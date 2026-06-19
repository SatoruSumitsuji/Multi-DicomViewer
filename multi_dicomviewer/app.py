"""QApplication bootstrap (+ a force-restart helper for hung sessions)."""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from multi_dicomviewer.config import APP_NAME
# NOTE: MainWindow is imported LAZILY inside main(), AFTER the splash is shown.
# It pulls in pydicom/numpy/pylibjpeg/etc. (~1s+ — the bulk of startup), so
# importing it at module load would block with nothing on screen.

#: Where the running session records its PID + how to relaunch itself, so the
#: external "restart" shortcut can force-kill a hung instance and reopen it
#: WITHOUT the user touching Task Manager. In the system temp dir so it works
#: regardless of where the app was launched from.
_SESSION_FILE = os.path.join(tempfile.gettempdir(), "multi_dicomviewer_session.json")


def _relaunch_argv() -> list[str]:
    """The command that re-launches this app exactly as it was started.

    * Frozen macOS .app: relaunch the BUNDLE via `open -n` (running the inner
      Mach-O binary directly can mis-set the bundle environment / Gatekeeper).
    * Frozen Windows/Linux: argv[0] IS the executable, so [exe, *args].
    * Dev (`python run.py …`): [python, run.py, *args]."""
    extra = list(sys.argv[1:])
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            i = sys.executable.find(".app/")
            if i != -1:
                app = sys.executable[: i + 4]      # ".../Multi-DicomViewer.app"
                return ["open", "-n", app] + (["--args", *extra] if extra else [])
        return [sys.executable, *extra]
    return [sys.executable, os.path.abspath(sys.argv[0]), *extra]


def _write_session() -> None:
    """Record this process so the restart shortcut can find and relaunch it."""
    try:
        data = {"pid": os.getpid(), "argv": _relaunch_argv(), "cwd": os.getcwd()}
        with open(_SESSION_FILE, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        atexit.register(_clear_session)
    except OSError:
        pass            # recording is best-effort; never block startup


def _clear_session() -> None:
    try:
        os.remove(_SESSION_FILE)
    except OSError:
        pass


def _kill_pid(pid: int) -> None:
    """Force-kill *pid* and its children (Windows: taskkill /F /T). Best-effort
    — a stale/already-gone PID is fine (the relaunch below still runs)."""
    if pid <= 0:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.kill(pid, 9)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass


def _restart() -> int:
    """`--restart` entry: kill the (possibly hung) recorded instance, then
    relaunch it. Runs as a SEPARATE short-lived process, so it works even when
    the GUI is completely frozen. Falls back to a plain launch if no session
    file exists."""
    argv = [sys.executable, os.path.abspath(sys.argv[0])]
    cwd = os.getcwd()
    try:
        with open(_SESSION_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        _kill_pid(int(data.get("pid", 0)))
        argv = data.get("argv") or argv
        cwd = data.get("cwd") or cwd
    except (OSError, ValueError):
        # No/garbled session file → just open a fresh instance below.
        argv = [a for a in argv if a != "--restart"]
    # Strip --restart from the relaunch command so we don't recurse.
    argv = [a for a in argv if a != "--restart"]
    # Detach the relaunched app so it outlives this helper (and the Terminal
    # the macOS .command ran in): Windows → its own process group; POSIX →
    # a new session.
    kwargs: dict = {"cwd": cwd, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen(argv, **kwargs)
    except OSError as exc:
        sys.stderr.write(f"[restart] failed to relaunch {argv!r}: {exc}\n")
        return 1
    return 0


def _make_splash() -> QSplashScreen:
    """A tiny in-code splash (no image asset) shown the instant Qt is up, so the
    user sees the app immediately while the heavy DICOM stack imports."""
    pm = QPixmap(440, 240)
    pm.fill(QColor(28, 30, 36))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(QColor(236, 239, 245))
    f = QFont()
    f.setPointSize(20)
    f.setBold(True)
    p.setFont(f)
    p.drawText(pm.rect().adjusted(0, -22, 0, -22),
               Qt.AlignmentFlag.AlignCenter, APP_NAME)
    f.setPointSize(11)
    f.setBold(False)
    p.setFont(f)
    p.setPen(QColor(150, 200, 255))
    p.drawText(pm.rect().adjusted(0, 58, 0, 58),
               Qt.AlignmentFlag.AlignCenter, "起動中… / Loading…")
    p.end()
    splash = QSplashScreen(pm)
    splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    return splash


def main(argv: list[str]) -> int:
    # External "restart" shortcut launches us with --restart: kill the hung
    # instance and reopen it, then exit (no GUI for this helper process).
    if "--restart" in argv:
        return _restart()

    app = QApplication(argv)
    app.setApplicationName(APP_NAME)

    # Show the splash BEFORE importing MainWindow (which loads pydicom/numpy/…),
    # so something is on screen within a moment instead of several blank seconds.
    splash = _make_splash()
    splash.show()
    app.processEvents()

    _write_session()
    from multi_dicomviewer.ui.main_window import MainWindow   # heavy import

    initial = argv[1] if len(argv) > 1 else None
    win = MainWindow(initial_folder=initial)
    win.showMaximized()
    splash.finish(win)                                        # close once shown
    return app.exec()
