"""QApplication bootstrap (+ a force-restart helper for hung sessions)."""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from multi_dicomviewer.config import APP_NAME, APP_VERSION

#: Minimum time (s) the start-up splash stays up — so the version/contact is
#: readable even when the heavy import finished sooner. Same on every platform.
_MIN_SPLASH_S = 3.0
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
    """In-code splash (no image asset) shown the instant Qt is up: app name,
    version, a one-line description and the contact address. The SAME content is
    shown on every platform while the heavy DICOM stack imports."""
    pm = QPixmap(480, 280)
    pm.fill(QColor(28, 30, 36))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    rect = pm.rect()
    top = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop

    def line(y, text, size, color, bold=False):
        f = QFont()
        f.setPointSize(size)
        f.setBold(bold)
        p.setFont(f)
        p.setPen(color)
        p.drawText(rect.adjusted(0, y, 0, 0), top, text)

    line(48, APP_NAME, 22, QColor(236, 239, 245), bold=True)
    line(96, f"v{APP_VERSION}", 11, QColor(150, 200, 255))
    line(142, "研究用DICOMビューワ", 13, QColor(210, 214, 222))
    line(186, "問い合わせ先：satoru@sumi2g.sakura.ne.jp",
         11, QColor(170, 176, 186))
    # Loading hint, bottom.
    f = QFont()
    f.setPointSize(10)
    p.setFont(f)
    p.setPen(QColor(120, 126, 136))
    p.drawText(rect.adjusted(0, 0, 0, -16),
               Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
               "起動中… / Loading…")
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
    app.setApplicationDisplayName(APP_NAME)

    # Windows: give the process an explicit AppUserModelID so every top-level
    # window it opens (main + CoSync + the other tool windows) is grouped under
    # ONE taskbar button — the app's own, not python.exe's. Hovering that button
    # then shows a live thumbnail per open window, each selectable. Without this
    # the windows scatter under Python's icon (or don't group at all).
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "Sumi2g.MultiDicomViewer")
        except Exception:
            pass          # best-effort; grouping is cosmetic, never block start

    # A frozen Windows/Linux build shows a NATIVE PyInstaller splash before
    # Python even starts (see the .spec) — the only thing that can cover the
    # first 2-3 s while the big Qt/DICOM DLLs cold-load. macOS / dev runs have
    # none.
    try:
        import pyi_splash                       # only exists in the native build
        have_native_splash = True
    except Exception:
        have_native_splash = False

    # Show our Qt info-splash (app name + version + contact) on EVERY platform
    # so the content is identical everywhere; close the native one once ours is
    # up so the two don't stack (option A).
    splash = _make_splash()
    splash.show()
    app.processEvents()
    if have_native_splash:
        try:
            import pyi_splash
            pyi_splash.close()
        except Exception:
            pass

    splash_start = time.monotonic()
    _write_session()
    from multi_dicomviewer.ui.main_window import MainWindow   # heavy import

    initial = argv[1] if len(argv) > 1 else None
    win = MainWindow(initial_folder=initial)

    # Keep the splash up for at least _MIN_SPLASH_S, even if the import finished
    # sooner, then reveal the window and close the splash. singleShot(0) simply
    # fires on the next event-loop pass when the minimum has already elapsed.
    remaining_ms = max(
        0, int((_MIN_SPLASH_S - (time.monotonic() - splash_start)) * 1000)
    )

    def _reveal():
        win.showMaximized()
        splash.finish(win)

    QTimer.singleShot(remaining_ms, _reveal)
    return app.exec()
