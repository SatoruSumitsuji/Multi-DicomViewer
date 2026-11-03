"""Opt-in mouse-input tracing for diagnosing "dead mouse" reports.

Enabled only when the environment variable MDV_INPUT_DEBUG=1 is set, e.g.

    MDV_INPUT_DEBUG=1 python run.py

It prints, to the terminal, which widget receives each mouse press/release,
plus any Qt mouse grabber / modal / popup widget at that moment — so a report
like "clicks do nothing, keyboard shortcuts still work" shows WHERE the clicks
go. Viewers call log() with their own gesture-state snapshot. Nothing about the
images or the patient is printed: only widget class names and state flags.
"""
from __future__ import annotations

import os
import sys
import time

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtWidgets import QApplication, QWidget

ENABLED = os.environ.get("MDV_INPUT_DEBUG", "") not in ("", "0")

_T0 = time.monotonic()


def log(msg: str) -> None:
    if ENABLED:
        print(f"[input {time.monotonic() - _T0:8.3f}] {msg}",
              file=sys.stderr, flush=True)


def _wname(w) -> str:
    if w is None:
        return "-"
    try:
        name = w.objectName()
    except Exception:                                    # noqa: BLE001
        name = ""
    cls = type(w).__name__
    txt = ""
    try:
        if hasattr(w, "text") and callable(w.text):
            txt = str(w.text())[:24]
    except Exception:                                    # noqa: BLE001
        pass
    return f"{cls}{'#' + name if name else ''}{'[' + txt + ']' if txt else ''}"


class _Filter(QObject):
    _TYPES = {QEvent.Type.MouseButtonPress: "press",
              QEvent.Type.MouseButtonRelease: "release"}

    def eventFilter(self, obj, ev):                      # noqa: N802
        kind = self._TYPES.get(ev.type())
        if kind is not None and isinstance(obj, QWidget):
            try:
                log(f"qt {kind} -> {_wname(obj)} | grabber="
                    f"{_wname(QWidget.mouseGrabber())} modal="
                    f"{_wname(QApplication.activeModalWidget())} popup="
                    f"{_wname(QApplication.activePopupWidget())}")
            except Exception:                            # noqa: BLE001
                pass
        return False


_filter = None


def install(app) -> None:
    """Install the app-wide filter (no-op unless MDV_INPUT_DEBUG is set)."""
    global _filter
    if not ENABLED or _filter is not None:
        return
    _filter = _Filter()
    app.installEventFilter(_filter)
    log("input tracing ON")
