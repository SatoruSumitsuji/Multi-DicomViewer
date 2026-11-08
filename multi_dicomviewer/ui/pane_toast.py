# -*- coding: utf-8 -*-
"""Short centred caption over a pane ("Zoom適応" when a tool key switches the
measure mode) — the frame blink alone was easy to miss.

A frameless, click-through TOP-LEVEL tool-tip window rather than a child
label: the CT panes are native OpenGL surfaces, which a sibling/child QLabel
cannot reliably paint over (Windows), whereas a separate window always does.
"""
from __future__ import annotations

from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtWidgets import QLabel

_toast: QLabel | None = None
_gen = 0

#: CT tool id → caption word (the tool buttons' names without the key).
_TOOL_WORDS = {"ZOOM": "Zoom", "MOVE": "Move", "ROTATE": "Rotate",
               "SPIN": "Spin", "PAGING": "Paging", "THICK": "Thick", "WL": "WL"}


def show_tool_toast(viewer, tool: str) -> None:
    """CT viewers: after a tool key really switched the tool, show
    "<Tool>適応" centred on the ACTIVE pane for the Settings time
    (計測モード変更表示時間; 0 = off)."""
    from multi_dicomviewer.core import settings
    from multi_dicomviewer.i18n import t
    try:
        ms = int(settings.load_display_quality().get("tool_toast_ms", 500))
    except (TypeError, ValueError):
        ms = 500
    if ms <= 0:
        return
    frames = getattr(viewer, "_frames", None) or {}
    w = frames.get(getattr(viewer, "_active_pane", None))
    word = _TOOL_WORDS.get(tool, str(tool).capitalize())
    show_pane_toast(w, t("{tool}適応", tool=word), ms)


def show_pane_toast(widget, text: str, ms: int) -> None:
    """Show *text* centred over *widget* for *ms* milliseconds (0 = nothing).
    A new toast replaces one still showing."""
    global _toast, _gen
    if widget is None or ms <= 0 or not text:
        return
    try:
        if not widget.isVisible():
            return
        if _toast is None:
            lab = QLabel()
            lab.setWindowFlags(Qt.WindowType.ToolTip
                               | Qt.WindowType.FramelessWindowHint
                               | Qt.WindowType.WindowStaysOnTopHint
                               | Qt.WindowType.WindowDoesNotAcceptFocus)
            lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            lab.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setStyleSheet(
                "QLabel{background:rgba(0,0,0,170);color:#ffe000;"
                "border:2px solid #ff2020;border-radius:8px;"
                "padding:10px 22px;font-size:26px;font-weight:bold;}")
            _toast = lab
        lab = _toast
        lab.setText(text)
        lab.adjustSize()
        c = widget.mapToGlobal(QPoint(widget.width() // 2, widget.height() // 2))
        lab.move(c.x() - lab.width() // 2, c.y() - lab.height() // 2)
        lab.show()
        lab.raise_()
    except RuntimeError:                      # widget deleted meanwhile
        return
    _gen += 1
    gen = _gen

    def _hide():
        if gen == _gen and _toast is not None:
            try:
                _toast.hide()
            except RuntimeError:
                pass

    QTimer.singleShot(int(ms), _hide)
