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

_toasts: list = []          # one label per pane shown at once (reused)
_gen = 0

#: CT tool id → caption word (the tool buttons' names without the key).
_TOOL_WORDS = {"ZOOM": "Zoom", "MOVE": "Move", "ROTATE": "Rotate",
               "SPIN": "Spin", "PAGING": "Paging", "THICK": "Thick", "WL": "WL"}


def tool_differs_between_panes(viewer, tool: str) -> bool:
    """True when *tool* does NOT act the same on the two CT panes, so the
    caption should only go on the active pane:
      - one pane is a VR (CPR VR表示 / Territory): a drag there turns the 3-D
        model (Mac: always; Windows: Paging/Thick become Rotate, Rotate turns
        the model instead of tilting the plane);
      - a CPR is built: on the short-axis pane Rotate/Spin turn the section,
        Paging steps along the centreline, Thick does not apply;
      - Epi tracing: Move is off on the reference (non-trace) pane."""
    vr = getattr(viewer, "_vr_on", None) or {}
    if bool(vr.get("A")) != bool(vr.get("B")):
        return True
    if (getattr(viewer, "_cpr", None) is not None
            and tool in ("ROTATE", "SPIN", "PAGING", "THICK")):
        return True
    lv = getattr(viewer, "_lv", None)
    if (tool == "MOVE" and isinstance(lv, dict) and lv.get("pass") == "epi"
            and lv.get("sax") is None):
        return True
    return False


def show_tool_toast(viewer, tool: str) -> None:
    """CT viewers: after a tool key really switched the tool, show
    "<Tool>適応" centred on the panes for the Settings time
    (計測モード変更表示時間; 0 = off) — on BOTH panes when the tool acts the
    same on both, else on the active pane only."""
    from multi_dicomviewer.core import settings
    from multi_dicomviewer.i18n import t
    try:
        ms = int(settings.load_display_quality().get("tool_toast_ms", 500))
    except (TypeError, ValueError):
        ms = 500
    if ms <= 0:
        return
    frames = getattr(viewer, "_frames", None) or {}
    act = getattr(viewer, "_active_pane", None)
    if tool_differs_between_panes(viewer, tool):
        keys = [act]
    else:
        keys = [k for k in ("A", "B") if k in frames]
    word = _TOOL_WORDS.get(tool, str(tool).capitalize())
    show_pane_toast([frames.get(k) for k in keys], t("{tool}適応", tool=word),
                    ms)


def _new_label() -> QLabel:
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
    return lab


def show_pane_toast(widgets, text: str, ms: int) -> None:
    """Show *text* centred over each visible widget in *widgets* (one widget
    or a list) for *ms* milliseconds (0 = nothing). A new toast replaces the
    ones still showing."""
    global _gen
    if not isinstance(widgets, (list, tuple)):
        widgets = [widgets]
    if ms <= 0 or not text:
        return
    shown = []
    for w in widgets:
        try:
            if w is not None and w.isVisible():
                shown.append(w)
        except RuntimeError:                  # widget deleted meanwhile
            pass
    for lab in _toasts:                       # drop the previous toast(s)
        try:
            lab.hide()
        except RuntimeError:
            pass
    if not shown:
        return
    while len(_toasts) < len(shown):
        _toasts.append(_new_label())
    for w, lab in zip(shown, _toasts):
        try:
            lab.setText(text)
            lab.adjustSize()
            c = w.mapToGlobal(QPoint(w.width() // 2, w.height() // 2))
            lab.move(c.x() - lab.width() // 2, c.y() - lab.height() // 2)
            lab.show()
            lab.raise_()
        except RuntimeError:
            continue
    _gen += 1
    gen = _gen

    def _hide():
        if gen == _gen:
            for lab in _toasts:
                try:
                    lab.hide()
                except RuntimeError:
                    pass

    QTimer.singleShot(int(ms), _hide)
