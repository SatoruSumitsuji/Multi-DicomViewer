"""Mouse-only operation grid for the cine (XA / IVUS) image canvas.

A 3×3 overlay whose cells each run ONE cine / view action, so the whole viewer
can be driven with the mouse alone. Click cells (series / frame / play) fire on
release; drag cells (pan / zoom / W-L / seek) act while dragging and the drag may
leave the cell once started (the grid captures it). The grid is an isolated layer:
when enabled it owns the mouse at the TOP of the canvas handlers, otherwise the
canvas behaves exactly as before. Shared by XA and IVUS (both use ImageCanvas),
so it works identically on Windows and Mac.

Click actions are forwarded to the viewer via canvas._grid_dispatch(action, arg);
pan / zoom / W-L are applied straight on the canvas (it already owns those).
"""
from __future__ import annotations

import math

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QApplication

from multi_dicomviewer.i18n import t


def _shift_held() -> bool:
    """True if Shift is down right now (read live so no event plumbing needed)."""
    return bool(QApplication.keyboardModifiers()
                & Qt.KeyboardModifier.ShiftModifier)


#: Shift+click on a frame / series step cell jumps to the first / last instead of
#: stepping by one.
_SHIFT_JUMP = {
    "prev_frame": "first_frame", "next_frame": "last_frame",
    "prev_series": "first_series", "next_series": "last_series",
}

#: Which actions are click-fired vs drag-driven. `playseek` is BOTH: a click
#: toggles play/stop, a drag (past _SEEK_DRAG_PX) scrubs the seek bar — so it is
#: handled specially and belongs to neither set.
_CLICK = {"prev_series", "next_series", "prev_frame", "next_frame"}
_DRAG = {"pan", "zoom", "wl", "rotate"}

#: Click cells that also auto-repeat while the button is held down, so a long
#: press steps through frames continuously (one step per interval) on top of the
#: single step a quick tap gives. Only the frame-step cells repeat — series /
#: play must fire exactly once per press.
_REPEAT = {"prev_frame", "next_frame"}
#: Hold this long after the press before continuous stepping kicks in (so a quick
#: tap stays a single step), then step every _REPEAT_INTERVAL_MS.
_REPEAT_DELAY_MS = 350
_REPEAT_INTERVAL_MS = 90
#: Pointer travel (px) that turns a press on the combined play/seek cell from a
#: play/stop CLICK into a seek DRAG.
_SEEK_DRAG_PX = 8

#: Short on-cell labels (kept terse so they read in a quarter-pane 2×2 layout).
_LABELS = {
    "prev_series": "◀ シリーズ",
    "next_series": "シリーズ ▶",
    "prev_frame":  "◀ フレーム",
    "next_frame":  "フレーム ▶",
    "playseek":    "▶ ■  ↔ シーク",
    "rotate":      "⟳ 回転",
    "pan":         "✥ 移動",
    "zoom":        "🔍 拡大",
    "wl":          "◐ 階調",
    # Legacy keys (pre-merge) kept so an old saved layout still labels sanely
    # until it is migrated to `playseek`.
    "play":        "▶ ■",
    "seek":        "↔ シーク",
}


class MouseGrid:
    """Per-canvas 3×3 mouse-operation overlay. Owns its enabled flag + layout and
    the in-progress drag; the canvas routes press/move/release here first."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.enabled = False
        # Overlay visibility is auto-managed by the canvas (shown after the mouse
        # is still for ~1 s, hidden on any move/action) so it never obscures the
        # image; interaction stays live whenever `enabled`, regardless.
        self.show_overlay = False
        self.layout = ["prev_frame", "rotate", "next_frame",
                       "prev_series", "playseek", "next_series",
                       "pan", "zoom", "wl"]
        self._drag_action = None          # action key of an in-progress drag
        self._press_cell = None           # cell pressed (for click-on-release)
        self._press_xy = None
        self._last_xy = None
        self._rot_last = None             # last pointer angle during a rotate drag
        self._rot_raw = 0.0               # unsnapped rotation accumulator (Shift=15°)
        self._press_fired = False         # True once a press already dispatched
        #                                   its action (repeat cells fire on press,
        #                                   so release must NOT fire it again).
        self._repeat_action = None        # action being auto-repeated on hold
        # Hold-to-repeat timers (owned by the canvas QWidget so they live/die
        # with it). The delay timer gates the start of continuous stepping; the
        # repeat timer then fires the step every _REPEAT_INTERVAL_MS.
        self._repeat_delay = QTimer(canvas)
        self._repeat_delay.setSingleShot(True)
        self._repeat_delay.setInterval(_REPEAT_DELAY_MS)
        self._repeat_delay.timeout.connect(self._start_repeat)
        self._repeat_timer = QTimer(canvas)
        self._repeat_timer.setInterval(_REPEAT_INTERVAL_MS)
        self._repeat_timer.timeout.connect(self._on_repeat_tick)

    # -------------------------------------------------- geometry / layout
    def set_enabled(self, on: bool) -> None:
        was = self.enabled
        self.enabled = bool(on)
        if was != self.enabled:
            self._stop_repeat()
            self._drag_action = self._press_cell = None
            self.canvas.update()

    def set_layout(self, layout) -> None:
        if isinstance(layout, (list, tuple)) and len(layout) == 9:
            self.layout = [str(k) for k in layout]
            if self.enabled:
                self.canvas.update()

    def _cell(self, sx, sy, w, h):
        """Row-major cell index 0..8 for widget px (sx,sy), or None if outside."""
        if w <= 0 or h <= 0 or sx < 0 or sy < 0 or sx >= w or sy >= h:
            return None
        col = min(2, int(sx * 3 / w))
        row = min(2, int(sy * 3 / h))
        return row * 3 + col

    def _action(self, sx, sy, w, h):
        c = self._cell(sx, sy, w, h)
        if c is None:
            return None
        try:
            return self.layout[c]
        except IndexError:
            return None

    # ------------------------------------------------------------- mouse
    def active(self) -> bool:
        return self.enabled

    def press(self, sx, sy) -> bool:
        """Begin a grid interaction. Returns True if the grid consumes the press
        (so the canvas skips its normal handling)."""
        if not self.enabled:
            return False
        w, h = self.canvas.width(), self.canvas.height()
        act = self._action(sx, sy, w, h)
        if act is None:
            return True                   # inside the grid but no action → eat it
        self._press_xy = (sx, sy)
        self._last_xy = (sx, sy)
        self._press_fired = False
        if act in _DRAG:
            self._drag_action = act
            self._press_cell = None
            if act == "seek":
                self._apply_seek(sx, w)   # jump on the initial press too
            elif act == "rotate":
                # Seed the rotate drag with the pointer's angle + the current
                # rotation so moves apply a delta (and Shift can snap to 15°).
                fn = getattr(self.canvas, "_free_rot_angle", None)
                self._rot_last = fn(sx, sy) if callable(fn) else None
                self._rot_raw = float(getattr(self.canvas, "_free_rot", 0.0))
        else:
            self._drag_action = None
            self._press_cell = act
            if act in _REPEAT:
                # Frame step: fire once immediately (responsive tap). With Shift it
                # JUMPS to the first/last frame instead (no auto-repeat); otherwise
                # it arms the hold-to-repeat. release() skips its own click since
                # _press_fired is set, so a tap = exactly one action.
                if _shift_held():
                    self._dispatch(_SHIFT_JUMP[act])
                    self._press_fired = True
                else:
                    self._dispatch(act)
                    self._press_fired = True
                    self._repeat_action = act
                    self._repeat_delay.start()
        return True

    def move(self, sx, sy) -> bool:
        if not self.enabled:
            return False
        # Combined play/seek cell: a press that travels past the threshold turns
        # into a seek drag (so it no longer fires the play/stop click on release);
        # within the threshold it stays a pending click and the move is eaten.
        if self._drag_action is None and self._press_cell == "playseek":
            px, py = self._press_xy or (sx, sy)
            if (abs(sx - px) >= _SEEK_DRAG_PX or abs(sy - py) >= _SEEK_DRAG_PX):
                self._drag_action = "playseek"
                self._press_cell = None
                self._last_xy = (sx, sy)
                self._apply_seek(sx, self.canvas.width())
            return True
        if self._drag_action is None:
            return False
        lx, ly = self._last_xy or (sx, sy)
        dx, dy = sx - lx, sy - ly
        self._last_xy = (sx, sy)
        a = self._drag_action
        if a == "pan":
            self.canvas._pan[0] += dx
            self.canvas._pan[1] += dy
            self.canvas.update()
        elif a == "zoom":
            factor = 1.0 + (sy - ly) * 0.006          # drag DOWN = zoom in (CT)
            if abs(factor - 1.0) > 1e-4:
                self.canvas._apply_zoom(
                    factor, self.canvas.width() / 2.0, self.canvas.height() / 2.0)
        elif a == "wl":
            if dx or dy:
                self.canvas.wl_dragged.emit(float(dx), float(dy))
        elif a == "playseek":
            self._apply_seek(sx, self.canvas.width())
        elif a == "rotate":
            self._apply_rotate(sx, sy)
        return True

    def _apply_rotate(self, sx, sy) -> None:
        """Spin the displayed image by the angular delta the pointer swept about
        the canvas centre, SCALED by how far the pointer is from the centre so a
        drag near the rim turns much faster / bigger than one near the hub. Holding
        Shift snaps the result to 15° steps. Updates canvas._free_rot (paint
        applies it)."""
        c = self.canvas
        fn = getattr(c, "_free_rot_angle", None)
        if not callable(fn) or not hasattr(c, "_free_rot"):
            return
        now = fn(sx, sy)
        prev = self._rot_last if self._rot_last is not None else now
        d = (now - prev + 180.0) % 360.0 - 180.0      # shortest signed delta
        self._rot_last = now
        # Distance-based gain: calm near the centre, faster the farther out.
        w, h = c.width(), c.height()
        dist = math.hypot(sx - w / 2.0, sy - h / 2.0)
        ref = max(1.0, 0.22 * min(w, h))
        gain = max(0.5, min(5.0, dist / ref))
        self._rot_raw = (self._rot_raw + d * gain) % 360.0
        c._free_rot = (round(self._rot_raw / 15.0) * 15.0) % 360.0 \
            if _shift_held() else self._rot_raw
        c.update()

    def double_click(self, sx, sy) -> bool:
        """Left double-click handling. On the 回転 cell → reset the rotation to 0°.
        Returns True if the grid consumed it."""
        if not self.enabled:
            return False
        w, h = self.canvas.width(), self.canvas.height()
        if self._action(sx, sy, w, h) == "rotate":
            c = self.canvas
            if hasattr(c, "set_free_rotation"):
                c.set_free_rotation(0.0)
            elif hasattr(c, "_free_rot"):
                c._free_rot = 0.0
            self._rot_raw = 0.0
            c.update()
            return True
        return False

    def release(self, sx, sy) -> bool:
        if not self.enabled:
            return False
        self._stop_repeat()
        handled = (self._drag_action is not None) or (self._press_cell is not None)
        if (self._press_cell is not None and self._drag_action is None
                and not self._press_fired):
            # A click (no drag) on a click-cell fires its action, but only if the
            # release is still on the same cell and the pointer barely moved.
            # (Repeat cells already fired on press — _press_fired — so they skip
            # this, otherwise a tap would step twice.)
            px, py = self._press_xy or (sx, sy)
            if abs(sx - px) < 12 and abs(sy - py) < 12:
                w, h = self.canvas.width(), self.canvas.height()
                cell = self._press_cell
                if self._action(sx, sy, w, h) == cell:
                    # Shift+click on a series step cell jumps to first / last.
                    if cell in _SHIFT_JUMP and _shift_held():
                        self._dispatch(_SHIFT_JUMP[cell])
                    else:
                        self._dispatch(cell)
        self._drag_action = self._press_cell = None
        self._press_xy = self._last_xy = None
        self._press_fired = False
        self._rot_last = None
        return handled

    # -------------------------------------------------- hold-to-repeat
    def _start_repeat(self) -> None:
        """Delay elapsed with the button still held → begin continuous stepping."""
        if self.enabled and self._repeat_action is not None:
            self._repeat_timer.start()

    def _on_repeat_tick(self) -> None:
        """Fire one more frame step while the button is held on a frame cell."""
        if self.enabled and self._repeat_action is not None:
            self._dispatch(self._repeat_action)
        else:
            self._stop_repeat()

    def _stop_repeat(self) -> None:
        self._repeat_delay.stop()
        self._repeat_timer.stop()
        self._repeat_action = None

    def _apply_seek(self, sx, w) -> None:
        # Seek is part of the combined play/seek action: a non-None arg (the
        # fraction) tells the viewer to scrub; a None arg is the play/stop click.
        frac = max(0.0, min(1.0, sx / float(max(1, w))))
        self._dispatch("playseek", frac)

    def _dispatch(self, action, arg=None) -> None:
        cb = getattr(self.canvas, "_grid_dispatch", None)
        if callable(cb):
            try:
                cb(action, arg)
            except Exception:                           # noqa: BLE001
                pass

    # ------------------------------------------------------------- paint
    def paint(self, p, w, h) -> None:
        """Draw the 3×3 grid + cell labels (call only while enabled)."""
        if not self.enabled or w <= 0 or h <= 0:
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        cw, ch = w / 3.0, h / 3.0
        # Grid lines.
        p.setPen(QPen(QColor(255, 196, 0, 150), 1.4))
        for i in (1, 2):
            p.drawLine(int(i * cw), 0, int(i * cw), h)
            p.drawLine(0, int(i * ch), w, int(i * ch))
        # Cell tint + label.
        fnt = QFont()
        fnt.setBold(True)
        fnt.setPointSizeF(max(8.0, min(13.0, ch * 0.16)))
        p.setFont(fnt)
        for idx, act in enumerate(self.layout[:9]):
            r, c = divmod(idx, 3)
            cell = QRectF(c * cw, r * ch, cw, ch)
            drag = act in _DRAG or act == "playseek"   # playseek has a seek drag
            p.fillRect(cell, QColor(40, 90, 160, 40) if drag
                       else QColor(20, 20, 20, 36))
            self._outlined_text(
                p, cell, int(Qt.AlignmentFlag.AlignCenter),
                t(_LABELS.get(act, act)),
                QColor(255, 230, 120) if drag else QColor(235, 235, 235))
        # A small corner hint that mouse-grid mode is ON.
        self._outlined_text(
            p, QRectF(4, 2, w - 8, 16),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
            t("マウス操作グリッド ON"), QColor(255, 196, 0, 220))
        p.restore()

    @staticmethod
    def _outlined_text(p, rect, align, text, color) -> None:
        """Draw *text* with a subtle 1-px dark outline so it stays legible over a
        bright background (ECG grids etc.) without the heavy look of a full
        stroke — the halo is semi-transparent black drawn in the 8 surrounding
        offsets, the foreground colour on top."""
        halo = QColor(0, 0, 0, 170)
        p.setPen(halo)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            p.drawText(rect.translated(dx, dy), align, text)
        p.setPen(color)
        p.drawText(rect, align, text)
