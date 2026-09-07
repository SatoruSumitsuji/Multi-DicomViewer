"""SnapDock — a QDockWidget whose FLOATING window gains normal-window sizing
gestures a Qt tool window otherwise lacks:

* full-screen ⇄ previous-size toggle (``toggle_maximize`` — call it from a
  button and/or a title double-click), and
* drag an edge to the screen top/bottom → vertical maximize (fill height).

Both are ONE reversible state (``_expanded``): whatever the pre-expand geometry
was is remembered, so maximize AND edge-snap can always be undone by toggling
back. Docking is unaffected: drag onto the main window and drop in the blue
region to re-dock. A floating QDockWidget is a Qt tool window with no native
maximize / Aero-Snap and ``showMaximized()`` is a no-op, hence the manual
geometry here. The title double-click is best-effort (the floating title bar is
platform-drawn and not always deliverable as a Qt event) — prefer a real button.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QRect, QTimer
from PyQt6.QtWidgets import QApplication, QDockWidget, QWidget


class SnapDock(QDockWidget):
    _SNAP_PX = 12                    # edge-proximity threshold (px)
    _SETTLE_MS = 140                 # drag-settle delay before an edge check

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._expanded = False       # currently maximized OR edge-snapped
        self._pre_geom = None        # geometry to restore to
        self._snapping = False       # guards our own setGeometry from re-firing
        self._snap_timer = QTimer(self)
        self._snap_timer.setSingleShot(True)
        self._snap_timer.timeout.connect(self._check_edge_snap)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # -- geometry primitives ---------------------------------------------
    def _avail_geom(self):
        scr = self.screen()
        return scr.availableGeometry() if scr is not None else None

    def _frame_margins(self):
        """(left, top, right, bottom) px the window FRAME (title bar / borders)
        adds around the client geometry — so an expand can keep the WHOLE window
        (title bar included) on-screen instead of pushing the bar off the top."""
        fg = self.frameGeometry()
        g = self.geometry()
        return (g.left() - fg.left(), g.top() - fg.top(),
                fg.right() - g.right(), fg.bottom() - g.bottom())

    def _apply(self, rect: QRect) -> None:
        self._snapping = True                # our own move must not re-trigger
        self.setGeometry(rect)
        self._snapping = False

    def _restore(self) -> None:
        if self._pre_geom is not None:
            self._apply(self._pre_geom)
        self._expanded = False

    def _expand_full(self) -> None:
        av = self._avail_geom()
        if av is None:
            return
        if not self._expanded:
            self._pre_geom = self.geometry()
        left, top, right, bottom = self._frame_margins()
        self._apply(QRect(av.x() + left, av.top() + top,
                          av.width() - left - right,
                          av.height() - top - bottom))
        self._expanded = True

    def _expand_vertical(self) -> None:
        av = self._avail_geom()
        if av is None:
            return
        if not self._expanded:
            self._pre_geom = self.geometry()
        g = self.geometry()
        _, top, _, bottom = self._frame_margins()
        # Fill the height but keep the title bar (top frame) on-screen.
        self._apply(QRect(g.x(), av.top() + top,
                          g.width(), av.height() - top - bottom))
        self._expanded = True

    # -- public toggle (button / double-click) ---------------------------
    def toggle_maximize(self) -> None:
        """Full-screen ⇄ previous size; also the way to undo an edge-snap."""
        if not self.isFloating():
            return
        if self._expanded:
            self._restore()
        else:
            self._expand_full()

    def is_expanded(self) -> bool:
        return self._expanded

    # -- edge snap on drag-settle ----------------------------------------
    def moveEvent(self, e):  # noqa: N802 (Qt override)
        super().moveEvent(e)
        if self.isFloating() and not self._snapping:
            self._snap_timer.start(self._SETTLE_MS)

    def _check_edge_snap(self) -> None:
        if not self.isFloating() or self._expanded:
            return
        av = self._avail_geom()
        if av is None:
            return
        g = self.frameGeometry()
        if (abs(g.top() - av.top()) <= self._SNAP_PX
                or abs(g.bottom() - av.bottom()) <= self._SNAP_PX):
            self._expand_vertical()

    # -- title-bar double-click (best-effort) ----------------------------
    def _on_titlebar(self, obj) -> bool:
        if not isinstance(obj, QWidget):
            return False
        content = self.widget()
        in_content = content is not None and (
            obj is content or content.isAncestorOf(obj))
        return (obj is self or self.isAncestorOf(obj)) and not in_content

    def eventFilter(self, obj, event):  # noqa: N802 (Qt override)
        if (event.type() == QEvent.Type.MouseButtonDblClick
                and self._on_titlebar(obj)):
            if self.isFloating():
                self.toggle_maximize()
            return True
        return super().eventFilter(obj, event)
