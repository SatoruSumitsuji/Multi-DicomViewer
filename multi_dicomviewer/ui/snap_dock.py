"""SnapDock — a QDockWidget whose FLOATING window gains the normal-window
gestures a Qt tool window otherwise lacks:

* title-bar double-click → full-screen ⇄ previous size, and
* dragging an edge to the screen top/bottom → vertical maximize (fill height).

Docking is unaffected: drag onto the main window and drop in the blue region
to re-dock (QDockWidget's native behaviour). A floating QDockWidget is a Qt
tool window, so it has no native maximize / Aero-Snap and ``showMaximized()``
is a no-op — hence the manual geometry handling here.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QTimer
from PyQt6.QtWidgets import QApplication, QDockWidget, QWidget


class SnapDock(QDockWidget):
    _SNAP_PX = 12                    # edge-proximity threshold
    _SETTLE_MS = 140                 # drag-settle delay before an edge check

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._maxed = False
        self._pre_max_geom = None
        self._snapping = False       # guards our own setGeometry from re-firing
        self._snap_timer = QTimer(self)
        self._snap_timer.setSingleShot(True)
        self._snap_timer.timeout.connect(self._check_edge_snap)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # -- geometry helpers ------------------------------------------------
    def _avail_geom(self):
        scr = self.screen()
        return scr.availableGeometry() if scr is not None else None

    def _toggle_maximize(self) -> None:
        """Title double-click while floating: full-screen ⇄ previous size."""
        if not self.isFloating():
            return
        av = self._avail_geom()
        if av is None:
            return
        self._snapping = True                # don't let setGeometry clear _maxed
        if self._maxed:
            if self._pre_max_geom is not None:
                self.setGeometry(self._pre_max_geom)
            self._maxed = False
        else:
            self._pre_max_geom = self.geometry()
            self.setGeometry(av)
            self._maxed = True
        self._snapping = False

    def moveEvent(self, e):  # noqa: N802 (Qt override)
        super().moveEvent(e)
        # A manual drag cancels full-screen (like a normal window) and schedules
        # an edge-snap check once the movement settles.
        if self.isFloating() and not self._snapping:
            self._maxed = False
            self._snap_timer.start(self._SETTLE_MS)

    def _check_edge_snap(self) -> None:
        if not self.isFloating() or self._maxed:
            return
        av = self._avail_geom()
        if av is None:
            return
        g = self.frameGeometry()
        if (abs(g.top() - av.top()) <= self._SNAP_PX
                or abs(g.bottom() - av.bottom()) <= self._SNAP_PX):
            self._snapping = True
            self.setGeometry(self.geometry().x(), av.top(),
                             self.width(), av.height())
            self._snapping = False

    # -- title-bar double-click interception -----------------------------
    def _on_titlebar(self, obj) -> bool:
        """True if *obj* is on this dock's title bar (not its content widget)."""
        if not isinstance(obj, QWidget):
            return False
        content = self.widget()
        in_content = content is not None and (
            obj is content or content.isAncestorOf(obj))
        return (obj is self or self.isAncestorOf(obj)) and not in_content

    def eventFilter(self, obj, event):  # noqa: N802 (Qt override)
        # Swallow the title double-click (Qt's default toggles float↔dock, which
        # keeps losing the panel) and maximize instead when floating.
        if (event.type() == QEvent.Type.MouseButtonDblClick
                and self._on_titlebar(obj)):
            if self.isFloating():
                self._toggle_maximize()
            return True
        return super().eventFilter(obj, event)
