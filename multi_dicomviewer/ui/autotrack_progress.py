"""Progress + result dialogs of the CPR Auto-Track (user 2026-11-18): the
trace runs in the foreground, so a progress window shows what it is doing
(trunk mm / branch k of n / total mm) with a 停止 button, and the result is
always shown in a dialog (also when nothing was found, with the reason).

While it runs a large red 「■ Auto-Track 停止」 button floats over the top of
the images (user 2026-11-25: how to stop was not obvious) — the progress
window's 停止 and a right-click anywhere stop it too."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import (QApplication, QMessageBox, QProgressDialog,
                             QPushButton, QWidget)

from multi_dicomviewer.i18n import t


class _RightClickStop(QObject):
    """While Auto-Track runs: a right-click ANYWHERE (panes included) stops
    it (user 2026-11-19); other clicks / wheel outside the progress window
    are swallowed so the trace is not disturbed mid-way."""

    _MOUSE = (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
              QEvent.Type.MouseButtonDblClick, QEvent.Type.Wheel)

    def __init__(self, owner):
        super().__init__()
        self._owner = owner

    def eventFilter(self, obj, ev):                      # noqa: N802
        try:
            if ev.type() not in self._MOUSE or not isinstance(obj, QWidget):
                return False
            for w in (self._owner._dlg, self._owner._stop_btn):
                if w is not None and (obj is w or w.isAncestorOf(obj)):
                    return False                     # the 停止 buttons etc.
            if (ev.type() == QEvent.Type.MouseButtonPress
                    and ev.button() == Qt.MouseButton.RightButton):
                self._owner._on_cancel()
            return True
        except Exception:                                # noqa: BLE001
            return False


class AutoTrackProgress:
    """``with AutoTrackProgress(parent) as pg: extend_vessel(...,
    cancel=pg.cancelled, progress=pg.update)``."""

    _STOP_STYLE = (
        "QPushButton { background:#d62828; color:white; font-weight:bold;"
        " font-size:%dpx; border:2px solid #7a0d0d; border-radius:7px;"
        " padding:%s; }"
        "QPushButton:hover { background:#e63946; }"
        "QPushButton:pressed { background:#9b1c1c; }")

    def __init__(self, parent=None, name: str = "", anchor=None):
        self._parent = parent
        self._name = name
        self._anchor = anchor          # the image area the stop button sits on
        self._dlg = None
        self._stop_btn = None
        self._cancel = False
        self._filter = None

    def __enter__(self):
        d = QProgressDialog(t("Auto-Track: 準備中…"), t("停止"), 0, 0,
                            self._parent)
        d.setWindowTitle(t("Auto-Track") + (f" — {self._name}"
                                            if self._name else ""))
        # not modal: the right-click stop has to reach the panes (the
        # filter below swallows every other click there meanwhile)
        d.setWindowModality(Qt.WindowModality.NonModal)
        d.setMinimumDuration(0)
        d.setAutoClose(False)
        d.setAutoReset(False)
        d.canceled.connect(self._on_cancel)
        cb = d.findChild(QPushButton)
        if cb is not None:                       # the dialog's 停止: red too
            cb.setStyleSheet(self._STOP_STYLE % (13, "4px 14px"))
        d.show()
        # keep the panes visible (the trace is watched step by step): the
        # window sits at the top-left of the main window, over the toolbar
        try:
            top = self._parent.window() if self._parent is not None else None
            if top is not None:
                g = top.frameGeometry()
                d.move(g.left() + 24, g.top() + 60)
        except Exception:                                # noqa: BLE001
            pass
        QApplication.processEvents()
        self._dlg = d
        self._show_stop_button()
        app = QApplication.instance()
        if app is not None:
            self._filter = _RightClickStop(self)
            app.installEventFilter(self._filter)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        return self

    def _area_rect(self):
        """Global rect the stop button is centred on: the union of the
        anchor widgets (the left + right image panes) — so it sits between
        the two images, just below their top."""
        from PyQt6.QtCore import QRect
        anc = self._anchor
        if anc is None:
            anc = self._parent.window() if self._parent is not None else None
        if anc is None:
            return None
        ws = list(anc) if isinstance(anc, (list, tuple)) else [anc]
        rect = None
        for w in ws:
            try:
                if w is None or not w.isVisible():
                    continue
                g = QRect(w.mapToGlobal(w.rect().topLeft()), w.size())
                rect = g if rect is None else rect.united(g)
            except Exception:                            # noqa: BLE001
                continue
        return rect

    def _show_stop_button(self) -> None:
        """The red stop button, floating between the two images just below
        their top (user 2026-11-25: closer, smaller) — a frameless
        always-on-top tool window: a child widget would hide under the GL
        canvas."""
        try:
            area = self._area_rect()
            b = QPushButton(t("■ Auto-Track 停止"))
            b.setWindowFlags(Qt.WindowType.Tool
                             | Qt.WindowType.FramelessWindowHint
                             | Qt.WindowType.WindowStaysOnTopHint)
            b.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(t("右クリックでも停止できます"))
            b.setStyleSheet(self._STOP_STYLE % (14, "4px 14px"))
            b.clicked.connect(self._on_cancel)
            b.adjustSize()
            if area is not None:
                b.move(area.x() + (area.width() - b.width()) // 2,
                       area.y() + 10)
            b.show()
            b.raise_()
            self._stop_btn = b
        except Exception:                                # noqa: BLE001
            self._stop_btn = None

    def __exit__(self, *exc):
        QApplication.restoreOverrideCursor()
        app = QApplication.instance()
        if app is not None and self._filter is not None:
            app.removeEventFilter(self._filter)
        self._filter = None
        if self._stop_btn is not None:
            self._stop_btn.close()
            self._stop_btn.deleteLater()
            self._stop_btn = None
        if self._dlg is not None:
            self._dlg.close()
            self._dlg.deleteLater()
            self._dlg = None
        return False

    def _on_cancel(self):
        self._cancel = True
        if self._stop_btn is not None:           # show that it was heard
            self._stop_btn.setEnabled(False)
            self._stop_btn.setText(t("停止中…"))

    def cancelled(self) -> bool:
        return self._cancel

    def text(self) -> str:
        return self._dlg.labelText() if self._dlg is not None else ""

    def update(self, stage, line_mm, total_mm, k=0, n=0, name="") -> None:
        if self._dlg is None:
            return
        if stage == "batch":                 # 一括描出: line k of n
            s = t("一括描出 {k} / {n}：「{nm}」を近位へ追跡中… {a:.0f} mm",
                  k=k, n=n, nm=name, a=line_mm)
        elif stage == "trunk":
            s = t("幹を追跡中… {a:.0f} mm（合計 {b:.0f} mm）", a=line_mm,
                  b=total_mm)
        else:
            s = t("分枝 {k} / {n} を追跡中…（合計 {b:.0f} mm）", k=k, n=n,
                  b=total_mm)
        self._dlg.setLabelText(s + "\n" + t("右クリックで停止"))
        QApplication.processEvents()


def show_result(parent, text: str, ok: bool = True, extra="",
                modeless: bool = False):
    """The Auto-Track result, always in a dialog. *extra* = one button text
    (→ returns True when chosen) or a list of them (→ returns the index of
    the one chosen, -1 for OK). *modeless*: the windows behind stay usable
    while it waits (a fork — the right pane can be rotated to compare the
    numbered candidates); it still returns only once answered."""
    many = isinstance(extra, (list, tuple))
    labels = list(extra) if many else ([extra] if extra else [])
    if not labels:
        (QMessageBox.information if ok else QMessageBox.warning)(
            parent, t("Auto-Track"), text)
        return -1 if many else False
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Information if ok
                else QMessageBox.Icon.Warning)
    box.setWindowTitle(t("Auto-Track"))
    box.setText(text)
    btns = [box.addButton(s, QMessageBox.ButtonRole.AcceptRole)
            for s in labels]
    box.addButton(QMessageBox.StandardButton.Ok)
    if modeless:
        from PyQt6.QtCore import QEventLoop
        box.setWindowModality(Qt.WindowModality.NonModal)
        loop = QEventLoop()
        box.finished.connect(loop.quit)
        box.show()
        box.raise_()
        loop.exec()
    else:
        box.exec()
    hit = box.clickedButton()
    k = next((i for i, b in enumerate(btns) if b is hit), -1)
    return k if many else k == 0
