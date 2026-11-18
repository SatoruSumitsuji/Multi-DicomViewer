"""Progress + result dialogs of the CPR Auto-Track (user 2026-11-18): the
trace runs in the foreground, so a progress window shows what it is doing
(trunk mm / branch k of n / total mm) with a 停止 button, and the result is
always shown in a dialog (also when nothing was found, with the reason)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from multi_dicomviewer.i18n import t


class AutoTrackProgress:
    """``with AutoTrackProgress(parent) as pg: extend_vessel(...,
    cancel=pg.cancelled, progress=pg.update)``."""

    def __init__(self, parent=None, name: str = ""):
        self._parent = parent
        self._name = name
        self._dlg = None
        self._cancel = False

    def __enter__(self):
        d = QProgressDialog(t("Auto-Track: 準備中…"), t("停止"), 0, 0,
                            self._parent)
        d.setWindowTitle(t("Auto-Track") + (f" — {self._name}"
                                            if self._name else ""))
        d.setWindowModality(Qt.WindowModality.WindowModal)
        d.setMinimumDuration(0)
        d.setAutoClose(False)
        d.setAutoReset(False)
        d.canceled.connect(self._on_cancel)
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
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        return self

    def __exit__(self, *exc):
        QApplication.restoreOverrideCursor()
        if self._dlg is not None:
            self._dlg.close()
            self._dlg.deleteLater()
            self._dlg = None
        return False

    def _on_cancel(self):
        self._cancel = True

    def cancelled(self) -> bool:
        return self._cancel

    def text(self) -> str:
        return self._dlg.labelText() if self._dlg is not None else ""

    def update(self, stage, line_mm, total_mm, k=0, n=0) -> None:
        if self._dlg is None:
            return
        if stage == "trunk":
            s = t("幹を追跡中… {a:.0f} mm（合計 {b:.0f} mm）", a=line_mm,
                  b=total_mm)
        else:
            s = t("分枝 {k} / {n} を追跡中…（合計 {b:.0f} mm）", k=k, n=n,
                  b=total_mm)
        self._dlg.setLabelText(s)
        QApplication.processEvents()


def show_result(parent, text: str, ok: bool = True, extra=""):
    """The Auto-Track result, always in a dialog. *extra* = one button text
    (→ returns True when chosen) or a list of them (→ returns the index of
    the one chosen, -1 for OK)."""
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
    box.exec()
    hit = box.clickedButton()
    k = next((i for i, b in enumerate(btns) if b is hit), -1)
    return k if many else k == 0
