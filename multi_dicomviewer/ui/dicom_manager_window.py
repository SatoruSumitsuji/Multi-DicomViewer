"""DicomManager — the combined DICOM-housekeeping tool.

ONE window that stacks the DicomCheck panel (delete non-DICOM files / empty folders)
over the DicomFolder panel (organize DICOM files into sub-folders). Dropping ONE
folder onto the window loads it into BOTH panels; whenever EITHER panel acts
(delete / Copy / Move) BOTH panels re-scan the folder so their views stay in sync
(the agreed "always reload both"). Replaces the former separate DicomCheck and
DicomFolder Tools entries.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QSplitter

from multi_dicomviewer.i18n import t
from multi_dicomviewer.ui.dicom_check_window import DicomCheckPanel
from multi_dicomviewer.ui.dicom_folder_window import DicomFolderPanel


class DicomManagerWindow(QMainWindow):
    def __init__(self, start_dir: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("DicomManager"))
        self.resize(1000, 980)
        self.setAcceptDrops(True)                  # drop a folder anywhere
        self._root: str | None = None

        split = QSplitter(Qt.Orientation.Vertical)
        self.setCentralWidget(split)
        self._check = DicomCheckPanel(start_dir)
        self._folder = DicomFolderPanel(start_dir)
        split.addWidget(self._check)
        split.addWidget(self._folder)
        # A wide, grabbable handle so the user can freely drag the top (DicomCheck)
        # panel DOWN to enlarge its file tree; neither panel fully collapses.
        split.setHandleWidth(8)
        split.setChildrenCollapsible(False)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([470, 510])                 # near-even; drag to taste
        self._split = split

        # One drop / pick on EITHER panel loads BOTH; either panel's action re-scans
        # BOTH so DicomCheck and DicomFolder always reflect the same folder state.
        for pnl in (self._check, self._folder):
            pnl.folder_dropped.connect(self._load_into_all)
            # After a Sort Files / Delete finishes, the folder changed — but a
            # re-scan isn't always wanted, so ASK (再スキャン / Clear / 終了)
            # instead of auto-reloading both panels.
            pnl.processed.connect(self._after_action)
        # "Clear Selection" (on the DicomFolder panel) clears BOTH views: the
        # folder panel has already reset itself, so the host clears DicomCheck too
        # and forgets the current root.
        self._folder.cleared.connect(self._clear_all)

    # ---- folder coordination -------------------------------------------
    def _load_into_all(self, path: str) -> None:
        if not path or not os.path.isdir(path):
            return
        self._root = path
        self._check.load_folder(path)
        self._folder.load_folder(path)

    def _reload_all(self) -> None:
        if self._root and os.path.isdir(self._root):
            self._check.load_folder(self._root)
            self._folder.load_folder(self._root)

    def _clear_all(self) -> None:
        """"Clear Selection" was pressed: the DicomFolder panel already reset
        itself, so clear the DicomCheck panel and forget the root."""
        self._root = None
        self._check.clear()

    # ---- what to do after a Sort Files / Delete ------------------------
    def _after_action(self) -> None:
        """A panel finished a Sort Files / Delete. Re-scanning isn't always
        needed, so defer and ask the user what to do next."""
        QTimer.singleShot(0, self._ask_next)

    def _ask_next(self) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(t("DicomManager"))
        box.setText(t("Processing finished. What would you like to do?"))
        rescan_btn = box.addButton(t("Re-scan"),
                                   QMessageBox.ButtonRole.AcceptRole)
        clear_btn = box.addButton(t("Clear Selection"),
                                  QMessageBox.ButtonRole.ActionRole)
        close_btn = box.addButton(t("Close"),
                                  QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(rescan_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is rescan_btn:
            self._reload_all()
        elif clicked is clear_btn:
            # Clear BOTH views; the folder panel's _clear emits `cleared`, which
            # the host routes to _clear_all to also clear DicomCheck.
            self._folder._clear()
        elif clicked is close_btn:
            self.close()

    # ---- drag&drop on the window chrome (splitter handle / margins) -----
    @staticmethod
    def _dropped_dir(ev):
        md = ev.mimeData()
        if not md.hasUrls():
            return None
        for u in md.urls():
            p = u.toLocalFile()
            if p and os.path.isdir(p):
                return p
        return None

    def dragEnterEvent(self, ev):                  # noqa: N802 (Qt override)
        if self._dropped_dir(ev) is not None:
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):                   # noqa: N802 (Qt override)
        self.dragEnterEvent(ev)

    def dropEvent(self, ev):                        # noqa: N802 (Qt override)
        d = self._dropped_dir(ev)
        if d:
            ev.acceptProposedAction()
            self._load_into_all(d)

    def closeEvent(self, ev):                       # noqa: N802 (Qt override)
        self._check.cleanup()
        self._folder.cleanup()
        super().closeEvent(ev)
