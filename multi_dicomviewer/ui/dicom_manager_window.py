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

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QSplitter

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
            pnl.processed.connect(self._reload_all)

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
