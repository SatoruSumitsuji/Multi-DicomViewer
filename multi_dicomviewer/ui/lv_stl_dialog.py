"""Export-STL dialog for the LV reconstructed surfaces.

Three checkboxes — Endo only, Epi only, Endo+Epi (both in one file) — plus the
output folder (defaulting to the displayed series' folder, the same place the
.lv.json is kept). Whatever is checked is written on OK, named after the series
(``name;date_SeNNN_Endo.stl`` / ``_Epi.stl`` / ``_EndoEpi.stl``).
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from multi_dicomviewer.i18n import t


class LVStlExportDialog(QDialog):
    """Pick which LV surfaces to export as STL and where. *have_endo* /
    *have_epi* grey out choices whose surface isn't built yet."""

    def __init__(self, default_dir: str, stem: str,
                 have_endo: bool, have_epi: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Export LV surfaces (STL)"))
        root = QVBoxLayout(self)

        gb = QGroupBox(t("Surfaces to export"))
        gl = QVBoxLayout(gb)
        self.cb_endo = QCheckBox(t("Endo only  ({f})", f=stem + "_Endo.stl"))
        self.cb_epi = QCheckBox(t("Epi only  ({f})", f=stem + "_Epi.stl"))
        self.cb_both = QCheckBox(
            t("Endo + Epi  ({f})", f=stem + "_EndoEpi.stl"))
        self.cb_endo.setChecked(have_endo)
        self.cb_epi.setChecked(have_epi)
        self.cb_both.setChecked(have_endo and have_epi)
        for cb, ok in ((self.cb_endo, have_endo), (self.cb_epi, have_epi),
                       (self.cb_both, have_endo or have_epi)):
            cb.setEnabled(ok)
            gl.addWidget(cb)
        root.addWidget(gb)

        row = QHBoxLayout()
        row.addWidget(QLabel(t("Folder:")))
        self._dir_edit = QLineEdit(default_dir)
        row.addWidget(self._dir_edit, 1)
        browse = QPushButton(t("Browse…"))
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        root.addLayout(row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, t("Choose output folder"), self._dir_edit.text())
        if d:
            self._dir_edit.setText(d)

    def out_dir(self) -> str:
        return self._dir_edit.text().strip()

    def choices(self) -> dict:
        """{'endo': bool, 'epi': bool, 'both': bool} — the checked exports."""
        return {"endo": self.cb_endo.isChecked(),
                "epi": self.cb_epi.isChecked(),
                "both": self.cb_both.isChecked()}
