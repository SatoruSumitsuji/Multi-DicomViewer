"""Tiny modal dialog shown after two Polygon/Ellipse shapes are picked for
Compare: choose which analysis to run.

- **%PA**  — percent-area difference (used for IVUS measurements).
- **Thickness** — radial gap colour map (used for CT LV wall-thickness).

The two are normally used one at a time (different clinical purposes), so the
user ticks whichever is wanted. Shared by both CT viewers (VTK + pygfx) so the
choice behaves identically on Windows and macOS.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from multi_dicomviewer.i18n import t


class CompareOptionsDialog(QDialog):
    """Two checkboxes (%PA, Thickness) + OK/Cancel. ``values()`` returns the
    chosen (want_pa, want_thickness) booleans once accepted."""

    def __init__(self, want_pa: bool = False, want_thickness: bool = True,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Compare"))
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(t("Required Data:")))
        self.cb_pa = QCheckBox(t("%PA  (percent area — IVUS)"))
        self.cb_thk = QCheckBox(t("Thickness  (radial gap — CT LV wall)"))
        self.cb_pa.setChecked(want_pa)
        self.cb_thk.setChecked(want_thickness)
        lay.addWidget(self.cb_pa)
        lay.addWidget(self.cb_thk)
        btns = QHBoxLayout()
        ok = QPushButton("OK")
        cancel = QPushButton(t("Cancel"))
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        lay.addLayout(btns)

    def values(self) -> tuple[bool, bool]:
        return self.cb_pa.isChecked(), self.cb_thk.isChecked()
