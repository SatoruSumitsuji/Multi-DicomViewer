"""Anonymize-settings dialog.

Lets the user pick which DICOM tags get blanked by BOTH the on-screen
Anonymize toggle and the "Export (Anon DICOM)" writer. A checked tag has
its value emptied (the tag itself stays). A separate toggle empties every
private (vendor) tag's value. Opened by right-clicking the Anonymous button.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.core.anonymize import ANON_TAG_CATALOG


def _tag_label(group: int, element: int) -> str:
    """'(0010,0010)  Patient's Name' for a catalog entry."""
    from pydicom.datadict import dictionary_description
    try:
        name = dictionary_description((group, element)) or ""
    except Exception:
        name = ""
    if not name and element == 0x0000:
        name = "Group Length"
    return f"({group:04X},{element:04X})  {name}".rstrip()


class AnonSettingsDialog(QDialog):
    """Choose the anonymization profile (tags to emptify + private toggle)."""

    def __init__(self, selected_tags, emptify_private: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Anonymize settings — items to anonymize")
        self.setMinimumWidth(460)
        sel = {(int(g), int(e)) for g, e in (selected_tags or ())}

        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "Checked items have their values cleared (the tags themselves remain).\n"
            "This setting applies to both on-screen anonymization (the Anonymous button)\n"
            "and Export (Anon DICOM)."
        ))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(280)
        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(4, 4, 4, 4)
        col.setSpacing(1)
        self._checks: dict[tuple[int, int], QCheckBox] = {}
        for (g, e) in ANON_TAG_CATALOG:
            cb = QCheckBox(_tag_label(g, e))
            cb.setChecked((g, e) in sel)
            self._checks[(g, e)] = cb
            col.addWidget(cb)
        col.addStretch(1)
        scroll.setWidget(host)
        root.addWidget(scroll, 1)

        self._priv = QCheckBox(
            "Also clear all Private (vendor) tag values"
        )
        self._priv.setChecked(bool(emptify_private))
        root.addWidget(self._priv)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def result_profile(self) -> tuple[list[tuple[int, int]], bool]:
        """(checked tags, emptify_private). Call after exec() == Accepted."""
        tags = [ge for ge, cb in self._checks.items() if cb.isChecked()]
        return tags, self._priv.isChecked()
