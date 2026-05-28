"""DICOM tag list dialog — pick which tags overlay on the image.

Shared by the XA and CT viewers. Shows the loaded series' header as a
filterable table; ticking a row adds that tag's identifier to the
overlay selection. Standard tags use their pydicom keyword
(``PatientName``); private / unknown tags use their tag-string literal
(``(0019,1099)``) — both are addressable by ``_lookup`` in
:mod:`dicom_tags`. Values honour the global anonymization toggle.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from multi_dicomviewer.core.dicom_tags import iter_tag_rows

_KW_ROLE = Qt.ItemDataRole.UserRole


class TagSelectionDialog(QDialog):
    """Modal picker returning the chosen overlay keywords in stable order."""

    def __init__(
        self,
        header,
        selected: list[str],
        anonymized: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("DICOM Tags — choose overlay items")
        self.resize(760, 560)
        self._prev_order = list(selected)
        self._anonymized = anonymized
        selected_set = set(selected)

        root = QVBoxLayout(self)

        note = QLabel(
            "Check the tags to overlay on the image."
            + ("   * Anonymized: case info is masked."
               if anonymized else "")
        )
        note.setWordWrap(True)
        root.addWidget(note)

        filt_row = QHBoxLayout()
        filt_row.addWidget(QLabel("Filter:"))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText(
            "Filter by tag name / keyword / value"
        )
        self.filter.textChanged.connect(self._apply_filter)
        filt_row.addWidget(self.filter, 1)
        root.addLayout(filt_row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Show", "Tag", "Name", "VR", "Value"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)

        self._populate(header, selected_set)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    # --------------------------------------------------------------- build
    def _populate(self, header, selected_set: set[str]) -> None:
        rows = iter_tag_rows(header, anonymized=self._anonymized)
        self.table.setRowCount(len(rows))
        for r, tr in enumerate(rows):
            # Standard tags identify themselves by their pydicom keyword;
            # private/unknown tags (no keyword) fall back to the tag-
            # string literal so they're still selectable and addressable.
            identifier = tr.keyword or tr.tag
            chk = QTableWidgetItem()
            chk.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
            )
            chk.setCheckState(
                Qt.CheckState.Checked
                if identifier in selected_set
                else Qt.CheckState.Unchecked
            )
            chk.setData(_KW_ROLE, identifier)
            if not tr.keyword:
                chk.setToolTip(
                    "Private/unknown tag — selected by its (group,"
                    "element) literal"
                )
            self.table.setItem(r, 0, chk)
            for c, text in enumerate(
                (tr.tag, tr.name, tr.vr, tr.value), start=1
            ):
                self.table.setItem(r, c, QTableWidgetItem(text))

    # -------------------------------------------------------------- filter
    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for r in range(self.table.rowCount()):
            if not needle:
                self.table.setRowHidden(r, False)
                continue
            hay = " ".join(
                self.table.item(r, c).text().lower()
                for c in (1, 2, 3, 4)
            )
            kw = (self.table.item(r, 0).data(_KW_ROLE) or "").lower()
            self.table.setRowHidden(r, needle not in hay and needle not in kw)

    # -------------------------------------------------------------- result
    def selected_keywords(self) -> list[str]:
        """Checked identifiers (pydicom keywords + private-tag literals),
        prior order kept then new ones in table order."""
        checked: list[str] = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item.checkState() == Qt.CheckState.Checked:
                ident = item.data(_KW_ROLE)
                if ident:
                    checked.append(ident)
        checked_set = set(checked)
        ordered = [k for k in self._prev_order if k in checked_set]
        ordered += [k for k in checked if k not in self._prev_order]
        return ordered
