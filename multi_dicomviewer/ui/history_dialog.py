"""Measurement-history window.

A small modeless dialog the shell pops up on "Measure History". It only
renders a snapshot it is handed; the shell owns the per-study history
(kept in memory until the app exits) and re-pushes it when it changes
while the window is open.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QDialog,
    QLabel,
    QListWidget,
    QVBoxLayout,
)


class MeasureHistoryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Measure History")
        self.resize(360, 440)

        self._header = QLabel("—")
        self._header.setStyleSheet("color:#ccc; padding:2px;")
        self._list = QListWidget()

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.close)
        btns.accepted.connect(self.close)

        lay = QVBoxLayout(self)
        lay.addWidget(self._header)
        lay.addWidget(self._list, 1)
        lay.addWidget(btns)

    def set_entries(self, title: str, measurements) -> None:
        """Render *measurements* (a list of Measurement) under *title*."""
        measurements = list(measurements or [])
        self._header.setText(f"{title} — {len(measurements)} item(s)")
        self._list.clear()
        if not measurements:
            self._list.addItem("(No measurements yet)")
            return
        for i, m in enumerate(measurements, 1):
            label = m.label() or "—"
            self._list.addItem(f"{i:>2}.  {m.kind}: {label}")
