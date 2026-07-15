"""Structured-Report viewer — renders a DICOM SR as readable text/tables.

SR files carry no pixel data, so the image viewers can't show them. An
X-Ray Radiation Dose SR ("Exam Protocol SR" from the angio system) is shown
as a dose report: study totals on top, then one table row per irradiation
event. Any other SR kind falls back to an indented plain-text content tree.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.core import sr_report
from multi_dicomviewer.core.dicom_io import LoadedSeries
from multi_dicomviewer.i18n import t
from multi_dicomviewer.ui.viewer_base import AbstractViewer


def _fmt(v, scale: float = 1.0, digits: int = 2) -> str:
    """Format an optional float for the table ('' when None)."""
    if v is None:
        return ""
    return f"{v * scale:.{digits}f}"


def _fmt_mmss(seconds) -> str:
    if seconds is None:
        return "—"
    s = int(round(float(seconds)))
    return f"{s // 60}:{s % 60:02d}"


class SRViewer(AbstractViewer):
    """Read-only report view for Modality=SR series (no image data)."""

    handles_modality = "SR"

    #: table columns: (header key, resize-to-contents?)
    _COLS = (
        "No", "Time", "Type", "Protocol", "Angle",
        "Pulses", "DAP (µGy·m²)", "Dose RP (mGy)",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ds = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)

        self._title_lbl = QLabel("")
        self._title_lbl.setStyleSheet("font-weight:bold;")
        self._title_lbl.setWordWrap(True)
        lay.addWidget(self._title_lbl)

        self._summary_lbl = QLabel("")
        self._summary_lbl.setWordWrap(True)
        self._summary_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self._summary_lbl)

        # The stretch item = the "image area" the pane's Max-Image keeps.
        self._stack = QStackedWidget()
        lay.addWidget(self._stack, 1)

        self._table = QTableWidget()
        self._table.setColumnCount(len(self._COLS))
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._stack.addWidget(self._table)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._text.setFont(mono)
        self._stack.addWidget(self._text)

    # ------------------------------------------------------ viewer contract
    def load_series(self, loaded: LoadedSeries, title: str) -> None:
        ds = loaded.header
        self._ds = ds
        self._title_lbl.setText(title)
        if ds is not None and sr_report.is_dose_sr(ds):
            self._show_dose_report(sr_report.parse_dose_sr(ds))
        else:
            self._summary_lbl.setText(
                t("Structured Report — no image data. Content shown below."))
            self._text.setPlainText(
                sr_report.generic_sr_text(ds) if ds is not None else "")
            self._stack.setCurrentWidget(self._text)

    def clear(self) -> None:
        self._ds = None
        self._title_lbl.setText("")
        self._summary_lbl.setText("")
        self._table.setRowCount(0)
        self._text.setPlainText("")

    def current_header(self):
        """The full SR dataset (lets the DICOM-tag dialog work as usual)."""
        return self._ds

    # ------------------------------------------------------------ rendering
    def _show_dose_report(self, rep: sr_report.DoseReport) -> None:
        bits = [t("Radiation Dose Report (no image data)")]
        if rep.device:
            bits.append(t("Device: {d}", d=rep.device))
        bits.append(t(
            "Runs: {na} acquisition / {nf} fluoro", na=rep.n_acq,
            nf=rep.n_fluoro))
        if rep.total_dap_gym2 is not None:
            bits.append(t("Total DAP: {v} µGy·m²",
                          v=f"{rep.total_dap_gym2 * 1e6:.1f}"))
        if rep.total_dose_rp_gy is not None:
            bits.append(t("Total Dose (RP): {v} mGy",
                          v=f"{rep.total_dose_rp_gy * 1e3:.1f}"))
        bits.append(t("Total fluoro time: {v}",
                      v=_fmt_mmss(rep.total_fluoro_time_s)))
        self._summary_lbl.setText("   |   ".join(bits))

        headers = [t("No"), t("Time"), t("Type"), t("Protocol"), t("Angle"),
                   t("Pulses"), t("DAP (µGy·m²)"), t("Dose RP (mGy)")]
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(rep.events))
        for r, ev in enumerate(rep.events):
            is_acq = "fluoro" not in ev.event_type.lower()
            cells = (
                str(r + 1),
                ev.datetime.split(" ")[-1],           # HH:MM:SS
                ev.event_type,
                ev.protocol,
                ev.angle_text,
                _fmt(ev.pulses, 1, 0),
                _fmt(ev.dap_gym2, 1e6, 1),
                _fmt(ev.dose_rp_gy, 1e3, 2),
            )
            for c, val in enumerate(cells):
                it = QTableWidgetItem(val)
                if c >= 5:                             # numeric columns
                    it.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter)
                if is_acq:
                    # Acquisition (cine) runs carry most of the dose — make
                    # them stand out from the fluoro rows.
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                self._table.setItem(r, c, it)
        hdr = self._table.horizontalHeader()
        for c in range(len(self._COLS)):
            hdr.setSectionResizeMode(
                c, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setStretchLastSection(True)
        self._stack.setCurrentWidget(self._table)
