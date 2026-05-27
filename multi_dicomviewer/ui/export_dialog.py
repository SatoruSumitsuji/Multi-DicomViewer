"""Settings dialog for the right-click Export (DICOM) / Export (MP4)
actions in the StudyBrowser.

The dialog gathers two things:

  * which DICOM tags to glue together (with underscores) into the output
    filename — one tag per checkbox;
  * for MP4 mode only, the target bitrate (Mbps) and the playback FPS.

The caller then asks for an output folder separately and runs the actual
export. Keeping the dialog focused on choices (no I/O, no series objects)
lets the same widget serve both formats with a single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


#: The 8 filename components the user can opt in/out of, in the order
#: they appear in the dialog AND in the order they are concatenated into
#: the output filename (so the order is predictable across runs).
FIELD_KEYS = (
    "date_time",       # acquisition date/time (e.g. 20260227_114526)
    "series_no",       # SeriesNumber, 3-digit zero-padded
    "acq_no",          # AcquisitionNumber, 3-digit zero-padded
    "type",            # DICOM Modality (XA/CT/IVUS/…)
    "description",     # SeriesDescription
    "images",          # frame count as Nimg
    "primary",         # PositionerPrimaryAngle as LAO/RAO + 3-digit
    "secondary",       # PositionerSecondaryAngle as CRA/CAU + 3-digit
)

FIELD_LABELS = {
    "date_time":   "Date/Time",
    "series_no":   "Series No",
    "acq_no":      "Acquisition No",
    "type":        "Type (Modality)",
    "description": "Description",
    "images":      "Images (count)",
    "primary":     "Positioner Primary Angle",
    "secondary":   "Positioner Secondary Angle",
}

#: Sensible defaults — enough to identify a series at a glance without a
#: hopelessly long filename. Both Series No and Acq No are pre-ticked
#: because in practice one of the two is empty per acquisition style;
#: ticking both means the populated one always shows up.
DEFAULT_FIELDS = (
    "date_time", "series_no", "acq_no", "type", "description"
)


@dataclass
class ExportSettings:
    """Result of the dialog, consumed by core.export."""
    fields: tuple[str, ...]   # ordered subset of FIELD_KEYS
    bitrate_mbps: int = 10    # MP4 only
    fps: float = 15.0         # MP4 only


class ExportDialog(QDialog):
    """One dialog for both formats. ``mode`` selects which extra widgets
    (bitrate / fps) are shown. The default FPS for XA/IVUS comes from the
    source DICOM and is supplied by the caller via ``default_fps`` (None
    falls back to 15 fps, also used for CT)."""

    def __init__(self, mode: str, n_series: int,
                 default_fps: Optional[float] = None,
                 show_filename_fields: bool = True,
                 title_override: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        if mode not in ("dicom", "mp4"):
            raise ValueError(f"unknown export mode: {mode!r}")
        self._mode = mode

        title = title_override or (
            "Export DICOM" if mode == "dicom" else "Export MP4"
        )
        plural = "" if n_series == 1 else f" ({n_series} series)"
        self.setWindowTitle(title + plural)
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)

        # --- filename components -------------------------------------------
        self._checks: dict[str, QCheckBox] = {}
        if show_filename_fields:
            box = QGroupBox("Filename components (joined with '_')")
            col = QVBoxLayout(box)
            col.addWidget(QLabel(
                "Tick what to put in each output filename. Components "
                "missing for a given series (e.g. no angle on a CT) are "
                "skipped."
            ))
            for key in FIELD_KEYS:
                cb = QCheckBox(FIELD_LABELS[key])
                cb.setChecked(key in DEFAULT_FIELDS)
                self._checks[key] = cb
                col.addWidget(cb)
            root.addWidget(box)

        # --- MP4 bitrate / fps ---------------------------------------------
        if mode == "mp4":
            mp4_box = QGroupBox("MP4 encoding")
            form = QFormLayout(mp4_box)
            self._bitrate = QSpinBox()
            self._bitrate.setRange(1, 10)
            self._bitrate.setValue(10)
            self._bitrate.setSuffix(" Mbps")
            self._bitrate.setToolTip(
                "Target H.264 bitrate. 10 Mbps ≈ near-lossless for typical "
                "512×512 XA cine; 4 Mbps still clinical-grade; 1 Mbps "
                "is the smallest the dialog accepts."
            )
            form.addRow("Bitrate:", self._bitrate)
            self._fps = QDoubleSpinBox()
            self._fps.setRange(1.0, 60.0)
            self._fps.setDecimals(1)
            self._fps.setSingleStep(1.0)
            self._fps.setSuffix(" fps")
            self._fps.setValue(
                float(default_fps) if default_fps and default_fps > 0
                else 15.0
            )
            self._fps.setToolTip(
                "Playback frame rate. XA/IVUS default = the source cine "
                "rate; CT default = 15 fps (slice scroll)."
            )
            form.addRow("Frame rate:", self._fps)
            root.addWidget(mp4_box)
        else:
            self._bitrate = None
            self._fps = None

        # --- OK / Cancel ---------------------------------------------------
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def result_settings(self) -> ExportSettings:
        """Returns the user's choices. Call only after exec() == Accepted."""
        chosen = tuple(
            k for k in FIELD_KEYS
            if k in self._checks and self._checks[k].isChecked()
        )
        if self._mode == "mp4":
            return ExportSettings(
                fields=chosen,
                bitrate_mbps=int(self._bitrate.value()),
                fps=float(self._fps.value()),
            )
        return ExportSettings(fields=chosen)
