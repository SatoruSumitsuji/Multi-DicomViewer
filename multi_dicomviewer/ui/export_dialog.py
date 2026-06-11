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
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


#: Filename components the user can opt in/out of, in the order they
#: appear in the dialog AND in the order they are concatenated into the
#: output filename (so the order is predictable across runs). Date and
#: Time are separate keys so CT can drop Time (often missing / not
#: meaningful) while XA/IVUS keep both.
FIELD_KEYS = (
    "date",            # AcquisitionDate (YYYYMMDD)
    "time",            # AcquisitionTime (HHMMSS)
    "series_no",       # SeriesNumber, 3-digit zero-padded
    "instance_no",     # InstanceNumber, 3-digit zero-padded
    "type",            # DICOM Modality (XA/CT/IVUS/…)
    "description",     # SeriesDescription
    "images",          # frame count as Nimg
    "primary",         # PositionerPrimaryAngle as LAO/RAO + 3-digit
    "secondary",       # PositionerSecondaryAngle as CRA/CAU + 3-digit
)

FIELD_LABELS = {
    "date":        "Date",
    "time":        "Time",
    "series_no":   "Series No",
    "instance_no": "Instance No",
    "type":        "Type (Modality)",
    "description": "Description",
    "images":      "Images (count)",
    "primary":     "Positioner Primary Angle",
    "secondary":   "Positioner Secondary Angle",
}

#: Sensible defaults — enough to identify a series at a glance without a
#: hopelessly long filename. Both Series No and Instance No are pre-ticked
#: because in practice one of the two is empty per acquisition style;
#: ticking both means the populated one always shows up.
DEFAULT_FIELDS = (
    "date", "time", "series_no", "instance_no", "type", "description"
)


@dataclass
class ExportSettings:
    """Result of the dialog, consumed by core.export."""
    fields: tuple[str, ...]   # ordered subset of FIELD_KEYS
    bitrate_mbps: int = 10    # MP4 only (used when crf is None)
    fps: float = 30.0         # MP4 only
    #: MP4 only. When set, encode at constant quality (x264 -crf) instead
    #: of a target bitrate — far smaller files at equal visual quality on
    #: content with large flat/black areas (IVUS composites). None → use
    #: bitrate_mbps. Lower = higher quality / bigger file.
    crf: Optional[int] = None


#: Preset frame-rate buttons shown next to the FPS spinbox.
FPS_PRESETS = (5, 10, 15, 30, 60)


class ExportDialog(QDialog):
    """One dialog for both formats. ``mode`` selects which extra widgets
    (bitrate / fps) are shown. The default FPS for XA/IVUS comes from the
    source DICOM and is supplied by the caller via ``default_fps`` (None
    falls back to 15 fps, also used for CT)."""

    def __init__(self, mode: str, n_series: int,
                 default_fps: Optional[float] = None,
                 show_filename_fields: bool = True,
                 title_override: Optional[str] = None,
                 dicom_tags: Optional[list] = None,
                 initial_fields: Optional[list] = None,
                 default_bitrate: Optional[int] = None,
                 default_crf: int = 20,
                 parent=None):
        """``dicom_tags`` is an optional list of ``(identifier, label)``
        tuples — typically the modality's tag-overlay selection — that
        the user can also tick to include in the filename. Identifiers
        are pydicom keywords (``PatientName``) or tag-string literals
        (``(0019,1099)``). ``initial_fields`` overrides the per-key
        default-checked state (used for per-modality memory)."""
        super().__init__(parent)
        if mode not in ("dicom", "mp4", "csv", "anon-dicom"):
            raise ValueError(f"unknown export mode: {mode!r}")
        self._mode = mode

        title = title_override or {
            "dicom": "Export DICOM",
            "mp4": "Export MP4",
            "csv": "Export CSV",
            "anon-dicom": "Export Anon DICOM",
        }.get(mode, "Export")
        plural = "" if n_series == 1 else f" ({n_series} series)"
        self.setWindowTitle(title + plural)
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)

        # --- filename components -------------------------------------------
        self._checks: dict[str, QCheckBox] = {}
        # Ordered identifiers seen by this dialog (predefined first,
        # then DICOM tags) — result_settings preserves this order.
        self._field_order: list[str] = []
        if show_filename_fields:
            # Use the caller's stored per-modality selection when given;
            # otherwise fall back to the static defaults.
            initial_set = (set(initial_fields)
                            if initial_fields is not None
                            else set(DEFAULT_FIELDS))

            box = QGroupBox("Filename components (joined with '_')")
            col = QVBoxLayout(box)
            col.addWidget(QLabel(
                "Tick what to put in each output filename. Components "
                "missing for a given series (e.g. no angle on a CT) are "
                "skipped."
            ))
            for key in FIELD_KEYS:
                cb = QCheckBox(FIELD_LABELS[key])
                cb.setChecked(key in initial_set)
                self._checks[key] = cb
                self._field_order.append(key)
                col.addWidget(cb)

            # DICOM tag list (the same items the user picked in the
            # DICOM-Tag-overlay dialog for this modality). Empty when
            # the caller passes nothing — in that case the section is
            # hidden so legacy callers still see exactly the old UI.
            tags = list(dicom_tags or [])
            if tags:
                col.addSpacing(6)
                col.addWidget(QLabel(
                    "DICOM tags (chosen in DICOM-Tag overlay) — tick to "
                    "include the tag's value in the filename:"
                ))
                # Long lists deserve a scroll area so the dialog doesn't
                # explode vertically on series with many tags selected.
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setMinimumHeight(60)
                scroll.setMaximumHeight(220)
                host = QWidget()
                hcol = QVBoxLayout(host)
                hcol.setContentsMargins(2, 2, 2, 2)
                hcol.setSpacing(1)
                for ident, label in tags:
                    cb = QCheckBox(label)
                    cb.setChecked(ident in initial_set)
                    cb.setToolTip(f"DICOM tag identifier: {ident}")
                    self._checks[ident] = cb
                    self._field_order.append(ident)
                    hcol.addWidget(cb)
                hcol.addStretch(1)
                scroll.setWidget(host)
                col.addWidget(scroll)

            root.addWidget(box)

        # --- MP4 bitrate / fps ---------------------------------------------
        if mode == "mp4":
            mp4_box = QGroupBox("MP4 encoding")
            form = QFormLayout(mp4_box)

            # Quality (CRF) vs target Bitrate. CRF is the default: constant
            # visual quality, and far smaller files than a fixed bitrate on
            # content with big flat/black areas (IVUS composites).
            self._mode_crf = QRadioButton("Quality (CRF)")
            self._mode_br = QRadioButton("Bitrate (Mbps)")
            self._mode_crf.setChecked(True)
            self._enc_group = QButtonGroup(self)
            self._enc_group.addButton(self._mode_crf)
            self._enc_group.addButton(self._mode_br)
            mode_row = QWidget()
            mode_lay = QHBoxLayout(mode_row)
            mode_lay.setContentsMargins(0, 0, 0, 0)
            mode_lay.addWidget(self._mode_crf)
            mode_lay.addWidget(self._mode_br)
            mode_lay.addStretch(1)
            form.addRow("Encode:", mode_row)

            self._crf = QSpinBox()
            self._crf.setRange(10, 30)
            self._crf.setValue(int(default_crf))
            self._crf.setToolTip(
                "Constant Rate Factor (x264). Lower = higher quality / "
                "larger file. 10–12 ≈ near-lossless (speckle preserved), "
                "18 visually lossless, 23 standard, 26+ compact. "
                "Each −6 ≈ double the file size."
            )
            form.addRow("CRF:", self._crf)

            self._bitrate = QSpinBox()
            self._bitrate.setRange(1, 100)
            self._bitrate.setValue(
                int(default_bitrate) if default_bitrate else 10
            )
            self._bitrate.setSuffix(" Mbps")
            self._bitrate.setToolTip(
                "Target H.264 bitrate (used only in Bitrate mode). "
                "Speckle-heavy IVUS / multi-pane composites need 30–40 Mbps "
                "@ 15 fps to match the on-screen view. Up to 100 Mbps."
            )
            form.addRow("Bitrate:", self._bitrate)

            def _sync_enc_mode():
                crf_on = self._mode_crf.isChecked()
                self._crf.setEnabled(crf_on)
                self._bitrate.setEnabled(not crf_on)

            self._mode_crf.toggled.connect(lambda _c: _sync_enc_mode())
            _sync_enc_mode()
            self._fps = QDoubleSpinBox()
            self._fps.setRange(1.0, 60.0)
            self._fps.setDecimals(1)
            self._fps.setSingleStep(1.0)
            self._fps.setSuffix(" fps")
            self._fps.setValue(
                float(default_fps) if default_fps and default_fps > 0
                else 30.0
            )
            self._fps.setToolTip(
                "Playback frame rate. XA/IVUS default = the source cine "
                "rate when available; otherwise 30 fps."
            )
            # Preset buttons next to the spinbox: click sets the value.
            fps_row = QWidget()
            fps_layout = QHBoxLayout(fps_row)
            fps_layout.setContentsMargins(0, 0, 0, 0)
            fps_layout.addWidget(self._fps, 1)
            for v in FPS_PRESETS:
                btn = QPushButton(str(v))
                btn.setFixedWidth(36)
                btn.setToolTip(f"Set frame rate to {v} fps")
                btn.clicked.connect(
                    lambda _c, val=v: self._fps.setValue(float(val))
                )
                fps_layout.addWidget(btn)
            form.addRow("Frame rate:", fps_row)
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
        """Returns the user's choices. Call only after exec() == Accepted.

        The returned ``fields`` tuple preserves the order the items were
        shown to the user — predefined components first, then any DICOM
        tag identifiers — so the output filename is reproducible."""
        chosen = tuple(
            k for k in self._field_order
            if k in self._checks and self._checks[k].isChecked()
        )
        if self._mode == "mp4":
            crf = (int(self._crf.value())
                   if self._mode_crf.isChecked() else None)
            return ExportSettings(
                fields=chosen,
                bitrate_mbps=int(self._bitrate.value()),
                fps=float(self._fps.value()),
                crf=crf,
            )
        return ExportSettings(fields=chosen)
