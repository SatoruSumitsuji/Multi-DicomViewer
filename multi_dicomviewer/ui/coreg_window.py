"""IVUS–XA co-registration window.

Shows one representative angiographic view (or up to a few) alongside one
or more IVUS pull-backs of the SAME vessel. The user traces a smooth
*guide* curve along the vessel on each angio, then pins *CoSync points*
(anatomical landmarks): for a landmark, scrub an IVUS to the frame that
shows it and slide the on-guide circle on each angio to where that
cross-section sits. Once ≥2 landmarks are pinned, scrubbing the active
IVUS drives a marker that rides each angio's guide, showing which
cross-section the current IVUS frame corresponds to.

Data model (kept here; the pure math is in :mod:`core.coreg`):

* **guide** — per angio pane: the user's control vertices + the smoothed
  curve, both in image-pixel coords. One vessel ⇒ one guide per view.
* **landmark** — a shared anatomical point: ``frames`` maps an IVUS
  pane-index → the frame that shows it (skippable per IVUS), ``fracs``
  maps an angio pane-index → the arc-length fraction on that view's guide
  (skippable per view). Both sides can be skipped where the landmark is
  not identifiable.

Driving is by a single *active* IVUS (master); switch it to review
pre/post/FU at the same anatomy. The panes are deliberately lightweight
(bare ImageCanvas + a scrub slider) so the window stays uncluttered.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.core import coreg, dicom_io
from multi_dicomviewer.core.study_model import Modality, Series
from multi_dicomviewer.i18n import t
from multi_dicomviewer.viewers.image_canvas import ImageCanvas
from multi_dicomviewer.viewers.xa_viewer import apply_window


def _grid_dims(n: int) -> int:
    """Number of columns for *n* panes: 1 for a single pane, 2 up to four,
    3 for five or six."""
    if n <= 1:
        return 1
    if n <= 4:
        return 2
    return 3


class _CoregPane:
    """One pane: an ImageCanvas + a scrub slider bound to a series.

    Role follows the series modality: an IVUS series is a scrubbable
    master candidate; any other (XA/angio) is a still view that carries
    the guide + marker overlay and its editing."""

    def __init__(self, index: int, owner: "CoregWindow"):
        self.index = index
        self.owner = owner
        self.series: Series | None = None
        self.plane = None
        self.total = 0
        self.cur = 0
        self.window = 1.0
        self.level = 0.5
        self.is_color = False
        self.is_ivus = False

        self.frame = QFrame()
        self.frame.setFrameShape(QFrame.Shape.Box)
        col = QVBoxLayout(self.frame)
        col.setContentsMargins(3, 3, 3, 3)
        col.setSpacing(2)

        self.title = QLabel("—")
        self.title.setStyleSheet("QLabel { color:#cfcfcf; }")
        col.addWidget(self.title)

        self.canvas = ImageCanvas()
        col.addWidget(self.canvas, 1)

        srow = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.lbl = QLabel("—")
        self.lbl.setMinimumWidth(90)
        srow.addWidget(self.slider, 1)
        srow.addWidget(self.lbl)
        col.addLayout(srow)

        # IVUS-only row: Master radio + a placement "include" checkbox.
        self.nav = QHBoxLayout()
        self.master_radio = QRadioButton(t("Master"))
        self.master_radio.setToolTip(
            t("Make this the active IVUS — its frame drives the markers"))
        self.master_radio.toggled.connect(self._on_master_toggled)
        self.include_cb = QCheckBox(t("Include in this point"))
        self.include_cb.setChecked(True)
        self.include_cb.setToolTip(
            t("While adding a CoSync point, include this IVUS's current frame"))
        self.include_cb.setVisible(False)
        self.nav.addWidget(self.master_radio)
        self.nav.addStretch(1)
        self.nav.addWidget(self.include_cb)
        col.addLayout(self.nav)

    def retranslate_ui(self) -> None:
        """Re-apply this pane's persistent strings on a live language change."""
        self.master_radio.setText(t("Master"))
        self.master_radio.setToolTip(
            t("Make this the active IVUS — its frame drives the markers"))
        self.include_cb.setText(t("Include in this point"))
        self.include_cb.setToolTip(
            t("While adding a CoSync point, include this IVUS's current frame"))

    # ---- loading ----
    def load(self, series: Series | None, plane_index: int = 0,
             start_frame: int = 0) -> None:
        """Load *series* into this pane. *plane_index* selects which biplane
        projection to show (0 = Lt / left, 1 = Rt / right) so the pane
        mirrors what the source pane displayed; ignored for single-plane.
        *start_frame* opens on the frame the source pane was showing (so an
        angio still keeps the frame the user picked, not frame 0)."""
        self.series = series
        if series is None:
            self.plane = None
            self.total = 0
            self.title.setText("—")
            self.slider.setEnabled(False)
            self.lbl.setText("—")
            self.canvas.set_frame(np.zeros((8, 8), np.uint8))
            return
        loaded = dicom_io.load_xa(series)
        planes = loaded.xa_planes or [None]
        pidx = plane_index if 0 <= plane_index < len(planes) else 0
        self.plane = planes[pidx]
        self.total = self.plane.total_frames if self.plane else 0
        self.is_color = bool(loaded.is_color
                             or getattr(self.plane, "is_color", False))
        self.is_ivus = (series.modality == Modality.IVUS)
        self._init_wl(loaded)
        self.canvas.spacing_mm = dicom_io.series_spacing_mm(series)
        self.master_radio.setVisible(self.is_ivus)
        # Angio panes carry the CoSync overlay; IVUS panes do not but get
        # MultiSync-style drag-to-rotate of the cross-section.
        self.canvas.set_coreg_visible(not self.is_ivus)
        self.canvas.set_free_rotation_enabled(self.is_ivus)
        mod = series.modality.value if hasattr(series.modality, "value") \
            else str(series.modality)
        side = f" {'Lt' if pidx == 0 else 'Rt'}" if len(planes) > 1 else ""
        self.title.setText(f"{mod}{side} — {getattr(series, 'label', '')}")
        start = (max(0, min(int(start_frame), self.total - 1))
                 if self.total > 0 else 0)
        self.cur = start
        self.slider.blockSignals(True)
        # IVUS panes scrub freely (they drive CoSync); an ANGIO pane is a still
        # — its seekbar is greyed out and only re-enabled while its Trace is on
        # (see _toggle_trace), so the shown angio frame stays fixed after trace.
        self.slider.setEnabled(self.total > 1 and self.is_ivus)
        self.slider.setRange(0, max(0, self.total - 1))
        self.slider.setValue(start)
        self.slider.blockSignals(False)
        self.show_frame(start)

    def _init_wl(self, loaded) -> None:
        """Default window/level: the loaded values when present, else the
        first frame's value range (mirrors what the IVUS viewer does)."""
        self.window = float(loaded.window or 0.0)
        self.level = float(loaded.level or 0.0)
        if self.window <= 0 and self.plane is not None and not self.is_color:
            f = self.plane.frame(0)
            vmin, vmax = float(np.min(f)), float(np.max(f))
            self.window = max(vmax - vmin, 1.0)
            self.level = (vmax + vmin) / 2.0

    def _display_frame(self, idx: int) -> np.ndarray:
        f = self.plane.frame(idx)
        if self.is_color:
            return f
        return apply_window(f, self.window, self.level)

    def show_frame(self, idx: int) -> None:
        if self.plane is None:
            return
        idx = max(0, min(int(idx), self.total - 1))
        self.cur = idx
        self.canvas.set_frame(self._display_frame(idx))
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.lbl.setText(f"F {idx + 1} / {self.total}")

    # ---- events ----
    def _on_slider(self, value: int) -> None:
        if self.plane is None:
            return
        self.show_frame(int(value))
        if self.is_ivus:
            self.owner._on_ivus_scrub(self.index)

    def _on_master_toggled(self, on: bool) -> None:
        if on:
            self.owner._set_active_ivus(self.index)

    def _on_slider_pressed(self) -> None:
        # Grabbing an IVUS seekbar means the user is positioning for a NEW
        # CoSync point → drop any landmark selected for modify.
        if self.is_ivus:
            self.owner._on_ivus_grab()


class CoregWindow(QMainWindow):
    """Standalone IVUS–XA co-registration window."""

    #: Marker-style toggles: (state key, English label, default on).
    _MARKER_LABELS = (
        ("guide", "Guide line", True),
        ("point", "Point", True),
        ("cutline", "Cut line", False),
        ("arrow", "Direction arrow", False),
    )

    def __init__(self, pane_specs, parent=None):
        """*pane_specs*: list of ``(Series, plane_index)`` — one per pane to
        create, mirroring the source panes on screen. plane_index picks the
        biplane projection (0=Lt, 1=Rt); a biplane pane shown as "Bi"
        contributes two specs (both planes)."""
        super().__init__(parent)
        self.setWindowTitle("CoSync")
        self.resize(1240, 900)

        self.panes: list[_CoregPane] = []
        #: pane_idx -> {"vertices": [...], "curve": [...]} for angio panes.
        self._guides: dict[int, dict] = {}
        #: landmarks: {"frames": {ivus_idx: frame}, "fracs": {xa_idx: s}}.
        self._landmarks: list[dict] = []
        self._active_ivus: int = -1
        self._tracing: int = -1          # angio pane-idx being traced, or -1
        self._placing: bool = False      # adding/modifying a CoSync point
        self._pending_fracs: dict[int, float] = {}   # xa_idx -> provisional s
        self._selected_lm: int = -1      # CoSync point selected (blue) / modify
        self._syncing: bool = False      # re-entrancy guard for IVUS↔IVUS sync
        #: Global long-axis seekbar state (unified pull-back timeline).
        self._global_flipped = False     # ⇄ reverses distal/proximal
        self._global_gmin = 0            # master-frame-extended range
        self._global_gmax = 0

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left column: the pane grid + a full-width global long-axis seekbar
        # under it (spanning the image area, NOT the right control panel).
        left = QWidget()
        lcol = QVBoxLayout(left)
        lcol.setContentsMargins(0, 0, 0, 0)
        lcol.setSpacing(2)
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        lcol.addWidget(self._grid_host, 1)
        lcol.addWidget(self._build_global_seek())
        root.addWidget(left, 1)
        root.addWidget(self._build_panel())

        self._load_series(pane_specs)
        self._refresh_global_seek()

    # ------------------------------------------------------------- build
    def _build_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(300)
        v = QVBoxLayout(panel)

        # Guide section — one Trace/Clear row per angio pane (filled later).
        self._gbox = QGroupBox(t("Guide line (vessel trace)"))
        self._guide_box = QVBoxLayout(self._gbox)
        v.addWidget(self._gbox)

        # CoSync points section.
        self._cbox = QGroupBox(t("CoSync points"))
        cv = QVBoxLayout(self._cbox)
        # Mode toggle: white = click to enter create/modify (goes blue); blue
        # = click to commit (確定). Modifies the selected CoSync point, or
        # creates a new one when none is selected.
        self._add_btn = QPushButton(t("Add / edit / commit CoSync point"))
        self._add_btn.clicked.connect(self._on_mode_button)
        cv.addWidget(self._add_btn)
        self._sort_btn = QPushButton(t("Sort CoSync points"))
        self._sort_btn.setToolTip(t("Sort by the active IVUS frame order"))
        self._sort_btn.clicked.connect(self._sort_landmarks)
        cv.addWidget(self._sort_btn)
        self._clear_all_btn = QPushButton(t("Delete all CoSync points"))
        self._clear_all_btn.clicked.connect(self._clear_all_landmarks)
        cv.addWidget(self._clear_all_btn)
        # Persist CoSync points + guide traces to a file, and reload them
        # (also imports a legacy MultiSync .txt as IVUS-only sync points).
        io_row = QHBoxLayout()
        io_row.setContentsMargins(0, 0, 0, 0)
        self._save_btn = QPushButton(t("Save CoSync…"))
        self._save_btn.setToolTip(
            t("Save the CoSync points and guide traces to a file"))
        self._save_btn.clicked.connect(self._save_coreg)
        io_row.addWidget(self._save_btn)
        self._load_btn = QPushButton(t("Load CoSync…"))
        self._load_btn.setToolTip(
            t("Load CoSync points / guide traces from a file "
              "(or import a MultiSync .txt)"))
        self._load_btn.clicked.connect(self._load_coreg)
        io_row.addWidget(self._load_btn)
        cv.addLayout(io_row)
        # Record the co-registration in motion: scrub the master IVUS end to
        # end, each frame driving the followers + the angio markers, and write
        # the composite of every pane to an MP4.
        self._mp4_btn = QPushButton(t("Export MP4…"))
        self._mp4_btn.setToolTip(
            t("Scrub the master IVUS start→end and record every pane "
              "(with the moving angio marker) to an MP4"))
        self._mp4_btn.clicked.connect(self._export_mp4)
        cv.addWidget(self._mp4_btn)
        self._cancel_btn = QPushButton(t("Cancel"))
        self._cancel_btn.clicked.connect(self._cancel_placing)
        self._cancel_btn.setVisible(False)
        cv.addWidget(self._cancel_btn)
        # Commit a CoSync point with NO XA position — for landmarks that are
        # only identifiable on IVUS (used for IVUS↔IVUS sync only).
        self._exclude_xa_cb = QCheckBox(t("Exclude XA (IVUS sync only)"))
        self._exclude_xa_cb.setToolTip(
            t("Don't set a position on XA; use this CoSync point only for "
              "IVUS sync"))
        self._exclude_xa_cb.setVisible(False)
        cv.addWidget(self._exclude_xa_cb)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("QLabel { color:#88ccff; }")
        cv.addWidget(self._status)
        # One row per CoSync point: [CoSync-N (jump)] [✕ (delete)].
        self._lm_host = QWidget()
        self._lm_box = QVBoxLayout(self._lm_host)
        self._lm_box.setContentsMargins(0, 0, 0, 0)
        self._lm_box.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._lm_host)
        cv.addWidget(scroll, 1)
        v.addWidget(self._cbox, 1)

        # Marker style toggles. Labels kept as English keys so retranslate_ui
        # can re-apply them on a live language change.
        self._sbox = QGroupBox(t("Marker display"))
        sv = QVBoxLayout(self._sbox)
        self._style_cbs: dict[str, QCheckBox] = {}
        for key, label, on in self._MARKER_LABELS:
            cb = QCheckBox(t(label))
            cb.setChecked(on)
            cb.toggled.connect(self._apply_style)
            sv.addWidget(cb)
            self._style_cbs[key] = cb
        v.addWidget(self._sbox)
        return panel

    def retranslate_ui(self) -> None:
        """Re-apply every persistent string after a LIVE language change (the
        shell calls this via its open-tool-window cascade). On-demand file /
        rename dialogs and the transient status line are left to their next
        use. The window title is the product name (kept)."""
        self._gbox.setTitle(t("Guide line (vessel trace)"))
        self._cbox.setTitle(t("CoSync points"))
        self._sbox.setTitle(t("Marker display"))
        self._add_btn.setText(t("Add / edit / commit CoSync point"))
        self._sort_btn.setText(t("Sort CoSync points"))
        self._sort_btn.setToolTip(t("Sort by the active IVUS frame order"))
        self._clear_all_btn.setText(t("Delete all CoSync points"))
        self._save_btn.setText(t("Save CoSync…"))
        self._save_btn.setToolTip(
            t("Save the CoSync points and guide traces to a file"))
        self._load_btn.setText(t("Load CoSync…"))
        self._load_btn.setToolTip(
            t("Load CoSync points / guide traces from a file "
              "(or import a MultiSync .txt)"))
        self._mp4_btn.setText(t("Export MP4…"))
        self._mp4_btn.setToolTip(
            t("Scrub the master IVUS start→end and record every pane "
              "(with the moving angio marker) to an MP4"))
        self._cancel_btn.setText(t("Cancel"))
        self._exclude_xa_cb.setText(t("Exclude XA (IVUS sync only)"))
        self._exclude_xa_cb.setToolTip(
            t("Don't set a position on XA; use this CoSync point only for "
              "IVUS sync"))
        for key, label, _on in self._MARKER_LABELS:
            cb = self._style_cbs.get(key)
            if cb is not None:
                cb.setText(t(label))
        for pane in self.panes:
            pane.retranslate_ui()
            trace = getattr(pane, "_trace_btn", None)
            if trace is not None:
                trace.setText(t("Trace"))
            clr = getattr(pane, "_clear_btn", None)
            if clr is not None:
                clr.setText(t("Clear"))
        self._refresh_list()          # landmark rows re-render via t()
        self._global_lbl.setText(t("Long-axis sync"))
        self._global_flip_btn.setToolTip(t("Flip the distal/proximal direction"))

    # ------------------------------------------- global long-axis seekbar
    def _build_global_seek(self) -> QWidget:
        """Full-width seekbar under the pane grid: a UNIFIED pull-back timeline
        that scrubs every IVUS/OCT together across the combined distal→proximal
        range (the master's frame axis, extended to cover followers whose
        pull-back reaches beyond it), so a range whose ends live in different
        pull-backs is still traversable with one bar. Separate from the Master
        radio. Hidden with <2 IVUS; greyed until a CoSync point links them."""
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(6, 0, 6, 2)
        self._global_lbl = QLabel(t("Long-axis sync"))
        self._global_slider = QSlider(Qt.Orientation.Horizontal)
        self._global_slider.valueChanged.connect(self._on_global_seek)
        self._global_flip_btn = QPushButton("⇄")
        self._global_flip_btn.setFixedWidth(32)
        self._global_flip_btn.setCheckable(True)
        self._global_flip_btn.setToolTip(t("Flip the distal/proximal direction"))
        self._global_flip_btn.toggled.connect(self._on_global_flip)
        row.addWidget(self._global_lbl)
        row.addWidget(self._global_slider, 1)
        row.addWidget(self._global_flip_btn)
        self._global_row = w
        return w

    def _ivus_panes(self):
        return [p for p in self.panes if p.is_ivus and p.total > 0]

    def _global_range(self, R):
        """(gmin, gmax) integer master-frame-extended range covering every
        IVUS's [0, total-1] mapped into master R's frame axis, or None."""
        if R < 0 or R >= len(self.panes):
            return None
        master = self.panes[R]
        gmin, gmax = 0.0, float(master.total - 1)
        for q in self.panes:
            if q is master or not q.is_ivus or q.total < 1:
                continue
            qm = [(lm["frames"][q.index], lm["frames"][R])
                  for lm in self._landmarks
                  if q.index in lm.get("frames", {})
                  and R in lm.get("frames", {})]
            for qf in (0, q.total - 1):
                g = coreg.map_between(qm, qf)
                if g is not None:
                    gmin, gmax = min(gmin, g), max(gmax, g)
        if gmax <= gmin:
            return None
        return int(math.floor(gmin)), int(math.ceil(gmax))

    def _slider_to_g(self, s):
        return (self._global_gmax - s if self._global_flipped
                else self._global_gmin + s)

    def _g_to_slider(self, g):
        return int(round(self._global_gmax - g if self._global_flipped
                         else g - self._global_gmin))

    def _refresh_global_seek(self):
        """Show the bar with ≥2 IVUS; enable + (re)range it once a CoSync point
        links them; keep the handle on the master's current frame."""
        if not hasattr(self, "_global_slider"):
            return
        self._global_row.setVisible(len(self._ivus_panes()) >= 2)
        R = self._active_ivus
        rng = self._global_range(R) if (R >= 0 and self._landmarks) else None
        on = len(self._ivus_panes()) >= 2 and rng is not None
        for w in (self._global_slider, self._global_flip_btn, self._global_lbl):
            w.setEnabled(on)
        if not on:
            return
        self._global_gmin, self._global_gmax = rng
        self._global_slider.blockSignals(True)
        self._global_slider.setRange(
            0, max(1, self._global_gmax - self._global_gmin))
        self._global_slider.setValue(self._g_to_slider(self.panes[R].cur))
        self._global_slider.blockSignals(False)

    def _reflect_global(self):
        """Move the global handle to reflect the master's current frame."""
        R = self._active_ivus
        if R < 0 or not self._global_slider.isEnabled():
            return
        self._global_slider.blockSignals(True)
        self._global_slider.setValue(self._g_to_slider(self.panes[R].cur))
        self._global_slider.blockSignals(False)

    def _on_global_flip(self, checked):
        self._global_flipped = bool(checked)
        self._reflect_global()

    def _on_global_seek(self, s):
        """Drive every IVUS from the unified timeline: the master clamps to its
        own range while a follower whose pull-back extends further keeps moving
        (map_between extrapolates), then markers + rotations follow."""
        if self._syncing or self._placing:
            return
        R = self._active_ivus
        if R < 0:
            return
        g = self._slider_to_g(s)
        self._syncing = True
        try:
            self.panes[R].show_frame(int(round(g)))     # clamps to master range
            for q in self.panes:
                if q is self.panes[R] or not q.is_ivus or q.total < 1:
                    continue
                pairs = [(lm["frames"][R], lm["frames"][q.index])
                         for lm in self._landmarks
                         if R in lm.get("frames", {})
                         and q.index in lm.get("frames", {})]
                f = coreg.map_between(pairs, g)
                if f is not None:
                    q.show_frame(int(round(f)))
        finally:
            self._syncing = False
        for p in self.panes:
            self._apply_rotation(p)
        # Drive the angio marker by the UNIFIED position g (not the master's
        # clamped frame) so the green point sweeps the WHOLE bar range.
        self._drive_markers(g)

    def _load_series(self, pane_specs) -> None:
        specs = [(s, p, f) for (s, p, f) in (pane_specs or [])
                 if s is not None][:6]
        cols = _grid_dims(len(specs))
        for i, (se, pidx, start) in enumerate(specs):
            pane = _CoregPane(i, self)
            self.panes.append(pane)
            self._grid.addWidget(pane.frame, i // cols, i % cols)
            pane.load(se, pidx, start)
            if not pane.is_ivus:
                self._wire_angio(pane)
                self._guides[i] = {"vertices": [], "curve": [], "frame": None}
                self._add_guide_row(pane)
        # Pick the first IVUS pane as the active master.
        for pane in self.panes:
            if pane.is_ivus:
                pane.master_radio.setChecked(True)
                break
        self._apply_style()
        self._refresh_list()

    def _wire_angio(self, pane: _CoregPane) -> None:
        c = pane.canvas
        c.set_coreg_visible(True)
        c.coreg_guide_changed.connect(
            lambda verts, pi=pane.index: self._on_guide_changed(pi, verts))
        c.coreg_guide_finished.connect(
            lambda pi=pane.index: self._on_guide_finished(pi))
        c.coreg_slider_moved.connect(
            lambda s, pi=pane.index: self._on_slider_moved(pi, s))
        c.coreg_anchor_clicked.connect(
            lambda idx, pi=pane.index: self._on_anchor_clicked(pi, idx))
        c.coreg_marker_dragged.connect(
            lambda s, pi=pane.index: self._on_marker_dragged(pi, s))

    def _add_guide_row(self, pane: _CoregPane) -> None:
        row = QHBoxLayout()
        # Number by angio order (#1, #2 …) so it matches the per-guide ✓
        # marks shown on each CoSync-point row. This pane is already appended.
        angio_no = sum(1 for p in self.panes if not p.is_ivus)
        mod = pane.series.modality.value if pane.series else "?"
        lbl = QLabel(f"#{angio_no} {mod}")
        trace = QPushButton(t("Trace"))
        trace.setCheckable(True)
        trace.toggled.connect(
            lambda on, pi=pane.index: self._toggle_trace(pi, on))
        clr = QPushButton(t("Clear"))
        clr.clicked.connect(lambda _=False, pi=pane.index: self._clear_guide(pi))
        row.addWidget(lbl)
        row.addStretch(1)
        row.addWidget(trace)
        row.addWidget(clr)
        self._guide_box.addLayout(row)
        pane._trace_btn = trace          # keep for programmatic untoggle
        pane._clear_btn = clr            # keep for live retranslate

    # ------------------------------------------------- guide editing
    def _toggle_trace(self, pane_idx: int, on: bool) -> None:
        pane = self.panes[pane_idx]
        if on:
            # Only one pane traces at a time.
            if self._tracing >= 0 and self._tracing != pane_idx:
                self.panes[self._tracing]._trace_btn.setChecked(False)
            self._tracing = pane_idx
            # If a guide already exists, jump back to the frame it was drawn
            # on so the edit happens on the same image.
            g = self._guides.get(pane_idx, {})
            if g.get("vertices") and g.get("frame") is not None:
                pane.show_frame(int(g["frame"]))
            pane.canvas.set_coreg_edit(True)
            pane.canvas.set_coreg_mode("trace")
            # Trace ON → let this angio pane's seekbar move so the user can pick
            # the frame the guide is drawn on.
            pane.slider.setEnabled(pane.total > 1)
            self._status.setText(t(
                "Click to add a vertex, double-click to commit. Once a vertex "
                "turns red, drag to move it / right-click to delete it. "
                "Right-click on the line to add a point."))
        else:
            if self._tracing == pane_idx:
                self._tracing = -1
            pane.canvas.set_coreg_mode("")
            pane.canvas.set_coreg_edit(False)
            # Trace finished → the angio is a still again; grey out its seekbar.
            pane.slider.setEnabled(False)
            self._status.setText("")
        self._update_coreg_controls_enabled()

    def _on_guide_changed(self, pane_idx: int, vertices) -> None:
        verts = [tuple(v) for v in vertices]
        curve = coreg.smooth_curve(verts)
        # Remember the angio frame the trace is drawn on, so re-entering
        # trace later jumps back to the same image.
        self._guides[pane_idx] = {
            "vertices": verts, "curve": curve,
            "frame": self.panes[pane_idx].cur,
        }
        self.panes[pane_idx].canvas.set_coreg_guide(verts, curve)
        self._drive_markers()

    def _on_guide_finished(self, pane_idx: int) -> None:
        btn = getattr(self.panes[pane_idx], "_trace_btn", None)
        if btn is not None:
            btn.setChecked(False)            # triggers _toggle_trace(off)

    def _clear_guide(self, pane_idx: int) -> None:
        self._guides[pane_idx] = {"vertices": [], "curve": [], "frame": None}
        self.panes[pane_idx].canvas.set_coreg_guide([], [])
        self.panes[pane_idx].canvas.set_coreg_marker(None)
        self.panes[pane_idx].canvas.set_coreg_anchors([])

    # ------------------------------------------------- CoSync placement
    def _style_blue(self, btn, on: bool) -> None:
        btn.setStyleSheet(
            "QPushButton { background:#2d7ff0; color:white; font-weight:bold; }"
            if on else "")

    def _on_mode_button(self) -> None:
        """Top toggle: white → enter create/modify (blue); blue → commit."""
        if self._placing:
            self._commit_placing()
        else:
            self._start_placing()

    def _start_placing(self) -> None:
        if self._active_ivus < 0:
            QMessageBox.information(self, "CoSync", t("There is no IVUS."))
            return
        # XA is optional: with no angio panes this is a pure multi-IVUS sync
        # point. (An untraced XA is handled at commit — the marker just has
        # no guide, or the user turns on XA除外.)
        has_angio = any(not p.is_ivus for p in self.panes)
        self._placing = True
        modifying = 0 <= self._selected_lm < len(self._landmarks)
        # Modify: seed from the point's existing angio positions. New: empty.
        self._pending_fracs = (
            dict(self._landmarks[self._selected_lm]["fracs"])
            if modifying else {})
        # XA-exclude toggle: only meaningful when angio panes exist; default
        # on when re-editing a point that has no XA position.
        self._exclude_xa_cb.setChecked(
            modifying and not self._landmarks[self._selected_lm]["fracs"])
        self._exclude_xa_cb.setVisible(has_angio)
        for p in self.panes:
            if p.is_ivus:
                # Frame(s) were chosen before entering — lock the seekbars.
                p.slider.setEnabled(False)
                p.include_cb.setVisible(not modifying)
            else:
                p.canvas.set_coreg_edit(True)
                p.canvas.set_coreg_mode("anchor")
                # Keep the 按分 (interpolated) green point as the starting
                # position for a NEW point so it doesn't vanish — adjust from
                # there. Modify: start from the point's existing position.
                s = self._pending_fracs.get(p.index)
                if s is None and not modifying:
                    s = self._interp_frac(p.index)
                    if s is not None:
                        self._pending_fracs[p.index] = s
                curve = self._guides.get(p.index, {}).get("curve", [])
                p.canvas.set_coreg_marker(
                    coreg.point_at_fraction(curve, s)
                    if (s is not None and curve) else None)
        self._style_blue(self._add_btn, True)
        self._cancel_btn.setVisible(True)
        self._sort_btn.setVisible(False)
        self._clear_all_btn.setVisible(False)
        self._update_coreg_controls_enabled()
        if not has_angio:
            self._status.setText(t(
                "Align each IVUS to the landmark frame/rotation, then press "
                "the button again to commit (IVUS sync point)."))
        elif modifying:
            self._status.setText(t(
                "Editing CoSync-{n}: move the circle on the angio guide to "
                "the target position, then press the button again to commit.",
                n=self._selected_lm + 1))
        else:
            self._status.setText(t(
                "New CoSync point: on an angio where it can be identified, "
                "move the circle on the guide to the target position, then "
                "press the button again to commit."))

    def _on_slider_moved(self, pane_idx: int, s: float) -> None:
        # Provisional anchor for this angio while placing a CoSync point.
        if self._placing:
            self._pending_fracs[pane_idx] = float(s)

    def _commit_placing(self) -> None:
        if not self._placing:
            return
        has_angio = any(not p.is_ivus for p in self.panes)
        exclude_xa = self._exclude_xa_cb.isChecked() or not has_angio
        if not exclude_xa and not self._pending_fracs:
            QMessageBox.information(
                self, "CoSync",
                t("Place the circle on at least one angio (if you cannot "
                  "decide on XA, turn on 'Exclude XA')."))
            return
        fracs = {} if exclude_xa else dict(self._pending_fracs)
        if 0 <= self._selected_lm < len(self._landmarks):
            lm = self._landmarks[self._selected_lm]
            lm["fracs"] = fracs
            lm["rots"] = {i: self.panes[i].canvas.free_rotation()
                          for i in lm["frames"]}
        else:
            frames = {p.index: p.cur for p in self.panes
                      if p.is_ivus and p.include_cb.isChecked()}
            if not frames:
                QMessageBox.information(
                    self, "CoSync", t("Select at least one IVUS to include."))
                return
            rots = {i: self.panes[i].canvas.free_rotation() for i in frames}
            self._landmarks.append(
                {"frames": frames, "rots": rots, "fracs": fracs})
        self._end_placing()
        self._refresh_list()
        self._drive_markers()

    def _cancel_placing(self) -> None:
        self._end_placing()
        self._refresh_list()
        self._drive_markers()

    def _end_placing(self) -> None:
        self._placing = False
        self._pending_fracs = {}
        self._style_blue(self._add_btn, False)
        self._cancel_btn.setVisible(False)
        self._exclude_xa_cb.setVisible(False)
        self._sort_btn.setVisible(True)
        self._clear_all_btn.setVisible(True)
        self._status.setText("")
        for p in self.panes:
            if p.is_ivus:
                p.slider.setEnabled(p.total > 1)
                p.include_cb.setVisible(False)
            else:
                p.canvas.set_coreg_mode("")
                p.canvas.set_coreg_edit(False)
        self._update_coreg_controls_enabled()

    # ------------------------------------------------- landmark list
    def _landmark_tip(self, lm) -> str:
        ivus_idx = [p.index for p in self.panes if p.is_ivus]
        xa_idx = [p.index for p in self.panes if not p.is_ivus]
        fr = " ".join(
            f"I{i + 1}:{lm['frames'][idx] + 1}" if idx in lm["frames"]
            else f"I{i + 1}:–" for i, idx in enumerate(ivus_idx))
        xa = " ".join(
            f"A{i + 1}:{'✓' if idx in lm['fracs'] else '–'}"
            for i, idx in enumerate(xa_idx))
        return f"{fr}   {xa}"

    def _landmark_info(self, lm) -> str:
        """Two-line summary shown right of the CoSync button: line 1 = each
        IVUS's frame/angle (all IVUS, like MultiSync); line 2 = a ✓ / · per
        angio guide (#1, #2 …) showing on which traces this point exists.
        Two lines so up to 3 IVUS × 3 XA still fit at the button font size."""
        ivus = [p for p in self.panes if p.is_ivus]
        xa = [p for p in self.panes if not p.is_ivus]
        toks = []
        for p in ivus:
            if p.index in lm["frames"]:
                fr = lm["frames"][p.index] + 1
                rot = lm.get("rots", {}).get(p.index)
                toks.append(f"{fr}/{round(rot)}°" if rot is not None
                            else str(fr))
            else:
                toks.append("–")
        guides = " ".join(
            f"#{k + 1}{'✓' if p.index in lm['fracs'] else '·'}"
            for k, p in enumerate(xa))
        return f"F {'  '.join(toks)}\n{guides}"

    def _refresh_list(self) -> None:
        while self._lm_box.count():
            it = self._lm_box.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        for n, lm in enumerate(self._landmarks, 1):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(3)
            x = QPushButton("✕")          # delete, moved to the left
            x.setFixedWidth(26)
            x.setToolTip(t("Delete this CoSync point"))
            x.clicked.connect(lambda _c, i=n - 1: self._delete_landmark(i))
            b = QPushButton(lm.get("name") or f"CoSync-{n}")   # jump / rename
            b.setFixedWidth(78)
            b.setToolTip(self._landmark_tip(lm) + t("\n(right-click to rename)"))
            b.clicked.connect(lambda _c, i=n - 1: self._goto_landmark(i))
            b.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            b.customContextMenuRequested.connect(
                lambda _pos, i=n - 1: self._rename_landmark(i))
            self._style_blue(b, (n - 1) == self._selected_lm)
            info = QLabel(self._landmark_info(lm))   # black, button-size text
            info.setStyleSheet("QLabel { color:black; }")
            h.addWidget(x)
            h.addWidget(b)
            h.addWidget(info, 1)
            self._lm_box.addWidget(row)
        self._refresh_global_seek()      # landmark set changed → re-range bar

    def _goto_landmark(self, idx: int) -> None:
        """Clicking CoSync-N selects it (blue) and jumps every IVUS that
        recorded this landmark to its frame; the active IVUS then drives the
        angio markers onto it. Ignored mid-placement."""
        if self._placing or not (0 <= idx < len(self._landmarks)):
            return
        self._selected_lm = idx
        lm = self._landmarks[idx]
        self._syncing = True
        try:
            for pane in self.panes:
                if pane.is_ivus and pane.index in lm["frames"]:
                    pane.show_frame(int(lm["frames"][pane.index]))
        finally:
            self._syncing = False
        for pane in self.panes:
            self._apply_rotation(pane)
        self._drive_markers()
        self._refresh_list()

    def _rename_landmark(self, idx: int) -> None:
        """Right-click a CoSync button → give it a short name (≤8 chars) like
        R1os / LCX / LAD / D9a instead of CoSync-N. Blank clears it."""
        if not (0 <= idx < len(self._landmarks)):
            return
        cur = self._landmarks[idx].get("name", "")
        text, ok = QInputDialog.getText(
            self, t("Rename"), t("Name (up to 8 characters):"), text=cur)
        if ok:
            self._landmarks[idx]["name"] = text.strip()[:8]
            self._refresh_list()

    def _delete_landmark(self, idx: int) -> None:
        if not (0 <= idx < len(self._landmarks)):
            return
        del self._landmarks[idx]
        if self._selected_lm == idx:
            self._selected_lm = -1
        elif self._selected_lm > idx:
            self._selected_lm -= 1
        self._refresh_list()
        self._drive_markers()

    def _sort_landmarks(self) -> None:
        """Order CoSync points along the active IVUS's pull-back (by its
        frame); points without an active-IVUS frame sort by their smallest
        frame and fall after."""
        R = self._active_ivus

        def key(lm):
            if R in lm["frames"]:
                return (0, lm["frames"][R])
            return (1, min(lm["frames"].values()) if lm["frames"] else 0)

        self._landmarks.sort(key=key)
        self._selected_lm = -1
        self._refresh_list()

    def _clear_all_landmarks(self) -> None:
        if not self._landmarks:
            return
        if QMessageBox.question(
                self, "CoSync", t("Delete all CoSync points?")
        ) == QMessageBox.StandardButton.Yes:
            self._landmarks.clear()
            self._selected_lm = -1
            self._refresh_list()
            self._drive_markers()

    # ------------------------------------------------- save / load
    def _save_coreg(self) -> None:
        """Write the CoSync points + guide traces to a JSON file. The guide
        curves are NOT stored (re-derived from the vertices on load)."""
        if not self._landmarks and not any(
                g.get("vertices") for g in self._guides.values()):
            QMessageBox.information(
                self, "CoSync", t("Nothing to save yet."))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, t("Save CoSync"), "", t("CoSync file (*.cosync.json)"))
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".cosync.json"
        data = {
            "format": "MDV-CoSync",
            "version": 1,
            "panes": [
                {"index": p.index, "ivus": bool(p.is_ivus),
                 "label": (p.series.label if p.series is not None else "")}
                for p in self.panes
            ],
            "guides": {
                str(i): {"vertices": [[float(x), float(y)] for (x, y) in
                                      g.get("vertices", [])],
                         "frame": g.get("frame")}
                for i, g in self._guides.items() if g.get("vertices")
            },
            "landmarks": [
                {"frames": {str(k): int(v) for k, v in lm.get("frames", {}).items()},
                 "rots": {str(k): float(v) for k, v in lm.get("rots", {}).items()},
                 "fracs": {str(k): float(v) for k, v in lm.get("fracs", {}).items()},
                 "name": lm.get("name", "")}
                for lm in self._landmarks
            ],
        }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=1)
        except OSError as exc:
            QMessageBox.warning(self, t("Save CoSync"), str(exc))
            return
        self._status.setText(
            t("Saved {n} CoSync point(s).", n=len(self._landmarks)))

    def _load_coreg(self) -> None:
        """Load CoSync points + guide traces from a native .cosync.json, or
        import a legacy MultiSync .txt as IVUS-only sync points."""
        path, _ = QFileDialog.getOpenFileName(
            self, t("Load CoSync"), "",
            t("CoSync / MultiSync (*.json *.txt);;All files (*)"))
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            QMessageBox.warning(self, t("Load CoSync"), str(exc))
            return
        if path.lower().endswith(".txt") or text.lstrip().startswith("#"):
            landmarks, guides = self._parse_multisync_txt(text), {}
        else:
            try:
                data = json.loads(text)
            except ValueError as exc:
                QMessageBox.warning(self, t("Load CoSync"), str(exc))
                return
            landmarks, guides = self._parse_coreg_json(data)
        if landmarks is None:
            QMessageBox.warning(
                self, t("Load CoSync"), t("Unrecognised CoSync file."))
            return
        self._apply_loaded(landmarks, guides)

    def _parse_coreg_json(self, data):
        """(landmarks, guides) from a native file, with int-keyed maps —
        or (None, None) if the file is not a CoSync file."""
        # Accept the current format and the pre-rename "MDV-CoReg" files.
        if (not isinstance(data, dict)
                or data.get("format") not in ("MDV-CoSync", "MDV-CoReg")):
            return None, None
        lms = []
        for lm in data.get("landmarks", []):
            lms.append({
                "frames": {int(k): int(v) for k, v in lm.get("frames", {}).items()},
                "rots": {int(k): float(v) for k, v in lm.get("rots", {}).items()},
                "fracs": {int(k): float(v) for k, v in lm.get("fracs", {}).items()},
                "name": str(lm.get("name", "")),
            })
        guides = {}
        for k, g in data.get("guides", {}).items():
            verts = [tuple(v) for v in g.get("vertices", [])]
            if verts:
                guides[int(k)] = {"vertices": verts, "frame": g.get("frame")}
        return lms, guides

    def _parse_multisync_txt(self, text: str):
        """Import a MultiSync '# MultiSync settings v1' .txt: each SyncPoint's
        per-slot frame/rotation becomes an IVUS-only CoSync landmark. Slot k
        maps to the k-th IVUS pane on screen (best effort; no XA fracs)."""
        ivus_panes = [p.index for p in self.panes if p.is_ivus]
        lms = []
        in_pts = False
        for raw in text.splitlines():
            line = raw.rstrip("\n")
            if line.startswith("[SyncPoints]"):
                in_pts = True
                continue
            if line.startswith("[") and line != "[SyncPoints]":
                in_pts = False
            if not in_pts or not line.strip() or line.lstrip().startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            name = cols[0].strip()
            rest = cols[1:]
            half = len(rest) // 2
            frames_raw, rots_raw = rest[:half], rest[half:]
            frames, rots = {}, {}
            for slot, fv in enumerate(frames_raw):
                if slot >= len(ivus_panes) or fv.strip() in ("", "-"):
                    continue
                pi = ivus_panes[slot]
                try:
                    frames[pi] = int(float(fv))
                    if slot < len(rots_raw):
                        rots[pi] = float(rots_raw[slot])
                except ValueError:
                    continue
            if frames:
                lms.append({"frames": frames, "rots": rots,
                            "fracs": {}, "name": name[:8]})
        return lms

    def _apply_loaded(self, landmarks, guides) -> None:
        """Replace the current CoSync points + guides with loaded ones and
        redraw. Guide curves are re-derived from vertices here."""
        self._landmarks = landmarks
        self._selected_lm = -1
        # Guides: keep only those whose pane still exists and is an angio pane.
        valid = {p.index for p in self.panes if not p.is_ivus}
        for i in list(self._guides):
            self._clear_guide(i)
        for i, g in (guides or {}).items():
            if i not in valid:
                continue
            verts = [tuple(v) for v in g["vertices"]]
            curve = coreg.smooth_curve(verts)
            self._guides[i] = {"vertices": verts, "curve": curve,
                               "frame": g.get("frame")}
            self.panes[i].canvas.set_coreg_guide(verts, curve)
        self._refresh_list()
        self._drive_markers()
        self._status.setText(
            t("Loaded {n} CoSync point(s).", n=len(self._landmarks)))

    # ------------------------------------------------- MP4 export
    @staticmethod
    def _qimage_to_rgb(img: QImage) -> np.ndarray:
        """QImage → owned contiguous (H, W, 3) uint8 RGB (a real copy, not a
        view — see the MultiSync note: a view aliases freed pixel memory)."""
        img = img.convertToFormat(QImage.Format.Format_RGB888)
        w, h = img.width(), img.height()
        bpl = img.bytesPerLine()
        ptr = img.constBits()
        ptr.setsize(h * bpl)
        arr = np.frombuffer(ptr, np.uint8).reshape(h, bpl)
        return np.array(arr[:, :w * 3].reshape(h, w, 3))

    def _compose_coreg_frame(self, cell: int, cols: int) -> np.ndarray:
        """Grab every pane's canvas (image + guide + moving marker, exactly as
        shown), CROP off the canvas letterbox so only the image region remains,
        and tile them into a cols-wide grid of square *cell* cells — so the
        image fills its cell instead of sitting small inside a double margin."""
        rows = (len(self.panes) + cols - 1) // cols
        img = QImage(cell * cols, cell * rows, QImage.Format.Format_RGB888)
        img.fill(QColor("#000000"))
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        try:
            for k, pane in enumerate(self.panes):
                canvas = pane.canvas
                qimg = canvas.grab().toImage()
                gw, gh = qimg.width(), qimg.height()
                if gw <= 0 or gh <= 0:
                    continue
                # Crop to the image's draw rect (drops the fit letterbox),
                # mapping logical widget px → grabbed px (HiDPI-safe).
                dr = canvas._draw_rect()
                cw, ch = canvas.width(), canvas.height()
                if dr.isValid() and cw > 0 and ch > 0:
                    sx, sy = gw / cw, gh / ch
                    cx = max(0, int(round(dr.x() * sx)))
                    cy = max(0, int(round(dr.y() * sy)))
                    cw2 = min(gw - cx, int(round(dr.width() * sx)))
                    ch2 = min(gh - cy, int(round(dr.height() * sy)))
                    if cw2 > 0 and ch2 > 0:
                        qimg = qimg.copy(cx, cy, cw2, ch2)
                w, h = qimg.width(), qimg.height()
                if w <= 0 or h <= 0:
                    continue
                r, c = divmod(k, cols)
                tx, ty = c * cell, r * cell
                scale = min(cell / w, cell / h)
                dw, dh = w * scale, h * scale
                p.save()
                p.setClipRect(tx, ty, cell, cell)
                p.drawImage(
                    QRectF(tx + (cell - dw) / 2.0, ty + (cell - dh) / 2.0,
                           dw, dh), qimg)
                p.restore()
        finally:
            p.end()
        return self._qimage_to_rgb(img)

    def _export_mp4(self) -> None:
        """Scrub the master IVUS from its first to last frame; each frame drives
        the follower IVUS + the angio markers (the live co-registration), and
        the composite of every pane is streamed to an MP4."""
        if self._placing:
            return
        if self._active_ivus < 0:
            QMessageBox.information(
                self, "CoSync", t("Select a master IVUS first."))
            return
        master = self.panes[self._active_ivus]
        if not master.is_ivus or master.total < 1:
            QMessageBox.information(
                self, "CoSync", t("The master IVUS has no frames."))
            return

        from multi_dicomviewer.ui.export_dialog import ExportDialog
        dlg_cfg = ExportDialog(
            "mp4", 1, default_fps=15.0, default_bitrate=40, default_crf=10,
            show_filename_fields=False,
            title_override=t("Export CoSync MP4"), parent=self)
        if dlg_cfg.exec() != dlg_cfg.DialogCode.Accepted:
            return
        cfg = dlg_cfg.result_settings()

        path, _ = QFileDialog.getSaveFileName(
            self, t("Export CoSync MP4"), "", t("MP4 video (*.mp4)"))
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"

        cols = _grid_dims(len(self.panes))
        # Square cell ≈ the largest pane canvas, clamped and made even.
        big = max((max(p.canvas.width(), p.canvas.height())
                   for p in self.panes), default=512)
        cell = max(480, min(int(big), 1024))
        cell -= cell % 2

        total = master.total
        prog = QProgressDialog(
            t("Exporting CoSync MP4…"), t("Cancel"), 0, total, self)
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)

        from multi_dicomviewer.core import export as exporter
        try:
            stream = exporter.open_mp4_stream(
                path, fps=cfg.fps, bitrate_mbps=cfg.bitrate_mbps, crf=cfg.crf)
        except Exception as e:      # noqa: BLE001
            prog.close()
            QMessageBox.critical(
                self, t("Export CoSync MP4"),
                t("Encoding failed:\n{e}", e=e))
            return

        saved = master.cur
        written, cancelled = 0, False
        try:
            for f in range(total):
                if prog.wasCanceled():
                    cancelled = True
                    break
                master.show_frame(f)
                self._on_ivus_scrub(master.index)   # sync followers + markers
                frame = self._compose_coreg_frame(cell, cols)
                stream.add(frame)
                written += 1
                if f % 8 == 0:
                    prog.setValue(f)
                    QApplication.processEvents()
        except Exception as e:      # noqa: BLE001
            stream.abort()
            prog.close()
            self._restore_master(master, saved)
            QMessageBox.critical(
                self, t("Export CoSync MP4"),
                t("Encoding failed:\n{e}", e=e))
            return
        prog.close()
        self._restore_master(master, saved)

        if cancelled or written == 0:
            stream.abort()
            return
        try:
            stream.close()
        except Exception as e:      # noqa: BLE001
            QMessageBox.critical(
                self, t("Export CoSync MP4"),
                t("Encoding failed:\n{e}", e=e))
            return
        self._status.setText(
            t("Exported {n} frame(s) to MP4.", n=written))

    def _restore_master(self, master, frame: int) -> None:
        master.show_frame(frame)
        self._on_ivus_scrub(master.index)

    def _on_ivus_grab(self) -> None:
        # User grabbed an IVUS seekbar (not during placement) → they are
        # positioning for a NEW point, so drop the modify selection.
        if self._placing:
            return
        if self._selected_lm != -1:
            self._selected_lm = -1
            self._refresh_list()

    def _update_coreg_controls_enabled(self) -> None:
        """Trace and CoSync-point editing are mutually exclusive: while a
        guide trace is active the CoSync-point controls are disabled. During
        placement the mode (確定) button stays enabled; sort / clear are
        hidden then, and the landmark rows are left enabled but their clicks
        are ignored (see _goto_landmark)."""
        tracing = self._tracing >= 0
        self._add_btn.setEnabled(not tracing)
        for b in (self._sort_btn, self._clear_all_btn):
            b.setEnabled(not tracing and not self._placing)
        self._lm_host.setEnabled(not tracing)

    # ------------------------------------------------- driving
    def _set_active_ivus(self, pane_idx: int) -> None:
        self._active_ivus = pane_idx
        self._drive_markers()
        self._refresh_global_seek()      # master changed → re-range on its axis

    def _on_ivus_scrub(self, pane_idx: int) -> None:
        """Scrubbing the MASTER IVUS moves the others in lockstep (once ≥2
        CoSync points exist) via the frames they share on the landmarks, then
        drives the angio markers. A non-master IVUS scrubs independently —
        it moves only itself and does not touch the markers."""
        if self._placing or self._syncing:
            return
        # Every scrubbed IVUS (master or not) updates its own 按分 rotation.
        self._apply_rotation(self.panes[pane_idx])
        if pane_idx != self._active_ivus:
            return
        self._syncing = True
        try:
            self._sync_ivus_from(pane_idx)
        finally:
            self._syncing = False
        for p in self.panes:
            self._apply_rotation(p)
        self._drive_markers()
        self._reflect_global()           # keep the global handle in step

    def _apply_rotation(self, pane) -> None:
        """Set *pane*'s IVUS display rotation to the angle interpolated from
        the CoSync landmarks at its current frame (按分). No-op when no
        rotation has been recorded (leaves the user's manual rotation)."""
        if not pane.is_ivus:
            return
        pairs = [(lm["frames"][pane.index], lm["rots"][pane.index])
                 for lm in self._landmarks
                 if pane.index in lm.get("frames", {})
                 and pane.index in lm.get("rots", {})]
        ang = coreg.map_rotation(pairs, pane.cur)
        if ang is not None:
            pane.canvas.set_free_rotation(ang)

    def _sync_ivus_from(self, src_idx: int) -> None:
        src = self.panes[src_idx]
        if not src.is_ivus:
            return
        for q in self.panes:
            if q is src or not q.is_ivus:
                continue
            pairs = [(lm["frames"][src_idx], lm["frames"][q.index])
                     for lm in self._landmarks
                     if src_idx in lm["frames"] and q.index in lm["frames"]]
            f = coreg.map_between(pairs, src.cur)
            if f is not None:
                q.show_frame(max(0, min(int(round(f)), q.total - 1)))

    def _interp_frac(self, pane_idx: int):
        """The 按分 (interpolated) guide fraction for angio *pane_idx* at the
        active IVUS's current frame, or None (too few anchors). Used to seed
        a new CoSync point's marker so it starts on the interpolated spot."""
        R = self._active_ivus
        if R < 0:
            return None
        anchors = [(lm["frames"][R], lm["fracs"][pane_idx])
                   for lm in self._landmarks
                   if R in lm["frames"] and pane_idx in lm["fracs"]]
        return coreg.map_frame_to_fraction(anchors, self.panes[R].cur)

    def _drive_markers(self, frame_override=None) -> None:
        """For every angio pane, map the active IVUS's frame to a point on that
        view's guide (from the landmarks common to the pair) and push the
        marker + the anchor foot-points to its canvas.

        *frame_override* lets the global long-axis seekbar drive the marker by
        the UNIFIED position (which can run past the master's own range): the
        fraction extrapolates and clamps to the guide ends, so the marker keeps
        moving over the WHOLE bar instead of freezing where the master clamps."""
        if self._active_ivus < 0 or self._placing:
            return
        R = self._active_ivus
        fR = self.panes[R].cur if frame_override is None else frame_override
        for pane in self.panes:
            if pane.is_ivus:
                continue
            curve = self._guides.get(pane.index, {}).get("curve", [])
            # Anchor foot-points on this guide (for the amber dots).
            feet = [coreg.point_at_fraction(curve, lm["fracs"][pane.index])
                    for lm in self._landmarks
                    if pane.index in lm["fracs"] and curve]
            pane.canvas.set_coreg_anchors([f for f in feet if f is not None])
            anchors = [(lm["frames"][R], lm["fracs"][pane.index])
                       for lm in self._landmarks
                       if R in lm["frames"] and pane.index in lm["fracs"]]
            s = coreg.map_frame_to_fraction(anchors, fR)
            pane.canvas.set_coreg_marker(
                None if (s is None or not curve)
                else coreg.point_at_fraction(curve, s))

    def _on_anchor_clicked(self, pane_idx: int, idx: int) -> None:
        """Review: clicking this view's idx-th amber anchor jumps the active
        IVUS to that landmark's frame, so the green marker lands on the dot."""
        if self._active_ivus < 0:
            return
        R = self._active_ivus
        lms = [lm for lm in self._landmarks if pane_idx in lm["fracs"]]
        if 0 <= idx < len(lms) and R in lms[idx]["frames"]:
            self.panes[R].slider.setValue(int(lms[idx]["frames"][R]))

    def _on_marker_dragged(self, pane_idx: int, s: float) -> None:
        """Review: dragging the green marker maps the guide fraction back to
        an IVUS frame and scrubs the active IVUS (reverse driving)."""
        if self._active_ivus < 0 or self._placing:
            return
        R = self._active_ivus
        anchors = [(lm["frames"][R], lm["fracs"][pane_idx])
                   for lm in self._landmarks
                   if R in lm["frames"] and pane_idx in lm["fracs"]]
        f = coreg.map_fraction_to_frame(anchors, s)
        if f is None:
            return
        total = self.panes[R].total
        self.panes[R].slider.setValue(max(0, min(int(round(f)), total - 1)))

    def _apply_style(self, *_a) -> None:
        style = {k: cb.isChecked() for k, cb in self._style_cbs.items()}
        for pane in self.panes:
            if not pane.is_ivus:
                pane.canvas.set_coreg_style(style)
