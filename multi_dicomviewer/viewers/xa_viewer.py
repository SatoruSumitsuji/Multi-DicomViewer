"""XA angiography viewer: multiframe cine + window/level + calibrated tools.

Handles single-plane and biplane acquisitions. The shell decides the
biplane layout via set_biplane_layout():

  * side-by-side  -> Front and Lateral shown together (used in "XA only")
  * stacked/single -> one plane at a time with Front/Lateral buttons
                      (used whenever CT shares the screen, or for mono cine)
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.config import DEFAULT_CINE_FPS
from multi_dicomviewer.core.dicom_io import (
    LoadedSeries,
    XAPlane,
    prefetch_planes,
)
from multi_dicomviewer.core.dicom_tags import overlay_lines
from multi_dicomviewer.ui.viewer_base import AbstractViewer
from multi_dicomviewer.viewers.image_canvas import ImageCanvas

#: Slider look matching the Windows build: light-grey groove, blue filled
#: (sub-page) track, and a handle that is a white circle with a blue dot in
#: the centre. Applied to the seek bar and the W/L sliders.
_SLIDER_QSS = (
    "QSlider::groove:horizontal {"
    " height:5px; background:#c9c9c9; border-radius:2px; }"
    "QSlider::sub-page:horizontal {"
    " background:#2f7fd1; border-radius:2px; }"
    "QSlider::handle:horizontal {"
    " width:16px; height:16px; margin:-6px 0; border-radius:8px;"
    " border:1px solid #9a9a9a;"
    " background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,"
    " stop:0 #2f7fd1, stop:0.55 #2f7fd1, stop:0.6 #ffffff, stop:1 #ffffff); }"
)


class _Prefetcher(QThread):
    """Decodes the remaining cine frames off the UI thread."""

    def __init__(self, planes: list[XAPlane], is_playing, parent=None):
        super().__init__(parent)
        self._planes = planes
        self._is_playing = is_playing      # () -> bool, the cine timer state
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        # Frame-major across planes so a biplane cine (which needs frame i
        # of BOTH planes before it can show i) advances in lockstep instead
        # of freezing at frame 0 until plane 0 is fully decoded.
        prefetch_planes(
            self._planes, lambda: self._stop, self._is_playing
        )


def apply_window(vol_frame: np.ndarray, window: float, level: float) -> np.ndarray:
    lo = level - window / 2.0
    out = (vol_frame - lo) / max(window, 1e-6)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _build_wl_lut(dtype, window, level):
    """uint8 lookup table so per-frame windowing is a single array gather
    (`lut[frame + offset]`) instead of float math every frame — the main
    cine-playback speed-up. Returns (lut, offset) or None for dtypes too
    wide to table (caller falls back to apply_window)."""
    if not np.issubdtype(dtype, np.integer):
        return None
    info = np.iinfo(dtype)
    n = int(info.max) - int(info.min) + 1
    if n > (1 << 16):                       # e.g. int32 — don't table
        return None
    vals = np.arange(info.min, int(info.max) + 1, dtype=np.float64)
    lo = level - window / 2.0
    out = (vals - lo) / max(window, 1e-6)
    lut = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return lut, -int(info.min)


class XAViewer(AbstractViewer):
    handles_modality = "XA"

    #: emitted by the series-navigation buttons ("first"/"prev"/"next"/"last")
    series_nav = pyqtSignal(str)

    #: emitted when the user clicks "DICOM Tags…" (shell opens the picker)
    tags_requested = pyqtSignal()

    #: emitted on every completed measurement (shell logs it per study)
    measurement_added = pyqtSignal(object)

    #: emitted when the user clicks "Measure History"
    history_requested = pyqtSignal()

    #: emitted whenever the displayed frame index changes (slider scrub or
    #: cine playback). MultiSync uses this to mirror the pane's frame into
    #: the matching slot. _suspend_frame_signal guards against echo when
    #: MultiSync drove the change itself.
    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._planes: list[XAPlane] = []
        self._frame = 0
        self._loaded_uid = ""     # SeriesInstanceUID of the loaded series
        #: per-series remembered frame index, so navigating between
        #: series and back resumes at the LAST frame the user saw
        #: ("本当に最後に見ていた画像").
        self._frame_by_series: dict[str, int] = {}
        self._active = 0          # plane shown in single layout
        self._want_dual = False   # set by the shell (XA-only + biplane)
        self._window = 1.0
        self._level = 0.5
        self._is_color = False
        self._wl_lut = None          # (uint8 lut, offset) or None
        self._wl_off = 0
        self._prefetch: _Prefetcher | None = None
        self._header = None                 # metadata of loaded series
        self._tag_keywords: list[str] = []  # overlay selection (shell-owned)
        self._anon = False                  # anonymized display toggle
        # Re-entrancy guard: True while MultiSync (or any other external
        # driver) is pushing a frame in, so the resulting _frame change
        # does NOT echo back via frame_changed.
        self._suspend_frame_signal = False

        self.canvas = ImageCanvas()    # primary / Front
        self.canvas2 = ImageCanvas()   # Lateral, only in side-by-side
        self.canvas2.hide()

        self.title_label = QLabel("—")
        self.title_label.setStyleSheet("color:#ccc; padding:2px 6px;")
        self.readout = QLabel("")
        self.readout.setStyleSheet("color:#27e0c0; padding:2px 6px;")

        # cine timer
        self._timer = QTimer(self)
        # Windows' default (coarse) timer coalesces to ~15 ms, so a 20-30
        # fps cine plays unevenly and slower than its rate; precise keeps it.
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._next_frame)
        # Cheap poll that holds Play in a "Buffering..." state until the
        # prefetch has a playback lead, so the GIL-heavy decoding happens
        # while nothing is animating instead of stuttering live playback.
        self._buffer_timer = QTimer(self)
        self._buffer_timer.setInterval(40)
        self._buffer_timer.timeout.connect(self._buffer_poll)
        self._fps = DEFAULT_CINE_FPS
        # Cine speed multiplier toggled by D (1.0 = 1×, 2.0 = 2×).
        self._play_speed: float = 1.0
        # ECG waveform strip visible? (W key — reader not yet implemented.)
        self._ecg_visible: bool = False

        img_row = QHBoxLayout()
        img_row.setContentsMargins(0, 0, 0, 0)
        img_row.addWidget(self.canvas, 1)
        img_row.addWidget(self.canvas2, 1)
        # Wrap the cross-section row in a QWidget so subclasses (IVUS)
        # can swap it into a QSplitter without touching XAViewer.
        self._canvas_area = QWidget()
        self._canvas_area.setLayout(img_row)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(self._build_series_nav())
        # Sub-bar with the 4 tool buttons + Clear, placed directly under
        # the series-nav row so it appears one row below the Measure
        # toggle (which now sits next to the Last button — CT-style).
        self._measure_bar = self._build_measure_bar()
        self._measure_bar.setVisible(False)
        layout.addWidget(self._measure_bar)
        layout.addWidget(self.title_label)
        layout.addWidget(self._canvas_area, 1)
        layout.addWidget(self._build_plane_bar())
        layout.addLayout(self._build_transport())
        layout.addLayout(self._build_image_controls())
        layout.addWidget(self.readout)

        self.canvas.measurement_done.connect(self._on_measurement)
        self.canvas2.measurement_done.connect(self._on_measurement)

    def _on_measurement(self, m):
        self.readout.setText(f"{m.kind}: {m.label()}")
        self.measurement_added.emit(m)

    # ----------------------------------------------------------- UI builders
    def _build_series_nav(self):
        row = QHBoxLayout()
        row.setContentsMargins(2, 0, 2, 0)
        row.addWidget(QLabel("Series:"))
        for label, where, tip in (
            ("⏮ First", "first", "First series (Home)"),
            ("◀ Prev (A)", "prev", "Previous series — shortcut: A"),
            ("Next (F) ▶", "next", "Next series — shortcut: F"),
            ("Last ⏭", "last", "Last series (End)"),
        ):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c, w=where: self.series_nav.emit(w))
            row.addWidget(b)
        # Measure toggle right after the Last button (CT-style placement);
        # the sub-bar (4 tool buttons + Clear) lives directly below this
        # row and only appears when Measure is checked.
        self._meas_btn = QPushButton("📏 Measure")
        self._meas_btn.setCheckable(True)
        self._meas_btn.setMinimumWidth(110)
        self._meas_btn.setStyleSheet(
            "QPushButton { font-weight: bold; }"
            "QPushButton:checked { background:#1f77b4; color:black; }"
        )
        self._meas_btn.setToolTip(
            "Toggle the measure bar (Line / Polyline / Ellipse / Polygon)"
        )
        self._meas_btn.clicked.connect(self._toggle_measure)
        row.addWidget(self._meas_btn)

        # Magnifier (click-to-zoom) buttons, right of Measure. 🔍+ / 🔍−
        # are sticky modes: after pressing one, each click on the image
        # zooms about that point (×1.1 / ×0.9). 🔍1 resets to the original
        # fit. Mutually exclusive with the measure tools.
        self._zoom_in_btn = QPushButton("🔍+")
        self._zoom_in_btn.setCheckable(True)
        self._zoom_in_btn.setToolTip(
            "Click-to-zoom IN — then click a point on the image to magnify "
            "×1.1 centred on it (click again to zoom further)"
        )
        self._zoom_in_btn.clicked.connect(lambda: self._set_zoom_click("in"))
        row.addWidget(self._zoom_in_btn)

        self._zoom_reset_btn = QPushButton("🔍1")
        self._zoom_reset_btn.setToolTip("Reset zoom to the original fit")
        self._zoom_reset_btn.clicked.connect(self._reset_zoom)
        row.addWidget(self._zoom_reset_btn)

        self._zoom_out_btn = QPushButton("🔍−")
        self._zoom_out_btn.setCheckable(True)
        self._zoom_out_btn.setToolTip(
            "Click-to-zoom OUT — then click a point on the image to shrink "
            "×0.9 centred on it"
        )
        self._zoom_out_btn.clicked.connect(lambda: self._set_zoom_click("out"))
        row.addWidget(self._zoom_out_btn)
        # Exposed so subclasses (IVUS) can insert their own toggles into
        # this toolbar via ``_insert_series_nav_widget`` — they land just
        # left of the flush-right Measure-History / DICOM-Tags group below.
        self._series_nav_row = row
        row.addStretch(1)
        # Measure History / DICOM Tags sit flush-right in the top toolbar
        # so they land at the image's top-right corner — matching the CT
        # viewer, where the same two actions live in the top toolbar.
        hist_btn = QPushButton("Measure History")
        hist_btn.setToolTip("Show this study's measurement history")
        hist_btn.clicked.connect(self.history_requested.emit)
        row.addWidget(hist_btn)
        self._series_nav_right_anchor = hist_btn
        tags_btn = QPushButton("DICOM Tags…")
        tags_btn.setToolTip("Choose DICOM tags to overlay on the image")
        tags_btn.clicked.connect(self.tags_requested.emit)
        row.addWidget(tags_btn)
        return row

    def _insert_series_nav_widget(self, widget) -> None:
        """Insert *widget* into the top series-nav toolbar, just left of
        the spacer that pushes the flush-right Measure-History / DICOM-Tags
        group to the corner. Subclasses (IVUS) use this so their toggles
        stay on the left while those two actions remain top-right."""
        row = self._series_nav_row
        anchor = row.indexOf(self._series_nav_right_anchor)
        row.insertWidget(max(0, anchor - 1), widget)

    def _build_plane_bar(self) -> QWidget:
        self.plane_bar = QWidget()
        row = QHBoxLayout(self.plane_bar)
        row.setContentsMargins(2, 0, 2, 0)
        row.addWidget(QLabel("Plane:"))
        self._plane_group = QButtonGroup(self)
        self._plane_group.setExclusive(True)
        self._plane_group.idClicked.connect(self._plane_chosen)
        self._plane_row = row
        row.addStretch(1)
        self.plane_bar.hide()
        return self.plane_bar

    def _build_transport(self):
        row = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._toggle_play)
        # Play is the primary transport control, so it stays a touch larger
        # than the default — ~1.3× the default font (scaling the font grows
        # the whole button, click area included, and adapts to the
        # ▶ Play / ⏸ Pause label swap).
        _pf = self.play_btn.font()
        _ps = _pf.pointSizeF()
        if _ps > 0:
            _pf.setPointSizeF(_ps * 1.3)
            self.play_btn.setFont(_pf)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self._seek)
        # Enlarge just the draggable handle (~1.2×) for an easier grab,
        # styled like the W/L sliders' native thumb — a white disc with a
        # blue inner dot (radial gradient) and a thin ring. We reserve
        # enough widget height so the taller handle is never clipped top /
        # bottom, while the groove keeps a normal height and the widget
        # spans the same width, so the click / seek reaction range along
        # the bar is unchanged.
        self.frame_slider.setMinimumHeight(24)
        self.frame_slider.setStyleSheet(
            "QSlider::groove:horizontal{height:6px;border-radius:3px;"
            "background:#c4c4c4;}"
            "QSlider::handle:horizontal{width:18px;height:18px;"
            "margin:-6px 0;border:1px solid #6a6a6a;border-radius:9px;"
            "background:qradialgradient(cx:0.5,cy:0.5,radius:0.5,"
            "fx:0.5,fy:0.5,stop:0 #1c6fd0,stop:0.32 #1c6fd0,"
            "stop:0.40 #ffffff,stop:1 #ffffff);}"
        )

        self.frame_lbl = QLabel("0/0")
        self.frame_lbl.setMinimumWidth(70)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 60.0)
        self.fps_spin.setValue(DEFAULT_CINE_FPS)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.valueChanged.connect(self._set_fps)

        row.addWidget(self.play_btn)
        row.addWidget(self.frame_slider, 1)
        row.addWidget(self.frame_lbl)
        row.addWidget(self.fps_spin)
        return row

    def _build_image_controls(self):
        row = QHBoxLayout()

        row.addWidget(QLabel("W"))
        self.win_slider = QSlider(Qt.Orientation.Horizontal)
        self.win_slider.setStyleSheet(_SLIDER_QSS)
        self.win_slider.valueChanged.connect(self._wl_changed)
        row.addWidget(self.win_slider, 1)

        row.addWidget(QLabel("L"))
        self.lvl_slider = QSlider(Qt.Orientation.Horizontal)
        self.lvl_slider.setStyleSheet(_SLIDER_QSS)
        self.lvl_slider.valueChanged.connect(self._wl_changed)
        row.addWidget(self.lvl_slider, 1)

        # W/L is rarely used for XA/IVUS — let the sliders take only ~half
        # the stretchable width. Measure History / DICOM Tags used to sit
        # here; they now live flush-right in the top series-nav toolbar
        # (matching CT), so the freed space just trails off to the right.
        row.addStretch(2)
        return row

    def _clear_measurements(self):
        for c in (self.canvas, self.canvas2):
            c.clear_measurements()

    # ----------------------------------------------- zoom (Z / Shift+Z)
    def zoom_in(self) -> None:
        for c in (self.canvas, self.canvas2):
            c.zoom_in()

    def zoom_out(self) -> None:
        for c in (self.canvas, self.canvas2):
            c.zoom_out()

    # --------------------------------------------------- biplane / planes
    @property
    def _is_biplane(self) -> bool:
        return len(self._planes) >= 2

    @property
    def _dual(self) -> bool:
        """Effective side-by-side layout (only possible when biplane)."""
        return self._want_dual and self._is_biplane

    def set_biplane_layout(self, side_by_side: bool) -> None:
        """Called by the shell. side_by_side is honoured only for biplane
        series; single-plane cine always stays single."""
        self._want_dual = side_by_side
        if self._planes:
            self._relayout()

    # -- Bi / Lt / Rt --------------------------------------------------
    @property
    def supports_side(self) -> bool:
        """True iff Bi/Lt/Rt has anything to switch (biplane only)."""
        return self._is_biplane

    def set_side(self, side: str, allow_dual: bool = True) -> None:
        """Bi/Lt/Rt switch driven by the layout-bar buttons.

        * ``Bi`` — side-by-side when ``allow_dual`` is True (which the
          shell sets to ``layout_key == "1x1"``); otherwise stay on
          whichever single plane was last active.
        * ``Lt`` — force single-pane mode showing plane 0.
        * ``Rt`` — force single-pane mode showing plane 1.

        Single-plane series ignore this (nothing to switch)."""
        if not self._is_biplane:
            return
        if side == "Bi":
            self._want_dual = bool(allow_dual)
        elif side == "Lt":
            self._want_dual = False
            self._active = 0
        elif side == "Rt":
            self._want_dual = False
            self._active = 1
        self._rebuild_plane_buttons()
        self._relayout()

    def _rebuild_plane_buttons(self) -> None:
        for b in list(self._plane_group.buttons()):
            self._plane_group.removeButton(b)
            b.setParent(None)
            b.deleteLater()
        # rebuild in front of the trailing stretch
        for i, plane in enumerate(self._planes):
            btn = QPushButton(plane.name)
            btn.setCheckable(True)
            btn.setChecked(i == self._active)
            self._plane_group.addButton(btn, i)
            self._plane_row.insertWidget(self._plane_row.count() - 1, btn)

    def _plane_chosen(self, idx: int) -> None:
        self._active = idx
        self._render()

    def _relayout(self) -> None:
        """Apply the current layout decision to the widgets."""
        if self._dual:
            self.canvas2.show()
            self.plane_bar.hide()
        else:
            self.canvas2.hide()
            self.plane_bar.setVisible(self._is_biplane)
        self._render()

    # ------------------------------------------------- AbstractViewer impl.
    def load_series(self, loaded: LoadedSeries, title: str) -> None:
        # Same series as the one already loaded? Preserve everything —
        # frame index, W/L, measurements, zoom — so returning to this
        # series from a different modality resumes exactly where the
        # user left off instead of restarting from frame 0.
        #
        # Use the SHELL's series UID (carries the synthesized "<orig>#N"
        # for packed-XA splits), NOT the DICOM file's SeriesInstanceUID
        # which is identical across every split row and would falsely
        # short-circuit reloads to "same series, skip".
        new_uid = (
            loaded.series_uid
            or (str(getattr(loaded.header, "SeriesInstanceUID", ""))
                if loaded.header is not None else "")
        )
        if (self._planes and new_uid
                and getattr(self, "_loaded_uid", "") == new_uid):
            return
        # Switching to a DIFFERENT series within the same viewer: save
        # the frame we were on for the outgoing series so a later
        # return to it picks up at that frame, not frame 0.
        if self._loaded_uid and self._planes:
            self._frame_by_series[self._loaded_uid] = self._frame
        self._loaded_uid = new_uid
        self._stop_prefetch()
        self._buffer_timer.stop()
        self._planes = list(loaded.xa_planes or [])
        self._header = loaded.header
        # Remembered frame for the incoming series (clamped below once
        # we know its length); fresh series default to 0.
        self._frame = self._frame_by_series.get(new_uid, 0)
        self._active = 0
        self._window = loaded.window or 1.0
        self._level = loaded.level or 0.0
        self._fps = loaded.cine_fps or DEFAULT_CINE_FPS
        self._is_color = bool(loaded.is_color)

        # Window/level is meaningless for RGB — disable the sliders for
        # color series; otherwise base their span on the decoded frame 0.
        for s in (self.win_slider, self.lvl_slider):
            s.blockSignals(True)
        if self._is_color:
            self.win_slider.setEnabled(False)
            self.lvl_slider.setEnabled(False)
        else:
            self.win_slider.setEnabled(True)
            self.lvl_slider.setEnabled(True)
            stacked = np.concatenate(
                [p.volume[0].ravel() for p in self._planes]
            )
            vmin = float(stacked.min())
            vmax = float(stacked.max())
            span = max(vmax - vmin, 1.0)
            self.win_slider.setRange(1, int(span * 2))
            self.win_slider.setValue(int(self._window))
            self.lvl_slider.setRange(int(vmin - span), int(vmax + span))
            self.lvl_slider.setValue(int(self._level))
        for s in (self.win_slider, self.lvl_slider):
            s.blockSignals(False)
        self._refresh_wl_lut()

        n = max(p.volume.shape[0] for p in self._planes)
        # Clamp the remembered frame index to the new series' length.
        self._frame = max(0, min(self._frame, n - 1))
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(n - 1)
        self.frame_slider.setValue(self._frame)
        self.frame_slider.blockSignals(False)
        self.fps_spin.setValue(self._fps)

        for c in (self.canvas, self.canvas2):
            c.set_spacing(loaded.spacing_mm)
            c.clear_measurements()

        kind = (
            f"biplane: {' + '.join(p.name for p in self._planes)}"
            if self._is_biplane
            else "single plane"
        )
        cal = "calibrated" if loaded.spacing_mm else "uncalibrated (px)"
        tone = "color" if self._is_color else "grayscale"
        self.title_label.setText(
            f"{title}   |   {n} frame(s)   |   {kind}   "
            f"|   {tone}   |   {cal}"
        )

        self._rebuild_plane_buttons()
        self._relayout()
        self._refresh_overlay()

        # Warm the rest of the cine in the background for smooth playback;
        # any frame reached before that is decoded on demand by .frame().
        self._prefetch = _Prefetcher(
            self._planes, lambda: self._timer.isActive(), self
        )
        self._prefetch.start()

    def clear(self) -> None:
        self._timer.stop()
        self._buffer_timer.stop()
        self.play_btn.setChecked(False)
        self._stop_prefetch()
        self._planes = []
        self._header = None
        self._loaded_uid = ""
        self._frame_by_series.clear()
        self.canvas2.hide()
        self.plane_bar.hide()
        self.title_label.setText("—")
        self.readout.setText("")
        for c in (self.canvas, self.canvas2):
            c.set_overlay([])

    # ----------------------------------------------------- DICOM-tag overlay
    def current_header(self):
        """Metadata of the loaded series (None if nothing loaded)."""
        return self._header

    def set_tag_keywords(self, keywords) -> None:
        """Set which DICOM tags overlay the image (shell-owned selection)."""
        self._tag_keywords = list(keywords or [])
        self._refresh_overlay()

    def set_anonymized(self, on: bool) -> None:
        self._anon = bool(on)
        self._refresh_overlay()

    def _refresh_overlay(self) -> None:
        # Biplane: each pane shows ITS OWN plane's DICOM tags (Frontal's
        # were previously shown on the Lateral pane too because the
        # overlay was computed from self._header alone). Each XAPlane
        # keeps its own ``_ds``, so per-canvas overlays are exact.
        for ci, c in enumerate((self.canvas, self.canvas2)):
            if ci < len(self._planes):
                ds = self._planes[ci]._ds
            else:
                ds = self._header
            c.set_overlay(
                overlay_lines(ds, self._tag_keywords, anonymized=self._anon)
            )

    # ----------------------------------------------------- play/stop API
    def play(self) -> None:
        self.play_btn.setChecked(True)

    def stop(self) -> None:
        self.play_btn.setChecked(False)

    # ----------------------------------------------- step / speed API (T/R/D/S)
    def step_frame(self, delta: int) -> None:
        """One-frame nudge in either direction. Stops playback so the
        user can scrub past the last frame without it wrapping back
        round (matches the spec: T = +1, R = -1)."""
        if not self._planes:
            return
        self.stop()
        n = max(p.volume.shape[0] for p in self._planes)
        if n < 1:
            return
        self._frame = max(0, min(self._frame + int(delta), n - 1))
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self._frame)
        self.frame_slider.blockSignals(False)
        self._render()
        if not self._suspend_frame_signal:
            self.frame_changed.emit(self._frame)

    def toggle_play_speed(self) -> None:
        """D: stopped → play 1×; 1× → 2×; 2× → 1×. (S = stop separately.)"""
        if not self._planes:
            return
        if not self.play_btn.isChecked():
            # Not playing: start at 1×.
            self._play_speed = 1.0
            self._apply_play_interval()
            self.play()
            return
        # Already playing: cycle 1 ↔ 2.
        self._play_speed = 2.0 if self._play_speed < 1.5 else 1.0
        self._apply_play_interval()
        # Live-update the timer if it is currently active.
        if self._timer.isActive():
            self._timer.start(int(1000.0 / max(self._effective_fps(), 1e-3)))

    def toggle_ecg(self) -> None:
        """W: ECG waveform strip toggle. The reader/widget are not yet
        wired in this build — flip the intent flag and tell the user."""
        self._ecg_visible = not getattr(self, "_ecg_visible", False)
        self.readout.setText(
            "ECG strip: ON (waveform display coming in a later build)"
            if self._ecg_visible
            else "ECG strip: OFF"
        )

    def _effective_fps(self) -> float:
        return float(self._fps) * float(getattr(self, "_play_speed", 1.0))

    def _apply_play_interval(self) -> None:
        """No-op when the timer is stopped; live-changes the cine speed
        when playing. Kept so toggle_play_speed reads as intent."""
        return

    # ------------------------------------------------------------- internals
    def _stop_prefetch(self) -> None:
        """Signal the prefetch thread to stop and detach. We don't block
        the UI thread waiting for it: prefetch_planes checks
        ``should_stop`` between every frame (~10-30 ms), so the worker
        exits on its own. The QThread is parented to the viewer so Qt
        owns its lifecycle and reaps it cleanly once it returns. A short
        ``wait`` is kept as a best-effort barrier — long enough to cover
        a single in-flight decode, short enough that rapid Next/Prev
        feels instant even when the previous series is mid-prefetch."""
        if self._prefetch is not None:
            self._prefetch.stop()
            # 80 ms is comfortably above one decoded frame on the slow
            # JPEG paths and well below the user's perceived-instant
            # ~150 ms threshold for click→action feedback.
            self._prefetch.wait(80)
            self._prefetch = None

    def _refresh_wl_lut(self) -> None:
        """(Re)build the W/L lookup table for the current grayscale
        cine. Call when the series or window/level changes."""
        self._wl_lut = None
        if self._is_color or not self._planes:
            return
        built = _build_wl_lut(
            self._planes[0].volume.dtype, self._window, self._level
        )
        if built is not None:
            self._wl_lut, self._wl_off = built

    def _frame_of(self, plane: XAPlane) -> np.ndarray:
        f = plane.frame(self._frame)
        if getattr(plane, "is_color", False):
            return f  # already display-ready uint8 RGB
        if (
            self._wl_lut is not None
            and np.issubdtype(f.dtype, np.integer)
        ):
            # Fast path: one array gather instead of float math/clip.
            return self._wl_lut[f + self._wl_off] if self._wl_off \
                else self._wl_lut[f]
        return apply_window(f, self._window, self._level)

    @staticmethod
    def _plane_angles(plane) -> tuple[float, float] | None:
        """(PositionerPrimary, Secondary) of a plane's dataset, or None when
        either tag is missing — used to stamp the angle onto measurements so
        Coaxial-Eval knows the exact view each line was drawn on."""
        ds = getattr(plane, "_ds", None)
        if ds is None:
            return None

        def _f(name):
            try:
                v = float(getattr(ds, name, float("nan")))
                return None if math.isnan(v) else v
            except (TypeError, ValueError):
                return None

        b = _f("PositionerPrimaryAngle")
        a = _f("PositionerSecondaryAngle")
        return (b, a) if (b is not None and a is not None) else None

    def _render(self):
        if not self._planes:
            return
        if self._dual:
            self.canvas.set_frame(self._frame_of(self._planes[0]))
            self.canvas2.set_frame(self._frame_of(self._planes[1]))
            self.canvas.view_angles = self._plane_angles(self._planes[0])
            self.canvas2.view_angles = self._plane_angles(self._planes[1])
        else:
            self._active = min(self._active, len(self._planes) - 1)
            self.canvas.set_frame(self._frame_of(self._planes[self._active]))
            self.canvas.view_angles = self._plane_angles(
                self._planes[self._active]
            )
        n = max(p.volume.shape[0] for p in self._planes)
        self.frame_lbl.setText(f"{self._frame + 1}/{n}")

    def _next_frame(self):
        if not self._planes:
            return
        n = max(p.volume.shape[0] for p in self._planes)
        nxt = (self._frame + 1) % n
        if self._dual:
            shown = self._planes
        else:
            self._active = min(self._active, len(self._planes) - 1)
            shown = [self._planes[self._active]]
        # Don't decode on the UI thread mid-cine: if the prefetch hasn't
        # warmed the next frame yet, hold on the current one (the timer
        # keeps firing and re-checks). The paced prefetch warms far faster
        # than any cine fps, so this only briefly holds at the very start
        # and never stutters; once warm it plays straight from cache.
        if nxt != 0 and not all(p.is_ready(nxt) for p in shown):
            return
        self._frame = nxt
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self._frame)
        self.frame_slider.blockSignals(False)
        self._render()
        if not self._suspend_frame_signal:
            self.frame_changed.emit(self._frame)

    def _shown_planes(self) -> list[XAPlane]:
        if self._dual:
            return self._planes
        self._active = min(self._active, len(self._planes) - 1)
        return [self._planes[self._active]]

    def _buffered_enough(self) -> bool:
        """True once the prefetch has decoded ~1 s of cine ahead of the
        play head on every shown plane (so playback runs from cache and
        the prefetch is finishing the tail far ahead, never contending
        with the timer). prefetch is strictly in index order, so the
        furthest needed frame being ready implies all earlier ones are."""
        if not self._planes:
            return False
        n = max(p.volume.shape[0] for p in self._planes)
        need = min(n - 1, max(2, int(round(self._fps))))
        last = min(self._frame + need, n - 1)
        return all(p.is_ready(last) for p in self._shown_planes())

    def _toggle_play(self, on: bool):
        n = max((p.volume.shape[0] for p in self._planes), default=0)
        if n < 2:
            self.play_btn.setChecked(False)
            return
        self.play_btn.setText("⏸ Pause" if on else "▶ Play")
        if on:
            if self._buffered_enough():
                self._start_cine()
            else:
                self.frame_lbl.setText("⏳ Buffering…")
                self._buffer_timer.start()
        else:
            self._buffer_timer.stop()
            self._timer.stop()
            if self._planes:
                self._render()                  # restore the "i/n" label

    def _buffer_poll(self):
        # Wait for a lead, but never wait past the prefetch finishing.
        if not self.play_btn.isChecked():       # paused while buffering
            self._buffer_timer.stop()
            return
        done = self._prefetch is None or not self._prefetch.isRunning()
        if self._buffered_enough() or done:
            self._buffer_timer.stop()
            self._start_cine()

    def _start_cine(self):
        self._render()                          # also restores the label
        self._timer.start(int(1000.0 / max(self._effective_fps(), 1e-3)))

    def _seek(self, value: int):
        self._frame = value
        self._render()
        if not self._suspend_frame_signal:
            self.frame_changed.emit(self._frame)

    def goto_frame(self, idx: int) -> None:
        """External entry point (used by MultiSync) to move to a frame
        without re-emitting frame_changed — prevents an A→B→A echo loop
        when both views drive each other."""
        if not self._planes:
            return
        n = max(p.volume.shape[0] for p in self._planes)
        idx = max(0, min(int(idx), n - 1))
        if idx == self._frame:
            return
        self._suspend_frame_signal = True
        try:
            self._frame = idx
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self._frame)
            self.frame_slider.blockSignals(False)
            self._render()
        finally:
            self._suspend_frame_signal = False

    def _set_fps(self, fps: float):
        self._fps = fps
        if self._timer.isActive():
            self._timer.start(int(1000.0 / max(self._effective_fps(), 1e-3)))

    def _wl_changed(self):
        self._window = float(self.win_slider.value())
        self._level = float(self.lvl_slider.value())
        self._refresh_wl_lut()
        self._render()

    # -------------------------------------------------- measure toolbar
    def _build_measure_bar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 6, 2)
        row.addWidget(QLabel("Measure:"))
        self._meas_btns: dict[str, QPushButton] = {}
        for label, key in (
            ("Line", "line"), ("Polyline", "polyline"),
            ("Ellipse", "ellipse"), ("Polygon", "polygon"),
            ("Angle", "angle"),
        ):
            b = QPushButton(label)
            b.setCheckable(True)
            b.clicked.connect(lambda _c, k=key: self._set_measure_type(k))
            self._meas_btns[key] = b
            row.addWidget(b)
        clr = QPushButton("Clear")
        clr.clicked.connect(self._clear_measurements)
        row.addWidget(clr)
        row.addWidget(QLabel(
            "  Left-click = add point /"
            " right-click finishes Polyline / Polygon"
        ))
        row.addStretch(1)
        return bar

    def _toggle_measure(self):
        on = self._meas_btn.isChecked()
        self._measure_bar.setVisible(on)
        if on:
            self._clear_zoom_click()
        if not on:
            for c in (self.canvas, self.canvas2):
                c.set_measure_type("")
            for b in self._meas_btns.values():
                b.setChecked(False)
                b.setStyleSheet("")

    def _set_measure_type(self, key: str):
        # Choosing a drawing tool cancels any active click-to-zoom mode.
        self._clear_zoom_click()
        for k, b in self._meas_btns.items():
            b.setChecked(k == key)
            b.setStyleSheet(
                "background:#1f77b4;color:black;" if k == key else ""
            )
        for c in (self.canvas, self.canvas2):
            c.set_measure_type(key)

    # -------------------------------------------------- click-to-zoom
    def _set_zoom_click(self, mode: str):
        """Toggle the 🔍+ / 🔍− click-to-zoom mode. Re-clicking the active
        button turns it off; switching modes turns the other off; enabling
        either cancels the measure tools."""
        btn = self._zoom_in_btn if mode == "in" else self._zoom_out_btn
        other = self._zoom_out_btn if mode == "in" else self._zoom_in_btn
        new_mode = mode if btn.isChecked() else ""
        other.setChecked(False)
        if new_mode and self._meas_btn.isChecked():
            self._meas_btn.setChecked(False)
            self._toggle_measure()
        self._style_zoom_btns()
        for c in (self.canvas, self.canvas2):
            c.set_zoom_click_mode(new_mode)

    def _clear_zoom_click(self):
        self._zoom_in_btn.setChecked(False)
        self._zoom_out_btn.setChecked(False)
        self._style_zoom_btns()
        for c in (self.canvas, self.canvas2):
            c.set_zoom_click_mode("")

    def _style_zoom_btns(self):
        for b in (self._zoom_in_btn, self._zoom_out_btn):
            b.setStyleSheet(
                "background:#1f77b4;color:black;" if b.isChecked() else ""
            )

    def _reset_zoom(self):
        for c in (self.canvas, self.canvas2):
            c.reset_zoom()
