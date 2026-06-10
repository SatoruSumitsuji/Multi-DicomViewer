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
from PyQt6.QtCore import QPoint, QPointF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygon, QPolygonF
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.config import DEFAULT_CINE_FPS
from multi_dicomviewer.core.ecg import ECGTrace, read_ecg
from multi_dicomviewer.core.dicom_io import (
    LoadedSeries,
    XAPlane,
    prefetch_planes,
)
from multi_dicomviewer.core.dicom_tags import overlay_lines
from multi_dicomviewer.core.image_export import export_image_as, safe_basename
from multi_dicomviewer.ui.tag_font import (
    TAG_FONT_PT_DEFAULT, build_tag_font_control,
)
from multi_dicomviewer.ui.viewer_base import AbstractViewer
from multi_dicomviewer.viewers.image_canvas import ImageCanvas

#: Inner-blue-dot blue used on the seek-bar handle (the "中青○" mark). The
#: Play range start/end triangle markers reuse this exact hue so they read
#: as the same control family.
_SEEK_DOT_BLUE = "#1c6fd0"

#: Slider look matching the Windows build: light-grey groove, blue filled
#: (sub-page) track, and a handle that is a white circle with a blue dot in
#: the centre. Applied to the W/L sliders. The inner blue dot here is the
#: SMALL one (radius ratio 0.32) — swapped with the seek bar's, which now
#: carries the LARGE inner dot. Handle pixel size is unchanged (16 px).
_SLIDER_QSS = (
    "QSlider::groove:horizontal {"
    " height:5px; background:#c9c9c9; border-radius:2px; }"
    "QSlider::sub-page:horizontal {"
    " background:#2f7fd1; border-radius:2px; }"
    "QSlider::handle:horizontal {"
    " width:16px; height:16px; margin:-6px 0; border-radius:8px;"
    " border:1px solid #9a9a9a;"
    " background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,"
    " stop:0 #2f7fd1, stop:0.32 #2f7fd1, stop:0.40 #ffffff, stop:1 #ffffff); }"
)


class _RangeMarks(QWidget):
    """Interactive strip drawn directly above the cine seek bar carrying two
    draggable down-triangle (▽) markers: the Play range START and END.

    Both triangles are painted in the seek handle's inner-blue
    (:data:`_SEEK_DOT_BLUE`) so they read as part of the same control. The
    strip shares the slider's x / width (they are stacked in one column),
    so the slider's groove geometry maps directly onto this widget's
    coordinates — the same trick MultiSync's mark strip uses.

    Dragging a marker bounds where Play loops and how far the handle can be
    scrubbed; the MP4 export honours the same [start, end] range. The DICOM
    export is unaffected (always full series, by design)."""

    range_changed = pyqtSignal(int, int)   # (start, end), 0-based frame idx

    _TRI_HW = 6        # triangle half-width, px

    def __init__(self, slider: QSlider, parent=None):
        super().__init__(parent)
        self._slider = slider
        self._start = 0
        self._end = 0
        self._drag: str | None = None      # "start" | "end" | None
        self.setFixedHeight(11)
        self.setMouseTracking(True)

    def set_bounds(self, start: int, end: int) -> None:
        self._start, self._end = int(start), int(end)
        self.update()

    # -- value <-> pixel mapping (mirrors the slider's own geometry) ----
    def _span_base(self) -> tuple[int, float]:
        sl = self._slider
        opt = QStyleOptionSlider()
        sl.initStyleOption(opt)
        style = sl.style()
        groove = style.subControlRect(
            QStyle.ComplexControl.CC_Slider, opt,
            QStyle.SubControl.SC_SliderGroove, sl,
        )
        handle = style.subControlRect(
            QStyle.ComplexControl.CC_Slider, opt,
            QStyle.SubControl.SC_SliderHandle, sl,
        )
        span = groove.width() - handle.width()
        base = groove.x() + handle.width() / 2.0
        return span, base

    def _value_x(self, value: int) -> float:
        sl = self._slider
        span, base = self._span_base()
        pos = QStyle.sliderPositionFromValue(
            sl.minimum(), sl.maximum(), int(value), span
        )
        return base + pos

    def _x_value(self, x: float) -> int:
        sl = self._slider
        span, base = self._span_base()
        return QStyle.sliderValueFromPosition(
            sl.minimum(), sl.maximum(), int(round(x - base)), span
        )

    def paintEvent(self, _e) -> None:
        sl = self._slider
        if sl.maximum() <= sl.minimum():
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(_SEEK_DOT_BLUE))
        h = self.height()
        w = self._TRI_HW
        for val in (self._start, self._end):
            cx = self._value_x(val)
            p.drawPolygon(QPolygon([
                QPoint(int(round(cx - w)), 0),
                QPoint(int(round(cx + w)), 0),
                QPoint(int(round(cx)), h - 1),
            ]))

    def resizeEvent(self, _e) -> None:
        self.update()

    def mousePressEvent(self, e) -> None:
        sl = self._slider
        if sl.maximum() <= sl.minimum():
            return
        x = e.position().x()
        ds = abs(x - self._value_x(self._start))
        de = abs(x - self._value_x(self._end))
        # Grab the nearer triangle. When both ends sit on the same frame
        # (a fresh full range collapses start==end only for 1-frame cines,
        # which hide the strip) bias to whichever side the press is toward.
        if ds <= de:
            self._drag = "start"
        else:
            self._drag = "end"
        self._apply_drag(x)

    def mouseMoveEvent(self, e) -> None:
        if self._drag is not None:
            self._apply_drag(e.position().x())

    def mouseReleaseEvent(self, _e) -> None:
        self._drag = None

    def _apply_drag(self, x: float) -> None:
        sl = self._slider
        v = max(sl.minimum(), min(self._x_value(x), sl.maximum()))
        if self._drag == "start":
            self._start = min(v, self._end)     # start never passes end
        else:
            self._end = max(v, self._start)     # end never passes start
        self.update()
        self.range_changed.emit(self._start, self._end)


class _EcgStrip(QWidget):
    """A physiological ECG strip drawn below the angio image.

    Shows the whole embedded ECG trace (read from the DICOM Waveform
    Sequence or legacy Curve Data) as a green polyline, with a yellow
    cursor that tracks the cine frame. The cursor position is given as a
    0–1 fraction by the viewer, which knows the frame↔time mapping; the
    strip itself just draws. Hidden until the user presses W on a series
    that actually carries an ECG."""

    _PAD = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(78)
        self._label = ""
        self._cursor = 0.0           # 0..1 across the trace
        self._xs: np.ndarray | None = None   # 0..1 sample x positions
        self._ys: np.ndarray | None = None   # 0..1 amplitude (1 = peak)

    def set_trace(self, trace: ECGTrace | None) -> None:
        """Cache a down-sampled, amplitude-normalised polyline for *trace*
        (None blanks the strip)."""
        self._xs = self._ys = None
        self._label = ""
        if trace is not None and trace.n >= 2:
            s = np.asarray(trace.samples, dtype=np.float32)
            lo = float(np.nanmin(s))
            hi = float(np.nanmax(s))
            rng = (hi - lo) if hi > lo else 1.0
            yn = (s - lo) / rng                       # 0..1, 1 = peak
            n = yn.shape[0]
            cap = 3000                                # plenty for any width
            if n > cap:
                idx = np.linspace(0, n - 1, cap).astype(np.int64)
                self._xs = (idx / (n - 1)).astype(np.float32)
                self._ys = yn[idx]
            else:
                self._xs = np.linspace(0.0, 1.0, n, dtype=np.float32)
                self._ys = yn
            self._label = trace.label or "ECG"
        self.update()

    def set_cursor_fraction(self, f: float) -> None:
        f = max(0.0, min(float(f), 1.0))
        if f != self._cursor:
            self._cursor = f
            self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0e1216"))
        w, h = self.width(), self.height()
        if self._xs is None or self._ys is None:
            p.setPen(QColor("#888"))
            p.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "ECG: no waveform in this series"
            )
            return
        pad = self._PAD
        top, bot = pad, h - pad
        span = max(1, bot - top)
        # Baseline (mid-line).
        p.setPen(QPen(QColor(70, 110, 85), 1, Qt.PenStyle.DashLine))
        p.drawLine(0, h // 2, w, h // 2)
        # Waveform polyline.
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor("#36e07a"), 1.3))
        xw = max(1, w - 1)
        poly = QPolygonF([
            QPointF(float(x) * xw, bot - float(y) * span)
            for x, y in zip(self._xs, self._ys)
        ])
        p.drawPolyline(poly)
        # Frame cursor.
        cx = int(round(self._cursor * xw))
        p.setPen(QPen(QColor(255, 210, 0, 230), 2))
        p.drawLine(cx, 0, cx, h)
        # Caption.
        p.setPen(QColor("#cfe8ff"))
        p.drawText(6, 15, f"ECG · {self._label}")


class _ClickLabel(QLabel):
    """A QLabel that emits ``clicked`` on a left mouse press. Used for the
    "frame/total" readout so a single click on it resets the Play range to
    the full clip."""

    clicked = pyqtSignal()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)


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

    #: emitted when the tag-text-size slider moves (shell broadcasts the new
    #: pt to every viewer so the overlay size stays uniform across modalities)
    overlay_font_changed = pyqtSignal(int)

    #: emitted on every completed measurement (shell logs it per study)
    measurement_added = pyqtSignal(object)

    #: emitted when the user clicks "Measure History"
    history_requested = pyqtSignal()

    #: emitted whenever the displayed frame index changes (slider scrub or
    #: cine playback). MultiSync uses this to mirror the pane's frame into
    #: the matching slot. _suspend_frame_signal guards against echo when
    #: MultiSync drove the change itself.
    frame_changed = pyqtSignal(int)

    #: emitted when the Play range markers move (or on series load), so the
    #: shell can remember per-series [start, end] for the MP4 export.
    #: (series_uid, start, end, total_frames) — all 0-based frame indices.
    play_range_changed = pyqtSignal(str, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._planes: list[XAPlane] = []
        self._frame = 0
        self._loaded_uid = ""     # SeriesInstanceUID of the loaded series
        #: per-series remembered frame index, so navigating between
        #: series and back resumes at the LAST frame the user saw
        #: ("本当に最後に見ていた画像").
        self._frame_by_series: dict[str, int] = {}
        #: per-series Play range [start, end] (0-based frame indices). Bounds
        #: cine looping + handle scrubbing and drives the MP4 export range.
        #: Defaults to the full series; remembered across series switches.
        self._range_by_series: dict[str, tuple[int, int]] = {}
        self._range_start = 0
        self._range_end = 0
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
        # ECG strip (W key): the trace read from this series' DICOM, the
        # per-frame time step used to drive the cursor, and visibility.
        self._ecg_visible: bool = False
        self._ecg_trace: ECGTrace | None = None
        self._ecg_available: bool = False
        self._ecg_frame_time_ms: float = 0.0
        self._ecg_strip = _EcgStrip()
        self._ecg_strip.setVisible(False)

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
        # ECG strip sits directly under the angio image (hidden until W
        # toggles it on, and only on series that carry an ECG waveform).
        layout.addWidget(self._ecg_strip)
        layout.addWidget(self._build_plane_bar())
        layout.addLayout(self._build_transport())
        layout.addLayout(self._build_image_controls())
        layout.addWidget(self.readout)

        self.canvas.measurement_done.connect(self._on_measurement)
        self.canvas2.measurement_done.connect(self._on_measurement)
        self.canvas.export_requested.connect(self._on_export_image)
        self.canvas2.export_requested.connect(self._on_export_image)

    def _on_measurement(self, m):
        self.readout.setText(f"{m.kind}: {m.label()}")
        self.measurement_added.emit(m)

    def _on_export_image(self, fmt_key):
        """Right-click export on a canvas (no active measure tool): save
        exactly what's on that canvas — image + measurement overlays +
        burned-in/tag text — in the chosen format."""
        canvas = self.sender()
        if not isinstance(canvas, ImageCanvas):
            canvas = self.canvas
        # grab() captures the painted widget verbatim (WYSIWYG), at the
        # display's devicePixelRatio so Retina exports stay full-res.
        export_image_as(self, canvas.grab(), fmt_key, self._export_basename())

    def _export_basename(self) -> str:
        """Suggested filename stem from the loaded series + current frame."""
        h = self._header
        parts: list[object] = []
        if h is not None:
            parts.append(getattr(h, "PatientID", "") or "")
            parts.append(getattr(h, "Modality", "") or "")
            parts.append(getattr(h, "AcquisitionDate", "")
                         or getattr(h, "StudyDate", "") or "")
        parts.append(f"frame{self._frame + 1}")
        return safe_basename(*parts)

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
        # DICOM Tags / Measure History sit flush-right in the top toolbar so
        # they land at the image's top-right corner — matching the CT viewer.
        # DICOM Tags is on the LEFT of the pair (always visible / more useful);
        # Measure History — less critical — sits to its right. The tag-text-size
        # slider is stacked ABOVE the Tags button (shared across modalities).
        tags_box, self._tag_font_slider, tags_btn = build_tag_font_control(
            TAG_FONT_PT_DEFAULT
        )
        tags_btn.clicked.connect(self.tags_requested.emit)
        self._tag_font_slider.valueChanged.connect(self.overlay_font_changed.emit)
        row.addWidget(tags_box)
        self._series_nav_right_anchor = tags_box   # left edge of the pair
        hist_btn = QPushButton("Measure History")
        hist_btn.setToolTip("Show this study's measurement history")
        hist_btn.clicked.connect(self.history_requested.emit)
        row.addWidget(hist_btn)
        return row

    def set_overlay_font_pt(self, pt: int) -> None:
        """Apply the shared DICOM-tag text size (pt) to both canvases and sync
        the slider. Called by the shell so every viewer stays in step."""
        pt = int(pt)
        sl = getattr(self, "_tag_font_slider", None)
        if sl is not None and sl.value() != pt:
            sl.blockSignals(True)
            sl.setValue(pt)
            sl.blockSignals(False)
        for c in (self.canvas, self.canvas2):
            c.set_overlay_font_pt(pt)

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
        # Seek-bar handle now carries the LARGE inner blue dot (radius ratio
        # 0.55) — swapped with the W/L sliders, which took the small one.
        # The 18 px handle pixel size is unchanged; only the inner-dot
        # proportion (the "mark") moved here.
        self.frame_slider.setStyleSheet(
            "QSlider::groove:horizontal{height:6px;border-radius:3px;"
            "background:#c4c4c4;}"
            "QSlider::handle:horizontal{width:18px;height:18px;"
            "margin:-6px 0;border:1px solid #6a6a6a;border-radius:9px;"
            "background:qradialgradient(cx:0.5,cy:0.5,radius:0.5,"
            "fx:0.5,fy:0.5,stop:0 #1c6fd0,stop:0.55 #1c6fd0,"
            "stop:0.60 #ffffff,stop:1 #ffffff);}"
        )

        # Draggable Play-range start/end triangles, painted on a thin strip
        # directly above the seek bar and sharing its x / width so they line
        # up with the slider positions.
        self._range_marks = _RangeMarks(self.frame_slider)
        self._range_marks.range_changed.connect(self._on_range_marks_changed)
        self._range_marks.setVisible(False)   # shown once a cine loads
        seek_col = QVBoxLayout()
        seek_col.setContentsMargins(0, 0, 0, 0)
        seek_col.setSpacing(0)
        seek_col.addWidget(self._range_marks)
        seek_col.addWidget(self.frame_slider)

        # Clicking the "frame/total" readout resets the Play range to the
        # full clip (one-click "select all"). Hand cursor + tooltip hint it.
        self.frame_lbl = _ClickLabel("0/0")
        self.frame_lbl.setMinimumWidth(70)
        self.frame_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self.frame_lbl.setToolTip(
            "クリックで再生範囲を全範囲に戻す "
            "(click to reset the Play range to the whole clip)"
        )
        self.frame_lbl.clicked.connect(self._reset_play_range)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 60.0)
        self.fps_spin.setValue(DEFAULT_CINE_FPS)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.valueChanged.connect(self._set_fps)

        row.addWidget(self.play_btn)
        row.addLayout(seek_col, 1)
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
            self._range_by_series[self._loaded_uid] = (
                self._range_start, self._range_end
            )
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
        # A single-image series (seek bar 1/1) can't play — grey out Play and
        # stop any running playback; multi-frame cines re-enable it. Modality-
        # agnostic: every cine (XA/IVUS/US/NM/…) loads through here.
        single = n <= 1
        if single and self.play_btn.isChecked():
            self.play_btn.setChecked(False)
        self.play_btn.setEnabled(not single)
        self.fps_spin.setValue(self._fps)

        # Restore (or default-to-full) this series' Play range and sync the
        # marker strip. Single-frame series have no range to set — hide the
        # strip. Tell the shell either way so its MP4-range map is current.
        lo, hi = self._range_by_series.get(new_uid, (0, n - 1))
        lo = max(0, min(int(lo), n - 1))
        hi = max(lo, min(int(hi), n - 1))
        self._range_start, self._range_end = lo, hi
        self._range_marks.set_bounds(lo, hi)
        self._range_marks.setVisible(not single)
        self.play_range_changed.emit(new_uid, lo, hi, n)

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
        # Read this series' ECG (if any) and prime the strip — hidden until W.
        self._read_ecg_for_series()

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
        self._range_by_series.clear()
        self._range_start = self._range_end = 0
        self._range_marks.setVisible(False)
        self._ecg_trace = None
        self._ecg_available = False
        self._ecg_visible = False
        self._ecg_strip.set_trace(None)
        self._ecg_strip.setVisible(False)
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
        # Nudge stays inside the Play range, matching the scrub clamp.
        self._frame = max(
            self._range_start,
            min(self._frame + int(delta), self._range_end),
        )
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
        """W: show / hide the ECG strip. Only series that actually carry a
        waveform can show it — on the rest the toggle just reports that no
        ECG is present and leaves the strip hidden."""
        if not self._ecg_available:
            self._ecg_visible = False
            self._ecg_strip.setVisible(False)
            self.readout.setText(
                "このシリーズに心電図データはありません "
                "(no ECG waveform in this series)."
            )
            return
        self._ecg_visible = not self._ecg_visible
        self._ecg_strip.setVisible(self._ecg_visible)
        if self._ecg_visible:
            self._ecg_strip.set_cursor_fraction(self._ecg_fraction(self._frame))
        lab = self._ecg_trace.label if self._ecg_trace else "ECG"
        self.readout.setText(
            f"ECG: ON · {lab}" if self._ecg_visible else "ECG: OFF"
        )

    def _frame_time_ms(self, ds) -> float:
        """Per-frame duration (ms) for mapping a frame onto the ECG time
        axis: the DICOM FrameTime when present, else 1000 / cine-fps."""
        if ds is not None:
            ft = getattr(ds, "FrameTime", None)
            try:
                if ft is not None:
                    v = float(ft)
                    if v > 0:
                        return v
            except (TypeError, ValueError):
                pass
        fps = float(self._fps or 0.0)
        return 1000.0 / fps if fps > 0 else 0.0

    def _ecg_fraction(self, frame: int) -> float:
        """Cursor position (0..1) along the ECG for cine *frame*.

        When the ECG sampling rate and the cine frame time are both known,
        map by absolute acquisition time (frame i → i·FrameTime → t/duration)
        so the cursor lands on the physiologically correct sample. Otherwise
        (legacy curves with no stated rate) fall back to a proportional map
        that stretches the trace across the whole cine."""
        n = max((p.volume.shape[0] for p in self._planes), default=1)
        if n <= 1:
            return 0.0
        t = self._ecg_trace
        if (t is not None and t.fs > 0 and self._ecg_frame_time_ms > 0
                and t.duration_s > 0):
            sec = frame * self._ecg_frame_time_ms / 1000.0
            return max(0.0, min(sec / t.duration_s, 1.0))
        return frame / (n - 1)

    def _read_ecg_for_series(self) -> None:
        """Read this series' ECG (Waveform Sequence or legacy Curve Data)
        from the metadata header and prime the strip. Resets visibility so
        a new series starts with the strip hidden until the user presses W."""
        self._ecg_trace = None
        self._ecg_available = False
        ds = self._header
        if ds is None and self._planes:
            ds = getattr(self._planes[0], "_ds", None)
        self._ecg_frame_time_ms = self._frame_time_ms(ds)
        trace = None
        try:
            trace = read_ecg(ds)
        except Exception:
            trace = None
        if trace is not None and trace.n >= 2:
            self._ecg_trace = trace
            self._ecg_available = True
            self._ecg_strip.set_trace(trace)
        else:
            self._ecg_strip.set_trace(None)
        self._ecg_visible = False
        self._ecg_strip.setVisible(False)

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
        # Keep the ECG cursor in step with the displayed frame.
        if self._ecg_visible and self._ecg_trace is not None:
            self._ecg_strip.set_cursor_fraction(self._ecg_fraction(self._frame))

    def _next_frame(self):
        if not self._planes:
            return
        # Loop within the Play range [start, end] instead of the whole
        # series, so playback stays inside the user-set window.
        lo, hi = self._range_start, self._range_end
        nxt = lo if self._frame >= hi else self._frame + 1
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
        if nxt != lo and not all(p.is_ready(nxt) for p in shown):
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
            # Snap into the Play range before looping so a handle parked
            # outside [start, end] doesn't crawl up to it one frame at a time.
            if self._frame < self._range_start or self._frame > self._range_end:
                self.frame_slider.setValue(
                    max(self._range_start,
                        min(self._frame, self._range_end))
                )
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
        # Keep the handle inside the Play range — dragging past either
        # marker sticks at it ("シークバーの移動範囲も開始点と終了点の範囲に").
        clamped = max(self._range_start, min(int(value), self._range_end))
        if clamped != value:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(clamped)
            self.frame_slider.blockSignals(False)
            value = clamped
        self._frame = value
        self._render()
        if not self._suspend_frame_signal:
            self.frame_changed.emit(self._frame)

    def _reset_play_range(self) -> None:
        """One-click reset of the Play range back to the whole clip — fired
        by clicking the 'frame/total' readout. No-op for single-frame
        series (which have no range)."""
        n = max((p.volume.shape[0] for p in self._planes), default=0)
        if n <= 1:
            return
        lo, hi = 0, n - 1
        if (self._range_start, self._range_end) == (lo, hi):
            return                              # already full — nothing to do
        self._range_start, self._range_end = lo, hi
        if self._loaded_uid:
            self._range_by_series[self._loaded_uid] = (lo, hi)
        self._range_marks.set_bounds(lo, hi)
        self.play_range_changed.emit(self._loaded_uid, lo, hi, n)
        self.readout.setText(f"Play range reset: all {n} frames")

    def _on_range_marks_changed(self, start: int, end: int) -> None:
        """A Play-range triangle was dragged. Store the new bounds, pull the
        current frame back inside them if it fell out, and tell the shell so
        the MP4 export range stays in step."""
        self._range_start, self._range_end = int(start), int(end)
        if self._loaded_uid:
            self._range_by_series[self._loaded_uid] = (start, end)
        if self._frame < start or self._frame > end:
            # setValue → _seek re-clamps and re-renders.
            self.frame_slider.setValue(max(start, min(self._frame, end)))
        n = max((p.volume.shape[0] for p in self._planes), default=0)
        self.play_range_changed.emit(self._loaded_uid, start, end, n)
        self.readout.setText(
            f"Play range: frames {start + 1}–{end + 1} "
            f"({end - start + 1} of {n})"
        )

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
