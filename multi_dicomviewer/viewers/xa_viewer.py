"""XA angiography viewer: multiframe cine + window/level + calibrated tools.

Handles single-plane and biplane acquisitions. Biplane series carry an
in-pane "Plane:" bar (Bi / Lt / Rt) so each pane independently shows both
planes side by side (Bi) or a single plane (Lt = plane 0, Rt = plane 1).
"""
from __future__ import annotations

import math
import time

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygon, QPolygonF
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
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

#: Qt's "no maximum" sentinel for clearing a setMaximumHeight/Width.
_QWIDGETSIZE_MAX = 16777215

#: Seek-bar stylesheet — full-size (handle 18 px) vs compact (handle 12 px,
#: used in multi-row layouts so the transport strip is roughly half height).
_SEEK_QSS = (
    "QSlider::groove:horizontal{height:6px;border-radius:3px;"
    "background:#c4c4c4;}"
    "QSlider::handle:horizontal{width:18px;height:18px;"
    "margin:-6px 0;border:1px solid #6a6a6a;border-radius:9px;"
    "background:qradialgradient(cx:0.5,cy:0.5,radius:0.5,"
    "fx:0.5,fy:0.5,stop:0 #1c6fd0,stop:0.55 #1c6fd0,"
    "stop:0.60 #ffffff,stop:1 #ffffff);}"
)
_SEEK_QSS_COMPACT = (
    "QSlider::groove:horizontal{height:4px;border-radius:2px;"
    "background:#c4c4c4;}"
    "QSlider::handle:horizontal{width:12px;height:12px;"
    "margin:-4px 0;border:1px solid #6a6a6a;border-radius:6px;"
    "background:qradialgradient(cx:0.5,cy:0.5,radius:0.5,"
    "fx:0.5,fy:0.5,stop:0 #1c6fd0,stop:0.55 #1c6fd0,"
    "stop:0.60 #ffffff,stop:1 #ffffff);}"
)
#: Grey inner dot for the same handle — used on single-frame series, where
#: there is nothing to play and the Play button is greyed out. A blue dot there
#: looked "active" and was mistaken for a live control, so it is desaturated to
#: match the disabled Play button.
_SEEK_DOT_GREY = "#9a9a9a"
_SEEK_QSS_DISABLED = _SEEK_QSS.replace(_SEEK_DOT_BLUE, _SEEK_DOT_GREY)
_SEEK_QSS_COMPACT_DISABLED = _SEEK_QSS_COMPACT.replace(
    _SEEK_DOT_BLUE, _SEEK_DOT_GREY)


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
    _TRI_H = 11        # triangle height, px (drawn bottom-anchored on the strip)
    #: Extra clickable height ABOVE the triangle (toward the image). Keeps the
    #: triangle's bottom edge (apex) as the LOWEST grab point so the strip never
    #: reaches down onto the seek handle's blue dot, while giving a little more
    #: room to grab from the image side ("上は画像の下境界まで").
    _GRAB_TOP_PAD = 5
    #: Horizontal grab radius around a triangle's centre: the triangle itself
    #: (±_TRI_HW) plus one triangle-WIDTH (2·_TRI_HW) of margin on each side, so
    #: only a click on/near a triangle grabs it. Clicking elsewhere on the strip
    #: (e.g. over the seek handle's column) no longer snatches the nearest
    #: triangle — that wide reach was the "反応範囲が大きすぎ" complaint.
    _GRAB_HX = 3 * _TRI_HW

    def __init__(self, slider: QSlider, parent=None):
        super().__init__(parent)
        self._slider = slider
        self._start = 0
        self._end = 0
        self._drag: str | None = None      # "start" | "end" | None
        # Strip = triangle height + a little top grab room (toward the image).
        self.setFixedHeight(self._TRI_H + self._GRAB_TOP_PAD)
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
        base_y = h - self._TRI_H        # bottom-anchored: apex sits at h-1
        for val in (self._start, self._end):
            cx = self._value_x(val)
            p.drawPolygon(QPolygon([
                QPoint(int(round(cx - w)), base_y),
                QPoint(int(round(cx + w)), base_y),
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
        # Only grab when the press is within a triangle's horizontal reach
        # (_GRAB_HX). A click elsewhere on the strip does nothing, so it can no
        # longer drag a far triangle or fight the seek handle below.
        if min(ds, de) > self._GRAB_HX:
            return
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
            # No waveform in this series — keep the (ON) strip visible with a
            # clear "No ECG" so the user knows the toggle is still on.
            p.setPen(QColor("#cfe8ff"))
            p.drawText(6, 15, "ECG")
            p.setPen(QColor("#888"))
            p.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No ECG"
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


class _AutoHideLabel(QLabel):
    """A QLabel that collapses (hides) when its text is empty, so an idle
    status line reserves NO height. Used for the bottom readout — otherwise
    an empty label left a one-line gap under the seek bar."""

    def setText(self, text) -> None:
        super().setText(text)
        self.setVisible(bool(text))


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

    #: image right-click ▸ Export DICOM / MP4 / CSV — asks the shell to run
    #: that export scoped to the clicked image. Args: (fmt, series_uid,
    #: plane_path). plane_path is the clicked biplane plane's own .dcm file
    #: ("" = whole series, e.g. single-plane / IVUS).
    plane_export_requested = pyqtSignal(str, str, str)

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
        # Load-time W/L baseline — what "Reset" in the W/L popup restores.
        self._window_init = 1.0
        self._level_init = 0.5
        #: Compact (half-height) transport — set by the shell in multi-row
        #: layouts so the Play/seek strip doesn't crowd the smaller panes.
        self._compact = False
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
        #: Wall-clock cine baseline: the perf_counter time and frame index when
        #: the cine timer was last (re)started. _next_frame advances to the
        #: frame real time says we should be on (not strictly +1/tick), so a
        #: late timer tick — macOS starves timers while the mouse moves —
        #: catches up instead of slowing/stalling the cine.
        self._cine_t0 = 0.0
        self._cine_f0 = 0
        #: True while a coalesced seek render is queued (see _seek): a long
        #: pull-back's slider fires many valueChanged per mouse move; we render
        #: once per event-loop pass instead of per step so the handle keeps up.
        self._seek_render_pending = False

        self.canvas = ImageCanvas()    # primary / Front
        self.canvas2 = ImageCanvas()   # Lateral, only in side-by-side
        self.canvas2.hide()

        self.title_label = QLabel("—")
        self.title_label.setStyleSheet("color:#ccc; padding:2px 6px;")
        self.readout = _AutoHideLabel("")
        self.readout.setStyleSheet("color:#27e0c0; padding:2px 6px;")
        self.readout.setVisible(False)   # empty → no wasted line under the seek bar

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
        #: Optional shell callback (viewer) -> bool that vetoes STARTING
        #: playback when too many panes already play at once (set via
        #: set_play_gate). None = no limit.
        self._play_gate = None
        self._fps = DEFAULT_CINE_FPS
        # Cine speed multiplier toggled by D (1.0 = 1×, 2.0 = 2×).
        self._play_speed: float = 1.0
        # ECG strip (W key): the trace read from this series' DICOM, the
        # per-frame time step used to drive the cursor, and visibility.
        #: User's ON/OFF choice — PERSISTS across series (Next/Prev) until the
        #: user toggles it. When ON, the strip stays shown on every series; a
        #: series without an ECG shows "No ECG" rather than hiding the strip.
        self._ecg_on: bool = False
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
        # Transport (Play + seek) wrapped in a widget flagged to SURVIVE the
        # shell's "Max Image" (Hide Buttons) so playback stays usable while
        # the rest of the chrome is hidden.
        self._transport_bar = QWidget()
        self._transport_bar.setLayout(self._build_transport())
        self._transport_bar._mdv_keep_on_max = True
        layout.addWidget(self._transport_bar)
        # W/L is rarely used on XA/IVUS and ate a permanent toolbar row, so it
        # now lives in a small popup opened from the image right-click ▸
        # Change W/L (built here so the sliders exist before load_series).
        self._build_wl_dialog()
        layout.addWidget(self.readout)
        # Left-align every toolbar button's caption so a too-narrow button shows
        # the START of its label (e.g. "Clear All Result", the tool names) rather
        # than centring it and clipping both ends. Include VERTICAL padding: a
        # stylesheet turns off the native macOS button height, so without it the
        # button collapses and the text overflows top/bottom.
        # Rounded-corner buttons (match the CT viewer's shape). The base rule
        # only sets shape/border/neutral background, so per-button background
        # colours (Clear All Result, the active tool) still apply over it.
        self.setStyleSheet(
            "QPushButton {"
            " text-align: left; padding: 9px 10px;"
            " border: 1px solid #9a9a9a; border-radius: 6px;"
            " background: #ededed; }")

        self.canvas.measurement_done.connect(self._on_measurement)
        self.canvas2.measurement_done.connect(self._on_measurement)
        # Right-click ▸ Change W/L on either canvas opens the W/L popup.
        self.canvas.wl_change_requested.connect(self.show_wl_dialog)
        self.canvas2.wl_change_requested.connect(self.show_wl_dialog)
        self.canvas.export_requested.connect(self._on_export_image)
        self.canvas2.export_requested.connect(self._on_export_image)
        self.canvas.arrow_pressed.connect(self._on_canvas_arrow)
        self.canvas2.arrow_pressed.connect(self._on_canvas_arrow)

    def _on_canvas_arrow(self, direction: str) -> None:
        """Arrow key on a focused cine image: Up/Down step Prev/Next series,
        Left/Right step one frame back/forward. Left/Right are a no-op on a
        single-frame image (step_frame clamps), matching 'no seek bar → no
        left/right' from the spec; Up/Down always navigate series."""
        if direction == "up":
            self.series_nav.emit("prev")
        elif direction == "down":
            self.series_nav.emit("next")
        elif direction == "left":
            self.step_frame(-1)
        elif direction == "right":
            self.step_frame(+1)

    def _on_measurement(self, m):
        self.readout.setText(f"{m.kind}: {m.label()}")
        self.measurement_added.emit(m)

    def _on_export_image(self, fmt_key):
        """Right-click export on a canvas (no active measure tool).

        PNG/JPEG/TIFF save exactly what's on the clicked canvas (WYSIWYG).
        DICOM/MP4/CSV are routed to the shell scoped to the RIGHT-CLICKED
        plane: for a biplane series we pass that plane's own .dcm file so
        only that side is exported; single-plane / IVUS export the whole
        series."""
        canvas = self.sender()
        if not isinstance(canvas, ImageCanvas):
            canvas = self.canvas
        if fmt_key in ("dicom", "mp4", "csv", "anon-dicom"):
            self.plane_export_requested.emit(
                fmt_key, getattr(self, "_loaded_uid", ""),
                self._clicked_plane_path(canvas),
            )
            return
        # grab() captures the painted widget verbatim (WYSIWYG), at the
        # display's devicePixelRatio so Retina exports stay full-res.
        export_image_as(self, canvas.grab(), fmt_key, self._export_basename())

    def _clicked_plane_path(self, canvas) -> str:
        """The .dcm file backing the right-clicked canvas, but only for a
        biplane series (so the export targets just that plane). Returns ""
        for single-plane series — the shell then exports the whole series."""
        if not self._is_biplane:
            return ""
        # canvas2 always shows plane 1 (Rt); canvas shows plane 0 in Bi mode
        # or the active plane in single (Lt/Rt) mode.
        if canvas is self.canvas2:
            pidx = 1
        else:
            pidx = 0 if self._dual else self._active
        if 0 <= pidx < len(self._planes):
            return getattr(self._planes[pidx], "path", "") or ""
        return ""

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

        # 2-D image transforms (rotate 90° / flip), right of the zoom buttons.
        # Useful mainly for static Secondary-Capture (SC) images. "Mirror" ==
        # Flip-H (a left-right mirror), so it is not a separate button.
        self._orient_btns = []
        for label, kind, tip in (
            ("Rt90°", "rt90", "Rotate the image 90° clockwise"),
            ("Lt90°", "lt90", "Rotate the image 90° counter-clockwise"),
            ("Flip-H", "fliph", "Flip horizontally (left-right mirror)"),
            ("Flip-V", "flipv", "Flip vertically (top-bottom)"),
        ):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c, k=kind: self._image_transform(k))
            self._orient_btns.append(b)
            row.addWidget(b)

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
        # The DICOM-tag controls now live in the shell's global top row; the
        # per-viewer copy is kept (for set_overlay_font_pt sync) but hidden.
        tags_box.setVisible(False)
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
        """In-pane Plane switch: Bi (both planes), Lt (left/plane 0), Rt
        (right/plane 1). Positional labels — not Front/Lateral — so they
        stay unambiguous regardless of how the planes map to the canvases.
        Shown only for biplane series; hidden for single-plane cine."""
        self.plane_bar = QWidget()
        row = QHBoxLayout(self.plane_bar)
        row.setContentsMargins(2, 0, 2, 0)
        row.addWidget(QLabel("Plane:"))
        self._plane_group = QButtonGroup(self)
        self._plane_group.setExclusive(True)
        self._side_btns: dict[str, QPushButton] = {}
        for key in ("Bi", "Lt", "Rt"):
            btn = QPushButton(key)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c, k=key: self.set_side(k, True))
            self._plane_group.addButton(btn)
            self._side_btns[key] = btn
            row.addWidget(btn)
        row.addStretch(1)
        self.plane_bar.hide()
        return self.plane_bar

    def _build_transport(self):
        row = QHBoxLayout()
        # No top/bottom padding — the default layout margins (≈11 px each on
        # macOS) otherwise opened a gap between the image and the seek bar.
        row.setContentsMargins(4, 0, 4, 0)
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
        # Remember the natural Play-button point size so set_compact() can
        # scale it down (multi-row layouts) and back without drift.
        self._play_base_pt = self.play_btn.font().pointSizeF()

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self._seek)
        # True while the user is dragging the seek handle — render with fast
        # (nearest) upscaling during the drag so scrubbing a long pull-back is
        # smooth, then settle to crisp bilinear on release.
        self._seeking = False
        self.frame_slider.sliderPressed.connect(self._on_seek_begin)
        self.frame_slider.sliderReleased.connect(self._on_seek_end)
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
        # proportion (the "mark") moved here. set_compact() swaps in a
        # smaller-handle stylesheet for multi-row layouts.
        # Track compact / playable state so the handle dot (blue when a cine
        # can play, grey when single-frame) and the compact size stay in sync
        # whichever changes. Start grey: no multi-frame series is loaded yet.
        self._seek_compact = False
        self._seek_playable = False
        self._refresh_seek_style()

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
            "Click to reset the playback range to the full range "
            "(click to reset the Play range to the whole clip)"
        )
        self.frame_lbl.clicked.connect(self._reset_play_range)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 60.0)
        self.fps_spin.setValue(DEFAULT_CINE_FPS)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.valueChanged.connect(self._set_fps)

        # Remember the natural fonts so set_compact() can shrink the frame
        # counter + fps box to match the smaller Play/seek and back.
        self._frame_lbl_base_pt = self.frame_lbl.font().pointSizeF()
        self._fps_base_pt = self.fps_spin.font().pointSizeF()

        row.addWidget(self.play_btn)
        row.addLayout(seek_col, 1)
        row.addWidget(self.frame_lbl)
        row.addWidget(self.fps_spin)
        return row

    def _build_wl_dialog(self) -> None:
        """Create the small, non-modal Window/Level popup. The sliders are
        the same objects the rest of the viewer drives (load_series ranges,
        the IVUS colour toggle …); only their home changed — from a permanent
        toolbar row to this popup, opened from the image right-click ▸ Change
        W/L. A numeric label beside each slider shows the current value."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Window / Level")
        dlg.setWindowFlag(Qt.WindowType.Tool, True)
        grid = QGridLayout(dlg)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(8)

        grid.addWidget(QLabel("W"), 0, 0)
        self.win_slider = QSlider(Qt.Orientation.Horizontal)
        self.win_slider.setStyleSheet(_SLIDER_QSS)
        self.win_slider.setMinimumWidth(240)
        self.win_slider.valueChanged.connect(self._wl_changed)
        grid.addWidget(self.win_slider, 0, 1)
        self._win_val_lbl = QLabel("—")
        self._win_val_lbl.setMinimumWidth(52)
        grid.addWidget(self._win_val_lbl, 0, 2)

        grid.addWidget(QLabel("L"), 1, 0)
        self.lvl_slider = QSlider(Qt.Orientation.Horizontal)
        self.lvl_slider.setStyleSheet(_SLIDER_QSS)
        self.lvl_slider.setMinimumWidth(240)
        self.lvl_slider.valueChanged.connect(self._wl_changed)
        grid.addWidget(self.lvl_slider, 1, 1)
        self._lvl_val_lbl = QLabel("—")
        self._lvl_val_lbl.setMinimumWidth(52)
        grid.addWidget(self._lvl_val_lbl, 1, 2)

        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Reset W/L to the default setting")
        reset_btn.clicked.connect(self._reset_wl)
        grid.addWidget(
            reset_btn, 2, 1, 1, 2, Qt.AlignmentFlag.AlignRight
        )

        self._wl_dialog = dlg

    def _reset_wl(self) -> None:
        """Restore W/L to this series' load-time baseline (popup ▸ Reset)."""
        if not self.win_slider.isEnabled():
            return
        # setValue fires _wl_changed (which re-renders + refreshes labels);
        # if a value is already at the baseline its signal won't fire, but the
        # state is then already correct, so an explicit label refresh covers it.
        self.win_slider.setValue(int(self._window_init))
        self.lvl_slider.setValue(int(self._level_init))
        self._update_wl_labels()

    def show_wl_dialog(self) -> None:
        """Open the Window/Level popup (image right-click ▸ Change W/L).
        No-op while W/L is disabled (e.g. a colour series has no W/L)."""
        if not self.win_slider.isEnabled():
            self.readout.setText("This series has no W/L setting")
            return
        self._update_wl_labels()
        self._wl_dialog.show()
        self._wl_dialog.raise_()
        self._wl_dialog.activateWindow()

    def _update_wl_labels(self) -> None:
        """Refresh the popup's numeric W/L readout from the current state."""
        if hasattr(self, "_win_val_lbl"):
            self._win_val_lbl.setText(f"{int(self._window)}")
            self._lvl_val_lbl.setText(f"{int(self._level)}")

    def _sync_wl_enabled(self) -> None:
        """Mirror W/L availability onto the canvases so the image right-click
        ▸ Change W/L greys out when W/L doesn't apply (e.g. a colour IVUS)."""
        on = self.win_slider.isEnabled()
        for c in (self.canvas, self.canvas2):
            c.wl_enabled = on

    def set_compact(self, on: bool) -> None:
        """Shrink the Play/seek transport to ~half height (and a smaller Play
        button) for multi-row layouts; restore full size when off. Called by
        the shell from _apply_layout. W/L already moved to a popup, so the
        transport row is the only height left to reclaim for the image."""
        on = bool(on)
        if self._compact == on:
            return
        self._compact = on
        # Play button: scale the (1.3×) natural font down, cap its size.
        if self._play_base_pt and self._play_base_pt > 0:
            f = self.play_btn.font()
            f.setPointSizeF(self._play_base_pt * (0.85 if on else 1.0))
            self.play_btn.setFont(f)
        self.play_btn.setMaximumHeight(22 if on else _QWIDGETSIZE_MAX)
        # Seek slider: shorter groove + smaller handle.
        self.frame_slider.setMinimumHeight(16 if on else 24)
        self.frame_slider.setMaximumHeight(16 if on else _QWIDGETSIZE_MAX)
        self._seek_compact = on
        self._refresh_seek_style()
        # Frame counter + fps box: match the shrunk Play/seek (smaller font,
        # capped height) so they don't tower over the compact transport.
        for widget, base in (
            (self.frame_lbl, self._frame_lbl_base_pt),
            (self.fps_spin, self._fps_base_pt),
        ):
            if base and base > 0:
                f = widget.font()
                f.setPointSizeF(base * (0.8 if on else 1.0))
                widget.setFont(f)
            widget.setMaximumHeight(18 if on else _QWIDGETSIZE_MAX)
        self.frame_lbl.setMinimumWidth(48 if on else 70)
        # Trim the Play-range strip's spare grab room (the mostly-empty band
        # above the seek bar) so the seek bar hugs the image.
        self._range_marks.setFixedHeight(
            _RangeMarks._TRI_H if on
            else _RangeMarks._TRI_H + _RangeMarks._GRAB_TOP_PAD
        )
        # Collapse the inter-row spacing of the viewer's main column so no gap
        # is left between the image and the transport strip.
        lay = self.layout()
        if lay is not None:
            if not hasattr(self, "_base_layout_spacing"):
                self._base_layout_spacing = lay.spacing()
            lay.setSpacing(0 if on else self._base_layout_spacing)

    def _refresh_seek_style(self) -> None:
        """Apply the seek-bar stylesheet for the current compact + playable
        state: blue inner dot when a multi-frame cine can play, grey when the
        series is single-frame (Play greyed out)."""
        compact = getattr(self, "_seek_compact", False)
        playable = getattr(self, "_seek_playable", False)
        if playable:
            qss = _SEEK_QSS_COMPACT if compact else _SEEK_QSS
        else:
            qss = _SEEK_QSS_COMPACT_DISABLED if compact else _SEEK_QSS_DISABLED
        self.frame_slider.setStyleSheet(qss)

    def _clear_measurements(self):
        for c in (self.canvas, self.canvas2):
            c.clear_measurements()

    # -------------------------------------------- 2-D image transforms (SC)
    def _image_transform(self, kind: str) -> None:
        """Rotate 90° / flip the shown image(s). Applied to both canvases so a
        biplane pair stays consistent; for single-plane SC only canvas matters."""
        for c in (self.canvas, self.canvas2):
            c.apply_orient(kind)

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

    # -- Bi / Lt / Rt --------------------------------------------------
    @property
    def supports_side(self) -> bool:
        """True iff Bi/Lt/Rt has anything to switch (biplane only)."""
        return self._is_biplane

    def set_side(self, side: str, allow_dual: bool = True) -> None:
        """In-pane Plane switch (Bi/Lt/Rt):

        * ``Bi`` — both planes side by side (``allow_dual`` is accepted for
          API parity but ignored: "Bi" always means both, even when small).
        * ``Lt`` — single plane 0.
        * ``Rt`` — single plane 1.

        Single-plane series ignore this (nothing to switch)."""
        if not self._is_biplane:
            return
        if side == "Bi":
            self._want_dual = True
        elif side == "Lt":
            self._want_dual = False
            self._active = 0
        elif side == "Rt":
            self._want_dual = False
            self._active = 1
        self._relayout()

    def current_side(self) -> str | None:
        """Current Bi/Lt/Rt state. None when not biplane (nothing to
        switch)."""
        if not self._is_biplane:
            return None
        if self._dual:
            return "Bi"
        return "Lt" if self._active == 0 else "Rt"

    def _refresh_side_buttons(self) -> None:
        """Check the Bi/Lt/Rt button matching the current side."""
        side = self.current_side()
        for key, btn in self._side_btns.items():
            btn.setChecked(side == key)

    def _relayout(self) -> None:
        """Apply the current layout decision to the widgets. The Plane bar
        stays visible for any biplane series (so Bi↔Lt↔Rt is always one
        click away); only the second canvas toggles with Bi/single."""
        self.canvas2.setVisible(self._dual)
        self.plane_bar.setVisible(self._is_biplane)
        self._refresh_side_buttons()
        # The active plane may have changed (Lt↔Rt), so the DICOM-tag overlay
        # must follow it.
        self._refresh_overlay()
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
        # A genuinely new series starts in its native orientation (clear any
        # Rt90/flip left over from the previously shown image).
        for c in (self.canvas, self.canvas2):
            c.reset_orient()
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
        # A freshly loaded biplane series defaults to "Bi" (both planes);
        # single-plane series ignore this (_dual gates on _is_biplane).
        self._want_dual = True
        self._window = loaded.window or 1.0
        self._level = loaded.level or 0.0
        # Remember this load's W/L so the popup's Reset can restore it.
        self._window_init = self._window
        self._level_init = self._level
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
        self._sync_wl_enabled()
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
        # Grey the seek handle's inner dot when there's nothing to play, so it
        # no longer looks like an active control next to the greyed Play button.
        self._seek_playable = not single
        self._refresh_seek_style()
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
        # Map each canvas to the plane it is ACTUALLY showing so the DICOM-tag
        # overlay matches the image. In Bi mode canvas→plane 0, canvas2→plane
        # 1; in single (Lt/Rt) mode the visible canvas shows self._active, so
        # Rt must read plane 1's header (not plane 0's). Each XAPlane keeps
        # its own ``_ds``.
        if self._dual:
            plane_for = {0: 0, 1: 1}
        else:
            active = (min(self._active, len(self._planes) - 1)
                      if self._planes else 0)
            plane_for = {0: active}
        for ci, c in enumerate((self.canvas, self.canvas2)):
            pi = plane_for.get(ci)
            if pi is not None and pi < len(self._planes):
                ds = self._planes[pi]._ds
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
            self._start_cine_timer()

    def toggle_ecg(self) -> None:
        """W: turn the ECG strip ON/OFF. The choice PERSISTS across series
        (Next/Prev) until toggled again. When ON the strip stays visible on
        every series — showing that series' ECG, or 'No ECG' if it has none."""
        self._ecg_on = not self._ecg_on
        self._apply_ecg_visibility()
        if self._ecg_on:
            lab = (self._ecg_trace.label if (self._ecg_available
                                             and self._ecg_trace) else "No ECG")
            self.readout.setText(f"ECG: ON · {lab}")
        else:
            self.readout.setText("ECG: OFF")

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
        from the metadata header and prime the strip, then RE-APPLY the user's
        persistent ON/OFF choice — so navigating Next/Prev keeps the strip on
        (showing this series' ECG, or 'No ECG' if it has none) instead of
        resetting it off each time."""
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
            self._ecg_strip.set_trace(None)      # strip paints "No ECG"
        self._apply_ecg_visibility()

    def _apply_ecg_visibility(self) -> None:
        """Show/hide the ECG strip per the persistent ON/OFF choice. When ON the
        strip is always shown (it renders the trace, or 'No ECG' if absent)."""
        self._ecg_visible = self._ecg_on
        self._ecg_strip.setVisible(self._ecg_on)
        if self._ecg_on and self._ecg_available and self._ecg_trace is not None:
            self._ecg_strip.set_cursor_fraction(self._ecg_fraction(self._frame))

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
            # Fast path: one array gather instead of float math/clip. Index in
            # a wide int — a signed dtype (e.g. int16 MR/PSIR, _wl_off=32768)
            # would otherwise overflow when the offset is added (NumPy 2 / NEP50
            # raises OverflowError instead of upcasting), which crashed the app.
            if self._wl_off:
                return self._wl_lut[f.astype(np.intp) + self._wl_off]
            return self._wl_lut[f]
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
        # Nearest-neighbour upscaling while the cine timer runs (smoothing every
        # frame is a real per-frame cost on a high-DPI display); crisp bilinear
        # when paused. set_fast_scaling only repaints on an actual change, so
        # this is free to call each frame.
        fast = self._timer.isActive() or self._seeking
        self.canvas.set_fast_scaling(fast)
        self.canvas2.set_fast_scaling(fast)
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
        span = hi - lo + 1
        if span <= 1:
            return
        # Wall-clock target: advance to the frame real time says we should be
        # on, given the tempo set when the timer (re)started. A timer tick that
        # arrives late — macOS starves timers while the mouse is moving — then
        # jumps to the right position instead of the cine slowing to a crawl /
        # stalling. On Windows (precise, on-time ticks) this resolves to the
        # usual +1 per tick, so behaviour there is unchanged.
        fps = max(self._effective_fps(), 1e-3)
        elapsed = int((time.perf_counter() - self._cine_t0) * fps)
        nxt = lo + (self._cine_f0 - lo + max(0, elapsed)) % span
        if nxt == self._frame:
            return                      # not yet time for a new frame
        if self._dual:
            shown = self._planes
        else:
            self._active = min(self._active, len(self._planes) - 1)
            shown = [self._planes[self._active]]
        # Don't decode on the UI thread mid-cine: if the prefetch hasn't
        # warmed the target frame yet, hold (the timer keeps firing and
        # re-checks). The paced prefetch warms far faster than any cine fps,
        # so this only briefly holds at the very start; once warm it plays
        # straight from cache.
        if not all(p.is_ready(nxt) for p in shown):
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

    def set_play_gate(self, fn) -> None:
        """Shell sets a callback ``fn(viewer) -> bool``; returning False vetoes
        starting playback (e.g. the simultaneous-play cap). fn shows its own
        message."""
        self._play_gate = fn

    def is_playing(self) -> bool:
        """True while this viewer's cine is engaged (Play button down)."""
        return self.play_btn.isChecked()

    def _toggle_play(self, on: bool):
        n = max((p.volume.shape[0] for p in self._planes), default=0)
        if n < 2:
            self.play_btn.setChecked(False)
            return
        # Shell cap on simultaneous playback: if vetoed, snap the button back
        # to "Play" and do nothing (the gate already told the user why).
        if on and self._play_gate is not None and not self._play_gate(self):
            self.play_btn.blockSignals(True)
            self.play_btn.setChecked(False)
            self.play_btn.blockSignals(False)
            self.play_btn.setText("▶ Play")
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
        self._start_cine_timer()

    def _start_cine_timer(self) -> None:
        """(Re)start the cine timer AND reset the wall-clock baseline to the
        current frame/time. Call this anywhere the timer (re)starts (play,
        speed change, fps change) so _next_frame's catch-up maths reference the
        new tempo from 'now', not a stale origin."""
        self._cine_t0 = time.perf_counter()
        self._cine_f0 = self._frame
        self._timer.start(int(1000.0 / max(self._effective_fps(), 1e-3)))

    def _on_seek_begin(self) -> None:
        """Seek handle pressed — switch the cross-section to fast upscaling so
        scrubbing a long pull-back stays smooth (see _render)."""
        self._seeking = True

    def _on_seek_end(self) -> None:
        """Seek handle released — settle the final frame at crisp bilinear,
        cancelling any pending coalesced render."""
        self._seeking = False
        self._seek_render_pending = False
        self._render()
        if not self._suspend_frame_signal:
            self.frame_changed.emit(self._frame)

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
        # If the user seeks WHILE playing, re-base the wall-clock so the cine
        # continues from the new position instead of _next_frame snapping back
        # to where the old baseline says it should be.
        if self._timer.isActive():
            self._cine_t0 = time.perf_counter()
            self._cine_f0 = self._frame
        # Coalesce: a 4000+-frame slider fires many valueChanged per mouse
        # move; rendering each one can't keep up and the handle lags the
        # cursor. Update the model now but render once on the next event-loop
        # pass, collapsing a burst of steps into a single paint of the latest
        # frame. The slider widget itself repaints immediately, so the handle
        # tracks the mouse while the image catches up a tick later.
        if not self._seek_render_pending:
            self._seek_render_pending = True
            QTimer.singleShot(0, self._flush_seek_render)

    def _flush_seek_render(self) -> None:
        """Render the latest sought frame (deferred from _seek). A no-op if a
        release already settled it (pending cleared in _on_seek_end)."""
        if not self._seek_render_pending:
            return
        # While the handle is held, NEVER block-decode a cold frame on the UI
        # thread: a compressed frame is ~10-15 ms and the JPEG codec holds the
        # GIL, and XAPlane.frame() takes a lock shared with the background
        # prefetch — so a synchronous decode here stalls the whole window. With
        # several series loaded across a multi-pane (2×3) grid the prefetch is
        # spread thin, scrub targets are far more often cold, and the stalls
        # pile up into a freeze. Hold instead: poll until the prefetch warms the
        # target (the strip keeps showing the last decoded frame), and let
        # _on_seek_end settle — and decode once — on release.
        if self._seeking and not self._seek_frame_ready():
            self._seek_render_pending = True
            QTimer.singleShot(40, self._flush_seek_render)
            return
        self._seek_render_pending = False
        self._render()
        if not self._suspend_frame_signal:
            self.frame_changed.emit(self._frame)

    def _seek_frame_ready(self) -> bool:
        """True when the current frame is already decoded on every shown plane
        — so rendering it won't block-decode on the UI thread."""
        return all(p.is_ready(self._frame) for p in self._shown_planes())

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
            self._start_cine_timer()

    def _wl_changed(self):
        self._window = float(self.win_slider.value())
        self._level = float(self.lvl_slider.value())
        self._update_wl_labels()
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
        # Compare two Ellipse/Polygon: %Area (%PA) and/or radial gap (Thickness)
        # — same flow as the CT viewers. Right of Angle, Clear All Result after.
        self._cmp_btn = QPushButton("Compare")
        self._cmp_btn.setCheckable(True)
        self._cmp_btn.setToolTip(
            "Compare two Ellipse/Polygon: click the two shapes, then pick %PA "
            "and/or Thickness")
        self._cmp_btn.clicked.connect(self._toggle_compare)
        row.addWidget(self._cmp_btn)
        # Hide/Show ALL results (lines + region colours + text) at once, between
        # Compare and Clear All Result. Disabled when there is nothing to hide.
        self._hideall_btn = QPushButton("Hide All Result")
        self._hideall_btn.setToolTip(
            "Hide / Show every measurement line, region colour and result text")
        self._hideall_btn.clicked.connect(self._toggle_hide_all)
        row.addWidget(self._hideall_btn)
        clr = QPushButton("Clear All Result")
        clr.setToolTip("Clear all measurements and comparison results")
        clr.setStyleSheet("background:#bdbdbd;color:black;")   # match 3DCT
        clr.clicked.connect(self._clear_measurements)
        row.addWidget(clr)
        row.addWidget(QLabel(
            "  Left-click = add point /"
            " right-click finishes Polyline / Polygon"
        ))
        row.addStretch(1)
        # A canvas that finishes/cancels a Compare un-checks the button; any
        # change to the result set refreshes the Hide/Show-All button.
        for c in (self.canvas, self.canvas2):
            c.compare_finished.connect(self._on_compare_finished)
            c.results_changed.connect(self._update_hideall_btn)
            c.measurement_done.connect(lambda _m: self._update_hideall_btn())
        self._update_hideall_btn()
        return bar

    def _toggle_hide_all(self):
        hide = not getattr(self.canvas, "_results_hidden", False)
        for c in (self.canvas, self.canvas2):
            c._results_hidden = hide
            if not hide:
                # Show All → reveal EVERYTHING, including results that were
                # individually hidden, regardless of their per-item Hide.
                for m in c.measures:
                    m.pop("hidden", None)
                for cmp in c._compares:
                    cmp.pop("hidden", None)
            c.update()
        self._update_hideall_btn()

    def _update_hideall_btn(self):
        btn = getattr(self, "_hideall_btn", None)
        if btn is None:
            return
        has = any(c.measures or c._compares
                  for c in (self.canvas, self.canvas2))
        hidden = getattr(self.canvas, "_results_hidden", False)
        btn.setEnabled(has)
        btn.setText("Show All Result" if hidden else "Hide All Result")

    def _toggle_compare(self):
        on = self._cmp_btn.isChecked()
        for c in (self.canvas, self.canvas2):
            c.set_compare_mode(on)

    def _on_compare_finished(self):
        self._cmp_btn.setChecked(False)
        for c in (self.canvas, self.canvas2):
            c.set_compare_mode(False)

    def _toggle_measure(self):
        on = self._meas_btn.isChecked()
        self._measure_bar.setVisible(on)
        if on:
            self._clear_zoom_click()
        if not on:
            for c in (self.canvas, self.canvas2):
                c.set_measure_type("")
                c.set_compare_mode(False)
            self._cmp_btn.setChecked(False)
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
