"""Pan/zoom-free fit-to-window image canvas with a full measurement
overlay matching the CT viewer.

Shared by the XA and IVUS viewers. Holds an 8-bit grayscale (or RGB)
frame plus a list of completed measurements drawn in image-pixel space.
Tools: Line / Polyline / Ellipse / Polygon — the same set as the CT
viewer — with running numbers, draggable handles, right-click delete,
spline-smoothed polygon outline (centripetal Catmull-Rom through the
clicked points), and faint thick dotted 長径/短径 (major/minor) caliper
guides on ellipses and polygons. Lengths/areas are in mm when the series
is calibrated (``spacing_mm`` set), in pixels otherwise.
"""
from __future__ import annotations

import math
import sys

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor, QCursor, QFont, QIcon, QImage, QPainter, QPainterPath, QPen,
    QPixmap, QPolygon, QPolygonF,
)
from PyQt6.QtWidgets import QMenu, QWidget

from multi_dicomviewer.core import measure_geom as G
from multi_dicomviewer.ui.compare_options import CompareOptionsDialog
from multi_dicomviewer.core.coaxial import VESSEL_LABELS
from multi_dicomviewer.core.image_export import pick_export_format
from multi_dicomviewer.core.measurements import Measurement
from multi_dicomviewer.ui.measure_style_menu import (
    add_color_submenu, add_transparency_submenu, transp_to_alpha,
)
from multi_dicomviewer.ui.tag_font import TAG_FONT_PT_DEFAULT, overlay_qfont


def to_qimage(frame8: np.ndarray) -> QImage:
    """uint8 frame -> QImage (copied, owns its buffer). Accepts 2-D
    (H, W) grayscale or 3-D (H, W, 3) RGB."""
    frame8 = np.ascontiguousarray(frame8)
    if frame8.ndim == 3:
        h, w = frame8.shape[:2]
        img = QImage(
            frame8.data, w, h, 3 * w, QImage.Format.Format_RGB888
        )
    else:
        h, w = frame8.shape
        img = QImage(
            frame8.data, w, h, w, QImage.Format.Format_Grayscale8
        )
    return img.copy()


# Back-compat alias.
gray_to_qimage = to_qimage


# Tool names that match the CT viewer.
_TOOLS = ("line", "polyline", "ellipse", "polygon", "angle")
_PRETTY = {"line": "Line", "polyline": "Polyline",
           "ellipse": "Ellipse", "polygon": "Polygon", "angle": "Angle"}

#: 16-colour palette for Change Color (high-contrast on grayscale).
COLOR_CHOICES: list[tuple[str, str]] = [
    ("Cyan", "#33E6FF"), ("Lime", "#7CFC00"),
    ("Yellow", "#FFD700"), ("Orange", "#FFA500"),
    ("Red", "#FF4040"), ("Magenta", "#FF40FF"),
    ("Pink", "#FF80C0"), ("Purple", "#A040FF"),
    ("Blue", "#4090FF"), ("Teal", "#40D0C0"),
    ("Light Green", "#A0FF80"), ("Light Yellow", "#FFFF80"),
    ("Light Red", "#FF8080"), ("Light Cyan", "#80FFFF"),
    ("White", "#FFFFFF"), ("Gray", "#A0A0A0"),
]
DEFAULT_MEAS_COLOR = COLOR_CHOICES[0][1]   # cyan


def draw_outlined_line(p: QPainter, x: float, y: float, text: str,
                       fill: QColor, width: float = 1.0) -> None:
    """Draw one baseline line of *text* in *fill* with a thin black outline —
    the SAME readability treatment the CT viewer gives its DICOM tags, so every
    modality's overlay reads cleanly over bright/white images. Black copies are
    placed on concentric rings spaced <=0.9 px apart (so neighbours overlap into
    a smooth edge), then the fill is drawn on top. Leaves the pen set to *fill*."""
    dirs = ((-1, -1), (0, -1), (1, -1), (-1, 0),
            (1, 0), (-1, 1), (0, 1), (1, 1))
    radii = []
    r = min(0.9, width)
    while r < width - 1e-6:
        radii.append(r)
        r += 0.9
    radii.append(width)
    p.setPen(QColor(0, 0, 0))
    for rad in radii:
        for ox, oy in dirs:
            p.drawText(QPointF(x + ox * rad, y + oy * rad), text)
    p.setPen(fill)
    p.drawText(QPointF(x, y), text)


class ImageCanvas(QWidget):
    measurement_done = pyqtSignal(object)  # emits Measurement
    #: IVUS long-axis: user dragged the rotation-centre marker. Emits the
    #: NEW (cx, cy) in image pixels — the viewer stores it per frame and
    #: re-renders the long-axis strip.
    ivus_center_changed = pyqtSignal(float, float)
    #: IVUS long-axis: right-click reset menu choice. "frame" → this
    #: frame's rotation centre back to the image centre; "all" → every
    #: frame's rotation centre back to the image centre.
    ivus_center_reset = pyqtSignal(str)
    #: IVUS long-axis: user dragged the yellow cut line on the cross-section
    #: to re-angle the long-axis plane. Emits the NEW absolute angle
    #: (radians, image coords); the end dragged toward becomes the strip's
    #: bottom. ``ivus_angle_finished`` fires on release so the host can
    #: rebuild the strip at full resolution (the drag previews at draft LOD).
    ivus_angle_changed = pyqtSignal(float)
    ivus_angle_finished = pyqtSignal()
    #: Right-click ▸ Change W/L on the image → the host (XA/IVUS viewer)
    #: opens its small Window/Level popup. The canvas doesn't own the W/L
    #: state, so it just forwards the request.
    wl_change_requested = pyqtSignal()
    #: Right-click on empty image area while NO measure tool is active →
    #: request a still-image export. Carries the chosen format key
    #: ('png'/'jpeg'/'tiff'); the viewer captures this canvas and saves it
    #: (it has the patient/series/frame context for the default filename).
    #: self.sender() identifies which canvas fired.
    export_requested = pyqtSignal(str)
    #: Arrow key pressed while THIS canvas has keyboard focus (click-to-focus).
    #: Carries "up"/"down"/"left"/"right"; the cine viewer maps up/down to
    #: Prev/Next series and left/right to prev/next frame. Because it only fires
    #: when the image itself is focused, clicking into a spin-box / slider hands
    #: arrow control back to that control, and the Tree / CT panes keep their own
    #: arrow behaviour when THEY are focused.
    arrow_pressed = pyqtSignal(str)

    #: Emitted when a Compare gesture ends (computed or cancelled) so the host
    #: viewer can un-check its "Compare" button.
    compare_finished = pyqtSignal()

    #: Emitted whenever the set of measurements / comparisons changes (add,
    #: delete, clear) so the host can refresh its "Hide/Show All Result" button.
    results_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(80, 80)
        # Click-to-focus so a click on the image makes it the keyboard target:
        # arrow keys then drive cine nav (see keyPressEvent / arrow_pressed).
        # Clicking a spin-box / slider moves focus there and hands the arrows
        # back to that control.
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        # Mouse tracking (move events with NO button held) is only needed for
        # the live hover preview while DRAWING a measurement. Leaving it on
        # floods the UI thread with move events whenever the cursor crosses the
        # image — on macOS that starves the cine timer, so just moving the
        # mouse briefly stalls playback. Enabled on demand in set_measure_type
        # (button-held drags — pan, handle edit, IVUS centre — still deliver
        # move events without it). Default off.
        self.setMouseTracking(False)
        # Qt fires contextMenuEvent on right-click under the default
        # policy, which on some Windows + OpenGL combinations races our
        # right-click-to-close-polygon handler. Force the right button to
        # only go through mousePressEvent.
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self._qimg: QImage | None = None
        self._img_size = (0, 0)            # (w, h) in image px
        self.spacing_mm = None             # (row, col) mm/px, or None
        self.overlay_lines: list[str] = []  # DICOM-tag text, top-left
        #: On-image text size (pt) for the DICOM-tag overlay AND the
        #: measurement readout — kept equal so the two match. Driven by the
        #: shared 'tag text size' slider via set_overlay_font_pt().
        self._overlay_font_pt = TAG_FONT_PT_DEFAULT

        # Measurement state — same model as the CT viewer.
        self.meas_type: str = ""           # "" | line | polyline | ellipse | polygon
        self.measures: list[dict] = []     # finalised {id, type, pts}
        self._meas_seq = 0
        self._draft: dict | None = None    # {type, pts}
        self._edit: dict | None = None     # {mi, vi}
        self._hover: tuple[float, float] | None = None
        # Compare (%PA + radial Thickness gap between two Ellipse/Polygon),
        # same flow as the CT viewers: arm with the Compare button, click two
        # shapes, pick analysis, results persist (right-click a fill → Delete).
        self._cmp_on: bool = False           # Compare-select mode
        self._cmp_sel: list[int] = []        # picked measure indices (max 2)
        self._compares: list[dict] = []      # persisted results
        self._results_hidden: bool = False   # "Hide/Show All Result" global toggle
        self._cmp_want_pa: bool = True       # last-used: %PA (IVUS default)
        self._cmp_want_thk: bool = False     # last-used: Thickness
        # Zoom (Z = zoom in / Shift+Z = zoom out / mouse wheel also).
        # Wheel zooms toward the cursor; Z/Shift+Z toward the view centre.
        self._zoom: float = 1.0
        # Pan offset in widget pixels, added on top of the centred fit so
        # the user can bring a specific region into view. Middle-button OR
        # Ctrl+left drag pans; reset_zoom clears both zoom and pan.
        self._pan: list[float] = [0.0, 0.0]
        self._panning: bool = False
        self._pan_anchor: tuple[float, float] | None = None
        # Click-to-zoom mode driven by the toolbar magnifier buttons:
        # "" off, "in" → each left-click zooms 1.1× about the clicked point,
        # "out" → 0.9×. Independent of the measure tool (they're made
        # mutually exclusive by the viewer).
        self._zoom_click: str = ""
        # C-arm (PositionerPrimary, Secondary) angles of whatever plane is
        # CURRENTLY shown on this canvas, set by the XA viewer on every
        # render. Stamped into a measurement when it is committed so the
        # Coaxial-Eval tool knows the exact view each line was drawn on —
        # essential for biplane shown single-plane, where one canvas shows
        # different planes (hence different angles) over time. None for
        # non-angio canvases.
        self.view_angles: tuple[float, float] | None = None
        # Center-Angle pick mode: when set to a measure index, the next
        # 3 left-clicks add perimeter points to that measure and an
        # arc angle is computed.
        self._center_angle_target: int = -1
        # Draggable measurement labels. ``_drag_label`` = index of the
        # measure whose id/vessel-tag label is being dragged (-1 = none).
        # ``_label_rects`` caches each label's on-screen rect from the last
        # paint for hit-testing. The chosen spot rides in ``m["label_pos"]``
        # (image coords) so it tracks the shape through zoom/pan and survives
        # the canvas state save/restore. Lets the user pull a label off an
        # overlapping neighbour.
        self._drag_label: int = -1
        self._label_rects: dict[int, QRect] = {}
        # IVUS long-axis: per-frame rotation-centre marker. Shown only
        # while the IVUS long-axis view is on; the IVUSViewer sets
        # ``ivus_show_center`` and pushes ``ivus_center_image`` whenever
        # the current frame's centre changes (default = image centre).
        self.ivus_show_center: bool = False
        self.ivus_center_image: tuple[float, float] | None = None
        self._ivus_dragging_center: bool = False
        #: True while the user is dragging the yellow cut line to re-angle
        #: the long-axis plane (set in mousePressEvent on a cut-line hit).
        self._ivus_dragging_angle: bool = False
        # True iff the CURRENT frame's rotation centre was explicitly
        # pinned by the user (= keyframe). Drives the marker colour:
        # red = pinned/fixed, blue = interpolated/movable. IVUSViewer
        # sets this whenever the active frame or the keyframe set
        # changes.
        self.ivus_center_keyed: bool = False
        # Whether W/L applies to the shown series — the host (XA/IVUS viewer)
        # sets this from its slider state. False on a colour series (e.g. a
        # colour IVUS), so the right-click ▸ Change W/L item greys out.
        self.wl_enabled: bool = True
        # Current long-axis cut angle (radians, image coords) pushed in by
        # IVUSViewer. Used to draw the MPR-style cut line + projection-
        # direction triangles through the rotation centre: the strip's TOP
        # is 90° counter-clockwise from the triangle direction, its BOTTOM
        # 90° clockwise (visual sense, y-down).
        self.ivus_la_angle: float = 0.0
        # 2-D display orientation (Rt90 / Lt90 / Flip), used mainly for static
        # Secondary-Capture (SC) images. Stored as a dihedral element: an
        # optional horizontal flip (_t_flip) followed by _t_rot ×90° clockwise.
        # Applied to every frame in set_frame so cine frames stay consistent.
        self._t_rot: int = 0
        self._t_flip: bool = False
        self._raw_frame8: np.ndarray | None = None
        # When True, scale the frame with fast nearest-neighbour instead of
        # bilinear in paintEvent. The viewer turns this on during cine playback
        # / seek-drag (where smoothing every frame costs real time on a high-DPI
        # display — the Mac "playback not as smooth as Windows" report) and off
        # when paused so a still frame is still crisply upscaled.
        self._fast_scale: bool = False

    # ---------------------------------------------------------------- public
    def set_frame(self, frame8: np.ndarray) -> None:
        self._raw_frame8 = frame8
        f = self._oriented(frame8)
        self._qimg = to_qimage(f)
        self._img_size = (f.shape[1], f.shape[0])
        self.update()

    def set_fast_scaling(self, on: bool) -> None:
        """Toggle fast (nearest) vs smooth (bilinear) frame upscaling. Set on
        during cine playback / seek-drag, off when paused. Only repaints when
        the flag actually changes (so it's free to call every frame)."""
        on = bool(on)
        if on != self._fast_scale:
            self._fast_scale = on
            self.update()

    def _oriented(self, frame8: np.ndarray) -> np.ndarray:
        """Apply the current Rt90/flip orientation to *frame8* (flip first,
        then rotate clockwise). Returns a contiguous array for to_qimage."""
        f = frame8
        if self._t_flip:
            f = np.fliplr(f)
        r = self._t_rot % 4
        if r:
            f = np.rot90(f, -r)            # -r → r×90° clockwise
        return np.ascontiguousarray(f)

    def apply_orient(self, kind: str) -> None:
        """Rotate 90° (rt90/lt90) or flip (fliph/flipv) the displayed image,
        composing with the current orientation (so each acts on what is shown).
        Clears measurements since the displayed pixel grid changes."""
        if kind == "rt90":
            self._t_rot = (self._t_rot + 1) % 4
        elif kind == "lt90":
            self._t_rot = (self._t_rot - 1) % 4
        elif kind == "fliph":            # M ∘ (R^r M^f) = R^-r M^(f+1)
            self._t_rot = (-self._t_rot) % 4
            self._t_flip = not self._t_flip
        elif kind == "flipv":            # (R^2 M) ∘ (R^r M^f) = R^(2-r) M^(f+1)
            self._t_rot = (2 - self._t_rot) % 4
            self._t_flip = not self._t_flip
        else:
            return
        self.clear_measurements()
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        if self._raw_frame8 is not None:
            self.set_frame(self._raw_frame8)

    def reset_orient(self) -> None:
        """Restore the native (un-rotated, un-flipped) orientation."""
        self._t_rot = 0
        self._t_flip = False
        if self._raw_frame8 is not None:
            self.set_frame(self._raw_frame8)

    def set_spacing(self, spacing_mm) -> None:
        self.spacing_mm = spacing_mm

    def set_overlay(self, lines) -> None:
        """Replace the top-left DICOM-tag overlay text (list of str)."""
        self.overlay_lines = list(lines or [])
        self.update()

    def clear_measurements(self) -> None:
        self.measures.clear()
        self._draft = None
        self._edit = None
        self._hover = None
        self._compares.clear()
        self._cmp_sel = []
        self._results_hidden = False
        self._center_angle_target = -1
        self.update()
        self.results_changed.emit()

    # ----------------------------------------------------- zoom (Z key)
    def _apply_zoom(self, factor: float, ax: float, ay: float) -> None:
        """Multiply zoom by *factor*, keeping the image point currently
        under widget pixel (ax, ay) anchored to that same pixel. This is
        what lets the wheel zoom toward the cursor and a specific region
        stay put instead of the image always scaling about its centre."""
        new_zoom = max(0.25, min(8.0, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-9:
            return
        anchor_img = self._widget_to_image_f(ax, ay)
        self._zoom = new_zoom
        after = self._image_to_widget_f(anchor_img)
        self._pan[0] += ax - after[0]
        self._pan[1] += ay - after[1]
        self._clamp_pan()
        self.update()

    def zoom_in(self) -> None:
        self._apply_zoom(1.25, self.width() / 2.0, self.height() / 2.0)

    def zoom_out(self) -> None:
        self._apply_zoom(1 / 1.25, self.width() / 2.0, self.height() / 2.0)

    def reset_zoom(self) -> None:
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self.update()

    def set_zoom_click_mode(self, mode: str) -> None:
        """Enable click-to-zoom: 'in' (1.1× per click), 'out' (0.9×), or ''
        to turn it off. The cursor becomes a pointing hand while active."""
        self._zoom_click = mode if mode in ("in", "out") else ""
        if self._zoom_click:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(
                Qt.CursorShape.CrossCursor if self.meas_type
                else Qt.CursorShape.ArrowCursor
            )

    def wheelEvent(self, e):
        """Mouse wheel zooms toward the cursor — natural counterpart to the
        Z / Shift+Z keyboard shortcuts (which zoom about the view centre)."""
        pos = e.position()
        factor = 1.25 if e.angleDelta().y() > 0 else 1 / 1.25
        self._apply_zoom(factor, pos.x(), pos.y())

    def _clamp_pan(self) -> None:
        """Keep the image rect overlapping the widget so it can never be
        panned completely out of sight. Allows the image edge to reach the
        widget centre, which is enough to inspect any corner."""
        w, h = self._img_size
        if w == 0 or h == 0:
            return
        cw, ch = self.width(), self.height()
        base = min(cw / w, ch / h) * self._zoom
        dw, dh = w * base, h * base
        # Centred-rect origin before pan.
        ox, oy = (cw - dw) / 2.0, (ch - dh) / 2.0
        # Permit panning until the image edge reaches the widget centre
        # (so any corner can be brought into view, but the image can never
        # be pushed completely off-screen).
        self._pan[0] = max(cw / 2.0 - dw - ox, min(cw / 2.0 - ox, self._pan[0]))
        self._pan[1] = max(ch / 2.0 - dh - oy, min(ch / 2.0 - oy, self._pan[1]))

    def set_measure_type(self, t: str) -> None:
        """Select a tool ('' to disable). The list of finished measures
        is kept across tool changes (same as the CT viewer)."""
        self.meas_type = t if t in _TOOLS else ""
        self._draft = None
        self._edit = None
        self._hover = None
        # Hover preview (and thus move-event tracking) is only needed while a
        # drawing tool is active; off otherwise so idle/playback mouse motion
        # doesn't flood the event loop (see __init__).
        self.setMouseTracking(bool(self.meas_type))
        self.setCursor(
            Qt.CursorShape.CrossCursor if self.meas_type
            else Qt.CursorShape.ArrowCursor
        )
        self.update()

    # Back-compat for the old distance/angle API.
    def set_measure_mode(self, mode: str) -> None:
        self.set_measure_type("line" if mode == "distance" else "")

    @property
    def measurements(self):                # back-compat read-only view
        return self.measures

    # ----------------------------------------------------- coord transforms
    def _draw_rect(self) -> QRect:
        w, h = self._img_size
        if w == 0 or h == 0:
            return QRect()
        cw, ch = self.width(), self.height()
        scale = min(cw / w, ch / h) * self._zoom    # zoom multiplies fit
        dw, dh = int(w * scale), int(h * scale)
        x = (cw - dw) // 2 + int(round(self._pan[0]))
        y = (ch - dh) // 2 + int(round(self._pan[1]))
        return QRect(x, y, dw, dh)

    def _widget_to_image(self, p: QPoint) -> tuple[float, float] | None:
        r = self._draw_rect()
        if not r.isValid() or not r.contains(p):
            return None
        w, h = self._img_size
        fx = (p.x() - r.x()) / r.width() * w
        fy = (p.y() - r.y()) / r.height() * h
        return (fx, fy)

    def _widget_to_image_f(self, x: float, y: float) -> tuple[float, float]:
        """Unbounded widget→image map (no rect-contains gate) for zoom-about
        -point maths, which must work even when the cursor sits in the
        letterbox margin outside the image."""
        r = self._draw_rect()
        w, h = self._img_size
        if not r.isValid():
            return (0.0, 0.0)
        return ((x - r.x()) / r.width() * w, (y - r.y()) / r.height() * h)

    def _image_to_widget(self, pt: tuple[float, float]) -> QPoint:
        wf = self._image_to_widget_f(pt)
        return QPoint(int(wf[0]), int(wf[1]))

    def _image_to_widget_f(self, pt: tuple[float, float]) -> tuple[float, float]:
        r = self._draw_rect()
        w, h = self._img_size
        if not r.isValid():
            return (0.0, 0.0)
        return (r.x() + pt[0] / w * r.width(),
                r.y() + pt[1] / h * r.height())

    # ----------------------------------------------------- IVUS centre helpers
    def _ivus_center_widget(self) -> QPoint | None:
        """Centre marker's position in widget pixels, or None when the
        marker isn't displayed yet (no image set, or feature off)."""
        if (not self.ivus_show_center
                or self.ivus_center_image is None
                or self._img_size == (0, 0)):
            return None
        return self._image_to_widget(self.ivus_center_image)

    def _hit_ivus_center(self, sx: float, sy: float,
                         radius: float = 12.0) -> bool:
        """True when (sx, sy) widget coords are within *radius* px of the
        centre marker. Default ~12 px for the left-drag grab; the right-
        click menu uses a wider 18 px band."""
        wp = self._ivus_center_widget()
        if wp is None:
            return False
        return math.hypot(sx - wp.x(), sy - wp.y()) <= radius

    def _ivus_cutline_widget_pts(self):
        """(wt, wb) widget-pixel endpoints of the long-axis cut line, or
        None when the guide isn't shown. Mirrors the geometry drawn in
        ``_paint_ivus_long_axis``."""
        if (not self.ivus_show_center
                or self.ivus_center_image is None
                or self._img_size == (0, 0)):
            return None
        cx_i, cy_i = self.ivus_center_image
        a = float(self.ivus_la_angle)
        w_img, h_img = self._img_size
        half = math.hypot(w_img, h_img) / 2.0
        ux, uy = math.cos(a), math.sin(a)
        wt = self._image_to_widget_f((cx_i - half * ux, cy_i - half * uy))
        wb = self._image_to_widget_f((cx_i + half * ux, cy_i + half * uy))
        return wt, wb

    def _hit_ivus_cutline(self, sx: float, sy: float) -> bool:
        """True when (sx, sy) is within ~15 px of the cut line segment
        (a wide grab band — 2.5× the original 6 px — so the thin dashed
        line is easy to catch). The rotation-centre grab region is checked
        first by the caller, so the unstable near-centre part never wins."""
        pts = self._ivus_cutline_widget_pts()
        if pts is None:
            return False
        (x1, y1), (x2, y2) = pts
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        if l2 < 1e-6:
            return False
        t = max(0.0, min(1.0, ((sx - x1) * dx + (sy - y1) * dy) / l2))
        px, py = x1 + t * dx, y1 + t * dy
        return math.hypot(sx - px, sy - py) <= 15.0

    def _set_cutline_from_widget(self, sx: float, sy: float) -> None:
        """Re-angle the long-axis plane so the cut line points from the
        rotation centre toward the cursor (image coords). Repaints
        immediately and emits ``ivus_angle_changed`` so the viewer
        rebuilds the strip; the end dragged toward becomes the strip's
        bottom (sample direction = +angle)."""
        if self.ivus_center_image is None:
            return
        ix, iy = self._widget_to_image_f(sx, sy)
        cx_i, cy_i = self.ivus_center_image
        dx, dy = ix - cx_i, iy - cy_i
        if math.hypot(dx, dy) < 1e-3:
            return
        a = math.atan2(dy, dx) % (2 * math.pi)
        self.ivus_la_angle = a
        self.update()
        self.ivus_angle_changed.emit(a)

    # ----------------------------------------------------- mm helpers
    def _mm_scale(self) -> tuple[float, float]:
        """(sx, sy) px -> mm. (1, 1) when uncalibrated => 'px' label."""
        if self.spacing_mm is None:
            return 1.0, 1.0
        sy, sx = self.spacing_mm           # (row mm, col mm)
        return float(sx), float(sy)

    def _to_mm(self, p):
        sx, sy = self._mm_scale()
        return (p[0] * sx, p[1] * sy)

    def _unit(self) -> str:
        return "mm" if self.spacing_mm is not None else "px"

    # ----------------------------------------------------- per-measure ops
    def _outline_px(self, m) -> list[tuple[float, float]]:
        """Drawn outline in IMAGE-pixel coords.
        Polygon → smoothed closed; Polyline → optionally smoothed (Spline
        toggle); Angle → two rays meeting at the vertex, clicked as
        end1 → vertex → end2 and drawn straight through in that order."""
        t = m["type"]
        if t == "line":
            return list(m["pts"][:2])
        if t == "polyline":
            pts = list(m["pts"])
            if m.get("smooth"):
                return G.smooth_open(pts)
            return pts
        if t == "polygon":
            return G.smooth_closed(m["pts"])
        if t == "angle":
            # Clicked end1 → vertex → end2; draw straight through so the
            # vertex sits between its two rays.
            return list(m["pts"][:3])
        return G.ellipse_outline(m["pts"])            # ellipse (rotated)

    def _axis_segs_px(self, m):
        """Major/minor caliper segments in IMAGE-pixel coords, computed
        from the points so the lines match what we report. (None list if
        the shape is degenerate.)"""
        if m["type"] == "ellipse":
            maj, mnr, _, _ = G.major_minor(m)
            return [maj, mnr]
        if m["type"] != "polygon":
            return []
        # Geometry done in mm-space (correct for anisotropic spacing),
        # then mapped back so the drawn segments line up with the shape.
        sx, sy = self._mm_scale()
        m_mm = {"type": "polygon",
                "pts": [self._to_mm(q) for q in m["pts"]]}
        maj_mm, mnr_mm, _, _ = G.major_minor(m_mm)
        segs = []
        for s in (maj_mm, mnr_mm):
            if s is None:
                continue
            (a, b) = s
            segs.append((
                (a[0] / sx, a[1] / sy),
                (b[0] / sx, b[1] / sy),
            ))
        return segs

    def _metrics(self, m) -> str:
        """Result text — mm if calibrated, px otherwise. No HU here
        because XA/IVUS pixels aren't Hounsfield units; lengths, area
        and 長径/短径 are the clinically meaningful numbers."""
        u = self._unit()
        pts_mm = [self._to_mm(q) for q in m["pts"]]
        t = m["type"]
        ca = m.get("center_angle")
        ca_str = (f"  CenterAngle:{ca['angle']:.1f}°"
                  if ca and "angle" in ca else "")
        if t == "line":
            L = G.dist(pts_mm[0], pts_mm[1])
            vtag = f" [{m['vessel']}]" if m.get("vessel") else ""
            return f"#{m['id']} Line{vtag}: {L:.1f} {u}"
        if t == "polyline":
            L = sum(G.dist(pts_mm[i], pts_mm[i + 1])
                    for i in range(len(pts_mm) - 1))
            tag = "Polyline (Spline)" if m.get("smooth") else "Polyline"
            return f"#{m['id']} {tag}: {L:.1f} {u}"
        if t == "angle":
            # Vertex is the middle (2nd) click; rays go to the 1st & 3rd.
            ang = G.angle_at(pts_mm[1], pts_mm[0], pts_mm[2])
            return f"#{m['id']} Angle: {ang:.1f}°"
        if t == "ellipse":
            _, _, a_mm, b_mm, _ = G.ellipse_params(pts_mm)
            return (f"#{m['id']} Ellipse  "
                    f"Area:{math.pi*a_mm*b_mm:.1f}{u}²  "
                    f"Dmax:{2*max(a_mm,b_mm):.1f}  "
                    f"Dmin:{2*min(a_mm,b_mm):.1f}{u}{ca_str}")
        # polygon
        area = G.poly_area(pts_mm)
        _, _, dmax, dmin = G.major_minor({"type": "polygon", "pts": pts_mm})
        return (f"#{m['id']} Polygon  Area:{area:.1f}{u}²  "
                f"Dmax:{dmax:.1f}  Dmin:{dmin:.1f}{u}{ca_str}")

    def _handles(self, m):
        return list(m["pts"])

    def _anchor(self, m):
        if m["type"] == "ellipse":
            cx, cy, _a, _b = G.ellipse_cab(m["pts"])
            return (cx, cy)
        if m["type"] == "polygon":
            xs = [q[0] for q in m["pts"]]
            ys = [q[1] for q in m["pts"]]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        if m["type"] == "angle":
            return m["pts"][1]                        # label at the vertex
        return m["pts"][0]

    def _shape_center(self, m) -> tuple[float, float]:
        """Centre of an Ellipse/Polygon — the apex of Center Angle annotations.
        Polygon uses the AREA centroid (physical centre of the region), not the
        vertex mean (which skews toward clustered vertices)."""
        if m["type"] == "polygon":
            return G.polygon_centroid(m["pts"])
        return self._anchor(m)

    # ----------------------------------------------------- label position
    def _label_topleft(self, m) -> QPoint:
        """Widget-pixel top-left where this measure's id/vessel label is
        drawn: a user-dragged offset (``m['label_pos']`` in image coords)
        when present, else the default just past the anchor."""
        pos = m.get("label_pos")
        if pos is not None:
            return self._image_to_widget(pos)
        return self._image_to_widget(self._anchor(m)) + QPoint(8, -8)

    def _pick_label(self, sx, sy):
        """Index of the measure whose drawn label contains widget pixel
        (sx, sy), topmost-first, or None. Uses the rects cached at paint."""
        pt = QPoint(int(sx), int(sy))
        for mi in range(len(self.measures) - 1, -1, -1):
            rect = self._label_rects.get(mi)
            if rect is not None and rect.adjusted(-3, -3, 3, 3).contains(pt):
                return mi
        return None

    # ----------------------------------------------------- hit testing
    def _pick_handle(self, sx, sy):
        """(measure_idx, vertex_idx) under widget pixel (sx, sy)."""
        for mi in range(len(self.measures) - 1, -1, -1):
            m = self.measures[mi]
            for vi, q in enumerate(m["pts"]):
                w = self._image_to_widget(q)
                if math.hypot(w.x() - sx, w.y() - sy) < 12.0:
                    return mi, vi
        return None

    # ------------------------------------------------- area (mm² / px²)
    def _area(self, m) -> float:
        """Area in mm² (or px² when uncalibrated). 0 for non-area shapes."""
        if m["type"] == "ellipse":
            # Measure the axes in mm-space so an oblique ellipse under
            # anisotropic spacing still reports the correct area.
            _, _, a, b, _ = G.ellipse_params([self._to_mm(q) for q in m["pts"]])
            return math.pi * a * b
        if m["type"] == "polygon":
            pts_mm = [self._to_mm(q) for q in m["pts"]]
            return G.poly_area(pts_mm)
        return 0.0

    # -------------------------------------------------- Compare (%PA + gap)
    def set_compare_mode(self, on: bool) -> None:
        """Arm/disarm Compare-select mode (host viewer's Compare button)."""
        self._cmp_on = bool(on)
        self._cmp_sel = []
        self.update()

    def _compare_pick(self, sx, sy) -> None:
        """Compare-mode left-click: toggle the Ellipse/Polygon under the cursor
        in the selection; the 2nd pick opens the analysis dialog and computes."""
        mi = self._pick_measure(sx, sy)
        if mi is not None and self.measures[mi]["type"] in ("ellipse", "polygon"):
            if mi in self._cmp_sel:
                self._cmp_sel.remove(mi)
            elif len(self._cmp_sel) < 2:
                self._cmp_sel.append(mi)
            if len(self._cmp_sel) == 2:
                QTimer.singleShot(0, self._compare_prompt)
        self.update()

    def _compare_prompt(self) -> None:
        if not self._cmp_on or len(self._cmp_sel) != 2:
            return
        dlg = CompareOptionsDialog(self._cmp_want_pa, self._cmp_want_thk, self)
        if not dlg.exec():
            self._cancel_compare()
            return
        self._cmp_want_pa, self._cmp_want_thk = dlg.values()
        if not (self._cmp_want_pa or self._cmp_want_thk):
            self._cancel_compare()
            return
        self._do_compare()

    def _cancel_compare(self) -> None:
        self._cmp_on = False
        self._cmp_sel = []
        self.compare_finished.emit()
        self.update()

    def _do_compare(self) -> None:
        if len(self._cmp_sel) != 2:
            return
        m1, m2 = self.measures[self._cmp_sel[0]], self.measures[self._cmp_sel[1]]
        o1, o2 = self._outline_px(m1), self._outline_px(m2)
        a1, a2 = self._area(m1), self._area(m2)
        if a2 > a1:                          # outer = larger
            m1, m2, o1, o2, a1, a2 = m2, m1, o2, o1, a2, a1
        cen = G.polygon_centroid(o1)
        radials = G.radial_gap_compare(o1, o2, cen, 1.0)
        for r in radials:                    # gap in mm (or px), via calibration
            r["gap"] = G.dist(self._to_mm(r["inner"]), self._to_mm(r["outer"]))
        self._compares.append({
            "big_id": m1["id"], "small_id": m2["id"],
            "pct": G.percent_area_diff(a1, a2), "centroid": cen,
            "outer": o1, "inner": o2, "radials": radials, "step": 1.0,
            "show_pa": self._cmp_want_pa, "show_thk": self._cmp_want_thk,
            "fill_hex": m1.get("color") or "#33e6ff",
        })
        self._cmp_on = False
        self._cmp_sel = []
        self.compare_finished.emit()
        self.results_changed.emit()
        self.update()

    def _recompute_compares(self) -> None:
        """Re-derive each compare result from the CURRENT outlines of the
        measures it references (by id), so editing a shape live-updates its
        comparison; a result whose shape was deleted is dropped."""
        if not self._compares:
            return
        by_id = {m["id"]: m for m in self.measures}
        out = []
        for c in self._compares:
            mb, ms = by_id.get(c["big_id"]), by_id.get(c["small_id"])
            if mb is None or ms is None:
                continue
            ob, os_ = self._outline_px(mb), self._outline_px(ms)
            ab, asm = self._area(mb), self._area(ms)
            if asm > ab:
                mb, ms, ob, os_, ab, asm = ms, mb, os_, ob, asm, ab
            cen = G.polygon_centroid(ob)
            radials = G.radial_gap_compare(ob, os_, cen, c.get("step", 1.0))
            for r in radials:
                r["gap"] = G.dist(self._to_mm(r["inner"]), self._to_mm(r["outer"]))
            c = dict(c)
            c.update({
                "big_id": mb["id"], "small_id": ms["id"],
                "outer": ob, "inner": os_, "centroid": cen,
                "pct": G.percent_area_diff(ab, asm), "radials": radials,
            })
            # Keep a user-chosen fill colour; otherwise track the outer outline.
            if not c.get("fill_custom"):
                c["fill_hex"] = mb.get("color") or "#33e6ff"
            out.append(c)
        self._compares = out

    def _compare_hit(self, sx, sy):
        """Index of the compare result whose filled region (inside outer, outside
        inner) contains widget point (sx,sy) — topmost first — else None."""
        pt = self._widget_to_image(QPoint(int(sx), int(sy)))
        if pt is None:
            return None
        for i in range(len(self._compares) - 1, -1, -1):
            c = self._compares[i]
            if (G.point_in_poly(pt[0], pt[1], c["outer"])
                    and not G.point_in_poly(pt[0], pt[1], c["inner"])):
                return i
        return None

    def _compare_delete_menu(self, target) -> None:
        """Right-click INSIDE a compare region → Hide·Show / Delete. Hide toggles
        only the region COLOUR fill; the defining outlines are separate measures
        and are unaffected."""
        if target not in self._compares:
            return
        menu = QMenu(self)
        # The %Area / Thickness fill colour is independently selectable here
        # (separate from the defining outlines' colours). Once set it sticks —
        # _recompute_compares preserves it via the "fill_custom" flag.
        color_actions = add_color_submenu(menu, COLOR_CHOICES)
        transp_actions = add_transparency_submenu(menu, target.get("transp", 50))
        vis_act = menu.addAction("Show" if target.get("hidden") else "Hide")
        del_act = menu.addAction("Delete")
        chosen = menu.exec(QCursor.pos())
        if chosen is vis_act:
            target["hidden"] = not target.get("hidden", False)
            self.update()
        elif chosen is del_act:
            try:
                self._compares.remove(target)
            except ValueError:
                pass
            self.update()
            self.results_changed.emit()
        else:
            for act, hexcol in color_actions:
                if chosen is act:
                    target["fill_hex"] = hexcol
                    target["fill_custom"] = True
                    self.update()
                    break
            for act, val in transp_actions:
                if chosen is act:
                    target["transp"] = val
                    self.update()
                    break

    # ----------------------------------------------- right-click menus
    def _ca_hit(self, sx, sy):
        """mi of a measure whose Center-Angle marker point OR spoke line is
        under the widget point (sx,sy), else None — for 'Delete Center Angle'."""
        for mi in range(len(self.measures) - 1, -1, -1):
            ca = self.measures[mi].get("center_angle")
            if not ca or not ca.get("pts"):
                continue
            for q in ca["pts"]:
                wq = self._image_to_widget(q)
                if (wq.x() - sx) ** 2 + (wq.y() - sy) ** 2 <= 144.0:   # ≤12px
                    return mi
            wc = self._image_to_widget(self._shape_center(self.measures[mi]))
            for q in ca["pts"]:
                wq = self._image_to_widget(q)
                if G.seg_dist(sx, sy, (wc.x(), wc.y()), (wq.x(), wq.y())) < 6.0:
                    return mi
        return None

    def _export_menu(self, sx, sy):
        """Right-click on empty image area (no active tool): pick a format,
        then let the viewer (which holds the patient/series/frame context)
        capture + save in response to export_requested."""
        if self._qimg is None:
            return
        # The cross-section canvas (XA / IVUS) is a real DICOM cine, so it
        # offers DICOM + MP4 in addition to the still formats + CSV.
        key = pick_export_format(
            self, self.mapToGlobal(QPoint(int(sx), int(sy))),
            include_dicom=True, include_mp4=True, include_anon=True,
            include_wl=True, wl_enabled=self.wl_enabled,
        )
        if key == "wl":
            self.wl_change_requested.emit()
        elif key:
            self.export_requested.emit(key)

    def _handle_menu(self, hit, sx, sy):
        mi, vi = hit
        m = self.measures[mi]
        menu = QMenu(self)
        del_pt = del_res = None
        if m["type"] in ("polyline", "polygon"):
            del_pt = menu.addAction("Delete point")
            # Don't let the user shrink below 2 vertices (Line minimum).
            if len(m["pts"]) <= 2:
                del_pt.setEnabled(False)
        # Change Color / Change Transparency — available on every result type
        # (incl. Line/Angle, which are most easily right-clicked on a handle).
        color_actions = add_color_submenu(menu, COLOR_CHOICES)
        transp_actions = add_transparency_submenu(menu, m.get("transp", 0))
        hide_act = menu.addAction("Show" if m.get("hidden") else "Hide")
        if m["type"] in ("polyline", "polygon"):
            del_res = menu.addAction("Delete result")
        else:
            del_res = menu.addAction("Delete")
        chosen = menu.exec(self.mapToGlobal(QPoint(int(sx), int(sy))))
        if del_pt is not None and chosen is del_pt:
            self._delete_point(mi, vi)
        elif chosen is hide_act:
            m["hidden"] = not m.get("hidden", False)   # hide THIS line only
        elif chosen is del_res:
            del self.measures[mi]
            self._recompute_compares()      # drop/refresh affected comparisons
            self.results_changed.emit()
        else:
            for act, hexcol in color_actions:
                if chosen is act:
                    m["color"] = hexcol
                    break
            for act, val in transp_actions:
                if chosen is act:
                    m["transp"] = val
                    break
        self._recompute_compares()          # a colour change refreshes any compare
        self.update()

    def _outline_menu(self, mi, sx, sy):
        m = self.measures[mi]
        menu = QMenu(self)
        add_pt = menu.addAction("Add point")
        spline_act = None
        if m["type"] == "polyline":
            spline_act = menu.addAction(
                "UnSpline" if m.get("smooth") else "Spline"
            )
        center_angle_act = None
        if m["type"] in ("ellipse", "polygon"):
            center_angle_act = menu.addAction("Center Angle")
        # Vessel type — tag a Line as the Guiding Catheter or a coronary
        # proximal segment, so the Coaxial-Eval tool can pick it up. Only
        # meaningful for straight lines; a "(none)" entry clears the tag.
        vessel_actions: list[tuple] = []
        if m["type"] == "line":
            vessel_menu = menu.addMenu("Vessel type")
            cur = m.get("vessel")
            none_act = vessel_menu.addAction("(none)")
            none_act.setCheckable(True)
            none_act.setChecked(cur is None)
            vessel_actions.append((none_act, None))
            vessel_menu.addSeparator()
            for label in VESSEL_LABELS:
                a = vessel_menu.addAction(label)
                a.setCheckable(True)
                a.setChecked(cur == label)
                vessel_actions.append((a, label))
        # Change Color — 16-colour submenu (each item carries a swatch).
        color_menu = menu.addMenu("Change Color")
        color_actions: list[tuple] = []
        for name, hexcol in COLOR_CHOICES:
            a = color_menu.addAction(name)
            pix = QPixmap(16, 16); pix.fill(QColor(hexcol))
            a.setIcon(QIcon(pix))
            color_actions.append((a, hexcol))
        transp_actions = add_transparency_submenu(menu, m.get("transp", 0))
        hide_act = menu.addAction("Show" if m.get("hidden") else "Hide")
        del_act = menu.addAction("Delete")
        chosen = menu.exec(self.mapToGlobal(QPoint(int(sx), int(sy))))
        if chosen is add_pt:
            self._add_point_at(mi, sx, sy)
        elif spline_act is not None and chosen is spline_act:
            m["smooth"] = not m.get("smooth", False)
        elif center_angle_act is not None and chosen is center_angle_act:
            self._center_angle_target = mi
            m.pop("center_angle", None)              # restart picking
        elif chosen is hide_act:
            m["hidden"] = not m.get("hidden", False)   # hide THIS line only
        elif chosen is del_act:
            del self.measures[mi]
            self._recompute_compares()      # drop/refresh affected comparisons
            self.results_changed.emit()
        else:
            for act, label in vessel_actions:
                if chosen is act:
                    if label is None:
                        m.pop("vessel", None)
                    else:
                        m["vessel"] = label
                    break
            else:
                for act, hexcol in color_actions:
                    if chosen is act:
                        m["color"] = hexcol
                        break
                for act, val in transp_actions:
                    if chosen is act:
                        m["transp"] = val
                        break
        # A colour / spline / vertex change to a shape that is part of a
        # comparison must refresh that comparison NOW (fill colour + outline),
        # not on some later action.
        self._recompute_compares()
        self.update()

    def _center_angle_add(self, pt) -> None:
        """One left-click while in Center-Angle pick mode adds a
        perimeter point to the targeted measure; the 3rd click finalises
        the arc angle."""
        mi = self._center_angle_target
        if not (0 <= mi < len(self.measures)):
            self._center_angle_target = -1
            return
        m = self.measures[mi]
        ca = m.setdefault("center_angle", {"pts": []})
        ca["pts"].append(self._snap_ca(m, pt))      # constrain to the outline
        if len(ca["pts"]) >= 3:
            centre = self._shape_center(m)
            # Click order is [endpoint, other endpoint, arc selector] — matching
            # Rupture-Predictor. The geometry helper wants (p1, through, p3), so
            # the selector (pts[2]) is passed as the middle "through" point.
            e1, e2, sel = ca["pts"][:3]
            span, t1, t3, ccw = G.central_arc_angle(centre, e1, sel, e2)
            m["center_angle"] = {
                "pts": [e1, e2, sel], "angle": span,
                "t1": t1, "t3": t3, "ccw": ccw,
            }
            self._center_angle_target = -1

    def _snap_ca(self, m, pt):
        """Constrain a Center-Angle marker to the measure's drawn outline, so
        the points always sit ON the polygon/ellipse line (matching CT)."""
        return G.project_to_polyline(pt, self._outline_px(m))

    def _pick_center_angle(self, sx, sy):
        """Pick a finalized Center-Angle marker point so it can be dragged like
        a polygon vertex. Returns (mi, ci) or None."""
        for mi in range(len(self.measures) - 1, -1, -1):
            ca = self.measures[mi].get("center_angle")
            if not ca or not ca.get("pts"):
                continue
            for ci, q in enumerate(ca["pts"]):
                wq = self._image_to_widget(q)
                if (wq.x() - sx) ** 2 + (wq.y() - sy) ** 2 <= 144.0:   # ≤12px
                    return mi, ci
        return None

    def _set_center_angle_point(self, m, ci, w):
        """Move one CA marker (snapped to the outline) and recompute the angle
        from the shape centre, mirroring the CT viewers' live update."""
        ca = m.get("center_angle")
        if not ca or "pts" not in ca or not (0 <= ci < len(ca["pts"])):
            return
        pts = list(ca["pts"])
        pts[ci] = self._snap_ca(m, w)
        centre = self._shape_center(m)
        # pts == [endpoint, other endpoint, arc selector]; selector is the
        # "through" point for the geometry helper (see _center_angle_add).
        span, t1, t3, ccw = G.central_arc_angle(centre, pts[0], pts[2], pts[1])
        m["center_angle"] = {"pts": pts, "angle": span,
                             "t1": t1, "t3": t3, "ccw": ccw}

    def _resnap_center_angle(self, m):
        """After the shape changes (vertex/handle drag, add/delete point), pull
        each CA marker back onto the new outline and recompute the angle."""
        ca = m.get("center_angle")
        if not ca or len(ca.get("pts", [])) < 3:
            return
        pts = [self._snap_ca(m, q) for q in ca["pts"]]
        centre = self._shape_center(m)
        # pts == [endpoint, other endpoint, arc selector]; selector is the
        # "through" point for the geometry helper (see _center_angle_add).
        span, t1, t3, ccw = G.central_arc_angle(centre, pts[0], pts[2], pts[1])
        m["center_angle"] = {"pts": pts, "angle": span,
                             "t1": t1, "t3": t3, "ccw": ccw}

    # ----------------------------------------------- point add/delete
    def _add_point_at(self, mi: int, sx: float, sy: float) -> None:
        """Insert a new editable vertex at the click on measure mi.
        Line→Polyline, Ellipse→Polygon (per user spec)."""
        m = self.measures[mi]
        pt = self._widget_to_image(QPoint(int(sx), int(sy)))
        if pt is None:
            return
        # Ellipse first becomes a 4-vertex Polygon at the axis endpoints,
        # in order around the perimeter (maj0 → min0 → maj1 → min1).
        if m["type"] == "ellipse":
            maj0, maj1, min0, min1 = m["pts"]
            m["type"] = "polygon"
            m["pts"] = [maj0, min0, maj1, min1]
        if m["type"] == "line":
            m["type"] = "polyline"
            m["pts"] = [m["pts"][0], pt, m["pts"][1]]
            return
        # Polyline or Polygon: insert at the closest raw segment.
        pts = list(m["pts"])
        n = len(pts)
        closed = (m["type"] == "polygon")
        edges = n if closed else n - 1
        best_i, best_d = 0, float("inf")
        for i in range(edges):
            a, b = pts[i], pts[(i + 1) % n]
            d = G.seg_dist(pt[0], pt[1], a, b)
            if d < best_d:
                best_d, best_i = d, i
        pts.insert(best_i + 1, pt)
        m["pts"] = pts
        self._resnap_center_angle(m)

    def _delete_point(self, mi: int, vi: int) -> None:
        """Remove one vertex from a Polyline/Polygon. Going from 3→2
        verts converts the shape to a Line (user-spec: 'delete until it
        becomes a Line')."""
        m = self.measures[mi]
        if m["type"] not in ("polyline", "polygon"):
            return
        pts = list(m["pts"])
        if len(pts) <= 2 or not (0 <= vi < len(pts)):
            return
        del pts[vi]
        if len(pts) == 2:
            m["type"] = "line"
        m["pts"] = pts
        self._resnap_center_angle(m)

    def _pick_measure(self, sx, sy):
        """Index of the measure whose outline is closest to (sx, sy)
        in widget pixels (within tol), or None."""
        best, bi = 10.0, None
        for mi, m in enumerate(self.measures):
            ol = self._outline_px(m)
            wpts = [self._image_to_widget(q) for q in ol]
            for i in range(len(wpts) - 1):
                a = (wpts[i].x(), wpts[i].y())
                b = (wpts[i + 1].x(), wpts[i + 1].y())
                d = G.seg_dist(sx, sy, a, b)
                if d < best:
                    best, bi = d, mi
        return bi

    # ----------------------------------------------------- mouse
    def keyPressEvent(self, e):
        """Arrow keys drive cine navigation when the image is the focused
        widget. Up/Down → Prev/Next series, Left/Right → prev/next frame (via
        arrow_pressed → the viewer). Anything else bubbles up unchanged."""
        direction = {
            Qt.Key.Key_Up: "up",
            Qt.Key.Key_Down: "down",
            Qt.Key.Key_Left: "left",
            Qt.Key.Key_Right: "right",
        }.get(e.key())
        if direction is not None:
            self.arrow_pressed.emit(direction)
            e.accept()
            return
        super().keyPressEvent(e)

    def mousePressEvent(self, e):
        # Take keyboard focus on click so arrow keys drive cine nav (this
        # override doesn't call the base mousePressEvent, which is what would
        # normally focus a ClickFocus widget, so do it explicitly).
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        pos = e.position()
        sx, sy = pos.x(), pos.y()
        # Middle-button drag pans the (zoomed) image — independent of the
        # active measurement tool, so panning never competes with drawing.
        # Ctrl + left-drag pans too, so a mouse without a middle button (or
        # a trackpad) can still move the image. Checked before the zoom /
        # measure handlers so panning always wins regardless of the tool.
        is_ctrl_pan = (
            e.button() == Qt.MouseButton.LeftButton
            and bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
        )
        if e.button() == Qt.MouseButton.MiddleButton or is_ctrl_pan:
            self._panning = True
            self._pan_anchor = (sx, sy)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        # Click-to-zoom mode (toolbar magnifier buttons): a left-click
        # zooms about the clicked point and consumes the click so it never
        # starts a measurement. Repeated clicks zoom progressively.
        if (e.button() == Qt.MouseButton.LeftButton
                and self._zoom_click in ("in", "out")):
            self._apply_zoom(1.1 if self._zoom_click == "in" else 0.9, sx, sy)
            return
        # IVUS long-axis rotation-centre marker takes priority over
        # everything else when the feature is on AND the click landed
        # near the marker. Left = start dragging it; right = Reset
        # frame / Reset All menu.
        if self.ivus_show_center:
            # Left-drag grabs the centre marker (any frame) to move it.
            if (e.button() == Qt.MouseButton.LeftButton
                    and self._hit_ivus_center(sx, sy)):
                self._ivus_dragging_center = True
                return
            # Right-click offers the Remove menu ONLY on a moved (red/keyed)
            # centre, within an 18 px band. A blue (interpolated) centre has
            # no menu — the click falls through to normal handling.
            if (e.button() == Qt.MouseButton.RightButton
                    and self.ivus_center_keyed
                    and self._hit_ivus_center(sx, sy, 18.0)):
                self._ivus_center_menu(sx, sy)
                return
        # Grab the yellow cut line (away from the centre) to re-angle the
        # long-axis plane — same priority as the centre marker so it wins
        # over starting a measurement on the line.
        if (self.ivus_show_center
                and e.button() == Qt.MouseButton.LeftButton
                and self._hit_ivus_cutline(sx, sy)):
            self._ivus_dragging_angle = True
            self._set_cutline_from_widget(sx, sy)
            return
        if e.button() == Qt.MouseButton.RightButton:
            # Cancel an in-progress Center-Angle pick on right-click.
            if self._center_angle_target >= 0:
                mi = self._center_angle_target
                if 0 <= mi < len(self.measures):
                    self.measures[mi].pop("center_angle", None)
                self._center_angle_target = -1
                self.update()
                return
            # Finish a polyline/polygon draft (≥2 pts) first.
            if self._draft and self._draft["type"] in ("polyline", "polygon") \
                    and len(self._draft["pts"]) >= 2:
                self._commit_draft()
                return
            # Right-click ON a Center-Angle marker or spoke deletes JUST the
            # Center Angle (the shape stays) — checked before the handle/outline
            # menus, matching the CT viewers.
            ca_mi = self._ca_hit(sx, sy)
            if ca_mi is not None:
                menu = QMenu(self)
                del_ca = menu.addAction("Delete Center Angle")
                chosen = menu.exec(self.mapToGlobal(QPoint(int(sx), int(sy))))
                if chosen is del_ca:
                    self.measures[ca_mi].pop("center_angle", None)
                    self.update()
                return
            # Handle has priority over outline (more specific target).
            hit = self._pick_handle(sx, sy)
            if hit is not None:
                self._handle_menu(hit, sx, sy)
                return
            # A measure LINE/outline takes priority next (its own Hide/Delete);
            # only an EMPTY spot inside a compare region falls through to the
            # region's colour Hide/Delete menu.
            mi = self._pick_measure(sx, sy)
            if mi is not None:
                self._outline_menu(mi, sx, sy)
                return
            ci = self._compare_hit(sx, sy)
            if ci is not None:
                self._compare_delete_menu(self._compares[ci])
                return
            # Empty area + no active measure tool → offer still export.
            # (In measure mode an empty right-click stays a no-op so it
            # can't interrupt drawing.)
            if not self.meas_type:
                self._export_menu(sx, sy)
            return

        if e.button() != Qt.MouseButton.LeftButton:
            return

        # Compare-select mode: a left-click picks the two shapes to compare.
        if self._cmp_on:
            self._compare_pick(sx, sy)
            return

        # Center-Angle pick mode consumes left-clicks until 3 perimeter
        # points have been added to the targeted Ellipse/Polygon.
        if self._center_angle_target >= 0:
            pt = self._widget_to_image(QPoint(int(sx), int(sy)))
            if pt is not None:
                self._center_angle_add(pt)
                self.update()
            return

        # Handle drag has priority — even when a tool is selected, so the
        # user can re-edit a previous shape.
        hit = self._pick_handle(sx, sy)
        if hit is not None:
            self._edit = {"mi": hit[0], "vi": hit[1]}
            self.update()                       # turn the handle green now
            return
        # A Center-Angle marker drags like a polygon vertex (snapped to outline).
        ca_hit = self._pick_center_angle(sx, sy)
        if ca_hit is not None:
            self._edit = {"mi": ca_hit[0], "vi": ca_hit[1], "ca": True}
            self.update()
            return

        # Label drag — pull an id/vessel label off an overlapping neighbour.
        # After handles (more specific) but before drawing, and works with no
        # tool selected so labels are always repositionable.
        li = self._pick_label(sx, sy)
        if li is not None:
            self._drag_label = li
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if not self.meas_type:
            return
        pt = self._widget_to_image(QPoint(int(sx), int(sy)))
        if pt is None:
            return
        d = self._draft
        if d is None or d["type"] != self.meas_type:
            d = {"type": self.meas_type, "pts": []}
            self._draft = d
        d["pts"].append(pt)
        if d["type"] in ("line", "ellipse") and len(d["pts"]) >= 2:
            self._commit_draft()
        elif d["type"] == "angle" and len(d["pts"]) >= 3:
            self._commit_draft()
        else:
            self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        sx, sy = pos.x(), pos.y()
        if self._panning and self._pan_anchor is not None:
            self._pan[0] += sx - self._pan_anchor[0]
            self._pan[1] += sy - self._pan_anchor[1]
            self._pan_anchor = (sx, sy)
            self._clamp_pan()
            self.update()
            return
        if self._ivus_dragging_center:
            pt = self._widget_to_image(QPoint(int(sx), int(sy)))
            if pt is not None:
                w, h = self._img_size
                cx = max(1.0, min(float(pt[0]), float(w - 1)))
                cy = max(1.0, min(float(pt[1]), float(h - 1)))
                self.ivus_center_image = (cx, cy)
                self.ivus_center_changed.emit(cx, cy)
                self.update()
            return
        if self._ivus_dragging_angle:
            self._set_cutline_from_widget(sx, sy)
            return
        if self._drag_label >= 0:
            # Store the new top-left in IMAGE coords so it tracks the image
            # through zoom/pan. Unbounded map so the label can be parked in
            # the letterbox margin too.
            if 0 <= self._drag_label < len(self.measures):
                self.measures[self._drag_label]["label_pos"] = (
                    self._widget_to_image_f(sx, sy)
                )
            self.update()
            return
        if self._edit is not None:
            pt = self._widget_to_image(QPoint(int(sx), int(sy)))
            if pt is None:
                return
            m = self.measures[self._edit["mi"]]
            vi = self._edit["vi"]
            if self._edit.get("ca"):
                self._set_center_angle_point(m, vi, pt)
            elif m["type"] == "ellipse":
                self._set_ellipse_handle(m, vi, pt)
                self._resnap_center_angle(m)
            else:
                m["pts"][vi] = pt
                self._resnap_center_angle(m)    # CA follows the reshaped polygon
            self._recompute_compares()          # live-update any comparison
            self.update()
            return
        if self._draft is not None:
            self._hover = self._widget_to_image(QPoint(int(sx), int(sy)))
            self.update()

    def mouseReleaseEvent(self, _e):
        if self._panning:
            self._panning = False
            self._pan_anchor = None
            self.setCursor(
                Qt.CursorShape.CrossCursor if self.meas_type
                else Qt.CursorShape.ArrowCursor
            )
            return
        if self._ivus_dragging_center:
            self._ivus_dragging_center = False
            return
        if self._ivus_dragging_angle:
            self._ivus_dragging_angle = False
            self.ivus_angle_finished.emit()
            return
        if self._drag_label >= 0:
            self._drag_label = -1
            self.setCursor(
                Qt.CursorShape.CrossCursor if self.meas_type
                else Qt.CursorShape.ArrowCursor
            )
            self.update()
            return
        if self._edit is not None:
            self._edit = None
            self.update()

    def _ivus_center_menu(self, sx: float, sy: float) -> None:
        """Remove this point / Remove all points — right-click on a moved
        (red/keyed) IVUS centre marker. "this point" discards just this
        frame's manual centre (it reverts to the value interpolated from the
        other manual centres); "all points" discards every manual centre.
        The actual edit is performed by the viewer (single source of truth
        for the per-frame centres array)."""
        menu = QMenu(self)
        a_one = menu.addAction("Remove this point")
        a_all = menu.addAction("Remove all points")
        chosen = menu.exec(self.mapToGlobal(QPoint(int(sx), int(sy))))
        if chosen is a_one:
            self.ivus_center_reset.emit("frame")
        elif chosen is a_all:
            self.ivus_center_reset.emit("all")

    def mouseDoubleClickEvent(self, _e):
        if self._draft and self._draft["type"] in ("polyline", "polygon") \
                and len(self._draft["pts"]) >= 2:
            self._commit_draft()

    # ----------------------------------------------------- mutation helpers
    def _set_ellipse_handle(self, m, vi, w):
        # Oblique-ellipse handle drag via the shared pure helper (same logic
        # used by both CT viewers): a major endpoint resizes + rotates keeping
        # the minor width; a minor endpoint sets the minor width perpendicular
        # to the major axis.
        m["pts"] = G.ellipse_drag(m["pts"], vi, w)

    def _commit_draft(self):
        d = self._draft
        self._draft = None
        self._hover = None
        if d is None or len(d["pts"]) < 2:
            self.update()
            return
        if d["type"] == "ellipse":
            # First two clicks are the MAJOR-axis endpoints (an oblique drag
            # makes an oblique ellipse); the minor radius starts at half the
            # major and is tuned afterwards via the minor handles.
            pts = G.ellipse_from_major(d["pts"][0], d["pts"][1])
        elif d["type"] == "line":
            pts = d["pts"][:2]
        else:
            pts = list(d["pts"])
        self._meas_seq += 1
        m = {"id": self._meas_seq, "type": d["type"], "pts": pts}
        # Stamp the view's C-arm angles so Coaxial-Eval reads the angle the
        # line was actually drawn at, not whatever plane happens to be shown
        # later (matters for biplane in single-plane mode).
        if self.view_angles is not None:
            m["beta"], m["alpha"] = self.view_angles
        self.measures.append(m)
        # Shell history: emit a Measurement with the metrics string as
        # label and the tool name as kind ("Line"/"Polyline"/etc.).
        meas = Measurement(
            kind=_PRETTY.get(m["type"], m["type"]),
            points=list(m["pts"]),
            spacing_mm=self.spacing_mm,
        )
        meas.text = self._metrics(m)
        # Defer the emit + repaint to the next event-loop tick so we never
        # repaint synchronously inside this mousePressEvent — a known
        # crash trigger on Windows / OpenGL when right-click also fires a
        # contextMenuEvent and the two events collide on the same widget.
        def _publish(self=self, meas=meas):
            self.measurement_done.emit(meas)
            self.results_changed.emit()
            self.update()
        QTimer.singleShot(0, _publish)

    # ---------------------------------------------------------------- paint
    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0a0a0a"))
        if self._qimg is None:
            return
        r = self._draw_rect()
        # Smooth (bilinear) upscaling — without this hint Qt uses fast
        # nearest-neighbour, which makes burned-in annotations (the "Frame N"
        # text, crosshairs) blocky/jagged when the 512² IVUS/XA frame is
        # scaled up to fill the canvas, while the noisy echo texture hides the
        # artefact. The echo DATA is bit-identical either way (≤1 LSB); this is
        # purely how the already-decoded frame is resampled to the widget.
        # During playback/seek the viewer sets _fast_scale so the per-frame
        # bilinear cost (heavy on a high-DPI Mac) doesn't stutter the cine; a
        # paused/still frame keeps the smooth upscale.
        if not self._fast_scale:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(r, self._qimg)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Rebuilt as labels are drawn below; clear so a deleted measure's
        # stale rect can't linger for the next drag hit-test.
        self._label_rects = {}

        # Major/minor axes (drawn UNDER outlines): thick dotted, in the
        # polygon-vertex colour (yellow) so the long/short-diameter lines read
        # as part of the shape — matching the CT viewers.
        for m in self.measures:
            if self._results_hidden or m.get("hidden"):
                continue
            try:
                segs = self._axis_segs_px(m)
            except Exception:
                continue
            axis_pen = QPen(QColor(255, 217, 0, 170), 2.4, Qt.PenStyle.DashLine)
            for seg in segs:
                if seg is None:
                    continue
                p.setPen(axis_pen)
                a = self._image_to_widget(seg[0])
                b = self._image_to_widget(seg[1])
                p.drawLine(a, b)

        # Outlines — per-measure colour (Change Color menu).
        for m in self.measures:
            if self._results_hidden or m.get("hidden"):
                continue
            try:
                outline = self._outline_px(m)
            except Exception:
                continue
            col = QColor(m.get("color", DEFAULT_MEAS_COLOR))
            col.setAlpha(transp_to_alpha(m.get("transp", 0)))   # Change Transparency
            p.setPen(QPen(col, 1.92))    # 1.6 ×1.2 — readability
            wpts = [self._image_to_widget(q) for q in outline]
            for a, b in zip(wpts, wpts[1:]):
                p.drawLine(a, b)

        # Center-Angle annotations (3 spokes from shape centre + small
        # dots on the picked perimeter points).
        for mi, m in enumerate(self.measures):
            if self._results_hidden or m.get("hidden"):
                continue
            ca = m.get("center_angle")
            if not ca or "pts" not in ca:
                continue
            centre = self._shape_center(m)
            wc = self._image_to_widget(centre)
            # Spokes in the CA-marker colour (orange) so the spokes + markers
            # read as one deletable unit — matching the CT viewers.
            spoke = QColor(255, 140, 0, 200)
            # CA spokes were anomalously thin (1.2); bump to the diameter-line
            # weight (2.4) so they're as visible as the rest of the measurement.
            p.setPen(QPen(spoke, 2.4, Qt.PenStyle.DashLine))
            for ci, q in enumerate(ca["pts"]):
                # pts == [endpoint, other endpoint, arc selector]. The 3rd point
                # only picks which arc is measured, so it gets no spoke — only
                # the two endpoints (ci 0 and 1) form the angle's arms.
                if ci == 2:
                    continue
                wq = self._image_to_widget(q)
                p.drawLine(wc, wq)
            # Solid orange arc on the outline between the two endpoints, passing
            # through the selector — only shown once all 3 points are placed.
            if "angle" in ca and len(ca["pts"]) >= 3:
                arc = G.arc_through(self._outline_px(m), ca["pts"][0],
                                    ca["pts"][2], ca["pts"][1])
                if len(arc) >= 2:
                    p.setPen(QPen(QColor(255, 140, 0), 2.88))   # 2.4 ×1.2
                    warc = [self._image_to_widget(q) for q in arc]
                    for a, b in zip(warc, warc[1:]):
                        p.drawLine(a, b)
            # Orange markers; the one being dragged turns green (like a vertex).
            ca_edit_vi = (self._edit["vi"] if (
                self._edit is not None and self._edit.get("ca")
                and self._edit["mi"] == mi) else -1)
            for ci, q in enumerate(ca["pts"]):
                wq = self._image_to_widget(q)
                if ci == ca_edit_vi:
                    p.setPen(QPen(QColor(60, 220, 90), 1.0))
                    p.setBrush(QColor(60, 220, 90))
                    p.drawEllipse(wq, 6, 6)
                else:
                    p.setPen(QPen(QColor(255, 140, 0), 1.0))
                    p.setBrush(QColor(255, 140, 0))
                    p.drawEllipse(wq, 4, 4)
            p.setBrush(Qt.BrushStyle.NoBrush)
            if "angle" in ca:
                p.setPen(QColor(255, 140, 0))
                p.drawText(wc + QPoint(10, -10), f"{ca['angle']:.1f}°")

        # Vertex handles + running-number labels.
        # The single handle currently being dragged turns green and
        # 1.5× larger so the edit state is unmistakable.
        edit_ca = bool(self._edit.get("ca")) if self._edit is not None else False
        edit_mi = self._edit["mi"] if (self._edit is not None
                                       and not edit_ca) else -1
        edit_vi = self._edit["vi"] if (self._edit is not None
                                       and not edit_ca) else -1
        yellow = QColor(255, 217, 0)
        green = QColor(60, 220, 90)
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        p.setFont(font)
        for mi, m in enumerate(self.measures):
            if self._results_hidden or m.get("hidden"):
                continue
            for vi, q in enumerate(self._handles(m)):
                is_edit = (mi == edit_mi and vi == edit_vi)
                col = green if is_edit else yellow
                p.setPen(QPen(col, 1.0))
                p.setBrush(col)
                w = self._image_to_widget(q)
                radius = 8 if is_edit else 5  # edit handle ~1.5× larger
                p.drawEllipse(w, radius, radius)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(yellow)
            tag = str(m["id"])
            if m.get("vessel"):
                tag += f" [{m['vessel']}]"
            # Draw at the dragged position when set, else past the anchor;
            # cache the rect so _pick_label can hit-test the drag.
            tl = self._label_topleft(m)
            fm = p.fontMetrics()
            self._label_rects[mi] = QRect(
                tl.x(), tl.y() - fm.ascent(),
                fm.horizontalAdvance(tag), fm.height(),
            )
            p.drawText(tl, tag)

        # Center-Angle pick-mode hint — order matches Rupture-Predictor:
        # endpoint, other endpoint, then a point on the arc to measure.
        if self._center_angle_target >= 0:
            picked = len(self.measures[self._center_angle_target]
                         .get("center_angle", {}).get("pts", []))
            step = (
                "click the 1st endpoint on the perimeter",
                "click the 2nd endpoint on the perimeter",
                "click a point on the arc you want to measure",
            )[min(picked, 2)]
            self._paint_hint(p, f"Center Angle: {step}")

        # Draft preview (yellow dashed).
        if self._draft and self._draft["pts"]:
            d = self._draft
            pen = QPen(QColor("#f4d03f"), 1.2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            preview_pts = list(d["pts"])
            if self._hover is not None and d["type"] != "ellipse":
                preview_pts.append(self._hover)
            wpts = [self._image_to_widget(q) for q in preview_pts]
            if d["type"] == "ellipse" and self._hover is not None:
                # Preview the oblique ellipse whose major axis is the drag.
                prev = G.ellipse_from_major(d["pts"][0], self._hover)
                wpts = [self._image_to_widget(q)
                        for q in G.ellipse_outline(prev)]
            for a, b in zip(wpts, wpts[1:]):
                p.drawLine(a, b)
            p.setBrush(QColor("#f4d03f"))
            p.setPen(QPen(QColor("#f4d03f"), 1.0))
            for q in d["pts"]:
                w = self._image_to_widget(q)
                p.drawEllipse(w, 3, 3)

        # Compare overlay (filled regions + radial gap map + selection / hint).
        self._paint_compare(p)

        # Result lines (top-right, beneath any overlay would be busy).
        self._paint_results(p)

        # DICOM-tag overlay (top-left, existing behaviour).
        if self.overlay_lines:
            self._paint_overlay(p)

        # IVUS long-axis guide (cut line + projection triangles + rotation-
        # centre marker). Drawn last so it sits on top of overlays.
        if (self.ivus_show_center
                and self.ivus_center_image is not None
                and self._img_size != (0, 0)):
            self._paint_ivus_long_axis(p)

    def _paint_ivus_long_axis(self, p: QPainter) -> None:
        """Draw the IVUS long-axis guide on the cross-section, MPR-style:

        * a dashed yellow *cut line* through the rotation centre at the
          current long-axis angle — the diameter that becomes the strip;
        * two cyan *projection-direction triangles* on the cut line,
          pointing the way the long-axis is "viewed from". By construction
          the strip's TOP is 90° counter-clockwise from the triangle
          direction and its BOTTOM 90° clockwise (visual sense, y-down),
          so the triangles tell the reader which wall is up/down;
        * the rotation-centre dot (red = keyed/fixed, blue = interpolated/
          movable), with a white crosshair for visibility on grey IVUS.
        """
        cx_i, cy_i = self.ivus_center_image
        a = float(self.ivus_la_angle)
        w_img, h_img = self._img_size
        half = math.hypot(w_img, h_img) / 2.0          # full lateral extent
        ux, uy = math.cos(a), math.sin(a)              # cut line: top↔bottom
        wt = self._image_to_widget_f((cx_i - half * ux, cy_i - half * uy))
        wb = self._image_to_widget_f((cx_i + half * ux, cy_i + half * uy))
        wc = self._image_to_widget_f((cx_i, cy_i))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Cut line — same yellow as the strip's axis line.
        p.setPen(QPen(QColor(255, 220, 80, 170), 1, Qt.PenStyle.DashLine))
        p.drawLine(int(round(wt[0])), int(round(wt[1])),
                   int(round(wb[0])), int(round(wb[1])))
        # Projection direction = long-axis angle − 90° (image coords); the
        # image→widget map is a uniform positive scale, so the same unit
        # vector orients the triangles in widget space.
        pdx, pdy = math.sin(a), -math.cos(a)
        # Place the two triangles 1.5 メモリ (1.5 mm) out from the centre along
        # the cut line — computed in IMAGE px (via PixelSpacing) and mapped to
        # widget so the marks ride the 1.5 mm ring through zoom/pan. Falls back
        # to a fraction of the frame on an uncalibrated series.
        sx_mm, sy_mm = self._mm_scale()
        denom = math.hypot(ux * sx_mm, uy * sy_mm)
        if self.spacing_mm is not None and denom > 1e-6:
            off_img = 1.5 / denom
        else:
            off_img = 0.08 * min(w_img, h_img)
        # Triangle size: ×1.25 the original (ts 8 → 10), aspect kept. The
        # APEX sits on the cut line at the 1.5 mm point and the body hangs on
        # the −proj side, so the apex *touches* the yellow line (mirroring the
        # long-axis cursor's arrowheads, whose apex touches the section line).
        ts = 10.0
        height = ts * 1.6          # apex→base distance, along proj
        half_w = ts                # base half-width, along the cut line
        # Colour the two triangles by orientation so up/down reads at a glance
        # and matches the long-axis cursor (cyan = strip top, dark red = strip
        # bottom). For the triangle offset by s·(ux,uy) from the centre, the
        # base→apex vector (= proj) is visually counter-clockwise from the
        # centre→triangle vector when (pos × proj) < 0 (screen y-down). That
        # cross product is s·(ux·pdy − uy·pdx) = −s, so s = +1 is the CCW
        # triangle. Colours chosen so the cross-section cyan triangle lines
        # up with the long-axis cursor's cyan (top) end: cyan at s = +1, red
        # at s = −1.
        cyan = QColor(0, 160, 210, 240)
        red = QColor(215, 45, 45, 240)
        for s in (1.0, -1.0):
            ax, ay = self._image_to_widget_f(
                (cx_i + s * off_img * ux, cy_i + s * off_img * uy)
            )
            apex = QPoint(int(round(ax)), int(round(ay)))   # on the cut line
            bcx, bcy = ax - pdx * height, ay - pdy * height  # base centre
            b1 = QPoint(int(round(bcx - ux * half_w)),
                        int(round(bcy - uy * half_w)))
            b2 = QPoint(int(round(bcx + ux * half_w)),
                        int(round(bcy + uy * half_w)))
            p.setPen(QPen(QColor(0, 0, 0, 180), 1))
            p.setBrush(cyan if s > 0 else red)
            p.drawConvexPolygon(QPolygon([apex, b1, b2]))
        p.setBrush(Qt.BrushStyle.NoBrush)
        # Rotation-centre marker, on top of the guide.
        wp = QPoint(int(round(wc[0])), int(round(wc[1])))
        fill = (QColor(230, 30, 30, 230) if self.ivus_center_keyed
                else QColor(50, 140, 255, 230))
        p.setPen(QPen(QColor(0, 0, 0, 180), 3))
        p.setBrush(fill)
        p.drawEllipse(wp, 6, 6)
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawLine(wp.x() - 10, wp.y(), wp.x() + 10, wp.y())
        p.drawLine(wp.x(), wp.y() - 10, wp.x(), wp.y() + 10)

    def _paint_hint(self, p: QPainter, text: str) -> None:
        """Status hint centred at the top of the canvas — used while
        Center-Angle is waiting for the next perimeter click."""
        font = QFont(); font.setPointSize(11); font.setBold(True)
        p.setFont(font)
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text); pad = 8
        box = QRect((self.width() - w - 2 * pad) // 2, 6,
                    w + 2 * pad, fm.height() + 2 * pad)
        p.fillRect(box, QColor(0, 0, 0, 180))
        p.setPen(QColor(255, 200, 80))
        p.drawText(box.x() + pad, box.y() + pad + fm.ascent(), text)

    def set_overlay_font_pt(self, pt: int) -> None:
        """Set the on-image DICOM-tag / readout text size (pt) and repaint."""
        pt = int(pt)
        if pt != self._overlay_font_pt:
            self._overlay_font_pt = pt
            self.update()

    @staticmethod
    def _wrap_px(lines, fm, max_px):
        """Word-wrap each string in *lines* so no line exceeds *max_px* pixels
        at the metrics *fm*; hard-splits any single token that alone overflows.
        Pixel-accurate (QFontMetrics), so the tag block (left 40%) and result
        block (right 40%) stay inside their share and never overlap."""
        out: list[str] = []
        for line in lines:
            cur = ""
            for word in line.split(" "):
                while fm.horizontalAdvance(word) > max_px:   # over-long token
                    lo, hi = 1, len(word)
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if fm.horizontalAdvance(word[:mid]) <= max_px:
                            lo = mid
                        else:
                            hi = mid - 1
                    if cur:
                        out.append(cur)
                        cur = ""
                    out.append(word[:lo])
                    word = word[lo:]
                trial = word if not cur else cur + " " + word
                if not cur or fm.horizontalAdvance(trial) <= max_px:
                    cur = trial
                else:
                    out.append(cur)
                    cur = word
            out.append(cur)
        return out

    def _paint_overlay(self, p: QPainter):
        font = overlay_qfont(self._overlay_font_pt)
        p.setFont(font)
        fm = p.fontMetrics()
        # Line pitch tuned (×1.0125) to match the CT viewer's spacing on
        # Windows (Meiryo). macOS falls back to Hiragino, whose metrics pack
        # the lines tighter, so give it more room there. Widened per Mac
        # feedback: ×1.25, then a further ×1.1 (cumulative ≈ ×1.392).
        _pitch = 1.0125 * 1.25 * 1.1 if sys.platform == "darwin" else 1.0125
        pad, lh = 6, round(fm.height() * _pitch)
        # Confine tags to the left ~40% and word-wrap, so a larger font can't
        # run them into the right-side results.
        max_px = max(20, int(self.width() * 0.40) - 2 * pad)
        lines = self._wrap_px(self.overlay_lines, fm, max_px)
        text_w = max(fm.horizontalAdvance(s) for s in lines)
        box = QRect(6, 6, text_w + 2 * pad, lh * len(lines) + 2 * pad)
        # No background fill behind the DICOM tags — the grey box was found to
        # hurt readability over angio images, so the white text is drawn
        # straight onto the image (per user request).
        white = QColor("#ffffff")
        y = box.y() + pad + fm.ascent()
        for line in lines:
            draw_outlined_line(p, box.x() + pad, y, line, white)
            y += lh

    def _paint_compare(self, p: QPainter):
        """Filled annulus + radial gap colour map per persisted comparison, plus
        the cyan selection highlight and instruction banner while picking."""
        def W(pt):
            wf = self._image_to_widget_f(pt)
            return QPointF(wf[0], wf[1])

        if self._cmp_on:
            for mi in self._cmp_sel:
                if 0 <= mi < len(self.measures):
                    ol = self._outline_px(self.measures[mi])
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(QColor(0, 229, 255), 6.0))   # cyan = picked
                    p.drawPolyline(QPolygonF([W(q) for q in ol]))
            f = QFont()
            f.setPointSize(self._overlay_font_pt + 1)
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor(0, 229, 255))
            n = len(self._cmp_sel)
            p.drawText(QRect(0, 6, self.width(), 26),
                       int(Qt.AlignmentFlag.AlignHCenter)
                       | int(Qt.AlignmentFlag.AlignTop),
                       "Click to select 2 Ellipse/Polygon data to compare"
                       f"  ({n}/2)")

        # "Hide/Show All Result" hides every region fill + legend at once.
        compares = [] if self._results_hidden else self._compares
        for c in compares:
            if c.get("hidden"):                          # Hidden → no fill/rays
                continue
            path = QPainterPath()                        # annulus = outer − inner
            path.setFillRule(Qt.FillRule.OddEvenFill)
            path.addPolygon(QPolygonF([W(q) for q in c["outer"]]))
            path.addPolygon(QPolygonF([W(q) for q in c["inner"]]))
            fr = QColor(c["fill_hex"])
            fr.setAlpha(transp_to_alpha(c.get("transp", 50)))   # Change Transparency (def 50%)
            p.setPen(Qt.PenStyle.NoPen)
            p.fillPath(path, fr)
            if c["show_thk"]:
                for r in c["radials"]:
                    p.setPen(QPen(QColor(G.gap_color(r["gap"])),
                                  G.gap_linewidth(r["gap"])))   # red = thicker
                    p.drawLine(W(r["inner"]), W(r["outer"]))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(0, 229, 255))
                p.drawEllipse(W(c["centroid"]), 3.0, 3.0)
        # Colour legend (lower-left) only when a VISIBLE Thickness result exists.
        # The %Area result text itself is listed top-right by _paint_results.
        bands = (G.gap_legend()
                 if any(c["show_thk"] and not c.get("hidden")
                        for c in compares) else [])
        if bands:
            f = QFont()
            f.setPointSize(self._overlay_font_pt)
            p.setFont(f)
            lh = 16.0
            y = self.height() - 10 - len(bands) * lh
            for lab, hexc in bands:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(hexc))
                p.drawRect(QRectF(10, y - 10, 12, 12))
                p.setPen(QColor(230, 230, 230))
                p.drawText(QPointF(28, y), lab)
                y += lh

    def _paint_results(self, p: QPainter):
        # "Hide/Show All Result" hides every measurement / compare result text.
        if self._results_hidden:
            return
        if not self.measures and not self._compares:
            return
        font = QFont()
        font.setPointSize(self._overlay_font_pt)
        p.setFont(font)
        fm = p.fontMetrics()
        pad, lh = 6, fm.height()
        # Confine results to the right ~40% and word-wrap, so a larger font
        # can't run them into the left-side DICOM tags.
        max_px = max(20, int(self.width() * 0.40) - 2 * pad)
        # Measure metrics (yellow) then the compare %Area results (cyan), listed
        # just below — same content/colour as before, only relocated here.
        meas_lines = self._wrap_px(
            [self._metrics(m) for m in self.measures], fm, max_px)
        cmp_raw = []
        for c in self._compares:
            ht = f"Compare #{c['big_id']} vs #{c['small_id']}"
            if c.get("show_pa"):
                ht += f"  %Area:{c['pct']:.1f}%"
            cmp_raw.append(ht)
        cmp_lines = self._wrap_px(cmp_raw, fm, max_px) if cmp_raw else []
        all_lines = meas_lines + cmp_lines
        if not all_lines:
            return
        text_w = max(fm.horizontalAdvance(s) for s in all_lines)
        box = QRect(
            self.width() - text_w - 2 * pad - 6, 6,
            text_w + 2 * pad, lh * len(all_lines) + 2 * pad,
        )
        p.fillRect(box, QColor(0, 0, 0, 140))
        y = box.y() + pad + fm.ascent()
        p.setPen(QColor(255, 217, 0))
        for line in meas_lines:
            p.drawText(box.x() + pad, y, line)
            y += lh
        p.setPen(QColor(0, 229, 255))               # compare results in cyan
        for line in cmp_lines:
            p.drawText(box.x() + pad, y, line)
            y += lh
