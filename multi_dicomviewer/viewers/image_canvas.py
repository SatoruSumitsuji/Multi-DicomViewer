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
from PyQt6.QtCore import QPoint, QPointF, QRect, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QMenu, QWidget

from multi_dicomviewer.core import measure_geom as G
from multi_dicomviewer.core.coaxial import VESSEL_LABELS
from multi_dicomviewer.core.measurements import Measurement
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(80, 80)
        self.setMouseTracking(True)
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
        # IVUS-only %PlaqueArea state. Set True by IVUSViewer; when there
        # are 3+ area measures the user picks 2 by ID via context menu.
        self.is_ivus: bool = False
        self._plaque_selected: list[int] = []   # ordered, max 2 ids
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
        # True iff the CURRENT frame's rotation centre was explicitly
        # pinned by the user (= keyframe). Drives the marker colour:
        # red = pinned/fixed, blue = interpolated/movable. IVUSViewer
        # sets this whenever the active frame or the keyframe set
        # changes.
        self.ivus_center_keyed: bool = False

    # ---------------------------------------------------------------- public
    def set_frame(self, frame8: np.ndarray) -> None:
        self._qimg = to_qimage(frame8)
        self._img_size = (frame8.shape[1], frame8.shape[0])
        self.update()

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
        self._plaque_selected.clear()
        self._center_angle_target = -1
        self.update()

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

    def _hit_ivus_center(self, sx: float, sy: float) -> bool:
        """True when (sx, sy) widget coords are within the marker grab
        radius (~10 px)."""
        wp = self._ivus_center_widget()
        if wp is None:
            return False
        return math.hypot(sx - wp.x(), sy - wp.y()) <= 12.0

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
        """Geometric centre of an Ellipse/Polygon — used as the apex of
        Center Angle annotations."""
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

    # ---------------------------------------- area & %PlaqueArea (IVUS)
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

    def _area_measures(self) -> list[dict]:
        return [m for m in self.measures
                if m["type"] in ("ellipse", "polygon")]

    def _plaque_compute(self):
        """Returns one of:
          None — IVUS off, or fewer than 2 area measures.
          ('auto', pct, small, large) — exactly 2 areas (auto-paired).
          ('selected', pct, small, large) — 3+ and user picked 2.
          ('prompt', [ids]) — 3+ and the user must pick 2.
        """
        if not self.is_ivus:
            return None
        areas = [(m["id"], self._area(m)) for m in self._area_measures()]
        if len(areas) < 2:
            return None
        if len(areas) == 2:
            a1, a2 = areas[0][1], areas[1][1]
        else:
            sel = [(mid, a) for (mid, a) in areas
                   if mid in self._plaque_selected]
            if len(sel) != 2:
                return ("prompt", [mid for mid, _ in areas])
            a1, a2 = sel[0][1], sel[1][1]
        large, small = max(a1, a2), min(a1, a2)
        if large < 1e-9:
            return None
        pct = (large - small) / large * 100.0
        kind = "auto" if len(areas) == 2 else "selected"
        return (kind, pct, small, large)

    def _cleanup_plaque_selection(self):
        ids = {m["id"] for m in self.measures}
        self._plaque_selected = [i for i in self._plaque_selected if i in ids]

    def _toggle_plaque(self, mid: int) -> None:
        if mid in self._plaque_selected:
            self._plaque_selected.remove(mid)
        else:
            self._plaque_selected.append(mid)
            if len(self._plaque_selected) > 2:
                self._plaque_selected = self._plaque_selected[-2:]

    # ----------------------------------------------- right-click menus
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
            del_res = menu.addAction("Delete result")
        else:
            del_res = menu.addAction("Delete")
        chosen = menu.exec(self.mapToGlobal(QPoint(int(sx), int(sy))))
        if del_pt is not None and chosen is del_pt:
            self._delete_point(mi, vi)
        elif chosen is del_res:
            del self.measures[mi]
            self._cleanup_plaque_selection()
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
        toggle_pl = None
        if (self.is_ivus
                and m["type"] in ("ellipse", "polygon")
                and len(self._area_measures()) >= 3):
            sel = m["id"] in self._plaque_selected
            toggle_pl = menu.addAction(
                "Unselect for %PlaqueArea" if sel
                else "Select for %PlaqueArea"
            )
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
        del_act = menu.addAction("Delete")
        chosen = menu.exec(self.mapToGlobal(QPoint(int(sx), int(sy))))
        if chosen is add_pt:
            self._add_point_at(mi, sx, sy)
        elif spline_act is not None and chosen is spline_act:
            m["smooth"] = not m.get("smooth", False)
        elif center_angle_act is not None and chosen is center_angle_act:
            self._center_angle_target = mi
            m.pop("center_angle", None)              # restart picking
        elif toggle_pl is not None and chosen is toggle_pl:
            self._toggle_plaque(m["id"])
        elif chosen is del_act:
            del self.measures[mi]
            self._cleanup_plaque_selection()
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
        ca["pts"].append(pt)
        if len(ca["pts"]) >= 3:
            centre = self._shape_center(m)
            p1, p2, p3 = ca["pts"][:3]
            span, t1, t3, ccw = G.central_arc_angle(centre, p1, p2, p3)
            m["center_angle"] = {
                "pts": [p1, p2, p3], "angle": span,
                "t1": t1, "t3": t3, "ccw": ccw,
            }
            self._center_angle_target = -1

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
    def mousePressEvent(self, e):
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
        if self.ivus_show_center and self._hit_ivus_center(sx, sy):
            if e.button() == Qt.MouseButton.LeftButton:
                self._ivus_dragging_center = True
                return
            if e.button() == Qt.MouseButton.RightButton:
                self._ivus_center_menu(sx, sy)
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
            # Handle has priority over outline (more specific target).
            hit = self._pick_handle(sx, sy)
            if hit is not None:
                self._handle_menu(hit, sx, sy)
                return
            mi = self._pick_measure(sx, sy)
            if mi is None:
                return
            self._outline_menu(mi, sx, sy)
            return

        if e.button() != Qt.MouseButton.LeftButton:
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
            if m["type"] == "ellipse":
                self._set_ellipse_handle(m, vi, pt)
            else:
                m["pts"][vi] = pt
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
        """Reset frame / Reset All — right-click on the IVUS rotation-
        centre marker. The actual reset is performed by the viewer
        (single source of truth for the per-frame centres array)."""
        menu = QMenu(self)
        a_frame = menu.addAction("Reset frame (this frame ➔ image centre)")
        a_all = menu.addAction("Reset All (every frame ➔ image centre)")
        chosen = menu.exec(self.mapToGlobal(QPoint(int(sx), int(sy))))
        if chosen is a_frame:
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
            self.update()
        QTimer.singleShot(0, _publish)

    # ---------------------------------------------------------------- paint
    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0a0a0a"))
        if self._qimg is None:
            return
        r = self._draw_rect()
        p.drawImage(r, self._qimg)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Rebuilt as labels are drawn below; clear so a deleted measure's
        # stale rect can't linger for the next drag hit-test.
        self._label_rects = {}

        # Major/minor axes (drawn UNDER outlines): thick dotted, in each
        # measure's own colour (alpha-modulated so the outline still
        # dominates).
        for m in self.measures:
            try:
                segs = self._axis_segs_px(m)
            except Exception:
                continue
            base = QColor(m.get("color", DEFAULT_MEAS_COLOR))
            base.setAlpha(170)
            axis_pen = QPen(base, 2, Qt.PenStyle.DashLine)
            for seg in segs:
                if seg is None:
                    continue
                p.setPen(axis_pen)
                a = self._image_to_widget(seg[0])
                b = self._image_to_widget(seg[1])
                p.drawLine(a, b)

        # Outlines — per-measure colour (Change Color menu).
        for m in self.measures:
            try:
                outline = self._outline_px(m)
            except Exception:
                continue
            col = QColor(m.get("color", DEFAULT_MEAS_COLOR))
            p.setPen(QPen(col, 1.6))
            wpts = [self._image_to_widget(q) for q in outline]
            for a, b in zip(wpts, wpts[1:]):
                p.drawLine(a, b)

        # Center-Angle annotations (3 spokes from shape centre + small
        # dots on the picked perimeter points).
        for m in self.measures:
            ca = m.get("center_angle")
            if not ca or "pts" not in ca:
                continue
            centre = self._shape_center(m)
            wc = self._image_to_widget(centre)
            col = QColor(m.get("color", DEFAULT_MEAS_COLOR))
            spoke = QColor(col); spoke.setAlpha(200)
            p.setPen(QPen(spoke, 1.2, Qt.PenStyle.DashLine))
            for q in ca["pts"]:
                wq = self._image_to_widget(q)
                p.drawLine(wc, wq)
            p.setPen(QPen(QColor(255, 140, 0), 1.0))
            p.setBrush(QColor(255, 140, 0))
            for q in ca["pts"]:
                wq = self._image_to_widget(q)
                p.drawEllipse(wq, 4, 4)
            p.setBrush(Qt.BrushStyle.NoBrush)
            if "angle" in ca:
                p.setPen(QColor(255, 140, 0))
                p.drawText(wc + QPoint(10, -10), f"{ca['angle']:.1f}°")

        # Vertex handles + running-number labels.
        # The single handle currently being dragged turns green and
        # 1.5× larger so the edit state is unmistakable.
        edit_mi = self._edit["mi"] if self._edit is not None else -1
        edit_vi = self._edit["vi"] if self._edit is not None else -1
        yellow = QColor(255, 217, 0)
        green = QColor(60, 220, 90)
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        p.setFont(font)
        for mi, m in enumerate(self.measures):
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

        # Center-Angle pick-mode hint.
        if self._center_angle_target >= 0:
            picked = len(self.measures[self._center_angle_target]
                         .get("center_angle", {}).get("pts", []))
            self._paint_hint(
                p,
                f"Center Angle: click {3 - picked} more point(s) "
                "on the perimeter",
            )

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

        # Result lines (top-right, beneath any overlay would be busy).
        self._paint_results(p)

        # DICOM-tag overlay (top-left, existing behaviour).
        if self.overlay_lines:
            self._paint_overlay(p)

        # IVUS long-axis rotation-centre marker. Drawn last so it sits
        # on top of overlays. Colour conveys the centre's "state" so
        # the user can tell at a glance whether this frame is a key:
        #   red  = pinned by the user on this exact frame (keyed)
        #   blue = interpolated from neighbouring keys (movable: drag
        #          to pin a new key here, or use the menu to reset)
        # Crosshair-on-filled-dot is preserved for visibility on
        # grayscale IVUS without dominating the cross-section.
        if (self.ivus_show_center
                and self.ivus_center_image is not None
                and self._img_size != (0, 0)):
            wp = self._image_to_widget(self.ivus_center_image)
            if self.ivus_center_keyed:
                fill = QColor(230, 30, 30, 230)     # red — fixed
            else:
                fill = QColor(50, 140, 255, 230)    # blue — movable
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
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
        p.fillRect(box, QColor(0, 0, 0, 140))
        p.setPen(QColor("#ffffff"))
        y = box.y() + pad + fm.ascent()
        for line in lines:
            p.drawText(box.x() + pad, y, line)
            y += lh

    def _paint_results(self, p: QPainter):
        if not self.measures:
            return
        font = QFont()
        font.setPointSize(self._overlay_font_pt)
        p.setFont(font)
        fm = p.fontMetrics()

        # IVUS %PlaqueArea (if applicable) — and a "★ " prefix on the
        # measures the user has selected to be the two compared.
        pa = self._plaque_compute()
        selected_ids: set[int] = set()
        if pa and pa[0] == "prompt":
            selected_ids = set(self._plaque_selected)
        elif pa and pa[0] == "selected":
            selected_ids = set(self._plaque_selected)

        lines = []
        for m in self.measures:
            prefix = "★ " if m["id"] in selected_ids else ""
            lines.append(prefix + self._metrics(m))
        if pa:
            unit = self._unit()
            if pa[0] in ("auto", "selected"):
                _, pct, small, large = pa
                lines.append(
                    f"%PlaqueArea: {pct:.1f}%  "
                    f"(small {small:.1f}{unit}² / large {large:.1f}{unit}²)"
                )
            else:                                # prompt: ids to pick from
                ids = " ".join(f"#{i}" for i in pa[1])
                lines.append(
                    f"%PlaqueArea: choose 2 from  {ids}"
                )
                lines.append(
                    "  (right-click an outline → Select for %PlaqueArea)"
                )

        pad, lh = 6, fm.height()
        # Confine results to the right ~40% and word-wrap, so a larger font
        # can't run them into the left-side DICOM tags.
        max_px = max(20, int(self.width() * 0.40) - 2 * pad)
        lines = self._wrap_px(lines, fm, max_px)
        text_w = max(fm.horizontalAdvance(s) for s in lines)
        box = QRect(
            self.width() - text_w - 2 * pad - 6, 6,
            text_w + 2 * pad, lh * len(lines) + 2 * pad,
        )
        p.fillRect(box, QColor(0, 0, 0, 140))
        y = box.y() + pad + fm.ascent()
        for line in lines:
            # %PlaqueArea result line in green so it stands out.
            colour = QColor(120, 230, 130) if line.startswith("%PlaqueArea:") \
                else QColor(255, 217, 0)
            p.setPen(colour)
            p.drawText(box.x() + pad, y, line)
            y += lh
