"""macOS CT MPR viewer — pygfx (wgpu→Metal) reimplementation of ct_viewer.py.

VTK's CT MPR renders through OpenGL, whose Apple OpenGL-on-Metal emulation
hangs on macOS. This module renders the same SSMview-style dual linked oblique
MPR with pygfx + wgpu (Metal native), so the Mac build can ship full CT. It is
selected on darwin by the shell's viewer factory; Windows keeps the VTK module.

It deliberately does NOT import ct_viewer.py (that module imports vtk at the top
and vtk is not installed on Mac). The pure-numpy geometry/state — identical in
behaviour to the VTK viewer — is reproduced here; only the render sink differs:

    VTK                                   pygfx
    vtkImageReslice (oblique resample)  → gfx.VolumeSliceMaterial.plane (GPU)
    vtkImageMapToColors + vtkLookupTable→ material.clim (+ material.map, Phase 4)
    vtkImageActor + vtkRenderer         → gfx.Volume in a Scene
    vtkCamera (ParallelProjection)      → gfx.OrthographicCamera looking down N

Coordinate model (the crux). pygfx places a Volume's grid at voxel coords; we
set the Volume's local.scale to the mm spacing (sx,sy,sz) so pygfx WORLD coords
== the VTK volume's mm coords — the same space the frame (U,V,N), _center and
_pc live in. The slice plane is evaluated in world space (volume_slice.wgsl), so
the plane through a pane's reslice centre is (N, -N·_pc[key]). The per-pane
orthographic camera is oriented with local axes (X,Y,Z) = (U,V,N): it looks down
-N with up=V, so the oblique slice is rendered face-on exactly like the VTK
reslice output. ParallelScale ↔ half the camera height in mm.

Phase 1 scope: load, dual/single pane, W/L, paging (drag + wheel), zoom, move,
3-D rotate, reset, presets. Overlays (crosshair/measures/text), HU colormap and
slab-MIP arrive in later phases.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import pygfx as gfx
import pylinalg as la
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPolygonF, QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from rendercanvas.pyqt6 import RenderCanvas

from multi_dicomviewer.config import CT_WL_PRESETS
from multi_dicomviewer.core.dicom_io import LoadedSeries
from multi_dicomviewer.core.dicom_tags import default_overlay_keywords, overlay_lines
from multi_dicomviewer.ui.tag_font import (
    TAG_FONT_PT_DEFAULT, build_tag_font_control, overlay_qfont,
)
from multi_dicomviewer.core.measurements import Measurement
from multi_dicomviewer.core.measure_geom import (
    angle_at as _angle_at,
    arc_through as _arc_through,
    central_arc_angle as _central_arc_angle,
    dist as _dist,
    ellipse_axes as _ellipse_axes,
    ellipse_cab as _ellipse_cab,
    ellipse_drag as _ellipse_drag,
    ellipse_from_major as _ellipse_from_major,
    ellipse_outline as _ellipse_outline,
    major_minor as _major_minor,
    point_in_poly as _point_in_poly,
    poly_area as _poly_area,
    polygon_centroid as _polygon_centroid,
    project_to_polyline as _project_to_polyline,
    seg_dist as _seg_dist,
    smooth_closed as _smooth_closed,
    smooth_open as _smooth_open,
)
from multi_dicomviewer.ui.viewer_base import AbstractViewer

#: SPIN sign. +1.0 matches the rotation direction expected on the Mac build.
_SPIN_SIGN = 1.0

# Slab-MIP level-of-detail. The slab MIP is a CPU max-over-N-oblique-planes
# image (see _slab_mip_hu); its cost scales with sample columns² × plane count.
# At rest we build it full quality; DURING an interactive drag/page we build a
# COARSER one — fewer sample columns and fewer MIP planes — so the THICK image
# KEEPS its slab look (instead of dropping to the thin GPU slice) while staying
# smooth on low-memory Macs. The debounce timer rebuilds full quality on settle.
# Tune these if motion is still heavy (lower) or too soft (raise) on the Mac.
_SLAB_IW_FULL = 480     # slab-MIP sample columns at rest
_SLAB_IW_LOD = 200      # ...during an interactive drag/page (coarse but smooth)
_SLAB_PLANES_FULL = 64  # MIP plane cap at rest
_SLAB_PLANES_LOD = 8    # ...during an interactive drag/page

_TOOLS = ("ZOOM", "MOVE", "ROTATE", "SPIN", "PAGING", "THICK", "WL")
_TOOL_KEYS = {
    Qt.Key.Key_Z: "ZOOM", Qt.Key.Key_V: "MOVE", Qt.Key.Key_S: "SPIN",
    Qt.Key.Key_G: "PAGING", Qt.Key.Key_W: "WL", Qt.Key.Key_R: "ROTATE",
    Qt.Key.Key_T: "THICK",
}  # C = ColorMap toggle (handled separately)


# --------------------------------------------------------------------- math
# (copied verbatim from ct_viewer.py — pure numpy, no vtk dependency)
def _norm(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _rotate(v, axis, deg):
    """Rodrigues rotation of vector *v* about unit *axis* by *deg*."""
    a = _norm(axis)
    th = math.radians(deg)
    return (
        v * math.cos(th)
        + np.cross(a, v) * math.sin(th)
        + a * np.dot(a, v) * (1 - math.cos(th))
    )


# ------------------------------------------------------------- HU colormap
#: Default HU colour bands (SSMview-style). Each colours [lo,hi]; "on" toggles.
#: opacity blends band colour over the windowed grayscale (0=gray, 1=colour).
_DEFAULT_BANDS = [
    {"rgb": (1.0, 0.0, 0.0), "lo": -1000, "hi": 0,    "on": True},
    {"rgb": (1.0, 1.0, 0.0), "lo": 0,     "hi": 50,   "on": True},
    {"rgb": (0.0, 1.0, 0.0), "lo": 50,    "hi": 250,  "on": True},
    {"rgb": (0.0, 0.0, 1.0), "lo": 250,   "hi": 350,  "on": False},  # blue off by default
    {"rgb": (1.0, 1.0, 1.0), "lo": 350,   "hi": 700,  "on": True},
    {"rgb": (1.0, 0.0, 1.0), "lo": 850,   "hi": 2000, "on": True},
]
_HU_LO, _HU_HI = -1000.0, 2000.0


def _hex_to_rgb(hexstr, default=(0x33, 0xE6, 0xFF)):
    if not hexstr:
        return default
    s = hexstr.lstrip("#")
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except Exception:
        return default


def _band_lut_array(bands, opacity, win, lvl) -> np.ndarray:
    """512×4 RGBA float32 colormap over HU [_HU_LO,_HU_HI]. Inside the FIRST
    enabled band containing a HU value: band colour blended over the windowed
    grayscale by *opacity*. Outside any band: grayscale. The numpy analogue of
    the VTK viewer's _band_lut (which produced a vtkLookupTable); here it feeds
    a pygfx 1-D colormap Texture assigned to VolumeSliceMaterial.map."""
    n = 512
    hu = _HU_LO + (_HU_HI - _HU_LO) * np.arange(n) / (n - 1)
    glo = lvl - win / 2.0
    span = max(1e-6, float(win))
    g = np.clip((hu - glo) / span, 0.0, 1.0).astype(np.float32)
    op = float(min(1.0, max(0.0, opacity)))
    out = np.stack([g, g, g, np.ones_like(g)], axis=1).astype(np.float32)
    assigned = np.zeros(n, dtype=bool)
    for b in bands:
        if not b["on"]:
            continue
        m = (hu >= b["lo"]) & (hu <= b["hi"]) & (~assigned)
        if not m.any():
            continue
        col = b["rgb"]
        for c in range(3):
            out[m, c] = op * col[c] + (1.0 - op) * g[m]
        assigned |= m
    return out


# ----------------------------------------------------------- HU sampling
def _trilinear_sample(vol: np.ndarray, fx, fy, fz) -> np.ndarray:
    """Trilinearly sample *vol* (z,y,x) at fractional voxel coords (fx=col,
    fy=row, fz=slice). Inputs are arrays; out-of-range is clamped to the
    border. Replaces the VTK reslice-output readback used for HU stats — no
    scipy. Runs only when a measurement is finalised/edited, not per frame."""
    nz, ny, nx = vol.shape
    fx = np.clip(np.asarray(fx, np.float64), 0, nx - 1)
    fy = np.clip(np.asarray(fy, np.float64), 0, ny - 1)
    fz = np.clip(np.asarray(fz, np.float64), 0, nz - 1)
    x0 = np.floor(fx).astype(int); y0 = np.floor(fy).astype(int)
    z0 = np.floor(fz).astype(int)
    x1 = np.minimum(x0 + 1, nx - 1); y1 = np.minimum(y0 + 1, ny - 1)
    z1 = np.minimum(z0 + 1, nz - 1)
    dx = fx - x0; dy = fy - y0; dz = fz - z0
    c000 = vol[z0, y0, x0]; c001 = vol[z0, y0, x1]
    c010 = vol[z0, y1, x0]; c011 = vol[z0, y1, x1]
    c100 = vol[z1, y0, x0]; c101 = vol[z1, y0, x1]
    c110 = vol[z1, y1, x0]; c111 = vol[z1, y1, x1]
    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx
    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy
    return c0 * (1 - dz) + c1 * dz


# ----------------------------------------------------------------- pane
class _PygfxPane:
    """One MPR pane: a wgpu canvas + scene + ortho camera + a Volume whose
    VolumeSliceMaterial cuts the oblique slice on the GPU."""

    def __init__(self):
        # ondemand: only redraw when the viewer calls render() (request_draw).
        self.canvas = RenderCanvas(update_mode="ondemand")
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.scene.add(gfx.Background(material=gfx.BackgroundMaterial(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))))
        self.cam = gfx.OrthographicCamera()
        self.cam.maintain_aspect = False     # we set width/height to the aspect
        self.mesh = None
        self.material = None
        self.canvas.request_draw(self._draw)

    def _draw(self):
        self.renderer.render(self.scene, self.cam)

    def set_volume(self, vol: np.ndarray, scale) -> None:
        if self.mesh is not None:
            self.scene.remove(self.mesh)
            self.mesh = None
        tex = gfx.Texture(np.ascontiguousarray(vol, dtype=np.float32), dim=3)
        geom = gfx.Geometry(grid=tex)
        self.material = gfx.VolumeSliceMaterial(
            clim=(-100.0, 700.0), interpolation="linear",
            plane=(0.0, 0.0, 1.0, 0.0))
        self.mesh = gfx.Volume(geom, self.material)
        self.mesh.local.scale = tuple(float(s) for s in scale)  # voxel→mm
        self.scene.add(self.mesh)

    def render(self) -> None:
        # force_draw renders synchronously so the GPU slice tracks the cursor
        # with no one-frame lag behind the QPainter overlay (MOVE/recenter feel).
        try:
            self.canvas.force_draw()
        except Exception:
            self.canvas.request_draw()


_BORDER = 3  # px; matches the active-pane QFrame border so children inset


class _PaneHost(QFrame):
    """Holds a pane's wgpu canvas with a transparent QPainter overlay stacked
    on top (proven to composite over the Metal surface in the overlay spike).
    No layout — both children are sized to the bordered content rect."""

    def __init__(self, canvas: QWidget, overlay: QWidget):
        super().__init__()
        self.setObjectName("ctpane")
        self._canvas = canvas
        self._overlay = overlay
        canvas.setParent(self)
        overlay.setParent(self)
        overlay.raise_()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        m = _BORDER
        rr = QRect(m, m, max(1, self.width() - 2 * m),
                   max(1, self.height() - 2 * m))
        self._canvas.setGeometry(rr)
        self._overlay.setGeometry(rr)


class _Overlay(QWidget):
    """Transparent QPainter layer drawing the crosshair, ▲ projection markers,
    slab-width guides, corner info text and the angio-angle readout — the VTK
    viewer's vtk* overlay actors, re-expressed as 2-D painting over the GPU
    slice. Mouse-transparent: pointer events fall through to the canvas."""

    def __init__(self, viewer: "CTViewer", key: str):
        super().__init__()
        self._v = viewer
        self._key = key
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def paintEvent(self, _e):
        v, key = self._v, self._key
        if v._vol is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        # slab-MIP base image (GPU slice is hidden in slab mode)
        mip = v._mip_img.get(key)
        if mip is not None:
            p.drawImage(self.rect(), mip)
        if v._cl_on:
            self._paint_cross(p, key, w, h)
        self._paint_measures(p, key, w, h)
        self._paint_info(p, key, w, h)

    # -- crosshair + ▲ markers + slab guides -------------------------------
    def _paint_cross(self, p, key, w, h):
        v = self._v
        ccx, ccy = v._cc(key)
        half = v._half
        a = math.radians(v._cross_ang[key])
        uh = (math.cos(a), math.sin(a))         # horizontal line dir (output)
        uv = (-math.sin(a), math.cos(a))        # vertical line dir

        def S(ox, oy):                          # output (ox,oy) -> screen pt
            sx, sy = v._world_to_screen(key, ox, oy)
            return QPointF(sx, sy)

        pen = QPen(QColor(255, 217, 0, 128), 1.0)
        p.setPen(pen)
        # full-extent crosshair lines through the crosshair centre
        p.drawLine(S(ccx - half * uh[0], ccy - half * uh[1]),
                   S(ccx + half * uh[0], ccy + half * uh[1]))
        p.drawLine(S(ccx - half * uv[0], ccy - half * uv[1]),
                   S(ccx + half * uv[0], ccy + half * uv[1]))

        # ▲ markers: the OTHER pane's projection direction, a constant
        # fraction of the viewport from the centre (size tied to ps).
        ps = v._ps[key]
        d = 0.255 * ps
        sz = 0.024 * ps
        p.setBrush(QColor(0, 242, 64))
        p.setPen(Qt.PenStyle.NoPen)
        for sgn in (1.0, -1.0):
            ax = ccx + sgn * d * uh[0]
            ay = ccy + sgn * d * uh[1]
            apex = S(ax + sz * uv[0], ay + sz * uv[1])
            b1 = S(ax - 0.6 * sz * uh[0], ay - 0.6 * sz * uh[1])
            b2 = S(ax + 0.6 * sz * uh[0], ay + 0.6 * sz * uh[1])
            p.drawPolygon(QPolygonF([apex, b1, b2]))

        # slab-width guides: two dashed lines parallel to the horizontal
        # line, offset by ±thick/2 of the OTHER pane.
        other = "B" if key == "A" else "A"
        t = v._thick[other]
        if t > 0:
            ht = t / 2.0
            dpen = QPen(QColor(255, 217, 0, 128), 1.0, Qt.PenStyle.DashLine)
            p.setPen(dpen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            for off in (ht, -ht):
                ox0 = ccx - half * uh[0] + off * uv[0]
                oy0 = ccy - half * uh[1] + off * uv[1]
                ox1 = ccx + half * uh[0] + off * uv[0]
                oy1 = ccy + half * uh[1] + off * uv[1]
                p.drawLine(S(ox0, oy0), S(ox1, oy1))

    # -- measurements (outlines, calipers, handles, labels, results) -------
    def _paint_measures(self, p, key, w, h):
        v = self._v

        def S(pt):
            sx, sy = v._world_to_screen(key, pt[0], pt[1])
            return QPointF(sx, sy)

        def poly(pts):
            return QPolygonF([S(q) for q in pts])

        def draw_outline(pts, rgb, width=1.8):    # 1.5 ×1.2 — readability
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(*rgb), width))
            if len(pts) >= 2:
                p.drawPolyline(poly(pts))

        def draw_dashed(seg, rgb):
            pen = QPen(QColor(*rgb), 2.64)         # 2.2 ×1.2 — readability
            pen.setStyle(Qt.PenStyle.DotLine)
            p.setPen(pen)
            p.drawLine(S(seg[0]), S(seg[1]))

        def dots(pts, color, r=4.0):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            for q in pts:
                p.drawEllipse(S(q), r, r)

        e = v._edit
        edit_mi = e["mi"] if (e and e["key"] == key) else -1
        edit_vi = e["vi"] if (e and e["key"] == key) else -1
        edit_ca = bool(e.get("ca")) if (e and e["key"] == key) else False

        for mi, m in enumerate(v._measures[key]):
            rgb = _hex_to_rgb(m.get("color"))
            draw_outline(v._outline(m), rgb)
            if m["type"] in ("ellipse", "polygon"):
                maj, mnr, _, _ = _major_minor(m)
                for seg in (maj, mnr):
                    if seg is not None:
                        # Long/short-diameter lines wear the polygon-vertex
                        # colour (yellow) so they read as part of the shape.
                        draw_dashed(seg, (255, 217, 0))
            ca = m.get("center_angle")
            if ca and ca.get("pts"):
                centre = v._shape_center(m)
                # Solid orange arc on the outline from p1→p3 through p2 — shows
                # exactly which part of the perimeter the central angle spans.
                if "angle" in ca and len(ca["pts"]) >= 3:
                    arc = _arc_through(v._outline(m), ca["pts"][0],
                                       ca["pts"][1], ca["pts"][2])
                    if len(arc) >= 2:
                        p.setPen(QPen(QColor(255, 140, 0), 2.88))  # 2.4 ×1.2
                        p.setBrush(Qt.BrushStyle.NoBrush)
                        p.drawPolyline(poly(arc))
                for ci, q in enumerate(ca["pts"]):
                    # The 2nd point only picks which way the angle is measured,
                    # so it gets no spoke. Spokes (orange, = marker colour) go to
                    # the 1st and 3rd points — the angle's two arms.
                    if ci == 1:
                        continue
                    draw_dashed((centre, q), (255, 140, 0))
                # Orange marker picks; the one being dragged turns green (like a
                # polygon vertex).
                ca_edit = edit_ca and mi == edit_mi and 0 <= edit_vi < len(ca["pts"])
                ca_idle = [q for ci, q in enumerate(ca["pts"])
                           if not (ca_edit and ci == edit_vi)]
                dots(ca_idle, QColor(255, 140, 0), 5.0)
                if ca_edit:
                    dots([ca["pts"][edit_vi]], QColor(59, 219, 90), 7.0)
            idle = [q for vi, q in enumerate(v._handles(m))
                    if not (mi == edit_mi and not edit_ca and vi == edit_vi)]
            dots(idle, QColor(255, 217, 0), 4.0)            # yellow handles
            if mi == edit_mi and not edit_ca and 0 <= edit_vi < len(m["pts"]):
                dots([m["pts"][edit_vi]], QColor(59, 219, 90), 7.0)  # green
            # numeric id label at the anchor
            p.setPen(QColor(255, 217, 0))
            fb = QFont("monospace", v._overlay_font_pt)
            fb.setBold(True)
            p.setFont(fb)
            ax, ay = v._world_to_screen(key, *v._anchor(m))
            p.drawText(QPointF(ax + 6, ay - 6), str(m["id"]))

        d = v._draft
        if d and d["pane"] == key and d["pts"]:
            # Yellow DASHED preview that follows the cursor while points are
            # being placed (matches the XA/IVUS canvas), incl. the angle tool.
            pen = QPen(QColor(244, 208, 63), 1.2)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            hover = v._meas_hover
            if d["type"] == "ellipse" and hover is not None:
                # Major axis = first click → cursor; preview the oblique ellipse.
                p.drawPolyline(poly(_ellipse_outline(
                    _ellipse_from_major(d["pts"][0], hover))))
            else:
                preview = list(d["pts"])
                if hover is not None and d["type"] != "ellipse":
                    preview.append(hover)
                if len(preview) >= 2:
                    p.drawPolyline(poly(preview))
            dots(list(d["pts"]), QColor(255, 217, 0), 4.0)

        # per-measure result strings, top-right, confined to the right 40% and
        # word-wrapped so growing the font can't make them overlap the tags.
        lines = v._metrics.get(key, [])
        if lines:
            p.setPen(QColor(255, 217, 0))   # yellow — match the other modalities
            p.setFont(QFont("monospace", v._overlay_font_pt))
            rx = w * 0.60
            flags = (int(Qt.AlignmentFlag.AlignRight)
                     | int(Qt.AlignmentFlag.AlignTop)
                     | int(Qt.TextFlag.TextWordWrap))
            p.drawText(QRectF(rx, 4, w - rx - 6, h - 40), flags,
                       "\n".join(lines))

    # -- corner info text + angio readout ----------------------------------
    def _paint_info(self, p, key, w, h):
        v = self._v
        p.setPen(QColor(255, 255, 255))         # white DICOM-tag / readout text
        if v._tags_on:
            # No explicit selection yet → show sensible defaults so first-time
            # users see tags without opening the dialog. A saved/edited
            # selection (non-empty) takes over and persists via core.settings.
            kws = v._tag_keywords or (default_overlay_keywords(v._header)
                                      if v._header is not None else [])
            head = overlay_lines(v._header, kws, anonymized=v._anon)
            if head:
                p.setFont(overlay_qfont(v._overlay_font_pt))
                # Confine tags to the left 40% and word-wrap, so a larger font
                # never runs them into the right-side measure results.
                flags = (int(Qt.AlignmentFlag.AlignLeft)
                         | int(Qt.AlignmentFlag.AlignTop)
                         | int(Qt.TextFlag.TextWordWrap))
                p.drawText(QRectF(6, 4, w * 0.40 - 6, h - 40), flags,
                           "\n".join(head))
        p.setFont(QFont("monospace", 12))       # corner readouts stay compact
        slab = v._thick[key]
        kind = f"Slab MIP {slab:.1f}mm" if slab > 0 else "MPR (thin)"
        p.drawText(QRectF(6, h - 28, w - 12, 24),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"WW {v._win:.0f}  WL {v._lvl:.0f}")
        p.drawText(QRectF(6, h - 28, w - 12, 24),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"{key}  |  {kind}")
        # angio readout (yellow, bottom-centre) — clinical, always shown
        ang = v._angio_angle(key)
        if ang:
            p.setPen(QColor(255, 230, 0))
            fb = QFont("monospace", 15)
            fb.setBold(True)
            p.setFont(fb)
            p.drawText(QRectF(0, h - 36, w, 30),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                       ang)


class _ColorMapDialog(QDialog):
    """SSMview-style HU colour-map editor (colour + HU Min/Max + enable/remove
    per band, Opacity slider, Add/Reset). Changes apply live via
    on_change(bands, opacity). Pure Qt — copied verbatim from the VTK viewer."""

    def __init__(self, bands, opacity, on_change, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ColorMap Setting")
        self.resize(560, 420)
        self._bands = [dict(b) for b in bands]
        self._opacity = float(opacity)
        self._on_change = on_change

        self._rows_host = QWidget()
        self._rows = QVBoxLayout(self._rows_host)
        self._rows.setContentsMargins(4, 4, 4, 4)
        self._rows.setSpacing(4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._rows_host)

        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacity"))
        self._op = QSlider(Qt.Orientation.Horizontal)
        self._op.setRange(0, 100)
        self._op.setValue(int(round(self._opacity * 100)))
        self._op.valueChanged.connect(self._op_changed)
        self._op_lbl = QLabel(f"{self._opacity:.2f}")
        op_row.addWidget(self._op, 1)
        op_row.addWidget(self._op_lbl)

        btns = QHBoxLayout()
        add = QPushButton("Add")
        add.clicked.connect(self._add_band)
        rst = QPushButton("Reset")
        rst.clicked.connect(self._reset)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        btns.addWidget(add)
        btns.addWidget(rst)
        btns.addStretch(1)
        btns.addWidget(close)

        col = QVBoxLayout(self)
        col.addWidget(scroll, 1)
        col.addLayout(op_row)
        col.addLayout(btns)
        self._rebuild()

    def _emit(self):
        self._on_change(self._bands, self._opacity)

    def _op_changed(self, v):
        self._opacity = v / 100.0
        self._op_lbl.setText(f"{self._opacity:.2f}")
        self._emit()

    def _add_band(self):
        self._bands.append(
            {"rgb": (1.0, 1.0, 1.0), "lo": 0, "hi": 100, "on": True})
        self._rebuild()
        self._emit()

    def _reset(self):
        self._bands = [dict(b) for b in _DEFAULT_BANDS]
        self._opacity = 0.25
        self._op.blockSignals(True)
        self._op.setValue(25)
        self._op.blockSignals(False)
        self._op_lbl.setText("0.25")
        self._rebuild()
        self._emit()

    def set_bands(self, bands, opacity) -> None:
        self._bands = [dict(b) for b in bands]
        self._opacity = float(opacity)
        self._op.blockSignals(True)
        self._op.setValue(int(round(self._opacity * 100)))
        self._op.blockSignals(False)
        self._op_lbl.setText(f"{self._opacity:.2f}")
        self._rebuild()

    def _rebuild(self):
        while self._rows.count():
            it = self._rows.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
        for idx in range(len(self._bands)):
            self._rows.addWidget(self._row_widget(idx))
        self._rows.addStretch(1)

    def _row_widget(self, idx) -> QWidget:
        b = self._bands[idx]
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(2, 2, 2, 2)
        sw = QPushButton()
        sw.setFixedWidth(40)
        r, g, bl = (int(c * 255) for c in b["rgb"])
        sw.setStyleSheet(f"background:rgb({r},{g},{bl});")
        sw.clicked.connect(lambda _c, i=idx: self._pick_color(i))
        h.addWidget(sw)
        h.addWidget(QLabel("Min"))
        lo = QSpinBox()
        lo.setRange(-1024, 4096)
        lo.setValue(int(b["lo"]))
        lo.valueChanged.connect(lambda v, i=idx: self._set(i, "lo", v))
        h.addWidget(lo)
        h.addWidget(QLabel("Max"))
        hi = QSpinBox()
        hi.setRange(-1024, 4096)
        hi.setValue(int(b["hi"]))
        hi.valueChanged.connect(lambda v, i=idx: self._set(i, "hi", v))
        h.addWidget(hi)
        en = QPushButton("Enabled" if b["on"] else "Disabled")
        en.setCheckable(True)
        en.setChecked(b["on"])
        en.clicked.connect(lambda _c, i=idx: self._toggle(i))
        h.addWidget(en)
        rm = QPushButton("Remove")
        rm.clicked.connect(lambda _c, i=idx: self._remove(i))
        h.addWidget(rm)
        return w

    def _set(self, idx, key, val):
        self._bands[idx][key] = val
        self._emit()

    def _toggle(self, idx):
        self._bands[idx]["on"] = not self._bands[idx]["on"]
        self._rebuild()
        self._emit()

    def _remove(self, idx):
        del self._bands[idx]
        self._rebuild()
        self._emit()

    def _pick_color(self, idx):
        c0 = self._bands[idx]["rgb"]
        col = QColorDialog.getColor(
            QColor(int(c0[0] * 255), int(c0[1] * 255), int(c0[2] * 255)),
            self, "Band colour")
        if col.isValid():
            self._bands[idx]["rgb"] = (col.redF(), col.greenF(), col.blueF())
            self._rebuild()
            self._emit()


# --------------------------------------------------------------- viewer
class CTViewer(AbstractViewer):
    handles_modality = "CT"
    tags_requested = pyqtSignal()
    #: emitted when the tag-text-size slider moves (shell broadcasts the pt)
    overlay_font_changed = pyqtSignal(int)
    #: emitted when the user clicks "Measure History" (shell opens the dialog)
    history_requested = pyqtSignal()
    #: emitted on every committed measurement (shell logs it per study)
    measurement_added = pyqtSignal(object)
    #: fired from a background debounce thread to wake the GUI thread and crisp
    #: up the slab LOD. A cross-thread queued signal posts an event that wakes a
    #: fully-idle Qt loop — which same-thread QTimer/aboutToBlock can't do
    #: reliably under rendercanvas ondemand (see _arm_lod / _lod_settle).
    _lod_wake = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._vol = None
        self._header = None
        self._pbasis = np.eye(3)
        self._tag_keywords: list[str] = []
        self._tags_on = True                 # DICOM tag overlay visible (Q toggles)
        self._anon = False
        self._tool = "PAGING"
        self._dims = (1.0, 1.0, 1.0)         # sx, sy, sz mm
        self._bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        self._diag = 1.0
        self._cam_off = 2.0                  # camera distance along N (mm)
        self._center = np.zeros(3)
        self._center0 = np.zeros(3)
        self._pc = {"A": np.zeros(3), "B": np.zeros(3)}
        self._frame = {
            "A": (np.array([1.0, 0.0, 0.0]),
                  np.array([0.0, 1.0, 0.0]),
                  np.array([0.0, 0.0, 1.0])),
            "B": (np.array([1.0, 0.0, 0.0]),
                  np.array([0.0, 0.0, 1.0]),
                  np.array([0.0, 1.0, 0.0])),
        }
        self._cross_ang = {"A": 0.0, "B": 0.0}
        self._half = 1.0
        self._ps = {"A": 1.0, "B": 1.0}      # camera half-height (mm) per pane
        self._pan = {"A": np.zeros(2), "B": np.zeros(2)}  # focal offset (u,v)
        self._roll = {"A": 0.0, "B": 0.0}    # camera roll (deg) for SPIN
        self._win, self._lvl = 800.0, 200.0
        self._win0, self._lvl0 = 800.0, 200.0
        self._thick = {"A": 0.0, "B": 5.0}
        self._active_pane = "A"
        self._view_initial = True
        self._cl_on = True                   # crosshair/slab overlay visible
        self._color = False                  # HU colormap on/off
        self._bands = [dict(b) for b in _DEFAULT_BANDS]
        self._opacity = 0.25
        self._cmap_dlg = None
        self._lut_key = None                 # cache key for the colormap tex
        self._lut_tex = None
        # measurements
        self._meas_on = False
        self._meas_type = None               # line|polyline|ellipse|polygon|angle
        self._measures = {"A": [], "B": []}  # finalized {id,type,pts,...}
        self._meas_seq = 0
        self._draft = None                   # {type, pane, pts} in progress
        self._edit = None                    # {key, mi, vi} handle drag
        self._center_angle_target = None     # {key, mi} during 3-pt pick
        self._metrics = {"A": [], "B": []}   # per-measure result strings
        self._meas_drag = False              # canvas is dragging a handle
        self._meas_hover = None              # cursor (output coords) for draft preview
        self._mip_img = {"A": None, "B": None}   # slab-MIP QImage per pane
        #: On-image DICOM-tag / readout text size (pt), shared across modalities
        #: via the shell. Read by the pane overlays' paint.
        self._overlay_font_pt = TAG_FONT_PT_DEFAULT
        # Interactive level-of-detail: during a drag / wheel-page the slab-MIP
        # is built coarse, then rebuilt full quality once the interaction
        # settles, so paging stays smooth on low-memory Macs.
        #
        # Crisping up reliably is the hard part. After a wheel/trackpad page the
        # Qt loop goes FULLY idle (no pointer-up to pump it) and, under
        # rendercanvas's ondemand canvas, a same-thread QTimer timeout (and even
        # aboutToBlock) is not serviced until the next OS input arrives — so the
        # coarse slab lingered until the user happened to move the mouse. The
        # robust fix is to wake the idle loop from ANOTHER thread: a background
        # debounce (threading.Timer) emits _lod_wake, whose cross-thread queued
        # delivery posts an event that interrupts the OS-level wait. Three paths
        # reach the rebuild — pointer-up (immediate), aboutToBlock (idle, when
        # it does fire), and the thread wake (guaranteed) — all coordinated by
        # _lod_pending so _lod_settle runs exactly once per interaction.
        self._lod_pending = False
        self._lod_due = None             # monotonic deadline for the rebuild
        self._lod_thread = None          # single reusable debounce worker
        self._lod_wake.connect(self._lod_settle)
        disp = QApplication.instance().eventDispatcher() if \
            QApplication.instance() is not None else None
        if disp is not None:
            disp.aboutToBlock.connect(self._on_about_to_block)
        self._loaded_uid = ""

        # drag state (rendercanvas pointer events)
        self._drag_btn = None
        self._last = (0.0, 0.0)
        self._cross_grab = False             # current drag is a crosshair grab
        self._cross_mode = "rotate"          # "rotate" | "move"
        self._cross_axis = None              # locked move axis (2-D unit)
        self._cross_ppt = (0.0, 0.0)         # prev world point (move mode)
        self._cross_prev = 0.0               # crosshair-rotate prev angle
        self._spin_prev = None               # SPIN previous cursor angle

        self.pane = {"A": _PygfxPane(), "B": _PygfxPane()}
        self._overlay = {"A": _Overlay(self, "A"), "B": _Overlay(self, "B")}

        # Each pane: bordered host holding the canvas + a QPainter overlay.
        self._frames = {}
        for key in ("A", "B"):
            self._frames[key] = _PaneHost(self.pane[key].canvas,
                                          self._overlay[key])
            self._wire_events(key)

        imgrow = QHBoxLayout()
        imgrow.setContentsMargins(0, 0, 0, 0)
        imgrow.setSpacing(2)
        imgrow.addWidget(self._frames["A"], 1)
        imgrow.addWidget(self._frames["B"], 1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self._build_toolbar())
        self._measure_bar = self._build_measure_bar()
        self._measure_bar.setVisible(False)
        lay.addWidget(self._measure_bar)
        lay.addLayout(imgrow, 1)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, False)
        for key in ("A", "B"):
            self.pane[key].canvas.setAttribute(
                Qt.WidgetAttribute.WA_InputMethodEnabled, False)
            # Mouse tracking so the measure draft preview (dashed line to the
            # cursor) follows the pointer without a button held.
            self.pane[key].canvas.setMouseTracking(True)
        # Tool shortcuts as QShortcuts (not keyPressEvent): like the XA/IVUS
        # cine keys, QShortcut accelerators are processed before the input
        # method, so they fire even with a Japanese IME active. Scoped to this
        # viewer (WidgetWithChildrenShortcut) so they only act while a CT pane
        # has focus. Q (tag overlay show/hide) is the shell's app-wide action.
        for seq, tool in (("Z", "ZOOM"), ("V", "MOVE"), ("R", "ROTATE"),
                          ("S", "SPIN"), ("G", "PAGING"), ("T", "THICK"),
                          ("W", "WL")):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda t=tool: self._set_tool(t))
        sc_c = QShortcut(QKeySequence("C"), self)
        sc_c.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_c.activated.connect(self._key_toggle_color)
        self._update_active_frames()

    # ------------------------------------------------------ event wiring
    def _wire_events(self, key):
        c = self.pane[key].canvas
        c.add_event_handler(lambda ev, k=key: self._on_down(k, ev), "pointer_down")
        c.add_event_handler(lambda ev, k=key: self._on_move(k, ev), "pointer_move")
        c.add_event_handler(lambda ev, k=key: self._on_up(k, ev), "pointer_up")
        c.add_event_handler(lambda ev, k=key: self._on_wheel(k, ev), "wheel")
        c.add_event_handler(lambda ev, k=key: self._on_dblclick(k, ev), "double_click")
        c.add_event_handler(lambda ev, k=key: self._on_resize(k, ev), "resize")

    def _on_down(self, key, ev):
        self._set_active(key)
        self._drag_btn = ev.get("button")
        x, y = ev["x"], ev["y"]
        self._last = (x, y)
        self._spin_prev = None
        if self._meas_on:
            self._cross_grab = False
            if self._drag_btn == 2:           # right-click: measure menu
                self._meas_drag = False
                self._measure_right(key, x, y)
                return
            started = self._measure_left(key, x, y)
            self._meas_drag = bool(started)
            return
        # Right-click (not measuring): force the slab to full quality. Manual
        # escape hatch for the rare case the coarse interactive LOD lingers
        # after wheel/trackpad paging (the idle Qt loop can fail to run the
        # auto crisp-up until the next input — see _lod_settle). Right-click is
        # otherwise unused here; on the Windows VTK viewer it does nothing, so
        # no behaviour or layout diverges between platforms.
        if self._drag_btn == 2:
            self._force_crisp()
            return
        # Pressing ON the crosshair grabs it (MOVE/ROTATE), overriding tool.
        self._cross_grab = (self._drag_btn == 1
                            and self._cross_press(key, x, y))

    def _on_move(self, key, ev):
        x, y = ev["x"], ev["y"]
        if self._meas_on:
            if self._meas_drag:
                self._measure_drag(key, x, y)
            elif self._draft and self._draft["pane"] == key:
                # Update the dashed draft preview that follows the cursor.
                self._meas_hover = self._disp_to_world(key, x, y)
                self._overlay[key].update()
            return
        if self._drag_btn != 1:               # left-drag drives tool/crosshair
            return
        if self._cross_grab:
            self._cross_move(key, x, y)
            self._last = (x, y)
            return
        dx, dy = x - self._last[0], y - self._last[1]
        self._last = (x, y)
        shift = "Shift" in (ev.get("modifiers") or ())
        self._drag(key, dx, dy, shift, x, y)

    def _on_up(self, key, ev):
        if self._meas_on and self._meas_drag:
            self._measure_release()
        self._meas_drag = False
        self._drag_btn = None
        self._cross_grab = False
        self._spin_prev = None
        # If an interactive (coarse) slab refresh is owed a quality upgrade, do
        # it NOW on release for the snappiest crisp-up (the idle backstop and
        # debounce timer would otherwise get it a moment later).
        if self._lod_pending:
            self._lod_settle()

    def _on_dblclick(self, key, ev):
        if self._meas_on:
            self._measure_finish_draft()
            return
        self._recenter(key, ev["x"], ev["y"])

    def _on_wheel(self, key, ev):
        # rendercanvas: wheel-up gives dy<0; page forward (+1) on wheel-up.
        self._wheel(key, 1 if ev["dy"] < 0 else -1)

    def _on_resize(self, key, ev):
        if self._vol is None:
            return
        if self._view_initial:
            self._fit_pane(key)
        else:
            self._config_cam(key)
        # Rebuild the slab-MIP image at the new size — otherwise the stale
        # QImage is stretched to the new rect and the aspect ratio distorts
        # (e.g. when the Measure bar shrinks the canvas height).
        if self._thick[key] > 0:
            self._mip_img[key] = self._build_slab_qimage(key)
        self.pane[key].render()
        self._overlay[key].update()

    # -- Bi / Lt / Rt --------------------------------------------------
    @property
    def supports_side(self) -> bool:
        return True

    def set_side(self, side: str, allow_dual: bool = True) -> None:
        self._frames["A"].setVisible(side != "Rt")
        self._frames["B"].setVisible(side != "Lt")

    # ------------------------------------------------------------ toolbar
    def _build_toolbar(self):
        row = QHBoxLayout()
        row.setContentsMargins(4, 2, 4, 2)
        self._tool_btns = {}
        for name in _TOOLS:
            b = QPushButton(name)
            b.setCheckable(True)
            b.clicked.connect(lambda _c, n=name: self._set_tool(n))
            self._tool_btns[name] = b
            row.addWidget(b)

        self._cl_btn = QPushButton("CenterLine")
        self._cl_btn.setCheckable(True)
        self._cl_btn.setChecked(True)
        self._cl_btn.setToolTip("Show/hide crosshair & slab lines")
        self._cl_btn.clicked.connect(self._toggle_centerline)
        row.addWidget(self._cl_btn)

        row.addWidget(QLabel("Slab:"))
        self._slab_spin = QDoubleSpinBox()
        self._slab_spin.setRange(0.0, 50.0)
        self._slab_spin.setSingleStep(0.5)
        self._slab_spin.setDecimals(1)
        self._slab_spin.valueChanged.connect(self._set_slab)
        row.addWidget(self._slab_spin)

        self._meas_btn = QPushButton("📏 Measure")
        self._meas_btn.setCheckable(True)
        self._meas_btn.setToolTip(
            "Measure on the image (Line / Polyline / Ellipse / Polygon / Angle)")
        self._meas_btn.clicked.connect(self._toggle_measure)
        row.addWidget(self._meas_btn)

        self._cmap_btn = QPushButton("ColorMap")
        self._cmap_btn.setCheckable(True)
        self._cmap_btn.clicked.connect(self._toggle_color)
        row.addWidget(self._cmap_btn)

        # Setting / Reset: a darker-grey, softer-text look so they read as
        # secondary controls, set apart from the tool buttons.
        _sr_qss = "QPushButton { background:#6e6e6e; color:#d8d8d8; }"
        setting = QPushButton("Setting")
        setting.setToolTip("HU colour-map settings (band colour, HU range, opacity)")
        setting.setStyleSheet(_sr_qss)
        setting.clicked.connect(self._open_setting)
        row.addWidget(setting)

        reset = QPushButton("Reset")
        reset.setStyleSheet(_sr_qss)
        reset.clicked.connect(self._reset)
        row.addWidget(reset)

        row.addWidget(QLabel("W/L:"))
        self._preset = QComboBox()
        self._preset.addItems(list(CT_WL_PRESETS.keys()))
        self._preset.currentTextChanged.connect(self._apply_preset)
        row.addWidget(self._preset)

        # DICOM Tags on the LEFT of the pair (kept always visible in the
        # scrollable strip); Measure History — less critical — sits to its
        # right. The tag-text-size slider is stacked above the Tags button.
        tags_box, self._tag_font_slider, tags = build_tag_font_control(
            TAG_FONT_PT_DEFAULT
        )
        tags.setToolTip(
            "Choose which DICOM tags overlay the image (key Q shows/hides)")
        tags.clicked.connect(self.tags_requested.emit)
        self._tag_font_slider.valueChanged.connect(self.overlay_font_changed.emit)
        row.addWidget(tags_box)

        hist = QPushButton("Measure History")
        hist.setToolTip("Show this study's measurement history")
        hist.clicked.connect(self.history_requested.emit)
        row.addWidget(hist)
        row.addStretch(1)
        self._set_tool("PAGING")

        # The CT pane is only half the window, so this many controls overflow
        # its width and Qt shrinks the buttons until the longest labels (e.g.
        # "Measure History") clip on both sides — worse on macOS, whose native
        # buttons reserve more horizontal padding. Pin every button to at least
        # its natural text width so labels never clip, and host the strip in a
        # horizontal scroll area so a narrow pane scrolls instead of squeezing.
        bar = QWidget()
        bar.setLayout(row)
        for b in bar.findChildren(QPushButton):
            b.setMinimumWidth(b.sizeHint().width())
        scroll = QScrollArea()
        scroll.setWidget(bar)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFixedHeight(bar.sizeHint().height() + 14)  # + scrollbar room
        return scroll

    def _set_tool(self, name):
        self._tool = name
        for n, b in self._tool_btns.items():
            b.setChecked(n == name)
            b.setStyleSheet("background:#c0392b;color:black;" if n == name else "")

    # ----------------------------------------------------- active pane
    def _set_active(self, which):
        self._active_pane = which
        self._sync_slab_spin()
        self._update_active_frames()

    def _update_active_frames(self):
        for key, f in self._frames.items():
            colr = "#ff2020" if key == self._active_pane else "transparent"
            f.setStyleSheet("QFrame#ctpane { border: 3px solid %s; }" % colr)

    def _sync_slab_spin(self):
        self._slab_spin.blockSignals(True)
        self._slab_spin.setValue(self._thick[self._active_pane])
        self._slab_spin.blockSignals(False)

    def _set_slab(self, mm):
        self._thick[self._active_pane] = float(mm)
        self._view_initial = False
        self._refresh()

    # --------------------------------------------------- AbstractViewer
    def load_series(self, loaded: LoadedSeries, title: str) -> None:
        new_uid = (
            loaded.series_uid
            or (str(getattr(loaded.header, "SeriesInstanceUID", ""))
                if loaded.header is not None else "")
        )
        if (self._vol is not None and new_uid
                and self._loaded_uid == new_uid):
            return
        self._loaded_uid = new_uid

        vol = np.ascontiguousarray(loaded.volume, dtype=np.float32)  # (z,y,x)
        self._vol = vol
        sr, sc = loaded.spacing_mm or (1.0, 1.0)
        sz = loaded.slice_mm or 1.0
        self._dims = (float(sc), float(sr), float(sz))   # x, y, z mm
        nz, ny, nx = vol.shape
        sx, sy, szz = self._dims
        # Bounds in mm (voxel centres span 0 .. (n-1)*spacing, origin 0) —
        # matches VTK vtkImageData.GetBounds for origin 0.
        self._bounds = (0.0, (nx - 1) * sx, 0.0, (ny - 1) * sy,
                        0.0, (nz - 1) * szz)
        self._header = loaded.header
        pb = loaded.patient_basis
        self._pbasis = (np.asarray(pb, dtype=np.float64)
                        if pb is not None else np.eye(3))
        self._win = self._win0 = float(loaded.window or 800.0)
        self._lvl = self._lvl0 = float(loaded.level or 200.0)
        self._thick = {"A": 0.0, "B": 5.0}

        b = self._bounds
        self._center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2,
                                 (b[4] + b[5]) / 2])
        self._center0 = self._center.copy()
        self._pc = {"A": self._center.copy(), "B": self._center.copy()}
        self._pan = {"A": np.zeros(2), "B": np.zeros(2)}
        self._roll = {"A": 0.0, "B": 0.0}
        self._init_frames()
        self._diag = math.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2
                               + (b[5] - b[4]) ** 2)
        self._half = self._diag / 2.0
        self._cam_off = max(2.0, 2.0 * self._diag)
        self._view_initial = True
        self._sync_slab_spin()

        for key in ("A", "B"):
            self.pane[key].set_volume(vol, self._dims)
        self._refresh(reset_cam=True)

    def clear(self) -> None:
        self._cancel_lod()
        self._lod_pending = False
        self._vol = None
        self._header = None
        self._measures = {"A": [], "B": []}
        self._metrics = {"A": [], "B": []}
        self._draft = None
        self._edit = None
        for key in ("A", "B"):
            p = self.pane[key]
            if p.mesh is not None:
                p.scene.remove(p.mesh)
                p.mesh = None
            p.render()
            self._overlay[key].update()

    def current_header(self):
        return self._header

    def set_tag_keywords(self, keywords) -> None:
        # The button only configures WHICH tags overlay; visibility is the Q
        # key's job alone, so don't touch _tags_on here.
        self._tag_keywords = list(keywords or [])
        self._refresh()

    def set_overlay_hidden(self, hidden: bool) -> None:
        """Show/hide the DICOM tag overlay. Driven by the shell's app-wide
        'Hide DICOM overlay' action (the Q key), so it works regardless of
        input-method state. The DICOM Tags button only edits WHICH tags."""
        self._tags_on = not bool(hidden)
        for k in ("A", "B"):
            self._overlay[k].update()

    def set_anonymized(self, on: bool) -> None:
        self._anon = bool(on)
        self._refresh()

    # ------------------------------------------------------- geometry
    def _init_frames(self):
        self._frame = {
            "A": (np.array([1.0, 0.0, 0.0]),
                  np.array([0.0, 1.0, 0.0]),
                  np.array([0.0, 0.0, 1.0])),
            "B": (np.array([1.0, 0.0, 0.0]),
                  np.array([0.0, 0.0, 1.0]),
                  np.array([0.0, 1.0, 0.0])),
        }
        self._cross_ang = {"A": 0.0, "B": 0.0}

    def _ortho(self, u, v):
        u = _norm(u)
        v = _norm(v - np.dot(v, u) * u)
        n = np.cross(u, v)
        return (u, v, n)

    def _axes_for(self, key):
        return self._frame[key]

    def _cc(self, key):
        """Crosshair centre (C projected into the pane plane, relative to
        the pane's reslice centre) — output coords mm. Used by overlays."""
        u, v, _n = self._axes_for(key)
        delta = self._center - self._pc[key]
        return float(np.dot(delta, u)), float(np.dot(delta, v))

    def _content_half_on_plane(self, key):
        """Half-widths (hu, hv) in the pane's (u,v) axes from _pc[key]
        needed to contain the volume's 8 corners projected onto the plane."""
        b = self._bounds
        u, v, _n = self._frame[key]
        pc = self._pc[key]
        hu = hv = 0.0
        for ix in (0, 1):
            for iy in (2, 3):
                for iz in (4, 5):
                    p = np.array([b[ix], b[iy], b[iz]], float) - pc
                    hu = max(hu, abs(float(np.dot(p, u))))
                    hv = max(hv, abs(float(np.dot(p, v))))
        return hu, hv

    # --------------------------------------------------------- render
    def _config_cam(self, key):
        """Orient/position the pane camera from the current frame, pan and
        zoom so the slice is shown face-on with U=right, V=up.

        The camera's view-back axis is cross(U,V), NOT the frame normal N:
        pane B's init frame is left-handed (U×V = -N), so a matrix with
        columns (U,V,N) would be a reflection (det -1) and quat_from_mat would
        produce a bad orientation (pane B rendered blank). cross(U,V) is always
        the proper right-handed third axis (= ±N); both give the same plane and
        the same U-right/V-up image, matching the VTK reslice output."""
        p = self.pane[key]
        ur, vr, w, focal = self._cam_basis(key)
        R = np.column_stack([ur, vr, w]).astype(np.float64)
        p.cam.local.rotation = la.quat_from_mat(R)
        p.cam.local.position = focal + w * self._cam_off
        ps = max(1e-3, self._ps[key])
        pw = max(1, p.canvas.width())
        ph = max(1, p.canvas.height())
        p.cam.height = 2.0 * ps
        p.cam.width = 2.0 * ps * (pw / ph)
        p.cam.depth_range = (0.1, 4.0 * self._cam_off + self._diag)

    def _cam_basis(self, key):
        """Camera right/up/back axes (3-D world) and focal point. Right/up are
        the in-plane (u,v) rotated by the SPIN roll; back = cross(u,v)."""
        u, v, _n = self._frame[key]
        w = _norm(np.cross(u, v))
        a = math.radians(self._roll[key])
        ca, sa = math.cos(a), math.sin(a)
        ur = ca * u + sa * v
        vr = -sa * u + ca * v
        px, py = self._pan[key]
        focal = self._pc[key] + px * u + py * v
        return ur, vr, w, focal

    # --------------------------------------------------- world<->screen
    def _scale_px(self, key):
        """Pixels per world-mm for a pane (ParallelScale → viewport px)."""
        ph = max(1, self.pane[key].canvas.height())
        return ph / (2.0 * max(1e-3, self._ps[key]))

    def _world_to_screen(self, key, wx, wy):
        """Pane output coords (wx,wy) in the unrolled (u,v) basis → widget px
        (y down). Inverse of _disp_to_world."""
        a = math.radians(self._roll[key])
        ca, sa = math.cos(a), math.sin(a)
        px, py = self._pan[key]
        dxo, dyo = wx - px, wy - py
        aa = dxo * ca + dyo * sa              # along camera right (ur)
        bb = -dxo * sa + dyo * ca             # along camera up (vr)
        s = self._scale_px(key)
        pw = max(1, self.pane[key].canvas.width())
        ph = max(1, self.pane[key].canvas.height())
        return pw / 2.0 + aa * s, ph / 2.0 - bb * s

    def _disp_to_world(self, key, sx, sy):
        """Widget px (sx,sy) → pane output coords (wx,wy) in the (u,v) basis."""
        s = self._scale_px(key)
        pw = max(1, self.pane[key].canvas.width())
        ph = max(1, self.pane[key].canvas.height())
        aa = (sx - pw / 2.0) / s
        bb = (ph / 2.0 - sy) / s
        a = math.radians(self._roll[key])
        ca, sa = math.cos(a), math.sin(a)
        dxo = aa * ca - bb * sa
        dyo = aa * sa + bb * ca
        px, py = self._pan[key]
        return px + dxo, py + dyo

    def _screen_center(self, key):
        """Widget px of the crosshair centre (output (ccx,ccy))."""
        ccx, ccy = self._cc(key)
        return self._world_to_screen(key, ccx, ccy)

    # ----------------------------------------------------- slab-MIP (CPU)
    def _slab_mip_hu(self, key, iw, ih, max_planes=_SLAB_PLANES_FULL):
        """(ih,iw) HU array = max over N parallel oblique planes within
        ±thick/2 of the pane plane, sampled across the current viewport.

        The sample grid is built in SCREEN space and unprojected with the same
        roll+pan as _disp_to_world, so the slab image rotates with SPIN/ROTATE
        and stays pixel-aligned with the crosshair (the image is drawn
        full-rect by the overlay). The slab depth direction is the frame
        normal N."""
        u, v, n = self._frame[key]
        pc = self._pc[key]
        sx, sy, sz = self._dims
        px, py = self._pan[key]
        pw = max(1, self.pane[key].canvas.width())
        ph = max(1, self.pane[key].canvas.height())
        scale = ph / (2.0 * max(1e-3, self._ps[key]))
        a = math.radians(self._roll[key])
        ca, sa = math.cos(a), math.sin(a)
        SX, SY = np.meshgrid(np.linspace(0.0, pw, iw),
                             np.linspace(0.0, ph, ih))
        aa = (SX - pw / 2.0) / scale
        bb = (ph / 2.0 - SY) / scale
        WX = px + aa * ca - bb * sa          # screen → output (roll/pan)
        WY = py + aa * sa + bb * ca
        bx = pc[0] + WX * u[0] + WY * v[0]
        by = pc[1] + WX * u[1] + WY * v[1]
        bz = pc[2] + WX * u[2] + WY * v[2]
        th = self._thick[key]
        step = max(1e-3, min(self._dims))
        nplanes = int(max(1, min(max_planes, round(th / step))))
        offs = (np.linspace(-th / 2.0, th / 2.0, nplanes)
                if nplanes > 1 else np.array([0.0]))
        mip = np.full((ih, iw), -np.inf, np.float32)
        for t in offs:
            hu = _trilinear_sample(self._vol, (bx + t * n[0]) / sx,
                                   (by + t * n[1]) / sy, (bz + t * n[2]) / sz)
            mip = np.maximum(mip, hu.astype(np.float32))
        return mip

    def _build_slab_qimage(self, key, lod=False):
        """Render the slab MIP for a pane to a viewport-filling RGB QImage
        (W/L or HU colormap applied CPU-side). *lod*=True builds a coarser
        image (fewer columns + MIP planes) for smooth interactive drag/page."""
        pane = self.pane[key]
        pw = max(1, pane.canvas.width())
        ph = max(1, pane.canvas.height())
        iw = min(pw, _SLAB_IW_LOD if lod else _SLAB_IW_FULL)
        ih = max(1, int(round(iw * ph / pw)))
        mip = self._slab_mip_hu(
            key, iw, ih,
            max_planes=_SLAB_PLANES_LOD if lod else _SLAB_PLANES_FULL)
        if self._color:
            lut = _band_lut_array(self._bands, self._opacity,
                                  self._win, self._lvl)        # (512,4)
            idx = np.clip((mip - _HU_LO) / (_HU_HI - _HU_LO) * 511.0,
                          0, 511).astype(np.int32)
            rgb = (lut[idx, :3] * 255.0).astype(np.uint8)       # (ih,iw,3)
        else:
            g = np.clip((mip - (self._lvl - self._win / 2.0))
                        / max(1e-6, self._win), 0.0, 1.0)
            gg = (g * 255.0).astype(np.uint8)
            rgb = np.stack([gg, gg, gg], axis=2)
        rgb = np.ascontiguousarray(rgb)
        return QImage(rgb.data, iw, ih, 3 * iw,
                      QImage.Format.Format_RGB888).copy()

    def _fit_pane(self, key):
        """Fit the volume content (projected onto the plane) to the viewport
        and record the resulting half-height (ParallelScale equivalent)."""
        p = self.pane[key]
        hu, hv = self._content_half_on_plane(key)
        pw = max(1, p.canvas.width())
        ph = max(1, p.canvas.height())
        self._ps[key] = max(1e-3, max(hv, hu * ph / pw))
        self._pan[key] = np.zeros(2)
        self._config_cam(key)

    def set_overlay_font_pt(self, pt: int) -> None:
        """Apply the shared DICOM-tag / readout text size (pt), sync the slider
        and repaint the overlays. Called by the shell."""
        pt = int(pt)
        self._overlay_font_pt = pt
        sl = getattr(self, "_tag_font_slider", None)
        if sl is not None and sl.value() != pt:
            sl.blockSignals(True)
            sl.setValue(pt)
            sl.blockSignals(False)
        for k in ("A", "B"):
            self._overlay[k].update()

    def _refresh(self, reset_cam=False, lod=False):
        # lod=True is an INTERACTIVE refresh (drag / wheel-page): the CPU slab-
        # MIP is built at REDUCED quality (coarse columns + fewer planes) rather
        # than skipped, so the THICK image keeps its slab look while staying
        # smooth on low-memory Macs. The debounce timer then rebuilds it at full
        # quality once the interaction settles.
        if self._vol is None:
            return
        for key in ("A", "B"):
            p = self.pane[key]
            if p.material is None:
                continue
            u, v, n = self._frame[key]
            pc = self._pc[key]
            p.material.plane = (float(n[0]), float(n[1]), float(n[2]),
                                float(-np.dot(n, pc)))
            if self._color:
                # Colormap bakes W/L into the LUT over [_HU_LO,_HU_HI]; clim
                # maps that HU span to the LUT's 0..1 domain.
                p.material.clim = (_HU_LO, _HU_HI)
                p.material.map = self._lut_texture()
            else:
                p.material.map = None
                p.material.clim = (self._lvl - self._win / 2.0,
                                   self._lvl + self._win / 2.0)
            if reset_cam:
                self._fit_pane(key)
            else:
                self._config_cam(key)
            # Slab-MIP (THICK): clipping planes can't bound a GPU MIP slab
            # (de-risk spike), so when thick>0 we hide the GPU slice and paint
            # a CPU max-over-N-oblique-planes image in the overlay instead.
            # During an interactive (lod) refresh we still paint the slab, just
            # at reduced quality, so the thickness never disappears mid-drag;
            # full quality returns when the drag settles (debounce timer).
            if self._thick[key] > 0:
                p.mesh.visible = False
                self._mip_img[key] = self._build_slab_qimage(key, lod=lod)
            else:
                p.mesh.visible = True
                self._mip_img[key] = None
            p.render()
            self._overlay[key].update()
        # (Re)arm the high-quality rebuild on interactive refreshes; a full
        # refresh has already painted the slab MIP, so cancel any pending one.
        if lod and any(self._thick[k] > 0 for k in ("A", "B")):
            self._lod_pending = True
            self._arm_lod()
        else:
            self._lod_pending = False
            self._cancel_lod()

    def _arm_lod(self):
        """(Re)arm the background debounce: push the rebuild deadline ~160ms out
        and make sure the worker is running. The worker emits _lod_wake once the
        deadline passes; the signal's cross-thread delivery wakes the idle GUI
        loop so _lod_settle runs even when no further input arrives (the
        wheel/trackpad lingering bug). One reusable worker that polls the
        deadline — not a new thread per mouse-move — keeps drag churn-free."""
        self._lod_due = time.monotonic() + 0.16
        t = self._lod_thread
        if t is None or not t.is_alive():
            t = threading.Thread(target=self._lod_worker, daemon=True)
            self._lod_thread = t
            t.start()

    def _cancel_lod(self):
        # Drop the deadline; the worker re-checks it (and _lod_pending, which the
        # caller has cleared) before firing and exits on its own.
        self._lod_due = None

    def _lod_worker(self):
        while True:
            due = self._lod_due
            if due is None or not self._lod_pending:
                return
            now = time.monotonic()
            if now >= due:
                if self._lod_pending:
                    self._lod_wake.emit()
                return
            time.sleep(min(0.04, due - now))

    def _on_about_to_block(self):
        """Qt fires this right before the event loop blocks (goes idle). If an
        interactive coarse slab is owed a full-quality rebuild, do it now so the
        crisp-up is instant when it fires; the background thread wake is the
        guaranteed backstop for when it (or the loop) stays parked."""
        if self._lod_pending and self._vol is not None:
            self._lod_settle()

    def _lod_settle(self):
        """Rebuild the slab at full quality and force a SYNCHRONOUS repaint.
        Runs at most once per interaction (guarded by _lod_pending), whichever
        path gets there first: pointer-up, aboutToBlock (idle), or the
        background thread wake. repaint() flushes immediately so the crisp image
        shows without waiting for the next input event to pump the loop."""
        if not self._lod_pending:
            return
        self._cancel_lod()
        self._refresh(lod=False)        # clears _lod_pending (non-lod branch)
        for k in ("A", "B"):
            self._overlay[k].repaint()

    def _force_crisp(self):
        """Unconditionally rebuild the slab at full quality and repaint now.
        Bound to right-click (Mac only) as a manual override when the coarse
        interactive LOD lingers; safe no-op feel on a thin (non-slab) view."""
        if self._vol is None:
            return
        self._cancel_lod()
        self._lod_pending = False
        self._refresh(lod=False)
        for k in ("A", "B"):
            self._overlay[k].repaint()

    # ----------------------------------------------------------- tools
    def _drag(self, which, dx, dy, shift=False, sx=None, sy=None):
        if self._vol is None:
            return
        t = self._tool
        if t != "WL":
            self._view_initial = False
        if t == "WL":
            self._win = max(1.0, self._win + dx * 2.0)
            self._lvl = self._lvl - dy * 2.0
        elif t == "PAGING":
            _, _, n = self._axes_for(which)
            # Mouse UP (dy<0) always moves toward the green ▲ apex of the other
            # pane's crossline, mouse DOWN toward its base — independent of the
            # pane's 3-D orientation (see _paging_sign).
            mv = n * (-dy) * self._paging_sign(which) * min(self._dims)
            # Page only THIS pane (its image scrolls through slices). The
            # OTHER pane's image stays put; its centreline slides to mark the
            # new slice position (only _center moves, not _pc[other]).
            self._center = self._center + mv
            self._pc[which] = self._pc[which] + mv
            self._clamp_center()
        elif t == "THICK":
            self._thick[which] = max(0.0, self._thick[which] + (dx - dy) * 0.3)
            if which == self._active_pane:
                self._sync_slab_spin()
        elif t == "ROTATE":
            u, v, n = self._frame[which]
            u = _rotate(_rotate(u, v, -dx * 0.5), u, -dy * 0.5)
            v = _rotate(_rotate(v, v, -dx * 0.5), u, -dy * 0.5)
            self._frame[which] = self._ortho(u, v)
            # Linked: re-derive the OTHER pane as the orthogonal section that
            # shares this pane's horizontal axis and contains its normal, so
            # the two panes stay coupled (product behaviour, unlike the VTK
            # viewer where ROTATE left the other pane unchanged).
            uw, _vw, nw = self._frame[which]
            other = "B" if which == "A" else "A"
            self._frame[other] = self._ortho(uw, nw)
            self._cross_ang[which] = 0.0
            self._cross_ang[other] = 0.0
            self._pc = {"A": self._center.copy(), "B": self._center.copy()}
        elif t == "ZOOM":
            factor = 1.0 - dy * 0.005
            for k in (("A", "B") if shift else (which,)):
                self._ps[k] = max(1e-3, self._ps[k] * factor)
        elif t == "MOVE":
            sc = self._ps[which] * 0.003
            px, py = self._pan[which]
            self._pan[which] = np.array([px - dx * sc, py + dy * sc])
        elif t == "SPIN":
            # Roll the camera by how far the cursor sweeps about the crosshair
            # centre (screen px, y-down) — image AND overlay rotate together.
            if sx is not None:
                cx, cy = self._screen_center(which)
                rx, ry = sx - cx, sy - cy
                if abs(rx) > 1e-3 or abs(ry) > 1e-3:
                    phi = math.atan2(ry, rx)
                    if self._spin_prev is None:
                        self._spin_prev = phi
                    else:
                        dphi = math.degrees(phi - self._spin_prev)
                        dphi = (dphi + 180.0) % 360.0 - 180.0
                        self._spin_prev = phi
                        self._roll[which] += _SPIN_SIGN * dphi
        # Build a reduced-quality slab MIP mid-drag for smoothness (the thick
        # image keeps its look; full quality returns when the drag settles).
        # THICK included: the coarse slab still updates live as it's adjusted.
        self._refresh(lod=True)

    def _wheel(self, which, delta):
        if self._vol is None:
            return
        _, _, n = self._axes_for(which)
        # Wheel up = toward the ▲ apex (same convention as drag-paging).
        d = 1.0 if delta > 0 else -1.0
        mv = n * d * self._paging_sign(which) * min(self._dims)
        self._center = self._center + mv
        self._pc[which] = self._pc[which] + mv     # page only this pane
        self._clamp_center()
        self._view_initial = False
        self._refresh(lod=True)            # smooth wheel-paging (slab MIP defers)

    def _paging_sign(self, which):
        """+1/-1 so that moving _center by +n advances the OTHER pane's
        crossline toward its green ▲ apex. The apex points +uv (perpendicular
        to that pane's horizontal crossline); we project this pane's normal n
        onto that apex direction so the up/down feel stays constant at any
        oblique orientation."""
        n = self._frame[which][2]
        other = "B" if which == "A" else "A"
        u_o, v_o, _n_o = self._frame[other]
        a = math.radians(self._cross_ang[other])
        apex = u_o * (-math.sin(a)) + v_o * math.cos(a)   # +uv of the other pane
        proj = float(np.dot(n, apex))
        return 1.0 if proj >= 0 else -1.0

    def _clamp_center(self):
        b = self._bounds
        self._center = np.array([
            min(max(self._center[0], b[0]), b[1]),
            min(max(self._center[1], b[2]), b[3]),
            min(max(self._center[2], b[4]), b[5]),
        ])

    # ----------------------------------------------------- crosshair
    def _cross_press(self, which, sx, sy) -> bool:
        """True (and arm a MOVE/ROTATE gesture) if the press is on the
        crosshair. Near the intersection → MOVE; on a line → ROTATE."""
        if self._vol is None:
            return False
        wx, wy = self._disp_to_world(which, sx, sy)
        ccx, ccy = self._cc(which)
        rx, ry = wx - ccx, wy - ccy
        a = math.radians(self._cross_ang[which])
        uh = (math.cos(a), math.sin(a))
        uv = (-math.sin(a), math.cos(a))
        tol = max(3.0, 0.02 * self._half)
        ctol = max(6.0, 0.06 * self._half)
        if math.hypot(rx, ry) < ctol:
            self._cross_mode = "move"
            self._cross_axis = None
            self._cross_ppt = (wx, wy)
            return True
        d_to_h = abs(rx * uv[0] + ry * uv[1])
        d_to_v = abs(rx * uh[0] + ry * uh[1])
        if d_to_h < tol or d_to_v < tol:
            self._cross_mode = "rotate"
            self._cross_prev = math.atan2(ry, rx)
            return True
        return False

    def _cross_move(self, which, sx, sy):
        wx, wy = self._disp_to_world(which, sx, sy)
        u, v, n = self._frame[which]
        other = "B" if which == "A" else "A"
        if self._cross_mode == "move":
            a = math.radians(self._cross_ang[which])
            uh = np.array([math.cos(a), math.sin(a)])
            uv = np.array([-math.sin(a), math.cos(a)])
            d2 = np.array([wx - self._cross_ppt[0], wy - self._cross_ppt[1]])
            self._cross_ppt = (wx, wy)
            if self._cross_axis is None:
                self._cross_axis = (uh if abs(np.dot(d2, uh))
                                    >= abs(np.dot(d2, uv)) else uv)
            amt = float(np.dot(d2, self._cross_axis))
            dir3 = u * self._cross_axis[0] + v * self._cross_axis[1]
            self._center = self._center + amt * dir3
            self._clamp_center()
            self._pc[other] = self._center.copy()
            self._view_initial = False
            self._refresh(lod=True)            # coarse slab while dragging
            return
        # ROTATE: crosshair follows the cursor; other pane re-derived.
        ccx, ccy = self._cc(which)
        rx, ry = wx - ccx, wy - ccy
        if abs(rx) < 1e-6 and abs(ry) < 1e-6:
            return
        cur = math.atan2(ry, rx)
        d = math.degrees(cur - self._cross_prev)
        d = (d + 180.0) % 360.0 - 180.0
        self._cross_prev = cur
        self._cross_ang[which] += d
        a = math.radians(self._cross_ang[which])
        crossdir = u * math.cos(a) + v * math.sin(a)
        self._frame[other] = self._ortho(crossdir, n)
        self._cross_ang[other] = 0.0
        self._pc[other] = self._center.copy()
        self._view_initial = False
        self._refresh(lod=True)                # coarse slab while dragging

    def _recenter(self, which, sx, sy):
        """Double-click: clicked point becomes the CrossLine centre AND the
        image centre in both panes."""
        if self._vol is None:
            return
        wx, wy = self._disp_to_world(which, sx, sy)
        u, v, _n = self._frame[which]
        self._center = self._pc[which] + wx * u + wy * v
        self._clamp_center()
        self._pc = {"A": self._center.copy(), "B": self._center.copy()}
        self._pan = {"A": np.zeros(2), "B": np.zeros(2)}
        self._view_initial = False
        self._refresh()

    def _angio_angle(self, key) -> str:
        """C-arm angle (LAO/RAO·CRA/CAU) of this pane's projection direction.
        Copied verbatim from the VTK viewer (pure numpy)."""
        n = np.asarray(self._frame[key][2], dtype=np.float64)
        nrm = float(np.linalg.norm(n))
        if nrm < 1e-9:
            return ""
        n = self._pbasis @ (n / nrm)
        nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
        if (ny > 1e-9
                or (abs(ny) <= 1e-9 and nz < -1e-9)
                or (abs(ny) <= 1e-9 and abs(nz) <= 1e-9 and nx < 0)):
            nx, ny, nz = -nx, -ny, -nz
        axial = math.hypot(nx, ny)
        prim = 0.0 if axial < 1e-9 else math.degrees(math.atan2(nx, -ny))
        sec = math.degrees(math.atan2(nz, axial))
        pi_, si_ = int(round(prim)), int(round(sec))
        lao = f"LAO{pi_}" if pi_ >= 0 else f"RAO{-pi_}"
        cra = f"CRA{si_}" if si_ >= 0 else f"CAU{-si_}"
        return f"{lao} {cra}"

    def _hu_stats(self, key, pts):
        """(min, max) HU inside the polygon *pts* (pane output coords) on the
        pane's oblique plane, by trilinearly sampling the source volume on a
        grid — replaces the VTK reslice-output readback. Runs on measure
        finalise/edit only."""
        if self._vol is None or len(pts) < 3:
            return 0.0, 0.0
        u, v, _n = self._frame[key]
        pc = self._pc[key]
        sx, sy, sz = self._dims
        arr = np.asarray(pts, float)
        xmn, ymn = arr.min(0)
        xmx, ymx = arr.max(0)
        diag = math.hypot(xmx - xmn, ymx - ymn)
        if diag < 1e-6:
            return 0.0, 0.0
        step = max(min(self._dims), diag / 160.0)   # cap ~160 samples/side
        gx = np.arange(xmn, xmx + step, step)
        gy = np.arange(ymn, ymx + step, step)
        WX, WY = np.meshgrid(gx, gy)
        fx_ = WX.ravel()
        fy_ = WY.ravel()
        inside = np.fromiter(
            (_point_in_poly(px, py, pts) for px, py in zip(fx_, fy_)),
            dtype=bool, count=fx_.size)
        if not inside.any():
            return 0.0, 0.0
        wx = fx_[inside]
        wy = fy_[inside]
        X = pc[0] + wx * u[0] + wy * v[0]
        Y = pc[1] + wx * u[1] + wy * v[1]
        Z = pc[2] + wx * u[2] + wy * v[2]
        hu = _trilinear_sample(self._vol, X / sx, Y / sy, Z / sz)
        return float(np.min(hu)), float(np.max(hu))

    # ------------------------------------------------------- measurements
    def _build_measure_bar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 6, 2)
        row.addWidget(QLabel("Measure:"))
        self._meas_btns = {}
        for label, key in (("Line", "line"), ("Polyline", "polyline"),
                           ("Ellipse", "ellipse"), ("Polygon", "polygon"),
                           ("Angle", "angle")):
            b = QPushButton(label)
            b.setCheckable(True)
            b.clicked.connect(lambda _c, k=key: self._set_measure_type(k))
            self._meas_btns[key] = b
            row.addWidget(b)
        clr = QPushButton("Clear")
        clr.clicked.connect(self._measure_clear)
        row.addWidget(clr)
        row.addWidget(QLabel("  Left-click = add point /"
                             " right-click finishes Polyline / Polygon"))
        row.addStretch(1)
        return bar

    def _toggle_measure(self):
        self._meas_on = self._meas_btn.isChecked()
        self._measure_bar.setVisible(self._meas_on)
        self._draft = None
        self._edit = None
        self._meas_hover = None
        if not self._meas_on:
            self._meas_type = None
            for b in self._meas_btns.values():
                b.setChecked(False)
                b.setStyleSheet("")
        for k in ("A", "B"):
            self._overlay[k].update()

    def _set_measure_type(self, key):
        self._meas_type = key
        self._draft = None
        self._meas_hover = None
        for k, b in self._meas_btns.items():
            b.setChecked(k == key)
            b.setStyleSheet("background:#1f77b4;color:black;" if k == key else "")

    def _measure_clear(self):
        self._measures = {"A": [], "B": []}
        self._draft = None
        self._edit = None
        self._meas_hover = None
        for k in ("A", "B"):
            self._redraw_meas(k)

    # ---- per-measure geometry ----
    def _ellipse_cab(self, m):
        return _ellipse_cab(m["pts"])

    def _outline(self, m):
        t = m["type"]
        if t == "line":
            return list(m["pts"][:2])
        if t == "polyline":
            pts = list(m["pts"])
            return _smooth_open(pts) if m.get("smooth") else pts
        if t == "polygon":
            return _smooth_closed(m["pts"])
        if t == "angle":
            # Vertex = MIDDLE point: endpoint1 → vertex → endpoint2.
            return list(m["pts"][:3])
        return _ellipse_outline(m["pts"])          # oblique ellipse

    def _handles(self, m):
        return list(m["pts"])

    def _anchor(self, m):
        if m["type"] == "ellipse":
            cx, cy, _a, _b = self._ellipse_cab(m)
            return (cx, cy)
        if m["type"] == "polygon":
            xs = [q[0] for q in m["pts"]]
            ys = [q[1] for q in m["pts"]]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        if m["type"] == "angle":
            return m["pts"][1]                        # label at vertex (middle)
        return m["pts"][0]

    def _shape_center(self, m):
        # Center-Angle apex = the physical (area) centroid of the region, not
        # the vertex mean (which skews toward clustered vertices).
        if m["type"] == "polygon":
            return _polygon_centroid(m["pts"])
        return self._anchor(m)

    def _metrics_text(self, key, m):
        t = m["type"]
        pts = m["pts"]
        ca = m.get("center_angle")
        ca_str = (f"  CenterAngle:{ca['angle']:.1f}°"
                  if ca and "angle" in ca else "")
        if t == "line":
            return f"#{m['id']} Line: {_dist(pts[0], pts[1]):.1f} mm"
        if t == "polyline":
            L = sum(_dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
            tag = "Polyline (Spline)" if m.get("smooth") else "Polyline"
            return f"#{m['id']} {tag}: {L:.1f} mm"
        if t == "angle":
            return f"#{m['id']} Angle: {_angle_at(pts[1], pts[0], pts[2]):.1f}°"
        if t == "ellipse":
            cx, cy, a, b = self._ellipse_cab(m)
            lo, hi = self._hu_stats(key, self._outline(m))
            return (f"#{m['id']} Ellipse  Area:{math.pi*a*b:.1f}mm²  "
                    f"CTmax:{hi:.0f}  CTmin:{lo:.0f}  "
                    f"Dmax:{2*max(a,b):.1f}  Dmin:{2*min(a,b):.1f}mm{ca_str}")
        lo, hi = self._hu_stats(key, pts + [pts[0]])
        _, _, dmax, dmin = _major_minor(m)
        return (f"#{m['id']} Polygon  Area:{_poly_area(pts):.1f}mm²  "
                f"CTmax:{hi:.0f}  CTmin:{lo:.0f}  "
                f"Dmax:{dmax:.1f}  Dmin:{dmin:.1f}mm{ca_str}")

    # ---- drawing (the overlay paints; these just trigger repaint) ----
    def _redraw_geom(self, key):
        self._overlay[key].update()

    def _redraw_meas(self, key):
        self._metrics[key] = [self._metrics_text(key, m)
                              for m in self._measures[key]]
        self._overlay[key].update()

    # ---- picking ----
    def _pick_handle(self, which, sx, sy):
        for mi in range(len(self._measures[which]) - 1, -1, -1):
            m = self._measures[which][mi]
            for vi, q in enumerate(m["pts"]):
                qx, qy = self._world_to_screen(which, q[0], q[1])
                if math.hypot(qx - sx, qy - sy) < 12.0:
                    return mi, vi
        return None

    def _pick_center_angle(self, which, sx, sy):
        """Pick a finalized Center-Angle marker point (the orange dots) so it
        can be dragged like a polygon vertex. Returns (mi, ci) or None."""
        for mi in range(len(self._measures[which]) - 1, -1, -1):
            ca = self._measures[which][mi].get("center_angle")
            if not ca or not ca.get("pts"):
                continue
            for ci, q in enumerate(ca["pts"]):
                qx, qy = self._world_to_screen(which, q[0], q[1])
                if math.hypot(qx - sx, qy - sy) < 12.0:
                    return mi, ci
        return None

    def _pick_measure(self, which, sx, sy):
        wx, wy = self._disp_to_world(which, sx, sy)
        tol = max(3.0, 0.02 * self._half)
        best, bi = tol, None
        for mi, m in enumerate(self._measures[which]):
            ol = self._outline(m)
            for i in range(len(ol) - 1):
                d = _seg_dist(wx, wy, ol[i], ol[i + 1])
                if d < best:
                    best, bi = d, mi
        return bi

    # ---- interaction ----
    def _measure_left(self, which, sx, sy) -> bool:
        if not self._meas_on:
            return False
        cat = self._center_angle_target
        if cat and cat.get("key") == which:
            self._center_angle_add(self._disp_to_world(which, sx, sy))
            return False
        hit = self._pick_handle(which, sx, sy)
        if hit is not None:
            self._edit = {"key": which, "mi": hit[0], "vi": hit[1]}
            self._redraw_geom(which)
            return True
        # A Center-Angle marker point can be dragged just like a polygon vertex.
        ca_hit = self._pick_center_angle(which, sx, sy)
        if ca_hit is not None:
            self._edit = {"key": which, "mi": ca_hit[0], "vi": ca_hit[1],
                          "ca": True}
            self._redraw_geom(which)
            return True
        if not self._meas_type:
            return False
        w = self._disp_to_world(which, sx, sy)
        d = self._draft
        if d is None or d["pane"] != which or d["type"] != self._meas_type:
            d = {"type": self._meas_type, "pane": which, "pts": []}
            self._draft = d
        d["pts"].append(w)
        if d["type"] in ("line", "ellipse") and len(d["pts"]) >= 2:
            self._commit_draft()
        elif d["type"] == "angle" and len(d["pts"]) >= 3:
            self._commit_draft()
        else:
            self._redraw_geom(which)
        return False

    def _measure_drag(self, which, sx, sy):
        e = self._edit
        if not e:
            return
        w = self._disp_to_world(e["key"], sx, sy)
        m = self._measures[e["key"]][e["mi"]]
        if e.get("ca"):
            self._set_center_angle_point(m, e["vi"], w)
        elif m["type"] == "ellipse":
            self._set_ellipse_handle(m, e["vi"], w)
            self._resnap_center_angle(m)
        else:
            m["pts"][e["vi"]] = w
            self._resnap_center_angle(m)
        self._redraw_geom(e["key"])

    def _resnap_center_angle(self, m):
        """After the shape itself changes (a vertex / ellipse-handle drag),
        pull each Center-Angle marker back onto the new outline and recompute
        the angle, so a finalized Center Angle tracks the reshaped polygon."""
        ca = m.get("center_angle")
        if not ca or len(ca.get("pts", [])) < 3:
            return
        pts = [self._snap_ca(m, q) for q in ca["pts"]]
        centre = self._shape_center(m)
        span, t1, t3, ccw = _central_arc_angle(centre, pts[0], pts[1], pts[2])
        m["center_angle"] = {"pts": pts, "angle": span,
                             "t1": t1, "t3": t3, "ccw": ccw}

    def _snap_ca(self, m, w):
        """Constrain a Center-Angle marker to the measure's drawn outline, so
        the three points always sit ON the polygon/ellipse line."""
        return _project_to_polyline(w, self._outline(m))

    def _set_center_angle_point(self, m, ci, w):
        """Move one Center-Angle marker point (snapped to the outline) and
        recompute the angle from the shape centre, mirroring how polygon-vertex
        drags update live."""
        ca = m.get("center_angle")
        if not ca or "pts" not in ca or not (0 <= ci < len(ca["pts"])):
            return
        pts = list(ca["pts"])
        pts[ci] = self._snap_ca(m, w)
        centre = self._shape_center(m)
        span, t1, t3, ccw = _central_arc_angle(centre, pts[0], pts[1], pts[2])
        m["center_angle"] = {"pts": pts, "angle": span,
                             "t1": t1, "t3": t3, "ccw": ccw}

    def _measure_release(self):
        if self._edit:
            key = self._edit["key"]
            self._edit = None
            self._redraw_meas(key)

    def _set_ellipse_handle(self, m, vi, w):
        m["pts"] = _ellipse_drag(m["pts"], vi, w)

    def _commit_draft(self):
        d = self._draft
        self._draft = None
        self._meas_hover = None
        if d is None or len(d["pts"]) < 2:
            return
        if d["type"] == "ellipse":
            # The two clicked points are the MAJOR-axis endpoints; the minor
            # axis starts at half the major and is then dragged to taste.
            pts = _ellipse_from_major(d["pts"][0], d["pts"][1])
        elif d["type"] == "line":
            pts = d["pts"][:2]
        else:
            pts = list(d["pts"])
        self._meas_seq += 1
        m = {"id": self._meas_seq, "type": d["type"], "pts": pts}
        self._measures[d["pane"]].append(m)
        self._redraw_meas(d["pane"])
        # Log to the study's measurement history (shell-owned).
        meas = Measurement(kind=d["type"].capitalize(), points=list(pts),
                           spacing_mm=None)
        meas.text = self._metrics_text(d["pane"], m)
        self.measurement_added.emit(meas)

    def _measure_finish_draft(self):
        d = self._draft
        if d and d["type"] in ("polyline", "polygon") and len(d["pts"]) >= 2:
            self._commit_draft()

    def _measure_right(self, which, sx, sy):
        cat = self._center_angle_target
        if cat and cat.get("key") == which:
            mi = cat["mi"]
            if 0 <= mi < len(self._measures[which]):
                self._measures[which][mi].pop("center_angle", None)
            self._center_angle_target = None
            self._redraw_meas(which)
            return
        if self._draft and self._draft["pane"] == which \
                and self._draft["type"] in ("polyline", "polygon"):
            self._measure_finish_draft()
            return
        # Right-click ON a Center-Angle marker or spoke deletes JUST the Center
        # Angle (the polygon/ellipse stays). Checked before the vertex/outline
        # menus since the markers sit on the outline.
        ca_mi = self._ca_hit(which, sx, sy)
        if ca_mi is not None:
            menu = QMenu(self)
            del_ca = menu.addAction("Delete Center Angle")
            chosen = menu.exec(self.pane[which].canvas.mapToGlobal(
                QPoint(int(sx), int(sy))))
            if chosen is del_ca:
                self._measures[which][ca_mi].pop("center_angle", None)
                self._redraw_meas(which)
            return
        hit = self._pick_handle(which, sx, sy)
        if hit is not None:
            self._handle_right(which, hit, sx, sy)
            return
        mi = self._pick_measure(which, sx, sy)
        if mi is None:
            return
        self._outline_right(which, mi, sx, sy)

    def _ca_hit(self, which, sx, sy):
        """mi of a measure whose Center-Angle marker point OR spoke line is
        under the screen point (sx,sy), else None — for 'Delete Center Angle'."""
        hit = self._pick_center_angle(which, sx, sy)
        if hit is not None:
            return hit[0]
        wx, wy = self._disp_to_world(which, sx, sy)
        tol = max(3.0, 0.02 * self._half)
        for mi in range(len(self._measures[which]) - 1, -1, -1):
            m = self._measures[which][mi]
            ca = m.get("center_angle")
            if not ca or not ca.get("pts"):
                continue
            centre = self._shape_center(m)
            for q in ca["pts"]:
                if _seg_dist(wx, wy, centre, q) < tol:
                    return mi
        return None

    def _handle_right(self, which, hit, sx, sy):
        mi, vi = hit
        m = self._measures[which][mi]
        menu = QMenu(self)
        del_pt = del_res = None
        if m["type"] in ("polyline", "polygon"):
            del_pt = menu.addAction("Delete point")
            if len(m["pts"]) <= 2:
                del_pt.setEnabled(False)
            del_res = menu.addAction("Delete result")
        else:
            del_res = menu.addAction("Delete")
        chosen = menu.exec(self.pane[which].canvas.mapToGlobal(
            QPoint(int(sx), int(sy))))
        if del_pt is not None and chosen is del_pt:
            self._delete_point(which, mi, vi)
        elif chosen is del_res:
            del self._measures[which][mi]
        self._redraw_meas(which)

    def _outline_right(self, which, mi, sx, sy):
        from PyQt6.QtGui import QIcon, QPixmap
        from multi_dicomviewer.viewers.image_canvas import COLOR_CHOICES
        m = self._measures[which][mi]
        menu = QMenu(self)
        add_pt = menu.addAction("Add point")
        spline_act = None
        if m["type"] == "polyline":
            spline_act = menu.addAction(
                "UnSpline" if m.get("smooth") else "Spline")
        center_angle_act = None
        if m["type"] in ("ellipse", "polygon"):
            center_angle_act = menu.addAction("Center Angle")
        color_menu = menu.addMenu("Change Color")
        color_actions = []
        for name, hexcol in COLOR_CHOICES:
            a = color_menu.addAction(name)
            pix = QPixmap(16, 16)
            pix.fill(QColor(hexcol))
            a.setIcon(QIcon(pix))
            color_actions.append((a, hexcol))
        del_act = menu.addAction("Delete")
        chosen = menu.exec(self.pane[which].canvas.mapToGlobal(
            QPoint(int(sx), int(sy))))
        if chosen is add_pt:
            self._add_point(which, mi, sx, sy)
        elif spline_act is not None and chosen is spline_act:
            m["smooth"] = not m.get("smooth", False)
        elif center_angle_act is not None and chosen is center_angle_act:
            self._center_angle_target = {"key": which, "mi": mi}
            m.pop("center_angle", None)
        elif chosen is del_act:
            del self._measures[which][mi]
        else:
            for act, hexcol in color_actions:
                if chosen is act:
                    m["color"] = hexcol
                    break
        self._redraw_meas(which)

    def _center_angle_add(self, w) -> None:
        cat = self._center_angle_target
        if not cat:
            return
        which, mi = cat["key"], cat["mi"]
        if not (0 <= mi < len(self._measures[which])):
            self._center_angle_target = None
            return
        m = self._measures[which][mi]
        ca = m.setdefault("center_angle", {"pts": []})
        ca["pts"].append(self._snap_ca(m, w))   # constrain to the outline
        if len(ca["pts"]) >= 3:
            centre = self._shape_center(m)
            p1, p2, p3 = ca["pts"][:3]
            span, t1, t3, ccw = _central_arc_angle(centre, p1, p2, p3)
            m["center_angle"] = {"pts": [p1, p2, p3], "angle": span,
                                 "t1": t1, "t3": t3, "ccw": ccw}
            self._center_angle_target = None
        self._redraw_meas(which)

    def _add_point(self, which, mi, sx, sy):
        m = self._measures[which][mi]
        wx, wy = self._disp_to_world(which, sx, sy)
        pt = (wx, wy)
        if m["type"] == "ellipse":
            e1, e2, m1, m2 = m["pts"]            # major ends, minor ends
            m["type"] = "polygon"
            m["pts"] = [e1, m1, e2, m2]          # around the ellipse
        if m["type"] == "line":
            m["type"] = "polyline"
            m["pts"] = [m["pts"][0], pt, m["pts"][1]]
            return
        pts = list(m["pts"])
        n = len(pts)
        closed = (m["type"] == "polygon")
        edges = n if closed else n - 1
        best_i, best_d = 0, float("inf")
        for i in range(edges):
            a, b = pts[i], pts[(i + 1) % n]
            d = _seg_dist(pt[0], pt[1], a, b)
            if d < best_d:
                best_d, best_i = d, i
        pts.insert(best_i + 1, pt)
        m["pts"] = pts
        self._resnap_center_angle(m)

    def _delete_point(self, which, mi, vi):
        m = self._measures[which][mi]
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

    def _toggle_centerline(self):
        self._cl_on = self._cl_btn.isChecked()
        for k in ("A", "B"):
            self._overlay[k].update()

    # ---------------------------------------------------- HU colormap
    def _lut_texture(self):
        """1-D colormap Texture for the current bands/opacity/W-L, rebuilt
        only when those change."""
        key = (tuple((tuple(b["rgb"]), b["lo"], b["hi"], b["on"])
                     for b in self._bands),
               self._opacity, self._win, self._lvl)
        if key != self._lut_key:
            arr = _band_lut_array(self._bands, self._opacity,
                                  self._win, self._lvl)
            self._lut_tex = gfx.Texture(arr, dim=1)
            self._lut_key = key
        return self._lut_tex

    def _toggle_color(self):
        self._color = self._cmap_btn.isChecked()
        self._refresh()

    def _open_setting(self):
        if self._cmap_dlg is None:
            self._cmap_dlg = _ColorMapDialog(
                self._bands, self._opacity, self._apply_colormap, self)
        else:
            self._cmap_dlg.set_bands(self._bands, self._opacity)
        self._cmap_dlg.show()
        self._cmap_dlg.raise_()
        self._cmap_dlg.activateWindow()

    def _apply_colormap(self, bands, opacity):
        self._bands = [dict(b) for b in bands]
        self._opacity = float(opacity)
        if not self._color:
            self._color = True
            self._cmap_btn.setChecked(True)
        self._refresh()

    def _reset(self):
        if self._vol is None:
            return
        if not self._view_initial:
            self._center = self._center0.copy()
            self._pc = {"A": self._center.copy(), "B": self._center.copy()}
            self._pan = {"A": np.zeros(2), "B": np.zeros(2)}
            self._roll = {"A": 0.0, "B": 0.0}
            self._init_frames()
            self._thick = {"A": 0.0, "B": 5.0}
            self._sync_slab_spin()
            self._view_initial = True
            self._refresh(reset_cam=True)
        else:
            self._win, self._lvl = self._win0, self._lvl0
            self._refresh()

    def _apply_preset(self, name):
        if name in CT_WL_PRESETS:
            self._win, self._lvl = (float(x) for x in CT_WL_PRESETS[name])
            self._refresh()

    def _key_toggle_color(self):
        """C shortcut → toggle the HU colormap (keep the toolbar button synced)."""
        self._cmap_btn.setChecked(not self._cmap_btn.isChecked())
        self._toggle_color()
