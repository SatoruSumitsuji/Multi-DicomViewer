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
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor, QCursor, QFont, QImage, QKeySequence, QPainter, QPainterPath, QPen,
    QPolygonF, QShortcut,
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
from multi_dicomviewer.core import settings
from multi_dicomviewer.core.dicom_io import LoadedSeries
from multi_dicomviewer.core.dicom_tags import default_overlay_keywords, overlay_lines
from multi_dicomviewer.core.image_export import (
    export_image_as, pick_export_format, safe_basename,
)
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
    gap_color as _gap_color,
    gap_legend as _gap_legend,
    gap_linewidth as _gap_linewidth,
    major_minor as _major_minor,
    percent_area_diff as _percent_area_diff,
    point_in_poly as _point_in_poly,
    poly_area as _poly_area,
    polygon_centroid as _polygon_centroid,
    radial_gap_compare as _radial_gap_compare,
    project_to_polyline as _project_to_polyline,
    seg_dist as _seg_dist,
    smooth_closed as _smooth_closed,
    smooth_open as _smooth_open,
)
from multi_dicomviewer.ui.viewer_base import AbstractViewer
from multi_dicomviewer.ui.study_browser import FitButton
from multi_dicomviewer.ui.compare_options import CompareOptionsDialog
from multi_dicomviewer.ui.measure_style_menu import (
    add_color_submenu, add_transparency_submenu, transp_to_alpha,
)

#: SPIN sign. +1.0 matches the rotation direction expected on the Mac build.
_SPIN_SIGN = 1.0

# Slab-MIP level-of-detail. The slab MIP is a CPU max-over-N-oblique-planes
# image (see _compute_slab_qimage); cost scales with sample columns² × planes.
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
#: Button captions carry the keyboard shortcut so it's discoverable on-screen.
_TOOL_LABELS = {
    "ZOOM": "Zoom (Z)", "MOVE": "Move (V)", "ROTATE": "Rotate (R)",
    "SPIN": "Spin (S)", "PAGING": "Paging (G)", "THICK": "Thick (T)",
    "WL": "WL (W)",
}
#: Tools that only make sense in 3-D MPR mode — disabled in 2-D (single-slice).
_MPR_ONLY_TOOLS = ("ROTATE", "SPIN", "THICK")
#: Series with this many slices or fewer default to 2-D display; more → 3-D.
_MODE_2D_MAX = 200
#: 2-D frame scrubber handle: a 20 px white disc with a blue inner dot (radial
#: gradient), matching the cine viewer's seek-bar thumb. The negative handle
#: margin overflows the 6 px groove, so the slider reserves extra height
#: (setMinimumHeight in _build_seek_bar) to keep the disc from clipping.
_SEEK_SLIDER_QSS = (
    "QSlider::groove:horizontal{height:6px;border-radius:3px;background:#c4c4c4;}"
    "QSlider::handle:horizontal{width:20px;height:20px;margin:-7px 0;"
    "border:1px solid #6a6a6a;border-radius:10px;"
    "background:qradialgradient(cx:0.5,cy:0.5,radius:0.5,fx:0.5,fy:0.5,"
    "stop:0 #1c6fd0,stop:0.55 #1c6fd0,stop:0.60 #ffffff,stop:1 #ffffff);}"
)
#: Compact (multi-row layout) seek slider — smaller groove + handle.
_SEEK_SLIDER_QSS_COMPACT = (
    "QSlider::groove:horizontal{height:4px;border-radius:2px;background:#c4c4c4;}"
    "QSlider::handle:horizontal{width:12px;height:12px;margin:-4px 0;"
    "border:1px solid #6a6a6a;border-radius:6px;"
    "background:qradialgradient(cx:0.5,cy:0.5,radius:0.5,fx:0.5,fy:0.5,"
    "stop:0 #1c6fd0,stop:0.55 #1c6fd0,stop:0.60 #ffffff,stop:1 #ffffff);}"
)
#: Qt's "no maximum" sentinel for clearing a setMaximumHeight.
_QWIDGETSIZE_MAX = 16777215
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


def _draw_outlined_text(p, rect, flags, text, fill, width=1.0, outline=None):
    """Draw *text* in *fill* colour with an *outline* halo (default black 黒枠;
    pass white for 白枠), by stamping outline-colour copies at 8 compass offsets
    then the fill on top. *width* ≈ outline thickness in px. To stay smooth (not
    blotchy) for thicker outlines, the copies are placed on CONCENTRIC rings
    spaced ≤~0.9px apart so neighbouring shadows always overlap into one
    continuous halo. Unlike a QPainterPath this works with multi-line /
    word-wrapped drawText, so it suits the tag block too. Leaves the painter pen
    set to *fill*."""
    dirs = ((-1, -1), (0, -1), (1, -1), (-1, 0),
            (1, 0), (-1, 1), (0, 1), (1, 1))
    radii = []
    r = min(0.9, width)
    while r < width - 1e-6:
        radii.append(r)
        r += 0.9
    radii.append(width)
    p.setPen(outline if outline is not None else QColor(0, 0, 0))
    for rad in radii:
        for ox, oy in dirs:
            p.drawText(rect.translated(ox * rad, oy * rad), flags, text)
    p.setPen(fill)
    p.drawText(rect, flags, text)


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


def _compute_slab_qimage(p):
    """Pure slab-MIP → RGB QImage from a snapshot dict (NO Qt-widget / no self
    access), so the heavy full-quality rebuild can run on a worker thread. The
    snapshot is taken on the GUI thread by CTViewerPygfx._slab_params; this is
    the same maths as the inline path, only parameterised."""
    u, v, n = p["u"], p["v"], p["n"]
    pc = p["pc"]
    sx, sy, sz = p["dims"]
    px, py = p["pan"]
    pw, ph, iw, ih = p["pw"], p["ph"], p["iw"], p["ih"]
    scale = ph / (2.0 * max(1e-3, p["ps"]))
    a = math.radians(p["roll"])
    ca, sa = math.cos(a), math.sin(a)
    SX, SY = np.meshgrid(np.linspace(0.0, pw, iw), np.linspace(0.0, ph, ih))
    aa = (SX - pw / 2.0) / scale
    bb = (ph / 2.0 - SY) / scale
    WX = px + aa * ca - bb * sa
    WY = py + aa * sa + bb * ca
    bx = pc[0] + WX * u[0] + WY * v[0]
    by = pc[1] + WX * u[1] + WY * v[1]
    bz = pc[2] + WX * u[2] + WY * v[2]
    th = p["thick"]
    vol = p["vol"]
    step = max(1e-3, min(p["dims"]))
    nplanes = int(max(1, min(p["max_planes"], round(th / step))))
    offs = (np.linspace(-th / 2.0, th / 2.0, nplanes)
            if nplanes > 1 else np.array([0.0]))
    nz, ny, nx = vol.shape
    mip = np.full((ih, iw), -np.inf, np.float32)
    for t in offs:
        vx = (bx + t * n[0]) / sx
        vy = (by + t * n[1]) / sy
        vz = (bz + t * n[2]) / sz
        hu = _trilinear_sample(vol, vx, vy, vz).astype(np.float32)
        oob = ((vx < 0) | (vx > nx - 1) | (vy < 0) | (vy > ny - 1)
               | (vz < 0) | (vz > nz - 1))
        np.putmask(hu, oob, _HU_LO)
        mip = np.maximum(mip, hu)
    if p["color"]:
        lut = _band_lut_array(p["bands"], p["opacity"], p["win"], p["lvl"])
        idx = np.clip((mip - _HU_LO) / (_HU_HI - _HU_LO) * 511.0,
                      0, 511).astype(np.int32)
        rgb = (lut[idx, :3] * 255.0).astype(np.uint8)
    else:
        g = np.clip((mip - (p["lvl"] - p["win"] / 2.0))
                    / max(1e-6, p["win"]), 0.0, 1.0)
        gg = (g * 255.0).astype(np.uint8)
        rgb = np.stack([gg, gg, gg], axis=2)
    rgb = np.ascontiguousarray(rgb)
    return QImage(rgb.data, iw, ih, 3 * iw, QImage.Format.Format_RGB888).copy()


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
        # Pad the GPU texture by one AIR (_HU_LO) voxel on every face. The GPU
        # sampler clamps to the edge texel outside the grid, which used to
        # smear the border slice (the first/last frame and the in-plane edges)
        # across the black margin — the Mac "flowing image" border bug. With an
        # air ring as the edge texel, every out-of-volume sample now reads air
        # and renders black. The CPU-side self._vol stays UNpadded (slab-MIP /
        # HU sampling assume voxel0==world0); only this GPU copy is padded, and
        # the mesh is shifted back by one voxel so original coords still line up.
        sx, sy, sz = (float(s) for s in scale)
        vp = np.pad(np.ascontiguousarray(vol, dtype=np.float32), 1,
                    mode="constant", constant_values=_HU_LO)
        tex = gfx.Texture(np.ascontiguousarray(vp, dtype=np.float32), dim=3)
        geom = gfx.Geometry(grid=tex)
        self.material = gfx.VolumeSliceMaterial(
            clim=(-100.0, 700.0), interpolation="linear",
            plane=(0.0, 0.0, 1.0, 0.0))
        self.mesh = gfx.Volume(geom, self.material)
        self.mesh.local.scale = (sx, sy, sz)              # voxel→mm
        self.mesh.local.position = (-sx, -sy, -sz)        # undo the 1-voxel pad
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
        # Draw-only: the LEFT pane (A)'s ▲ points the opposite way (apex on the
        # −uv side). Visual only — the image/frame, the angle readout and the
        # paging-sense are all unchanged. Painted here every repaint, so it
        # persists through ROTATE/SPIN and reset. (Parity with VTK 948d500.)
        apex_sgn = -1.0 if key == "A" else 1.0
        p.setBrush(QColor(0, 242, 64))
        p.setPen(Qt.PenStyle.NoPen)
        for sgn in (1.0, -1.0):
            ax = ccx + sgn * d * uh[0]
            ay = ccy + sgn * d * uh[1]
            apex = S(ax + apex_sgn * sz * uv[0], ay + apex_sgn * sz * uv[1])
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

        def draw_outline(pts, rgb, width=1.8, alpha=255):  # 1.5 ×1.2 — readability
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(rgb[0], rgb[1], rgb[2], alpha), width))
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
            # Hidden by "Hide/Show All Result" (global) or this measure's own
            # right-click Hide → skip its line, handles and id label entirely.
            if v._results_hidden or m.get("hidden"):
                continue
            rgb = _hex_to_rgb(m.get("color"))
            draw_outline(v._outline(m), rgb,
                         alpha=transp_to_alpha(m.get("transp", 0)))
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
                # Solid orange arc on the outline between the two endpoints,
                # passing through the selector — only shown once all 3 points
                # are placed.
                if "angle" in ca and len(ca["pts"]) >= 3:
                    arc = _arc_through(v._outline(m), ca["pts"][0],
                                       ca["pts"][2], ca["pts"][1])
                    if len(arc) >= 2:
                        p.setPen(QPen(QColor(255, 140, 0), 2.88))  # 2.4 ×1.2
                        p.setBrush(Qt.BrushStyle.NoBrush)
                        p.drawPolyline(poly(arc))
                for ci, q in enumerate(ca["pts"]):
                    # pts == [endpoint, other endpoint, arc selector]. The 3rd
                    # point only picks which arc is measured, so it gets no
                    # spoke. Spokes (orange, = marker colour) go to the two
                    # endpoints (ci 0 and 1) — the angle's two arms.
                    if ci == 2:
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
        if lines and not v._results_hidden:
            p.setPen(QColor(255, 217, 0))   # yellow — match the other modalities
            p.setFont(QFont("monospace", v._overlay_font_pt))
            rx = w * 0.60
            flags = (int(Qt.AlignmentFlag.AlignRight)
                     | int(Qt.AlignmentFlag.AlignTop)
                     | int(Qt.TextFlag.TextWordWrap))
            p.drawText(QRectF(rx, 4, w - rx - 6, h - 40), flags,
                       "\n".join(lines))

        # ---- Compare: selection highlight + radial gap colour map ----
        if v._cmp_on:
            for (skey, smi) in v._cmp_sel:
                if skey == key and 0 <= smi < len(v._measures[key]):
                    draw_outline(v._outline(v._measures[key][smi]),
                                 (0, 229, 255), width=6.0)   # cyan = picked
            # instruction banner (top-centre) so the workflow is discoverable
            fb = QFont("monospace", 13)
            fb.setBold(True)
            p.setFont(fb)
            p.setPen(QColor(0, 229, 255))
            n_sel = sum(1 for s in v._cmp_sel if s[0] == key)
            p.drawText(QRectF(0, 8, w, 26),
                       int(Qt.AlignmentFlag.AlignHCenter)
                       | int(Qt.AlignmentFlag.AlignTop),
                       f"Click to select 2 Ellipse/Polygon data to compare"
                       f"  ({n_sel}/2)")
        cmps = [] if v._results_hidden else [c for c in v._compares
                                             if c["key"] == key]
        for c in cmps:
            if c.get("hidden"):                          # Hidden → no fill
                continue
            if c["show_thk"]:
                # Thickness: FILL each angular sector of the annulus by its gap
                # band colour (heatmap) at ~35% alpha = 65% transparent (same as
                # the IVUS fill) — a COMPLETE fill, not radial lines. The pen is
                # the same colour as the brush so adjacent sectors leave no
                # anti-aliased seam (which read as faint radial lines).
                rad = c["radials"]
                nr = len(rad)
                for i in range(nr):
                    a, b = rad[i], rad[(i + 1) % nr]
                    da = abs(b["ang"] - a["ang"]) % 360.0
                    if 2.5 * c["step"] < da < 360.0 - 2.5 * c["step"]:
                        continue                         # a skipped-ray gap
                    col = QColor(_gap_color(a["gap"]))
                    col.setAlpha(transp_to_alpha(c.get("transp", 50)))  # Change Transparency
                    p.setBrush(col)
                    p.setPen(QPen(col, 0.8))             # cover sector seams
                    p.drawPolygon(QPolygonF([S(a["inner"]), S(a["outer"]),
                                             S(b["outer"]), S(b["inner"])]))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(0, 229, 255))          # centroid dot
                p.drawEllipse(S(c["centroid"]), 3.0, 3.0)
            else:
                p.setPen(Qt.PenStyle.NoPen)
                # %PA: single outer-colour annulus fill (65% transparent). Even
                # -odd fill of outer minus inner = annulus; also the Delete area.
                path = QPainterPath()
                path.setFillRule(Qt.FillRule.OddEvenFill)
                path.addPolygon(poly(c["outer"]))
                path.addPolygon(poly(c["inner"]))
                fr = c["fill_rgb"]
                p.fillPath(path, QColor(fr[0], fr[1], fr[2],
                                        transp_to_alpha(c.get("transp", 50))))
        if cmps:
            # one summary line per result + a single colour legend if any result
            # is a VISIBLE Thickness run, lower-left above the WW/WL readout.
            # Legend text is white with a thin black 枠 (readable on any colour).
            p.setFont(QFont("monospace", 11))
            _black, _white = QColor(0, 0, 0), QColor(255, 255, 255)
            bands = (_gap_legend()
                     if any(c["show_thk"] and not c.get("hidden") for c in cmps)
                     else [])
            # NOTE: this legend style (white text + thin black 枠) is intended to
            # be ported to the Windows viewer too once approved on Mac.
            heads = []
            for c in cmps:
                ht = f"Compare #{c['big_id']} vs #{c['small_id']}"
                if c["show_pa"]:
                    ht += f"  %Area:{c['pct']:.1f}%"
                heads.append(ht)
            lh = 16.0
            y = h - 40 - (len(heads) + len(bands)) * lh
            fl = int(Qt.AlignmentFlag.AlignLeft) | int(Qt.AlignmentFlag.AlignVCenter)
            for ht in heads:
                _draw_outlined_text(p, QRectF(10, y - lh, 360, lh), fl, ht,
                                    QColor(0, 229, 255), 1.0, _black)
                y += lh
            for lab, hexc in bands:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(hexc))
                # Lift the swatch 30% of its height (12 px → 3.6 px) to line up
                # with its label, matching the Windows viewer's nudge.
                p.drawRect(QRectF(10, y - 10 - 3.6, 12, 12))
                # White text with a thin black 枠 on every row (readable on any
                # background; the white-on-<5mm-red row was unreadable before).
                # Start the label one swatch-width (12 px) clear of the square's
                # right edge (10+12=22 → 34): the square stays put, the text
                # shifts right so the two don't crowd.
                _draw_outlined_text(p, QRectF(34, y - lh, 200, lh), fl, lab,
                                    _white, 1.0, _black)
                y += lh

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
                # never runs them into the right-side measure results. A thin
                # black outline keeps the white tags legible over bright slices.
                flags = (int(Qt.AlignmentFlag.AlignLeft)
                         | int(Qt.AlignmentFlag.AlignTop)
                         | int(Qt.TextFlag.TextWordWrap))
                _draw_outlined_text(p, QRectF(6, 4, w * 0.40 - 6, h - 40),
                                    flags, "\n".join(head),
                                    QColor(255, 255, 255), width=1.0)
        p.setFont(QFont("monospace", 12))       # corner readouts stay compact
        slab = v._thick[key]
        kind = f"Slab MIP {slab:.1f}mm" if slab > 0 else "MPR (thin)"
        p.drawText(QRectF(6, h - 28, w - 12, 24),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"WW {v._win:.0f}  WL {v._lvl:.0f}")
        p.drawText(QRectF(6, h - 28, w - 12, 24),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"{key}  |  {kind}")
        # angio readout (yellow, bottom-centre) — clinical, always shown.
        # Font 18 = the old 15 ×1.2 (clinician asked for a larger tag), drawn
        # with a black outline so the yellow stays legible over a white slice.
        ang = v._angio_angle(key)
        if ang:
            fb = QFont("monospace", 18)
            fb.setBold(True)
            p.setFont(fb)
            rect = QRectF(0, h - 42, w, 36)
            flags = (Qt.AlignmentFlag.AlignHCenter
                     | Qt.AlignmentFlag.AlignBottom)
            _draw_outlined_text(p, rect, flags, ang,
                                QColor(255, 230, 0), width=2.0)


class _AngioAngleDialog(QDialog):
    """Pick a C-arm view (LAO/RAO primary + CRA/CAU secondary, each with a
    degree value) to rotate the CT slice to. Opened by right-clicking the
    bottom-centre angio readout; pre-filled with the pane's current angle.
    values() returns signed degrees (LAO+/RAO−, CRA+/CAU−)."""

    def __init__(self, prim, sec, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Angio Angle")
        v = QVBoxLayout(self)
        v.addWidget(QLabel("対応するアンギオ像の角度に回転します"))

        r1 = QHBoxLayout()
        self._lr = QComboBox()
        self._lr.addItems(["LAO", "RAO"])
        self._lr.setCurrentIndex(0 if prim >= 0 else 1)
        self._lr_val = QSpinBox()
        self._lr_val.setRange(0, 180)
        self._lr_val.setSuffix(" °")
        self._lr_val.setValue(abs(int(prim)))
        r1.addWidget(QLabel("Primary:"))
        r1.addWidget(self._lr, 1)
        r1.addWidget(self._lr_val, 1)
        v.addLayout(r1)

        r2 = QHBoxLayout()
        self._cc = QComboBox()
        self._cc.addItems(["CRA", "CAU"])
        self._cc.setCurrentIndex(0 if sec >= 0 else 1)
        self._cc_val = QSpinBox()
        self._cc_val.setRange(0, 90)
        self._cc_val.setSuffix(" °")
        self._cc_val.setValue(abs(int(sec)))
        r2.addWidget(QLabel("Secondary:"))
        r2.addWidget(self._cc, 1)
        r2.addWidget(self._cc_val, 1)
        v.addLayout(r2)

        btns = QHBoxLayout()
        btns.addStretch(1)
        ok = QPushButton("OK")
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)

    def values(self):
        prim = self._lr_val.value() * (1 if self._lr.currentIndex() == 0 else -1)
        sec = self._cc_val.value() * (1 if self._cc.currentIndex() == 0 else -1)
        return prim, sec


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
    #: image right-click ▸ Export DICOM / CSV → shell runs that export for the
    #: shown CT series. Args: (fmt, series_uid, plane_path); CT always passes
    #: plane_path="" (one volume — A/B panes are reformats of the same data).
    plane_export_requested = pyqtSignal(str, str, str)
    #: emitted on every committed measurement (shell logs it per study)
    measurement_added = pyqtSignal(object)
    #: fired from a background debounce thread to wake the GUI thread and crisp
    #: up the slab LOD. A cross-thread queued signal posts an event that wakes a
    #: fully-idle Qt loop — which same-thread QTimer/aboutToBlock can't do
    #: reliably under rendercanvas ondemand (see _arm_lod / _lod_settle).
    _lod_wake = pyqtSignal()

    #: Carries a finished high-res slab rebuild (gen, {key: QImage}) from the
    #: compute worker back to the GUI thread (see _lod_settle / _on_slab_done).
    _slab_done = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Unified button look (matches the Angio/IVUS viewer): a light-grey
        # rounded border + consistent padding/background on EVERY button. Each
        # active/coloured button only overrides background+colour, so it keeps
        # this shape and size (the closer per-button rule wins for colour only).
        self.setStyleSheet(
            "QPushButton {"
            " border:1px solid #c8c8c8; border-radius:6px;"
            " padding:3px 8px; background:#ededed; color:#101010; }")
        self._vol = None
        self._header = None
        self._pbasis = np.eye(3)
        self._tag_keywords: list[str] = []
        self._tags_on = True                 # DICOM tag overlay visible (Q toggles)
        self._anon = False
        self._tool = "PAGING"
        self._mode = "3D"                    # "3D" MPR | "2D" native slices
        self._slice2d = 0                    # current slice index in 2-D mode
        self._page_accum = 0.0               # 2-D drag-paging pixel accumulator
        self._side = "Bi"                    # last 3-D Plane choice (Bi/Lt/Rt)
        # 2-D display in-plane axes (output right = U, up = V); rotated/flipped
        # by the Rt90/Lt90/Flip buttons. N stays +z (the paging axis).
        self._axes2d = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
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
        # Compare (%Area + radial gap map between two Polygon/Ellipse outlines)
        self._cmp_on = False                 # Compare-select mode: click 2 shapes
        self._cmp_sel = []                   # [(key, mi)] picked shapes (max 2)
        self._compares = []                  # persisted results (right-click→Delete)
        self._results_hidden = False         # "Hide/Show All Result" global toggle
        self._cmp_want_pa = False            # last-used: compute %PA (IVUS)
        self._cmp_want_thk = True            # last-used: compute Thickness (CT LV)
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
        # Persisted image-quality prefs; ct_full_quality=True turns the coarse
        # interactive LOD OFF so a fast Mac always shows full-quality MPR.
        self._dq = settings.load_display_quality()
        self._lod_off = bool(self._dq.get("ct_full_quality"))
        self._lod_due = None             # monotonic deadline for the rebuild
        self._lod_thread = None          # single reusable debounce worker
        self._slab_gen = 0               # generation token for async slab builds
        self._lod_wake.connect(self._lod_settle)
        self._slab_done.connect(self._on_slab_done)
        disp = QApplication.instance().eventDispatcher() if \
            QApplication.instance() is not None else None
        if disp is not None:
            disp.aboutToBlock.connect(self._on_about_to_block)
        self._loaded_uid = ""

        # drag state (rendercanvas pointer events)
        self._drag_btn = None
        self._last = (0.0, 0.0)
        # Right-click single-vs-double discrimination. A single right-click
        # exports a still image; a right DOUBLE-click forces the full-quality
        # ("high-res") slab rebuild. The export menu is modal and would
        # swallow the second click, so the single-click export is deferred by
        # one double-click interval and a second right-press within that
        # window cancels it and crisps instead. We detect the double ourselves
        # (consecutive right-downs) rather than rely on the canvas' own
        # double_click event, which is not guaranteed for the right button.
        self._pending_rclick = None          # (key, x, y) awaiting the timer
        self._last_rdown_t = None            # monotonic time of last right-down
        self._rclick_timer = QTimer(self)
        self._rclick_timer.setSingleShot(True)
        self._rclick_timer.timeout.connect(self._rclick_timeout)
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
        lay.addWidget(self._build_seek_bar())

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
        # Arrow keys drive the active tool (see _key_arrow). QShortcuts (not
        # keyPressEvent) so they fire over the wgpu canvas' own focus handling.
        for seq, direction in (("Up", "up"), ("Down", "down"),
                               ("Left", "left"), ("Right", "right")):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda d=direction: self._key_arrow(d))
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
        # Auto-recover from a lost pointer-up: if a previous gesture left state
        # behind (drag off-pane, or an event dropped while the GUI thread was
        # busy), clear it AND release any stuck mouse grab so this fresh press —
        # and the toolbar buttons — work again. (The Mac dead-buttons bug.)
        if self._drag_btn is not None or self._cross_grab or self._meas_drag:
            self._reset_pointer_state()
        self._drag_btn = ev.get("button")
        x, y = ev["x"], ev["y"]
        self._last = (x, y)
        self._spin_prev = None
        # Compare-select mode: a left-click picks the two shapes to compare.
        if self._cmp_on and self._drag_btn == 1:
            self._compare_pick(key, x, y)
            return
        # Shift + right-click (single two-finger tap + Shift) = full-quality
        # ("high-res") rebuild. A single gesture so trackpad users don't need
        # the hard-to-do right-DOUBLE-click (which is kept). It only re-renders,
        # so it's safe in any tool/measure mode; check it before everything else.
        if (self._drag_btn == 2
                and "Shift" in (ev.get("modifiers") or ())):
            self._force_crisp()
            return
        # Right-click ON the bottom-centre angio readout → angle dialog
        # (rotate the slice to match a chosen LAO/RAO·CRA/CAU view). Checked
        # first, in any tool/measure mode, since it's a fixed screen target.
        if self._drag_btn == 2 and self._angio_hit(key, x, y):
            self._open_angio_dialog(key)
            return
        # Right-click priority: a measure LINE / handle gets its own menu first
        # (Hide that line / Delete); only an EMPTY spot inside a compare region
        # falls through to the region's colour Hide/Delete menu. Defer the modal
        # region menu out of the pointer handler (pointer-up safety).
        if self._drag_btn == 2:
            self._cross_grab = False
            self._meas_drag = False
            if self._meas_on and self._measure_right(key, x, y):
                return                        # handled a measure line/handle
            ci = self._compare_hit(key, x, y)
            if ci is not None:
                target = self._compares[ci]
                QTimer.singleShot(0, lambda t=target: self._compare_delete_menu(t))
                return
            if self._meas_on:
                return                        # right-click in measure mode = no-op
        if self._meas_on:
            self._cross_grab = False
            started = self._measure_left(key, x, y)
            self._meas_drag = bool(started)
            return
        # Right-click (not measuring): a single click exports a still image;
        # a double click forces the full-quality ("high-res") rebuild. Defer
        # the export by one double-click interval so a second right-press can
        # preempt it (the export menu is modal and would block the second
        # click otherwise).
        if self._drag_btn == 2:
            try:
                dbl_ms = max(150, int(QApplication.doubleClickInterval()))
            except Exception:
                dbl_ms = 400
            now = time.monotonic()
            if (self._last_rdown_t is not None
                    and (now - self._last_rdown_t) * 1000.0 <= dbl_ms):
                # Second right-click within the window → high-res rebuild.
                self._rclick_timer.stop()
                self._pending_rclick = None
                self._last_rdown_t = None
                self._force_crisp()
                return
            self._last_rdown_t = now
            self._pending_rclick = (key, x, y)
            self._rclick_timer.start(dbl_ms)
            return
        # Pressing within the (now 5%) crosshair grab band grabs the centreline
        # (MOVE/ROTATE), overriding the tool — for ALL tools incl. SPIN. The band
        # is small, so SPIN still owns the drag everywhere off the lines; on the
        # lines, grabbing the centreline takes priority (per request).
        self._cross_grab = (self._drag_btn == 1
                            and self._cross_press(key, x, y))

    def _on_move(self, key, ev):
        x, y = ev["x"], ev["y"]
        if self._cmp_on:                      # Compare-select: clicks pick, no drag
            return
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
        # Proactively drop any mouse grab so it can never linger into the next
        # gesture and deaden the toolbar buttons.
        try:
            gw = QWidget.mouseGrabber()
            if gw is not None:
                gw.releaseMouse()
        except Exception:
            pass
        # If an interactive (coarse) slab refresh is owed a quality upgrade, do
        # it NOW on release for the snappiest crisp-up (the idle backstop and
        # debounce timer would otherwise get it a moment later).
        if self._lod_pending:
            self._lod_settle()

    def _reset_pointer_state(self):
        """Clear stale drag/gesture flags and release any stuck Qt mouse grab —
        the recovery for a lost pointer-up that otherwise leaves the canvas
        holding the grab, diverting clicks away from the toolbar buttons."""
        self._drag_btn = None
        self._cross_grab = False
        self._meas_drag = False
        self._spin_prev = None
        try:
            gw = QWidget.mouseGrabber()
            if gw is not None:
                gw.releaseMouse()
        except Exception:
            pass

    def _on_dblclick(self, key, ev):
        # Right double-click is the high-res ("force crisp") gesture, detected
        # from consecutive right-downs in _on_down — never recenter on it.
        if ev.get("button") == 2:
            return
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

    # -- still-image export -------------------------------------------
    def _rclick_timeout(self):
        """No second right-click arrived within the double-click window, so
        the pending right-click was a single click → run the still-image
        export."""
        pend, self._pending_rclick = self._pending_rclick, None
        self._last_rdown_t = None
        if pend is not None:
            self._export_pane(*pend)

    def _export_pane(self, key, x, y):
        """Right-click export on a CT pane (no active measure tool): save
        what's on that pane — GPU slice (or slab-MIP) plus the crosshair,
        measurements and tag/result text from the overlay — in the chosen
        format. (*key* here is the pane id; *fmt* is the format choice.)"""
        if self._header is None:        # nothing loaded → no export offered
            return
        # Capture at full quality, not the coarse interactive LOD that wheel/
        # trackpad paging can leave behind.
        self._force_crisp()
        canvas = self.pane[key].canvas
        # CT offers DICOM + Anon DICOM + CSV but NOT MP4 (slice scroll, not
        # a cine).
        fmt = pick_export_format(
            self, canvas.mapToGlobal(QPoint(int(x), int(y))),
            include_dicom=True, include_mp4=False, include_anon=True,
        )
        if not fmt:
            return
        if fmt in ("dicom", "csv", "anon-dicom"):
            # One volume — A/B panes are reformats of the same series.
            self.plane_export_requested.emit(
                fmt, getattr(self, "_loaded_uid", ""), ""
            )
            return
        img = self._grab_pane_qimage(key)
        if img is not None:
            export_image_as(self, img, fmt, self._export_basename(key))

    def _grab_pane_qimage(self, key):
        """Composite the pane's GPU render (read back from wgpu) with the
        QPainter overlay into one RGB QImage. Returns None on failure."""
        pane = self.pane[key]
        pane.render()                   # synchronous force_draw before readback
        try:
            rgba = pane.renderer.snapshot()      # (H, W, 4), physical pixels
        except Exception:
            return None
        if rgba is None or rgba.size == 0:
            return None
        if rgba.dtype != np.uint8:      # HDR blender returns float (0..1)
            rgba = np.clip(np.asarray(rgba, np.float32) * 255.0, 0, 255) \
                .astype(np.uint8)
        rgba = np.ascontiguousarray(rgba[..., :4])
        h, w = rgba.shape[:2]
        base = QImage(rgba.data, w, h, 4 * w,
                      QImage.Format.Format_RGBA8888).copy()
        # Paint the transparent overlay (crosshair / measures / slab-MIP / info)
        # on top, scaled from its logical size to the snapshot's physical size
        # so HiDPI (Retina) exports line up.
        ov = self._overlay.get(key)
        if ov is not None and ov.width() > 0 and ov.height() > 0:
            p = QPainter(base)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.scale(w / ov.width(), h / ov.height())
            ov.render(p, QPoint(0, 0))
            p.end()
        return base

    def _export_basename(self, key="") -> str:
        """Suggested filename stem from the loaded CT series + pane."""
        h = self._header
        parts: list[object] = []
        if h is not None:
            parts.append(getattr(h, "PatientID", "") or "")
            parts.append(getattr(h, "Modality", "") or "")
            parts.append(getattr(h, "StudyDate", "")
                         or getattr(h, "AcquisitionDate", "") or "")
        if key:
            parts.append(f"pane{key}")
        return safe_basename(*parts)

    # -- Bi / Lt / Rt --------------------------------------------------
    @property
    def supports_side(self) -> bool:
        return True

    def set_side(self, side: str, allow_dual: bool = True) -> None:
        self._side = side
        self._frames["A"].setVisible(side != "Rt")
        self._frames["B"].setVisible(side != "Lt")
        self._refresh_side_buttons()

    def _refresh_side_buttons(self) -> None:
        """Check the Plane button (Bi/Lt/Rt) matching the current state."""
        btns = getattr(self, "_side_btns", None)
        if not btns:
            return
        side = self.current_side()
        for key, b in btns.items():
            on = (side == key)
            b.setChecked(on)
            # The base stylesheet de-natives the button, so give the active
            # Plane its own blue+white fill (colour-only → keeps shape/size).
            b.setStyleSheet("background:#1f77b4;color:white;" if on else "")

    def current_side(self) -> str:
        """Current Bi/Lt/Rt state (derived from pane visibility) so the
        shell's toolbar buttons can mirror THIS pane's choice."""
        a = self._frames["A"].isVisible()
        b = self._frames["B"].isVisible()
        if a and b:
            return "Bi"
        if a:
            return "Lt"
        if b:
            return "Rt"
        # Neither frame reports visible yet — happens right after load while
        # the pane/window is still being shown (Qt isVisible() stays False
        # even after setVisible(True) until an ancestor is shown). Fall back
        # to the intended plane so the buttons don't wrongly default to "Rt".
        return self._side

    # ------------------------------------------------------------ toolbar
    def _build_toolbar(self):
        # Two rows so the (now longer, shortcut-labelled) controls don't grow
        # the strip too wide: row 1 = view/plane/measure controls, row 2 = the
        # interaction tools (each captioned with its keyboard shortcut).
        col = QVBoxLayout()
        col.setContentsMargins(4, 2, 4, 2)
        col.setSpacing(2)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)

        # In-pane Plane switch: Bi (both MPR panes) / Lt (left) / Rt (right).
        row.addWidget(QLabel("Plane:"))
        self._side_btns: dict[str, QPushButton] = {}
        for key, tip in (
            ("Bi", "Show both MPR panes"),
            ("Lt", "Show only the left MPR pane"),
            ("Rt", "Show only the right MPR pane"),
        ):
            b = FitButton(key)
            b.setCheckable(True)
            b.setChecked(key == "Bi")        # both panes shown by default
            b.setHelpToolTip(tip)
            b.clicked.connect(lambda _c, k=key: self.set_side(k, True))
            self._side_btns[key] = b
            row.addWidget(b)
        row.addSpacing(8)

        # 3-D MPR vs 2-D (native single-slice) display. Default chosen per
        # series on load (≥201 slices → 3D, else 2D); see load_series. Tinted a
        # dark (goldenrod) yellow so the 3D/2D pair reads apart from the grey
        # Plane (Bi/Lt/Rt) buttons; the active mode is the brighter shade.
        _mode_css = (
            "QPushButton { background:white; color:black; }"
            "QPushButton:checked { background:#edc63a; color:black; }"
        )
        self._mode_btns: dict[str, QPushButton] = {}
        for key, tip in (
            ("3D", "3-D MPR reconstruction (dual oblique reslice)"),
            ("2D", "Show native acquisition slices one at a time (paging)"),
        ):
            b = FitButton(key)
            b.setCheckable(True)
            b.setChecked(key == "3D")
            b.setHelpToolTip(tip)
            b.setStyleSheet(_mode_css)
            b.clicked.connect(lambda _c, k=key: self._set_mode(k))
            self._mode_btns[key] = b
            row.addWidget(b)
        row.addSpacing(8)

        # Setting / Reset: a darker-grey, softer-text look so they read as
        # secondary controls, set apart from the tool buttons.
        _sr_qss = "QPushButton { background:#6e6e6e; color:#d8d8d8; }"
        reset = FitButton("Reset")
        reset.setStyleSheet(_sr_qss)
        reset.clicked.connect(self._reset)
        row.addWidget(reset)

        # ReCalc: same size/font as Reset, a slightly lighter grey so it reads
        # as a sibling utility yet is easy to tell apart. Rebuilds the OTHER
        # pane from the selected pane's green-▲ centre line (un-mirror / fix a
        # companion that drifted after complex rotations) without a full Reset.
        self._recalc_btn = recalc = FitButton("ReCalc")
        recalc.setStyleSheet("QPushButton { background:#8a8a8a; color:#101010; }")
        recalc.setHelpToolTip(
            "Re-derive the OTHER pane from the selected pane's green-▲ centre "
            "line — fixes a mirrored / wrong companion after complex rotations")
        recalc.clicked.connect(self._recalc_companion)
        row.addWidget(recalc)

        self._cmap_btn = FitButton("ColorMap")
        self._cmap_btn.setCheckable(True)
        self._cmap_btn.clicked.connect(self._toggle_color)
        row.addWidget(self._cmap_btn)

        self._meas_btn = FitButton("📏 Measure")
        self._meas_btn.setCheckable(True)
        self._meas_btn.setStyleSheet(            # blue when in Measure mode (= IVUS)
            "QPushButton:checked { background:#1f77b4; color:white; }")
        self._meas_btn.setHelpToolTip(
            "Measure on the image (Line / Polyline / Ellipse / Polygon / Angle)")
        self._meas_btn.clicked.connect(self._toggle_measure)
        row.addWidget(self._meas_btn)

        row.addWidget(QLabel("Slab:"))
        self._slab_spin = QDoubleSpinBox()
        self._slab_spin.setRange(0.0, 50.0)
        self._slab_spin.setSingleStep(0.5)
        self._slab_spin.setDecimals(1)
        self._slab_spin.valueChanged.connect(self._set_slab)
        row.addWidget(self._slab_spin)

        self._cl_btn = FitButton("CenterLine")
        self._cl_btn.setCheckable(True)
        self._cl_btn.setChecked(True)
        self._cl_btn.setHelpToolTip("Show/hide crosshair & slab lines")
        self._cl_btn.clicked.connect(self._toggle_centerline)
        row.addWidget(self._cl_btn)

        # HiRes: disable the coarse interactive LOD so drag/zoom stays full
        # quality (smoother on a fast Mac, heavier on a slow one). Default OFF
        # = keep the LOD. Persisted across restarts.
        self._hires_btn = FitButton("HiRes")
        self._hires_btn.setCheckable(True)
        self._hires_btn.setChecked(self._lod_off)
        self._hires_btn.setHelpToolTip(
            "Always full-quality MPR: turn OFF the coarse preview shown while "
            "dragging/zooming. Smoother on a fast Mac; heavier on a slow one.")
        self._hires_btn.toggled.connect(self._toggle_hires)
        row.addWidget(self._hires_btn)

        setting = FitButton("Setting")
        setting.setHelpToolTip(
            "HU colour-map settings (band colour, HU range, opacity)")
        setting.setStyleSheet(_sr_qss)
        setting.clicked.connect(self._open_setting)
        row.addWidget(setting)

        row.addWidget(QLabel("W/L:"))
        self._preset = QComboBox()
        self._preset.addItems(list(CT_WL_PRESETS.keys()))
        self._preset.currentTextChanged.connect(self._apply_preset)
        row.addWidget(self._preset)

        # DICOM Tags on the LEFT of the pair (kept always visible in the
        # scrollable strip); Measure History — less critical — sits to its
        # right. The tag-text-size slider is stacked above the Tags button
        # (kept a 2-row control, matching the two-row toolbar height).
        tags_box, self._tag_font_slider, tags = build_tag_font_control(
            TAG_FONT_PT_DEFAULT
        )
        tags.setToolTip(
            "Choose which DICOM tags overlay the image (key Q shows/hides)")
        tags.clicked.connect(self.tags_requested.emit)
        self._tag_font_slider.valueChanged.connect(self.overlay_font_changed.emit)
        row.addWidget(tags_box)
        # DICOM-tag controls moved to the shell's global top row; hide the
        # per-viewer copy (kept only for set_overlay_font_pt slider sync).
        tags_box.setVisible(False)

        hist = FitButton("Measure History")
        hist.setHelpToolTip("Show this study's measurement history")
        hist.clicked.connect(self.history_requested.emit)
        row.addWidget(hist)
        row.addStretch(1)

        # Row 2: the interaction tools, each captioned with its shortcut key.
        # Arrow keys ↑↓←→ drive the active tool (see _key_arrow).
        self._tool_btns = {}
        for name in _TOOLS:
            b = FitButton(_TOOL_LABELS[name])
            b.setCheckable(True)
            b.clicked.connect(lambda _c, n=name: self._set_tool(n))
            self._tool_btns[name] = b
            row2.addWidget(b)

        # 2-D image transforms (rotate 90° / flip), right of the tools (whose
        # last entry is WL). Disabled (greyed) in 3-D. "Mirror" == Flip-H, so it
        # is not a separate button. Kept on this second row so they stay visible
        # on a narrow pane (row 1 overflows).
        row2.addSpacing(12)
        self._t2d_btns = []
        for label, kind, tip in (
            ("Rt90°", "rt90", "Rotate the image 90° clockwise"),
            ("Lt90°", "lt90", "Rotate the image 90° counter-clockwise"),
            ("Flip-H", "fliph", "Flip horizontally (left-right mirror)"),
            ("Flip-V", "flipv", "Flip vertically (top-bottom)"),
        ):
            b = FitButton(label)
            b.setHelpToolTip(tip)
            b.clicked.connect(lambda _c, k=kind: self._2d_transform(k))
            self._t2d_btns.append(b)
            row2.addWidget(b)
        row2.addStretch(1)

        col.addLayout(row)
        col.addLayout(row2)
        self._set_tool("PAGING")

        # The CT pane is only half the window, so these controls can overflow
        # its width — worse on macOS, whose native buttons reserve more
        # horizontal padding. Give every button a SMALL minimum width (a floor,
        # not its full natural width) so on a narrow / low-res monitor the
        # labels first elide from the right — keeping the START of each caption
        # readable, full text in the tooltip (FitButton) — and only fall back to
        # the horizontal scroll bar when even that won't fit.
        bar = QWidget()
        bar.setLayout(col)
        for b in bar.findChildren(QPushButton):
            b.setMinimumWidth(min(b.sizeHint().width(), 56))
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
        # MPR-only tools are unavailable in 2-D native-slice mode (their
        # keyboard shortcuts are otherwise still live).
        if getattr(self, "_mode", "3D") == "2D" and name in _MPR_ONLY_TOOLS:
            return
        self._tool = name
        for n, b in self._tool_btns.items():
            b.setChecked(n == name)
            # Active = red background + WHITE text; only colour is overridden so
            # the button keeps the base border/radius/padding (no size change).
            b.setStyleSheet(
                "background:#c0392b;color:white;" if n == name else "")

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

    # ------------------------------------------------------- 3D / 2D mode
    def _set_mode(self, mode, reset_cam=False):
        """Switch between 3-D MPR (dual oblique reslice) and 2-D native-slice
        display. In 2-D only pane A is shown, locked to the acquisition
        (axial) plane, paging native slices; the MPR-only tools/controls are
        disabled. Default mode is chosen per series on load."""
        if mode not in ("3D", "2D") or self._vol is None:
            return
        self._mode = mode
        for k, b in self._mode_btns.items():
            b.setChecked(k == mode)
        is2d = (mode == "2D")
        for name in _MPR_ONLY_TOOLS:
            self._tool_btns[name].setEnabled(not is2d)
        self._slab_spin.setEnabled(not is2d)
        self._cl_btn.setEnabled(not is2d)
        for b in self._side_btns.values():
            b.setEnabled(not is2d)
        # The 2-D image transforms only apply to the native slice (2-D mode).
        for b in self._t2d_btns:
            b.setEnabled(is2d)
        if is2d:
            if self._tool in _MPR_ONLY_TOOLS:
                self._set_tool("PAGING")
            self._active_pane = "A"
            self._frames["A"].setVisible(True)
            self._frames["B"].setVisible(False)
            self._update_active_frames()
            self._init_frames(native=True)           # lock to native axial
            self._apply_2d_axes()                    # re-apply rotate/flip state
            self._thick = {"A": 0.0, "B": 0.0}
            self._roll = {"A": 0.0, "B": 0.0}
            self._pan = {"A": np.zeros(2), "B": np.zeros(2)}
            self._cl_on = False                      # no crosshair in 2-D
            # Snap the slice plane onto the nearest native slice so no z
            # interpolation occurs (the image is the acquired slice as-is).
            nz = self._vol.shape[0]
            sz = self._dims[2]
            k = int(round(self._center[2] / sz)) if sz > 1e-6 else 0
            self._slice2d = min(max(k, 0), max(0, nz - 1))
            z = self._slice2d * sz if sz > 1e-6 else 0.0
            self._center[2] = z
            self._pc = {"A": self._center.copy(), "B": self._center.copy()}
        else:
            self._frames["A"].setVisible(self._side != "Rt")
            self._frames["B"].setVisible(self._side != "Lt")
            self._thick = {"A": 0.0, "B": 5.0}
            self._cl_on = self._cl_btn.isChecked()
            self._refresh_side_buttons()
        self._sync_slab_spin()
        self._refresh(reset_cam=reset_cam or is2d)
        self._sync_seek()

    def _page2d(self, step):
        """Page by *step* native slices in 2-D mode (integer slice index)."""
        if self._vol is None:
            return
        nz = self._vol.shape[0]
        sz = self._dims[2]
        self._slice2d = int(min(max(self._slice2d + step, 0), max(0, nz - 1)))
        z = self._slice2d * sz if sz > 1e-6 else 0.0
        self._center[2] = z
        self._pc["A"][2] = z
        self._clamp_center()
        self._view_initial = False
        self._refresh()
        self._sync_seek()

    # ------------------------------------------------------ 2-D frame seek bar
    def _build_seek_bar(self) -> QWidget:
        """A bottom scrubber for 2-D mode: shows N/total and lets the user drag
        through the native slices (so a multi-frame series doesn't look like a
        single image). Hidden in 3-D MPR and for single-frame series."""
        self._seek_wrap = QWidget()
        # Survive the shell's "Max Image" (Hide Buttons): the slice scrubber +
        # its Frame/count labels stay visible so paging is still usable.
        self._seek_wrap._mdv_keep_on_max = True
        row = QHBoxLayout(self._seek_wrap)
        row.setContentsMargins(8, 2, 8, 2)

        self._seek_base_pt = QLabel("x").font().pointSizeF() or 9.0

        def _big(lbl):                          # ~1.55× label (readable, compact)
            f = lbl.font()
            f.setPointSizeF(self._seek_base_pt * 1.55)
            f.setBold(True)
            lbl.setFont(f)
            return lbl

        self._seek_frame_lbl = _big(QLabel("Frame:"))
        row.addWidget(self._seek_frame_lbl)
        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setMinimum(0)
        self._seek_slider.setMaximum(0)
        self._seek_slider.setMinimumHeight(26)   # room for the 20px disc handle
        self._seek_slider.setStyleSheet(_SEEK_SLIDER_QSS)
        self._seek_slider.valueChanged.connect(self._on_seek)
        row.addWidget(self._seek_slider, 1)
        self._seek_lbl = _big(QLabel("1 / 1"))
        self._seek_lbl.setMinimumWidth(96)
        row.addWidget(self._seek_lbl)
        self._seek_wrap.setVisible(False)
        if getattr(self, "_ct_compact", False):
            self._apply_seek_compact(True)
        return self._seek_wrap

    def set_compact(self, on: bool) -> None:
        """Shrink the bottom slice scrubber (Frame label + slider + N/total)
        for multi-row layouts so it matches the cine viewers' compact
        transport. Called by the shell from _apply_layout."""
        on = bool(on)
        if getattr(self, "_ct_compact", False) == on:
            return
        self._ct_compact = on
        if hasattr(self, "_seek_slider"):
            self._apply_seek_compact(on)

    def _apply_seek_compact(self, on: bool) -> None:
        base = getattr(self, "_seek_base_pt", 9.0) or 9.0
        for lbl in (self._seek_frame_lbl, self._seek_lbl):
            f = lbl.font()
            f.setPointSizeF(base * (1.0 if on else 1.55))
            f.setBold(not on)               # big = bold, compact = normal
            lbl.setFont(f)
        self._seek_lbl.setMinimumWidth(60 if on else 96)
        self._seek_slider.setMinimumHeight(16 if on else 26)
        self._seek_slider.setMaximumHeight(16 if on else _QWIDGETSIZE_MAX)
        self._seek_slider.setStyleSheet(
            _SEEK_SLIDER_QSS_COMPACT if on else _SEEK_SLIDER_QSS
        )

    def _on_seek(self, val):
        """Scrubber moved → jump to that native slice (2-D mode)."""
        if self._mode != "2D" or self._vol is None:
            return
        nz = self._vol.shape[0]
        sz = self._dims[2]
        self._slice2d = int(min(max(int(val), 0), max(0, nz - 1)))
        z = self._slice2d * sz if sz > 1e-6 else 0.0
        self._center[2] = z
        self._pc["A"][2] = z
        self._clamp_center()
        self._view_initial = False
        self._refresh()
        self._seek_lbl.setText(f"{self._slice2d + 1} / {nz}")

    def _sync_seek(self):
        """Show/refresh the scrubber to match the current mode and slice."""
        nz = self._vol.shape[0] if self._vol is not None else 1
        show = (self._mode == "2D" and nz > 1)
        self._seek_wrap.setVisible(show)
        if not show:
            return
        self._seek_slider.blockSignals(True)
        self._seek_slider.setMaximum(nz - 1)
        self._seek_slider.setValue(self._slice2d)
        self._seek_slider.blockSignals(False)
        self._seek_lbl.setText(f"{self._slice2d + 1} / {nz}")

    # ------------------------------------------------- 2-D image transforms
    def _apply_2d_axes(self):
        """Set pane A's in-plane display axes (U, V) from the 2-D rotate/flip
        state, keeping the slice normal at +z (the plane the GPU cuts). A flip
        makes cross(U, V) = -z, so the per-pane camera views the slice from the
        other side — i.e. mirrored — while the cut plane stays the same slice."""
        u, v = self._axes2d
        ez = np.array([0.0, 0.0, 1.0])
        self._frame["A"] = (np.asarray(u, float).copy(),
                            np.asarray(v, float).copy(), ez)

    def _2d_transform(self, kind):
        """Rotate the 2-D image 90° (rt90/lt90) or flip it (fliph/flipv).
        Applied incrementally to the current display axes (composable)."""
        if self._mode != "2D" or self._vol is None:
            return
        u, v = self._axes2d
        if kind == "rt90":          # 90° clockwise
            self._axes2d = (v.copy(), (-u).copy())
        elif kind == "lt90":        # 90° counter-clockwise
            self._axes2d = ((-v).copy(), u.copy())
        elif kind == "fliph":       # left-right mirror (== "Mirror")
            self._axes2d = ((-u).copy(), v.copy())
        elif kind == "flipv":       # top-bottom flip
            self._axes2d = (u.copy(), (-v).copy())
        else:
            return
        self._apply_2d_axes()
        self._view_initial = False
        self._refresh(reset_cam=True)   # refit (a 90° turn swaps the aspect)

    def _page_step(self, step):
        """One paging notch: a native slice in 2-D, a wheel step in 3-D."""
        if self._mode == "2D":
            self._page2d(step)
        else:
            self._wheel(self._active_pane, step)

    def _key_arrow(self, direction):
        """Drive the currently-selected tool from an arrow key. Mapping:
        Zoom/Paging/Thick = ↑/↓ only; Move = ↑↓←→; Rotate = ↑↓←→ (orthogonal,
        about the centreline, no diagonal); Spin = →/↓ CW, ←/↑ CCW; WL = same
        as the mouse drag (←/→ window, ↑/↓ level)."""
        if self._vol is None:
            return
        k = self._active_pane
        t = self._tool
        S = 12.0
        if t == "PAGING":
            # Paging is up/down only (left/right do nothing).
            if direction == "up":
                self._page_step(1)
            elif direction == "down":
                self._page_step(-1)
            return
        if t == "ZOOM":
            # Up = zoom OUT (shrink), Down = zoom IN (enlarge) — same as the
            # mouse drag (arrow up mirrors a mouse-up = negative dy).
            if direction == "up":
                self._drag(k, 0, -S)
            elif direction == "down":
                self._drag(k, 0, S)
            return
        if t == "THICK":
            if self._mode == "2D":
                return
            if direction == "up":
                self._drag(k, 0, -S)       # (dx-dy)*0.3 → thicker
            elif direction == "down":
                self._drag(k, 0, S)
            return
        if t == "MOVE":
            d = {"up": (0, -S), "down": (0, S),
                 "left": (-S, 0), "right": (S, 0)}[direction]
            self._drag(k, d[0], d[1])
            return
        if t == "WL":
            d = {"left": (-S, 0), "right": (S, 0),
                 "up": (0, -S), "down": (0, S)}[direction]
            self._drag(k, d[0], d[1])
            return
        if t == "ROTATE":
            if self._mode == "2D":
                return
            d = {"up": (0, -S), "down": (0, S),
                 "left": (-S, 0), "right": (S, 0)}[direction]
            self._drag(k, d[0], d[1])      # one axis → no diagonal tilt
            return
        if t == "SPIN":
            if self._mode == "2D":
                return
            sign = 1.0 if direction in ("right", "down") else -1.0
            self._roll[k] += _SPIN_SIGN * sign * 5.0
            self._refresh()
            return

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
        # Reset any 2-D rotate/flip to the native orientation for the new series.
        self._axes2d = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))

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
        # Default 3-D MPR for thin-slice volumes (≥201 slices), 2-D native
        # paging for ordinary (≤200-slice) series. _set_mode also fits & draws.
        self._set_mode("3D" if nz >= _MODE_2D_MAX + 1 else "2D", reset_cam=True)

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
    def _init_frames(self, native=False):
        """Set the two panes' initial (u=right, v=up, n=normal) frames.

        3-D MPR default (native=False): orient the panes from the patient
        coordinate system (via patient_basis) rather than the raw volume axes,
        so the first image is anatomically correct regardless of how the
        series was stored:

          * Pane A — axial, viewed from below ("下から見上げる"): screen-right =
            patient LEFT, screen-up = ANTERIOR (sternum), screen-down =
            POSTERIOR (spine). This puts the left ventricle on the image right
            and the right ventricle on the image left.
          * Pane B — frontal/coronal companion: screen-right = patient LEFT,
            screen-up = SUPERIOR.

        native=True restores the raw volume-axis frames (used by the 2-D lock,
        which pages the acquired slices as-is)."""
        pb = getattr(self, "_pbasis", None)
        if native or pb is None:
            self._frame = {
                "A": (np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 1.0, 0.0]),
                      np.array([0.0, 0.0, 1.0])),
                "B": (np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 0.0, 1.0]),
                      np.array([0.0, 1.0, 0.0])),
            }
        else:
            # patient = pbasis @ volume  →  volume = inv(pbasis) @ patient.
            # Build the desired axes in patient LPS (+X=Left, +Y=Posterior,
            # +Z=Superior) and map them back into volume space.
            try:
                inv = np.linalg.inv(np.asarray(pb, dtype=np.float64))
            except np.linalg.LinAlgError:
                inv = np.eye(3)
            left = inv @ np.array([1.0, 0.0, 0.0])     # patient Left
            ant = inv @ np.array([0.0, -1.0, 0.0])     # patient Anterior
            sup = inv @ np.array([0.0, 0.0, 1.0])      # patient Superior
            self._frame = {
                "A": self._ortho(left, ant),           # axial from below
                "B": self._ortho(left, sup),           # frontal/coronal
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
    def _slab_params(self, key, lod=False) -> dict:
        """Snapshot (on the GUI thread) everything _compute_slab_qimage needs:
        a plain dict of numpy/scalars + the (read-only, shared) volume — no Qt
        widget access — so the build can run on a worker thread."""
        pane = self.pane[key]
        pw = max(1, pane.canvas.width())
        ph = max(1, pane.canvas.height())
        iw = min(pw, _SLAB_IW_LOD if lod else _SLAB_IW_FULL)
        ih = max(1, int(round(iw * ph / pw)))
        u, v, n = self._frame[key]
        return {
            "u": np.asarray(u, float), "v": np.asarray(v, float),
            "n": np.asarray(n, float), "pc": np.asarray(self._pc[key], float),
            "dims": tuple(self._dims),
            "pan": (float(self._pan[key][0]), float(self._pan[key][1])),
            "pw": pw, "ph": ph, "iw": iw, "ih": ih,
            "ps": float(self._ps[key]), "roll": float(self._roll[key]),
            "thick": float(self._thick[key]), "vol": self._vol,
            "max_planes": _SLAB_PLANES_LOD if lod else _SLAB_PLANES_FULL,
            "color": bool(self._color),
            "bands": [dict(b) for b in self._bands],
            "opacity": float(self._opacity),
            "win": float(self._win), "lvl": float(self._lvl),
        }

    def _build_slab_qimage(self, key, lod=False):
        """Synchronous slab MIP → QImage (used by the non-interactive refresh).
        Interactive full-quality rebuilds go through the async worker instead."""
        return _compute_slab_qimage(self._slab_params(key, lod=lod))

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
        # "HiRes" (ct_full_quality): never use the coarse interactive LOD — a
        # fast Mac rebuilds full quality every frame instead.
        if self._lod_off:
            lod = False
        # Any refresh (interactive frame or full) supersedes an in-flight async
        # high-res slab build, so bump the generation to discard a late result.
        self._slab_gen += 1
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
        """Build the slab at full quality OFF the GUI thread, then swap it in.

        Runs at most once per interaction (guarded by _lod_pending), whichever
        path gets there first: pointer-up, aboutToBlock (idle), or the worker
        wake. The heavy CPU MIP must NOT run on the GUI thread — a multi-second
        rebuild there froze the event loop and dropped the pointer-up, leaving
        macOS holding a stuck mouse grab so the buttons went dead. Here we only
        SNAPSHOT params on the GUI thread and hand the maths to a worker; the
        coarse image stays on screen until the crisp one arrives."""
        if not self._lod_pending:
            return
        self._cancel_lod()
        self._lod_pending = False
        params = {k: self._slab_params(k, lod=False)
                  for k in ("A", "B") if self._thick[k] > 0}
        if not params:
            return
        self._slab_gen += 1
        gen = self._slab_gen
        threading.Thread(target=self._slab_compute_worker,
                         args=(gen, params), daemon=True).start()

    def _slab_compute_worker(self, gen, params):
        """Worker thread: build the full-quality slab QImage(s) and post them
        back to the GUI thread via the _slab_done signal (queued connection)."""
        out = {}
        for k, pr in params.items():
            try:
                out[k] = _compute_slab_qimage(pr)
            except Exception:
                out[k] = None
        self._slab_done.emit((gen, out))

    def _on_slab_done(self, payload):
        """GUI thread: adopt a finished async slab build unless it was
        superseded (a newer interaction/refresh bumped the generation)."""
        gen, out = payload
        if gen != self._slab_gen:
            return
        for k, img in out.items():
            if img is not None and self._thick[k] > 0:
                self._mip_img[k] = img
                self._overlay[k].update()

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
            if self._mode == "2D":
                # 2-D: page integer native slices, ~6 px of drag per slice.
                self._page_accum -= dy
                ppx = 6.0
                while self._page_accum >= ppx:
                    self._page_accum -= ppx
                    self._page2d(1)
                while self._page_accum <= -ppx:
                    self._page_accum += ppx
                    self._page2d(-1)
                return
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
            un, vn, _nn = self._frame[which]
            # KEEP the crossline angle exactly as it was (the crossline is fixed
            # to the pane's frame and rotates WITH it). This branch used to reset
            # _cross_ang to 0 — snapping an oblique crossline back to orthogonal
            # on every plane rotation. (Re-projecting it into the new plane,
            # tried earlier, also drifted toward orthogonal on large rotations.)
            a = math.radians(self._cross_ang[which])
            crossdir = _norm(un * math.cos(a) + vn * math.sin(a))
            # Re-derive the companion so it still marks that crossline, with a
            # continuous orientation (see _couple_companion).
            self._couple_companion(which, crossdir)
            self._pc = {"A": self._center.copy(), "B": self._center.copy()}
        elif t == "ZOOM":
            # Drag (and arrow) UP = zoom OUT (shrink), DOWN = zoom IN (enlarge):
            # dy<0 (up) → factor>1 → larger half-height (_ps) → wider view = shrink.
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
        if self._mode == "2D":
            self._page2d(1 if delta > 0 else -1)
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
    def _couple_companion(self, which, crossdir) -> None:
        """Re-derive the OTHER pane as the plane ⟂ to *which* that contains the
        unit crossline *crossdir* (a 3-D vector lying in *which*'s plane),
        keeping the companion's image orientation CONTINUOUS and its crossline
        still marking that line — instead of rebuilding a fresh ortho() that
        snaps the companion's crossline back to straight. Shared by the
        crosshair-rotate gesture and the ROTATE tool so neither resets the
        crossline."""
        other = "B" if which == "A" else "A"
        n = self._frame[which][2]
        _ou, ov, on = self._frame[other]
        new_n = _norm(np.cross(crossdir, n))            # companion plane normal
        if float(np.dot(new_n, on)) < 0.0:
            new_n = -new_n                              # keep the viewing side stable
        v_new = ov - float(np.dot(ov, new_n)) * new_n   # old up projected in-plane
        if float(np.linalg.norm(v_new)) < 1e-6:         # old up ⟂ new plane
            v_new = np.cross(new_n, crossdir)
        v_new = _norm(v_new)
        u_new = np.cross(v_new, new_n)                  # u×v = new_n (ortho convention)
        self._frame[other] = (u_new, v_new, new_n)
        self._cross_ang[other] = math.degrees(math.atan2(
            float(np.dot(crossdir, v_new)), float(np.dot(crossdir, u_new))))
        self._pc[other] = self._center.copy()

    def _patient_axis_vol(self, p):
        """A patient-LPS direction (e.g. (1,0,0)=Left) in volume coords."""
        pb = getattr(self, "_pbasis", None)
        try:
            inv = (np.linalg.inv(np.asarray(pb, float))
                   if pb is not None else np.eye(3))
        except np.linalg.LinAlgError:
            inv = np.eye(3)
        return _norm(inv @ np.array(p, float))

    def _flash_recalc(self):
        """Briefly flash the ReCalc button green so the click is visibly
        acknowledged even when nothing in the image changes."""
        btn = getattr(self, "_recalc_btn", None)
        if btn is None:
            return
        btn.setStyleSheet("QPushButton { background:#2ecc71; color:#101010; }")
        QTimer.singleShot(380, lambda: btn.setStyleSheet(
            "QPushButton { background:#8a8a8a; color:#101010; }"))

    def _recalc_companion(self):
        """ReCalc: rebuild the OTHER pane as the plane that cuts the ACTIVE pane
        along its green-▲ centre line, fixing a MIRROR while keeping the view.

        _couple_companion rebuilds it right-handed but PRESERVES the companion's
        existing viewing side, so a mirrored companion would stay mirrored. We
        then force the non-mirror side: screen-right ≈ patient LEFT (the
        _init_frames convention). Flipping u (and n) keeps the up vector, so the
        zoom/roll are preserved and only the left-right mirror is corrected.
        The master pane is untouched."""
        if self._vol is None:
            return
        master = self._active_pane
        other = "B" if master == "A" else "A"
        u, v, _n = self._frame[master]
        a = math.radians(self._cross_ang[master])
        crossdir = u * math.cos(a) + v * math.sin(a)
        self._couple_companion(master, crossdir)
        ou, ov, on = self._frame[other]
        left = self._patient_axis_vol((1.0, 0.0, 0.0))      # patient Left in vol
        if float(np.dot(ou, left)) < 0.0:                   # mirrored → un-mirror
            ou, on = -ou, -on
            self._frame[other] = (ou, ov, on)
            self._cross_ang[other] = math.degrees(math.atan2(
                float(np.dot(crossdir, ov)), float(np.dot(crossdir, ou))))
        self._view_initial = False
        self._refresh()
        self._flash_recalc()

    def _cross_press(self, which, sx, sy) -> bool:
        """True (and arm a MOVE/ROTATE gesture) if the press lands ON the
        crosshair, else False so the selected tool handles the drag.

        Distances are measured in NORMALISED screen space — each pane axis
        scaled to [-1, 1] (centre→edge = 1). So the catch band is 10% of the
        actual screen on EACH side of a crossline (10% left/right of the
        vertical line, 10% up/down of the horizontal line), on both axes
        regardless of the pane's aspect ratio (vital for the extreme aspect of
        side-by-side multi-pane) and at any zoom. (It used to be tied to the
        fixed volume diagonal self._half, which ballooned to most of a
        zoomed-in thin-MIP pane and hijacked paging / tool drags.) Of the
        caught span, the INNER half (near the centre) translates the plane; the
        OUTER half rotates it."""
        if self._vol is None:
            return False
        wx, wy = self._disp_to_world(which, sx, sy)   # world (gesture state)
        ccx, ccy = self._cc(which)
        cx, cy = self._screen_center(which)           # crosshair centre, px
        pane = self.pane[which]
        hx = max(1.0, pane.canvas.width() / 2.0)
        hy = max(1.0, pane.canvas.height() / 2.0)
        a = math.radians(self._cross_ang[which])

        def _ndir(ux, uy):
            """Output-basis crossline direction (ux,uy) → unit vector in
            normalised screen space (carries roll + pixel aspect)."""
            px, py = self._world_to_screen(which, ccx + ux, ccy + uy)
            dx, dy = (px - cx) / hx, (py - cy) / hy
            n = math.hypot(dx, dy) or 1.0
            return dx / n, dy / n

        uh = _ndir(math.cos(a), math.sin(a))          # along the H crossline
        uv = _ndir(-math.sin(a), math.cos(a))         # along the V crossline
        # Press point in normalised screen space, relative to the centre.
        rx, ry = (sx - cx) / hx, (sy - cy) / hy
        # [-1,1] per axis → centre-to-edge = 1.0, full screen = 2.0. A 5%-of-
        # screen catch on each side is therefore 0.10 in these units.
        band = 0.10                           # perpendicular catch = 5% screen/side
        mid = 0.50                            # inner half → move, outer → rotate
        # Perpendicular distance to each crossline (|r × û|) + distance ALONG it
        # from the centre (|r · û|).
        d_to_h = abs(rx * uh[1] - ry * uh[0])
        along_h = abs(rx * uh[0] + ry * uh[1])
        d_to_v = abs(rx * uv[1] - ry * uv[0])
        along_v = abs(rx * uv[0] + ry * uv[1])
        on_h, on_v = d_to_h < band, d_to_v < band
        if not (on_h or on_v):
            return False                      # off the crosshair → tool runs
        # The ▲ markers sit on the HORIZONTAL crossline (uh) → that is the
        # green-▲ line. Where the two 5% bands overlap (the central square) the
        # green-▲ line WINS; otherwise grab whichever line was hit.
        grab_h = on_h                         # green-▲ (H); True also if both
        along = along_h if grab_h else along_v
        if along <= mid:
            self._cross_mode = "move"
            # Lock the slide to the grabbed line so the grab is deterministic
            # (no drag-direction auto-detect): the green-▲ (H) line slides ⟂ to
            # itself = along uv (→ reslice/repage); the non-▲ (V) line slides
            # along uh (→ edge-capped centre-line slide). Output-basis vectors.
            ouh = np.array([math.cos(a), math.sin(a)])
            ouv = np.array([-math.sin(a), math.cos(a)])
            self._cross_axis = ouv if grab_h else ouh
            self._cross_ppt = (wx, wy)
        else:
            self._cross_mode = "rotate"
            self._cross_prev = math.atan2(wy - ccy, wx - ccx)
        return True

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
            # Which crossline is being dragged? Moving the centre ALONG uh — the
            # green-▲ line direction, the shared crossline that also lies in the
            # companion plane — is the "non-▲ line" translate.
            along_uh = (abs(float(np.dot(self._cross_axis, uh)))
                        >= abs(float(np.dot(self._cross_axis, uv))))
            if along_uh:
                # Keep BOTH images fixed (never pan) and the crosshair slide
                # along dir3 only. Do NOT _clamp_center here: that axis-aligned
                # box clamp pulls the centre OFF the slide line at the data edge
                # and drifts the ▲ line. Instead cap the slide at the view edge
                # in either pane, so the non-▲ centre line stops at the edge;
                # past the data it just shows black. (Recentre to go further.)
                dn = dir3 / (float(np.linalg.norm(dir3)) or 1.0)
                over = 0.0
                for pk in (which, other):
                    offp = float(np.dot(self._center - self._pc[pk], dn))
                    limp = self._ps[pk]            # half-view (the edge)
                    op = offp - max(-limp, min(limp, offp))
                    if abs(op) > abs(over):
                        over = op                  # the binding pane's overflow
                if over:
                    self._center = self._center - over * dn
            else:
                self._clamp_center()               # ▲ line: reslice → box-clamp OK
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
        # Re-derive the companion plane so it stays ⟂ to this pane and still
        # contains the rotated crossline, keeping its orientation continuous
        # (see _couple_companion — it does NOT snap the companion straight).
        self._couple_companion(which, crossdir)
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

        The pane normal (= cross(u,v), pointing toward the observer/detector)
        is used with its real sign — we deliberately do NOT fold it into the
        anterior hemisphere. Folding made a fully-reversed view (e.g. spinning
        the cross-line 180° so the companion looks from the opposite side)
        collapse back to the same reading; without it a reversed LAO30 now
        correctly reads RAO150, etc."""
        vals = self._angio_angle_vals(key)
        if vals is None:
            return ""
        pi_, si_ = vals
        lao = f"LAO{pi_}" if pi_ >= 0 else f"RAO{-pi_}"
        cra = f"CRA{si_}" if si_ >= 0 else f"CAU{-si_}"
        return f"{lao} {cra}"

    def _angio_angle_vals(self, key):
        """The readout's (primary, secondary) C-arm angles as signed ints
        (LAO + / RAO −, CRA + / CAU −), or None when the frame is degenerate.
        Shared by the readout text and the right-click angle dialog."""
        n = np.asarray(self._frame[key][2], dtype=np.float64)
        nrm = float(np.linalg.norm(n))
        if nrm < 1e-9:
            return None
        n = self._pbasis @ (n / nrm)
        nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
        axial = math.hypot(nx, ny)
        prim = 0.0 if axial < 1e-9 else math.degrees(math.atan2(nx, -ny))
        sec = math.degrees(math.atan2(nz, axial))
        return int(round(prim)), int(round(sec))

    def _frame_from_angio(self, prim_deg, sec_deg):
        """Inverse of _angio_angle_vals: build a pane frame (u, v, n) whose
        normal projects to the C-arm primary (LAO + / RAO −) and secondary
        (CRA + / CAU −) angle. Screen-up = patient SUPERIOR, matching the
        frontal default (LAO0 CRA0 → coronal viewed from the front)."""
        pr, se = math.radians(prim_deg), math.radians(sec_deg)
        n_pat = np.array([math.cos(se) * math.sin(pr),
                          -math.cos(se) * math.cos(pr),
                          math.sin(se)], dtype=np.float64)
        try:
            inv = np.linalg.inv(np.asarray(self._pbasis, dtype=np.float64))
        except np.linalg.LinAlgError:
            inv = np.eye(3)
        n = _norm(inv @ n_pat)                       # normal in volume coords
        sup = _norm(inv @ np.array([0.0, 0.0, 1.0]))  # patient superior
        if abs(float(np.dot(sup, n))) > 0.999:       # looking down the SI axis
            sup = _norm(inv @ np.array([0.0, -1.0, 0.0]))  # fall back: anterior
        u = _norm(np.cross(sup, n))                  # screen-right
        v = np.cross(n, u)                           # screen-up (u×v = n)
        return (u, v, n)

    def _set_angio_angle(self, which, prim_deg, sec_deg):
        """Re-orient pane *which* so it projects from the given C-arm angle,
        pivoting about the CrossLine intersection (_center). The companion
        pane re-derives as the coupled orthogonal section — same linkage as
        the ROTATE tool."""
        if self._vol is None or self._mode != "3D":
            return
        self._frame[which] = self._frame_from_angio(prim_deg, sec_deg)
        uw, _vw, nw = self._frame[which]
        other = "B" if which == "A" else "A"
        _ou, ov, _on = self._frame[other]
        if float(np.dot(nw, ov)) < 0.0:
            nw = -nw
        self._frame[other] = self._ortho(uw, nw)
        self._cross_ang[which] = 0.0
        self._cross_ang[other] = 0.0
        self._roll[which] = 0.0
        self._pc = {"A": self._center.copy(), "B": self._center.copy()}
        self._view_initial = False
        self._refresh()

    def _angio_hit(self, which, x, y):
        """True if canvas point (x, y) is over the bottom-centre angio readout
        of pane *which* — the right-click target opening the angle dialog."""
        if self._vol is None or self._mode != "3D":
            return False
        if not self._angio_angle(which):
            return False
        c = self.pane[which].canvas
        w, h = c.width(), c.height()
        band = 50.0
        return (h - band) <= y <= (h - 2) and (w * 0.30) <= x <= (w * 0.70)

    def _open_angio_dialog(self, which):
        """Right-click on the readout → pick LAO/RAO + CRA/CAU and rotate the
        pane to that C-arm angle (to line the slice up with an angio view)."""
        vals = self._angio_angle_vals(which) or (0, 0)
        dlg = _AngioAngleDialog(vals[0], vals[1], self)
        if dlg.exec():
            prim, sec = dlg.values()
            self._set_angio_angle(which, prim, sec)

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
            b = FitButton(label)
            b.setMinimumWidth(min(b.sizeHint().width(), 56))
            b.setCheckable(True)
            b.clicked.connect(lambda _c, k=key: self._set_measure_type(k))
            self._meas_btns[key] = b
            row.addWidget(b)
        # Compare two Polygon/Ellipse: %Area difference + radial gap colour map.
        # Placed right of Angle, with Clear All Result to its right.
        self._cmp_btn = FitButton("Compare")
        self._cmp_btn.setMinimumWidth(min(self._cmp_btn.sizeHint().width(), 64))
        self._cmp_btn.setCheckable(True)
        self._cmp_btn.setHelpToolTip(
            "Compare two Polygon/Ellipse: click the two shapes — shows %Area "
            "difference and a radial gap colour map (<5 / 5–7 / 7–9 / >9 mm)")
        self._cmp_btn.clicked.connect(self._toggle_compare)
        row.addWidget(self._cmp_btn)
        # Hide/Show ALL results (lines + region colours + text) at once, between
        # Compare and Clear All Result. Same grey as ReCalc; disabled when there
        # is nothing to hide.
        self._hideall_btn = FitButton("Hide All Result")
        self._hideall_btn.setMinimumWidth(
            min(self._hideall_btn.sizeHint().width(), 64))
        self._hideall_btn.setHelpToolTip(
            "Hide / Show every measurement line, region colour and result text")
        self._hideall_btn.setStyleSheet(                     # light grey, black text
            "QPushButton { background:#bdbdbd; color:#101010; }")
        self._hideall_btn.clicked.connect(self._toggle_hide_all)
        row.addWidget(self._hideall_btn)
        clr = FitButton("Clear All Result")
        clr.setMinimumWidth(min(clr.sizeHint().width(), 56))
        clr.setHelpToolTip("Clear all measurements and comparison results")
        clr.setStyleSheet(                                    # Reset's darker grey
            "QPushButton { background:#6e6e6e; color:#d8d8d8; }")
        clr.clicked.connect(self._measure_clear)
        row.addWidget(clr)
        self._cmp_hint = QLabel("  Left-click = add point /"
                                " right-click finishes Polyline / Polygon")
        row.addWidget(self._cmp_hint)
        row.addStretch(1)
        self._update_hideall_btn()
        return bar

    def _toggle_hide_all(self):
        """Hide / Show ALL results (every measurement line, region colour and
        result text) at once. Show reveals EVERYTHING, including results that
        were individually hidden, regardless of their per-item Hide."""
        self._results_hidden = not self._results_hidden
        if not self._results_hidden:
            for k in ("A", "B"):
                for m in self._measures[k]:
                    m.pop("hidden", None)
            for c in self._compares:
                c.pop("hidden", None)
        for k in ("A", "B"):
            self._overlay[k].update()
        self._update_hideall_btn()

    def _update_hideall_btn(self):
        """Sync the Hide/Show-All button: greyed when there is nothing to hide,
        else labelled Hide (results visible) or Show (results hidden)."""
        btn = getattr(self, "_hideall_btn", None)
        if btn is None:
            return
        has = (any(self._measures[k] for k in ("A", "B"))
               or bool(self._compares))
        btn.setEnabled(has)
        btn.setText("Show All Result" if self._results_hidden
                    else "Hide All Result")

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
            # Active = blue + WHITE text; colour-only override keeps size/shape.
            b.setStyleSheet(
                "background:#1f77b4;color:white;" if k == key else "")

    def _measure_clear(self):
        self._measures = {"A": [], "B": []}
        self._draft = None
        self._edit = None
        self._meas_hover = None
        self._compares = []
        self._cmp_sel = []
        self._results_hidden = False
        for k in ("A", "B"):
            self._redraw_meas(k)

    # ---- Compare: %Area + radial gap between two Polygon/Ellipse shapes ----
    def _toggle_compare(self):
        """Enter/leave Compare-select mode. While on, a left-click picks a
        Polygon/Ellipse (toggles); picking the 2nd computes and ADDS a result
        (prior results persist — right-click a filled region to Delete one)."""
        self._cmp_on = self._cmp_btn.isChecked()
        self._cmp_sel = []
        for k in ("A", "B"):
            self._overlay[k].update()

    def _compare_pick(self, key, sx, sy) -> bool:
        """Compare-mode left-click: toggle the Polygon/Ellipse under the cursor
        in the selection (one pane only). Returns True if the click was consumed
        (i.e. Compare mode is active)."""
        if not self._cmp_on:
            return False
        mi = self._pick_measure(key, sx, sy)
        if mi is not None and self._measures[key][mi]["type"] in (
                "polygon", "ellipse"):
            # Selecting on a different pane restarts the selection there.
            if self._cmp_sel and self._cmp_sel[0][0] != key:
                self._cmp_sel = []
            item = (key, mi)
            if item in self._cmp_sel:
                self._cmp_sel.remove(item)
            elif len(self._cmp_sel) < 2:
                self._cmp_sel.append(item)
            if len(self._cmp_sel) == 2:
                # Defer the modal options dialog out of the pointer handler (so
                # a blocking exec() can't swallow the pointer-up).
                QTimer.singleShot(0, self._compare_prompt)
        self._overlay[key].update()
        return True

    def _compare_prompt(self):
        """Ask which analysis to run (%PA / Thickness), then compute."""
        if not self._cmp_on or len(self._cmp_sel) != 2:
            return
        dlg = CompareOptionsDialog(self._cmp_want_pa, self._cmp_want_thk, self)
        if not dlg.exec():
            self._cancel_compare()
            return
        self._cmp_want_pa, self._cmp_want_thk = dlg.values()
        if not (self._cmp_want_pa or self._cmp_want_thk):
            self._cancel_compare()           # nothing ticked → abort
            return
        self._do_compare()

    def _cancel_compare(self):
        self._cmp_on = False
        self._cmp_sel = []
        self._cmp_btn.setChecked(False)
        for k in ("A", "B"):
            self._overlay[k].update()

    def _do_compare(self):
        sel = self._cmp_sel
        if len(sel) != 2 or sel[0][0] != sel[1][0]:
            return
        key = sel[0][0]
        m1 = self._measures[key][sel[0][1]]
        m2 = self._measures[key][sel[1][1]]
        o1, o2 = self._outline(m1), self._outline(m2)
        a1, a2 = _poly_area(o1), _poly_area(o2)
        # The LARGER shape is the outer reference (centroid + denominator).
        if a2 > a1:
            m1, m2, o1, o2, a1, a2 = m2, m1, o2, o1, a2, a1
        cen = _polygon_centroid(o1)
        # Radials are always computed (they also drive the filled-region hit
        # area); they're only DRAWN as a colour map when Thickness is wanted.
        radials = _radial_gap_compare(o1, o2, cen, 1.0)
        self._compares.append({
            "key": key, "big_id": m1["id"], "small_id": m2["id"],
            "pct": _percent_area_diff(a1, a2), "centroid": cen,
            "outer": o1, "inner": o2, "radials": radials, "step": 1.0,
            "show_pa": self._cmp_want_pa, "show_thk": self._cmp_want_thk,
            "fill_rgb": _hex_to_rgb(m1.get("color")),   # outer (larger) colour
        })
        self._cmp_on = False
        self._cmp_sel = []
        self._cmp_btn.setChecked(False)
        for k in ("A", "B"):
            self._overlay[k].update()
        self._update_hideall_btn()

    def _recompute_compares(self, key):
        """Re-derive any compare results on *key* from the CURRENT outlines of
        the measures they reference (by id), so editing a shape live-updates its
        comparison. A result whose shape was deleted is dropped."""
        if not self._compares:
            return
        by_id = {m["id"]: m for m in self._measures[key]}
        out = []
        for c in self._compares:
            if c["key"] != key:
                out.append(c)
                continue
            mb, ms = by_id.get(c["big_id"]), by_id.get(c["small_id"])
            if mb is None or ms is None:
                continue                     # a referenced shape is gone → drop
            ob, os_ = self._outline(mb), self._outline(ms)
            ab, asm = _poly_area(ob), _poly_area(os_)
            if asm > ab:                     # keep outer = larger
                mb, ms, ob, os_, ab, asm = ms, mb, os_, ob, asm, ab
            cen = _polygon_centroid(ob)
            c = dict(c)
            c.update({
                "big_id": mb["id"], "small_id": ms["id"],
                "outer": ob, "inner": os_, "centroid": cen,
                "pct": _percent_area_diff(ab, asm),
                "radials": _radial_gap_compare(ob, os_, cen, c.get("step", 1.0)),
            })
            # Keep a user-chosen fill colour; otherwise track the outer outline.
            if not c.get("fill_custom"):
                c["fill_rgb"] = _hex_to_rgb(mb.get("color"))
            out.append(c)
        self._compares = out

    def _compare_hit(self, key, sx, sy):
        """Index in self._compares of the result whose filled region (inside the
        outer outline, outside the inner) contains screen point (sx,sy) on pane
        *key* — topmost first — else None."""
        wx, wy = self._disp_to_world(key, sx, sy)
        for i in range(len(self._compares) - 1, -1, -1):
            c = self._compares[i]
            if c["key"] != key:
                continue
            if (_point_in_poly(wx, wy, c["outer"])
                    and not _point_in_poly(wx, wy, c["inner"])):
                return i
        return None

    def _compare_delete_menu(self, target):
        """Right-click INSIDE a compare result's filled region → Hide·Show /
        Delete. Hide toggles only the region COLOUR (its fill); the defining
        outlines are separate measures and are unaffected."""
        if target not in self._compares:
            return
        from multi_dicomviewer.viewers.image_canvas import COLOR_CHOICES
        menu = QMenu(self)
        # Independently-selectable fill colour (separate from the outlines);
        # preserved across recompute via the "fill_custom" flag.
        color_actions = add_color_submenu(menu, COLOR_CHOICES)
        transp_actions = add_transparency_submenu(menu, target.get("transp", 50))
        vis_act = menu.addAction("Show" if target.get("hidden") else "Hide")
        del_act = menu.addAction("Delete")
        chosen = menu.exec(QCursor.pos())
        if chosen is vis_act:
            target["hidden"] = not target.get("hidden", False)
        elif chosen is del_act:
            try:
                self._compares.remove(target)
            except ValueError:
                pass
        else:
            hit = False
            for act, hexcol in color_actions:
                if chosen is act:
                    target["fill_rgb"] = _hex_to_rgb(hexcol)
                    target["fill_custom"] = True
                    hit = True
                    break
            if not hit:
                for act, val in transp_actions:
                    if chosen is act:
                        target["transp"] = val
                        hit = True
                        break
            if not hit:
                return
        for k in ("A", "B"):
            self._overlay[k].update()
        self._update_hideall_btn()

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
        self._recompute_compares(key)      # keep comparisons in sync on edit/delete
        self._metrics[key] = [self._metrics_text(key, m)
                              for m in self._measures[key]]
        self._overlay[key].update()
        self._update_hideall_btn()

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
        # Pixel-based catch (constant on-screen width, zoom/DPR independent), so
        # the boundary band can't balloon when zoomed in and shadow the filled
        # compare region — a click ≥5 px inside the annulus now selects the fill.
        tol = 5.0                              # screen px, each side of the line
        best, bi = tol, None
        for mi, m in enumerate(self._measures[which]):
            wpts = [self._world_to_screen(which, q[0], q[1])
                    for q in self._outline(m)]
            for i in range(len(wpts) - 1):
                d = _seg_dist(sx, sy, wpts[i], wpts[i + 1])
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
        self._recompute_compares(e["key"])     # live-update any comparison
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
        # pts == [endpoint, other endpoint, arc selector]; selector is the
        # "through" point for the geometry helper (see _center_angle_add).
        span, t1, t3, ccw = _central_arc_angle(centre, pts[0], pts[2], pts[1])
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
        # pts == [endpoint, other endpoint, arc selector]; selector is the
        # "through" point for the geometry helper (see _center_angle_add).
        span, t1, t3, ccw = _central_arc_angle(centre, pts[0], pts[2], pts[1])
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

    def _measure_right(self, which, sx, sy) -> bool:
        """Right-click on a measure (handle / outline / Center-Angle marker) →
        its context menu. Returns True if a measure was hit/handled, False if
        the click landed on nothing (so the caller can try the compare region)."""
        cat = self._center_angle_target
        if cat and cat.get("key") == which:
            mi = cat["mi"]
            if 0 <= mi < len(self._measures[which]):
                self._measures[which][mi].pop("center_angle", None)
            self._center_angle_target = None
            self._redraw_meas(which)
            return True
        if self._draft and self._draft["pane"] == which \
                and self._draft["type"] in ("polyline", "polygon"):
            self._measure_finish_draft()
            return True
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
            return True
        hit = self._pick_handle(which, sx, sy)
        if hit is not None:
            self._handle_right(which, hit, sx, sy)
            return True
        mi = self._pick_measure(which, sx, sy)
        if mi is None:
            return False
        self._outline_right(which, mi, sx, sy)
        return True

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
        # Change Color / Change Transparency — on every result type (incl.
        # Line/Angle, most easily right-clicked on a handle).
        from multi_dicomviewer.viewers.image_canvas import COLOR_CHOICES
        color_actions = add_color_submenu(menu, COLOR_CHOICES)
        transp_actions = add_transparency_submenu(menu, m.get("transp", 0))
        hide_act = menu.addAction("Show" if m.get("hidden") else "Hide")
        if m["type"] in ("polyline", "polygon"):
            del_res = menu.addAction("Delete result")
        else:
            del_res = menu.addAction("Delete")
        chosen = menu.exec(self.pane[which].canvas.mapToGlobal(
            QPoint(int(sx), int(sy))))
        if del_pt is not None and chosen is del_pt:
            self._delete_point(which, mi, vi)
        elif chosen is hide_act:
            m["hidden"] = not m.get("hidden", False)   # hide THIS line only
        elif chosen is del_res:
            del self._measures[which][mi]
        else:
            for act, hexcol in color_actions:
                if chosen is act:
                    m["color"] = hexcol
                    break
            for act, val in transp_actions:
                if chosen is act:
                    m["transp"] = val
                    break
        self._recompute_compares(which)     # a colour change refreshes any compare
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
        transp_actions = add_transparency_submenu(menu, m.get("transp", 0))
        hide_act = menu.addAction("Show" if m.get("hidden") else "Hide")
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
        elif chosen is hide_act:
            m["hidden"] = not m.get("hidden", False)   # hide THIS line only
        elif chosen is del_act:
            del self._measures[which][mi]
        else:
            for act, hexcol in color_actions:
                if chosen is act:
                    m["color"] = hexcol
                    break
            for act, val in transp_actions:
                if chosen is act:
                    m["transp"] = val
                    break
        self._recompute_compares(which)     # a colour change refreshes any compare
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
            # Click order is [endpoint, other endpoint, arc selector] — matching
            # Rupture-Predictor. The geometry helper wants (p1, through, p3), so
            # the selector (pts[2]) is passed as the middle "through" point.
            e1, e2, sel = ca["pts"][:3]
            span, t1, t3, ccw = _central_arc_angle(centre, e1, sel, e2)
            m["center_angle"] = {"pts": [e1, e2, sel], "angle": span,
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

    def _toggle_hires(self, on: bool) -> None:
        """HiRes toggle: disable/enable the coarse interactive LOD, persist the
        choice, and repaint at full quality now."""
        self._lod_off = bool(on)
        self._dq["ct_full_quality"] = self._lod_off
        settings.save_display_quality(self._dq)
        if self._vol is not None:
            self._refresh(lod=False)

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
            # Restore the native 2-D orientation (clear any rotate/flip).
            self._axes2d = (np.array([1.0, 0.0, 0.0]),
                            np.array([0.0, 1.0, 0.0]))
            self._sync_slab_spin()
            self._view_initial = True
            # Re-apply the current mode (re-locks 2-D / restores dual MPR) and
            # refits the camera.
            self._set_mode(self._mode, reset_cam=True)
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
