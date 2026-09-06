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
from PyQt6.QtCore import (
    QEvent, QPoint, QPointF, QRect, QRectF, Qt, QThread, QTimer, pyqtSignal,
)
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
from multi_dicomviewer.i18n import t
from multi_dicomviewer.viewers.cpr_mixin import CPRMixin
from multi_dicomviewer.core import settings
from multi_dicomviewer.core.centerline import CenterLine
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
_SLAB_IW_FULL = 480     # slab-MIP sample columns for the immediate at-rest build
_SLAB_IW_LOD = 200      # ...during an interactive drag/page (coarse but smooth)
_SLAB_IW_NATIVE = 1000  # ...the crisp settle build (off-thread, ~native res)
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
#: Tools DISABLED (greyed + unclickable) while an LV pass is axis-locked or in
#: SAX (Trace/SAX): Rotate would re-tilt the locked reslice frame, Paging would
#: shift the long-axis level (hard to tell which pane paged). Zoom/Move/Thick/WL
#: AND Spin (camera roll only) stay usable via the Alt/Option passthrough. NOT
#: applied in plain 3-D MPR. Mirrors the VTK viewer.
_LV_LOCK_DISABLED = ("ROTATE", "PAGING")
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


def _smooth_lut_edges(col: np.ndarray, alpha: np.ndarray, sigma: float = 5.0):
    """Soften the hard band edges of a step colour map: Gaussian-blur the
    premultiplied colour + alpha along the HU axis so adjacent bands (and the
    band↔grayscale boundary) blend over a short HU ramp — removes the
    posterised "blocky / speckly" colour look on noisy CT. Edge-padded so the
    HU extremes don't dip. Returns (colour, alpha). Mirrors ct_viewer._smooth_
    lut_edges so VTK and pygfx colour maps match."""
    r = int(max(1, round(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()

    def _sm(a):
        pad = np.pad(a, ((r, r),) + ((0, 0),) * (a.ndim - 1), mode="edge")
        if a.ndim == 1:
            return np.convolve(pad, k, mode="valid")
        out = np.empty((a.shape[0], a.shape[1]), np.float64)
        for c in range(a.shape[1]):
            out[:, c] = np.convolve(pad[:, c], k, mode="valid")
        return out

    pm = _sm(col * alpha[:, None])
    a_s = _sm(alpha)
    col_s = pm / np.maximum(a_s[:, None], 1e-6)
    return col_s, a_s


def _band_lut_array(bands, opacity, win, lvl) -> np.ndarray:
    """512×4 RGBA float32 colormap over HU [_HU_LO,_HU_HI] with CRISP/hard band
    edges (matching SSMView). Inside the first enabled band: band colour blended
    over the windowed grayscale by *opacity*; outside: grayscale. On-image
    boundary smoothness is handled spatially, not in the LUT. Feeds a pygfx 1-D
    colormap Texture; mirrors the VTK viewer's _band_lut_rgb."""
    n = 512
    hu = _HU_LO + (_HU_HI - _HU_LO) * np.arange(n) / (n - 1)
    glo = lvl - win / 2.0
    span = max(1e-6, float(win))
    g = np.clip((hu - glo) / span, 0.0, 1.0)
    op = float(min(1.0, max(0.0, opacity)))
    col = np.zeros((n, 3), np.float64)
    alpha = np.zeros(n, np.float64)
    assigned = np.zeros(n, dtype=bool)
    for b in bands:
        if not b["on"]:
            continue
        m = (hu >= b["lo"]) & (hu <= b["hi"]) & (~assigned)
        if not m.any():
            continue
        col[m] = b["rgb"]
        alpha[m] = 1.0
        assigned |= m
    eff = (op * alpha)[:, None]
    rgb = np.clip(g[:, None] * (1.0 - eff) + col * eff, 0.0, 1.0)
    out = np.concatenate(
        [rgb.astype(np.float32), np.ones((n, 1), np.float32)], axis=1)
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


_WALL_RAMP = ((0.70, 0.0, 0.0),          # dark red (thin, critical)
              (1.0, 0.549, 0.0),         # orange
              (0.945, 0.769, 0.059),     # yellow
              (0.18, 0.80, 0.44))        # green (thick, normal)
_WALL_DEFAULT_THR = (5.0, 7.0, 9.0)      # Measure-Compare 5/7/9 mm bands


def _wall_band_colors(count: int):
    """*count* RGB colours evenly sampled along _WALL_RAMP (count>=1). count==4 →
    the exact anchors (the Measure-Compare colours)."""
    count = max(1, int(count))
    if count == 1:
        return [_WALL_RAMP[0]]
    out = []
    m = len(_WALL_RAMP) - 1
    for i in range(count):
        x = i / (count - 1) * m
        j = min(int(x), m - 1)
        f = x - j
        a, b = _WALL_RAMP[j], _WALL_RAMP[j + 1]
        out.append(tuple(a[k] + (b[k] - a[k]) * f for k in range(3)))
    return out


def _wall_rgba_from_mm(mm, thresholds, alpha: float = 0.55):
    """Vectorised banded wall-thickness colouring: *mm* (float array) → (…,4) uint8
    RGBA using the clinical bands. Values <= 0.05 (no myocardium) → transparent.
    N ascending *thresholds* → N+1 bands red→green along _WALL_RAMP."""
    thr = sorted(float(x) for x in (thresholds or [])) or [5.0, 7.0, 9.0]
    cols = _wall_band_colors(len(thr) + 1)          # N+1 band colours
    mm = np.asarray(mm, np.float64)
    idx = np.zeros(mm.shape, int)
    for tv in thr:                                  # band index = #thresholds below
        idx += (mm >= tv).astype(int)
    a = int(round(max(0.0, min(1.0, alpha)) * 255))
    rgba = np.zeros(mm.shape + (4,), np.uint8)
    for bi, (r, g, b) in enumerate(cols):
        sel = (idx == bi) & (mm > 0.05)
        if sel.any():
            rgba[sel] = (int(r * 255), int(g * 255), int(b * 255), a)
    return rgba


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
        if p.get("invert"):                       # WB reverse (grayscale only)
            g = 1.0 - g
        gg = (g * 255.0).astype(np.uint8)
        rgb = np.stack([gg, gg, gg], axis=2)
    rgb = np.ascontiguousarray(rgb)
    return QImage(rgb.data, iw, ih, 3 * iw, QImage.Format.Format_RGB888).copy()


class _LvvWorker(QThread):
    """Runs the (few-second) LV blood-pool volume computation off the UI thread
    so a busy progress bar can animate; emits the result dict. Ported verbatim
    from the VTK viewer — the compute is pure-numpy so it is safe off-thread."""
    finished_result = pyqtSignal(object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            res = self._fn()
        except Exception:                                # noqa: BLE001
            import traceback
            res = {"error": "exc", "msg": traceback.format_exc()}
        self.finished_result.emit(res)


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
        # LV Vol voxel tints: cyan (in-range blood, 血流領域表示) then red
        # (measured region, 計測領域) drawn OVER the grayscale but UNDER the
        # crosshair / measures / markers. Cached RGBA images (see _refresh).
        if v._lvv is not None:
            cyan = v._lvv_cyan_img.get(key)
            if cyan is not None:
                p.drawImage(self.rect(), cyan)
            red = v._lvv_red_img.get(key)
            if red is not None:
                p.drawImage(self.rect(), red)
            thick = v._lvv_thick_img.get(key)
            if thick is not None:
                p.drawImage(self.rect(), thick)
                self._paint_wall_legend(p, v, w, h)
        # Short-axis (CPR): pane A shows a centred crosshair + the editable
        # control-point marker instead of the normal MPR crosshair / measures.
        if v._cpr is not None and key == "A":
            self._paint_cpr(p, w, h)
            self._paint_info(p, key, w, h)
            return
        if v._cl_on and not v._lv_cross_suppressed():
            self._paint_cross(p, key, w, h)
        self._paint_measures(p, key, w, h)
        if v._lv is not None:
            self._paint_lv(p, key, w, h)
        if v._lvv is not None:
            self._paint_lvv(p, key, w, h)
        self._paint_info(p, key, w, h)

    def _paint_cpr(self, p, w, h):
        """Short-axis overlay: a centred amber crosshair (the cross-section is
        centred on the vessel) plus the nearest control-point marker as a
        draggable dot."""
        v = self._v
        cx, cy = v._world_to_screen("A", 0.0, 0.0)   # section centre = output 0,0
        if v._cl_on:
            pen = QPen(QColor(255, 217, 0, 128), 1.0)
            p.setPen(pen)
            p.drawLine(QPointF(0, cy), QPointF(w, cy))
            p.drawLine(QPointF(cx, 0), QPointF(cx, h))
        # Editable pseudo-centre marker(s): the nearest control point, drawn at
        # its in-plane offset from the centreline, draggable to fine-tune it.
        for _ci, (du, dv) in v._cpr_marker_geom():
            mx, my = v._world_to_screen("A", du, dv)
            p.setPen(QPen(QColor(0, 0, 0, 200), 1.4))
            p.setBrush(QColor(255, 235, 0))
            p.drawEllipse(QPointF(mx, my), 5.0, 5.0)

    def _paint_wall_legend(self, p, v, w, h):
        """Colour-band legend for the 壁厚 heat map: a vertical bar (green=thick on
        top → red=thin) with the mm thresholds labelled at the colour boundaries."""
        thr = v._wall_thresholds()
        cols = _wall_band_colors(len(thr) + 1)
        n = len(cols)
        bw, seg = 22, 26                      # bar width / per-band height (2×)
        x0 = w - bw - 60
        y0 = h - seg * n - 44
        p.setFont(QFont("monospace", 12))
        for i in range(n):
            r, g, b = cols[n - 1 - i]         # top = last band (thick/green)
            p.fillRect(QRectF(x0, y0 + i * seg, bw, seg),
                       QColor(int(r * 255), int(g * 255), int(b * 255)))
        p.setPen(QColor(0, 0, 0, 160))
        p.drawRect(QRectF(x0, y0, bw, seg * n))
        for j, tv in enumerate(reversed(thr)):    # boundaries top→down = 9,7,5…
            yy = y0 + (j + 1) * seg
            _draw_outlined_text(
                p, QRectF(x0 + bw + 3, yy - 10, 56, 20),
                int(Qt.AlignmentFlag.AlignLeft) | int(Qt.AlignmentFlag.AlignVCenter),
                t("{v:g}mm").format(v=float(tv)), QColor(255, 255, 255), width=1.0)

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

        # Hover/drag highlight: the caught line goes vivid (dimmed) yellow and
        # opaque; the rotate zone also draws small double-headed arrows.
        hi = v._cross_hi.get(key)
        hl_line = hi[0] if hi else None
        both = hi is not None and hi[1] == "center"   # intersection → both lines
        base_pen = QPen(QColor(255, 217, 0, 128), 1.0)      # amber, 50%
        hi_pen = QPen(QColor(204, 204, 0, 255), 1.6)        # yellow, opaque
        # full-extent crosshair lines through the crosshair centre
        p.setPen(hi_pen if (both or hl_line == "H") else base_pen)
        p.drawLine(S(ccx - half * uh[0], ccy - half * uh[1]),
                   S(ccx + half * uh[0], ccy + half * uh[1]))
        p.setPen(hi_pen if (both or hl_line == "V") else base_pen)
        p.drawLine(S(ccx - half * uv[0], ccy - half * uv[1]),
                   S(ccx + half * uv[0], ccy + half * uv[1]))
        if hi is not None and hi[1] == "rotate":
            self._paint_rot_arrow(p, S, ccx, ccy, v._ps[key], uh, uv, hi[0])

        # ▲ markers: the OTHER pane's projection direction, a constant
        # fraction of the viewport from the centre (size tied to ps).
        ps = v._ps[key]
        d = 0.255 * ps
        sz = 0.024 * ps
        # Draw-only: the LEFT pane (A)'s ▲ points the opposite way (apex on the
        # −uv side). Visual only — the image/frame, the angle readout and the
        # paging-sense are all unchanged. Painted here every repaint, so it
        # persists through ROTATE/SPIN and reset. (Parity with VTK 948d500.)
        apex_sgn = ((-1.0 if key == "A" else 1.0)
                    * getattr(v, "_apex_flip", {}).get(key, 1.0))
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

    def _paint_rot_arrow(self, p, S, ccx, ccy, ps, uh, uv, line):
        """Two small double-headed curved arrows on BOTH sides of the caught
        line's outer ends — the 'rotates either way' hint. Drawn in output
        coords so it follows the crossline (redrawn each paint at the current
        angle). Compact + dimmed yellow to match the highlight."""
        base = uh if line == "H" else uv
        base_ang = math.atan2(base[1], base[0])
        r = 0.60 * ps
        span = math.radians(3.75)
        steps = 6
        hs = 0.0125 * ps
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(204, 204, 0, 255), 1.4))
        for side in (0.0, math.pi):                     # both ends of the line
            ca0 = base_ang + side
            arc = []
            for i in range(steps + 1):
                ang = ca0 - span + (2.0 * span) * i / steps
                arc.append((ccx + r * math.cos(ang), ccy + r * math.sin(ang)))
            p.drawPolyline(QPolygonF([S(ox, oy) for (ox, oy) in arc]))
            for tip, nxt in ((arc[-1], arc[-2]), (arc[0], arc[1])):
                tx, ty = tip[0] - nxt[0], tip[1] - nxt[1]
                tl = math.hypot(tx, ty) or 1.0
                tx, ty = tx / tl, ty / tl
                for deg in (28.0, -28.0):
                    cA = math.cos(math.radians(deg))
                    sA = math.sin(math.radians(deg))
                    bx = (-tx) * cA - (-ty) * sA
                    by = (-tx) * sA + (-ty) * cA
                    p.drawLine(S(tip[0], tip[1]),
                               S(tip[0] + bx * hs, tip[1] + by * hs))

    # -- measurements (outlines, calipers, handles, labels, results) -------
    # -- LV EF overlay (endo/epi splines, crossing dots, level/centre line,
    #    wall-thickness fill) — the pygfx equivalent of the VTK _redraw_lv ----
    def _paint_lv(self, p, key, w, h):
        v = self._v
        lv = v._lv
        if lv is None or lv["model"].axis is None:
            return
        ax = lv["model"].axis

        def S(pt):
            sx, sy = v._world_to_screen(key, pt[0], pt[1])
            return QPointF(sx, sy)

        # Apex markers stay visible in EVERY LV phase once an axis exists — but
        # ONLY the ACTIVE pass's marker is drawn (endo=red / epi=green), on the
        # long-axis (trace) pane and, in SAX, on the cross-section pane too.
        if key in (lv.get("pane"), lv.get("sax_pane")):
            tgt = lv.get("pass")
            if tgt == "endo":
                Pa, argb = lv["model"].endo_apex, (255, 64, 64)
            elif tgt == "epi":
                Pa, argb = lv["model"].epi_apex, (64, 200, 80)
            else:
                Pa, argb = None, None
            if Pa is not None:
                c = S(v._world3d_to_out(key, Pa))
                p.setPen(Qt.PenStyle.NoPen)
                # GLOW when a border point is within the convergence range (it
                # will snap to the apex): translucent halo + brighter, bigger dot.
                if v._lv_apex_glow(key):
                    p.setBrush(QColor(argb[0], argb[1], argb[2], 90))
                    p.drawEllipse(c, 13.0, 13.0)
                    p.setBrush(QColor(*[min(255, x + 130) for x in argb]))
                    p.drawEllipse(c, 8.0, 8.0)
                else:
                    p.setBrush(QColor(*argb))
                    p.drawEllipse(c, 6.0, 6.0)

        if lv.get("phase") != "contour":
            return                                # only apex markers pre-trace

        yellow = QColor(255, 210, 0)
        if lv.get("sax") is not None:
            along0 = float(lv["sax"])
            if key == lv.get("sax_pane"):
                # While a long-axis border VERTEX is dragged, colour the ONE
                # crossing that follows it GREEN so it's clear which yellow dot
                # moves. Identify its meridian from the edited vertex.
                edit_which, edit_mu = None, None
                e = getattr(v, "_edit", None)
                if e is not None and e.get("key") == lv.get("pane"):
                    em = v._measures[e["key"]][e["mi"]]
                    etag = em.get("_lv")
                    if (etag is not None and etag[1] in ("endo", "epi")
                            and em.get("pts3d")
                            and 0 <= e["vi"] < len(em["pts3d"])):
                        Pv = np.asarray(em["pts3d"][e["vi"]], float)
                        angs2 = lv["model"].plane_angles()
                        phi = angs2[etag[0] % len(angs2)]
                        s = float(np.dot(Pv - ax.apex, ax.meridian_dir(phi)))
                        edit_which = etag[1]
                        edit_mu = (phi % 360.0) if s >= 0 \
                            else ((phi + 180.0) % 360.0)
                border_sm, mark = {}, []
                for which in v._lv_sax_borders():   # single pass, or both
                    sp = lv["model"].short_axis_border_pts(along0, which)
                    if sp is None or len(sp) < 3:
                        continue
                    xy = [v._world3d_to_out(key, P) for P in sp]
                    border_sm[which] = _smooth_closed(xy)
                    gi = -1
                    if edit_which == which and edit_mu is not None:
                        bd = 1e9
                        for i, P in enumerate(sp):
                            d = np.asarray(P, float) - ax.apex
                            th = math.degrees(math.atan2(
                                float(d @ ax.binormal),
                                float(d @ ax.radial0))) % 360.0
                            dd = min(abs(th - edit_mu),
                                     360.0 - abs(th - edit_mu))
                            if dd < bd:
                                bd, gi = dd, i
                        # Only follow the crossing that TRULY matches the edited
                        # meridian (within half a meridian spacing). If this
                        # level has none there (asymmetric/apical "missing left
                        # dot"), highlight nothing rather than a different
                        # section-line's point.
                        if bd > 90.0 / max(1, lv["model"].n_planes):
                            gi = -1
                    for i, q in enumerate(xy):
                        mark.append((q, i == gi))
                if v._lv_wall and "endo" in border_sm and "epi" in border_sm:
                    self._paint_lv_wall(p, key, border_sm["endo"],
                                        border_sm["epi"])
                    self._paint_lv_wall_legend(p, w, h)
                for which, colr in (("endo", (255, 64, 64)),
                                    ("epi", (64, 192, 64))):
                    sm = border_sm.get(which)
                    if not sm:
                        continue
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(QColor(*colr), 2.2))
                    p.drawPolyline(QPolygonF([S(q) for q in sm]))
                p.setPen(Qt.PenStyle.NoPen)
                green = QColor(64, 220, 64)
                for q, is_edit in mark:               # fixed screen-size dots
                    p.setBrush(green if is_edit else yellow)
                    r = 7.0 if is_edit else 3.5       # green = 2× the yellow
                    p.drawEllipse(S(q), r, r)
                angs = lv["model"].plane_angles()
                md = ax.meridian_dir(angs[lv["plane_idx"] % len(angs)])
                u, vv, _n = v._frame[key]
                dx, dy = float(np.dot(md, u)), float(np.dot(md, vv))
                nrm = math.hypot(dx, dy) or 1.0
                dx, dy = dx / nrm, dy / nrm
                X = float(v._half)
                lw = 3.8 if v._lv_line_hi.get(key) else 2.4
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(yellow, lw))
                p.drawLine(S((-dx * X, -dy * X)), S((dx * X, dy * X)))
                hs = v._lv_handle_screen(key)   # ○ handle pinned to the pane edge
                if hs is not None:
                    p.drawEllipse(QPointF(hs[0], hs[1]), 9.0, 9.0)
            elif key == lv.get("pane"):
                _, y = v._world3d_to_out(key, ax.apex + along0 * ax.axis)
                X = float(v._half)
                lw = 3.8 if v._lv_line_hi.get(key) else 2.4
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(yellow, lw))
                p.drawLine(S((-X, y)), S((X, y)))
                hs = v._lv_handle_screen(key)   # ○ handle pinned to the pane edge
                if hs is not None:
                    p.drawEllipse(QPointF(hs[0], hs[1]), 9.0, 9.0)
            return
        # LONG-AXIS view: base-cut line ⟂ the axis at the common basal level.
        if key != lv.get("pane"):
            return
        rng = (lv["model"].along_range("endo")
               or lv["model"].along_range("epi"))
        if rng is not None:
            base = rng[1]
            X = float(v._half)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(yellow, 2.4))
            p.drawLine(S((-X, base)), S((X, base)))

    def _paint_lv_wall(self, p, key, endo_sm, epi_sm):
        v = self._v
        outer = [tuple(q) for q in epi_sm]
        inner = [tuple(q) for q in endo_sm]
        if len(outer) < 3 or len(inner) < 3:
            return
        cen = _polygon_centroid(outer)
        radials = _radial_gap_compare(outer, inner, cen, 1.0)
        n = len(radials)

        def S(pt):
            sx, sy = v._world_to_screen(key, pt[0], pt[1])
            return QPointF(sx, sy)

        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            a, b = radials[i], radials[(i + 1) % n]
            da = abs(b["ang"] - a["ang"]) % 360.0
            if 2.5 < da < 357.5:
                continue
            rgb = _hex_to_rgb(_gap_color(a["gap"]))
            p.setBrush(QColor(rgb[0], rgb[1], rgb[2], 140))
            p.drawPolygon(QPolygonF([S(a["inner"]), S(a["outer"]),
                                     S(b["outer"]), S(b["inner"])]))

    def _paint_lv_wall_legend(self, p, w, h):
        """Bottom-left wall-thickness colour key (red <5 / orange 5-7 /
        yellow 7-9 / green >9 mm), shown only while the wall map is up — the
        QPainter equivalent of the VTK viewer's _lv_update_wall_legend."""
        bands = _gap_legend()
        if not bands:
            return
        p.setFont(QFont("monospace", 11))
        _black, _white = QColor(0, 0, 0), QColor(255, 255, 255)
        lh = 16.0
        y = h - 40 - len(bands) * lh
        fl = int(Qt.AlignmentFlag.AlignLeft) | int(Qt.AlignmentFlag.AlignVCenter)
        for lab, hexc in bands:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(hexc))
            p.drawRect(QRectF(10, y - 10 - 3.6, 12, 12))
            _draw_outlined_text(p, QRectF(34, y - lh, 200, lh), fl, lab,
                                _white, 1.0, _black)
            y += lh

    # -- LV Vol (blood-pool LVEF) markers: apex (red) / seed (cyan) 3-D dots,
    #    and the Epi border (green) where it crosses the pane. The cyan/red
    #    voxel TINTS are drawn as RGBA images in paintEvent; this draws the
    #    point markers on top. Mirrors the VTK viewer's _lvv_add_marker /
    #    _lvv_show_epi (re-expressed as QPainter dots). --------------------
    def _paint_lvv(self, p, key, w, h):
        v = self._v
        lvv = v._lvv
        # Never draw the LV Vol overlay (Epi dots / landmarks) while contour LV
        # is active — its display belongs to the LV Vol bar's Epi button only.
        if lvv is None or v._lv is not None:
            return

        def Sd(P3):
            ox, oy = v._world3d_to_out(key, P3)
            sx, sy = v._world_to_screen(key, ox, oy)
            return QPointF(sx, sy)

        # Epi border (green dots) where the surface crosses this pane.
        if getattr(v, "_lvv_epi_show", False) and v._lvv_epi_surf is not None:
            try:
                pts = np.asarray(v._lvv_epi_surf._all_ring_points(), float)
            except Exception:                            # noqa: BLE001
                pts = None
            if pts is not None and len(pts):
                _u, _vv, n = v._axes_for(key)
                n = np.asarray(n, float)
                o = np.asarray(v._pc[key], float)
                dist = (pts - o) @ n
                tol = 0.75 * max(v._dims)
                near = pts[np.abs(dist) <= tol]
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(80, 220, 80))
                for P in near:
                    p.drawEllipse(Sd(P), 2.0, 2.0)

        # Auto-Endo表示 overlay: the auto endocardial envelope where it crosses
        # this pane (orange dots), independent of the Epi/blood overlays.
        if (getattr(v, "_lvv_endo_show", False)
                and getattr(v, "_lv_endo_auto_surf", None) is not None):
            try:
                pts = np.asarray(v._lv_endo_auto_surf._all_ring_points(), float)
            except Exception:                            # noqa: BLE001
                pts = None
            if pts is not None and len(pts):
                _u, _vv, n = v._axes_for(key)
                n = np.asarray(n, float)
                o = np.asarray(v._pc[key], float)
                dist = (pts - o) @ n
                tol = 0.75 * max(v._dims)
                near = pts[np.abs(dist) <= tol]
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 140, 40))
                for P in near:
                    p.drawEllipse(Sd(P), 2.0, 2.0)

        # LV Diameter: show WHERE it was measured — the SHORT-axis pane draws the
        # chord itself; the LONG-axis pane draws the section line at that level.
        surf = getattr(v, "_lv_endo_auto_surf", None)
        if surf is not None and getattr(surf, "axis", None) is not None:
            v._lvv_lv_diameter_mm()                   # populate _lvv_diam_pts
            pts = getattr(v, "_lvv_diam_pts", None)
            if pts is not None:
                axis = np.asarray(surf.axis.axis, float)
                axis = axis / (float(np.linalg.norm(axis)) or 1.0)
                u_ax, _v_ax, n_ax = v._axes_for(key)
                perp = abs(float(np.dot(np.asarray(n_ax, float), axis)))
                p.setPen(QPen(QColor(255, 235, 0), 2.4))
                p.setBrush(Qt.BrushStyle.NoBrush)
                if perp >= 0.6:                       # short-axis → the chord
                    p.drawLine(Sd(pts[0]), Sd(pts[1]))
                else:                                 # long-axis → section line
                    c = 0.5 * (pts[0] + pts[1])
                    d = np.asarray(u_ax, float)
                    d = d - float(np.dot(d, axis)) * axis
                    dn = float(np.linalg.norm(d))
                    if dn > 1e-6:
                        d = d / dn
                        wmm = 0.9 * float(v._ps.get(key, 60.0))
                        p.drawLine(Sd(c - wmm * d), Sd(c + wmm * d))

        # Apex (red) and seed (cyan) landmark dots.
        apex = lvv.get("apex")
        if apex is not None:
            p.setPen(QPen(QColor(0, 0, 0, 200), 1.4))
            p.setBrush(QColor(255, 64, 64))
            p.drawEllipse(Sd(np.asarray(apex, float)), 6.0, 6.0)
        seed = lvv.get("seed")
        if seed is not None:
            p.setPen(QPen(QColor(0, 0, 0, 200), 1.4))
            p.setBrush(QColor(64, 192, 255))
            p.drawEllipse(Sd(np.asarray(seed, float)), 5.0, 5.0)

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

        def dots_hollow(pts, color, r=4.0, width=1.4):
            # Off-plane depth cue: a hollow ring (中抜き) rather than a filled
            # dot, so an off-slice pseudo-centre reads as "behind/in front".
            # *r* is the OUTER radius (matches the filled dot's r); the stroke
            # straddles the path, so draw at r-width/2 to keep the outer edge
            # on r — the ring and the filled dot then share an outer diameter.
            rr = max(0.5, r - width / 2.0)
            p.setPen(QPen(color, width))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for q in pts:
                p.drawEllipse(S(q), rr, rr)

        # This pane's cutting-plane frame — used to fade the parts of a 3-D
        # vessel trace that lie OFF the shown slice (depth cue), matching the
        # VTK viewer's off-plane handling in _redraw_geom.
        _pu, _pv, _pn = v._axes_for(key)
        _po = v._pc[key]

        def off_flags(m):
            """Per-vertex True where the trace's 3-D point is >1 mm off this
            plane; None when the measure carries no matching 3-D points."""
            p3 = m.get("pts3d") if m["type"] == "polyline" else None
            if p3 is None or len(p3) != len(m["pts"]):
                return None
            return [abs(float(np.dot(np.asarray(P, float) - _po, _pn))) > 1.0
                    for P in p3]

        e = v._edit
        edit_mi = e["mi"] if (e and e["key"] == key) else -1
        edit_vi = e["vi"] if (e and e["key"] == key) else -1
        edit_ca = bool(e.get("ca")) if (e and e["key"] == key) else False
        # Hover highlight: the control point under the cursor turns green so the
        # user knows it will be grabbed before pressing (twin of the drag green).
        hh = v._meas_hover_handle
        hov_here = bool(hh and hh["key"] == key)
        hov_mi = hh["mi"] if hov_here else -1
        hov_vi = hh["vi"] if hov_here else -1
        hov_ca = bool(hh.get("ca")) if hov_here else False
        # Outline hover: the movable shape under the cursor draws green (it will
        # be grabbed to MOVE).
        ho = v._meas_hover_outline
        hov_out_mi = ho[1] if (ho and ho[0] == key) else -1

        for mi, m in enumerate(v._measures[key]):
            # Hidden by "Hide/Show All Result" (global) or this measure's own
            # right-click Hide → skip its line, handles and id label entirely.
            if v._results_hidden or m.get("hidden"):
                continue
            # MV/AoV valve rings are LOCKED: draw the outline ONLY (no vertex
            # handles, no long/short-diameter lines) — to change, redraw + Save.
            locked = bool(m.get("_lv_valve"))
            rgb = _hex_to_rgb(m.get("color"))
            if mi == hov_out_mi and not locked:  # outline hover → green
                rgb = (80, 220, 80)
            a4 = transp_to_alpha(m.get("transp", 0))
            # Point HU probe → a fixed-size "+" (two ~12 px segments) + #id.
            if m["type"] == "point":
                sp = S(m["pts"][0])
                p.setPen(QPen(QColor(rgb[0], rgb[1], rgb[2], a4), 1.8))
                p.drawLine(QPointF(sp.x() - 6, sp.y()),
                           QPointF(sp.x() + 6, sp.y()))
                p.drawLine(QPointF(sp.x(), sp.y() - 6),
                           QPointF(sp.x(), sp.y() + 6))
                p.setPen(QPen(QColor(255, 217, 0)))
                p.drawText(QPointF(sp.x() + 8, sp.y() - 6), f"#{m['id']}")
                continue
            of = off_flags(m)
            if of is not None and not m.get("smooth"):
                # Per-segment outline, three states (same as the VTK viewer):
                #   both endpoints ON  → solid, full alpha
                #   one endpoint OFF   → solid, 50% alpha (leaving the slice)
                #   both endpoints OFF → DOTTED, 50% alpha (fully out of range)
                # so "in the cross-section" vs "out of it" reads at a glance.
                verts = list(m["pts"])
                half_a = max(1, a4 // 2)
                p.setBrush(Qt.BrushStyle.NoBrush)
                for i in range(len(verts) - 1):
                    o0, o1 = of[i], of[i + 1]
                    seg_a = half_a if (o0 or o1) else a4
                    pen = QPen(QColor(rgb[0], rgb[1], rgb[2], seg_a), 1.8)
                    if o0 and o1:
                        pen.setStyle(Qt.PenStyle.DotLine)
                    p.setPen(pen)
                    p.drawLine(S(verts[i]), S(verts[i + 1]))
            else:
                draw_outline(v._outline(m), rgb, alpha=a4)
            if m["type"] in ("ellipse", "polygon") and not locked:
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
                ca_hov = (hov_ca and mi == hov_mi
                          and 0 <= hov_vi < len(ca["pts"]) and not ca_edit)
                ca_idle = [q for ci, q in enumerate(ca["pts"])
                           if not (ca_edit and ci == edit_vi)
                           and not (ca_hov and ci == hov_vi)]
                dots(ca_idle, QColor(255, 140, 0), 5.0)
                if ca_hov:
                    dots([ca["pts"][hov_vi]], QColor(59, 219, 90), 6.0)
                if ca_edit:
                    dots([ca["pts"][edit_vi]], QColor(59, 219, 90), 7.0)
            # Trace vertices (the vessel's pseudo-centres): on-plane ones draw
            # as solid yellow dots, off-plane ones as 50% hollow yellow rings so
            # the user sees which pseudo-centres sit in the shown cross-section.
            idle_on, idle_off, hov_pts = [], [], []
            if not locked:                          # valve ring = no handles
                for vi, q in enumerate(v._handles(m)):
                    if mi == edit_mi and not edit_ca and vi == edit_vi:
                        continue                      # the dragged one → green
                    if (not hov_ca and mi == hov_mi and vi == hov_vi):
                        hov_pts.append(q)             # hovered one → green
                        continue
                    if of is not None and vi < len(of) and of[vi]:
                        idle_off.append(q)
                    else:
                        idle_on.append(q)
            # In-range dot a touch larger than the off-plane ring so the
            # in-plane pseudo-centre reads as the more prominent of the two.
            dots(idle_on, QColor(255, 217, 0), 4.4)              # yellow handles
            dots_hollow(idle_off, QColor(255, 217, 0, 128), 3.3)  # off-plane 50%
            dots(hov_pts, QColor(59, 219, 90), 6.0)             # hover green
            if (not locked and mi == edit_mi and not edit_ca
                    and 0 <= edit_vi < len(m["pts"])):
                dots([m["pts"][edit_vi]], QColor(59, 219, 90), 7.0)  # green
            # numeric id label at the anchor
            p.setPen(QColor(255, 217, 0))
            fb = QFont("monospace", v._overlay_font_pt)
            fb.setBold(True)
            p.setFont(fb)
            ax, ay = v._world_to_screen(key, *v._anchor(m))
            p.drawText(QPointF(ax + 6, ay - 6), str(m["id"]))

        # CPR: on the MAP pane, mark where the short-axis is currently cut (the
        # scrubbed centreline point projected onto this plane) — the CT analogue
        # of the IVUS pull-back position marker. Green, matching the VTK viewer.
        if v._cpr is not None and key == v._cpr.get("src"):
            cl = v._cpr["cl"]
            i = int(v._cpr["idx"])
            if 0 <= i < len(cl.points):
                pw = np.asarray(cl.points[i], float) - _po
                dots([(float(np.dot(pw, _pu)), float(np.dot(pw, _pv)))],
                     QColor(59, 219, 90), 7.0)

        d = v._draft
        if d and d["pane"] == key and d["pts"]:
            hover = v._meas_hover
            # LV Endo/Epi trace: preview the committed points with the SAME
            # centripetal Catmull-Rom the final border uses, in the target's
            # colour (what you see is what you'll get); the last point → cursor
            # rubber-band stays straight.
            lv_trace = (v._lv is not None
                        and v._lv.get("phase") == "contour"
                        and v._lv.get("target") in ("endo", "epi")
                        and d["type"] == "polyline")
            if lv_trace:
                col = ((211, 47, 47) if v._lv["target"] == "endo"
                       else (46, 139, 87))
                pts = list(d["pts"])
                src = _smooth_open(pts) if len(pts) >= 3 else pts
                pen = QPen(QColor(*col), 2.0)
                pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                if len(src) >= 2:
                    p.drawPolyline(poly(src))
                if hover is not None and pts:
                    p.drawLine(S(pts[-1]), S(hover))
                dots(pts, QColor(255, 217, 0), 4.0)
            else:
                # Yellow DASHED preview that follows the cursor while points are
                # being placed (matches the XA/IVUS canvas), incl. the angle tool.
                pen = QPen(QColor(244, 208, 63), 1.2)
                pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                if d["type"] == "ellipse" and hover is not None:
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
        # In LV mode, prepend the LV status / volume result lines.
        lines = list(v._metrics.get(key, []))
        if v._lv is not None:
            lines = v._lv_status_lines() + lines
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
                       t("Click to select 2 Ellipse/Polygon data to compare"
                         "  ({n_sel}/2)", n_sel=n_sel))
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

        # Point-probe hover HU — follows the cursor, painted last (on top).
        ph = v._probe_hover
        if ph is not None and ph[0] == key:
            _, hx, hy, htxt = ph
            p.setFont(QFont("monospace", 11))
            p.setPen(QColor(255, 255, 102))
            p.drawText(QPointF(hx + 12, hy - 4), htxt)

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
        # LV volume breakdown readout (top-right, nested Epi ⊇ Endo ⊇ Blood):
        # Epi / Endo / Blood + LV Compact (Epi−Endo) + LV PapTned (Endo−Blood).
        lvv = getattr(v, "_lvv", None)
        if lvv is not None:
            vlines = []
            epi_ml = (v._lvv_epi_volume_ml()
                      if hasattr(v, "_lvv_epi_volume_ml") else None)
            endo_ml = (v._lvv_endo_volume_ml()
                       if hasattr(v, "_lvv_endo_volume_ml") else None)
            blood_ml = (float(lvv["last_ml"])
                        if lvv.get("last_ml") is not None else None)
            if epi_ml is not None:
                vlines.append(t("Epi Volume: {v:.1f} mL").format(v=epi_ml))
            if endo_ml is not None:
                vlines.append(t("Endo Volume: {v:.1f} mL").format(v=endo_ml))
            if blood_ml is not None:
                vlines.append(t("Blood Volume: {v:.1f} mL").format(v=blood_ml))
            if epi_ml is not None and endo_ml is not None:
                vlines.append(t("LV Compact Volume: {v:.1f} mL").format(
                    v=max(0.0, epi_ml - endo_ml)))
            if endo_ml is not None and blood_ml is not None:
                vlines.append(t("LV PapTned Volume: {v:.1f} mL").format(
                    v=max(0.0, endo_ml - blood_ml)))
            diam = (v._lvv_lv_diameter_mm()
                    if hasattr(v, "_lvv_lv_diameter_mm") else None)
            if diam is not None:
                vlines.append(t("LV Diameter: {v:.1f} mm").format(v=diam))
            if vlines:
                fb = QFont("monospace", 13)
                fb.setBold(True)
                p.setFont(fb)
                _draw_outlined_text(
                    p, QRectF(w * 0.50, 6, w * 0.50 - 6, 24 * len(vlines)),
                    int(Qt.AlignmentFlag.AlignRight)
                    | int(Qt.AlignmentFlag.AlignTop),
                    "\n".join(vlines), QColor(120, 220, 255), width=1.0)
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
        self.setWindowTitle(t("Angio Angle"))
        v = QVBoxLayout(self)
        v.addWidget(QLabel(t("Rotate to the angle of the corresponding "
                             "angiography view")))

        # Combos sized to their contents (+ the dropdown arrow) so the 3-letter
        # LAO/RAO·CRA/CAU labels aren't clipped on the right.
        def _fit_combo(cb: QComboBox) -> QComboBox:
            cb.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToContents)
            cb.setMinimumContentsLength(4)
            cb.setMinimumWidth(78)
            return cb

        r1 = QHBoxLayout()
        self._lr = _fit_combo(QComboBox())
        self._lr.addItems(["LAO", "RAO"])
        self._lr.setCurrentIndex(0 if prim >= 0 else 1)
        self._lr_val = QSpinBox()
        self._lr_val.setRange(0, 180)
        self._lr_val.setSuffix(" °")
        self._lr_val.setValue(abs(int(prim)))
        r1.addWidget(QLabel(t("Primary:")))
        r1.addWidget(self._lr, 1)
        r1.addWidget(self._lr_val, 1)
        v.addLayout(r1)

        r2 = QHBoxLayout()
        self._cc = _fit_combo(QComboBox())
        self._cc.addItems(["CRA", "CAU"])
        self._cc.setCurrentIndex(0 if sec >= 0 else 1)
        self._cc_val = QSpinBox()
        self._cc_val.setRange(0, 90)
        self._cc_val.setSuffix(" °")
        self._cc_val.setValue(abs(int(sec)))
        r2.addWidget(QLabel(t("Secondary:")))
        r2.addWidget(self._cc, 1)
        r2.addWidget(self._cc_val, 1)
        v.addLayout(r2)

        btns = QHBoxLayout()
        btns.addStretch(1)
        ok = QPushButton(t("OK"))
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        cancel = QPushButton(t("Cancel"))
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

    def __init__(self, bands, opacity, on_change, win=400.0, lvl=40.0,
                 smooth_mm=0.4, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("ColorMap Setting"))
        self.resize(560, 560)
        self._bands = [dict(b) for b in bands]
        self._opacity = float(opacity)
        self._on_change = on_change
        self._win, self._lvl = float(win), float(lvl)
        self._smooth_mm = float(smooth_mm)

        self._rows_host = QWidget()
        self._rows = QVBoxLayout(self._rows_host)
        self._rows.setContentsMargins(4, 4, 4, 4)
        self._rows.setSpacing(4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._rows_host)

        op_row = QHBoxLayout()
        op_row.addWidget(QLabel(t("Opacity")))
        self._op = QSlider(Qt.Orientation.Horizontal)
        self._op.setRange(0, 100)
        self._op.setValue(int(round(self._opacity * 100)))
        self._op.valueChanged.connect(self._op_changed)
        self._op_lbl = QLabel(f"{self._opacity:.2f}")
        op_row.addWidget(self._op, 1)
        op_row.addWidget(self._op_lbl)

        btns = QHBoxLayout()
        add = QPushButton(t("Add"))
        add.clicked.connect(self._add_band)
        rst = QPushButton(t("Reset"))
        rst.clicked.connect(self._reset)
        close = QPushButton(t("Close"))
        close.clicked.connect(self.accept)
        btns.addWidget(add)
        btns.addWidget(rst)
        btns.addStretch(1)
        btns.addWidget(close)

        from multi_dicomviewer.ui.hu_legend import HuLegend
        self._legend = HuLegend(
            lambda b, o, w, l: _band_lut_array(b, o, w, l)[:, :3],
            _HU_LO, _HU_HI)
        _leg_lbl = QLabel(t("Legend — groups / grayscale (W/L) / colour"))
        sm_row = QHBoxLayout()
        sm_row.addWidget(QLabel(t("Boundary smoothing")))
        self._smooth_sld = QSlider(Qt.Orientation.Horizontal)
        self._smooth_sld.setRange(0, 20)        # 0.0–2.0 mm, /10
        self._smooth_sld.setValue(int(round(self._smooth_mm * 10)))
        self._smooth_sld.setToolTip(
            t("Smooths the colour band boundaries so they read as curves, not "
              "a voxel-grid staircase. 0 = crisp. Higher softens fine detail."))
        self._smooth_sld.valueChanged.connect(self._on_smooth_changed)
        self._smooth_lbl = QLabel(f"{self._smooth_mm:.1f} mm")
        sm_row.addWidget(self._smooth_sld, 1)
        sm_row.addWidget(self._smooth_lbl)

        col = QVBoxLayout(self)
        col.addWidget(scroll, 1)
        col.addLayout(op_row)
        col.addLayout(sm_row)
        col.addWidget(_leg_lbl)
        col.addWidget(self._legend)
        col.addLayout(btns)
        self._rebuild()
        self._refresh_legend()

    def _on_smooth_changed(self, v):
        self._smooth_mm = v / 10.0
        self._smooth_lbl.setText(f"{self._smooth_mm:.1f} mm")
        self._emit()

    def _refresh_legend(self):
        self._legend.set_params(self._bands, self._opacity,
                                self._win, self._lvl)

    def _emit(self):
        self._on_change(self._bands, self._opacity, self._smooth_mm)
        self._refresh_legend()

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
        if getattr(self, "_legend", None) is not None:
            self._refresh_legend()

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
        h.addWidget(QLabel(t("Min")))
        lo = QSpinBox()
        lo.setRange(-1024, 4096)
        lo.setValue(int(b["lo"]))
        lo.valueChanged.connect(lambda v, i=idx: self._set(i, "lo", v))
        h.addWidget(lo)
        h.addWidget(QLabel(t("Max")))
        hi = QSpinBox()
        hi.setRange(-1024, 4096)
        hi.setValue(int(b["hi"]))
        hi.valueChanged.connect(lambda v, i=idx: self._set(i, "hi", v))
        h.addWidget(hi)
        en = QPushButton(t("Enabled") if b["on"] else t("Disabled"))
        en.setCheckable(True)
        en.setChecked(b["on"])
        en.clicked.connect(lambda _c, i=idx: self._toggle(i))
        h.addWidget(en)
        rm = QPushButton(t("Remove"))
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
            self, t("Band colour"))
        if col.isValid():
            self._bands[idx]["rgb"] = (col.redF(), col.greenF(), col.blueF())
            self._rebuild()
            self._emit()


# --------------------------------------------------------------- viewer
class CTViewer(CPRMixin, AbstractViewer):
    handles_modality = "CT"
    #: emitted by the series-navigation buttons ("first"/"prev"/"next"/"last")
    #: — the shell steps through this study's CT series (angio-style F/A nav)
    series_nav = pyqtSignal(str)
    #: CoSync short-axis broadcast — display index / rotation° (see CPRMixin)
    cpr_index_changed = pyqtSignal(int)
    cpr_rotation_changed = pyqtSignal(float)
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
    #: emitted with a measurement id when a committed result is un-committed
    #: (Resume trace) so the shell drops its stale Measure-History entry; the
    #: entry comes back under the SAME id when the trace is committed again.
    measurement_removed = pyqtSignal(int)
    #: HU colour map edited here — shell persists it and mirrors it onto every
    #: other CT pane so the colour map is global. Args:
    #: (bands, opacity, smooth_mm).
    colormap_changed = pyqtSignal(object, float, float)
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
        # by the Rt90/Lt90/Flip buttons. Default V = -y so the stored slice is
        # shown in raster order (pixel row 0 at the TOP, like any 2-D DICOM
        # viewer): the camera puts +V up, while DICOM rows grow downward.
        # V = +y (the old default) showed every native slice upside-down.
        self._axes2d = (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]))
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
        # ▲ apex-marker side per pane (±1). Flips on each Flip-H / Flip-V so the
        # ▲ mirrors WITH the image (the crosshair is drawn in output coords, so a
        # frame mirror doesn't auto-flip the directed ▲); rotations don't change
        # it. Reset to +1 when the default frames are rebuilt.
        self._apex_flip = {"A": 1.0, "B": 1.0}
        # Hover/drag centreline highlight: (line 'H'/'V', mode 'move'/'rotate')
        # or None per pane — drives the vivid-yellow highlight + rotate arrow.
        self._cross_hi = {"A": None, "B": None}
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
        # HU colour map is GLOBAL + persisted: load the shared bands so every
        # CT pane (and a fresh restart) starts from the same colour map.
        _cm = settings.load_ct_colormap()
        self._bands = [dict(b) for b in _cm["bands"]]
        self._opacity = float(_cm["opacity"])
        self._cmap_smooth_mm = float(_cm.get("smooth_mm", 0.4))
        self._cmap_dlg = None
        self._lut_key = None                 # cache key for the colormap tex
        self._lut_tex = None
        self._invert = False                 # WB reverse (grayscale negative)
        self._inv_key = None                 # cache key for the inverted ramp
        self._inv_tex = None
        # short-axis (CPR) state — shared logic lives in CPRMixin. None unless
        # the user turns a polyline trace into a vessel centreline (_enter_cpr);
        # pane A then becomes the cross-section scroller.
        self._cpr = None
        self._cpr_drag = None                # grabbed control-point index
        self._cpr_rot_prev = None            # dial-rotation anchor angle
        self._cpr_marker_pts = []            # [(ctrl_idx, (du,dv))] for hit-test
        # measurements
        self._meas_on = False
        self._meas_type = None               # line|polyline|ellipse|polygon|angle
        self._measures = {"A": [], "B": []}  # finalized {id,type,pts,...}
        self._meas_seq = 0
        self._snap_lumen = True              # snap trace clicks to the lumen
        self._draft = None                   # {type, pane, pts} in progress
        self._undo_clear()                   # unified Ctrl+Z / Ctrl+Y state
        self._edit = None                    # {key, mi, vi} handle drag
        self._meas_hover_handle = None       # {key, mi, vi, ca} handle under cursor
        self._meas_hover_outline = None      # (key, mi) shape outline under cursor
        self._meas_edit_before = None        # (pane, snapshot) at a handle press
        self._meas_edit_moved = False        # did the current measure drag move
        self._center_angle_target = None     # {key, mi} during 3-pt pick
        self._metrics = {"A": [], "B": []}   # per-measure result strings
        self._meas_drag = False              # canvas is dragging a handle
        self._alt_tool = False               # Alt/Option-press runs the tool, not measure
        self._meas_circle = False            # Shift while drawing → 正円 (Ellipse)
        self._meas_ortho = False             # Shift while drawing → 縦横直線 (Line)
        self._meas_hover = None              # cursor (output coords) for draft preview
        self._probe_hover = None             # (key, sx, sy, text) for the Point HU probe
        # Compare (%Area + radial gap map between two Polygon/Ellipse outlines)
        self._cmp_on = False                 # Compare-select mode: click 2 shapes
        self._cmp_sel = []                   # [(key, mi)] picked shapes (max 2)
        self._compares = []                  # persisted results (right-click→Delete)
        self._results_hidden = False         # "Hide/Show All Result" global toggle
        self._cmp_want_pa = False            # last-used: compute %PA (IVUS)
        self._cmp_want_thk = True            # last-used: compute Thickness (CT LV)
        self._mip_img = {"A": None, "B": None}   # slab-MIP QImage per pane
        # LV EF (ported from the VTK viewer): whole state in self._lv (None when
        # off). Rendering is done in the _Overlay (QPainter), not VTK actors.
        self._lv = None
        self._lv_view_free = False   # Epi領域表示 inspect state: lock lifted
        # Common valve planes (MV / AoV) shared by Endo/Epi/Blood as the LV base —
        # each is (centre_xyz, normal_xyz, radius) in volume mm, or None.
        self._lv_valves = {"mitral": None, "aortic": None}
        self._lv_valve_shown = {"mitral": True, "aortic": True}
        self._lv_dirty = False
        self._lv_result_lines = []
        self._lv_wall = False
        self._lv_line_drag = None             # "level"/"meridian" while grabbing
        self._lv_apex_drag = None             # "endo"/"epi" while dragging an apex
        self._lv_apex_hot = False             # apex glows: cursor in range while
        #                                       tracing (cleared on point confirm)
        self._lv_line_hi = {"A": False, "B": False}
        # LV Vol (blood-pool LVEF) mode, ported from the VTK viewer. self._lvv is
        # None when off, else a dict of landmarks (apex/aortic/mitral/seed/…). The
        # Epi surface (outer bound) is stashed from contour-LV mode on Exit LV, or
        # rebuilt on Load. Rendering is QPainter (markers) + cached RGBA tint
        # images (cyan in-range blood, red measured region) drawn in the overlay.
        self._lvv = None
        self._lvv_epi_surf = None             # LVSurface (Epi outer bound)
        self._lvv_epi_apex = None             # Epi apex (world mm)
        self._lvv_epi_model_dict = None       # LVModel dict for Save
        self._lvv_epi_ml = None               # cached epicardial volume (mL)
        self._lvv_hl_on = True                # 血流領域表示 (in-range tint) on
        self._lvv_epi_show = False            # Epi境界 (green border) on
        self._lvv_mask_on = False             # 計測領域 (red region) on
        self._lvv_mask_vol = None             # measured-region mask (0/1 float32)
        self._lvv_cyan_img = {"A": None, "B": None}   # in-range tint RGBA per pane
        self._lvv_red_img = {"A": None, "B": None}    # measured-region RGBA per pane
        self._lvv_thick_img = {"A": None, "B": None}  # 壁厚 heat-map RGBA per pane
        # 壁厚 (wall thickness) state — None/"3d"/"sax"; the field is a full-grid
        # numpy mm volume; cache lets mode re-entry skip the EDT/radial recompute.
        self._lvv_thick_mode = None
        self._lvv_thick_vol = None
        self._lvv_thick_stats = None
        self._lvv_thick_cache = {}
        try:
            from multi_dicomviewer.core import settings as _st
            self._lv_wall_thresholds = list(
                _st.load_lv_wall_bands()["thresholds"])
        except Exception:                     # noqa: BLE001
            self._lv_wall_thresholds = list(_WALL_DEFAULT_THR)
        self._lvv_worker = None               # keep the compute thread ref
        self._lvv_calc_then = None            # chained action after a measure
        self._lvv_blood_comp = None           # last Blood mask (bbox sub-volume)
        self._lvv_blood_bbox = None           # its (z0,z1,y0,y1,x0,x1)
        self._lvv_blood_apex = None           # apex used for that Blood measure
        # Auto-Endo表示 (display-only) vs Manual-Endo (edit) — two independent
        # Endo borders. endo_auto re-derived from blood on HU change; endo_manual
        # is hand-edited and RETAINED across HU changes.
        self._lv_endo_auto_model = None
        self._lv_endo_auto_surf = None
        self._lv_endo_auto_sig = None
        self._lv_endo_close_mm = 5.0      # Auto-Endo papillary/trabecula bridging
        self._lvv_endo_show = False
        self._lv_endo_manual_dict = None
        self._lv_endo_manual_mode = False
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
        # Persisted image-quality prefs. ct_quality_mode: 'high' = never coarse,
        # 'adaptive' = coarse only while moving (crisp when still, default),
        # 'low' = always coarse. (All set from Settings ▸ CT Image Quality.)
        self._dq = settings.load_display_quality()
        self._ct_quality = self._dq.get("ct_quality_mode", "adaptive")
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
        # Built BEFORE the toolbar (whose _set_tool touches the mode/side
        # buttons) but placed BELOW the image — angio-style: the top row
        # hosts Series First/Prev/Next/Last, the Plane/3D/2D switch sits
        # under the image.
        plane_bar = self._build_plane_bar()
        lay.addWidget(self._build_toolbar())
        self._measure_bar = self._build_measure_bar()
        self._measure_bar.setVisible(False)
        lay.addWidget(self._measure_bar)
        lay.addLayout(imgrow, 1)
        lay.addWidget(plane_bar)
        lay.addWidget(self._build_lv_bar())     # unified bar (builds the lvv bar)
        lay.addWidget(self._build_seek_bar())
        lay.addWidget(self._build_cpr_bar())

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
                          ("G", "PAGING"), ("T", "THICK"),
                          ("W", "WL")):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda t=tool: self._set_tool(t))
        # S is dual-purpose: Spin tool in 3-D, STOP the auto-page in 2-D
        # (Spin is MPR-only, so there is no conflict). D = play / ×2 toggle
        # in 2-D (angio cine-key parity); no 3-D action.
        sc_s = QShortcut(QKeySequence("S"), self)
        sc_s.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_s.activated.connect(self._key_s)
        sc_d = QShortcut(QKeySequence("D"), self)
        sc_d.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_d.activated.connect(self._key_d)
        sc_c = QShortcut(QKeySequence("C"), self)
        sc_c.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_c.activated.connect(self._key_toggle_color)
        # Ctrl+Z = unified undo, Ctrl+Y = redo. macOS note: Qt's portable "Ctrl"
        # maps to ⌘, so StandardKey.Undo/Redo are ⌘Z / ⌘⇧Z and "Ctrl+Y" is ⌘Y.
        # Physical-Ctrl ("Meta") shortcuts are NOT bound: on macOS a physical
        # Ctrl+letter is a control character, not a shortcut, so those never fire
        # (verified on device) — the toolbar Undo/Redo buttons cover the mouse /
        # remote-desktop case instead.
        sc_undo = QShortcut(QKeySequence.StandardKey.Undo, self)   # ⌘Z on macOS
        sc_undo.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_undo.activated.connect(self._undo_last)
        sc_redo = QShortcut(QKeySequence.StandardKey.Redo, self)   # ⌘⇧Z on macOS
        sc_redo.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_redo.activated.connect(self._redo_last)
        sc_redo2 = QShortcut(QKeySequence("Ctrl+Y"), self)         # ⌘Y on macOS
        sc_redo2.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_redo2.activated.connect(self._redo_last)
        # NB: A / F are app-wide ApplicationShortcuts (cine/series nav, see
        # MainWindow._nav_active). A viewer-level A/F QShortcut would collide
        # (two matching shortcuts → "ambiguous" → NEITHER fires), so LV plane-
        # stepping is routed via lv_nav_key() (called first by _nav_active).
        # Arrow keys drive the active tool (see _key_arrow). QShortcuts (not
        # keyPressEvent) so they fire over the wgpu canvas' own focus handling.
        for seq, direction in (("Up", "up"), ("Down", "down"),
                               ("Left", "left"), ("Right", "right")):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda d=direction: self._key_arrow(d))
        self._update_active_frames()

        # Last-resort safety net for the Mac "dead toolbar buttons" class of
        # bug. The primary fix defers every modal out of the pointer handler
        # so a grab can't get stuck — this guards anything that slips through:
        # if a canvas ever still holds the Qt mouse grab with NO gesture in
        # progress, a press elsewhere would be diverted to it and the toolbar
        # would go dead. Watch presses app-wide and drop such a stray grab so
        # the click reaches its real target. See eventFilter for why this is
        # cheap and can't disturb a legitimate grab.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def eventFilter(self, obj, ev):  # noqa: N802 (Qt override)
        """Drop a STRAY Qt mouse grab held by one of this viewer's canvases.

        Fires only on a mouse-button PRESS (so there is no per-frame / hover
        cost), and only acts when the grabber is one of OUR canvases AND no
        gesture is active (a real drag sets _drag_btn / _cross_grab /
        _meas_drag, so it is never touched; other widgets' grabs — sliders,
        menus — are never ours). A legitimate press that STARTS a drag is
        safe too: the grab is established only after this filter runs, so at
        filter time there is no grab yet. Never consumes the event."""
        if ev.type() == QEvent.Type.MouseButtonPress:
            gw = QWidget.mouseGrabber()
            if (gw is not None
                    and gw in (self.pane["A"].canvas, self.pane["B"].canvas)
                    and self._drag_btn is None
                    and not self._cross_grab
                    and not self._meas_drag):
                try:
                    gw.releaseMouse()
                except Exception:
                    pass
        return super().eventFilter(obj, ev)

    # ------------------------------------------------------ event wiring
    def _wire_events(self, key):
        c = self.pane[key].canvas
        c.add_event_handler(lambda ev, k=key: self._on_down(k, ev), "pointer_down")
        c.add_event_handler(lambda ev, k=key: self._on_move(k, ev), "pointer_move")
        c.add_event_handler(lambda ev, k=key: self._on_up(k, ev), "pointer_up")
        c.add_event_handler(lambda ev, k=key: self._on_leave(k, ev), "pointer_leave")
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
        # Right-click ON the bottom-centre angio readout → angle dialog
        # (rotate the slice to match a chosen LAO/RAO·CRA/CAU view). Checked
        # first, in any tool/measure mode, since it's a fixed screen target.
        if self._drag_btn == 2 and self._angio_hit(key, x, y):
            # Defer the modal dialog OUT of the pointer-down handler (same as
            # the compare-delete menu below). Running dlg.exec() inline here
            # swallows the matching pointer-UP, so the canvas keeps its mouse
            # grab and EVERY toolbar button goes dead afterwards — the Mac
            # "buttons unclickable after setting the angle" bug. Clear pointer
            # state now so no drag/grab is live while the dialog is up.
            self._reset_pointer_state()
            QTimer.singleShot(0, lambda k=key: self._open_angio_dialog(k))
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
        # LV apex points: place the two apex vertices (apex phase) or grab an
        # existing marker to drag it. Checked before Measure so a click near a
        # marker moves the apex instead of adding a trace point; _lv_apex_press
        # yields while a line is being drawn so tracing always wins.
        if self._drag_btn == 1 and self._lv is not None:
            # Alt/Option (not Shift) adjusts the view before placing the apex,
            # matching the measure passthrough (Shift is now the draw constraint).
            _adj = "Alt" in (ev.get("modifiers") or ())
            r = self._lv_apex_press(key, x, y, _adj)
            if r == "place":
                return
            if r in ("endo", "epi"):
                self._lv_apex_drag = r
                # Snapshot apex + borders now so the drag is one Ctrl+Z step.
                self._lv_apex_snap = self._lv_geom_snap()
                self._cross_grab = False
                self._meas_drag = False
                self._last = (x, y)
                return
        # SAX: grab the thick ○-marked LEVEL line (long-axis pane → translate the
        # cross-section level) or the CENTRELINE (short-axis pane → rotate the
        # meridian) to review the endo/epi borders. Checked BEFORE Measure so the
        # line wins, but _lv_line_press yields to a measure-handle grab so border
        # points still edit. The line thickens slightly while held.
        if self._drag_btn == 1 and self._lv_sax_active():
            kind = self._lv_line_press(key, x, y)
            if kind:
                self._lv_line_drag = kind
                # Snapshot SAX level/meridian now → one Ctrl+Z step on release.
                self._gesture_begin()
                self._lv_line_set_grabbed(key, True)
                self._cross_grab = False
                self._meas_drag = False
                self._last = (x, y)
                return
        # ALT/OPTION while measuring temporarily runs the SELECTED tool instead
        # of drawing (Zoom/Move/Thick/WL help you look / adjust the slab) — so
        # you can adjust the view mid-trace without leaving Measure. SHIFT is now
        # the DRAW CONSTRAINT (正円 / 縦横直線), matching PowerPoint and the VTK
        # (Windows) viewer. _alt_tool keeps _on_move on the tool path for the
        # rest of the drag (Alt may be released mid-gesture).
        _mods = ev.get("modifiers") or ()
        _alt = "Alt" in _mods
        self._alt_tool = bool(self._meas_on and _alt and self._drag_btn == 1)
        if self._meas_on and not _alt:
            self._cross_grab = False
            # SHIFT held while drawing constrains the shape: Ellipse → 正円,
            # Line → 縦横 (axis-aligned). Consumed by the draft/commit.
            _sh = "Shift" in _mods
            self._meas_circle = _sh
            self._meas_ortho = _sh
            # Left-click MEASURES while a type is selected or a Center-Angle
            # pick is in progress, and it also grabs an existing measure's
            # handle to edit it. Otherwise (Measure on but no type chosen) it
            # falls through to the active tool below, so Zoom/Move/… still
            # work — which is why the tools are only greyed once a type is
            # selected.
            capturing = (bool(self._meas_type)
                         or self._center_angle_target is not None)
            started = self._measure_left(key, x, y)
            if capturing or started:
                self._meas_drag = bool(started)
                return
            # idle Measure mode → fall through to the tool / crosshair setup
        # Right-click (not measuring): a single click exports a still image;
        # a double click forces the full-quality ("high-res") rebuild. Defer
        # the export by one double-click interval so a second right-press can
        # preempt it (the export menu is modal and would block the second
        # click otherwise).
        if self._drag_btn == 2:
            # Shift + right-click = full-quality ("high-res") rebuild — a
            # trackpad-friendly single gesture (the right-DOUBLE-click is kept
            # too). Checked HERE, not before the measure handling above, so a
            # Shift+right-click ON a measure element still reaches the measure
            # menu (Resume trace / Delete) — Windows parity, where Shift is
            # ignored on right-click and the measure menu always wins.
            if "Shift" in (ev.get("modifiers") or ()):
                self._force_crisp()
                return
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
        # Short-axis (CPR) pane A: a left-press grabs a control-point marker to
        # edit it, or (Rotate/Spin) starts the dial rotation. WL/Zoom/Move fall
        # through to the normal drag in _on_move; Paging is handled there too.
        # There is no MPR crosshair on the cross-section, so never grab it.
        if self._cpr is not None and key == "A":
            if self._drag_btn == 1:
                if self._cpr_grab(x, y):
                    return
                if self._tool in ("ROTATE", "SPIN"):
                    self._cpr_rot_start(x, y)
            return
        # A view-changing drag begins here (crosshair move/rotate OR the active
        # tool). Snapshot for Ctrl+Z now; committed as one step on release.
        if self._drag_btn == 1:
            self._gesture_begin()
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
        # LV apex drag: slide the grabbed apex along its long axis (border points
        # follow). Takes priority over everything while held.
        if self._lv_apex_drag is not None:
            if self._drag_btn == 1:
                self._lv_apex_move(key, x, y)
                self._last = (x, y)
            return
        # SAX line drag: move the level (long-axis pane) or rotate the meridian
        # (short-axis pane). Takes priority over measure/tool while held.
        if self._lv_line_drag is not None:
            if self._drag_btn == 1:
                self._lv_line_move(key, x, y)
                self._last = (x, y)
            return
        if self._meas_on and not self._alt_tool:
            if self._meas_drag:
                # Keep the Shift 縦横/正円 constraint live during the drag.
                _sh = "Shift" in (ev.get("modifiers") or ())
                self._meas_ortho = _sh
                self._meas_circle = _sh
                self._measure_drag(key, x, y)
                return
            if self._draft and self._draft["pane"] == key:
                # Update the dashed draft preview that follows the cursor.
                self._clear_hover_handle()
                self._meas_hover = self._disp_to_world(key, x, y)
                self._lv_apex_hover(key, x, y)     # glow apex if cursor in range
                self._overlay[key].update()
                return
            # Not drawing a draft here → hover-highlight (green) an existing
            # control point so the user sees it will be grabbed before pressing.
            self._measure_hover_handle(key, x, y)
            self._lv_apex_hover(key, x, y)         # glow apex if cursor in range
            # SAX: with no border point under the cursor, thicken the ○ line
            # handle so the user still sees the level / meridian is grabbable.
            if self._lv_sax_active() and self._meas_hover_handle is None:
                self._lv_line_hover(key, x, y)
            elif self._lv_sax_active():
                self._lv_line_set_grabbed(key, False)
            # Measuring with a type / Center-Angle pick (but not dragging) →
            # don't drive the tool. Idle Measure (no type chosen) falls
            # through so Zoom/Move/… still work.
            if bool(self._meas_type) or self._center_angle_target is not None:
                if self._meas_type == "point":
                    self._measure_hover(key, x, y)
                return
        else:
            self._clear_hover_handle()
        if self._drag_btn != 1:               # left-drag drives tool/crosshair
            # SAX: hover-thicken the level / centre line so the user sees where a
            # click will grab it (mirrors the crosshair hover preview).
            if self._lv_sax_active():
                self._lv_line_hover(key, x, y)
                return
            self._hover_cross(key, x, y)      # no button → preview centreline grab
            return
        if self._cross_grab:
            self._cross_move(key, x, y)
            self._last = (x, y)
            return
        dx, dy = x - self._last[0], y - self._last[1]
        self._last = (x, y)
        # Short-axis (CPR) pane A: dispatch the drag to the CPR handlers. A
        # grabbed marker moves; Rotate/Spin dials the section; Paging scrolls
        # the pull-back. WL/Zoom/Move fall through to the normal _drag (they
        # operate on the cross-section's window-level / camera).
        if self._cpr is not None and key == "A":
            if self._cpr_drag is not None:
                self._cpr_drag_move(x, y)
                return
            if self._tool in ("ROTATE", "SPIN"):
                self._cpr_rot_move(x, y)
                return
            if self._tool == "PAGING":
                self._cpr_page_drag(dy)
                return
        mods = ev.get("modifiers") or ()
        shift = "Shift" in mods
        # Ctrl (or Cmd/Meta on Mac) — used by the ZOOM tool so that WHILE tracing
        # a border, Shift zooms only THIS pane and Ctrl+Shift zooms both.
        ctrl = ("Control" in mods) or ("Meta" in mods)
        self._drag(key, dx, dy, shift, x, y, ctrl=ctrl)

    def _on_up(self, key, ev):
        # End an LV apex drag (record apex + borders as one Ctrl+Z step).
        if self._lv_apex_drag is not None:
            self._lv_apex_drag = None
            self._lv_record_geom(self._lv_apex_snap)
            self._lv_apex_snap = None
            self._drag_btn = None
            self._last = None
            return
        # End a SAX level/meridian line drag and drop its highlight.
        if self._lv_line_drag is not None:
            self._lv_line_drag = None
            self._lv_line_set_grabbed(key, False)
            self._lv_line_hi[key] = False        # re-hover on next move
            self._gesture_commit()               # commit level/meridian drag
            self._drag_btn = None
            self._last = None
            return
        if self._meas_on and self._meas_drag:
            self._measure_release()
        # End any short-axis marker drag / dial rotation.
        if self._cpr is not None:
            self._cpr_drag_end()
            self._cpr_rot_end()
        # Commit an intersection recentre: the point held under the cursor during
        # the drag jumps to the image centre in both panes (done on RELEASE).
        if self._cross_grab and self._cross_mode == "center":
            self._recenter(key, ev["x"], ev["y"])
        # Commit the drag (Zoom/Move/Rotate/Spin/Thick/W-L or a centreline
        # move·rotate) as one Ctrl+Z step. A click/double-click recentre records
        # itself (leaves _gesture_moved False), so this won't double-record.
        self._gesture_commit()
        self._meas_drag = False
        self._alt_tool = False
        self._drag_btn = None
        self._cross_grab = False
        self._spin_prev = None
        # End the centreline gesture and drop its highlight.
        if self._cross_hi.get(key) is not None:
            self._cross_hi[key] = None
            self._overlay[key].update()
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
        self._alt_tool = False
        self._meas_edit_before = None
        self._meas_edit_moved = False
        self._spin_prev = None
        self._lv_line_drag = None
        self._lv_apex_drag = None
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
        # Shift+double-click recentres even while Measuring (the trace follows
        # the moved image, see _recenter → _redraw_meas). A plain double-click
        # in Measure mode still finishes the polyline draft.
        if self._meas_on and "Shift" not in (ev.get("modifiers") or ()):
            self._measure_finish_draft()       # LV capture handled inside now
            return
        if self._cpr is not None and key == "A":
            return                                # no recenter on the section
        self._recenter(key, ev["x"], ev["y"])

    def _on_wheel(self, key, ev):
        # rendercanvas: wheel-up gives dy<0; page forward (+1) on wheel-up.
        step = 1 if ev["dy"] < 0 else -1
        # Short-axis: the wheel scrolls the cross-section along the vessel.
        if self._cpr is not None and key == "A":
            self._cpr_set_index(self._cpr_disp(self._cpr["idx"]) + step)
            return
        self._wheel(key, step)

    def showEvent(self, e):  # noqa: N802 (Qt override)
        """Re-fit and repaint once the pane is actually on screen.

        DEFENSIVE / belt-and-suspenders — mirrors the VTK viewer's showEvent.
        The shell calls ``load_series`` BEFORE it brings this viewer to the front
        of the pane's QStackedWidget, so the initial fit + draw in load_series can
        run while the wgpu canvas is still the hidden page: it then has no final
        size (``_fit_pane`` reads ``canvas.width()/height()`` for the aspect) and
        draws into an unexposed surface, which is the suspected cause of the
        intermittent BLACK-CT-until-reload symptom. Now that we are being shown,
        redo it — deferred to the next event-loop turn so Qt has settled the
        canvas geometry first.

        The ``_view_initial`` guard means a user who has already zoomed/panned
        keeps their view (we only repaint, not refit). This COMPLEMENTS the
        canvas ``resize``-event refit (``_on_resize``): if wgpu fires no resize on
        first show (unchanged size) that path never runs, so this closes the gap.
        The effect is unverified (the symptom is intermittent / low-repro), hence
        purely defensive — keep it a minimal, easily-reverted mirror of VTK."""
        super().showEvent(e)
        if self._vol is not None:
            QTimer.singleShot(0, self._refit_on_show)

    def _refit_on_show(self) -> None:
        # Guard: the viewer may have been cleared/destroyed before this fires.
        if self._vol is None:
            return
        self._refresh(reset_cam=self._view_initial)

    def _on_resize(self, key, ev):
        if self._vol is None:
            return
        # Short-axis (CPR) pane A has its OWN camera + plane (the oblique
        # cross-section, normal = vessel tangent). The normal MPR fit path
        # below would re-point the camera at the MPR frame and re-scale from
        # _ps["A"], leaving the camera mismatched with the CPR material plane
        # — the distorted sliver seen when switching Plane Bi->Lt in CPR.
        # Re-render through the CPR path so _config_cpr_cam picks up the new
        # aspect ratio (its "half" zoom is preserved).
        if self._cpr is not None and key == "A":
            p = self.pane[key]
            if p.material is not None:
                self._render_cpr_pane(p)
                self._overlay[key].update()
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

    # ---------------------------------------------------- live i18n switch
    def retranslate_ui(self) -> None:
        """Re-apply every persistent user-facing string via t() after a live
        UI-language switch (no restart), mirroring how the constructor built
        the toolbar / measure bar / seek bar.

        Safe to call whether or not a CT volume is loaded: all controls are
        guarded (they may be lazily built or None). On-image text drawn in
        paintEvent, right-click context menus and on-demand dialogs are NOT
        touched here — the overlays redraw (forced at the end) and the menus /
        dialogs are rebuilt every time they open.
        """
        # -- toolbar / measure-bar section labels ------------------------
        lbl = getattr(self, "_series_nav_lbl", None)
        if lbl is not None:
            lbl.setText(t("Series:"))
        nav_tips = (
            t("First series (Home)"),
            t("Previous series — shortcut: A"),
            t("Next series — shortcut: F"),
            t("Last series (End)"),
        )
        for b, tip in zip(getattr(self, "_nav_btns", []) or [], nav_tips):
            b.setHelpToolTip(tip)
        lbl = getattr(self, "_plane_lbl", None)
        if lbl is not None:
            lbl.setText(t("Plane:"))
        lbl = getattr(self, "_slab_lbl", None)
        if lbl is not None:
            lbl.setText(t("Slab:"))
        if getattr(self, "_slab_spin", None) is not None:
            self._slab_spin.setToolTip(
                t("Slab-MIP thickness of the active pane (0 = thin MPR)"))
        lbl = getattr(self, "_measure_lbl", None)
        if lbl is not None:
            lbl.setText(t("Measure:"))

        # -- DICOM-tag overlay font control (button text + tooltips) -----
        if getattr(self, "_tag_font_slider", None) is not None:
            self._tag_font_slider.setToolTip(t("DICOM tag text size"))
        if getattr(self, "_tags_btn", None) is not None:
            self._tags_btn.setText(t("DICOM Tags"))
            self._tags_btn.setToolTip(
                t("Choose which DICOM tags overlay the image (key Q shows/hides)"))

        # -- Plane (Bi/Lt/Rt) button tooltips ----------------------------
        side_tips = {
            "Bi": t("Show both MPR panes"),
            "Lt": t("Show only the left MPR pane"),
            "Rt": t("Show only the right MPR pane"),
        }
        for key, b in (getattr(self, "_side_btns", None) or {}).items():
            if key in side_tips:
                b.setHelpToolTip(side_tips[key])

        # -- 3D / 2D display-mode button tooltips ------------------------
        mode_tips = {
            "3D": t("3-D MPR reconstruction (dual oblique reslice)"),
            "2D": t("Show native acquisition slices one at a time (paging)"),
        }
        for key, b in (getattr(self, "_mode_btns", None) or {}).items():
            if key in mode_tips:
                b.setHelpToolTip(mode_tips[key])

        # -- ReCalc / Measure / CenterLine tooltips ----------------------
        b = getattr(self, "_recalc_btn", None)
        if b is not None:
            b.setHelpToolTip(t(
                "Re-derive the OTHER pane from the selected pane's green-▲ "
                "centre line — fixes a mirrored / wrong companion after "
                "complex rotations"))
        b = getattr(self, "_meas_btn", None)
        if b is not None:
            b.setHelpToolTip(t(
                "Measure on the image (Line / Polyline / Ellipse / Polygon "
                "/ Angle)"))
        b = getattr(self, "_cl_btn", None)
        if b is not None:
            b.setHelpToolTip(t("Show/hide crosshair & slab lines"))

        # -- Setting / Measure History / DICOM-tag button ----------------
        b = getattr(self, "_setting_btn", None)
        if b is not None:
            b.setText(t("Setting"))
            b.setHelpToolTip(t(
                "HU colour-map settings (band colour, HU range, opacity)"))
        b = getattr(self, "_hist_btn", None)
        if b is not None:
            b.setText(t("Measure History"))
            b.setHelpToolTip(t("Show this study's measurement history"))
        b = getattr(self, "_tags_btn", None)
        if b is not None:
            b.setToolTip(t(
                "Choose which DICOM tags overlay the image (key Q "
                "shows/hides)"))

        # -- 2-D image-transform (rotate / flip) tooltips ----------------
        t2d_tips = [
            t("Rotate the image 90° clockwise"),
            t("Rotate the image 90° counter-clockwise"),
            t("Flip horizontally (left-right mirror)"),
            t("Flip vertically (top-bottom)"),
        ]
        for b, tip in zip(getattr(self, "_t2d_btns", None) or [], t2d_tips):
            b.setHelpToolTip(tip)
        sb = getattr(self, "_spin_snap_btn", None)
        if sb is not None:
            sb.setText(t("Spin+"))
            sb.setHelpToolTip(t(
                "Snap the centreline to the nearest vertical/horizontal "
                "(45° snaps clockwise)"))
        b = getattr(self, "_invert_btn", None)
        if b is not None:
            b.setText(t("WB reverse"))
            b.setHelpToolTip(t("Invert grayscale (black↔white negative)"))

        # -- bottom 2-D seek bar -----------------------------------------
        lbl = getattr(self, "_seek_frame_lbl", None)
        if lbl is not None:
            lbl.setText(t("Frame:"))
        lbl = getattr(self, "_seek_series_cap", None)
        if lbl is not None:
            lbl.setText(t("Series:"))
        lbl = getattr(self, "_seek_series_lbl", None)
        if lbl is not None:
            lbl.setToolTip(t(
                "Series position in this study (current / total)"))

        # -- measure bar -------------------------------------------------
        b = getattr(self, "_cmp_btn", None)
        if b is not None:
            b.setText(t("Compare"))
            b.setHelpToolTip(t(
                "Compare two Polygon/Ellipse: click the two shapes — shows "
                "%Area difference and a radial gap colour map "
                "(<5 / 5–7 / 7–9 / >9 mm)"))
        b = getattr(self, "_hideall_btn", None)
        if b is not None:
            b.setHelpToolTip(t(
                "Hide / Show every measurement line, region colour and "
                "result text"))
        b = getattr(self, "_clr_btn", None)
        if b is not None:
            b.setText(t("Clear All Result"))
            b.setHelpToolTip(t(
                "Clear all measurements and comparison results"))
        lbl = getattr(self, "_cmp_hint", None)
        if lbl is not None:
            lbl.setText(t("  Left-click = add point /"
                          " right-click finishes Polyline / Polygon"))

        # Two-state toggle label (Hide All Result / Show All Result):
        # re-derive from the CURRENT state via its helper (never flip).
        if hasattr(self, "_update_hideall_btn"):
            self._update_hideall_btn()

        # Force on-image text (paintEvent overlays) + child widgets to
        # repaint so the DICOM-tag / readout text picks up the new language,
        # and cheaply re-render the current slice (self-guards vol=None).
        self.update()
        for ov in (getattr(self, "_overlay", None) or {}).values():
            ov.update()
        for name in ("_measure_bar", "_seek_wrap"):
            w = getattr(self, name, None)
            if w is not None:
                w.update()
        try:
            self._refresh()
        except Exception:
            pass

    # ------------------------------------------------------------ toolbar
    # -------------------------------------------- plane bar (below the image)
    def _build_plane_bar(self) -> QWidget:
        """Plane (Bi/Lt/Rt) + 3D/2D mode switch in a slim bar BELOW the image
        — angio-style layout: the top toolbar row hosts the Series
        First/Prev/Next/Last navigation instead. Hidden by "Max Image"
        together with the other toolbars (not flagged _mdv_keep_on_max)."""
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 0, 8, 0)

        # In-pane Plane switch: Bi (both MPR panes) / Lt (left) / Rt (right).
        self._plane_lbl = QLabel(t("Plane:"))
        row.addWidget(self._plane_lbl)
        self._side_btns: dict[str, QPushButton] = {}
        for key, tip in (
            ("Bi", t("Show both MPR panes")),
            ("Lt", t("Show only the left MPR pane")),
            ("Rt", t("Show only the right MPR pane")),
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
            ("3D", t("3-D MPR reconstruction (dual oblique reslice)")),
            ("2D", t("Show native acquisition slices one at a time (paging)")),
        ):
            b = FitButton(key)
            b.setCheckable(True)
            b.setChecked(key == "3D")
            b.setHelpToolTip(tip)
            b.setStyleSheet(_mode_css)
            b.clicked.connect(lambda _c, k=key: self._set_mode(k))
            self._mode_btns[key] = b
            row.addWidget(b)
        row.addStretch(1)
        for b in bar.findChildren(QPushButton):
            b.setMinimumWidth(min(b.sizeHint().width(), 56))
        return bar

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

        # Series navigation (angio-style): First/Prev/Next/Last step through
        # this study's CT series (the list behind the "Series: x/y" counter).
        # Icons sit LEFT of the text so narrow multi-pane layouts still show
        # which button is which (a right-side icon is the first thing elided).
        self._series_nav_lbl = QLabel(t("Series:"))
        row.addWidget(self._series_nav_lbl)
        self._nav_btns: list[QPushButton] = []
        for label, where, tip in (
            ("⏮ First", "first", t("First series (Home)")),
            ("◀ Prev (A)", "prev", t("Previous series — shortcut: A")),
            ("▶ Next (F)", "next", t("Next series — shortcut: F")),
            ("⏭ Last", "last", t("Last series (End)")),
        ):
            b = FitButton(label)
            b.setHelpToolTip(tip)
            b.clicked.connect(lambda _c, w=where: self.series_nav.emit(w))
            self._nav_btns.append(b)
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
        recalc.setHelpToolTip(t(
            "Re-derive the OTHER pane from the selected pane's green-▲ centre "
            "line — fixes a mirrored / wrong companion after complex rotations"))
        recalc.clicked.connect(self._recalc_companion)
        row.addWidget(recalc)

        self._cmap_btn = FitButton("ColorMap")
        self._cmap_btn.setCheckable(True)
        # Colour ON → a soft, muted (low-saturation) pale-yellow background.
        self._cmap_btn.setStyleSheet(
            "QPushButton:checked { background:#e3ddaa; color:#101010; }")
        self._cmap_btn.clicked.connect(self._toggle_color)
        row.addWidget(self._cmap_btn)

        self._meas_btn = FitButton("📏 Measure")
        self._meas_btn.setCheckable(True)
        self._meas_btn.setStyleSheet(            # blue when in Measure mode (= IVUS)
            "QPushButton:checked { background:#1f77b4; color:white; }")
        self._meas_btn.setHelpToolTip(t(
            "Measure on the image (Line / Polyline / Ellipse / Polygon / Angle)"))
        self._meas_btn.clicked.connect(self._toggle_measure)
        row.addWidget(self._meas_btn)

        self._slab_lbl = QLabel(t("Slab:"))
        row.addWidget(self._slab_lbl)
        self._slab_spin = QDoubleSpinBox()
        self._slab_spin.setRange(0.0, 50.0)
        self._slab_spin.setSingleStep(0.5)
        self._slab_spin.setDecimals(1)
        self._slab_spin.setToolTip(
            t("Slab-MIP thickness of the active pane (0 = thin MPR)"))
        self._slab_spin.valueChanged.connect(self._set_slab)
        row.addWidget(self._slab_spin)

        self._cl_btn = FitButton("CenterLine")
        self._cl_btn.setCheckable(True)
        self._cl_btn.setChecked(True)
        self._cl_btn.setHelpToolTip(t("Show/hide crosshair & slab lines"))
        self._cl_btn.clicked.connect(self._toggle_centerline)
        row.addWidget(self._cl_btn)

        # CT image quality (the old HQ-Img toolbar toggle) now lives entirely in
        # Settings ▸ CT Image Quality (Only Mac): high / adaptive / low.
        self._hires_btn = None

        # The per-pane "Setting" button (opened the HU colour-map editor) was
        # removed: the colour map is now global and edited from the shell's
        # "Settings" popup (CT colour ▸ Color setting…). _open_setting stays —
        # the shell calls it for the active CT pane.
        self._setting_btn = None

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
        self._tags_btn = tags
        tags.setToolTip(t(
            "Choose which DICOM tags overlay the image (key Q shows/hides)"))
        tags.clicked.connect(self.tags_requested.emit)
        self._tag_font_slider.valueChanged.connect(self.overlay_font_changed.emit)
        row.addWidget(tags_box)
        # DICOM-tag controls moved to the shell's global top row; hide the
        # per-viewer copy (kept only for set_overlay_font_pt slider sync).
        tags_box.setVisible(False)

        self._hist_btn = hist = FitButton(t("Measure History"))
        hist.setHelpToolTip(t("Show this study's measurement history"))
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
            ("Rt90°", "rt90", t("Rotate the image 90° clockwise")),
            ("Lt90°", "lt90", t("Rotate the image 90° counter-clockwise")),
            ("Flip-H", "fliph", t("Flip horizontally (left-right mirror)")),
            ("Flip-V", "flipv", t("Flip vertically (top-bottom)")),
        ):
            b = FitButton(label)
            b.setHelpToolTip(tip)
            b.clicked.connect(lambda _c, k=kind: self._2d_transform(k))
            self._t2d_btns.append(b)
            row2.addWidget(b)
        # Spin+ : snap the ACTIVE pane's centreline to the nearest vertical /
        # horizontal (a 45° tie snaps clockwise). Works in 2-D and 3-D — it just
        # rolls the on-screen view; the frame / measurements are unchanged.
        self._spin_snap_btn = FitButton(t("Spin+"))
        self._spin_snap_btn.setHelpToolTip(t(
            "Snap the centreline to the nearest vertical/horizontal "
            "(45° snaps clockwise)"))
        self._spin_snap_btn.clicked.connect(self._spin_snap)
        row2.addWidget(self._spin_snap_btn)
        # Grayscale invert (black↔white negative) — right of Spin+.
        self._invert_btn = FitButton(t("WB reverse"))
        self._invert_btn.setCheckable(True)
        self._invert_btn.setHelpToolTip(
            t("Invert grayscale (black↔white negative)"))
        self._invert_btn.clicked.connect(self._toggle_invert)
        row2.addWidget(self._invert_btn)
        # Undo / Redo buttons — set off to the right of WB reverse (same gap the
        # transforms have from WL). Mouse-clickable, so undo/redo works over
        # remote desktop (Parsec) where ⌘ can't reliably be sent. The shortcut in
        # the label is platform-correct (Cmd on macOS).
        import sys as _sys
        _mod = "Cmd" if _sys.platform == "darwin" else "Ctrl"
        row2.addSpacing(12)
        self._undo_btn = FitButton(f"Undo ({_mod}+Z)")
        self._undo_btn.setHelpToolTip(t("Undo the last action"))
        self._undo_btn.clicked.connect(self._undo_last)
        row2.addWidget(self._undo_btn)
        self._redo_btn = FitButton(f"Redo ({_mod}+Y)")
        self._redo_btn.setHelpToolTip(t("Redo the last undone action"))
        self._redo_btn.clicked.connect(self._redo_last)
        row2.addWidget(self._redo_btn)
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
        # Rotate/Paging can't be selected while an LV pass is locked/in SAX.
        if (self._lv_axis_locked() or self._lv_sax_active()) \
                and name in _LV_LOCK_DISABLED:
            return
        self._tool = name
        for n, b in self._tool_btns.items():
            b.setChecked(n == name)
        self._refresh_tool_availability()

    def _refresh_tool_availability(self):
        """Grey the interaction tools (Zoom/Move/…) while a measure TYPE is
        active: left-click then measures, so those tools can't be driven. The
        selected tool keeps a dimmed red so it stays identifiable. With no
        measure type (Measure off, or on but idle) they are their normal
        colour and usable (selected = red). MPR-only tools also stay disabled
        in 2-D native-slice mode."""
        is2d = getattr(self, "_mode", "3D") == "2D"
        # LV Trace/SAX (axis-locked long-axis OR SAX): Rotate/Paging disabled;
        # Zoom/Move/Thick/WL AND Spin stay live (via the Alt/Option passthrough).
        lv_lock = (self._lv_axis_locked() or self._lv_sax_active()) \
            if hasattr(self, "_lv") else False

        def _disabled(n):
            return ((is2d and n in _MPR_ONLY_TOOLS)
                    or (lv_lock and n in _LV_LOCK_DISABLED))

        # If the ACTIVE tool just became unavailable, fall back to Move (re-enters
        # _set_tool, which re-runs this refresh with a safe tool).
        cur = getattr(self, "_tool", None)
        if _disabled(cur) and cur != "MOVE":
            self._set_tool("MOVE")
            return
        for n, b in self._tool_btns.items():
            active = (n == cur)
            dis = _disabled(n)
            b.setEnabled(not dis)
            b.setChecked(active)
            if dis:                                  # greyed + unclickable
                b.setStyleSheet("background:#e6e6e6;color:#a8a8a8;"
                                "border:1px solid #d8d8d8;")
            elif active:                             # selected tool = red
                b.setStyleSheet("background:#c0392b;color:white;")
            else:
                # Non-selected but selectable → plain BLACK-text look so a
                # greyed/disabled tool is clearly distinguishable (VTK parity).
                b.setStyleSheet("")
        # CenterLine button: greyed + unclickable while the LV crosshair is
        # SUPPRESSED (apex placed → tracing) and in 2-D; auto-re-enables.
        if getattr(self, "_cl_btn", None) is not None:
            self._cl_btn.setEnabled(
                (not is2d) and not self._lv_cross_suppressed())
            self._style_cl()
        # WB reverse (grayscale invert) and the slab-thickness spin are disabled
        # throughout LV mode (thin slices, fixed grayscale).
        in_lv = getattr(self, "_lv", None) is not None
        if getattr(self, "_invert_btn", None) is not None:
            # WB reverse: 2-D only (unneeded in 3-D MPR, disabled in LV) — VTK
            # parity. Revivable by relaxing this to `not in_lv`.
            self._invert_btn.setEnabled(is2d and not in_lv)
        # Slab spin: available in ALL 3-D LV sub-modes (Endo/Epi AND Blood) so
        # the operator can adjust each pane's slab — Endo/Epi default to left 0 /
        # right 5 mm but may be changed. Disabled only in 2-D (VTK parity).
        if getattr(self, "_slab_spin", None) is not None:
            self._slab_spin.setEnabled(not is2d)

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
        prev = getattr(self, "_mode", None)
        if prev is not None and prev != mode:
            self._undo_clear()              # 2-D↔3-D: drop stale view undos
        self._mode = mode
        for k, b in self._mode_btns.items():
            b.setChecked(k == mode)
        is2d = (mode == "2D")
        self._refresh_tool_availability()   # MPR-only tools off in 2-D
        self._slab_spin.setEnabled(not is2d)
        self._cl_btn.setEnabled(
            (not is2d) and not self._lv_cross_suppressed())
        self._style_cl()
        for b in self._side_btns.values():
            b.setEnabled(not is2d)
        # Rt90/Lt90/Flip act on the native slice in 2-D and on the ACTIVE pane's
        # reslice frame in 3-D MPR — enabled in both (and in CPR).
        for b in self._t2d_btns:
            b.setEnabled(True)
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
            # Rebuild the anatomical (pbasis) frames when coming from 2-D:
            # pane A still carries the raster 2-D axes, which are NOT the
            # anatomical axial view (upside-down for a standard series) and
            # must not leak into the MPR. (Re-clicking "3D" while already in
            # 3-D keeps the user's oblique rotations.)
            if prev == "2D":
                self._init_frames()
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
        new_slice = int(min(max(self._slice2d + step, 0), max(0, nz - 1)))
        if new_slice == self._slice2d:
            return                          # at the end → nothing to page/undo
        before = self._view_snapshot()
        self._slice2d = new_slice
        z = self._slice2d * sz if sz > 1e-6 else 0.0
        self._center[2] = z
        self._pc["A"][2] = z
        self._clamp_center()
        self._view_initial = False
        self._refresh()
        self._sync_seek()
        self._undo_view(before, self._view_snapshot())

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

        # "Play" auto-pages through the native slices head → feet (the
        # conventional CT paging direction), looping back to the head end.
        # Fixed 10 slices/s — CT slices carry no cine rate to honour.
        self._play2d_btn = QPushButton("▶ Play")
        self._play2d_btn.setCheckable(True)
        _pf = self._play2d_btn.font()
        _pf.setPointSizeF(self._seek_base_pt * 1.55)
        _pf.setBold(True)
        self._play2d_btn.setFont(_pf)
        self._play2d_btn.setToolTip(
            t("Auto-page through the slices head → feet (loops; click again "
              "to stop). Keys: D = play / ×2 speed, S = stop"))
        self._play2d_btn.toggled.connect(self._toggle_play2d)
        row.addWidget(self._play2d_btn)
        self._play2d_speed = 1.0                 # ×2 via the D key
        self._play2d_timer = QTimer(self)
        self._play2d_timer.setInterval(100)      # 10 slices/s
        self._play2d_timer.timeout.connect(self._play2d_tick)
        #: True after a manual Pause → the next Play RESUMES from the current
        #: slice instead of rewinding to Frame 1. Cleared by forced stops
        #: (mode switch, new series, clear).
        self._play2d_resume = False
        self._seek_frame_lbl = _big(QLabel(t("Frame:")))
        row.addWidget(self._seek_frame_lbl)
        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setMinimum(0)
        self._seek_slider.setMaximum(0)
        self._seek_slider.setMinimumHeight(26)   # room for the 20px disc handle
        self._seek_slider.setStyleSheet(_SEEK_SLIDER_QSS)
        # The volume's slice index ascends toward the head (load_ct sorts by
        # IPP z), but conventional CT paging runs head → feet, so the slider
        # is inverted: left end = head (Frame 1), right end = feet.
        self._seek_slider.setInvertedAppearance(True)
        self._seek_slider.setInvertedControls(True)
        self._seek_slider.valueChanged.connect(self._on_seek)
        row.addWidget(self._seek_slider, 1)
        self._seek_lbl = _big(QLabel("1 / 1"))
        self._seek_lbl.setMinimumWidth(96)
        row.addWidget(self._seek_lbl)
        # Series position within the study's CT series (current / total), right
        # of the slice counter. It lives in the 2-D scrubber on purpose: that
        # bar is shown for native-slice (auxiliary, ≤200-slice) series and
        # hidden in 3-D MPR, so the counter appears exactly where the user
        # wants it (scout / Ca-score / thin recons) and not on the full 3-D
        # volume. A "Series:" caption (mirroring "Frame:") keeps it apart from
        # the adjacent slice "N / total". Fed by the shell via
        # set_series_position.
        self._seek_series_cap = _big(QLabel(t("Series:")))
        row.addWidget(self._seek_series_cap)
        self._seek_series_lbl = _big(QLabel(""))
        self._seek_series_lbl.setMinimumWidth(66)
        self._seek_series_lbl.setToolTip(t(
            "Series position in this study (current / total)"))
        row.addWidget(self._seek_series_lbl)
        self._seek_wrap.setVisible(False)
        if getattr(self, "_ct_compact", False):
            self._apply_seek_compact(True)
        return self._seek_wrap

    def set_series_position(self, index: int, total: int) -> None:
        """Show '<index+1>/<total>' (1-based) of this series within the study's
        CT list, beside the 2-D slice counter. Cleared when index<0/total<=0.
        Lives in the 2-D scrubber, so it is naturally hidden in 3-D MPR mode
        (the full recon) and shown for native-slice auxiliary series."""
        lbl = getattr(self, "_seek_series_lbl", None)
        if lbl is None:
            return
        cap = getattr(self, "_seek_series_cap", None)
        if total > 0 and 0 <= index < total:
            lbl.setText(f"{index + 1}/{total}")
            if cap is not None:
                cap.setVisible(True)
        else:
            lbl.setText("")
            if cap is not None:
                cap.setVisible(False)   # no dangling "Series:" caption

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
        for lbl in (self._seek_frame_lbl, self._seek_lbl,
                    self._seek_series_cap, self._seek_series_lbl):
            f = lbl.font()
            f.setPointSizeF(base * (1.0 if on else 1.55))
            f.setBold(not on)               # big = bold, compact = normal
            lbl.setFont(f)
        self._seek_lbl.setMinimumWidth(60 if on else 96)
        self._seek_series_lbl.setMinimumWidth(46 if on else 66)
        self._seek_slider.setMinimumHeight(16 if on else 26)
        self._seek_slider.setMaximumHeight(16 if on else _QWIDGETSIZE_MAX)
        self._seek_slider.setStyleSheet(
            _SEEK_SLIDER_QSS_COMPACT if on else _SEEK_SLIDER_QSS
        )
        pf = self._play2d_btn.font()
        pf.setPointSizeF(base * (1.0 if on else 1.55))
        pf.setBold(not on)
        self._play2d_btn.setFont(pf)

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
        # Counted from the head end (Frame 1 = most cranial slice), matching
        # the inverted slider direction.
        self._seek_lbl.setText(f"{nz - self._slice2d} / {nz}")

    def _sync_seek(self):
        """Show/refresh the scrubber to match the current mode and slice."""
        nz = self._vol.shape[0] if self._vol is not None else 1
        show = (self._mode == "2D" and nz > 1)
        if not show:
            self._play2d_btn.setChecked(False)   # stops the auto-page timer
            self._play2d_resume = False          # next Play starts at Frame 1
        self._seek_wrap.setVisible(show)
        if not show:
            return
        self._seek_slider.blockSignals(True)
        self._seek_slider.setMaximum(nz - 1)
        self._seek_slider.setValue(self._slice2d)
        self._seek_slider.blockSignals(False)
        self._seek_lbl.setText(f"{nz - self._slice2d} / {nz}")

    def _toggle_play2d(self, on):
        """Start/stop auto-paging (head → feet, looping) in 2-D mode.
        A fresh Play REWINDS to Frame 1 (the head end) first — the mid-stack
        slice shown on load is only the initial preview, not the start
        point. After a manual Pause, Play resumes from the paused slice."""
        if on:
            nz = self._vol.shape[0] if self._vol is not None else 1
            if self._mode != "2D" or nz <= 1:
                self._play2d_btn.setChecked(False)
                return
            self._play2d_btn.setText(
                "⏸ Pause ×2" if self._play2d_speed >= 1.5 else "⏸ Pause")
            if not self._play2d_resume:
                self._seek_slider.setValue(nz - 1)   # Frame 1 (fires _on_seek)
            self._play2d_timer.start()
        else:
            self._play2d_btn.setText("▶ Play")
            self._play2d_timer.stop()
            # Reaching here means playback WAS running (toggled only fires on
            # a state change) — treat as a manual Pause; the forced-stop
            # sites reset this right after their setChecked(False).
            self._play2d_resume = True

    def _play2d_tick(self):
        """One auto-page step toward the feet (the volume index ascends
        toward the head, so head → feet = descending index), looping back to
        the head end after the last slice."""
        if self._mode != "2D" or self._vol is None:
            self._play2d_btn.setChecked(False)
            return
        nz = self._vol.shape[0]
        nxt = self._slice2d - 1
        if nxt < 0:
            nxt = nz - 1
        # Drive through the slider so the handle follows (fires _on_seek).
        self._seek_slider.setValue(nxt)

    def _play2d_speed_toggle(self):
        """D (2-D mode): stopped → play at 1×; playing 1× → 2×; 2× → 1× —
        the same cine key the angio viewer uses."""
        if self._mode != "2D" or self._vol is None:
            return
        if not self._play2d_btn.isChecked():
            self._set_play2d_speed(1.0)
            self._play2d_btn.setChecked(True)    # fresh/resume rules apply
            return
        self._set_play2d_speed(2.0 if self._play2d_speed < 1.5 else 1.0)

    def _set_play2d_speed(self, speed: float) -> None:
        """Apply the auto-page speed (1× = 10 slices/s) and mirror it in the
        Pause label while playing."""
        self._play2d_speed = float(speed)
        self._play2d_timer.setInterval(int(round(100 / self._play2d_speed)))
        if self._play2d_btn.isChecked():
            self._play2d_btn.setText(
                "⏸ Pause ×2" if self._play2d_speed >= 1.5 else "⏸ Pause")

    def _key_s(self):
        """S: STOP the 2-D auto-page; Spin tool in 3-D (MPR-only)."""
        if self._mode == "2D":
            self._play2d_btn.setChecked(False)
        else:
            self._set_tool("SPIN")

    def _key_d(self):
        """D: 2-D play / ×2 speed toggle (no 3-D action)."""
        if self._mode == "2D":
            self._play2d_speed_toggle()

    # ------------------------------------------------- 2-D image transforms
    def _apply_2d_axes(self):
        """Set pane A's in-plane display axes (U, V) from the 2-D rotate/flip
        state. N = U×V (the actual viewing normal — the camera already views
        from the cross(U,V) side, and the LAO/CRA angle text reads this N);
        the cut plane is the same native slice either way, and 2-D paging
        moves along absolute z (_page2d), not along N."""
        u, v = self._axes2d
        u = np.asarray(u, float).copy()
        v = np.asarray(v, float).copy()
        self._frame["A"] = (u, v, np.cross(u, v))

    #: Rt90/Lt90/Flip as 2x2 matrices acting on the short-axis (u,v) frame —
    #: bu = T[0,0]·u0 + T[0,1]·v0, etc. (see CPRMixin._cpr_apply_xform).
    _XFORM_2X2 = {
        "rt90": np.array([[0.0, 1.0], [-1.0, 0.0]]),    # (u,v) -> (v, -u)
        "lt90": np.array([[0.0, -1.0], [1.0, 0.0]]),    # (u,v) -> (-v, u)
        "fliph": np.array([[-1.0, 0.0], [0.0, 1.0]]),   # (u,v) -> (-u, v)
        "flipv": np.array([[1.0, 0.0], [0.0, -1.0]]),   # (u,v) -> (u, -v)
    }

    def _2d_transform(self, kind):
        """Rotate the 2-D image 90° (rt90/lt90) or flip it (fliph/flipv).
        Applied incrementally to the current display axes (composable).
        In short-axis (CPR) mode the same buttons transform the cross-section
        via the CPR display transform T instead of the 2-D axes."""
        if self._cpr is not None:
            M = self._XFORM_2X2.get(kind)
            if M is None:
                return
            before = self._view_snapshot()
            self._cpr["T"] = M @ self._cpr["T"]
            self._cpr_apply_xform()
            self._view_initial = False
            self._refresh(reset_cam=True)
            self._undo_view(before, self._view_snapshot())
            return
        if self._vol is None:
            return
        if self._mode != "2D":
            # 3-D MPR: rotate / mirror the ACTIVE pane by transforming its reslice
            # frame (u, v, n). Pivot about the VISIBLE centre by transforming the
            # camera pan the SAME way, so the image flips/rotates in place (else a
            # panned/LV view would swing off-centre). pygfx derives the camera
            # from cross(u,v), so a mirror that flips handedness is fine; the
            # plane normal is kept consistent (rotations keep n, mirrors flip n).
            k = self._active_pane
            before = self._view_snapshot()
            u, v, n = (np.asarray(a, float) for a in self._frame[k])
            px, py = float(self._pan[k][0]), float(self._pan[k][1])
            if kind == "rt90":          # 90° clockwise
                self._frame[k] = (v, -u, n)
                npx, npy = py, -px
            elif kind == "lt90":        # 90° counter-clockwise
                self._frame[k] = (-v, u, n)
                npx, npy = -py, px
            elif kind == "fliph":       # left-right mirror
                self._frame[k] = (-u, v, -n)
                npx, npy = -px, py
            elif kind == "flipv":       # top-bottom flip
                self._frame[k] = (u, -v, -n)
                npx, npy = px, -py
            else:
                return
            # Turn the crosshair WITH the image: keep its world direction fixed
            # so it re-expresses in the new (u,v) basis (else the centreline stays
            # put while the anatomy rotates/flips around it).
            ca = self._cross_ang[k]
            self._cross_ang[k] = {"rt90": ca - 90.0, "lt90": ca + 90.0,
                                  "fliph": 180.0 - ca, "flipv": -ca}[kind]
            if kind in ("fliph", "flipv"):     # mirror → flip the ▲ side too
                self._apex_flip[k] = -self._apex_flip.get(k, 1.0)
            self._pan[k] = np.array([npx, npy])
            self._view_initial = False
            self._refresh(reset_cam=False, only=k)
            self._undo_view(before, self._view_snapshot())
            return
        before = self._view_snapshot()
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
        self._undo_view(before, self._view_snapshot())

    def _spin_snap(self) -> None:
        """Spin+ : roll the ACTIVE pane's view so its centreline (crosshair)
        snaps to the nearest vertical / horizontal (a 45° tie snaps CLOCKWISE).
        The camera roll rotates the on-screen view only — frame / measurements
        are unchanged. pygfx maps a roll R to on-screen angle = R − base, so the
        roll delta equals the screen-angle delta measured here."""
        if self._vol is None:
            return
        key = self._active_pane
        ccx, ccy = self._cc(key)
        a = math.radians(self._cross_ang[key])
        uh = (math.cos(a), math.sin(a))            # a crossline dir (output uv)
        s0 = self._world_to_screen(key, ccx, ccy)
        s1 = self._world_to_screen(key, ccx + uh[0], ccy + uh[1])
        sa = math.degrees(math.atan2(s1[1] - s0[1], s1[0] - s0[0]))
        # Nearest 90°; a 45° tie rounds so it snaps clockwise on screen.
        target = math.ceil(sa / 90.0 - 0.5) * 90.0
        delta = ((target - sa + 180.0) % 360.0) - 180.0
        if abs(delta) < 1e-4:
            return
        before = self._view_snapshot()
        self._roll[key] += delta                    # snaps the crossline
        self._view_initial = False
        self._refresh(only=key)
        self._undo_view(before, self._view_snapshot())

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
        self._undo_clear()                   # fresh Ctrl+Z history for the series

        # A genuinely new series → leave any LV mode and clear its state (the
        # borders/axis are tied to the previous series' volume coordinates).
        if self._lv is not None:
            self._lv = None
            self._lv_result_lines = []
            self._lv_wall = False
            if getattr(self, "_lv_btn", None) is not None:
                self._lv_btn.setChecked(False)
                self._lv_wall_btn.setChecked(False)
                self._lv_sax_btn.setChecked(False)
                self._lv_plane_lbl.setText("0/6")
                self._lv_sync_buttons()          # reset colours + grey the bar
        # A new series → drop any LV Vol state (landmarks/Epi/mask are in the
        # previous series' voxel coordinates).
        self._lvv_reset_state()

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
        self._src_dir = getattr(loaded, "source_dir", "") or ""   # data folder
        pb = loaded.patient_basis
        self._pbasis = (np.asarray(pb, dtype=np.float64)
                        if pb is not None else np.eye(3))
        self._win = self._win0 = float(loaded.window or 800.0)
        self._lvl = self._lvl0 = float(loaded.level or 200.0)
        self._thick = {"A": 0.0, "B": 5.0}
        # Reset any 2-D rotate/flip to the native orientation for the new series.
        self._axes2d = (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]))
        self._play2d_btn.setChecked(False)       # stop any running auto-page
        self._play2d_resume = False              # next Play starts at Frame 1
        self._set_play2d_speed(1.0)              # back to 1× for a new series

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

    def lv_active(self) -> bool:
        """True while an LV analysis session is in progress. The shell must NOT
        auto-demote such a pane to a memory-saving 'still': the LV state (mode,
        long axis, traced borders, computed volume) isn't captured by the still
        snapshot, so demoting garbles the image and loses the whole session."""
        return getattr(self, "_lv", None) is not None

    def snapshot(self):
        """A QPixmap of just the rendered CT image(s) — no toolbars — for the
        shell's memory-saving 'still' pane. Reuses the GPU-readback + overlay
        compositor (_grab_pane_qimage), so the frozen image matches what's on
        screen (WYSIWYG, correct aspect). Visible panes are composed side by
        side into one raw-pixel image. None on failure."""
        from PyQt6.QtGui import QColor, QImage, QPixmap
        imgs = []
        for key in ("A", "B"):
            if not self._frames[key].isVisible():
                continue
            qi = self._grab_pane_qimage(key)
            if qi is not None and not qi.isNull():
                imgs.append(qi)
        if not imgs:
            return None
        if len(imgs) == 1:
            return QPixmap.fromImage(imgs[0])
        h = max(i.height() for i in imgs)
        w = sum(i.width() for i in imgs) + 2 * (len(imgs) - 1)
        out = QImage(w, h, QImage.Format.Format_RGB32)
        out.fill(QColor(0, 0, 0))
        painter = QPainter(out)
        x = 0
        for i in imgs:
            painter.drawImage(x, 0, i)
            x += i.width() + 2
        painter.end()
        return QPixmap.fromImage(out)

    def clear(self) -> None:
        self._play2d_btn.setChecked(False)       # stop any running auto-page
        self._play2d_resume = False              # next Play starts at Frame 1
        self._cancel_lod()
        self._lod_pending = False
        self._lvv_reset_state()
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

        native=True gives the raw volume-axis frames (used by the 2-D lock,
        which pages the acquired slices as-is). V = -y so the stored slice
        shows in raster order (pixel row 0 at the top — DICOM rows grow
        downward while the camera puts +V up); these equal the pbasis frames
        for an identity basis, so the pb-None fallback behaves like a
        standard axial supine volume."""
        self._apex_flip = {"A": 1.0, "B": 1.0}   # fresh frames → default ▲ side
        pb = getattr(self, "_pbasis", None)
        if native or pb is None:
            self._frame = {
                "A": self._ortho(np.array([1.0, 0.0, 0.0]),
                                 np.array([0.0, -1.0, 0.0])),
                "B": self._ortho(np.array([1.0, 0.0, 0.0]),
                                 np.array([0.0, 0.0, 1.0])),
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

    # A traced polyline stores BOTH 2-D pane coords and absolute 3-D volume
    # coords (pts3d); these two convert between them for the plane live NOW.
    def _out_to_world3d(self, which, wx, wy):
        u, v, _n = self._axes_for(which)
        o = self._pc[which]
        return o + float(wx) * u + float(wy) * v

    def _world3d_to_out(self, which, P):
        u, v, _n = self._axes_for(which)
        d = np.asarray(P, dtype=float) - self._pc[which]
        return (float(np.dot(d, u)), float(np.dot(d, v)))

    # ---- lumen (high-HU) snapping for vessel tracing ----
    def _hu_along(self, P, n, ds):
        """Trilinear HU samples of the volume at P + d·n for each d in *ds*
        (world mm). Out-of-volume samples read as very low HU so they never
        win the brightest-peak search. Mirrors the VTK viewer."""
        if self._vol is None:
            return None
        sx, sy, sz = self._dims
        pts = np.asarray(P, float)[None, :] + np.asarray(ds, float)[:, None] * n
        fx = pts[:, 0] / sx
        fy = pts[:, 1] / sy
        fz = pts[:, 2] / sz
        nz, ny, nx = self._vol.shape
        inb = ((fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
               & (fz >= 0) & (fz <= nz - 1))
        out = np.full(len(ds), -2000.0)
        if not inb.any():
            return out
        val = _trilinear_sample(self._vol, fx, fy, fz)
        out[inb] = np.asarray(val, float)[inb]
        return out

    def _snap_to_lumen(self, P, n, reach=8.0, floor_hu=150.0):
        """Move P along ±*reach* mm of the plane normal *n* to the centre of
        the nearest contrast-bright (lumen) run — the depth the slab MIP hid.

        Picks the brightest run whose centre is closest to the click (so it
        can't jump to a distant bright structure), then returns its intensity-
        weighted centroid. No-op (returns P) if nothing rises above *floor_hu*
        or the volume isn't available."""
        n = np.asarray(n, float)
        nn = float(np.linalg.norm(n))
        if self._vol is None or nn < 1e-9:
            return np.asarray(P, float)
        n = n / nn
        step = max(0.25, min(self._dims) * 0.5)
        ds = np.arange(-reach, reach + step, step)
        hu = self._hu_along(P, n, ds)
        if hu is None:
            return np.asarray(P, float)
        peak = float(hu.max())
        if peak < floor_hu:
            return np.asarray(P, float)          # no lumen in reach → leave it
        thr = max(floor_hu, 0.5 * peak)
        bright = hu >= thr
        best = None                    # contiguous bright runs; pick nearest d=0
        i = 0
        N = len(ds)
        while i < N:
            if not bright[i]:
                i += 1
                continue
            j = i
            while j < N and bright[j]:
                j += 1
            centre = 0.5 * (ds[i] + ds[j - 1])
            if best is None or abs(centre) < abs(best[2]):
                best = (i, j, centre)
            i = j
        if best is None:
            return np.asarray(P, float)
        i, j, _c = best
        w = hu[i:j] - thr
        wsum = float(w.sum())
        d_star = float(np.dot(w, ds[i:j]) / wsum) if wsum > 1e-9 \
            else float(ds[(i + j) // 2])
        return np.asarray(P, float) + d_star * n

    def _snap_trace(self, which, mi):
        """Re-snap every vertex of a 3-D trace to the contrast lumen along the
        CURRENT plane normal (orient the plane roughly along the vessel first
        for the best axis). Redraws the trace afterwards."""
        m = self._measures[which][mi]
        p3 = m.get("pts3d")
        if not p3:
            return
        _, _, nrm = self._axes_for(which)
        m["pts3d"] = [self._snap_to_lumen(P, nrm) for P in p3]
        self._redraw_meas(which)

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
    def _slab_params(self, key, lod=False, native=False) -> dict:
        """Snapshot (on the GUI thread) everything _compute_slab_qimage needs:
        a plain dict of numpy/scalars + the (read-only, shared) volume — no Qt
        widget access — so the build can run on a worker thread.

        Three quality tiers: lod (coarse, during a drag), full (the immediate
        at-rest paint), and native (the off-thread settle build at ~2× the pane
        pixels so the STATIC slab is crisp on a Retina display)."""
        pane = self.pane[key]
        pw = max(1, pane.canvas.width())
        ph = max(1, pane.canvas.height())
        if lod:
            iw = min(pw, _SLAB_IW_LOD)
        elif native:
            iw = min(_SLAB_IW_NATIVE, pw * 2)
        else:
            iw = min(pw, _SLAB_IW_FULL)
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
            "invert": bool(self._invert),
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

    def _refresh(self, reset_cam=False, lod=False, only=None):
        # only="A"/"B" re-renders JUST that pane (single-pane drags: Move / Spin /
        # Thick / single-pane Zoom), where the companion cannot have changed — so
        # re-rendering (and rebuilding its slab MIP) every mouse-move was waste.
        # lod=True is an INTERACTIVE refresh (drag / wheel-page): the CPU slab-
        # MIP is built at REDUCED quality (coarse columns + fewer planes) rather
        # than skipped, so the THICK image keeps its slab look while staying
        # smooth on low-memory Macs. The debounce timer then rebuilds it at full
        # quality once the interaction settles.
        if self._vol is None:
            return
        # CT quality mode: 'high' never uses the coarse LOD (full every frame);
        # 'low' is always coarse (never crisps up); 'adaptive' keeps the caller's
        # lod (coarse only while moving, then a crisp settle when still).
        if self._ct_quality == "high":
            lod = False
        elif self._ct_quality == "low":
            lod = True
        # Any refresh (interactive frame or full) supersedes an in-flight async
        # high-res slab build, so bump the generation to discard a late result.
        self._slab_gen += 1
        for key in ("A", "B"):
            if only is not None and key != only:
                continue
            p = self.pane[key]
            if p.material is None:
                continue
            # Short-axis (CPR): pane A is the oblique cross-section, rendered by
            # its own plane + camera (the map pane keeps its normal MPR).
            if self._cpr is not None and key == "A":
                self._render_cpr_pane(p)
                self._overlay["A"].update()
                continue
            u, v, n = self._frame[key]
            pc = self._pc[key]
            p.material.plane = (float(n[0]), float(n[1]), float(n[2]),
                                float(-np.dot(n, pc)))
            # Anatomy-anchored traces (pts3d) re-project on ANY view change —
            # recentre / move / rotate / spin / page — not only on measure
            # edits, so the pseudo-centre points AND their line follow the
            # image even after Measure is switched off.
            self._reproject_traces(key)
            if self._color:
                # Colormap bakes W/L into the LUT over [_HU_LO,_HU_HI]; clim
                # maps that HU span to the LUT's 0..1 domain.
                p.material.clim = (_HU_LO, _HU_HI)
                p.material.map = self._lut_texture()
            else:
                self._gray_material(p)
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
            # LV Vol voxel tints follow the plane on any view change (recentre /
            # move / rotate / spin / page), so rebuild them whenever the pane is
            # refreshed while in LV Vol mode.
            if self._lvv is not None:
                self._lvv_refresh_overlays(key)
            p.render()
            self._overlay[key].update()
        # Whenever a slab is on screen, (re)arm the off-thread NATIVE-resolution
        # rebuild so the STATIC image crisps up once interaction settles — for
        # the coarse interactive frames AND the immediate full paint. In 'low'
        # mode the user asked for always-coarse, so skip the crisp settle.
        if self._ct_quality != "low" and any(self._thick[k] > 0
                                             for k in ("A", "B")):
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
        params = {k: self._slab_params(k, native=True)
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
    def _drag(self, which, dx, dy, shift=False, sx=None, sy=None, ctrl=False):
        if self._vol is None:
            return
        t = self._tool
        # LV short-axis is a DERIVED view: a Paging drag moves the cross-section
        # LEVEL; ROTATE is blocked because tilting the reslice FRAME would
        # corrupt the locked short-axis / long-axis geometry (use ◀ ▶ to rotate
        # the reference centreline instead). SPIN is allowed — it only rolls the
        # camera (the image AND the overlay rotate together), leaving the frame
        # and the reconstructed data untouched.
        if self._lv_sax_active():
            # Rotate/Paging disabled in SAX (Paging would move the long-axis level
            # on the long-axis pane; the ○ level line handle scrubs the level).
            # Zoom/Move/Thick/WL AND Spin (camera roll) still work.
            if t in ("PAGING", "ROTATE"):
                return
        # Long-axis view after Set axis: the axis is locked, so Rotate/Paging
        # (re-tilt the frame / shift the long-axis level) are blocked. Zoom/Move/
        # WL/Thick/Spin still work (they don't change the axis relationship).
        if self._lv_axis_locked() and t in ("ROTATE", "PAGING"):
            return
        if t != "WL":
            self._view_initial = False
        # This drag changes the view (W/L included, captured in the snapshot) →
        # its whole gesture becomes one Ctrl+Z step. Single-pane tools also skip
        # re-rendering the companion each move (only_pane).
        self._gesture_moved = True
        only_pane = None
        if t == "WL":
            self._win = max(1.0, self._win + dx * 2.0)
            self._lvl = self._lvl - dy * 2.0
        elif t == "PAGING":
            if self._mode == "2D":
                # 2-D: page integer native slices, ~6 px of drag per slice.
                # _page2d records each step itself → don't ALSO gesture-record.
                self._gesture_moved = False
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
            only_pane = which
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
            # Shift = zoom BOTH panes, else just this one — EXCEPT while actively
            # tracing a border, where a plain left-drag is taken by the trace so
            # Shift is the "run the tool" gate: there Shift alone zooms only THIS
            # pane (individual L/R zoom) and Ctrl+Shift zooms both.
            factor = 1.0 - dy * 0.005
            # While tracing a border OR anywhere in LV mode (align / SAX / trace,
            # so the two panes can be zoomed independently), Shift zooms only THIS
            # pane; add Ctrl (or Cmd/Meta on macOS) for both. Outside LV, Shift =
            # both, as before.
            indiv = ((self._meas_on and bool(self._meas_type))
                     or self._lv is not None)
            both = (shift and ctrl) if indiv else shift
            only_pane = None if both else which      # single-pane zoom skips other
            for k in (("A", "B") if both else (which,)):
                self._ps[k] = max(1e-3, self._ps[k] * factor)
        elif t == "MOVE":
            only_pane = which
            sc = self._ps[which] * 0.003
            px, py = self._pan[which]
            self._pan[which] = np.array([px - dx * sc, py + dy * sc])
        elif t == "SPIN":
            only_pane = which
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
        self._refresh(lod=True, only=only_pane)

    def _wheel(self, which, delta):
        if self._vol is None:
            return
        # SAX: the wheel scrolls the cross-section LEVEL (up = toward the apex),
        # so you can page through the short-axis stack without grabbing the line.
        if self._lv_sax_active():
            self._lv_step_level(1 if delta > 0 else -1)
            return
        if self._mode == "2D":
            self._page2d(1 if delta > 0 else -1)
            return
        before = self._view_snapshot()
        _, _, n = self._axes_for(which)
        # Wheel up = toward the ▲ apex (same convention as drag-paging).
        d = 1.0 if delta > 0 else -1.0
        mv = n * d * self._paging_sign(which) * min(self._dims)
        self._center = self._center + mv
        self._pc[which] = self._pc[which] + mv     # page only this pane
        self._clamp_center()
        self._view_initial = False
        self._refresh(lod=True)            # smooth wheel-paging (slab MIP defers)
        self._undo_view(before, self._view_snapshot())

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

    def _rotate_companion_by(self, which, d_deg) -> None:
        """Incremental companion coupling for a crossLINE ROTATE: the crossline
        turns by *d_deg* in *which*'s plane, so the companion plane turns by the
        same amount AROUND the shared axis n (= which's normal). Rotating the
        companion frame RIGIDLY around n keeps its CURRENT orientation (no snap to
        a fresh derivation) and holds its no-▲ centreline (which lies along n)
        fixed while its image turns about it — no drift/lock over many turns.
        (Ported from the VTK viewer.)"""
        if abs(float(d_deg)) < 1e-9:
            return
        other = "B" if which == "A" else "A"
        n = _norm(self._frame[which][2])
        u, v, _nn = self._frame[other]
        u2 = _rotate(u, n, d_deg)
        v2 = _rotate(v, n, d_deg)
        self._frame[other] = self._ortho(u2, v2)
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

    def _cross_zone(self, which, sx, sy):
        """Classify (sx, sy) against *which* pane's crosshair. Returns
        ``(caught, line, mode)`` — caught=False → off the crosshair (tool owns
        the drag); else line ∈ {"H","V"} (H = green-▲) and mode ∈
        {"move","rotate"}. Pure, so it drives BOTH press and hover.

        Distances are in NORMALISED screen space (each pane axis → [-1,1]); the
        catch band is 5%-of-screen on each side of a crossline, at any aspect /
        zoom. Of the caught span the INNER half translates the plane, the OUTER
        half rotates it."""
        if self._vol is None:
            return (False, None, None)
        ccx, ccy = self._cc(which)
        cx, cy = self._screen_center(which)           # crosshair centre, px
        pane = self.pane[which]
        hx = max(1.0, pane.canvas.width() / 2.0)
        hy = max(1.0, pane.canvas.height() / 2.0)
        a = math.radians(self._cross_ang[which])

        def _ndir(ux, uy):
            px, py = self._world_to_screen(which, ccx + ux, ccy + uy)
            dx, dy = (px - cx) / hx, (py - cy) / hy
            n = math.hypot(dx, dy) or 1.0
            return dx / n, dy / n

        uh = _ndir(math.cos(a), math.sin(a))          # along the H crossline
        uv = _ndir(-math.sin(a), math.cos(a))         # along the V crossline
        rx, ry = (sx - cx) / hx, (sy - cy) / hy
        band = 0.05                           # perpendicular catch = 2.5% screen/side
        mid = 0.50                            # inner half → move, outer → rotate
        d_to_h = abs(rx * uh[1] - ry * uh[0])
        along_h = abs(rx * uh[0] + ry * uh[1])
        d_to_v = abs(rx * uv[1] - ry * uv[0])
        along_v = abs(rx * uv[0] + ry * uv[1])
        on_h, on_v = d_to_h < band, d_to_v < band
        if not (on_h or on_v):
            return (False, None, None)
        # BOTH bands overlap → the crossline INTERSECTION: dragging here moves
        # the whole crosshair (a live recentre), like the VTK viewer.
        if on_h and on_v:
            return (True, "C", "center")
        grab_h = on_h
        along = along_h if grab_h else along_v
        mode = "move" if along <= mid else "rotate"
        return (True, "H" if grab_h else "V", mode)

    def _cross_press(self, which, sx, sy) -> bool:
        """Arm a MOVE/ROTATE crosshair gesture (above the tool) if the press
        lands ON the crosshair; else False. Locks the vivid highlight for the
        whole drag."""
        caught, line, mode = self._cross_zone(which, sx, sy)
        if not caught:
            return False
        # Intersection grab → live recentre (handled in _cross_move; committed on
        # release). No per-line state needed.
        if mode == "center":
            self._cross_mode = "center"
            self._cross_hi[which] = (line, mode)
            self._overlay[which].update()
            return True
        wx, wy = self._disp_to_world(which, sx, sy)   # world (gesture state)
        ccx, ccy = self._cc(which)
        a = math.radians(self._cross_ang[which])
        grab_h = (line == "H")
        if mode == "move":
            # Lock the slide to the grabbed line (parallel move of that line):
            # green-▲ (H) slides ⟂ itself = along uv; the V line slides along uh.
            ouh = np.array([math.cos(a), math.sin(a)])
            ouv = np.array([-math.sin(a), math.cos(a)])
            self._cross_mode = "move"
            self._cross_axis = ouv if grab_h else ouh
            self._cross_ppt = (wx, wy)
        else:
            self._cross_mode = "rotate"
            self._cross_prev = math.atan2(wy - ccy, wx - ccx)
        self._cross_hi[which] = (line, mode)
        self._overlay[which].update()
        return True

    def _hover_cross(self, which, sx, sy) -> None:
        """Mouse moved over a pane with no button down: preview whether a press
        would grab the centreline (vivid highlight + rotate arrow) or fall
        through to the tool (normal amber crosshair)."""
        if self._cross_grab or self._vol is None or not self._cl_on:
            return
        caught, line, mode = self._cross_zone(which, sx, sy)
        new = (line, mode) if caught else None
        if self._cross_hi.get(which) != new:
            self._cross_hi[which] = new
            self._overlay[which].update()

    def _on_leave(self, key, ev) -> None:
        # Cursor left the pane → drop any hover highlight (not mid-drag) + the
        # point-probe HU readout.
        if not self._cross_grab and self._cross_hi.get(key) is not None:
            self._cross_hi[key] = None
            self._overlay[key].update()
        self._measure_hover_clear(key)

    def _cross_move(self, which, sx, sy):
        self._gesture_moved = True             # centreline drag = one Ctrl+Z step
        wx, wy = self._disp_to_world(which, sx, sy)
        u, v, n = self._frame[which]
        other = "B" if which == "A" else "A"
        if self._cross_mode == "center":
            # Intersection drag: the crosshair centre FOLLOWS the cursor while
            # the background images stay put (only _center moves, not _pc); the
            # actual recentre happens on release (see _on_up → _recenter).
            self._center = self._pc[which] + wx * u + wy * v
            self._clamp_center()
            self._view_initial = False
            self._refresh()
            return
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
        # Turn the companion by the SAME increment about the shared axis, from
        # its CURRENT orientation — no snap, no drift/lock over many turns, and
        # the companion's no-▲ line (the shared axis) stays put while its image
        # rotates about it (matches the fixed VTK viewer).
        self._rotate_companion_by(which, d)
        self._view_initial = False
        self._refresh(lod=True)                # coarse slab while dragging

    def _recenter(self, which, sx, sy):
        """Double-click: clicked point becomes the CrossLine centre AND the
        image centre in both panes."""
        if self._vol is None:
            return
        # A click / double-click recentre records its own undo; a crosshair-CENTRE
        # drag is committed by the gesture instead (leaves _gesture_moved True),
        # so skip self-recording then to keep it one Ctrl+Z step.
        before = (None if getattr(self, "_gesture_moved", False)
                  else self._view_snapshot())
        wx, wy = self._disp_to_world(which, sx, sy)
        u, v, _n = self._frame[which]
        self._center = self._pc[which] + wx * u + wy * v
        self._clamp_center()
        self._pc = {"A": self._center.copy(), "B": self._center.copy()}
        self._pan = {"A": np.zeros(2), "B": np.zeros(2)}
        self._view_initial = False
        self._refresh()
        if before is not None:
            self._undo_view(before, self._view_snapshot())

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
        u = np.asarray(self._frame[key][0], dtype=np.float64)
        v = np.asarray(self._frame[key][1], dtype=np.float64)
        # Patient-space viewing normal = cross of the MAPPED in-plane axes.
        # (pbasis @ n would flip the sign when the volume is stored along
        # −x/−y — a left-handed pbasis, e.g. a coronal/sagittal stack sorted
        # A→P / R→L — misreporting the viewed side by 180°.)
        n = np.cross(self._pbasis @ u, self._pbasis @ v)   # -> patient LPS
        nrm = float(np.linalg.norm(n))
        if nrm < 1e-9:
            return None
        n = n / nrm
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
        sup_pat = np.array([0.0, 0.0, 1.0])          # patient superior
        if abs(float(np.dot(sup_pat, n_pat))) > 0.999:   # looking down SI axis
            sup_pat = np.array([0.0, -1.0, 0.0])     # fall back: anterior
        # Build the screen axes in PATIENT space and map EACH into volume
        # coords. (Mapping only the normal via inv loses a sign for a
        # left-handed pbasis — see _angio_angle_vals — so the pane would
        # show the opposite side to the angle requested.)
        u_pat = _norm(np.cross(sup_pat, n_pat))      # screen-right
        v_pat = np.cross(n_pat, u_pat)               # screen-up
        u = _norm(inv @ u_pat)
        v = _norm(inv @ v_pat)
        n = np.cross(u, v)                           # volume coords (u×v = n)
        return (u, v, n)

    def _set_angio_angle(self, which, prim_deg, sec_deg):
        """Re-orient pane *which* so it projects from the given C-arm angle,
        pivoting about the CrossLine intersection (_center). The companion
        pane re-derives as the coupled orthogonal section — same linkage as
        the ROTATE tool."""
        if self._vol is None or self._mode != "3D":
            return
        before = self._view_snapshot()
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
        self._undo_view(before, self._view_snapshot())   # Ctrl+Z / Ctrl+Y

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
        pane to that C-arm angle (to line the slice up with an angio view).

        NON-MODAL by design. A modal ``exec()`` spins a nested Qt event loop
        which, on macOS, hijacks the in-flight TRACKPAD gesture / first
        responder: after the dialog closes the trackpad can no longer click
        the toolbar (the mouse still works, and clicking with the mouse frees
        it). The stray "TSMSendMessageToUIServer FAILED" console spam is the
        same first-responder disturbance. A modeless dialog has no nested
        loop, so input routing is never taken from the trackpad. The result
        is applied from the dialog's ``accepted`` signal instead of a return
        value, and focus is handed back to the viewer when it closes."""
        vals = self._angio_angle_vals(which) or (0, 0)
        # Close any dialog still open from a previous click. Guard against its
        # C++ object already being gone (RuntimeError) so a stale reference
        # can't crash the reopen. (Do NOT use WA_DeleteOnClose: closing via
        # the window's ✕ deletes the dialog without firing `finished`, which
        # left _angio_dlg dangling and crashed the next open.)
        old = getattr(self, "_angio_dlg", None)
        self._angio_dlg = None
        if old is not None:
            try:
                old.close()
            except RuntimeError:
                pass
        dlg = _AngioAngleDialog(vals[0], vals[1], self)
        self._angio_dlg = dlg
        dlg.setModal(False)

        def _finished(result):
            # Fires for OK (Accepted) AND Cancel/Esc/✕ (QDialog routes the
            # window-close to reject → Rejected), so cleanup always runs.
            if result == QDialog.DialogCode.Accepted:
                try:
                    prim, sec = dlg.values()
                    self._set_angio_angle(which, prim, sec)
                except RuntimeError:
                    pass
            if self._angio_dlg is dlg:
                self._angio_dlg = None
            dlg.deleteLater()
            self._after_angio_dialog()

        dlg.finished.connect(_finished)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _after_angio_dialog(self):
        """Hand keyboard / first-responder focus back to the viewer after the
        angle dialog closes, so macOS routes trackpad + IME input to it again,
        and clear any stale pointer state."""
        self._reset_pointer_state()
        try:
            win = self.window()
            if win is not None:
                win.activateWindow()
            self.setFocus(Qt.FocusReason.OtherFocusReason)
        except Exception:
            pass

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
        self._measure_lbl = QLabel(t("Measure:"))
        row.addWidget(self._measure_lbl)
        self._meas_btns = {}
        for label, key in (("Point", "point"),
                           ("Line", "line"), ("Polyline", "polyline"),
                           ("Ellipse", "ellipse"), ("Polygon", "polygon"),
                           ("Angle", "angle")):
            b = FitButton(t(label) if key == "point" else label)
            if key == "point":
                b.setHelpToolTip(
                    t("Probe HU at a point: hover shows the value, click drops "
                      "a + marker and lists it"))
            b.setMinimumWidth(min(b.sizeHint().width(), 56))
            b.setCheckable(True)
            b.clicked.connect(lambda _c, k=key: self._set_measure_type(k))
            self._meas_btns[key] = b
            row.addWidget(b)
        # Compare two Polygon/Ellipse: %Area difference + radial gap colour map.
        # Placed right of Angle, with Clear All Result to its right.
        self._cmp_btn = FitButton(t("Compare"))
        self._cmp_btn.setMinimumWidth(min(self._cmp_btn.sizeHint().width(), 64))
        self._cmp_btn.setCheckable(True)
        self._cmp_btn.setHelpToolTip(t(
            "Compare two Polygon/Ellipse: click the two shapes — shows %Area "
            "difference and a radial gap colour map (<5 / 5–7 / 7–9 / >9 mm)"))
        self._cmp_btn.clicked.connect(self._toggle_compare)
        row.addWidget(self._cmp_btn)
        # Hide/Show ALL results (lines + region colours + text) at once, between
        # Compare and Clear All Result. Same grey as ReCalc; disabled when there
        # is nothing to hide.
        self._hideall_btn = FitButton(t("Hide All Result"))
        self._hideall_btn.setMinimumWidth(
            min(self._hideall_btn.sizeHint().width(), 64))
        self._hideall_btn.setHelpToolTip(t(
            "Hide / Show every measurement line, region colour and result text"))
        self._hideall_btn.setStyleSheet(                     # light grey, black text
            "QPushButton { background:#bdbdbd; color:#101010; }")
        self._hideall_btn.clicked.connect(self._toggle_hide_all)
        row.addWidget(self._hideall_btn)
        self._clr_btn = clr = FitButton(t("Clear All Result"))
        clr.setMinimumWidth(min(clr.sizeHint().width(), 56))
        clr.setHelpToolTip(t("Clear all measurements and comparison results"))
        clr.setStyleSheet(                                    # Reset's darker grey
            "QPushButton { background:#6e6e6e; color:#d8d8d8; }")
        clr.clicked.connect(self._measure_clear)
        row.addWidget(clr)
        self._cmp_hint = QLabel(t("  Left-click = add point /"
                                  " right-click finishes Polyline / Polygon"))
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
        btn.setText(t("Show All Result") if self._results_hidden
                    else t("Hide All Result"))

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
        self._refresh_tool_availability()   # grey/restore the interaction tools
        for k in ("A", "B"):
            self._overlay[k].update()

    def _set_measure_type(self, key):
        self._meas_type = key
        self._draft = None
        self._meas_hover = None
        if key != "point":
            self._measure_hover_clear()
        for k, b in self._meas_btns.items():
            b.setChecked(k == key)
            # Active = blue + WHITE text; colour-only override keeps size/shape.
            b.setStyleSheet(
                "background:#1f77b4;color:white;" if k == key else "")
        # A type is now active → left-click measures → grey the tools.
        self._refresh_tool_availability()

    def _measure_clear(self):
        self._measures = {"A": [], "B": []}
        self._draft = None
        self._edit = None
        self._meas_hover = None
        self._measure_hover_clear()
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
        vis_act = menu.addAction(t("Show") if target.get("hidden") else t("Hide"))
        del_act = menu.addAction(t("Delete"))
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
        if t == "point":
            return list(m["pts"][:1])
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

    def _hu_at(self, P):
        """Trilinear HU at a single 3-D world point *P*, or None outside the
        volume."""
        if P is None or self._vol is None:
            return None
        sx, sy, sz = self._dims
        vx, vy, vz = P[0] / sx, P[1] / sy, P[2] / sz
        nz, ny, nx = self._vol.shape
        if not (0 <= vx <= nx - 1 and 0 <= vy <= ny - 1 and 0 <= vz <= nz - 1):
            return None
        return float(_trilinear_sample(
            self._vol, np.array([vx]), np.array([vy]), np.array([vz]))[0])

    def _measure_hover(self, which, sx, sy):
        """Point-probe hover: show the HU under the cursor (painted next to it
        by the overlay). Repaints only the hovered pane."""
        if self._vol is None:
            return
        w = self._disp_to_world(which, sx, sy)
        try:
            hu = self._hu_at(self._out_to_world3d(which, *w))
        except Exception:
            hu = None
        self._probe_hover = (which, sx, sy, "—" if hu is None else f"HU {hu:.0f}")
        self._overlay[which].update()

    def _measure_hover_clear(self, which=None):
        if self._probe_hover is None:
            return
        k = self._probe_hover[0]
        self._probe_hover = None
        self._overlay[k].update()

    def _metrics_text(self, key, m):
        t = m["type"]
        pts = m["pts"]
        ca = m.get("center_angle")
        ca_str = (f"  CenterAngle:{ca['angle']:.1f}°"
                  if ca and "angle" in ca else "")
        if t == "point":
            p3 = (m.get("pts3d") or [None])[0]
            hu = self._hu_at(p3) if p3 is not None else None
            return (f"#{m['id']} Point: HU {hu:.0f}" if hu is not None
                    else f"#{m['id']} Point: HU —")
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
    def _reproject_traces(self, key):
        """A vessel-trace polyline stores ABSOLUTE 3-D control points (pts3d);
        re-derive its 2-D pane coords from them against the plane live NOW, so
        the trace stays glued to the anatomy when the plane is recentred /
        moved / rotated / paged instead of sitting still on screen. A plain 2-D
        measure has no pts3d and is untouched. 3-D MPR only."""
        if self._mode != "3D":
            return
        for m in self._measures[key]:
            p3 = m.get("pts3d")
            if p3 and m["type"] == "polyline" and len(p3) == len(m["pts"]):
                m["pts"] = [self._world3d_to_out(key, P) for P in p3]
        d = self._draft
        if (d is not None and d.get("pane") == key and d.get("pts3d")
                and len(d["pts3d"]) == len(d["pts"])):
            d["pts"] = [self._world3d_to_out(key, P) for P in d["pts3d"]]

    def _redraw_geom(self, key):
        self._reproject_traces(key)
        self._overlay[key].update()

    def _redraw_meas(self, key):
        self._reproject_traces(key)
        self._recompute_compares(key)      # keep comparisons in sync on edit/delete
        self._metrics[key] = [self._metrics_text(key, m)
                              for m in self._measures[key]
                              if m.get("_lv") is None]   # LV borders aren't results
        self._overlay[key].update()
        self._update_hideall_btn()

    # ---- picking ----
    def _pick_handle(self, which, sx, sy):
        # In LV mode, endo & epi points overlap near the apex/base; picking the
        # topmost blindly grabs (or deletes) the WRONG border and shadows the one
        # you meant. So when an Endo/Epi target is armed, the ARMED border's
        # point wins; a point that belongs to the OTHER LV border is ignored (so
        # you never grab/delete it by mistake); plain (non-LV) measures still
        # match as a fallback.
        lv_t = self._lv.get("target") if self._lv is not None else None
        fallback = None
        for mi in range(len(self._measures[which]) - 1, -1, -1):
            m = self._measures[which][mi]
            if m.get("hidden") or m.get("_lv_valve"):
                continue          # invisible / LOCKED valve ring → not grabbable
            for vi, q in enumerate(m["pts"]):
                qx, qy = self._world_to_screen(which, q[0], q[1])
                if math.hypot(qx - sx, qy - sy) < 12.0:
                    tag = m.get("_lv")
                    if lv_t not in ("endo", "epi"):
                        # No Endo/Epi armed → do NOT grab an LV border (endo & epi
                        # overlap in SAX, so an un-armed grab edits the WRONG one).
                        if tag is None:
                            return mi, vi
                        continue
                    if tag is not None and tag[1] == lv_t:
                        return mi, vi                  # armed border wins
                    if tag is None and fallback is None:
                        fallback = (mi, vi)            # non-LV measure fallback
        return fallback

    def _measure_hover_handle(self, which, sx, sy) -> None:
        """Highlight (green) the existing control point under the cursor so the
        user sees it will be grabbed BEFORE pressing — the hover twin of the
        green drag colour, so hover→grab→drag reads as one continuous state.
        Covers both ordinary vertices and Center-Angle marker points."""
        hit = self._pick_handle(which, sx, sy)
        if hit is not None:
            new = {"key": which, "mi": hit[0], "vi": hit[1], "ca": False}
        else:
            ca = self._pick_center_angle(which, sx, sy)
            new = ({"key": which, "mi": ca[0], "vi": ca[1], "ca": True}
                   if ca is not None else None)
        # With no handle/vertex under the cursor, hovering a movable shape's
        # OUTLINE turns the whole outline green (it will be grabbed to MOVE).
        out_new = None
        if new is None:
            om = self._pick_measure(which, sx, sy, tol=8.0)
            if om is not None:
                m = self._measures[which][om]
                if (not m.get("_lv") and not m.get("hidden")
                        and m.get("type") in ("line", "ellipse", "polygon")):
                    out_new = (which, om)
        if (new == self._meas_hover_handle
                and out_new == self._meas_hover_outline):
            return
        old = self._meas_hover_handle
        old_out = self._meas_hover_outline
        self._meas_hover_handle = new
        self._meas_hover_outline = out_new
        keys = {which}
        if old:
            keys.add(old["key"])
        if old_out:
            keys.add(old_out[0])
        for k in keys:
            self._overlay[k].update()

    def _clear_hover_handle(self) -> None:
        if (self._meas_hover_handle is None
                and self._meas_hover_outline is None):
            return
        keys = set()
        if self._meas_hover_handle:
            keys.add(self._meas_hover_handle["key"])
        if self._meas_hover_outline:
            keys.add(self._meas_hover_outline[0])
        self._meas_hover_handle = None
        self._meas_hover_outline = None
        for k in keys:
            self._overlay[k].update()

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

    def _pick_measure(self, which, sx, sy, tol=5.0):
        # Pixel-based catch (constant on-screen width, zoom/DPR independent), so
        # the boundary band can't balloon when zoomed in and shadow the filled
        # compare region. tol=5 for right-click routing; ~8 for hover / move-grab.
        # STRONGLY honour the armed Endo/Epi selection: when a target is armed,
        # ONLY that border is a candidate — so right-click "Add point" (and any
        # outline pick) lands on the SELECTED border, never the nearer other one.
        lv_t = self._lv.get("target") if self._lv is not None else None
        best, bi = tol, None
        for mi, m in enumerate(self._measures[which]):
            if m.get("hidden") or m.get("_lv_valve"):   # locked valve ring
                continue
            if lv_t in ("endo", "epi"):
                tag = m.get("_lv")
                if tag is None or tag[1] != lv_t:
                    continue
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
        # LV: while ACTIVELY drawing an endo/epi border, a click adds the next
        # point (never grabs a nearby other border's handle).
        lv_target = self._lv.get("target") if self._lv is not None else None
        lv_drawing = (lv_target in ("endo", "epi")
                      and self._draft is not None
                      and self._draft.get("pane") == which
                      and len(self._draft.get("pts", [])) >= 1)
        hit = None if lv_drawing else self._pick_handle(which, sx, sy)
        # Starting a NEW LV border must not grab a DIFFERENT border's point.
        if (hit is not None and lv_target in ("endo", "epi")
                and self._draft is None):
            tag = self._measures[which][hit[0]].get("_lv")
            if tag is None or tag[1] != lv_target:
                hit = None
        if hit is not None:
            self._edit = {"key": which, "mi": hit[0], "vi": hit[1]}
            is_lv_border = (self._lv is not None
                            and self._measures[which][hit[0]].get("_lv"))
            if is_lv_border:
                self._lv_push_undo(which, hit[0])   # snapshot before the drag
            else:
                self._meas_edit_before = (which, self._meas_pane_snap(which))
                self._meas_edit_moved = False
            self._redraw_geom(which)
            # Grabbing an LV border vertex → light the linked SAX crossing green
            # at once (not only after the first drag).
            if self._lv_sax_active() and self._measures[which][hit[0]].get("_lv"):
                self._overlay[self._lv["sax_pane"]].update()
            return True
        # A Center-Angle marker point can be dragged just like a polygon vertex.
        ca_hit = None if lv_drawing else self._pick_center_angle(which, sx, sy)
        if ca_hit is not None:
            self._edit = {"key": which, "mi": ca_hit[0], "vi": ca_hit[1],
                          "ca": True}
            self._meas_edit_before = (which, self._meas_pane_snap(which))
            self._meas_edit_moved = False
            self._redraw_geom(which)
            return True
        # No handle/vertex hit: pressing on a shape's OUTLINE grabs the WHOLE
        # shape to MOVE it (Line/Ellipse/Polygon, shape preserved). Non-LV only,
        # and NOT while a trace is in progress (draw a fresh shape a few px off).
        if not lv_drawing and self._draft is None:
            om = self._pick_measure(which, sx, sy, tol=8.0)
            if om is not None:
                m = self._measures[which][om]
                if (not m.get("_lv") and not m.get("hidden")
                        and m.get("type") in ("line", "ellipse", "polygon")):
                    w0 = self._disp_to_world(which, sx, sy)
                    ca0 = m.get("center_angle")
                    self._edit = {
                        "key": which, "mi": om, "vi": None, "move": True,
                        "anchor": w0,
                        "orig": [tuple(q) for q in m["pts"]],
                        "orig3d": ([list(map(float, P)) for P in m["pts3d"]]
                                   if m.get("pts3d") else None),
                        "orig_ca": ([tuple(q) for q in ca0["pts"]]
                                    if ca0 and ca0.get("pts") else None)}
                    self._meas_edit_before = (which, self._meas_pane_snap(which))
                    self._meas_edit_moved = False
                    self._redraw_geom(which)
                    return True
        if not self._meas_type:
            return False
        w = self._disp_to_world(which, sx, sy)
        # Point HU probe: each click drops a persistent "+" and lists its HU in
        # the top-right result block (a single click finishes it; no draft).
        if self._meas_type == "point":
            try:
                P = self._out_to_world3d(which, *w)
            except Exception:
                P = None
            _pt_before = self._meas_pane_snap(which)
            self._meas_seq += 1
            self._measures[which].append(
                {"id": self._meas_seq, "type": "point", "pts": [w],
                 "pts3d": [P] if P is not None else []})
            self._redraw_meas(which)
            self._meas_record(which, _pt_before)
            return False
        # Line: support press-drag-release (the natural gesture for a straight
        # line) ON TOP of click-click. The first press starts a rubber-band line
        # whose 2nd point tracks the drag; a release that actually moved commits
        # it, one that didn't falls back to two-click (see _measure_drag /
        # _measure_release). Nothing here is LV-gated, so Line works in LV Vol
        # (and everywhere) by either gesture.
        if self._meas_type == "line" and (
                self._draft is None or self._draft.get("pane") != which
                or self._draft.get("type") != "line"):
            self._draft = {"type": "line", "pane": which, "pts": [w, w],
                           "_drag_new": True}
            self._draft_redo = []
            self._redraw_geom(which)
            return True                    # caller keeps the drag (_meas_drag)
        # LV: a NEW polyline may start ONLY with Endo/Epi active and no captured
        # border for this plane yet; blocked in SAX (confirm/edit only).
        if (self._lv is not None and self._lv.get("phase") == "contour"
                and self._meas_type == "polyline" and self._draft is None
                and (self._lv.get("sax") is not None
                     or lv_target not in ("endo", "epi")
                     or self._lv_has_border(which, lv_target))):
            return False
        d = self._draft
        if d is None or d["pane"] != which or d["type"] != self._meas_type:
            d = {"type": self._meas_type, "pane": which, "pts": []}
            self._draft = d
        d["pts"].append(w)
        self._draft_redo = []          # a new point forks the trace's redo history
        # Capture the ABSOLUTE 3-D position of each click on the plane active
        # when it was made (world mm), so a polyline traced across rotated /
        # paged planes lifts to a correct 3-D centreline (Short-axis / CPR) —
        # mirrors the VTK viewer. Only meaningful for polylines.
        if d["type"] == "polyline":
            P = self._out_to_world3d(which, *w)
            # Optionally snap the DEPTH to the contrast lumen along the plane
            # normal (the slab MIP hid it) — same as the VTK viewer. NOT for LV
            # borders: endo/epi aren't the vessel lumen, so snapping would push
            # the traced point off the plane (a hollow depth-cue dot) and break
            # the WYSIWYG capture.
            if self._snap_lumen and self._mode == "3D" and self._lv is None:
                _, _, nrm = self._axes_for(which)
                P = self._snap_to_lumen(P, nrm)
                d["pts"][-1] = self._world3d_to_out(which, P)  # keep 2-D in step
            # LV: if this click is within the apex convergence range, snap it
            # EXACTLY onto the apex NOW (overlaps the marker immediately, not
            # only after the finishing double-click).
            Pa = self._lv_active_apex()
            if Pa is not None:
                ax0, ay0 = self._world3d_to_out(which, Pa)
                if (math.hypot(w[0] - ax0, w[1] - ay0)
                        <= self._lv_apex_range_mm(which)):
                    P = np.asarray(Pa, float)
                    d["pts"][-1] = (ax0, ay0)
            d.setdefault("pts3d", []).append(P)
        if d["type"] in ("line", "ellipse") and len(d["pts"]) >= 2:
            self._commit_draft()
        elif d["type"] == "angle" and len(d["pts"]) >= 3:
            self._commit_draft()
        else:
            self._redraw_geom(which)
            self._lv_apex_clear_glow()         # a point was confirmed → normal
        return False

    def _measure_drag(self, which, sx, sy):
        e = self._edit
        if not e:
            # Rubber-band the 2nd point of a Line being press-dragged.
            d = self._draft
            if (d is not None and d.get("_drag_new")
                    and d.get("type") == "line" and d.get("pane") == which):
                w = self._disp_to_world(which, sx, sy)
                if getattr(self, "_meas_ortho", False):   # Shift → 縦横直線 preview
                    w = self._ortho_snap(d["pts"][0], w)
                d["pts"][1] = w
                self._redraw_geom(which)
            return
        m = self._measures[e["key"]][e["mi"]]
        if e.get("move"):
            # Translate the WHOLE shape by the drag delta (shape preserved).
            w = self._disp_to_world(e["key"], sx, sy)
            dx = w[0] - e["anchor"][0]
            dy = w[1] - e["anchor"][1]
            m["pts"] = [(q[0] + dx, q[1] + dy) for q in e["orig"]]
            if e.get("orig3d"):
                P0 = np.asarray(self._out_to_world3d(e["key"], *e["anchor"]),
                                float)
                P1 = np.asarray(self._out_to_world3d(e["key"], *w), float)
                dP = P1 - P0
                m["pts3d"] = [list(np.asarray(P, float) + dP)
                              for P in e["orig3d"]]
            if e.get("orig_ca") and m.get("center_angle"):
                m["center_angle"]["pts"] = [(q[0] + dx, q[1] + dy)
                                            for q in e["orig_ca"]]
                self._resnap_center_angle(m)
            self._meas_edit_moved = True
            self._redraw_geom(e["key"])
            return
        w = self._disp_to_world(e["key"], sx, sy)
        if self._meas_edit_before is not None:
            self._meas_edit_moved = True     # a non-LV handle actually moved
        if e.get("ca"):
            self._set_center_angle_point(m, e["vi"], w)
        elif m["type"] == "ellipse":
            self._set_ellipse_handle(m, e["vi"], w)
            self._resnap_center_angle(m)
        else:
            m["pts"][e["vi"]] = w
            # Keep the absolute 3-D trace in step: the dragged vertex moves on
            # the CURRENT plane, so its new 3-D is that 2-D point lifted here
            # (then snapped to the lumen along the normal, if enabled).
            if m.get("pts3d") and 0 <= e["vi"] < len(m["pts3d"]):
                P = self._out_to_world3d(e["key"], *w)
                if self._snap_lumen and self._mode == "3D" \
                        and self._lv is None:
                    _, _, nrm = self._axes_for(e["key"])
                    P = self._snap_to_lumen(P, nrm)
                    m["pts"][e["vi"]] = self._world3d_to_out(e["key"], P)
                m["pts3d"][e["vi"]] = P
            self._resnap_center_angle(m)
        self._recompute_compares(e["key"])     # live-update any comparison
        self._redraw_geom(e["key"])
        self._lv_live_recapture(e["key"], m)   # edited LV border → refresh SAX

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
            mi = self._edit.get("mi")
            self._edit = None
            self._redraw_meas(key)
            if self._lv_sax_active():        # revert the green SAX crossing
                self._overlay[self._lv["sax_pane"]].update()
            # Commit an LV-border DRAG (snapshot stashed at press) as one step.
            if getattr(self, "_lv_edit_before", None) is not None \
                    and mi is not None:
                self._lv_record_border(self._lv_edit_before, key, mi)
            self._lv_edit_before = None
            # Commit a general-Measure handle/move DRAG as one undo step (only if
            # the shape/point actually moved).
            if self._meas_edit_before is not None:
                bpane, bsnap = self._meas_edit_before
                self._meas_edit_before = None
                if self._meas_edit_moved:
                    self._meas_record(bpane, bsnap)
                self._meas_edit_moved = False
            return
        # Line press-drag-release: a real drag commits the 2-point line; a press
        # with (almost) no movement reverts to a 1-point draft so the classic
        # click-click still finishes it on the next click.
        d = self._draft
        if d is not None and d.get("_drag_new") and d.get("type") == "line":
            p0, p1 = d["pts"][0], d["pts"][1]
            d.pop("_drag_new", None)
            if abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) > 1e-6:
                self._commit_draft()
            else:
                d["pts"] = [p0]
                self._redraw_geom(d["pane"])
        self._lv_edit_before = None

    @staticmethod
    def _ortho_snap(p0, p1):
        """Snap *p1* to a horizontal or vertical line through *p0* (whichever the
        drag is closer to) — the 縦横直線 constraint when Shift is held."""
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        if abs(dx) >= abs(dy):
            return (p1[0], p0[1])       # horizontal
        return (p0[0], p1[1])           # vertical

    def _set_ellipse_handle(self, m, vi, w):
        m["pts"] = _ellipse_drag(m["pts"], vi, w, circle=bool(m.get("circle")))

    def _make_circle(self, which, mi) -> None:
        """Turn an existing Ellipse into a TRUE CIRCLE (正円化): minor axis = major
        (same centre + major direction). Recorded as one undo step; pts3d re-
        derived if 3-D anchored."""
        if not (0 <= mi < len(self._measures.get(which, []))):
            return
        m = self._measures[which][mi]
        if m.get("type") != "ellipse" or len(m.get("pts", [])) < 2:
            return
        before = None if m.get("_lv") else self._meas_pane_snap(which)
        m["pts"] = _ellipse_from_major(m["pts"][0], m["pts"][1], minor_ratio=1.0)
        m["circle"] = True
        if m.get("pts3d") and len(m["pts3d"]) == 4:
            m["pts3d"] = [self._out_to_world3d(which, *q) for q in m["pts"]]
        if before is not None:
            self._meas_record(which, before)
        self._redraw_meas(which)

    def _commit_draft(self):
        d = self._draft
        self._draft = None
        self._meas_hover = None
        if d is None or len(d["pts"]) < 2:
            return
        # Snapshot for a general-Measure create undo (skipped for LV borders,
        # which keep their own _lv_record_create).
        lv_border = (self._lv is not None and self._lv.get("phase") == "contour")
        meas_before = None if lv_border else self._meas_pane_snap(d["pane"])
        if d["type"] == "ellipse":
            # The two clicked points are the MAJOR-axis endpoints; the minor
            # axis starts at half the major (or EQUAL to it → 正円 when Shift is
            # held) and is then dragged to taste.
            circ = bool(getattr(self, "_meas_circle", False))
            pts = _ellipse_from_major(d["pts"][0], d["pts"][1],
                                      minor_ratio=1.0 if circ else 0.5)
        elif d["type"] == "line":
            pts = d["pts"][:2]
            # Shift held → constrain the line to horizontal/vertical (縦横直線).
            if getattr(self, "_meas_ortho", False) and len(pts) == 2:
                pts = [pts[0], self._ortho_snap(pts[0], pts[1])]
        else:
            pts = list(d["pts"])
        # A resumed trace keeps its ORIGINAL id (and colour / spline / alpha) so
        # it reads as the same result continued, not a brand-new one; a fresh
        # draft takes the next sequence number.
        if d.get("resume_id") is not None:
            rid = int(d["resume_id"])
            self._meas_seq = max(self._meas_seq, rid)
        else:
            self._meas_seq += 1
            rid = self._meas_seq
        m = {"id": rid, "type": d["type"], "pts": pts}
        for k in ("color", "smooth", "transp"):
            if d.get(k) is not None:
                m[k] = d[k]
        # A Shift-drawn ellipse is a TRUE CIRCLE; remember it so handle edits
        # keep it circular (see _set_ellipse_handle).
        if d["type"] == "ellipse" and getattr(self, "_meas_circle", False):
            m["circle"] = True
        # Carry the per-click 3-D control points (polyline only) so the trace
        # can seed a short-axis centreline and its control markers.
        if d["type"] == "polyline" and d.get("pts3d") \
                and len(d["pts3d"]) == len(pts):
            m["pts3d"] = [np.asarray(P, float) for P in d["pts3d"]]
        self._measures[d["pane"]].append(m)
        self._redraw_meas(d["pane"])
        if meas_before is not None:                  # general-Measure create undo
            self._meas_record(d["pane"], meas_before)
        # Log to the study's measurement history (shell-owned).
        meas = Measurement(kind=d["type"].capitalize(), points=list(pts),
                           spacing_mm=None, mid=rid)
        meas.text = self._metrics_text(d["pane"], m)
        self.measurement_added.emit(meas)

    def _measure_finish_draft(self):
        d = self._draft
        if d and d["type"] in ("polyline", "polygon") and len(d["pts"]) >= 2:
            self._commit_draft()
            # LV EF: capture the finished border to the current target (spline +
            # endo/epi colour) for BOTH double-click AND right-click finishes, and
            # force the full re-render so it appears at once (right-click parity).
            if self._lv is not None and self._lv.get("phase") == "contour":
                self._lv_on_border_committed()
                if not self._lv_sax_active():
                    self._lv_show_plane()

    def _resume_trace(self, which, mi, endpoint_vi):
        """Un-commit polyline *mi* back into the in-progress draft so the user
        can keep clicking to EXTEND it from the *endpoint_vi* end (0 = start,
        last = end). New clicks always append to the draft, so a start-resume
        reverses the vertex order first (the geometry is identical; CPR handles
        direction). The stale Measure-History entry is dropped here and re-added
        when the trace is committed again, keeping the same result id."""
        if not (0 <= mi < len(self._measures[which])):
            return
        m = self._measures[which][mi]
        if m["type"] != "polyline" or len(m["pts"]) < 2:
            return
        # Arm the Polyline tool so the next clicks EXTEND this trace. Sync the
        # toolbar buttons WITHOUT calling _set_measure_type, which would wipe
        # the draft we're about to install.
        self._meas_type = "polyline"
        self._meas_hover = None
        for k, b in self._meas_btns.items():
            b.setChecked(k == "polyline")
            b.setStyleSheet(
                "background:#1f77b4;color:white;" if k == "polyline" else "")
        self._refresh_tool_availability()
        # Prefer the absolute 3-D control points (re-projected onto the CURRENT
        # plane) as the source of truth; fall back to the stored 2-D vertices.
        p3 = m.get("pts3d")
        if p3 is not None and len(p3) == len(m["pts"]) and self._mode == "3D":
            pts3d = [np.asarray(P, float) for P in p3]
            pts = [self._world3d_to_out(which, P) for P in pts3d]
        else:
            pts3d = None
            pts = [tuple(q) for q in m["pts"]]
        if endpoint_vi == 0:                     # resume from the START end
            pts = list(reversed(pts))
            if pts3d is not None:
                pts3d = list(reversed(pts3d))
        d = {"type": "polyline", "pane": which, "pts": pts,
             "resume_id": m["id"]}
        if pts3d is not None:
            d["pts3d"] = pts3d
        for k in ("color", "smooth", "transp"):  # keep the trace's appearance
            if k in m:
                d[k] = m[k]
        # Drop the committed result (and its history entry); it lives on as the
        # draft now and re-commits under the same id.
        self.measurement_removed.emit(int(m["id"]))
        del self._measures[which][mi]
        self._draft = d
        self._redraw_meas(which)

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
        # The three context menus below are MODAL (menu.exec). They must NOT
        # run inline here — this is reached from _on_down (the pointer-down
        # handler), and a modal loop entered before the pointer-up swallows
        # that up, leaving the canvas holding the Qt mouse grab so every
        # toolbar button goes dead (the Mac dead-buttons bug; see
        # _open_angio_dialog). Detect the hit synchronously (so the bool
        # return is correct for the caller's fall-through), then DEFER the
        # menu out of the handler via QTimer.singleShot(0) after clearing
        # pointer state. Mirrors the compare-delete menu.
        #
        # Right-click ON a Center-Angle marker or spoke deletes JUST the Center
        # Angle (the polygon/ellipse stays). Checked before the vertex/outline
        # menus since the markers sit on the outline.
        ca_mi = self._ca_hit(which, sx, sy)
        if ca_mi is not None:
            self._reset_pointer_state()
            QTimer.singleShot(
                0, lambda: self._center_angle_delete_menu(which, ca_mi, sx, sy))
            return True
        hit = self._pick_handle(which, sx, sy)
        if hit is not None:
            self._reset_pointer_state()
            QTimer.singleShot(0, lambda: self._handle_right(which, hit, sx, sy))
            return True
        mi = self._pick_measure(which, sx, sy)
        if mi is None:
            return False
        self._reset_pointer_state()
        QTimer.singleShot(0, lambda: self._outline_right(which, mi, sx, sy))
        return True

    def _center_angle_delete_menu(self, which, ca_mi, sx, sy):
        """Deferred 'Delete Center Angle' menu (see _measure_right for why it
        must run outside the pointer-down handler)."""
        if not (0 <= ca_mi < len(self._measures[which])):
            return
        menu = QMenu(self)
        del_ca = menu.addAction(t("Delete Center Angle"))
        try:
            chosen = menu.exec(self.pane[which].canvas.mapToGlobal(
                QPoint(int(sx), int(sy))))
        finally:
            self._reset_pointer_state()
        if chosen is del_ca:
            self._measures[which][ca_mi].pop("center_angle", None)
            self._redraw_meas(which)

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
        del_pt = del_res = resume_act = None
        if m["type"] in ("polyline", "polygon"):
            del_pt = menu.addAction(t("Delete point"))
            if len(m["pts"]) <= 2:
                del_pt.setEnabled(False)
        # Right-click on a polyline END vertex (the 断端) → "Resume trace":
        # un-commit it so the user can keep clicking points to EXTEND that end
        # (Add point only inserts BETWEEN existing vertices, never past an end).
        if m["type"] == "polyline" and vi in (0, len(m["pts"]) - 1):
            resume_act = menu.addAction(t("Resume trace"))
        # Change Color / Change Transparency — on every result type (incl.
        # Line/Angle, most easily right-clicked on a handle).
        from multi_dicomviewer.viewers.image_canvas import COLOR_CHOICES
        color_actions = add_color_submenu(menu, COLOR_CHOICES)
        transp_actions = add_transparency_submenu(menu, m.get("transp", 0))
        circle_act = (menu.addAction(t("Make Circle"))
                      if m["type"] == "ellipse" else None)
        hide_act = menu.addAction(t("Show") if m.get("hidden") else t("Hide"))
        if m["type"] in ("polyline", "polygon"):
            del_res = menu.addAction(t("Delete result"))
        else:
            del_res = menu.addAction(t("Delete"))
        try:
            chosen = menu.exec(self.pane[which].canvas.mapToGlobal(
                QPoint(int(sx), int(sy))))
        finally:
            self._reset_pointer_state()   # never leave a stuck grab (Mac dead-buttons)
        if circle_act is not None and chosen is circle_act:
            self._make_circle(which, mi)         # records its own undo + redraw
            return
        _mr_before = None if m.get("_lv") else self._meas_pane_snap(which)
        _mr_changed = False
        if del_pt is not None and chosen is del_pt:
            self._delete_point(which, mi, vi)
            _mr_changed = True
        elif resume_act is not None and chosen is resume_act:
            self._resume_trace(which, mi, vi)
            return                               # _resume_trace redraws itself
        elif chosen is hide_act:
            m["hidden"] = not m.get("hidden", False)   # hide THIS line only
            _mr_changed = True
        elif chosen is del_res:
            if m.get("id") is not None:
                self.measurement_removed.emit(int(m["id"]))
            del self._measures[which][mi]
            _mr_changed = True
        else:
            for act, hexcol in color_actions:
                if chosen is act:
                    m["color"] = hexcol
                    _mr_changed = True
                    break
            for act, val in transp_actions:
                if chosen is act:
                    m["transp"] = val
                    _mr_changed = True
                    break
        if _mr_before is not None and _mr_changed:
            self._meas_record(which, _mr_before)
        self._recompute_compares(which)     # a colour change refreshes any compare
        self._redraw_meas(which)

    def _outline_right(self, which, mi, sx, sy):
        from PyQt6.QtGui import QIcon, QPixmap
        from multi_dicomviewer.viewers.image_canvas import COLOR_CHOICES
        m = self._measures[which][mi]
        menu = QMenu(self)
        add_pt = menu.addAction(t("Add point"))
        spline_act = None
        cpr_act = None
        if m["type"] == "polyline":
            spline_act = menu.addAction(
                t("UnSpline") if m.get("smooth") else t("Spline"))
            # Curved-MPR: build a centreline from this trace and scroll the
            # short-axis cross-sections in pane A. 3-D MPR only, >=2 points.
            if self._mode == "3D" and len(m["pts"]) >= 2:
                cpr_act = menu.addAction(t("Short-axis MPR (CPR)"))
        # Lumen-snap controls (3-D traces only): re-snap this trace now, and a
        # checkable auto-snap toggle for future clicks.
        snap_now_act = snap_auto_act = None
        if m["type"] == "polyline" and self._mode == "3D" and m.get("pts3d"):
            snap_now_act = menu.addAction(t("Snap trace to lumen"))
            snap_auto_act = menu.addAction(t("Auto-snap to lumen"))
            snap_auto_act.setCheckable(True)
            snap_auto_act.setChecked(self._snap_lumen)
        center_angle_act = None
        if m["type"] in ("ellipse", "polygon"):
            center_angle_act = menu.addAction(t("Center Angle"))
        circle_act = (menu.addAction(t("Make Circle"))
                      if m["type"] == "ellipse" else None)
        color_menu = menu.addMenu(t("Change Color"))
        color_actions = []
        for name, hexcol in COLOR_CHOICES:
            a = color_menu.addAction(name)
            pix = QPixmap(16, 16)
            pix.fill(QColor(hexcol))
            a.setIcon(QIcon(pix))
            color_actions.append((a, hexcol))
        transp_actions = add_transparency_submenu(menu, m.get("transp", 0))
        hide_act = menu.addAction(t("Show") if m.get("hidden") else t("Hide"))
        del_act = menu.addAction(t("Delete"))
        try:
            chosen = menu.exec(self.pane[which].canvas.mapToGlobal(
                QPoint(int(sx), int(sy))))
        finally:
            self._reset_pointer_state()   # never leave a stuck grab (Mac dead-buttons)
        if circle_act is not None and chosen is circle_act:
            self._make_circle(which, mi)         # records its own undo + redraw
            return
        _mr_before = None if m.get("_lv") else self._meas_pane_snap(which)
        _mr_changed = False
        if chosen is add_pt:
            self._add_point(which, mi, sx, sy)
            _mr_changed = True
        elif cpr_act is not None and chosen is cpr_act:
            self._enter_cpr(which, mi)
            return                               # _enter_cpr redraws itself
        elif snap_now_act is not None and chosen is snap_now_act:
            self._snap_trace(which, mi)
            _mr_changed = True
        elif snap_auto_act is not None and chosen is snap_auto_act:
            self._snap_lumen = snap_auto_act.isChecked()
        elif spline_act is not None and chosen is spline_act:
            m["smooth"] = not m.get("smooth", False)
            _mr_changed = True
        elif center_angle_act is not None and chosen is center_angle_act:
            self._center_angle_target = {"key": which, "mi": mi}
            m.pop("center_angle", None)
        elif chosen is hide_act:
            m["hidden"] = not m.get("hidden", False)   # hide THIS line only
            _mr_changed = True
        elif chosen is del_act:
            if m.get("id") is not None:
                self.measurement_removed.emit(int(m["id"]))
            del self._measures[which][mi]
            _mr_changed = True
        else:
            for act, hexcol in color_actions:
                if chosen is act:
                    m["color"] = hexcol
                    _mr_changed = True
                    break
            for act, val in transp_actions:
                if chosen is act:
                    m["transp"] = val
                    _mr_changed = True
                    break
        if _mr_before is not None and _mr_changed:
            self._meas_record(which, _mr_before)
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
        _lv_before = (self._lv_border_snap(which, mi)
                      if self._lv is not None and m.get("_lv") else None)
        wx, wy = self._disp_to_world(which, sx, sy)
        pt = (wx, wy)
        # 3-D lift of the new point on the CURRENT plane, so a trace that carries
        # per-vertex 3-D (LV borders / CPR centrelines) stays consistent — an
        # inserted point with no 3-D desynced pts/pts3d and dropped later points
        # onto the wrong cross-section.
        p3d = None
        try:
            p3d = self._out_to_world3d(which, wx, wy)
        except Exception:
            p3d = None
        if m["type"] == "ellipse":
            e1, e2, m1, m2 = m["pts"]            # major ends, minor ends
            m["type"] = "polygon"
            m["pts"] = [e1, m1, e2, m2]          # around the ellipse
        if m["type"] == "line":
            m["type"] = "polyline"
            m["pts"] = [m["pts"][0], pt, m["pts"][1]]
            if m.get("pts3d") and len(m["pts3d"]) == 2:
                m["pts3d"] = [m["pts3d"][0], p3d, m["pts3d"][1]]
            self._lv_after_point_edit(which, m)
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
        if m.get("pts3d") and len(m["pts3d"]) == n:
            m["pts3d"].insert(best_i + 1, p3d)
        self._resnap_center_angle(m)
        self._lv_after_point_edit(which, m)
        self._lv_record_border(_lv_before, which, mi)   # Ctrl+Z / Ctrl+Y

    def _delete_point(self, which, mi, vi):
        m = self._measures[which][mi]
        if m["type"] not in ("polyline", "polygon"):
            return
        pts = list(m["pts"])
        if len(pts) <= 2 or not (0 <= vi < len(pts)):
            return
        _lv_before = (self._lv_border_snap(which, mi)
                      if self._lv is not None and m.get("_lv") else None)
        del pts[vi]
        if m.get("pts3d") and 0 <= vi < len(m["pts3d"]):
            del m["pts3d"][vi]
        if len(pts) == 2:
            m["type"] = "line"
        m["pts"] = pts
        self._resnap_center_angle(m)
        self._lv_after_point_edit(which, m)
        self._lv_record_border(_lv_before, which, mi)   # Ctrl+Z / Ctrl+Y

    def _lv_after_point_edit(self, which, m) -> None:
        """After Add/Delete point on an LV border, push the reshaped trace back
        into the model and refresh the short-axis so add/delete behave like a
        vertex drag (which already re-captures live)."""
        if self._lv is not None and m.get("_lv") is not None:
            self._lv_live_recapture(which, m)

    def _lv_cross_suppressed(self) -> bool:
        """The crosshair is hidden + non-interactive once the active LV pass's
        APEX is placed (tracing has begun) — from there the plane is driven only
        by the SAX handles / ◀▶ and the crosshair would only corrupt a trace.
        Active through align/ready and up to the first apex click. Matches VTK."""
        if getattr(self, "_lv_view_free", False):     # Epi領域表示 inspect
            return False
        return (getattr(self, "_lv", None) is not None
                and self._lv_active_apex() is not None)

    def _style_cl(self):
        if not self._cl_btn.isEnabled():          # suppressed / 2-D → greyed out
            self._cl_btn.setStyleSheet(
                "background:#e6e6e6;color:#a8a8a8;border:1px solid #d8d8d8;")
        else:
            self._cl_btn.setStyleSheet("")        # default checkable look

    def _toggle_centerline(self):
        self._cl_on = self._cl_btn.isChecked()
        for k in ("A", "B"):
            self._overlay[k].update()

    # ==================================================================
    # LV EF (ported from the VTK viewer). Logic reuses the shared measure/
    # frame/model infrastructure; LV drawing is done by _Overlay._paint_lv
    # (QPainter) instead of VTK actors, so here "redraw" = overlay.update().
    # ==================================================================
    def _lv_redraw_all(self) -> None:
        for k in ("A", "B"):
            self._overlay[k].update()

    # ===== Unified LV mode: common valves + sub-mode selector (VTK parity) =====
    def _lv_current_submode(self):
        """Which sub-mode is active: 'blood', 'endo', 'epi', or None."""
        if self._lvv is not None:
            return "blood"
        if self._lv is not None:
            return self._lv.get("pass")
        return None

    def _lv_valves_ready(self) -> bool:
        return (self._lv_valves.get("mitral") is not None
                and self._lv_valves.get("aortic") is not None)

    def _lv_capture_valve_common(self, which) -> None:
        """Set the COMMON MV/AoV plane from the latest fresh Ellipse."""
        from PyQt6.QtWidgets import QMessageBox
        if self._vol is None:
            return
        m, key, best = None, None, -1
        for k in ("A", "B"):
            for cand in self._measures.get(k, []):
                if (cand.get("type") == "ellipse"
                        and cand.get("_lv_valve") is None
                        and cand.get("id", -1) > best):
                    best = cand.get("id", -1); m, key = cand, k
        vname = "AoV" if which == "aortic" else "MV"
        if m is None:
            if self._lv_valves.get(which) is not None:
                self._lv_toggle_valve_visibility(which)
                return
            box = QMessageBox(self.window())
            box.setWindowTitle(t("LV")); box.setIcon(QMessageBox.Icon.Information)
            box.setText(t("Set the {v} plane:").format(v=vname))
            b_load = box.addButton(t("Load"), QMessageBox.ButtonRole.AcceptRole)
            b_make = box.addButton(t("Create (draw Ellipse)"),
                                   QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is b_load:
                self._lv_load_valve(which)
            elif clicked is b_make:
                QMessageBox.information(
                    self.window(), t("LV"),
                    t("Draw an Ellipse on the {v} annulus (Measure→Ellipse), "
                      "then press {v} plane again.").format(v=vname))
            return
        self._lv_valve_shown[which] = True
        cx, cy = self._shape_center(m)
        center = np.asarray(self._out_to_world3d(key, cx, cy), float)
        _u, _v, n = self._axes_for(key)
        _ecx, _ecy, ea, eb = self._ellipse_cab(m)
        radius = float(max(ea, eb))
        self._lv_valves[which] = (center, np.asarray(n, float), radius)
        for k in ("A", "B"):
            self._measures[k] = [mm for mm in self._measures.get(k, [])
                                 if mm is not m and mm.get("_lv_valve") != which]
        self._lv_valve_show_from_geom(which)
        if self._meas_on:
            self._meas_btn.setChecked(False)
            self._toggle_measure()
        self._lv_update_valve_buttons()
        self._lv_update_submode_ui()

    def _lv_valve_show_from_geom(self, which) -> None:
        v = self._lv_valves.get(which)
        if v is None or self._vol is None:
            return
        c, n, r = v
        c = np.asarray(c, float); n = np.asarray(n, float)
        n = n / (np.linalg.norm(n) or 1.0)
        ref = (np.array([1.0, 0.0, 0.0]) if abs(float(n[0])) < 0.9
               else np.array([0.0, 1.0, 0.0]))
        u = np.cross(n, ref); u = u / (np.linalg.norm(u) or 1.0)
        w = np.cross(n, u)
        ths = np.linspace(0.0, 2.0 * np.pi, 33)[:-1]
        pts3d = [tuple(map(float, c + r * np.cos(t) * u + r * np.sin(t) * w))
                 for t in ths]
        color = "#ffd24d" if which == "aortic" else "#4dd0ff"
        hidden = not self._lv_valve_shown.get(which, True)
        for k in ("A", "B"):
            self._measures[k] = [mm for mm in self._measures.get(k, [])
                                 if mm.get("_lv_valve") != which]
            self._meas_seq += 1
            self._measures[k].append({
                "id": self._meas_seq, "type": "polygon",
                "pts": [self._world3d_to_out(k, P) for P in pts3d],
                "pts3d": list(pts3d), "color": color, "_lv_valve": which,
                "transp": 50, "hidden": hidden})
            self._redraw_meas(k)

    def _lv_toggle_valve_visibility(self, which) -> None:
        from PyQt6.QtWidgets import QMessageBox
        shown = not self._lv_valve_shown.get(which, True)
        if (which == "mitral" and shown
                and self._lv_valves.get("mitral") is not None):
            box = QMessageBox(self.window())
            box.setWindowTitle(t("MV plane")); box.setIcon(QMessageBox.Icon.Question)
            box.setText(t("Show the MV plane:"))
            b_asis = box.addButton(t("As-is"), QMessageBox.ButtonRole.AcceptRole)
            b_perp = box.addButton(t("MV-perpendicular view"),
                                   QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is None or clicked not in (b_asis, b_perp):
                return
            self._lv_valve_shown["mitral"] = True
            for k in ("A", "B"):
                for mm in self._measures.get(k, []):
                    if mm.get("_lv_valve") == "mitral":
                        mm["hidden"] = False
                self._redraw_meas(k)
            self._lv_update_valve_buttons()
            if clicked is b_perp:
                self._lv_view_mv_perpendicular()
            return
        self._lv_valve_shown[which] = shown
        for k in ("A", "B"):
            for mm in self._measures.get(k, []):
                if mm.get("_lv_valve") == which:
                    mm["hidden"] = not shown
            self._redraw_meas(k)
        self._lv_update_valve_buttons()

    def _lv_view_mv_perpendicular(self) -> None:
        mv = self._lv_valves.get("mitral")
        if mv is None or self._vol is None:
            return
        c = np.asarray(mv[0], float); n = np.asarray(mv[1], float)
        n = n / (np.linalg.norm(n) or 1.0)
        ref = (np.array([1.0, 0.0, 0.0]) if abs(float(n[0])) < 0.9
               else np.array([0.0, 1.0, 0.0]))
        a = np.cross(n, ref); a = a / (np.linalg.norm(a) or 1.0)
        b = np.cross(n, a)
        self._frame["A"] = self._ortho(a, b)
        self._frame["B"] = self._ortho(a, -n)
        self._pc["A"] = c.copy(); self._pc["B"] = c.copy()
        self._center = c.copy()
        self._cross_ang = {"A": 0.0, "B": 0.0}
        self.set_side("Bi")
        self._view_initial = True
        self._refresh()
        for k in ("A", "B"):
            self._overlay[k].update()

    def _lv_update_valve_buttons(self) -> None:
        if getattr(self, "_lv_mv_btn", None) is None:
            return
        for which, btn, color in (("mitral", self._lv_mv_btn, "#2b6cb0"),
                                   ("aortic", self._lv_aov_btn, "#b8860b")):
            if self._lv_valves.get(which) is None:
                btn.setStyleSheet(self._BTN_DIS)
            elif self._lv_valve_shown.get(which, True):
                btn.setStyleSheet("QPushButton{background:%s;color:white;}%s"
                                  % (color, self._BTN_DIS))
            else:
                btn.setStyleSheet(
                    "QPushButton{background:palette(button);color:%s;"
                    "border:2px solid %s;}%s" % (color, color, self._BTN_DIS))

    def _lv_save_valve(self, which) -> None:
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import json, os
        v = self._lv_valves.get(which)
        if v is None:
            QMessageBox.information(
                self.window(), t("LV"), t("Set the {w} plane first.").format(
                    w="MV" if which == "mitral" else "AoV"))
            return
        c, n, r = v
        data = {"type": "valve", "valve": which,
                "series": (self._lv_series_meta()
                           if hasattr(self, "_lv_series_meta") else {}),
                "c": list(map(float, c)), "n": list(map(float, n)), "r": float(r)}
        suffix = ".MVLv.json" if which == "mitral" else ".AoVLv.json"
        d = self._lv_save_dir() if hasattr(self, "_lv_save_dir") else ""
        stem = (self._lv_default_stem() if hasattr(self, "_lv_default_stem")
                else "valve")
        default = os.path.join(d, stem + suffix) if d else stem + suffix
        flt = (("MV plane (*.MVLv.json)" if which == "mitral"
                else "AoV plane (*.AoVLv.json)") + ";;JSON (*.json)")
        path, _ = QFileDialog.getSaveFileName(
            self.window(), t("Save valve plane"), default, flt)
        if not path:
            return
        if not path.endswith(".json"):
            path += suffix
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as exc:                        # noqa: BLE001
            QMessageBox.warning(self.window(), t("LV"),
                                t("Save failed: {err}", err=str(exc)))
            return
        QMessageBox.information(self.window(), t("LV"),
                               t("Saved: {p}", p=os.path.basename(path)))

    def _lv_load_valve(self, which) -> None:
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import json
        if self._vol is None:
            return
        flt = (("MV plane (*.MVLv.json)" if which == "mitral"
                else "AoV plane (*.AoVLv.json)") + ";;JSON (*.json)")
        d = self._lv_save_dir() if hasattr(self, "_lv_save_dir") else ""
        path, _ = QFileDialog.getOpenFileName(
            self.window(), t("Load valve plane"), d, flt)
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._lv_valves[which] = (np.asarray(data["c"], float),
                                      np.asarray(data["n"], float),
                                      float(data.get("r", 20.0)))
            self._lv_valve_shown[which] = True
            self._lv_valve_show_from_geom(which)
            self._lv_update_valve_buttons()
            self._lv_update_submode_ui()
            QMessageBox.information(
                self.window(), t("LV"), t("Loaded the {w} plane.").format(
                    w="MV" if which == "mitral" else "AoV"))
        except Exception as exc:                        # noqa: BLE001
            import traceback
            QMessageBox.critical(self.window(), t("LV (valve load error)"),
                                 traceback.format_exc() or repr(exc))

    def _lv_stash_epi_for_blood(self, model) -> None:
        try:
            if (model is not None and model.epi_axis is not None
                    and len(model.epi_contours) >= 3):
                model.build()
                if model.epi is not None:
                    self._lvv_epi_surf = model.epi
                    self._lvv_epi_apex = np.asarray(model.epi_axis.apex, float)
                    d = model.to_dict()
                    # Carry the epicardial volume so Blood/Endo can show
                    # Epi-Volume (and Myocardium = Epi − Blood) with no recompute.
                    try:
                        spacing = max(0.5, float(min(self._dims)))
                        epi_ml = model.volume_ml(spacing, "epi")
                        if epi_ml is not None:
                            d.setdefault("volume", {})["epi_ml"] = float(epi_ml)
                            self._lvv_epi_ml = float(epi_ml)
                    except Exception:                   # noqa: BLE001
                        pass
                    self._lvv_epi_model_dict = d
        except Exception:                               # noqa: BLE001
            pass

    def _lvv_epi_volume_ml(self):
        """Epicardial volume (mL) for the Blood/Endo readout, from the stashed Epi
        model dict — cached, no recompute. None if unknown."""
        v = getattr(self, "_lvv_epi_ml", None)
        if v is not None:
            return v
        d = getattr(self, "_lvv_epi_model_dict", None)
        try:
            v = float(d["volume"]["epi_ml"]) if d else None
        except (KeyError, TypeError, ValueError):
            v = None
        self._lvv_epi_ml = v
        return v

    def _lvv_setup_axis_views(self) -> None:
        """On entering Blood/Endo, lay the panes out on the LV long axis: the
        RIGHT pane (B) shows the long-axis view (axis vertical) and the LEFT pane
        (A) the orthogonal SHORT-axis cut (⟂ the long axis, cardiology view),
        centred at the mid-ventricle. From the Epi axis (apex→MV centre)."""
        epi = getattr(self, "_lvv_epi_surf", None)
        ax = getattr(epi, "axis", None) if epi is not None else None
        if ax is None or self._vol is None:
            return
        apex = np.asarray(ax.apex, float)
        axis_dir = np.asarray(ax.axis, float)
        axis_dir = axis_dir / (float(np.linalg.norm(axis_dir)) or 1.0)
        length = float(getattr(ax, "length_mm", 0.0)) or 80.0
        along_mid = 0.5 * length
        mid = apex + along_mid * axis_dir
        # RIGHT (B): long-axis — output y = axis (apex→base) → vertical.
        _o, e_s, e_t, nrm = ax.long_axis_basis(0.0)
        self._frame["B"] = (np.asarray(e_s, float), np.asarray(e_t, float),
                            np.asarray(nrm, float))
        self._pc["B"] = mid.copy()
        self._cross_ang["B"] = 0.0
        self._roll["B"] = 0.0
        # LEFT (A): short-axis (cardiology: LV right, RV left, diaphragm down —
        # same convention as the SAX pane).
        _o2, ex, ey, nn = ax.short_axis_basis(along_mid)
        self._frame["A"] = (-np.asarray(ex, float), np.asarray(ey, float),
                            -np.asarray(nn, float))
        self._pc["A"] = np.asarray(_o2, float).copy()
        self._cross_ang["A"] = 0.0
        self._roll["A"] = 0.0
        self._center = mid.copy()
        self._view_initial = True
        self._refresh(reset_cam=True)
        for k in ("A", "B"):
            self._overlay[k].update()

    def _lvv_endo_mask_cached(self):
        """Rasterise the auto Endo surface (valve-clipped) to (comp, bbox) once
        and cache it by surface identity — shared by the Endo volume + LV Diameter
        readouts so the per-paint result block doesn't re-rasterise. (None, None)
        until the Auto-Endo surface is built."""
        surf = getattr(self, "_lv_endo_auto_surf", None)
        if surf is None or self._vol is None:
            return None, None
        cache = getattr(self, "_lvv_endo_mask_cache", None)
        if cache is not None and cache[0] is surf:
            return cache[1], cache[2]
        lvv = self._lvv or {}
        av, mv, apex = lvv.get("aortic"), lvv.get("mitral"), lvv.get("apex")
        comp = bb = None
        if av is not None and mv is not None and apex is not None:
            try:
                planes = [(np.asarray(av[0], float), np.asarray(av[1], float)),
                          (np.asarray(mv[0], float), np.asarray(mv[1], float))]
                comp, bb = surf.inside_mask_bbox(
                    self._dims, self._vol.shape, planes, np.asarray(apex, float))
            except Exception:                        # noqa: BLE001
                comp = bb = None
        self._lvv_endo_mask_cache = (surf, comp, bb)
        return comp, bb

    def _lvv_endo_volume_ml(self):
        """Endo (endocardial-envelope) volume in mL = the auto Endo mask voxel
        count × voxel volume. Endo ⊇ Blood (Endo − Blood = pap/trab tissue) and
        Epi ⊇ Endo (Epi − Endo = compact wall). None until the Auto-Endo surface
        is built (Auto-Endo表示 / 壁厚)."""
        comp, _bb = self._lvv_endo_mask_cached()
        if comp is None:
            return None
        sx, sy, sz = self._dims
        return (float(np.count_nonzero(comp))
                * (float(sx) * float(sy) * float(sz)) / 1000.0)

    def _lvv_lv_diameter_mm(self):
        """LV Diameter = the maximum endocardial diameter on the planes ⟂ the LV
        long axis (clinical LVDd-style widest short-axis chord: per along-axis
        level the true max chord, then the max over levels). NOT 2·max-radius,
        which overreads on a flared basal / LVOT slice. None until the Auto-Endo
        surface is built."""
        comp, bb = self._lvv_endo_mask_cached()
        surf = getattr(self, "_lv_endo_auto_surf", None)
        ax = getattr(surf, "axis", None) if surf is not None else None
        if comp is None or bb is None or ax is None:
            self._lvv_diam_pts = None
            return None
        try:
            from multi_dicomviewer.core.lv_compact import max_perp_diameter
            det = max_perp_diameter(comp, bb, ax.apex, ax.axis, ax.radial0,
                                    self._dims, return_detail=True)
            if det is None:
                self._lvv_diam_pts = None
                return None
            self._lvv_diam_pts = (np.asarray(det[1], float),
                                  np.asarray(det[2], float))
            return float(det[0])
        except Exception:                            # noqa: BLE001
            self._lvv_diam_pts = None
            return None

    def _lv_mode_has_unsaved(self, mode) -> bool:
        if mode == "blood":
            return (self._lvv is not None
                    and self._lvv.get("last_ml") is not None
                    and getattr(self, "_lvv_dirty", False))
        return (self._lv is not None
                and bool(self._lv["model"].endo_planes
                         or self._lv["model"].epi_planes)
                and getattr(self, "_lv_dirty", False))

    def _lv_confirm_drop(self, mode) -> bool:
        from PyQt6.QtWidgets import QMessageBox
        if not self._lv_mode_has_unsaved(mode):
            return True
        return QMessageBox.question(
            self.window(), t("LV"),
            t("This sub-mode has unsaved data. Switch without saving?")) \
            == QMessageBox.StandardButton.Yes

    def _lv_style_selectors(self) -> None:
        sm = self._lv_current_submode()
        m = self._lv["model"] if self._lv is not None else None
        epi_loaded = (getattr(self, "_lvv_epi_surf", None) is not None
                      or (m is not None and len(m.epi_contours) >= 3))
        endo_loaded = (m is not None and len(m.endo_contours) >= 3)
        blood_loaded = getattr(self, "_lvv_blood_comp", None) is not None
        off = self._LV_STY.get("off", "")
        self._lv_epi_btn.setStyleSheet(
            self._LV_STY.get("epi", "") if (sm == "epi" or epi_loaded) else off)
        be_on = (sm in ("blood", "endo") or blood_loaded or endo_loaded)
        self._lvv_start_btn.setStyleSheet(
            ("QPushButton{background:#2e8b57;color:white;}" + self._BTN_DIS)
            if be_on
            else ("QPushButton:checked{background:#2e8b57;color:white;}"
                  + self._BTN_DIS))

    def _lv_update_submode_ui(self) -> None:
        if not hasattr(self, "_lv_grp_trace"):
            return
        sm = self._lv_current_submode()
        endoepi = sm in ("endo", "epi")
        blood = sm == "blood"

        def _vis(w, on):
            if w.isVisible() != on:
                w.setVisible(on)
        _vis(self._lv_grp_trace, endoepi)
        _vis(self._lv_grp_blood, blood)
        _vis(self._lv_grp_r2_trace, endoepi)
        _vis(self._lv_grp_r2_blood, blood)
        if getattr(self, "_lv_grp_r2_valves", None) is not None:
            _vis(self._lv_grp_r2_valves, sm is None)
        self._lv_update_valve_buttons()
        ready = self._lv_valves_ready()
        if not ready:
            self._lv_epi_btn.setEnabled(sm == "epi")
            self._lvv_start_btn.setEnabled(sm == "blood")
        else:
            self._lv_epi_btn.setEnabled(True)
            self._lvv_start_btn.setEnabled(True)
        if self._lvv_start_btn.isChecked() != blood:
            self._lvv_start_btn.setChecked(blood)
        if self._lv is None or self._lv.get("sax") is None:
            self._lv_style_selectors()

    def _lv_select_submode(self, sm) -> None:
        from PyQt6.QtWidgets import QMessageBox
        cur = self._lv_current_submode()
        if sm == cur:
            if sm == "blood":
                if not self._lv_confirm_drop("blood"):
                    return
                self._lvv_toggle()
            elif self._lv is not None and self._lv.get("sax") is None:
                m = self._lv["model"]
                if bool(m.endo_planes or m.epi_planes):
                    if not self._lv_confirm_drop("contour"):
                        return
                    self._lv_exit()
                else:
                    self._lv["pass"] = None
                    self._lv_apply_target(None)
                    self._lv_sync_buttons()
            self._lv_update_submode_ui()
            return
        if not self._lv_valves_ready():
            QMessageBox.information(
                self.window(), t("LV"),
                t("Set the MV and AoV planes first. Draw an Ellipse on each "
                  "annulus and press MV plane / AoV plane (or Load them); then "
                  "Epi / Blood/Endo become available."))
            return
        if sm == "epi":
            if self._lvv is not None:
                self._lvv_deactivate()
            # Returning to Epi with no current Epi trace to resume — whether we
            # came straight from Blood/Endo or it was closed first (entering it
            # runs _lv_exit(), dropping the contour _lv) — restore the Epi that
            # was carried into Blood (_lvv_epi_model_dict) so the border shows
            # immediately, with no re-load / recompute.
            has_cur_epi = (self._lv is not None
                           and len(self._lv["model"].epi_contours) >= 3)
            retain_epi = (None if has_cur_epi
                          else getattr(self, "_lvv_epi_model_dict", None))
            if retain_epi is not None:
                try:
                    from multi_dicomviewer.core.lv_measure import LVModel
                    model = LVModel.from_dict(retain_epi)
                    model.build()
                    if (model.epi_axis is not None
                            and len(model.epi_contours) >= 3):
                        self._lv_apply_model(
                            model, volume=retain_epi.get("volume"))
                        self._lv_update_submode_ui()
                        return
                except Exception:                    # noqa: BLE001
                    pass
            self._lv_select_pass("epi")
        elif sm == "blood":
            if self._lv is not None:
                self._lv_stash_epi_for_blood(self._lv["model"])
                self._lv_exit()
            self._lvv_toggle()
        self._lv_update_submode_ui()

    def _lv_exit_all(self) -> None:
        if self._lvv is not None:
            if not self._lv_confirm_drop("blood"):
                return
            self._lvv_clear_markers()
            self._lvv = None
            self._lvv_sync()
        if self._lv is not None:
            self._lv_exit_confirm()
        self._lv_update_submode_ui()

    def _build_lv_bar(self) -> QWidget:
        """LV EF bar (below the image): Endo/Epi ENTER LV mode and select the
        pass; Set axis captures the current long-axis view as that pass's axis;
        Trace places the apex then traces. Buttons pack from the LEFT. The
        non-entry controls are enabled only while in LV mode (refined by phase
        in _lv_sync_buttons). Mirrors the VTK viewer's 2-pass flow."""
        self._lv_wrap = QWidget()
        self._lv_wrap._mdv_keep_on_max = True
        outer = QVBoxLayout(self._lv_wrap)
        outer.setContentsMargins(8, 2, 8, 2); outer.setSpacing(2)
        row1 = QHBoxLayout(); row1.setSpacing(4)
        row2 = QHBoxLayout(); row2.setSpacing(4)
        outer.addLayout(row1); outer.addLayout(row2)
        cap = QLabel(t("LV:"))
        f = cap.font(); f.setBold(True); cap.setFont(f)
        row1.addWidget(cap)
        self._lv_btn = FitButton(t("Trace"))     # hidden internal mode flag
        self._lv_btn.setCheckable(True); self._lv_btn.setVisible(False)
        # Build the Blood GROUP first (creates _lvv_start_btn / _lv_grp_blood /
        # _lv_grp_r2_blood); embedded into row1/row2 below.
        _blood_grp = self._build_lvv_bar()
        # ---- Common valve planes (MV / AoV): set once, shared by every sub-mode.
        self._lv_mv_btn = FitButton(t("MV plane"))
        self._lv_mv_btn.setHelpToolTip(
            t("Draw an Ellipse on the mitral annulus (Measure→Ellipse), then "
              "press this to set the COMMON MV plane"))
        self._lv_mv_btn.clicked.connect(
            lambda: self._lv_capture_valve_common("mitral"))
        row1.addWidget(self._lv_mv_btn)
        self._lv_aov_btn = FitButton(t("AoV plane"))
        self._lv_aov_btn.setHelpToolTip(
            t("Draw an Ellipse on the aortic annulus (Measure→Ellipse), then "
              "press this to set the COMMON AoV plane"))
        self._lv_aov_btn.clicked.connect(
            lambda: self._lv_capture_valve_common("aortic"))
        row1.addWidget(self._lv_aov_btn)
        row1.addSpacing(8)
        # ---- Sub-mode selector: Epi → Blood/Endo (Endo merged into Blood/Endo).
        self._lv_epi_btn = FitButton(t("Epi"))
        self._lv_epi_btn.setHelpToolTip(
            t("Epi (myocardial) pass — align its long-axis view, Set axis, then "
              "Trace"))
        self._lv_epi_btn.clicked.connect(lambda: self._lv_select_submode("epi"))
        row1.addWidget(self._lv_epi_btn)
        row1.addWidget(self._lvv_start_btn)      # "Blood/Endo" (built in lvv bar)
        self._lv_endo_btn = FitButton(t("Endo"))
        self._lv_endo_btn.clicked.connect(lambda: self._lv_select_submode("endo"))
        self._lv_endo_btn.setVisible(False)
        row1.addSpacing(8)
        # ---- Endo/Epi trace GROUP (row 1) ----
        self._lv_grp_trace = QWidget()
        gt = QHBoxLayout(self._lv_grp_trace)
        gt.setContentsMargins(0, 0, 0, 0); gt.setSpacing(4)
        # 'Set axis' retired — Trace auto-captures the axis. Hidden placeholder.
        self._lv_setaxis_btn = FitButton(t("Set axis"))
        self._lv_setaxis_btn.setVisible(False)
        self._lv_setaxis_btn.clicked.connect(self._lv_set_axis)
        # Apex button: set this pass's apex at the centreline crossing (no
        # image-click). Move the crossing onto the apex, then press Apex → Trace.
        self._lv_apex_btn = FitButton(t("Apex"))
        self._lv_apex_btn.setHelpToolTip(
            t("Set the LV apex at the centreline crossing (move the crossing "
              "onto the apex first), then press Trace."))
        self._lv_apex_btn.clicked.connect(self._lv_confirm_apex_trace)
        gt.addWidget(self._lv_apex_btn)
        self._lv_trace_btn = FitButton(t("Trace"))
        self._lv_trace_btn.clicked.connect(self._lv_start_trace)
        gt.addWidget(self._lv_trace_btn)
        self._lv_prev_btn = FitButton(t("◀ Prev plane (A)"))
        self._lv_prev_btn.clicked.connect(lambda: self._lv_step_plane(-1))
        gt.addWidget(self._lv_prev_btn)
        self._lv_plane_lbl = QLabel("0/6")
        self._lv_plane_lbl.setMinimumWidth(78)
        fl = self._lv_plane_lbl.font(); fl.setBold(True)
        self._lv_plane_lbl.setFont(fl)
        self._lv_plane_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gt.addWidget(self._lv_plane_lbl)
        self._lv_next_btn = FitButton(t("▶ Next plane (F)"))
        self._lv_next_btn.clicked.connect(lambda: self._lv_step_plane(1))
        gt.addWidget(self._lv_next_btn)
        self._lv_sax_btn = FitButton(t("SAX"))
        self._lv_sax_btn.setCheckable(True)
        self._lv_sax_btn.setStyleSheet(
            "QPushButton:checked{background:#b8860b;color:white;}")
        self._lv_sax_btn.clicked.connect(self._lv_toggle_sax)
        gt.addWidget(self._lv_sax_btn)
        row1.addWidget(self._lv_grp_trace)
        # ---- Blood GROUP (row 1) — built above ----
        row1.addWidget(_blood_grp)
        row1.addStretch(1)

        # ================= Row 2 =================
        self._lv_grp_r2_trace = QWidget()
        r2t = QHBoxLayout(self._lv_grp_r2_trace)
        r2t.setContentsMargins(0, 0, 0, 0); r2t.setSpacing(4)
        self._lv_vol_btn = FitButton(t("Calc Vol"))
        self._lv_vol_btn.setStyleSheet(self._LV_STY["vol_todo"])
        self._lv_vol_btn.clicked.connect(self._lv_compute_volume)
        r2t.addWidget(self._lv_vol_btn)
        # Epi領域表示: toggle the red measured region + free the view for 3-D
        # inspection (Rotate/Spin/Paging/CenterLine unlocked while ON).
        self._lv_region_btn = FitButton(t("Epi領域表示"))
        self._lv_region_btn.setCheckable(True)
        self._lv_region_btn.setHelpToolTip(
            t("Show/hide the red measured region (after Calc Vol) and free the "
              "view (Rotate/Spin/Paging/CenterLine) to inspect it in 3-D."))
        self._lv_region_btn.clicked.connect(self._lv_toggle_region)
        r2t.addWidget(self._lv_region_btn)
        self._lv_wall_btn = FitButton(t("Wall"))
        self._lv_wall_btn.setCheckable(True)
        self._lv_wall_btn.setStyleSheet(
            "QPushButton{background:palette(button);border:2px solid #9b59b6;}"
            "QPushButton:checked{background:#8e44ad;color:white;"
            "border:2px solid #8e44ad;}")
        self._lv_wall_btn.clicked.connect(self._lv_toggle_wall)
        self._lv_wall_btn.setVisible(False)
        self._lv_save_btn = FitButton(t("Save"))
        self._lv_save_btn.clicked.connect(self._lv_save)
        r2t.addWidget(self._lv_save_btn)
        self._lv_load_btn = FitButton(t("Load"))
        self._lv_load_btn.clicked.connect(self._lv_load)
        r2t.addWidget(self._lv_load_btn)
        self._lv_stl_btn = FitButton(t("STL"))
        self._lv_stl_btn.clicked.connect(self._lv_export_stl)
        r2t.addWidget(self._lv_stl_btn)
        self._lv_redo_btn = FitButton(t("Clear borders"))
        self._lv_redo_btn.clicked.connect(self._lv_clear_confirm)
        r2t.addWidget(self._lv_redo_btn)
        self._lv_exit_btn = FitButton(t("Exit LV"))
        self._lv_exit_btn.clicked.connect(self._lv_exit_all)
        r2t.addWidget(self._lv_exit_btn)
        row2.addWidget(self._lv_grp_r2_trace)
        row2.addWidget(self._lv_grp_r2_blood)     # built by _build_lvv_bar
        # Valve-setup row-2 group (shown when NO sub-mode active): Save/Load valves.
        self._lv_grp_r2_valves = QWidget()
        r2v = QHBoxLayout(self._lv_grp_r2_valves)
        r2v.setContentsMargins(0, 0, 0, 0); r2v.setSpacing(4)
        self._lv_mv_save_btn = FitButton(t("Save MV"))
        self._lv_mv_save_btn.clicked.connect(lambda: self._lv_save_valve("mitral"))
        r2v.addWidget(self._lv_mv_save_btn)
        self._lv_mv_load_btn = FitButton(t("Load MV"))
        self._lv_mv_load_btn.clicked.connect(lambda: self._lv_load_valve("mitral"))
        r2v.addWidget(self._lv_mv_load_btn)
        self._lv_aov_save_btn = FitButton(t("Save AoV"))
        self._lv_aov_save_btn.clicked.connect(lambda: self._lv_save_valve("aortic"))
        r2v.addWidget(self._lv_aov_save_btn)
        self._lv_aov_load_btn = FitButton(t("Load AoV"))
        self._lv_aov_load_btn.clicked.connect(lambda: self._lv_load_valve("aortic"))
        r2v.addWidget(self._lv_aov_load_btn)
        for b in (self._lv_mv_save_btn, self._lv_mv_load_btn,
                  self._lv_aov_save_btn, self._lv_aov_load_btn):
            b.setStyleSheet(self._BTN_DIS)
        row2.addWidget(self._lv_grp_r2_valves)
        row2.addStretch(1)
        for b in (self._lv_prev_btn, self._lv_next_btn, self._lv_redo_btn,
                  self._lv_save_btn, self._lv_stl_btn, self._lv_load_btn,
                  self._lv_exit_btn):
            b.setStyleSheet(self._BTN_DIS)
        self._lv_bar_btns = [
            self._lv_setaxis_btn, self._lv_trace_btn, self._lv_prev_btn,
            self._lv_next_btn, self._lv_sax_btn, self._lv_vol_btn,
            self._lv_wall_btn, self._lv_redo_btn, self._lv_save_btn,
            self._lv_stl_btn, self._lv_exit_btn]
        # Explicit INITIAL group visibility (no sub-mode active): the idempotent
        # _vis() in _lv_update_submode_ui can't take effect at build time because
        # isVisible() is False before the parent is shown, so set it directly
        # here — only the valve Save/Load row shows until Epi/Blood is picked.
        for _g in (self._lv_grp_trace, self._lv_grp_blood,
                   self._lv_grp_r2_trace, self._lv_grp_r2_blood):
            _g.setVisible(False)
        self._lv_grp_r2_valves.setVisible(True)
        self._lv_sync_buttons()
        self._lv_update_submode_ui()
        return self._lv_wrap

    #: LV bar button styles by state (default = plain grey/black).
    _LV_STY = {
        "endo": "QPushButton{background:#d32f2f;color:white;}",
        "epi": "QPushButton{background:#2e8b57;color:white;}",
        "setaxis": "QPushButton{background:#b8860b;color:white;}",
        "trace": "QPushButton{background:#c0392b;color:white;}",
        # CalcVol: the SAME native background as every other button BEFORE a
        # volume is computed (only a clear blue 2px outline sets it apart — a
        # hint it turns blue once computed); solid blue AFTER (valid result).
        "vol_todo": ("QPushButton{background:palette(button);color:black;"
                     "border:2px solid #1f77b4;}"),
        "vol_done": "QPushButton{background:#1f77b4;color:white;}",
        # SAX/refine neutral (grey/black): the 4 trace buttons reset to this on
        # SAX entry; Endo/Epi re-colour only to show the armed edit target.
        "neutral": "QPushButton{background:#d0d0d0;color:black;}",
    }

    #: Appended to LV Vol button styles so a DISABLED button clearly greys out
    #: (a custom background otherwise overrides Qt's native disabled look).
    _BTN_DIS = ("QPushButton:disabled{background:#e6e6e6;color:#a8a8a8;"
                "border:1px solid #d8d8d8;}")

    def _lv_set_bar_enabled(self, on: bool) -> None:
        """Enable/disable the non-entry LV controls (Endo/Epi/Load stay live)."""
        for b in getattr(self, "_lv_bar_btns", []):
            b.setEnabled(bool(on))

    # ================================================================= LV Vol
    # Blood-pool LVEF: trace Epi (contour LV mode), Exit LV, then measure the
    # blood volume inside the Epi surface, apex-side of the MV/AoV planes, in a
    # HU (contrast) range and 3-D connected to a seed. Ported method-for-method
    # from the VTK viewer; VTK reslice/LUT overlays are re-expressed as QPainter
    # markers + cached RGBA tint images (see _paint_lvv / _lvv_plane_rgba). The
    # heavy volume runs off-thread (_LvvWorker) behind a busy progress bar.
    def _build_lvv_bar(self) -> QWidget:
        # Blood row-1 GROUP inside the unified LV bar (the caption + Start button +
        # common MV/AoV live in the selector row built by _build_lv_bar). Returned
        # as _lv_grp_blood; shown only while the Blood/Endo sub-mode is active.
        self._lv_grp_blood = QWidget()
        self._lvv_wrap = self._lv_grp_blood
        row = QHBoxLayout(self._lv_grp_blood)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        # Blood/Endo selector button (lives in the selector row; kept as an attr
        # for _lvv_sync). Relabelled from the old standalone "Start".
        self._lvv_start_btn = FitButton(t("Blood/Endo"))
        self._lvv_start_btn.setCheckable(True)
        self._lvv_start_btn.setHelpToolTip(
            t("Blood + Endo sub-mode: set the blood HU range, compute the blood "
              "volume, then derive & edit the Endo border (needs Epi + MV/AoV)"))
        self._lvv_start_btn.clicked.connect(lambda: self._lv_select_submode("blood"))
        self._lvv_apex_btn = FitButton(t("Apex"))
        self._lvv_apex_btn.setHelpToolTip(
            t("Confirm the LV apex at the crosshair (move it there first)"))
        self._lvv_apex_btn.clicked.connect(self._lvv_confirm_apex)
        row.addWidget(self._lvv_apex_btn)
        # MV/AoV are the COMMON planes (selector row) in the unified bar; the
        # blood-local capture buttons are kept (attrs) but not shown.
        self._lvv_mv_btn = FitButton(t("MV plane"))
        self._lvv_mv_btn.clicked.connect(lambda: self._lvv_capture_valve("mitral"))
        self._lvv_mv_btn.setVisible(False)
        self._lvv_aov_btn = FitButton(t("AoV plane"))
        self._lvv_aov_btn.clicked.connect(lambda: self._lvv_capture_valve("aortic"))
        self._lvv_aov_btn.setVisible(False)
        # 内腔ROI (Polygon HU sampling) is retired — the blood HU range starts at
        # 200–500 and is fine-tuned with the spin arrows (the seed is auto = the
        # largest in-range component inside the Epi). Hidden placeholder keeps
        # old references harmless.
        self._lvv_thr_btn = FitButton(t("内腔ROI"))
        self._lvv_thr_btn.setVisible(False)
        self._lvv_thr_btn.setHelpToolTip(
            t("Draw a Polygon inside the LV cavity (Measure→Polygon), then press "
              "this: it colours the ROI, seeds it, and sets the HU range from "
              "the pixels inside — adjust 下限/上限 below."))
        self._lvv_thr_btn.clicked.connect(self._lvv_capture_roi)
        # (not added to the row — retired)
        # Blood HU RANGE (lower / upper). Defaults 200–500; adjust with the arrows.
        self._lvv_lo_lbl = QLabel(t("下限"))
        self._lvv_lo_spin = QSpinBox()
        self._lvv_lo_spin.setRange(-1000, 4000)
        self._lvv_lo_spin.setSingleStep(10)
        self._lvv_lo_spin.setValue(200)                # blood-pool default lo
        self._lvv_lo_spin.setSuffix(" HU")
        self._lvv_lo_spin.setKeyboardTracking(False)
        self._lvv_hi_lbl = QLabel(t("上限"))
        self._lvv_hi_spin = QSpinBox()
        self._lvv_hi_spin.setRange(-1000, 4000)
        self._lvv_hi_spin.setSingleStep(10)
        self._lvv_hi_spin.setValue(500)                # blood-pool default hi
        self._lvv_hi_spin.setSuffix(" HU")
        self._lvv_hi_spin.setKeyboardTracking(False)
        self._lvv_lo_spin.valueChanged.connect(lambda _v: self._lvv_hu_changed())
        self._lvv_hi_spin.valueChanged.connect(lambda _v: self._lvv_hu_changed())
        for _w in (self._lvv_lo_lbl, self._lvv_lo_spin,
                   self._lvv_hi_lbl, self._lvv_hi_spin):
            row.addWidget(_w)
        # 全域HU表示: instant tint of ALL in-range voxels (no compute) — the
        # DEFAULT view on entering the mode, for tuning 下限/上限. Exclusive with
        # LV-Blood表示 (the computed, Epi-clipped region).
        self._lvv_hl_on = True
        self._lvv_hl_btn = FitButton(t("全域HU表示"))
        self._lvv_hl_btn.setCheckable(True)
        self._lvv_hl_btn.setChecked(True)
        self._lvv_hl_btn.setHelpToolTip(
            t("Instantly tint every voxel whose HU is in the 下限–上限 range on "
              "both panes (no compute) — adjust 下限/上限 to optimise. Turned off "
              "automatically while LV-Blood表示 is shown."))
        self._lvv_hl_btn.clicked.connect(self._lvv_toggle_highlight)
        row.addWidget(self._lvv_hl_btn)
        # LV-Blood表示: compute the blood WITHIN the Epi surface (largest in-range
        # connected component, apex-side of MV/AoV) and show it in 水色 + report
        # the volume. Recomputes when the HU range changed since.
        self._lvv_mask_btn = FitButton(t("LV-Blood表示"))
        self._lvv_mask_btn.setCheckable(True)
        self._lvv_mask_btn.setChecked(False)
        self._lvv_mask_btn.setHelpToolTip(
            t("Compute + show the LV blood inside the Epi surface (水色) and its "
              "volume. Exclusive with 全域HU表示; recomputes on HU-range change."))
        self._lvv_mask_btn.clicked.connect(self._lvv_toggle_blood)
        row.addWidget(self._lvv_mask_btn)
        # "Epi境界" toggle: draw the Epi surface where it crosses each pane
        # (green) — to judge whether coronary voxels contaminate the cavity.
        self._lvv_epi_show = False
        self._lvv_epi_btn = FitButton(t("Epi境界"))
        self._lvv_epi_btn.setCheckable(True)
        self._lvv_epi_btn.setHelpToolTip(
            t("Show the Epi border on both panes (green)"))
        self._lvv_epi_btn.clicked.connect(self._lvv_toggle_epi)
        row.addWidget(self._lvv_epi_btn)
        # Auto-Endo表示: display-only overlay of the endocardial envelope derived
        # from the blood pool (orange). Recomputes for the current HU range.
        self._lvv_auto_endo_btn = FitButton(t("Auto-Endo表示"))
        self._lvv_auto_endo_btn.setCheckable(True)
        self._lvv_auto_endo_btn.setHelpToolTip(
            t("Show the auto Endo border (from the blood pool, papillary filled) "
              "as a display-only overlay — recomputes for the current HU range."))
        self._lvv_auto_endo_btn.clicked.connect(self._lvv_toggle_auto_endo)
        row.addWidget(self._lvv_auto_endo_btn)
        # Auto-Endo 係数: papillary/trabecula BRIDGING radius (close_mm). Larger =
        # smoother (trabeculae included); smaller = follows the blood indents.
        self._lvv_close_lbl = QLabel(t("肉柱"))
        self._lvv_close_spin = QSpinBox()
        self._lvv_close_spin.setRange(1, 12)
        self._lvv_close_spin.setValue(5)
        self._lvv_close_spin.setSuffix(" mm")
        self._lvv_close_spin.setKeyboardTracking(False)
        self._lvv_close_spin.setToolTip(
            t("Auto-Endo の肉柱/乳頭筋の凹凸を橋渡しする量 (close_mm): 大きいほど"
              "滑らか＝緻密層寄り、小さいほど血流に忠実"))
        self._lvv_close_spin.valueChanged.connect(
            lambda _v: self._lvv_close_changed())
        row.addWidget(self._lvv_close_lbl)
        row.addWidget(self._lvv_close_spin)
        # Manual-Endo: enter the Endo edit mode (13 handles). Seeds from Auto-Endo
        # the first time; the hand-edited border is retained across HU changes.
        self._lvv_manual_endo_btn = FitButton(t("Manual-Endo"))
        self._lvv_manual_endo_btn.setHelpToolTip(
            t("Edit the Endo border by hand. Seeded from Auto-Endo the first "
              "time; kept when the HU range changes."))
        self._lvv_manual_endo_btn.clicked.connect(self._lvv_manual_endo)
        row.addWidget(self._lvv_manual_endo_btn)
        # 壁厚 (wall thickness) heat maps — mutually exclusive.
        self._lvv_thick3d_btn = FitButton(t("壁厚3D"))
        self._lvv_thick3d_btn.setCheckable(True)
        self._lvv_thick3d_btn.setHelpToolTip(
            t("Myocardial wall thickness = Endo→Epi nearest distance (EDT), "
              "coloured with the clinical bands. Needs Blood + Epi."))
        self._lvv_thick3d_btn.clicked.connect(
            lambda: self._lvv_set_thick_mode("3d"))
        row.addWidget(self._lvv_thick3d_btn)
        self._lvv_thicksax_btn = FitButton(t("壁厚短軸"))
        self._lvv_thicksax_btn.setCheckable(True)
        self._lvv_thicksax_btn.setHelpToolTip(
            t("Myocardial wall thickness measured radially (short-axis / "
              "echo-style) from the LV long axis. Needs Blood + Epi."))
        self._lvv_thicksax_btn.clicked.connect(
            lambda: self._lvv_set_thick_mode("sax"))
        row.addWidget(self._lvv_thicksax_btn)
        # (Per-pane slab thickness is changed with the toolbar "Slab(mm)"
        # control — it stays enabled in LV Vol mode. No dedicated spinboxes.)
        # Calc Vol is retired — LV-Blood表示 computes + shows the volume. Hidden
        # placeholders keep old references harmless.
        self._lvv_calc_btn = FitButton(t("LV Vol計測"))
        self._lvv_calc_btn.setVisible(False)
        self._lvv_calc_btn.clicked.connect(lambda: self._lvv_calc())
        self._lvv_vol_lbl = QLabel("--")
        self._lvv_vol_lbl.setVisible(False)
        # Blood row-2 GROUP: Save / Load / Exit (shown while Blood/Endo active).
        self._lv_grp_r2_blood = QWidget()
        r2b = QHBoxLayout(self._lv_grp_r2_blood)
        r2b.setContentsMargins(0, 0, 0, 0); r2b.setSpacing(4)
        self._lvv_save_btn = FitButton(t("Save"))
        self._lvv_save_btn.setHelpToolTip(
            t("Save the LV Vol landmarks, HU range, Epi surface and volume"))
        self._lvv_save_btn.clicked.connect(self._lvv_save)
        r2b.addWidget(self._lvv_save_btn)
        self._lvv_load_btn = FitButton(t("Load"))
        self._lvv_load_btn.setHelpToolTip(t("Load a saved LV Vol dataset"))
        self._lvv_load_btn.clicked.connect(self._lvv_load)
        r2b.addWidget(self._lvv_load_btn)
        self._lvv_exit_btn = FitButton(t("Exit"))
        self._lvv_exit_btn.clicked.connect(self._lv_exit_all)
        r2b.addWidget(self._lvv_exit_btn)
        self._lvv_ctrl_btns = [
            self._lvv_apex_btn, self._lvv_aov_btn, self._lvv_mv_btn,
            self._lvv_thr_btn, self._lvv_calc_btn, self._lvv_save_btn,
            self._lvv_exit_btn, self._lvv_manual_endo_btn]
        for b in self._lvv_ctrl_btns:
            b.setStyleSheet(self._BTN_DIS)
        self._lvv_load_btn.setStyleSheet(self._BTN_DIS)
        self._lvv_start_btn.setStyleSheet(
            "QPushButton:checked{background:#2e8b57;color:white;}" + self._BTN_DIS)
        self._lvv_sync()
        return self._lv_grp_blood

    def _lvv_sync(self) -> None:
        on = self._lvv is not None
        self._lvv_start_btn.setChecked(on)
        g = (lambda k: on and self._lvv.get(k) is not None)
        apex_done, aov_done = g("apex"), g("aortic")
        mv_done = g("mitral")
        # Ready to compute LV-Blood: apex + valves + an Epi surface (no ROI now —
        # the seed is auto = the largest in-range component inside the Epi).
        ready = on and apex_done and aov_done and (self._lvv_epi_surf is not None)
        self._lvv_apex_btn.setEnabled(on)
        self._lvv_mv_btn.setEnabled(on and apex_done)
        self._lvv_aov_btn.setEnabled(on and mv_done)
        self._lvv_mask_btn.setEnabled(ready)        # LV-Blood表示
        self._lvv_save_btn.setEnabled(on and apex_done)
        self._lvv_load_btn.setEnabled(self._vol is not None)
        self._lvv_exit_btn.setEnabled(on)

        def _done(btn, is_set, color):
            """Colour *btn* when its landmark is set (still greys when off)."""
            if on and is_set:
                btn.setStyleSheet(
                    "QPushButton{background:%s;color:white;}%s"
                    % (color, self._BTN_DIS))
            else:
                btn.setStyleSheet(self._BTN_DIS)

        _done(self._lvv_apex_btn, apex_done, "#d32f2f")     # apex red
        _done(self._lvv_aov_btn, aov_done, "#b8860b")       # aortic amber
        _done(self._lvv_mv_btn, mv_done, "#2b6cb0")         # mitral blue
        # HU spins + the 全域HU / LV-Blood toggles are available whenever Blood is
        # active (the 全域HU tint is the default view; adjust 下限/上限 then press
        # LV-Blood表示 to compute the Epi-clipped region).
        for w in (self._lvv_lo_lbl, self._lvv_lo_spin,
                  self._lvv_hi_lbl, self._lvv_hi_spin, self._lvv_hl_btn):
            w.setVisible(on)
        if getattr(self, "_lvv_close_spin", None) is not None:
            self._lvv_close_lbl.setVisible(on)
            self._lvv_close_spin.setVisible(on)
        self._lvv_style_toggle(self._lvv_hl_btn, "#40c0ff", "black")
        self._lvv_mask_btn.setVisible(on)
        self._lvv_style_toggle(self._lvv_mask_btn, "#40e0ff", "black")
        # Auto-Endo表示 / Manual-Endo: available once the region CAN be computed
        # (apex + valves + Epi) — they compute the blood internally.
        blood_ok = self._lvv_blood_comp is not None
        if getattr(self, "_lvv_auto_endo_btn", None) is not None:
            self._lvv_auto_endo_btn.setEnabled(on and (ready or blood_ok))
            self._lvv_auto_endo_btn.setVisible(on)
            self._lvv_auto_endo_btn.setChecked(bool(self._lvv_endo_show))
            self._lvv_style_toggle(self._lvv_auto_endo_btn, "#ff8c28", "black")
        if getattr(self, "_lvv_manual_endo_btn", None) is not None:
            self._lvv_manual_endo_btn.setEnabled(
                on and (ready or blood_ok
                        or self._lv_endo_manual_dict is not None))
            self._lvv_manual_endo_btn.setVisible(on)
        # Epi-border toggle: available in the mode.
        self._lvv_epi_btn.setVisible(on and self._lvv_epi_surf is not None)
        # 壁厚 buttons: visible in the mode, enabled once computable.
        for _m, (_attr, _col, _lbl) in getattr(
                self, "_THICK_MODES", {}).items():
            btn = getattr(self, _attr, None)
            if btn is not None:
                btn.setVisible(on)
        if on:
            self._lvv_thick_sync_buttons()

    def _lvv_prompt(self, text) -> None:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self.window(), t("LV Vol"), text)

    def _lvv_toggle(self, *args) -> None:
        from PyQt6.QtWidgets import QMessageBox
        try:
            if self._lvv is None:
                if self._vol is None:
                    QMessageBox.information(
                        self.window(), t("LV Vol"),
                        t("Load a CT first (no volume in this pane)."))
                    return
                if self._lv is not None:          # leave contour LV mode first
                    self._lv_exit()
                if self._lvv_epi_surf is None:
                    QMessageBox.information(
                        self.window(), t("LV Vol"),
                        t("Trace the EPI border first (contour LV mode: Epi → "
                          "Set axis → Trace), then Exit LV and start LV Vol."))
                    return
                self._lvv = {"apex": None, "aortic": None, "mitral": None,
                             "hu_lo": None, "hu_hi": None, "seed": None,
                             "step": "apex", "last_ml": None, "calc_sig": None}
                # Default the apex to the EPI apex so LV-Blood / Auto-Endo work
                # WITHOUT a manual Apex step (Apex button can override).
                if getattr(self, "_lvv_epi_apex", None) is not None:
                    self._lvv["apex"] = np.asarray(self._lvv_epi_apex, float)
                    self._lvv["step"] = "ready"
                # Inherit the COMMON MV/AoV planes (unified LV bar) so Blood uses
                # the same valves as Epi — no separate per-Blood valve capture.
                if self._lv_valves.get("mitral") is not None:
                    self._lvv["mitral"] = self._lv_valves["mitral"]
                if self._lv_valves.get("aortic") is not None:
                    self._lvv["aortic"] = self._lv_valves["aortic"]
                # Blood/Endo evaluates HU on THIN slices — force BOTH panes' slab
                # to 0 (the right pane often carried a 5mm MPR slab).
                self._thick["A"] = 0.0
                self._thick["B"] = 0.0
                if hasattr(self, "_sync_slab_spin"):
                    self._sync_slab_spin()
                # Lay the panes on the LV long axis: right = long-axis view, left
                # = orthogonal short-axis cut (from the Epi axis).
                self._lvv_setup_axis_views()
                # 全域HU tint ON by default; LV-Blood off until computed.
                self._lvv_hl_on = True
                self._lvv_mask_on = False
                self._lvv_hl_btn.setChecked(True)
                self._lvv_mask_btn.setChecked(False)
                self._lvv_sync()
                for k in ("A", "B"):
                    self._overlay[k].update()
                self._lvv_prompt(
                    t("Epi border loaded. Move the crosshair onto the LV apex "
                      "(left double-click recentres), then press Apex. (This "
                      "apex is for the valve orientation — it can differ from "
                      "the Epi apex.)"))
                return
            else:
                self._lvv_clear_markers()
                self._lvv = None
            self._lvv_sync()
            for k in ("A", "B"):
                self._overlay[k].update()
        except Exception as exc:                        # noqa: BLE001
            import traceback
            QMessageBox.critical(self.window(), t("LV Vol (toggle error)"),
                                 traceback.format_exc() or repr(exc))

    def _lvv_confirm_apex(self) -> None:
        """Apex button → confirm the apex at the current crosshair centre."""
        lvv = self._lvv
        if lvv is None or self._center is None:
            return
        P = np.asarray(self._center, float).copy()
        lvv["apex"] = P
        lvv["step"] = "mv"
        self._lvv_sync()
        for k in ("A", "B"):
            self._overlay[k].update()
        self._lvv_prompt(
            t("Identify the mitral valve: draw an Ellipse on the MV annulus "
              "(Measure→Ellipse), then press 'MV plane'."))

    def _lvv_hu_at(self, P) -> float:
        """Nearest-voxel HU at world point *P* (mm) in self._vol (indexed
        vol[z, y, x], world = index × spacing = self._dims). Clamped to the
        volume (mirrors the VTK viewer's _lvv_hu_at)."""
        sx, sy, sz = self._dims
        nz, ny, nx = self._vol.shape
        ix = min(max(int(round(P[0] / sx)), 0), nx - 1)
        iy = min(max(int(round(P[1] / sy)), 0), ny - 1)
        iz = min(max(int(round(P[2] / sz)), 0), nz - 1)
        return float(self._vol[iz, iy, ix])

    def _lvv_plane_rgba(self, key, kind):
        """Per-pane RGBA tint image sampled on the pane's current oblique plane:
        kind='cyan' tints voxels with hu_lo<=HU<=hu_hi (blood); kind='red' tints
        the measured-region mask (self._lvv_mask_vol) voxels. Drawn stretched
        over the pane in paintEvent. Returns None if nothing to tint. Mirrors
        _compute_slab_qimage's plane geometry (single centre plane)."""
        if self._vol is None:
            return None
        par = self._slab_params(key)
        u, v, n = par["u"], par["v"], par["n"]
        pc = par["pc"]
        sx, sy, sz = par["dims"]
        px, py = par["pan"]
        pw, ph, iw, ih = par["pw"], par["ph"], par["iw"], par["ih"]
        scale = ph / (2.0 * max(1e-3, par["ps"]))
        a = math.radians(par["roll"])
        ca, sa = math.cos(a), math.sin(a)
        SX, SY = np.meshgrid(np.linspace(0.0, pw, iw), np.linspace(0.0, ph, ih))
        aa = (SX - pw / 2.0) / scale
        bb = (ph / 2.0 - SY) / scale
        WX = px + aa * ca - bb * sa
        WY = py + aa * sa + bb * ca
        bx = pc[0] + WX * u[0] + WY * v[0]
        by = pc[1] + WX * u[1] + WY * v[1]
        bz = pc[2] + WX * u[2] + WY * v[2]
        vx = bx / sx; vy = by / sy; vz = bz / sz
        nz, ny, nx = self._vol.shape
        oob = ((vx < 0) | (vx > nx - 1) | (vy < 0) | (vy > ny - 1)
               | (vz < 0) | (vz > nz - 1))
        if kind == "wall":
            if getattr(self, "_lvv_thick_vol", None) is None:
                return None
            mm = _trilinear_sample(self._lvv_thick_vol, vx, vy, vz)
            rgba = _wall_rgba_from_mm(mm, self._wall_thresholds(), 0.55)
            rgba[oob] = 0
            if not (rgba[:, :, 3] > 0).any():
                return None
            rgba = np.ascontiguousarray(rgba)
            return QImage(rgba.data, iw, ih, 4 * iw,
                          QImage.Format.Format_RGBA8888).copy()
        if kind == "cyan":
            lo = float(self._lvv_lo_spin.value())
            hi = float(self._lvv_hi_spin.value())
            hu = _trilinear_sample(self._vol, vx, vy, vz)
            inmask = (hu >= lo) & (hu <= hi) & ~oob
            col = (26, 230, 255, 115)          # cyan (0.1,0.9,1.0,0.45)
        else:
            if self._lvv_mask_vol is None:
                return None
            mv = _trilinear_sample(self._lvv_mask_vol, vx, vy, vz)
            inmask = (mv >= 0.5) & ~oob
            col = (64, 191, 255, 140)          # LV-Blood = 水色 (light blue)
        if not inmask.any():
            return None
        rgba = np.zeros((ih, iw, 4), np.uint8)
        rgba[inmask] = col
        rgba = np.ascontiguousarray(rgba)
        return QImage(rgba.data, iw, ih, 4 * iw,
                      QImage.Format.Format_RGBA8888).copy()

    def _lvv_refresh_overlays(self, key) -> None:
        """(Re)build the cyan (in-range blood), red (measured region) and 壁厚
        heat-map tint images for one pane from the current toggles. Called per
        pane from _refresh and from the toggle/spin handlers."""
        cyan = red = thick = None
        if self._lvv is not None:
            # Exclusive fills: 全域HU tint (cyan) when LV-Blood is NOT shown; the
            # computed 水色 region (red image slot) when it is.
            if self._lvv_hl_on and not self._lvv_mask_on:
                cyan = self._lvv_plane_rgba(key, "cyan")
            if self._lvv_mask_vol is not None and self._lvv_mask_on:
                red = self._lvv_plane_rgba(key, "red")
            if (getattr(self, "_lvv_thick_mode", None) is not None
                    and getattr(self, "_lvv_thick_vol", None) is not None):
                thick = self._lvv_plane_rgba(key, "wall")
        self._lvv_cyan_img[key] = cyan
        self._lvv_red_img[key] = red
        self._lvv_thick_img[key] = thick

    def _lvv_redraw(self) -> None:
        for k in ("A", "B"):
            self._lvv_refresh_overlays(k)
            self._overlay[k].update()

    def _lvv_update_highlight(self) -> None:
        """Rebuild the in-range (blood) voxel tint on both panes."""
        self._lvv_redraw()

    def _lvv_signature(self):
        """Hashable snapshot of the inputs that define the measured region: apex,
        AoV & MV planes (centre + normal), ROI seed, and the HU range read LIVE
        from the spins. Stored at each successful measure (lvv['calc_sig']) and
        compared on the next LV Vol計測 press — equal → toggle the red overlay
        only (no recompute); different → recompute. Returns None if incomplete."""
        lvv = self._lvv
        if lvv is None:
            return None

        def rt(v, nd=3):
            return tuple(round(float(x), nd)
                         for x in np.asarray(v, float).ravel())
        try:
            apex = lvv["apex"]
            aortic, mitral = lvv["aortic"], lvv["mitral"]
            if apex is None or aortic is None or mitral is None:
                return None
            c_a, n_a = aortic[0], aortic[1]
            c_m, n_m = mitral[0], mitral[1]
        except (KeyError, TypeError, IndexError):
            return None
        lo = round(float(self._lvv_lo_spin.value()), 3)
        hi = round(float(self._lvv_hi_spin.value()), 3)
        return (rt(apex), rt(c_a), rt(n_a, 4), rt(c_m), rt(n_m, 4), lo, hi)

    def _lvv_toggle_highlight(self, *args) -> None:
        # 全域HU表示 and LV-Blood表示 are exclusive.
        self._lvv_hl_on = self._lvv_hl_btn.isChecked()
        if self._lvv_hl_on and self._lvv_mask_on:
            self._lvv_mask_on = False
            self._lvv_mask_btn.setChecked(False)
            self._lvv_style_toggle(self._lvv_mask_btn, "#40e0ff", "black")
        self._lvv_style_toggle(self._lvv_hl_btn, "#40c0ff", "black")
        self._lvv_redraw()

    def _lvv_close_changed(self) -> None:
        """Auto-Endo 肉柱 bridging (close_mm) changed → drop the stale auto Endo +
        hide Auto-Endo表示 so a re-press recomputes. Manual Endo is unaffected."""
        self._lv_endo_close_mm = float(self._lvv_close_spin.value())
        self._lv_endo_auto_model = None
        self._lv_endo_auto_surf = None
        self._lv_endo_auto_sig = None
        if getattr(self, "_lvv_endo_show", False):
            self._lvv_endo_show = False
            if getattr(self, "_lvv_auto_endo_btn", None) is not None:
                self._lvv_auto_endo_btn.setChecked(False)
                self._lvv_style_toggle(self._lvv_auto_endo_btn, "#ff8c28", "black")
        if getattr(self, "_lvv_thick_mode", None) is not None or getattr(
                self, "_lvv_thick_cache", None):
            self._lvv_thick_clear()        # Endo changed → wall thickness stale
        for k in ("A", "B"):
            self._overlay[k].update()

    def _lvv_hu_changed(self) -> None:
        """HU 下限/上限 changed: the computed LV-Blood region no longer matches, so
        drop the 水色 region and fall back to the instant 全域HU tint. Re-press
        LV-Blood表示 to recompute for the new range."""
        if self._lvv_mask_on:
            self._lvv_mask_on = False
            self._lvv_mask_btn.setChecked(False)
            self._lvv_style_toggle(self._lvv_mask_btn, "#40e0ff", "black")
            self._lvv_hl_on = True
            self._lvv_hl_btn.setChecked(True)
            self._lvv_style_toggle(self._lvv_hl_btn, "#40c0ff", "black")
        # The auto Endo is derived from the blood pool → HU change makes it stale:
        # drop it + hide Auto-Endo表示. The hand-edited Manual Endo is RETAINED.
        self._lv_endo_auto_model = None
        self._lv_endo_auto_surf = None
        self._lv_endo_auto_sig = None
        if getattr(self, "_lvv_endo_show", False):
            self._lvv_endo_show = False
            if getattr(self, "_lvv_auto_endo_btn", None) is not None:
                self._lvv_auto_endo_btn.setChecked(False)
                self._lvv_style_toggle(self._lvv_auto_endo_btn, "#ff8c28",
                                       "black")
        if getattr(self, "_lvv_thick_mode", None) is not None or getattr(
                self, "_lvv_thick_cache", None):
            self._lvv_thick_clear()        # Endo changed → wall thickness stale
        self._lvv_redraw()

    def _lvv_blood_vol_from_comp(self) -> bool:
        """(Re)build the 水色 display volume (numpy 0/1 float32, full grid) from the
        retained blood component (``_lvv_blood_comp`` + ``_lvv_blood_bbox``)
        WITHOUT re-running the blood-pool segmentation. Used on LV-Blood re-entry
        so a retained mask is reused rather than recomputed."""
        comp = getattr(self, "_lvv_blood_comp", None)
        bbox = getattr(self, "_lvv_blood_bbox", None)
        if comp is None or bbox is None or self._vol is None:
            return False
        full = np.zeros(self._vol.shape, np.float32)
        z0, z1, y0, y1, x0, x1 = bbox
        full[z0:z1, y0:y1, x0:x1][np.asarray(comp, bool)] = 1.0
        self._lvv_mask_vol = full
        return True

    def _lvv_toggle_blood(self, *args) -> None:
        """LV-Blood表示: show the computed, Epi-clipped blood region (水色). When
        turning ON, compute it for the current HU range if stale/absent; the
        全域HU tint is hidden while it is shown. Turning OFF restores the tint."""
        if not self._lvv_mask_btn.isChecked():
            self._lvv_mask_on = False
            self._lvv_style_toggle(self._lvv_mask_btn, "#40e0ff", "black")
            self._lvv_hl_on = True
            self._lvv_hl_btn.setChecked(True)
            self._lvv_style_toggle(self._lvv_hl_btn, "#40c0ff", "black")
            self._lvv_redraw()
            return
        # The display volume is dropped on leaving LV mode, but the segmented
        # component (_lvv_blood_comp) is retained — rebuild the volume cheaply
        # from it rather than re-running the pool segmentation.
        have = (self._lvv is not None and self._lvv.get("last_ml") is not None
                and self._lvv.get("calc_sig") is not None
                and self._lvv_signature() == self._lvv.get("calc_sig")
                and (self._lvv_mask_vol is not None
                     or getattr(self, "_lvv_blood_comp", None) is not None))
        if have:
            if self._lvv_mask_vol is None:
                self._lvv_blood_vol_from_comp()
            self._lvv_mask_on = True
            self._lvv_hl_on = False
            self._lvv_hl_btn.setChecked(False)
            self._lvv_style_toggle(self._lvv_hl_btn, "#40c0ff", "black")
            self._lvv_style_toggle(self._lvv_mask_btn, "#40e0ff", "black")
            self._lvv_redraw()
            return
        self._lvv_mask_btn.setChecked(False)     # recompute (re-checks on success)
        self._lvv_calc()

    def _lvv_toggle_epi(self, *args) -> None:
        self._lvv_epi_show = self._lvv_epi_btn.isChecked()
        self._lvv_style_toggle(self._lvv_epi_btn, "#50dc50", "black")
        for k in ("A", "B"):
            self._overlay[k].update()

    def _lvv_style_toggle(self, btn, color, text="white") -> None:
        """Colour a checkable overlay button by its checked state."""
        if btn.isChecked():
            btn.setStyleSheet("QPushButton{background:%s;color:%s;}%s"
                              % (color, text, self._BTN_DIS))
        else:
            btn.setStyleSheet(self._BTN_DIS)

    def _lvv_update_mask(self) -> None:
        """Rebuild the measured-region (red) overlay on both panes."""
        self._lvv_redraw()

    def _lvv_toggle_mask(self, *args) -> None:
        self._lvv_mask_on = self._lvv_mask_btn.isChecked()
        self._lvv_style_toggle(self._lvv_mask_btn, "#ff5a5a", "black")
        self._lvv_update_mask()

    # -------- Auto-Endo表示 (display) / Manual-Endo (edit) --------
    def _lvv_build_endo_model(self):
        """Build (but do NOT open) an Endo LVModel from the retained blood pool
        (compact-layer envelope, papillary filled; 13 handles/plane). Returns the
        built LVModel or None (with a message on failure). Mirrors the VTK viewer."""
        from PyQt6.QtCore import Qt, QThread
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog
        from multi_dicomviewer.core.lv_measure import LVModel
        from multi_dicomviewer.core.lv_compact import endo_contours_from_blood
        epi = getattr(self, "_lvv_epi_surf", None)
        apex = getattr(self, "_lvv_blood_apex", None)
        mv = self._lvv.get("mitral") if self._lvv is not None else None
        if (self._vol is None or self._lvv_blood_comp is None
                or self._lvv_blood_bbox is None or epi is None
                or apex is None or mv is None):
            QMessageBox.information(
                self.window(), t("LV"),
                t("自動Endo needs Blood, Epi, apex and the MV plane."))
            return None
        ax = epi.axis
        apex = np.asarray(apex, float)
        axis_dir = np.asarray(ax.axis, float)
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) or 1.0)
        radial0 = np.asarray(ax.radial0, float)
        n_planes = 6
        along_base = float((np.asarray(mv[0], float) - apex) @ axis_dir)
        if along_base <= 6.0:
            QMessageBox.information(
                self.window(), t("LV"),
                t("The MV plane is not basal to the apex — re-check apex / MV."))
            return None
        z0, z1, y0, y1, x0, x1 = self._lvv_blood_bbox
        blood = np.zeros(self._vol.shape, bool)
        blood[z0:z1, y0:y1, x0:x1] = self._lvv_blood_comp
        dims = self._dims
        close = float(getattr(self, "_lv_endo_close_mm", 5.0))  # 肉柱 bridging
        result: dict = {}

        class _EndoWorker(QThread):
            def run(self_) -> None:
                try:
                    result["prof"] = endo_contours_from_blood(
                        blood, dims, apex, axis_dir, radial0, 2 * n_planes,
                        along_apex=1.0, along_base=along_base - 0.5,
                        sax_step_mm=1.0, close_mm=close, half_mm=70.0, grid_mm=0.8)
                except Exception as exc:                  # noqa: BLE001
                    result["err"] = str(exc)

        dlg = QProgressDialog(t("Deriving Endo from blood…"), "", 0, 0,
                              self.window())
        dlg.setWindowTitle(t("Endo (Auto)"))
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        worker = _EndoWorker()
        worker.finished.connect(dlg.reset)
        worker.start()
        dlg.exec()
        worker.wait()
        worker.deleteLater()

        prof = result.get("prof")
        if result.get("err") or not prof or (isinstance(prof, dict)
                                             and prof.get("error")):
            QMessageBox.information(
                self.window(), t("LV"),
                t("Could not derive Endo from the blood pool: {e}").format(
                    e=result.get("err") or (prof or {}).get("error", "empty")))
            return None
        fracs = [0.125, 0.25, 0.5, 0.75, 0.875, 1.0]
        levels = [f * along_base for f in fracs]

        def _r_at(prof_theta, al):
            return float(np.interp(al, prof_theta[:, 0], prof_theta[:, 1]))

        model = LVModel(n_planes=n_planes)
        model.set_axis_from_frame(apex, axis_dir, radial0, which="endo")
        axm = model._axis_for("endo")
        built = 0
        for i in range(n_planes):
            phi = i * 180.0 / n_planes
            pos, neg = phi % 360.0, (phi + 180.0) % 360.0
            pp, pn = prof.get(pos), prof.get(neg)
            if pp is None or pn is None or len(pp) < 2 or len(pn) < 2:
                continue
            pts = [axm.to_world(neg, _r_at(pn, L), L) for L in reversed(levels)]
            pts.append(tuple(map(float, apex)))
            pts += [axm.to_world(pos, _r_at(pp, L), L) for L in levels]
            model.set_long_axis_contour(phi, np.asarray(pts, float), which="endo")
            built += 1
        if built < 3:
            QMessageBox.information(
                self.window(), t("LV"),
                t("Too few meridians recovered from the blood pool ({n}).").format(
                    n=built))
            return None
        model.set_apex_point("endo", apex)
        model.build()
        return model

    def _lvv_endo_stale(self) -> bool:
        if self._lv_endo_auto_model is None or self._lv_endo_auto_surf is None:
            return True
        sig = self._lvv_signature() if self._lvv is not None else None
        return sig is None or sig != self._lv_endo_auto_sig

    def _lvv_toggle_auto_endo(self, *args) -> None:
        """Auto-Endo表示: display-only overlay of the auto Endo border. Computes
        the blood pool INTERNALLY if needed (no change to the 水色 / 全域HU fill),
        builds the Endo when stale, then shows it."""
        from PyQt6.QtWidgets import QMessageBox
        if not self._lvv_auto_endo_btn.isChecked():
            self._lvv_endo_show = False
            self._lvv_style_toggle(self._lvv_auto_endo_btn, "#ff8c28", "black")
            self._lvv_redraw()
            return
        lvv = self._lvv
        av = lvv.get("aortic") if lvv else None
        mv = lvv.get("mitral") if lvv else None
        ready = (lvv is not None and lvv.get("apex") is not None
                 and av is not None and mv is not None
                 and self._lvv_epi_surf is not None)
        if not (ready or self._lvv_blood_comp is not None):
            self._lvv_auto_endo_btn.setChecked(False)
            self._lvv_style_toggle(self._lvv_auto_endo_btn, "#ff8c28", "black")
            QMessageBox.information(
                self.window(), t("Auto-Endo"),
                t("Set the apex, MV/AoV planes and an Epi surface first."))
            return
        sig = self._lvv_signature() if lvv is not None else None
        blood_fresh = (self._lvv_blood_comp is not None
                       and lvv.get("calc_sig") is not None
                       and lvv.get("calc_sig") == sig)
        if not blood_fresh:
            self._lvv_calc(then=self._lvv_finish_auto_endo, display=False)
            return
        self._lvv_finish_auto_endo()

    def _lvv_finish_auto_endo(self) -> None:
        if getattr(self, "_lvv_blood_comp", None) is None:
            self._lvv_auto_endo_btn.setChecked(False)
            self._lvv_style_toggle(self._lvv_auto_endo_btn, "#ff8c28", "black")
            return
        if self._lvv_endo_stale():
            model = self._lvv_build_endo_model()
            if model is None:
                self._lvv_auto_endo_btn.setChecked(False)
                self._lvv_style_toggle(self._lvv_auto_endo_btn, "#ff8c28", "black")
                return
            self._lv_endo_auto_model = model
            self._lv_endo_auto_surf = model.endo
            self._lv_endo_auto_sig = (self._lvv_signature()
                                      if self._lvv is not None else None)
        self._lvv_endo_show = True
        self._lvv_auto_endo_btn.setChecked(True)
        self._lvv_style_toggle(self._lvv_auto_endo_btn, "#ff8c28", "black")
        self._lvv_redraw()

    # -------- 壁厚 (wall thickness) heat maps ---------------------------------
    #: 壁厚 mode → (button attr, chip colour, label)
    _THICK_MODES = {"3d": ("_lvv_thick3d_btn", "#ffd040", "壁厚3D"),
                    "sax": ("_lvv_thicksax_btn", "#ff40d0", "壁厚短軸")}

    def _wall_thresholds(self) -> list:
        return list(getattr(self, "_lv_wall_thresholds", None)
                    or _WALL_DEFAULT_THR)

    def _lvv_thick_ready(self) -> bool:
        """True if 壁厚 can be computed: an Epi surface plus either an already-built
        Endo surface, a computed blood pool, OR the inputs to build one."""
        if self._lvv is None or self._lvv_epi_surf is None:
            return False
        if getattr(self, "_lv_endo_auto_surf", None) is not None:
            return True
        if getattr(self, "_lvv_blood_comp", None) is not None:
            return True
        apex_ok = self._lvv.get("apex") is not None
        valves_ok = (self._lv_valves_ready()
                     or self._lvv.get("aortic") is not None)
        return bool(apex_ok and valves_ok)

    def _lvv_thick_sync_buttons(self) -> None:
        """Reflect the 壁厚 mode on both buttons (only one can be on)."""
        mode = getattr(self, "_lvv_thick_mode", None)
        can = self._lvv_thick_ready()
        for m, (attr, col, _lbl) in self._THICK_MODES.items():
            btn = getattr(self, attr, None)
            if btn is None:
                continue
            btn.setChecked(mode == m)
            btn.setEnabled(can)
            self._lvv_style_toggle(btn, col, "black")

    def _lvv_thick_clear(self) -> None:
        """Drop the 壁厚 heat map + cache (used when the Endo becomes stale)."""
        self._lvv_thick_mode = None
        self._lvv_thick_vol = None
        self._lvv_thick_stats = None
        self._lvv_thick_cache = {}
        self._lvv_thick_refresh_display()
        self._lvv_thick_sync_buttons()

    def _lvv_thick_refresh_display(self) -> None:
        """Rebuild the 壁厚 heat-map overlays on both panes and repaint."""
        for k in ("A", "B"):
            self._lvv_refresh_overlays(k)
            self._overlay[k].update()

    def _lv_wall_bands_refresh(self) -> None:
        """Re-read the wall-thickness colour bands (Settings changed) and repaint
        any live 壁厚 heat map with them."""
        try:
            from multi_dicomviewer.core import settings as _st
            self._lv_wall_thresholds = list(
                _st.load_lv_wall_bands()["thresholds"])
        except Exception:                                # noqa: BLE001
            return
        self._lvv_thick_refresh_display()

    def _lvv_thick_vol_from_sub(self, sub, bbox):
        """Expand a cached wall-thickness sub-volume into a full-grid numpy field
        (mm); cheap, no EDT/radial recompute."""
        full = np.zeros(self._vol.shape, np.float32)
        z0, z1, y0, y1, x0, x1 = bbox
        full[z0:z1, y0:y1, x0:x1] = np.asarray(sub, np.float32)
        return full

    def _lvv_build_wall_thickness(self, mode: str = "3d"):
        """Build the myocardial wall-thickness field (mm) between the auto Endo
        surface and the Epi surface, valve-clipped. mode '3d' = Endo→Epi nearest
        distance (EDT); 'sax' = radial (short-axis) from the LV long axis. Returns
        (thick_sub[z,y,x] float32, epi_bbox, stats) or None; off-thread."""
        from PyQt6.QtCore import Qt, QThread
        from PyQt6.QtWidgets import QProgressDialog
        epi = getattr(self, "_lvv_epi_surf", None)
        endo = getattr(self, "_lv_endo_auto_surf", None)
        av = (self._lvv or {}).get("aortic")
        mv = (self._lvv or {}).get("mitral")
        apex = (self._lvv or {}).get("apex")
        if (self._vol is None or epi is None or endo is None or av is None
                or mv is None or apex is None):
            return None
        if mode == "sax" and getattr(epi, "axis", None) is None:
            return None
        dims = self._dims                       # (sx, sy, sz)
        shape = self._vol.shape                 # (nz, ny, nx)
        c_a, n_a = np.asarray(av[0], float), np.asarray(av[1], float)
        c_m, n_m = np.asarray(mv[0], float), np.asarray(mv[1], float)
        apex = np.asarray(apex, float)
        ax = getattr(epi, "axis", None)
        axis_dir = None if ax is None else np.asarray(ax.axis, float)
        radial0 = None if ax is None else np.asarray(ax.radial0, float)
        result: dict = {}

        class _ThickWorker(QThread):
            def run(self_) -> None:
                try:
                    from multi_dicomviewer.core.lv_wallthickness import (
                        wall_thickness_field, wall_thickness_radial_field)
                    planes = [(c_a, n_a), (c_m, n_m)]
                    epi_comp, epi_bb = epi.inside_mask_bbox(
                        dims, shape, planes, apex)
                    endo_comp, endo_bb = endo.inside_mask_bbox(
                        dims, shape, planes, apex)
                    if epi_comp is None or endo_comp is None:
                        return
                    ez0, ez1, ey0, ey1, ex0, ex1 = epi_bb
                    endo_in = np.zeros(epi_comp.shape, bool)
                    nz0, nz1, ny0, ny1, nx0, nx1 = endo_bb
                    z0, z1 = max(ez0, nz0), min(ez1, nz1)
                    y0, y1 = max(ey0, ny0), min(ey1, ny1)
                    x0, x1 = max(ex0, nx0), min(ex1, nx1)
                    if z1 > z0 and y1 > y0 and x1 > x0:
                        endo_in[z0 - ez0:z1 - ez0, y0 - ey0:y1 - ey0,
                                x0 - ex0:x1 - ex0] = endo_comp[
                            z0 - nz0:z1 - nz0, y0 - ny0:y1 - ny0,
                            x0 - nx0:x1 - nx0]
                    sp = (dims[2], dims[1], dims[0])   # (sz, sy, sx)
                    if mode == "sax":
                        origin = np.array([ex0 * dims[0], ey0 * dims[1],
                                           ez0 * dims[2]], float)
                        thick_sub, stats = wall_thickness_radial_field(
                            endo_in, epi_comp, sp, apex - origin,
                            axis_dir, radial0)
                    else:
                        thick_sub, stats = wall_thickness_field(
                            endo_in, epi_comp, sp)
                    result["sub"] = np.ascontiguousarray(thick_sub, np.float32)
                    result["bbox"] = epi_bb
                    result["stats"] = stats
                except Exception as exc:            # noqa: BLE001
                    result["err"] = str(exc)

        dlg = QProgressDialog(t("Measuring wall thickness…"), "", 0, 0,
                              self.window())
        dlg.setWindowTitle(t("壁厚"))
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        worker = _ThickWorker()
        worker.finished.connect(dlg.reset)
        worker.start()
        dlg.exec()
        worker.wait()
        worker.deleteLater()
        if result.get("err") or result.get("sub") is None:
            return None
        return result["sub"], result["bbox"], result["stats"]

    def _lvv_set_thick_mode(self, mode: str) -> None:
        """壁厚3D / 壁厚短軸: mutually-exclusive heat map. Clicking the active mode
        turns it OFF; clicking the other SWITCHES. Retained fields are reused so a
        re-press skips the EDT/radial recompute. Needs Blood/Endo + Epi."""
        from PyQt6.QtWidgets import QMessageBox
        cur = getattr(self, "_lvv_thick_mode", None)
        target = None if mode == cur else mode      # re-click active → off
        if target is not None:
            if not self._lvv_thick_ready():
                self._lvv_thick_sync_buttons()
                QMessageBox.information(
                    self.window(), t("壁厚"),
                    t("Set the apex, MV/AoV planes and an Epi surface first."))
                return
            # Build the Endo surface on demand (needs the blood pool).
            if (getattr(self, "_lv_endo_auto_surf", None) is None
                    or self._lvv_endo_stale()):
                if getattr(self, "_lvv_blood_comp", None) is None:
                    self._lvv_thick_sync_buttons()
                    QMessageBox.information(
                        self.window(), t("壁厚"),
                        t("Press LV-Blood表示 (or Auto-Endo表示) first to compute "
                          "the blood pool, then 壁厚."))
                    return
                model = self._lvv_build_endo_model()
                if model is None:
                    self._lvv_thick_sync_buttons()
                    return
                self._lv_endo_auto_model = model
                self._lv_endo_auto_surf = model.endo
                self._lv_endo_auto_sig = (self._lvv_signature()
                                          if self._lvv is not None else None)
            # Reuse a retained field if the Epi + Endo surfaces are unchanged.
            cache = getattr(self, "_lvv_thick_cache", {})
            epi = getattr(self, "_lvv_epi_surf", None)
            endo = getattr(self, "_lv_endo_auto_surf", None)
            hit = cache.get(target)
            if (hit is not None and hit.get("epi") is epi
                    and hit.get("endo") is endo):
                sub, bbox, stats = hit["sub"], hit["bbox"], hit["stats"]
            else:
                built = self._lvv_build_wall_thickness(target)
                if not built:
                    self._lvv_thick_sync_buttons()
                    QMessageBox.information(
                        self.window(), t("壁厚"),
                        t("Could not compute wall thickness — check "
                          "Endo/Epi/valves."))
                    return
                sub, bbox, stats = built
                cache[target] = {"sub": sub, "bbox": bbox, "stats": stats,
                                 "epi": epi, "endo": endo}
                self._lvv_thick_cache = cache
            self._lvv_thick_mode = target
            self._lvv_thick_vol = self._lvv_thick_vol_from_sub(sub, bbox)
            self._lvv_thick_stats = stats
            # Hide the blood/tint fills so the heat map reads clearly.
            self._lvv_mask_on = False
            self._lvv_hl_on = False
            self._lvv_mask_btn.setChecked(False)
            self._lvv_hl_btn.setChecked(False)
            self._lvv_style_toggle(self._lvv_mask_btn, "#40e0ff", "black")
            self._lvv_style_toggle(self._lvv_hl_btn, "#40c0ff", "black")
        else:
            self._lvv_thick_mode = None
            self._lvv_thick_vol = None
            self._lvv_thick_stats = None
        self._lv_update_text()
        self._lvv_thick_refresh_display()
        self._lvv_thick_sync_buttons()

    def _lvv_manual_endo(self, *args) -> None:
        """Manual-Endo: enter the Endo edit pass. Continues the RETAINED hand-
        edited Endo if one exists; else seeds from the auto Endo. Clear the manual
        Endo to re-seed from a fresh auto."""
        from PyQt6.QtWidgets import QMessageBox
        from multi_dicomviewer.core.lv_measure import LVModel
        if getattr(self, "_lv_endo_manual_dict", None) is not None:
            model = LVModel.from_dict(self._lv_endo_manual_dict)
            model.build()
        else:
            if getattr(self, "_lvv_blood_comp", None) is None:
                QMessageBox.information(
                    self.window(), t("Manual-Endo"),
                    t("Compute the LV blood first (press 'LV-Blood表示'), then "
                      "Manual-Endo."))
                return
            model = self._lvv_build_endo_model()
            if model is None:
                return
        self._lv_endo_manual_mode = True
        self._lv_apply_model(model)
        QMessageBox.information(
            self.window(), t("Manual-Endo"),
            t("Editing the Endo border. Changes are kept when the HU range "
              "changes; Clear to re-seed from Auto-Endo."))

    def _lv_stash_manual_endo(self, model) -> None:
        """Serialize a hand-edited Endo model so it is RETAINED across HU changes.
        A CLEARED Endo (<3 meridians) drops the retained manual (→ re-seed)."""
        try:
            if (model is not None
                    and getattr(model, "endo_axis", None) is not None
                    and len(model.endo_contours) >= 3):
                self._lv_endo_manual_dict = model.to_dict()
            else:
                self._lv_endo_manual_dict = None
        except Exception:                               # noqa: BLE001
            pass

    def _lvv_polygon_hu(self, m, which):
        """Sample HU (from self._vol) at grid points inside polygon *m* drawn on
        pane *which*. Returns a 1-D array of HU (empty if none)."""
        import cv2
        pts = np.asarray(m["pts"], float)
        if len(pts) < 3:
            return np.array([])
        ox0, oy0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        ox1, oy1 = float(pts[:, 0].max()), float(pts[:, 1].max())
        step = 0.6
        gw = max(2, int((ox1 - ox0) / step) + 1)
        gh = max(2, int((oy1 - oy0) / step) + 1)
        poly = np.round((pts - [ox0, oy0]) / step).astype(np.int32)
        mask = np.zeros((gh, gw), np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        ys, xs = np.where(mask > 0)
        hus = []
        for gx, gy in zip(xs, ys):
            P = self._out_to_world3d(which, ox0 + gx * step, oy0 + gy * step)
            hus.append(self._lvv_hu_at(P))
        return np.asarray(hus, float)

    def _lvv_capture_roi(self) -> None:
        """内腔ROI button → capture the latest Polygon as the LV cavity ROI:
        colour it, seed connectivity at its centroid, and suggest the blood HU
        range (下限/上限) from the pixels inside it (then user-adjustable)."""
        from PyQt6.QtWidgets import QMessageBox
        try:
            lvv = self._lvv
            if lvv is None:
                return
            m, which, best = None, None, -1
            for k in ("A", "B"):
                for cand in self._measures.get(k, []):
                    if (cand.get("type") == "polygon"
                            and cand.get("id", -1) > best):
                        best = cand.get("id", -1)
                        m, which = cand, k
            if m is None:
                QMessageBox.information(
                    self.window(), t("LV Vol"),
                    t("Draw a Polygon inside the LV cavity first "
                      "(Measure→Polygon)."))
                return
            cx, cy = self._shape_center(m)
            seed = np.asarray(self._out_to_world3d(which, cx, cy), float)
            hus = self._lvv_polygon_hu(m, which)
            if hus.size < 4:
                QMessageBox.information(self.window(), t("LV Vol"),
                                       t("ROI too small — draw a larger Polygon."))
                return
            lo = float(np.percentile(hus, 2))
            hi = float(np.percentile(hus, 99))
            lvv["seed"] = seed
            lvv["hu_lo"] = lo
            lvv["hu_hi"] = hi
            m["color"] = "#40c0ff"                       # ROI tint
            m["_lvv"] = "roi"
            self._lvv_lo_spin.blockSignals(True)
            self._lvv_lo_spin.setValue(int(round(lo)))
            self._lvv_lo_spin.blockSignals(False)
            self._lvv_hi_spin.blockSignals(True)
            self._lvv_hi_spin.setValue(int(round(hi)))
            self._lvv_hi_spin.blockSignals(False)
            self._redraw_meas(which)
            if self._meas_on:
                self._meas_btn.setChecked(False)
                self._toggle_measure()
            lvv["step"] = "ready"
            self._lvv_sync()
            self._lvv_update_highlight()                 # in-range voxel tint
            self._lvv_prompt(
                t("ROI captured. Blood HU range set to {lo:.0f}–{hi:.0f} from "
                  "the ROI (min {mn:.0f} / max {mx:.0f}). Adjust 下限/上限 if "
                  "needed, then press CalcVol.").format(
                    lo=lo, hi=hi, mn=float(hus.min()), mx=float(hus.max())))
        except Exception as exc:                        # noqa: BLE001
            import traceback
            QMessageBox.critical(self.window(), t("LV Vol (ROI error)"),
                                 traceback.format_exc() or repr(exc))

    def _lvv_capture_valve(self, valve) -> None:
        """Capture the most-recent Ellipse on either pane as this valve's plane
        (centre + the pane's current normal)."""
        from PyQt6.QtWidgets import QMessageBox
        if self._lvv is None:
            return
        # Newest Ellipse across BOTH panes (drawn on either side).
        m, which = None, None
        best = -1
        for k in ("A", "B"):
            for cand in self._measures.get(k, []):
                if cand.get("type") == "ellipse" and cand.get("id", -1) > best:
                    best = cand.get("id", -1)
                    m, which = cand, k
        if m is None:
            QMessageBox.information(
                self.window(), t("LV Vol"),
                t("Draw an Ellipse on the {v} annulus first (Measure→Ellipse).")
                .format(v=t("aortic") if valve == "aortic" else t("mitral")))
            return
        cx, cy = self._shape_center(m)
        center = np.asarray(self._out_to_world3d(which, cx, cy), float)
        _u, _v, n = self._axes_for(which)
        ecx, ecy, ea, eb = self._ellipse_cab(m)          # semi-axes (mm)
        radius = float(max(ea, eb))
        self._lvv[valve] = (center, np.asarray(n, float), radius)
        m["color"] = "#ffd24d" if valve == "aortic" else "#4dd0ff"
        m["_lvv"] = valve
        self._redraw_meas(which)
        # Turn Measure OFF so the user can navigate to the next landmark without
        # a click starting a new ellipse (they re-arm Measure→Ellipse next).
        if self._meas_on:
            self._meas_btn.setChecked(False)
            self._toggle_measure()
        if valve == "mitral":
            self._lvv["step"] = "aov"
            self._lvv_sync()
            self._lvv_prompt(
                t("Mitral plane captured. Now identify the aortic valve: draw "
                  "an Ellipse on the AoV annulus, then press 'AoV plane'."))
        else:
            self._lvv["step"] = "thr"
            self._lvv_sync()
            self._lvv_prompt(
                t("Aortic plane captured. Draw a Polygon inside the LV cavity "
                  "(Measure→Polygon), then press the 内腔ROI button."))

    def _lvv_calc(self, then=None, display=True) -> None:
        from PyQt6.QtWidgets import QMessageBox
        try:
            lvv = self._lvv
            if lvv is None or self._vol is None:
                return
            self._lvv_calc_display = display   # finish flips the 水色 fill or not
            # Already measured AND the inputs are unchanged since that measure
            # (region + HU range signature matches)? Then this button does NOT
            # recompute — it just toggles the red measured-region overlay on/off
            # (the data is kept). If the region or HU range changed, fall through
            # and recompute. The Save path (then) saves directly when unchanged.
            have = (lvv.get("last_ml") is not None
                    and self._lvv_mask_vol is not None)
            sig = lvv.get("calc_sig")
            if have and sig is not None and self._lvv_signature() == sig:
                if then is not None:
                    then()
                    return
                self._lvv_mask_on = True
                self._lvv_hl_on = False
                self._lvv_mask_btn.setChecked(True)
                self._lvv_hl_btn.setChecked(False)
                self._lvv_redraw()
                self._lvv_sync()
                return
            miss = [n for n, k in ((t("apex"), "apex"), (t("aortic plane"),
                    "aortic"), (t("mitral plane"), "mitral"))
                    if lvv.get(k) is None]
            if miss:
                QMessageBox.information(
                    self.window(), t("LV Vol"),
                    t("Set these first: {m}").format(m=", ".join(miss)))
                return
            if self._lvv_epi_surf is None:
                QMessageBox.information(self.window(), t("LV Vol"),
                                       t("No Epi surface — trace Epi first."))
                return
            from multi_dicomviewer.core.lv_bloodpool import bloodpool_volume_epi
            from PyQt6.QtWidgets import QProgressDialog
            seed = None                          # auto: largest in-range component
            hu_lo = float(self._lvv_lo_spin.value())
            hu_hi = float(self._lvv_hi_spin.value())
            c_a, n_a, _r_a = lvv["aortic"]
            c_m, n_m, _r_m = lvv["mitral"]
            epi = self._lvv_epi_surf
            apex = tuple(lvv["apex"])
            vol, dims = self._vol, self._dims

            def _job():
                return bloodpool_volume_epi(
                    vol, dims, epi._all_ring_points(),
                    lambda p: epi.contains(p, extend_base=True),
                    [(c_a, n_a), (c_m, n_m)], apex, hu_lo, hu_hi, seed)

            # Busy progress bar while the (few-second) volume runs off-thread.
            dlg = QProgressDialog(t("Measuring LV volume…"), None, 0, 0,
                                  self.window())
            dlg.setWindowTitle(t("LV Vol"))
            dlg.setWindowModality(Qt.WindowModality.WindowModal)
            dlg.setCancelButton(None)
            dlg.setMinimumDuration(0)
            dlg.setAutoClose(False)
            dlg.setAutoReset(False)
            self._lvv_calc_btn.setEnabled(False)
            self._lvv_calc_then = then           # run after a successful measure
            dlg.show()
            worker = _LvvWorker(_job, self)
            self._lvv_worker = worker            # keep a ref so it isn't GC'd
            worker.finished_result.connect(
                lambda res: self._lvv_calc_finish(res, dlg))
            worker.start()
        except Exception as exc:                        # noqa: BLE001
            import traceback
            QMessageBox.critical(self.window(), t("LV Vol (error)"),
                                 traceback.format_exc() or repr(exc))

    def _lvv_calc_finish(self, res, dlg) -> None:
        from PyQt6.QtWidgets import QMessageBox
        try:
            dlg.close()
            self._lvv_calc_btn.setEnabled(True)
            lvv = self._lvv
            if lvv is None:
                return
            if res is None:
                QMessageBox.information(
                    self.window(), t("LV Vol"),
                    t("No cavity found — check the HU range and the ROI."))
                return
            if res.get("error") == "exc":
                QMessageBox.critical(self.window(), t("LV Vol (error)"),
                                     res.get("msg", "error"))
                return
            if res.get("error") == "seed_out":
                QMessageBox.information(
                    self.window(), t("LV Vol"),
                    t("Could not measure — {m}.\nAdjust 下限/上限, move the ROI "
                      "to a clearly-contrast part of the cavity, or re-check "
                      "the Epi / MV / AoV.").format(m=res["msg"]))
                return
            if res.get("error") == "too_large":
                QMessageBox.warning(
                    self.window(), t("LV Vol"),
                    t("Region too large to compute ({v:,} voxels) — re-check "
                      "the Epi surface.").format(v=res["voxels"]))
                return
            lvv["last_ml"] = res["volume_ml"]
            lvv["calc_sig"] = self._lvv_signature()      # inputs at this measure
            self._lvv_vol_lbl.setText(t("{v:.1f} mL").format(v=res["volume_ml"]))
            # Build the LV-Blood region mask volume (0/1 float32) for the 水色
            # overlay — a plain numpy array sampled by _lvv_plane_rgba (no VTK).
            comp = res.get("comp")
            if comp is not None:
                z0, z1, y0, y1, x0, x1 = res["bbox"]
                full = np.zeros(self._vol.shape, np.float32)
                full[z0:z1, y0:y1, x0:x1][np.asarray(comp, bool)] = 1.0
                self._lvv_mask_vol = full
                self._lvv_blood_comp = np.asarray(comp, bool)
                self._lvv_blood_bbox = tuple(res["bbox"])
                if lvv.get("apex") is not None:
                    self._lvv_blood_apex = np.asarray(lvv["apex"], float)
                if getattr(self, "_lvv_calc_display", True):
                    # Show 水色, hide the 全域HU tint (they are exclusive).
                    self._lvv_mask_on = True
                    self._lvv_mask_btn.setChecked(True)
                    self._lvv_hl_on = False
                    self._lvv_hl_btn.setChecked(False)
                    self._lvv_style_toggle(self._lvv_hl_btn, "#40c0ff", "black")
                    self._lvv_redraw()                   # build + show 水色 now
            self._lvv_style_toggle(self._lvv_mask_btn, "#40e0ff", "black")
            self._lvv_sync()
            then = getattr(self, "_lvv_calc_then", None)
            self._lvv_calc_then = None
            if then is not None:
                then()                               # e.g. continue to Save
        except Exception as exc:                        # noqa: BLE001
            import traceback
            QMessageBox.critical(self.window(), t("LV Vol (error)"),
                                 traceback.format_exc() or repr(exc))

    def _lvv_save(self) -> None:
        """Save the LV Vol dataset. If no volume yet, offer to save without it,
        compute-then-save, or cancel (mirrors the contour LV Save)."""
        from PyQt6.QtWidgets import QMessageBox
        lvv = self._lvv
        if lvv is None or lvv.get("apex") is None:
            QMessageBox.information(self.window(), t("LV Vol"),
                                   t("Set the apex first."))
            return
        if lvv.get("last_ml") is None:
            box = QMessageBox(self.window())
            box.setWindowTitle(t("LV Vol"))
            box.setIcon(QMessageBox.Icon.Question)
            box.setText(t("Save without volume data?"))
            b_no = box.addButton(t("Save without volume"),
                                 QMessageBox.ButtonRole.AcceptRole)
            b_yes = box.addButton(t("Calculate volume, then save"),
                                  QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is b_yes:
                self._lvv_calc(then=self._lvv_do_save)   # compute → then save
            elif clicked is b_no:
                self._lvv_do_save()
            return
        self._lvv_do_save()

    def _lvv_do_save(self) -> None:
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import json
        import os
        lvv = self._lvv
        if lvv is None or lvv.get("apex") is None:
            return
        c_a, n_a, r_a = lvv["aortic"]
        c_m, n_m, r_m = lvv["mitral"]
        data = {
            "type": "lvvol",
            "series": (self._lv_series_meta()
                       if hasattr(self, "_lv_series_meta") else {}),
            "apex": list(map(float, lvv["apex"])),
            "aortic": {"c": list(map(float, c_a)), "n": list(map(float, n_a)),
                       "r": float(r_a)},
            "mitral": {"c": list(map(float, c_m)), "n": list(map(float, n_m)),
                       "r": float(r_m)},
            "hu_lo": float(self._lvv_lo_spin.value()),
            "hu_hi": float(self._lvv_hi_spin.value()),
            "volume_ml": (None if lvv.get("last_ml") is None
                          else float(lvv["last_ml"])),
            "epi_model": self._lvv_epi_model_dict,
        }
        d = self._lv_save_dir() if hasattr(self, "_lv_save_dir") else ""
        # Same auto name as the Epi .lv.json — "名前;日付_Se番号.lvvol.json".
        stem = (self._lv_default_stem() if hasattr(self, "_lv_default_stem")
                else "lvvol")
        name = stem + ".lvvol.json"
        default = os.path.join(d, name) if d else name
        path, _ = QFileDialog.getSaveFileName(
            self.window(), t("Save LV Vol"), default,
            "LV Vol (*.lvvol.json);;JSON (*.json)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".lvvol.json"
        self._lv_stamp_axis_def(data)        # apex→MV-centre long-axis marker
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as exc:                        # noqa: BLE001
            QMessageBox.warning(self.window(), t("LV Vol"),
                                t("Save failed: {err}", err=str(exc)))
            return
        QMessageBox.information(self.window(), t("LV Vol"),
                               t("Saved: {p}", p=os.path.basename(path)))

    def _lvv_load(self) -> None:
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from multi_dicomviewer.core.lv_measure import LVModel
        import json
        if self._vol is None:
            return
        d = self._lv_save_dir() if hasattr(self, "_lv_save_dir") else ""
        path, _ = QFileDialog.getOpenFileName(
            self.window(), t("Load LV Vol"), d,
            "LV Vol (*.lvvol.json);;JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._lv_warn_if_axis_stale(data)
            # (B) Warn on a series mismatch — the apex/seed/valve/Epi are all in
            # THIS series' volume mm, so a file saved for another series won't
            # line up. Let the user override (same as the contour LV load).
            saved = (data.get("series") or {}).get("series_uid", "")
            cur = (self._lv_series_meta().get("series_uid", "")
                   if hasattr(self, "_lv_series_meta") else "")
            if saved and cur and saved != cur:
                if QMessageBox.question(
                        self.window(), t("LV Vol"),
                        t("This file was saved for a DIFFERENT series — the "
                          "landmarks and Epi may not line up. Load anyway?")) \
                        != QMessageBox.StandardButton.Yes:
                    return
            # (C) Contour LV and LV Vol are mutually exclusive: leave contour LV
            # first so both modes can't be active at once (this used to leave
            # self._lv AND self._lvv set). _lv_exit stashes the contour Epi, but
            # we overwrite _lvv_epi_surf with THIS file's Epi just below, so the
            # loaded file stays the single Epi source — (A).
            if self._lv is not None:
                self._lv_exit()
            # Drop any prior LV Vol overlay/markers so a reload doesn't stack.
            if self._lvv is not None:
                self._lvv_clear_markers()
            model = LVModel.from_dict(data["epi_model"])
            model.build()
            if model.epi is None:
                raise ValueError("no Epi surface in file")
            self._lvv_epi_surf = model.epi
            self._lvv_epi_apex = np.asarray(model.epi_axis.apex, float)
            self._lvv_epi_model_dict = data["epi_model"]
            self._lvv_epi_ml = None          # re-derive from the loaded dict
            if self._lvv is None:
                self._lvv = {"apex": None, "aortic": None, "mitral": None,
                             "hu_lo": None, "hu_hi": None, "seed": None,
                             "step": "apex", "last_ml": None, "calc_sig": None}
            lvv = self._lvv
            lvv["apex"] = np.asarray(data["apex"], float)
            if data.get("seed") is not None:         # legacy files only
                lvv["seed"] = np.asarray(data["seed"], float)
            a, m = data["aortic"], data["mitral"]
            lvv["aortic"] = (np.asarray(a["c"], float), np.asarray(a["n"], float),
                             float(a.get("r", 20.0)))
            lvv["mitral"] = (np.asarray(m["c"], float), np.asarray(m["n"], float),
                             float(m.get("r", 20.0)))
            lvv["hu_lo"] = float(data["hu_lo"])
            lvv["hu_hi"] = float(data["hu_hi"])
            lvv["last_ml"] = data.get("volume_ml")
            lvv["step"] = "ready"
            for spin, vv in ((self._lvv_lo_spin, data["hu_lo"]),
                             (self._lvv_hi_spin, data["hu_hi"])):
                spin.blockSignals(True)
                spin.setValue(int(round(float(vv))))
                spin.blockSignals(False)
            if lvv.get("last_ml") is not None:
                self._lvv_vol_lbl.setText(
                    t("{v:.1f} mL").format(v=float(lvv["last_ml"])))
            self._lvv_sync()
            self._lvv_update_highlight()
            for k in ("A", "B"):
                self._overlay[k].update()
            # (A) Make the Epi source explicit, and (B) remind about the two-file
            # drift: the Epi lives in BOTH the .lv and .lvvol files, so an edit
            # in one must be re-saved to the other to stay in sync.
            QMessageBox.information(
                self.window(), t("LV Vol"),
                t("Loaded — the volume is measured against the Epi border stored "
                  "in THIS .lvvol file. If you later edit the Epi in contour LV "
                  "and re-save the .lv file, re-save the .lvvol too so they stay "
                  "in sync. Press LV Vol計測 to (re)compute the volume."))
        except Exception as exc:                        # noqa: BLE001
            import traceback
            QMessageBox.critical(self.window(), t("LV Vol (load error)"),
                                 traceback.format_exc() or repr(exc))

    def _lvv_clear_markers(self) -> None:
        # NOTE: the retained blood/Endo/Epi surfaces and the 壁厚 cache survive
        # leaving the mode — only the on-screen overlays are dropped, so a
        # re-entry rebuilds the display cheaply (no recompute).
        self._lvv_mask_vol = None
        self._lvv_mask_on = False
        self._lvv_epi_show = False
        self._lvv_endo_show = False        # hide Auto-Endo overlay (model kept)
        self._lvv_thick_mode = None
        self._lvv_thick_vol = None
        self._lvv_thick_stats = None
        for k in ("A", "B"):
            self._measures[k] = [m for m in self._measures.get(k, [])
                                 if m.get("_lvv") is None]
            self._lvv_cyan_img[k] = None
            self._lvv_red_img[k] = None
            self._lvv_thick_img[k] = None
            self._redraw_meas(k)

    def _lvv_deactivate(self) -> None:
        """Fully leave LV Vol mode: clear its overlays (Epi dots / blood tint /
        red region) and drop the mode, KEEPING the stashed Epi surface. Called
        when entering contour LV so a loaded .lvvol dataset's Epi border can't
        linger on-screen there — its display belongs to the LV Vol Epi button."""
        if self._lvv is None:
            return
        self._lvv_clear_markers()
        self._lvv = None
        self._lvv_sync()
        for k in ("A", "B"):
            self._overlay[k].update()

    def _lvv_reset_state(self) -> None:
        """Drop ALL LV Vol state (mode + stashed Epi surface) — used when a new
        series is loaded or the viewer is cleared, since every landmark and the
        mask are in the old series' voxel coordinates."""
        if getattr(self, "_lvv", None) is not None:
            self._lvv_clear_markers()
        self._lvv = None
        self._lvv_epi_surf = None
        self._lvv_epi_apex = None
        self._lvv_epi_model_dict = None
        self._lvv_epi_ml = None
        self._lvv_mask_vol = None
        self._lvv_mask_on = False
        self._lvv_epi_show = False
        self._lvv_blood_comp = None
        self._lvv_blood_bbox = None
        self._lvv_blood_apex = None
        self._lv_endo_auto_model = None
        self._lv_endo_auto_surf = None
        self._lv_endo_auto_sig = None
        self._lvv_endo_show = False
        self._lv_endo_manual_dict = None
        self._lvv_thick_mode = None
        self._lvv_thick_vol = None
        self._lvv_thick_stats = None
        self._lvv_thick_cache = {}
        for k in ("A", "B"):
            self._lvv_cyan_img[k] = None
            self._lvv_red_img[k] = None
            self._lvv_thick_img[k] = None
        if getattr(self, "_lvv_start_btn", None) is not None:
            self._lvv_vol_lbl.setText("--")
            self._lvv_sync()

    def _lv_sync_buttons(self) -> None:
        """Colour + enable the LV bar by the current pass/phase (mirror the VTK
        viewer). Only the active pass's Endo/Epi is coloured; Set axis is
        dark-yellow while its axis is in use (ready/apex/contour); Trace is red
        while tracing (apex/contour). LIFO enable: only the last-turned-on of
        {Set axis, Trace, SAX} can be turned off."""
        lv = self._lv
        endo_btn, epi_btn = self._lv_endo_btn, self._lv_epi_btn
        setax, trace = self._lv_setaxis_btn, self._lv_trace_btn
        apexb = getattr(self, "_lv_apex_btn", None)
        endo_btn.setEnabled(True)
        epi_btn.setEnabled(True)
        self._lv_load_btn.setEnabled(True)
        if lv is None:                                # not in LV mode
            for b in (endo_btn, epi_btn, setax, trace):
                b.setStyleSheet("")
            if apexb is not None:
                apexb.setStyleSheet("")
            self._lv_vol_btn.setStyleSheet(self._LV_STY["vol_todo"])
            self._lv_set_bar_enabled(False)
            self._lv_exit_btn.setEnabled(False)
            self._refresh_tool_availability()        # restore WB reverse / tools
            return
        ph = lv.get("phase")
        pas = lv.get("pass")
        sax_on = lv.get("sax") is not None
        if sax_on:
            # SAX / refine mode: neutral bar. Endo/Epi colour only to show which
            # border is armed for editing (lv["sax_edit"]); Set axis stays off.
            # Trace is enabled once a border is armed → Endo/Epi + Trace LEAVES
            # SAX into that pass's trace (Endo restores its original trace).
            ed = lv.get("sax_edit")
            endo_btn.setStyleSheet(self._LV_STY["endo"] if ed == "endo"
                                   else self._LV_STY["neutral"])
            epi_btn.setStyleSheet(self._LV_STY["epi"] if ed == "epi"
                                  else self._LV_STY["neutral"])
            setax.setStyleSheet(self._LV_STY["neutral"])
            setax.setEnabled(False)
            trace.setStyleSheet(self._LV_STY["neutral"])
            trace.setEnabled(ed in ("endo", "epi"))
            if apexb is not None:                     # apex is set pre-SAX only
                apexb.setStyleSheet(self._LV_STY["neutral"])
                apexb.setEnabled(False)
        else:
            endo_btn.setStyleSheet(self._LV_STY["endo"] if pas == "endo" else "")
            epi_btn.setStyleSheet(self._LV_STY["epi"] if pas == "epi" else "")
            setax.setStyleSheet(self._LV_STY["setaxis"]
                                if ph in ("ready", "apex", "contour") else "")
            # Trace red while tracing; neutral once toggled to VIEW (trace_view).
            _tracing = (ph == "apex"
                        or (ph == "contour" and not lv.get("trace_view")))
            trace.setStyleSheet(self._LV_STY["trace"] if _tracing else "")
            # LIFO enable: you can only turn OFF the LAST button turned on.
            setax.setEnabled(ph in ("align", "ready"))
            trace.setEnabled(ph in ("ready", "apex", "contour"))
            # Apex: coloured (pass colour) once this pass's apex is set — it never
            # toggles off. Settable in align/ready only, once a pass is chosen.
            if apexb is not None:
                has_pass = pas in ("endo", "epi")
                mdl = lv["model"]
                apex_set = ((mdl.endo_apex if pas == "endo"
                             else mdl.epi_apex if pas == "epi" else None)
                            is not None)
                if apex_set:
                    apexb.setStyleSheet(self._LV_STY["endo"] if pas == "endo"
                                        else self._LV_STY["epi"])
                else:
                    apexb.setStyleSheet("")
                apexb.setEnabled(has_pass and ph in ("align", "ready"))
        self._lv_sax_btn.setEnabled(ph == "contour")
        contour = ph == "contour"
        for b in (self._lv_prev_btn, self._lv_next_btn, self._lv_vol_btn,
                  self._lv_wall_btn, self._lv_redo_btn, self._lv_save_btn,
                  self._lv_stl_btn):
            b.setEnabled(contour)
        # CalcVol: blue once a volume has been computed for the CURRENT trace,
        # grey again after any edit (result stale). See lv["vol_done"].
        self._lv_vol_btn.setStyleSheet(
            self._LV_STY["vol_done"] if lv.get("vol_done")
            else self._LV_STY["vol_todo"])
        self._lv_exit_btn.setEnabled(True)
        self._refresh_tool_availability()   # grey Rotate/Spin/Thick + WB reverse

    def _toggle_lv(self) -> None:
        """Legacy toggle (kept for callers): enter LV mode on the Endo pass, or
        leave it."""
        if self._lv is None:
            if self._lv_enter_mode():
                self._lv_select_pass("endo")
        else:
            self._lv_exit(from_toggle=True)

    def _lv_enter_mode(self) -> bool:
        """Create the LV model + state (no axis yet — set per pass). Returns False
        if there's no volume. LV always works on the RIGHT pane (B) long axis; the
        LEFT (A) becomes the short-axis side."""
        if self._vol is None:
            return False
        from multi_dicomviewer.core.lv_measure import LVModel
        if self._meas_on:
            self._meas_btn.setChecked(False)
            self._toggle_measure()
        self._lvv_deactivate()            # contour LV and LV Vol are exclusive
        self._lv = {"model": LVModel(n_planes=6), "phase": "align",
                    "plane_idx": 0, "target": None, "pane": "B",
                    "sax": None, "pass": None,
                    "prev_side": self.current_side()}
        self._lv_reset_undo()                        # fresh Ctrl+Z stack
        self._lv_btn.setChecked(True)               # internal mode flag
        self.set_side("Bi")
        return True

    def _lv_thick_trace_both(self) -> None:
        """Slab thickness for the LV panes on Endo/Epi entry: BOTH panes 0 mm —
        the Endo and Epi borders are both traced/judged on THIN cross-sections
        (no MIP slab), matching the Blood/Endo sub-mode."""
        self._thick["A"] = 0.0
        self._thick["B"] = 0.0
        if hasattr(self, "_sync_slab_spin"):
            self._sync_slab_spin()

    # ---- pass flow: align the view → Set axis → place apex → trace ----------
    def _lv_axis_from_view(self):
        """(origin, axis_dir, radial0) of the CURRENT long-axis view on the trace
        pane — the rotation axis = the no-arrow centreline, θ=0 = the green-▲
        radial."""
        key0 = self._lv["pane"]
        u, v, _n = self._frame[key0]
        a = math.radians(self._cross_ang[key0])
        axis_dir = -math.sin(a) * u + math.cos(a) * v
        radial0 = math.cos(a) * u + math.sin(a) * v
        origin = np.asarray(self._pc[key0], dtype=float).copy()
        return origin, axis_dir, radial0

    def _lv_enter_align(self) -> None:
        """ALIGN sub-phase for the active pass: free long-axis MPR so the user
        orients the view; Trace then captures it. Trace/analysis controls are
        disabled until the axis is set."""
        lv = self._lv
        self._lv_region_reset()
        lv["phase"] = "align"
        lv["target"] = None
        lv["keep_view"] = False                      # fresh alignment → normal fit
        self.set_side("Bi")
        self._lv_thick_trace_both()                  # slab 5mm both panes
        # Hide the other pass's border while aligning this one.
        for mm in self._measures[lv["pane"]]:
            tag = mm.get("_lv")
            if tag is not None:
                mm["hidden"] = (tag[1] != lv.get("pass"))
        if self._meas_on:
            self._meas_btn.setChecked(False)
            self._toggle_measure()
        self._lv_sync_buttons()
        self._lv_update_text()
        self._view_initial = False
        self._refresh()
        self._lv_redraw_all()

    def _lv_set_axis(self) -> None:
        """'Set axis' button. ALIGN → READY: capture the current view as this
        pass's long axis. READY → ALIGN (press again): UNDO — back to the
        Endo/Epi-only state with the IMAGE KEPT, unlocking Spin/Rotate/Thick."""
        lv = self._lv
        if lv is None or lv.get("pass") is None:
            return
        ph = lv.get("phase")
        if ph == "align":
            which = lv["pass"]
            origin, axis_dir, radial0 = self._lv_axis_from_view()
            lv["model"].set_axis_from_frame(origin, axis_dir, radial0,
                                            which=which)
            # Snap the global CrossLine centre onto the axis origin so the
            # crosshair COINCIDES with the LV long axis. Otherwise the crosshair
            # stays at the stale _center (off the axis after aligning) while the
            # apex is constrained to the axis (output x=0) — placing the apex
            # would then land visibly offset from the crosshair (the reported
            # "心尖部点がズレる"). With this, clicking on the crosshair puts the
            # apex under the cursor.
            self._center = np.asarray(origin, dtype=float).copy()
            lv["phase"] = "ready"
            lv["plane_idx"] = 0
            self._lv_sync_buttons()
            self._lv_show_plane()                    # right pane: axis vertical
        elif ph == "ready":
            self._lv_enter_align()                   # undo (keeps the image)

    def _lv_start_trace(self) -> None:
        """'Trace' button. READY → tracing: place this pass's apex (first plain
        click; Shift-click adjusts the view) then trace — or resume if the apex
        is already set. APEX/CONTOUR (press again, SAX off) → UNDO back to READY."""
        lv = self._lv
        if lv is None:
            return
        # In SAX: Trace acts on the ARMED border (Endo/Epi button) — LEAVE SAX
        # into that pass's long-axis trace. For Endo, restore its ORIGINAL
        # independent-axis trace (undo the Epi-axis promotion, non-destructive).
        if lv.get("sax") is not None:
            ed = lv.get("sax_edit")
            if ed not in ("endo", "epi"):
                return
            self._lv_leave_sax()
            if ed == "endo" and lv["model"].endo_promoted:
                lv["model"].restore_endo_original()
                self._lv_rebuild_measures()
            lv["sax_edit"] = None
            self._lv_select_pass(ed)             # resume that pass's trace
            return
        ph = lv.get("phase")
        if ph == "align":
            # The LV long axis is now DETERMINISTIC (apex → MV centre), set by the
            # Apex button — not captured from the view. Point the user there.
            self._lvv_prompt(t(
                "Set the apex first: move the centreline crossing onto the "
                "apex, then press 'Apex' (the LV long axis = apex→MV centre)."))
            return
        if ph == "ready":
            m = lv["model"]
            apex = m.endo_apex if lv["pass"] == "endo" else m.epi_apex
            if apex is not None:                     # apex already set → resume
                lv["apex_target"] = None
                self._lv_enter_contour()
                return
            # No apex yet → the apex is set with the Apex button now, not by
            # clicking the image. Point the user there.
            self._lvv_prompt(t(
                "Set the apex first: move the centreline crossing onto the "
                "apex, then press 'Apex'."))
        elif ph == "contour" and lv.get("sax") is None and (
                lv["model"].endo_planes or lv["model"].epi_planes):
            # Border traced → Trace toggles TRACE ⇄ VIEW instead of undoing: VIEW
            # frees Rotate/Spin/Paging/CenterLine to inspect the result in 3-D.
            lv["trace_view"] = not lv.get("trace_view", False)
            if lv["trace_view"] and self._meas_on:
                self._meas_btn.setChecked(False)
                self._toggle_measure()
            self._lv_sync_buttons()
            self._lv_apply_view_free()
        elif ph in ("apex", "contour") and lv.get("sax") is None:
            lv["phase"] = "ready"                    # UNDO trace → ready
            lv["apex_target"] = None
            # Clear this pass's placed apex so its marker disappears and it can be
            # re-placed (re-Trace).
            lv["model"].set_apex_point(lv["pass"], None)
            self._lv_result_lines = []
            if self._meas_on:
                self._meas_btn.setChecked(False)
                self._toggle_measure()
            self._lv_sync_buttons()
            self._lv_update_text()
            self._lv_show_plane()

    def _lv_select_pass(self, which: str) -> None:
        """Endo / Epi button = choose that analysis pass (the sole active one).
        In SAX it ISOLATES that pass (long + short show only it, on its own axis)
        for editing; otherwise it enters/resumes the long-axis trace on its axis.
        Unset axis → align it first."""
        if self._lv is None:
            if not self._lv_enter_mode():
                return
        lv = self._lv
        m = lv["model"]
        if lv.get("sax") is not None:               # in SAX → ARM this border
            # After promotion both borders live on the Epi axis and are shown on
            # the long-axis plane; the Endo/Epi button picks WHICH border's
            # points are editable (the other is display-only). Endo/Epi + Trace
            # then leaves SAX into that pass's trace (see _lv_start_trace).
            store = m.endo_contours if which == "endo" else m.epi_contours
            if m._axis_for(which) is not None and len(store) >= 3:
                lv["sax_edit"] = which
                lv["pass"] = which
                self._lv_apply_target(which)        # only this border grabbable
                self._lv_sync_buttons()
                self._lv_show_sax_both()
            return
        lv["pass"] = which
        self._lv_thick_trace_both()
        ax = m._axis_for(which)
        if ax is None:
            self._lv_enter_align()
            return
        m.axis = ax                                 # activate this pass's axis
        lv["phase"] = "contour"
        lv["plane_idx"] = 0
        self._lv_enter_contour()

    # ---- apex points ----
    #: LV files written with the apex→MV-centre long-axis definition carry this
    #: marker; files without it predate the change (data-reliability warning).
    _LV_AXIS_DEF = "apex-mvcenter"

    def _lv_stamp_axis_def(self, data: dict) -> None:
        """Stamp a saved LV file with the current long-axis definition + a
        timestamp, so a later load can tell new (apex→MV-centre) files from old
        ("垂線") ones."""
        try:
            from datetime import datetime
            data["axis_def"] = self._LV_AXIS_DEF
            data["saved_at"] = datetime.now().strftime("%Y%m%d;%H%M")
        except Exception:                                # noqa: BLE001
            data["axis_def"] = self._LV_AXIS_DEF

    def _lv_warn_if_axis_stale(self, data: dict) -> None:
        """Warn if a loaded LV file predates the apex→MV-centre long axis (marker
        absent). Non-blocking information dialog."""
        from PyQt6.QtWidgets import QMessageBox
        if isinstance(data, dict) and data.get("axis_def") == self._LV_AXIS_DEF:
            return
        QMessageBox.information(
            self.window(), t("LV"),
            t("長軸データが古いためデータ信頼性に問題があります。"))

    def _lv_long_axis_from_apex(self, apex):
        """Deterministic LV long axis = apex → MV 3-D centre (the MV-plane circle
        centre). This is the clinically standard long axis (apex-to-mitral-
        centroid), replacing the old apex→MV-plane-normal ("垂線") axis. Returns
        (axis_dir, radial0) or None if the MV plane isn't set yet.

        radial0 (θ=0 reference) is ANY unit vector ⊥ axis — the Epi/volume/wall-
        thickness analyses are rotationally symmetric so its direction doesn't
        change their results; anchored to the AoV centre for a reproducible θ=0."""
        apex = np.asarray(apex, float)
        mv = self._lv_valves.get("mitral") or (self._lvv or {}).get("mitral")
        if mv is None:
            return None
        c_m = np.asarray(mv[0], float)
        axis_dir = c_m - apex
        nrm = float(np.linalg.norm(axis_dir))
        if nrm < 1e-6:
            return None
        axis_dir = axis_dir / nrm
        ref = None
        av = self._lv_valves.get("aortic") or (self._lvv or {}).get("aortic")
        if av is not None:
            ref = np.asarray(av[0], float) - apex
        if ref is None or np.linalg.norm(
                np.asarray(ref, float)
                - np.dot(ref, axis_dir) * axis_dir) < 1e-6:
            ref = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(ref, axis_dir))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
        radial0 = ref - np.dot(ref, axis_dir) * axis_dir
        radial0 = radial0 / (float(np.linalg.norm(radial0)) or 1.0)
        return axis_dir, radial0

    def _lv_confirm_apex_trace(self) -> None:
        """'Apex' button (trace flow): SET this pass's apex at the centreline
        crossing (the trace pane's plane centre); the LV long axis is then
        DETERMINISTIC = apex → MV centre (no view alignment). Needs the MV plane
        set first. Re-pressing just re-sets the apex + axis at the crossing."""
        lv = self._lv
        if lv is None:
            return
        if lv.get("sax") is not None:                # not used in SAX review
            return
        pas = lv.get("pass")
        if pas not in ("endo", "epi"):
            self._lvv_prompt(t("Choose Endo or Epi first, then set the apex."))
            return
        P = np.asarray(self._pc[lv["pane"]], float).copy()   # crossing = plane ctr
        axinfo = self._lv_long_axis_from_apex(P)
        if axinfo is None:
            self._lvv_prompt(t(
                "Set the MV plane first — the LV long axis runs from the apex to "
                "the MV centre."))
            return
        from multi_dicomviewer.core.lv_axis import LVAxis
        axis_dir, radial0 = axinfo
        ax = LVAxis.from_frame(P, axis_dir, radial0)
        # Assign directly (non-clearing, like an apex re-pin) so any already-
        # captured meridians survive an apex nudge.
        if pas == "endo":
            lv["model"].endo_axis = ax
        else:
            lv["model"].epi_axis = ax
        lv["model"].axis = ax
        lv["model"].set_apex_point(pas, P)
        self._center = P.copy()
        lv["apex_target"] = None
        # Keep the EXACT Apex-set view (zoom AND pan) through Trace + tracing so
        # the crossing doesn't jump to screen centre.
        lv["keep_view"] = True
        if lv.get("phase") == "align":
            lv["phase"] = "ready"                    # axis is set → ready to trace
            lv["plane_idx"] = 0
        self._lv_sync_buttons()
        self._lv_update_text()
        self._lv_show_plane()
        self._lv_redraw_all()
        self._lvv_prompt(t(
            "Apex set + LV long axis = apex→MV centre. Press 'Trace' to trace "
            "the border."))

    def _lv_apex_on_axis(self, tgt, sx, sy):
        """3-D point for an apex click/drag at screen (sx,sy), CONSTRAINED to the
        pass's long (rotation) axis — the apex can slide ALONG the axis but never
        leave it. None if that axis isn't set."""
        ax = self._lv["model"]._axis_for(tgt)
        if ax is None:
            return None
        which = self._lv["pane"]
        wx, wy = self._disp_to_world(which, sx, sy)
        P = self._out_to_world3d(which, wx, wy)
        along = float(np.dot(P - ax.apex, ax.axis))     # project onto the axis
        return ax.apex + along * ax.axis

    def _lv_follow_apex(self, tgt, P_old, P_new) -> None:
        """Move every border vertex that had CONVERGED to this apex (snapped to
        *P_old*) to *P_new*, so the traced Endo/Epi lines track the apex as it
        slides along the axis. Re-stores the affected planes' contours."""
        if P_old is None:
            return
        model = self._lv["model"]
        angs = model.plane_angles()
        pane = self._lv["pane"]
        P_old = np.asarray(P_old, float)
        eps = 0.05                                     # snapped pts sit ON P_old
        for m in self._measures[pane]:
            tag = m.get("_lv")
            if tag is None or tag[1] != tgt or not m.get("pts3d"):
                continue
            moved = False
            p3 = [list(map(float, P)) for P in m["pts3d"]]
            for i, P in enumerate(p3):
                if np.linalg.norm(np.asarray(P) - P_old) <= eps:
                    p3[i] = list(map(float, P_new))
                    moved = True
            if moved:
                m["pts3d"] = [tuple(x) for x in p3]
                try:
                    model.set_long_axis_contour(
                        angs[tag[0] % len(angs)], m["pts3d"], tgt)
                except Exception:
                    pass

    def _lv_place_apex(self, which, sx, sy) -> bool:
        """Place the ACTIVE pass's apex at the clicked point (phase 'apex') and
        pin the long axis to pass THROUGH it, then start tracing. Returns True if
        the click was consumed.

        The apex marks the true ventricular tip, so it must land exactly under
        the cursor — NOT be projected onto the axis line set before the tip was
        known (that made a tip clicked off the axis snap sideways onto it, the
        reported "心尖部点がズレる"). We keep the view-aligned axis DIRECTION and
        radial reference but move the axis ORIGIN to the clicked point, so the
        apex stays ON the rotation axis (meridians converge there) while sitting
        where the user clicked."""
        lv = self._lv
        if (lv is None or lv.get("phase") != "apex"
                or lv.get("apex_target") is None or which != lv.get("pane")):
            return False
        tgt = lv["apex_target"]
        ax = lv["model"]._axis_for(tgt)
        if ax is None:
            return False
        # Full clicked 3-D point in the long-axis plane (u,v through _pc).
        wx, wy = self._disp_to_world(which, sx, sy)
        P = np.asarray(self._out_to_world3d(which, wx, wy), dtype=float)
        # Re-pin the axis through the tip: same direction + radial0, new origin.
        new_ax = type(ax).from_frame(P, ax.axis, ax.radial0)
        if tgt == "endo":
            lv["model"].endo_axis = new_ax
        else:
            lv["model"].epi_axis = new_ax
        lv["model"].axis = new_ax
        lv["model"].set_apex_point(tgt, P)
        # Snap the CrossLine centre onto the tip so the crosshair coincides with
        # the axis through it (crosshair, axis and apex marker all agree).
        self._center = P.copy()
        self._lv_result_lines = []                 # invalidate any volume result
        lv["apex_target"] = None
        self._lv_enter_contour()                   # apex set → trace this pass
        return True

    def _lv_apex_press(self, which, sx, sy, shift=False):
        """Left-press hit-test for the apex markers. Returns "place" if a click
        was consumed as an apex placement, "endo"/"epi" if an existing marker was
        grabbed to drag, else None. In the APEX phase a plain click places the
        apex; a SHIFT-click yields (None) so the view can be adjusted first.
        Grabbing is allowed only when NOT mid-trace."""
        lv = self._lv
        if lv is None or lv.get("phase") not in ("apex", "contour"):
            return None
        if lv.get("phase") == "apex" and lv.get("apex_target") is not None:
            if shift:                               # adjust the view, don't place
                return None
            return "place" if self._lv_place_apex(which, sx, sy) else None
        if which != lv.get("pane"):
            return None
        drafting = (self._draft is not None
                    and self._draft.get("pane") == which)
        if drafting:                                # never grab while tracing
            return None
        # Only the ACTIVE pass's apex is grabbable.
        tgt = lv.get("pass")
        P = (lv["model"].endo_apex if tgt == "endo"
             else lv["model"].epi_apex if tgt == "epi" else None)
        if P is None:
            return None
        wx, wy = self._disp_to_world(which, sx, sy)
        rgrab = self._lv_px_to_mm(which, 14.0)
        ox, oy = self._world3d_to_out(which, P)
        if math.hypot(wx - ox, wy - oy) <= rgrab:
            return tgt
        return None

    def _lv_apex_move(self, which, sx, sy) -> None:
        """Drag the grabbed apex ALONG its long axis (it can't leave the axis),
        carrying with it the border points that had converged to it."""
        tgt = self._lv_apex_drag
        if tgt is None or self._lv is None:
            return
        P_new = self._lv_apex_on_axis(tgt, sx, sy)
        if P_new is None:
            return
        model = self._lv["model"]
        P_old = model.endo_apex if tgt == "endo" else model.epi_apex
        self._lv_follow_apex(tgt, P_old, P_new)    # border pts track the apex
        model.set_apex_point(tgt, P_new)
        self._lv_result_lines = []
        self._redraw_meas(self._lv["pane"])        # moved border points
        self._lv_redraw_all()

    def _lv_snap_apex(self, pts3d, target, tol=3.0):
        """Return *pts3d* (list of 3-tuples) with any vertex within *tol* mm of
        the *target* apex vertex moved exactly onto it, and consecutive
        duplicates collapsed. No apex set → unchanged."""
        lv = self._lv
        P = (lv["model"].endo_apex if target == "endo"
             else lv["model"].epi_apex)
        arr = np.asarray(pts3d, float).reshape(-1, 3)
        if P is None or len(arr) == 0:
            return [tuple(q) for q in arr]
        snapped = np.where(
            (np.linalg.norm(arr - P, axis=1) <= tol)[:, None], P, arr)
        out = [tuple(snapped[0])]
        for q in snapped[1:]:                       # drop consecutive duplicates
            if np.linalg.norm(np.asarray(out[-1]) - q) > 1e-6:
                out.append(tuple(q))
        return out

    def _lv_apex_range_mm(self, key) -> float:
        """Convergence radius in output-mm = TWICE the apex marker's radius
        (6 px → 12 px), so it reads as a circle twice the marker (area ×4)."""
        return self._lv_px_to_mm(key, 12.0)

    def _lv_active_apex(self):
        """The ACTIVE pass's apex vertex (3-D volume mm), or None when not
        tracing / no apex set."""
        lv = self._lv
        if lv is None or lv.get("phase") != "contour":
            return None
        tgt = lv.get("pass")
        return (lv["model"].endo_apex if tgt == "endo"
                else lv["model"].epi_apex if tgt == "epi" else None)

    def _lv_apex_glow(self, key) -> bool:
        """True while the tracing CURSOR is inside the convergence range of the
        apex (a hover cue that the next click will snap there). Cleared once the
        point is confirmed, so the marker returns to a plain red/green dot."""
        lv = self._lv
        return bool(lv is not None and key == lv.get("pane")
                    and self._lv_apex_hot)

    def _lv_apex_hover(self, which, sx, sy) -> None:
        """Update the apex GLOW from the live cursor while tracing a border."""
        lv = self._lv
        hot = False
        if (lv is not None and lv.get("phase") == "contour"
                and lv.get("sax") is None and which == lv.get("pane")
                and lv.get("target") in ("endo", "epi")):
            tgt = lv["pass"]
            P = (lv["model"].endo_apex if tgt == "endo"
                 else lv["model"].epi_apex if tgt == "epi" else None)
            if P is not None:
                ax0, ay0 = self._world3d_to_out(which, P)
                wx, wy = self._disp_to_world(which, sx, sy)
                hot = (math.hypot(wx - ax0, wy - ay0)
                       <= self._lv_apex_range_mm(which))
        if hot != self._lv_apex_hot:
            self._lv_apex_hot = hot
            self._overlay[which].update()

    def _lv_apex_clear_glow(self) -> None:
        """Turn the apex glow off (a point was confirmed / mode changed)."""
        if self._lv_apex_hot:
            self._lv_apex_hot = False
            lv = self._lv
            if lv is not None and lv.get("pane") is not None:
                self._overlay[lv["pane"]].update()

    # ---- contour phase: step the rotated long-axis planes, trace this pass ----
    def _lv_enter_contour(self) -> None:
        """Axis + apex are set → trace the ACTIVE pass's border on its rotated
        long-axis planes (the target is LOCKED to the pass — Endo pass traces
        endo, Epi pass traces epi)."""
        lv = self._lv
        lv["phase"] = "contour"
        lv["plane_idx"] = 0
        lv["target"] = lv["pass"]                  # capture is locked to the pass
        lv["model"].axis = lv["model"]._axis_for(lv["pass"])
        self.set_side("Bi")
        if not self._meas_on:
            self._meas_btn.setChecked(True)
            self._toggle_measure()
        self._set_measure_type("polyline")
        self._lv_sync_buttons()
        self._lv_show_plane()

    def _lv_show_plane(self) -> None:
        lv = self._lv
        ax = lv["model"].axis
        if ax is None:
            return
        angs = lv["model"].plane_angles()
        idx = lv["plane_idx"] % len(angs)
        lv["plane_idx"] = idx
        phi = angs[idx]
        pane = lv["pane"]
        u, v, n = self._ortho(ax.meridian_dir(phi), ax.axis)
        self._frame[pane] = (u, v, n)
        # keep_view (set at Apex): preserve the user's exact zoom/pan through
        # Trace + tracing — do NOT recentre onto the ventricle mid or auto-fit.
        keep = lv.get("keep_view", False)
        if not keep:
            self._pc[pane] = ax.apex + 0.5 * ax.length_mm * ax.axis
        self._cross_ang[pane] = 0.0
        self._roll[pane] = 0.0          # long axis EXACTLY vertical: clear any
        #                                SPIN roll done before Set axis
        first = (not lv.get("fitted", False)) and not keep
        lv["fitted"] = True
        self._view_initial = first
        # Show only THIS plane's border for the ACTIVE pass. Endo and Epi are on
        # DIFFERENT axes, so the other pass's border would project onto a wrong
        # cross-section here — hide it; both show on the SAX pane.
        for mm in self._measures[pane]:
            tag = mm.get("_lv")
            if tag is not None:
                mm["hidden"] = (tag[0] != idx) or (tag[1] != lv.get("pass"))
        self._lv_plane_lbl.setText(f"{idx + 1}/{len(angs)}")
        self._refresh(reset_cam=first)
        self._overlay[pane].update()
        self._lv_update_text()

    def _lv_toggle_sax(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        self._lv_region_reset()
        lv = self._lv
        if lv is None or lv.get("phase") != "contour":
            self._lv_sax_btn.setChecked(False)
            return
        if self._lv_sax_btn.isChecked():
            self._lv_capture_current()
            m = lv["model"]
            endo_ok = m.endo_axis is not None and len(m.endo_contours) >= 3
            epi_ok = m.epi_axis is not None and len(m.epi_contours) >= 3
            # BOTH traced → combined view (endo resampled to the epi axis + epi),
            # the basis for Vol/STL. Otherwise a single-pass short-axis of the
            # active/available pass on ITS OWN axis, for reviewing/editing it.
            if endo_ok and epi_ok:
                which = "both"
                # Promote Endo onto the Epi axis so both borders share ONE frame
                # (edit Endo on the Epi plane; true per-ray wall thickness).
                # Silent + non-destructive (original endo trace stashed; Endo →
                # Trace restores it). Re-place the Endo points on the epi planes.
                if m.promote_endo_to_epi_axis():
                    self._lv_rebuild_measures()
                    self._lv_result_lines = []
            elif lv.get("pass") == "endo" and endo_ok:
                which = "endo"
            elif lv.get("pass") == "epi" and epi_ok:
                which = "epi"
            elif endo_ok:
                which = "endo"
            elif epi_ok:
                which = "epi"
            else:
                self._lv_sax_btn.setChecked(False)
                QMessageBox.information(
                    self.window(), t("LV EF"),
                    t("Trace this pass on at least 3 planes first."))
                return
            lv["sax_which"] = which
            # Point the model's active axis at the SAX reference axis so the SAX
            # code slices against it (short_axis_border_pts default ref_axis =
            # model.axis): endo/epi single → that axis; both → epi.
            m.axis = self._lv_sax_axis()
            rng = self._lv_level_range()
            if rng is None:
                self._lv_sax_btn.setChecked(False)
                return
            common = self._lv_common_range() or rng
            lv["sax"] = 0.5 * (common[0] + common[1])
            sa = "A" if lv["pane"] == "B" else "B"
            lv["sax_pane"] = sa
            lv["sax_saved"] = (
                tuple(np.asarray(a).copy() for a in self._frame[sa]),
                np.asarray(self._pc[sa]).copy(),
                self._cross_ang[sa], self._thick[sa])
            lv["fitted_sax"] = False
            lv["sax_edit"] = None                    # no border armed for editing
            self._lv_apply_target(None)             # no capture in short-axis
            self.set_side("Bi")
            self._lv_sync_buttons()                  # SAX entry → all 4 grey
            self._lv_show_sax_both()
        else:
            # SAX is the last-on button → pressing it turns SAX OFF (back to the
            # long-axis trace).
            self._lv_leave_sax()
            self._lv_show_plane()

    def _lv_show_sax_both(self) -> None:
        lv = self._lv
        ax = lv["model"].axis
        if ax is None or lv.get("sax") is None:
            return
        la, sa = lv["pane"], lv["sax_pane"]
        angs = lv["model"].plane_angles()
        idx = lv["plane_idx"] % len(angs)
        u, v, n = self._ortho(ax.meridian_dir(angs[idx]), ax.axis)
        self._frame[la] = (u, v, n)
        self._pc[la] = ax.apex + 0.5 * ax.length_mm * ax.axis
        self._cross_ang[la] = 0.0
        # The reference long-axis pane is resliced on the SAX axis. For a single
        # pass show only that border; for 'both' (Endo promoted onto the Epi
        # axis) show BOTH borders of THIS plane so Endo can be edited against
        # Epi. The Endo/Epi button (lv["target"]) decides which is grabbable.
        w = lv.get("sax_which")
        show_both = w not in ("endo", "epi")
        ref_which = w if w in ("endo", "epi") else None
        for mm in self._measures[la]:
            tag = mm.get("_lv")
            if tag is not None:
                mm["hidden"] = (tag[0] != idx) or (
                    not show_both and tag[1] != ref_which)
        self._lv_set_short_frame()
        first = not lv.get("fitted_sax", False)
        lv["fitted_sax"] = True
        self._view_initial = first
        self._lv_update_sax_label()
        self._refresh(reset_cam=first)
        for k in (la, sa):
            self._overlay[k].update()

    def _lv_set_short_frame(self) -> None:
        lv = self._lv
        ax = lv["model"].axis
        sa = lv["sax_pane"]
        o, ex, ey, nn = ax.short_axis_basis(float(lv["sax"]))
        # View from the APEX toward the BASE (cardiology convention): horizontal
        # MIRROR so LV is on the viewer's right, RV on the left, diaphragm down.
        # Negates the in-plane horizontal axis (and normal, keeping right-handed).
        # Display-only — measurements use the axis/borders, so are unaffected.
        self._frame[sa] = (-ex, ey, -nn)
        self._pc[sa] = o
        self._cross_ang[sa] = 0.0
        self._thick[sa] = 0.0
        for mm in self._measures[sa]:
            if mm.get("_lv") is not None:
                mm["hidden"] = True

    def _lv_update_sax_label(self) -> None:
        rng = self._lv_level_range()
        pos = (float(self._lv["sax"]) - rng[0]) if rng else 0.0
        self._lv_plane_lbl.setText(t("SAX {mm:.0f}mm", mm=pos))

    def _lv_reslice_short(self) -> None:
        lv = self._lv
        if lv is None or lv.get("sax") is None:
            return
        self._lv_set_short_frame()
        self._lv_update_sax_label()
        self._view_initial = False
        self._refresh()
        for k in (lv["pane"], lv["sax_pane"]):
            self._overlay[k].update()

    def _lv_sax_axis(self):
        """Reference axis for the short-axis display, by what it shows: a single
        pass → that pass's own axis; 'both' → the EPI (myocardial) axis (endo is
        resampled onto it)."""
        m = self._lv["model"]
        w = self._lv.get("sax_which")
        if w == "endo":
            return m.endo_axis or m.axis
        if w == "epi":
            return m.epi_axis or m.axis
        return m.epi_axis or m.endo_axis or m.axis        # both → epi

    def _lv_sax_borders(self):
        """Which border(s) the short-axis shows: a single pass, or both."""
        w = self._lv.get("sax_which")
        return ["endo", "epi"] if w not in ("endo", "epi") else [w]

    def _lv_sax_stores(self):
        """The contour store(s) whose along values are in the CURRENT SAX axis'
        frame — the single pass shown, or epi for 'both' (its axis is the SAX
        reference; endo is resampled onto it at draw time)."""
        m = self._lv["model"]
        w = self._lv.get("sax_which")
        if w == "endo":
            return (m.endo_contours,)
        if w == "epi":
            return (m.epi_contours,)
        if w == "both":
            return (m.epi_contours,)
        return (m.endo_contours, m.epi_contours)          # fallback

    def _lv_leave_sax(self) -> None:
        lv = self._lv
        sa = lv.get("sax_pane")
        saved = lv.get("sax_saved")
        if sa is not None and saved is not None:
            self._frame[sa] = tuple(np.asarray(a).copy() for a in saved[0])
            self._pc[sa] = np.asarray(saved[1]).copy()
            self._cross_ang[sa] = saved[2]
            self._thick[sa] = saved[3]
        lv["sax"] = None
        lv["sax_saved"] = None
        lv["model"].axis = lv["model"]._axis_for(lv.get("pass"))
        lv["target"] = lv.get("pass")           # resume tracing this pass
        self._lv_sax_btn.setChecked(False)

    def _lv_capture_current(self) -> None:
        lv = self._lv
        if lv is None or lv.get("phase") != "contour":
            return
        if lv.get("target") not in ("endo", "epi"):
            return
        pane = lv["pane"]
        if self._draft and self._draft.get("pane") == pane \
                and len(self._draft.get("pts", [])) >= 2:
            self._commit_draft()
        m = None
        for cand in reversed(self._measures[pane]):
            if (cand.get("type") == "polyline"
                    and len(cand.get("pts3d", [])) >= 2
                    and cand.get("_lv") is None):
                m = cand
                break
        if m is None:
            return
        # Snap vertices within the convergence range (= twice the apex marker
        # radius) of this surface's apex onto it, so every meridian passes
        # through the shared apex (the displayed border ends there too).
        m["pts3d"] = self._lv_snap_apex(
            m["pts3d"], lv["target"], tol=self._lv_apex_range_mm(pane))
        phi = lv["model"].plane_angles()[lv["plane_idx"]]
        try:
            lv["model"].set_long_axis_contour(phi, m["pts3d"],
                                              which=lv["target"])
        except Exception:
            return
        tag = (lv["plane_idx"], lv["target"])
        self._measures[pane] = [mm for mm in self._measures[pane]
                                if mm is m or mm.get("_lv") != tag]
        m["_lv"] = tag
        m["color"] = "#ff4040" if lv["target"] == "endo" else "#40c040"
        m["smooth"] = True
        self._lv_invalidate_volume()   # a changed border invalidates the volume
        self._draft = None
        self._draft_redo = []          # the trace is consumed
        self._lv_apex_hot = False      # border confirmed → marker back to normal
        self._redraw_meas(pane)
        self._lv_redraw_all()
        self._lv_record_create(pane, m)   # Ctrl+Z removes the whole new border

    def lv_nav_key(self, where: str) -> bool:
        """A / F (via MainWindow._nav_active) step the long-axis plane while
        LV-tracing, instead of navigating CT series. Returns True if handled."""
        if self._lv is not None and self._lv.get("phase") == "contour":
            if where == "prev":
                self._lv_step_plane(-1)
                return True
            if where == "next":
                self._lv_step_plane(1)
                return True
        return False

    def _lv_step_plane(self, delta) -> None:
        if self._lv is None or self._lv.get("phase") != "contour":
            return
        if self._lv.get("sax") is not None:
            before = self._lv_scalar_snap()
            self._lv["plane_idx"] += int(delta)
            self._lv_show_sax_both()
            self._lv_record_scalar(before)         # Ctrl+Z / Ctrl+Y
            return
        self._lv_capture_current()
        pane = self._lv["pane"]
        self._measures[pane] = [
            m for m in self._measures[pane]
            if not (m.get("type") == "polyline" and m.get("_lv") is None)]
        self._draft = None
        self._lv["plane_idx"] += int(delta)
        self._lv_apply_target(self._lv["pass"])   # target locked to this pass
        self._lv_show_plane()

    def _lv_sax_active(self) -> bool:
        return (self._lv is not None
                and self._lv.get("phase") == "contour"
                and self._lv.get("sax") is not None)

    def _lv_axis_locked(self) -> bool:
        """True once the active pass's long axis is SET (Set axis pressed) and we
        are on the long-axis view — the 3DCT↔vertical-axis relationship is then
        fixed, so Rotate/Spin/Thick (which would re-tilt the reslice frame) are
        blocked. Zoom/Move/WL and the cross-section level still work."""
        lv = self._lv
        if getattr(self, "_lv_view_free", False):     # Epi領域表示 inspect
            return False
        return (lv is not None
                and lv.get("phase") in ("ready", "apex", "contour")
                and lv.get("sax") is None)

    def _lv_apply_view_free(self) -> None:
        """Recompute the free-view state from the Trace⇄View toggle and the
        Epi領域表示 button, then lift/restore the tool + CenterLine locks."""
        lv = self._lv
        trace_view = bool(lv is not None and lv.get("trace_view"))
        region_on = bool(getattr(self, "_lv_region_btn", None) is not None
                         and self._lv_region_btn.isChecked())
        self._lv_view_free = trace_view or region_on
        self._refresh_tool_availability()
        for k in ("A", "B"):
            self._overlay[k].update()

    def _lv_toggle_region(self) -> None:
        """Epi領域表示: show/hide the red measured region (computing it if not done
        yet) and, as one free-view source, allow 3-D inspection."""
        on = self._lv_region_btn.isChecked()
        self._lv_region_btn.setStyleSheet(
            "QPushButton{background:#ff5a5a;color:black;}" if on else "")
        if on and self._lvv_mask_vol is None:
            self._lv_apply_view_free()
            self._lv_compute_volume()
            return
        self._lvv_mask_on = on and (self._lvv_mask_vol is not None)
        self._lvv_redraw()
        self._lv_apply_view_free()

    def _lv_region_reset(self) -> None:
        if self._lv is not None:
            self._lv["trace_view"] = False
        if getattr(self, "_lv_region_btn", None) is not None:
            self._lv_region_btn.setChecked(False)
            self._lv_region_btn.setStyleSheet("")
        self._lv_view_free = False

    def _lv_level_range(self):
        """The along span (in the SAX axis' frame) to SCROLL the level over:
        apex = the most-apical traced point (min of minima), base = the common
        base (min of maxima). Uses only the store(s) in the SAX axis' frame."""
        mins, maxs = [], []
        for store in self._lv_sax_stores():
            for c in store.values():
                a = np.asarray(c, float).reshape(-1, 2)[:, 0]
                mins.append(float(a.min()))
                maxs.append(float(a.max()))
        if not mins:
            return None
        lo, hi = min(mins), min(maxs)     # apex = deepest, base = common
        # Let the level scroll all the way to the DRAWN base-cut line even when
        # the shortest meridian of the shown store stops a touch short of it
        # (reported on Mac: SAX scroll halted just before the base cut). The cut
        # line is at along_range(endo|epi)[1] — which may use the OTHER border
        # than the one shown in SAX, so its base can sit beyond this store's
        # common base. Extend hi to reach it; never shrink below the common base.
        m = self._lv["model"]
        br = m.along_range("endo") or m.along_range("epi")
        if br is not None:
            hi = max(hi, float(br[1]))
        return (lo, hi) if hi > lo else None

    def _lv_common_range(self):
        """COMMON along-range (SAX axis frame) where every meridian of the shown
        border reaches — used to pick the SAX entry level so the border shows on
        entry."""
        mins, maxs = [], []
        for store in self._lv_sax_stores():
            for c in store.values():
                a = np.asarray(c, float).reshape(-1, 2)[:, 0]
                mins.append(float(a.min()))
                maxs.append(float(a.max()))
        if not mins:
            return None
        lo, hi = max(mins), min(maxs)
        return (lo, hi) if hi > lo else (min(mins), min(maxs))

    def _lv_step_level(self, delta) -> None:
        rng = self._lv_level_range()
        if rng is None:
            return
        before = self._lv_scalar_snap()
        step = (rng[1] - rng[0]) / 24.0
        self._lv["sax"] = min(rng[1], max(
            rng[0], float(self._lv["sax"]) + float(delta) * step))
        self._lv_reslice_short()
        self._lv_record_scalar(before)             # Ctrl+Z / Ctrl+Y

    def _lv_drag_level(self, dy) -> None:
        rng = self._lv_level_range()
        if rng is None:
            return
        span = rng[1] - rng[0]
        self._lv["sax"] = min(rng[1], max(
            rng[0], float(self._lv["sax"]) + (dy / 200.0) * span))
        self._lv_reslice_short()

    def _lv_px_to_mm(self, which, px) -> float:
        ps = float(self._ps[which])
        return float(px) * (2.0 * ps / max(1, self.pane[which].canvas.height()))

    def _lv_line_press(self, which, sx, sy):
        # Grab the SAX line ONLY near its ○ handle (out at the view edge, clear
        # of the heart) — NOT along its whole length, which crosses the trace and
        # would otherwise steal clicks meant to place / edit a border point (a
        # near-miss moved the level instead, dropping the point onto a different
        # cross-section). Also yield outright while a trace is in progress or the
        # cursor is on a border point, so tracing / editing always wins.
        if not self._lv_sax_active() or self._lv["model"].axis is None:
            return None
        if self._draft is not None and self._draft.get("pane") == which:
            return None                       # mid-trace → clicks add points
        if self._pick_handle(which, sx, sy) is not None:
            return None                       # let the border point be edited
        hs = self._lv_handle_screen(which)    # the visible ○ (pinned to the edge)
        if hs is None:
            return None
        if math.hypot(sx - hs[0], sy - hs[1]) <= 22.0:
            return "level" if which == self._lv.get("pane") else "meridian"
        return None

    def _lv_line_move(self, which, sx, sy) -> None:
        kind = self._lv_line_drag
        if kind is None:
            return
        lv = self._lv
        ax = lv["model"].axis
        wx, wy = self._disp_to_world(which, sx, sy)
        if kind == "level":
            along = wy + float(np.dot(self._pc[which] - ax.apex, ax.axis))
            rng = self._lv_level_range()
            if rng is not None:
                along = min(rng[1], max(rng[0], along))
            lv["sax"] = along
            self._lv_reslice_short()
        elif kind == "meridian":
            th = math.degrees(math.atan2(wy, wx)) % 180.0
            angs = lv["model"].plane_angles()
            idx = min(range(len(angs)), key=lambda i: min(
                abs(th - angs[i]), 180.0 - abs(th - angs[i])))
            if idx != (lv["plane_idx"] % len(angs)):
                lv["plane_idx"] = idx
                self._lv_show_sax_both()

    def _lv_drop_border(self, m) -> None:
        if self._lv is None:
            return
        tag = m.get("_lv")
        if tag is None:
            return
        idx, target = tag
        angs = self._lv["model"].plane_angles()
        if 0 <= idx < len(angs):
            self._lv["model"].clear_contour(angs[idx], which=target)
        self._lv_result_lines = []
        if self._lv_sax_active():
            self._overlay[self._lv["sax_pane"]].update()

    def _lv_line_set_grabbed(self, which, on: bool) -> None:
        if self._lv_line_hi.get(which) == bool(on):
            return
        self._lv_line_hi[which] = bool(on)
        self._overlay[which].update()

    def _lv_line_hover(self, which, sx, sy) -> None:
        if self._lv_line_drag is not None:
            return
        on = bool(self._lv_sax_active()
                  and self._lv_line_press(which, sx, sy) is not None)
        self._lv_line_set_grabbed(which, on)

    def _lv_apply_target(self, target) -> None:
        self._lv["target"] = target
        self._lv_endo_btn.setChecked(target == "endo")
        self._lv_epi_btn.setChecked(target == "epi")
        self._lv_update_text()

    def _lv_set_target(self, target) -> None:
        if self._lv is None or self._lv.get("phase") != "contour":
            return
        self._lv_capture_current()
        new = None if self._lv.get("target") == target else target
        self._lv_apply_target(new)

    def _lv_on_border_committed(self) -> None:
        lv = self._lv
        if lv is not None and lv.get("target") in ("endo", "epi"):
            self._lv_capture_current()
            self._lv_update_text()

    def _lv_clear_confirm(self) -> None:
        """'Clear borders' button → confirm before discarding the traced borders."""
        from PyQt6.QtWidgets import QMessageBox
        if self._lv is None or self._lv.get("phase") != "contour":
            return
        if QMessageBox.question(
                self.window(), t("LV EF"),
                t("Clear all traced borders? This cannot be undone.")) \
                != QMessageBox.StandardButton.Yes:
            return
        self._lv_clear_contours()

    def _lv_clear_contours(self) -> None:
        if self._lv is None or self._lv.get("phase") != "contour":
            return
        pane = self._lv["pane"]
        self._lv["model"].endo_contours.clear()
        self._lv["model"].epi_contours.clear()
        self._lv["model"].endo_planes.clear()
        self._lv["model"].epi_planes.clear()
        self._lv["model"].endo_orig = None          # drop the promotion stash
        self._lv_reset_undo()
        self._measures[pane] = [m for m in self._measures[pane]
                                if m.get("type") != "polyline"]
        self._draft = None
        self._lv_result_lines = []
        self._lv["plane_idx"] = 0
        if self._lv.get("sax") is not None:
            self._lv_leave_sax()
        self._lv_apply_target(self._lv.get("pass"))
        self._lv_show_plane()
        self._redraw_meas(pane)

    def _lv_has_border(self, which, target) -> bool:
        if self._lv is None:
            return False
        idx = self._lv.get("plane_idx", 0)
        for m in self._measures.get(which, []):
            tag = m.get("_lv")
            if tag is not None and tag[0] == idx and tag[1] == target:
                return True
        return False

    _UNDO_MAX = 80                         # Ctrl+Z / Ctrl+Y depth (unified)

    # ---- unified undo / redo stack (Ctrl+Z / Ctrl+Y) ----------------------
    # A single command list + cursor covering EVERY undoable action: image
    # transforms, Spin+, centreline move/rotate, Zoom/Move/Paging/Thick, W/L,
    # angio-angle, recentre, SAX level / meridian, apex drag, and LV border
    # edits (drag / add / delete / create / per-point tracing). Each command is
    # {"undo": fn, "redo": fn}. Most snapshot the whole VIEW state (frames,
    # pc, zoom/pan/roll, cross angles, thickness, W/L, 2-D axes/slice, CPR)
    # before AND after, so undo/redo restore exactly. Cleared on series load /
    # 2-D↔3-D switch / LV enter·exit·clear so no command restores a stale view.
    def _undo_clear(self) -> None:
        self._undo_cmds = []
        self._undo_idx = 0
        self._lv_edit_before = None   # stashed LV-border snap during a drag
        self._gesture_snap = None     # view snap captured at a drag's press
        self._gesture_lv = None       # LV level/meridian snap at a drag's press
        self._gesture_moved = False   # did the current drag change anything
        self._lv_apex_snap = None     # LV geometry snap while dragging an apex
        self._draft_redo = []         # points popped from an in-progress trace

    def _undo_record(self, undo_fn, redo_fn) -> None:
        if not hasattr(self, "_undo_cmds"):
            self._undo_clear()
        if self._undo_idx < len(self._undo_cmds):
            del self._undo_cmds[self._undo_idx:]
        self._undo_cmds.append({"undo": undo_fn, "redo": redo_fn})
        if len(self._undo_cmds) > self._UNDO_MAX:
            del self._undo_cmds[:len(self._undo_cmds) - self._UNDO_MAX]
        self._undo_idx = len(self._undo_cmds)

    def _undo_view(self, before, after) -> None:
        if before is None or after is None:
            return
        self._undo_record(lambda b=before: self._view_restore(b),
                          lambda a=after: self._view_restore(a))

    # ---- general Measure (Line/Polyline/Ellipse/Polygon/Point) undo --------
    # NON-LV measures snapshot the pane list before/after so Ctrl+Z / Ctrl+Y
    # restore it exactly (LV borders keep their own _lv_record_* undo).
    def _meas_pane_snap(self, pane):
        import copy
        return copy.deepcopy([m for m in self._measures.get(pane, [])
                              if not m.get("_lv")])

    def _meas_pane_restore(self, pane, snap) -> None:
        import copy
        cur = self._measures.get(pane, [])
        lv_keep = [m for m in cur if m.get("_lv")]
        old_ids = {m.get("id") for m in cur if not m.get("_lv")}
        self._measures[pane] = copy.deepcopy(snap) + lv_keep
        new_ids = {m.get("id") for m in self._measures[pane]
                   if not m.get("_lv")}
        try:
            for mid in old_ids - new_ids:
                if mid is not None:
                    self.measurement_removed.emit(int(mid))
            add_back = new_ids - old_ids
            for m in self._measures[pane]:
                if not m.get("_lv") and m.get("id") in add_back:
                    meas = Measurement(kind=m["type"].capitalize(),
                                       points=list(m["pts"]), spacing_mm=None,
                                       mid=int(m["id"]))
                    meas.text = self._metrics_text(pane, m)
                    self.measurement_added.emit(meas)
        except Exception:                               # noqa: BLE001
            pass
        self._redraw_meas(pane)

    def _meas_record(self, pane, before) -> None:
        if before is None:
            return
        after = self._meas_pane_snap(pane)
        self._undo_record(
            lambda p=pane, b=before: self._meas_pane_restore(p, b),
            lambda p=pane, a=after: self._meas_pane_restore(p, a))

    def _undo_last(self) -> None:
        """Ctrl+Z: while tracing drop the last point; else step one command
        back. A command whose undo() returns False (cancelled confirm) is kept."""
        if self._draft is not None and self._draft.get("pts"):
            self._draft_pop_point()
            return
        if not getattr(self, "_undo_cmds", None) or self._undo_idx <= 0:
            return
        self._undo_idx -= 1
        try:
            res = self._undo_cmds[self._undo_idx]["undo"]()
        except Exception:                              # noqa: BLE001
            res = None
        if res is False:
            self._undo_idx += 1

    def _redo_last(self) -> None:
        """Ctrl+Y: while tracing re-place a dropped point; else step forward."""
        if self._draft is not None and getattr(self, "_draft_redo", None):
            self._draft_push_point()
            return
        if not getattr(self, "_undo_cmds", None) \
                or self._undo_idx >= len(self._undo_cmds):
            return
        try:
            res = self._undo_cmds[self._undo_idx]["redo"]()
        except Exception:                              # noqa: BLE001
            res = None
        if res is not False:
            self._undo_idx += 1

    def _view_snapshot(self) -> dict:
        """Full display state — pygfx configures the camera each frame from these
        scalars (_ps/_pan/_roll), so capturing them restores zoom/pan/roll."""
        return {
            "frame": {k: tuple(np.asarray(a, float).copy()
                               for a in self._frame[k]) for k in ("A", "B")},
            "pc": {k: np.asarray(self._pc[k], float).copy() for k in ("A", "B")},
            "ps": dict(self._ps),
            "pan": {k: np.asarray(self._pan[k], float).copy() for k in ("A", "B")},
            "roll": dict(self._roll),
            "cross_ang": dict(self._cross_ang),
            "apex_flip": dict(self._apex_flip),
            "thick": dict(self._thick),
            "center": np.asarray(self._center, float).copy(),
            "axes2d": (None if self._axes2d is None else
                       (np.asarray(self._axes2d[0], float).copy(),
                        np.asarray(self._axes2d[1], float).copy())),
            "slice2d": int(self._slice2d),
            "win": float(self._win), "lvl": float(self._lvl),
            "side": self._side,
            "cpr_T": (None if self._cpr is None
                      else np.asarray(self._cpr["T"], float).copy()),
            "cpr_rot": (None if self._cpr is None else self._cpr.get("rot")),
            "cpr_idx": (None if self._cpr is None else self._cpr.get("idx")),
        }

    def _view_restore(self, snap) -> None:
        if self._vol is None or snap is None:
            return
        for k in ("A", "B"):
            self._frame[k] = tuple(np.asarray(a, float).copy()
                                   for a in snap["frame"][k])
            self._pc[k] = np.asarray(snap["pc"][k], float).copy()
            self._ps[k] = float(snap["ps"][k])
            self._pan[k] = np.asarray(snap["pan"][k], float).copy()
            self._roll[k] = float(snap["roll"][k])
        self._cross_ang = dict(snap["cross_ang"])
        if snap.get("apex_flip") is not None:
            self._apex_flip = dict(snap["apex_flip"])
        self._thick = dict(snap["thick"])
        self._center = np.asarray(snap["center"], float).copy()
        if snap.get("axes2d") is not None:
            self._axes2d = (snap["axes2d"][0].copy(), snap["axes2d"][1].copy())
        self._slice2d = int(snap.get("slice2d", self._slice2d))
        self._win = float(snap.get("win", self._win))
        self._lvl = float(snap.get("lvl", self._lvl))
        self._side = snap.get("side", self._side)
        if self._cpr is not None and snap.get("cpr_T") is not None:
            self._cpr["T"] = np.asarray(snap["cpr_T"], float).copy()
            if snap.get("cpr_rot") is not None:
                self._cpr["rot"] = snap["cpr_rot"]
            if snap.get("cpr_idx") is not None:
                self._cpr["idx"] = snap["cpr_idx"]
            self._cpr_apply_xform()
        if self._mode == "2D":
            self._apply_2d_axes()
        self._view_initial = False
        self._refresh(reset_cam=False)
        for k in ("A", "B"):
            self._redraw_meas(k)
        if self._lv is not None:
            self._lv_redraw_all()
        if self._mode == "2D":
            self._sync_seek()

    def _gesture_begin(self) -> None:
        """A view-changing drag starts: snapshot so its whole gesture collapses
        into ONE Ctrl+Z step, committed on release."""
        self._gesture_snap = self._view_snapshot()
        self._gesture_lv = self._lv_scalar_snap()   # None outside LV
        self._gesture_moved = False

    def _gesture_commit(self) -> None:
        """Drag released: record the view change (only if it moved) and the LV
        level/meridian change (only if it changed) as one step each."""
        snap = getattr(self, "_gesture_snap", None)
        if snap is not None and getattr(self, "_gesture_moved", False):
            self._undo_view(snap, self._view_snapshot())
        self._lv_record_scalar(getattr(self, "_gesture_lv", None))
        self._gesture_snap = None
        self._gesture_lv = None
        self._gesture_moved = False

    # ---- LV short-axis LEVEL + shown MERIDIAN (navigation) undo -----------
    def _lv_scalar_snap(self):
        if self._lv is None:
            return None
        return {"sax": self._lv.get("sax"),
                "plane_idx": self._lv.get("plane_idx")}

    def _lv_scalar_restore(self, snap) -> bool:
        if self._lv is None or snap is None:
            return False
        if snap.get("sax") is not None:
            self._lv["sax"] = snap["sax"]
        if snap.get("plane_idx") is not None:
            self._lv["plane_idx"] = snap["plane_idx"]
        if self._lv_sax_active():
            self._lv_show_sax_both()
        return True

    def _lv_record_scalar(self, before) -> None:
        if before is None:
            return
        after = self._lv_scalar_snap()
        if after is None or before == after:
            return
        self._undo_record(lambda b=before: self._lv_scalar_restore(b),
                          lambda a=after: self._lv_scalar_restore(a))

    # ---- LV apex-drag undo (apex point + the borders that follow it) ------
    def _lv_geom_snap(self):
        if self._lv is None:
            return None
        model = self._lv["model"]
        borders = {}
        for pane in ("A", "B"):
            for m in self._measures.get(pane, []):
                tag = m.get("_lv")
                if tag is not None and m.get("pts3d"):
                    borders[(pane, tuple(tag))] = [list(map(float, P))
                                                   for P in m["pts3d"]]
        return {
            "endo_apex": (None if model.endo_apex is None
                          else list(map(float, model.endo_apex))),
            "epi_apex": (None if model.epi_apex is None
                         else list(map(float, model.epi_apex))),
            "borders": borders,
        }

    def _lv_geom_restore(self, snap) -> bool:
        if self._lv is None or snap is None:
            return False
        model = self._lv["model"]
        if snap.get("endo_apex") is not None:
            model.set_apex_point("endo", np.asarray(snap["endo_apex"], float))
        if snap.get("epi_apex") is not None:
            model.set_apex_point("epi", np.asarray(snap["epi_apex"], float))
        angs = model.plane_angles()
        for (pane, tag), pts in snap["borders"].items():
            for m in self._measures.get(pane, []):
                if m.get("_lv") == tag:
                    m["pts3d"] = [np.asarray(P, float) for P in pts]
                    m["pts"] = [self._world3d_to_out(pane, P)
                                for P in m["pts3d"]]
                    model.set_long_axis_contour(
                        angs[tag[0] % len(angs)], m["pts3d"], tag[1])
                    break
        self._lv_invalidate_volume()
        for k in ("A", "B"):
            self._redraw_meas(k)
        self._lv_redraw_all()
        return True

    def _lv_record_geom(self, before) -> None:
        if before is None:
            return
        after = self._lv_geom_snap()
        if after is None or before == after:
            return
        self._undo_record(lambda b=before: self._lv_geom_restore(b),
                          lambda a=after: self._lv_geom_restore(a))

    # ---- LV border edits (drag / add / delete) ---------------------------
    def _lv_reset_undo(self) -> None:
        self._undo_clear()

    def _lv_border_snap(self, pane, mi):
        if self._lv is None or not (0 <= mi < len(self._measures.get(pane, []))):
            return None
        m = self._measures[pane][mi]
        tag = m.get("_lv")
        if tag is None or not m.get("pts3d"):
            return None
        return {"pane": pane, "tag": tuple(tag),
                "pts3d": [list(map(float, P)) for P in m["pts3d"]]}

    def _lv_record_border(self, before, pane, mi) -> None:
        after = self._lv_border_snap(pane, mi)
        if before is None or after is None:
            return
        self._undo_record(lambda b=before: self._lv_restore_border(b),
                          lambda a=after: self._lv_restore_border(a))

    def _lv_push_undo(self, pane, mi) -> None:
        """Stash an LV border BEFORE a DRAG; committed on release."""
        self._lv_edit_before = self._lv_border_snap(pane, mi)

    def _lv_restore_border(self, snap) -> bool:
        if self._lv is None or self._lv.get("phase") != "contour":
            return False
        pane, tag = snap["pane"], tuple(snap["tag"])
        for m in self._measures.get(pane, []):
            if m.get("_lv") == tag:
                m["pts3d"] = [np.asarray(P, float) for P in snap["pts3d"]]
                # Regenerate the 2-D pts to MATCH — add/delete change the count,
                # and the re-projection only runs when counts already agree.
                m["pts"] = [self._world3d_to_out(pane, P) for P in m["pts3d"]]
                if len(m["pts"]) >= 3:
                    m["type"] = "polyline"
                angs = self._lv["model"].plane_angles()
                self._lv["model"].set_long_axis_contour(
                    angs[tag[0] % len(angs)], m["pts3d"], tag[1])
                self._lv_invalidate_volume()
                self._redraw_meas(pane)
                if self._lv_sax_active():
                    self._overlay[self._lv["sax_pane"]].update()
                return True
        return False

    # ---- LV border CREATION undo (whole traced border) -------------------
    def _lv_measure_copy(self, m):
        rec = {"id": m.get("id"), "type": m.get("type", "polyline"),
               "pts": [tuple(map(float, p)) for p in m.get("pts", [])],
               "_lv": tuple(m["_lv"]),
               "color": m.get("color"),
               "smooth": bool(m.get("smooth", True))}
        if m.get("pts3d"):
            rec["pts3d"] = [list(map(float, P)) for P in m["pts3d"]]
        return rec

    def _lv_record_create(self, pane, m) -> None:
        if self._lv is None or m.get("_lv") is None:
            return
        snap = {"pane": pane, "tag": tuple(m["_lv"]),
                "rec": self._lv_measure_copy(m)}
        self._undo_record(lambda s=snap: self._lv_undo_create(s),
                          lambda s=snap: self._lv_redo_create(s))

    def _lv_undo_create(self, snap) -> bool:
        from PyQt6.QtWidgets import QMessageBox
        if self._lv is None:
            return True
        if QMessageBox.question(
                self.window(), t("LV EF"),
                t("This deletes the whole border. OK?")) \
                != QMessageBox.StandardButton.Yes:
            return False
        pane, tag = snap["pane"], tuple(snap["tag"])
        for i, mm in enumerate(list(self._measures.get(pane, []))):
            if mm.get("_lv") == tag:
                self._lv_drop_border(mm)
                del self._measures[pane][i]
                break
        self._lv_invalidate_volume()
        self._redraw_meas(pane)
        self._lv_redraw_all()
        return True

    def _lv_redo_create(self, snap) -> bool:
        if self._lv is None:
            return True
        pane, tag = snap["pane"], tuple(snap["tag"])
        rec = self._lv_measure_copy(snap["rec"])
        p3 = [np.asarray(P, float) for P in rec.get("pts3d", [])]
        rec["pts3d"] = p3
        self._measures[pane] = [mm for mm in self._measures.get(pane, [])
                                if mm.get("_lv") != tag]
        self._measures[pane].append(rec)
        if p3:
            angs = self._lv["model"].plane_angles()
            self._lv["model"].set_long_axis_contour(
                angs[tag[0] % len(angs)], p3, tag[1])
        self._lv_invalidate_volume()
        self._redraw_meas(pane)
        self._lv_redraw_all()
        return True

    # ---- in-progress trace: per-point Ctrl+Z / Ctrl+Y --------------------
    def _draft_pop_point(self) -> None:
        d = self._draft
        if not d or not d.get("pts"):
            return
        pt = d["pts"].pop()
        p3 = None
        if d.get("pts3d") and len(d["pts3d"]) > len(d["pts"]):
            p3 = d["pts3d"].pop()
        self._draft_redo.append((pt, p3))
        self._redraw_meas(d["pane"])

    def _draft_push_point(self) -> None:
        d = self._draft
        if not d or not getattr(self, "_draft_redo", None):
            return
        pt, p3 = self._draft_redo.pop()
        d["pts"].append(pt)
        if p3 is not None:
            d.setdefault("pts3d", []).append(p3)
        self._redraw_meas(d["pane"])

    def _lv_invalidate_volume(self) -> None:
        """A border changed → the computed volume is stale: clear the result
        readout, forget the numbers and turn CalcVol grey. Re-pressing CalcVol
        recomputes + re-lights blue (and Save then persists the fresh value)."""
        lv = self._lv
        if lv is None:
            return
        if lv.get("vol_done") or lv.get("vol_endo_ml") is not None \
                or self._lv_result_lines:
            lv["vol_done"] = False
            lv["vol_endo_ml"] = None
            lv["vol_myo_ml"] = None
            self._lv_result_lines = []
            self._lv_sync_buttons()
            self._lv_update_text()

    def _lv_live_recapture(self, key, m) -> None:
        if not self._lv_sax_active():
            return
        tag = m.get("_lv")
        if (tag is None or key != self._lv.get("pane")
                or not m.get("pts3d") or tag[1] not in ("endo", "epi")):
            return
        angs = self._lv["model"].plane_angles()
        self._lv["model"].set_long_axis_contour(
            angs[tag[0] % len(angs)], m["pts3d"], tag[1])
        self._lv_invalidate_volume()         # edited border → volume stale
        self._overlay[self._lv["sax_pane"]].update()

    def _lv_compute_volume(self) -> None:
        """Build the endo/epi surfaces and report LV cavity + myocardial volume.

        Runs on a worker thread behind a busy progress dialog: the click gets
        immediate "computing" feedback, the window stays painted, and the bar
        animates. The compute is pure-numpy on the model (no Qt/GPU), so it is
        safe off the UI thread."""
        from PyQt6.QtCore import Qt, QThread
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog
        if self._lv is None or self._lv.get("phase") != "contour":
            return
        self._lv_capture_current()
        m = self._lv["model"]
        top = self.window()
        spacing = max(0.5, float(min(self._dims)))

        result: dict = {}

        class _VolWorker(QThread):
            def run(self_) -> None:
                try:
                    m.build()
                    result["endo"] = m.volume_ml(spacing, "endo")
                    result["myo"] = m.myocardial_volume_ml(spacing)
                except Exception as exc:                  # noqa: BLE001
                    result["err"] = str(exc)

        dlg = QProgressDialog(t("Computing LV volume…"), "", 0, 0, top)
        dlg.setWindowTitle(t("LV EF"))
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        worker = _VolWorker()
        worker.finished.connect(dlg.reset)
        worker.start()
        dlg.exec()
        worker.wait()
        worker.deleteLater()

        if "err" in result:
            QMessageBox.information(
                top, t("LV EF"),
                t("Could not build the LV surface: {err}", err=result["err"]))
            return
        endo_ml = result.get("endo")
        if endo_ml is None:
            QMessageBox.information(
                top, t("LV EF"),
                t("Trace the endo border on at least 3 planes first."))
            return
        lines = [t("LV cavity volume: {v:.1f} mL", v=endo_ml)]
        myo_ml = result.get("myo")
        if myo_ml is not None:
            lines.append(t("Myocardial volume: {v:.1f} mL", v=myo_ml))
        self._lv_result_lines = lines
        self._lv["vol_done"] = True          # CalcVol button → blue (valid result)
        # Remember the numbers so Save can persist them and Load can redisplay.
        self._lv["vol_endo_ml"] = float(endo_ml)
        self._lv["vol_myo_ml"] = None if myo_ml is None else float(myo_ml)
        self._lv_sync_buttons()
        self._lv_update_text()

    def _lv_toggle_wall(self) -> None:
        """Toggle the short-axis wall-thickness colour map. Turning it ON shows
        the COMBINED short-axis (Endo + Epi borders on the epi axis) and colours
        the Epi−Endo gap; needs both passes traced."""
        from PyQt6.QtWidgets import QMessageBox
        lv = self._lv
        if lv is None or lv.get("phase") != "contour":
            self._lv_wall_btn.setChecked(False)
            return
        self._lv_wall = self._lv_wall_btn.isChecked()
        if not self._lv_wall:                       # turning OFF → just redraw
            if self._lv_sax_active():
                self._overlay[lv["sax_pane"]].update()
            return
        m = lv["model"]
        both = (m.endo_axis is not None and len(m.endo_contours) >= 3
                and m.epi_axis is not None and len(m.epi_contours) >= 3)
        if not both:
            self._lv_wall_btn.setChecked(False)
            self._lv_wall = False
            QMessageBox.information(
                self.window(), t("LV EF"),
                t("Trace BOTH Endo and Epi (≥3 planes each) first — the wall "
                  "map needs both borders."))
            return
        # Ensure the COMBINED (both-border, epi-axis) short-axis is shown.
        if not self._lv_sax_active():
            self._lv_sax_btn.setChecked(True)
            self._lv_toggle_sax()                   # both done → sax_which='both'
        elif lv.get("sax_which") != "both":
            lv["sax_which"] = "both"
            m.axis = self._lv_sax_axis()
            lv["fitted_sax"] = False
            rng = self._lv_common_range() or self._lv_level_range()
            if rng:
                lv["sax"] = 0.5 * (rng[0] + rng[1])
            self._lv_show_sax_both()
        if self._lv_sax_active():
            self._overlay[lv["sax_pane"]].update()

    def _lv_series_meta(self) -> dict:
        h = self._header
        if h is None:
            return {}
        pn = str(getattr(h, "PatientName", "") or "")
        return {
            "patient": pn.split("^")[0].strip() or pn.strip(),
            "date": str(getattr(h, "StudyDate", "")
                        or getattr(h, "AcquisitionDate", "") or ""),
            "series_number": str(getattr(h, "SeriesNumber", "") or ""),
            "series_uid": str(getattr(h, "SeriesInstanceUID", "") or ""),
        }

    def _lv_series_dir(self) -> str:
        import os
        # Prefer the folder recorded at load time (dirname of the series' first
        # file) — reliable for every modality; fall back to the header filename.
        sd = getattr(self, "_src_dir", "") or ""
        if sd and os.path.isdir(sd):
            return sd
        h = self._header
        fn = getattr(h, "filename", None) if h is not None else None
        if fn:
            d = os.path.dirname(str(fn))
            if os.path.isdir(d):
                return d
        return ""

    def _lv_save_dir(self) -> str:
        """Default folder for SAVE / EXPORT / LOAD dialogs = the PARENT of the
        source-data folder (one level ABOVE where the CT series was read)."""
        import os
        d = self._lv_series_dir()
        if not d:
            return d
        parent = os.path.dirname(d.rstrip("\\/"))
        return parent if parent and os.path.isdir(parent) else d

    def _lv_default_stem(self) -> str:
        """Series-named file stem, e.g. 'ARIFIN;20260629_Se006' — the base for
        the .lv.json and the exported _Endo/_Epi/_EndoEpi.stl filenames."""
        import re
        meta = self._lv_series_meta()
        name = meta.get("patient", "")
        date = meta.get("date", "")
        sn = meta.get("series_number", "")
        seno = ""
        if sn:
            try:
                seno = "Se%03d" % int(sn)
            except (TypeError, ValueError):
                seno = "Se" + sn
        stem = ";".join(p for p in (name, date) if p)
        if seno:
            stem = (stem + "_" + seno) if stem else seno
        return re.sub(r'[\\/:*?"<>|]', "_", stem).strip() or "LV_borders"

    def _lv_default_name(self) -> str:
        return self._lv_default_stem() + ".lv.json"

    def _lv_export_stl(self) -> None:
        from PyQt6.QtWidgets import QDialog, QMessageBox
        from multi_dicomviewer.ui.lv_stl_dialog import LVStlExportDialog
        from multi_dicomviewer.core.stl_io import write_stl
        from multi_dicomviewer.core.lv_surface import myocardial_shell_mesh
        import os
        if self._lv is None or self._lv.get("model") is None:
            return
        self._lv_capture_current()
        m = self._lv["model"]
        try:
            m.build()
        except Exception:                                  # noqa: BLE001
            pass
        endo = getattr(m, "endo", None)
        epi = getattr(m, "epi", None)
        if endo is None and epi is None:
            QMessageBox.information(
                self.window(), t("LV EF"),
                t("Trace at least 3 planes and press Calc Vol first — there is "
                  "no surface to export yet."))
            return
        stem = self._lv_default_stem()
        dlg = LVStlExportDialog(self._lv_save_dir(), stem,
                                endo is not None, epi is not None, self.window())
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        ch = dlg.choices()
        outdir = dlg.out_dir() or self._lv_save_dir() or os.getcwd()
        if not os.path.isdir(outdir):
            QMessageBox.warning(self.window(), t("LV EF"),
                                t("Output folder does not exist."))
            return
        jobs = []
        if ch["endo"] and endo is not None:
            jobs.append(("_Endo.stl", [endo.to_mesh(close_base=False)]))
        if ch["epi"] and epi is not None:
            jobs.append(("_Epi.stl", [epi.to_mesh(close_base=False)]))
        if ch["both"]:
            shell = (myocardial_shell_mesh(endo, epi)
                     if (endo is not None and epi is not None) else None)
            if shell is not None:
                jobs.append(("_EndoEpi.stl", [shell]))
            else:                                   # only one surface → open cup
                surf = endo if endo is not None else epi
                if surf is not None:
                    jobs.append(("_EndoEpi.stl", [surf.to_mesh(close_base=False)]))
        if not jobs:
            return
        written = []
        try:
            for suffix, meshes in jobs:
                path = os.path.join(outdir, stem + suffix)
                write_stl(path, meshes, header="MDV LV " + stem)
                written.append(os.path.basename(path))
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(self.window(), t("LV EF"),
                                t("STL export failed: {err}", err=str(exc)))
            return
        self._lv_result_lines = [t("Exported STL: {f}", f=", ".join(written))]
        self._lv_update_text()
        QMessageBox.information(
            self.window(), t("LV EF"),
            t("Exported to {d}:\n{f}", d=outdir, f="\n".join(written)))

    def _lv_save(self) -> None:
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import json
        import os
        if self._lv is None or self._lv.get("model") is None:
            return
        self._lv_capture_current()
        m = self._lv["model"]
        if not (m.endo_planes or m.epi_planes):
            QMessageBox.information(self.window(), t("LV EF"),
                                    t("No borders to save yet."))
            return
        # No valid volume yet → ask whether to save without it or compute first.
        has_vol = bool(self._lv.get("vol_done")
                       and self._lv.get("vol_endo_ml") is not None)
        if not has_vol:
            box = QMessageBox(self.window())
            box.setWindowTitle(t("LV EF"))
            box.setIcon(QMessageBox.Icon.Question)
            box.setText(t("Save without volume data?"))
            b_no = box.addButton(t("Save without volume"),
                                 QMessageBox.ButtonRole.AcceptRole)
            b_yes = box.addButton(t("Calculate volume, then save"),
                                  QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is b_yes:
                self._lv_compute_volume()      # runs CalcVol (blocks on its dialog)
            elif clicked is not b_no:
                return                         # Cancel / closed → abort the save
        d = self._lv_save_dir()
        default = os.path.join(d, self._lv_default_name()) if d \
            else self._lv_default_name()
        path, _ = QFileDialog.getSaveFileName(
            self.window(), t("Save LV borders"), default,
            "LV (*.lv.json);;JSON (*.json)")
        if not path:
            return
        data = m.to_dict()
        data["series"] = self._lv_series_meta()
        # Persist the computed volume (only while a VALID result is showing —
        # vol_done is cleared on any edit) so Load can redisplay it.
        if self._lv.get("vol_done") and self._lv.get("vol_endo_ml") is not None:
            data["volume"] = {"endo_ml": float(self._lv["vol_endo_ml"]),
                              "myo_ml": (None if self._lv.get("vol_myo_ml") is None
                                         else float(self._lv["vol_myo_ml"]))}
        self._lv_stamp_axis_def(data)        # apex→MV-centre long-axis marker
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.warning(self.window(), t("LV EF"),
                                t("Save failed: {err}", err=str(exc)))
            return
        # Keep the volume readout on screen after saving (append the saved note,
        # don't replace it) so the result stays visible.
        note = t("Saved: {p}", p=os.path.basename(path))
        lines = []
        if self._lv.get("vol_done") and self._lv.get("vol_endo_ml") is not None:
            lines.append(t("LV cavity volume: {v:.1f} mL",
                           v=float(self._lv["vol_endo_ml"])))
            if self._lv.get("vol_myo_ml") is not None:
                lines.append(t("Myocardial volume: {v:.1f} mL",
                               v=float(self._lv["vol_myo_ml"])))
        lines.append(note)
        self._lv_result_lines = lines
        self._lv_update_text()

    def _lv_load(self) -> None:
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from multi_dicomviewer.core.lv_measure import LVModel
        import json
        if self._vol is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self.window(), t("Load LV borders"), self._lv_save_dir(),
            "LV (*.lv.json *.lvef.json);;JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._lv_warn_if_axis_stale(data)
            model = LVModel.from_dict(data)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.warning(self.window(), t("LV EF"),
                                t("Load failed: {err}", err=str(exc)))
            return
        if model.axis is None:
            QMessageBox.warning(self.window(), t("LV EF"),
                                t("The file has no LV axis."))
            return
        saved = (data.get("series") or {}).get("series_uid", "")
        cur = self._lv_series_meta().get("series_uid", "")
        if saved and cur and saved != cur:
            if QMessageBox.question(
                    self.window(), t("LV EF"),
                    t("This file was saved for a DIFFERENT series — the borders "
                      "may not line up. Load anyway?")) != \
                    QMessageBox.StandardButton.Yes:
                return
        self._lv_apply_model(model, volume=data.get("volume"))

    def _lv_apply_model(self, model, volume=None) -> None:
        if self._meas_on:
            self._meas_btn.setChecked(False)
            self._toggle_measure()
        self._lvv_deactivate()            # contour LV and LV Vol are exclusive
        self._lv = {"model": model, "phase": "contour", "plane_idx": 0,
                    "target": None, "pane": "B", "sax": None,
                    "pass": "epi" if model.epi_axis is not None else "endo",
                    "prev_side": self.current_side()}
        self._lv_reset_undo()
        self._lv_btn.setChecked(True)
        self._lv_enter_contour()
        self._lv_rebuild_measures()
        self._lv_apply_target(self._lv["pass"])
        self._lv_show_plane()
        self._lv_result_lines = [
            t("Loaded borders: endo {ne} / epi {nep} planes",
              ne=len(model.endo_planes), nep=len(model.epi_planes))]
        if (len(model.endo_contours) >= 3 or len(model.epi_contours) >= 3):
            self._lv_sax_btn.setChecked(True)
            self._lv_toggle_sax()
        # Redisplay a SAVED volume result (after SAX entry, which clears the
        # result panel) and light CalcVol blue — so Load shows the computed EF.
        if volume and self._lv is not None:
            lines = [t("Loaded borders: endo {ne} / epi {nep} planes",
                       ne=len(model.endo_planes), nep=len(model.epi_planes))]
            ev, mv = volume.get("endo_ml"), volume.get("myo_ml")
            if ev is not None:
                lines.append(t("LV cavity volume: {v:.1f} mL", v=float(ev)))
                self._lv["vol_endo_ml"] = float(ev)
            if mv is not None:
                lines.append(t("Myocardial volume: {v:.1f} mL", v=float(mv)))
                self._lv["vol_myo_ml"] = float(mv)
            if ev is not None:
                self._lv["vol_done"] = True
                self._lv_result_lines = lines
                self._lv_sync_buttons()
        self._lv_update_text()

    def _lv_rebuild_measures(self) -> None:
        lv = self._lv
        pane = lv["pane"]
        m = lv["model"]
        self._measures[pane] = [mm for mm in self._measures[pane]
                                if mm.get("_lv") is None]
        angs = m.plane_angles()
        for which, store, col in (("endo", m.endo_planes, "#ff4040"),
                                  ("epi", m.epi_planes, "#40c040")):
            for phi, pts3d in store.items():
                arr = np.asarray(pts3d, float).reshape(-1, 3)
                if len(arr) < 2:
                    continue
                idx = min(range(len(angs)), key=lambda i: min(
                    abs(angs[i] - phi), abs(angs[i] - phi + 360.0),
                    abs(angs[i] - phi - 360.0)))
                p3 = [arr[j].copy() for j in range(len(arr))]
                self._meas_seq = getattr(self, "_meas_seq", 0) + 1
                self._measures[pane].append({
                    "id": self._meas_seq, "type": "polyline",
                    "pts3d": p3,
                    "pts": [self._world3d_to_out(pane, P) for P in p3],
                    "color": col, "smooth": True, "_lv": (idx, which)})

    def _lv_exit_confirm(self) -> None:
        """'Exit LV' button → confirm before leaving LV analysis."""
        from PyQt6.QtWidgets import QMessageBox
        if self._lv is None:
            return
        if QMessageBox.question(
                self.window(), t("LV EF"),
                t("Exit LV analysis? Unsaved borders/results are kept only if "
                  "you Saved them.")) != QMessageBox.StandardButton.Yes:
            return
        self._lv_exit()

    def _lv_exit(self, from_toggle=False) -> None:
        self._lv_region_reset()
        if self._lv is not None and self._lv.get("phase") == "contour":
            self._lv_capture_current()
        # Stash the traced EPI surface so LV Vol mode can use it as the outer
        # bound (Epi border → myocardial envelope). Only when Epi was traced.
        try:
            if self._lv is not None:
                model = self._lv["model"]
                if (model.epi_axis is not None
                        and len(model.epi_contours) >= 3):
                    model.build()
                    if model.epi is not None:
                        self._lvv_epi_surf = model.epi
                        self._lvv_epi_apex = np.asarray(
                            model.epi_axis.apex, float)
                        d = model.to_dict()
                        try:
                            spacing = max(0.5, float(min(self._dims)))
                            epi_ml = model.volume_ml(spacing, "epi")
                            if epi_ml is not None:
                                d.setdefault("volume", {})["epi_ml"] = float(
                                    epi_ml)
                                self._lvv_epi_ml = float(epi_ml)
                        except Exception:               # noqa: BLE001
                            pass
                        self._lvv_epi_model_dict = d
        except Exception:                               # noqa: BLE001
            pass
        # Keep a hand-edited Endo (Manual-Endo) so it is retained across HU
        # changes; a cleared Endo drops it (→ re-seed next Manual-Endo).
        if self._lv is not None and getattr(self, "_lv_endo_manual_mode", False):
            self._lv_stash_manual_endo(self._lv["model"])
            self._lv_endo_manual_mode = False
        self._lv_reset_undo()
        for k in ("A", "B"):
            self._measures[k] = [m for m in self._measures[k]
                                 if m.get("_lv") is None]
        self._lv = None
        self._lv_result_lines = []
        self._lv_wall = False
        self._lv_wall_btn.setChecked(False)
        self._lv_sax_btn.setChecked(False)
        self._lv_btn.setChecked(False)      # internal mode flag off
        self._lv_sync_buttons()             # reset colours + grey out the bar
        self._lv_plane_lbl.setText("0/6")
        if self._meas_on:
            self._meas_btn.setChecked(False)
            self._toggle_measure()
        self.set_side("Bi")
        self._init_frames()
        self._view_initial = True
        self._lv_update_text()
        # Re-enable the tools/controls contour LV greyed out (esp. the Slab(mm)
        # spin): _refresh_tool_availability only ever DISABLED them in LV and
        # nothing re-ran it on the way out, so entering LV Vol afterwards left
        # the slab stuck disabled.
        self._refresh_tool_availability()
        self._refresh(reset_cam=True)
        self._lv_redraw_all()

    def _lv_update_text(self) -> None:
        for k in ("A", "B"):
            self._redraw_meas(k)

    def _lv_status_lines(self) -> list:
        lv = self._lv
        if lv is None:
            return []
        lines = list(getattr(self, "_lv_result_lines", []))
        pas = lv.get("pass")
        if pas is None:
            lines.append(t("LV EF — choose Endo or Epi to start a pass"))
            return lines
        name = t("Endo (lumen)") if pas == "endo" else t("Epi (myocardial)")
        ph = lv.get("phase")
        if ph == "align":
            lines.append(t("LV EF [{p} pass] — align the {p} long-axis view, "
                           "then press 'Set axis'", p=name))
            return lines
        if ph == "ready":
            lines.append(t("LV EF [{p} pass] — axis set. Final Zoom/Move, then "
                           "press 'Trace'", p=name))
            return lines
        if ph == "apex":
            lines.append(t("LV EF [{p} pass] — click the {p} apex "
                           "(Shift-click to adjust the view first)", p=name))
            return lines
        if ph == "contour":
            m = lv["model"]
            head = (t("tracing Endo (red) — double-click to finish")
                    if pas == "endo"
                    else t("tracing Epi (green) — double-click to finish"))
            lines.append(
                t("LV EF [{p} pass] — {head}\ncaptured: endo {ne} / epi {nep} "
                  "meridians", p=name, head=head,
                  ne=len(m.endo_contours), nep=len(m.epi_contours)))
        return lines

    @staticmethod
    def _circle_poly(cx, cy, r, n=20):
        return [(cx + r * math.cos(2.0 * math.pi * i / n),
                 cy + r * math.sin(2.0 * math.pi * i / n))
                for i in range(n + 1)]

    def _lv_view_half(self, key):
        ps = float(self._ps[key])
        w = max(1, self.pane[key].canvas.width())
        h = max(1, self.pane[key].canvas.height())
        return ps * (w / h), ps

    def _lv_edge_xy(self, key, ux, uy, frac=0.9):
        hw, hh = self._lv_view_half(key)
        nrm = math.hypot(ux, uy) or 1.0
        ux, uy = ux / nrm, uy / nrm
        tx = hw / abs(ux) if abs(ux) > 1e-6 else 1e18
        ty = hh / abs(uy) if abs(uy) > 1e-6 else 1e18
        d = frac * min(tx, ty)
        return ux * d, uy * d

    def _lv_handle_screen(self, key):
        """Screen px (hx, hy) of the SAX line's ○ grab handle for *key* — the
        + end of the line, always pinned JUST INSIDE the pane edge so it stays
        on-screen and reachable at any zoom / pan (a world-fixed position drifts
        off when the pane is panned/zoomed). None if this pane has no SAX line."""
        lv = self._lv
        if (lv is None or lv.get("model") is None
                or lv["model"].axis is None or lv.get("sax") is None):
            return None
        ax = lv["model"].axis
        if key == lv.get("pane"):
            _, y = self._world3d_to_out(key, ax.apex + float(lv["sax"]) * ax.axis)
            cx, cy = self._world_to_screen(key, 0.0, y)
            ex, ey = self._world_to_screen(key, 1.0, y)     # +x world direction
        elif key == lv.get("sax_pane"):
            angs = lv["model"].plane_angles()
            md = ax.meridian_dir(angs[lv["plane_idx"] % len(angs)])
            u, v, _n = self._frame[key]
            mx, my = float(np.dot(md, u)), float(np.dot(md, v))
            nrm = math.hypot(mx, my) or 1.0
            cx, cy = self._world_to_screen(key, 0.0, 0.0)
            ex, ey = self._world_to_screen(key, mx / nrm, my / nrm)
        else:
            return None
        dx, dy = ex - cx, ey - cy
        L = math.hypot(dx, dy) or 1.0
        dx, dy = dx / L, dy / L
        W = max(1, self.pane[key].canvas.width())
        H = max(1, self.pane[key].canvas.height())
        m = 14.0                                    # inset = handle radius + pad
        # March the line from its centre in the + direction to the pane boundary.
        tx = (((W - m) - cx) / dx if dx > 1e-9
              else ((m - cx) / dx if dx < -1e-9 else 1e18))
        ty = (((H - m) - cy) / dy if dy > 1e-9
              else ((m - cy) / dy if dy < -1e-9 else 1e18))
        t = max(0.0, min(tx, ty))
        hx = min(W - m, max(m, cx + dx * t))        # clamp inside as a safety net
        hy = min(H - m, max(m, cy + dy * t))
        return hx, hy

    # ==================================================================
    # Short-axis (CPR). Shared state + control logic live in CPRMixin; the
    # backend-specific pieces are here: the scrubber bar, entering/leaving the
    # mode, the live oblique GPU slice (material.plane + oriented camera) and
    # the QPainter overlay. Mirrors the VTK viewer feature-for-feature.
    # ==================================================================
    def _build_cpr_bar(self) -> QWidget:
        """Bottom scrubber for short-axis mode: scroll the cross-section along
        the traced vessel, reverse the direction, and exit. Hidden unless CPR
        is active."""
        self._cpr_wrap = QWidget()
        row = QHBoxLayout(self._cpr_wrap)
        row.setContentsMargins(8, 2, 8, 2)
        cap = QLabel(t("Short-axis:"))
        f = cap.font(); f.setBold(True); cap.setFont(f)
        self._cpr_cap = cap
        row.addWidget(cap)
        self._cpr_rev_btn = FitButton(t("Reverse"))
        self._cpr_rev_btn.setCheckable(True)
        self._cpr_rev_btn.setHelpToolTip(
            t("Reverse the scroll order to distal->proximal (match an IVUS "
              "pull-back). Cross-section content is unchanged."))
        self._cpr_rev_btn.clicked.connect(self._cpr_toggle_reverse)
        row.addWidget(self._cpr_rev_btn)
        self._cpr_slider = QSlider(Qt.Orientation.Horizontal)
        self._cpr_slider.setMinimum(0)
        self._cpr_slider.setMaximum(0)
        self._cpr_slider.setMinimumHeight(26)
        self._cpr_slider.setStyleSheet(_SEEK_SLIDER_QSS)
        self._cpr_slider.valueChanged.connect(self._cpr_set_index)
        row.addWidget(self._cpr_slider, 1)
        self._cpr_lbl = QLabel("")
        self._cpr_lbl.setMinimumWidth(170)
        fl = self._cpr_lbl.font(); fl.setBold(True); self._cpr_lbl.setFont(fl)
        row.addWidget(self._cpr_lbl)
        self._cpr_exit_btn = FitButton(t("Exit CPR"))
        self._cpr_exit_btn.setHelpToolTip(
            t("Leave short-axis mode and restore the normal MPR"))
        self._cpr_exit_btn.clicked.connect(self._exit_cpr)
        row.addWidget(self._cpr_exit_btn)
        self._cpr_wrap.setVisible(False)
        return self._cpr_wrap

    def _enter_cpr(self, which, mi):
        """Turn polyline *mi* on pane *which* into a vessel centreline and put
        pane A into short-axis (cross-section) scroll mode. Mirrors the VTK
        viewer's _enter_cpr."""
        m = self._measures[which][mi]
        u, v, nrm = self._axes_for(which)
        p3 = m.get("pts3d")
        if p3 and len(p3) >= 2:
            ctrl = [np.asarray(P, dtype=float) for P in p3]
        else:
            pts2d = self._outline(m)
            if len(pts2d) < 2:
                return
            o = self._pc[which]
            ctrl = [np.asarray(o, float) + float(x) * u + float(y) * v
                    for (x, y) in pts2d]
        step = max(1e-3, min(self._dims))
        cl = CenterLine.from_points(ctrl, step_mm=step)
        if cl.n < 2:
            return
        fu, fv = cl.frames(ref_up=nrm)
        fu = -fu                                   # view proximal->distal
        self._cpr = {
            "cl": cl, "u": fu, "v": fv, "idx": cl.n // 2,
            "u0": fu.copy(), "v0": fv.copy(),
            "T": np.eye(2), "rot": 0.0, "reversed": False,
            "half": 25.0, "src": which, "src_mi": mi,
            "ref_up": np.asarray(nrm, float),
        }
        self.set_side("Bi")                        # A = cross-section, src = map
        for b in self._t2d_btns:                   # Rt90/Lt90/Flip work in CPR
            b.setEnabled(True)
        self._cpr_rev_btn.setChecked(False)
        self._cpr_sync_bar()
        self._refresh(reset_cam=True)

    def _exit_cpr(self):
        """Leave short-axis mode and restore pane A's normal MPR."""
        if self._cpr is None:
            return
        self._cpr = None
        self._cpr_drag = None
        self._cpr_rot_prev = None
        self._cpr_marker_pts = []
        self._cpr_wrap.setVisible(False)
        for b in self._t2d_btns:                   # work in 2-D and 3-D MPR
            b.setEnabled(True)
        self._init_frames()                        # rebuild pane A's MPR frame
        self._refresh(reset_cam=True)

    def _cpr_sync_bar(self):
        c = self._cpr
        if c is None:
            return
        cl = c["cl"]
        d = self._cpr_disp(c["idx"])
        self._cpr_wrap.setVisible(True)
        self._cpr_slider.blockSignals(True)
        self._cpr_slider.setMaximum(cl.n - 1)
        self._cpr_slider.setValue(d)
        self._cpr_slider.blockSignals(False)
        pos_mm = float(cl.arclen[c["idx"]])
        self._cpr_lbl.setText(
            f"{d + 1} / {cl.n}   ({pos_mm:.1f} / {cl.length_mm:.1f} mm)")

    # ---- live oblique slice (GPU) ----
    def _render_cpr_pane(self, p):
        """Render pane A as the short-axis cross-section: an oblique GPU slice
        (plane normal = tangent) with the camera oriented to (u, v). Reuses the
        same VolumeSliceMaterial + OrthographicCamera path the MPR panes use."""
        o, u, v, tg = self._cpr_frame()
        n = _norm(np.asarray(tg, float))
        p.material.plane = (float(n[0]), float(n[1]), float(n[2]),
                            float(-np.dot(n, o)))
        if self._color:
            p.material.clim = (_HU_LO, _HU_HI)
            p.material.map = self._lut_texture()
        else:
            self._gray_material(p)
        p.mesh.visible = True
        self._mip_img["A"] = None
        self._config_cpr_cam(p, o, u, v)
        p.render()

    def _config_cpr_cam(self, p, o, u, v):
        """Orient/position pane A's camera to view the cross-section face-on
        with U=right, V=up, FOV = +-half mm (mirrors _config_cam)."""
        ur = _norm(np.asarray(u, float))
        vr = _norm(np.asarray(v, float))
        w = _norm(np.cross(ur, vr))
        R = np.column_stack([ur, vr, w]).astype(np.float64)
        p.cam.local.rotation = la.quat_from_mat(R)
        p.cam.local.position = np.asarray(o, float) + w * self._cam_off
        ps = max(1e-3, float(self._cpr["half"]))
        pw = max(1, p.canvas.width())
        ph = max(1, p.canvas.height())
        p.cam.height = 2.0 * ps
        p.cam.width = 2.0 * ps * (pw / ph)
        p.cam.depth_range = (0.1, 4.0 * self._cam_off + self._diag)

    # ---- control-point markers (edit the pseudo-centres in the section) ----
    def _cpr_marker_geom(self):
        """Recompute the single nearest control-point marker's in-plane offset
        (du,dv) and cache it in _cpr_marker_pts for hit-testing. Returns the
        list of (ctrl_idx, (du,dv))."""
        self._cpr_marker_pts = []
        p3 = self._cpr_ctrl_pts3d()
        if not p3:
            return self._cpr_marker_pts
        o, u, vv, n = self._cpr_frame()
        dns = [abs(float(np.dot(np.asarray(P, float) - o, n))) for P in p3]
        near = int(np.argmin(dns))
        P = np.asarray(p3[near], float)
        du = float(np.dot(P - o, u))
        dv = float(np.dot(P - o, vv))
        self._cpr_marker_pts.append((near, (du, dv)))
        return self._cpr_marker_pts

    def _cpr_grab(self, sx, sy) -> bool:
        """A press near a control-point marker on the section starts a drag."""
        if self._cpr is None:
            return False
        for ci, (du, dv) in self._cpr_marker_geom():
            mx, my = self._world_to_screen("A", du, dv)
            if (mx - sx) ** 2 + (my - sy) ** 2 <= 14.0 ** 2:
                self._cpr_drag = ci
                return True
        return False

    def _cpr_drag_move(self, sx, sy):
        """Move the grabbed control point IN the cross-section plane (its along-
        vessel depth is preserved); the map-pane trace follows live."""
        if self._cpr_drag is None:
            return
        p3 = self._cpr_ctrl_pts3d()
        ci = self._cpr_drag
        if not p3 or not (0 <= ci < len(p3)):
            return
        o, u, vv, n = self._cpr_frame()
        du, dv = self._disp_to_world("A", sx, sy)
        dn = float(np.dot(np.asarray(p3[ci], float) - o, n))     # keep depth
        p3[ci] = np.asarray(o, float) + du * u + dv * vv + dn * n
        self._overlay["A"].update()
        self._redraw_meas(self._cpr["src"])        # map-pane trace follows

    def reload_display_quality(self) -> None:
        """Re-read the app-wide display-quality prefs (the shell calls this after
        the Settings dialog): apply the chosen CT quality mode live and repaint.
        CT image quality is set only from Settings ▸ CT Image Quality."""
        self._dq = settings.load_display_quality()
        mode = self._dq.get("ct_quality_mode", "adaptive")
        if mode != self._ct_quality:
            self._ct_quality = mode
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

    def _invert_texture(self):
        """1-D white→black gray ramp with the W/L window baked in, over the
        full HU domain [_HU_LO,_HU_HI] — the same mechanism (and clim) as the
        HU colormap, which is the proven-good path on Metal.

        Why not just swap the clim ends: a reversed clim (low > high) is
        undefined behaviour for the material's clamp, so it renders garbage on
        gradients. Why 4096 entries: the window is baked into the ramp, so a
        narrow window must still resolve >256 grey levels or it would band."""
        key = (self._win, self._lvl)
        if key != self._inv_key:
            n = 4096
            hu = _HU_LO + (_HU_HI - _HU_LO) * np.arange(n) / (n - 1)
            g = np.clip((hu - (self._lvl - self._win / 2.0))
                        / max(1e-6, self._win), 0.0, 1.0).astype(np.float32)
            g = (1.0 - g).astype(np.float32)          # the negative
            arr = np.stack([g, g, g, np.ones_like(g)], axis=1)
            self._inv_tex = gfx.Texture(arr, dim=1)
            self._inv_key = key
        return self._inv_tex

    def _gray_material(self, p):
        """Apply the plain (non-colormap) grayscale mapping to a pane material.
        WB reverse renders through an inverted gray ramp; the normal path is
        left exactly as it was (direct clim, no map)."""
        if self._invert:
            p.material.clim = (_HU_LO, _HU_HI)
            p.material.map = self._invert_texture()
        else:
            p.material.map = None
            p.material.clim = (self._lvl - self._win / 2.0,
                               self._lvl + self._win / 2.0)

    def _toggle_color(self):
        self._color = self._cmap_btn.isChecked()
        self._refresh()

    def _toggle_invert(self):
        """WB reverse: invert the grayscale (black↔white). Applies to every
        pane, including the short-axis (all use the gray ramp)."""
        self._invert = self._invert_btn.isChecked()
        self._refresh()

    def _open_setting(self, parent=None, modal=False):
        """Open the HU colour-map editor. *parent*/*modal* let the shell open it
        ON TOP of (and modal to) the Settings popup — otherwise a Settings-modal
        dialog would sit in front of it and block its controls. A fresh dialog
        is built each time (the viewer's bands are the source of truth)."""
        dlg = _ColorMapDialog(self._bands, self._opacity,
                              self._apply_colormap, self._win, self._lvl,
                              self._cmap_smooth_mm, parent or self)
        self._cmap_dlg = dlg
        if modal:
            dlg.setWindowModality(Qt.WindowModality.WindowModal)
            dlg.exec()
            self._cmap_dlg = None
        else:
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()

    def _apply_colormap(self, bands, opacity, smooth_mm=None):
        self._bands = [dict(b) for b in bands]
        self._opacity = float(opacity)
        if smooth_mm is not None:
            self._cmap_smooth_mm = float(smooth_mm)
        if not self._color:
            self._color = True
            self._cmap_btn.setChecked(True)
        self._refresh()
        # The colour map is global: persist it and let the shell mirror it onto
        # every other CT pane.
        settings.save_ct_colormap(self._bands, self._opacity,
                                  self._cmap_smooth_mm)
        self.colormap_changed.emit([dict(b) for b in self._bands],
                                   self._opacity, self._cmap_smooth_mm)

    def apply_global_colormap(self, bands, opacity, smooth_mm=None):
        """Adopt a colour map edited in ANOTHER CT pane (shell propagation).
        Updates the bands + any open editor and redraws, but does NOT force
        colour mode on, persist, or re-emit (which would loop)."""
        self._bands = [dict(b) for b in bands]
        self._opacity = float(opacity)
        if smooth_mm is not None:
            self._cmap_smooth_mm = float(smooth_mm)
        if self._cmap_dlg is not None:
            self._cmap_dlg.set_bands(self._bands, self._opacity)
        if self._color:
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
                            np.array([0.0, -1.0, 0.0]))
            self._sync_slab_spin()
            self._view_initial = True
            # Re-apply the current mode (re-locks 2-D / restores dual MPR) and
            # refits the camera.
            self._set_mode(self._mode, reset_cam=True)
        else:
            before = self._view_snapshot()
            self._win, self._lvl = self._win0, self._lvl0
            self._refresh()
            self._undo_view(before, self._view_snapshot())   # Ctrl+Z / Ctrl+Y

    def _apply_preset(self, name):
        if name in CT_WL_PRESETS:
            before = self._view_snapshot()
            self._win, self._lvl = (float(x) for x in CT_WL_PRESETS[name])
            self._refresh()
            self._undo_view(before, self._view_snapshot())   # Ctrl+Z / Ctrl+Y

    def _key_toggle_color(self):
        """C shortcut → toggle the HU colormap (keep the toolbar button synced)."""
        self._cmap_btn.setChecked(not self._cmap_btn.isChecked())
        self._toggle_color()
