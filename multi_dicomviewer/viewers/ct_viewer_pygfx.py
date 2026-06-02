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

import numpy as np
import pygfx as gfx
import pylinalg as la
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
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
from multi_dicomviewer.core.dicom_tags import overlay_lines
from multi_dicomviewer.core.measure_geom import (
    angle_at as _angle_at,
    central_arc_angle as _central_arc_angle,
    dist as _dist,
    ellipse_cab as _ellipse_cab,
    major_minor as _major_minor,
    point_in_poly as _point_in_poly,
    poly_area as _poly_area,
    seg_dist as _seg_dist,
    smooth_closed as _smooth_closed,
    smooth_open as _smooth_open,
)
from multi_dicomviewer.ui.viewer_base import AbstractViewer

#: SPIN sign. +1.0 matches the rotation direction expected on the Mac build.
_SPIN_SIGN = 1.0

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

        def draw_outline(pts, rgb, width=1.5):
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(*rgb), width))
            if len(pts) >= 2:
                p.drawPolyline(poly(pts))

        def draw_dashed(seg, rgb):
            pen = QPen(QColor(*rgb), 2.2)
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

        for mi, m in enumerate(v._measures[key]):
            rgb = _hex_to_rgb(m.get("color"))
            draw_outline(v._outline(m), rgb)
            if m["type"] in ("ellipse", "polygon"):
                maj, mnr, _, _ = _major_minor(m)
                for seg in (maj, mnr):
                    if seg is not None:
                        draw_dashed(seg, rgb)
            ca = m.get("center_angle")
            if ca and ca.get("pts"):
                centre = v._shape_center(m)
                for q in ca["pts"]:
                    draw_dashed((centre, q), rgb)
                dots(ca["pts"], QColor(255, 140, 0), 5.0)   # orange picks
            idle = [q for vi, q in enumerate(v._handles(m))
                    if not (mi == edit_mi and vi == edit_vi)]
            dots(idle, QColor(255, 217, 0), 4.0)            # yellow handles
            if mi == edit_mi and 0 <= edit_vi < len(m["pts"]):
                dots([m["pts"][edit_vi]], QColor(59, 219, 90), 7.0)  # green
            # numeric id label at the anchor
            p.setPen(QColor(255, 217, 0))
            fb = QFont("monospace", 14)
            fb.setBold(True)
            p.setFont(fb)
            ax, ay = v._world_to_screen(key, *v._anchor(m))
            p.drawText(QPointF(ax + 6, ay - 6), str(m["id"]))

        d = v._draft
        if d and d["pane"] == key and d["pts"]:
            cyan = _hex_to_rgb(None)
            if d["type"] == "ellipse" and len(d["pts"]) >= 2:
                p0, p1 = d["pts"][0], d["pts"][1]
                cx, cy = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
                a, b = abs(p1[0] - p0[0]) / 2, abs(p1[1] - p0[1]) / 2
                draw_outline([(cx + a * math.cos(t), cy + b * math.sin(t))
                              for t in (i * 2 * math.pi / 48
                                        for i in range(49))], cyan)
            else:
                draw_outline(list(d["pts"]), cyan)
            dots(list(d["pts"]), QColor(255, 217, 0), 4.0)

        # per-measure result strings, stacked top-right
        lines = v._metrics.get(key, [])
        if lines:
            p.setPen(QColor(102, 255, 153))
            p.setFont(QFont("monospace", 11))
            p.drawText(QRectF(0, 4, w - 6, h * 0.5),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                       "\n".join(lines))

    # -- corner info text + angio readout ----------------------------------
    def _paint_info(self, p, key, w, h):
        v = self._v
        p.setPen(QColor(102, 255, 153))         # green like vtk corner text
        f = QFont("monospace", 12)
        p.setFont(f)
        head = overlay_lines(v._header, v._tag_keywords, anonymized=v._anon)
        if head:
            p.drawText(QRectF(6, 4, w - 12, h * 0.6),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                       "\n".join(head))
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._vol = None
        self._header = None
        self._pbasis = np.eye(3)
        self._tag_keywords: list[str] = []
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
        self._mip_img = {"A": None, "B": None}   # slab-MIP QImage per pane
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
        lay.addLayout(self._build_toolbar())
        self._measure_bar = self._build_measure_bar()
        self._measure_bar.setVisible(False)
        lay.addWidget(self._measure_bar)
        lay.addLayout(imgrow, 1)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
        c.add_event_handler(lambda ev, k=key: self._on_key(k, ev), "key_down")

    def _on_key(self, key, ev):
        """Canvas keyboard shortcuts (rendercanvas key_down). The canvas has
        focus, so Qt's keyPressEvent on the viewer never fires — route here."""
        k = ev.get("key", "")
        kl = k.lower() if isinstance(k, str) else k
        self._set_active(key)
        if kl == "c":
            self._cmap_btn.setChecked(not self._cmap_btn.isChecked())
            self._toggle_color()
            return
        tool = {"z": "ZOOM", "v": "MOVE", "s": "SPIN", "g": "PAGING",
                "w": "WL", "r": "ROTATE", "t": "THICK"}.get(kl)
        if tool:
            self._set_tool(tool)

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
        # Pressing ON the crosshair grabs it (MOVE/ROTATE), overriding tool.
        self._cross_grab = (self._drag_btn == 1
                            and self._cross_press(key, x, y))

    def _on_move(self, key, ev):
        x, y = ev["x"], ev["y"]
        if self._meas_on:
            if self._meas_drag:
                self._measure_drag(key, x, y)
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
        self.pane[key].render()

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

        row.addWidget(QLabel("Slab(mm):"))
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

        setting = QPushButton("Setting")
        setting.setToolTip("HU colour-map settings (band colour, HU range, opacity)")
        setting.clicked.connect(self._open_setting)
        row.addWidget(setting)

        reset = QPushButton("Reset")
        reset.clicked.connect(self._reset)
        row.addWidget(reset)

        row.addWidget(QLabel("W/L preset:"))
        self._preset = QComboBox()
        self._preset.addItems(list(CT_WL_PRESETS.keys()))
        self._preset.currentTextChanged.connect(self._apply_preset)
        row.addWidget(self._preset)

        tags = QPushButton("DICOM Tags…")
        tags.clicked.connect(self.tags_requested.emit)
        row.addWidget(tags)
        row.addStretch(1)
        self._set_tool("PAGING")
        return row

    def _set_tool(self, name):
        self._tool = name
        for n, b in self._tool_btns.items():
            b.setChecked(n == name)
            b.setStyleSheet("background:#c0392b;color:white;" if n == name else "")

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
        self._tag_keywords = list(keywords or [])
        self._refresh()

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
    def _slab_mip_hu(self, key, iw, ih):
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
        nplanes = int(max(1, min(64, round(th / step))))
        offs = (np.linspace(-th / 2.0, th / 2.0, nplanes)
                if nplanes > 1 else np.array([0.0]))
        mip = np.full((ih, iw), -np.inf, np.float32)
        for t in offs:
            hu = _trilinear_sample(self._vol, (bx + t * n[0]) / sx,
                                   (by + t * n[1]) / sy, (bz + t * n[2]) / sz)
            mip = np.maximum(mip, hu.astype(np.float32))
        return mip

    def _build_slab_qimage(self, key):
        """Render the slab MIP for a pane to a viewport-filling RGB QImage
        (W/L or HU colormap applied CPU-side)."""
        pane = self.pane[key]
        pw = max(1, pane.canvas.width())
        ph = max(1, pane.canvas.height())
        iw = min(pw, 480)
        ih = max(1, int(round(iw * ph / pw)))
        mip = self._slab_mip_hu(key, iw, ih)
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

    def _refresh(self, reset_cam=False):
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
            if self._thick[key] > 0:
                p.mesh.visible = False
                self._mip_img[key] = self._build_slab_qimage(key)
            else:
                p.mesh.visible = True
                self._mip_img[key] = None
            p.render()
            self._overlay[key].update()

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
            mv = n * dy * min(self._dims)
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
            u = _rotate(_rotate(u, v, dx * 0.5), u, dy * 0.5)
            v = _rotate(_rotate(v, v, dx * 0.5), u, dy * 0.5)
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
        self._refresh()

    def _wheel(self, which, delta):
        if self._vol is None:
            return
        _, _, n = self._axes_for(which)
        mv = n * (1 if delta > 0 else -1) * min(self._dims)
        self._center = self._center + mv
        self._pc[which] = self._pc[which] + mv     # page only this pane
        self._clamp_center()
        self._view_initial = False
        self._refresh()

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
            self._refresh()
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
        self._refresh()

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
        for k, b in self._meas_btns.items():
            b.setChecked(k == key)
            b.setStyleSheet("background:#1f77b4;color:white;" if k == key else "")

    def _measure_clear(self):
        self._measures = {"A": [], "B": []}
        self._draft = None
        self._edit = None
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
            a, b, c = m["pts"][:3]
            return [b, a, c]
        cx, cy, a, b = self._ellipse_cab(m)
        return [(cx + a * math.cos(th), cy + b * math.sin(th))
                for th in (i * 2 * math.pi / 48 for i in range(49))]

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
        return m["pts"][0]

    def _shape_center(self, m):
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
            return f"#{m['id']} Angle: {_angle_at(pts[0], pts[1], pts[2]):.1f}°"
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
        if m["type"] == "ellipse":
            self._set_ellipse_handle(m, e["vi"], w)
        else:
            m["pts"][e["vi"]] = w
        self._redraw_geom(e["key"])

    def _measure_release(self):
        if self._edit:
            key = self._edit["key"]
            self._edit = None
            self._redraw_meas(key)

    def _set_ellipse_handle(self, m, vi, w):
        pts = [list(q) for q in m["pts"]]    # lft,rgt,top,bot
        if vi == 0:
            pts[0][0] = w[0]
        elif vi == 1:
            pts[1][0] = w[0]
        elif vi == 2:
            pts[2][1] = w[1]
        else:
            pts[3][1] = w[1]
        cx = (pts[0][0] + pts[1][0]) / 2.0
        cy = (pts[2][1] + pts[3][1]) / 2.0
        pts[0][1] = pts[1][1] = cy
        pts[2][0] = pts[3][0] = cx
        m["pts"] = [tuple(q) for q in pts]

    def _commit_draft(self):
        d = self._draft
        self._draft = None
        if d is None or len(d["pts"]) < 2:
            return
        if d["type"] == "ellipse":
            p0, p1 = d["pts"][0], d["pts"][1]
            cx, cy = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            a, b = abs(p1[0] - p0[0]) / 2, abs(p1[1] - p0[1]) / 2
            pts = [(cx - a, cy), (cx + a, cy), (cx, cy - b), (cx, cy + b)]
        elif d["type"] == "line":
            pts = d["pts"][:2]
        else:
            pts = list(d["pts"])
        self._meas_seq += 1
        self._measures[d["pane"]].append(
            {"id": self._meas_seq, "type": d["type"], "pts": pts})
        self._redraw_meas(d["pane"])

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
        hit = self._pick_handle(which, sx, sy)
        if hit is not None:
            self._handle_right(which, hit, sx, sy)
            return
        mi = self._pick_measure(which, sx, sy)
        if mi is None:
            return
        self._outline_right(which, mi, sx, sy)

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
        ca["pts"].append(w)
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
            lft, rgt, top, bot = m["pts"]
            m["type"] = "polygon"
            m["pts"] = [lft, top, rgt, bot]
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

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_C:               # C = toggle ColorMap
            self._cmap_btn.setChecked(not self._cmap_btn.isChecked())
            self._toggle_color()
            return
        tool = _TOOL_KEYS.get(e.key())
        if tool:
            self._set_tool(tool)
        else:
            super().keyPressEvent(e)
