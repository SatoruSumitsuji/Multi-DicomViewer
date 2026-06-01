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
from PyQt6.QtCore import QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from rendercanvas.pyqt6 import RenderCanvas

from multi_dicomviewer.config import CT_WL_PRESETS
from multi_dicomviewer.core.dicom_io import LoadedSeries
from multi_dicomviewer.core.dicom_tags import overlay_lines
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
        if v._cl_on:
            self._paint_cross(p, key, w, h)
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

    # -- corner info text + angio readout ----------------------------------
    def _paint_info(self, p, key, w, h):
        v = self._v
        p.setPen(QColor(102, 255, 153))         # green like vtk corner text
        f = QFont("monospace", 9)
        p.setFont(f)
        head = overlay_lines(v._header, v._tag_keywords, anonymized=v._anon)
        if head:
            p.drawText(QRectF(6, 4, w - 12, h * 0.6),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                       "\n".join(head))
        slab = v._thick[key]
        kind = f"Slab MIP {slab:.1f}mm" if slab > 0 else "MPR (thin)"
        p.drawText(QRectF(6, h - 22, w - 12, 18),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"WW {v._win:.0f}  WL {v._lvl:.0f}")
        p.drawText(QRectF(6, h - 22, w - 12, 18),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"{key}  |  {kind}")
        # angio readout (yellow, bottom-centre) — clinical, always shown
        ang = v._angio_angle(key)
        if ang:
            p.setPen(QColor(255, 230, 0))
            fb = QFont("monospace", 11)
            fb.setBold(True)
            p.setFont(fb)
            p.drawText(QRectF(0, h - 28, w, 22),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                       ang)


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

    def _on_down(self, key, ev):
        self._set_active(key)
        self._drag_btn = ev.get("button")
        self._last = (ev["x"], ev["y"])
        self._spin_prev = None
        # Pressing ON the crosshair grabs it (MOVE/ROTATE), overriding the
        # active tool — SSMview behaviour.
        self._cross_grab = (self._drag_btn == 1
                            and self._cross_press(key, ev["x"], ev["y"]))

    def _on_move(self, key, ev):
        if self._drag_btn != 1:               # left-drag drives tool/crosshair
            return
        x, y = ev["x"], ev["y"]
        if self._cross_grab:
            self._cross_move(key, x, y)
            self._last = (x, y)
            return
        dx, dy = x - self._last[0], y - self._last[1]
        self._last = (x, y)
        shift = "Shift" in (ev.get("modifiers") or ())
        self._drag(key, dx, dy, shift, x, y)

    def _on_up(self, key, ev):
        self._drag_btn = None
        self._cross_grab = False
        self._spin_prev = None

    def _on_dblclick(self, key, ev):
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
        # Slab-MIP rendering arrives in Phase 7; for now store the value.
        self._thick[self._active_pane] = float(mm)
        self._view_initial = False

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
        for key in ("A", "B"):
            p = self.pane[key]
            if p.mesh is not None:
                p.scene.remove(p.mesh)
                p.mesh = None
            p.render()

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
            p.material.clim = (self._lvl - self._win / 2.0,
                               self._lvl + self._win / 2.0)
            if reset_cam:
                self._fit_pane(key)
            else:
                self._config_cam(key)
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

    def _toggle_centerline(self):
        self._cl_on = self._cl_btn.isChecked()
        for k in ("A", "B"):
            self._overlay[k].update()

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
        tool = _TOOL_KEYS.get(e.key())
        if tool:
            self._set_tool(tool)
        else:
            super().keyPressEvent(e)
