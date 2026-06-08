"""SSMview-style cardiac CT viewer: two linked oblique MPR panes with
sliding-slab MIP and an optional HU colour map.

Replaces the old orthogonal axial/coronal/sagittal vtkImageViewer2 layout.

Geometry
--------
One shared vtkImageData (HU). A right-handed orthonormal basis
(e0, e1, e2) in patient mm and a CrossLine ``center`` define the views:

* Pane A : plane spanned by (e0, e1), normal e2
* Pane B : plane spanned by (e0, e2), normal e1  (orthogonal, shares e0)

Each pane is  vtkImageReslice -> vtkImageMapToColors -> vtkImageActor.
``ResliceAxes`` columns are (U, V, N) with translation = ``center``, so
output world (wx, wy) maps back to volume as  M·(wx, wy, 0, 1)  — used
for double-click recenter.

Tools (toolbar button = red when active; left-hand keys R/T/W/S/G/Z/V):
ZOOM, MOVE, ROTATE (3-D about center), SPIN (in-plane), PAGING (slide
along the active normal), THICK (slab-MIP thickness), WL. Double-click
recenters the CrossLine and relinks the other pane. ColorMap toggles a
HU pseudo-colour LUT; Reset restores the default window/level.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PyQt6.QtCore import QPoint as QtPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor
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

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import (
    VTK_FLOAT, vtkLookupTable, vtkPoints, vtkUnsignedCharArray,
)
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkImageData,
    vtkPolyData,
)
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkFiltersSources import vtkLineSource
from vtkmodules.vtkImagingCore import vtkImageMapToColors, vtkImageReslice
from vtkmodules.vtkRenderingAnnotation import vtkCornerAnnotation
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkBillboardTextActor3D,
    vtkImageActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkTextActor,
)

# Rendering / interaction implementations VTK loads lazily.
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
import vtkmodules.vtkInteractionStyle  # noqa: F401
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser

from multi_dicomviewer.config import CT_WL_PRESETS
from multi_dicomviewer.core.dicom_io import LoadedSeries
from multi_dicomviewer.core.dicom_tags import overlay_lines
from multi_dicomviewer.core.measurements import Measurement
from multi_dicomviewer.ui.tag_font import (
    TAG_FONT_FILE, TAG_FONT_PT_DEFAULT, VTK_FONT_FILE, build_tag_font_control,
    wrap_lines_to_chars,
)


#: vtkCornerAnnotation sizes its font in pixels (smaller than the Qt point
#: sizes the other viewers draw) and auto-scales to the viewport, so the CT tag
#: text looked smaller. Convert pt -> px (~96 dpi) and pin min=max so CT matches
#: the other modalities' size and stays fixed regardless of window size.
_VTK_TAG_FONT_SCALE = 4 / 3

#: vtkCornerAnnotation packs lines tighter than the Qt-drawn overlays in the
#: other viewers; loosen the line spacing so the CT tag block matches their
#: roominess (the CT lines looked cramped / squeezed together).
_VTK_TAG_LINE_SPACING = 1.5525
#: vtkCornerAnnotation renders NOTHING when its minimum font size is larger than
#: what fits the corner, so keep the minimum low and only drive the MAXIMUM with
#: the slider — the text then scales down to fit instead of vanishing.
_VTK_MIN_FONT = 4


def _vtk_font_px(pt) -> int:
    return max(1, round(int(pt) * _VTK_TAG_FONT_SCALE))


def _set_vtk_tag_font(tp) -> None:
    """Point a vtkTextProperty at the shared overlay font file (Meiryo) when
    present, so CT tags use the same Japanese-capable typeface as the other
    viewers. The caller sets Arial first as the fallback — Yu Gothic's .ttc
    renders blank in VTK's FreeType, but Meiryo's loads fine."""
    if os.path.exists(TAG_FONT_FILE):
        tp.SetFontFamily(VTK_FONT_FILE)
        tp.SetFontFile(TAG_FONT_FILE)
from multi_dicomviewer.core.measure_geom import (
    angle_at as _angle_at,
    central_arc_angle as _central_arc_angle,
    convex_hull as _convex_hull,
    dist as _dist,
    ellipse_cab as _ellipse_cab_pure,
    ellipse_drag as _ellipse_drag,
    ellipse_from_major as _ellipse_from_major,
    ellipse_outline as _ellipse_outline,
    major_minor as _major_minor_pure,
    min_width as _min_width,
    point_in_poly as _point_in_poly,
    poly_area as _poly_area,
    project_to_polyline as _project_to_polyline,
    seg_dist as _seg_dist,
    smooth_closed as _smooth_closed,
    smooth_open as _smooth_open,
)
from multi_dicomviewer.ui.viewer_base import AbstractViewer

_TOOLS = ("ZOOM", "MOVE", "ROTATE", "SPIN", "PAGING", "THICK", "WL")
_TOOL_KEYS = {
    Qt.Key.Key_Z: "ZOOM", Qt.Key.Key_V: "MOVE", Qt.Key.Key_S: "SPIN",
    Qt.Key.Key_G: "PAGING", Qt.Key.Key_W: "WL", Qt.Key.Key_R: "ROTATE",
    Qt.Key.Key_T: "THICK",
}  # C = ColorMap toggle (handled separately)

#: SPIN sign. Flip to -1.0 if the rotation direction feels reversed.
_SPIN_SIGN = -1.0


# --------------------------------------------------------------------- math
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


def _tri_pd(apex, b1, b2) -> vtkPolyData:
    pd = vtkPolyData()
    pts = vtkPoints()
    polys = vtkCellArray()
    for p in (apex, b1, b2):
        pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    polys.InsertNextCell(3)
    for i in range(3):
        polys.InsertCellPoint(i)
    pd.SetPoints(pts)
    pd.SetPolys(polys)
    return pd


def _tris_pd(tris) -> vtkPolyData:
    """Polydata holding several triangles; each tri = (apex, b1, b2)."""
    pd = vtkPolyData()
    pts = vtkPoints()
    polys = vtkCellArray()
    idx = 0
    for tri in tris:
        for p in tri:
            pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        polys.InsertNextCell(3)
        for _ in range(3):
            polys.InsertCellPoint(idx)
            idx += 1
    pd.SetPoints(pts)
    pd.SetPolys(polys)
    return pd


def _dashed_pd(p0, p1, dash, gap) -> vtkPolyData:
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    pd = vtkPolyData()
    pts = vtkPoints()
    lines = vtkCellArray()
    L = float(np.linalg.norm(p1 - p0))
    if L > 1e-6 and dash > 0:
        d = (p1 - p0) / L
        s, idx = 0.0, 0
        while s < L:
            e = min(s + dash, L)
            a = p0 + d * s
            b = p0 + d * e
            pts.InsertNextPoint(*a)
            pts.InsertNextPoint(*b)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(idx)
            lines.InsertCellPoint(idx + 1)
            idx += 2
            s = e + gap
    pd.SetPoints(pts)
    pd.SetLines(lines)
    return pd


def _dashed_multi_pd(segments, z=0.66) -> vtkPolyData:
    """One polydata of faint dotted lines for several 2-D segments
    (major/minor diameter guides). Dash length scales with each
    segment so it reads as dots at any zoom."""
    pd = vtkPolyData()
    pts = vtkPoints()
    lines = vtkCellArray()
    idx = 0
    for p0, p1 in segments:
        a = np.asarray(p0, float)
        b = np.asarray(p1, float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        d = (b - a) / L
        dash = max(0.4, L / 44.0)
        s = 0.0
        while s < L:
            e = min(s + dash, L)
            qa = a + d * s
            qb = a + d * e
            pts.InsertNextPoint(float(qa[0]), float(qa[1]), z)
            pts.InsertNextPoint(float(qb[0]), float(qb[1]), z)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(idx)
            lines.InsertCellPoint(idx + 1)
            idx += 2
            s = e + dash            # gap == dash -> evenly dotted
    pd.SetPoints(pts)
    pd.SetLines(lines)
    return pd


def _polyline_pd(pts) -> vtkPolyData:
    """Open polyline through 2-D points (z=0.6, above the image)."""
    pd = vtkPolyData()
    vp = vtkPoints()
    lines = vtkCellArray()
    for q in pts:
        vp.InsertNextPoint(float(q[0]), float(q[1]), 0.6)
    if len(pts) >= 2:
        lines.InsertNextCell(len(pts))
        for k in range(len(pts)):
            lines.InsertCellPoint(k)
    pd.SetPoints(vp)
    pd.SetLines(lines)
    return pd


def _multi_pd(polylines) -> vtkPolyData:
    """One polydata holding several open polylines (z=0.6)."""
    pd = vtkPolyData()
    vp = vtkPoints()
    lines = vtkCellArray()
    base = 0
    for pl in polylines:
        if len(pl) < 2:
            for q in pl:
                vp.InsertNextPoint(float(q[0]), float(q[1]), 0.6)
            base += len(pl)
            continue
        for q in pl:
            vp.InsertNextPoint(float(q[0]), float(q[1]), 0.6)
        lines.InsertNextCell(len(pl))
        for k in range(len(pl)):
            lines.InsertCellPoint(base + k)
        base += len(pl)
    pd.SetPoints(vp)
    pd.SetLines(lines)
    return pd


def _hex_to_rgb(hexstr, default=(0x33, 0xE6, 0xFF)):
    if not hexstr:
        return default
    s = hexstr.lstrip("#")
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except Exception:
        return default


def _colored_multi_pd(polylines, colors) -> vtkPolyData:
    """Same as _multi_pd but each polyline cell carries its own RGB
    colour (uint8 triple). Caller's mapper must enable cell-scalar
    direct-colour to see the per-measure colours."""
    pd = vtkPolyData()
    vp = vtkPoints()
    lines = vtkCellArray()
    cell_rgb = vtkUnsignedCharArray()
    cell_rgb.SetNumberOfComponents(3)
    cell_rgb.SetName("Colors")
    base = 0
    for pl, rgb in zip(polylines, colors):
        n = len(pl)
        for q in pl:
            vp.InsertNextPoint(float(q[0]), float(q[1]), 0.6)
        if n >= 2:
            lines.InsertNextCell(n)
            for k in range(n):
                lines.InsertCellPoint(base + k)
            cell_rgb.InsertNextTuple3(*rgb)
        base += n
    pd.SetPoints(vp)
    pd.SetLines(lines)
    if cell_rgb.GetNumberOfTuples() > 0:
        pd.GetCellData().SetScalars(cell_rgb)
    return pd


def _colored_dashed_pd(segments, colors, z=0.66) -> vtkPolyData:
    """_dashed_multi_pd + per-segment RGB colours (one colour per input
    segment, applied to every dash cell of that segment)."""
    pd = vtkPolyData()
    pts = vtkPoints()
    lines = vtkCellArray()
    cell_rgb = vtkUnsignedCharArray()
    cell_rgb.SetNumberOfComponents(3)
    cell_rgb.SetName("Colors")
    idx = 0
    for (p0, p1), rgb in zip(segments, colors):
        a = np.asarray(p0, float)
        b = np.asarray(p1, float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        d = (b - a) / L
        dash = max(0.4, L / 44.0)
        s = 0.0
        while s < L:
            e = min(s + dash, L)
            qa = a + d * s; qb = a + d * e
            pts.InsertNextPoint(float(qa[0]), float(qa[1]), z)
            pts.InsertNextPoint(float(qb[0]), float(qb[1]), z)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(idx); lines.InsertCellPoint(idx + 1)
            cell_rgb.InsertNextTuple3(*rgb)
            idx += 2
            s = e + dash
    pd.SetPoints(pts); pd.SetLines(lines)
    if cell_rgb.GetNumberOfTuples() > 0:
        pd.GetCellData().SetScalars(cell_rgb)
    return pd


def _points_pd(pts) -> vtkPolyData:
    """Vertex-cell polydata so each clicked point shows as a dot
    (even a single start point)."""
    pd = vtkPolyData()
    vp = vtkPoints()
    verts = vtkCellArray()
    for k, q in enumerate(pts):
        vp.InsertNextPoint(float(q[0]), float(q[1]), 0.7)
        verts.InsertNextCell(1)
        verts.InsertCellPoint(k)
    pd.SetPoints(vp)
    pd.SetVerts(verts)
    return pd


def numpy_to_vtk_image(vol: np.ndarray, sx, sy, sz) -> vtkImageData:
    """vol: (z, y, x) HU float32 -> vtkImageData (x fastest, deep copy)."""
    z, y, x = vol.shape
    flat = np.ascontiguousarray(vol.ravel(order="C"), dtype=np.float32)
    arr = numpy_to_vtk(flat, deep=True, array_type=VTK_FLOAT)
    img = vtkImageData()
    img.SetDimensions(x, y, z)
    img.SetSpacing(float(sx), float(sy), float(sz))
    img.GetPointData().SetScalars(arr)
    return img


def _placeholder_image() -> vtkImageData:
    """Tiny stub kept on each reslice until a real CT loads, so empty
    start-up renders don't log 'input port 0 has 0 connections'."""
    img = vtkImageData()
    img.SetDimensions(2, 2, 2)
    img.SetSpacing(1.0, 1.0, 1.0)
    arr = numpy_to_vtk(
        np.zeros(8, dtype=np.float32), deep=True, array_type=VTK_FLOAT
    )
    img.GetPointData().SetScalars(arr)
    return img


def _gray_lut(width: float, level: float) -> vtkLookupTable:
    lut = vtkLookupTable()
    lut.SetHueRange(0.0, 0.0)
    lut.SetSaturationRange(0.0, 0.0)
    lut.SetValueRange(0.0, 1.0)
    lut.SetTableRange(level - width / 2.0, level + width / 2.0)
    lut.Build()
    return lut


#: Default HU colour bands (SSMview-style). Each band colours the HU
#: range [lo, hi]; "on" toggles it. Opacity blends band colour over the
#: windowed grayscale (0 = grayscale, 1 = full colour).
_DEFAULT_BANDS = [
    {"rgb": (1.0, 0.0, 0.0), "lo": -1000, "hi": 0,    "on": True},
    {"rgb": (1.0, 1.0, 0.0), "lo": 0,     "hi": 50,   "on": True},
    {"rgb": (0.0, 1.0, 0.0), "lo": 50,    "hi": 250,  "on": True},
    {"rgb": (0.0, 0.0, 1.0), "lo": 250,   "hi": 350,  "on": False},
    {"rgb": (1.0, 1.0, 1.0), "lo": 350,   "hi": 700,  "on": True},
    {"rgb": (1.0, 0.0, 1.0), "lo": 850,   "hi": 2000, "on": True},
]
_HU_LO, _HU_HI = -1000.0, 2000.0


def _band_lut(bands, opacity, win, lvl) -> vtkLookupTable:
    """HU -> RGB table. Inside an enabled band: band colour blended over
    the windowed grayscale by *opacity*. Outside any band: grayscale."""
    n = 512
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(n)
    lut.SetTableRange(_HU_LO, _HU_HI)
    glo = lvl - win / 2.0
    span = max(1e-6, float(win))
    op = min(1.0, max(0.0, float(opacity)))
    for i in range(n):
        hu = _HU_LO + (_HU_HI - _HU_LO) * i / (n - 1)
        g = min(1.0, max(0.0, (hu - glo) / span))
        col = None
        for b in bands:
            if b["on"] and b["lo"] <= hu <= b["hi"]:
                col = b["rgb"]
                break
        if col is None:
            r = gg = bb = g
        else:
            r = op * col[0] + (1 - op) * g
            gg = op * col[1] + (1 - op) * g
            bb = op * col[2] + (1 - op) * g
        lut.SetTableValue(i, r, gg, bb, 1.0)
    lut.Build()
    return lut


# ----------------------------------------------------------------- pane
class _PaneCanvas(QVTKRenderWindowInteractor):
    """QVTK widget whose mouse/keyboard go to the owning viewer's tools
    instead of VTK's interactor style."""

    def __init__(self, owner: "CTViewer", which: str, parent=None):
        super().__init__(parent)
        self._owner = owner
        self._which = which
        self._last = None
        self._cross = False
        self._meas_drag = False
        style = vtkInteractorStyleUser()  # neutralise default VTK style
        self.SetInteractorStyle(style)

    def mousePressEvent(self, e):
        self._owner._set_active(self._which)
        if self._owner._meas_on:
            self._cross = False
            if e.button() == Qt.MouseButton.RightButton:
                self._meas_drag = False
                self._last = None
                self._owner._measure_right(
                    self._which, e.position().x(), e.position().y()
                )
                return
            started = self._owner._measure_left(
                self._which, e.position().x(), e.position().y()
            )
            self._meas_drag = bool(started)
            self._last = e.position() if started else None
            return
        self._owner._spin_prev = None        # restart SPIN wheel angle
        # Pressing ON the crosshair (with tolerance) rotates it about the
        # centre, overriding the selected tool (SSMview behaviour).
        self._cross = self._owner._cross_press(
            self._which, e.position().x(), e.position().y()
        )
        self._last = e.position()

    def mouseMoveEvent(self, e):
        if self._owner._meas_on:
            if self._meas_drag:
                self._owner._measure_drag(
                    self._which, e.position().x(), e.position().y()
                )
            return
        if self._last is None:
            return
        p = e.position()
        if self._cross:
            self._owner._cross_move(self._which, p.x(), p.y())
            self._last = p
            return
        shift = bool(
            e.modifiers() & Qt.KeyboardModifier.ShiftModifier
        )
        self._owner._drag(
            self._which, p.x() - self._last.x(), p.y() - self._last.y(),
            shift, p.x(), p.y(),
        )
        self._last = p

    def mouseReleaseEvent(self, e):
        if self._owner._meas_on and self._meas_drag:
            self._owner._measure_release()
        self._meas_drag = False
        self._last = None
        self._cross = False
        self._owner._spin_prev = None

    def mouseDoubleClickEvent(self, e):
        if self._owner._meas_on:
            self._owner._measure_finish_draft()
            return
        self._owner._recenter(self._which, e.position().x(), e.position().y())

    def wheelEvent(self, e):
        self._owner._wheel(self._which, e.angleDelta().y())

    def keyPressEvent(self, e):
        self._owner.keyPressEvent(e)  # tool/colormap shortcuts

    def resizeEvent(self, e):
        # Override the upstream QVTKRenderWindowInteractor.resizeEvent so the
        # render window is sized using THIS widget's own DPR, not the DPR of
        # whichever monitor the cursor happens to be on. Keeps every DPR-
        # dependent quantity (render window size, our mouse <-> world maths)
        # in lockstep on mixed-DPR multi-monitor setups.
        dpr = self.devicePixelRatioF()
        w = int(round(dpr * self.width()))
        h = int(round(dpr * self.height()))
        self._RenderWindow.SetDPI(int(round(72 * dpr)))
        vtkRenderWindow.SetSize(self._RenderWindow, w, h)
        self._Iren.SetSize(w, h)
        self._Iren.ConfigureEvent()
        self.update()

        # Refit while the view is still at its initial state (first
        # layout, Info hidden, window resize) so the right edge stays
        # visible; once the user has zoomed/panned, leave it alone.
        o = self._owner
        if (
            getattr(o, "_image", None) is not None
            and getattr(o, "_view_initial", False)
        ):
            o._fit_pane(self._which)


class _Pane:
    """One reslice view: pipeline + renderer + CrossLine actors."""

    def __init__(self, canvas: _PaneCanvas):
        self.canvas = canvas
        self.reslice = vtkImageReslice()
        self.reslice.SetInputData(_placeholder_image())
        self.reslice.SetOutputDimensionality(2)
        self.reslice.SetInterpolationModeToLinear()
        self.reslice.SetBackgroundLevel(-1000.0)
        self.colors = vtkImageMapToColors()
        self.colors.SetOutputFormatToRGB()
        self.colors.SetLookupTable(_gray_lut(400.0, 40.0))  # default
        self.colors.SetInputConnection(self.reslice.GetOutputPort())
        self.actor = vtkImageActor()
        self.actor.GetMapper().SetInputConnection(self.colors.GetOutputPort())
        self.ren = vtkRenderer()
        self.ren.SetBackground(0.0, 0.0, 0.0)
        self.ren.GetActiveCamera().ParallelProjectionOn()
        self.ren.AddActor(self.actor)

        self.cross = []
        self._overlay_actors = []
        for _ in range(2):
            src = vtkLineSource()
            m = vtkPolyDataMapper()
            m.SetInputConnection(src.GetOutputPort())
            a = vtkActor()
            a.SetMapper(m)
            a.GetProperty().SetColor(1.0, 0.85, 0.0)
            a.GetProperty().SetLineWidth(1.0)
            a.GetProperty().SetOpacity(0.5)        # 50% transparent
            self.ren.AddActor(a)
            self.cross.append((src, a))
            self._overlay_actors.append(a)

        # ▲ = the other pane's projection direction.
        self.tri_mapper = vtkPolyDataMapper()
        self.tri_mapper.SetInputData(vtkPolyData())
        tri = vtkActor()
        tri.SetMapper(self.tri_mapper)
        tri.GetProperty().SetColor(0.0, 0.95, 0.25)   # green: stands out
        self.ren.AddActor(tri)
        self._overlay_actors.append(tri)
        # Two dashed lines = the other pane's slab-MIP width.
        self.slab_mappers = []
        for _ in range(2):
            mp = vtkPolyDataMapper()
            mp.SetInputData(vtkPolyData())
            a = vtkActor()
            a.SetMapper(mp)
            a.GetProperty().SetColor(1.0, 0.85, 0.0)
            a.GetProperty().SetLineWidth(1.0)
            a.GetProperty().SetOpacity(0.5)        # 50% transparent
            self.ren.AddActor(a)
            self.slab_mappers.append(mp)
            self._overlay_actors.append(a)

        # Measurement overlay (filled/edges + vertex markers).
        # Per-measure colour: polylines/segments carry per-cell RGB
        # scalars via _colored_multi_pd, and the mapper renders them
        # directly. Fallback (no scalars) is cyan via SetColor.
        self.meas_mapper = vtkPolyDataMapper()
        self.meas_mapper.SetInputData(vtkPolyData())
        self.meas_mapper.ScalarVisibilityOn()
        self.meas_mapper.SetScalarModeToUseCellData()
        self.meas_mapper.SetColorModeToDirectScalars()
        ma = vtkActor()
        ma.SetMapper(self.meas_mapper)
        ma.GetProperty().SetColor(0.2, 0.9, 1.0)   # cyan fallback
        ma.GetProperty().SetLineWidth(1.5)
        self.ren.AddActor(ma)
        # In-progress draft outline — drawn DASHED (geometric dashes, the
        # same technique as the axis / center-angle guides) so a shape
        # being placed reads as "not yet committed". On commit it moves to
        # meas_mapper above and re-renders solid.
        self.meas_draft_mapper = vtkPolyDataMapper()
        self.meas_draft_mapper.SetInputData(vtkPolyData())
        self.meas_draft_mapper.ScalarVisibilityOn()
        self.meas_draft_mapper.SetScalarModeToUseCellData()
        self.meas_draft_mapper.SetColorModeToDirectScalars()
        mda = vtkActor()
        mda.SetMapper(self.meas_draft_mapper)
        mda.GetProperty().SetColor(0.2, 0.9, 1.0)  # cyan fallback
        mda.GetProperty().SetLineWidth(1.5)
        self.ren.AddActor(mda)
        # 長径/短径 (major/minor diameter) guides for ellipse & polygon:
        # thin + faint + dotted, drawn under the outline.
        self.meas_axis_mapper = vtkPolyDataMapper()
        self.meas_axis_mapper.SetInputData(vtkPolyData())
        self.meas_axis_mapper.ScalarVisibilityOn()
        self.meas_axis_mapper.SetScalarModeToUseCellData()
        self.meas_axis_mapper.SetColorModeToDirectScalars()
        mxa = vtkActor()
        mxa.SetMapper(self.meas_axis_mapper)
        mxa.GetProperty().SetColor(0.2, 0.9, 1.0)  # cyan fallback
        mxa.GetProperty().SetLineWidth(2.2)
        mxa.GetProperty().SetOpacity(0.65)
        self.ren.AddActor(mxa)
        # Center-Angle spokes (dashed lines from shape centre to each
        # picked perimeter point) + the picked dots themselves.
        self.meas_ca_mapper = vtkPolyDataMapper()
        self.meas_ca_mapper.SetInputData(vtkPolyData())
        self.meas_ca_mapper.ScalarVisibilityOn()
        self.meas_ca_mapper.SetScalarModeToUseCellData()
        self.meas_ca_mapper.SetColorModeToDirectScalars()
        mca = vtkActor()
        mca.SetMapper(self.meas_ca_mapper)
        mca.GetProperty().SetLineWidth(1.2)
        mca.GetProperty().SetOpacity(0.8)
        self.ren.AddActor(mca)
        self.meas_ca_pts_mapper = vtkPolyDataMapper()
        self.meas_ca_pts_mapper.SetInputData(vtkPolyData())
        mcap = vtkActor()
        mcap.SetMapper(self.meas_ca_pts_mapper)
        mcap.GetProperty().SetColor(1.0, 0.55, 0.0)   # orange dots
        mcap.GetProperty().SetPointSize(11.0)
        if hasattr(mcap.GetProperty(), "SetRenderPointsAsSpheres"):
            mcap.GetProperty().SetRenderPointsAsSpheres(True)
        self.ren.AddActor(mcap)
        # Clicked-point markers (so the start point is always visible).
        self.meas_pts_mapper = vtkPolyDataMapper()
        self.meas_pts_mapper.SetInputData(vtkPolyData())
        mp = vtkActor()
        mp.SetMapper(self.meas_pts_mapper)
        mp.GetProperty().SetColor(1.0, 0.85, 0.0)  # yellow idle dots
        mp.GetProperty().SetPointSize(10.0)
        if hasattr(mp.GetProperty(), "SetRenderPointsAsSpheres"):
            mp.GetProperty().SetRenderPointsAsSpheres(True)
        self.ren.AddActor(mp)
        # Same dots but the ONE point under an active drag is green,
        # slightly larger, so the user sees which handle they're editing.
        self.meas_pts_edit_mapper = vtkPolyDataMapper()
        self.meas_pts_edit_mapper.SetInputData(vtkPolyData())
        mpe = vtkActor()
        mpe.SetMapper(self.meas_pts_edit_mapper)
        mpe.GetProperty().SetColor(0.23, 0.86, 0.35)  # green
        mpe.GetProperty().SetPointSize(18.0)          # 1.5× idle (10)
        if hasattr(mpe.GetProperty(), "SetRenderPointsAsSpheres"):
            mpe.GetProperty().SetRenderPointsAsSpheres(True)
        self.ren.AddActor(mpe)
        self.meas_labels = []                 # number billboards

        self.info = vtkCornerAnnotation()
        self.info.SetMaximumFontSize(_vtk_font_px(TAG_FONT_PT_DEFAULT))
        self.info.SetMinimumFontSize(_VTK_MIN_FONT)
        _tp = self.info.GetTextProperty()
        _tp.SetColor(1.0, 1.0, 1.0)                  # white, like the others
        _tp.SetFontFamilyToArial()                   # restored (see _vtk_tag_font)
        _set_vtk_tag_font(_tp)                        # Yu Gothic via file if VTK can
        _tp.SetLineSpacing(_VTK_TAG_LINE_SPACING)    # loosen cramped CT lines
        self.ren.AddViewProp(self.info)

        # SSMview-style angio-angle readout: yellow, image bottom-centre.
        # vtkCornerAnnotation only has the 4 corners, so this is its own
        # text actor pinned to the bottom-centre in normalised viewport
        # coords (stays centred & same size at any pane/zoom).
        self.angle = vtkTextActor()
        self.angle.SetTextScaleModeToNone()
        self.angle.GetTextProperty().SetColor(1.0, 0.9, 0.0)
        self.angle.GetTextProperty().SetFontSize(15)
        self.angle.GetTextProperty().SetBold(True)
        self.angle.GetTextProperty().SetJustificationToCentered()
        self.angle.GetTextProperty().SetVerticalJustificationToBottom()
        self.angle.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.angle.GetPositionCoordinate().SetValue(0.5, 0.012)
        self.angle.SetInput("")
        # Always visible (like the WW/WL corner), independent of the
        # CenterLine overlay toggle — it's clinical info, not clutter.
        self.ren.AddViewProp(self.angle)

        # DICOM tags (top-left) and measure results (top-right) render as their
        # OWN fixed-size text actors, NOT vtkCornerAnnotation corners. The
        # annotation auto-fits its font to the viewport, which made the tag-size
        # slider ineffective and let long lines sprawl across the image. These
        # honour the slider's exact pixel size; the viewer word-wraps their text
        # to ~40% of the pane width so left tags and right results never collide.
        def _mk_overlay_text(justify_right):
            a = vtkTextActor()
            a.SetTextScaleModeToNone()              # fixed font size, no auto-fit
            tp = a.GetTextProperty()
            tp.SetColor(1.0, 1.0, 1.0)
            tp.SetFontFamilyToArial()
            _set_vtk_tag_font(tp)                   # Meiryo via file when present
            tp.SetLineSpacing(_VTK_TAG_LINE_SPACING)
            tp.SetFontSize(_vtk_font_px(TAG_FONT_PT_DEFAULT))
            tp.SetVerticalJustificationToTop()
            tp.SetJustificationToRight() if justify_right \
                else tp.SetJustificationToLeft()
            a.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            a.GetPositionCoordinate().SetValue(0.988 if justify_right else 0.012,
                                               0.985)
            a.SetInput("")
            return a

        self.tagact = _mk_overlay_text(False)       # top-left
        self.resultact = _mk_overlay_text(True)     # top-right
        self.ren.AddViewProp(self.tagact)
        self.ren.AddViewProp(self.resultact)

        self.canvas.GetRenderWindow().AddRenderer(self.ren)

    def set_overlay_visible(self, on: bool) -> None:
        for a in self._overlay_actors:
            a.SetVisibility(bool(on))

    def render(self):
        self.canvas.GetRenderWindow().Render()


class _ColorMapDialog(QDialog):
    """SSMview-style HU colour-map editor: a list of colour bands (colour
    + HU Min/Max + enable/remove), an Opacity slider, Add and Reset.
    Changes apply live via on_change(bands, opacity)."""

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

    # -------------------------------------------------------- helpers
    def _emit(self):
        self._on_change(self._bands, self._opacity)

    def _op_changed(self, v):
        self._opacity = v / 100.0
        self._op_lbl.setText(f"{self._opacity:.2f}")
        self._emit()

    def _add_band(self):
        self._bands.append(
            {"rgb": (1.0, 1.0, 1.0), "lo": 0, "hi": 100, "on": True}
        )
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
        """Force-sync the dialog's display to *bands* / *opacity*. Used
        when the dialog is reopened so its UI always reflects the
        viewer's current band state — the user's disabled/colour edits
        are then visibly preserved across close-and-reopen of the
        Setting dialog, not just remembered internally."""
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
            self,
            "Band colour",
        )
        if col.isValid():
            self._bands[idx]["rgb"] = (
                col.redF(), col.greenF(), col.blueF()
            )
            self._rebuild()
            self._emit()


# --------------------------------------------------------------- viewer
class CTViewer(AbstractViewer):
    handles_modality = "CT"
    tags_requested = pyqtSignal()
    #: emitted when the tag-text-size slider moves (shell broadcasts the pt to
    #: every viewer so the overlay size matches across modalities)
    overlay_font_changed = pyqtSignal(int)
    #: emitted when a measurement is committed — the shell files it under
    #: the current study so it shows in the shared Measure History.
    measurement_added = pyqtSignal(object)
    #: emitted when the user clicks "Measure History"
    history_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = None
        self._header = None
        self._overlay_font_pt = TAG_FONT_PT_DEFAULT
        #: last-computed (unwrapped) result strings per pane, so a font-size
        #: change can re-wrap them without recomputing the costly HU stats.
        self._metric_lines = {"A": [], "B": []}
        self._pbasis = np.eye(3)             # voxel->patient LPS (set on load)
        self._tag_keywords: list[str] = []
        self._anon = False
        self._tool = "PAGING"
        self._dims = (1.0, 1.0, 1.0)        # sx, sy, sz mm
        self._bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        self._center = np.zeros(3)           # C: CrossLine 3-D point
        self._center0 = np.zeros(3)          # initial CrossLine center
        # Per-pane reslice centre (output (0,0) maps here). Normally == C;
        # PAGING moves only the paged pane's, so the OTHER pane's image
        # stays put while its crosshair (drawn at C's projection) slides.
        self._pc = {"A": np.zeros(3), "B": np.zeros(3)}
        # Independent (U, V, N) frame per pane. Linkage is re-established
        # only when the user drags a crosshair (the OTHER pane is then
        # derived from the dragged crossline).
        self._frame = {
            "A": (np.array([1.0, 0.0, 0.0]),
                  np.array([0.0, 1.0, 0.0]),
                  np.array([0.0, 0.0, 1.0])),
            "B": (np.array([1.0, 0.0, 0.0]),
                  np.array([0.0, 0.0, 1.0]),
                  np.array([0.0, 1.0, 0.0])),
        }
        self._half = 1.0                     # symmetric FOV half-size mm
        self._npx = 64                       # output pixels / side
        self._win, self._lvl = 800.0, 200.0
        self._win0, self._lvl0 = 800.0, 200.0
        self._thick = {"A": 0.0, "B": 5.0}   # slab mm per pane
        self._color = False
        self._bands = [dict(b) for b in _DEFAULT_BANDS]
        self._opacity = 0.25
        self._cmap_dlg = None
        self._meas_on = False
        self._meas_type = None          # line|polyline|ellipse|polygon
        self._measures = {"A": [], "B": []}   # finalized {id,type,pts}
        self._meas_seq = 0              # type-independent running number
        self._draft = None              # {type, pane, pts} in progress
        self._edit = None               # {key, mi, vi} handle drag
        self._center_angle_target = None  # {key, mi} during 3-pt pick
        self._active_pane = "A"
        self._view_initial = True            # for the 2-stage Reset
        self._cross_prev = 0.0               # CrossLine-rotate prev angle
        self._spin_prev = None               # SPIN wheel previous angle
        self._cross_mode = "rotate"          # "rotate" | "move"
        self._cross_axis = None              # locked move axis (2-D unit)
        self._cross_ppt = (0.0, 0.0)         # prev world point (move mode)
        # Drawn crosshair rotation per pane (deg); follows the cursor
        # while CrossLine-dragging so the crosshair tracks the drag.
        self._cross_ang = {"A": 0.0, "B": 0.0}

        self.canvas_a = _PaneCanvas(self, "A")
        self.canvas_b = _PaneCanvas(self, "B")
        self.pane = {"A": _Pane(self.canvas_a), "B": _Pane(self.canvas_b)}

        # Wrap each canvas so the active pane can show a yellow border
        # (the GL surface can't take a stylesheet itself).
        self._frames = {}
        for key, canvas in (("A", self.canvas_a), ("B", self.canvas_b)):
            f = QFrame()
            f.setObjectName("ctpane")
            fl = QVBoxLayout(f)
            fl.setContentsMargins(3, 3, 3, 3)
            fl.setSpacing(0)
            fl.addWidget(canvas)
            self._frames[key] = f

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

        for c in (self.canvas_a, self.canvas_b):
            c.Initialize()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._update_active_frames()

    # -- Bi / Lt / Rt --------------------------------------------------
    @property
    def supports_side(self) -> bool:
        """CT always has two panes, so Bi/Lt/Rt always applies."""
        return True

    def set_side(self, side: str, allow_dual: bool = True) -> None:
        """Show only pane A (Lt), only pane B (Rt), or both (Bi).

        ``allow_dual`` is accepted for API parity with the XA viewer
        but ignored here — CT has its own A/B pane pair independent of
        the shell's grid layout."""
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
        self._style_cl()

        row.addWidget(QLabel("Slab(mm):"))
        self._slab_spin = QDoubleSpinBox()
        self._slab_spin.setRange(0.0, 50.0)
        self._slab_spin.setSingleStep(0.5)
        self._slab_spin.setDecimals(1)
        self._slab_spin.setToolTip(
            "Slab-MIP thickness of the active pane (0 = thin MPR)"
        )
        self._slab_spin.valueChanged.connect(self._set_slab)
        row.addWidget(self._slab_spin)

        # Order: Measure → ColorMap → Setting (Setting is the ColorMap's
        # config dialog, so it sits next to ColorMap on the right).
        self._meas_btn = QPushButton("📏 Measure")
        self._meas_btn.setCheckable(True)
        self._meas_btn.setToolTip(
            "Measure on the image (Line / Polyline / Ellipse / Polygon)"
        )
        self._meas_btn.clicked.connect(self._toggle_measure)
        row.addWidget(self._meas_btn)

        self._cmap_btn = QPushButton("ColorMap")
        self._cmap_btn.setCheckable(True)
        self._cmap_btn.clicked.connect(self._toggle_color)
        row.addWidget(self._cmap_btn)

        # Setting / Reset are utility actions (not tools): tint them a light
        # grey so they read as distinct from the tool / preset buttons.
        _util_btn_css = "background:#e0e0e0;color:black;"
        setting = QPushButton("Setting")
        setting.setToolTip(
            "HU colour-map settings (band colour, HU range, opacity)"
        )
        setting.setStyleSheet(_util_btn_css)
        setting.clicked.connect(self._open_setting)
        row.addWidget(setting)

        reset = QPushButton("Reset")
        reset.setToolTip(
            "1st click: keep W/L, reset the view position / "
            "click again at the initial position: also reset W/L"
        )
        reset.setStyleSheet(_util_btn_css)
        reset.clicked.connect(self._reset)
        row.addWidget(reset)

        row.addWidget(QLabel("W/L preset:"))
        self._preset = QComboBox()
        self._preset.addItems(list(CT_WL_PRESETS.keys()))
        self._preset.currentTextChanged.connect(self._apply_preset)
        row.addWidget(self._preset)

        # DICOM Tags on the LEFT of the pair (always visible); Measure History
        # — less critical — to its right. Tag-text-size slider stacked above.
        tags_box, self._tag_font_slider, tags = build_tag_font_control(
            TAG_FONT_PT_DEFAULT
        )
        tags.clicked.connect(self.tags_requested.emit)
        self._tag_font_slider.valueChanged.connect(self.overlay_font_changed.emit)
        row.addWidget(tags_box)

        hist = QPushButton("Measure History")
        hist.setToolTip("Show this study's measurement history")
        hist.clicked.connect(self.history_requested.emit)
        row.addWidget(hist)
        row.addStretch(1)
        self._set_tool("PAGING")
        return row

    def set_overlay_font_pt(self, pt: int) -> None:
        """Apply the shared DICOM-tag text size (pt) to every pane's corner
        annotation and sync the slider. Called by the shell."""
        pt = int(pt)
        self._overlay_font_pt = pt
        sl = getattr(self, "_tag_font_slider", None)
        if sl is not None and sl.value() != pt:
            sl.blockSignals(True)
            sl.setValue(pt)
            sl.blockSignals(False)
        vf = _vtk_font_px(pt)
        for key, p in self.pane.items():
            p.info.SetMaximumFontSize(vf)   # bottom WW/WL & key|kind readouts
            # Tags + results are fixed-size actors: set their EXACT pixel size
            # (no auto-fit), then re-wrap to the new 40% budget. Tags are cheap
            # to rebuild; results re-wrap the stored (unwrapped) lines so we
            # don't recompute HU stats on every slider tick.
            p.tagact.GetTextProperty().SetFontSize(vf)
            p.resultact.GetTextProperty().SetFontSize(vf)
            if self._header is not None:
                self._update_info(key, False)
                p.resultact.SetInput("\n".join(wrap_lines_to_chars(
                    self._metric_lines.get(key, []), self._wrap_budget(key))))
            p.render()

    def _set_tool(self, name):
        self._tool = name
        for n, b in self._tool_btns.items():
            b.setChecked(n == name)
            b.setStyleSheet(
                "background:#c0392b;color:white;" if n == name else ""
            )

    # --------------------------------------------------- CenterLine
    def _style_cl(self):
        on = self._cl_btn.isChecked()
        self._cl_btn.setStyleSheet(
            "background:#b8860b;color:black;" if on
            else "background:white;color:black;"
        )

    def _toggle_centerline(self):
        on = self._cl_btn.isChecked()
        for k in ("A", "B"):
            self.pane[k].set_overlay_visible(on)
            self.pane[k].render()
        self._style_cl()

    # ------------------------------------------------------- Measure
    def _build_measure_bar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 6, 2)
        row.addWidget(QLabel("Measure:"))
        self._meas_btns = {}
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
        clr.clicked.connect(self._measure_clear)
        row.addWidget(clr)
        row.addWidget(QLabel(
            "  Left-click = add point /"
            " right-click finishes Polyline / Polygon"
        ))
        row.addStretch(1)
        return bar

    _JP = {"line": "Line", "polyline": "Polyline",
           "ellipse": "Ellipse", "polygon": "Polygon", "angle": "Angle"}

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

    def _set_measure_type(self, key):
        self._meas_type = key
        self._draft = None
        for k, b in self._meas_btns.items():
            b.setChecked(k == key)
            b.setStyleSheet(
                "background:#1f77b4;color:white;" if k == key else ""
            )

    def _measure_clear(self):
        self._measures = {"A": [], "B": []}
        self._draft = None
        self._edit = None
        for k in ("A", "B"):
            self._redraw_meas(k)

    # ---- world<->screen ----
    def _world_to_qt(self, key, wx, wy):
        canvas = self.pane[key].canvas
        ren = self.pane[key].ren
        ren.SetWorldPoint(float(wx), float(wy), 0.0, 1.0)
        ren.WorldToDisplay()
        dx, dy, _dz = ren.GetDisplayPoint()
        # VTK display coords are in render-window physical pixels; this
        # canvas always sizes the render window using its own DPR (see
        # _PaneCanvas.resizeEvent), so divide by the SAME DPR to recover
        # Qt-logical coords. Robust across mixed-DPR multi-monitor.
        dpr = canvas.devicePixelRatioF()
        return dx / dpr, canvas.height() - dy / dpr

    # ---- per-measure geometry ----
    def _ellipse_cab(self, m):
        return _ellipse_cab_pure(m["pts"])

    def _outline(self, m):
        t = m["type"]
        if t == "line":
            return list(m["pts"][:2])
        if t == "polyline":
            pts = list(m["pts"])
            if m.get("smooth"):                       # Spline toggle on
                return _smooth_open(pts)
            return pts
        if t == "polygon":
            # Smooth closed curve from the vertices (drawn curve AND the
            # ROI used for area / HU / diameters); no corner overshoot.
            return _smooth_closed(m["pts"])
        if t == "angle":
            # Clicked end1 → vertex → end2; draw straight through so the
            # vertex sits between its two rays.
            return list(m["pts"][:3])
        return _ellipse_outline(m["pts"])          # ellipse (rotated)

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
            return m["pts"][1]                       # label at the vertex
        return m["pts"][0]

    def _shape_center(self, m):
        return self._anchor(m)

    def _major_minor(self, m):
        return _major_minor_pure(m)

    def _metrics_text(self, key, m):
        t = m["type"]
        pts = m["pts"]
        ca = m.get("center_angle")
        ca_str = (f"  CenterAngle:{ca['angle']:.1f}°"
                  if ca and "angle" in ca else "")
        if t == "line":
            return f"#{m['id']} Line: {_dist(pts[0], pts[1]):.1f} mm"
        if t == "polyline":
            L = sum(_dist(pts[i], pts[i + 1])
                    for i in range(len(pts) - 1))
            tag = "Polyline (Spline)" if m.get("smooth") else "Polyline"
            return f"#{m['id']} {tag}: {L:.1f} mm"
        if t == "angle":
            # Vertex is the middle (2nd) click; rays go to the 1st & 3rd.
            ang = _angle_at(pts[1], pts[0], pts[2])
            return f"#{m['id']} Angle: {ang:.1f}°"
        if t == "ellipse":
            cx, cy, a, b = self._ellipse_cab(m)
            poly = self._outline(m)
            lo, hi = self._hu_stats(key, poly)
            return (f"#{m['id']} Ellipse  Area:{math.pi*a*b:.1f}mm²  "
                    f"CTmax:{hi:.0f}  CTmin:{lo:.0f}  "
                    f"Dmax:{2*max(a,b):.1f}  Dmin:{2*min(a,b):.1f}mm{ca_str}")
        poly = pts + [pts[0]]                       # polygon (vertices)
        lo, hi = self._hu_stats(key, poly)
        _, _, dmax, dmin = self._major_minor(m)     # vertex-based, drawn
        return (f"#{m['id']} Polygon  Area:{_poly_area(pts):.1f}mm²  "
                f"CTmax:{hi:.0f}  CTmin:{lo:.0f}  "
                f"Dmax:{dmax:.1f}  Dmin:{dmin:.1f}mm{ca_str}")

    # ---- drawing ----
    def _rebuild_labels(self, p, labels):
        for a in getattr(p, "meas_labels", []):
            p.ren.RemoveActor(a)
        p.meas_labels = []
        for text, (x, y) in labels:
            ta = vtkBillboardTextActor3D()
            ta.SetInput(text)
            ta.SetPosition(float(x), float(y), 0.8)
            ta.GetTextProperty().SetColor(1.0, 0.85, 0.0)
            ta.GetTextProperty().SetFontSize(16)
            ta.GetTextProperty().SetBold(True)
            p.ren.AddActor(ta)
            p.meas_labels.append(ta)

    def _redraw_geom(self, key):
        p = self.pane[key]
        polylines, handles, labels = [], [], []
        outline_colors: list[tuple[int, int, int]] = []
        axis_segs: list = []
        axis_colors: list[tuple[int, int, int]] = []
        ca_segs: list = []           # center-angle spokes
        ca_colors: list[tuple[int, int, int]] = []
        ca_pts: list = []            # picked perimeter dots
        # The one vertex currently being dragged lives in *edit_pts*
        # (drawn green by a second points mapper); the rest stay yellow.
        edit_pts = []
        e = self._edit
        edit_mi = e["mi"] if (e and e["key"] == key) else -1
        edit_vi = e["vi"] if (e and e["key"] == key) else -1
        edit_ca = bool(e.get("ca")) if (e and e["key"] == key) else False
        for mi, m in enumerate(self._measures[key]):
            rgb = _hex_to_rgb(m.get("color"))
            polylines.append(self._outline(m))
            outline_colors.append(rgb)
            for vi, q in enumerate(self._handles(m)):
                if mi == edit_mi and not edit_ca and vi == edit_vi:
                    edit_pts.append(q)
                else:
                    handles.append(q)
            labels.append((str(m["id"]), self._anchor(m)))
            if m["type"] in ("ellipse", "polygon"):
                maj, mnr, _, _ = self._major_minor(m)
                # Long/short-diameter lines wear the polygon-vertex colour
                # (yellow) so they read as part of the shape.
                if maj is not None:
                    axis_segs.append(maj); axis_colors.append((255, 217, 0))
                if mnr is not None:
                    axis_segs.append(mnr); axis_colors.append((255, 217, 0))
            ca = m.get("center_angle")
            if ca and ca.get("pts"):
                centre = self._shape_center(m)
                for ci, q in enumerate(ca["pts"]):
                    # The 2nd point only picks which way the angle is measured,
                    # so it gets no spoke; spokes (orange = marker colour) go to
                    # the 1st and 3rd points (the angle's two arms).
                    if ci != 1:
                        ca_segs.append((centre, q))
                        ca_colors.append((255, 140, 0))
                    # The marker being dragged turns green (like a vertex);
                    # the rest stay orange.
                    if mi == edit_mi and edit_ca and ci == edit_vi:
                        edit_pts.append(q)
                    else:
                        ca_pts.append(q)
        # The in-progress draft is drawn DASHED via its own mapper (below)
        # so it reads as not-yet-committed; on commit it re-renders solid
        # through the outline path above.
        draft_segs: list = []
        d = self._draft
        if d and d["pane"] == key and d["pts"]:
            if d["type"] == "ellipse" and len(d["pts"]) >= 2:
                # Preview the oblique ellipse whose major axis is the drag.
                dpts = _ellipse_outline(
                    _ellipse_from_major(d["pts"][0], d["pts"][1]))
            else:
                dpts = list(d["pts"])
            draft_segs = list(zip(dpts, dpts[1:]))
            handles += list(d["pts"])
        p.meas_mapper.SetInputData(
            _colored_multi_pd(polylines, outline_colors)
        )
        p.meas_draft_mapper.SetInputData(
            _colored_dashed_pd(draft_segs, [_hex_to_rgb(None)] * len(draft_segs))
        )
        p.meas_axis_mapper.SetInputData(
            _colored_dashed_pd(axis_segs, axis_colors)
        )
        p.meas_ca_mapper.SetInputData(
            _colored_dashed_pd(ca_segs, ca_colors)
        )
        p.meas_ca_pts_mapper.SetInputData(_points_pd(ca_pts))
        p.meas_pts_mapper.SetInputData(_points_pd(handles))
        p.meas_pts_edit_mapper.SetInputData(_points_pd(edit_pts))
        self._rebuild_labels(p, labels)
        p.render()

    def _redraw_meas(self, key):
        self._redraw_geom(key)
        p = self.pane[key]
        lines = [self._metrics_text(key, m) for m in self._measures[key]]
        self._metric_lines[key] = lines        # keep unwrapped for re-wrapping
        # Confine the result block to ~40% width (right) by word-wrapping it to
        # the fixed-size result actor (which honours the exact slider font).
        p.resultact.SetInput("\n".join(
            wrap_lines_to_chars(lines, self._wrap_budget(key))))
        p.render()

    def _wrap_budget(self, key) -> int:
        """Characters that fit within ~40% of the pane width at the current
        (fixed) overlay font — the wrap width for both the tag block (left) and
        result block (right). 0.6 char-width keeps the block strictly ≤40%."""
        px = max(1, self.pane[key].canvas.width())
        fpx = max(1, _vtk_font_px(
            getattr(self, "_overlay_font_pt", TAG_FONT_PT_DEFAULT)))
        return max(8, int(0.40 * px / (fpx * 0.6)))

    # ---- picking ----
    def _pick_handle(self, which, sx, sy):
        for mi in range(len(self._measures[which]) - 1, -1, -1):
            m = self._measures[which][mi]
            for vi, q in enumerate(m["pts"]):
                qx, qy = self._world_to_qt(which, q[0], q[1])
                if math.hypot(qx - sx, qy - sy) < 12.0:
                    return mi, vi
        return None

    def _pick_center_angle(self, which, sx, sy):
        """Pick a finalized Center-Angle marker point (the orange perimeter
        dots) so it can be dragged like a polygon vertex. Returns (mi, ci)."""
        for mi in range(len(self._measures[which]) - 1, -1, -1):
            ca = self._measures[which][mi].get("center_angle")
            if not ca or not ca.get("pts"):
                continue
            for ci, q in enumerate(ca["pts"]):
                qx, qy = self._world_to_qt(which, q[0], q[1])
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
        """Return True if a handle-drag started (canvas keeps the drag)."""
        if not self._meas_on:
            return False
        # Center-Angle pick mode consumes left-clicks until 3 perimeter
        # points have been added.
        cat = getattr(self, "_center_angle_target", None)
        if cat and cat.get("key") == which:
            w = self._disp_to_world(which, sx, sy)
            self._center_angle_add(w)
            return False
        hit = self._pick_handle(which, sx, sy)
        if hit is not None:
            self._edit = {"key": which, "mi": hit[0], "vi": hit[1]}
            self._redraw_geom(which)            # show the green dot now
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
        # Oblique-ellipse handle drag via the shared pure helper (same logic
        # used by the XA/IVUS canvas and the pygfx CT viewer).
        m["pts"] = _ellipse_drag(m["pts"], vi, w)

    def _commit_draft(self):
        d = self._draft
        self._draft = None
        if d is None or len(d["pts"]) < 2:
            return
        if d["type"] == "ellipse":
            # First two clicks are the MAJOR-axis endpoints (an oblique drag
            # makes an oblique ellipse); the minor radius starts at half the
            # major and is tuned afterwards via the minor handles.
            pts = _ellipse_from_major(d["pts"][0], d["pts"][1])
        elif d["type"] == "line":
            pts = d["pts"][:2]
        else:
            pts = list(d["pts"])
        self._meas_seq += 1
        self._measures[d["pane"]].append(
            {"id": self._meas_seq, "type": d["type"], "pts": pts}
        )
        self._redraw_meas(d["pane"])
        # File it under the current study's shared Measure History. The
        # pre-formatted metrics string is the label; points/kind travel
        # along so the entry is self-describing (mm units already baked in).
        m_dict = self._measures[d["pane"]][-1]
        self.measurement_added.emit(Measurement(
            kind=self._JP.get(d["type"], d["type"]),
            points=[tuple(q) for q in pts],
            text=self._metrics_text(d["pane"], m_dict),
        ))

    def _measure_finish_draft(self):
        d = self._draft
        if d and d["type"] in ("polyline", "polygon") \
                and len(d["pts"]) >= 2:
            self._commit_draft()

    def _measure_right(self, which, sx, sy):
        # Cancel an in-progress Center-Angle pick on right-click.
        cat = getattr(self, "_center_angle_target", None)
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
            chosen = menu.exec(
                self.pane[which].canvas.mapToGlobal(QtPoint(int(sx), int(sy))))
            if chosen is del_ca:
                self._measures[which][ca_mi].pop("center_angle", None)
                self._redraw_meas(which)
            return
        # Handle is more specific than outline — try it first.
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
        """Right-click on a measure handle: 'Delete point' + 'Delete
        result' for Polyline/Polygon, just 'Delete' for Line/Ellipse."""
        mi, vi = hit
        m = self._measures[which][mi]
        menu = QMenu(self)
        del_pt = del_res = None
        if m["type"] in ("polyline", "polygon"):
            del_pt = menu.addAction("Delete point")
            if len(m["pts"]) <= 2:                # never shrink below Line
                del_pt.setEnabled(False)
            del_res = menu.addAction("Delete result")
        else:
            del_res = menu.addAction("Delete")
        chosen = menu.exec(
            self.pane[which].canvas.mapToGlobal(
                QtPoint(int(sx), int(sy))
            )
        )
        if del_pt is not None and chosen is del_pt:
            self._delete_point(which, mi, vi)
        elif chosen is del_res:
            del self._measures[which][mi]
        self._redraw_meas(which)

    def _outline_right(self, which, mi, sx, sy):
        """Right-click on a measure outline: Add point / Spline (Polyline)
        / Center Angle (Ellipse/Polygon) / Change Color / Delete."""
        from multi_dicomviewer.viewers.image_canvas import COLOR_CHOICES
        from PyQt6.QtGui import QIcon, QPixmap, QColor as _QColor
        m = self._measures[which][mi]
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
        color_menu = menu.addMenu("Change Color")
        color_actions: list[tuple] = []
        for name, hexcol in COLOR_CHOICES:
            a = color_menu.addAction(name)
            pix = QPixmap(16, 16); pix.fill(_QColor(hexcol))
            a.setIcon(QIcon(pix))
            color_actions.append((a, hexcol))
        del_act = menu.addAction("Delete")
        chosen = menu.exec(
            self.pane[which].canvas.mapToGlobal(
                QtPoint(int(sx), int(sy))
            )
        )
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
        cat = getattr(self, "_center_angle_target", None)
        if not cat:
            return
        which = cat["key"]; mi = cat["mi"]
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
            m["center_angle"] = {
                "pts": [p1, p2, p3], "angle": span,
                "t1": t1, "t3": t3, "ccw": ccw,
            }
            self._center_angle_target = None
        self._redraw_meas(which)

    def _add_point(self, which, mi, sx, sy):
        m = self._measures[which][mi]
        wx, wy = self._disp_to_world(which, sx, sy)
        pt = (wx, wy)
        if m["type"] == "ellipse":
            # Ellipse → 4-vertex polygon at the axis endpoints, in order
            # around the perimeter (maj0 → min0 → maj1 → min1).
            maj0, maj1, min0, min1 = m["pts"]
            m["type"] = "polygon"
            m["pts"] = [maj0, min0, maj1, min1]
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

    # ----- measurement geometry / sampling -----
    def _ellipse_params(self, p0, p1):
        cx, cy = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
        a = abs(p1[0] - p0[0]) / 2.0
        b = abs(p1[1] - p0[1]) / 2.0
        return (cx, cy), max(a, 1e-6), max(b, 1e-6)

    def _ellipse_poly(self, p0, p1):
        (cx, cy), a, b = self._ellipse_params(p0, p1)
        return [
            (cx + a * math.cos(t), cy + b * math.sin(t))
            for t in (i * 2 * math.pi / 48 for i in range(49))
        ]

    def _hu_stats(self, key, poly):
        """Min/max HU of the reslice-output pixels inside *poly*."""
        p = self.pane[key]
        p.reslice.Update()
        img = p.reslice.GetOutput()
        ox, oy, _oz = img.GetOrigin()
        sxp, syp, _sz = img.GetSpacing()
        dx, dy, _dz = img.GetDimensions()
        xs = [q[0] for q in poly]
        ys = [q[1] for q in poly]
        i0 = max(0, int((min(xs) - ox) / sxp))
        i1 = min(dx - 1, int((max(xs) - ox) / sxp) + 1)
        j0 = max(0, int((min(ys) - oy) / syp))
        j1 = min(dy - 1, int((max(ys) - oy) / syp) + 1)
        lo, hi = 1e9, -1e9
        for j in range(j0, j1 + 1):
            wy = oy + j * syp
            for i in range(i0, i1 + 1):
                wx = ox + i * sxp
                if _point_in_poly(wx, wy, poly):
                    val = img.GetScalarComponentAsDouble(i, j, 0, 0)
                    lo = min(lo, val)
                    hi = max(hi, val)
        if lo > hi:
            return 0.0, 0.0
        return lo, hi

    # --------------------------------------------------- AbstractViewer
    def load_series(self, loaded: LoadedSeries, title: str) -> None:
        # If this is the same CT series already loaded, keep the viewer's
        # state (camera, crosshair, slab, W/L, measures) so returning to
        # CT from another modality resumes where the user left off
        # instead of reloading from the initial view.
        # Prefer the shell's UID so split rows are not falsely deduped.
        new_uid = (
            loaded.series_uid
            or (str(getattr(loaded.header, "SeriesInstanceUID", ""))
                if loaded.header is not None else "")
        )
        if (self._image is not None and new_uid
                and getattr(self, "_loaded_uid", "") == new_uid):
            return
        self._loaded_uid = new_uid
        vol = loaded.volume
        sr, sc = loaded.spacing_mm or (1.0, 1.0)
        sz = loaded.slice_mm or 1.0
        self._dims = (float(sc), float(sr), float(sz))   # x, y, z mm
        self._image = numpy_to_vtk_image(vol, sc, sr, sz)
        self._bounds = self._image.GetBounds()
        self._header = loaded.header
        # voxel-axis -> patient-LPS rotation (cols=x, rows=y, slices=z).
        # None -> standard axial supine head-first (x=Left, y=Post, z=Head).
        pb = loaded.patient_basis
        self._pbasis = (
            np.asarray(pb, dtype=np.float64)
            if pb is not None else np.eye(3)
        )
        self._win = self._win0 = float(loaded.window or 800.0)
        self._lvl = self._lvl0 = float(loaded.level or 200.0)
        self._thick = {"A": 0.0, "B": 5.0}

        b = self._bounds
        self._center = np.array(
            [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
        )
        self._center0 = self._center.copy()
        self._pc = {"A": self._center.copy(), "B": self._center.copy()}
        self._init_frames()
        # Symmetric square FOV big enough for any oblique plane, so the
        # CrossLine center is always output (0,0) = the pane middle.
        step = max(1e-3, min(self._dims))
        diag = math.sqrt(
            (b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2
        )
        self._half = diag / 2.0
        self._npx = min(1200, max(64, int(2 * self._half / step) + 1))
        self._view_initial = True
        self._sync_slab_spin()
        for key in ("A", "B"):
            p = self.pane[key]
            p.reslice.SetInputData(self._image)
            p.colors.SetLookupTable(self._lut())
        self._refresh(reset_cam=True)

    def clear(self) -> None:
        self._image = None
        self._header = None
        for key in ("A", "B"):
            self.pane[key].reslice.SetInputData(_placeholder_image())
            self.pane[key].info.SetText(0, "")
            self.pane[key].info.SetText(1, "")
            self.pane[key].tagact.SetInput("")
            self.pane[key].resultact.SetInput("")
            self.pane[key].angle.SetInput("")
            self.pane[key].render()

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
        """Right-handed orthonormal (U, V, N) from rough U, V."""
        u = _norm(u)
        v = _norm(v - np.dot(v, u) * u)
        n = np.cross(u, v)
        return (u, v, n)

    def _axes_for(self, key):
        """(U, V, N) world unit vectors for a pane."""
        return self._frame[key]

    def _apex_dir3(self, key):
        """World-space direction the green ▲ on *key* points (its apex) —
        i.e. +uv (perpendicular to that pane's crossline) in 3-D. PAGING
        keys its sign off this so 'up' always slides toward the ▲ apex
        even after crossline rotations re-sign a pane's normal (which
        otherwise silently reverses the page direction)."""
        u, v, _n = self._frame[key]
        a = math.radians(self._cross_ang[key])
        return u * (-math.sin(a)) + v * math.cos(a)

    def _matrix(self, key) -> vtkMatrix4x4:
        u, v, n = self._axes_for(key)
        o = self._pc[key]                     # output (0,0) maps here
        m = vtkMatrix4x4()
        for r in range(3):
            m.SetElement(r, 0, float(u[r]))
            m.SetElement(r, 1, float(v[r]))
            m.SetElement(r, 2, float(n[r]))
            m.SetElement(r, 3, float(o[r]))
        return m

    def _cc(self, key):
        """Crosshair centre for a pane = C projected into its plane,
        relative to that pane's reslice centre (output coords, mm)."""
        u, v, _n = self._axes_for(key)
        delta = self._center - self._pc[key]
        return float(np.dot(delta, u)), float(np.dot(delta, v))

    def _lut(self):
        if self._color:
            return _band_lut(
                self._bands, self._opacity, self._win, self._lvl
            )
        return _gray_lut(self._win, self._lvl)

    def _open_setting(self):
        if self._cmap_dlg is None:
            self._cmap_dlg = _ColorMapDialog(
                self._bands, self._opacity, self._apply_colormap, self
            )
        else:
            # Re-sync the dialog with the viewer's current bands. The
            # viewer is the source of truth (it stored the user's last
            # edit); without this push, the cached dialog could redraw
            # from its own stale internal state and the user's
            # previously-disabled band would appear Enabled again.
            self._cmap_dlg.set_bands(self._bands, self._opacity)
        self._cmap_dlg.show()
        self._cmap_dlg.raise_()
        self._cmap_dlg.activateWindow()

    def _apply_colormap(self, bands, opacity):
        self._bands = [dict(b) for b in bands]
        self._opacity = float(opacity)
        if not self._color:                 # show the result immediately
            self._color = True
            self._cmap_btn.setChecked(True)
        self._refresh()

    def _refresh(self, reset_cam=False):
        if self._image is None:
            return
        step = max(1e-3, min(self._dims))
        half, n = self._half, self._npx
        for key in ("A", "B"):
            p = self.pane[key]
            p.reslice.SetResliceAxes(self._matrix(key))
            p.reslice.SetOutputSpacing(step, step, step)
            p.reslice.SetOutputOrigin(-half, -half, 0.0)
            p.reslice.SetOutputExtent(0, n - 1, 0, n - 1, 0, 0)
            th = self._thick[key]
            if th > 0 and hasattr(p.reslice, "SetSlabModeToMax"):
                p.reslice.SetSlabModeToMax()
                p.reslice.SetSlabNumberOfSlices(
                    max(1, int(round(th / step)))
                )
                if hasattr(p.reslice, "SetSlabSliceSpacingFraction"):
                    p.reslice.SetSlabSliceSpacingFraction(1.0)
            elif hasattr(p.reslice, "SetSlabNumberOfSlices"):
                p.reslice.SetSlabNumberOfSlices(1)
            p.reslice.Modified()
            p.colors.SetLookupTable(self._lut())
            p.colors.Modified()
            # Fit BEFORE drawing the ▲ markers: their size is now tied to
            # the camera's parallel scale, so the camera must already be at
            # its final zoom when _update_cross runs (otherwise the initial
            # view's markers would be sized for the pre-fit scale).
            if reset_cam:
                self._fit_pane(key)
            self._update_cross(key)
            self._update_info(key, title_only=False)
            p.render()

    def _fit_pane(self, key):
        """Fit the actual volume content (projected onto the pane plane)
        into the viewport, not the oversized diagonal-sized FOV square.
        Without this the visible image only fills ~max(W,H)/diag of the
        pane in the initial axial view."""
        p = self.pane[key]
        p.reslice.Update()
        cam = p.ren.GetActiveCamera()
        cam.SetViewUp(0.0, 1.0, 0.0)
        fz = cam.GetFocalPoint()[2]
        pz = cam.GetPosition()[2]
        cam.SetFocalPoint(0.0, 0.0, fz)
        cam.SetPosition(0.0, 0.0, pz)
        hu, hv = self._content_half_on_plane(key)
        pw = max(1, p.canvas.width())
        ph = max(1, p.canvas.height())
        # ParallelScale = half the viewport height in world units. To make
        # the box of half-widths (hu, hv) fit tightly, pick the larger of
        # "fit by height" and "fit by width (converted via aspect ratio)".
        ps = max(hv, hu * ph / pw)
        cam.SetParallelScale(max(1e-3, ps))
        p.render()

    def _content_half_on_plane(self, key):
        """Half-widths (hu, hv) — measured in the pane's plane axes (u, v)
        from output (0,0) — needed to contain the volume's 8 corners
        projected onto that plane. Output (0,0) is at _pc[key] in volume
        coords, so we project (corner - _pc[key]) onto u and v."""
        b = self._image.GetBounds()        # xmin xmax ymin ymax zmin zmax
        u, v, _n = self._frame[key]
        pc = self._pc[key]
        hu = hv = 0.0
        for ix in (0, 1):
            for iy in (2, 3):
                for iz in (4, 5):
                    p = np.array(
                        [b[ix], b[iy], b[iz]], dtype=float
                    ) - pc
                    hu = max(hu, abs(float(np.dot(p, u))))
                    hv = max(hv, abs(float(np.dot(p, v))))
        return hu, hv

    def _update_cross(self, key):
        """Crosshair at world (0,0) — the projected CrossLine center,
        kept at the pane middle by the symmetric FOV. While the user
        drags on the crosshair it rotates to follow the cursor
        (_cross_ang[key]); the horizontal line is where the OTHER pane
        cuts this one and carries the ▲ projection markers and, in
        slab-MIP, the two dashed slab-width lines — all rotating with
        the crosshair."""
        p = self.pane[key]
        h = self._half
        zc = 0.5
        th = math.radians(self._cross_ang[key])
        c, s_ = math.cos(th), math.sin(th)
        uh = np.array([c, s_])           # horizontal line direction
        uv = np.array([-s_, c])          # perpendicular (= ▲ apex dir)
        ccx, ccy = self._cc(key)         # crosshair centre (C's projection)
        cc = np.array([ccx, ccy])

        def pt(a, b, z=zc):              # a along uh, b along uv, from cc
            v = cc + a * uh + b * uv
            return (float(v[0]), float(v[1]), z)

        for (src, _a), (p0, p1) in zip(
            p.cross,
            (((-h, 0.0), (h, 0.0)), ((0.0, -h), (0.0, h))),
        ):
            src.SetPoint1(*pt(*p0))
            src.SetPoint2(*pt(*p1))
            src.Modified()

        # ▲ markers: size & distance are tied to the camera's parallel
        # scale (= half the visible height in mm), NOT the fixed FOV, so
        # they keep a constant on-screen size and a constant fraction of
        # the viewport from the centre at ANY zoom — they no longer
        # balloon or drift off-screen when the user zooms in.
        ps = self.pane[key].ren.GetActiveCamera().GetParallelScale()
        sz = 0.024 * ps
        d = 0.255 * ps
        z = zc + 0.1
        p.tri_mapper.SetInputData(
            _tris_pd([
                (pt(a, sz, z), pt(a - 0.6 * sz, 0.0, z),
                 pt(a + 0.6 * sz, 0.0, z))
                for a in (d, -d)
            ])
        )

        other = "B" if key == "A" else "A"
        t = self._thick[other]
        if t > 0:
            ht = t / 2.0
            dash = max(1.0, 0.012 * h)
            for mp, off in zip(p.slab_mappers, (ht, -ht)):
                mp.SetInputData(
                    _dashed_pd(pt(-h, off), pt(h, off), dash, dash)
                )
        else:
            for mp in p.slab_mappers:
                mp.SetInputData(vtkPolyData())

    def _update_info(self, key, title_only):
        p = self.pane[key]
        head = overlay_lines(
            self._header, self._tag_keywords, anonymized=self._anon
        )
        # Confine the tag block to ~40% width (left corner) by word-wrapping, so
        # a larger font can't run it into the right-corner measure results.
        head = wrap_lines_to_chars(head, self._wrap_budget(key))
        slab = self._thick[key]
        kind = f"Slab MIP {slab:.1f}mm" if slab > 0 else "MPR (thin)"
        p.tagact.SetInput("\n".join(head))          # top-left (fixed-size actor)
        p.info.SetText(
            0, f"WW {self._win:.0f}  WL {self._lvl:.0f}"
        )                                           # bottom-left
        p.info.SetText(1, f"{key}  |  {kind}")      # bottom-right
        p.angle.SetInput(self._angio_angle(key))    # bottom-centre (yellow)

    def _angio_angle(self, key) -> str:
        """SSMview-style C-arm angle of this pane's projection direction.

        The pane is viewed along its normal N. Map N into patient LPS
        (x = Left, y = Posterior, z = Head) via the volume's patient
        basis, then decompose into the DICOM Positioner primary angle
        (LAO + / RAO −, azimuth in the axial plane) and secondary angle
        (CRA + / CAU −, elevation toward the head). Anchors verified
        against the SSMview screen: a coronal view -> "LAO0 CRA0", an
        axial view -> "LAO0 CRA90".
        """
        n = np.asarray(self._frame[key][2], dtype=np.float64)
        nrm = float(np.linalg.norm(n))
        if nrm < 1e-9:
            return ""
        n = self._pbasis @ (n / nrm)                 # -> patient LPS
        nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
        # Resolve the ±N ambiguity to the anterior hemisphere so a
        # frontal/coronal projection reads LAO0 CRA0 (SSMview).
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

    # ----------------------------------------------------------- tools
    def _set_active(self, which):
        self._active_pane = which
        self._sync_slab_spin()
        self._update_active_frames()

    def _update_active_frames(self):
        """Yellow border around the active CT pane (transparent on the
        other, same width so the layout doesn't shift)."""
        for key, f in self._frames.items():
            colr = "#ff2020" if key == self._active_pane else "transparent"
            f.setStyleSheet(
                "QFrame#ctpane { border: 3px solid %s; }" % colr
            )

    def _sync_slab_spin(self):
        self._slab_spin.blockSignals(True)
        self._slab_spin.setValue(self._thick[self._active_pane])
        self._slab_spin.blockSignals(False)

    def _set_slab(self, mm):
        self._thick[self._active_pane] = float(mm)
        self._view_initial = False
        self._refresh()

    def _center_camera(self, key):
        """Put world (0,0) (the crosshair/center) at the pane middle,
        keeping the current zoom."""
        cam = self.pane[key].ren.GetActiveCamera()
        fx, fy, fz = cam.GetFocalPoint()
        px, py, pz = cam.GetPosition()
        cam.SetFocalPoint(0.0, 0.0, fz)
        cam.SetPosition(0.0, 0.0, pz)

    def _drag(self, which, dx, dy, shift=False, sx=None, sy=None):
        if self._image is None:
            return
        t = self._tool
        if t != "WL":
            self._view_initial = False
        if t == "WL":
            self._win = max(1.0, self._win + dx * 2.0)
            self._lvl = self._lvl - dy * 2.0
        elif t == "PAGING":
            # PAGING pages the pane under the cursor itself: the visible
            # image scrolls through slices. We step C and THIS pane's
            # reslice origin together along this pane's normal — their
            # difference is unchanged, so this pane's crosslines stay put
            # while its image pages. The OTHER pane's image is fixed
            # (_pc[other] untouched); C's shift lies in the other pane's
            # plane, so its cross-section line slides there. Sign: dragging
            # UP moves toward the other pane's green-▲ apex. We derive the
            # sign from the ACTUAL displayed ▲ (its apex direction in 3-D),
            # not from the raw normal: a crossline rotation can flip this
            # pane's normal (N_other = crossdir × N_which), which used to
            # silently reverse paging while the ▲ kept pointing the same
            # way. Projecting N onto the other pane's apex keeps "up = apex"
            # invariant (and is +1 in the initial axial/coronal setup).
            _, _, n = self._axes_for(which)
            other = "B" if which == "A" else "A"
            s = 1.0 if float(np.dot(n, self._apex_dir3(other))) >= 0.0 else -1.0
            mv = n * (-dy) * s * min(self._dims)
            self._center = self._center + mv
            self._pc[which] = self._pc[which] + mv
            self._clamp_center()
        elif t == "THICK":
            self._thick[which] = max(
                0.0, self._thick[which] + (dx - dy) * 0.3
            )
            if which == self._active_pane:
                self._sync_slab_spin()
        elif t == "ROTATE":
            # 3-D tilt of THIS pane's plane about its own axes; the
            # other pane is left unchanged. (Direction reversed.)
            u, v, n = self._frame[which]
            u = _rotate(_rotate(u, v, -dx * 0.5), u, -dy * 0.5)
            v = _rotate(_rotate(v, v, -dx * 0.5), u, -dy * 0.5)
            self._frame[which] = self._ortho(u, v)
        elif t == "SPIN":
            # SSMview SPIN = "whole-screen rotation": roll the camera so
            # image AND crosshair/▲/slab rotate together (relationship
            # kept); the other pane is unchanged. Wheel-style: rotate by
            # how far the cursor sweeps AROUND the crosshair centre, so
            # vertical drags work too — right+up / left+down → CCW,
            # right+down / left+up → CW (sign via _SPIN_SIGN).
            if sx is not None:
                # Angle of the cursor about the crosshair centre measured
                # in *screen* pixels (Qt, y-down), so the quadrant feel is
                # exactly as seen: right half drag down → CW, up → CCW;
                # left half reversed; top half L→R → CW, R→L → CCW;
                # bottom half reversed. (Screen space avoids the world
                # y-up / camera-roll confusion.)
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
                        self.pane[which].ren.GetActiveCamera().Roll(
                            _SPIN_SIGN * dphi
                        )
        elif t == "ZOOM":
            # Shift = zoom BOTH panes together, else just this one.
            factor = 1.0 - dy * 0.005
            keys = ("A", "B") if shift else (which,)
            for k in keys:
                cam = self.pane[k].ren.GetActiveCamera()
                cam.SetParallelScale(
                    max(1e-3, cam.GetParallelScale() * factor)
                )
        elif t == "MOVE":
            cam = self.pane[which].ren.GetActiveCamera()
            sc = cam.GetParallelScale() * 0.003
            cam.SetFocalPoint(
                cam.GetFocalPoint()[0] - dx * sc,
                cam.GetFocalPoint()[1] + dy * sc,
                cam.GetFocalPoint()[2],
            )
            cam.SetPosition(
                cam.GetPosition()[0] - dx * sc,
                cam.GetPosition()[1] + dy * sc,
                cam.GetPosition()[2],
            )
        self._refresh()

    def _wheel(self, which, delta):
        if self._image is None:
            return
        # Same contract as the PAGING tool: page the wheeled pane itself —
        # the visible image scrolls through slices. Step C and THIS pane's
        # reslice origin together along this pane's normal. Wheel-up moves
        # toward the other pane's ▲ apex (same sign derivation as PAGING so
        # a crossline-flipped normal can't reverse it).
        _, _, n = self._axes_for(which)
        other = "B" if which == "A" else "A"
        s = 1.0 if float(np.dot(n, self._apex_dir3(other))) >= 0.0 else -1.0
        mv = n * (3 if delta > 0 else -3) * s * min(self._dims)
        self._center = self._center + mv
        self._pc[which] = self._pc[which] + mv
        self._clamp_center()
        self._view_initial = False
        self._refresh()

    def _screen_center(self, which):
        """Qt-widget pixel position (y down) of the crosshair centre."""
        canvas = self.pane[which].canvas
        ren = self.pane[which].ren
        ccx, ccy = self._cc(which)
        ren.SetWorldPoint(float(ccx), float(ccy), 0.0, 1.0)
        ren.WorldToDisplay()
        dx, dy, _dz = ren.GetDisplayPoint()
        h = canvas.height()
        # VTK display coords are render-window physical pixels; our
        # _PaneCanvas.resizeEvent sizes the render window using this
        # widget's own devicePixelRatioF(), so the inverse uses the same.
        dpr = canvas.devicePixelRatioF()
        return dx / dpr, h - dy / dpr          # VTK display y-up -> Qt

    def _disp_to_world(self, which, sx, sy):
        """Screen (sx, sy) on a pane -> that pane's reslice output world
        (wx, wy). Output (0,0) is the CrossLine centre."""
        canvas = self.pane[which].canvas
        ren = self.pane[which].ren
        h = canvas.height()
        # Scale Qt-logical mouse coords to render-window physical pixels
        # using THIS widget's DPR (matches _PaneCanvas.resizeEvent).
        dpr = canvas.devicePixelRatioF()
        ren.SetDisplayPoint(float(sx * dpr), float((h - sy) * dpr), 0.0)
        ren.DisplayToWorld()
        wx, wy, _wz, ww = ren.GetWorldPoint()
        if ww:
            wx, wy = wx / ww, wy / ww
        return wx, wy

    def _recenter(self, which, sx, sy):
        """Double-click: the clicked point becomes the CrossLine center
        AND the image centre, in BOTH panes."""
        if self._image is None:
            return
        wx, wy = self._disp_to_world(which, sx, sy)
        m = self._matrix(which)
        vol = m.MultiplyPoint((wx, wy, 0.0, 1.0))
        self._center = np.array([vol[0], vol[1], vol[2]])
        self._clamp_center()
        # Both panes recentre on the clicked point (image centre + cross
        # centre) -> reset each pane's reslice centre to C.
        self._pc = {"A": self._center.copy(), "B": self._center.copy()}
        self._view_initial = False
        for k in ("A", "B"):
            self._center_camera(k)
        self._refresh()

    def _cross_press(self, which, sx, sy) -> bool:
        """Start a CrossLine gesture, overriding the active tool, when
        the press is on the crosshair. Near the intersection -> MOVE
        (translate the crossline along one of its 2 directions); on a
        line away from the centre -> ROTATE. Follows the current
        crosshair angle so it keeps working after rotation/paging."""
        if self._image is None:
            return False
        wx, wy = self._disp_to_world(which, sx, sy)
        ccx, ccy = self._cc(which)
        rx, ry = wx - ccx, wy - ccy           # relative to crosshair centre
        a = math.radians(self._cross_ang[which])
        uh = (math.cos(a), math.sin(a))       # horizontal-line direction
        uv = (-math.sin(a), math.cos(a))      # vertical-line direction
        tol = max(3.0, 0.02 * self._half)
        ctol = max(6.0, 0.06 * self._half)    # "near the intersection"
        if math.hypot(rx, ry) < ctol:
            self._cross_mode = "move"
            self._cross_axis = None           # locked on first move
            self._cross_ppt = (wx, wy)
            return True
        d_to_h = abs(rx * uv[0] + ry * uv[1])  # dist to horizontal line
        d_to_v = abs(rx * uh[0] + ry * uh[1])  # dist to vertical line
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
            # Translate the crossline along ONE of its 2 directions
            # (locked on first move). The dragged pane's image stays;
            # the OTHER pane reslices through the moved crossline.
            a = math.radians(self._cross_ang[which])
            uh = np.array([math.cos(a), math.sin(a)])
            uv = np.array([-math.sin(a), math.cos(a)])
            d2 = np.array([wx - self._cross_ppt[0],
                           wy - self._cross_ppt[1]])
            self._cross_ppt = (wx, wy)
            if self._cross_axis is None:
                if abs(np.dot(d2, uh)) >= abs(np.dot(d2, uv)):
                    self._cross_axis = uh
                else:
                    self._cross_axis = uv
            amt = float(np.dot(d2, self._cross_axis))
            dir3 = u * self._cross_axis[0] + v * self._cross_axis[1]
            self._center = self._center + amt * dir3
            self._clamp_center()
            self._pc[other] = self._center.copy()   # other follows line
            self._view_initial = False
            self._refresh()
            return

        # ROTATE: crosshair follows the cursor about its centre; the
        # dragged pane's image is fixed; the OTHER pane is re-derived
        # as the section cut along the rotated crossline.
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

    def _clamp_center(self):
        b = self._bounds
        self._center = np.array([
            min(max(self._center[0], b[0]), b[1]),
            min(max(self._center[1], b[2]), b[3]),
            min(max(self._center[2], b[4]), b[5]),
        ])

    def _toggle_color(self):
        self._color = self._cmap_btn.isChecked()
        self._refresh()

    def _reset(self):
        """1st click (view moved): keep W/L, restore the view position.
        Click again at the initial position: also reset W/L."""
        if self._image is None:
            return
        if not self._view_initial:
            self._center = self._center0.copy()
            self._pc = {"A": self._center.copy(), "B": self._center.copy()}
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
