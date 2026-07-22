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
from PyQt6.QtCore import QPoint as QtPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QKeySequence, QShortcut
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
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
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
    vtkActor2D,
    vtkBillboardTextActor3D,
    vtkImageActor,
    vtkPolyDataMapper,
    vtkPolyDataMapper2D,
    vtkRenderer,
    vtkRenderWindow,
    vtkTextActor,
    vtkWindowToImageFilter,
)

# Rendering / interaction implementations VTK loads lazily.
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
import vtkmodules.vtkInteractionStyle  # noqa: F401
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser

from multi_dicomviewer.config import CT_WL_PRESETS
from multi_dicomviewer.i18n import t
from multi_dicomviewer.viewers.cpr_mixin import CPRMixin
from multi_dicomviewer.core.dicom_io import LoadedSeries
from multi_dicomviewer.core.dicom_tags import overlay_lines
from multi_dicomviewer.core.image_export import (
    export_image_as, pick_export_format, safe_basename,
)
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
    arc_through as _arc_through,
    central_arc_angle as _central_arc_angle,
    convex_hull as _convex_hull,
    dist as _dist,
    ellipse_cab as _ellipse_cab_pure,
    ellipse_drag as _ellipse_drag,
    ellipse_from_major as _ellipse_from_major,
    ellipse_outline as _ellipse_outline,
    gap_color as _gap_color,
    gap_legend as _gap_legend,
    gap_linewidth as _gap_linewidth,
    major_minor as _major_minor_pure,
    min_width as _min_width,
    percent_area_diff as _percent_area_diff,
    point_in_poly as _point_in_poly,
    poly_area as _poly_area,
    polygon_centroid as _polygon_centroid,
    project_to_polyline as _project_to_polyline,
    radial_gap_compare as _radial_gap_compare,
    seg_dist as _seg_dist,
    smooth_closed as _smooth_closed,
    smooth_open as _smooth_open,
)
from multi_dicomviewer.core.centerline import CenterLine
from multi_dicomviewer.ui.compare_options import CompareOptionsDialog
from multi_dicomviewer.ui.viewer_base import AbstractViewer
from multi_dicomviewer.ui.study_browser import FitButton
from multi_dicomviewer.ui.measure_style_menu import (
    add_color_submenu, add_transparency_submenu, transp_to_alpha,
)

_TOOLS = ("ZOOM", "MOVE", "ROTATE", "SPIN", "PAGING", "THICK", "WL")
#: Button captions carry the keyboard shortcut so it's discoverable on-screen.
_TOOL_LABELS = {
    "ZOOM": "Zoom (Z)", "MOVE": "Move (V)", "ROTATE": "Rotate (R)",
    "SPIN": "Spin (S)", "PAGING": "Paging (G)", "THICK": "Thick (T)",
    "WL": "WL (W)",
}
#: Tools that only make sense in 3-D MPR mode — disabled in 2-D (single-slice)
#: mode, where the pane shows native acquisition slices.
_MPR_ONLY_TOOLS = ("ROTATE", "SPIN", "THICK")
#: Series with this many slices or fewer default to 2-D (single-slice) display;
#: more than this defaults to 3-D MPR reconstruction.
_MODE_2D_MAX = 200
#: 2-D frame scrubber handle: a 20 px white disc with a blue inner dot (radial
#: gradient), matching the cine viewer's seek-bar thumb. The negative handle
#: margin makes it overflow the 6 px groove, so the slider reserves extra height
#: (setMinimumHeight in _build_seek_bar) to keep the disc from clipping top/bottom.
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


def _polylines_pd(polylines, z=0.72) -> vtkPolyData:
    """One polydata holding several connected polylines (each a list of (x, y)
    output-plane points) at depth *z*. Used for the rotate-hint curved arrow."""
    pd = vtkPolyData()
    pts = vtkPoints()
    lines = vtkCellArray()
    idx = 0
    for poly in polylines:
        n = len(poly)
        if n < 2:
            continue
        lines.InsertNextCell(n)
        for (x, y) in poly:
            pts.InsertNextPoint(float(x), float(y), float(z))
            lines.InsertCellPoint(idx)
            idx += 1
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


def _rgba(c):
    """Normalise a colour to a 4-tuple RGBA uint8 (pad opaque if no alpha)."""
    return (int(c[0]), int(c[1]), int(c[2]), int(c[3]) if len(c) > 3 else 255)


def _colored_multi_pd(polylines, colors) -> vtkPolyData:
    """Same as _multi_pd but each polyline cell carries its own RGBA colour
    (uint8; 3-tuples are treated as opaque). Caller's mapper must enable
    cell-scalar direct-colour; alpha < 255 makes that cell translucent."""
    pd = vtkPolyData()
    vp = vtkPoints()
    lines = vtkCellArray()
    cell_rgb = vtkUnsignedCharArray()
    cell_rgb.SetNumberOfComponents(4)
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
            cell_rgb.InsertNextTuple4(*_rgba(rgb))
        base += n
    pd.SetPoints(vp)
    pd.SetLines(lines)
    if cell_rgb.GetNumberOfTuples() > 0:
        pd.GetCellData().SetScalars(cell_rgb)
    return pd


def _filled_tris_pd(tris, colors) -> vtkPolyData:
    """Polydata of filled triangles (each ``(p0, p1, p2)`` of (x,y) points) with
    a per-cell RGBA colour (uint8; 3-tuples opaque). Used for the translucent
    compare-fill annulus — alpha now lives in the cells (per-result)."""
    pd = vtkPolyData()
    vp = vtkPoints()
    polys = vtkCellArray()
    cell_rgb = vtkUnsignedCharArray()
    cell_rgb.SetNumberOfComponents(4)
    cell_rgb.SetName("Colors")
    base = 0
    for (p0, p1, p2), rgb in zip(tris, colors):
        vp.InsertNextPoint(float(p0[0]), float(p0[1]), 0.55)
        vp.InsertNextPoint(float(p1[0]), float(p1[1]), 0.55)
        vp.InsertNextPoint(float(p2[0]), float(p2[1]), 0.55)
        polys.InsertNextCell(3)
        polys.InsertCellPoint(base)
        polys.InsertCellPoint(base + 1)
        polys.InsertCellPoint(base + 2)
        cell_rgb.InsertNextTuple4(*_rgba(rgb))
        base += 3
    pd.SetPoints(vp)
    pd.SetPolys(polys)
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


def _gray_lut(width: float, level: float, invert: bool = False) -> vtkLookupTable:
    lut = vtkLookupTable()
    lut.SetHueRange(0.0, 0.0)
    lut.SetSaturationRange(0.0, 0.0)
    # invert = black↔white negative (low HU → white, high HU → black).
    lut.SetValueRange(1.0, 0.0) if invert else lut.SetValueRange(0.0, 1.0)
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
        # Track motion with no button down so the crosshair can preview (vivid
        # highlight + rotate arrow) whether a press would grab the centreline.
        self.setMouseTracking(True)

    def mousePressEvent(self, e):
        self._owner._set_active(self._which)
        # Short-axis (CPR) pane: a left-press on a control-point marker grabs it
        # to fine-tune the pseudo-centre in the cross-section (highest priority).
        if (e.button() == Qt.MouseButton.LeftButton
                and self._owner._cpr is not None and self._which == "A"
                and self._owner._cpr_grab(e.position().x(), e.position().y())):
            self._last = e.position()
            self._cross = False
            self._meas_drag = False
            return
        # Otherwise a left-drag on the short-axis is dispatched by the SELECTED
        # tool so the toolbar works here too: Rotate/Spin turn the cross-section
        # (feeds the CoSync rotation), Paging scrolls the pull-back, and
        # WL/Zoom/Move fall through to the normal _drag (window-level, camera
        # zoom / pan all act on this pane).
        if (e.button() == Qt.MouseButton.LeftButton
                and self._owner._cpr is not None and self._which == "A"):
            self._cross = False
            self._meas_drag = False
            self._last = e.position()
            self._cpr_page_anchor = e.position().y()
            if self._owner._tool in ("ROTATE", "SPIN"):
                self._owner._cpr_rot_start(
                    e.position().x(), e.position().y())
            # PAGING scrolls; WL / ZOOM / MOVE / THICK → _drag (mouseMoveEvent)
            return
        # Compare-select mode: a left-click picks the two shapes to compare
        # (no drag: clear _last so a move can't pan/rotate the image).
        if (e.button() == Qt.MouseButton.LeftButton
                and self._owner._cmp_on):
            self._last = None
            self._cross = False
            self._owner._compare_pick(
                self._which, e.position().x(), e.position().y()
            )
            return
        # Right-click ON the bottom-centre angio readout → angle dialog
        # (rotate the slice to match a chosen LAO/RAO·CRA/CAU view). Checked
        # first, in any tool/measure mode, since it's a fixed screen target.
        if e.button() == Qt.MouseButton.RightButton and self._owner._angio_hit(
                self._which, e.position().x(), e.position().y(),
                self.width(), self.height()):
            self._owner._open_angio_dialog(self._which)
            return
        # Right-click priority: a measure LINE/handle gets its own menu first
        # (Hide that line / Delete); only an EMPTY spot inside a compare region
        # falls through to the region's colour Hide/Delete menu.
        if e.button() == Qt.MouseButton.RightButton:
            self._cross = False
            self._meas_drag = False
            self._last = None
            if self._owner._meas_on and self._owner._measure_right(
                    self._which, e.position().x(), e.position().y()):
                return                        # handled a measure line/handle
            ci = self._owner._compare_hit(
                self._which, e.position().x(), e.position().y())
            if ci is not None:
                self._owner._compare_delete_menu(self._owner._compares[ci])
                return
            if self._owner._meas_on:
                return                        # right-click in measure mode = no-op
            self._owner._export_pane(         # not measuring → still-image export
                self._which, e.position().x(), e.position().y()
            )
            return
        # Shift while measuring temporarily runs the SELECTED tool instead of
        # drawing (Rotate/Spin/Paging move the cutting plane to chase a vessel
        # off the slab; Move/Zoom just help you look) — so you can reorient the
        # plane mid-trace without leaving Measure. Falls through to the same
        # tool path idle-Measure uses.
        _shift = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if self._owner._meas_on and not _shift:
            self._cross = False
            self._meas_drag = False
            # Left-click MEASURES while a type is selected or a Center-Angle
            # pick is in progress, and it also grabs an existing measure's
            # handle. Otherwise (Measure on but no type chosen) it falls
            # through to the active tool below, so Zoom/Move/… still work —
            # which is why the tools are only greyed once a type is selected.
            capturing = (bool(self._owner._meas_type)
                         or self._owner._center_angle_target is not None)
            started = self._owner._measure_left(
                self._which, e.position().x(), e.position().y()
            )
            if capturing or started:
                self._meas_drag = bool(started)
                self._last = e.position() if started else None
                return
            # idle Measure mode → fall through to the tool / crosshair setup
        self._owner._spin_prev = None        # restart SPIN wheel angle
        # Pressing within the (5%) crosshair grab band grabs the centreline
        # (reslice move / rotate), overriding the tool — for ALL tools. The
        # centreline gesture is deliberately ABOVE every tool button: on the
        # lines, grabbing the centreline always wins. Hover feedback (the vivid
        # highlight + rotate arrow, see _hover_cross) tells the user, before the
        # click, whether the press will pan/tool, parallel-move or rotate.
        self._cross = self._owner._cross_press(
            self._which, e.position().x(), e.position().y()
        )
        self._last = e.position()

    def mouseMoveEvent(self, e):
        # Short-axis (CPR) pane, left-drag in progress → dispatch by tool.
        if (self._owner._cpr is not None and self._which == "A"
                and self._last is not None):
            if self._owner._cpr_drag is not None:        # moving a control point
                self._owner._cpr_drag_move(
                    e.position().x(), e.position().y())
                return
            _tool = self._owner._tool
            if _tool in ("ROTATE", "SPIN"):
                self._owner._cpr_rot_move(e.position().x(), e.position().y())
                return
            if _tool == "PAGING":
                self._owner._cpr_page_drag(
                    e.position().y() - self._cpr_page_anchor)
                self._cpr_page_anchor = e.position().y()
                return
            # WL / ZOOM / MOVE / THICK → the normal tool drag on this pane.
            p = e.position()
            shift = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self._owner._drag(self._which, p.x() - self._last.x(),
                              p.y() - self._last.y(), shift, p.x(), p.y())
            self._last = p
            return
        if self._owner._meas_on and self._meas_drag:
            self._owner._measure_drag(
                self._which, e.position().x(), e.position().y()
            )
            return
        # Measure-on but idle (no active measure drag) falls through to the
        # tool, so Zoom/Move/… work when no measure type is selected. When a
        # type IS active, the press left _last = None, so the guard below
        # returns and no tool drag happens.
        if self._last is None:
            # No button down → hover-preview the centreline grab (unless a
            # measure type is armed, where a click starts a measurement).
            if not (self._owner._meas_on and self._owner._meas_type):
                self._owner._hover_cross(
                    self._which, e.position().x(), e.position().y())
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
        if self._owner._cpr_drag is not None:
            self._owner._cpr_drag_end()          # rebuild centreline on release
            self._last = None
            return
        if self._owner._cpr_rot_prev is not None:
            self._owner._cpr_rot_end()
            self._last = None
            return
        if self._owner._meas_on and self._meas_drag:
            self._owner._measure_release()
        # Commit an intersection recentre: the point held under the cursor
        # during the drag now jumps to the image centre (the background was
        # fixed while dragging). Done on RELEASE only.
        if self._cross and self._owner._cross_mode == "center":
            self._owner._recenter(
                self._which, e.position().x(), e.position().y())
        self._meas_drag = False
        self._last = None
        self._cross = False
        self._owner._spin_prev = None
        # End the centreline gesture and drop its highlight (a fresh hover will
        # re-preview if the cursor is still on a line).
        self._owner._cross_dragging = None
        self._owner._set_cross_highlight(self._which, None, None)

    def leaveEvent(self, e):
        # Cursor left the pane → clear any hover highlight.
        if self._owner._cross_dragging is None:
            self._owner._set_cross_highlight(self._which, None, None)
        super().leaveEvent(e)

    def mouseDoubleClickEvent(self, e):
        _shift = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        # Shift+double-click recentres even while Measuring (the trace follows
        # the moved image, see _recenter → _redraw_meas). A plain double-click
        # in Measure mode still finishes the polyline draft.
        if self._owner._meas_on and not _shift:
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
        # Curved rotate-arrow shown beside a caught centreline while hovering /
        # dragging in the OUTER (rotate) zone — a hint that the gesture rotates
        # rather than parallel-moves. Empty + hidden until _set_cross_highlight
        # fills it. NOT added to _overlay_actors (it must stay hidden in the
        # "Max Image" presentation mode where overlays are toggled off).
        self.rot_arrow_mapper = vtkPolyDataMapper()
        self.rot_arrow_mapper.SetInputData(vtkPolyData())
        self.rot_arrow = vtkActor()
        self.rot_arrow.SetMapper(self.rot_arrow_mapper)
        self.rot_arrow.GetProperty().SetColor(0.8, 0.8, 0.0)    # yellow (dimmed)
        self.rot_arrow.GetProperty().SetLineWidth(1.4)
        self.rot_arrow.SetVisibility(False)
        self.ren.AddActor(self.rot_arrow)
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
        ma.GetProperty().SetLineWidth(1.8)         # 1.5 ×1.2 — readability
        # GL line width is clamped to 1px on many GPUs, so widths above only
        # show when lines are rendered as tubes.
        if hasattr(ma.GetProperty(), "SetRenderLinesAsTubes"):
            ma.GetProperty().SetRenderLinesAsTubes(True)
        self.ren.AddActor(ma)
        # Compare filled region (annulus between the two outlines), translucent
        # (~35% opacity = 65% transparent); per-cell RGB = outer shape colour.
        # Also the right-click → Delete hit area.
        self.cmp_fill_mapper = vtkPolyDataMapper()
        self.cmp_fill_mapper.SetInputData(vtkPolyData())
        self.cmp_fill_mapper.ScalarVisibilityOn()
        self.cmp_fill_mapper.SetScalarModeToUseCellData()
        self.cmp_fill_mapper.SetColorModeToDirectScalars()
        cfa = vtkActor()
        cfa.SetMapper(self.cmp_fill_mapper)
        cfa.GetProperty().SetOpacity(1.0)    # per-result alpha lives in the cells
        self.ren.AddActor(cfa)
        # Compare radial gap map: thin lines between the two outlines, each
        # cell coloured by its gap band (set by _redraw_compare).
        self.cmp_mapper = vtkPolyDataMapper()
        self.cmp_mapper.SetInputData(vtkPolyData())
        self.cmp_mapper.ScalarVisibilityOn()
        self.cmp_mapper.SetScalarModeToUseCellData()
        self.cmp_mapper.SetColorModeToDirectScalars()
        cma = vtkActor()
        cma.SetMapper(self.cmp_mapper)
        cma.GetProperty().SetLineWidth(1.6)
        if hasattr(cma.GetProperty(), "SetRenderLinesAsTubes"):
            cma.GetProperty().SetRenderLinesAsTubes(True)
        self.ren.AddActor(cma)
        # Compare radial gap map — the hottest (<5 mm red) band, drawn thicker
        # so it stands out (separate actor: VTK line width is per-actor).
        self.cmp_red_mapper = vtkPolyDataMapper()
        self.cmp_red_mapper.SetInputData(vtkPolyData())
        self.cmp_red_mapper.ScalarVisibilityOn()
        self.cmp_red_mapper.SetScalarModeToUseCellData()
        self.cmp_red_mapper.SetColorModeToDirectScalars()
        crm = vtkActor()
        crm.SetMapper(self.cmp_red_mapper)
        crm.GetProperty().SetLineWidth(4.0)
        if hasattr(crm.GetProperty(), "SetRenderLinesAsTubes"):
            crm.GetProperty().SetRenderLinesAsTubes(True)
        self.ren.AddActor(crm)
        # Compare selection highlight: the two picked outlines (cyan, thick).
        self.cmp_sel_mapper = vtkPolyDataMapper()
        self.cmp_sel_mapper.SetInputData(vtkPolyData())
        self.cmp_sel_mapper.ScalarVisibilityOn()
        self.cmp_sel_mapper.SetScalarModeToUseCellData()
        self.cmp_sel_mapper.SetColorModeToDirectScalars()
        csa = vtkActor()
        csa.SetMapper(self.cmp_sel_mapper)
        csa.GetProperty().SetLineWidth(6.0)
        if hasattr(csa.GetProperty(), "SetRenderLinesAsTubes"):
            csa.GetProperty().SetRenderLinesAsTubes(True)
        self.ren.AddActor(csa)
        self.cmp_text = []      # legend vtkTextActors (rebuilt each draw)
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
        mxa.GetProperty().SetLineWidth(2.64)        # 2.2 ×1.2 — readability
        mxa.GetProperty().SetOpacity(0.65)
        if hasattr(mxa.GetProperty(), "SetRenderLinesAsTubes"):
            mxa.GetProperty().SetRenderLinesAsTubes(True)
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
        # Match the diameter/spoke weight used on Mac (2.64); CA spokes were
        # anomalously thin (1.2) before.
        mca.GetProperty().SetLineWidth(2.64)
        mca.GetProperty().SetOpacity(0.8)
        if hasattr(mca.GetProperty(), "SetRenderLinesAsTubes"):
            mca.GetProperty().SetRenderLinesAsTubes(True)
        self.ren.AddActor(mca)
        # Center-Angle arc: the perimeter span p1→p3 (through p2), drawn SOLID
        # in orange and thicker than the outline so it stands out (parity with
        # the Mac viewer); its own actor so it isn't tied to the outline width.
        self.meas_arc_mapper = vtkPolyDataMapper()
        self.meas_arc_mapper.SetInputData(vtkPolyData())
        self.meas_arc_mapper.ScalarVisibilityOn()
        self.meas_arc_mapper.SetScalarModeToUseCellData()
        self.meas_arc_mapper.SetColorModeToDirectScalars()
        mar = vtkActor()
        mar.SetMapper(self.meas_arc_mapper)
        mar.GetProperty().SetColor(1.0, 0.55, 0.0)
        mar.GetProperty().SetLineWidth(2.88)
        if hasattr(mar.GetProperty(), "SetRenderLinesAsTubes"):
            mar.GetProperty().SetRenderLinesAsTubes(True)
        self.ren.AddActor(mar)
        # VTK line width is in render-window PHYSICAL pixels, and this canvas
        # sizes its render window by devicePixelRatio — so on a scaled Windows
        # display the lines render thinner than the equivalent Qt-drawn IVUS
        # lines. Re-apply these base widths × DPR every redraw (see _redraw_geom)
        # so CT matches. (base, actor) pairs:
        # Base widths ×1.2 again (VTK still read a touch thinner than the Qt
        # viewers after DPR scaling): 1.8→2.16, 2.64→3.168, 2.88→3.456.
        self._meas_line_actors = [
            (2.16, ma), (3.168, mxa), (3.168, mca), (3.456, mar),
        ]
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
        # Vessel-trace vertices that lie OFF the current cutting plane: drawn
        # at 50% opacity as a depth cue (they sit in front of / behind the
        # shown slice). Same yellow, half alpha.
        self.meas_pts_off_mapper = vtkPolyDataMapper()
        self.meas_pts_off_mapper.SetInputData(vtkPolyData())
        mpo = vtkActor()
        mpo.SetMapper(self.meas_pts_off_mapper)
        mpo.GetProperty().SetColor(1.0, 0.85, 0.0)
        mpo.GetProperty().SetPointSize(10.0)
        mpo.GetProperty().SetOpacity(0.5)             # off-plane = 50%
        if hasattr(mpo.GetProperty(), "SetRenderPointsAsSpheres"):
            mpo.GetProperty().SetRenderPointsAsSpheres(True)
        self.ren.AddActor(mpo)
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
        # coords (stays centred & same size at any pane/zoom). Font 18 =
        # the old 15 ×1.2 (clinician asked for a larger, easier-to-read tag).
        #
        # Black outline ("縁取り"): vtkTextProperty has no glyph stroke, so the
        # outline is 8 black copies of the same text nudged ±a few px around
        # the centre, drawn UNDER the yellow one — a halo that keeps the
        # yellow legible even when the slice background turns white.
        # A single ring of copies looks blotchy (斑) once the offset is large
        # enough that the 8 shadows no longer overlap — you see them as
        # discrete images. So use TWO concentric rings: a tight inner ring
        # (like the clean DICOM-tag outline) fills the near gaps, an outer ring
        # adds a touch of thickness; their union reads as one smooth outline.
        _ANGLE_FONT = 18
        self.angle_halo = []
        for _r in (0.002, 0.004):                # inner (fill) + outer (thickness)
            for _ox, _oy in ((-1, -1), (0, -1), (1, -1), (-1, 0),
                             (1, 0), (-1, 1), (0, 1), (1, 1)):
                ha = vtkTextActor()
                ha.SetTextScaleModeToNone()
                htp = ha.GetTextProperty()
                htp.SetColor(0.0, 0.0, 0.0)
                htp.SetFontSize(_ANGLE_FONT)
                htp.SetBold(True)
                htp.SetJustificationToCentered()
                htp.SetVerticalJustificationToBottom()
                ha.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                ha.GetPositionCoordinate().SetValue(0.5 + _ox * _r,
                                                    0.012 + _oy * _r)
                ha.SetInput("")
                self.ren.AddViewProp(ha)         # added BEFORE the yellow actor
                self.angle_halo.append(ha)
        self.angle = vtkTextActor()
        self.angle.SetTextScaleModeToNone()
        self.angle.GetTextProperty().SetColor(1.0, 0.9, 0.0)
        self.angle.GetTextProperty().SetFontSize(_ANGLE_FONT)
        self.angle.GetTextProperty().SetBold(True)
        self.angle.GetTextProperty().SetJustificationToCentered()
        self.angle.GetTextProperty().SetVerticalJustificationToBottom()
        self.angle.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.angle.GetPositionCoordinate().SetValue(0.5, 0.012)
        self.angle.SetInput("")
        # Always visible (like the WW/WL corner), independent of the
        # CenterLine overlay toggle — it's clinical info, not clutter.
        # Added AFTER the halo so the yellow text sits on top of the outline.
        self.ren.AddViewProp(self.angle)

        # DICOM tags (top-left) and measure results (top-right) render as their
        # OWN fixed-size text actors, NOT vtkCornerAnnotation corners. The
        # annotation auto-fits its font to the viewport, which made the tag-size
        # slider ineffective and let long lines sprawl across the image. These
        # honour the slider's exact pixel size; the viewer word-wraps their text
        # to ~40% of the pane width so left tags and right results never collide.
        def _mk_overlay_text(justify_right, rgb, outline=False):
            base_x = 0.988 if justify_right else 0.012

            def _one(color):
                a = vtkTextActor()
                a.SetTextScaleModeToNone()          # fixed font size, no auto-fit
                tp = a.GetTextProperty()
                tp.SetColor(*color)
                tp.SetFontFamilyToArial()
                _set_vtk_tag_font(tp)               # Meiryo via file when present
                tp.SetLineSpacing(_VTK_TAG_LINE_SPACING)
                tp.SetFontSize(_vtk_font_px(TAG_FONT_PT_DEFAULT))
                tp.SetVerticalJustificationToTop()
                tp.SetJustificationToRight() if justify_right \
                    else tp.SetJustificationToLeft()
                a.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                a.SetInput("")
                return a

            # Thin black outline ("細めの黒枠"): black copies nudged ±~1px around
            # the text, drawn UNDER it so e.g. white tags stay legible on a white
            # slice. Added before the main actor so the colour sits on top.
            halos = []
            if outline:
                _td = 0.002
                for ox, oy in ((-1, -1), (0, -1), (1, -1), (-1, 0),
                               (1, 0), (-1, 1), (0, 1), (1, 1)):
                    ha = _one((0.0, 0.0, 0.0))
                    ha.GetPositionCoordinate().SetValue(base_x + ox * _td,
                                                        0.985 + oy * _td)
                    self.ren.AddViewProp(ha)
                    halos.append(ha)
            a = _one(rgb)
            a.GetPositionCoordinate().SetValue(base_x, 0.985)
            return a, halos

        # Tags white (with a thin black outline so they read over bright/white
        # slices); results yellow (255,217,0) to match the other modalities.
        self.tagact, self.tagact_halo = _mk_overlay_text(
            False, (1.0, 1.0, 1.0), outline=True)               # top-left
        self.resultact, _ = _mk_overlay_text(True, (1.0, 0.851, 0.0))  # top-right
        self.ren.AddViewProp(self.tagact)
        self.ren.AddViewProp(self.resultact)

        self.canvas.GetRenderWindow().AddRenderer(self.ren)

    def set_overlay_visible(self, on: bool) -> None:
        for a in self._overlay_actors:
            a.SetVisibility(bool(on))
        # The rotate-hint arrow is not in _overlay_actors (so it stays hidden by
        # default), but if presentation mode turns overlays OFF mid-hover, drop
        # it too so no stray arrow lingers.
        if not on:
            self.rot_arrow.SetVisibility(False)

    def render(self):
        # Skip the GL Render while this pane's canvas is not actually on screen
        # (a hidden QStackedWidget page during load_series, or the inactive pane
        # in single-view). Rendering an unmapped native window binds VTK's GL
        # context to a bad HDC — wglMakeCurrent fails ("error 2004" flood) and
        # the pane comes up BLACK until a manual reload. CTViewer.showEvent
        # re-renders once we are shown, so the first Render always targets a
        # mapped, DPI-settled window. (Windows/WGL; harmless no-op elsewhere.)
        if not self.canvas.isVisible():
            return
        self.canvas.GetRenderWindow().Render()


class _AngioAngleDialog(QDialog):
    """Pick a C-arm view (LAO/RAO primary + CRA/CAU secondary, each with a
    degree value) to rotate the CT slice to. Opened by right-clicking the
    bottom-centre angio readout; pre-filled with the pane's current angle so
    small tweaks are easy. values() returns signed degrees (LAO+/RAO−,
    CRA+/CAU−) to feed _set_angio_angle."""

    def __init__(self, prim, sec, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Angio Angle"))
        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            t("Rotates the CT slice to match the angle of the "
              "corresponding angio view")))

        r1 = QHBoxLayout()
        self._lr = QComboBox()
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
        self._cc = QComboBox()
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
    """SSMview-style HU colour-map editor: a list of colour bands (colour
    + HU Min/Max + enable/remove), an Opacity slider, Add and Reset.
    Changes apply live via on_change(bands, opacity)."""

    def __init__(self, bands, opacity, on_change, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("ColorMap Setting"))
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
            self,
            t("Band colour"),
        )
        if col.isValid():
            self._bands[idx]["rgb"] = (
                col.redF(), col.greenF(), col.blueF()
            )
            self._rebuild()
            self._emit()


# --------------------------------------------------------------- viewer
class CTViewer(CPRMixin, AbstractViewer):
    handles_modality = "CT"
    #: emitted by the series-navigation buttons ("first"/"prev"/"next"/"last")
    #: — the shell steps through this study's CT series (angio-style F/A nav)
    series_nav = pyqtSignal(str)
    #: CPR short-axis scrub position changed (arc-length sample index) — the CT
    #: analogue of a cine's frame_changed, so the shell can CoSync it against an
    #: IVUS pull-back / another short-axis.
    cpr_index_changed = pyqtSignal(int)
    #: CPR short-axis in-plane rotation changed (degrees) — the analogue of the
    #: IVUS cross-section rotation, for the CoSync rotation interpolation (按分).
    cpr_rotation_changed = pyqtSignal(float)
    tags_requested = pyqtSignal()
    #: emitted when the tag-text-size slider moves (shell broadcasts the pt to
    #: every viewer so the overlay size matches across modalities)
    overlay_font_changed = pyqtSignal(int)
    #: emitted when a measurement is committed — the shell files it under
    #: the current study so it shows in the shared Measure History.
    measurement_added = pyqtSignal(object)
    #: emitted when the user clicks "Measure History"
    history_requested = pyqtSignal()
    #: image right-click ▸ Export DICOM / CSV → shell runs that export for the
    #: shown CT series. Args: (fmt, series_uid, plane_path); CT always passes
    #: plane_path="" (one volume — A/B panes are reformats of the same data).
    plane_export_requested = pyqtSignal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Unified button look (matches the Angio/IVUS + Mac viewers): a light-
        # grey rounded border + consistent padding/background on EVERY button.
        # Active/coloured buttons override only background+colour, so they keep
        # this shape and size.
        self.setStyleSheet(
            "QPushButton {"
            " border:1px solid #c8c8c8; border-radius:6px;"
            " padding:3px 8px; background:#ededed; color:#101010; }")
        self._results_hidden = False         # "Hide/Show All Result" global toggle
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
        self._mode = "3D"                    # "3D" MPR | "2D" native slices
        # Curved-MPR / short-axis reformat state. None = off; when active it
        # holds the centreline + per-sample short-axis frames and drives pane
        # A as a cross-section scroller (see _enter_cpr / _refresh). Pane B
        # keeps its normal MPR so the trace stays visible as a map.
        self._cpr = None
        self._cpr_marker_pts = []            # [(ctrl_idx, (du,dv))] for hit-test
        self._cpr_drag = None                # control index being dragged
        self._cpr_rot_prev = None            # cursor angle while rotating (rad)
        self._vol = None                     # (z,y,x) HU volume for lumen snap
        # Auto-snap a traced vertex to the brightest (contrast lumen) point
        # along the plane normal, near the click — the MIP shows WHERE the
        # vessel is bright, snap recovers its DEPTH. Toggle in the trace menu.
        self._snap_lumen = True
        self._slice2d = 0                    # current slice index in 2-D mode
        self._page_accum = 0.0               # 2-D drag-paging pixel accumulator
        self._side = "Bi"                    # last 3-D Plane choice (Bi/Lt/Rt)
        # 2-D display in-plane axes (output right = U, up = V); rotated/flipped
        # by the Rt90/Lt90/Flip buttons. Default V = -y so the stored slice is
        # shown in raster order (pixel row 0 at the TOP, like any 2-D DICOM
        # viewer): the camera puts +V up, while DICOM rows grow downward.
        # V = +y (the old default) showed every native slice upside-down.
        self._axes2d = (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]))
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
        self._invert = False                 # grayscale black↔white negative
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
        # Compare (%Area + radial gap map between two Polygon/Ellipse outlines)
        self._cmp_on = False                 # Compare-select mode: click 2 shapes
        self._cmp_sel = []                   # [(key, mi)] picked shapes (max 2)
        self._compares = []                  # persisted results (right-click→Delete)
        self._cmp_want_pa = False            # last-used: compute %PA (IVUS)
        self._cmp_want_thk = True            # last-used: compute Thickness (CT LV)
        self._active_pane = "A"
        self._view_initial = True            # for the 2-stage Reset
        self._cross_prev = 0.0               # CrossLine-rotate prev angle
        self._spin_prev = None               # SPIN wheel previous angle
        self._cross_mode = "rotate"          # "rotate" | "move"
        self._cross_axis = None              # locked move axis (2-D unit)
        self._cross_ppt = (0.0, 0.0)         # prev world point (move mode)
        # Hover/drag centreline highlight: which pane is mid-drag, and the
        # (line, mode) currently highlighted per pane (None = normal crosshair).
        self._cross_dragging = None          # "A"/"B" while a cross gesture runs
        self._cross_hi = {"A": None, "B": None}   # (line, mode) or None
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
        lay.addWidget(self._build_seek_bar())
        lay.addWidget(self._build_cpr_bar())

        for c in (self.canvas_a, self.canvas_b):
            c.Initialize()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Arrow keys drive the active tool. Register them as QShortcuts scoped
        # to this viewer (WidgetWithChildrenShortcut) rather than handling them
        # in keyPressEvent: a focused child (the Slab spin-box, W/L combo or the
        # tag-size slider) would otherwise swallow the arrows for its own value
        # stepping, making the tool-arrow operation fire only sometimes (the
        # "unstable / changes the tag size instead of rotating" bug). A shortcut
        # is processed before the focused child sees the key, so arrows always
        # drive the tool regardless of which control last had focus.
        for seq, direction in (("Up", "up"), ("Down", "down"),
                               ("Left", "left"), ("Right", "right")):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda d=direction: self._key_arrow(d))
        self._update_active_frames()

    def showEvent(self, e):
        """Fit and paint the FIRST Render once the pane is actually on screen.

        The shell calls ``load_series`` BEFORE it brings this viewer to the front
        of the pane's QStackedWidget (see MainWindow.show_series), so load_series
        runs while our canvases are still a hidden page. ``_Pane.render`` suppr-
        esses any GL Render while a canvas is off screen (rendering an unmapped
        native window binds VTK's context to a bad HDC → ``wglMakeCurrent``
        "error 2004" flood → BLACK pane), so load_series only prepares the
        pipeline and camera. Now that we are shown, do the real first Render —
        deferred to the next event-loop turn so Qt has settled (mapped, DPI-
        resolved) the canvas geometry first, which is also when ``_fit_pane`` can
        read the true canvas size. The ``_view_initial`` guard keeps a user's own
        zoom/pan (repaint only); a fresh load refits to the real size."""
        super().showEvent(e)
        if self._image is not None:
            QTimer.singleShot(0, self._refit_on_show)

    def _refit_on_show(self) -> None:
        # Guard: the viewer may have been cleared/destroyed before this fires.
        if self._image is None:
            return
        self._refresh(reset_cam=self._view_initial)

    def finalize_gl(self) -> None:
        """Release each pane's VTK OpenGL context while our native windows are
        still valid — called from MainWindow.closeEvent on app quit. Without it,
        Qt destroys the HWNDs first and VTK's later teardown floods the terminal
        with 'wglMakeCurrent failed ... invalid handle (code 6)' during Clean().
        Windows/VTK only (duck-typed: pygfx has no such method)."""
        for key in ("A", "B"):
            try:
                self.pane[key].canvas.Finalize()
            except Exception:
                pass

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

    # --------------------------------------------- curved-MPR scrubber bar
    def _build_cpr_bar(self) -> QWidget:
        """A bottom scrubber for short-axis (CPR) mode: scroll the cross-
        section along the traced vessel, plus an Exit button. Hidden unless
        CPR is active. Survives 'Max Image' so scrolling stays usable."""
        self._cpr_wrap = QWidget()
        self._cpr_wrap._mdv_keep_on_max = True
        row = QHBoxLayout(self._cpr_wrap)
        row.setContentsMargins(8, 2, 8, 2)
        cap = QLabel(t("Short-axis:"))
        f = cap.font(); f.setBold(True); cap.setFont(f)
        self._cpr_cap = cap
        row.addWidget(cap)
        # Reverse the scroll direction (distal→proximal) to match an IVUS
        # pull-back; the cross-section content is unchanged.
        self._cpr_rev_btn = FitButton(t("Reverse"))
        self._cpr_rev_btn.setCheckable(True)
        self._cpr_rev_btn.setHelpToolTip(
            t("Reverse the scroll order to distal→proximal (match an IVUS "
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

        # 3-D MPR vs 2-D (native single-slice) display. Default is chosen per
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

        # Setting / Reset are utility actions (not tools): tint them a light
        # grey so they read as distinct from the tool / preset buttons.
        _util_btn_css = "background:#6e6e6e;color:#d8d8d8;"   # match Mac

        self._reset_btn = reset = FitButton("Reset")
        reset.setHelpToolTip(
            t("1st click: keep W/L, reset the view position / "
              "click again at the initial position: also reset W/L")
        )
        reset.setStyleSheet(_util_btn_css)
        reset.clicked.connect(self._reset)
        row.addWidget(reset)

        # ReCalc: same size/font as Reset, a slightly darker grey so it reads as
        # a sibling utility yet is easy to tell apart. Rebuilds the OTHER pane
        # from the selected pane's green-▲ centre line (un-mirror / fix a
        # companion that drifted after complex rotations) without a full Reset.
        self._recalc_btn = recalc = FitButton("ReCalc")
        recalc.setHelpToolTip(
            t("Re-derive the OTHER pane from the selected (active) pane's "
              "green-▲ centre line — fixes a mirrored / wrong companion after "
              "complex rotations, without resetting your view")
        )
        recalc.setStyleSheet("background:#8a8a8a;color:#101010;")   # match Mac
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
            t("Measure on the image (Line / Polyline / Ellipse / Polygon)")
        )
        self._meas_btn.clicked.connect(self._toggle_measure)
        row.addWidget(self._meas_btn)

        self._slab_lbl = QLabel(t("Slab(mm):"))
        row.addWidget(self._slab_lbl)
        self._slab_spin = QDoubleSpinBox()
        self._slab_spin.setRange(0.0, 50.0)
        self._slab_spin.setSingleStep(0.5)
        self._slab_spin.setDecimals(1)
        self._slab_spin.setToolTip(
            t("Slab-MIP thickness of the active pane (0 = thin MPR)")
        )
        self._slab_spin.valueChanged.connect(self._set_slab)
        row.addWidget(self._slab_spin)

        self._cl_btn = FitButton("CenterLine")
        self._cl_btn.setCheckable(True)
        self._cl_btn.setChecked(True)
        self._cl_btn.setHelpToolTip(t("Show/hide crosshair & slab lines"))
        self._cl_btn.clicked.connect(self._toggle_centerline)
        row.addWidget(self._cl_btn)
        self._style_cl()

        self._setting_btn = setting = FitButton(t("Setting"))
        setting.setHelpToolTip(
            t("HU colour-map settings (band colour, HU range, opacity)")
        )
        setting.setStyleSheet(_util_btn_css)
        setting.clicked.connect(self._open_setting)
        row.addWidget(setting)

        row.addWidget(QLabel("W/L:"))
        self._preset = QComboBox()
        self._preset.addItems(list(CT_WL_PRESETS.keys()))
        self._preset.currentTextChanged.connect(self._apply_preset)
        row.addWidget(self._preset)

        # DICOM Tags on the LEFT of the pair (always visible); Measure History
        # — less critical — to its right. Tag-text-size slider stacked above
        # (kept a 2-row control, matching the two-row toolbar height).
        tags_box, self._tag_font_slider, self._tags_font_btn = (
            build_tag_font_control(TAG_FONT_PT_DEFAULT)
        )
        tags = self._tags_font_btn
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
        # last entry is WL). Only meaningful for the single native slice in 2-D
        # mode, so they are disabled (greyed) in 3-D. "Mirror" == Flip-H (a
        # left-right mirror), so it is not a separate button. Kept on this
        # second row so they stay visible on a narrow pane (row 1 overflows).
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
        # Grayscale invert (black↔white negative) — right of Flip-V.
        self._invert_btn = FitButton(t("WB reverse"))
        self._invert_btn.setCheckable(True)
        self._invert_btn.setHelpToolTip(
            t("Invert grayscale (black↔white negative)"))
        self._invert_btn.clicked.connect(self._toggle_invert)
        row2.addWidget(self._invert_btn)
        row2.addStretch(1)

        col.addLayout(row)
        col.addLayout(row2)
        self._set_tool("PAGING")

        # Host the two rows in a horizontal scroll area and give every button a
        # SMALL minimum width (a floor, not its full natural width) so that on a
        # narrow / low-res monitor the labels first elide from the right —
        # keeping the START of each caption readable, full text in the tooltip
        # (FitButton) — and only fall back to scrolling when even that won't fit.
        bar = QWidget()
        bar.setLayout(col)
        for b in bar.findChildren(QPushButton):
            b.setMinimumWidth(min(b.sizeHint().width(), 56))
        scroll = QScrollArea()
        scroll.setWidget(bar)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFixedHeight(bar.sizeHint().height() + 14)
        return scroll

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
            for _ha in p.tagact_halo:           # outline tracks the tag size
                _ha.GetTextProperty().SetFontSize(vf)
            p.resultact.GetTextProperty().SetFontSize(vf)
            if self._header is not None:
                self._update_info(key, False)
                p.resultact.SetInput("\n".join(wrap_lines_to_chars(
                    self._metric_lines.get(key, []), self._wrap_budget(key))))
            p.render()

    def retranslate_ui(self) -> None:
        """Re-apply every persistent, user-facing string via ``t()`` so a live
        language switch (no restart) updates the CT toolbar / controls in place.

        Mirrors the toolbar / measure-bar / seek-bar construction. Safe to call
        whether or not a CT volume is loaded — every control is guarded with
        getattr (some are built lazily). Two-state toggle labels are re-derived
        from the CURRENT state (never flipped) by calling their own helper.
        On-demand dialogs (Angio Angle, ColorMap) and per-frame VTK annotations
        are NOT touched here — they are rebuilt / redrawn when next shown."""
        # ---- toolbar row 1: series nav / view / measure controls ----
        if getattr(self, "_series_nav_lbl", None) is not None:
            self._series_nav_lbl.setText(t("Series:"))
        nav_tips = (
            t("First series (Home)"),
            t("Previous series — shortcut: A"),
            t("Next series — shortcut: F"),
            t("Last series (End)"),
        )
        for b, tip in zip(getattr(self, "_nav_btns", []), nav_tips):
            b.setHelpToolTip(tip)
        # ---- plane bar (below the image): Plane Bi/Lt/Rt + 3D/2D ----
        if getattr(self, "_plane_lbl", None) is not None:
            self._plane_lbl.setText(t("Plane:"))
        side_tips = {
            "Bi": t("Show both MPR panes"),
            "Lt": t("Show only the left MPR pane"),
            "Rt": t("Show only the right MPR pane"),
        }
        for key, b in getattr(self, "_side_btns", {}).items():
            if key in side_tips:
                b.setHelpToolTip(side_tips[key])
        mode_tips = {
            "3D": t("3-D MPR reconstruction (dual oblique reslice)"),
            "2D": t("Show native acquisition slices one at a time (paging)"),
        }
        for key, b in getattr(self, "_mode_btns", {}).items():
            if key in mode_tips:
                b.setHelpToolTip(mode_tips[key])
        if getattr(self, "_reset_btn", None) is not None:
            self._reset_btn.setHelpToolTip(
                t("1st click: keep W/L, reset the view position / "
                  "click again at the initial position: also reset W/L"))
        if getattr(self, "_recalc_btn", None) is not None:
            self._recalc_btn.setHelpToolTip(
                t("Re-derive the OTHER pane from the selected (active) pane's "
                  "green-▲ centre line — fixes a mirrored / wrong companion "
                  "after complex rotations, without resetting your view"))
        if getattr(self, "_meas_btn", None) is not None:
            self._meas_btn.setHelpToolTip(
                t("Measure on the image (Line / Polyline / Ellipse / Polygon)"))
        if getattr(self, "_slab_lbl", None) is not None:
            self._slab_lbl.setText(t("Slab(mm):"))
        if getattr(self, "_slab_spin", None) is not None:
            self._slab_spin.setToolTip(
                t("Slab-MIP thickness of the active pane (0 = thin MPR)"))
        if getattr(self, "_cl_btn", None) is not None:
            self._cl_btn.setHelpToolTip(t("Show/hide crosshair & slab lines"))
        if getattr(self, "_setting_btn", None) is not None:
            self._setting_btn.setText(t("Setting"))
            self._setting_btn.setHelpToolTip(
                t("HU colour-map settings (band colour, HU range, opacity)"))
        if getattr(self, "_hist_btn", None) is not None:
            self._hist_btn.setText(t("Measure History"))
            self._hist_btn.setHelpToolTip(
                t("Show this study's measurement history"))
        # DICOM-tag overlay font control (hidden per-viewer copy, kept in sync).
        if getattr(self, "_tag_font_slider", None) is not None:
            self._tag_font_slider.setToolTip(t("DICOM tag text size"))
        if getattr(self, "_tags_font_btn", None) is not None:
            self._tags_font_btn.setText(t("DICOM Tags"))
            self._tags_font_btn.setToolTip(
                t("Choose DICOM tags to overlay on the image"))
        # ---- toolbar row 2: 2-D image transforms (rotate 90° / flip) ----
        t2d_tips = (
            t("Rotate the image 90° clockwise"),
            t("Rotate the image 90° counter-clockwise"),
            t("Flip horizontally (left-right mirror)"),
            t("Flip vertically (top-bottom)"),
        )
        for b, tip in zip(getattr(self, "_t2d_btns", []), t2d_tips):
            b.setHelpToolTip(tip)
        # ---- measure bar ----
        if getattr(self, "_measure_lbl", None) is not None:
            self._measure_lbl.setText(t("Measure:"))
        if getattr(self, "_cmp_btn", None) is not None:
            self._cmp_btn.setText(t("Compare"))
            self._cmp_btn.setHelpToolTip(
                t("Compare two Polygon/Ellipse: click the two shapes — shows "
                  "%Area difference and a radial gap colour map "
                  "(<5 / 5–7 / 7–9 / >9 mm)"))
        if getattr(self, "_hideall_btn", None) is not None:
            self._hideall_btn.setHelpToolTip(
                t("Hide / Show every measurement line, region colour and "
                  "result text"))
            # Re-derives the Hide / Show All Result label from current state.
            self._update_hideall_btn()
        if getattr(self, "_clr_btn", None) is not None:
            self._clr_btn.setText(t("Clear All Result"))
            self._clr_btn.setHelpToolTip(
                t("Clear all measurements and comparison results"))
        if getattr(self, "_measure_hint_lbl", None) is not None:
            self._measure_hint_lbl.setText(
                t("  Left-click = add point /"
                  " right-click finishes Polyline / Polygon"))
        # ---- seek bar (2-D native-slice scrubber) ----
        if getattr(self, "_seek_frame_lbl", None) is not None:
            self._seek_frame_lbl.setText(t("Frame:"))
        if getattr(self, "_seek_series_cap", None) is not None:
            self._seek_series_cap.setText(t("Series:"))
        if getattr(self, "_seek_series_lbl", None) is not None:
            self._seek_series_lbl.setToolTip(
                t("Series position in this study (current / total)"))
        # Repaint the Qt controls, then re-render the panes so on-image text
        # (DICOM tag overlay / measure results, drawn by VTK) also refreshes.
        # _refresh is a no-op while no volume is loaded, so this stays safe.
        self.update()
        for c in (getattr(self, "canvas_a", None),
                  getattr(self, "canvas_b", None)):
            if c is not None:
                c.update()
        self._refresh()

    def _set_tool(self, name):
        # MPR-only tools are unavailable in 2-D native-slice mode (their
        # keyboard shortcuts are otherwise still live).
        if getattr(self, "_mode", "3D") == "2D" and name in _MPR_ONLY_TOOLS:
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
        measuring = getattr(self, "_meas_on", False) and bool(
            getattr(self, "_meas_type", None))
        is2d = getattr(self, "_mode", "3D") == "2D"
        for n, b in self._tool_btns.items():
            active = (n == getattr(self, "_tool", None))
            # Keep the tools clickable WHILE measuring so the user can pick which
            # tool a Shift+drag uses (the shortcut keys already switch them); the
            # dimmed-red styling below still signals "hold Shift to use it here".
            b.setEnabled(not (is2d and n in _MPR_ONLY_TOOLS))
            if measuring:
                b.setStyleSheet("background:#7a4b46;color:#d0d0d0;" if active
                                else "color:#9a9a9a;")
            else:
                b.setStyleSheet("background:#c0392b;color:white;" if active
                                else "")

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
        # In CPR the crosshair / ▲ on pane A are drawn by _draw_cpr_overlay
        # (sized to the CPR zoom), not the normal per-render _update_cross — so
        # redraw them here when the CenterLine toggle flips on.
        if self._cpr is not None:
            self._draw_cpr_overlay()
            self.pane["A"].render()

    # ------------------------------------------------------- Measure
    def _build_measure_bar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 6, 2)
        self._measure_lbl = QLabel(t("Measure:"))
        row.addWidget(self._measure_lbl)
        self._meas_btns = {}
        for label, key in (
            ("Line", "line"), ("Polyline", "polyline"),
            ("Ellipse", "ellipse"), ("Polygon", "polygon"),
            ("Angle", "angle"),
        ):
            b = FitButton(label)
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
        self._cmp_btn.setHelpToolTip(
            t("Compare two Polygon/Ellipse: click the two shapes — shows %Area "
              "difference and a radial gap colour map (<5 / 5–7 / 7–9 / >9 mm)")
        )
        self._cmp_btn.clicked.connect(self._toggle_compare)
        row.addWidget(self._cmp_btn)
        # Hide/Show ALL results (lines + region colours + text) at once, between
        # Compare and Clear All Result. Disabled when there is nothing to hide.
        self._hideall_btn = FitButton(t("Hide All Result"))
        self._hideall_btn.setMinimumWidth(
            min(self._hideall_btn.sizeHint().width(), 64))
        self._hideall_btn.setHelpToolTip(
            t("Hide / Show every measurement line, region colour and result "
              "text"))
        self._hideall_btn.setStyleSheet("background:#bdbdbd;color:#101010;")
        self._hideall_btn.clicked.connect(self._toggle_hide_all)
        row.addWidget(self._hideall_btn)
        self._clr_btn = clr = FitButton(t("Clear All Result"))
        clr.setMinimumWidth(min(clr.sizeHint().width(), 56))
        clr.setHelpToolTip(t("Clear all measurements and comparison results"))
        clr.setStyleSheet("background:#6e6e6e;color:#d8d8d8;")   # Reset's grey
        clr.clicked.connect(self._measure_clear)
        row.addWidget(clr)
        self._measure_hint_lbl = QLabel(
            t("  Left-click = add point /"
              " right-click finishes Polyline / Polygon")
        )
        row.addWidget(self._measure_hint_lbl)
        row.addStretch(1)
        self._update_hideall_btn()
        return bar

    def _toggle_hide_all(self):
        """Hide / Show ALL results at once. Show reveals EVERYTHING, including
        individually-hidden results (clears their per-item Hide too)."""
        self._results_hidden = not self._results_hidden
        if not self._results_hidden:
            for k in ("A", "B"):
                for m in self._measures[k]:
                    m.pop("hidden", None)
            for c in self._compares:
                c.pop("hidden", None)
        for k in ("A", "B"):
            self._redraw_meas(k)
            self._redraw_compare(k)
        self._update_hideall_btn()

    def _update_hideall_btn(self):
        btn = getattr(self, "_hideall_btn", None)
        if btn is None:
            return
        has = (any(self._measures[k] for k in ("A", "B"))
               or bool(self._compares))
        btn.setEnabled(has)
        btn.setText(t("Show All Result") if self._results_hidden
                    else t("Hide All Result"))

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
        self._refresh_tool_availability()   # grey/restore the interaction tools

    def _set_measure_type(self, key):
        self._meas_type = key
        self._draft = None
        for k, b in self._meas_btns.items():
            b.setChecked(k == key)
            b.setStyleSheet(
                "background:#1f77b4;color:white;" if k == key else ""
            )
        # A type is now active → left-click measures → grey the tools.
        self._refresh_tool_availability()

    def _measure_clear(self):
        self._measures = {"A": [], "B": []}
        self._draft = None
        self._edit = None
        self._compares = []
        self._cmp_sel = []
        self._results_hidden = False
        for k in ("A", "B"):
            self._redraw_meas(k)
            self._redraw_compare(k)

    # ---- Compare: %Area + radial gap between two Polygon/Ellipse shapes ----
    def _toggle_compare(self):
        """Enter/leave Compare-select mode. While on, a left-click picks a
        Polygon/Ellipse (toggles); picking the 2nd computes and shows the
        %Area difference and the radial gap colour map."""
        self._cmp_on = self._cmp_btn.isChecked()
        self._cmp_sel = []
        for k in ("A", "B"):
            self._redraw_compare(k)

    def _compare_pick(self, which, sx, sy) -> bool:
        """Compare-mode left-click: toggle the Polygon/Ellipse under the cursor
        in the selection (one pane only). Returns True if Compare mode is on
        (so the canvas consumes the click)."""
        if not self._cmp_on:
            return False
        mi = self._pick_measure(which, sx, sy)
        if mi is not None and self._measures[which][mi]["type"] in (
                "polygon", "ellipse"):
            if self._cmp_sel and self._cmp_sel[0][0] != which:
                self._cmp_sel = []           # restart selection on a new pane
            item = (which, mi)
            if item in self._cmp_sel:
                self._cmp_sel.remove(item)
            elif len(self._cmp_sel) < 2:
                self._cmp_sel.append(item)
            if len(self._cmp_sel) == 2:
                # Defer the modal options dialog out of the mouse handler (so a
                # blocking exec() can't swallow the button-release).
                QTimer.singleShot(0, self._compare_prompt)
        self._redraw_compare(which)
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
            self._redraw_compare(k)

    def _do_compare(self):
        sel = self._cmp_sel
        if len(sel) != 2 or sel[0][0] != sel[1][0]:
            return
        key = sel[0][0]
        m1 = self._measures[key][sel[0][1]]
        m2 = self._measures[key][sel[1][1]]
        o1, o2 = self._outline(m1), self._outline(m2)
        a1, a2 = _poly_area(o1), _poly_area(o2)
        if a2 > a1:                          # the LARGER shape is the reference
            m1, m2, o1, o2, a1, a2 = m2, m1, o2, o1, a2, a1
        cen = _polygon_centroid(o1)
        # Radials are always computed (they also drive the filled-region geometry
        # and hit area); only DRAWN as a colour map when Thickness is wanted.
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
            self._redraw_compare(k)
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

    def _compare_hit(self, which, sx, sy):
        """Index in self._compares of the result whose filled region (inside the
        outer outline, outside the inner) contains screen point (sx,sy) on pane
        *which* — topmost first — else None."""
        wx, wy = self._disp_to_world(which, sx, sy)
        for i in range(len(self._compares) - 1, -1, -1):
            c = self._compares[i]
            if c["key"] != which:
                continue
            if (_point_in_poly(wx, wy, c["outer"])
                    and not _point_in_poly(wx, wy, c["inner"])):
                return i
        return None

    def _compare_delete_menu(self, target):
        """Right-click INSIDE a compare region → Hide·Show / Delete. Hide toggles
        only the region COLOUR fill; the defining outlines are unaffected."""
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
            self._redraw_compare(k)
        self._update_hideall_btn()

    def _redraw_compare(self, key):
        """Rebuild the compare overlay for a pane: cyan selection outlines, the
        radial gap colour map and the legend / %Area text actors."""
        p = self.pane[key]
        # selection highlight (cyan) while picking
        sel_lines, sel_cols = [], []
        if self._cmp_on:
            for (skey, smi) in self._cmp_sel:
                if skey == key and 0 <= smi < len(self._measures[key]):
                    sel_lines.append(self._outline(self._measures[key][smi]))
                    sel_cols.append((0, 229, 255))
        p.cmp_sel_mapper.SetInputData(_colored_multi_pd(sel_lines, sel_cols))
        # All persisted results on this pane, drawn as a translucent (65%) annulus
        # FILL: Thickness colours each angular sector by its gap band (heatmap);
        # %PA fills with the outer shape's single colour. No radial lines.
        # "Hide/Show All Result" (global) suppresses everything here.
        cmps = ([] if self._results_hidden
                else [c for c in self._compares if c["key"] == key])
        fill_tris, fill_cols = [], []
        for c in cmps:
            if c.get("hidden"):                      # Hidden → no fill
                continue
            rad = c["radials"]
            n = len(rad)
            thk = bool(c.get("show_thk"))
            alpha = transp_to_alpha(c.get("transp", 50))   # Change Transparency
            for i in range(n):                       # annulus fill triangles
                a, b = rad[i], rad[(i + 1) % n]
                da = abs(b["ang"] - a["ang"]) % 360.0
                if 2.5 * c["step"] < da < 360.0 - 2.5 * c["step"]:
                    continue                         # a skipped-ray gap
                # Thickness → this sector's gap-band colour; %PA → outer colour.
                rgb = _hex_to_rgb(_gap_color(a["gap"])) if thk else c["fill_rgb"]
                col = (rgb[0], rgb[1], rgb[2], alpha)
                fill_tris.append((a["inner"], a["outer"], b["outer"]))
                fill_tris.append((a["inner"], b["outer"], b["inner"]))
                fill_cols += [col, col]
        p.cmp_fill_mapper.SetInputData(_filled_tris_pd(fill_tris, fill_cols))
        p.cmp_mapper.SetInputData(_colored_multi_pd([], []))      # no radial lines
        p.cmp_red_mapper.SetInputData(_colored_multi_pd([], []))
        # text actors (hint + per-result summary + a single colour legend)
        for a in p.cmp_text:
            p.ren.RemoveActor(a)
        p.cmp_text = []

        def _add_cmp_text(txt, rgb, halo, fx, fy, size=14, centered=False):
            """A text actor + a thin 8-offset halo (枠) of colour *halo* (0–1
            triple) behind it, so it stays legible over the image."""
            for ox, oy in ((-1, -1), (0, -1), (1, -1), (-1, 0),
                           (1, 0), (-1, 1), (0, 1), (1, 1)):
                ha = vtkTextActor()
                ha.SetInput(txt)
                htp = ha.GetTextProperty()
                htp.SetColor(*halo)
                htp.SetFontSize(size)
                htp.SetBold(True)
                if centered:
                    htp.SetJustificationToCentered()
                ha.GetPositionCoordinate(
                    ).SetCoordinateSystemToNormalizedViewport()
                ha.SetPosition(fx + ox * 0.0016, fy + oy * 0.0016)
                p.ren.AddActor(ha)
                p.cmp_text.append(ha)
            ta = vtkTextActor()
            ta.SetInput(txt)
            tp = ta.GetTextProperty()
            tp.SetColor(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
            tp.SetFontSize(size)
            tp.SetBold(True)
            if centered:
                tp.SetJustificationToCentered()
            ta.GetPositionCoordinate(
                ).SetCoordinateSystemToNormalizedViewport()
            ta.SetPosition(fx, fy)
            p.ren.AddActor(ta)
            p.cmp_text.append(ta)

        # Instruction banner (top-centre) while picking — black 枠.
        if self._cmp_on:
            n_sel = sum(1 for s in self._cmp_sel if s[0] == key)
            _add_cmp_text(
                t("Click to select 2 Ellipse/Polygon data to compare"
                  "  ({n_sel}/2)", n_sel=n_sel), (0, 229, 255), (0.0, 0.0, 0.0),
                0.5, 0.94, size=15, centered=True)
        row_i = 0
        for c in cmps:                               # one summary line each (cyan)
            head = f"Compare #{c['big_id']} vs #{c['small_id']}"
            if c.get("show_pa"):
                head += f"  %Area:{c['pct']:.1f}%"
            _add_cmp_text(head, (0, 229, 255), (0.0, 0.0, 0.0),
                          0.02, 0.34 - row_i * 0.04)
            row_i += 1
        # Colour legend: a band-coloured swatch + a WHITE label (matches Mac).
        # The swatch is a small SQUARE 2-D quad sized in pixels (a fixed-size
        # square, aspect-independent — matching Mac's 12 px square), rather than
        # a text-background box whose width tracked the glyph (→ a rectangle).
        if any(c.get("show_thk") and not c.get("hidden") for c in cmps):
            sq = 13                                  # swatch side, pixels
            sq_dy = sq * 0.30                        # lift 30% of its height to
            #                                          align with the text row
            for lab, hexc in _gap_legend():
                rgb = _hex_to_rgb(hexc)
                fy = 0.34 - row_i * 0.04
                sq_pd = vtkPolyData()
                sq_pts = vtkPoints()
                # Points are DISPLAY-space pixel offsets from the actor's
                # normalized-viewport PositionCoordinate, so the box is a true
                # square regardless of the viewport's aspect ratio.
                for px, py in ((0, 0), (sq, 0), (sq, sq), (0, sq)):
                    sq_pts.InsertNextPoint(px, py + sq_dy, 0.0)
                sq_quad = vtkCellArray()
                sq_quad.InsertNextCell(4)
                for k in range(4):
                    sq_quad.InsertCellPoint(k)
                sq_pd.SetPoints(sq_pts)
                sq_pd.SetPolys(sq_quad)
                sq_map = vtkPolyDataMapper2D()
                sq_map.SetInputData(sq_pd)
                sw = vtkActor2D()
                sw.SetMapper(sq_map)
                sw.GetPositionCoordinate(
                    ).SetCoordinateSystemToNormalizedViewport()
                sw.GetPositionCoordinate().SetValue(0.02, fy)
                sw.GetProperty().SetColor(rgb[0] / 255.0, rgb[1] / 255.0,
                                          rgb[2] / 255.0)
                sw.GetProperty().SetOpacity(1.0)
                p.ren.AddActor(sw)
                p.cmp_text.append(sw)
                _add_cmp_text(lab, (255, 255, 255), (0.0, 0.0, 0.0), 0.05, fy)
                row_i += 1
        p.render()

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
        # Center-Angle apex = the physical (area) centroid of the region, not
        # the vertex mean (which skews toward clustered vertices).
        if m["type"] == "polygon":
            return _polygon_centroid(m["pts"])
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
        # A vessel-trace polyline stores absolute 3-D control points; re-derive
        # its 2-D outline from them on the CURRENT plane so the trace follows
        # the anatomy when the plane is rotated / paged (a plain 2-D measure has
        # no pts3d and is untouched). 3-D MPR only.
        if self._mode == "3D":
            for m in self._measures[key]:
                p3 = m.get("pts3d")
                if p3 and m["type"] == "polyline" and len(p3) == len(m["pts"]):
                    m["pts"] = [self._world3d_to_out(key, P) for P in p3]
            d = self._draft
            if (d is not None and d.get("pane") == key and d.get("pts3d")
                    and len(d["pts3d"]) == len(d["pts"])):
                d["pts"] = [self._world3d_to_out(key, P) for P in d["pts3d"]]
        p = self.pane[key]
        # Scale the measurement line widths by the render-window DPR so they
        # render at the intended on-screen thickness on scaled displays.
        dpr = max(1.0, p.canvas.devicePixelRatioF())
        for base_w, act in p._meas_line_actors:
            act.GetProperty().SetLineWidth(base_w * dpr)
        polylines, handles, labels = [], [], []
        outline_colors: list[tuple[int, int, int]] = []
        axis_segs: list = []
        axis_colors: list[tuple[int, int, int]] = []
        ca_segs: list = []           # center-angle spokes
        ca_colors: list[tuple[int, int, int]] = []
        ca_pts: list = []            # picked perimeter dots
        arc_lines: list = []         # center-angle arc (solid, own actor)
        # The one vertex currently being dragged lives in *edit_pts*
        # (drawn green by a second points mapper); the rest stay yellow.
        edit_pts = []
        off_pts = []                 # trace vertices off the current plane (50%)
        _, _, _pn = self._axes_for(key)          # this plane's normal
        _po = self._pc[key]
        e = self._edit
        edit_mi = e["mi"] if (e and e["key"] == key) else -1
        edit_vi = e["vi"] if (e and e["key"] == key) else -1
        edit_ca = bool(e.get("ca")) if (e and e["key"] == key) else False
        for mi, m in enumerate(self._measures[key]):
            # Hidden by "Hide/Show All Result" (global) or this measure's own
            # right-click Hide → skip its line, handles, axes and id label.
            if self._results_hidden or m.get("hidden"):
                continue
            rgb = _hex_to_rgb(m.get("color"))
            a4 = transp_to_alpha(m.get("transp", 0))   # Change Transparency
            # Off-plane depth cue for a 3-D trace: a vertex whose 3-D point is
            # > 1 mm off this plane (along its normal) is "off-plane".
            p3 = m.get("pts3d") if m["type"] == "polyline" else None
            off_flag = None
            if p3 is not None and len(p3) == len(m["pts"]):
                off_flag = [abs(float(np.dot(np.asarray(P, float) - _po,
                                             _pn))) > 1.0 for P in p3]
            if off_flag is not None and not m.get("smooth"):
                # Draw the outline as PER-SEGMENT cells so the off-plane parts
                # of the LINE (not just the point dots) fade to 50%. A trace is
                # normally un-splined; a smooth trace keeps one line alpha.
                verts = list(m["pts"])
                half_a = max(1, a4 // 2)
                for i in range(len(verts) - 1):
                    polylines.append([verts[i], verts[i + 1]])
                    seg_a = half_a if (off_flag[i] or off_flag[i + 1]) else a4
                    outline_colors.append((rgb[0], rgb[1], rgb[2], seg_a))
            else:
                polylines.append(self._outline(m))
                outline_colors.append((rgb[0], rgb[1], rgb[2], a4))
            # Solid orange arc on the outline between the two endpoints, passing
            # through the selector — only shown once all 3 points are placed
            # (drawn over the outline via the same solid-line mapper).
            ca0 = m.get("center_angle")
            if ca0 and "angle" in ca0 and len(ca0.get("pts", [])) >= 3:
                arc = _arc_through(self._outline(m), ca0["pts"][0],
                                   ca0["pts"][2], ca0["pts"][1])
                if len(arc) >= 2:
                    arc_lines.append(arc)
            # Off-plane point dots: a trace vertex > 1 mm off this plane is
            # drawn faint (50%) via the separate off-plane points actor.
            for vi, q in enumerate(self._handles(m)):
                if mi == edit_mi and not edit_ca and vi == edit_vi:
                    edit_pts.append(q)
                elif (off_flag is not None and vi < len(off_flag)
                      and off_flag[vi]):
                    off_pts.append(q)
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
                    # pts == [endpoint, other endpoint, arc selector]. The 3rd
                    # point only picks which arc is measured, so it gets no
                    # spoke; spokes (orange = marker colour) go to the two
                    # endpoints (ci 0 and 1) — the angle's two arms.
                    if ci != 2:
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
        p.meas_arc_mapper.SetInputData(
            _colored_multi_pd(arc_lines, [(255, 140, 0)] * len(arc_lines))
        )
        # CPR: on the MAP pane, mark where the short-axis is currently cut
        # (the scrubbed centreline point projected onto this plane) — the CT
        # analogue of the IVUS pull-back position marker. Drawn green via the
        # edit-points actor (no vertex is being edited on the map pane in CPR).
        if self._cpr is not None and key == self._cpr.get("src"):
            i = self._cpr["idx"]
            edit_pts.append(
                self._world3d_to_out(key, self._cpr["cl"].points[i]))
        p.meas_ca_pts_mapper.SetInputData(_points_pd(ca_pts))
        p.meas_pts_mapper.SetInputData(_points_pd(handles))
        p.meas_pts_edit_mapper.SetInputData(_points_pd(edit_pts))
        p.meas_pts_off_mapper.SetInputData(_points_pd(off_pts))
        self._rebuild_labels(p, labels)
        p.render()

    def _redraw_meas(self, key):
        self._recompute_compares(key)      # keep comparisons in sync on edit/delete
        self._redraw_geom(key)
        self._redraw_compare(key)
        p = self.pane[key]
        # "Hide/Show All Result" hides the measure result text too.
        lines = ([] if self._results_hidden
                 else [self._metrics_text(key, m) for m in self._measures[key]])
        self._metric_lines[key] = lines        # keep unwrapped for re-wrapping
        # Confine the result block to ~40% width (right) by word-wrapping it to
        # the fixed-size result actor (which honours the exact slider font).
        p.resultact.SetInput("\n".join(
            wrap_lines_to_chars(lines, self._wrap_budget(key))))
        p.render()
        self._update_hideall_btn()

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
        # Pixel-based catch (constant on-screen width, zoom/DPR independent), so
        # the boundary band can't balloon when zoomed in and shadow the filled
        # compare region — a click ≥5 px inside the annulus now selects the fill.
        tol = 5.0                              # screen px, each side of the line
        best, bi = tol, None
        for mi, m in enumerate(self._measures[which]):
            wpts = [self._world_to_qt(which, q[0], q[1])
                    for q in self._outline(m)]
            for i in range(len(wpts) - 1):
                d = _seg_dist(sx, sy, wpts[i], wpts[i + 1])
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
        # Capture the absolute 3-D position of each polyline vertex (a vessel
        # trace may cross rotated / paged planes). Only polylines in 3-D MPR —
        # the plane doesn't move in 2-D native mode. Optionally snap the depth
        # to the contrast lumen along the plane normal (the MIP hid the depth).
        if d["type"] == "polyline" and self._mode == "3D":
            P = self._out_to_world3d(which, *w)
            if self._snap_lumen:
                _, _, nrm = self._axes_for(which)
                P = self._snap_to_lumen(P, nrm)
                d["pts"][-1] = self._world3d_to_out(which, P)   # keep 2-D in step
            d.setdefault("pts3d", []).append(P)
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
            # Keep the absolute 3-D trace in step: the dragged vertex moves on
            # the CURRENT plane, so its new 3-D is that 2-D point lifted here
            # (then snapped to the lumen along the normal, if enabled).
            if m.get("pts3d") and 0 <= e["vi"] < len(m["pts3d"]):
                P = self._out_to_world3d(e["key"], *w)
                if self._snap_lumen:
                    _, _, nrm = self._axes_for(e["key"])
                    P = self._snap_to_lumen(P, nrm)
                    m["pts"][e["vi"]] = self._world3d_to_out(e["key"], P)
                m["pts3d"][e["vi"]] = P
            self._resnap_center_angle(m)
        self._recompute_compares(e["key"])     # live-update any comparison
        self._redraw_geom(e["key"])
        self._redraw_compare(e["key"])

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
        rec = {"id": self._meas_seq, "type": d["type"], "pts": pts}
        # Carry the absolute 3-D trace (vessel centreline control points) so a
        # later Short-axis MPR uses the real geometry regardless of how the
        # plane was moved while tracing.
        if d["type"] == "polyline" and d.get("pts3d") \
                and len(d["pts3d"]) == len(pts):
            rec["pts3d"] = list(d["pts3d"])
        self._measures[d["pane"]].append(rec)
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

    def _measure_right(self, which, sx, sy) -> bool:
        """Right-click on a measure (handle / outline / Center-Angle) → its menu.
        Returns True if a measure was hit/handled, False if nothing was (so the
        caller can fall through to the compare region)."""
        # Cancel an in-progress Center-Angle pick on right-click.
        cat = getattr(self, "_center_angle_target", None)
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
            del_ca = menu.addAction(t("Delete Center Angle"))
            chosen = menu.exec(
                self.pane[which].canvas.mapToGlobal(QtPoint(int(sx), int(sy))))
            if chosen is del_ca:
                self._measures[which][ca_mi].pop("center_angle", None)
                self._redraw_meas(which)
            return True
        # Handle is more specific than outline — try it first.
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

    def _export_pane(self, which, sx, sy):
        """Right-click export on a CT pane (no active measure tool): capture
        the pane's VTK render window verbatim — slice, measurements,
        crosshair and tag/result text are all VTK actors, so one grab gets
        the whole WYSIWYG view — and save in the chosen format."""
        if self._header is None:        # nothing loaded → no export offered
            return
        canvas = self.pane[which].canvas
        # CT offers DICOM + Anon DICOM + CSV but NOT MP4 (a slice scroll
        # isn't a cine).
        key = pick_export_format(
            self, canvas.mapToGlobal(QtPoint(int(sx), int(sy))),
            include_dicom=True, include_mp4=False, include_anon=True,
        )
        if not key:
            return
        if key in ("dicom", "csv", "anon-dicom"):
            # One volume — A/B panes are reformats of the same series, so
            # always the whole series (plane_path="").
            self.plane_export_requested.emit(
                key, getattr(self, "_loaded_uid", ""), ""
            )
            return
        img = self._grab_pane_qimage(which)
        if img is not None:
            export_image_as(self, img, key, self._export_basename(which))

    def _grab_pane_qimage(self, which):
        """VTK render window of pane *which* → RGB QImage, or None."""
        from multi_dicomviewer.viewers.image_canvas import to_qimage
        rw = self.pane[which].canvas.GetRenderWindow()
        rw.Render()
        w2i = vtkWindowToImageFilter()
        w2i.SetInput(rw)
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()        # read the buffer we just rendered
        w2i.Update()
        vimg = w2i.GetOutput()
        cols, rows, _ = vimg.GetDimensions()
        scalars = vimg.GetPointData().GetScalars()
        if scalars is None or cols == 0 or rows == 0:
            return None
        arr = vtk_to_numpy(scalars).reshape(rows, cols, -1)
        # VTK's image origin is bottom-left; flip to top-left for QImage.
        arr = np.ascontiguousarray(arr[::-1, :, :3])
        return to_qimage(arr)

    def _export_basename(self, which="") -> str:
        """Suggested filename stem from the loaded CT series + pane."""
        h = self._header
        parts: list[object] = []
        if h is not None:
            parts.append(getattr(h, "PatientID", "") or "")
            parts.append(getattr(h, "Modality", "") or "")
            parts.append(getattr(h, "StudyDate", "")
                         or getattr(h, "AcquisitionDate", "") or "")
        if which:
            parts.append(f"pane{which}")
        return safe_basename(*parts)

    def _handle_right(self, which, hit, sx, sy):
        """Right-click on a measure handle: 'Delete point' + 'Delete
        result' for Polyline/Polygon, just 'Delete' for Line/Ellipse."""
        from multi_dicomviewer.viewers.image_canvas import COLOR_CHOICES
        mi, vi = hit
        m = self._measures[which][mi]
        menu = QMenu(self)
        del_pt = del_res = None
        if m["type"] in ("polyline", "polygon"):
            del_pt = menu.addAction(t("Delete point"))
            if len(m["pts"]) <= 2:                # never shrink below Line
                del_pt.setEnabled(False)
        # Change Color / Change Transparency — on every result type (incl.
        # Line/Angle, most easily right-clicked on a handle).
        color_actions = add_color_submenu(menu, COLOR_CHOICES)
        transp_actions = add_transparency_submenu(menu, m.get("transp", 0))
        hide_act = menu.addAction(t("Show") if m.get("hidden") else t("Hide"))
        if m["type"] in ("polyline", "polygon"):
            del_res = menu.addAction(t("Delete result"))
        else:
            del_res = menu.addAction(t("Delete"))
        chosen = menu.exec(
            self.pane[which].canvas.mapToGlobal(
                QtPoint(int(sx), int(sy))
            )
        )
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
        self._redraw_meas(which)              # recomputes + redraws compares too

    def _outline_right(self, which, mi, sx, sy):
        """Right-click on a measure outline: Add point / Spline (Polyline)
        / Center Angle (Ellipse/Polygon) / Change Color / Delete."""
        from multi_dicomviewer.viewers.image_canvas import COLOR_CHOICES
        from PyQt6.QtGui import QIcon, QPixmap, QColor as _QColor
        m = self._measures[which][mi]
        menu = QMenu(self)
        add_pt = menu.addAction(t("Add point"))
        spline_act = None
        cpr_act = None
        if m["type"] == "polyline":
            spline_act = menu.addAction(
                t("UnSpline") if m.get("smooth") else t("Spline")
            )
            # Curved-MPR: build a centreline from this trace and scroll the
            # short-axis (perpendicular) cross-sections in the other pane —
            # only meaningful in 3-D MPR (the default slab view the user
            # traces on) and with ≥2 points.
            if self._mode == "3D" and len(m["pts"]) >= 2:
                cpr_act = menu.addAction(t("Short-axis MPR (CPR)"))
        # Lumen-snap controls (3-D traces only): re-snap this trace now, and a
        # checkable auto-snap toggle for future clicks.
        snap_now_act = snap_auto_act = None
        if m["type"] == "polyline" and self._mode == "3D" \
                and m.get("pts3d"):
            snap_now_act = menu.addAction(t("Snap trace to lumen"))
            snap_auto_act = menu.addAction(t("Auto-snap to lumen"))
            snap_auto_act.setCheckable(True)
            snap_auto_act.setChecked(self._snap_lumen)
        center_angle_act = None
        if m["type"] in ("ellipse", "polygon"):
            center_angle_act = menu.addAction(t("Center Angle"))
        color_menu = menu.addMenu(t("Change Color"))
        color_actions: list[tuple] = []
        for name, hexcol in COLOR_CHOICES:
            a = color_menu.addAction(name)
            pix = QPixmap(16, 16); pix.fill(_QColor(hexcol))
            a.setIcon(QIcon(pix))
            color_actions.append((a, hexcol))
        transp_actions = add_transparency_submenu(menu, m.get("transp", 0))
        hide_act = menu.addAction(t("Show") if m.get("hidden") else t("Hide"))
        del_act = menu.addAction(t("Delete"))
        chosen = menu.exec(
            self.pane[which].canvas.mapToGlobal(
                QtPoint(int(sx), int(sy))
            )
        )
        if chosen is add_pt:
            self._add_point(which, mi, sx, sy)
        elif cpr_act is not None and chosen is cpr_act:
            self._enter_cpr(which, mi)
            return                               # _enter_cpr redraws itself
        elif snap_now_act is not None and chosen is snap_now_act:
            self._snap_trace(which, mi)
        elif snap_auto_act is not None and chosen is snap_auto_act:
            self._snap_lumen = snap_auto_act.isChecked()
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
            # Click order is [endpoint, other endpoint, arc selector] — matching
            # Rupture-Predictor. The geometry helper wants (p1, through, p3), so
            # the selector (pts[2]) is passed as the middle "through" point.
            e1, e2, sel = ca["pts"][:3]
            span, t1, t3, ccw = _central_arc_angle(centre, e1, sel, e2)
            m["center_angle"] = {
                "pts": [e1, e2, sel], "angle": span,
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
        # Insert the matching 3-D control point (the click lifted on this plane)
        # so the vessel trace keeps pts3d aligned with pts.
        if m.get("pts3d") and len(m["pts3d"]) == len(pts) - 1:
            m["pts3d"].insert(best_i + 1, self._out_to_world3d(which, wx, wy))
        self._resnap_center_angle(m)

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

    def _delete_point(self, which, mi, vi):
        m = self._measures[which][mi]
        if m["type"] not in ("polyline", "polygon"):
            return
        pts = list(m["pts"])
        if len(pts) <= 2 or not (0 <= vi < len(pts)):
            return
        del pts[vi]
        p3 = m.get("pts3d")
        if p3 and 0 <= vi < len(p3):
            del p3[vi]
        if len(pts) == 2:
            m["type"] = "line"
            m.pop("pts3d", None)               # a Line is a plain 2-D measure
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
        self._vol = np.asarray(vol)                      # (z,y,x) for lumen snap
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
        # Reset any 2-D rotate/flip to the native orientation for the new series.
        self._axes2d = (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]))
        self._play2d_btn.setChecked(False)       # stop any running auto-page
        self._play2d_resume = False              # next Play starts at Frame 1
        self._set_play2d_speed(1.0)              # back to 1× for a new series
        self._cpr = None                         # drop any short-axis session
        self._cpr_wrap.setVisible(False)

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
        # Default 3-D MPR for thin-slice volumes (≥201 slices), 2-D native
        # paging for ordinary (≤200-slice) series. _set_mode also fits & draws.
        nz = self._image.GetDimensions()[2]
        self._set_mode("3D" if nz >= _MODE_2D_MAX + 1 else "2D", reset_cam=True)

    def clear(self) -> None:
        self._play2d_btn.setChecked(False)       # stop any running auto-page
        self._play2d_resume = False              # next Play starts at Frame 1
        self._cpr = None                         # drop any short-axis session
        self._cpr_wrap.setVisible(False)
        self._vol = None
        self._image = None
        self._header = None
        for key in ("A", "B"):
            self.pane[key].reslice.SetInputData(_placeholder_image())
            self.pane[key].info.SetText(0, "")
            self.pane[key].info.SetText(1, "")
            self.pane[key].tagact.SetInput("")
            for _ha in self.pane[key].tagact_halo:
                _ha.SetInput("")
            self.pane[key].resultact.SetInput("")
            self.pane[key].angle.SetInput("")
            for _ha in self.pane[key].angle_halo:
                _ha.SetInput("")
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

    # ---- 2-D output point  <->  absolute 3-D volume point ----
    # A measure point is stored as (wx, wy) in a pane's reslice OUTPUT frame,
    # which only makes sense while that pane keeps one cutting plane. For a
    # vessel trace the plane is rotated / paged between clicks, so each point
    # is ALSO captured as an absolute 3-D volume coordinate (pts3d) using the
    # frame live at that click; the 2-D pts are re-derived from pts3d on every
    # redraw so the trace stays anchored to the anatomy as the plane moves.
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
        win the brightest-peak search."""
        if self._vol is None:
            return None
        sx, sy, sz = self._dims
        pts = np.asarray(P, float)[None, :] + np.asarray(ds, float)[:, None] * n
        # world mm -> fractional voxel index (origin 0, vol is (z,y,x))
        fx = pts[:, 0] / sx
        fy = pts[:, 1] / sy
        fz = pts[:, 2] / sz
        nz, ny, nx = self._vol.shape
        inb = ((fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
               & (fz >= 0) & (fz <= nz - 1))
        out = np.full(len(ds), -2000.0)
        if not inb.any():
            return out
        x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
        z0 = np.clip(np.floor(fz).astype(int), 0, nz - 2)
        tx, ty, tz = fx - x0, fy - y0, fz - z0
        V = self._vol
        val = np.zeros(len(ds))
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = ((tx if dx else 1 - tx) * (ty if dy else 1 - ty)
                         * (tz if dz else 1 - tz))
                    val += w * V[z0 + dz, y0 + dy, x0 + dx]
        out[inb] = val[inb]
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
        # contiguous bright runs; choose the one nearest d=0 (the click)
        best = None
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

    def _cc(self, key):
        """Crosshair centre for a pane = C projected into its plane,
        relative to that pane's reslice centre (output coords, mm)."""
        u, v, _n = self._axes_for(key)
        delta = self._center - self._pc[key]
        return float(np.dot(delta, u)), float(np.dot(delta, v))

    def _toggle_invert(self):
        """WB reverse: invert the grayscale (black↔white). Applies to every
        pane, including the short-axis (all use the gray LUT)."""
        self._invert = self._invert_btn.isChecked()
        for k in ("A", "B"):
            self.pane[k].colors.SetLookupTable(self._lut())
            self.pane[k].colors.Modified()
        self._refresh()

    def _lut(self):
        if self._color:
            return _band_lut(
                self._bands, self._opacity, self._win, self._lvl
            )
        return _gray_lut(self._win, self._lvl, invert=self._invert)

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

    # ------------------------------------------------ curved-MPR / short-axis
    def _enter_cpr(self, which, mi):
        """Turn the polyline *mi* on pane *which* into a vessel centreline and
        put pane A into short-axis (cross-section) scroll mode.

        The trace is a 2-D outline in pane *which*'s reslice plane; lift each
        point to 3-D volume mm (origin + x·u + y·v) and hand it to CenterLine.
        Pane A then reslices the plane ⟂ the vessel tangent at the scrubbed
        arc-length index; pane *which* keeps its slab MPR so the trace stays
        visible as a map."""
        m = self._measures[which][mi]
        u, v, nrm = self._axes_for(which)
        p3 = m.get("pts3d")
        if p3 and len(p3) >= 2:
            # Absolute 3-D control points captured while tracing (correct even
            # if the plane was rotated / paged between clicks).
            ctrl = [np.asarray(P, dtype=float) for P in p3]
        else:
            # Fallback: a flat trace on this one plane (older / 2-D-lifted).
            pts2d = self._outline(m)              # honours the Spline toggle
            if len(pts2d) < 2:
                return
            o = self._pc[which]
            ctrl = [o + float(x) * u + float(y) * v for (x, y) in pts2d]
        step = max(1e-3, min(self._dims))
        cl = CenterLine.from_points(ctrl, step_mm=step)
        if cl.n < 2:
            return
        fu, fv = cl.frames(ref_up=nrm)            # short-axis axes ⟂ tangent
        # Look at each cross-section FROM the first control point TOWARD the
        # last (proximal→distal — the usual tracing order). The RMF gives
        # u×v = +tangent (an upstream view = mirror image); flip u so
        # u×v = −tangent, i.e. the viewer sits on the proximal side and the
        # short-axis is non-mirrored.
        fu = -fu
        self._cpr = {
            "cl": cl, "u": fu, "v": fv, "idx": cl.n // 2,
            "u0": fu.copy(), "v0": fv.copy(),     # base frame (pre-transform)
            "T": np.eye(2),                       # Rt90/Flip display transform
            "rot": 0.0,                           # continuous in-plane rotation°
            "reversed": False,                    # scroll distal→proximal (IVUS)
            "half": 25.0,                         # ±25 mm cross-section FOV
            "src": which,                         # pane the trace lives on
            "src_mi": mi,                         # index of the trace measure
            "ref_up": np.asarray(nrm, float),     # RMF seed (kept for rebuilds)
        }
        # Show both panes: A = cross-section, the traced pane = map.
        self.set_side("Bi")
        self.pane["A"].set_overlay_visible(False)   # no crosshair on the disc
        for b in self._t2d_btns:                    # Rt90/Lt90/Flip work on CPR
            b.setEnabled(True)
        self._cpr_rev_btn.setChecked(False)         # fresh: proximal→distal
        self._cpr_sync_bar()
        self._refresh(reset_cam=True)

    def _exit_cpr(self):
        """Leave short-axis mode and restore pane A's normal MPR."""
        if self._cpr is None:
            return
        self._cpr = None
        self._cpr_wrap.setVisible(False)
        self.pane["A"].set_overlay_visible(self._cl_btn.isChecked())
        for b in self._t2d_btns:                  # 2-D-only again once out of CPR
            b.setEnabled(self._mode == "2D")
        self._init_frames()                       # rebuild pane A's MPR frame
        self._refresh(reset_cam=True)

    # ---- CoSync interface + scrub/rotate/reverse/paging/rebuild:
    #      shared, in CPRMixin (viewers/cpr_mixin.py). ----

    def _cpr_matrix(self):
        """Reslice matrix for pane A in short-axis mode: axes (u, v, tangent)
        at the current sample, origin = that centreline point (volume mm)."""
        c = self._cpr
        i = c["idx"]
        u = c["u"][i]; v = c["v"][i]; nrm = c["cl"].tangents[i]
        o = c["cl"].points[i]
        mtx = vtkMatrix4x4()
        for r in range(3):
            mtx.SetElement(r, 0, float(u[r]))
            mtx.SetElement(r, 1, float(v[r]))
            mtx.SetElement(r, 2, float(nrm[r]))
            mtx.SetElement(r, 3, float(o[r]))
        return mtx

    def _cpr_sync_bar(self):
        c = self._cpr
        if c is None:
            return
        cl = c["cl"]
        d = self._cpr_disp(c["idx"])              # scrubber = display position
        self._cpr_wrap.setVisible(True)
        self._cpr_slider.blockSignals(True)
        self._cpr_slider.setMaximum(cl.n - 1)
        self._cpr_slider.setValue(d)
        self._cpr_slider.blockSignals(False)
        pos_mm = float(cl.arclen[c["idx"]])
        self._cpr_lbl.setText(
            f"{d + 1} / {cl.n}   ({pos_mm:.1f} / {cl.length_mm:.1f} mm)"
        )

    def _refresh(self, reset_cam=False):
        if self._image is None:
            return
        step = max(1e-3, min(self._dims))
        half, n = self._half, self._npx
        for key in ("A", "B"):
            p = self.pane[key]
            # Pane A in short-axis (CPR) mode: reslice the cross-section plane
            # with a tight FOV instead of the normal MPR matrix.
            if self._cpr is not None and key == "A":
                hcpr = self._cpr["half"]
                ncpr = max(64, int(2 * hcpr / step) + 1)
                p.reslice.SetResliceAxes(self._cpr_matrix())
                p.reslice.SetOutputSpacing(step, step, step)
                p.reslice.SetOutputOrigin(-hcpr, -hcpr, 0.0)
                p.reslice.SetOutputExtent(0, ncpr - 1, 0, ncpr - 1, 0, 0)
                p.reslice.SetSlabNumberOfSlices(1)
                p.reslice.Modified()
                p.colors.SetLookupTable(self._lut())
                p.colors.Modified()
                if reset_cam:
                    self._fit_cpr_pane()
                # No crosshair / C-arm angle on a cross-section; label the
                # pane and its arc-length position instead.
                c = self._cpr
                p.info.SetText(0, f"WW {self._win:.0f}  WL {self._lvl:.0f}")
                p.info.SetText(
                    1, "A  |  " + t("Short-axis {i}/{n}",
                                    i=c["idx"] + 1, n=c["cl"].n))
                p.angle.SetInput("")
                for _ha in p.angle_halo:
                    _ha.SetInput("")
                # DICOM-tag overlay honours the current keywords / anon state
                # here too (the normal _update_info is skipped for the CPR pane,
                # so Q / DICOM-info would otherwise leave stale tags on screen).
                _head = wrap_lines_to_chars(
                    overlay_lines(self._header, self._tag_keywords,
                                  anonymized=self._anon),
                    self._wrap_budget("A"))
                _tt = "\n".join(_head)
                p.tagact.SetInput(_tt)
                for _ha in p.tagact_halo:
                    _ha.SetInput(_tt)
                # CenterLine overlay in the cross-section: a centred crosshair
                # and a ▲ sized to the CPR zoom (the normal _update_cross is
                # skipped here, which is why the ▲ used to keep its huge pre-CPR
                # world size). Also draws the editable control-point markers.
                self._draw_cpr_overlay()
                p.render()
                continue
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
        # Anatomy-anchored traces (pts3d) re-project on ANY view change —
        # rotate / spin / page / move / recentre — not only on measure edits,
        # so the pseudo-centre points AND their lines follow the image even
        # after Measure is turned off. Cheap guard: only runs when such a trace
        # exists (a plain measure has no pts3d).
        if self._mode == "3D" and any(
                m.get("pts3d") for kk in ("A", "B")
                for m in self._measures[kk]):
            for kk in ("A", "B"):
                # In CPR, pane A is the cross-section (its overlay is the
                # control-point markers, drawn separately) — don't overwrite it.
                if self._cpr is not None and kk == "A":
                    continue
                self._redraw_geom(kk)

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

    def _draw_cpr_overlay(self):
        """Short-axis (CPR) pane overlay: a CENTRED crosshair and a ▲ sized to
        the CPR zoom (the normal per-render _update_cross is skipped for the
        CPR pane, so without this the ▲ kept its huge pre-CPR world size).
        Drawn only while CenterLine is on; also refreshes the editable
        control-point markers."""
        p = self.pane["A"]
        if p.cross and p.cross[0][1].GetVisibility():    # CenterLine on
            h = self._half
            zc = 0.5
            for (src, _a), (p0, p1) in zip(
                    p.cross,
                    (((-h, 0.0), (h, 0.0)), ((0.0, -h), (0.0, h)))):
                src.SetPoint1(p0[0], p0[1], zc)
                src.SetPoint2(p1[0], p1[1], zc)
                src.Modified()
            ps = p.ren.GetActiveCamera().GetParallelScale()
            sz = 0.024 * ps                              # same on-screen size…
            d = 0.255 * ps                               # …as the normal panes
            z = zc + 0.1
            p.tri_mapper.SetInputData(_tris_pd([
                ((a, -sz, z), (a - 0.6 * sz, 0.0, z), (a + 0.6 * sz, 0.0, z))
                for a in (d, -d)
            ]))
            for mp in p.slab_mappers:                    # no slab in a section
                mp.SetInputData(vtkPolyData())
        self._draw_cpr_ctrl_markers()

    def _draw_cpr_ctrl_markers(self):
        """Show the control points near the current cross-section as draggable
        dots at their in-plane offset from the centreline, so the user can see
        and fine-tune how each pseudo-centre sits versus the lumen."""
        p = self.pane["A"]
        p3 = self._cpr_ctrl_pts3d()
        pts = []
        self._cpr_marker_pts = []
        if p3:
            o, u, vv, n = self._cpr_frame()
            # Show exactly ONE dot: the control point nearest this cross-section
            # (along the vessel), at its in-plane offset. Always visible (never a
            # gap) and unambiguous — scroll to a pseudo-centre, then drag it.
            dns = [abs(float(np.dot(np.asarray(P, float) - o, n))) for P in p3]
            near = int(np.argmin(dns))
            P = np.asarray(p3[near], float)
            du = float(np.dot(P - o, u))
            dv = float(np.dot(P - o, vv))
            pts.append((du, dv))
            self._cpr_marker_pts.append((near, (du, dv)))
        p.meas_pts_mapper.SetInputData(_points_pd(pts))
        p.meas_pts_off_mapper.SetInputData(vtkPolyData())
        p.meas_pts_edit_mapper.SetInputData(vtkPolyData())

    def _cpr_grab(self, sx, sy) -> bool:
        """A press on the CPR pane near a control-point marker starts a drag."""
        if self._cpr is None or not self._cpr_marker_pts:
            return False
        for ci, (du, dv) in self._cpr_marker_pts:
            mx, my = self._world_to_qt("A", du, dv)
            if (mx - sx) ** 2 + (my - sy) ** 2 <= 14.0 ** 2:
                self._cpr_drag = ci
                return True
        return False

    def _cpr_drag_move(self, sx, sy):
        """Move the grabbed control point IN the cross-section plane (its along-
        vessel position is preserved); the trace on the map pane follows live."""
        if self._cpr_drag is None:
            return
        p3 = self._cpr_ctrl_pts3d()
        ci = self._cpr_drag
        if not p3 or not (0 <= ci < len(p3)):
            return
        o, u, vv, n = self._cpr_frame()
        du, dv = self._disp_to_world("A", sx, sy)
        dn = float(np.dot(np.asarray(p3[ci], float) - o, n))   # keep depth
        p3[ci] = o + du * u + dv * vv + dn * n
        self._draw_cpr_ctrl_markers()
        self.pane["A"].render()
        self._redraw_meas(self._cpr["src"])        # map-pane trace follows

    # ---- _cpr_drag_end / _cpr_cursor_angle / _cpr_rot_* / _cpr_page_drag /
    #      _cpr_rebuild: shared, in CPRMixin (viewers/cpr_mixin.py). ----

    def _fit_cpr_pane(self):
        """Fit pane A's camera to the short-axis FOV (±half mm, centred)."""
        p = self.pane["A"]
        p.reslice.Update()
        cam = p.ren.GetActiveCamera()
        cam.SetViewUp(0.0, 1.0, 0.0)
        fz = cam.GetFocalPoint()[2]
        pz = cam.GetPosition()[2]
        cam.SetFocalPoint(0.0, 0.0, fz)
        cam.SetPosition(0.0, 0.0, pz)
        cam.SetParallelScale(max(1e-3, self._cpr["half"]))
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
        # Draw-only: the LEFT pane (A)'s ▲ points the opposite way (apex on the
        # −uv side). Visual only — the image/frame, the angle readout and the
        # paging-sense are all unchanged. Applied here in the per-render
        # crosshair update, so it persists through ROTATE/SPIN and every redraw.
        apex_sgn = -1.0 if key == "A" else 1.0
        p.tri_mapper.SetInputData(
            _tris_pd([
                (pt(a, apex_sgn * sz, z), pt(a - 0.6 * sz, 0.0, z),
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

        # Keep the rotate-hint arrow glued to the (possibly rotating) crossline:
        # rebuild it from the current _cross_ang whenever this pane is
        # highlighted in rotate mode, so it follows a rotate drag / SPIN instead
        # of drifting off the line.
        hi = self._cross_hi.get(key)
        if hi is not None and hi[1] == "rotate":
            p.rot_arrow_mapper.SetInputData(self._rot_arrow_pd(key, hi[0]))

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
        _tagtext = "\n".join(head)
        p.tagact.SetInput(_tagtext)                 # top-left (fixed-size actor)
        for _ha in p.tagact_halo:                   # thin black outline copies
            _ha.SetInput(_tagtext)
        p.info.SetText(
            0, f"WW {self._win:.0f}  WL {self._lvl:.0f}"
        )                                           # bottom-left
        p.info.SetText(1, f"{key}  |  {kind}")      # bottom-right
        _ang = self._angio_angle(key)
        p.angle.SetInput(_ang)                      # bottom-centre (yellow)
        for _ha in p.angle_halo:                    # black outline copies
            _ha.SetInput(_ang)

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
        # NOTE: the old code folded ±N into the anterior hemisphere here. That
        # made a fully-reversed view (e.g. spinning the cross-line 180° so the
        # companion looks from the opposite side) collapse back to the same
        # reading. We now keep the real signed normal so a reversed LAO30
        # correctly reads RAO150, etc.
        axial = math.hypot(nx, ny)
        prim = 0.0 if axial < 1e-9 else math.degrees(math.atan2(nx, -ny))
        sec = math.degrees(math.atan2(nz, axial))
        return int(round(prim)), int(round(sec))

    def _frame_from_angio(self, prim_deg, sec_deg):
        """Inverse of _angio_angle_vals: build a pane frame (u, v, n) whose
        normal projects to the C-arm primary (LAO + / RAO −) and secondary
        (CRA + / CAU −) angle. Screen-up = patient SUPERIOR, matching the
        frontal default (LAO0 CRA0 → coronal viewed from the front).

        The detector/observer normal in patient LPS is
          n = (cos·sin prim, −cos·cos prim, sin sec)·(axial=cos sec)
        which round-trips through _angio_angle_vals; map it back into volume
        coords via the inverse patient basis."""
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
        the ROTATE tool — so the pair stays consistent."""
        if self._image is None or self._mode != "3D":
            return
        self._frame[which] = self._frame_from_angio(prim_deg, sec_deg)
        uw, _vw, nw = self._frame[which]
        other = "B" if which == "A" else "A"
        # Keep the companion's "up" continuous (see the ROTATE handler).
        _ou, ov, _on = self._frame[other]
        if float(np.dot(nw, ov)) < 0.0:
            nw = -nw
        self._frame[other] = self._ortho(uw, nw)
        self._cross_ang[which] = 0.0
        self._cross_ang[other] = 0.0
        self._pc = {"A": self._center.copy(), "B": self._center.copy()}
        self._view_initial = False
        self._refresh()

    def _angio_hit(self, which, x, y, w, h):
        """True if widget point (x, y) (Qt px, y-down) is over the bottom-centre
        angio readout of pane *which* — the right-click target that opens the
        angle dialog. Only in 3-D MPR with a real readout shown."""
        if self._image is None or self._mode != "3D":
            return False
        if not self._angio_angle(which):
            return False
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

    # ------------------------------------------------------- 3D / 2D mode
    def _set_mode(self, mode, reset_cam=False):
        """Switch between 3-D MPR (dual oblique reslice) and 2-D native-slice
        display. In 2-D only pane A is shown, locked to the acquisition
        (axial) plane, paging through native slices; the MPR-only tools and
        controls are disabled. Default mode is chosen per series on load."""
        if mode not in ("3D", "2D") or self._image is None:
            return
        # A Plane/2D/3D switch leaves short-axis mode (it repurposes pane A).
        if self._cpr is not None:
            self._cpr = None
            self._cpr_wrap.setVisible(False)
        prev = getattr(self, "_mode", None)
        self._mode = mode
        for k, b in self._mode_btns.items():
            b.setChecked(k == mode)
        is2d = (mode == "2D")
        # Disable the MPR-only tools/controls in 2-D (handled with the
        # measure-mode greying in one place).
        self._refresh_tool_availability()
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
            # Snap the reslice plane onto the nearest native slice so no z
            # interpolation occurs (the image is the acquired slice as-is).
            nz = self._image.GetDimensions()[2]
            sz = self._dims[2]
            k = int(round(self._center[2] / sz)) if sz > 1e-6 else 0
            self._slice2d = min(max(k, 0), max(0, nz - 1))
            z = self._slice2d * sz if sz > 1e-6 else 0.0
            self._center[2] = z
            self._pc = {"A": self._center.copy(), "B": self._center.copy()}
            self.pane["A"].set_overlay_visible(False)  # no crosshair in 2-D
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
            for k in ("A", "B"):
                self.pane[k].set_overlay_visible(self._cl_btn.isChecked())
            self._refresh_side_buttons()
        self._sync_slab_spin()
        self._refresh(reset_cam=reset_cam or is2d)
        self._sync_seek()

    def _page2d(self, step):
        """Page by *step* native slices in 2-D mode (integer slice index)."""
        if self._image is None:
            return
        nz = self._image.GetDimensions()[2]
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
        single image). Hidden in 3-D MPR (paging there is continuous, not
        discrete frames) and for single-frame series."""
        self._seek_wrap = QWidget()
        # Survive the shell's "Max Image" (Hide Buttons): the slice scrubber +
        # its Frame/count labels stay visible so paging is still usable.
        self._seek_wrap._mdv_keep_on_max = True
        row = QHBoxLayout(self._seek_wrap)
        row.setContentsMargins(8, 2, 8, 2)

        # Natural label point size; set_compact() toggles between the big
        # (1.55× bold) presentation size and a compact normal size.
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
        # bar shows for native-slice (auxiliary, ≤200-slice) series and is
        # hidden in 3-D MPR, so the counter appears for scout / Ca-score / thin
        # recons and not on the full 3-D volume. A "Series:" caption (mirroring
        # "Frame:") keeps it apart from the adjacent slice "N / total". Fed by
        # the shell via set_series_position.
        self._seek_series_cap = _big(QLabel(t("Series:")))
        row.addWidget(self._seek_series_cap)
        self._seek_series_lbl = _big(QLabel(""))
        self._seek_series_lbl.setMinimumWidth(66)
        self._seek_series_lbl.setToolTip(
            t("Series position in this study (current / total)"))
        row.addWidget(self._seek_series_lbl)
        self._seek_wrap.setVisible(False)
        # Apply the current compact state (set before this bar was built).
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
        if self._mode != "2D" or self._image is None:
            return
        nz = self._image.GetDimensions()[2]
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
        nz = self._image.GetDimensions()[2] if self._image is not None else 1
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
            nz = (self._image.GetDimensions()[2]
                  if self._image is not None else 1)
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
        if self._mode != "2D" or self._image is None:
            self._play2d_btn.setChecked(False)
            return
        nz = self._image.GetDimensions()[2]
        nxt = self._slice2d - 1
        if nxt < 0:
            nxt = nz - 1
        # Drive through the slider so the handle follows (fires _on_seek).
        self._seek_slider.setValue(nxt)

    def _play2d_speed_toggle(self):
        """D (2-D mode): stopped → play at 1×; playing 1× → 2×; 2× → 1× —
        the same cine key the angio viewer uses."""
        if self._mode != "2D" or self._image is None:
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

    # ------------------------------------------------- 2-D image transforms
    def _apply_2d_axes(self):
        """Set pane A's in-plane display axes (U, V) from the 2-D rotate/flip
        state. N = U×V (the actual viewing normal, so the LAO/CRA angle text
        reports the side the slice is seen from); the cut plane itself is the
        same native slice either way, and 2-D paging moves along absolute z
        (_page2d), not along N, so the paging direction is unaffected."""
        u, v = self._axes2d
        u = np.asarray(u, float).copy()
        v = np.asarray(v, float).copy()
        self._frame["A"] = (u, v, np.cross(u, v))

    #: Rt90/Lt90/Flip transforms as 2×2 matrices acting on the (u, v) frame
    #: — u' = M·(u,v). Same visual result as the 2-D-mode _axes2d swaps.
    _XFORM_2X2 = {
        "rt90": np.array([[0.0, 1.0], [-1.0, 0.0]]),    # (u,v) → (v, -u)
        "lt90": np.array([[0.0, -1.0], [1.0, 0.0]]),    # (u,v) → (-v, u)
        "fliph": np.array([[-1.0, 0.0], [0.0, 1.0]]),   # (u,v) → (-u, v)
        "flipv": np.array([[1.0, 0.0], [0.0, -1.0]]),   # (u,v) → (u, -v)
    }


    def _2d_transform(self, kind):
        """Rotate the 2-D image 90° (rt90/lt90) or flip it (fliph/flipv).
        Applied incrementally to the current display axes (composable).
        Works on the CPR short-axis too (transforms the whole stack)."""
        if self._cpr is not None:
            M = self._XFORM_2X2.get(kind)
            if M is None:
                return
            self._cpr["T"] = M @ self._cpr["T"]
            self._cpr_apply_xform()
            self._view_initial = False
            self._refresh(reset_cam=True)
            return
        if self._mode != "2D" or self._image is None:
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
        """One paging notch (keyboard / arrow): a native slice in 2-D, a
        wheel-equivalent step in 3-D."""
        if self._mode == "2D":
            self._page2d(step)
        else:
            self._wheel(self._active_pane, step)

    def _key_arrow(self, direction):
        """Drive the currently-selected tool from an arrow key. Mapping:
        Zoom/Paging/Thick = ↑/↓ only; Move = ↑↓←→; Rotate = ↑↓←→ (orthogonal,
        about the centreline, no diagonal); Spin = →/↓ CW, ←/↑ CCW; WL = same
        as the mouse drag (←/→ window, ↑/↓ level)."""
        if self._image is None:
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
            # one axis at a time → no diagonal tilt
            d = {"up": (0, -S), "down": (0, S),
                 "left": (-S, 0), "right": (S, 0)}[direction]
            self._drag(k, d[0], d[1])
            return
        if t == "SPIN":
            if self._mode == "2D":
                return
            sign = 1.0 if direction in ("right", "down") else -1.0
            self.pane[k].ren.GetActiveCamera().Roll(_SPIN_SIGN * sign * 5.0)
            self._refresh()
            return

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
            # 3-D tilt of THIS pane's plane about its own axes.
            u, v, n = self._frame[which]
            u = _rotate(_rotate(u, v, -dx * 0.5), u, -dy * 0.5)
            v = _rotate(_rotate(v, v, -dx * 0.5), u, -dy * 0.5)
            self._frame[which] = self._ortho(u, v)
            un, vn, _nn = self._frame[which]
            # KEEP the crossline angle exactly as it was (the crossline is fixed
            # to the pane's frame and rotates WITH it). This branch used to reset
            # _cross_ang to 0 — snapping an oblique crossline back to orthogonal
            # on every plane rotation. Matches the pygfx/Mac viewer.
            a = math.radians(self._cross_ang[which])
            crossdir = _norm(un * math.cos(a) + vn * math.sin(a))
            # Re-derive the companion so it still marks that crossline, with a
            # continuous orientation (see _couple_companion).
            self._couple_companion(which, crossdir)
            self._pc = {"A": self._center.copy(), "B": self._center.copy()}
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
            # Drag (and arrow) UP = zoom OUT (shrink), DOWN = zoom IN (enlarge):
            # dy<0 (up) → factor>1 → larger ParallelScale → wider view = shrink.
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
        if self._mode == "2D":
            self._page2d(1 if delta > 0 else -1)
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
        # Re-project any anatomy-anchored trace (pts3d) onto the moved planes so
        # the pseudo-centre points / lines travel WITH the image instead of
        # staying frozen on screen.
        for k in ("A", "B"):
            self._redraw_meas(k)

    def _couple_companion(self, which, crossdir) -> None:
        """Re-derive the OTHER pane as the plane ⟂ to *which* that contains the
        unit crossline *crossdir* (a 3-D vector lying in *which*'s plane),
        keeping the companion's image orientation CONTINUOUS and its crossline
        still marking that line — instead of rebuilding a fresh ortho() that
        snaps the companion's crossline back to straight. Shared by the
        crosshair-rotate gesture and the ROTATE tool so neither resets the
        crossline. Mirrors the pygfx/Mac viewer."""
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

    def _cross_zone(self, which, sx, sy):
        """Classify a screen point (sx, sy) against *which* pane's crosshair.

        Returns ``(caught, line, mode)``: caught=False → off the crosshair (the
        active tool owns the drag); else line ∈ {"H","V"} (H = the green-▲ line)
        and mode ∈ {"move","rotate"}. Pure / side-effect-free so it can drive
        BOTH the press action and the hover highlight.

        Distances are measured in NORMALISED screen space — each pane axis
        scaled to [-1, 1] (centre→edge = 1) via VTK WorldToDisplay (so camera
        roll/zoom are included). The catch band is 5%-of-screen on EACH side of
        a crossline, on both axes regardless of aspect ratio and at any zoom. Of
        the caught span, the INNER half (near the centre) translates the plane;
        the OUTER half rotates it. Mirrors the pygfx/Mac viewer."""
        if self._image is None:
            return (False, None, None)
        ccx, ccy = self._cc(which)
        canvas = self.pane[which].canvas
        ren = self.pane[which].ren
        hpix = canvas.height()
        dpr = canvas.devicePixelRatioF()

        def _w2s(ox, oy):
            """Reslice-output point (ox,oy) → Qt-widget pixels (y down)."""
            ren.SetWorldPoint(float(ox), float(oy), 0.0, 1.0)
            ren.WorldToDisplay()
            ddx, ddy, _ddz = ren.GetDisplayPoint()
            return ddx / dpr, hpix - ddy / dpr

        hx = max(1.0, canvas.width() / 2.0)
        hy = max(1.0, hpix / 2.0)
        cx, cy = _w2s(ccx, ccy)                        # crosshair centre, px

        def _ndir(ux, uy):
            """Output-basis crossline direction → unit vector in normalised
            screen space (carries camera roll + pixel aspect)."""
            px, py = _w2s(ccx + ux, ccy + uy)
            ddx, ddy = (px - cx) / hx, (py - cy) / hy
            nlen = math.hypot(ddx, ddy) or 1.0
            return ddx / nlen, ddy / nlen

        a = math.radians(self._cross_ang[which])
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
        # The central square (BOTH bands overlap = the crossline INTERSECTION):
        # dragging here recentres — the grabbed point moves to the crosshair
        # centre AND to the middle of the screen (a live version of the double-
        # click recentre). The lines themselves (only one band) keep their
        # move / rotate behaviour and do NOT pan the view.
        if on_h and on_v:
            return (True, "C", "center")
        # The ▲ markers sit on the HORIZONTAL crossline → green-▲ line.
        grab_h = on_h
        along = along_h if grab_h else along_v
        mode = "move" if along <= mid else "rotate"
        return (True, "H" if grab_h else "V", mode)

    def _cross_press(self, which, sx, sy) -> bool:
        """Start a CrossLine gesture (overriding the tool) when the press lands
        ON the crosshair; else False so the active tool handles the drag. The
        caught line/mode also lock the vivid highlight for the whole drag."""
        caught, line, mode = self._cross_zone(which, sx, sy)
        if not caught:
            return False
        # Intersection grab → live recentre (handled in _cross_move); no
        # per-line gesture state needed.
        if mode == "center":
            self._cross_mode = "center"
            self._cross_dragging = which
            self._set_cross_highlight(which, line, mode)
            return True
        wx, wy = self._disp_to_world(which, sx, sy)   # world (gesture state)
        ccx, ccy = self._cc(which)
        a = math.radians(self._cross_ang[which])
        grab_h = (line == "H")
        if mode == "move":
            self._cross_mode = "move"
            # Lock the slide to the grabbed line so the grab is deterministic:
            # the green-▲ (H) line slides ⟂ to itself = along uv (parallel move
            # of the line); the non-▲ (V) line slides along uh. Output basis.
            ouh = np.array([math.cos(a), math.sin(a)])
            ouv = np.array([-math.sin(a), math.cos(a)])
            self._cross_axis = ouv if grab_h else ouh
            self._cross_ppt = (wx, wy)
        else:
            self._cross_mode = "rotate"
            self._cross_prev = math.atan2(wy - ccy, wx - ccx)
        self._cross_dragging = which
        self._set_cross_highlight(which, line, mode)
        return True

    # ---- centreline hover / drag highlight -----------------------------
    _CROSS_BASE = (1.0, 0.85, 0.0)          # normal crosshair: amber, 50%
    _CROSS_HI = (0.8, 0.8, 0.0)             # caught line: yellow, opaque (dimmed)

    def _hover_cross(self, which, sx, sy) -> None:
        """Mouse moved over a pane with NO button down: preview whether a press
        here would grab the centreline (vivid highlight + rotate arrow) or fall
        through to the active tool (normal crosshair)."""
        if self._cross_dragging is not None or self._image is None:
            return
        caught, line, mode = self._cross_zone(which, sx, sy)
        if caught:
            self._set_cross_highlight(which, line, mode)
        else:
            self._set_cross_highlight(which, None, None)

    def _set_cross_highlight(self, which, line, mode) -> None:
        """Colour *which* pane's crosshair to show the pending/active gesture:
        the caught line goes vivid cyan (thicker, opaque) and, in the rotate
        zone, a curved arrow appears beside it; line=None restores the normal
        amber crosshair. Cheap: only actor properties change, then one render.
        No-op in Max-Image mode (crosshair actors hidden)."""
        p = self.pane[which]
        # In presentation ("Max Image") mode the crosshair is hidden — never
        # light it up or show the arrow there.
        if not p.cross[0][1].GetVisibility():
            line = None
        new = (line, mode) if line else None
        if self._cross_hi.get(which) == new:
            return
        self._cross_hi[which] = new
        for _src, act in p.cross:
            pr = act.GetProperty()
            pr.SetColor(*self._CROSS_BASE)
            pr.SetLineWidth(1.0)
            pr.SetOpacity(0.5)
        p.rot_arrow.SetVisibility(False)
        if line == "C":
            # Intersection (recentre) zone: light up BOTH crosslines, no arrow.
            for _src, act in p.cross:
                pr = act.GetProperty()
                pr.SetColor(*self._CROSS_HI)
                pr.SetLineWidth(1.6)
                pr.SetOpacity(1.0)
        elif line is not None:
            pr = p.cross[0 if line == "H" else 1][1].GetProperty()
            pr.SetColor(*self._CROSS_HI)
            pr.SetLineWidth(1.6)
            pr.SetOpacity(1.0)
            if mode == "rotate":
                p.rot_arrow_mapper.SetInputData(self._rot_arrow_pd(which, line))
                p.rot_arrow.SetVisibility(True)
        p.render()

    def _rot_arrow_pd(self, which, line) -> vtkPolyData:
        """Two small double-headed curved arrows tangent to a circle about the
        crosshair centre — one on EACH side of the caught line's outer ends —
        the 'this rotates (either way)' hint. Compact (~¼ the first pass)."""
        th = math.radians(self._cross_ang[which])
        c_, s_ = math.cos(th), math.sin(th)
        base = (c_, s_) if line == "H" else (-s_, c_)   # caught line direction
        ccx, ccy = self._cc(which)
        ps = self.pane[which].ren.GetActiveCamera().GetParallelScale()
        r = 0.60 * ps                                   # arc radius (outer zone)
        base_ang = math.atan2(base[1], base[0])
        span = math.radians(3.75)                       # ⅛ of the first pass
        steps = 6
        hs = 0.0125 * ps                                # ⅛ head size

        def _head(tip, nxt):
            """Two barbs at *tip*, fanned back from the outward tangent tip←nxt."""
            tx, ty = tip[0] - nxt[0], tip[1] - nxt[1]
            tl = math.hypot(tx, ty) or 1.0
            tx, ty = tx / tl, ty / tl
            out = []
            for deg in (28.0, -28.0):
                ca, sa = math.cos(math.radians(deg)), math.sin(math.radians(deg))
                bx = (-tx) * ca - (-ty) * sa
                by = (-tx) * sa + (-ty) * ca
                out.append([tip, (tip[0] + bx * hs, tip[1] + by * hs)])
            return out

        polylines = []
        for side in (0.0, math.pi):                     # both ends of the line
            ca0 = base_ang + side
            arc = []
            for i in range(steps + 1):
                ang = ca0 - span + (2.0 * span) * i / steps
                arc.append((ccx + r * math.cos(ang), ccy + r * math.sin(ang)))
            polylines.append(arc)
            polylines += _head(arc[-1], arc[-2])        # head at one end …
            polylines += _head(arc[0], arc[1])          # … and the other
        return _polylines_pd(polylines)

    def _cross_move(self, which, sx, sy):
        # Intersection drag: the crosshair centre FOLLOWS the cursor while the
        # background image stays put (only _center moves, not _pc / the camera).
        # The actual recentre — moving that point to the middle of the screen —
        # happens on RELEASE (see _PaneCanvas.mouseReleaseEvent → _recenter).
        if self._cross_mode == "center":
            wx, wy = self._disp_to_world(which, sx, sy)
            m = self._matrix(which)
            vol = m.MultiplyPoint((wx, wy, 0.0, 1.0))
            self._center = np.array([vol[0], vol[1], vol[2]])
            self._clamp_center()
            self._view_initial = False
            self._refresh()                # crosshair moves; images unchanged
            return
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
            # Moving the centre ALONG uh — the green-▲ line direction, the shared
            # crossline that also lies in the companion plane — is the "non-▲
            # line" translate.
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
                    limp = self.pane[pk].ren.GetActiveCamera().GetParallelScale()
                    op = offp - max(-limp, min(limp, offp))
                    if abs(op) > abs(over):
                        over = op                  # the binding pane's overflow
                if over:
                    self._center = self._center - over * dn
            else:
                self._clamp_center()               # ▲ line: reslice → box-clamp OK
                self._pc[other] = self._center.copy()
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
        # Re-derive the companion plane so it stays ⟂ to this pane and still
        # contains the rotated crossline, keeping its orientation continuous
        # (see _couple_companion — it does NOT snap the companion straight).
        self._couple_companion(which, crossdir)
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
            # Restore the native 2-D orientation (clear any rotate/flip).
            self._axes2d = (np.array([1.0, 0.0, 0.0]),
                            np.array([0.0, -1.0, 0.0]))
            self._sync_slab_spin()
            self._view_initial = True
            # Re-apply the current mode (re-locks 2-D / restores dual MPR) and
            # refits the camera.
            self._set_mode(self._mode, reset_cam=True)
        else:
            self._win, self._lvl = self._win0, self._lvl0
            self._refresh()

    def _patient_axis_vol(self, p):
        """A patient-LPS direction (e.g. (1,0,0)=Left) expressed in volume
        coords, via the inverse patient basis. Falls back to the raw axis."""
        pb = getattr(self, "_pbasis", None)
        try:
            inv = (np.linalg.inv(np.asarray(pb, float))
                   if pb is not None else np.eye(3))
        except np.linalg.LinAlgError:
            inv = np.eye(3)
        return _norm(inv @ np.array(p, float))

    def _flash_recalc(self):
        """Briefly flash the ReCalc button green so the user can SEE the click
        registered (ReCalc often changes nothing visible when already correct)."""
        btn = getattr(self, "_recalc_btn", None)
        if btn is None:
            return
        btn.setStyleSheet("background:#2ecc71;color:#101010;")   # flash green
        QTimer.singleShot(
            380, lambda: btn.setStyleSheet("background:#8a8a8a;color:#101010;"))

    def _recalc_companion(self):
        """ReCalc: rebuild the OTHER pane as the plane that cuts the ACTIVE pane
        along its green-▲ centre line, fixing a MIRROR while keeping the view.

        _couple_companion rebuilds it right-handed but PRESERVES the companion's
        existing viewing side — so a mirrored companion would stay mirrored. We
        then force the non-mirror side: screen-right ≈ patient LEFT (the
        _init_frames convention). Flipping u (and n) keeps the up vector, so the
        zoom/rotation are preserved and only the left-right mirror is corrected.
        The master pane is untouched."""
        if self._image is None:
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
        self._pc[other] = self._center.copy()
        self._view_initial = False
        self._refresh()
        self._flash_recalc()

    def _apply_preset(self, name):
        if name in CT_WL_PRESETS:
            self._win, self._lvl = (float(x) for x in CT_WL_PRESETS[name])
            self._refresh()

    def keyPressEvent(self, e):
        # Arrow keys are handled by QShortcuts (see __init__) so a focused
        # spin-box / combo / slider can't swallow them; only letter tool keys
        # and C (ColorMap) are handled here.
        if e.key() == Qt.Key.Key_C:               # C = toggle ColorMap
            self._cmap_btn.setChecked(not self._cmap_btn.isChecked())
            self._toggle_color()
            return
        # Angio-parity cine keys in 2-D mode: D = play / ×2 toggle, S = stop.
        # In 3-D, S keeps selecting the Spin tool (MPR-only, so no conflict).
        if self._mode == "2D":
            if e.key() == Qt.Key.Key_D:
                self._play2d_speed_toggle()
                return
            if e.key() == Qt.Key.Key_S:
                self._play2d_btn.setChecked(False)
                return
        tool = _TOOL_KEYS.get(e.key())
        if tool:
            self._set_tool(tool)
        else:
            super().keyPressEvent(e)
