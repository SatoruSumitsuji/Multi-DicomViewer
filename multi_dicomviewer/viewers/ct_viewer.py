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
from PyQt6.QtCore import QPoint as QtPoint, Qt, QThread, QTimer, pyqtSignal
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
from vtkmodules.vtkImagingGeneral import vtkImageGaussianSmooth
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
from multi_dicomviewer.core import settings
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


#: Idle measure-vertex dot size (physical px, VTK PointSize).
_MEAS_PT_PX = 11.0
#: Off-plane hollow-ring OUTER radius (physical px). Kept a touch SMALLER than
#: the in-plane dot's radius (_MEAS_PT_PX/2) so the in-range filled dot reads as
#: the more prominent of the two.
_CPR_RING_OUTER_PX = 4.0


def _ring_polylines(centers, radius, seg=20):
    """Closed-circle outlines (one polyline per centre) for hollow off-plane
    markers — the 中抜き (ring) depth cue, matching the Mac viewer."""
    ang = [2.0 * math.pi * k / seg for k in range(seg + 1)]  # +1 closes the loop
    return [[(cx + radius * math.cos(a), cy + radius * math.sin(a))
             for a in ang]
            for (cx, cy) in centers]


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


def _lv_pts_pd(pts_xy, colors, z=0.8) -> vtkPolyData:
    """Vert-cell polydata for the LV axis pick markers — one point per (x, y)
    output-plane coord with a per-cell RGB colour (apex vs basal)."""
    from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
    pd = vtkPolyData()
    vp = vtkPoints()
    verts = vtkCellArray()
    col = vtkUnsignedCharArray()
    col.SetNumberOfComponents(3)
    col.SetName("Colors")
    for i, (x, y) in enumerate(pts_xy):
        vp.InsertNextPoint(float(x), float(y), float(z))
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)
        r, g, b = colors[i]
        col.InsertNextTuple3(int(r), int(g), int(b))
    pd.SetPoints(vp)
    pd.SetVerts(verts)
    pd.GetCellData().SetScalars(col)
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


def _colored_dashed_rgba_pd(segments, colors, z=0.62) -> vtkPolyData:
    """Like _colored_dashed_pd but per-dash cell carries RGBA (uint8; a 3-tuple
    is padded opaque). Used for the off-plane 点線 outline, whose 50% alpha must
    survive into each dash cell — the caller's mapper must direct-colour cells."""
    pd = vtkPolyData()
    pts = vtkPoints()
    lines = vtkCellArray()
    cell_rgb = vtkUnsignedCharArray()
    cell_rgb.SetNumberOfComponents(4)
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
            cell_rgb.InsertNextTuple4(*_rgba(rgb))
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


def _lvv_transparent_lut() -> vtkLookupTable:
    """A fully transparent LUT (the LV Vol highlight overlay is invisible until
    a HU range is armed)."""
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(2)
    lut.SetTableRange(-1000.0, 3000.0)
    lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)
    lut.SetTableValue(1, 0.0, 0.0, 0.0, 0.0)
    lut.Build()
    return lut


def _lvv_highlight_lut(lo: float, hi: float,
                       rgb=(0.1, 0.9, 1.0), alpha: float = 0.45) -> vtkLookupTable:
    """LUT that tints only HU in [lo, hi] (semi-transparent), rest transparent —
    the LV Vol in-range voxel highlight over the grayscale reslice."""
    lut = vtkLookupTable()
    lo_r, hi_r = -1000.0, 3000.0
    n = 400
    lut.SetNumberOfTableValues(n)
    lut.SetTableRange(lo_r, hi_r)
    for i in range(n):
        hu = lo_r + (hi_r - lo_r) * i / (n - 1)
        if lo <= hu <= hi:
            lut.SetTableValue(i, rgb[0], rgb[1], rgb[2], alpha)
        else:
            lut.SetTableValue(i, 0.0, 0.0, 0.0, 0.0)
    lut.Build()
    return lut


def _lvv_mask_lut(on: bool, rgb=(1.0, 0.25, 0.25),
                  alpha: float = 0.9) -> vtkLookupTable:
    """LUT for the measured-region mask reslice: value 1 → red (when on), value
    0 → transparent. Alpha is high and the colour saturated so the red clearly
    DOMINATES the cyan blood tint below ('全面表示' — red on top)."""
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(2)
    lut.SetTableRange(0.0, 1.0)
    lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)
    lut.SetTableValue(1, rgb[0], rgb[1], rgb[2], alpha if on else 0.0)
    lut.Build()
    return lut


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

#: Max colour-reslice pixels/side (A): in colour mode the MPR is resliced at
#: this resolution over the whole FOV so magnified band boundaries stay smooth
#: curves instead of a voxel-grid staircase. Grayscale keeps voxel resolution.
_RESLICE_NPX_CAP = 2048
#: Tighter per-side cap used ONLY during an interactive drag (pan/zoom/rotate),
#: so a zoomed-out thick-slab MPR stays responsive; the full cap returns on
#: release when _refresh() repaints at full quality.
_RESLICE_DRAG_NPX_CAP = 1024


def _smooth_lut_edges(col: np.ndarray, alpha: np.ndarray, sigma: float = 5.0):
    """Soften the hard band edges of a step colour map: Gaussian-blur the
    premultiplied colour + alpha along the HU axis so adjacent bands (and the
    band↔grayscale boundary) blend over a short HU ramp. This removes the
    posterised "blocky / speckly" look colour mapping gave on noisy CT, where
    HU jittering across a hard band edge flipped whole voxels between colours.
    Edge-padded so the HU extremes don't dip. Returns (colour, alpha)."""
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


def _band_lut_rgb(bands, opacity, win, lvl, n: int = 512) -> np.ndarray:
    """(n, 3) float RGB for HU in [_HU_LO, _HU_HI]: enabled band colours blended
    over the windowed grayscale by *opacity*, with CRISP/hard band edges
    (matching SSMView). Outside any band → grayscale. Smoothness of the on-image
    band boundaries is handled SPATIALLY (a Gaussian on the reslice), not in the
    LUT. Shared by the VTK LUT and (mirrored) the pygfx colormap."""
    hu = _HU_LO + (_HU_HI - _HU_LO) * np.arange(n) / (n - 1)
    glo = lvl - win / 2.0
    span = max(1e-6, float(win))
    g = np.clip((hu - glo) / span, 0.0, 1.0)
    op = min(1.0, max(0.0, float(opacity)))
    col = np.zeros((n, 3), np.float64)
    alpha = np.zeros(n, np.float64)
    assigned = np.zeros(n, bool)
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
    rgb = g[:, None] * (1.0 - eff) + col * eff
    return np.clip(rgb, 0.0, 1.0)


def _band_lut(bands, opacity, win, lvl) -> vtkLookupTable:
    """HU -> RGB table (crisp bands). Inside an enabled band: band colour blended
    over the windowed grayscale by *opacity*; outside: grayscale."""
    rgb = _band_lut_rgb(bands, opacity, win, lvl)
    n = rgb.shape[0]
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(n)
    lut.SetTableRange(_HU_LO, _HU_HI)
    for i in range(n):
        lut.SetTableValue(i, float(rgb[i, 0]), float(rgb[i, 1]),
                          float(rgb[i, 2]), 1.0)
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
        # (LV blood-pool volume apex/threshold points are placed by DOUBLE-click
        # — see mouseDoubleClickEvent — so single-click still navigates.)
        # LV apex points: place the two apex vertices (apex phase) or grab an
        # existing marker to drag it. Checked before Measure so a click near a
        # marker moves the apex instead of adding a trace point; _lv_apex_press
        # yields while a line is being drawn so tracing always wins.
        if (e.button() == Qt.MouseButton.LeftButton
                and self._owner._lv is not None):
            _sh = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            r = self._owner._lv_apex_press(
                self._which, e.position().x(), e.position().y(), _sh)
            if r == "place":
                return
            if r in ("endo", "epi"):
                self._owner._lv_apex_drag = r
                # Snapshot apex + borders now so the drag is one Ctrl+Z step.
                self._owner._lv_apex_snap = self._owner._lv_geom_snap()
                self._cross = False
                self._meas_drag = False
                self._lv_line = False
                self._last = e.position()
                return
        # SAX: grab the thick ○-marked LEVEL line (long-axis pane → translate the
        # cross-section level) or CENTRELINE (short-axis pane → rotate the
        # meridian) to review the endo/epi borders. Checked before Measure so the
        # line wins, but _lv_line_press yields to a measure-handle grab so border
        # points still edit. The line thickens slightly while held.
        if (e.button() == Qt.MouseButton.LeftButton
                and self._owner._lv_sax_active()):
            kind = self._owner._lv_line_press(
                self._which, e.position().x(), e.position().y())
            if kind:
                self._owner._lv_line_drag = kind
                # Snapshot the SAX level/meridian now → one Ctrl+Z step on release.
                self._owner._gesture_begin()
                self._owner._lv_line_set_grabbed(self._which, True)
                self._lv_line = True
                self._cross = False
                self._meas_drag = False
                self._last = e.position()
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
        # A view-changing drag begins here (crosshair move/rotate OR the active
        # tool: Zoom/Move/Rotate/Spin/Thick/Paging). Snapshot for Ctrl+Z now;
        # the gesture is committed as one step on release.
        self._owner._gesture_begin()
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
            ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
            self._owner._drag(self._which, p.x() - self._last.x(),
                              p.y() - self._last.y(), shift, p.x(), p.y(),
                              ctrl=ctrl)
            self._last = p
            return
        if self._owner._lv_apex_drag is not None and self._last is not None:
            self._owner._lv_apex_move(
                self._which, e.position().x(), e.position().y())
            self._last = e.position()
            return
        if getattr(self, "_lv_line", False) and self._last is not None:
            self._owner._lv_line_move(
                self._which, e.position().x(), e.position().y())
            self._last = e.position()
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
            # No button down. Point-probe armed → show the HU under the cursor;
            # any other measure type armed → a click starts a measurement (no
            # hover); otherwise hover-preview the centreline.
            x, y = e.position().x(), e.position().y()
            # Hover-highlight (green) an existing control point so the user sees
            # it will be grabbed before pressing. Skip while a draft trace is
            # active here (then a click adds a point, it doesn't grab a handle).
            drafting = (self._owner._draft is not None
                        and self._owner._draft.get("pane") == self._which)
            if self._owner._meas_on and not drafting:
                self._owner._measure_hover_handle(self._which, x, y)
            else:
                self._owner._clear_hover_handle()
            # Glow the apex while the tracing cursor is within its snap range.
            self._owner._lv_apex_hover(self._which, x, y)
            # SAX: with no border point under the cursor, thicken the ○ line
            # handle so the user still sees the level / meridian is grabbable.
            if (self._owner._lv_sax_active()
                    and self._owner._meas_hover_handle is None):
                self._owner._lv_line_hover(self._which, x, y)
                return
            if self._owner._lv_sax_active():
                self._owner._lv_line_set_grabbed(self._which, False)
                return
            if (self._owner._meas_on
                    and self._owner._meas_type == "point"):
                self._owner._measure_hover(
                    self._which, e.position().x(), e.position().y())
            elif not (self._owner._meas_on and self._owner._meas_type):
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
        ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self._owner._drag(
            self._which, p.x() - self._last.x(), p.y() - self._last.y(),
            shift, p.x(), p.y(), ctrl=ctrl,
        )
        self._last = p

    def mouseReleaseEvent(self, e):
        if self._owner._lv_apex_drag is not None:
            self._owner._lv_apex_drag = None
            self._owner._lv_record_geom(self._owner._lv_apex_snap)  # Ctrl+Z step
            self._owner._lv_apex_snap = None
            self._last = None
            return
        if getattr(self, "_lv_line", False):
            self._owner._lv_line_drag = None
            self._owner._lv_line_set_grabbed(self._which, False)
            self._owner._lv_line_hi[self._which] = False   # re-hover next move
            self._lv_line = False
            self._owner._gesture_commit()      # commit the SAX level/meridian drag
            self._last = None
            return
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
        # Commit the drag (Zoom/Move/Rotate/Spin/Thick/Paging or a centreline
        # move·rotate) as one Ctrl+Z step. A click/double-click recentre records
        # itself (it leaves _gesture_moved False), so this won't double-record.
        self._owner._gesture_commit()
        # End the centreline gesture and drop its highlight (a fresh hover will
        # re-preview if the cursor is still on a line).
        self._owner._cross_dragging = None
        self._owner._set_cross_highlight(self._which, None, None)

    def leaveEvent(self, e):
        # Cursor left the pane → clear any hover highlight + point-probe HU.
        if self._owner._cross_dragging is None:
            self._owner._set_cross_highlight(self._which, None, None)
        self._owner._measure_hover_clear(self._which)
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


class _LvvWorker(QThread):
    """Runs the (few-second) LV blood-pool volume computation off the UI thread
    so a busy progress bar can animate; emits the result dict."""
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


class _Pane:
    """One reslice view: pipeline + renderer + CrossLine actors."""

    def __init__(self, canvas: _PaneCanvas):
        self.canvas = canvas
        self.reslice = vtkImageReslice()
        self.reslice.SetInputData(_placeholder_image())
        self.reslice.SetOutputDimensionality(2)
        self.reslice.SetInterpolationModeToLinear()
        self.reslice.SetBackgroundLevel(-1000.0)
        # Optional spatial smoothing of the HU reslice BEFORE colour mapping, so
        # the hard colour-band boundaries read as smooth curves instead of the
        # voxel-grid staircase (see CTViewer._refresh; std set per-redraw from
        # the mm strength). colours is wired to the reslice by default (smoothing
        # off) and re-routed through gauss when the strength is > 0.
        self.gauss = vtkImageGaussianSmooth()
        self.gauss.SetDimensionality(2)
        self.gauss.SetInputConnection(self.reslice.GetOutputPort())
        self.colors = vtkImageMapToColors()
        self.colors.SetOutputFormatToRGB()
        self.colors.SetLookupTable(_gray_lut(400.0, 40.0))  # default
        self.colors.SetInputConnection(self.reslice.GetOutputPort())
        self.actor = vtkImageActor()
        self.actor.GetMapper().SetInputConnection(self.colors.GetOutputPort())
        # Linear (bilinear) display interpolation so magnifying the colour
        # reslice smooths the pixels instead of showing a hard nearest-neighbour
        # staircase at the band boundaries.
        self.actor.SetInterpolate(True)
        # LV Vol HU-range HIGHLIGHT overlay: a second colour mapper on the SAME
        # reslice output (so it tracks the plane for free), tinting only voxels
        # whose HU is in the chosen blood range. Transparent LUT until armed.
        self.colors_hl = vtkImageMapToColors()
        self.colors_hl.SetOutputFormatToRGBA()
        self.colors_hl.SetLookupTable(_lvv_transparent_lut())
        self.colors_hl.SetInputConnection(self.reslice.GetOutputPort())
        self.actor_hl = vtkImageActor()
        self.actor_hl.GetMapper().SetInputConnection(
            self.colors_hl.GetOutputPort())
        self.actor_hl.SetInterpolate(False)
        # Just in front of the grayscale (z=0) but behind the crosshair (0.5) /
        # markers, so the tint sits on the image without z-fighting.
        self.actor_hl.SetPosition(0.0, 0.0, 0.05)
        # LV Vol MEASURED-region overlay: reslice a separate 0/1 mask volume on
        # the SAME plane and tint the 1s light red. Drawn above the blood tint.
        self.reslice_mask = vtkImageReslice()
        self.reslice_mask.SetInputData(_placeholder_image())
        self.reslice_mask.SetOutputDimensionality(2)
        self.reslice_mask.SetInterpolationModeToNearestNeighbor()
        self.reslice_mask.SetBackgroundLevel(0.0)
        self.colors_mask = vtkImageMapToColors()
        self.colors_mask.SetOutputFormatToRGBA()
        self.colors_mask.SetLookupTable(_lvv_mask_lut(False))
        self.colors_mask.SetInputConnection(self.reslice_mask.GetOutputPort())
        self.actor_mask = vtkImageActor()
        self.actor_mask.GetMapper().SetInputConnection(
            self.colors_mask.GetOutputPort())
        self.actor_mask.SetInterpolate(False)
        self.actor_mask.SetPosition(0.0, 0.0, 0.10)
        self.ren = vtkRenderer()
        self.ren.SetBackground(0.0, 0.0, 0.0)
        self.ren.GetActiveCamera().ParallelProjectionOn()
        self.ren.AddActor(self.actor)
        self.ren.AddActor(self.actor_hl)              # blood tint over grayscale
        self.ren.AddActor(self.actor_mask)           # measured region (red) on top

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
        self.slab_actors = []
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
            self.slab_actors.append(a)
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
        # LV EF (Phase 1): the apex→base long axis line + the 3 picked axis
        # points (apex green, basal cyan). Own actors so they draw over the MPR.
        self.lv_line_mapper = vtkPolyDataMapper()
        self.lv_line_mapper.SetInputData(vtkPolyData())
        lla = vtkActor()
        lla.SetMapper(self.lv_line_mapper)
        lla.GetProperty().SetColor(1.0, 0.85, 0.0)      # yellow long axis
        lla.GetProperty().SetLineWidth(2.4)
        if hasattr(lla.GetProperty(), "SetRenderLinesAsTubes"):
            lla.GetProperty().SetRenderLinesAsTubes(True)
        self.ren.AddActor(lla)
        self.lv_line_actor = lla       # kept so the SAX level/centre line can be
        self.lv_line_w = 2.4           # thickened while grabbed (see _lv_line_*)
        # LV wall-thickness colour map (translucent annulus fill between the
        # short-axis endo & epi borders; per-cell RGBA set by _redraw_lv).
        self.lv_wall_mapper = vtkPolyDataMapper()
        self.lv_wall_mapper.SetInputData(vtkPolyData())
        self.lv_wall_mapper.ScalarVisibilityOn()
        self.lv_wall_mapper.SetScalarModeToUseCellData()
        self.lv_wall_mapper.SetColorModeToDirectScalars()
        lwa = vtkActor()
        lwa.SetMapper(self.lv_wall_mapper)
        lwa.GetProperty().SetOpacity(1.0)    # alpha lives per-cell
        self.ren.AddActor(lwa)
        self.lv_pts_mapper = vtkPolyDataMapper()
        self.lv_pts_mapper.SetInputData(vtkPolyData())
        self.lv_pts_mapper.ScalarVisibilityOn()
        self.lv_pts_mapper.SetScalarModeToUseCellData()
        self.lv_pts_mapper.SetColorModeToDirectScalars()
        lpa = vtkActor()
        lpa.SetMapper(self.lv_pts_mapper)
        lpa.GetProperty().SetPointSize(8.0)     # SCREEN px → constant under zoom
        if hasattr(lpa.GetProperty(), "SetRenderPointsAsSpheres"):
            lpa.GetProperty().SetRenderPointsAsSpheres(True)
        self.ren.AddActor(lpa)
        self.lv_pts_actor = lpa
        # LV user-defined APEX markers (endo=red, epi=green) — the vertices the
        # reconstructed surface converges to. Bigger than the crossing dots and
        # own actor so they draw over everything and stay grabbable.
        self.lv_apex_mapper = vtkPolyDataMapper()
        self.lv_apex_mapper.SetInputData(vtkPolyData())
        self.lv_apex_mapper.ScalarVisibilityOn()
        self.lv_apex_mapper.SetScalarModeToUseCellData()
        self.lv_apex_mapper.SetColorModeToDirectScalars()
        lapx = vtkActor()
        lapx.SetMapper(self.lv_apex_mapper)
        lapx.GetProperty().SetPointSize(15.0)   # SCREEN px → constant under zoom
        if hasattr(lapx.GetProperty(), "SetRenderPointsAsSpheres"):
            lapx.GetProperty().SetRenderPointsAsSpheres(True)
        self.ren.AddActor(lapx)
        self.lv_apex_actor = lapx
        # LV Vol envelope ("loose bag") preview: dotted boundary of the region
        # near this pane's plane. Small yellow points; own actor.
        self.lvv_env_mapper = vtkPolyDataMapper()
        self.lvv_env_mapper.SetInputData(vtkPolyData())
        self.lvv_env_mapper.ScalarVisibilityOn()
        self.lvv_env_mapper.SetScalarModeToUseCellData()
        self.lvv_env_mapper.SetColorModeToDirectScalars()
        lenv = vtkActor()
        lenv.SetMapper(self.lvv_env_mapper)
        lenv.GetProperty().SetPointSize(3.0)
        self.ren.AddActor(lenv)
        self.lvv_env_actor = lenv
        # Highlighted SAX crossing (the one that follows an active long-axis
        # edit) — green, TWICE the yellow crossing-dot radius (own actor so it
        # can be a larger point).
        self.lv_hi_mapper = vtkPolyDataMapper()
        self.lv_hi_mapper.SetInputData(vtkPolyData())
        self.lv_hi_mapper.ScalarVisibilityOn()
        self.lv_hi_mapper.SetScalarModeToUseCellData()
        self.lv_hi_mapper.SetColorModeToDirectScalars()
        lhi = vtkActor()
        lhi.SetMapper(self.lv_hi_mapper)
        lhi.GetProperty().SetPointSize(16.0)    # 2× the 8 px crossing dots
        if hasattr(lhi.GetProperty(), "SetRenderPointsAsSpheres"):
            lhi.GetProperty().SetRenderPointsAsSpheres(True)
        self.ren.AddActor(lhi)
        self.lv_hi_actor = lhi
        # Captured endo (red) / epi (green) borders — redrawn from the model so
        # a traced border stays visible (and re-appears on revisiting a plane).
        self.lv_endo_mapper = vtkPolyDataMapper()
        self.lv_endo_mapper.SetInputData(vtkPolyData())
        lea = vtkActor()
        lea.SetMapper(self.lv_endo_mapper)
        lea.GetProperty().SetColor(1.0, 0.25, 0.25)     # endo red
        lea.GetProperty().SetLineWidth(2.4)
        if hasattr(lea.GetProperty(), "SetRenderLinesAsTubes"):
            lea.GetProperty().SetRenderLinesAsTubes(True)
        self.ren.AddActor(lea)
        self.lv_epi_mapper = vtkPolyDataMapper()
        self.lv_epi_mapper.SetInputData(vtkPolyData())
        lpe = vtkActor()
        lpe.SetMapper(self.lv_epi_mapper)
        lpe.GetProperty().SetColor(0.25, 0.85, 0.35)    # epi green
        lpe.GetProperty().SetLineWidth(2.4)
        if hasattr(lpe.GetProperty(), "SetRenderLinesAsTubes"):
            lpe.GetProperty().SetRenderLinesAsTubes(True)
        self.ren.AddActor(lpe)
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
        mp.GetProperty().SetPointSize(_MEAS_PT_PX)
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
        # Vessel-trace vertices that lie OFF the current cutting plane: drawn as
        # HOLLOW 50% yellow rings (中抜き) — a depth cue that they sit in front
        # of / behind the shown slice. Ring outlines (per-cell RGBA so the 50%
        # alpha lives in the cell), NOT filled dots, matching the Mac viewer.
        self.meas_pts_off_mapper = vtkPolyDataMapper()
        self.meas_pts_off_mapper.SetInputData(vtkPolyData())
        self.meas_pts_off_mapper.ScalarVisibilityOn()
        self.meas_pts_off_mapper.SetScalarModeToUseCellData()
        self.meas_pts_off_mapper.SetColorModeToDirectScalars()
        mpo = vtkActor()
        mpo.SetMapper(self.meas_pts_off_mapper)
        mpo.GetProperty().SetColor(1.0, 0.85, 0.0)    # fallback (cells carry RGBA)
        mpo.GetProperty().SetLineWidth(2.0)
        if hasattr(mpo.GetProperty(), "SetRenderLinesAsTubes"):
            mpo.GetProperty().SetRenderLinesAsTubes(True)
        self.ren.AddActor(mpo)
        # Keep the ring width DPR-scaled like the other measure lines (the list
        # was built before this actor existed, so append it here).
        self._meas_line_actors.append((2.4, mpo))    # off-plane hollow rings
        # Fully off-plane trace SEGMENTS (both endpoints off the slice): drawn
        # as a 点線 (dotted) at 50% so you can tell at a glance which stretch of
        # the pseudo-centreline is out of the shown cross-section. Per-cell RGBA
        # (alpha lives in the dash cells), tube-rendered so the dashes show.
        self.meas_off_dash_mapper = vtkPolyDataMapper()
        self.meas_off_dash_mapper.SetInputData(vtkPolyData())
        self.meas_off_dash_mapper.ScalarVisibilityOn()
        self.meas_off_dash_mapper.SetScalarModeToUseCellData()
        self.meas_off_dash_mapper.SetColorModeToDirectScalars()
        mod = vtkActor()
        mod.SetMapper(self.meas_off_dash_mapper)
        mod.GetProperty().SetColor(1.0, 0.85, 0.0)   # fallback (cells carry RGBA)
        mod.GetProperty().SetLineWidth(2.0)
        if hasattr(mod.GetProperty(), "SetRenderLinesAsTubes"):
            mod.GetProperty().SetRenderLinesAsTubes(True)
        self.ren.AddActor(mod)
        self._meas_line_actors.append((2.4, mod))    # off-plane dotted outline
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

        # LV wall-thickness legend (bottom-left) — one coloured line per gap
        # band (red <5 / orange 5-7 / yellow 7-9 / green >9 mm). Shown only while
        # the short-axis wall map is up (see _lv_update_wall_legend).
        self.lv_wall_legend = []
        for i in range(4):
            a = vtkTextActor()
            a.SetTextScaleModeToNone()
            tp = a.GetTextProperty()
            tp.SetFontFamilyToArial()
            _set_vtk_tag_font(tp)
            tp.SetFontSize(16)
            tp.SetBold(True)
            tp.SetJustificationToLeft()
            tp.SetVerticalJustificationToBottom()
            a.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            a.GetPositionCoordinate().SetValue(0.012, 0.012 + i * 0.045)
            a.SetInput("")
            a.SetVisibility(False)
            self.ren.AddViewProp(a)
            self.lv_wall_legend.append(a)

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

        # Follows the cursor to show the HU under it while the Point probe tool
        # is armed (positioned in display pixels on each hover; see
        # CTViewer._measure_hover).
        self.hover_hu = vtkTextActor()
        _htp = self.hover_hu.GetTextProperty()
        _htp.SetFontSize(15)
        _htp.SetColor(1.0, 1.0, 0.4)
        _htp.SetBold(True)
        self.hover_hu.GetPositionCoordinate().SetCoordinateSystemToDisplay()
        self.hover_hu.SetInput("")
        self.ren.AddViewProp(self.hover_hu)

        self.canvas.GetRenderWindow().AddRenderer(self.ren)

    def set_slab_visible(self, on: bool) -> None:
        """Show/hide just the two slab-width parallel lines (not the whole
        crosshair) — LV trace mode hides these but keeps the centreline."""
        for a in self.slab_actors:
            a.SetVisibility(bool(on))

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
        self._legend = HuLegend(_band_lut_rgb, _HU_LO, _HU_HI)
        _leg_lbl = QLabel(t("Legend — groups / grayscale (W/L) / colour"))
        # Spatial smoothing of the colour boundaries (mm) — de-jaggs the band
        # edges (a weak Gaussian on the colour reslice). 0 = crisp/blocky.
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

    # -------------------------------------------------------- helpers
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
    #: emitted when a committed measure is un-committed ("resumed" to keep
    #: extending its trace) — carries the measure id so the shell drops that
    #: stale history entry; re-committing then re-adds it fresh.
    measurement_removed = pyqtSignal(int)
    #: emitted when the user clicks "Measure History"
    history_requested = pyqtSignal()
    #: HU colour map edited here — shell persists it and mirrors it onto every
    #: other CT pane so the colour map is global. Args:
    #: (bands, opacity, smooth_mm).
    colormap_changed = pyqtSignal(object, float, float)
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
        self._undo_clear()                   # unified Ctrl+Z / Ctrl+Y state
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
        # HU colour map is GLOBAL + persisted: load the shared bands so every
        # CT pane (and a fresh restart) starts from the same colour map.
        _cm = settings.load_ct_colormap()
        self._bands = [dict(b) for b in _cm["bands"]]
        self._opacity = float(_cm["opacity"])
        #: Spatial colour-smoothing strength (mm): a weak Gaussian on the colour
        #: reslice so band boundaries are smooth curves, not the voxel-grid
        #: staircase. Global + persisted; 0 = crisp.
        self._cmap_smooth_mm = float(_cm.get("smooth_mm", 0.4))
        self._cmap_dlg = None
        # LV EF measurement state (Phase 1): None = off; else
        # {"model": LVModel, "phase": "contour", "plane_idx": int,
        # "target": "endo"|"epi", "plane_done": bool, "prev_side": str}.
        self._lv = None
        # Common valve planes (MV / AoV) shared by Endo/Epi/Blood as the LV base.
        # Each is (centre_xyz, normal_xyz, radius) in volume mm, or None.
        self._lv_valves = {"mitral": None, "aortic": None}
        # Whether each valve's ellipse is currently SHOWN (its button toggles it
        # once the plane is set, so it can be hidden while tracing Endo/Epi).
        self._lv_valve_shown = {"mitral": True, "aortic": True}
        self._lvv = None                 # LV blood-pool volume (LVEF) session
        self._lvv_epi_surf = None        # Epi surface captured from contour mode
        self._lvv_epi_apex = None
        self._lvv_epi_model_dict = None  # epi model (for LV Vol save/load)
        self._lvv_mask_vol = None        # measured-region 0/1 vtkImageData
        self._lvv_mask_on = False        # red measured-region overlay visible
        self._meas_on = False
        self._meas_type = None          # line|polyline|ellipse|polygon
        self._measures = {"A": [], "B": []}   # finalized {id,type,pts}
        self._meas_seq = 0              # type-independent running number
        self._draft = None              # {type, pane, pts} in progress
        self._edit = None               # {key, mi, vi} handle drag
        self._meas_hover_handle = None  # {key, mi, vi, ca} handle under cursor
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
        self._lv_line_drag = None            # "level"/"meridian" while grabbing
        #                                      the SAX level / centre line
        self._lv_line_hi = {"A": False, "B": False}   # line thickened on hover
        self._lv_apex_drag = None            # "endo"/"epi" while dragging an apex
        #                                      marker (None = not dragging)
        self._lv_apex_hot = False            # apex glows: cursor in range while
        #                                      tracing (cleared on point confirm)
        # Drawn crosshair rotation per pane (deg); follows the cursor
        # while CrossLine-dragging so the crosshair tracks the drag.
        self._cross_ang = {"A": 0.0, "B": 0.0}
        # ▲ apex-marker side per pane (±1). Flips on each Flip-H / Flip-V so the
        # ▲ mirrors WITH the image (the crosshair is drawn in output coords, so
        # a frame mirror doesn't auto-flip the directed ▲). Rotations don't
        # change it. Reset to +1 whenever the default frames are rebuilt.
        self._apex_flip = {"A": 1.0, "B": 1.0}

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
        lay.addWidget(self._build_lv_bar())

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
        # NB: A / F are app-wide ApplicationShortcuts (cine/series nav, see
        # MainWindow._nav_active). Registering a viewer-level A/F QShortcut here
        # collides with those → Qt sees two matching shortcuts → "ambiguous" →
        # NEITHER fires. So LV plane-stepping on A/F is routed the other way:
        # _nav_active calls lv_nav_key() below FIRST when a CT pane is active.
        # Ctrl+Z = unified undo, Ctrl+Y = redo — covering EVERY view action
        # (Rt90/Lt90/Flip-H/Flip-V, Spin+, centreline move·rotate, Zoom/Move/
        # Paging/Thick, recentre) and LV border edits. Viewer-scoped so a focused
        # child can't swallow them.
        sc_undo = QShortcut(QKeySequence.StandardKey.Undo, self)
        sc_undo.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_undo.activated.connect(self._undo_last)
        sc_redo = QShortcut(QKeySequence.StandardKey.Redo, self)
        sc_redo.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_redo.activated.connect(self._redo_last)
        # Ctrl+Y is the Windows redo; also bind Ctrl+Shift+Z (the common
        # alternative) so both habits work.
        sc_redo2 = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        sc_redo2.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_redo2.activated.connect(self._redo_last)
        self._update_active_frames()

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

    def _build_lv_bar(self) -> QWidget:
        """Unified LV bar (2 rows, always visible below the image). Row 1: the
        LV: caption, the Endo/Epi/Blood SUB-MODE selector, then ONLY the active
        sub-mode's operation group (Endo/Epi tracing controls, or Blood volume
        controls). Row 2: the shared file/measure controls (Calc Vol / Save /
        Load / STL / Clear / Exit), showing the active sub-mode's set. Survives
        'Max Image'. Endo/Epi/Blood ENTER their sub-mode; Exit leaves LV."""
        self._lv_wrap = QWidget()
        self._lv_wrap._mdv_keep_on_max = True
        outer = QVBoxLayout(self._lv_wrap)
        outer.setContentsMargins(8, 2, 8, 2)
        outer.setSpacing(2)
        row1 = QHBoxLayout(); row1.setSpacing(4)
        row2 = QHBoxLayout(); row2.setSpacing(4)
        outer.addLayout(row1)
        outer.addLayout(row2)

        cap = QLabel(t("LV:"))
        f = cap.font(); f.setBold(True); cap.setFont(f)
        row1.addWidget(cap)
        # Internal mode flag (hidden) — kept for the many _lv_btn.isChecked()
        # state checks throughout the viewer.
        self._lv_btn = FitButton(t("Trace"))
        self._lv_btn.setCheckable(True)
        self._lv_btn.setVisible(False)

        # ---- Common valve planes (MV / AoV): set ONCE at the start, shared by
        # every sub-mode as the LV base — Endo/Epi base cut + wall normalisation,
        # and the Blood region's basal bound. Saved to their own MVLv.json /
        # AoVLv.json (single source, one per 3DCT phase). ----
        self._lv_mv_btn = FitButton(t("MV plane"))
        self._lv_mv_btn.setHelpToolTip(
            t("Draw an Ellipse on the mitral annulus (Measure→Ellipse), then "
              "press this to set the COMMON MV plane (shared by Endo/Epi/Blood)"))
        self._lv_mv_btn.clicked.connect(
            lambda: self._lv_capture_valve_common("mitral"))
        row1.addWidget(self._lv_mv_btn)
        self._lv_aov_btn = FitButton(t("AoV plane"))
        self._lv_aov_btn.setHelpToolTip(
            t("Draw an Ellipse on the aortic annulus (Measure→Ellipse), then "
              "press this to set the COMMON AoV plane (shared by Endo/Epi/Blood)"))
        self._lv_aov_btn.clicked.connect(
            lambda: self._lv_capture_valve_common("aortic"))
        row1.addWidget(self._lv_aov_btn)
        row1.addSpacing(8)

        # ---- Sub-mode selector: Endo / Epi / Blood (always visible) ----
        self._lv_endo_btn = FitButton(t("Endo"))
        self._lv_endo_btn.setHelpToolTip(
            t("Endo (lumen) pass — align its long-axis view, Set axis, then Trace"))
        self._lv_endo_btn.clicked.connect(lambda: self._lv_select_submode("endo"))
        row1.addWidget(self._lv_endo_btn)
        self._lv_epi_btn = FitButton(t("Epi"))
        self._lv_epi_btn.setHelpToolTip(
            t("Epi (myocardial) pass — align its long-axis view, Set axis, then "
              "Trace"))
        self._lv_epi_btn.clicked.connect(lambda: self._lv_select_submode("epi"))
        row1.addWidget(self._lv_epi_btn)
        self._lvv_start_btn = FitButton(t("Blood"))
        self._lvv_start_btn.setCheckable(True)
        self._lvv_start_btn.setHelpToolTip(
            t("Blood-pool volume sub-mode (needs a traced/loaded Epi border)"))
        self._lvv_start_btn.clicked.connect(lambda: self._lv_select_submode("blood"))
        row1.addWidget(self._lvv_start_btn)
        row1.addSpacing(8)

        # ================= Endo/Epi operation group (row 1) =================
        self._lv_grp_trace = QWidget()
        gt = QHBoxLayout(self._lv_grp_trace)
        gt.setContentsMargins(0, 0, 0, 0); gt.setSpacing(4)
        self._lv_setaxis_btn = FitButton(t("Set axis"))
        self._lv_setaxis_btn.setHelpToolTip(
            t("Use the current long-axis view as this pass's rotation axis"))
        self._lv_setaxis_btn.clicked.connect(self._lv_set_axis)
        gt.addWidget(self._lv_setaxis_btn)
        self._lv_trace_btn = FitButton(t("Trace"))
        self._lv_trace_btn.setHelpToolTip(
            t("Place this pass's apex (first click; Shift-click to adjust the "
              "view) then trace its border"))
        self._lv_trace_btn.clicked.connect(self._lv_start_trace)
        gt.addWidget(self._lv_trace_btn)
        self._lv_prev_btn = FitButton(t("◀ Prev plane (A)"))
        self._lv_prev_btn.setHelpToolTip(t("Previous long-axis plane"))
        self._lv_prev_btn.clicked.connect(lambda: self._lv_step_plane(-1))
        gt.addWidget(self._lv_prev_btn)
        self._lv_plane_lbl = QLabel("0/6")     # 0/6 until a pass is started
        self._lv_plane_lbl.setMinimumWidth(78)
        fl = self._lv_plane_lbl.font(); fl.setBold(True)
        self._lv_plane_lbl.setFont(fl)
        self._lv_plane_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gt.addWidget(self._lv_plane_lbl)
        self._lv_next_btn = FitButton(t("▶ Next plane (F)"))
        self._lv_next_btn.setHelpToolTip(t("Next long-axis plane"))
        self._lv_next_btn.clicked.connect(lambda: self._lv_step_plane(1))
        gt.addWidget(self._lv_next_btn)
        self._lv_sax_btn = FitButton(t("SAX"))
        self._lv_sax_btn.setCheckable(True)
        self._lv_sax_btn.setStyleSheet(
            "QPushButton:checked{background:#b8860b;color:white;}" + self._BTN_DIS)
        self._lv_sax_btn.setHelpToolTip(
            t("Short-axis view: show the endo/epi borders on cross-sections ⟂ "
              "the long axis (◀ ▶ scroll the level)"))
        self._lv_sax_btn.clicked.connect(self._lv_toggle_sax)
        gt.addWidget(self._lv_sax_btn)
        row1.addWidget(self._lv_grp_trace)

        # ================= Blood operation group (row 1) =================
        self._lv_grp_blood = QWidget()
        gb = QHBoxLayout(self._lv_grp_blood)
        gb.setContentsMargins(0, 0, 0, 0); gb.setSpacing(4)
        self._lvv_apex_btn = FitButton(t("Apex"))
        self._lvv_apex_btn.setHelpToolTip(
            t("Confirm the LV apex at the crosshair (move it there first)"))
        self._lvv_apex_btn.clicked.connect(self._lvv_confirm_apex)
        gb.addWidget(self._lvv_apex_btn)
        self._lvv_mv_btn = FitButton(t("MV plane"))
        self._lvv_mv_btn.setHelpToolTip(
            t("Draw an Ellipse on the mitral annulus (Measure→Ellipse), then "
              "press this to capture its plane"))
        self._lvv_mv_btn.clicked.connect(lambda: self._lvv_capture_valve("mitral"))
        gb.addWidget(self._lvv_mv_btn)
        self._lvv_aov_btn = FitButton(t("AoV plane"))
        self._lvv_aov_btn.setHelpToolTip(
            t("Draw an Ellipse on the aortic annulus (Measure→Ellipse), then "
              "press this to capture its plane"))
        self._lvv_aov_btn.clicked.connect(lambda: self._lvv_capture_valve("aortic"))
        gb.addWidget(self._lvv_aov_btn)
        self._lvv_thr_btn = FitButton(t("内腔ROI"))
        self._lvv_thr_btn.setHelpToolTip(
            t("Draw a Polygon inside the LV cavity (Measure→Polygon), then press "
              "this: it colours the ROI, seeds it, and sets the HU range from "
              "the pixels inside — adjust 下限/上限 below."))
        self._lvv_thr_btn.clicked.connect(self._lvv_capture_roi)
        gb.addWidget(self._lvv_thr_btn)
        self._lvv_lo_lbl = QLabel(t("下限"))
        self._lvv_lo_spin = QSpinBox()
        self._lvv_lo_spin.setRange(-1000, 4000)
        self._lvv_lo_spin.setSingleStep(10)
        self._lvv_lo_spin.setSuffix(" HU")
        self._lvv_lo_spin.setKeyboardTracking(False)
        self._lvv_hi_lbl = QLabel(t("上限"))
        self._lvv_hi_spin = QSpinBox()
        self._lvv_hi_spin.setRange(-1000, 4000)
        self._lvv_hi_spin.setSingleStep(10)
        self._lvv_hi_spin.setValue(3000)
        self._lvv_hi_spin.setSuffix(" HU")
        self._lvv_hi_spin.setKeyboardTracking(False)
        self._lvv_lo_spin.valueChanged.connect(
            lambda _v: self._lvv_update_highlight())
        self._lvv_hi_spin.valueChanged.connect(
            lambda _v: self._lvv_update_highlight())
        for _w in (self._lvv_lo_lbl, self._lvv_lo_spin,
                   self._lvv_hi_lbl, self._lvv_hi_spin):
            gb.addWidget(_w)
        self._lvv_hl_on = True
        self._lvv_hl_btn = FitButton(t("血流領域表示"))
        self._lvv_hl_btn.setCheckable(True)
        self._lvv_hl_btn.setChecked(True)
        self._lvv_hl_btn.setHelpToolTip(
            t("Tint every voxel whose HU is in the 下限–上限 range on both panes "
              "(in-range = blood); adjust 下限/上限 to optimise"))
        self._lvv_hl_btn.clicked.connect(self._lvv_toggle_highlight)
        gb.addWidget(self._lvv_hl_btn)
        self._lvv_mask_btn = FitButton(t("計測領域"))
        self._lvv_mask_btn.setCheckable(True)
        self._lvv_mask_btn.setChecked(True)
        self._lvv_mask_btn.setHelpToolTip(
            t("Show the measured LV blood region (red) — independent of 血流領域表示"))
        self._lvv_mask_btn.clicked.connect(self._lvv_toggle_mask)
        gb.addWidget(self._lvv_mask_btn)
        self._lvv_epi_show = False
        # Epi読み込み: ALWAYS pick an EpiLv.json (replace the in-memory Epi).
        self._lvv_epi_load_btn = FitButton(t("Epi読み込み"))
        self._lvv_epi_load_btn.setHelpToolTip(
            t("Load an EpiLv.json as the Epi surface bounding the Blood region "
              "(replaces the current Epi)"))
        self._lvv_epi_load_btn.clicked.connect(self._lvv_epi_load_click)
        gb.addWidget(self._lvv_epi_load_btn)
        # Epi表示: toggle the green Epi border; if none in memory, load one first.
        self._lvv_epi_btn = FitButton(t("Epi表示"))
        self._lvv_epi_btn.setCheckable(True)
        self._lvv_epi_btn.setHelpToolTip(
            t("Show the Epi border on both panes (green); loads an EpiLv.json "
              "first if none is in memory"))
        self._lvv_epi_btn.clicked.connect(self._lvv_toggle_epi)
        gb.addWidget(self._lvv_epi_btn)
        row1.addWidget(self._lv_grp_blood)
        row1.addStretch(1)

        # ================= Row 2: shared file / measure controls =========
        # Endo/Epi set: Calc Vol / Save / Load / STL / Clear / Exit.
        self._lv_grp_r2_trace = QWidget()
        r2t = QHBoxLayout(self._lv_grp_r2_trace)
        r2t.setContentsMargins(0, 0, 0, 0); r2t.setSpacing(4)
        self._lv_vol_btn = FitButton(t("Calc Vol"))
        self._lv_vol_btn.setStyleSheet(self._LV_STY["vol_todo"])   # grey until calc
        self._lv_vol_btn.setHelpToolTip(
            t("Compute the volume enclosed by this sub-mode's traced border"))
        self._lv_vol_btn.clicked.connect(self._lv_compute_volume)
        r2t.addWidget(self._lv_vol_btn)
        # Wall button: kept (referenced by _lv_sync_buttons) but NOT shown in the
        # bar — wall-thickness moves to the Tools「心機能」tool (planned).
        self._lv_wall_btn = FitButton(t("Wall"))
        self._lv_wall_btn.setCheckable(True)
        self._lv_wall_btn.setStyleSheet(
            "QPushButton{background:palette(button);border:2px solid #9b59b6;}"
            "QPushButton:checked{background:#8e44ad;color:white;"
            "border:2px solid #8e44ad;}" + self._BTN_DIS)
        self._lv_wall_btn.setHelpToolTip(
            t("Short-axis WALL THICKNESS colour map (Epi−Endo), on every level"))
        self._lv_wall_btn.clicked.connect(self._lv_toggle_wall)
        self._lv_wall_btn.setVisible(False)
        self._lv_save_btn = FitButton(t("Save"))
        self._lv_save_btn.setHelpToolTip(
            t("Save this sub-mode's border to a file"))
        self._lv_save_btn.clicked.connect(self._lv_save)
        r2t.addWidget(self._lv_save_btn)
        self._lv_load_btn = FitButton(t("Load"))
        self._lv_load_btn.setHelpToolTip(
            t("Load a previously-saved border and apply it"))
        self._lv_load_btn.clicked.connect(self._lv_load)
        r2t.addWidget(self._lv_load_btn)
        self._lv_stl_btn = FitButton(t("STL"))
        self._lv_stl_btn.setHelpToolTip(
            t("Export the reconstructed surface as STL (mm scale)"))
        self._lv_stl_btn.clicked.connect(self._lv_export_stl)
        r2t.addWidget(self._lv_stl_btn)
        self._lv_redo_btn = FitButton(t("Clear"))
        self._lv_redo_btn.setHelpToolTip(
            t("Discard all traced borders and start again from plane 1"))
        self._lv_redo_btn.clicked.connect(self._lv_clear_confirm)
        r2t.addWidget(self._lv_redo_btn)
        self._lv_exit_btn = FitButton(t("Exit"))
        self._lv_exit_btn.clicked.connect(self._lv_exit_all)
        r2t.addWidget(self._lv_exit_btn)
        row2.addWidget(self._lv_grp_r2_trace)

        # Blood set: Calc Vol / (mL) / Save / Load / Exit. (STL/Clear come with
        # the 3-file split in the next increment.)
        self._lv_grp_r2_blood = QWidget()
        r2b = QHBoxLayout(self._lv_grp_r2_blood)
        r2b.setContentsMargins(0, 0, 0, 0); r2b.setSpacing(4)
        self._lvv_calc_btn = FitButton(t("Calc Vol"))
        self._lvv_calc_btn.setHelpToolTip(
            t("Measure the blood volume inside the Epi surface, apex-side of "
              "MV/AoV, within the 下限–上限 HU range"))
        self._lvv_calc_btn.clicked.connect(lambda: self._lvv_calc())
        r2b.addWidget(self._lvv_calc_btn)
        self._lvv_vol_lbl = QLabel("--")
        fv = self._lvv_vol_lbl.font(); fv.setBold(True)
        self._lvv_vol_lbl.setFont(fv)
        self._lvv_vol_lbl.setMinimumWidth(90)
        r2b.addWidget(self._lvv_vol_lbl)
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
        row2.addWidget(self._lv_grp_r2_blood)

        # Valve-setup row-2 group: shown when NO sub-mode is active (the initial
        # "set the common MV / AoV planes" step). Save/Load the two valve files.
        self._lv_grp_r2_valves = QWidget()
        r2v = QHBoxLayout(self._lv_grp_r2_valves)
        r2v.setContentsMargins(0, 0, 0, 0); r2v.setSpacing(4)
        self._lv_mv_save_btn = FitButton(t("Save MV"))
        self._lv_mv_save_btn.setHelpToolTip(t("Save the MV plane to MVLv.json"))
        self._lv_mv_save_btn.clicked.connect(lambda: self._lv_save_valve("mitral"))
        r2v.addWidget(self._lv_mv_save_btn)
        self._lv_mv_load_btn = FitButton(t("Load MV"))
        self._lv_mv_load_btn.setHelpToolTip(t("Load an MVLv.json"))
        self._lv_mv_load_btn.clicked.connect(lambda: self._lv_load_valve("mitral"))
        r2v.addWidget(self._lv_mv_load_btn)
        self._lv_aov_save_btn = FitButton(t("Save AoV"))
        self._lv_aov_save_btn.setHelpToolTip(t("Save the AoV plane to AoVLv.json"))
        self._lv_aov_save_btn.clicked.connect(lambda: self._lv_save_valve("aortic"))
        r2v.addWidget(self._lv_aov_save_btn)
        self._lv_aov_load_btn = FitButton(t("Load AoV"))
        self._lv_aov_load_btn.setHelpToolTip(t("Load an AoVLv.json"))
        self._lv_aov_load_btn.clicked.connect(lambda: self._lv_load_valve("aortic"))
        r2v.addWidget(self._lv_aov_load_btn)
        for b in (self._lv_mv_save_btn, self._lv_mv_load_btn,
                  self._lv_aov_save_btn, self._lv_aov_load_btn):
            b.setStyleSheet(self._BTN_DIS)
        row2.addWidget(self._lv_grp_r2_valves)
        row2.addStretch(1)

        # Plain-button disabled-grey + the button lists the sync methods use.
        for b in (self._lv_prev_btn, self._lv_next_btn, self._lv_redo_btn,
                  self._lv_save_btn, self._lv_stl_btn, self._lv_load_btn,
                  self._lv_exit_btn):
            b.setStyleSheet(self._BTN_DIS)
        self._lv_bar_btns = [
            self._lv_setaxis_btn, self._lv_trace_btn, self._lv_prev_btn,
            self._lv_next_btn, self._lv_sax_btn, self._lv_vol_btn,
            self._lv_wall_btn, self._lv_redo_btn, self._lv_save_btn,
            self._lv_stl_btn, self._lv_exit_btn]
        self._lvv_ctrl_btns = [
            self._lvv_apex_btn, self._lvv_aov_btn, self._lvv_mv_btn,
            self._lvv_thr_btn, self._lvv_calc_btn, self._lvv_save_btn,
            self._lvv_exit_btn]
        for b in self._lvv_ctrl_btns:
            b.setStyleSheet(self._BTN_DIS)
        self._lvv_load_btn.setStyleSheet(self._BTN_DIS)
        self._lvv_start_btn.setStyleSheet(
            "QPushButton:checked{background:#2e8b57;color:white;}" + self._BTN_DIS)
        self._lv_sync_buttons()           # initial (not in LV mode) state
        self._lvv_sync()
        self._lv_update_submode_ui()
        return self._lv_wrap

    # ==================================================================
    # LV blood-pool volume (LVEF) — region = apex + aortic/mitral valve
    # planes; count contrast voxels (HU >= threshold) connected to a cavity
    # seed. The "Blood" sub-mode of the unified LV bar (buttons built in
    # _build_lv_bar above).
    # ==================================================================
    def _lv_current_submode(self):
        """Which sub-mode is active: 'blood', 'endo', 'epi', or None."""
        if self._lvv is not None:
            return "blood"
        if self._lv is not None:
            return self._lv.get("pass")           # 'endo' / 'epi' / None
        return None

    def _lv_mode_has_unsaved(self, mode) -> bool:
        """True if *mode* holds traced/measured data not saved since the last
        change (see the _lv_dirty / _lvv_dirty flags). Used to warn before a
        sub-mode switch discards it (sub-modes are mutually exclusive)."""
        if mode == "blood":
            return (self._lvv is not None
                    and (self._lvv.get("seed") is not None
                         or self._lvv.get("last_ml") is not None)
                    and getattr(self, "_lvv_dirty", False))
        # contour (endo/epi share one model/session)
        return (self._lv is not None
                and bool(self._lv["model"].endo_planes
                         or self._lv["model"].epi_planes)
                and getattr(self, "_lv_dirty", False))

    def _lv_confirm_drop(self, mode) -> bool:
        """If *mode* has unsaved data, ask before discarding it. Returns True to
        proceed (nothing unsaved, or the user confirmed), False to cancel."""
        from PyQt6.QtWidgets import QMessageBox
        if not self._lv_mode_has_unsaved(mode):
            return True
        return QMessageBox.question(
            self.window(), t("LV"),
            t("This sub-mode has unsaved data. Switch without saving?")) \
            == QMessageBox.StandardButton.Yes

    def _lv_select_submode(self, sm) -> None:
        """Enter/resume a sub-mode from the Endo/Epi/Blood selector. All three are
        INDEPENDENT (each its own session + its own EndoLv/EpiLv/BldLv file);
        switching ends the current one, warning first if it has unsaved work.
        Wall thickness / EF (which need both borders) are combined later in the
        Tools「心機能」tool from the saved files."""
        cur = self._lv_current_submode()
        # Re-click the ACTIVE sub-mode → DEACTIVATE it. Because the selector greys
        # the other two while one is active, this 2nd click is the ONLY way to
        # leave a sub-mode, so the "unsaved data will be lost" warning belongs
        # HERE (the moment it goes inactive) — not later when another button is
        # picked. On confirm the data is dropped so the sub-mode is fully clear.
        if sm == cur:
            if sm == "blood":
                if not self._lv_confirm_drop("blood"):
                    return
                self._lvv_toggle()                    # leave Blood (drops it)
            elif self._lv is not None and self._lv.get("sax") is None:
                m = self._lv["model"]
                if bool(m.endo_planes or m.epi_planes):
                    if not self._lv_confirm_drop("contour"):
                        return
                    self._lv_reset_contour_empty()    # drop the pass's data
                else:
                    self._lv["pass"] = None           # nothing traced → just clear
                    self._lv_apply_target(None)
                    self._lv_sync_buttons()
            else:
                # In SAX, re-click ARMS this border for editing (existing flow).
                self._lv_select_pass(sm)
            self._lv_update_submode_ui()
            return
        # Switch to a DIFFERENT sub-mode. The current one is already inactive
        # (deactivated by its own 2nd click, warned there), so no warning here —
        # just start the new one. Any stray leftover is cleared without a prompt.
        if sm in ("endo", "epi"):
            if self._lvv is not None:                 # (defensive) leftover Blood
                self._lvv_clear_markers()
                self._lvv = None
                self._lvv_sync()
            elif self._lv is not None:
                m = self._lv["model"]
                other_planes = (m.endo_planes if sm == "epi"
                                else m.epi_planes)
                if other_planes:                      # (defensive) fresh session
                    self._lv_switch_pass_independent(sm)
                    self._lv_update_submode_ui()
                    return
            self._lv_select_pass(sm)                  # enter/arm this pass
        elif sm == "blood":
            self._lvv_toggle()                        # start Blood (needs Epi)
        self._lv_update_submode_ui()

    def _lv_exit_all(self) -> None:
        """Exit the whole LV mode (both the contour and the Blood sub-modes)."""
        if self._lvv is not None:
            if not self._lv_confirm_drop("blood"):   # warn on unsaved Blood
                return
            self._lvv_clear_markers()
            self._lvv = None
            self._lvv_sync()
        if self._lv is not None:
            self._lv_exit_confirm()                  # has its own confirm
        self._lv_update_submode_ui()

    def _lv_reset_contour_empty(self) -> None:
        """Drop the current pass's model + on-screen borders and return to the
        no-pass state (still in LV mode). Used when a pass is DEACTIVATED (its
        data is discarded then) and before starting a fresh independent pass."""
        from multi_dicomviewer.core.lv_measure import LVModel
        lv = self._lv
        lv["model"] = LVModel(n_planes=lv["model"].n_planes)
        for k in ("A", "B"):
            self._measures[k] = [mm for mm in self._measures[k]
                                 if mm.get("_lv") is None]
        lv["pass"] = None
        lv["sax"] = None
        lv["sax_edit"] = None
        lv["plane_idx"] = 0
        lv["phase"] = "align"
        lv["fitted"] = False
        self._lv_dirty = False
        self._lv_result_lines = []
        if getattr(self, "_lv_sax_btn", None) is not None:
            self._lv_sax_btn.setChecked(False)
        self._lv_apply_target(None)
        self._lv_reset_undo()
        self._lv_sync_buttons()
        self._redraw_all_lv()

    def _lv_switch_pass_independent(self, sm) -> None:
        """Endo and Epi are independent sub-modes — one border per session, each
        saved to its own EndoLv/EpiLv file (wall thickness / EF are combined
        later in the 心機能 tool). Reset to an empty model, then enter the new
        pass's Align."""
        self._lv_reset_contour_empty()
        self._lv_select_pass(sm)                      # → fresh Align for this pass

    def _lv_update_submode_ui(self) -> None:
        """Show ONLY the active sub-mode's operation group (row 1) and file
        controls (row 2): Endo/Epi share the trace group, Blood has its own.
        The within-group grey-out (step availability + existing rules) still
        comes from _lv_sync_buttons / _lvv_sync."""
        if not hasattr(self, "_lv_grp_trace"):
            return
        sm = self._lv_current_submode()               # 'endo'/'epi'/'blood'/None
        endoepi = sm in ("endo", "epi")
        blood = sm == "blood"

        # Idempotent show/hide: only toggle a group's visibility when it actually
        # changes. Calling setVisible every sync (many per view manipulation)
        # churned the layout and could momentarily drop the trace buttons
        # mid-align (reported: Set axis vanished while orienting the view).
        def _vis(w, on):
            if w.isVisible() != on:
                w.setVisible(on)
        _vis(self._lv_grp_trace, endoepi)
        _vis(self._lv_grp_blood, blood)
        _vis(self._lv_grp_r2_trace, endoepi)
        _vis(self._lv_grp_r2_blood, blood)
        # Valve-setup file controls (row 2) show when NO sub-mode is active — the
        # initial "set the common MV / AoV planes" step. The MV/AoV capture
        # buttons (row 1) stay visible always.
        if getattr(self, "_lv_grp_r2_valves", None) is not None:
            _vis(self._lv_grp_r2_valves, sm is None)
        self._lv_update_valve_buttons()
        # Sub-mode selector: once one is chosen, grey the other two (only the
        # active one stays clickable — re-click it to deselect and bring the
        # others back). All three live when nothing is selected. Runs AFTER
        # _lv_sync_buttons / _lvv_sync (which enable them), so this wins.
        # EXCEPTION — in SAX you may switch which traced border you edit, so keep
        # Endo/Epi clickable there (whichever has a border); Blood stays greyed.
        if self._lv is not None and self._lv.get("sax") is not None:
            m = self._lv["model"]
            self._lv_endo_btn.setEnabled(
                m.endo_axis is not None and len(m.endo_contours) >= 3)
            self._lv_epi_btn.setEnabled(
                m.epi_axis is not None and len(m.epi_contours) >= 3)
            self._lvv_start_btn.setEnabled(False)
        else:
            self._lv_endo_btn.setEnabled(sm in (None, "endo"))
            self._lv_epi_btn.setEnabled(sm in (None, "epi"))
            self._lvv_start_btn.setEnabled(sm in (None, "blood"))
        # Keep the Blood selector's checked look in step even if it was clicked
        # while already active (the checkable button toggles itself on click).
        if self._lvv_start_btn.isChecked() != blood:
            self._lvv_start_btn.setChecked(blood)

    def _lvv_sync(self) -> None:
        on = self._lvv is not None
        self._lvv_start_btn.setChecked(on)
        g = (lambda k: on and self._lvv.get(k) is not None)
        apex_done, aov_done = g("apex"), g("aortic")
        mv_done, roi_done = g("mitral"), g("seed")
        # Wizard: enable only the next step (previous steps stay live for redo).
        # Order: Apex → MV → AoV → 内腔ROI.
        self._lvv_apex_btn.setEnabled(on)
        self._lvv_mv_btn.setEnabled(on and apex_done)
        self._lvv_aov_btn.setEnabled(on and mv_done)
        self._lvv_thr_btn.setEnabled(on and aov_done)
        self._lvv_calc_btn.setEnabled(on and roi_done)
        self._lvv_save_btn.setEnabled(on and roi_done)
        self._lvv_load_btn.setEnabled(self._image is not None)
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
        _done(self._lvv_thr_btn, roi_done, "#2e8b57")       # ROI green
        for w in (self._lvv_lo_lbl, self._lvv_lo_spin,
                  self._lvv_hi_lbl, self._lvv_hi_spin, self._lvv_hl_btn):
            w.setVisible(roi_done)
        # 血流領域表示 button colours cyan while active.
        if roi_done:
            self._lvv_style_toggle(self._lvv_hl_btn, "#40c0ff", "black")
        # Once a volume is measured the LV Vol計測 button acts as the red
        # measured-region on/off toggle (no recompute): light-red WHILE the red
        # overlay is showing, plain when hidden. The separate 計測領域 toggle
        # mirrors the same state.
        measured = on and self._lvv.get("last_ml") is not None
        showing = measured and getattr(self, "_lvv_mask_on", False)
        self._lvv_calc_btn.setStyleSheet(
            ("QPushButton{background:#ff5a5a;color:black;}" + self._BTN_DIS)
            if showing else self._BTN_DIS)
        self._lvv_mask_btn.setVisible(measured)
        if measured:
            self._lvv_style_toggle(self._lvv_mask_btn, "#ff5a5a", "black")
        # Epi buttons: both available throughout Blood mode. Epi読み込み always
        # picks a file; Epi表示 toggles the border (and loads one first if none
        # is in memory), so it stays enabled even before an Epi is loaded.
        self._lvv_epi_load_btn.setVisible(on)
        self._lvv_epi_btn.setVisible(on)
        self._lvv_epi_btn.setChecked(bool(getattr(self, "_lvv_epi_show", False)))
        self._lvv_style_toggle(self._lvv_epi_btn, "#50dc50", "black")
        self._lv_update_submode_ui()        # show only the active sub-mode's group

    def _lvv_prompt(self, text) -> None:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self.window(), t("LV Vol"), text)

    def _lvv_load_epi(self) -> bool:
        """Load an EpiLv.json as the Epi surface that bounds the Blood region.
        Returns True on success. Folder defaults to the 3DCT series folder; warns
        on a series-UID mismatch (same as the other loads)."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from multi_dicomviewer.core.lv_measure import LVModel
        import json
        if self._image is None:
            return False
        d = self._lv_series_dir() if hasattr(self, "_lv_series_dir") else ""
        path, _ = QFileDialog.getOpenFileName(
            self.window(), t("Load Epi data"), d,
            "Epi LV (*.EpiLv.json);;JSON (*.json)")
        if not path:
            return False
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            saved = (data.get("series") or {}).get("series_uid", "")
            cur = (self._lv_series_meta().get("series_uid", "")
                   if hasattr(self, "_lv_series_meta") else "")
            if saved and cur and saved != cur:
                if QMessageBox.question(
                        self.window(), t("LV Vol"),
                        t("This Epi file was saved for a DIFFERENT series — it "
                          "may not line up. Load anyway?")) \
                        != QMessageBox.StandardButton.Yes:
                    return False
            model = LVModel.from_dict(data)
            model.build()
            if model.epi is None:
                raise ValueError("no Epi surface in file")
            self._lvv_epi_surf = model.epi
            self._lvv_epi_apex = np.asarray(model.epi_axis.apex, float)
            self._lvv_epi_model_dict = data
            return True
        except Exception as exc:                        # noqa: BLE001
            import traceback
            QMessageBox.critical(self.window(), t("LV Vol (Epi load error)"),
                                 traceback.format_exc() or repr(exc))
            return False

    def _lvv_toggle(self, *args) -> None:
        from PyQt6.QtWidgets import QMessageBox
        try:
            if self._lvv is None:
                if self._image is None:
                    QMessageBox.information(
                        self.window(), t("LV Vol"),
                        t("Load a CT first (no volume in this pane)."))
                    return
                if self._lv is not None:          # leave contour LV mode first
                    self._lv_exit()
                if self._lvv_epi_surf is None:
                    # Blood needs an Epi surface as the outer bound. Offer to
                    # load an EpiLv.json or switch to Epi tracing.
                    box = QMessageBox(self.window())
                    box.setWindowTitle(t("LV Vol"))
                    box.setIcon(QMessageBox.Icon.Information)
                    box.setText(t("Blood mode needs Epi data."))
                    b_load = box.addButton(t("Load Epi data"),
                                           QMessageBox.ButtonRole.AcceptRole)
                    b_make = box.addButton(t("Create Epi data"),
                                           QMessageBox.ButtonRole.ActionRole)
                    box.addButton(QMessageBox.StandardButton.Cancel)
                    box.exec()
                    clicked = box.clickedButton()
                    if clicked is b_load:
                        if not self._lvv_load_epi():
                            return                # load cancelled/failed → abort
                        # Epi now in memory → fall through and start Blood.
                    elif clicked is b_make:
                        self._lv_select_submode("epi")   # go trace Epi instead
                        return
                    else:
                        return                     # Cancel
                self._lvv = {"apex": None, "aortic": None, "mitral": None,
                             "hu_lo": None, "hu_hi": None, "seed": None,
                             "step": "apex", "last_ml": None, "calc_sig": None}
                self._lvv_dirty = False              # fresh Blood session
                self._lvv_sync()
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
        self._lvv_add_marker("apex", P, "#ff4040")
        lvv["step"] = "mv"
        self._lvv_sync()
        self._lvv_prompt(
            t("Identify the mitral valve: draw an Ellipse on the MV annulus "
              "(Measure→Ellipse), then press 'MV plane'."))

    def _lvv_dbg(self, msg) -> None:
        """Append a diagnostic line to ~/.mdv_lvv_debug.log (survives a crash)."""
        try:
            import os
            with open(os.path.expanduser("~/.mdv_lvv_debug.log"), "a",
                      encoding="utf-8") as fh:
                fh.write(str(msg) + "\n")
        except Exception:                               # noqa: BLE001
            pass

    def _lvv_hu_at(self, P) -> float:
        """Nearest-voxel HU at world point *P* (mm) in self._vol (indexed
        vol[z, y, x], world = index × spacing = self._dims)."""
        sx, sy, sz = self._dims
        nz, ny, nx = self._vol.shape
        ix = min(max(int(round(P[0] / sx)), 0), nx - 1)
        iy = min(max(int(round(P[1] / sy)), 0), ny - 1)
        iz = min(max(int(round(P[2] / sz)), 0), nz - 1)
        return float(self._vol[iz, iy, ix])

    def _lvv_update_highlight(self) -> None:
        """Tint the in-range (blood) voxels on both panes via the overlay LUT."""
        on = (self._lvv is not None and self._lvv.get("seed") is not None
              and getattr(self, "_lvv_hl_on", True))
        if on:
            lut = _lvv_highlight_lut(float(self._lvv_lo_spin.value()),
                                     float(self._lvv_hi_spin.value()))
        else:
            lut = _lvv_transparent_lut()
        for k in ("A", "B"):
            self.pane[k].colors_hl.SetLookupTable(lut)
            self.pane[k].colors_hl.Modified()
            self.pane[k].render()

    def _lvv_toggle_highlight(self, *args) -> None:
        self._lvv_hl_on = self._lvv_hl_btn.isChecked()
        self._lvv_style_toggle(self._lvv_hl_btn, "#40c0ff", "black")
        self._lvv_update_highlight()

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
            apex, seed = lvv["apex"], lvv["seed"]
            # Effective valves: common MV/AoV if set, else the Blood-wizard ones.
            aortic = self._lv_valves.get("aortic") or lvv.get("aortic")
            mitral = self._lv_valves.get("mitral") or lvv.get("mitral")
            if apex is None or seed is None or aortic is None or mitral is None:
                return None
            c_a, n_a = aortic[0], aortic[1]
            c_m, n_m = mitral[0], mitral[1]
        except (KeyError, TypeError, IndexError):
            return None
        lo = round(float(self._lvv_lo_spin.value()), 3)
        hi = round(float(self._lvv_hi_spin.value()), 3)
        return (rt(apex), rt(c_a), rt(n_a, 4), rt(c_m), rt(n_m, 4),
                rt(seed), lo, hi)

    def _lvv_show_epi(self, render=True) -> None:
        """Draw the Epi surface where it crosses each pane (green dots), so the
        operator can see the Epi border and judge coronary contamination."""
        on = (self._lvv is not None and self._lv is None
              and self._lvv_epi_surf is not None
              and getattr(self, "_lvv_epi_show", False))
        pts = self._lvv_epi_surf._all_ring_points() if on else None
        for key in ("A", "B"):
            p = self.pane[key]
            if pts is None:
                p.lvv_env_mapper.SetInputData(vtkPolyData())
            else:
                _u, _v, n = self._axes_for(key)
                n = np.asarray(n, float)
                o = np.asarray(self._pc[key], float)
                dist = (pts - o) @ n
                tol = 0.75 * max(self._dims)
                near = pts[np.abs(dist) <= tol]
                if len(near):
                    out = [self._world3d_to_out(key, P) for P in near]
                    p.lvv_env_mapper.SetInputData(
                        _lv_pts_pd(out, [(80, 220, 80)] * len(out), z=0.72))
                else:
                    p.lvv_env_mapper.SetInputData(vtkPolyData())
            if render:
                p.render()

    def _lvv_epi_load_click(self, *args) -> None:
        """Epi読み込み: always pick an EpiLv.json (replace the in-memory Epi). On
        success, show the border and refresh."""
        if self._lvv_load_epi():
            self._lvv_epi_show = True
            self._lvv_epi_btn.setChecked(True)
            self._lvv_style_toggle(self._lvv_epi_btn, "#50dc50", "black")
            self._lvv_show_epi()
            self._lvv_sync()

    def _lvv_toggle_epi(self, *args) -> None:
        # Epi表示: turning ON with no Epi in memory → load one first; if that is
        # cancelled/fails, leave the toggle off.
        if self._lvv_epi_btn.isChecked() and self._lvv_epi_surf is None:
            if not self._lvv_load_epi():
                self._lvv_epi_btn.setChecked(False)
                self._lvv_style_toggle(self._lvv_epi_btn, "#50dc50", "black")
                return
            self._lvv_sync()
        self._lvv_epi_show = self._lvv_epi_btn.isChecked()
        self._lvv_style_toggle(self._lvv_epi_btn, "#50dc50", "black")
        self._lvv_show_epi()

    def _lvv_style_toggle(self, btn, color, text="white") -> None:
        """Colour a checkable overlay button by its checked state."""
        if btn.isChecked():
            btn.setStyleSheet("QPushButton{background:%s;color:%s;}%s"
                              % (color, text, self._BTN_DIS))
        else:
            btn.setStyleSheet(self._BTN_DIS)

    def _lvv_update_mask(self) -> None:
        """Show/hide the measured-region (red) overlay on both panes."""
        on = (self._lvv_mask_vol is not None
              and getattr(self, "_lvv_mask_on", False))
        for k in ("A", "B"):
            self.pane[k].colors_mask.SetLookupTable(_lvv_mask_lut(on))
            self.pane[k].colors_mask.Modified()
            self.pane[k].render()

    def _lvv_toggle_mask(self, *args) -> None:
        self._lvv_mask_on = self._lvv_mask_btn.isChecked()
        self._lvv_style_toggle(self._lvv_mask_btn, "#ff5a5a", "black")
        self._lvv_update_mask()

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
            self._lvv_dirty = True         # ROI/seed set → unsaved Blood work
            lvv["hu_lo"] = lo
            lvv["hu_hi"] = hi
            m["color"] = "#40c0ff"                       # ROI tint
            m["_lvv"] = "roi"
            # Anchor to its 3-D anatomical position (re-projects with the view)
            # and draw at 50% so the ROI outline doesn't dominate the image.
            m["pts3d"] = [self._out_to_world3d(which, wx, wy)
                          for (wx, wy) in m["pts"]]
            m["transp"] = 50
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

    def _lvv_add_marker(self, tag, P3, color) -> None:
        """Add a 3-D-anchored marker (apex / seed) on BOTH panes. Stored with
        pts3d so it is re-projected onto the current plane on every redraw (stays
        fixed to the anatomy when the MPR is panned / recentred / rotated)."""
        P3 = np.asarray(P3, float)
        for which in ("A", "B"):
            self._measures[which] = [m for m in self._measures.get(which, [])
                                     if m.get("_lvv") != tag]
            wx, wy = self._world3d_to_out(which, P3)
            self._meas_seq += 1
            self._measures[which].append(
                {"id": self._meas_seq, "type": "point", "pts": [(wx, wy)],
                 "pts3d": [tuple(map(float, P3))], "color": color, "_lvv": tag})
            self._redraw_meas(which)

    def _lv_capture_valve_common(self, which) -> None:
        """Set the COMMON MV/AoV plane from the latest Ellipse (centre + the
        pane's normal). Shared by Endo/Epi/Blood; independent of any sub-mode."""
        from PyQt6.QtWidgets import QMessageBox
        if self._image is None:
            return
        # Newest FRESH (untagged) ellipse — a new Measure→Ellipse to (re)capture.
        m, key, best = None, None, -1
        for k in ("A", "B"):
            for cand in self._measures.get(k, []):
                if (cand.get("type") == "ellipse"
                        and cand.get("_lv_valve") is None
                        and cand.get("id", -1) > best):
                    best = cand.get("id", -1)
                    m, key = cand, k
        if m is None:
            # No fresh ellipse. If this valve is already set, the button just
            # toggles its ellipse's visibility (hide it while tracing Endo/Epi).
            if self._lv_valves.get(which) is not None:
                self._lv_toggle_valve_visibility(which)
                return
            QMessageBox.information(
                self.window(), t("LV"),
                t("Draw an Ellipse on the {v} annulus first (Measure→Ellipse).")
                .format(v=t("aortic") if which == "aortic" else t("mitral")))
            return
        # A fresh ellipse → (re)capture this valve; make it shown.
        self._lv_valve_shown[which] = True
        # Drop any previous ellipse for this valve so only the newest remains.
        for k in ("A", "B"):
            self._measures[k] = [mm for mm in self._measures.get(k, [])
                                 if mm.get("_lv_valve") != which]
        cx, cy = self._shape_center(m)
        center = np.asarray(self._out_to_world3d(key, cx, cy), float)
        _u, _v, n = self._axes_for(key)
        _ecx, _ecy, ea, eb = self._ellipse_cab(m)
        radius = float(max(ea, eb))
        self._lv_valves[which] = (center, np.asarray(n, float), radius)
        m["color"] = "#ffd24d" if which == "aortic" else "#4dd0ff"
        m["_lv_valve"] = which
        # Anchor the ellipse to its 3-D anatomical position (re-projects as the
        # view moves) and draw it at 50% so it doesn't dominate the image.
        m["pts3d"] = [self._out_to_world3d(key, wx, wy) for (wx, wy) in m["pts"]]
        m["transp"] = 50
        self._redraw_meas(key)
        if self._meas_on:                       # so the next click doesn't draw
            self._meas_btn.setChecked(False)
            self._toggle_measure()
        self._lv_update_valve_buttons()

    def _lv_toggle_valve_visibility(self, which) -> None:
        """Show/hide this valve's ellipse (button toggle once the plane is set),
        so MV/AoV can be hidden while tracing Endo/Epi. The valve plane geometry
        is unaffected."""
        shown = not self._lv_valve_shown.get(which, True)
        self._lv_valve_shown[which] = shown
        for k in ("A", "B"):
            for mm in self._measures.get(k, []):
                if mm.get("_lv_valve") == which:
                    mm["hidden"] = not shown
            self._redraw_meas(k)
        self._lv_update_valve_buttons()

    def _lv_update_valve_buttons(self) -> None:
        """MV / AoV buttons: plain when unset; solid blue/amber when set AND
        shown; a coloured outline when set but hidden (toggled off for tracing)."""
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
        """Save the common MV or AoV plane to its own MVLv.json / AoVLv.json."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import json
        import os
        v = self._lv_valves.get(which)
        if v is None:
            QMessageBox.information(
                self.window(), t("LV"),
                t("Set the {w} plane first.").format(
                    w="MV" if which == "mitral" else "AoV"))
            return
        c, n, r = v
        data = {"type": "valve", "valve": which,
                "series": (self._lv_series_meta()
                           if hasattr(self, "_lv_series_meta") else {}),
                "c": list(map(float, c)), "n": list(map(float, n)),
                "r": float(r)}
        suffix = ".MVLv.json" if which == "mitral" else ".AoVLv.json"
        d = self._lv_series_dir() if hasattr(self, "_lv_series_dir") else ""
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
        self._unlink_case_variant(path)
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
        """Load an MVLv.json / AoVLv.json into the common valve planes."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import json
        if self._image is None:
            return
        flt = (("MV plane (*.MVLv.json)" if which == "mitral"
                else "AoV plane (*.AoVLv.json)") + ";;JSON (*.json)")
        d = self._lv_series_dir() if hasattr(self, "_lv_series_dir") else ""
        path, _ = QFileDialog.getOpenFileName(
            self.window(), t("Load valve plane"), d, flt)
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            saved = (data.get("series") or {}).get("series_uid", "")
            cur = (self._lv_series_meta().get("series_uid", "")
                   if hasattr(self, "_lv_series_meta") else "")
            if saved and cur and saved != cur:
                if QMessageBox.question(
                        self.window(), t("LV"),
                        t("This valve file was saved for a DIFFERENT series — it "
                          "may not line up. Load anyway?")) \
                        != QMessageBox.StandardButton.Yes:
                    return
            self._lv_valves[which] = (np.asarray(data["c"], float),
                                      np.asarray(data["n"], float),
                                      float(data.get("r", 20.0)))
            self._lv_update_valve_buttons()
            QMessageBox.information(
                self.window(), t("LV"),
                t("Loaded the {w} plane.").format(
                    w="MV" if which == "mitral" else "AoV"))
        except Exception as exc:                        # noqa: BLE001
            import traceback
            QMessageBox.critical(self.window(), t("LV (valve load error)"),
                                 traceback.format_exc() or repr(exc))

    def _lvv_capture_valve(self, valve) -> None:
        """Capture the most-recent Ellipse on the active pane as this valve's
        plane (centre + the pane's current normal)."""
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
        m["pts3d"] = [self._out_to_world3d(which, wx, wy) for (wx, wy) in m["pts"]]
        m["transp"] = 50                     # anchored + 50% (not dominating)
        self._redraw_meas(which)
        # Turn Measure OFF so the user can navigate to the next landmark without
        # a click starting a new ellipse (they re-arm Measure→Ellipse for the
        # next valve).
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

    def _lvv_calc(self, then=None) -> None:
        from PyQt6.QtWidgets import QMessageBox
        try:
            lvv = self._lvv
            if lvv is None or self._vol is None:
                return
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
                self._lvv_mask_on = not self._lvv_mask_on
                self._lvv_mask_btn.setChecked(self._lvv_mask_on)
                self._lvv_update_mask()
                self._lvv_sync()
                return
            # Valve planes: prefer the COMMON MV/AoV (shared step); fall back to
            # any captured in the Blood wizard itself.
            av = self._lv_valves.get("aortic") or lvv.get("aortic")
            mv = self._lv_valves.get("mitral") or lvv.get("mitral")
            miss = [n for n, val in ((t("apex"), lvv.get("apex")),
                    (t("aortic plane"), av), (t("mitral plane"), mv),
                    (t("ROI"), lvv.get("seed"))) if val is None]
            if miss:
                QMessageBox.information(
                    self.window(), t("LV Vol"),
                    t("Set these first: {m}").format(m=", ".join(miss)))
                return
            if self._lvv_epi_surf is None:
                QMessageBox.information(
                    self.window(), t("LV Vol"),
                    t("No Epi surface — press 'Epi読み込み' to load an EpiLv.json "
                      "(or trace Epi in the Epi sub-mode) first."))
                return
            from multi_dicomviewer.core.lv_bloodpool import bloodpool_volume_epi
            from PyQt6.QtWidgets import QProgressDialog
            seed = tuple(lvv["seed"])
            hu_lo = float(self._lvv_lo_spin.value())
            hu_hi = float(self._lvv_hi_spin.value())
            c_a, n_a, _r_a = av
            c_m, n_m, _r_m = mv
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
            self._lvv_dirty = True         # a fresh measure is unsaved
            lvv["calc_sig"] = self._lvv_signature()      # inputs at this measure
            self._lvv_vol_lbl.setText(t("{v:.1f} mL").format(v=res["volume_ml"]))
            # Build the measured-region mask volume (0/1) for the red overlay.
            comp = res.get("comp")
            if comp is not None:
                z0, z1, y0, y1, x0, x1 = res["bbox"]
                full = np.zeros(self._vol.shape, np.float32)
                full[z0:z1, y0:y1, x0:x1][np.asarray(comp, bool)] = 1.0
                sx, sy, sz = self._dims
                self._lvv_mask_vol = numpy_to_vtk_image(full, sx, sy, sz)
                for k in ("A", "B"):
                    self.pane[k].reslice_mask.SetInputData(self._lvv_mask_vol)
                self._lvv_mask_on = True
                self._lvv_mask_btn.setChecked(True)
                self._refresh()                          # reslice the mask now
                self._lvv_update_mask()
            # LV Vol計測 button → light red once measured.
            self._lvv_calc_btn.setStyleSheet(
                "QPushButton{background:#ff5a5a;color:black;}" + self._BTN_DIS)
            self._lvv_style_toggle(self._lvv_mask_btn, "#ff5a5a", "black")
            self._lvv_sync()
            self._lv_update_text()     # show "Blood-Volume:" in the result block
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
        if lvv is None or lvv.get("seed") is None:
            QMessageBox.information(self.window(), t("LV Vol"),
                                   t("Set the ROI first."))
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
        if lvv is None or lvv.get("seed") is None:
            return
        c_a, n_a, r_a = lvv["aortic"]
        c_m, n_m, r_m = lvv["mitral"]
        data = {
            "type": "lvvol",
            "series": (self._lv_series_meta()
                       if hasattr(self, "_lv_series_meta") else {}),
            "apex": list(map(float, lvv["apex"])),
            "seed": list(map(float, lvv["seed"])),
            "aortic": {"c": list(map(float, c_a)), "n": list(map(float, n_a)),
                       "r": float(r_a)},
            "mitral": {"c": list(map(float, c_m)), "n": list(map(float, n_m)),
                       "r": float(r_m)},
            "hu_lo": float(self._lvv_lo_spin.value()),
            "hu_hi": float(self._lvv_hi_spin.value()),
            "volume_ml": (None if lvv.get("last_ml") is None
                          else float(lvv["last_ml"])),
        }
        # The Epi surface is NOT embedded — it lives in its own EpiLv.json and is
        # loaded via "Epi読み込み"/"Epi表示". Record the Epi source's series (if
        # known) so a mismatch can be flagged, but the geometry stays single-
        # source in the EpiLv file (no drift between two copies).
        em = getattr(self, "_lvv_epi_model_dict", None)
        if isinstance(em, dict) and em.get("series"):
            data["epi_series"] = em.get("series")
        d = self._lv_series_dir() if hasattr(self, "_lv_series_dir") else ""
        # Auto name "名前;日付_Se番号.BldLv.json" (Blood sub-mode file).
        stem = (self._lv_default_stem() if hasattr(self, "_lv_default_stem")
                else "BldLv")
        name = stem + ".BldLv.json"
        default = os.path.join(d, name) if d else name
        path, _ = QFileDialog.getSaveFileName(
            self.window(), t("Save LV Vol"), default,
            "Blood LV (*.BldLv.json);;JSON (*.json)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".BldLv.json"
        self._unlink_case_variant(path)      # force BldLv exact casing
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as exc:                        # noqa: BLE001
            QMessageBox.warning(self.window(), t("LV Vol"),
                                t("Save failed: {err}", err=str(exc)))
            return
        self._lvv_dirty = False              # Blood saved → no unsaved-switch warn
        QMessageBox.information(self.window(), t("LV Vol"),
                               t("Saved: {p}", p=os.path.basename(path)))

    def _lvv_load(self) -> None:
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import json
        if self._image is None:
            return
        d = self._lv_series_dir() if hasattr(self, "_lv_series_dir") else ""
        path, _ = QFileDialog.getOpenFileName(
            self.window(), t("Load LV Vol"), d,
            "Blood LV (*.BldLv.json);;JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
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
                          "landmarks may not line up. Load anyway?")) \
                        != QMessageBox.StandardButton.Yes:
                    return
            # Contour LV and LV Vol are mutually exclusive: leave contour LV first
            # so both modes can't be active at once. _lv_exit stashes the contour
            # Epi into _lvv_epi_surf, which the Blood measure then uses.
            if self._lv is not None:
                self._lv_exit()
            # Drop any prior LV Vol overlay/markers so a reload doesn't stack.
            if self._lvv is not None:
                self._lvv_clear_markers()
            # BldLv.json no longer embeds the Epi — the Epi lives in its own
            # EpiLv.json (single source). Keep whatever Epi is already in memory
            # (from an Epi trace or a previous Epi読み込み); if there is none, the
            # user must load one via Epi読み込み before Calc Vol can run.
            if self._lvv is None:
                self._lvv = {"apex": None, "aortic": None, "mitral": None,
                             "hu_lo": None, "hu_hi": None, "seed": None,
                             "step": "apex", "last_ml": None, "calc_sig": None}
            self._lvv_dirty = False              # loaded = matches the file
            lvv = self._lvv
            lvv["apex"] = np.asarray(data["apex"], float)
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
            for spin, v in ((self._lvv_lo_spin, data["hu_lo"]),
                            (self._lvv_hi_spin, data["hu_hi"])):
                spin.blockSignals(True)
                spin.setValue(int(round(float(v))))
                spin.blockSignals(False)
            self._lvv_add_marker("apex", lvv["apex"], "#ff4040")
            if lvv.get("last_ml") is not None:
                self._lvv_vol_lbl.setText(
                    t("{v:.1f} mL").format(v=float(lvv["last_ml"])))
            self._lvv_sync()
            self._lvv_update_highlight()
            self._lv_update_text()     # show "Blood-Volume:" in the result block
            # The Epi is NOT in this file. If none is in memory, tell the user to
            # load one (Epi読み込み) before Calc Vol; otherwise the current Epi is
            # used.
            if self._lvv_epi_surf is None:
                QMessageBox.information(
                    self.window(), t("LV Vol"),
                    t("Loaded the Blood landmarks. This file does not contain the "
                      "Epi border — press 'Epi読み込み' to load the matching "
                      "EpiLv.json, then Calc Vol."))
            else:
                QMessageBox.information(
                    self.window(), t("LV Vol"),
                    t("Loaded. The Epi currently in memory will be used — press "
                      "'Epi読み込み' to use a different EpiLv.json. Press Calc Vol "
                      "to (re)compute the volume."))
        except Exception as exc:                        # noqa: BLE001
            import traceback
            QMessageBox.critical(self.window(), t("LV Vol (load error)"),
                                 traceback.format_exc() or repr(exc))

    def _lvv_clear_markers(self) -> None:
        self._lvv_mask_vol = None
        self._lvv_mask_on = False
        self._lvv_epi_show = False
        for k in ("A", "B"):
            self._measures[k] = [m for m in self._measures.get(k, [])
                                 if m.get("_lvv") is None]
            self.pane[k].lvv_env_mapper.SetInputData(vtkPolyData())
            self.pane[k].colors_hl.SetLookupTable(_lvv_transparent_lut())
            self.pane[k].colors_hl.Modified()
            self.pane[k].colors_mask.SetLookupTable(_lvv_mask_lut(False))
            self.pane[k].reslice_mask.SetInputData(_placeholder_image())
            self.pane[k].colors_mask.Modified()
            self._redraw_meas(k)

    def _lvv_deactivate(self) -> None:
        """Fully leave LV Vol mode: clear its overlays (Epi dots / blood tint /
        red region) and drop the mode. Called when entering contour LV so a
        loaded .lvvol dataset's Epi border can't linger on-screen there — its
        display belongs to the LV Vol bar's Epi button only."""
        if self._lvv is None:
            return
        self._lvv_clear_markers()
        self._lvv = None
        self._lvv_sync()

    def _active_pane(self) -> str:
        """The pane the user is working in (current side; default B)."""
        s = self.current_side() if hasattr(self, "current_side") else "Bi"
        return "A" if s == "Lt" else "B"

    #: Appended to EVERY LV/tool button style so a DISABLED button clearly greys
    #: out (a custom background otherwise overrides Qt's native disabled look, so
    #: an unavailable button would keep its colour). Matches the "Show Buttons"
    #: greyed look the user expects.
    _BTN_DIS = ("QPushButton:disabled{background:#e6e6e6;color:#a8a8a8;"
                "border:1px solid #d8d8d8;}")

    #: LV bar button styles by state (default = plain grey/black). Every entry
    #: carries the disabled-grey rule so unusable buttons read as greyed out.
    _LV_STY = {
        "endo": "QPushButton{background:#d32f2f;color:white;}" + _BTN_DIS,
        "epi": "QPushButton{background:#2e8b57;color:white;}" + _BTN_DIS,
        "setaxis": "QPushButton{background:#b8860b;color:white;}" + _BTN_DIS,
        "trace": "QPushButton{background:#c0392b;color:white;}" + _BTN_DIS,
        # CalcVol: the SAME native background as every other button BEFORE a
        # volume is computed (only a clear blue 2px outline sets it apart — a
        # hint it turns blue once computed); solid blue AFTER (valid result).
        "vol_todo": ("QPushButton{background:palette(button);color:black;"
                     "border:2px solid #1f77b4;}" + _BTN_DIS),
        "vol_done": "QPushButton{background:#1f77b4;color:white;}" + _BTN_DIS,
        # SAX/refine neutral (grey/black): the 4 trace buttons reset to this on
        # SAX entry; Endo/Epi re-colour only to show the armed edit target.
        "neutral": "QPushButton{background:#d0d0d0;color:black;}" + _BTN_DIS,
        # Unselected/default: native background when ENABLED, grey when disabled.
        "off": _BTN_DIS,
    }

    def _lv_set_bar_enabled(self, on: bool) -> None:
        """Enable/disable the non-entry LV controls (Endo/Epi/Load stay live)."""
        for b in getattr(self, "_lv_bar_btns", []):
            b.setEnabled(bool(on))

    def _lv_sync_buttons(self) -> None:
        """Colour + enable the LV bar by the current pass/phase (see the agreed
        flow in [[lv-apex-point-feature]]). Endo/Epi are red/green when selected
        or already set; Set axis turns dark-yellow once this pass's axis is set;
        Trace turns red once apex/tracing is armed."""
        lv = self._lv
        endo_btn, epi_btn = self._lv_endo_btn, self._lv_epi_btn
        setax, trace = self._lv_setaxis_btn, self._lv_trace_btn
        endo_btn.setEnabled(True)
        epi_btn.setEnabled(True)
        self._lv_load_btn.setEnabled(True)
        if lv is None:                                # not in LV mode
            for b in (endo_btn, epi_btn, setax, trace):
                b.setStyleSheet(self._LV_STY["off"])
            self._lv_vol_btn.setStyleSheet(self._LV_STY["vol_todo"])
            self._lv_set_bar_enabled(False)
            self._lv_exit_btn.setEnabled(False)
            self._refresh_tool_availability()        # restore WB reverse / tools
            self._lv_update_submode_ui()             # hide the trace group
            return
        ph = lv.get("phase")
        pas = lv.get("pass")
        sax_on = lv.get("sax") is not None
        # Exactly ONE pass is active — only its button is coloured (the rotation
        # axis is that pass's axis too, switched in _lv_select_pass). Set axis is
        # dark-yellow while its axis is in use (ready/apex/contour); Trace is red
        # while tracing (apex/contour); both go grey when undone.
        if sax_on:
            # SAX / refine mode: neutral bar. Endo/Epi colour only to show which
            # border is armed for editing (lv["sax_edit"]); Set axis stays off.
            # Trace is enabled once a border is armed → Endo/Epi + Trace LEAVES
            # SAX into that pass's long-axis trace (Endo restores its original
            # independent-axis trace; Epi resumes on its axis).
            ed = lv.get("sax_edit")
            endo_btn.setStyleSheet(self._LV_STY["endo"] if ed == "endo"
                                   else self._LV_STY["neutral"])
            epi_btn.setStyleSheet(self._LV_STY["epi"] if ed == "epi"
                                  else self._LV_STY["neutral"])
            setax.setStyleSheet(self._LV_STY["neutral"])
            setax.setEnabled(False)
            trace.setStyleSheet(self._LV_STY["neutral"])
            trace.setEnabled(ed in ("endo", "epi"))
        else:
            off = self._LV_STY["off"]
            endo_btn.setStyleSheet(self._LV_STY["endo"] if pas == "endo" else off)
            epi_btn.setStyleSheet(self._LV_STY["epi"] if pas == "epi" else off)
            setax.setStyleSheet(self._LV_STY["setaxis"]
                                if ph in ("ready", "apex", "contour") else off)
            trace.setStyleSheet(self._LV_STY["trace"]
                                if ph in ("apex", "contour") else off)
            # LIFO enable: you can only turn OFF the LAST button turned on.
            #   align → Set axis armed (set)      ready → Set axis (undo) + Trace
            #   apex/contour → Trace (undo) + SAX (on)
            # Set axis / Trace need a pass (Endo/Epi) chosen first — greyed until
            # then so the current step's usable buttons stand out.
            has_pass = pas in ("endo", "epi")
            setax.setEnabled(has_pass and ph in ("align", "ready"))
            trace.setEnabled(has_pass and ph in ("ready", "apex", "contour"))
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
        self._lv_update_submode_ui()        # show only the active sub-mode's group

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
        # Colour ON → a soft, muted (low-saturation) pale-yellow background.
        self._cmap_btn.setStyleSheet(
            "QPushButton:checked { background:#e3ddaa; color:#101010; }")
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
            b.setStyleSheet(self._BTN_DIS)          # clear grey when disabled
            b.clicked.connect(lambda _c, k=kind: self._2d_transform(k))
            self._t2d_btns.append(b)
            row2.addWidget(b)
        # Spin+ : snap the active pane's centreline to the nearest vertical /
        # horizontal (a 45° tie snaps clockwise). Works in 2-D and 3-D.
        self._spin_snap_btn = FitButton(t("Spin+"))
        self._spin_snap_btn.setStyleSheet(self._BTN_DIS)
        self._spin_snap_btn.setHelpToolTip(
            t("Snap the centreline to the nearest vertical/horizontal "
              "(45° snaps clockwise)"))
        self._spin_snap_btn.clicked.connect(self._spin_snap)
        row2.addWidget(self._spin_snap_btn)
        # Grayscale invert (black↔white negative) — right of Flip-V.
        self._invert_btn = FitButton(t("WB reverse"))
        self._invert_btn.setCheckable(True)
        self._invert_btn.setStyleSheet(self._BTN_DIS)   # clear grey when disabled
        self._invert_btn.setHelpToolTip(
            t("Invert grayscale (black↔white negative)"))
        self._invert_btn.clicked.connect(self._toggle_invert)
        row2.addWidget(self._invert_btn)
        # Undo / Redo buttons — set off to the right of WB reverse (same gap the
        # transforms have from WL). The shortcut in the label is platform-correct
        # (Ctrl on Windows/Linux, Cmd on macOS) so the hint matches the actual
        # key. Mouse-clickable → works over remote desktop where ⌘ can't be sent.
        import sys as _sys
        _mod = "Cmd" if _sys.platform == "darwin" else "Ctrl"
        row2.addSpacing(12)
        self._undo_btn = FitButton(f"Undo ({_mod}+Z)")
        self._undo_btn.setStyleSheet(self._BTN_DIS)
        self._undo_btn.setHelpToolTip(t("Undo the last action"))
        self._undo_btn.clicked.connect(self._undo_last)
        row2.addWidget(self._undo_btn)
        self._redo_btn = FitButton(f"Redo ({_mod}+Y)")
        self._redo_btn.setStyleSheet(self._BTN_DIS)
        self._redo_btn.setHelpToolTip(t("Redo the last undone action"))
        self._redo_btn.clicked.connect(self._redo_last)
        row2.addWidget(self._redo_btn)
        row2.addStretch(1)
        # (LV EF entry lives in the LV bar below as "Trace" — see _build_lv_bar.)

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
        locked = self._lv_axis_locked()          # LV axis set → no re-tilt tools
        for n, b in self._tool_btns.items():
            active = (n == getattr(self, "_tool", None))
            # Keep the tools clickable WHILE measuring so the user can pick which
            # tool a Shift+drag uses (the shortcut keys already switch them); the
            # dimmed-red styling below still signals "hold Shift to use it here".
            b.setEnabled(not ((is2d or locked) and n in _MPR_ONLY_TOOLS))
            if measuring:
                b.setStyleSheet("background:#7a4b46;color:#d0d0d0;" if active
                                else "color:#9a9a9a;")
            else:
                b.setStyleSheet("background:#c0392b;color:white;" if active
                                else "")
        # WB reverse (grayscale invert) and the slab-thickness spin are disabled
        # throughout LV mode (thin slices, fixed grayscale). It is ALSO disabled
        # in 3-D MPR per user request (not needed for 3DCT for now) — REVIVABLE:
        # drop the `and is2d` below to restore WB reverse in 3-D.
        if getattr(self, "_invert_btn", None) is not None:
            self._invert_btn.setEnabled(self._lv is None and is2d)
        # Rt90/Lt90/Flip-H/Flip-V all work in BOTH 2-D and 3-D (they transform the
        # active pane's frame), so keep all four enabled. Applied HERE too (not
        # only in _set_mode) so the state is right on every refresh / load path.
        for b in getattr(self, "_t2d_btns", []):
            b.setEnabled(True)
        # Slab(mm) is available in ALL 3-D LV sub-modes (Endo/Epi AND Blood) so
        # the operator can adjust each pane's slab — Endo/Epi default to left 0 /
        # right 5 mm (set on pass entry) but may be changed. Disabled only in 2-D.
        if getattr(self, "_slab_spin", None) is not None:
            self._slab_spin.setEnabled(self._mode == "3D")

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
            ("Point", "point"),
            ("Line", "line"), ("Polyline", "polyline"),
            ("Ellipse", "ellipse"), ("Polygon", "polygon"),
            ("Angle", "angle"),
        ):
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
            self._measure_hover_clear()
            # Arm Reset: after a measure/LV session the view is off its initial
            # position, so the first Reset click should restore it (without this
            # the 2-stage Reset only reset W/L until the user first dragged).
            self._view_initial = False
        self._refresh_tool_availability()   # grey/restore the interaction tools

    def _set_measure_type(self, key):
        self._meas_type = key
        self._draft = None
        for k, b in self._meas_btns.items():
            b.setChecked(k == key)
            b.setStyleSheet(
                "background:#1f77b4;color:white;" if k == key else ""
            )
        if key != "point":
            self._measure_hover_clear()
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
        dlg = CompareOptionsDialog(self._cmp_want_pa, self._cmp_want_thk,
                                   self.window())
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
        if t == "point":
            return list(m["pts"][:1])
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
        if t == "point":
            p3 = (m.get("pts3d") or [None])[0]
            hu = self._hu_at(p3) if p3 is not None else None
            return (f"#{m['id']} Point: HU {hu:.0f}" if hu is not None
                    else f"#{m['id']} Point: HU —")
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
                # Re-derive 2-D from the absolute 3-D points on the CURRENT plane
                # so the shape follows the anatomy when the plane is rotated /
                # paged. Ellipse/Polygon (MV/AoV valve rings, Blood ROI) are
                # anchored this way too — their handles/vertices carry pts3d.
                if (p3 and m["type"] in ("polyline", "point", "ellipse",
                                         "polygon")
                        and len(p3) == len(m["pts"])):
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
        off_dash_segs: list = []     # segments fully off-plane → 点線 (dotted)
        off_dash_cols: list = []
        _, _, _pn = self._axes_for(key)          # this plane's normal
        _po = self._pc[key]
        e = self._edit
        edit_mi = e["mi"] if (e and e["key"] == key) else -1
        edit_vi = e["vi"] if (e and e["key"] == key) else -1
        edit_ca = bool(e.get("ca")) if (e and e["key"] == key) else False
        # Hover highlight: the control point under the cursor turns green (the
        # same green as while dragging) so the user knows it will be grabbed
        # before pressing. Drawn through the edit-points actor.
        hh = self._meas_hover_handle
        hov_here = bool(hh and hh["key"] == key)
        hov_mi = hh["mi"] if hov_here else -1
        hov_vi = hh["vi"] if hov_here else -1
        hov_ca = bool(hh.get("ca")) if hov_here else False
        for mi, m in enumerate(self._measures[key]):
            # Hidden by "Hide/Show All Result" (global) or this measure's own
            # right-click Hide → skip its line, handles, axes and id label.
            if self._results_hidden or m.get("hidden"):
                continue
            # Point HU probe → a fixed-size "+" (two short segments, ~12 px) at
            # the point plus its #id; skip the generic outline/handle drawing.
            if m["type"] == "point":
                rgb = _hex_to_rgb(m.get("color"))
                a4 = transp_to_alpha(m.get("transp", 0))
                wx, wy = m["pts"][0]
                qx, qy = self._world_to_qt(key, wx, wy)
                lp = self._disp_to_world(key, qx - 6, qy)
                rp = self._disp_to_world(key, qx + 6, qy)
                up = self._disp_to_world(key, qx, qy - 6)
                dp = self._disp_to_world(key, qx, qy + 6)
                polylines.append([lp, rp])
                outline_colors.append((rgb[0], rgb[1], rgb[2], a4))
                polylines.append([up, dp])
                outline_colors.append((rgb[0], rgb[1], rgb[2], a4))
                labels.append((f"#{m['id']}", (wx, wy)))
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
            if m.get("_lv"):
                off_flag = None      # LV borders draw uniform (no off-plane cue)
            if off_flag is not None and not m.get("smooth"):
                # Draw the outline as PER-SEGMENT cells so the parts of the LINE
                # leaving the slice read differently. A trace is normally
                # un-splined; a smooth trace keeps one line alpha. Three states:
                #   both endpoints in-plane  → solid, full alpha
                #   one endpoint off-plane   → solid, 50% (the transition)
                #   both endpoints off-plane → 点線 (dotted), 50%
                # so at a glance you can tell which stretch is out of range.
                verts = list(m["pts"])
                half_a = max(1, a4 // 2)
                for i in range(len(verts) - 1):
                    o0, o1 = off_flag[i], off_flag[i + 1]
                    if o0 and o1:
                        off_dash_segs.append((verts[i], verts[i + 1]))
                        off_dash_cols.append((rgb[0], rgb[1], rgb[2], half_a))
                    else:
                        polylines.append([verts[i], verts[i + 1]])
                        seg_a = half_a if (o0 or o1) else a4
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
                elif not hov_ca and mi == hov_mi and vi == hov_vi:
                    edit_pts.append(q)            # hovered → green
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
                    # The marker being dragged (or hovered) turns green (like a
                    # vertex); the rest stay orange.
                    if mi == edit_mi and edit_ca and ci == edit_vi:
                        edit_pts.append(q)
                    elif hov_ca and mi == hov_mi and ci == hov_vi:
                        edit_pts.append(q)
                    else:
                        ca_pts.append(q)
        # The in-progress draft is drawn DASHED via its own mapper (below)
        # so it reads as not-yet-committed; on commit it re-renders solid
        # through the outline path above.
        draft_segs: list = []
        draft_col = _hex_to_rgb(None)
        d = self._draft
        if d and d["pane"] == key and d["pts"]:
            if d["type"] == "ellipse" and len(d["pts"]) >= 2:
                # Preview the oblique ellipse whose major axis is the drag.
                dpts = _ellipse_outline(
                    _ellipse_from_major(d["pts"][0], d["pts"][1]))
                draft_segs = list(zip(dpts, dpts[1:]))
            else:
                dpts = list(d["pts"])
                # LV Endo/Epi trace: preview the committed points with the SAME
                # centripetal Catmull-Rom the final border uses, in the target's
                # colour, so what you see IS what you'll get (place fewer points).
                lv_trace = (self._lv is not None
                            and self._lv.get("phase") == "contour"
                            and self._lv.get("target") in ("endo", "epi")
                            and d["type"] == "polyline")
                src = (_smooth_open(dpts)
                       if (lv_trace and len(dpts) >= 3) else dpts)
                draft_segs = list(zip(src, src[1:]))
                if lv_trace:
                    draft_col = ((211, 47, 47) if self._lv["target"] == "endo"
                                 else (46, 139, 87))
            handles += list(d["pts"])
        p.meas_mapper.SetInputData(
            _colored_multi_pd(polylines, outline_colors)
        )
        p.meas_draft_mapper.SetInputData(
            _colored_dashed_pd(draft_segs, [draft_col] * len(draft_segs))
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
        # Off-plane pseudo-centres → hollow 50% yellow rings (中抜き). Ring
        # radius tracks the pane's parallel scale so the on-screen size stays
        # constant across zoom (same trick the ▲ markers use).
        if off_pts:
            ps_off = p.ren.GetActiveCamera().GetParallelScale()
            # Size the hollow ring's OUTER edge to _CPR_RING_OUTER_PX — kept a
            # little smaller than the in-plane dot's radius (_MEAS_PT_PX/2) so
            # the in-range filled dot stands out more. The ring is a tube of
            # width 2.4*dpr, so its CENTRELINE radius must be (outer − tube_half)
            # for the tube's outer edge to land on the target. Convert that
            # screen radius to world mm via the parallel scale (= half the
            # viewport's world height) so it stays a constant on-screen size.
            h_phys = max(1.0, p.canvas.height() * dpr)
            ring_r_px = max(1.0, _CPR_RING_OUTER_PX - 1.2 * dpr)
            ring_r_world = ring_r_px * (2.0 * ps_off) / h_phys
            rings = _ring_polylines(off_pts, ring_r_world)
            p.meas_pts_off_mapper.SetInputData(
                _colored_multi_pd(rings, [(255, 217, 0, 128)] * len(rings)))
        else:
            p.meas_pts_off_mapper.SetInputData(vtkPolyData())
        # Fully off-plane segments → dotted (点線) at 50%.
        p.meas_off_dash_mapper.SetInputData(
            _colored_dashed_rgba_pd(off_dash_segs, off_dash_cols))
        self._rebuild_labels(p, labels)
        p.render()

    def _redraw_meas(self, key):
        self._recompute_compares(key)      # keep comparisons in sync on edit/delete
        self._redraw_geom(key)
        self._redraw_compare(key)
        p = self.pane[key]
        # "Hide/Show All Result" hides the measure result text too. In LV mode,
        # the border traces show NO length (they're LV contours, not rulers);
        # the LV volume result + tracing guidance are shown instead.
        meas_lines = ([] if self._results_hidden
                      else [self._metrics_text(key, m)
                            for m in self._measures[key]
                            if m.get("_lv") is None])
        lines = self._lv_status_lines() + meas_lines
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
            # Skip HIDDEN measures — their handles aren't drawn, so picking one
            # would grab an invisible point (another pass/plane's border
            # reprojected here) and shadow the visible point you meant to edit.
            if self._results_hidden or m.get("hidden"):
                continue
            for vi, q in enumerate(m["pts"]):
                qx, qy = self._world_to_qt(which, q[0], q[1])
                if math.hypot(qx - sx, qy - sy) < 12.0:
                    tag = m.get("_lv")
                    if lv_t not in ("endo", "epi"):
                        # No Endo/Epi armed → do NOT grab an LV border (endo & epi
                        # overlap in SAX, so an un-armed grab edits the WRONG one).
                        # Non-LV measures still grab as before.
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
        green drag colour. Covers ordinary vertices and Center-Angle points."""
        hit = self._pick_handle(which, sx, sy)
        if hit is not None:
            new = {"key": which, "mi": hit[0], "vi": hit[1], "ca": False}
        else:
            ca = self._pick_center_angle(which, sx, sy)
            new = ({"key": which, "mi": ca[0], "vi": ca[1], "ca": True}
                   if ca is not None else None)
        if new == self._meas_hover_handle:
            return
        old = self._meas_hover_handle
        self._meas_hover_handle = new
        for k in ({which} | ({old["key"]} if old else set())):
            self._redraw_geom(k)

    def _clear_hover_handle(self) -> None:
        if self._meas_hover_handle is None:
            return
        k = self._meas_hover_handle["key"]
        self._meas_hover_handle = None
        self._redraw_geom(k)

    def _lv_has_border(self, which, target) -> bool:
        """True if a captured Endo/Epi (*target*) border already exists for the
        current LV plane on *which* pane."""
        if self._lv is None:
            return False
        idx = self._lv.get("plane_idx", 0)
        for m in self._measures.get(which, []):
            tag = m.get("_lv")
            if tag is not None and tag[0] == idx and tag[1] == target:
                return True
        return False

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
        # STRONGLY honour the armed Endo/Epi selection: when a target is armed,
        # ONLY that border is a candidate — so right-click "Add point" (and any
        # outline pick) lands on the SELECTED border, never the nearer other one
        # (endo & epi overlap in SAX).
        lv_t = self._lv.get("target") if self._lv is not None else None
        best, bi = tol, None
        for mi, m in enumerate(self._measures[which]):
            if m.get("hidden"):
                continue
            if lv_t in ("endo", "epi"):
                tag = m.get("_lv")
                if tag is None or tag[1] != lv_t:
                    continue
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
        # While ACTIVELY drawing an LV border (Endo/Epi armed AND a draft already
        # in progress), a click ADDS the next point to that line — it never grabs
        # a nearby existing border's handle, so you can trace a 2nd line right
        # next to the first. When NOT mid-draw (no draft yet, or the line is
        # finished), a click on a handle still grabs it to MOVE the point.
        lv_target = self._lv.get("target") if self._lv is not None else None
        lv_drawing = (lv_target in ("endo", "epi")
                      and self._draft is not None
                      and self._draft.get("pane") == which
                      and len(self._draft.get("pts", [])) >= 1)
        hit = None if lv_drawing else self._pick_handle(which, sx, sy)
        # Starting a NEW LV border (target armed, no draft yet): don't let the
        # first click grab a DIFFERENT border's point (e.g. an Epi vertex when
        # you're starting an Endo right next to it) — start the new line instead.
        # A point of the SAME target's border is still grabbable (to edit it).
        if (hit is not None and lv_target in ("endo", "epi")
                and self._draft is None):
            tag = self._measures[which][hit[0]].get("_lv")
            if tag is None or tag[1] != lv_target:
                hit = None
        if hit is not None:
            self._edit = {"key": which, "mi": hit[0], "vi": hit[1]}
            if self._lv is not None and self._measures[which][hit[0]].get("_lv"):
                self._lv_push_undo(which, hit[0])   # snapshot before the drag
            self._redraw_geom(which)            # show the green dot now
            # Grabbing an LV border vertex → light the linked SAX crossing green
            # at once (not only after the first drag).
            if self._lv_sax_active() and self._measures[which][hit[0]].get("_lv"):
                self._redraw_lv(self._lv["sax_pane"])
                self.pane[self._lv["sax_pane"]].render()
            return True
        # A Center-Angle marker point can be dragged just like a polygon vertex.
        ca_hit = None if lv_drawing else self._pick_center_angle(which, sx, sy)
        if ca_hit is not None:
            self._edit = {"key": which, "mi": ca_hit[0], "vi": ca_hit[1],
                          "ca": True}
            self._redraw_geom(which)
            return True
        if not self._meas_type:
            return False
        w = self._disp_to_world(which, sx, sy)
        # Point HU probe: each click drops a persistent "+" and lists its HU in
        # the top-right result block (no draft; a single click finishes it).
        if self._meas_type == "point":
            try:
                P = self._out_to_world3d(which, *w)
            except Exception:
                P = None
            self._meas_seq += 1
            self._measures[which].append(
                {"id": self._meas_seq, "type": "point", "pts": [w],
                 "pts3d": [P] if P is not None else []})
            self._redraw_meas(which)
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
        # LV mode: a NEW polyline may start ONLY when Endo or Epi is active AND
        # that border isn't captured yet. So it's blocked when neither Endo nor
        # Epi is selected (a click does nothing), while the short-axis is shown
        # (confirm/edit only), or on a plane+target that already has a captured
        # border (edit that one instead). Point MOVE (handle grab above) and
        # ADD/DELETE (right-click) still work.
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
        # Capture the absolute 3-D position of each polyline vertex (a vessel
        # trace may cross rotated / paged planes). Only polylines in 3-D MPR —
        # the plane doesn't move in 2-D native mode. Optionally snap the depth
        # to the contrast lumen along the plane normal (the MIP hid the depth).
        if d["type"] == "polyline" and self._mode == "3D":
            P = self._out_to_world3d(which, *w)
            # NOT for LV borders: endo/epi aren't the vessel lumen, so snapping
            # would push the point off the traced plane (its 3-D depth jumps to
            # the bright blood pool), so the short-axis crossing lands off the
            # drawn border — the "point off the section line / balloon" bug.
            if self._snap_lumen and self._lv is None:
                _, _, nrm = self._axes_for(which)
                P = self._snap_to_lumen(P, nrm)
                d["pts"][-1] = self._world3d_to_out(which, P)   # keep 2-D in step
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

    def _measure_hover(self, which, sx, sy):
        """Point-probe hover: show the HU under the cursor, following it (the
        text sits just to the cursor's right). '—' outside the volume."""
        p = self.pane[which]
        if self._vol is None:
            p.hover_hu.SetInput("")
            p.render()
            return
        w = self._disp_to_world(which, sx, sy)
        try:
            hu = self._hu_at(self._out_to_world3d(which, *w))
        except Exception:
            hu = None
        p.hover_hu.SetInput("—" if hu is None else f"HU {hu:.0f}")
        dpr = max(1.0, p.canvas.devicePixelRatioF())
        dx = (sx + 12) * dpr
        dy = (p.canvas.height() - sy - 4) * dpr      # VTK display origin = bottom
        p.hover_hu.GetPositionCoordinate().SetValue(dx, dy)
        p.render()

    def _measure_hover_clear(self, which=None):
        for k in (("A", "B") if which is None else (which,)):
            pane = self.pane.get(k)
            if pane is not None:
                pane.hover_hu.SetInput("")
                pane.render()

    def _measure_drag(self, which, sx, sy):
        e = self._edit
        if not e:
            # Rubber-band the 2nd point of a Line being press-dragged.
            d = self._draft
            if (d is not None and d.get("_drag_new")
                    and d.get("type") == "line" and d.get("pane") == which):
                d["pts"][1] = self._disp_to_world(which, sx, sy)
                self._redraw_geom(which)
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
                # NOT for LV borders (see _measure_left): snapping an endo/epi
                # point to the lumen moves its 3-D depth off the drawn plane, so
                # the short-axis crossing jumps off the border (the balloon).
                if self._snap_lumen and self._lv is None:
                    _, _, nrm = self._axes_for(e["key"])
                    P = self._snap_to_lumen(P, nrm)
                    m["pts"][e["vi"]] = self._world3d_to_out(e["key"], P)
                m["pts3d"][e["vi"]] = P
            self._resnap_center_angle(m)
        self._recompute_compares(e["key"])     # live-update any comparison
        self._redraw_geom(e["key"])
        self._redraw_compare(e["key"])
        self._lv_live_recapture(e["key"], m)   # edited LV border → refresh SAX

    _UNDO_MAX = 80                         # Ctrl+Z / Ctrl+Y depth (unified)

    # ---- unified undo / redo stack (Ctrl+Z / Ctrl+Y) ----------------------
    # A single command list + cursor covering EVERY undoable action: image
    # transforms (Rt90/Lt90/Flip-H/Flip-V), Spin+, centreline move/rotate,
    # Zoom/Move/Paging/Thick, recentre, and LV border edits. Each command is a
    # {"undo": fn, "redo": fn} pair — undo() reverts one step, redo() re-applies
    # it. Most commands snapshot the whole VIEW state (frames, 2-D axes, centre,
    # per-pane reslice centre, cross angles, thickness, per-pane camera, CPR
    # transform, 2-D slice) BEFORE and AFTER the action, so undo/redo restore
    # exactly — no per-action drift (this is why LV-mode Rt90/Lt90 no longer
    # leaves the measurements shifted after Ctrl+Z). Cleared on series load / 2-D
    # ↔ 3-D switch / LV enter·exit·clear so no command restores a stale context.
    def _undo_clear(self) -> None:
        self._undo_cmds = []          # [{"undo": fn, "redo": fn}, …]
        self._undo_idx = 0            # cmds[:idx] are applied; cmds[idx:] are redoable
        self._lv_edit_before = None   # stashed LV-border snap during a drag
        self._gesture_snap = None     # view snap captured at a drag's mouse-press
        self._gesture_lv = None       # LV level/meridian snap at a drag's press
        self._gesture_moved = False   # did the current drag actually change anything
        self._lv_apex_snap = None     # LV geometry snap while dragging an apex
        self._draft_redo = []         # points popped from an in-progress trace
        self._lod_drag = False        # interactive low-res reslice while dragging

    def _undo_record(self, undo_fn, redo_fn) -> None:
        """Append one undo/redo command, dropping any redo-future first."""
        if not hasattr(self, "_undo_cmds"):
            self._undo_clear()
        if self._undo_idx < len(self._undo_cmds):
            del self._undo_cmds[self._undo_idx:]        # a new action forks history
        self._undo_cmds.append({"undo": undo_fn, "redo": redo_fn})
        if len(self._undo_cmds) > self._UNDO_MAX:
            del self._undo_cmds[:len(self._undo_cmds) - self._UNDO_MAX]
        self._undo_idx = len(self._undo_cmds)

    def _undo_view(self, before, after) -> None:
        """Record a view-state change from *before* to *after* snapshots."""
        if before is None or after is None:
            return
        self._undo_record(lambda b=before: self._view_restore(b),
                          lambda a=after: self._view_restore(a))

    def _gesture_begin(self) -> None:
        """A view-changing mouse drag is starting: snapshot the state so its
        WHOLE gesture (Zoom/Move/Rotate/Spin/Thick/Paging/WL, a centreline
        move·rotate, or an LV level/meridian drag) collapses into ONE Ctrl+Z
        step, committed on release."""
        self._gesture_snap = self._view_snapshot()
        self._gesture_lv = self._lv_scalar_snap()   # None outside LV
        self._gesture_moved = False
        self._lod_drag = True                       # reslice coarse until release

    def _gesture_commit(self) -> None:
        """Mouse released: record the drag as one undo step. The view part is
        recorded only if it actually moved; the LV level/meridian part only if
        it changed (an SAX line-drag moves no view state, so it lands here).
        Then drop interactive LOD and repaint once at full quality."""
        moved = getattr(self, "_gesture_moved", False)
        snap = getattr(self, "_gesture_snap", None)
        if snap is not None and moved:
            self._undo_view(snap, self._view_snapshot())
        self._lv_record_scalar(getattr(self, "_gesture_lv", None))
        self._gesture_snap = None
        self._gesture_lv = None
        self._gesture_moved = False
        if getattr(self, "_lod_drag", False):
            self._lod_drag = False
            if moved and self._image is not None:
                self._refresh()                     # final full-resolution pass

    # ---- LV short-axis LEVEL + shown MERIDIAN (navigation) undo -----------
    def _lv_scalar_snap(self):
        """Snapshot the SAX level + shown meridian (or None outside LV)."""
        if self._lv is None:
            return None
        return {"sax": self._lv.get("sax"), "plane_idx": self._lv.get("plane_idx")}

    def _lv_scalar_restore(self, snap) -> bool:
        if self._lv is None or snap is None:
            return False
        if snap.get("sax") is not None:
            self._lv["sax"] = snap["sax"]
        if snap.get("plane_idx") is not None:
            self._lv["plane_idx"] = snap["plane_idx"]
        # Re-derive the SAX view: _lv_show_sax_both re-slices BOTH the long-axis
        # meridian (plane_idx) and the short-axis level (sax) pane.
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
        """Snapshot both apex points + every LV border's 3-D points. Used for
        the apex drag, which also carries the border points that converged to
        the apex (so restoring only the apex would leave the borders shifted)."""
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
                    model.set_long_axis_contour(
                        angs[tag[0] % len(angs)], m["pts3d"], tag[1])
                    break
        self._lv_invalidate_volume()
        self._redraw_all_lv()
        for k in ("A", "B"):
            self._redraw_meas(k)
            self.pane[k].render()
        return True

    def _lv_record_geom(self, before) -> None:
        if before is None:
            return
        after = self._lv_geom_snap()
        if after is None or before == after:
            return
        self._undo_record(lambda b=before: self._lv_geom_restore(b),
                          lambda a=after: self._lv_geom_restore(a))

    # ---- LV border CREATION undo (whole traced border) --------------------
    def _lv_measure_copy(self, m):
        """Plain-data copy of an LV border measure (for the create-undo redo)."""
        rec = {"id": m.get("id"), "type": m.get("type", "polyline"),
               "pts": [tuple(map(float, p)) for p in m.get("pts", [])],
               "_lv": tuple(m["_lv"]),
               "color": m.get("color"),
               "smooth": bool(m.get("smooth", True))}
        if m.get("pts3d"):
            rec["pts3d"] = [list(map(float, P)) for P in m["pts3d"]]
        return rec

    def _lv_record_create(self, pane, m) -> None:
        """Record a freshly-captured border: Ctrl+Z removes the WHOLE border
        (after a confirm), Ctrl+Y recreates it."""
        if self._lv is None or m.get("_lv") is None:
            return
        snap = {"pane": pane, "tag": tuple(m["_lv"]),
                "rec": self._lv_measure_copy(m)}
        self._undo_record(lambda s=snap: self._lv_undo_create(s),
                          lambda s=snap: self._lv_redo_create(s))

    def _lv_undo_create(self, snap) -> bool:
        """Ctrl+Z on a created border → confirm, then delete it + its contour.
        Returns False if the user cancels (the border is kept)."""
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
                self._lv_drop_border(mm)          # clear the model contour too
                del self._measures[pane][i]
                break
        self._lv_invalidate_volume()
        self._redraw_meas(pane)
        if self._lv_sax_active():
            self._redraw_lv(self._lv["sax_pane"])
            self.pane[self._lv["sax_pane"]].render()
        return True

    def _lv_redo_create(self, snap) -> bool:
        """Ctrl+Y after undoing a create → re-add the border + its contour."""
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
        if self._lv_sax_active():
            self._redraw_lv(self._lv["sax_pane"])
            self.pane[self._lv["sax_pane"]].render()
        return True

    def _undo_last(self) -> None:
        """Ctrl+Z: while tracing, drop the last-placed point; otherwise step one
        command back. A command whose undo() returns False (e.g. the user
        cancelled the 'delete the whole border?' prompt) is left applied."""
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
        if res is False:                               # refused → keep it applied
            self._undo_idx += 1

    def _redo_last(self) -> None:
        """Ctrl+Y: while tracing, re-place a point dropped by Ctrl+Z; otherwise
        step one command forward (re-apply an undone action)."""
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

    def _draft_pop_point(self) -> None:
        """Ctrl+Z during a trace: remove the last-placed vertex (kept for redo)."""
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
        """Ctrl+Y during a trace: re-place the last vertex Ctrl+Z removed."""
        d = self._draft
        if not d or not getattr(self, "_draft_redo", None):
            return
        pt, p3 = self._draft_redo.pop()
        d["pts"].append(pt)
        if p3 is not None:
            d.setdefault("pts3d", []).append(p3)
        self._redraw_meas(d["pane"])

    def _view_snapshot(self) -> dict:
        """Capture the full display state so undo/redo can restore it exactly."""
        def cam(k):
            c = self.pane[k].ren.GetActiveCamera()
            return {"vu": tuple(c.GetViewUp()), "fp": tuple(c.GetFocalPoint()),
                    "pos": tuple(c.GetPosition()),
                    "ps": float(c.GetParallelScale())}
        return {
            "frame": {k: tuple(np.asarray(a, float).copy()
                               for a in self._frame[k]) for k in ("A", "B")},
            "axes2d": (None if self._axes2d is None else
                       (np.asarray(self._axes2d[0], float).copy(),
                        np.asarray(self._axes2d[1], float).copy())),
            "center": self._center.copy(),
            "pc": {k: self._pc[k].copy() for k in ("A", "B")},
            "cross_ang": dict(self._cross_ang),
            "apex_flip": dict(self._apex_flip),
            "thick": dict(self._thick),
            "slice2d": int(self._slice2d),
            "win": float(self._win),
            "lvl": float(self._lvl),
            "cam": {k: cam(k) for k in ("A", "B")},
            "cpr_T": (None if self._cpr is None else self._cpr["T"].copy()),
            "cpr_idx": (None if self._cpr is None else self._cpr.get("idx")),
        }

    def _view_restore(self, snap) -> None:
        if self._image is None or snap is None:
            return
        for k in ("A", "B"):
            self._frame[k] = tuple(np.asarray(a, float).copy()
                                   for a in snap["frame"][k])
            self._pc[k] = snap["pc"][k].copy()
        if snap.get("axes2d") is not None:
            self._axes2d = (snap["axes2d"][0].copy(), snap["axes2d"][1].copy())
        self._center = snap["center"].copy()
        self._cross_ang = dict(snap["cross_ang"])
        if snap.get("apex_flip") is not None:
            self._apex_flip = dict(snap["apex_flip"])
        self._thick = dict(snap["thick"])
        self._slice2d = int(snap.get("slice2d", self._slice2d))
        if snap.get("win") is not None:
            self._win = float(snap["win"])
        if snap.get("lvl") is not None:
            self._lvl = float(snap["lvl"])
        if self._cpr is not None and snap.get("cpr_T") is not None:
            self._cpr["T"] = np.asarray(snap["cpr_T"], float).copy()
            if snap.get("cpr_idx") is not None:
                self._cpr["idx"] = snap["cpr_idx"]
            self._cpr_apply_xform()
        if self._mode == "2D":
            self._apply_2d_axes()
        for k in ("A", "B"):
            c = self.pane[k].ren.GetActiveCamera()
            cc = snap["cam"][k]
            c.SetViewUp(*cc["vu"])
            c.SetFocalPoint(*cc["fp"])
            c.SetPosition(*cc["pos"])
            c.SetParallelScale(max(1e-3, cc["ps"]))
        self._view_initial = False
        self._refresh(reset_cam=False)
        for k in ("A", "B"):
            self._redraw_meas(k)
        if self._mode == "2D":
            self._sync_seek()

    # LV kept the same public names; they now feed the unified stack.
    def _lv_reset_undo(self) -> None:
        self._undo_clear()

    def _lv_border_snap(self, pane, mi):
        """A restorable snapshot of one LV border's 3-D points (or None)."""
        if self._lv is None or not (0 <= mi < len(self._measures.get(pane, []))):
            return None
        m = self._measures[pane][mi]
        tag = m.get("_lv")
        if tag is None or not m.get("pts3d"):
            return None
        return {"pane": pane, "tag": tuple(tag),
                "pts3d": [list(map(float, P)) for P in m["pts3d"]]}

    def _lv_record_border(self, before, pane, mi) -> None:
        """Record an LV-border edit (before → after) as one undo/redo command."""
        after = self._lv_border_snap(pane, mi)
        if before is None or after is None:
            return
        self._undo_record(lambda b=before: self._lv_restore_border(b),
                          lambda a=after: self._lv_restore_border(a))

    def _lv_push_undo(self, pane, mi) -> None:
        """Stash an LV border's 3-D points BEFORE a DRAG; committed on release
        (see _measure_release) once the final points are known."""
        self._lv_edit_before = self._lv_border_snap(pane, mi)

    def _lv_restore_border(self, snap) -> bool:
        if self._lv is None or self._lv.get("phase") != "contour":
            return False
        pane, tag = snap["pane"], tuple(snap["tag"])
        for m in self._measures.get(pane, []):
            if m.get("_lv") == tag:
                m["pts3d"] = [np.asarray(P, float) for P in snap["pts3d"]]
                # Regenerate the 2-D points to MATCH — add/delete change the point
                # count, and _redraw_geom only re-projects pts3d when the counts
                # already agree, so without this an add/delete undo restored the
                # 3-D border but left the on-screen polyline unchanged.
                m["pts"] = [self._world3d_to_out(pane, P) for P in m["pts3d"]]
                if len(m["pts"]) >= 3:
                    m["type"] = "polyline"
                angs = self._lv["model"].plane_angles()
                self._lv["model"].set_long_axis_contour(
                    angs[tag[0] % len(angs)], m["pts3d"], tag[1])
                self._lv_invalidate_volume()
                self._redraw_meas(pane)
                if self._lv_sax_active():
                    sa = self._lv["sax_pane"]
                    self._redraw_lv(sa)
                    self.pane[sa].render()
                return True
        return False

    def _lv_invalidate_volume(self) -> None:
        """A border changed → the computed volume is stale: clear the result
        readout, forget the numbers and turn CalcVol grey. Re-pressing CalcVol
        recomputes + re-lights blue (and Save then persists the fresh value)."""
        lv = self._lv
        if lv is None:
            return
        if lv.get("vol_done") or lv.get("vol_endo_ml") is not None \
                or lv.get("vol_epi_ml") is not None or self._lv_result_lines:
            lv["vol_done"] = False
            lv["vol_endo_ml"] = None
            lv["vol_epi_ml"] = None
            lv["vol_myo_ml"] = None
            self._lv_result_lines = []
            self._lv_sync_buttons()
            self._lv_update_text()

    def _lv_live_recapture(self, key, m) -> None:
        """If *m* is an LV endo/epi border being edited on the long-axis pane
        while short-axis is shown, feed the edited 3-D border back into the model
        and redraw the short-axis at once, so the cross-section tracks the edit."""
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
        sa = self._lv["sax_pane"]
        self._redraw_lv(sa)
        self.pane[sa].render()

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
            if (mi is not None and 0 <= mi < len(self._measures[key])):
                self._lv_live_recapture(key, self._measures[key][mi])
            self._redraw_meas(key)
            # Commit an LV-border DRAG (snapshot stashed at mouse-press) as one
            # undo/redo step, now that the final dragged points are known.
            if getattr(self, "_lv_edit_before", None) is not None \
                    and mi is not None:
                self._lv_record_border(self._lv_edit_before, key, mi)
            self._lv_edit_before = None
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
        # A resumed trace keeps its ORIGINAL id (and colour / spline / alpha) so
        # it reads as the same result continued, not a brand-new one; a fresh
        # draft takes the next sequence number.
        if d.get("resume_id") is not None:
            rid = int(d["resume_id"])
            self._meas_seq = max(self._meas_seq, rid)
        else:
            self._meas_seq += 1
            rid = self._meas_seq
        rec = {"id": rid, "type": d["type"], "pts": pts}
        for k in ("color", "smooth", "transp"):
            if d.get(k) is not None:
                rec[k] = d[k]
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
            mid=rid,
        ))

    def _measure_finish_draft(self):
        d = self._draft
        if d and d["type"] in ("polyline", "polygon") \
                and len(d["pts"]) >= 2:
            self._commit_draft()
            # LV EF: a finished border is captured to the current target and the
            # per-plane sequence advances endo → epi (both on THIS plane before
            # the user steps to the next long-axis plane).
            if self._lv is not None and self._lv.get("phase") == "contour":
                self._lv_on_border_committed()
                # Force the full re-render so the spline + endo/epi colour appear
                # AT ONCE — a right-click finish otherwise left the border un-
                # smoothed / still the draft colour until the next F/A plane step.
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
        # Switch the tool to Polyline so the next clicks EXTEND this trace.
        # (Right-click only reaches this menu with Measure mode already on, so
        # left-clicks route to measuring — we just need the type armed.) Sync
        # the toolbar buttons WITHOUT calling _set_measure_type, which would
        # wipe the draft we're about to install.
        self._meas_type = "polyline"
        for k, b in self._meas_btns.items():
            b.setChecked(k == "polyline")
            b.setStyleSheet(
                "background:#1f77b4;color:white;" if k == "polyline" else "")
        self._refresh_tool_availability()
        # Prefer the absolute 3-D control points (re-projected onto the CURRENT
        # plane) as the source of truth; fall back to the stored 2-D vertices.
        p3 = m.get("pts3d")
        if p3 and len(p3) == len(m["pts"]) and self._mode == "3D":
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
        del_pt = del_res = resume_act = None
        if m["type"] in ("polyline", "polygon"):
            del_pt = menu.addAction(t("Delete point"))
            if len(m["pts"]) <= 2:                # never shrink below Line
                del_pt.setEnabled(False)
        # Right-click on a polyline END vertex (the 断端) → "Resume trace":
        # un-commit it so the user can keep clicking points to EXTEND that end
        # (Add point only inserts BETWEEN existing vertices, never past an end).
        if m["type"] == "polyline" and vi in (0, len(m["pts"]) - 1):
            resume_act = menu.addAction(t("Resume trace"))
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
        elif resume_act is not None and chosen is resume_act:
            self._resume_trace(which, mi, vi)
            return                               # _resume_trace redraws itself
        elif chosen is hide_act:
            m["hidden"] = not m.get("hidden", False)   # hide THIS line only
        elif chosen is del_res:
            self._lv_drop_border(m)              # clear its model contour (if LV)
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
        _lv_before = (self._lv_border_snap(which, mi)
                      if self._lv is not None and m.get("_lv") else None)
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
        self._lv_live_recapture(which, m)     # edited LV border → refresh SAX
        self._lv_record_border(_lv_before, which, mi)   # Ctrl+Z / Ctrl+Y

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
        _lv_before = (self._lv_border_snap(which, mi)
                      if self._lv is not None and m.get("_lv") else None)
        del pts[vi]
        p3 = m.get("pts3d")
        if p3 and 0 <= vi < len(p3):
            del p3[vi]
        if len(pts) == 2:
            m["type"] = "line"
            m.pop("pts3d", None)               # a Line is a plain 2-D measure
        m["pts"] = pts
        self._resnap_center_angle(m)
        self._lv_live_recapture(which, m)     # edited LV border → refresh SAX
        self._lv_record_border(_lv_before, which, mi)   # Ctrl+Z / Ctrl+Y

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
        """Min/max HU inside *poly* (output-plane U,V mm), sampled DIRECTLY from
        the volume on a voxel-pitch grid — independent of the display reslice
        (which now only covers the zoomed-in viewport, so reading its pixels
        would miss an off-screen ROI)."""
        if self._vol is None or len(poly) < 3:
            return 0.0, 0.0
        u, v, _n = self._axes_for(key)
        pc = self._pc[key]
        xs = [q[0] for q in poly]
        ys = [q[1] for q in poly]
        step = max(0.1, min(self._dims))          # ~voxel pitch
        # Gather the (u,v) grid points inside the polygon, lift to 3-D world.
        pts = []
        y = min(ys)
        while y <= max(ys):
            x = min(xs)
            while x <= max(xs):
                if _point_in_poly(x, y, poly):
                    pts.append(pc + x * u + y * v)
                x += step
            y += step
        if not pts:
            return 0.0, 0.0
        # Batched trilinear sample of the volume (voxel index = world / spacing).
        P = np.asarray(pts, float)
        sx, sy, sz = self._dims
        fx = P[:, 0] / sx
        fy = P[:, 1] / sy
        fz = P[:, 2] / sz
        nz, ny, nx = self._vol.shape
        inb = ((fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
               & (fz >= 0) & (fz <= nz - 1))
        if not inb.any():
            return 0.0, 0.0
        fx, fy, fz = fx[inb], fy[inb], fz[inb]
        x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
        z0 = np.clip(np.floor(fz).astype(int), 0, nz - 2)
        tx, ty, tz = fx - x0, fy - y0, fz - z0
        V = self._vol
        val = np.zeros(len(fx))
        for dz in (0, 1):
            for dy2 in (0, 1):
                for dx2 in (0, 1):
                    w = ((tx if dx2 else 1 - tx) * (ty if dy2 else 1 - ty)
                         * (tz if dz else 1 - tz))
                    val += w * V[z0 + dz, y0 + dy2, x0 + dx2]
        return float(val.min()), float(val.max())

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
        self._undo_clear()               # fresh Ctrl+Z history for the new series
        vol = loaded.volume
        sr, sc = loaded.spacing_mm or (1.0, 1.0)
        sz = loaded.slice_mm or 1.0
        self._dims = (float(sc), float(sr), float(sz))   # x, y, z mm
        self._vol = np.asarray(vol)                      # (z,y,x) for lumen snap
        self._image = numpy_to_vtk_image(vol, sc, sr, sz)
        self._bounds = self._image.GetBounds()
        self._header = loaded.header
        self._src_dir = getattr(loaded, "source_dir", "") or ""   # data folder
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
        self._lv = None                          # drop any LV EF session
        self._lvv = None                         # drop LV blood-pool session
        self._lvv_epi_surf = None                # a new series invalidates the Epi
        self._lvv_epi_model_dict = None
        if hasattr(self, "_lvv_start_btn"):
            self._lvv_sync()
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

    def lv_active(self) -> bool:
        """True while an LV analysis session is in progress. The shell must NOT
        auto-demote such a pane to a memory-saving 'still': the LV state (mode,
        long axis, traced borders, computed volume) isn't captured by the still
        snapshot, so demoting garbles the image and loses the whole session."""
        return getattr(self, "_lv", None) is not None

    def snapshot(self):
        """A QPixmap of just the image frames — no toolbars — for the shell's
        memory-saving 'still' pane.

        MUST read the VTK render window via vtkWindowToImageFilter (glReadPixels),
        NOT QWidget.grab(): the canvas is a NATIVE OpenGL window with no Qt paint
        engine, so grab() paints black and logs
        "QPainter::begin: Paint device returned engine == 0". That black grab is
        what made a demoted CT pane go black. The window-to-image capture keeps
        the image + overlay text actors (the earlier text-size cosmetic quirk is
        far preferable to a black still). Visible panes are composed side by side.
        Returns None on failure."""
        from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
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
        self._cpr = None                         # drop any short-axis session
        self._lv = None                          # drop any LV EF session
        self._lv_valves = {"mitral": None, "aortic": None}   # new series → clear
        self._lv_valve_shown = {"mitral": True, "aortic": True}
        self._lvv = None                         # drop LV blood-pool session
        self._lvv_epi_surf = None                # a new series invalidates the Epi
        self._lvv_epi_model_dict = None
        if hasattr(self, "_lvv_start_btn"):
            self._lvv_sync()
        self._cpr_wrap.setVisible(False)
        self._vol = None
        self._image = None
        self._header = None
        for key in ("A", "B"):
            self.pane[key].reslice.SetInputData(_placeholder_image())
            self.pane[key].hover_hu.SetInput("")
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
        Also resets the ▲ apex-marker side to default (fresh unmirrored frames).

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

    # ------------------------------------------------ LV EF (Phase 1)
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
        if there's no image. LV always works on the RIGHT pane (B) long axis; the
        LEFT (A) becomes the short-axis side."""
        if self._image is None:
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
        self._lv_dirty = False                       # nothing traced yet
        self._lv_reset_undo()                        # fresh Ctrl+Z stack
        self._lv_btn.setChecked(True)               # internal mode flag
        self.set_side("Bi")
        return True

    def _lv_thick_trace_both(self) -> None:
        """Slab thickness for the LV panes: 5 mm on the long-axis TRACE pane (a
        thin MIP slab helps see the endo/epi border), 0 mm on the other
        (cross-section) pane."""
        pane = self._lv["pane"] if self._lv is not None else "B"
        for k in ("A", "B"):
            self._thick[k] = 5.0 if k == pane else 0.0

    # ---- pass flow: align the view → Set axis → place apex → trace ----------
    def _lv_axis_from_view(self):
        """(origin, axis_dir, radial0) of the CURRENT long-axis view on the trace
        pane — the rotation axis = the no-arrow centreline, θ=0 = the green-▲
        radial (same convention the single-axis entry used)."""
        key0 = self._lv["pane"]
        u, v, _n = self._frame[key0]
        a = math.radians(self._cross_ang[key0])
        axis_dir = -math.sin(a) * u + math.cos(a) * v
        radial0 = math.cos(a) * u + math.sin(a) * v
        origin = np.asarray(self._pc[key0], dtype=float).copy()
        return origin, axis_dir, radial0

    def _lv_enter_align(self) -> None:
        """ALIGN sub-phase for the active pass: free long-axis MPR so the user
        orients the view; 'Set axis' then captures it. Trace/analysis controls
        are disabled until the axis is set."""
        lv = self._lv
        lv["phase"] = "align"
        lv["target"] = None
        self.set_side("Bi")
        self._lv_thick_trace_both()                  # slab 5mm both panes
        # Hide the other pass's border while aligning this one (e.g. the Endo
        # line must vanish once Epi is active) — _lv_show_plane isn't called in
        # the align phase, so set the visibility here.
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
        for k in ("A", "B"):
            self.pane[k].render()

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
        # independent-axis trace (undo the Epi-axis promotion, non-destructive)
        # so re-tracing happens on the axis through the endo apex.
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
        if ph == "ready":
            m = lv["model"]
            apex = m.endo_apex if lv["pass"] == "endo" else m.epi_apex
            if apex is not None:                     # apex already set → resume
                lv["apex_target"] = None
                self._lv_enter_contour()
                return
            lv["phase"] = "apex"
            lv["apex_target"] = lv["pass"]
            if self._meas_on:                        # clicks place the apex first
                self._meas_btn.setChecked(False)
                self._toggle_measure()
            self._lv_sync_buttons()
            self._lv_update_text()
            for k in ("A", "B"):
                self.pane[k].render()
        elif ph in ("apex", "contour") and lv.get("sax") is None:
            lv["phase"] = "ready"                    # UNDO trace → ready
            lv["apex_target"] = None
            # Clear this pass's placed apex so its marker disappears and it can be
            # re-placed (re-Trace) — or the axis redone via Set axis.
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
            # After promotion both borders live on the Epi axis and both are
            # shown on the long-axis plane; the Endo/Epi button picks WHICH
            # border's points are editable (the other is display-only). Endo/Epi
            # + Trace then leaves SAX into that pass's trace (see _lv_start_trace).
            store = m.endo_contours if which == "endo" else m.epi_contours
            if m._axis_for(which) is not None and len(store) >= 3:
                lv["sax_edit"] = which
                lv["pass"] = which
                self._lv_apply_target(which)        # only this border grabbable
                self._lv_sync_buttons()
                self._lv_show_sax_both()
            return
        # Toggle OFF before Set axis: clicking the ALREADY-selected pass again
        # while still aligning (no axis set yet) DESELECTS it → back to the
        # no-pass state, so Endo/Epi return to their default look and the rest of
        # the bar (Set axis, …) greys out again.
        if lv.get("pass") == which and lv.get("phase") == "align":
            lv["pass"] = None
            self._lv_apply_target(None)
            self._lv_sync_buttons()
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

    def _lv_apex_on_axis(self, tgt, sx, sy):
        """3-D point for an apex click/drag at screen (sx,sy), CONSTRAINED to the
        pass's long (rotation) axis — the apex can slide ALONG the axis but never
        leave it. None if that axis isn't set."""
        ax = self._lv["model"]._axis_for(tgt)
        if ax is None:
            return None
        which = self._lv["pane"]
        wx, wy = self._disp_to_world(which, sx, sy)
        vol = self._matrix(which).MultiplyPoint((wx, wy, 0.0, 1.0))
        P = np.array([vol[0], vol[1], vol[2]], dtype=float)
        along = float(np.dot(P - ax.apex, ax.axis))    # project onto the axis
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
        vol = self._matrix(which).MultiplyPoint((wx, wy, 0.0, 1.0))
        P = np.array([vol[0], vol[1], vol[2]], dtype=float)
        # Preserve the long-axis VIEW across the axis re-pin: remember the world
        # point at the view centre now. Re-pinning moves the plane centre (_pc)
        # to the new axis midpoint, which — with the camera unchanged — shifted
        # the anatomy so it looked smaller (reported: image shrinks on apex).
        cam = self.pane[which].ren.GetActiveCamera()
        fp0 = cam.GetFocalPoint()
        try:
            wc = self._out_to_world3d(which, float(fp0[0]), float(fp0[1]))
        except Exception:                          # noqa: BLE001
            wc = None
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
        # Re-centre the long-axis pane on the SAME world point so the view (size
        # AND position) is unchanged by the apex placement.
        if wc is not None:
            try:
                ox, oy = self._world3d_to_out(which, wc)
                pos = cam.GetPosition()
                fp = cam.GetFocalPoint()
                cam.SetFocalPoint(ox, oy, fp[2])
                cam.SetPosition(ox, oy, pos[2])
                self._refresh(only=which)          # reslice around the new centre
                self._update_cross(which)
                self.pane[which].render()
            except Exception:                      # noqa: BLE001
                pass
        return True

    def _lv_apex_press(self, which, sx, sy, shift=False):
        """Left-press hit-test for the apex markers. Returns "place" if a click
        was consumed as an apex placement, "endo"/"epi" if an existing marker was
        grabbed to drag, else None. In the APEX phase a plain click places the
        apex; a SHIFT-click yields (None) so the view can be zoomed/moved/rotated
        to adjust before placing it. Grabbing is allowed only when NOT mid-trace."""
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
        # Only the ACTIVE pass's apex is grabbable — the other pass's apex is
        # shown for reference but must not be selected/moved from here.
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
        self._redraw_all_lv()
        self._redraw_meas(self._lv["pane"])        # moved border points
        for k in ("A", "B"):
            self.pane[k].render()

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
        """Reslice pane A to the current rotated long-axis plane; update label."""
        lv = self._lv
        ax = lv["model"].axis
        if ax is None:
            return
        angs = lv["model"].plane_angles()
        idx = lv["plane_idx"] % len(angs)
        lv["plane_idx"] = idx
        phi = angs[idx]
        pane = lv["pane"]
        u, v, n = self._ortho(ax.meridian_dir(phi), ax.axis)  # v = apex→base up
        self._frame[pane] = (u, v, n)
        self._pc[pane] = ax.apex + 0.5 * ax.length_mm * ax.axis
        self._cross_ang[pane] = 0.0
        # Just after Set axis (ready): sync the cross-section pane's reslice
        # centre to the crosshair (_center), like a recenter does
        # (self._pc[other] = self._center). Set axis moves _center onto the axis
        # but did NOT re-cut the other pane, so its section was stale until the
        # user nudged the level (reported: right crosshair vs left section
        # mismatch on entry). Only in 'ready' — contour keeps its own section.
        if lv.get("phase") == "ready":
            other = "A" if pane == "B" else "B"
            self._pc[other] = np.asarray(self._center, float).copy()
        # Fit the view only the FIRST time; keep the user's zoom/pan when they
        # step planes (angle change) afterwards.
        first = not lv.get("fitted", False)
        lv["fitted"] = True
        self._view_initial = first
        # Show only THIS plane's border for the ACTIVE pass. Endo and Epi are on
        # DIFFERENT axes, so the other pass's border would project onto a wrong
        # cross-section here (looks broken) — hide it; both show on the SAX pane.
        for mm in self._measures[pane]:
            tag = mm.get("_lv")
            if tag is not None:
                mm["hidden"] = (tag[0] != idx) or (tag[1] != lv.get("pass"))
        self._lv_plane_lbl.setText(f"{idx + 1}/{len(angs)}")   # e.g. 1/6
        # Preserve the ZOOM the user aligned at: the first show (right after Set
        # axis) fits the camera, which would otherwise rescale (shrink) the image.
        # Keep the parallel scale across that fit so the scale never changes on
        # Set axis; the fit still recenters the plane. (Reported: image shrinks.)
        cam = self.pane[pane].ren.GetActiveCamera()
        ps0 = float(cam.GetParallelScale())
        self._refresh(reset_cam=first)
        # Force the long axis EXACTLY vertical: the reslice frame's v = ax.axis,
        # so reset the camera roll (any SPIN done before Set axis would otherwise
        # leave the axis diagonal). The output plane is (x=u, y=v), so up=(0,1,0).
        cam.SetViewUp(0.0, 1.0, 0.0)
        if first:
            cam.SetParallelScale(ps0)                # keep the aligned zoom
            # The ▲ markers are sized 0.024·ParallelScale in _update_cross, which
            # ran during _refresh at the pre-restore (fitted) scale — leaving them
            # oversized. Re-size them for the restored scale.
            self._update_cross(pane)
        # Keep the centreline (crosshair) but drop only the slab-width parallel
        # lines in LV trace mode.
        self.pane[pane].set_overlay_visible(self._cl_btn.isChecked())
        self.pane[pane].set_slab_visible(False)
        self.pane[pane].render()
        self._lv_update_text()

    def _lv_toggle_sax(self) -> None:
        """Enter/leave the short-axis DISPLAY. On: show BOTH panes — the traced
        long-axis view stays on its pane (with a movable level line), and the
        OTHER pane shows the cross-section ⟂ the axis at that level with the
        endo/epi borders. ◀ ▶ translate the level; the short-axis updates."""
        from PyQt6.QtWidgets import QMessageBox
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
                # Silent + non-destructive (the original endo trace is stashed;
                # Endo → Trace restores it). Then re-place the on-screen Endo
                # correction points on the Epi meridian planes.
                if m.promote_endo_to_epi_axis():
                    self._lv_rebuild_measures()
                    self._lv_result_lines = []       # promoted geometry → restate
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
            # Point the model's active axis at the SAX reference axis so the
            # existing SAX code slices against it (short_axis_border_pts default
            # ref_axis = model.axis): endo/epi single → that axis; both → epi.
            m.axis = self._lv_sax_axis()
            rng = self._lv_level_range()
            if rng is None:
                self._lv_sax_btn.setChecked(False)
                return
            # Start at the mid of the COMMON range (where the border is drawn),
            # not the scroll range (which extends past the apex), so the border
            # shows on entry.
            common = self._lv_common_range() or rng
            lv["sax"] = 0.5 * (common[0] + common[1])
            sa = "A" if lv["pane"] == "B" else "B"
            lv["sax_pane"] = sa
            # Remember the cross-section pane's current image so leaving SAX can
            # restore it (we're already in Bi — no layout switch).
            lv["sax_saved"] = (
                tuple(np.asarray(a).copy() for a in self._frame[sa]),
                np.asarray(self._pc[sa]).copy(),
                self._cross_ang[sa], self._thick[sa])
            lv["fitted_sax"] = False
            # Arm the CURRENTLY-SELECTED pass for editing right away, so the user
            # can correct its border in SAX without re-clicking Endo/Epi first
            # (that extra step is now skipped). Only if that pass has a border.
            armed = lv.get("pass") if lv.get("pass") in ("endo", "epi") else None
            if armed == "endo" and not endo_ok:
                armed = None
            elif armed == "epi" and not epi_ok:
                armed = None
            lv["sax_edit"] = armed
            self._lv_apply_target(armed)            # arm that border for editing
            self.set_side("Bi")                      # long-axis + short-axis
            self._lv_sync_buttons()                  # SAX entry → all 4 buttons
            #                        neutral grey (Endo/Epi UNARMED — the stale
            #                        green Epi no longer implies it's editable)
            self._lv_show_sax_both()
        else:
            # SAX is the last-on button → pressing it turns SAX OFF (back to the
            # long-axis trace). The endo/epi single-vs-both content is chosen via
            # the Endo/Epi buttons and Wall, not by re-pressing SAX.
            self._lv_leave_sax()
            self._lv_show_plane()                    # back to the long-axis view

    def _lv_show_sax_both(self) -> None:
        """Full setup on ENTERING short-axis mode: the traced long-axis pane on
        one side + the short-axis cross-section on the other. The long-axis pane
        is set up ONCE here — rotating (◀ ▶) or scrolling the level afterwards
        must NOT re-slice it, so its image/axis stay fixed."""
        lv = self._lv
        ax = lv["model"].axis
        if ax is None or lv.get("sax") is None:
            return
        la, sa = lv["pane"], lv["sax_pane"]
        # long-axis pane: the current rotated plane (borders of THIS plane only)
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
        self._lv_set_short_frame()                   # short-axis (level) pane
        first = not lv.get("fitted_sax", False)
        lv["fitted_sax"] = True
        self._view_initial = first
        self._lv_update_sax_label()
        # Keep the LONG-AXIS pane's zoom on SAX entry — the fit would otherwise
        # rescale (shrink/grow) it like Set axis did (reported). The SHORT-AXIS
        # pane still fits (it's a fresh view). Restore the long-axis parallel
        # scale after the fit and re-size its ▲ markers for that scale.
        la_cam = self.pane[la].ren.GetActiveCamera()
        la_ps0 = float(la_cam.GetParallelScale())
        self._refresh(reset_cam=first)
        if first:
            la_cam.SetParallelScale(la_ps0)
            self._update_cross(la)
        for k in (la, sa):
            self.pane[k].set_overlay_visible(self._cl_btn.isChecked())
            self.pane[k].set_slab_visible(False)
            self.pane[k].render()

    def _lv_set_short_frame(self) -> None:
        """Reslice ONLY the short-axis (cross-section) pane to the current level.
        Never touches the long-axis pane, so rotating/scrolling can't disturb
        it."""
        lv = self._lv
        ax = lv["model"].axis
        sa = lv["sax_pane"]
        o, ex, ey, nn = ax.short_axis_basis(float(lv["sax"]))
        # View the short axis from the APEX toward the BASE (cardiology
        # convention): a horizontal MIRROR so the LV sits on the viewer's right
        # and the RV on the left, with the diaphragm still at the bottom. This
        # negates the in-plane horizontal axis (and the normal, keeping a
        # right-handed frame). Display-only — the volume uses the axis/borders,
        # so measurements are unaffected.
        self._frame[sa] = (-ex, ey, -nn)
        self._pc[sa] = o
        self._cross_ang[sa] = 0.0
        self._thick[sa] = 0.0                        # thin cross-section
        for mm in self._measures[sa]:                # no long-axis borders here
            if mm.get("_lv") is not None:
                mm["hidden"] = True

    def _lv_update_sax_label(self) -> None:
        rng = self._lv_level_range()
        pos = (float(self._lv["sax"]) - rng[0]) if rng else 0.0
        self._lv_plane_lbl.setText(t("SAX {mm:.0f}mm", mm=pos))

    def _lv_reslice_short(self) -> None:
        """Redraw SAX after a LEVEL or ROTATION change: reslice ONLY the
        short-axis pane and redraw all overlays via the same path as entry. The
        long-axis pane's frame is untouched, so its image/axis stay fixed (only
        its level-line overlay moves). Uses the full _refresh + visibility path
        so the endo/epi splines always re-appear."""
        lv = self._lv
        if lv is None or lv.get("sax") is None:
            return
        self._lv_set_short_frame()
        self._lv_update_sax_label()
        self._view_initial = False
        self._refresh()                  # long-axis frame unchanged → image fixed
        for k in (lv["pane"], lv["sax_pane"]):
            self.pane[k].set_overlay_visible(self._cl_btn.isChecked())
            self.pane[k].set_slab_visible(False)
            self.pane[k].render()

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

    def _lv_leave_sax(self) -> None:
        """Return from short-axis to the long-axis view (staying in Bi): restore
        the cross-section pane to the image it showed before SAX and re-activate
        the current pass's axis for tracing."""
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

    def _lv_capture_current(self) -> None:
        """Capture the freshly-traced polyline on the LV pane into the model for
        the current (plane, target), then KEEP it on screen — recoloured
        (endo=red, epi=green) and tagged by (plane, target). The displayed
        border is thus EXACTLY what you traced (no reconstruction). Re-tracing
        the same border replaces the previous one."""
        lv = self._lv
        if lv is None or lv.get("phase") != "contour":
            return
        if lv.get("target") not in ("endo", "epi"):
            return                                # no target armed → plain polyline
        pane = lv["pane"]
        if self._draft and self._draft.get("pane") == pane \
                and len(self._draft.get("pts", [])) >= 2:
            self._commit_draft()                  # finish an un-committed trace
        m = None
        for cand in reversed(self._measures[pane]):
            if (cand.get("type") == "polyline" and len(cand.get("pts3d", [])) >= 2
                    and cand.get("_lv") is None):     # a fresh (untagged) trace
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
            lv["model"].set_long_axis_contour(phi, m["pts3d"], which=lv["target"])
        except Exception:
            return
        self._lv_dirty = True          # a traced/changed border is now unsaved
        tag = (lv["plane_idx"], lv["target"])
        # re-trace: drop the previous captured border for this (plane, target)
        self._measures[pane] = [mm for mm in self._measures[pane]
                                if mm is m or mm.get("_lv") != tag]
        m["_lv"] = tag
        m["color"] = "#ff4040" if lv["target"] == "endo" else "#40c040"
        m["smooth"] = True             # AUTO-spline the LV border on finish
        #                                (LV traces only; other traces unchanged)
        self._lv_invalidate_volume()   # a changed border invalidates the volume
        self._draft = None
        self._draft_redo = []          # the trace is consumed
        self._lv_apex_hot = False      # border confirmed → marker back to normal
        self._redraw_meas(pane)
        self._lv_record_create(pane, m)   # Ctrl+Z removes the whole new border
        self._redraw_all_lv()          # refresh the base-cut line from the model

    def _lv_step_plane(self, delta) -> None:
        if self._lv is None or self._lv.get("phase") != "contour":
            return
        # Short-axis mode: ◀ ▶ ROTATE — step the meridian. The LONG-AXIS pane
        # (right) reslices to that meridian plane (so its border can be edited
        # against the matching image) and the short-axis centreline rotates; the
        # short-axis LEVEL is unchanged. Level itself moves by paging/mouse-drag.
        if self._lv.get("sax") is not None:
            before = self._lv_scalar_snap()
            self._lv["plane_idx"] += int(delta)
            self._lv_show_sax_both()       # reslice long-axis to the new meridian
            self._lv_record_scalar(before)     # Ctrl+Z / Ctrl+Y
            return
        self._lv_capture_current()
        pane = self._lv["pane"]
        # drop any leftover un-captured scratch polyline (plain, untagged)
        self._measures[pane] = [
            m for m in self._measures[pane]
            if not (m.get("type") == "polyline" and m.get("_lv") is None)]
        self._draft = None
        self._lv["plane_idx"] += int(delta)
        self._lv_apply_target(self._lv["pass"])   # target locked to this pass
        self._lv_show_plane()

    def _lv_sax_active(self) -> bool:
        """True while the short-axis DISPLAY is up (level is scrollable)."""
        return (self._lv is not None
                and self._lv.get("phase") == "contour"
                and self._lv.get("sax") is not None)

    def _lv_axis_locked(self) -> bool:
        """True once the active pass's long axis is SET (Set axis pressed) and we
        are on the long-axis view — the 3DCT↔vertical-axis relationship is then
        fixed, so Rotate/Spin/Thick (which would re-tilt the reslice frame) are
        blocked. Zoom/Move/WL and the cross-section level still work."""
        lv = self._lv
        return (lv is not None
                and lv.get("phase") in ("ready", "apex", "contour")
                and lv.get("sax") is None)

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

    def _lv_level_range(self):
        """The along span (in the SAX axis' frame) to SCROLL the level over:
        apex = the most-apical traced point (min of minima), base = the common
        base (min of maxima). Uses only the store(s) in the SAX axis' frame so
        endo/epi on different axes aren't mixed."""
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
        """Move the short-axis LEVEL one paging notch (wheel / arrow keys)."""
        rng = self._lv_level_range()
        if rng is None:
            return
        before = self._lv_scalar_snap()
        step = (rng[1] - rng[0]) / 24.0            # ~24 levels apex→base
        self._lv["sax"] = min(rng[1], max(
            rng[0], float(self._lv["sax"]) + float(delta) * step))
        self._lv_reslice_short()
        self._lv_record_scalar(before)             # Ctrl+Z / Ctrl+Y

    def _lv_drag_level(self, dy) -> None:
        """Move the short-axis LEVEL by a mouse drag (Paging tool): drag down →
        toward the base, up → toward the apex. Full span ≈ 200 px of drag."""
        rng = self._lv_level_range()
        if rng is None:
            return
        span = rng[1] - rng[0]
        self._lv["sax"] = min(rng[1], max(
            rng[0], float(self._lv["sax"]) + (dy / 200.0) * span))
        self._lv_reslice_short()

    # ---- SAX: grab the ○-marked level / centre line directly ----
    def _lv_px_to_mm(self, which, px) -> float:
        """Screen px → output-plane mm for *which* pane (for a constant-width
        grab band that tracks zoom)."""
        p = self.pane[which]
        ps = float(p.ren.GetActiveCamera().GetParallelScale())
        return float(px) * (2.0 * ps / max(1, p.canvas.height()))

    def _lv_line_press(self, which, sx, sy):
        """Hit-test the SAX lines: returns "level" if the press lands on the
        long-axis pane's level-line ○ handle, "meridian" if on the short-axis
        pane's centreline ○ handle, else None. Grabs ONLY near the ○ handle (out
        at the view edge, clear of the heart) — NOT along the whole line, which
        crosses the trace and would steal clicks meant to place / edit a border
        point (a near-miss moved the level, dropping the point onto a different
        cross-section). Yields while a trace is in progress or a border point is
        under the cursor, so tracing / editing always wins."""
        if not self._lv_sax_active() or self._lv["model"].axis is None:
            return None
        if self._draft is not None and self._draft.get("pane") == which:
            return None                       # mid-trace → clicks add points
        if self._pick_handle(which, sx, sy) is not None:
            return None                       # let the border point be edited
        lv = self._lv
        ax = lv["model"].axis
        wx, wy = self._disp_to_world(which, sx, sy)
        rgrab = self._lv_px_to_mm(which, 22.0)    # radius around the ○ handle
        if which == lv.get("pane"):
            _, y = self._world3d_to_out(
                which, ax.apex + float(lv["sax"]) * ax.axis)
            hx, hy = self._lv_ring_xy(which, 0.0, y, 1.0, 0.0)
            if math.hypot(wx - hx, wy - hy) <= rgrab:
                return "level"
        elif which == lv.get("sax_pane"):
            angs = lv["model"].plane_angles()
            md = ax.meridian_dir(angs[lv["plane_idx"] % len(angs)])
            u, v, _n = self._frame[which]
            dx, dy = float(np.dot(md, u)), float(np.dot(md, v))
            nrm = math.hypot(dx, dy) or 1.0
            dx, dy = dx / nrm, dy / nrm
            cx, cy = self._lv_view_center(which)   # same anchor as the drawn line
            ex, ey = self._lv_ring_xy(which, cx, cy, dx, dy)
            if math.hypot(wx - ex, wy - ey) <= rgrab:
                return "meridian"
        return None

    def _lv_line_move(self, which, sx, sy) -> None:
        """Drag the grabbed SAX line: level line → translate the cross-section
        level (parallel move); centreline → rotate the meridian (snapping to the
        nearest traced plane)."""
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
        """A captured LV border measure is being deleted: drop its meridian
        contour from the model too and refresh the short-axis, so nothing stale
        lingers (and the volume recomputes clean)."""
        if self._lv is None:
            return
        tag = m.get("_lv")
        if tag is None:
            return
        idx, target = tag
        angs = self._lv["model"].plane_angles()
        if 0 <= idx < len(angs):
            self._lv["model"].clear_contour(angs[idx], which=target)
        self._lv_result_lines = []               # invalidate any volume result
        if self._lv_sax_active():
            self._redraw_lv(self._lv["sax_pane"])
            self.pane[self._lv["sax_pane"]].render()

    def _lv_line_set_grabbed(self, which, on: bool) -> None:
        """Thicken the SAX line a little while it's hovered/held (feedback, like
        the crosshair) — kept modest since the line is already fairly thick."""
        p = self.pane[which]
        a = getattr(p, "lv_line_actor", None)
        if a is None:
            return
        a.GetProperty().SetLineWidth(3.8 if on else getattr(p, "lv_line_w", 2.4))
        p.render()

    def _lv_line_hover(self, which, sx, sy) -> None:
        """Cursor moved with no button down: thicken the level/centre line while
        the cursor is inside its (unchanged) grab band, so 'click here to grab'
        is visible; restore it otherwise. Only re-renders on a state change."""
        if self._lv_line_drag is not None:
            return                                # a drag owns the thickness
        on = bool(self._lv_sax_active()
                  and self._lv_line_press(which, sx, sy) is not None)
        if self._lv_line_hi.get(which) == on:
            return
        self._lv_line_hi[which] = on
        self._lv_line_set_grabbed(which, on)

    def _lv_apply_target(self, target) -> None:
        """Set the active border target + sync the Endo/Epi buttons + text
        (no capture — the caller handles that)."""
        self._lv["target"] = target
        self._lv_endo_btn.setChecked(target == "endo")
        self._lv_epi_btn.setChecked(target == "epi")
        self._lv_update_text()

    def _lv_set_target(self, target) -> None:
        """Endo/Epi button: EXPLICITLY choose which border the next trace is.
        Clicking the already-active one turns capture OFF → a plain polyline.
        Captures any pending trace under the previous target first."""
        if self._lv is None or self._lv.get("phase") != "contour":
            return
        self._lv_capture_current()
        new = None if self._lv.get("target") == target else target
        self._lv_apply_target(new)

    def _lv_on_border_committed(self) -> None:
        """A border was just finished (double-click). Capture it ONLY if an
        Endo/Epi target is armed; otherwise leave it as a plain polyline. No
        auto endo→epi switch — the user picks each target explicitly."""
        lv = self._lv
        if lv.get("target") in ("endo", "epi"):
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
        """Discard all captured borders and start tracing again from plane 0
        (the long axis / view is kept)."""
        if self._lv is None or self._lv.get("phase") != "contour":
            return
        pane = self._lv["pane"]
        m = self._lv["model"]
        m.endo_contours.clear()
        m.epi_contours.clear()
        m.endo_planes.clear()               # also drop the raw borders (else SAX
        m.epi_planes.clear()                # / rebuild would resurrect them)
        m.endo_orig = None                  # and the promotion stash
        self._lv_reset_undo()
        self._measures[pane] = [mm for mm in self._measures[pane]
                                if mm.get("type") != "polyline"]
        self._draft = None
        self._lv_result_lines = []
        self._lv["plane_idx"] = 0
        if self._lv.get("sax") is not None:     # leaving short-axis (stay Bi)
            self._lv_leave_sax()
        self._lv_apply_target(self._lv.get("pass"))
        self._lv_show_plane()
        self._redraw_meas(pane)

    def _lv_compute_volume(self) -> None:
        """Build the endo/epi surfaces from the traced borders and report the LV
        cavity volume (voxels inside the endo surface) + myocardial mass.

        The reconstruction (surface loft + voxel counting) takes several seconds,
        so it runs on a worker thread behind a busy progress dialog: the click
        gets IMMEDIATE "computing" feedback (no more "did it even start?"), the
        window stays painted, and the bar animates because the UI event loop is
        free. The compute is pure-numpy on the model (no Qt/VTK), so it is safe
        off the UI thread; results are read back here after it finishes."""
        from PyQt6.QtCore import Qt, QThread
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog
        if self._lv is None or self._lv.get("phase") != "contour":
            return
        self._lv_capture_current()          # capture any pending trace
        m = self._lv["model"]
        # Per-sub-mode: Calc Vol computes the ACTIVE pass's own enclosed volume
        # (Endo → LV cavity, Epi → epicardial). Myocardium / EF (Epi − Endo) is a
        # separate cross-file tool, so this no longer needs BOTH borders.
        pas = self._lv.get("pass")
        if pas not in ("endo", "epi"):
            pas = "endo" if m.endo_planes else "epi"
        planes = m.endo_planes if pas == "endo" else m.epi_planes
        if len(planes) < 3:
            QMessageBox.information(
                self.window(), t("LV Volume"),
                t("Trace the {p} border on at least 3 planes first.").format(
                    p=pas.capitalize()))
            return
        # Parent dialogs to the TOP-LEVEL window, not this embedded viewer, to
        # avoid Qt's "must be a top level window" console warning.
        top = self.window()
        spacing = max(0.5, float(min(self._dims)))

        result: dict = {}

        class _VolWorker(QThread):
            def run(self_) -> None:
                try:
                    m.build()
                    result["vol"] = m.volume_ml(spacing, pas)
                except Exception as exc:                  # noqa: BLE001
                    result["err"] = str(exc)

        dlg = QProgressDialog(t("Computing LV volume…"), "", 0, 0, top)
        dlg.setWindowTitle(t("LV Volume"))
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setCancelButton(None)                 # not cancellable — it's short
        dlg.setMinimumDuration(0)                 # show at once
        dlg.setValue(0)
        worker = _VolWorker()
        # finished is delivered via the UI event loop that dlg.exec() runs, so
        # even if the worker finishes before exec() starts, reset() is queued and
        # closes the dialog (no hang).
        worker.finished.connect(dlg.reset)
        worker.start()
        dlg.exec()
        worker.wait()
        worker.deleteLater()

        if "err" in result:
            QMessageBox.information(
                top, t("LV Volume"),
                t("Could not build the LV surface: {err}", err=result["err"]))
            return
        vol_ml = result.get("vol")
        if vol_ml is None:
            QMessageBox.information(
                top, t("LV Volume"),
                t("Trace the {p} border on at least 3 planes first.").format(
                    p=pas.capitalize()))
            return
        # Show the result in the pane's RESULT block (not a dialog), labelled by
        # sub-mode: "Endo-LV Volume:" or "Epi-LV Volume:".
        label = (t("Endo-LV Volume: {v:.1f} mL") if pas == "endo"
                 else t("Epi-LV Volume: {v:.1f} mL"))
        self._lv_result_lines = [label.format(v=vol_ml)]
        self._lv["vol_done"] = True          # CalcVol button → blue (valid result)
        # Remember the number so Save can persist it and Load can redisplay.
        if pas == "endo":
            self._lv["vol_endo_ml"] = float(vol_ml)
        else:
            self._lv["vol_epi_ml"] = float(vol_ml)
        self._lv_sync_buttons()
        self._lv_update_text()

    # ---- wall-thickness colour map (short axis) ----
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
                self._redraw_lv(lv["sax_pane"])
                self.pane[lv["sax_pane"]].render()
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
            self._redraw_lv(lv["sax_pane"])
            self.pane[lv["sax_pane"]].render()

    def _lv_update_wall_legend(self) -> None:
        """Show/hide the bottom-left wall-thickness colour legend (red <5 /
        orange 5-7 / yellow 7-9 / green >9 mm) on the short-axis pane — visible
        only while the wall map is up."""
        lv = self._lv
        show = bool(getattr(self, "_lv_wall", False)) and self._lv_sax_active()
        sa = lv.get("sax_pane") if lv is not None else None
        # bottom→top: green (thick) … red (thin, critical). ASCII labels so they
        # render regardless of the VTK font.
        bands = [("wall > 9 mm", (0.18, 0.80, 0.44)),
                 ("wall 7-9 mm", (0.95, 0.77, 0.06)),
                 ("wall 5-7 mm", (1.0, 0.55, 0.0)),
                 ("wall < 5 mm", (0.70, 0.0, 0.0))]
        for k in ("A", "B"):
            legend = getattr(self.pane[k], "lv_wall_legend", [])
            on = show and k == sa
            for i, a in enumerate(legend):
                if on and i < len(bands):
                    label, (r, g, b) = bands[i]
                    a.GetTextProperty().SetColor(r, g, b)
                    a.SetInput(label)
                    a.SetVisibility(True)
                else:
                    a.SetVisibility(False)

    def _lv_draw_wall(self, p, endo_sm, epi_sm) -> None:
        """Fill the annulus between the short-axis endo (inner) and epi (outer)
        splines with the Epi−Endo wall-thickness heatmap (Compare's gap
        colours), translucent, on the cross-section pane."""
        outer = [tuple(q) for q in epi_sm]
        inner = [tuple(q) for q in endo_sm]
        if len(outer) < 3 or len(inner) < 3:
            return
        cen = _polygon_centroid(outer)
        radials = _radial_gap_compare(outer, inner, cen, 1.0)
        n = len(radials)
        tris, cols = [], []
        for i in range(n):
            a, b = radials[i], radials[(i + 1) % n]
            da = abs(b["ang"] - a["ang"]) % 360.0
            if 2.5 < da < 357.5:                # skip a large angular gap
                continue
            rgb = _hex_to_rgb(_gap_color(a["gap"]))
            col = (rgb[0], rgb[1], rgb[2], 140)    # ~55% opacity
            tris.append((a["inner"], a["outer"], b["outer"]))
            tris.append((a["inner"], b["outer"], b["inner"]))
            cols += [col, col]
        if tris:
            p.lv_wall_mapper.SetInputData(_filled_tris_pd(tris, cols))

    # ---- save / load the traced Endo/Epi 3-D borders ----
    def _lv_series_meta(self) -> dict:
        """Series identity (name/date/series-no + UID) recorded in the .lvef.json
        so a load can confirm it's being applied to the same 3-D CT."""
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
        """Folder the displayed CT series was read from (where the .lvef.json is
        kept), so Save/Load open there. '' if unknown."""
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

    @staticmethod
    def _unlink_case_variant(path) -> None:
        """Remove any file in *path*'s folder whose name matches case-
        INSENSITIVELY but differs in exact case, so a following write creates the
        entry with *path*'s intended casing. Windows preserves an existing
        entry's case on overwrite, so a stale 'Epilv.json' would otherwise keep
        its lowercase 'l' when saving 'EpiLv.json'."""
        import os
        try:
            d = os.path.dirname(path) or "."
            base = os.path.basename(path)
            for nm in os.listdir(d):
                if nm != base and nm.lower() == base.lower():
                    os.remove(os.path.join(d, nm))
        except Exception:                               # noqa: BLE001
            pass

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
        """Suggested .lv.json filename, e.g. 'ARIFIN;20260629_Se006.lv.json'."""
        return self._lv_default_stem() + ".lv.json"

    def _lv_export_stl(self) -> None:
        """Export the reconstructed Endo/Epi surfaces as binary STL (mm scale).
        A dialog picks which of Endo / Epi / Endo+Epi to write; files go to the
        series folder as 'stem_Endo.stl' / '_Epi.stl' / '_EndoEpi.stl'."""
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
        dlg = LVStlExportDialog(self._lv_series_dir(), stem,
                                endo is not None, epi is not None, self.window())
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        ch = dlg.choices()
        outdir = dlg.out_dir() or self._lv_series_dir() or os.getcwd()
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
        # Save the ACTIVE sub-mode's border only (Endo → EndoLv.json, Epi →
        # EpiLv.json). Pick the pass; require it to have a border.
        pas = self._lv.get("pass")
        if pas not in ("endo", "epi"):
            pas = "endo" if m.endo_planes else "epi"
        planes = m.endo_planes if pas == "endo" else m.epi_planes
        if not planes:
            QMessageBox.information(
                self.window(), t("LV EF"),
                t("No {p} border to save yet.").format(p=pas.capitalize()))
            return
        # No valid volume yet → ask whether to save without it or compute first.
        vol_key = "vol_endo_ml" if pas == "endo" else "vol_epi_ml"
        has_vol = bool(self._lv.get("vol_done")
                       and self._lv.get(vol_key) is not None)
        if not has_vol:
            box = QMessageBox(self.window())
            box.setWindowTitle(t("LV Volume"))
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
        suffix = ".EndoLv.json" if pas == "endo" else ".EpiLv.json"
        d = self._lv_series_dir()
        fname = self._lv_default_stem() + suffix
        default = os.path.join(d, fname) if d else fname
        flt = (("Endo LV (*.EndoLv.json)" if pas == "endo"
                else "Epi LV (*.EpiLv.json)") + ";;JSON (*.json)")
        path, _ = QFileDialog.getSaveFileName(
            self.window(), t("Save LV borders"), default, flt)
        if not path:
            return
        if not path.endswith(".json"):
            path += suffix
        # EndoLv.json = endo only, EpiLv.json = epi only: filter the combined
        # model dict down to just this sub-mode's border.
        data = m.to_dict()
        if pas == "endo":
            data["epi_axis"] = None
            data["epi_apex"] = None
            data["epi_planes"] = {}
        else:
            data["endo_axis"] = None
            data["endo_apex"] = None
            data["endo_planes"] = {}
            data.pop("endo_orig", None)
        data["series"] = self._lv_series_meta()      # for the load-time match
        # Persist THIS sub-mode's computed volume (if a VALID result is showing —
        # vol_done is cleared on any edit) so Load can redisplay it.
        vv = self._lv.get(vol_key)
        if self._lv.get("vol_done") and vv is not None:
            data["volume"] = {("endo_ml" if pas == "endo" else "epi_ml"):
                              float(vv)}
        self._unlink_case_variant(path)      # force EndoLv/EpiLv exact casing
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.warning(self.window(), t("LV Volume"),
                                t("Save failed: {err}", err=str(exc)))
            return
        self._lv_dirty = False               # borders saved → no unsaved-switch warn
        # Keep the volume readout on screen after saving (append the saved note,
        # don't replace it) so the result stays visible.
        note = t("Saved: {p}", p=os.path.basename(path))
        lines = []
        if self._lv.get("vol_done") and vv is not None:
            label = (t("Endo-LV Volume: {v:.1f} mL") if pas == "endo"
                     else t("Epi-LV Volume: {v:.1f} mL"))
            lines.append(label.format(v=float(vv)))
        lines.append(note)
        self._lv_result_lines = lines
        self._lv_update_text()

    def _lv_load(self) -> None:
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from multi_dicomviewer.core.lv_measure import LVModel
        import json
        if self._image is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self.window(), t("Load LV borders"), self._lv_series_dir(),
            "LV border (*.EndoLv.json *.EpiLv.json);;JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            model = LVModel.from_dict(data)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.warning(self.window(), t("LV EF"),
                                t("Load failed: {err}", err=str(exc)))
            return
        if model.axis is None:
            QMessageBox.warning(self.window(), t("LV EF"),
                                t("The file has no LV axis."))
            return
        # The 3-D borders are in THIS series' volume coordinates — warn (but let
        # the user override) if the file was saved for a different series.
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
        """Enter LV contour mode with a loaded model (axis + borders from file)
        and re-create the on-screen Endo/Epi border measures from it."""
        if self._meas_on:
            self._meas_btn.setChecked(False)
            self._toggle_measure()
        self._lvv_deactivate()            # contour LV and LV Vol are exclusive
        self._lv = {"model": model, "phase": "contour", "plane_idx": 0,
                    "target": None, "pane": "B", "sax": None,
                    "pass": "epi" if model.epi_axis is not None else "endo",
                    "prev_side": self.current_side()}
        self._lv_dirty = False                       # loaded = matches the file
        self._lv_reset_undo()
        self._lv_btn.setChecked(True)
        self._lv_enter_contour()
        self._lv_rebuild_measures()
        self._lv_apply_target(self._lv["pass"])
        self._lv_show_plane()
        self._lv_result_lines = [
            t("Loaded borders: endo {ne} / epi {nep} planes",
              ne=len(model.endo_planes), nep=len(model.epi_planes))]
        # Go straight to the short-axis (left pane) so the loaded borders are
        # shown there immediately and the level/centre lines are draggable.
        if (len(model.endo_contours) >= 3 or len(model.epi_contours) >= 3):
            self._lv_sax_btn.setChecked(True)
            self._lv_toggle_sax()
        # Redisplay a SAVED volume result (after SAX entry, which clears the
        # result panel) and light CalcVol blue — so Load shows the computed value.
        if volume and self._lv is not None:
            lines = [t("Loaded borders: endo {ne} / epi {nep} planes",
                       ne=len(model.endo_planes), nep=len(model.epi_planes))]
            ev = volume.get("endo_ml")
            pv = volume.get("epi_ml")
            if ev is not None:
                lines.append(t("Endo-LV Volume: {v:.1f} mL", v=float(ev)))
                self._lv["vol_endo_ml"] = float(ev)
            if pv is not None:
                lines.append(t("Epi-LV Volume: {v:.1f} mL", v=float(pv)))
                self._lv["vol_epi_ml"] = float(pv)
            if ev is not None or pv is not None:
                self._lv["vol_done"] = True
                self._lv_result_lines = lines
                self._lv_sync_buttons()
        self._lv_update_text()

    def _lv_rebuild_measures(self) -> None:
        """(Re)create the per-plane Endo/Epi border measures on the LV pane from
        the model's stored 3-D borders (used after a Load)."""
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
        """Leave LV mode entirely, restoring the normal MPR view."""
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
                        self._lvv_epi_model_dict = model.to_dict()
        except Exception:                               # noqa: BLE001
            pass
        self._lv_reset_undo()
        # Remove the on-screen LV border traces (kept in the model for volume).
        for k in ("A", "B"):
            self._measures[k] = [m for m in self._measures[k]
                                 if m.get("_lv") is None]
        self._lv = None
        self._lv_result_lines = []
        self._lv_wall = False
        self._lv_wall_btn.setChecked(False)
        self._lv_sax_btn.setChecked(False)  # leave short-axis display
        self._lv_btn.setChecked(False)      # internal mode flag off
        self._lv_sync_buttons()             # reset colours + grey out the bar
        self._lv_plane_lbl.setText("0/6")   # reset the plane counter
        if self._meas_on:
            self._meas_btn.setChecked(False)
            self._toggle_measure()
        self.set_side("Bi")
        self._init_frames()
        self._view_initial = True
        # Restore the crosshair + slab overlay (LV trace mode hid the slab).
        for k in ("A", "B"):
            self.pane[k].set_overlay_visible(self._cl_btn.isChecked())
            self.pane[k].set_slab_visible(True)
        self._lv_update_text()
        # Re-enable the tools/controls that contour LV greyed out (esp. the
        # Slab(mm) spin) — _refresh_tool_availability only ever DISABLED them in
        # LV and nothing re-ran it on the way out, so entering LV Vol afterwards
        # left the slab stuck disabled.
        self._refresh_tool_availability()
        self._refresh(reset_cam=True)
        self._redraw_all_lv()

    def _lv_update_text(self) -> None:
        """Refresh the LV status/result text via the unified result path so the
        volume result and border metrics never fight over the result block."""
        for k in ("A", "B"):
            self._redraw_meas(k)

    def _lv_status_lines(self) -> list:
        """Result-block lines for LV mode: the computed volume result (if any),
        then the current tracing guidance. Empty when not in LV mode."""
        # Blood sub-mode: show the measured volume in the SAME pane result block
        # as Endo/Epi (consistent overlay), labelled "Blood-Volume:".
        lvv = self._lvv
        if lvv is not None:
            if lvv.get("last_ml") is not None:
                return [t("Blood-Volume: {v:.1f} mL", v=float(lvv["last_ml"]))]
            return []
        lv = self._lv
        if lv is None:
            return []
        lines = list(getattr(self, "_lv_result_lines", []))
        pas = lv.get("pass")
        if pas is None:
            lines.append(t("LV — choose Endo or Epi to start a pass"))
            return lines
        name = t("Endo (lumen)") if pas == "endo" else t("Epi (myocardial)")
        if lv.get("phase") == "align":
            lines.append(t("LV [{p} pass] — align the {p} long-axis view, "
                           "then press 'Set axis'", p=name))
            return lines
        if lv.get("phase") == "ready":
            lines.append(t("LV [{p} pass] — axis set. Final Zoom/Move, then "
                           "press 'Trace'", p=name))
            return lines
        if lv.get("phase") == "apex":
            lines.append(t("LV [{p} pass] — click the {p} apex "
                           "(Shift-click to adjust the view first)", p=name))
            return lines
        if lv.get("phase") == "contour":
            m = lv["model"]
            head = (t("tracing Endo (red) — double-click to finish")
                    if pas == "endo"
                    else t("tracing Epi (green) — double-click to finish"))
            lines.append(
                t("LV [{p} pass] — {head}\ncaptured: endo {ne} / epi {nep} "
                  "meridians", p=name, head=head,
                  ne=len(m.endo_contours), nep=len(m.epi_contours)))
        return lines

    @staticmethod
    def _circle_poly(cx, cy, r, n=20):
        """A small closed circle polyline (output coords) used as a direction
        marker at one end of the LV level / centre lines."""
        return [(cx + r * math.cos(2.0 * math.pi * i / n),
                 cy + r * math.sin(2.0 * math.pi * i / n))
                for i in range(n + 1)]

    def _lv_view_half(self, key):
        """(half_width, half_height) of the pane's CURRENTLY VISIBLE region in
        output-mm (tracks zoom), so overlay markers can sit near the visible
        edge instead of the fixed FOV corner."""
        p = self.pane[key]
        ps = float(p.ren.GetActiveCamera().GetParallelScale())   # half-height
        w = max(1, p.canvas.width())
        h = max(1, p.canvas.height())
        return ps * (w / h), ps

    def _lv_view_center(self, key):
        """(cx, cy) of the pane's CURRENTLY VISIBLE region in output-mm — the
        camera focal point (which lives in the reslice output frame). The origin
        (0,0) is the LV axis point, NOT the visible centre once the pane is
        panned/zoomed onto the heart, so overlay handles must be anchored here,
        not at the origin, or they drift off-screen."""
        fp = self.pane[key].ren.GetActiveCamera().GetFocalPoint()
        return float(fp[0]), float(fp[1])

    def _lv_ring_radius(self, key):
        """Output-mm radius of the ○ grab handle for *key* (grows with zoom-out
        so it stays a sensible on-screen size)."""
        return max(2.0, 0.04 * self._lv_view_half(key)[1])

    def _lv_line_clip(self, key, ax0, ay0, ux, uy):
        """Clip the (infinite) line through (ax0, ay0) with direction (ux, uy) to
        the pane's visible rectangle [cx±hw, cy±hh] shrunk by the ○ handle radius.
        Returns ((x0,y0), (x1,y1), grabbed) — the −θ and +θ visible endpoints and
        whether the line actually crosses the rect. If it misses (LV centre panned
        far off-screen), both endpoints collapse to the clamped anchor so the ○
        still shows at the nearest inside corner and stays grabbable."""
        hw, hh = self._lv_view_half(key)
        cx, cy = self._lv_view_center(key)
        mg = 1.8 * self._lv_ring_radius(key)
        xlo, xhi = cx - hw + mg, cx + hw - mg
        ylo, yhi = cy - hh + mg, cy + hh - mg
        if xlo > xhi:
            xlo = xhi = cx
        if ylo > yhi:
            ylo = yhi = cy
        nrm = math.hypot(ux, uy) or 1.0
        ux, uy = ux / nrm, uy / nrm
        INF = 1e18
        fallback = (min(xhi, max(xlo, ax0)), min(yhi, max(ylo, ay0)))
        if abs(ux) > 1e-9:
            ta, tb = (xlo - ax0) / ux, (xhi - ax0) / ux
            txlo, txhi = min(ta, tb), max(ta, tb)
        elif xlo <= ax0 <= xhi:
            txlo, txhi = -INF, INF
        else:
            return fallback, fallback, False
        if abs(uy) > 1e-9:
            ta, tb = (ylo - ay0) / uy, (yhi - ay0) / uy
            tylo, tyhi = min(ta, tb), max(ta, tb)
        elif ylo <= ay0 <= yhi:
            tylo, tyhi = -INF, INF
        else:
            return fallback, fallback, False
        t0, t1 = max(txlo, tylo), min(txhi, tyhi)
        if t0 > t1:                                   # line misses the rect
            return fallback, fallback, False
        return ((ax0 + ux * t0, ay0 + uy * t0),
                (ax0 + ux * t1, ay0 + uy * t1), True)

    def _lv_ring_xy(self, key, ax0, ay0, ux, uy):
        """Output-mm (x, y) of the ○ grab handle — the +θ end of the line's
        visible segment (see _lv_line_clip), so it is ALWAYS on-screen and on the
        drawn line at any zoom / pan."""
        _p0, p1, _ok = self._lv_line_clip(key, ax0, ay0, ux, uy)
        return p1

    def _lv_draw_apex_markers(self, key, p) -> None:
        """Draw ONLY the ACTIVE pass's apex marker (endo=red / epi=green) on the
        long-axis (trace) pane and, while short-axis is shown, on the short-axis
        pane. The inactive pass's apex is hidden (Endo active → no epi apex, and
        vice-versa)."""
        lv = self._lv
        if lv is None or lv["model"].axis is None:
            return
        if key not in (lv.get("pane"), lv.get("sax_pane")):
            return
        tgt = lv.get("pass")
        if tgt == "endo":
            P, rgb = lv["model"].endo_apex, (255, 64, 64)
        elif tgt == "epi":
            P, rgb = lv["model"].epi_apex, (64, 200, 80)
        else:
            return
        if P is None:
            return
        # GLOW when a border point is within the convergence range (it will snap
        # to the apex): brighten + enlarge the marker so "収束します" is obvious.
        glow = self._lv_apex_glow(key)
        if glow:
            rgb = tuple(min(255, int(c) + 130) for c in rgb)
            p.lv_apex_actor.GetProperty().SetPointSize(30.0)
        else:
            p.lv_apex_actor.GetProperty().SetPointSize(15.0)
        p.lv_apex_mapper.SetInputData(
            _lv_pts_pd([self._world3d_to_out(key, P)], [rgb], z=0.9))

    def _lv_apex_range_mm(self, key) -> float:
        """Convergence radius in output-mm = TWICE the apex marker's radius
        (PointSize 15 → ~7.5 px radius) so it reads as a circle twice the marker
        (area ×4). Screen-relative, so it tracks zoom."""
        return self._lv_px_to_mm(key, 15.0)

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
        """Update the apex GLOW from the live cursor while tracing a border: glow
        when the cursor is within the convergence range of the active pass's
        apex. Called from the idle mouse-move path."""
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
            self._redraw_lv(which)
            self.pane[which].render()

    def _lv_apex_clear_glow(self) -> None:
        """Turn the apex glow off (a point was confirmed / mode changed)."""
        if self._lv_apex_hot:
            self._lv_apex_hot = False
            lv = self._lv
            if lv is not None and lv.get("pane") is not None:
                self._redraw_lv(lv["pane"])
                self.pane[lv["pane"]].render()

    def _redraw_lv(self, key) -> None:
        """Draw the base-cut line for the current long-axis plane on the LV pane.
        The endo/epi BORDERS are the user's own traced polylines (kept on screen
        recoloured), not a reconstruction — so nothing is redrawn for them here.
        Re-run on every view change so the base line stays anchored."""
        p = self.pane[key]
        lv = self._lv
        p.lv_pts_mapper.SetInputData(vtkPolyData())     # no pick markers
        p.lv_line_mapper.SetInputData(vtkPolyData())
        p.lv_endo_mapper.SetInputData(vtkPolyData())    # borders = the measures
        p.lv_epi_mapper.SetInputData(vtkPolyData())
        p.lv_wall_mapper.SetInputData(vtkPolyData())    # wall-thickness colour map
        p.lv_apex_mapper.SetInputData(vtkPolyData())    # user apex markers
        p.lv_hi_mapper.SetInputData(vtkPolyData())      # green edited crossing
        self._lv_update_wall_legend()                   # bottom-left colour key
        if lv is None:
            return
        # Apex markers stay visible in EVERY LV phase once an axis exists (so the
        # endo apex remains on screen while the epi pass is being set up).
        if lv["model"].axis is not None:
            self._lv_draw_apex_markers(key, p)
        if lv.get("phase") != "contour" or lv["model"].axis is None:
            return                                       # only markers pre-trace
        ax = lv["model"].axis
        # SHORT-AXIS mode (both panes visible):
        if lv.get("sax") is not None:
            along0 = float(lv["sax"])
            if key == lv.get("sax_pane"):
                # the cross-section pane: endo (red) / epi (green) closed splines
                # through the level×meridian border crossings, with the crossing
                # points themselves marked as fixed-SCREEN-size dots (so they
                # don't grow when the pane is zoomed).
                mark_xy, hi_xy = [], []
                border_sm = {}
                # While a long-axis border VERTEX is being dragged, colour the
                # ONE short-axis crossing that follows it GREEN (of the two yellow
                # dots on the meridian line) so it's clear which point moves.
                edit_which, edit_mu = None, None
                e = getattr(self, "_edit", None)
                if e is not None and e.get("key") == lv.get("pane"):
                    em = self._measures[e["key"]][e["mi"]]
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
                show = self._lv_sax_borders()       # single pass, or both
                for which, mapper in (("endo", p.lv_endo_mapper),
                                      ("epi", p.lv_epi_mapper)):
                    if which not in show:
                        continue
                    sp = lv["model"].short_axis_border_pts(along0, which)
                    if sp is None or len(sp) < 3:
                        continue
                    xy = [self._world3d_to_out(key, P) for P in sp]
                    sm = _smooth_closed(xy)         # closed Catmull-Rom
                    mapper.SetInputData(_polylines_pd([[tuple(q) for q in sm]]))
                    if edit_which == which and edit_mu is not None:
                        bi, bd = None, 1e9
                        for i, P in enumerate(sp):
                            d = np.asarray(P, float) - ax.apex
                            th = math.degrees(math.atan2(
                                float(d @ ax.binormal),
                                float(d @ ax.radial0))) % 360.0
                            dd = min(abs(th - edit_mu), 360.0 - abs(th - edit_mu))
                            if dd < bd:
                                bd, bi = dd, i
                        # Only light the crossing that TRULY corresponds to the
                        # edited meridian (within half a meridian spacing). When
                        # this level has no crossing on that meridian (an
                        # asymmetric trace / apical level where one wall drops
                        # out — the "missing left dot"), highlight NOTHING rather
                        # than a different section-line's point, which read as
                        # "the wrong point got selected".
                        n_pl = max(1, lv["model"].n_planes)
                        if bi is not None and bd <= 90.0 / n_pl:
                            hi_xy.append(xy[bi])        # the following point
                    mark_xy.extend(xy)              # the crossing points
                    border_sm[which] = sm
                # WALL-THICKNESS colour map: translucent annulus between endo &
                # epi, each angular sector coloured by its Epi−Endo gap (the same
                # heatmap Compare uses). Recomputed at every level.
                if (getattr(self, "_lv_wall", False)
                        and "endo" in border_sm and "epi" in border_sm):
                    self._lv_draw_wall(p, border_sm["endo"], border_sm["epi"])
                if mark_xy:
                    # yellow crossing dots, fixed screen size (constant under
                    # zoom); the one following a long-axis edit is overdrawn green
                    # at 2× radius via lv_hi_mapper.
                    p.lv_pts_mapper.SetInputData(
                        _lv_pts_pd(mark_xy, [(255, 210, 0)] * len(mark_xy)))
                if hi_xy:
                    p.lv_hi_mapper.SetInputData(
                        _lv_pts_pd(hi_xy, [(64, 220, 64)] * len(hi_xy), z=0.95))
                # centreline showing the current long-axis plane direction
                # (rotated by ◀ ▶); it lies in this short-axis plane. A ring
                # marks the +θ end so the long/short views share an orientation.
                angs = lv["model"].plane_angles()
                md = ax.meridian_dir(angs[lv["plane_idx"] % len(angs)])
                u, v, _n = self._frame[key]
                dx, dy = float(np.dot(md, u)), float(np.dot(md, v))
                nrm = math.hypot(dx, dy) or 1.0
                dx, dy = dx / nrm, dy / nrm
                X = float(getattr(self, "_half", 100.0))
                # centreline + ○ handle: anchor the meridian line at the VISIBLE
                # CENTRE (focal point), not the axis origin, so it always passes
                # through the middle of the pane → a consistent FULL-SPAN line
                # edge-to-edge with the ○ near one edge (a line through an
                # off-centre origin cut a short, angle-dependent chord — the
                # "lengths vary" report). Direction (dx,dy) still shows the plane.
                cx, cy = self._lv_view_center(key)
                (sx0, sy0), (ex, ey), _ok = self._lv_line_clip(
                    key, cx, cy, dx, dy)
                cr = self._lv_ring_radius(key)
                line = ([(sx0, sy0), (ex, ey)] if _ok
                        else [(cx - dx * X, cy - dy * X),
                              (cx + dx * X, cy + dy * X)])
                p.lv_line_mapper.SetInputData(_polylines_pd([
                    line, self._circle_poly(ex, ey, cr)]))
            elif key == lv.get("pane"):
                # the long-axis pane: the movable LEVEL line ⟂ the axis (a
                # horizontal line at output-y = the current cross-section level).
                # A ring marks the +θ (meridian) end — same direction as the
                # short-axis centreline's ring — kept near the visible edge.
                _, y = self._world3d_to_out(key, ax.apex + along0 * ax.axis)
                X = float(getattr(self, "_half", 100.0))
                cr = self._lv_ring_radius(key)
                # level line + ○ handle, clipped to the visible rect so the ○
                # sits near the visible right edge, on the line, at any zoom / pan.
                (lx0, ly0), (hx, hy), _ok = self._lv_line_clip(
                    key, 0.0, y, 1.0, 0.0)
                line = [(lx0, ly0), (hx, hy)] if _ok else [(-X, y), (X, y)]
                p.lv_line_mapper.SetInputData(_polylines_pd([
                    line, self._circle_poly(hx, hy, cr)]))
            return
        if key != lv.get("pane"):
            return
        # LONG-AXIS view: base-cut line ⟂ the axis at the most-basal common
        # along-level (along maps to output-y, radius to output-x, so it's a
        # horizontal line at y = base_along across the FOV).
        rng = (lv["model"].along_range("endo")
               or lv["model"].along_range("epi"))
        if rng is not None:
            base = rng[1]
            X = float(getattr(self, "_half", 100.0))
            p.lv_line_mapper.SetInputData(
                _polylines_pd([[(-X, base), (X, base)]]))

    def _lv_border_polys(self, key, which, phi):
        """Output-plane polylines of the captured *which* border for the two
        meridians (phi, phi+180) lying in the long-axis plane at rotation phi."""
        ax = self._lv["model"].axis
        store = (self._lv["model"].endo_contours if which == "endo"
                 else self._lv["model"].epi_contours)
        polys = []
        for th in (phi % 360.0, (phi + 180.0) % 360.0):
            prof = None
            for kk, vv in store.items():
                if abs((kk - th + 180.0) % 360.0 - 180.0) < 1e-3:
                    prof = vv
                    break
            if prof is None:
                continue
            pts = [self._world3d_to_out(
                       key, ax.to_world(th, float(r), float(al)))
                   for (al, r) in prof]
            if len(pts) >= 2:
                polys.append(pts)
        return polys

    def _redraw_all_lv(self) -> None:
        for k in ("A", "B"):
            self._redraw_lv(k)
            self.pane[k].render()

    # ---- lumen (high-HU) snapping for vessel tracing ----
    def _hu_at(self, P):
        """Trilinear HU at a single 3-D world point *P*, or None if it falls
        outside the volume."""
        if P is None or self._vol is None:
            return None
        r = self._hu_along(P, np.array([0.0, 0.0, 1.0]), [0.0])
        if r is None:
            return None
        v = float(r[0])
        return None if v <= -1999.0 else v

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

    def _open_setting(self, parent=None, modal=False):
        """Open the HU colour-map editor. *parent*/*modal* let the shell open it
        ON TOP of (and modal to) the Settings popup — otherwise a Settings-modal
        dialog would sit in front of it and block all its controls. A fresh
        dialog is built each time (the viewer's bands are the source of truth,
        so nothing is lost) to avoid a stale parent when opened from Settings."""
        dlg = _ColorMapDialog(self._bands, self._opacity,
                              self._apply_colormap, self._win, self._lvl,
                              self._cmap_smooth_mm, parent or self.window())
        self._cmap_dlg = dlg
        if modal:
            dlg.setWindowModality(Qt.WindowModality.WindowModal)
            dlg.exec()                     # blocks the Settings popup only
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
        if not self._color:                 # show the result immediately
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
        colour mode on (each pane keeps its own ColorMap toggle), persist, or
        re-emit (which would loop)."""
        self._bands = [dict(b) for b in bands]
        self._opacity = float(opacity)
        if smooth_mm is not None:
            self._cmap_smooth_mm = float(smooth_mm)
        if self._cmap_dlg is not None:
            self._cmap_dlg.set_bands(self._bands, self._opacity)
        if self._color:                     # only repaint if colour is showing
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

    def _wire_color_smoothing(self, p, step):
        """(C) Route the colour reslice through the Gaussian when colour mode is
        on and a smoothing strength is set, else straight to the LUT. *step* is
        the reslice output spacing (mm) so the mm strength maps to a pixel std."""
        if self._color and self._cmap_smooth_mm > 0.0:
            # mm strength → output pixels; capped so a fine (zoomed) reslice
            # can't make a huge, slow kernel. The display-resolution reslice
            # already removes the staircase, so this mainly calms voxel noise.
            sd = max(0.01, min(6.0, self._cmap_smooth_mm / max(1e-6, step)))
            p.gauss.SetStandardDeviations(sd, sd, 0.0)
            p.gauss.SetRadiusFactors(2.0, 2.0, 0.0)
            p.gauss.Modified()
            p.colors.SetInputConnection(p.gauss.GetOutputPort())
        else:
            p.colors.SetInputConnection(p.reslice.GetOutputPort())

    def _refresh(self, reset_cam=False, only=None):
        """Rebuild the reslice(s) + overlays. *only* = "A"/"B" reslices JUST that
        pane and skips the other — used for single-pane drags (Move / single Zoom
        / Spin / Thick), where the companion's image can't have changed, so
        re-reslicing it every mouse-move was pure waste. A full _refresh() on
        release repaints both."""
        if self._image is None:
            return
        base_step = max(1e-3, min(self._dims))
        for key in ("A", "B"):
            if only is not None and key != only:
                continue
            p = self.pane[key]
            # Pane A in short-axis (CPR) mode: reslice the cross-section plane
            # with a tight FOV instead of the normal MPR matrix.
            if self._cpr is not None and key == "A":
                step = base_step
                hcpr = self._cpr["half"]
                ncpr = min(_RESLICE_NPX_CAP, max(64, int(2 * hcpr / step) + 1))
                p.reslice.SetResliceAxes(self._cpr_matrix())
                p.reslice.SetOutputSpacing(step, step, step)
                p.reslice.SetOutputOrigin(-hcpr, -hcpr, 0.0)
                p.reslice.SetOutputExtent(0, ncpr - 1, 0, ncpr - 1, 0, 0)
                p.reslice.SetSlabNumberOfSlices(1)
                p.reslice.Modified()
                p.colors.SetLookupTable(self._lut())
                p.colors.Modified()
                self._wire_color_smoothing(p, step)
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
            th = self._thick[key]
            # Fit the camera FIRST on a reset so the zoom-adaptive reslice below
            # samples exactly the fitted view (the ▲ markers are also sized to
            # the camera's parallel scale, so it must be final before _update_
            # cross).
            if reset_cam:
                self._fit_pane(key)
            # --- Zoom-adaptive, display-resolution reslice -------------------
            # Reslice ONLY the VISIBLE viewport (from the camera) at the screen's
            # pixel density, instead of the whole FOV at the voxel grid. This is
            # what keeps the colour band boundaries smooth curves at ANY zoom
            # (a fixed-grid reslice becomes a staircase when magnified). The
            # camera lives in the reslice OUTPUT frame: focal point = visible
            # centre (mm from _pc), ParallelScale = half the visible height (mm).
            dpr = max(1.0, p.canvas.devicePixelRatioF())
            cam = p.ren.GetActiveCamera()
            ps = max(1e-3, cam.GetParallelScale())
            fp = cam.GetFocalPoint()
            fx, fy = float(fp[0]), float(fp[1])
            pw = max(1, int(round(p.canvas.width() * dpr)))
            ph = max(1, int(round(p.canvas.height() * dpr)))
            half_u = ps * pw / ph                       # half visible width (mm)
            # SPIN rolls the camera → the visible rect is rotated in the output
            # (u,v) plane; widen the axis-aligned sampled box to its bounding
            # box so the rolled corners aren't left unsampled. The camera lives
            # in the OUTPUT frame (x=U, y=V), so read the roll straight from its
            # ViewUp's in-plane components — NOT via the world plane axes.
            vup = cam.GetViewUp()
            mag = math.hypot(float(vup[0]), float(vup[1])) or 1.0
            c = abs(float(vup[1])) / mag                # |cos(roll)|
            s = abs(float(vup[0])) / mag                # |sin(roll)|
            box_u = half_u * c + ps * s
            box_v = half_u * s + ps * c
            spacing = max(base_step * 0.05, 2.0 * ps / ph)   # ≈ display pixel
            # Interactive LOD: while a pan / zoom / rotate / centreline drag is
            # live (_lod_drag, set in _gesture_begin), reslice at HALF the linear
            # resolution (¼ the pixels) and sample a thick slab ~3× coarser, then
            # snap back to full quality once on release (_gesture_commit). A 5 mm
            # MIP trace-slab resliced at full display resolution EVERY mouse-move
            # was the "LV-align Move is laggy" cause — the slab multiplies the
            # per-pixel work by its through-plane sample count.
            lod = getattr(self, "_lod_drag", False)
            # While dragging, halve the linear resolution AND cap the reslice at
            # 1024 px/side (vs 2048): a zoomed-out view otherwise samples up to
            # 2048² × slab-slices per pane every mouse-move. Full cap on release.
            npx_cap = _RESLICE_DRAG_NPX_CAP if lod else _RESLICE_NPX_CAP
            if lod:
                spacing *= 2.0
            nu = min(npx_cap, max(64, int(2.0 * box_u / spacing) + 1))
            nv = min(npx_cap, max(64, int(2.0 * box_v / spacing) + 1))
            p.reslice.SetOutputSpacing(spacing, spacing, base_step)
            p.reslice.SetOutputOrigin(fx - box_u, fy - box_v, 0.0)
            p.reslice.SetOutputExtent(0, nu - 1, 0, nv - 1, 0, 0)
            if th > 0 and hasattr(p.reslice, "SetSlabModeToMax"):
                p.reslice.SetSlabModeToMax()
                # Slab depth sampled at the VOXEL pitch (through-plane), not the
                # fine in-plane display pitch. While dragging, take ~1/3 as many
                # samples but keep the SAME total depth (spacing-fraction ×3), so
                # the thick slab stays responsive and snaps sharp on release.
                slab_frac = 3.0 if lod else 1.0
                p.reslice.SetSlabNumberOfSlices(
                    max(1, int(round(th / (base_step * slab_frac))))
                )
                if hasattr(p.reslice, "SetSlabSliceSpacingFraction"):
                    p.reslice.SetSlabSliceSpacingFraction(slab_frac)
            elif hasattr(p.reslice, "SetSlabNumberOfSlices"):
                p.reslice.SetSlabNumberOfSlices(1)
            p.reslice.Modified()
            # Keep the measured-region mask reslice on the SAME plane/output so
            # the red overlay tracks the image (only when a mask is loaded).
            if getattr(self, "_lvv_mask_vol", None) is not None:
                p.reslice_mask.SetResliceAxes(self._matrix(key))
                p.reslice_mask.SetOutputSpacing(spacing, spacing, base_step)
                p.reslice_mask.SetOutputOrigin(fx - box_u, fy - box_v, 0.0)
                p.reslice_mask.SetOutputExtent(0, nu - 1, 0, nv - 1, 0, 0)
                if hasattr(p.reslice_mask, "SetSlabNumberOfSlices"):
                    p.reslice_mask.SetSlabNumberOfSlices(1)
                p.reslice_mask.Modified()
            p.colors.SetLookupTable(self._lut())
            p.colors.Modified()
            self._wire_color_smoothing(p, spacing)
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
                if only is not None and kk != only:
                    continue
                # In CPR, pane A is the cross-section (its overlay is the
                # control-point markers, drawn separately) — don't overwrite it.
                if self._cpr is not None and kk == "A":
                    continue
                self._redraw_geom(kk)
        # LV EF axis + points re-project onto the (possibly moved) planes too.
        if self._lv is not None:
            for kk in ("A", "B"):
                if only is not None and kk != only:
                    continue
                self._redraw_lv(kk)
        # LV Vol Epi-border dots follow plane/zoom changes (data only).
        if self._lvv is not None and getattr(self, "_lvv_epi_show", False):
            self._lvv_show_epi(render=False)

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
        p.meas_off_dash_mapper.SetInputData(vtkPolyData())
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
        apex_sgn = (-1.0 if key == "A" else 1.0) * self._apex_flip.get(key, 1.0)
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
        before = self._view_snapshot()
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
        self._undo_view(before, self._view_snapshot())   # Ctrl+Z / Ctrl+Y

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
        dlg = _AngioAngleDialog(vals[0], vals[1], self.window())
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
        if prev is not None and prev != mode:
            self._undo_clear()           # frames re-derived → drop stale undos
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
        # Rt90/Lt90/Flip-H/Flip-V all work in BOTH 2-D (native slice) and 3-D MPR
        # (they transform the ACTIVE pane's frame), so they stay enabled in both.
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
        new_slice = int(min(max(self._slice2d + step, 0), max(0, nz - 1)))
        if new_slice == self._slice2d:
            return                      # already at the end — nothing to undo
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
            before = self._view_snapshot()
            self._cpr["T"] = M @ self._cpr["T"]
            self._cpr_apply_xform()
            self._view_initial = False
            self._refresh(reset_cam=True)
            self._undo_view(before, self._view_snapshot())
            return
        if self._image is None:
            return
        if self._mode != "2D":
            # 3-D MPR: rotate / mirror the ACTIVE pane by transforming its reslice
            # frame (u, v, n) — the anatomy AND the crosshair turn with it, and it
            # is coordinate-safe (traces are absolute 3-D pts3d and re-derive to
            # the new 2-D, so measurements/volumes are unchanged). Right-handed is
            # preserved (rotations keep n; mirrors flip n).
            k = self._active_pane
            before = self._view_snapshot()
            # The frame flip / rotate mirrors·turns the anatomy about the reslice
            # CENTRE (output 0,0). On its own that throws the view off-centre when
            # the reslice centre isn't at the screen centre — which is exactly the
            # case in the LV long-axis view (reslice centred on the axis, not the
            # crosshair) and after any pan — so Flip-V appeared to flip about an
            # offset line. Pivot INSTEAD about the visible centre: move the camera
            # focal point by the SAME transform, so the world point at the screen
            # centre stays put and the image flips / rotates in place.
            cam = self.pane[k].ren.GetActiveCamera()
            fp = cam.GetFocalPoint()
            ps = cam.GetPosition()
            fx, fy = float(fp[0]), float(fp[1])
            u, v, n = (np.asarray(a, float) for a in self._frame[k])
            if kind == "rt90":          # 90° clockwise
                self._frame[k] = (v, -u, n)
                nfx, nfy = fy, -fx
            elif kind == "lt90":        # 90° counter-clockwise
                self._frame[k] = (-v, u, n)
                nfx, nfy = -fy, fx
            elif kind == "fliph":       # left-right mirror
                self._frame[k] = (-u, v, -n)
                nfx, nfy = -fx, fy
            elif kind == "flipv":       # top-bottom flip
                self._frame[k] = (u, -v, -n)
                nfx, nfy = fx, -fy
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
            cam.SetFocalPoint(nfx, nfy, fp[2])
            cam.SetPosition(nfx, nfy, ps[2])
            self._view_initial = False
            self._refresh(reset_cam=False)
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
        """Spin+ : roll the ACTIVE pane's camera so its centreline (crosshair)
        snaps to the nearest vertical / horizontal. A 45° tie snaps CLOCKWISE.
        Works in 2-D and 3-D (it just rotates the on-screen view; the frame /
        measurements are unchanged)."""
        if self._image is None:
            return
        key = self._active_pane
        ren = self.pane[key].ren
        cam = ren.GetActiveCamera()
        ccx, ccy = self._cc(key)
        th = math.radians(self._cross_ang[key])
        uh = (math.cos(th), math.sin(th))          # a crosshair line's direction

        def crossline_angle():
            """On-screen angle (deg) of the crossline, via the live camera."""
            ren.SetWorldPoint(ccx, ccy, 0.5, 1.0)
            ren.WorldToDisplay()
            a0 = ren.GetDisplayPoint()
            ren.SetWorldPoint(ccx + uh[0], ccy + uh[1], 0.5, 1.0)
            ren.WorldToDisplay()
            a1 = ren.GetDisplayPoint()
            return math.degrees(math.atan2(a1[1] - a0[1], a1[0] - a0[0]))

        sa = crossline_angle()
        # Probe the Roll→display-angle sign at runtime (no render needed) so the
        # snap direction is always correct regardless of VTK's Roll convention.
        cam.Roll(1.0)
        per = ((crossline_angle() - sa + 180.0) % 360.0) - 180.0   # display °/1° roll
        cam.Roll(-1.0)
        if abs(per) < 1e-6:
            return
        # Nearest 90°; a 45° tie rounds DOWN = CLOCKWISE (display CW = decreasing).
        target = math.ceil(sa / 90.0 - 0.5) * 90.0
        delta = ((target - sa + 180.0) % 360.0) - 180.0            # shortest move
        if abs(delta) < 1e-4:
            return
        before = self._view_snapshot()
        cam.Roll(delta / per)                       # snaps the crossline to target
        self._view_initial = False
        self._refresh()
        self._undo_view(before, self._view_snapshot())

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
        # LV short-axis: don't let Rotate/Spin tilt the locked derived frames.
        if self._lv_sax_active() and t in ("ROTATE", "SPIN"):
            return
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

    def _drag(self, which, dx, dy, shift=False, sx=None, sy=None, ctrl=False):
        if self._image is None:
            return
        t = self._tool
        # LV short-axis is a DERIVED view: a Paging drag moves the cross-section
        # LEVEL; ROTATE is blocked because tilting the reslice FRAME would
        # corrupt the locked short-axis / long-axis geometry (use ◀ ▶ to rotate
        # the reference centreline instead). SPIN is allowed — it only rolls the
        # camera (the image AND the overlay rotate together), leaving the frame
        # and the reconstructed data untouched.
        if self._lv_sax_active():
            if t == "PAGING":
                self._view_initial = False
                self._lv_drag_level(dy)
                return
            if t == "ROTATE":
                return
        # Long-axis view after Set axis: the axis is locked, so Rotate/Spin/Thick
        # (which would re-tilt the reslice frame) are blocked. Zoom/Move/WL/Paging
        # still work (they don't change the axis relationship).
        if self._lv_axis_locked() and t in ("ROTATE", "SPIN", "THICK"):
            return
        if t != "WL":
            self._view_initial = False
        # This drag changes the view (W/L included, now captured in the
        # snapshot) → its whole gesture becomes one Ctrl+Z step.
        self._gesture_moved = True
        # Single-pane tools (Move / Spin / Thick / single-pane Zoom) leave the
        # companion pane untouched → reslice ONLY the dragged pane each move.
        only_pane = None
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
            only_pane = which
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
            only_pane = which                       # SPIN rolls only this pane
        elif t == "ZOOM":
            # Shift = zoom BOTH panes together, else just this one — EXCEPT while
            # actively tracing a border, where a plain left-drag is taken by the
            # trace so Shift is the "run the tool" gate: there Shift alone zooms
            # only THIS pane (individual L/R zoom), and Ctrl+Shift zooms both.
            # Drag (and arrow) UP = zoom OUT (shrink), DOWN = zoom IN (enlarge):
            # dy<0 (up) → factor>1 → larger ParallelScale → wider view = shrink.
            factor = 1.0 - dy * 0.005
            # Individual (single-pane) Shift-zoom while tracing a border OR
            # anywhere in LV mode (align/SAX/trace) so the two panes can be zoomed
            # independently; Ctrl+Shift zooms both. Outside LV, Shift = both.
            indiv = ((self._meas_on and bool(self._meas_type))
                     or self._lv is not None)
            both = (shift and ctrl) if indiv else shift
            keys = ("A", "B") if both else (which,)
            only_pane = None if both else which     # single-pane zoom → skip other
            for k in keys:
                cam = self.pane[k].ren.GetActiveCamera()
                cam.SetParallelScale(
                    max(1e-3, cam.GetParallelScale() * factor)
                )
        elif t == "MOVE":
            only_pane = which
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
        self._refresh(only=only_pane)

    def _wheel(self, which, delta):
        if self._image is None:
            return
        # LV short-axis: the wheel pages the cross-section LEVEL along the axis.
        if self._lv_sax_active():
            self._lv_step_level(1 if delta > 0 else -1)
            return
        if self._mode == "2D":
            self._page2d(1 if delta > 0 else -1)
            return
        # Same contract as the PAGING tool: page the wheeled pane itself —
        # the visible image scrolls through slices. Step C and THIS pane's
        # reslice origin together along this pane's normal. Wheel-up moves
        # toward the other pane's ▲ apex (same sign derivation as PAGING so
        # a crossline-flipped normal can't reverse it).
        before = self._view_snapshot()
        _, _, n = self._axes_for(which)
        other = "B" if which == "A" else "A"
        s = 1.0 if float(np.dot(n, self._apex_dir3(other))) >= 0.0 else -1.0
        mv = n * (3 if delta > 0 else -3) * s * min(self._dims)
        self._center = self._center + mv
        self._pc[which] = self._pc[which] + mv
        self._clamp_center()
        self._view_initial = False
        self._refresh()
        self._undo_view(before, self._view_snapshot())

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
        # A double-click / single-click recentre records its own undo; a
        # crosshair-CENTRE *drag* is committed by the drag gesture instead (from
        # its mouse-press snapshot), so skip self-recording then to keep it one
        # Ctrl+Z step. The distinguishing factor is whether a drag actually moved.
        before = (None if getattr(self, "_gesture_moved", False)
                  else self._view_snapshot())
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
        if before is not None:
            self._undo_view(before, self._view_snapshot())

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
        _ou, _ov, on = self._frame[other]
        new_n = _norm(np.cross(crossdir, n))            # companion plane normal
        if float(np.dot(new_n, on)) < 0.0:
            new_n = -new_n                              # keep the viewing side stable
        # Companion UP = which's through-plane axis n (the axis SHARED by the two
        # planes). It always lies in the companion plane (n ⟂ new_n) and is FIXED
        # while the crossline rotates, so the companion's image can't drift or
        # lock over many turns — the old code projected the PREVIOUS up each step,
        # which parallel-transported an accumulating error and folded (snapping
        # the image "stuck" once the up passed the plane normal). n is also the
        # anatomically-correct vertical for an orthogonal companion (e.g. axial's
        # S–I becomes the sagittal/coronal up).
        v_new = _norm(n - float(np.dot(n, new_n)) * new_n)
        if float(np.linalg.norm(v_new)) < 1e-6:         # n ⟂ new plane (unreachable)
            v_new = _norm(np.cross(new_n, crossdir))
        u_new = _norm(np.cross(v_new, new_n))           # u×v = new_n (ortho convention)
        self._frame[other] = (u_new, v_new, new_n)
        self._cross_ang[other] = math.degrees(math.atan2(
            float(np.dot(crossdir, v_new)), float(np.dot(crossdir, u_new))))
        self._pc[other] = self._center.copy()

    def _rotate_companion_by(self, which, d_deg) -> None:
        """Incremental companion coupling for a crossLINE ROTATE: the crossline
        turns by *d_deg* in *which*'s plane, so the companion plane turns by the
        same amount AROUND the shared axis n (= which's normal). Rotating the
        companion frame RIGIDLY around n keeps its CURRENT orientation (no snap
        to a fresh derivation) and, because n is the rotation axis, holds the
        companion's no-▲ centreline (which lies along n) fixed while its image
        turns about it — with no parallel-transport drift/lock over many turns.
        The crossline mark rides with the frame (its frame-relative angle is
        unchanged)."""
        if abs(float(d_deg)) < 1e-9:
            return
        other = "B" if which == "A" else "A"
        n = _norm(self._frame[which][2])
        u, v, _nn = self._frame[other]
        u2 = _rotate(u, n, d_deg)
        v2 = _rotate(v, n, d_deg)
        self._frame[other] = self._ortho(u2, v2)
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
        caught line/mode also lock the vivid highlight for the whole drag.

        The crosshair grab wins over EVERY tool — including SPIN: a press that
        lands on a crossline rotates/moves that line (and couples the other pane
        via the now-smooth _rotate_companion_by), while a press OFF the crosshair
        falls through to SPIN (whole-pane roll, other pane untouched)."""
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
        self._gesture_moved = True             # centreline drag = one Ctrl+Z step
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
        # Turn the companion by the SAME increment about the shared axis, from
        # its CURRENT orientation — no snap to a fresh derivation, no drift/lock
        # over many turns, and the companion's no-▲ line (the shared axis) stays
        # put while its image rotates about it. (see _rotate_companion_by)
        self._rotate_companion_by(which, d)
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
            before = self._view_snapshot()
            self._win, self._lvl = self._win0, self._lvl0
            self._refresh()
            self._undo_view(before, self._view_snapshot())   # Ctrl+Z / Ctrl+Y

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
            before = self._view_snapshot()
            self._win, self._lvl = (float(x) for x in CT_WL_PRESETS[name])
            self._refresh()
            self._undo_view(before, self._view_snapshot())   # Ctrl+Z / Ctrl+Y

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
