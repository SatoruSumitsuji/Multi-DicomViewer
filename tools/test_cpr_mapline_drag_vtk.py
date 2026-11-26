# -*- coding: utf-8 -*-
"""Drag the right pane's plane line on the short axis — real VTK CTViewer
(user 2026-11-26): mouse press / move / release ON the light-blue line turn
the right pane 90°; with Measure on (橙) only with Alt (a plain click there
moves the edit point, as before). Synthetic data (made-up patient).

    python tools/test_cpr_mapline_drag_vtk.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
os.environ["QT_QPA_PLATFORM"] = ("windows" if sys.platform == "win32"
                                 else os.environ.get("QT_QPA_PLATFORM", "offscreen"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
for nm in ("information", "warning", "critical"):
    setattr(QMessageBox, nm, staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Ok))
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.viewers import ct_viewer as ctv  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


sx, sy, sz = 0.5, 0.5, 0.5
nz, ny, nx = 100, 120, 120
z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
X, Y, Z = x * sx, y * sy, z * sz
vol = np.full((nz, ny, nx), -80.0, np.float32)
vol[(Y - 30) ** 2 + (Z - 25) ** 2 <= 1.5 ** 2] = 380      # vessel along x
s = LoadedSeries(modality=Modality.CT, volume=vol.astype(np.int16),
                 spacing_mm=(0.5, 0.5), cine_fps=None, window=800.0,
                 level=200.0, slice_mm=0.5, patient_basis=None,
                 series_uid="map-line-drag-uid")
v = ctv.CTViewer()
v.resize(1400, 800)
v.load_series(s, "t")
v._set_mode("3D")
app.processEvents()
v._coronary_mode = True
p3 = [[15.0, 30.0, 25.0], [25.0, 30.0, 25.0], [35.0, 30.0, 25.0]]
pts = [tuple(map(float, v._world3d_to_out("B", np.array(q)))) for q in p3]
v._measures["B"].append({"id": 5, "type": "polyline", "pts": pts,
                         "pts3d": p3, "color": "#ffff00"})
mi = len(v._measures["B"]) - 1
check("(setup) short-axis built",
      v._cpr_ensure_built(("B", mi)) and v._cpr is not None
      and v._cpr.get("src") == "B")
app.processEvents()


def seg3d():
    segs = v._cpr_map_plane_section_2d()
    if not segs:
        return None
    o, u, vv, _n = (np.asarray(a, float) for a in v._cpr_frame())
from PyQt6.QtCore import QPointF, Qt, QEvent  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402
cv = v.pane["A"].canvas
cv.setFocus()
app.processEvents()
segs = v._cpr_map_plane_section_2d()
(a0, b0), (a1, b1) = segs[0]
# a screen point ON the line: world→display via the renderer
ren = v.pane["A"].ren
dpr = cv.devicePixelRatioF()


def w2s(wx, wy):
    ren.SetWorldPoint(float(wx), float(wy), 0.0, 1.0)
    ren.WorldToDisplay()
    dx, dy, _ = ren.GetDisplayPoint()
    return dx / dpr, cv.height() - dy / dpr


L = np.hypot(a1 - a0, b1 - b0)
fx, fy = (a1 - a0) / L, (b1 - b0) / L
p_on = w2s(8 * fx, 8 * fy)
nb0 = np.asarray(v._axes_for("B")[2], float).copy()


def ev(t, x, y, mods=Qt.KeyboardModifier.NoModifier, btn=Qt.MouseButton.LeftButton):
    btns = btn if t != QEvent.Type.MouseButtonRelease else Qt.MouseButton.NoButton
    return QMouseEvent(t, QPointF(x, y), QPointF(x, y), btn, btns, mods)


mx = my = 0.0


def drag(mods):
    cv.mousePressEvent(ev(QEvent.Type.MouseButtonPress, *p_on, mods))
    tgt = w2s(mx + 8 * -fy, my + 8 * fx)          # 90° round the pivot
    mid = w2s(mx + 8 * (fx - fy) / 1.414, my + 8 * (fy + fx) / 1.414)
    for q in (mid, tgt):
        cv.mouseMoveEvent(ev(QEvent.Type.MouseMove, *q, mods))
    cv.mouseReleaseEvent(ev(QEvent.Type.MouseButtonRelease, *tgt, mods))
    app.processEvents()
    nb = np.asarray(v._axes_for("B")[2], float)
    return np.degrees(np.arccos(min(1.0, abs(float(nb @ nb0)))))


v._meas_on = False
a = drag(Qt.KeyboardModifier.NoModifier)
check("VTK, Measure off: dragging the line turns the right pane 90°",
      abs(a - 90) < 3, a)
nb0 = np.asarray(v._axes_for("B")[2], float).copy()
segs = v._cpr_map_plane_section_2d()
(a0, b0), (a1, b1) = segs[0]
L = np.hypot(a1 - a0, b1 - b0)
fx, fy = (a1 - a0) / L, (b1 - b0) / L
p_on = w2s(5 * fx, 5 * fy)
v._meas_on = True
a = drag(Qt.KeyboardModifier.NoModifier)
check("VTK, 橙 without Alt: not turned", a < 1, a)
def reset():
    global nb0, fx, fy, p_on, mx, my
    nb0 = np.asarray(v._axes_for("B")[2], float).copy()
    (a0, b0), (a1, b1) = v._cpr_map_plane_section_2d()[0]
    L = np.hypot(a1 - a0, b1 - b0)
    fx, fy = (a1 - a0) / L, (b1 - b0) / L
    mx, my = (a0 + a1) / 2.0, (b0 + b1) / 2.0
    p_on = w2s(mx + 8 * fx, my + 8 * fy)


reset()
a = drag(Qt.KeyboardModifier.AltModifier)
check("VTK, 橙 with Alt: turned 90°", abs(a - 90) < 3, a)
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
os._exit(1 if fails else 0)
