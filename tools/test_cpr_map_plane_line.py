# -*- coding: utf-8 -*-
"""CPR point setting (user 2026-11-25): the right (long-axis) pane's cutting
plane is drawn as a line on the left short-axis section, so a point set /
moved on the right pane shows which way it goes on the left. The line lies in
BOTH planes and follows the right pane when it is paged / turned.

Real VTK CTViewer on synthetic data (made-up patient).

    python tools/test_cpr_map_plane_line.py
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
                 series_uid="map-line-uid")
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
    return [o + a * u + b * vv for (a, b) in segs[0]]


def in_plane_b(P):
    _u, _v, nb = (np.asarray(a, float) for a in v._axes_for("B"))
    nb = nb / np.linalg.norm(nb)
    return abs(float(nb @ (P - np.asarray(v._pc["B"], float))))


S = seg3d()
check("a line is given on the short axis", S is not None)
check("both ends lie in the right pane's plane",
      S is not None and max(in_plane_b(P) for P in S) < 1e-6,
      None if S is None else [in_plane_b(P) for P in S])
o, u, vv, n = (np.asarray(a, float) for a in v._cpr_frame())
check("… and in the section's plane",
      S is not None and max(abs(float(n @ (P - o))) for P in S) < 1e-6)
_bu, _bv, nb = (np.asarray(a, float) for a in v._axes_for("B"))
d = (S[1] - S[0]) / np.linalg.norm(S[1] - S[0]) if S is not None else None
check("its direction is the in-plane direction of the right pane",
      d is not None and abs(float(d @ nb)) < 1e-6
      and abs(float(d @ n)) < 1e-6)

# VTK: drawn in the short-axis overlay with its own colour
v._draw_cpr_ctrl_markers()
pd = v.pane["A"].meas_pts_off_mapper.GetInput()
cols = pd.GetCellData().GetScalars() if pd is not None else None
rgb = []
if cols is not None:
    for i in range(cols.GetNumberOfTuples()):
        rgb.append(tuple(int(c) for c in cols.GetTuple4(i))[:3])
check("VTK overlay carries the cyan plane line",
      tuple(v._CPR_MAP_LINE_RGB) in rgb, rgb)

# the right pane paged 2 mm along its normal → the line follows
nbu = nb / np.linalg.norm(nb)
v._pc["B"] = np.asarray(v._pc["B"], float) + 2.0 * nbu
S2 = seg3d()
check("paging the right pane moves the line (still in its plane)",
      S2 is not None and max(in_plane_b(P) for P in S2) < 1e-6
      and abs(abs(float(nbu @ (S2[0] - S[0]))) - 2.0) < 1e-6)

# parallel planes → no line
fr = v._frame["B"]
v._frame["B"] = (fr[0], fr[1], np.asarray(v._cpr_frame()[3], float))
check("no line when the right pane is parallel to the section",
      v._cpr_map_plane_section_2d() == [])
v._frame["B"] = fr

# a VR right pane → no line
v._vr_on["B"] = True
check("no line while the right pane is VR", v._cpr_map_plane_section_2d() == [])
v._vr_on["B"] = False

mac = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(
    encoding="utf-8")
check("Mac draws it too (dashed)",
      "v._cpr_map_plane_section_2d()" in mac)
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
os._exit(1 if fails else 0)
