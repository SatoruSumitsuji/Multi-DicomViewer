# -*- coding: utf-8 -*-
"""The image the user is working on does not jump (user 2026-11-19).

* confirming a CPR line keeps the map pane B's zoom / pan;
* Auto-Track: pane B follows each new point (turns with the course);
* Resume trace after Auto-Track (right-click the distal point) keeps pane B
  where Auto-Track left it (it used to jump back to the proximal view the
  short-axis was built on) and brings pane A's MPR there too.

Real VTK CTViewer on a synthetic curved vessel; dialogs / menus mocked.

    python tools/test_cpr_keep_view.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
# The VTK viewer needs a real GL context (offscreen has none on Windows);
# nothing is shown.
os.environ["QT_QPA_PLATFORM"] = ("windows" if sys.platform == "win32"
                                 else os.environ.get("QT_QPA_PLATFORM", "offscreen"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMenu, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
for nm in ("information", "warning", "critical"):
    setattr(QMessageBox, nm, staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Ok))
QMessageBox.question = staticmethod(
    lambda *a, **k: QMessageBox.StandardButton.Yes)
QMessageBox.exec = lambda self: 0
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.viewers import ct_viewer as ctv  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


# a straight then curving vessel (380 HU, r 1.5 mm) in -80 HU
sx, sy, sz = 0.4, 0.4, 0.5
nz, ny, nx = 160, 200, 200
z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
X, Y, Z = x * sx, y * sy, z * sz
vol = np.full((nz, ny, nx), -80.0, np.float32)
cen = [(10 + s_, 40, 30) for s_ in np.arange(0, 30, 0.2)]
cen += [(40 + 15 * np.sin(a), 40 + 15 * (1 - np.cos(a)), 30 + 6 * a)
        for a in np.arange(0, 2.0, 0.2 / 15)]
for c in cen:
    r = 1.5
    sl = (slice(int((c[2] - r - 1) / sz), int((c[2] + r + 1) / sz) + 1),
          slice(int((c[1] - r - 1) / sy), int((c[1] + r + 1) / sy) + 1),
          slice(int((c[0] - r - 1) / sx), int((c[0] + r + 1) / sx) + 1))
    msk = ((X[sl] - c[0]) ** 2 + (Y[sl] - c[1]) ** 2
           + (Z[sl] - c[2]) ** 2 <= r * r)
    vol[sl][msk] = 380
vol += np.random.default_rng(0).normal(0, 25, vol.shape).astype(np.float32)
ser = LoadedSeries(modality=Modality.CT, volume=vol.astype(np.int16),
                   spacing_mm=(0.4, 0.4), cine_fps=None, window=800.0,
                   level=200.0, slice_mm=0.5, patient_basis=None,
                   series_uid="keep-view-uid")
ctv.CTViewer._AUTOTRACK_STEP_S = 0.0
v = ctv.CTViewer()
v.resize(1400, 800)
v.load_series(ser, "keep")
v._set_mode("3D")
app.processEvents()
v._coronary_mode = True
p3 = [[15.0, 40, 30], [22.0, 40, 30]]


def add_line():
    pts = [tuple(map(float, v._world3d_to_out("B", np.array(q)))) for q in p3]
    v._measures["B"].append({"id": 77, "type": "polyline", "pts": pts,
                             "pts3d": [list(q) for q in p3],
                             "color": "#ffff00"})
    return len(v._measures["B"]) - 1


def mi_of():
    return next(i for i, q in enumerate(v._measures["B"]) if q.get("id") == 77)


def state_b():
    c = v.pane["B"].ren.GetActiveCamera()
    return (np.round(v._pc["B"], 2).tolist(),
            np.round(c.GetFocalPoint()[:2], 2).tolist(),
            round(c.GetParallelScale(), 2),
            [np.round(a, 3).tolist() for a in v._frame["B"]])


# the user zooms / pans pane B, then confirms the line
mi = add_line()
cam = v.pane["B"].ren.GetActiveCamera()
cam.SetParallelScale(cam.GetParallelScale() * 0.6)
fp, po = cam.GetFocalPoint(), cam.GetPosition()
cam.SetFocalPoint(fp[0] + 5, fp[1] + 3, fp[2])
cam.SetPosition(po[0] + 5, po[1] + 3, po[2])
v._view_initial = False
s0 = state_b()
v._enter_cpr("B", mi)
app.processEvents()
check("confirm: pane B keeps its zoom / pan", state_b() == s0,
      (s0, state_b()))
# Auto-Track: pane B pans to each point, its orientation unchanged
fr0 = state_b()[3]
out = v._cpr_autotrack("B", mi_of())
app.processEvents()
tip = np.asarray(v._measures["B"][mi_of()]["pts3d"][-1], float)
check("Auto-Track ran distally", out is not None and out["trunk_mm"] > 20,
      out and out["trunk_mm"])
check("Auto-Track: pane B follows the last edit point (turns with the "
      "course, centred on the distal point)",
      state_b()[3] != fr0 and np.linalg.norm(v._pc["B"] - tip) < 1e-6,
      (state_b()[3], fr0))
# Resume trace from the distal point (the handle menu)
s1 = state_b()


def _exec(self, *a, **k):
    for act in self.actions():
        if act.text() == t("Resume trace"):
            return act
    return None


QMenu.exec = _exec
mi = mi_of()
v._handle_right("B", (mi, len(v._measures["B"][mi]["pts"]) - 1), 10, 10)
app.processEvents()
check("Resume after Auto-Track: pane B stays where it was (distal)",
      state_b() == s1 and v._draft is not None, (s1, state_b()))
check("Resume after Auto-Track: pane A's MPR is at the distal point too",
      np.linalg.norm(v._pc["A"] - v._pc["B"]) < 1e-6
      and np.linalg.norm(v._center - v._pc["B"]) < 1e-6,
      (np.round(v._pc["A"], 1), np.round(v._pc["B"], 1)))

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
