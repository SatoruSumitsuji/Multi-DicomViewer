# -*- coding: utf-8 -*-
"""The point probe's HU on the SHORT AXIS (user 2026-11-26): the HU shown
next to the cursor on the left (short-axis) pane is read on that section —
it used to be read on the MPR plane of pane A at the same screen spot
(a bright lumen read 「HU 26」).
Real VTK CTViewer; the volume's HU rises steadily with x / y / z (made-up
data), so each position has its own value.

    python tools/test_cpr_short_axis_hu.py
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
QMessageBox.question = staticmethod(
    lambda *a, **k: QMessageBox.StandardButton.Yes)
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.viewers import ct_viewer as ctv  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


sx, sy, sz = 0.5, 0.5, 0.5
nz, ny, nx = 120, 120, 120
z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
vol = (10.0 * x * sx + 4.0 * y * sy + 1.0 * z * sz).astype(np.int16)
ser = LoadedSeries(modality=Modality.CT, volume=vol, spacing_mm=(sx, sy),
                   cine_fps=None, window=800.0, level=200.0, slice_mm=sz,
                   patient_basis=None, series_uid="sa-hu-uid")
v = ctv.CTViewer()
v.resize(1400, 800)
v.load_series(ser, "sahu")
v._set_mode("3D")
v.show()
app.processEvents()
v._coronary_mode = True
# an oblique line, so the short axis is NOT the MPR plane of pane A
p3 = [[20.0, 22.0, 18.0], [30.0, 31.0, 30.0], [40.0, 36.0, 42.0]]
pts = [tuple(map(float, v._world3d_to_out("B", np.array(q)))) for q in p3]
v._measures["B"].append({"id": 5, "type": "polyline", "pts": pts,
                         "pts3d": [list(q) for q in p3], "color": "#ffff00"})
v._enter_cpr("B", len(v._measures["B"]) - 1)
for _ in range(5):
    app.processEvents()
check("short axis open", v._cpr is not None)

cv = v.pane["A"].canvas
probes = [(cv.width() * 0.5 + 40, cv.height() * 0.5 + 25),
          (cv.width() * 0.5 - 30, cv.height() * 0.5 - 35)]
ok_all, bad_any = True, False
for (qx, qy) in probes:
    v._measure_hover("A", qx, qy)
    txt = v.pane["A"].hover_hu.GetInput()
    du, dv = v._disp_to_world("A", qx, qy)
    o, u, vv, _n = v._cpr_frame()
    want = v._hu_at(np.asarray(o) + du * np.asarray(u) + dv * np.asarray(vv))
    old = v._hu_at(v._out_to_world3d("A", du, dv))
    ok_all &= txt == f"HU {want:.0f}"
    bad_any |= abs(want - old) > 5
    print("   probe", (round(qx), round(qy)), txt, "want", round(want),
          "MPR-plane value", None if old is None else round(old))
check("the HU shown is the value ON the short-axis section", ok_all)
check("(and it differs from the MPR plane's — the old reading)", bad_any)

# the Mac viewer reads it the same way (source check: pygfx is not here)
src = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(
    "utf-8")
i = src.find("def _measure_hover(")
check("Mac viewer: the short axis reads its own section",
      "_cpr_offset_to_world3d(" in src[i:i + 1200]
      and "_cpr_screen_to_offset(sx, sy)" in src[i:i + 1200])

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
app.processEvents()
os._exit(1 if fails else 0)
