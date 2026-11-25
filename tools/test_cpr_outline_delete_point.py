# -*- coding: utf-8 -*-
"""The right-click menu on a CPR line being edited (「この線を Edit」) has
「点の削除」 right below 「点の追加」 (user 2026-11-25): it deletes the edit
point nearest the click and the short-axis follows. Real VTK CTViewer on
synthetic data (made-up patient); the menu is answered by a patch.

    python tools/test_cpr_outline_delete_point.py
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
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

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
from PyQt6.QtWidgets import QMenu  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


sx, sy, sz = 0.4, 0.4, 0.5
nz, ny, nx = 160, 200, 200
z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
X, Y, Z = x * sx, y * sy, z * sz
vol = np.full((nz, ny, nx), -80.0, np.float32)
cen = [(10 + t, 40, 30) for t in np.arange(0, 30, 0.2)]
cen += [(40 + 15 * np.sin(a), 40 + 15 * (1 - np.cos(a)), 30 + 6 * a)
        for a in np.arange(0, 2.0, 0.2 / 15)]
for c in cen:
    r = 1.5
    sl = (slice(int((c[2] - r - 1) / sz), int((c[2] + r + 1) / sz) + 1),
          slice(int((c[1] - r - 1) / sy), int((c[1] + r + 1) / sy) + 1),
          slice(int((c[0] - r - 1) / sx), int((c[0] + r + 1) / sx) + 1))
    m = (X[sl] - c[0]) ** 2 + (Y[sl] - c[1]) ** 2 + (Z[sl] - c[2]) ** 2 <= r * r
    vol[sl][m] = 380
vol += np.random.default_rng(0).normal(0, 25, vol.shape).astype(np.float32)
s = LoadedSeries(modality=Modality.CT, volume=vol.astype(np.int16), spacing_mm=(0.4, 0.4), cine_fps=None,
                 window=800.0, level=200.0, slice_mm=0.5, patient_basis=None, series_uid="live-uid")
v = ctv.CTViewer(); v.resize(1400, 800); v.load_series(s, "live"); v._set_mode("3D")
app.processEvents()
v._coronary_mode = True
p3 = [[15.0, 40, 30], [22.0, 40, 30]]
pts = [tuple(map(float, v._world3d_to_out("B", np.array(q)))) for q in p3]
v._measures["B"].append({"id": 77, "type": "polyline", "pts": pts, "pts3d": p3, "color": "#ffff00"})
mi = len(v._measures["B"]) - 1
check("(setup) short-axis built", v._cpr_ensure_built(("B", mi)) and v._cpr is not None)
v._measures["B"][mi]["pts3d"] = [[15.0, 40, 30], [22.0, 40, 30],
                                 [30.0, 40, 30], [40.0, 41.0, 31.0]]
v._measures["B"][mi]["pts"] = [
    tuple(map(float, v._world3d_to_out("B", np.array(q))))
    for q in v._measures["B"][mi]["pts3d"]]
v._cpr_rebuild()
v.show()
for _ in range(10):
    app.processEvents()
m = v._measures["B"][mi]
n0 = len(m["pts"])
target = list(map(float, m["pts3d"][2]))
sx, sy = v._world_to_qt("B", float(m["pts"][2][0]), float(m["pts"][2][1]))
seen = {}


def _exec(self, *a, **k):
    acts = [x for x in self.actions() if not x.isSeparator()]
    seen["texts"] = [x.text() for x in acts]
    for x in acts:
        if x.text() == t("Delete point"):
            return x
    return None


_orig = QMenu.exec
QMenu.exec = _exec
v._outline_right("B", mi, sx + 3, sy + 2)
QMenu.exec = _orig
tx = seen.get("texts", [])
i_add = tx.index(t("Add point")) if t("Add point") in tx else -1
check("「点の削除」 right below 「点の追加」",
      i_add >= 0 and i_add + 1 < len(tx) and tx[i_add + 1] == t("Delete point"),
      tx)
m = v._measures["B"][mi]
check("the edit point nearest the click is deleted",
      len(m["pts"]) == n0 - 1
      and all(not np.allclose(p, target) for p in m["pts3d"]),
      (n0, len(m["pts"])))
check("the short-axis follows (still on this line)",
      v._cpr is not None and v._cpr_is_src("B", mi))
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
app.processEvents()
os._exit(1 if fails else 0)
