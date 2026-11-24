# -*- coding: utf-8 -*-
"""Right-click on the distal edit point's short-axis (user 2026-11-25):
the menu offers 「Auto-Track 再開」, which resumes Auto-Track from the point
AS IT IS — the right-click no longer moves it (its position is adjusted
with a left-click beforehand). The marker stays put while the menu is open.

Real VTK CTViewer on synthetic data (made-up patient); QMenu.exec patched.

    python tools/test_cpr_center_preview.py
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
app.processEvents()

# go to the distal edit point's section
idxs = v._cpr_ctrl_indices()
v._cpr["idx"] = int(idxs[-1])
v._cpr_sync_bar()
v._refresh()
app.processEvents()
v._cpr_autotrack_center_ok = lambda: True        # (Coronary Tree work)
orig = [float(q) for q in v._cpr_ctrl_pts3d()[-1]]
cw, ch = v.pane["A"].canvas.width(), v.pane["A"].canvas.height()
csx, csy = cw / 2 + 25, ch / 2 - 15              # off the current point
seen = {}


def menu_cancel(self, *a, **k):
    seen["texts"] = [x.text() for x in self.actions()]
    seen["during"] = [float(q) for q in v._cpr_ctrl_pts3d()[-1]]
    return None


QMenu.exec = menu_cancel
v._cpr_context_menu(csx, csy)
check("menu offers 「Auto-Track 再開」 (no 「ここを中心に」)",
      t("Auto-Track 再開") in seen.get("texts", [])
      and not any("ここを中心" in x for x in seen.get("texts", [])),
      seen.get("texts"))
check("the right-click does not move the marker",
      np.allclose(seen.get("during"), orig)
      and np.allclose([float(q) for q in v._cpr_ctrl_pts3d()[-1]], orig))

called = {}
v._cpr_autotrack = lambda which, mi_, guided=False: called.update(
    which=which, mi=mi_, guided=guided,
    end=[float(q) for q in v._cpr_ctrl_pts3d()[-1]]) or None


def menu_pick(self, *a, **k):
    return next(x for x in self.actions()
                if x.text() == t("Auto-Track 再開"))


QMenu.exec = menu_pick
v._cpr_context_menu(csx, csy)
check("「Auto-Track 再開」 resumes (guided) from the point as it is",
      called.get("guided") is True and called.get("mi") == mi
      and np.allclose(called.get("end"), orig), called)

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
app.processEvents()
os._exit(1 if fails else 0)
