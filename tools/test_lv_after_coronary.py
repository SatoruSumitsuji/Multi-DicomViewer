# -*- coding: utf-8 -*-
"""LV after CoronaryTree (user 2026-11-25): with a coronary tree loaded and no
LV data yet, the LV row (MV-AoV+Apex …) is greyed only IN CoronaryTreeモード —
back in Simple View the LV analysis can start. A 「Load」 button right of
MV-AoV+Apex opens a .FullLv.json like a drop. Real VTK CTViewer on synthetic
data (made-up patient); the file dialog and the shell are fakes.

    python tools/test_lv_after_coronary.py
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
v.show()
for _ in range(5):
    app.processEvents()
v._coro_overlay_vr = {"vessels": []}          # a docked tree drives this CT
v._ct_mode_sel = "coronary"
v._lv_update_submode_ui()
check("CoronaryTreeモード with a tree and no LV: LV row greyed (as before)",
      not v._lv_mv_btn.isEnabled())
v._ct_mode_sel = "standard"
v._lv_update_submode_ui()
check("Simple View with the tree kept: MV-AoV+Apex usable",
      v._lv_mv_btn.isEnabled() and v._lv_full_load_btn.isEnabled())
app.processEvents()
gm = v._lv_full_load_btn.mapTo(v, v._lv_full_load_btn.rect().topRight())
gl = v._lv_mv_btn.mapTo(v, v._lv_mv_btn.rect().topLeft())
check("Load sits LEFT of MV-AoV+Apex (next to it)",
      v._lv_full_load_btn.isVisible() and 0 <= gl.x() - gm.x() <= 12
      and abs(gl.y() - gm.y()) <= 4, (gm, gl))
seq = [v._lv_full_load_btn, v._lv_mv_btn, v._lv_epi_btn, v._lvv_start_btn,
       v._full_lv_btn]
ix = [x.mapTo(v, x.rect().topLeft()).x() for x in seq]
check("LV row order: Load, MV-AoV+Apex, 心外膜, Blood/Endo, FullLV",
      all(x.isVisible() for x in seq) and ix == sorted(ix)
      and len(set(ix)) == 5, ix)
g = v._ct_terr_button()
tl = g.layout()
names = [tl.itemAt(i).widget().text() for i in range(tl.count())
         if tl.itemAt(i).widget() is not None]
check("Territory order: Load, Calc, Save",
      [n for n in names if n in ("Load", "Calc", "Save")]
      == ["Load", "Calc", "Save"], names)
opened = []
import PyQt6.QtWidgets as _QW  # noqa: E402
_QW.QFileDialog.getOpenFileName = staticmethod(
    lambda *a, **k: ("C:/x/case.FullLv.json", ""))
v.window()._open_full_lv_files = lambda paths: opened.append(paths)
v._lv_full_load_btn.click()
app.processEvents()
check("Load opens the picked .FullLv.json like a drop",
      opened == [["C:/x/case.FullLv.json"]], opened)
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
app.processEvents()
os._exit(1 if fails else 0)
