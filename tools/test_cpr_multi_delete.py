# -*- coding: utf-8 -*-
"""「複数点の削除」 (user 2026-11-26): on the CPR line open in the short axis
the right-click menu has it right below 「点の削除」; picking it arms the
Polygon, and the edit points inside the polygon the user then draws on the
map pane are deleted (one Ctrl+Z step) — no Polygon tool / extra menu
first. Real VTK CTViewer, synthetic data (made-up patient), no dialog.

    python tools/test_cpr_multi_delete.py
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
from PyQt6.QtWidgets import QApplication, QMenu, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
for nm in ("information", "warning", "critical", "question"):
    setattr(QMessageBox, nm, staticmethod(lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("unexpected dialog"))))
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.viewers import ct_viewer as ctv  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


sx, sy, sz = 0.5, 0.5, 0.5
nz, ny, nx = 80, 100, 120
z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
vol = np.full((nz, ny, nx), -80.0, np.float32)
vol[(y * sy - 25) ** 2 + (z * sz - 20) ** 2 <= 1.5 ** 2] = 380   # along x
s = LoadedSeries(modality=Modality.CT, volume=vol.astype(np.int16),
                 spacing_mm=(0.5, 0.5), cine_fps=None, window=800.0,
                 level=200.0, slice_mm=0.5, patient_basis=None,
                 series_uid="multi-delete-uid")
v = ctv.CTViewer()
v.resize(1400, 800)
v.load_series(s, "t")
v._set_mode("3D")
app.processEvents()
v._coronary_mode = True
p3 = [[x_, 25.0, 20.0] for x_ in (8.0, 14.0, 20.0, 26.0, 32.0, 38.0, 44.0, 50.0)]
pts = [tuple(map(float, v._world3d_to_out("B", np.array(q)))) for q in p3]
v._measures["B"].append({"id": 3, "type": "polyline", "pts": pts,
                         "pts3d": p3, "color": "#ffff00"})
mi = len(v._measures["B"]) - 1
check("(setup) short-axis built", v._cpr_ensure_built(("B", mi))
      and v._cpr is not None and v._cpr.get("src") == "B")
v._meas_btn.setChecked(True)
v._toggle_measure()
app.processEvents()
m = v._cpr_src_measure()
n0 = len(m["pts3d"])

menu = QMenu(v)
menu.addAction(t("Add point"))
dp = menu.addAction(t("Delete point"))
a = v._cpr_multi_del_add(menu, "B", mi)
menu.addAction(t("Delete"))
txt = [x.text() for x in menu.actions()]
check("menu: 「複数点の削除」 right below 「点の削除」",
      a is not None and txt.index(a.text()) == txt.index(dp.text()) + 1
      and a.text() == t("Delete multiple points"), txt)
check("not offered on another (non-CPR) line",
      v._cpr_multi_del_add(QMenu(v), "A", 0) is None)

v._cpr_multi_del_start()
check("picked: the Polygon is armed (lasso)", v._meas_type == "polygon"
      and v._cpr_lasso_stash is not None and v._cpr_multi_del)

# a polygon round edit points 3..5 (x = 26 / 32 / 38 mm), in the map plane
q3, q5 = m["pts"][3], m["pts"][5]
x0, x1 = min(q3[0], q5[0]) - 3.0, max(q3[0], q5[0]) + 3.0
yc = 0.5 * (q3[1] + q5[1])
poly = [(x0, yc - 5.0), (x1, yc - 5.0), (x1, yc + 5.0), (x0, yc + 5.0)]
v._cpr_lasso_commit("B", poly)
for _ in range(5):
    app.processEvents()
m = v._cpr_src_measure()
xs = [round(float(q[0])) for q in m["pts3d"]]
check("closing the polygon deleted the 3 points inside",
      len(m["pts3d"]) == n0 - 3 and not any(q in xs for q in (26, 32, 38)), xs)
check("… the others kept, ends untouched", xs[0] == 8 and xs[-1] == 50, xs)
check("… back to the edit (Polygon off, no lasso)", v._cpr_lasso_stash is None
      and not v._cpr_multi_del and v._meas_type != "polygon", v._meas_type)
check("… no Polygon measurement made",
      not any(mm.get("type") == "polygon" for k in ("A", "B")
              for mm in v._measures[k]))
v._undo_measure() if hasattr(v, "_undo_measure") else None

# a plain Polygon (not via the menu) still opens the selection menu path
v._set_measure_type("polygon")
check("plain Polygon: not a delete", not v._cpr_multi_del)
v._cpr_lasso_restore()

for fn in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    src = (ROOT / "multi_dicomviewer/viewers" / fn).read_text("utf-8")
    check(f"{fn}: in both line menus + dispatched",
          src.count("multi_del = self._cpr_multi_del_add(menu, which, mi)") == 2
          and src.count("self._cpr_multi_del_start()") == 2)
v.close()
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
