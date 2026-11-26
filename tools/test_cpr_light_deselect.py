# -*- coding: utf-8 -*-
"""Deselect on a set CPR line (user 2026-11-26): after a CPR line is set,
all its edit points stay shown while Draw is on, and turning Draw off was
the only way out. Its right-click menu now has 「選択を解除 (Deselect)」 at
the top: the points go (not drawn, not grabbable), the line stays; editing
it again (ReCalc / Resume) brings them back. Real VTK CTViewer, synthetic
data (made-up patient), no dialog.

    python tools/test_cpr_light_deselect.py
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


vol = np.full((60, 80, 80), -80, np.int16)
s = LoadedSeries(modality=Modality.CT, volume=vol, spacing_mm=(0.5, 0.5),
                 cine_fps=None, window=800.0, level=200.0, slice_mm=0.5,
                 patient_basis=None, series_uid="light-deselect-uid")
v = ctv.CTViewer()
v.resize(1200, 700)
v.load_series(s, "t")
v._set_mode("3D")
app.processEvents()
v._coronary_mode = True
v._meas_on = True                                 # Draw still on
p3 = [[10.0, 20.0, 15.0], [20.0, 20.0, 15.0], [30.0, 20.0, 15.0]]
pts = [tuple(map(float, v._world3d_to_out("B", np.array(q)))) for q in p3]
v._measures["B"].append({"id": 7, "type": "polyline", "pts": pts,
                         "pts3d": p3, "color": "#ffff00", "_cpr_src": True,
                         "_cpr_saved": True})
mi = len(v._measures["B"]) - 1
m = v._measures["B"][mi]
check("(setup) no short-axis open", v._cpr is None)


def n_handles():
    v._redraw_meas("B")
    pd = v.pane["B"].meas_pts_mapper.GetInput()
    return 0 if pd is None else pd.GetNumberOfPoints()


def menu_texts():
    menu = QMenu(v)
    menu.addAction("x")
    a = v._cpr_edit_deselect_add(menu, "B", mi)
    return a, [x.text() for x in menu.actions() if not x.isSeparator()]


check("set line, Draw on: all its edit points shown", n_handles() >= 3,
      n_handles())
sx, sy = v._world_to_qt("B", *pts[1])
check("… and grabbable", v._pick_handle("B", sx, sy) == (mi, 1))
a, txt = menu_texts()
check("right-click menu: 「選択を解除 (Deselect)」 at the top",
      a is not None and txt and txt[0] == t("選択を解除 (Deselect)"), txt)
check("Deselect runs (no dialog)", v._cpr_edit_deselect("B", mi) is True)
check("… the edit points are gone", n_handles() == 0, n_handles())
check("… not grabbable", v._pick_handle("B", sx, sy) is None)
check("… the line itself stays", m in v._measures["B"]
      and not m.get("hidden") and m.get("_cpr_src"))
check("… Deselect no longer offered", menu_texts()[0] is None)
check("Draw still on", v._meas_on)
v._resume_trace("B", mi, len(m["pts"]) - 1)
check("Resume trace: the points come back", not m.get("_cpr_desel"))

src = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text("utf-8")
check("pygfx: Deselect hides / un-grabs the points too",
      'or bool(m.get("_cpr_desel")))' in src
      and 'or m.get("_lv_valve") or m.get("_cpr_desel")):' in src
      and src.count('pop("_cpr_desel", None)') == 2
      and src.count("self._cpr_edit_deselect(which, mi)") == 2)
v.close()
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
