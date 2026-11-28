# -*- coding: utf-8 -*-
"""Epi 「Draw」 (user 2026-11-28): after the dialog's Draw starts the trace,
the row-3 Draw button is coloured (Epi green), so it does not look as if it
still had to be pressed. Plain before Draw / outside Epi. Real VTK CTViewer
on synthetic data (made-up patient); Mac viewer: the same code (source).

    python tools/test_lv_epi_draw_btn_color.py
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
# the Epi dialog: exec() "clicks" Draw (the first added button)
_orig_add = QMessageBox.addButton


def _add(self, *a):
    b = _orig_add(self, *a)
    if not hasattr(self, "_first"):
        self._first = b
    return b


QMessageBox.addButton = _add
QMessageBox.exec = lambda self: 0
QMessageBox.clickedButton = lambda self: getattr(self, "_first", None)
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.viewers import ct_viewer as ctv  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


vol = np.random.default_rng(0).normal(0, 25, (80, 100, 100)).astype(np.int16)
s = LoadedSeries(modality=Modality.CT, volume=vol, spacing_mm=(0.5, 0.5),
                 cine_fps=None, window=800.0, level=200.0, slice_mm=0.5,
                 patient_basis=None, series_uid="draw-uid")
v = ctv.CTViewer()
v.resize(1400, 800)
v.load_series(s, "draw")
v._set_mode("3D")
v.show()
for _ in range(5):
    app.processEvents()
b = v._lv_epi_draw_btn
green = "#2e8b57"

# an Epi pass at the selector step (common apex set, nothing traced yet)
v._lv_enter_mode()
v._lv_apex = np.array([25.0, 25.0, 10.0])
v._lv_valves["mitral"] = (np.array([25.0, 25.0, 30.0]), np.array([0, 0, 1.0]),
                          10.0)
v._lv["pass"] = "epi"
v._lv_epi_armed = False
v._lv_update_submode_ui()
check("Epi before Draw: Draw shown, plain", b.isVisibleTo(v)
      and green not in b.styleSheet(), b.styleSheet())
v._lv_epi_draw()
for _ in range(3):
    app.processEvents()
check("after the dialog's Draw: tracing armed", v._lv_epi_armed)
check("… the row-3 Draw is coloured (Epi green)", green in b.styleSheet(),
      b.styleSheet())
v._lv_epi_armed = False
v._lv["model"].epi_contours = []
v._lv["model"].epi_planes = {}
v._lv_update_submode_ui()
check("not armed → plain again", green not in b.styleSheet(), b.styleSheet())

mac = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(
    encoding="utf-8")
check("Mac viewer: the same colouring",
      'self._lv_epi_draw_btn.setProperty("drawing", _drawing)' in mac
      and 'self._LV_STY["epi"] if _drawing' in mac)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(0 if not fails else 1)
