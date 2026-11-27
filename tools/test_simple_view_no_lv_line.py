# -*- coding: utf-8 -*-
"""Simple View is a clean CT (user 2026-11-27): after an LV analysis, leaving
LVモード for Simple View left the thick yellow base-cut line on the long-axis
pane. Now nothing of the LV pane overlay is drawn while the LV is hidden
(Simple View / CoronaryTree), and it comes back in LVモード. Real VTK
CTViewer with a stand-in LV model (synthetic, no dialog); pygfx by source.

    python tools/test_simple_view_no_lv_line.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
os.environ["QT_QPA_PLATFORM"] = ("windows" if sys.platform == "win32"
                                 else os.environ.get("QT_QPA_PLATFORM", "offscreen"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


class _Ax:
    apex = np.array([17.0, 17.0, 5.0])
    axis = np.array([0.0, 0.0, 1.0])


class _Model:
    axis = _Ax()
    epi_axis = None
    endo_contours = epi_contours = {}
    n_planes = 6

    def along_range(self, which):
        return (0.0, 15.0)


s = LoadedSeries(modality=Modality.CT, volume=np.full((40, 50, 50), -100, np.int16),
                 spacing_mm=(0.7, 0.7), cine_fps=None, window=800.0, level=200.0,
                 slice_mm=0.7, patient_basis=None, series_uid="lvline-ct")
v = CTViewer()
v.resize(1100, 650)
v.show()
v.load_series(s, "s")
app.processEvents()
v._lv_draw_apex_markers = lambda *a, **k: None
v._lv = {"model": _Model(), "phase": "contour", "plane_idx": 0,
         "target": None, "pane": "B", "sax": None, "pass": None}


def n_line():
    pd = v.pane["B"].lv_line_mapper.GetInput()
    return 0 if pd is None else pd.GetNumberOfPoints()


v._ct_mode_sel = "lv"
v._ct_lv_hidden = False
v._redraw_lv("B")
check("LVモード: the base-cut line is drawn", n_line() >= 2, n_line())

v._ct_show_lv(False)                     # what Simple View does
check("Simple View: the base-cut line is gone at once", n_line() == 0, n_line())
v._redraw_lv("B")                        # any later view change / refresh
v._refresh()
app.processEvents()
check("… and stays gone on later redraws", n_line() == 0, n_line())

v._ct_show_lv(True)                      # back to LVモード
check("LVモード again: the line is back", n_line() >= 2, n_line())

src = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text("utf-8")
i = src.index("def _paint_lv(self, p, key, w, h):")
check("pygfx: _paint_lv draws nothing while the LV is hidden",
      'if getattr(v, "_ct_lv_hidden", False):' in src[i:i + 600])
v._lv = None
v.close()
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
