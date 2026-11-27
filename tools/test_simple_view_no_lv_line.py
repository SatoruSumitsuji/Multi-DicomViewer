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

# 壁厚3D / 壁厚短軸 heat map shown in LVモード → gone in Simple View, back after
v._lvv_thick_ready = lambda: True
for mode in ("3d", "sax"):
    v._lvv_thick_mode = mode
    v._lvv_thick_vol = v._lvv_thick_vol_from_sub(
        np.full((10, 10, 10), 6.0, np.float32), (5, 15, 5, 15, 5, 15))
    v._lvv_thick_stats = {"max": 6.0, "mean": 6.0}
    v._lvv_thick_refresh_display()
    bar = v.pane["A"].thick_bar
    check(f"壁厚{mode}: shown in LVモード (colour bar on)", bool(bar.GetVisibility()))
    v._ct_show_lv(False)
    check(f"壁厚{mode}: Simple View → the heat map is off",
          v._lvv_thick_mode is None and not bar.GetVisibility())
    v._ct_show_lv(True)
    check(f"壁厚{mode}: LVモード again → back ({mode})",
          v._lvv_thick_mode == mode and v._lvv_thick_vol is not None
          and bool(bar.GetVisibility()))
v._lvv_thick_mode = v._lvv_thick_vol = None
v._lvv_thick_refresh_display()
# the short-axis wall map legend (contour LV SAX) is hidden too
v._lv_wall = True
v._lv_sax_active = lambda: True
v._lv["sax_pane"] = "A"
v._lv_update_wall_legend()
leg = v.pane["A"].lv_wall_legend
check("SAX wall legend shown in LVモード", any(a.GetVisibility() for a in leg))
v._ct_show_lv(False)
v._lv_update_wall_legend()
check("… hidden in Simple View", not any(a.GetVisibility() for a in leg))
v._ct_show_lv(True)
v._lv_wall = False
del v._lv_sax_active

src = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text("utf-8")
i = src.index("def _paint_lv(self, p, key, w, h):")
check("pygfx: _paint_lv draws nothing while the LV is hidden",
      'if getattr(v, "_ct_lv_hidden", False):' in src[i:i + 600])
v._lv = None
v.close()
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
