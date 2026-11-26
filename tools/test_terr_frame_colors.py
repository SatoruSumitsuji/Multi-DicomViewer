# -*- coding: utf-8 -*-
"""Territory review frames (user 2026-11-27): the RIGHT pane only observes /
sets the Territory range (a left-drag always turns the VR) → always a red
frame; the LEFT pane follows Measure — orange (Alt+ for the view tools)
with Measure on, red with it off. Outside the review both panes follow
Measure as before. Real VTK CTViewer, synthetic data, no dialog.

    python tools/test_terr_frame_colors.py
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


s = LoadedSeries(modality=Modality.CT, volume=np.full((30, 40, 40), -100, np.int16),
                 spacing_mm=(0.7, 0.7), cine_fps=None, window=800.0, level=200.0,
                 slice_mm=1.5, patient_basis=None, series_uid="frame-ct")
v = CTViewer()
v.resize(1200, 700)
v.load_series(s, "f")
v._set_mode("3D")
app.processEvents()


def frame(key):
    v._set_active(key)
    ss = v._frames[key].styleSheet()
    if "#ff8c00" in ss and "6px" in ss:
        return "orange"
    if "#ff2020" in ss and "3px" in ss:
        return "red"
    return ss


v._meas_on = True
check("normal view, Measure on: both panes orange",
      frame("A") == "orange" and frame("B") == "orange")
v._territory_display = True
check("Territory review, Measure on: left orange, right RED",
      frame("A") == "orange" and frame("B") == "red", (frame("A"), frame("B")))
v._meas_on = False
check("Territory review, Measure off: both red",
      frame("A") == "red" and frame("B") == "red")
v._territory_display = False
v._meas_on = True
check("review left: the right pane follows Measure again",
      frame("B") == "orange")

src = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text("utf-8")
check("pygfx: same rule",
      'alt = meas and not (terr and key == "B")' in src)
mb = (ROOT / "multi_dicomviewer/viewers/ct_mode_bar.py").read_text("utf-8")
check("review enter / leave redraw the frames",
      mb.count('"_update_active_frames"') == 2)
v.close()
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
