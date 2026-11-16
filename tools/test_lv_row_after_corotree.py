"""LV steps reachable after a corotree load (2026-11-16): a loaded corotree
narrows the bottom bar to its CoronaryTree half (LV row hidden). Picking
標準 / LVモード afterwards must bring back 「LV: MV-AoV+Apex  Epi  Blood/Endo
FullLV」 (it stayed hidden in every mode, so no LV step could be entered).

    python tools/test_lv_row_after_corotree.py
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
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


vol = np.full((60, 64, 64), -100.0, np.float32)
s = LoadedSeries(modality=Modality.CT, volume=vol, spacing_mm=(0.7, 0.7),
                 cine_fps=None, window=800.0, level=200.0, slice_mm=1.5,
                 patient_basis=None, series_uid="lvrow-ct")
v = CTViewer()
v.resize(1600, 900)
v.show()
v.load_series(s, "lvrow")
v._set_mode("3D")
app.processEvents()


def lv_row_shown():
    app.processEvents()
    return all(not w.isHidden() for w in (v._lv_cap, v._lv_mv_btn,
                                          v._lv_epi_btn, v._lvv_start_btn))


check("plain CT: LV row shown", lv_row_shown())
v.set_bottom_file_context("coronary", "x.corotree.json")
check("corotree loaded (CoronaryTreeモード): LV row hidden", not lv_row_shown()
      and not v._coro_cap.isHidden())
v._ct_mode_clicked("standard")
check("→ 標準モード: LV row back (MV-AoV+Apex reachable)", lv_row_shown()
      and not v._coro_cap.isHidden())
v._ct_switch_inplace("coronary")
check("→ CoronaryTreeモード: LV row hidden again", not lv_row_shown())
v._ct_switch_inplace("lv")
check("→ LVモード: LV row back", lv_row_shown())
mac = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(encoding="utf-8")
check("Mac: same rule",
      "ctx = self._ct_bar_ctx()" in mac)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
