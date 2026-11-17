"""Loaded files readout (2026-11-17): besides FullLv / corotree it names the
.territory.json — the loaded / saved file, 「計算済み（未保存）」 for a fresh
unsaved Calc, なし without a Territory. A fresh Calc forgets the old file.

    python tools/test_ct_files_line_territory.py
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
from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.ui.coronary_tree_window import CoronaryTreeWindow  # noqa: E402
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


vol = np.full((40, 48, 48), -100.0, np.float32)
s = LoadedSeries(modality=Modality.CT, volume=vol, spacing_mm=(0.7, 0.7),
                 cine_fps=None, window=800.0, level=200.0, slice_mm=1.5,
                 patient_basis=None, series_uid="files-ct")
v = CTViewer()
v.resize(1400, 800)
v.show()
v.load_series(s, "files")
app.processEvents()


class Shell:
    state = {}

    def ct_mode_state(self, viewer):
        return dict(self.state)


sh = Shell()
v._ct_mode_shell = lambda: sh


def line(state):
    sh.state = state
    v.refresh_ct_mode_bar()
    return v._ct_files_lbl.text()


txt = line({"full": "a.FullLv.json", "tree": "a.corotree.json", "terr": True,
            "terr_file": "a.territory.json"})
check("loaded territory file named on the line",
      "territory: a.territory.json" in txt and "FullLv: a.FullLv.json" in txt
      and "corotree: a.corotree.json" in txt, txt)
txt = line({"full": "a.FullLv.json", "tree": "a.corotree.json", "terr": True,
            "terr_file": ""})
check("fresh Calc (unsaved) → 計算済み（未保存）",
      "territory: " + t("計算済み（未保存）") in txt, txt)
txt = line({"full": "a.FullLv.json", "tree": "a.corotree.json"})
check("no Territory → なし", txt.endswith("territory: " + t("なし")), txt)

# dock: the path getter + a fresh Calc clears it
check("dock getter exists",
      hasattr(CoronaryTreeWindow, "territory_path_for_series"))
src = (ROOT / "multi_dicomviewer/ui/main_window.py").read_text(encoding="utf-8")
i = src.index("    def ct_territory_compute(")
check("Territory Calc forgets the old .territory.json",
      'w._bundle()["terr_path"] = ""' in src[i:i + 4000])
check("shell passes the file name", 'out["terr_file"]' in src)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
