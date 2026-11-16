"""CPR line Resume trace (2026-11-16): Auto goes OFF on a Resume of an existing
CPR line (so finishing it neither re-traces nor auto-Fits the hand-corrected
line) and back ON after SaveNext / Save&Exit. A user click on Auto wins.

    python tools/test_cpr_resume_auto.py
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

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


v = CTViewer()
v._coronary_mode = True
v._coronary_sync_ui = lambda: None
v._redraw_meas = lambda which: None
v._cpr_hide_all_results = lambda: None
v._cpr_register_to_tree_impl = lambda save_as=False: True
v._cpr_source_measure = lambda: None
exits = []
v._coronary_exit = lambda: exits.append(1)


def add_line():
    v._measures["A"] = [{"type": "polyline", "id": 7, "pane": "A",
                         "pts": [(0.0, 0.0), (5.0, 5.0), (9.0, 2.0)]}]


check("Auto ON by default", v._cpr_auto_on() and v._cpr_auto_btn.isChecked())

# Resume → Auto OFF; SaveNext → ON
add_line()
v._resume_trace("A", 0, 2)
check("Resume trace: Auto OFF (button too)",
      not v._cpr_auto_on() and not v._cpr_auto_btn.isChecked())
v._cpr = None
v._cpr_register_to_tree()
check("SaveNext: Auto back ON", v._cpr_auto_on() and v._cpr_auto_btn.isChecked())

# Resume → OFF; overwrite Save (menu, rearm=False) keeps OFF; Save&Exit → ON
add_line()
v._resume_trace("A", 0, 2)
v._cpr_register_to_tree(save_as=False, rearm=False)
check("right-click 上書き保存: Auto stays OFF (still editing)", not v._cpr_auto_on())
v._cpr_save_and_exit()
check("Save&Exit: Auto back ON", v._cpr_auto_on() and v._cpr_auto_btn.isChecked()
      and exits == [1])

# User choice wins
add_line()
v._resume_trace("A", 0, 2)
v._cpr_auto_btn.click()                     # user turns it on again
v._cpr_auto_btn.click()                     # …and off
v._cpr_register_to_tree()
check("user turned Auto OFF by hand: SaveNext leaves it OFF",
      not v._cpr_auto_on() and not v._cpr_auto_btn.isChecked())
v._cpr_auto_btn.click()
check("user click ON", v._cpr_auto_on())

# Resume with Auto already OFF: stays OFF afterwards
v._cpr_auto_btn.click()
add_line()
v._resume_trace("A", 0, 2)
v._cpr_register_to_tree()
check("Auto was OFF before the Resume: stays OFF", not v._cpr_auto_on())
v._cpr_auto_btn.click()

# Plain measurement Resume (not coronary): Auto untouched
v._coronary_mode = False
add_line()
v._resume_trace("A", 0, 2)
check("non-coronary Resume: Auto untouched", v._cpr_auto_on())

src = (ROOT / "multi_dicomviewer/viewers/cpr_mixin.py").read_text(encoding="utf-8")
fin = src[src.index("    def _cpr_finish_trace"):src.index("    def _cpr_auto_run")]
check("finishing a trace never auto-Fits (Fit removed)", "_cpr_fit()" not in fin)
mac = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(encoding="utf-8")
check("Mac: same Resume / Auto wiring",
      "self._cpr_resume_auto_off()" in mac
      and "self._cpr_auto_btn.toggled.connect(self._cpr_auto_user_toggle)" in mac)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
