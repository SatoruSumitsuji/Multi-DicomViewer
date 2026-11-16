"""Resume trace of a saved CPR line (2026-11-16): the re-committed line used to
lose its .cpr.json → SaveNext wrote a 2nd serial file and registered a 2nd
tree vessel ("二重保存"). Now its save asks 「元のCPR線とそのファイル … を削除して
いいですか？」: Yes = the original file is replaced (same file / tree vessel,
only the new line remains), No = the original stays, the new line goes to a
new serial file. Asked once, at save time.

    python tools/test_cpr_resume_replace.py
"""
import json
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

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


DIR = tempfile.mkdtemp(dir=HOME)
ORIG = os.path.join(DIR, "01.cpr.json")
with open(ORIG, "w", encoding="utf-8") as f:
    json.dump({"format": "MDV-CPR", "ctrl": [[0, 0, 0], [1, 1, 1]]}, f)

v = CTViewer()
v._coronary_mode = True
v._coronary_sync_ui = lambda: None
v._redraw_meas = lambda which: None
v._cpr_next_dir = lambda: DIR
asked = []
answer = {"a": QMessageBox.StandardButton.Yes}


def fake_q(*a, **k):
    asked.append(a[2] if len(a) > 2 else "")
    return answer["a"]


QMessageBox.question = staticmethod(fake_q)


def resume_and_recommit(extra=None):
    m = {"type": "polyline", "id": 7, "pane": "A",
         "pts": [(0.0, 0.0), (5.0, 5.0)]}
    m.update(extra or {})
    v._measures["A"] = [m]
    v._cpr_resume_capture("A", 0)
    v._resume_trace("A", 0, 1)
    new = {"type": "polyline", "id": 7, "pane": "A",
           "pts": [(0.0, 0.0), (5.0, 5.0), (9.0, 9.0)]}
    v._measures["A"] = [new]
    v._draft = None
    v._cpr_source_measure = lambda: new
    return new


# Yes → the original file is the target (replaced in place)
resume_and_recommit({"_cpr_file": ORIG})
answer["a"] = QMessageBox.StandardButton.Yes
tgt = v._cpr_next_target()
check("resumed line: asked once, naming the file",
      len(asked) == 1 and "01.cpr.json" in asked[0], asked)
check("Yes: the save replaces the original file", tgt == (ORIG, False), tgt)
asked.clear()
tgt2 = v._cpr_next_target()
check("asked only once per Resume", asked == [] and tgt2[1] is True, tgt2)

# No → original kept, a new serial file
resume_and_recommit({"_cpr_saved": ORIG})
answer["a"] = QMessageBox.StandardButton.No
tgt = v._cpr_next_target()
check("No: new serial file, original kept",
      tgt == (os.path.join(DIR, "02.cpr.json"), True)
      and os.path.isfile(ORIG), tgt)

# Through SaveNext: Yes → registered with the ORIGINAL path (tree dedupes by
# path → that vessel is updated in place, no 2nd vessel)
reg = []


class Shell:
    def coronary_register_cpr(self, data, name, path):
        reg.append(path)
        return True

    def statusBar(self):
        raise AttributeError


v.window = lambda: Shell()
v._cpr_build_data = lambda: {"format": "MDV-CPR",
                             "ctrl": [[0, 0, 0], [1, 1, 1], [2, 2, 2]]}


def fake_write(path, data, quiet=False):
    v._cpr_last_path = path
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return True


v._cpr_write = fake_write
v._cpr_hide_all_results = lambda: None
resume_and_recommit({"_cpr_file": ORIG})
answer["a"] = QMessageBox.StandardButton.Yes
before = sorted(os.listdir(DIR))
v._cpr = None
v._cpr_register_to_tree()
with open(ORIG, encoding="utf-8") as f:
    now = json.load(f)
check("SaveNext + Yes: original file now holds the new line, no new file",
      len(now["ctrl"]) == 3 and sorted(os.listdir(DIR)) == before, os.listdir(DIR))
check("SaveNext + Yes: registered under the original file (vessel updated)",
      reg == [ORIG], reg)

# A line never saved: no question
asked.clear()
resume_and_recommit()
tgt = v._cpr_next_target()
check("unsaved line: no question, new serial", asked == [] and tgt[1] is True)

# Edited from the tree: the vessel's file (via _coro_edit_vid) is the origin
asked.clear()
m = {"type": "polyline", "id": 9, "pane": "A", "pts": [(0.0, 0.0), (5.0, 5.0)]}
v._measures["A"] = [m]
v._cpr = {"src": "A", "src_mi": 0}
v._coro_edit_vid = "v3"
v._cpr_last_path = ORIG
v._cpr_resume_capture("A", 0)
check("tree-vessel edit: origin = the vessel's file",
      (v._cpr_resume_origin or {}).get("path") == ORIG)
v._cpr = None

src = (ROOT / "multi_dicomviewer/viewers/ct_viewer.py").read_text(encoding="utf-8")
mac = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(encoding="utf-8")
mix = (ROOT / "multi_dicomviewer/viewers/cpr_mixin.py").read_text(encoding="utf-8")
check("both viewers + the line menu capture before leaving the short-axis",
      src.count("self._cpr_resume_capture(which, mi)") == 1
      and mac.count("self._cpr_resume_capture(which, mi)") == 1
      and mix.count("self._cpr_resume_capture(which, mi)") == 1)
check("Exit forgets an unsaved Resume",
      src.count("self._cpr_resume_origin = None") == 1
      and mac.count("self._cpr_resume_origin = None") == 1)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
