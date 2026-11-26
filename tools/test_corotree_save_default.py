# -*- coding: utf-8 -*-
"""ツリー保存 proposes the file saved last (user 2026-11-26: it is usually an
overwrite): this CT's tree file saved / loaded before (same folder + name),
else the file NAME saved last — remembered across sessions — in the default
folder, else coronary.corotree.json. Fake HOME, stubbed save dialog, made-up
names.

    python tools/test_corotree_save_default.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
for _n in ("information", "warning", "critical", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("unexpected dialog"))))
from multi_dicomviewer.ui import coronary_tree_window as ctw  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


D = tempfile.mkdtemp(prefix="mdv_tree_", dir=HOME)


def cpr(name, uid):
    p = os.path.join(D, name + ".cpr.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"format": "MDV-CPR", "series": {"series_uid": uid},
                   "ctrl": [[0.0, 0.0, z] for z in (0.0, 10.0, 20.0)]}, f)
    return p


proposed = []
answer = [""]


def fake_dialog(_parent, _title, default, _filt):
    proposed.append(default)
    return answer[0], ""


ctw.QFileDialog.getSaveFileName = staticmethod(fake_dialog)

w = ctw.CoronaryTreeWindow(None)
w._show_source_ct = lambda *a, **k: ""
w._load_cpr_paths([cpr("LAD", "save-default-ct")])
w._save_as()
check("first save: coronary.corotree.json in the CPR folder",
      proposed[-1] == os.path.join(D, "coronary.corotree.json"), proposed[-1:])
mine = os.path.join(D, "Case12_tree.corotree.json")
answer[0] = mine
w._save_as()
check("saved under the name the user typed", os.path.isfile(mine))
answer[0] = ""
w._save_as()
check("next save: that same file is proposed (overwrite)",
      proposed[-1] == mine, proposed[-1:])
check("the name is remembered across sessions",
      settings.load_ui_state("corotree_last_name") == "Case12_tree.corotree.json")

# a new session, another CT (another folder): the last NAME is proposed there
D2 = tempfile.mkdtemp(prefix="mdv_tree2_", dir=HOME)
D_keep, D = D, D2
w2 = ctw.CoronaryTreeWindow(None)
w2._show_source_ct = lambda *a, **k: ""
w2._load_cpr_paths([cpr("RCA", "save-default-ct2")])
w2._save_as()
check("new session: the name saved last, in this CT's folder",
      proposed[-1] == os.path.join(D2, "Case12_tree.corotree.json"),
      proposed[-1:])
# a tree loaded from a file → that file is proposed
w3 = ctw.CoronaryTreeWindow(None)
w3._show_source_ct = lambda *a, **k: ""
check("(setup) load the saved tree", w3.load_file(mine))
w3._save_as()
check("a loaded tree: its own file is proposed", proposed[-1] == mine,
      proposed[-1:])
settings.save_ui_state("corotree_last_name", "../x/evil.corotree.json")
w2._last_path = None
w2._save_as()
check("a bad remembered name is not used",
      proposed[-1] == os.path.join(w2._tree_save_dir(), "coronary.corotree.json"),
      proposed[-1:])

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
