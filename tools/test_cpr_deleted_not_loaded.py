# -*- coding: utf-8 -*-
"""_deleted is never read (user 2026-11-26): CPR lines deleted with their
files (moved into the _deleted folder beside them) came back in the
Coronary Tree. The folder stays where it is, but the app reads nothing in
it — not via a .corotree.json saved before the delete (its sub-folder
search found the moved files), a dropped / chosen folder (also the _deleted
folder itself, any case), or files dropped from inside it. Temp files only
(made-up names), no real dialog.

    python tools/test_cpr_deleted_not_loaded.py
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

import numpy as np  # noqa: E402
from PyQt6.QtCore import QUrl  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
for _n in ("information", "warning", "critical", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("unexpected dialog"))))
from multi_dicomviewer.ui import coronary_tree_window as ctw  # noqa: E402
from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


UID = "deleted-not-loaded-ct"
D = tempfile.mkdtemp(prefix="mdv_cpr_", dir=HOME)


def write(name, x):
    p = os.path.join(D, name + ".cpr.json")
    ctrl = [[x, 0.0, z] for z in (0.0, 10.0, 20.0, 30.0)]
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"format": "MDV-CPR", "version": 1, "type": "cpr",
                   "series": {"series_uid": UID}, "ctrl": ctrl}, f)
    return p


paths = [write(n, x) for n, x in (("LAD", 0.0), ("RCA", 30.0), ("LCX", 60.0))]
w = ctw.CoronaryTreeWindow(None)
w._show_source_ct = lambda *a, **k: ""
w._load_cpr_paths(paths)
names = lambda: sorted(v.name for v in w._tree.vessels.values())  # noqa: E731
check("(setup) three lines loaded", names() == ["LAD", "LCX", "RCA"], names())
tree_p = os.path.join(D, "coronary.corotree.json")
w._write_to(tree_p)
check("(setup) tree saved", os.path.isfile(tree_p))

vid = next(k for k, v in w._tree.vessels.items() if v.name == "RCA")
w._tree_w.clearSelection()
w._item_for_vid(vid).setSelected(True)
w._ask_delete_mode = lambda *a, **k: "file"
w._delete_selected()
dd = os.path.join(D, "_deleted")
check("(setup) RCA deleted with its file → _deleted",
      names() == ["LAD", "LCX"] and os.path.isfile(os.path.join(dd, "RCA.cpr.json"))
      and not os.path.exists(os.path.join(D, "RCA.cpr.json")))

# the tree saved BEFORE the delete is loaded again (not re-saved)
w2 = ctw.CoronaryTreeWindow(None)
w2._show_source_ct = lambda *a, **k: ""
check("old .corotree.json loads", w2.load_file(tree_p))
n2 = sorted(v.name for v in w2._tree.vessels.values())
check("… the deleted line does NOT come back", n2 == ["LAD", "LCX"], n2)
check("… no path into _deleted",
      not any(ctw.in_deleted_dir(p) for p in w2._bundle()["paths"].values()),
      w2._bundle()["paths"])

# dropped / chosen folders and files
fs = MainWindow._folder_cpr_files(D)
check("dropped parent folder: nothing from _deleted",
      fs and not any(ctw.in_deleted_dir(p) for p in fs), fs)
check("dropped _deleted folder itself: nothing",
      MainWindow._folder_cpr_files(dd) == [])
got = ctw.CoronaryTreeWindow._collect_cpr_paths(
    [QUrl.fromLocalFile(dd), QUrl.fromLocalFile(os.path.join(dd, "RCA.cpr.json")),
     QUrl.fromLocalFile(D)])
check("drop on the tab: _deleted folder / its files skipped",
      got and not any(ctw.in_deleted_dir(p) for p in got), got)
w3 = ctw.CoronaryTreeWindow(None)
w3._show_source_ct = lambda *a, **k: ""
w3._load_cpr_paths([os.path.join(dd, "RCA.cpr.json")])
check("a file inside _deleted is not loaded", not w3._tree.vessels)
check("any case (_Deleted) and any depth",
      ctw.in_deleted_dir(os.path.join(D, "_Deleted", "x", "a.cpr.json"))
      and not ctw.in_deleted_dir(os.path.join(D, "a_deleted.cpr.json"))
      and not ctw.in_deleted_dir(os.path.join(D, "_deleted.cpr.json")))
opened = []
mw = MainWindow.__new__(MainWindow)
mw._is_cpr_json = lambda p: opened.append(p) or True
MainWindow._open_cpr_files(mw, [os.path.join(dd, "RCA.cpr.json")])
check("shell drop of a _deleted file: not opened", opened == [], opened)

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
