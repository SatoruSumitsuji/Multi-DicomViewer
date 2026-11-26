# -*- coding: utf-8 -*-
"""Deleting CPR lines from the Coronary Tree (user 2026-11-26): the prompt
offers 「元ファイルごと削除」 / 「リストから消去」 / 「キャンセル」.
「元ファイルごと削除」 moves each line's .cpr.json into a _deleted folder
beside it (a name already there gets a time stamp) — kept, not erased;
folder scans (フォルダ更新 / 読込 / a dropped folder) skip _deleted, so the
line does not come back. 「リストから消去」 keeps the files; キャンセル
changes nothing. Temp files only (made-up names), no real dialog.

    python tools/test_cpr_delete_files.py
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
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.information = staticmethod(lambda *a, **k: None)
from multi_dicomviewer.ui import coronary_tree_window as ctw  # noqa: E402
from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


D = tempfile.mkdtemp(prefix="mdv_del_")


def line(a, b, n=41):
    return np.linspace(np.asarray(a, float), np.asarray(b, float), n)


def write(name):
    p = os.path.join(D, name + ".cpr.json")
    json.dump({"format": "MDV-CPR", "ctrl": [[0, 0, 0], [0, 0, 9]]},
              open(p, "w", encoding="utf-8"))
    return p


w = ctw.CoronaryTreeWindow(None)
w._use_uid("1.2.9")
b = w._bundle()
tr = b["tree"]
tr.add_root("a", "01", "LAD", line([0, 0, 0], [0, 0, 40]))
tr.add_vessel("a1", "02", "branch", line([0, 0, 20], [15, 0, 20]))
tr.vessels["a1"].parent, tr.vessels["a1"].junction = "a", 20
tr.add_root("c", "03", "RCA", line([30, 0, 0], [30, 0, 40]))
tr.add_root("e", "04", "LCX", line([60, 0, 0], [60, 0, 40]))
for vid, nm in (("a", "01"), ("a1", "02"), ("c", "03"), ("e", "04")):
    b["paths"][vid] = write(nm)
w._populate()


def select(*vids):
    w._tree_w.clearSelection()
    for v in vids:
        w._item_for_vid(v).setSelected(True)


asked = []


def answer(mode):
    def f(text):
        asked.append(text)
        return mode
    return f


# キャンセル → nothing changes
select("c")
w._ask_delete_mode = answer(None)
w._delete_selected()
check("キャンセル: line and file kept",
      "c" in tr.vessels and os.path.isfile(os.path.join(D, "03.cpr.json")))

# リストから消去 → the file stays
w._ask_delete_mode = answer("list")
w._delete_selected()
check("リストから消去: off the list, the file stays",
      "c" not in tr.vessels and os.path.isfile(os.path.join(D, "03.cpr.json")))

# 元ファイルごと削除 → the line and its branch, files moved to _deleted
select("a")
w._ask_delete_mode = answer("file")
w._delete_selected()
dd = os.path.join(D, "_deleted")
check("元ファイルごと削除: the line + its downstream branch removed",
      "a" not in tr.vessels and "a1" not in tr.vessels and "e" in tr.vessels)
check("… their files moved into _deleted (not erased)",
      not os.path.exists(os.path.join(D, "01.cpr.json"))
      and not os.path.exists(os.path.join(D, "02.cpr.json"))
      and os.path.isfile(os.path.join(dd, "01.cpr.json"))
      and os.path.isfile(os.path.join(dd, "02.cpr.json")),
      os.listdir(D))
check("… their paths forgotten", "a" not in b["paths"]
      and "a1" not in b["paths"] and "e" in b["paths"])
check("the prompt says how many", asked and "2" in asked[-1], asked[-1:])

# a name already in _deleted → kept, the new one gets a time stamp
write("01")
tr.add_root("a", "01", "LAD", line([0, 0, 0], [0, 0, 40]))
b["paths"]["a"] = os.path.join(D, "01.cpr.json")
w._populate()
select("a")
w._delete_selected()
got = sorted(os.listdir(dd))
check("same name again: both kept in _deleted (time-stamped)",
      len([n for n in got if n.startswith("01")]) == 2
      and "01.cpr.json" in got, got)

# folder scans skip _deleted
files = MainWindow._folder_cpr_files(D)
check("a dropped folder: _deleted is skipped",
      all("_deleted" not in f for f in files)
      and any(f.endswith("04.cpr.json") for f in files), files)
src = Path(ctw.__file__).read_text(encoding="utf-8")
check("フォルダ更新 / 読込 and a dropped folder skip _deleted",
      src.count('!= self.DELETED_DIR]') == 1
      and src.count('!= "_deleted"]') == 1)
check("the single-line delete asks the same way",
      src.count("self._ask_delete_mode(") == 2)
check("no 「元のCPRファイルは残ります」 prompt left",
      "(元のCPRファイルは残ります)\", name" not in src
      and "(元のCPRファイルは残ります)\", n=" not in src)

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
