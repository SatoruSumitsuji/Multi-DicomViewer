# -*- coding: utf-8 -*-
"""Coronary Tree ↔ its .cpr.json files (user 2026-11-25):
- FixName 「@007」 + フォルダ更新 renames the files to <name>@007.cpr.json;
- Open Folder opens the folder of the listed lines' .cpr.json (not the CT's
  parent folder, where the .corotree.json sits);
- the .corotree.json remembers each line's file, so a reload knows it.
Synthetic lines, fake HOME, dialogs patched (no real dialog).

    python tools/test_coro_cpr_folder_paths.py
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
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(
    lambda *a, **k: QMessageBox.StandardButton.Yes)
from multi_dicomviewer.core.centerline import CenterLine  # noqa: E402
from multi_dicomviewer.ui import coronary_tree_window as ctw  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


# layout: <case>/CT (the 3-D CT), <case>/x.corotree.json, <case>/cpr/*.cpr.json
case = tempfile.mkdtemp(dir=HOME)
ctd = os.path.join(case, "CT")
cprd = os.path.join(case, "cpr")
os.makedirs(ctd)
os.makedirs(cprd)
UID = "1.2.3"
lines = {"LAD": np.c_[np.linspace(0, 40, 6), np.zeros(6), np.zeros(6)],
         "D1": np.c_[np.full(6, 20.0), np.linspace(0, 30, 6), np.zeros(6)]}
for nm, c in lines.items():
    with open(os.path.join(cprd, nm + ".cpr.json"), "w", encoding="utf-8") as f:
        json.dump({"format": "MDV-CPR", "version": 1, "type": "cpr",
                   "series": {"series_uid": UID}, "ctrl": c.tolist(),
                   "state": {"idx": 0}}, f)
w = ctw.CoronaryTreeWindow(None)
w._use_uid(UID, ct_dir=ctd)
b = w._bundle()
tr = b["tree"]
tr.add_root("lad", "LAD", "LAD",
            CenterLine.from_points(lines["LAD"], step_mm=0.5).points)
tr.vessels["lad"].ctrl = lines["LAD"]
tr.add_vessel("d1", "D1", "branch",
              CenterLine.from_points(lines["D1"], step_mm=0.5).points,
              ctrl=lines["D1"])
tr.vessels["d1"].parent, tr.vessels["d1"].junction = "lad", 40
tree_path = os.path.join(case, "x.corotree.json")
w._write_to(tree_path)

opened = []
import PyQt6.QtGui as _G  # noqa: E402
_G.QDesktopServices.openUrl = staticmethod(
    lambda u: opened.append(os.path.normpath(u.toLocalFile())))

# ---- a tree loaded with its files in a sub-folder (no record yet)
w2 = ctw.CoronaryTreeWindow(None)
w2.load_file(tree_path)
w2._open_cpr_folder()
check("Open Folder: the .cpr.json folder (found in a sub-folder)",
      opened[-1:] == [os.path.normpath(cprd)], (opened, cprd))

# ---- FixName @007 + フォルダ更新
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: cprd)
w2._fix_edit.setText("@007")
w2._on_fix_edited("@007")
w2._folder_update()
names = sorted(os.listdir(cprd))
check("フォルダ更新 with FixName → files renamed to <name>@007.cpr.json",
      names == ["D1@007.cpr.json", "LAD@007.cpr.json"], names)
check("… the lines renamed too, the tree intact",
      sorted(v.name for v in w2._tree.vessels.values())
      == ["D1@007", "LAD@007"]
      and w2._tree.vessels["d1"].parent == "lad", 
      [(v.name, v.parent) for v in w2._tree.vessels.values()])
with open(tree_path, encoding="utf-8") as f:
    saved = json.load(f)
cf = sorted(d.get("cpr_file", "") for d in saved["vessels"])
check("the .corotree.json records each line's file (relative)",
      cf == ["cpr/D1@007.cpr.json", "cpr/LAD@007.cpr.json"], cf)

# ---- reload from the record: Open Folder still right
w3 = ctw.CoronaryTreeWindow(None)
w3.load_file(tree_path)
w3._open_cpr_folder()
check("reload: Open Folder from the recorded files",
      opened[-1] == os.path.normpath(cprd), opened[-1])
check("reload: every line knows its file",
      all(os.path.isfile(w3._bundle()["paths"].get(v, ""))
          for v in w3._tree.vessels), w3._bundle()["paths"])
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
os._exit(1 if fails else 0)
