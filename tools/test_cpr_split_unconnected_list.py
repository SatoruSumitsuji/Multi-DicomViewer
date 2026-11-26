# -*- coding: utf-8 -*-
"""「ここで separate」 on an UNCONNECTED line (user 2026-11-26): a line just
drawn (LM-L8, not yet connected) was cut in two on the image, but the
Coronary Tree list showed only one — the 未接続 group listed its lines
without the lines hung below them (the distal part is hung on the
proximal one). Now each is listed with its children. Real dock, synthetic
data, no dialog.

    python tools/test_cpr_split_unconnected_list.py
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
from PyQt6.QtWidgets import (QApplication, QMessageBox,  # noqa: E402
                             QTreeWidgetItemIterator)

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.ui import coronary_tree_window as ctw  # noqa: E402

for _n in ("information", "warning", "critical", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("unexpected dialog"))))

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


UID = "split-loose-ct"
DIR = tempfile.mkdtemp(dir=HOME)
ctrl = np.array([[10.0, 10.0, 5.0], [10.0, 10.0, 15.0], [10.0, 10.0, 25.0],
                 [10.0, 10.0, 35.0], [10.0, 10.0, 45.0]])
data = {"format": "MDV-CPR", "version": 1, "type": "cpr",
        "series": {"series_uid": UID}, "src_dir": "", "ctrl": ctrl.tolist()}
p = os.path.join(DIR, "LM-L8.cpr.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(data, f)

w = ctw.CoronaryTreeWindow(None)
w._show_source_ct = lambda *a, **k: ""
check("(setup) registered", w.register_cpr(data, "LM-L8", p))
vid = next(k for k, v in w._tree.vessels.items() if v.name == "LM-L8")
check("(setup) an unconnected line", vid in w._tree.unconnected())


def listed():
    out = []
    it = QTreeWidgetItemIterator(w._tree_w)
    while it.value():
        x = it.value()
        v_ = x.data(0, ctw._UID_ROLE)
        if v_:
            out.append(v_)
        it += 1
    return out


v = w._tree.vessels[vid]
new = w.split_vessel(vid, v.n // 2, folder=DIR)
check("separate made a 2nd line (its own file)",
      bool(new) and new in w._tree.vessels
      and os.path.isfile(w._bundle()["paths"].get(new, "")))
ls = listed()
check("the list shows BOTH lines", vid in ls and new in ls, ls)
check("… the distal one under the proximal one",
      any(x.data(0, ctw._UID_ROLE) == new and x.parent() is not None
          and x.parent().data(0, ctw._UID_ROLE) == vid
          for x in w._tree_w.findItems(
              "", ctw.Qt.MatchFlag.MatchContains
              | ctw.Qt.MatchFlag.MatchRecursive)))
new2 = w.split_vessel(new, w._tree.vessels[new].n // 2, folder=DIR)
check("separating again: all three listed",
      bool(new2) and set(listed()) >= {vid, new, new2}, listed())
check("each line listed once", len(listed()) == len(set(listed())))

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
