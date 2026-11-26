# -*- coding: utf-8 -*-
"""A confirmed CPR line is DRAWN (user 2026-11-26): after Auto-Track →
「これ以上追えません」 OK, the line 「08」 was in the Coronary Tree but nothing
showed on the image until ツリー + 表示 — the image line is put away on save
(Hide All Result) and the tree overlay draws only with ツリー表示 on. Now
registering / overwriting / Auto-Track (and its branches) turn ツリー表示 on
and un-hide those vessels. Synthetic data, no dialog.

    python tools/test_cpr_confirmed_shown.py
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
from multi_dicomviewer.ui.coronary_tree_window import (  # noqa: E402
    CoronaryTreeWindow)

for _n in ("information", "warning", "critical", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("unexpected dialog"))))

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


UID = "shown-ct"
DIR = tempfile.mkdtemp(dir=HOME)
ctrl = np.array([[10.0, 10.0, 5.0], [10.0, 10.0, 15.0], [10.0, 10.0, 25.0]])


def data():
    return {"format": "MDV-CPR", "version": 1, "type": "cpr",
            "series": {"series_uid": UID}, "src_dir": "",
            "ctrl": ctrl.tolist()}


def vids_drawn(w):
    return {e.get("vid") for e in (w.overlay_spec(uid=UID) or [])}


w = CoronaryTreeWindow(None)
w._show_source_ct = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("the CT is not re-shown"))
check("start: ツリー表示 off → nothing drawn",
      not w._overlay_on and not w.overlay_spec(uid=UID))

p08 = os.path.join(DIR, "08.cpr.json")
with open(p08, "w", encoding="utf-8") as f:
    json.dump(data(), f)
ok = w.register_cpr(data(), "08", p08)
vid = next(k for k, v in w._tree.vessels.items() if v.name == "08")
check("registered 「08」 is drawn at once (ツリー表示 turned on)",
      ok and w._overlay_on and vid in vids_drawn(w), vids_drawn(w))
check("… the button reads 非表示", w._overlay_btn is None
      or w._overlay_btn.text() in ("非表示", "Hide"), w._overlay_btn
      and w._overlay_btn.text())

# a hidden vessel overwritten (上書き保存) → shown again
w._bundle()["hidden"].add(vid)
w._set_overlay = None                     # never the re-show path
w._overlay_on = False
check("hidden + ツリー非表示: not drawn", vid not in vids_drawn(w))
ok = w.update_vessel_cpr(vid, ctrl + [0.5, 0, 0], p08)
check("overwritten → drawn again (un-hidden, ツリー表示 on)",
      ok and w._overlay_on and vid in vids_drawn(w)
      and vid not in w._bundle()["hidden"])

# Auto-Track of a registered line (nothing to extend in an empty volume)
w._overlay_on = False
w._bundle()["hidden"].add(vid)
vol = np.zeros((40, 40, 40), np.float32)
out = w.auto_extend(vid, vol, (1.0, 1.0, 1.0), folder=DIR)
check("Auto-Track result → the line is drawn",
      out is not None and w._overlay_on and vid in vids_drawn(w), out)

src = (ROOT / "multi_dicomviewer" / "ui" / "coronary_tree_window.py"
       ).read_text(encoding="utf-8")
i = src.index("    def add_auto_branches(")
check("auto-added branches are shown with their line",
      "self._show_confirmed([vid] + made)" in src[i:i + 3000])

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
