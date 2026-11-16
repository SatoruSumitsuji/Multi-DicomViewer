"""Territory ▸ Load (2026-11-17): no 「置き換えます。続行しますか？」 when the
open corotree / FullLv ARE the .territory.json's linked files and nothing is
unsaved (replacing them changes nothing). Asked when the tree was edited,
another file is open, or the LV has unsaved edits.

    python tools/test_territory_load_noask.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.core.coronary_territory import CoronaryTree  # noqa: E402
from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


D = tempfile.mkdtemp(dir=HOME)
coro_p = os.path.join(D, "case.corotree.json")
lv_p = os.path.join(D, "case.FullLv.json")
other_p = os.path.join(D, "other.corotree.json")
tree = CoronaryTree()
tree.add_root("lad", "LAD", "LAD",
              np.c_[np.linspace(0, 29.3, 30), np.zeros(30), np.zeros(30)])
with open(coro_p, "w", encoding="utf-8") as f:
    json.dump(tree.to_json(series={"series_uid": "1.2.3"}), f)
with open(other_p, "w", encoding="utf-8") as f:
    json.dump(tree.to_json(), f)
with open(lv_p, "w", encoding="utf-8") as f:
    json.dump({"format": "MDV-FullLv"}, f)
with open(coro_p, encoding="utf-8") as f:
    loaded = CoronaryTree.from_json(json.load(f))   # as the dock loads it


class W:
    def __init__(self, tree, coro, lv, full=None):
        self._tree, self._coro, self._lv, self._full = tree, coro, lv, full

    def corotree_path_for_series(self, uid):
        return self._coro

    def full_lv_path_for_series(self, uid):
        return self._lv

    def full_lv_for_series(self, uid):
        return self._full


class V:
    _lv_dirty = False


match = MainWindow._territory_links_match
check("same corotree + FullLv files, nothing unsaved → no question",
      match(W(loaded, coro_p, lv_p, {"x": 1}), V(), "1.2.3", coro_p, lv_p))
check("same corotree, no FullLv loaded yet → no question",
      match(W(loaded, coro_p, ""), V(), "1.2.3", coro_p, lv_p))

edited = CoronaryTree.from_json(json.load(open(coro_p, encoding="utf-8")))
edited.vessels["lad"].name = "LAD-edited"
check("tree edited (unsaved) → asked",
      not match(W(edited, coro_p, lv_p), V(), "1.2.3", coro_p, lv_p))
check("another corotree open → asked",
      not match(W(loaded, other_p, lv_p), V(), "1.2.3", coro_p, lv_p))
check("another FullLv open → asked",
      not match(W(loaded, coro_p, os.path.join(D, "b.FullLv.json")), V(),
                "1.2.3", coro_p, lv_p))
check("in-memory (unsaved) FullLv → asked",
      not match(W(loaded, coro_p, "", {"x": 1}), V(), "1.2.3", coro_p, lv_p))
dv = V()
dv._lv_dirty = True
check("unsaved LV edit in the viewer → asked",
      not match(W(loaded, coro_p, lv_p), dv, "1.2.3", coro_p, lv_p))

src = (ROOT / "multi_dicomviewer/ui/main_window.py").read_text(encoding="utf-8")
check("Territory ▸ Load uses it before the replace question",
      "if confirm and self._territory_links_match(" in src)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
