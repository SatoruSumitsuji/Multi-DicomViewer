"""Territory ▸ Save / Load (.territory.json, core/territory_file + the
coronary-tree panel's save_territory / load_territory) on synthetic data.

The panel methods run on a light fake ``self`` (no dialogs: an explicit path).
Fake HOME — never the real settings.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_home = tempfile.mkdtemp(prefix="mdv_home_")
os.environ["HOME"] = os.environ["USERPROFILE"] = _home
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.core import full_lv as full_lv_mod  # noqa: E402
from multi_dicomviewer.core import territory_file as tf  # noqa: E402
from multi_dicomviewer.core.coronary_territory import CoronaryTree  # noqa: E402
from multi_dicomviewer.ui.coronary_tree_window import CoronaryTreeWindow  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


d = tempfile.mkdtemp(prefix="mdv_terr_")
coro_p = os.path.join(d, "case.corotree.json")
lv_p = os.path.join(d, "case.FullLv.json")
tree = CoronaryTree()
tree.add_root("lad", "LAD", "LAD",
              np.c_[np.arange(30.0), np.zeros(30), np.zeros(30)])
coro = tree.to_json(series={"series_uid": "1.2.3"})
full = {"format": full_lv_mod.FORMAT, "src": {"series_uid": "1.2.3"}}
json.dump(coro, open(coro_p, "w", encoding="utf-8"))
json.dump(full, open(lv_p, "w", encoding="utf-8"))

# ---- core/territory_file ------------------------------------------------
data, err = tf.build({"series_uid": "1.2.3"}, "", lv_p)
check("build refuses without a corotree link", data is None and "corotree" in err)
data, err = tf.build({"series_uid": "1.2.3"}, coro_p, "")
check("build refuses without a FullLv link", data is None and "FullLv" in err)
data, err = tf.build({"series_uid": "1.2.3"}, coro_p, lv_p,
                     targets=[{"vid": "lad", "idx": 7}, {"vid": None}],
                     params={"territory_opacity": 0.3},
                     result={"systems": ["LAD 40.0% / 30.0mL"]})
check("build: links + sha1 + targets + result",
      err == "" and tf.is_territory(data) and tf.corotree_path(data) == coro_p
      and len(tf.corotree_sha1(data)) == 40
      and tf.targets(data) == [{"vid": "lad", "idx": 7, "hidden": False}]
      and tf.series_uid(data) == "1.2.3")
check("unchanged sources → nothing reported",
      tf.changed_sources(data, coro_p, lv_p) == [])
with open(lv_p, "a", encoding="utf-8") as f:
    f.write("\n")
check("edited FullLv → reported", tf.changed_sources(data, coro_p, lv_p)
      == ["FullLv.json"])
tf.set_corotree_path(data, "/x/y.corotree.json")
check("re-link writes the new path", tf.corotree_path(data) == "/x/y.corotree.json")


# ---- panel save_territory / load_territory on a fake self ----------------
class Hint:
    def setText(self, s):
        self.text = s


class Fake:
    territory_default_path = CoronaryTreeWindow.territory_default_path
    save_territory = CoronaryTreeWindow.save_territory
    load_territory = CoronaryTreeWindow.load_territory

    def __init__(self):
        self._tree = tree
        self._territory = object()
        self._full_lv_data = full
        self._last_path = coro_p
        self._full_lv_path = lv_p
        self._ct_uid, self._ct_dir = "1.2.3", d
        self._myo_ml = 75.0
        self._targets = [{"vid": "lad", "idx": 5, "pct": 12.34, "ml": 9.25}]
        self._hidden = set()
        self._b = {}
        self._hint = Hint()
        self.warns = []

    def _warn(self, s):
        self.warns.append(s)

    def _bundle(self):
        return self._b

    def _corotree_series_block(self):
        return {}

    def _territory_summary_text(self):
        return "LAD 40.0% / 30.0mL\nRCA 60.0% / 45.0mL"

    def uid_for_series(self, u):
        return u

    def _use_uid(self, u, **k):
        self.used = u

    def _dedupe_tree(self):
        pass

    def _sync_terr_alpha_combo(self):
        pass

    def _populate(self):
        pass

    def _compute_territory(self):
        self._territory = "computed"


w = Fake()
check("default path: next to the FullLv, named after it",
      w.territory_default_path() == os.path.join(d, "case.territory.json"))
out = os.path.join(d, "saved")
ok = w.save_territory(out)
sp = out + ".territory.json"
check("save: .territory.json suffix enforced, file written",
      ok and os.path.isfile(sp) and w._b.get("terr_path") == sp)
saved = json.load(open(sp, encoding="utf-8"))
check("save: result lines use 'XX.X% / XX.XmL'",
      saved["result"]["systems"] == ["LAD 40.0% / 30.0mL", "RCA 60.0% / 45.0mL"]
      and saved["result"]["targets"] == ["Target-1 12.3% / 9.2mL"]
      and saved["result"]["myocardium_ml"] == 75.0, str(saved["result"]))
w2 = Fake()
w2._last_path = ""
check("save refuses while the tree is not saved",
      not w2.save_territory(out + "2") and w2.warns)

w3 = Fake()
w3._territory = None
w3._targets = []
ok = w3.load_territory(saved, coro, full, path=sp, coro_path=coro_p, lv_path=lv_p)
check("load: tree / FullLv / targets restored and recomputed",
      ok and w3._territory == "computed" and "lad" in w3._tree.vessels
      and w3._targets == [{"vid": "lad", "idx": 5, "hidden": False}]
      and w3._last_path == coro_p and w3._full_lv_path == lv_p
      and w3._b.get("terr_path") == sp and w3.used == "1.2.3")
check("load refuses a non-territory file",
      not Fake().load_territory({"format": "x"}, coro, full))

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
