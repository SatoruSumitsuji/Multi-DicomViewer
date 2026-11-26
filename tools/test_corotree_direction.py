# -*- coding: utf-8 -*-
"""Drawn direction vs connected proximal / distal (user 2026-11-27): branches
are now drawn proximal → distal, older ones distal → proximal, and a D9a
territory came out wrong. The connection point on the parent (anchor, 3-D)
now decides the proximal end, and every junction FOLLOWS it when the
parent's line is replaced (re-edit / Auto-Track / file refresh), reversed,
split or merged — an index alone went stale. 接続 prefers the drawing
start when both ends are near the tree. Synthetic lines, no patient data.

    python tools/test_corotree_direction.py
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

from multi_dicomviewer.core.coronary_territory import CoronaryTree  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


def line(a, b, n=None, step=1.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = n or max(2, int(np.linalg.norm(b - a) / step) + 1)
    return np.linspace(a, b, n)


def build():
    tr = CoronaryTree()
    tr.add_root("lad", "LAD", "LAD", line([0, 0, 0], [0, 0, 80]))
    # D9 leaves the LAD at z = 30, drawn proximal → distal (new way)
    tr.add_vessel("d9", "D9", "branch", line([1, 0, 30], [30, 0, 50]))
    # D9a leaves D9 at its middle, drawn distal → proximal (old way)
    mid = line([1, 0, 30], [30, 0, 50])[15]
    tr.add_vessel("d9a", "D9a", "branch", line(mid + [10, 12, 6], mid + [0.5, 0.5, 0]))
    return tr, mid


def near(tr, vid, P, tol=1.6):
    v = tr.vessels[vid]
    p = tr.vessels[v.parent]
    return float(np.linalg.norm(p.points[v.junction] - P)) <= tol


def prox_end(tr, vid):
    v = tr.vessels[vid]
    return v.points[tr.proximal_idx(vid)]


tr, mid = build()
res = tr.connect_all(snap_tol_mm=5.0)
check("接続: D9 and D9a connected", set(res["connected"]) == {"d9", "d9a"},
      res)
check("D9a's junction on D9 at the branch point", near(tr, "d9a", mid))
check("D9a's proximal end is the end at D9",
      np.linalg.norm(prox_end(tr, "d9a") - mid) < 1.5, prox_end(tr, "d9a"))
_, st, full = tr.distal_set("d9", tr.proximal_idx("d9"))
check("D9's territory: D9a fully distal", "d9a" in full, full)

# the parent re-edited in the DRAWN (file) order = reversed vs the tree, and
# a little longer (Auto-Track): D9a's junction must follow the 3-D point
d9 = tr.vessels["d9"]
new = line([1, 0, 30], [36, 0, 54])[::-1]
flipped = tr.replace_points("d9", new, new.copy())
check("re-edit: the tree keeps D9's direction (proximal first)",
      flipped and np.linalg.norm(d9.points[0] - [1, 0, 30]) < 1e-6)
check("… D9a's junction followed (same place on D9)", near(tr, "d9a", mid))
check("… D9a's proximal end still at D9",
      np.linalg.norm(prox_end(tr, "d9a") - mid) < 1.5)
_, st, full = tr.distal_set("d9", tr.proximal_idx("d9"))
check("… D9's territory still holds D9a", "d9a" in full)
# a target on D9 BEFORE the D9a branch → D9a distal; AFTER → not
j = tr.vessels["d9a"].junction
check("target on D9 before D9a: D9a distal",
      "d9a" in tr.distal_set("d9", max(0, j - 5))[2])
check("target on D9 after D9a: D9a not distal",
      "d9a" not in tr.distal_set("d9", min(d9.n - 1, j + 5))[2])

# reversing the parent (右クリック 逆向き): junction remapped
tr.reverse_vessel("d9")
check("parent reversed: D9a's junction remapped to the same place",
      near(tr, "d9a", mid))
check("… target after D9a (now lower index): D9a not distal",
      "d9a" not in tr.distal_set(
          "d9", max(0, tr.vessels["d9a"].junction - 5))[2])
check("… target before D9a: D9a distal",
      "d9a" in tr.distal_set(
          "d9", min(d9.n - 1, tr.vessels["d9a"].junction + 5))[2])
tr.reverse_vessel("d9")

# the child itself re-edited in the other order: proximal end unchanged
c = tr.vessels["d9a"].points.copy()
tr.replace_points("d9a", c[::-1], c[::-1].copy())
check("child re-edit in the other order: its proximal end stays at D9",
      np.linalg.norm(prox_end(tr, "d9a") - mid) < 1.5)

# an old saved tree with a STALE junction index → derived from the geometry
data = tr.to_json()
for d in data["vessels"]:
    if d["vid"] == "d9a":
        d["junction"] = 1                      # far from D9a's ends
t2 = CoronaryTree.from_json(json.loads(json.dumps(data)))
check("old file, stale junction: the proximal end is still the end at D9",
      np.linalg.norm(prox_end(t2, "d9a") - mid) < 1.5
      and near(t2, "d9a", mid, tol=2.0))
check("the save format is unchanged (no anchor field)",
      all("anchor" not in d for d in tr.to_json()["vessels"]))

# 接続 with BOTH ends near the tree: the drawing start wins
t3 = CoronaryTree()
t3.add_root("lad", "LAD", "LAD", line([0, 0, 0], [0, 0, 80]))
t3.add_root("lcx", "LCX", "LCX", line([20, 0, 0], [20, 0, 80]))
# drawn from the LAD (start 2.0 mm off) to near the LCX (end 1.2 mm off)
t3.add_vessel("x", "X", "branch", line([2.0, 0, 40], [18.8, 0, 40]))
t3.connect_all(snap_tol_mm=5.0)
check("both ends near: attached at the drawing start (LAD)",
      t3.vessels["x"].parent == "lad", t3.vessels["x"].parent)
t4 = CoronaryTree()
t4.add_root("lad", "LAD", "LAD", line([0, 0, 0], [0, 0, 80]))
t4.add_root("lcx", "LCX", "LCX", line([20, 0, 0], [20, 0, 80]))
t4.add_vessel("x", "X", "branch", line([4.5, 0, 40], [19.5, 0, 40]))
t4.connect_all(snap_tol_mm=5.0)
check("… unless the other end is clearly nearer (> 1.5 mm)",
      t4.vessels["x"].parent == "lcx", t4.vessels["x"].parent)

# Coronary Tree dock: 上書き保存 of the parent in the drawn order, a split of
# a connected line stored distal → proximal
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
for _n in ("information", "warning", "critical", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("unexpected dialog"))))
from multi_dicomviewer.ui import coronary_tree_window as ctw  # noqa: E402

w = ctw.CoronaryTreeWindow(None)
w._show_source_ct = lambda *a, **k: ""
w._use_uid("direction-ct")
tw, mid = build()
tw.connect_all(snap_tol_mm=5.0)
w._bundle()["tree"] = tw
w._tree.vessels  # noqa: B018
D = tempfile.mkdtemp(dir=HOME)
for vid in tw.vessels:
    pth = os.path.join(D, vid + ".cpr.json")
    with open(pth, "w", encoding="utf-8") as f:
        json.dump({"format": "MDV-CPR", "ctrl": tw.vessels[vid].points.tolist()}, f)
    w._bundle()["paths"][vid] = pth
w._populate()
drawn = line([1, 0, 30], [30, 0, 50])[::-1]
check("dock 上書き保存 (drawn order)", w.update_vessel_cpr("d9", drawn))
check("… D9 kept proximal first, D9a's junction followed",
      np.linalg.norm(w._tree.vessels["d9"].points[0] - [1, 0, 30]) < 1e-6
      and near(w._tree, "d9a", mid))
# make D9 stored distal → proximal in the tree, then separate it
w._tree.reverse_vessel("d9")
v9 = w._tree.vessels["d9"]
check("(setup) D9 now runs distal → proximal", w._tree.runs_reversed("d9"))
cut = int(v9.n * 0.8)                         # near D9's proximal end
newv = w.split_vessel("d9", cut, folder=D)
v9 = w._tree.vessels["d9"]
check("separate: D9 keeps its PROXIMAL part (at the LAD)",
      bool(newv) and np.linalg.norm(v9.points[0] - [1, 0, 30]) < 1.5
      and not w._tree.runs_reversed("d9"),
      v9.points[[0, -1]].round(1).tolist())
check("… the distal part hangs on D9's end",
      w._tree.vessels[newv].parent == "d9")
check("… D9a (beyond the cut) moved to the distal part, same place",
      w._tree.vessels["d9a"].parent == newv and near(w._tree, "d9a", mid),
      w._tree.vessels["d9a"].parent)

# .corotree.json load: a parent that 接続 turned round (its .cpr.json keeps
# the drawn order) is NOT taken as changed — the child's junction stays
D2 = tempfile.mkdtemp(dir=HOME)
t5 = CoronaryTree()
t5.add_root("lad", "LAD", "LAD", line([0, 0, 0], [0, 0, 80]))
d9_drawn = line([30, 0, 50], [1, 0, 30])           # drawn distal → proximal
t5.add_vessel("d9", "D9", "branch", d9_drawn, ctrl=d9_drawn.copy())
mid5 = line([1, 0, 30], [30, 0, 50])[15]
d9a_drawn = line(mid5 + [0.5, 0.5, 0], mid5 + [10, 12, 6])
t5.add_vessel("d9a", "D9a", "branch", d9a_drawn, ctrl=d9a_drawn.copy())
t5.connect_all(snap_tol_mm=5.0)
check("(setup) 接続 turned D9 round in the tree",
      np.linalg.norm(t5.vessels["d9"].points[0] - [1, 0, 30]) < 1e-6)
w5 = ctw.CoronaryTreeWindow(None)
w5._show_source_ct = lambda *a, **k: ""
w5._use_uid("direction-ct5")
w5._bundle()["tree"] = t5
for vid, drawn_pts in (("lad", t5.vessels["lad"].points), ("d9", d9_drawn),
                       ("d9a", d9a_drawn)):
    pth = os.path.join(D2, t5.vessels[vid].name + ".cpr.json")
    with open(pth, "w", encoding="utf-8") as f:
        json.dump({"format": "MDV-CPR", "ctrl": np.asarray(drawn_pts).tolist()}, f)
    w5._bundle()["paths"][vid] = pth
tp = os.path.join(D2, "t.corotree.json")
w5._write_to(tp)
w6 = ctw.CoronaryTreeWindow(None)
w6._show_source_ct = lambda *a, **k: ""
check("reload the tree", w6.load_file(tp))
t6 = w6._tree
check("… D9 not 'refreshed' from its drawn-order file (same line)",
      np.linalg.norm(t6.vessels["d9"].points[0] - [1, 0, 30]) < 1e-6)
check("… D9a's junction still at its branch point", near(t6, "d9a", mid5))
check("… D9's territory holds D9a",
      "d9a" in t6.distal_set("d9", t6.proximal_idx("d9"))[2])

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
