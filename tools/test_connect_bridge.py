# -*- coding: utf-8 -*-
"""接続 closes the gap (user 2026-11-27): a line connected to its parent
from a few mm away is extended in a straight line (the shortest way) to
its connection point, so it visibly joins ("つながってない？" no more).

  - 接続: the line's proximal end = the connection point on the parent;
    the extension is straight; the rest of the line is unchanged
  - its .cpr.json is rewritten (the drawing direction kept — a line drawn
    distal → proximal gets the new point at its END), so フォルダ更新 keeps it
  - a line already touching its parent is left alone
  - 親を変更 (right-click) does the same

    python tools/test_connect_bridge.py
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
from PyQt6.QtWidgets import (QApplication, QInputDialog,  # noqa: E402
                             QMessageBox)

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
warned = []
QMessageBox.warning = staticmethod(lambda *a, **k: warned.append(a))
QMessageBox.information = staticmethod(lambda *a, **k: warned.append(a))
from multi_dicomviewer.core.centerline import CenterLine  # noqa: E402
from multi_dicomviewer.ui.coronary_tree_window import CoronaryTreeWindow  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


D = tempfile.mkdtemp(dir=HOME)
UID = "bridge-ct"


def write_cpr(path, ctrl):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"format": "MDV-CPR", "version": 1, "type": "cpr",
                   "series": {"series_uid": UID}, "src_dir": "",
                   "ctrl": np.asarray(ctrl, float).tolist(),
                   "state": {"idx": 0}, "auto_traced": True}, f)


def read_ctrl(path):
    with open(path, encoding="utf-8") as f:
        return np.asarray(json.load(f)["ctrl"], float)


lad_ctrl = np.c_[np.linspace(0, 40, 9), np.zeros(9), np.zeros(9)]
# D1 drawn FROM near the LAD (start 3 mm off it at x=20)
d1_ctrl = np.array([[20.0, 3.0, 0.0], [20.0, 12.0, 0.0], [20.0, 22.0, 0.0]])
# D2 drawn the other way (distal → proximal), its end 4 mm off at x=30
d2_ctrl = np.array([[30.0, -24.0, 0.0], [30.0, -14.0, 0.0], [30.0, -4.0, 0.0]])
# D3 touches the LAD already
d3_ctrl = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 10.0], [10.0, 0.0, 20.0]])
paths = {}
for vid, c in (("d1", d1_ctrl), ("d2", d2_ctrl), ("d3", d3_ctrl)):
    paths[vid] = os.path.join(D, vid + ".cpr.json")
    write_cpr(paths[vid], c)

w = CoronaryTreeWindow(None)
w._use_uid(UID)
b = w._bundle()
tree = b["tree"]
tree.add_root("lad", "LAD", "LAD",
              CenterLine.from_points(lad_ctrl, step_mm=0.5).points)
tree.vessels["lad"].ctrl = lad_ctrl
for vid, c in (("d1", d1_ctrl), ("d2", d2_ctrl), ("d3", d3_ctrl)):
    tree.add_vessel(vid, vid.upper(), "branch",
                    CenterLine.from_points(c, step_mm=0.5).points, ctrl=c)
    b["paths"][vid] = paths[vid]
d3_before = tree.vessels["d3"].points.copy()

w._connect()
hint = w._hint.text()
check("all three connected", all(tree.vessels[k].parent == "lad"
                                 for k in ("d1", "d2", "d3")), hint)
check("the message says 2 lines were extended",
      ("2 本は合流点まで" in hint) or ("2 extended" in hint), hint)
for vid, J in (("d1", (20.0, 0.0, 0.0)), ("d2", (30.0, 0.0, 0.0))):
    v = tree.vessels[vid]
    A = np.asarray(tree.anchor(vid))
    check(f"{vid}: the proximal end IS the connection point",
          np.linalg.norm(v.points[0] - A) < 0.05
          and np.linalg.norm(A - np.array(J)) < 0.3, (v.points[0], A))
    seg = v.points[v.points[:, 1] * np.sign(v.points[-1, 1]) <= 4.01]
    off = np.abs(seg[:, 0] - J[0]).max()
    check(f"{vid}: the extension is straight (shortest way)", off < 0.05, off)
    gaps = np.linalg.norm(np.diff(v.points, axis=0), axis=1)
    check(f"{vid}: no gap left along the line", gaps.max() < 0.8, gaps.max())
c1 = read_ctrl(paths["d1"])
check("d1 file: the connection point added at its START (drawn from the LAD)",
      np.allclose(c1[0], [20, 0, 0], atol=0.3) and np.allclose(c1[1:], d1_ctrl),
      c1)
c2 = read_ctrl(paths["d2"])
check("d2 file: drawn distal → proximal, the point added at its END",
      np.allclose(c2[-1], [30, 0, 0], atol=0.3) and np.allclose(c2[:-1], d2_ctrl),
      c2)
with open(paths["d1"], encoding="utf-8") as f:
    check("its marks are kept (auto_traced)", json.load(f).get("auto_traced"))
check("d3 (already touching): left alone",
      np.allclose(tree.vessels["d3"].points, d3_before)
      and np.allclose(read_ctrl(paths["d3"]), d3_ctrl))

# 親を変更 (right-click) closes the gap too
d4_ctrl = np.array([[5.0, 2.5, 0.0], [5.0, 12.0, 0.0]])
p4 = os.path.join(D, "d4.cpr.json")
write_cpr(p4, d4_ctrl)
tree.add_vessel("d4", "D4", "branch",
                CenterLine.from_points(d4_ctrl, step_mm=0.5).points, ctrl=d4_ctrl)
b["paths"]["d4"] = p4
w._populate()
QInputDialog.getItem = staticmethod(lambda *a, **k: ("LAD", True))
w.selected_vid = lambda: "d4"
for name in ("_set_parent", "_change_parent", "_reparent"):
    if hasattr(w, name):
        getattr(w, name)()
        break
v4 = tree.vessels["d4"]
check("親を変更: connected and extended to the LAD",
      v4.parent == "lad" and np.linalg.norm(v4.points[0] - [5, 0, 0]) < 0.3,
      (v4.parent, v4.points[0]))

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(0 if not fails else 1)
