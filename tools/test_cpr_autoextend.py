# -*- coding: utf-8 -*-
"""CPR 自動追跡 (user 2026-11-18): extend a line distally through the lumen,
the trunk continuing (A案: the clearly wider / straighter way), every other
branch its own line hung on its parent; never guessing (ambiguous
bifurcation → the trunk stops, both become branches; no jumping a gap or
onto a neighbour; stop before a chamber).

Synthetic volumes (0.5 mm voxels, 400 HU tubes in 40 HU, noise 20 HU), the
dock's auto_extend (files + serial names + parent links) and the viewer's
distal-edit-point 自動追跡 on a fake host.

    python tools/test_cpr_autoextend.py
"""
import copy
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
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.information = staticmethod(lambda *a, **k: None)
from multi_dicomviewer.core.centerline import CenterLine  # noqa: E402
from multi_dicomviewer.core.cpr_autoextend import extend_vessel  # noqa: E402
from multi_dicomviewer.ui.coronary_tree_window import CoronaryTreeWindow  # noqa: E402
from multi_dicomviewer.viewers.cpr_mixin import CPRMixin  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


sp = 0.5
nz, ny, nx = 160, 120, 120
z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32) * sp
D3 = (sp, sp, sp)


def new():
    return np.full((nz, ny, nx), 40.0, np.float32)


def tube(vol, P0, P1, r, hu=400.0):
    P0 = np.asarray(P0, float)
    P1 = np.asarray(P1, float)
    d = P1 - P0
    L = np.linalg.norm(d)
    d /= L
    X = np.stack([x - P0[0], y - P0[1], z - P0[2]], -1)
    t = np.clip(X @ d, 0, L)
    dist = np.linalg.norm(X - t[..., None] * d, axis=-1)
    vol[dist <= r] = hu


def noise(vol, s=0):
    return vol + np.random.default_rng(s).normal(
        0, 20, vol.shape).astype(np.float32)


ctrl = np.array([[30, 30, 8.0], [30, 30, 14.0], [30, 30, 20.0]])

# 1. trunk + a thinner side branch
v = new()
tube(v, [30, 30, 5], [30, 30, 40], 1.8)
tube(v, [30, 30, 40], [30, 30, 75], 1.5)
tube(v, [30, 30, 40], [50, 30, 60], 1.0)
VY = noise(v)
r = extend_vessel(VY, D3, ctrl)
tp = r.trunk.path if r.trunk is not None else np.zeros((1, 3))
check("Y: the trunk continues to its end, centred",
      tp[-1][2] > 72 and np.abs(tp[:, :2] - 30).max() < 0.6, tp[-1].round(1))
check("Y: the side branch is its own line, traced to its end",
      len(r.branches) == 1 and r.branches[0].parent == -1
      and np.linalg.norm(r.branches[0].path[-1] - [50, 30, 60]) < 2.0
      and abs(r.branches[0].path[0][2] - 40) < 2.5)
check("edit points about 3 mm apart",
      r.trunk is not None and 2.0 < np.median(np.linalg.norm(
          np.diff(r.trunk.ctrl, axis=0), axis=1)) < 4.0)

# 2. two equal daughters: no trunk decision
v = new()
tube(v, [30, 30, 5], [30, 30, 40], 1.6)
tube(v, [30, 30, 40], [45, 30, 65], 1.3)
tube(v, [30, 30, 40], [15, 30, 65], 1.3)
r = extend_vessel(noise(v), D3, ctrl)
ends = sorted(round(float(b.path[-1][0])) for b in r.branches)
check("equal bifurcation: the trunk stops there, both daughters = branches",
      r.stop == "bifurcation" and len(r.branches) == 2
      and abs(ends[0] - 15) < 2 and abs(ends[1] - 45) < 2, (r.stop, ends))

# 3. a gap of 4.5 mm fat, then another vessel on the same axis
v = new()
tube(v, [30, 30, 5], [30, 30, 40], 1.5)
tube(v, [30, 30, 44.5], [30, 30, 75], 1.5)
r = extend_vessel(noise(v, 1), D3, ctrl)
check("no jump across a gap", r.trunk is not None
      and r.trunk.path[:, 2].max() < 42 and not r.branches)

# 4. a neighbour beside the trunk's end
v = new()
tube(v, [30, 30, 5], [30, 30, 40], 1.5)
tube(v, [34.5, 30, 30], [34.5, 30, 75], 1.5)
r = extend_vessel(noise(v, 2), D3, ctrl)
check("no jump onto a neighbouring vessel", r.trunk is not None
      and r.trunk.path[:, 2].max() < 42
      and np.abs(r.trunk.path[:, 0] - 30).max() < 1.0 and not r.branches)

# 5. runs into a chamber
v = new()
tube(v, [30, 30, 5], [30, 30, 48], 1.5)
v[(np.sqrt((x - 30) ** 2 + (y - 30) ** 2 + (z - 60) ** 2) < 14)] = 400
r = extend_vessel(noise(v, 3), D3, ctrl)
check("stops before a chamber", r.trunk is not None
      and r.trunk.path[:, 2].max() < 47.5 and not r.branches,
      r.trunk.path[-1].round(1) if r.trunk is not None else None)

# 6. a curve (radius 25 mm)
v = new()
tube(v, [30, 30, 5], [30, 30, 20], 1.4)
cx, cz, R = 55.0, 20.0, 25.0
ang = np.linspace(np.pi, np.pi * 0.45, 60)
P = np.c_[cx + R * np.cos(ang), np.full(60, 30.0), cz + R * np.sin(ang)]
for a, b in zip(P[:-1], P[1:]):
    tube(v, a, b, 1.4)
r = extend_vessel(noise(v, 4), D3, ctrl)
dev = np.abs(np.hypot(r.trunk.path[:, 0] - cx, r.trunk.path[:, 2] - cz) - R)
check("follows a curve (within 0.5 mm, to its end)",
      dev[r.trunk.path[:, 2] > 20.5].max() < 0.5
      and np.linalg.norm(r.trunk.path[-1] - P[-1]) < 3.0)

# ---- dock: registered line → extended + branch registered ----------------
DIR = tempfile.mkdtemp(dir=HOME)
UID = "ae-ct"
p01 = os.path.join(DIR, "01.cpr.json")
with open(p01, "w", encoding="utf-8") as f:
    json.dump({"format": "MDV-CPR", "version": 1, "type": "cpr",
               "series": {"series_uid": UID}, "src_dir": "",
               "ctrl": ctrl.tolist(), "state": {"idx": 3}}, f)
w = CoronaryTreeWindow(None)
w._use_uid(UID)
b = w._bundle()
tree = b["tree"]
tree.add_root("lad", "01", "LAD",
              CenterLine.from_points(ctrl, step_mm=0.5).points)
tree.vessels["lad"].ctrl = ctrl.copy()
b["paths"]["lad"] = p01
out = w.auto_extend("lad", VY, D3)
lad = tree.vessels["lad"]
with open(p01, encoding="utf-8") as f:
    f01 = json.load(f)
check("dock: the line itself is extended (file rewritten)",
      out is not None and out["trunk_mm"] > 45 and lad.points[-1][2] > 72
      and f01["ctrl"][-1][2] > 72, out)
kids = [vv for vv in tree.vessels.values() if vv.parent == "lad"]
check("dock: the branch = next serial line 02, hung on it at the fork",
      len(kids) == 1 and kids[0].name == "02" and kids[0].auto_traced
      and os.path.isfile(os.path.join(DIR, "02.cpr.json"))
      and abs(lad.points[kids[0].junction][2] - 40) < 2.5)
check("dock: result text", "mm" in w._hint.text(), w._hint.text())


# ---- viewer: distal edit point → 自動追跡 (fake host) -----------------------
class Shell:
    def __init__(self):
        self.calls = []

    def coronary_add_auto_branches(self, v, path, segs):
        self.calls.append(("branches", path, len(segs)))
        return ["x1"] * len(segs)

    def coronary_auto_extend_report(self, v, out, name="", w=None):
        self.calls.append(("report", out["trunk_mm"], len(out["branches"])))


class Host(QWidget, CPRMixin):
    def __init__(self):
        QWidget.__init__(self)
        self._vol = VY
        self._dims = D3
        self._mode = "3D"
        self._coronary_mode = True
        self._cpr = None
        self._vr_on = {"A": False, "B": False}
        self._measures = {"A": [], "B": [{
            "id": 1, "type": "polyline", "_cpr_src": True,
            "pts3d": ctrl.tolist(),
            "pts": [(float(p[0]), float(p[2])) for p in ctrl]}]}
        self.sh = Shell()
        self.undo = []
        self.saved = 0
        self._cpr_last_path = p01

    def _ct_mode_shell(self):
        return self.sh

    def _world3d_to_out(self, key, P):
        return (float(P[0]), float(P[2]))

    def _meas_pane_snap(self, pane):
        return copy.deepcopy(self._measures[pane])

    def _meas_record(self, pane, before):
        self.undo.append((pane, before))

    def _redraw_meas(self, key):
        pass

    def _cpr_ensure_built(self, target=None):
        return True

    def _cpr_register_to_tree(self, save_as=False, rearm=True):
        self.saved += 1
        return True


h = Host()
check("自動追跡 offered on the distal edit point only",
      h._cpr_autotrack_ok("B", 0, 2) and not h._cpr_autotrack_ok("B", 0, 0))
res = h._cpr_autotrack("B", 0)
m = h._measures["B"][0]
check("viewer: the line is extended in place (one undo step)",
      len(m["pts3d"]) > 15 and len(m["pts"]) == len(m["pts3d"])
      and m["pts3d"][-1][2] > 72 and len(h.undo) == 1)
check("viewer: saved + its branch registered on it",
      h.saved == 1 and ("branches", p01, 1) in h.sh.calls
      and res["branches"] == ["x1"], h.sh.calls)
for nm in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    s = (ROOT / "multi_dicomviewer/viewers" / nm).read_text(encoding="utf-8")
    check(f"{nm}: handle menu offers 自動追跡",
          "self._cpr_autotrack_ok(which, mi, vi)" in s
          and "self._cpr_autotrack(which, mi)" in s)
s = (ROOT / "multi_dicomviewer/ui/main_window.py").read_text(encoding="utf-8")
check("shell: auto-extend entry points",
      "def coronary_auto_extend" in s and "def coronary_add_auto_branches" in s)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
