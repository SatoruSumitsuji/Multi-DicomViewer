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
shown = []
QMessageBox.warning = staticmethod(lambda *a, **k: shown.append(("warn", a)))
QMessageBox.information = staticmethod(lambda *a, **k: shown.append(("info", a)))
from multi_dicomviewer.core.centerline import CenterLine  # noqa: E402
import multi_dicomviewer.core.cpr_autoextend as CAE  # noqa: E402
# The trunk now STOPS at a fork for the user to choose (ask_at_fork). The
# automatic A案 choice (still used for the side branches) is tested with
# ask_at_fork off — also through the viewer / dock (patched module func).
_EV_ASK = CAE.extend_vessel


def extend_vessel(*a, prm=None, **k):
    return _EV_ASK(*a, prm=prm or CAE.ExtendParams(ask_at_fork=False), **k)


CAE.extend_vessel = extend_vessel
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

    _AUTOTRACK_STEP_S = 0.0
    _AUTOTRACK_DRAW_S = 0.0     # draw every step (the live-view checks)

    def _redraw_meas(self, key):
        pass

    def _refresh(self, *a, **k):
        # live view: how many edit points the line had at each refresh
        self.__dict__.setdefault("seen", []).append(
            len(self._measures["B"][0]["pts3d"]))

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
# a tight bend (curvature radius 3.5 mm, r 1.6) is followed, not stopped as
# "turn"; a 2-point start drawn 25° off the vessel still calibrates
def _bend_vol(Rc, r):
    sx_, sy_, sz_ = 0.4, 0.4, 0.5
    zz, yy, xx = np.mgrid[0:120, 0:200, 0:150].astype(np.float32)
    Xb, Yb, Zb = xx * sx_, yy * sy_, zz * sz_
    vb = np.full(Xb.shape, -80.0, np.float32)
    cen = [(10 + t, 30, 30) for t in np.arange(0, 15, 0.2)]
    cen += [(25 + Rc * np.sin(a), 30 + Rc * (1 - np.cos(a)), 30)
            for a in np.arange(0, np.pi, 0.2 / Rc)]
    cen += [(25 - t, 30 + 2 * Rc, 30) for t in np.arange(0, 15, 0.2)]
    for c in cen:
        sl = (slice(int((c[2] - r - 1) / sz_), int((c[2] + r + 1) / sz_) + 1),
              slice(int((c[1] - r - 1) / sy_), int((c[1] + r + 1) / sy_) + 1),
              slice(int((c[0] - r - 1) / sx_), int((c[0] + r + 1) / sx_) + 1))
        m = ((Xb[sl] - c[0]) ** 2 + (Yb[sl] - c[1]) ** 2
             + (Zb[sl] - c[2]) ** 2 <= r * r)
        vb[sl][m] = 380
    vb += np.random.default_rng(3).normal(0, 25, vb.shape).astype(np.float32)
    return vb, (sx_, sy_, sz_), 5 + np.pi * Rc + 15
for off in (0, 25):
    vb, db, full = _bend_vol(3.5, 1.6)
    o = np.radians(off)
    p2 = np.array([20.0, 30, 30])
    p1 = p2 - 8 * np.array([np.cos(o), np.sin(o), 0])
    rb = extend_vessel(vb, db, np.array([p1, p2]))
    Lb = (0.0 if rb.trunk is None else float(np.linalg.norm(
        np.diff(rb.trunk.path, axis=0), axis=1).sum()))
    check(f"tight bend followed (start {off}° off): {Lb:.0f}/{full:.0f} mm",
          rb.stop != "turn" and Lb > 0.85 * full, rb.stop)
# a line ending AT / past the LM carina (LAD 45°, LCX 90° off its axis):
# was 0 mm 「急カーブ」; now both found as branches, no duplicate
def _carina_vol():
    sx_, sy_, sz_ = 0.4, 0.4, 0.5
    zz, yy, xx = np.mgrid[0:120, 0:200, 0:200].astype(np.float32)
    Xc, Yc, Zc = xx * sx_, yy * sy_, zz * sz_
    vc = np.full(Xc.shape, -80.0, np.float32)
    def tb(P0, P1, r):
        P0 = np.asarray(P0, float); P1 = np.asarray(P1, float)
        dd = P1 - P0; LL = np.linalg.norm(dd); dd /= LL
        Qq = np.stack([Xc - P0[0], Yc - P0[1], Zc - P0[2]], -1)
        tt = np.clip(Qq @ dd, 0, LL)
        vc[np.linalg.norm(Qq - tt[..., None] * dd, axis=-1) <= r] = 380
    Bc = np.array([40.0, 40, 30])
    tb([20, 40, 30], Bc, 2.3)
    tb(Bc, Bc + 35 * np.array([np.cos(np.radians(45)),
                               np.sin(np.radians(45)), 0]), 1.8)
    tb(Bc, Bc + 34 * np.array([0, -1.0, 0]), 1.6)
    vc += np.random.default_rng(1).normal(0, 25, vc.shape).astype(np.float32)
    return vc, (sx_, sy_, sz_)
vc, dc = _carina_vol()
rc = extend_vessel(vc, dc, np.array([[23.0, 40, 30], [30, 40, 30],
                                     [41.0, 40, 30]]))
ends = [b.path[-1] for b in rc.branches]
check("line ending at the carina: LAD + LCX as branches (not 「急カーブ」)",
      rc.stop != "turn" and len(rc.branches) == 2
      and any(e[0] > 60 and e[1] > 60 for e in ends)
      and any(e[1] < 10 for e in ends),
      (rc.stop, [np.round(e, 0) for e in ends]))
# diagnostic log: every step recorded, saved under the (fake) home only
from multi_dicomviewer.core.cpr_autoextend import save_log  # noqa: E402
rl = extend_vessel(VY, D3, ctrl)
steps = [e for e in rl.log if "step" in e]
lp = save_log(rl.log)
with open(lp, encoding="utf-8") as f:
    jl = json.load(f)
check("diagnostic log: steps with position / radius / candidates, saved "
      "under the settings folder",
      len(steps) >= 10 and all("P" in e and "r" in e and "cands" in e
                               for e in steps)
      and os.path.abspath(lp).startswith(os.path.abspath(HOME))
      and jl["format"] == "MDV-AutoTrackLog"
      and not any(k in json.dumps(jl).lower()
                  for k in ("patient", "name\"")), lp)
txt = CoronaryTreeWindow.auto_extend_text(
    "x", {"trunk_mm": 5.0, "branches": [], "stop": "lumen", "log_path": lp})
check("result text names the log file", lp in txt)
# live view: the line grows one step at a time, the right pane turns to
# contain the course and centres on the new point
class LiveHost(Host):
    def __init__(self):
        super().__init__()
        self._frame = {"A": (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]),
                       "B": (np.eye(3)[0], np.eye(3)[2], -np.eye(3)[1])}
        self._pc = {"A": np.zeros(3), "B": np.zeros(3)}
        self._cross_ang = {"A": 0.0, "B": 0.0}
        self._center = np.zeros(3)
        self.views = []

    def _ortho(self, u, v):
        u = u / np.linalg.norm(u)
        v = v - (v @ u) * u
        v = v / np.linalg.norm(v)
        return (u, v, np.cross(u, v))

    def _cpr_autotrack_view(self, which, P, course):
        super()._cpr_autotrack_view(which, P, course)
        self.views.append((np.asarray(P, float).copy(),
                           np.asarray(course, float).copy(),
                           tuple(np.asarray(x).copy()
                                 for x in self._frame["B"])))


check("full speed by default: no pause per step, the view redrawn "
      "at most every 0.15 s (user 2026-11-26)",
      CPRMixin._AUTOTRACK_STEP_S == 0.0
      and 0.0 < CPRMixin._AUTOTRACK_DRAW_S <= 0.2)
lh = LiveHost()
lh._cpr_autotrack("B", 0)
seen = lh.__dict__.get("seen", [])
grow = [b - a for a, b in zip(seen, seen[1:])]
check("live: the line grows step by step (one point per refresh)",
      len(seen) >= 10 and max(grow) <= 1 and seen[-1] > 3, seen[:12])
okv = all(abs(abs(float(fr[0] @ (dc / np.linalg.norm(dc)))) - 1) < 1e-6
          and np.linalg.norm(lh._pc["B"] - lh.views[-1][0]) < 1e-9
          for (P_, dc, fr) in lh.views[-1:])
check("live: the right pane follows the course, centred on the new point",
      lh.views and okv)
# 停止 mid-way: the trace so far is kept, no branches traced after it
import multi_dicomviewer.ui.autotrack_progress as _atp  # noqa: E402
_orig_c = _atp.AutoTrackProgress.cancelled
_n = {"k": 0}


def _cancel_after(self):
    _n["k"] += 1
    return _n["k"] > 12


_atp.AutoTrackProgress.cancelled = _cancel_after
sh_ = Host()
rs = sh_._cpr_autotrack("B", 0)
_atp.AutoTrackProgress.cancelled = _orig_c
n_pts = len(sh_._measures["B"][0]["pts3d"])
check("停止: stops at once, keeps the trace so far (one undo step)",
      rs is not None and rs["stop"] == "cancel" and 3 < n_pts < 20
      and len(sh_.undo) == 1 and not rs["branches"], (rs, n_pts))
# 右クリックで停止: a right-click anywhere during Auto-Track stops it; other
# clicks there are swallowed; the filter is gone afterwards
from PyQt6.QtCore import QPointF  # noqa: E402
from PyQt6.QtCore import QEvent as _QE  # noqa: E402
from PyQt6.QtCore import Qt as _Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402

pane_w = QWidget()
pane_w.resize(100, 100)
pane_w.show()
_got_left = []


class _Spy(QWidget):
    def mousePressEvent(self, e):                       # noqa: N802
        _got_left.append(e.button())


spy = _Spy()
spy.resize(50, 50)
spy.show()


def _click(w, btn):
    ev = QMouseEvent(_QE.Type.MouseButtonPress, QPointF(5, 5),
                     QPointF(5, 5), btn, btn,
                     _Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(w, ev)


with _atp.AutoTrackProgress(None) as _pg:
    _click(spy, _Qt.MouseButton.LeftButton)
    c_left = _pg.cancelled()
    _click(pane_w, _Qt.MouseButton.RightButton)
    c_right = _pg.cancelled()
_click(spy, _Qt.MouseButton.LeftButton)
check("右クリックで停止: right-click stops; left clicks swallowed meanwhile",
      not c_left and c_right and _got_left == [_Qt.MouseButton.LeftButton],
      (c_left, c_right, _got_left))

# the red 「■ Auto-Track 停止」 button between the two images, just below
# their top, not too big (user 2026-11-25)
from multi_dicomviewer.i18n import t  # noqa: E402
from PyQt6.QtWidgets import QWidget as _QW  # noqa: E402
spy.resize(200, 150)
spy.move(300, 300)
spy2 = _QW()
spy2.resize(200, 150)
spy2.move(spy.geometry().right() + 1, 300)
spy2.show()
app.processEvents()
with _atp.AutoTrackProgress(None, anchor=[spy, spy2]) as _pg:
    sb = _pg._stop_btn
    sb_shown = (sb is not None and sb.isVisible()
                and sb.text() == t("■ Auto-Track 停止")
                and "#d62828" in sb.styleSheet()
                and "font-size:25px" in sb.styleSheet())
    top = spy.mapToGlobal(spy.rect().topLeft())
    right = spy2.mapToGlobal(spy2.rect().topRight())
    mid = (top.x() + right.x()) / 2.0
    at_top = (sb is not None and abs(sb.y() - (top.y() + 40)) <= 2
              and abs(sb.x() + sb.width() / 2.0 - mid) <= 2
              and 40 <= sb.height() < 80)
    from PyQt6.QtTest import QTest  # noqa: E402
    QTest.mouseClick(sb, _Qt.MouseButton.LeftButton)
    c_btn = _pg.cancelled()
    heard = (not sb.isEnabled()) and sb.text() == t("停止中…")
check("Auto-Track: the stop button sits between the images, below the top",
      sb_shown and at_top, (sb_shown, at_top, sb.geometry() if sb else None))
check("… clicking it stops (and it says 停止中…)", c_btn and heard)
check("… it is gone afterwards", _pg._stop_btn is None)


class RCHost(Host):
    """a right-click arrives while the 6th step is being shown"""
    def _refresh(self, *a, **k):
        self.__dict__.setdefault("nref", 0)
        self.nref += 1
        if self.nref == 6:
            _click(pane_w, _Qt.MouseButton.RightButton)


rc_ = RCHost()
rr_ = rc_._cpr_autotrack("B", 0)
n_rc = len(rc_._measures["B"][0]["pts3d"])
check("右クリックで停止 during a live Auto-Track: stops, keeps the trace",
      rr_ is not None and rr_["stop"] == "cancel" and 3 < n_rc < 15,
      (rr_ and rr_["stop"], n_rc))
# every step = the section 3 mm ahead: edit points / sections 3 mm apart
sp_ = np.linalg.norm(np.diff(extend_vessel(VY, D3, ctrl).trunk.path,
                             axis=0), axis=1)
check("steps are 3 mm apart (no shorter retry steps)",
      len(sp_) > 10 and sp_.min() > 2.8 and sp_.max() < 3.4,
      (round(float(sp_.min()), 2), round(float(sp_.max()), 2)))
# a short narrowing (2.5 mm of faint lumen) is bridged — still 3 mm apart
v = new()
tube(v, [30, 30, 5], [30, 30, 40], 1.5)
tube(v, [30, 30, 40], [30, 30, 42.5], 1.5, hu=180.0)
tube(v, [30, 30, 42.5], [30, 30, 75], 1.5)
rn_ = extend_vessel(noise(v, 5), D3, ctrl)
tpn = rn_.trunk.path if rn_.trunk is not None else np.zeros((2, 3))
spn = np.linalg.norm(np.diff(tpn, axis=0), axis=1)
# (a faint ROUND narrowing is now usually taken as the vessel itself —
# "faint round lumen", user 2026-11-25, or its HU peak, 2026-11-26 — else
# it is bridged)
check("a short narrowing is bridged, steps still 3 mm",
      tpn[-1][2] > 70 and spn.max() < 3.4
      and any(e.get("event") in ("bridged a narrowing", "faint round lumen",
                                 "HU peak")
              for e in rn_.log),
      (tpn[-1].round(1), round(float(spn.max()), 2)))
# why it stopped: the last 3 mm section's regions and why each was not
# taken (values) — in the log and in the result text
from multi_dicomviewer.core.cpr_autoextend import stop_detail  # noqa: E402
v = new()
tube(v, [30, 30, 5], [30, 30, 40], 1.5)
tube(v, [30, 30, 40], [30, 30, 75], 1.5, hu=240.0)
# (the faint-but-round / HU-peak searches would follow it: off, to test
# the report)
rf_ = extend_vessel(noise(v, 6), D3, ctrl,
                    prm=CAE.ExtendParams(ask_at_fork=False,
                                         round_rescue=False,
                                         peak_rescue=False))
dt_ = stop_detail(rf_.log)
txt_ = CoronaryTreeWindow.auto_extend_text(
    "x", {"trunk_mm": 10.0, "branches": [], "stop": rf_.stop, "detail": dt_})
check("stop detail: faint lumen ahead named with its HU and the need",
      "low_hu" in dt_.get("why", {}) and dt_["regions"] >= 1
      and dt_["why"]["low_hu"][0]["need"] > dt_["why"]["low_hu"][0]["hu"]
      and "HU" in txt_ and str(dt_["step"]) in txt_, txt_)
v = new()
tube(v, [30, 30, 5], [30, 30, 40], 1.5)
tube(v, [30, 30, 44.5], [30, 30, 75], 1.5)
dg_ = stop_detail(extend_vessel(noise(v, 1), D3, ctrl).log)
check("stop detail: a vessel beyond fat = not connected",
      "not_connected" in dg_.get("why", {}), dg_.get("why", {}).keys())
# the shell shows the result dialog even with the Coronary Tree dock closed
from types import SimpleNamespace  # noqa: E402
from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402
shown.clear()
MainWindow.coronary_auto_extend_report(
    SimpleNamespace(_corotree_win=None, statusBar=lambda: None), None,
    {"trunk_mm": 0.0, "branches": [], "stop": "lumen", "detail": dt_})
check("result dialog shown without the Coronary Tree dock", len(shown) == 1)
# the stop section: q = 3 mm beyond the last step's point along its course
lst = [e for e in rf_.log if e.get("line") == 0 and "step" in e][-1]
check("stop detail carries the stop section point (3 mm ahead)",
      dt_.get("q") is not None and abs(np.linalg.norm(
          np.asarray(dt_["q"]) - np.asarray(lst["P"])) - 3.0) < 0.05)
# a vessel in fat running next to LUNG (-800 HU 2 mm away on one side): the
# lung must not pull the background down (background = fixed 50 HU; was
# a measured bg -758 / threshold -198 →
# fat counted as lumen → one big "chamber" → stop at step 1)
v = np.full((nz, ny, nx), -90.0, np.float32)
v[x < 28.0] = -800.0                    # lung 2 mm beside the vessel (x = 30)
tube(v, [30, 30, 5], [30, 30, 75], 1.5, hu=360.0)
rl_ = extend_vessel(noise(v, 7), D3, ctrl)
hd_ = next(e for e in rl_.log if "bg" in e)
tl_ = rl_.trunk.path if rl_.trunk is not None else np.zeros((2, 3))
check("next to lung: background stays tissue, the vessel is traced",
      hd_["bg"] == 50.0 and tl_[-1][2] > 70 and rl_.stop != "widening",
      (hd_["bg"], rl_.stop, tl_[-1].round(1)))
# progress reported while tracing (trunk mm, then branch k / n)
stages = []
extend_vessel(VY, D3, ctrl, progress=lambda st, a, tot, k, n: stages.append(
    (st, k, n)))
check("progress: trunk then each branch reported",
      any(x[0] == "trunk" for x in stages)
      and ("branch", 1, 1) in stages, stages[-3:])
# a CT whose bundle key is "" (no SeriesInstanceUID): branches still added
w._by_uid[""] = w._by_uid.pop(UID)
w._use_uid("")
n0 = len(w._bundle()["tree"].vessels)
res2 = extend_vessel(VY, D3, ctrl)
made2 = w.add_auto_branches("lad", res2.branches)
check("CT without a UID: branches still registered (was the silent drop)",
      len(made2) == 1 and len(w._bundle()["tree"].vessels) == n0 + 1)
# no shell: the result still shows in a dialog; an error shows too
h2 = Host()
h2._ct_mode_shell = lambda: None
shown.clear()
h2._cpr_autotrack("B", 0)
check("result always shown in a dialog (no shell)",
      shown and shown[-1][0] == "info" and "Auto-Track" in shown[-1][1][1])
h3 = Host()
h3._vol = None
shown.clear()
h3._cpr_autotrack("B", 0)
check("an error is shown, not swallowed",
      shown and shown[-1][0] == "warn")
for nm in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    s = (ROOT / "multi_dicomviewer/viewers" / nm).read_text(encoding="utf-8")
    check(f"{nm}: handle menu offers 自動追跡",
          "self._cpr_autotrack_ok(which, mi, vi)" in s
          and "self._cpr_autotrack(which, mi)" in s
          and 't("Auto-Track")' in s)
s = (ROOT / "multi_dicomviewer/ui/main_window.py").read_text(encoding="utf-8")
check("shell: auto-extend entry points",
      "def coronary_auto_extend" in s and "def coronary_add_auto_branches" in s)

# ---- the trunk stops at a fork; the user chooses the lumen ---------------
CAE.extend_vessel = _EV_ASK
rk = _EV_ASK(VY, D3, ctrl)
check("fork: the trunk stops there with both lumens as candidates",
      rk.stop == "fork" and len(rk.fork) == 2 and not rk.branches
      and rk.fork[0]["ang"] <= rk.fork[1]["ang"]
      and rk.fork[0]["r"] > rk.fork[1]["r"], (rk.stop, rk.fork))
# following candidate k = a new trace from its centre
ck = np.asarray(rk.fork[1]["c"], float)
rk2 = _EV_ASK(VY, D3, np.vstack([rk.trunk.path[-2:], ck[None]]),
              guided=True)
check("fork: following the side candidate traces the side vessel "
      "(no new fork with the other lumen right beside)",
      rk2.trunk is not None and rk2.stop != "fork"
      and np.linalg.norm(rk2.trunk.path[-1] - [50, 30, 60]) < 3.5,
      rk2.trunk.path[-1].round(1) if rk2.trunk is not None else None)
from multi_dicomviewer.ui import autotrack_progress as _ap  # noqa: E402
_clicks = {"i": 1}
_orig_exec = QMessageBox.exec


def _fake_exec(self):
    bs = [b_ for b_ in self.buttons()
          if self.buttonRole(b_) == QMessageBox.ButtonRole.AcceptRole]
    tgt = [b_ for b_ in bs if b_.text().startswith(("候補 2", "Follow candidate 2"))]
    if tgt:
        tgt[0].click()
    return 0


QMessageBox.exec = _fake_exec
kk = _ap.show_result(None, "x", extra=["候補 1 へ進む", "候補 2 へ進む",
                                       "中心を指定して続ける"])
QMessageBox.exec = _orig_exec
check("result dialog with several buttons returns the one chosen", kk == 1)
txt_f = CoronaryTreeWindow.auto_extend_text("x", {
    "trunk_mm": 21.0, "branches": [], "stop": "fork",
    "fork_view": [dict(f, pos="右", dist=1.2) for f in rk.fork]})
check("fork text lists each candidate (position / diameter / angle)",
      txt_f.count("mm") >= 3 and ("2" in txt_f), txt_f)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
