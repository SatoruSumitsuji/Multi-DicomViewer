# -*- coding: utf-8 -*-
"""CPR line separate / connect + Polygon lasso (user 2026-11-18).

  - 「ここで separate」: a registered line is cut at the clicked sample; the
    proximal part keeps the vessel + file, the distal part gets the next
    serial name (NN, NN.cpr.json next to the original), hangs on the
    proximal end and takes the side branches beyond the cut
  - 「connect」: two line ends within the 接続 distance (5 mm) → ONE line
    (the line ending at the join keeps its identity; the other is appended,
    reversed if needed; its file goes to _merged/; its branches re-attach)
  - line ends are selected by Shift+click (toggle), Shift+drag rectangle or
    a Polygon (lasso); the Polygon also selects the open line's edit points
    (→ the bulk delete / Fit / smooth menu); Draw stops for the lasso and
    comes back after it

    python tools/test_cpr_separate_connect.py
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
from PyQt6.QtWidgets import QApplication, QMenu, QMessageBox, QWidget  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
warned = []
QMessageBox.warning = staticmethod(lambda *a, **k: warned.append(a))
QMessageBox.information = staticmethod(lambda *a, **k: warned.append(a))
from multi_dicomviewer.core.centerline import CenterLine  # noqa: E402
from multi_dicomviewer.ui.coronary_tree_window import CoronaryTreeWindow  # noqa: E402
from multi_dicomviewer.viewers.cpr_mixin import CPRMixin  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


# ---- dock: separate ----------------------------------------------------------
D = tempfile.mkdtemp(dir=HOME)
UID = "sc-ct"


def write_cpr(path, ctrl):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"format": "MDV-CPR", "version": 1, "type": "cpr",
                   "series": {"series_uid": UID}, "src_dir": "",
                   "ctrl": np.asarray(ctrl, float).tolist(),
                   "state": {"idx": 7}, "ctrl_manual": [[0, 0, 0]]}, f)


lad_ctrl = np.c_[np.linspace(0, 40, 9), np.zeros(9), np.zeros(9)]
d1_ctrl = np.array([[30.0, 0.0, 0.0], [30.0, 10.0, 0.0], [30.0, 20.0, 0.0]])
p_lad = os.path.join(D, "01.cpr.json")
p_d1 = os.path.join(D, "D1.cpr.json")
write_cpr(p_lad, lad_ctrl)
write_cpr(p_d1, d1_ctrl)
w = CoronaryTreeWindow(None)
w._use_uid(UID)
b = w._bundle()
tree = b["tree"]
pts = CenterLine.from_points(lad_ctrl, step_mm=0.5).points
tree.add_root("lad", "LAD", "LAD", pts)
tree.vessels["lad"].ctrl = lad_ctrl
dp = CenterLine.from_points(d1_ctrl, step_mm=0.5).points
tree.add_vessel("d1", "D1", "branch", dp, ctrl=d1_ctrl)
tree.set_parent("d1", "lad")
b["paths"].update({"lad": p_lad, "d1": p_d1})
n = tree.vessels["lad"].n
cut = int(np.argmin(np.linalg.norm(pts - [20.0, 0, 0], axis=1)))
new = w.split_vessel("lad", cut)
check("separate → a new line", bool(new) and new in tree.vessels, new)
nv = tree.vessels.get(new)
lad = tree.vessels["lad"]
check("distal part: next serial name (02) + its 02.cpr.json",
      nv is not None and nv.name == "02"
      and os.path.isfile(os.path.join(D, "02.cpr.json"))
      and b["paths"].get(new) == os.path.join(D, "02.cpr.json"))
check("proximal part keeps LAD (root), ends at the cut",
      lad.role == "LAD" and abs(lad.points[-1][0] - 20.0) < 0.6
      and abs(nv.points[0][0] - 20.0) < 0.6 and abs(nv.points[-1][0] - 40) < 0.6)
check("distal part hangs on the proximal end",
      nv.parent == "lad" and nv.junction == lad.n - 1)
check("the side branch beyond the cut moves to the distal part",
      tree.vessels["d1"].parent == new)
with open(p_lad, encoding="utf-8") as f:
    fl = json.load(f)
with open(os.path.join(D, "02.cpr.json"), encoding="utf-8") as f:
    fd = json.load(f)
check("files: proximal rewritten, distal written (no stale manual / idx)",
      abs(fl["ctrl"][-1][0] - 20.0) < 0.6 and abs(fd["ctrl"][0][0] - 20.0) < 0.6
      and "ctrl_manual" not in fl and fd["state"]["idx"] == 0
      and fd["format"] == "MDV-CPR")
check("separate at the very end is refused",
      w.split_vessel("lad", 0) == "" and len(tree.vessels) == 3)

# ---- dock: connect -----------------------------------------------------------
kept = w.connect_vessels(new, "first", "lad", "last")
check("connect → ONE line, LAD kept (it ends at the join)",
      kept == "lad" and new not in tree.vessels and len(tree.vessels) == 2)
lad = tree.vessels["lad"]
check("merged line runs 0 → 40 mm",
      abs(lad.points[0][0]) < 0.6 and abs(lad.points[-1][0] - 40) < 0.6)
check("its side branch re-attached to the merged line",
      tree.vessels["d1"].parent == "lad"
      and abs(lad.points[tree.vessels["d1"].junction][0] - 30) < 1.0)
check("merged-away file moved to _merged/",
      not os.path.isfile(os.path.join(D, "02.cpr.json"))
      and os.path.isfile(os.path.join(D, "_merged", "02.cpr.json")))
with open(p_lad, encoding="utf-8") as f:
    fl = json.load(f)
check("LAD file holds the whole line again",
      abs(fl["ctrl"][-1][0] - 40) < 0.6 and abs(fl["ctrl"][0][0]) < 0.6)

# reversed second line (both "last" ends meet) + too far
rc = np.array([[60.0, 0, 0], [50.0, 0, 0], [42.0, 0, 0]])
tree.add_vessel("rv", "RV", "branch",
                CenterLine.from_points(rc, step_mm=0.5).points, ctrl=rc)
warned.clear()
check("too far (> 5 mm) → refused with a message",
      w.connect_vessels("lad", "first", "rv", "last") == "" and warned)
kept = w.connect_vessels("lad", "last", "rv", "last")
check("drawn the other way → reversed and appended (0 → 60 mm)",
      kept == "lad" and abs(tree.vessels["lad"].points[-1][0] - 60) < 0.6
      and "rv" not in tree.vessels)

# ---- viewer: ends selection / connect / separate menu / lasso ---------------


class Shell:
    calls = []

    def coronary_connect_vessels(self, v, a, ea, b_, eb):
        self.calls.append(("connect", a, ea, b_, eb))
        return True

    def coronary_split_vessel(self, v, vid, idx):
        self.calls.append(("split", vid, idx))
        return True


class Host(QWidget, CPRMixin):
    """Pane B shows the x-y plane, 10 px per mm."""

    def __init__(self):
        QWidget.__init__(self)
        self._dims = (0.5, 0.5, 0.5)
        self._mode = "3D"
        self._meas_on = True
        self._meas_type = None
        self._draft = None
        self._cpr = None
        self._measures = {"A": [], "B": []}
        self._coronary_mode = True
        self._vr_on = {"A": False, "B": False}
        self._coro_overlay = [
            {"vid": "a", "points": [[0, 0, 0], [10, 0, 0], [20, 0, 0]]},
            {"vid": "b", "points": [[23, 0, 0], [30, 0, 0], [40, 0, 0]]}]
        self.sh = Shell()
        self.pane = {"A": None, "B": None}

    def _ct_mode_shell(self):
        return self.sh

    def _redraw_geom(self, key):
        pass

    def _redraw_meas(self, key):
        pass

    def _coronary_sync_ui(self):
        pass

    def _set_measure_type(self, key):
        self._cpr_lasso_begin(key)
        self._meas_type = key
        self._draft = None

    def _world3d_to_out(self, key, P):
        return (float(P[0]), float(P[1]))

    def _world_to_qt(self, key, wx, wy):
        return wx * 10.0, wy * 10.0 + 100.0

    def _disp_to_world(self, key, sx, sy):
        return sx / 10.0, (sy - 100.0) / 10.0

    def _coronary_pick_sample(self, which, sx, sy, tol=10.0):
        return "a", 1


h = Host()
check("ends selectable in CoronaryTree work", h._coro_end_sel_ok("B"))
h._cpr_band_start("B", 195, 100, ends=True)        # Shift+click on a.last
h._cpr_band_end()
check("Shift+click toggles the nearest end",
      h._coro_end_sel_get() == {("a", "last")})
h._cpr_band_start("B", 225, 90, ends=True)          # rectangle 22.5..24 mm
h._cpr_band_move(245, 110)
h._cpr_band_end()
check("Shift+drag selects the ends inside (replacing)",
      h._coro_end_sel_get() == {("b", "first")})
h._cpr_band_start("B", 200, 100, ends=True)         # add a.last by Shift+click
h._cpr_band_end()
check("two ends of two lines → connect possible",
      h._coro_connect_pair() == (("a", "last"), ("b", "first")))
check("selected ends drawn like selected edit points",
      len(h._cpr_sel_world_pts("B")) == 2)


class FakeCanvas:
    def mapToGlobal(self, q):
        return q


class FakePane:
    canvas = FakeCanvas()


h.pane = {"A": FakePane(), "B": FakePane()}
orig_exec = QMenu.exec
seen = []


def pick(*label_parts):
    """Pick the menu action whose text holds one of *label_parts* (case-
    insensitive; the Japanese key or its translation)."""
    def _exec(self, *a, **k):
        acts = [x for x in self.actions() if x.text()]
        seen.append([x.text() for x in acts])
        for x in acts:
            if any(p.lower() in x.text().lower() for p in label_parts):
                return x
        return None
    return _exec


QMenu.exec = pick("connect", "conectar", "连接")
h._coro_end_menu("B", 200, 100)
QMenu.exec = orig_exec
check("right-click ▸ connect → the shell joins the two lines",
      h.sh.calls[-1] == ("connect", "a", "last", "b", "first"), h.sh.calls)
check("selection cleared after connect", not h._coro_end_sel_get())

# separate from the tree-line menu
QMenu.exec = pick("separate", "separar", "分离")
h._cpr_tree_menu("B", 100, 100, None, "a")
QMenu.exec = orig_exec
check("tree line menu ▸ ここで separate → split at the clicked sample",
      h.sh.calls[-1] == ("split", "a", 1), (h.sh.calls, seen[-1]))

# Polygon lasso: Draw stops, the trace is kept, ends inside are selected
h._coronary_mpr_pending = True
h._meas_type = "polyline"
h._draft = {"type": "polyline", "pane": "B", "pts": [(1, 1)]}
h._set_measure_type("polygon")
check("Polygon while drawing: Draw stops (trace stashed)",
      h._meas_type == "polygon" and not h._coronary_mpr_pending
      and h._cpr_lasso_stash["draft"] is not None)
QMenu.exec = pick("選択解除", "deselect", "Deseleccionar", "取消选择")
poly = [(18.0, -3.0), (26.0, -3.0), (26.0, 3.0), (18.0, 3.0)]
done = h._cpr_lasso_commit("B", poly)
for _ in range(5):
    app.processEvents()
QMenu.exec = orig_exec
check("lasso consumed the polygon (no measurement)", done)
check("… its menu offered connect for the two ends inside",
      any("connect" in x.lower() or "conectar" in x.lower() or "连接" in x for x in seen[-1]), seen[-1])
check("after the lasso: Draw back with its unfinished trace",
      h._meas_type == "polyline" and h._coronary_mpr_pending
      and h._draft is not None and h._draft["pts"] == [(1, 1)])

# lasso on the open line's edit points → bulk-edit menu (delete)
ctrl = [np.array([x, 20.0, 15.0]) for x in np.linspace(5.0, 55.0, 11)]
h._measures["B"] = [{"id": 1, "type": "polyline", "_cpr_src": True,
                     "pts3d": [list(map(float, p)) for p in ctrl],
                     "pts": [(float(p[0]), float(p[1])) for p in ctrl]}]
cl = CenterLine.from_points(np.asarray(ctrl, float), step_mm=0.5)
fu, fv = cl.frames(ref_up=np.array([0.0, 0.0, 1.0]))
h._cpr = {"cl": cl, "u0": -fu, "v0": fv, "idx": 0, "T": np.eye(2),
          "rot": 0.0, "reversed": False, "half": 10.0, "src": "B",
          "src_mi": 0, "ref_up": np.array([0.0, 0.0, 1.0])}
h._coronary_mpr_pending = False
h._meas_type = None
h._set_measure_type("polygon")
deleted = []
h._cpr_sel_apply = lambda op: deleted.append((op, sorted(h._cpr_sel_get())))
QMenu.exec = lambda self, *a, **k: (seen.append(
    [x.text() for x in self.actions() if x.text()]) or self.actions()[0])
poly = [(18.0, 15.0), (42.0, 15.0), (42.0, 25.0), (18.0, 25.0)]
done = h._cpr_lasso_commit("B", poly)
for _ in range(5):
    app.processEvents()
QMenu.exec = orig_exec
check("Polygon around edit points → their bulk menu ▸ delete",
      done and deleted == [("delete", [3, 4, 5, 6, 7])], deleted)
check("… and the edit state is back (no draw type)", h._meas_type is None)

# backends wired (Windows + Mac)
for nm in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    src = (ROOT / "multi_dicomviewer/viewers" / nm).read_text(encoding="utf-8")
    check(f"{nm}: lasso / ends / Draw-restore hooks",
          "self._cpr_lasso_begin(key)" in src
          and "self._cpr_lasso_commit(d[\"pane\"], pts)" in src
          and "_cpr_band_start(" in src and "ends=True" in src
          and "_coro_end_menu(" in src
          and "if self._cpr_lasso_restore():" in src)
src = (ROOT / "multi_dicomviewer/ui/main_window.py").read_text(encoding="utf-8")
check("shell: split / connect entry points",
      "def coronary_split_vessel" in src and "def coronary_connect_vessels" in src)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
