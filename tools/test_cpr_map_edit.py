# -*- coding: utf-8 -*-
"""Headless checks for the CPR map-pane edit-point tools in cpr_mixin:
click → show that point's section, Shift+drag rubber-band selection (and
Shift+click toggle), bulk Delete / Fit / Smooth as one undo step, and the
short-axis following an undo.

A tiny fake host stands in for the viewer: the map pane B shows the x-y plane
with screen = 10 px per mm, so screen rectangles are easy to reason about.

Run: python tools/test_cpr_map_edit.py
"""
import os
import pathlib
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from PyQt6.QtWidgets import QApplication

from multi_dicomviewer.viewers.cpr_mixin import CPRMixin
from multi_dicomviewer.core.centerline import CenterLine

app = QApplication.instance() or QApplication([])
fails = []


def chk(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), name, extra)
    if not cond:
        fails.append(name)


class _Sig:
    def __init__(self):
        self.got = []

    def emit(self, *a):
        self.got.append(a)


class Host(CPRMixin):
    def __init__(self, ctrl):
        self._dims = (0.5, 0.5, 0.5)
        self._vol = np.full((60, 60, 120), -100, np.int16)
        self._meas_on = True
        self._meas_type = None
        self._mode = "3D"
        self._measures = {"A": [], "B": [{
            "id": 1, "type": "polyline", "_cpr_src": True,
            "pts3d": [list(map(float, p)) for p in ctrl],
            "pts": [(float(p[0]), float(p[1])) for p in ctrl]}]}
        self.cpr_index_changed = _Sig()
        self.refreshes = 0
        self.undo = []
        cl = CenterLine.from_points(np.asarray(ctrl, float), step_mm=0.5)
        fu, fv = cl.frames(ref_up=np.array([0.0, 0.0, 1.0]))
        self._cpr = {"cl": cl, "u0": -fu, "v0": fv, "idx": 0,
                     "T": np.eye(2), "rot": 0.0, "reversed": False,
                     "half": 10.0, "src": "B", "src_mi": 0,
                     "ref_up": np.array([0.0, 0.0, 1.0])}
        self._cpr_apply_xform()

    # backend hooks
    def _refresh(self, **_k):
        self.refreshes += 1

    def _cpr_sync_bar(self):
        pass

    def _redraw_geom(self, key):
        pass

    def _redraw_meas(self, key):
        pass

    def _world3d_to_out(self, key, P):
        return (float(P[0]), float(P[1]))

    def _world_to_qt(self, key, wx, wy):
        return wx * 10.0, wy * 10.0

    def _disp_to_world(self, key, sx, sy):
        return sx / 10.0, sy / 10.0

    def _meas_pane_snap(self, pane):
        import copy
        return copy.deepcopy(self._measures[pane])

    def _meas_record(self, pane, before):
        self.undo.append((pane, before))

    def _meas_pane_restore(self, pane, snap):
        import copy
        self._measures[pane] = copy.deepcopy(snap)
        if self._cpr is not None and self._cpr.get("src") == pane:
            self._cpr_rebuild()


# 11 control points along x (y = 20 + a bump), z = 15 mm.
xs = np.linspace(5.0, 55.0, 11)
ctrl = [np.array([x, 20.0 + (3.0 if i == 5 else 0.0), 15.0])
        for i, x in enumerate(xs)]
h = Host(ctrl)

# ---- click → section
h._cpr_goto_ctrl(4)
idx4 = h._cpr_ctrl_indices()[4]
chk("click: section jumps to that edit point", h._cpr["idx"] == idx4
    and h._cpr_at_ctrl() == 4, f"idx {h._cpr['idx']} vs {idx4}")
chk("click: CoSync told", bool(h.cpr_index_changed.got))
chk("map edit ok in edit mode", h._cpr_map_edit_ok("B"))
chk("map edit off on the short-axis pane", not h._cpr_map_edit_ok("A"))
h._meas_type = "polyline"
chk("map edit off while a draw type is armed", not h._cpr_map_edit_ok("B"))
h._meas_type = None

# ---- rubber band: x 18..42 mm, y 0..40 mm → points 3,4,5,6,7 (x 20..40)
h._cpr_band_start("B", 180, 0)
h._cpr_band_move(420, 400)
chk("band: rectangle drawable", h._cpr_band_world_rect("B") is not None
    and len(h._cpr_band_world_rect("B")) == 5)
h._cpr_band_end()
chk("band: selects the points inside", h._cpr_sel_get() == {3, 4, 5, 6, 7},
    f"{sorted(h._cpr_sel_get())}")
chk("band: selected points drawable", len(h._cpr_sel_world_pts("B")) == 5)
# Shift+click toggles one off / on.
h._cpr_band_start("B", 400, 200)
h._cpr_band_end()
chk("shift+click: toggles a point off", h._cpr_sel_get() == {3, 4, 5, 6},
    f"{sorted(h._cpr_sel_get())}")
h._cpr_band_start("B", 400, 200)
h._cpr_band_end()
chk("shift+click: toggles it back on", h._cpr_sel_get() == {3, 4, 5, 6, 7})

# ---- smooth: the bump at #5 relaxes, ends/unselected fixed
before = [list(p) for p in h._measures["B"][0]["pts3d"]]
h._cpr_sel_apply("smooth")
after = h._measures["B"][0]["pts3d"]
chk("smooth: the bump is reduced", abs(after[5][1] - 20.0) < 1.5,
    f"y5 {before[5][1]:.2f} → {after[5][1]:.2f}")
chk("smooth: unselected points unchanged",
    all(np.allclose(after[k], before[k]) for k in (0, 1, 2, 8, 9, 10)))
chk("smooth: one undo step", len(h.undo) == 1)
chk("smooth: selection kept (repeatable)", h._cpr_sel_get() == {3, 4, 5, 6, 7})
chk("smooth: centreline rebuilt", abs(
    max(np.asarray(h._cpr["cl"].points)[:, 1]) - max(a[1] for a in after))
    < 0.3)

# ---- fit: no contrast in the fake volume → points stay put
h._cpr_sel_apply("fit")
chk("fit: no lumen nearby → unchanged",
    np.allclose(h._measures["B"][0]["pts3d"], after))

# ---- delete
n0 = len(h._measures["B"][0]["pts3d"])
h._cpr_sel_apply("delete")
m = h._measures["B"][0]
chk("delete: 5 points removed", len(m["pts3d"]) == n0 - 5
    and len(m["pts"]) == n0 - 5, f"{len(m['pts3d'])}")
chk("delete: selection cleared", h._cpr_sel_get() == set())
chk("delete: centreline rebuilt over the gap",
    abs(h._cpr["cl"].length_mm - 50.0) < 1.0, f"{h._cpr['cl'].length_mm:.1f}")
chk("delete: ctrl index cache matches", len(h._cpr_ctrl_indices()) == n0 - 5)

# ---- undo restores the points AND the short-axis
pane, snap = h.undo[-1]
h._meas_pane_restore(pane, snap)
chk("undo: points back", len(h._measures["B"][0]["pts3d"]) == n0)
chk("undo: short-axis follows", len(h._cpr_ctrl_indices()) == n0)

# ---- delete guard: cannot go below 2 points
h._cpr_sel_set(set(range(n0 - 1)))
from PyQt6.QtWidgets import QMessageBox
QMessageBox.information = staticmethod(lambda *a, **k: None)
h._cpr_sel_apply("delete")
chk("delete: keeps at least 2 points", len(h._measures["B"][0]["pts3d"]) == n0)

# ---- selection drops when the point count changes elsewhere
h._cpr_sel_set({1, 2})
h._measures["B"][0]["pts3d"].pop()
chk("selection: stale after count change → empty", h._cpr_sel_get() == set())

# ---- map drag end → rebuild + land on that point
h._measures["B"][0]["pts3d"].append([55.0, 20.0, 15.0])
h._measures["B"][0]["pts3d"][2] = [15.0, 24.0, 15.0]
h._cpr_map_point_moved(2)
chk("drag end: section on the moved point", h._cpr_at_ctrl() == 2)
P = np.asarray(h._cpr["cl"].points)[h._cpr["idx"]]
chk("drag end: centreline passes the moved point",
    np.linalg.norm(P - [15.0, 24.0, 15.0]) < 0.5, f"{P.round(2)}")

print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
