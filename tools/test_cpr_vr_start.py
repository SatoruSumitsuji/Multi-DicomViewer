# -*- coding: utf-8 -*-
"""Headless checks for the CPR "VR始点" (pick the vessel's proximal start on
the VR) shared logic in cpr_mixin: the visible-surface ray march, the snap to
the lumen centre along the view ray, the heart-sphere crop, and how the pick
starts / re-starts the trace draft on the map pane B.

The HU line sampler is the real VTK-viewer implementation (borrowed onto a
tiny fake host); the ray itself is faked (the backends' _cpr_vr_ray).

Run: python tools/test_cpr_vr_start.py
"""
import os
import pathlib
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from PyQt6.QtWidgets import QApplication

from multi_dicomviewer.viewers.cpr_mixin import CPRMixin
from multi_dicomviewer.viewers.ct_viewer import CTViewer

app = QApplication.instance() or QApplication([])
fails = []


def chk(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), name, extra)
    if not cond:
        fails.append(name)


# Synthetic CT (0.5 mm iso): a 3 mm coronary along x at (y=30, z=30), a big
# bright chamber (r = 12 mm) centred at (60, 30, 30), soft tissue elsewhere.
S = 0.5
nz = ny = nx = 160
zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32) * S
vol = np.full((nz, ny, nx), 40, np.int16)
vessel = ((yy - 30.0) ** 2 + (zz - 30.0) ** 2 <= 1.5 ** 2) & (xx < 45.0)
vol[vessel] = 450
chamber = (xx - 60.0) ** 2 + (yy - 30.0) ** 2 + (zz - 30.0) ** 2 <= 12.0 ** 2
vol[chamber] = 450
del zz, yy, xx


class Host(CPRMixin):
    _hu_along = CTViewer._hu_along               # the app's trilinear sampler

    def __init__(self):
        self._vol = vol
        self._dims = (S, S, S)
        self._cpr = None
        self._cpr_exclude = []
        self._mode = "3D"
        self._vr_on = {"A": True, "B": False}
        self._pc = {"A": np.zeros(3), "B": np.array([40.0, 40.0, 40.0])}
        self._draft = None
        self._draft_redo = []
        self.refreshes = 0
        self.ray = None

    # backend hooks
    def _cpr_vr_ray(self, key, sx, sy, dvr=None):
        return self.ray

    def _cpr_recenter_map_pane(self, mapkey, P):
        self._pc[mapkey] = np.asarray(P, float).copy()

    def _world3d_to_out(self, key, P):        # B = axial plane through _pc
        d = np.asarray(P, float) - self._pc[key]
        return (float(d[0]), float(d[1]))

    def _refresh(self, **_k):
        self.refreshes += 1

    def _redraw_geom(self, key):
        pass

    def _cpr_vr_line_update(self):
        pass


h = Host()

# ---- first hit: a ray straight down onto the vessel hits its TOP surface
P0 = h._cpr_vr_first_hit([20.0, 30.0, 70.0], [0.0, 0.0, -1.0])
chk("first hit lands on the vessel's near surface",
    P0 is not None and 30.0 < P0[2] <= 31.6 and abs(P0[0] - 20) < 1e-6,
    f"{None if P0 is None else P0.round(2)}")
chk("first hit: nothing under the cursor → None",
    h._cpr_vr_first_hit([20.0, 60.0, 70.0], [0.0, 0.0, -1.0]) is None)
chk("first hit: a ray that misses the volume → None",
    h._cpr_vr_first_hit([-5.0, -5.0, 70.0], [0.0, 0.0, -1.0]) is None)
# The VR W/L shift slides the ramp: +400 HU makes the 450 HU vessel faint.
chk("first hit follows the VR W/L shift",
    h._cpr_vr_first_hit([20.0, 30.0, 70.0], [0.0, 0.0, -1.0], shift=400.0)
    is None)
# Crop: a mask that excludes the vessel → see through it.
m = np.ones(vol.shape, bool)
m[:, :, : int(30 / S)] = False
chk("first hit honours the VR crop mask",
    h._cpr_vr_first_hit([20.0, 30.0, 70.0], [0.0, 0.0, -1.0], mask=m) is None)

# ---- pick: surface hit snapped to the lumen CENTRE along the view ray
h.ray = (np.array([20.0, 30.0, 70.0]), np.array([0.0, 0.0, -2.0]), None, 0.0)
ok = h._cpr_vr_pick_start("A", 0, 0)
d = h._draft
chk("pick: start set", ok and d is not None and d["pane"] == "B"
    and d["type"] == "polyline" and len(d["pts3d"]) == 1)
P = np.asarray(d["pts3d"][0])
chk("pick: snapped to the vessel centre (z≈30)", abs(P[2] - 30.0) < 0.4
    and abs(P[1] - 30.0) < 1e-6, f"{P.round(2)}")
chk("pick: map pane B centred on the start", np.allclose(h._pc["B"], P))
chk("pick: 2-D draft point in step with the moved plane",
    np.allclose(d["pts"][0], (0.0, 0.0)))
# Oblique ray (45° in x-z) still lands in the lumen.
D = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
h.ray = (np.array([20.0, 30.0, 30.0]) - 30.0 * D, D, None, 0.0)
h._draft = None
h._cpr_vr_pick_start("A", 0, 0)
P = np.asarray(h._draft["pts3d"][0])
chk("pick: oblique view ray → still inside the lumen",
    np.hypot(P[1] - 30.0, P[2] - 30.0) <= 1.0, f"{P.round(2)}")

# A click on the vessel's FLANK (1.2 mm off its axis) → pulled onto the axis.
h.ray = (np.array([20.0, 31.2, 70.0]), np.array([0.0, 0.0, -1.0]), None, 0.0)
h._draft = None
h._cpr_vr_pick_start("A", 0, 0)
P = np.asarray(h._draft["pts3d"][0])
chk("pick on the flank → re-centred on the axis",
    np.hypot(P[1] - 30.0, P[2] - 30.0) <= 0.35 and abs(P[0] - 20.0) < 0.3,
    f"{P.round(2)}")

# Chamber: a chamber-sized bright run is NOT pulled to its centre (the snap
# rejects cavities) → the start stays on the surface the user clicked.
h.ray = (np.array([60.0, 30.0, 70.0]), np.array([0.0, 0.0, -1.0]), None, 0.0)
h._draft = None
h._cpr_vr_pick_start("A", 0, 0)
P = np.asarray(h._draft["pts3d"][0])
chk("pick on a chamber: stays at its surface (not dragged inside)",
    P[2] > 40.0, f"{P.round(2)}")
chk("pick on empty space → False, draft untouched",
    (lambda d0: (setattr(h, "ray", (np.array([20.0, 60.0, 70.0]),
                                    np.array([0.0, 0.0, -1.0]), None, 0.0)),
                 not h._cpr_vr_pick_start("A", 0, 0) and h._draft is d0)[1])(
        h._draft))

# ---- re-pick keeps the distal clicks, replaces only the start
h._draft = {"type": "polyline", "pane": "B",
            "pts": [(0, 0), (1, 1), (2, 2)],
            "pts3d": [np.array([1.0, 1, 1]), np.array([30.0, 30, 30]),
                      np.array([40.0, 30, 30])]}
h.ray = (np.array([20.0, 30.0, 70.0]), np.array([0.0, 0.0, -1.0]), None, 0.0)
h._cpr_vr_pick_start("A", 0, 0)
d = h._draft
chk("re-pick: same trace, 3 points kept", len(d["pts3d"]) == 3
    and np.allclose(d["pts3d"][2], [40, 30, 30]))
chk("re-pick: only the start replaced", abs(d["pts3d"][0][2] - 30.0) < 0.4
    and abs(d["pts3d"][0][0] - 20.0) < 1e-6)
chk("re-pick: 2-D re-derived for every point", len(d["pts"]) == 3
    and np.allclose(d["pts"][2], (40 - h._pc["B"][0], 30 - h._pc["B"][1])))

# ---- VR overlay points
h._cpr_vr_on = True
pts, start = h._cpr_vr_line_pts()
chk("overlay: the trace + start", len(pts) == 3 and np.allclose(start, pts[0]))
h._cpr_vr_on = False
chk("overlay: nothing when VR始点 is off", h._cpr_vr_line_pts() == ([], None))

# ---- pick gate
h._cpr_vr_on = True
chk("pick gate: VR pane A in VR始点", h._cpr_vr_pick_ok("A"))
chk("pick gate: not pane B", not h._cpr_vr_pick_ok("B"))
h._cpr = {"cl": None}
chk("pick gate: not once a CPR is built", not h._cpr_vr_pick_ok("A"))
h._cpr = None
h._cpr_vr_on = False

# ---- heart-sphere crop
h._cpr_vr_center = np.array([5.0, 5.0, 5.0])
cm = h._cpr_vr_crop_mask()
R = h._CPR_VR_CROP_MM
chk("crop: centre inside", cm[10, 10, 10])
chk("crop: just inside / outside the radius",
    cm[10, 10, int((5 + R - 1) / S)] and not cm[10, 10, int((5 + R + 1) / S)])
chk("crop: far corner outside", not cm[-1, -1, -1])
vox = int(cm.sum())
exp = 4.0 / 3.0 * np.pi * R ** 3 / S ** 3
chk("crop: clipped by the volume (≤ full sphere)", 0 < vox <= exp * 1.02,
    f"{vox} vs {exp:.0f}")

# ---- toggle / stop bookkeeping (backend calls faked)
calls = []
h._coronary_mpr_pending = False
h._coronary_draw = lambda: (calls.append("draw"),
                            setattr(h, "_coronary_mpr_pending", True))
h._vr_set = lambda k, on: (calls.append(("vr", k, on)), h._vr_on.__setitem__(k, on))
h._vr_on = {"A": False, "B": False}
h._cpr_vr_toggle()
chk("toggle on: arms Draw, VR on pane A", calls[:2] == ["draw", ("vr", "A", True)]
    and h._cpr_vr_active())
chk("toggle on: crop centred on the map pane centre",
    np.allclose(h._cpr_vr_center, h._pc["B"]))
h._cpr_vr_toggle()
chk("toggle off: VR off, state cleared", calls[-1] == ("vr", "A", False)
    and not h._cpr_vr_active() and h._cpr_vr_start is None)
h._cpr_vr_stop()
chk("stop when off is a no-op", calls[-1] == ("vr", "A", False)
    and len(calls) == 3)

print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
