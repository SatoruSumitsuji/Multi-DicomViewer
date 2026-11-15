"""CPR + Epi border (2026-11-16):
- auto-trace gap edit points (the hollow-ring points where the lumen trace
  could not follow — the most distal vessel added from the angiogram) go onto
  the nearest Epi surface point when this CT has an Epi; traced points and the
  user's own clicks are never moved; nothing moves without an Epi;
- CPR row → Epi shows the Epi border on the short axis only (2-D segments of
  the section); both viewers have the button and draw it."""
import math
import sys
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from multi_dicomviewer.core.lv_measure import LVModel  # noqa: E402
from multi_dicomviewer.viewers.cpr_mixin import CPRMixin  # noqa: E402

app = QApplication.instance() or QApplication([])
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


# A synthetic Epi: a prolate ellipsoid tent, axis along z (apex at 0).
R, L = 30.0, 90.0


def r_of(t):
    return R * math.sqrt(max(0.0, 1.0 - ((t - L / 2) / (L / 2)) ** 2))


model = LVModel(n_planes=8)
model.set_axis(np.array([1.0, 0, L]), np.array([-1.0, 0, L]), np.zeros(3))
model.epi_tent = True
for phi in model.plane_angles():
    ts = np.linspace(L, 0.0, 80)
    a = [model.axis.to_world(phi, r_of(t), t) for t in ts]
    b = [model.axis.to_world(phi + 180, r_of(t), t) for t in ts[::-1]]
    model.set_long_axis_contour(phi, np.asarray(a + b), which="epi")
model.build()
EPI = model.epi


class V(CPRMixin):
    def __init__(self, epi=None, epi_dict=None):
        self._lvv_epi_surf = epi
        self._lvv_epi_model_dict = epi_dict
        self._lv = None
        self._measures = {"A": [], "B": []}
        self._cpr = None
        self.refreshes = 0

    def _world3d_to_out(self, which, P):
        return (float(P[0]), float(P[1]))

    def _refresh(self, *a, **k):
        self.refreshes += 1

    def window(self):
        return None


def trace(gap_pts, clicks):
    pts = [[45.0, 0, 45.0], [40.0, 0, 50.0], [36.0, 0, 60.0], [33.0, 0, 70.0]]
    m = {"type": "polyline", "pts3d": [list(q) for q in pts],
         "_cpr_gap_pts": [list(q) for q in gap_pts],
         "_cpr_manual": [list(q) for q in clicks]}
    return m


# ---- gap points onto the Epi ---------------------------------------------------
v = V(EPI)
m = trace(gap_pts=[[40.0, 0, 50.0], [36.0, 0, 60.0], [45.0, 0, 45.0]],
          clicks=[[45.0, 0, 45.0], [33.0, 0, 70.0]])
v._measures["B"].append(m)
before = [list(q) for q in m["pts3d"]]
n = v._cpr_epi_snap_gaps(m, m["_cpr_manual"])
check("2 gap points moved (the gap click kept)", n == 2, str(n))
p3 = np.asarray(m["pts3d"])
for k in (1, 2):
    q = p3[k]
    check(f"point {k} on the Epi", abs(math.hypot(q[0], q[1]) - r_of(q[2])) < 1.0,
          str(np.round(q, 2)))
check("traced point 3 unchanged", m["pts3d"][3] == before[3])
check("hand click 0 unchanged", m["pts3d"][0] == before[0])
check("still marked as gap points", sorted(v._cpr_gap_idx(m)) == [0, 1, 2],
      str(v._cpr_gap_idx(m)))
check("pts follow pts3d", np.allclose(np.asarray(m["pts"])[:, :2], p3[:, :2]))

# No Epi: nothing moves.
v0 = V(None)
m0 = trace(gap_pts=[[40.0, 0, 50.0]], clicks=[])
b0 = [list(q) for q in m0["pts3d"]]
check("no Epi → nothing moves", v0._cpr_epi_snap_gaps(m0) == 0 and m0["pts3d"] == b0)
# No gap points (the lumen trace worked everywhere): nothing moves.
m1 = trace(gap_pts=[], clicks=[])
b1 = [list(q) for q in m1["pts3d"]]
check("no gaps → nothing moves", v._cpr_epi_snap_gaps(m1) == 0 and m1["pts3d"] == b1)

# A loaded EpiLv / FullLv dict (no surface yet) is built once and cached.
vd = V(None, epi_dict=model.to_dict())
s1 = vd._cpr_epi_surface()
check("Epi built from the loaded dict", s1 is not None)
check("cached", vd._cpr_epi_surface() is s1)


# ---- short-axis Epi line ---------------------------------------------------------
class Pts:
    def __init__(self):
        self.points = np.array([[0.0, 0, 45.0]])
        self.tangents = np.array([[0.0, 0, 1.0]])


v2 = V(EPI)
v2._cpr = {"idx": 0, "cl": Pts(), "u": [np.array([1.0, 0, 0])],
           "v": [np.array([0.0, 1, 0])]}
check("off → no segments", v2._cpr_epi_section_2d() == [])
v2._cpr_epi_show_toggle(True)
segs = v2._cpr_epi_section_2d()
rr = [math.hypot(*a) for a, _b in segs]
check("on → the Epi ring of this section", len(segs) >= 20
      and abs(float(np.mean(rr)) - r_of(45.0)) < 1.0,
      f"{len(segs)} segs, r {np.mean(rr) if rr else 0:.1f}")
check("toggle refreshes", v2.refreshes == 1)
v2._cpr_epi_show_toggle(False)
check("off again", v2._cpr_epi_section_2d() == [])

# Without an Epi the toggle stays off and tells the user (no real dialog).
v3 = V(None)
v3._cpr = v2._cpr
with mock.patch.object(QMessageBox, "information") as info:
    v3._cpr_epi_show_toggle(True)
check("no Epi → stays off + message", not v3._cpr_epi_show and info.called)

# ---- the CPR row button + drawing in both viewers (source checks) --------------------
for f in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    s = (ROOT / "multi_dicomviewer" / "viewers" / f).read_text(encoding="utf-8")
    check(f"{f}: Epi button on the CPR row",
          "self._cpr_epi_show_btn = FitButton(t(\"Epi\"))" in s
          and "toggled.connect(self._cpr_epi_show_toggle)" in s)
    check(f"{f}: draws the section", "_cpr_epi_section_2d()" in s)
mx = (ROOT / "multi_dicomviewer" / "viewers" / "cpr_mixin.py").read_text(encoding="utf-8")
check("auto store snaps gaps", "self._cpr_epi_snap_gaps(m, m.get(\"_cpr_manual\"))" in mx)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
