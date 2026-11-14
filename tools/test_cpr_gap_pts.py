"""AutoTrace gap edit points: the edit points in a low-contrast stretch
(stenosis / occlusion / calcium?) are flagged, drawn as an orange-red ring on
the CPR map pane and around the short-axis dot in both viewers, saved in the
.cpr.json (gap_pts) and dropped once the point is moved.
"""
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_dicomviewer.core.cpr_autotrack import gap_ctrl_pts  # noqa: E402
from multi_dicomviewer.viewers.cpr_mixin import CPRMixin  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


path = np.stack([np.linspace(0, 50, 201), np.zeros(201), np.zeros(201)], 1)
ctrl = np.array([[x, 0, 0] for x in (0, 10, 20, 24, 30, 40, 50)], float)
gp = gap_ctrl_pts(path, ctrl, [(19.0, 25.0)])
check("edit points inside a gap are flagged",
      sorted(q[0] for q in gp) == [20.0, 24.0], str(gp))
gp2 = gap_ctrl_pts(path, ctrl, [(33.0, 36.0)])
check("a gap with no edit point inside flags the nearest one",
      [q[0] for q in gp2] == [30.0], str(gp2))
check("no gaps → nothing flagged", gap_ctrl_pts(path, ctrl, []) == [])


class M(CPRMixin):
    def __init__(self, m):
        self._m = m
        self._cpr = {"src": "B", "src_mi": 0, "idx": 0}
        self._measures = {"A": [], "B": [m]}

    def _cpr_source_measure(self):
        return self._m

    def _cpr_src_measure(self):
        return self._m

    def _cpr_at_ctrl(self):
        return self._k


m = {"pts3d": ctrl.tolist(), "pts": [(q[0], q[1]) for q in ctrl],
     "_cpr_auto": True, "_cpr_gap_pts": gp}
v = M(m)
check("gap indices on the line", CPRMixin._cpr_gap_idx(m) == [2, 3])
check("map pane: the gap points' positions",
      v._cpr_gap_world_pts("B") == [(20.0, 0.0), (24.0, 0.0)]
      and v._cpr_gap_world_pts("A") == [])
v._k = 3
check("short-axis on a gap point", v._cpr_at_gap())
v._k = 1
check("short-axis on an ordinary point", not v._cpr_at_gap())
meta = v._cpr_save_meta()
check("saved in the .cpr.json as gap_pts",
      len(meta.get("gap_pts", [])) == 2)
m["pts3d"][2] = [20.0, 1.2, 0.0]                # the user dragged it
check("a moved point is no longer a gap point",
      CPRMixin._cpr_gap_idx(m) == [3]
      and len(v._cpr_save_meta().get("gap_pts", [])) == 1)
m2 = {"pts3d": ctrl.tolist()}
v2 = M(m2)
v2._cpr_adopt_file_meta({"ctrl": ctrl.tolist(), "gap_pts": [[24, 0, 0]]})
check("a loaded .cpr.json's gap_pts are adopted",
      CPRMixin._cpr_gap_idx(m2) == [3])
check("a line without gaps saves no gap_pts",
      "gap_pts" not in M({"pts3d": ctrl.tolist()})._cpr_save_meta())

src_vtk = (ROOT / "multi_dicomviewer/viewers/ct_viewer.py").read_text(encoding="utf-8")
src_gfx = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(
    encoding="utf-8")
src_mix = (ROOT / "multi_dicomviewer/viewers/cpr_mixin.py").read_text(encoding="utf-8")
src_ctw = (ROOT / "multi_dicomviewer/ui/coronary_tree_window.py").read_text(
    encoding="utf-8")
for nm, src in (("ct_viewer.py", src_vtk), ("ct_viewer_pygfx.py", src_gfx)):
    check(f"{nm}: gap rings on the map pane and the short-axis",
          src.count("_cpr_gap_world_pts(key)") == 1
          and src.count("_cpr_at_gap()") == 1 and "_CPR_GAP_RGB" in src)
check("viewer AutoTrace stores the gap points",
      "m[\"_cpr_gap_pts\"] = gap_ctrl_pts(res.path_mm, ctrl, res.gaps)" in src_mix)
check("batch AutoTrace writes gap_pts into each .cpr.json",
      "gap_ctrl_pts(res.path_mm, ctrl, res.gaps)" in src_ctw
      and 'nd["gap_pts"]' in src_ctw)

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
