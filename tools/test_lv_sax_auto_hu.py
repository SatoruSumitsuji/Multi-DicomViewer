"""LV mode (2026-11-16): (2) the trace guides (dotted lines ⟂ the long axis at
1/16 … 15/16) stay on the long-axis pane in SAX. (1) Entering SAX applies Epi HU＋スプライン補正
automatically (once — not after it has run, so hand corrections stay), with
no dialog; the short-axis spline keeps the MV-AoV tent (every wall ends AT
the MV-AoV point; the degenerate MV "plane" is never used as terminal plane).

    python tools/test_lv_sax_auto_hu.py
"""
import ast
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from multi_dicomviewer.core import lv_epi_autocontour as ea  # noqa: E402
from multi_dicomviewer.core.lv_measure import LVModel  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


# ---- core: the spline keeps the tent ------------------------------------------
R, H, L = 24.0, 60.0, 80.0


def r_of(t):
    if t <= H:
        return R * math.sqrt(max(0.0, 1.0 - ((H - t) / H) ** 2))
    return R * (L - t) / (L - H)


m = LVModel(n_planes=8)
m.set_axis(np.array([1.0, 0, L]), np.array([-1.0, 0, L]), np.zeros(3))
m.epi_tent = True
for phi in m.plane_angles():
    ts = np.linspace(L, 0.0, 60)
    a = [m.axis.to_world(phi, r_of(t) * (1.0 + 0.05 * math.sin(math.radians(phi))), t)
         for t in ts]
    b = [m.axis.to_world(phi + 180, r_of(t), t) for t in ts[::-1]]
    m.set_long_axis_contour(phi, np.asarray(a + b), which="epi")
P = np.array([0.0, 0.0, L])                        # the MV-AoV point (on the axis)
st = ea.refine_epi_short_axis(None, None, m, tent_point=P)
check("spline ran on every plane", st["planes"] == 8, str(st))
ends_ok = all(np.allclose(np.asarray(c)[0], P) and np.allclose(np.asarray(c)[-1], P)
              for c in m.epi_planes.values())
check("every wall ends AT the MV-AoV point", ends_ok)
below = all(float(np.asarray(c)[1:-1, 2].max()) < L - 0.5 for c in m.epi_planes.values())
check("no point at / past the point's level", below)
m.build()
check("tent still builds", m.epi is not None)

# Without tent_point the old behaviour (no forced terminal) is unchanged.
m2 = LVModel(n_planes=4)
m2.set_axis(np.array([1.0, 0, L]), np.array([-1.0, 0, L]), np.zeros(3))
for phi in m2.plane_angles():
    ts = np.linspace(L, 0.0, 60)
    a = [m2.axis.to_world(phi, r_of(min(t, H)) + 5, t) for t in ts]
    b = [m2.axis.to_world(phi + 180, r_of(min(t, H)) + 5, t) for t in ts[::-1]]
    m2.set_long_axis_contour(phi, np.asarray(a + b), which="epi")
ea.refine_epi_short_axis(None, None, m2)
check("no tent → ends not forced to a point",
      not any(np.allclose(np.asarray(c)[0], P) for c in m2.epi_planes.values()))

# ---- both viewers --------------------------------------------------------------------
for f in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    s = (ROOT / "multi_dicomviewer" / "viewers" / f).read_text(encoding="utf-8")
    tree = ast.parse(s)

    def seg(name):
        n = next(x for x in ast.walk(tree)
                 if isinstance(x, ast.FunctionDef) and x.name == name)
        return ast.get_source_segment(s, n)
    sax = seg("_lv_toggle_sax")
    hu = seg("_lv_hu_snap_epi")
    mvp = seg("_lv_mv_plane")
    check(f"{f}: SAX entry runs HU+Spline (auto)",
          "self._lv_hu_snap_epi(auto=True)" in sax
          and sax.index("self._lv_hu_snap_epi(auto=True)") < sax.index("lv[\"sax_which\"] = which"))
    check(f"{f}: auto = once (epi_hu_done skips) and no dialog",
          "if auto and (model is None or model.epi_hu_done" in hu
          and "if auto:                                 # SAX entry: no dialog" in hu)
    check(f"{f}: degenerate MV plane never a terminal plane",
          "if mv is None or self._lv_mvaov_on():" in mvp)
    check(f"{f}: spline gets the tent point (both calls)",
          s.count("tent_point=self._lv_tent_point()") == 2)

# ---- (2) the trace guides stay on the long-axis pane in SAX ---------------------------
vs = (ROOT / "multi_dicomviewer" / "viewers" / "ct_viewer.py").read_text(
    encoding="utf-8").replace("\r\n", "\n")
gs = (ROOT / "multi_dicomviewer" / "viewers" / "ct_viewer_pygfx.py").read_text(
    encoding="utf-8").replace("\r\n", "\n")
check("VTK: guides drawn in the SAX long-axis branch",
      "[line, self._circle_poly(rx, ry, cr)] + arrows))\n"
      "                # The trace guides" in vs
      and vs.count("self._lv_draw_axis_guides(key, p)") == 2)
check("Mac: guides painted in the SAX long-axis branch",
      gs.count("self._paint_lv_axis_guides(p, key, S)") == 2)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
