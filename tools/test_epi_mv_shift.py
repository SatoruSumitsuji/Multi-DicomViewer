"""Epi volume: the MV plane is moved toward the atrium so no drawn Epi point
is cut off (LVModel.epi_mv_plane). Endo keeps the plain MV plane.

Synthetic: an ellipsoidal Epi (apex at z=10, base around z=90) traced on 4
long-axis planes, its base drawn up to z=94 while the MV plane sits at z=90.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multi_dicomviewer.core.lv_measure import LVModel  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


C0 = np.array([60.0, 60.0, 0.0])
apex = C0 + [0, 0, 10.0]
m = LVModel(n_planes=4)
m.set_axis_from_frame(apex, np.array([0, 0, 1.0]), np.array([1.0, 0, 0]),
                      which="epi")
m.set_apex_point("epi", apex)
ax = m.epi_axis
A, R = 84.0, 28.0                       # half-length along z, equator radius
for phi in (0.0, 45.0, 90.0, 135.0):
    e = ax.meridian_dir(phi)
    pts = []
    for zz in np.linspace(94.0, 10.0, 40):          # wall +e: base → apex
        r = R * np.sqrt(max(0.0, 1 - ((zz - 10.0 - A) / A) ** 2))
        pts.append(C0 + [0, 0, zz] + r * e)
    for zz in np.linspace(10.0, 94.0, 40)[1:]:      # apex → base, wall −e
        r = R * np.sqrt(max(0.0, 1 - ((zz - 10.0 - A) / A) ** 2))
        pts.append(C0 + [0, 0, zz] - r * e)
    m.set_long_axis_contour(phi, np.array(pts), which="epi")
m.build()

mv = (C0 + [0, 0, 90.0], np.array([0, 0, 1.0]), 15.0)
mv2, d = m.epi_mv_plane(mv)
check("shift = protrusion (4 mm)", abs(d - 4.0) < 1e-6, f"d={d:.3f}")
check("moved toward the atrium (+z)",
      abs(float(mv2[0][2]) - 94.0) < 1e-6, f"c={np.round(mv2[0], 2)}")
check("normal / radius kept", np.allclose(mv2[1], mv[1]) and mv2[2] == 15.0)

# Flipped normal sign → same plane
mvf = (mv[0], -mv[1], 15.0)
mvf2, df = m.epi_mv_plane(mvf)
check("normal sign independent", abs(df - 4.0) < 1e-6
      and abs(float(mvf2[0][2]) - 94.0) < 1e-6)

# Nothing protrudes → unchanged
mv_hi = (C0 + [0, 0, 96.0], np.array([0, 0, 1.0]), 15.0)
mvh, dh = m.epi_mv_plane(mv_hi)
check("no protrusion → unchanged", dh == 0.0 and mvh is mv_hi)

# Volume: the Epi volume now reaches the drawn base (= the 94 mm cut)
dims, shape = (1.0, 1.0, 1.0), (110, 120, 120)
_c, _b, v_auto, _ = m.inside_mask_volumes(dims, shape, "epi", mv, None)
_c, _b, v_94, _ = m.inside_mask_volumes(dims, shape, "epi", mv2, None)
check("Epi volume uses the moved plane", abs(v_auto - v_94) < 1e-6,
      f"{v_auto:.1f} vs {v_94:.1f} mL")
surf = m._full_surface("epi")
_c, _b, v_90, _ = surf.inside_mask_volumes(dims, shape, mv, None, apex)
check("larger than the plain-MV cut", v_auto > v_90 + 1.0,
      f"{v_auto:.1f} > {v_90:.1f} mL")

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
