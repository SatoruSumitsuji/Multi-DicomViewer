"""Old EpiLv.json → MV-AoV tent conversion (core.lv_epi_convert, 2026-11-16).

A synthetic old Epi (a half-ellipsoid traced on 8 planes about an OLD, tilted
axis, ending at a flat base) is converted onto a NEW axis (common apex →
MV-AoV point): every new plane is a tent polyline point → apex → point, the
walls stay apical of the point, the radii follow the old surface, and the
result round-trips as an EpiLv dict with "epi_tent".

    python tools/test_lv_epi_convert.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_dicomviewer.core import lv_epi_convert as cv  # noqa: E402
from multi_dicomviewer.core.lv_axis import LVAxis  # noqa: E402
from multi_dicomviewer.core.lv_measure import LVModel, tent_terminal_ok  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


# Ellipsoid LV: centre C, semi-axes a (long) and b (short); apex = C - a·z.
C = np.array([60.0, 60.0, 70.0])
A_LONG, B_SHORT = 45.0, 25.0
Z = np.array([0.0, 0.0, 1.0])


def radius_at(p_along_from_apex):
    """Ellipsoid radius at an along-z level measured from its apex."""
    u = (p_along_from_apex - A_LONG) / A_LONG
    return B_SHORT * np.sqrt(max(0.0, 1.0 - u * u))


apex_true = C - A_LONG * Z
# OLD axis: tilted 8° from z, through the true apex (as an old apex → MV line).
tilt = np.radians(8.0)
old_dir = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
old = LVModel(n_planes=8)
old.set_axis_from_frame(apex_true, old_dir, np.array([0.0, 1.0, 0.0]), which="epi")
old.set_apex_point("epi", apex_true)
oax = old.epi_axis
for phi in old.plane_angles():
    e_s = oax.meridian_dir(phi)
    pts = []
    for sgn in (1.0, -1.0):
        for lv in (np.linspace(40.0, 2.0, 12) if sgn > 0 else np.linspace(2.0, 40.0, 12)):
            # point on the ellipsoid along the ray from the old-axis point at lv
            o = apex_true + lv * old_dir
            d = sgn * e_s
            lo, hi = 0.0, 60.0
            for _ in range(40):          # bisection on the ellipsoid boundary
                mid = 0.5 * (lo + hi)
                q = o + mid * d - C
                inside = (q[0] ** 2 + q[1] ** 2) / B_SHORT ** 2 + q[2] ** 2 / A_LONG ** 2 < 1
                lo, hi = (mid, hi) if inside else (lo, mid)
            pts.append(o + lo * d)
        if sgn > 0:
            pts.append(apex_true)
    old.set_long_axis_contour(phi, np.array(pts), which="epi")
old_d = old.to_dict()
old_d["series"] = {"series_uid": "1.2.3"}
old_d["region"] = {"x": 1}
old_d["axis_def"] = "apex-mvcenter"

check("old file detected", cv.is_old_epi(old_d))
new_style = dict(old_d, epi_tent=True)
check("a tent file is not old", not cv.is_old_epi(new_style))
check("a non-Epi dict is not old", not cv.is_old_epi({"type": "valve"}))

# NEW axis: common apex = the true apex, MV-AoV point 38 mm up the true z axis.
point = apex_true + 38.0 * Z
new_dir = Z
out, err = cv.convert(old_d, apex_true, new_dir, np.array([1.0, 0.0, 0.0]),
                      point, 12)
check("converted", out is not None, err)
check("tent marker + new plane count", out["epi_tent"] is True
      and out["n_planes"] == 12 and len(out["epi_planes"]) == 12)
check("stale keys dropped / identity kept", "region" not in out
      and "axis_def" not in out and out["series"] == {"series_uid": "1.2.3"}
      and out["converted_from"]["axis_def"] == "apex-mvcenter")
m = LVModel.from_dict(out)
ax = m.epi_axis
check("new axis = apex → MV-AoV point",
      np.allclose(ax.apex, apex_true) and np.allclose(ax.axis, Z, atol=1e-9))
ok_ends = ok_tent = ok_rad = True
worst = 0.0
for phi, P in m.epi_planes.items():
    P = np.asarray(P)
    ok_ends &= np.allclose(P[0], point) and np.allclose(P[-1], point)
    inner = P[1:-1]
    ok_tent &= bool(tent_terminal_ok(ax, inner, point).all())
    for q in inner:
        lv = float((q - ax.apex) @ ax.axis)
        r = float(np.linalg.norm((q - ax.apex) - lv * ax.axis))
        if lv > 3.0:
            worst = max(worst, abs(r - radius_at(lv)))
ok_rad = worst < 1.5
check("both ends at the MV-AoV point", ok_ends)
check("walls stay apical of the point (tent rule)", ok_tent)
check("radii follow the old surface (< 1.5 mm)", ok_rad, f"worst {worst:.2f} mm")
m.build()
check("converted model builds a closed tent surface",
      m.epi is not None and m.epi.tent is True)

# Failure cases
bad, why = cv.convert(old_d, apex_true, Z, np.array([1.0, 0.0, 0.0]),
                      apex_true + 5.0 * Z, 8)
check("point too close to the apex → refused", bad is None and why, why)
empty = dict(old_d, epi_planes={})
bad, why = cv.convert(empty, apex_true, Z, np.array([1.0, 0.0, 0.0]), point, 8)
check("no Epi border → refused", bad is None and why, why)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
