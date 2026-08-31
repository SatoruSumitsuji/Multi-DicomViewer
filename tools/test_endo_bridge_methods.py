"""Headless comparison of the Auto-Endo papillary/trabecula bridging methods in
endo_contours_from_blood: "close" (2-D morphological closing), "polar" (angular
dip-bridging on the radial profile), and "hull" (per-level convex hull).

Scenario 1 — a blood cylinder (radius R) with a WALL-ATTACHED PAPILLARY: a
wedge of blood removed near the wall on one side (an inward notch). The endo
radius at the notch meridian should be RESTORED (~R) once the papillary is
bridged; "close" with a small disk leaves it dipped.

Scenario 2 — a genuinely CONCAVE (kidney-shaped) cavity: shows "hull" over-
includes the concavity more than "polar".
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core.lv_compact import (          # noqa: E402
    endo_contours_from_blood,
)


def _profile_radius_at(prof, theta_deg):
    """Mean radius across levels for the meridian nearest *theta_deg*."""
    if not prof or "error" in prof:
        return None
    key = min(prof.keys(), key=lambda k: min(abs(k - theta_deg),
                                             abs(k - theta_deg + 360),
                                             abs(k - theta_deg - 360)))
    arr = prof[key]
    return float(np.mean(arr[:, 1])) if len(arr) else 0.0


def _cavity_area_mm2(prof, n_mer):
    """Cross-sectional area (mm^2) from the per-meridian mean radii (polygon)."""
    if not prof or "error" in prof:
        return None
    rs = []
    for i in range(n_mer):
        th = 360.0 * i / n_mer
        r = _profile_radius_at(prof, th)
        rs.append(r if r else 0.0)
    # area of the star polygon r(θ): 1/2 Σ r_i r_{i+1} sin(Δθ)
    dth = 2 * math.pi / n_mer
    return 0.5 * math.sin(dth) * sum(
        rs[i] * rs[(i + 1) % n_mer] for i in range(n_mer))


def scenario_papillary():
    sx = sy = sz = 1.0
    nx = ny = 80
    nz = 60
    cx, cy = 40.0, 40.0
    apex_z, base_z = 10.0, 50.0
    R = 25.0
    NOTCH_DEG = 20.0          # half-width of the papillary wedge
    NOTCH_IN = 15.0           # blood kept inside this radius (notch depth → 15)

    xc = (np.arange(nx) * sx).reshape(1, 1, -1)
    yc = (np.arange(ny) * sy).reshape(1, -1, 1)
    zc = (np.arange(nz) * sz).reshape(-1, 1, 1)
    dxg = xc - cx
    dyg = yc - cy
    r2 = dxg ** 2 + dyg ** 2
    ang = np.arctan2(dyg, dxg)                       # radians, θ=0 along +x
    inz = (zc >= apex_z) & (zc <= base_z)

    blood = (r2 <= R ** 2) & inz
    # Papillary: remove blood in the wedge |θ|<NOTCH_DEG for r in (NOTCH_IN, R]
    wedge = (np.abs(ang) <= math.radians(NOTCH_DEG)) & (r2 > NOTCH_IN ** 2)
    blood = blood & ~np.broadcast_to(wedge, blood.shape)

    apex = (cx, cy, apex_z)
    axis_dir = (0.0, 0.0, 1.0)
    radial0 = (1.0, 0.0, 0.0)
    n_mer = 24
    common = dict(n_meridians=n_mer, along_apex=2.0,
                  along_base=(base_z - apex_z) - 2.0, sax_step_mm=2.0,
                  half_mm=40.0, grid_mm=0.8)

    print(f"--- Scenario 1: wall-attached papillary (R={R}, notch to "
          f"{NOTCH_IN} mm over ±{NOTCH_DEG}°) ---")
    print(f"{'method':8}  {'r@notch(0°)':>12}  {'r@wall(180°)':>13}"
          f"  {'area mm2':>9}")
    res = {}
    for method in ("close", "polar", "hull"):
        prof = endo_contours_from_blood(
            blood, (sx, sy, sz), apex, axis_dir, radial0,
            close_mm=5.0, method=method, bridge_deg=60.0, **common)
        r_notch = _profile_radius_at(prof, 0.0)
        r_wall = _profile_radius_at(prof, 180.0)
        area = _cavity_area_mm2(prof, n_mer)
        res[method] = (r_notch, r_wall, area)
        print(f"{method:8}  {r_notch:12.1f}  {r_wall:13.1f}  {area:9.0f}")

    # Assertions: polar & hull RESTORE the notch (~R); close leaves it dipped.
    assert res["close"][0] < 0.7 * R, \
        f"close should dip at the notch, got {res['close'][0]:.1f}"
    assert res["polar"][0] > 0.85 * R, \
        f"polar should bridge the notch, got {res['polar'][0]:.1f}"
    assert res["hull"][0] > 0.85 * R, \
        f"hull should bridge the notch, got {res['hull'][0]:.1f}"
    # The un-notched wall must be ~R for every method (no over-reach there).
    for m in ("close", "polar", "hull"):
        assert 0.9 * R <= res[m][1] <= 1.12 * R, \
            f"{m} wall radius off: {res[m][1]:.1f}"
    print("OK: polar & hull bridge the papillary; close dips.\n")


def scenario_concave():
    """A concave (bean) cavity: two overlapping disks removed on one side make a
    real concavity. Hull should OVER-include it more than polar."""
    sx = sy = sz = 1.0
    nx = ny = 90
    nz = 40
    cx, cy = 45.0, 45.0
    apex_z, base_z = 8.0, 32.0
    R = 26.0

    xc = (np.arange(nx) * sx).reshape(1, 1, -1)
    yc = (np.arange(ny) * sy).reshape(1, -1, 1)
    zc = (np.arange(nz) * sz).reshape(-1, 1, 1)
    r2 = (xc - cx) ** 2 + (yc - cy) ** 2
    inz = (zc >= apex_z) & (zc <= base_z)
    blood = (r2 <= R ** 2) & inz
    # Carve a big concavity from the +y side (a wide, deep bite, NOT a thin notch)
    bite = ((xc - cx) ** 2 + (yc - (cy + R)) ** 2) <= (R * 0.8) ** 2
    blood = blood & ~np.broadcast_to(bite, blood.shape)

    apex = (cx, cy, apex_z)
    n_mer = 24
    common = dict(n_meridians=n_mer, along_apex=2.0,
                  along_base=(base_z - apex_z) - 2.0, sax_step_mm=2.0,
                  half_mm=40.0, grid_mm=0.8)
    print("--- Scenario 2: genuine concavity (bean) — over-inclusion check ---")
    areas = {}
    for method in ("close", "polar", "hull"):
        prof = endo_contours_from_blood(
            blood, (sx, sy, sz), apex, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0),
            close_mm=5.0, method=method, bridge_deg=60.0, **common)
        areas[method] = _cavity_area_mm2(prof, n_mer)
        print(f"{method:8}  area mm2 = {areas[method]:9.0f}")
    # Hull fills the concavity → largest area; polar bridges less aggressively.
    assert areas["hull"] >= areas["polar"] >= areas["close"] - 1.0, \
        f"expected hull >= polar >= close, got {areas}"
    print("OK: hull over-includes the concavity most; polar is in between.\n")


if __name__ == "__main__":
    scenario_papillary()
    scenario_concave()
    print("PASS")
