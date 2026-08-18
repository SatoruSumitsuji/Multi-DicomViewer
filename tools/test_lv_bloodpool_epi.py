"""Headless test for the epi-surface-bounded LV blood-pool volume.

Synthetic: an EPI surface (cylinder radius 20 mm) built from meridian contours;
inside it a contrast blood cylinder (radius 15, HU 400) with myocardium (HU 100)
in the 15–20 shell; valve planes cap the base; a disconnected contrast blob
(mock aorta) sits outside. The measured volume must match the analytic blood
cylinder below the planes and exclude the myocardium and the blob.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core.lv_axis import LVAxis            # noqa: E402
from multi_dicomviewer.core.lv_surface import LVSurface      # noqa: E402
from multi_dicomviewer.core.lv_bloodpool import (            # noqa: E402
    bloodpool_volume_epi)


def main():
    sx = sy = sz = 0.5
    nx, ny, nz = 200, 200, 240
    vol = np.full((nz, ny, nx), 100.0, np.float32)           # myocardium
    cx, cy = 50.0, 50.0
    apex_z, base_z = 20.0, 90.0

    zc = (np.arange(nz) * sz).reshape(-1, 1, 1)
    yc = (np.arange(ny) * sy).reshape(1, -1, 1)
    xc = (np.arange(nx) * sx).reshape(1, 1, -1)
    r2 = (xc - cx) ** 2 + (yc - cy) ** 2
    vol[(r2 <= 15.0 ** 2) & (zc >= apex_z)] = 400.0          # blood (r<15)
    vol[((xc - 15.0) ** 2 + (yc - 50.0) ** 2 <= 6.0 ** 2)    # disconnected aorta
        & (zc >= 10.0)] = 420.0

    # Epi surface: cylinder radius 20 along +z from the apex.
    ax = LVAxis.from_frame((cx, cy, apex_z), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    alongs = np.linspace(0.0, 70.0, 15)
    contours = {float(t): np.column_stack([alongs, np.full_like(alongs, 20.0)])
                for t in range(0, 360, 30)}
    epi = LVSurface.from_meridian_contours(ax, contours, level_step=2.0,
                                           n_theta=48)

    # contains: on-axis mid inside; far point outside.
    c = epi.contains(np.array([[cx, cy, 55.0], [cx + 40, cy, 55.0]]))
    assert c[0] and not c[1], c

    apex = (cx, cy, apex_z)
    seed = (cx, cy, (apex_z + base_z) / 2.0)
    planes = [((cx, cy, base_z), (0.0, 0.0, 1.0)),
              ((cx, cy, base_z), (0.0, 0.0, 1.0))]
    res = bloodpool_volume_epi(
        vol, (sx, sy, sz), epi._all_ring_points(), epi.contains, planes,
        apex, hu_lo=250.0, hu_hi=3000.0, seed_xyz=seed)
    assert res is not None and "volume_ml" in res, res

    analytic = np.pi * 15.0 ** 2 * (base_z - apex_z) / 1000.0
    got = res["volume_ml"]
    err = abs(got - analytic) / analytic
    print(f"analytic = {analytic:7.3f} mL   measured = {got:7.3f} mL  "
          f"({err*100:.2f}% , count={res['count']})")
    assert err < 0.10, f"off by {err*100:.1f}%"
    print("OK: epi-bounded blood volume matches (myocardium + aorta excluded)")

    # Over-high range -> seed_out(hu).
    r_hi = bloodpool_volume_epi(
        vol, (sx, sy, sz), epi._all_ring_points(), epi.contains, planes,
        apex, hu_lo=500.0, hu_hi=3000.0, seed_xyz=seed)
    assert r_hi.get("error") == "seed_out" and r_hi.get("reason") == "hu", r_hi
    print("OK: over-high range -> seed_out(hu)")
    print("PASS")


if __name__ == "__main__":
    main()
