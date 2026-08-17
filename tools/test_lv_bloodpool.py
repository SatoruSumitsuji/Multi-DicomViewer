"""Headless test for core.lv_bloodpool.bloodpool_volume.

Synthetic volume: an LV 'cavity' cylinder of contrast (HU 400) surrounded by
myocardium (HU 100), capped basally by two (slightly tilted) valve planes, plus
a SEPARATE disconnected contrast blob (mock descending aorta) that sits below
the planes.  The measured volume must match the analytic cylinder volume (below
the planes) and must EXCLUDE the disconnected blob.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core.lv_bloodpool import bloodpool_volume  # noqa: E402


def build():
    sx = sy = sz = 0.5                      # mm
    nx, ny, nz = 200, 200, 240
    vol = np.full((nz, ny, nx), 100.0, np.float32)      # myocardium background

    # World coords: (x,y,z) = index*spacing. Cylinder along z, centred in x,y.
    cx, cy = nx * sx / 2.0, ny * sy / 2.0               # 50, 50 mm
    R = 20.0                                             # cavity radius mm
    apex_z = 20.0                                        # apex world z (mm)
    base_z = 90.0                                        # valve level world z
    top_z = nz * sz                                      # 120 mm

    zc = (np.arange(nz) * sz).reshape(-1, 1, 1)
    yc = (np.arange(ny) * sy).reshape(1, -1, 1)
    xc = (np.arange(nx) * sx).reshape(1, 1, -1)
    r2 = (xc - cx) ** 2 + (yc - cy) ** 2

    # LV cavity: contrast cylinder from apex_z up to the top (extends ABOVE the
    # valve planes on purpose — the planes must clip it at base_z).
    cavity = (r2 <= R ** 2) & (zc >= apex_z)
    vol[cavity] = 400.0

    # Disconnected 'descending aorta': a separate contrast cylinder off to the
    # side, spanning low z (below the valve planes) — must be EXCLUDED.
    ax_, ay_ = 15.0, 50.0
    aorta = ((xc - ax_) ** 2 + (yc - ay_) ** 2 <= 8.0 ** 2) & (zc >= 10.0)
    vol[aorta] = 420.0

    apex = (cx, cy, apex_z)
    base = (cx, cy, base_z)                              # axis end (valve centres)
    seed = (cx, cy, (apex_z + base_z) / 2.0)            # mid-cavity
    r_max = R * 1.5                                      # generous bag radius

    # Two valve planes at z=base_z (flat, normal +z) so the analytic flat-cut
    # volume is exact. bloodpool_volume orients normals toward the apex
    # internally. (Tilt is exercised separately below.)
    planes = [((cx, cy, base_z), (0.0, 0.0, 1.0)),
              ((cx, cy, base_z), (0.0, 0.0, 1.0))]

    return vol, (sx, sy, sz), apex, base, r_max, planes, seed, dict(
        cx=cx, cy=cy, R=R, apex_z=apex_z, base_z=base_z)


def main():
    vol, spacing, apex, base, r_max, planes, seed, g = build()
    res = bloodpool_volume(vol, spacing, apex, base, r_max, planes,
                           thr=250.0, seed_xyz=seed)
    assert res is not None, "seed not in region"

    # Analytic: cylinder from apex_z to base_z (both planes ~ z=base_z through
    # the axis; the tilt averages out at the centre). Volume in mL.
    analytic_ml = np.pi * g["R"] ** 2 * (g["base_z"] - g["apex_z"]) / 1000.0
    got = res["volume_ml"]
    err = abs(got - analytic_ml) / analytic_ml
    print(f"analytic  = {analytic_ml:8.3f} mL")
    print(f"measured  = {got:8.3f} mL  (count={res['count']})")
    print(f"rel error = {err*100:5.2f} %")

    assert err < 0.08, f"volume off by {err*100:.1f}%"

    # The disconnected 'aorta' blob (~ pi*8^2*(base-10) below planes) would add a
    # large volume if connectivity failed — confirm it did NOT.
    aorta_ml = np.pi * 8.0 ** 2 * (g["base_z"] - 10.0) / 1000.0
    assert got < analytic_ml + 0.5 * aorta_ml, "descending-aorta blob leaked in"
    print("OK: disconnected aorta excluded")

    # Threshold too high (above blood HU) -> seed off pool -> None.
    assert bloodpool_volume(vol, spacing, apex, base, r_max, planes,
                            thr=500.0, seed_xyz=seed) is None
    print("OK: over-high threshold returns None")

    # A tilted valve plane must clip MORE (smaller volume) than the flat cut.
    n_tilt = np.array([0.25, 0.0, 1.0]); n_tilt /= np.linalg.norm(n_tilt)
    tilted = [((g["cx"], g["cy"], g["base_z"]), tuple(n_tilt)),
              ((g["cx"], g["cy"], g["base_z"]), (0.0, 0.0, 1.0))]
    res_t = bloodpool_volume(vol, spacing, apex, base, r_max, tilted,
                             thr=250.0, seed_xyz=seed)
    assert res_t is not None and res_t["volume_ml"] < got, "tilt did not clip"
    print(f"OK: tilted plane clips more ({res_t['volume_ml']:.1f} < {got:.1f} mL)")

    # A too-small bag radius must clip the cavity (smaller vol + hit_wall flag).
    res_s = bloodpool_volume(vol, spacing, apex, base, g["R"] * 0.5, planes,
                             thr=250.0, seed_xyz=seed)
    assert res_s is not None and res_s["volume_ml"] < got and res_s["hit_wall"]
    print(f"OK: small bag clips ({res_s['volume_ml']:.1f} mL, hit_wall=True)")
    print("PASS")


if __name__ == "__main__":
    main()
