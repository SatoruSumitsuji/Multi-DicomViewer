"""Headless test: bloodpool_volume_epi auto-seed (seed_xyz=None) picks the
LARGEST connected in-range/inside-Epi component as the LV cavity — no manual ROI.

Synthetic: a blood cylinder (r=15 mm) along +z inside an Epi cylinder (r=22 mm),
plus a SEPARATE bright blob (a fake "aorta", r=6 mm) that is also inside the Epi
box and in the HU range but NOT connected to the cavity. The auto path must
return the big cavity volume, ignoring the blob.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core.lv_bloodpool import bloodpool_volume_epi  # noqa: E402


def main():
    sx = sy = sz = 0.5
    nx, ny, nz = 140, 140, 200
    cx, cy = 35.0, 35.0
    apex_z, base_z = 10.0, 85.0
    R = 15.0            # blood cavity radius
    Repi = 22.0         # epi radius
    HU_LO, HU_HI = 200.0, 500.0

    xc = (np.arange(nx) * sx).reshape(1, 1, -1)
    yc = (np.arange(ny) * sy).reshape(1, -1, 1)
    zc = (np.arange(nz) * sz).reshape(-1, 1, 1)
    r2 = (xc - cx) ** 2 + (yc - cy) ** 2
    inz = (zc >= apex_z) & (zc <= base_z)

    vol = np.zeros((nz, ny, nx), np.float32)     # ~0 HU background (out of range)
    vol[np.broadcast_to((r2 <= R ** 2) & inz, vol.shape)] = 350.0   # cavity blood

    # A disconnected bright blob well away from the cavity (fake aorta), same HU.
    bx, by = cx + 30.0, cy
    blob = ((xc - bx) ** 2 + (yc - by) ** 2 <= 6.0 ** 2) & inz
    vol[np.broadcast_to(blob, vol.shape)] = 400.0

    # Epi "surface": a cylinder radius Repi about (cx,cy), apex_z..base_z.
    def epi_contains(pts, extend_base=False):
        p = np.asarray(pts, float).reshape(-1, 3)
        rr = (p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2
        return (rr <= Repi ** 2) & (p[:, 2] >= apex_z) & (p[:, 2] <= base_z)

    # Ring points sampling the epi cylinder (for the work-box bounds).
    ths = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    zs = np.linspace(apex_z, base_z, 10)
    ring = np.array([[cx + Repi * np.cos(t), cy + Repi * np.sin(t), z]
                     for z in zs for t in ths], float)

    # Valve planes far basal/apical so they don't clip the cavity here.
    planes = [((cx, cy, base_z + 5.0), (0.0, 0.0, 1.0)),
              ((cx, cy, base_z + 5.0), (0.0, 0.0, 1.0))]
    apex = (cx, cy, apex_z)

    L = base_z - apex_z
    cyl_ml = np.pi * R ** 2 * L / 1000.0
    blob_ml = np.pi * 6.0 ** 2 * L / 1000.0

    res = bloodpool_volume_epi(vol, (sx, sy, sz), ring, epi_contains, planes,
                               apex, HU_LO, HU_HI, seed_xyz=None)
    assert res is not None and "error" not in res, res
    print(f"analytic cavity = {cyl_ml:7.2f} mL")
    print(f"blob (ignored)  = {blob_ml:7.2f} mL")
    print(f"auto-seed vol   = {res['volume_ml']:7.2f} mL")
    # Must be ~the cavity, NOT cavity+blob and NOT the blob.
    err = abs(res["volume_ml"] - cyl_ml) / cyl_ml
    assert err < 0.06, f"auto-seed off by {err*100:.1f}% (blob leaked in?)"
    assert res["volume_ml"] > blob_ml * 2, "auto-seed picked the small blob!"
    print("OK: auto-seed picks the largest cavity, ignores the disconnected blob")
    print("PASS")


if __name__ == "__main__":
    main()
