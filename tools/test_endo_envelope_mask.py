"""Headless test: endo_envelope_mask builds a 3-D endocardial envelope that is
GUARANTEED to contain the blood pool (endo ⊇ blood) while bridging papillary /
trabecular indentations — the root fix for "LV-Blood pokes out past the Endo".

Scenario: a blood cylinder (radius R) with a WALL-ATTACHED papillary (a wedge of
blood removed near the wall) AND a free-floating papillary island (blood removed
in an interior blob). The envelope must (1) contain every blood voxel, and
(2) fill the papillary notch + island (volume > blood volume), for all methods.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core.lv_compact import endo_envelope_mask  # noqa: E402


def main():
    sx = sy = sz = 1.0
    nx = ny = 80
    nz = 60
    cx, cy = 40.0, 40.0
    apex_z, base_z = 10.0, 50.0
    R = 25.0

    xc = (np.arange(nx) * sx).reshape(1, 1, -1)
    yc = (np.arange(ny) * sy).reshape(1, -1, 1)
    zc = (np.arange(nz) * sz).reshape(-1, 1, 1)
    dxg, dyg = xc - cx, yc - cy
    r2 = dxg ** 2 + dyg ** 2
    ang = np.arctan2(dyg, dxg)
    inz = (zc >= apex_z) & (zc <= base_z)

    blood = (r2 <= R ** 2) & inz
    # wall-attached papillary wedge (blood removed near the wall)
    wedge = (np.abs(ang) <= math.radians(20.0)) & (r2 > 15.0 ** 2)
    blood = blood & ~np.broadcast_to(wedge, blood.shape)
    # free-floating papillary island (interior blob of removed blood)
    island = ((xc - (cx - 8.0)) ** 2 + (yc - cy) ** 2 <= 4.0 ** 2)
    blood = blood & ~np.broadcast_to(island & (r2 <= R ** 2), blood.shape)

    apex = (cx, cy, apex_z)
    axis_dir = (0.0, 0.0, 1.0)
    radial0 = (1.0, 0.0, 0.0)
    blood_vox = int(blood.sum())

    print(f"blood voxels = {blood_vox}")
    for method in ("close", "polar", "hull"):
        res = endo_envelope_mask(
            blood, (sx, sy, sz), apex, axis_dir, radial0,
            along_apex=2.0, along_base=(base_z - apex_z) - 2.0,
            sax_step_mm=1.0, close_mm=6.0, half_mm=40.0, grid_mm=0.8,
            method=method, bridge_deg=60.0)
        assert res is not None, f"{method}: no envelope"
        comp, bbox = res
        z0, z1, y0, y1, x0, x1 = bbox
        endo = np.zeros(blood.shape, bool)
        endo[z0:z1, y0:y1, x0:x1] = comp
        # (1) endo MUST contain every blood voxel.
        missed = int((blood & ~endo).sum())
        # (2) endo must FILL the papillary (notch + island) → strictly bigger.
        endo_vox = int(endo.sum())
        print(f"{method:6}  endo = {endo_vox:6d}  missed blood = {missed}"
              f"  fill = +{endo_vox - blood_vox}")
        assert missed == 0, f"{method}: {missed} blood voxels outside endo!"
        assert endo_vox > blood_vox * 1.02, \
            f"{method}: envelope didn't fill papillaries ({endo_vox} vs {blood_vox})"
    print("OK: endo ⊇ blood for all methods; papillaries filled.")
    print("PASS")


if __name__ == "__main__":
    main()
