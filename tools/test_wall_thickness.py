"""Headless test for core.lv_wallthickness (concentric-sphere shell)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core.lv_wallthickness import wall_thickness_field  # noqa: E402


def test_spherical_shell():
    # Two concentric spheres → known wall thickness = r_epi - r_endo everywhere.
    N = 120
    c = N / 2.0
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
    r = np.sqrt((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2)
    r_endo, r_epi = 25.0, 40.0          # voxels; spacing 1 mm → mm
    endo = r <= r_endo
    epi = r <= r_epi
    thick, stats = wall_thickness_field(endo, epi, (1.0, 1.0, 1.0))
    myo = epi & ~endo
    vals = thick[myo]
    true_t = r_epi - r_endo             # 15 mm
    # Interior of the shell (away from the two staircased boundaries) should be
    # very close to the true thickness; allow a small voxelization margin.
    med = float(np.median(vals))
    print(f"true={true_t}  median={med:.2f}  mean={stats['mean']:.2f}  "
          f"min={stats['min']:.2f} max={stats['max']:.2f}")
    assert abs(med - true_t) <= 1.5, med
    assert abs(stats["max"] - true_t) <= 2.5, stats["max"]
    # thickness is 0 outside the myocardium
    assert thick[~myo].max() == 0.0

    # anisotropic spacing: stretch z by 2 → thickness scales in mm, not voxels
    thick2, stats2 = wall_thickness_field(endo, epi, (2.0, 1.0, 1.0))
    assert stats2["max"] > stats["max"]        # z-direction walls now thicker mm
    print("aniso max=", round(stats2["max"], 2))

    # empty myocardium → zeros, zero stats
    z = np.zeros((10, 10, 10), bool)
    t0, s0 = wall_thickness_field(z, z, (1.0, 1.0, 1.0))
    assert t0.max() == 0.0 and s0["mean"] == 0.0
    print("WALL THICKNESS OK")


if __name__ == "__main__":
    test_spherical_shell()
