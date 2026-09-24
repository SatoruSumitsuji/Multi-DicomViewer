"""Headless tests for core/chamber_mask (CPR lumen-snap chamber exclusion).

    python tools/test_chamber_mask.py
"""
import sys

sys.path.insert(0, r"C:\CC_Product\Multi-DicomViewer")

import numpy as np  # noqa: E402

from multi_dicomviewer.core.chamber_mask import (   # noqa: E402
    chamber_component, point_excluded)


def _phantom():
    """A 60^3 HU volume (spacing 1mm): a bright LV 'chamber' blob, a separate
    bright 'coronary' tube divided from it by a low-HU 'myocardium' gap, and a
    separate bright 'aorta' blob far away."""
    v = np.full((60, 60, 60), -50.0)         # soft tissue background
    v[20:40, 20:40, 20:40] = 400.0           # LV chamber (bright)
    # coronary tube along x at z=30, y=48 (separated from chamber by y=40..47 gap)
    v[29:32, 47:50, 10:55] = 350.0
    v[10:20, 10:20, 10:20] = 500.0           # aorta blob (separate, far)
    return v


def test_chamber_flood_bounded():
    v = _phantom()
    sp = (1.0, 1.0, 1.0)
    comp, bbox = chamber_component(v, sp, seed_zyx=(30, 30, 30),
                                   hu_lo=200, hu_hi=1000, r_max_mm=60)
    assert comp is not None
    z0, z1, y0, y1, x0, x1 = bbox
    full = np.zeros(v.shape, bool)
    full[z0:z1, y0:y1, x0:x1] = comp
    # The chamber voxels are captured...
    assert full[30, 30, 30] and full[22, 22, 22], "chamber not filled"
    # ...but the flood does NOT leak across the myocardial gap into the coronary
    assert not full[30, 48, 30], "leaked into coronary tube"
    # ...nor into the far aorta blob
    assert not full[15, 15, 15], "leaked into aorta blob"
    # roughly the chamber's voxel count (20^3 = 8000)
    assert 6000 <= int(full.sum()) <= 10000, int(full.sum())
    print("OK chamber flood: fills the seeded chamber, no leak to coronary/aorta")


def test_seed_guards():
    v = _phantom()
    sp = (1.0, 1.0, 1.0)
    # seed in soft tissue (below hu_lo) → nothing
    comp, bbox = chamber_component(v, sp, (5, 5, 5))
    assert comp is None and bbox is None, "should reject a non-bright seed"
    # seed out of range → nothing
    comp, bbox = chamber_component(v, sp, (100, 100, 100))
    assert comp is None, "should reject an out-of-range seed"
    print("OK seed guards (non-bright / out-of-range → None)")


def test_point_excluded():
    v = _phantom()
    sp = (1.0, 1.0, 1.0)
    m = chamber_component(v, sp, (30, 30, 30))
    masks = [m]
    # world = index * spacing (x,y,z). Inside chamber (voxel 30,30,30):
    assert point_excluded((30, 30, 30), sp, masks) is True
    # coronary tube point (voxel z30,y48,x30) → world (30,48,30): NOT excluded
    assert point_excluded((30, 48, 30), sp, masks) is False
    # far background
    assert point_excluded((5, 5, 5), sp, masks) is False
    print("OK point_excluded (chamber in, coronary/background out)")


def main():
    test_chamber_flood_bounded()
    test_seed_guards()
    test_point_excluded()
    print("\nAll chamber-mask tests passed.")


if __name__ == "__main__":
    main()
