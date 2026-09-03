"""Myocardial WALL-THICKNESS field between the Endo and Epi surfaces.

Nearest-distance definition: the thickness at a myocardium voxel =
(distance to the Endo surface) + (distance to the Epi surface). For a
locally-parallel wall this equals the true wall thickness at *every* voxel of
the wall (both nearest points lie on the same perpendicular), so it makes a
smooth, correct on-section heat map. Both distances use an exact Euclidean
distance transform on the (anisotropic) voxel grid, so it stays fast (~1-2 s on
a heart-sized sub-box) rather than an O(N²) surface-to-surface search.
"""
from __future__ import annotations

import numpy as np


def wall_thickness_field(endo_mask, epi_mask, spacing_zyx):
    """*endo_mask*, *epi_mask*: boolean (z, y, x) on the SAME grid, endo ⊆ epi.
    *spacing_zyx*: (sz, sy, sx) mm.

    Returns (thick[z,y,x] float32 mm, stats). *thick* is 0 outside the
    myocardium (= epi AND NOT endo); inside it holds the local wall thickness in
    mm. *stats* = {min, mean, max, myo_ml} over the myocardium.
    """
    from scipy.ndimage import distance_transform_edt

    endo = np.asarray(endo_mask, bool)
    epi = np.asarray(epi_mask, bool)
    myo = epi & ~endo
    thick = np.zeros(epi.shape, np.float32)
    if not myo.any():
        return thick, {"min": 0.0, "mean": 0.0, "max": 0.0, "myo_ml": 0.0}
    sp = tuple(float(s) for s in spacing_zyx)
    # d_epi: inside epi, distance to the epi surface. d_endo: outside endo,
    # distance to the endo surface. Their sum = local wall thickness.
    d_epi = distance_transform_edt(epi, sampling=sp)
    d_endo = distance_transform_edt(~endo, sampling=sp)
    t = (d_epi + d_endo).astype(np.float32)
    thick[myo] = t[myo]
    vals = thick[myo]
    voxel_ml = (sp[0] * sp[1] * sp[2]) / 1000.0
    stats = {
        "min": float(vals.min()),
        "mean": float(vals.mean()),
        "max": float(vals.max()),
        "myo_ml": float(int(myo.sum()) * voxel_ml),
    }
    return thick, stats
