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

import math

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


def _fill_theta(grid):
    """Fill 0 cells of a (level, θ) grid by CIRCULAR linear interpolation along θ
    (per level), so bins a coarse voxel circle missed take a sensible value."""
    out = grid.copy()
    n_lvl, n_theta = grid.shape
    xs = np.arange(n_theta)
    for i in range(n_lvl):
        row = out[i]
        nz = np.nonzero(row)[0]
        if 0 < len(nz) < n_theta:
            out[i] = np.interp(xs, nz, row[nz], period=n_theta)
    return out


def wall_thickness_radial_field(endo_mask, epi_mask, spacing_zyx, apex_xyz,
                                axis_dir, radial0, n_theta=120,
                                level_mm=1.0):
    """RADIAL (short-axis / echo-style) wall thickness: at each along-axis LEVEL
    and angle θ around the LV long axis, thickness = (outermost Epi radius) −
    (outermost Endo radius) from the axis — i.e. the wall measured on a
    short-axis slice, the way echocardiography measures it. Each myocardium
    voxel of that (level, θ) column gets that thickness.

    *endo_mask*, *epi_mask*: bool (z, y, x), same grid, endo ⊆ epi.
    *spacing_zyx*: (sz, sy, sx) mm. *apex_xyz*, *axis_dir*, *radial0*: the LV
    long axis (world mm; radial0 = θ=0, ⟂ axis). Returns (thick float32 mm,
    stats) with *thick* 0 outside the myocardium.
    """
    endo = np.asarray(endo_mask, bool)
    epi = np.asarray(epi_mask, bool)
    myo = epi & ~endo
    thick = np.zeros(epi.shape, np.float32)
    sz, sy, sx = (float(s) for s in spacing_zyx)
    apex = np.asarray(apex_xyz, float).ravel()
    ax = np.asarray(axis_dir, float).ravel()
    ax = ax / (np.linalg.norm(ax) or 1.0)
    r0 = np.asarray(radial0, float).ravel()
    r0 = r0 - ax * float(r0 @ ax)                 # make ⟂ axis
    r0 = r0 / (np.linalg.norm(r0) or 1.0)
    binm = np.cross(ax, r0)                        # θ=90° direction

    def _cyl(mask):
        """(level_idx, theta_idx, radius) for the True voxels of *mask*."""
        zz, yy, xx = np.nonzero(mask)
        P = np.column_stack([xx * sx, yy * sy, zz * sz]) - apex
        lvl = P @ ax
        perp = P - np.outer(lvl, ax)
        rad = np.linalg.norm(perp, axis=1)
        th = np.arctan2(perp @ binm, perp @ r0)   # −π..π
        return zz, yy, xx, lvl, th, rad

    ez, ey, ex, e_lvl, e_th, e_rad = _cyl(epi)
    if len(e_rad) == 0:
        return thick, {"min": 0.0, "mean": 0.0, "max": 0.0, "myo_ml": 0.0}
    lvl_lo = float(min(e_lvl.min(), 0.0))
    lvl_hi = float(e_lvl.max())
    n_lvl = max(1, int(math.ceil((lvl_hi - lvl_lo) / max(0.1, level_mm))))

    def _bin(lvl, th):
        li = np.clip(((lvl - lvl_lo) / max(1e-6, lvl_hi - lvl_lo)
                      * n_lvl).astype(int), 0, n_lvl - 1)
        ti = np.clip((((th + math.pi) / (2 * math.pi)) * n_theta).astype(int),
                     0, n_theta - 1)
        return li * n_theta + ti

    r_epi = np.zeros(n_lvl * n_theta, np.float64)
    np.maximum.at(r_epi, _bin(e_lvl, e_th), e_rad)
    r_endo = np.zeros(n_lvl * n_theta, np.float64)
    if endo.any():
        _z, _y, _x, o_lvl, o_th, o_rad = _cyl(endo)
        np.maximum.at(r_endo, _bin(o_lvl, o_th), o_rad)
    # Fill angular bins left empty by the voxel grid (a small-radius circle can't
    # populate every fine θ bin) so an empty endo bin doesn't read as full-wall.
    r_epi = _fill_theta(r_epi.reshape(n_lvl, n_theta))
    r_endo = _fill_theta(r_endo.reshape(n_lvl, n_theta))
    t_grid = np.maximum(0.0, r_epi - r_endo).ravel()   # radial thickness / cell

    mz, my, mx = np.nonzero(myo)
    if len(mz):
        Pm = np.column_stack([mx * sx, my * sy, mz * sz]) - apex
        m_lvl = Pm @ ax
        m_perp = Pm - np.outer(m_lvl, ax)
        m_th = np.arctan2(m_perp @ binm, m_perp @ r0)
        tvals = t_grid[_bin(m_lvl, m_th)].astype(np.float32)
        thick[mz, my, mx] = tvals

    vals = thick[myo]
    if vals.size == 0 or float(vals.max()) <= 0.0:
        return thick, {"min": 0.0, "mean": 0.0, "max": 0.0, "myo_ml": 0.0}
    voxel_ml = (sz * sy * sx) / 1000.0
    pos = vals[vals > 0]
    return thick, {
        "min": float(pos.min()),
        "mean": float(pos.mean()),
        "max": float(pos.max()),
        "myo_ml": float(int(myo.sum()) * voxel_ml),
    }
