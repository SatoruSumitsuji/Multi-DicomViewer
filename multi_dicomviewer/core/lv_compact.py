"""Compact-layer identification from the LV blood pool.

Separate the LV myocardium's COMPACT layer from the papillary muscles /
trabeculae using the blood-pool GEOMETRY — not HU, since compact myocardium and
papillary muscle share the same (enhanced-tissue) HU. On each short-axis (SAX)
slice ⟂ the long axis:

  * fully-enclosed papillary "islands" (surrounded by blood) are filled
    (``binary_fill_holes``);
  * wall-attached papillary "notches" (open concavities) are bridged by a
    morphological ``closing`` with a disk the size of the largest papillary
    indentation.

That yields the smooth ENDOCARDIAL ENVELOPE S — the compact-layer inner surface,
i.e. the clinical "compacted" Endo contour (papillaries INCLUDED in the cavity).
From it:

  * endo cavity (papillary-included)  = S               → the standard Endo/EF
  * blood-only cavity                 = the blood pool   → true contrast volume
  * papillary + trabeculae            = S − blood        → intra-cavity muscle
  * compact layer                     = (inside Epi) − S (computed by the caller)

Volumes are integrated over the SAX stack from the apex to the valve base cut.

Pure numpy + SciPy ndimage (morphology). Coordinates are volume mm: a world
point (x, y, z) maps to voxel index (x/sx, y/sy, z/sz) and the mask is indexed
``blood[z, y, x]`` (matching the CT viewers' ``_dims`` = (sx, sy, sz)).
"""

from __future__ import annotations

import math

import numpy as np


def _disk(radius_px: float) -> np.ndarray:
    """Boolean disk structuring element of the given pixel radius (>=1)."""
    r = int(max(1, round(radius_px)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _envelope_radius(env: np.ndarray, ctr: int, dx: float, dy: float,
                     grid_mm: float) -> float:
    """Radius (mm) of the *env* boolean SAX grid along the ray (dx, dy) from the
    centre pixel *ctr* — the OUTERMOST True along that direction (the envelope is
    filled, so that is its boundary). env is indexed [row=v, col=u]; dx is the u
    (column) step, dy the v (row) step. 0 if the ray hits nothing."""
    npix = env.shape[0]
    last = 0
    s = 1
    while True:
        j = int(round(ctr + s * dx))
        i = int(round(ctr + s * dy))
        if not (0 <= i < npix and 0 <= j < npix):
            break
        if env[i, j]:
            last = s
        s += 1
    return last * grid_mm


def endo_contours_from_blood(blood, spacing_xyz, apex_xyz, axis_dir, radial0,
                             n_meridians, along_apex, along_base,
                             sax_step_mm=1.0, close_mm=4.0, half_mm=60.0,
                             grid_mm=0.6):
    """Per-meridian (along, radius) profiles of the ENDOCARDIAL ENVELOPE derived
    from the blood pool — the compact-layer inner surface (papillary/trabeculae
    INCLUDED in the cavity), ready to load as an editable Endo border.

    For each short-axis level the blood is filled (enclosed papillary islands)
    and closed (wall-attached notches) into the smooth envelope; its radial
    extent is then measured along each of *n_meridians* directions (θ=0 along
    *radial0*, matching LVModel's meridian convention).

    Returns ``{theta_deg: np.ndarray[(along, radius), …]}`` for the meridian
    angles ``i*360/n_meridians`` (only those with ≥2 samples), or
    ``{'error': 'no_scipy'}`` if SciPy is unavailable.
    """
    try:
        from scipy import ndimage
    except Exception:                                   # noqa: BLE001
        return {"error": "no_scipy"}

    apex = np.asarray(apex_xyz, float)
    n = np.asarray(axis_dir, float)
    n = n / (np.linalg.norm(n) or 1.0)
    u = np.asarray(radial0, float)
    u = u - u.dot(n) * n
    u = u / (np.linalg.norm(u) or 1.0)
    v = np.cross(n, u)
    se = _disk(close_mm / grid_mm)
    thetas = [360.0 * i / n_meridians for i in range(int(n_meridians))]
    rays = [(th, math.cos(math.radians(th)), math.sin(math.radians(th)))
            for th in thetas]
    out: dict = {th: [] for th in thetas}
    slab = float(sax_step_mm)
    t = float(along_apex)
    while t <= float(along_base) + 1e-6:
        centre = apex + t * n
        g, _cell = _sample_on_plane(blood, spacing_xyz, centre, u, v,
                                    half_mm, grid_mm)
        if g.any():
            env = ndimage.binary_fill_holes(g)
            env = ndimage.binary_closing(env, structure=se)
            env = ndimage.binary_fill_holes(env)
            ctr = env.shape[0] // 2                      # (u,v)=(0,0) pixel
            for th, dx, dy in rays:
                r = _envelope_radius(env, ctr, dx, dy, grid_mm)
                if r > 0.0:
                    out[th].append((float(t), r))
        t += slab
    return {th: np.asarray(vals, float)
            for th, vals in out.items() if len(vals) >= 2}


def _sample_on_plane(blood, spacing_xyz, centre, u, v, half_mm, grid_mm):
    """Nearest-voxel boolean sample of *blood* on a SAX plane grid centred at
    *centre* with in-plane axes (u, v), extent +/-half_mm, pitch grid_mm.
    Returns (grid[n, n] bool, cell_area_mm2)."""
    sx, sy, sz = spacing_xyz
    nz, ny, nx = blood.shape
    n = max(2, int(2.0 * half_mm / grid_mm) + 1)
    ax = np.linspace(-half_mm, half_mm, n)
    aa, bb = np.meshgrid(ax, ax)                       # (n, n)
    px = centre[0] + aa * u[0] + bb * v[0]
    py = centre[1] + aa * u[1] + bb * v[1]
    pz = centre[2] + aa * u[2] + bb * v[2]
    ix = np.round(px / sx).astype(np.int64)
    iy = np.round(py / sy).astype(np.int64)
    iz = np.round(pz / sz).astype(np.int64)
    ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
          & (iz >= 0) & (iz < nz))
    g = np.zeros((n, n), bool)
    g[ok] = blood[iz[ok], iy[ok], ix[ok]]
    return g, float(grid_mm * grid_mm)


def compact_from_blood(blood, spacing_xyz, apex_xyz, axis_dir, radial0,
                       along_apex, along_base, sax_step_mm=1.0,
                       close_mm=4.0, half_mm=60.0, grid_mm=0.6):
    """Segment the endocardial envelope / papillary volume from the blood pool.

    blood       : (nz, ny, nx) bool blood-pool mask (volume index space).
    spacing_xyz : (sx, sy, sz) mm.
    apex_xyz    : long-axis apex (mm).  axis_dir: unit apex->base direction.
    radial0     : an in-plane reference direction (theta=0); orthonormalised.
    along_apex / along_base : along-axis range (mm from the apex) to integrate
                  — the apex end and the valve base cut.
    sax_step_mm : SAX slice thickness for the integral.
    close_mm    : disk radius (mm) that bridges wall-attached papillary notches.
    half_mm     : half the SAX sampling FOV (>= expected cavity radius).
    grid_mm     : SAX sampling pitch.

    Returns a dict with volumes (mL) — blood_ml, envelope_ml (= papillary-
    included Endo cavity), papillary_ml (= envelope - blood) — and per-slice
    (along, blood_area_mm2, envelope_area_mm2). {'error': 'no_scipy'} if SciPy
    is unavailable.
    """
    try:
        from scipy import ndimage
    except Exception:                                   # noqa: BLE001
        return {"error": "no_scipy"}

    apex = np.asarray(apex_xyz, float)
    n = np.asarray(axis_dir, float)
    n = n / (np.linalg.norm(n) or 1.0)
    u = np.asarray(radial0, float)
    u = u - u.dot(n) * n
    u = u / (np.linalg.norm(u) or 1.0)
    v = np.cross(n, u)
    se = _disk(close_mm / grid_mm)

    blood_ml = env_ml = pap_ml = 0.0
    levels = []
    slab = float(sax_step_mm)
    t = float(along_apex)
    while t <= float(along_base) + 1e-6:
        centre = apex + t * n
        g, cell = _sample_on_plane(blood, spacing_xyz, centre, u, v,
                                   half_mm, grid_mm)
        if g.any():
            filled = ndimage.binary_fill_holes(g)        # enclosed islands
            env = ndimage.binary_closing(filled, structure=se)  # notches
            env = ndimage.binary_fill_holes(env)         # any holes the close left
            b_area = float(g.sum()) * cell
            e_area = float(env.sum()) * cell
            blood_ml += b_area * slab / 1000.0
            env_ml += e_area * slab / 1000.0
            pap_ml += max(0.0, e_area - b_area) * slab / 1000.0
            levels.append((float(t), b_area, e_area))
        t += slab

    return {"blood_ml": blood_ml, "envelope_ml": env_ml,
            "papillary_ml": pap_ml, "levels": levels}
