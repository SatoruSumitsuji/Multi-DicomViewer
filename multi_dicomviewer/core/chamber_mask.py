"""Heart-chamber blood-pool extraction for the CPR lumen-snap exclusion.

A CPR trace's "snap to the brightest lumen" can be pulled into a bright cardiac
chamber (LV / RV) when the coronary itself is dark (occluded / low-HU). Clicking
a seed INSIDE the chamber and flood-filling the connected blood pool (an HU
range) gives a mask the snap can ignore. Limiting to LV/RV is inherent: the user
seeds the chamber, and the surrounding myocardium (soft tissue, < hu_lo) plus
the closed diastolic aortic valve confine the connected component to that
chamber — it does not reach the epicardial coronaries or the aorta. A radius cap
bounds a runaway flood.

Pure numpy + scipy.ndimage (already a dependency); GUI-free and headless-tested.
Voxel/world convention matches the app: world_mm = voxel_index * spacing,
arrays indexed [z, y, x].
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:                       # pragma: no cover - fallback below
    _ndi = None


def chamber_component(vol_zyx, spacing_xyz, seed_zyx,
                      hu_lo: float = 200.0, hu_hi: float = 1000.0,
                      r_max_mm: float = 60.0):
    """Connected blood-pool component containing *seed_zyx* within [hu_lo, hu_hi],
    bounded to ±*r_max_mm* around the seed.

    *vol_zyx* is the HU volume [z,y,x]; *spacing_xyz* = (sx, sy, sz) mm;
    *seed_zyx* = (z, y, x) voxel index of the click. Returns ``(comp, bbox)``:
    ``comp`` is a bool sub-mask and ``bbox`` = (z0, z1, y0, y1, x0, x1) its offset
    into the full volume — so a point at full-volume voxel (z,y,x) is excluded iff
    ``bbox`` contains it and ``comp[z-z0, y-y0, x-x0]`` is True. Returns
    ``(None, None)`` if the seed is out of range or not in [hu_lo, hu_hi]."""
    vol = np.asarray(vol_zyx)
    if vol.ndim != 3:
        return None, None
    sx, sy, sz = (float(s) for s in spacing_xyz)
    Z, Y, X = vol.shape
    sz_i, sy_i, sx_i = (int(round(v)) for v in seed_zyx)
    if not (0 <= sz_i < Z and 0 <= sy_i < Y and 0 <= sx_i < X):
        return None, None
    if not (hu_lo <= float(vol[sz_i, sy_i, sx_i]) <= hu_hi):
        return None, None                       # seed not in a bright pool
    rz = int(r_max_mm / max(sz, 1e-6)) + 1
    ry = int(r_max_mm / max(sy, 1e-6)) + 1
    rx = int(r_max_mm / max(sx, 1e-6)) + 1
    z0, z1 = max(0, sz_i - rz), min(Z, sz_i + rz + 1)
    y0, y1 = max(0, sy_i - ry), min(Y, sy_i + ry + 1)
    x0, x1 = max(0, sx_i - rx), min(X, sx_i + rx + 1)
    sub = vol[z0:z1, y0:y1, x0:x1]
    inrange = (sub >= hu_lo) & (sub <= hu_hi)
    lz, ly, lx = sz_i - z0, sy_i - y0, sx_i - x0
    if _ndi is not None:
        lab, _n = _ndi.label(inrange)           # 6-connectivity by default
        sl = int(lab[lz, ly, lx])
        comp = (lab == sl) if sl > 0 else np.zeros_like(inrange)
    else:                                        # pragma: no cover
        comp = inrange                           # no CC available → whole range
    bbox = (z0, z1, y0, y1, x0, x1)
    return comp, bbox


def point_excluded(world_xyz, spacing_xyz, masks) -> bool:
    """True if world point *world_xyz* (mm) falls in any (comp, bbox) in *masks*.
    *masks* is an iterable of the tuples returned by :func:`chamber_component`."""
    sx, sy, sz = (float(s) for s in spacing_xyz)
    x, y, z = (float(c) for c in world_xyz)
    xi = int(round(x / max(sx, 1e-6)))
    yi = int(round(y / max(sy, 1e-6)))
    zi = int(round(z / max(sz, 1e-6)))
    for comp, bbox in masks:
        if comp is None:
            continue
        z0, z1, y0, y1, x0, x1 = bbox
        if z0 <= zi < z1 and y0 <= yi < y1 and x0 <= xi < x1:
            if comp[zi - z0, yi - y0, xi - x0]:
                return True
    return False
