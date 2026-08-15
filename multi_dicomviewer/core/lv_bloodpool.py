"""LV blood-pool volume for LVEF.

The LV cavity is bounded basally by the AORTIC- and MITRAL-valve planes and
converges toward the apex.  Within that region every voxel whose HU is at or
above a blood (contrast) threshold and that is 3-D connected to a seed in the
cavity is counted; volume = voxel count x voxel volume.  The two valve planes
cut off the aorta/LVOT and the left atrium; the connectivity filter drops
disconnected high-HU structures (e.g. the descending aorta).

Backend-independent: pure numpy, with SciPy used for fast labelling when it is
importable and a numpy flood-fill fallback otherwise.  Coordinates are in
patient/volume mm where a world point (x, y, z) maps to the voxel index
(x/sx, y/sy, z/sz) and the array is indexed vol[z, y, x] (matching the CT
viewers' ``_dims`` = (sx, sy, sz) and ``_trilinear_grid``).
"""

from __future__ import annotations

import numpy as np


def _seed_component(mask: np.ndarray, seed_ijk) -> np.ndarray:
    """Boolean array: the connected component of *mask* containing *seed_ijk*
    ((z, y, x) index).  Uses SciPy (26-connectivity) when available, else a
    numpy 6-connectivity flood-fill."""
    try:
        from scipy import ndimage                      # fast C labelling
        lbl, _n = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
        sl = int(lbl[seed_ijk])
        if sl == 0:
            return np.zeros_like(mask)
        return lbl == sl
    except Exception:                                   # noqa: BLE001
        return _flood(mask, seed_ijk)


def _flood(mask: np.ndarray, seed_ijk) -> np.ndarray:
    """6-connectivity flood-fill of *mask* from *seed_ijk* (numpy fallback)."""
    cur = np.zeros_like(mask)
    cur[seed_ijk] = mask[seed_ijk]
    if not cur.any():
        return cur
    while True:
        prev = int(cur.sum())
        nxt = cur.copy()
        nxt[1:, :, :] |= cur[:-1, :, :]
        nxt[:-1, :, :] |= cur[1:, :, :]
        nxt[:, 1:, :] |= cur[:, :-1, :]
        nxt[:, :-1, :] |= cur[:, 1:, :]
        nxt[:, :, 1:] |= cur[:, :, :-1]
        nxt[:, :, :-1] |= cur[:, :, 1:]
        nxt &= mask
        if int(nxt.sum()) == prev:
            return nxt
        cur = nxt


def _oriented_normal(center, normal, apex) -> np.ndarray:
    """Unit *normal* flipped so the apex lies on its POSITIVE side."""
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    if np.dot(np.asarray(apex, float) - np.asarray(center, float), n) < 0:
        n = -n
    return n


def bloodpool_volume(vol, spacing_xyz, apex_xyz, planes, thr, seed_xyz,
                     pad_mm: float = 15.0):
    """Blood-pool volume (mL) of the LV cavity.

    *vol*        : (nz, ny, nx) HU array, indexed vol[z, y, x].
    *spacing_xyz*: (sx, sy, sz) mm per voxel.
    *apex_xyz*   : (x, y, z) mm apex point.
    *planes*     : iterable of (center_xyz, normal_xyz) — the aortic & mitral
                   valve planes (normal orientation is fixed internally so the
                   apex side is kept).
    *thr*        : blood HU threshold (voxels with HU >= thr are blood).
    *seed_xyz*   : (x, y, z) mm seed inside the cavity (connectivity anchor).
    *pad_mm*     : margin added around {apex, seed, plane centres} for the work
                   sub-volume.

    Returns dict(volume_ml, count, voxel_ml, bbox) or None if the seed does not
    fall inside the thresholded region (e.g. threshold too high / off cavity).
    """
    vol = np.asarray(vol)
    sx, sy, sz = (float(s) for s in spacing_xyz)
    nz, ny, nx = vol.shape

    def to_ijk(p):                                       # (x,y,z) mm -> (z,y,x) idx
        return np.array([p[2] / sz, p[1] / sy, p[0] / sx], float)

    anchors = np.array(
        [to_ijk(p) for p in ([apex_xyz, seed_xyz] + [c for (c, _n) in planes])])
    lo_a, hi_a = anchors.min(0), anchors.max(0)
    spac = np.array([sz, sy, sx])

    # Adaptive bounding box: the LV cavity extends laterally well past the
    # on-axis anchors by an unknown radius, so grow the work sub-box until the
    # connected blood component no longer touches a (non-volume-edge) face.
    pad = float(pad_mm)
    comp = None
    z0 = y0 = x0 = 0
    while True:
        p = np.array([pad, pad, pad]) / spac
        lo = np.maximum(np.floor(lo_a - p).astype(int), 0)
        hi = np.minimum(np.ceil(hi_a + p).astype(int), [nz, ny, nx])
        if np.any(hi <= lo):
            return None
        z0, y0, x0 = (int(v) for v in lo)
        z1, y1, x1 = (int(v) for v in hi)
        sub = vol[z0:z1, y0:y1, x0:x1]
        zc = (np.arange(z0, z1) * sz).reshape(-1, 1, 1)
        yc = (np.arange(y0, y1) * sy).reshape(1, -1, 1)
        xc = (np.arange(x0, x1) * sx).reshape(1, 1, -1)
        mask = sub >= float(thr)
        for (c, nrm) in planes:
            c = np.asarray(c, float)
            n = _oriented_normal(c, nrm, apex_xyz)
            d = (xc - c[0]) * n[0] + (yc - c[1]) * n[1] + (zc - c[2]) * n[2]
            mask &= (d >= 0.0)
        si = (int(round(seed_xyz[2] / sz)) - z0,
              int(round(seed_xyz[1] / sy)) - y0,
              int(round(seed_xyz[0] / sx)) - x0)
        if not (0 <= si[0] < sub.shape[0] and 0 <= si[1] < sub.shape[1]
                and 0 <= si[2] < sub.shape[2]):
            return None
        if not mask[si]:
            return None                                 # seed off the blood pool
        comp = _seed_component(mask, si)
        # Grow if the component reaches a sub-box face that is not the volume
        # edge (i.e. it was clipped by the box, not by anatomy).
        touch = ((comp[0].any() and z0 > 0) or (comp[-1].any() and z1 < nz)
                 or (comp[:, 0].any() and y0 > 0) or (comp[:, -1].any() and y1 < ny)
                 or (comp[:, :, 0].any() and x0 > 0)
                 or (comp[:, :, -1].any() and x1 < nx))
        if touch and pad < 80.0:
            pad = min(80.0, pad * 1.8)
            continue
        break

    count = int(comp.sum())
    voxel_ml = (sx * sy * sz) / 1000.0
    return {"volume_ml": count * voxel_ml, "count": count,
            "voxel_ml": voxel_ml, "bbox": (z0, comp.shape[0] + z0,
                                           y0, comp.shape[1] + y0,
                                           x0, comp.shape[2] + x0)}
