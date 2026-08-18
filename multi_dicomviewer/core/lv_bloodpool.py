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


def bloodpool_volume(vol, spacing_xyz, apex_xyz, base_xyz, r_max, planes,
                     hu_lo, hu_hi, seed_xyz, apex_margin_mm: float = 8.0,
                     pad_mm: float = 3.0):
    """Blood-pool volume (mL) of the LV cavity inside a "loose bag" envelope.

    The envelope is a finite region so a threshold/plane mistake can't let the
    cavity leak into the aorta/whole heart (no runaway growth, no OOM crash):
      * a CYLINDER of radius *r_max* around the apex→base axis (generous, so the
        mid-cavity bulge is never clipped — size it from the valve annuli),
      * capped basally by the two valve *planes* (apex side kept),
      * capped apically a little below the apex (apex_margin_mm).
    Within it the counted voxels are hu_lo <= HU <= hu_hi (blood range) and 3-D
    connected to *seed_xyz*.

    *vol*        : (nz, ny, nx) HU array, indexed vol[z, y, x].
    *spacing_xyz*: (sx, sy, sz) mm per voxel.
    *apex_xyz*   : (x, y, z) mm apex point (axis start).
    *base_xyz*   : (x, y, z) mm base point (axis end; e.g. midpoint of the two
                   valve-ellipse centres).
    *r_max*      : cylinder radius mm (valve annulus size × a bulge factor).
    *planes*     : iterable of (center_xyz, normal_xyz) valve planes.
    *hu_lo/hu_hi*: blood HU range. *seed_xyz*: connectivity seed (mm).

    Returns dict(volume_ml, count, voxel_ml, bbox) or None if the seed is not in
    the thresholded region inside the envelope.
    """
    vol = np.asarray(vol)
    sx, sy, sz = (float(s) for s in spacing_xyz)
    nz, ny, nx = vol.shape
    apex = np.asarray(apex_xyz, float)
    base = np.asarray(base_xyz, float)
    axis = base - apex
    L = float(np.linalg.norm(axis))
    if L < 1e-3:
        return None
    axis = axis / L
    r_max = float(r_max)

    # Work sub-box: the capped cylinder's extent = the apex/base span padded by
    # the radius on every side (a hard, finite bound — no growth).
    ext = r_max + float(pad_mm)
    lo_w = np.minimum(apex, base) - ext
    hi_w = np.maximum(apex, base) + ext
    lo = np.maximum(np.floor([lo_w[2] / sz, lo_w[1] / sy, lo_w[0] / sx]),
                    0).astype(int)
    hi = np.minimum(np.ceil([hi_w[2] / sz, hi_w[1] / sy, hi_w[0] / sx]),
                    [nz, ny, nx]).astype(int)
    if np.any(hi <= lo):
        return None
    # Hard ceiling so a mis-scaled radius / axis can never allocate a huge
    # sub-volume and OOM-crash the app (converted to a caller-visible error).
    box = int(np.prod(np.maximum(hi - lo, 0)))
    if box > 50_000_000:
        return {"error": "too_large", "voxels": box, "r_max": float(r_max),
                "L": float(L)}
    z0, y0, x0 = (int(v) for v in lo)
    z1, y1, x1 = (int(v) for v in hi)
    sub = vol[z0:z1, y0:y1, x0:x1]
    zc = (np.arange(z0, z1) * sz).reshape(-1, 1, 1)
    yc = (np.arange(y0, y1) * sy).reshape(1, -1, 1)
    xc = (np.arange(x0, x1) * sx).reshape(1, 1, -1)

    # Envelope: cylinder about the axis + apical/basal caps + valve planes.
    dx, dy, dz = xc - apex[0], yc - apex[1], zc - apex[2]
    along = dx * axis[0] + dy * axis[1] + dz * axis[2]
    perp2 = (dx * dx) + (dy * dy) + (dz * dz) - along * along
    mask = (sub >= float(hu_lo)) & (sub <= float(hu_hi))
    mask &= (perp2 <= r_max * r_max)
    mask &= (along >= -float(apex_margin_mm))
    mask &= (along <= L + float(apex_margin_mm))
    for (c, nrm) in planes:
        c = np.asarray(c, float)
        n = _oriented_normal(c, nrm, apex)
        d = (xc - c[0]) * n[0] + (yc - c[1]) * n[1] + (zc - c[2]) * n[2]
        mask &= (d >= 0.0)

    si = (int(round(seed_xyz[2] / sz)) - z0,
          int(round(seed_xyz[1] / sy)) - y0,
          int(round(seed_xyz[0] / sx)) - x0)
    if not (0 <= si[0] < sub.shape[0] and 0 <= si[1] < sub.shape[1]
            and 0 <= si[2] < sub.shape[2]):
        return {"error": "seed_out", "reason": "box",
                "msg": "seed is outside the work box"}
    if not mask[si]:
        # Diagnose WHICH envelope condition the seed fails, so the caller can
        # give an actionable message instead of a vague "no cavity".
        seed = np.asarray(seed_xyz, float)
        hu_seed = float(sub[si])
        dv = seed - apex
        along_s = float(dv @ axis)
        perp_s = float(np.sqrt(max(0.0, dv @ dv - along_s * along_s)))
        plane_bad = False
        for (c, nrm) in planes:
            c = np.asarray(c, float)
            n = _oriented_normal(c, nrm, apex)
            if float((seed - c) @ n) < 0:
                plane_bad = True
        if not (hu_lo <= hu_seed <= hu_hi):
            reason, msg = "hu", ("seed HU %.0f is outside the range %.0f-%.0f"
                                 % (hu_seed, hu_lo, hu_hi))
        elif perp_s > r_max:
            reason, msg = "radius", ("seed is %.0f mm off the axis, beyond the "
                                     "bag radius %.0f mm" % (perp_s, r_max))
        elif along_s < -apex_margin_mm or along_s > L + apex_margin_mm:
            reason, msg = "along", ("seed is beyond the apex/base extent "
                                    "(%.0f mm along a %.0f mm axis)"
                                    % (along_s, L))
        elif plane_bad:
            reason, msg = "plane", ("seed is on the far (non-apex) side of a "
                                    "valve plane — re-check AoV/MV placement")
        else:
            reason, msg = "other", "seed not in the region"
        return {"error": "seed_out", "reason": reason, "msg": msg,
                "hu": hu_seed, "perp": perp_s, "r_max": float(r_max),
                "along": along_s, "L": float(L)}
    comp = _seed_component(mask, si)
    count = int(comp.sum())
    voxel_ml = (sx * sy * sz) / 1000.0
    # The region can still hit the cylinder wall if r_max is too small for the
    # true cavity — flag it so the caller can suggest enlarging the bag.
    hit_wall = bool(((perp2 > (r_max - max(sx, sy, sz)) ** 2) & comp).any())
    return {"volume_ml": count * voxel_ml, "count": count,
            "voxel_ml": voxel_ml, "hit_wall": hit_wall,
            "bbox": (z0, z1, y0, y1, x0, x1)}


def bloodpool_volume_epi(vol, spacing_xyz, epi_ring_pts, epi_contains, planes,
                         apex_xyz, hu_lo, hu_hi, seed_xyz, pad_mm: float = 2.0):
    """LV blood-pool volume bounded by the EPICARDIAL surface (instead of the
    loose-bag cylinder): count voxels that are inside the epi surface, on the
    apex side of both valve *planes*, with hu_lo <= HU <= hu_hi, and 3-D
    connected to *seed_xyz*.

    *epi_ring_pts* : (M,3) world points on the epi surface (for the work box).
    *epi_contains* : callable pts(N,3) -> bool mask (points inside the surface).
    *planes*       : ((center, normal), …) valve planes (oriented to apex here).
    Returns dict(volume_ml, count, …) or an {"error": …} dict.
    """
    vol = np.asarray(vol)
    sx, sy, sz = (float(s) for s in spacing_xyz)
    nz, ny, nx = vol.shape
    apex = np.asarray(apex_xyz, float)
    verts = np.asarray(epi_ring_pts, float).reshape(-1, 3)
    lo_w = verts.min(0) - float(pad_mm)
    hi_w = verts.max(0) + float(pad_mm)
    lo = np.maximum(np.floor([lo_w[2] / sz, lo_w[1] / sy, lo_w[0] / sx]),
                    0).astype(int)
    hi = np.minimum(np.ceil([hi_w[2] / sz, hi_w[1] / sy, hi_w[0] / sx]),
                    [nz, ny, nx]).astype(int)
    if np.any(hi <= lo):
        return None
    if int(np.prod(hi - lo)) > 60_000_000:
        return {"error": "too_large", "voxels": int(np.prod(hi - lo))}
    z0, y0, x0 = (int(v) for v in lo)
    z1, y1, x1 = (int(v) for v in hi)
    sub = vol[z0:z1, y0:y1, x0:x1]

    # Orient valve normals toward the apex once.
    oplanes = [(np.asarray(c, float), _oriented_normal(c, nrm, apex))
               for (c, nrm) in planes]

    # Only the HU-in-range voxels need the (costly) inside-epi / plane tests.
    hu_mask = (sub >= float(hu_lo)) & (sub <= float(hu_hi))
    zz, yy, xx = np.where(hu_mask)
    mask = np.zeros(sub.shape, dtype=bool)
    if len(zz):
        world = np.column_stack([(x0 + xx) * sx, (y0 + yy) * sy, (z0 + zz) * sz])
        keep = np.asarray(epi_contains(world), bool)
        for (c, n) in oplanes:
            keep &= ((world - c) @ n) >= 0.0
        mask[zz[keep], yy[keep], xx[keep]] = True

    si = (int(round(seed_xyz[2] / sz)) - z0,
          int(round(seed_xyz[1] / sy)) - y0,
          int(round(seed_xyz[0] / sx)) - x0)
    if not (0 <= si[0] < sub.shape[0] and 0 <= si[1] < sub.shape[1]
            and 0 <= si[2] < sub.shape[2]):
        return {"error": "seed_out", "reason": "box",
                "msg": "seed is outside the epicardial box"}
    if not mask[si]:
        seed = np.asarray(seed_xyz, float)
        hu_seed = float(sub[si])
        in_epi = bool(np.asarray(epi_contains(seed.reshape(1, 3)), bool)[0])
        plane_bad = any(float((seed - c) @ n) < 0 for (c, n) in oplanes)
        if not (hu_lo <= hu_seed <= hu_hi):
            reason, msg = "hu", ("seed HU %.0f is outside the range %.0f-%.0f"
                                 % (hu_seed, hu_lo, hu_hi))
        elif not in_epi:
            reason, msg = "epi", ("seed is outside the epicardial surface — "
                                  "move the ROI inside it / re-trace Epi")
        elif plane_bad:
            reason, msg = "plane", ("seed is on the far (non-apex) side of a "
                                    "valve plane — re-check MV/AoV")
        else:
            reason, msg = "other", "seed not in the region"
        return {"error": "seed_out", "reason": reason, "msg": msg}
    comp = _seed_component(mask, si)
    count = int(comp.sum())
    voxel_ml = (sx * sy * sz) / 1000.0
    return {"volume_ml": count * voxel_ml, "count": count, "voxel_ml": voxel_ml,
            "bbox": (z0, z1, y0, y1, x0, x1), "comp": comp}
