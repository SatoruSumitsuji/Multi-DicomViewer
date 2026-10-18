"""Virtual BiXA — pure compositing math for the bilateral-angiography
feature.

"Virtual BiXA" fuses an LCA cine and an RCA cine that were shot at the
(nearly) same C-arm projection into a single bilateral-contrast cine, as
if both coronary trees had been injected simultaneously. This module is
the Qt-free / DICOM-free core (same discipline as
:mod:`multi_dicomviewer.core.coaxial` and
:mod:`multi_dicomviewer.core.multisync`), so it is unit-testable in
isolation:

* **spatial registration** — a 2-landmark similarity transform
  (rotation + uniform scale + translation) mapping the moving image
  (RCA) onto the reference image (LCA).
* **fusion** — per-pixel ``min`` of the two display images. Angiographic
  contrast is dark on a bright background (MONOCHROME2), so the darkest
  of the two pixels keeps whichever vessel tree is present, leaving both
  trees black in the composite.
* **view-match metric** — the 3-D angular separation of the two C-arm
  beams (delegated to :mod:`coaxial`), so the UI can warn when the two
  acquisitions are not really the same projection.

Temporal alignment (LCA frame -> RCA frame) is NOT re-implemented here:
the feature reuses :func:`multi_dicomviewer.core.multisync.map_frame`
directly.
"""
from __future__ import annotations

import math

import numpy as np

from . import coaxial


# --------------------------------------------------------------- registration
def similarity_from_landmarks(src_pts, dst_pts):
    """Similarity transform (rotation + uniform scale + translation) that
    maps the two ``src_pts`` onto the two ``dst_pts``.

    Each point is an ``(x, y)`` pixel coordinate. With ``v_s = s1 - s0``
    and ``v_d = d1 - d0`` the transform is

        dst = scale * R(rot) @ (src - s0) + d0

    so it sends ``s0 -> d0`` and ``s1 -> d1`` exactly. Two points fully
    determine a similarity (4 DOF: scale, angle, tx, ty), which is the
    right model for two same-projection angiograms differing only by a
    small magnification (SID) and pan/tilt offset.

    Returns ``(scale, rot_rad, tx, ty)`` such that
    ``[x', y'] = scale * R @ [x, y] + [tx, ty]``.

    Raises ``ValueError`` if the two source points coincide (no scale /
    angle can be derived from a zero-length baseline).
    """
    (sx0, sy0), (sx1, sy1) = src_pts
    (dx0, dy0), (dx1, dy1) = dst_pts
    vsx, vsy = (sx1 - sx0), (sy1 - sy0)
    vdx, vdy = (dx1 - dx0), (dy1 - dy0)
    len_s = math.hypot(vsx, vsy)
    if len_s < 1e-9:
        raise ValueError("source landmarks coincide — baseline has no length")
    len_d = math.hypot(vdx, vdy)
    scale = len_d / len_s
    rot = math.atan2(vdy, vdx) - math.atan2(vsy, vsx)
    ca, sa = math.cos(rot), math.sin(rot)
    # tx,ty chosen so s0 maps exactly to d0:
    #   d0 = scale * R @ s0 + t  ->  t = d0 - scale * R @ s0
    tx = dx0 - scale * (ca * sx0 - sa * sy0)
    ty = dy0 - scale * (sa * sx0 + ca * sy0)
    return scale, rot, tx, ty


def apply_similarity(frame, scale, rot_rad, tx, ty,
                     out_shape=None, fill=255):
    """Warp ``frame`` by the forward similarity ``(scale, rot, tx, ty)``.

    The transform is the one returned by :func:`similarity_from_landmarks`
    (forward: source pixel -> destination pixel). We render the output by
    inverse-mapping every destination pixel back to a source pixel and
    sampling nearest-neighbour — the same approach as
    ``multisync_window._rotate_rgb``, generalised from pure rotation to a
    full similarity.

    ``out_shape`` defaults to the input's (H, W). Works for 2-D
    grayscale or (H, W, C) arrays. Destination pixels whose inverse maps
    outside the source are set to ``fill`` — default **255** (bright
    background) so a subsequent ``min`` fusion is not contaminated by
    black borders.
    """
    arr = np.asarray(frame)
    h_in, w_in = arr.shape[:2]
    if out_shape is None:
        h_out, w_out = h_in, w_in
    else:
        h_out, w_out = int(out_shape[0]), int(out_shape[1])

    ca, sa = math.cos(rot_rad), math.sin(rot_rad)
    # Forward:  d = scale * R @ s + t. Invert:
    #   s = (1/scale) * R^T @ (d - t)
    inv = 1.0 / scale if abs(scale) > 1e-12 else 0.0
    yy, xx = np.indices((h_out, w_out), dtype=np.float64)
    dx = xx - tx
    dy = yy - ty
    # R^T = [[ca, sa], [-sa, ca]]
    sx = inv * (ca * dx + sa * dy)
    sy = inv * (-sa * dx + ca * dy)
    sxi = np.rint(sx).astype(np.int64)
    syi = np.rint(sy).astype(np.int64)
    ok = (sxi >= 0) & (sxi < w_in) & (syi >= 0) & (syi < h_in)

    if arr.ndim == 2:
        out = np.full((h_out, w_out), fill, dtype=arr.dtype)
    else:
        out = np.full((h_out, w_out) + arr.shape[2:], fill, dtype=arr.dtype)
    cyi = np.clip(syi, 0, h_in - 1)
    cxi = np.clip(sxi, 0, w_in - 1)
    sampled = arr[cyi, cxi]
    out[ok] = sampled[ok]
    return out


# --------------------------------------------------------------------- fusion
def blend_min(a, b):
    """Per-pixel minimum of two display images (uint8 expected).

    Angiographic contrast is dark, so ``min`` keeps the darker (vessel)
    pixel from either image: both coronary trees survive into the
    composite. The two arrays must already share the same shape (the
    caller warps the moving image to the reference's geometry first).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(
            f"blend_min needs matching shapes, got {a.shape} vs {b.shape}"
        )
    return np.minimum(a, b)


# ------------------------------------------------------------ view-match metric
def view_separation_deg(view_a, view_b) -> float:
    """3-D angular separation (deg, [0, 90]) between two C-arm beams.

    Each view is ``(primary_deg, secondary_deg)`` = (LAO+/RAO-,
    CRA+/CAU-). 0 deg means the two acquisitions share the same
    projection line (ideal for BiXA); the UI grades the badge on this
    value. Thin wrapper over :func:`coaxial.beam_direction` +
    :func:`coaxial.angle_between_directions`.
    """
    (ba, aa) = view_a
    (bb, ab) = view_b
    da = coaxial.beam_direction(ba, aa)
    db = coaxial.beam_direction(bb, ab)
    return coaxial.angle_between_directions(da, db)
