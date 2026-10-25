"""HU-guided refinement of the LV epicardial (Epi) contour — shared core.

The user traces a ROUGH Epi border on each long-axis plane; this module snaps
that border to the real epicardial edge in the CT **where the HU gradient is
clear** (anterior / lateral walls: myocardium ~soft-tissue → pericardial fat /
lung, a big HU drop) and **leaves it on the manual spline where the HU
difference is small** (inferior wall against diaphragm / liver — the user's
"HU差が小さい条件は手動より").

That behaviour falls out of a single idea: for each point we look outward along
the short-axis radial direction, measure the strongest soft-tissue→outside HU
drop near the traced radius, and blend the detected edge with the manual radius
by a weight that is ~0 for a small drop and ~1 for a large, clean drop. No hard
on/off threshold → no jagged seams between the auto and manual stretches.

Pure numpy; operates on an LVAxis + the HU volume. No Qt, no VTK — headless
testable. Works for BOTH viewers (Windows VTK / Mac pygfx); they only pass
``self._vol`` ([z,y,x] HU) and ``self._dims`` (sx,sy,sz mm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multi_dicomviewer.core.lv_measure import _cr_densify_3d


@dataclass
class EpiSnapParams:
    """Tuning for the Epi HU-snap (all mm unless noted)."""
    search_in_mm: float = 9.0      # how far INWARD of the trace to look
    search_out_mm: float = 9.0     # how far OUTWARD of the trace to look
    step_mm: float = 0.5           # radial sampling step
    smooth_mm: float = 1.5         # 1-D smoothing of the HU profile before d/dr
    plaus_mm: float = 2.5          # window each side of a candidate edge for the drop
    r_min_mm: float = 5.0          # below this radius (apex region) keep the manual line
    max_move_mm: float = 9.0       # reject an edge farther than this from the trace
    myo_floor_hu: float = -20.0    # INNER side must be ≥ this (soft tissue = myocardium)
    outside_ceil_hu: float = 0.0   # OUTER side must be ≤ this (fat / lung). Liver (~50)
    #                                is above it → inferior wall stays manual (by design).
    drop_lo_hu: float = 40.0       # HU drop below this → weight 0 (手動より)
    drop_hi_hu: float = 140.0      # HU drop above this → weight 1 (follow the edge)
    prox_w: float = 6.0            # HU-per-mm penalty for edges away from the trace
    per_seg: int = 6               # densify: samples per traced segment
    outside_hu: float = -1000.0    # HU returned for samples outside the volume


def sample_hu(vol, dims, pts, outside=-1000.0):
    """Trilinear HU at world-mm *pts* ((N,3)); world = voxel·spacing, origin 0,
    array indexed ``vol[z, y, x]``. Points outside the volume → *outside*."""
    sx, sy, sz = (float(d) for d in dims)
    p = np.asarray(pts, float).reshape(-1, 3)
    fx, fy, fz = p[:, 0] / sx, p[:, 1] / sy, p[:, 2] / sz
    nz, ny, nx = vol.shape
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    z0 = np.floor(fz).astype(int)
    out = np.full(len(p), float(outside), float)
    ok = ((x0 >= 0) & (x0 < nx - 1) & (y0 >= 0) & (y0 < ny - 1)
          & (z0 >= 0) & (z0 < nz - 1))
    if not ok.any():
        return out
    xi, yi, zi = x0[ok], y0[ok], z0[ok]
    tx = (fx[ok] - xi)[:, None]
    ty = (fy[ok] - yi)[:, None]
    tz = (fz[ok] - zi)
    V = vol
    c000 = V[zi, yi, xi];       c100 = V[zi, yi, xi + 1]
    c010 = V[zi, yi + 1, xi];   c110 = V[zi, yi + 1, xi + 1]
    c001 = V[zi + 1, yi, xi];   c101 = V[zi + 1, yi, xi + 1]
    c011 = V[zi + 1, yi + 1, xi]; c111 = V[zi + 1, yi + 1, xi + 1]
    tx = tx[:, 0]; ty = ty[:, 0]
    c00 = c000 * (1 - tx) + c100 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    out[ok] = c0 * (1 - tz) + c1 * tz
    return out


def _smooth1d(a, k):
    if k <= 1:
        return a
    ker = np.ones(int(k), float) / float(int(k))
    return np.convolve(a, ker, mode="same")


def _smoothstep(x, lo, hi):
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = max(0.0, min(1.0, (x - lo) / (hi - lo)))
    return t * t * (3.0 - 2.0 * t)


def detect_epi_radius(vol, dims, axis, theta_deg, along, r0, prm):
    """Find the epicardial radius along the radial ray (short-axis plane at
    *along*, direction = meridian θ) near the traced radius *r0*.

    Returns ``(r_edge_mm, weight)`` with weight ∈ [0,1]: 0 = no clear edge (keep
    the manual radius), 1 = a clean, large soft-tissue→outside drop."""
    r_lo = max(prm.r_min_mm * 0.5, r0 - prm.search_in_mm)
    rs = np.arange(r_lo, r0 + prm.search_out_mm + 1e-6, prm.step_mm)
    if len(rs) < 5:
        return r0, 0.0
    rdir = np.asarray(axis.meridian_dir(theta_deg), float)
    base = np.asarray(axis.apex, float) + float(along) * np.asarray(axis.axis, float)
    pts = base[None, :] + rs[:, None] * rdir[None, :]
    hu = sample_hu(vol, dims, pts, outside=prm.outside_hu)
    hus = _smooth1d(hu, max(1, int(round(prm.smooth_mm / prm.step_mm))))
    d = np.gradient(hus)                     # <0 where HU drops going outward
    drop = -d                                # per-step drop magnitude
    win = max(2, int(round(prm.plaus_mm / prm.step_mm)))
    n = len(rs)
    # The EPICARDIAL edge is a soft-tissue(myocardium) → fat/lung transition:
    # INNER side ≥ myo_floor AND OUTER side ≤ outside_ceil. This excludes the
    # endocardial edge (blood→myocardium: outer is still soft tissue) and the
    # inferior wall against liver (outer ~50 HU > ceil → stays manual).
    best_i, best_sc, best_dm = -1, -np.inf, 0.0
    for i in range(1, n - 1):
        if abs(rs[i] - r0) > prm.max_move_mm:
            continue
        hu_in = float(np.median(hus[max(0, i - win):i + 1]))
        hu_out = float(np.median(hus[i:min(n, i + win + 1)]))
        if hu_in < prm.myo_floor_hu or hu_out > prm.outside_ceil_hu:
            continue
        dm = hu_in - hu_out
        if dm <= 0.0:
            continue
        # Locate by steepness (per-step drop), biased toward the traced radius.
        sc = float(drop[i]) - prm.prox_w * abs(rs[i] - r0)
        if sc > best_sc:
            best_sc, best_i, best_dm = sc, i, dm
    if best_i < 0:
        return r0, 0.0
    w = _smoothstep(best_dm, prm.drop_lo_hu, prm.drop_hi_hu)
    return float(rs[best_i]), float(w)


def refine_epi_plane(vol, dims, axis, pts3d, prm=None):
    """Refine one plane's traced Epi polyline (base→apex→base, (N,3) world mm).

    Densifies the manual trace (Catmull-Rom, the fallback where HU is weak), then
    for each dense point blends the detected epicardial radius with the manual
    radius by the detection weight. Returns ``(out_pts(M,3), n_refined,
    mean_weight)``."""
    prm = prm or EpiSnapParams()
    dense = _cr_densify_3d(np.asarray(pts3d, float), per_seg=prm.per_seg)
    out = np.asarray(dense, float).copy()
    n_ref = 0
    wsum = 0.0
    for i in range(len(out)):
        along, radius, theta = axis.project(out[i])
        if radius < prm.r_min_mm:
            continue                         # apex region: keep the manual line
        r_edge, w = detect_epi_radius(vol, dims, axis, theta, along, radius, prm)
        if w <= 0.0:
            continue
        r_final = w * r_edge + (1.0 - w) * radius
        out[i] = axis.to_world(theta, r_final, along)
        n_ref += 1
        wsum += w
    return out, n_ref, (wsum / n_ref if n_ref else 0.0)


def refine_epi_model(vol, dims, model, prm=None):
    """Refine EVERY traced Epi plane of *model* in place (rewrites
    ``epi_planes`` / ``epi_contours`` via ``set_long_axis_contour``). The caller
    then does ``model.build()`` + redisplay. Returns a stats dict."""
    prm = prm or EpiSnapParams()
    stats = {"planes": 0, "refined_pts": 0, "mean_weight": 0.0}
    axis = getattr(model, "epi_axis", None)
    if vol is None or axis is None or not getattr(model, "epi_planes", None):
        return stats
    wsum = 0.0
    for phi, pts in list(model.epi_planes.items()):
        if pts is None or len(pts) < 3:
            continue
        refined, n_ref, meanw = refine_epi_plane(vol, dims, axis, pts, prm)
        model.set_long_axis_contour(float(phi), refined, which="epi")
        stats["planes"] += 1
        stats["refined_pts"] += n_ref
        wsum += meanw
    if stats["planes"]:
        stats["mean_weight"] = wsum / stats["planes"]
    return stats
