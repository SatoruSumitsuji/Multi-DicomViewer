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

from multi_dicomviewer.core.lv_measure import _cr_densify_3d, mv_terminal_point


@dataclass
class EpiSnapParams:
    """Tuning for the Epi HU-snap (all mm unless noted)."""
    search_in_mm: float = 6.0      # how far INWARD of the trace to look
    search_out_mm: float = 6.0     # how far OUTWARD of the trace to look
    step_mm: float = 0.5           # radial sampling step
    smooth_mm: float = 1.5         # 1-D smoothing of the HU profile before d/dr
    plaus_mm: float = 2.5          # window each side of a candidate edge for the drop
    r_min_mm: float = 5.0          # below this radius (apex region) keep the manual line
    max_move_mm: float = 4.0       # reject an edge farther than this from the trace —
    #                                keeps HU補正 a LOCAL snap near the rough Epi line, so
    #                                it can't drift out to the pericardium / PV / lung
    #                                (combined with the innermost-edge choice below).
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
    def _edge_dm(i):
        hu_in = float(np.median(hus[max(0, i - win):i + 1]))
        hu_out = float(np.median(hus[i:min(n, i + win + 1)]))
        if hu_in < prm.myo_floor_hu or hu_out > prm.outside_ceil_hu:
            return None
        dm = hu_in - hu_out
        return dm if dm > 0.0 else None

    cands = []                                   # (i, dm) valid myo→outside edges
    for i in range(1, n - 1):
        if abs(rs[i] - r0) > prm.max_move_mm:
            continue
        dm = _edge_dm(i)
        if dm is not None:
            cands.append((i, dm))
    if not cands:
        return r0, 0.0
    # Pick the INNERMOST real myocardium→fat transition (drop ≥ drop_lo), NOT the
    # globally steepest one — the epicardium (myo→pericardial fat, ~100-150 HU)
    # sits INSIDE the far, much larger fat→lung / pleura drop (~600 HU), which
    # otherwise won the old steepness score and flung the border outward. The endo
    # edge (outer still soft tissue) was already excluded by _edge_dm.
    strong = [c for c in cands if c[1] >= prm.drop_lo_hu]
    i0 = min((strong or cands), key=lambda c: rs[c[0]])[0]
    # Localise within that innermost edge to its STEEPEST step so we land ON the
    # transition (the median-window first-crossing sits ~1 mm inside it).
    near = [c for c in cands if abs(c[0] - i0) <= win]
    i_e = max(near, key=lambda c: float(drop[c[0]]))[0]
    dm_e = _edge_dm(i_e) or 0.0
    w = _smoothstep(dm_e, prm.drop_lo_hu, prm.drop_hi_hu)
    return float(rs[i_e]), float(w)


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


# ---------------------------------------------------------------------------
# Short-axis correction (ellipse / spline) — regularise the Epi surface on a
# stack of short-axis rings between the base and an apical cap, then loft the
# result back into the meridian contours. An ADDITIONAL mode next to the
# per-meridian HU補正 above (not a replacement).
# ---------------------------------------------------------------------------

def _prior_profile(model, theta):
    """(along[], radius[]) of the current epi contour at meridian *theta* (sorted
    by along), or None when that meridian has no usable profile."""
    prof = model.epi_contours.get(theta % 360.0)
    if prof is None or len(prof) < 2:
        return None
    a = np.asarray(prof, float)[:, 0]
    r = np.asarray(prof, float)[:, 1]
    o = np.argsort(a)
    return a[o], r[o]


def _prior_radius(model, theta, along):
    """Traced epi radius at (meridian *theta*, *along*) by linear interpolation
    of the stored profile (clamped at the ends). None if no profile."""
    pr = _prior_profile(model, theta)
    if pr is None:
        return None
    a, r = pr
    return float(np.interp(float(along), a, r))


def _fill_nan_periodic(r):
    """Fill NaNs in a ring array by periodic linear interpolation."""
    r = np.asarray(r, float).copy()
    n = len(r)
    good = np.isfinite(r)
    if not good.any():
        return r
    idx = np.arange(n)
    xs, ys = idx[good], r[good]
    xs2 = np.concatenate([xs - n, xs, xs + n])
    ys2 = np.concatenate([ys, ys, ys])
    r[~good] = np.interp(idx[~good], xs2, ys2)
    return r


def _smooth_ring(theta_deg, r, harmonics):
    """Periodic circumferential low-pass of a UNIFORMLY-sampled ring r(θ): keep
    harmonics 0..*harmonics* (drops high-frequency noise, keeps the D-shape).
    Returns the smoothed radii at the same angles."""
    rr = np.asarray(r, float)
    n = len(rr)
    if n < 4:
        return rr
    F = np.fft.rfft(rr)
    if harmonics + 1 < len(F):
        F[harmonics + 1:] = 0.0
    return np.fft.irfft(F, n)


#: Short-axis correction levels as fractions apex(0)→base(1). The construction:
#: 4 equal divisions, then HALVE the apical-most and basal-most segments again,
#: plus one more at each end: between the last ring and the MV plane (fraction 1 —
#: the basal terminal the border converges to) and between the first ring and the
#: apex tip, so those corners aren't cut straight. The rings sit denser near the
#: apex and the base. 9 levels total; fraction 0 is the apex tip (radius 0), the
#: other 8 are the correction rings. This gives
#: a SPARSE contour (≈8 edit points per wall + apex) instead of a dense polyline.
_SAX_LEVEL_FRACS = (0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 0.875, 0.9375, 1.0)


def _edit_tol_mm(amax):
    """Along distance (mm) within which a control point counts as sitting ON a
    short-axis level (i.e. not a long-axis edit that needs its own ring)."""
    return max(1.5, 0.03 * float(amax))


def _merge_levels(vals, tol):
    """Sort + merge along values closer than *tol* (cluster mean)."""
    out = []
    for v in sorted(float(x) for x in vals):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(c)) for c in out]


def _edited_levels(model, amax, prev_levels):
    """Along positions of the long-axis control points that do NOT sit on a
    level of the previous spline run (*prev_levels*) — points the user added
    (add point) or slid along the axis since. Apex tip / above-base excluded."""
    tol = _edit_tol_mm(amax)
    prev = np.asarray(prev_levels, float)
    hits = []
    for prof in model.epi_contours.values():
        a = np.asarray(prof, float).reshape(-1, 2)[:, 0]
        for al in a[:-1]:                      # the last one is the MV terminal
            if al <= tol or al > amax - tol * 0.5:
                continue                       # apex tip / the base ring
            if len(prev) and np.min(np.abs(prev - al)) <= tol:
                continue                       # unmoved point on an old level
            hits.append(float(al))
    return _merge_levels(hits, tol)


def refine_epi_short_axis(vol, dims, model, prm=None, harmonics=4,
                          level_fracs=_SAX_LEVEL_FRACS, keep_edits=False,
                          mv_plane=None):
    """Short-axis SPLINE correction of the Epi surface — PURELY GEOMETRIC (no HU).

    The user traces/edits the Epi on the long axis (incl. the long-axis range
    handle); this harmonises the result CIRCUMFERENTIALLY. At short-axis rings at
    *level_fracs* of the apex→base length (default: the 9-level, apex-/base-dense
    scheme ``_SAX_LEVEL_FRACS``) it samples the CURRENT contour radius at each
    meridian (so every hand edit is INCLUDED as input, not re-detected) and
    low-passes the ring circumferentially (keep harmonics 0..*harmonics*) → a
    smooth closed short-axis shape. A manually moved point is thus blended with
    its neighbours (the user asked for this), not pinned. The rings become the
    contour's control points directly → the long-axis pane shows only ~8 edit
    points per wall, not a dense polyline. The apex tip stays as traced. No HU:
    accuracy comes from the manual trace + long-axis editing, which the HU
    detector could not match. *vol*/*dims* are unused (kept for signature
    parity). Returns a stats dict.

    *keep_edits* (the re-run after long-axis hand edits): keep the rings of the
    previous run (so one replaced by move_epi_section stays gone) and add a ring
    at the along position of any control point not on one of them, so those edits
    are spline INPUT rather than being resampled away. The levels used are
    remembered on the model (``epi_sax_levels``; ``epi_sax_extra`` = those off the
    default rings); without a previous run keep_edits falls back to the default
    rings.

    *mv_plane* ((c, n) of the MV plane): each wall's basal terminal is placed by
    mv_terminal_point instead of on the base ring."""
    stats = {"mode": "spline-geom", "levels": 0, "planes": 0, "extra_levels": 0}
    axis = getattr(model, "epi_axis", None)
    if axis is None or len(getattr(model, "epi_contours", {})) < 3:
        return stats
    thetas = model.meridian_angles()                  # uniform ring angles
    nth = len(thetas)
    amaxs = [float(_prior_profile(model, th)[0][-1])
             for th in thetas if _prior_profile(model, th) is not None]
    if not amaxs:
        return stats
    amax = float(np.median(amaxs))
    ring_fracs = [float(f) for f in level_fracs if f > 1e-6]   # non-apex rings
    levels = amax * np.asarray(ring_fracs)
    extra = []
    prev = getattr(model, "epi_sax_levels", None)
    if keep_edits and prev is not None:
        tol = _edit_tol_mm(amax)
        default = levels
        kept = [float(x) for x in sorted(prev)[:-1] if x < amax - tol * 0.5]
        edits = [e for e in _edited_levels(model, amax, prev)
                 if tol < e <= amax - tol * 0.5
                 and (not kept or np.min(np.abs(np.asarray(kept) - e)) > tol)]
        inner = sorted(kept + edits)
        extra = [x for x in inner if np.min(np.abs(default - x)) > tol]
        levels = np.asarray(inner + [amax], float)
    ring = np.full((len(levels), nth), np.nan)

    def _col(th):
        return int(np.argmin([abs(((t - th + 180.0) % 360.0) - 180.0)
                              for t in thetas]))

    for li, al in enumerate(levels):
        r = np.array([(_prior_radius(model, th, al)
                       if _prior_radius(model, th, al) is not None else np.nan)
                      for th in thetas], float)
        r = _fill_nan_periodic(r)
        if not np.isfinite(r).all():
            continue
        ring[li] = _smooth_ring(np.asarray(thetas), r, harmonics)
        stats["levels"] += 1

    if stats["levels"] < 1:
        return stats
    apex_pt = (np.asarray(model.epi_apex, float)
               if getattr(model, "epi_apex", None) is not None
               else np.asarray(axis.apex, float))
    for phi in model.plane_angles():
        opp = (phi + 180.0) % 360.0
        cphi, copp = _col(phi), _col(opp)

        def _wall(th, ci):
            w = [axis.to_world(th, max(0.0, float(ring[li, ci])), levels[li])
                 for li in range(len(levels)) if np.isfinite(ring[li, ci])]
            if mv_plane is not None and len(w) >= 2:
                end = mv_terminal_point(axis, th, mv_plane[0], mv_plane[1], w[-2])
                a_end = float((end - axis.apex) @ axis.axis)
                w = [P for P in w[:-1]
                     if float((P - axis.apex) @ axis.axis) < a_end - 0.5] + [end]
            return w

        wall_phi, wall_opp = _wall(phi, cphi), _wall(opp, copp)
        if len(wall_phi) < 2 or len(wall_opp) < 2:
            continue
        pts3d = (list(reversed(wall_phi))            # base → near-apex (φ wall)
                 + [apex_pt]                         # apex tip (shared)
                 + wall_opp)                         # near-apex → base (φ+180 wall)
        model.set_long_axis_contour(float(phi), np.asarray(pts3d, float),
                                    which="epi")
        stats["planes"] += 1
    if stats["planes"]:
        model.epi_sax_levels = [float(x) for x in levels]
        model.epi_sax_extra = list(extra)
        stats["extra_levels"] = len(extra)
    return stats


#: Along distance (mm) within which a wall's control point IS a given section.
_SECTION_MATCH_MM = 0.5


def _epi_apex_index(model, axis, pts):
    """Index of the apex vertex of an Epi polyline (as set_long_axis_contour)."""
    if model.epi_apex is not None:
        return int(np.argmin(np.linalg.norm(
            pts - np.asarray(model.epi_apex, float), axis=1)))
    return int(np.argmin((pts - axis.apex) @ axis.axis))


def _wall_radial_at(axis, e_s, wall, along):
    """Signed radial of a wall's DISPLAYED curve at *along*, None if outside it."""
    p = _cr_densify_3d(np.asarray(wall, float))
    a = (p - axis.apex) @ axis.axis
    s = (p - axis.apex) @ e_s
    o = np.argsort(a)
    if along < a[o[0]] or along > a[o[-1]]:
        return None
    return float(np.interp(along, a[o], s[o]))


def move_epi_section(model, plane_angle, vi, along_before, merge_mm=5.0,
                     still_mm=1.0):
    """A long-axis Epi edit point was dragged (the sections exist: after the
    spline). The point is already at its new place in ``epi_planes``; give the
    move its section, so every wall keeps one point per section:

    * moved less than *still_mm* along the axis → it stays on its section (only
      the radial changed), so a hand tremor doesn't shift a whole section;
    * else a section is made at the new along position: the dragged point plus,
      on every other wall, a point on that wall's displayed curve;
    * the existing section nearest the new position, if within *merge_mm*, is
      replaced by it (the curves are sampled first, so its edits carry over);
    * the section the point left, if it survives, gets a curve point on the
      dragged wall. The apex and the MV terminals are never sections.

    Returns {"moved", "level", "removed"} or None when this is not a section point."""
    ax = model.epi_axis
    levels = model.epi_sax_levels
    key = float(plane_angle) % 360.0
    Q = model.epi_planes.get(key)
    if ax is None or not levels or len(levels) < 2 or Q is None:
        return None
    Q = np.asarray(Q, float).reshape(-1, 3)
    api = _epi_apex_index(model, ax, Q)
    if vi in (0, len(Q) - 1, api) or not 0 <= vi < len(Q):
        return None

    def _al(P):
        return float((np.asarray(P, float) - ax.apex) @ ax.axis)

    drag_end = Q[0] if vi < api else Q[-1]
    a0 = float(along_before)
    a = float(np.clip(_al(Q[vi]), 1.0, max(1.0, _al(drag_end) - 1.0)))
    if abs(a - a0) < still_mm:
        Q = Q.copy()
        Q[vi] = Q[vi] + (a0 - _al(Q[vi])) * ax.axis
        model.set_long_axis_contour(key, Q, which="epi")
        return {"moved": False, "level": a0, "removed": None}
    P_new = Q[vi] + (a - _al(Q[vi])) * ax.axis
    inner = sorted(float(x) for x in levels)[:-1]
    base = max(float(x) for x in levels)
    removed = None
    if inner:
        near = min(inner, key=lambda x: abs(x - a))
        if abs(near - a) <= merge_mm:
            removed = near
    source = None
    if inner:
        near = min(inner, key=lambda x: abs(x - a0))
        if abs(near - a0) <= _SECTION_MATCH_MM and near != removed:
            source = near

    def _rewall(idx, e_s, dragged):
        """Wall *idx* (apex first, terminal last) with the new section applied."""
        W = [P.copy() for P in Q_i[idx]]
        pts = list(W[1:-1])
        if dragged:
            pts = [Q_i[j].copy() for j in idx[1:-1] if j != vi]
            W_src = [W[0]] + pts + [W[-1]]
        else:
            W_src = W
        add = []
        if dragged:
            add.append(P_new)
            if source is not None:
                s = _wall_radial_at(ax, e_s, W_src, source)
                if s is not None:
                    add.append(ax.apex + source * ax.axis + s * e_s)
        elif _al(W[-1]) > a + _SECTION_MATCH_MM:
            s = _wall_radial_at(ax, e_s, W_src, a)
            if s is not None:
                add.append(ax.apex + a * ax.axis + s * e_s)
        if removed is not None and pts:
            j = int(np.argmin([abs(_al(P) - removed) for P in pts]))
            if abs(_al(pts[j]) - removed) <= _SECTION_MATCH_MM:
                del pts[j]
        return [W[0]] + sorted(pts + add, key=_al) + [W[-1]]

    for phi in list(model.epi_planes):
        Q_i = np.asarray(model.epi_planes[phi], float).reshape(-1, 3)
        ai = _epi_apex_index(model, ax, Q_i)
        if not 0 < ai < len(Q_i) - 1:
            continue                     # no two walls: nothing to section
        e_s = ax.meridian_dir(phi)
        ia = list(range(ai, -1, -1))     # apex → first end
        ib = list(range(ai, len(Q_i)))   # apex → last end
        same = float(phi) % 360.0 == key
        wa = _rewall(ia, e_s, same and vi < ai)
        wb = _rewall(ib, e_s, same and vi > ai)
        model.set_long_axis_contour(
            float(phi), np.asarray(wa[::-1] + wb[1:], float), which="epi")
    new_inner = sorted([x for x in inner if x != removed] + [a])
    fracs = base * np.asarray([f for f in _SAX_LEVEL_FRACS if f > 1e-6])
    tol = _edit_tol_mm(base)
    model.epi_sax_levels = new_inner + [base]
    model.epi_sax_extra = [x for x in new_inner
                           if np.min(np.abs(fracs - x)) > tol]
    return {"moved": True, "level": a, "removed": removed}
