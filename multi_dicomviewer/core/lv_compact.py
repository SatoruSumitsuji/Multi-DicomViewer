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


def clip_mask_by_planes(comp, bbox, spacing_xyz, apex_xyz, planes):
    """Keep only the voxels of *comp* on the APEX side of each (centre, normal)
    valve plane — the SAME clip LVSurface.inside_mask_bbox applies to the Epi/Endo
    surfaces. Used to trim the Auto-Endo envelope's flat basal cut back to the
    (tilted) MV / AoV planes so its base coincides with the valve-clipped Epi.
    *spacing_xyz* = (sx,sy,sz) mm; *bbox* = (z0,z1,y0,y1,x0,x1). Returns a new
    boolean comp of the same shape."""
    comp = np.asarray(comp, bool)
    planes = [(np.asarray(c, float), np.asarray(n, float))
              for (c, n) in (planes or []) if c is not None and n is not None]
    if not planes or not comp.any():
        return comp
    z0, z1, y0, y1, x0, x1 = bbox
    sx, sy, sz = spacing_xyz
    apex = np.asarray(apex_xyz, float)
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
    wx = xx * sx
    wy = yy * sy
    wz = zz * sz
    out = comp.copy()
    for c, nrm in planes:
        nn = nrm / (float(np.linalg.norm(nrm)) or 1.0)
        if float((apex - c) @ nn) < 0.0:            # point the normal apex-ward
            nn = -nn
        d = (wx - c[0]) * nn[0] + (wy - c[1]) * nn[1] + (wz - c[2]) * nn[2]
        out &= (d >= 0.0)
    return out


def max_perp_diameter(comp, bbox, apex_xyz, axis_dir, radial0, spacing_xyz,
                      level_mm: float = 1.5, n_dir: int = 90,
                      min_pts: int = 6, frac_lo: float = 0.15,
                      frac_hi: float = 0.85, return_detail: bool = False):
    """Maximum in-plane diameter of a mask on the planes PERPENDICULAR to the LV
    long axis — the clinical LVDd-style "max short-axis diameter".

    With *return_detail* True, returns ``(mm, p1_xyz, p2_xyz)`` where p1/p2 are
    the world-mm endpoints of the winning chord (for drawing where it was
    measured), or None. Otherwise returns the diameter (mm) or None.

    *comp* is a boolean sub-volume [dz,dy,dx] over *bbox* (z0,z1,y0,y1,x0,x1),
    *spacing_xyz* = (sx,sy,sz) mm. Voxels are projected onto the axis (level) and
    onto the perpendicular plane; per along-axis level the TRUE max chord = the
    max over *n_dir* directions of (max−min projection) — this equals the largest
    pairwise distance in that slice. Returns the max over levels (mm), or None.

    LVDd site: the per-level diameter profile is scanned from the BASE toward the
    apex; the wide sub-aortic / LVOT slices near the base are skipped by taking
    the max only APICAL of the first profile valley (the basal narrowing that
    marks entry into the true LV body). [*frac_lo*, *frac_hi*] is only the
    fallback band used when the profile has no such valley (monotonic).

    NB this is the real diameter (widest cavity slice), NOT 2·max-radius — the
    latter overreads on an off-centre / flared basal slice."""
    comp = np.asarray(comp, bool)
    zz, yy, xx = np.nonzero(comp)
    if zz.size < 2:
        return None
    z0, _z1, y0, _y1, x0, _x1 = bbox
    sx, sy, sz = spacing_xyz
    apex = np.asarray(apex_xyz, float)
    a = np.asarray(axis_dir, float)
    a = a / (float(np.linalg.norm(a)) or 1.0)
    e1 = np.asarray(radial0, float)
    e1 = e1 - float(e1 @ a) * a                  # orthogonalise to the axis
    n1 = float(np.linalg.norm(e1))
    if n1 < 1e-9:                                # radial0 ∥ axis → any perp basis
        e1 = np.array([1.0, 0.0, 0.0]) - a[0] * a
        n1 = float(np.linalg.norm(e1)) or 1.0
    e1 = e1 / n1
    e2 = np.cross(a, e1)
    P = np.column_stack([(xx + x0) * sx, (yy + y0) * sy,
                         (zz + z0) * sz]).astype(float) - apex
    along = P @ a
    u = P @ e1
    w = P @ e2
    # Build the per-level short-axis diameter PROFILE over the whole apex→base
    # span (true max chord per level + its endpoints).
    lvl = np.round(along / float(level_mm)).astype(int)
    th = np.linspace(0.0, np.pi, int(n_dir), endpoint=False)
    cs, sn = np.cos(th), np.sin(th)
    prof = []                                        # (along_c, dia, P1, P2)
    for L in np.unique(lvl):
        m = lvl == L
        if int(m.sum()) < int(min_pts):
            continue
        um, wm, am = u[m], w[m], along[m]
        proj = np.outer(um, cs) + np.outer(wm, sn)   # (n, n_dir)
        wd = proj.max(0) - proj.min(0)               # (n_dir,)
        ki = int(np.argmax(wd))
        col = proj[:, ki]
        i_hi, i_lo = int(np.argmax(col)), int(np.argmin(col))
        prof.append((
            float(am.mean()), float(wd[ki]),
            apex + am[i_hi] * a + um[i_hi] * e1 + wm[i_hi] * e2,
            apex + am[i_lo] * a + um[i_lo] * e1 + wm[i_lo] * e2))
    if not prof:
        return None
    prof.sort(key=lambda r: r[0])                    # apex (low along) → base
    dia = np.array([r[1] for r in prof], float)
    n = len(dia)
    # LVDd site: scan from the BASE toward the apex; the sub-aortic / LVOT slices
    # are wide, then the cavity NARROWS (a valley) entering the true LV body,
    # then widens again to the LV max. Take the max APICAL of that first valley,
    # so the wide sub-aortic base is excluded. Light 3-tap smoothing avoids noise
    # valleys. If there's no such valley (monotonic), fall back to the mid
    # [frac_lo, frac_hi] band.
    sm = dia if n < 3 else np.convolve(dia, np.ones(3) / 3.0, mode="same")
    valley = None
    for i in range(n - 2, 0, -1):                    # base → apex
        if sm[i] <= sm[i + 1] and sm[i] < sm[i - 1]:
            valley = i
            break
    if valley is not None:
        lo_i, hi_i = 0, valley + 1                   # apical side of the valley
    else:
        lo_i = int(np.floor(frac_lo * (n - 1)))
        hi_i = int(np.ceil(frac_hi * (n - 1))) + 1
    lo_i = max(0, lo_i)
    hi_i = min(n, hi_i)
    if hi_i <= lo_i:
        lo_i, hi_i = 0, n
    bi = lo_i + int(np.argmax(dia[lo_i:hi_i]))
    best = float(dia[bi])
    best_ep = (prof[bi][2], prof[bi][3])
    if best <= 0:
        return None
    if return_detail:
        return float(best), best_ep[0], best_ep[1]
    return float(best)


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


def _polar_close(r: np.ndarray, k: int) -> np.ndarray:
    """Circular 1-D grey CLOSING (dilate then erode) of a radial profile r[θ]
    with a flat window of half-width *k* samples. Fills INWARD dips (papillary /
    trabecular notches up to ~2k samples wide) while leaving the broad cavity
    shape intact — the angular analogue of the 2-D closing, but it only bridges
    radial dips instead of rounding the whole contour."""
    m = len(r)
    if k < 1 or m == 0:
        return r
    idx = np.arange(m)
    dil = r.copy()
    for s in range(-k, k + 1):                         # dilation = rolling max
        dil = np.maximum(dil, r[(idx + s) % m])
    ero = dil.copy()
    for s in range(-k, k + 1):                         # erosion  = rolling min
        ero = np.minimum(ero, dil[(idx + s) % m])
    return ero


def _hull_radius_profile(filled: np.ndarray, ctr: int, rays, grid_mm: float):
    """Convex-hull radius (mm) along each ray from the grid centre. The blood
    cross-section's convex hull bridges ALL inward notches (papillary muscles /
    trabeculae) with straight chords — the most aggressive "papillary-included"
    envelope (matches the smooth curve a human draws), at the cost of a little
    over-inclusion where the cavity is genuinely concave (base / LVOT)."""
    ys, xs = np.nonzero(filled)
    n = len(rays)
    if len(xs) < 3:
        return [0.0] * n
    pu = xs.astype(float) - ctr                        # u = column
    pv = ys.astype(float) - ctr                        # v = row
    pts = np.column_stack([pu, pv])
    try:
        from scipy.spatial import ConvexHull
        verts = pts[ConvexHull(pts).vertices]          # ordered boundary loop
    except Exception:                                   # noqa: BLE001
        return [0.0] * n
    nH = len(verts)
    out = []
    for _th, dx, dy in rays:
        # Cast the ray  P = t·(dx,dy), t>0  against each hull edge a→b; the exit
        # crossing (largest valid t with 0≤s≤1) is the hull boundary radius.
        best_t = 0.0
        for i in range(nH):
            a = verts[i]
            e = verts[(i + 1) % nH] - a
            denom = dx * e[1] - dy * e[0]              # d × e
            if abs(denom) < 1e-9:
                continue
            tt = (a[0] * e[1] - a[1] * e[0]) / denom    # (a × e)/(d × e)
            ss = (a[0] * dy - a[1] * dx) / denom        # (a × d)/(d × e)
            if tt > best_t and -1e-9 <= ss <= 1.0 + 1e-9:
                best_t = tt
        out.append(best_t * grid_mm)
    return out


def endo_convex3d_mask(blood, spacing_xyz, apex_xyz, axis_dir,
                       along_apex, along_base):
    """3-D CONVEX HULL of the blood pool within the valve-to-apex range. Being
    convex, EVERY planar section (short-axis OR any oblique long-axis) is convex
    — no inward dips — and it contains ALL blood; a rounded bullet Endo, far
    tighter than a per-level circumscribing circle. Returns (comp, bbox) or None.
    """
    try:
        from scipy import ndimage
        from scipy.spatial import ConvexHull
    except Exception:                                   # noqa: BLE001
        return None
    blood = np.asarray(blood, bool)
    bz, by, bx = np.where(blood)
    if len(bz) < 8:
        return None
    sx, sy, sz = spacing_xyz
    apex = np.asarray(apex_xyz, float)
    n = np.asarray(axis_dir, float)
    n = n / (np.linalg.norm(n) or 1.0)
    # Work in the blood's tight bbox (the blood is a small region in a big
    # volume) so erosion / the voxel grid stay small.
    z0, z1 = int(bz.min()), int(bz.max()) + 1
    y0, y1 = int(by.min()), int(by.max()) + 1
    x0, x1 = int(bx.min()), int(bx.max()) + 1
    bsub = blood[z0:z1, y0:y1, x0:x1]
    # The hull is set by EXTREME points only, so feed ConvexHull just the SURFACE
    # voxels (blood minus its erosion) — a few thousand instead of millions.
    surf = bsub & ~ndimage.binary_erosion(bsub)
    szs, sys_, sxs = np.where(surf)
    P = np.column_stack([(sxs + x0) * sx, (sys_ + y0) * sy,
                         (szs + z0) * sz]).astype(float)
    along = (P - apex) @ n
    keep = (along >= float(along_apex)) & (along <= float(along_base))
    if int(keep.sum()) < 8:
        return None
    P = P[keep]
    try:
        from scipy.spatial import Delaunay
        hull = ConvexHull(P)
        tri = Delaunay(P[hull.vertices])   # tiny (hull vertices) → fast queries
    except Exception:                                   # noqa: BLE001
        return None
    gz, gy, gx = np.mgrid[z0:z1, y0:y1, x0:x1]
    gp = np.column_stack([gx.ravel() * sx, gy.ravel() * sy,
                          gz.ravel() * sz]).astype(float)
    # Inside-hull test via a Delaunay triangulation of the hull VERTICES (a few
    # hundred pts) — find_simplex is C-optimised and ~7× faster than testing
    # every voxel against every face equation.
    comp = (tri.find_simplex(gp) >= 0).reshape(gz.shape)
    if not comp.any():
        return None
    return comp, (z0, z1, y0, y1, x0, x1)


def endo_envelope_mask(blood, spacing_xyz, apex_xyz, axis_dir, radial0,
                       along_apex, along_base, sax_step_mm=1.0, close_mm=4.0,
                       half_mm=70.0, grid_mm=0.8, method="close",
                       bridge_deg=40.0, n_meridians=180, roundness=0.0,
                       bulge_frac=0.20, min_chord_mm=5.0):
    """3-D ENDOCARDIAL ENVELOPE mask = the blood pool with its papillary /
    trabecular indentations bridged, per short-axis level, rasterised back into
    the volume grid. Returns ``(comp bool[dz,dy,dx], bbox)`` or None.

    Unlike the per-meridian radial loft (endo_contours_from_blood → LVModel),
    this is a VOLUME mask built as ``endo = blood | (bridged additions)``, so the
    Endo cavity is GUARANTEED to contain the blood pool everywhere (no blood
    poking out on the myocardium side between sparse meridians). *method* bridges
    the same way as endo_contours_from_blood (close / polar / hull).
    """
    try:
        from scipy import ndimage
    except Exception:                                   # noqa: BLE001
        return None
    blood = np.asarray(blood, bool)
    if not blood.any():
        return None
    endo = blood.copy()
    nz, ny, nx = blood.shape
    sx, sy, sz = spacing_xyz
    apex = np.asarray(apex_xyz, float)
    n = np.asarray(axis_dir, float)
    n = n / (np.linalg.norm(n) or 1.0)
    u = np.asarray(radial0, float)
    u = u - u.dot(n) * n
    u = u / (np.linalg.norm(u) or 1.0)
    v = np.cross(n, u)
    se = _disk(close_mm / grid_mm)
    n_mer = max(8, int(n_meridians))
    thetas = np.array([360.0 * i / n_mer for i in range(n_mer)] + [360.0])
    rays = [(360.0 * i / n_mer,
             math.cos(math.radians(360.0 * i / n_mer)),
             math.sin(math.radians(360.0 * i / n_mer))) for i in range(n_mer)]
    deg_per = 360.0 / n_mer
    k_win = max(1, int(round((bridge_deg * 0.5) / deg_per)))
    ng = max(2, int(2.0 * half_mm / grid_mm) + 1)
    ctr = ng // 2
    ax = np.linspace(-half_mm, half_mm, ng)
    uu, vv = np.meshgrid(ax, ax)
    ang_grid = np.degrees(np.arctan2(vv, uu)) % 360.0    # (ng,ng)
    rad_grid = np.hypot(uu, vv)                          # mm from centre
    t = float(along_apex)
    while t <= float(along_base) + 1e-6:
        centre = apex + t * n
        g, _cell = _sample_on_plane(blood, spacing_xyz, centre, u, v,
                                    half_mm, grid_mm)
        if g.any():
            filled = ndimage.binary_fill_holes(g)
            if method == "circle":
                # 外接円: the min-enclosing circle of the blood at this level —
                # centre = blood centroid, radius = the farthest blood pixel — so
                # EVERY blood voxel of the level is inside. Stacked apex→base this
                # is the classic bullet Endo; base/apex taper naturally as the
                # blood narrows.
                ys_p, xs_p = np.nonzero(filled)
                cu = float(xs_p.mean())
                cv = float(ys_p.mean())
                rr = float(np.sqrt((xs_p - cu) ** 2
                                   + (ys_p - cv) ** 2).max()) + 1.0
                gy, gx = np.ogrid[0:filled.shape[0], 0:filled.shape[1]]
                env = ((gx - cu) ** 2 + (gy - cv) ** 2) <= rr * rr
            elif method == "hull_smooth":
                # Per-level convex hull, then a Catmull-Rom spline through its
                # vertices so the straight chords arc smoothly OUTWARD — a clean
                # rounded Endo that still contains the blood (spline ⊇ hull).
                try:
                    import cv2
                    ys_p, xs_p = np.nonzero(filled)
                    hv = cv2.convexHull(
                        np.column_stack([xs_p, ys_p]).astype(np.int32)
                    ).reshape(-1, 2).astype(float)
                    # Bulge ONLY the long chords (concavities); short hull edges
                    # stay on the blood so convex walls aren't pushed into myo.
                    sm = _hull_bulge_long_chords(
                        hv, grid_mm, min_chord_mm=min_chord_mm,
                        bulge_frac=bulge_frac)
                    env = np.zeros(filled.shape, np.uint8)
                    cv2.fillPoly(env, [np.round(sm).astype(np.int32)], 1)
                    env = env.astype(bool)
                except Exception:                        # noqa: BLE001
                    env = ndimage.binary_fill_holes(filled)
            elif method in ("hull", "hull_round"):
                rs = _hull_radius_profile(filled, ctr, rays, grid_mm)
                if method == "hull_round" and roundness > 0.0:
                    # Blend the convex-hull radius toward the circumscribing
                    # circle (its max radius) so the polygon rounds toward a
                    # circle: roundness 0 = hull, 1 = full circle.
                    arr = np.asarray(rs, float)
                    rmax = float(arr.max()) if arr.size else 0.0
                    rs = list(arr + float(roundness) * (rmax - arr))
                env = _radius_to_mask(rs, thetas, ang_grid, rad_grid)
            elif method == "polar":
                rs = [_envelope_radius(filled, ctr, dx, dy, grid_mm)
                      for _th, dx, dy in rays]
                rs = list(_polar_close(np.asarray(rs, float), k_win))
                env = _radius_to_mask(rs, thetas, ang_grid, rad_grid)
            else:                                        # "close"
                env = ndimage.binary_fill_holes(
                    ndimage.binary_closing(filled, structure=se))
            sel = np.where(env)
            uo = -half_mm + sel[1].astype(float) * grid_mm   # col → u
            vo = -half_mm + sel[0].astype(float) * grid_mm   # row → v
            px = centre[0] + uo * u[0] + vo * v[0]
            py = centre[1] + uo * u[1] + vo * v[1]
            pz = centre[2] + uo * u[2] + vo * v[2]
            ix = np.round(px / sx).astype(np.int64)
            iy = np.round(py / sy).astype(np.int64)
            iz = np.round(pz / sz).astype(np.int64)
            ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
                  & (iz >= 0) & (iz < nz))
            endo[iz[ok], iy[ok], ix[ok]] = True
        t += float(sax_step_mm)
    zs, ys, xs = np.where(endo)
    if len(zs) == 0:
        return None
    # 3-D CLOSING scaled by close_mm (肉柱), applied for EVERY method in a cropped
    # box (blood extent + margin) for speed: it bridges the papillary/trabecular
    # indentations in 3-D — not just per short-axis level — so an obliquely-cut
    # Endo no longer dips into the notches but sits smoothly further OUT. Closing
    # only fills concavities (convex parts stay put) and we re-union blood, so
    # endo ⊇ blood still holds. Larger 肉柱 → smoother / further out.
    it = max(1, min(32, int(round(float(close_mm) / max(1e-3, min(spacing_xyz))))))
    mrg = it + 4
    bz0, bz1 = max(0, int(zs.min()) - mrg), min(nz, int(zs.max()) + 1 + mrg)
    by0, by1 = max(0, int(ys.min()) - mrg), min(ny, int(ys.max()) + 1 + mrg)
    bx0, bx1 = max(0, int(xs.min()) - mrg), min(nx, int(xs.max()) + 1 + mrg)
    sub = endo[bz0:bz1, by0:by1, bx0:bx1]
    sub = ndimage.binary_closing(sub, iterations=it)
    sub |= blood[bz0:bz1, by0:by1, bx0:bx1]             # re-guarantee ⊇ blood
    sub = ndimage.binary_fill_holes(sub)
    endo[bz0:bz1, by0:by1, bx0:bx1] = sub
    zs, ys, xs = np.where(endo)
    bbox = (int(zs.min()), int(zs.max()) + 1, int(ys.min()), int(ys.max()) + 1,
            int(xs.min()), int(xs.max()) + 1)
    z0, z1, y0, y1, x0, x1 = bbox
    return endo[z0:z1, y0:y1, x0:x1].copy(), bbox


def _hull_bulge_long_chords(hull, grid_mm, min_chord_mm=5.0, bulge_frac=0.20):
    """Smooth the convex hull by arcing ONLY its LONG edges (those that span an
    inward concavity — the straight chords the user wants rounded OUT) while
    leaving SHORT edges (where the hull already hugs the blood) exactly on the
    blood, so convex walls are NOT pushed out into myocardium. Each long edge is
    replaced by an OUTWARD quadratic-Bezier arc; short edges stay straight.
    *hull* = ordered (N,2) hull vertices (pixels). Returns dense (M,2) pixels."""
    hull = np.asarray(hull, float)
    n = len(hull)
    if n < 3:
        return hull
    cen = hull.mean(axis=0)
    min_px = float(min_chord_mm) / max(1e-6, float(grid_mm))
    out = []
    for i in range(n):
        v0 = hull[i]
        v1 = hull[(i + 1) % n]
        seg = v1 - v0
        L = float(np.hypot(seg[0], seg[1]))
        if L <= min_px:
            out.append(v0)                               # short → keep on the blood
            continue
        nrm = np.array([-seg[1], seg[0]], float)         # perpendicular
        nn = np.hypot(nrm[0], nrm[1]) or 1.0
        nrm = nrm / nn
        mid = 0.5 * (v0 + v1)
        if float(np.dot(mid - cen, nrm)) < 0.0:          # point OUTWARD
            nrm = -nrm
        ctrl = mid + nrm * (bulge_frac * L)              # bulge amount ∝ chord
        m = max(4, min(24, int(L / 2.0)))                # arc subdivisions
        for t in np.linspace(0.0, 1.0, m, endpoint=False):
            omt = 1.0 - t                                # quadratic Bezier
            out.append(omt * omt * v0 + 2 * omt * t * ctrl + t * t * v1)
    return np.asarray(out, float)


def _catmull_closed(P, subdiv: int = 10):
    """Closed uniform Catmull-Rom spline through points *P* (Nx2). Between the
    control points the curve bulges OUTWARD for a convex polygon (the tangents
    follow the hull), so hull vertices → a smooth convex curve that arcs out past
    the straight chords. Returns the densified (M,2) points."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 3:
        return P
    ts = np.linspace(0.0, 1.0, int(subdiv), endpoint=False)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        for t in ts:
            t2 = t * t
            t3 = t2 * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.asarray(out, float)


def _radius_to_mask(rs, thetas, ang_grid, rad_grid):
    """Fill a star polygon r(θ) into the SAX grid: pixel inside iff its radius
    (mm) <= the profile radius at its angle (circular-interpolated)."""
    rr = np.asarray(list(rs) + [rs[0]], float)          # wrap for interp
    r_at = np.interp(ang_grid, thetas, rr)
    return rad_grid <= r_at


def region_outline_on_plane(comp, bbox, spacing_xyz, origin, u, v,
                            half_mm=100.0, step_mm=0.8, convex=False):
    """Outline polygon(s) of a 3-D region's CROSS-SECTION on an arbitrary plane.

    Samples the boolean region (``comp`` = bbox-local mask, ``bbox`` =
    (z0,z1,y0,y1,x0,x1) into the full volume) on a square grid centred at
    *origin* (world mm) with in-plane axes *u*, *v* (unit, world), extent
    +/-half_mm at pitch *step_mm*, then traces the mask boundary. Returns a list
    of polygons, each a list of (out_u, out_v) points in the plane's OUTPUT
    coordinates (origin = (0,0), same frame the viewers reslice in) — ready to
    draw as the region's border that TRACKS free rotation (it is the section of
    the reconstructed solid, so it always coincides with the resliced fill).

    [] if the plane misses the region or OpenCV is unavailable.
    """
    try:
        import cv2
    except Exception:                                   # noqa: BLE001
        return []
    comp = np.asarray(comp, bool)
    if comp.size == 0 or not comp.any():
        return []
    z0, z1, y0, y1, x0, x1 = bbox
    sx, sy, sz = spacing_xyz
    o = np.asarray(origin, float)
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    n = max(2, int(2.0 * half_mm / step_mm) + 1)
    ax = np.linspace(-half_mm, half_mm, n)
    uu, vv = np.meshgrid(ax, ax)                        # uu[i,j]=ax[j], vv[i,j]=ax[i]
    px = o[0] + uu * u[0] + vv * v[0]
    py = o[1] + uu * u[1] + vv * v[1]
    pz = o[2] + uu * u[2] + vv * v[2]
    ix = np.round(px / sx).astype(np.int64)
    iy = np.round(py / sy).astype(np.int64)
    iz = np.round(pz / sz).astype(np.int64)
    inb = ((ix >= x0) & (ix < x1) & (iy >= y0) & (iy < y1)
           & (iz >= z0) & (iz < z1))
    g = np.zeros((n, n), np.uint8)
    sel = np.where(inb)
    g[sel] = comp[iz[sel] - z0, iy[sel] - y0, ix[sel] - x0]
    if not g.any():
        return []
    cnts, _h = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if convex:
        # A single CONVEX outline that CIRCUMSCRIBES the section (an "外接" curve)
        # — the smooth envelope the user wants for Auto-Endo: hull all the section
        # points, then resample + Taubin-smooth so the corners round off. Since
        # the mask ⊇ blood, the hull ⊇ blood too (no blood pokes outside it).
        allpts = np.vstack([c.reshape(-1, 2) for c in cnts
                            if len(c) >= 3]) if cnts else None
        if allpts is None or len(allpts) < 3:
            return []
        hull = cv2.convexHull(allpts.astype(np.int32)).reshape(-1, 2)
        ring = np.column_stack([-half_mm + hull[:, 0].astype(float) * step_mm,
                                -half_mm + hull[:, 1].astype(float) * step_mm])
        return [_smooth_ring(_resample_closed(ring, 96), passes=16)]
    polys = []
    for c in cnts:
        pc = c.reshape(-1, 2)                           # (col=j, row=i)
        if len(pc) < 3:
            continue
        ring = np.column_stack([-half_mm + pc[:, 0].astype(float) * step_mm,
                                -half_mm + pc[:, 1].astype(float) * step_mm])
        # Arc-length resample dense, then periodic low-pass to erase the pixel
        # staircase (the caller Catmull-Roms the result) — a smooth outline.
        polys.append(_smooth_ring(_resample_closed(ring, 96), passes=16))
    # Keep the biggest ring first (the cavity); tiny specks (noise) sort after.
    polys.sort(key=len, reverse=True)
    return polys


def _smooth_ring(pts, passes=12, lam=0.63, mu=-0.67):
    """TAUBIN (λ|μ) smoothing of a closed ring: erases the pixel-staircase wiggle
    WITHOUT the inward shrink a plain moving-average causes (the μ step re-
    inflates), so the smoothed Endo outline stays out at the wall. Returns a list
    of (x, y)."""
    p = np.asarray(pts, float)
    n = len(p)
    if n < 5:
        return [tuple(map(float, q)) for q in p]
    idx = np.arange(n)

    def _lap(q):                                       # neighbour-average − self
        return 0.5 * (q[(idx - 1) % n] + q[(idx + 1) % n]) - q

    for _ in range(max(1, int(passes))):
        p = p + lam * _lap(p)
        p = p + mu * _lap(p)
    return [(float(x), float(y)) for x, y in p]


def _resample_closed(ring: np.ndarray, n: int):
    """Resample a closed polygon to *n* points EVENLY BY ARC LENGTH, so the
    pixel-jagged contour becomes a smooth, uniformly-sampled ring ready for a
    Catmull-Rom spline. Returns a list of (x, y)."""
    p = np.asarray(ring, float)
    if len(p) < 3:
        return [tuple(map(float, q)) for q in p]
    closed = np.vstack([p, p[:1]])                      # wrap to close
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 1e-6:
        return [tuple(map(float, q)) for q in p]
    targets = np.linspace(0.0, total, n, endpoint=False)
    xs = np.interp(targets, s, closed[:, 0])
    ys = np.interp(targets, s, closed[:, 1])
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def endo_contours_from_blood(blood, spacing_xyz, apex_xyz, axis_dir, radial0,
                             n_meridians, along_apex, along_base,
                             sax_step_mm=1.0, close_mm=4.0, half_mm=60.0,
                             grid_mm=0.6, method="close", bridge_deg=40.0):
    """Per-meridian (along, radius) profiles of the ENDOCARDIAL ENVELOPE derived
    from the blood pool — the compact-layer inner surface (papillary/trabeculae
    INCLUDED in the cavity), ready to load as an editable Endo border.

    For each short-axis level the blood is filled (enclosed papillary islands)
    then the wall-attached notches (papillary muscles / trabeculae) are bridged
    into a smooth envelope; its radial extent is measured along each of
    *n_meridians* directions (θ=0 along *radial0*, LVModel's meridian convention).

    *method* selects how the notches are bridged:
      * ``"close"`` — 2-D morphological closing with a disk of radius *close_mm*
        (the original; isotropic, rounds the whole contour — a blunt lever for
        broad papillaries).
      * ``"polar"`` — measure the raw outermost-blood radius per ray on the
        FILLED blood, then bridge inward dips with a 1-D circular closing over θ
        of angular width ~*bridge_deg* (targeted: fills papillary dips, keeps
        the rest of the shape).
      * ``"hull"`` — per-level convex hull of the blood: bridges every inward
        notch with a straight chord (most aggressive; may slightly over-include
        where the cavity is genuinely concave).

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
    n_mer = int(n_meridians)
    thetas = [360.0 * i / n_mer for i in range(n_mer)]
    rays = [(th, math.cos(math.radians(th)), math.sin(math.radians(th)))
            for th in thetas]
    # Polar-close half-width: samples spanning ~bridge_deg of arc.
    deg_per = 360.0 / max(1, n_mer)
    k_win = max(1, int(round((bridge_deg * 0.5) / deg_per)))
    out: dict = {th: [] for th in thetas}
    slab = float(sax_step_mm)
    t = float(along_apex)
    while t <= float(along_base) + 1e-6:
        centre = apex + t * n
        g, _cell = _sample_on_plane(blood, spacing_xyz, centre, u, v,
                                    half_mm, grid_mm)
        if g.any():
            filled = ndimage.binary_fill_holes(g)       # enclosed islands
            ctr = filled.shape[0] // 2                   # (u,v)=(0,0) pixel
            if method == "hull":
                rs = _hull_radius_profile(filled, ctr, rays, grid_mm)
            elif method == "polar":
                rs = [_envelope_radius(filled, ctr, dx, dy, grid_mm)
                      for _th, dx, dy in rays]
                rs = list(_polar_close(np.asarray(rs, float), k_win))
            else:                                        # "close" (original)
                env = ndimage.binary_closing(filled, structure=se)
                env = ndimage.binary_fill_holes(env)
                rs = [_envelope_radius(env, ctr, dx, dy, grid_mm)
                      for _th, dx, dy in rays]
            for (th, _dx, _dy), r in zip(rays, rs):
                if r > 0.0:
                    out[th].append((float(t), float(r)))
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
