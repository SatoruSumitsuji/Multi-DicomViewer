"""CPR auto-tracking — trace a coronary centreline through the CT contrast lumen
between points the user clicks (start, optional via points, end).

Phase 1 core (headless). For each consecutive pair of clicked points:

1. Resample an isotropic ROI (``work_mm``) around the pair.
2. Score every voxel's "contrast vessel likelihood" L in [0, 1]: a Frangi
   tubularity (bright tubes, several radii) times an HU gate. The gate is high
   in the contrast range and low for fat, myocardium and calcium. Chambers and
   the aorta are bright but NOT tubular at coronary scales, so they score low
   and the path does not short-cut through them.
3. Run a minimal-cost path (Dijkstra on the 26-neighbour voxel graph, cost per
   mm = 1 / (eps + L)). A short low-contrast gap (stenosis, soft plaque) is
   crossed at a high but finite cost instead of breaking the trace.

4. Re-centre the path sub-voxel in each cross-section on the native volume
   (half-max lumen region around the path, so a neighbouring vein in the same
   cross-section is ignored) — decisive for thin distal vessels.

The result is a dense, smoothed path plus ADAPTIVE control points (≤
``ctrl_step_mm`` apart, denser where the course bends, so the CPR's
Catmull-Rom centreline stays on the path), in world mm, ready to become a CPR
trace's ``pts3d``. Stretches whose centre intensity drops well below the
vessel's own nearby lumen (stenosis / occlusion) are reported as gaps, so the
UI can flag them for review.

Thin-vessel design notes: the gate uses LOCAL contrast (white top-hat), not an
absolute HU threshold (a thin distal lumen blurs to ~100 HU); Frangi uses a
fixed structure constant and the cost saturates at L_sat, so a brighter or
straighter vein next to the artery is not cheaper; the clicked ends snap to
the NEAREST lumen, not the brightest one in reach.

Pure numpy + SciPy — no Qt / VTK / scikit-image — shared by both CT viewers.
Volumes are ``vol[z, y, x]`` with ``dims = (sx, sy, sz)`` mm, world = voxel ·
spacing (origin 0), the convention of ``viewers/cpr_mixin.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

# 13 "forward" offsets of the 26-neighbourhood (the other 13 are their negatives;
# the graph is undirected).
_OFFSETS = np.array([(dz, dy, dx)
                     for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                     if (dz, dy, dx) > (0, 0, 0)], dtype=np.int64)


@dataclass
class TraceParams:
    work_mm: float = 0.4            # isotropic working resolution (finest)
    margin_mm: float = 10.0         # ROI margin around each point pair
    margin_frac: float = 0.25       # ... or this fraction of the pair distance
    roi_retries: int = 3            # grow the ROI when the path hugs its wall
    max_roi_voxels: int = 9_000_000  # coarsen work_mm when the ROI is larger
    scales_mm: tuple = (0.45, 0.7, 1.1, 1.6, 2.3)   # vessel radii (Frangi)
    frangi_c: float = 15.0          # Frangi structure term (HU): saturates for any
    #                                 real vessel, so a THIN dim tube scores like a
    #                                 thick bright one (no bias to bigger veins)
    # LOCAL contrast gate: a thin distal lumen is blurred to ~100 HU, so an
    # absolute HU threshold drops it. Gate on the white top-hat (voxel minus a
    # local opening ≈ the surrounding tissue) instead, plus a loose floor.
    tophat_mm: float = 3.0          # opening size (> the largest lumen radius)
    contrast_lo: float = 45.0       # top-hat contrast gate midpoint (HU)
    contrast_width: float = 15.0
    hu_abs_lo: float = -10.0        # absolute floor (fat / lung never lumen)
    hu_abs_width: float = 20.0
    hu_hi: float = 850.0            # calcium/metal upper edge (gate midpoint)
    hu_hi_width: float = 80.0
    hu_floor: float = -60.0         # voxels below this (fat, lung) are not graph nodes
    eps: float = 0.01               # cost floor term: max cost per mm = 1/eps
    L_sat: float = 0.35             # L above this = "clearly lumen" → same cost,
    #                                 so a brighter / straighter neighbour vein is
    #                                 no cheaper than the clicked thin artery
    gap_rel: float = 0.5            # gap: centre HU below bg + rel·(local lumen − bg)
    gap_ref_mm: float = 12.0        # local lumen reference window (running median)
    snap_radius_mm: float = 1.5     # clicked point → NEAREST lumen within
    snap_sigma_mm: float = 0.7      # distance weighting of that snap
    smooth_mm: float = 0.6          # Gaussian smoothing of the dense path
    sample_mm: float = 0.25         # dense path resampling step
    center_iters: int = 2           # cross-section re-centring passes
    center_radius_mm: float = 2.0   # cross-section half-size for re-centring
    center_max_move_mm: float = 0.8  # per-pass limit (never jumps vessels)
    ctrl_step_mm: float = 5.0       # MAX control-point spacing for pts3d
    ctrl_min_mm: float = 1.2        # min spacing (where the course bends)
    ctrl_tol_mm: float = 0.12       # max deviation of the CPR spline from path
    gap_L: float = 0.15             # L below this counts as "outside lumen"
    gap_min_mm: float = 2.0         # report gaps at least this long


@dataclass
class TraceResult:
    path_mm: np.ndarray                     # (N, 3) dense smoothed path, world mm
    ctrl_mm: np.ndarray                     # (M, 3) control points (pts3d)
    length_mm: float
    gaps: list = field(default_factory=list)   # [(arclen_start, arclen_end)] mm
    seg_costs: list = field(default_factory=list)
    ok: bool = True
    message: str = ""


# --------------------------------------------------------------------------- ROI
def _vol_extent(dims, vol_shape):
    return np.array([vol_shape[2] - 1, vol_shape[1] - 1, vol_shape[0] - 1],
                    float) * np.asarray(dims, float)     # x, y, z extent (mm)


def _roi_grid(p0, p1, dims, vol_shape, prm: TraceParams, grow: float = 1.0):
    """World-mm origin, voxel step and shape of an isotropic ROI enclosing the
    two points (+ margin × grow), clipped to the volume extent."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = float(np.linalg.norm(p1 - p0))
    m = grow * max(prm.margin_mm, prm.margin_frac * d)
    lo = np.minimum(p0, p1) - m
    hi = np.maximum(p0, p1) + m
    ext = _vol_extent(dims, vol_shape)
    lo = np.clip(lo, 0.0, ext); hi = np.clip(hi, 0.0, ext)
    step = float(prm.work_mm)
    while True:
        shape_xyz = np.floor((hi - lo) / step).astype(int) + 1
        if int(np.prod(shape_xyz)) <= prm.max_roi_voxels:
            break
        step *= 1.25
    return lo, step, tuple(int(v) for v in shape_xyz[::-1])   # shape z, y, x


def _sample_roi(vol, dims, origin, step, shape_zyx):
    """Trilinear resample of ``vol`` on the ROI grid → float32 [z, y, x]."""
    sx, sy, sz = (float(v) for v in dims)
    nz, ny, nx = shape_zyx
    zc = (origin[2] + step * np.arange(nz)) / sz
    yc = (origin[1] + step * np.arange(ny)) / sy
    xc = (origin[0] + step * np.arange(nx)) / sx
    # Separable linear interpolation via map_coordinates on a broadcast grid.
    Z, Y, X = np.meshgrid(zc, yc, xc, indexing="ij")
    out = ndimage.map_coordinates(np.asarray(vol, np.float32),
                                  [Z.ravel(), Y.ravel(), X.ravel()],
                                  order=1, mode="nearest")
    return out.reshape(shape_zyx).astype(np.float32)


# -------------------------------------------------------------------- likelihood
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _eig3_sym(H):
    """Eigenvalues (ascending) of many real symmetric 3x3 matrices, closed
    form (trigonometric method) — far faster than batched LAPACK here."""
    a, b, c = H[:, 0, 0].astype(np.float64), H[:, 1, 1].astype(np.float64),         H[:, 2, 2].astype(np.float64)
    d, e, f = H[:, 0, 1].astype(np.float64), H[:, 1, 2].astype(np.float64),         H[:, 0, 2].astype(np.float64)
    p1 = d * d + e * e + f * f
    q = (a + b + c) / 3.0
    p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2.0 * p1
    p = np.sqrt(p2 / 6.0)
    ps = np.where(p > 1e-12, p, 1.0)
    B00, B11, B22 = (a - q) / ps, (b - q) / ps, (c - q) / ps
    B01, B12, B02 = d / ps, e / ps, f / ps
    detB = (B00 * (B11 * B22 - B12 * B12) - B01 * (B01 * B22 - B12 * B02)
            + B02 * (B01 * B12 - B11 * B02))
    r = np.clip(detB / 2.0, -1.0, 1.0)
    phi = np.arccos(r) / 3.0
    e1 = q + 2.0 * p * np.cos(phi)                       # largest
    e3 = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)   # smallest
    e2 = 3.0 * q - e1 - e3
    flat = p <= 1e-12                                    # multiple of identity
    e1 = np.where(flat, q, e1); e2 = np.where(flat, q, e2)
    e3 = np.where(flat, q, e3)
    return np.stack([e3, e2, e1], axis=1).astype(np.float32)


def _frangi_bright(img, step, scales_mm, cand, c_hu=15.0):
    """Max-over-scales Frangi vesselness for BRIGHT tubes, evaluated only at the
    voxels in boolean mask ``cand`` (returns a full-size float32 array). The
    structure term uses a FIXED ``c_hu`` (not the ROI maximum) and the result is
    NOT renormalised, so a dim thin vessel is not out-scored by a bright vein."""
    V = np.zeros(img.shape, np.float32)
    idx = np.nonzero(cand)
    if idx[0].size == 0:
        return V
    alpha2 = 2 * 0.5 ** 2
    beta2 = 2 * 0.5 ** 2
    best = np.zeros(idx[0].size, np.float32)
    # Neighbour indices (clamped) for central finite differences at candidates.
    z, y, x = (a.astype(np.int64) for a in idx)
    nz, ny, nx = img.shape
    zp, zm = np.minimum(z + 1, nz - 1), np.maximum(z - 1, 0)
    yp, ym = np.minimum(y + 1, ny - 1), np.maximum(y - 1, 0)
    xp, xm = np.minimum(x + 1, nx - 1), np.maximum(x - 1, 0)
    for s_mm in scales_mm:
        s = max(0.5, float(s_mm) / step)
        # ONE Gaussian smoothing per scale; the Hessian comes from central
        # differences of it, evaluated only at the candidate voxels (was six
        # full-volume derivative filters per scale — the dominant cost).
        G = ndimage.gaussian_filter(img, s, mode="nearest")
        c0 = G[z, y, x]
        H = np.empty((idx[0].size, 3, 3), np.float32)
        H[:, 0, 0] = G[zp, y, x] - 2 * c0 + G[zm, y, x]
        H[:, 1, 1] = G[z, yp, x] - 2 * c0 + G[z, ym, x]
        H[:, 2, 2] = G[z, y, xp] - 2 * c0 + G[z, y, xm]
        H[:, 0, 1] = H[:, 1, 0] = 0.25 * (G[zp, yp, x] - G[zp, ym, x]
                                          - G[zm, yp, x] + G[zm, ym, x])
        H[:, 0, 2] = H[:, 2, 0] = 0.25 * (G[zp, y, xp] - G[zp, y, xm]
                                          - G[zm, y, xp] + G[zm, y, xm])
        H[:, 1, 2] = H[:, 2, 1] = 0.25 * (G[z, yp, xp] - G[z, yp, xm]
                                          - G[z, ym, xp] + G[z, ym, xm])
        H *= (s * s)                                   # scale-normalised
        del G
        lam = _eig3_sym(H)                             # ascending (signed)
        order_abs = np.argsort(np.abs(lam), axis=1)
        lam = np.take_along_axis(lam, order_abs, axis=1)   # |l1|<=|l2|<=|l3|
        l1, l2, l3 = lam[:, 0], lam[:, 1], lam[:, 2]
        bright = (l2 < 0) & (l3 < 0)
        a2 = np.abs(l2); a3 = np.abs(l3) + 1e-6
        Ra = a2 / a3
        Rb = np.abs(l1) / np.sqrt(a2 * a3 + 1e-12)
        S = np.sqrt(l1 * l1 + l2 * l2 + l3 * l3)
        c = float(c_hu)
        v = ((1 - np.exp(-(Ra * Ra) / alpha2)) * np.exp(-(Rb * Rb) / beta2)
             * (1 - np.exp(-(S * S) / (2 * c * c))))
        v = np.where(bright, v, 0.0).astype(np.float32)
        best = np.maximum(best, v)
    V[idx] = best
    return V


def _local_contrast(sm, step, size_mm):
    """White top-hat: voxel minus a grey opening (flat cube, separable min/max
    filters) — the brightness above the surrounding tissue for structures
    narrower than ``size_mm``. Chambers / aorta (wider) stay ~0 inside."""
    k = max(3, int(round(size_mm / step)) | 1)
    opened = ndimage.maximum_filter(ndimage.minimum_filter(sm, size=k), size=k)
    return sm - opened


def vessel_likelihood(img, step, prm: TraceParams):
    """L in [0, 1] per ROI voxel: tubularity × gate. The gate needs LOCAL
    contrast (top-hat, so thin dim distal lumen passes), a loose absolute floor
    (fat / lung never), and stays below calcium / metal."""
    sm = ndimage.gaussian_filter(img, max(0.5, 0.3 / step), mode="nearest")
    th = _local_contrast(sm, step, 2.0 * prm.tophat_mm)
    gate = (_sigmoid((th - prm.contrast_lo) / prm.contrast_width)
            * _sigmoid((sm - prm.hu_abs_lo) / prm.hu_abs_width)
            * _sigmoid((prm.hu_hi - sm) / prm.hu_hi_width)).astype(np.float32)
    cand = gate > 0.3                 # top-hat ≳ 33 HU: below that L ≈ 0 anyway
    V = _frangi_bright(img, step, prm.scales_mm, cand, prm.frangi_c)
    return np.clip(gate * V, 0.0, 1.0), sm, gate


# -------------------------------------------------------------------- min path
def _snap(L, k, radius_vox, sigma_vox):
    """Move voxel index k (z, y, x) to the NEAREST good lumen voxel: maximise
    L · exp(-d²/2σ²) within radius, so a clicked thin artery is not abandoned
    for a brighter vein a few mm away."""
    r = int(max(0, round(radius_vox)))
    z, y, x = k
    zs = slice(max(0, z - r), z + r + 1)
    ys = slice(max(0, y - r), y + r + 1)
    xs = slice(max(0, x - r), x + r + 1)
    sub = L[zs, ys, xs]
    gz, gy, gx = np.mgrid[zs, ys, xs]
    d2 = (gz - z) ** 2 + (gy - y) ** 2 + (gx - x) ** 2
    inside = d2 <= r * r
    w = np.where(inside, sub * np.exp(-d2 / (2.0 * max(0.5, sigma_vox) ** 2)),
                 -1.0)
    j = np.unravel_index(int(np.argmax(w)), w.shape)
    if w[j] <= 0:
        return k
    return (zs.start + j[0], ys.start + j[1], xs.start + j[2])


def _min_path(cost, node_mask, a, b, step):
    """Dijkstra from voxel a to b over voxels in node_mask (26-neighbour,
    edge = mean cost × length). Returns (list of (z,y,x), total cost) or
    (None, inf) when b is unreachable."""
    shape = cost.shape
    node_mask = node_mask.copy()
    node_mask[a] = True; node_mask[b] = True
    ids = -np.ones(shape, np.int64)
    flat = np.flatnonzero(node_mask)
    ids.ravel()[flat] = np.arange(flat.size)
    rows, cols, wts = [], [], []
    for off in _OFFSETS:
        dz, dy, dx = (int(v) for v in off)
        src = tuple(slice(max(0, -d), shape[i] - max(0, d))
                    for i, d in enumerate((dz, dy, dx)))
        dst = tuple(slice(max(0, d), shape[i] - max(0, -d))
                    for i, d in enumerate((dz, dy, dx)))
        m = node_mask[src] & node_mask[dst]
        if not m.any():
            continue
        length = step * float(np.sqrt(dz * dz + dy * dy + dx * dx))
        rows.append(ids[src][m]); cols.append(ids[dst][m])
        wts.append(0.5 * (cost[src][m] + cost[dst][m]) * length)
    n = flat.size
    G = coo_matrix((np.concatenate(wts), (np.concatenate(rows),
                                          np.concatenate(cols))),
                   shape=(n, n)).tocsr()
    ia = int(ids[a]); ib = int(ids[b])
    dist, pred = dijkstra(G, directed=False, indices=ia,
                          return_predecessors=True)
    if not np.isfinite(dist[ib]):
        return None, float("inf")
    seq = [ib]
    while seq[-1] != ia:
        seq.append(int(pred[seq[-1]]))
    seq.reverse()
    zyx = np.column_stack(np.unravel_index(flat[np.asarray(seq)], shape))
    return [tuple(int(v) for v in r) for r in zyx], float(dist[ib])


# ----------------------------------------------------------------- post-process
def _resample(P, step):
    P = np.asarray(P, float)
    if len(P) < 2:
        return P.copy()
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-9:
        return P[:1].copy()
    n = max(2, int(np.floor(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, P[:, k]) for k in range(3)])


def _smooth(P, sigma_samples, keep_ends=True):
    if len(P) < 5 or sigma_samples <= 0:
        return P
    Q = ndimage.gaussian_filter1d(P, sigma_samples, axis=0, mode="nearest")
    if keep_ends:
        Q[0] = P[0]; Q[-1] = P[-1]              # keep the clicked end points
    return Q


def _tangents(P):
    T = np.gradient(P, axis=0)
    n = np.linalg.norm(T, axis=1, keepdims=True)
    return T / np.maximum(n, 1e-9)


def _center_path(volf, dims, P, prm: TraceParams):
    """Sub-voxel re-centring: move every dense sample to the intensity
    centroid of ITS OWN lumen in the cross-section ⟂ the path, sampled from the
    native volume (trilinear, 0.1 mm grid). The lumen is the connected region
    above half-maximum (between the local peak and the local background) that
    contains the bright spot nearest the sample — so an adjacent vein in the
    same cross-section is ignored. Each pass moves ≤ center_max_move_mm."""
    from scipy.ndimage import label
    sx, sy, sz = (float(v) for v in dims)
    R = prm.center_radius_mm
    g = np.arange(-R, R + 1e-9, 0.1)
    A, B = np.meshgrid(g, g, indexing="ij")
    disc = (A * A + B * B) <= R * R
    near = (A * A + B * B) <= 0.8 ** 2
    c0 = len(g) // 2
    for _ in range(max(0, prm.center_iters)):
        T = _tangents(_smooth(P, 2.0, keep_ends=False))
        ref = np.where(np.abs(T[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]],
                       [[1.0, 0.0, 0.0]])
        U = np.cross(T, ref); U /= np.linalg.norm(U, axis=1, keepdims=True)
        V = np.cross(T, U)
        Q = (P[:, None, None, :] + A[None, :, :, None] * U[:, None, None, :]
             + B[None, :, :, None] * V[:, None, None, :])          # N,g,g,3
        coords = [Q[..., 2].ravel() / sz, Q[..., 1].ravel() / sy,
                  Q[..., 0].ravel() / sx]
        I = ndimage.map_coordinates(volf, coords, order=1,
                                    mode="nearest").reshape(Q.shape[:3])
        newP = P.copy()
        for i in range(len(P)):
            im = I[i]
            peak = float(im[near].max())
            bg = float(np.percentile(im[disc], 20))
            if peak - bg < prm.contrast_lo:         # no lumen here (gap): keep
                continue
            thr = 0.5 * (peak + bg)
            lab, _n = label((im > thr) & disc)
            # Seed = brightest pixel within 0.8 mm of the current sample.
            nm = np.where(near, im, -1e9)
            sa, sb = np.unravel_index(int(np.argmax(nm)), nm.shape)
            lid = lab[sa, sb]
            if lid == 0:
                continue
            reg = lab == lid
            w = (im - thr) * reg
            ws = float(w.sum())
            if ws <= 0:
                continue
            da = float((w * A).sum() / ws); db = float((w * B).sum() / ws)
            mv = np.hypot(da, db)
            if mv > prm.center_max_move_mm:
                da *= prm.center_max_move_mm / mv
                db *= prm.center_max_move_mm / mv
            newP[i] = P[i] + da * U[i] + db * V[i]
        P = newP
    return P


def _adaptive_ctrl(Pd, prm: TraceParams):
    """Control points for the CPR spline: as FEW as possible (≤ ctrl_step_mm
    apart) but dense enough where the course bends that the Catmull-Rom
    centreline (core.centerline) stays within ctrl_tol_mm of the traced path.
    Greedy refinement: start with ctrl_step_mm spacing, then split the segment
    with the worst deviation until the tolerance holds (or ctrl_min_mm)."""
    from multi_dicomviewer.core.centerline import CenterLine
    seg = np.linalg.norm(np.diff(Pd, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    n0 = max(2, int(np.ceil(L / prm.ctrl_step_mm)) + 1)
    knots = list(np.linspace(0.0, L, n0))
    from scipy.spatial import cKDTree
    tree_d = cKDTree(Pd)

    def pts_at(kn):
        return np.column_stack([np.interp(kn, s, Pd[:, k]) for k in range(3)])
    for _ in range(400):
        C = pts_at(np.asarray(knots))
        cl = CenterLine.from_points(C, step_mm=0.2)
        d, _i = tree_d.query(cl.points)
        if d.max() <= prm.ctrl_tol_mm:
            break
        # Arc position (on the dense path) of the worst spline sample → split
        # the knot interval containing it.
        j = int(np.argmax(d))
        sj = s[_i[j]]
        k = int(np.searchsorted(knots, sj))
        k = min(max(k, 1), len(knots) - 1)
        a, b = knots[k - 1], knots[k]
        if b - a < 2 * prm.ctrl_min_mm:
            # Cannot split this one further; try the next worst elsewhere.
            d[np.abs(s[_i] - sj) < (b - a)] = 0.0
            if d.max() <= prm.ctrl_tol_mm:
                break
            j = int(np.argmax(d)); sj = s[_i[j]]
            k = min(max(int(np.searchsorted(knots, sj)), 1), len(knots) - 1)
            a, b = knots[k - 1], knots[k]
            if b - a < 2 * prm.ctrl_min_mm:
                break
        knots.insert(k, 0.5 * (a + b))
    return pts_at(np.asarray(knots))


def thin_ctrl(path_mm, ctrl_mm, keep_mm, spacing_mm):
    """Thin the auto-trace control points *ctrl_mm* to about *spacing_mm*
    apart along the dense path *path_mm*. The points nearest to *keep_mm*
    (the hand-clicked start / via / end) are always kept; an auto point is
    dropped when it is closer than *spacing_mm* (along the path) to the last
    kept point or to the next kept click. Points that are already farther
    apart stay; no point is added. Returns an (M, 3) array in path order."""
    P = np.asarray(path_mm, float)
    C = np.asarray(ctrl_mm, float)
    if len(P) < 2 or len(C) < 2 or float(spacing_mm) <= 0.0:
        return C
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0),
                                                        axis=1))])
    from scipy.spatial import cKDTree
    tree = cKDTree(P)

    def arc(Q):
        return s[tree.query(np.asarray(Q, float).reshape(-1, 3))[1]]
    sc = arc(C)
    # Start / end clicks = the path ends (the trace starts / ends there); the
    # via clicks are kept at their nearest arc position.
    vias = (list(arc(np.asarray(keep_mm, float)[1:-1]))
            if keep_mm is not None and len(keep_mm) > 2 else [])
    lo, hi = float(min(sc[0], 0.0)), float(max(sc[-1], s[-1]))
    keep = sorted(set([lo, hi] + [float(k) for k in vias if lo < k < hi]))
    sp = float(spacing_mm)
    out = []                                  # (arc, point, is_click)
    # Clicks are placed on the path at their arc position; the end ctrl points
    # are the path ends, so they coincide with the start / end clicks.
    for k in keep:
        out.append((k, None, True))
    for a, q in zip(sc, C):
        out.append((float(a), q, False))
    out.sort(key=lambda e: (e[0], not e[2]))
    res, last = [], None
    kept_arcs = np.array(keep)
    for a, q, is_click in out:
        if is_click:
            if last is not None and abs(a - last) < 1e-6:
                continue
            res.append(np.array([np.interp(a, s, P[:, i]) for i in range(3)]))
            last = a
            continue
        if last is not None and a - last < sp:
            continue
        nxt = kept_arcs[kept_arcs > a + 1e-6]
        if len(nxt) and float(nxt[0]) - a < sp:
            continue
        res.append(np.asarray(q, float))
        last = a
    return np.asarray(res, float)


def gap_ctrl_pts(path_mm, ctrl_mm, gaps, pad_mm: float = 1.0) -> list:
    """The edit points *ctrl_mm* that lie in (or within *pad_mm* of) a
    low-contrast gap of the trace (``TraceResult.gaps``, arc lengths along
    *path_mm*). A gap with no edit point inside flags the nearest one, so every
    gap is marked. Returns their positions ([x, y, z] lists, world mm)."""
    P = np.asarray(path_mm, float).reshape(-1, 3)
    C = np.asarray(ctrl_mm, float).reshape(-1, 3)
    if len(P) < 2 or len(C) == 0 or not gaps:
        return []
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0),
                                                        axis=1))])
    from scipy.spatial import cKDTree
    sc = s[cKDTree(P).query(C)[1]]
    flag = np.zeros(len(C), bool)
    for a, b in gaps:
        a, b = float(a), float(b)
        inside = (sc >= a - pad_mm) & (sc <= b + pad_mm)
        if inside.any():
            flag |= inside
        else:
            flag[int(np.argmin(np.abs(sc - 0.5 * (a + b))))] = True
    return [list(map(float, q)) for q in C[flag]]


class TraceCancelled(Exception):
    """Raised inside trace_vessel when the progress callback asks to stop."""


def trace_vessel(vol, dims, points_mm, prm: TraceParams | None = None,
                 progress=None) -> TraceResult:
    """Trace a centreline through the contrast lumen along ``points_mm``
    (start, via…, end; world mm). See the module docstring.

    *progress* (optional) is called as ``progress(stage, i, n, frac)`` —
    stage ∈ {"enhance", "path", "center", "ctrl"}, segment i of n, overall
    fraction 0..1 — and may return False to cancel (→ TraceCancelled)."""
    prm = prm or TraceParams()

    def report(stage, i, n, frac):
        if progress is not None and progress(stage, i, n, frac) is False:
            raise TraceCancelled()
    pts = [np.asarray(p, float) for p in points_mm]
    if len(pts) < 2:
        return TraceResult(np.zeros((0, 3)), np.zeros((0, 3)), 0.0,
                           ok=False, message="need at least 2 points")
    dense, Lprof, costs = [], [], []
    ext = _vol_extent(dims, vol.shape)
    nseg = len(pts) - 1
    for si, (p0, p1) in enumerate(zip(pts[:-1], pts[1:])):
        grow = 1.0
        for _attempt in range(prm.roi_retries + 1):
            report("enhance", si + 1, nseg, 0.8 * si / nseg)
            origin, step, shape = _roi_grid(p0, p1, dims, vol.shape, prm, grow)
            img = _sample_roi(vol, dims, origin, step, shape)
            L, sm, gate = vessel_likelihood(img, step, prm)
            cost = (1.0 / (prm.eps + np.minimum(1.0, L / prm.L_sat))
                    ).astype(np.float32)

            def to_idx(p, origin=origin, step=step, shape=shape):
                k = np.round((p - origin) / step).astype(int)[::-1]
                return tuple(int(np.clip(k[i], 0, shape[i] - 1))
                             for i in range(3))
            a = _snap(L, to_idx(p0), prm.snap_radius_mm / step,
                      prm.snap_sigma_mm / step)
            b = _snap(L, to_idx(p1), prm.snap_radius_mm / step,
                      prm.snap_sigma_mm / step)
            # Graph = the gated (contrast-lumen-like) region dilated ~1 mm, so
            # chambers / myocardium / fat are not even nodes; fall back to the
            # wider soft-tissue graph if that leaves the ends disconnected
            # (long occlusion), then to the full ROI.
            report("path", si + 1, nseg, 0.8 * (si + 0.6) / nseg)
            core = ndimage.binary_dilation(
                gate > 0.1, iterations=max(1, int(round(1.0 / step))))
            path, c = _min_path(cost, core & (sm > prm.hu_floor), a, b, step)
            if path is None:
                path, c = _min_path(cost, sm > prm.hu_floor, a, b, step)
            if path is None:                     # pruned graph disconnected
                path, c = _min_path(cost, np.ones(shape, bool), a, b, step)
            if path is None:
                return TraceResult(np.zeros((0, 3)), np.zeros((0, 3)), 0.0,
                                   ok=False, message="no path between points")
            # A curved vessel can bulge outside the box around its two ends; the
            # path then hugs the ROI wall. Grow the ROI and trace again — but a
            # wall that is the VOLUME's own edge cannot be grown past.
            zyx = np.asarray(path)
            hi_idx = np.array(shape) - 1
            touch = np.zeros(3, bool)                       # per axis z, y, x
            lo_open = origin[::-1] > 1e-6
            hi_open = (origin + step * hi_idx[::-1])[::-1] < ext[::-1] - 1e-6
            touch |= (zyx.min(axis=0) <= 1) & lo_open
            touch |= (zyx.max(axis=0) >= hi_idx - 1) & hi_open
            if not touch.any():
                break
            grow *= 1.8
        costs.append(c)
        zyx = np.asarray(path, float)
        xyz = origin + step * zyx[:, ::-1]
        Lp = L[tuple(np.asarray(path).T)]
        if dense:                                # drop the duplicated joint
            xyz, Lp = xyz[1:], Lp[1:]
        dense.append(xyz); Lprof.append(Lp)
    P = np.vstack(dense)
    Pd = _resample(P, prm.sample_mm)
    Pd = _smooth(Pd, prm.smooth_mm / prm.sample_mm)
    volf = np.asarray(vol, np.float32)
    # Sub-voxel re-centring on the native volume (the voxel path is only as good
    # as the working grid — decisive for thin distal vessels).
    report("center", nseg, nseg, 0.82)
    if prm.center_iters > 0:
        Pd = _center_path(volf, dims, Pd, prm)
        Pd = _smooth(Pd, prm.smooth_mm / prm.sample_mm, keep_ends=False)
    Pd = _resample(Pd, prm.sample_mm)
    length = float(np.sum(np.linalg.norm(np.diff(Pd, axis=0), axis=1)))
    gaps = _find_gaps(volf, dims, Pd, prm)
    report("ctrl", nseg, nseg, 0.93)
    ctrl = _adaptive_ctrl(Pd, prm)
    return TraceResult(Pd, ctrl, length, gaps=gaps, seg_costs=costs, ok=True)


def _find_gaps(volf, dims, Pd, prm: TraceParams):
    """Low-contrast stretches along the final path, RELATIVE to the vessel's own
    lumen: the centre intensity drops below bg + gap_rel·(lumen − bg), where the
    lumen reference is a running median over gap_ref_mm (so a gradual distal fade
    is not a gap, an abrupt stenosis / occlusion is). Arc lengths in mm."""
    sx, sy, sz = (float(v) for v in dims)
    sm = ndimage.map_coordinates(
        volf, [Pd[:, 2] / sz, Pd[:, 1] / sy, Pd[:, 0] / sx], order=1,
        mode="nearest")
    sm = ndimage.gaussian_filter1d(sm, 0.3 / prm.sample_mm)
    w = max(3, int(round(prm.gap_ref_mm / prm.sample_mm)) | 1)
    # Reference = upper quantile of the window (the healthy lumen around it).
    ref = ndimage.percentile_filter(sm, 80, size=w, mode="nearest")
    bg = prm.hu_floor
    low = sm < bg + prm.gap_rel * (ref - bg)
    s = np.arange(len(Pd)) * prm.sample_mm
    gaps, run0 = [], None
    for i, lo in enumerate(low):
        if lo and run0 is None:
            run0 = i
        if (not lo or i == len(low) - 1) and run0 is not None:
            j = i if lo else i - 1
            if s[j] - s[run0] >= prm.gap_min_mm:
                gaps.append((float(s[run0]), float(s[j])))
            run0 = None
    return gaps
