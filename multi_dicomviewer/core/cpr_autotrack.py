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

The result is a dense, smoothed path plus control points every ``ctrl_step_mm``
in world mm, ready to become a CPR trace's ``pts3d`` (then Fit / ReCalc as
usual). Segments that ran through low-likelihood voxels are reported as gaps,
so the UI can flag them for review.

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
    work_mm: float = 0.5            # isotropic working resolution of the ROI
    margin_mm: float = 10.0         # ROI margin around each point pair
    margin_frac: float = 0.25       # ... or this fraction of the pair distance
    roi_retries: int = 3            # grow the ROI when the path hugs its wall
    max_roi_voxels: int = 6_000_000  # coarsen work_mm when the ROI is larger
    scales_mm: tuple = (0.7, 1.1, 1.6, 2.3)   # vessel radii probed (Frangi)
    hu_lo: float = 150.0            # contrast lumen lower edge (gate midpoint)
    hu_lo_width: float = 40.0
    hu_hi: float = 850.0            # calcium/metal upper edge (gate midpoint)
    hu_hi_width: float = 80.0
    hu_floor: float = -60.0         # voxels below this (fat, lung) are not graph nodes
    eps: float = 0.02               # cost floor term: max cost per mm = 1/eps
    snap_radius_mm: float = 2.0     # clicked point → best lumen voxel within
    smooth_mm: float = 1.0          # Gaussian smoothing of the dense path
    sample_mm: float = 0.5          # dense path resampling step
    ctrl_step_mm: float = 5.0       # control-point spacing for pts3d
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


def _frangi_bright(img, step, scales_mm, cand):
    """Max-over-scales Frangi vesselness for BRIGHT tubes, evaluated only at the
    voxels in boolean mask ``cand`` (returns a full-size float32 array)."""
    V = np.zeros(img.shape, np.float32)
    idx = np.nonzero(cand)
    if idx[0].size == 0:
        return V
    alpha2 = 2 * 0.5 ** 2
    beta2 = 2 * 0.5 ** 2
    best = np.zeros(idx[0].size, np.float32)
    orders = {(2, 0, 0): (0, 0), (0, 2, 0): (1, 1), (0, 0, 2): (2, 2),
              (1, 1, 0): (0, 1), (1, 0, 1): (0, 2), (0, 1, 1): (1, 2)}
    for s_mm in scales_mm:
        s = max(0.5, float(s_mm) / step)
        H = np.empty((idx[0].size, 3, 3), np.float32)
        for order, (i, j) in orders.items():
            d = ndimage.gaussian_filter(img, s, order=order, mode="nearest")
            d = d[idx] * (s * s)                       # scale-normalised
            H[:, i, j] = d
            H[:, j, i] = d
        lam = np.linalg.eigvalsh(H)                    # ascending (signed)
        order_abs = np.argsort(np.abs(lam), axis=1)
        lam = np.take_along_axis(lam, order_abs, axis=1)   # |l1|<=|l2|<=|l3|
        l1, l2, l3 = lam[:, 0], lam[:, 1], lam[:, 2]
        bright = (l2 < 0) & (l3 < 0)
        a2 = np.abs(l2); a3 = np.abs(l3) + 1e-6
        Ra = a2 / a3
        Rb = np.abs(l1) / np.sqrt(a2 * a3 + 1e-12)
        S = np.sqrt(l1 * l1 + l2 * l2 + l3 * l3)
        c = 0.5 * float(S.max()) + 1e-6
        v = ((1 - np.exp(-(Ra * Ra) / alpha2)) * np.exp(-(Rb * Rb) / beta2)
             * (1 - np.exp(-(S * S) / (2 * c * c))))
        v = np.where(bright, v, 0.0).astype(np.float32)
        best = np.maximum(best, v)
    m = float(best.max())
    if m > 0:
        best /= m
    V[idx] = best
    return V


def vessel_likelihood(img, step, prm: TraceParams):
    """L in [0, 1] per ROI voxel: tubularity × HU gate (contrast, not calcium)."""
    sm = ndimage.gaussian_filter(img, max(0.5, 0.5 / step), mode="nearest")
    gate = (_sigmoid((sm - prm.hu_lo) / prm.hu_lo_width)
            * _sigmoid((prm.hu_hi - sm) / prm.hu_hi_width)).astype(np.float32)
    cand = gate > 0.05
    V = _frangi_bright(img, step, prm.scales_mm, cand)
    # Tubularity is weak at the very centre of the largest vessels (LM) at small
    # scales; blend a little pure-HU evidence so the lumen stays cheap there.
    return np.clip(gate * (0.8 * V + 0.2 * gate * (V > 0.02)), 0.0, 1.0), sm


# -------------------------------------------------------------------- min path
def _snap(L, k, radius_vox):
    """Move voxel index k (z, y, x) to the max-L voxel within radius."""
    r = int(max(0, round(radius_vox)))
    z, y, x = k
    zs = slice(max(0, z - r), z + r + 1)
    ys = slice(max(0, y - r), y + r + 1)
    xs = slice(max(0, x - r), x + r + 1)
    sub = L[zs, ys, xs]
    gz, gy, gx = np.mgrid[zs, ys, xs]
    inside = (gz - z) ** 2 + (gy - y) ** 2 + (gx - x) ** 2 <= r * r
    w = np.where(inside, sub, -1.0)
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


def _smooth(P, sigma_samples):
    if len(P) < 5 or sigma_samples <= 0:
        return P
    Q = ndimage.gaussian_filter1d(P, sigma_samples, axis=0, mode="nearest")
    Q[0] = P[0]; Q[-1] = P[-1]                  # keep the clicked end points
    return Q


def trace_vessel(vol, dims, points_mm, prm: TraceParams | None = None
                 ) -> TraceResult:
    """Trace a centreline through the contrast lumen along ``points_mm``
    (start, via…, end; world mm). See the module docstring."""
    prm = prm or TraceParams()
    pts = [np.asarray(p, float) for p in points_mm]
    if len(pts) < 2:
        return TraceResult(np.zeros((0, 3)), np.zeros((0, 3)), 0.0,
                           ok=False, message="need at least 2 points")
    dense, Lprof, costs = [], [], []
    ext = _vol_extent(dims, vol.shape)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        grow = 1.0
        for _attempt in range(prm.roi_retries + 1):
            origin, step, shape = _roi_grid(p0, p1, dims, vol.shape, prm, grow)
            img = _sample_roi(vol, dims, origin, step, shape)
            L, sm = vessel_likelihood(img, step, prm)
            cost = (1.0 / (prm.eps + L)).astype(np.float32)

            def to_idx(p, origin=origin, step=step, shape=shape):
                k = np.round((p - origin) / step).astype(int)[::-1]
                return tuple(int(np.clip(k[i], 0, shape[i] - 1))
                             for i in range(3))
            a = _snap(L, to_idx(p0), prm.snap_radius_mm / step)
            b = _snap(L, to_idx(p1), prm.snap_radius_mm / step)
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
    Lall = np.concatenate(Lprof)
    # Gaps: runs of low likelihood along the raw path (arc length in mm).
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    gaps, run0 = [], None
    for i, low in enumerate(Lall < prm.gap_L):
        if low and run0 is None:
            run0 = i
        if (not low or i == len(Lall) - 1) and run0 is not None:
            j = i if low else i - 1
            if s[j] - s[run0] >= prm.gap_min_mm:
                gaps.append((float(s[run0]), float(s[j])))
            run0 = None
    Pd = _resample(P, prm.sample_mm)
    Pd = _smooth(Pd, prm.smooth_mm / prm.sample_mm)
    Pd = _resample(Pd, prm.sample_mm)
    length = float(np.sum(np.linalg.norm(np.diff(Pd, axis=0), axis=1)))
    ctrl = _resample(Pd, prm.ctrl_step_mm)
    tail = float(np.linalg.norm(ctrl[-1] - Pd[-1]))
    if tail > 1e-6:
        if tail < 0.5 * prm.ctrl_step_mm and len(ctrl) >= 2:
            ctrl[-1] = Pd[-1]                    # short remainder: move last
        else:
            ctrl = np.vstack([ctrl, Pd[-1]])
    return TraceResult(Pd, ctrl, length, gaps=gaps, seg_costs=costs, ok=True)
