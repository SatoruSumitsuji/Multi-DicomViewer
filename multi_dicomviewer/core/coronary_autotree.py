"""全冠動脈 一括描出 — extract the whole coronary tree at once (user 2026-11-28).

The user sets the ostia by hand (LM, or LAD + LCX when there is no LM, and
RCA); everything else is automatic:

1. Search space. The LV Epi surface is the map: coronary arteries run just
   OUTSIDE it. A signed distance to the Epi (mm, negative inside) gives the
   Epi shell  −3 … +8 mm  (−3: a myocardial bridge dives under the surface;
   the cavity is never searched). The RCA (#1-3 and the proximal 4 / 16)
   runs outside the LV Epi in the right AV groove, so a corridor around the
   LV base is searched too, and a ball round every ostium (the LM / RCA
   leave the aorta outside the shell).
2. Lumen. Contrast lumen = HU above the half-maximum between the
   myocardium and the ostium blood (calcium / stents count as lumen, so they
   never break a line). Thick bright bodies (aorta, chambers, atria: a
   lumen radius > 3.5 mm) are removed with a margin, leaving the tubes.
3. Connected = an artery. The tube network connected to an ostium is taken
   — lumen continuity first (that is all the RCA corridor relies on). A
   tube piece separated by a short gap (≤ 4 mm: a stenosis / a calcified
   or blurred segment) whose gap is not fat is joined to it.
4. Centrelines (TEASAR): a path from the ostium to the farthest point of
   the network through the lumen centre (a centre-seeking cost), then
   again to the farthest point not covered yet, and so on; each new path
   ends on the path it meets → a branch at that junction. Very short side
   paths (< min_branch_mm) are dropped (wall bumps).
5. The LM is cut at its first big bifurcation into LM + the two trunks;
   the trunk ending nearer the apex is the LAD, the other the LCX.
6. 未接続血管: a tube in the Epi shell that is long enough (≥ 15 mm), bright
   like an artery and NOT connected to an ostium (an occluded vessel filled
   by collaterals) is returned as an unconnected line (connected False).
   Veins share the grooves but are darker in the arterial phase, so they
   fail the brightness test; a vein that passes it still never becomes part
   of the tree (it is not connected to the aorta) — it is listed as 未接続
   for the user to delete.

Volumes are ``vol[z, y, x]``, ``dims = (sx, sy, sz)`` mm, world = voxel ·
spacing (origin 0) — the convention of ``viewers/cpr_mixin.py``. No patient
data in the log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

#: the ostium roles the user sets (in this order)
OSTIUM_ROLES = ("LM", "LAD", "LCX", "RCA")


@dataclass
class AutoTreeParams:
    h_mm: float = 0.6               # working grid (isotropic) mm
    margin_mm: float = 15.0         # ROI margin around the Epi / ostia
    shell_in_mm: float = 3.0        # Epi shell: this far INSIDE (bridges)
    shell_out_mm: float = 8.0       # … and this far outside
    ostium_ball_mm: float = 15.0    # searched round every ostium
    rca_below_base_mm: float = 45.0  # RCA corridor: from the LV base toward
    rca_above_base_mm: float = 20.0  # the apex / toward the atria
    rca_radial_mm: float = 45.0     # … up to this far outside the Epi radius
    no_epi_box_mm: float = 80.0     # no Epi: the box round the ostia
    lumen_frac: float = 0.45        # threshold: bg + frac · (ostium − bg)
    min_thr_hu: float = 120.0
    big_radius_mm: float = 3.5      # thicker than this → aorta / chamber
    big_margin_mm: float = 1.0
    seed_mm: float = 6.0            # ostium → nearest tube voxel
    gap_mm: float = 4.0             # a short gap bridged …
    gap_floor_frac: float = 0.10    # … when no sample in it is below
    #                                 bg + frac · (ostium − bg) (= not fat)
    min_branch_mm: float = 8.0      # a side path shorter than this is dropped
    inval_scale: float = 2.0        # TEASAR: covered = within scale·radius
    inval_add_mm: float = 1.5       # + this of a path
    max_paths: int = 300
    lm_max_mm: float = 25.0         # LM bifurcation within this of the ostium
    trunk_min_mm: float = 20.0      # … onto a branch at least this long
    unc_min_mm: float = 15.0        # 未接続: a line at least this long
    unc_shell_frac: float = 0.6     # … mostly in the Epi shell
    unc_bright_frac: float = 0.55   # … median HU ≥ bg + frac · (ostium − bg)
    unc_max_radius_mm: float = 2.5  # … and thin (median lumen radius)
    unc_max: int = 40
    ctrl_step_mm: float = 2.5       # edit points along a line
    smooth_n: int = 5               # moving-average window (nodes)


@dataclass
class AutoVessel:
    key: int
    role: str                       # LM / LAD / LCX / RCA, or "branch"
    parent: int | None              # key of the parent line (None: root/loose)
    points: np.ndarray              # (M, 3) world mm, proximal → distal
    ctrl: np.ndarray                # edit points (thinned)
    connected: bool = True          # False = 未接続血管
    system: str = ""                # LCA / RCA / "" (unconnected)
    length_mm: float = 0.0


@dataclass
class AutoTreeResult:
    vessels: list = field(default_factory=list)
    stop: str = "ok"                # ok / cancel / no-seed / empty
    log: dict = field(default_factory=dict)


class _Cancelled(Exception):
    pass


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _resample(arr, spacing_zyx, h, order=1, offset_vox=(0.0, 0.0, 0.0),
              out_shape=None, cval=0.0):
    """*arr* (zyx, its own spacing) onto an isotropic *h* grid. Output index
    k ↔ input coordinate offset + k·h/s (exact, grid aligned)."""
    m = np.array([h / s for s in spacing_zyx], float)
    if out_shape is None:
        out_shape = tuple(int(np.floor((n - 1) * s / h)) + 1
                          for n, s in zip(arr.shape, spacing_zyx))
    return ndimage.affine_transform(
        np.asarray(arr, np.float32), m, offset=np.asarray(offset_vox, float),
        output_shape=out_shape, order=order, mode="constant", cval=cval)


_OFFS = np.array([(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1)
                  for dx in (-1, 0, 1)
                  if (dz, dy, dx) > (0, 0, 0)], int)       # 13 half-offsets


def _graph(mask, Ed, h):
    """26-neighbour graph of the True voxels: (coords (n,3) zyx, plain-length
    matrix, centre-seeking (TEASAR) matrix)."""
    shape = mask.shape
    idx = np.flatnonzero(mask)
    n = len(idx)
    lut = np.full(mask.size, -1, np.int64)
    lut[idx] = np.arange(n)
    coords = np.stack(np.unravel_index(idx, shape), axis=1)
    dbf = Ed.ravel()[idx].astype(float)
    mx = max(float(dbf.max()) if n else 1.0, h)
    pen = 5000.0 * (1.0 - np.clip(dbf / mx, 0.0, 1.0)) ** 16
    R, C, L = [], [], []
    for off in _OFFS:
        nb = coords + off
        ok = np.all((nb >= 0) & (nb < np.array(shape)), axis=1)
        src = np.nonzero(ok)[0]
        j = lut[np.ravel_multi_index(nb[ok].T, shape)]
        good = j >= 0
        R.append(src[good])
        C.append(j[good])
        L.append(np.full(int(good.sum()), h * float(np.linalg.norm(off))))
    R = np.concatenate(R) if R else np.zeros(0, int)
    C = np.concatenate(C) if C else np.zeros(0, int)
    L = np.concatenate(L) if L else np.zeros(0)
    W = L * (1.0 + 0.5 * (pen[R] + pen[C]))
    Gl = coo_matrix((L, (R, C)), shape=(n, n)).tocsr()
    Gp = coo_matrix((W, (R, C)), shape=(n, n)).tocsr()
    return coords, dbf, Gl, Gp


def _teasar(mask, Ed, h, roots, prm, check):
    """Centreline paths of the voxel set *mask* from the root voxels
    (list of zyx). Returns (coords_zyx, paths) with paths = [{nodes
    (indices root/junction → tip), parent (path id or None), source (root
    index), length_mm}]."""
    coords, dbf, Gl, Gp = _graph(mask, Ed, h)
    n = len(coords)
    if n == 0:
        return coords, []
    lut = {tuple(c): i for i, c in enumerate(coords[:0])}  # (unused)
    del lut
    sidx = []
    tree = cKDTree(coords)
    for r in roots:
        _d, i = tree.query(np.asarray(r, float))
        sidx.append(int(i))
    sidx = list(dict.fromkeys(sidx))
    De = dijkstra(Gl, directed=False, indices=sidx, min_only=True)
    _Dp, pred, srcs = dijkstra(Gp, directed=False, indices=sidx,
                               min_only=True, return_predecessors=True)
    alive = np.isfinite(De)
    on_path = np.full(n, -1, np.int64)
    pos = np.full(n, -1, np.int64)
    pts_mm = coords * h
    kdt = cKDTree(pts_mm)
    paths = []
    tries = 0
    while tries < prm.max_paths * 4 and len(paths) < prm.max_paths:
        tries += 1
        check()
        cand = np.nonzero(alive)[0]
        if len(cand) == 0:
            break
        tgt = int(cand[np.argmax(De[cand])])
        seq = [tgt]
        cur = tgt
        while on_path[cur] < 0 and pred[cur] >= 0:
            cur = int(pred[cur])
            seq.append(cur)
        seq.reverse()
        seq = np.asarray(seq, np.int64)
        parent = int(on_path[seq[0]]) if on_path[seq[0]] >= 0 else None
        seg = np.diff(pts_mm[seq], axis=0)
        length = float(np.linalg.norm(seg, axis=1).sum()) if len(seq) > 1 else 0.0
        keep = (parent is None and len(seq) > 1) or \
            (parent is not None and length >= prm.min_branch_mm)
        rad = prm.inval_scale * dbf[seq] + prm.inval_add_mm
        for p_, r_ in zip(pts_mm[seq], rad):
            alive[kdt.query_ball_point(p_, float(r_))] = False
        alive[seq] = False
        if keep:
            pid = len(paths)
            own = seq[1:] if parent is not None else seq
            on_path[own] = pid
            pos[own] = np.arange(len(own)) + (1 if parent is not None else 0)
            paths.append({"nodes": seq, "parent": parent,
                          "source": int(srcs[seq[0]]) if parent is None
                          else None,
                          "junction": (int(pos[seq[0]]) if parent is not None
                                       else None),
                          "length_mm": length})
    # a root path's own nodes: pos along it; a branch path's junction index
    # is the parent's node position at its first node
    return coords, paths, sidx


def _smooth(P, n):
    P = np.asarray(P, float)
    if len(P) < 3 or n < 2:
        return P
    k = min(n, len(P))
    pad = np.vstack([np.repeat(P[:1], k, 0), P, np.repeat(P[-1:], k, 0)])
    ker = np.ones(k) / k
    S = np.stack([np.convolve(pad[:, a], ker, mode="same")
                  for a in range(3)], axis=1)[k:-k]
    S[0], S[-1] = P[0], P[-1]
    return S


def _thin(P, step):
    P = np.asarray(P, float)
    if len(P) < 3:
        return P.copy()
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    marks = np.arange(0.0, d[-1], step)
    keep = sorted(set(int(np.argmin(np.abs(d - m))) for m in marks)
                  | {len(P) - 1})
    return P[keep].copy()


def extract_coronary_tree(vol, dims, ostia, epi=None, epi_origin_mm=None,
                          apex=None, axis_dir=None,
                          prm: AutoTreeParams | None = None,
                          progress=None, cancel=None) -> AutoTreeResult:
    """Extract the coronary tree of *vol* from the hand-set *ostia*
    (``[(role, xyz mm), …]``, role in OSTIUM_ROLES). *epi* = the LV Epi
    region (bool zyx, the CT's spacing) whose voxel 0 sits at
    *epi_origin_mm* (x, y, z); None → lumen continuity only (a box round
    the ostia). *apex* / *axis_dir* (mm / unit) when known. *progress*
    (text) is called between the stages; *cancel*() → stop."""
    prm = prm or AutoTreeParams()
    t0 = time.time()
    out = AutoTreeResult()
    log = out.log

    def say(s):
        if progress is not None:
            progress(s)

    def check():
        if cancel is not None and cancel():
            raise _Cancelled()

    try:
        return _extract(vol, dims, ostia, epi, epi_origin_mm, apex, axis_dir,
                        prm, say, check, out, log, t0)
    except _Cancelled:
        out.vessels = []
        out.stop = "cancel"
        return out


def _extract(vol, dims, ostia, epi, epi_origin_mm, apex, axis_dir, prm,
             say, check, out, log, t0):
    vol = np.asarray(vol)
    sx, sy, sz = (float(d) for d in dims)
    sp_xyz = np.array([sx, sy, sz])
    h = float(prm.h_mm)
    ost = [(str(r), np.asarray(p, float).reshape(3)) for r, p in ostia
           if r in OSTIUM_ROLES and p is not None]
    if not ost:
        out.stop = "no-seed"
        return out
    O = np.array([p for _r, p in ost])
    # ---- ROI (world mm) ---------------------------------------------------
    have_epi = epi is not None and np.asarray(epi).any()
    if have_epi:
        epi = np.asarray(epi, bool)
        eo = np.asarray(epi_origin_mm if epi_origin_mm is not None
                        else (0, 0, 0), float)
        ez, ey, ex = np.nonzero(epi[::2, ::2, ::2])
        epts = np.stack([ex * 2 * sx, ey * 2 * sy, ez * 2 * sz], 1) + eo
        lo = np.minimum(epts.min(0), O.min(0) - 20.0) - prm.margin_mm
        hi = np.maximum(epts.max(0), O.max(0) + 20.0) + prm.margin_mm
    else:
        epts = None
        lo = O.min(0) - prm.no_epi_box_mm
        hi = O.max(0) + prm.no_epi_box_mm
    nz, ny, nx = vol.shape
    i0 = np.maximum(np.floor(lo / sp_xyz).astype(int), 0)
    i1 = np.minimum(np.ceil(hi / sp_xyz).astype(int) + 1, [nx, ny, nz])
    crop = vol[i0[2]:i1[2], i0[1]:i1[1], i0[0]:i1[0]]
    roi0 = i0 * sp_xyz                                    # world of ROI k=0
    say("ROI")
    V = _resample(crop, (sz, sy, sx), h, order=1, cval=-1000.0)
    shape = V.shape
    log["roi_shape"] = list(shape)
    check()

    def w2k(p):                                          # world → ROI zyx
        q = (np.asarray(p, float) - roi0) / h
        return np.array([q[2], q[1], q[0]])

    def k2w(k):                                          # ROI zyx → world
        k = np.asarray(k, float)
        return np.stack([k[..., 2], k[..., 1], k[..., 0]], -1) * h + roi0

    # ---- Epi signed distance ---------------------------------------------
    sd = None
    if have_epi:
        say("Epi")
        off = ((roi0 - eo) / sp_xyz)[::-1]                # epi zyx index of k=0
        E = _resample(epi, (sz, sy, sx), h, order=0, offset_vox=off,
                      out_shape=shape) > 0.5
        sd = (ndimage.distance_transform_edt(~E) * h
              - ndimage.distance_transform_edt(E) * h).astype(np.float32)
        del E
        check()
    # ---- intensities ------------------------------------------------------
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    vals = []
    for _r, p in ost:
        k = w2k(p)
        r = 1.5 / h
        lo_ = np.maximum(np.floor(k - r).astype(int), 0)
        hi_ = np.minimum(np.ceil(k + r).astype(int) + 1, shape)
        sub = V[lo_[0]:hi_[0], lo_[1]:hi_[1], lo_[2]:hi_[2]]
        if sub.size:
            vals.append(float(np.percentile(sub, 75)))
    I_ost = float(np.median(vals)) if vals else 350.0
    if sd is not None:
        band = V[(sd > -5.0) & (sd < -1.5)]
        bg = float(np.percentile(band, 30)) if band.size > 50 else 80.0
    else:
        bg = 80.0
    if I_ost - bg < 80.0:
        I_ost = bg + 80.0
    thr = max(bg + prm.lumen_frac * (I_ost - bg), prm.min_thr_hu)
    log.update(ostium_hu=round(I_ost, 1), bg_hu=round(bg, 1),
               thr_hu=round(thr, 1), epi=bool(have_epi))
    # ---- search space -----------------------------------------------------
    say("space")
    if sd is not None:
        S = (sd >= -prm.shell_in_mm) & (sd <= prm.shell_out_mm)
        # the LV axis (apex → base)
        c = epts.mean(0)
        if axis_dir is not None and np.linalg.norm(axis_dir) > 0:
            a = _unit(axis_dir)
        else:
            u, _s, vt = np.linalg.svd(epts[::7] - c, full_matrices=False)
            a = _unit(vt[0])
        if np.dot(O.mean(0) - c, a) < 0:                 # → toward the ostia
            a = -a
        ap = (np.asarray(apex, float) if apex is not None
              else epts[np.argmin((epts - c) @ a)])
        hgt = (epts - ap) @ a
        hb = float(hgt.max())                            # the LV base level
        rad = np.linalg.norm((epts - ap) - np.outer(hgt, a), axis=1)
        rmax = float(np.percentile(rad, 95))
        log.update(base_mm=round(hb, 1), epi_radius_mm=round(rmax, 1))
        if any(r == "RCA" for r, _p in ost):
            W = k2w(np.stack(np.broadcast_arrays(zz, yy, xx), -1)
                    .reshape(-1, 3)).reshape(shape + (3,))
            Wr = W - ap
            hh = Wr @ a
            rr = np.linalg.norm(Wr - hh[..., None] * a, axis=-1)
            corr = ((hh >= hb - prm.rca_below_base_mm)
                    & (hh <= hb + prm.rca_above_base_mm)
                    & (rr <= rmax + prm.rca_radial_mm) & (sd > 0))
            S |= corr
            del W, Wr, hh, rr, corr
        for _r, p in ost:
            k = w2k(p)
            S |= ((zz - k[0]) ** 2 + (yy - k[1]) ** 2 + (xx - k[2]) ** 2
                  <= (prm.ostium_ball_mm / h) ** 2)
    else:
        S = np.ones(shape, bool)
        ap, a = None, None
    check()
    # ---- tubes ------------------------------------------------------------
    say("lumen")
    L = V >= thr
    Ed = (ndimage.distance_transform_edt(L) * h).astype(np.float32)
    core = Ed > prm.big_radius_mm
    if core.any():
        near_big = ndimage.distance_transform_edt(~core) * h \
            <= prm.big_radius_mm + prm.big_margin_mm
    else:
        near_big = np.zeros(shape, bool)
    T = L & S & ~near_big
    del near_big, core
    check()
    st = np.ones((3, 3, 3), bool)
    lab, nlab = ndimage.label(T, structure=st)
    log["components"] = int(nlab)
    if nlab == 0:
        out.stop = "empty"
        return out
    # ---- seeds ------------------------------------------------------------
    seeds = []                                           # (role, zyx, label)
    for r, p in ost:
        k = w2k(p)
        rs = prm.seed_mm / h
        lo_ = np.maximum(np.floor(k - rs).astype(int), 0)
        hi_ = np.minimum(np.ceil(k + rs).astype(int) + 1, shape)
        sub = lab[lo_[0]:hi_[0], lo_[1]:hi_[1], lo_[2]:hi_[2]]
        nzi = np.argwhere(sub > 0)
        if len(nzi) == 0:
            log.setdefault("seed_missing", []).append(r)
            continue
        g = nzi + lo_
        d = np.linalg.norm(g - k, axis=1) * h
        j = int(np.argmin(d))
        if d[j] > prm.seed_mm:
            log.setdefault("seed_missing", []).append(r)
            continue
        seeds.append((r, g[j], int(lab[tuple(g[j])])))
    if not seeds:
        out.stop = "no-seed"
        return out
    conn = {s[2] for s in seeds}
    # ---- bridge short gaps (stenosis / calcium blur) ----------------------
    say("gaps")
    floor_hu = bg + prm.gap_floor_frac * (I_ost - bg)
    sizes = np.bincount(lab.ravel())
    min_vox = max(int(round(3.0 / h ** 3)), 4)
    bridges = 0
    for _it in range(4):
        check()
        C = np.isin(lab, list(conn))
        dC, ind = ndimage.distance_transform_edt(~C, return_indices=True)
        dC *= h
        merged = False
        for lb in range(1, nlab + 1):
            if lb in conn or sizes[lb] < min_vox:
                continue
            m = lab == lb
            dm = np.where(m, dC, np.inf)
            j = np.unravel_index(int(np.argmin(dm)), shape)
            if not np.isfinite(dm[j]) or dm[j] > prm.gap_mm:
                continue
            q = np.array([ind[0][j], ind[1][j], ind[2][j]], float)
            pj = np.array(j, float)
            nseg = max(int(np.ceil(np.linalg.norm(q - pj) * 3)), 2)
            line = pj + np.linspace(0, 1, nseg)[:, None] * (q - pj)
            hu = ndimage.map_coordinates(V, line.T, order=1)
            if float(hu.min()) < floor_hu:
                continue                                  # fat between → no
            li = np.unique(np.rint(line).astype(int), axis=0)
            T[tuple(li.T)] = True
            lab[tuple(li.T)] = lb
            conn.add(lb)
            bridges += 1
            merged = True
        del dC, ind
        if not merged:
            break
    log["bridged_gaps"] = bridges
    # ---- centrelines of the connected network -----------------------------
    say("centreline")
    M = np.isin(lab, list(conn))
    coords, paths, sidx = _teasar(M, Ed, h, [s[1] for s in seeds], prm, check)
    src_role = {}
    kd = cKDTree(coords) if len(coords) else None
    for r, g, _l in seeds:
        if kd is not None:
            src_role.setdefault(int(kd.query(g.astype(float))[1]), r)
    vessels = _to_vessels(coords, paths, src_role, h, k2w, prm, ap, a)
    # ---- 未接続血管 ------------------------------------------------------
    if sd is not None:
        say("unconnected")
        bright = bg + prm.unc_bright_frac * (I_ost - bg)
        shell = (sd >= -prm.shell_in_mm) & (sd <= prm.shell_out_mm)
        cands = []
        objs = ndimage.find_objects(lab)
        for lb in range(1, nlab + 1):
            if lb in conn or sizes[lb] < min_vox or objs[lb - 1] is None:
                continue
            sl = objs[lb - 1]
            m = lab[sl] == lb
            ext = np.array([s.stop - s.start for s in sl]) * h
            if float(np.linalg.norm(ext)) < prm.unc_min_mm:
                continue
            if float(shell[sl][m].mean()) < prm.unc_shell_frac:
                continue
            if float(np.median(V[sl][m])) < bright:
                continue
            if float(np.median(Ed[sl][m])) > prm.unc_max_radius_mm:
                continue
            cands.append((float(np.linalg.norm(ext)), lb))
        cands.sort(reverse=True)
        n_unc = 0
        for _e, lb in cands[:prm.unc_max]:
            check()
            m = lab == lb
            g = np.argwhere(m)
            gw = k2w(g)
            if a is not None:
                rt = g[int(np.argmax((gw - ap) @ a))]      # toward the base
            else:
                rt = g[int(np.argmin(np.linalg.norm(gw - O.mean(0), axis=1)))]
            cc, pp, _s = _teasar(m, Ed, h, [rt], prm, check)
            if not pp or pp[0]["length_mm"] < prm.unc_min_mm:
                continue
            vs = _to_vessels(cc, pp, {}, h, k2w, prm, None, None,
                             key0=len(vessels), connected=False)
            vessels.extend(vs)
            n_unc += 1
        log["unconnected"] = n_unc
    out.vessels = vessels
    log["lines"] = len(vessels)
    log["seconds"] = round(time.time() - t0, 1)
    out.stop = "ok" if vessels else "empty"
    return out


def _to_vessels(coords, paths, src_role, h, k2w, prm, apex, a, key0=0,
                connected=True):
    """TEASAR paths → AutoVessels (parents first); an LM root is cut at its
    first big bifurcation into LM + LAD + LCX."""
    if not paths:
        return []
    P = [k2w(coords[p["nodes"]]) for p in paths]
    par = [p["parent"] for p in paths]
    junc = [p["junction"] for p in paths]
    role = []
    for p in paths:
        if p["parent"] is None:
            role.append(src_role.get(p["source"], "branch") if connected
                        else "branch")
        else:
            role.append("branch")
    # ---- LM → LM + two trunks -------------------------------------------
    for i in range(len(paths)):
        if role[i] != "LM":
            continue
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P[i], axis=0), axis=1))]
        best = None
        for c in range(len(paths)):
            if par[c] != i or junc[c] is None:
                continue
            jm = d[min(junc[c], len(d) - 1)]
            if jm > prm.lm_max_mm or paths[c]["length_mm"] < prm.trunk_min_mm:
                continue
            if best is None or paths[c]["length_mm"] > paths[best]["length_mm"]:
                best = c
        if best is None:
            continue
        j = int(junc[best])
        rest = P[i][j:]
        new = len(P)
        P.append(rest)
        par.append(None)
        junc.append(None)
        role.append("trunk")
        paths.append({"length_mm": float(d[-1] - d[j])})
        P[i] = P[i][:j + 1]
        for c in range(len(paths) - 1):
            if par[c] == i and c != best and junc[c] is not None \
                    and junc[c] > j:
                par[c] = new
                junc[c] = junc[c] - j
        par[best] = None
        role[best] = "trunk"
        # the trunk ending nearer the apex = LAD
        t1, t2 = new, best
        if apex is not None:
            lad = t1 if (np.linalg.norm(P[t1][-1] - apex)
                         <= np.linalg.norm(P[t2][-1] - apex)) else t2
        else:
            lad = t1 if paths[t1]["length_mm"] >= paths[t2]["length_mm"] else t2
        role[lad] = "LAD"
        role[t2 if lad == t1 else t1] = "LCX"
        # the trunks hang from the LM end (so they start ON it)
        P[t2] = np.vstack([P[i][-1:], P[t2]]) if np.linalg.norm(
            P[t2][0] - P[i][-1]) > 1e-6 else P[t2]
    # ---- order: parents first; system per root ------------------------------
    order, seen = [], set()

    def visit(k):
        if k in seen:
            return
        if par[k] is not None:
            visit(par[k])
        seen.add(k)
        order.append(k)

    for k in range(len(P)):
        visit(k)
    sysm = {}
    for k in order:
        if role[k] in ("LM", "LAD", "LCX"):
            sysm[k] = "LCA"
        elif role[k] == "RCA":
            sysm[k] = "RCA"
        elif par[k] is not None:
            sysm[k] = sysm.get(par[k], "")
        else:
            sysm[k] = ""
    keymap = {k: key0 + n for n, k in enumerate(order)}
    out = []
    for k in order:
        pts = _smooth(P[k], prm.smooth_n)
        if par[k] is not None:
            pts[0] = P[k][0]                    # start ON the parent line
        L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) \
            if len(pts) > 1 else 0.0
        if len(pts) < 2:
            continue
        out.append(AutoVessel(
            key=keymap[k],
            role=(role[k] if role[k] in OSTIUM_ROLES else "branch"),
            parent=(keymap[par[k]] if par[k] is not None else None),
            points=pts, ctrl=_thin(pts, prm.ctrl_step_mm),
            connected=connected, system=(sysm[k] if connected else ""),
            length_mm=L))
    return out
