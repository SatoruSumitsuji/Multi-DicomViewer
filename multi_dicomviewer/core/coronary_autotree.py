"""全冠動脈 一括描出 — extract the whole coronary tree at once (user 2026-11-28).

The user sets the ostia by hand (LM, or LAD + LCX when there is no LM, and
RCA); everything else is automatic:

1. Search space, one per system. The LV Epi surface is the map: coronary
   arteries run just OUTSIDE it. A signed distance to the Epi (mm, negative
   inside) gives the Epi shell  −3 … +8 mm  (−3: a myocardial bridge dives
   under the surface; the cavity is never searched). Above the base plane
   (the MV cap the Epi is cut at) only the AV-groove ring is kept — not the
   LA floor / the annulus. The RCA (#1-3 and the proximal 4 / 16) runs
   outside the LV Epi in the right AV groove, so the RCA (only) also gets a
   corridor: from its ostium's side round the right (away from the LCA,
   230°), from 60 mm below to 20 mm above the base, near the heart surface
   (≤ 15 mm from a chamber / the aorta, or ≤ 20 mm from the LV Epi). Each
   system adds a 10 mm ball round its own ostia. (user 2026-11-28: the
   first version's corridor was a full ring open to the LCA too, and the
   shell covered the cap — lines in the LA / LAA / chest wall.)
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
   paths (< min_branch_mm) are dropped (wall bumps). Each new path is
   JUDGED first: a branch thicker than its parent, clearly darker than it,
   or plate-like (Hessian: an atrial wall rim / trabeculae) is dropped; a
   path that turns lastingly darker without narrowing (a stenosis narrows)
   or wider + darker is cut there (a vein met at a contact — the GCV
   crossing the LCX). What is reached THROUGH a rejected part is taken out
   of the search, so a vein never comes back as another path.
4b. Tips that still are thick (a vessel cut where it touches the RA / RV,
   a gap longer than the bridged one, a faint stretch) are continued with
   the CPR Auto-Track (core.cpr_autoextend, trunk only) while they stay in
   their system's space and off the other lines.
5. The LM is cut at its first big bifurcation into LM + the two trunks;
   the trunk ending nearer the apex is the LAD, the other the LCX.
6. 未接続血管: a tube in the Epi shell that is long enough (≥ 20 mm), bright
   like an artery, thin, NOT connected to an ostium (an occluded vessel
   filled by collaterals), running ALONG the surface (≥ 80 % in the shell,
   median ≤ 5 mm outside the Epi, its distance to the Epi steady) and not
   on an extracted line is returned as an unconnected line (connected
   False).
   Veins share the grooves but are darker in the arterial phase, so they
   fail the brightness test; a vein that passes it still never becomes part
   of the tree (it is not connected to the aorta) — it is listed as 未接続
   for the user to delete.

Volumes are ``vol[z, y, x]``, ``dims = (sx, sy, sz)`` mm, world = voxel ·
spacing (origin 0) — the convention of ``viewers/cpr_mixin.py``. No patient
data in the log (the viewer saves it as autotree_log.json next to the
lines, for tuning on real cases).
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
    ostium_ball_mm: float = 10.0    # searched round every ostium (its
    #                                 own system only)
    cap_tol_mm: float = 1.0         # above the LV base plane (+ this) …
    ring_in_mm: float = 6.0         # … only the AV-groove ring is searched:
    #                                 radius ≥ the base Epi radius − this
    rca_below_base_mm: float = 60.0  # RCA corridor: from the LV base toward
    rca_above_base_mm: float = 20.0  # the apex / toward the atria
    rca_sector_deg: float = 230.0   # … from the RCA ostium's side, away from
    rca_sector_back_deg: float = 30.0  # the LCA (+ this back)
    rca_core_mm: float = 15.0       # … within this of a chamber / the aorta
    rca_epi_mm: float = 20.0        # … or of the LV Epi (the heart surface)
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
    unc_min_mm: float = 20.0        # 未接続: a line at least this long
    unc_shell_frac: float = 0.8     # … mostly in the Epi shell
    unc_max_sd_mm: float = 5.0      # … its median distance outside the Epi
    unc_sd_std_mm: float = 2.5      # … along the surface (not leaving it)
    unc_near_line_mm: float = 2.0   # … not on an extracted line (≥ 30 %)
    unc_bright_frac: float = 0.55   # … median HU ≥ bg + frac · (ostium − bg)
    unc_max_radius_mm: float = 2.5  # … and thin (median lumen radius)
    unc_max: int = 40
    # artery / vein at a contact (judged while the centrelines are made;
    # a rejected part is cut off with everything reached through it)
    vein_win_mm: float = 5.0        # compare the 5 mm before / after
    vein_hu_drop: float = 0.22      # darker by this · contrast, not narrower
    vein_widen: float = 1.4         # … or wider by this and darker by
    vein_widen_hu: float = 0.12     # this · contrast
    vein_min_r_mm: float = 1.2
    vein_parent_ratio: float = 1.15  # a branch thicker than its parent
    plate_sigma_mm: float = 1.0     # Hessian: a plate (atrial wall rim,
    plate_ratio: float = 0.25       # trabeculae) |l2| < ratio · |l3|
    plate_frac: float = 0.3         # … on this much of a branch → dropped
    plate_nb_frac: float = 0.35     # … or of the lumen round it (≤ 2 mm)
    # tips continued with Auto-Track (cpr_autoextend, trunk only)
    ext_min_r_mm: float = 0.9       # a tip still this thick
    ext_min_len_mm: float = 20.0    # a branch at least this long
    ext_max_mm: float = 100.0       # per tip
    ext_max_tips: int = 30
    ext_near_mm: float = 1.5        # stop on another line
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


def _subtree(pred, n):
    """The shortest-path tree as a child CSR graph (for cutting off what is
    reached THROUGH a node)."""
    from scipy.sparse import csr_matrix
    c = np.nonzero(pred >= 0)[0]
    return csr_matrix((np.ones(len(c), np.int8), (pred[c], c)), shape=(n, n))


def _teasar(mask, Ed, h, roots, prm, check, judge=None):
    """Centreline paths of the voxel set *mask* from the root voxels
    (list of zyx). Returns (coords_zyx, paths, root idx) with paths =
    [{nodes (indices root/junction → tip), parent (path id or None), source
    (root index), length_mm}]. *judge*(seq, coords, parent_nodes, jpos) →
    ("keep" | "cut" | "drop", index, reason): a cut path keeps seq[:index+1]
    and a dropped one nothing; what is reached through the rejected part is
    taken out of the search (a vein met at a contact, with its whole
    tree), without covering the neighbours."""
    from scipy.sparse.csgraph import breadth_first_order
    coords, dbf, Gl, Gp = _graph(mask, Ed, h)
    n = len(coords)
    if n == 0:
        return coords, [], []
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
    tree_g = _subtree(pred, n) if judge is not None else None
    judged = {}

    def kill_from(node):
        sub = breadth_first_order(tree_g, int(node), directed=True,
                                  return_predecessors=False)
        alive[sub] = False
        return len(sub)

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
        if judge is not None and len(seq) > 2:
            pn = paths[parent]["nodes"] if parent is not None else None
            jp = int(pos[seq[0]]) if parent is not None else None
            verdict, k, why = judge(seq, coords, pn, jp)
            if verdict == "drop":
                judged[why] = judged.get(why, 0) + 1
                kill_from(seq[1] if parent is not None else seq[0])
                alive[seq] = False
                continue
            if verdict == "cut" and k is not None and 0 < k < len(seq) - 1:
                judged[why] = judged.get(why, 0) + 1
                kill_from(seq[k + 1])
                seq = seq[:k + 1]
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
    if judge is not None:
        _teasar.judged = judged
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
    # ---- lumen + thick bodies (the RCA corridor follows the chambers) -----
    say("lumen")
    L = V >= thr
    Ed = (ndimage.distance_transform_edt(L) * h).astype(np.float32)
    core = Ed > prm.big_radius_mm
    if core.any():
        dcore = (ndimage.distance_transform_edt(~core) * h).astype(np.float32)
    else:
        dcore = np.full(shape, 1e4, np.float32)
    del core
    near_big = dcore <= prm.big_radius_mm + prm.big_margin_mm
    check()
    # ---- search space: one per system -------------------------------------
    say("space")
    sys_of = {r: ("RCA" if r == "RCA" else "LCA") for r, _p in ost}
    balls = {"LCA": np.zeros(shape, bool), "RCA": np.zeros(shape, bool)}
    for r, p in ost:
        k = w2k(p)
        balls[sys_of[r]] |= ((zz - k[0]) ** 2 + (yy - k[1]) ** 2
                             + (xx - k[2]) ** 2
                             <= (prm.ostium_ball_mm / h) ** 2)
    shell = None
    if sd is not None:
        shell = (sd >= -prm.shell_in_mm) & (sd <= prm.shell_out_mm)
        # the LV axis (apex → base)
        c = epts.mean(0)
        if axis_dir is not None and np.linalg.norm(axis_dir) > 0:
            a = _unit(axis_dir)
        else:
            _u, _s, vt = np.linalg.svd(epts[::7] - c, full_matrices=False)
            a = _unit(vt[0])
        if np.dot(O.mean(0) - c, a) < 0:                 # → toward the ostia
            a = -a
        ap = (np.asarray(apex, float) if apex is not None
              else epts[np.argmin((epts - c) @ a)])
        hgt = (epts - ap) @ a
        hb = float(hgt.max())                            # the LV base level
        rad = np.linalg.norm((epts - ap) - np.outer(hgt, a), axis=1)
        top = hgt >= hb - 8.0
        r_ring = float(np.percentile(rad[top] if top.sum() > 20 else rad,
                                     95))                # the base Epi radius
        e1 = _unit(np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9
                            else [0, 1.0, 0]))
        e2 = np.cross(a, e1)
        wx = (roi0[0] + np.arange(shape[2]) * h - ap[0]).astype(np.float32)
        wy = (roi0[1] + np.arange(shape[1]) * h - ap[1]).astype(np.float32)
        wz = (roi0[2] + np.arange(shape[0]) * h - ap[2]).astype(np.float32)

        def sep(v):                                      # (W − apex) · v
            return (wz[:, None, None] * np.float32(v[2])
                    + wy[None, :, None] * np.float32(v[1])
                    + wx[None, None, :] * np.float32(v[0]))

        hh = sep(a)
        u1, u2 = sep(e1), sep(e2)
        rr = np.hypot(u1, u2)
        # the base cap (the MV plane the Epi is cut at): above it only the
        # AV-groove ring is searched — not the LA floor / the annulus
        cap = (hh > hb + prm.cap_tol_mm) & (rr < r_ring - prm.ring_in_mm)
        shell &= ~cap
        log.update(base_mm=round(hb, 1), ring_radius_mm=round(r_ring, 1))
        space = {"LCA": shell | balls["LCA"], "RCA": shell | balls["RCA"]}
        rca_p = [p for r, p in ost if r == "RCA"]
        if rca_p:
            def azof(p):
                q = np.asarray(p, float) - ap
                return float(np.degrees(np.arctan2(q @ e2, q @ e1)))

            az = np.degrees(np.arctan2(u2, u1))
            ph_r = azof(rca_p[0])
            lca_p = [p for r, p in ost if r != "RCA"]
            if lca_p:
                ang = np.radians([azof(p) for p in lca_p])
                ph_l = float(np.degrees(np.arctan2(np.sin(ang).mean(),
                                                   np.cos(ang).mean())))
                sgn = 1.0 if (ph_l - ph_r) % 360.0 >= 180.0 else -1.0
                tt = (sgn * (az - ph_r)) % 360.0
                sect = (tt <= prm.rca_sector_deg) | \
                    (tt >= 360.0 - prm.rca_sector_back_deg)
                del tt
            else:
                sect = np.ones(shape, bool)
            corr = (sect & (sd > 0)
                    & (hh >= hb - prm.rca_below_base_mm)
                    & (hh <= hb + prm.rca_above_base_mm)
                    & ((dcore <= prm.rca_core_mm) | (sd <= prm.rca_epi_mm)))
            log["rca_corridor_ml"] = round(float(corr.sum()) * h ** 3
                                           / 1000.0, 1)
            space["RCA"] |= corr
            del az, sect, corr
        del hh, u1, u2, rr, cap
    else:
        space = {"LCA": np.ones(shape, bool), "RCA": np.ones(shape, bool)}
        ap, a = None, None
    check()
    # ---- tubes connected to the ostia, per system -------------------------
    Tb = L & ~near_big
    del near_big
    st = np.ones((3, 3, 3), bool)
    floor_hu = bg + prm.gap_floor_frac * (I_ost - bg)
    min_vox = max(int(round(3.0 / h ** 3)), 4)
    seeds = []                                           # (role, zyx)
    M = np.zeros(shape, bool)
    bridges = 0
    ncomp = 0
    for sysk in ("LCA", "RCA"):
        mine = [(r, p) for r, p in ost if sys_of[r] == sysk]
        if not mine:
            continue
        T = Tb & space[sysk]
        lab, nlab = ndimage.label(T, structure=st)
        ncomp += int(nlab)
        conn = set()
        for r, p in mine:
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
            seeds.append((r, g[j]))
            conn.add(int(lab[tuple(g[j])]))
        if not conn:
            continue
        # bridge short gaps (stenosis / calcium blur)
        say("gaps")
        sizes = np.bincount(lab.ravel())
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
                    continue                              # fat between → no
                li = np.unique(np.rint(line).astype(int), axis=0)
                lab[tuple(li.T)] = lb
                conn.add(lb)
                bridges += 1
                merged = True
            del dC, ind
            if not merged:
                break
        M |= np.isin(lab, list(conn))
        del lab, T
    log["components"] = ncomp
    log["bridged_gaps"] = bridges
    if not seeds:
        out.stop = "no-seed"
        return out
    # ---- centrelines, judged on the way (veins / plates cut off) ----------
    say("centreline")
    delta = max(I_ost - bg, 1.0)
    wn = max(int(round(prm.vein_win_mm / h)), 3)
    G = ndimage.gaussian_filter(V, max(prm.plate_sigma_mm / h, 0.5))
    gmax = np.array(shape) - 1

    def plate_at(cz):                                    # (n, 3) zyx int
        from multi_dicomviewer.core.cpr_autotrack import _eig3_sym
        z = np.clip(cz[:, 0], 1, gmax[0] - 1)
        y = np.clip(cz[:, 1], 1, gmax[1] - 1)
        x = np.clip(cz[:, 2], 1, gmax[2] - 1)
        c0 = G[z, y, x]
        H = np.empty((len(z), 3, 3), np.float32)
        H[:, 0, 0] = G[z + 1, y, x] - 2 * c0 + G[z - 1, y, x]
        H[:, 1, 1] = G[z, y + 1, x] - 2 * c0 + G[z, y - 1, x]
        H[:, 2, 2] = G[z, y, x + 1] - 2 * c0 + G[z, y, x - 1]
        H[:, 0, 1] = H[:, 1, 0] = 0.25 * (G[z + 1, y + 1, x] - G[z + 1, y - 1, x]
                                          - G[z - 1, y + 1, x] + G[z - 1, y - 1, x])
        H[:, 0, 2] = H[:, 2, 0] = 0.25 * (G[z + 1, y, x + 1] - G[z + 1, y, x - 1]
                                          - G[z - 1, y, x + 1] + G[z - 1, y, x - 1])
        H[:, 1, 2] = H[:, 2, 1] = 0.25 * (G[z, y + 1, x + 1] - G[z, y + 1, x - 1]
                                          - G[z, y - 1, x + 1] + G[z, y - 1, x - 1])
        lam = _eig3_sym(H)
        lam = np.take_along_axis(lam, np.argsort(np.abs(lam), axis=1), axis=1)
        l2, l3 = np.abs(lam[:, 1]), lam[:, 2]
        return (l3 < 0) & (l2 < prm.plate_ratio * np.abs(l3))

    _rn = int(np.ceil(2.0 / h))
    _o = np.mgrid[-_rn:_rn + 1, -_rn:_rn + 1, -_rn:_rn + 1].reshape(3, -1).T
    nb_off = _o[(np.linalg.norm(_o, axis=1) * h) <= 2.0]

    def judge(seq, cc, pn, jp):
        cz = cc[seq]
        r = Ed[cz[:, 0], cz[:, 1], cz[:, 2]]
        hu = V[cz[:, 0], cz[:, 1], cz[:, 2]]
        if pn is not None:
            # a branch: not thicker than its parent, not clearly darker,
            # not a plate (an atrial wall rim / trabeculae)
            pz = cc[pn[max(0, jp - wn):jp + wn + 1]]
            pr = float(np.median(Ed[pz[:, 0], pz[:, 1], pz[:, 2]]))
            phu = float(np.median(V[pz[:, 0], pz[:, 1], pz[:, 2]]))
            k0 = max(int(round(2.0 / h)), 1)
            br, bh = r[k0:k0 + 2 * wn], hu[k0:k0 + 2 * wn]
            if len(br) >= 3:
                mr = float(np.median(br))
                if mr > prm.vein_parent_ratio * pr and mr >= prm.vein_min_r_mm:
                    return "drop", None, "wider_than_parent"
                if float(np.median(bh)) < phu - prm.vein_hu_drop * delta:
                    return "drop", None, "darker_than_parent"
            if len(seq) >= 6:
                # the centre AND the lumen round it (≤ 2 mm): the edge of a
                # sheet looks like a tube at its centre, not around it
                pf = float(plate_at(cz[1:]).mean())
                nb = (cz[1::2, None, :] + nb_off[None]).reshape(-1, 3)
                nb = nb[np.all((nb >= 0) & (nb <= gmax), axis=1)]
                nb = np.unique(nb, axis=0)
                nb = nb[L[nb[:, 0], nb[:, 1], nb[:, 2]]]
                pn_ = float(plate_at(nb).mean()) if len(nb) else 0.0
                pfl = log.setdefault("plate_fracs", [])
                if len(pfl) < 300:
                    pfl.append([round(pf, 2), round(pn_, 2)])
                if pf > prm.plate_frac or pn_ > prm.plate_nb_frac:
                    return "drop", None, "plate"
        # along the line: a lasting darker stretch that does not narrow
        # (a vein / chamber met at a contact; a stenosis narrows), or a
        # widening + darker one
        k_start = max(int(round(5.0 / h)), wn)
        for k in range(k_start, len(seq) - 2 * wn):
            rb = float(np.median(r[k - wn:k]))
            hb_ = float(np.median(hu[k - wn:k]))
            ra = float(np.median(r[k:k + wn]))
            ha = float(np.median(hu[k:k + wn]))
            ha2 = float(np.median(hu[k:k + 2 * wn]))
            if (ha < hb_ - prm.vein_hu_drop * delta
                    and ha2 < hb_ - prm.vein_hu_drop * delta
                    and ra >= 0.9 * rb):
                return "cut", k, "darker_on"
            if (ra > prm.vein_widen * rb and ra >= prm.vein_min_r_mm
                    and ha < hb_ - prm.vein_widen_hu * delta):
                return "cut", k, "wider_darker_on"
        return "keep", None, ""

    coords, paths, sidx = _teasar(M, Ed, h, [g for _r, g in seeds], prm,
                                  check, judge=judge)
    log["judged"] = dict(getattr(_teasar, "judged", {}))
    del G
    src_role = {}
    kd = cKDTree(coords) if len(coords) else None
    for r, g in seeds:
        if kd is not None:
            src_role.setdefault(int(kd.query(g.astype(float))[1]), r)
    vessels = _to_vessels(coords, paths, src_role, h, k2w, prm, ap, a)
    # ---- tips continued with Auto-Track -----------------------------------
    say("extend")
    log["extended"] = _extend_tips(vessels, vol, dims, Ed, space, w2k, shape,
                                   h, prm, check)
    # ---- 未接続血管 ------------------------------------------------------
    if sd is not None and shell is not None:
        say("unconnected")
        bright = bg + prm.unc_bright_frac * (I_ost - bg)
        Tu = Tb & shell & ~M
        lab, nlab = ndimage.label(Tu, structure=st)
        del Tu
        sizes = np.bincount(lab.ravel())
        allp = (np.vstack([v.points for v in vessels]) if vessels
                else np.zeros((0, 3)))
        kdl = cKDTree(allp) if len(allp) else None
        cands = []
        objs = ndimage.find_objects(lab)
        for lb in range(1, nlab + 1):
            if sizes[lb] < min_vox or objs[lb - 1] is None:
                continue
            sl = objs[lb - 1]
            m = lab[sl] == lb
            ext = np.array([s_.stop - s_.start for s_ in sl]) * h
            if float(np.linalg.norm(ext)) < prm.unc_min_mm:
                continue
            if float(np.median(V[sl][m])) < bright:
                continue
            if float(np.median(Ed[sl][m])) > prm.unc_max_radius_mm:
                continue
            sdv = sd[sl][m]
            if float((sdv <= prm.shell_out_mm).mean()) < prm.unc_shell_frac:
                continue
            if float(np.median(sdv)) > prm.unc_max_sd_mm:
                continue
            if float(np.std(sdv)) > prm.unc_sd_std_mm:
                continue                                  # leaves the surface
            if kdl is not None:
                g = np.argwhere(m)[::3] + np.array([s_.start for s_ in sl])
                dd, _i = kdl.query(k2w(g))
                if float((dd <= prm.unc_near_line_mm).mean()) >= 0.3:
                    continue                              # an extended line
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


def _extend_tips(vessels, vol, dims, Ed, space, w2k, shape, h, prm, check):
    """Continue the tips that stop while still thick (a vessel cut where it
    touches a chamber, a gap longer than the bridged one, a faint stretch)
    with the CPR Auto-Track (trunk only, it bridges a short narrowing). A
    continuation ends where it leaves its system's search space (2 points
    in a row) or meets another line. Returns the number of tips continued."""
    from multi_dicomviewer.core.cpr_autoextend import (ExtendParams,
                                                       extend_vessel)

    def cancelled():
        try:
            check()
        except _Cancelled:
            return True
        return False

    ep = ExtendParams(trunk_only=True, ask_at_fork=False,
                      max_len_mm=prm.ext_max_mm, max_total_mm=prm.ext_max_mm)
    gmax = np.array(shape) - 1

    def kidx(p):
        return tuple(np.clip(np.rint(w2k(p)).astype(int), 0, gmax))

    def tip_r(P):
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        # 2–6 mm back: the last mm taper into whatever stopped the lumen
        last = P[(d >= d[-1] - 6.0) & (d <= d[-1] - 2.0)]
        if len(last) == 0:
            last = P[-1:]
        return float(np.median([Ed[kidx(p)] for p in last]))

    cand = []
    for i, v in enumerate(vessels):
        if not v.connected or v.role == "LM" or len(v.points) < 3:
            continue
        if v.parent is not None and v.length_mm < prm.ext_min_len_mm:
            continue
        r = tip_r(v.points)
        if r >= prm.ext_min_r_mm:
            cand.append((r, i))
    cand.sort(reverse=True)
    n = 0
    for _r, i in cand[:prm.ext_max_tips]:
        check()
        v = vessels[i]
        S = space.get(v.system) if v.system else None
        P = v.points
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        ctrl = _thin(P[d >= d[-1] - 15.0], 3.0)
        if len(ctrl) < 2:
            continue
        try:
            res = extend_vessel(vol, dims, ctrl, ep, cancel=cancelled)
        except Exception:                                # noqa: BLE001
            continue
        if res.trunk is None or len(res.trunk.path) < 2:
            continue
        others = [w.points for j, w in enumerate(vessels) if j != i]
        kdo = cKDTree(np.vstack(others)) if others else None
        new, miss = [], 0
        for q in res.trunk.path[1:]:
            if S is not None and not S[kidx(q)]:
                miss += 1
                if miss >= 2:
                    break
            else:
                miss = 0
            if kdo is not None and kdo.query(q)[0] < prm.ext_near_mm:
                break
            new.append(q)
        while new and S is not None and not S[kidx(new[-1])]:
            new.pop()
        if not new:
            continue
        add = np.asarray(new)
        L_add = float(np.linalg.norm(np.diff(np.vstack([P[-1:], add]),
                                             axis=0), axis=1).sum())
        if L_add < 3.0:
            continue
        v.points = np.vstack([P, add])
        v.ctrl = _thin(v.points, prm.ctrl_step_mm)
        v.length_mm = float(np.linalg.norm(np.diff(v.points, axis=0),
                                           axis=1).sum())
        n += 1
    return n


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
