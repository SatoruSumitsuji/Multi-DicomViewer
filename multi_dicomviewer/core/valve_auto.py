# -*- coding: utf-8 -*-
"""Automatic MV / AoV annulus plane from a rough click (contrast cardiac CT).

Both return the plane the LV module stores for a valve — (centre, normal,
radius) in volume mm (x, y, z; voxel index = P / dims, vol is [z, y, x]) — plus
the annulus points they were fitted to, so the user can check / correct them.

AoV (detect_aov): the aortic root is a bright tube. The tube axis at the click
is the direction whose cross-section is smallest; the root is then followed
along that axis (re-centred / re-aimed at every step) and the cross-section
area profile is read from the LVOT to the ascending aorta. The annulus (the
"virtual basal ring", through the three cusp hinge points) is taken at the
BASE of the sinus-of-Valsalva bulge: just below where the area starts to rise
into the sinus. (After Elattar et al., Int J Cardiovasc Imaging 2016, who split
LVOT / sinus / STJ on the root's cross-sectional shape along its centreline.)

MV (detect_mv): the annulus is the hinge line of the leaflets, where the LA and
LV walls meet at the AV junction; the annular plane is the least-squares plane
of the (saddle-shaped) annulus (Lopes et al., Front Bioeng Biotechnol 2022;
long-axis rotation around an MV-centre / apex axis as in Hashimoto et al.,
Struct Heart 2022). Long-axis half-planes are rotated around the current axis;
in each, the hinge is where the THICK LV myocardium (apex side) gives way to
the THIN LA wall (base side) — the AV junction — read from the soft-tissue
thickness just outside the blood pool (up to the AV-groove / epicardial fat or
the next blood pool) at each height; the pool's outer edge there is the hinge
point. (The pool's narrowest point is NOT used: in systole the LV base is no
wider than the annulus and papillary muscles / trabeculae narrow the LV pool
more — that picked the annulus too far apex-ward on real CT.) Thin leaflets
are filled by a 2-D closing first. A robust weighted plane fit gives the new axis
(its normal) and centre, and the rotation is repeated, so the rough apex only
seeds the first pass. Two phases: run each one (the first result can seed the
other: pass it as mv_mm / apex_mm = centre ∓ normal).

Pure numpy / scipy, no Qt — headless-testable (tools/test_valve_auto.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


_DEBUG = False
#: A tube section closes on (almost) every ray; a chamber leaves rays open.
_TUBE_OPEN = 0.08


@dataclass
class ValveResult:
    center: np.ndarray                      # (3,) mm
    normal: np.ndarray                      # (3,) unit
    radius: float                           # mm (area-equivalent)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ok: bool = True
    message: str = ""


# ------------------------------------------------------------------ helpers
def _sample(vol, dims, P):
    """Trilinear HU at world-mm points P (..., 3)."""
    P = np.asarray(P, float)
    sh = P.shape[:-1]
    Q = P.reshape(-1, 3)
    sx, sy, sz = (float(d) for d in dims)
    out = ndimage.map_coordinates(
        vol, [Q[:, 2] / sz, Q[:, 1] / sy, Q[:, 0] / sx], order=1,
        mode="nearest", output=np.float32, prefilter=False)
    return out.reshape(sh)


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _frame(n):
    """Two unit vectors ⊥ n (and each other)."""
    n = _unit(n)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(n, ref))
    return u, np.cross(n, u)


def blood_threshold(vol, dims, P, r_mm=12.0):
    """Blood / wall threshold near P: half-way between the local contrast blood
    (95th percentile in an r_mm sphere) and myocardium (~100 HU); ≥ 150 HU."""
    g = np.linspace(-r_mm, r_mm, 13)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    m = X ** 2 + Y ** 2 + Z ** 2 <= r_mm ** 2
    pts = np.asarray(P, float) + np.column_stack([X[m], Y[m], Z[m]])
    hb = float(np.percentile(_sample(vol, dims, pts), 95))
    return max(150.0, 0.5 * (hb + 100.0))


def _snap_into_blood(vol, dims, P, thr, r_mm=8.0):
    """P if it is in the blood, else the nearest blood point within r_mm."""
    P = np.asarray(P, float)
    if _sample(vol, dims, P[None])[0] >= thr:
        return P
    g = np.arange(-r_mm, r_mm + 0.01, 0.5)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    off = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    d = np.linalg.norm(off, axis=1)
    keep = d <= r_mm
    off, d = off[keep], d[keep]
    hu = _sample(vol, dims, P + off)
    ok = hu >= thr
    if not ok.any():
        return P
    return P + off[ok][int(np.argmin(d[ok]))]


def _ray_extent(hu, step, gap_mm, lead_mm=2.0):
    """Blood extent along rays: hu (..., K) sampled from the axis outwards at
    *step* mm. Returns (radius, open): the last blood sample before a dark run
    of ≥ gap_mm (thinner dark runs — leaflets — are bridged); a dark lead-in of
    ≤ lead_mm at the axis (a coaptation line) is skipped. open = never closed."""
    inside = np.asarray(hu, bool)
    K = inside.shape[-1]
    g = max(1, int(round(gap_mm / step)))
    flat = inside.reshape(-1, K)
    rad = np.zeros(flat.shape[0])
    opn = np.zeros(flat.shape[0], bool)
    lead = int(round(lead_mm / step))
    for i, row in enumerate(flat):
        idx = np.flatnonzero(row)
        if len(idx) == 0 or idx[0] > lead:
            rad[i] = 0.0
            continue
        # gaps between consecutive blood samples
        jumps = np.flatnonzero(np.diff(idx) > g)
        last = idx[jumps[0]] if len(jumps) else idx[-1]
        rad[i] = last * step
        opn[i] = (not len(jumps)) and idx[-1] >= K - 1
    sh = inside.shape[:-1]
    return rad.reshape(sh), opn.reshape(sh)


def _cross_section(vol, dims, c, n, thr, rmax=24.0, nray=48, step=0.5,
                   gap_mm=1.2):
    """Ray-cast lumen cross-section in the plane (c, n): area (mm²), centroid,
    fraction of rays that did not close within rmax, radii."""
    u, w = _frame(n)
    th = np.linspace(0.0, 2 * np.pi, nray, endpoint=False)
    D = np.cos(th)[:, None] * u + np.sin(th)[:, None] * w          # (R, 3)
    rs = np.arange(0.0, rmax + 1e-6, step)
    P = c + D[:, None, :] * rs[None, :, None]
    hu = _sample(vol, dims, P) >= thr
    rad, opn = _ray_extent(hu, step, gap_mm)
    area = 0.5 * float(np.sum(rad ** 2)) * (2 * np.pi / nray)
    cen = np.asarray(c, float)
    if rad.max() > 0:
        # Area centroid of the ray polygon (not the mean of its vertices, which
        # is biased when c is off-centre in an elongated section).
        x, y = rad * np.cos(th), rad * np.sin(th)
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        a2 = float(cr.sum())
        if abs(a2) > 1e-6:
            cx = float(((x + x1) * cr).sum()) / (3.0 * a2)
            cy = float(((y + y1) * cr).sum()) / (3.0 * a2)
            cen = cen + cx * u + cy * w
    return area, cen, float(opn.mean()), rad


def _dirs_cone(n0, max_deg, count):
    """~count unit directions within max_deg of n0 (Fibonacci cap), n0 first."""
    n0 = _unit(n0)
    u, w = _frame(n0)
    out = [n0]
    cmin = np.cos(np.deg2rad(max_deg))
    ga = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(1, count):
        z = 1.0 - (1.0 - cmin) * i / (count - 1)
        r = np.sqrt(max(0.0, 1.0 - z * z))
        a = ga * i
        out.append(z * n0 + r * (np.cos(a) * u + np.sin(a) * w))
    return np.asarray(out)


def _min_area_dir(vol, dims, c, thr, n0=None, max_deg=90.0, count=120,
                  rmax=24.0):
    """Tube axis at c = the plane normal with the smallest cross-section."""
    if n0 is None:
        n0 = np.array([0.0, 0.0, 1.0])
    best, bn = np.inf, _unit(n0)
    for d in _dirs_cone(n0, max_deg, count):
        a, _c, opn, _r = _cross_section(vol, dims, c, d, thr, rmax=rmax,
                                        nray=32, step=0.7)
        a = a * (1.0 + 2.0 * opn)                # open rays = not a tube here
        if a < best:
            best, bn = a, _unit(d)
    return bn


# ---------------------------------------------------------------------- AoV
def detect_aov(vol, dims, click_mm, lv_hint_mm=None) -> ValveResult:
    """Aortic annulus plane from a rough click in / near the aortic root.
    *lv_hint_mm* (optional, e.g. the MV centre or apex) tells which way along
    the root the LV is; otherwise the side that opens into a chamber is."""
    vol = np.asarray(vol)
    c = np.asarray(click_mm, float)
    thr = blood_threshold(vol, dims, c)
    c = _snap_into_blood(vol, dims, c, thr)
    n = _min_area_dir(vol, dims, c, thr)
    # The smallest section is only a first guess (the sinus bulge biases it).
    # Refine on the root's centreline: the centroids of PARALLEL sections of a
    # tube lie on its axis whatever their tilt, so a line through them is the
    # axis; re-slice perpendicular to it and repeat.
    for _ in range(6):
        cens = []
        for sgn in (+1.0, -1.0):
            p = c.copy()
            for _k in range(12):                 # ±12 mm: the root, not the LV
                p = p + sgn * 1.0 * n
                _a, cen, opn, _r = _cross_section(vol, dims, p, n, thr,
                                                  nray=32, step=0.6)
                if opn > _TUBE_OPEN:             # left the tube (LV / chamber)
                    break
                p = cen
                cens.append(p.copy())
        _a, c0, _o, _r = _cross_section(vol, dims, c, n, thr, nray=32, step=0.6)
        cens.append(c0)
        Cn = np.asarray(cens)
        if len(Cn) < 6:
            break
        m = Cn.mean(axis=0)
        _u, _s, vt = np.linalg.svd(Cn - m, full_matrices=False)
        n_new = _unit(vt[0])
        if np.dot(n_new, n) < 0:
            n_new = -n_new
        c = m + np.dot(c - m, n_new) * n_new     # back onto the fitted axis
        done = np.dot(n_new, n) > np.cos(np.deg2rad(0.5))
        n = n_new
        if done:
            break
    a0, c, _o, _r = _cross_section(vol, dims, c, n, thr)

    def walk(sign, n_steps=30, ds=1.0):
        prof = []
        p, d = c.copy(), n * sign
        for k in range(n_steps):
            p = p + ds * d
            a, cen, opn, _r = _cross_section(vol, dims, p, d, thr)
            if opn < 0.5:
                p = p + 0.7 * (cen - p)          # follow the lumen centre
            prof.append(((k + 1) * ds * sign, a, p.copy(), d * sign, opn))
            if opn > 0.6:                        # into a chamber / out of tube
                break
        return prof
    fwd, bwd = walk(+1.0), walk(-1.0)

    # Which side is the LV?
    def opens(prof):
        for s, _a, _p, _d, opn in prof:
            if opn > 0.35:
                return abs(s)
        return np.inf
    if lv_hint_mm is not None:
        lv_sign = -1.0 if np.dot(np.asarray(lv_hint_mm, float) - c, n) < 0 else 1.0
    else:
        of, ob = opens(fwd), opens(bwd)
        if np.isfinite(of) or np.isfinite(ob):
            lv_sign = 1.0 if of < ob else -1.0
        else:
            lv_sign = (1.0 if np.mean([e[1] for e in fwd[-5:]])
                       > np.mean([e[1] for e in bwd[-5:]]) else -1.0)
    lv_side = fwd if lv_sign > 0 else bwd
    ao_side = bwd if lv_sign > 0 else fwd
    # Profile from the LV (s < 0) through the click (0) to the aorta (s > 0).
    prof = ([(-abs(s), a, p, d, o) for s, a, p, d, o in reversed(lv_side)]
            + [(0.0, a0, c.copy(), n * (-lv_sign), 0.0)]
            + [(abs(s), a, p, d, o) for s, a, p, d, o in ao_side])
    S = np.array([e[0] for e in prof])
    A = np.array([e[1] for e in prof])
    O = np.array([e[4] for e in prof])
    if _DEBUG:
        print("lv_sign", lv_sign, "n", np.round(n, 3))
        for e in prof:
            print(f"  s {e[0]:+5.0f} area {e[1]:6.0f} open {e[4]:.2f}")
    valid = O <= _TUBE_OPEN                      # closed sections = the tube
    if valid.sum() < 5:
        return ValveResult(c, n, float(np.sqrt(max(a0, 1.0) / np.pi)), ok=False,
                           message="aortic root not found")
    As = ndimage.median_filter(A, 3)
    # Read the profile LV → aorta: the LVOT waist (narrowest section from 25 mm
    # below to 15 mm above the click — the LV beyond it only widens, closed or
    # not), then the sinus bulge (largest section within 25 mm above the
    # waist), and the annulus at the BASE of the bulge: the most sinus-ward
    # section between them still within 15 % of the waist area.
    win = np.flatnonzero(valid & (S >= -25.0) & (S <= 15.0))
    if len(win) == 0:
        return ValveResult(c, n, float(np.sqrt(max(a0, 1.0) / np.pi)), ok=False,
                           message="aortic root not found")
    i_min = int(win[np.argmin(As[win])])
    up = np.flatnonzero(valid & (S > S[i_min]) & (S <= S[i_min] + 25.0))
    if len(up) == 0:
        return ValveResult(c, n, float(np.sqrt(max(a0, 1.0) / np.pi)), ok=False,
                           message="aortic sinus not found")
    i_sin = int(up[np.argmax(As[up])])
    amin = float(As[i_min])
    lim = amin + 0.15 * (float(As[i_sin]) - amin)
    mid = np.arange(i_min, i_sin + 1)
    cand = mid[As[mid] <= lim]
    i_ann = int(cand.max()) if len(cand) else i_min
    _s, _a, p, d, _o = prof[i_ann]
    a, cen, _o, rad = _cross_section(vol, dims, p, d, thr)
    u, w = _frame(d)
    th = np.linspace(0.0, 2 * np.pi, len(rad), endpoint=False)
    pts = p + (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w) * rad[:, None]
    return ValveResult(np.asarray(cen, float), _unit(d),
                       float(np.sqrt(max(a, 1.0) / np.pi)), points=pts,
                       weights=np.ones(len(pts)))


# ----------------------------------------------------------------------- MV
def _plane_fit(P, w):
    """Weighted least-squares plane: (centroid, unit normal, residuals)."""
    w = np.asarray(w, float)
    c = (P * w[:, None]).sum(0) / w.sum()
    Q = (P - c) * np.sqrt(w)[:, None]
    _u, _s, vt = np.linalg.svd(Q, full_matrices=False)
    nrm = _unit(vt[-1])
    return c, nrm, (P - c) @ nrm


def _disk(r_px):
    r = int(np.ceil(r_px))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r_px * r_px + 1e-6


def _hinges(vol, dims, c, n, thr, H, n_half=16, rmax=40.0, step=0.5,
            gap_mm=1.5, close_mm=1.5):
    """One hinge (waist) point per long-axis half-plane around (c, n).

    Each full long-axis plane (both half-planes) is sampled as an (h, r) image
    and its blood mask morphologically CLOSED (disk close_mm) first: that fills
    the thin dark leaflets — closed in systole they cut the pool into LA and LV
    and, crossed obliquely, are too wide to bridge along a ray — but not the
    wall + fat of the AV junction, so the waist stays."""
    u, w = _frame(n)
    hs = np.arange(-H, H + 1e-6, step)
    rs = np.arange(0.0, rmax + 1e-6, step)
    rr = np.concatenate([-rs[:0:-1], rs])                 # −rmax … +rmax
    i0 = len(rs) - 1                                      # r = 0 column
    st = _disk(close_mm / step)
    pad = st.shape[0]
    pts, wts = [], []
    halves = []
    for th in np.linspace(0.0, np.pi, max(1, n_half // 2), endpoint=False):
        e = np.cos(th) * u + np.sin(th) * w
        P = (c + hs[:, None, None] * n + rr[None, :, None] * e)        # (H, 2R, 3)
        hu = _sample(vol, dims, P)
        inside = hu >= thr
        m = np.pad(inside, pad, mode="edge")
        m = ndimage.binary_closing(m, structure=st)[pad:-pad, pad:-pad]
        halves.append((e, m[:, i0:], hu[:, i0:]))
        halves.append((-e, m[:, i0::-1], hu[:, i0::-1]))
    win = max(4, int(round(_WALL_WIN_MM / step)))
    for e, inside, hu in halves:
        R, opn = _ray_extent(inside, step, gap_mm, lead_mm=4.0)
        R = ndimage.median_filter(R, 5, mode="nearest")
        Tw = _wall_thickness(inside, hu, R, step)
        Tw = ndimage.median_filter(Tw, 3, mode="nearest")
        # The hinge = where the THICK LV myocardium (below, apex side) gives way
        # to the THIN LA wall (above): the AV junction. (The blood pool's waist
        # is not reliable there — in systole the LV base is no wider than the
        # annulus, and papillary muscles / trabeculae narrow the LV pool more.)
        best, k_best = -np.inf, None
        for k in range(win, len(hs) - win):
            if R[k] < 6.0 or opn[k]:
                continue
            lv_t = float(np.median(Tw[k - win:k]))
            la_t = float(np.median(Tw[k + 1:k + 1 + win]))
            if lv_t < _WALL_LV_MM or la_t > _WALL_LA_MM:
                continue
            sc = (lv_t - la_t) - 0.1 * abs(float(hs[k]))
            if sc > best:
                best, k_best = sc, k
        if k_best is None:
            continue
        k = k_best
        rk = float(np.median(R[max(0, k - 2):k + 3]))
        pts.append(c + hs[k] * n + rk * e)
        wts.append(float(np.clip((best + 0.1 * abs(float(hs[k]))) / 6.0,
                                 0.15, 1.0)))
    return np.asarray(pts, float).reshape(-1, 3), np.asarray(wts, float)


#: Hinge search (MV): wall thickness = the soft tissue just outside the blood
#: pool, up to fat (epicardial / AV-groove, < _FAT_HU) or another blood pool.
_FAT_HU = -20.0
_WALL_LV_MM = 5.0        # LV myocardium below the annulus is at least this thick
_WALL_LA_MM = 4.0        # the LA wall above it is at most this thick
_WALL_WIN_MM = 5.0       # each side's median window along the axis


def _wall_thickness(inside, hu, R, step, fat_run_mm=1.5, blood_run_mm=2.0):
    """Per row (height h): soft-tissue thickness from the pool's outer edge R
    to the first fat run (≥ fat_run_mm) or blood run (≥ blood_run_mm)."""
    nh, nr = inside.shape
    fr = max(1, int(round(fat_run_mm / step)))
    br = max(1, int(round(blood_run_mm / step)))
    fat = hu < _FAT_HU
    out = np.zeros(nh)
    for i in range(nh):
        j0 = int(round(R[i] / step)) + 1
        if j0 >= nr:
            out[i] = 0.0
            continue
        f = fat[i, j0:]
        b = inside[i, j0:]
        stop = len(f)
        # first index starting a run of fr fat / br blood samples
        if len(f) >= fr:
            cf = np.convolve(f.astype(int), np.ones(fr, int), "valid") >= fr
            if cf.any():
                stop = min(stop, int(np.argmax(cf)))
        if len(b) >= br:
            cb = np.convolve(b.astype(int), np.ones(br, int), "valid") >= br
            if cb.any():
                stop = min(stop, int(np.argmax(cb)))
        out[i] = stop * step
    return out


def detect_mv(vol, dims, mv_mm, apex_mm, iters=4) -> ValveResult:
    """Mitral annulus plane from a rough MV-centre click and a rough apex (it
    only seeds the first rotation axis; later passes rotate around the fitted
    plane's normal)."""
    vol = np.asarray(vol)
    c = np.asarray(mv_mm, float)
    n = _unit(c - np.asarray(apex_mm, float))            # apex → base (LA)
    thr = blood_threshold(vol, dims, c)
    c = _snap_into_blood(vol, dims, c, thr)
    pts = np.zeros((0, 3))
    wts = np.zeros(0)
    for it in range(int(iters)):
        H = 18.0 if it == 0 else 10.0
        P, W = _hinges(vol, dims, c, n, thr, H)
        if len(P) < 6:
            if it > 0:                           # keep the last good fit
                break
            return ValveResult(c, n, 15.0, points=P, weights=W, ok=False,
                               message="mitral annulus not found")
        keep = np.ones(len(P), bool)
        # Robust: drop outliers — not on the first pass, whose half-planes are
        # still oblique to the annulus (its spread is expected, not outliers).
        for _ in range(3 if it > 0 else 0):
            cc, nn, res = _plane_fit(P[keep], W[keep])
            rr = np.linalg.norm((P - cc) - np.outer((P - cc) @ nn, nn), axis=1)
            med_r = float(np.median(rr[keep]))
            res_all = np.abs((P - cc) @ nn)
            mad = float(np.median(np.abs(res_all[keep]))) + 1e-6
            new = (res_all <= max(3.0, 3.0 * mad)) & (rr <= 1.5 * med_r) \
                & (rr >= 0.5 * med_r)
            if new.sum() < 6 or np.array_equal(new, keep):
                break
            keep = new
        cc, nn, _res = _plane_fit(P[keep], W[keep])
        if np.dot(nn, n) < 0:
            nn = -nn
        moved = float(np.linalg.norm(cc - c))
        turned = float(np.degrees(np.arccos(np.clip(np.dot(nn, n), -1, 1))))
        c, n = cc, nn
        pts, wts = P[keep], W[keep]
        if it > 0 and moved < 0.3 and turned < 0.5:
            break
    # Area-equivalent radius of the annulus projected on the plane.
    u, w = _frame(n)
    Q = pts - c
    ang = np.arctan2(Q @ w, Q @ u)
    o = np.argsort(ang)
    x, y = (Q @ u)[o], (Q @ w)[o]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return ValveResult(c, n, float(np.sqrt(max(area, 1.0) / np.pi)),
                       points=pts, weights=wts)
