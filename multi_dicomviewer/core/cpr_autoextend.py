"""CPR 自動追跡 — extend a coronary centreline DISTALLY from its last edit
point through the contrast lumen, following every branch it meets.

Conservative by design (user 2026-11-18): it never guesses. Each step
re-centres in the vessel's cross-section and only goes on while the lumen
is clearly continuous; any doubt ends that line, so a line may come out
short but what it traces is right.

Per step (``step_mm`` ahead):

1. Look ahead: rays from the current centre to a shell (``shell_k`` × the
   lumen radius) inside a cone (``cone_deg``) around the course. A ray
   counts only if its WEAKEST sample stays above the lumen threshold — a
   continuous lumen, not a jump across fat to a neighbour.
2. The rays group into candidates ≥ ``sep_deg`` apart; each is measured
   (its cross-section lumen radius).
3. Choose (A案): the clearly WIDER candidate continues the line (by
   ``wide_ratio``); with similar widths the clearly STRAIGHTER one; if
   neither is clear the line STOPS there and every candidate becomes its
   own branch. The other candidates (wide enough) become side branches,
   each traced the same way (to ``max_depth`` levels).
4. Re-centre in the cross-section (half-max lumen region connected to the
   centre) and update the radius / reference lumen intensity.

A line stops when the lumen fades (centre below the threshold), gets too
thin, suddenly widens (a chamber / the aorta), turns too sharply, runs into
something already traced, leaves the volume or reaches its length cap.
Branches shorter than ``min_branch_mm`` are dropped.

Volumes are ``vol[z, y, x]`` with ``dims = (sx, sy, sz)`` mm, world = voxel ·
spacing (origin 0) — the convention of ``viewers/cpr_mixin.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class ExtendParams:
    step_mm: float = 0.8
    shell_k: float = 2.5           # look-ahead shell = k × lumen radius …
    shell_min_mm: float = 2.0      # … clamped to [min, max]
    shell_max_mm: float = 5.0
    cone_deg: float = 70.0         # look-ahead cone half-angle
    sep_deg: float = 30.0          # candidates at least this far apart
    max_turn_deg: float = 35.0     # per-step course change cap
    wide_ratio: float = 1.25       # "clearly wider" (radius ratio)
    straight_gap_deg: float = 25.0  # "clearly straighter" (angle gap)
    straight_max_deg: float = 30.0  # … and itself within this of the course
    min_radius_mm: float = 0.6     # thinner → stop
    branch_min_radius_mm: float = 0.75
    max_radius_mm: float = 3.5     # wider → a chamber / the aorta → stop
    radius_jump: float = 1.7       # sudden widening (× running radius) → stop
    half_frac: float = 0.5         # lumen threshold = bg + f·(lumen − bg)
    min_contrast_hu: float = 120.0  # lumen − background must exceed this
    lost_steps: int = 2            # centre below threshold this many → stop
    max_len_mm: float = 160.0      # per line
    max_total_mm: float = 600.0    # all lines together
    min_branch_mm: float = 10.0
    max_depth: int = 3
    ctrl_mm: float = 3.0           # output edit-point spacing
    revisit_mm: float = 1.0        # closer than this to a traced line → stop


@dataclass
class Segment:
    """One traced, unbranched line (world mm, proximal → distal)."""
    path: np.ndarray                 # dense centreline
    ctrl: np.ndarray                 # edit points (≈ ctrl_mm apart)
    parent: int                      # -1 = the trunk extension's parent (the
    #                                  input line); else index of a Segment
    depth: int
    stop: str = ""                   # why it ended


@dataclass
class ExtendResult:
    trunk: Segment | None            # the input line's continuation (or None)
    branches: list = field(default_factory=list)   # [Segment]
    stop: str = ""                   # why the trunk ended


class _Vol:
    def __init__(self, vol, dims):
        # sampled in place (no float copy of a whole CT: several 100 MB)
        self.v = np.asarray(vol)
        self.sx, self.sy, self.sz = (float(d) for d in dims)
        nz, ny, nx = self.v.shape
        self.hi = np.array([(nx - 1) * self.sx, (ny - 1) * self.sy,
                            (nz - 1) * self.sz])

    def inside(self, P, margin=1.0) -> bool:
        P = np.asarray(P, float)
        return bool(np.all(P >= margin) and np.all(P <= self.hi - margin))

    def sample(self, P):
        """Trilinear HU at world points P (..., 3); outside → -1000."""
        P = np.asarray(P, float)
        sh = P.shape[:-1]
        Q = P.reshape(-1, 3)
        coords = np.vstack([Q[:, 2] / self.sz, Q[:, 1] / self.sy,
                            Q[:, 0] / self.sx])
        out = ndimage.map_coordinates(self.v, coords, order=1, mode="constant",
                                      cval=-1000.0, output=np.float32)
        return out.reshape(sh)


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _frame(d):
    d = _unit(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(d, a))
    return u, np.cross(d, u)


def _angle(a, b) -> float:
    c = float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _sphere_dirs(n=600):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi),
                 np.cos(phi)]


_DIRS = _sphere_dirs()


def _section(V, P, d, thr, R=4.0, h=0.25):
    """Cross-section at P ⟂ d: (radius_mm, centroid_offset_world, centre
    HU) of the above-*thr* region connected to the centre, or None."""
    u, v = _frame(d)
    g = np.arange(-R, R + 1e-6, h)
    A, B = np.meshgrid(g, g, indexing="ij")
    pts = P + A[..., None] * u + B[..., None] * v
    img = V.sample(pts)
    m = img > thr
    c = len(g) // 2
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    # the component at the centre, else the nearest one within 1 mm
    lc = lab[c, c]
    if lc == 0:
        rr = np.hypot(A, B)
        cand = (lab > 0) & (rr <= 1.0)
        if not cand.any():
            return None
        ii = np.argmin(np.where(cand, rr, np.inf))
        lc = lab.flat[ii]
    comp = lab == lc
    # Inscribed-circle radius (robust where a branch joins: the section is
    # elongated there, its area would overstate the vessel) at the deepest
    # point within 1 mm of the centre; the centroid only over that disc.
    edt = ndimage.distance_transform_edt(comp) * h
    rr = np.hypot(A, B)
    near = comp & (rr <= 1.0)
    if not near.any():
        return None
    k = int(np.argmax(np.where(near, edt, -1.0)))
    r_in = float(edt.flat[k])
    big = bool(comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any()
               or comp[:, -1].any()) and r_in >= R * 0.6
    ka, kb = float(A.flat[k]), float(B.flat[k])
    disc = comp & (np.hypot(A - ka, B - kb) <= r_in + 0.5)
    w = np.clip(img - thr, 0, None) * disc
    s = float(w.sum())
    if s <= 0:
        return None
    ca = float((w * A).sum() / s)
    cb = float((w * B).sum() / s)
    return {"r": 99.0 if big else r_in + 0.5 * h, "off": ca * u + cb * v,
            "hu": float(V.sample(P[None])[0])}


class _Tracer:
    def __init__(self, V: _Vol, prm: ExtendParams, cancel=None):
        self.V = V
        self.p = prm
        self.done_pts = []             # traced points (for the revisit check)
        self.total = 0.0
        self.cancel = cancel

    def _near_traced(self, P, own_recent) -> bool:
        if not self.done_pts:
            return False
        D = np.asarray(self.done_pts)
        d = np.linalg.norm(D - P, axis=1)
        return bool((d < self.p.revisit_mm).any())

    def _candidates(self, P, d, r, thr, cone_deg=None, merge=True):
        prm = self.p
        Rs = float(np.clip(prm.shell_k * r, prm.shell_min_mm,
                           prm.shell_max_mm))
        cosc = np.cos(np.radians(prm.cone_deg if cone_deg is None
                                 else cone_deg))
        W = _DIRS[_DIRS @ _unit(d) >= cosc]
        ts = np.linspace(0.35 * Rs, Rs, 8)
        rays = P + ts[None, :, None] * W[:, None, :]
        val = self.V.sample(rays)                       # (nW, nt)
        score = val.min(axis=1)
        ok = score > thr
        if not ok.any():
            return []
        order = np.argsort(-score)
        picked = []
        for i in order:
            if not ok[i]:
                break
            if all(_angle(W[i], W[j]) >= prm.sep_deg for j in picked):
                picked.append(i)
        out = []
        for i in picked:
            Q = P + Rs * W[i]
            sec = _section(self.V, Q, W[i], thr)
            if sec is None:
                continue
            Qc = Q + sec["off"]
            wc = _unit(Qc - P)
            # same lumen as a stronger candidate (their cross-sections
            # overlap) → the same vessel, not a branch
            if merge and any(np.linalg.norm(Qc - o["Q"]) < min(sec["r"], 3.0)
                   + min(o["r"], 3.0) for o in out):
                continue
            out.append({"w": wc, "r": sec["r"], "Q": Qc,
                        "score": float(score[i]), "ang": _angle(wc, d)})
        return out

    def _choose(self, cands):
        """(main, others, ambiguous) per A案."""
        prm = self.p
        good = [c for c in cands if c["r"] <= prm.max_radius_mm * 1.5]
        if not good:
            return None, [], False
        if len(good) == 1:
            return good[0], [], False
        g = sorted(good, key=lambda c: -c["r"])
        if g[0]["r"] >= prm.wide_ratio * g[1]["r"]:
            return g[0], g[1:], False
        a = sorted(good, key=lambda c: c["ang"])
        if (a[0]["ang"] <= prm.straight_max_deg
                and a[1]["ang"] - a[0]["ang"] >= prm.straight_gap_deg):
            return a[0], a[1:], False
        return None, good, True                         # can't tell → stop

    def trace(self, P0, d0, r0, lumen0, bg, depth):
        """Trace one line from P0 heading d0. Returns (path, stop, seeds)
        where seeds = [(P, d, r, at_index)] side-branch starts."""
        prm = self.p
        P = np.asarray(P0, float)
        d = _unit(d0)
        r = float(r0)
        lumen = float(lumen0)
        path = [P.copy()]
        seeds = []
        lost = 0
        length = 0.0
        stop = "length"
        while True:
            if self.cancel is not None and self.cancel():
                stop = "cancel"
                break
            if length >= prm.max_len_mm or self.total >= prm.max_total_mm:
                stop = "length"
                break
            if lumen - bg < prm.min_contrast_hu:
                stop = "contrast"
                break
            thr = bg + prm.half_frac * (lumen - bg)
            start = depth > 0 and length < 4.0
            cands = self._candidates(P, d, r, thr,
                                     cone_deg=25.0 if start else None,
                                     merge=not start)
            if start:
                # a branch still inside its parent's lumen: follow the seed
                # direction (the parent's own course is not this branch)
                cands = sorted(cands, key=lambda c: c["ang"])
                main, others, amb = (cands[0] if cands else None), [], False
            else:
                main, others, amb = self._choose(cands)
            if amb:
                for c in others:
                    if c["r"] >= prm.branch_min_radius_mm:
                        seeds.append((P.copy(), c["w"], c["r"], len(path) - 1,
                                      np.asarray(c["Q"], float)))
                stop = "bifurcation"
                break
            if main is None:
                lost += 1
                if lost >= prm.lost_steps:
                    stop = "lumen"
                    break
                main = {"w": d, "r": r}
            else:
                lost = 0
            for c in others:
                if c["r"] >= prm.branch_min_radius_mm \
                        and not self._seed_known(seeds, P, c["w"]):
                    seeds.append((P.copy(), c["w"], c["r"], len(path) - 1,
                                      np.asarray(c["Q"], float)))
            w = _unit(main["w"])
            if _angle(w, d) > prm.max_turn_deg:
                stop = "turn"
                break
            Pn = P + prm.step_mm * w
            if not self.V.inside(Pn):
                stop = "edge"
                break
            sec = _section(self.V, Pn, w, thr)
            if sec is None:
                lost += 1
                if lost >= prm.lost_steps:
                    stop = "lumen"
                    break
                P, d = Pn, w
                path.append(P.copy())
                length += prm.step_mm
                continue
            off = sec["off"]
            n = float(np.linalg.norm(off))
            lim = 0.35 * prm.step_mm / 0.8      # ≤ ~24° per step
            if n > lim:
                off = off * (lim / n)
            Pn = Pn + off
            rn = sec["r"]
            if rn > prm.max_radius_mm or rn > prm.radius_jump * max(r, 0.8):
                stop = "widening"
                break
            if rn < prm.min_radius_mm:
                stop = "thin"
                break
            if (np.linalg.norm(Pn - P0) > 3.0 * float(r0) + 1.5
                    and self._near_traced(Pn, path)):
                stop = "revisit"
                break
            dn = _unit(Pn - P)
            if _angle(dn, d) > prm.max_turn_deg:
                stop = "turn"
                break
            hu = float(self.V.sample(Pn[None])[0])
            if hu < thr:
                lost += 1
                if lost >= prm.lost_steps:
                    stop = "lumen"
                    break
            d = _unit(0.6 * dn + 0.4 * d)
            r = 0.8 * r + 0.2 * rn
            lumen = 0.9 * lumen + 0.1 * max(hu, thr)
            seg = float(np.linalg.norm(Pn - P))
            length += seg
            self.total += seg
            P = Pn
            path.append(P.copy())
        # keep the traced points so other lines don't run back into this one
        self.done_pts.extend(path)
        return np.asarray(path), stop, seeds

    @staticmethod
    def _seed_known(seeds, P, w) -> bool:
        for (S, sw, _r, _i, _q) in seeds:
            if np.linalg.norm(S - P) < 4.0 and _angle(sw, w) < 35.0:
                return True
        return False


def _thin(path, step):
    """Edit points ≈ *step* mm apart along *path* (ends kept)."""
    P = np.asarray(path, float)
    if len(P) < 2:
        return P
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    L = float(s[-1])
    if L <= step:
        return P[[0, -1]]
    n = max(2, int(round(L / step)) + 1)
    t = np.linspace(0.0, L, n)
    return np.c_[np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1]),
                 np.interp(t, s, P[:, 2])]


def _lumen_stats(V, ctrl):
    """Reference lumen HU, background HU and radius from the last ~6 mm of
    the given line (its own vessel, so the tracker starts calibrated)."""
    C = np.asarray(ctrl, float)
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    L = float(s[-1])
    ts = np.linspace(max(0.0, L - 6.0), L, 8)
    pts = np.c_[np.interp(ts, s, C[:, 0]), np.interp(ts, s, C[:, 1]),
                np.interp(ts, s, C[:, 2])]
    d = _unit(C[-1] - C[max(0, len(C) - 2)])
    if len(pts) >= 2 and np.linalg.norm(pts[-1] - pts[0]) > 1.0:
        d = _unit(pts[-1] - pts[0])
    hu = V.sample(pts)
    lumen = float(np.median(hu))
    u, v = _frame(d)
    ring = []
    for P in pts:
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            ring.append(P + 4.5 * (np.cos(a) * u + np.sin(a) * v))
    bg = float(np.percentile(V.sample(np.asarray(ring)), 30))
    thr = bg + 0.5 * (lumen - bg)
    rs = []
    for P in pts[-4:]:
        sec = _section(V, P, d, thr)
        if sec is not None and sec["r"] < 50:
            rs.append(sec["r"])
    r = float(np.median(rs)) if rs else 1.2
    return lumen, bg, r, d


def extend_vessel(vol, dims, ctrl, prm: ExtendParams | None = None,
                  cancel=None) -> ExtendResult:
    """Trace on from the LAST point of *ctrl* (world mm, ≥ 2 points,
    proximal → distal). The trunk's continuation (A案) and every branch met
    on the way (each its own unbranched Segment)."""
    prm = prm or ExtendParams()
    V = _Vol(vol, dims)
    C = np.asarray(ctrl, float).reshape(-1, 3)
    if len(C) < 2:
        return ExtendResult(None, [], "short")
    lumen, bg, r, d = _lumen_stats(V, C)
    if lumen - bg < prm.min_contrast_hu:
        return ExtendResult(None, [], "contrast")
    T = _Tracer(V, prm, cancel)
    # the given line counts as traced (nothing runs back into it)
    dense = []
    for a, b in zip(C[:-1], C[1:]):
        k = max(2, int(np.linalg.norm(b - a) / 0.5) + 1)
        dense.extend(np.linspace(a, b, k)[:-1])
    T.done_pts.extend(dense[:-6] if len(dense) > 6 else [])
    path, stop, seeds = T.trace(C[-1], d, r, lumen, bg, 0)
    trunk = None
    if len(path) >= 2 and np.linalg.norm(path[-1] - path[0]) >= 1.5:
        trunk = Segment(path, _thin(path, prm.ctrl_mm), -1, 0, stop)
    branches = []
    # breadth-first over the side branches: (seed, parent index, depth)
    queue = [(s, -1, 1) for s in seeds]
    while queue:
        (P, w, rr, _i, Q), parent, depth = queue.pop(0)
        if depth > prm.max_depth or T.total >= prm.max_total_mm:
            continue
        if cancel is not None and cancel():
            break
        # start in the branch's own lumen (the candidate point, outside the
        # parent), the bifurcation point prepended so the line joins there
        if not T.V.inside(Q):
            continue
        bp, bstop, bseeds = T.trace(Q, w, max(rr, prm.branch_min_radius_mm),
                                    lumen, bg, depth)
        bp = np.vstack([np.asarray(P, float)[None], bp])
        if len(bp) < 2:
            continue
        L = float(np.linalg.norm(np.diff(bp, axis=0), axis=1).sum())
        if L < prm.min_branch_mm:
            continue
        branches.append(Segment(bp, _thin(bp, prm.ctrl_mm), parent, depth,
                                bstop))
        me = len(branches) - 1
        queue.extend((s, me, depth + 1) for s in bseeds)
    return ExtendResult(trunk, branches, stop)
