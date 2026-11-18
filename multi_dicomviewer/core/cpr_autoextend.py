"""CPR Auto-Track — extend a coronary centreline DISTALLY from its last edit
point through the contrast lumen, following every branch it meets.

Predict → search → correct (user 2026-11-18, rebuilt):

1. Predict: the course = the 3-D vector of the line's most distal 3 mm;
   the predicted point Q = the current centre + 3 mm along it.
2. Search: the cross-section ⟂ the course at Q, radius 5 mm. Lumen regions
   (above the half-max threshold) are measured: inscribed-circle radius,
   centre (two vessels merged in one region near a carina are split by
   their inscribed-circle peaks), mean HU. A region counts only if
   - the straight segment from the current centre to it stays in the lumen
     (a vessel 2 mm away across fat — a vein, a neighbour — is not taken),
   - it is not on something already traced,
   - its size and HU are plausible for this vessel.
   A large region touching the search edge near Q = a chamber / the aorta
   → stop.
3. Choose (A案): one region → it; several → the clearly WIDER (×1.25),
   else the clearly STRAIGHTER; else the line stops there and every region
   becomes its own branch. The others become side branches.
4. Correct: the course is re-measured from the found centre and the centre
   taken again in the section ⟂ that course (an oblique cut in a bend is
   off-centre). Then the next course = the last 3 mm of the traced line.
5. Nothing at 3 mm → try 1.5 mm (a tight bend / the vessel's end); still
   nothing → a SHORT narrowing is bridged at 4.5 mm only when the region
   there lies on the course, is the same size, and the gap is not fat
   (its weakest sample ≥ bg + 25 % of the contrast). Else the line ends.

At the trunk's start (and on every step for side branches leaving at
55–120°) rays from the centre look for lumen that leaves sideways — a line
ending at a carina sees LCX as well as LAD.

Each step is logged (``ExtendResult.log``: positions mm, radius, HU,
candidates, choice, events — no patient data) for tuning on real data.

Volumes are ``vol[z, y, x]`` with ``dims = (sx, sy, sz)`` mm, world = voxel ·
spacing (origin 0) — the convention of ``viewers/cpr_mixin.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class ExtendParams:
    step_mm: float = 3.0           # predicted point ahead (the user's 3 mm)
    short_step_mm: float = 1.5     # retry when nothing at step_mm
    bridge_mm: float = 4.5         # a short narrowing is bridged up to this
    dir_mm: float = 3.0            # course = the line's last dir_mm vector
    search_mm: float = 5.0         # search radius in the section
    grid_mm: float = 0.25
    max_ang_deg: float = 75.0      # a region further off the course: ignored
    start_ray_deg: float = 120.0   # trunk start: sideways lumen up to this
    side_ray_min_deg: float = 55.0  # side branches by rays: this …
    side_ray_max_deg: float = 120.0  # … to this off the course
    ray_mm: float = 3.0
    uturn_deg: float = 80.0        # course change per step above → stop
    hairpin_deg: float = 100.0     # nothing ahead → connected lumen up to
    #                                this far off the course (a hairpin)
    wide_ratio: float = 1.25       # "clearly wider" (radius ratio)
    straight_gap_deg: float = 20.0  # "clearly straighter" (angle gap)
    straight_max_deg: float = 30.0  # … and itself within this of the course
    min_radius_mm: float = 0.6     # thinner → stop
    branch_min_radius_mm: float = 0.75
    max_radius_mm: float = 3.5     # wider → a chamber / the aorta → stop
    radius_jump: float = 1.7       # sudden widening (× running radius) → stop
    half_frac: float = 0.5         # lumen threshold = bg + f·(lumen − bg)
    hu_frac: float = 0.6           # region mean HU ≥ bg + f·(lumen − bg)
    bridge_frac: float = 0.25      # bridged gap's weakest ≥ bg + f·contrast
    min_contrast_hu: float = 120.0  # lumen − background must exceed this
    traced_mm: float = 1.5         # a region this close to a traced line
    max_len_mm: float = 160.0      # per line
    max_total_mm: float = 600.0    # all lines together
    min_branch_mm: float = 10.0
    max_depth: int = 3
    ctrl_mm: float = 3.0           # output edit-point spacing


@dataclass
class Segment:
    """One traced, unbranched line (world mm, proximal → distal)."""
    path: np.ndarray                 # traced centres (first = start point)
    ctrl: np.ndarray                 # edit points (≈ ctrl_mm apart)
    parent: int                      # -1 = the input line; else a Segment
    depth: int
    stop: str = ""                   # why it ended


@dataclass
class ExtendResult:
    trunk: Segment | None            # the input line's continuation (or None)
    branches: list = field(default_factory=list)   # [Segment]
    stop: str = ""                   # why the trunk ended
    log: list = field(default_factory=list)        # per-step diagnostics


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


def _sphere_dirs(n=900):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi),
                 np.cos(phi)]


_DIRS = _sphere_dirs()


def _r3(a):
    return [round(float(x), 2) for x in np.asarray(a, float).ravel()]


def _regions(V, Q, d, thr, R=5.0, h=0.25):
    """Lumen regions of the section ⟂ *d* at *Q* (disc radius *R*):
    [{c (world), r (inscribed radius), hu, big, area}]. A region holding
    two vessels (merged near a carina) gives one entry per vessel."""
    u, v = _frame(d)
    g = np.arange(-R, R + 1e-6, h)
    A, B = np.meshgrid(g, g, indexing="ij")
    img = V.sample(Q + A[..., None] * u + B[..., None] * v)
    rr = np.hypot(A, B)
    m = (img > thr) & (rr <= R)
    lab, n = ndimage.label(m)
    out = []
    if n == 0:
        return out
    edge = rr > R - 1.5 * h
    min_area = np.pi * 0.45 ** 2
    for i in range(1, n + 1):
        comp = lab == i
        area = float(comp.sum()) * h * h
        if area < min_area:
            continue
        edt = ndimage.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1] * h
        emax = float(edt.max())
        touch = bool((comp & edge).any())
        big = (touch and emax >= 2.5) or area > np.pi * 16.0
        # inscribed-circle peaks (≥ 55 % of the deepest), non-max suppressed
        k = max(3, int(round(1.0 / h)) * 2 + 1)
        mx = ndimage.maximum_filter(edt, size=k)
        pk = np.argwhere(comp & (edt >= mx - 1e-9)
                         & (edt >= max(0.45, 0.55 * emax)))
        pk = sorted(pk.tolist(), key=lambda ij: -edt[ij[0], ij[1]])
        keep = []
        for ij in pk:
            e = edt[ij[0], ij[1]]
            ok = True
            for kj in keep:
                ek = edt[kj[0], kj[1]]
                dd = np.hypot((ij[0] - kj[0]) * h, (ij[1] - kj[1]) * h)
                if dd < max(1.2, 1.6 * max(e, ek)):
                    ok = False
                    break
                # two vessels only with a waist between them (a vessel cut
                # lengthwise in a bend is one long region with no waist)
                tt = np.linspace(0.0, 1.0, 24)
                ii = np.rint(ij[0] + tt * (kj[0] - ij[0])).astype(int)
                jj = np.rint(ij[1] + tt * (kj[1] - ij[1])).astype(int)
                if float(edt[ii, jj].min()) >= 0.75 * min(e, ek):
                    ok = False
                    break
            if ok:
                keep.append(ij)
        if big:
            keep = keep[:1]
        for ij in keep:
            e = float(edt[ij[0], ij[1]])
            a0, b0 = A[ij[0], ij[1]], B[ij[0], ij[1]]
            disc = comp & (np.hypot(A - a0, B - b0) <= e + 0.5)
            w = np.clip(img - thr, 0, None) * disc
            s = float(w.sum())
            ca, cb = ((float((w * A).sum() / s), float((w * B).sum() / s))
                      if s > 0 else (a0, b0))
            core = comp & (np.hypot(A - ca, B - cb) <= max(0.5, 0.6 * e))
            hu = float(img[core].mean()) if core.any() else float(
                img[ij[0], ij[1]])
            out.append({"c": Q + ca * u + cb * v, "r": e + 0.5 * h, "hu": hu,
                        "big": big, "area": area})
    return out


def _seg_min(V, P, C, n=None):
    """Weakest HU along the straight segment P → C (ends excluded)."""
    L = float(np.linalg.norm(np.asarray(C) - np.asarray(P)))
    n = n or max(6, int(L / 0.25))
    t = np.linspace(0.0, 1.0, n + 2)[1:-1]
    pts = np.asarray(P)[None] + t[:, None] * (np.asarray(C) - np.asarray(P))
    return float(V.sample(pts).min())


class _Tracer:
    def __init__(self, V: _Vol, prm: ExtendParams, lumen, bg, cancel=None,
                 progress=None, log=None, on_step=None):
        self.V = V
        self.p = prm
        self.lumen0 = float(lumen)
        self.bg = float(bg)
        self.done = np.zeros((0, 3))   # dense traced points
        self.total = 0.0
        self.cancel = cancel
        self.progress = progress       # fn(stage, line_mm, total_mm, k, n)
        self.stage = ("trunk", 0, 0)
        self.log = log if log is not None else []
        # fn(line_id, depth, path (N,3), course) after every accepted point;
        # returning False stops here (the trace so far is kept)
        self.on_step = on_step

    # ---- helpers ---------------------------------------------------------
    def add_done(self, pts):
        P = np.asarray(pts, float).reshape(-1, 3)
        if len(P) == 0:
            return
        dense = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            k = max(2, int(np.linalg.norm(b - a) / 0.5) + 1)
            dense.extend(np.linspace(a, b, k)[1:])
        self.done = np.vstack([self.done, np.asarray(dense)])

    def near_done(self, C, tol) -> bool:
        if len(self.done) == 0:
            return False
        return bool((np.linalg.norm(self.done - C, axis=1) < tol).any())

    def _course(self, path):
        """The 3-D vector of the traced line's most distal dir_mm."""
        P = np.asarray(path, float)
        end = P[-1]
        acc = 0.0
        for k in range(len(P) - 2, -1, -1):
            acc += float(np.linalg.norm(P[k + 1] - P[k]))
            if acc >= self.p.dir_mm - 1e-6:
                return _unit(end - P[k])
        return _unit(end - P[0])

    def _cands(self, P, d, s, thr, lumen, r, own):
        """Plausible lumen regions at s mm ahead (see the module doc).
        Returns (cands, chamber, n_traced)."""
        prm = self.p
        Q = P + s * d
        regs = _regions(self.V, Q, d, thr, prm.search_mm, prm.grid_mm)
        out, chamber, n_tr = [], False, 0
        hu_min = self.bg + prm.hu_frac * (lumen - self.bg)
        for g in regs:
            c = g["c"]
            w = _unit(c - P)
            ang = _angle(w, d)
            if ang > prm.max_ang_deg:
                continue
            if _seg_min(self.V, P, c) <= thr:
                continue                         # not connected (fat gap)
            if g["big"] or g["r"] > prm.max_radius_mm * 1.3:
                if np.linalg.norm(c - Q) < max(2.5, 2.0 * r):
                    chamber = True
                continue
            if self._traced(c, own):
                n_tr += 1
                continue
            if g["hu"] < hu_min:
                continue
            out.append({"c": c, "w": w, "r": g["r"], "hu": g["hu"],
                        "ang": ang, "s": s, "how": "disc"})
        return out, chamber, n_tr

    def _traced(self, c, own) -> bool:
        """*c* lies on an already traced line (not this line's own last
        few mm, which the step starts from)."""
        if len(self.done) == 0:
            return False
        dd = np.linalg.norm(self.done - c, axis=1)
        hit = dd < self.p.traced_mm
        if not hit.any():
            return False
        if own is not None and len(own):
            # points of this line's recent path don't count
            o = np.asarray(own, float)
            near_own = np.linalg.norm(
                self.done[hit][:, None, :] - o[None, :, :], axis=2).min(
                axis=1) < 0.3
            return bool((~near_own).any())
        return True

    def _ray_cands(self, P, d, thr, lumen, lo, hi, own):
        """Lumen leaving sideways: rays (lo..hi° off the course) whose
        weakest sample stays in the lumen for ray_mm, then centred."""
        prm = self.p
        cs = _DIRS @ _unit(d)
        W = _DIRS[(cs <= np.cos(np.radians(lo)))
                  & (cs >= np.cos(np.radians(hi)))]
        if len(W) == 0:
            return []
        ts = np.linspace(0.4, prm.ray_mm, 10)
        val = self.V.sample(P + ts[None, :, None] * W[:, None, :])
        score = val.min(axis=1)
        okm = score > thr
        if not okm.any():
            return []
        order = np.argsort(-score)
        picked = []
        for i in order:
            if not okm[i]:
                break
            if all(_angle(W[i], W[j]) >= 25.0 for j in picked):
                picked.append(i)
        hu_min = self.bg + prm.hu_frac * (lumen - self.bg)
        out = []
        for i in picked:
            Qr = P + prm.ray_mm * W[i]
            regs = _regions(self.V, Qr, W[i], thr, 2.5, prm.grid_mm)
            if not regs:
                continue
            g = min(regs, key=lambda x: np.linalg.norm(x["c"] - Qr))
            if np.linalg.norm(g["c"] - Qr) > 1.5 or g["big"]:
                continue
            c = g["c"]
            if _seg_min(self.V, P, c) <= thr or self._traced(c, own) \
                    or g["hu"] < hu_min:
                continue
            w = _unit(c - P)
            out.append({"c": c, "w": w, "r": g["r"], "hu": g["hu"],
                        "ang": _angle(w, d), "s": prm.ray_mm, "how": "ray"})
        return out

    def _choose(self, cands):
        """(main, others, ambiguous) per A案."""
        prm = self.p
        if not cands:
            return None, [], False
        if len(cands) == 1:
            return cands[0], [], False
        g = sorted(cands, key=lambda c: -c["r"])
        if g[0]["r"] >= prm.wide_ratio * g[1]["r"]:
            return g[0], g[1:], False
        a = sorted(cands, key=lambda c: c["ang"])
        if (a[0]["ang"] <= prm.straight_max_deg
                and a[1]["ang"] - a[0]["ang"] >= prm.straight_gap_deg):
            return a[0], a[1:], False
        return None, cands, True

    def _refine(self, P, c, thr):
        """Re-take the centre in the section ⟂ the corrected course."""
        w = _unit(c - P)
        s = float(np.linalg.norm(c - P))
        regs = _regions(self.V, P + s * w, w, thr, 2.5, self.p.grid_mm)
        if not regs:
            return None
        g = min(regs, key=lambda x: np.linalg.norm(x["c"] - c))
        if np.linalg.norm(g["c"] - c) > 1.5 or g["big"]:
            return None
        if _seg_min(self.V, P, g["c"]) <= thr:
            return None
        return g

    @staticmethod
    def _seed_known(seeds, P, w, Q=None) -> bool:
        """The same side branch already found (from this or a nearby step):
        a close start in a similar direction, or its point on the first
        3 mm of a known branch."""
        for sd in seeds:
            if np.linalg.norm(sd["P"] - P) < 4.0 and _angle(sd["w"], w) < 35.0:
                return True
            if Q is not None:
                A = np.asarray(sd["P"], float)
                B = np.asarray(sd["Q"], float) + 3.0 * _unit(sd["w"])
                ab = B - A
                t = float(np.clip((Q - A) @ ab / max(ab @ ab, 1e-9), 0, 1))
                if np.linalg.norm(Q - (A + t * ab)) < max(2.0,
                                                          1.5 * sd["r"]):
                    return True
        return False

    def _bridge(self, P, d, thr, lumen, r, own):
        prm = self.p
        Q = P + prm.bridge_mm * d
        regs = _regions(self.V, Q, d, thr, 2.5, prm.grid_mm)
        floor = self.bg + prm.bridge_frac * (lumen - self.bg)
        hu_min = self.bg + prm.hu_frac * (lumen - self.bg)
        for g in sorted(regs, key=lambda x: np.linalg.norm(x["c"] - Q)):
            c = g["c"]
            if np.linalg.norm(c - Q) > 1.5 or g["big"]:
                continue
            if not (0.6 * r <= g["r"] <= 1.5 * max(r, 0.8)):
                continue
            if g["hu"] < hu_min or self._traced(c, own):
                continue
            if _seg_min(self.V, P, c) < floor:
                continue                         # fat in between: not ours
            w = _unit(c - P)
            return {"c": c, "w": w, "r": g["r"], "hu": g["hu"],
                    "ang": _angle(w, d), "s": prm.bridge_mm, "how": "bridge"}
        return None

    # ---- one line ----------------------------------------------------------
    def trace(self, P0, d0, r0, depth, line_id, prefix=None):
        """Trace one line from centre *P0* with course *d0*. *prefix* = the
        points before P0 (for the course). Returns (path, stop, seeds)."""
        prm = self.p
        P = np.asarray(P0, float)
        hist = [np.asarray(q, float) for q in (prefix or [])] + [P.copy()]
        d = _unit(d0)
        r = float(r0)
        lumen = self.lumen0
        path = [P.copy()]
        seeds = []
        length = 0.0
        stop = "length"
        step = 0
        while True:
            step += 1
            if self.progress is not None:
                st, k_, n_ = self.stage
                self.progress(st, length, self.total, k_, n_)
            if self.cancel is not None and self.cancel():
                stop = "cancel"
                break
            if length >= prm.max_len_mm or self.total >= prm.max_total_mm:
                stop = "length"
                break
            if lumen - self.bg < prm.min_contrast_hu:
                stop = "contrast"
                break
            thr = self.bg + prm.half_frac * (lumen - self.bg)
            own = path[-3:]
            rec = {"line": line_id, "step": step, "P": _r3(P), "d": _r3(d),
                   "r": round(r, 2), "lumen": round(lumen, 1),
                   "thr": round(thr, 1)}
            starting = depth > 0 and length < 4.0
            cands, chamber, n_tr = [], False, 0
            for s in (prm.step_mm, prm.short_step_mm):
                cands, chamber, n_tr = self._cands(P, d, s, thr, lumen, r,
                                                   own)
                if cands or chamber:
                    break
            if depth == 0 and length < 1e-6 and not chamber:
                # the trunk's start: lumen leaving sideways (carina)
                for c in self._ray_cands(P, d, thr, lumen, 0.0,
                                         prm.start_ray_deg, own):
                    if all(_angle(c["w"], o["w"]) >= 25.0 for o in cands):
                        cands.append(c)
            rec["cands"] = [{"c": _r3(c["c"]), "r": round(c["r"], 2),
                             "hu": round(c["hu"], 1),
                             "ang": round(c["ang"], 1), "s": c["s"],
                             "how": c["how"]} for c in cands]
            if chamber and not cands:
                stop = "widening"
                rec["event"] = "chamber ahead"
                self.log.append(rec)
                break
            if not cands and not chamber:
                # a hairpin: the lumen goes on sideways (connected rays)
                cands = self._ray_cands(P, d, thr, lumen, 0.0,
                                        prm.hairpin_deg, own)
                if cands:
                    rec["event"] = "sharp bend (rays)"
                    rec["cands"] = [{"c": _r3(c["c"]), "r": round(c["r"], 2),
                                     "ang": round(c["ang"], 1),
                                     "how": c["how"]} for c in cands]
            if not cands:
                br = self._bridge(P, d, thr, lumen, r, own)
                if br is None:
                    stop = "revisit" if n_tr else "lumen"
                    rec["event"] = "nothing found"
                    self.log.append(rec)
                    break
                cands = [br]
                rec["event"] = "bridged a narrowing"
            if starting or rec.get("event") == "sharp bend (rays)":
                # a branch still beside its parent / a hairpin: the region nearest the
                # predicted point, no new branches yet
                main = min(cands, key=lambda c: c["ang"])
                others, amb = [], False
            else:
                main, others, amb = self._choose(cands)
            if amb:
                for c in others:
                    if c["r"] >= prm.branch_min_radius_mm:
                        seeds.append({"P": P.copy(), "w": c["w"],
                                      "r": c["r"], "Q": c["c"]})
                stop = "bifurcation"
                rec["event"] = "fork: no clear trunk"
                self.log.append(rec)
                break
            for c in others:
                if c["r"] >= prm.branch_min_radius_mm \
                        and not self._seed_known(seeds, P, c["w"], c["c"]):
                    seeds.append({"P": P.copy(), "w": c["w"], "r": c["r"],
                                  "Q": c["c"]})
            g = self._refine(P, main["c"], thr) if main["how"] != "bridge" \
                else None
            C = g["c"] if g is not None else main["c"]
            rn = g["r"] if g is not None else main["r"]
            hu = g["hu"] if g is not None else main["hu"]
            rec["pick"] = {"c": _r3(C), "r": round(rn, 2), "hu": round(hu, 1),
                           "how": main["how"]}
            if not self.V.inside(C):
                stop = "edge"
                rec["event"] = "volume edge"
                self.log.append(rec)
                break
            if rn > prm.max_radius_mm or rn > prm.radius_jump * max(r, 0.8):
                stop = "widening"
                rec["event"] = "widens"
                self.log.append(rec)
                break
            if rn < prm.min_radius_mm:
                stop = "thin"
                rec["event"] = "too thin"
                self.log.append(rec)
                break
            dn = _unit(C - P)
            lim = (prm.start_ray_deg if (depth == 0 and length < 1e-6)
                   else prm.hairpin_deg + 5.0 if main["how"] == "ray"
                   else prm.uturn_deg)
            if _angle(dn, d) > lim:
                stop = "turn"
                rec["event"] = "U-turn"
                self.log.append(rec)
                break
            # side branches leaving sideways at this step
            if not starting:
                for c in self._ray_cands(P, d, thr, lumen,
                                         prm.side_ray_min_deg,
                                         prm.side_ray_max_deg, own):
                    if (c["r"] >= prm.branch_min_radius_mm
                            and _angle(c["w"], dn) >= 35.0
                            and not self._seed_known(seeds, P, c["w"],
                                                     c["c"])):
                        seeds.append({"P": P.copy(), "w": c["w"],
                                      "r": c["r"], "Q": c["c"]})
                        rec.setdefault("side", []).append(_r3(c["c"]))
            self.log.append(rec)
            seg = float(np.linalg.norm(C - P))
            length += seg
            self.total += seg
            P = C
            path.append(P.copy())
            hist.append(P.copy())
            d = self._course(hist)
            r = 0.7 * r + 0.3 * rn
            lumen = 0.85 * lumen + 0.15 * max(hu, thr)
            if self.on_step is not None and self.on_step(
                    line_id, depth, np.asarray(path), d) is False:
                stop = "cancel"
                break
        self.add_done(path)
        return np.asarray(path), stop, seeds


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


def _duplicate(path, others, tol=1.5, frac=0.6) -> bool:
    """Most of *path* (past its first 3 mm) runs within *tol* mm of one of
    *others* → a duplicate trace of that vessel."""
    P = np.asarray(path, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s_ = np.r_[0.0, np.cumsum(seg)]
    P = P[s_ >= 3.0]
    if len(P) < 2:
        return False
    for O in others:
        O = np.asarray(O, float)
        if len(O) < 2:
            continue
        # densify the other line so its 3 mm gaps don't hide an overlap
        dense = [O[0]]
        for a, b in zip(O[:-1], O[1:]):
            k = max(2, int(np.linalg.norm(b - a) / 0.5) + 1)
            dense.extend(np.linspace(a, b, k)[1:])
        Dn = np.asarray(dense)
        d = np.linalg.norm(P[:, None, :] - Dn[None, :, :], axis=2).min(axis=1)
        if float((d < tol).mean()) >= frac:
            return True
    return False


def _arc_point(C, back_mm):
    """The point *back_mm* before the end along polyline *C*."""
    C = np.asarray(C, float)
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    t = max(0.0, float(s[-1]) - back_mm)
    return np.array([np.interp(t, s, C[:, i]) for i in range(3)])


def _lumen_stats(V, C, dir_mm=3.0):
    """Reference lumen HU, background HU, radius and course from the given
    line's distal part (its own vessel, so the tracker starts calibrated)."""
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    L = float(s[-1])
    ts = np.linspace(max(0.0, L - 6.0), L, 8)
    pts = np.c_[np.interp(ts, s, C[:, 0]), np.interp(ts, s, C[:, 1]),
                np.interp(ts, s, C[:, 2])]
    d = _unit(C[-1] - _arc_point(C, dir_mm))
    if np.linalg.norm(d) < 0.5:
        d = _unit(C[-1] - C[-2])
    u, v = _frame(d)
    # the brightest sample within ~1 mm of each line point (a straight
    # hand-drawn segment on a curved / thin vessel leaves the lumen)
    offs = [np.zeros(3)]
    for rad in (0.5, 1.0):
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            offs.append(rad * (np.cos(a) * u + np.sin(a) * v))
    offs = np.asarray(offs)
    near = V.sample((pts[:, None, :] + offs[None, :, :]).reshape(-1, 3))
    lumen = float(np.median(near.reshape(len(pts), len(offs)).max(axis=1)))
    ring = []
    for P in pts:
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            ring.append(P + 4.5 * (np.cos(a) * u + np.sin(a) * v))
    bg = float(np.percentile(V.sample(np.asarray(ring)), 30))
    thr = bg + 0.5 * (lumen - bg)
    rs = []
    for P in pts[-4:]:
        regs = _regions(V, P, d, thr, 4.0, 0.25)
        if regs:
            g = min(regs, key=lambda x: np.linalg.norm(x["c"] - P))
            if np.linalg.norm(g["c"] - P) < 1.5 and not g["big"]:
                rs.append(g["r"])
    r = float(np.median(rs)) if rs else 1.2
    return lumen, bg, r, d, thr


def extend_vessel(vol, dims, ctrl, prm: ExtendParams | None = None,
                  cancel=None, progress=None, on_step=None) -> ExtendResult:
    """Trace on from the LAST point of *ctrl* (world mm, ≥ 2 points,
    proximal → distal). The trunk's continuation (A案) and every branch met
    on the way (each its own unbranched Segment)."""
    prm = prm or ExtendParams()
    V = _Vol(vol, dims)
    C = np.asarray(ctrl, float).reshape(-1, 3)
    log = []
    if len(C) < 2:
        return ExtendResult(None, [], "short", log)
    lumen, bg, r, d, thr = _lumen_stats(V, C, prm.dir_mm)
    log.append({"start": _r3(C[-1]), "course": _r3(d), "lumen": round(lumen, 1),
                "bg": round(bg, 1), "r0": round(r, 2),
                "params": asdict(prm)})
    if lumen - bg < prm.min_contrast_hu:
        return ExtendResult(None, [], "contrast", log)
    T = _Tracer(V, prm, lumen, bg, cancel, progress, log, on_step)
    # the given line counts as traced (nothing runs back into it), except
    # its last 3 mm the trace starts from
    keep = C[np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C, axis=0),
                                                 axis=1))]
             <= max(0.0, float(np.linalg.norm(np.diff(C, axis=0),
                                              axis=1).sum()) - 3.0)]
    if len(keep):
        T.add_done(np.vstack([keep, _arc_point(C, 3.0)[None]]))
    # start from the lumen centre near the last point (a click slightly
    # off-centre would break the "connected" test of the first step)
    P0 = C[-1].copy()
    regs = _regions(V, P0, d, thr, 2.5, prm.grid_mm)
    if regs:
        g = min(regs, key=lambda x: np.linalg.norm(x["c"] - P0))
        if np.linalg.norm(g["c"] - P0) < 1.5 and not g["big"]:
            P0 = g["c"]
    path, stop, seeds = T.trace(P0, d, r, 0, 0, prefix=[_arc_point(C, 3.0)])
    path = np.vstack([C[-1][None], path[1:]]) if len(path) > 1 else path
    trunk = None
    if len(path) >= 2 and np.linalg.norm(path[-1] - path[0]) >= 1.5:
        trunk = Segment(path, _thin(path, prm.ctrl_mm), -1, 0, stop)
    branches = []
    queue = [(s, -1, 1) for s in seeds]
    done_n = 0
    while queue:
        sd, parent, depth = queue.pop(0)
        done_n += 1
        T.stage = ("branch", done_n, done_n + len(queue))
        if progress is not None:
            progress("branch", 0.0, T.total, done_n, done_n + len(queue))
        if depth > prm.max_depth or T.total >= prm.max_total_mm:
            continue
        if (cancel is not None and cancel()) or stop == "cancel":
            break
        Q = np.asarray(sd["Q"], float)
        Pf = np.asarray(sd["P"], float)
        if not V.inside(Q):
            continue
        log.append({"branch": done_n, "fork": _r3(Pf), "start": _r3(Q)})
        bp, bstop, bseeds = T.trace(Q, _unit(Q - Pf),
                                    max(sd["r"], prm.branch_min_radius_mm),
                                    depth, done_n, prefix=[Pf])
        bp = np.vstack([Pf[None], bp])
        L = float(np.linalg.norm(np.diff(bp, axis=0), axis=1).sum())
        if L < prm.min_branch_mm:
            log.append({"branch": done_n, "dropped": "short",
                        "mm": round(L, 1)})
            continue
        if _duplicate(bp, [b_.path for b_ in branches]
                      + ([trunk.path] if trunk is not None else [])):
            log.append({"branch": done_n, "dropped": "duplicate"})
            continue
        branches.append(Segment(bp, _thin(bp, prm.ctrl_mm), parent, depth,
                                bstop))
        me = len(branches) - 1
        queue.extend((s, me, depth + 1) for s in bseeds)
    log.append({"result": {"trunk_mm": round(float(np.linalg.norm(
        np.diff(path, axis=0), axis=1).sum()) if trunk is not None else 0.0,
        1), "stop": stop, "branches": len(branches)}})
    return ExtendResult(trunk, branches, stop, log)


def save_log(log, folder=None, keep: int = 30) -> str:
    """Write an Auto-Track diagnostic log (positions mm / radius / HU /
    candidates / events — no patient data) as JSON under
    ~/.multi-dicomviewer/autotrack_logs (the newest *keep* are kept).
    Returns the path, or "" when it could not be written."""
    import datetime
    import json
    import os
    try:
        if folder is None:
            from multi_dicomviewer.core.settings import SETTINGS_DIR
            folder = os.path.join(str(SETTINGS_DIR), "autotrack_logs")
        os.makedirs(folder, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = os.path.join(folder, f"autotrack_{stamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"format": "MDV-AutoTrackLog", "version": 1,
                       "log": log}, f, ensure_ascii=False, indent=1,
                      default=float)
        old = sorted(n for n in os.listdir(folder)
                     if n.startswith("autotrack_") and n.endswith(".json"))
        for n in old[:-keep]:
            try:
                os.remove(os.path.join(folder, n))
            except OSError:
                pass
        return path
    except Exception:                                    # noqa: BLE001
        return ""
