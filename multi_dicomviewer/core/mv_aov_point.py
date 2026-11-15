# -*- coding: utf-8 -*-
"""MV-AoV transition point (aorto-mitral continuity) + LV apex from ONE rough
click — the Epi long axis of the LV mode (point → apex).

Definition (chosen for reproducibility): the point ON THE AORTIC ANNULUS
(the virtual basal ring found by valve_auto.detect_aov) in the direction of
the mitral inflow, i.e. the aortic end of the aorto-mitral curtain, under the
left / non-coronary commissure. Only the robust aortic ring fixes the point;
the mitral side gives only the DIRECTION:

1. detect_aov from the click → annulus centre c, normal n (toward the aorta),
   ring points.
2. Sections parallel to the annulus on the LV side, down to where the LVOT
   opens into the LV: their blood (thin leaflets bridged by a closing)
   connected to the LVOT. The centroid lies toward the mitral inflow — the
   LV base is wide and the LVOT sits at its antero-septal edge. Direction = LVOT centre → centroid, in the annulus
   plane (an area integral, so a few mm of noise barely turns it).
3. The point = the annulus ring in that direction.
4. The apex: the farthest LV-cavity end from the base (a wide cone toward the
   LV), then valve_auto's epicardial apex read on that ray (_apex_from_click).

Pure numpy / scipy (headless, tools/test_mv_aov_point.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import valve_auto as va

_DEPTH_MM = 4.0          # first LV-side section below the aortic annulus
_HALF_MM = 38.0          # half-size of that section
_STEP_MM = 0.5
_CLOSE_MM = 1.5          # bridges thin leaflets (closing)
_MIN_SHIFT_MM = 3.0      # centroid must sit this far off the LVOT centre
_MAX_DEPTH_MM = 34.0     # deepest section searched for the LV opening
_OPEN_RATIO = 1.6        # section this much wider than the LVOT = the LV
_N_DIRS = 3              # min. agreeing sections for the mitral direction
_CLUSTER_DEG = 10.0      # sections within this angle agree
_AOV_RESEED = 1          # re-runs of detect_aov from its own centre
_AOV_JUMP_MM = 8.0       # a re-run AoV centre farther than this = not trusted
_APEX_CONE_DEG = 60.0    # first, wide apex search around the LV direction


@dataclass
class MvAovResult:
    point: np.ndarray                        # (3,) mm, the transition point
    apex: np.ndarray                         # (3,) mm, epicardial LV apex
    aov_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    aov_normal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mv_dir: np.ndarray = field(default_factory=lambda: np.zeros(3))
    aov: object = None                       # the valve_auto.ValveResult used
    ok: bool = True
    message: str = ""


def _fail(msg, P=None):
    P = np.zeros(3) if P is None else np.asarray(P, float)
    return MvAovResult(P, P, ok=False, message=msg)


def _section_region(vol, dims, q, u, w, thr):
    """(A, B, region) of the blood connected to *q* in the plane (q; u, w)."""
    g = np.arange(-_HALF_MM, _HALF_MM + 1e-6, _STEP_MM)
    A, B = np.meshgrid(g, g, indexing="ij")
    P = q + A[..., None] * u + B[..., None] * w
    blood = va._sample(vol, dims, P) >= thr
    r = max(1, int(round(_CLOSE_MM / _STEP_MM)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    blood = ndimage.binary_closing(blood, structure=(xx ** 2 + yy ** 2) <= r * r)
    lab, _n = ndimage.label(blood)
    i0 = len(g) // 2
    seed = lab[i0, i0]
    if seed == 0:                            # centre not in blood: nearest
        idx = np.argwhere(lab > 0)
        if not len(idx):
            return A, B, None
        k = int(np.argmin(((idx - i0) ** 2).sum(axis=1)))
        if np.hypot(*(idx[k] - i0)) * _STEP_MM > 6.0:
            return A, B, None
        seed = lab[tuple(idx[k])]
    reg = (lab == seed) & (A ** 2 + B ** 2 <= _HALF_MM ** 2)
    return A, B, (reg if reg.sum() >= 20 else None)


def _mitral_dir(vol, dims, c, n, thr, r_ann=11.0):
    """Unit direction (in the annulus plane) from the LVOT centre toward the
    mitral inflow, or None. Sections parallel to the annulus are read from
    _DEPTH_MM down the LV side until the LVOT tube has opened into the LV
    (the section clearly wider than the LVOT, its centroid off the LVOT
    centre); the directions of all such sections down to _MAX_DEPTH_MM are
    combined by consensus."""
    u, w = va._frame(n)
    a0 = None
    dirs = []
    q = c - _DEPTH_MM * n                    # LV side (n points to the aorta)
    for depth in np.arange(_DEPTH_MM, _MAX_DEPTH_MM + 1e-6, 2.0):
        q = va._snap_into_blood(vol, dims, c - depth * n, thr, r_mm=4.0)
        q = q - ((q - c) @ n + depth) * n     # stay on this section
        A, B, reg = _section_region(vol, dims, q, u, w, thr)
        if reg is None:
            continue
        area = float(reg.sum()) * _STEP_MM ** 2
        if a0 is None:
            a0 = area                         # the LVOT (first section)
            continue
        # The LV part only: the LVOT tube (the annulus radius around the AoV
        # axis) left out, so a narrow (systolic) LV base is not outweighed by it.
        off = (q - c) - ((q - c) @ n) * n
        ou, ow = float(off @ u), float(off @ w)
        lv = reg & ((A + ou) ** 2 + (B + ow) ** 2 > (r_ann + 1.0) ** 2)
        if lv.sum() < 20:
            continue
        v = float((A[lv] + ou).mean()) * u + float((B[lv] + ow).mean()) * w
        if area >= _OPEN_RATIO * a0 and np.linalg.norm(v) >= _MIN_SHIFT_MM:
            dirs.append(va._unit(v))
    if len(dirs) < _N_DIRS:
        return None
    # Consensus: the direction with the most others within _CLUSTER_DEG (a
    # section leaking into a neighbouring chamber points elsewhere), averaged
    # over that cluster.
    D = np.asarray(dirs)
    cosd = np.cos(np.radians(_CLUSTER_DEG))
    near = (D @ D.T) >= cosd
    k = int(np.argmax(near.sum(axis=1)))
    if int(near[k].sum()) < _N_DIRS:
        return None
    return va._unit(D[near[k]].sum(axis=0))


def _ring_point(c, n, ring, d):
    """Point of the annulus ring *ring* (N,3) in direction *d* from *c*."""
    u, w = va._frame(n)
    Q = np.asarray(ring, float) - c
    ang = np.arctan2(Q @ w, Q @ u)
    rad = np.linalg.norm(Q - np.outer(Q @ n, n), axis=1)
    o = np.argsort(ang)
    ang, rad = ang[o], rad[o]
    a = float(np.arctan2(d @ w, d @ u))
    r = float(np.interp(a, ang, rad, period=2 * np.pi))
    return c + r * (np.cos(a) * u + np.sin(a) * w)


def _apex(vol, dims, base, lv_dir, thr):
    """Epicardial apex: the farthest LV-cavity end from *base* in a wide cone
    around *lv_dir*, then valve_auto's apex read toward it."""
    step = 0.5
    rs = np.arange(0.0, va._APEX_RMAX_MM + 1e-6, step)
    D = va._dirs_cone(lv_dir, _APEX_CONE_DEG, 400)
    hu = va._sample(vol, dims, base + D[:, None, :] * rs[None, :, None])
    ends = []
    for k in range(len(D)):
        e = va._cavity_end(hu[k], hu[k] >= thr, step)
        if e is not None and e > 0:
            ends.append((rs[e], D[k]))
    if not ends:
        return None
    dv = va._apex_avg_dir([x[0] for x in ends], [x[1] for x in ends])
    e_max = max(x[0] for x in ends)
    res = va._apex_from_click(vol, dims, base, base + e_max * dv, thr)
    return res


def _robust_aov(vol, dims, click):
    """detect_aov made independent of where exactly the user clicked: from the
    click (or, if that fails, seeds 5 mm around it), then RE-RUN from the found
    annulus centre (_AOV_RESEED rounds). A centre that wanders
    > _AOV_JUMP_MM from the previous one is not trusted."""
    seeds = [click] + [click + 5.0 * np.asarray(e, float) for e in (
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    res = None
    for sd in seeds:
        r = va.detect_aov(vol, dims, sd)
        if r.ok and len(r.points) >= 8:
            res = r
            break
    if res is None:
        return None
    for _ in range(_AOV_RESEED):
        r = va.detect_aov(vol, dims, res.center)
        if not r.ok or len(r.points) < 8:
            break
        jump = float(np.linalg.norm(r.center - res.center))
        if jump > _AOV_JUMP_MM:
            break
        res = r
        if jump < 0.5:
            break
    return res


def detect_mv_aov_point(vol, dims, click_mm) -> MvAovResult:
    """The MV-AoV transition point + the LV apex from a rough click near the
    aorto-mitral continuity (see the module docstring)."""
    vol = np.asarray(vol)
    click = np.asarray(click_mm, float)
    aov = _robust_aov(vol, dims, click)
    if aov is None:
        return _fail("aortic annulus not found", click)
    c, n = np.asarray(aov.center, float), va._unit(aov.normal)
    thr = va.blood_threshold(vol, dims, c)
    d = _mitral_dir(vol, dims, c, n, thr, float(aov.radius))
    if d is None:
        return _fail("mitral side not found", click)
    P = _ring_point(c, n, aov.points, d)
    base = va._snap_into_blood(vol, dims, c - 12.0 * n, thr)
    ap = _apex(vol, dims, base, -n, thr)
    if ap is None or not ap.ok:
        return MvAovResult(P, P, c, n, d, aov=aov, ok=False,
                           message=(getattr(ap, "message", "") or "LV apex not found"))
    return MvAovResult(P, np.asarray(ap.apex, float), c, n, d, aov=aov,
                       message=ap.message)
