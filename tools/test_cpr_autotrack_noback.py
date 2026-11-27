# -*- coding: utf-8 -*-
"""Auto-Track between clicked points never turns back and never balloons
(user 2026-11-27: "CPR線が戻る" / "CPR線が異常に長くなる" forbidden — a true
hairpin is set by hand later). Synthetic tubes, no patient data.

Case 1 (戻る): the straight lumen between the clicks has a gap; a bright
detour runs PAST the end click and comes back to it. The old search took
the detour (backtracking); now the path keeps going forward.
Case 2 (異常に長い): the only bright route bulges far sideways; the line stays
near the chord, within the length limit.

    python tools/test_cpr_autotrack_noback.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from multi_dicomviewer.core.cpr_autotrack import (TraceParams,  # noqa: E402
                                                  trace_vessel)

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


SP = 0.5
dims = (SP, SP, SP)
NZ, NY, NX = 40, 90, 130


def tube_vol(polylines, r=1.2):
    vol = np.full((NZ, NY, NX), 40, np.int16)
    zz, yy, xx = np.meshgrid(np.arange(NZ) * SP, np.arange(NY) * SP,
                             np.arange(NX) * SP, indexing="ij")
    W = np.stack([xx, yy, zz], -1)
    for pl in polylines:
        pl = np.asarray(pl, float)
        for a, b in zip(pl[:-1], pl[1:]):
            d = b - a
            t = np.clip(((W - a) @ d) / float(d @ d), 0.0, 1.0)
            dist = np.linalg.norm(W - (a + t[..., None] * d), axis=-1)
            vol[dist <= r] = 400
    return vol


def back_max(path, p0, p1):
    """Largest retreat along the chord (mm) — 0 for a forward-only line."""
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    s = (np.asarray(path) - p0) @ u
    return float(np.max(np.maximum.accumulate(s) - s))


def length(path):
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


z = 10.0
# a wide ROI so the detours are reachable at all (else the box hides them)
WIDE = TraceParams(margin_mm=30.0)
p0 = np.array([8.0, 15.0, z])
p1 = np.array([34.0, 15.0, z])
# case 1: lumen 8→22 and 27→34 (gap 22..27), detour 22 → past 34 (to 52) → 34
vol1 = tube_vol([
    [[6, 15, z], [22, 15, z]],
    [[27, 15, z], [36, 15, z]],
    [[22, 15, z], [26, 22, z], [52, 22, z], [52, 15, z], [36, 15, z]],
])
old = trace_vessel(vol1, dims, [p0, p1], TraceParams(no_backtrack=False, margin_mm=30.0))
check("(test valid) the old search backtracks through the detour",
      old.ok and back_max(old.path_mm, p0, p1) > 8.0,
      back_max(old.path_mm, p0, p1))
new = trace_vessel(vol1, dims, [p0, p1], WIDE)
bm = back_max(new.path_mm, p0, p1)
check("戻る: the line never turns back (≤ 1 mm)", new.ok and bm <= 1.0, bm)
L = length(new.path_mm)
ch = float(np.linalg.norm(p1 - p0))
check("… and is not long (≤ 1.3 × the chord)", L <= 1.3 * ch, (L, ch))
check("… ends at the clicks",
      np.linalg.norm(new.path_mm[0] - p0) < 2.0
      and np.linalg.norm(new.path_mm[-1] - p1) < 2.0)

# case 2: lumen 8→14 and 28→34, the only bright route bulges 25 mm sideways
vol2 = tube_vol([
    [[6, 15, z], [14, 15, z]],
    [[28, 15, z], [36, 15, z]],
    [[14, 15, z], [14, 40, z], [28, 40, z], [28, 15, z]],
])
old2 = trace_vessel(vol2, dims, [p0, p1], TraceParams(no_backtrack=False, margin_mm=30.0))
check("(test valid) the old search takes the long bulge",
      old2.ok and length(old2.path_mm) > 2.0 * ch, length(old2.path_mm))
new2 = trace_vessel(vol2, dims, [p0, p1], WIDE)
L2 = length(new2.path_mm)
check("異常に長い: the line stays within the limit (1.8 × chord or +10 mm)",
      new2.ok and L2 <= max(1.8 * ch, ch + 10.0), (L2, ch))
check("… never turns back", back_max(new2.path_mm, p0, p1) <= 1.0)

# a normal curved vessel (quarter-circle-ish arc) is still followed
arc = [[8 + 13 * (1 - np.cos(a)), 15 + 13 * np.sin(a), z]
       for a in np.linspace(0, np.pi, 25)]
vol3 = tube_vol([arc])
q0, q1 = np.array(arc[0]), np.array(arc[-1])
r3 = trace_vessel(vol3, dims, [q0, q1])
A = np.asarray(arc)
dev = max(float(np.min(np.linalg.norm(A - p, axis=1))) for p in r3.path_mm)
check("a half-circle vessel (no turning back) is still followed (≤ 1.5 mm)",
      r3.ok and dev <= 1.5, dev)

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
