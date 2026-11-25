# -*- coding: utf-8 -*-
"""Territory of a 「枝」 target (user 2026-11-25): a branch drawn / batch-
traced from its DISTAL end is stored distal→proximal, and its territory came
out on the PROXIMAL side of the target (target → the branch point). Now a
branch's direction is read from its junction on the parent; LM / LAD / LCX /
RCA keep the old proximal→distal rule. Synthetic lines only.

    python tools/test_territory_branch_direction.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from multi_dicomviewer.core.coronary_territory import CoronaryTree  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


def line(a, b, n=41):
    return np.linspace(np.asarray(a, float), np.asarray(b, float), n)


def build(branch_reversed: bool):
    t = CoronaryTree()
    t.add_root("lad", "LAD", "LAD", line((0, 0, 0), (40, 0, 0)))  # ostium at x=0
    pts = line((20, 0, 0), (20, 40, 0))       # branch off LAD at x=20, toward +y
    if branch_reversed:
        pts = pts[::-1]                       # drawn from the distal end
    t.add_vessel("d1", "D1", "branch", pts)
    t.vessels["d1"].parent = "lad"
    t.vessels["d1"].junction = 20
    # a sub-branch off D1 at y=30 (distal of a target at y=20)
    sub = line((20, 30, 0), (35, 30, 0))
    t.add_vessel("s1", "S1", "branch", sub)
    t.vessels["s1"].parent = "d1"
    t.vessels["s1"].junction = 10 if branch_reversed else 30
    return t


# voxels: one next to every sample of D1 (y = 0…40) and of S1
vox = np.array([[21.0, y, 0] for y in np.linspace(1, 40, 40)]
               + [[x, 31.0, 0] for x in np.linspace(22, 35, 10)])
for rev in (False, True):
    t = build(rev)
    tag = "reversed" if rev else "proximal-first"
    check(f"{tag}: direction detected", t.runs_reversed("d1") == rev)
    a = t.assign(vox)
    n = t.vessels["d1"].n
    target = 20 if not rev else n - 1 - 20         # the sample at y = 20
    m = t.territory_mask(a, "d1", target)
    ys = vox[:40][m[:40], 1]
    check(f"{tag}: territory is DISTAL of the target (y ≥ 20)",
          len(ys) > 0 and ys.min() >= 19.0 and ys.max() >= 39.0,
          (ys.min() if len(ys) else None, ys.max() if len(ys) else None))
    check(f"{tag}: the sub-branch beyond the target is included",
          bool(m[40:].all()))
    whole = t.territory_mask(a, "d1", t.proximal_idx("d1"))
    check(f"{tag}: whole branch from its proximal end", bool(whole.all()))
    check(f"{tag}: LAD keeps the old rule (idx 0 = ostium)",
          not t.runs_reversed("lad") and t.proximal_idx("lad") == 0)
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
