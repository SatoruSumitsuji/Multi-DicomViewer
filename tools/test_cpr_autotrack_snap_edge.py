# -*- coding: utf-8 -*-
"""Draw ▸ Auto (core/cpr_autotrack._snap): a click near the FAR edge of the
traced region no longer fails with "operands could not be broadcast"
(user log 2026-11-25) — the search block is clipped at both ends.

    python tools/test_cpr_autotrack_snap_edge.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from multi_dicomviewer.core.cpr_autotrack import _snap  # noqa: E402

fails = []
L = np.zeros((20, 8, 20), np.float32)
L[17, 6, 18] = 1.0                                # the lumen near the corner
ok = True
for k in ((19, 7, 19), (0, 0, 0), (19, 0, 10), (10, 7, 0), (10, 4, 10)):
    try:
        j = _snap(L, k, 4, 2)
        ok &= all(0 <= int(a) < n for a, n in zip(j, L.shape))
    except Exception as exc:                      # noqa: BLE001
        print("FAIL", k, exc)
        ok = False
print(("PASS " if ok else "FAIL ") + "a click at any edge / corner is handled")
if not ok:
    fails.append(1)
j = _snap(L, (19, 7, 19), 4, 2)
good = tuple(int(a) for a in j) == (17, 6, 18)
print(("PASS " if good else "FAIL ") + "… and still moves to the lumen there", j)
if not good:
    fails.append(2)
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
os._exit(1 if fails else 0)
