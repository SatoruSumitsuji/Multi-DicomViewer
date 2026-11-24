# -*- coding: utf-8 -*-
"""一括描出 core: trace_to_tree (user 2026-11-25). A short line at a
branch's distal end is traced PROXIMALLY until it meets the tree (stops
~1–2 mm short of the branch point); the seed may be given in either order;
an occluded branch stops on its way (not reached); at a confluence with a
vessel not yet drawn the WIDER (proximal) one is followed to the tree.
Synthetic tubes, no patient data.

    python tools/test_cpr_trace_to_tree.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from multi_dicomviewer.core import cpr_autoextend as CAE  # noqa: E402

FAILS = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        FAILS.append(name)


h = 0.4
n = 150
z, y, x = (np.mgrid[0:n // 3, 0:n, 0:n].astype(np.float32) * h)
BG = -60.0
dims = (h, h, h)


def seg(vol, a, b, r, hu):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    L2 = float(ab @ ab)
    t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]
                 + (z - a[2]) * ab[2]) / L2, 0, 1)
    d2 = ((x - a[0] - t * ab[0]) ** 2 + (y - a[1] - t * ab[1]) ** 2
          + (z - a[2] - t * ab[2]) ** 2)
    rr = r[0] + (r[1] - r[0]) * t if isinstance(r, tuple) else r
    vol[d2 <= rr * rr] = hu


def finish(vol, seed=0):
    vol = ndimage.gaussian_filter(vol, 0.5 / h)
    return vol + np.random.default_rng(seed).normal(0, 15, vol.shape).astype(
        np.float32)


ZC = 9.0
PARENT = np.c_[np.linspace(4, 56, 105), np.full(105, 10.0), np.full(105, ZC)]


def line_pts(a, b, k=40):
    return np.linspace(np.asarray(a, float), np.asarray(b, float), k)


# --- 1. a branch leaving the parent at (20, 10) going to (45, 50) ----------
vol = np.full(x.shape, BG, np.float32)
seg(vol, PARENT[0], PARENT[-1], 1.8, 420.0)
J = np.array([20.0, 10.0, ZC])
E = np.array([45.0, 50.0, ZC])
seg(vol, J, E, (1.4, 0.8), 400.0)
V1 = finish(vol)
u = (E - J) / np.linalg.norm(E - J)
seed = np.array([E - 7.0 * u, E - 3.5 * u, E - 0.5 * u])   # proximal → distal
r = CAE.trace_to_tree(V1, dims, seed, [PARENT])
d_j = float(np.linalg.norm(r.ctrl[0] - J))
check("reached the parent line", r.reached and r.stop == "reached"
      and r.target == 0, (r.reached, r.stop, r.dist_mm))
check("stops short of the branch point, not inside the parent",
      1.6 <= r.dist_mm <= 4.5 and d_j < 6.0, (r.dist_mm, d_j))
off = max(float(np.linalg.norm(np.cross(p - J, u))) for p in r.ctrl)
check("on the branch all the way (≤ 0.6 mm off)", off <= 0.6, off)
check("the line comes out proximal → distal, the seed kept at its end",
      np.allclose(r.ctrl[-3:], seed) and r.added_mm > 30.0,
      (r.ctrl[-1], r.added_mm))
check("junction = the nearest parent sample",
      abs(PARENT[r.junction][0] - r.ctrl[0][0]) < 3.5,
      (PARENT[r.junction], r.ctrl[0]))
r2 = CAE.trace_to_tree(V1, dims, seed[::-1], [PARENT])
check("the seed drawn distal → proximal gives the same line",
      r2.reached and np.allclose(r2.ctrl, r.ctrl, atol=1e-6))

# --- 2. occluded on its way: stops, not reached -----------------------------
vol = np.full(x.shape, BG, np.float32)
seg(vol, PARENT[0], PARENT[-1], 1.8, 420.0)
M = J + 0.45 * (E - J)
seg(vol, J, M - 2.5 * u, 1.3, 400.0)
seg(vol, M + 2.5 * u, E, (1.1, 0.8), 400.0)                 # a 5 mm gap
r = CAE.trace_to_tree(finish(vol, 1), dims, seed, [PARENT])
check("an occlusion: stopped on its way, not reached",
      not r.reached and r.target == -1
      and np.linalg.norm(r.ctrl[0] - M) < 6.0, (r.stop, r.ctrl[0]))

# --- 3. joins a vessel not drawn yet: the wider way leads to the tree ------
#   K (not drawn): from the parent at (36, 10) up to (36, 55), tapering;
#   the seeded branch joins K at (36, 34) from the side.
vol = np.full(x.shape, BG, np.float32)
seg(vol, PARENT[0], PARENT[-1], 1.8, 420.0)
K0, K1 = np.array([36.0, 10.0, ZC]), np.array([36.0, 56.0, ZC])
seg(vol, K0, K1, (1.6, 0.9), 410.0)
B0, B1 = np.array([36.0, 34.0, ZC]), np.array([12.0, 52.0, ZC])
seg(vol, B0, B1, (1.0, 0.75), 400.0)
ub = (B1 - B0) / np.linalg.norm(B1 - B0)
sb = np.array([B1 - 6.0 * ub, B1 - 3.0 * ub, B1 - 0.3 * ub])
r = CAE.trace_to_tree(finish(vol, 2), dims, sb, [PARENT])
ys = r.ctrl[:, 1]
check("at the confluence the wider (proximal) way is taken → the tree",
      r.reached and r.ctrl[0][1] < 15.0 and ys.max() < 54.0,
      (r.stop, r.ctrl[0], ys.max()))

# --- 4. already at the tree: nothing traced ---------------------------------
near = np.array([J + 1.0 * u, J + 4.0 * u, J + 7.0 * u])
r = CAE.trace_to_tree(V1, dims, near, [PARENT])
check("a seed touching the tree: reached at once, nothing added",
      r.reached and r.added_mm == 0.0)
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
sys.stdout.flush()
os._exit(1 if FAILS else 0)
