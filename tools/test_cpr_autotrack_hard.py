# -*- coding: utf-8 -*-
"""Harder synthetic cases for core/cpr_autotrack.py: a tapering vessel
(r 2.2 → 0.8 mm), a side branch that must not be followed, and a vessel
running only ~1.5 mm from a bright chamber wall.

Run: python tools/test_cpr_autotrack_hard.py
"""
import pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from multi_dicomviewer.core.cpr_autotrack import trace_vessel

fails = []
def chk(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), name, extra)
    if not cond:
        fails.append(name)

rng = np.random.default_rng(1)
dims = (0.4, 0.4, 0.5)
nz, ny, nx = 180, 220, 220
sx, sy, sz = dims
zz, yy, xx = np.meshgrid(np.arange(nz) * sz, np.arange(ny) * sy,
                         np.arange(nx) * sx, indexing="ij")
P = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
vol = np.full(P.shape[0], -90.0, np.float32)

# Chamber: a big contrast slab-like box face at x >= 60 mm (wall plane x=60).
vol[P[:, 0] >= 60.0] = 340.0
vol[(P[:, 0] >= 56.0) & (P[:, 0] < 60.0)] = 340.0   # contrast up to the wall
# Main vessel: S-curve in y-z at x ~ 52 → approaching x = 54.3 (1.7 mm gap
# from the 56 mm contrast edge for r ~1) in its middle part.
t = np.linspace(0, 1, 700)
main = np.column_stack([50.0 + 4.3 * np.sin(np.pi * t),
                        15.0 + 60.0 * t,
                        30.0 + 12.0 * np.sin(2 * np.pi * t)])
r_main = 2.2 - 1.4 * t                                    # taper 2.2 → 0.8 mm
# Side branch leaving at t=0.35 towards -x.
j0 = int(0.35 * 699)
tb = np.linspace(0, 1, 300)
branch = main[j0] + np.column_stack([-25.0 * tb, 6.0 * tb, 3.0 * tb])
r_br = np.full(300, 1.2)

def paint(cl, rr, hu):
    tr = cKDTree(cl)
    d, i = tr.query(P, distance_upper_bound=3.0)
    ok = np.isfinite(d)
    m = np.zeros(P.shape[0], bool)
    m[ok] = d[ok] < rr[i[ok]]
    vol[m] = hu
paint(main, r_main, 390.0)
paint(branch, r_br, 390.0)
vol = vol.reshape(nz, ny, nx)
vol = ndimage.gaussian_filter(vol, 0.6) + rng.normal(0, 25, vol.shape)
vol = vol.astype(np.int16)

tree = cKDTree(main)
p0 = main[4] + [0.5, 0.5, -0.4]
p1 = main[-5] + [-0.3, 0.4, 0.3]
t0 = time.time()
res = trace_vessel(vol, dims, [p0, p1])
dt = time.time() - t0
d, _ = tree.query(res.path_mm)
print(f"   time {dt:.1f}s len {res.length_mm:.1f} err mean {d.mean():.2f} "
      f"p95 {np.percentile(d, 95):.2f} max {d.max():.2f} gaps {res.gaps}")
chk("hard: ok", res.ok, res.message)
chk("hard: mean error < 0.5 mm", d.mean() < 0.5, f"{d.mean():.2f}")
chk("hard: max error < 1.5 mm (no branch / chamber detour)", d.max() < 1.5,
    f"{d.max():.2f}")
chk("hard: never in the chamber (x < 56)", res.path_mm[:, 0].max() < 56.0,
    f"max x {res.path_mm[:, 0].max():.1f}")
chk("hard: no gaps reported", len(res.gaps) == 0, f"{res.gaps}")

# Branch tracing: from the main origin into the branch end.
trb = cKDTree(np.vstack([main[:j0 + 1], branch]))
res_b = trace_vessel(vol, dims, [p0, branch[-5]])
db, _ = trb.query(res_b.path_mm)
chk("branch: follows main then branch", res_b.ok and db.max() < 1.5,
    f"mean {db.mean():.2f} max {db.max():.2f}")

print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
