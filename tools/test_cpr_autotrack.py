# -*- coding: utf-8 -*-
"""Headless checks for core/cpr_autotrack.py on a synthetic coronary phantom.

The phantom is a contrast tube (r 1.6 mm, ~380 HU) running in fat over the
surface of a bright "chamber" sphere (~350 HU) inside myocardium, with a
low-HU stenosis, a calcium nodule beside the lumen and image noise. The straight
line between the end points crosses the chamber, so a short-cut would show.

Run: python tools/test_cpr_autotrack.py
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import ndimage

from multi_dicomviewer.core.cpr_autotrack import TraceParams, trace_vessel

fails = []


def chk(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), name, extra)
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- phantom
rng = np.random.default_rng(0)
dims = (0.45, 0.45, 0.6)                       # sx, sy, sz mm (anisotropic z)
nz, ny, nx = 150, 200, 200
sx, sy, sz = dims
zz, yy, xx = np.meshgrid(np.arange(nz) * sz, np.arange(ny) * sy,
                         np.arange(nx) * sx, indexing="ij")
vol = np.full((nz, ny, nx), -100.0, np.float32)            # epicardial fat

C = np.array([45.0, 45.0, 45.0])                           # chamber centre x,y,z
R_ch = 20.0
r_c = np.sqrt((xx - C[0]) ** 2 + (yy - C[1]) ** 2 + (zz - C[2]) ** 2)
vol[r_c < R_ch + 8.0] = 60.0                               # myocardium shell
vol[r_c < R_ch] = 350.0                                    # contrast chamber

# Vessel: an arc over the chamber at radius R_ch + 11 mm (in the fat just
# outside the myocardium), 150 degrees in the x-y plane, rising in z.
R_v = R_ch + 11.0
th = np.linspace(np.deg2rad(-75), np.deg2rad(75), 600)
centre = np.column_stack([C[0] + R_v * np.cos(th), C[1] + R_v * np.sin(th),
                          C[2] - 6.0 + 12.0 * (th - th[0]) / (th[-1] - th[0])])
arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centre, axis=0),
                                                      axis=1))])
r_lumen = 1.6
# Distance of every voxel to the centreline polyline (sampled points; fine).
from scipy.spatial import cKDTree
tree = cKDTree(centre)
pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
near = r_c.ravel() > R_ch + 6.0                 # only fat region can hold the vessel
dist = np.full(pts.shape[0], 99.0)
dist[near], nn = tree.query(pts[near], distance_upper_bound=4.0)
nn_full = np.zeros(pts.shape[0], int)
nn_full[near] = np.minimum(nn, len(centre) - 1)
dist = dist.reshape(vol.shape)
nnv = nn_full.reshape(vol.shape)
lumen = dist < r_lumen
s_at = arc[nnv]
sten = lumen & (s_at > 0.55 * arc[-1]) & (s_at < 0.55 * arc[-1] + 3.0)
vol[lumen] = 380.0
vol[sten] = 60.0                                            # 3 mm soft-plaque gap
# Calcium nodule touching the lumen at 30 % length.
i_ca = int(np.searchsorted(arc, 0.30 * arc[-1]))
Pca = centre[i_ca] + np.array([0.0, 0.0, 2.4])
vol[np.sqrt((xx - Pca[0]) ** 2 + (yy - Pca[1]) ** 2
            + (zz - Pca[2]) ** 2) < 1.4] = 1100.0
vol = ndimage.gaussian_filter(vol, 0.6)                     # scanner blur
vol += rng.normal(0, 25, vol.shape).astype(np.float32)      # noise
vol = vol.astype(np.int16)

# Clicks: slightly off the true centre (as a user would click).
p_start = centre[5] + np.array([0.7, -0.5, 0.4])
p_end = centre[-6] + np.array([-0.6, 0.6, -0.5])

# Sanity: the straight line crosses the chamber (else the test proves nothing).
seg = np.linspace(p_start, p_end, 200)
chk("phantom: straight line crosses the chamber",
    np.min(np.linalg.norm(seg - C, axis=1)) < R_ch,
    f"min dist to C {np.min(np.linalg.norm(seg - C, axis=1)):.1f} mm")

# ---------------------------------------------------------------- trace
t0 = time.time()
res = trace_vessel(vol, dims, [p_start, p_end])
dt = time.time() - t0
chk("trace: ok", res.ok, res.message)
d_true, _ = tree.query(res.path_mm)
print(f"   time {dt:.1f}s, length {res.length_mm:.1f} mm (true "
      f"{arc[-6] - arc[5]:.1f}), ctrl {len(res.ctrl_mm)}, "
      f"err mean {d_true.mean():.2f} / p95 {np.percentile(d_true, 95):.2f} / "
      f"max {d_true.max():.2f} mm, gaps {res.gaps}")
chk("trace: never enters the chamber",
    np.min(np.linalg.norm(res.path_mm - C, axis=1)) > R_ch,
    f"min {np.min(np.linalg.norm(res.path_mm - C, axis=1)):.1f} mm")
chk("trace: mean error < 0.5 mm", d_true.mean() < 0.5, f"{d_true.mean():.2f}")
chk("trace: max error < 1.2 mm", d_true.max() < 1.2, f"{d_true.max():.2f}")
chk("trace: length within 5 % of the true arc",
    abs(res.length_mm - (arc[-6] - arc[5])) < 0.05 * (arc[-6] - arc[5]))
chk("trace: under 30 s", dt < 30.0, f"{dt:.1f}s")
s_sten = 0.55 * arc[-1] - arc[5]
chk("gaps: the 3 mm stenosis is reported",
    any(a - 2.0 <= s_sten + 1.5 <= b + 2.0 for a, b in res.gaps), f"{res.gaps}")
chk("gaps: no spurious gaps (<= 2)", len(res.gaps) <= 2, f"{res.gaps}")
step = np.linalg.norm(np.diff(res.ctrl_mm, axis=0), axis=1)
chk("ctrl: spacing ~5 mm (last may be shorter)",
    np.all(step[:-1] > 4.0) and np.all(step < 6.0), f"{step.round(1)}")
chk("ctrl: starts and ends at the path ends",
    np.allclose(res.ctrl_mm[0], res.path_mm[0])
    and np.allclose(res.ctrl_mm[-1], res.path_mm[-1]))

# ---------------------------------------------------------------- via point
mid = centre[300] + np.array([0.5, 0.5, 0.0])
res2 = trace_vessel(vol, dims, [p_start, mid, p_end])
d2, _ = tree.query(res2.path_mm)
chk("via point: ok and accurate", res2.ok and d2.mean() < 0.5,
    f"mean {d2.mean():.2f} max {d2.max():.2f}")

# ---------------------------------------------------------------- degenerate
chk("one point → not ok", not trace_vessel(vol, dims, [p_start]).ok)

print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
