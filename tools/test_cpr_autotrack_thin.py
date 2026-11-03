# -*- coding: utf-8 -*-
"""Thin distal vessel accuracy for core/cpr_autotrack.py — the case users hit
on real CT ("細い部分の精度が著しく低下").

Phantom: an artery tapering r 1.8 → 0.5 mm whose contrast fades 430 → 250 HU
distally, with a tortuous distal course (±1.5 mm wiggle, 9 mm period), a
contrast vein (r 1.2 mm, 300 HU) running parallel (walls ≥ ~1 mm apart) along the
distal half, scanner blur (PSF σ 0.55 mm) and noise (σ 30 HU) on a 0.39 ×
0.39 × 0.5 mm grid.

What is scored is what the user sees: the CPR centreline that
CenterLine.from_points builds from the returned control points, compared
with the true centreline, separately for the THIN part (r < 0.8 mm).

Run: python tools/test_cpr_autotrack_thin.py
"""
import pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from multi_dicomviewer.core.centerline import CenterLine
from multi_dicomviewer.core.cpr_autotrack import trace_vessel

fails = []
def chk(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), name, extra)
    if not cond:
        fails.append(name)

rng = np.random.default_rng(7)
dims = (0.39, 0.39, 0.5)
sx, sy, sz = dims
nz, ny, nx = 140, 240, 200
zz, yy, xx = np.meshgrid(np.arange(nz) * sz, np.arange(ny) * sy,
                         np.arange(nx) * sx, indexing="ij")
P = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
vol = np.full(P.shape[0], -80.0, np.float32)

# Artery: along +y from y=8 to y=85 mm, gentle bend in x/z, distal wiggle.
t = np.linspace(0, 1, 1600)
y = 8.0 + 77.0 * t
wig = np.where(t > 0.55, 1.5 * np.sin(2 * np.pi * (y - y[0]) / 9.0), 0.0)
art = np.column_stack([38.0 + 6.0 * np.sin(np.pi * t) + wig,
                       y,
                       35.0 + 4.0 * np.sin(1.3 * np.pi * t)])
r_art = 1.8 - 1.3 * t                         # 1.8 → 0.5 mm
hu_art = 430.0 - 180.0 * t                    # 430 → 250 HU
# Vein: parallel to the distal half, 4.5 mm to the -x side of the
# NON-wiggled course; the wiggle brings the walls to ~1 mm apart.
tv = t[t > 0.45]
vein = np.column_stack([38.0 + 6.0 * np.sin(np.pi * tv) - 4.5,
                        8.0 + 77.0 * tv,
                        35.0 + 4.0 * np.sin(1.3 * np.pi * tv)])

def paint(cl, rr, hu):
    tr = cKDTree(cl)
    d, i = tr.query(P, distance_upper_bound=3.0)
    ok = np.isfinite(d)
    m = np.zeros(P.shape[0], bool)
    m[ok] = d[ok] < np.broadcast_to(rr, (len(cl),))[i[ok]]
    hv = np.broadcast_to(hu, (len(cl),))
    vol[m] = hv[i[m]]
paint(vein, np.full(len(vein), 1.2), np.full(len(vein), 300.0))
paint(art, r_art, hu_art)
vol = vol.reshape(nz, ny, nx)
psf = 0.55
vol = ndimage.gaussian_filter(vol, (psf / sz, psf / sy, psf / sx))
vol = (vol + rng.normal(0, 30, vol.shape)).astype(np.int16)

tree = cKDTree(art)
thin = r_art < 0.8
tree_thin = cKDTree(art[thin])
p0 = art[10] + [0.5, 0.4, -0.4]
p1 = art[-10] + [-0.3, 0.3, 0.3]

t0 = time.time()
res = trace_vessel(vol, dims, [p0, p1])
dt = time.time() - t0
chk("thin: ok", res.ok, res.message)
cl = CenterLine.from_points(res.ctrl_mm, step_mm=0.2)
d_all, _ = tree.query(cl.points)
# Thin part: centreline samples whose nearest true point is in the thin part.
_, near = tree.query(cl.points)
in_thin = thin[near]
d_thin = d_all[in_thin]
print(f"   time {dt:.1f}s, ctrl {len(res.ctrl_mm)}, len {cl.length_mm:.1f}; "
      f"all mean {d_all.mean():.2f} max {d_all.max():.2f}; thin mean "
      f"{d_thin.mean():.2f} p95 {np.percentile(d_thin, 95):.2f} max "
      f"{d_thin.max():.2f} mm; gaps {res.gaps}")
chk("thin: whole-vessel mean < 0.25 mm", d_all.mean() < 0.25, f"{d_all.mean():.2f}")
chk("thin: thin-part mean < 0.25 mm", d_thin.mean() < 0.25, f"{d_thin.mean():.2f}")
chk("thin: thin-part max < 0.6 mm (no vein jump, follows wiggle)",
    d_thin.max() < 0.6, f"{d_thin.max():.2f}")
chk("thin: reaches the distal end", d_all[-1] < 0.6, f"{d_all[-1]:.2f}")
chk("thin: under 30 s", dt < 30, f"{dt:.1f}s")

print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
