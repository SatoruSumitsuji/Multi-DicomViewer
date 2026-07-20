# -*- coding: utf-8 -*-
"""Headless geometry checks for core/centerline.py — no GUI, no VTK.

Run: python tools/test_centerline.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from multi_dicomviewer.core.centerline import CenterLine

fails = []


def chk(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), name, ("" if cond else extra))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- straight line
# A straight segment along +x from 0..100mm. Everything must be exact.
cl = CenterLine.from_points([[0, 0, 0], [100, 0, 0]], step_mm=1.0)
chk("straight: sample count", cl.n == 101, f"n={cl.n}")
chk("straight: length", abs(cl.length_mm - 100.0) < 1e-6, f"L={cl.length_mm}")
d = np.diff(cl.arclen)
chk("straight: uniform 1mm spacing", np.allclose(d, 1.0, atol=1e-6),
    f"spacings {d.min():.4f}..{d.max():.4f}")
chk("straight: tangent = +x", np.allclose(cl.tangents, [1, 0, 0], atol=1e-6))
u, v = cl.frames(ref_up=[0, 0, 1])            # through-plane = +z
# plane ⟂ +x is spanned by y,z; u should be ±z (ref_up), v ⟂ both
chk("straight: u ⟂ tangent", np.allclose(u @ [1, 0, 0], 0, atol=1e-6))
chk("straight: v ⟂ tangent", np.allclose(v @ [1, 0, 0], 0, atol=1e-6))
chk("straight: u,v unit", np.allclose(np.linalg.norm(u, axis=1), 1, atol=1e-6)
    and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-6))
chk("straight: u,v orthogonal", np.allclose(np.einsum("ij,ij->i", u, v), 0, atol=1e-6))
chk("straight: u aligns ref_up +z", np.allclose(u, [0, 0, 1], atol=1e-6),
    f"u0={u[0]}")
chk("straight: no twist (u constant)",
    np.allclose(u, u[0], atol=1e-6) and np.allclose(v, v[0], atol=1e-6))

# ---------------------------------------------------------------- quarter circle
# Planar arc radius R in the x-y plane: (R cosθ, R sinθ, 0), θ 0..90°.
R = 50.0
th = np.linspace(0, np.pi / 2, 9)
ctrl = np.c_[R * np.cos(th), R * np.sin(th), np.zeros_like(th)]
cl = CenterLine.from_points(ctrl, step_mm=0.5)
chk("arc: length ≈ πR/2", abs(cl.length_mm - (np.pi * R / 2)) < 0.5,
    f"L={cl.length_mm:.3f} vs {np.pi*R/2:.3f}")
d = np.diff(cl.arclen)
chk("arc: uniform spacing", d.std() < 1e-6, f"std={d.std():.2e}")
chk("arc: tangents unit", np.allclose(np.linalg.norm(cl.tangents, axis=1), 1, atol=1e-6))
# tangent ⟂ radius for a circle centred at origin. Interior tangents are
# full 2nd-order central differences → near-exact; the two endpoints use
# one-sided data (only one neighbour exists) so a few-degree error is
# expected and clinically irrelevant (the trace ends).
rad = cl.points / np.linalg.norm(cl.points, axis=1, keepdims=True)
dots = np.abs(np.einsum("ij,ij->i", rad, cl.tangents))
chk("arc: tangent ⟂ radius (interior)", dots[3:-3].max() < 0.04,
    f"interior max|dot|={dots[3:-3].max():.4f}")
chk("arc: tangent ⟂ radius (endpoints)", dots.max() < 0.12,
    f"endpoint max|dot|={dots.max():.4f}")
u, v = cl.frames(ref_up=[0, 0, 1])
# plane ⟂ tangent for a planar curve with ref +z: one axis stays ~+z (through
# plane), the other lies in-plane (x-y). u ⟂ tangent everywhere.
chk("arc: u ⟂ tangent", np.abs(np.einsum("ij,ij->i", u, cl.tangents)).max() < 1e-6)
chk("arc: v ⟂ tangent", np.abs(np.einsum("ij,ij->i", v, cl.tangents)).max() < 1e-6)
chk("arc: frames orthonormal",
    np.allclose(np.einsum("ij,ij->i", u, v), 0, atol=1e-6)
    and np.allclose(np.linalg.norm(u, axis=1), 1, atol=1e-6))
# one of the two axes should stay close to the through-plane +z (small twist)
zalign = np.maximum(np.abs(u @ [0, 0, 1]), np.abs(v @ [0, 0, 1]))
chk("arc: an axis tracks through-plane +z", zalign.min() > 0.99,
    f"min z-alignment {zalign.min():.4f}")

# ---------------------------------------------------------------- helix (3-D, twist)
tt = np.linspace(0, 4 * np.pi, 40)
helix = np.c_[10 * np.cos(tt), 10 * np.sin(tt), 2.0 * tt]
cl = CenterLine.from_points(helix, step_mm=0.5)
u, v = cl.frames(ref_up=[0, 0, 1])
chk("helix: frames ⟂ tangent",
    np.abs(np.einsum("ij,ij->i", u, cl.tangents)).max() < 1e-6
    and np.abs(np.einsum("ij,ij->i", v, cl.tangents)).max() < 1e-6)
chk("helix: frames orthonormal",
    np.allclose(np.einsum("ij,ij->i", u, v), 0, atol=1e-6))
# RMF: consecutive u should rotate smoothly, never flip (dot>0 between steps)
consec = np.einsum("ij,ij->i", u[:-1], u[1:])
chk("helix: RMF no flips (smooth u)", consec.min() > 0.9,
    f"min consec dot {consec.min():.4f}")

# ---------------------------------------------------- cross-section sampling test
# Synthetic volume: a bright cylinder of radius 4mm along +z through (30,30).
# A centreline straight up the cylinder axis → every short-axis frame samples
# a disc; the bright area must be ~constant (πr²) and centred.
NZ, NY, NX = 60, 60, 60
zz, yy, xx = np.mgrid[0:NZ, 0:NY, 0:NX].astype(np.float64)
cyl = (((xx - 30) ** 2 + (yy - 30) ** 2) <= 16).astype(np.float32)  # r=4 vox
vol = cyl  # 1mm isotropic assumed for the test


def sample(volume, origin, u, v, half=12, ns=49):
    """Trilinear-sample a short-axis plane; origin/u/v in (x,y,z) mm=vox."""
    gs = np.linspace(-half, half, ns)
    gu, gv = np.meshgrid(gs, gs)
    P = (origin[None, None, :]
         + gu[..., None] * u[None, None, :]
         + gv[..., None] * v[None, None, :])
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    x0 = np.clip(np.floor(x).astype(int), 0, NX - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, NY - 2)
    z0 = np.clip(np.floor(z).astype(int), 0, NZ - 2)
    fx, fy, fz = x - x0, y - y0, z - z0
    out = np.zeros_like(x)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
                out += w * volume[z0 + dz, y0 + dy, x0 + dx]
    return out


cl = CenterLine.from_points([[30, 30, 5], [30, 30, 55]], step_mm=1.0)
u, v = cl.frames(ref_up=[1, 0, 0])
areas = []
cxs, cys = [], []
for i in range(5, cl.n - 5, 10):
    disc = sample(vol, cl.points[i], u[i], v[i])
    mask = disc > 0.5
    areas.append(mask.sum())
    if mask.any():
        gs = np.linspace(-12, 12, 49)
        gu, gv = np.meshgrid(gs, gs)
        cxs.append(gu[mask].mean()); cys.append(gv[mask].mean())
areas = np.array(areas)
# expected disc area ≈ π·4² = 50.3 mm² → in a 0.5mm grid (49 pts / 24mm span)
px_area = (24.0 / 48) ** 2
mm2 = areas * px_area
chk("cyl: cross-section area ≈ πr²",
    np.all(np.abs(mm2 - np.pi * 16) < 12), f"areas mm² = {np.round(mm2,1)}")
chk("cyl: disc centred on centreline",
    max(abs(np.mean(cxs)), abs(np.mean(cys))) < 0.8,
    f"centroid ({np.mean(cxs):.2f},{np.mean(cys):.2f})")
chk("cyl: area stable along vessel", areas.std() / areas.mean() < 0.1,
    f"cv={areas.std()/areas.mean():.3f}")

print("\nRESULT:", "ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(0 if not fails else 1)
