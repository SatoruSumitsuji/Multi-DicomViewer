# -*- coding: utf-8 -*-
"""Headless checks for core/valve_auto.py on a synthetic left heart.

Phantom (contrast CTA-like, 0.6 mm): LV cavity and LA joined at a SADDLE-shaped
mitral annulus (r 15 mm, ±2 mm saddle) that narrows the blood pool to a waist
(the AV junction), myocardium around the blood, fat outside; an LVOT → aortic
root with a three-lobed sinus bulge → ascending aorta. Planes are tilted off
the volume axes. Systole variant: closed MV leaflets (thin dark dome).

Run: python tools/test_valve_auto.py
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import ndimage

from multi_dicomviewer.core.valve_auto import _frame, _unit, detect_aov, detect_mv

fails = []


def chk(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), name, extra)
    if not cond:
        fails.append(name)


S = 0.6
N = 220
dims = (S, S, S)
rng = np.random.default_rng(1)
g = (np.arange(N) * S).astype(np.float32)
Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
Pw = np.stack([X, Y, Z], axis=-1)                           # (z, y, x, 3)

C_mv = np.array([58.0, 62.0, 60.0])
n_mv = _unit([0.30, 0.20, 0.93])
u_mv, w_mv = _frame(n_mv)
Ra = 15.0


def mv_coords():
    Q = Pw - C_mv.astype(np.float32)
    h = Q @ n_mv.astype(np.float32)
    a = Q @ u_mv.astype(np.float32)
    b = Q @ w_mv.astype(np.float32)
    return h, np.sqrt(a * a + b * b), np.arctan2(b, a)


h, r, phi = mv_coords()
hs = 2.0 * np.cos(2.0 * phi)                                # saddle height
rho_lv = 24.0 * np.sqrt(np.clip(1.0 - ((h + 30.0) / 45.0) ** 2, 0, None))
rho_la = 23.0 * np.sqrt(np.clip(1.0 - ((h - 22.0) / 32.0) ** 2, 0, None))
rho = np.where(h < hs, rho_lv, rho_la)
rho = np.minimum(rho, Ra + 1.2 * np.abs(h - hs))            # AV-junction waist
blood = r < rho

# Aortic root: anterior (+u), axis tilted towards +u.
n_ao = _unit(n_mv + 0.8 * u_mv)
u_ao, w_ao = _frame(n_ao)
C_ao = C_mv + 34.0 * u_mv - 4.0 * n_mv
Qa = Pw - C_ao.astype(np.float32)
ta = Qa @ n_ao.astype(np.float32)
aa = Qa @ u_ao.astype(np.float32)
ba = Qa @ w_ao.astype(np.float32)
ra = np.sqrt(aa * aa + ba * ba)
psi = np.arctan2(ba, aa)
R_ann = 12.0
r_lvot = np.where(ta < 0, 11.0 + 1.0 * np.clip(1 + ta / 10.0, 0, 1), R_ann)
bul = np.where((ta >= 0) & (ta <= 20),
               4.5 * np.sin(np.pi * np.clip(ta, 0, 20) / 20.0)
               * (1.0 + 0.35 * np.cos(3 * psi)), 0.0)
r_root = np.where(ta < 0, r_lvot, np.where(ta <= 20, R_ann + bul, 13.0))
aorta = (ra < r_root) & (ta > -26.0) & (ta < 60.0)
blood |= aorta
del Qa, aa, ba, psi

myo = ndimage.distance_transform_edt(~blood, sampling=S) <= 6.0
vol = np.full(blood.shape, -80.0, np.float32)
vol[myo] = 110.0
vol[blood] = 420.0
vol_dia = vol.copy()
# Systole: closed leaflets = thin dome from the hinge to 5 mm into the LV.
leaf = ((r < Ra) & (np.abs(h - (hs - 5.0 * (1.0 - r / Ra))) < 0.6)
        & ~aorta)
vol_sys = vol.copy()
vol_sys[leaf] = 110.0


def finish(v):
    v = ndimage.gaussian_filter(v, 0.7)
    v += rng.normal(0, 20, v.shape).astype(np.float32)
    return v.astype(np.int16)


vol_dia, vol_sys = finish(vol_dia), finish(vol_sys)
del X, Y, Z, Pw, h, r, phi, hs, rho, rho_lv, rho_la, myo, vol, leaf


def ang(a, b):
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(_unit(a), _unit(b))))))))


apex_true = C_mv - 75.0 * n_mv
for label, v in (("diastole", vol_dia), ("systole", vol_sys)):
    for off_deg, seed_off in ((10, np.array([3.0, -2.0, 2.0])),
                              (25, np.array([-4.0, 3.0, -3.0])),
                              (40, np.array([5.0, 3.0, -2.0]))):
        apex_rough = (C_mv - 75.0 * np.cos(np.deg2rad(off_deg)) * n_mv
                      + 75.0 * np.sin(np.deg2rad(off_deg)) * w_mv)
        t0 = time.time()
        res = detect_mv(v, dims, C_mv + seed_off, apex_rough)
        dt = time.time() - t0
        a = ang(res.normal, n_mv)
        d = float(np.linalg.norm(res.center - C_mv))
        tag = f"MV {label}, apex {off_deg}° off"
        chk(f"{tag}: found", res.ok, res.message)
        chk(f"{tag}: plane angle ≤ 6°", a <= 6.0, f"{a:.1f}°")
        chk(f"{tag}: centre ≤ 3 mm", d <= 3.0, f"{d:.1f} mm")
        chk(f"{tag}: radius within 2.5 mm of {Ra}", abs(res.radius - Ra) <= 2.5,
            f"{res.radius:.1f} mm, {len(res.points)} pts, {dt:.1f}s")

for label, v in (("diastole", vol_dia), ("systole", vol_sys)):
    for click in (C_ao + 7.0 * n_ao + 3.0 * u_ao, C_ao + 2.0 * n_ao - 3.0 * w_ao):
        t0 = time.time()
        res = detect_aov(v, dims, click)
        dt = time.time() - t0
        a = ang(res.normal, n_ao)
        # the centre may slide a little along the axis; judge in-plane and axial
        dv = res.center - C_ao
        ax_err = abs(float(np.dot(dv, n_ao)))
        rad_err = float(np.linalg.norm(dv - np.dot(dv, n_ao) * n_ao))
        tag = f"AoV {label}, click {np.round(click - C_ao, 1)}"
        chk(f"{tag}: found", res.ok, res.message)
        chk(f"{tag}: axis angle ≤ 6°", a <= 6.0, f"{a:.1f}°")
        chk(f"{tag}: centre in-plane ≤ 2 mm", rad_err <= 2.0, f"{rad_err:.1f} mm")
        chk(f"{tag}: annulus level ≤ 3 mm", ax_err <= 3.0, f"{ax_err:.1f} mm")
        chk(f"{tag}: radius within 2 mm of {R_ann}", abs(res.radius - R_ann) <= 2.0,
            f"{res.radius:.1f} mm, {dt:.1f}s")

print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
