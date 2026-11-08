# -*- coding: utf-8 -*-
"""Headless checks for core/valve_auto.py on a synthetic left heart.

Phantom (contrast CTA-like, 0.6 mm): LV cavity and LA joined at a SADDLE-shaped
mitral annulus (r 15 mm, ±2 mm saddle) with only a mild waist; a THICK LV wall
(10 mm) that ends at the annulus and a THIN LA wall (2 mm); papillary muscles
and trabeculae in the LV (they narrow the pool more than the annulus does);
RV beyond the septum, RA beyond the atrial septum; LCx and coronary sinus in
the AV-groove fat; an LVOT → aortic root with a three-lobed sinus bulge →
ascending aorta. Planes tilted off the volume axes. Systole variant: closed
MV leaflets (thin dark dome).

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


def build(LV_PEAK, LEAFLETS):
    global n_ao, u_ao, w_ao, C_ao, R_ann
    h, r, phi = mv_coords()
    hs = 2.0 * np.cos(2.0 * phi)                                # saddle height
    # LV: diastole wider than the annulus below it; systole ≈ annulus size at the
    # base (no waist on the LV side), tapering to the apex.
    rho_lv = np.where(h > -12.0, Ra + (LV_PEAK - Ra) * np.clip(-h / 12.0, 0, 1),
                      LV_PEAK * np.sqrt(np.clip(1.0 - ((h + 12.0) / 63.0) ** 2, 0, None)))
    rho_la = 23.0 * np.sqrt(np.clip(1.0 - ((h - 22.0) / 32.0) ** 2, 0, None))
    rho = np.where(h < hs, rho_lv, rho_la)
    rho = np.where(h >= hs, np.minimum(rho, Ra + 0.6 * (h - hs)), rho)   # LA side
    lv_blood = (r < rho) & (h < hs)
    la_blood = (r < rho) & (h >= hs)


    def sector(phi0, half):
        return np.abs(np.angle(np.exp(1j * (phi - phi0)))) <= half


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
    del Qa, aa, ba, psi

    # Walls: THICK LV myocardium (10 mm) that ends at the annulus, THIN LA wall
    # (2 mm), thin aortic wall; epicardial fat everywhere else.
    lv_cav = lv_blood | (aorta & (ta < 0))
    myo_lv = ((ndimage.distance_transform_edt(~lv_cav, sampling=S) <= 10.0)
              & (h < hs + 1.5))
    la_wall = ndimage.distance_transform_edt(~la_blood, sampling=S) <= 2.0
    ao_wall = ndimage.distance_transform_edt(~aorta, sampling=S) <= 2.5
    # RV beyond the septum (−w side) and RA beyond the thin atrial septum.
    rv = (sector(-np.pi / 2, np.deg2rad(60)) & (h > -60) & (h < -4)
          & (r > rho_lv + 10.0) & (r < rho_lv + 24.0))
    rho_la_raw = 23.0 * np.sqrt(np.clip(1.0 - ((h - 22.0) / 32.0) ** 2, 0, None))
    ra_b = (sector(-np.pi / 2, np.deg2rad(50)) & (h > 3) & (h < 40)
            & (r > np.minimum(rho_la_raw, Ra + 0.6 * np.abs(h - hs)) + 3.0)
            & (r < rho_la_raw + 22.0))
    # Papillary muscles + trabeculae inside the LV (thick, dark).
    pap = np.zeros_like(lv_blood)
    for ph0 in (0.75 * np.pi, -0.6 * np.pi):
        cxy = (LV_PEAK - 6.0) * np.array([np.cos(ph0), np.sin(ph0)])
        a_ = r * np.cos(phi) - cxy[0]
        b_ = r * np.sin(phi) - cxy[1]
        pap |= (a_ * a_ + b_ * b_ < 6.0 ** 2) & (h > -42) & (h < -16)
    trab = np.zeros_like(lv_blood)
    for _i in range(40):
        hh = rng.uniform(-55, -14)
        ph = rng.uniform(-np.pi, np.pi)
        rr0 = LV_PEAK * np.sqrt(max(0.0, 1.0 - ((hh + 12.0) / 63.0) ** 2)) - 1.5
        Pc = C_mv + hh * n_mv + rr0 * (np.cos(ph) * u_mv + np.sin(ph) * w_mv)
        trab |= ((h - hh) ** 2 + (r * np.cos(phi) - rr0 * np.cos(ph)) ** 2
                 + (r * np.sin(phi) - rr0 * np.sin(ph)) ** 2) < 2.2 ** 2
    # AV groove vessels in the fat: LCx (lateral) and coronary sinus (posterior).
    lcx = (sector(0.55 * np.pi, 0.35 * np.pi)
           & ((h - 4.0) ** 2 + (r - 22.0) ** 2 < 1.8 ** 2))
    cs = (sector(np.pi, 0.4 * np.pi)
          & ((h - 5.0) ** 2 + (r - 21.5) ** 2 < 4.0 ** 2))

    vol = np.full(lv_blood.shape, -90.0, np.float32)            # fat
    vol[myo_lv | la_wall | ao_wall] = 110.0
    vol[lv_blood | la_blood] = 420.0
    vol[aorta] = 440.0
    vol[rv | ra_b] = 380.0
    vol[(pap | trab) & lv_blood] = 110.0
    vol[lcx | cs] = 400.0
    if LEAFLETS:   # systole: closed leaflets = thin dome, hinge → 5 mm into the LV
        leaf = ((r < Ra) & (np.abs(h - (hs - 5.0 * (1.0 - r / Ra))) < 0.6)
                & ~aorta)
        vol[leaf] = 110.0


    return vol


def finish(v):
    v = ndimage.gaussian_filter(v, 0.7)
    v += rng.normal(0, 20, v.shape).astype(np.float32)
    return v.astype(np.int16)


vol_dia = finish(build(24.0, False))
vol_sys = finish(build(15.5, True))
del X, Y, Z, Pw


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
        dv = res.center - C_mv
        print(f"   MV {label} {off_deg}°: centre offset along the axis "
              f"{float(np.dot(dv, n_mv)):+.1f} mm (− = apex-ward), in-plane "
              f"{float(np.linalg.norm(dv - np.dot(dv, n_mv) * n_mv)):.1f} mm")
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
