# -*- coding: utf-8 -*-
"""Headless checks for core/valve_auto.detect_apex on a synthetic LV.

Phantom (0.8 mm): a half-ellipsoid LV cavity (24 mm radius, 80 mm long) whose
long axis is tilted 15° from the MV-plane normal; myocardium 10 mm at the side
walls, 5 mm at the apex; the LA on the other side of the MV; the RV beyond the
septum; liver / diaphragm (soft tissue, no fat) touching the inferior wall;
lung lateral; epicardial fat elsewhere; apical trabeculae. The true epicardial
apex = the farthest epicardial point from the MV centre = C + 85·a.

Run: python tools/test_apex_auto.py
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import ndimage

from multi_dicomviewer.core.valve_auto import _frame, _unit, detect_apex

fails = []


def chk(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), name, extra)
    if not cond:
        fails.append(name)


S = 0.8
N = 200
dims = (S, S, S)
rng = np.random.default_rng(2)
g = (np.arange(N) * S).astype(np.float32)
Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
Pw = np.stack([X, Y, Z], axis=-1)

C = np.array([70.0, 75.0, 100.0])
n_mv = _unit([0.25, 0.15, 0.95])           # MV normal, apex → LA (+)
u0, w0 = _frame(n_mv)
tilt = np.deg2rad(15.0)
a = _unit(-np.cos(tilt) * n_mv + np.sin(tilt) * u0)   # LV axis, MV → apex
ua, wa = _frame(a)
L_ENDO, R_ENDO = 80.0, 24.0
L_EPI, R_EPI = 85.0, 34.0

Q = Pw - C.astype(np.float32)
h = Q @ a.astype(np.float32)
qa = Q @ ua.astype(np.float32)
qb = Q @ wa.astype(np.float32)
rr = np.sqrt(qa * qa + qb * qb)
cav = (h > -2.0) & ((h / L_ENDO) ** 2 + (rr / R_ENDO) ** 2 < 1.0)
myo = (h > -2.0) & ((h / L_EPI) ** 2 + (rr / R_EPI) ** 2 < 1.0) & ~cav
hn = Q @ n_mv.astype(np.float32)
la = (hn > 0) & (np.linalg.norm(Q - (24.0 * n_mv).astype(np.float32), axis=-1)
                 < 24.0)
la_wall = (ndimage.distance_transform_edt(~la, sampling=S) <= 2.0) & ~la & ~cav
# RV beyond the septum (−wa side), mid-ventricle
rv = ((np.abs(qa) < 18.0) & (qb < -R_EPI - 1.0) & (qb > -R_EPI - 16.0)
      & (h > 5.0) & (h < 60.0))
rv_wall = (ndimage.distance_transform_edt(~rv, sampling=S) <= 3.0) & ~rv
# liver / diaphragm touching the inferior wall (−ua side), no fat between
liver = (qa < -R_EPI * np.sqrt(np.clip(1 - (h / L_EPI) ** 2, 0, 1)) + 0.5) \
    & (qa > -R_EPI - 40.0) & (h > 10.0) & (h < 60.0)
# lung lateral (+ua side, beyond 6 mm of fat)
lung = qa > R_EPI + 6.0
trab = np.zeros_like(cav)
for _i in range(25):
    hh = rng.uniform(45, 74)
    ph = rng.uniform(-np.pi, np.pi)
    r0 = R_ENDO * np.sqrt(max(0.0, 1 - (hh / L_ENDO) ** 2)) - 1.0
    Pc = C + hh * a + r0 * (np.cos(ph) * ua + np.sin(ph) * wa)
    trab |= np.linalg.norm(Pw - Pc.astype(np.float32), axis=-1) < 1.8

vol = np.full(cav.shape, -90.0, np.float32)
vol[lung] = -800.0
vol[liver & ~myo & ~cav] = 70.0
vol[myo | la_wall | rv_wall] = 110.0
vol[cav | la] = 420.0
vol[rv] = 380.0
vol[trab & cav] = 110.0
vol = ndimage.gaussian_filter(vol, 0.7)
vol += rng.normal(0, 20, vol.shape).astype(np.float32)
vol = vol.astype(np.int16)
del X, Y, Z, Pw, Q, h, qa, qb, rr, hn

apex_true = C + L_EPI * a
endo_true = C + L_ENDO * a
for label, c, n in (("exact", C, n_mv),
                    ("normal flipped", C, -n_mv),
                    ("centre +3 mm, normal 8° off", C + np.array([2.0, -2.0, 1.0]),
                     _unit(n_mv + 0.14 * w0))):
    t0 = time.time()
    res = detect_apex(vol, dims, c, n)
    dt = time.time() - t0
    d = float(np.linalg.norm(res.apex - apex_true))
    de = float(np.linalg.norm(res.endo - endo_true))
    chk(f"Apex {label}: found", res.ok, res.message)
    chk(f"Apex {label}: epicardial apex ≤ 4 mm", d <= 4.0,
        f"{d:.1f} mm (wall {res.wall_mm:.1f} mm, {dt:.1f}s)")
    chk(f"Apex {label}: endocardial apex ≤ 6 mm", de <= 6.0, f"{de:.1f} mm")

print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
