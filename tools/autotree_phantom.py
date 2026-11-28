# -*- coding: utf-8 -*-
"""A synthetic heart CT for the 全冠動脈描出 tests (no patient data):
an LV (half ellipsoid, apex at the bottom), an aorta, LM → LAD (a short
stenosis gap) + LCX + a diagonal, an RCA outside the LV Epi down to a PD,
a darker vein along the LAD and an occluded bright segment on the Epi.
Imported by tools/test_coronary_autotree.py / test_autotree_flow.py."""
import numpy as np


S = 0.5                                     # mm (isotropic test CT)
NX, NY, NZ = 200, 200, 250
vol = np.full((NZ, NY, NX), -60.0, np.float32)          # fat
zz, yy, xx = np.meshgrid(np.arange(NZ) * S, np.arange(NY) * S,
                         np.arange(NX) * S, indexing="ij")
CX, CY, ZB, ZA = 50.0, 50.0, 80.0, 10.0     # LV centre, base z, apex z
RE, RI = 25.0, 17.0                         # Epi / Endo radius at the base


def ell(r, c):
    return ((xx - CX) / r) ** 2 + ((yy - CY) / r) ** 2 + ((zz - ZB) / c) ** 2


epi = (ell(RE, ZB - ZA) <= 1.0) & (zz <= ZB)
endo = (ell(RI, ZB - ZA - 8) <= 1.0) & (zz <= ZB)
vol[epi] = 100.0                            # myocardium
vol[endo] = 420.0                           # LV blood
aorta = ((xx - 50) ** 2 + (yy - 62) ** 2 <= 11.0 ** 2) & (zz >= 86) & (zz <= 124)
vol[aorta] = 420.0


def surf(theta_deg, z, off=2.0):
    """A point just outside the Epi at angle theta, height z."""
    q = 1.0 - ((z - ZB) / (ZB - ZA)) ** 2
    r = RE * np.sqrt(max(q, 0.0)) + off
    t = np.radians(theta_deg)
    return np.array([CX + r * np.cos(t), CY + r * np.sin(t), z])


def tube(pts, r, hu, gap=None):
    """Paint a tube of radius r round the polyline pts (world mm)."""
    pts = np.asarray(pts, float)
    dense = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(np.linalg.norm(b - a) / 0.25), 1)
        dense += [a + (b - a) * k / n for k in range(1, n + 1)]
    for p in dense:
        lo = np.maximum(((p - r - 1) / S).astype(int), 0)
        hi = np.minimum(((p + r + 1) / S).astype(int) + 1, [NX, NY, NZ])
        sl = (slice(lo[2], hi[2]), slice(lo[1], hi[1]), slice(lo[0], hi[0]))
        d2 = ((xx[sl] - p[0]) ** 2 + (yy[sl] - p[1]) ** 2 + (zz[sl] - p[2]) ** 2)
        val = hu
        if gap is not None and gap[0] <= p[2] <= gap[1]:
            val = gap[2]
        vol[sl][d2 <= r * r] = val


LM0 = np.array([39.3, 62.0, 92.0])          # left ostium (aorta wall)
BIF = np.array([33.0, 70.0, 86.0])
lad = [BIF] + [surf(105 - (82 - z) * 0.2, z) for z in np.arange(82, 12, -2.0)]
lcx = [BIF] + [surf(th, 81.0) for th in np.arange(130, 250, 6.0)]
lm = [LM0, BIF]
diag = [surf(105 - (82 - 60) * 0.2, 60.0)] + \
    [surf(th, z) for th, z in zip(np.linspace(95, 55, 12), np.linspace(58, 36, 12))]
RC0 = np.array([60.7, 62.0, 92.0])          # right ostium
rca = [RC0, np.array([70.0, 72.0, 88.0])] + \
    [surf(th, 84.0, off=14.0) for th in np.arange(50, -91, -8.0)] + \
    [surf(-90, z, off=2.0) for z in np.arange(80, 44, -3.0)]
tube(lm, 2.2, 420.0)
tube(lad, 1.5, 420.0, gap=(46.0, 47.6, 190.0))     # a short stenosis
tube(lcx, 1.4, 420.0)
tube(diag, 1.1, 420.0)
tube(rca, 1.6, 420.0)
vein = [surf(128, z, off=2.2) for z in np.arange(79, 22, -3.0)]
tube(vein, 1.6, 200.0)                       # darker: not an artery
occl = [surf(-20, z) for z in np.arange(68, 38, -2.0)]
tube(occl, 1.3, 400.0)                       # occluded vessel, collateral filled
rng = np.random.default_rng(3)
vol += rng.normal(0, 12, vol.shape).astype(np.float32)

PHANTOM = dict(vol=vol, S=S, epi=epi, endo=endo, LM0=LM0, RC0=RC0,
               apex=np.array([CX, CY, ZA]), CX=CX, CY=CY, ZA=ZA, RE=RE,
               NX=NX, NY=NY, NZ=NZ, occl=occl, vein=vein)
