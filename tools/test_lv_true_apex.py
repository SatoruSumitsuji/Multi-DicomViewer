"""LV mode, MV-AoV point flow (2026-11-16) — core parts, headless:
- valve_auto.apex_from_point: the true (epicardial) apex from the MV-AoV
  point + a provisional apex (the crossing at Apex Set), with the MV+Apex
  apex search; a provisional apex too far off / too close is rejected;
- the MV-AoV tent closes the Epi: LVSurface.contains never extends the base;
- bloodpool_volume_epi with NO valve planes (the tent is the base).

    python tools/test_lv_true_apex.py
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from multi_dicomviewer.core import valve_auto as va  # noqa: E402
from multi_dicomviewer.core.lv_bloodpool import bloodpool_volume_epi  # noqa: E402
from multi_dicomviewer.core.lv_measure import LVModel  # noqa: E402

FAILS = []


def chk(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


# ---- synthetic LV: blood cavity (ellipsoid) in a 10-mm myocardium in fat -------
sp = 1.0
nz, ny, nx = 140, 100, 100
z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(float)
C = np.array([50.0, 50.0, 80.0])              # (x, y, z) centre of the cavity
a, c = 22.0, 45.0                              # cavity radius / half-length
W = 10.0                                       # wall


def ell(ra, rc):
    return (((x - C[0]) / ra) ** 2 + ((y - C[1]) / ra) ** 2
            + ((z - C[2]) / rc) ** 2) <= 1.0


vol = np.full((nz, ny, nx), -100.0, np.float32)    # fat
vol[ell(a + W, c + W)] = 100.0                      # myocardium
vol[ell(a, c)] = 400.0                              # contrast blood
vol[(z > C[2] + 20)] = np.where(vol[(z > C[2] + 20)] > 0, 400.0, -100.0)
dims = (sp, sp, sp)
apex_epi = np.array([C[0], C[1], C[2] - c - W])     # z = 25
apex_endo = np.array([C[0], C[1], C[2] - c])        # z = 35
point = np.array([C[0] + 10.0, C[1], C[2] + 30.0])  # a basal point (z = 110)

found = []
for off in ([0, 0, 0], [6, -4, 3], [-5, 5, -2]):
    prov = apex_epi + np.array(off, float)
    r = va.apex_from_point(vol, dims, point, prov)
    err = float(np.linalg.norm(np.asarray(r.apex) - apex_epi)) if r.ok else -1
    chk(f"true apex from provisional {off}", r.ok and 0 <= err <= 5.0
        and abs(float(r.apex[2]) - apex_epi[2]) <= 1.5,
        f"err {err:.1f} mm, wall {r.wall_mm:.1f}")
    found.append(np.asarray(r.apex, float))
F = np.asarray(found)
spread = float(np.linalg.norm(F - F.mean(axis=0), axis=1).max())
chk("same true apex whatever the provisional offset", spread <= 1.5,
    f"{spread:.1f} mm")
r = va.apex_from_point(vol, dims, point, point + [0, 0, -10.0])
chk("provisional too close → not ok", not r.ok)
r = va.apex_from_point(vol, dims, point, apex_epi + [0, 0, -1.0],
                       max_shift_mm=0.01)
chk("far from provisional → not ok", not r.ok and "far" in r.message, r.message)

# ---- tent: contains never extends the base; blood pool with no planes ----------
R, H, L = 20.0, 50.0, 70.0


def r_of(t):
    if t <= H:
        return R * math.sqrt(max(0.0, 1.0 - ((H - t) / H) ** 2))
    return R * (L - t) / (L - H)


m = LVModel(n_planes=8)
ap = np.array([50.0, 50.0, 20.0])
m.set_axis(ap + [1, 0, L], ap + [-1, 0, L], ap)
m.epi_tent = True
for phi in m.plane_angles():
    ts = np.linspace(L, 0.0, 70)
    A = [m.axis.to_world(phi, r_of(t), t) for t in ts]
    B = [m.axis.to_world(phi + 180, r_of(t), t) for t in ts[::-1]]
    m.set_long_axis_contour(phi, np.asarray(A + B), which="epi")
m.build()
S = m.epi
above = ap + [3.0, 0, L + 5.0]                  # past the tent top (the point)
chk("tent: base not extended", not S.contains([above], extend_base=True)[0])
chk("tent: inside point contained", bool(S.contains([ap + [3.0, 0, 30.0]])[0]))

bvol = np.full((nz, ny, nx), 400.0, np.float32)      # blood everywhere
res = bloodpool_volume_epi(bvol, dims, S._all_ring_points(),
                           lambda p: S.contains(p, extend_base=True),
                           [], tuple(ap), 200.0, 700.0, None)
want = (2 / 3 * math.pi * R * R * H + math.pi * R * R * (L - H) / 3) / 1000.0
got = None if res is None else res.get("ml", res.get("volume_ml"))
chk("blood pool with no planes = the tent volume",
    got is not None and abs(got - want) / want < 0.08, f"{got} vs {want:.1f}")

print("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL: {FAILS}")
sys.exit(1 if FAILS else 0)
