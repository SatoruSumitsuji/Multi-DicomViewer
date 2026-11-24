"""Auto-Track centre with a WIDER CT window (user 2026-11-25): the centre of
a found lumen is the centre of the whole bright round shape (voxels above
bg + 30 % of the contrast, weights saturating at the lumen HU), with calcium
and its blooming rim left out — not the HU-weighted centre above the
half-max threshold, which a calcified wall pulled ~1 mm off. Synthetic data.

    python tools/test_cpr_autotrack_centre.py
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
nx, ny, nz = 100, 40, 40
z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32) * h
CY, CZ, R = 8.0, 8.0, 1.5                       # vessel along x
LUMEN, BG = 420.0, 20.0


def make(seed, calc):
    vol = np.where((y - CY) ** 2 + (z - CZ) ** 2 <= R * R, LUMEN, BG)
    vol = vol.astype(np.float32)
    if calc:                                    # calcified wall at +y, x≈20
        m = (x - 20.0) ** 2 + (y - (CY + R + 0.3)) ** 2 + (z - CZ) ** 2 \
            <= 0.9 ** 2
        vol[m] = 2200.0
    vol = ndimage.gaussian_filter(vol, 0.7 / h)
    vol += np.random.default_rng(seed).normal(0, 35, vol.shape).astype(
        np.float32)
    return vol


dims = (h, h, h)
d = np.array([1.0, 0.0, 0.0])
true_c = np.array([20.0, CY, CZ])
thr = BG + 0.5 * (LUMEN - BG)
win = CAE._centre_window(BG, LUMEN, CAE.ExtendParams())
check("default window: floor bg+30 %, cap = lumen, calcium above lumen+80 %",
      abs(win["lo"] - (BG + 0.3 * 400)) < 1e-6 and win["cap"] == LUMEN
      and abs(win["hi"] - (LUMEN + 0.8 * 400)) < 1e-6, win)

old_e, new_e = {}, {}
for calc in (False, True):
    oe, ne = [], []
    for seed in range(3):
        V = CAE._Vol(make(seed, calc), dims)
        start = true_c + np.array([0.0, 0.3, 0.0])
        g0 = min(CAE._regions(V, start, d, thr, 5.0, 0.25),
                 key=lambda r: np.linalg.norm(r["c"] - true_c))
        g1 = min(CAE._regions(V, start, d, thr, 5.0, 0.25, **win),
                 key=lambda r: np.linalg.norm(r["c"] - true_c))
        oe.append(float(np.linalg.norm(g0["c"] - true_c)))
        ne.append(float(np.linalg.norm(g1["c"] - true_c)))
    old_e[calc], new_e[calc] = float(np.mean(oe)), float(np.mean(ne))
    print(f"   {'calcium' if calc else 'clean  '}: old {old_e[calc]:.3f} mm"
          f" → new {new_e[calc]:.3f} mm")
check("clean lumen: the centre stays as good (≤ 0.15 mm)",
      new_e[False] <= 0.15 and new_e[False] <= old_e[False] + 0.02,
      (old_e[False], new_e[False]))
check("calcified wall: the centre is no longer pulled to the calcium",
      new_e[True] < 0.45 and new_e[True] < old_e[True] - 0.4,
      (old_e[True], new_e[True]))

# Auto-Track itself on the calcified vessel stays on the centreline
vol = make(0, True)
ctrl = np.array([[4.0, CY, CZ], [8.0, CY, CZ], [12.0, CY, CZ]])


def trace(**kw):
    r = CAE.extend_vessel(vol, dims, ctrl,
                          CAE.ExtendParams(ask_at_fork=False, **kw))
    P = r.trunk.path if r.trunk is not None else np.zeros((0, 3))
    return P, (np.hypot(P[:, 1] - CY, P[:, 2] - CZ) if len(P)
               else np.array([9.0]))


P, off = trace()
_Po, off_old = trace(center_frac=0.5)          # = the old half-max centre
away = np.abs(P[:, 0] - 20.0) > 2.0 if len(P) else np.array([True])
check("Auto-Track runs past the calcified wall",
      len(P) > 5 and P[:, 0].max() > 30.0, len(P))
check("… on the centre away from the calcium (< 0.2 mm)",
      float(off[away].max()) < 0.2, off.round(2))
check("… and closer than before at the calcium",
      float(off.max()) < float(off_old.max()) - 0.1,
      (off.max(), off_old.max()))
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
sys.stdout.flush()
os._exit(1 if FAILS else 0)
