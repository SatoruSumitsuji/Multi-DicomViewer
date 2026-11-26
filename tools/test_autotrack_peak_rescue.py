# -*- coding: utf-8 -*-
"""Auto-Track PEAK rescue (user 2026-11-26): a lumen of HU ≈ 190-200,
nearly round, was not taken — the thresholds still came from the brighter
proximal lumen (≈ 284 HU) and, lower down, the vessel merged with the
myocardium at 3-4 o'clock. Now, when nothing is found, the HU PEAKS are
searched (the section 3 mm ahead, then — a hairpin — the sphere of 3 mm
round the point up to 100°) and the best by HU + closeness to the course +
roundness is taken; a ~100 HU round structure beside it is not.
Synthetic volumes (made-up data).

    python tools/test_autotrack_peak_rescue.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
import multi_dicomviewer.core.cpr_autoextend as C  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


SP = 0.4


def fainting(dist_hu=220.0, myo_hu=125.0, myo_gap=0.8, seed=1):
    """bright (380) along z to 45 mm, then fainter / thinner to 85 mm;
    from 50 mm the myocardium touches it at 3-4 o'clock and a ~120 HU
    round structure runs 3.2 mm below it."""
    nz, ny, nx = 230, 110, 110
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32) * SP
    vol = np.full((nz, ny, nx), 10.0, np.float32)
    rad = np.where(z < 45, 1.5, 0.9)
    hu = np.where(z < 45, 380.0, dist_hu)
    ins = ((x - 22) ** 2 + (y - 22) ** 2 <= rad ** 2) & (z >= 4) & (z <= 85)
    vol[(x >= 22 + myo_gap) & (x <= 34) & (y >= 21) & (y <= 32)
        & (z >= 50)] = myo_hu
    vol[((x - 22) ** 2 + (y - 18.8) ** 2 <= 1.0) & (z >= 50)] = 120
    vol[ins] = hu[ins]
    vol = ndimage.gaussian_filter(vol, 0.6 / SP)
    vol += np.random.default_rng(seed).normal(0, 12, vol.shape).astype(
        np.float32)
    return vol


def hairpin(dh=256.0, seed=2):
    """bright along z, fainter from 40 mm, a hairpin at 60 mm (bend radius
    3.5 mm) running back down at x = 27."""
    nz, ny, nx = 200, 90, 90
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32) * SP
    vol = np.full((nz, ny, nx), 10.0, np.float32)
    pts = [(20.0, 20.0, zz, 1.5 if zz < 40 else 0.9,
            380.0 if zz < 40 else dh) for zz in np.arange(4, 60, 0.2)]
    for a in np.linspace(0, np.pi, 60):
        pts.append((23.5 - 3.5 * np.cos(a), 20.0, 60 + 3.5 * np.sin(a),
                    0.9, dh))
    pts += [(27.0, 20.0, zz, 0.9, dh) for zz in np.arange(60, 25, -0.2)]
    for (cx, cy, cz, r, h) in pts:
        sl = (slice(int((cz - r - 1) / SP), int((cz + r + 1) / SP) + 1),
              slice(int((cy - r - 1) / SP), int((cy + r + 1) / SP) + 1),
              slice(int((cx - r - 1) / SP), int((cx + r + 1) / SP) + 1))
        mm = ((x[sl] - cx) ** 2 + (y[sl] - cy) ** 2 + (z[sl] - cz) ** 2
              <= r * r)
        vol[sl][mm] = h
    vol = ndimage.gaussian_filter(vol, 0.6 / SP)
    vol += np.random.default_rng(seed).normal(0, 12, vol.shape).astype(
        np.float32)
    return vol


def run(vol, ctrl, **kw):
    res = C.extend_vessel(vol, (SP, SP, SP), ctrl,
                          prm=C.ExtendParams(**kw))
    P = (np.asarray(res.trunk.path) if res.trunk is not None
         else np.asarray(ctrl, float))
    ev = [e.get("event") for e in res.log
          if e.get("line") == 0 and e.get("event")]
    return res, P, ev


# ---- 1. fainter + thinner, myocardium beside, a ~120 HU structure below
CTRL = [[22.0, 22.0, 8.0], [22.0, 22.0, 14.0]]
for dh, mh, gap in ((220.0, 125.0, 0.8), (205.0, 145.0, 0.6)):
    v = fainting(dh, mh, gap)
    _, P0, _ = run(v, CTRL, peak_rescue=False)
    res, P1, ev = run(v, CTRL)
    dist = P1[P1[:, 2] > 50]
    tag = f"(lumen {dh:.0f} / myocardium {mh:.0f})"
    check(f"old rule stopped at the faint part {tag}", P0[-1][2] < 55,
          P0[-1])
    check(f"now traced on to the vessel's end {tag}", P1[-1][2] > 78,
          (P1[-1], ev[-3:]))
    check(f"… on the vessel (≤ 1 mm), not the myocardium {tag}",
          len(dist) and float(np.abs(dist[:, :2] - 22).max()) <= 1.0,
          float(np.abs(dist[:, :2] - 22).max()) if len(dist) else None)
    check(f"… never on the ~120 HU structure below {tag}",
          len(dist) and float(dist[:, 1].min()) > 21.0)
    check(f"… by the HU-peak rescue {tag}", "HU peak" in ev, ev[-4:])

# ---- 2. a faint vessel in a hairpin: the sphere stage
h = hairpin()
_, H0, _ = run(h, [[20.0, 20.0, 8.0], [20.0, 20.0, 14.0]],
               peak_rescue=False)
_, H1, evh = run(h, [[20.0, 20.0, 8.0], [20.0, 20.0, 14.0]])
back = H1[H1[:, 0] > 25.5]
check("hairpin (faint): the old rule stopped at the bend",
      float(H0[:, 0].max()) < 23.0, H0[-1])
check("hairpin (faint): now round the bend and down the far limb",
      len(back) and float(back[:, 2].min()) < 35.0, (H1[-1], evh[-3:]))
check("hairpin: the far limb followed on its centre (≤ 1 mm)",
      len(back) and float(np.abs(back[:, 0] - 27.0).max()) <= 1.0)
check("hairpin: never back onto its own way",
      not any(np.linalg.norm(H1[k + 3:] - H1[k], axis=1).min() < 1.0
              for k in range(len(H1) - 4)))

# ---- 3. the peak helper itself
V = C._Vol(fainting(220.0, 125.0, 0.8), (SP, SP, SP))
d = np.array([0.0, 0.0, 1.0])
pk = C._peak_lumen(V, [22.0, 22.0, 70.0], d, 50.0, find_mm=0.75)
check("a round faint lumen: peak ≥ 150, closed ≥ 0.55, radius ≈ 0.9",
      pk["peak"] >= 150 and pk["closed"] >= 0.55
      and 0.5 <= pk["r"] <= 1.4, pk)
pm = C._peak_lumen(V, [29.0, 27.0, 70.0], d, 50.0, find_mm=0.75)
check("the myocardium: not a lumen (below the gate or not closed / "
      "not standing out)",
      pm is None or pm["peak"] < 150 or pm["closed"] < 0.55
      or pm.get("prom", 0) < 40, pm)
pb = C._peak_lumen(V, [22.0, 18.8, 70.0], d, 50.0, find_mm=0.5)
check("the ~120 HU structure: below the gate (bg + 100)",
      pb is None or pb["peak"] < 150, pb)
prm = C.ExtendParams()
check("defaults: peak rescue on, gate bg+100, ≥ 40 above surroundings, "
      "up to 100° (hairpin)",
      prm.peak_rescue and prm.peak_min_contrast == 100.0
      and prm.peak_min_prom == 40.0 and prm.hairpin_deg == 100.0)

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
