# -*- coding: utf-8 -*-
"""全冠動脈 一括描出 core (user 2026-11-28) on a synthetic heart:

  LV (half ellipsoid, apex at the bottom) with myocardium + a bright cavity,
  an aorta above the base, LM → LAD (to the apex, with a short stenosis
  gap) + LCX (round the AV groove) + a diagonal, an RCA leaving the aorta
  OUTSIDE the LV Epi (its corridor) down to a PD, a darker vein along the
  LAD, and an occluded (unconnected) bright segment on the Epi.

  - LM / LAD / LCX / RCA roots, LAD = the trunk reaching the apex
  - the diagonal is a branch of the LAD, the PD part follows the RCA
  - the stenosis gap is bridged (the LAD reaches the apex)
  - the vein is not taken; the occluded segment is 未接続 (connected False)
  - no line runs inside the LV cavity / the aorta

    python tools/test_coronary_autotree.py
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from multi_dicomviewer.core.coronary_autotree import (  # noqa: E402
    AutoTreeParams, extract_coronary_tree)

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


from tools.autotree_phantom import *  # noqa: E402,F401,F403


t0 = time.time()
res = extract_coronary_tree(
    vol, (S, S, S), [("LM", LM0), ("RCA", RC0)], epi=epi,
    epi_origin_mm=(0, 0, 0), apex=np.array([CX, CY, ZA]),
    prm=AutoTreeParams(), progress=lambda s: None)
dt = time.time() - t0
print("log", res.log, f"{dt:.1f}s")
V = res.vessels
roles = {v.role: v for v in V if v.role != "branch"}
check("extraction ran", res.stop == "ok", res.stop)
check("LM / LAD / LCX / RCA roots", set(roles) == {"LM", "LAD", "LCX", "RCA"},
      sorted(roles))
check("roots have no parent", all(roles[r].parent is None for r in roles))
if "LAD" in roles:
    d = np.linalg.norm(roles["LAD"].points[-1] - [CX, CY, ZA])
    check("LAD reaches the apex (stenosis gap bridged)", d < 12.0, round(d, 1))
    check("the gap was bridged", res.log.get("bridged_gaps", 0) >= 1, res.log)
if "LCX" in roles:
    e = roles["LCX"].points[-1]
    check("LCX runs round the AV groove", abs(e[2] - 81) < 6
          and np.degrees(np.arctan2(e[1] - CY, e[0] - CX)) % 360 > 215, e)
if "RCA" in roles:
    e = roles["RCA"].points[-1]
    check("RCA → PD (posterior, toward the apex)", e[1] < CY - 15 and e[2] < 60, e)
    pts = roles["RCA"].points
    check("RCA prox-mid runs OUTSIDE the Epi shell (the corridor)",
          np.any(np.linalg.norm(pts[:, :2] - [CX, CY], axis=1) > RE + 9), "")
keys = {v.key: v for v in V}
lad_key = roles["LAD"].key if "LAD" in roles else None
dg = [v for v in V if v.role == "branch" and v.connected
      and v.parent == lad_key and v.length_mm > 15]
check("the diagonal is a branch of the LAD", len(dg) >= 1,
      [(v.parent, round(v.length_mm)) for v in V if v.role == "branch"])
unc = [v for v in V if not v.connected]
check("the occluded segment → one 未接続 line", len(unc) == 1
      and np.linalg.norm(unc[0].points.mean(0) - np.mean(occl, 0)) < 6,
      [np.round(u.points.mean(0), 1) for u in unc])
allp = np.vstack([v.points for v in V])
iz = np.clip((allp[:, 2] / S).round().astype(int), 0, NZ - 1)
iy = np.clip((allp[:, 1] / S).round().astype(int), 0, NY - 1)
ix = np.clip((allp[:, 0] / S).round().astype(int), 0, NX - 1)
check("no line inside the LV cavity", not endo[iz, iy, ix].any())
inner_ao = ((allp[:, 0] - 50) ** 2 + (allp[:, 1] - 62) ** 2 <= 8.0 ** 2) & \
    (allp[:, 2] >= 88)
check("no line inside the aorta", not inner_ao.any())
vd = np.vstack(vein)
near_vein = [v for v in V if np.min(np.linalg.norm(
    v.points[:, None, :] - vd[None], axis=2), axis=1).mean() < 1.5]
check("the vein is not taken", not near_vein)
check("branches start ON their parent line", all(
    np.min(np.linalg.norm(keys[v.parent].points - v.points[0], axis=1)) < 1.0
    for v in V if v.parent is not None and v.parent in keys
    and v.role == "branch"))
check("parents come before their children",
      all(v.parent is None or v.parent < v.key for v in V))
check("reasonable time on a 100×100×125 mm CT", dt < 90, f"{dt:.1f}s")

# no Epi → lumen continuity only (still finds the arteries from the ostia)
r2 = extract_coronary_tree(vol, (S, S, S), [("LM", LM0), ("RCA", RC0)])
r2r = {v.role for v in r2.vessels}
check("no Epi: still LM / LAD / LCX / RCA", {"LM", "LAD", "LCX", "RCA"} <= r2r,
      sorted(r2r))
check("no Epi: no 未接続 search", all(v.connected for v in r2.vessels))
# LMなし: LAD and LCX set separately (their first points)
r4 = extract_coronary_tree(vol, (S, S, S), [("LAD", np.asarray(lad[2])),
                                            ("LCX", np.asarray(lcx[2])),
                                            ("RCA", RC0)],
                           epi=epi, epi_origin_mm=(0, 0, 0),
                           apex=np.array([CX, CY, ZA]))
r4r = [v.role for v in r4.vessels if v.role != "branch"]
check("LMなし: LAD / LCX / RCA roots, no LM", sorted(r4r) == ["LAD", "LCX", "RCA"],
      r4r)
# cancel
r3 = extract_coronary_tree(vol, (S, S, S), [("LM", LM0)], cancel=lambda: True)
check("cancel → stop 'cancel', nothing returned",
      r3.stop == "cancel" and not r3.vessels)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(0 if not fails else 1)
