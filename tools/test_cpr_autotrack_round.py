"""Auto-Track: a FAINT but ROUND lumen is still the vessel (user 2026-11-25).
When the contrast drops (the distal vessel only a little above the
background) the normal half-max threshold finds nothing; the section is then
searched from a lower threshold and a region that is nearly a circle and
brighter than just around it is taken. A faint FLAT band (not round) and
plain background are not. Synthetic data.

    python tools/test_cpr_autotrack_round.py
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
nx, ny, nz = 160, 40, 40
z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32) * h
CY, CZ, R = 8.0, 8.0, 1.4
BG = -60.0


def make(faint_hu, end_x=58.0, band=False, seed=0):
    """A vessel along x: 420 HU up to x = 24, then *faint_hu* to *end_x*.
    *band*: past the end, a faint flat band (1.6 mm × 9 mm) on the course."""
    vol = np.full(x.shape, BG, np.float32)
    tube = (y - CY) ** 2 + (z - CZ) ** 2 <= R * R
    vol[tube & (x <= 24.0)] = 420.0
    vol[tube & (x > 24.0) & (x <= end_x)] = faint_hu
    if band:
        vol[(np.abs(z - CZ) <= 0.8) & (np.abs(y - CY) <= 4.5)
            & (x > end_x + 1.0)] = faint_hu
    vol = ndimage.gaussian_filter(vol, 0.6 / h)
    vol += np.random.default_rng(seed).normal(0, 20, vol.shape).astype(
        np.float32)
    return vol


dims = (h, h, h)
ctrl = np.array([[4.0, CY, CZ], [8.0, CY, CZ], [12.0, CY, CZ]])


def run(vol, **kw):
    r = CAE.extend_vessel(vol, dims, ctrl,
                          CAE.ExtendParams(ask_at_fork=False, **kw))
    P = r.trunk.path if r.trunk is not None else np.zeros((0, 3))
    end = float(P[:, 0].max()) if len(P) else 12.0
    off = (float(np.hypot(P[:, 1] - CY, P[:, 2] - CZ).max()) if len(P)
           else 0.0)
    ev = [e.get("event") for e in r.log if e.get("event")]
    return end, off, ev


# the rescues switched off (the HU-peak one, 2026-11-26, follows it too)
OFF = dict(round_rescue=False, peak_rescue=False)
for seed in (0, 1):
    vol = make(190.0, seed=seed)
    e0, _o0, _ = run(vol, **OFF)
    e1, o1, ev = run(vol)
    check(f"seed {seed}: without it the trace stops where the contrast drops",
          e0 < 30.0, e0)
    check(f"seed {seed}: with it the faint round vessel is followed on",
          e1 > 50.0 and ("faint round lumen" in ev or "HU peak" in ev),
          (e1, ev[:4]))
    check(f"seed {seed}: … on its centre (< 0.5 mm)", o1 < 0.5, o1)

# a vessel that simply ends: no rescue into the background
e, _o, ev = run(make(420.0, end_x=40.0))
check("a plain vessel end still stops (nothing taken from noise)",
      38.0 <= e <= 44.0, (e, ev[-2:]))
# a faint flat band past the end: not round → not taken
e, _o, ev = run(make(190.0, end_x=40.0, band=True))
check("a faint FLAT band past the end is not taken",
      e <= 44.0 and ev.count("faint round lumen") <= 6, (e, ev[-3:]))
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
sys.stdout.flush()
os._exit(1 if FAILS else 0)
