"""MV-AoV tent (the Epi base of the LV mode) — headless.

The Epi long axis runs apex → MV-AoV point; every Epi border runs into that
point at both basal ends, so the 2 × N lines close the Epi like a tent and the
Epi volume is read WITHOUT any MV / AoV plane clip. Also: plane counts
(Settings → Epi境界設定断面数), tent_terminal_ok, LVSurface.nearest_point /
plane_section (used by the CPR Epi snap + short-axis Epi line).

    python tools/test_lv_epi_tent.py
"""
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from multi_dicomviewer.core import settings  # noqa: E402
from multi_dicomviewer.core.lv_measure import (LV_PLANE_COUNTS, LVModel,  # noqa: E402
                                               tent_terminal_ok)

FAILS = []


def chk(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


R, H, L = 24.0, 60.0, 80.0          # radius, apex-part height, point height


def r_of(t):
    """Half-ellipsoid apex (0..H), then a straight cone into the point (H..L)."""
    if t <= H:
        return R * math.sqrt(max(0.0, 1.0 - ((H - t) / H) ** 2))
    return R * (L - t) / (L - H)


def border(ax, phi, n=80):
    """One plane's border: point → apex on +phi, apex → point on phi+180."""
    ts = np.linspace(L, 0.0, n)
    a = [ax.to_world(phi, r_of(t), t) for t in ts]
    b = [ax.to_world(phi + 180.0, r_of(t), t) for t in ts[::-1]]
    return np.asarray(a + b)


def tent_model(n_planes):
    m = LVModel(n_planes=n_planes)
    m.set_axis(np.array([1.0, 0.0, L]), np.array([-1.0, 0.0, L]),
               np.array([0.0, 0.0, 0.0]))
    m.epi_tent = True
    for phi in m.plane_angles():
        m.set_long_axis_contour(phi, border(m.axis, phi), which="epi")
    m.build()
    return m


V_true = (2.0 / 3.0 * math.pi * R * R * H + math.pi * R * R * (L - H) / 3.0) / 1000.0

# ---- plane counts ---------------------------------------------------------------
chk("plane counts 4..16", LV_PLANE_COUNTS == (4, 6, 8, 10, 12, 16))
for npl in (10, 12, 16):
    mm = LVModel(n_planes=npl)
    chk(f"n_planes {npl} angles", len(mm.plane_angles()) == npl
        and abs(mm.plane_angles()[1] - 180.0 / npl) < 1e-9)
try:
    LVModel(n_planes=5)
    chk("n_planes 5 rejected", False)
except ValueError:
    chk("n_planes 5 rejected", True)

# ---- tent volume, no plane clip ----------------------------------------------------
m = tent_model(8)
chk("epi built", m.epi is not None)
chk("surface flagged tent", bool(getattr(m.epi, "tent", False)))
v0 = m.volume_ml_valves(0.5, "epi")
mv = (np.array([0.0, 0.0, 70.0]), np.array([0.0, 0.0, 1.0]), 20.0)
aov = (np.array([10.0, 0.0, 65.0]), np.array([0.3, 0.0, 1.0]), 11.0)
v1 = m.volume_ml_valves(0.5, "epi", mv, aov)
chk("tent volume ~ analytic", v0 is not None and abs(v0 - V_true) / V_true < 0.05,
    f"{v0} vs {V_true:.2f}")
chk("tent ignores MV/AoV planes", v1 is not None and abs(v1 - v0) < 1e-6,
    f"{v1} vs {v0}")
# Without the tent the same planes DO cut the volume (sanity of the test).
m.epi_tent = False
m.epi.tent = False
v2 = m.volume_ml_valves(0.5, "epi", None, aov)
chk("non-tent AoV plane cuts", v2 is not None and v2 < v0 - 1.0, f"{v2} vs {v0}")
m.epi_tent = True
m.epi.tent = True

# ---- persistence ----------------------------------------------------------------------
d = m.to_dict()
chk("to_dict epi_tent", d.get("epi_tent") is True)
m2 = LVModel.from_dict(d)
chk("from_dict epi_tent", m2.epi_tent is True and m2.n_planes == 8)
d_bad = dict(d, n_planes=7)
chk("from_dict bad n_planes → 8", LVModel.from_dict(d_bad).n_planes == 8)

# ---- tent_terminal_ok ---------------------------------------------------------------------
P = np.array([[0, 0, 10.0], [0, 0, 79.0], [0, 0, 79.6], [0, 0, 85.0]])
ok = tent_terminal_ok(m.axis, P, np.array([0.0, 0.0, L]))
chk("tent_terminal_ok", ok.tolist() == [True, True, False, False], str(ok))

# ---- nearest_point / plane_section ----------------------------------------------------------
S = m.epi
q = S.nearest_point(np.array([40.0, 0.0, 30.0]))   # outside, at mid-height
rq = math.hypot(q[0], q[1])
chk("nearest_point on the wall", abs(rq - r_of(q[2])) < 1.0
    and abs(q[2] - 30.0) < 6.0, str(np.round(q, 2)))
q2 = S.nearest_point(np.array([0.0, 5.0, 30.0]))   # inside
chk("nearest_point from inside", abs(math.hypot(q2[0], q2[1]) - r_of(q2[2])) < 1.0)
segs = S.plane_section(np.array([0.0, 0.0, 30.0]), np.array([0.0, 0.0, 1.0]))
chk("plane_section shape", segs.ndim == 3 and segs.shape[1:] == (2, 3)
    and len(segs) >= 20, str(segs.shape))
rr = np.hypot(segs[..., 0], segs[..., 1])
chk("plane_section radius", abs(float(rr.mean()) - r_of(30.0)) < 1.0
    and np.allclose(segs[..., 2], 30.0, atol=1e-6),
    f"{rr.mean():.2f} vs {r_of(30.0):.2f}")
chk("plane_section miss → empty", len(S.plane_section(
    np.array([0.0, 0.0, 200.0]), np.array([0.0, 0.0, 1.0]))) == 0)

# ---- 16 planes also works -------------------------------------------------------------
m16 = tent_model(16)
v16 = m16.volume_ml_valves(0.5, "epi", mv, aov)
chk("16-plane tent volume", v16 is not None and abs(v16 - V_true) / V_true < 0.05,
    f"{v16}")

# ---- Settings → Epi境界設定断面数 (a temp file, never the real settings) -------------------
with tempfile.TemporaryDirectory() as td:
    orig = settings.LV_ENDO_PATH
    settings.LV_ENDO_PATH = Path(td) / "lv_endo.json"
    try:
        chk("default 8", settings.lv_epi_n_planes() == 8)
        settings.save_lv_endo_params(dict(settings.load_lv_endo_params(),
                                          epi_n_planes=12))
        chk("saved 12", settings.lv_epi_n_planes() == 12)
        settings.LV_ENDO_PATH.write_text(json.dumps({"epi_n_planes": 7}),
                                         encoding="utf-8")
        chk("invalid 7 → 8", settings.lv_epi_n_planes() == 8)
        settings.save_lv_endo_params(dict(settings.load_lv_endo_params(),
                                          epi_n_planes=5))
        chk("save invalid 5 → 8", settings.lv_epi_n_planes() == 8)
    finally:
        settings.LV_ENDO_PATH = orig

print("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL: {FAILS}")
sys.exit(1 if FAILS else 0)
