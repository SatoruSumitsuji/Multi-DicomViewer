"""MV-AoV transition point from one rough click (core/mv_aov_point) — headless.

Uses the aortic-root + LV phantom of tools/test_valve_auto.py (diastole and
systole). The point (the aortic annulus point toward the mitral centre) must be
found within 2.5 mm from clicks scattered inside the aortic root, and the
result must not depend much on where exactly the user clicked.

    python tools/test_mv_aov_point.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from multi_dicomviewer.core.mv_aov_point import detect_mv_aov_point  # noqa: E402

FAILS = []


def chk(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


src = (ROOT / "tools" / "test_valve_auto.py").read_text(encoding="utf-8")
head = src.split("vol_dia = finish(build(24.0, False))")[0]
ns = {"__name__": "phantom", "__file__": str(ROOT / "tools" / "test_valve_auto.py")}
exec(compile(head, "phantom", "exec"), ns)

for name, args in (("diastole", (24.0, False)), ("systole", (15.5, True))):
    vol = ns["finish"](ns["build"](*args))
    C_ao, n_ao, C_mv, R = ns["C_ao"], ns["n_ao"], ns["C_mv"], ns["R_ann"]
    d = C_mv - C_ao
    d = d - (d @ n_ao) * n_ao
    d /= np.linalg.norm(d)
    P_true = C_ao + R * d
    pts = []
    for off in ([0, 0, 0], [4, -3, 2], [-3, 4, -4]):
        click = C_ao + np.array(off, float) + 6 * n_ao
        r = detect_mv_aov_point(vol, ns["dims"], click)
        err = float(np.linalg.norm(r.point - P_true))
        # (The phantom's apex lies outside the image → ok may be False with
        # the point still set; the point is what is checked here.)
        chk(f"{name} click {off}: point", err <= 2.5, f"err {err:.1f} mm")
        ang = float(np.degrees(np.arccos(np.clip(r.mv_dir @ d, -1, 1))))
        chk(f"{name} click {off}: mitral direction", ang <= 12.0, f"{ang:.0f} deg")
        pts.append(r.point)
    P = np.asarray(pts)
    spread = float(np.linalg.norm(P - P.mean(axis=0), axis=1).max())
    chk(f"{name}: reproducible between clicks", spread <= 2.0, f"{spread:.1f} mm")

# A click in air: a clean failure, no exception.
vol = ns["finish"](ns["build"](24.0, False))
r = detect_mv_aov_point(vol, ns["dims"], np.array([1.0, 1.0, 1.0]))
chk("click outside → not ok", not r.ok and bool(r.message), r.message)

print("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL: {FAILS}")
sys.exit(1 if FAILS else 0)
