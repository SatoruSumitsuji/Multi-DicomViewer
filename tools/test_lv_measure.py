"""End-to-end headless test of the LV measurement controller (core/lv_measure).

Feeds long-axis borders the way the GUI will (one 3-D polyline per rotated
plane, each covering both walls), then checks cavity volume, myocardial mass,
and a two-phase LVEF against analytic ellipsoid values. Pure numpy:

    python tools/test_lv_measure.py
"""
import math
import sys

sys.path.insert(0, r"C:\CC_Product\Multi-DicomViewer")

import numpy as np  # noqa: E402

from multi_dicomviewer.core.lv_axis import LVAxis        # noqa: E402
from multi_dicomviewer.core.lv_measure import LVModel     # noqa: E402


def _ellip_r(a_x, a_y, theta_deg, t, c):
    a = math.radians(theta_deg)
    denom = math.sqrt((math.cos(a) / a_x) ** 2 + (math.sin(a) / a_y) ** 2)
    return math.sqrt(max(0.0, 1.0 - ((t - c) / c) ** 2)) / denom


def _plane_border(ax: LVAxis, a_x, a_y, phi, n=50):
    """A continuous endo/epi border on the long-axis plane at rotation *phi*:
    base->apex on the +radial wall, apex->base on the -radial wall — exactly
    the single stroke a user traces. Returns (2n,3) volume-mm points."""
    length = ax.length_mm
    c = length / 2.0
    pts = []
    for t in np.linspace(length, 0.0, n):
        pts.append(ax.to_world(phi, _ellip_r(a_x, a_y, phi, t, c), t))
    for t in np.linspace(0.0, length, n):
        pts.append(ax.to_world(phi + 180.0,
                               _ellip_r(a_x, a_y, phi + 180.0, t, c), t))
    return np.asarray(pts)


def _fill(model, a_x, a_y, which):
    for phi in model.plane_angles():
        model.set_long_axis_contour(phi, _plane_border(model.axis, a_x, a_y, phi),
                                    which=which)


def _ellip_vol_ml(a_x, a_y, length):
    return 4.0 / 3.0 * math.pi * a_x * a_y * (length / 2.0) / 1000.0


def _check(name, got, want, tol_frac):
    err = abs(got - want) / abs(want)
    ok = err <= tol_frac
    print(f"  {name}: got {got:.2f}, want {want:.2f} "
          f"({err*100:.2f}% vs {tol_frac*100:.0f}%) {'OK' if ok else 'FAIL'}")
    assert ok, f"{name}: {err*100:.2f}% > {tol_frac*100:.0f}%"


ax_pts = (np.array([16.0, 0.0, 85.0]), np.array([-16.0, 0.0, 85.0]),
          np.array([0.0, 0.0, 0.0]))            # basal1, basal2, apex; L=85

# ---- cavity volume from long-axis borders (6 planes / 30 deg) --------------
m = LVModel(n_planes=6)
m.set_axis(*ax_pts)
assert m.plane_angles() == [0, 30, 60, 90, 120, 150]
assert len(m.meridian_angles()) == 12
_fill(m, 22.0, 22.0, "endo")
m.build()
V_endo = _ellip_vol_ml(22.0, 22.0, 85.0)
print("A) cavity volume from long-axis borders:")
_check("endo volume mL", m.volume_ml(0.5), V_endo, 0.03)

# ---- non-axisymmetric cavity (4 planes / 45 deg) --------------------------
m2 = LVModel(n_planes=4)
m2.set_axis(*ax_pts)
assert m2.plane_angles() == [0, 45, 90, 135]
_fill(m2, 26.0, 18.0, "endo")
m2.build()
print("B) non-axisymmetric cavity (4 planes):")
_check("endo volume mL", m2.volume_ml(0.5), _ellip_vol_ml(26.0, 18.0, 85.0), 0.04)

# ---- myocardial mass (endo + epi) -----------------------------------------
_fill(m, 27.0, 27.0, "epi")
m.build()
myo_vol = _ellip_vol_ml(27.0, 27.0, 85.0) - _ellip_vol_ml(22.0, 22.0, 85.0)
print("C) myocardial volume + mass (endo/epi):")
_check("myo volume mL", m.myocardial_volume_ml(0.5), myo_vol, 0.05)
_check("myo mass g", m.myocardial_mass_g(0.5), myo_vol * 1.05, 0.05)

# ---- two-phase LVEF (ED + ES) ---------------------------------------------
ed = LVModel(n_planes=6); ed.set_axis(*ax_pts); _fill(ed, 24.0, 24.0, "endo"); ed.build()
es = LVModel(n_planes=6); es.set_axis(*ax_pts); _fill(es, 16.0, 16.0, "endo"); es.build()
EDV = ed.volume_ml(0.5)
ESV = es.volume_ml(0.5)
EF = (EDV - ESV) / EDV * 100.0
EDV_t = _ellip_vol_ml(24.0, 24.0, 85.0)
ESV_t = _ellip_vol_ml(16.0, 16.0, 85.0)
EF_t = (EDV_t - ESV_t) / EDV_t * 100.0
print("D) two-phase LVEF:")
_check("EDV mL", EDV, EDV_t, 0.03)
_check("ESV mL", ESV, ESV_t, 0.03)
_check("LVEF %", EF, EF_t, 0.03)
print(f"     LVEF = {EF:.1f}% (analytic {EF_t:.1f}%)")

print("\nALL PASS")
