"""Test the view-derived LV axis path (no apex/basal picks): the rotation axis
comes from the current view frame, and the apex/base extent + base-cut plane are
derived from the traced borders. Pure numpy:

    python tools/test_lv_viewaxis.py
"""
import math
import sys

sys.path.insert(0, r"C:\CC_Product\Multi-DicomViewer")

import numpy as np  # noqa: E402

from multi_dicomviewer.core.lv_measure import LVModel   # noqa: E402


def _rad(a_x, a_y, phi, al, c):
    """Half-ellipsoid radius: apex pole (r=0) at along=0, open base (r=max) at
    along=c — a realistic LV shape (base is an annulus, not a pole)."""
    z = a_x * a_y  # not used; keep signature simple
    base = math.sqrt(max(0.0, 1.0 - ((al - c) / c) ** 2))
    denom = math.sqrt((math.cos(math.radians(phi)) / a_x) ** 2
                      + (math.sin(math.radians(phi)) / a_y) ** 2)
    return base / denom


def _plane_border(ax, a_x, a_y, phi, c, n=60):
    """Continuous border on the long-axis plane at rotation phi: base(al=c) →
    apex(al=0) on the +radial wall, then apex → base on the −radial wall."""
    pts = [ax.to_world(phi, _rad(a_x, a_y, phi, al, c), al)
           for al in np.linspace(c, 0.0, n)]
    pts += [ax.to_world(phi + 180, _rad(a_x, a_y, phi + 180, al, c), al)
            for al in np.linspace(0.0, c, n)]
    return np.asarray(pts)


def _check(name, got, want, tol):
    err = abs(got - want) / abs(want)
    print(f"  {name}: {got:.2f} vs {want:.2f} ({err*100:.2f}% / {tol*100:.0f}%) "
          f"{'OK' if err <= tol else 'FAIL'}")
    assert err <= tol, name


# view frame (oblique) — no apex/basal picks
origin = np.array([5.0, 10.0, -2.0])
axis_dir = np.array([1.0, 1.0, 4.0])
radial0 = np.array([1.0, 0.0, 0.0])
a_x = a_y = 20.0
c = 40.0                                   # half-length → full length 80

m = LVModel(n_planes=6)
m.set_axis_from_frame(origin, axis_dir, radial0)
assert m.axis is not None
for phi in m.plane_angles():
    m.set_long_axis_contour(phi, _plane_border(m.axis, a_x, a_y, phi, c),
                            which="endo")

rng = m.along_range("endo")
print("A) view-axis along range (expect ~[0, 40]):",
      tuple(round(x, 1) for x in rng))
assert abs(rng[0]) < 1.5 and abs(rng[1] - c) < 1.0, rng

m.build()
V_true = 2.0 / 3.0 * math.pi * a_x * a_y * c / 1000.0   # half-ellipsoid
print("B) volume from view-derived axis (no picks):")
_check("volume mL", m.volume_ml(0.5), V_true, 0.04)

# C) editing a border's basal end moves the base plane (base = min of maxes)
th0 = sorted(m.endo_contours.keys())[0]
prof = m.endo_contours[th0]
prof[:, 0] = np.clip(prof[:, 0], None, 20.0)   # cap this meridian's basal end
rng2 = m.along_range("endo")
print("C) base moves when a basal end is edited (expect base ~20):",
      round(rng2[1], 1))
assert abs(rng2[1] - 20.0) < 1.0, rng2

print("\nALL PASS")
