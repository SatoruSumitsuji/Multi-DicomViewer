"""Headless tests for the LV geometry core (core/lv_axis + core/lv_surface).

Validates the Phase-0 volume engine against shapes with a known analytic
volume (ellipsoids), for both an axis-aligned and an oblique long axis, plus
the ring-edit path. Pure numpy - no Qt / VTK - so it runs anywhere:

    python tools/test_lv_volume.py
"""
import math
import sys

sys.path.insert(0, r"C:\CC_Product\Multi-DicomViewer")

import numpy as np  # noqa: E402

from multi_dicomviewer.core.lv_axis import LVAxis          # noqa: E402
from multi_dicomviewer.core.lv_measure import LVModel      # noqa: E402
from multi_dicomviewer.core.lv_surface import LVSurface    # noqa: E402


def _ellipsoid_contours(axis: LVAxis, a_x: float, a_y: float,
                        n_meridians: int = 12, n_along: int = 60) -> dict:
    """Meridian (along, radius) contours of a prolate ellipsoid with radial
    semi-axes *a_x* (along radial0) and *a_y* (along binormal), long semi-axis
    = length_mm / 2. Apex and base are the two poles (radius -> 0). Analytic
    volume = 4/3 π a_x a_y c."""
    length = axis.length_mm
    c = length / 2.0
    t = np.linspace(0.0, length, n_along)
    ax_term = (t - c) ** 2 / (c * c)
    contours = {}
    for m in range(n_meridians):
        th = 360.0 * m / n_meridians
        a = math.radians(th)
        denom = math.sqrt((math.cos(a) / a_x) ** 2 + (math.sin(a) / a_y) ** 2)
        r = np.sqrt(np.clip(1.0 - ax_term, 0.0, None)) / denom
        contours[th] = np.column_stack([t, r])
    return contours


def _ellipsoid_plane_pts(axis, a_x, a_y, phi, length, n=40):
    """A traced long-axis border polyline (N,3) for the rotation plane *phi* of a
    prolate ellipsoid (semi-axes a_x, a_y, long = length/2): the φ+180 wall
    base→apex then the φ wall apex→base (through the apex pole). Mirrors what the
    viewer captures per plane so the model gets both endo_planes + endo_contours
    (via set_long_axis_contour)."""
    c = length / 2.0

    def r_at(theta, t):
        a = math.radians(theta)
        denom = math.sqrt((math.cos(a) / a_x) ** 2 + (math.sin(a) / a_y) ** 2)
        return math.sqrt(max(0.0, 1.0 - (t - c) ** 2 / (c * c))) / denom

    pts = []
    for t in np.linspace(length, 0.0, n):            # φ+180 wall: base → apex
        pts.append(axis.to_world(phi + 180.0, r_at(phi + 180.0, t), t))
    for t in np.linspace(0.0, length, n):            # φ wall: apex → base
        pts.append(axis.to_world(phi, r_at(phi, t), t))
    return np.asarray(pts, float)


def _check(name, got, want, tol_frac):
    err = abs(got - want) / want
    ok = err <= tol_frac
    print(f"  {name}: got {got:.2f} mL, want {want:.2f} mL "
          f"({err*100:.2f}% vs {tol_frac*100:.0f}% tol) "
          f"{'OK' if ok else 'FAIL'}")
    assert ok, f"{name}: {err*100:.2f}% > {tol_frac*100:.0f}%"


# ---- LVAxis geometry -------------------------------------------------------
apex = np.array([0.0, 0.0, 0.0])
b1 = np.array([15.0, 0.0, 80.0])
b2 = np.array([-15.0, 0.0, 80.0])
ax = LVAxis.from_points(b1, b2, apex)
assert abs(ax.length_mm - 80.0) < 1e-9
assert np.allclose(ax.axis, [0, 0, 1])
assert np.allclose(ax.base_center, [0, 0, 80])
assert np.allclose(ax.radial0, [1, 0, 0]), ax.radial0
assert np.allclose(ax.binormal, [0, 1, 0]), ax.binormal
# orthonormal, right-handed
for u in (ax.radial0, ax.binormal, ax.axis):
    assert abs(np.linalg.norm(u) - 1.0) < 1e-9
assert np.allclose(np.cross(ax.radial0, ax.binormal), ax.axis)
# to_world / project round-trip
P = ax.to_world(37.0, 12.0, 55.0)
al, r, th = ax.project(P)
assert abs(al - 55.0) < 1e-6 and abs(r - 12.0) < 1e-6 and abs(th - 37.0) < 1e-4
print("LVAxis geometry + round-trip: OK")

# ---- A: axis-aligned sphere-ish ellipsoid ---------------------------------
a_x = a_y = 20.0
surf = LVSurface.from_meridian_contours(ax, _ellipsoid_contours(ax, a_x, a_y))
V_true = 4.0 / 3.0 * math.pi * a_x * a_y * (ax.length_mm / 2.0) / 1000.0
print("A) axis-aligned prolate spheroid:")
_check("volume", surf.voxel_volume_ml(0.5), V_true, 0.03)

# ---- B: non-axisymmetric ellipsoid (a_x != a_y) ---------------------------
apexB = np.array([0.0, 0.0, 0.0])
axB = LVAxis.from_points([14, 0, 90], [-14, 0, 90], apexB)
a_x, a_y = 28.0, 18.0
surfB = LVSurface.from_meridian_contours(axB, _ellipsoid_contours(axB, a_x, a_y))
V_true = 4.0 / 3.0 * math.pi * a_x * a_y * (axB.length_mm / 2.0) / 1000.0
print("B) non-axisymmetric ellipsoid:")
_check("volume", surfB.voxel_volume_ml(0.5), V_true, 0.04)

# ---- C: oblique long axis (same shape, tilted) - volume is invariant ------
d = np.array([1.0, 2.0, 6.0])
d = d / np.linalg.norm(d)
apexC = np.array([10.0, 5.0, -3.0])
baseC = apexC + 80.0 * d
# any perpendicular for the basal offset
perp = np.cross(d, [0, 0, 1]); perp = perp / np.linalg.norm(perp)
axC = LVAxis.from_points(baseC + 15 * perp, baseC - 15 * perp, apexC)
assert abs(axC.length_mm - 80.0) < 1e-6
a_x = a_y = 20.0
surfC = LVSurface.from_meridian_contours(axC, _ellipsoid_contours(axC, a_x, a_y))
V_true = 4.0 / 3.0 * math.pi * a_x * a_y * (axC.length_mm / 2.0) / 1000.0
print("C) oblique long axis (orientation-invariant):")
_check("volume", surfC.voxel_volume_ml(0.5), V_true, 0.03)

# ---- D: ring edit changes the volume. Scaling every ring radially by 0.5
# halves the in-plane extent at each level but leaves the axial spacing alone,
# so area x0.25 and volume x0.25 (NOT x1/8 - that would need axial scaling too).
V0 = surf.voxel_volume_ml(0.6)
surf.rings[:] = surf.rings * 0.5           # short-axis correction, uniform
V1 = surf.voxel_volume_ml(0.6)
print("D) ring edit (x0.5 radius -> 1/4 volume):")
_check("scaled volume", V1, V0 / 4.0, 0.05)

# ---- mesh sanity -----------------------------------------------------------
verts, faces = surfB.to_mesh()
assert verts.ndim == 2 and verts.shape[1] == 3
assert faces.ndim == 2 and faces.shape[1] == 3
assert faces.max() < len(verts) and faces.min() >= 0
print(f"mesh: {len(verts)} verts, {len(faces)} faces - OK")

# ---- E: promote Endo onto the Epi axis (independent axes → one frame) -------
# Endo axis: straight up. Epi axis: apex a few mm more apical + slightly tilted
# (a genuinely INDEPENDENT axis, as the 2-pass trace produces). Build the model
# the way the viewer does — set each axis from its view frame, then feed traced
# 3-D border polylines per plane (populates endo_planes AND endo_contours).
ax_endo = LVAxis.from_points([15, 0, 80], [-15, 0, 80], [0, 0, 0])
d = np.array([0.05, 0.03, 1.0]); d = d / np.linalg.norm(d)
apex_epi = np.array([1.5, -1.0, -4.0])
base_epi = apex_epi + 84.0 * d
perp = np.cross(d, [0, 0, 1]); perp = perp / np.linalg.norm(perp)
ax_epi = LVAxis.from_points(base_epi + 15 * perp, base_epi - 15 * perp, apex_epi)

m = LVModel(n_planes=6)
m.set_axis_from_frame(ax_endo.apex, ax_endo.axis, ax_endo.radial0, which="endo")
m.set_axis_from_frame(ax_epi.apex, ax_epi.axis, ax_epi.radial0, which="epi")
for phi in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0):
    m.set_long_axis_contour(phi, _ellipsoid_plane_pts(ax_endo, 18, 18, phi, 80.0),
                            which="endo")
    m.set_long_axis_contour(phi, _ellipsoid_plane_pts(ax_epi, 23, 23, phi, 84.0),
                            which="epi")
m.set_apex_point("endo", ax_endo.apex)
m.set_apex_point("epi", ax_epi.apex)

m.build()
V_endo_pre = m.volume_ml(0.5, "endo")
endo_apex_before = m.endo_apex.copy()

ok = m.promote_endo_to_epi_axis()
assert ok, "promotion returned False"
assert m.endo_axis is m.epi_axis, "endo not on the epi axis after promotion"
assert m.endo_promoted, "endo_promoted flag not set"
assert np.allclose(m.endo_apex, endo_apex_before), "endo apex point changed"
assert len(m.endo_contours) >= 3, "too few endo contours after promotion"

m.build()
V_endo_post = m.volume_ml(0.5, "endo")
print("E) promote Endo onto the Epi axis:")
_check("endo volume (pre vs post promote)", V_endo_post, V_endo_pre, 0.10)
V_myo = m.myocardial_volume_ml(0.5)
assert V_myo is not None and V_myo > 0, f"myo volume invalid: {V_myo}"
assert m.promote_endo_to_epi_axis() is True            # idempotent
V_promoted = m.volume_ml(0.5, "endo")

# ---- F: non-destructive restore (Endo → original independent-axis trace) ----
ok2 = m.restore_endo_original()
assert ok2 and not m.endo_promoted, "restore failed / still promoted"
assert m.endo_axis is not m.epi_axis, "endo axis not restored to its own"
m.build()
print("F) restore original Endo trace:")
_check("endo volume (restored vs original)", m.volume_ml(0.5, "endo"),
       V_endo_pre, 0.01)

# ---- G: v4 persistence — save in the PROMOTED state, reload keeps both ------
import copy  # noqa: E402
m.promote_endo_to_epi_axis()
saved = m.to_dict()
assert saved["version"] == 4 and "endo_orig" in saved, "v4 endo_orig not saved"
m2 = LVModel.from_dict(copy.deepcopy(saved))
assert m2.endo_promoted, "promoted stash lost across save/reload"
m2.build()
print("G) v4 persistence round-trip (promoted state):")
_check("reloaded promoted endo volume", m2.volume_ml(0.5, "endo"),
       V_promoted, 0.02)
assert m2.restore_endo_original(), "reloaded model can't restore original"
m2.build()
_check("reloaded then restored endo volume", m2.volume_ml(0.5, "endo"),
       V_endo_pre, 0.05)

print("\nALL PASS")
