"""Unit tests for core.lv_epi_autocontour — HU-guided Epi contour refinement.

Headless, Qt/VTK-free. A synthetic cylinder-ish myocardium (soft-tissue wall in
a known radius band, fat outside) lets us check that a trace placed INSIDE the
true epicardium snaps outward to the wall edge where the HU drop is large, and
stays put where the surrounding HU is made ambiguous (small drop).
"""
import numpy as np
import pytest

from multi_dicomviewer.core.lv_axis import LVAxis
from multi_dicomviewer.core import lv_epi_autocontour as ea


def _axis():
    # Apex at bottom, long axis +z; radial0 = +x, binormal = +y.
    return LVAxis.from_frame(origin=np.array([60.0, 60.0, 10.0]),
                             axis_dir=np.array([0.0, 0.0, 1.0]),
                             radial0=np.array([1.0, 0.0, 0.0]))


def _volume(true_epi_r=24.0, ambiguous_x_neg=False):
    """[z,y,x] HU: blood pool (300) in the cavity, myocardium (90) in a wall band
    up to true_epi_r, fat (-90) outside. If *ambiguous_x_neg*, the -x side outside
    the wall is set to 70 HU (≈myocardium) so there is NO clear epi edge there."""
    nz, ny, nx = 60, 120, 120
    sx = sy = sz = 1.0
    cx, cy = 60.0, 60.0
    Y, X = np.mgrid[0:ny, 0:nx].astype(float)
    rr = np.hypot(X - cx, Y - cy)
    endo_r = 14.0
    slice2d = np.full((ny, nx), -90.0)       # fat outside
    slice2d[rr <= true_epi_r] = 90.0         # myocardium wall
    slice2d[rr <= endo_r] = 300.0            # blood cavity
    if ambiguous_x_neg:
        amb = (X < cx) & (rr > true_epi_r)   # left half, outside the wall
        slice2d[amb] = 70.0                  # ~soft tissue → no clear edge
    vol = np.repeat(slice2d[None, :, :], nz, axis=0).astype(np.float32)
    return vol, (sx, sy, sz)


def test_snaps_outward_to_wall_edge():
    axis = _axis()
    vol, dims = _volume(true_epi_r=24.0)
    # A radial point traced at r=19 on the +x meridian, mid-level (along=20).
    along, theta, r_trace = 20.0, 0.0, 19.0
    r_edge, w = ea.detect_epi_radius(vol, dims, axis, theta, along, r_trace,
                                     ea.EpiSnapParams())
    assert w > 0.6, f"expected a confident edge, got w={w}"
    assert 22.5 <= r_edge <= 25.5, f"edge should be ~24 mm, got {r_edge}"


def test_keeps_manual_where_hu_ambiguous():
    axis = _axis()
    vol, dims = _volume(true_epi_r=24.0, ambiguous_x_neg=True)
    # -x meridian (theta=180) is the ambiguous side (70 HU outside the wall).
    r_edge, w = ea.detect_epi_radius(vol, dims, axis, 180.0, 20.0, 19.0,
                                     ea.EpiSnapParams())
    assert w < 0.25, f"ambiguous side should stay manual, got w={w}"


def test_refine_plane_moves_points_outward():
    axis = _axis()
    vol, dims = _volume(true_epi_r=24.0)
    # A rough Epi polyline on the x-z plane: base(+x) -> apex -> base(-x),
    # traced a few mm INSIDE the true epicardium (r≈19).
    along_levels = [8, 14, 20, 26, 30]
    pos = [axis.to_world(0.0, 19.0, a) for a in along_levels]      # +x wall
    apex = [axis.to_world(0.0, 0.0, 0.0)]
    neg = [axis.to_world(180.0, 19.0, a) for a in reversed(along_levels)]  # -x wall
    trace = np.array(pos + apex + neg)
    out, n_ref, meanw = ea.refine_epi_plane(vol, dims, axis, trace)
    assert n_ref > 0 and meanw > 0.5
    # Mid-wall points should now sit ~24 mm out (within the wall band), not 19.
    radii = np.array([axis.project(p)[1] for p in out])
    mid = radii[(radii > 6)]                      # exclude apex region
    assert mid.mean() > 22.0, f"mean refined radius {mid.mean():.1f} should rise toward 24"


def test_refine_model_updates_contours():
    from multi_dicomviewer.core.lv_measure import LVModel
    axis = _axis()
    vol, dims = _volume(true_epi_r=24.0)
    m = LVModel(n_planes=8)
    m.set_axis_from_frame(origin=np.array([60.0, 60.0, 10.0]),
                          axis_dir=np.array([0.0, 0.0, 1.0]),
                          radial0=np.array([1.0, 0.0, 0.0]), which="epi")
    m.set_apex_point("epi", axis.to_world(0.0, 0.0, 0.0))
    # Trace all 8 planes a few mm inside the true epicardium.
    for phi in m.plane_angles():
        pos = [axis.to_world(phi, 19.0, a) for a in (8, 16, 24, 30)]
        apex = [axis.to_world(phi, 0.0, 0.0)]
        neg = [axis.to_world((phi + 180) % 360, 19.0, a) for a in (30, 24, 16, 8)]
        m.set_long_axis_contour(phi, np.array(pos + apex + neg), which="epi")
    before = {k: v.copy() for k, v in m.epi_contours.items()}
    stats = ea.refine_epi_model(vol, dims, m)
    assert stats["planes"] == 8 and stats["refined_pts"] > 0
    # At least one meridian's max radius grew toward the wall.
    grew = 0
    for k, prof in m.epi_contours.items():
        if k in before and prof[:, 1].max() > before[k][:, 1].max() + 1.0:
            grew += 1
    assert grew >= 6, f"expected most meridians to grow outward, got {grew}"


def _traced_model(true_epi_r=24.0, r_trace=19.0):
    from multi_dicomviewer.core.lv_measure import LVModel
    axis = _axis()
    vol, dims = _volume(true_epi_r=true_epi_r)
    m = LVModel(n_planes=8)
    m.set_axis_from_frame(origin=np.array([60.0, 60.0, 10.0]),
                          axis_dir=np.array([0.0, 0.0, 1.0]),
                          radial0=np.array([1.0, 0.0, 0.0]), which="epi")
    m.set_apex_point("epi", axis.to_world(0.0, 0.0, 0.0))
    for phi in m.plane_angles():
        pos = [axis.to_world(phi, r_trace, a) for a in (8, 16, 24, 30)]
        apex = [axis.to_world(phi, 0.0, 0.0)]
        neg = [axis.to_world((phi + 180) % 360, r_trace, a) for a in (30, 24, 16, 8)]
        m.set_long_axis_contour(phi, np.array(pos + apex + neg), which="epi")
    return m, vol, dims


def test_short_axis_ellipse_grows_outward():
    m, vol, dims = _traced_model()
    before = {k: v[:, 1].max() for k, v in m.epi_contours.items()}
    stats = ea.refine_epi_short_axis(vol, dims, m, mode="ellipse")
    assert stats["planes"] == 8 and stats["levels"] > 0
    grew = sum(1 for k, prof in m.epi_contours.items()
               if prof[:, 1].max() > before.get(k, 0.0) + 1.0)
    assert grew >= 6, f"ellipse should pull most meridians to ~24, got {grew}"
    # And the fitted body radius should be near the true 24 mm (not the 19 trace).
    rmax = max(prof[:, 1].max() for prof in m.epi_contours.values())
    assert 22.0 <= rmax <= 26.0, f"ellipse body radius {rmax:.1f} off"


def test_short_axis_spline_grows_outward():
    m, vol, dims = _traced_model()
    before = {k: v[:, 1].max() for k, v in m.epi_contours.items()}
    stats = ea.refine_epi_short_axis(vol, dims, m, mode="spline")
    assert stats["planes"] == 8 and stats["levels"] > 0
    grew = sum(1 for k, prof in m.epi_contours.items()
               if prof[:, 1].max() > before.get(k, 0.0) + 1.0)
    assert grew >= 6, f"spline should pull most meridians to ~24, got {grew}"


def test_short_axis_safe_without_edges():
    # A uniform (no-edge) volume must not blow up the contours — keep manual.
    m, _vol, dims = _traced_model()
    flat = np.full_like(_vol, 50.0)                # soft tissue everywhere, no drop
    before = {k: v[:, 1].max() for k, v in m.epi_contours.items()}
    ea.refine_epi_short_axis(flat, dims, m, mode="ellipse")
    for k, prof in m.epi_contours.items():
        assert prof[:, 1].max() <= before[k] + 1.5, "no-edge case must stay ~manual"
