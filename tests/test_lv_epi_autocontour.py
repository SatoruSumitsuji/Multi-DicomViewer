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
from multi_dicomviewer.core.lv_measure import LVModel


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
    # A radial point traced at r=21 on the +x meridian (3 mm inside the 24 mm
    # wall — within the tighter max_move), mid-level (along=20).
    along, theta, r_trace = 20.0, 0.0, 21.0
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
    # traced ~3 mm INSIDE the true epicardium (r≈21, within the tighter max_move).
    along_levels = [8, 14, 20, 26, 30]
    pos = [axis.to_world(0.0, 21.0, a) for a in along_levels]      # +x wall
    apex = [axis.to_world(0.0, 0.0, 0.0)]
    neg = [axis.to_world(180.0, 21.0, a) for a in reversed(along_levels)]  # -x wall
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
    # Trace all 8 planes ~3 mm inside the true epicardium (within max_move).
    for phi in m.plane_angles():
        pos = [axis.to_world(phi, 21.0, a) for a in (8, 16, 24, 30)]
        apex = [axis.to_world(phi, 0.0, 0.0)]
        neg = [axis.to_world((phi + 180) % 360, 21.0, a) for a in (30, 24, 16, 8)]
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


def _bumped_model(bump_r=30.0, base_r=19.0):
    """A traced model whose θ=0 meridian is bumped OUT to *bump_r* (a hand edit),
    the rest a uniform *base_r* circle — for the geometric (HU-free) short-axis
    spline tests."""
    from multi_dicomviewer.core.lv_measure import LVModel
    axis = _axis()
    m = LVModel(n_planes=8)
    m.set_axis_from_frame(origin=np.array([60.0, 60.0, 10.0]),
                          axis_dir=np.array([0.0, 0.0, 1.0]),
                          radial0=np.array([1.0, 0.0, 0.0]), which="epi")
    m.set_apex_point("epi", axis.to_world(0.0, 0.0, 0.0))
    for phi in m.plane_angles():
        rp = bump_r if abs(phi) < 1e-6 else base_r      # θ=0 wall bumped out
        pos = [axis.to_world(phi, rp, a) for a in (8, 16, 24, 30)]
        apex = [axis.to_world(phi, 0.0, 0.0)]
        neg = [axis.to_world((phi + 180) % 360, base_r, a) for a in (30, 24, 16, 8)]
        m.set_long_axis_contour(phi, np.array(pos + apex + neg), which="epi")
    return m, axis


def test_short_axis_geometric_smooths_bump():
    # No HU: the correction circumferentially smooths the CURRENT contour, so a
    # single hand-bumped meridian is reduced (blended) but still above baseline,
    # and its neighbours are pulled out a little (influence spread).
    m, _axis = _bumped_model(bump_r=30.0, base_r=19.0)
    before0 = m.epi_contours[0.0][:, 1].max()           # ~30
    stats = ea.refine_epi_short_axis(None, None, m)
    assert stats["planes"] == 8 and stats["levels"] > 0
    after0 = m.epi_contours[0.0][:, 1].max()
    assert 19.5 < after0 < before0, f"bump should smooth, got {after0:.1f}"
    neigh = sorted(k for k in m.epi_contours if 1e-6 < k < 180.0)[0]
    assert m.epi_contours[neigh][:, 1].max() > 19.3, "neighbour should feel the bump"
    # Sparse output: ~8 ring points per wall + the shared apex (not a dense line).
    for phi in m.plane_angles():
        assert len(m.epi_planes[phi % 360.0]) <= 17, "contour should be sparse"


def test_short_axis_geometric_keeps_a_circle():
    # A uniform circle stays a circle (smoothing a constant ring is a no-op).
    m, _vol, dims = _traced_model(r_trace=19.0)
    ea.refine_epi_short_axis(None, None, m)
    for prof in m.epi_contours.values():
        assert abs(prof[:, 1].max() - 19.0) < 1.0, "circle should be preserved"


def test_short_axis_rerun_is_stable():
    m, _axis = _bumped_model()
    ea.refine_epi_short_axis(None, None, m)
    first = {k: v[:, 1].max() for k, v in m.epi_contours.items()}
    ea.refine_epi_short_axis(None, None, m)
    for k, prof in m.epi_contours.items():
        assert abs(prof[:, 1].max() - first[k]) <= 1.0, "re-run should be stable"


def _add_point(m, axis, phi, along, r):
    """Hand edit: insert a long-axis control point on plane *phi*'s φ wall."""
    pts = [np.asarray(p, float) for p in m.epi_planes[phi % 360.0]]
    new = axis.to_world(phi, r, along)
    al = [float((p - axis.apex) @ axis.axis) for p in pts]
    api = int(np.argmin(al))                      # φ wall = base → apex, before api
    i = next((j for j in range(api) if al[j] < along), api)
    pts.insert(i, new)
    m.set_long_axis_contour(phi, np.array(pts), which="epi")


def test_short_axis_keep_edits_adds_ring_for_added_point():
    # HU＋spline → default 8 rings; an add-point between two rings is then the
    # spline INPUT on the re-run (its own ring), not resampled away.
    m, axis = _bumped_model(bump_r=19.0, base_r=19.0)          # plain circle
    st = ea.refine_epi_short_axis(None, None, m)
    assert st["levels"] == 8 and st["extra_levels"] == 0
    amax = max(m.epi_sax_levels)
    a_new = 0.375 * amax                                        # between rings
    _add_point(m, axis, 0.0, a_new, 27.0)
    old = ea.refine_epi_short_axis(None, None, LVModel.from_dict(m.to_dict()))
    st = ea.refine_epi_short_axis(None, None, m, keep_edits=True)
    assert old["levels"] == 8
    assert st["levels"] == 9 and st["extra_levels"] == 1
    prof = m.epi_contours[0.0]
    j = int(np.argmin(np.abs(prof[:, 0] - a_new)))
    assert abs(prof[j, 0] - a_new) < 1.0, "a ring at the added point's height"
    assert prof[j, 1] > 20.0, "the hand edit must survive the spline"
    # The added ring is remembered: a further re-run keeps it, no new rings.
    st = ea.refine_epi_short_axis(None, None, m, keep_edits=True)
    assert st["levels"] == 9 and st["extra_levels"] == 1
    # …and round-trips through Save / Load.
    m2 = LVModel.from_dict(m.to_dict())
    assert m2.epi_sax_extra == m.epi_sax_extra
    st = ea.refine_epi_short_axis(None, None, m2, keep_edits=True)
    assert st["levels"] == 9


def test_short_axis_keep_edits_no_edit_no_ring():
    # A radial-only move ON a ring adds no ring (it is already sampled there).
    m, axis = _bumped_model(bump_r=19.0, base_r=19.0)
    ea.refine_epi_short_axis(None, None, m)
    pts = np.array(m.epi_planes[0.0], float)
    a1 = float((pts[1] - axis.apex) @ axis.axis)
    pts[1] = axis.to_world(0.0, 25.0, a1)
    m.set_long_axis_contour(0.0, pts, which="epi")
    st = ea.refine_epi_short_axis(None, None, m, keep_edits=True)
    assert st["levels"] == 8 and st["extra_levels"] == 0
    assert m.epi_contours[0.0][:, 1].max() > 20.0


def test_short_axis_keep_edits_without_previous_run_is_default():
    m, _axis = _bumped_model()
    st = ea.refine_epi_short_axis(None, None, m, keep_edits=True)
    assert st["levels"] == 8 and st["extra_levels"] == 0


def test_short_axis_default_has_ring_between_last_ring_and_mv():
    # Default rings: one more between 0.875 and the MV plane (the basal terminal)
    # and one more between the apex tip and 0.125.
    m, _axis = _bumped_model(bump_r=19.0, base_r=19.0)
    st = ea.refine_epi_short_axis(None, None, m)
    assert st["levels"] == 8
    amax = max(m.epi_sax_levels)
    fr = [round(x / amax, 4) for x in m.epi_sax_levels]
    assert fr == [0.0625, 0.125, 0.25, 0.5, 0.75, 0.875, 0.9375, 1.0]
    # 8 rings per wall + the shared apex.
    for phi in m.plane_angles():
        assert len(m.epi_planes[phi % 360.0]) == 17
