"""Known-answer 'practice problems' for core/coreg.py (IVUS-XA CoReg).

Run directly:    python tools/test_coreg.py
Or with pytest:  pytest tools/test_coreg.py

Each test builds a synthetic guide curve / anchor set where the answer is
known by hand, then checks the interpolation math reproduces it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core import coreg


# --------------------------------------------------------------------------
# 1. A 2-vertex guide is a straight segment; the smooth curve passes
#    through both endpoints unchanged.
# --------------------------------------------------------------------------
def test_smooth_two_points_is_straight():
    curve = coreg.smooth_curve([(0.0, 0.0), (10.0, 0.0)])
    assert curve[0] == (0.0, 0.0)
    assert curve[-1] == (10.0, 0.0)
    assert len(curve) == 2


# --------------------------------------------------------------------------
# 2. The smooth spline still starts and ends exactly on the first / last
#    traced vertex (interpolating, not approximating).
# --------------------------------------------------------------------------
def test_smooth_passes_through_endpoints():
    verts = [(0.0, 0.0), (10.0, 5.0), (20.0, 0.0), (30.0, 5.0)]
    curve = coreg.smooth_curve(verts, samples_per_seg=8)
    assert abs(curve[0][0] - 0.0) < 1e-9 and abs(curve[0][1] - 0.0) < 1e-9
    assert abs(curve[-1][0] - 30.0) < 1e-9 and abs(curve[-1][1] - 5.0) < 1e-9
    # A collinear straight trace stays on the y=0 line (spline adds no wobble).
    flat = coreg.smooth_curve([(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)])
    assert all(abs(y) < 1e-9 for _, y in flat)


# --------------------------------------------------------------------------
# 3. Arc-length addressing on a length-10 horizontal segment: fraction s
#    lands at x = 10*s, and total length is 10.
# --------------------------------------------------------------------------
def test_point_at_fraction_straight():
    curve = [(0.0, 0.0), (10.0, 0.0)]
    assert abs(coreg.arc_length(curve) - 10.0) < 1e-9
    assert coreg.point_at_fraction(curve, 0.0) == (0.0, 0.0)
    assert abs(coreg.point_at_fraction(curve, 0.25)[0] - 2.5) < 1e-9
    assert abs(coreg.point_at_fraction(curve, 0.5)[0] - 5.0) < 1e-9
    # Out-of-range fractions clamp to the ends.
    assert abs(coreg.point_at_fraction(curve, 2.0)[0] - 10.0) < 1e-9
    assert abs(coreg.point_at_fraction(curve, -1.0)[0] - 0.0) < 1e-9


# --------------------------------------------------------------------------
# 4. point_at_fraction respects arc length across a 2-segment L, not vertex
#    count: an L of legs 6 (horizontal) + 4 (vertical) has total 10, and
#    s=0.8 is 8 units along -> 2 units up the vertical leg -> (6, 2).
# --------------------------------------------------------------------------
def test_point_at_fraction_multiseg():
    curve = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.0)]
    assert abs(coreg.arc_length(curve) - 10.0) < 1e-9
    p = coreg.point_at_fraction(curve, 0.8)
    assert abs(p[0] - 6.0) < 1e-9 and abs(p[1] - 2.0) < 1e-9
    # The corner (6,0) sits at s = 6/10 = 0.6.
    c = coreg.point_at_fraction(curve, 0.6)
    assert abs(c[0] - 6.0) < 1e-9 and abs(c[1] - 0.0) < 1e-9


# --------------------------------------------------------------------------
# 5. Projecting a click onto the guide snaps to the nearest foot and
#    reports the perpendicular distance (used as the snap tolerance).
# --------------------------------------------------------------------------
def test_project_fraction():
    curve = [(0.0, 0.0), (10.0, 0.0)]
    s, d = coreg.project_fraction(curve, (3.0, 2.0))     # 3 along, 2 off
    assert abs(s - 0.3) < 1e-9
    assert abs(d - 2.0) < 1e-9
    # A click past the far end clamps to s=1 (foot at the endpoint).
    s2, d2 = coreg.project_fraction(curve, (99.0, 0.0))
    assert abs(s2 - 1.0) < 1e-9


# --------------------------------------------------------------------------
# 6. Frame -> fraction: fewer than two anchors gives None (guide only);
#    two anchors interpolate linearly and clamp outside [0, 1].
# --------------------------------------------------------------------------
def test_map_frame_to_fraction():
    assert coreg.map_frame_to_fraction([], 5) is None
    assert coreg.map_frame_to_fraction([(0, 0.2)], 5) is None
    anchors = [(0, 0.0), (100, 1.0)]
    assert abs(coreg.map_frame_to_fraction(anchors, 0) - 0.0) < 1e-9
    assert abs(coreg.map_frame_to_fraction(anchors, 50) - 0.5) < 1e-9
    assert abs(coreg.map_frame_to_fraction(anchors, 100) - 1.0) < 1e-9
    # Extrapolation beyond the anchors is clamped into [0, 1].
    assert coreg.map_frame_to_fraction(anchors, 999) == 1.0
    assert coreg.map_frame_to_fraction(anchors, -50) == 0.0


# --------------------------------------------------------------------------
# 7. Three anchors interpolate piecewise-linearly between the bracketing
#    pair (a skipped-in-the-middle segment still bends the mapping).
# --------------------------------------------------------------------------
def test_map_frame_piecewise():
    anchors = [(0, 0.0), (40, 0.2), (100, 1.0)]
    # Between frames 40 and 100: s goes 0.2 -> 1.0, so frame 70 -> 0.6.
    assert abs(coreg.map_frame_to_fraction(anchors, 70) - 0.6) < 1e-9
    # Between 0 and 40: frame 20 -> 0.1.
    assert abs(coreg.map_frame_to_fraction(anchors, 20) - 0.1) < 1e-9


# --------------------------------------------------------------------------
# 8. is_monotonic rejects a fold-back (frame rises but fraction dips) and
#    duplicate frames; accepts a clean rising set.
# --------------------------------------------------------------------------
def test_is_monotonic():
    assert coreg.is_monotonic([(0, 0.0), (50, 0.5), (100, 1.0)])
    assert coreg.is_monotonic([(0, 1.0), (50, 0.5), (100, 0.0)])   # falling ok
    assert not coreg.is_monotonic([(0, 0.0), (50, 0.6), (100, 0.5)])
    assert not coreg.is_monotonic([(0, 0.0), (0, 0.5)])            # dup frame


# --------------------------------------------------------------------------
# 9. End-to-end: on a length-10 straight guide, frame 50 of a 0->100
#    pull-back lands the marker at the midpoint (5, 0).
# --------------------------------------------------------------------------
def test_marker_point_end_to_end():
    curve = coreg.smooth_curve([(0.0, 0.0), (10.0, 0.0)])
    anchors = [(0, 0.0), (100, 1.0)]
    p = coreg.marker_point(curve, anchors, 50)
    assert abs(p[0] - 5.0) < 1e-9 and abs(p[1] - 0.0) < 1e-9
    # Too few anchors -> no marker.
    assert coreg.marker_point(curve, [(0, 0.0)], 50) is None


# --------------------------------------------------------------------------
# 10. map_fraction_to_frame is the inverse of map_frame_to_fraction: on a
#     0->100 pull-back, fraction 0.5 maps back to frame 50 (used when the
#     user drags the on-guide marker to scrub the IVUS).
# --------------------------------------------------------------------------
def test_map_fraction_to_frame_inverse():
    anchors = [(0, 0.0), (100, 1.0)]
    assert coreg.map_fraction_to_frame([(0, 0.0)], 0.5) is None
    assert abs(coreg.map_fraction_to_frame(anchors, 0.0) - 0.0) < 1e-9
    assert abs(coreg.map_fraction_to_frame(anchors, 0.5) - 50.0) < 1e-9
    assert abs(coreg.map_fraction_to_frame(anchors, 1.0) - 100.0) < 1e-9
    # Round-trip a piecewise set: frame -> s -> frame returns the frame.
    an = [(0, 0.0), (40, 0.2), (100, 1.0)]
    s70 = coreg.map_frame_to_fraction(an, 70)
    assert abs(coreg.map_fraction_to_frame(an, s70) - 70.0) < 1e-6


# --------------------------------------------------------------------------
# 11. map_between syncs one IVUS's frame to another's from shared landmarks:
#     pairs (frameA, frameB) with A=0->B=0 and A=100->B=50 maps A=40 -> 20.
# --------------------------------------------------------------------------
def test_map_between():
    pairs = [(0, 0), (100, 50)]
    assert coreg.map_between([], 40) is None             # 0 pairs → no sync
    # 1 pair → 1:1 offset: anchor (10, 30), master 15 → other 35 (moved +5).
    assert abs(coreg.map_between([(10, 30)], 15) - 35.0) < 1e-9
    assert abs(coreg.map_between([(10, 30)], 5) - 25.0) < 1e-9
    assert abs(coreg.map_between(pairs, 40) - 20.0) < 1e-9
    assert abs(coreg.map_between(pairs, 100) - 50.0) < 1e-9
    # Extrapolates past the ends (caller clamps to the frame range).
    assert abs(coreg.map_between(pairs, 200) - 100.0) < 1e-9


# --------------------------------------------------------------------------
# 12. map_rotation interpolates the cross-section angle (shortest path),
#     constant-clamped outside; 350°→10° sweeps forward through 0.
# --------------------------------------------------------------------------
def test_map_rotation():
    assert coreg.map_rotation([], 5) is None
    pairs = [(0, 0.0), (100, 90.0)]
    assert abs(coreg.map_rotation(pairs, 50) - 45.0) < 1e-9
    assert abs(coreg.map_rotation(pairs, -10) - 0.0) < 1e-9      # clamp lo
    assert abs(coreg.map_rotation(pairs, 200) - 90.0) < 1e-9     # clamp hi
    # Shortest path 350 -> 10 sweeps +20 through 0, midpoint ≈ 0/360.
    assert abs(coreg.map_rotation([(0, 350.0), (100, 10.0)], 50) % 360.0) < 1e-9


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
