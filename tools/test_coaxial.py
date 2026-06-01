"""Known-answer 'practice problems' for core/coaxial.py.

Run directly:    python tools/test_coaxial.py
Or with pytest:  pytest tools/test_coaxial.py

Each test builds synthetic angio lines for a vessel whose true 3-D
direction we already know, then checks the reconstruction matches.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core import coaxial as cx


def _angle(a, b):
    """Acute angle (deg) between two vectors, sign-insensitive."""
    return cx.angle_between_directions(a, b)


def _vertical_line():
    """A top-to-bottom line on the image (x fixed, y grows)."""
    return ((100.0, 50.0), (100.0, 150.0))


def _horizontal_line():
    """A left-to-right line on the image (y fixed, x grows)."""
    return ((50.0, 100.0), (150.0, 100.0))


# --------------------------------------------------------------------------
# 1. Vessel along patient Head-Foot axis (+z), seen frontal and lateral.
#    In both projections a head-foot vessel looks like a vertical line.
#    Expected reconstructed direction: (0, 0, 1) up to sign.
# --------------------------------------------------------------------------
def test_head_foot_vessel():
    obs = [
        {"beta": 0.0, "alpha": 0.0, "line_2d": _vertical_line()},   # frontal
        {"beta": 90.0, "alpha": 0.0, "line_2d": _vertical_line()},  # lateral
    ]
    d = cx.vessel_direction(obs)
    err = _angle(d, [0.0, 0.0, 1.0])
    assert err < 1e-6, f"head-foot vessel off by {err:.4f} deg"


# --------------------------------------------------------------------------
# 2. Vessel along patient Left-Right axis (+x). Frontal view sees it
#    horizontal; a CRA/CAU-tilted second view still sees it (LAO90 would
#    project it to a point, so we use LAO0/CRA40 as the second view).
#    Expected direction: (1, 0, 0) up to sign.
# --------------------------------------------------------------------------
def test_left_right_vessel():
    obs = [
        {"beta": 0.0, "alpha": 0.0, "line_2d": _horizontal_line()},
        {"beta": 0.0, "alpha": 40.0, "line_2d": _horizontal_line()},
    ]
    d = cx.vessel_direction(obs)
    err = _angle(d, [1.0, 0.0, 0.0])
    assert err < 1e-6, f"left-right vessel off by {err:.4f} deg"


# --------------------------------------------------------------------------
# 3. Coaxial GC and vessel: both lines drawn the SAME way on the SAME views
#    must give a 0 deg angle.
# --------------------------------------------------------------------------
def test_coaxial_is_zero():
    lines = []
    for beta, alpha in [(0.0, 0.0), (90.0, 0.0)]:
        lines.append({"label": "GC", "beta": beta, "alpha": alpha,
                      "line_2d": _vertical_line()})
        lines.append({"label": "proxLAD", "beta": beta, "alpha": alpha,
                      "line_2d": _vertical_line()})
    res = cx.compute_coaxial_angles(lines)
    assert not res["warnings"], res["warnings"]
    ang = res["angles"]["proxLAD"]
    assert ang < 1e-6, f"identical lines should be 0 deg, got {ang:.4f}"


# --------------------------------------------------------------------------
# 4. A known 90 deg case: GC along +z (vertical in both views), vessel along
#    +x (horizontal frontal, still visible in a CRA-tilted view).
# --------------------------------------------------------------------------
def test_perpendicular_is_ninety():
    lines = [
        {"label": "GC", "beta": 0.0, "alpha": 0.0, "line_2d": _vertical_line()},
        {"label": "GC", "beta": 90.0, "alpha": 0.0, "line_2d": _vertical_line()},
        {"label": "proxRCA", "beta": 0.0, "alpha": 0.0,
         "line_2d": _horizontal_line()},
        {"label": "proxRCA", "beta": 0.0, "alpha": 40.0,
         "line_2d": _horizontal_line()},
    ]
    res = cx.compute_coaxial_angles(lines)
    assert not res["warnings"], res["warnings"]
    ang = res["angles"]["proxRCA"]
    assert abs(ang - 90.0) < 1e-4, f"expected 90 deg, got {ang:.4f}"


# --------------------------------------------------------------------------
# 5. Pixel spacing (anisotropic) must be honoured: a line that is diagonal
#    in pixels but axis-aligned in millimetres should behave like the mm
#    line. Here 2x taller pixels turn a 1:2 pixel slope into a 1:1 mm slope.
# --------------------------------------------------------------------------
def test_spacing_is_honoured():
    iso = cx.line_direction_3d(((0, 0), (100, 100)), 0.0, 0.0, (1.0, 1.0))
    # Same physical line but pixels are 0.5 mm wide, 1.0 mm tall: to keep the
    # mm direction 1:1 the pixel deltas must be 200 (x) by 100 (y).
    aniso = cx.line_direction_3d(((0, 0), (200, 100)), 0.0, 0.0, (1.0, 0.5))
    assert _angle(iso, aniso) < 1e-6, "spacing not applied to line direction"


# --------------------------------------------------------------------------
# 6. Views that are too close together must be rejected with a warning, not
#    silently produce a garbage direction.
# --------------------------------------------------------------------------
def test_close_views_warn():
    lines = [
        {"label": "GC", "beta": 0.0, "alpha": 0.0, "line_2d": _vertical_line()},
        {"label": "GC", "beta": 5.0, "alpha": 0.0, "line_2d": _vertical_line()},
    ]
    res = cx.compute_coaxial_angles(lines)
    assert "GC" not in res["directions"], "5 deg apart should be rejected"
    assert any("apart" in w for w in res["warnings"]), res["warnings"]


# --------------------------------------------------------------------------
# 7. A single view for a vessel cannot be reconstructed -> warning.
# --------------------------------------------------------------------------
def test_single_view_warn():
    lines = [
        {"label": "GC", "beta": 0.0, "alpha": 0.0, "line_2d": _vertical_line()},
        {"label": "GC", "beta": 90.0, "alpha": 0.0, "line_2d": _vertical_line()},
        {"label": "proxLAD", "beta": 0.0, "alpha": 0.0,
         "line_2d": _horizontal_line()},  # only one view
    ]
    res = cx.compute_coaxial_angles(lines)
    assert "proxLAD" not in res["angles"]
    assert any("need 2+" in w for w in res["warnings"]), res["warnings"]


# --------------------------------------------------------------------------
# 8. Three views (least-squares path) with slightly noisy picks should still
#    land within a degree of the true head-foot direction.
# --------------------------------------------------------------------------
def test_three_views_least_squares():
    obs = [
        {"beta": 0.0, "alpha": 0.0, "line_2d": ((100, 50), (101, 150))},
        {"beta": 60.0, "alpha": 0.0, "line_2d": ((100, 50), (99, 150))},
        {"beta": 120.0, "alpha": 0.0, "line_2d": ((100, 50), (100, 150))},
    ]
    d = cx.vessel_direction(obs)
    err = _angle(d, [0.0, 0.0, 1.0])
    assert err < 1.0, f"3-view LS off by {err:.4f} deg"


# --------------------------------------------------------------------------
# 9. line_angle_2d: two perpendicular image lines = 90°, identical = 0°,
#    and anisotropic spacing is honoured.
# --------------------------------------------------------------------------
def test_line_angle_2d_basic():
    perp = cx.line_angle_2d(_vertical_line(), _horizontal_line())
    assert abs(perp - 90.0) < 1e-6, f"perp lines: {perp}"
    same = cx.line_angle_2d(_vertical_line(), _vertical_line())
    assert same < 1e-6, f"identical lines: {same}"
    # A 1:1 pixel diagonal becomes 1:2 in mm when columns are half as wide;
    # vs a vertical line that is 45° in pixels but should differ in mm.
    aniso = cx.line_angle_2d(((0, 0), (100, 100)), ((0, 0), (0, 100)),
                             (1.0, 0.5))
    pix = cx.line_angle_2d(((0, 0), (100, 100)), ((0, 0), (0, 100)),
                           (1.0, 1.0))
    assert abs(aniso - pix) > 1.0, "spacing not honoured in 2-D angle"


# --------------------------------------------------------------------------
# 10. per_view_2d / spread_2d are reported, and a perfectly coaxial case
#     shows ~0° in every view with ~0° spread.
# --------------------------------------------------------------------------
def test_per_view_2d_reported():
    lines = []
    for beta, alpha in [(0.0, 0.0), (90.0, 0.0)]:
        lines.append({"label": "GC", "beta": beta, "alpha": alpha,
                      "line_2d": _vertical_line()})
        lines.append({"label": "proxLAD", "beta": beta, "alpha": alpha,
                      "line_2d": _vertical_line()})
    res = cx.compute_coaxial_angles(lines)
    det = res["details"]["proxLAD"]
    assert len(det["per_view_2d"]) == 2, det["per_view_2d"]
    assert all(p["angle_2d"] < 1e-6 for p in det["per_view_2d"])
    assert det["spread_2d"] is not None and det["spread_2d"] < 1e-6


# --------------------------------------------------------------------------
# 11. Confidence cues: a 3-view case reports pairwise 2-view angles, a
#     pairwise spread, leave-one-out entries, and finite conditioning.
# --------------------------------------------------------------------------
def test_confidence_three_views():
    # Three well-separated views (β 0/60/120), GC and vessel both drawn as
    # near-vertical lines so every 2-view pair reconstructs cleanly.
    lines = []
    for beta in (0.0, 60.0, 120.0):
        lines.append({"label": "GC", "beta": beta, "alpha": 0.0,
                      "line_2d": _vertical_line()})
        lines.append({"label": "proxLAD", "beta": beta, "alpha": 0.0,
                      "line_2d": _vertical_line()})
    res = cx.compute_coaxial_angles(lines)
    conf = res["details"]["proxLAD"]["confidence"]
    assert len(conf["shared_views"]) == 3
    assert len(conf["pairwise"]) == 3, conf["pairwise"]
    assert conf["pairwise_spread"] is not None
    assert len(conf["leave_one_out"]) == 3
    # All views consistent (perpendicular vessel) → tiny spread.
    assert conf["pairwise_spread"] < 1.0
    assert conf["cond_gc"] is not None and conf["cond_vessel"] is not None


# --------------------------------------------------------------------------
# 12. reconstruction_condition: well-separated views give a small kappa;
#     two near-identical views give a large (ill-conditioned) one.
# --------------------------------------------------------------------------
def test_close_pair_does_not_block_when_a_good_pair_exists():
    """Case A: adding a 3rd view that sits CLOSE to one existing view must
    not turn a solvable 2-view case unsolvable. Mirrors the user's RCA
    report: LAO0/CRA30 + LAO50/CRA0 (56 deg apart, fine) plus a LAO30/CRA30
    view that is only ~26 deg from the first. The vessel must still be
    reconstructed, with a non-blocking 'two views are close' note."""
    def line(b, a, lab):
        return {"label": lab, "beta": b, "alpha": a,
                "line_2d": _vertical_line()}
    views = [(0.0, 30.0), (50.0, 0.0), (30.0, 30.0)]
    lines = []
    for (b, a) in views:
        lines.append(line(b, a, "GC"))
        lines.append(line(b, a, "proxRCA"))
    res = cx.compute_coaxial_angles(lines)
    assert "proxRCA" in res["angles"], res["warnings"]
    det = res["details"]["proxRCA"]
    # Worst pair is below threshold, best pair is well above it.
    assert det["separation_deg"] < cx.MIN_VIEW_SEPARATION_DEG
    assert det["best_separation_deg"] >= cx.MIN_VIEW_SEPARATION_DEG
    # A note (not a hard failure) mentions the close pair.
    assert any("close" in w or "add little" in w for w in res["warnings"])


def test_two_close_views_still_rejected():
    """With ONLY a close pair (no well-separated pair), it is still
    rejected — the best pair is the only pair and it is too close."""
    lines = [
        {"label": "GC", "beta": 0.0, "alpha": 0.0, "line_2d": _vertical_line()},
        {"label": "GC", "beta": 5.0, "alpha": 0.0, "line_2d": _vertical_line()},
        {"label": "proxRCA", "beta": 0.0, "alpha": 0.0,
         "line_2d": _horizontal_line()},
        {"label": "proxRCA", "beta": 5.0, "alpha": 0.0,
         "line_2d": _horizontal_line()},
    ]
    res = cx.compute_coaxial_angles(lines)
    assert "proxRCA" not in res["angles"]
    assert any("most-separated pair" in w for w in res["warnings"])


def test_angles_from_beam_roundtrip():
    """angles_from_beam is the exact inverse of beam_direction."""
    for beta, alpha in [(0, 0), (30, -20), (-45, 25), (90, 0), (0, 40)]:
        beam = cx.beam_direction(beta, alpha)
        b2, a2 = cx.angles_from_beam(beam)
        assert abs(b2 - beta) < 1e-6 and abs(a2 - alpha) < 1e-6, \
            f"{(beta, alpha)} -> {(b2, a2)}"


def test_optimal_projection_perpendicular():
    """For two perpendicular vessels along +z and +x, the optimal view is
    along +/- y (i.e. AP / PA: beta 0, alpha 0), and looking down it the
    two truly are 90 deg apart."""
    d_gc = [0.0, 0.0, 1.0]
    d_ves = [1.0, 0.0, 0.0]
    opt = cx.optimal_projection_angles(d_gc, d_ves)
    assert opt is not None and len(opt) == 2
    # Normal is +/- y; angles_from_beam(+y) = (0,0), (-y) = (180,0)/(0,...) —
    # both must reproduce the y axis as their beam.
    for (b, a) in opt:
        beam = cx.beam_direction(b, a)
        assert abs(abs(beam[1]) - 1.0) < 1e-6, f"beam not along y: {beam}"


def test_optimal_projection_coaxial_none():
    """Near-coaxial directions have no unique optimal view."""
    assert cx.optimal_projection_angles([0, 0, 1], [0.0, 1e-7, 1]) is None


def test_optimal_view_in_details():
    lines = []
    for beta in (0.0, 60.0, 120.0):
        lines.append({"label": "GC", "beta": beta, "alpha": 0.0,
                      "line_2d": _vertical_line()})
        lines.append({"label": "proxRCA", "beta": beta, "alpha": 0.0,
                      "line_2d": _horizontal_line()})
    res = cx.compute_coaxial_angles(lines)
    det = res["details"]["proxRCA"]
    assert "optimal_view" in det
    assert det["optimal_view"] is None or len(det["optimal_view"]) == 2


def test_reconstruction_condition():
    good = [
        {"beta": 0.0, "alpha": 0.0, "line_2d": _vertical_line()},
        {"beta": 90.0, "alpha": 0.0, "line_2d": _vertical_line()},
    ]
    bad = [
        {"beta": 0.0, "alpha": 0.0, "line_2d": _vertical_line()},
        {"beta": 3.0, "alpha": 0.0, "line_2d": _vertical_line()},
    ]
    kg = cx.reconstruction_condition(good)
    kb = cx.reconstruction_condition(bad)
    assert kg < kb, f"well-separated should be better-conditioned: {kg} vs {kb}"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
