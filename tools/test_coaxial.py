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
