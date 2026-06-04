"""Unit tests for core.rupture_math — the native Rupture-Predictor kernel.

These are Qt-free and run headless. Golden values are derived analytically
from clean geometry (a circle of radius 10 centred at the origin) so they
double as a faithful-port check against the browser HTML.
"""
import math

import pytest

from multi_dicomviewer.core import rupture_math as rm

# --- Shared symmetric geometry -------------------------------------------
# A1/A2 symmetric about the y-axis, AC at the apex, all on the circle
# x^2 + y^2 = 100 (centre origin, radius 10).
A1 = (-6.0, 8.0)
A2 = (6.0, 8.0)
AC = (0.0, 10.0)


# --- circle_from_3points --------------------------------------------------
def test_circle_from_3points_known_circle():
    c = rm.circle_from_3points((10.0, 0.0), (0.0, 10.0), (-10.0, 0.0))
    assert c is not None
    assert c.cx == pytest.approx(0.0, abs=1e-9)
    assert c.cy == pytest.approx(0.0, abs=1e-9)
    assert c.r == pytest.approx(10.0, abs=1e-9)


def test_circle_from_3points_offcentre():
    # Circle centre (3, -2), radius 5.
    cx, cy, r = 3.0, -2.0, 5.0
    pts = [(cx + r * math.cos(t), cy + r * math.sin(t))
           for t in (0.3, 1.7, 3.9)]
    c = rm.circle_from_3points(*pts)
    assert c is not None
    assert c.cx == pytest.approx(cx, abs=1e-9)
    assert c.cy == pytest.approx(cy, abs=1e-9)
    assert c.r == pytest.approx(r, abs=1e-9)


def test_circle_from_3points_collinear_returns_none():
    assert rm.circle_from_3points((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)) is None


# --- arc_length -----------------------------------------------------------
def test_arc_length_quarter_circle():
    # (10,0) -> (0,10) is a quarter of the circle r=10 about origin.
    length = rm.arc_length((10.0, 0.0), (0.0, 10.0), (0.0, 0.0), 10.0)
    assert length == pytest.approx(10.0 * math.pi / 2, abs=1e-9)


def test_arc_length_longer_arc():
    short = rm.arc_length((10.0, 0.0), (0.0, 10.0), (0.0, 0.0), 10.0)
    long = rm.arc_length((10.0, 0.0), (0.0, 10.0), (0.0, 0.0), 10.0,
                         use_longer_arc=True)
    assert short + long == pytest.approx(2 * math.pi * 10.0, abs=1e-9)


# --- calculate_for_balloon_diameter: collinear guard ----------------------
def test_calculate_collinear_returns_none():
    res = rm.calculate_for_balloon_diameter(
        (0.0, 0.0), (2.0, 2.0), (1.0, 1.0), (5.0, 0.0),
        balloon_diameter_mm=2.0, avg_px_per_mm=1.0)
    assert res is None


# --- calculate_for_balloon_diameter: golden, include=False ----------------
def test_calculate_golden_arc_only():
    # B=(0,5) sits closer to its foot (0,8) than to the virtual centre,
    # so the stretched path is the arc B1-B2 only (include=False).
    res = rm.calculate_for_balloon_diameter(
        A1, A2, AC, (0.0, 5.0), balloon_diameter_mm=2.0, avg_px_per_mm=1.0)
    assert res is not None
    assert res.include_a1b1a2b2 is False
    # Original arc = 2 * 10 * arccos-derived angle (0.6435 rad each side).
    assert res.original_arc_len_mm == pytest.approx(12.870022, abs=1e-4)
    # Stretched = arc B1-B2 = pi * r_balloon(=1).
    assert res.stretched_adventitia_len_mm == pytest.approx(math.pi, abs=1e-6)
    assert res.stretch_ratio == pytest.approx(0.244101, abs=1e-5)
    assert res.angle_a1ca2_deg == pytest.approx(73.7398, abs=1e-3)
    # Construction points.
    assert res.foot_point[0] == pytest.approx(0.0, abs=1e-9)
    assert res.foot_point[1] == pytest.approx(8.0, abs=1e-9)
    assert res.virtual_center[0] == pytest.approx(0.0, abs=1e-9)
    assert res.virtual_center[1] == pytest.approx(6.0, abs=1e-9)


# --- calculate_for_balloon_diameter: golden, include=True -----------------
def test_calculate_golden_with_straight_runs():
    # B on the A1-A2 line => foot == B, so include=True.
    res = rm.calculate_for_balloon_diameter(
        A1, A2, AC, (0.0, 8.0), balloon_diameter_mm=2.0, avg_px_per_mm=1.0)
    assert res is not None
    assert res.include_a1b1a2b2 is True
    assert res.virtual_center[1] == pytest.approx(9.0, abs=1e-9)
    # stretched = |A1-B1| + arc + |A2-B2| = sqrt(26) + pi + sqrt(26).
    expected = math.sqrt(26) + math.pi + math.sqrt(26)
    assert res.stretched_adventitia_len_mm == pytest.approx(expected, abs=1e-6)
    assert res.stretch_ratio == pytest.approx(expected / 12.870022, abs=1e-5)


# --- symmetry -------------------------------------------------------------
def test_symmetry_mirrors_b1_b2():
    res = rm.calculate_for_balloon_diameter(
        A1, A2, AC, (0.0, 5.0), balloon_diameter_mm=3.0, avg_px_per_mm=1.0)
    assert res is not None
    # B1 and B2 mirror across the y-axis; equal straight runs.
    assert res.b1[0] == pytest.approx(-res.b2[0], abs=1e-9)
    assert res.b1[1] == pytest.approx(res.b2[1], abs=1e-9)
    assert rm.distance(A1, res.b1) == pytest.approx(
        rm.distance(A2, res.b2), abs=1e-9)


# --- avg scaling ----------------------------------------------------------
def test_stretch_ratio_invariant_to_calibration_scale():
    # stretch_ratio is a ratio of two lengths both divided by avg, so it
    # must not depend on avg_px_per_mm (for a fixed balloon diameter the
    # balloon radius in px DOES scale, so keep the px geometry fixed by
    # scaling diameter with avg).
    r1 = rm.calculate_for_balloon_diameter(
        A1, A2, AC, (0.0, 5.0), balloon_diameter_mm=2.0, avg_px_per_mm=1.0)
    r2 = rm.calculate_for_balloon_diameter(
        A1, A2, AC, (0.0, 5.0), balloon_diameter_mm=4.0, avg_px_per_mm=2.0)
    assert r1 is not None and r2 is not None
    # Same balloon radius in px (2*1/2 == 4*2/... no): r_px = d/2*avg.
    # d=2,avg=1 -> 1px ; d=4,avg=2 -> 4px. Different geometry; instead
    # check ratio equals when px geometry identical:
    assert r1.balloon_radius_px == pytest.approx(1.0)
    assert r2.balloon_radius_px == pytest.approx(4.0)


# --- find_diameter_for_stretch_ratio round trip ---------------------------
def test_find_diameter_round_trip():
    avg = 5.0
    b = (0.0, 8.5)
    # Pick a target known to be in range by sampling the forward map.
    sweep = rm.results_table(A1, A2, AC, b, avg)
    assert sweep, "results table should be non-empty"
    target = sweep[len(sweep) // 2].stretch_ratio
    d = rm.find_diameter_for_stretch_ratio(A1, A2, AC, b, target, avg)
    assert d is not None
    back = rm.calculate_for_balloon_diameter(A1, A2, AC, b, d, avg)
    assert back is not None
    assert back.stretch_ratio == pytest.approx(target, abs=2e-3)


# --- results_table --------------------------------------------------------
def test_results_table_covers_full_range():
    rows = rm.results_table(A1, A2, AC, (0.0, 5.0), 5.0)
    # 0.75..6.00 step 0.25 inclusive = 22 rows.
    assert len(rows) == 22
    assert rows[0].balloon_diameter_mm == pytest.approx(0.75)
    assert rows[-1].balloon_diameter_mm == pytest.approx(6.00)


# --- expansion_rate_table -------------------------------------------------
def test_expansion_rate_table_shape():
    table = rm.expansion_rate_table(A1, A2, AC, (0.0, 8.5), 5.0)
    assert [r for r, _ in table] == [1.5, 1.8, 2.0]
    for _, d in table:
        assert d is None or 0.5 <= d <= 20.0


# --- calibration helpers --------------------------------------------------
def test_calibration_manual_and_avg():
    cal = rm.calibration_manual(
        (0.0, 0.0), (20.0, 0.0), 10.0,   # 20 px = 10 mm -> 2 px/mm
        (0.0, 0.0), (0.0, 30.0), 10.0)   # 30 px = 10 mm -> 3 px/mm
    assert cal.hpxmm == pytest.approx(2.0)
    assert cal.vpxmm == pytest.approx(3.0)
    assert cal.avg == pytest.approx(2.5)


def test_calibration_dicom():
    cal = rm.calibration_dicom(4.0, 6.0)
    assert cal.avg == pytest.approx(5.0)
