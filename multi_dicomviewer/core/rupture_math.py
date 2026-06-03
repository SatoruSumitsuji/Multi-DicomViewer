"""Rupture-Predictor clinical math — pure geometry, no Qt.

Goal
----
Reproduce, 1:1, the coronary adventitia *stretch-ratio* calculation that
the bundled ``resources/Rupture-Predictor.html`` performs in JavaScript,
so the tool can run fully native (PyQt6) instead of in an external
browser. Keeping this kernel Qt-free lets it be unit-tested headless and
validated numerically against the browser version.

The clinical idea (one paragraph)
---------------------------------
On an angiogram the operator marks the two stumps of the expanding
adventitia (``A1``/``A2``) and a mid-arc point (``AC``); a circle through
those three points models the vessel wall arc *before* dilation. Point
``B`` marks the most vessel-side point the balloon reaches. For a chosen
balloon diameter we build a *virtual balloon* circle tangent at ``B`` and
measure the stretched adventitia path along it. ``stretch_ratio =
stretched_length / original_arc_length`` is the predicted over-stretch —
a rupture-risk cue.

Coordinates & units
-------------------
Every point is a plain ``(x, y)`` tuple in **image pixels**. Calibration
is in **pixels per mm**; horizontal (``hpxmm``) and vertical (``vpxmm``)
are averaged for the isotropic ``avg`` used throughout, exactly as the
HTML's ``getCalib().avg``. The math here mirrors HTML lines 2672-2922 and
3325-3351; see those for the canonical source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# A point in image-pixel space.
Point = tuple[float, float]

# Balloon-diameter dropdown range in the HTML (mm): 0.75 .. 6.00 step 0.25.
DIAMETER_START_MM = 0.75
DIAMETER_STOP_MM = 6.00
DIAMETER_STEP_MM = 0.25

# Preset stretch ratios whose required balloon diameter the panel reports.
PRESET_STRETCH_RATES = (1.5, 1.8, 2.0)


@dataclass(frozen=True)
class Circle:
    """A fitted circle in image-pixel space."""

    cx: float
    cy: float
    r: float

    @property
    def center(self) -> Point:
        return (self.cx, self.cy)


@dataclass
class Calibration:
    """Pixels-per-mm scale. ``avg`` is the isotropic value used for all
    length conversions (matches the HTML ``getCalib().avg``)."""

    hpxmm: float
    vpxmm: float

    @property
    def avg(self) -> float:
        return (self.hpxmm + self.vpxmm) / 2.0


@dataclass(frozen=True)
class BalloonResult:
    """Outcome of :func:`calculate_for_balloon_diameter` — the clinical
    numbers plus every construction point needed to draw the overlay."""

    balloon_diameter_mm: float
    original_arc_len_mm: float
    stretched_adventitia_len_mm: float
    stretch_ratio: float
    angle_a1ca2_deg: float
    angle_a1ca2_rad: float
    original_circle: Circle
    virtual_center: Point          # C — virtual balloon centre
    virtual_radius_px: float
    balloon_radius_px: float
    foot_point: Point              # foot of perpendicular from B onto A1-A2
    b1: Point                      # balloon/parallel intersection, A1 side
    b2: Point                      # balloon/parallel intersection, A2 side
    include_a1b1a2b2: bool


def distance(p1: Point, p2: Point) -> float:
    """Euclidean distance (HTML ``distance``)."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def circle_from_3points(p1: Point, p2: Point, p3: Point) -> Circle | None:
    """Circle through three points, or ``None`` if they are (near-)
    collinear. Verbatim port of HTML ``circleFrom3Points`` (2677-2691)."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-10:
        return None

    ux = ((ax * ax + ay * ay) * (by - cy)
          + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx)
          + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d

    radius = math.hypot(ax - ux, ay - uy)
    return Circle(ux, uy, radius)


def arc_length(p1: Point, p2: Point, center: Point, radius: float,
               use_longer_arc: bool = False) -> float:
    """Arc length p1→p2 on the circle. ``use_longer_arc`` picks the major
    arc (2π−θ). Verbatim port of HTML ``arcLength`` (2695-2710)."""
    angle1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
    angle2 = math.atan2(p2[1] - center[1], p2[0] - center[0])
    angle_diff = angle2 - angle1

    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi

    arc_angle = abs(angle_diff)
    final_angle = (2 * math.pi - arc_angle) if use_longer_arc else arc_angle
    return final_angle * radius


def _normalize_pi(angle: float) -> float:
    """Normalize to (-π, π], matching the HTML ``while`` loops."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def calculate_for_balloon_diameter(
    a1: Point, a2: Point, ac: Point, b: Point,
    balloon_diameter_mm: float, avg_px_per_mm: float,
) -> BalloonResult | None:
    """Predict the stretched adventitia for one balloon diameter.

    Verbatim port of HTML ``calculateForBalloonDiameter`` (2713-2922).
    Returns ``None`` when A1/AC/A2 are collinear (circle fit fails) —
    the HTML alerts and returns ``undefined`` there.
    """
    original_circle = circle_from_3points(a1, ac, a2)
    if original_circle is None:
        return None
    cx, cy = original_circle.center
    radius = original_circle.r

    # Angles of A1/A2/AC about the original circle centre.
    angle1 = math.atan2(a1[1] - cy, a1[0] - cx)
    angle2 = math.atan2(a2[1] - cy, a2[0] - cx)
    angle_ac = math.atan2(ac[1] - cy, ac[0] - cx)

    angle_diff_a1_a2 = _normalize_pi(angle2 - angle1)
    angle_diff_a1_ac = _normalize_pi(angle_ac - angle1)

    # Is AC on the shorter A1→A2 arc? Picks which arc the reported angle uses.
    ac_on_shorter_arc = (
        (angle_diff_a1_a2 > 0 and angle_diff_a1_ac > 0
         and angle_diff_a1_ac < angle_diff_a1_a2)
        or (angle_diff_a1_a2 < 0 and angle_diff_a1_ac < 0
            and angle_diff_a1_ac > angle_diff_a1_a2)
    )
    angle_a1ca2 = (abs(angle_diff_a1_a2) if ac_on_shorter_arc
                   else (2 * math.pi - abs(angle_diff_a1_a2)))
    angle_a1ca2_deg = angle_a1ca2 * 180 / math.pi

    # Original adventitia arc length = A1→AC arc + AC→A2 arc (via AC).
    arc_a1_ac = abs(angle_diff_a1_ac) * radius
    arc_ac_a2 = abs(_normalize_pi(angle2 - angle_ac)) * radius
    original_arc_len_px = arc_a1_ac + arc_ac_a2
    original_arc_len_mm = original_arc_len_px / avg_px_per_mm

    # Balloon circle radius (px) for the chosen diameter.
    balloon_radius_px = (balloon_diameter_mm / 2) * avg_px_per_mm

    # Direction of line A1-A2 (unit).
    dx_a1a2 = a2[0] - a1[0]
    dy_a1a2 = a2[1] - a1[1]
    len_a1a2 = math.hypot(dx_a1a2, dy_a1a2)
    unit_x = dx_a1a2 / len_a1a2
    unit_y = dy_a1a2 / len_a1a2

    # Foot of perpendicular from B onto line A1-A2.
    dx_a1b = b[0] - a1[0]
    dy_a1b = b[1] - a1[1]
    projection = dx_a1b * unit_x + dy_a1b * unit_y
    foot = (a1[0] + projection * unit_x, a1[1] + projection * unit_y)

    # Two candidate virtual-balloon centres perpendicular to A1-A2 at B;
    # choose the one closer to AC.
    perp1 = (-unit_y, unit_x)
    perp2 = (unit_y, -unit_x)
    candidate1 = (b[0] + balloon_radius_px * perp1[0],
                  b[1] + balloon_radius_px * perp1[1])
    candidate2 = (b[0] + balloon_radius_px * perp2[0],
                  b[1] + balloon_radius_px * perp2[1])
    virtual_center = (candidate1 if distance(candidate1, ac)
                      < distance(candidate2, ac) else candidate2)
    virtual_radius = balloon_radius_px

    # B1/B2: intersections of the parallel-through-C line with the balloon
    # circle (C ± r·unit). Swap so B1 is the A1-side point.
    b1 = (virtual_center[0] - virtual_radius * unit_x,
          virtual_center[1] - virtual_radius * unit_y)
    b2 = (virtual_center[0] + virtual_radius * unit_x,
          virtual_center[1] + virtual_radius * unit_y)
    if distance(b1, a1) > distance(b2, a1):
        b1, b2 = b2, b1

    # Stretched path: include the straight A1-B1 / A2-B2 runs only when B
    # is farther from C than from its foot on A1-A2.
    dist_b_center = distance(b, virtual_center)
    dist_b_foot = distance(b, foot)

    angle_b1 = math.atan2(b1[1] - virtual_center[1], b1[0] - virtual_center[0])
    angle_b2 = math.atan2(b2[1] - virtual_center[1], b2[0] - virtual_center[0])
    arc_b1_b2 = abs(_normalize_pi(angle_b2 - angle_b1)) * virtual_radius

    len_a1_b1 = distance(a1, b1)
    len_a2_b2 = distance(a2, b2)

    include = dist_b_center > dist_b_foot
    if include:
        stretched_px = len_a1_b1 + arc_b1_b2 + len_a2_b2
    else:
        stretched_px = arc_b1_b2
    stretched_mm = stretched_px / avg_px_per_mm

    stretch_ratio = stretched_mm / original_arc_len_mm

    return BalloonResult(
        balloon_diameter_mm=balloon_diameter_mm,
        original_arc_len_mm=original_arc_len_mm,
        stretched_adventitia_len_mm=stretched_mm,
        stretch_ratio=stretch_ratio,
        angle_a1ca2_deg=angle_a1ca2_deg,
        angle_a1ca2_rad=angle_a1ca2,
        original_circle=original_circle,
        virtual_center=virtual_center,
        virtual_radius_px=virtual_radius,
        balloon_radius_px=balloon_radius_px,
        foot_point=foot,
        b1=b1,
        b2=b2,
        include_a1b1a2b2=include,
    )


def find_diameter_for_stretch_ratio(
    a1: Point, a2: Point, ac: Point, b: Point,
    target_ratio: float, avg_px_per_mm: float,
) -> float | None:
    """Balloon diameter (mm) yielding ``target_ratio``, by bisection, or
    ``None`` if out of range. Verbatim port of HTML
    ``findDiameterForStretchRatio`` (3325-3351), including its quirky
    out-of-range guard and increasing/decreasing handling."""
    lo, hi = 0.5, 20.0
    result_lo = calculate_for_balloon_diameter(a1, a2, ac, b, lo, avg_px_per_mm)
    result_hi = calculate_for_balloon_diameter(a1, a2, ac, b, hi, avg_px_per_mm)
    if result_lo is None or result_hi is None:
        return None
    if (target_ratio < result_lo.stretch_ratio
            or target_ratio > result_hi.stretch_ratio):
        if (target_ratio > result_lo.stretch_ratio
                or target_ratio < result_hi.stretch_ratio):
            return None

    increasing = result_hi.stretch_ratio > result_lo.stretch_ratio
    for _ in range(50):
        mid = (lo + hi) / 2
        result = calculate_for_balloon_diameter(
            a1, a2, ac, b, mid, avg_px_per_mm)
        if result is None:
            return None
        diff = result.stretch_ratio - target_ratio
        if abs(diff) < 0.001:
            return mid
        if increasing:
            if diff > 0:
                hi = mid
            else:
                lo = mid
        else:
            if diff > 0:
                lo = mid
            else:
                hi = mid
    return (lo + hi) / 2


def results_table(
    a1: Point, a2: Point, ac: Point, b: Point, avg_px_per_mm: float,
) -> list[BalloonResult]:
    """Per-diameter results for the 0.75..6.00 step-0.25 table. Integer
    counting avoids float drift in the step."""
    out: list[BalloonResult] = []
    n = round((DIAMETER_STOP_MM - DIAMETER_START_MM) / DIAMETER_STEP_MM) + 1
    for i in range(n):
        d = DIAMETER_START_MM + i * DIAMETER_STEP_MM
        res = calculate_for_balloon_diameter(a1, a2, ac, b, d, avg_px_per_mm)
        if res is not None:
            out.append(res)
    return out


def expansion_rate_table(
    a1: Point, a2: Point, ac: Point, b: Point, avg_px_per_mm: float,
    rates: tuple[float, ...] = PRESET_STRETCH_RATES,
) -> list[tuple[float, float | None]]:
    """For each preset stretch ratio, the required balloon diameter (mm)
    or ``None`` (out of range). Mirrors HTML ``updateExpansionRateResults``."""
    return [
        (rate, find_diameter_for_stretch_ratio(
            a1, a2, ac, b, rate, avg_px_per_mm))
        for rate in rates
    ]


def calibration_manual(
    ch1: Point, ch2: Point, ch_dist_mm: float,
    cv1: Point, cv2: Point, cv_dist_mm: float,
) -> Calibration:
    """Pixels-per-mm from the manually placed CH/CV point pairs and their
    real-world distances (HTML ``getCalib`` manual branch)."""
    h = distance(ch1, ch2) / ch_dist_mm
    v = distance(cv1, cv2) / cv_dist_mm
    return Calibration(hpxmm=h, vpxmm=v)


def calibration_dicom(hpxmm: float, vpxmm: float) -> Calibration:
    """Pixels-per-mm taken straight from DICOM pixel spacing (the MDV
    hand-off path that skips manual CH/CV calibration)."""
    return Calibration(hpxmm=hpxmm, vpxmm=vpxmm)
