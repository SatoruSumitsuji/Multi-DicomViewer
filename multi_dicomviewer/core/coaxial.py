"""Coaxiality evaluation — pure geometry, no Qt.

Goal
----
Given a straight line drawn on the SAME vessel in two or more angiographic
views (each shot at a different C-arm angle), recover that vessel's 3-D
direction in patient space, then report the angle between the Guiding
Catheter (GC) and each coronary-proximal segment.

How it works (one paragraph)
----------------------------
An angiogram is a 2-D shadow of a 3-D scene. A straight line drawn on one
view does not by itself fix the vessel's 3-D direction — but it does fix a
*plane*: the plane swept by back-projecting that 2-D line through the X-ray
source. The true 3-D vessel direction must lie inside that plane. Draw the
line on a second view shot from a different angle and you get a second
plane; the vessel direction is the line where the two planes intersect
(``d = n1 x n2``). With three or more views we solve it in the
least-squares sense (smallest singular vector). Once each vessel has a 3-D
direction, the GC-to-vessel angle is just the angle between two vectors.

Coordinate system (matches ui/orthogonal_view.py)
-------------------------------------------------
Patient LPS: x = Left, y = Posterior, z = Head.
PositionerPrimaryAngle   beta  (LAO + / RAO -), rotation about z.
PositionerSecondaryAngle alpha (CRA + / CAU -), rotation about x.

    u_right(b, a) = ( cos b,          sin b,          0    )   image +x / mm
    v_down (b, a) = ( sin b * sin a, -cos b * sin a, -cos a)   image +y / mm
    beam   (b, a) = (-sin b * cos a,  cos b * cos a, -sin a)   source -> detector

These satisfy ``u_right x v_down = beam`` (a right-handed image triad).
"""
from __future__ import annotations

import math

import numpy as np

# Below this pairwise C-arm separation the two projection planes are nearly
# parallel and the reconstructed direction is numerically unstable, so the
# caller should warn / refuse the result.
MIN_VIEW_SEPARATION_DEG = 30.0

# Coronary-proximal vessel labels the UI will offer on the Line tool.
GC = "GC"
VESSEL_LABELS = (GC, "LM", "proxLAD", "proxLCX", "proxRCA")


def image_axes_lps(beta_deg: float, alpha_deg: float):
    """(u_right, v_down) unit vectors in patient LPS for a C-arm at the
    given primary / secondary angles."""
    b = math.radians(beta_deg)
    a = math.radians(alpha_deg)
    cb, sb = math.cos(b), math.sin(b)
    ca, sa = math.cos(a), math.sin(a)
    u_right = np.array([cb, sb, 0.0])
    v_down = np.array([sb * sa, -cb * sa, -ca])
    return u_right, v_down


def beam_direction(beta_deg: float, alpha_deg: float) -> np.ndarray:
    """X-ray beam direction (source -> detector) in patient LPS."""
    b = math.radians(beta_deg)
    a = math.radians(alpha_deg)
    return np.array([
        -math.sin(b) * math.cos(a),
        math.cos(b) * math.cos(a),
        -math.sin(a),
    ])


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def line_direction_3d(line_2d, beta_deg, alpha_deg, spacing=(1.0, 1.0)):
    """The in-plane 3-D direction a 2-D image line points along.

    Parameters
    ----------
    line_2d  : ((x0, y0), (x1, y1)) in image pixel coordinates (x right,
               y down). Only the direction matters, not which endpoint.
    spacing  : (row_mm, col_mm) = (y, x) pixel spacing in millimetres.

    Returns a unit vector in patient LPS lying in the image plane.
    """
    (x0, y0), (x1, y1) = line_2d
    row_mm, col_mm = float(spacing[0]), float(spacing[1])
    du = (x1 - x0) * col_mm
    dv = (y1 - y0) * row_mm
    if math.hypot(du, dv) < 1e-9:
        raise ValueError("line endpoints coincide")
    u_right, v_down = image_axes_lps(beta_deg, alpha_deg)
    return _unit(du * u_right + dv * v_down)


def plane_normal(line_2d, beta_deg, alpha_deg, spacing=(1.0, 1.0)):
    """Unit normal of the back-projection plane of a 2-D line.

    The plane contains the line's in-plane direction and the beam, so its
    normal is their cross product. The true vessel direction is perpendicular
    to this normal.
    """
    v3d = line_direction_3d(line_2d, beta_deg, alpha_deg, spacing)
    beam = beam_direction(beta_deg, alpha_deg)
    return _unit(np.cross(v3d, beam))


def intersect_planes(normals) -> np.ndarray:
    """3-D direction lying in every plane, given the planes' unit normals.

    Two planes  -> cross product of their normals.
    Three+      -> least-squares: the unit vector most orthogonal to all
                   normals (smallest right-singular vector of the stacked
                   normals). This averages out picking error across views.
    """
    arr = np.asarray([_unit(n) for n in normals], dtype=np.float64)
    if arr.shape[0] < 2:
        raise ValueError("need at least two views")
    if arr.shape[0] == 2:
        return _unit(np.cross(arr[0], arr[1]))
    # Smallest singular vector of the normals = direction closest to lying
    # in all the planes at once.
    _, _, vt = np.linalg.svd(arr)
    return _unit(vt[-1])


def view_separation_deg(observations) -> float:
    """Smallest pairwise C-arm beam-direction angle among the views, in
    degrees. Small values mean the views are nearly the same projection and
    the reconstruction is unreliable. ``observations`` is a list of dicts
    each with ``beta`` and ``alpha``."""
    beams = [_unit(beam_direction(o["beta"], o["alpha"])) for o in observations]
    worst = 180.0
    for i in range(len(beams)):
        for j in range(i + 1, len(beams)):
            d = abs(float(np.dot(beams[i], beams[j])))
            d = max(-1.0, min(1.0, d))
            worst = min(worst, math.degrees(math.acos(d)))
    return worst


def vessel_direction(observations) -> np.ndarray:
    """Reconstruct a vessel's 3-D unit direction from 2+ labelled views.

    ``observations`` : list of dicts, each with keys
        ``beta``, ``alpha``   : C-arm angles (degrees)
        ``line_2d``           : ((x0,y0),(x1,y1)) pixel line on that view
        ``spacing``           : optional (row_mm, col_mm), default (1,1)

    Returns a unit vector in patient LPS. Sign is arbitrary (a drawn line
    has no head/tail), so callers comparing directions must use |dot|.
    """
    if len(observations) < 2:
        raise ValueError("need at least two views to reconstruct 3-D")
    normals = [
        plane_normal(o["line_2d"], o["beta"], o["alpha"],
                     o.get("spacing", (1.0, 1.0)))
        for o in observations
    ]
    return intersect_planes(normals)


def angle_between_directions(d1, d2) -> float:
    """Acute angle between two 3-D directions, in degrees [0, 90].

    0 deg means the two run along the same line (coaxial); 90 deg means
    perpendicular. Uses |dot| because drawn lines are undirected.
    """
    a = _unit(d1)
    b = _unit(d2)
    c = abs(float(np.dot(a, b)))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def compute_coaxial_angles(labeled_lines):
    """High-level entry: GC-to-each-vessel angles from labelled lines.

    Parameters
    ----------
    labeled_lines : list of dicts, each describing ONE line drawn on ONE
        view, with keys:
            ``label``   : one of VESSEL_LABELS
            ``beta``    : PositionerPrimaryAngle  (deg)
            ``alpha``   : PositionerSecondaryAngle (deg)
            ``line_2d`` : ((x0,y0),(x1,y1)) pixel coordinates
            ``spacing`` : optional (row_mm, col_mm)

    Returns
    -------
    dict with:
        ``directions`` : {label: 3-D unit vector} for every vessel that had
                         >= 2 views.
        ``angles``     : {label: GC-to-label angle in deg} for non-GC
                         vessels (only when GC itself was reconstructed).
        ``details``    : {label: dict} per-vessel calculation trace used to
                         explain the result in the UI. Keys:
                            ``views``         : [(beta, alpha), ...] used
                            ``separation_deg``: smallest pairwise C-arm angle
                            ``direction``     : reconstructed 3-D unit vector
                            ``cos_to_gc``     : |GC . v| (non-GC vessels only)
        ``warnings``   : list of human-readable strings (too few views,
                         views too close together, GC missing, ...).
    """
    by_label: dict[str, list] = {}
    for ln in labeled_lines:
        by_label.setdefault(ln["label"], []).append(ln)

    directions: dict[str, np.ndarray] = {}
    details: dict[str, dict] = {}
    warnings: list[str] = []

    for label, obs in by_label.items():
        if len(obs) < 2:
            warnings.append(
                f"{label}: only {len(obs)} view with a line — need 2+ "
                f"views to reconstruct a 3-D direction."
            )
            continue
        sep = view_separation_deg(obs)
        if sep < MIN_VIEW_SEPARATION_DEG:
            warnings.append(
                f"{label}: views are only {sep:.0f} deg apart "
                f"(< {MIN_VIEW_SEPARATION_DEG:.0f} deg) — reconstruction is "
                f"unreliable; use more separated C-arm angles."
            )
            continue
        d = vessel_direction(obs)
        directions[label] = d
        details[label] = {
            "views": [(float(o["beta"]), float(o["alpha"])) for o in obs],
            "separation_deg": sep,
            "direction": d,
        }

    angles: dict[str, float] = {}
    if GC in directions:
        gc = _unit(directions[GC])
        for label, d in directions.items():
            if label == GC:
                continue
            c = abs(float(np.dot(gc, _unit(d))))
            c = max(-1.0, min(1.0, c))
            angles[label] = math.degrees(math.acos(c))
            details[label]["cos_to_gc"] = c
    elif directions:
        warnings.append(
            "No GC line reconstructed — label a Guiding-Catheter line on "
            "2+ views to get GC-to-vessel angles."
        )

    return {
        "directions": directions,
        "angles": angles,
        "details": details,
        "warnings": warnings,
    }
