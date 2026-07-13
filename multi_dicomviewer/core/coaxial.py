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
from itertools import combinations

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


def angles_from_beam(beam) -> tuple[float, float]:
    """Inverse of :func:`beam_direction`: a 3-D beam (source -> detector)
    unit vector in patient LPS back to (PositionerPrimary, Secondary) =
    (beta = LAO+/RAO-, alpha = CRA+/CAU-) in degrees.

    beam = (-sin b cos a,  cos b cos a,  -sin a), so
        a = asin(-beam_z)              (secondary, [-90, 90])
        b = atan2(-beam_x, beam_y)     (primary)
    """
    bx, by, bz = (float(beam[0]), float(beam[1]), float(beam[2]))
    a = math.asin(max(-1.0, min(1.0, -bz)))
    b = math.atan2(-bx, by)
    return math.degrees(b), math.degrees(a)


def _clean_angulation(beam):
    """(beta, alpha) for a beam, with negative-zero scrubbed so the UI never
    prints 'CRA -0' / 'RAO -0'. Does NOT flip to the opposite beam — callers
    that want both ends of a viewing line pass +n and -n separately."""
    b, a = angles_from_beam(_unit(beam))
    b = 0.0 if abs(b) < 5e-4 else b + 0.0
    a = 0.0 if abs(a) < 5e-4 else a + 0.0
    return (b, a)


def _primary_reachable(angulation) -> bool:
    """True if the primary (LAO/RAO) angle is within a typical C-arm's gantry
    travel (|primary| <= 90 deg). Used only to decide which of the two
    opposite optimal beams to list first (the bedside-reachable one)."""
    return abs(angulation[0]) <= 90.0 + 1e-6


def optimal_projection_angles(d1, d2):
    """The two C-arm angulations whose view shows the angle between 3-D
    directions ``d1`` and ``d2`` with NO foreshortening — i.e. looking
    straight down the normal of the plane the two vectors span, so both
    vessels lie in the image plane and their on-screen angle equals the true
    3-D angle.

    The viewing axis is ``d1 x d2`` — a line, so it has two opposite beams
    that show the SAME projection (the view and its counter-view from the
    other side of the patient). BOTH are returned. They are ordered so the
    gantry-reachable one (primary LAO/RAO within +/-90 deg) comes first; its
    partner is the geometric opposite (often a > 90 deg "counter-view" a
    standard C-arm cannot be driven to, shown for completeness).

    Returns ``[(beta1, alpha1), (beta2, alpha2)]``, or ``None`` when ``d1``
    and ``d2`` are nearly parallel (coaxial) — then the spanned plane, and
    so the optimal view, is undefined (any orthogonal view is equally good).
    """
    a = _unit(d1)
    b = _unit(d2)
    n = np.cross(a, b)
    if float(np.linalg.norm(n)) < 1e-4:        # ~< 0.006 deg apart: degenerate
        return None
    n = _unit(n)
    first = _clean_angulation(n)
    second = _clean_angulation(-n)
    # List the bedside-reachable angulation first.
    if not _primary_reachable(first) and _primary_reachable(second):
        first, second = second, first
    return [first, second]


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


def line_angle_2d(line_a, line_b, spacing=(1.0, 1.0)) -> float:
    """Acute angle (deg, [0,90]) between two 2-D image lines, in millimetre
    space (so anisotropic pixel spacing is honoured).

    This is the angle as it APPEARS in a single projection — it carries
    foreshortening and is NOT the true 3-D angle. It is only a per-view
    sanity / confidence cue: if every view shows a similar GC-to-vessel
    angle the 3-D reconstruction is well constrained; a wide spread flags
    a less trustworthy result. Sign-insensitive (drawn lines are
    undirected), matching angle_between_directions.
    """
    row_mm, col_mm = float(spacing[0]), float(spacing[1])

    def _vec(line):
        (x0, y0), (x1, y1) = line
        return col_mm * (x1 - x0), row_mm * (y1 - y0)

    ax, ay = _vec(line_a)
    bx, by = _vec(line_b)
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na < 1e-9 or nb < 1e-9:
        raise ValueError("line endpoints coincide")
    c = abs(ax * bx + ay * by) / (na * nb)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


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
    degrees. Small values mean SOME pair of views is nearly the same
    projection. ``observations`` is a list of dicts each with ``beta`` and
    ``alpha``."""
    beams = [_unit(beam_direction(o["beta"], o["alpha"])) for o in observations]
    worst = 180.0
    for i in range(len(beams)):
        for j in range(i + 1, len(beams)):
            d = abs(float(np.dot(beams[i], beams[j])))
            d = max(-1.0, min(1.0, d))
            worst = min(worst, math.degrees(math.acos(d)))
    return worst


def best_view_separation_deg(observations) -> float:
    """LARGEST pairwise C-arm beam-direction angle among the views, in
    degrees. This is the separation of the best-conditioned pair: as long as
    ONE pair is well separated the 3-D reconstruction is anchored, even if
    other views sit close to each other (they only add information when the
    answer is solved jointly, never remove it). Used to decide whether a
    reconstruction is worth doing; ``view_separation_deg`` (the worst pair)
    is reported separately as a confidence cue."""
    beams = [_unit(beam_direction(o["beta"], o["alpha"])) for o in observations]
    best = 0.0
    for i in range(len(beams)):
        for j in range(i + 1, len(beams)):
            d = abs(float(np.dot(beams[i], beams[j])))
            d = max(-1.0, min(1.0, d))
            best = max(best, math.degrees(math.acos(d)))
    return best


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


def _view_key(o):
    """Hashable (beta, alpha) key, rounded so float noise doesn't split
    what is really the same view."""
    return (round(float(o["beta"]), 3), round(float(o["alpha"]), 3))


def _theta_from_obs(gc_obs, vessel_obs) -> float:
    """GC-to-vessel 3-D angle reconstructed from the given observation
    subsets. Each subset must have >= 2 views."""
    return angle_between_directions(
        vessel_direction(gc_obs), vessel_direction(vessel_obs)
    )


def reconstruction_condition(observations) -> float | None:
    """Geometric well-posedness of a vessel's 3-D reconstruction.

    Returns ``sigma1 / sigma2`` of the stacked plane normals (the two
    LARGEST singular values). The smallest singular vector is the answer
    direction; its stability is governed by how large the *second* singular
    value is. So:

    * ~1   -> the views are geometrically diverse; the direction is pinned
              from two well-separated angles. Well-conditioned.
    * large -> the planes are nearly co-axial (views cluster, or their
              normals nearly align); the direction is weakly constrained
              and small picking errors blow up. Ill-conditioned.

    ``inf`` when sigma2 collapses (effectively a single view direction);
    ``None`` for < 2 views. This is independent of the angle value itself,
    so it flags the "two views with the same tilt" case the C-arm
    separation check can miss.
    """
    if len(observations) < 2:
        return None
    normals = np.array([
        plane_normal(o["line_2d"], o["beta"], o["alpha"],
                     o.get("spacing", (1.0, 1.0)))
        for o in observations
    ])
    sv = np.linalg.svd(normals, compute_uv=False)
    s1, s2 = float(sv[0]), float(sv[1])
    if s2 < 1e-12:
        return float("inf")
    return s1 / s2


def coaxial_confidence(gc_obs, vessel_obs) -> dict:
    """Confidence cues for one GC-to-vessel angle. None of these change the
    reported 3-D angle — they only let the reader judge how trustworthy it
    is, especially the "does adding a 3rd view move the answer?" question.

    Built from the views that carry BOTH a GC and a vessel line (shared
    views), so every cue is an apples-to-apples GC-vs-vessel comparison.

    Returns a dict with:
        ``shared_views``    : [(beta, alpha), ...] used by the cues below.
        ``pairwise``        : list of {views:[(b,a),(b,a)], theta} — the
            angle reconstructed from EACH PAIR of shared views on its own
            (the minimal 2-view answer). Comparing these to the headline
            (all-views) angle shows directly how much a 2-view estimate
            would have differed.
        ``pairwise_spread`` : max-min of the pairwise thetas (None if <2
            pairs). Small = every pair agrees = trustworthy.
        ``leave_one_out``   : list of {dropped:(b,a), theta} — the all-but-
            one-view angle, one entry per dropped shared view. Only useful
            (and only populated) with >= 3 shared views; at exactly 3 views
            it equals ``pairwise`` so callers may hide it then.
        ``loo_swing``       : max-min of the leave-one-out thetas (None if
            < 2). How much the answer moves when any single view is removed.
        ``cond_gc`` / ``cond_vessel`` : reconstruction_condition() of the
            GC and vessel from the shared views (None / inf as documented).
    """
    gc_by = {_view_key(o): o for o in gc_obs}
    ve_by = {_view_key(o): o for o in vessel_obs}
    shared = sorted(k for k in ve_by if k in gc_by)

    out: dict = {
        "shared_views": [(k[0], k[1]) for k in shared],
        "pairwise": [],
        "pairwise_spread": None,
        "leave_one_out": [],
        "loo_swing": None,
        "cond_gc": None,
        "cond_vessel": None,
    }

    # Pairwise minimal (2-view) reconstructions.
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            ks = (shared[i], shared[j])
            try:
                th = _theta_from_obs(
                    [gc_by[k] for k in ks], [ve_by[k] for k in ks]
                )
            except ValueError:
                continue
            out["pairwise"].append({
                "views": [(ks[0][0], ks[0][1]), (ks[1][0], ks[1][1])],
                "theta": th,
            })
    pw = [p["theta"] for p in out["pairwise"]]
    if len(pw) >= 2:
        out["pairwise_spread"] = max(pw) - min(pw)

    # Leave-one-out (needs >= 3 shared so >= 2 remain).
    if len(shared) >= 3:
        for k_drop in shared:
            keep = [k for k in shared if k != k_drop]
            try:
                th = _theta_from_obs(
                    [gc_by[k] for k in keep], [ve_by[k] for k in keep]
                )
            except ValueError:
                continue
            out["leave_one_out"].append({
                "dropped": (k_drop[0], k_drop[1]), "theta": th,
            })
    loo = [p["theta"] for p in out["leave_one_out"]]
    if len(loo) >= 2:
        out["loo_swing"] = max(loo) - min(loo)

    # Per-reconstruction conditioning (on the shared views).
    if len(shared) >= 2:
        out["cond_gc"] = reconstruction_condition([gc_by[k] for k in shared])
        out["cond_vessel"] = reconstruction_condition(
            [ve_by[k] for k in shared]
        )
    return out


def subset_angles(gc_obs, vessel_obs) -> dict:
    """GC↔vessel 3-D angle reconstructed from EVERY subset (size ≥ 2) of the
    views that carry BOTH a GC and a vessel line — a transparent breakdown so
    the reader can see how each image-combination's angle compares (and whether
    the all-image value is representative).

    Returns:
        ``views``  : ordered ``[(beta, alpha), ...]`` — the image numbering
                     1..N used by the ``idx`` tuples below (same stable
                     (beta, alpha) order as the confidence cues).
        ``combos`` : list of ``{idx, size, theta, best_sep}`` for each subset,
                     size ascending then lexicographic. ``idx`` is the 1-based
                     image numbers; ``theta`` is the angle in degrees, or
                     ``None`` when the subset's most-separated view pair is
                     below :data:`MIN_VIEW_SEPARATION_DEG` (too parallel to
                     reconstruct — the UI shows "N/A").
    """
    gc_by = {_view_key(o): o for o in gc_obs}
    ve_by = {_view_key(o): o for o in vessel_obs}
    shared = sorted(k for k in ve_by if k in gc_by)
    views = [(k[0], k[1]) for k in shared]
    combos = []
    n = len(shared)
    for size in range(2, n + 1):
        for idx in combinations(range(n), size):
            ks = [shared[i] for i in idx]
            gco = [gc_by[k] for k in ks]
            veo = [ve_by[k] for k in ks]
            best_sep = best_view_separation_deg(gco)
            theta = None
            if best_sep >= MIN_VIEW_SEPARATION_DEG:
                try:
                    theta = _theta_from_obs(gco, veo)
                except ValueError:
                    theta = None
            combos.append({
                "idx": tuple(i + 1 for i in idx),   # 1-based image numbers
                "size": size,
                "theta": theta,
                "best_sep": best_sep,
            })
    return {"views": views, "combos": combos}


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
                                (worst pair — a confidence cue)
                            ``best_separation_deg`` : largest pairwise C-arm
                                angle (best pair — the solvability gate)
                            ``direction``     : reconstructed 3-D unit vector
                            ``cos_to_gc``     : |GC . v| (non-GC vessels only)
                            ``optimal_view``  : (non-GC only) the two opposite
                                C-arm angulations [(beta,alpha),(beta,alpha)]
                                that show this angle without foreshortening
                                (view down the GC/vessel plane normal), or
                                None when GC and the vessel are near-coaxial.
                            ``per_view_2d``   : (non-GC only) list of
                                {beta, alpha, angle_2d} — the GC-to-vessel
                                angle AS SEEN in each view that has both a GC
                                and a vessel line. Foreshortened, NOT the 3-D
                                answer; a confidence cue (see line_angle_2d).
                            ``spread_2d``     : (non-GC only) max-min of the
                                per-view 2-D angles, or None if <2 shared
                                views. Small = views agree = more trust.
                            ``confidence``    : (non-GC only) dict from
                                coaxial_confidence() — pairwise 2-view
                                angles, leave-one-out swing, and per-
                                reconstruction conditioning. Confidence cues
                                only; they do NOT change the reported angle.
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
        # Gate on the BEST-separated pair, not the worst: as long as one
        # pair of views is well apart the joint reconstruction is anchored.
        # Extra views that happen to sit close to another view only add
        # information (they never make the least-squares solution worse), so
        # adding a 3rd view must never turn a solvable case unsolvable.
        sep = view_separation_deg(obs)          # worst pair — confidence cue
        best_sep = best_view_separation_deg(obs)  # best pair — solvability
        if best_sep < MIN_VIEW_SEPARATION_DEG:
            warnings.append(
                f"{label}: the most-separated pair of views is only "
                f"{best_sep:.0f} deg apart (< {MIN_VIEW_SEPARATION_DEG:.0f} "
                f"deg) — reconstruction is unreliable; use more separated "
                f"C-arm angles."
            )
            continue
        d = vessel_direction(obs)
        directions[label] = d
        details[label] = {
            "views": [(float(o["beta"]), float(o["alpha"])) for o in obs],
            "separation_deg": sep,            # worst pair (kept for callers)
            "best_separation_deg": best_sep,  # best pair (the gate value)
            "direction": d,
        }
        # A close pair no longer blocks the result, but flag it so the
        # reader knows some views are near-redundant.
        if sep < MIN_VIEW_SEPARATION_DEG:
            warnings.append(
                f"{label}: two of its views are only {sep:.0f} deg apart "
                f"(< {MIN_VIEW_SEPARATION_DEG:.0f} deg); they add little and "
                f"the angle leans on the better-separated views — check the "
                f"per-pair spread / conditioning below."
            )

    angles: dict[str, float] = {}
    if GC in directions:
        gc = _unit(directions[GC])
        # Index the GC lines by their view so we can pair a GC line with a
        # vessel line drawn on the SAME view for the per-view 2-D cue.
        gc_by_view = {
            (round(float(o["beta"]), 3), round(float(o["alpha"]), 3)): o
            for o in by_label[GC]
        }
        for label, d in directions.items():
            if label == GC:
                continue
            c = abs(float(np.dot(gc, _unit(d))))
            c = max(-1.0, min(1.0, c))
            angles[label] = math.degrees(math.acos(c))
            details[label]["cos_to_gc"] = c
            # The two C-arm angulations that show this GC-to-vessel angle
            # without foreshortening (view down the normal of the GC/vessel
            # plane). None when GC and the vessel are near-coaxial.
            details[label]["optimal_view"] = optimal_projection_angles(
                gc, d
            )
            # Per-view 2-D GC-to-vessel angle, for views that carry BOTH a
            # GC and a vessel line. Confidence cue only (foreshortened).
            per_view = []
            for o in by_label[label]:
                key = (round(float(o["beta"]), 3), round(float(o["alpha"]), 3))
                gco = gc_by_view.get(key)
                if gco is None:
                    continue
                try:
                    a2d = line_angle_2d(
                        gco["line_2d"], o["line_2d"],
                        o.get("spacing", (1.0, 1.0)),
                    )
                except ValueError:
                    continue
                per_view.append({
                    "beta": float(o["beta"]),
                    "alpha": float(o["alpha"]),
                    "angle_2d": a2d,
                })
            details[label]["per_view_2d"] = per_view
            if len(per_view) >= 2:
                vals = [p["angle_2d"] for p in per_view]
                details[label]["spread_2d"] = max(vals) - min(vals)
            else:
                details[label]["spread_2d"] = None
            # Reconstruction-stability cues (pairwise 2-view angles,
            # leave-one-out swing, conditioning). 3-D answer unchanged.
            details[label]["confidence"] = coaxial_confidence(
                by_label[GC], by_label[label]
            )
            # Full per-image-combination breakdown (2..N views), each labelled
            # by C-arm angulation, with N/A for combinations too parallel to
            # solve. The all-image entry is the headline angle above.
            details[label]["subsets"] = subset_angles(
                by_label[GC], by_label[label]
            )
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
