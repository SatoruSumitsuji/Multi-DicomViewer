"""Pure geometry / interpolation engine for IVUS–XA co-registration.

For each angiographic view the user traces a smooth vessel *guide* curve
and pins a handful of *anchors*, each tying one IVUS pull-back frame to a
position along that view's guide. Given the IVUS master frame, this
module maps it to a fractional arc-length position (0..1) along the guide
and returns the ``(x, y)`` image-pixel point on the smooth curve. No Qt,
no DICOM, no numpy — just math, so it is unit-testable in isolation (same
discipline as :mod:`multi_dicomviewer.core.multisync`).

Conventions
-----------
* A *point* is an ``(x, y)`` tuple in image-pixel coordinates.
* A *guide* is the ordered list of user-clicked control vertices. The
  smooth display curve is derived from it via a centripetal Catmull-Rom
  spline (:func:`smooth_curve`) that passes through every vertex.
  Positions along the smooth curve are addressed by arc-length *fraction*
  ``s in [0, 1]`` (0 = first vertex, 1 = last).
* An *anchor* is a ``(frame, s)`` pair: IVUS frame index -> guide
  fraction. Anchors are stored per view; a view the user skipped for a
  given landmark simply has fewer of them, and its marker is interpolated
  from the anchors it does have (piecewise-linear, end-extrapolated —
  exactly like :func:`multi_dicomviewer.core.multisync.map_frame`). A
  view with fewer than two anchors cannot define a slope, so it shows the
  guide only and :func:`map_frame_to_fraction` returns ``None``.
"""
from __future__ import annotations


def _f(v):
    return float(v)


def _clamp01(s):
    return 0.0 if s < 0.0 else 1.0 if s > 1.0 else s


def _dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _lerp_pt(a, b, t):
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


# --------------------------------------------------------------------------
# Smooth curve from traced control vertices (centripetal Catmull-Rom)
# --------------------------------------------------------------------------

def _cr_segment(p0, p1, p2, p3, n, alpha):
    """Sample the centripetal Catmull-Rom curve on the p1->p2 span.

    Returns ``n + 1`` points from p1 to p2 inclusive (Barry-Goldman
    pyramid). Degenerate spans fall back to a straight p1->p2 line."""
    def tj(ti, pi, pj):
        d = _dist(pi, pj)
        return ti + (d ** alpha if d > 0 else 0.0)

    t0 = 0.0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)
    if t1 == t0 or t2 == t1 or t3 == t2:
        return [_lerp_pt(p1, p2, i / n) for i in range(n + 1)]

    out = []
    for i in range(n + 1):
        t = t1 + (t2 - t1) * (i / n)
        a1 = _lerp_pt(p0, p1, (t - t0) / (t1 - t0))
        a2 = _lerp_pt(p1, p2, (t - t1) / (t2 - t1))
        a3 = _lerp_pt(p2, p3, (t - t2) / (t3 - t2))
        b1 = _lerp_pt(a1, a2, (t - t0) / (t2 - t0))
        b2 = _lerp_pt(a2, a3, (t - t1) / (t3 - t1))
        out.append(_lerp_pt(b1, b2, (t - t1) / (t2 - t1)))
    return out


def smooth_curve(vertices, samples_per_seg=16, alpha=0.5):
    """Dense smooth polyline through the traced control *vertices*.

    Consecutive duplicate vertices are dropped. 0/1 vertices pass through
    unchanged; 2 vertices give a straight line. The phantom endpoints
    needed by Catmull-Rom are reflected (``2*P0 - P1``) so the spline
    passes cleanly through the first and last vertex without a coincident
    control point."""
    vs = [(_f(x), _f(y)) for x, y in vertices]
    dedup = []
    for p in vs:
        if not dedup or p != dedup[-1]:
            dedup.append(p)
    vs = dedup
    if len(vs) <= 2:
        return list(vs)

    head = (2 * vs[0][0] - vs[1][0], 2 * vs[0][1] - vs[1][1])
    tail = (2 * vs[-1][0] - vs[-2][0], 2 * vs[-1][1] - vs[-2][1])
    pad = [head] + vs + [tail]

    out = []
    for k in range(len(vs) - 1):
        seg = _cr_segment(pad[k], pad[k + 1], pad[k + 2], pad[k + 3],
                          samples_per_seg, alpha)
        out.extend(seg[1:] if out else seg)
    return out


# --------------------------------------------------------------------------
# Arc-length addressing along a curve
# --------------------------------------------------------------------------

def _cumlen(curve):
    cum = [0.0]
    for i in range(1, len(curve)):
        cum.append(cum[-1] + _dist(curve[i - 1], curve[i]))
    return cum


def arc_length(curve):
    """Total length of the polyline *curve* (0 for <2 points)."""
    if len(curve) < 2:
        return 0.0
    return _cumlen(curve)[-1]


def point_at_fraction(curve, s):
    """Point at arc-length fraction *s* in [0, 1] along *curve*.

    *s* is clamped to [0, 1]. Returns ``None`` for an empty curve, the
    lone vertex for a 1-point curve."""
    if not curve:
        return None
    if len(curve) == 1:
        return tuple(curve[0])
    s = _clamp01(s)
    cum = _cumlen(curve)
    total = cum[-1]
    if total <= 0:
        return tuple(curve[0])
    target = s * total
    for i in range(1, len(cum)):
        if cum[i] >= target:
            seg = cum[i] - cum[i - 1]
            t = (target - cum[i - 1]) / seg if seg > 0 else 0.0
            return _lerp_pt(curve[i - 1], curve[i], t)
    return tuple(curve[-1])


def project_fraction(curve, pt):
    """Nearest position on *curve* to *pt*: returns ``(s, distance)``.

    *s* is the arc-length fraction of the foot of the perpendicular
    (clamped to each segment); *distance* is the pixel distance from *pt*
    to that foot — the caller uses it as a snap tolerance. For a curve
    with fewer than two points, snaps to the lone vertex (or
    ``(0.0, inf)`` when empty)."""
    if len(curve) < 2:
        if curve:
            return (0.0, _dist(pt, curve[0]))
        return (0.0, float("inf"))
    cum = _cumlen(curve)
    total = cum[-1]
    best_s, best_d = 0.0, float("inf")
    for i in range(1, len(curve)):
        a, b = curve[i - 1], curve[i]
        abx, aby = b[0] - a[0], b[1] - a[1]
        seg2 = abx * abx + aby * aby
        if seg2 <= 0:
            t = 0.0
        else:
            t = ((pt[0] - a[0]) * abx + (pt[1] - a[1]) * aby) / seg2
            t = 0.0 if t < 0 else 1.0 if t > 1 else t
        fx, fy = a[0] + t * abx, a[1] + t * aby
        d = ((pt[0] - fx) ** 2 + (pt[1] - fy) ** 2) ** 0.5
        if d < best_d:
            arclen = cum[i - 1] + t * (seg2 ** 0.5)
            best_s = arclen / total if total > 0 else 0.0
            best_d = d
    return (best_s, best_d)


# --------------------------------------------------------------------------
# Frame -> guide-fraction mapping (per view; sparse anchors + interpolation)
# --------------------------------------------------------------------------

def _interp(pts, x):
    """Piecewise-linear y at *x* over sorted ``(x, y)`` *pts*, with the end
    intervals' slope used to extrapolate beyond the first / last point.
    Assumes ``len(pts) >= 2`` (mirrors
    :func:`multi_dicomviewer.core.multisync.map_frame`)."""
    if x <= pts[0][0]:
        (a0, b0), (a1, b1) = pts[0], pts[1]
        r = (b1 - b0) / (a1 - a0) if a1 != a0 else 0.0
        return b0 + (x - a0) * r
    if x >= pts[-1][0]:
        (a0, b0), (a1, b1) = pts[-2], pts[-1]
        r = (b1 - b0) / (a1 - a0) if a1 != a0 else 0.0
        return b1 + (x - a1) * r
    for i in range(len(pts) - 1):
        (a0, b0), (a1, b1) = pts[i], pts[i + 1]
        if a0 <= x <= a1:
            t = (x - a0) / (a1 - a0) if a1 != a0 else 0.0
            return b0 + t * (b1 - b0)
    return pts[-1][1]


def map_frame_to_fraction(anchors, frame):
    """IVUS *frame* -> guide fraction in [0, 1], or ``None``.

    *anchors* is an iterable of ``(frame, s)`` pairs. Fewer than two
    anchors -> ``None`` (no slope; the view shows the guide only). Two or
    more -> piecewise-linear between the bracketing anchors, end-interval
    extrapolated, then clamped to [0, 1]."""
    pts = sorted(((_f(f), _f(s)) for f, s in anchors), key=lambda fs: fs[0])
    if len(pts) < 2:
        return None
    return _clamp01(_interp(pts, _f(frame)))


def map_fraction_to_frame(anchors, s):
    """Inverse map: guide fraction *s* -> IVUS frame (float), or ``None``.

    The reverse of :func:`map_frame_to_fraction` — used when the user drags
    the on-guide marker to scrub the IVUS. *anchors* is the same list of
    ``(frame, s)`` pairs; here they are inverted to ``(s, frame)`` and the
    fraction is interpolated to a frame. The result is NOT clamped to a
    frame range (the caller clamps to ``[0, total - 1]``). Fewer than two
    anchors -> ``None``."""
    pts = sorted(((_f(sv), _f(f)) for f, sv in anchors), key=lambda sf: sf[0])
    if len(pts) < 2:
        return None
    return _interp(pts, _f(s))


def map_between(pairs, x):
    """Map *x* -> y over ``(x, y)`` *pairs*. Used to sync one IVUS
    pull-back's frame to another's from the frames they share on the CoReg
    landmarks (frame_A -> frame_B):

    * 0 pairs -> ``None`` (no correspondence yet — no sync).
    * 1 pair  -> 1:1 offset from the single anchor (``b0 + (x - a0)``), so
      moving the master by N frames moves the other by N.
    * ≥2 pairs -> piecewise-linear with end-interval extrapolation."""
    pts = sorted(((_f(a), _f(b)) for a, b in pairs), key=lambda ab: ab[0])
    if not pts:
        return None
    if len(pts) == 1:
        a0, b0 = pts[0]
        return b0 + (_f(x) - a0)
    return _interp(pts, _f(x))


def map_rotation(pairs, x):
    """Interpolated angle (deg) at *x* over ``(x, angle)`` *pairs* along the
    SHORTEST angular path, constant-clamped before the first / after the
    last pair; ``None`` with no pairs. Mirrors
    :func:`multi_dicomviewer.core.multisync.map_rotation` — 按分 (interpolate)
    each IVUS's cross-section rotation across the CoReg landmarks."""
    pts = sorted(((_f(a), _f(r)) for a, r in pairs), key=lambda ar: ar[0])
    if not pts:
        return None
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        (a0, r0), (a1, r1) = pts[i], pts[i + 1]
        if a0 <= x <= a1:
            t = (x - a0) / (a1 - a0) if a1 != a0 else 0.0
            delta = ((r1 - r0 + 180.0) % 360.0) - 180.0
            return r0 + t * delta
    return pts[-1][1]


def marker_point(curve, anchors, frame):
    """Convenience: the on-image marker point for *frame*, or ``None``.

    ``None`` when the view has too few anchors (see
    :func:`map_frame_to_fraction`) or the curve is empty."""
    s = map_frame_to_fraction(anchors, frame)
    if s is None:
        return None
    return point_at_fraction(curve, s)


def is_monotonic(anchors):
    """True if *anchors* map frame -> fraction monotonically.

    Frames must be strictly increasing and the fractions strictly
    monotonic (all rising or all falling); otherwise the mapping would
    fold back on itself and the marker would jump. The UI uses this to
    warn before committing a new anchor. Fewer than two anchors is
    vacuously monotonic."""
    pts = sorted(((_f(f), _f(s)) for f, s in anchors), key=lambda fs: fs[0])
    for i in range(len(pts) - 1):
        if pts[i + 1][0] == pts[i][0]:
            return False
    ss = [s for _, s in pts]
    inc = all(ss[i + 1] > ss[i] for i in range(len(ss) - 1))
    dec = all(ss[i + 1] < ss[i] for i in range(len(ss) - 1))
    return inc or dec
