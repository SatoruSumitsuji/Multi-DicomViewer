"""Pure 2-D measurement geometry shared by the CT viewer (VTK) and the
XA / IVUS image canvas (QPainter), so Line / Polyline / Ellipse / Polygon
behave identically everywhere.

Everything here is plain math on (x, y) tuples — no Qt, no VTK. The CT
viewer keeps its own rendering; the XA/IVUS canvas renders with QPainter;
both call these for distances, areas, the smoothed polygon outline and
the 長径/短径 (major/minor) caliper lines.
"""
from __future__ import annotations

import math

import numpy as np


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def seg_dist(px, py, a, b) -> float:
    """Distance from point (px,py) to segment a-b (2-D)."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def closest_point_on_segment(px, py, a, b):
    """Nearest point on segment a-b to (px,py), returned as (x,y)."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return (ax, ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return (ax + t * dx, ay + t * dy)


def project_to_polyline(p, pts):
    """Nearest point on polyline *pts* (list of (x,y)) to point *p*=(x,y).
    Constrains a marker to a measure's drawn outline. Returns (x,y); falls
    back to *p* when *pts* has fewer than two points."""
    px, py = float(p[0]), float(p[1])
    if not pts or len(pts) < 2:
        return (px, py)
    best, bestd = None, float("inf")
    for i in range(len(pts) - 1):
        c = closest_point_on_segment(px, py, pts[i], pts[i + 1])
        d = (c[0] - px) ** 2 + (c[1] - py) ** 2
        if d < bestd:
            bestd, best = d, c
    return best


def poly_area(pts) -> float:
    """Shoelace area of a simple polygon (caller scales to mm²)."""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def polygon_centroid(pts):
    """Area (shoelace) centroid of polygon *pts* — the physical centre of the
    enclosed region, used as the apex of a Center Angle. Falls back to the
    vertex mean for a degenerate (near-zero-area) polygon."""
    n = len(pts)
    if n == 0:
        return (0.0, 0.0)
    if n < 3:
        return (sum(q[0] for q in pts) / n, sum(q[1] for q in pts) / n)
    a2 = cx = cy = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a2) < 1e-9:                            # degenerate → vertex mean
        return (sum(q[0] for q in pts) / n, sum(q[1] for q in pts) / n)
    return (cx / (3.0 * a2), cy / (3.0 * a2))


def point_in_poly(x, y, poly) -> bool:
    """Ray-cast point-in-polygon (poly may repeat its first vertex)."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def convex_hull(pts):
    p = sorted(set((round(a, 6), round(b, 6)) for a, b in pts))
    if len(p) < 3:
        return p

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lo = []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    up = []
    for q in reversed(p):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    return lo[:-1] + up[:-1]


def min_width(pts) -> float:
    """Minimum caliper width of the points' convex hull."""
    h = convex_hull(pts)
    if len(h) < 3:
        return 0.0
    best = float("inf")
    for i in range(len(h)):
        ax, ay = h[i]
        bx, by = h[(i + 1) % len(h)]
        ex, ey = bx - ax, by - ay
        L = math.hypot(ex, ey)
        if L < 1e-9:
            continue
        nx, ny = -ey / L, ex / L
        ds = [(px - ax) * nx + (py - ay) * ny for px, py in h]
        best = min(best, max(ds) - min(ds))
    return 0.0 if best == float("inf") else best


def smooth_closed(pts, per_seg: int = 14):
    """Smooth CLOSED curve through polygon *pts* (periodic centripetal
    Catmull-Rom). Interpolates every vertex; centripetal keeps corner
    overshoot tiny and loop-free. Purely the drawn outline — area / HU /
    diameters are measured from the original vertices so smoothing never
    biases a clinical number. <3 points -> closed straight loop."""
    n = len(pts)
    if n < 3:
        return [tuple(q) for q in pts] + ([tuple(pts[0])] if n else [])
    P = [np.asarray(q, dtype=np.float64) for q in pts]

    def _kt(ti, a, b):                         # centripetal: alpha = 0.5
        d = float(np.linalg.norm(b - a))
        return ti + (math.sqrt(d) if d > 1e-9 else 1e-6)

    out = []
    for i in range(n):
        p0, p1 = P[(i - 1) % n], P[i]
        p2, p3 = P[(i + 1) % n], P[(i + 2) % n]
        t0 = 0.0
        t1 = _kt(t0, p0, p1)
        t2 = _kt(t1, p1, p2)
        t3 = _kt(t2, p2, p3)
        for s in range(per_seg):
            t = t1 + (t2 - t1) * (s / per_seg)
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
            out.append((float(c[0]), float(c[1])))
    out.append(out[0])
    return out


def smooth_open(pts, per_seg: int = 14):
    """Smooth OPEN curve interpolating *pts* (centripetal Catmull-Rom
    with endpoints duplicated so the curve starts at pts[0] and ends at
    pts[-1], passing through every interior vertex). For Polyline 'Spline'
    mode — the curve is open (not closed)."""
    n = len(pts)
    if n < 2:
        return [tuple(q) for q in pts]
    if n == 2:
        return [tuple(pts[0]), tuple(pts[1])]
    P = [np.asarray(q, dtype=np.float64) for q in pts]
    ext = [P[0]] + P + [P[-1]]

    def _kt(ti, a, b):
        d = float(np.linalg.norm(b - a))
        return ti + (math.sqrt(d) if d > 1e-9 else 1e-6)

    out = []
    for i in range(n - 1):
        p0, p1 = ext[i], ext[i + 1]
        p2, p3 = ext[i + 2], ext[i + 3]
        t0 = 0.0
        t1 = _kt(t0, p0, p1)
        t2 = _kt(t1, p1, p2)
        t3 = _kt(t2, p2, p3)
        for s in range(per_seg):
            t = t1 + (t2 - t1) * (s / per_seg)
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
            out.append((float(c[0]), float(c[1])))
    out.append((float(P[-1][0]), float(P[-1][1])))
    return out


def angle_at(a, b, c) -> float:
    """Degrees between rays a→b and a→c (vertex at *a*). User spec:
    Angle ABC = angle of line AB and line AC, vertex at A."""
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - a[0], c[1] - a[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def arc_through(outline, p1, p2, p3):
    """Sub-path of the CLOSED polyline *outline* running from p1 to p3 the way
    that passes through p2 — i.e. the very arc whose central angle a Center
    Angle measures. Returns a point list incl. the p1/p3 endpoints. The three
    points are assumed to lie on (or near) the outline (CA markers are snapped
    to it); each is located by nearest projection onto the outline."""
    pts = [(float(q[0]), float(q[1])) for q in outline]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    n = len(pts)
    if n < 3:
        return [tuple(p1), tuple(p3)]

    def pos(q):                                   # float position in [0, n)
        best_d, best = float("inf"), 0.0
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            ux, uy = bx - ax, by - ay
            L2 = ux * ux + uy * uy
            u = 0.0 if L2 < 1e-12 else max(0.0, min(
                1.0, ((q[0] - ax) * ux + (q[1] - ay) * uy) / L2))
            cx, cy = ax + u * ux, ay + u * uy
            d = (q[0] - cx) ** 2 + (q[1] - cy) ** 2
            if d < best_d:
                best_d, best = d, i + u
        return best

    s1, s2, s3 = pos(p1), pos(p2), pos(p3)

    def walk(sa, sb):                             # outline vertices in (sa, sb)
        end = sa + (sb - sa) % n
        cur, out, guard = math.floor(sa) + 1, [], 0
        while cur < end - 1e-9 and guard <= n:
            out.append(pts[cur % n])
            cur += 1
            guard += 1
        return out

    if (s2 - s1) % n <= (s3 - s1) % n:            # forward arc contains p2
        return [tuple(p1)] + walk(s1, s3) + [tuple(p3)]
    return [tuple(p1)] + list(reversed(walk(s3, s1))) + [tuple(p3)]


def central_arc_angle(center, p1, p2, p3):
    """Central angle (deg, 0..360) of the arc from p1 → p3 *passing
    through* p2, measured at *center*. Returns (span, theta1, theta3,
    going_ccw). theta values are in degrees in [0, 360)."""
    cx, cy = center

    def _th(p):
        return math.degrees(math.atan2(p[1] - cy, p[0] - cx)) % 360

    t1, t2, t3 = _th(p1), _th(p2), _th(p3)
    ccw = (t3 - t1) % 360
    if (t2 - t1) % 360 <= ccw:               # arc CCW from t1 contains t2
        return ccw, t1, t3, True
    return (360 - ccw), t1, t3, False        # arc goes CW


def ellipse_params(pts):
    """(cx, cy, a, b, theta) for a (possibly rotated) ellipse stored as four
    axis endpoints ``[maj0, maj1, min0, min1]`` — the two major-axis endpoints
    followed by the two minor-axis endpoints. ``a`` / ``b`` are the semi-major /
    semi-minor radii and ``theta`` is the major-axis angle (radians). ``a``/``b``
    come straight from the stored axes, so a minor axis dragged longer than the
    major one is tolerated; callers wanting max/min use major_minor's Dmax/Dmin.
    """
    maj0, maj1, min0, min1 = pts[:4]
    cx = (maj0[0] + maj1[0]) / 2.0
    cy = (maj0[1] + maj1[1]) / 2.0
    a = max(dist(maj0, maj1) / 2.0, 1e-6)
    b = max(dist(min0, min1) / 2.0, 1e-6)
    theta = math.atan2(maj1[1] - maj0[1], maj1[0] - maj0[0])
    return cx, cy, a, b, theta


def ellipse_cab(pts):
    """(cx, cy, a, b) — centre and semi-axes of a (possibly rotated) ellipse;
    see ellipse_params. Kept for callers that only need the centre or radii."""
    cx, cy, a, b, _ = ellipse_params(pts)
    return cx, cy, a, b


def ellipse_axes(pts):
    """Unit major direction (u) and minor direction (v ⟂ u) of an oblique
    ellipse ``[maj0, maj1, min0, min1]``. Same orientation that
    ``ellipse_params`` returns as an angle — exposed as direction vectors for
    callers (e.g. the pygfx CT viewer) that want u/v directly."""
    e1, e2 = pts[0], pts[1]
    dx, dy = e2[0] - e1[0], e2[1] - e1[1]
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    return (ux, uy), (-uy, ux)


def ellipse_outline(pts, n: int = 48):
    """Closed polyline (``n+1`` points, last == first) tracing a rotated
    ellipse stored as four axis endpoints; see ellipse_params."""
    cx, cy, a, b, th = ellipse_params(pts)
    ct, st = math.cos(th), math.sin(th)
    out = []
    for i in range(n + 1):
        t = i * 2 * math.pi / n
        x, y = a * math.cos(t), b * math.sin(t)
        out.append((cx + x * ct - y * st, cy + x * st + y * ct))
    return out


def ellipse_from_major(p0, p1, minor_ratio: float = 0.5):
    """Four axis endpoints ``[maj0, maj1, min0, min1]`` for an ellipse whose
    MAJOR axis runs ``p0 → p1`` (so an oblique drag yields an oblique ellipse).
    The minor radius defaults to ``minor_ratio`` × the semi-major length and is
    meant to be tuned afterwards by dragging the minor handles."""
    cx, cy = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
    a = dist(p0, p1) / 2.0
    b = max(a * minor_ratio, 1e-6)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    px, py = -math.sin(ang), math.cos(ang)
    return [tuple(p0), tuple(p1),
            (cx - b * px, cy - b * py), (cx + b * px, cy + b * py)]


def ellipse_drag(pts, vi, w, circle=False):
    """New oblique-ellipse points ``[maj0, maj1, min0, min1]`` after dragging
    handle *vi* to world point *w*. Shared by every viewer's
    ``_set_ellipse_handle`` so XA/IVUS and both CT renderers edit identically:
    a MAJOR endpoint (vi 0/1) moves freely (resize + rotate) keeping the
    current minor width; a MINOR endpoint (vi 2/3) changes only the minor width
    along the perpendicular through the centre.

    *circle* True locks the shape to a TRUE CIRCLE (minor = major) through every
    edit: dragging a major endpoint resizes/rotates about the centre with the
    minor kept equal, and dragging a minor endpoint resizes the whole circle
    uniformly (major re-fit to the same radius, direction and centre kept)."""
    e1, e2, m1, m2 = (list(q) for q in pts)
    if vi == 0:
        e1 = [w[0], w[1]]
    elif vi == 1:
        e2 = [w[0], w[1]]
    cx, cy = (e1[0] + e2[0]) / 2.0, (e1[1] + e2[1]) / 2.0
    dx, dy = e2[0] - e1[0], e2[1] - e1[1]
    L = math.hypot(dx, dy) or 1e-6
    ux, uy = dx / L, dy / L                        # major (axis) dir
    vx, vy = -uy, ux                               # minor (perpendicular) dir
    if vi in (0, 1):
        b = (L / 2.0 if circle                     # circle: minor = major
             else math.hypot(m2[0] - m1[0], m2[1] - m1[1]) / 2.0)  # keep width
    else:
        b = max(abs((w[0] - cx) * vx + (w[1] - cy) * vy), 1e-3)
        if circle:                                 # uniform resize about centre
            e1 = [cx - b * ux, cy - b * uy]
            e2 = [cx + b * ux, cy + b * uy]
    m1 = [cx - b * vx, cy - b * vy]
    m2 = [cx + b * vx, cy + b * vy]
    return [tuple(e1), tuple(e2), tuple(m1), tuple(m2)]


def major_minor(m):
    """Long-axis & short-axis of an ellipse/polygon as drawable 2-D
    segments plus lengths: ((p1,p2),(q1,q2),Dmax,Dmin). Ellipse -> its
    exact axes. Polygon -> longest chord (長径) + minimum caliper width
    (短径), measured from the ORIGINAL vertices so the drawn caliper
    lines match the reported numbers."""
    if m["type"] == "ellipse":
        # The stored axis endpoints ARE the caliper segments (rotated).
        maj0, maj1, min0, min1 = m["pts"][:4]
        maj, mnr = (maj0, maj1), (min0, min1)
        da, db = dist(maj0, maj1), dist(min0, min1)
        return (maj, mnr, da, db) if da >= db else (mnr, maj, db, da)
    # Measure on the SMOOTHED outline that is actually drawn (smooth_closed),
    # not the raw vertices: a smoothed/concave polygon departs from its raw
    # convex hull, so caliper endpoints taken off the raw hull floated off the
    # drawn curve (most visible on the short axis). Hull of the dense outline
    # hugs the curve, so both diameters' endpoints land on the line.
    outline = smooth_closed(list(m["pts"]))
    if len(outline) >= 2 and outline[-1] == outline[0]:
        outline = outline[:-1]
    hull = convex_hull(outline)
    if len(hull) < 3:
        return None, None, 0.0, 0.0
    best = (-1.0, hull[0], hull[0])
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            d = dist(hull[i], hull[j])
            if d > best[0]:
                best = (d, hull[i], hull[j])
    dmax, p1, p2 = best
    # Minimum caliper width via rotating-calipers over the hull edges. Track the
    # winning edge's base point + normal so the SHORT-axis segment can be drawn
    # as the real caliper (far vertex → its foot on the opposite supporting
    # edge) rather than floating around the centroid.
    bw, bn, bbase = float("inf"), (0.0, 1.0), hull[0]
    for i in range(len(hull)):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % len(hull)]
        ex, ey = bx - ax, by - ay
        L = math.hypot(ex, ey)
        if L < 1e-9:
            continue
        nx, ny = -ey / L, ex / L
        ds = [(qx - ax) * nx + (qy - ay) * ny for qx, qy in hull]
        w = max(ds) - min(ds)
        if w < bw:
            bw, bn, bbase = w, (nx, ny), (ax, ay)
    if dmax < 1e-9 or bw == float("inf"):
        return None, None, 0.0, 0.0
    nx, ny = bn
    bax, bay = bbase
    proj = [((qx - bax) * nx + (qy - bay) * ny, (qx, qy)) for qx, qy in hull]
    d_far, vfar = max(proj, key=lambda t: t[0])     # farthest vertex (on edge)
    d_base = min(proj, key=lambda t: t[0])[0]        # supporting (min) side
    width = d_far - d_base
    foot = (vfar[0] - nx * width, vfar[1] - ny * width)  # foot on the far line
    # Both endpoints lie on the polygon boundary; the segment length == bw.
    return (p1, p2), (vfar, foot), dmax, bw


# --------------------------------------------------------------------------
# Two-shape comparison: %Area difference + radial gap map between two closed
# outlines (Polygon / Ellipse). Shared by both CT viewers so Windows (VTK) and
# macOS (pygfx) compute identically; each viewer only renders the result.
# --------------------------------------------------------------------------

#: Gap colour bands (upper bound in mm, hex colour). A SMALL gap is the
#: clinically notable case, so it is the "hot" red end: <5 red, 5–7 orange,
#: 7–9 yellow, >9 green.
GAP_BANDS = (
    (5.0, "#b30000"),          # dark red — < 5 mm  (deep, distinct from orange)
    (7.0, "#ff8c00"),          # orange   — 5–7 mm  (bright, clearly orange)
    (9.0, "#f1c40f"),          # yellow   — 7–9 mm
    (float("inf"), "#2ecc71"),  # green   — > 9 mm
)


def gap_color(gap_mm: float) -> str:
    """Hex colour for a radial gap length (mm) per GAP_BANDS."""
    for hi, col in GAP_BANDS:
        if gap_mm < hi:
            return col
    return GAP_BANDS[-1][1]


def gap_linewidth(gap_mm: float, base: float = 1.6) -> float:
    """Radial line width for a gap: the hottest (smallest-gap) band is drawn
    thicker so the clinically-critical <5 mm red stands out."""
    return base * 2.4 if gap_mm < GAP_BANDS[0][0] else base


def gap_legend():
    """Legend rows [(label, hex)] derived from GAP_BANDS, so the legend always
    matches the actual colouring (single source of truth)."""
    rows = []
    lo = 0.0
    for hi, col in GAP_BANDS:
        if hi == float("inf"):
            rows.append((f">{lo:g} mm", col))
        elif lo == 0.0:
            rows.append((f"<{hi:g} mm", col))
        else:
            rows.append((f"{lo:g}-{hi:g} mm", col))
        lo = hi
    return rows


def percent_area_diff(area_big: float, area_small: float) -> float:
    """|big − small| / big × 100  (divide by the LARGER area)."""
    if area_big <= 1e-9:
        return 0.0
    return abs(area_big - area_small) / area_big * 100.0


def _ray_polyline_hit(cx, cy, dx, dy, poly):
    """Nearest t>0 where ray (cx,cy)+t·(dx,dy) crosses the CLOSED polyline
    *poly* (list of (x,y); segment n→0 closes it). Returns t or None.
    With (dx,dy) a unit vector, t is the distance in the same units as poly."""
    best_t = None
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-12:                       # ray ∥ segment
            continue
        rx, ry = ax - cx, ay - cy
        t = (rx * ey - ry * ex) / denom              # along the ray
        s = (rx * dy - ry * dx) / denom              # along the segment
        if t > 1e-9 and -1e-9 <= s <= 1.0 + 1e-9:
            if best_t is None or t < best_t:
                best_t = t
    return best_t


def radial_gap_compare(outer, inner, centroid, step_deg: float = 1.0):
    """Cast rays every *step_deg* over 360° from *centroid* (the LARGER shape's
    area centroid). For each ray, intersect the OUTER and INNER closed outlines
    and record the gap between the two crossings.

    Returns a list of dicts: {ang, inner:(x,y), outer:(x,y), gap}. Rays that
    miss either outline are skipped. Units follow the outlines (mm)."""
    cx, cy = centroid
    out = []
    n = max(1, int(round(360.0 / step_deg)))
    for k in range(n):
        ang = math.radians(k * step_deg)
        dx, dy = math.cos(ang), math.sin(ang)
        to = _ray_polyline_hit(cx, cy, dx, dy, outer)
        ti = _ray_polyline_hit(cx, cy, dx, dy, inner)
        if to is None or ti is None:
            continue
        out.append({
            "ang": k * step_deg,
            "inner": (cx + ti * dx, cy + ti * dy),
            "outer": (cx + to * dx, cy + to * dy),
            "gap": abs(to - ti),
        })
    return out
