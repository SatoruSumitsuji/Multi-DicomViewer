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


def poly_area(pts) -> float:
    """Shoelace area of a simple polygon (caller scales to mm²)."""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


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
    hull = convex_hull(list(m["pts"]))
    if len(hull) < 3:
        return None, None, 0.0, 0.0
    best = (-1.0, hull[0], hull[0])
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            d = dist(hull[i], hull[j])
            if d > best[0]:
                best = (d, hull[i], hull[j])
    dmax, p1, p2 = best
    cx = sum(q[0] for q in hull) / len(hull)
    cy = sum(q[1] for q in hull) / len(hull)
    bw, bn = float("inf"), (0.0, 1.0)
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
            bw, bn = w, (nx, ny)
    if dmax < 1e-9 or bw == float("inf"):
        return None, None, 0.0, 0.0
    nx, ny = bn
    q1 = (cx - nx * bw / 2.0, cy - ny * bw / 2.0)
    q2 = (cx + nx * bw / 2.0, cy + ny * bw / 2.0)
    return (p1, p2), (q1, q2), dmax, bw
