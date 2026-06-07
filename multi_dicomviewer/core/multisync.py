"""Pure frame/rotation sync-mapping engine for the MultiSync IVUS viewer.

Up to 4 IVUS pull-backs are shown together. A SyncPoint pins, for every
loaded slot, a frame index and a rotation angle. One slot is the
*master*; given the master's current frame, each other slot's frame and
rotation are piecewise-linear interpolated between the bracketing sync
points (the end intervals' ratio is used to extrapolate beyond the first
/ last point). No Qt, no DICOM here — just math, so it is unit-testable.

A SyncPoint is a dict: ``{"frames": [f0..f3], "rots": [r0..r3]}`` where
each entry is None when that slot's frame is not set yet.
"""
from __future__ import annotations


def _pairs(sync_points, master, slot):
    """Sorted (master_frame, slot_frame) pairs for points that have both
    frames set."""
    out = []
    for p in sync_points:
        a = p["frames"][master]
        b = p["frames"][slot]
        if a is not None and b is not None:
            out.append((float(a), float(b)))
    out.sort(key=lambda ab: ab[0])
    return out


def map_frame(sync_points, master, slot, fm,
              fps_master=30.0, fps_slot=30.0):
    """Master frame *fm* -> slot frame (float; caller rounds + clamps).

    0 sync pairs -> FPS-ratio passthrough; 1 pair -> offset + FPS ratio;
    ≥2 -> piecewise-linear with end-interval extrapolation."""
    if slot == master:
        return float(fm)
    pts = _pairs(sync_points, master, slot)
    ratio = (fps_slot or 30.0) / (fps_master or 30.0)
    if not pts:
        return fm * ratio
    if len(pts) == 1:
        a0, b0 = pts[0]
        return b0 + (fm - a0) * ratio
    if fm <= pts[0][0]:
        (a0, b0), (a1, b1) = pts[0], pts[1]
        r = (b1 - b0) / (a1 - a0) if a1 != a0 else ratio
        return b0 + (fm - a0) * r
    if fm >= pts[-1][0]:
        (a0, b0), (a1, b1) = pts[-2], pts[-1]
        r = (b1 - b0) / (a1 - a0) if a1 != a0 else ratio
        return b1 + (fm - a1) * r
    for i in range(len(pts) - 1):
        (a0, b0), (a1, b1) = pts[i], pts[i + 1]
        if a0 <= fm <= a1:
            t = (fm - a0) / (a1 - a0) if a1 != a0 else 0.0
            return b0 + t * (b1 - b0)
    return float(fm)


def map_rotation(sync_points, master, slot, fm):
    """Interpolated rotation (deg) for *slot* at master frame *fm*.
    Uses sync points that have the master frame set; constant-clamped
    before the first / after the last point.

    Between two points the rotation is interpolated along the SHORTEST
    angular path: the delta is wrapped to (-180, 180] so e.g. 350° → 10°
    sweeps +20° forward, not -340° backward (the "spins almost all the
    way round to line up" artefact the user saw)."""
    pts = []
    for p in sync_points:
        a = p["frames"][master]
        if a is not None:
            pts.append((float(a), float(p["rots"][slot] or 0.0)))
    pts.sort(key=lambda ar: ar[0])
    if not pts:
        return 0.0
    if fm <= pts[0][0]:
        return pts[0][1]
    if fm >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        (a0, r0), (a1, r1) = pts[i], pts[i + 1]
        if a0 <= fm <= a1:
            t = (fm - a0) / (a1 - a0) if a1 != a0 else 0.0
            delta = ((r1 - r0 + 180.0) % 360.0) - 180.0
            return r0 + t * delta
    return pts[-1][1]


def has_conflict(sync_points, master, n_slots):
    """True if, for some slot, the sync points' frames are NOT strictly
    monotonic with the master's — which would fold the mapping back on
    itself (the "矛盾" the user must be warned about)."""
    for slot in range(n_slots):
        if slot == master:
            continue
        pts = _pairs(sync_points, master, slot)
        for i in range(len(pts) - 1):
            if pts[i + 1][0] == pts[i][0]:        # same master frame
                return True
            if pts[i + 1][1] <= pts[i][1]:        # slot frame not rising
                return True
    return False


def would_conflict(sync_points, master, n_slots, pi, slot, value):
    """Would setting ``sync_points[pi]['frames'][slot] = value`` create a
    conflict? Evaluated on a shallow copy so the caller can reject the
    edit before committing it."""
    sim = [
        {"frames": list(p["frames"]), "rots": list(p["rots"])}
        for p in sync_points
    ]
    sim[pi]["frames"][slot] = value
    return has_conflict(sim, master, n_slots)
