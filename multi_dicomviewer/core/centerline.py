"""Vessel centreline geometry for CT curved-MPR / short-axis reformat.

GUI-free (pure numpy, no VTK / Qt / scipy) so it can be unit-tested headless
and shared by both CT viewers (Windows VTK and macOS pygfx).

The pipeline the viewers drive:

    control points (3-D, volume mm)          # a polyline lifted off one pane
        │  CenterLine.from_points()
        ▼
    Catmull-Rom spline, resampled at a
    uniform arc-length step                  # smooth, evenly spaced samples
        │  .frames(ref_up)
        ▼
    per-sample short-axis frame              # (origin, u, v, tangent):
    (rotation-minimising, no twist)          #   u,v span the plane ⟂ tangent

Each frame feeds a normal oblique reslice (the MPR engine already takes an
arbitrary u/v/n basis), so scrolling the sample index walks a continuous
stack of cross-sections perpendicular to the vessel — the CT analogue of an
IVUS pullback, ready to CoSync against one.

Coordinates are plain 3-vectors in the volume's physical (mm) space — the
same space the reslice matrix's origin/axes live in — so the viewer converts
a pane's 2-D measure point ``(x, y)`` to 3-D with ``origin + x*u + y*v`` and
hands the list straight in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        if fallback is not None:
            return np.asarray(fallback, dtype=np.float64)
        return np.array([0.0, 0.0, 0.0])
    return v / n


def _catmull_rom(p0, p1, p2, p3, ts: np.ndarray) -> np.ndarray:
    """Centripetal-flavoured Catmull-Rom segment from p1→p2 sampled at the
    parameters *ts* (0..1). Uniform parameterisation — good enough once the
    control points are the user's clicked vertices; the arc-length resample
    afterwards removes any speed non-uniformity."""
    p0, p1, p2, p3 = (np.asarray(p, np.float64) for p in (p0, p1, p2, p3))
    ts = ts[:, None]
    t2 = ts * ts
    t3 = t2 * ts
    return (
        0.5
        * (
            (2.0 * p1)
            + (-p0 + p2) * ts
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
    )


def _dense_spline(control: np.ndarray, per_seg: int = 24) -> np.ndarray:
    """Densely sample a Catmull-Rom spline through *control* (N,3). Endpoints
    are duplicated so the curve passes through the first / last vertex."""
    n = len(control)
    if n == 1:
        return control.copy()
    if n == 2:
        ts = np.linspace(0.0, 1.0, per_seg + 1)
        return control[0][None, :] * (1 - ts)[:, None] + control[1][None, :] * ts[:, None]
    # Duplicated (clamped) phantom endpoints: conservative — the curve never
    # extrapolates past the first/last clicked vertex. The trade-off is a small
    # one-sided tangent error at the very ends (~few°, decaying inward); the
    # interior tangents are full 2nd-order accurate. Extrapolating (reflected)
    # phantoms would over-shoot a curving vessel past where the user traced.
    ext = np.vstack([control[0], control, control[-1]])
    out = []
    ts = np.linspace(0.0, 1.0, per_seg, endpoint=False)
    for i in range(1, len(ext) - 2):
        seg = _catmull_rom(ext[i - 1], ext[i], ext[i + 1], ext[i + 2], ts)
        out.append(seg)
    out.append(control[-1][None, :])       # close on the final vertex exactly
    return np.vstack(out)


def _resample_arclength(dense: np.ndarray, step_mm: float):
    """Resample a dense polyline to a uniform arc-length *step_mm*.

    Returns (points (M,3), arclen (M,)). Guarantees at least the two
    endpoints; collapses duplicate points so a zero-length curve can't divide
    by zero."""
    d = np.diff(dense, axis=0)
    seg = np.linalg.norm(d, axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    dense = dense[keep]
    if len(dense) < 2:
        return dense.copy(), np.zeros(len(dense))
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    step_mm = max(float(step_mm), 1e-3)
    m = max(1, int(round(total / step_mm)))
    targets = np.linspace(0.0, total, m + 1)
    pts = np.empty((len(targets), 3))
    for k in range(3):
        pts[:, k] = np.interp(targets, cum, dense[:, k])
    return pts, targets


def _tangents(points: np.ndarray) -> np.ndarray:
    """Unit tangents via central differences (one-sided at the ends)."""
    n = len(points)
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]])
    t = np.zeros_like(points)
    t[1:-1] = points[2:] - points[:-2]        # 2nd-order central (interior)
    if n >= 3:
        # 2nd-order one-sided at the ends (−3p₀+4p₁−p₂ / 3p_last−4p₋₂+p₋₃):
        # a plain forward difference is only 1st-order and left the endpoint
        # tangent visibly off-axis on a curved trace.
        t[0] = -3.0 * points[0] + 4.0 * points[1] - points[2]
        t[-1] = 3.0 * points[-1] - 4.0 * points[-2] + points[-3]
    else:
        t[0] = points[1] - points[0]
        t[-1] = points[-1] - points[-2]
    prev = np.array([0.0, 0.0, 1.0])
    for i in range(n):
        t[i] = _unit(t[i], fallback=prev)
        prev = t[i]
    return t


def _rmf_frames(points: np.ndarray, tangents: np.ndarray, ref_up: np.ndarray):
    """Rotation-minimising frames (double-reflection method, Wang et al.).

    Returns (u (M,3), v (M,3)) orthonormal axes spanning the plane ⟂ tangent
    at every sample, evolving with minimal twist so a scrolled cross-section
    doesn't spin. The first frame's u is *ref_up* projected ⟂ the first
    tangent (falls back to an arbitrary perpendicular if ref_up ∥ tangent),
    which keeps the reformat's 'up' close to the plane the user traced on."""
    n = len(points)
    u = np.zeros_like(points)
    v = np.zeros_like(points)
    t0 = tangents[0]
    r = _unit(np.asarray(ref_up, np.float64) - np.dot(ref_up, t0) * t0)
    if float(np.linalg.norm(r)) < 1e-9:            # ref_up ∥ tangent
        alt = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(alt, t0))) > 0.9:
            alt = np.array([0.0, 1.0, 0.0])
        r = _unit(alt - np.dot(alt, t0) * t0)
    u[0] = r
    v[0] = _unit(np.cross(t0, r))
    for i in range(n - 1):
        v1 = points[i + 1] - points[i]
        c1 = float(np.dot(v1, v1))
        if c1 < 1e-18:                              # coincident samples
            u[i + 1], v[i + 1] = u[i], v[i]
            continue
        rL = u[i] - (2.0 / c1) * np.dot(v1, u[i]) * v1
        tL = tangents[i] - (2.0 / c1) * np.dot(v1, tangents[i]) * v1
        v2 = tangents[i + 1] - tL
        c2 = float(np.dot(v2, v2))
        rN = rL if c2 < 1e-18 else rL - (2.0 / c2) * np.dot(v2, rL) * v2
        u[i + 1] = _unit(rN, fallback=u[i])
        v[i + 1] = _unit(np.cross(tangents[i + 1], u[i + 1]), fallback=v[i])
    return u, v


@dataclass
class CenterLine:
    """A resampled vessel centreline in volume mm space.

    points   : (M,3) uniform-arc-length samples along the curve
    arclen   : (M,)  arc length (mm) of each sample from the start
    tangents : (M,3) unit tangent at each sample
    """
    points: np.ndarray
    arclen: np.ndarray
    tangents: np.ndarray

    @classmethod
    def from_points(cls, control: Sequence[Sequence[float]],
                    step_mm: float = 0.5) -> "CenterLine":
        """Build from clicked 3-D control vertices (≥1). *step_mm* is the
        cross-section spacing along the vessel (the pullback pitch)."""
        control = np.asarray(control, dtype=np.float64).reshape(-1, 3)
        if len(control) == 0:
            raise ValueError("centreline needs at least one control point")
        dense = _dense_spline(control)
        pts, arclen = _resample_arclength(dense, step_mm)
        tang = _tangents(pts)
        return cls(points=pts, arclen=arclen, tangents=tang)

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def length_mm(self) -> float:
        return float(self.arclen[-1]) if len(self.arclen) else 0.0

    def frames(self, ref_up: Sequence[float]):
        """Per-sample short-axis frames (u, v) spanning the plane ⟂ tangent,
        rotation-minimising from *ref_up* (typically the traced pane's
        normal). Returns (u (M,3), v (M,3))."""
        ref = np.asarray(ref_up, np.float64)
        return _rmf_frames(self.points, self.tangents, ref)
