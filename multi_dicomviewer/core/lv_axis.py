"""Left-ventricle long axis + basal plane geometry for LVEF measurement.

GUI-free (pure numpy, no VTK / Qt / scipy) so it can be unit-tested headless
and shared by both CT viewers (Windows VTK and macOS pygfx).

The workflow this supports (see also [[lv_surface]]):

    3 picked points on a reference long-axis cross-section
        basal-1, basal-2  (the two mitral-annulus hinge corners)
        apex              (the LV tip)
        │  LVAxis.from_points()
        ▼
    long axis   = apex → midpoint(basal-1, basal-2)      (rotation axis)
    basal plane = ⟂ the long axis, through the basal midpoint   (v1: flat cut;
                  the mitral-annulus tilt is a later refinement)
    radial-0    = the in-plane radial toward basal-1 (the θ = 0° meridian)

The long-axis cross-section at rotation angle θ is the half-plane pair spanned
by the axis and the radial direction meridian_dir(θ); it shows BOTH the θ and
the θ+180° walls of the LV. Rotating θ by 45° (→ 4 planes / 8 meridians) or 30°
(→ 6 planes / 12 meridians) samples the whole ventricle.

All coordinates are plain 3-vectors in the volume's physical (mm) space — the
same space the reslice matrix's origin/axes live in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        if fallback is not None:
            return _unit(np.asarray(fallback, dtype=np.float64))
        return np.array([0.0, 0.0, 0.0])
    return v / n


@dataclass
class LVAxis:
    """LV long-axis frame in volume mm space.

    apex        : (3,)  the ventricular tip (along = 0)
    base_center : (3,)  midpoint of the two basal points (along = length_mm)
    axis        : (3,)  unit direction apex → base (the rotation axis)
    radial0     : (3,)  unit in-plane radial toward basal-1 (θ = 0° meridian)
    binormal    : (3,)  unit = axis × radial0 (θ = 90° meridian); the frame
                        (radial0, binormal, axis) is right-handed
    length_mm   : float apex → base distance
    """
    apex: np.ndarray
    base_center: np.ndarray
    axis: np.ndarray
    radial0: np.ndarray
    binormal: np.ndarray
    length_mm: float

    # ------------------------------------------------------------------ build
    @classmethod
    def from_points(cls, basal1, basal2, apex) -> "LVAxis":
        """Build from the two basal points and the apex (each a 3-vector in
        volume mm)."""
        b1 = np.asarray(basal1, dtype=np.float64).reshape(3)
        b2 = np.asarray(basal2, dtype=np.float64).reshape(3)
        ap = np.asarray(apex, dtype=np.float64).reshape(3)
        base_center = 0.5 * (b1 + b2)
        span = base_center - ap
        length = float(np.linalg.norm(span))
        if length < 1e-9:
            raise ValueError("apex and basal midpoint coincide — no long axis")
        axis = span / length
        # radial-0 = the component of (basal-1 - base_center) perpendicular to
        # the axis → the in-plane direction the reference long-axis plane was
        # drawn on. Fallback to any axis-perpendicular vector if degenerate
        # (the two basal points lay on the axis).
        v = b1 - base_center
        radial0 = _unit(v - float(np.dot(v, axis)) * axis,
                        fallback=_perp_any(axis))
        binormal = _unit(np.cross(axis, radial0))
        return cls(apex=ap, base_center=base_center, axis=axis,
                   radial0=radial0, binormal=binormal, length_mm=length)

    @classmethod
    def from_frame(cls, origin, axis_dir, radial0) -> "LVAxis":
        """Build the axis directly from a view frame: *origin* a point on the
        axis (the along = 0 reference, e.g. the crosshair centre), *axis_dir*
        the rotation-axis direction (the long-axis view's up), *radial0* the
        θ = 0 in-plane direction (the view's right).

        Used when the LV long axis is taken from the current long-axis MPR view
        instead of picked points — the apex/base extent is then defined by the
        traced borders (see LVSurface.from_meridian_contours), so ``apex`` here
        is just the along-origin and ``length_mm`` is a nominal placeholder."""
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        axis = _unit(axis_dir)
        if float(np.linalg.norm(axis)) < 1e-9:
            raise ValueError("degenerate axis direction")
        r = np.asarray(radial0, dtype=np.float64).reshape(3)
        radial0 = _unit(r - float(np.dot(r, axis)) * axis,
                        fallback=_perp_any(axis))
        binormal = _unit(np.cross(axis, radial0))
        return cls(apex=o, base_center=o + axis, axis=axis,
                   radial0=radial0, binormal=binormal, length_mm=0.0)

    # --------------------------------------------------------- meridian / xform
    def meridian_dir(self, theta_deg: float) -> np.ndarray:
        """Unit radial direction of the θ meridian (θ=0 → radial0,
        θ=90 → binormal). Rotates radial0 about the axis by θ."""
        a = np.radians(float(theta_deg))
        return np.cos(a) * self.radial0 + np.sin(a) * self.binormal

    def to_world(self, theta_deg: float, radius: float, along: float
                 ) -> np.ndarray:
        """3-D point at meridian *theta_deg*, *radius* mm from the axis, and
        *along* mm from the apex toward the base."""
        return (self.apex + float(along) * self.axis
                + float(radius) * self.meridian_dir(theta_deg))

    def project(self, p) -> tuple[float, float, float]:
        """Inverse of to_world: world point → (along, radius, theta_deg).
        *along* is the signed distance along the axis from the apex; *radius*
        the perpendicular distance from the axis; *theta_deg* in [0, 360)."""
        d = np.asarray(p, dtype=np.float64).reshape(3) - self.apex
        along = float(np.dot(d, self.axis))
        perp = d - along * self.axis
        radius = float(np.linalg.norm(perp))
        theta = np.degrees(np.arctan2(float(np.dot(perp, self.binormal)),
                                      float(np.dot(perp, self.radial0))))
        return along, radius, theta % 360.0

    def project_array(self, pts: np.ndarray):
        """Vectorised project() for an (N,3) array → (along (N,), x (N,),
        y (N,)) where (x, y) are the short-axis in-plane coords in the
        (radial0, binormal) basis (so radius = hypot(x, y))."""
        d = np.asarray(pts, dtype=np.float64).reshape(-1, 3) - self.apex
        along = d @ self.axis
        x = d @ self.radial0
        y = d @ self.binormal
        return along, x, y

    # -------------------------------------------------------- plane bases (UI)
    def short_axis_basis(self, along: float):
        """(origin, e_x, e_y, normal) of the short-axis plane at *along* mm.
        e_x = radial0, e_y = binormal, normal = axis."""
        origin = self.apex + float(along) * self.axis
        return origin, self.radial0.copy(), self.binormal.copy(), self.axis.copy()

    def long_axis_basis(self, theta_deg: float):
        """(origin, e_s, e_t, normal) of the long-axis cross-section at rotation
        *theta_deg*. e_s = meridian_dir(θ) (radial across; negative = the θ+180°
        wall), e_t = axis (apex→base), normal = e_s × e_t."""
        e_s = self.meridian_dir(theta_deg)
        e_t = self.axis.copy()
        normal = _unit(np.cross(e_s, e_t))
        return self.apex.copy(), e_s, e_t, normal

    @property
    def basal_plane(self):
        """(point, normal) of the v1 basal cut: ⟂ the axis, through the basal
        midpoint."""
        return self.base_center.copy(), self.axis.copy()


def _perp_any(axis: np.ndarray) -> np.ndarray:
    """Any unit vector perpendicular to *axis* (degenerate-radial0 fallback)."""
    axis = _unit(axis)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    return _unit(ref - float(np.dot(ref, axis)) * axis)
