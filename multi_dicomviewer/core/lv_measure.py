"""LV measurement workflow controller (Phase 1 foundation).

GUI-free (pure numpy) orchestration the CT viewers drive to measure LV volume:

    1. set_axis(basal1, basal2, apex)          # 3 picks on a reference long-axis
    2. for each φ in plane_angles():           # 4×45° or 6×30° rotated planes
         draw the endo (and epi) border →
         set_long_axis_contour(φ, pts3d)       # a 3-D polyline on that plane
    3. build()                                  # → short-axis ring stacks
    4. volume_ml() / myocardial_volume_ml()     # voxel count inside the surface

Each long-axis plane at rotation φ shows BOTH walls of the LV (the φ meridian
and the φ+180° meridian), so one traced border splits into two meridian
profiles by the sign of its in-plane radial coordinate. The viewer supplies the
border as absolute 3-D control points (volume mm) — the same ``pts3d`` it
already captures for a CPR trace — so this stays backend-independent and
testable headless. See [[lv_axis]], [[lv_surface]], [[lvef-feature]].
"""
from __future__ import annotations

import numpy as np

from .lv_axis import LVAxis
from .lv_surface import LV_LEVEL_STEP_MM, LV_RING_POINTS, LVSurface


class LVModel:
    """Holds the LV measurement state for one cardiac phase (ED or ES)."""

    def __init__(self, n_planes: int = 6):
        if n_planes not in (4, 6, 8):
            raise ValueError("n_planes must be 4, 6 or 8")
        self.n_planes = int(n_planes)
        self.axis: LVAxis | None = None
        # meridian angle (deg) -> (T,2) array of (along, radius)
        self.endo_contours: dict[float, np.ndarray] = {}
        self.epi_contours: dict[float, np.ndarray] = {}
        self.endo: LVSurface | None = None
        self.epi: LVSurface | None = None

    # ------------------------------------------------------------------- axis
    def set_axis(self, basal1, basal2, apex) -> None:
        """Define the long axis from the two basal points + the apex. Clears
        any contours built against a previous axis."""
        self.axis = LVAxis.from_points(basal1, basal2, apex)
        self.endo_contours.clear()
        self.epi_contours.clear()
        self.endo = self.epi = None

    def set_axis_from_frame(self, origin, axis_dir, radial0) -> None:
        """Define the long axis from the current long-axis VIEW instead of
        picked points: *origin* a point on the axis (crosshair), *axis_dir* the
        rotation axis (view up), *radial0* the θ=0 direction (view right). The
        apex/base extent is then derived from the traced borders. Clears any
        contours built against a previous axis."""
        self.axis = LVAxis.from_frame(origin, axis_dir, radial0)
        self.endo_contours.clear()
        self.epi_contours.clear()
        self.endo = self.epi = None

    def along_range(self, which: str = "endo"):
        """(apex_along, base_along) of the currently-captured *which* borders —
        the along span where every meridian has a point (apex = max of the
        per-meridian minima, base = min of the per-meridian maxima, the base
        cut plane ⟂ the axis). None if <1 border or they don't overlap."""
        store = self.endo_contours if which == "endo" else self.epi_contours
        if not store:
            return None
        mins, maxs = [], []
        for c in store.values():
            a = np.asarray(c, float).reshape(-1, 2)[:, 0]
            mins.append(float(a.min()))
            maxs.append(float(a.max()))
        apex, base = max(mins), min(maxs)
        return (apex, base) if base > apex else None

    def plane_angles(self) -> list[float]:
        """Rotation angles (deg) of the long-axis drawing planes. n planes span
        0..180° (each plane also covers its +180° wall)."""
        return [i * 180.0 / self.n_planes for i in range(self.n_planes)]

    def meridian_angles(self) -> list[float]:
        """The 2×n meridian angles the planes produce (deg, sorted)."""
        out: set[float] = set()
        for phi in self.plane_angles():
            out.add(phi % 360.0)
            out.add((phi + 180.0) % 360.0)
        return sorted(out)

    # -------------------------------------------------------------- contours
    def set_long_axis_contour(self, plane_angle: float, pts3d,
                              which: str = "endo") -> None:
        """Store a border traced on the long-axis plane at *plane_angle*.

        *pts3d* is an (N,3) array of volume-mm points along the border. It is
        split into the φ and φ+180° meridian profiles by the sign of the
        in-plane radial coordinate, each stored as (along, radius) sorted by
        along. *which* is "endo" or "epi"."""
        if self.axis is None:
            raise RuntimeError("set_axis() before adding contours")
        ax = self.axis
        e_s = ax.meridian_dir(plane_angle)
        pts = np.asarray(pts3d, dtype=float).reshape(-1, 3)
        d = pts - ax.apex
        along = d @ ax.axis
        s = d @ e_s                                   # signed radial in-plane
        pos_theta = plane_angle % 360.0
        neg_theta = (plane_angle + 180.0) % 360.0
        store = self.endo_contours if which == "endo" else self.epi_contours
        # On-axis points (s ≈ 0: the apex/base poles) belong to BOTH walls, so
        # each meridian reaches the pole — otherwise a pole assigned to only one
        # wall shortens the other meridian's along-range (a false apex/base cut).
        for theta, mask in ((pos_theta, s >= 0), (neg_theta, s <= 0)):
            if not np.any(mask):
                continue
            prof = np.column_stack([along[mask], np.abs(s[mask])])
            prof = prof[np.argsort(prof[:, 0])]
            store[theta] = prof

    def clear_contour(self, plane_angle: float, which: str = "endo") -> None:
        store = self.endo_contours if which == "endo" else self.epi_contours
        store.pop(plane_angle % 360.0, None)
        store.pop((plane_angle + 180.0) % 360.0, None)

    # ----------------------------------------------------------------- build
    def build(self, level_step: float = LV_LEVEL_STEP_MM,
              n_theta: int = LV_RING_POINTS) -> None:
        """(Re)build the endo (and epi, if traced) ring stacks from the stored
        meridian contours. Needs ≥3 meridians for a surface."""
        if self.axis is None:
            raise RuntimeError("set_axis() before build()")
        self.endo = (LVSurface.from_meridian_contours(
            self.axis, self.endo_contours, level_step, n_theta)
            if len(self.endo_contours) >= 3 else None)
        self.epi = (LVSurface.from_meridian_contours(
            self.axis, self.epi_contours, level_step, n_theta)
            if len(self.epi_contours) >= 3 else None)

    # ---------------------------------------------------------------- volume
    def volume_ml(self, spacing: float, which: str = "endo") -> float | None:
        """LV cavity (endo) or epicardial volume in mL, or None if that surface
        is not built yet."""
        surf = self.endo if which == "endo" else self.epi
        return None if surf is None else surf.voxel_volume_ml(spacing)

    def myocardial_volume_ml(self, spacing: float) -> float | None:
        """Myocardial volume = epi − endo (mL), or None if either is missing."""
        if self.endo is None or self.epi is None:
            return None
        return (self.epi.voxel_volume_ml(spacing)
                - self.endo.voxel_volume_ml(spacing))

    def myocardial_mass_g(self, spacing: float, density: float = 1.05
                          ) -> float | None:
        """Myocardial mass (g) = myocardial volume (mL) × density (g/mL,
        1.05 for myocardium)."""
        v = self.myocardial_volume_ml(spacing)
        return None if v is None else v * density
