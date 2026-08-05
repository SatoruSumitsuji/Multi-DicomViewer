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


def _cr_densify_3d(pts: np.ndarray, per_seg: int = 8) -> np.ndarray:
    """Centripetal Catmull-Rom densification of an OPEN (N,3) polyline — the SAME
    spline the viewer draws the long-axis border with. Intersecting the DENSE
    curve (not the raw chords) with a short-axis level makes the crossing lie on
    the displayed border even where it curves hard (the apex), so the short-axis
    points match the long-axis line there."""
    P = np.asarray(pts, float).reshape(-1, 3)
    n = len(P)
    if n < 3:
        return P
    ext = np.vstack([P[0], P, P[-1]])          # duplicate endpoints
    out = []

    def _kt(ti, a, b):
        d = float(np.linalg.norm(b - a))
        return ti + (d ** 0.5 if d > 1e-9 else 1e-6)

    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
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
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(ext[-2])
    return np.asarray(out)


class LVModel:
    """Holds the LV measurement state for one cardiac phase (ED or ES)."""

    def __init__(self, n_planes: int = 6):
        if n_planes not in (4, 6, 8):
            raise ValueError("n_planes must be 4, 6 or 8")
        self.n_planes = int(n_planes)
        self.axis: LVAxis | None = None
        # meridian angle (deg) -> (T,2) array of (along, radius) [for the volume]
        self.endo_contours: dict[float, np.ndarray] = {}
        self.epi_contours: dict[float, np.ndarray] = {}
        # plane angle (deg) -> (N,3) RAW traced border in volume mm [for the
        # short-axis display: intersect this polyline with the level plane]
        self.endo_planes: dict[float, np.ndarray] = {}
        self.epi_planes: dict[float, np.ndarray] = {}
        self.endo: LVSurface | None = None
        self.epi: LVSurface | None = None

    # ------------------------------------------------------------------- axis
    def set_axis(self, basal1, basal2, apex) -> None:
        """Define the long axis from the two basal points + the apex. Clears
        any contours built against a previous axis."""
        self.axis = LVAxis.from_points(basal1, basal2, apex)
        self.endo_contours.clear()
        self.epi_contours.clear()
        self.endo_planes.clear()
        self.epi_planes.clear()
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
        self.endo_planes.clear()
        self.epi_planes.clear()
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

    def level_range(self, which: str = "endo"):
        """Apex→base span for SCROLLING the short-axis display level: apex = the
        MOST-apical point of any meridian (min of per-meridian minima) so the
        level can reach the true apex even if one meridian's trace stopped
        early; base = the common base cut (min of per-meridian maxima). Wider on
        the apical side than along_range() (which needs every meridian present).
        None if no borders."""
        store = self.endo_contours if which == "endo" else self.epi_contours
        if not store:
            return None
        mins, maxs = [], []
        for c in store.values():
            a = np.asarray(c, float).reshape(-1, 2)[:, 0]
            mins.append(float(a.min()))
            maxs.append(float(a.max()))
        lo, hi = min(mins), min(maxs)
        return (lo, hi) if hi > lo else None

    def short_axis_border_pts(self, along0: float, which: str = "endo"):
        """The border points (3-D volume mm) where the traced *which* border
        crosses the short-axis plane at axial position *along0* (⟂ the rotation
        axis), ORDERED by angle θ around the axis so a closed spline can be
        drawn straight through them. None if <3 crossings.

        Method (the DIRECT one, per the clinician's spec): intersect the level
        plane at *along0* with each traced long-axis border POLYLINE — every
        segment whose two ends straddle along0 yields one crossing point, right
        on the drawn border. This makes no assumption about the border being a
        clean function of along (so a wall that dips toward the axis, or a trace
        that wiggles, can't misplace a point the way the along/radius split
        could); the short-axis passes exactly through the long-axis crossings.
        A plane whose border doesn't reach this level contributes nothing (near
        the apex the ring simply shrinks through the planes that still cross)."""
        if self.axis is None:
            return None
        planes = self.endo_planes if which == "endo" else self.epi_planes
        if len(planes) < 2:                    # <2 planes ⇒ <4 crossings
            return None
        ax = self.axis
        # For each PLANE, intersect its traced border with the level and assign
        # every crossing to that plane's own meridian by the SIGN of the in-plane
        # radial (e_s): s≥0 → φ wall, s<0 → φ+180 wall. This is robust even when
        # the crossing is near the axis (small radius) — where a global arctan2(θ)
        # would be noisy and mis-bin the point. Keep the OUTERMOST crossing per
        # meridian (drops papillary/wiggle inner crossings) and order the result
        # by meridian angle (deterministic — no θ), giving one clean point per
        # direction that sits exactly on the long-axis border crossing.
        best: dict[float, tuple[float, np.ndarray]] = {}
        for phi, pts3d in planes.items():
            # Densify to the DISPLAYED spline first, so the crossing lies on the
            # drawn long-axis border (matches it even at the curved apex).
            p = _cr_densify_3d(pts3d)
            if len(p) < 2:
                continue
            along = (p - ax.apex) @ ax.axis
            e_s = ax.meridian_dir(phi)
            for i in range(len(p) - 1):
                a0, a1 = float(along[i]), float(along[i + 1])
                if a0 == a1:
                    continue
                t = (along0 - a0) / (a1 - a0)
                if not (-1e-9 <= t <= 1.0 + 1e-9):
                    continue
                P = p[i] + t * (p[i + 1] - p[i])
                s = float((P - ax.apex) @ e_s)          # signed in-plane radial
                mth = (phi % 360.0) if s >= 0.0 else ((phi + 180.0) % 360.0)
                if mth not in best or abs(s) > best[mth][0]:
                    best[mth] = (abs(s), P)
        if len(best) < 3:
            return None
        return np.asarray([best[a][1] for a in sorted(best)])

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
        # Keep the RAW traced polyline for this plane — the short-axis display
        # intersects it with the level plane directly (see short_axis_border_pts).
        planes = self.endo_planes if which == "endo" else self.epi_planes
        planes[plane_angle % 360.0] = pts.copy()
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
            # Keep the RAW (along, radius) samples (strictly-increasing along) —
            # short_axis_border_pts intersects them with the level directly.
            keep = np.concatenate(([True], np.diff(prof[:, 0]) > 1e-9))
            store[theta] = prof[keep]

    def clear_contour(self, plane_angle: float, which: str = "endo") -> None:
        store = self.endo_contours if which == "endo" else self.epi_contours
        store.pop(plane_angle % 360.0, None)
        store.pop((plane_angle + 180.0) % 360.0, None)
        planes = self.endo_planes if which == "endo" else self.epi_planes
        planes.pop(plane_angle % 360.0, None)
        planes.pop((plane_angle + 180.0) % 360.0, None)

    # ----------------------------------------------------- persistence (JSON)
    def to_dict(self) -> dict:
        """Serialise the axis + traced Endo/Epi borders as 3-D volume-mm data —
        enough to fully re-apply them to the same series later (LVEF workflow)."""
        ax = self.axis
        return {
            "kind": "mdv-lvef", "version": 1,
            "n_planes": self.n_planes,
            "axis": None if ax is None else {
                "origin": [float(x) for x in ax.apex],
                "axis": [float(x) for x in ax.axis],
                "radial0": [float(x) for x in ax.radial0],
            },
            "endo_planes": {f"{k:g}": np.asarray(v, float).reshape(-1, 3).tolist()
                            for k, v in self.endo_planes.items()},
            "epi_planes": {f"{k:g}": np.asarray(v, float).reshape(-1, 3).tolist()
                           for k, v in self.epi_planes.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LVModel":
        """Rebuild a model from to_dict() output (axis + Endo/Epi 3-D borders)."""
        m = cls(n_planes=int(d.get("n_planes", 6)))
        ax = d.get("axis")
        if ax:
            m.set_axis_from_frame(ax["origin"], ax["axis"], ax["radial0"])
        for which, key in (("endo", "endo_planes"), ("epi", "epi_planes")):
            for k, pts in (d.get(key) or {}).items():
                arr = np.asarray(pts, float).reshape(-1, 3)
                if len(arr) >= 2:
                    m.set_long_axis_contour(float(k), arr, which=which)
        return m

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
