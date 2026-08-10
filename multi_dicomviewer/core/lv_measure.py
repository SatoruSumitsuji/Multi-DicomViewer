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


def _outermost_by_level(items, step):
    """From [(along, radius, P3d), …] keep the OUTERMOST (largest radius)
    crossing per axial-level bin (bin width *step* mm). Drops a papillary /
    inner re-entry so one wall yields one clean point per level."""
    best: dict = {}
    for along, rad, P in items:
        key = int(round(along / max(step, 1e-6)))
        if key not in best or rad > best[key][1]:
            best[key] = (along, rad, P)
    return list(best.values())


class LVModel:
    """Holds the LV measurement state for one cardiac phase (ED or ES)."""

    def __init__(self, n_planes: int = 6):
        if n_planes not in (4, 6, 8):
            raise ValueError("n_planes must be 4, 6 or 8")
        self.n_planes = int(n_planes)
        # Endo (lumen) and Epi (myocardium) are INDEPENDENT analyses, each traced
        # on its OWN long axis (their apexes are offset, so a single shared axis
        # can't put both apexes at the bottom of their trace planes — see
        # [[lv-apex-point-feature]]). ``axis`` points at the ACTIVE pass's axis
        # (so the tracing code stays axis-agnostic); build()/SAX pick per-surface.
        self.endo_axis: LVAxis | None = None
        self.epi_axis: LVAxis | None = None
        self.axis: LVAxis | None = None          # = the active pass's axis
        # meridian angle (deg) -> (T,2) array of (along, radius) [for the volume]
        self.endo_contours: dict[float, np.ndarray] = {}
        self.epi_contours: dict[float, np.ndarray] = {}
        # plane angle (deg) -> (N,3) RAW traced border in volume mm [for the
        # short-axis display: intersect this polyline with the level plane]
        self.endo_planes: dict[float, np.ndarray] = {}
        self.epi_planes: dict[float, np.ndarray] = {}
        # User-defined apex vertices (volume mm) each surface converges to — the
        # endocardial (lumen) apex and the epicardial (myocardial) apex, set
        # before tracing. None → synthesise a rounded cap from the traced rings.
        self.endo_apex: np.ndarray | None = None
        self.epi_apex: np.ndarray | None = None
        self.endo: LVSurface | None = None
        self.epi: LVSurface | None = None

    # ------------------------------------------------------------------- axis
    def _axis_for(self, which: str) -> "LVAxis | None":
        return self.endo_axis if which == "endo" else self.epi_axis

    def set_axis(self, basal1, basal2, apex) -> None:
        """Define BOTH long axes from the two basal points + the apex (legacy /
        shared-axis entry). Clears all contours."""
        ax = LVAxis.from_points(basal1, basal2, apex)
        self.endo_axis = ax
        self.epi_axis = LVAxis.from_points(basal1, basal2, apex)
        self.axis = self.epi_axis
        self.endo_contours.clear()
        self.epi_contours.clear()
        self.endo_planes.clear()
        self.epi_planes.clear()
        self.endo_apex = self.epi_apex = None
        self.endo = self.epi = None

    def set_axis_from_frame(self, origin, axis_dir, radial0,
                            which: str | None = None) -> None:
        """Define a long axis from the current long-axis VIEW: *origin* a point
        on the axis (crosshair), *axis_dir* the rotation axis (view up),
        *radial0* the θ=0 direction (view right). The apex/base extent is derived
        from the traced borders. *which* ("endo"/"epi") sets that pass's axis and
        clears only that surface's data; None sets BOTH (legacy) and clears all.
        ``axis`` is left pointing at the axis just set (the active pass)."""
        ax = LVAxis.from_frame(origin, axis_dir, radial0)
        if which in (None, "endo"):
            self.endo_axis = ax if which == "endo" \
                else LVAxis.from_frame(origin, axis_dir, radial0)
            self.endo_contours.clear()
            self.endo_planes.clear()
            self.endo_apex = None
            self.endo = None
        if which in (None, "epi"):
            self.epi_axis = ax if which == "epi" \
                else LVAxis.from_frame(origin, axis_dir, radial0)
            self.epi_contours.clear()
            self.epi_planes.clear()
            self.epi_apex = None
            self.epi = None
        self.axis = self._axis_for(which) if which else self.epi_axis

    def set_apex_point(self, which: str, p3d) -> None:
        """Set the user-defined apex vertex (volume mm) for *which* surface
        ("endo" = lumen apex, "epi" = myocardial apex). None clears it (falls
        back to a synthesised cap). Applied to a built surface on the next
        build()."""
        p = None if p3d is None else np.asarray(p3d, float).reshape(3)
        if which == "endo":
            self.endo_apex = p
        elif which == "epi":
            self.epi_apex = p
        else:
            raise ValueError("which must be 'endo' or 'epi'")

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

    def short_axis_border_pts(self, along0: float, which: str = "endo",
                              ref_axis: "LVAxis | None" = None):
        """The border points (3-D volume mm) where the traced *which* border
        crosses the short-axis plane at axial position *along0* along *ref_axis*
        (the SAX reference axis — the EPI axis, so BOTH borders show on the same
        cut even though Endo/Epi were traced on their own axes), ORDERED by angle
        θ around *ref_axis*. None if <3 crossings.

        Intersect the level plane at *along0* with each traced border POLYLINE —
        every segment straddling along0 yields one crossing right on the drawn
        border. Bin the crossings by θ around ref_axis (2×n_planes bins), keep
        the OUTERMOST per bin (drops papillary/wiggle inner crossings), and
        return them θ-ordered so a closed spline can be drawn through them."""
        ax = ref_axis if ref_axis is not None else self.axis
        if ax is None:
            return None
        planes = self.endo_planes if which == "endo" else self.epi_planes
        if len(planes) < 2:                    # <2 planes ⇒ <4 crossings
            return None
        # SAME-axis (the border's own axis): bin each crossing to that PLANE's
        # meridian by the SIGN of its in-plane radial (robust near the axis where
        # a global θ is noisy) — points land exactly on the 12 meridians. Only
        # for a CROSS-axis reference (endo shown on the epi axis) fall back to
        # global-θ binning, since the plane φ no longer maps to this axis.
        same_axis = (ax is self._axis_for(which))
        best: dict = {}
        for phi, pts3d in planes.items():
            # Densify to the DISPLAYED spline first, so the crossing lies on the
            # drawn long-axis border (matches it even at the curved apex).
            p = _cr_densify_3d(pts3d)
            if len(p) < 2:
                continue
            along = (p - ax.apex) @ ax.axis
            e_s = ax.meridian_dir(phi) if same_axis else None
            for i in range(len(p) - 1):
                a0, a1 = float(along[i]), float(along[i + 1])
                if a0 == a1:
                    continue
                t = (along0 - a0) / (a1 - a0)
                if not (-1e-9 <= t <= 1.0 + 1e-9):
                    continue
                P = p[i] + t * (p[i + 1] - p[i])
                if same_axis:
                    s = float((P - ax.apex) @ e_s)      # signed in-plane radial
                    key = (phi % 360.0) if s >= 0.0 else ((phi + 180.0) % 360.0)
                    rad = abs(s)
                    ordv = key
                else:
                    d = P - ax.apex
                    x, y = float(d @ ax.radial0), float(d @ ax.binormal)
                    rad = float(np.hypot(x, y))
                    ordv = float(np.degrees(np.arctan2(y, x))) % 360.0
                    key = int(round(ordv / 360.0 * max(6, 2 * self.n_planes)))
                if key not in best or rad > best[key][0]:
                    best[key] = (rad, P, ordv)
        if len(best) < 3:
            return None
        order = sorted(best.values(), key=lambda t: t[2])   # by meridian / θ
        return np.asarray([P for _r, P, _o in order])

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
        ax = self._axis_for(which)
        if ax is None:
            raise RuntimeError("set_axis() before adding contours")
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
    @staticmethod
    def _axis_dict(ax):
        return None if ax is None else {
            "origin": [float(x) for x in ax.apex],
            "axis": [float(x) for x in ax.axis],
            "radial0": [float(x) for x in ax.radial0],
        }

    def to_dict(self) -> dict:
        """Serialise the Endo/Epi axes + apexes + traced borders as 3-D volume-mm
        data — enough to fully re-apply them to the same series later."""
        return {
            "kind": "mdv-lvef", "version": 3,
            "n_planes": self.n_planes,
            "endo_axis": self._axis_dict(self.endo_axis),
            "epi_axis": self._axis_dict(self.epi_axis),
            "endo_apex": (None if self.endo_apex is None
                          else [float(x) for x in self.endo_apex]),
            "epi_apex": (None if self.epi_apex is None
                         else [float(x) for x in self.epi_apex]),
            "endo_planes": {f"{k:g}": np.asarray(v, float).reshape(-1, 3).tolist()
                            for k, v in self.endo_planes.items()},
            "epi_planes": {f"{k:g}": np.asarray(v, float).reshape(-1, 3).tolist()
                           for k, v in self.epi_planes.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LVModel":
        """Rebuild a model from to_dict() output. v3 has independent endo/epi
        axes; v1/v2 have a single 'axis' → use it for BOTH (fallback)."""
        m = cls(n_planes=int(d.get("n_planes", 6)))
        ea, pa = d.get("endo_axis"), d.get("epi_axis")
        if ea or pa:                                   # v3: independent axes
            if ea:
                m.set_axis_from_frame(ea["origin"], ea["axis"], ea["radial0"],
                                      which="endo")
            if pa:
                m.set_axis_from_frame(pa["origin"], pa["axis"], pa["radial0"],
                                      which="epi")
        elif d.get("axis"):                            # v1/v2: shared axis
            ax = d["axis"]
            m.set_axis_from_frame(ax["origin"], ax["axis"], ax["radial0"])
        for which, key in (("endo", "endo_planes"), ("epi", "epi_planes")):
            for k, pts in (d.get(key) or {}).items():
                arr = np.asarray(pts, float).reshape(-1, 3)
                if len(arr) >= 2:
                    m.set_long_axis_contour(float(k), arr, which=which)
        if d.get("endo_apex") is not None:
            m.set_apex_point("endo", d["endo_apex"])
        if d.get("epi_apex") is not None:
            m.set_apex_point("epi", d["epi_apex"])
        return m

    def _common_base(self):
        """Common basal cut for Endo & Epi as (endo_base_along, epi_base_along) in
        each surface's OWN axis, judged on the EPI (myocardial) axis: take the
        most-apical of the two surfaces' own most-basal common levels (so borders
        are present everywhere up to the cut), then express that one world plane
        on each axis. (None, None) if either surface can't build."""
        er = self.along_range("endo")
        pr = self.along_range("epi")
        if er is None or pr is None \
                or self.endo_axis is None or self.epi_axis is None:
            return None, None
        pap, pa = self.epi_axis.apex, self.epi_axis.axis
        eap, ea = self.endo_axis.apex, self.endo_axis.axis
        # endo's own basal plane, measured along the EPI axis
        endo_base_on_epi = float(((eap + er[1] * ea) - pap) @ pa)
        common_epi = min(endo_base_on_epi, pr[1])       # most-apical → common
        base_pt = pap + common_epi * pa                 # one world basal plane
        endo_base = float((base_pt - eap) @ ea)         # onto endo axis
        return endo_base, common_epi

    # ----------------------------------------------------------------- build
    def build(self, level_step: float = LV_LEVEL_STEP_MM,
              n_theta: int = LV_RING_POINTS) -> None:
        """(Re)build the endo/epi ring stacks — each from ITS OWN axis + stored
        meridian contours. When both exist, cut them at a COMMON basal level
        judged on the epi axis. Needs ≥3 meridians for a surface."""
        endo_ok = self.endo_axis is not None and len(self.endo_contours) >= 3
        epi_ok = self.epi_axis is not None and len(self.epi_contours) >= 3
        endo_base = epi_base = None
        if endo_ok and epi_ok:
            endo_base, epi_base = self._common_base()
        self.endo = (LVSurface.from_meridian_contours(
            self.endo_axis, self.endo_contours, level_step, n_theta,
            base_along=endo_base) if endo_ok else None)
        self.epi = (LVSurface.from_meridian_contours(
            self.epi_axis, self.epi_contours, level_step, n_theta,
            base_along=epi_base) if epi_ok else None)
        # converge each built surface to its user-defined apex vertex (if set)
        if self.endo is not None:
            self.endo.apex_world = self.endo_apex
        if self.epi is not None:
            self.epi.apex_world = self.epi_apex

    # -------------------------------------------------- promote endo → epi axis
    def promote_endo_to_epi_axis(self, level_step: float = LV_LEVEL_STEP_MM,
                                 n_theta: int = LV_RING_POINTS) -> bool:
        """Re-express the traced Endo border on the EPI axis so both borders
        share ONE coordinate frame (common axis + meridians) — the prerequisite
        for editing Endo against Epi on one long-axis plane and for true
        per-ray wall thickness.

        Endo & Epi are traced on INDEPENDENT axes (each through its own apex).
        Here Endo is 'promoted' onto the Epi axis:
          1. build the Endo surface on its OWN axis (rings faithful to the trace),
          2. reslice those rings along each EPI meridian plane → the Endo border
             curve lying in that plane,
          3. store it as endo_planes/endo_contours on the EPI axis
             (``endo_axis := epi_axis``), so from then on Endo & Epi are
             symmetric and every edit / SAX / build / STL path treats them
             identically.

        The Endo APEX point is KEPT unchanged as an OFF-AXIS apex the
        reconstruction still converges to (build() sets
        ``endo.apex_world = endo_apex`` regardless of axis), so the Endo apex
        accuracy is preserved even though the Epi axis misses it, and apical wall
        thickness = the gap between the two apex points.

        Returns True on success, False (no change) if either border isn't traced
        or the reslice found too few crossings. Idempotent: returns True without
        change if Endo already shares the Epi axis."""
        if (self.epi_axis is None or self.endo_axis is None
                or len(self.endo_contours) < 3 or len(self.epi_contours) < 3):
            return False
        if self.endo_axis is self.epi_axis:
            return True                                    # already promoted
        # 1. Endo surface on its OWN axis (basal cut judged commonly on epi axis).
        endo_base, _epi_base = self._common_base()
        surf = LVSurface.from_meridian_contours(
            self.endo_axis, self.endo_contours, level_step, n_theta,
            base_along=endo_base)
        if surf is None:
            return False
        rings = [surf.ring_world(k) for k in range(len(surf.along))]

        ax = self.epi_axis
        # 2. Reslice the endo rings along each EPI meridian plane.
        new_planes: dict[float, np.ndarray] = {}
        for phi in self.plane_angles():
            _o, e_s, _e_t, nrm = ax.long_axis_basis(phi)   # nrm ⟂ meridian plane
            pos, neg = [], []                              # +e_s / -e_s sides
            for ring in rings:
                b = (ring - ax.apex) @ nrm                 # signed dist to plane
                m = len(ring)
                for i in range(m):
                    j = (i + 1) % m
                    b0, b1 = float(b[i]), float(b[j])
                    if b0 == b1:
                        continue
                    if (b0 <= 0.0 <= b1) or (b1 <= 0.0 <= b0):
                        t = b0 / (b0 - b1)
                        P = ring[i] + t * (ring[j] - ring[i])
                        dp = P - ax.apex
                        along = float(dp @ ax.axis)
                        s = float(dp @ e_s)
                        (pos if s >= 0.0 else neg).append((along, abs(s), P))
            pos = _outermost_by_level(pos, level_step)
            neg = _outermost_by_level(neg, level_step)
            if len(pos) + len(neg) < 3:
                continue
            neg.sort(key=lambda r: -r[0])                  # base → apex
            pos.sort(key=lambda r: r[0])                   # apex → base
            pts = [P for _a, _r, P in neg] + [P for _a, _r, P in pos]
            new_planes[phi] = np.asarray(pts, float)
        if len(new_planes) < 3:
            return False
        # 3. Commit: Endo now lives on the Epi axis; keep the Endo apex point.
        self.endo_axis = ax
        self.endo_contours = {}
        self.endo_planes = {}
        for phi, pts in new_planes.items():
            self.set_long_axis_contour(phi, pts, "endo")
        return True

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
