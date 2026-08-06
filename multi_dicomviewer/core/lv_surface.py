"""LV endo/epi boundary surface from rotated long-axis contours → voxel volume.

GUI-free (pure numpy, no VTK / Qt / scipy) so it can be unit-tested headless
and shared by both CT viewers. Builds on [[lv_axis]].

Pipeline the viewers drive:

    per-meridian long-axis contours              # drawn on the rotated long-axis
    {θ: [(along, radius), …]}                     #   planes (4×45° or 6×30°)
        │  LVSurface.from_meridian_contours()
        ▼
    short-axis contour stack                      # one closed ring per axial
    rings : (K, Nθ, 2) in (radial0, binormal)     #   level, spline-smoothed
        │  .voxel_volume_ml(spacing)              #   round θ (single/long-axis
        ▼                                         #   corrections edit the rings)
    count CT voxels whose centre lies inside the closed endo surface
    × voxel volume  →  LV cavity volume (papillary muscle / trabeculae INCLUDED)

The rings are stored as explicit closed polygons (not a radius-only field) so a
later editing pass can push a ring into a non-star-shaped outline without
changing the volume engine — voxel counting uses a general point-in-polygon
test per axial level, so it is correct for any (convex or concave) ring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lv_axis import LVAxis

#: default axial spacing (mm) of the short-axis ring stack
LV_LEVEL_STEP_MM = 0.5
#: default number of points around each short-axis ring
LV_RING_POINTS = 120


def _resample_periodic(theta_src, r_src, n_out: int) -> np.ndarray:
    """Interpolate radii *r_src* sampled at angles *theta_src* (deg, any order)
    onto *n_out* uniform angles in [0, 360), wrapping periodically. Returns the
    (n_out,) radii."""
    th = np.asarray(theta_src, float) % 360.0
    r = np.asarray(r_src, float)
    order = np.argsort(th)
    th, r = th[order], r[order]
    # wrap one period on each end so np.interp is continuous across 0/360.
    th_ext = np.concatenate([th - 360.0, th, th + 360.0])
    r_ext = np.concatenate([r, r, r])
    out_ang = np.linspace(0.0, 360.0, n_out, endpoint=False)
    return np.interp(out_ang, th_ext, r_ext)


def _points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised crossing-number point-in-polygon. *pts* (P,2), *poly* (V,2)
    closed implicitly (last→first). Returns (P,) bool. Works for concave rings."""
    x = pts[:, 0]
    y = pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi)
        inside ^= cond
        j = i
    return inside


@dataclass
class LVSurface:
    """A stack of short-axis rings describing one LV boundary (endo or epi).

    axis  : the LVAxis this surface is built on
    along : (K,) axial level positions, apex (0) → base (length_mm)
    rings : (K, Nθ, 2) closed polygons in the short-axis (radial0, binormal)
            2-D basis, one per level
    """
    axis: LVAxis
    along: np.ndarray
    rings: np.ndarray

    # ------------------------------------------------------------------ build
    @classmethod
    def from_meridian_contours(cls, axis: LVAxis, contours: dict,
                               level_step: float = LV_LEVEL_STEP_MM,
                               n_theta: int = LV_RING_POINTS) -> "LVSurface":
        """Build the ring stack from per-meridian long-axis contours.

        *contours* maps a meridian angle θ (deg) → an (T,2) array of
        (along, radius) samples for that meridian (apex→base). A long-axis
        plane at rotation φ contributes TWO meridians: φ (the +radial wall) and
        φ+180 (the −radial wall). Provide them already split by wall.
        """
        if len(contours) < 3:
            raise ValueError("need at least 3 meridians to build a surface")
        thetas = np.array(sorted(contours.keys()), dtype=float)
        # along-range = the span where EVERY meridian has a point (no
        # extrapolation): apex = max of per-meridian minima, base = min of
        # per-meridian maxima. The base is thus a plane ⟂ the axis at the
        # most-basal level common to all borders — editing a border's basal end
        # moves it. (The axis's own apex/length are not used here: the axis may
        # come from the current view, with the extent defined by the borders.)
        mins, maxs = [], []
        for th in thetas:
            a = np.asarray(contours[th], float).reshape(-1, 2)[:, 0]
            mins.append(float(a.min()))
            maxs.append(float(a.max()))
        apex_along, base_along = max(mins), min(maxs)
        if base_along <= apex_along:
            raise ValueError("border along-ranges do not overlap")
        k = max(2, int(round((base_along - apex_along)
                             / max(1e-3, level_step))) + 1)
        along = np.linspace(apex_along, base_along, k)
        # radius of each meridian interpolated onto the common axial levels.
        r_km = np.zeros((k, len(thetas)))
        for j, th in enumerate(thetas):
            c = np.asarray(contours[th], float).reshape(-1, 2)
            cc = c[np.argsort(c[:, 0])]
            r_km[:, j] = np.interp(along, cc[:, 0], cc[:, 1],
                                   left=cc[0, 1], right=cc[-1, 1])
        # per level, resample radii around θ to a uniform ring, then → 2-D pts.
        out_ang = np.radians(np.linspace(0.0, 360.0, n_theta, endpoint=False))
        cos_t, sin_t = np.cos(out_ang), np.sin(out_ang)
        rings = np.zeros((k, n_theta, 2))
        for ki in range(k):
            r_theta = _resample_periodic(thetas, r_km[ki], n_theta)
            r_theta = np.clip(r_theta, 0.0, None)
            rings[ki, :, 0] = r_theta * cos_t
            rings[ki, :, 1] = r_theta * sin_t
        return cls(axis=axis, along=along, rings=rings)

    # ---------------------------------------------------------------- accessors
    @property
    def n_levels(self) -> int:
        return len(self.along)

    def ring_world(self, k: int) -> np.ndarray:
        """(Nθ,3) world points of ring *k* (for drawing the short-axis outline
        in 3-D / on a pane)."""
        o = self.axis.apex + float(self.along[k]) * self.axis.axis
        xy = self.rings[k]
        return (o + np.outer(xy[:, 0], self.axis.radial0)
                + np.outer(xy[:, 1], self.axis.binormal))

    def long_axis_profile(self, theta_deg: float) -> np.ndarray:
        """(K,2) array of (along, radius) along one meridian — the outline seen
        on the long-axis cross-section at rotation *theta_deg* (its +radial
        wall). Sampled from the rings."""
        ang = np.radians(theta_deg % 360.0)
        d = np.array([np.cos(ang), np.sin(ang)])
        # radius at this angle per level = ring point nearest that direction.
        idx = int(round((theta_deg % 360.0) / 360.0 * self.rings.shape[1])
                  ) % self.rings.shape[1]
        r = np.hypot(self.rings[:, idx, 0], self.rings[:, idx, 1])
        _ = d
        return np.column_stack([self.along, r])

    def set_ring(self, k: int, poly_xy: np.ndarray) -> None:
        """Replace ring *k* with an explicit closed polygon (Nθ,2) in the
        short-axis basis — the short-axis-correction edit path."""
        poly = np.asarray(poly_xy, float).reshape(-1, 2)
        if poly.shape != self.rings[k].shape:
            raise ValueError("ring vertex count must match")
        self.rings[k] = poly

    # ------------------------------------------------------------------- volume
    def voxel_volume_ml(self, spacing: float, *, return_count: bool = False):
        """LV volume by counting the voxels of an axis-aligned grid (voxel size
        *spacing* mm, isotropic) whose centres fall inside the closed surface.
        Returns millilitres (mm³ / 1000), or (ml, voxel_count) if requested.

        Voxels are classified per axial level with a general point-in-polygon
        test, so concave rings are handled correctly. The base is a flat cut at
        along = length_mm; the apex closes the bottom."""
        verts = self._all_ring_points()
        lo = verts.min(axis=0) - spacing
        hi = verts.max(axis=0) + spacing
        gx = np.arange(lo[0], hi[0] + spacing, spacing)
        gy = np.arange(lo[1], hi[1] + spacing, spacing)
        gz = np.arange(lo[2], hi[2] + spacing, spacing)
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        along, px, py = self.axis.project_array(pts)
        lo_a = float(self.along[0])
        hi_a = float(self.along[-1])
        step = (hi_a - lo_a) / (self.n_levels - 1)
        inside = np.zeros(len(pts), dtype=bool)
        in_range = (along >= lo_a) & (along <= hi_a)
        lvl = np.clip(np.round((along - lo_a) / max(1e-9, step)).astype(int),
                      0, self.n_levels - 1)
        sa = np.column_stack([px, py])
        for k in range(self.n_levels):
            sel = in_range & (lvl == k)
            if not np.any(sel):
                continue
            inside[sel] = _points_in_polygon(sa[sel], self.rings[k])
        count = int(np.count_nonzero(inside))
        ml = count * (spacing ** 3) / 1000.0
        return (ml, count) if return_count else ml

    # ---------------------------------------------------------------- meshing
    def to_mesh(self, close_base: bool = True):
        """Triangulated surface (vertices (V,3), faces (F,3) int). Rings lofted
        with quad strips; the apex is fanned to a point (rounded bottom).

        *close_base* True → also fan a flat base cap, giving a CLOSED solid (for
        voxelisation). False → leave the basal rim OPEN: a thin single-wall
        'cup'/bowl (wall + apex only, no fill), which is what the Endo/Epi STL
        export wants (a filled solid is not what a surface should look like)."""
        k, nth, _ = self.rings.shape
        ring_pts = np.concatenate([self.ring_world(i) for i in range(k)], 0)
        apex = self.axis.apex.reshape(1, 3)
        verts_list = [ring_pts, apex]
        apex_i = k * nth
        faces = []
        for i in range(k - 1):                     # loft between rings (outward)
            a0 = i * nth
            a1 = (i + 1) * nth
            for j in range(nth):
                jn = (j + 1) % nth
                faces.append([a0 + j, a1 + jn, a1 + j])
                faces.append([a0 + j, a0 + jn, a1 + jn])
        for j in range(nth):                       # apex cap (rounded bottom)
            jn = (j + 1) % nth
            faces.append([apex_i, jn, j])
        if close_base:
            base = self.axis.base_center.reshape(1, 3)
            base_i = apex_i + 1
            verts_list.append(base)
            base0 = (k - 1) * nth
            for j in range(nth):                   # base cap (flat) → solid
                jn = (j + 1) % nth
                faces.append([base_i, base0 + j, base0 + jn])
        verts = np.concatenate(verts_list, 0)
        return verts, np.asarray(faces, dtype=np.int64)

    # ------------------------------------------------------------------ private
    def _all_ring_points(self) -> np.ndarray:
        return np.concatenate([self.ring_world(i)
                               for i in range(self.n_levels)], 0)


def myocardial_shell_mesh(inner: "LVSurface", outer: "LVSurface"):
    """Watertight myocardial 'cup' between the INNER (endo) and OUTER (epi)
    surfaces — the wall has the real myocardial thickness. Built as: outer wall
    + apex cap (normals out), inner wall + apex cap (normals in), joined at the
    base by an annular rim so the cavity stays open at the base (you can see
    into the cup) while the solid is closed everywhere else. Both surfaces share
    the same axis / ring-point count. Vertices in world mm."""
    ki, nth, _ = inner.rings.shape
    ko = outer.rings.shape[0]
    if ki < 2 or ko < 2 or outer.rings.shape[1] != nth:
        # Degenerate → fall back to the two open cups side by side.
        return None
    iring = np.concatenate([inner.ring_world(i) for i in range(ki)], 0)
    oring = np.concatenate([outer.ring_world(i) for i in range(ko)], 0)
    apex = inner.axis.apex.reshape(1, 3)           # shared apex point
    verts = np.concatenate([iring, oring, apex], 0)
    ib0, ob0, ap = 0, ki * nth, ki * nth + ko * nth
    faces = []
    # INNER (endo) wall — normals face the CAVITY (radially inward = out of the
    # myocardial solid on the cavity side).
    for i in range(ki - 1):
        a0 = ib0 + i * nth
        a1 = ib0 + (i + 1) * nth
        for j in range(nth):
            jn = (j + 1) % nth
            faces.append([a0 + j, a1 + j, a1 + jn])
            faces.append([a0 + j, a1 + jn, a0 + jn])
    for j in range(nth):                           # inner apex cap (into cavity)
        jn = (j + 1) % nth
        faces.append([ap, ib0 + j, ib0 + jn])
    # OUTER (epi) wall — normals point OUTWARD.
    for i in range(ko - 1):
        a0 = ob0 + i * nth
        a1 = ob0 + (i + 1) * nth
        for j in range(nth):
            jn = (j + 1) % nth
            faces.append([a0 + j, a1 + jn, a1 + j])
            faces.append([a0 + j, a0 + jn, a1 + jn])
    for j in range(nth):                           # outer apex cap (outward)
        jn = (j + 1) % nth
        faces.append([ap, ob0 + jn, ob0 + j])
    # BASE RIM: annulus joining the endo base ring to the epi base ring, closing
    # the wall thickness at the top (exterior normal points basally, +axis) while
    # the cavity mouth stays open.
    ibase = ib0 + (ki - 1) * nth
    obase = ob0 + (ko - 1) * nth
    for j in range(nth):
        jn = (j + 1) % nth
        faces.append([ibase + j, obase + j, obase + jn])
        faces.append([ibase + j, obase + jn, ibase + jn])
    return verts, np.asarray(faces, dtype=np.int64)
