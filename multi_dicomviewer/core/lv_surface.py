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


def _gauss_kernel(sigma: float) -> np.ndarray:
    r = int(max(1, round(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def _smooth_ring_stack(rings, ang_sigma: float = 5.0, lon_sigma: float = 3.0,
                       apex_frac: float = 0.4):
    """Gaussian smoothing of a (K, Nθ, 2) ring stack for a nicer DISPLAY mesh —
    applied ONLY near the apex, where sparse meridians make the rings POLYGONAL
    (a hexagon from 6 meridians) and the changing meridian set per level makes
    the loft 'spiral'. The smoothing is blended in with a per-level weight that
    is 1 at the apex and fades to 0 over the apical *apex_frac* of the length,
    so the BODY (the real 12-meridian shape) and the basal rim are kept exactly.
    Purely cosmetic — the volume is measured from the ORIGINAL rings, not this."""
    R = np.asarray(rings, float)
    K = len(R)
    if K < 3:
        return R.copy()
    ka = _gauss_kernel(ang_sigma)                  # periodic (angular)
    ra = len(ka) // 2
    A = np.zeros_like(R)
    for i, w in enumerate(ka):
        A += w * np.roll(R, i - ra, axis=1)
    kl = _gauss_kernel(lon_sigma)                  # edge-clamped (longitudinal)
    rl = len(kl) // 2
    idx = np.arange(K)
    S = np.zeros_like(A)
    for i, w in enumerate(kl):
        S += w * A[np.clip(idx + (i - rl), 0, K - 1)]
    # per-level blend weight: 1 at the apex (level 0) → 0 by apex_frac up.
    t = idx / max(1, K - 1)
    wv = np.clip(1.0 - t / max(1e-6, apex_frac), 0.0, 1.0)
    wv = wv * wv * (3.0 - 2.0 * wv)                # smoothstep
    wv = wv[:, None, None]
    out = wv * S + (1.0 - wv) * R
    out[-1] = R[-1]                                # keep the basal rim crisp
    return out


def _apex_tip(axis, along, rings):
    """A point on the axis a SHORT hemisphere-height (≈ the apical-ring radius)
    below the apical ring, so the apex closes as a small rounded cap — NOT a
    long spike (extrapolating the shallow taper to radius 0 overshot into a
    pointed spike) and not a flat nub."""
    k = len(along)
    step = float(along[1] - along[0]) if k >= 2 else 1.0
    r0 = float(np.hypot(rings[0, :, 0], rings[0, :, 1]).mean())
    ext = float(max(r0, 0.6 * step))
    return axis.apex + (float(along[0]) - ext) * axis.axis


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
    reach: np.ndarray = None       # (K,) #meridians defining each level (or None)

    def _reliable_start(self) -> int:
        """Most-apical ring index to KEEP for the display mesh. Below it the
        wall has tapered to a thin, sparsely-sampled 'neck' that reads as a spike
        / nub; the mesh instead caps the apex from HERE with a rounded dome. The
        cut is where the (smoothed) mean ring radius first reaches a MODERATE
        target (~a quarter of the widest ring, 3.5–6 mm), and ≥3 meridians define
        the ring — so the dome starts from a sensibly-broad, reliable level. 0 if
        unknown / already broad everywhere."""
        k = len(self.along)
        if k < 3:
            return 0
        mr = np.hypot(self.rings[:, :, 0], self.rings[:, :, 1]).mean(axis=1)
        sm = mr.copy()                                 # light smoothing
        sm[1:-1] = 0.25 * mr[:-2] + 0.5 * mr[1:-1] + 0.25 * mr[2:]
        rt = float(np.clip(0.25 * float(sm.max()), 3.5, 6.0))
        ok = sm >= rt
        r = getattr(self, "reach", None)
        if r is not None and len(r) == k:
            ok = ok & (np.asarray(r) >= 3)
        idx = np.nonzero(ok)[0]
        # keep at least the basal two-thirds so a badly-sampled case can't gut it
        return int(min(idx[0], (k - 1) // 3)) if len(idx) else 0

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
        # along-range: the APEX extends to the DEEPEST traced point of ANY
        # meridian (min of minima), so a short trace on one wall no longer clips
        # the whole surface off partway. Each level is then rebuilt from ONLY the
        # meridians that actually REACH it (angular interpolation across the
        # gaps) — no vertical extrapolation, so an aneurysmal / atypical apex is
        # taken from the real traces, not invented. The BASE stays the common
        # flat cut ⟂ the axis at the most-basal level common to all borders.
        mins, maxs = [], []
        for th in thetas:
            a = np.asarray(contours[th], float).reshape(-1, 2)[:, 0]
            mins.append(float(a.min()))
            maxs.append(float(a.max()))
        apex_along, base_along = min(mins), min(maxs)
        if base_along <= apex_along:
            raise ValueError("border along-ranges do not overlap")
        k = max(2, int(round((base_along - apex_along)
                             / max(1e-3, level_step))) + 1)
        along = np.linspace(apex_along, base_along, k)
        # radius of each meridian at each level, or NaN where the level is
        # OUTSIDE that meridian's own traced range (so it doesn't contribute).
        r_km = np.full((k, len(thetas)), np.nan)
        for j, th in enumerate(thetas):
            c = np.asarray(contours[th], float).reshape(-1, 2)
            cc = c[np.argsort(c[:, 0])]
            lo, hi = float(cc[0, 0]), float(cc[-1, 0])
            m = (along >= lo - 1e-6) & (along <= hi + 1e-6)
            r_km[m, j] = np.interp(along[m], cc[:, 0], cc[:, 1])
        # per level, resample radii around θ using ONLY the meridians present at
        # that level, then → 2-D pts. Deep apical levels are built from the few
        # meridians that reach there; the basal levels from all of them.
        out_ang = np.radians(np.linspace(0.0, 360.0, n_theta, endpoint=False))
        cos_t, sin_t = np.cos(out_ang), np.sin(out_ang)
        rings = np.zeros((k, n_theta, 2))
        reach = np.zeros(k, dtype=int)             # #meridians defining each level
        for ki in range(k):
            valid = ~np.isnan(r_km[ki])
            nv = int(valid.sum())
            reach[ki] = nv
            if nv >= 3:
                r_theta = _resample_periodic(thetas[valid], r_km[ki, valid],
                                             n_theta)
            elif nv >= 1:                          # too few for a ring → a disc
                r_theta = np.full(n_theta, float(np.nanmean(r_km[ki, valid])))
            else:
                r_theta = np.zeros(n_theta)
            r_theta = np.clip(r_theta, 0.0, None)
            rings[ki, :, 0] = r_theta * cos_t
            rings[ki, :, 1] = r_theta * sin_t
        surf = cls(axis=axis, along=along, rings=rings)
        surf.reach = reach
        return surf

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
    def _rings_world(self, rings, along=None):
        """World points ((len(rings))*Nθ, 3) for an explicit ring stack (may be a
        smoothed / trimmed copy) at levels *along* (defaults to self.along)."""
        ax = self.axis
        aa = self.along if along is None else along
        out = []
        for i in range(len(rings)):
            o = ax.apex + float(aa[i]) * ax.axis
            xy = rings[i]
            out.append(o + np.outer(xy[:, 0], ax.radial0)
                       + np.outer(xy[:, 1], ax.binormal))
        return np.concatenate(out, 0)

    def to_mesh(self, close_base: bool = True, smooth: bool = True):
        """Triangulated surface (vertices (V,3), faces (F,3) int). Rings lofted
        with quad strips; the apex is closed with a rounded DOME.

        *close_base* True → also fan a flat base cap, giving a CLOSED solid (for
        voxelisation). False → leave the basal rim OPEN: a thin single-wall
        'cup'/bowl. *smooth* trims the unreliable apical 'neck' the 1–2 apical
        meridians produce and rounds the apex from the deepest well-sampled ring,
        and lightly smooths the rings — all DISPLAY-only; the measured volume
        uses the raw rings and is UNAFFECTED."""
        _, nth, _ = self.rings.shape
        ax = self.axis
        rings_full = _smooth_ring_stack(self.rings) if smooth else self.rings
        ts = self._reliable_start() if smooth else 0
        rings = rings_full[ts:]                     # wall = well-sampled levels
        along = self.along[ts:]
        km = len(rings)
        ring_pts = self._rings_world(rings, along)
        verts_list = [ring_pts]
        faces = []
        for i in range(km - 1):                     # loft the wall (outward)
            a0 = i * nth
            a1 = (i + 1) * nth
            for j in range(nth):
                jn = (j + 1) % nth
                faces.append([a0 + j, a1 + jn, a1 + j])
                faces.append([a0 + j, a0 + jn, a1 + jn])
        nxt = km * nth
        if close_base:
            base = (ax.apex + float(along[-1]) * ax.axis).reshape(1, 3)
            base_i = nxt
            verts_list.append(base)
            nxt += 1
            base0 = (km - 1) * nth
            for j in range(nth):                   # flat base cap → solid
                jn = (j + 1) % nth
                faces.append([base_i, base0 + j, base0 + jn])
        # Rounded APEX DOME below the deepest RELIABLE ring: rings shrink along a
        # quarter-ellipse (height ≈ that ring's radius) to a tip, so the bottom
        # follows the wall's curve into a smooth rounded apex — no thin neck, no
        # spike, no nub.
        r0 = float(np.hypot(rings[0, :, 0], rings[0, :, 1]).mean())
        step = float(along[1] - along[0]) if km >= 2 else 1.0
        h = max(r0, 0.6 * step)                    # dome height (apical)
        c0 = ax.apex + float(along[0]) * ax.axis   # reliable-ring plane centre
        P0 = ring_pts[:nth]
        n_dome = 7
        prev = 0
        for s in range(1, n_dome):
            u = s / float(n_dome)
            scale = float(np.cos(u * np.pi / 2.0))
            drop = h * float(np.sin(u * np.pi / 2.0))
            off = nxt
            verts_list.append((c0 - drop * ax.axis) + scale * (P0 - c0))
            nxt += nth
            for j in range(nth):                   # loft ring→dome (apical dir)
                jn = (j + 1) % nth
                faces.append([off + j, prev + jn, prev + j])
                faces.append([off + j, off + jn, prev + jn])
            prev = off
        tip_i = nxt
        verts_list.append((c0 - h * ax.axis).reshape(1, 3))
        nxt += 1
        for j in range(nth):                       # fan last dome ring → tip
            jn = (j + 1) % nth
            faces.append([tip_i, prev + jn, prev + j])
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
    _, nth, _ = inner.rings.shape
    if inner.rings.shape[0] < 2 or outer.rings.shape[0] < 2 \
            or outer.rings.shape[1] != nth:
        return None
    ax = outer.axis
    irings = _smooth_ring_stack(inner.rings)       # cosmetic (volume unaffected)
    orings = _smooth_ring_stack(outer.rings)
    its = inner._reliable_start()
    ots = outer._reliable_start()
    iw_rings, iw_along = irings[its:], inner.along[its:]
    ow_rings, ow_along = orings[ots:], outer.along[ots:]
    kmi, kmo = len(iw_rings), len(ow_rings)
    iw = inner._rings_world(iw_rings, iw_along)    # kmi*nth  (reliable→base)
    ow = outer._rings_world(ow_rings, ow_along)    # kmo*nth
    # Rounded apex domes from each surface's deepest RELIABLE ring, meeting at a
    # shared tip (so the myocardium closes to a smooth rounded point — no thin
    # neck / spike). Tip = the more apical of the two hemisphere tips.
    Ri = float(np.hypot(iw_rings[0, :, 0], iw_rings[0, :, 1]).mean())
    Ro = float(np.hypot(ow_rings[0, :, 0], ow_rings[0, :, 1]).mean())
    ci_a, co_a = float(iw_along[0]), float(ow_along[0])
    tip_a = min(ci_a - max(Ri, 0.6), co_a - max(Ro, 0.6))
    tip = (ax.apex + tip_a * ax.axis).reshape(1, 3)
    n_dome = 7
    c0i, c0o = ax.apex + ci_a * ax.axis, ax.apex + co_a * ax.axis
    P0i, P0o = iw[:nth], ow[:nth]
    hi, ho = ci_a - tip_a, co_a - tip_a
    idome, odome = [], []
    for s in range(1, n_dome):
        u = s / float(n_dome)
        sc, sn = float(np.cos(u * np.pi / 2)), float(np.sin(u * np.pi / 2))
        idome.append((c0i - hi * sn * ax.axis) + sc * (P0i - c0i))
        odome.append((c0o - ho * sn * ax.axis) + sc * (P0o - c0o))
    verts = np.concatenate(
        [iw, np.concatenate(idome, 0), ow, np.concatenate(odome, 0), tip], 0)
    IB = 0
    IDB = kmi * nth
    OB = IDB + (n_dome - 1) * nth
    ODB = OB + kmo * nth
    TIP = ODB + (n_dome - 1) * nth
    faces = []
    # INNER wall (normals into the cavity), reliable→base.
    for i in range(kmi - 1):
        a0, a1 = IB + i * nth, IB + (i + 1) * nth
        for j in range(nth):
            jn = (j + 1) % nth
            faces.append([a0 + j, a1 + j, a1 + jn])
            faces.append([a0 + j, a1 + jn, a0 + jn])
    prev = IB                                       # inner apex dome (inward)
    for d in range(n_dome - 1):
        off = IDB + d * nth
        for j in range(nth):
            jn = (j + 1) % nth
            faces.append([off + j, prev + j, prev + jn])
            faces.append([off + j, prev + jn, off + jn])
        prev = off
    for j in range(nth):                            # inner tip fan (inward)
        jn = (j + 1) % nth
        faces.append([TIP, prev + j, prev + jn])
    # OUTER wall (normals outward), reliable→base.
    for i in range(kmo - 1):
        a0, a1 = OB + i * nth, OB + (i + 1) * nth
        for j in range(nth):
            jn = (j + 1) % nth
            faces.append([a0 + j, a1 + jn, a1 + j])
            faces.append([a0 + j, a0 + jn, a1 + jn])
    prev = OB                                       # outer apex dome (outward)
    for d in range(n_dome - 1):
        off = ODB + d * nth
        for j in range(nth):
            jn = (j + 1) % nth
            faces.append([off + j, prev + jn, prev + j])
            faces.append([off + j, off + jn, prev + jn])
        prev = off
    for j in range(nth):                            # outer tip fan (outward)
        jn = (j + 1) % nth
        faces.append([TIP, prev + jn, prev + j])
    # BASE RIM: annulus joining the inner base ring to the outer base ring.
    ibase, obase = IB + (kmi - 1) * nth, OB + (kmo - 1) * nth
    for j in range(nth):
        jn = (j + 1) % nth
        faces.append([ibase + j, obase + j, obase + jn])
        faces.append([ibase + j, obase + jn, ibase + jn])
    return verts, np.asarray(faces, dtype=np.int64)
