"""CAG→CPR core — reconstruct a 3-D coronary centreline from points drawn on
one or more angiographic views, snapped to the CT contrast lumen.

Workflow this supports (see also [[coaxial-evaluation-plan]]):

    On each angiographic view (a real CAG or a virtual CAG = a CT MIP at a
    C-arm-standard angle) the user marks the vessel centreline per segment
    (LM / LAD / D1 …) and, to fix the depth, a FRONT point (表点, near) and/or a
    BACK point (裏点, far). Each 2-D point back-projects to a ray in CT world
    space; the 3-D centreline point is where that ray meets the contrast lumen
    in the CT (HU ~200-1000) on the chosen (near/far) side. Several views can be
    combined by triangulation (closest point to the rays). Where there is no CT
    contrast (CTO / heavy calcium) the point falls back to a geometric prior
    (e.g. the Epi surface) handled by the caller.

Pure numpy — no Qt / VTK — so it is headless-testable and shared by both CT
viewers. Volumes are indexed ``vol[z, y, x]`` with ``dims = (sx, sy, sz)`` mm,
world = voxel · spacing (origin 0), matching ``core.lv_epi_autocontour``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from multi_dicomviewer.core.lv_epi_autocontour import sample_hu


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


@dataclass
class ProjectionView:
    """A projection geometry mapping CT world (mm) ↔ a 2-D view.

    *center* is the world point imaged at view coords (0, 0); *u*, *v* are unit
    world directions of the view's +x / +y (so a view point ``(x, y)`` in mm maps
    to ``center + x·u + y·v`` on the image plane); *normal* is the unit viewing
    direction (into the volume).

    - **Parallel** (``source is None``; a MIP / virtual CAG): the back-projection
      of ``(x, y)`` is the line through ``center + x·u + y·v`` along *normal*.
    - **Perspective** (``source`` = the X-ray focal spot; a real cone-beam CAG):
      ``(x, y)`` is a point on the detector plane and the back-projection is the
      ray from *source* through it.
    """
    center: np.ndarray
    u: np.ndarray
    v: np.ndarray
    normal: np.ndarray
    source: np.ndarray | None = None

    def __post_init__(self):
        self.center = np.asarray(self.center, float).reshape(3)
        self.u = _unit(self.u)
        self.v = _unit(self.v)
        self.normal = _unit(self.normal)
        if self.source is not None:
            self.source = np.asarray(self.source, float).reshape(3)

    def backproject(self, x, y):
        """View point (x, y) → a 3-D ray ``(origin, unit_dir)`` in world mm."""
        p = self.center + float(x) * self.u + float(y) * self.v
        if self.source is None:
            return p, self.normal.copy()
        return self.source.copy(), _unit(p - self.source)

    def project(self, P):
        """World point *P* → view coords (x, y) in mm."""
        P = np.asarray(P, float).reshape(3)
        if self.source is None:
            d = P - self.center
            return float(d @ self.u), float(d @ self.v)
        o = self.source
        dirv = P - o
        denom = float(dirv @ self.normal)
        if abs(denom) < 1e-9:
            return float("nan"), float("nan")
        t = float((self.center - o) @ self.normal) / denom
        d = (o + t * dirv) - self.center
        return float(d @ self.u), float(d @ self.v)


def triangulate_rays(rays):
    """Least-squares 3-D point closest to every ray ``(origin, dir)`` (dir need
    not be unit). Returns the point, or None if the rays are (near) parallel."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    n = 0
    for o, d in rays:
        o = np.asarray(o, float).reshape(3)
        d = _unit(d)
        proj = np.eye(3) - np.outer(d, d)     # projector ⟂ to the ray
        A += proj
        b += proj @ o
        n += 1
    if n < 2 or np.linalg.matrix_rank(A, tol=1e-6) < 3:
        return None
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def _contrast_clusters(ts, mask):
    """Contiguous runs of True in *mask* → list of (t_centre, i0, i1) by *ts*."""
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((0.5 * (ts[i] + ts[j]), i, j))
            i = j + 1
        else:
            i += 1
    return out


def snap_to_contrast(vol, dims, ray_o, ray_dir, t_range=(-150.0, 150.0),
                     step=0.5, hu_lo=200.0, hu_hi=1000.0, radius_mm=4.0,
                     prefer="near"):
    """March the ray ``ray_o + t·ray_dir`` over *t_range* mm; find where the CT is
    in the contrast window ``[hu_lo, hu_hi]``; pick the NEAR (``prefer='near'``,
    smallest t) or FAR (``'far'``) contrast cluster, and return its 3-D centre,
    laterally refined to the contrast centroid within *radius_mm* of the ray.
    Returns ``(point, t)`` or ``(None, None)`` when no contrast is on the ray."""
    ray_o = np.asarray(ray_o, float).reshape(3)
    ray_dir = _unit(ray_dir)
    ts = np.arange(float(t_range[0]), float(t_range[1]) + 1e-6, float(step))
    if len(ts) < 2:
        return None, None
    pts = ray_o[None, :] + ts[:, None] * ray_dir[None, :]
    hu = sample_hu(vol, dims, pts, outside=-1000.0)
    mask = (hu >= hu_lo) & (hu <= hu_hi)
    clusters = _contrast_clusters(ts, mask)
    if not clusters:
        return None, None
    tc, i0, i1 = clusters[0] if prefer == "near" else clusters[-1]
    # On-ray centre weighted by contrast; then a lateral centroid refine on the
    # short-axis disc of the ray (so the point sits at the lumen centre, not
    # wherever the ray happened to clip the vessel).
    seg_t = ts[i0:i1 + 1]
    seg_w = np.clip(hu[i0:i1 + 1], hu_lo, hu_hi) - hu_lo + 1.0
    t_star = float(np.sum(seg_t * seg_w) / np.sum(seg_w))
    centre = ray_o + t_star * ray_dir
    e1 = _unit(np.cross(ray_dir, [0.0, 0.0, 1.0]))
    if float(np.linalg.norm(e1)) < 1e-6:
        e1 = _unit(np.cross(ray_dir, [0.0, 1.0, 0.0]))
    e2 = _unit(np.cross(ray_dir, e1))
    gs = np.arange(-radius_mm, radius_mm + 1e-6, max(0.5, step))
    gx, gy = np.meshgrid(gs, gs)
    within = (gx ** 2 + gy ** 2) <= radius_mm ** 2
    disc = (centre[None, :]
            + gx[within][:, None] * e1[None, :]
            + gy[within][:, None] * e2[None, :])
    dhu = sample_hu(vol, dims, disc, outside=-1000.0)
    dm = (dhu >= hu_lo) & (dhu <= hu_hi)
    if dm.any():
        centre = disc[dm].mean(axis=0)
    return centre, t_star


def reconstruct_point(views_uv, vol=None, dims=None, hu_lo=200.0, hu_hi=1000.0,
                      radius_mm=4.0, prefer="near", t_range=(-150.0, 150.0)):
    """Reconstruct ONE 3-D centreline point from its 2-D marks.

    *views_uv* is a list of ``(ProjectionView, (x, y))`` — the same anatomical
    point marked on one or more views. Strategy:
      * 2+ views → triangulate the rays (geometry alone), then, if a volume is
        given, snap that estimate to the nearest contrast lumen along the ray;
      * 1 view → snap along its back-projection ray to the near/far contrast.
    Returns ``(point_or_None, info)`` where info has 'method' and 'rays'."""
    rays = [v.backproject(x, y) for v, (x, y) in views_uv]
    info = {"rays": rays, "method": None}
    tri = triangulate_rays(rays) if len(rays) >= 2 else None
    if tri is not None:
        info["method"] = "triangulate"
        if vol is not None:
            o, d = rays[0]
            t0 = float((tri - np.asarray(o, float)) @ _unit(d))
            p, _t = snap_to_contrast(
                vol, dims, o, d, t_range=(t0 - 20.0, t0 + 20.0), hu_lo=hu_lo,
                hu_hi=hu_hi, radius_mm=radius_mm, prefer=prefer)
            if p is not None:
                info["method"] = "triangulate+snap"
                return p, info
        return tri, info
    if vol is not None and rays:
        o, d = rays[0]
        p, _t = snap_to_contrast(vol, dims, o, d, t_range=t_range, hu_lo=hu_lo,
                                 hu_hi=hu_hi, radius_mm=radius_mm, prefer=prefer)
        if p is not None:
            info["method"] = "snap"
            return p, info
    return None, info
