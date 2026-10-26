"""Unit tests for core.cag_cpr — CAG→CPR geometry + CT contrast snap (headless)."""
import numpy as np
import pytest

from multi_dicomviewer.core import cag_cpr as cc


def _parallel_view(normal, up):
    """A parallel (MIP) view centred at the volume mid, looking along *normal*."""
    n = np.asarray(normal, float)
    u = np.cross(up, n)
    v = np.cross(n, u)
    return cc.ProjectionView(center=np.array([60.0, 60.0, 30.0]), u=u, v=v,
                             normal=n)


def test_parallel_project_backproject_roundtrip():
    view = _parallel_view([0, 0, 1], [0, 1, 0])     # look along +z
    P = np.array([75.0, 48.0, 50.0])
    x, y = view.project(P)
    o, d = view.backproject(x, y)
    # The back-projection ray must pass through P (P = o + t d for some t).
    w = P - o
    perp = w - (w @ d) * d
    assert np.linalg.norm(perp) < 1e-6, "ray should pass through P"


def test_perspective_project_backproject_roundtrip():
    view = cc.ProjectionView(center=np.array([60.0, 60.0, 60.0]),
                             u=np.array([1.0, 0, 0]), v=np.array([0, 1.0, 0]),
                             normal=np.array([0, 0, 1.0]),
                             source=np.array([60.0, 60.0, -600.0]))
    P = np.array([80.0, 40.0, 70.0])
    x, y = view.project(P)
    o, d = view.backproject(x, y)
    assert np.linalg.norm(o - view.source) < 1e-9, "perspective ray starts at source"
    w = P - o
    perp = w - (w @ d) * d
    assert np.linalg.norm(perp) < 1e-6, "ray should pass through P"


def test_triangulate_two_rays():
    P = np.array([70.0, 55.0, 42.0])
    v1 = _parallel_view([0, 0, 1], [0, 1, 0])        # along +z
    v2 = _parallel_view([1, 0, 0], [0, 0, 1])        # along +x
    rays = [v1.backproject(*v1.project(P)), v2.backproject(*v2.project(P))]
    est = cc.triangulate_rays(rays)
    assert est is not None and np.linalg.norm(est - P) < 1e-5


def test_triangulate_parallel_rays_returns_none():
    v = _parallel_view([0, 0, 1], [0, 1, 0])
    r = v.backproject(0.0, 0.0)
    assert cc.triangulate_rays([r, r]) is None       # same direction → degenerate


def _contrast_volume(center=(75.0, 48.0, 50.0), r=3.0, hu=400.0):
    """[z,y,x] HU: air (-1000) with a single bright contrast sphere."""
    nz, ny, nx = 90, 120, 120
    sx = sy = sz = 1.0
    Z, Y, X = np.mgrid[0:nz, 0:ny, 0:nx].astype(float)
    cx, cy, cz = center
    d2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    vol = np.full((nz, ny, nx), -1000.0, np.float32)
    vol[d2 <= r ** 2] = hu
    return vol, (sx, sy, sz)


def test_snap_to_contrast_finds_blob():
    blob = (75.0, 48.0, 50.0)
    vol, dims = _contrast_volume(center=blob, r=3.0, hu=400.0)
    view = _parallel_view([0, 0, 1], [0, 1, 0])      # look along +z
    x, y = view.project(np.asarray(blob))
    o, d = view.backproject(x, y)
    p, t = cc.snap_to_contrast(vol, dims, o, d, t_range=(-40, 40),
                               hu_lo=200, hu_hi=1000, radius_mm=4.0)
    assert p is not None
    assert np.linalg.norm(p - np.asarray(blob)) < 1.5, f"snapped to {p}, want {blob}"


def test_snap_near_vs_far():
    # Two blobs along the +z ray (front at z=40, back at z=70); near/far select.
    vol, dims = _contrast_volume(center=(75.0, 48.0, 40.0), r=3.0)
    v2, _ = _contrast_volume(center=(75.0, 48.0, 70.0), r=3.0)
    vol = np.maximum(vol, v2)
    view = _parallel_view([0, 0, 1], [0, 1, 0])
    x, y = view.project(np.array([75.0, 48.0, 55.0]))
    o, d = view.backproject(x, y)
    pn, _ = cc.snap_to_contrast(vol, dims, o, d, t_range=(-60, 60), prefer="near")
    pf, _ = cc.snap_to_contrast(vol, dims, o, d, t_range=(-60, 60), prefer="far")
    assert pn[2] < pf[2], "near cluster should be the front (smaller z)"
    assert abs(pn[2] - 40.0) < 1.5 and abs(pf[2] - 70.0) < 1.5


def test_reconstruct_point_two_views_plus_snap():
    blob = (75.0, 48.0, 50.0)
    vol, dims = _contrast_volume(center=blob, r=3.0)
    v1 = _parallel_view([0, 0, 1], [0, 1, 0])
    v2 = _parallel_view([1, 0, 0], [0, 0, 1])
    marks = [(v1, v1.project(np.asarray(blob))),
             (v2, v2.project(np.asarray(blob)))]
    p, info = cc.reconstruct_point(marks, vol=vol, dims=dims)
    assert p is not None and np.linalg.norm(p - np.asarray(blob)) < 1.5
    assert "triangulate" in info["method"]
