"""Headless test: region_outline_on_plane traces the cross-section of a 3-D
region on an arbitrary plane, so the Epi border can be drawn as the section of
the reconstructed solid (tracking free rotation, always coincident with the
resliced red fill).

Scenario: a solid sphere (radius R) in a volume. On a plane through the centre
the section is a circle of radius ~R; on a plane offset by d along the normal it
is a circle of radius ~sqrt(R^2 - d^2). Verify the traced outline's mean radius
matches, for an AXIS-ALIGNED and an OBLIQUE plane.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core.lv_compact import (          # noqa: E402
    region_outline_on_plane,
)


def _poly_mean_radius(poly, cu=0.0, cv=0.0):
    p = np.asarray(poly, float)
    return float(np.mean(np.hypot(p[:, 0] - cu, p[:, 1] - cv)))


def main():
    sx = sy = sz = 1.0
    nx = ny = nz = 100
    cx, cy, cz = 50.0, 50.0, 50.0
    R = 25.0
    xc = (np.arange(nx) * sx).reshape(1, 1, -1)
    yc = (np.arange(ny) * sy).reshape(1, -1, 1)
    zc = (np.arange(nz) * sz).reshape(-1, 1, 1)
    sphere = ((xc - cx) ** 2 + (yc - cy) ** 2 + (zc - cz) ** 2) <= R ** 2
    bbox = (0, nz, 0, ny, 0, nx)

    # 1) Axis-aligned plane through the centre (u=+x, v=+y, origin=centre).
    polys = region_outline_on_plane(
        sphere, bbox, (sx, sy, sz), (cx, cy, cz),
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), half_mm=40.0, step_mm=0.8)
    assert polys, "no outline on the central axial plane"
    r0 = max(_poly_mean_radius(p) for p in polys)
    print(f"axial centre: outline r = {r0:6.2f} (expect ~{R})")
    assert abs(r0 - R) < 1.5, f"axial centre radius off: {r0:.2f}"

    # 2) OBLIQUE plane through the centre (u,v tilted 45° about x). A great circle
    #    of a sphere is still radius R regardless of orientation.
    th = math.radians(45.0)
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, math.cos(th), math.sin(th)])
    polys = region_outline_on_plane(
        sphere, bbox, (sx, sy, sz), (cx, cy, cz), u, v,
        half_mm=40.0, step_mm=0.8)
    assert polys, "no outline on the oblique central plane"
    r1 = max(_poly_mean_radius(p) for p in polys)
    print(f"oblique centre: outline r = {r1:6.2f} (expect ~{R})")
    assert abs(r1 - R) < 1.5, f"oblique centre radius off: {r1:.2f}"

    # 3) Plane OFFSET d along its normal → smaller section circle.
    d = 15.0
    origin = np.array([cx, cy, cz]) + d * np.array([0.0, 0.0, 1.0])
    polys = region_outline_on_plane(
        sphere, bbox, (sx, sy, sz), origin,
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), half_mm=40.0, step_mm=0.8)
    assert polys, "no outline on the offset plane"
    r2 = max(_poly_mean_radius(p) for p in polys)
    exp = math.sqrt(R ** 2 - d ** 2)
    print(f"offset {d}mm: outline r = {r2:6.2f} (expect ~{exp:.2f})")
    assert abs(r2 - exp) < 1.5, f"offset radius off: {r2:.2f} vs {exp:.2f}"

    # 4) Plane entirely OUTSIDE the sphere → no outline.
    origin = np.array([cx, cy, cz]) + (R + 5.0) * np.array([0.0, 0.0, 1.0])
    polys = region_outline_on_plane(
        sphere, bbox, (sx, sy, sz), origin,
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), half_mm=40.0, step_mm=0.8)
    assert not polys, "expected no outline beyond the sphere"
    print("beyond sphere: no outline (correct)")
    print("PASS")


if __name__ == "__main__":
    main()
