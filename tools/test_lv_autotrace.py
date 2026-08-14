"""Headless sanity test for the Phase-1 LV cavity auto-trace core.

Builds a synthetic HU plane resembling a contrast LV cavity: a bright blob with
(a) internal dark inclusions (papillary muscles / trabeculae) and (b) concave
outer indentations, surrounded by mid-grey myocardium. The auto-trace must
return an OUTER envelope that FILLS the inclusions and BRIDGES the indentations
(area close to the blob's convex-ish outer area, not the swiss-cheese area).

Run:  python tools/test_lv_autotrace.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core.lv_autotrace import auto_cavity_contour  # noqa: E402


def _disk(shape, cy, cx, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def build_plane(n=240):
    hu = np.full((n, n), 60.0, np.float32)          # myocardium ~60 HU
    hu[:20, :] = -200.0                             # some lung/air outside
    cy = cx = n // 2
    R = 70
    pool = _disk((n, n), cy, cx, R)
    # a concave bite out of the outer boundary (trabecular cleft)
    bite = _disk((n, n), cy - R, cx, 22)
    pool = pool & ~bite
    hu[pool] = 420.0                                # contrast blood ~420 HU
    # internal dark inclusions (papillary muscles) fully inside the pool
    for (dy, dx, rr) in [(15, -18, 12), (18, 20, 10), (-25, 5, 8)]:
        hu[_disk((n, n), cy + dy, cx + dx, rr)] = 70.0
    return hu, (cy, cx), R


def main():
    hu, seed, R = build_plane()
    cnt = auto_cavity_contour(hu, seed, roi_radius_px=110, n_points=80)
    assert cnt is not None, "auto_cavity_contour returned None"
    assert len(cnt) == 80, f"expected 80 points, got {len(cnt)}"

    area = _polygon_area(cnt)
    full_disk = np.pi * R * R
    # Holes filled + bite bridged → area should be a large fraction of the full
    # outer disk (NOT the swiss-cheese blob). Allow the closing/threshold slack.
    frac = area / full_disk
    print(f"contour points : {len(cnt)}")
    print(f"contour area   : {area:.0f} px^2")
    print(f"full outer disk: {full_disk:.0f} px^2  (ratio {frac:.2f})")

    # Centroid near the seed (didn't leak elsewhere).
    cxy = cnt.mean(axis=0)
    dist = float(np.hypot(cxy[0] - seed[1], cxy[1] - seed[0]))
    print(f"centroid offset from seed: {dist:.1f} px")

    ok = (0.75 <= frac <= 1.15) and dist <= 15.0
    print("RESULT:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
