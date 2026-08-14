"""Phase-1 prototype: automatic LV cavity (endo) contour on a 2-D plane.

EXPERIMENTAL — lives on the `feature/lv-auto-trace` branch, not wired into a
release. The idea: on a resliced plane whose HU image contains the contrast-
filled LV blood pool, find the OUTER envelope of that bright region (bridging
trabeculae / papillary-muscle indentations, filling internal dark inclusions),
seeded from a point known to lie inside the cavity (the LV axis / crosshair).

Pure numpy + OpenCV (`cv2`) — both already bundled, so NO new dependency. This
produces an EDITABLE proposal, not a final segmentation: the caller turns the
returned polygon into a normal endo border the user then drags / adds / deletes
/ Ctrl+Z like a hand trace.

Design notes / known limits (see the feasibility write-up):
  * Robust on well-isolated mid-ventricular planes (high contrast).
  * Weak near the base (cavity merges with LA / LVOT / aorta) — clip with the
    caller's basal geometry and/or a ROI radius; hand-edit the rest.
  * Threshold varies with contrast timing — estimate per-plane from the seed,
    and expose it so the UI can offer a slider.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:                                   # pragma: no cover
    cv2 = None


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill interior holes of a 0/1 mask (papillary muscles / trabeculae that
    read as myocardium inside the bright pool). Flood the exterior background
    from a corner, then anything still background is an interior hole."""
    h, w = binary.shape
    ff = binary.copy().astype(np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Seed the flood from a corner assumed to be exterior background.
    cv2.floodFill(ff, mask, (0, 0), 1)
    holes = (ff == 0).astype(np.uint8)
    return ((binary.astype(np.uint8) | holes) > 0).astype(np.uint8)


def _resample_closed(poly: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polygon (Nx2, x,y) to *n* points evenly by arc length."""
    p = np.asarray(poly, float)
    if len(p) < 3:
        return p
    loop = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-6:
        return p
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 2), float)
    j = 0
    for i, tdist in enumerate(targets):
        while j < len(seg) and cum[j + 1] < tdist:
            j += 1
        segd = seg[j] if seg[j] > 1e-9 else 1.0
        f = (tdist - cum[j]) / segd
        out[i] = loop[j] * (1.0 - f) + loop[j + 1] * f
    return out


def snap_seed(hu: np.ndarray, seed_ij, radius: int = 6):
    """Nudge the seed to the brightest pixel within *radius* — so a slightly
    off-centre crosshair still lands inside the blood pool."""
    h, w = hu.shape
    si, sj = int(round(seed_ij[0])), int(round(seed_ij[1]))
    i0, i1 = max(0, si - radius), min(h, si + radius + 1)
    j0, j1 = max(0, sj - radius), min(w, sj + radius + 1)
    if i0 >= i1 or j0 >= j1:
        return si, sj
    sub = hu[i0:i1, j0:j1]
    di, dj = np.unravel_index(int(np.argmax(sub)), sub.shape)
    return i0 + di, j0 + dj


def estimate_threshold(hu: np.ndarray, seed_ij, roi_radius_px: float | None = None):
    """A per-plane blood/myocardium HU threshold. Otsu within a seed-centred ROI
    (falls back to a fixed split if Otsu degenerates on a near-uniform patch)."""
    h, w = hu.shape
    si, sj = int(seed_ij[0]), int(seed_ij[1])
    if roi_radius_px:
        yy, xx = np.ogrid[:h, :w]
        roi = ((yy - si) ** 2 + (xx - sj) ** 2) <= float(roi_radius_px) ** 2
        vals = hu[roi]
    else:
        vals = hu.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size < 16:
        return 200.0
    lo, hi = float(np.percentile(vals, 2)), float(np.percentile(vals, 99))
    if hi - lo < 1e-3:
        return lo
    u8 = np.clip((vals - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    t8, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = lo + (float(t8) / 255.0) * (hi - lo)
    blood = float(np.median(hu[max(0, si - 3):si + 4, max(0, sj - 3):sj + 4]))
    # Keep the threshold safely below the blood pool so the seed survives.
    return float(min(thr, blood - 40.0))


def auto_cavity_contour(hu: np.ndarray, seed_ij, thr: float | None = None,
                        roi_radius_px: float | None = None,
                        close_frac: float = 0.04, n_points: int = 64,
                        min_area_px: int = 60):
    """Outer contour of the contrast-filled cavity around *seed_ij* (row, col).

    Returns an (n_points, 2) array of (x=col, y=row) pixel coordinates, or None
    if nothing plausible was found. The caller maps pixels → plane mm → 3-D.
    """
    if cv2 is None or hu is None or hu.ndim != 2:
        return None
    h, w = hu.shape
    si, sj = snap_seed(hu, seed_ij)
    if not (0 <= si < h and 0 <= sj < w):
        return None
    if thr is None:
        thr = estimate_threshold(hu, (si, sj), roi_radius_px)
    mask = (hu >= float(thr)).astype(np.uint8)
    if roi_radius_px:
        yy, xx = np.ogrid[:h, :w]
        roi = ((yy - si) ** 2 + (xx - sj) ** 2) <= float(roi_radius_px) ** 2
        mask = (mask & roi.astype(np.uint8)).astype(np.uint8)
    if mask[si, sj] == 0:
        return None                                 # seed not bright → give up
    num, lbl = cv2.connectedComponents(mask, connectivity=8)
    seed_label = int(lbl[si, sj])
    if seed_label == 0:
        return None
    comp = (lbl == seed_label).astype(np.uint8)
    if int(comp.sum()) < min_area_px:
        return None
    comp = _fill_holes(comp)
    k = max(3, int(close_frac * max(h, w)))
    k = k + 1 if k % 2 == 0 else k                  # odd kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, kernel)
    comp = _fill_holes(comp)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)  # (x, y)
    if len(cnt) < 3:
        return None
    return _resample_closed(cnt, n_points)
