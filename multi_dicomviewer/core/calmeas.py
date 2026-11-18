"""Cal-Measure — calibration-first measurement for 2-D images (XA angio,
echo, …) whose pixel size is not trusted or not known (user 2026-11-18).

Pure numpy / SciPy, no Qt:

* calibration — two points + their known distance, or a CATHETER: the
  contrast-filled catheter inside a user polygon is extracted (the
  elongated dark / bright object), its centreline found and the full width
  at half maximum (FWHM) measured on cross-section profiles along it; the
  catheter's French size gives the true outer diameter (Fr / 3 mm).
* measurements — straight distance, polyline length and the FWHM diameter
  on a user line drawn across a vessel. The FWHM is the primary value as
  measured on that line; the vessel's direction near the crossing is
  estimated so an oblique line can be flagged, and (only when the estimate
  is reliable) a perpendicular re-measure is offered as a reference value.

Images are 2-D float arrays ``img[y, x]``; points are (x, y) in pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

FR_MM = 1.0 / 3.0            # 1 French = 1/3 mm (outer diameter)


def to_gray(frame) -> np.ndarray:
    """A frame as 2-D float32 (RGB → luminance)."""
    f = np.asarray(frame)
    if f.ndim == 3 and f.shape[-1] in (3, 4):
        f = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2])
    return np.asarray(f, np.float32)


def sample_line(img, p0, p1, step=0.25):
    """Bilinear samples along p0 → p1 every *step* px. (values, t_px)."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    n = max(2, int(math.ceil(L / step)) + 1)
    t = np.linspace(0.0, L, n)
    u = (p1 - p0) / (L or 1.0)
    xs = p0[0] + t * u[0]
    ys = p0[1] + t * u[1]
    v = ndimage.map_coordinates(np.asarray(img, np.float32), [ys, xs],
                                order=1, mode="nearest")
    return v, t


@dataclass
class FWHM:
    ok: bool
    width_px: float = float("nan")
    left: float = float("nan")          # along-line positions (px)
    right: float = float("nan")
    centre: float = float("nan")
    contrast: float = 0.0               # object − background (signed)
    cnr: float = 0.0                    # |contrast| / background noise
    dark: bool = True                   # object darker than background
    dip: bool = False                   # two peaks (an empty catheter's walls)
    reason: str = ""


def fwhm_profile(v, t, end_frac=0.15, min_cnr=3.0) -> FWHM:
    """FWHM of the object crossed by a profile. The background is the line
    straight between the means of the two end segments (*end_frac* of the
    length each — the line must reach beyond the object on both sides);
    the object's polarity (dark or bright) is whichever deviation is the
    larger in the middle. Sub-sample edges by linear interpolation."""
    v = np.asarray(v, float)
    t = np.asarray(t, float)
    n = len(v)
    if n < 12:
        return FWHM(False, reason="short")
    k = max(2, int(round(n * end_frac)))
    b0, b1 = float(np.mean(v[:k])), float(np.mean(v[-k:]))
    base = b0 + (b1 - b0) * (t - t[0]) / ((t[-1] - t[0]) or 1.0)
    d = v - base
    noise = float(np.std(np.r_[v[:k] - b0, v[-k:] - b1])) or 1e-6
    mid = d[k:n - k]
    if len(mid) < 3:
        return FWHM(False, reason="short")
    dark = abs(float(mid.min())) >= abs(float(mid.max()))
    s = -d if dark else d                          # object → positive
    sm = ndimage.uniform_filter1d(s, 3)
    i_pk = k + int(np.argmax(sm[k:n - k]))
    peak = float(sm[i_pk])
    cnr = peak / noise
    res = FWHM(False, contrast=(-peak if dark else peak), cnr=cnr, dark=dark)
    if peak <= 0 or cnr < min_cnr:
        res.reason = "contrast"
        return res
    half = 0.5 * peak
    i = i_pk
    while i > 0 and s[i] >= half:
        i -= 1
    j = i_pk
    while j < n - 1 and s[j] >= half:
        j += 1
    if s[i] >= half or s[j] >= half:
        res.reason = "edge"                         # object reaches the ends
        return res

    def cross(a, b):                               # s[a] < half <= s[b]
        fa, fb = s[a], s[b]
        f = (half - fa) / ((fb - fa) or 1e-12)
        return t[a] + f * (t[b] - t[a])
    left = cross(i, i + 1)
    right = cross(j, j - 1)
    seg = sm[i + 1:j]
    dip = False
    # a second hump of similar height just beside the measured one (an
    # empty catheter's other wall) → the profile is two-peaked
    span = max(j - i, 2)
    lo_, hi_ = max(k, i - 4 * span), min(n - k, j + 4 * span)
    for a0, a1 in ((lo_, i), (j + 1, hi_)):
        part = sm[a0:a1]
        if len(part) and float(part.max()) >= 0.6 * peak:
            dip = True
    if len(seg) >= 5:
        # two humps with a clear valley between them (catheter walls only)
        a_ = int(np.argmax(seg[:len(seg) // 2 + 1]))
        b_ = len(seg) // 2 + int(np.argmax(seg[len(seg) // 2:]))
        if b_ - a_ >= 3:
            valley = float(seg[a_:b_ + 1].min())
            dip = valley < 0.7 * min(seg[a_], seg[b_])
    return FWHM(True, width_px=float(right - left), left=float(left),
                right=float(right), centre=float(0.5 * (left + right)),
                contrast=(-peak if dark else peak), cnr=cnr, dark=dark,
                dip=dip)


def line_fwhm(img, p0, p1, step=0.25) -> FWHM:
    v, t = sample_line(img, p0, p1, step)
    return fwhm_profile(v, t)


# ---- vessel direction at a diameter line ------------------------------------
@dataclass
class Diameter:
    fwhm: FWHM
    p0: tuple
    p1: tuple
    edges: list = field(default_factory=list)     # [(x, y), (x, y)]
    angle_deg: float = float("nan")               # line vs vessel-normal
    direction_ok: bool = False
    direction_reason: str = ""
    corrected: FWHM | None = None                 # perpendicular re-measure
    corrected_line: list = field(default_factory=list)
    corrected_edges: list = field(default_factory=list)


def measure_diameter(img, p0, p1, step=0.25, min_angle_deg=10.0) -> Diameter:
    """FWHM diameter on the user line p0 → p1 (the primary value), plus an
    estimate of how far the line is from perpendicular to the vessel.

    The vessel direction: parallel copies of the line are shifted along the
    vessel (± about one diameter); the object's centre on each copy traces
    the vessel axis. It is trusted only when every copy finds the object
    with a similar width (no stenosis edge / branch / overlap there) and the
    centres lie on a straight line. When trusted and the line is ≥
    *min_angle_deg* off, the diameter is re-measured on the perpendicular
    through the same centre (a REFERENCE value; the primary is unchanged)."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    f = line_fwhm(img, p0, p1, step)
    L = float(np.linalg.norm(p1 - p0))
    u = (p1 - p0) / (L or 1.0)
    out = Diameter(f, tuple(map(float, p0)), tuple(map(float, p1)))
    if not f.ok:
        return out
    out.edges = [tuple(map(float, p0 + f.left * u)),
                 tuple(map(float, p0 + f.right * u))]
    nrm = np.array([-u[1], u[0]])
    w = f.width_px
    offs = np.linspace(-1.0, 1.0, 7) * max(w, 3.0)
    cs, ws = [], []
    for s in offs:
        g = line_fwhm(img, p0 + s * nrm, p1 + s * nrm, step)
        if not g.ok or g.dark != f.dark:
            out.direction_reason = "lost"
            return out
        cs.append(g.centre)
        ws.append(g.width_px)
    cs = np.asarray(cs)
    ws = np.asarray(ws)
    if ws.max() > 1.35 * ws.min() + 0.5:
        out.direction_reason = "width"          # narrowing / branch here
        return out
    A = np.c_[offs, np.ones_like(offs)]
    coef, *_ = np.linalg.lstsq(A, cs, rcond=None)
    slope = float(coef[0])                      # along-line shift per offset
    resid = cs - A @ coef
    if float(np.sqrt(np.mean(resid ** 2))) > max(0.6, 0.12 * w):
        out.direction_reason = "curved"
        return out
    # vessel axis a ∝ nrm + slope·u ; the line's angle off the axis normal
    out.angle_deg = float(math.degrees(math.atan(abs(slope))))
    out.direction_ok = True
    if out.angle_deg >= min_angle_deg:
        a = nrm + slope * u
        a /= np.linalg.norm(a)
        perp = np.array([-a[1], a[0]])
        c = p0 + f.centre * u
        h = 0.5 * L
        q0, q1 = c - h * perp, c + h * perp
        g = line_fwhm(img, q0, q1, step)
        if g.ok:
            out.corrected = g
            out.corrected_line = [tuple(map(float, q0)), tuple(map(float, q1))]
            out.corrected_edges = [tuple(map(float, q0 + g.left * perp)),
                                   tuple(map(float, q0 + g.right * perp))]
    return out


# ---- catheter calibration ------------------------------------------------------
def polygon_mask(shape, poly) -> np.ndarray:
    """Boolean mask of the pixels whose centres lie inside *poly* [(x, y)]."""
    h, w = shape
    P = np.asarray(poly, float)
    x0, y0 = np.floor(P.min(axis=0)).astype(int)
    x1, y1 = np.ceil(P.max(axis=0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w - 1), min(y1, h - 1)
    m = np.zeros((h, w), bool)
    if x1 < x0 or y1 < y0:
        return m
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    X = xs.astype(float)
    Y = ys.astype(float)
    inside = np.zeros(X.shape, bool)
    n = len(P)
    for i in range(n):
        xa, ya = P[i]
        xb, yb = P[(i + 1) % n]
        cond = (ya > Y) != (yb > Y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = xa + (Y - ya) * (xb - xa) / (yb - ya)
        inside ^= cond & (X < xc)
    m[y0:y1 + 1, x0:x1 + 1] = inside
    return m


def _otsu(vals) -> float:
    v = np.asarray(vals, float)
    hist, edges = np.histogram(v, bins=128)
    c = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * c)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = m0 / w0
        mu1 = (mt - m0) / w1
        sb = w0 * w1 * (mu0 - mu1) ** 2
    sb[~np.isfinite(sb)] = -1
    return float(c[int(np.argmax(sb))])


@dataclass
class Catheter:
    ok: bool
    width_px: float = float("nan")
    iqr_px: float = float("nan")
    n_profiles: int = 0
    centerline: list = field(default_factory=list)   # [(x, y)]
    dark: bool = True
    dip_frac: float = 0.0
    warnings: list = field(default_factory=list)
    reason: str = ""


def catheter_width(img, poly, step=0.25, every_px=1.0) -> Catheter:
    """Width (px) of the catheter inside polygon *poly*: the elongated
    object (dark or bright, whichever is the more elongated Otsu class) →
    its centreline (centroids of slabs along the main axis, smoothed) →
    FWHM on a perpendicular profile every *every_px* along it (ends
    trimmed) → the median. Warnings: an empty catheter's two-wall profile,
    a large spread, too few profiles."""
    img = np.asarray(img, np.float32)
    mask = polygon_mask(img.shape, poly)
    if mask.sum() < 30:
        return Catheter(False, reason="polygon")
    thr = _otsu(img[mask])
    best = None
    for dark in (True, False):
        cls = (img < thr) if dark else (img > thr)
        lab, n = ndimage.label(cls & mask)
        if n == 0:
            continue
        sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        comp = lab == k
        ys, xs = np.nonzero(comp)
        if len(xs) < 20:
            continue
        P = np.c_[xs, ys].astype(float)
        mu = P.mean(axis=0)
        ev, evec = np.linalg.eigh(np.cov((P - mu).T))
        elong = math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))
        if best is None or elong > best[0]:
            best = (elong, dark, P, mu, evec[:, 1])
    if best is None or best[0] < 2.5:
        return Catheter(False, reason="shape")
    elong, dark, P, mu, ax = best
    proj = (P - mu) @ ax
    lo, hi = float(proj.min()), float(proj.max())
    length = hi - lo
    est_w = len(P) / max(length, 1.0)
    bin_w = max(2.0, 0.5 * est_w)
    nb = max(3, int(length / bin_w))
    edges = np.linspace(lo, hi, nb + 1)
    cl = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (proj >= a) & (proj < b)
        if sel.sum() >= 3:
            cl.append(P[sel].mean(axis=0))
    if len(cl) < 3:
        return Catheter(False, reason="shape")
    cl = np.asarray(cl)
    if len(cl) >= 5:
        cl = np.c_[ndimage.uniform_filter1d(cl[:, 0], 3, mode="nearest"),
                   ndimage.uniform_filter1d(cl[:, 1], 3, mode="nearest")]
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    S = float(s[-1])
    trim = 0.12 * S + 0.5 * est_w
    ss = np.arange(trim, S - trim, every_px)
    if len(ss) < 5:
        return Catheter(False, reason="short")
    cx = np.interp(ss, s, cl[:, 0])
    cy = np.interp(ss, s, cl[:, 1])
    tx = np.gradient(cx)
    ty = np.gradient(cy)
    # profiles span the polygon's width (so an empty catheter's OTHER wall
    # is on the profile too), within 3…8 × the object's own width
    ym, xm = np.nonzero(mask)
    perp = np.array([-ax[1], ax[0]])
    pw = np.c_[xm, ym] @ perp
    half = float(np.clip(0.5 * (pw.max() - pw.min()), 3.0 * est_w,
                         8.0 * est_w))
    half = max(half, 6.0)
    widths, dips = [], 0
    for x, y, a, b in zip(cx, cy, tx, ty):
        nn = math.hypot(a, b) or 1.0
        nx, ny = -b / nn, a / nn
        f = line_fwhm(img, (x - half * nx, y - half * ny),
                      (x + half * nx, y + half * ny), step)
        if f.ok and f.dark == dark:
            widths.append(f.width_px)
            dips += int(f.dip)
    out = Catheter(False, centerline=[tuple(map(float, q)) for q in
                                      np.c_[cx, cy]], dark=dark)
    if len(widths) < 5:
        out.reason = "profiles"
        return out
    w = np.asarray(widths)
    med = float(np.median(w))
    q1, q3 = np.percentile(w, [25, 75])
    out.ok = True
    out.width_px = med
    out.iqr_px = float(q3 - q1)
    out.n_profiles = len(w)
    out.dip_frac = dips / len(w)
    if out.dip_frac > 0.5:                  # walls only: not contrast-filled
        out.ok = False
        out.reason = "dip"
        return out
    if out.dip_frac > 0.2:
        out.warnings.append("dip")
    if out.iqr_px > 0.15 * med:
        out.warnings.append("spread")
    if out.n_profiles < 15:
        out.warnings.append("few")
    return out


def path_length(pts) -> float:
    P = np.asarray(pts, float)
    if len(P) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


# ---- DICOM pixel size (reference option) ------------------------------------
def dicom_mm_per_px(ds) -> tuple[float, str] | None:
    """(mm/px, source) from the dataset: an ultrasound B-mode region
    (PhysicalDeltaX in cm), else PixelSpacing, else ImagerPixelSpacing (the
    detector plane — an XA's magnification is NOT corrected). None if none."""
    if ds is None:
        return None
    try:
        regs = getattr(ds, "SequenceOfUltrasoundRegions", None) or []
        best = None
        for r in regs:
            if int(getattr(r, "RegionSpatialFormat", 0)) != 1:
                continue
            if int(getattr(r, "PhysicalUnitsXDirection", 0)) != 3:  # cm
                continue
            dx = abs(float(getattr(r, "PhysicalDeltaX", 0) or 0))
            if dx <= 0:
                continue
            area = (abs(int(getattr(r, "RegionLocationMaxX1", 0))
                        - int(getattr(r, "RegionLocationMinX0", 0)))
                    * abs(int(getattr(r, "RegionLocationMaxY1", 0))
                          - int(getattr(r, "RegionLocationMinY0", 0))))
            if best is None or area > best[0]:
                best = (area, dx * 10.0)
        if best is not None:
            return best[1], "US region"
    except (TypeError, ValueError, AttributeError):
        pass
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        try:
            v = getattr(ds, tag, None)
            if v is not None and len(v) >= 2 and float(v[0]) > 0:
                return float(v[0]), tag
        except (TypeError, ValueError):
            continue
    return None
