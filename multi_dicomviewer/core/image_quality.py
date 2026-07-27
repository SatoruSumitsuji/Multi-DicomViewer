"""Optional OpenCV-backed image-quality processing for the 2-D (Angio/IVUS)
viewer: edge-preserving noise reduction and Lanczos high-quality upscaling.

Both are opt-in user toggles (``core.settings`` keys ``xa_denoise`` /
``xa_smooth``) that default OFF, so a fresh install renders exactly as before.
OpenCV (``cv2``) is bundled, but every function degrades gracefully to a
no-op / "let the caller fall back" if it is somehow unavailable — the same
philosophy as the optional ``imagecodecs`` fast decoder, so the viewer always
keeps running.
"""
from __future__ import annotations

import numpy as np

try:                                  # OpenCV is bundled; tolerate its absence.
    import cv2                        # type: ignore
    HAVE_CV2 = True
except Exception:                     # pragma: no cover - exercised only sans cv2
    cv2 = None                        # type: ignore
    HAVE_CV2 = False


def available() -> bool:
    """True iff the OpenCV backend is importable (so the UI can disable or
    annotate the Smooth/Denoise toggles when it is not)."""
    return HAVE_CV2


def denoise(frame8: np.ndarray | None, sigma: float = 50.0) -> np.ndarray | None:
    """Edge-preserving noise reduction on an 8-bit grayscale ``(H, W)`` or RGB
    ``(H, W, 3)`` frame.

    Bilateral filtering smooths flat / noisy regions (IVUS echo speckle,
    angiographic quantum noise) while keeping vessel and catheter edges crisp —
    unlike a plain Gaussian blur, which would soften the borders the reader
    cares about. *sigma* is the colour sigma (strength): higher = stronger
    smoothing (50 is the tuned default; 0 = off). Returns the input unchanged
    when cv2 is unavailable, *sigma* ≤ 0, or the frame is not a uint8 2-D / 3-D
    image (so a colour-mapped or odd frame can never crash the paint path)."""
    if not HAVE_CV2 or frame8 is None or sigma <= 0:
        return frame8
    if frame8.dtype != np.uint8 or frame8.ndim not in (2, 3):
        return frame8
    src = np.ascontiguousarray(frame8)
    # d=5 → a small 5-px neighbourhood (fast); sigmaSpace=5 keeps it local.
    out = cv2.bilateralFilter(src, 5, float(sigma), 5)
    return np.ascontiguousarray(out)


def sharpen(frame8: np.ndarray | None, amount: float = 0.0) -> np.ndarray | None:
    """Unsharp-mask sharpening — accentuates vessel / catheter edges. *amount*
    is a percentage (0 = off, 100 ≈ 1.0× the high-pass detail added back).
    No-op without cv2 or for a non-uint8 frame."""
    if not HAVE_CV2 or frame8 is None or amount <= 0:
        return frame8
    if frame8.dtype != np.uint8 or frame8.ndim not in (2, 3):
        return frame8
    src = np.ascontiguousarray(frame8)
    blur = cv2.GaussianBlur(src, (0, 0), 1.2)
    a = float(amount) / 100.0
    out = cv2.addWeighted(src, 1.0 + a, blur, -a, 0)   # uint8 auto-clamped
    return np.ascontiguousarray(out)


def clahe(frame8: np.ndarray | None, clip: float = 0.0) -> np.ndarray | None:
    """Contrast-Limited Adaptive Histogram Equalisation — boosts *local*
    contrast so faint low-contrast vessels stand out. *clip* is the clip limit
    (0 = off; ~2.0 is a moderate boost). Applied to luma only for colour frames
    so hue is preserved. No-op without cv2 or for a non-uint8 frame."""
    if not HAVE_CV2 or frame8 is None or clip <= 0:
        return frame8
    if frame8.dtype != np.uint8 or frame8.ndim not in (2, 3):
        return frame8
    obj = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
    if frame8.ndim == 2:
        return np.ascontiguousarray(obj.apply(np.ascontiguousarray(frame8)))
    ycc = cv2.cvtColor(np.ascontiguousarray(frame8), cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = obj.apply(np.ascontiguousarray(ycc[..., 0]))
    return np.ascontiguousarray(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB))


def enhance(frame8: np.ndarray | None, denoise_sigma: float = 0.0,
            sharpen_amount: float = 0.0,
            clahe_clip: float = 0.0) -> np.ndarray | None:
    """Angio / IVUS enhancement pipeline: denoise → CLAHE → sharpen. Each step
    is skipped when its parameter is 0, so the default (0, 0, 0) is a no-op and
    the classic single-step Denoise is just ``enhance(f, 50, 0, 0)``. Denoise
    runs first so CLAHE / sharpen don't amplify the noise it removes."""
    f = denoise(frame8, denoise_sigma)
    f = clahe(f, clahe_clip)
    f = sharpen(f, sharpen_amount)
    return f


def lanczos_resize(frame8: np.ndarray | None, w: int, h: int):
    """High-quality Lanczos-4 resample of *frame8* to ``(w, h)`` pixels —
    sharper than the bilinear upscaling Qt applies at paint time.

    Returns ``None`` when cv2 is unavailable or the inputs are unusable, so the
    caller can fall back to Qt's own scaling."""
    if not HAVE_CV2 or frame8 is None:
        return None
    if frame8.dtype != np.uint8 or frame8.ndim not in (2, 3):
        return None
    w, h = int(w), int(h)
    if w <= 0 or h <= 0:
        return None
    out = cv2.resize(np.ascontiguousarray(frame8), (w, h),
                     interpolation=cv2.INTER_LANCZOS4)
    return np.ascontiguousarray(out)
