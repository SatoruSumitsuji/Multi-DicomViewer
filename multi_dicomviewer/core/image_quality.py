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


def denoise(frame8: np.ndarray | None) -> np.ndarray | None:
    """Edge-preserving noise reduction on an 8-bit grayscale ``(H, W)`` or RGB
    ``(H, W, 3)`` frame.

    Bilateral filtering smooths flat / noisy regions (IVUS echo speckle,
    angiographic quantum noise) while keeping vessel and catheter edges crisp —
    unlike a plain Gaussian blur, which would soften the borders the reader
    cares about. Returns the input unchanged when cv2 is unavailable or the
    frame is not a uint8 2-D / 3-D image (so a colour-mapped or odd frame can
    never crash the paint path)."""
    if not HAVE_CV2 or frame8 is None:
        return frame8
    if frame8.dtype != np.uint8 or frame8.ndim not in (2, 3):
        return frame8
    src = np.ascontiguousarray(frame8)
    # d=5 → a small 5-px neighbourhood (fast); sigmaColor=50 calms speckle
    # without bleeding across strong edges; sigmaSpace=5 keeps it local. Tuned
    # for medical grayscale where over-smoothing hides real low-contrast detail.
    out = cv2.bilateralFilter(src, 5, 50, 5)
    return np.ascontiguousarray(out)


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
