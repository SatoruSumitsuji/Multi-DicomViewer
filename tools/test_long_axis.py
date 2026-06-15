"""build_long_axis: vectorised (ndarray) path must equal the per-frame loop.

The long-axis composite gained a vectorised fast path (used when `frames` is a
contiguous (n,H,W[,3]) volume) to stop a long colour pull-back freezing the UI
on a rotate/rebuild. These tests pin it bit-identical to the original
frame-by-frame fallback (exercised by passing a Python list of frames).
"""
import math

import numpy as np

from multi_dicomviewer.viewers.long_axis_canvas import build_long_axis


def _gray_u8(col):
    """W/L-style map for uint8: identity passthrough (matches IVUSViewer.to_u8
    uint8 branch)."""
    return col.astype(np.uint8)


def _lut_map(col):
    """A LUT gather like the grayscale-integer W/L path (uint16 -> uint8)."""
    lut = (np.arange(65536) // 256).astype(np.uint8)
    return lut[col]


def _centers(n, w, h, seed):
    rng = np.random.default_rng(seed)
    # Mix in-image and deliberately off-image centres so out-of-bounds
    # masking is exercised on both paths.
    cx = rng.uniform(-5, w + 5, n)
    cy = rng.uniform(-5, h + 5, n)
    return [(float(a), float(b)) for a, b in zip(cx, cy)]


def _check(vol, centers, lateral, apply_u8):
    for angle in (0.0, 0.3, math.pi / 2, 2.1, math.pi):
        for step in (1.0, 2.0, 3.0):
            fast = build_long_axis(vol, centers, angle, lateral, apply_u8, step)
            slow = build_long_axis(list(vol), centers, angle, lateral,
                                   apply_u8, step)
            assert fast.shape == slow.shape, (fast.shape, slow.shape)
            assert fast.dtype == np.uint8
            assert np.array_equal(fast, slow), (
                f"mismatch angle={angle} step={step} "
                f"maxdiff={int(np.abs(fast.astype(int) - slow.astype(int)).max())}"
            )


def test_gray_uint16_lut_matches_loop():
    n, h, w = 40, 48, 64
    rng = np.random.default_rng(1)
    vol = rng.integers(0, 65536, size=(n, h, w), dtype=np.uint16)
    lateral = int(round(math.sqrt(h * h + w * w)))
    _check(vol, _centers(n, w, h, 11), lateral, _lut_map)


def test_gray_float32_window_matches_loop():
    from multi_dicomviewer.viewers.xa_viewer import apply_window
    n, h, w = 30, 50, 50
    rng = np.random.default_rng(2)
    vol = (rng.random((n, h, w), dtype=np.float32) * 4000.0).astype(np.float32)
    lateral = int(round(math.sqrt(h * h + w * w)))
    _check(vol, _centers(n, w, h, 22), lateral,
           lambda c: apply_window(c, 2000.0, 1000.0))


def test_color_uint8_matches_loop():
    n, h, w = 35, 40, 56
    rng = np.random.default_rng(3)
    vol = rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)
    lateral = int(round(math.sqrt(h * h + w * w)))
    _check(vol, _centers(n, w, h, 33), lateral, _gray_u8)


def test_list_with_none_uses_loop_path():
    """A list containing a None frame stays on the fallback path and that
    frame's column is all-zero (after apply_u8)."""
    n, h, w = 8, 20, 20
    rng = np.random.default_rng(4)
    frames = [rng.integers(0, 256, (h, w), dtype=np.uint8) for _ in range(n)]
    frames[3] = None
    centers = [(w / 2.0, h / 2.0)] * n
    lateral = int(round(math.sqrt(h * h + w * w)))
    img = build_long_axis(frames, centers, 0.0, lateral, _gray_u8)
    assert img.shape == (lateral, n)
    assert np.all(img[:, 3] == 0)            # None frame -> zero column
