"""Unit tests for core.image_quality (Angio/IVUS Smooth + Denoise).

Run:  python -m pytest tools/test_image_quality.py
  or: python tools/test_image_quality.py
"""
import numpy as np

from multi_dicomviewer.core import image_quality as iq


def test_denoise_preserves_shape_and_dtype_grayscale():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(64, 80), dtype=np.uint8)
    out = iq.denoise(frame)
    assert out is not None
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_denoise_preserves_shape_rgb():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=(48, 60, 3), dtype=np.uint8)
    out = iq.denoise(frame)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_denoise_reduces_noise_variance():
    # A flat field + noise: denoise should lower the variance (smooths speckle)
    # while keeping the mean roughly intact.
    if not iq.available():
        return                       # no cv2 → denoise is a no-op; skip
    rng = np.random.default_rng(2)
    base = np.full((128, 128), 128, np.uint8)
    noisy = np.clip(base + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8)
    out = iq.denoise(noisy)
    assert out.var() < noisy.var()
    assert abs(float(out.mean()) - float(noisy.mean())) < 5.0


def test_denoise_noop_on_non_uint8():
    frame = np.zeros((10, 10), np.float32)
    assert iq.denoise(frame) is frame   # unsupported dtype → unchanged


def test_lanczos_resize_target_size():
    frame = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    out = iq.lanczos_resize(frame, 100, 80)
    if iq.available():
        assert out is not None
        assert out.shape == (80, 100)   # (h, w)
        assert out.dtype == np.uint8
    else:
        assert out is None              # caller falls back to Qt


def test_lanczos_resize_rejects_bad_inputs():
    frame = np.zeros((16, 16), np.uint8)
    assert iq.lanczos_resize(frame, 0, 10) is None
    assert iq.lanczos_resize(None, 10, 10) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"\nall {len(fns)} tests passed (cv2 available={iq.available()})")
