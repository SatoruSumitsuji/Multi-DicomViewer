"""Regression tests for the XA window/level LUT sizing.

A series can mix integer widths — some vendors store an 8-bit and a 16-bit
copy of the same still under one SeriesUID. The shared W/L lookup table must
cover the WIDEST plane, or indexing a 16-bit frame into a table sized for an
8-bit plane (256 entries) throws IndexError and crashes the viewer. This was
the "6 files cannot be displayed / app crashes" report (CHEMAOU SAAD/XA).

Run:  python -m pytest tools/test_xa_wl_lut.py
"""
import numpy as np

from multi_dicomviewer.viewers.xa_viewer import _build_wl_lut


def test_wl_lut_covers_combined_uint8_uint16_range():
    # uint8 (0..255) + uint16 (0..65535) → table must span the full union.
    lut, off = _build_wl_lut(0, 65535, window=1400.0, level=700.0)
    assert off == 0
    assert lut.shape[0] == 65536          # large 16-bit values stay in-bounds
    # The crashing index from the real data (value 1403) must now be valid.
    assert 0 <= int(lut[1403]) <= 255
    assert int(lut[65535]) == 255         # above-window saturates to white


def test_wl_lut_uint8_only_is_256_entries():
    lut, off = _build_wl_lut(0, 255, window=255.0, level=127.5)
    assert off == 0
    assert lut.shape[0] == 256


def test_wl_lut_signed_int16_offset():
    # Signed data (e.g. int16 MR/PSIR) tables with a +32768 index offset.
    lut, off = _build_wl_lut(-32768, 32767, window=1000.0, level=0.0)
    assert off == 32768
    assert lut.shape[0] == 65536


def test_wl_lut_range_too_wide_returns_none():
    # Combined span > 65536 entries can't be tabled → caller uses apply_window.
    assert _build_wl_lut(-32768, 65535, 100.0, 0.0) is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
