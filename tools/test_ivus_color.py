"""Known-answer tests for the IVUS colour-display toggle (core side).

Run directly:    python tools/test_ivus_color.py
Or with pytest:  pytest tools/test_ivus_color.py

Covers the header-level colour decision (_is_color_capable / _is_color_ds),
the grayscale↔colour re-decode round-trip (apply_color_mode_to_planes), and
the per-series settings persistence. The GUI toggle itself (button + viewer
wiring) is exercised by running the app on real NIRS-IVUS data — these tests
pin the pure logic underneath it so a regression is caught headlessly.
"""
import os
import sys
import tempfile

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core import dicom_io, settings


# --------------------------------------------------------------------------
# Synthetic, UNCOMPRESSED multi-frame IVUS so pydicom.pixel_array decodes it
# natively (no codec dependency). PI=RGB, SamplesPerPixel=3, with a coloured
# square so the colour path has something to show.
# --------------------------------------------------------------------------
def _make_ivus_color_ds(nframes=4, h=16, w=16):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Modality = "IVUS"
    ds.Rows, ds.Columns = h, w
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.NumberOfFrames = nframes
    px = np.zeros((nframes, h, w, 3), dtype=np.uint8)
    # A saturated red block in the centre of every frame, mid-grey elsewhere.
    px[:, :, :, :] = 80
    px[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = (255, 0, 0)
    ds.PixelData = px.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def _make_plane(ds):
    """Build an XAPlane the way load_xa would for a DEFAULT (grayscale) IVUS:
    frame 0 decoded grayscale, the rest lazy."""
    nf = int(ds.NumberOfFrames)
    f0 = dicom_io._decode_frame(ds, 0)            # default → 2-D grayscale
    assert f0.ndim == 2, "IVUS must default to grayscale"
    vol = np.zeros((nf,) + f0.shape, dtype=f0.dtype)
    vol[0] = f0
    return dicom_io.XAPlane("IVUS", "mem", ds, nf, vol, is_color=False)


# --------------------------------------------------------------------------
# 1. Header decision: an IVUS RGB dataset is colour-CAPABLE but the default
#    decision still forces grayscale (the long-standing tint-avoidance rule).
# --------------------------------------------------------------------------
def test_header_decision():
    ds = _make_ivus_color_ds()
    assert dicom_io._is_color_capable(ds) is True
    assert dicom_io._is_color_ds(ds) is False         # IVUS → grayscale default

    # A non-IVUS RGB dataset is colour by default.
    ds.Modality = "XA"
    assert dicom_io._is_color_ds(ds) is True


# --------------------------------------------------------------------------
# 2. apply_color_mode_to_planes flips the plane to colour and back, reshaping
#    the volume and re-decoding frame 0 each way.
# --------------------------------------------------------------------------
def test_color_roundtrip():
    plane = _make_plane(_make_ivus_color_ds(nframes=4, h=16, w=16))
    assert plane.volume.ndim == 3 and not plane.is_color   # (F,H,W) grayscale

    achieved = dicom_io.apply_color_mode_to_planes([plane], True)
    assert achieved is True
    assert plane.is_color is True
    assert plane.volume.shape == (4, 16, 16, 3)
    assert plane.volume.dtype == np.uint8
    # Frame 0's centre is the red block; corner is the grey background.
    f0 = plane.frame(0)
    assert tuple(f0[8, 8]) == (255, 0, 0)
    assert f0[0, 0, 0] == f0[0, 0, 1] == f0[0, 0, 2] == 80
    # A lazily-decoded later frame is colour too (force_color persisted).
    f3 = plane.frame(3)
    assert f3.ndim == 3 and tuple(f3[8, 8]) == (255, 0, 0)

    # Back to grayscale: shape collapses, frame is 2-D again.
    achieved = dicom_io.apply_color_mode_to_planes([plane], False)
    assert achieved is False
    assert plane.is_color is False
    assert plane.volume.shape == (4, 16, 16)
    assert plane.frame(2).ndim == 2


# --------------------------------------------------------------------------
# 3. A genuinely monochrome IVUS cannot be forced to colour — the toggle
#    reports "no colour" (achieved False) so the viewer can revert the button.
# --------------------------------------------------------------------------
def test_monochrome_cannot_force_color():
    ds = _make_ivus_color_ds()
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.full((4, 16, 16), 120, np.uint8).tobytes()
    assert dicom_io._is_color_capable(ds) is False
    plane = _make_plane(ds)
    achieved = dicom_io.apply_color_mode_to_planes([plane], True)
    assert achieved is False
    assert plane.is_color is False
    assert plane.volume.ndim == 3


# --------------------------------------------------------------------------
# 4. Settings persistence: a saved colour choice round-trips per SeriesUID,
#    grayscale is the unstored default, and turning colour off removes it.
# --------------------------------------------------------------------------
def test_settings_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        settings.IVUS_COLOR_PATH = os.path.join(d, "ivus_color.json")
        from pathlib import Path
        settings.IVUS_COLOR_PATH = Path(settings.IVUS_COLOR_PATH)

        assert settings.load_ivus_color("UID-A") is False     # unset → default
        settings.save_ivus_color("UID-A", True)
        assert settings.load_ivus_color("UID-A") is True
        assert settings.load_ivus_color("UID-B") is False     # independent

        settings.save_ivus_color("UID-A", False)              # off → removed
        assert settings.load_ivus_color("UID-A") is False
        # Empty UID is a no-op both ways.
        settings.save_ivus_color("", True)
        assert settings.load_ivus_color("") is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
