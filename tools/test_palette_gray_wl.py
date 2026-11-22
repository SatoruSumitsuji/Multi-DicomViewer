# -*- coding: utf-8 -*-
"""PALETTE COLOR that is really grayscale (user 2026-11-22: an IVUS exported
as US by the scanner with a Windows-style palette — a grey ramp plus a few
system colours no pixel uses — showed fine but W/L could not be adjusted,
because it was treated as a colour series). Such a dataset now decodes to grey
levels (2-D float), so W/L works; a palette image whose pixels DO use colour
entries stays colour, and the IVUS colour toggle (force_color) still gets the
palette colours. Synthetic files only.

    python tools/test_palette_gray_wl.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

from multi_dicomviewer.core import dicom_io  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


def make(frames: np.ndarray) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.Modality = "US"
    ds.Rows, ds.Columns = frames.shape[1:]
    ds.NumberOfFrames = frames.shape[0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "PALETTE COLOR"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    # 16-bit palette: a NON-monotone grey ramp (entry i → grey (255 - i))
    # with system colours at 1..4 and 250..253 (red, green, blue, yellow)
    r = np.array([(255 - i) * 257 for i in range(256)], np.uint16)
    g, b = r.copy(), r.copy()
    for i, (cr, cg, cb) in zip((1, 2, 3, 4, 250, 251, 252, 253),
                               [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                                (255, 255, 0)] * 2):
        r[i], g[i], b[i] = cr * 257, cg * 257, cb * 257
    for name, arr in (("Red", r), ("Green", g), ("Blue", b)):
        setattr(ds, f"{name}PaletteColorLookupTableDescriptor", [256, 0, 16])
        setattr(ds, f"{name}PaletteColorLookupTableData", arr.tobytes())
    ds.PixelData = frames.astype(np.uint8).tobytes()
    return ds


rng = np.random.default_rng(0)
grey_idx = np.array([0] + list(range(5, 250)) + [254, 255], np.uint8)
frames = grey_idx[rng.integers(0, len(grey_idx), (3, 16, 16))]
ds = make(frames)
f = dicom_io._decode_frame(ds, 1)
check("grey-only palette image → 2-D grey levels (W/L usable)",
      f.ndim == 2 and f.dtype == np.float32, (f.shape, f.dtype))
check("grey level = the palette's grey, not the raw index",
      np.allclose(f, 255.0 - frames[1].astype(np.float32), atol=0.6),
      float(np.abs(f - (255.0 - frames[1])).max()))
fc = dicom_io._decode_frame(ds, 1, force_color=True)
check("IVUS colour toggle (force_color) still gives the palette colours",
      fc.ndim == 3 and fc.shape[-1] == 3, fc.shape)

col = frames.copy()
col[0, 0, :4] = [1, 2, 3, 4]                  # pixels on colour entries
ds2 = make(col)
f2 = dicom_io._decode_frame(ds2, 0)
check("palette image that USES colours stays colour", f2.ndim == 3, f2.shape)
f2b = dicom_io._decode_frame(ds2, 1)
check("…decided once per series (every frame colour)", f2b.ndim == 3)

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
