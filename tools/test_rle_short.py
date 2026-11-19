# -*- coding: utf-8 -*-
"""RLE Lossless written one byte short per segment (seen from an ultrasound
device: 147455 vs 147456 bytes) — pydicom refuses the file ("Unable to
decode as exceptions were raised by all available plugins"); the viewer
decodes it leniently (the missing byte padded). Synthetic files only.

    python tools/test_rle_short.py
"""
import os
import struct
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pydicom  # noqa: E402
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.encaps import encapsulate  # noqa: E402
from pydicom.uid import RLELossless, generate_uid  # noqa: E402

from multi_dicomviewer.core import dicom_io  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


def packbits_literal(b: bytes) -> bytes:
    out = bytearray()
    for i in range(0, len(b), 128):
        chunk = b[i:i + 128]
        out.append(len(chunk) - 1)
        out += chunk
    return bytes(out)


def rle_frame(planes, short=1):
    segs = [packbits_literal(p.tobytes()[:len(p.tobytes()) - short])
            for p in planes]
    offs, pos = [], 64
    for s in segs:
        offs.append(pos)
        pos += len(s) + (len(s) & 1)
    hdr = struct.pack("<16I", len(segs), *(offs + [0] * (15 - len(offs))))
    return hdr + b"".join(s + (b"\0" if len(s) & 1 else b"") for s in segs)


def make(path, frames, spp=1, bits=8, palette=False):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = RLELossless
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientName, ds.PatientID = "TEST^RLE", "TESTRLE"
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.Modality = "US"
    ds.Rows, ds.Columns = frames[0].shape[:2]
    ds.NumberOfFrames = len(frames)
    ds.SamplesPerPixel = spp
    ds.BitsAllocated = ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.PixelRepresentation = 0
    if spp == 3:
        ds.PlanarConfiguration = 0
    ds.PhotometricInterpretation = ("PALETTE COLOR" if palette else
                                    ("RGB" if spp == 3 else "MONOCHROME2"))
    if palette:
        lut = np.arange(256, dtype=np.uint16) * 257
        for c, arr in (("Red", lut), ("Green", lut[::-1].copy()),
                       ("Blue", lut)):
            setattr(ds, f"{c}PaletteColorLookupTableDescriptor", [256, 0, 16])
            setattr(ds, f"{c}PaletteColorLookupTableData", arr.tobytes())
    enc = []
    for f in frames:
        if spp == 3:
            planes = [f[..., k] for k in range(3)]
        elif bits == 16:
            planes = [(f >> 8).astype(np.uint8), (f & 255).astype(np.uint8)]
        else:
            planes = [f]
        enc.append(rle_frame(planes))
    ds.PixelData = encapsulate(enc)
    ds["PixelData"].VR = "OB"
    ds.save_as(path, enforce_file_format=True)
    return path


tmp = tempfile.mkdtemp(dir=HOME)
rng = np.random.default_rng(0)
fr8 = [rng.integers(0, 255, (24, 20), dtype=np.uint8) for _ in range(5)]
p = make(os.path.join(tmp, "pal.dcm"), fr8, palette=True)
ds = pydicom.dcmread(p, force=True)
try:
    ds.pixel_array
    refused = False
except Exception as e:                                   # noqa: BLE001
    refused = "decode" in str(e)
check("pydicom itself refuses the short RLE (the reported error)", refused)
ds = pydicom.dcmread(p, force=True)
a = dicom_io._decode_frame(ds, 3)
exp = fr8[3].astype(int)
check("palette 8-bit: decoded to RGB, pixels right except the padded last",
      a.shape == (24, 20, 3)
      and np.array_equal(a[..., 1].astype(int).ravel()[:-1],
                         (255 - exp).ravel()[:-1]), a.shape)
check("every frame decodes",
      all(dicom_io._decode_frame(ds, i).shape == (24, 20, 3)
          for i in range(5)))
fr16 = [rng.integers(0, 4000, (16, 12), dtype=np.uint16) for _ in range(3)]
ds = pydicom.dcmread(make(os.path.join(tmp, "g16.dcm"), fr16, bits=16),
                     force=True)
g = dicom_io._raw_frame(ds, 2)
check("16-bit grey: both byte planes recombined",
      g.shape == (16, 12)
      and np.array_equal(g.ravel()[:-1], fr16[2].ravel()[:-1]), g.dtype)
frc = [rng.integers(0, 255, (10, 8, 3), dtype=np.uint8) for _ in range(2)]
ds = pydicom.dcmread(make(os.path.join(tmp, "rgb.dcm"), frc, spp=3),
                     force=True)
c = dicom_io._raw_frame(ds, 1)
check("RGB: 3 segments → (H, W, 3)",
      c.shape == (10, 8, 3)
      and np.array_equal(c.reshape(-1, 3)[:-1], frc[1].reshape(-1, 3)[:-1]))
pts = dicom_io.index_files([p])
ser = [se for pt in pts.values() for st in pt.studies.values()
       for se in st.series.values()]
ls = dicom_io.load_series(ser[0])
check("the series loads in the app (was the error dialog)",
      ls is not None and ls.is_color and ls.volume is not None)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
