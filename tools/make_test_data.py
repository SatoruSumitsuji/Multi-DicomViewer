"""Generate synthetic DICOM so Multi-DICOMviewer is runnable without patient data.

Creates ONE patient with BOTH modalities so the CT|XA compare layout and
the ⚹ CT+XA marker can be exercised:

  * a multiframe XA cine (moving synthetic "vessel")
  * a CT series (~60 axial slices of a heart-ish phantom in HU)

Usage:
    python tools/make_test_data.py [output-dir]      # default: ./sample_data
"""
from __future__ import annotations

import os
import sys

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    XRayAngiographicImageStorage,
    generate_uid,
)

PATIENT_ID = "XAV-TEST-001"
PATIENT_NAME = "TEST^Cardiac"
STUDY_UID = generate_uid()
STUDY_DATE = "20260515"


def _base(sop_class_uid: str, modality: str) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class_uid
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientID = PATIENT_ID
    ds.PatientName = PATIENT_NAME
    ds.StudyInstanceUID = STUDY_UID
    ds.StudyDate = STUDY_DATE
    ds.StudyDescription = "Multi-DICOMviewer synthetic test"
    ds.Modality = modality
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def make_xa(out_dir: str, frames: int = 40, size: int = 512) -> None:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    vol = np.full((frames, size, size), 40.0, dtype=np.float32)

    for f in range(frames):
        # A curved "vessel" that pulses/shifts across the cine.
        t = f / frames * 2 * np.pi
        cx = size / 2 + 60 * np.sin(t)
        path_y = np.linspace(40, size - 40, 400)
        path_x = cx + 90 * np.sin(path_y / size * 3.5 * np.pi)
        radius = 5 + 2 * np.sin(t * 2)
        for py, px in zip(path_y, path_x):
            d2 = (xx - px) ** 2 + (yy - py) ** 2
            vol[f][d2 < radius ** 2] = 210.0
    vol += np.random.normal(0, 4, vol.shape)
    px8 = np.clip(vol, 0, 255).astype(np.uint8)

    ds = _base(XRayAngiographicImageStorage, "XA")
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 2
    ds.SeriesDescription = "LCA cine (synthetic)"
    ds.Rows, ds.Columns = size, size
    ds.NumberOfFrames = frames
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.ImagerPixelSpacing = [0.184, 0.184]
    ds.PixelSpacing = [0.184, 0.184]
    ds.CineRate = 15
    ds.FrameTime = 1000.0 / 15
    ds.PixelData = px8.tobytes()

    path = os.path.join(out_dir, "xa_cine.dcm")
    ds.save_as(path, enforce_file_format=True)
    print(f"  XA  -> {path}  ({frames} frames, {size}x{size})")


def make_xa_biplane(out_dir: str, frames: int = 40, size: int = 512) -> None:
    """One XA series with TWO instances (Front + Lateral) so the browser
    shows [2 img] and the biplane layout/Front-Lateral toggle can be tested."""
    series_uid = generate_uid()
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    # (instance label, PositionerPrimaryAngle, vessel-shape phase)
    views = (("Front", 30.0, 0.0), ("Lateral", 90.0, np.pi / 2))
    for label, primary, phase in views:
        vol = np.full((frames, size, size), 40.0, dtype=np.float32)
        for f in range(frames):
            t = f / frames * 2 * np.pi
            cx = size / 2 + 60 * np.sin(t + phase)
            path_y = np.linspace(40, size - 40, 400)
            path_x = cx + 90 * np.sin(path_y / size * 3.5 * np.pi + phase)
            radius = 5 + 2 * np.sin(t * 2)
            for py, px in zip(path_y, path_x):
                d2 = (xx - px) ** 2 + (yy - py) ** 2
                vol[f][d2 < radius ** 2] = 210.0
        vol += np.random.normal(0, 4, vol.shape)
        px8 = np.clip(vol, 0, 255).astype(np.uint8)

        ds = _base(XRayAngiographicImageStorage, "XA")
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = 3
        ds.SeriesDescription = "Biplane cine (synthetic)"
        ds.ViewName = label
        ds.PositionerPrimaryAngle = primary
        ds.PositionerSecondaryAngle = 0.0
        ds.Rows, ds.Columns = size, size
        ds.NumberOfFrames = frames
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.ImagerPixelSpacing = [0.184, 0.184]
        ds.PixelSpacing = [0.184, 0.184]
        ds.CineRate = 15
        ds.FrameTime = 1000.0 / 15
        ds.PixelData = px8.tobytes()

        path = os.path.join(out_dir, f"xa_biplane_{label.lower()}.dcm")
        ds.save_as(path, enforce_file_format=True)
        print(f"  XA  -> {path}  (biplane {label}, {frames} frames)")


def make_ct(out_dir: str, slices: int = 60, size: int = 256) -> None:
    series_uid = generate_uid()
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = size / 2

    ct_dir = os.path.join(out_dir, "ct")
    os.makedirs(ct_dir, exist_ok=True)

    for i in range(slices):
        hu = np.full((size, size), -500.0, dtype=np.float32)  # lung-ish bg
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        # Heart blood pool (soft tissue) shrinking toward apex.
        heart_r = 70 - 0.6 * abs(i - slices / 2)
        hu[r < heart_r] = 45.0
        # Myocardium ring.
        hu[(r < heart_r) & (r > heart_r - 12)] = 90.0
        # Contrast-filled coronary moving with slice index.
        ang = i / slices * 2 * np.pi
        ox = cx + 45 * np.cos(ang)
        oy = cy + 45 * np.sin(ang)
        cor = np.sqrt((xx - ox) ** 2 + (yy - oy) ** 2)
        hu[cor < 4] = 350.0
        # A rib-like bone arc.
        hu[(r < size / 2 - 6) & (r > size / 2 - 12)] = 700.0
        hu += np.random.normal(0, 8, hu.shape)

        stored = np.clip(hu + 1024.0, 0, 4095).astype(np.uint16)

        ds = _base(CTImageStorage, "CT")
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = 1
        ds.SeriesDescription = "CCTA (synthetic)"
        ds.InstanceNumber = i + 1
        ds.Rows, ds.Columns = size, size
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        ds.PixelSpacing = [0.4, 0.4]
        ds.SliceThickness = 1.0
        ds.ImagePositionPatient = [-51.2, -51.2, float(i)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelData = stored.tobytes()

        ds.save_as(
            os.path.join(ct_dir, f"ct_{i:03d}.dcm"),
            enforce_file_format=True,
        )
    print(f"  CT  -> {ct_dir}  ({slices} slices, {size}x{size}, HU)")


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_data"
    os.makedirs(out, exist_ok=True)
    print(f"Generating synthetic DICOM in: {os.path.abspath(out)}")
    make_xa(out)
    make_xa_biplane(out)
    make_ct(out)
    print("Done. Run:  python run.py", os.path.abspath(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
