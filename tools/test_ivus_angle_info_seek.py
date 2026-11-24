"""IVUS: the "Angle keyframes — …" line (Multi-Rot Angle Set) sits in the
below-image column with the readout, so it no longer squeezes the transport row
and the seek-bar handle stays fully visible (user 2026-11-24). Synthetic data."""
import os
import sys
import tempfile

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402
from PyQt6.QtCore import QPoint  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.core import dicom_io, settings  # noqa: E402
from multi_dicomviewer.viewers.ivus_viewer import IVUSViewer  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
FAILS = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        FAILS.append(name)


d = tempfile.mkdtemp()
p = os.path.join(d, "ivus.dcm")
meta = FileMetaDataset()
meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
meta.MediaStorageSOPInstanceUID = generate_uid()
meta.TransferSyntaxUID = ExplicitVRLittleEndian
ds = FileDataset(p, {}, file_meta=meta, preamble=b"\0" * 128)
ds.SOPClassUID = meta.MediaStorageSOPClassUID
ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
ds.StudyInstanceUID = generate_uid()
ds.SeriesInstanceUID = generate_uid()
ds.PatientID = "TEST"
ds.PatientName = "Test^Synthetic"
ds.Modality = "IVUS"
ds.NumberOfFrames = 40
ds.Rows = ds.Columns = 64
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"
ds.BitsAllocated = ds.BitsStored = 8
ds.HighBit = 7
ds.PixelRepresentation = 0
ds.PixelSpacing = [0.02, 0.02]
ds.PixelData = np.random.default_rng(0).integers(
    0, 255, (40, 64, 64), dtype=np.uint8).tobytes()
ds.save_as(p, enforce_file_format=True)

pats = dicom_io.index_files([p])
se = [s for pt in pats.values() for st in pt.studies.values()
      for s in st.series.values()][0]
v = IVUSViewer()
v.resize(1100, 800)
v.show()
v.load_series(dicom_io.load_xa(se, None), "t")
app.processEvents()


def handle_fully_visible():
    sl = v.frame_slider
    sc = v._below_scroll
    vp = sc.viewport()
    top = sl.mapTo(vp, QPoint(0, 0)).y()
    return (top >= 0 and top + sl.height() <= vp.height()
            and sl.height() >= sl.sizeHint().height()), (
        top, sl.height(), sl.sizeHint().height(), vp.height())


for long_view in (False, True):
    if long_view and hasattr(v, "_toggle_long_axis"):
        v._toggle_long_axis()
    v._angle_kf.clear()
    v._refresh_angle_marks()
    app.processEvents()
    ok0, info0 = handle_fully_visible()
    v._angle_kf[5] = 65.0
    v._angle_kf[20] = 30.0
    v._refresh_angle_marks()
    app.processEvents()
    app.processEvents()
    tag = "long view" if long_view else "cross-section"
    check(f"{tag}: the angle-keyframe line is shown",
          v._angle_info.isVisible() and "F6" in v._angle_info.text())
    check(f"{tag}: it lives in the below-image column (with the readout)",
          v._angle_info.parentWidget() is v.readout.parentWidget())
    ok1, info1 = handle_fully_visible()
    check(f"{tag}: seek bar fully visible without keyframes", ok0, info0)
    check(f"{tag}: seek bar still fully visible with the keyframe line",
          ok1, info1)
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
sys.stdout.flush()
os._exit(1 if FAILS else 0)
