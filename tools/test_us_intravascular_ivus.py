"""An older IVUS exported as plain Modality "US" (US Multi-frame, ImageType
value 3 = INTRAVASCULAR) is routed to the IVUS viewer — so it rotates, has the
long axis, etc. — like Modality "IVUS" (user 2026-11-24). Synthetic data only."""
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
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.core import dicom_io, settings  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.ui import main_window as mw  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
FAILS = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        FAILS.append(name)


# --- the mapping itself
F = Modality.from_dicom
check("IVUS → IVUS", F("IVUS") is Modality.IVUS)
check("US + INTRAVASCULAR → IVUS",
      F("US", ["ORIGINAL", "PRIMARY", "INTRAVASCULAR"]) is Modality.IVUS)
check("US + INTRAVASCULAR (4 values) → IVUS",
      F("US", ["ORIGINAL", "PRIMARY", "INTRAVASCULAR", "0001"]) is Modality.IVUS)
check("US (lower-case code) + intravascular → IVUS",
      F("us", ["DERIVED", "PRIMARY", "intravascular "]) is Modality.IVUS)
check("US, other body part (CARDIAC) stays OTHER",
      F("US", ["ORIGINAL", "PRIMARY", "CARDIAC"]) is Modality.OTHER)
check("US without ImageType stays OTHER", F("US") is Modality.OTHER)
check("US with a single-string ImageType", F("US", "INTRAVASCULAR")
      is Modality.IVUS)
check("XA with INTRAVASCULAR is still XA",
      F("XA", ["ORIGINAL", "PRIMARY", "INTRAVASCULAR"]) is Modality.XA)
check("CT stays CT", F("CT", ["ORIGINAL", "PRIMARY", "AXIAL"]) is Modality.CT)


# --- through a scanned file
def write_us(path, image_type):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientID = "TEST"
    ds.PatientName = "Test^Synthetic"
    ds.Modality = "US"
    if image_type is not None:
        ds.ImageType = image_type
    ds.NumberOfFrames = 4
    ds.Rows = ds.Columns = 32
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [0.02, 0.02]
    rng = np.random.default_rng(0)
    ds.PixelData = rng.integers(0, 255, (4, 32, 32), dtype=np.uint8).tobytes()
    ds.save_as(path, enforce_file_format=True)


d = tempfile.mkdtemp()
p_iv = os.path.join(d, "iv.dcm")
p_us = os.path.join(d, "us.dcm")
write_us(p_iv, ["ORIGINAL", "PRIMARY", "INTRAVASCULAR"])
write_us(p_us, ["ORIGINAL", "PRIMARY", "CARDIAC"])


def series_of(p):
    pats = dicom_io.index_files([p])
    return [s for pt in pats.values() for st in pt.studies.values()
            for s in st.series.values()][0]


si, su = series_of(p_iv), series_of(p_us)
check("scanned US/INTRAVASCULAR: modality IVUS, kind still shown as US",
      si.modality is Modality.IVUS and si.kind == "US", (si.modality, si.kind))
check("scanned US/CARDIAC: modality OTHER", su.modality is Modality.OTHER)
check("US/INTRAVASCULAR opens in the IVUS viewer",
      mw._VIEWER_FACTORY[si.modality] is mw.IVUSViewer)

v = mw.IVUSViewer()
v.resize(700, 500)
v.show()
v.load_series(dicom_io.load_xa(si, None), "t")
app.processEvents()
c = v.canvas
check("IVUS viewer: free rotation is on", bool(c._free_rot_on))
w, h = c.width(), c.height()
c._free_rot_drag = c._free_rot_angle(w * 0.8, h * 0.5)
now = c._free_rot_angle(w * 0.5, h * 0.2)
c._free_rot = (c._free_rot + (now - c._free_rot_drag)) % 360.0
check("a rotate drag turns the image", abs(c.free_rotation()) > 1,
      c.free_rotation())
v.close()

# --- right-click "Open as Intravascular Image" / "Open as Original Image"
import time  # noqa: E402

from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402

su_uid = su.series_uid
m = MainWindow()
m._patients = dicom_io.index_files([p_us, p_iv])
m._reindex_series_maps()
pane = m._panes[0]


def wait_load():
    t0 = time.time()
    while pane in m._loads and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


se = m._series_by_uid[su_uid]
m._open_series(se, pane)
wait_load()
v = pane.current_viewer()
check("US/CARDIAC opens in the XA (cine) viewer",
      type(v) is mw.XAViewer, type(v).__name__)
check("…its image right-click offers 'Open as Intravascular Image'",
      v.canvas.iv_menu_item == ("Open as Intravascular Image", "iv-open"),
      v.canvas.iv_menu_item)
v._on_export_image("iv-open")
wait_load()
v2 = pane.current_viewer()
check("after 'Open as Intravascular Image': the IVUS viewer, rotation on",
      type(v2) is mw.IVUSViewer and v2.canvas._free_rot_on,
      type(v2).__name__)
check("…the item turns into 'Open as Original Image'",
      v2.canvas.iv_menu_item == ("Open as Original Image", "iv-orig"),
      v2.canvas.iv_menu_item)
check("…remembered in the settings (not in the DICOM file)",
      su_uid in settings.load_intravascular_series()
      and settings.INTRAVASCULAR_PATH.exists()
      and str(settings.INTRAVASCULAR_PATH).startswith(HOME))
re = series_of(p_us)
check("…kept after a restart (a fresh scan opens it as intravascular)",
      re.modality is Modality.IVUS and re.iv_forced and re.kind == "US",
      (re.modality, re.iv_forced))
check("the DICOM file is untouched (still US / CARDIAC)",
      list(__import__("pydicom").dcmread(p_us).ImageType)[-1] == "CARDIAC")
v2._on_export_image("iv-orig")
wait_load()
v3 = pane.current_viewer()
check("'Open as Original Image': back in the XA viewer",
      type(v3) is mw.XAViewer, type(v3).__name__)
check("…the item is 'Open as Intravascular Image' again",
      v3.canvas.iv_menu_item == ("Open as Intravascular Image", "iv-open"))
check("…and forgotten in the settings",
      su_uid not in settings.load_intravascular_series()
      and series_of(p_us).modality is Modality.OTHER)
# never on a real IVUS (Modality IVUS / US+INTRAVASCULAR) or on XA
m._open_series(m._series_by_uid[si.series_uid], pane)
wait_load()
check("no item on a recognised intravascular image",
      pane.current_viewer().canvas.iv_menu_item is None)
xa_item = mw.XAViewer()
xa_item._set_iv_menu_item(type("L", (), {"modality": Modality.XA})())
check("no item on an XA angiogram", xa_item.canvas.iv_menu_item is None)
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
sys.stdout.flush()
os._exit(1 if FAILS else 0)
