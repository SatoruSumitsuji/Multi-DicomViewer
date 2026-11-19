# -*- coding: utf-8 -*-
"""検査ごとの書き出し (user 2026-11-19): right-click a STUDY node of an XA or
IVUS / OCT / OFDI kind → Export study (DICOM / Anon DICOM / MP4 / CSV): every
un-hidden series of the study in one export, one folder per study
(<date>_<kind>_study, no patient name in it). CT and UCG (US) study nodes
do not offer it (CT stays per series; UCG is exported per series).

Synthetic DICOM files only (a made-up patient); dialogs / menus mocked.

    python tools/test_study_export.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pydicom  # noqa: E402
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402
from PyQt6.QtWidgets import (QApplication, QFileDialog, QMenu,  # noqa: E402
                             QMessageBox)

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
for nm in ("information", "warning", "critical"):
    setattr(QMessageBox, nm, staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Ok))
from multi_dicomviewer.core import dicom_io  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.ui import study_browser as sb  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


SRC = tempfile.mkdtemp(dir=HOME)
PAT = ("TEST^STUDYEXPORT", "TESTID01")


def write(folder, modality, study_uid, series_uid, sno, n=2, date="20250102"):
    os.makedirs(folder, exist_ok=True)
    for i in range(n):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.PatientName, ds.PatientID = PAT
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = modality
        ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = date
        ds.AcquisitionTime = "101500"
        ds.SeriesNumber = sno
        ds.InstanceNumber = i + 1
        ds.Rows = ds.Columns = 8
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = (np.arange(64, dtype=np.uint16) * (i + 1)).tobytes()
        ds.save_as(os.path.join(folder, f"{modality}_{sno}_{i}.dcm"),
                   enforce_file_format=True)


st_xa, st_ct, st_us, st_oct = (generate_uid() for _ in range(4))
for k in range(3):
    write(os.path.join(SRC, "xa", str(k)), "XA", st_xa, generate_uid(), k + 1)
write(os.path.join(SRC, "ct"), "CT", st_ct, generate_uid(), 1)
write(os.path.join(SRC, "us"), "US", st_us, generate_uid(), 1)
for k in range(2):
    write(os.path.join(SRC, "oct", str(k)), "OCT", st_oct, generate_uid(),
          k + 1, date="20250103")
patients = dicom_io.scan_folder(SRC)
tree = sb.StudyBrowser()
tree.populate(patients)


def study_item(uid):
    for (u, k), it in tree._study_items.items():
        if u == uid:
            return it, k
    return None, None


xa_ser = tree.series_in_study(*[(u, k) for (u, k) in tree._study_items
                                if u == st_xa][0])
tree.set_series_hidden([xa_ser[2]], True)      # one XA series Hidden

captured = {}


def menu_texts(uid):
    it, _k = study_item(uid)
    tree.clearSelection()
    tree.setCurrentItem(it)
    it.setSelected(True)
    acts = []

    def _exec(self, *a, **k):
        acts.extend(self.actions())
        return None
    old = QMenu.exec
    QMenu.exec = _exec
    try:
        tree._context_menu(tree.visualItemRect(it).center())
    finally:
        QMenu.exec = old
    return acts


acts = menu_texts(st_xa)
txt = [a.text() for a in acts]
want = [t("Export study (DICOM)"), t("Export study (Anon DICOM)"),
        t("Export study (MP4)"), t("Export study (CSV)")]
check("XA study: the 4 study exports in the menu (Hidden excluded)",
      all(any(x.startswith(w) for x in txt) for w in want)
      and any(t(" — {h} hidden excluded", h=1) in x for x in txt), txt[:6])
check("… alongside Select / Show / Close / Delete",
      any(x.startswith(t("Select (show unhide only)")) for x in txt)
      and any(x.startswith(t("Close (close list)")) for x in txt))
check("IVUS-like (OCT) study: offered",
      any(a.text().startswith(want[0]) for a in menu_texts(st_oct)))
check("CT study: not offered (stays per series)",
      not any(a.text().startswith(want[0]) for a in menu_texts(st_ct)))
check("UCG (US) study: not offered (per series)",
      not any(a.text().startswith(want[0]) for a in menu_texts(st_us)))
got = []
tree.export_requested.connect(lambda f, g: got.append((f, g)))
acts = menu_texts(st_xa)
next(a for a in acts if a.text().startswith(want[0])).trigger()
check("triggering → export request with the study's un-hidden series",
      got and got[-1][0] == "study-dicom" and len(got[-1][1]) == 1
      and [s.series_uid for s in got[-1][1][0]["series"]]
      == [s.series_uid for s in xa_ser[:2]], got and got[-1][0])
groups_xa = got[-1][1]

# ---- the shell runs it: one folder per study ------------------------------
from multi_dicomviewer.ui import export_dialog as edlg  # noqa: E402
from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402

OUT = tempfile.mkdtemp(dir=HOME)
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: OUT)
edlg.ExportDialog.exec = lambda self: edlg.ExportDialog.DialogCode.Accepted
edlg.ExportDialog.result_settings = lambda self: edlg.ExportSettings(
    fields=("series_no", "instance_no"))
mw = MainWindow()
mw._on_export_requested("study-dicom", groups_xa)
top = sorted(os.listdir(OUT))
check("DICOM: one study folder <date>_XA_study (no patient name)",
      top == ["20250102_XA_study"], top)
inner = sorted(os.listdir(os.path.join(OUT, top[0]))) if top else []
nfiles = sum(len(fs) for _d, _s, fs in os.walk(os.path.join(OUT, top[0]))) \
    if top else 0
check("… inside: the 2 un-hidden series, each its own folder, 2 files each",
      len(inner) == 2 and nfiles == 4, (inner, nfiles))
mw._on_export_requested("study-anon-dicom", groups_xa)
top = sorted(os.listdir(OUT))
anon = [d for d in top if d.endswith("_anon")]
dcm = [os.path.join(dp, f) for dp, _s, fs in os.walk(os.path.join(OUT, anon[0]))
       for f in fs] if anon else []
check("Anon DICOM: its own study folder …_study_anon, identifiers emptied",
      anon == ["20250102_XA_study_anon"] and len(dcm) == 4
      and all(str(pydicom.dcmread(p).PatientName) == "" for p in dcm), top)
mw._on_export_requested("study-csv", groups_xa)
top2 = sorted(os.listdir(OUT))
new = [d for d in top2 if d not in top]
csvs = os.listdir(os.path.join(OUT, new[0])) if new else []
check("CSV: a new study folder with one .csv per series",
      len(new) == 1 and len(csvs) == 2
      and all(c.endswith(".csv") for c in csvs), (new, csvs))
# two studies selected at once → two study folders
OUT2 = tempfile.mkdtemp(dir=HOME)
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: OUT2)
oct_ser = tree.series_in_study(*[(u, k) for (u, k) in tree._study_items
                                 if u == st_oct][0])
mw._on_export_requested("study-dicom", groups_xa + [
    {"kind": "OCT", "series": oct_ser, "hidden": 0}])
check("two studies → two study folders",
      sorted(os.listdir(OUT2)) == ["20250102_XA_study", "20250103_OCT_study"],
      sorted(os.listdir(OUT2)))
# per-series export unchanged
OUT3 = tempfile.mkdtemp(dir=HOME)
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: OUT3)
mw._on_export_requested("dicom", xa_ser[:1])
check("per-series Export (DICOM) unchanged (series folder directly)",
      len(os.listdir(OUT3)) == 1
      and not os.listdir(OUT3)[0].endswith("_study"), os.listdir(OUT3))

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
