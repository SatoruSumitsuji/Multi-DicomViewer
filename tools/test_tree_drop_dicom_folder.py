"""A DICOM folder dropped on the Studies tree / window is IMPORTED ONLY, even
when LV-analysis / .cpr.json files were saved inside it — no pane opens and no
partial LV is applied (that raised the "MV・AoV・Apex …" warning on a plain
drop). JSON-only analysis folders and analysis FILE drops behave as before. A
pane drop of a CT folder is DICOM-only too (2026-11-16: analysis data loads only
when the user asks for it). Synthetic DICOM only.
"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pydicom  # noqa: E402
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

from multi_dicomviewer.ui import main_window as mw  # noqa: E402

M = mw.MainWindow
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


def write_dcm(path):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.Modality = "CT"
    ds.save_as(path, enforce_file_format=True)


root = tempfile.mkdtemp(prefix="mdv_drop_")
ct = os.path.join(root, "study", "CT_3D")             # DICOM + saved LV/CPR json
os.makedirs(ct)
write_dcm(os.path.join(ct, "IM0001"))
for nm in ("case.epi.json", "case.bld.json", "LAD.cpr.json"):
    open(os.path.join(ct, nm), "w").write("{}")
ana = os.path.join(root, "analysis")                 # JSON-only analysis folder
os.makedirs(ana)
for nm in ("case.epi.json", "case.mv.json"):
    open(os.path.join(ana, nm), "w").write("{}")
cpr_only = os.path.join(root, "vessels")             # JSON-only CPR folder
os.makedirs(cpr_only)
open(os.path.join(cpr_only, "RCA.cpr.json"), "w").write("{}")


def make():
    calls = SimpleNamespace(lv=[], cpr=[], index=[], tree=0)
    o = SimpleNamespace(_last_dir="")
    for name in ("_load_paths", "_folder_lv_analysis_files"):
        setattr(o, name, M.__dict__[name].__get__(o))
    o._folder_holds_dicom = M._folder_holds_dicom
    o._folder_cpr_files = M._folder_cpr_files
    no = lambda p: False                                   # noqa: E731
    o._is_case_presentation_json = o._is_coronary_tree_json = no
    o._is_full_lv_json = o._is_cpr_json = no
    o._is_lv_analysis_json = lambda p: p.endswith((".epi.json", ".bld.json",
                                                    ".mv.json"))
    o._open_lv_analysis_files = lambda paths, pane=None: calls.lv.append(
        (sorted(os.path.basename(p) for p in paths), pane))

    class TW:
        def _load_cpr_paths(self, cprs):
            calls.cpr.append(sorted(os.path.basename(p) for p in cprs))

        def _set_overlay(self, on):
            pass

    def open_tree():
        calls.tree += 1
        return TW()
    o._open_coronary_tree = open_tree
    o._load_index = lambda msg, fn, open_in_pane=None, spread_roots=None: \
        calls.index.append(open_in_pane)
    return o, calls


check("_folder_holds_dicom: CT folder / its parent yes, JSON-only no",
      M._folder_holds_dicom(ct) and M._folder_holds_dicom(os.path.join(root, "study"))
      and not M._folder_holds_dicom(ana))

o, c = make()
o._load_paths([ct])                                  # tree / window drop
check("tree drop of a CT folder: imported (scanned), not opened in a pane",
      c.index == [None], str(c.index))
check("tree drop of a CT folder: saved LV json NOT applied", c.lv == [], str(c.lv))
check("tree drop of a CT folder: saved .cpr.json NOT loaded",
      c.cpr == [] and c.tree == 0, str(c.cpr))

o, c = make()
o._load_paths([os.path.join(root, "study")])          # parent folder too
check("tree drop of the study folder: import only", c.index == [None]
      and c.lv == [] and c.cpr == [])

o, c = make()
o._load_paths([ana])                                 # JSON-only LV folder
check("JSON-only LV folder: still applied like before",
      c.lv and c.lv[0][0] == ["case.epi.json", "case.mv.json"]
      and c.index == [], str(c.lv))
o, c = make()
o._load_paths([cpr_only])                            # JSON-only CPR folder
check("JSON-only CPR folder: still loaded like before",
      c.cpr == [["RCA.cpr.json"]] and c.index == [], str(c.cpr))

o, c = make()
pane = object()
o._load_paths([ct], pane)                            # dropped ON a pane
check("pane drop of a CT folder: DICOM only (no .cpr / LV sweep)",
      c.cpr == [] and c.lv == [] and c.index == [pane], str((c.cpr, c.lv, c.index)))

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
