"""A DICOM folder drop reads DICOM ONLY (2026-11-16): the analysis JSONs saved
beside the CT (.cpr.json / corotree / FullLv / MV・AoV・Apex・Epi・Blood) are
not swept in — wherever it is dropped (tree / window / an image pane). A
JSON-only analysis folder dropped on purpose is still loaded.

    python tools/test_drop_dicom_only.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


class Host:
    _load_paths = MainWindow._load_paths

    def __init__(self, dicom_dirs):
        self.dicom_dirs = set(dicom_dirs)
        self.swept = []
        self.indexed = []

    # helpers used by _load_paths
    def _is_case_presentation_json(self, p):
        return False

    _is_coronary_tree_json = _is_full_lv_json = _is_cpr_json = \
        _is_case_presentation_json

    def _is_lv_analysis_json(self, p):
        return False

    def _folder_holds_dicom(self, d):
        return d in self.dicom_dirs

    def _folder_cpr_files(self, d):
        self.swept.append(("cpr", d))
        return []                            # (an LV-only analysis folder)

    def _folder_lv_analysis_files(self, d):
        self.swept.append(("lv", d))
        return [os.path.join(d, "x.EpiLv.json")]

    def _open_coronary_tree(self):
        return None

    def _open_lv_analysis_files(self, files, pane):
        self.swept.append(("lv-open", tuple(files)))

    def _load_index(self, msg, fn, open_in_pane=None, spread_roots=None):
        self.indexed.append(open_in_pane)


with tempfile.TemporaryDirectory() as td:
    ct = os.path.join(td, "CT")
    js = os.path.join(td, "LV")
    os.makedirs(ct)
    os.makedirs(js)
    pane = object()
    for where, op in (("an image pane", pane), ("the tree", None)):
        h = Host([ct])
        h._load_paths([ct], open_in_pane=op)
        check(f"DICOM folder on {where}: analysis JSONs not swept", h.swept == [],
              str(h.swept))
        check(f"DICOM folder on {where}: DICOM indexed", h.indexed == [op])
    h = Host([])
    h._load_paths([js], open_in_pane=pane)
    check("JSON-only analysis folder still loads",
          any(k == "lv-open" for k, _ in h.swept) and h.indexed == [], str(h.swept))

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
