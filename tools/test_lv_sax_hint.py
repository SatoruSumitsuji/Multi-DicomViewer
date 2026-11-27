# -*- coding: utf-8 -*-
"""「SAX」 guide (user 2026-11-27): once the Epi border is traced on all the
long-axis planes (6), the SAX button gets a bright frame so the next step is
found at once. No frame with a plane missing, in the Endo pass, or once SAX
is on; a removed plane takes it away again. Real VTK CTViewer + LVModel
(synthetic, no dialog); pygfx by source.

    python tools/test_lv_sax_hint.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
os.environ["QT_QPA_PLATFORM"] = ("windows" if sys.platform == "win32"
                                 else os.environ.get("QT_QPA_PLATFORM", "offscreen"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.core.lv_measure import LVModel  # noqa: E402
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


v = CTViewer()
m = LVModel(n_planes=6)
v._lv = {"model": m, "phase": "contour", "plane_idx": 0, "target": "epi",
         "pane": "B", "sax": None, "pass": "epi"}
b = v._lv_sax_btn


def framed():
    return v._LV_SAX_HINT_CSS in b.styleSheet()


v._lv_sax_hint_sync()
check("no Epi plane yet: plain SAX", not framed())
angs = m.plane_angles()
for a in angs[:-1]:
    m.epi_planes[a % 360.0] = np.zeros((3, 3))
v._lv_sax_hint_sync()
check("5 of 6 planes: still plain", not framed())
m.epi_planes[angs[-1] % 360.0] = np.zeros((3, 3))
v._lv_sax_hint_sync()
check("all 6 Epi planes traced: SAX framed", framed(), b.styleSheet())
check("… the checked (SAX on) colour is kept in the style",
      "QPushButton:checked{background:#b8860b" in b.styleSheet())
v._lv["sax"] = {"level": 0.0}
v._lv_sax_hint_sync()
check("in SAX: the frame goes", not framed())
v._lv["sax"] = None
v._lv_sax_hint_sync()
check("back on the long axis (all traced): framed again", framed())
m.epi_planes.pop(angs[2] % 360.0)
v._lv_sax_hint_sync()
check("a plane removed: plain", not framed())
m.epi_planes[angs[2] % 360.0] = np.zeros((3, 3))
v._lv["pass"] = "endo"
v._lv_sax_hint_sync()
check("Endo pass: no frame (Epi guide only)", not framed())
v._lv = None
v._lv_sax_hint_sync()
check("LV left: plain", not framed())

for fn in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    src = (ROOT / "multi_dicomviewer/viewers" / fn).read_text("utf-8")
    check(f"{fn}: synced on capture, plane removal and every button sync",
          src.count("self._lv_sax_hint_sync()") >= 3
          and "def _lv_sax_hint_due(self)" in src)
v.close()
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
