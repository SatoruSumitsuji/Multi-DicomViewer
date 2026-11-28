# -*- coding: utf-8 -*-
"""全冠動脈描出 button (user 2026-11-28: 「何も起こりません」): a click on the
real VTK viewer inside the real MainWindow opens the ostium window over the
CT pane; an early stop is never silent (box + status bar + console).
Synthetic CT, no patient data.

    python tools/test_autotree_button_click.py
"""
import os, sys, tempfile, traceback
HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
os.environ["QT_QPA_PLATFORM"] = "windows"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox
app = QApplication.instance() or QApplication([])
sys.excepthook = lambda *a: traceback.print_exception(*a)
from multi_dicomviewer.core import settings
assert str(settings.SETTINGS_DIR).startswith(HOME)
boxes = []
for nm in ("information", "warning", "critical", "question"):
    setattr(QMessageBox, nm, staticmethod(lambda *a, **k: (boxes.append(a), QMessageBox.StandardButton.Ok)[1]))
from multi_dicomviewer.core.dicom_io import LoadedSeries
from multi_dicomviewer.core.study_model import Modality
from multi_dicomviewer.viewers import ct_viewer as ctv
from multi_dicomviewer.ui.main_window import MainWindow
m = MainWindow()
m.show()
vol = np.random.default_rng(0).normal(0, 25, (80, 100, 100)).astype(np.int16)
s = LoadedSeries(modality=Modality.CT, volume=vol, spacing_mm=(0.5, 0.5),
                 cine_fps=None, window=800.0, level=200.0, slice_mm=0.5,
                 patient_basis=None, series_uid="r-uid")
v = ctv.CTViewer()
v.load_series(s, "r")
v._set_mode("3D")
pane = m._panes[0]
print("pane type", type(pane))
v.setParent(pane)
v.show()
for _ in range(5):
    app.processEvents()
b = v._cpr_autotree_btn
print("btn visible", b.isVisible(), "enabled", b.isEnabled(), "shell", v._ct_mode_shell())
print("pending", getattr(v, "_coronary_mpr_pending", None))
b.click()
for _ in range(5):
    app.processEvents()
d = getattr(m, "_autotree_dlg", None)
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


check("click → the ostium window is shown", d is not None and d.isVisible(),
      d.geometry() if d else None)
if d is not None:
    c = v.mapToGlobal(v.rect().center())
    check("… over the CT pane", d.geometry().contains(c), (d.geometry(), c))
    d.close()
# CoronaryTreeモード arms Draw at once: the button stays usable, the empty
# Draw is disarmed and the ostium window opens (user 2026-11-28)
v._coronary_mode = True
v._coronary_mpr_pending = True
v._draft = None
v._coronary_sync_ui()
check("Draw armed (nothing traced): the button stays enabled", b.isEnabled())
m._autotree_dlg = None
b.click()
for _ in range(3):
    app.processEvents()
d = getattr(m, "_autotree_dlg", None)
check("… click → Draw disarmed + the ostium window", d is not None
      and d.isVisible() and not v._coronary_mpr_pending and not boxes, boxes)
if d is not None:
    d.close()
# a line half drawn → told to finish it first (not lost)
v._coronary_mpr_pending = True
v._draft = {"type": "polyline", "pane": "A", "pts": [np.zeros(3), np.ones(3)]}
v._coro_auto_tree()
check("line in progress → told in a box, Draw kept", len(boxes) == 1
      and v._coronary_mpr_pending, boxes)
v._coronary_mpr_pending = False
v._draft = None
m._autotree_dlg = None
v2 = ctv.CTViewer()
v2.setParent(pane)
v2._coro_auto_tree()
check("no CT → told in a box (not silent)", len(boxes) == 2, boxes[-1:])
print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(0 if not fails else 1)
