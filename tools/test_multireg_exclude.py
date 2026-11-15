"""MultiReg 除外 (Exclude): while adding / editing a MultiReg point, a pane can
be left out (IVUS, CT short-axis or angio). The point then has no frame /
position for it, and that pane keeps following the points it IS part of.
Synthetic panes only (no DICOM); dialogs mocked.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer import i18n  # noqa: E402
from multi_dicomviewer.core import coreg  # noqa: E402
from multi_dicomviewer.ui import coreg_window as cw  # noqa: E402

i18n.set_language("ja")
fails = []
msgs = []
QMessageBox.information = staticmethod(lambda *a, **k: msgs.append(a[2] if len(a) > 2 else ""))


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


def make():
    w = cw.CoregWindow([])
    w.panes = []
    for i, (ivus, total) in enumerate(((True, 100), (True, 80), (False, 30),
                                       (False, 30))):
        p = cw._CoregPane(i, w)
        p.is_ivus, p.total, p.cur = ivus, total, 0
        p._stack = np.zeros((total, 8, 8), np.float32)
        p.is_color = True
        w.panes.append(p)
    w._active_ivus = 0
    verts = [(0.0, 0.0), (100.0, 0.0)]
    for i in (2, 3):
        w._guides[i] = {"vertices": verts, "curve": coreg.smooth_curve(verts),
                        "frame": 0}
    return w


w = make()
check("除外 hidden outside Add/edit/commit",
      all(p.excl_btn.isHidden() for p in w.panes))
w.panes[0].cur, w.panes[1].cur = 10, 20
w._start_placing()
check("Add: 除外 shown on every pane, all off",
      all(not p.excl_btn.isHidden() and not p.excl_btn.isChecked() for p in w.panes))
check("除外 sits right of WL in the pane row",
      w.panes[0].nav.indexOf(w.panes[0].excl_btn)
      == w.panes[0].nav.indexOf(w.panes[0].wl_btn) + 1)
w._pending_fracs = {2: 0.2, 3: 0.3}
w.panes[1].excl_btn.setChecked(True)          # IVUS #2 can't identify it
w.panes[3].excl_btn.setChecked(True)          # angio #2 neither
check("excluded angio loses its circle", 3 not in w._pending_fracs
      and w.panes[3].canvas.coreg_marker_pt is None)
check("除外 on = red outline", "border:2px solid" in w.panes[3].excl_btn.styleSheet())
w._commit_placing()
lm = w._landmarks[-1]
check("point uses only the non-excluded images",
      lm["frames"] == {0: 10} and lm["fracs"] == {2: 0.2}, str(lm))
check("除外 hidden + reset after commit",
      all(p.excl_btn.isHidden() and not p.excl_btn.isChecked() for p in w.panes))

# Edit that point: the excluded panes start excluded; put IVUS #2 back in.
w._selected_lm = 0
w.panes[1].cur = 33
w._start_placing()
check("Edit: panes not in the point start 除外",
      [p.excl_btn.isChecked() for p in w.panes] == [False, True, False, True])
w.panes[1].excl_btn.setChecked(False)
w.panes[3].excl_btn.setChecked(False)         # angio #2 back: circle returns
check("angio put back gets a circle again (interpolated / own spot)",
      w.panes[3].canvas.coreg_marker_pt is not None or 3 not in w._pending_fracs)
w.panes[2].excl_btn.setChecked(True)          # angio #1 out now
w._pending_fracs[3] = 0.6
w._commit_placing()
lm = w._landmarks[0]
check("Edit commit: IVUS #2 joins at its current frame, angio #1 dropped",
      lm["frames"] == {0: 10, 1: 33} and lm["fracs"] == {3: 0.6}, str(lm))

# Every angio excluded = an IVUS-only point (no 'place the circle' nag).
msgs.clear()
w._selected_lm = -1
w._start_placing()
for i in (2, 3):
    w.panes[i].excl_btn.setChecked(True)
w._commit_placing()
check("all angios 除外 → IVUS-only point, no warning",
      not msgs and w._landmarks[-1]["fracs"] == {}, str(msgs))

# Every IVUS excluded → refused.
msgs.clear()
w._start_placing()
for i in (0, 1):
    w.panes[i].excl_btn.setChecked(True)
n = len(w._landmarks)
w._commit_placing()
check("all IVUS 除外 → refused with a message", len(w._landmarks) == n and msgs)
w._cancel_placing()

check("i18n: 除外", i18n.t("Exclude") == "除外")

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
