# -*- coding: utf-8 -*-
"""Seek-bar Play-range ▽ (user 2026-11-23): right-clicking either range
triangle puts BOTH ends back at the very first / last frame. Shared by every
cine seek bar (XA, IVUS = XAViewer subclass). Synthetic frames only.

    python tools/test_range_marks_right_click.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.viewers.xa_viewer import XAViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


class _Plane:
    def __init__(self, n):
        self.volume = np.zeros((n, 8, 8), np.float32)
        self.total_frames = n


n = 20
v = XAViewer()
v.resize(800, 600)
v._planes = [_Plane(n)]
v.frame_slider.setRange(0, n - 1)
v._frame = 10
v._render = lambda *a, **k: None       # fake plane: no drawing
v.show()
app.processEvents()
rm = v._range_marks
rm.setVisible(True)
app.processEvents()


def set_range(a, b):
    v._range_start, v._range_end = a, b
    rm.set_bounds(a, b)


def click(val, button):
    x = int(round(rm._value_x(val)))
    QTest.mouseClick(rm, button, Qt.KeyboardModifier.NoModifier,
                     QPoint(x, rm.height() - 3))
    app.processEvents()


emitted = []
v.play_range_changed.connect(lambda *a: emitted.append(a))

for which, val in (("start", 5), ("end", 14)):
    set_range(5, 14)
    emitted.clear()
    click(val, Qt.MouseButton.RightButton)
    check(f"right-click {which} ▽ → full range",
          (v._range_start, v._range_end) == (0, n - 1)
          and (rm._start, rm._end) == (0, n - 1),
          (v._range_start, v._range_end, rm._start, rm._end))
    check(f"right-click {which} ▽ → shell told (MP4 range)",
          bool(emitted) and emitted[-1][1:] == (0, n - 1, n), emitted)

set_range(5, 14)
click(10, Qt.MouseButton.RightButton)           # far from both triangles
check("right-click away from the ▽s does nothing",
      (v._range_start, v._range_end) == (5, 14))

set_range(5, 14)
click(5, Qt.MouseButton.LeftButton)              # left click still grabs
check("left-click on a ▽ still only drags (range kept)",
      (v._range_start, v._range_end) == (5, 14))

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
os._exit(1 if fails else 0)
