# -*- coding: utf-8 -*-
"""XA / IVUS (user 2026-11-22):
1. While the mouse-operation grid is ON, 🔍+ / 🔍− / 🔍⬚ and the drag Zoom /
   W/L modes are greyed out (and any armed one is turned off); Measure (which
   suspends the grid) or grid OFF brings them back.
2. With a measure tool armed, a plain CLICK on an existing edit point places a
   NEW point there (it used to grab the old point); a DRAG still re-edits it.

    python tools/test_grid_lock_and_point_on_point.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_home = tempfile.mkdtemp(prefix="mdv_home_")
os.environ["HOME"] = os.environ["USERPROFILE"] = _home
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.viewers.image_canvas import ImageCanvas  # noqa: E402
from multi_dicomviewer.viewers.xa_viewer import XAViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- 1. grid lock
v = XAViewer()
locked = [v._zoom_in_btn, v._zoom_out_btn, v._zoom_rect_btn,
          v._drag_zoom_btn, v._drag_wl_btn]
v._grid_btn.setChecked(False)
v._toggle_mouse_grid()
check("grid OFF: zoom / W/L buttons enabled", all(b.isEnabled() for b in locked))
v._drag_wl_btn.setChecked(True)
v._toggle_drag_wl()
v._grid_btn.setChecked(True)
v._toggle_mouse_grid()
check("grid ON: 🔍+ / 🔍− / 🔍⬚ / Zoom / W/L greyed out",
      not any(b.isEnabled() for b in locked))
check("grid ON: the armed W/L mode is turned off",
      not v._drag_wl_btn.isChecked() and v.canvas._drag_mode == "rotate")
check("grid ON: 🔍1 (reset) stays usable", v._zoom_reset_btn.isEnabled())
check("grid ON: Rt90° / Lt90° / Flip-H / Flip-V stay usable",
      len(v._orient_btns) == 4 and all(b.isEnabled() for b in v._orient_btns))
v._meas_btn.setChecked(True)
v._toggle_measure()
check("Measure ON (grid suspended): buttons usable again",
      all(b.isEnabled() for b in locked))
v._meas_btn.setChecked(False)
v._toggle_measure()
check("Measure OFF (grid back): greyed out again",
      not any(b.isEnabled() for b in locked))
v._grid_btn.setChecked(False)
v._toggle_mouse_grid()
check("grid OFF again: enabled", all(b.isEnabled() for b in locked))
check("greyed-out style rule present",
      "QPushButton:disabled" in v.styleSheet())

# ----------------------------------------------------- 2. point on a point
c = ImageCanvas()
c.resize(300, 300)
c.set_frame(np.full((100, 100), 120, np.uint8))
c.show()
app.processEvents()


def click(x, y):
    QTest.mouseClick(c, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, QPoint(x, y))


c.set_measure_type("line")
click(60, 60)
click(200, 60)
check("first line measured", len(c.measures) == 1, len(c.measures))
old = list(c.measures[0]["pts"])
# a new line starting ON the old line's end point (a click, no drag)
click(200, 60)
click(200, 220)
check("click on an old edit point starts a NEW line there",
      len(c.measures) == 2, len(c.measures))
check("the old line is untouched", list(c.measures[0]["pts"]) == old)
if len(c.measures) == 2:
    check("new line starts at the old point",
          c.measures[1]["pts"][0] == old[1], (c.measures[1]["pts"][0], old[1]))
# a drag on an edit point still re-edits it
QTest.mousePress(c, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                 QPoint(60, 60))
for k in range(1, 6):
    QTest.mouseMove(c, QPoint(60 + 6 * k, 60 + 6 * k))
QTest.mouseRelease(c, Qt.MouseButton.LeftButton,
                   Qt.KeyboardModifier.NoModifier, QPoint(90, 90))
check("drag on an edit point still moves it",
      c.measures[0]["pts"][0] != old[0] and len(c.measures) == 2,
      (old[0], c.measures[0]["pts"][0], len(c.measures)))
check("no stray draft left", c._draft is None)
# polyline: a vertex placed on an old point while drawing
c.set_measure_type("polyline")
click(30, 250)
click(200, 220)          # the second line's end point
click(250, 250)
check("polyline vertex can sit on an old point",
      c._draft is not None and len(c._draft["pts"]) == 3,
      c._draft and len(c._draft["pts"]))

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
