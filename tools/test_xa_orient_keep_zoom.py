# -*- coding: utf-8 -*-
"""XA / IVUS (user 2026-11-22):
- Rt90° / Lt90° / Flip-H / Flip-V keep the zoom (like WB reverse): the region
  that was magnified stays magnified and centred.
- Reset puts EVERYTHING back to the initial view: rotation / flip, free
  rotation, zoom / pan, WB reverse AND W/L.

    python tools/test_xa_orient_keep_zoom.py
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
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.viewers.image_canvas import ImageCanvas  # noqa: E402
from multi_dicomviewer.viewers.xa_viewer import XAViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


# 200 (w) x 100 (h) frame with one bright marker pixel block at x=150, y=30
frame = np.zeros((100, 200), np.uint8)
frame[28:33, 148:153] = 255
c = ImageCanvas()
c.resize(400, 400)
c.set_frame(frame)
app.processEvents()


def marker_at_centre():
    """The frame pixel shown at the widget centre (in RAW frame coords)."""
    x, y = c._widget_to_image_f(c.width() / 2.0, c.height() / 2.0)
    img = c._oriented(frame)
    xi, yi = int(x), int(y)
    return img[yi - 3:yi + 4, xi - 3:xi + 4].max() == 255


# zoom ×3 about the marker so it sits at the widget centre
mx, my = c._image_to_widget_f((150.5, 30.5))
c._apply_zoom(3.0, mx, my)
c._pan[0] += c.width() / 2.0 - c._image_to_widget_f((150.5, 30.5))[0]
c._pan[1] += c.height() / 2.0 - c._image_to_widget_f((150.5, 30.5))[1]
check("set-up: marker at the centre, zoom 3", marker_at_centre() and
      abs(c._zoom - 3.0) < 1e-6)
for kind in ("rt90", "rt90", "lt90", "fliph", "flipv", "lt90"):
    c.apply_orient(kind)
    check(f"{kind}: zoom kept", abs(c._zoom - 3.0) < 1e-6, c._zoom)
    check(f"{kind}: the magnified region stays centred", marker_at_centre(),
          c._widget_to_image_f(c.width() / 2.0, c.height() / 2.0))

# unzoomed: rotation keeps the plain fit
c.reset_zoom()
c.apply_orient("rt90")
check("no zoom: rotation stays at the fit",
      c._zoom == 1.0 and c._pan == [0.0, 0.0], (c._zoom, c._pan))

# ---------------------------------------------------- Reset = everything
v = XAViewer()
for cv in (v.canvas, v.canvas2):
    cv.resize(400, 400)
    cv.set_frame(frame)
    cv._apply_zoom(2.5, 300, 120)
    cv.apply_orient("rt90")
    cv.set_free_rotation(30.0)
v._wb_btn.setChecked(True)
v.win_slider.setEnabled(True)
v._window_init, v._level_init = 400.0, 200.0
called = []
_orig_reset_wl = v._reset_wl
v._reset_wl = lambda: (called.append(1), _orig_reset_wl())
v._wl_override = (900.0, 50.0)
v._reset_transform()
check("Reset: WB reverse off (button + canvases)",
      not v._wb_btn.isChecked() and not v.canvas._invert
      and not v.canvas2._invert)
check("Reset: W/L back to the series default (sticky override dropped)",
      called == [1] and v._wl_override is None)
for i, cv in enumerate((v.canvas, v.canvas2)):
    check(f"Reset canvas{i + 1}: zoom 1 / no pan",
          cv._zoom == 1.0 and cv._pan == [0.0, 0.0], (cv._zoom, cv._pan))
    check(f"Reset canvas{i + 1}: orientation back",
          cv._t_rot == 0 and not cv._t_flip)
    check(f"Reset canvas{i + 1}: free rotation back",
          abs(getattr(cv, "_free_rot", 0.0)) < 1e-9)

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
