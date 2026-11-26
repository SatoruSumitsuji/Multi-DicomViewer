# -*- coding: utf-8 -*-
"""The active pane's frame (user 2026-11-25): red 3 px normally; ORANGE and
twice as thick (6 px) while Measure is on — a type chosen or not (橙なら
Alt+, user 2026-11-26: the view tools then ALWAYS need Alt+drag; a plain
drag with Measure on and no type no longer runs them). The images never
shift (the frames always take 6 px).
Real VTK CTViewer on synthetic data (made-up patient).

    python tools/test_ct_frame_measure_alt.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
# The VTK viewer needs a real GL context (offscreen has none on Windows);
# nothing is shown.
os.environ["QT_QPA_PLATFORM"] = ("windows" if sys.platform == "win32"
                                 else os.environ.get("QT_QPA_PLATFORM", "offscreen"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
for nm in ("information", "warning", "critical"):
    setattr(QMessageBox, nm, staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Ok))
QMessageBox.question = staticmethod(
    lambda *a, **k: QMessageBox.StandardButton.Yes)
QMessageBox.exec = lambda self: 0
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.viewers import ct_viewer as ctv  # noqa: E402
from PyQt6.QtWidgets import QMenu  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


sx, sy, sz = 0.4, 0.4, 0.5
nz, ny, nx = 160, 200, 200
z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
X, Y, Z = x * sx, y * sy, z * sz
vol = np.full((nz, ny, nx), -80.0, np.float32)
cen = [(10 + t, 40, 30) for t in np.arange(0, 30, 0.2)]
cen += [(40 + 15 * np.sin(a), 40 + 15 * (1 - np.cos(a)), 30 + 6 * a)
        for a in np.arange(0, 2.0, 0.2 / 15)]
for c in cen:
    r = 1.5
    sl = (slice(int((c[2] - r - 1) / sz), int((c[2] + r + 1) / sz) + 1),
          slice(int((c[1] - r - 1) / sy), int((c[1] + r + 1) / sy) + 1),
          slice(int((c[0] - r - 1) / sx), int((c[0] + r + 1) / sx) + 1))
    m = (X[sl] - c[0]) ** 2 + (Y[sl] - c[1]) ** 2 + (Z[sl] - c[2]) ** 2 <= r * r
    vol[sl][m] = 380
vol += np.random.default_rng(0).normal(0, 25, vol.shape).astype(np.float32)
s = LoadedSeries(modality=Modality.CT, volume=vol.astype(np.int16), spacing_mm=(0.4, 0.4), cine_fps=None,
                 window=800.0, level=200.0, slice_mm=0.5, patient_basis=None, series_uid="live-uid")
v = ctv.CTViewer(); v.resize(1400, 800); v.load_series(s, "live"); v._set_mode("3D")
v.show()
for _ in range(10):
    app.processEvents()


def state():
    f = v._frames[v._active_pane]
    o = [k for k in v._frames if k != v._active_pane][0]
    cg = {k: v.pane[k].canvas.geometry().getRect() for k in ("A", "B")}
    return f.styleSheet(), v._frames[o].styleSheet(), cg


s0, o0, g0 = state()
check("normal: red 3 px", "3px solid #ff2020" in s0 and "transparent" in o0, s0)
v._meas_btn.setChecked(True)
v._toggle_measure()
s1, _o1, _g1 = state()
check("Measure on, no type yet: orange too (橙なら Alt+)",
      "6px solid #ff8c00" in s1, s1)

# a plain left-drag with Measure on (no type) no longer runs the view tool;
# Alt+drag does
from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402


def _drag(alt):
    cv = v.pane["B"].canvas
    mods = (Qt.KeyboardModifier.AltModifier if alt
            else Qt.KeyboardModifier.NoModifier)
    L = Qt.MouseButton.LeftButton
    a, b = QPointF(150, 150), QPointF(190, 175)
    cv.mousePressEvent(QMouseEvent(QMouseEvent.Type.MouseButtonPress, a,
                                   a, L, L, mods))
    cv.mouseMoveEvent(QMouseEvent(QMouseEvent.Type.MouseMove, b, b,
                                  Qt.MouseButton.NoButton, L, mods))
    cv.mouseReleaseEvent(QMouseEvent(QMouseEvent.Type.MouseButtonRelease, b,
                                     b, L, Qt.MouseButton.NoButton, mods))


_got = []
_orig_drag = v._drag
v._drag = lambda *a, **k: _got.append(a)
v._set_tool("MOVE")
_drag(False)
check("Measure on, no type: a plain drag runs no view tool", not _got, _got)
_drag(True)
check("Measure on, no type: Alt+drag runs it", bool(_got))
v._drag = _orig_drag
v._set_measure_type("polyline")
for _ in range(5):
    app.processEvents()
s2, o2, g2 = state()
check("measuring (Alt needed): orange, 6 px", "6px solid #ff8c00" in s2
      and "transparent" in o2, s2)
# (the height changes with the Measure bar shown below; the frame itself
# must not move the image: same x / y / width inside the frame)
check("the images do not shift (frame width constant)",
      all(g2[k][:3] == g0[k][:3] for k in g0), (g0, g2))
v._meas_btn.setChecked(False)
v._toggle_measure()
for _ in range(5):
    app.processEvents()
s3, _o3, g3 = state()
check("Measure off: red again", "3px solid #ff2020" in s3 and g3 == g0, (s3, g3))
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
app.processEvents()
os._exit(1 if fails else 0)
