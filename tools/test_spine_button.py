# -*- coding: utf-8 -*-
"""「Spine=」 (user 2026-11-27, right of Spin+): on an oblique section (e.g. an
LV short axis, thickness 0) roll the image so the patient's head is up and
feet down — the head-foot axis projected on the section vertical on screen.
Uses the patient basis (also a z-reversed stack). An axial-like section
(no head-foot direction in it) is left alone. Real VTK CTViewer, synthetic
data, no dialog.

    python tools/test_spine_button.py
"""
import math
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
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


s = LoadedSeries(modality=Modality.CT, volume=np.full((40, 50, 50), -100, np.int16),
                 spacing_mm=(0.7, 0.7), cine_fps=None, window=800.0, level=200.0,
                 slice_mm=0.7, patient_basis=None, series_uid="spine-ct")
v = CTViewer()
v.resize(1200, 700)
v.show()
v.load_series(s, "s")
v._set_mode("3D")
app.processEvents()


def oblique(spin_deg):
    """An LV-short-axis-like section (normal tilted 40° from the body axis),
    turned in-plane by *spin_deg*."""
    n = np.array([0.0, math.sin(math.radians(40)), math.cos(math.radians(40))])
    a = np.cross(n, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    r = math.radians(spin_deg)
    u = math.cos(r) * a + math.sin(r) * b
    w = np.cross(n, u)
    return v._ortho(u, w)


def foot_screen_angle(key):
    d = v._spine_foot_dir(key)
    ren = v.pane[key].ren
    ccx, ccy = v._cc(key)
    ren.SetWorldPoint(ccx, ccy, 0.5, 1.0)
    ren.WorldToDisplay()
    a0 = ren.GetDisplayPoint()
    ren.SetWorldPoint(ccx + d[0], ccy + d[1], 0.5, 1.0)
    ren.WorldToDisplay()
    a1 = ren.GetDisplayPoint()
    return math.degrees(math.atan2(a1[1] - a0[1], a1[0] - a0[0]))


check("Spine= button right of Spin+",
      v._spine_btn.text() == t("Spine=")
      and v._spine_btn.parentWidget() is v._spin_snap_btn.parentWidget())
g1 = v._spin_snap_btn.mapTo(v, v._spin_snap_btn.rect().topRight())
g2 = v._spine_btn.mapTo(v, v._spine_btn.rect().topLeft())
check("… directly after it (same row, next to it)",
      0 <= g2.x() - g1.x() <= 12 and abs(g2.y() - g1.y()) <= 4, (g1, g2))

for spin in (0.0, 37.0, 130.0, -100.0):
    v._frame["A"] = oblique(spin)
    v._thick["A"] = 0.0
    v._set_active("A")
    v._refresh()
    app.processEvents()
    v._spine_snap()
    app.processEvents()
    a = foot_screen_angle("A")
    check(f"in-plane turn {spin:+.0f}°: feet straight down after Spine=",
          abs(((a + 90.0 + 180.0) % 360.0) - 180.0) < 0.5, a)

# a second click changes nothing (already there)
before = v.pane["A"].ren.GetActiveCamera().GetViewUp()
v._spine_snap()
check("again: no change",
      np.allclose(before, v.pane["A"].ren.GetActiveCamera().GetViewUp(),
                  atol=1e-6))

# z-reversed stack (volume +z = patient FOOT): the image turns the other way
v._pbasis = np.diag([1.0, 1.0, -1.0])
d_rev = v._spine_foot_dir("A")
v._pbasis = np.eye(3)
d_std = v._spine_foot_dir("A")
check("patient basis used (z-reversed stack → opposite foot direction)",
      np.allclose(d_rev, -np.asarray(d_std), atol=1e-6), (d_rev, d_std))

# an axial section: no head-foot direction in it → left alone
ax = v._ortho(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
v._frame["A"] = ax
check("axial section: no direction → nothing to do",
      v._spine_foot_dir("A") is None)
cam = v.pane["A"].ren.GetActiveCamera()
vu = cam.GetViewUp()
v._spine_snap()
check("… the view is not rolled", np.allclose(vu, cam.GetViewUp()))

src = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text("utf-8")
check("pygfx: Spine= next to Spin+, same rule",
      'self._spine_btn = FitButton(t("Spine="))' in src
      and "def _spine_snap_pane(self, key)" in src
      and "def _spine_foot_dir(self, key)" in src)
v.close()
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
