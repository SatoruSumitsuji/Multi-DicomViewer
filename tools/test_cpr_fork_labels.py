# -*- coding: utf-8 -*-
"""Auto-Track stopped at a fork (user 2026-11-25): the candidates 1, 2, …
are numbered beside their lumens on the right (long-axis) pane AND the left
short axis; the numbers are re-placed from 3-D, so they stay on the branch
when the right pane is rotated. The choice dialog is modeless (the panes can
be turned meanwhile) and the numbers go when it is answered.

Real VTK CTViewer on synthetic data (made-up patient); Mac painter via a
stub viewer.

    python tools/test_cpr_fork_labels.py
"""
import math
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
os.environ["QT_QPA_PLATFORM"] = ("windows" if sys.platform == "win32"
                                 else os.environ.get("QT_QPA_PLATFORM", "offscreen"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
for nm in ("information", "warning", "critical"):
    setattr(QMessageBox, nm, staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Ok))
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.ui import autotrack_progress as atp  # noqa: E402
from multi_dicomviewer.viewers import ct_viewer as ctv  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


h = 0.5
nz, ny, nx = 100, 120, 120
z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32) * h
vol = np.full((nz, ny, nx), -80.0, np.float32)
vol[(y - 30) ** 2 + (z - 25) ** 2 <= 1.5 ** 2] = 380
s = LoadedSeries(modality=Modality.CT, volume=vol.astype(np.int16),
                 spacing_mm=(h, h), cine_fps=None, window=800.0, level=200.0,
                 slice_mm=h, patient_basis=None, series_uid="fork-uid")
v = ctv.CTViewer()
v.resize(1400, 800)
v.load_series(s, "t")
v._set_mode("3D")
v.show()
app.processEvents()
v._coronary_mode = True
p3 = [[15.0, 30.0, 25.0], [25.0, 30.0, 25.0], [35.0, 30.0, 25.0]]
pts = [tuple(map(float, v._world3d_to_out("B", np.array(q)))) for q in p3]
v._measures["B"].append({"id": 9, "type": "polyline", "pts": pts,
                         "pts3d": p3, "color": "#ffff00"})
check("(setup) short-axis built",
      v._cpr_ensure_built(("B", len(v._measures["B"]) - 1))
      and v._cpr.get("src") == "B")
v._cpr["idx"] = int(v._cpr_ctrl_indices()[-1])
v._refresh()
app.processEvents()

fork = [{"c": [38.0, 32.0, 25.5]}, {"c": [38.0, 28.0, 23.0]}]
v._cpr_fork_marks_set(fork)
app.processEvents()


def expect(key):
    if key == "A":
        o, u, vv, _n = (np.asarray(a, float) for a in v._cpr_frame())
        return [(float((np.asarray(f["c"]) - o) @ u),
                 float((np.asarray(f["c"]) - o) @ vv)) for f in fork]
    return [tuple(map(float, v._world3d_to_out("B", np.asarray(f["c"]))))
            for f in fork]


def actors(key):
    return [(a.GetInput(), tuple(a.GetPosition()[:2]))
            for a in getattr(v.pane[key], "fork_lbls", []) or []]


for key in ("A", "B"):
    got = actors(key)
    ex = expect(key)
    check(f"pane {key}: numbers 1, 2 beside each candidate",
          [g[0] for g in got] == ["1", "2"]
          and all(np.allclose(g[1], e, atol=1e-6) for g, e in zip(got, ex)),
          (got, ex))

acts = {k: list(getattr(v.pane[k], "fork_lbls", []) or []) for k in "AB"}
check("right pane: small numbers ON the candidates, no frame / box",
      all(not a.GetTextProperty().GetFrame()
          and a.GetTextProperty().GetFontSize() <= 16
          and a.GetDisplayOffset()[1] == 0 for a in acts["B"]),
      [(a.GetTextProperty().GetFontSize(), a.GetTextProperty().GetFrame())
       for a in acts["B"]])
scr = [np.add(v._world_to_qt("B", *a.GetPosition()[:2]),
              (a.GetDisplayOffset()[0], 0)) for a in acts["B"]]
check("right pane: numbers one behind the other are spread apart",
      len(scr) == 2 and float(np.hypot(*(scr[0] - scr[1]))) >= 12.0, scr)
check("short axis: framed numbers beside the rings (as before)",
      all(a.GetTextProperty().GetFrame() for a in acts["A"]))

# rotate the right pane 40° about the vessel: the numbers follow the branch
fu, fv, fn = (np.asarray(a, float) for a in v._frame["B"])
ax, th = np.array([1.0, 0, 0]), math.radians(40)


def rot(w):
    return (w * math.cos(th) + np.cross(ax, w) * math.sin(th)
            + ax * (ax @ w) * (1 - math.cos(th)))


before = actors("B")
v._frame["B"] = (rot(fu), rot(fv), rot(fn))
v._refresh()
app.processEvents()
got = actors("B")
check("right pane rotated: the numbers are re-placed on the candidates",
      all(np.allclose(g[1], e, atol=1e-6) for g, e in zip(got, expect("B")))
      and not all(np.allclose(a[1], b[1]) for a, b in zip(got, before)),
      (before, got))
pd = v.pane["A"].meas_pts_off_mapper.GetInput()
cols = pd.GetCellData().GetScalars()
rgb = {tuple(int(c) for c in cols.GetTuple4(i))[:3]
       for i in range(cols.GetNumberOfTuples())} if cols else set()
check("short axis: a ring round each candidate", v._CPR_FORK_RGB in rgb, rgb)

v._cpr_fork_marks_set(None)
app.processEvents()
check("cleared: no numbers left", actors("A") == [] and actors("B") == [])

# the choice dialog is modeless and still returns the chosen candidate
seen = {}


def answer():
    box = next(w for w in QApplication.topLevelWidgets()
               if isinstance(w, QMessageBox) and w.isVisible())
    seen["modal"] = box.isModal()
    seen["main_enabled"] = v.isEnabled() and not QApplication.activeModalWidget()
    next(b for b in box.buttons() if b.text() == "候補 2").click()


QTimer.singleShot(200, answer)
k = atp.show_result(v, "fork", ok=False, extra=["候補 1", "候補 2"],
                    modeless=True)
check("fork dialog: modeless (panes usable) and returns the choice",
      k == 1 and seen.get("modal") is False and seen.get("main_enabled"),
      (k, seen))

# Mac painter: ring + number at each candidate (stub viewer); pygfx is
# not installed everywhere → then the source is checked
mac = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(
    encoding="utf-8")
check("Mac: painted on the short axis and the long-axis pane",
      mac.count("self._paint_cpr_forks(p, key)") == 2)
try:
    from multi_dicomviewer.viewers.ct_viewer_pygfx import _Overlay  # noqa: E402
except ImportError:
    _Overlay = None
    print("   (pygfx not installed: Mac painter not run)")
if _Overlay is not None:


    class _V:
        _CPR_FORK_RGB = (255, 140, 0)
        _cpr_spread_px = staticmethod(lambda scr: [0.0] * len(scr))

        def _cpr_fork_label_pts(self, key):
            return [("1", (0.0, 0.0)), ("2", (10.0, 0.0))]

        def _cpr_offset_to_screen(self, du, dv):
            return 100 + 5 * du, 100 - 5 * dv

        def _world_to_screen(self, key, wx, wy):
            return 100 + 5 * wx, 100 - 5 * wy


    img = QImage(300, 200, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0))
    pnt = QPainter(img)
    _Overlay._paint_cpr_forks(type("O", (), {"_v": _V()})(), pnt, "B")
    pnt.end()
    c0 = img.pixelColor(100 + 9, 100)               # on the ring of 1
    c1 = img.pixelColor(150 + 9, 100)               # on the ring of 2
    orange = lambda c: c.red() > 200 and 80 < c.green() < 200 and c.blue() < 80  # noqa: E731
    check("Mac: a ring at each candidate", orange(c0) and orange(c1),
          (c0.name(), c1.name()))
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
os._exit(1 if fails else 0)
