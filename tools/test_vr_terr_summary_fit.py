# -*- coding: utf-8 -*-
"""Territory VR top-right summary (Myocardium / LM / LAD …) fits the pane
(user 2026-11-25: its right was cut off in a narrow pane): the block is
measured, right-hugged and shrunk when wider than the pane. Real VTK
CTViewer on synthetic data (made-up patient); the Mac painter by source.

    python tools/test_vr_terr_summary_fit.py
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
for _ in range(5):
    app.processEvents()
TXT = ("Myocardium 123.4mL\nLM 12.3% / 15.2mL\nLAD 45.6% / 56.3mL\n"
       "LCX 22.2% / 27.4mL\nRCA 30.0% / 37.0mL")
v._terr_summary_text = TXT
for width in (1400, 700, 420, 300):
    v.resize(width, 700)
    for _ in range(5):
        app.processEvents()
    for k in ("A", "B"):
        v._vr_on[k] = True
        try:
            v._vr_update_summary(k)
        finally:
            v._vr_on[k] = False
        p = v.pane[k]
        W = p.ren.GetSize()[0]
        bb = [0.0] * 4
        p.terract.GetBoundingBox(p.ren, bb)
        x0 = p.terract.GetPositionCoordinate().GetValue()[0] * W
        right = x0 + (bb[1] - bb[0])
        check(f"width {width} pane {k}: summary inside the pane",
              x0 >= 0 and right <= W + 1,
              (x0, right, W, p.terract.GetTextProperty().GetFontSize()))
src = (ROOT / "multi_dicomviewer" / "viewers" / "ct_viewer_pygfx.py").read_text(
    encoding="utf-8")
check("Mac painter: shrinks instead of clipping",
      "_block_terr_tr(text, max(8, int(pt * (w - 16) / maxw)))" in src
      and "x0 = max(6, w - maxw - 10)" in src)
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
app.processEvents()
os._exit(1 if fails else 0)
