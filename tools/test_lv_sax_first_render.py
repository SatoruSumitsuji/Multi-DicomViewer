# -*- coding: utf-8 -*-
"""LV SAX entry (user 2026-11-25): the first long-axis image after 短軸 was
black in its lower third (the apex) until any nudge. The reslice samples only
the camera's visible viewport; the SAX default scale/centre changed the camera
AFTER it was built. Now the default-scale step re-reslices. Real VTK CTViewer
on synthetic data (made-up patient); the LV model is faked — this checks the
reslice covers the view after the deferred SAX refit.

    python tools/test_lv_sax_first_render.py
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
v._refresh(reset_cam=True)
app.processEvents()


def covered(k):
    """Does pane k's reslice output box cover the camera's visible rect?"""
    p = v.pane[k]
    cam = p.ren.GetActiveCamera()
    ps = cam.GetParallelScale()
    fx, fy = cam.GetFocalPoint()[:2]
    w, h = p.canvas.width(), p.canvas.height()
    hu = ps * w / max(1, h)
    ox, oy, _ = p.reslice.GetOutputOrigin()
    sx, sy, _ = p.reslice.GetOutputSpacing()
    e = p.reslice.GetOutputExtent()
    x1, y1 = ox + sx * e[1], oy + sy * e[3]
    tol = 2 * sx
    return (ox <= fx - hu + tol and x1 >= fx + hu - tol
            and oy <= fy - ps + tol and y1 >= fy + ps - tol), \
        (ox, x1, oy, y1, fx - hu, fx + hu, fy - ps, fy + ps)


# the SAX default step: a zoom-out + pan (as _lv_sax_apply_default_scale does)
def fake_default():
    for k in ("A", "B"):
        cam = v.pane[k].ren.GetActiveCamera()
        cam.SetParallelScale(cam.GetParallelScale() * 1.8)
        fp = list(cam.GetFocalPoint()); po = list(cam.GetPosition())
        fp[1] -= 30.0; po[1] -= 30.0
        cam.SetFocalPoint(*fp); cam.SetPosition(*po)
    v._view_initial = False


v._lv_sax_apply_default_scale = fake_default
v._redraw_lv = lambda k: None
v._lv = {"sax": 0.0, "pane": "B", "sax_pane": "A"}
fake_default()                                    # the old order: camera moved …
ok0, _ = covered("B")
check("(repro) camera changed after the reslice → part of the view unsampled",
      not ok0)
v._lv_sax_refit_default()                         # the deferred SAX refit
app.processEvents()
okA, ia = covered("A")
okB, ib = covered("B")
check("after the SAX refit the long-axis pane is resliced over the whole view",
      okB, ib)
check("… and the short-axis pane too", okA, ia)
src = (ROOT / "multi_dicomviewer" / "viewers" / "ct_viewer.py").read_text(
    encoding="utf-8").replace("\r\n", "\n")
i = src.index("        if first:\n            self._lv_sax_apply_default_scale()")
check("SAX entry: re-reslice right after the default scale",
      "self._refresh(reset_cam=False)" in src[i:i + 600])
v._lv = None
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
app.processEvents()
os._exit(1 if fails else 0)
