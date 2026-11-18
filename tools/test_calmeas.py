# -*- coding: utf-8 -*-
"""Tools ▸ Cal-Measure (user 2026-11-18): calibration-first measurement on
2-D series (XA, echo …), separate from Measure.

  - core: FWHM on a line (dark / bright object), the oblique-line estimate
    (primary value unchanged; a perpendicular reference only when the vessel
    direction is reliable; "unreliable" at a narrowing), catheter width
    from a polygon (contrast-filled vs. walls-only warning), DICOM pixel size
    (US region / PixelSpacing / ImagerPixelSpacing)
  - window: frame choice, per-plane calibration (2 points / catheter /
    DICOM), distance / polyline / diameter rows in mm, recalibration updates
    the values, save as <name>.CalMeas.json (calibration + measurements) and
    load back, CSV, unsaved-close question
  - shell: Tools ▸ Cal-Measure… hands over the active pane's plane(s)/frame;
    the LV spline buttons' colours (HU+spline grey, spline orange)

    python tools/test_calmeas.py
"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
msgs = []
QMessageBox.information = staticmethod(lambda *a, **k: msgs.append(a))
QMessageBox.warning = staticmethod(lambda *a, **k: msgs.append(a))
answer = {"q": QMessageBox.StandardButton.Discard}
QMessageBox.question = staticmethod(lambda *a, **k: answer["q"])
from multi_dicomviewer.core import calmeas as cm  # noqa: E402
from multi_dicomviewer.ui.calmeas_window import CalMeasWindow  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


H, W = 300, 400
yy, xx = np.mgrid[0:H, 0:W].astype(float)


def band(img, p0, p1, width, val):
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    d /= L
    X = np.stack([xx - p0[0], yy - p0[1]], -1)
    t = np.clip(X @ d, 0, L)
    img[np.linalg.norm(X - t[..., None] * d, axis=-1) <= width / 2] = val


def make(seed=0):
    img = np.full((H, W), 800.0)
    band(img, (50, 60), (350, 60 + 300 * math.tan(math.radians(30))), 12, 300)
    band(img, (300, 20), (320, 280), 8, 250)          # catheter, 8 px
    band(img, (40, 250), (140, 250), 10, 300)          # vessel with a
    band(img, (140, 250), (160, 250), 4, 300)          # sharp narrowing
    band(img, (160, 250), (260, 250), 10, 300)
    img = ndimage.gaussian_filter(img, 1.2)
    return img + np.random.default_rng(seed).normal(0, 15, img.shape)


IMG = make()
C = np.array([200.0, 60 + 150 * math.tan(math.radians(30))])


def line_at(off_deg, half=25.0, c=C, vessel_deg=30.0):
    a = math.radians(vessel_deg + 90 + off_deg)
    u = np.array([math.cos(a), math.sin(a)])
    return c - half * u, c + half * u


# ---- core ----------------------------------------------------------------
d = cm.measure_diameter(IMG, *line_at(0))
check("FWHM of a 12 px vessel (perpendicular line)",
      d.fwhm.ok and abs(d.fwhm.width_px - 12) < 0.6 and d.fwhm.dark,
      round(d.fwhm.width_px, 2))
d = cm.measure_diameter(IMG, *line_at(25))
check("oblique line 25°: primary = as measured (larger), angle found",
      d.fwhm.width_px > 12.8 and abs(d.angle_deg - 25) < 3 and d.direction_ok,
      (round(d.fwhm.width_px, 2), round(d.angle_deg, 1)))
check("… perpendicular reference ≈ true 12 px",
      d.corrected is not None and abs(d.corrected.width_px - 12) < 0.6)
d = cm.measure_diameter(IMG, *line_at(5))
check("nearly perpendicular (5°): no reference offered",
      d.direction_ok and d.corrected is None)
d = cm.measure_diameter(IMG, (150, 225), (150, 275))
check("inside a narrowing: measured (blur widens a 4 px lumen a little)",
      d.fwhm.ok and abs(d.fwhm.width_px - 4) < 1.5, round(d.fwhm.width_px, 2))
d = cm.measure_diameter(IMG, (141, 225), (141, 275))
check("at the edge of a narrowing: direction 'unreliable', no reference",
      d.fwhm.ok and not d.direction_ok and d.corrected is None,
      d.direction_reason)
f = cm.line_fwhm(IMG, (195, 140), (205, 150))
check("line not reaching beyond the lumen → refused (edge)",
      not f.ok and f.reason in ("edge", "contrast"), f.reason)
bright = 1600 - IMG
check("bright object (echo-like polarity) works too",
      abs(cm.line_fwhm(bright, *line_at(0)).width_px - 12) < 0.6)
k = cm.catheter_width(IMG, [(290, 60), (335, 60), (340, 220), (295, 220)])
check("catheter width from a polygon ≈ 8 px",
      k.ok and abs(k.width_px - 8) < 0.5 and k.n_profiles > 50
      and not k.warnings, (round(k.width_px, 2), k.warnings))
walls = np.full((H, W), 800.0)
band(walls, (300, 20), (320, 280), 10, 300)
band(walls, (300, 20), (320, 280), 6, 800)            # lumen not filled
walls = ndimage.gaussian_filter(walls, 0.8) + np.random.default_rng(1).normal(
    0, 10, walls.shape)
k2 = cm.catheter_width(walls, [(290, 60), (335, 60), (340, 220), (295, 220)])
check("empty catheter (walls only) → refused as two-peaked",
      not k2.ok and k2.reason == "dip", (k2.ok, k2.warnings, k2.reason))
us = SimpleNamespace(SequenceOfUltrasoundRegions=[SimpleNamespace(
    RegionSpatialFormat=1, PhysicalUnitsXDirection=3, PhysicalDeltaX=0.02,
    RegionLocationMinX0=0, RegionLocationMaxX1=600, RegionLocationMinY0=0,
    RegionLocationMaxY1=400)])
check("DICOM: US B-mode region (cm) → mm/px",
      cm.dicom_mm_per_px(us) == (0.2, "US region"))
check("DICOM: XA ImagerPixelSpacing",
      cm.dicom_mm_per_px(SimpleNamespace(ImagerPixelSpacing=[0.3, 0.3]))
      == (0.3, "ImagerPixelSpacing"))


# ---- window ----------------------------------------------------------------
class Plane:
    def __init__(self, name, frames, path, ds=None):
        self.name = name
        self.volume = np.stack(frames).astype(np.float32)
        self.total_frames = len(frames)
        self.path = path
        self._ds = ds
        self.is_color = False

    def frame(self, i):
        return self.volume[max(0, min(int(i), self.total_frames - 1))]


DIR = tempfile.mkdtemp(dir=HOME)
pa = Plane("A", [make(s) for s in range(3)], os.path.join(DIR, "a.dcm"),
           SimpleNamespace(ImagerPixelSpacing=[0.25, 0.25]))
pb = Plane("B", [make(9) for _ in range(2)], os.path.join(DIR, "b.dcm"))
w = CalMeasWindow([{"plane": pa, "index": 0, "name": "A"},
                   {"plane": pb, "index": 1, "name": "B"}], frame=1,
                  window=1000, level=600, title="test run", series_uid="u1",
                  modality="XA")
w.show()
app.processEvents()
check("starts on the pane's frame; both planes offered",
      w._frame == 1 and w._plane_cb.count() == 2)
w._set_tool("dist")
check("measuring needs a calibration first", w._tool is None and msgs)
# 2-point calibration: 100 px = 25 mm
w._set_tool("cal2")
w.canvas.clicked.emit(100.0, 100.0)
w.canvas.clicked.emit(200.0, 100.0)
w._dist_spin.setValue(25.0)
w._cal_confirm()
check("2-point calibration → 0.25 mm/px",
      abs(w._mmpp() - 0.25) < 1e-9 and "0.2500" in w._cal_lbl.text())
w._set_tool("dist")
w.canvas.clicked.emit(10.0, 10.0)
w.canvas.clicked.emit(50.0, 40.0)
w._set_tool("poly")
for q in ((10, 10), (40, 10), (40, 50)):
    w.canvas.clicked.emit(float(q[0]), float(q[1]))
w.canvas.finished.emit()
w._set_tool("diam")
p0, p1 = line_at(25)
w.canvas.clicked.emit(*map(float, p0))
w.canvas.clicked.emit(*map(float, p1))
ms = w._state[0]["meas"]
vals = [round(w._value_mm(m), 2) for m in ms]
check("distance 50 px → 12.50 mm, polyline 70 px → 17.50 mm",
      vals[:2] == [12.5, 17.5], vals)
check("diameter row: value (as measured) + note with angle and reference",
      len(ms) == 3 and ms[2]["type"] == "diameter"
      and "25" in w.note_text(ms[2]) and "mm" in w.note_text(ms[2])
      and w._table.rowCount() == 3, w.note_text(ms[2]))
# catheter recalibration (6 Fr = 2 mm for 8 px → 0.25 mm/px) updates values
w._set_tool("calcath")
for q in [(290, 60), (335, 60), (340, 220), (295, 220)]:
    w.canvas.clicked.emit(float(q[0]), float(q[1]))
w.canvas.finished.emit()
w._fr_cb.setCurrentIndex(cm_idx := [4, 5, 6, 7, 8].index(6))
w._cal_confirm()
c = w._cal()
check("catheter calibration: 6 Fr / ≈8 px",
      c["method"] == "catheter" and c["fr"] == 6
      and abs(c["mm_per_px"] - 2.0 / 8.0) < 0.02, c["mm_per_px"])
check("recalibration re-computes the rows",
      abs(w._value_mm(ms[0]) - 50 * c["mm_per_px"]) < 1e-9)
# plane B: own calibration
w._plane_cb.setCurrentIndex(1)
check("plane B has no calibration of its own yet", w._cal() is None
      and not w._dist_btn.isEnabled())
w._plane_cb.setCurrentIndex(0)
w._cal_dicom()
check("DICOM value offered as pending (detector-plane note)",
      w._pending is not None and w._pending["mm_per_px"] == 0.25
      and w._pend_lbl.text())
w._pending = None
w._refresh()
# save / load
QFileDialog.getSaveFileName = staticmethod(
    lambda *a, **k: (os.path.join(DIR, "LAD_pre"), ""))
path = w.save()
check("saved as <name>.CalMeas.json",
      path == os.path.join(DIR, "LAD_pre.CalMeas.json") and os.path.isfile(path))
with open(path, encoding="utf-8") as f:
    data = json.load(f)
pl0 = data["planes"][0]
check("file holds the calibration and the measurements (mm)",
      data["format"] == "MDV-CalMeas" and pl0["calibration"]["method"]
      == "catheter" and pl0["calibration"]["polygon"]
      and len(pl0["measurements"]) == 3
      and abs(pl0["measurements"][0]["value_mm"]
              - 50 * c["mm_per_px"]) < 1e-6
      and pl0["measurements"][2]["corrected_mm"] > 0)
w2 = CalMeasWindow([{"plane": pa, "index": 0, "name": "A"},
                    {"plane": pb, "index": 1, "name": "B"}], series_uid="u1")
ok = w2.load(path)
check("loaded back: calibration + rows restored",
      ok and w2._mmpp() == w._mmpp() and len(w2._state[0]["meas"]) == 3
      and w2._table.rowCount() == 3)
csvp = w.export_csv(os.path.join(DIR, "x.csv"))
with open(csvp, encoding="utf-8-sig") as f:
    lines = f.read().splitlines()
check("CSV: header + 3 rows", len(lines) == 4 and "value_mm" in lines[0])
# unsaved close asks (Cancel keeps the window)
w._dirty = True
answer["q"] = QMessageBox.StandardButton.Cancel
w.close()
check("unsaved → close asks; Cancel keeps it", w.isVisible())
answer["q"] = QMessageBox.StandardButton.Discard
w.close()
check("… Discard closes", not w.isVisible())

# ---- shell / buttons (source checks) ----------------------------------------
s = (ROOT / "multi_dicomviewer/ui/main_window.py").read_text(encoding="utf-8")
check("Tools ▸ Cal-Measure… wired",
      'QAction(t("Cal-Measure…")' in s and "def _open_calmeas" in s
      and "v._shown_planes()" in s)
for nm in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    src = (ROOT / "multi_dicomviewer/viewers" / nm).read_text(encoding="utf-8")
    i = src.index("self._lv_husnap_btn.setStyleSheet(\n                \"QPush")
    seg = src[i:i + 700]
    check(f"{nm}: HU+spline grey, spline orange after the HU pass",
          "#c4c4c4" in seg and "_lv_saxspl_btn.setStyleSheet" in seg
          and "#e0a030" in seg.split("_lv_saxspl_btn.setStyleSheet")[1])

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
