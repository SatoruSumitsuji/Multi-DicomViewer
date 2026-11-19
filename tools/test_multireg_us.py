# -*- coding: utf-8 -*-
"""MultiReg without angio (user 2026-11-19): an IVUS stored as Modality "US"
and a CT short-axis are both pull-backs (no Modality judgement); two sync
points with no XA position, then scrubbing the IVUS drives the CT
short-axis. The CT long-axis view (MPR / slab MIP as shown) joins as an
angio-like guide pane whose guide is the projected centreline: its marker
follows the CT short-axis by geometry (no tracing, no MultiReg points).
Also: the 2-CT link is named SyncCT (was SyncView).

Real VTK CTViewer + CoregWindow on synthetic data (made-up patient).

    python tools/test_multireg_us.py
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
app.processEvents()
v._coronary_mode = True
p3 = [[15.0, 40, 30], [22.0, 40, 30]]
pts = [tuple(map(float, v._world3d_to_out("B", np.array(q)))) for q in p3]
v._measures["B"].append({"id": 77, "type": "polyline", "pts": pts, "pts3d": p3, "color": "#ffff00"})
mi = len(v._measures["B"]) - 1
print("built", v._cpr_ensure_built(("B", mi)), v._cpr is not None)
app.processEvents()
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from multi_dicomviewer.core import dicom_io
from multi_dicomviewer.ui.coreg_window import CoregWindow
spec_ct = v.cpr_cosync_spec()
check("CT short-axis spec", spec_ct is not None)
usd = tempfile.mkdtemp(dir=HOME)
meta = FileMetaDataset(); meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
meta.MediaStorageSOPInstanceUID = generate_uid(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
ds = FileDataset(None, {}, file_meta=meta, preamble=bytes(1) * 128)
ds.SOPClassUID = meta.MediaStorageSOPClassUID; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
ds.PatientName, ds.PatientID = "TEST^US", "TESTUS"; ds.StudyInstanceUID = generate_uid(); ds.SeriesInstanceUID = generate_uid()
ds.Modality = "US"; ds.Rows = ds.Columns = 64; ds.NumberOfFrames = 20; ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"; ds.BitsAllocated = ds.BitsStored = 8; ds.HighBit = 7; ds.PixelRepresentation = 0
ds.PixelData = (np.random.default_rng(1).integers(0, 255, (20, 64, 64), dtype=np.uint8)).tobytes()
ds.save_as(os.path.join(usd, "us.dcm"), enforce_file_format=True)
pts_ = dicom_io.index_files([os.path.join(usd, "us.dcm")])
us = [se for p in pts_.values() for st in p.studies.values() for se in st.series.values()][0]
check("US series is not Modality.IVUS", us.kind == "US")
w = CoregWindow([(us, 0, 0, 0.0, False), spec_ct]); app.processEvents()
check("MultiReg: the US IVUS and the CT short-axis are both pull-backs (no XA)",
      [p.is_ivus for p in w.panes] == [True, True] and w.panes[0].total == 20,
      [(p.is_ivus, p.total) for p in w.panes])
P_us, P_ct = w.panes
def lm(fu, fc):
    P_us.slider.setValue(fu); P_ct.slider.setValue(fc); app.processEvents()
    w._add_btn.click(); app.processEvents(); w._add_btn.click(); app.processEvents()
w._set_active_ivus(P_us.index)
lm(2, 5); lm(15, 40)
check("two sync points without any XA position",
      len(w._landmarks) == 2 and all(not lm_["fracs"] for lm_ in w._landmarks),
      [lm_["frames"] for lm_ in w._landmarks])
P_us.slider.setValue(8); app.processEvents()
f2 = w._landmarks[1]["frames"][1]
exp = round(5 + (8 - 2) / 13 * (f2 - 5))
check("scrubbing the US IVUS drives the CT short-axis (interpolated)",
      abs(P_ct.cur - exp) <= 1, (P_ct.cur, exp))

# ---- CT long-axis view as an automatic guide pane (no tracing) ------------
v._thick["B"] = 40.0                  # slab MIP holding the whole vessel
g = v.cpr_cosync_guide_spec()
img = np.asarray(g["image"]) if g else None
cv_ = np.asarray(g["curve"]) if g else np.zeros((0, 2))
vals = [img[int(round(y)), int(round(x))] for x, y in cv_
        if 0 <= int(round(y)) < img.shape[0]
        and 0 <= int(round(x)) < img.shape[1]] if g else []
check("guide spec: slab-MIP image + the centreline projected (1 pt / section)",
      g is not None and len(cv_) == len(spec_ct["frames"])
      and len(vals) == len(cv_) and float(np.min(vals)) > 250,
      (img.shape if g else None, len(cv_),
       round(float(np.min(vals))) if vals else None))
w2 = CoregWindow([(us, 0, 0, 0.0, False), spec_ct, g])
app.processEvents()
G = w2.panes[2]
check("guide pane: angio-like, linked to the CT short-axis, no Trace needed",
      not G.is_ivus and G.link_ct == 1
      and len(w2._guides[2]["curve"]) == len(cv_))
w2._set_active_ivus(1)
w2.panes[1].slider.setValue(7)
app.processEvents()
check("CT short-axis scrubbed → the long-axis marker is at that section",
      G.canvas.coreg_marker_pt is not None
      and np.allclose(G.canvas.coreg_marker_pt, cv_[7]),
      (G.canvas.coreg_marker_pt, tuple(cv_[7])))
U2, C2 = w2.panes[0], w2.panes[1]
w2._set_active_ivus(0)
for fu, fc in ((2, 3), (15, 16)):
    U2.slider.setValue(fu)
    C2.slider.setValue(fc)
    app.processEvents()
    w2._add_btn.click()
    app.processEvents()
    w2._add_btn.click()
    app.processEvents()
check("MultiReg points still commit (the guide pane takes no part)",
      len(w2._landmarks) == 2
      and all(not lm_["fracs"] for lm_ in w2._landmarks),
      [lm_["frames"] for lm_ in w2._landmarks])
U2.slider.setValue(8)
app.processEvents()
check("IVUS master → CT follows → the long-axis marker follows the CT",
      np.allclose(G.canvas.coreg_marker_pt, cv_[C2.cur]),
      (C2.cur, G.canvas.coreg_marker_pt))
G.canvas.coreg_marker_dragged.emit(0.9)
app.processEvents()
check("dragging the long-axis marker scrolls the CT short-axis there",
      abs(C2.cur - round(0.9 * (C2.total - 1))) <= 1, (C2.cur, C2.total))
C2._toggle_reverse()
app.processEvents()
w2._drive_linked()
check("CT short-axis reversed → the marker stays on the same section",
      np.allclose(G.canvas.coreg_marker_pt,
                  cv_[C2.total - 1 - C2.cur]),
      (C2.cur, G.canvas.coreg_marker_pt))

# ---- as on screen (user 2026-11-19): the short-axis pane's own W/L, and the
# HU colour map in colour when it is on ------------------------------------
v._wl["A"] = [600.0, 150.0]
sp_g = v.cpr_cosync_spec()
check("short-axis spec takes pane A's window / level (as shown)",
      (sp_g["window"], sp_g["level"]) == (600.0, 150.0)
      and not sp_g["color"], (sp_g["window"], sp_g["level"]))
v._color = True
sp_c = v.cpr_cosync_spec()
fr_c = np.asarray(sp_c["frames"])
check("colour map on → the short-axis goes to MultiReg in colour (RGB)",
      sp_c["color"] and fr_c.dtype == np.uint8 and fr_c.ndim == 4
      and fr_c.shape[-1] == 3
      and int(np.abs(fr_c[..., 0].astype(int)
                     - fr_c[..., 2].astype(int)).max()) > 0,
      (fr_c.shape, fr_c.dtype))
g_c = v.cpr_cosync_guide_spec()
check("colour map on → the long-axis guide (option) in colour too",
      g_c is not None and g_c["color"]
      and np.asarray(g_c["image"]).shape[-1] == 3)
w3 = CoregWindow([(us, 0, 0, 0.0, False), sp_c, g_c])
app.processEvents()
P3, G3 = w3.panes[1], w3.panes[2]
d3 = P3._display_frame(P3.cur)
check("MultiReg shows the CT short-axis / long-axis as coloured",
      P3.is_color and G3.is_color and d3.ndim == 3 and d3.shape[-1] == 3,
      (P3.is_color, G3.is_color, d3.shape))
v._color = False
# zoom / pan / rotation of the short-axis pane as shown (user 2026-11-19)
import math as _m  # noqa: E402
c_ = v._cpr
camA = v.pane["A"].ren.GetActiveCamera()
_f0, _p0 = camA.GetFocalPoint(), camA.GetParallelScale()
camA.SetParallelScale(float(c_["half"]) / 2.0)          # zoomed in ×2
camA.SetFocalPoint(1.2, -0.8, _f0[2])                    # panned
v.set_cpr_rotation(30.0, silent=True)
sp_z = v.cpr_cosync_spec(px=97)
k_ = sp_z["start"]
i_ = (c_["cl"].n - 1 - k_) if c_.get("reversed") else k_
o_ = np.asarray(c_["cl"].points[i_], float)
want = float(v._sample_vol_grid((o_ + 1.2 * c_["u"][i_]
                                 - 0.8 * c_["v"][i_])[None, None, :])[0, 0])
got = float(np.asarray(sp_z["frames"])[k_, 48, 48])
cvw = v.pane["A"].canvas
hv = (float(c_["half"]) / 2.0) * min(cvw.width(), cvw.height()) \
    / max(1, cvw.height())
check("MultiReg short-axis = the pane's zoom + pan (centre = the panned "
      "point, field = the shown square), rotation carried separately",
      abs(got - want) < 1e-3
      and abs(sp_z["spacing_mm"] - 2 * hv / 96) < 1e-6
      and abs(sp_z["rotation"] - 30.0) < 1e-9,
      (round(got, 2), round(want, 2), sp_z["spacing_mm"], 2 * hv / 96))
v.set_cpr_rotation(0.0, silent=True)
camA.SetParallelScale(_p0)
camA.SetFocalPoint(*_f0)
mw_src = (ROOT / "multi_dicomviewer" / "ui" / "main_window.py").read_text(
    encoding="utf-8")
check("the CT long-axis guide is an option (Tools menu, off by default)",
      "multireg_ct_long_axis" in mw_src
      and "if long_on and hasattr(v, \"cpr_cosync_guide_spec\")" in mw_src
      and 'load_ui_state("multireg_ct_long_axis", False)' in mw_src)

# SyncCT: the 2-CT link's visible name
import inspect  # noqa: E402
from multi_dicomviewer.ui import main_window as _mwm  # noqa: E402
src_mw = inspect.getsource(_mwm)
check("the 2-CT link is called SyncCT (menu / bar / End button)",
      't("SyncCT")' in src_mw and 't("SyncCT終了")' in src_mw
      and 't("SyncView")' not in src_mw
      and t("SyncCT終了") != "", t("SyncCT終了"))

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
