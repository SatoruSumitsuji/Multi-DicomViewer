# -*- coding: utf-8 -*-
"""Auto-Track stopped (not at a fork), user 2026-11-25:
- while the result dialog is read, BOTH panes are centred on the final
  traced point (short axis on it, the long-axis pane and the crossing
  there) — no longer on the stop section 3 mm beyond;
- 「中心を指定して続ける」 then shows the stop section (its point = the new
  distal edit point) with the search range as a light-orange circle
  (radius = the 5 mm search disc), gone when Auto-Track runs again.

Real VTK CTViewer on synthetic data (made-up patient); dialogs patched.

    python tools/test_cpr_stop_center.py
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
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
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
check("(setup) short-axis built", v._cpr_ensure_built(("B", mi)) and v._cpr is not None)
app.processEvents()
import multi_dicomviewer.ui.autotrack_progress as atp  # noqa: E402
from multi_dicomviewer.core.cpr_autoextend import ExtendParams  # noqa: E402

v._AUTOTRACK_STEP_S = 0.0
seen = {}


def snap():
    o = np.asarray(v._cpr_frame()[0], float)
    return {"secA": o, "pcB": np.asarray(v._pc["B"], float).copy(),
            "center": np.asarray(v._center, float).copy(),
            "last": np.asarray(v._cpr_ctrl_pts3d()[-1], float).copy(),
            "ring": v._cpr_search_ring_2d()}


def answer(pick_extra):
    def fake(parent, text, ok=True, extra="", modeless=False):
        seen["extras"] = list(extra) if isinstance(extra, list) else [extra]
        seen["during"] = snap()
        if pick_extra:
            return len(seen["extras"]) - 1
        return -1
    atp.show_result = fake


answer(False)
n0 = len(v._cpr_ctrl_pts3d())
out = v._cpr_autotrack("B", mi)
d = seen["during"]
check("(setup) the trace ran and stopped (not a fork)",
      out is not None and out["stop"] not in ("fork", "cancel")
      and out["trunk_mm"] > 5, out and out["stop"])
check("dialog: both panes centred on the FINAL traced point",
      np.allclose(d["secA"], d["last"], atol=1e-6)
      and np.allclose(d["pcB"], d["last"], atol=1e-6)
      and np.allclose(d["center"], d["last"], atol=1e-6),
      {k: d[k].round(2) for k in ("secA", "pcB", "center", "last")})
check("dialog: no stop-section point added, no search ring yet",
      d["ring"] is None and not out.get("probe_shown"))
check("dialog offers 「中心を指定して続ける」",
      t("中心を指定して続ける") in seen["extras"], seen["extras"])
after = snap()
check("OK: stays on the final point",
      np.allclose(after["secA"], d["last"]) and after["ring"] is None)

# again from there, now choosing 「中心を指定して続ける」
v._cpr_set_ctrl(v._measures["B"][mi], "B", [list(q) for q in p3])
v._cpr_rebuild()
answer(True)
out = v._cpr_autotrack("B", mi)
s2 = snap()
q = np.asarray(out["detail"]["q"], float)
check("中心を指定: the stop section's point is the new distal edit point",
      np.allclose(s2["last"], q, atol=1e-6)
      and np.allclose(s2["secA"], q, atol=1e-6)
      and np.allclose(s2["center"], q, atol=1e-6))
check("… with the search range circle (5 mm) on the short axis",
      s2["ring"] is not None
      and abs(s2["ring"][1] - ExtendParams().search_mm) < 1e-9
      and np.allclose(s2["ring"][0], (0.0, 0.0), atol=1e-6), s2["ring"])
v._draw_cpr_ctrl_markers()
pd = v.pane["A"].meas_pts_off_mapper.GetInput()
cols = pd.GetCellData().GetScalars()
rgba = {tuple(int(c) for c in cols.GetTuple4(i))
        for i in range(cols.GetNumberOfTuples())} if cols else set()
check("… drawn light orange (translucent)",
      tuple(v._CPR_SEARCH_RGB) + (150,) in rgba, rgba)
# another section → no ring there; back → shown again
idxs = v._cpr_ctrl_indices()
v._cpr["idx"] = int(idxs[-2])
check("another section: no ring", v._cpr_search_ring_2d() is None)
v._cpr["idx"] = int(idxs[-1])
check("back on the stop section: ring again",
      v._cpr_search_ring_2d() is not None)
answer(False)
v._cpr_autotrack("B", mi)
check("a new Auto-Track clears it", v._cpr_search_ring is None
      or v._cpr_search_ring_2d() is None)
mac = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(
    encoding="utf-8")
check("Mac draws the search circle too", "v._cpr_search_ring_2d()" in mac)
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
app.processEvents()
os._exit(1 if fails else 0)
