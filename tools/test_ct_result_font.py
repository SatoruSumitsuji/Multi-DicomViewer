# -*- coding: utf-8 -*-
"""結果文字サイズ (user 2026-11-25): one slider (right of the CT mode
buttons) sizes every RESULT text on BOTH panes alike — WW/WL + pane kind,
the angle (+ outline), the result block, the legends, the hover HU — while
the top-row DICOM-tag slider sizes only the tags. Real VTK CTViewer on
synthetic data (made-up patient); the Mac painter by source.

    python tools/test_ct_result_font.py
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
from multi_dicomviewer.ui.tag_font import TAG_FONT_PT_DEFAULT  # noqa: E402
base = ctv._vtk_font_px(TAG_FONT_PT_DEFAULT)


def sizes(k):
    p = v.pane[k]
    return {"info": p.info.GetMaximumFontSize(),
            "result": p.resultact.GetTextProperty().GetFontSize(),
            "angle": p.angle.GetTextProperty().GetFontSize(),
            "halo": {a.GetTextProperty().GetFontSize() for a in p.angle_halo},
            "legend": {a.GetTextProperty().GetFontSize()
                       for a in p.lv_wall_legend},
            "hover": p.hover_hu.GetTextProperty().GetFontSize(),
            "tag": p.tagact.GetTextProperty().GetFontSize()}


s0 = {k: sizes(k) for k in ("A", "B")}
sl = v._result_font_slider
check("slider sits in the mode row, 50–150 %",
      sl.minimum() == 50 and sl.maximum() == 150 and sl.isVisible())
sl.setValue(150)
app.processEvents()
s1 = {k: sizes(k) for k in ("A", "B")}
check("both panes take the SAME sizes", s1["A"] == s1["B"], s1)
a = s1["A"]
check("150 %: WW/WL + pane kind, result block",
      a["info"] == round(base * 1.5) and a["result"] == round(base * 1.5), a)
check("150 %: angle + its outline", a["angle"] == 27 and a["halo"] == {27}, a)
check("150 %: wall legend, hover HU",
      a["legend"] == {24} and a["hover"] == round(15 * 1.5), a)
check("the DICOM tags are NOT changed by it", a["tag"] == s0["A"]["tag"])
v.set_overlay_font_pt(TAG_FONT_PT_DEFAULT + 6)
app.processEvents()
s2 = {k: sizes(k) for k in ("A", "B")}
check("the tag slider changes the tags only",
      s2["A"]["tag"] != a["tag"]
      and {kk: vv for kk, vv in s2["A"].items() if kk != "tag"}
      == {kk: vv for kk, vv in a.items() if kk != "tag"}, s2["A"])
v._redraw_meas("A")
v._redraw_meas("B")
check("a result redraw keeps the size",
      sizes("A")["result"] == sizes("B")["result"] == round(base * 1.5))
sl.setValue(50)
app.processEvents()
check("50 %: smaller on both panes",
      sizes("A")["angle"] == sizes("B")["angle"] == 9, sizes("A"))
# the shell: every CT viewer + saved
saved = {}
import multi_dicomviewer.core.settings as st  # noqa: E402
st_save = st.save_ui_state
st.save_ui_state = lambda k, val: saved.__setitem__(k, val)
from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402


class _Pane:
    def all_viewers(self):
        return [v]


class _Shell:
    _panes = [_Pane()]
    ct_result_font_set = MainWindow.ct_result_font_set


_Shell().ct_result_font_set(120)
st.save_ui_state = st_save
check("shell: applied + saved",
      v._result_font_pct == 120 and sl.value() == 120
      and saved.get("ct_result_font_pct") == 120, (v._result_font_pct, saved))
src = (ROOT / "multi_dicomviewer" / "viewers" / "ct_viewer_pygfx.py").read_text(
    encoding="utf-8")
check("Mac painter: results / WW-WL / angle / legends at _rfs",
      src.count("v._rfs(") >= 14 and "def _apply_result_font" in src
      and 'QFont("monospace", v._overlay_font_pt)' not in src)
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
app.processEvents()
os._exit(1 if fails else 0)
