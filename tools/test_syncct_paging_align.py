# -*- coding: utf-8 -*-
"""SyncCT (user 2026-11-23):
- Paging is mirrored on every route: 2-D wheel / Paging drag / arrow keys,
  3-D wheel / drag / keys, and under the LV level link (LV-volume analysis)
  the Paging drag sends the level (按分 / mm) like the wheel does.
- Start dialog 「画像位置合わせ：今のまま / MV-AoV点 / Apex点 / 再設定」: a
  point choice moves each CT so that point is the image centre (orientation
  and zoom kept) and the mm level link counts from that point.
Two real (VTK) CT viewers, synthetic volumes; no dialog is shown.

    python tools/test_syncct_paging_align.py
"""
import copy
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
os.environ["QT_QPA_PLATFORM"] = ("windows" if sys.platform == "win32"
                                 else os.environ.get("QT_QPA_PLATFORM",
                                                     "offscreen"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.ui import sync_view as sv  # noqa: E402
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


def mk(uid):
    vol = np.full((60, 64, 64), -100.0, np.float32)
    s = LoadedSeries(modality=Modality.CT, volume=vol, spacing_mm=(0.7, 0.7),
                     cine_fps=None, window=800.0, level=200.0, slice_mm=1.5,
                     patient_basis=None, series_uid=uid)
    v = CTViewer()
    v.resize(1200, 700)
    v.show()
    v.load_series(s, uid)
    return v


a, b = mk("a"), mk("b")
app.processEvents()
INIT = {id(v): (v._center.copy(), copy.deepcopy(v._pc)) for v in (a, b)}


def home():
    for v in (a, b):
        c, pc = INIT[id(v)]
        v._center, v._pc = c.copy(), copy.deepcopy(pc)
        v._slice2d = 30
        v._center[2] = 30 * 1.5


def set_lv(lv):
    for v in (a, b):
        v._lv_apex = np.array([22.0, 22.0, 20.0]) if lv else None
        v._lv_valves = ({"mitral": (np.array([22.0, 22.0, 70.0]),),
                         "aortic": None} if lv
                        else {"mitral": None, "aortic": None})


# ------------------------------------------------------------ paging routes
for mode in ("2D", "3D"):
    for lv in (False, True):
        for v in (a, b):
            v._set_mode(mode)
        app.processEvents()
        set_lv(lv)
        link = sv.SyncViewLink(a, b)
        for name, fn in (
                ("wheel", lambda: a._wheel("A", 1)),
                ("Paging drag", lambda: (setattr(a, "_tool", "PAGING"),
                                         a._drag("A", 0, -40))),
                ("arrow key", lambda: (setattr(a, "_tool", "PAGING"),
                                       a._key_arrow("up")))):
            home()
            ca, cb = a._center.copy(), b._center.copy()
            fn()
            app.processEvents()
            da = float(np.linalg.norm(a._center - ca))
            db = float(np.linalg.norm(b._center - cb))
            check(f"{mode}{' +LV axis' if lv else ''}: {name} mirrored",
                  da > 0.5 and abs(da - db) < 1e-6, (round(da, 2), round(db, 2)))
        link.teardown()

# LV level link with LV-volume analysis open: drag sends the LEVEL
for v in (a, b):
    v._set_mode("3D")
app.processEvents()
home()
for v in (a, b):
    v._lv_apex = np.array([22.0, 22.0, 20.0])
a._lv_valves = {"mitral": (np.array([22.0, 22.0, 70.0]),), "aortic": None}
b._lv_valves = {"mitral": (np.array([22.0, 22.0, 60.0]),), "aortic": None}
for v in (a, b):
    v._lvv = {}
link = sv.SyncViewLink(a, b)
link.set_level_fraction(True)
a._tool = "PAGING"
a._drag("A", 0, -10)
app.processEvents()
fa, fb = a.lv_cosync_level_fraction(), b.lv_cosync_level_fraction()
check("LV level link: Paging drag → same 按分 level (LV 50 vs 40 mm)",
      abs(fa - 0.5) > 0.05 and abs(fa - fb) < 1e-6, (fa, fb))
link.set_level_fraction(False)
a._drag("A", 0, 10)
app.processEvents()
check("LV level link: Paging drag → same mm level",
      abs(a.lv_cosync_level_mm() - b.lv_cosync_level_mm()) < 1e-6)
link.teardown()
for v in (a, b):
    v._lvv = None

# ------------------------------------------------- point alignment (centre)
for v in (a, b):
    v._set_mode("3D")
app.processEvents()
home()
apex_a, mv_a = np.array([20.0, 20.0, 20.0]), np.array([24.0, 22.0, 70.0])
apex_b, mv_b = np.array([26.0, 18.0, 25.0]), np.array([28.0, 21.0, 65.0])
a._lv_apex, b._lv_apex = apex_a, apex_b
a._lv_valves = {"mitral": None, "aortic": None}
b._lv_valves = {"mitral": (mv_b,), "aortic": None}
check("MV-AoV point missing on one CT → not offered",
      not sv.point_available([a, b], "mv"))
check("Apex point on both CTs → offered", sv.point_available([a, b], "apex"))
a._lv_valves = {"mitral": (mv_a,), "aortic": None}
a._drag("A", 30, 20, tool="ROTATE")            # some oblique view
frames_before = {id(v): {k: tuple(np.asarray(x, float).copy()
                                  for x in v._frame[k]) for k in ("A", "B")}
                 for v in (a, b)}
ps_before = {id(v): v.pane["A"].ren.GetActiveCamera().GetParallelScale()
             for v in (a, b)}
for which, pa, pb in (("mv", mv_a, mv_b), ("apex", apex_a, apex_b)):
    ok = sv.center_on_point([a, b], which)
    app.processEvents()
    check(f"{which}: moved", ok)
    for v, p in ((a, pa), (b, pb)):
        nm = "left" if v is a else "right"
        check(f"{which}: {nm} crosshair centre = its point",
              np.allclose(v._center, p, atol=1e-6), (v._center, p))
        check(f"{which}: {nm} both planes pass through the point (image centre)",
              all(np.allclose(v._pc[k], p, atol=1e-6) for k in ("A", "B")))
        cam = v.pane["A"].ren.GetActiveCamera()
        fp = cam.GetFocalPoint()
        check(f"{which}: {nm} camera centred on it (no pan)",
              abs(fp[0]) < 1e-6 and abs(fp[1]) < 1e-6, fp)
        check(f"{which}: {nm} orientation unchanged",
              all(np.allclose(np.asarray(v._frame[k][i], float),
                              frames_before[id(v)][k][i], atol=1e-9)
                  for k in ("A", "B") for i in range(3)))
        check(f"{which}: {nm} zoom unchanged",
              abs(cam.GetParallelScale() - ps_before[id(v)]) < 1e-6)

# mm level link counted from the chosen point
link = sv.SyncViewLink(a, b)
check("level link on (both CTs have both points)", link.level_link_active())
La = float(np.linalg.norm(mv_a - apex_a))
Lb = float(np.linalg.norm(mv_b - apex_b))
for ref in ("apex", "mv"):
    link.set_mm_ref(ref)
    link.set_level_fraction(False)
    for v in (a, b):
        v._lvv = {}
    a._tool = "PAGING"
    a._drag("A", 0, -15)
    app.processEvents()
    la, lb = a.lv_cosync_level_mm(), b.lv_cosync_level_mm()
    if ref == "apex":
        check("mm from the Apex: same mm from each Apex", abs(la - lb) < 1e-6,
              (la, lb))
    else:
        check("mm from the MV-AoV point: same mm from each MV-AoV point",
              abs((la - La) - (lb - Lb)) < 1e-6, (la - La, lb - Lb))
    for v in (a, b):
        v._lvv = None
link.teardown()

# 2-D: centring on a point also pages to that native slice
for v in (a, b):
    v._set_mode("2D")
app.processEvents()
sv.center_on_point([a, b], "apex")
check("2-D: slice = the Apex slice",
      a._slice2d == int(round(apex_a[2] / a._dims[2]))
      and b._slice2d == int(round(apex_b[2] / b._dims[2])),
      (a._slice2d, b._slice2d))

# ------------------------------------------------------------ start dialog
from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402

host = QWidget()
seen = {}


def fake_exec(pick):
    def _exec(box):
        bt = {x.text(): x for x in box.buttons()}
        seen["texts"] = list(bt)
        seen["enabled"] = {k: bt[t(k)].isEnabled()
                           for k in ("MV-AoV point", "Apex point")}
        bt[t(pick)].click()
        return 0
    return _exec


_real = QMessageBox.exec
both = {"mv": True, "apex": True}
for pick, pts, want in (("As it is", both, "current"),
                        ("MV-AoV point", both, "mv"),
                        ("Apex point", both, "apex"),
                        ("Re-adjust", both, None),
                        ("MV-AoV point", {"mv": False, "apex": True}, None),
                        ("Apex point", {"mv": False, "apex": True}, "apex")):
    QMessageBox.exec = fake_exec(pick)
    got = MainWindow._sv_ask_start_mode(host, pts)
    check(f"dialog: {pick} {pts} → {want}",
          got == want and seen["enabled"] == {"MV-AoV point": pts["mv"],
                                              "Apex point": pts["apex"]},
          (got, seen["enabled"]))
QMessageBox.exec = _real
check("dialog: 今のまま / MV-AoV点 / Apex点 / 再設定",
      seen["texts"] == [t("As it is"), t("MV-AoV point"), t("Apex point"),
                        t("Re-adjust")], seen["texts"])

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
os._exit(1 if fails else 0)
