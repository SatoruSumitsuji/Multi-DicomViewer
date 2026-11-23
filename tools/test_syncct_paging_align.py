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


def mk(uid, shape=(60, 64, 64), px=0.7, sl=1.5):
    vol = np.full(shape, -100.0, np.float32)
    s = LoadedSeries(modality=Modality.CT, volume=vol, spacing_mm=(px, px),
                     cine_fps=None, window=800.0, level=200.0, slice_mm=sl,
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

# User report 2026-11-23: both CTs on Rt (pane B = the short axis), aligned on
# the MV-AoV point, mm level link → Paging moved only the crossline of the
# peer, not its image. Pane B must page on BOTH CTs (drag and wheel).
for v in (a, b):
    v.set_side("Rt")
    ex, ey, ez = np.eye(3)
    v._frame["B"] = (ex, ey, ez)            # B = short axis (⟂ apex→MV, z)
    v._frame["A"] = (ex, ez, -ey)           # A = its orthogonal companion
app.processEvents()
sv.center_on_point([a, b], "mv")
link = sv.SyncViewLink(a, b)
link.set_mm_ref("mv")
link.set_level_fraction(False)


def depth(v, k="B"):
    n = np.asarray(v._frame[k][2], float)
    return float(np.asarray(v._pc[k], float) @ n)


for lvv in (None, {}):
    for v in (a, b):
        v._lvv = lvv
    for name, fn in (("Paging drag", lambda: a._drag("B", 0, -15,
                                                     tool="PAGING")),
                     ("wheel", lambda: a._wheel("B", -1))):
        sv.center_on_point([a, b], "mv")    # start on the MV-AoV point
        da0, db0 = depth(a), depth(b)
        fn()
        app.processEvents()
        da, db = depth(a) - da0, depth(b) - db0
        tag = "LV analysis open" if lvv is not None else "no LV analysis"
        check(f"Rt / MV-AoV / mm, {tag}: {name} pages the shown pane on both",
              abs(da) > 0.3 and abs(db) > 0.3, (round(da, 2), round(db, 2)))
        if True:   # short-axis pane → matched by the LEVEL (with or w/o LV)
            # LV analysis: matched by mm along each long axis, from MV-AoV
            ra = a.lv_cosync_level_mm() - float(
                (mv_a - a._lv_cosync_axis()[0]) @ a._lv_cosync_axis()[1])
            rb = b.lv_cosync_level_mm() - float(
                (mv_b - b._lv_cosync_axis()[0]) @ b._lv_cosync_axis()[1])
            check(f"…{tag}: {name}: same mm from each MV-AoV point",
                  abs(ra - rb) < 1e-6, (ra, rb))
        check(f"…{tag}: {name}: both planes still pass through the centre",
              all(abs(float((np.asarray(v._pc[k], float) - v._center)
                            @ np.asarray(v._frame[k][2], float))) < 1e-6
                  for v in (a, b) for k in ("A", "B")))
for v in (a, b):
    v._lvv = None
    v.set_side("Bi")
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

# 2-D edge: b reaches its last slice first; paging back re-aligns the pair
link = sv.SyncViewLink(a, b)
a._slice2d = 5
a._page2d(0)
b._slice2d = int(b._image.GetDimensions()[2]) - 3
b._page2d(0)
a0, b0 = a._slice2d, b._slice2d
for _ in range(6):
    a._page2d(1)
app.processEvents()
check("2-D edge: the peer stops at its last slice",
      b._slice2d == int(b._image.GetDimensions()[2]) - 1 and a._slice2d == a0 + 6,
      (a._slice2d, b._slice2d))
for _ in range(8):
    a._page2d(-1)
app.processEvents()
check("2-D edge: aligned again after paging back",
      a._slice2d - a0 == b._slice2d - b0 == -2, (a._slice2d - a0, b._slice2d - b0))
link.teardown()

# ------------------------------- different voxel sizes: no drift (2026-11-23)
# The two CTs page by the SAME mm (the old pixel / notch replay moved each by
# its own voxel size, so they drifted apart over use).
c = mk("c", shape=(60, 64, 64), px=0.7, sl=1.5)
d = mk("d", shape=(90, 80, 80), px=0.5, sl=1.0)
app.processEvents()
for v in (c, d):
    v._set_mode("3D")
    v._lv_apex = None
    v._lv_valves = {"mitral": None, "aortic": None}
app.processEvents()


MID = {id(v): np.asarray(v._center, float).copy() for v in (c, d)}


def mid(v):
    p = MID[id(v)].copy()                     # the volume centre (as loaded)
    v._center = p.copy()
    v._pc = {"A": p.copy(), "B": p.copy()}


def updepth(v, k):
    return float(np.asarray(v._center, float) @ v._sync_page_up(k))


for v in (c, d):
    mid(v)
link = sv.SyncViewLink(c, d)
for k in ("A", "B"):
    c0, d0 = updepth(c, k), updepth(d, k)
    for step in range(6):                      # back and forth, drag + wheel
        c._drag(k, 0, -12 if step % 2 == 0 else 7, tool="PAGING")
        c._wheel(k, 1 if step % 3 else -1)
    app.processEvents()
    dc, dd = updepth(c, k) - c0, updepth(d, k) - d0
    check(f"different voxels, no LV axis, pane {k}: same mm after repeated "
          f"drag / wheel", abs(dc) > 1 and abs(dc - dd) < 1e-6,
          (round(dc, 3), round(dd, 3)))
link.teardown()

# with the LV axis: short-axis-like pane → level (mm / 按分), the long-axis
# pane → real mm
for v, apex, mvp in ((c, (22.0, 22.0, 15.0), (22.0, 22.0, 75.0)),
                     (d, (20.0, 20.0, 20.0), (20.0, 20.0, 70.0))):
    v._lv_apex = np.array(apex)
    v._lv_valves = {"mitral": (np.array(mvp),), "aortic": None}
    ex, ey, ez = np.eye(3)
    v._frame["A"] = (ex, ey, ez)              # A = short axis (⟂ z = axis)
    v._frame["B"] = (ex, ez, -ey)             # B = long axis
    mid(v)
link = sv.SyncViewLink(c, d)
for frac in (False, True):
    link.set_level_fraction(frac)
    for step in range(6):
        c._drag("A", 0, -10 if step % 2 == 0 else 6, tool="PAGING")
        c._wheel("A", 1 if step % 3 else -1)
    app.processEvents()
    if frac:
        got = (c.lv_cosync_level_fraction(), d.lv_cosync_level_fraction())
    else:
        got = (c.lv_cosync_level_mm(), d.lv_cosync_level_mm())
    check(f"different voxels, short axis, {'按分' if frac else 'mm'}: "
          f"levels still equal after repeated paging",
          abs(got[0] - got[1]) < 1e-6, got)
c0, d0 = updepth(c, "B"), updepth(d, "B")
for step in range(4):
    c._drag("B", 0, -9 if step % 2 == 0 else 4, tool="PAGING")
app.processEvents()
dc, dd = updepth(c, "B") - c0, updepth(d, "B") - d0
check("different voxels, long-axis pane: same mm",
      abs(dc) > 1 and abs(dc - dd) < 1e-6, (round(dc, 3), round(dd, 3)))

# volume edge: the peer hits its edge first (d is smaller along B's normal);
# after paging back, the pair is aligned again (no lasting offset)
for v in (c, d):
    mid(v)
link.set_level_fraction(False)
c0, d0 = updepth(c, "B"), updepth(d, "B")
d_ref = d._center.copy()
for _ in range(3):
    c._drag("B", 0, -10, tool="PAGING")        # 3 x 7 mm = 21 mm
app.processEvents()
dc, dd = updepth(c, "B") - c0, updepth(d, "B") - d0
check("volume edge: the peer stops at its edge", abs(dd) < abs(dc) - 0.5,
      (round(dc, 3), round(dd, 3)))
check("volume edge: the peer stays on the paging line (no in-plane slide)",
      np.allclose(np.cross(d._center - d_ref, d._sync_page_up("B")), 0,
                  atol=1e-6), (d._center, d_ref))
for _ in range(5):
    c._drag("B", 0, 8, tool="PAGING")          # back past the start
app.processEvents()
dc, dd = updepth(c, "B") - c0, updepth(d, "B") - d0
check("volume edge: aligned again after paging back",
      abs(dc) > 1 and abs(dc - dd) < 1e-6, (round(dc, 3), round(dd, 3)))
d.sync_center_on(d._center)
check("volume edge: a fresh alignment clears the debt",
      not d._sync_pg_debt)

# double-click recentre is mirrored (same in-plane move, image recentred)
for v in (c, d):
    mid(v)
cc0, dc0 = c._center.copy(), d._center.copy()
cw, ch = c.pane["A"].canvas.width(), c.pane["A"].canvas.height()
c._gesture_moved = False
c._recenter("A", cw * 0.5 + 40, ch * 0.5 - 25)
app.processEvents()
mvc, mvd = c._center - cc0, d._center - dc0
check("double-click recentre: the peer centre moves the same way",
      float(np.linalg.norm(mvc)) > 1
      and np.allclose(mvc, mvd, atol=1e-6), (mvc, mvd))
check("double-click recentre: the peer image is recentred on it",
      all(np.allclose(d._pc[k], d._center, atol=1e-6) for k in ("A", "B")))

# SyncCT bar 位置合わせ: redo the alignment without leaving SyncCT
from multi_dicomviewer.ui.main_window import MainWindow as _MW  # noqa: E402


class _Shell:
    _syncview_link = link
    synced = []

    def _sv_ask_start_mode(self, pts):
        return "apex"

    def _sv_sync_bar_state(self):
        self.synced.append(1)


link.set_level_fraction(True)
_MW._sv_realign(_Shell())
app.processEvents()
check("位置合わせ: both CTs centred on their Apex point, SyncCT kept",
      np.allclose(c._center, c._lv_apex) and np.allclose(d._center, d._lv_apex)
      and link.viewers() == [c, d])
check("位置合わせ: level link → mm from the chosen point, bar refreshed",
      link.mm_ref == "apex" and not link.level_fraction and _Shell.synced)
link.teardown()

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
