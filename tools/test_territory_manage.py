# -*- coding: utf-8 -*-
"""Territory Manage (user 2026-11-27): a separate window for the targets —
show/hide, name, 心筋%, 灌流域 mL, 元血管名, ターゲット位置. A named target
replaces 「Target-N」 in the right pane's bottom-right result (and is saved in
the .territory.json). Opened by row 3's 「Territory Manage」 (right of
Territory Calc) or a right-click on that result. Ticking 除外 keeps the
calculated Territory (and the right pane) — counted at the next Calc.
Synthetic tree, faked engine, fake HOME, no real dialog.

    python tools/test_territory_manage.py
"""
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
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
for _n in ("information", "warning", "critical", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("unexpected dialog"))))
from multi_dicomviewer.core import territory_file  # noqa: E402
from multi_dicomviewer.core.centerline import CenterLine  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.ui import coronary_tree_window as ctw  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


def pump():
    for _ in range(3):
        app.processEvents()


ENG_CALLS = []


class _Eng:
    def territory(self, vid, idx):
        ENG_CALLS.append((vid, idx))
        return np.zeros(4, bool), 10.0 + idx


class _LVF:
    def myocardial_volume_ml(self):
        return 100.0


ctw.format_territory_report = lambda tree, lvf: (["report"], _Eng())
ctw.LVFunction.from_full = staticmethod(lambda d: _LVF())

w = ctw.CoronaryTreeWindow(None)
w._run_busy = lambda _msg, fn: fn()
w._use_uid("1.2.3")
tr = w._bundle()["tree"]
pts = np.c_[np.linspace(0, 40, 9), np.zeros(9), np.zeros(9)]
tr.add_root("lad", "LAD", "LAD", CenterLine.from_points(pts, step_mm=0.5).points)
tr.add_root("rca", "RCA", "RCA",
            CenterLine.from_points(pts + [0, 30, 0], step_mm=0.5).points)
w._full_lv_data = {"fake": True}
w._compute_territory(calc=True)
check("(setup) Territory calculated", w._territory is not None)
n = tr.vessels["lad"].n
w.add_target("lad", n // 4)
w.add_target("lad", n // 2)
check("(setup) two targets", len(w._targets) == 2)

w.open_target_manager()
pump()
dlg = w._target_mgr
tb = dlg._tbl
check("window opened (separate, non-modal)",
      dlg.isVisible() and not dlg.isModal())
check("columns: 表示 No. 名称 心筋% 灌流域mL 元血管名 ターゲット位置",
      [tb.horizontalHeaderItem(c).text() for c in range(7)]
      == [t("表示"), t("No."), t("名称"), t("心筋%"), t("灌流域mL"),
          t("元血管名"), t("ターゲット位置")])
check("one row per target", tb.rowCount() == 2)
tg0 = w._targets[0]
check("row 1: No. / 心筋% / mL / vessel",
      tb.item(0, 1).text() == "1"
      and tb.item(0, 3).text() == f"{tg0['pct']:.1f}%"
      and tb.item(0, 4).text() == f"{tg0['ml']:.1f}"
      and tb.item(0, 5).text() == "LAD")
check("row 1: position from the proximal end (% and mm)",
      tb.item(0, 6).text().startswith("25%") and "mm" in tb.item(0, 6).text(),
      tb.item(0, 6).text())

# name the first target
check("before: the VR result says Target-1",
      w._target_summary_text().splitlines()[0].startswith("Target-1"))
tb.item(0, 2).setText("D1 distal")
pump()
check("name stored", w._targets[0].get("name") == "D1 distal")
lines = w._target_summary_text().splitlines()
check("the VR result shows the name instead of Target-1",
      lines[0].startswith("D1 distal ") and "Target-1" not in lines[0], lines)
check("… the unnamed one stays Target-2", lines[1].startswith("Target-2"))

# show / hide one, all
tb.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
pump()
check("checkbox hides target 2", w._targets[1].get("hidden") is True)
check("… and its line leaves the VR result",
      len(w._target_summary_text().splitlines()) == 1)

def boxes():
    return [tb.item(r, 0).checkState() == Qt.CheckState.Checked
            for r in range(tb.rowCount())]


dlg._toggle_all()
pump()
check("全ターゲットを非表示: nothing drawn", w._target_summary_text() == "")
check("… every checkbox unchecked (matches the display)",
      boxes() == [False, False], boxes())
check("… the button now offers 全ターゲットを表示",
      dlg._all_btn.text() == t("全ターゲットを表示"))
dlg._toggle_all()
pump()
check("全ターゲットを表示: both drawn, both checked",
      len(w._target_summary_text().splitlines()) == 2
      and boxes() == [True, True], boxes())
# the tab's own 表示/非表示 button does the same (one set of flags)
w._toggle_targets_shown()
pump()
check("tab 非表示 → the window's checkboxes follow", boxes() == [False, False])
w._toggle_targets_shown()
pump()
tb.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
pump()
check("one unchecked → only that one hidden",
      [tg["hidden"] for tg in w._targets] == [True, False])
tb.item(0, 0).setCheckState(Qt.CheckState.Checked)
pump()

# show / hide = DRAW only: no recompute, no full overlay (tube) rebuild
pushes = {"overlay": 0, "targets": 0, "terr": 0}
_orig = (w._push_overlay, w._push_targets_only, w._push_territory)
w._push_overlay = lambda: pushes.__setitem__("overlay", pushes["overlay"] + 1)
w._push_targets_only = lambda: pushes.__setitem__("targets", pushes["targets"] + 1)
w._push_territory = lambda: (pushes.__setitem__("terr", pushes["terr"] + 1),
                             _orig[2]())
n0 = len(ENG_CALLS)
w._targets_shown_before = True
for _ in range(3):
    w.toggle_target_n(1)
    w._toggle_targets_shown()
    w.set_target_name(2, "x%d" % _)
pump()
check("show / hide / rename: no full overlay rebuild",
      pushes["overlay"] == 0 and pushes["targets"] > 0 and pushes["terr"] > 0,
      pushes)
check("… and no territory recompute (cached masks)",
      len(ENG_CALLS) == n0, ENG_CALLS[n0:])
w._push_overlay, w._push_targets_only, w._push_territory = _orig
w.set_target_name(2, "")
for tg in w._targets:
    tg["hidden"] = False
w._refresh_targets_table()

# a SELECTED row in the CoronaryTree tab no longer limits the tint: the
# show checkboxes alone decide
eng = w._territory
eng.centers = np.array([[1.0, 1, 1], [3.0, 3, 3], [5.0, 5, 5], [7.0, 7, 7]])
eng.assignment = {"code": np.array([0, 0, 0, 0]), "code_to_vid": ["lad"]}
i1, i2 = w._targets[0]["idx"], w._targets[1]["idx"]
eng.territory = lambda vid, idx: (np.array([idx == i1, idx == i2, False, False]),
                                  10.0)
w._terr_mask_cache = None
w._terr_base_cache = None
w._full_lv_data = {"epi": {"spacing": [1.0, 1.0, 1.0],
                           "region": {"vol_shape": [10, 10, 10]}}}
w._territory_on = True
w._targets_w.setCurrentItem(w._targets_w.topLevelItem(0))
lab, _cols = w._territory_systems_full()
labs = set(np.unique(lab).tolist()) if lab is not None else set()
check("row 1 selected in the tab: BOTH checked targets are tinted",
      {5, 6} <= labs, labs)
w.toggle_target_n(2)
lab, _cols = w._territory_systems_full()
check("… an unchecked one is not", 6 not in set(np.unique(lab).tolist()))
w.toggle_target_n(2)
w._targets_w.setCurrentItem(None)

# every column width can be dragged
from PyQt6.QtWidgets import QHeaderView  # noqa: E402
hh = tb.horizontalHeader()
check("columns are user-resizable",
      all(hh.sectionResizeMode(c) == QHeaderView.ResizeMode.Interactive
          for c in range(7)))

# the name survives a recompute and goes into the .territory.json
w._recompute_targets()
check("name kept by a target recompute", w._targets[0].get("name") == "D1 distal")
data, err = territory_file.build(
    src_meta={}, corotree_path=os.path.join(HOME, "a.corotree.json"),
    full_lv_path=os.path.join(HOME, "a.FullLv.json"),
    targets=[dict(x) for x in w._targets])
check("saved with the name",
      data is not None and territory_file.targets(data)[0].get("name")
      == "D1 distal", err)
check("… an unnamed target has no name field",
      "name" not in territory_file.targets(data)[1])

# 除外 on RCA: the calculated Territory stays (no blank → the VR stays)
eng = w._territory
w._set_excluded(["rca"], True)
pump()
check("除外: the Territory result is kept", w._territory is eng)
check("… the user is told it counts at the next Calc",
      "Territory Calc" in w._hint.text(), w._hint.text())
check("… the vessel is marked excluded", tr.vessels["rca"].excluded)

# delete
tb.selectRow(1)
dlg._delete_selected()
pump()
check("delete the selected target", len(w._targets) == 1 and tb.rowCount() == 1)
dlg.close()

# main window routing: "manage" opens the window
from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402


class _Shell:
    _corotree_win = w
    coronary_target_action = MainWindow.coronary_target_action


opened = []
w.open_target_manager = lambda: opened.append(1)
_Shell().coronary_target_action(0, "manage")
check("shell: action 'manage' opens Territory Manage", opened == [1])

# row 3: [Territory Calc][Territory Manage]; the VR result right-click
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

s = LoadedSeries(modality=Modality.CT, volume=np.full((20, 40, 40), -100, np.int16),
                 spacing_mm=(0.7, 0.7), cine_fps=None, window=800.0, level=200.0,
                 slice_mm=0.7, patient_basis=None, series_uid="tm-ct")
v = CTViewer()
v.resize(1200, 700)
v.show()
v.load_series(s, "s")
pump()
mb = getattr(v, "_ct_terr_mgr_btn", None)
check("row 3 has 「Territory Manage」", mb is not None
      and mb.text() == t("Territory Manage"))
if mb is not None:
    g1 = v._ct_terr_btn.mapTo(v, v._ct_terr_btn.rect().topRight())
    g2 = mb.mapTo(v, mb.rect().topLeft())
    check("… right of Territory Calc",
          0 <= g2.x() - g1.x() <= 12 and abs(g2.y() - g1.y()) <= 4, (g1, g2))
    got = []

    class _S2:
        def coronary_target_action(self, n, a):
            got.append((n, a))
    v._ct_mode_shell = lambda: _S2()
    v._ct_territory_manage()
    check("… its click asks the shell for 'manage'", got == [(0, "manage")])

p = v.pane["B"]
v._tgt_summary_text = "D1 distal 12.0% / 12.0mL"
p.tgtact.SetInput(v._tgt_summary_text)
p.tgtact.SetVisibility(True)
p.render()
pump()
bb = [0.0, 0.0, 0.0, 0.0]
p.tgtact.GetBoundingBox(p.ren, bb)
dpr = max(1.0, p.canvas.devicePixelRatioF())
_W, H = p.ren.GetRenderWindow().GetSize()
cx = (bb[0] + bb[1]) / 2 / dpr
cy = (H - (bb[2] + bb[3]) / 2) / dpr
check("VR: a right-click ON the target result is a hit",
      v._vr_tgt_text_hit("B", cx, cy), (bb, cx, cy))
check("… elsewhere is not", not v._vr_tgt_text_hit("B", 10, 10))

# viewer: a uint8 map is used as is; the same map again does nothing; a
# TARGET-only change (labels 5..10) is applied (the old sig missed it)
lab = np.zeros(v._vol.shape, np.uint8)
lab[5:10, 10:20, 10:20] = 1
cols = [(1, 0, 0, .4)] * 10
v.set_territory_mask(lab, cols)
check("viewer: a uint8 label map is not copied", v._terr_label_np is lab)
img1 = v._terr_mask_vol
v.set_territory_mask(lab.copy(), cols)
check("… the same map again: nothing rebuilt", v._terr_mask_vol is img1)
lab2 = lab.copy()
lab2[6:8, 12:14, 12:14] = 5                       # a target on top
v.set_territory_mask(lab2, cols)
check("… a target change is applied", v._terr_label_np is lab2
      and v._terr_mask_vol is not img1)

src = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text("utf-8")
check("pygfx: the result rect is kept and its right-click opens the window",
      "v._tgt_text_rect[key] = (x, y, maxw + 6, bh)" in src
      and 'self._coro_target_action_cb(0, "manage")' in src)
v.close()
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
os._exit(1 if fails else 0)
