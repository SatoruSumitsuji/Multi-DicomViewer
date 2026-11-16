"""CT bar / overlay rules (2026-11-16):
- LVモード: row 2 starts with 「LV: 壁厚3D 壁厚短軸 CoronaryTree」 (left of
  CoronaryTree:), only in LVモード (no LV edit step); 壁厚 like Territory
  (coloured ON, exclusive), CoronaryTree draws the tree over the LV.
- Territory Target points: Territoryモード only.
- No meaningless text: 2-D labels never on a VR pane; an internal point marker
  (LV apex …) keeps its "+" but loses the "#id".
- Open VR / 回転連動 show with the CoronaryTree: group (before CPR too).

    python tools/test_ct_lv_group.py
"""
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

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.viewers import ct_viewer as ctv  # noqa: E402
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


v = CTViewer()
lay = v._coro_cap.parentWidget().layout()


def pos(w):
    for i in range(lay.count()):
        sub = lay.itemAt(i).layout()
        if sub is None:
            continue
        for j in range(sub.count()):
            if sub.itemAt(j).widget() is w:
                return i, j
    return None


grp = v._ctlv_group
p_lv = [pos(w) for w in grp]
p_cor = pos(v._coro_cap)
check("LV: 壁厚3D 壁厚短軸 CoronaryTree, in that order, left of CoronaryTree:",
      None not in p_lv and p_cor is not None
      and len({p[0] for p in p_lv + [p_cor]}) == 1
      and [p[1] for p in p_lv] == sorted(p[1] for p in p_lv)
      and p_lv[-1][1] < p_cor[1], f"{p_lv} / {p_cor}")


class Shell:
    tree = True
    shown = []

    def ct_tree_overlay_on(self, viewer):
        return False if self.tree else None

    def ct_tree_show(self, viewer, on, raise_tab=False):
        self.shown.append(on)

    def ct_mode_state(self, viewer):
        return {}


sh = Shell()
v._ct_mode_shell = lambda: sh
mode = {"m": "standard"}
v._ct_current_mode = lambda: mode["m"]
v._lv_mode_active = lambda: False
v._mode = "3D"

for m, want in (("standard", False), ("coronary", False), ("lv", True)):
    mode["m"] = m
    v._ctlv_sync()
    check(f"{m}: LV group {'shown' if want else 'hidden'}",
          all(w.isHidden() != want for w in grp))
v._lv_mode_active = lambda: True
v._ctlv_sync()
check("LVモード with an LV edit step open: group hidden",
      all(w.isHidden() for w in grp))
v._lv_mode_active = lambda: False

# 壁厚 like Territory: coloured, exclusive (the LV handler faked)
v._lvv_thick_mode = "3d"
v._ctlv_sync()
check("壁厚3D ON: checked + coloured, 壁厚短軸 greyed",
      v._ctlv_thick3d_btn.isChecked() and "#ffd040" in v._ctlv_thick3d_btn.styleSheet()
      and not v._ctlv_thicksax_btn.isEnabled())
v._lvv_thick_mode = None
v._ctlv_sync()
check("壁厚 OFF: both clickable", v._ctlv_thick3d_btn.isEnabled()
      and v._ctlv_thicksax_btn.isEnabled() and not v._ctlv_thick3d_btn.isChecked())
calls = []
v._lvv_set_thick_mode = lambda m: calls.append(m)
v._ctlv_thicksax_btn.click()
check("壁厚短軸 button → the shared 壁厚 handler", calls == ["sax"])

# CoronaryTree over the LV
applied = []
v._ct_tree_apply = lambda: applied.append(v._ct_tree_suppressed)
v._ct_tree_suppressed = True
v._ctlv_tree_btn.click()
check("CoronaryTree ON: tree drawn over the LV (dock ツリー表示 on)",
      v._ct_lv_tree_on and applied[-1] is False and sh.shown[-1] is True)
check("CoronaryTree ON: coloured background",
      v._ctlv_tree_btn.isChecked() and "#b8860b" in v._ctlv_tree_btn.styleSheet())
v._ctlv_tree_btn.click()
check("CoronaryTree OFF: tree hidden again", not v._ct_lv_tree_on
      and applied[-1] is True and not v._ctlv_tree_btn.isChecked())
sh.tree = False
v._ctlv_sync()
check("no tree: CoronaryTree button greyed", not v._ctlv_tree_btn.isEnabled())
sh.tree = True
src = (ROOT / "multi_dicomviewer/viewers/ct_mode_bar.py").read_text(encoding="utf-8")
check("LVモード re-entry keeps the tree when its toggle is on",
      'or (key == "lv" and getattr(self, "_ct_lv_tree_on", False)))' in src)

# Targets: Territory only
v._territory_display = False
check("targets hidden outside Territory", not v._ct_targets_visible())
v._territory_display = True
check("targets shown in Territory", v._ct_targets_visible())
v._territory_display = False

# Labels: never on a VR pane; internal markers without "#id"
v._vr_on = {"A": False, "B": True}
v._rebuild_labels(v.pane["B"], [("#10", (0.0, 0.0))])
v._rebuild_labels(v.pane["A"], [("#11", (0.0, 0.0))])
check("measurement label hidden on the VR pane",
      not v.pane["B"].meas_labels[0].GetVisibility())
check("measurement label kept on an MPR pane",
      bool(v.pane["A"].meas_labels[0].GetVisibility()))
v._rebuild_coro_labels(v.pane["B"], [("1", (0.0, 0.0), (255, 0, 0), False)])
check("target number hidden on the VR pane",
      not v.pane["B"].coro_labels[0].GetVisibility())
v._vr_on = {"A": False, "B": False}
check("internal marker (LV apex) → no #id",
      ctv._internal_marker({"_lvv": "apex"}) and ctv._internal_marker({"_lv_valve": "mitral"})
      and not ctv._internal_marker({"type": "point"}))

# Open VR / 回転連動: CoronaryTreeモード tools (before CPR too); not in
# Simple View (2026-11-17: there only the Load / CPR entry shows).
v._territory_display = False
v._coronary_mode = False
v._cpr = None
v._bottom_ctx = "both"
v._coronary_mpr_btn.setVisible(True)
mode["m"] = "coronary"
v._cpr_vr_group_sync()
check("Open VR / 回転連動 shown before entering CPR (CoronaryTreeモード)",
      not v._cpr_vr_btn.isHidden() and not v._cpr_vr_rot_btn.isHidden())
mode["m"] = "standard"
v._cpr_vr_group_sync()
check("…not in Simple View", v._cpr_vr_btn.isHidden())
mode["m"] = "coronary"
v._territory_display = True
v._cpr_vr_group_sync()
check("…not in the Territory review", v._cpr_vr_btn.isHidden())

mac = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(encoding="utf-8")
check("Mac: same LV group + target / marker rules",
      "self._ctlv_tree_btn = FitButton(t(\"CoronaryTree\"))" in mac
      and mac.count("_ct_targets_visible()") >= 3
      and "if not _internal_marker(m):" in mac)

tw = (ROOT / "multi_dicomviewer/ui/coronary_tree_window.py").read_text(encoding="utf-8")
check("CoronaryTree tab: no Territory VR row",
      "_vr_show_btn" not in tw and 't("Territory VR：")' not in tw
      and "def _ensure_vr_on" in tw)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
