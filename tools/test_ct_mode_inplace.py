"""CT mode buttons, in-place (CTModeBarMixin): 標準 / LV / CoronaryTree switch by
hiding / restoring overlays (data kept), availability from in-session data,
auto modes, the tree view in LVモード; 「Territory計算」 (CoronaryTreeモード only)
and the Territory review entered / left in place (no CT re-open).

Fake viewer + fake shell; no real dialogs (QMessageBox patched).
"""
import ast
import os
import sys
from pathlib import Path

import numpy as np

import tempfile  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_home = tempfile.mkdtemp(prefix="mdv_home_")       # never the real settings
os.environ["HOME"] = os.environ["USERPROFILE"] = _home
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtWidgets import QApplication, QHBoxLayout, QMessageBox, QWidget  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.viewers.ct_mode_bar import CTModeBarMixin  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


class Shell:
    def __init__(self):
        self.state = {"full": "", "tree": ""}
        self.overlay_on = True
        self.shows, self.set_modes = [], []
        self.computed, self.left = [], []

    def ct_mode_state(self, v):
        return dict(self.state)

    def ct_tree_overlay_on(self, v):
        has = self.state.get("tree") or self.state.get("tree_mem")
        return self.overlay_on if has else None

    def ct_tree_show(self, v, on, raise_tab=False):
        has = self.state.get("tree") or self.state.get("tree_mem")
        if has:
            self.shows.append((bool(on), raise_tab))
            self.overlay_on = bool(on)

    def ct_set_mode(self, mode, v):
        self.set_modes.append(mode)

    def ct_territory_compute(self, v):
        full = v._ct_full_lv_build() or {"file": 1}
        self.computed.append(full)
        self.state["terr"] = True
        return v._ct_territory_enter(full)

    def ct_territory_show(self, v):
        return v._ct_territory_enter({"file": 1})

    def ct_territory_left(self, v):
        self.left.append(True)


class V(QWidget, CTModeBarMixin):
    def __init__(self, shell):
        super().__init__()
        self.shell = shell
        self._territory_display = False
        self._bottom_ctx = "both"
        self._ct_mode_reset()
        # LV data + overlays as after a finished LV analysis
        self._lv = None
        self._lvv = {"apex": (0, 0, 0)}
        self._lvv_epi_surf = None
        self._lv_valves = {"mitral": ((0, 0, 0), (0, 0, 1), 15.0), "aortic": None}
        self._lv_apex = np.zeros(3)
        self._lv_border_show = False
        self._lvv_epi_show = True
        self._lvv_mask_on = True            # LV-Blood (in a _lvv session)
        self._lvv_hl_on = False
        self._lvv_endo_show = True
        self._lvv_lvd_shown = False
        self._lv_valve_shown = {"mitral": True, "aortic": True}
        self._lv_apex_shown = True
        self._lvv_apex_shown = True
        self._measures = {"A": [{"_lv_valve": "mitral"}, {"_lvv": "apex"}],
                          "B": []}
        self._coronary_mode = False
        self._edit_active = False
        self.calls = []
        self._coro_overlay_req = None
        self._coro_overlay = None
        # view state + the Territory review plumbing (the viewers' real names)
        self._mode = "3D"
        self._frame = {"A": (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]),
                       "B": (np.eye(3)[1], np.eye(3)[2], np.eye(3)[0])}
        self._pc = {"A": np.array([1.0, 2.0, 3.0]),
                    "B": np.array([4.0, 5.0, 6.0])}
        self._cross_ang = {"A": 0.3, "B": 0.0}
        self._thick = {"A": 0.0, "B": 5.0}
        self._center = np.array([7.0, 8.0, 9.0])
        self._vr_on = {"A": False, "B": False}
        self._lv_missing = []
        self._applied = []
        row = QHBoxLayout(self)
        self._build_ct_mode_buttons(row)
        row.addWidget(self._build_ct_files_line())

    def _ct_mode_shell(self):
        return self.shell

    # LV setters (the viewers' real names)
    def _lv_set_epi_blood_border(self, on):
        self.calls.append(("epi", on)); self._lvv_epi_show = on

    def _lv_set_auto_endo(self, on):
        self.calls.append(("endo", on)); self._lvv_endo_show = on

    def _lv_set_blood(self, mode):
        self.calls.append(("blood", mode))
        self._lvv_mask_on = mode == "lv"; self._lvv_hl_on = mode == "all"

    def set_lv_border_visible(self, on):
        self.calls.append(("border", on)); self._lv_border_show = on

    def _lv_apex_marker_draw(self):
        pass

    def _lv_mode_active(self):
        return self._edit_active

    def _coronary_exit(self):
        self.calls.append(("cpr_exit",)); self._coronary_mode = False

    def set_coronary_overlay(self, spec):
        self._coro_overlay_req = spec or None
        self._coro_overlay = None if self._ct_tree_suppressed else (spec or None)

    def _full_lv_missing(self):
        return list(self._lv_missing)

    def _full_lv_bld_for_save(self):
        return {"endo": {}}

    def _full_lv_epi_dict(self):
        return {"region": {}}

    def _lvv_setup_axis_views(self):
        self.calls.append(("axis",))
        self._frame = {"A": ("lvA",), "B": ("lvB",)}
        self._pc = {"A": np.zeros(3), "B": np.zeros(3)}
        self._center = np.zeros(3)

    def _lvv_observe_exit(self):
        self._lvv["observe"] = True

    def _refresh(self, reset_cam=False):
        self.calls.append(("refresh", reset_cam))

    def _set_mode(self, m):
        self._mode = m

    def _init_frames(self):
        self.calls.append(("init_frames",))

    def _vr_set(self, k, on):
        self._vr_on[k] = bool(on)
        if not on and not any(self._vr_on.values()):
            self._territory_display = False

    def _terr_bar_sync(self):
        self.calls.append(("terr_bar", self._territory_display))

    def apply_full_lv(self, full):
        self._applied.append(full)
        self._lvv_setup_axis_views()
        self._territory_display = True
        return True


sh = Shell()
v = V(sh)
b = v._ct_mode_btns
v.refresh_ct_mode_bar()
check("LV measured in session (no file) → LVモード selected",
      v._ct_current_mode() == "lv" and b["lv"].isChecked()
      and not b["standard"].isChecked())
check("enabled: 標準 + LV; CoronaryTree / Territory not (no tree)",
      b["standard"].isEnabled() and b["lv"].isEnabled()
      and not b["coronary"].isEnabled() and not b["territory"].isEnabled())
check("readout says measured (unsaved)", t("計測済み（未保存）") in v._ct_files_lbl.text(),
      v._ct_files_lbl.text())

# 標準: everything hidden, data kept
v._ct_mode_clicked("standard")
check("標準: LV overlays hidden, data kept",
      v._ct_current_mode() == "standard" and v._ct_lv_hidden
      and not v._lvv_epi_show and not v._lvv_endo_show and not v._lvv_mask_on
      and v._lvv is not None and v._lv_valves["mitral"] is not None,
      str(v.calls))
check("標準: valve ring + apex hidden (flags)",
      v._lv_valve_shown["mitral"] is False and v._lv_apex_shown is False
      and v._measures["A"][0].get("hidden") is True
      and v._measures["A"][1].get("hidden") is True)
check("標準: the Epi border that was OFF is not touched",
      ("border", False) not in v.calls and ("border", True) not in v.calls)
check("標準: LV still selectable", b["lv"].isEnabled() and b["standard"].isChecked())

# back to LV: exactly what was shown
v.calls.clear()
v._ct_mode_clicked("lv")
check("LV: overlays restored exactly (LV-Blood, Epi, Endo; border stays off)",
      v._lvv_epi_show and v._lvv_endo_show and v._lvv_mask_on
      and not v._lvv_hl_on and not v._lv_border_show and not v._ct_lv_hidden,
      str(v.calls))
check("LV: rings + apex back", v._lv_valve_shown["mitral"] is True
      and v._lv_apex_shown and v._measures["A"][0].get("hidden") is False)

# a tree is registered → CoronaryTree enabled; tree lines suppressed in 標準
sh.state["tree_mem"] = True
v.set_coronary_overlay([{"vid": 1}])
v.refresh_ct_mode_bar()
check("tree registered → CoronaryTree selectable",
      b["coronary"].isEnabled()
      and t("ツリーあり（未保存）") in v._ct_files_lbl.text())
v._ct_mode_clicked("coronary")
check("CoronaryTree: LV hidden, tab raised + ツリー表示 ON",
      v._ct_current_mode() == "coronary" and v._ct_lv_hidden
      and sh.shows[-1] == (True, True) and v._coro_overlay is not None)
v._ct_mode_clicked("standard")
check("標準: tree lines hidden here (kept)", v._coro_overlay is None
      and v._coro_overlay_req is not None)
v._ct_mode_clicked("lv")
check("LV with a tree: tree drawn over the LV, tab raised",
      v._coro_overlay is not None and sh.shows[-1] == (True, True)
      and not v._ct_lv_hidden)
sh.overlay_on = False                      # the user turns ツリー表示 off in LV
v._ct_mode_clicked("coronary")
check("CoronaryTree forces ツリー表示 ON", sh.shows[-1] == (True, True))
v._ct_mode_clicked("lv")
check("back in LV: ツリー表示 OFF remembered for LVモード",
      sh.shows[-1] == (False, True))

# an LV edit step blocks a manual switch away from LV (dialog patched)
asked = []
_orig = QMessageBox.information
QMessageBox.information = staticmethod(lambda *a, **k: asked.append(a))
try:
    v._edit_active = True
    v._ct_mode_clicked("standard")
    check("LV edit in progress: switch refused with a message",
          v._ct_current_mode() == "lv" and len(asked) == 1)
finally:
    QMessageBox.information = _orig
    v._edit_active = False

# auto modes
v._ct_mode_clicked("standard")
v._ct_mode_auto("lv")
check("LV edit step from 標準 → LVモード (auto)", v._ct_current_mode() == "lv"
      and not v._ct_lv_hidden)
v._coronary_mode = True                    # CPR button pressed
v._ct_mode_auto("coronary")
check("CPR → CoronaryTreeモード (auto), CPR session kept",
      v._ct_current_mode() == "coronary" and v._coronary_mode)
v._ct_mode_clicked("lv")
check("leaving CoronaryTree ends the CPR session",
      ("cpr_exit",) in v.calls and not v._coronary_mode)

# 「Territory計算」: shown only in CoronaryTreeモード; enabled with a tree + a
# complete LV (in-session) or a FullLv.
import multi_dicomviewer.core.full_lv as _flm  # noqa: E402
_flm.build = lambda epi, bld: ({"epi": epi, "bld": bld, "mem": True}, "")
from PyQt6.QtCore import QTimer  # noqa: E402
_ss = QTimer.singleShot
single = []


def run_deferred(fn, *a):
    single.clear()
    QTimer.singleShot = staticmethod(lambda ms, f: single.append(f))
    try:
        fn(*a)
    finally:
        QTimer.singleShot = _ss
    for f in single:
        f()


tb = v._ct_terr_btn
v._ct_mode_clicked("coronary")
v.refresh_ct_mode_bar()
check("Territory計算 visible in CoronaryTreeモード, enabled (tree + LV ready)",
      not tb.isHidden() and tb.isEnabled())
check("Territoryモード not selectable before a result",
      not b["territory"].isEnabled())
v._ct_mode_clicked("lv")
check("Territory計算 hidden in LVモード", tb.isHidden())
v._ct_mode_clicked("standard")
check("Territory計算 hidden in 標準モード", tb.isHidden())
v._ct_mode_clicked("coronary")
v._lv_missing = ["Epi(region)"]
v.refresh_ct_mode_bar()
check("Territory計算 disabled while the LV is incomplete and no FullLv",
      not tb.isEnabled())
v._lv_missing = []
v.refresh_ct_mode_bar()

pre_center = v._center.copy()
pre_a = v._frame["A"]
v._lvv_epi_surf = object()                 # a complete in-session LV
v.calls.clear()
sh.set_modes.clear()
run_deferred(tb.click)
check("Territory計算 → FullLv built in memory from the in-session LV",
      len(sh.computed) == 1 and sh.computed[0].get("mem") is True,
      str(sh.computed)[:120])
check("in place: Territory review on (light path: LV kept, no apply_full_lv)",
      v._ct_current_mode() == "territory" and not v._applied
      and ("axis",) in v.calls and v._vr_on["B"] and not sh.set_modes)
check("review shows the LV (+ tree) again", not v._ct_lv_hidden
      and v._coro_overlay is not None)
check("Territoryモード checked + Territory計算 hidden in the review",
      b["territory"].isChecked() and tb.isHidden())
v.calls.clear()
v._ct_mode_clicked("coronary")
check("leave → VR off, previous view restored (frames / centre), refit",
      not v._vr_on["B"]
      and all(np.allclose(x, y) for x, y in zip(v._frame["A"], pre_a))
      and np.allclose(v._center, pre_center) and ("refresh", True) in v.calls
      and sh.left == [True] and not sh.set_modes, str(v.calls))
check("back in CoronaryTreeモード (LV hidden, tree on)",
      v._ct_current_mode() == "coronary" and v._ct_lv_hidden
      and v._coro_overlay is not None)
v.refresh_ct_mode_bar()
check("result kept → Territoryモード selectable", b["territory"].isEnabled())
v._ct_mode_clicked("lv")
run_deferred(v._ct_mode_clicked, "territory")
check("Territoryモード from LV: shown in place, not recomputed",
      v._ct_current_mode() == "territory" and len(sh.computed) == 1)
v._ct_mode_clicked("lv")
check("Territory → LV: LV shown, view back",
      v._ct_current_mode() == "lv" and not v._ct_lv_hidden
      and np.allclose(v._center, pre_center))
v._lv_missing = ["Blood/Endo"]
run_deferred(v._ct_mode_clicked, "territory")
check("no complete in-session LV → apply_full_lv(FullLv) path",
      v._applied == [{"file": 1}] and v._ct_current_mode() == "territory")
v._ct_mode_clicked("standard")
check("Territory → 標準: plain, view restored",
      v._ct_current_mode() == "standard" and v._ct_lv_hidden
      and np.allclose(v._center, pre_center))
v._lv_missing = []

# a fresh series resets
v._ct_mode_reset()
check("reset: nothing hidden, mode derived", v._ct_mode_sel is None
      and not v._ct_lv_hidden and not v._ct_tree_suppressed)

# Both viewers: set_coronary_overlay keeps the request and hides it while
# _ct_tree_suppressed; the LV readouts honour _ct_lv_hidden.
root = Path(__file__).resolve().parents[1] / "multi_dicomviewer" / "viewers"
for fn in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    src = (root / fn).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "set_coronary_overlay")
    ns = {}
    exec(compile(ast.Module([node], []), fn, "exec"), ns)
    f = ns["set_coronary_overlay"]

    class O:
        _vr_on = {}
        _overlay = {}
        _ct_tree_suppressed = True

        def _redraw_geom(self, k):
            pass

        def _lv_update_submode_ui(self):
            pass
    o = O()
    f(o, [{"vid": 1}])
    ok1 = o._coro_overlay is None and o._coro_overlay_req == [{"vid": 1}]
    o._ct_tree_suppressed = False
    f(o, [{"vid": 1}])
    check(f"{fn}: tree overlay hidden while suppressed, kept as requested",
          ok1 and o._coro_overlay == [{"vid": 1}])
    check(f"{fn}: LV readouts gated by _ct_lv_hidden",
          src.count("_ct_lv_hidden") >= 1 and "_ct_mode_reset()" in src)

# FullLv save: no automatic Territory any more (both viewers).
for fn in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    src = (root / fn).read_text(encoding="utf-8")
    check(f"{fn}: FullLv save no longer enters Territory",
          "on_full_lv_created" not in src
          and "_full_lv_saved_path = path" in src)

# 「Territory計算」 (Calc Territory) is not in the mode row: the viewers put it
# at the right end of the Coronary / VR / LV row.
check("Territory計算 not added to the mode row",
      "row.addWidget(tb)" not in (root / "ct_mode_bar.py").read_text(encoding="utf-8"))
for fn in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    src = (root / fn).read_text(encoding="utf-8")
    check(f"{fn}: Territory計算 at the end of the Coronary row",
          "row1.addWidget(self._ct_terr_button())\n"
          "        row1.addStretch(1)" in src)

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
