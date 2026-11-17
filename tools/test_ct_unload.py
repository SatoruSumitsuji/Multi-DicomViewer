"""Loaded Files ⊗ Unload + the CoronaryTree tab closed (2026-11-17).

  - each loaded FILE (FullLv / corotree / territory) has a ⊗ button; none for
    「なし」 / 未保存 entries
  - Unload keeps the data but it counts as not loaded and is not shown
    (the tree is not drawn, the Territory goes with its corotree / FullLv,
    the LV overlays hide and LVモード is left); loading a file of that kind
    shows it again
  - the CoronaryTree tab closed → no tree drawn on the CT
  - the right pane's depth peeling is on only while it shows the VR (left
    on, it hid the blood on that pane after the Territory review)
  - en: "Loaded Files"

    python tools/test_ct_unload.py
"""
import json
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

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
asked = []
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(
    lambda *a, **k: asked.append(a) or QMessageBox.StandardButton.No)
from multi_dicomviewer.core.coronary_territory import CoronaryTree  # noqa: E402
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.ui.coronary_tree_window import CoronaryTreeWindow  # noqa: E402
from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402
from multi_dicomviewer.viewers import ct_viewer as ctv  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


# ---- dock: Unload keeps the data, counts as not loaded --------------------
D = tempfile.mkdtemp(dir=HOME)
coro_p = os.path.join(D, "case.corotree.json")
tree = CoronaryTree()
tree.add_root("lad", "LAD", "LAD",
              np.c_[np.linspace(0, 29.3, 30), np.zeros(30), np.zeros(30)])
with open(coro_p, "w", encoding="utf-8") as f:
    json.dump(tree.to_json(series={"series_uid": "ul-ct"}), f)
w = CoronaryTreeWindow(None)
check("dock loads the corotree", w.load_file(coro_p))
uid = "ul-ct"
w._overlay_on = True
b = w._bundle()
b["full_lv"] = {"format": "MDV-FullLv"}
b["full_lv_path"] = os.path.join(D, "case.FullLv.json")
b["territory"] = object()
b["terr_path"] = os.path.join(D, "case.territory.json")
check("loaded: tree / FullLv / territory present",
      w.tree_vessels_for_series(uid) and w.full_lv_for_series(uid)
      and w.territory_for_series(uid) and w.overlay_spec())
w.unload_for_series(uid, "tree")
check("tree Unloaded: not loaded, not drawn, data kept",
      not w.tree_vessels_for_series(uid)
      and w.corotree_path_for_series(uid) == ""
      and w.overlay_spec(force=True) == [] and b["tree"].vessels)
check("… and its Territory goes with it",
      not w.territory_for_series(uid)
      and w.territory_path_for_series(uid) == ""
      and b["territory"] is not None)
check("FullLv still loaded", bool(w.full_lv_for_series(uid)))
w._last_path = coro_p                 # loading a corotree again
check("tree loaded again → shown again",
      w.tree_vessels_for_series(uid) and w.territory_for_series(uid))
w.unload_for_series(uid, "full")
check("FullLv Unloaded: not loaded (data kept), Territory goes too",
      w.full_lv_for_series(uid) is None and w.full_lv_path_for_series(uid) == ""
      and b["full_lv"] is not None and not w.territory_for_series(uid))
w._full_lv_data = {"format": "MDV-FullLv", "new": 1}   # a FullLv loaded
check("a FullLv loaded → back (replaced)",
      w.full_lv_for_series(uid) == {"format": "MDV-FullLv", "new": 1})
w.unload_for_series(uid, "terr")
check("territory Unloaded only: tree + FullLv kept loaded",
      not w.territory_for_series(uid) and w.tree_vessels_for_series(uid)
      and w.full_lv_for_series(uid) is not None)
w._territory = object()               # a Calc / load
check("Calc / load → Territory back", w.territory_for_series(uid))

# ---- viewer: ⊗ buttons, FullLv Unload --------------------------------------
vol = np.full((40, 48, 48), 100.0, np.float32)
s = LoadedSeries(modality=Modality.CT, volume=vol, spacing_mm=(0.7, 0.7),
                 cine_fps=None, window=800.0, level=200.0, slice_mm=1.5,
                 patient_basis=None, series_uid="ul-ct")
v = ctv.CTViewer()
v.resize(1400, 800)
v.show()
v.load_series(s, "ul")
v._set_mode("3D")
app.processEvents()


class Shell:
    state = {}
    calls = []

    def ct_mode_state(self, viewer):
        return dict(self.state)

    def ct_unload_file(self, viewer, kind):
        self.calls.append(kind)


sh = Shell()
v._ct_mode_shell = lambda: sh


def vis(state):
    sh.state = state
    v.refresh_ct_mode_bar()
    return {k: not b.isHidden() for k, b in v._ct_unload_btns.items()}


check("⊗ next to every loaded file",
      vis({"full": "a.FullLv.json", "tree": "a.corotree.json", "terr": True,
           "terr_file": "a.territory.json"})
      == {"full": True, "tree": True, "terr": True})
check("no ⊗ for なし / 未保存",
      vis({"full": "", "tree": "", "tree_mem": True, "terr": True})
      == {"full": False, "tree": False, "terr": False}, v._ct_files_text())
vis({"full": "a.FullLv.json", "tree": "a.corotree.json"})
v._ct_unload_btns["tree"].click()
check("⊗ click → shell.ct_unload_file(kind)", sh.calls == ["tree"], sh.calls)
check("caption reads 「Loaded Files」 in English",
      json.loads((ROOT / "multi_dicomviewer/resources/i18n/en.json")
                 .read_text(encoding="utf-8"))["読込ファイル"] == "Loaded Files")

v._lv_apex = np.array([10.0, 20.0, 30.0])
check("an LV → LV data", v._ct_lv_data())
v._ct_mode_sel = "lv"
v._ct_lv_unload()
check("FullLv Unload: LVモード left, LV counts as absent, hidden",
      v._ct_current_mode() == "standard" and not v._ct_lv_data()
      and not v._ct_lv_ready() and v._ct_lv_hidden)
v.apply_lv_analysis({})
check("a FullLv load (apply) → LV back", v._ct_lv_data()
      and not v._ct_lv_unloaded)

# ---- shell: Unload asks only with unsaved changes --------------------------


class FakeW:
    _territory_on = False
    unloaded = []

    def unload_for_series(self, u, kind):
        self.unloaded.append(kind)

    def set_territory_color(self, on):
        pass


class Pane:
    def shown_series_uid(self):
        return "ul-ct"


class FakeShell:
    _corotree_win = FakeW()
    ct_unload_file = MainWindow.ct_unload_file
    _ct_leave_to_standard = MainWindow._ct_leave_to_standard
    st = {}
    refreshed = 0

    def ct_mode_state(self, viewer):
        return dict(self.st)

    def _pane_of_viewer(self, viewer):
        return Pane()

    def coronary_overlay_refresh(self):
        FakeShell.refreshed += 1

    def refresh_ct_mode_bars(self):
        pass


fs = FakeShell()
fs.st = {"tree": "a.corotree.json", "tree_dirty": True}
asked.clear()
check("unsaved changes → asked; No keeps it",
      fs.ct_unload_file(v, "tree") is False and len(asked) == 1
      and FakeW.unloaded == [])
fs.st = {"tree": "a.corotree.json"}
asked.clear()
ok = fs.ct_unload_file(v, "tree")
check("unchanged → no question, Unloaded, overlay re-pushed",
      ok and not asked and FakeW.unloaded == ["tree"]
      and FakeShell.refreshed == 1)

# ---- shell: the CoronaryTree tab closed → no tree pushed -------------------


class OV:
    got = "unset"
    vr = "unset"

    def set_coronary_overlay(self, spec):
        OV.got = spec

    def set_vr_coronary(self, spec):
        OV.vr = spec


class PaneOV:
    def current_viewer(self):
        return OV()

    def shown_series_uid(self):
        return "ul-ct"


class Shell2:
    coronary_overlay_refresh = MainWindow.coronary_overlay_refresh
    _corotree_dock_hidden = MainWindow._corotree_dock_hidden
    _panes = [PaneOV()]

    def __getattr__(self, n):
        if n.startswith("coronary_target"):
            return lambda *a, **k: None
        raise AttributeError(n)


s2 = Shell2()
s2._corotree_win = w
w._overlay_on = True
w.show()
app.processEvents()
s2.coronary_overlay_refresh()
check("tab open → the tree is pushed", bool(OV.got), OV.got)
w.hide()
app.processEvents()
s2.coronary_overlay_refresh()
check("tab closed (×) → no tree on the CT (2-D nor VR)",
      not OV.got and not OV.vr, (OV.got, OV.vr))
src = (ROOT / "multi_dicomviewer/ui/coronary_tree_window.py").read_text(
    encoding="utf-8")
check("the dock tells the shell when it is closed / opened",
      "def hideEvent" in src and "corotree_dock_shown_changed" in src)

# ---- depth peeling only while the VR is shown ------------------------------
pB = v.pane["B"]
v._vr_crop_mask = lambda: None
v._run_busy = lambda a, b_, fn: fn()
v._vr_set("B", True)
app.processEvents()
on_peel = bool(pB.ren.GetUseDepthPeeling())
v._vr_set("B", False)
app.processEvents()
check("depth peeling on with the VR, off again after it",
      on_peel and not pB.ren.GetUseDepthPeeling())

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
