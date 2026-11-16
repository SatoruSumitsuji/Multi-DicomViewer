"""MV-AoV method file consistency (2026-11-16):

1. the MV-AoV point is saved as .MVAoVLv.json (old .MVLv.json still loads);
2. a 5-file LV drop with the MV-AoV point needs only MV-AoV + Apex (no AoV);
3. LV Analysis 「保存」 needs only MV-AoV + Apex, writes .MVAoVLv.json with the
   point and no .AoVLv.json;
4. BldLv / FullLv carry the MV-AoV point; a FullLv re-opened for editing does
   not load the MV stand-in as an AoV plane;
5. an OLD EpiLv asks 変換 / そのまま / キャンセル (no real dialogs).

    python tools/test_lv_mvaov_files.py
"""
import ast
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402
from multi_dicomviewer.viewers.valve_auto_mixin import ValveAutoMixin  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


def src(path):
    return (ROOT / path).read_text(encoding="utf-8")


def seg(path, name):
    s = src(path)
    n = next(x for x in ast.walk(ast.parse(s))
             if isinstance(x, ast.FunctionDef) and x.name == name)
    return ast.get_source_segment(s, n)


VTK = "multi_dicomviewer/viewers/ct_viewer.py"
MAC = "multi_dicomviewer/viewers/ct_viewer_pygfx.py"
MIX = "multi_dicomviewer/viewers/valve_auto_mixin.py"

# ---- 1. names ---------------------------------------------------------------
check("MV+Apex save writes .MVAoVLv.json",
      '".MVAoVLv.json", self._lv_mv_payload()' in seg(MIX, "_lv_mvapex_save"))
ld = seg(MIX, "_lv_mvapex_load")
check("MV+Apex load finds .MVAoVLv.json, else old .MVLv.json",
      'find(".mvaovlv.json") or find(".mvlv.json")' in ld)
check("stem strips both names",
      ValveAutoMixin._lv_mvapex_stem("/a/case.MVAoVLv.json") == "/a/case"
      and ValveAutoMixin._lv_mvapex_stem("/a/case.MVLv.json") == "/a/case")
for f in (VTK, MAC):
    sv = seg(f, "_lv_save_valve")
    check(f"{Path(f).name}: single MV save = .MVAoVLv.json + the point",
          '".MVAoVLv.json" if which == "mitral"' in sv
          and "self._lv_mv_payload()" in sv)

# ---- 2. drop needs MV-AoV + Apex only ----------------------------------------
for f in (VTK, MAC):
    a = seg(f, "apply_lv_analysis")
    check(f"{Path(f).name}: drop with the MV-AoV point skips the AoV check",
          'isinstance(mvf.get("mvaov"), dict)' in a and "have_av = True" in a)
    check(f"{Path(f).name}: old Epi in a drop → convert prompt",
          'self._lv_epi_maybe_convert(files["epi"])' in a
          and "self._lv_epi_converted_notice(" in a)
    check(f"{Path(f).name}: Load converts an old Epi, left unsaved",
          "self._lv_epi_maybe_convert(data)" in seg(f, "_lv_load")
          and "self._lv_dirty = True" in seg(f, "_lv_load"))
    b = seg(f, "_lvv_build_dict")
    check(f"{Path(f).name}: BldLv MV carries the MV-AoV point",
          '"mvaov": dict(self._lv_mvaov)' in b)

# ---- 3. LV Analysis save -------------------------------------------------------
class FakeV:
    def __init__(self, mvaov=True):
        self._lv_valves = {"mitral": (np.zeros(3), np.array([0, 0, 1.0]), 12.0),
                           "aortic": None}
        self._lv_apex = np.array([0, 0, -60.0])
        self._mvaov = mvaov

    def _lv_mvaov_on(self):
        return self._mvaov

    def _lv_mv_payload(self):
        return {"type": "valve", "valve": "mitral", "c": [0, 0, 0], "n": [0, 0, 1],
                "r": 12.0, "mvaov": {"x": 1}}

    def _lvv_build_dict(self, warn=False):
        return {"type": "lvvol", "endo": {"m": 1}, "series": {"series_uid": "1"},
                "mitral": {"c": [0, 0, 0], "n": [0, 0, 1], "r": 12.0,
                           "mvaov": {"x": 1}}}

    def _full_lv_epi_dict(self):
        return {"kind": "mdv-lvef", "region": {"r": 1}, "epi_tent": True}

    def _lv_series_dir(self):
        return ""

    def _lv_default_stem(self):
        return "case"

    def _lv_series_meta(self):
        return {}


class Pane:
    def __init__(self, v):
        self.v = v

    def current_viewer(self):
        return self.v


class Host:
    lv_save_active_full = MainWindow.lv_save_active_full

    def __init__(self, v):
        self._active = Pane(v)


with tempfile.TemporaryDirectory() as td:
    out = os.path.join(td, "case.FullLv.json")
    with mock.patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName",
                    return_value=(out, "")), \
            mock.patch("PyQt6.QtWidgets.QMessageBox.information") as info, \
            mock.patch("PyQt6.QtWidgets.QMessageBox.warning") as warn:
        res = Host(FakeV()).lv_save_active_full()
    names = sorted(os.listdir(td))
    check("LV Analysis save works with MV-AoV + Apex (no AoV)",
          bool(res) and not info.called and not warn.called, str(names))
    check("writes .MVAoVLv.json, no .AoVLv.json / .MVLv.json",
          "case.MVAoVLv.json" in names and not any(
              n.endswith((".AoVLv.json", ".MVLv.json")) for n in names), str(names))
    mv = json.load(open(os.path.join(td, "case.MVAoVLv.json"), encoding="utf-8"))
    check(".MVAoVLv.json holds the MV-AoV point", mv.get("mvaov") == {"x": 1})
with mock.patch("PyQt6.QtWidgets.QMessageBox.information") as info:
    res = Host(FakeV(mvaov=False)).lv_save_active_full()
check("old MV-plane analysis without AoV is still refused",
      res is False and info.called)

# ---- 4. FullLv → editable files -------------------------------------------------
bld = {"mitral": {"c": [0, 0, 0], "n": [0, 0, 1], "mvaov": {"x": 1}},
       "aortic": {"c": [0, 0, 0], "n": [0, 0, 1]}, "apex": [0, 0, -60]}
fl = MainWindow._full_lv_to_files({"bld": bld, "epi": {"kind": "mdv-lvef"}})
check("FullLv with the MV-AoV point: no fake AoV plane",
      "valve_aortic" not in fl and fl["valve_mitral"]["mvaov"] == {"x": 1})
old = dict(bld, mitral={"c": [0, 0, 0], "n": [0, 0, 1]})
check("old FullLv (MV plane) still gives the AoV plane",
      "valve_aortic" in MainWindow._full_lv_to_files({"bld": old}))
check("VTK FullLv review records the MV-AoV point, skips the stand-in",
      "self._lv_mvaov = dict(_mvaov)" in seg(VTK, "apply_full_lv"))


# ---- 5. old Epi prompt ------------------------------------------------------
class _Sig:
    def __init__(self):
        self.cbs = []

    def connect(self, cb):
        self.cbs.append(cb)


class _Btn:
    def __init__(self, label):
        self.label = label
        self.clicked = _Sig()


class FakeBox:
    pick = None                       # label of the button to press
    Icon = QMessageBox.Icon
    ButtonRole = QMessageBox.ButtonRole
    StandardButton = QMessageBox.StandardButton
    information = mock.MagicMock()
    warning = mock.MagicMock()

    def __init__(self, *a):
        self.btns = []

    def setWindowTitle(self, *a):
        pass

    setIcon = setText = setInformativeText = setWindowTitle

    def addButton(self, label, *a):
        b = _Btn(label)
        self.btns.append(b)
        return b

    def exec(self):
        for b in self.btns:
            if b.label == FakeBox.pick:
                for cb in b.clicked.cbs:
                    cb()


class V(ValveAutoMixin):
    def __init__(self, ready=True):
        self._lv_valves = {"mitral": (np.array([0, 0, 38.0]),
                                      np.array([0, 0, 1.0]), 10.0)}
        self._lv_mvaov = {"x": 1} if ready else None
        self._lv_apex = np.array([0, 0, 0.0])

    def window(self):
        return None

    def _lv_long_axis_from_apex(self, apex):
        return np.array([0, 0, 1.0]), np.array([1.0, 0, 0])


old_epi = {"kind": "mdv-lvef", "epi_planes": {"0": [[0, 0, 0]]}}
new_epi = {"kind": "mdv-lvef", "epi_planes": {"0": [[0, 0, 0]]}, "epi_tent": True}
with mock.patch("PyQt6.QtWidgets.QMessageBox", FakeBox), \
        mock.patch("multi_dicomviewer.core.lv_epi_convert.convert",
                   return_value=({"converted": True}, "")) as conv:
    check("a current-method Epi passes through (no question)",
          V()._lv_epi_maybe_convert(new_epi) == (new_epi, False))
    FakeBox.pick = t("そのまま読み込む")
    check("そのまま読み込む → the file as is",
          V()._lv_epi_maybe_convert(old_epi) == (old_epi, False))
    FakeBox.pick = None
    check("キャンセル → nothing applied", V()._lv_epi_maybe_convert(old_epi) is None)
    FakeBox.pick = t("変換")
    r = V()._lv_epi_maybe_convert(old_epi)
    check("変換 → converted dict", r == ({"converted": True}, True), str(r))
    a = conv.call_args[0]
    check("converted on apex → MV-AoV point", np.allclose(a[1], [0, 0, 0])
          and np.allclose(a[4], [0, 0, 38.0]))
    FakeBox.information.reset_mock()
    check("変換 without the MV-AoV point → refused with a hint",
          V(ready=False)._lv_epi_maybe_convert(old_epi) is None
          and FakeBox.information.called)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
