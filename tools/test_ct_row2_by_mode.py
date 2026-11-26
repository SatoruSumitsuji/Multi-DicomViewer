"""Bottom-bar row by mode (2026-11-17): Simple View = the 道しるべ (entry
steps of every mode), LV / CoronaryTree = only that mode's tools; row order
LV → CoronaryTree → Territory (the row-1 mode order) with ONE 「LV:」 caption.

  Simple View : LV MV-AoV+Apex / Epi / Blood/Endo / FullLV,
                CoronaryTree Load / CPR
  LVモード     : LV steps + 壁厚3D / 壁厚短軸 / CoronaryTree (no Coronary group)
  CoronaryTree : Load / CPR / Open VR / 回転連動 (no LV)
  Territory Calc / Load / Save: row 3 (読込ファイル), every mode (2026-11-27):
  FullLv ＋ corotree ▶ [Territory Calc] territory [Load] [Save]

    python tools/test_ct_row2_by_mode.py
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

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


vol = np.full((60, 64, 64), -100.0, np.float32)
s = LoadedSeries(modality=Modality.CT, volume=vol, spacing_mm=(0.7, 0.7),
                 cine_fps=None, window=800.0, level=200.0, slice_mm=1.5,
                 patient_basis=None, series_uid="row2-ct")
v = CTViewer()
v.resize(1800, 900)
v.show()
v.load_series(s, "row2")
v._set_mode("3D")
app.processEvents()


def shown(w):
    return w is not None and not w.isHidden()


def state():
    app.processEvents()
    return {
        "lv": all(shown(w) for w in (v._lv_cap, v._lv_mv_btn, v._lv_epi_btn,
                                     v._lvv_start_btn, v._full_lv_btn)),
        "lv_view": all(shown(w) for w in (v._ctlv_thick3d_btn,
                                          v._ctlv_thicksax_btn,
                                          v._ctlv_tree_btn)),
        "coro": all(shown(w) for w in (v._coro_cap, v._coro_load_btn,
                                       v._coronary_mpr_btn)),
        "vr": any(shown(w) for w in (v._cpr_vr_group or [])),
        "terr": shown(v._ct_terr_btn) and shown(v._ct_files_wrap),
        "terr_load": shown(v._ct_terr_load_btn),
        "terr_save": shown(v._ct_terr_save_btn),
        "lv_caps": sum(shown(w) for w in (v._lv_cap, v._ctlv_cap)),
    }


# ---- row order -----------------------------------------------------------
row1 = v._lv_wrap.layout().itemAt(0).layout()
pos = {n: row1.indexOf(w) for n, w in (
    ("lv_cap", v._lv_cap), ("mv", v._lv_mv_btn), ("full", v._full_lv_btn),
    ("thick3d", v._ctlv_thick3d_btn), ("tree", v._ctlv_tree_btn),
    ("coro_cap", v._coro_cap), ("cpr", v._coronary_mpr_btn))}
check("order: LV steps → LV view tools → CoronaryTree",
      0 <= pos["lv_cap"] < pos["mv"] < pos["full"] < pos["thick3d"]
      < pos["tree"] < pos["coro_cap"] < pos["cpr"], pos)
check("Territory no longer in row 1", row1.indexOf(v._ct_terr_btn) < 0
      and row1.indexOf(v._ct_terr_load_btn) < 0)
r3 = v._ct_files_wrap.layout()
p3 = {n: r3.indexOf(w) for n, w in (
    ("full", v._ct_file_lbls["full"]), ("tree", v._ct_file_lbls["tree"]),
    ("calc", v._ct_terr_btn), ("terr", v._ct_file_lbls["terr"]),
    ("terr_x", v._ct_unload_btns["terr"]), ("load", v._ct_terr_load_btn),
    ("save", v._ct_terr_save_btn))}
check("row 3: FullLv ＋ corotree ▶ [Territory Calc] territory ⊗ [Load] [Save]",
      0 <= p3["full"] < p3["tree"] < p3["calc"] < p3["terr"] < p3["terr_x"]
      < p3["load"] < p3["save"], p3)
seps = [r3.itemAt(i).widget().text() for i in range(r3.count())
        if r3.itemAt(i).widget() is not None
        and r3.itemAt(i).widget().text() in ("&nbsp;＋&nbsp;", "&nbsp;▶&nbsp;")]
check("row 3 separators: ＋ then ▶", seps == ["&nbsp;＋&nbsp;", "&nbsp;▶&nbsp;"],
      seps)

# ---- Simple View ----------------------------------------------------------
v._ct_switch_inplace("standard")
st = state()
check("Simple View: LV steps + CoronaryTree Load/CPR + Territory (row 3)",
      st["lv"] and st["coro"] and st["terr"] and st["terr_load"]
      and st["terr_save"], st)
check("Simple View: no LV view tools / Open VR",
      not st["lv_view"] and not st["vr"], st)

# ---- LVモード --------------------------------------------------------------
v._ct_switch_inplace("lv")
st = state()
check("LVモード: LV steps + view tools, ONE 「LV:」 caption",
      st["lv"] and st["lv_view"] and st["lv_caps"] == 1, st)
check("LVモード: no CoronaryTree group; Territory (row 3) still there",
      not st["coro"] and not st["vr"] and st["terr"] and st["terr_load"], st)

# ---- CoronaryTreeモード ----------------------------------------------------
v._ct_switch_inplace("coronary")
st = state()
check("CoronaryTreeモード: Load/CPR/Open VR + Territory (row 3)",
      st["coro"] and st["vr"] and st["terr"] and st["terr_load"], st)
check("CoronaryTreeモード: no LV", not st["lv"] and not st["lv_view"], st)

# ---- back to Simple View --------------------------------------------------
v._ct_switch_inplace("standard")
st = state()
check("back to Simple View: the 道しるべ again",
      st["lv"] and st["coro"] and st["terr_load"] and not st["vr"], st)

# ---- LVモード: Epi境界 / Endo境界 / 血流 show toggles ----------------------
sb = v._ctlv_show_btns
check("toggles placed after FullLV, before 壁厚3D",
      pos["full"] < row1.indexOf(sb["epi"]) < row1.indexOf(sb["endo"])
      < row1.indexOf(sb["blood"]) < row1.indexOf(v._ctlv_thick3d_btn),
      [row1.indexOf(sb[k]) for k in ("epi", "endo", "blood")])
v._ct_switch_inplace("lv")
app.processEvents()
check("LVモード: the three toggles shown (no LV data → greyed)",
      all(shown(sb[k]) for k in sb)
      and not any(sb[k].isEnabled() for k in sb))
v._ct_switch_inplace("standard")
app.processEvents()
check("Simple View: toggles hidden", not any(shown(sb[k]) for k in sb))
calls = []
v._lv_set_epi_blood_border = lambda on: calls.append(("epi", on))
v.set_lv_border_visible = lambda on: calls.append(("border", on))
v._lv_set_auto_endo = lambda on: calls.append(("endo", on))
v._lv_set_blood = lambda m: calls.append(("blood", m))
v._lvv = {"observe": True}
v._lvv_epi_surf = object()
v._lvv_blood_comp = object()
v._ct_switch_inplace("lv")
for k in ("epi", "endo", "blood"):
    sb[k].setChecked(True)
    v._ctlv_show_toggle(k)
    sb[k].setChecked(False)
    v._ctlv_show_toggle(k)
check("toggles drive the existing show/hide (blood = LV-Blood / off)",
      ("epi", True) in calls and ("epi", False) in calls
      and ("border", False) in calls
      and ("endo", True) in calls and ("endo", False) in calls
      and ("blood", "lv") in calls and ("blood", None) in calls, calls)
v._lvv = None
v._lvv_epi_surf = None
v._lvv_blood_comp = None

# ---- Calc enable rule + Mac parity ----------------------------------------
src = (ROOT / "multi_dicomviewer/viewers/ct_mode_bar.py").read_text(encoding="utf-8")
check("Calc enabled in every mode once a tree + an LV exist",
      "tb.setEnabled(has_tree and has_lv)" in src
      and 'cur in ("coronary", "standard")' not in src)
mac = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(encoding="utf-8")
check("Mac: same mode rule + row order",
      "ctx = self._ct_bar_ctx()" in mac and "self._ct_bar_reorder(row1)" in mac)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
