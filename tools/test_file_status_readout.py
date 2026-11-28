"""Loaded files readout status (2026-11-17): each named file shows
「（変更あり・未保存）」 once its content changed after it was loaded / saved,
and goes back to the bare name after a save.

  corotree : the tree's saved content vs the loaded / saved one (a vessel
             hidden is display-only → not a change)
  FullLv   : a fingerprint of the three LV steps (MV-AoV + Apex, Epi,
             Blood/Endo) vs the loaded / saved one
  territory: Targets / opacity changed, or its corotree / FullLv changed

    python tools/test_file_status_readout.py
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
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.information = staticmethod(lambda *a, **k: None)
from multi_dicomviewer.core.coronary_territory import CoronaryTree  # noqa: E402
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.ui.coronary_tree_window import CoronaryTreeWindow  # noqa: E402
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


# ---- dock: corotree changed since load / save -----------------------------
D = tempfile.mkdtemp(dir=HOME)
coro_p = os.path.join(D, "case.corotree.json")
tree = CoronaryTree()
tree.add_root("lad", "LAD", "LAD",
              np.c_[np.linspace(0, 29.3, 30), np.zeros(30), np.zeros(30)])
with open(coro_p, "w", encoding="utf-8") as f:
    json.dump(tree.to_json(series={"series_uid": "st-ct"}), f)
w = CoronaryTreeWindow(None)
check("dock loads the corotree", w.load_file(coro_p))
uid = "st-ct"
check("just loaded → unchanged", not w.corotree_dirty_for_series(uid))
w._hidden.add("lad")
w.treeChanged.emit()
check("a vessel hidden (display only) → still unchanged",
      not w.corotree_dirty_for_series(uid))
w._hidden.discard("lad")
w._tree.vessels["lad"].name = "LAD-edited"
w.treeChanged.emit()
check("vessel edited → 変更あり", w.corotree_dirty_for_series(uid))
w._write_to(coro_p)
check("saved → unchanged again", not w.corotree_dirty_for_series(uid))

# Territory settings vs the loaded / saved .territory.json
b = w._bundle()
b["terr_path"] = os.path.join(D, "case.territory.json")
b["terr_base"] = w._terr_sig()
check("territory settings as saved → unchanged",
      not w.territory_changed_for_series(uid))
w._targets = [{"vid": "lad", "idx": 5, "hidden": False}]
check("a Target added → territory 変更あり",
      w.territory_changed_for_series(uid))

# ---- viewer: FullLv fingerprint + readout --------------------------------
vol = np.full((40, 48, 48), -100.0, np.float32)
s = LoadedSeries(modality=Modality.CT, volume=vol, spacing_mm=(0.7, 0.7),
                 cine_fps=None, window=800.0, level=200.0, slice_mm=1.5,
                 patient_basis=None, series_uid="st-ct")
v = CTViewer()
v.resize(1400, 800)
v.show()
v.load_series(s, "st")
app.processEvents()
check("fresh CT: no FullLv baseline → not modified", not v._full_lv_modified())
v._lv_apex = np.array([10.0, 20.0, 30.0])
v._full_lv_mark_clean()
check("baseline taken → unchanged", not v._full_lv_modified())
v._lv_apex = np.array([10.0, 21.0, 30.0])
check("Apex moved (step 1) → 変更あり", v._full_lv_modified())
v._full_lv_mark_clean()
v._lvv_epi_surf = object()
check("Epi re-traced (step 2) → 変更あり", v._full_lv_modified())
v._full_lv_mark_clean()
v._lvv_blood_comp = np.zeros((2, 2, 2), bool)
check("Blood re-calculated (step 3) → 変更あり", v._full_lv_modified())
v._full_lv_mark_clean()
v._lvv_mask_on = not getattr(v, "_lvv_mask_on", False)
check("blood shown / hidden (display only) → unchanged",
      not v._full_lv_modified())


class Shell:
    state = {}

    def ct_mode_state(self, viewer):
        return dict(self.state)


sh = Shell()
v._ct_mode_shell = lambda: sh
mark = t("変更あり・未保存")


def line(state):
    sh.state = state
    v.refresh_ct_mode_bar()
    return v._ct_files_text()


base = {"full": "a.FullLv.json", "tree": "a.corotree.json", "terr": True,
        "terr_file": "a.territory.json"}
txt = line(base)
check("nothing changed → bare names, no mark", mark not in txt, txt)
txt = line(dict(base, tree_dirty=True))
check("corotree edited → corotree + territory marked, FullLv not",
      txt.count(mark) == 2 and txt.index("corotree") < txt.index(mark)
      and txt.index("FullLv: a.FullLv.json") < txt.index("corotree"), txt)
v._lv_apex = np.array([1.0, 2.0, 3.0])
txt = line(base)
check("FullLv changed → FullLv + territory marked",
      txt.count(mark) == 2 and txt.index(mark) < txt.index("corotree"), txt)
v._full_lv_mark_clean()
txt = line(dict(base, terr_changed=True))
check("Targets changed → only territory marked", txt.count(mark) == 1
      and txt.index(mark) > txt.index("territory"), txt)
txt = line({"full": "", "tree": "", "tree_mem": True})
check("unsaved tree (no file) → no mark (already 未保存)", mark not in txt, txt)

# ---- 保存済み / 未保存 button right of each file (user 2026-11-28) ----------
sbt = v._ct_file_save_btns
U, S = t("未保存"), t("保存済み")


def states():
    return {k: (sbt[k].text() if not sbt[k].isHidden() else None)
            for k in ("full", "tree", "terr")}


v._full_lv_mark_clean()
line(base)
check("all three saved → three 保存済み", states() == {
    "full": S, "tree": S, "terr": S}, states())
line(dict(base, tree_dirty=True))
check("corotree edited → corotree + territory 未保存", states() == {
    "full": S, "tree": U, "terr": U}, states())
v._full_lv_saved = False
line({"full": "", "tree": "", "tree_mem": True})
check("LV measured (no file) + tree registered (no file) → both 未保存, "
      "no territory button", states() == {"full": U, "tree": U, "terr": None},
      states())
asked, saved_kinds = [], []
sh.ct_file_save = lambda viewer, kind: saved_kinds.append(kind)
QMessageBox.question = staticmethod(
    lambda *a, **k: (asked.append(a[2]), QMessageBox.StandardButton.No)[1])
sbt["tree"].click()
check("未保存 click → 「保存しますか？」; No → nothing saved",
      len(asked) == 1 and "corotree" in asked[0] and not saved_kinds, asked)
QMessageBox.question = staticmethod(
    lambda *a, **k: QMessageBox.StandardButton.Yes)
sbt["tree"].click()
check("… Yes → that file's save (the Save dialog)", saved_kinds == ["tree"],
      saved_kinds)
line(base)
asked.clear()
sbt["tree"].click()
check("保存済み click → nothing", saved_kinds == ["tree"] and not asked)

# .bak: the previous version is kept on every overwrite
from multi_dicomviewer.core.file_backup import backup_existing  # noqa: E402
with open(coro_p, encoding="utf-8") as f:
    before = f.read()
w._tree.vessels["lad"].name = "LAD-again"
w._write_to(coro_p)
bak = coro_p + ".bak"
with open(bak, encoding="utf-8") as f:
    kept = f.read()
check("corotree overwrite → the previous version in .corotree.json.bak",
      kept == before and "LAD-again" not in kept, bak)
check("no file yet → no .bak", backup_existing(os.path.join(D, "x.json")) == "")
src_all = "".join((ROOT / p).read_text(encoding="utf-8") for p in (
    "multi_dicomviewer/viewers/ct_viewer.py",
    "multi_dicomviewer/viewers/ct_viewer_pygfx.py",
    "multi_dicomviewer/ui/coronary_tree_window.py"))
check("FullLv (both viewers) / corotree / territory saves keep a .bak",
      src_all.count("backup_existing(path)") == 4)

# FullLV button: green only while saved AND unchanged
v._full_lv_saved = True
v._full_lv_mark_clean()
v._full_lv_refresh_btn()
green = "#2e8b57" in v._full_lv_btn.styleSheet()
v._lv_apex = np.array([9.0, 9.0, 9.0])
v._full_lv_refresh_btn()
check("FullLV button: green when saved, plain once the LV changed",
      green and "#2e8b57" not in v._full_lv_btn.styleSheet())

mac = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(encoding="utf-8")
check("Mac: save marks clean + button rule",
      "self._full_lv_mark_clean()   # 変更あり baseline" in mac
      and "and not self._full_lv_modified())" in mac)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
