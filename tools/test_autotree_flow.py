# -*- coding: utf-8 -*-
"""全冠動脈描出 UI flow (user 2026-11-28):

  - the ostium window: LM → RCA (or LMなし: LAD → LCX → RCA) set at the
    crosshair one after another; スキップ / 戻る; 抽出開始 only when done
  - the extracted lines → the tree: LM / LAD / LCX / RCA roots, branches on
    their parents, 未接続 left loose, one NN.cpr.json each (auto_traced) in
    the AutoTree folder; 置き換える replaces the old tree (its files kept)
  - the 全冠動脈描出 button next to 一括描出 (both viewers); the CPR seek
    buttons read ◀(A) / ▶(F) (|◀ / ▶| unchanged)

    python tools/test_autotree_flow.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.information = staticmethod(lambda *a, **k: None)
from multi_dicomviewer.core.coronary_autotree import AutoVessel  # noqa: E402
from multi_dicomviewer.ui.autotree_ostia import OstiaDialog  # noqa: E402
from multi_dicomviewer.ui.coronary_tree_window import CoronaryTreeWindow  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


# ---- the ostium window ------------------------------------------------------
cross = {"p": np.array([10.0, 20.0, 30.0])}
ran = []
dlg = OstiaDialog(None, lambda: cross["p"], lambda ost: ran.append(ost))
check("LMあり: steps LM → RCA", dlg.roles() == ["LM", "RCA"])
check("抽出開始 disabled at first", not dlg._run.isEnabled())
dlg._ok.click()
cross["p"] = np.array([40.0, 20.0, 30.0])
check("after LM: asks for the RCA", "RCA" in dlg._ask.text(), dlg._ask.text())
dlg._ok.click()
check("both set → 抽出開始 enabled", dlg._run.isEnabled())
dlg._back.click()
check("戻る → RCA unset again", not dlg._run.isEnabled()
      and "RCA" in dlg._ask.text())
dlg._skip.click()
check("RCA skipped → LM only, still runnable",
      dlg._run.isEnabled() and [r for r, _p in dlg.ostia()] == ["LM"])
dlg._nolm_rb.setChecked(True)
check("LMなし: steps LAD → LCX → RCA", dlg.roles() == ["LAD", "LCX", "RCA"])
for p in ([1.0, 2, 3], [4.0, 5, 6], [7.0, 8, 9]):
    cross["p"] = np.array(p)
    dlg._ok.click()
dlg._run.click()
check("抽出開始 → the ostia in order (crosshair positions)",
      len(ran) == 1 and [r for r, _p in ran[0]] == ["LAD", "LCX", "RCA"]
      and np.allclose(ran[0][1][1], [4, 5, 6]), ran)

# ---- install into the tree -------------------------------------------------
D = tempfile.mkdtemp(dir=HOME)
w = CoronaryTreeWindow(None)
w._use_uid("at-ct")
old_p = os.path.join(D, "old.cpr.json")
with open(old_p, "w", encoding="utf-8") as f:
    f.write("{}")
w._tree.add_vessel("old", "OLD", "branch", np.c_[np.arange(5.0), np.zeros(5),
                                                np.zeros(5)])
w._bundle()["paths"]["old"] = old_p


def line(a, b, n=20):
    return np.linspace(a, b, n)


lm = line([0, 0, 0], [0, 10, 0])
lad = line([0, 10, 0], [0, 60, -20])
lcx = line([0, 10, 0], [30, 20, 0])
d1 = line(lad[8], lad[8] + [15, 5, -5])
rca = line([40, 0, 0], [60, -40, -10])
unc = line([20, 40, -10], [25, 60, -20])
V = [AutoVessel(0, "LM", None, lm, lm[::4], system="LCA"),
     AutoVessel(1, "LAD", None, lad, lad[::3], system="LCA"),
     AutoVessel(2, "LCX", None, lcx, lcx[::3], system="LCA"),
     AutoVessel(3, "branch", 1, d1, d1[::3], system="LCA"),
     AutoVessel(4, "RCA", None, rca, rca[::3], system="RCA"),
     AutoVessel(5, "branch", None, unc, unc[::3], connected=False)]
F = os.path.join(D, "AutoTree_x")
out = w.install_auto_tree(V, F, series={"series_uid": "at-ct"})
tr = w._tree
roles = sorted(v.role for v in tr.vessels.values())
check("replaced: the old line is gone, its file kept",
      "old" not in tr.vessels and os.path.isfile(old_p))
check("6 lines: 4 roots, 1 branch, 1 未接続",
      out["lines"] == 6 and out["roots"] == 4 and out["branches"] == 1
      and out["unconnected"] == 1, out)
check("roles LM / LAD / LCX / RCA + 2 branches",
      roles == sorted(["LM", "LAD", "LCX", "RCA", "branch", "branch"]), roles)
br = [v for v in tr.vessels.values() if v.role == "branch" and v.parent]
check("the diagonal hangs on the LAD", len(br) == 1
      and tr.vessels[br[0].parent].role == "LAD"
      and br[0].junction is not None, [(b.parent, b.junction) for b in br])
check("the 未接続 line is unconnected", len(tr.unconnected()) == 1)
files = sorted(os.listdir(F))
check("one NN.cpr.json each in the AutoTree folder",
      files == [f"{i:02d}.cpr.json" for i in range(1, 7)], files)
with open(os.path.join(F, files[0]), encoding="utf-8") as f:
    rec = json.load(f)
check("… MDV-CPR, auto_traced, the CT's series",
      rec.get("format") == "MDV-CPR" and rec.get("auto_traced")
      and rec.get("series", {}).get("series_uid") == "at-ct")
check("all lines marked auto_traced",
      all(v.auto_traced for v in tr.vessels.values()))
check("the summary names the counts", "6" in w._hint.text(), w._hint.text())
out2 = w.install_auto_tree(V[:2], os.path.join(D, "AutoTree_y"), replace=False)
check("追加する: added to the tree", len(tr.vessels) == 8, len(tr.vessels))

# ---- buttons -----------------------------------------------------------------
for fn in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    s = (ROOT / "multi_dicomviewer/viewers" / fn).read_text(encoding="utf-8")
    check(f"{fn}: 全冠動脈描出 right of 一括描出",
          0 < s.index('FitButton(t("全冠動脈描出"))')
          - s.index('FitButton(t("一括描出"))') < 1500)
    check(f"{fn}: ◀(A) / ▶(F), |◀ / ▶| unchanged",
          'FitButton("◀(A)")' in s and 'FitButton("▶(F)")' in s
          and 'FitButton("|◀")' in s and 'FitButton("▶|")' in s)
mw = (ROOT / "multi_dicomviewer/ui/main_window.py").read_text(encoding="utf-8")
check("shell: the flow + the Epi source",
      "def coronary_auto_tree(self, viewer)" in mw
      and "def coronary_auto_tree_run(self, viewer, ostia)" in mw
      and "def _autotree_epi(self, viewer)" in mw)

# ---- end to end through the shell (stub viewer, the synthetic heart) -------
import base64  # noqa: E402
import zlib  # noqa: E402

from PyQt6.QtWidgets import QWidget  # noqa: E402

import multi_dicomviewer.ui.autotrack_progress as atp  # noqa: E402
from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402
from tools.autotree_phantom import PHANTOM as PH  # noqa: E402

epi = PH["epi"]
zs, ys, xs = np.nonzero(epi)
bb = (int(zs.min()), int(zs.max()) + 1, int(ys.min()), int(ys.max()) + 1,
      int(xs.min()), int(xs.max()) + 1)
sub = epi[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
region = {"shape": list(sub.shape), "bbox": list(bb),
          "packed": base64.b64encode(zlib.compress(
              np.packbits(sub.ravel()).tobytes())).decode()}


class Viewer(QWidget):
    _vol = PH["vol"]
    _dims = (PH["S"],) * 3
    _center = np.zeros(3)
    _lv_apex = PH["apex"]

    def _lv_series_meta(self):
        return {"series_uid": "e2e-ct"}

    def _lv_series_dir(self):
        return ""

    def _full_lv_epi_dict(self):
        return {"region": region}


class Prog:
    def __init__(self, *a, **k):
        self.stages = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cancelled(self):
        return False

    def update(self, stage, *a, name=""):
        self.stages.append((stage, name))


results, progs = [], []
atp.show_result = lambda parent, text, ok=True, **k: results.append((ok, text))
atp.AutoTrackProgress = lambda *a, **k: (progs.append(Prog()), progs[-1])[1]
E2E = tempfile.mkdtemp(dir=HOME)
w2 = CoronaryTreeWindow(None)


class Shell(QWidget):
    coronary_auto_tree_run = MainWindow.coronary_auto_tree_run
    _autotree_epi = MainWindow._autotree_epi
    _corotree_win = w2

    def _open_coronary_tree(self):
        return w2

    def _pane_of_viewer(self, v):
        return None

    def _viewer_next_dir(self, v):
        return E2E

    def coronary_overlay_refresh(self):
        pass

    def refresh_ct_mode_bars(self):
        pass


sh = Shell()
vw = Viewer()
ep = sh._autotree_epi(vw)
check("the Epi comes from the viewer's in-session Epi (mask + origin)",
      ep is not None and ep[0].shape == sub.shape
      and np.allclose(ep[1], (bb[4] * PH["S"], bb[2] * PH["S"], bb[0] * PH["S"])))
o2 = sh.coronary_auto_tree_run(vw, [("LM", PH["LM0"]), ("RCA", PH["RC0"])])
t2 = w2._tree
r2 = sorted(v.role for v in t2.vessels.values() if v.role != "branch")
check("end to end: LM / LAD / LCX / RCA registered", r2 == ["LAD", "LCX", "LM", "RCA"],
      (r2, results))
check("end to end: one 未接続 line (the occluded segment)",
      len(t2.unconnected()) == 1, len(t2.unconnected()))
sub_d = [d for d in os.listdir(E2E) if d.startswith("AutoTree_")]
check("files in a new AutoTree_<date> folder",
      len(sub_d) == 1 and len(os.listdir(os.path.join(E2E, sub_d[0])))
      == len(t2.vessels), sub_d)
check("the stages were shown, the result reported",
      progs and any(s == "autotree" for s, _n in progs[0].stages)
      and results and results[-1][0], results[-1:] if results else None)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(0 if not fails else 1)
