# -*- coding: utf-8 -*-
"""Territory is computed ONLY on Calc (user 2026-11-25): an edit in the
CoronaryTree mode does not re-run the slow compute any more — the result
goes blank (心筋量 column 「—」, 「未計算（Calc で計算）」) until Calc is
pressed again. Synthetic tree, the engine faked (counts the computes),
fake HOME, no real dialog.

    python tools/test_territory_calc_only.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(
    lambda *a, **k: QMessageBox.StandardButton.Yes)
from multi_dicomviewer.core.centerline import CenterLine  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.ui import coronary_tree_window as ctw  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


calls = []


class _Eng:
    def territory(self, vid, idx):
        return None, 12.5


class _LVF:
    def myocardial_volume_ml(self):
        return 100.0


def _report(tree, lvf):
    calls.append(1)
    return ["report"], _Eng()


ctw.format_territory_report = _report
ctw.LVFunction.from_full = staticmethod(lambda d: _LVF())

w = ctw.CoronaryTreeWindow(None)
w._run_busy = lambda _msg, fn: fn()
w._use_uid("1.2.3")
tr = w._bundle()["tree"]
pts = np.c_[np.linspace(0, 40, 6), np.zeros(6), np.zeros(6)]
tr.add_root("lad", "LAD", "LAD", CenterLine.from_points(pts, step_mm=0.5).points)
tr.vessels["lad"].ctrl = pts
w._full_lv_data = {"fake": True}

# a FullLv / tree change before Calc: nothing computed
w._compute_territory()
check("no compute before Calc", not calls and w._territory is None)
check("心筋量 shows 「—」 when not computed", w._myo_text("lad") == "—",
      w._myo_text("lad"))

# Calc (main_window arms the one-shot flag, then the FullLv load computes)
w._bundle()["terr_calc"] = True
w._compute_territory()
check("Calc computes once", len(calls) == 1 and w._territory is not None)
check("心筋量 filled after Calc", "%" in w._myo_text("lad"), w._myo_text("lad"))
w._targets = [{"vid": "lad", "idx": 3, "ml": 12.5, "pct": 12.5}]
w._targets_shown = True
w._refresh_targets_table()
check("Target % / mL filled after Calc",
      w._targets_w.topLevelItem(0).text(4) == "12.5%"
      and "12.5%" in w._target_summary_text())

# a NAME change keeps the result (no blank, no recompute)
w.rename_vessel("lad", "LAD2")
check("a rename keeps the Territory result",
      len(calls) == 1 and w._territory is not None
      and "%" in w._myo_text("lad"), (len(calls), w._myo_text("lad")))
w._batch_rename(ask=False)               # FixName / file rename path
check("a file rename keeps it too", w._territory is not None)

# edits: no recompute, the result goes blank
w.treeChanged.emit()
check("an edit does NOT recompute", len(calls) == 1, len(calls))
check("an edit blanks the result",
      w._territory is None and w._myo_text("lad") == "—"
      and w._terr_lbl.text() == t("Territory: 未計算（Calc で計算）")
      and w._terr_out.toPlainText() == "")
check("every %/mL readout shows 「—」 (targets table, VR summaries)",
      w._targets_w.topLevelItem(0).text(4) == "—"
      and w._targets_w.topLevelItem(0).text(5) == "—"
      and w._target_summary_text() == "Target-1 —"
      and w._territory_summary_text() == "Myocardium —\nLAD —",
      (w._target_summary_text(), w._territory_summary_text()))
w.treeChanged.emit()
check("further edits: still no compute", len(calls) == 1, len(calls))

# Calc again computes again
w._bundle()["terr_calc"] = True
w._compute_territory()
check("Calc again computes", len(calls) == 2 and w._territory is not None)
w._compute_territory(calc=True)          # a loaded .territory.json
check(".territory.json load computes", len(calls) == 3)
check("the one-shot flag is consumed", not w._bundle().get("terr_calc"))
src = (ROOT / "multi_dicomviewer" / "ui" / "main_window.py").read_text(
    encoding="utf-8")
check("shell never leaves the flag armed",
      'w._bundle().pop("terr_calc", None)' in src)

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
