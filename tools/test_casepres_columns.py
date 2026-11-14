"""Case Presentation: a column hidden with the header ✕ can always be brought
back — 列表示 shows how many are hidden (outlined), its menu has すべて表示,
and the header tooltip points there.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QMainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer import i18n  # noqa: E402
from multi_dicomviewer.ui import case_presentation_window as cpw  # noqa: E402

i18n.set_language("ja")
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


shell = QMainWindow()
w = cpw.CasePresentationWindow(shell)
b = w._cols_btn
check("nothing hidden: plain 列表示, すべて表示 disabled",
      b.text() == "列表示" and b.styleSheet() == ""
      and not w._col_show_all.isEnabled())
w._hide_col(1)
w._hide_col(3)
check("header ✕ ×2 → 列表示 (2列非表示), outlined",
      b.text() == "列表示 (2列非表示)" and "#e0a000" in b.styleSheet()
      and w._col_show_all.isEnabled(), b.text())
check("menu entries unchecked for the hidden columns",
      not w._col_actions[1].isChecked() and not w._col_actions[3].isChecked())
acts = b.menu().actions()
check("すべて表示 is the first menu item", acts[0] is w._col_show_all
      and acts[0].text() == "すべて表示")
w._col_actions[3].setChecked(True)
check("re-checking one in the menu → 1 hidden", b.text() == "列表示 (1列非表示)"
      and not w._table.isColumnHidden(3))
w._col_show_all.trigger()
check("すべて表示 → every column back, button plain again",
      not w._hidden_cols() and b.text() == "列表示" and b.styleSheet() == ""
      and all(a.isChecked() for a in w._col_actions.values()))
check("header tooltip tells where to show a column again",
      "列表示" in w._table.horizontalHeader().toolTip())

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
