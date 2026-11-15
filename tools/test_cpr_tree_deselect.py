"""CPR line right-click ▸ 選択を解除 (Deselect): a click on a Coronary Tree line
selects it (orange); the short menu (編集 / 名前を表示) now also offers
選択を解除 under them, which clears the tree selection so the orange highlight
goes. Shared CPRMixin menu → both viewers. Fake menu (nothing pops up).
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_FAKE_HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = _FAKE_HOME

import PyQt6.QtWidgets as QtW  # noqa: E402
from PyQt6.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer import i18n  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.viewers.cpr_mixin import CPRMixin  # noqa: E402
from multi_dicomviewer.ui import coronary_tree_window as ctw  # noqa: E402

i18n.set_language("ja")
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


class FakeAct:
    def __init__(self, text):
        self.text, self.enabled = text, True

    def setEnabled(self, b):
        self.enabled = bool(b)


class FakeMenu:
    pick, seen = None, []

    def __init__(self, *_a):
        self.items = []

    def addAction(self, text):
        a = FakeAct(text)
        self.items.append(a)
        return a

    def addSeparator(self):
        pass

    def exec(self, *_a):
        FakeMenu.seen.append(self.items)
        for a in self.items:
            if FakeMenu.pick and a.text == FakeMenu.pick:
                return a
        return None


class _Canvas:
    def mapToGlobal(self, p):
        return p


class _Pane:
    canvas = _Canvas()


class Host(CPRMixin):
    def __init__(self):
        self._coro_names = set()
        self.pane = {"A": _Pane(), "B": _Pane()}
        self.sel = []
        self._coro_select_cb = self.sel.append
        self._coro_overlay = [{"vid": "v7", "selected": True},
                              {"vid": "v8", "selected": False}]

    def _redraw_geom(self, k):
        pass


QtW.QMenu = FakeMenu
h = Host()
FakeMenu.pick = None
h._cpr_tree_menu("B", 3, 4, None, "v7")
texts = [a.text for a in FakeMenu.seen[-1]]
check("menu: 編集 / 名前を表示 / 選択を解除 (in that order)",
      texts == [t("編集 (Edit)"), t("名前を表示"), t("選択を解除 (Deselect)")],
      str(texts))
check("選択を解除 enabled while a line is selected (orange)",
      FakeMenu.seen[-1][-1].enabled)
FakeMenu.pick = t("選択を解除 (Deselect)")
h._cpr_tree_menu("B", 3, 4, None, "v7")
check("選択を解除 → select callback with '' (clear)", h.sel == [""], str(h.sel))
h._coro_overlay = [{"vid": "v7", "selected": False}]
FakeMenu.pick = None
h._cpr_tree_menu("B", 3, 4, None, "v7")
check("nothing selected → 選択を解除 greyed", not FakeMenu.seen[-1][-1].enabled)
check("i18n en", i18n.t("選択を解除 (Deselect)") == "選択を解除 (Deselect)")


# Tree panel: select_vessel("") clears the selection + pushes the overlay.
class TW:
    select_vessel = ctw.CoronaryTreeWindow.select_vessel

    def __init__(self):
        self._tree_w = QTreeWidget()
        self.it = QTreeWidgetItem(["05"])
        self._tree_w.addTopLevelItem(self.it)
        self._tree_w.setCurrentItem(self.it)
        self.pushed = self.selcalls = 0

    def _on_selection(self):
        self.selcalls += 1

    def _push_overlay(self):
        self.pushed += 1


tw = TW()
tw.select_vessel("")
check("tree: '' clears the current row + selection",
      tw._tree_w.currentItem() is None and not tw._tree_w.selectedItems())
check("tree: overlay pushed (orange line redrawn normal)", tw.pushed == 1)
tw.select_vessel("")
check("tree: already clear → no extra push", tw.pushed == 1)

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
