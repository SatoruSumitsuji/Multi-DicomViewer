# -*- coding: utf-8 -*-
"""Renaming a CPR line onto a name whose file already exists (user
2026-11-26): it used to stop with 「同名ファイルが既にある」. Now it asks
「上書き」 (default; the old file goes to _deleted) / 「-1 などを付けた名前に
する」 / 「変更しない」; 上書き is not offered when another line uses that
file. Synthetic files, the prompt answered by the test.

    python tools/test_cpr_rename_clash.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.ui.coronary_tree_window import (  # noqa: E402
    CoronaryTreeWindow as W)

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


class V:
    def __init__(self, name):
        self.name, self.detail_name, self.fix_name = name, "", ""


class Tree:
    def __init__(self):
        self.vessels = {"a": V("01"), "b": V("RCA")}
        self.fix_name = ""


class TW:
    DELETED_DIR = W.DELETED_DIR
    _display_name = staticmethod(W._display_name)
    _read_cpr = staticmethod(W._read_cpr)
    _write_cpr = staticmethod(W._write_cpr)
    _tree_fix = W._tree_fix
    _rename_base = W._rename_base
    _batch_rename = W._batch_rename
    _emit_renamed = W._emit_renamed
    _retarget_viewer_path = W._retarget_viewer_path
    _file_owner = W._file_owner
    _retire_path = W._retire_path

    def __init__(self, folder, answer):
        self._tree = Tree()
        self._last_path = None
        self.b = {"paths": {}}
        for vid, v in self._tree.vessels.items():
            p = os.path.join(folder, v.name + ".cpr.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"ctrl": [[0, 0, 0], [1, 1, 1]], "who": vid}, f)
            self.b["paths"][vid] = p
        self.answer, self.asked, self.warned = answer, [], []
        self._hint = type("H", (), {"setText": lambda *_a: None})()
        self.treeChanged = type("S", (), {"emit": lambda *_a: None})()

    def _ask_name_clash(self, fname, can_overwrite=True):
        self.asked.append((fname, can_overwrite))
        return self.answer

    def _bundle(self):
        return self.b

    def _populate(self):
        pass

    def _push_overlay(self):
        pass

    def _refresh_rename_btn(self):
        pass

    def _source_viewer(self):
        return None

    def _warn(self, m):
        self.warned.append(m)


def setup(answer):
    d = tempfile.mkdtemp(dir=HOME)
    w = TW(d, answer)
    stray = os.path.join(d, "LAD.cpr.json")      # a file not in the tree
    with open(stray, "w", encoding="utf-8") as f:
        json.dump({"ctrl": [[5, 5, 5], [6, 6, 6]], "who": "stray"}, f)
    w._tree.vessels["a"].detail_name = "LAD"     # rename 01 → LAD
    return d, w, stray


def who(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("who")


# ---- 上書き
d, w, stray = setup("overwrite")
w._batch_rename(only={"a"}, ask=False)
dl = os.path.join(d, W.DELETED_DIR)
check("asked, with 上書き offered (no line uses that file)",
      w.asked == [("LAD.cpr.json", True)], w.asked)
check("上書き: 01 → LAD.cpr.json, the old LAD file in _deleted",
      who(stray) == "a" and not os.path.exists(os.path.join(d, "01.cpr.json"))
      and os.path.isdir(dl) and who(os.path.join(dl, "LAD.cpr.json")) == "stray"
      and w._tree.vessels["a"].name == "LAD"
      and w.b["paths"]["a"] == stray, os.listdir(d))

# ---- -1 などを付ける
d, w, stray = setup("suffix")
w._batch_rename(only={"a"}, ask=False)
p1 = os.path.join(d, "LAD-1.cpr.json")
check("-1: 01 → LAD-1.cpr.json, LAD.cpr.json untouched",
      os.path.isfile(p1) and who(p1) == "a" and who(stray) == "stray"
      and w._tree.vessels["a"].name == "LAD-1"
      and W._display_name(w._tree.vessels["a"]) == "LAD-1"
      and w.b["paths"]["a"] == p1, os.listdir(d))
with open(os.path.join(d, "LAD-2.cpr.json"), "w", encoding="utf-8") as f:
    json.dump({"ctrl": []}, f)
w._tree.vessels["b"].detail_name = "LAD"     # RCA → LAD: -1 / -2 taken
w._batch_rename(only={"b"}, ask=False)
check("… the next free number (-3 when -1 / -2 are there)",
      w._tree.vessels["b"].name == "LAD-3"
      and os.path.isfile(os.path.join(d, "LAD-3.cpr.json")), os.listdir(d))

# ---- 変更しない
d, w, stray = setup(None)
w._batch_rename(only={"a"}, ask=False)
check("変更しない: nothing renamed, user told",
      os.path.isfile(os.path.join(d, "01.cpr.json")) and who(stray) == "stray"
      and w._tree.vessels["a"].name == "01" and w.warned, w.warned)

# ---- a file another line uses: no 上書き
d, w, stray = setup("suffix")
w.b["paths"]["b"] = stray                    # RCA now owns LAD.cpr.json
w._batch_rename(only={"a"}, ask=False)
check("another line's file: 上書き not offered",
      w.asked == [("LAD.cpr.json", False)] and who(stray) == "stray", w.asked)

# ---- the prompt itself (real QMessageBox, clicked by a timer)
from PyQt6.QtWidgets import QWidget  # noqa: E402

dk = QWidget()                               # the prompt's parent
seen = {}


def click(label_fn):
    def go():
        box = QApplication.activeModalWidget()
        if box is None:
            QTimer.singleShot(20, go)
            return
        seen["labels"] = [b.text() for b in box.buttons()]
        seen["default"] = box.defaultButton().text()
        label_fn(box)
    QTimer.singleShot(20, go)


from multi_dicomviewer.i18n import t  # noqa: E402

click(lambda box: box.defaultButton().click())
r = W._ask_name_clash(dk, "LAD.cpr.json", True)
check("prompt: 上書き / -1 / 変更しない, 上書き the default",
      r == "overwrite" and seen["default"] == t("上書き")
      and set(seen["labels"]) == {t("上書き"), t("「-1」などを付けた名前にする"),
                                  t("変更しない")}, seen)
click(lambda box: box.defaultButton().click())
r = W._ask_name_clash(dk, "LAD.cpr.json", False)
check("prompt without 上書き: -1 is the default",
      r == "suffix" and t("上書き") not in seen["labels"], seen)
click(lambda box: box.reject())
check("Esc / close = 変更しない", W._ask_name_clash(dk, "x", True) is None)

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
