"""CoronaryTree: bar (2026-11-16): caption 「CoronaryTree:」, first button Load
(.corotree.json → the shell's corotree open, CoronaryTreeモード), the VR support
tools in the same row with no 「VR:」 caption, 回転連動 ON by default.

    python tools/test_coro_bar_load.py
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
# The VTK viewer needs a real GL context (offscreen has none on Windows);
# nothing is shown.
os.environ["QT_QPA_PLATFORM"] = ("windows" if sys.platform == "win32"
                                 else os.environ.get("QT_QPA_PLATFORM", "offscreen"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


v = CTViewer()
check("caption is CoronaryTree:", v._coro_cap.text() == t("CoronaryTree:"),
      v._coro_cap.text())
lay = v._coro_cap.parentWidget().layout()


def row_of(w):
    """(row layout, index) of widget *w* in the bar."""
    for i in range(lay.count()):
        sub = lay.itemAt(i).layout()
        if sub is None:
            continue
        for j in range(sub.count()):
            if sub.itemAt(j).widget() is w:
                return i, j
    return None


order = [v._coro_cap, v._coro_load_btn, v._coronary_mpr_btn, v._cpr_vr_btn,
         v._cpr_vr_rot_btn]
pos = [row_of(w) for w in order]
check("one row: CoronaryTree: Load CPR Open VR 回転連動",
      None not in pos and len({p[0] for p in pos}) == 1
      and [p[1] for p in pos] == sorted(p[1] for p in pos), str(pos))
check("first button is Load", v._coro_load_btn._full_text == t("Load")
      and pos[1][1] == pos[0][1] + 1)
labels = [lay.itemAt(i).layout().itemAt(j).widget()
          for i in range(lay.count()) if lay.itemAt(i).layout() is not None
          for j in range(lay.itemAt(i).layout().count())]
check("no 「VR:」 caption", not any(isinstance(w, QLabel) and w.text() in
                                   ("VR:", t("VR：")) for w in labels if w))
check("Load is part of the CoronaryTree group",
      v._coro_load_btn in v._coro_group_widgets)
check("回転連動 is ON by default", v._cpr_vr_rot_btn.isChecked())
check("no Epi status label", not hasattr(v, "_cpr_epi_lbl"))
check("no Set Epi-Data button", not hasattr(v, "_cpr_epi_btn")
      and not any(getattr(w, "_full_text", None) == t("Set Epi-Data") for w in labels))


class Shell:
    def __init__(self):
        self.opened = []

    def _open_coronary_tree_file(self, path, to_territory=True):
        self.opened.append((path, to_territory))


sh = Shell()
with mock.patch.object(CTViewer, "window", return_value=sh), \
        mock.patch("PyQt6.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=("C:/x/case.corotree.json", "")) as dlg:
    v._coro_load_btn.click()
check("Load opens the picked .corotree.json in CoronaryTreeモード",
      sh.opened == [("C:/x/case.corotree.json", False)], str(sh.opened))
check("the picker filters .corotree.json", "corotree.json" in dlg.call_args[0][3])
sh.opened.clear()
with mock.patch.object(CTViewer, "window", return_value=sh), \
        mock.patch("PyQt6.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=("", "")):
    v._coro_load_btn.click()
check("cancelled pick → nothing opened", sh.opened == [])

mac = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(encoding="utf-8")
check("Mac: same caption + Load button",
      't("CoronaryTree:")' in mac and "self._coro_load_btn = FitButton(t(\"Load\"))" in mac
      and "self._coro_group_widgets = [self._coro_cap, self._coro_load_btn," in mac)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
