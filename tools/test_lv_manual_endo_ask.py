# -*- coding: utf-8 -*-
"""LV Blood/Endo: Endo-Border(Manual) is shown pale (the grey text of the
disabled row-2 buttons) yet clickable, and asks 「本当に Manual Endo-Border を
使いますか？」 first — only Yes enters the manual Endo edit (user 2026-11-18).
Both viewers (VTK + pygfx).

    python tools/test_lv_manual_endo_ask.py
"""
import ast
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

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


answer = {"r": QMessageBox.StandardButton.No}
asked = []


def _question(*a, **k):
    asked.append(a)
    return answer["r"]


QMessageBox.question = staticmethod(_question)


class Host:
    def __init__(self):
        self.entered = 0

    def window(self):
        return None

    def _lvv_manual_endo(self, *a):
        self.entered += 1


for nm in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    src = (ROOT / "multi_dicomviewer/viewers" / nm).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "_lvv_manual_endo_ask")
    ns = {}
    code = ast.Module(body=[fn], type_ignores=[])
    from multi_dicomviewer.i18n import t
    exec(compile(code, nm, "exec"), {"t": t}, ns)
    ask = ns["_lvv_manual_endo_ask"]
    h = Host()
    answer["r"] = QMessageBox.StandardButton.No
    asked.clear()
    ask(h)
    check(f"{nm}: asks first; No → nothing", asked and h.entered == 0)
    answer["r"] = QMessageBox.StandardButton.Yes
    ask(h)
    check(f"{nm}: Yes → manual Endo edit", h.entered == 1)
    check(f"{nm}: the button goes through the question",
          "self._lvv_manual_endo_btn.clicked.connect(self._lvv_manual_endo_ask)"
          in src)
    check(f"{nm}: pale text (#a8a8a8) while enabled",
          '"QPushButton{color:#a8a8a8;}" + self._BTN_DIS' in src)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
