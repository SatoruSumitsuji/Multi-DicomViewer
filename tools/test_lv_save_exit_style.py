"""LV steps: Save / Exit get a coloured OUTLINE only (no fill — they act at once,
no running state), Save and Exit in different colours, disabled = plain grey.
The style is applied after every construction-time re-style in both viewers.
"""
import ast
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.viewers import valve_auto_mixin as vam  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


SAVE = ("_lv_save_btn", "_lvv_save_btn", "_lv_ve_save_btn", "_lv_ax_save_btn")
EXIT = ("_lv_exit_btn", "_lvv_exit_btn", "_lv_ve_exit_btn", "_lv_ax_exit_btn")


class V(vam.ValveAutoMixin):
    _BTN_DIS = "QPushButton:disabled{background:#e6e6e6;}"


v = V()
for nm in SAVE + EXIT:
    setattr(v, nm, QPushButton())
v._lv_style_save_exit()
for nm in SAVE:
    s = getattr(v, nm).styleSheet()
    check(f"{nm}: Save outline, no fill colour", vam._SAVE_OUTLINE in s
          and "background:palette(button)" in s and s.endswith(V._BTN_DIS))
for nm in EXIT:
    s = getattr(v, nm).styleSheet()
    check(f"{nm}: Exit outline", vam._EXIT_OUTLINE in s and s.endswith(V._BTN_DIS))
check("Save and Exit colours differ", vam._SAVE_OUTLINE != vam._EXIT_OUTLINE)

for fn in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    src = (ROOT / "multi_dicomviewer" / "viewers" / fn).read_text(encoding="utf-8")
    tree = ast.parse(src)
    call = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_lv_style_save_exit"]
    later = []
    names = set(SAVE + EXIT) | {"_lvv_ctrl_btns"}
    for n in ast.walk(tree):
        if isinstance(n, ast.For):
            it = {a.attr for a in ast.walk(n.iter) if isinstance(a, ast.Attribute)}
            sty = any(isinstance(c, ast.Attribute) and c.attr == "setStyleSheet"
                      for c in ast.walk(ast.Module(n.body, [])))
            if it & names and sty and call and n.lineno > call[0]:
                later.append(n.lineno)
    check(f"{fn}: outline applied once, after the construction re-styles",
          len(call) == 1 and not later, f"{call} {later}")

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
