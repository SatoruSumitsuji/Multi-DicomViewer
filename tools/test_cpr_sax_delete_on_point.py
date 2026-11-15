"""CPR short-axis (left pane) right-click ▸ Add Point works only BETWEEN edit
points; Delete Point works ONLY while the
section sits exactly on an edit point (both viewers). Between edit points the
item is disabled ("only on an edit point") and nothing is deleted — before, it
deleted the NEAREST edit point, which could be the wrong one.
"""
import ast
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from multi_dicomviewer import i18n  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.viewers.cpr_mixin import CPRMixin  # noqa: E402

i18n.set_language("ja")
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


class _CL:
    def __init__(self, n):
        self.points = np.array([[0.0, 0.0, float(i)] for i in range(n)])


class V(CPRMixin):
    def __init__(self):
        self.rebuilt = 0
        pts = [[0.0, 0.0, z] for z in (0.0, 10.0, 20.0, 30.0)]
        self._measures = {"B": [{"pts3d": pts}], "A": []}
        self._cpr = {"src": "B", "src_mi": 0, "idx": 10, "cl": _CL(31)}

    def _cpr_set_ctrl(self, m, src, p3):
        m["pts3d"] = [list(map(float, q)) for q in p3]
        self._cpr.pop("_ctrl_idx", None)

    def _cpr_edit_rebuild(self):
        self.rebuilt += 1


v = V()
check("on an edit point → _cpr_at_ctrl gives it", v._cpr_at_ctrl() == 1)
v._cpr_delete_ctrl_near()
zs = [q[2] for q in v._measures["B"][0]["pts3d"]]
check("Delete removes THAT edit point", zs == [0.0, 20.0, 30.0] and v.rebuilt == 1,
      str(zs))
v._cpr["idx"] = 14                      # between 0 and 20 → not on a point
check("between edit points → not on a point", v._cpr_at_ctrl() is None)
v._cpr_delete_ctrl_near()
zs = [q[2] for q in v._measures["B"][0]["pts3d"]]
check("between edit points → nothing deleted", zs == [0.0, 20.0, 30.0]
      and v.rebuilt == 1, str(zs))


# Menu: Delete item disabled (with the hint text) between edit points.
class _Act:
    def __init__(self, text):
        self.text, self.enabled = text, True

    def setEnabled(self, on):
        self.enabled = bool(on)


class _Menu:
    last = None

    def __init__(self, *a):
        self.acts = []
        _Menu.last = self

    def addAction(self, text):
        a = _Act(text)
        self.acts.append(a)
        return a

    def exec(self, *a):
        return None


qtw = types.ModuleType("PyQt6.QtWidgets")
qtw.QMenu = _Menu
qtg = types.ModuleType("PyQt6.QtGui")
qtg.QCursor = types.SimpleNamespace(pos=lambda: None)
real = {k: sys.modules.get(k) for k in ("PyQt6.QtWidgets", "PyQt6.QtGui")}
sys.modules["PyQt6.QtWidgets"], sys.modules["PyQt6.QtGui"] = qtw, qtg
try:
    for fn in ("ct_viewer.py", "ct_viewer_pygfx.py"):
        src = (ROOT / "multi_dicomviewer/viewers" / fn).read_text(encoding="utf-8")
        node = next(n for n in ast.walk(ast.parse(src))
                    if isinstance(n, ast.FunctionDef)
                    and n.name == "_cpr_context_menu")
        ns = {"t": t, "np": np}
        exec(compile(ast.Module([node], []), fn, "exec"), ns)
        for idx, want in ((20, True), (14, False)):
            v._cpr["idx"] = idx
            ns["_cpr_context_menu"](v, 0, 0)
            d = _Menu.last.acts[1]
            check(f"{fn}: idx {idx} → Delete enabled={want}",
                  d.enabled == want
                  and d.text == (t("Delete Point") if want else
                                 t("Delete Point (only on an edit point)")),
                  d.text)
            a = _Menu.last.acts[0]
            check(f"{fn}: idx {idx} → Add enabled={not want}",
                  a.enabled == (not want)
                  and a.text == (t("Add Point (only between edit points)")
                                 if want else t("Add Point")), a.text)
finally:
    for k, m in real.items():
        if m is not None:
            sys.modules[k] = m

# Add on an edit point is a no-op (would stack a duplicate there).
v._cpr["idx"] = 20
n0 = len(v._measures["B"][0]["pts3d"])
v._cpr_add_ctrl_at(np.array([1.0, 0.0, 20.0]))
check("Add on an edit point → nothing added",
      len(v._measures["B"][0]["pts3d"]) == n0)
check("i18n ja add hint", t("Add Point (only between edit points)")
      == "点を追加（編集点の間でのみ）")
check("i18n ja hint", t("Delete Point (only on an edit point)")
      == "点を削除（編集点にいる時のみ）")
print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
