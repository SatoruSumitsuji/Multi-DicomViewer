"""LV Epi re-entry (both CT viewers, VTK + pygfx):

1. Entering Epi with NO traced border re-seeds the Epi long axis from the
   common apex + the CURRENT MV centre. Before, an axis seeded by an earlier
   Draw stayed on the OLD MV centre after the MV was re-done (the long axis
   passed the apex but not the MV centre).
2. Draw while already in the trace ('contour') re-enters the contour; it no
   longer calls _lv_start_trace, which read that as UNDO → ready (apex
   cleared), so Draw had to be pressed twice and the dialog came back.
3. Mac: the Epi HU補正 band is drawn thicker / more opaque than the old thin
   1.2 px DotLine.

The viewers need VTK / pygfx (not in the sandbox), so the methods are pulled
out with AST and run on a small stand-in object.
"""
import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_dicomviewer.core import lv_axis  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


class _Btn:
    def __init__(self, box):
        self.box, self._cb = box, []

    @property
    def clicked(self):
        btn = self

        class _Sig:
            def connect(self, f):
                btn._cb.append(f)
        return _Sig()


class _Box:
    """QMessageBox stand-in: exec() 'clicks' the Draw (AcceptRole) button."""
    class Icon:
        Information = 0

    class ButtonRole:
        AcceptRole = 0

    class StandardButton:
        Cancel = 1

    def __init__(self, *a):
        self._draw = None

    def setWindowTitle(self, *a): pass
    def setIcon(self, *a): pass
    def setText(self, *a): pass

    def addButton(self, *a):
        b = _Btn(self)
        if self._draw is None:
            self._draw = b
        return b

    def exec(self):
        for f in self._draw._cb:
            f()

    def clickedButton(self):
        return self._draw


class _Timer:
    @staticmethod
    def singleShot(*a):
        pass


class _Model:
    def __init__(self):
        self.epi_contours, self.epi_planes = [], {}
        self.endo_contours, self.endo_planes = [], {}
        self.epi_axis = self.endo_axis = self.axis = None
        self.epi_apex = self.endo_apex = None

    def set_apex_point(self, which, p):
        setattr(self, which + "_apex", None if p is None else np.asarray(p, float))

    def _axis_for(self, which):
        return self.epi_axis if which == "epi" else self.endo_axis


def methods(fn, names):
    src = (ROOT / "multi_dicomviewer" / "viewers" / fn).read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names \
                and node.name not in out:
            mod = ast.Module(body=[node], type_ignores=[])
            ns = {"np": np, "t": t, "QTimer": _Timer}
            exec(compile(mod, fn, "exec"), ns)
            out[node.name] = ns[node.name]
    return src, out


NAMES = ("_lv_pass_has_border", "_lv_refresh_pass_axis",
         "_lv_long_axis_from_apex", "_lv_epi_draw",
         "_lv_apply_common_apex_to_pass")

import types  # noqa: E402
qtw = types.ModuleType("PyQt6.QtWidgets")
qtw.QMessageBox = _Box
import PyQt6  # noqa: E402,F401
sys.modules["PyQt6.QtWidgets"] = qtw            # only the stubs above use it

for fn in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    src, M = methods(fn, NAMES)
    check(f"{fn}: helpers present", set(M) == set(NAMES), str(sorted(M)))

    class V:
        pass
    for k, f in M.items():
        setattr(V, k, f)
    v = V()
    apex = np.array([0.0, 0.0, 0.0])
    old_mv, new_mv = np.array([0.0, 0.0, 80.0]), np.array([20.0, 0.0, 80.0])
    v._lv_apex = apex
    v._lvv = None
    v._lv_valves = {"mitral": (old_mv, np.array([0, 0, 1.0]), 15.0)}
    v._lv = {"model": _Model(), "phase": "contour", "pass": "epi"}
    v._lv_refresh_pass_axis("epi")
    a0 = np.asarray(v._lv["model"].epi_axis.axis)
    v._lv_valves["mitral"] = (new_mv, np.array([0, 0, 1.0]), 15.0)   # MV re-done
    v._lv_refresh_pass_axis("epi")
    a1 = np.asarray(v._lv["model"].epi_axis.axis)
    want = (new_mv - apex) / np.linalg.norm(new_mv - apex)
    check(f"{fn}: refreshed Epi axis = apex → NEW MV centre",
          np.allclose(a1, want) and not np.allclose(a0, a1), f"{a1}")
    check(f"{fn}: no border → has_border False",
          not v._lv_pass_has_border("epi"))
    v._lv["model"].epi_planes = {0: [1]}
    check(f"{fn}: traced border → has_border True", v._lv_pass_has_border("epi"))
    v._lv["model"].epi_planes = {}

    # Select-pass: the refresh runs (when no border) BEFORE the axis is used.
    fsrc = ast.get_source_segment(src, next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_lv_select_pass"))
    i_ref = fsrc.find("self._lv_refresh_pass_axis(which)")
    i_ax = fsrc.find("ax = m._axis_for(which)")
    check(f"{fn}: Epi/Endo entry refreshes the axis before using it",
          0 <= i_ref < i_ax and "_lv_pass_has_border(which)" in fsrc)

    # Draw while already tracing → enter contour, NOT _lv_start_trace (UNDO).
    calls = []
    v._lv_current_submode = lambda: "epi"
    v.window = lambda: None
    v._lv_enter_contour = lambda: calls.append("enter")
    v._lv_start_trace = lambda: calls.append("start")
    v._lv_update_submode_ui = lambda: None
    v._lv_sync_buttons = v._lv_update_text = v._lv_show_plane = \
        v._lv_redraw_all = v._reset_pointer_state = lambda *a: None
    v._ps = {}

    class _Cam:
        def SetParallelScale(self, *a): pass

    class _Ren:
        def GetActiveCamera(self): return _Cam()

    class _P:
        ren = _Ren()

        def render(self): pass
    v.pane = {"A": _P(), "B": _P()}
    v._lv["phase"] = "contour"
    v._lv["sax"] = None
    v._lv_epi_draw()
    check(f"{fn}: Draw in 'contour' re-enters the trace (no UNDO)",
          calls == ["enter"], str(calls))
    check(f"{fn}: Draw keeps the Epi apex", v._lv["model"].epi_apex is not None)
    calls.clear()
    v._lv["phase"] = "ready"
    v._lv_epi_draw()
    check(f"{fn}: Draw in 'ready' still starts the trace", calls == ["start"],
          str(calls))

pg = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(encoding="utf-8")
i = pg.find("_ebands = v._lv_epi_band_2d(key)")
seg = pg[i:i + 700]
check("Mac Epi band: thicker + more opaque dashes",
      "QColor(46, 139, 87, 230), 2.2)" in seg and "PenStyle.DotLine" not in seg)

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
