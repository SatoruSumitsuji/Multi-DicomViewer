"""Leaving Epi with Exit BEFORE any border is traced (both CT viewers):

* the white dotted guide lines ⟂ the LV long axis are no longer drawn — they
  belong to an ACTIVE Endo/Epi pass only;
* the trace is left completely, so the Rotate / Paging lock (phase 'contour')
  is lifted: Windows → _lv_reset_contour_empty ('align') + Measure off + the
  tool locks recomputed; Mac → _lv_exit (as with a traced border).

VTK / pygfx can't run in the sandbox: the guide painters are run on a stand-in
object, the Exit branches are checked in the source.
"""
import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


def func(src, name):
    tree = ast.parse(src)
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)


class _Ax:
    apex = np.zeros(3)


class _Model:
    axis = _Ax()


def run_guides(fn, name, pas):
    src = (ROOT / "multi_dicomviewer/viewers" / fn).read_text(encoding="utf-8")
    node = func(src, name)
    drawn = []

    class _Any:
        def __getattr__(self, k):
            return _Call(k)

    class _Call:
        def __init__(self, k):
            self.k = k

        def __call__(self, *a, **kw):
            drawn.append(self.k)

        def __getattr__(self, k):
            return _Call(k)

    class _Pen(_Any):
        def __init__(self, *a): pass

    ns = {"np": np, "QPen": _Pen, "QColor": lambda *a: None, "Qt": _Any(),
          "_polylines_pd": lambda segs: segs}
    exec(compile(ast.Module([node], []), fn, "exec"), ns)
    v = type("V", (), {})()
    v._lv = {"model": _Model(), "pass": pas, "phase": "contour"}
    v._lvv = None
    v._lv_valves = {"mitral": (np.array([0.0, 50.0, 0.0]), None, 10.0)}
    v._half = 100.0
    v._world3d_to_out = lambda key, P: (float(P[0]), float(P[1]))
    p = _Any()
    if fn == "ct_viewer.py":
        p.lv_axis_guide_mapper = _Any()
        ns[name](v, "B", p)
    else:
        host = type("H", (), {})()
        host._v = v
        ns[name](host, p, "B", lambda q: q)
    return bool(drawn)


for fn, name in (("ct_viewer.py", "_lv_draw_axis_guides"),
                 ("ct_viewer_pygfx.py", "_paint_lv_axis_guides")):
    check(f"{fn}: guides drawn while Epi is active", run_guides(fn, name, "epi"))
    check(f"{fn}: no guides after Exit (pass None)",
          not run_guides(fn, name, None))

src = (ROOT / "multi_dicomviewer/viewers/ct_viewer.py").read_text(encoding="utf-8")
body = ast.get_source_segment(src, func(src, "_lv_submode_exit"))
tail = body[body.find("elif has_border:"):]
check("Windows Exit, nothing traced → full reset (no lingering 'contour')",
      'self._lv["pass"] = None' not in tail
      and tail.count("self._lv_reset_contour_empty()") == 2)
check("Windows Exit → Measure off + tool locks recomputed",
      "self._toggle_measure()" in tail and "self._lv_apply_view_free()" in tail)

pg = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(encoding="utf-8")
body = ast.get_source_segment(pg, func(pg, "_lv_submode_exit"))
check("Mac Exit → _lv_exit whether traced or not",
      "self._lv_exit()" in body and 'self._lv["pass"] = None' not in body)

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
