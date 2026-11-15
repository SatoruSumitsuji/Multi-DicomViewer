"""Mac (pygfx) CPR short-axis pane: the Zoom tool works. Its camera is framed
from the section half-width, so Zoom scales a CPR view scale (as the VTK
camera ParallelScale zoom does on Windows), and the marker / drag mapping
(_cpr_section_scale) follows it. pygfx is not importable here, so the methods
are pulled out of the source and run on a stand-in object.
"""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(
    encoding="utf-8")
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


tree = ast.parse(SRC)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
           and any(isinstance(f, ast.FunctionDef) and f.name == "_cpr_view_scale"
                   for f in n.body))
fns = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
ns = {}
for name in ("_cpr_view_scale", "_cpr_section_scale"):
    exec(compile(ast.Module([fns[name]], []), "<src>", "exec"), ns)


class Canvas:
    def height(self):
        return 500


o = SimpleNamespace(_cpr={"half": 25.0},
                    pane={"A": SimpleNamespace(canvas=Canvas())})
o._cpr_view_scale = ns["_cpr_view_scale"].__get__(o)
o._cpr_section_scale = ns["_cpr_section_scale"].__get__(o)
check("default: ±half fills the pane (10 px/mm here)",
      abs(o._cpr_section_scale() - 10.0) < 1e-9)
o._cpr["view_scale"] = 0.5                       # zoomed in 2×
check("zoomed in: markers map at 2× (20 px/mm)",
      abs(o._cpr_section_scale() - 20.0) < 1e-9)
o._cpr["view_scale"] = 999
check("zoom clamped", o._cpr_view_scale() == 20.0)

zoom = ast.get_source_segment(SRC, fns["_drag"])
zoom = zoom[zoom.index('elif t == "ZOOM":'):zoom.index('elif t == "MOVE":')]
check("Zoom drag on pane A during CPR scales the CPR view",
      'k == "A" and self._cpr is not None' in zoom and "view_scale" in zoom)
cam = ast.get_source_segment(SRC, fns["_config_cpr_cam"])
check("the CPR camera uses the view scale",
      "self._cpr_view_scale()" in cam)

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
