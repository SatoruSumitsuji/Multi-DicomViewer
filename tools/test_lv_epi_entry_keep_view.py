"""LV mode (2026-11-16): entering Epi / Endo after MV-AoV Set + Apex Set must
NOT move the image — the long-axis view the user set up before entering is
kept (no automatic MV-perpendicular re-orient in the MV-AoV point mode). The
old MV-plane mode keeps its automatic re-orient. (The Mac viewer has no
automatic re-orient on entry.)"""
import ast
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
fails = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


src = (ROOT / "multi_dicomviewer" / "viewers" / "ct_viewer.py").read_text(
    encoding="utf-8")
fn = next(n for n in ast.walk(ast.parse(src))
          if isinstance(n, ast.FunctionDef) and n.name == "_lv_show_mv_perp_auto")
ns = {}
exec(textwrap.dedent(ast.get_source_segment(src, fn)), ns)
show = ns["_lv_show_mv_perp_auto"]


class V:
    def __init__(self):
        self._lv_valve_shown = {"mitral": False}
        self._measures = {"A": [{"_lv_valve": "mitral", "hidden": True}],
                          "B": [{"_lv_valve": "mitral", "hidden": True}]}
        self.views = 0

    def _redraw_meas(self, k):
        pass

    def _lv_update_valve_buttons(self):
        pass

    def _lv_view_mv_perpendicular(self, **k):
        self.views += 1


v = V()
show(v, reorient=False)
check("MV-AoV mode: point shown", v._lv_valve_shown["mitral"]
      and not v._measures["B"][0]["hidden"])
check("MV-AoV mode: view NOT re-oriented", v.views == 0)
v2 = V()
show(v2)
check("MV-plane mode: re-orient kept", v2.views == 1)
check("entry passes reorient=not MV-AoV",
      "self._lv_show_mv_perp_auto(reorient=not self._lv_mvaov_on())" in src)
gfx = (ROOT / "multi_dicomviewer" / "viewers" / "ct_viewer_pygfx.py").read_text(
    encoding="utf-8")
g = next(n for n in ast.walk(ast.parse(gfx))
         if isinstance(n, ast.FunctionDef) and n.name == "_lv_enter_align")
check("Mac: no re-orient on entry",
      "_lv_view_mv_perpendicular" not in ast.get_source_segment(gfx, g))
print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
