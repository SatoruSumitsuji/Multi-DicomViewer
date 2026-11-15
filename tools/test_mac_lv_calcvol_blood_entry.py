"""Mac (pygfx) LV:

1. Calc Vol turns Epi-Area ON and shows the red region at once (Windows
   parity) — the region appearing tells the user the calculation finished.
   Before, the region showed only if Epi-Area was already on.
2. Entering Blood/Endo shows the right image straight away: the panes are
   re-rendered after the Epi mask / 全域HU tint are set, and the deferred 1.5×
   zoom does a full refresh. The tint images are sampled for the pane's
   zoom / pan (_lvv_plane_rgba uses par["ps"]), so a zoom change with only
   render() left them stale until the centreline was nudged.

pygfx can't run in the sandbox → source (AST) checks.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


src = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(encoding="utf-8")
tree = ast.parse(src)


def body(name):
    return ast.get_source_segment(src, next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name))


cv = body("_lv_compute_volume")
i_mask = cv.find("self._lv_show_measured_mask(")
tail = cv[i_mask:]
check("Calc Vol → Epi-Area button checked + red style",
      "self._lv_region_btn.setChecked(True)" in tail and "#ff5a5a" in tail)
check("Calc Vol → region shown (mask on + redraw + view-free)",
      "self._lvv_mask_on = True" in tail and "self._lvv_redraw()" in tail
      and "self._lv_apply_view_free()" in tail)

tg = body("_lvv_toggle")
i_setup = tg.find("self._lvv_setup_axis_views()")
i_hl = tg.find("self._lvv_hl_on = True")
i_ref = tg.find("self._refresh()", i_hl)
check("Blood/Endo entry re-renders AFTER the tint flags / Epi mask",
      0 <= i_setup < i_hl < i_ref, f"{i_setup} {i_hl} {i_ref}")

rz = body("_lvv_reapply_default_zoom")
check("deferred 1.5× zoom does a full refresh (tints rebuilt)",
      rz.find("self._lvv_enlarge_default_zoom()") < rz.find("self._refresh()"))
check("tint image depends on the zoom (why the refresh is needed)",
      'par["ps"]' in body("_lvv_plane_rgba"))

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
