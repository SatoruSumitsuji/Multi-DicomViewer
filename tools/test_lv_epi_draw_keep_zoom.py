"""Epi 'Draw' keeps the zoom (2026-11-16, 「見た目が変わらない」): the right pane
keeps the scale the user had before Draw — no default 0.75·length scale — on
both viewers, and the Mac first plane fit restores the scale like the VTK one.

    python tools/test_lv_epi_draw_keep_zoom.py
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


def code(text):
    return "\n".join(ln.split("#")[0] for ln in text.splitlines())


def seg(path, name):
    src = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    n = next(x for x in ast.walk(tree)
             if isinstance(x, ast.FunctionDef) and x.name == name)
    return ast.get_source_segment(src, n)


VTK = "multi_dicomviewer/viewers/ct_viewer.py"
MAC = "multi_dicomviewer/viewers/ct_viewer_pygfx.py"

d = code(seg(VTK, "_lv_epi_draw"))
check("VTK Draw sets no scale", "SetParallelScale" not in d and "0.75" not in d)
d = code(seg(MAC, "_lv_epi_draw"))
check("Mac Draw sets no scale", "_ps[" not in d and "0.75" not in d)

s = seg(VTK, "_lv_show_plane")
check("VTK first fit restores the zoom", "ps0 = float(cam.GetParallelScale())" in s
      and "cam.SetParallelScale(ps0)" in s)
s = seg(MAC, "_lv_show_plane")
check("Mac first fit restores the zoom",
      "ps0 = self._ps.get(pane) if first else None" in s
      and s.index("self._refresh(reset_cam=first)") < s.index("self._ps[pane] = ps0"))


# behaviour: the Mac show-plane restore on a fake host
class FakeMac:
    def __init__(self):
        self._ps = {"A": 1.0, "B": 42.0}
        self.calls = []

    def _refresh(self, reset_cam=False, only=None):
        self.calls.append(("refresh", reset_cam, only))
        if reset_cam:
            self._ps["B"] = 99.0              # the fit rescales

    def _config_cam(self, k):
        self.calls.append(("cam", k))


src = seg(MAC, "_lv_show_plane")
i0 = src.index("        ps0 = self._ps.get(pane) if first else None")
i1 = src.index("        self._overlay[pane].update()")
body = "\n".join(line[8:] for line in src[i0:i1].splitlines())
for first, want in ((True, 42.0), (False, 99.0)):
    h = FakeMac()
    if not first:
        h._ps["B"] = 99.0
    exec(body, {}, {"self": h, "first": first, "pane": "B"})
    check(f"Mac plane show (first={first}) keeps scale {want}", h._ps["B"] == want,
          str(h._ps))

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
