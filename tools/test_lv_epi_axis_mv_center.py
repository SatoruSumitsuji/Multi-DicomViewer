"""Epi Draw after the AoV step: the 8 planes rotate about the apex → MV-centre
long axis, and the plane centre / crosshair sit ON that axis — not on the AoV
centre left over from the AoV step's view (keep_view used to keep the stale
centre, sliding the planes sideways so the centreline ran apex → AoV).

Both viewers: _lv_apply_common_apex_to_pass runs on a fake self (AST
extraction, no VTK / pygfx); _lv_show_plane is checked for the on-axis
projection of the kept centre.
"""
import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


def extract(fn, name):
    src = (ROOT / "multi_dicomviewer/viewers" / fn).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"np": np}
    exec(compile(ast.Module([node], []), fn, "exec"), ns)
    return ns[name], ast.get_source_segment(src, node)


class Model:
    endo_axis = epi_axis = axis = None

    def set_apex_point(self, pas, p):
        self.apex_pt = p


APEX = np.array([0.0, 0.0, 0.0])
MV_C = np.array([0.0, 0.0, 90.0])
AOV_C = np.array([25.0, 10.0, 85.0])


for fn in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    long_axis, _ = extract(fn, "_lv_long_axis_from_apex")
    apply_fn, _ = extract(fn, "_lv_apply_common_apex_to_pass")

    class V:
        _lv_long_axis_from_apex = long_axis

        def __init__(self):
            self._lv = {"model": Model(), "phase": "align", "pass": "epi"}
            self._lv_apex = APEX.copy()
            self._lv_valves = {"mitral": (MV_C, np.array([0, 0, 1.0]), 15.0),
                               "aortic": (AOV_C, np.array([0, 0, 1.0]), 12.0)}
            self._lvv = None
            self._center = AOV_C.copy()           # the AoV step's crosshair
            self._lv_up_ref = None

            class _P:
                def render(self):
                    pass
            self.pane = {"A": _P(), "B": _P()}

        def __getattr__(self, k):                 # UI refreshes → no-op
            return lambda *a, **kw: None

    v = V()
    apply_fn(v, "epi")
    ax = v._lv["model"].epi_axis
    check(f"{fn}: Epi axis = apex → MV centre",
          ax is not None and np.allclose(ax.axis, [0, 0, 1.0])
          and np.allclose(ax.apex, APEX), str(getattr(ax, "axis", None)))
    c = np.asarray(v._center, float)
    off = np.linalg.norm(c - (APEX + np.dot(c - APEX, [0, 0, 1.0]) * np.array([0, 0, 1.0])))
    check(f"{fn}: crosshair moved ONTO the axis (same level), not the AoV centre",
          off < 1e-9 and abs(c[2] - AOV_C[2]) < 1e-9, str(c))

    _, show_src = extract(fn, "_lv_show_plane")
    check(f"{fn}: _lv_show_plane keeps the centre on the axis under keep_view",
          "along = float(np.dot(pc0 - ax.apex, ax.axis))" in show_src
          and "self._pc[pane] = ax.apex + along * ax.axis" in show_src)

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
