"""SAX entry: the long-axis pane is centred on the LEVEL point (the yellow
cross-section line), not left at the camera's earlier pan — the line was
"somewhere else every time". _lv_sax_apply_default_scale is lifted from both
viewers via ast (no VTK / pygfx) onto fakes.
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


def lift(fn, name):
    src = (ROOT / "multi_dicomviewer" / "viewers" / fn).read_text(encoding="utf-8")
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            ns = {"np": np}
            exec(compile(ast.Module([n], []), fn, "exec"), ns)
            return ns[name]
    raise LookupError(name)


class Ax:
    length_mm = 80.0


class Cam:
    def __init__(self):
        self.focal, self.pos, self.ps = (12.0, -30.0, 5.0), (12.0, -30.0, 500.0), 1
        self.calls = []

    def GetFocalPoint(self):
        return self.focal

    def GetPosition(self):
        return self.pos

    def SetFocalPoint(self, *a):
        self.focal = a

    def SetPosition(self, *a):
        self.pos = a

    def SetParallelScale(self, s):
        self.ps = s


class Pane:
    def __init__(self):
        self.cam = Cam()
        self.ren = self

    def GetActiveCamera(self):
        return self.cam


class Base:
    def __init__(self):
        self._lv = {"sax": 40.0, "model": type("M", (), {"axis": Ax()})(),
                    "pane": "B", "sax_pane": "A"}
        self._view_initial = True
        self._vr_on = {}
        self.pane = {"A": Pane(), "B": Pane()}
        self._pan = {"A": np.array([3.0, 4.0]), "B": np.array([-25.0, 18.0])}
        self._ps = {"A": 1.0, "B": 1.0}
        self.cfg = []

    def _lv_sax_short_scale(self, ax, along):
        return 30.0

    def _fit_pane(self, k):
        pass

    def _config_cam(self, k):
        self.cfg.append((k, self._pan[k].copy(), self._ps[k]))


# --- VTK
V = type("V", (Base,), {
    "_lv_sax_apply_default_scale": lift("ct_viewer.py", "_lv_sax_apply_default_scale"),
    "_center_camera": lift("ct_viewer.py", "_center_camera")})
v = V()
v._lv_sax_apply_default_scale()
cam = v.pane["B"].cam
check("VTK: long-axis camera centred on the level point (output 0,0)",
      cam.focal[:2] == (0.0, 0.0) and cam.pos[:2] == (0.0, 0.0), str(cam.focal))
check("VTK: default scale kept (0.75 x MV-apex length)", cam.ps == 60.0)
check("VTK: the short-axis pane camera not moved",
      v.pane["A"].cam.focal[:2] == (12.0, -30.0))

# --- pygfx
G = type("G", (Base,), {
    "_lv_sax_apply_default_scale": lift("ct_viewer_pygfx.py",
                                        "_lv_sax_apply_default_scale")})
g = G()
g._lv_sax_apply_default_scale()
check("pygfx: long-axis pan cleared → level line at the pane centre",
      np.allclose(g._pan["B"], 0) and ("B" in [c[0] for c in g.cfg]))
check("pygfx: default scale kept", g._ps["B"] == 60.0)
check("pygfx: the short-axis pane pan untouched", np.allclose(g._pan["A"], [3, 4]))

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
