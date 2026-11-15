"""MultiReg keeps the main-screen layout: 2×2 XA | IVUS#1 / CT short-axis |
IVUS#10 opens as the same 2×2 (the CT short-axis used to be appended last, so
the bottom row came out swapped). The MP4 tiles follow the same cells.
Also: the Mac (pygfx) CPR short-axis pane pans with Move (as on Windows).
Synthetic stand-ins only; CoregWindow is replaced by a recorder.
"""
import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.ui import coreg_window as cw  # noqa: E402
from multi_dicomviewer.ui import main_window as mw  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


G = mw._MAX_GRID_COLS


class V:
    def __init__(self, cpr=False):
        self._frame, self._reversed = 0, False
        self.handles_modality = "CT" if cpr else "XA"
        self._cpr = cpr

    def current_side(self):
        return None

    def cpr_active(self):
        return self._cpr

    def cpr_cosync_spec(self):
        return {"kind": "ct_cpr", "label": "CT"}


class Pane:
    def __init__(self, uid, cpr=False):
        self.uid, self.v = uid, V(cpr)

    def shown_series_uid(self):
        return self.uid

    def current_viewer(self):
        return self.v


def series(uid, mod):
    return SimpleNamespace(uid=uid, modality=mod, label=uid)


xa, iv1, ct, iv10 = (Pane("xa"), Pane("iv1"), Pane("ct", cpr=True),
                     Pane("iv10"))
filler = [Pane(f"f{i}") for i in range(G * 2)]
order = list(filler)
order[0], order[1], order[G], order[G + 1] = xa, iv1, ct, iv10
got = {}


class FakeCoreg:
    def __init__(self, specs, grid_cells=None):
        got["specs"], got["cells"] = specs, grid_cells

    def showMaximized(self):
        pass

    raise_ = showMaximized


real = cw.CoregWindow
cw.CoregWindow = FakeCoreg
try:
    o = SimpleNamespace(
        _layout_key="2x2", _order=order, _layout_cells=[0, 1, G, G + 1],
        _series_by_uid={"xa": series("xa", Modality.XA),
                        "iv1": series("iv1", Modality.IVUS),
                        "iv10": series("iv10", Modality.IVUS)},
        _as_taskbar_window=lambda w: None)
    for name in ("_shown_panes", "_current_layout_rect", "_open_coreg_ivus"):
        setattr(o, name, mw.MainWindow.__dict__[name].__get__(o))
    o._open_coreg_ivus()
finally:
    cw.CoregWindow = real

labels = [s["label"] if isinstance(s, dict) else s[0].uid for s in got["specs"]]
check("specs in screen order, CT short-axis in place",
      labels == ["xa", "iv1", "CT", "iv10"], str(labels))
check("cells = the 2×2 positions", got["cells"] == [(0, 0), (0, 1), (1, 0), (1, 1)],
      str(got["cells"]))

L = cw.CoregWindow._layout_cells
check("MultiReg grid uses those cells",
      L([(0, 0), (0, 1), (1, 0), (1, 1)], 4) == [(0, 0), (0, 1), (1, 0), (1, 1)])
check("1×3 stays one row", L([(0, 0), (0, 1), (0, 2)], 3) == [(0, 0), (0, 1), (0, 2)])
check("an empty column is squeezed out",
      L([(0, 0), (0, 2), (1, 0), (1, 2)], 4) == [(0, 0), (0, 1), (1, 0), (1, 1)])
check("Bi (two views, one cell) → reading-order grid",
      L([(0, 0), (0, 0), (0, 1)], 3) == [(0, 0), (0, 1), (1, 0)])
check("no cells → reading-order grid", L(None, 4) == [(0, 0), (0, 1), (1, 0), (1, 1)])

# MP4 tiles follow the cells: a 1×3 row composes 3 wide, 1 high.
from PyQt6.QtWidgets import QLabel  # noqa: E402
import numpy as np  # noqa: E402
from multi_dicomviewer.viewers.image_canvas import ImageCanvas  # noqa: E402


class CP:
    def __init__(self):
        self.canvas = ImageCanvas()
        self.canvas.resize(200, 200)
        self.canvas.set_frame(np.full((32, 32), 100, np.uint8))
        self.title, self.lbl = QLabel("t"), QLabel("F 1")


host = SimpleNamespace(panes=[CP(), CP(), CP()], _grid_pos=[(0, 0), (0, 1), (0, 2)],
                       _qimage_to_rgb=cw.CoregWindow._qimage_to_rgb,
                       _draw_band_text=cw.CoregWindow._draw_band_text)
img = cw.CoregWindow._compose_coreg_frame(host, 480, 2)
check("MP4 follows the layout (1×3 → one row of 3)",
      img.shape[0] == 480 and img.shape[1] > 3 * 480, str(img.shape))

# Mac CPR pane Move: mapping functions honour the pan (source run on a stand-in).
SRC = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(encoding="utf-8")
tree = ast.parse(SRC)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
           and any(isinstance(f, ast.FunctionDef) and f.name == "_cpr_view_pan"
                   for f in n.body))
fns = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
ns = {}
for name in ("_cpr_view_pan", "_cpr_view_scale", "_cpr_section_scale",
             "_cpr_offset_to_screen", "_cpr_screen_to_offset"):
    exec(compile(ast.Module([fns[name]], []), "<src>", "exec"), ns)


class Canvas:
    def width(self):
        return 500

    def height(self):
        return 500


s = SimpleNamespace(_cpr={"half": 25.0, "view_pan": (5.0, -2.0)},
                    pane={"A": SimpleNamespace(canvas=Canvas())})
for name in ("_cpr_view_pan", "_cpr_view_scale", "_cpr_section_scale",
             "_cpr_offset_to_screen", "_cpr_screen_to_offset"):
    setattr(s, name, ns[name].__get__(s))
check("panned: the view centre shows the panned point",
      s._cpr_offset_to_screen(5.0, -2.0) == (250.0, 250.0))
back = s._cpr_screen_to_offset(*s._cpr_offset_to_screen(1.0, 3.0))
check("screen ↔ section mapping round-trips with a pan",
      abs(back[0] - 1.0) < 1e-9 and abs(back[1] - 3.0) < 1e-9)
drag = ast.get_source_segment(SRC, fns["_drag"])
check("Move on the CPR pane pans the section view",
      'elif t == "MOVE" and which == "A" and self._cpr is not None:' in drag
      and "view_pan" in drag)
check("the CPR camera follows the pan",
      "self._cpr_view_pan()" in ast.get_source_segment(SRC, fns["_config_cpr_cam"]))

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
