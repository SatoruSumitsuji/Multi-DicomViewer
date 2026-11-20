# -*- coding: utf-8 -*-
"""Mac Territory VR (user 2026-11-20): during the Territory review only the
territory-coloured voxels are drawn. The crop is the whole Epi region but the
territory covers the compact myocardium only, so uncoloured voxels inside it
(trabeculae / PMT / low-HU spots) used to show as a mottled tissue colour over
the territory colours.

The pure CPU ray-caster _vr_render_image is lifted from ct_viewer_pygfx.py with
ast (pygfx / wgpu need not be installed) and run on a tiny synthetic volume.

    python tools/test_mac_territory_vr_only.py
"""
import ast
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtGui import QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


src = (ROOT / "multi_dicomviewer" / "viewers" / "ct_viewer_pygfx.py").read_text(
    encoding="utf-8")
tree = ast.parse(src)
keep = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "_vr_render_image":
        keep.append(node)
    elif isinstance(node, ast.Assign):
        names = [t_.id for t_ in node.targets if isinstance(t_, ast.Name)]
        names += [e.id for t_ in node.targets if isinstance(t_, ast.Tuple)
                  for e in t_.elts if isinstance(e, ast.Name)]
        if any(n.startswith(("_HU_", "_VR_OP_", "_VR_COL_")) for n in names):
            keep.append(node)
ns = {"np": np, "QImage": QImage}
exec(compile(ast.Module(keep, []), "ct_viewer_pygfx.py", "exec"), ns)
render = ns["_vr_render_image"]
check("ray-caster lifted", callable(render))

# 40³ mm volume: a slab of myocardium-like HU (150) across the view; its left
# half carries territory label 1 (red), the right half no label.
n = 40
vol = np.full((n, n, n), -1000.0, np.float32)
vol[10:30, 5:35, 5:35] = 150.0
tvol = np.zeros((n, n, n), np.float32)
tvol[10:30, 5:35, 5:20] = 1.0


def params(terr_only):
    return {"vol": vol, "dims": (1.0, 1.0, 1.0),
            "ctr": np.array([20.0, 20.0, 20.0]),
            "right": np.array([1.0, 0.0, 0.0]),
            "up": np.array([0.0, -1.0, 0.0]),
            "back": np.array([0.0, 0.0, -1.0]),
            "halfW": 20.0, "halfH": 20.0, "pw": 120, "ph": 120,
            "lod": False, "ih_drag": 150, "mode": "dvr",
            "win": 800.0, "lvl": 200.0, "mask": None,
            "tvol": tvol, "tcols": [(1.0, 0.0, 0.0, 0.45)],
            "terr_only": terr_only}


def px(img, x, y):
    c = img.pixelColor(x, y)
    return c.red(), c.green(), c.blue()


img0, _ = render(params(False))
img1, _ = render(params(True))
W, H = img1.width(), img1.height()
lab_px = (int(W * 0.30), H // 2)        # over the labelled half
unl_px = (int(W * 0.70), H // 2)        # over the unlabelled half
r1, g1, b1 = px(img1, *lab_px)
check("territory voxels still drawn in their colour (red)",
      r1 > 60 and g1 < r1 // 3 and b1 < r1 // 3, (r1, g1, b1))
check("before: the uncoloured tissue was drawn (tissue colour)",
      sum(px(img0, *unl_px)) > 60, px(img0, *unl_px))
check("Territory review: the uncoloured tissue is not drawn (black)",
      sum(px(img1, *unl_px)) < 10, px(img1, *unl_px))
src_p = src[src.index("    def _vr_params(self, key, lod):"):][:1600]
check("terr_only = the Territory review (_territory_display)",
      '"terr_only": bool(getattr(self, "_territory_display", False))' in src_p)

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
