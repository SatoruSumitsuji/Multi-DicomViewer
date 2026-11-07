# -*- coding: utf-8 -*-
"""自動CPR編集点距離: Auto-Trace edit points thinned to the Settings spacing,
the hand-clicked start / via / end always kept.

Run: python tools/test_cpr_ctrl_thin.py
"""
import os
import pathlib
import sys
import tempfile
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
_tmp = tempfile.mkdtemp()
os.environ["HOME"] = _tmp
os.environ["USERPROFILE"] = _tmp

import numpy as np

from multi_dicomviewer.core import settings
from multi_dicomviewer.core.cpr_autotrack import thin_ctrl

fails = []


def chk(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), name, extra)
    if not cond:
        fails.append(name)


# Dense path: a helix, 0.25 mm sampling (like trace_vessel's path_mm).
th = np.linspace(0, 3 * np.pi, 2000)
P = np.column_stack([20 * np.cos(th), 20 * np.sin(th), 4 * th])
seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
s = np.concatenate([[0.0], np.cumsum(seg)])
L = float(s[-1])


def at(a):
    return np.array([np.interp(a, s, P[:, k]) for k in range(3)])


ctrl = np.array([at(a) for a in np.arange(0.0, L, 1.2)] + [P[-1]])  # dense auto
clicks = [P[0] + [0.3, 0, 0], at(0.31 * L) + [0, 0.4, 0],
          at(0.72 * L) + [0.2, 0.2, 0], P[-1] + [0, 0, 0.3]]


def arcs(Q):
    return np.array([s[int(np.argmin(np.linalg.norm(P - q, axis=1)))] for q in Q])


click_arcs = [0.0, *arcs(clicks[1:-1]), L]
prev_n = None
for sp in settings.CPR_CTRL_MM_CHOICES:
    out = thin_ctrl(P, ctrl, clicks, sp)
    a = arcs(out)
    chk(f"{sp} mm: in path order", np.all(np.diff(a) > 0))
    chk(f"{sp} mm: start / end kept", np.allclose(out[0], P[0])
        and np.allclose(out[-1], P[-1]))
    dk = [float(np.min(np.abs(a - c))) for c in click_arcs]
    chk(f"{sp} mm: every clicked point kept", max(dk) < 0.3, f"{max(dk):.2f}")
    gaps = np.diff(a)
    chk(f"{sp} mm: no two points closer than the spacing",
        gaps.min() >= sp - 0.3, f"min {gaps.min():.2f}")
    chk(f"{sp} mm: gaps stay < 2 × spacing + auto step", gaps.max() < 2 * sp + 1.2,
        f"max {gaps.max():.2f}")
    if prev_n is not None:
        chk(f"{sp} mm: fewer points than {sp - 1} mm", len(out) < prev_n,
            f"{len(out)} vs {prev_n}")
    prev_n = len(out)
    print(f"   {sp} mm → {len(out)} points (auto {len(ctrl)}), L {L:.0f} mm")

# Two clicks only (start + end): ends kept, nothing else forced.
out = thin_ctrl(P, ctrl, [P[0], P[-1]], 5)
chk("start+end only: ends kept", np.allclose(out[0], P[0]) and np.allclose(out[-1], P[-1]))

# Sparse auto points (already farther apart than the spacing) are not dropped.
sparse = np.array([at(a) for a in np.arange(0.0, L, 8.0)] + [P[-1]])
out = thin_ctrl(P, sparse, [P[0], P[-1]], 3)
chk("already sparse: kept (only the one next to the end may go)",
    len(out) >= len(sparse) - 1,
    f"{len(out)} vs {len(sparse)}")

# ---- Settings round trip
cp = settings.load_coronary_params()
chk("settings: default 3 mm", cp["cpr_ctrl_mm"] == 3)
settings.save_coronary_params({"cpr_ctrl_mm": 5})
chk("settings: saved 5 mm", settings.load_coronary_params()["cpr_ctrl_mm"] == 5)
settings.save_coronary_params({"cpr_ctrl_mm": 9})
chk("settings: invalid → default", settings.load_coronary_params()["cpr_ctrl_mm"] == 3)
settings.save_coronary_params({"tube_diameter_mm": 2.0, "cpr_ctrl_mm": 2})
settings.save_coronary_params({"tube_diameter_mm": 1.5})
chk("settings: kept when other params saved",
    settings.load_coronary_params()["cpr_ctrl_mm"] == 2)

# ---- CPRMixin._cpr_auto_store applies it (pts3d = clicks before the store)
from multi_dicomviewer.viewers.cpr_mixin import CPRMixin


class H(CPRMixin):
    pass


h = H()
h._measures = {"A": [], "B": []}
settings.save_coronary_params({"cpr_ctrl_mm": 4})
m = {"type": "polyline", "pts3d": [list(map(float, q)) for q in clicks]}
h._measures["B"].append(m)
res = types.SimpleNamespace(path_mm=P, ctrl_mm=ctrl, gaps=[])
h._cpr_auto_store(m, res)
a = arcs(np.asarray(m["pts3d"]))
chk("store: thinned to the setting", np.diff(a).min() >= 4 - 0.3
    and len(m["pts3d"]) < len(ctrl), f"{len(m['pts3d'])} pts")
chk("store: clicks kept", max(float(np.min(np.abs(a - c))) for c in click_arcs) < 0.3)
chk("store: marked auto", m.get("_cpr_auto") is True)

print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
