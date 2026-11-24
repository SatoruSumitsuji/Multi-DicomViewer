# -*- coding: utf-8 -*-
"""Coronary Tree tab ▸ 役割 ▸ LM / LAD / LCX / RCA asks first whether the
vessel takes the role's name (user 2026-11-25). Yes = renamed like 名前を変更
(its .cpr.json follows); No = the role only; 枝 / a vessel already so named =
no question. Synthetic lines, fake HOME, dialogs patched.

    python tools/test_coro_role_rename.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
asked = []
answer = [QMessageBox.StandardButton.Yes]
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(
    lambda *a, **k: (asked.append(a[2] if len(a) > 2 else ""), answer[0])[1])
from multi_dicomviewer.core.centerline import CenterLine  # noqa: E402
from multi_dicomviewer.ui.coronary_tree_window import CoronaryTreeWindow  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


D = tempfile.mkdtemp(dir=HOME)
UID = "rr-ct"


def line(vid, name, x0):
    ctrl = np.c_[np.full(5, x0), np.linspace(0, 30, 5), np.zeros(5)]
    p = os.path.join(D, f"{name}.cpr.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"format": "MDV-CPR", "version": 1, "type": "cpr",
                   "series": {"series_uid": UID}, "src_dir": "",
                   "ctrl": ctrl.tolist(), "state": {"idx": 0}}, f)
    tree.add_vessel(vid, name, "branch",
                    CenterLine.from_points(ctrl, step_mm=0.5).points,
                    ctrl=ctrl)
    b["paths"][vid] = p


w = CoronaryTreeWindow(None)
w._use_uid(UID)
b = w._bundle()
tree = b["tree"]
line("a", "01", 0.0)
line("b", "02", 20.0)
line("c", "03", 40.0)

answer[0] = QMessageBox.StandardButton.Yes
w._apply_role("a", "LAD")
va = tree.vessels["a"]
check("Yes: role set AND the vessel named after it",
      va.role == "LAD" and w._display_name(va) == "LAD" and len(asked) == 1,
      (va.role, va.name, asked))
check("the question names the old and the new name",
      "01" in asked[-1] and "LAD" in asked[-1], asked[-1])
check("Yes: its .cpr.json follows the name",
      os.path.basename(b["paths"]["a"]).startswith("LAD")
      and os.path.exists(b["paths"]["a"]), b["paths"]["a"])

answer[0] = QMessageBox.StandardButton.No
w._apply_role("b", "RCA")
vb = tree.vessels["b"]
check("No: the role only, the name kept",
      vb.role == "RCA" and w._display_name(vb) == "02"
      and os.path.basename(b["paths"]["b"]) == "02.cpr.json", (vb.name,))

n = len(asked)
w._apply_role("c", "branch")
check("枝: no question", len(asked) == n and tree.vessels["c"].role == "branch")
w._apply_role("a", "LAD")
check("already named after the role: no question", len(asked) == n)
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
os._exit(1 if fails else 0)
