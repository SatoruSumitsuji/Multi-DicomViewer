# -*- coding: utf-8 -*-
"""XA / IVUS transport (user 2026-11-22):
- the Play button shows its shortcut keys: 「▶ Play (D)」 / 「⏸ Pause (S)」;
- Reverse ON colours the Reverse button and keeps the ORIGINAL frame numbers
  in the "current/total" readout: seek bar at the left end = total, at the
  right end = 1. Synthetic frames only.

    python tools/test_xa_play_keys_reverse.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.viewers.xa_viewer import XAViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


class _Plane:
    """Minimal stand-in for an XAPlane (volume + reverse + total_frames)."""

    def __init__(self, n):
        self.volume = np.stack([np.full((8, 8), i, np.float32)
                                for i in range(n)])
        self.total_frames = n

    def reverse(self):
        self.volume = self.volume[::-1].copy()


v = XAViewer()
check("Play label shows D", v.play_btn.text() == "▶ Play (D)", v.play_btn.text())
v.play_btn.blockSignals(True)
v.play_btn.setChecked(True)
v.play_btn.blockSignals(False)
v.retranslate_ui()
check("Pause label shows S", v.play_btn.text() == "⏸ Pause (S)",
      v.play_btn.text())
v.play_btn.blockSignals(True)
v.play_btn.setChecked(False)
v.play_btn.blockSignals(False)

src = (ROOT / "multi_dicomviewer" / "viewers" / "xa_viewer.py").read_text(
    encoding="utf-8")
check("no plain 「▶ Play」 / 「⏸ Pause」 label left",
      '"▶ Play"' not in src and '"⏸ Pause"' not in src)
check("Reverse button has a checked background",
      ":checked" in v.reverse_btn.styleSheet()
      and "background" in v.reverse_btn.styleSheet())

# readout with Reverse: drive the label code with a fake 10-frame plane
n = 10
v._planes = [_Plane(n)]
v._frame_of = lambda p: p.volume[min(v._frame, p.volume.shape[0] - 1)]
v._plane_angles = lambda p: None


def label_at(frame):
    v._frame = frame
    v._render()
    return v.frame_lbl.text()


v._reversed = False
check("forward: left end = 1", label_at(0) == f"1/{n}", label_at(0))
check("forward: right end = n", label_at(n - 1) == f"{n}/{n}")
v._reversed = True
check("reversed: left end = n (original number)", label_at(0) == f"{n}/{n}",
      label_at(0))
check("reversed: right end = 1", label_at(n - 1) == f"1/{n}", label_at(n - 1))
check("reversed: frame 3 from the left = original n-3",
      label_at(3) == f"{n - 3}/{n}", label_at(3))

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
os._exit(1 if fails else 0)
