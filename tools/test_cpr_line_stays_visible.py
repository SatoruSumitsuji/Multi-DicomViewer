# -*- coding: utf-8 -*-
"""A confirmed CPR line stays drawn (user 2026-11-26): saving while its
short-axis stays open (上書き保存 / Auto-Track with branches) used to run
Hide All Result over the LIVE line too, and the tree overlay left the
edited vessel out until something else re-pushed it — so nothing showed.
Now: hide-all skips the live line, ending a tree-vessel edit re-pushes the
overlay, and the short-axis entry un-hides the line it takes over.
Mixin-level, no real viewer / dialog.

    python tools/test_cpr_line_stays_visible.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_dicomviewer.viewers.cpr_mixin import CPRMixin  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


class _Shell:
    def __init__(self):
        self.n = 0

    def coronary_overlay_refresh(self):
        self.n += 1


class V(CPRMixin):
    def __init__(self):
        self.live = {"type": "polyline", "_cpr_src": True}
        self.other = {"type": "polyline"}
        self._measures = {"A": [], "B": [self.live, self.other]}
        self._cpr = {"src": "B", "src_mi": 0}
        self._coro_edit_vid = ""
        self._cpr_suggested_name = ""
        self.shell = _Shell()
        self.redrawn = []

    def window(self):
        return self.shell

    def _hideall_items(self):
        return [m for m in self._measures["B"] if not m.get("_bulk_hidden")]

    def _redraw_meas(self, k):
        self.redrawn.append(k)


v = V()
v._cpr_hide_all_results()
check("hide-all keeps the line open in the short-axis",
      not v.live.get("_bulk_hidden") and v.other.get("_bulk_hidden"))
v._cpr = None
v._cpr_hide_all_results()
check("with the short-axis left, the done line is hidden as before",
      v.live.get("_bulk_hidden"))

v._coro_edit_vid = "lad"
v._cpr_suggested_name = "LAD"
v._coro_edit_end()
check("ending a vessel edit re-pushes the tree overlay",
      v.shell.n == 1 and v._coro_edit_vid == "" and v._cpr_suggested_name == "")
v._coro_edit_end()
check("no edit → no extra push", v.shell.n == 1)

for f in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    src = (ROOT / "multi_dicomviewer" / "viewers" / f).read_text(encoding="utf-8")
    i = src.index("    def _exit_cpr(")
    check(f + ": _exit_cpr re-pushes the overlay",
          "self._coro_edit_end()" in src[i:i + 1500])
    j = src.index("    def _enter_cpr(")
    check(f + ": _enter_cpr un-hides the line it takes over",
          'self._measures[which][mi].pop("_bulk_hidden", None)'
          in src[j:j + 3500])
mix = (ROOT / "multi_dicomviewer" / "viewers" / "cpr_mixin.py").read_text(
    encoding="utf-8")
check("Next / headless save end the edit through _coro_edit_end",
      mix.count("self._coro_edit_end()") >= 2)

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
