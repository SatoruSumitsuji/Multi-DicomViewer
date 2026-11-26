# -*- coding: utf-8 -*-
"""(user 2026-11-26)
1. The short axis shows the right pane's SLAB: with a thickness > 0 its two
   faces are drawn as lines parallel to the plane line (± half the
   thickness), dotted, 25 % — none when the thickness is 0.
2. 「Auto-Track 終了」 (left-pane right-click, under 「Auto-Track 開始」): while
   「中心を指定して続ける」 waits, it ends Auto-Track — the temporary
   stop-section point is removed (one undo step), the search range and the
   hint go. Resuming Auto-Track or leaving the short axis drops it too.
Mixin stub, no real dialog, no patient data.

    python tools/test_cpr_slab_lines_at_end.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from multi_dicomviewer.viewers.cpr_mixin import CPRMixin  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


class SB:
    def __init__(self):
        self.cleared = 0

    def clearMessage(self):
        self.cleared += 1


class Win:
    def __init__(self):
        self.sb = SB()

    def statusBar(self):
        return self.sb


class Host(CPRMixin):
    def __init__(self):
        self._cpr = {"src": "B", "src_mi": 0, "half": 20.0}
        self._vr_on = {"A": False, "B": False}
        self._thick = {"A": 0.0, "B": 0.0}
        # section at the origin, in-plane u = x, v = y; the right pane's
        # plane: normal x through the origin → the line is x = 0
        self._pc = {"B": np.zeros(3)}
        self._measures = {"B": [{"pts3d": [[0, 0, 0], [0, 0, 5]]}]}
        self.undo = []
        self.win = Win()
        self.refreshed = 0

    def _cpr_frame(self):
        return (np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                np.array([0, 0, 1.0]))

    def _axes_for(self, key):
        return (np.array([0, 1.0, 0]), np.array([0, 0, 1.0]),
                np.array([1.0, 0, 0]))

    def window(self):
        return self.win

    def _meas_pane_snap(self, pane):
        return [dict(m, pts3d=[list(p) for p in m["pts3d"]])
                for m in self._measures[pane]]

    def _meas_record(self, pane, before):
        self.undo.append(before)

    def _cpr_set_ctrl(self, m, which, pts):
        m["pts3d"] = [list(map(float, p)) for p in pts]

    def _cpr_is_src(self, which, mi):
        return False

    def _redraw_meas(self, key):
        pass

    def _refresh(self, *a, **k):
        self.refreshed += 1


# ---- 1. slab lines
h = Host()
pl = h._cpr_map_plane_section_2d()
check("plane line: x = 0", len(pl) == 1
      and all(abs(q[0]) < 1e-9 for q in pl[0]), pl)
check("no slab lines at thickness 0", h._cpr_map_slab_section_2d() == [])
h._thick["B"] = 5.0
sl = h._cpr_map_slab_section_2d()
xs = sorted(round(q[0], 6) for sg in sl for q in sg)
check("slab 5 mm: two lines at x = ∓2.5, parallel to the plane line",
      len(sl) == 2 and xs == [-2.5, -2.5, 2.5, 2.5]
      and all(abs(sg[0][1] - pl[0][0][1]) < 1e-9
              and abs(sg[1][1] - pl[0][1][1]) < 1e-9 for sg in sl), sl)
check("slab lines 25 % (half the former 50 %)",
      CPRMixin._CPR_SLAB_ALPHA == 64)
for f in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    s_ = (ROOT / "multi_dicomviewer/viewers" / f).read_text("utf-8")
    check(f"{f}: slab lines use _CPR_SLAB_ALPHA", "_CPR_SLAB_ALPHA" in s_)
h._vr_on["B"] = True
check("VR on the right: no slab lines", h._cpr_map_slab_section_2d() == [])
h._vr_on["B"] = False

# ---- 2. Auto-Track 終了
keep = [[0.0, 0, 0], [0.0, 0, 5]]
h._measures["B"][0]["pts3d"] = keep + [[0.0, 0, 8]]   # + stop-section point
h._cpr_search_ring = {"q": [0, 0, 8], "r": 3.0}
check("not offered without a waiting 「中心を指定して続ける」",
      not h._cpr_autotrack_end_ok())
h._cpr_at_pending = {"which": "B", "mi": 0, "keep": keep}
check("offered while it waits", h._cpr_autotrack_end_ok())
check("終了 runs", h._cpr_autotrack_end() is True)
check("the temporary point is removed",
      h._measures["B"][0]["pts3d"] == keep, h._measures["B"][0]["pts3d"])
check("… as one undo step (holding the point)", len(h.undo) == 1
      and len(h.undo[0][0]["pts3d"]) == 3)
check("search range and hint gone, no longer offered",
      h._cpr_search_ring is None and h.win.sb.cleared == 1
      and not h._cpr_autotrack_end_ok())
# the line edited since → only the waiting state ends, the line is kept
h._measures["B"][0]["pts3d"] = [[0.0, 0, 0], [1.0, 0, 5], [0.0, 0, 8]]
h._cpr_at_pending = {"which": "B", "mi": 0, "keep": keep}
h._cpr_autotrack_end()
check("edited since: the line is left as it is",
      len(h._measures["B"][0]["pts3d"]) == 3 and len(h.undo) == 1)

# ---- source: menu + resets in both viewers
src = (ROOT / "multi_dicomviewer/viewers/cpr_mixin.py").read_text("utf-8")
check("resuming Auto-Track drops it",
      "self._cpr_at_pending = None      # (resumed" in src)
for f in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    s = (ROOT / "multi_dicomviewer/viewers" / f).read_text("utf-8")
    i_at = s.find('t("Auto-Track 開始")')
    i_end = s.find('t("Auto-Track 終了")')
    check(f"{f}: 「Auto-Track 終了」 right under 「Auto-Track 開始」",
          0 < i_at < i_end < i_at + 400
          and "self._cpr_autotrack_end()" in s)
    check(f"{f}: leaving the short axis drops it",
          "self._cpr_at_pending = None                   # nor" in s)
    check(f"{f}: slab lines drawn", "_cpr_map_slab_section_2d()" in s)

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
