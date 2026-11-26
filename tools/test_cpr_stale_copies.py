"""Pre-edit CPR lines are hidden: a session copy of a .cpr.json (e.g. the line
kept after Save&Exit) is hidden once the same file is resumed / re-saved, and
on フォルダ更新 when its shape no longer matches the file. Show All never
brings it back. Also: Coronary Tree tab buttons are left-aligned."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_dicomviewer.viewers.cpr_mixin import CPRMixin  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


d = tempfile.mkdtemp(prefix="mdv_cpr_")
fp = os.path.join(d, "01.cpr.json")
new_ctrl = [[0.0, 0.0, 0.0], [0.0, 5.0, 10.0], [0.0, 0.0, 20.0]]
json.dump({"ctrl": new_ctrl}, open(fp, "w", encoding="utf-8"))


class V(CPRMixin):
    def __init__(self):
        self._cpr = None
        self.old = {"type": "polyline", "_cpr_saved": fp,
                    "pts3d": [[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]]}
        self.cur = {"type": "polyline", "pts3d": new_ctrl}
        self.other = {"type": "polyline", "_cpr_saved": os.path.join(d, "02.cpr.json"),
                      "pts3d": [[1.0, 0, 0], [1.0, 0, 9]]}
        self._measures = {"A": [], "B": [self.old, self.other, self.cur]}
        self.redraws = 0

    def _cpr_parked_target(self):
        return ("B", 2)

    def _redraw_meas(self, k):
        self.redraws += 1

    def _cpr_vr_active(self):
        return False


v = V()
n = v._cpr_retire_stale_copies(os.path.join(d, ".", "01.cpr.json"), keep=v.cur)
check("resume / save: the older copy of the same file is hidden",
      n == 1 and v.old.get("hidden") and v.old.get("_cpr_stale")
      and "_cpr_saved" not in v.old and not v._cpr_is_cpr_line(v.old))
check("other files' lines and the edited line untouched",
      not v.other.get("hidden") and not v.cur.get("hidden"))

v2 = V()
v2.cur["_cpr_saved"] = fp                 # the re-saved line = the file
n = v2._cpr_retire_outdated()
check("フォルダ更新: a line whose shape differs from its file is hidden",
      v2.old.get("_cpr_stale") and not v2.cur.get("_cpr_stale") and n == 1)

for fn in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    src = (ROOT / "multi_dicomviewer/viewers" / fn).read_text(encoding="utf-8")
    check(f"{fn}: Show All skips outdated CPR copies",
          'if not m.get("_cpr_stale")] + list(self._compares)' in src)
mw = (ROOT / "multi_dicomviewer/ui/main_window.py").read_text(encoding="utf-8")
check("resume (編集) hides the session copies of the vessel's file",
      "v._cpr_retire_stale_copies(" in mw)
cw = (ROOT / "multi_dicomviewer/ui/coronary_tree_window.py").read_text(
    encoding="utf-8")
check("フォルダ更新 calls _cpr_retire_outdated", "_sv._cpr_retire_outdated()" in cw)
check("Coronary Tree tab buttons left-aligned",
      '_BTN_CSS = "padding: 2px 6px; text-align: left;"' in cw
      and cw.count("setStyleSheet(_BTN_CSS)") == 2   # (Territory VR row removed)
      and 'setStyleSheet("padding: 2px 6px;")' not in cw)

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
