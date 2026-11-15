"""CPR line setting (2026-11-15):
- a left click on a CPR line does not select it while setting CPR lines, and a
  Draw click on another CPR line's point starts the new line there;
- 選択を解除 (Deselect) on the line being edited leaves the edit at once (the
  tree line comes back via the overlay push; unsaved changes are confirmed);
- the 📁 Next-folder button is gone;
- short-axis Delete Point lands on the nearest remaining edit point;
- auto-trace edit points are never nearer than the spacing (also in space)."""
import sys
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from multi_dicomviewer.core.cpr_autotrack import thin_ctrl  # noqa: E402
from multi_dicomviewer.viewers.cpr_mixin import CPRMixin  # noqa: E402

app = QApplication.instance() or QApplication([])
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


class Shell:
    def __init__(self):
        self.refreshed = 0

    def coronary_overlay_refresh(self):
        self.refreshed += 1


class V(CPRMixin):
    def __init__(self, vid):
        ctrl = [[0.0, 0, 0], [0, 0, 10.0], [0, 0, 20.0]]
        self.m = {"type": "polyline", "_cpr_src": True,
                  "pts3d": [list(q) for q in ctrl],
                  "_cpr_orig_ctrl": [list(q) for q in ctrl]}
        self._measures = {"A": [], "B": [self.m]}
        self._cpr = {"src": "B", "src_mi": 0}
        self._coro_edit_vid = vid
        self.shell = Shell()
        self.log = []
        self.sel = []
        self._coro_select_cb = self.sel.append

    def window(self):
        return self.shell

    def _view_snapshot(self):
        return {}

    def _exit_cpr(self, keep_view=False):
        self.log.append("exit")
        self._cpr = None
        self._coro_edit_vid = ""

    def _coronary_draw(self):
        self.log.append("draw")
        if self._cpr is not None:
            self._cpr = None             # park + exit

    def _redraw_meas(self, k):
        pass

    def _cpr_vr_active(self):
        return False


v = V("v1")
ok = v._cpr_edit_deselect()
check("tree vessel: Deselect leaves the edit, drops the editing copy",
      ok and v._cpr is None and v._measures["B"] == [] and v.log == ["exit", "draw"])
check("tree line pushed back at once + selection cleared",
      v.shell.refreshed == 1 and v.sel == [""])

v = V("v1")
v.m["pts3d"][1] = [0, 3.0, 10.0]           # edited, not saved
with mock.patch.object(QMessageBox, "question",
                       return_value=QMessageBox.StandardButton.No) as q:
    ok = v._cpr_edit_deselect()
check("unsaved edit: asked, No keeps the edit",
      q.called and not ok and v._cpr is not None and v._measures["B"] == [v.m])

v = V("")
ok = v._cpr_edit_deselect()
check("line not in the tree: kept (Draw parks it)",
      ok and v._measures["B"] == [v.m] and v.log == ["draw"])


class Menu:
    pass


v = V("v1")
check("Deselect offered only on the line being edited",
      v._cpr_edit_deselect_add(None, "A", 0) is None)

# Delete Point → nearest edit point
class CL:
    def __init__(self):
        self.points = np.array([[0, 0, z] for z in range(0, 41)], float)
        self.n = len(self.points)


class D(CPRMixin):
    def __init__(self):
        self._cpr = {"cl": CL(), "idx": 0}
        self.calls = []

    def _cpr_ctrl_indices(self):
        return [0, 18, 40]

    def _cpr_sync_bar(self):
        self.calls.append("bar")

    def _cpr_center_map_on_current(self):
        self.calls.append("map")

    def _refresh(self):
        pass


d = D()
d._cpr_goto_nearest_ctrl([0, 0, 25.0])
check("Delete Point: section + map go to the nearest edit point",
      d._cpr["idx"] == 18 and "map" in d.calls)

# thin_ctrl: no auto point nearer than the spacing in space
t = np.linspace(0, 1, 400)
path = np.stack([np.zeros_like(t), 2.0 * np.sin(t * 40), t * 20.0], 1)  # meander
ctrl = path[::8]
out = thin_ctrl(path, ctrl, [path[0], path[-1]], 3.0)
dd = np.linalg.norm(np.diff(out, axis=0), axis=1)
check("auto edit points ≥ 3 mm apart in space", bool(np.all(dd >= 3.0 - 1e-6)),
      f"min={dd.min():.2f}")

for fn in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    src = (ROOT / "multi_dicomviewer/viewers" / fn).read_text(encoding="utf-8")
    check(f"{fn}: no click-select while setting CPR lines",
          'and not getattr(o, "_coronary_mode", False)):' in src
          or 'and not getattr(self, "_coronary_mode", False)):' in src)
    check(f"{fn}: Draw click ignores other CPR lines' points",
          "or self._cpr_is_cpr_line(self._measures[which][hit[0]]))):" in src)
    check(f"{fn}: Deselect in both line menus",
          src.count("self._cpr_edit_deselect_add(menu, which, mi)") == 2
          and src.count("self._cpr_edit_deselect()") == 2)
    check(f"{fn}: 📁 Next-folder button removed",
          "self._cpr_next_dir_btn = None" in src
          and 'FitButton("📁 "' not in src)
mw = (ROOT / "multi_dicomviewer/ui/main_window.py").read_text(encoding="utf-8")
check("resume remembers the vessel shape", '_sm["_cpr_orig_ctrl"]' in mw)
print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
