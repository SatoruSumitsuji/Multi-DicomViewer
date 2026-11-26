# -*- coding: utf-8 -*-
"""Auto-Track stops at an existing CPR line (user 2026-11-26): tracing a
branch proximally ran past its parent line. Now when the lumen centre 3 mm
ahead comes within 3 mm of an existing tree line (hidden ones too) the
trace stops and its end point is put ON that line (the tree then connects
it — ≤ 5 mm). A branch origin lying on the traced vessel (the zone near a
line's proximal end) and the line's own saved copy do not stop it.
Synthetic tube volume (made-up data), fake host / shell, no real dialog.

    python tools/test_autotrack_stop_existing.py
"""
import copy
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.information = staticmethod(lambda *a, **k: None)
import multi_dicomviewer.core.cpr_autoextend as CAE  # noqa: E402
from multi_dicomviewer.core.centerline import CenterLine  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.ui.coronary_tree_window import (  # noqa: E402
    CoronaryTreeWindow)
from multi_dicomviewer.viewers.cpr_mixin import CPRMixin  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


# a straight 400 HU tube along z (x = y = 30 mm), 0.5 mm voxels
sp = 0.5
nz, ny, nx = 170, 120, 120
z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32) * sp
vol = np.full((nz, ny, nx), 40.0, np.float32)
vol[((x - 30) ** 2 + (y - 30) ** 2 <= 1.6 ** 2) & (z >= 4) & (z <= 80)] = 400
vol += np.random.default_rng(0).normal(0, 15, vol.shape).astype(np.float32)
D3 = (sp, sp, sp)
CTRL = [[30.0, 30.0, 8.0], [30.0, 30.0, 14.0]]


def line(a, b, n=81):
    return np.linspace(np.asarray(a, float), np.asarray(b, float), n)


class Shell:
    def __init__(self, lines):
        self.lines = lines
        self.out = None

    def coronary_stop_lines(self, v):
        return self.lines

    def coronary_auto_extend_report(self, v, out, name="", w=None):
        self.out = out


class Host(QWidget, CPRMixin):
    _AUTOTRACK_STEP_S = 0.0

    def __init__(self, lines):
        QWidget.__init__(self)
        self._vol, self._dims, self._mode = vol, D3, "3D"
        self._coronary_mode = True
        self._cpr = None
        self._vr_on = {"A": False, "B": False}
        self._measures = {"A": [], "B": [{
            "id": 1, "type": "polyline", "_cpr_src": True,
            "pts3d": copy.deepcopy(CTRL),
            "pts": [(p[0], p[2]) for p in CTRL]}]}
        self.sh = Shell(lines)

    def _ct_mode_shell(self):
        return self.sh

    def _world3d_to_out(self, key, P):
        return (float(P[0]), float(P[2]))

    def _meas_pane_snap(self, pane):
        return copy.deepcopy(self._measures[pane])

    def _meas_record(self, pane, before):
        pass

    def _redraw_meas(self, key):
        pass

    def _refresh(self, *a, **k):
        pass

    def _cpr_ensure_built(self, target=None):
        return True

    def _cpr_register_to_tree(self, save_as=False, rearm=True):
        return True


def run(lines):
    h = Host(lines)
    h._cpr_autotrack("B", 0)
    return np.asarray(h._measures["B"][0]["pts3d"], float), h.sh.out


# no existing line: it runs to the tube's end
P0, o0 = run([])
check("no line: traced on to the end", P0[-1][2] > 70, P0[-1])

# a line crossing the tube at z = 45 (its ends far away)
cross = line([5, 30, 45], [55, 30, 45])
P1, o1 = run([(cross, True, False)])
d_on = float(np.linalg.norm(cross - P1[-1], axis=1).min())
check("stops at the existing line", 40 <= P1[-1][2] <= 46, P1[-1])
check("its end point lies ON that line (connects ≤ 5 mm)", d_on < 0.3, d_on)
check("the result says why",
      o1 is not None and o1["stop"] == "existing"
      and CoronaryTreeWindow.stop_text(o1["stop"])
      == t("既存の CPR 線に到達（終点をその線上に置きました）"),
      (o1 or {}).get("stop"))
check("no 「中心を指定して続ける」 after it", not o1.get("extras"),
      o1.get("extras"))

# a branch whose ORIGIN sits on the tube at z = 45 → passed (not stopped)
br = line([30, 30, 45], [55, 30, 45])
P2, _ = run([(br, True, False)])
check("a branch origin on the path does not stop it", P2[-1][2] > 70, P2[-1])

# the line's own saved copy is ignored
own = np.asarray(CenterLine.from_points(np.asarray(CTRL), step_mm=0.5).points)
P3, _ = run([(own, True, False)])
check("its own saved copy does not stop it", P3[-1][2] > 70, P3[-1])

# the core helper
ls = CAE.LineStops([(cross, True, False)])
check("LineStops: within 3 mm hits, beyond not",
      ls.hit([30, 30, 42.5]) is not None and ls.hit([30, 30, 41.5]) is None)
e = CAE.end_on_line([[0, 0, 0], [0, 0, 2.6], [0, 0, 2.9]], [0, 0, 3.0])
check("end_on_line: near points dropped, the line point last",
      len(e) == 2 and np.allclose(e[-1], [0, 0, 3.0]), e)

# the tree dock gives every line of the CT — hidden ones too
w = CoronaryTreeWindow(None)
w._use_uid("1.2.3")
b = w._bundle()
b["tree"].add_root("lad", "LAD", "LAD", line([0, 0, 0], [0, 0, 40]))
b["tree"].add_vessel("d1", "D1", "branch", line([0, 0, 20], [20, 0, 20]))
b["tree"].vessels["d1"].parent, b["tree"].vessels["d1"].junction = "lad", 40
b["tree"].add_vessel("x", "X", "branch", line([40, 0, 0], [40, 20, 0]))
b["hidden"].add("lad")
got = {round(float(P[-1][0])): (i0, i1)
       for P, i0, i1 in w.stop_lines_for_series("1.2.3")}
check("dock: hidden lines included", len(got) == 3, got)
check("dock: proximal ends ignored (root / branch), unconnected both",
      got.get(0) == (True, False) and got.get(20) == (True, False)
      and got.get(40) == (True, True), got)
check("dock: the edited line left out",
      len(w.stop_lines_for_series("1.2.3", skip_vid="d1")) == 2)

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
