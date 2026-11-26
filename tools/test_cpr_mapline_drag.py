# -*- coding: utf-8 -*-
"""Drag the right pane's plane line on the short axis (user 2026-11-26):
whenever the light-blue line shows, a left-drag ON it turns it round the
section centre, and the right (long-axis) pane turns with it round the
course; with Measure on (橙) only with Alt. Mixin stub (10 px / mm), no
real viewer / dialog, no patient data.

    python tools/test_cpr_mapline_drag.py
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


O = np.array([10.0, 20.0, 30.0])


class Host(CPRMixin):
    def __init__(self):
        self._cpr = {"src": "B", "src_mi": 0, "half": 20.0}
        self._vr_on = {"A": False, "B": False}
        self._thick = {"A": 0.0, "B": 0.0}
        self._meas_on = False
        # section at O, in-plane u = x, v = y, normal (course) z; the right
        # pane: horizontal = course, normal x → the line is du = 0
        self._frame = {"B": (np.array([0, 0, 1.0]), np.array([0, 1.0, 0]),
                             np.array([1.0, 0, 0]))}
        self._cross_ang = {"A": 0.0, "B": 12.0}
        self._pc = {"B": O + [0.0, 0, 5]}
        self._center = O + [0.0, 0, 5]
        self.refreshed = 0

    def _cpr_frame(self):
        return (O.copy(), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                np.array([0, 0, 1.0]))

    def _axes_for(self, key):
        return self._frame[key]

    def _ortho(self, u, v):
        u = u / np.linalg.norm(u)
        v = v - (v @ u) * u
        v = v / np.linalg.norm(v)
        return (u, v, np.cross(u, v))

    def _cpr_screen_to_offset(self, sx, sy):     # 10 px / mm, centre 100
        return (sx - 100.0) / 10.0, (100.0 - sy) / 10.0

    def _cpr_recenter_map_pane(self, key, P):
        self._pc[key] = np.asarray(P, float).copy()

    def _redraw_meas(self, key):
        pass

    def _refresh(self, *a, **k):
        self.refreshed += 1


h = Host()
pl = h._cpr_map_plane_section_2d()
check("start: the line is du = 0", len(pl) == 1
      and all(abs(q[0]) < 1e-9 for q in pl[0]), pl)
check("a press ON the line (≤ 6 px) is caught",
      h._cpr_mapline_hit(104, 50) and not h._cpr_mapline_hit(110, 50))
h._meas_on = True
check("橙 (Measure on): not without Alt", not h._cpr_mapline_grab(100, 50))
check("橙: with Alt it is grabbed", h._cpr_mapline_grab(100, 50, alt=True))
h._cpr_mapline_end()
h._meas_on = False
check("Measure off: a plain press grabs it", h._cpr_mapline_grab(100, 50))
h._cpr_mapline_move(150, 100)                    # (0, 5) → (5, 0): −90°
h._cpr_mapline_end()
u1, v1, n1 = h._frame["B"]
check("dragged 90°: the right pane's normal is now ±y (line along du)",
      abs(abs(n1[1]) - 1.0) < 1e-6, n1)
check("… its horizontal still the course (+z), so the vessel runs as before",
      np.allclose(u1, [0, 0, 1.0]), u1)
pl = h._cpr_map_plane_section_2d()
check("… the line drawn on the short axis turned with it (dv = 0)",
      len(pl) == 1 and all(abs(q[1]) < 1e-6 for q in pl[0]), pl)
check("… the right pane through / centred on the section centre",
      np.allclose(h._pc["B"], O) and np.allclose(h._center, O)
      and h._cross_ang["B"] == 0.0 and h.refreshed >= 1)
check("released: no longer turning", h._cpr_mapline_prev is None)
h._cpr_mapline_move(100, 50)
check("a move after release does nothing", np.allclose(h._frame["B"][2], n1))
# small steps follow the cursor (45° in 3 moves)
h._cpr_mapline_grab(150, 100)
for a in (15, 30, 45):
    r = np.radians(a)
    h._cpr_mapline_move(100 + 50 * np.cos(r), 100 - 50 * np.sin(r))
h._cpr_mapline_end()
n2 = h._frame["B"][2]
ang = np.degrees(np.arccos(abs(float(n2 @ n1))))
check("a gradual drag turns it by the cursor's angle (45°)",
      abs(ang - 45.0) < 0.5, ang)
h._vr_on["B"] = True
check("VR on the right (no line): nothing to grab",
      not h._cpr_mapline_grab(100, 50))
h._vr_on["B"] = False

for f, keys in (("ct_viewer.py", ("_owner._cpr_mapline_grab(",
                                  "_owner._cpr_mapline_move(",
                                  "_owner._cpr_mapline_end()")),
                ("ct_viewer_pygfx.py", ("self._cpr_mapline_grab(x, y, alt=",
                                        "self._cpr_mapline_move(x, y)",
                                        "self._cpr_mapline_end()"))):
    s_ = (ROOT / "multi_dicomviewer/viewers" / f).read_text("utf-8")
    check(f"{f}: press / drag / release wired", all(k in s_ for k in keys))

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
