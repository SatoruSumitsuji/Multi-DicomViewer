"""MV / AoV redraw (ValveAutoMixin): a right-click on the MV / AoV line (the
detected outline or the ring) → 「Ellipseで描き直す」/「Polygonで描き直す」. A
finished Polygon sets the plane at once; an Ellipse waits for its handles, then
its right-click 「このEllipseで…面を設定」. The plane is computed like Auto's:
centre = the area centroid, normal = the drawn pane's, radius = the equal-area
circle. One Ctrl+Z step; Esc cancels; the shape is never a Measure-History
figure. Both viewers route the right-click / the commit through the mixin.

The mixin runs on a fake viewer (no VTK / pygfx); QMenu.exec is patched.
"""
import math
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_home = tempfile.mkdtemp(prefix="mdv_home_")       # never the real settings
os.environ["HOME"] = os.environ["USERPROFILE"] = _home
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMenu, QPushButton, QWidget  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.core.measure_geom import (  # noqa: E402
    ellipse_from_major, ellipse_outline, smooth_closed)
from multi_dicomviewer.viewers.valve_auto_mixin import ValveAutoMixin  # noqa: E402
from multi_dicomviewer import i18n  # noqa: E402

i18n.set_language("ja")                 # menu texts = the Japanese keys

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


class V(ValveAutoMixin, QWidget):
    """Pane A = the z = 10 plane (u = x, v = y); output coords = mm = screen."""

    def __init__(self):
        super().__init__()
        self._measures = {"A": [], "B": []}
        self._draft = None
        self._meas_seq = 0
        self._meas_on = False
        self._meas_type = None
        self._meas_btn = QPushButton()
        self._meas_btn.setCheckable(True)
        self._lv_valves = {"mitral": None, "aortic": None}
        self._lv_valve_shown = {}
        self._lv_valve_saved_sig = {}
        self._lv_apex = None
        self._lv_apex_shown = True
        self._frame = {"A": (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                             np.array([0, 0, 1.0])),
                       "B": (np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                             np.array([0, -1.0, 0]))}
        self._pc = {"A": np.array([0, 0, 10.0]), "B": np.zeros(3)}
        self.prompts, self.changed, self.undo, self.views = [], [], [], []

    # --- viewer API the mixin uses
    def _world_to_qt(self, key, wx, wy):
        return float(wx), float(wy)

    def _outline(self, m):
        if m["type"] == "polygon":
            return smooth_closed(m["pts"])
        return ellipse_outline(m["pts"])

    def _axes_for(self, key):
        return self._frame[key]

    def _out_to_world3d(self, key, wx, wy):
        u, v, _n = self._frame[key]
        return self._pc[key] + wx * u + wy * v

    def _world3d_to_out(self, key, P):
        u, v, _n = self._frame[key]
        d = np.asarray(P, float) - self._pc[key]
        return float(d @ u), float(d @ v)

    def _pick_handle(self, key, sx, sy):
        for mi, m in enumerate(self._measures[key]):
            if m.get("_lv_valve"):
                continue
            for vi, q in enumerate(m["pts"]):
                if math.hypot(q[0] - sx, q[1] - sy) < 4:
                    return mi, vi
        return None

    def _set_measure_type(self, key):
        self._meas_type = key
        self._draft = None

    def _toggle_measure(self):
        self._meas_on = self._meas_btn.isChecked()
        if not self._meas_on:
            self._meas_type = None

    def _redraw_meas(self, k):
        pass

    def _lvv_prompt(self, s):
        self.prompts.append(s)

    def _lv_view_mv_perpendicular(self, reset_cam=True, which="mitral"):
        self.views.append(which)

    def _lv_valve_show_from_geom(self, which):
        c, n, r = self._lv_valves[which]
        for k in ("A", "B"):
            self._measures[k] = [m for m in self._measures[k]
                                 if m.get("_lv_valve") != which]
            th = np.linspace(0, 2 * np.pi, 33)[:-1]
            p3 = [np.asarray(c) + r * np.array([math.cos(a), math.sin(a), 0])
                  for a in th]
            self._meas_seq += 1
            self._measures[k].append({
                "id": self._meas_seq, "type": "polygon", "_lv_valve": which,
                "pts": [self._world3d_to_out(k, q) for q in p3],
                "pts3d": p3, "color": "#4dd0ff"})

    def _lv_update_valve_buttons(self):
        pass

    def _lv_update_submode_ui(self):
        pass

    def _lv_on_valve_changed(self, which):
        self.changed.append(which)

    def _lv_apex_marker_draw(self):
        pass

    def _undo_record(self, undo, redo):
        self.undo.append((undo, redo))

    # the viewers' _commit_draft hook, reduced
    def commit(self, pane, typ, pts):
        self._meas_seq += 1
        rec = {"id": self._meas_seq, "type": typ, "pts": pts}
        if self._lv_valve_redraw_commit(pane, rec):
            return "redraw"
        self._measures[pane].append(rec)
        return "measure"


picks = []
def _exec(self, *a, **k):
    want = picks.pop(0)
    return next((x for x in self.actions() if x.text() == want), None)


QMenu.exec = _exec

v = V()
v._lv_valves["mitral"] = (np.array([0, 0, 10.0]), np.array([0, 0.3, 1.0]), 12.0)
v._lv_valve_show_from_geom("mitral")
# the detected (magenta) outline next to the ring
v._lv_valve_auto_contour("mitral", type("R", (), {
    "points": np.array([[14 * math.cos(a), 14 * math.sin(a), 10.0]
                        for a in np.linspace(0, 6.2, 20)]),
    "center": np.array([0, 0, 10.0]), "normal": np.array([0, 0, 1.0])})())
n_valve = len([m for m in v._measures["A"] if m.get("_lv_valve")])

check("right-click on the ring → ('line', mitral)",
      v._lv_valve_hit("A", 12.0, 0.0) == ("line", "mitral"))
check("right-click on the detected outline → ('line', mitral)",
      v._lv_valve_hit("A", 0.0, 14.0) == ("line", "mitral"))
check("right-click away from the lines → None",
      v._lv_valve_hit("A", 40.0, 40.0) is None)
v._lv_valve_shown["mitral"] = False
for m in v._measures["A"]:
    m["hidden"] = True
check("hidden lines → no menu", v._lv_valve_hit("A", 12.0, 0.0) is None)
for m in v._measures["A"]:
    m["hidden"] = False
v._draft = {"type": "polygon", "pts": [(0, 0)]}
check("while a shape is drawn → None (the right-click finishes it)",
      v._lv_valve_hit("A", 12.0, 0.0) is None)
v._draft = None

# ---- Polygon: the plane at once
picks[:] = ["MVをPolygonで描き直す"]
v._lv_valve_menu("A", ("line", "mitral"))
check("Polygon redraw armed: Measure on, Polygon type, pane A face-on",
      v._meas_on and v._meas_type == "polygon" and v.views == ["mitral"])
sq = [(5.0, 5.0), (25.0, 5.0), (25.0, 25.0), (5.0, 25.0)]   # centred (15, 15)
check("an Ellipse while Polygon is armed is a plain measure",
      v.commit("A", "ellipse", ellipse_from_major((0, 0), (4, 0))) == "measure")
v._measures["A"].pop()
res = v.commit("A", "polygon", sq)
c, n, r = v._lv_valves["mitral"]
area = abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1)
               in zip(smooth_closed(sq), smooth_closed(sq)[1:] + smooth_closed(sq)[:1]))) / 2
check("Polygon consumed (no measure, no history)", res == "redraw"
      and not any(m.get("type") == "polygon" and not m.get("_lv_valve")
                  for m in v._measures["A"]))
check("centre = the area centroid on pane A (3-D)",
      np.allclose(c, [15, 15, 10], atol=0.3), str(c))
check("normal = pane A's normal (the old tilt replaced)",
      np.allclose(n, [0, 0, 1]))
check("radius = the equal-area circle", abs(r - math.sqrt(area / math.pi)) < 1e-6,
      f"{r:.2f}")
check("unsaved, caches told, one undo step",
      v._lv_valve_saved_sig.get("mitral") is None and v.changed == ["mitral"]
      and len(v.undo) == 1)
outl = [m for m in v._measures["A"] if m.get("_lv_valve_auto")]
check("the drawn outline replaced the detected one (same magenta tag)",
      len(outl) == 1 and np.allclose(np.mean(outl[0]["pts3d"], 0)[:2], [15, 15],
                                     atol=1.5))
check("Measure back off (as before the redraw), redraw disarmed",
      not v._meas_on and v._lv_valve_redraw is None)
v.undo[0][0]()
check("Ctrl+Z → the previous plane and outline",
      np.allclose(v._lv_valves["mitral"][2], 12.0)
      and len([m for m in v._measures["A"] if m.get("_lv_valve")]) == n_valve)

# ---- Ellipse: adjust, then 「このEllipseで…面を設定」
v.changed.clear()
v._meas_btn.setChecked(True)
v._toggle_measure()
v._set_measure_type("line")
picks[:] = ["MVをEllipseで描き直す"]
v._lv_valve_menu("A", ("line", "mitral"))
check("Ellipse redraw armed (Measure was on with Line)",
      v._meas_type == "ellipse" and v._lv_valve_redraw["meas_on"]
      and v._lv_valve_redraw["type"] == "line")
e1 = ellipse_from_major((-10.0, 0.0), (10.0, 0.0))
check("a finished Ellipse waits (kept, not set yet)",
      v.commit("A", "ellipse", e1) == "redraw"
      and v._lv_valve_shape() is not None and not v.changed)
e2 = ellipse_from_major((-6.0, 2.0), (14.0, 2.0))
v.commit("A", "ellipse", e2)
shapes = [m for m in v._measures["A"] if m.get("_lv_vredraw")]
check("a new Ellipse replaces the waiting one", len(shapes) == 1
      and shapes[0]["pts"] == e2)
sh = shapes[0]
sh["pts"] = [tuple(p) for p in e2]
sh["pts"][2] = (4.0, 2.0 + 8.0)          # minor-axis handle dragged to 8 mm
sh["pts"][3] = (4.0, 2.0 - 8.0)
check("right-click on the waiting Ellipse → ('shape', None)",
      v._lv_valve_hit("A", 14.0, 2.0) == ("shape", None))
picks[:] = ["このEllipseでMV面を設定"]
v._lv_valve_menu("A", ("shape", None))
c, n, r = v._lv_valves["mitral"]
check("Ellipse set: centre = its centre, radius = sqrt(a·b)",
      np.allclose(c, [4, 2, 10], atol=0.2) and abs(r - math.sqrt(10 * 8)) < 0.3,
      f"{c} {r:.2f}")
check("the waiting Ellipse is gone; Measure back to Line",
      not any(m.get("_lv_vredraw") for m in v._measures["A"])
      and v._meas_on and v._meas_type == "line" and v._lv_valve_redraw is None)

# ---- Esc cancels (and drops a waiting Ellipse); AoV the same
before = v._lv_valves["mitral"]
v._lv_valves["aortic"] = (np.array([30, 0, 10.0]), np.array([0, 0, 1.0]), 10.0)
picks[:] = ["AoVをEllipseで描き直す"]
v._lv_valve_menu("A", ("line", "aortic"))
v.commit("A", "ellipse", ellipse_from_major((20.0, 0.0), (40.0, 0.0)))
check("Esc consumed", v._lv_pick_escape())
check("Esc: nothing set, the waiting Ellipse removed",
      v._lv_valves["mitral"] is before and np.allclose(v._lv_valves["aortic"][0],
                                                       [30, 0, 10])
      and not any(m.get("_lv_vredraw") for m in v._measures["A"])
      and v._lv_valve_redraw is None and v._meas_type == "line")
check("Esc without a redraw → not consumed", not v._lv_pick_escape())
picks[:] = ["AoVをPolygonで描き直す"]
v._lv_valve_menu("A", ("line", "aortic"))
v.commit("A", "polygon", [(26.0, -4.0), (34.0, -4.0), (34.0, 4.0), (26.0, 4.0)])
check("AoV redraw by Polygon", np.allclose(v._lv_valves["aortic"][0], [30, 0, 10],
                                           atol=0.2) and v.views[-1] == "aortic")
picks[:] = ["MVをPolygonで描き直す"]
v._lv_valve_menu("A", ("line", "mitral"))
v.commit("A", "polygon", [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)])
check("a tiny shape → cancelled, the plane kept",
      v._lv_valve_redraw is None and v._lv_valves["mitral"][2] == r)

# ---- the viewers route through the mixin
for fn, n_circle in (("ct_viewer.py", 2), ("ct_viewer_pygfx.py", 3)):
    s = (ROOT / "multi_dicomviewer" / "viewers" / fn).read_text(encoding="utf-8")
    check(f"{fn}: right-click → _lv_valve_hit / _lv_valve_menu",
          "_lv_valve_hit(" in s and "_lv_valve_menu(" in s)
    check(f"{fn}: _commit_draft hands the shape to the redraw",
          "if self._lv_valve_redraw_commit(d[\"pane\"]," in s)
    check(f"{fn}: a redraw Polygon is not an LV border",
          "redraw = self._lv_valve_redraw_wants(d[\"type\"])" in s)
    check(f"{fn}: no forced 正円 during a redraw",
          len(re.findall(r"_lv_valve_redraw is None\)", s)) == n_circle)
    check(f"{fn}: face-on view for either valve",
          'mv = self._lv_valves.get(which)' in s)

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
