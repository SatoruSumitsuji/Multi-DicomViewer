"""Epi volume: the MV plane is moved toward the atrium so no drawn Epi point
is cut off (LVModel.epi_mv_plane). Endo keeps the plain MV plane.

Synthetic: an ellipsoidal Epi (apex at z=10, base around z=90) traced on 4
long-axis planes, its base drawn up to z=94 while the MV plane sits at z=90.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multi_dicomviewer.core.lv_measure import LVModel  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


C0 = np.array([60.0, 60.0, 0.0])
apex = C0 + [0, 0, 10.0]
m = LVModel(n_planes=4)
m.set_axis_from_frame(apex, np.array([0, 0, 1.0]), np.array([1.0, 0, 0]),
                      which="epi")
m.set_apex_point("epi", apex)
ax = m.epi_axis
A, R = 84.0, 28.0                       # half-length along z, equator radius
for phi in (0.0, 45.0, 90.0, 135.0):
    e = ax.meridian_dir(phi)
    pts = []
    for zz in np.linspace(94.0, 10.0, 40):          # wall +e: base → apex
        r = R * np.sqrt(max(0.0, 1 - ((zz - 10.0 - A) / A) ** 2))
        pts.append(C0 + [0, 0, zz] + r * e)
    for zz in np.linspace(10.0, 94.0, 40)[1:]:      # apex → base, wall −e
        r = R * np.sqrt(max(0.0, 1 - ((zz - 10.0 - A) / A) ** 2))
        pts.append(C0 + [0, 0, zz] - r * e)
    m.set_long_axis_contour(phi, np.array(pts), which="epi")
m.build()

mv = (C0 + [0, 0, 90.0], np.array([0, 0, 1.0]), 15.0)
mv2, d = m.epi_mv_plane(mv)
check("shift = protrusion (4 mm)", abs(d - 4.0) < 1e-6, f"d={d:.3f}")
check("moved toward the atrium (+z)",
      abs(float(mv2[0][2]) - 94.0) < 1e-6, f"c={np.round(mv2[0], 2)}")
check("normal / radius kept", np.allclose(mv2[1], mv[1]) and mv2[2] == 15.0)

# Flipped normal sign → same plane
mvf = (mv[0], -mv[1], 15.0)
mvf2, df = m.epi_mv_plane(mvf)
check("normal sign independent", abs(df - 4.0) < 1e-6
      and abs(float(mvf2[0][2]) - 94.0) < 1e-6)

# Nothing protrudes → unchanged
mv_hi = (C0 + [0, 0, 96.0], np.array([0, 0, 1.0]), 15.0)
mvh, dh = m.epi_mv_plane(mv_hi)
check("no protrusion → unchanged", dh == 0.0 and mvh is mv_hi)

# Along the apex ↔ MV-centre line (angle kept): apex point 16 mm off the
# normal through the centre → the centre slides on that line to the 94 mm level.
ap_off = C0 + [16.0, 0, 10.0]
mvl, dl = m.epi_mv_plane(mv, toward=ap_off)
line = mv[0] - ap_off
cr = np.cross(mvl[0] - ap_off, line)
check("tilted line: plane reaches the drawn base (z = 94)",
      abs(float(mvl[0][2]) - 94.0) < 1e-6, f"c={np.round(mvl[0], 2)}")
check("tilted line: centre stays on the apex–MV line",
      float(np.linalg.norm(cr)) / float(np.linalg.norm(line)) < 1e-6)
check("tilted line: normal kept, shift = centre move",
      np.allclose(mvl[1], mv[1]) and
      abs(dl - float(np.linalg.norm(mvl[0] - mv[0]))) < 1e-6, f"{dl:.2f} mm")

# Both ways (apical=True, all planes traced): an MV plane ABOVE the drawn
# base (z = 99) comes down to the myocardium's basal end (z = 94); atrial-only
# leaves it.
mv_up = (C0 + [0, 0, 99.0], np.array([0, 0, 1.0]), 15.0)
check("plane above the drawn base, atrial-only → unchanged",
      m.epi_mv_plane(mv_up)[1] == 0.0)
mvd, dd = m.epi_mv_plane(mv_up, apical=True)
check("apical=True: down to the drawn base (z = 94), shift < 0",
      abs(float(mvd[0][2]) - 94.0) < 1e-6 and abs(dd + 5.0) < 1e-6, f"{dd:.2f}")
# The automatic extension ends are not the user's drawing: skipped, the level
# comes from the drawn points (z 94 → the next drawn point ≈ 91.9 when the
# z = 94 ends of every plane are flagged as extensions).
skip = {round(float(ph), 4): (True, True) for ph in m.epi_planes}
mvs, ds = m.epi_mv_plane(mv_up, apical=True, skip=skip)
check("extension ends skipped: level from the drawn points only",
      float(mvs[0][2]) < 93.0, f"z={float(mvs[0][2]):.2f}")

# ---- Viewer (ValveAutoMixin): the COMMON MV plane itself is replaced (all
# analyses); the automatic extensions are stripped first (and re-made by the
# viewer's valve-changed hook); ONE undo step; >8 mm asks; Endo leaves it.
import copy  # noqa: E402
from types import SimpleNamespace  # noqa: E402
import PyQt6.QtWidgets as _qw  # noqa: E402
from multi_dicomviewer.core.lv_measure import mv_terminal_point  # noqa: E402
from multi_dicomviewer.viewers.valve_auto_mixin import ValveAutoMixin  # noqa: E402


class _V(ValveAutoMixin):
    def __init__(self, model, mv0, apex_pt):
        self._lv = {"model": model, "phase": "contour"}
        self._lv_valves = {"mitral": mv0, "aortic": None}
        self._lv_apex = apex_pt
        self._lv_valve_saved_sig = {"mitral": "x"}
        self._lv_valve_shown = {}
        self._measures = {"A": [], "B": []}
        self.applied, self.undo = [], []

    def window(self):
        return None

    def _undo_record(self, u, r):
        self.undo.append((u, r))

    def _lv_vset_snap(self):
        return {"mv": self._lv_valves["mitral"]}

    def _lv_vset_restore(self, sn):
        self._lv_valves["mitral"] = sn["mv"]
        self._undo_record(None, None)              # must be muted

    def _lv_geom_snap(self):
        return {k: np.array(v) for k, v in self._lv["model"].epi_planes.items()}

    def _lv_geom_restore(self, sn):
        for k, v in sn.items():
            self._lv["model"].set_long_axis_contour(k, v, which="epi")

    def _lv_rebuild_measures(self):
        pass

    def _lv_apply_valve_data(self, which, data):
        self.applied.append(which)
        self._lv_valves[which] = (np.asarray(data["c"]), np.asarray(data["n"]),
                                  data["r"])
        self._undo_record(None, None)              # nested record: muted


def _with_ext(model, mv0):
    """every Epi plane gets the automatic terminals on *mv0* at both ends."""
    mm = copy.deepcopy(model)
    ax2 = mm.epi_axis
    c, n = np.asarray(mv0[0]), np.asarray(mv0[1])
    for ph, arr in list(mm.epi_planes.items()):
        P = np.asarray(arr)
        b0 = mv_terminal_point(ax2, ph, c, n, P[0])
        bl = mv_terminal_point(ax2, ph, c, n, P[-1])
        mm.set_long_axis_contour(ph, np.vstack([b0, P, bl]), which="epi")
    return mm


v = _V(copy.deepcopy(m), mv, ap_off)
sh = v._lv_epi_mv_adjust("epi")
check("viewer: common MV plane moved (all analyses)",
      v.applied == ["mitral"] and abs(float(v._lv_valves["mitral"][0][2]) - 94.0)
      < 1e-6 and abs(sh - dl) < 1e-6, f"{sh:.2f} mm")
check("viewer: ONE undo step (nested records muted), plane marked unsaved",
      len(v.undo) == 1 and v.undo[0][0] is not None
      and v._lv_valve_saved_sig["mitral"] is None, f"{len(v.undo)}")
check("viewer: again → nothing more to move", v._lv_epi_mv_adjust("epi") == 0.0
      and v.applied == ["mitral"])
v.undo[0][0]()
check("viewer: Ctrl+Z → the plane back", np.allclose(v._lv_valves["mitral"][0],
                                                      mv[0]) and len(v.undo) == 1)
v2 = _V(copy.deepcopy(m), mv, ap_off)
check("viewer: Endo pass leaves the MV plane alone",
      v2._lv_epi_mv_adjust("endo") == 0.0 and not v2.applied)

# Extensions on a plane ABOVE the drawn base (z = 99): the old atrial-only rule
# saw them "on the plane" and never moved; now they are recognised, skipped,
# stripped, and the plane comes down to the myocardium end (z = 94) on Calc Vol
# (apical=True) — not during tracing (apical=False).
mx = _with_ext(m, mv_up)
v3 = _V(mx, mv_up, apex)
ends = v3._lv_base_ext_ends("epi", mv_up)
check("extensions recognised by shape (both ends, every plane)",
      len(ends) == 4 and all(e == (True, True) for e in ends.values()))
check("while tracing (atrial only): no move", v3._lv_epi_mv_adjust("epi") == 0.0)
sh3 = v3._lv_epi_mv_adjust("epi", apical=True)
n_pts = [len(np.asarray(a)) for a in mx.epi_planes.values()]
check("Calc Vol (apical): plane down to the myocardium end, extensions stripped",
      abs(float(v3._lv_valves["mitral"][0][2]) - 94.0) < 1e-6
      and abs(sh3 + 5.0) < 1e-6 and n_pts == [79] * 4, f"{sh3:.2f}, {n_pts}")

# > 8 mm asks first; a declined target is not asked again.
asked = []
_orig_q = _qw.QMessageBox.question
_qw.QMessageBox.question = staticmethod(
    lambda *a, **k: (asked.append(1), _qw.QMessageBox.StandardButton.No)[1])
try:
    mv_lo = (C0 + [0, 0, 80.0], np.array([0, 0, 1.0]), 15.0)
    v4 = _V(copy.deepcopy(m), mv_lo, apex)
    r1 = v4._lv_epi_mv_adjust("epi")
    r2 = v4._lv_epi_mv_adjust("epi")
    check("> 8 mm: asked, No → not moved, not asked again",
          r1 == 0.0 and r2 == 0.0 and len(asked) == 1 and not v4.applied)
    _qw.QMessageBox.question = staticmethod(
        lambda *a, **k: _qw.QMessageBox.StandardButton.Yes)
    v5 = _V(copy.deepcopy(m), mv_lo, apex)
    check("> 8 mm: Yes → moved 14 mm", abs(v5._lv_epi_mv_adjust("epi") - 14.0)
          < 1e-6)
finally:
    _qw.QMessageBox.question = _orig_q

# The viewers' base snap (VTK + pygfx, lifted from source): on the plane the
# myocardium end touches (MV moved to it) NO extension is drawn; elsewhere the
# extension is made as before and recorded as automatic.
import ast  # noqa: E402


def _lift(path, name):
    src = Path(path).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"np": np}
            exec(compile(ast.Module([node], []), path, "exec"), ns)
            return ns[name]
    raise LookupError(name)


root = Path(__file__).resolve().parents[1] / "multi_dicomviewer" / "viewers"
for fn in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    snap = _lift(root / fn, "_lv_snap_base_to_mv")
    for z, want in ((94.0, False), (96.0, True)):
        vv = SimpleNamespace(_LV_EXT_TOL_MM=0.3, _lv_geom_snap=lambda: None,
                             _lv_rebuild_measures=lambda: None,
                             _lv_show_plane=lambda: None,
                             _lv_record_geom=lambda b: None)
        vv._lv = {"model": copy.deepcopy(m)}
        vv._lv_valves = {"mitral": (C0 + [0, 0, z], np.array([0, 0, 1.0]), 15.0)}
        moved = snap(vv, "epi")
        check(f"{fn} base snap, MV at z={z:g}: extension "
              f"{'made + recorded' if want else 'NOT drawn (end on the plane)'}",
              bool(moved) == want
              and (len(vv._lv.get("base_ext", ())) == 8) == want,   # 4 planes × 2 ends
              f"moved={moved} ext={len(vv._lv.get('base_ext', ()))}")

# Volume: the Epi volume now reaches the drawn base (= the 94 mm cut)
dims, shape = (1.0, 1.0, 1.0), (110, 120, 120)
_c, _b, v_auto, _ = m.inside_mask_volumes(dims, shape, "epi", mv, None)
_c, _b, v_94, _ = m.inside_mask_volumes(dims, shape, "epi", mv2, None)
check("Epi volume uses the moved plane", abs(v_auto - v_94) < 1e-6,
      f"{v_auto:.1f} vs {v_94:.1f} mL")
surf = m._full_surface("epi")
_c, _b, v_90, _ = surf.inside_mask_volumes(dims, shape, mv, None, apex)
check("larger than the plain-MV cut", v_auto > v_90 + 1.0,
      f"{v_auto:.1f} > {v_90:.1f} mL")

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
