"""MV+Apex / AoV Auto: after the last click, 「AutoTraceを開始しますか？」
Yes → the AutoTrace runs; No → the Auto stops; Cancel (戻る) / Esc → the last
point is clicked again (MV+Apex keeps the MV click). Fake host, no real dialog
(QMessageBox.exec is patched), no detection (the run methods are stubs).
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_FAKE_HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = _FAKE_HOME
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:                                   # noqa: BLE001
    pass

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from multi_dicomviewer.viewers.valve_auto_mixin import ValveAutoMixin  # noqa: E402

app = QApplication.instance() or QApplication([])
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


class Host(ValveAutoMixin):
    def __init__(self, which):
        self._lv_valve_edit = which
        self._measures = {"A": [], "B": []}
        self._meas_seq = 0
        self.runs, self.prompts, self.answer = [], [], "yes"

    def window(self):
        return None

    def _lv_auto_mark(self, btn, on):
        pass

    def _redraw_meas(self, k):
        pass

    def _world3d_to_out(self, k, q):
        return (q[0], q[1])

    def _lv_pick_repaint(self):
        pass

    def _lv_pick_marker_clear_later(self):
        pass

    def _lvv_prompt(self, s):
        self.prompts.append(s)

    def _lv_auto_confirm(self, which):
        return self.answer

    def _lv_run_aov_auto(self, P):
        self.runs.append(("aov", P))
        self._lv_valve_auto_seed = None

    def _lv_run_mv_apex_auto(self, mv_p, P):
        self.runs.append(("mv", mv_p, P))
        self._lv_valve_auto_seed = None


def pick_markers(h):
    return [m["color"] for m in h._measures["A"] if m.get("_lv_pick")]


MV, AP = np.array([10.0, 0, 0]), np.array([10.0, 60, 0])
# ---- MV+Apex
for ans, want_run, want_seed in (("yes", True, None), ("no", False, None),
                                  ("back", False, "apex")):
    h = Host("mitral")
    h._lv_valve_auto_seed = ("mitral", "mv", None)
    h.answer = ans
    h._lv_pick_point(MV)
    h._lv_pick_point(AP)
    seed = h._lv_valve_auto_seed
    check(f"MV+Apex {ans}: AutoTrace {'runs' if want_run else 'does not run'}",
          bool(h.runs) == want_run, f"{h.runs!r:.60}")
    check(f"MV+Apex {ans}: state", (seed[1] if seed else None) == want_seed,
          f"{seed!r:.60}")
    if ans == "back":
        check("MV+Apex back: MV click kept", np.allclose(seed[2], MV))
        check("MV+Apex back: only the MV marker left",
              pick_markers(h) == [Host._LV_PICK_COLOR], f"{pick_markers(h)}")
        h.answer = "yes"
        h._lv_pick_point(AP + 5)
        check("MV+Apex back → new apex click → runs with it",
              h.runs and np.allclose(h.runs[0][2], AP + 5)
              and np.allclose(h.runs[0][1], MV))
    if ans == "no":
        check("MV+Apex no: markers removed", pick_markers(h) == [])
# ---- AoV
C = np.array([0.0, 0, 50])
for ans, want_run, want_seed in (("yes", True, None), ("no", False, None),
                                  ("back", False, "aov")):
    h = Host("aortic")
    h._lv_valve_auto_seed = ("aortic", "aov", None)
    h.answer = ans
    h._lv_pick_point(C)
    seed = h._lv_valve_auto_seed
    check(f"AoV {ans}: AutoTrace {'runs' if want_run else 'does not run'}",
          bool(h.runs) == want_run)
    check(f"AoV {ans}: state", (seed[1] if seed else None) == want_seed,
          f"{seed!r:.60}")
    if ans != "yes":
        check(f"AoV {ans}: marker removed", pick_markers(h) == [])

# ---- Ctrl+Z after the MV click → the MV click is dropped (not a view undo);
#      Ctrl+Y puts it back; outside that state both fall through (False).
h = Host("mitral")
h._lv_valve_auto_seed = ("mitral", "mv", None)
check("Ctrl+Z before the MV click → not consumed", h._lv_pick_undo() is False)
h._lv_pick_point(MV)
check("Ctrl+Z after the MV click → consumed", h._lv_pick_undo() is True)
check("… back to the MV click, marker gone",
      h._lv_valve_auto_seed[1] == "mv" and pick_markers(h) == [])
check("Ctrl+Y → MV click restored", h._lv_pick_redo() is True
      and h._lv_valve_auto_seed[1] == "apex"
      and np.allclose(h._lv_valve_auto_seed[2], MV)
      and pick_markers(h) == [Host._LV_PICK_COLOR])
check("Ctrl+Y again → not consumed", h._lv_pick_redo() is False)
h._lv_pick_undo()
h._lv_pick_point(MV + 3)                       # a new MV click ends the redo
check("new MV click → Ctrl+Y not consumed", h._lv_pick_redo() is False
      and np.allclose(h._lv_valve_auto_seed[2], MV + 3))
h.answer = "yes"
h._lv_pick_point(AP)
check("then apex + Yes → runs with the new MV click",
      h.runs and np.allclose(h.runs[0][1], MV + 3))
check("after the run Ctrl+Z → not consumed (normal undo)",
      h._lv_pick_undo() is False)
h = Host("aortic")
h._lv_valve_auto_seed = ("aortic", "aov", None)
check("AoV pick: Ctrl+Z not consumed", h._lv_pick_undo() is False)

# ---- Ctrl+Z / Ctrl+Y of MV+Apex Auto, AoV Auto, MV Set, AoV Set, Apex Set.
# The undo stack + _undo_last/_redo_last are the viewers' REAL code (lifted
# from the source with ast — no VTK / pygfx DLL has to load).
import ast  # noqa: E402
import textwrap  # noqa: E402

_VDIR = Path(__file__).resolve().parents[1] / "multi_dicomviewer" / "viewers"
_UNDO = ("_undo_clear", "_undo_record", "_undo_last", "_redo_last")


def _lift(fname, names):
    src = (_VDIR / fname).read_text(encoding="utf-8")
    ns = {"np": np}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name in names \
                and node.name not in ns:
            exec(compile(textwrap.dedent(ast.get_source_segment(src, node)),
                         fname, "exec"), ns)
    assert all(n in ns for n in names), fname
    return {n: ns[n] for n in names}


def _set_src_ok(fname):
    """Each Set handler snapshots before and records after."""
    src = (_VDIR / fname).read_text(encoding="utf-8")
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name in (
                "_lv_valve_confirm", "_lv_apex_set"):
            seg = ast.get_source_segment(src, node)
            out.append("_lv_vset_snap()" in seg and "_lv_vset_record(before)" in seg)
    return len(out) == 2 and all(out)


class SetHost(Host):
    _UNDO_MAX = 50

    def __init__(self, fns):
        super().__init__("mitral")
        for n, f in fns.items():
            setattr(self, n, f.__get__(self))
        self._draft = None
        self._lv_valves = {"mitral": None, "aortic": None}
        self._lv_valve_shown = {"mitral": True, "aortic": True}
        self._lv_valve_saved_sig = {"mitral": None, "aortic": None}
        self._lv_apex, self._lv_apex_shown = None, True
        self.changed = []
        self._undo_clear()

    def _lv_on_valve_changed(self, w):
        self.changed.append(w)

    def _lv_update_valve_buttons(self):
        pass

    def _lv_update_submode_ui(self):
        pass

    def _lv_apex_marker_draw(self):
        for k in ("A", "B"):
            self._measures[k] = [m for m in self._measures[k] if not m.get("_lv_apex")]
            if self._lv_apex is not None:
                self._measures[k].append({"_lv_apex": True,
                                          "pts3d": [tuple(self._lv_apex)]})

    # stand-ins for the Set operations (same state changes as the real ones)
    def set_valve(self, w, c, r):
        before = self._lv_vset_snap()
        self._lv_valves[w] = (np.asarray(c, float), np.array([0, 0, 1.0]), r)
        for k in ("A", "B"):
            self._measures[k] = [m for m in self._measures[k]
                                 if m.get("_lv_valve") != w]
            self._measures[k].append({"_lv_valve": w, "r": r})
        self._lv_vset_record(before)

    def set_apex(self, P):
        before = self._lv_vset_snap()
        self._lv_apex = np.asarray(P, float)
        self._lv_apex_marker_draw()
        self._lv_vset_record(before)


def state(h):
    v = h._lv_valves
    rings = sorted((m["_lv_valve"], m.get("r")) for m in h._measures["A"]
                   if m.get("_lv_valve"))
    return (None if v["mitral"] is None else v["mitral"][2],
            None if v["aortic"] is None else v["aortic"][2],
            None if h._lv_apex is None else float(h._lv_apex[1]),
            tuple(rings),
            sum(1 for m in h._measures["A"] if m.get("_lv_apex")))


for fname in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    nm = "VTK" if fname == "ct_viewer.py" else "pygfx"
    check(f"{nm}: MV/AoV Set + Apex Set record an undo step", _set_src_ok(fname))
    h = SetHost(_lift(fname, _UNDO))
    s0 = state(h)
    h.set_valve("mitral", [0, 0, 0], 15.0)          # MV Set / MV+Apex Auto
    s1 = state(h)
    h.set_apex([0, 80.0, 0])                        # Apex Set
    s2 = state(h)
    h.set_valve("aortic", [20, 0, 0], 11.0)         # AoV Set / AoV Auto
    s3 = state(h)
    h.set_valve("mitral", [1, 0, 0], 16.0)          # MV re-Set
    s4 = state(h)
    h.set_apex([0, 80.0, 0])                        # same apex → no step
    seq = []
    for _ in range(4):
        h._undo_last()
        seq.append(state(h))
    check(f"{nm}: Ctrl+Z ×4 steps back through every Set",
          seq == [s3, s2, s1, s0], f"{seq}")
    check(f"{nm}: Ctrl+Z → MV ring/apex marker removed", s0 == (None, None, None, (), 0))
    h._undo_last()
    check(f"{nm}: Ctrl+Z at the start → nothing", state(h) == s0)
    seq = []
    for _ in range(4):
        h._redo_last()
        seq.append(state(h))
    check(f"{nm}: Ctrl+Y ×4 re-applies them", seq == [s1, s2, s3, s4], f"{seq}")
    check(f"{nm}: valve caches refreshed on MV/AoV undo/redo",
          "mitral" in h.changed and "aortic" in h.changed)
    # MV+Apex Auto (real mixin code, detection stubbed): MV + apex = ONE step.
    h2 = SetHost(_lift(fname, _UNDO))

    class _R:
        ok, center, normal, radius = True, np.zeros(3), np.array([0, 0, 1.0]), 14.0

    class _A:
        apex, wall_mm, message = np.array([0, 75.0, 0]), 7.0, ""
    h2._vol, h2._dims = np.zeros((2, 2, 2)), (1, 1, 1)
    h2._lv_auto_running = lambda b: __import__("contextlib").nullcontext()
    h2._lv_ve_auto_btn = None
    h2._run_busy = lambda *a: (_R(), _A())
    h2._lv_apply_valve_data = lambda w, d: h2._lv_valves.__setitem__(
        w, (np.asarray(d["c"]), np.asarray(d["n"]), d["r"]))
    h2._lv_valve_auto_contour = lambda w, r: None
    ValveAutoMixin._lv_run_mv_apex_auto(h2, MV, AP)
    check(f"{nm}: MV+Apex Auto set MV + apex",
          h2._lv_valves["mitral"] is not None and h2._lv_apex is not None)
    h2._undo_last()
    check(f"{nm}: one Ctrl+Z undoes MV + apex of the Auto",
          h2._lv_valves["mitral"] is None and h2._lv_apex is None)
    h2._redo_last()
    check(f"{nm}: Ctrl+Y brings both back",
          h2._lv_valves["mitral"] is not None and h2._lv_apex is not None)

# ---- the real dialog: buttons → answers (exec patched, nothing pops up)
h = Host("mitral")
for label, want in (("Yes", "yes"), ("No", "no"), ("Cancel", "back"), (None, "back")):
    def fake_exec(box, label=label):
        for b in box.buttons():
            if label and b.text().startswith(label):
                b.click()
                return 0
        return 0                                  # closed / Esc: no click
    QMessageBox.exec = fake_exec
    got = ValveAutoMixin._lv_auto_confirm(h, "mitral")
    check(f"dialog {label or 'closed'} → {want}", got == want, got)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
