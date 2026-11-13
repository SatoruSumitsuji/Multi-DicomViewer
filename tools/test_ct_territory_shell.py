"""Shell side of the in-place Territory (main_window): 「Territory計算」 takes the
in-session LV (a FullLv built in memory, no save) else the docked .FullLv.json,
computes in the Coronary Tree dock and enters the review on the SAME viewer;
the Territoryモード button re-shows a computed result; leaving turns 領域色 off.

The methods are lifted from main_window.py via ast onto a fake shell (no VTK),
with a fake dock + viewer; QMessageBox is patched (no real dialogs).
"""
import ast
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_home = tempfile.mkdtemp(prefix="mdv_home_")       # never the real settings
os.environ["HOME"] = os.environ["USERPROFILE"] = _home
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.i18n import t  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


NAMES = ("ct_mode_state", "ct_territory_compute", "ct_territory_show",
         "_ct_territory_open", "ct_territory_left", "_full_lv_to_files")
src = (ROOT / "multi_dicomviewer" / "ui" / "main_window.py").read_text(
    encoding="utf-8")
cls = next(n for n in ast.walk(ast.parse(src))
           if isinstance(n, ast.ClassDef) and any(
               isinstance(m, ast.FunctionDef) and m.name == "ct_territory_compute"
               for m in n.body))
nodes = [m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name in NAMES]
check("all shell methods found", {n.name for n in nodes} == set(NAMES))
ns = {"t": t}
exec(compile(ast.Module(nodes, []), "main_window.py", "exec"), ns)


class Dock:
    def __init__(self):
        self.tree = True
        self.full = None
        self._territory = None
        self._full_lv_path = ""
        self._last_path = "C:/x/a.corotree.json"
        self.loads, self.colors, self.used = [], [], []

    def tree_vessels_for_series(self, uid):
        return self.tree

    def full_lv_for_series(self, uid):
        return self.full

    def full_lv_path_for_series(self, uid):
        return self._full_lv_path if self.full else ""

    def corotree_path_for_series(self, uid):
        return self._last_path

    def territory_for_series(self, uid):
        return self._territory is not None

    def uid_for_series(self, uid):
        return uid

    def _use_uid(self, u):
        self.used.append(u)

    def load_full_lv(self, full, src_path=""):
        self.loads.append((full, src_path))
        self.full = full
        if src_path:
            self._full_lv_path = src_path
        self._territory = object()
        return True

    def set_territory_color(self, on):
        self.colors.append(bool(on))


class Pane:
    def shown_series_uid(self):
        return "1.2.3"


class Viewer:
    def __init__(self):
        self.mem = None
        self._full_lv_saved = False
        self._full_lv_saved_path = ""
        self.entered = []
        self._lv_apex = 1
        self.apex = []

    def _ct_full_lv_build(self):
        return self.mem

    def _ct_territory_enter(self, full):
        self.entered.append(full)
        return True

    def _lv_apply_apex_data(self, d):
        self.apex.append(d)


class Shell:
    def __init__(self):
        self._corotree_win = Dock()
        self.pane = Pane()
        self.tree_shown = []

    def _pane_of_viewer(self, v):
        return self.pane

    def _set_active_pane(self, p):
        pass

    def _corotree_dock_hidden(self):
        return False

    def ct_tree_show(self, v, on, raise_tab=False):
        self.tree_shown.append(on)


for n in NAMES:
    f = ns[n]
    setattr(Shell, n, staticmethod(f) if n == "_full_lv_to_files" else f)

msgs = []
_orig = QMessageBox.information
QMessageBox.information = staticmethod(lambda *a, **k: msgs.append(a[2]))
try:
    sh, v = Shell(), Viewer()
    sh._corotree_win.tree = False
    check("no tree → message, nothing computed",
          sh.ct_territory_compute(v) is False and len(msgs) == 1
          and not sh._corotree_win.loads)

    sh, v = Shell(), Viewer()
    v.mem = {"epi": {}, "bld": {}, "mem": 1}
    ok = sh.ct_territory_compute(v)
    d = sh._corotree_win
    check("in-session LV (unsaved) → FullLv built in memory, no file name",
          ok and d.loads == [(v.mem, "")] and d._full_lv_path == ""
          and v.entered == [v.mem])
    check("review: tree ON + 領域色 ON", sh.tree_shown == [True]
          and d.colors == [True])
    st = sh.ct_mode_state(v)
    check("state: FullLv from memory (no file name) + Territory result",
          st.get("full") == "" and st.get("full_mem") and st.get("terr"),
          str(st))

    sh, v = Shell(), Viewer()
    v.mem = {"m": 1}
    v._full_lv_saved, v._full_lv_saved_path = True, "C:/x/a.FullLv.json"
    sh.ct_territory_compute(v)
    check("in-session LV saved → its .FullLv.json named",
          sh._corotree_win.loads[-1][1] == "C:/x/a.FullLv.json"
          and sh.ct_mode_state(v).get("full") == "a.FullLv.json")

    sh, v = Shell(), Viewer()
    sh._corotree_win.full = {"docked": 1}
    sh._corotree_win._full_lv_path = "C:/x/b.FullLv.json"
    sh.ct_territory_compute(v)
    check("no in-session LV → the docked FullLv (file) is used",
          sh._corotree_win.loads == [({"docked": 1}, "C:/x/b.FullLv.json")]
          and v.entered == [{"docked": 1}])

    msgs.clear()
    sh, v = Shell(), Viewer()
    check("no LV at all → message", sh.ct_territory_compute(v) is False
          and len(msgs) == 1 and not v.entered)

    sh, v = Shell(), Viewer()
    check("Territoryモード without a result → nothing",
          sh.ct_territory_show(v) is False and not v.entered)
    sh._corotree_win.full = {"docked": 1}
    sh._corotree_win._territory = object()
    check("Territoryモード with a result → re-shown, no recompute",
          sh.ct_territory_show(v) and v.entered == [{"docked": 1}]
          and not sh._corotree_win.loads)

    sh, v = Shell(), Viewer()
    sh._corotree_win.full = {"bld": {"apex": [1, 2, 3]}, "epi": {}}
    v._lv_apex = None
    sh.ct_territory_left(v)
    check("leaving: 領域色 OFF; apex put back when the LV came from FullLv",
          sh._corotree_win.colors == [False]
          and v.apex == [{"apex": [1, 2, 3]}])
finally:
    QMessageBox.information = _orig

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
