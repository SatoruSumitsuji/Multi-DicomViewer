"""CPR line right-click on the Coronary Tree lines + Coronary Tree tab tweaks.

* Every tree CPR line (overlay line, or a CPR line of this session that is not
  the one being edited) gets the SAME short menu — 編集 (Edit) / 名前を表示 —
  with Measure on or off; 編集 opens the full edit menu (a tree vessel is
  resumed first). The line being edited and ordinary measures keep their menus.
* The edit menu: no Auto-Trace / lumen snap; no Hide on CPR lines; Save split
  into 上書き保存 (Save) and 名前を付けて保存….
* Results hidden by Next (Hide All Result) are not picked any more.
* Coronary Tree tab: Vessel column = DetailName; FixName header "+" copies the
  selected FixName to every vessel (after a confirm).
Fake hosts, menus and dialogs patched (nothing pops up). The viewer methods are
lifted from the viewer SOURCE with ast (no VTK / pygfx DLL has to load).
"""
import ast
import os
import sys
import tempfile
import textwrap
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

import PyQt6.QtWidgets as QtW  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.viewers.cpr_mixin import CPRMixin  # noqa: E402

app = QApplication.instance() or QApplication([])
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


_VDIR = Path(__file__).resolve().parents[1] / "multi_dicomviewer" / "viewers"
_SRC = {f: (_VDIR / f).read_text(encoding="utf-8")
        for f in ("ct_viewer.py", "ct_viewer_pygfx.py")}


def _func_src(fname, name):
    for node in ast.walk(ast.parse(_SRC[fname])):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return textwrap.dedent(ast.get_source_segment(_SRC[fname], node))
    raise KeyError(name)


# ---------------------------------------------------------------- fake menu
class FakeAct:
    def __init__(self, text):
        self.text = text
        self.enabled = True

    def setEnabled(self, b):
        self.enabled = bool(b)

    def setVisible(self, _b):
        pass

    def setCheckable(self, _b):
        pass

    def setChecked(self, _b):
        pass

    def setIcon(self, _i):
        pass

    def isChecked(self):
        return False


class FakeMenu:
    pick = None                     # label (prefix) to choose
    seen = []

    def __init__(self, *_a):
        self.items = []

    def addAction(self, text):
        a = FakeAct(text)
        self.items.append(a)
        return a

    def addMenu(self, text):
        self.items.append(FakeAct(text))      # submenu title (never chosen)
        return FakeMenu()

    def addSeparator(self):
        pass

    def exec(self, *_a):
        FakeMenu.seen.append([a.text for a in self.items])
        for a in self.items:
            if FakeMenu.pick and a.text == FakeMenu.pick:
                return a
        return None


class _Canvas:
    def mapToGlobal(self, p):
        return p

    def mapFromGlobal(self, p):
        return p


class _Pane:
    canvas = _Canvas()


# ---------------------------------------------------------------- host
def _poly(**kw):
    m = {"type": "polyline", "pts": [(0, 0), (5, 0), (9, 0)],
         "pts3d": [(0, 0, 0), (5, 0, 0), (9, 0, 0)], "id": 1}
    m.update(kw)
    return m


class Host(CPRMixin):
    def __init__(self):
        self._mode = "3D"
        self._draft = None
        self._cpr = None
        self._cpr_parked = None
        self._lv = None
        self._coronary_mode = True
        self._vr_on = {}
        self._coro_names = set()
        self._measures = {"A": [], "B": []}
        self.pane = {"A": _Pane(), "B": _Pane()}
        self.hit_mi = None          # what _pick_measure returns
        self.hit_vid = None         # what _coronary_pick returns
        self.calls = []
        self._coro_resume_cb = self._resume

    def window(self):
        return None

    def _pick_handle(self, which, sx, sy):
        return None

    def _pick_measure(self, which, sx, sy):
        return self.hit_mi

    def _coronary_pick(self, which, sx, sy):
        return self.hit_vid

    def _redraw_geom(self, k):
        pass

    def _outline_right(self, which, mi, sx, sy):
        self.calls.append(("outline", which, mi))

    def _resume(self, vid):
        # like main_window.coronary_resume_edit → apply_cpr_data: the vessel
        # becomes the live (short-axis source) line on pane B
        self.calls.append(("resume", vid))
        self._measures["B"].append(_poly(_cpr_src=True, id=99))
        self._cpr = {"src": "B", "src_mi": len(self._measures["B"]) - 1}


QtW.QMenu = FakeMenu                                 # mixin imports it inline

h = Host()
check("nothing under the cursor → normal flow", h._cpr_tree_hit("B", 1, 1) is None)
h.hit_vid = "v7"
check("overlay line only → short menu (None, vid)",
      h._cpr_tree_hit("B", 1, 1) == (None, "v7"))
h._measures["B"] = [_poly(_cpr_saved="x.cpr.json")]
h.hit_mi, h.hit_vid = 0, None
check("saved CPR line of this session → short menu (0, None)",
      h._cpr_tree_hit("B", 1, 1) == (0, None))
h._measures["B"] = [_poly()]
check("ordinary polyline → its own menu", h._cpr_tree_hit("B", 1, 1) is None)
h._measures["B"] = [_poly(_cpr_src=True)]
h._cpr = {"src": "B", "src_mi": 0}
check("the line being edited → its own (full) menu",
      h._cpr_tree_hit("B", 1, 1) is None)
h._cpr = None
h._draft = {"pane": "B", "type": "polyline", "pts": []}
check("a trace being drawn → right-click finishes it (normal flow)",
      h._cpr_tree_hit("B", 1, 1) is None)
h._draft = None
h._vr_on = {"B": True}
check("VR pane → its own routing", h._cpr_tree_hit("B", 1, 1) is None)
h._vr_on = {}
h._mode = "2D"
check("2-D → normal flow", h._cpr_tree_hit("B", 1, 1) is None)

# ---- the short menu
h = Host()
FakeMenu.seen, FakeMenu.pick = [], None
h._cpr_tree_menu("B", 3, 4, None, "v7")
items = FakeMenu.seen[-1]
check("short menu = 編集 + 名前を表示 (nothing else)",
      items == [t("編集 (Edit)"), t("名前を表示")], f"{items}")
FakeMenu.pick = t("名前を表示")
h._cpr_tree_menu("B", 3, 4, None, "v7")
check("名前を表示 → name shown", h._coro_names == {"v7"})
FakeMenu.pick = t("編集 (Edit)")
h._cpr_tree_menu("B", 3, 4, None, "v7")
check("編集 on a tree vessel → resumed, then the full menu on the live line",
      h.calls == [("resume", "v7"), ("outline", "B", 0)], f"{h.calls}")
h = Host()
h._measures["A"] = [_poly(_cpr_saved=True)]
FakeMenu.pick = t("編集 (Edit)")
h._cpr_tree_menu("A", 3, 4, 0, None)
check("編集 on a session CPR line → the full menu on it",
      h.calls == [("outline", "A", 0)], f"{h.calls}")

# ---------------------------------------------------------------- full menu
# The REAL _outline_right / _handle_right of both viewers on a fake host.
from multi_dicomviewer.ui.measure_style_menu import (  # noqa: E402
    add_color_submenu, add_transparency_submenu)

for fname in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    nm = "VTK" if fname == "ct_viewer.py" else "pygfx"
    from PyQt6.QtGui import QColor, QIcon, QPixmap
    ns = {"np": np, "t": t, "QMenu": FakeMenu, "QColor": QColor,
          "QIcon": QIcon, "QPixmap": QPixmap, "QPoint": lambda *a: a,
          "QtPoint": lambda *a: a, "add_color_submenu": add_color_submenu,
          "add_transparency_submenu": add_transparency_submenu,
          "QTimer": type("QT", (), {"singleShot": staticmethod(lambda *a: None)})}
    for fn in ("_outline_right", "_handle_right"):
        exec(compile(_func_src(fname, fn), fname, "exec"), ns)

    class VHost(Host):
        _outline_right = ns["_outline_right"]
        _handle_right = ns["_handle_right"]

        def _meas_pane_snap(self, w):
            return []

        def _meas_record(self, w, b):
            pass

        def _redraw_meas(self, w):
            pass

        def _recompute_compares(self, w):
            pass

        def _redraw_compare(self, w):
            pass

        def _reset_pointer_state(self):
            pass

        def _cpr_ensure_built(self, tgt=None):
            self.calls.append(("built", tgt))
            return True

        def _cpr_register_to_tree(self, save_as=False, rearm=True):
            self.calls.append(("save", save_as, rearm))
            return True

    v = VHost()
    v._measures["B"] = [_poly(_cpr_saved=True), _poly(id=2)]
    FakeMenu.seen, FakeMenu.pick = [], None
    v._outline_right("B", 0, 1, 1)
    cpr_items = FakeMenu.seen[-1]
    v._outline_right("B", 1, 1, 1)
    plain_items = FakeMenu.seen[-1]
    v._handle_right("B", (0, 2), 1, 1)
    handle_items = FakeMenu.seen[-1]
    for lbl in ("Add point", "Resume trace", "Change Color",
                "Copy (Ctrl+V to paste)", "Delete"):
        check(f"{nm}: CPR edit menu keeps {lbl}", t(lbl) in cpr_items)
    check(f"{nm}: CPR edit menu keeps Create/ReCalc short-axis",
          t("このCPR線を短軸で再計算 (ReCalc)") in cpr_items
          or t("この線で短軸CPRを作成 (Coronary)") in cpr_items, f"{cpr_items}")
    for lbl in ("Auto-Trace", "Snap trace to lumen", "Auto-snap to lumen", "Hide"):
        check(f"{nm}: CPR edit menu has no {lbl}", t(lbl) not in cpr_items)
    check(f"{nm}: Save split into 上書き保存 (Save) + 名前を付けて保存…",
          t("上書き保存 (Save)") in cpr_items and t("名前を付けて保存…") in cpr_items)
    check(f"{nm}: CPR vertex menu has no Hide", t("Hide") not in handle_items)
    check(f"{nm}: ordinary polyline keeps Hide", t("Hide") in plain_items)
    v.calls = []
    FakeMenu.pick = t("上書き保存 (Save)")
    v._outline_right("B", 0, 1, 1)
    check(f"{nm}: 上書き保存 → save without dialog (Next-style), no re-arm",
          v.calls == [("built", ("B", 0)), ("save", False, False)], f"{v.calls}")
    v.calls = []
    FakeMenu.pick = t("名前を付けて保存…")
    v._outline_right("B", 0, 1, 1)
    check(f"{nm}: 名前を付けて保存… → Save As dialog",
          v.calls == [("built", ("B", 0)), ("save", True, True)], f"{v.calls}")

    # source checks: routing + Next-hidden results not pickable
    src = _SRC[fname]
    i_tree = src.find("_cpr_tree_hit(")
    i_meas = src.find("_measure_right(", src.find("Right-click priority"))
    check(f"{nm}: tree-line menu is checked before the measure menu",
          0 < i_tree < i_meas)
    check(f"{nm}: _pick_measure skips results hidden by Next",
          "_bulk_hidden" in _func_src(fname, "_pick_measure"))
    check(f"{nm}: _pick_handle skips results hidden by Next",
          "_bulk_hidden" in _func_src(fname, "_pick_handle"))
check("VTK overlay right-click uses the short menu",
      "_cpr_tree_menu(" in _func_src("ct_viewer.py", "_coronary_right"))
check("pygfx overlay right-click uses the short menu",
      "_cpr_tree_menu(" in _func_src("ct_viewer_pygfx.py", "_coronary_menu"))
# Mac = Windows: MPR line menu without ここにターゲット設定; the VR pane's vessel
# menu = ここにターゲット設定 / この血管を非表示 (the REAL pygfx methods).
check("pygfx: MPR tree-line menu has no target item",
      "with_target=True" not in _SRC["ct_viewer_pygfx.py"])
ns = {"np": np, "t": t, "QMenu": FakeMenu, "QPoint": lambda *a: a}
for fn in ("_coronary_menu", "_vr_vessel_menu"):
    exec(compile(_func_src("ct_viewer_pygfx.py", fn), "pygfx", "exec"), ns)


class MacHost(Host):
    _coronary_menu = ns["_coronary_menu"]
    _vr_vessel_menu = ns["_vr_vessel_menu"]

    def __init__(self):
        super().__init__()
        self._coro_target_cb = lambda v, i: self.calls.append(("target", v, i))
        self._coro_vessel_hide_cb = lambda v: self.calls.append(("hide", v))

    def _reset_pointer_state(self):
        pass

    def _coronary_pick_sample_vr(self, which, sx, sy):
        return "v3", 17


mh = MacHost()
FakeMenu.seen, FakeMenu.pick = [], None
mh._coronary_menu("B", "v3", 1, 1)
check("pygfx: MPR overlay line → 編集 / 名前を表示 only",
      FakeMenu.seen[-1] == [t("編集 (Edit)"), t("名前を表示")], f"{FakeMenu.seen[-1]}")
mh._vr_on = {"A": True}
mh._coronary_menu("A", "v3", 1, 1)
check("pygfx: VR vessel → ここにターゲット設定 / この血管を非表示 (Windows)",
      FakeMenu.seen[-1] == [t("ここにターゲット設定"), t("この血管を非表示")],
      f"{FakeMenu.seen[-1]}")
FakeMenu.pick = t("ここにターゲット設定")
mh._coronary_menu("A", "v3", 1, 1)
FakeMenu.pick = t("この血管を非表示")
mh._coronary_menu("A", "v3", 1, 1)
check("pygfx: VR target set at the clicked sample / vessel hidden",
      mh.calls == [("target", "v3", 17), ("hide", "v3")], f"{mh.calls}")
check("pygfx: set_coronary_vessel_hide_cb exists (main window wires it)",
      "def set_coronary_vessel_hide_cb" in _SRC["ct_viewer_pygfx.py"])
check("VTK: list-selected vessel drawn orange 1.5x",
      "_CORO_SEL_RGB" in _SRC["ct_viewer.py"]
      and "2.6 * 1.5" in _SRC["ct_viewer.py"])
check("pygfx: list-selected vessel drawn orange 1.5x",
      "rgb = _CORO_SEL_RGB" in _SRC["ct_viewer_pygfx.py"]
      and "2.2 * 1.5 if sel" in _SRC["ct_viewer_pygfx.py"])

# ---------------------------------------------------------------- tree tab
from PyQt6.QtCore import QPoint  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QTreeWidget  # noqa: E402

from multi_dicomviewer.ui import coronary_tree_window as ctw  # noqa: E402


class V:
    def __init__(self, name, det="", fix=""):
        self.name, self.detail_name, self.fix_name = name, det, fix


W = ctw.CoronaryTreeWindow
check("Vessel column = DetailName", W._display_name(V("LAD7@030", "LAD7")) == "LAD7")
check("Vessel column = name until a DetailName is given",
      W._display_name(V("14")) == "14")


class FakeTree:
    def __init__(self):
        self.vessels = {"a": V("1", fix="@030"), "b": V("2", fix=""),
                        "c": V("3", fix="@x")}


class TW:
    def __init__(self, sel):
        self._tree = FakeTree()
        self.sel = sel
        self.populated = 0

    def selected_vid(self):
        return self.sel

    def _populate(self):
        self.populated += 1

    def _refresh_rename_btn(self):
        pass


real_q, real_i = QMessageBox.question, QMessageBox.information
asked = []
QMessageBox.information = staticmethod(lambda *a, **k: asked.append("info"))
QMessageBox.question = staticmethod(
    lambda *a, **k: QMessageBox.StandardButton.Yes)
w = TW("a")
W._fix_name_copy_all(w)
check("+ → every FixName = the selected row's",
      [v.fix_name for v in w._tree.vessels.values()] == ["@030"] * 3
      and w.populated == 1)
QMessageBox.question = staticmethod(
    lambda *a, **k: QMessageBox.StandardButton.No)
w = TW("a")
W._fix_name_copy_all(w)
check("+ → No keeps every FixName",
      [v.fix_name for v in w._tree.vessels.values()] == ["@030", "", "@x"])
w = TW(None)
W._fix_name_copy_all(w)
check("+ with no row selected → a note, nothing changed",
      asked == ["info"] and w.populated == 0)
QMessageBox.question, QMessageBox.information = real_q, real_i

tw = QTreeWidget()
tw.setColumnCount(ctw._N_COLS)
hdr = ctw._XHeader(tw)
tw.setHeader(hdr)
tw.setHeaderLabels([str(i) for i in range(ctw._N_COLS)])
for c in range(ctw._N_COLS):
    tw.setColumnWidth(c, 90)
tw.resize(900, 200)
tw.show()
app.processEvents()
got = []
hdr.plusClicked.connect(got.append)
hdr.toggled.connect(lambda c: got.append(("x", c)))
pr = hdr._plus_rect(ctw._C_FIX)
QTest.mouseClick(hdr.viewport(), QtW.QApplication.instance() and
                 __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.MouseButton.LeftButton,
                 pos=pr.center())
check("FixName header + → plusClicked", got == [ctw._C_FIX], f"{got}")
got.clear()
xr = hdr._x_rect(ctw._C_FIX)
QTest.mouseClick(hdr.viewport(),
                 __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.MouseButton.LeftButton,
                 pos=xr.center())
check("FixName header × still folds", got == [("x", ctw._C_FIX)], f"{got}")
check("+ only on FixName", hdr._plus_rect(ctw._C_DETAIL) is None)
hdr.folded.add(ctw._C_FIX)
check("no + while FixName is folded", hdr._plus_rect(ctw._C_FIX) is None)
_ = QPoint

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
