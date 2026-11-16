"""CPR line setting (2026-11-16):
- Rotate / Paging picked (key or button) while setting CPR lines → Draw on
  hold: the left-click runs the tool in every pane (no trace point added, no
  edit point grabbed). Right-click ▸ 「ReDraw」 (or the Draw button) resumes;
  an armed Draw / unfinished trace is kept. Exit / Resume clear the hold.
- Fit / Snap buttons removed; the lumen snap is off; no auto-Fit on finish.

    python tools/test_cpr_draw_hold.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
os.environ["QT_QPA_PLATFORM"] = ("windows" if sys.platform == "win32"
                                 else os.environ.get("QT_QPA_PLATFORM", "offscreen"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMenu  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


v = CTViewer()
check("Fit / Snap buttons gone, lumen snap off",
      v._cpr_fit_btn is None and v._cpr_snap_btn is None
      and v._snap_lumen is False)

# Setting CPR lines with Draw armed (no image needed for the gating)
v._coronary_mode = True
v._coronary_mpr_pending = True
v._meas_on = True
v._meas_type = "polyline"
v._mode = "3D"
v._draft = {"type": "polyline", "pane": "B", "pts": [(0.0, 0.0)]}
v._coronary_sync_ui()
armed_sty = v._cpr_draw_btn.styleSheet()

v._set_tool("MOVE")
check("Move picked: no hold", not v._cpr_tool_hold_on())
v._set_tool_by_key("ROTATE")
check("Rotate (key): Draw on hold", v._cpr_tool_hold_on())
check("on hold: Draw button shows paused (dashed)",
      "dashed" in v._cpr_draw_btn.styleSheet()
      and v._cpr_draw_btn.styleSheet() != armed_sty)
check("hold keeps the armed Draw + unfinished trace",
      v._coronary_mpr_pending and v._draft is not None)

# Left-press on the right pane: the tool, not the measure
called = []
v._measure_left = lambda *a, **k: called.append(a) or True
canvas = v.pane["B"].canvas


def press(btn=Qt.MouseButton.LeftButton):
    ev = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(40, 40),
                     QPointF(40, 40), btn, btn,
                     Qt.KeyboardModifier.NoModifier)
    canvas.mousePressEvent(ev)
    rel = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(40, 40),
                      QPointF(40, 40), btn, Qt.MouseButton.NoButton,
                      Qt.KeyboardModifier.NoModifier)
    canvas.mouseReleaseEvent(rel)


press()
check("on hold: right-pane left-click never reaches the measure", called == [])

# Right-click → ReDraw menu
picked = []
orig_exec = QMenu.exec


def fake_exec(self, *a, **k):
    acts = self.actions()
    picked.append([x.text() for x in acts])
    return acts[0] if acts else None


QMenu.exec = fake_exec
press(Qt.MouseButton.RightButton)
QMenu.exec = orig_exec
check("on hold: right-click ▸ 「ReDraw」 only", picked == [["ReDraw"]], picked)
check("ReDraw: hold ended, Draw still armed, trace kept",
      not v._cpr_tool_hold_on() and v._coronary_mpr_pending
      and v._draft is not None and v._cpr_draw_btn.styleSheet() == armed_sty)
press()
check("after ReDraw: left-click draws again", len(called) == 1)

# Paging (button) → hold; Draw button resumes instead of disarming
v._tool_btns["PAGING"].click() if "PAGING" in v._tool_btns else v._set_tool("PAGING")
check("Paging (button): Draw on hold", v._cpr_tool_hold_on())
v._coronary_draw()
check("Draw button on hold: resumes (not disarmed)",
      not v._cpr_tool_hold_on() and v._coronary_mpr_pending)

# Not while measuring is off / outside coronary mode
v._set_tool("MOVE")
v._meas_on = False
v._set_tool("ROTATE")
check("Measure off: no hold", not v._cpr_tool_hold_on())
v._meas_on = True
v._set_tool("ROTATE")
check("hold again", v._cpr_tool_hold_on())
v._cpr_exit_calls = 0
src = (ROOT / "multi_dicomviewer/viewers/ct_viewer.py").read_text(encoding="utf-8")
ex = src[src.index("    def _coronary_exit(self):"):][:900]
check("Exit clears the hold", "self._cpr_tool_hold = False" in ex)
v._resume_trace = v._resume_trace           # Resume clears it (shared helper)
v._cpr_resume_auto_off()
check("Resume trace clears the hold", not v._cpr_tool_hold_on())

mac = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(encoding="utf-8")
check("Mac: same hold / ReDraw / no Fit-Snap",
      "self._cpr_tool_hold_check(name)" in mac
      and '_alt = "Alt" in _mods or self._cpr_tool_hold_on()' in mac
      and "QTimer.singleShot(0, self._cpr_redraw_menu)" in mac
      and "self._cpr_fit_btn = None" in mac and "self._snap_lumen = False" in mac
      and 't("Fit")' not in mac and 't("Snap")' not in mac)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
