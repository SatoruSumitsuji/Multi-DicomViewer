"""CPR line setting — no "CPR mode OFF" (user 2026-11-25):
- Picking Rotate / Paging while setting CPR lines no longer puts Draw on hold:
  a plain left-click keeps making / editing CPR points, and every view tool
  runs with Alt+drag (the selected tool). The ReDraw right-click menu and the
  「CPRモードオン / オフ」 notices are gone.
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
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


v = CTViewer()
check("Fit / Snap buttons gone, lumen snap off",
      v._cpr_fit_btn is None and v._cpr_snap_btn is None
      and v._snap_lumen is False)
check("no Draw-hold / ReDraw / notice helpers left",
      not any(hasattr(v, n) for n in (
          "_cpr_tool_hold_on", "_cpr_tool_hold_check", "_cpr_redraw",
          "_cpr_redraw_menu", "_cpr_hold_notice")))

# Setting CPR lines with Draw armed (no image needed for the gating)
v._coronary_mode = True
v._coronary_mpr_pending = True
v._meas_on = True
v._meas_type = "polyline"
v._mode = "3D"
v._draft = {"type": "polyline", "pane": "B", "pts": [(0.0, 0.0)]}
v._coronary_sync_ui()
armed_sty = v._cpr_draw_btn.styleSheet()

called = []
v._measure_left = lambda *a, **k: called.append(a) or True
canvas = v.pane["B"].canvas


def press(mods=Qt.KeyboardModifier.NoModifier):
    btn = Qt.MouseButton.LeftButton
    ev = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(40, 40),
                     QPointF(40, 40), btn, btn, mods)
    canvas.mousePressEvent(ev)
    rel = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(40, 40),
                      QPointF(40, 40), btn, Qt.MouseButton.NoButton, mods)
    canvas.mouseReleaseEvent(rel)


for how, pick in (("key", lambda n: v._set_tool_by_key(n)),
                  ("button", lambda n: v._set_tool(n))):
    for name in ("ROTATE", "PAGING"):
        pick(name)
        n0 = len(called)
        press()
        check(f"{name} ({how}) picked: a plain left-click still draws",
              len(called) == n0 + 1)
        check(f"{name} ({how}): Draw stays armed, trace kept, normal style",
              v._coronary_mpr_pending and v._draft is not None
              and v._cpr_draw_btn.styleSheet() == armed_sty
              and "dashed" not in v._cpr_draw_btn.styleSheet())
        n0 = len(called)
        press(Qt.KeyboardModifier.AltModifier)
        check(f"{name} ({how}): Alt+left-click runs the tool (no point)",
              len(called) == n0)

tr = {}
for lang in ("en", "ja", "es", "zh"):
    import json
    tr[lang] = json.loads((ROOT / f"multi_dicomviewer/resources/i18n/{lang}.json")
                          .read_text(encoding="utf-8"))
check("i18n: the CPR mode ON/OFF notice strings are removed",
      all(not any(k.startswith("CPRモードオ") for k in d) for d in tr.values()))

for fn in ("ct_viewer.py", "ct_viewer_pygfx.py", "cpr_mixin.py"):
    src = (ROOT / "multi_dicomviewer/viewers" / fn).read_text(encoding="utf-8")
    check(f"{fn}: no hold / ReDraw left",
          "_cpr_tool_hold" not in src and "_cpr_redraw" not in src
          and "CPRモードオ" not in src)
mac = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(encoding="utf-8")
check("Mac: Alt = tool, no Fit-Snap",
      '_alt = "Alt" in _mods\n' in mac.replace("\r\n", "\n")
      and "self._cpr_fit_btn = None" in mac and "self._snap_lumen = False" in mac
      and 't("Fit")' not in mac and 't("Snap")' not in mac)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
