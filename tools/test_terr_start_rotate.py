"""Territory review start (2026-11-16): the VR comes on in the right pane,
which becomes the active pane with the ROTATE tool (it started on Paging,
but the first action there is turning the VR). Waits while the VR builds.

    python tools/test_terr_start_rotate.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
os.environ["QT_QPA_PLATFORM"] = ("windows" if sys.platform == "win32"
                                 else os.environ.get("QT_QPA_PLATFORM", "offscreen"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication  # noqa: E402

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
check("default tool is Paging", v._tool == "PAGING")


def vr_now(k, on):
    v._vr_on[k] = bool(on)


v._vr_set = vr_now
v._territory_display = True
v._set_active("A")
v._terr_vr_rotate()
check("Territory start: right pane active, Rotate",
      v._active_pane == "B" and v._tool == "ROTATE" and v._vr_on["B"],
      f"{v._active_pane} {v._tool}")

# VR built a moment later → still ends on Rotate
v._vr_on["B"] = False
v._set_active("A")
v._set_tool("PAGING")
v._vr_set = lambda k, on: None              # not on yet
v._terr_vr_rotate()
check("VR not ready yet: tool not forced", v._tool == "PAGING")
v._vr_on["B"] = True                         # the VR comes up
t0 = time.time()
while time.time() - t0 < 1.5 and v._tool != "ROTATE":
    app.processEvents()
    time.sleep(0.02)
check("…then Rotate on the VR pane", v._tool == "ROTATE" and v._active_pane == "B")

src = {f: (ROOT / "multi_dicomviewer/viewers" / f).read_text(encoding="utf-8")
       for f in ("ct_mode_bar.py", "ct_viewer.py", "ct_viewer_pygfx.py")}
check("every Territory entry uses it (in-place, Windows / Mac FullLv review)",
      all("QTimer.singleShot(0, self._terr_vr_rotate)" in s for s in src.values())
      and not any('QTimer.singleShot(0, lambda: self._vr_set("B", True))' in s
                  for s in src.values()))

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
