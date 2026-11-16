"""Territory: bar (2026-11-16) — Blood is its own ON/OFF toggle (LV-Blood ⇔ no
blood, never the All-Blood tint), independent of 壁厚; Blood / 壁厚3D / 壁厚短軸
have a coloured background while ON; 壁厚3D ⇔ 壁厚短軸 exclusive (the other is
greyed out while one is on); no 壁厚 legend on a VR pane.

The LV handlers are faked with their real side effects (a 壁厚 ON hides the
blood fill, LV mode behaviour); the Territory logic is the real mixin.

    python tools/test_terr_bar_toggles.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.viewers.ct_mode_bar import CTModeBarMixin  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


def mk():
    b = QPushButton()
    b.setCheckable(True)
    return b


class Host(CTModeBarMixin):
    _THICK_MODES = {"3d": ("_lvv_thick3d_btn", "#ffd040", "壁厚3D"),
                    "sax": ("_lvv_thicksax_btn", "#ff40d0", "壁厚短軸")}

    def __init__(self):
        self._territory_display = True
        self._lvv_mask_on = False
        self._lvv_hl_on = True                   # LV mode left All-Blood on
        self._lvv_thick_mode = None
        self._terr_blood_btn = mk()
        self._terr_thick3d_btn = mk()
        self._terr_thicksax_btn = mk()
        for b, f in ((self._terr_blood_btn, self._terr_toggle_blood),
                     (self._terr_thick3d_btn, lambda: self._terr_toggle_thick("3d")),
                     (self._terr_thicksax_btn, lambda: self._terr_toggle_thick("sax"))):
            b.clicked.connect(f)
        self.styles = {}

    # --- the LV side, as the real handlers behave
    def _lv_set_blood(self, mode):
        self._lvv_mask_on = mode == "lv"
        self._lvv_hl_on = mode == "all"

    def _lvv_set_thick_mode(self, mode):
        cur = self._lvv_thick_mode
        self._lvv_thick_mode = None if mode == cur else mode
        if self._lvv_thick_mode is not None:     # LV mode: heat map hides blood
            self._lvv_mask_on = self._lvv_hl_on = False

    def _lvv_style_toggle(self, btn, color, text="white"):
        self.styles[id(btn)] = color if btn.isChecked() else None


h = Host()
B, W3, WS = h._terr_blood_btn, h._terr_thick3d_btn, h._terr_thicksax_btn
h._terr_btn_states()
check("entry: an All-Blood tint left from LV mode reads as Blood ON", B.isChecked())
B.click()
check("Blood click (ON→OFF): no blood at all",
      not h._lvv_mask_on and not h._lvv_hl_on and not B.isChecked())
B.click()
check("Blood click (OFF→ON): LV-Blood, not the All-Blood tint",
      h._lvv_mask_on and not h._lvv_hl_on and B.isChecked())
check("Blood ON → coloured background", h.styles.get(id(B)) == "#40e0ff")
B.click()
check("Blood click again → OFF (a real toggle)", not h._lvv_mask_on
      and not B.isChecked() and h.styles.get(id(B)) is None)
B.click()

W3.click()
check("壁厚3D ON: Blood kept", h._lvv_mask_on and B.isChecked())
check("壁厚3D ON: coloured, 壁厚短軸 greyed out",
      W3.isChecked() and h.styles.get(id(W3)) == "#ffd040"
      and not WS.isEnabled() and W3.isEnabled())
W3.click()
check("壁厚3D OFF (2nd click): Blood still shown", h._lvv_mask_on and B.isChecked())
check("壁厚3D OFF: both 壁厚 buttons clickable, no colour",
      W3.isEnabled() and WS.isEnabled() and not W3.isChecked()
      and h.styles.get(id(W3)) is None)
WS.click()
check("壁厚短軸 ON: 壁厚3D greyed out, magenta background",
      not W3.isEnabled() and h.styles.get(id(WS)) == "#ff40d0")
B.click()
check("Blood OFF while 壁厚短軸 is on: the heat map stays",
      not h._lvv_mask_on and h._lvv_thick_mode == "sax")
WS.click()
check("壁厚短軸 OFF: Blood stays off (independent)", not h._lvv_mask_on
      and not B.isChecked() and W3.isEnabled())

h._territory_display = False
h._lvv_mask_on = True
h._terr_btn_states()
check("outside Territory: the bar is left alone", not B.isChecked())

# ---- both viewers: wired to the shared logic, legend never on a VR pane
for f in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    s = (ROOT / "multi_dicomviewer/viewers" / f).read_text(encoding="utf-8")
    check(f"{f}: no per-viewer Territory toggles (mixin's)",
          "    def _terr_toggle_blood(self)" not in s
          and "    def _terr_toggle_thick(self" not in s)
    check(f"{f}: fills / 壁厚 changes sync the Territory bar",
          s.count("self._terr_btn_states()") >= 3)
vtk = (ROOT / "multi_dicomviewer/viewers/ct_viewer.py").read_text(encoding="utf-8")
check("Windows: legend hidden on a VR pane",
      "p.thick_bar.SetVisibility(not self._vr_on.get(k, False))" in vtk
      and "def _lvv_thick_bar_vis(self)" in vtk)
mac = (ROOT / "multi_dicomviewer/viewers/ct_viewer_pygfx.py").read_text(encoding="utf-8")
check("Mac: legend hidden on a VR pane",
      'if not getattr(v, "_vr_on", {}).get(key):   # no legend on VR' in mac)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
