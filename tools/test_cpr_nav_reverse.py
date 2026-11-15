"""CPR short-axis edit-point buttons follow the SEEK BAR: ◀ / |◀ always move
LEFT on the bar, ▶ / ▶| RIGHT — also after Reverse (distal→proximal), where
the bar runs backwards along the vessel. A / F / Shift+A / Shift+F likewise.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_dicomviewer.viewers.cpr_mixin import CPRMixin  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


N = 100
CTRL = [10, 40, 70, 90]                        # centreline indices (prox→dist)


def make(reversed_, idx):
    o = SimpleNamespace(_cpr={"reversed": reversed_, "idx": idx, "n": N})
    for name in ("_cpr_jump_ctrl", "_cpr_jump_ctrl_end", "_cpr_bar_flipped",
                 "_cpr_bar_step_ctrl", "_cpr_bar_ctrl_end", "cpr_key_nav"):
        setattr(o, name, getattr(CPRMixin, name).__get__(o))
    o._cpr_ctrl_indices = lambda: list(CTRL)
    o._cpr_sync_bar = o._cpr_center_map_on_current = o._refresh = lambda: None
    return o


def bar_pos(o):
    """Seek-bar position of the current section (0 = left)."""
    i = o._cpr["idx"]
    return N - 1 - i if o._cpr["reversed"] else i


for rev in (False, True):
    tag = "reversed" if rev else "normal"
    o = make(rev, 40)
    before = bar_pos(o)
    o._cpr_bar_step_ctrl(-1)
    check(f"{tag}: ◀ moves LEFT on the bar", bar_pos(o) < before,
          f"{before}→{bar_pos(o)}")
    before = bar_pos(o)
    o._cpr_bar_step_ctrl(+1)
    check(f"{tag}: ▶ moves RIGHT on the bar", bar_pos(o) > before,
          f"{before}→{bar_pos(o)}")
    o._cpr_bar_ctrl_end(False)
    left = bar_pos(o)
    o._cpr_bar_ctrl_end(True)
    right = bar_pos(o)
    allpos = [N - 1 - i if rev else i for i in CTRL]
    check(f"{tag}: |◀ = leftmost, ▶| = rightmost edit point",
          left == min(allpos) and right == max(allpos), f"{left}, {right}")
    o = make(rev, 40)
    b = bar_pos(o)
    o.cpr_key_nav("prev")
    check(f"{tag}: key A moves LEFT too", bar_pos(o) < b)

o = make(True, 40)
o._cpr_jump_ctrl_end(False)
check("anatomical 'most proximal' jump (used when a CPR opens) unchanged",
      o._cpr["idx"] == CTRL[0])

for p in ("multi_dicomviewer/viewers/ct_viewer.py",
          "multi_dicomviewer/viewers/ct_viewer_pygfx.py"):
    src = (ROOT / p).read_text(encoding="utf-8")
    check(f"{Path(p).name}: the 4 buttons use the seek-bar direction",
          src.count("self._cpr_bar_step_ctrl(") >= 2
          and src.count("self._cpr_bar_ctrl_end(") == 2)

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
