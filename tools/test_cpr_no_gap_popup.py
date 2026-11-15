"""No low-contrast popup after a CPR auto trace (the gap edit points are marked
on the short axis by their orange-red ring instead); the gap marking stays."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "multi_dicomviewer/viewers/cpr_mixin.py").read_text(encoding="utf-8")
fails = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


check("no _cpr_auto_report popup any more", "_cpr_auto_report" not in src
      and "造影の弱い区間を通過しました" not in src)
check("gap edit points still marked", 'm["_cpr_gap_pts"] = gap_ctrl_pts(' in src)
print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
