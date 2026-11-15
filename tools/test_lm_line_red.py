"""LM line colour = red (the most important trunk; the old pale purple read as
grey). Saved settings holding the OLD LM default are migrated to red; a colour
the user chose stays. Fake HOME — never the real settings."""
import json
import os
import sys
import tempfile
from pathlib import Path

_home = tempfile.mkdtemp(prefix="mdv_home_")
os.environ["HOME"] = os.environ["USERPROFILE"] = _home
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multi_dicomviewer.core import settings  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


RED = "#e53935"
src = (Path(__file__).resolve().parents[1]
       / "multi_dicomviewer/ui/coronary_tree_window.py").read_text(encoding="utf-8")
check("shipped default LM = red (settings + tree panel)",
      settings.CORONARY_PARAMS_DEFAULT["line_colors"]["LM"] == RED
      and f'"LM":  "{RED}"' in src)

p = Path(tempfile.mkdtemp(prefix="mdv_cp_")) / "coronary_params.json"
settings.CORONARY_PARAMS_PATH = p
p.write_text(json.dumps({"line_colors": {"LM": "#B4A7D6", "LAD": "#000001"}}),
             encoding="utf-8")
lc = settings.load_coronary_params()["line_colors"]
check("old saved LM default → red; other saved colours kept",
      lc["LM"] == RED and lc["LAD"] == "#000001", str(lc))
p.write_text(json.dumps({"line_colors": {"LM": "#123456"}}), encoding="utf-8")
check("a user-chosen LM colour is kept",
      settings.load_coronary_params()["line_colors"]["LM"] == "#123456")

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
