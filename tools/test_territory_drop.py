"""Drop a .territory.json (2026-11-17): like a corotree / FullLv drop, the
source 3-D CT is found (through the linked corotree.json) and loaded, then
the Territory is loaded (tree + FullLv + Targets, recompute) and shown in the
review — no "replace the tree?" question for the tree the drop just loaded.

    python tools/test_territory_drop.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.core import territory_file as tf  # noqa: E402
from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


D = tempfile.mkdtemp(dir=HOME)
coro_p = os.path.join(D, "case.corotree.json")
lv_p = os.path.join(D, "case.FullLv.json")
with open(coro_p, "w", encoding="utf-8") as f:
    json.dump({"format": "MDV-CoroTree", "vessels": []}, f)
with open(lv_p, "w", encoding="utf-8") as f:
    json.dump({"format": "MDV-FullLv"}, f)
data, err = tf.build({"series_uid": "1.2.3"}, coro_p, lv_p)
assert data is not None, err
terr_p = os.path.join(D, "case.territory.json")
with open(terr_p, "w", encoding="utf-8") as f:
    json.dump(data, f)

check("a .territory.json is recognised as a drop", MainWindow._is_territory_json(terr_p)
      and not MainWindow._is_territory_json(coro_p)
      and not MainWindow._is_coronary_tree_json(terr_p))

QMessageBox.question = staticmethod(lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("unexpected question")))
calls = []


class V:
    _image = object()

    def _ct_territory_enter(self, full):
        return True


viewer = V()


class Shell:
    _corotree_win = None
    _territory_resolve_link = MainWindow._territory_resolve_link
    _open_territory_file = MainWindow._open_territory_file
    _poll_territory_load = MainWindow._poll_territory_load

    def __init__(self):
        self.ready = False
        self._active = self

    def current_viewer(self):
        return viewer if self.ready else None

    def _open_coronary_tree_file(self, p, to_territory=True):
        calls.append(("coro", p, to_territory))
        self.ready_at = time.time() + 0.5          # CT loads a moment later

    def _coronary_display_uid(self, uid):
        calls.append(("show", uid))
        self.ready = time.time() >= getattr(self, "ready_at", 0)
        return self.ready

    def ct_territory_load(self, v, path, confirm=True):
        calls.append(("load", v is viewer, path, confirm))
        return True


sh = Shell()
sh._open_territory_file(terr_p)
t0 = time.time()
while time.time() - t0 < 5 and not any(c[0] == "load" for c in calls):
    app.processEvents()
    time.sleep(0.02)
coro_calls = [c for c in calls if c[0] == "coro"]
load_calls = [c for c in calls if c[0] == "load"]
check("source CT opened through the linked corotree (plain, no auto-review)",
      coro_calls == [("coro", coro_p, False)], coro_calls)
check("waits for the CT, then loads the Territory on it (no replace question)",
      load_calls == [("load", True, terr_p, False)], load_calls)
check("the CT was looked up by the Territory's series UID",
      ("show", "1.2.3") in calls)

src = (ROOT / "multi_dicomviewer/ui/main_window.py").read_text(encoding="utf-8")
i = src.index("    def _load_paths(")
body = src[i:i + 6000]
check("drop routing: .territory.json peeled off before corotree / FullLv",
      body.index("_is_territory_json") < body.index("_is_coronary_tree_json")
      < body.index("_is_full_lv_json"))
check("Territory ▸ Load keeps its replace question (confirm=True default)",
      "if confirm and getattr(w, \"_tree\", None) is not None" in src)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
