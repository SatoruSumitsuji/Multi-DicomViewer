"""LV mode MV-AoV Set / Apex Set at the centreline crossing (2026-11-16).

(1) MV-AoV Set: Set → the point at the crossing; Adjust → a message only;
    Cancel → nothing.
(2) Apex Set: needs the MV-AoV point; Set Auto -> the crossing as the
    provisional apex, then the true apex (apex_from_point) at once (no extra
    question; a failed search keeps the provisional apex); Set Manual -> Yes:
    the crossing is the apex, No: back to the 4 choices; Adjust -> message;
    Cancel -> nothing.
Blood/Endo in the MV-AoV mode: no AoV plane needed, no valve-plane clip with
the Epi tent. No real dialogs (all mocked)."""
import sys
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from multi_dicomviewer.core import valve_auto as va  # noqa: E402
from multi_dicomviewer.viewers.valve_auto_mixin import ValveAutoMixin  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer import i18n  # noqa: E402
i18n.set_language("ja")                  # the messages are checked in Japanese
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


class V(ValveAutoMixin):
    def __init__(self):
        self._lv_valve_auto_seed_v = None
        self._lv_valves = {"mitral": None, "aortic": None}
        self._lv_valve_shown = {"mitral": True, "aortic": True}
        self._lv_valve_saved_sig = {}
        self._lv_apex = None
        self._lv_apex_shown = True
        self._lv_mvaov = None
        self._lvv = None
        self._measures = {"A": [], "B": []}
        self._center = np.array([10.0, 20.0, 100.0])
        self._vol = np.zeros((4, 4, 4), np.float32)
        self._dims = (1.0, 1.0, 1.0)
        self.prompts, self.undo = [], []

    # viewer stubs
    def window(self):
        return None

    def _lv_apply_valve_data(self, which, data):
        if which == "mitral":
            self._lv_mvaov = (dict(data["mvaov"]) if isinstance(
                data.get("mvaov"), dict) else None)
        self._lv_valves[which] = (np.asarray(data["c"], float),
                                  np.asarray(data["n"], float), float(data["r"]))

    def _lv_apex_marker_draw(self):
        pass

    def _lv_update_submode_ui(self):
        pass

    def _lv_update_valve_buttons(self):
        pass

    def _redraw_meas(self, k):
        pass

    def _lvv_prompt(self, s):
        self.prompts.append(s)

    def _undo_record(self, u, r):
        self.undo.append((u, r))

    def _run_busy(self, title, text, fn):
        return fn()


# ---- (1) MV-AoV Set ------------------------------------------------------------
v = V()
with mock.patch.object(V, "_lv_ask_set_adjust", return_value=None):
    v._lv_mvaov_set_ask()
check("Cancel → nothing", v._lv_valves["mitral"] is None and not v._lv_mvaov_on())
with mock.patch.object(V, "_lv_ask_set_adjust", return_value="adjust"), \
        mock.patch.object(QMessageBox, "information") as info:
    v._lv_mvaov_set_ask()
check("Adjust → message only", info.called and v._lv_valves["mitral"] is None
      and "MV-AoV Set" in str(info.call_args))
with mock.patch.object(V, "_lv_ask_set_adjust", return_value="set"):
    v._lv_mvaov_set_ask()
check("Set → the point at the crossing", v._lv_mvaov_on()
      and np.allclose(v._lv_valves["mitral"][0], [10, 20, 100]))
check("Set → one undo step", len(v.undo) == 1)

# ---- (2) Apex Set ----------------------------------------------------------------
w = V()
with mock.patch.object(QMessageBox, "information") as info:
    w._lv_apex_set_ask()
check("Apex Set without the MV-AoV point → told, nothing set",
      info.called and w._lv_apex is None)

v._center = np.array([12.0, 22.0, 30.0])            # the crossing
true_apex = np.array([13.0, 21.0, 26.0])
ok_res = va.ApexResult(true_apex, true_apex + [0, 0, 10], 10.0)
Q = QMessageBox.StandardButton

# Set Auto: the true apex at once — no further question.
with mock.patch.object(V, "_lv_ask_apex_mode", return_value="auto"), \
        mock.patch.object(QMessageBox, "question") as q, \
        mock.patch.object(va, "apex_from_point", return_value=ok_res) as afp:
    v._lv_apex_set_ask()
check("Set Auto → true apex", np.allclose(v._lv_apex, true_apex))
check("Set Auto → no confirmation question", not q.called)
check("search from the point + provisional",
      afp.called and np.allclose(afp.call_args[0][2], [10, 20, 100])
      and np.allclose(afp.call_args[0][3], [12, 22, 30]))
check("result message", "真Apex" in v.prompts[-1] and "Set Manual" in v.prompts[-1])
n = np.asarray(v._lv_valves["mitral"][1])
d = np.array([10, 20, 100.]) - true_apex
check("point normal = apex → point", np.allclose(n, d / np.linalg.norm(d)))

bad = va.ApexResult(true_apex, true_apex, 0.0, ok=False, message="far")
with mock.patch.object(V, "_lv_ask_apex_mode", return_value="auto"), \
        mock.patch.object(va, "apex_from_point", return_value=bad):
    v._lv_apex_set_ask()
check("failed search → provisional apex kept", np.allclose(v._lv_apex, [12, 22, 30])
      and "Set Manual" in v.prompts[-1])

# Set Manual: Yes → the crossing itself.
v._center = np.array([14.0, 23.0, 28.0])
with mock.patch.object(V, "_lv_ask_apex_mode", return_value="manual"), \
        mock.patch.object(QMessageBox, "question", return_value=Q.Yes), \
        mock.patch.object(va, "apex_from_point") as afp:
    v._lv_apex_set_ask()
check("Set Manual + Yes → the crossing", np.allclose(v._lv_apex, [14, 23, 28])
      and not afp.called)

# Set Manual: No → back to the 4 choices (then Cancel here).
v._center = np.array([0.0, 0.0, 5.0])
modes = iter(["manual", None])
with mock.patch.object(V, "_lv_ask_apex_mode", side_effect=lambda: next(modes)) as am, \
        mock.patch.object(QMessageBox, "question", return_value=Q.No):
    v._lv_apex_set_ask()
check("Set Manual + No → back to the choices", am.call_count == 2
      and np.allclose(v._lv_apex, [14, 23, 28]))

with mock.patch.object(V, "_lv_ask_apex_mode", return_value="adjust"), \
        mock.patch.object(QMessageBox, "information") as info:
    v._lv_apex_set_ask()
check("Apex Adjust → message, apex unchanged", info.called
      and np.allclose(v._lv_apex, [14, 23, 28]))
with mock.patch.object(V, "_lv_ask_apex_mode", return_value=None), \
        mock.patch.object(QMessageBox, "information") as info:
    v._lv_apex_set_ask()
check("Apex Cancel → nothing", not info.called and np.allclose(v._lv_apex, [14, 23, 28]))

# ---- Blood/Endo against the tent ---------------------------------------------------
av = v._lv_aov_eff()
check("MV-AoV mode: no AoV plane needed", av is not None)


class _Tent:
    tent = True


class _Open:
    tent = False


check("tent → no blood planes", v._lv_blood_planes(av, v._lv_valves["mitral"], _Tent()) == [])
check("non-tent Epi → the point's plane only",
      len(v._lv_blood_planes(av, v._lv_valves["mitral"], _Open())) == 1)
plain = V()
plain._lv_valves = {"mitral": (np.zeros(3), np.array([0, 0, 1.0]), 15.0),
                    "aortic": (np.ones(3), np.array([0, 1.0, 0]), 11.0)}
check("old MV + AoV mode: both planes",
      len(plain._lv_blood_planes(plain._lv_aov_eff(), plain._lv_valves["mitral"])) == 2)
plain._lv_valves["aortic"] = None
check("old mode without AoV → still required", plain._lv_aov_eff() is None)

# ---- both viewers: wiring ------------------------------------------------------------
for f in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    s = (ROOT / "multi_dicomviewer" / "viewers" / f).read_text(encoding="utf-8")
    check(f"{f}: Apex Set → provisional/true flow",
          "clicked.connect(self._lv_apex_set_ask)" in s)
    check(f"{f}: MV-AoV Set → Set/Adjust/Cancel", "self._lv_mvaov_set_ask()" in s)
    check(f"{f}: no Auto on the MV-AoV+Apex row",
          "self._lv_ve_auto_btn.setVisible(not mvx)" in s)
    check(f"{f}: AoV not in the LV selectors",
          "self._lv_selectors = [self._lv_cap, self._lv_mv_btn,\n" in s.replace("\r\n", "\n"))
    check(f"{f}: MV-AoV point drawn as a point",
          '"type": "point",\n                    "pts": [self._world3d_to_out(k, np.asarray(P))]'
          in s.replace("\r\n", "\n"))
    check(f"{f}: blood pool planes via the tent helper",
          "self._lv_blood_planes(" in s)
check("one-click MV-AoV detection removed",
      not (ROOT / "multi_dicomviewer" / "core" / "mv_aov_point.py").exists())

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
