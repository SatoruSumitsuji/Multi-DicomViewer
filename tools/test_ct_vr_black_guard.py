"""CT Territory VR came up BLACK (2026-11-16, "LAO0 CRA90" on the right pane, in
every mode): (1) the 2-D MPR fit (_fit_pane — the canvas resize refit / the
deferred Blood/Endo zoom re-fit) ran on the VR pane and moved its perspective
camera to the origin looking along +z; (2) a FullLv re-applied while its VR was
open re-toggled VR and the "last VR off → leave Territory" rule dropped the
Territory state with VR still on (mode bar → LV, no switch turned VR off).

    python tools/test_ct_vr_black_guard.py
"""
import ast
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402


class _NoPane(dict):
    def __getitem__(self, k):
        raise AssertionError("the VR pane was touched by the MPR fit")


class Fake:
    def __init__(self, vr):
        self._vr_on = {"A": False, "B": vr}
        self.pane = _NoPane()
        self._view_initial = True


# (1) the MPR fit leaves a VR pane alone
try:
    CTViewer._fit_pane(Fake(True), "B")
    ok = True
except AssertionError:
    ok = False
check("_fit_pane skips a VR pane (camera untouched)", ok)
try:
    CTViewer._fit_pane(Fake(False), "B")
    ok = False                       # must reach the pane for a normal MPR pane
except AssertionError:
    ok = True
check("_fit_pane still fits a normal MPR pane", ok)

# (2) a VR re-toggle keeps the Territory state
src = (ROOT / "multi_dicomviewer" / "viewers" / "ct_viewer.py").read_text(
    encoding="utf-8")
tree = ast.parse(src)


def seg(name):
    n = next(x for x in ast.walk(tree)
             if isinstance(x, ast.FunctionDef) and x.name == name)
    return ast.get_source_segment(src, n)


reset = seg("_vr_pane_reset_if_on")
vrset = seg("_vr_set_impl")
check("re-toggle flagged around the internal VR off",
      "self._vr_retoggle = True" in reset and "self._vr_retoggle = False" in reset)
check("last-VR-off leaves Territory only on a real exit",
      'not getattr(self, "_vr_retoggle", False)' in vrset
      and vrset.index('not getattr(self, "_vr_retoggle", False)')
      < vrset.index("self._territory_display = False"))

# (3) re-entrancy: a VR ON queued while another ON's busy dialog runs (the
# corotree drop's 領域色 ON + apply_full_lv's deferred ON) must not run nested.
from unittest import mock  # noqa: E402


class Host:
    def __init__(self):
        self._image = object()
        self._vr_on = {"A": False, "B": False}
        self.calls = []

    def _vr_pane_reset_if_on(self, key, on):
        return None

    def _vr_set_impl(self, key, on):
        self.calls.append((key, on))
        self._vr_on[key] = on
        if on and len(self.calls) == 1:
            CTViewer._vr_set(self, key, True)    # nested ON (from the busy loop)
            CTViewer._vr_set(self, key, False)   # nested OFF


h = Host()
with mock.patch("multi_dicomviewer.viewers.ct_viewer.QTimer.singleShot") as ss:
    CTViewer._vr_set(h, "B", True)
check("nested VR ON during a build is dropped", h.calls == [("B", True)], str(h.calls))
check("nested VR OFF during a build is deferred", ss.call_count == 1)
check("build flag cleared afterwards", not h._vr_building)
CTViewer._vr_set(h, "B", True)
check("a later ON runs normally", h.calls[-1] == ("B", True) and len(h.calls) == 2)

check("a parallel (MPR) camera is never saved / restored as the VR view",
      'snap_view["cam"][0]' in vrset or 'snap_view["cam"][0]' in seg("_vr_set_impl"))
impl = seg("_vr_set_impl")
check("VR off skips saving an MPR pose", "if cam.GetParallelProjection():" in impl
      and "saved.pop(key, None)" in impl)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
