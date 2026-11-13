"""Settings → 「LV Epi HU補正範囲」: the ± band of the Epi HU snap.

* epi_snap_params(range) — range 4.0 reproduces the original defaults; the band
  limits how far the snap may move the trace (edge 6 mm off: 4 mm → stays manual,
  7 mm → snaps).
* epi_band_curves — the displayed band = trace ± range along the short-axis
  radial direction, broken near the apex (r < r_min), inner side floored.
* settings persistence — clamp to 1–10 mm, 0.5 mm steps, cached getter follows
  a re-save. (Temp file — the real ~/.multi-dicomviewer is never touched.)
* Settings dialog — the combo offers ±1.0 … ±10.0 mm (19 items).
Synthetic data only.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Fake HOME before any app import: the real ~/.multi-dicomviewer is never read
# or written by this test.
_FAKE_HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = _FAKE_HOME
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:                                   # noqa: BLE001
    pass

from multi_dicomviewer.core import lv_epi_autocontour as ea  # noqa: E402
from multi_dicomviewer.core import settings as st  # noqa: E402
from multi_dicomviewer.core.lv_axis import LVAxis  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- params
d0, p4 = ea.EpiSnapParams(), ea.epi_snap_params(4.0)
check("range 4.0 == original defaults",
      (p4.search_in_mm, p4.search_out_mm, p4.max_move_mm)
      == (d0.search_in_mm, d0.search_out_mm, d0.max_move_mm))
p9 = ea.epi_snap_params(9.0)
check("range 9.0 widens the search", p9.max_move_mm == 9.0
      and p9.search_out_mm >= 9.0 + 2.0, f"{p9.search_out_mm}")
for v, want in ((0.2, 1.0), (12, 10.0), (3.26, 3.5), (3.74, 3.5), ("x", 4.0),
                (float("nan"), 4.0)):
    check(f"clamp {v!r} → {want}", ea.clamp_epi_hu_range(v) == want,
          f"{ea.clamp_epi_hu_range(v)}")

# ---------------------------------------------------------------- band geometry
apex = np.array([60.0, 60.0, 10.0])
axis = LVAxis.from_frame(apex, np.array([0, 0, 1.0]), np.array([1.0, 0, 0]))
e = axis.meridian_dir(0.0)
trace = [apex + [0, 0, z] + 20.0 * e for z in (80.0, 60.0, 40.0, 20.0)]
curves = ea.epi_band_curves(axis, trace, 3.0)
check("straight wall → 2 band curves", len(curves) == 2, f"{len(curves)}")
rad = sorted(float(np.median([axis.project(q)[1] for q in c])) for c in curves)
check("band radii = 20 -/+ 3", np.allclose(rad, [17.0, 23.0], atol=1e-6), f"{rad}")
al_tr = sorted(axis.project(q)[0] for q in trace)
al_c = [axis.project(q)[0] for q in curves[0]]
check("band spans the trace along the axis",
      abs(min(al_c) - al_tr[0]) < 1e-6 and abs(max(al_c) - al_tr[-1]) < 1e-6)
# A trace through the apex (r → 0) breaks the band there; inner side floored.
thru = ([apex + [0, 0, z] + r * e for z, r in ((60, 20), (30, 14), (8, 6))]
        + [apex + [0, 0, 0.0]]
        + [apex + [0, 0, z] - r * e for z, r in ((8, 6), (30, 14), (60, 20))])
cv = ea.epi_band_curves(axis, thru, 8.0)
check("apex breaks the band → 4 runs", len(cv) == 4, f"{len(cv)}")
rmin = min(axis.project(q)[1] for c in cv for q in c)
check("inner side floored at r_min/2", rmin >= ea.EpiSnapParams().r_min_mm * 0.5
      - 1e-9, f"{rmin:.2f}")
check("no band without axis", ea.epi_band_curves(None, trace, 4.0) == [])

# ---------------------------------------------------------------- snap reach
# Cylinder of myocardium (60 HU, r < 26 mm) in fat (−100 HU) around the axis;
# the trace sits at r = 20 → the real edge is 6 mm outside it.
sp = 1.0
zz, yy, xx = np.mgrid[0:100, 0:120, 0:120].astype(float)
vol = np.where(np.hypot(xx - 60.0, yy - 60.0) < 26.0, 60.0, -100.0)
dims = (sp, sp, sp)
dense_tr = np.array(trace)
_o4, n4, _w = ea.refine_epi_plane(vol, dims, axis, dense_tr, ea.epi_snap_params(4.0))
out7, n7, _w = ea.refine_epi_plane(vol, dims, axis, dense_tr, ea.epi_snap_params(7.0))
check("edge 6 mm off, range ±4 → stays manual", n4 == 0, f"n={n4}")
r7 = float(np.median([axis.project(q)[1] for q in out7]))
check("edge 6 mm off, range ±7 → snaps to it", n7 > 0 and abs(r7 - 26.0) < 1.0,
      f"n={n7} r={r7:.2f}")

# ---------------------------------------------------------------- persistence
tmp = Path(tempfile.mkdtemp())
st.LV_ENDO_PATH = tmp / "lv_endo_params.json"            # never the real HOME
check("default 4.0 (no file)", st.load_lv_endo_params()["epi_hu_range_mm"] == 4.0)
for v, want in ((3.3, 3.5), (15.0, 10.0), (0.4, 1.0), (7.5, 7.5)):
    prm = st.load_lv_endo_params()
    prm["epi_hu_range_mm"] = v
    st.save_lv_endo_params(prm)
    os.utime(st.LV_ENDO_PATH, ns=(0, 10 ** 18 + int(v * 1000)))   # mtime ticks
    got = st.load_lv_endo_params()["epi_hu_range_mm"]
    check(f"save {v} → load {want}", got == want, f"{got}")
    check(f"cached getter follows ({want})", st.lv_epi_hu_range_mm() == want,
          f"{st.lv_epi_hu_range_mm()}")
check("other Auto-Endo keys kept", st.load_lv_endo_params()["method"] == "hull_smooth")

# ---------------------------------------------------------------- dialog
from PyQt6.QtWidgets import QApplication  # noqa: E402
from multi_dicomviewer.ui.settings_dialog import SettingsDialog  # noqa: E402

app = QApplication.instance() or QApplication([])
dlg = SettingsDialog({"CT": 4, "XA": 4}, st.load_display_quality(), None,
                     lv_endo={"epi_hu_range_mm": 6.5})
cb = dlg._epi_hu_range_combo
items = [cb.itemData(i) for i in range(cb.count())]
check("combo ±1.0…±10.0 step 0.5 (19)",
      items == [1.0 + 0.5 * i for i in range(19)], f"{len(items)}")
check("combo shows the saved 6.5", cb.currentData() == 6.5, f"{cb.currentData()}")
cb.setCurrentIndex(0)
check("lv_endo() returns the choice", dlg.lv_endo()["epi_hu_range_mm"] == 1.0)
check("label text", cb.itemText(18) == "± 10.0 mm", cb.itemText(18))

# ---------------------------------------------------------------- viewer band
# The real _lv_epi_band_2d of BOTH viewers on a tiny fake host: pane B = the
# θ=0 meridian plane (u = radial0, v = axis), the Epi draft + this plane's
# committed Epi border get a band; Endo / other planes / after HU補正 → none.
# The methods are lifted from the viewer SOURCE (ast) so no VTK / pygfx DLL has
# to load — the code under test is exactly what the viewers run.
import ast  # noqa: E402
import textwrap  # noqa: E402

_VDIR = Path(__file__).resolve().parents[1] / "multi_dicomviewer" / "viewers"


def _methods(fname, names):
    src = (_VDIR / fname).read_text(encoding="utf-8")
    ns = {"np": np}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name in names                 and node.name not in ns:
            seg = textwrap.dedent(ast.get_source_segment(src, node))
            exec(compile(seg, fname, "exec"), ns)
    missing = [n for n in names if n not in ns]
    assert not missing, (fname, missing)
    return type("V", (), {n: ns[n] for n in names})


_NAMES = ("_lv_epi_band_2d", "_out_to_world3d", "_world3d_to_out")
CTViewer = _methods("ct_viewer.py", _NAMES)
GfxViewer = _methods("ct_viewer_pygfx.py", _NAMES)


class _Model:
    def __init__(self):
        self.epi_axis = axis
        self.epi_hu_done = False


def _host(cls):
    class H:
        _lv_epi_band_2d = cls._lv_epi_band_2d
        _out_to_world3d = cls._out_to_world3d
        _world3d_to_out = cls._world3d_to_out

        def __init__(self):
            self._pc = {"B": apex.copy()}
            self._lv = {"phase": "contour", "target": "epi", "pane": "B",
                        "plane_idx": 0, "model": _Model()}
            pts2 = [self._world3d_to_out("B", q) for q in trace]
            self._draft = {"type": "polyline", "pane": "B", "pts": pts2}
            self._measures = {"A": [], "B": [
                {"type": "polyline", "_lv": (0, "epi"), "pts3d": list(trace)},
                {"type": "polyline", "_lv": (1, "epi"), "pts3d": list(trace)},
                {"type": "polyline", "_lv": (0, "endo"), "pts3d": list(trace)}]}

        def _axes_for(self, which):
            return (axis.radial0, axis.axis, axis.binormal)
    return H()


st.save_lv_endo_params(dict(st.load_lv_endo_params(), epi_hu_range_mm=3.0))
os.utime(st.LV_ENDO_PATH, ns=(0, 2 * 10 ** 18))
for nm, cls in (("VTK", CTViewer), ("pygfx", GfxViewer)):
    h = _host(cls)
    b = h._lv_epi_band_2d("B")
    check(f"{nm}: draft + current-plane Epi → 4 curves", len(b) == 4, f"{len(b)}")
    xs = sorted({round(float(np.median([q[0] for q in c])), 6) for c in b})
    check(f"{nm}: band at u = 20 -/+ 3 (Settings 3.0)", xs == [17.0, 23.0], f"{xs}")
    check(f"{nm}: other pane → none", h._lv_epi_band_2d("A") == [])
    h._lv["target"] = "endo"
    check(f"{nm}: Endo trace → none", h._lv_epi_band_2d("B") == [])
    h._lv["target"] = "epi"
    h._lv["model"].epi_hu_done = True
    check(f"{nm}: after HU補正 → none", h._lv_epi_band_2d("B") == [])

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
