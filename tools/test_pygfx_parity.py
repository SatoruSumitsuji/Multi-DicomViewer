"""Windows↔Mac CT-viewer parity checks for the pygfx (Mac) viewer.

Headless-ish: builds the real CTViewer (ct_viewer_pygfx) on a synthetic CT
volume in an off-screen window and exercises the features that used to exist
only in the VTK (Windows) viewer:

    1. Resume trace   — right-click a polyline 断端 → un-commit into the draft,
                        same result id on re-commit, measurement_removed fired.
    2. Lumen snap     — _snap_to_lumen pulls a click onto the contrast lumen;
                        _snap_trace re-snaps a whole trace; menu flag exists.
    3. Off-plane cue  — trace segments with BOTH endpoints off-plane paint as a
                        DOTTED half-alpha line (in/out of range at a glance).
    4. WB reverse     — the invert button flips the grayscale (GPU map + the
                        CPU slab-MIP path).

Run:  python tools/test_pygfx_parity.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from multi_dicomviewer.core.dicom_io import LoadedSeries
from multi_dicomviewer.core.study_model import Modality
from multi_dicomviewer.viewers.ct_viewer_pygfx import (
    CTViewer, _compute_slab_qimage,
)

FAILED: list[str] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  ({extra})" if extra else ""), flush=True)
    if not ok:
        FAILED.append(name)


def make_volume(nz=64, ny=96, nx=112) -> np.ndarray:
    """(z,y,x) HU volume: air background, a soft-tissue sphere and a bright
    'contrast' cylinder along z through the centre (the lumen to snap to)."""
    z, y, x = np.indices((nz, ny, nx), dtype=np.float32)
    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
    rx, ry, rz = x - cx, y - cy, z - cz
    vol = np.full((nz, ny, nx), -1000.0, dtype=np.float32)
    rad = 0.35 * min(nx, ny)
    vol[(rx * rx + ry * ry + rz * rz) <= rad * rad] = 50.0
    cyl = 0.06 * min(nx, ny)
    vol[(rx * rx + ry * ry) <= cyl * cyl] = 400.0
    return vol


def make_series() -> LoadedSeries:
    return LoadedSeries(
        modality=Modality.CT, volume=make_volume(), spacing_mm=(0.7, 0.7),
        cine_fps=None, window=800.0, level=200.0, slice_mm=1.5,
        patient_basis=None, series_uid="parity-ct",
    )


class RecordingPainter:
    """Stub QPainter: records the pen used for every drawLine, so the overlay
    paint path can be checked without a real paint device."""

    def __init__(self):
        self.lines = []          # (pen_style, alpha, p0, p1)
        self._pen = None

    def setPen(self, pen):
        self._pen = pen

    def setBrush(self, _b):
        pass

    def setFont(self, _f):
        pass

    def drawLine(self, a, b):
        style = alpha = None
        if hasattr(self._pen, "style"):
            style = self._pen.style()
            alpha = self._pen.color().alpha()
        self.lines.append((style, alpha, (a.x(), a.y()), (b.x(), b.y())))

    def drawPolyline(self, _p):
        pass

    def drawEllipse(self, *_a):
        pass

    def drawText(self, *_a):
        pass


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    win.resize(900, 600)
    v = CTViewer()
    win.setCentralWidget(v)
    win.show()
    v.load_series(make_series(), "parity")
    # A short series loads in native-2D mode; the 3-D trace features (pts3d,
    # lumen snap, CPR) only apply to 3-D MPR — same rule as the VTK viewer.
    v._set_mode("3D")
    app.processEvents()

    # ---------------------------------------------------------- 1. resume
    print("\n1. Resume trace")
    removed: list[int] = []
    added: list[object] = []
    check("measurement_removed signal exists",
          hasattr(v, "measurement_removed"))
    v.measurement_removed.connect(removed.append)
    v.measurement_added.connect(added.append)

    key = "A"
    u, vv, _n = v._axes_for(key)
    pc = v._pc[key]
    pts2d = [(-6.0, -6.0), (-2.0, -1.0), (3.0, 4.0)]
    v._measures[key] = []
    v._meas_seq = 0
    v._draft = {"type": "polyline", "pane": key, "pts": list(pts2d),
                "pts3d": [pc + a * u + b * vv for (a, b) in pts2d]}
    v._commit_draft()
    check("polyline committed", len(v._measures[key]) == 1)
    mid0 = v._measures[key][0]["id"]
    check("Measurement carries its id (mid)",
          bool(added) and getattr(added[-1], "mid", None) == mid0,
          f"mid={getattr(added[-1], 'mid', None)}")

    # menu wiring: an END vertex offers Resume, a MIDDLE vertex does not
    import multi_dicomviewer.viewers.ct_viewer_pygfx as mod
    seen: list[list[str]] = []

    class FakeAct:
        def __init__(self, text):
            self.text = text

        def setEnabled(self, _b):
            pass

        def setCheckable(self, _b):
            pass

        def setChecked(self, _b):
            pass

        def setIcon(self, _i):
            pass

    class FakeMenu:
        def __init__(self, *_a):
            self.items: list[str] = []

        def addAction(self, text):
            self.items.append(text)
            return FakeAct(text)

        def addMenu(self, _text):
            return FakeMenu()

        def exec(self, *_a):
            seen.append(list(self.items))
            return None

    real_menu = mod.QMenu
    mod.QMenu = FakeMenu
    try:
        v._handle_right(key, (0, 2), 10, 10)          # last vertex → end
        v._handle_right(key, (0, 1), 10, 10)          # middle vertex
        v._outline_right(key, 0, 10, 10)              # outline → lumen snap
    finally:
        mod.QMenu = real_menu
    from multi_dicomviewer.i18n import t
    check("end-vertex menu offers Resume trace", t("Resume trace") in seen[0],
          str(seen[0]))
    check("middle-vertex menu has no Resume trace",
          t("Resume trace") not in seen[1], str(seen[1]))
    check("outline menu offers Snap trace to lumen",
          t("Snap trace to lumen") in seen[2], str(seen[2]))
    check("outline menu offers Auto-snap to lumen",
          t("Auto-snap to lumen") in seen[2])

    # the real thing: resume from the END vertex
    v._resume_trace(key, 0, 2)
    check("committed result un-committed", len(v._measures[key]) == 0)
    check("draft installed with all vertices",
          v._draft is not None and len(v._draft["pts"]) == 3)
    check("draft remembers the original id",
          v._draft.get("resume_id") == mid0)
    check("history entry removed", removed == [mid0], str(removed))
    check("polyline tool armed", v._meas_type == "polyline")
    # extend and re-commit → SAME id, longer trace
    v._draft["pts"].append((7.0, 8.0))
    v._draft["pts3d"].append(pc + 7.0 * u + 8.0 * vv)
    v._commit_draft()
    m = v._measures[key][0]
    check("re-committed under the same id", m["id"] == mid0,
          f"{m['id']} vs {mid0}")
    check("extended to 4 points", len(m["pts"]) == 4)

    # resume from the START vertex reverses the order
    v._resume_trace(key, 0, 0)
    first = v._draft["pts"][0]
    check("start-resume reverses the vertex order",
          abs(first[0] - 7.0) < 1e-6 and abs(first[1] - 8.0) < 1e-6,
          str(first))
    v._commit_draft()

    # --------------------------------------------------------- 2. lumen snap
    print("\n2. Lumen snap")
    check("_snap_lumen defaults on", v._snap_lumen is True)
    # The bright cylinder runs ALONG z, so pane A (axial, normal = z) has no
    # depth peak to find — use pane B (coronal), whose normal crosses the
    # cylinder, exactly like tracing a vessel that dives through the slice.
    snap_key = "B"
    u2, _v2, nrm = v._axes_for(snap_key)
    centre = np.asarray(v._center, float)
    P = centre + 3.0 * np.asarray(nrm, float)
    Q = v._snap_to_lumen(P, nrm)
    d_before = abs(float(np.dot(P - centre, nrm)))
    d_after = abs(float(np.dot(np.asarray(Q, float) - centre, nrm)))
    check("snap moves the point toward the lumen", d_after < d_before,
          f"{d_before:.2f} mm -> {d_after:.2f} mm")
    far = centre + 500.0 * np.asarray(nrm, float)      # nothing bright in reach
    check("no-op outside the volume",
          np.allclose(np.asarray(v._snap_to_lumen(far, nrm), float), far))
    v._measures[snap_key] = [{"id": 1, "type": "polyline",
                              "pts": [(-2.0, 0.0), (2.0, 0.0)],
                              "pts3d": [centre + 3.0 * np.asarray(nrm, float)
                                        - 2.0 * u2,
                                        centre + 3.0 * np.asarray(nrm, float)
                                        + 2.0 * u2]}]
    before = [np.array(P, float) for P in v._measures[snap_key][0]["pts3d"]]
    v._snap_trace(snap_key, 0)
    after = v._measures[snap_key][0]["pts3d"]
    check("_snap_trace moves every vertex",
          all(not np.allclose(a, b) for a, b in zip(before, after)))

    # ------------------------------------------------------- 3. dotted cue
    print("\n3. Off-plane dotted segments")
    # 5 vertices: 0,1 on-plane, 2,3,4 pushed 6 mm off the plane, so segment
    # 0-1 = solid/full, 1-2 = solid/half, 2-3 and 3-4 = dotted/half.
    off_k = {2, 3, 4}
    pts, p3 = [], []
    for k in range(5):
        a, b = (k - 2.0) * 4.0, 0.0
        pts.append((a, b))
        P = pc + a * u + b * vv
        if k in off_k:
            P = P + 6.0 * np.asarray(_n, float)
        p3.append(P)
    v._measures[key] = [{"id": 9, "type": "polyline", "pts": pts, "pts3d": p3}]
    v._draft = None
    v._edit = None
    rec = RecordingPainter()
    v._overlay[key]._paint_measures(rec, key, 400, 400)
    styles = [(s, a) for (s, a, _p0, _p1) in rec.lines]
    dotted = [s for (s, a) in styles if s == Qt.PenStyle.DotLine]
    solid = [s for (s, a) in styles if s == Qt.PenStyle.SolidLine]
    check("both-off segments paint DOTTED", len(dotted) == 2,
          f"{len(dotted)} dotted, {len(solid)} solid")
    alphas = sorted({a for (s, a) in styles})
    check("dotted segments are half-alpha",
          all(a <= 128 for (s, a) in styles if s == Qt.PenStyle.DotLine),
          str(alphas))
    check("fully in-range segment stays full alpha",
          any(a > 200 for (s, a) in styles if s == Qt.PenStyle.SolidLine),
          str(alphas))

    # -------------------------------------------------------- 4. WB reverse
    print("\n4. WB reverse (grayscale invert)")
    check("invert button exists", getattr(v, "_invert_btn", None) is not None)
    check("invert starts off", v._invert is False)
    v._invert_btn.setChecked(True)
    v._toggle_invert()
    check("_toggle_invert sets the flag", v._invert is True)
    lo = v._lvl - v._win / 2.0
    hi = v._lvl + v._win / 2.0
    check("grayscale panes get the inverted ramp",
          all(v.pane[k].material.map is not None for k in ("A", "B")),
          str(v.pane["A"].material.clim))
    # ...and the GPU slice really renders as the negative (not just the state).
    v._thick["A"] = 0.0                 # thin MPR → the GPU path, not the slab
    v._invert = False
    v._refresh()
    app.processEvents()
    pos = np.asarray(v.pane["A"].renderer.snapshot(), float)[..., 0]
    v._invert = True
    v._refresh()
    app.processEvents()
    inv = np.asarray(v.pane["A"].renderer.snapshot(), float)[..., 0]
    # Check the transfer function on pixels of KNOWN HU (the synthetic volume's
    # 400-HU cylinder at the centre and its 50-HU sphere): with W/L 800/200
    # those render 191 and 80, so the negative must be exactly 64 and 175.
    # (Pixel-diffing the whole frame would compare against out-of-window HU,
    # which saturates rather than mirrors, so only known HU is meaningful.)
    h, w = pos.shape
    probes = [(h // 2, w // 2), (h // 2, w // 4), (h // 3, w // 2)]
    pairs = [(pos[r, c], inv[r, c]) for (r, c) in probes]
    check("GPU slice renders an exact negative",
          all(abs(q - (255.0 - p_)) <= 1.0 for (p_, q) in pairs),
          " ".join(f"{p_:.0f}->{q:.0f}" for (p_, q) in pairs))
    prm = v._slab_params("A")
    check("slab params carry invert", prm.get("invert") is True)
    prm["thick"] = 6.0
    img_inv = _compute_slab_qimage(prm)
    prm2 = dict(prm)
    prm2["invert"] = False
    img_pos = _compute_slab_qimage(prm2)
    cx, cy = img_inv.width() // 2, img_inv.height() // 2
    g_inv = img_inv.pixelColor(cx, cy).red()
    g_pos = img_pos.pixelColor(cx, cy).red()
    check("slab MIP is inverted", abs((255 - g_pos) - g_inv) <= 1,
          f"pos={g_pos} inv={g_inv}")
    v._invert_btn.setChecked(False)
    v._toggle_invert()
    check("invert off restores the normal window",
          all(v.pane[k].material.map is None
              and v.pane[k].material.clim == (lo, hi) for k in ("A", "B")))

    # ------------------------------------ 5. traces follow the image
    print("\n5. Anatomy-anchored traces follow the view")
    v._invert = False
    v._measures = {"A": [], "B": []}
    v._draft = None
    key = "A"
    u, vv, _n = v._axes_for(key)
    pc0 = np.array(v._pc[key], float)
    pts2d = [(-6.0, -6.0), (0.0, 0.0), (5.0, 5.0)]
    p3 = [pc0 + a * u + b * vv for (a, b) in pts2d]
    v._measures[key] = [{"id": 1, "type": "polyline",
                         "pts": list(pts2d), "pts3d": p3}]
    # a committed trace must keep its 3-D anchor and re-project its 2-D coords
    v._recenter(key, 40, 40)                     # double-click recentre
    app.processEvents()
    moved = float(np.linalg.norm(np.array(v._pc[key], float) - pc0))
    m = v._measures[key][0]
    check("recentre actually moved the plane centre", moved > 0.5,
          f"{moved:.1f} mm")
    check("trace keeps its 3-D anchor",
          all(np.allclose(a, b) for a, b in zip(m["pts3d"], p3)))
    expect = [v._world3d_to_out(key, P) for P in p3]
    check("trace 2-D coords re-projected onto the moved plane",
          all(abs(q[0] - e[0]) < 1e-6 and abs(q[1] - e[1]) < 1e-6
              for q, e in zip(m["pts"], expect))
          and not np.allclose(np.asarray(m["pts"], float),
                              np.asarray(pts2d, float)),
          f"{pts2d[0]} -> {m['pts'][0]}")
    # a rotate/page must do the same (the re-projection hangs off _refresh)
    before = [tuple(q) for q in m["pts"]]
    v._pc[key] = np.array(v._pc[key], float) + 3.0 * np.asarray(u, float)
    v._refresh()
    app.processEvents()
    check("trace re-projects on any view change",
          not np.allclose(np.asarray(m["pts"], float),
                          np.asarray(before, float)),
          f"{before[0]} -> {m['pts'][0]}")

    # ------------------------------------------- 6. Shift modifiers
    print("\n6. Shift gestures (parity with the VTK viewer)")
    v._meas_on = True
    v._meas_type = "polyline"
    v._draft = {"type": "polyline", "pane": key, "pts": [(0.0, 0.0),
                                                         (2.0, 2.0)],
                "pts3d": [pc0, pc0 + 2.0 * u + 2.0 * vv]}
    n_before = len(v._measures[key])
    v._on_dblclick(key, {"button": 1, "x": 30, "y": 30,
                         "modifiers": ()})
    check("plain double-click finishes the draft while measuring",
          v._draft is None and len(v._measures[key]) == n_before + 1)
    pc_before = np.array(v._pc[key], float)
    v._on_dblclick(key, {"button": 1, "x": 60, "y": 60,
                         "modifiers": ("Shift",)})
    check("Shift+double-click recentres while measuring",
          float(np.linalg.norm(np.array(v._pc[key], float)
                               - pc_before)) > 0.5)
    v._draft = None
    v._on_down(key, {"button": 1, "x": 50, "y": 50,
                     "modifiers": ("Shift",)})
    check("Shift+press runs the tool instead of measuring",
          v._shift_tool is True and v._draft is None)
    v._on_up(key, {"button": 1, "x": 50, "y": 50, "modifiers": ()})
    check("the Shift-tool flag clears on release", v._shift_tool is False)
    v._on_down(key, {"button": 1, "x": 50, "y": 50, "modifiers": ()})
    check("plain press still measures",
          v._shift_tool is False and v._draft is not None)
    v._on_up(key, {"button": 1, "x": 50, "y": 50, "modifiers": ()})
    v._draft = None
    v._meas_on = False

    print("\n" + ("ALL PASS" if not FAILED
                  else f"{len(FAILED)} FAILED: {FAILED}"))
    win.close()
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
