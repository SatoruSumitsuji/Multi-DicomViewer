"""Mac CT redraw timing probe (pygfx viewer) — measure where a redraw's time goes.

Loads the LARGEST CT series under a folder into the real pygfx CTViewer, runs a
fixed, scripted sequence (no mouse needed), and prints a per-step breakdown:

    * present only   — pane.render() with nothing changed (GPU slice + readback
                       + Qt present). The "bitmap vs screen" number.
    * paging (drag)  — move the plane + _refresh(lod=True), like a wheel/drag.
    * at rest        — _refresh(lod=False), the immediate full-quality paint.
    * slab compute   — the CPU slab-MIP alone, at each quality tier.

for three thickness setups: thin/thin, default (A thin, B 5 mm), 5 mm/5 mm.
After the scripted run the window STAYS OPEN: use it normally (wheel, drag,
CrossLine…) and the same timers keep counting; a "manual" summary is printed
when you close the window. Everything is also written to a text file in the
current directory (mac_perf_<present>_<date>.txt).

Usage (from the repo root, in the Mac venv):
    python tools/mac_perf_probe.py /path/to/CT_folder
    python tools/mac_perf_probe.py /path/to/CT_folder --present screen

--present screen is a SPIKE: rendercanvas's direct-to-window path. Check
visually whether the crosshair / text overlays still show and whether anything
flickers or freezes — that is the open question for option (2).

Reads only; never writes settings or touches the DICOM files.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import platform
import statistics
import sys
import threading
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6 import QtWidgets  # noqa: E402

_LINES: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    _LINES.append(s)


# ---- timers -----------------------------------------------------------------
class Timers:
    def __init__(self):
        self.t = defaultdict(list)

    def add(self, name, dt):
        self.t[name].append(dt * 1000.0)

    def reset(self):
        self.t.clear()

    def summary(self, title):
        out(f"  {title}")
        if not self.t:
            out("    (no samples)")
            return
        out(f"    {'item':26s} {'n':>5s} {'median':>8s} {'p90':>8s} {'max':>8s}  ms")
        for name in sorted(self.t):
            v = sorted(self.t[name])
            p90 = v[min(len(v) - 1, int(len(v) * 0.9))]
            out(f"    {name:26s} {len(v):5d} {statistics.median(v):8.1f} "
                f"{p90:8.1f} {v[-1]:8.1f}")


T = Timers()


def wrap(obj, attr, name, main_only=False):
    orig = getattr(obj, attr)

    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            tag = name
            if main_only and threading.current_thread() is not threading.main_thread():
                tag = name + " (worker)"
            T.add(tag, time.perf_counter() - t0)
    setattr(obj, attr, w)


def find_largest_ct(path):
    from multi_dicomviewer.core import dicom_io
    from multi_dicomviewer.core.study_model import Modality
    out(f"scanning {path} ...")
    best = None
    for pat in dicom_io.scan_folder(path).values():
        for study in pat.studies.values():
            for se in study.series.values():
                if se.modality == Modality.CT and (best is None
                                                   or len(se.files) > len(best.files)):
                    best = se
    if best is None:
        raise SystemExit(f"no CT series under {path}")
    out(f"loading largest CT series: {len(best.files)} files  '{best.description}'")
    t0 = time.perf_counter()
    loaded = dicom_io.load_series(best)
    out(f"  load time {time.perf_counter() - t0:.1f} s")
    return loaded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--present", choices=("bitmap", "screen"), default="bitmap")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cpu-slab", action="store_true",
                    help="force the old CPU slab MIP (to compare with the GPU one)")
    ap.add_argument("--no-manual", action="store_true",
                    help="quit right after the scripted run (no manual phase)")
    ap.add_argument("--size", default="1600x1000",
                    help="viewer window size WxH in points (default 1600x1000)")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    import multi_dicomviewer.viewers.ct_viewer_pygfx as M

    if args.present == "screen":
        _RC = M.RenderCanvas

        class _ScreenRC(_RC):
            def __init__(self, *a, **k):
                k.setdefault("present_method", "screen")
                super().__init__(*a, **k)
        M.RenderCanvas = _ScreenRC

    # Instrument the real code paths (module-level slab fn: _build_slab_qimage
    # and the async settle both call it through the module global).
    wrap(M.CTViewer, "_refresh", "_refresh total")
    wrap(M._PygfxPane, "render", "pane.render (GPU+present)")
    wrap(M, "_compute_slab_qimage", "slab MIP (CPU)", main_only=True)
    wrap(M._Overlay, "paintEvent", "overlay paint (QPainter)")
    try:
        from multi_dicomviewer.viewers import gpu_slab
        wrap(gpu_slab.GpuSlab, "mip", "slab MIP (GPU)")
    except ImportError:
        gpu_slab = None

    loaded = find_largest_ct(args.folder)
    v = M.CTViewer()
    if args.cpu_slab:
        v._gpu_slab_error = "forced off (--cpu-slab)"
    w, h = (int(x) for x in args.size.lower().split("x"))
    v.resize(w, h)
    v.setWindowTitle(f"mac_perf_probe — present={args.present}")
    v.show()
    v.load_series(loaded, "perf")
    for _ in range(30):
        app.processEvents()
        time.sleep(0.02)

    # ---- environment ----
    out("")
    out("=" * 72)
    out(f"Mac perf probe  {_dt.datetime.now():%Y-%m-%d %H:%M:%S}  present={args.present}")
    out(f"macOS {platform.mac_ver()[0]}  {platform.machine()}  python {platform.python_version()}")
    try:
        import pygfx, wgpu, rendercanvas
        out(f"pygfx {pygfx.__version__}  wgpu {wgpu.__version__}  "
            f"rendercanvas {rendercanvas.__version__}")
        info = pygfx.renderers.wgpu.get_shared().adapter.info
        out(f"GPU: {info.get('device', '?')}  ({info.get('backend_type', '?')})")
    except Exception as exc:                             # noqa: BLE001
        out(f"(version/GPU info unavailable: {exc})")
    try:
        import subprocess
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        out(f"RAM: {mem / 2**30:.0f} GB")
    except Exception:                                    # noqa: BLE001
        pass
    vol = v._vol
    out(f"volume {vol.shape} (z,y,x) {vol.dtype}  {vol.nbytes / 2**20:.0f} MB  "
        f"spacing {tuple(round(d, 3) for d in v._dims)} mm")
    for k in ("A", "B"):
        c = v.pane[k].canvas
        out(f"pane {k}: {c.width()}x{c.height()} pt  physical {c.get_physical_size()}  "
            f"dpr {c.devicePixelRatioF():.1f}")
    out(f"CT quality mode: {v._ct_quality}   "
        f"slab tiers: LOD iw{M._SLAB_IW_LOD}/{M._SLAB_PLANES_LOD} planes, "
        f"FULL iw{M._SLAB_IW_FULL}/{M._SLAB_PLANES_FULL}, NATIVE iw{M._SLAB_IW_NATIVE}")

    def settle():
        app.processEvents()

    def page(key, step):
        n = v._frame[key][2]
        v._pc[key] = v._pc[key] + step * n

    def present_only():
        T.reset()
        v._thick = {"A": 0.0, "B": 0.0}
        v._refresh(); settle()
        T.reset()
        for _ in range(30):
            for k in ("A", "B"):
                v.pane[k].render()
            settle()
        T.summary("present only (thin, nothing changed, 30x per pane)")

    def scenario(label, thick):
        v._thick = dict(thick)
        v._refresh(); settle()
        # paging: oscillate around the start so we stay inside the volume
        step = min(v._dims)
        T.reset()
        t_frames = []
        for i in range(args.steps):
            s = step if (i // 10) % 2 == 0 else -step
            t0 = time.perf_counter()
            for k in ("A", "B"):
                page(k, s)
            v._refresh(lod=True)
            settle()
            t_frames.append((time.perf_counter() - t0) * 1000)
        T.t["== frame (both panes)"] = t_frames
        T.summary(f"[{label}] paging / drag (lod=True), {args.steps} steps")
        # at rest: full-quality synchronous paint
        T.reset()
        for _ in range(5):
            v._refresh(lod=False); settle()
        T.summary(f"[{label}] at rest (lod=False), 5x")
        # wait for the async native crisp-up to finish so it doesn't overlap
        t_end = time.time() + 5
        while getattr(v, "_lod_pending", False) and time.time() < t_end:
            settle(); time.sleep(0.02)
        for _ in range(20):
            settle(); time.sleep(0.02)

    def slab_tiers():
        if v._gpu_slab is not None:
            p = v._slab_params("B", gpu=True)
            ts = []
            for _ in range(10):
                t0 = time.perf_counter()
                v._build_slab_qimage("B")
                ts.append((time.perf_counter() - t0) * 1000)
            out(f"    slab 5 mm GPU    iw={p['iw']:4d} ih={p['ih']:4d} "
                f"(+W/L->QImage): median {statistics.median(ts):7.1f} ms/pane")
        T.reset()
        v._thick = {"A": 5.0, "B": 5.0}
        v._refresh(); settle()
        for tier, kw in (("LOD", dict(lod=True)), ("FULL", {}), ("NATIVE", dict(native=True))):
            p = v._slab_params("B", **kw)
            M._compute_slab_qimage(p)             # warm
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                M._compute_slab_qimage(p)
                ts.append((time.perf_counter() - t0) * 1000)
            out(f"    slab 5 mm {tier:6s} iw={p['iw']:4d} ih={p['ih']:4d} "
                f"planes<={p['max_planes']:3d}: median {statistics.median(ts):7.1f} ms/pane")

    v._thick = {"A": 5.0, "B": 5.0}
    v._refresh(); settle()
    out(f"slab MIP path: {'GPU' if v._gpu_slab is not None else 'CPU'}"
        + (f"  ({v._gpu_slab_error})" if v._gpu_slab_error else ""))
    out("-" * 72)
    present_only()
    for label, th in (("thin/thin", {"A": 0.0, "B": 0.0}),
                      ("default A thin, B 5mm", {"A": 0.0, "B": 5.0}),
                      ("5mm/5mm", {"A": 5.0, "B": 5.0})):
        scenario(label, th)
    out("  slab compute alone (pane B, 5 mm)")
    slab_tiers()
    out("-" * 72)
    out("Scripted run done. The window stays open: use it normally (wheel paging,")
    out("CrossLine drag, Rotate, Thick) for ~30 s, then close the window.")

    v._thick = {"A": 0.0, "B": 5.0}
    v._refresh()
    T.reset()

    def on_quit():
        T.summary("manual use (everything since the scripted run)")
        name = f"mac_perf_{args.present}_{_dt.datetime.now():%Y%m%d_%H%M%S}.txt"
        with open(name, "w", encoding="utf-8") as f:
            f.write("\n".join(_LINES) + "\n")
        print(f"\nsaved: {os.path.abspath(name)}", flush=True)
    app.aboutToQuit.connect(on_quit)
    if args.no_manual:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
