"""Phase 0 feasibility spike: prove pygfx can do CT MPR on Mac (Metal) and
Windows (DX12) without OpenGL.

Run this on both OSes. If both render a synthetic volume with an oblique
slice that updates as you drag, pygfx is the right replacement for VTK.

What this validates:
    1. pygfx + wgpu installs and runs on the target OS.
    2. 3-D texture upload of CT-sized volume works.
    3. ``VolumeSliceMaterial`` does what we need for MPR (oblique slice
       through a volume, sampled with linear interpolation).
    4. Render performance is interactive (>= 15 fps for a 512x512x512
       volume on the user's hardware).
    5. The backend on each OS is Metal (Mac) / DX12 (Windows), NOT
       OpenGL — so we have escaped the M1 OpenGL→Metal translation
       bug that hangs VTK CT load.

Run:
    python tools/pygfx_spike.py

Controls:
    Left-drag        rotate the slice plane around the volume centre
    Right-drag       window / level (CT-style)
    Mouse wheel      zoom in / out
    R                reset
    ESC              quit
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import pygfx as gfx
import wgpu
from PyQt6 import QtCore, QtWidgets
from rendercanvas.pyqt6 import RenderCanvas


# ------------------------------------------------------------- synthetic CT
def make_synthetic_ct(n: int = 256) -> np.ndarray:
    """Build a CT-like (n, n, n) float32 volume with HU-like values:
    * background = -1000 (air),
    * a soft-tissue sphere in the middle (~50 HU),
    * a denser cylinder running along Z through the sphere (~300 HU),
    * a high-density spot off-centre (~1200 HU).

    Distinctive enough that any orientation error in the slice is
    obvious from the rendered image."""
    z, y, x = np.indices((n, n, n), dtype=np.float32)
    cx = cy = cz = (n - 1) / 2.0
    rx, ry, rz = x - cx, y - cy, z - cz
    r2 = rx * rx + ry * ry + rz * rz

    vol = np.full((n, n, n), -1000.0, dtype=np.float32)

    sphere = r2 <= (n * 0.35) ** 2
    vol[sphere] = 50.0

    cyl = (rx * rx + ry * ry) <= (n * 0.08) ** 2
    vol[cyl] = 300.0

    dot_centre = np.array([cx + n * 0.18, cy - n * 0.10, cz + n * 0.05])
    dx = x - dot_centre[0]
    dy = y - dot_centre[1]
    dz = z - dot_centre[2]
    dot = (dx * dx + dy * dy + dz * dz) <= (n * 0.04) ** 2
    vol[dot] = 1200.0

    return vol


# ---------------------------------------------------------------- main app
class SpikeWindow(QtWidgets.QMainWindow):
    def __init__(self, vol: np.ndarray):
        super().__init__()
        self.setWindowTitle("pygfx MPR spike — Phase 0 feasibility")
        self.resize(900, 720)

        # rendercanvas-pyqt6 widget — embeds a wgpu surface in a Qt window.
        self._canvas = RenderCanvas(parent=self, update_mode="continuous")
        self.setCentralWidget(self._canvas)

        # pygfx renderer + scene + camera ----------------------------------
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()
        # Orthographic camera looking down at the slice quad from +Z.
        self._cam = gfx.OrthographicCamera(width=2.2, height=2.2)
        self._cam.local.position = (0, 0, 3)
        self._cam.look_at((0, 0, 0))

        # Volume texture ---------------------------------------------------
        # pygfx's Volume expects a Geometry whose ``grid`` attribute is
        # the 3-D texture (D, H, W). The material's ``plane`` parameter
        # cuts an arbitrary slice through the texture.
        tex = gfx.Texture(vol, dim=3)
        geom = gfx.Geometry(grid=tex)
        self._mat = gfx.VolumeSliceMaterial(
            clim=(-100.0, 700.0),   # default CT soft-tissue window
            interpolation="linear",
            plane=(0.0, 0.0, 1.0, 0.0),   # ax+by+cz+d=0 plane at z=0
        )
        self._mesh = gfx.Volume(geom, self._mat)
        self._scene.add(self._mesh)

        # Light grey background so a black volume is visible.
        self._scene.add(gfx.Background(material=gfx.BackgroundMaterial(
            (0.05, 0.05, 0.05), (0.05, 0.05, 0.05)
        )))

        # Plane rotation state (yaw, pitch around volume centre) -----------
        self._yaw = 0.0
        self._pitch = 0.0
        self._win = 800.0
        self._lvl = 300.0
        self._update_plane()

        # Reporting --------------------------------------------------------
        self._fps_t0 = time.perf_counter()
        self._fps_frames = 0
        self._fps_lbl = QtWidgets.QLabel(self)
        self._fps_lbl.setStyleSheet(
            "color:white; background:rgba(0,0,0,160); "
            "padding:4px 8px; font: bold 11pt 'Consolas';"
        )
        self._fps_lbl.move(8, 8)
        self._fps_lbl.setText("warming up…")
        self._fps_lbl.adjustSize()
        self._info_lbl = QtWidgets.QLabel(self)
        self._info_lbl.setStyleSheet(
            "color:#ffd700; background:rgba(0,0,0,160); "
            "padding:4px 8px; font: 10pt 'Consolas';"
        )
        self._info_lbl.move(8, 38)
        self._info_lbl.setText(self._backend_info())
        self._info_lbl.adjustSize()

        # Drive the render loop via Qt's event tick.
        self._canvas.request_draw(self._draw)

        # Mouse / key state for plane manipulation -------------------------
        self._drag_btn: int | None = None
        self._drag_x = 0.0
        self._drag_y = 0.0
        self._canvas.add_event_handler(self._on_pointer_down, "pointer_down")
        self._canvas.add_event_handler(self._on_pointer_up, "pointer_up")
        self._canvas.add_event_handler(self._on_pointer_move, "pointer_move")
        self._canvas.add_event_handler(self._on_wheel, "wheel")
        self._canvas.add_event_handler(self._on_key, "key_down")

    # ---------------------------------------------------------------- API
    def _backend_info(self) -> str:
        try:
            adapter = self._renderer.device.adapter
            info = adapter.info
            return (f"backend: {info.get('backend_type', '?')}   "
                    f"GPU: {info.get('device', '?')}")
        except Exception as exc:
            return f"backend info unavailable: {exc}"

    def _update_plane(self):
        # Plane normal in volume-local coords (unit cube spans -1..1).
        cy = math.cos(self._yaw)
        sy = math.sin(self._yaw)
        cp = math.cos(self._pitch)
        sp = math.sin(self._pitch)
        nx, ny, nz = sy * cp, sp, cy * cp
        self._mat.plane = (nx, ny, nz, 0.0)
        self._mat.clim = (
            self._lvl - self._win / 2.0,
            self._lvl + self._win / 2.0,
        )

    def _draw(self):
        self._renderer.render(self._scene, self._cam)
        self._fps_frames += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 0.5:
            fps = self._fps_frames / (now - self._fps_t0)
            self._fps_lbl.setText(
                f"{fps:5.1f} fps   "
                f"yaw={math.degrees(self._yaw):+6.1f}°  "
                f"pitch={math.degrees(self._pitch):+6.1f}°  "
                f"W={self._win:.0f} L={self._lvl:.0f}"
            )
            self._fps_lbl.adjustSize()
            self._fps_t0 = now
            self._fps_frames = 0

    # ---------------------------------------------------------------- input
    def _on_pointer_down(self, e):
        self._drag_btn = e["button"]
        self._drag_x = e["x"]
        self._drag_y = e["y"]

    def _on_pointer_up(self, _e):
        self._drag_btn = None

    def _on_pointer_move(self, e):
        if self._drag_btn is None:
            return
        dx = e["x"] - self._drag_x
        dy = e["y"] - self._drag_y
        self._drag_x, self._drag_y = e["x"], e["y"]
        if self._drag_btn == 1:                       # left = rotate
            self._yaw = (self._yaw + dx * 0.01) % (2 * math.pi)
            self._pitch = max(-1.4, min(1.4, self._pitch - dy * 0.01))
        elif self._drag_btn in (2, 3):                # right / middle = W/L
            self._win = max(1.0, self._win + dx * 2.0)
            self._lvl = self._lvl - dy * 2.0
        self._update_plane()
        self._canvas.request_draw()

    def _on_wheel(self, e):
        f = 1.1 if e["dy"] < 0 else (1 / 1.1)
        self._cam.width *= f
        self._cam.height *= f
        self._canvas.request_draw()

    def _on_key(self, e):
        if e["key"] == "r":
            self._yaw = self._pitch = 0.0
            self._win, self._lvl = 800.0, 300.0
            self._update_plane()
            self._canvas.request_draw()
        elif e["key"] == "Escape":
            self.close()


def main() -> int:
    print("[spike] building synthetic CT-like volume (256x256x256)...", flush=True)
    t0 = time.perf_counter()
    vol = make_synthetic_ct(256)
    print(f"[spike] volume shape={vol.shape} dtype={vol.dtype} "
          f"size={vol.nbytes / 1024 / 1024:.1f} MB  "
          f"build={time.perf_counter()-t0:.2f}s", flush=True)
    print(f"[spike] HU range: min={vol.min():.0f} max={vol.max():.0f}",
          flush=True)
    print(f"[spike] wgpu version: {wgpu.__version__}", flush=True)
    print(f"[spike] pygfx version: {gfx.__version__}", flush=True)
    adapters = wgpu.gpu.enumerate_adapters_sync()
    for a in adapters:
        info = a.info
        print(f"[spike] adapter: type={info.get('adapter_type','?')}, "
              f"device={info.get('device','?')}, "
              f"backend={info.get('backend_type','?')}", flush=True)

    app = QtWidgets.QApplication(sys.argv)
    w = SpikeWindow(vol)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
