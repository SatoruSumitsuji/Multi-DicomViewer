"""Pre-0.5 de-risk spike: can a transparent QPainter QWidget composite OVER a
rendercanvas (wgpu/Metal) surface, and do mouse events still reach the canvas?

This is the one untested assumption behind the planned pygfx CT viewer: we want
to draw the crosshair / measurements / text with QPainter on a transparent
overlay sitting on top of the GPU-rendered MPR slice, instead of building pygfx
scene primitives. Two things must hold on macOS/Metal:

    1. COMPOSITING: the painted overlay (bright crosshair + a translucent panel
       + text) must be visible on top of the rendered slice. If the native
       Metal layer "punches through" and hides sibling Qt widgets, this fails
       and we fall back to pygfx scene primitives.
    2. EVENT ROUTING: the overlay is WA_TransparentForMouseEvents, so pointer
       events pass through to the RenderCanvas, whose pygfx handlers update the
       crosshair state and call overlay.update(). Dragging must move the painted
       crosshair live.

Run:
    python tools/overlay_spike.py

PASS if: you SEE a yellow crosshair + "OVERLAY OK" panel over the grey slice,
and left-dragging moves the crosshair to follow the cursor. Press Esc to quit.
"""
from __future__ import annotations

import sys

import numpy as np
import pygfx as gfx
from PyQt6 import QtCore, QtGui, QtWidgets
from rendercanvas.pyqt6 import RenderCanvas


def make_volume(n: int = 128) -> np.ndarray:
    """Small synthetic CT-like volume: air background, soft-tissue sphere,
    dense central cylinder. Enough to render a recognisable grey slice."""
    z, y, x = np.indices((n, n, n), dtype=np.float32)
    c = (n - 1) / 2.0
    rx, ry, rz = x - c, y - c, z - c
    vol = np.full((n, n, n), -1000.0, dtype=np.float32)
    vol[(rx * rx + ry * ry + rz * rz) <= (n * 0.35) ** 2] = 50.0
    vol[(rx * rx + ry * ry) <= (n * 0.10) ** 2] = 300.0
    return vol


class Overlay(QtWidgets.QWidget):
    """Transparent QPainter layer. Draws a crosshair at self._pt (widget px)
    plus a translucent info panel, to prove it composites over the GPU surface."""

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._pt = QtCore.QPointF(0, 0)
        self._painted = False

    def set_point(self, x: float, y: float):
        self._pt = QtCore.QPointF(x, y)
        self.update()

    def paintEvent(self, _e):
        if not self._painted:
            print("[overlay] paintEvent fired -> QPainter overlay is drawing", flush=True)
            self._painted = True
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        # Crosshair following the cursor.
        pen = QtGui.QPen(QtGui.QColor(255, 215, 0), 1.5)
        p.setPen(pen)
        x, y = self._pt.x(), self._pt.y()
        p.drawLine(QtCore.QPointF(0, y), QtCore.QPointF(w, y))
        p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h))
        p.drawEllipse(self._pt, 6, 6)
        # Translucent info panel — proves alpha compositing, not just thin lines.
        p.fillRect(QtCore.QRectF(8, 8, 240, 54),
                   QtGui.QColor(0, 0, 0, 140))
        p.setPen(QtGui.QColor(255, 215, 0))
        f = p.font(); f.setBold(True); f.setPointSize(12); p.setFont(f)
        p.drawText(QtCore.QRectF(16, 12, 230, 22),
                   QtCore.Qt.AlignmentFlag.AlignLeft, "OVERLAY OK (QPainter)")
        f.setBold(False); f.setPointSize(9); p.setFont(f)
        p.drawText(QtCore.QRectF(16, 34, 230, 22),
                   QtCore.Qt.AlignmentFlag.AlignLeft,
                   "left-drag moves crosshair  ·  Esc quits")
        p.end()


class SpikeWindow(QtWidgets.QMainWindow):
    def __init__(self, vol: np.ndarray):
        super().__init__()
        self.setWindowTitle("overlay spike — QPainter over wgpu/Metal")
        self.resize(800, 640)

        container = QtWidgets.QWidget(self)
        self.setCentralWidget(container)
        self._container = container

        # GPU canvas (bottom layer).
        self._canvas = RenderCanvas(parent=container, update_mode="continuous")
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()
        self._scene.add(gfx.Background(material=gfx.BackgroundMaterial(
            (0.05, 0.05, 0.05), (0.05, 0.05, 0.05))))
        self._cam = gfx.OrthographicCamera()

        nz, ny, nx = vol.shape
        centre = ((nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0)
        tex = gfx.Texture(vol, dim=3)
        geom = gfx.Geometry(grid=tex)
        mat = gfx.VolumeSliceMaterial(
            clim=(-100.0, 700.0), interpolation="linear",
            plane=(0.0, 0.0, 1.0, -centre[2]),
        )
        self._mesh = gfx.Volume(geom, mat)
        self._scene.add(self._mesh)
        self._cam.show_object(self._mesh, view_dir=(0, 0, -1), up=(0, 1, 0), scale=1.2)
        self._canvas.request_draw(lambda: self._renderer.render(self._scene, self._cam))

        # Overlay (top layer) — must composite over the GPU surface.
        self._overlay = Overlay(container)
        self._overlay.raise_()

        # Route canvas pointer events -> crosshair state -> overlay repaint.
        self._drag = False
        self._canvas.add_event_handler(self._on_down, "pointer_down")
        self._canvas.add_event_handler(self._on_up, "pointer_up")
        self._canvas.add_event_handler(self._on_move, "pointer_move")
        self._canvas.add_event_handler(self._on_key, "key_down")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        r = self._container.rect()
        self._canvas.setGeometry(r)
        self._overlay.setGeometry(r)

    def _on_down(self, ev):
        self._drag = True
        self._route(ev)

    def _on_up(self, _ev):
        self._drag = False

    def _on_move(self, ev):
        if self._drag:
            self._route(ev)

    def _route(self, ev):
        # rendercanvas reports logical (DIP) coords; QWidget paints in the same.
        self._overlay.set_point(ev["x"], ev["y"])
        print(f"[overlay] canvas pointer -> ({ev['x']:.0f},{ev['y']:.0f}) "
              f"-> overlay updated", flush=True)

    def _on_key(self, ev):
        if ev["key"] == "Escape":
            self.close()


def main() -> int:
    print("[overlay] building synthetic volume (128^3)...", flush=True)
    vol = make_volume(128)
    print(f"[overlay] pygfx={gfx.__version__}", flush=True)
    app = QtWidgets.QApplication(sys.argv)
    w = SpikeWindow(vol)
    w.show()
    # Seed the crosshair near the middle so it's visible before any drag.
    QtCore.QTimer.singleShot(
        200, lambda: w._overlay.set_point(w._overlay.width() / 2,
                                          w._overlay.height() / 2))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
