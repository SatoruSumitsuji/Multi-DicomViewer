"""Pre-0.7 de-risk spike: how to do slab-MIP (the THICK tool) in pygfx.

The display path uses VolumeSliceMaterial (a single flat oblique slice). The
THICK tool needs a slab maximum-intensity projection: MIP over a slab of
thickness t centred on the slice, along the slice normal N. The spike's Phase 0
did NOT cover this. We test approach (B):

    VolumeMipMaterial (ray MIP) + an orthographic camera looking down N, with
    two opposing clipping planes bounding a slab of thickness t about the centre.

Open question this spike answers: do the material's clipping_planes actually
bound the MIP *ray integration* to the slab, or do they only clip the box
surface fragments (which would NOT give a slab)? Adjust thickness with [ and ]
and watch whether the projected content changes (the off-centre high-density
dot should appear only once the slab grows to include its depth).

If clipping does NOT bound the integration, Phase 7 falls back to a CPU slab:
sample N oblique planes with the numpy trilinear sampler (built for HU stats)
and max-composite — guaranteed correct, reuses existing code.

Run:
    python tools/slab_spike.py

Controls:  [ thinner   ] thicker   M cycle MIP/slice   R reset   Esc quit
"""
from __future__ import annotations

import sys

import numpy as np
import pygfx as gfx
from PyQt6 import QtCore, QtWidgets
from rendercanvas.pyqt6 import RenderCanvas


def make_volume(n: int = 160) -> np.ndarray:
    """Air background, soft-tissue sphere, dense axial cylinder, and an
    OFF-CENTRE high-density dot displaced along +Z so a thin centred slab
    excludes it and a thick slab includes it — the slab test signal."""
    z, y, x = np.indices((n, n, n), dtype=np.float32)
    c = (n - 1) / 2.0
    rx, ry, rz = x - c, y - c, z - c
    vol = np.full((n, n, n), -1000.0, dtype=np.float32)
    vol[(rx * rx + ry * ry + rz * rz) <= (n * 0.35) ** 2] = 50.0
    vol[(rx * rx + ry * ry) <= (n * 0.06) ** 2] = 300.0
    # Dot centred at z = c + 0.22*n (well off the central slab), x=y=c.
    dz = z - (c + n * 0.22)
    vol[(rx * rx + ry * ry + dz * dz) <= (n * 0.05) ** 2] = 1500.0
    return vol


class SlabWindow(QtWidgets.QMainWindow):
    def __init__(self, vol: np.ndarray):
        super().__init__()
        self.setWindowTitle("slab-MIP spike — VolumeMipMaterial + clipping planes")
        self.resize(820, 680)
        self._vol = vol
        nz, ny, nx = vol.shape
        self._centre = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0])
        self._n_along = nz  # slab along +Z (world z), camera looks down -Z

        self._canvas = RenderCanvas(parent=self, update_mode="continuous")
        self.setCentralWidget(self._canvas)
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()
        self._scene.add(gfx.Background(material=gfx.BackgroundMaterial(
            (0.04, 0.04, 0.04), (0.04, 0.04, 0.04))))
        self._cam = gfx.OrthographicCamera()

        tex = gfx.Texture(vol, dim=3)
        geom = gfx.Geometry(grid=tex)
        # MIP material — clim is the contrast window; clipping_mode ANY keeps
        # only fragments inside ALL planes (intersection = slab).
        self._mip = gfx.VolumeMipMaterial(clim=(-100.0, 1500.0))
        self._mip.clipping_mode = "ANY"
        self._slice = gfx.VolumeSliceMaterial(
            clim=(-100.0, 700.0), interpolation="linear",
            plane=(0.0, 0.0, 1.0, -self._centre[2]))
        self._mesh = gfx.Volume(geom, self._mip)
        self._scene.add(self._mesh)
        self._cam.show_object(self._mesh, view_dir=(0, 0, -1), up=(0, 1, 0), scale=1.2)

        self._thick = 4.0            # slab thickness in voxels (world units here)
        self._mode = "mip"
        self._clip = False           # start with NO clipping = full-thickness MIP
        self._apply_slab()

        self._lbl = QtWidgets.QLabel(self)
        self._lbl.setStyleSheet("color:#ffd700;background:rgba(0,0,0,150);"
                                "padding:4px 8px;font:bold 11pt monospace;")
        self._lbl.move(8, 8)
        self._update_lbl()

        self._canvas.request_draw(lambda: self._renderer.render(self._scene, self._cam))
        self._canvas.add_event_handler(self._on_key, "key_down")

    def _apply_slab(self):
        zc = self._centre[2]
        half = self._thick / 2.0
        zlo, zhi = zc - half, zc + half
        # Plane (a,b,c,d): fragment kept where a*x+b*y+c*z+d >= 0.
        # Keep z >= zlo  -> (0,0,1,-zlo);  keep z <= zhi -> (0,0,-1,zhi).
        if self._clip:
            self._mip.clipping_planes = [(0, 0, 1, -zlo), (0, 0, -1, zhi)]
        else:
            self._mip.clipping_planes = []
        if self._mode == "mip":
            if self._mesh.material is not self._mip:
                self._mesh.material = self._mip
        else:
            if self._mesh.material is not self._slice:
                self._mesh.material = self._slice
        self._canvas.request_draw()

    def _update_lbl(self):
        self._lbl.setText(f" mode={self._mode}  clip={'ON' if self._clip else 'OFF'}  "
                          f"slab_thick={self._thick:.0f} vox "
                          f" (dot +{self._vol.shape[0]*0.22:.0f} vox from centre) ")
        self._lbl.adjustSize()

    def _on_key(self, ev):
        k = ev["key"]
        if k == "[":
            self._thick = max(1.0, self._thick - 2.0)
        elif k == "]":
            self._thick = min(self._vol.shape[0], self._thick + 2.0)
        elif k in ("m", "M"):
            self._mode = "slice" if self._mode == "mip" else "mip"
        elif k in ("c", "C"):
            self._clip = not self._clip
        elif k == "r":
            self._thick, self._mode, self._clip = 4.0, "mip", False
        elif k == "Escape":
            self.close(); return
        else:
            return
        self._apply_slab()
        self._update_lbl()
        print(f"[slab] mode={self._mode} thick={self._thick:.0f}vox "
              f"clip={self._mip.clipping_planes}", flush=True)


def main() -> int:
    print("[slab] building synthetic volume (160^3) with off-centre dot...", flush=True)
    vol = make_volume(160)
    print(f"[slab] pygfx={gfx.__version__}  HU range {vol.min():.0f}..{vol.max():.0f}",
          flush=True)
    print("[slab] start: MIP, thin slab (dot should be HIDDEN). Press ] to widen "
          "until the bright dot appears -> clipping bounds the slab (approach B works).",
          flush=True)
    app = QtWidgets.QApplication(sys.argv)
    w = SlabWindow(vol)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
