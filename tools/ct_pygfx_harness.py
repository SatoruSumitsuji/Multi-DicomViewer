"""Phase 1+ verification harness for the pygfx CT viewer.

Loads a synthetic CT-like series (asymmetric so any orientation/spacing error
is obvious) into the real multi_dicomviewer.viewers.ct_viewer_pygfx.CTViewer
inside a plain Qt window — no DICOM files, no shell. Use it to eyeball each
phase on the Mac (Metal) without real CT data.

Run:
    python tools/ct_pygfx_harness.py

The synthetic volume is intentionally ANISOTROPIC (in-plane 0.7 mm, slice
1.5 mm) and NON-cubic in voxel count, so a wrong spacing/aspect shows up as a
stretched circle. Contents:
    * soft-tissue sphere (~50 HU) centred,
    * dense cylinder (~300 HU) along the slice (z) axis through the centre,
    * a high-density dot (~1200 HU) off-centre (so paging/rotate reveal it).

Pane A starts axial (normal = z), pane B coronal (normal = y). Default tool is
PAGING; pick WL/ZOOM/MOVE/ROTATE from the toolbar (or keys W/Z/V/R, G=paging).
"""
from __future__ import annotations

import sys

import numpy as np
from PyQt6 import QtWidgets

from multi_dicomviewer.core.dicom_io import LoadedSeries
from multi_dicomviewer.core.study_model import Modality
from multi_dicomviewer.viewers.ct_viewer_pygfx import CTViewer


def make_synthetic_ct(nz=160, ny=224, nx=256) -> np.ndarray:
    """(z, y, x) float32 HU volume, asymmetric contents."""
    z, y, x = np.indices((nz, ny, nx), dtype=np.float32)
    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
    rx, ry, rz = x - cx, y - cy, z - cz
    vol = np.full((nz, ny, nx), -1000.0, dtype=np.float32)
    # sphere (radius ~ 35% of the in-plane size)
    rad = 0.35 * min(nx, ny)
    vol[(rx * rx + ry * ry + rz * rz) <= rad * rad] = 50.0
    # cylinder along z (in-plane radius ~ 8%)
    cyl = 0.08 * min(nx, ny)
    vol[(rx * rx + ry * ry) <= cyl * cyl] = 300.0
    # off-centre high-density dot
    dx = x - (cx + 0.18 * nx)
    dy = y - (cy - 0.10 * ny)
    dz = z - (cz + 0.05 * nz)
    dot = 0.04 * min(nx, ny)
    vol[(dx * dx + dy * dy + dz * dz) <= dot * dot] = 1200.0
    return vol


def main() -> int:
    print("[harness] building synthetic CT (160x224x256, "
          "spacing 0.7/0.7/1.5 mm)...", flush=True)
    vol = make_synthetic_ct()
    loaded = LoadedSeries(
        modality=Modality.CT,
        volume=vol,
        spacing_mm=(0.7, 0.7),     # (row, col) in-plane mm
        cine_fps=None,
        window=800.0,
        level=200.0,
        slice_mm=1.5,              # z spacing mm
        patient_basis=None,        # identity = standard axial supine
        series_uid="synthetic-ct",
    )
    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("pygfx CT viewer — Phase 1 harness")
    win.resize(1100, 720)
    viewer = CTViewer()
    win.setCentralWidget(viewer)
    win.show()
    viewer.load_series(loaded, "synthetic")
    print("[harness] loaded. Pane A=axial, B=coronal. Tool=PAGING. "
          "Try W/Z/V/R/G + drag, wheel pages, Reset.", flush=True)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
