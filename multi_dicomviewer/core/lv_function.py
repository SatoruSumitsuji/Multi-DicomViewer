"""Offline LV cardiac-function analysis from a saved BldLv.json.

The BldLv file embeds the blood / Endo / Epi masks plus the voxel spacing and
the LV long axis, so every function metric — cavity volume, myocardial volume /
mass, and wall thickness (3-D and short-axis) — can be computed from the FILE
ALONE, with no CT images and no Qt/VTK. That's the contract the future 心機能
tool builds on: ``LVFunction.from_json(json.load(f)).summary()``.

Pure numpy; wall thickness pulls in scipy (via core.lv_wallthickness) lazily, so
volume / mass work even without it.
"""
from __future__ import annotations

import base64
import zlib

import numpy as np

#: Myocardial density (g/mL), the usual value for LV mass.
MYO_DENSITY_G_PER_ML = 1.05


def decode_mask(entry):
    """A saved packed-mask dict {bbox, shape, packed, …} → (comp bool [z,y,x],
    bbox) or (None, None) if absent/corrupt."""
    if not isinstance(entry, dict):
        return None, None
    try:
        shape = tuple(int(s) for s in entry["shape"])
        bbox = tuple(int(x) for x in entry["bbox"])
        raw = zlib.decompress(base64.b64decode(entry["packed"]))
        comp = np.unpackbits(np.frombuffer(raw, np.uint8))[
            :int(np.prod(shape))].reshape(shape).astype(bool)
        return comp, bbox
    except Exception:                                    # noqa: BLE001
        return None, None


class LVFunction:
    """LV masks placed on a COMMON grid (Endo/Epi on the Epi bbox) + spacing and
    axis, with the function metrics. Build via :meth:`from_json`."""

    def __init__(self, endo, epi, spacing_zyx, apex=None, axis_dir=None,
                 radial0=None, origin=None):
        self.endo = np.asarray(endo, bool)     # on the epi sub-box grid
        self.epi = np.asarray(epi, bool)
        self.spacing_zyx = tuple(float(s) for s in spacing_zyx)   # (sz, sy, sx)
        self.apex = None if apex is None else np.asarray(apex, float)
        self.axis_dir = None if axis_dir is None else np.asarray(axis_dir, float)
        self.radial0 = None if radial0 is None else np.asarray(radial0, float)
        self.origin = None if origin is None else np.asarray(origin, float)

    @classmethod
    def from_json(cls, data):
        """Build from a parsed BldLv.json dict. None if the Endo/Epi masks or
        spacing are missing."""
        endo_c, endo_bb = decode_mask(data.get("endo"))
        epi_c, epi_bb = decode_mask(data.get("epi"))
        sp = data.get("spacing")
        if endo_c is None or epi_c is None or not sp:
            return None
        sx, sy, sz = float(sp[0]), float(sp[1]), float(sp[2])
        # Place the Endo mask (its own bbox) onto the Epi sub-box grid.
        ez0, ez1, ey0, ey1, ex0, ex1 = epi_bb
        endo_in = np.zeros(epi_c.shape, bool)
        nz0, nz1, ny0, ny1, nx0, nx1 = endo_bb
        z0, z1 = max(ez0, nz0), min(ez1, nz1)
        y0, y1 = max(ey0, ny0), min(ey1, ny1)
        x0, x1 = max(ex0, nx0), min(ex1, nx1)
        if z1 > z0 and y1 > y0 and x1 > x0:
            endo_in[z0 - ez0:z1 - ez0, y0 - ey0:y1 - ey0,
                    x0 - ex0:x1 - ex0] = endo_c[
                z0 - nz0:z1 - nz0, y0 - ny0:y1 - ny0, x0 - nx0:x1 - nx0]
        origin = np.array([ex0 * sx, ey0 * sy, ez0 * sz], float)
        ax = data.get("axis") or {}
        apex = np.asarray(ax["apex"], float) if ax.get("apex") else None
        axis_dir = np.asarray(ax["dir"], float) if ax.get("dir") else None
        radial0 = np.asarray(ax["radial0"], float) if ax.get("radial0") else None
        return cls(endo_in, epi_c, (sz, sy, sx), apex, axis_dir, radial0, origin)

    # ---- metrics (masks + spacing only) --------------------------------
    def _voxel_ml(self) -> float:
        sz, sy, sx = self.spacing_zyx
        return (sz * sy * sx) / 1000.0

    def cavity_volume_ml(self) -> float:
        """LV cavity (Endo) volume."""
        return float(int(self.endo.sum())) * self._voxel_ml()

    def epi_volume_ml(self) -> float:
        return float(int(self.epi.sum())) * self._voxel_ml()

    def myocardial_volume_ml(self) -> float:
        """Myocardium = Epi minus Endo."""
        return float(int((self.epi & ~self.endo).sum())) * self._voxel_ml()

    def myocardial_mass_g(self, density=MYO_DENSITY_G_PER_ML) -> float:
        return self.myocardial_volume_ml() * float(density)

    def wall_thickness(self, mode="3d"):
        """Wall-thickness stats {min, mean, max, myo_ml} in mm. mode '3d' =
        Endo→Epi nearest distance; 'sax' = radial (needs the axis). None if the
        radial method is asked for but no axis was saved."""
        from multi_dicomviewer.core.lv_wallthickness import (
            wall_thickness_field, wall_thickness_radial_field)
        if mode == "sax":
            if (self.apex is None or self.axis_dir is None
                    or self.radial0 is None):
                return None
            apex_local = self.apex - (self.origin
                                      if self.origin is not None else 0.0)
            _t, stats = wall_thickness_radial_field(
                self.endo, self.epi, self.spacing_zyx, apex_local,
                self.axis_dir, self.radial0)
        else:
            _t, stats = wall_thickness_field(
                self.endo, self.epi, self.spacing_zyx)
        return stats

    def summary(self, density=MYO_DENSITY_G_PER_ML) -> dict:
        """All metrics in one dict — what the 心機能 tool renders."""
        return {
            "cavity_ml": self.cavity_volume_ml(),
            "epi_ml": self.epi_volume_ml(),
            "myo_ml": self.myocardial_volume_ml(),
            "myo_mass_g": self.myocardial_mass_g(density),
            "wall_3d": self.wall_thickness("3d"),
            "wall_sax": self.wall_thickness("sax"),
        }
