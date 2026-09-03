"""Offline LV cardiac-function analysis from the saved EpiLv.json + BldLv.json.

Function analysis needs BOTH borders, so it reads the two files together (the
smart, drift-free split): the **Epi** stays single-source in EpiLv.json (its
region mask), and BldLv.json carries the **Endo** mask plus the voxel spacing
and LV long axis. Every metric — cavity volume, myocardial volume / mass, wall
thickness (3-D and short-axis) — is then computed from the MASKS alone, with no
CT images and no Qt/VTK. That's the contract the future 心機能 tool builds on::

    LVFunction.from_files(json.load(epilv), json.load(bldlv)).summary()

Because the Epi is never duplicated into BldLv, EpiLv and BldLv can't fall out
of sync. Pure numpy; wall thickness pulls in scipy (via core.lv_wallthickness)
lazily, so volume / mass work even without it.
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
    def from_masks(cls, endo, endo_bb, epi, epi_bb, spacing_sxyz, axis=None):
        """*endo* / *epi*: bool masks in their OWN bboxes; *spacing_sxyz* =
        (sx, sy, sz) mm; *axis* = {apex, dir, radial0} or None. Places the Endo
        onto the Epi bbox grid and returns an LVFunction (None if inputs bad)."""
        if endo is None or epi is None or not spacing_sxyz:
            return None
        sx, sy, sz = (float(s) for s in spacing_sxyz)
        ez0, ez1, ey0, ey1, ex0, ex1 = epi_bb
        endo_in = np.zeros(epi.shape, bool)
        nz0, nz1, ny0, ny1, nx0, nx1 = endo_bb
        z0, z1 = max(ez0, nz0), min(ez1, nz1)
        y0, y1 = max(ey0, ny0), min(ey1, ny1)
        x0, x1 = max(ex0, nx0), min(ex1, nx1)
        if z1 > z0 and y1 > y0 and x1 > x0:
            endo_in[z0 - ez0:z1 - ez0, y0 - ey0:y1 - ey0,
                    x0 - ex0:x1 - ex0] = endo[
                z0 - nz0:z1 - nz0, y0 - ny0:y1 - ny0, x0 - nx0:x1 - nx0]
        origin = np.array([ex0 * sx, ey0 * sy, ez0 * sz], float)
        ax = axis or {}
        apex = np.asarray(ax["apex"], float) if ax.get("apex") else None
        axis_dir = np.asarray(ax["dir"], float) if ax.get("dir") else None
        radial0 = np.asarray(ax["radial0"], float) if ax.get("radial0") else None
        return cls(endo_in, epi, (sz, sy, sx), apex, axis_dir, radial0, origin)

    @classmethod
    def from_files(cls, epilv_data, bldlv_data):
        """Combine EpiLv.json (Epi region mask = the single source of the Epi) +
        BldLv.json (Endo mask + spacing + axis) → a full LVFunction, with NO
        duplicated Epi so the two files can never drift out of sync. None if the
        Epi region or Endo mask (or spacing) is missing."""
        epi_c, epi_bb = decode_mask((epilv_data or {}).get("region"))
        endo_c, endo_bb = decode_mask((bldlv_data or {}).get("endo"))
        spacing = ((bldlv_data or {}).get("spacing")
                   or (epilv_data or {}).get("spacing"))
        axis = (bldlv_data or {}).get("axis") or (epilv_data or {}).get("axis")
        if epi_c is None or endo_c is None or not spacing:
            return None
        return cls.from_masks(endo_c, endo_bb, epi_c, epi_bb, spacing, axis)

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
