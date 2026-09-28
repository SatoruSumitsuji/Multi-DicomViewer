"""FullLv.json — the CT-Territory LV bundle.

``corotree.json`` + ``FullLv.json`` are the two inputs to CT Territory analysis.
A FullLv file packs the **EXACT** EpiLv dict (the Epi region mask) and BldLv
dict (Endo / blood / spacing / LV axis / MV / AoV / Apex + source-CT identity)
into one file, so the myocardium (Epi ∧ ¬Endo) rebuilds via
:meth:`LVFunction.from_full` with no new mask code and no drift.

It carries no territory-unique information of its own — it is a *derived*
snapshot the user regenerates (the "FullLV" button) whenever the underlying
Epi/Endo analysis changes, exactly the way ``corotree.json`` is a compiled
snapshot of the coronary tree. The five per-analysis files stay the source of
truth; FullLv is the convenience container that pairs with corotree.

Shape::

    {"format": "MDV-FullLV", "version": 1,
     "src": {"series_uid": <uid>, "src_dir": <path>, "spacing": [sx,sy,sz]},
     "epi": { <EpiLv.json dict, must contain "region"> },
     "bld": { <BldLv.json dict: endo/blood/spacing/axis/mitral/aortic/apex> }}
"""
from __future__ import annotations

#: File-format marker + version (see the module docstring on why this exists).
FORMAT = "MDV-FullLV"
VERSION = 1


def build(epi_dict, bld_dict):
    """Package a loaded EpiLv dict + a built BldLv dict into a FullLv dict.

    Returns ``(full_dict, "")`` on success, or ``(None, reason)`` when a piece
    territory needs is missing — so the caller can warn instead of writing a
    file that can't build a myocardium."""
    if not isinstance(bld_dict, dict):
        return None, "no Blood/Endo (BldLv) analysis in memory"
    if not isinstance(epi_dict, dict):
        return None, "no Epi (EpiLv) data loaded"
    if not isinstance(epi_dict.get("region"), dict):
        return None, ("the loaded Epi has no region mask — save the EpiLv "
                      "after Calc Vol so it carries the Epi voxels")
    if not isinstance(bld_dict.get("endo"), dict):
        return None, "no Endo mask — derive the Endo before saving"
    ser = bld_dict.get("series") or {}
    src = {
        "series_uid": ser.get("series_uid", "") or "",
        "src_dir": bld_dict.get("src_dir", "") or "",
        "spacing": [float(s) for s in (bld_dict.get("spacing") or [])],
    }
    return ({"format": FORMAT, "version": VERSION, "src": src,
             "epi": epi_dict, "bld": bld_dict}, "")


def is_full_lv(data) -> bool:
    """True if *data* looks like a parsed FullLv.json."""
    return isinstance(data, dict) and data.get("format") == FORMAT


def series_uid(data) -> str:
    """Source-CT SeriesInstanceUID recorded in a FullLv (or '')."""
    fd = data or {}
    return (((fd.get("src") or {}).get("series_uid"))
            or ((fd.get("bld") or {}).get("series") or {}).get("series_uid")
            or "")


def spacing(data) -> list:
    """Voxel spacing (sx, sy, sz) recorded in a FullLv (or [])."""
    fd = data or {}
    sp = ((fd.get("src") or {}).get("spacing")
          or (fd.get("bld") or {}).get("spacing"))
    return [float(s) for s in sp] if sp else []
