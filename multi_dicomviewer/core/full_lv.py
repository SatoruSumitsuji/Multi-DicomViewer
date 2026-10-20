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
        # Human-readable source identity (症例 / 検査日 / Se番号) so a later load
        # WITHOUT the source CT open can still label the dataset clearly in the
        # LV Analysis / Coronary Tree tabs, not by the bare UID. Pulled from the
        # BldLv's embedded series meta (viewer._lv_series_meta()).
        "patient": ser.get("patient", "") or "",
        "date": ser.get("date", "") or "",
        "series_number": ser.get("series_number", "") or "",
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


def src_dir(data) -> str:
    """Source-CT folder recorded in a FullLv (or '')."""
    fd = data or {}
    return (((fd.get("src") or {}).get("src_dir"))
            or (fd.get("bld") or {}).get("src_dir")
            or "")


def label_meta(data) -> dict:
    """Readable source identity for a dataset label: patient / date /
    series_number / series_uid / src_dir. Prefers the FullLv ``src`` block, then
    falls back to the embedded BldLv ``series`` meta. Empty strings when absent
    (an older FullLv saved before these fields were embedded)."""
    fd = data or {}
    src = fd.get("src") or {}
    ser = (fd.get("bld") or {}).get("series") or {}

    def pick(*keys):
        for store in (src, ser):
            for k in keys:
                v = store.get(k)
                if v not in (None, ""):
                    return str(v).strip()
        return ""

    return {
        "patient": pick("patient", "patient_id"),
        "date": pick("date", "study_date"),
        "series_number": pick("series_number"),
        "series_uid": series_uid(fd),
        "src_dir": src_dir(fd),
    }
