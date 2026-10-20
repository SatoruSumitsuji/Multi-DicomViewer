"""territory_file — the ``.territory.json`` bundle (a CT Territory session).

A Territory session = a source CT + its coronary tree (**corotree**) + its LV
(**FullLv**) + the chosen **Targets** + display **params**. ``corotree.json`` and
``FullLv.json`` are each their own editable source file; a ``.territory.json`` is
the convenience container that pairs them for a one-drop restore of the whole
perfusion-territory review — just as corotree + FullLv already pair as the two
inputs to CT Territory.

**Hybrid storage** (agreed 2026-10-21): the corotree + FullLv DATA are EMBEDDED
(self-contained → the review opens even if the originals were moved / renamed)
AND their source paths / names are RECORDED (shown in the UI as the linked
files, and used to re-link or re-save the originals). Nothing *computed* (the
per-voxel territory assignment, myocardium mask) is stored — it is rebuilt on
load from the embedded corotree + FullLv, exactly as a fresh analysis would.

Shape::

    {"format": "MDV-Territory", "version": 1,
     "src": {"series_uid","src_dir","patient","date","series_number"},
     "corotree": {"data": <corotree.json dict>, "path": <str>},
     "full_lv":  {"data": <FullLv.json dict>,   "path": <str>},
     "targets":  [ <target spec dict>, ... ],
     "params":   {"territory_opacity": float, ...}}
"""
from __future__ import annotations

from multi_dicomviewer.core import full_lv as _full_lv

#: File-format marker + version.
FORMAT = "MDV-Territory"
VERSION = 1


def build(src_meta: dict | None = None,
          corotree_data: dict | None = None, corotree_path: str = "",
          full_lv_data: dict | None = None, full_lv_path: str = "",
          targets=None, params=None):
    """Package a Territory session into a ``.territory.json`` dict.

    Returns ``(data, "")`` on success, or ``(None, reason)`` when a required
    piece is missing (so the caller can warn instead of writing a file that
    can't reopen the review)."""
    if not isinstance(corotree_data, dict) or not corotree_data.get("vessels"):
        return None, "no coronary tree (corotree) to save"
    if not _full_lv.is_full_lv(full_lv_data):
        return None, "no LV (FullLv) loaded to save"
    sm = dict(src_meta or {})
    # Fill any missing readable identity from the embedded FullLv (its src block
    # carries 症例/検査日/Se#), so the bundle labels itself even without the CT.
    fmeta = _full_lv.label_meta(full_lv_data)
    for k in ("series_uid", "src_dir", "patient", "date", "series_number"):
        if not sm.get(k) and fmeta.get(k):
            sm[k] = fmeta[k]
    data = {
        "format": FORMAT,
        "version": VERSION,
        "src": sm,
        "corotree": {"data": corotree_data, "path": corotree_path or ""},
        "full_lv": {"data": full_lv_data, "path": full_lv_path or ""},
        "targets": list(targets or []),
        "params": dict(params or {}),
    }
    return data, ""


def is_territory(data) -> bool:
    """True if *data* looks like a parsed ``.territory.json``."""
    return isinstance(data, dict) and data.get("format") == FORMAT


def corotree_data(data) -> dict | None:
    return ((data or {}).get("corotree") or {}).get("data")


def corotree_path(data) -> str:
    return ((data or {}).get("corotree") or {}).get("path", "") or ""


def full_lv_data(data) -> dict | None:
    return ((data or {}).get("full_lv") or {}).get("data")


def full_lv_path(data) -> str:
    return ((data or {}).get("full_lv") or {}).get("path", "") or ""


def targets(data) -> list:
    return list((data or {}).get("targets") or [])


def params(data) -> dict:
    return dict((data or {}).get("params") or {})


def series_uid(data) -> str:
    """Source-CT SeriesInstanceUID (from the src block, else the embedded
    FullLv / corotree series)."""
    fd = data or {}
    uid = (fd.get("src") or {}).get("series_uid")
    if uid:
        return uid
    uid = _full_lv.series_uid(full_lv_data(fd))
    if uid:
        return uid
    return ((corotree_data(fd) or {}).get("series") or {}).get("series_uid", "") \
        or ""


def src_dir(data) -> str:
    """Source-CT folder (src block, else the embedded FullLv / corotree)."""
    fd = data or {}
    sd = (fd.get("src") or {}).get("src_dir")
    if sd:
        return sd
    sd = _full_lv.src_dir(full_lv_data(fd))
    if sd:
        return sd
    return ((corotree_data(fd) or {}).get("series") or {}).get("src_dir", "") \
        or ""


def label_meta(data) -> dict:
    """Readable source identity (patient / date / series_number / uid / src_dir)
    for a dataset label — the src block, enriched from the embedded FullLv."""
    fd = data or {}
    src = fd.get("src") or {}
    fmeta = _full_lv.label_meta(full_lv_data(fd))

    def pick(*keys):
        for store in (src, fmeta):
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
