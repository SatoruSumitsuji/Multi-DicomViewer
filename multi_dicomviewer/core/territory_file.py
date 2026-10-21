"""territory_file — the ``.territory.json`` bundle (a CT Territory session).

A Territory session = a source CT + its coronary tree (**corotree**) + its LV
(**FullLv**) + the chosen **Targets** + display **params**. ``corotree.json`` and
``FullLv.json`` are each their own self-contained file (each holds only its own
data + a link to the source 3-D CT). A ``.territory.json`` is the small
**LINK** file that pairs them: it stores the PATHS to the corotree.json + the
FullLv.json (plus the readable source identity, the Targets and the display
params) so a one-drop restores the whole review.

**Link model** (agreed 2026-10-21): the corotree / FullLv DATA is NOT embedded
— only their file paths are. On drop the links are resolved and the two files
are loaded fresh; if a linked file is missing the caller alerts and lets the
user re-associate (locate) it, writing the new path back. Old hybrid files that
embedded data are NOT read for that data — they go through the same
locate-and-link flow, so everything is steered onto links going forward.

Shape::

    {"format": "MDV-Territory", "version": 2,
     "src": {"series_uid","src_dir","patient","date","series_number"},
     "corotree": {"path": <str>},
     "full_lv":  {"path": <str>},
     "targets":  [ <target spec dict>, ... ],
     "params":   {"territory_opacity": float, ...}}
"""
from __future__ import annotations

#: File-format marker + version (2 = link-only; 1 was the hybrid embed).
FORMAT = "MDV-Territory"
VERSION = 2


def build(src_meta: dict | None = None,
          corotree_path: str = "", full_lv_path: str = "",
          targets=None, params=None):
    """Package a Territory session into a link-only ``.territory.json`` dict.

    Returns ``(data, "")`` on success, or ``(None, reason)`` when a link is
    missing — a territory.json links to the corotree.json + FullLv.json FILES,
    so both must have been saved first."""
    if not corotree_path:
        return None, "corotree.json がまだ保存されていません（リンク先がありません）"
    if not full_lv_path:
        return None, "FullLv.json がまだ保存されていません（リンク先がありません）"
    data = {
        "format": FORMAT,
        "version": VERSION,
        "src": dict(src_meta or {}),
        "corotree": {"path": corotree_path},
        "full_lv": {"path": full_lv_path},
        "targets": list(targets or []),
        "params": dict(params or {}),
    }
    return data, ""


def is_territory(data) -> bool:
    """True if *data* looks like a parsed ``.territory.json`` (any version)."""
    return isinstance(data, dict) and data.get("format") == FORMAT


def corotree_path(data) -> str:
    return ((data or {}).get("corotree") or {}).get("path", "") or ""


def full_lv_path(data) -> str:
    return ((data or {}).get("full_lv") or {}).get("path", "") or ""


def targets(data) -> list:
    return list((data or {}).get("targets") or [])


def params(data) -> dict:
    return dict((data or {}).get("params") or {})


def series_uid(data) -> str:
    """Source-CT SeriesInstanceUID recorded in the src block (or '')."""
    return ((data or {}).get("src") or {}).get("series_uid", "") or ""


def src_dir(data) -> str:
    """Source-CT folder recorded in the src block (or '')."""
    return ((data or {}).get("src") or {}).get("src_dir", "") or ""


def label_meta(data) -> dict:
    """Readable source identity (patient / date / series_number / uid / src_dir)
    for a dataset label — from the src block only (link files aren't read here)."""
    src = (data or {}).get("src") or {}
    return {
        "patient": str(src.get("patient", "") or "").strip(),
        "date": str(src.get("date", "") or "").strip(),
        "series_number": str(src.get("series_number", "") or "").strip(),
        "series_uid": series_uid(data),
        "src_dir": src_dir(data),
    }


def set_corotree_path(data, path: str) -> None:
    """Re-link the corotree.json (used by the locate-on-missing flow)."""
    if isinstance(data, dict):
        data.setdefault("corotree", {})["path"] = path or ""


def set_full_lv_path(data, path: str) -> None:
    """Re-link the FullLv.json (used by the locate-on-missing flow)."""
    if isinstance(data, dict):
        data.setdefault("full_lv", {})["path"] = path or ""
