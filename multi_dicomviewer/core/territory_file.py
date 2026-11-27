"""territory_file — the ``.territory.json`` file (a saved CT Territory result).

A Territory = a source CT + its coronary tree (**corotree.json**) + its LV
(**FullLv.json**) + the chosen **Targets** + display **params**. The two source
files stay their own self-contained files; a ``.territory.json`` LINKS to them
(their paths, not their data) and adds:

* a content **signature** (SHA-1) of each linked file, so a load can warn when
  a source changed since the Territory was saved (the result may differ);
* the **Targets** (vessel + position) and display **params**;
* the **result** as saved (myocardium mL, per-system and per-target
  ``pct% / mL`` lines) — a readable record of what was computed.

Loading resolves the links, loads both files, restores the Targets and
recomputes the Territory, then shows it at once (Territoryモード).

Shape::

    {"format": "MDV-Territory", "version": 3, "saved_at": "YYYYMMDD;HHMM",
     "src": {"series_uid","src_dir","patient","date","series_number"},
     "corotree": {"path": <str>, "sha1": <hex>},
     "full_lv":  {"path": <str>, "sha1": <hex>},
     "targets":  [{"vid","idx","hidden"}, ...],
     "params":   {"territory_opacity": float, "targets_shown": bool},
     "result":   {"myocardium_ml": float, "systems": [str], "targets": [str]}}

Version 2 files (link-only, no signature / result) still load.
"""
from __future__ import annotations

import hashlib

#: File-format marker + version (3 = links + signatures + result).
FORMAT = "MDV-Territory"
VERSION = 3


def file_sha1(path: str) -> str:
    """SHA-1 of a file's bytes ("" if it can't be read)."""
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def build(src_meta: dict | None = None,
          corotree_path: str = "", full_lv_path: str = "",
          targets=None, params=None, result=None, saved_at: str = ""):
    """Package a Territory into a ``.territory.json`` dict.

    Returns ``(data, "")``, or ``(None, reason)`` when a link is missing — the
    file links to the corotree.json + FullLv.json FILES, so both must be saved
    first."""
    if not corotree_path:
        return None, "corotree.json がまだ保存されていません（リンク先がありません）"
    if not full_lv_path:
        return None, "FullLv.json がまだ保存されていません（リンク先がありません）"
    data = {
        "format": FORMAT,
        "version": VERSION,
        "saved_at": saved_at or "",
        "src": dict(src_meta or {}),
        "corotree": {"path": corotree_path, "sha1": file_sha1(corotree_path)},
        "full_lv": {"path": full_lv_path, "sha1": file_sha1(full_lv_path)},
        "targets": [dict({"vid": tg.get("vid"), "idx": int(tg.get("idx", 0)),
                          "hidden": bool(tg.get("hidden", False))},
                         **({"name": str(tg["name"])} if tg.get("name") else {}))
                    for tg in (targets or []) if tg.get("vid")],
        "params": dict(params or {}),
        "result": dict(result or {}),
    }
    return data, ""


def is_territory(data) -> bool:
    """True if *data* looks like a parsed ``.territory.json`` (any version)."""
    return isinstance(data, dict) and data.get("format") == FORMAT


def corotree_path(data) -> str:
    return ((data or {}).get("corotree") or {}).get("path", "") or ""


def full_lv_path(data) -> str:
    return ((data or {}).get("full_lv") or {}).get("path", "") or ""


def corotree_sha1(data) -> str:
    return ((data or {}).get("corotree") or {}).get("sha1", "") or ""


def full_lv_sha1(data) -> str:
    return ((data or {}).get("full_lv") or {}).get("sha1", "") or ""


def targets(data) -> list:
    return list((data or {}).get("targets") or [])


def params(data) -> dict:
    return dict((data or {}).get("params") or {})


def result(data) -> dict:
    return dict((data or {}).get("result") or {})


def series_uid(data) -> str:
    """Source-CT SeriesInstanceUID recorded in the src block (or '')."""
    return ((data or {}).get("src") or {}).get("series_uid", "") or ""


def src_dir(data) -> str:
    """Source-CT folder recorded in the src block (or '')."""
    return ((data or {}).get("src") or {}).get("src_dir", "") or ""


def changed_sources(data, coro_path: str, lv_path: str) -> list:
    """Names of the linked files whose content differs from when the Territory
    was saved (by SHA-1). A file without a stored signature (version 2) is not
    reported."""
    out = []
    for name, want, p in (("corotree.json", corotree_sha1(data), coro_path),
                          ("FullLv.json", full_lv_sha1(data), lv_path)):
        if want and file_sha1(p) != want:
            out.append(name)
    return out


def set_corotree_path(data, path: str) -> None:
    """Re-link the corotree.json (used by the locate-on-missing flow)."""
    if isinstance(data, dict):
        data.setdefault("corotree", {})["path"] = path or ""


def set_full_lv_path(data, path: str) -> None:
    """Re-link the FullLv.json (used by the locate-on-missing flow)."""
    if isinstance(data, dict):
        data.setdefault("full_lv", {})["path"] = path or ""
