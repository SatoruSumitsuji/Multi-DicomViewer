"""User-persisted settings (the DICOM-tag display conditions).

The overlay tag selection survives app restarts by being written to a
small JSON file in the user's home dir on every change, and read back at
startup. The same on-disk shape is used for the manual Export/Import so a
condition set can be shared between machines.

This is intentionally separate from :mod:`config` (static, code-owned
constants) — this module owns mutable per-user state.
"""
from __future__ import annotations

import json
from pathlib import Path

SETTINGS_DIR = Path.home() / ".multi-dicomviewer"
TAG_CONDITIONS_PATH = SETTINGS_DIR / "tag_conditions.json"
EXPORT_FIELDS_PATH = SETTINGS_DIR / "export_fields.json"
ANON_PROFILE_PATH = SETTINGS_DIR / "anon_profile.json"
_SCHEMA_VERSION = 2

#: Modalities that get their own persisted tag list. Anything else
#: falls into "OTHER" so the auto-recall still works for NM/OCT/etc.
TAG_MODALITIES = ("XA", "IVUS", "CT", "OTHER")


def _parse(text: str):
    """Extract a clean keyword list (legacy v1) or per-modality dict
    (v2) from a conditions-file body.

    Returns either a ``list[str]`` (legacy) or a ``dict[str, list[str]]``
    keyed by modality. Raises ValueError/JSONDecodeError on malformed
    content so explicit Import can report it; auto-load wraps this and
    falls back instead.
    """
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("malformed settings file")
    by_mod = data.get("tag_keywords_by_modality")
    if isinstance(by_mod, dict):
        return {
            mod: [k for k in (by_mod.get(mod) or []) if isinstance(k, str) and k]
            for mod in TAG_MODALITIES
        }
    kws = data.get("tag_keywords")
    if isinstance(kws, list):
        return [k for k in kws if isinstance(k, str) and k]
    raise ValueError("no 'tag_keywords' or 'tag_keywords_by_modality' found")


def _write_keywords(path: Path, by_mod: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _SCHEMA_VERSION,
        "tag_keywords_by_modality": {
            mod: list(by_mod.get(mod, [])) for mod in TAG_MODALITIES
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_tag_keywords_by_modality() -> dict:
    """Persisted per-modality conditions, or empty lists per modality if
    absent. Legacy single-list files are silently widened so the same
    list applies to every modality (no behaviour change on upgrade)."""
    empty = {mod: [] for mod in TAG_MODALITIES}
    try:
        parsed = _parse(TAG_CONDITIONS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty
    if isinstance(parsed, list):
        return {mod: list(parsed) for mod in TAG_MODALITIES}
    out = dict(empty)
    out.update(parsed)
    return out


def save_tag_keywords_by_modality(by_mod: dict) -> None:
    """Best-effort persist; a failed write must not break the session."""
    try:
        _write_keywords(TAG_CONDITIONS_PATH, by_mod)
    except OSError:
        pass


# ---- legacy single-list helpers (kept for the Export/Import menu) ----
def load_tag_keywords() -> list[str]:
    """Back-compat helper: the XA modality's persisted list (which used
    to be the global one in v1) — used as a reasonable default when a
    legacy importer wants a single list."""
    return load_tag_keywords_by_modality().get("XA", [])


def save_tag_keywords(keywords) -> None:
    """Back-compat helper: write the same list to all modalities."""
    save_tag_keywords_by_modality(
        {mod: list(keywords) for mod in TAG_MODALITIES}
    )


def export_tag_keywords(path: str, keywords) -> None:
    """Write conditions to *path*; errors propagate for the UI to report.
    Accepts either a list (legacy) or a per-modality dict."""
    if isinstance(keywords, dict):
        _write_keywords(Path(path), keywords)
    else:
        _write_keywords(
            Path(path),
            {mod: list(keywords) for mod in TAG_MODALITIES},
        )


def import_tag_keywords(path: str):
    """Read conditions from *path*; raises on missing/invalid file.
    Returns a list (legacy file) or per-modality dict (v2 file)."""
    return _parse(Path(path).read_text(encoding="utf-8"))


# ---- per-modality export-fields memory ------------------------------
def _migrate_export_field(f: str) -> list[str]:
    """Map legacy field keys to the current set. ``date_time`` was
    split into ``date`` + ``time`` so the per-modality memory keeps
    working after the upgrade."""
    if f == "date_time":
        return ["date", "time"]
    return [f]


def load_export_fields_by_modality() -> dict:
    """Persisted Export-dialog field selection per modality, or empty."""
    empty = {mod: [] for mod in TAG_MODALITIES}
    try:
        data = json.loads(EXPORT_FIELDS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty
    if not isinstance(data, dict):
        return empty
    by_mod = data.get("export_fields_by_modality")
    if not isinstance(by_mod, dict):
        return empty
    out = {}
    for mod in TAG_MODALITIES:
        raw = [f for f in (by_mod.get(mod) or [])
                if isinstance(f, str) and f]
        flat: list[str] = []
        seen: set[str] = set()
        for f in raw:
            for migrated in _migrate_export_field(f):
                if migrated not in seen:
                    seen.add(migrated)
                    flat.append(migrated)
        out[mod] = flat
    return out


def save_export_fields_by_modality(by_mod: dict) -> None:
    """Best-effort persist of the per-modality Export-dialog selection."""
    try:
        EXPORT_FIELDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _SCHEMA_VERSION,
            "export_fields_by_modality": {
                mod: list(by_mod.get(mod, [])) for mod in TAG_MODALITIES
            },
        }
        EXPORT_FIELDS_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


# ----------------------------------------------------- anonymization profile
def load_anon_profile():
    """Return ``(tags, emptify_private)`` where *tags* is a list of
    ``(group, element)`` int pairs, or ``None`` when no profile is saved yet
    (caller then uses the built-in default)."""
    try:
        data = json.loads(ANON_PROFILE_PATH.read_text(encoding="utf-8"))
        tags = [(int(g), int(e)) for g, e in data.get("tags", [])]
        return tags, bool(data.get("emptify_private", True))
    except Exception:
        return None


def save_anon_profile(tags, emptify_private: bool) -> None:
    """Best-effort persist of the anonymization profile."""
    try:
        ANON_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "tags": [[int(g), int(e)] for g, e in tags],
            "emptify_private": bool(emptify_private),
        }
        ANON_PROFILE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass
