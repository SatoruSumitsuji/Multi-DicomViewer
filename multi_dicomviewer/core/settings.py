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
_SCHEMA_VERSION = 1


def _parse(text: str) -> list[str]:
    """Extract a clean keyword list from a conditions-file body.

    Raises ValueError/JSONDecodeError on malformed content so explicit
    Import can report it; auto-load wraps this and falls back instead.
    """
    data = json.loads(text)
    kws = data.get("tag_keywords") if isinstance(data, dict) else None
    if not isinstance(kws, list):
        raise ValueError("'tag_keywords' array not found")
    return [k for k in kws if isinstance(k, str) and k]


def _write(path: Path, keywords) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": _SCHEMA_VERSION, "tag_keywords": list(keywords)}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_tag_keywords() -> list[str]:
    """Persisted conditions, or [] if absent/unreadable (never raises)."""
    try:
        return _parse(TAG_CONDITIONS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def save_tag_keywords(keywords) -> None:
    """Best-effort persist; a failed write must not break the session."""
    try:
        _write(TAG_CONDITIONS_PATH, keywords)
    except OSError:
        pass


def export_tag_keywords(path: str, keywords) -> None:
    """Write conditions to *path*; errors propagate for the UI to report."""
    _write(Path(path), keywords)


def import_tag_keywords(path: str) -> list[str]:
    """Read conditions from *path*; raises on missing/invalid file."""
    return _parse(Path(path).read_text(encoding="utf-8"))
