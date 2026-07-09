"""Lightweight UI translation.

``t("English source")`` returns the string in the active language. English
is the canonical KEY (identity), so any untranslated / partially-translated
string safely falls back to the readable English source — migration can be
done module by module without ever showing a blank or a raw key.

Other languages provide an ``{english: translated}`` JSON table at
``resources/i18n/<code>.json``. Placeholders use ``str.format`` so callers
pass named values: ``t("At most {n} panes can play.", n=4)``.

The active language is chosen by the shell at startup (persisted via
:mod:`multi_dicomviewer.core.settings`); this module just holds the current
code and the lazily-loaded tables. ``t()`` reads the current code on every
call, so switching language is LIVE: the shell calls ``set_language`` then
walks the persistent widgets' ``retranslate_ui`` to re-apply strings in
place (no restart). On-demand dialogs / menus are rebuilt on open and pick
up the new language for free.
"""
from __future__ import annotations

import json
import os

#: code -> native display name, in menu order. English first (the keys).
LANGUAGES = {
    "en": "English",
    "ja": "日本語",
    "zh": "中文",
    "es": "Español",
}
DEFAULT = "en"

_lang = DEFAULT
_tables: dict[str, dict] = {}


def _table_path(code: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "resources", "i18n", f"{code}.json")


def _table(code: str) -> dict:
    """The {english: translated} map for *code*, loaded once. English (and
    any missing/broken table) is the empty map → t() returns its key."""
    if code not in _tables:
        tbl: dict = {}
        if code != "en":
            try:
                with open(_table_path(code), encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    tbl = {str(k): str(v) for k, v in loaded.items()}
            except Exception:      # noqa: BLE001 — missing table ⇒ English
                tbl = {}
        _tables[code] = tbl
    return _tables[code]


def enabled_languages() -> dict:
    """Languages actually offered in the UI: English (the keys) plus any
    whose table file exists yet. Adding resources/i18n/<code>.json later
    (e.g. zh/es) makes it appear automatically — no code change."""
    out = {"en": LANGUAGES["en"]}
    for code, name in LANGUAGES.items():
        if code != "en" and os.path.exists(_table_path(code)):
            out[code] = name
    return out


def set_language(code: str) -> None:
    global _lang
    _lang = code if code in LANGUAGES else DEFAULT
    _table(_lang)              # warm the table so first t() is cheap


def get_language() -> str:
    return _lang


def t(text: str, **kwargs) -> str:
    """Translate *text* (English key) to the active language, filling any
    ``{name}`` placeholders from *kwargs*. Unknown keys fall back to the
    English source."""
    s = _table(_lang).get(text, text)
    return s.format(**kwargs) if kwargs else s
