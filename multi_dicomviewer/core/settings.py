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
IVUS_COLOR_PATH = SETTINGS_DIR / "ivus_color.json"
DISPLAY_QUALITY_PATH = SETTINGS_DIR / "display_quality.json"
LANGUAGE_PATH = SETTINGS_DIR / "language.json"
DICOMFOLDER_SORT_PATH = SETTINGS_DIR / "dicomfolder_sort.json"
DICOMFOLDER_OPTIONS_PATH = SETTINGS_DIR / "dicomfolder_options.json"
LIVE_CAPS_PATH = SETTINGS_DIR / "live_caps.json"
_SCHEMA_VERSION = 2

#: How many panes of a modality may hold their full data (volume / clip) live
#: at once before the least-recently-used one is frozen to a memory-light still.
#: Kept low by default (a 600-slice cardiac CT is ~0.7 GB) and user-raisable in
#: the Display-count settings dialog up to the max.
LIVE_CAPS_DEFAULT = {"CT": 1, "XA": 3}
LIVE_CAPS_MIN = {"CT": 1, "XA": 1}
LIVE_CAPS_MAX = {"CT": 4, "XA": 8}


def load_dicomfolder_sort(default=(0, 0)) -> tuple[int, int]:
    """The persisted DicomFolder file-list sort as ``(column, order)`` — order
    is a ``Qt.SortOrder`` int (0 = ascending, 1 = descending). Falls back to
    *default* (Group column, ascending)."""
    try:
        data = json.loads(DICOMFOLDER_SORT_PATH.read_text(encoding="utf-8"))
        col = int(data.get("column"))
        order = int(data.get("order"))
        if col >= 0 and order in (0, 1):
            return col, order
    except (OSError, ValueError, TypeError):
        pass
    return default


def save_dicomfolder_sort(column: int, order: int) -> None:
    """Best-effort persist of the DicomFolder sort column + order."""
    try:
        DICOMFOLDER_SORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        DICOMFOLDER_SORT_PATH.write_text(
            json.dumps({"column": int(column), "order": int(order),
                        "version": 1}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    except OSError:
        pass


#: DicomFolder output-option checkboxes. Defaults match the pre-persistence
#: behaviour (no DICOMDIR, XA@STILL split on) so a fresh install is unchanged.
_DICOMFOLDER_OPTION_DEFAULTS = {
    "with_dicomdir": False,
    "separate_xa_still": True,
}


def load_dicomfolder_options() -> dict:
    """The persisted DicomFolder option checkboxes ("With DICOMDIR",
    "Separate XA single-frame"), falling back to the defaults for any
    missing/unreadable key."""
    out = dict(_DICOMFOLDER_OPTION_DEFAULTS)
    try:
        data = json.loads(DICOMFOLDER_OPTIONS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for k in _DICOMFOLDER_OPTION_DEFAULTS:
                if isinstance(data.get(k), bool):
                    out[k] = data[k]
    except (OSError, ValueError):
        pass
    return out


def save_dicomfolder_options(options: dict) -> None:
    """Best-effort persist of the DicomFolder option checkboxes."""
    try:
        out = {k: bool(options.get(k, _DICOMFOLDER_OPTION_DEFAULTS[k]))
               for k in _DICOMFOLDER_OPTION_DEFAULTS}
        out["version"] = 1
        DICOMFOLDER_OPTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        DICOMFOLDER_OPTIONS_PATH.write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def load_language(default: str = "en") -> str:
    """The persisted UI language code (e.g. "en"/"ja"), or *default*."""
    try:
        data = json.loads(LANGUAGE_PATH.read_text(encoding="utf-8"))
        code = data.get("language")
        if isinstance(code, str) and code:
            return code
    except (OSError, ValueError):
        pass
    return default


def save_language(code: str) -> None:
    """Best-effort persist of the chosen UI language code."""
    try:
        LANGUAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        LANGUAGE_PATH.write_text(
            json.dumps({"language": str(code), "version": 1},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
    except OSError:
        pass

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


# ----------------------------------------------------- IVUS colour display
# IVUS is decoded grayscale by default (see core.dicom_io._is_color_ds); the
# rare genuinely-colour IVUS (NIRS chemogram, VH tissue map) is opted into
# per-series by the user via the viewer's "colour display" toggle. We remember
# that choice keyed by SeriesInstanceUID so re-opening the same series restores
# colour automatically. Grayscale (the default) is NOT stored — absence means
# grayscale — so the file only ever lists series the user explicitly coloured.
def load_ivus_color(series_uid: str) -> bool:
    """True if the user previously chose colour display for this IVUS series.
    Best-effort; defaults to False (grayscale) when unset or unreadable."""
    if not series_uid:
        return False
    try:
        data = json.loads(IVUS_COLOR_PATH.read_text(encoding="utf-8"))
        series = data.get("series") if isinstance(data, dict) else None
        return bool(isinstance(series, dict) and series.get(series_uid, False))
    except (OSError, ValueError):
        return False


def save_ivus_color(series_uid: str, color: bool) -> None:
    """Best-effort persist of the IVUS colour-display choice for one series.
    A failed write must not break the session. Turning colour OFF removes the
    entry (grayscale is the unstored default)."""
    if not series_uid:
        return
    try:
        try:
            data = json.loads(IVUS_COLOR_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        series = data.get("series")
        if not isinstance(series, dict):
            series = {}
        if color:
            series[series_uid] = True
        else:
            series.pop(series_uid, None)
        data["series"] = series
        data["version"] = 1
        IVUS_COLOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        IVUS_COLOR_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


# --------------------------------------------------- display-quality prefs
#: App-wide image-quality toggles (default OFF / current behaviour, so a fresh
#: install renders exactly as before). Persisted so the user sets them once.
#: Mac 3DCT quality modes: high = always full; adaptive = crisp when still,
#: coarse while moving (default); low = always coarse.
CT_QUALITY_MODES = ("high", "adaptive", "low")

_DQ_DEFAULTS = {
    "xa_hq_cine": False,      # Angio/IVUS: smooth (bilinear) frames during cine
    "xa_smooth": False,       # Angio/IVUS: high-quality (Lanczos) upscaling
    "xa_denoise": False,      # Angio/IVUS: edge-preserving noise reduction
    "ct_quality_mode": "adaptive",   # Mac 3DCT: high | adaptive | low
}


def load_display_quality() -> dict:
    """Return the persisted image-quality prefs, falling back to defaults for
    any missing/unreadable key."""
    out = dict(_DQ_DEFAULTS)
    try:
        data = json.loads(DISPLAY_QUALITY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for k, dv in _DQ_DEFAULTS.items():
                v = data.get(k)
                if isinstance(dv, bool):
                    if isinstance(v, bool):
                        out[k] = v
                elif k == "ct_quality_mode":
                    if v in CT_QUALITY_MODES:
                        out[k] = v
            # Migrate the legacy boolean (ct_full_quality=True → always high).
            if data.get("ct_quality_mode") not in CT_QUALITY_MODES \
                    and data.get("ct_full_quality") is True:
                out["ct_quality_mode"] = "high"
    except (OSError, ValueError):
        pass
    return out


def save_display_quality(prefs: dict) -> None:
    """Best-effort persist of the image-quality toggles. A failed write must
    not break the session."""
    try:
        out = {}
        for k, dv in _DQ_DEFAULTS.items():
            v = prefs.get(k, dv)
            if k == "ct_quality_mode":
                out[k] = v if v in CT_QUALITY_MODES else dv
            else:
                out[k] = bool(v)
        out["version"] = 1
        DISPLAY_QUALITY_PATH.parent.mkdir(parents=True, exist_ok=True)
        DISPLAY_QUALITY_PATH.write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


# ----------------------------------------------- live pane count (memory cap)
def _clamp_cap(mod: str, val) -> int:
    lo, hi = LIVE_CAPS_MIN[mod], LIVE_CAPS_MAX[mod]
    try:
        return max(lo, min(hi, int(val)))
    except (TypeError, ValueError):
        return LIVE_CAPS_DEFAULT[mod]


def load_live_caps() -> dict:
    """Return the per-modality 'live at once' caps {"CT": n, "XA": m},
    clamped to the allowed range, falling back to the defaults."""
    out = dict(LIVE_CAPS_DEFAULT)
    try:
        data = json.loads(LIVE_CAPS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for k in LIVE_CAPS_DEFAULT:
                if k in data:
                    out[k] = _clamp_cap(k, data[k])
    except (OSError, ValueError):
        pass
    return out


def save_live_caps(caps: dict) -> None:
    """Best-effort persist of the live-pane caps. A failed write must not
    break the session."""
    try:
        out = {k: _clamp_cap(k, caps.get(k, LIVE_CAPS_DEFAULT[k]))
               for k in LIVE_CAPS_DEFAULT}
        out["version"] = 1
        LIVE_CAPS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LIVE_CAPS_PATH.write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


# --------------------------------------- advanced angio/IVUS image quality
ADVANCED_QUALITY_PATH = SETTINGS_DIR / "advanced_quality.json"

#: Fine-grained enhancement parameters for the 2-D (XA / IVUS) viewer, tuned in
#: the "Advanced" quality dialog. Defaults reproduce the classic single-step
#: Denoise: denoise sigma 50, no sharpen, no CLAHE. CT is NOT affected.
_ADV_QUALITY_DEFAULTS = {"denoise": 50.0, "sharpen": 0.0, "clahe": 0.0}
_ADV_QUALITY_RANGE = {
    "denoise": (0.0, 150.0),     # bilateral colour sigma
    "sharpen": (0.0, 200.0),     # unsharp amount, %
    "clahe": (0.0, 4.0),         # CLAHE clip limit
}


def load_advanced_quality() -> dict:
    """Return the advanced XA/IVUS enhancement params, clamped, with defaults
    for any missing/unreadable key."""
    out = dict(_ADV_QUALITY_DEFAULTS)
    try:
        data = json.loads(ADVANCED_QUALITY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for k, (lo, hi) in _ADV_QUALITY_RANGE.items():
                if k in data:
                    try:
                        out[k] = max(lo, min(hi, float(data[k])))
                    except (TypeError, ValueError):
                        pass
    except (OSError, ValueError):
        pass
    return out


def save_advanced_quality(params: dict) -> None:
    """Best-effort persist of the advanced XA/IVUS enhancement params."""
    try:
        out = {}
        for k, (lo, hi) in _ADV_QUALITY_RANGE.items():
            try:
                out[k] = max(lo, min(hi, float(
                    params.get(k, _ADV_QUALITY_DEFAULTS[k]))))
            except (TypeError, ValueError):
                out[k] = _ADV_QUALITY_DEFAULTS[k]
        out["version"] = 1
        ADVANCED_QUALITY_PATH.parent.mkdir(parents=True, exist_ok=True)
        ADVANCED_QUALITY_PATH.write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


# ------------------------------------------------- CT HU colour map (global)
CT_COLORMAP_PATH = SETTINGS_DIR / "ct_colormap.json"

#: Factory-default HU colour bands (must match the viewers' _DEFAULT_BANDS).
#: Each band colours HU range [lo, hi]; "on" toggles it. Shared by every CT
#: pane and persisted, so a colour edit in one pane applies everywhere and
#: survives a restart.
CT_COLORMAP_DEFAULT_BANDS = [
    {"rgb": (1.0, 0.0, 0.0), "lo": -1000, "hi": 0,    "on": True},
    {"rgb": (1.0, 1.0, 0.0), "lo": 0,     "hi": 50,   "on": True},
    {"rgb": (0.0, 1.0, 0.0), "lo": 50,    "hi": 250,  "on": True},
    {"rgb": (0.0, 0.0, 1.0), "lo": 250,   "hi": 350,  "on": False},
    {"rgb": (1.0, 1.0, 1.0), "lo": 350,   "hi": 700,  "on": True},
    {"rgb": (1.0, 0.0, 1.0), "lo": 850,   "hi": 2000, "on": True},
]
CT_COLORMAP_DEFAULT_OPACITY = 0.25


def _sanitize_band(b):
    """Coerce one persisted band dict into the canonical shape, or None."""
    try:
        rgb = tuple(float(c) for c in b["rgb"])
        if len(rgb) != 3:
            return None
        return {"rgb": rgb, "lo": int(b["lo"]), "hi": int(b["hi"]),
                "on": bool(b["on"])}
    except (KeyError, TypeError, ValueError):
        return None


#: Default spatial colour-smoothing strength (mm) — a weak Gaussian on the
#: reslice before colour mapping so the band boundaries read as smooth curves
#: (like SSMView) instead of the voxel-grid staircase. 0 = off (crisp/blocky).
CT_COLOR_SMOOTH_MM_DEFAULT = 0.4
CT_COLOR_SMOOTH_MM_MAX = 2.0


def load_ct_colormap() -> dict:
    """Return the global CT colour map as {"bands": [...], "opacity": float,
    "smooth_mm": float}, falling back to the factory default when absent/
    unreadable. *smooth_mm* is the spatial Gaussian strength in mm applied to
    the colour reslice (0 = crisp; ~0.4 = a gentle default that de-jaggs the
    band boundaries)."""
    default = {"bands": [dict(b) for b in CT_COLORMAP_DEFAULT_BANDS],
               "opacity": CT_COLORMAP_DEFAULT_OPACITY,
               "smooth_mm": CT_COLOR_SMOOTH_MM_DEFAULT}
    try:
        data = json.loads(CT_COLORMAP_PATH.read_text(encoding="utf-8"))
        bands = [sb for sb in (_sanitize_band(b)
                               for b in data.get("bands", [])) if sb]
        if not bands:
            return default
        op = float(data.get("opacity", CT_COLORMAP_DEFAULT_OPACITY))
        try:
            sm = float(data.get("smooth_mm", CT_COLOR_SMOOTH_MM_DEFAULT))
        except (TypeError, ValueError):
            sm = CT_COLOR_SMOOTH_MM_DEFAULT
        return {"bands": bands, "opacity": max(0.0, min(1.0, op)),
                "smooth_mm": max(0.0, min(CT_COLOR_SMOOTH_MM_MAX, sm))}
    except (OSError, ValueError, TypeError):
        return default


def save_ct_colormap(bands, opacity, smooth_mm=CT_COLOR_SMOOTH_MM_DEFAULT) -> None:
    """Best-effort persist of the global CT colour map. A failed write must
    not break the session."""
    try:
        try:
            sm = max(0.0, min(CT_COLOR_SMOOTH_MM_MAX, float(smooth_mm)))
        except (TypeError, ValueError):
            sm = CT_COLOR_SMOOTH_MM_DEFAULT
        out = {"bands": [sb for sb in (_sanitize_band(b) for b in bands) if sb],
               "opacity": max(0.0, min(1.0, float(opacity))),
               "smooth_mm": sm,
               "version": 1}
        # store rgb as a list (JSON has no tuple)
        for b in out["bands"]:
            b["rgb"] = list(b["rgb"])
        CT_COLORMAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        CT_COLORMAP_PATH.write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError):
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
