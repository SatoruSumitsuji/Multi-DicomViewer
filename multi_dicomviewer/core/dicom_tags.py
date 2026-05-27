"""DICOM header tag enumeration + on-image overlay lines.

Feeds two UI consumers from one place so they always agree on tag names,
formatting, and anonymization:

  * the tag-selection dialog — the full header as a filterable table
  * the viewer overlay — only the user-picked keywords, as short lines

Both run masked values through :mod:`anonymize` so the Anonymize toggle
hides case-identifying fields here exactly as it does elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .anonymize import mask_text

# Bulk pixel/overlay payloads — never useful as text, sometimes huge.
_SKIP_KEYWORDS = frozenset(
    {
        "PixelData",
        "FloatPixelData",
        "DoubleFloatPixelData",
        "OverlayData",
        "EncapsulatedDocument",
        "SpectroscopyData",
    }
)
_BINARY_VRS = frozenset({"OB", "OW", "OF", "OD", "OL", "OV", "UN"})


@dataclass(frozen=True)
class TagRow:
    """One displayable header element."""

    tag: str        # "(0010,0010)"
    keyword: str     # pydicom keyword ("" for private/unknown)
    name: str        # human-readable element name
    vr: str
    value: str       # already anonymization-masked, display-ready


def _format_value(elem, max_len: int = 160) -> str:
    """Compact one-line display string for *elem*'s value."""
    vr = elem.VR
    if vr == "SQ":
        try:
            return f"<sequence: {len(elem.value)} item(s)>"
        except TypeError:
            return "<sequence>"
    if vr in _BINARY_VRS:
        try:
            return f"<binary: {len(elem.value)} bytes>"
        except TypeError:
            return "<binary>"
    try:
        text = str(elem.value)
    except Exception:
        return "<unreadable>"
    text = " ".join(text.split())  # collapse newlines/runs of whitespace
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def iter_tag_rows(ds, anonymized: bool = False) -> list[TagRow]:
    """Every top-level header element as a :class:`TagRow`, in tag order.

    Pixel/overlay bulk data and group-length elements are dropped; PHI
    values are replaced with the placeholder when *anonymized*.
    """
    if ds is None:
        return []
    rows: list[TagRow] = []
    for elem in ds:
        if elem.tag.element == 0x0000:  # group length — noise
            continue
        keyword = elem.keyword or ""
        if keyword in _SKIP_KEYWORDS:
            continue
        name = elem.name or keyword or "Unknown"
        value = mask_text(keyword, _format_value(elem), anonymized)
        rows.append(
            TagRow(
                tag=f"({elem.tag.group:04X},{elem.tag.element:04X})",
                keyword=keyword,
                name=name,
                vr=str(elem.VR),
                value=value,
            )
        )
    return rows


def _overlay_value(keyword: str, elem, max_len: int) -> str:
    """Per-keyword display formatting for the on-image overlay (the tag
    table keeps raw values). Falls through to the generic formatter for
    anything not specifically handled."""
    try:
        if keyword == "AcquisitionTime":
            # DICOM TM "HHMMSS.FFFFFF" — drop the fractional seconds.
            return str(elem.value).split(".", 1)[0]
        if keyword in ("PositionerPrimaryAngle",
                       "PositionerSecondaryAngle"):
            return str(int(round(float(elem.value))))
    except Exception:
        pass
    return _format_value(elem, max_len=max_len)


def overlay_lines(
    ds,
    keywords: Iterable[str],
    anonymized: bool = False,
) -> list[str]:
    """"Name: value" lines for *keywords* present in *ds*, in given order.

    Keywords absent from the header are skipped. Used for the corner
    overlay, so values are clipped shorter than in the table and a few
    keys get a cleaner display (integer angles, no fractional seconds).
    """
    if ds is None:
        return []
    lines: list[str] = []
    for kw in keywords:
        elem = _lookup(ds, kw)
        if elem is None:
            continue
        name = elem.name or kw
        value = mask_text(kw, _overlay_value(kw, elem, 64), anonymized)
        lines.append(f"{name}: {value}")
    return lines


def _lookup(ds, keyword: str):
    """Header element for *keyword*, or None if absent/empty."""
    try:
        if keyword not in ds:
            return None
        elem = ds[keyword]
    except (KeyError, TypeError):
        return None
    return elem


def default_overlay_keywords(ds) -> list[str]:
    """A sensible starter selection from those present in *ds*."""
    preferred = (
        "PatientName",
        "PatientID",
        "StudyDate",
        "Modality",
        "SeriesNumber",
        "SeriesDescription",
    )
    return [kw for kw in preferred if _lookup(ds, kw) is not None]


def first_present(ds, keywords: Iterable[str]) -> Optional[str]:
    """First keyword in *keywords* that exists in *ds* (helper for tests)."""
    for kw in keywords:
        if _lookup(ds, kw) is not None:
            return kw
    return None
