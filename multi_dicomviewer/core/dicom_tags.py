"""DICOM header tag enumeration + on-image overlay lines.

Feeds two UI consumers from one place so they always agree on tag names,
formatting, and anonymization:

  * the tag-selection dialog — the full header as a filterable table
  * the viewer overlay — only the user-picked keywords, as short lines

Both run masked values through :mod:`anonymize` so the Anonymize toggle
hides case-identifying fields here exactly as it does elsewhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from pydicom.tag import Tag

from .anonymize import mask_value

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
    #: Stable selection identifier used by the overlay/dialog: the pydicom
    #: keyword for standard tags, a private-creator key
    #: ``(GGGG,"CREATOR",EE)`` for private DATA elements (so the selection
    #: survives the per-file block-number reassignment that makes the raw
    #: (group,element) literal unstable across series), or the raw literal
    #: as a last resort. Empty only for back-compat default construction.
    ident: str = ""


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


def _private_ident(group: int, creator: str, offset: int) -> str:
    """Stable identifier for a private DATA element: group + private-creator
    name + element offset. Independent of the per-file block number."""
    return f'({group:04X},"{creator}",{offset:02X})'


_PRIV_IDENT_RE = re.compile(r'^\(([0-9A-Fa-f]{4}),"(.*)",([0-9A-Fa-f]{2})\)$')


def _parse_private_ident(s):
    """Parse ``(7005,"CREATOR",05)`` → (0x7005, "CREATOR", 0x05), else None."""
    if not isinstance(s, str):
        return None
    m = _PRIV_IDENT_RE.match(s)
    if m is None:
        return None
    return (int(m.group(1), 16), m.group(2), int(m.group(3), 16))


def _stable_ident(ds, elem, keyword: str, tag_literal: str) -> str:
    """The selection identifier for *elem*: keyword if standard, a
    private-creator key for a private data element whose creator we can
    resolve, else the raw (group,element) literal."""
    if keyword:
        return keyword
    g = elem.tag.group
    e = elem.tag.element
    # Private data elements live in odd groups with a non-zero block byte
    # (the high byte 0x10-0xFF); the matching private creator string sits
    # at (group, 0x00bb) where bb is that block byte.
    if g % 2 == 1 and (e >> 8) >= 0x10:
        block = e >> 8
        offset = e & 0xFF
        creator_tag = Tag(g, block)
        if creator_tag in ds:
            cv = ds[creator_tag].value
            if cv is not None:
                return _private_ident(g, str(cv).strip(), offset)
    return tag_literal


def upgrade_private_literal(ds, ident: str) -> str:
    """If *ident* is a raw private ``(group,element)`` literal resolvable in
    *ds*, return its stable private-creator key; otherwise return *ident*
    unchanged. Keywords and already-stable keys pass through. Lets a
    selection saved before the private-creator change self-migrate the
    moment the originating series is shown — no dialog re-confirm needed."""
    if not isinstance(ident, str) or _parse_private_ident(ident) is not None:
        return ident
    tag = _parse_tag_string(ident)
    if tag is None:
        return ident                      # pydicom keyword — leave as-is
    g, e = tag
    if g % 2 == 1 and (e >> 8) >= 0x10 and ds is not None:
        block = e >> 8
        offset = e & 0xFF
        creator_tag = Tag(g, block)
        if creator_tag in ds:
            cv = ds[creator_tag].value
            if cv is not None:
                return _private_ident(g, str(cv).strip(), offset)
    return ident


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
        value = mask_value(elem, _format_value(elem), anonymized)
        tag_literal = f"({elem.tag.group:04X},{elem.tag.element:04X})"
        rows.append(
            TagRow(
                tag=tag_literal,
                keyword=keyword,
                name=name,
                vr=str(elem.VR),
                value=value,
                ident=_stable_ident(ds, elem, keyword, tag_literal),
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
        value = mask_value(elem, _overlay_value(kw, elem, 64), anonymized)
        lines.append(f"{name}: {value}")
    return lines


def _parse_tag_string(s: str):
    """Parse '(0019,1099)' → (0x0019, 0x1099). None when *s* is not a
    tag-string literal — pydicom keywords ('PatientName' etc.) round-trip
    through the keyword path instead."""
    if not (isinstance(s, str) and s.startswith("(") and s.endswith(")")
            and "," in s):
        return None
    try:
        g, e = s[1:-1].split(",", 1)
        return (int(g, 16), int(e, 16))
    except ValueError:
        return None


def _resolve_private(ds, group: int, creator: str, offset: int):
    """Resolve a private DATA element by its creator + offset in *ds*,
    finding whatever block that creator currently occupies. Returns the
    element or None. This is what makes a private-tag selection survive
    the per-file block reassignment."""
    try:
        block = ds.private_block(group, creator)
    except (KeyError, ValueError):
        return None
    try:
        return block[offset]
    except (KeyError, ValueError):
        return None


def _lookup(ds, keyword: str):
    """Header element for *keyword*, or None if absent/empty.

    Accepts a pydicom keyword (``PatientName``), a stable private-creator
    identifier (``(7005,"CREATOR",05)``) resolved via the creator's current
    block, or a raw tag-string literal (``(0019,1099)``) — the last kept so
    selections saved before the private-creator change still load."""
    priv = _parse_private_ident(keyword)
    if priv is not None:
        return _resolve_private(ds, *priv)
    tag = _parse_tag_string(keyword)
    try:
        if tag is not None:
            if tag in ds:
                return ds[tag]
            return None
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
