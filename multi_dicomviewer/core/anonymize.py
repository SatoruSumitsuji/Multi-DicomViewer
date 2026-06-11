"""Anonymization profile shared by the on-screen Anonymize toggle AND the
"Export (Anon DICOM)" writer, so both blank exactly the same fields.

The profile is a set of DICOM tags (group, element) whose values are
"emptified" (value cleared, tag kept) plus a flag to emptify every private
(odd-group) element's value. The same profile drives:

  * display masking — when the Anonymize toggle is on, the viewer overlay
    and the tag-list dialog show the placeholder in place of a profile tag's
    value (pixels are never touched);
  * export — :func:`deidentify_dataset` writes a copy with those values
    emptied.

UIDs, the pixel module (group 0028 / 7FE0) and the transfer syntax are never
in the default profile, so anonymized files still decode and display.
"""
from __future__ import annotations

from typing import Iterable

from pydicom.tag import Tag

# Shown in place of a masked value. Kept short so it fits a corner overlay.
ANON_PLACEHOLDER = "(anonymized)"

#: All tags offered in the Anonymize-settings dialog, in display order.
#: Two are listed but OFF by default:
#:  * Modality (0008,0060) — not PHI; required so a re-opened file routes to
#:    the right viewer (a blank Modality would show a CT as a generic cine).
#:  * Specific Character Set (0008,0005) — emptying it can mangle any
#:    remaining multibyte (e.g. Japanese) text on save.
ANON_TAG_CATALOG: tuple[tuple[int, int], ...] = (
    (0x0002, 0x0016),  # Source Application Entity Title (file meta)
    (0x0008, 0x0005),  # Specific Character Set  (default OFF — see note)
    (0x0008, 0x0008),  # Image Type
    (0x0008, 0x0012),  # Instance Creation Date
    (0x0008, 0x0020),  # Study Date
    (0x0008, 0x0021),  # Series Date
    (0x0008, 0x0022),  # Acquisition Date
    (0x0008, 0x0023),  # Content Date
    (0x0008, 0x0050),  # Accession Number
    (0x0008, 0x0060),  # Modality  (default OFF — see note above)
    (0x0008, 0x0070),  # Manufacturer
    (0x0008, 0x0080),  # Institution Name
    (0x0008, 0x0081),  # Institution Address
    (0x0008, 0x0090),  # Referring Physician's Name
    (0x0008, 0x1010),  # Station Name
    (0x0008, 0x1030),  # Study Description
    (0x0008, 0x1040),  # Institutional Department Name
    (0x0008, 0x1090),  # Manufacturer's Model Name
    (0x0010, 0x0000),  # Group Length (auto-managed; emptied if present)
    (0x0010, 0x0010),  # Patient's Name
    (0x0010, 0x0020),  # Patient ID
    (0x0010, 0x0030),  # Patient's Birth Date
    (0x0010, 0x0040),  # Patient's Sex
    (0x0010, 0x1010),  # Patient's Age
    (0x0010, 0x1030),  # Patient's Weight
    (0x0020, 0x0010),  # Study ID
)

#: Tags emptified out of the box — everything in the catalog except Modality
#: (0008,0060) and Specific Character Set (0008,0005), which stay selectable
#: but OFF by default (see the catalog note above).
DEFAULT_ANON_TAGS: frozenset[tuple[int, int]] = (
    frozenset(ANON_TAG_CATALOG) - {(0x0008, 0x0060), (0x0008, 0x0005)}
)
DEFAULT_EMPTIFY_PRIVATE = True

# ---- current profile (process-global; loaded from settings at startup) ----
_anon_tags: set[tuple[int, int]] = set(DEFAULT_ANON_TAGS)
_emptify_private: bool = DEFAULT_EMPTIFY_PRIVATE


def set_anon_profile(
    tags: Iterable[tuple[int, int]], emptify_private: bool
) -> None:
    """Replace the active profile (called by the shell after the settings
    dialog or on startup from saved settings)."""
    global _anon_tags, _emptify_private
    _anon_tags = {(int(g), int(e)) for g, e in tags}
    _emptify_private = bool(emptify_private)


def get_anon_profile() -> tuple[set[tuple[int, int]], bool]:
    """Current (tag set, emptify_private)."""
    return set(_anon_tags), _emptify_private


def _is_private(group: int) -> bool:
    """Private data lives in odd-numbered groups."""
    return group % 2 == 1


def should_anonymize(group: int, element: int) -> bool:
    """True if a tag's value should be blanked under the active profile."""
    if (group, element) in _anon_tags:
        return True
    return _emptify_private and _is_private(group)


def mask_value(elem, value: str, anonymized: bool) -> str:
    """Display value, or the placeholder when *anonymized* and *elem* is in
    the profile. Used by the overlay / tag-list / CSV paths, which all have
    the element in hand."""
    if anonymized and should_anonymize(elem.tag.group, elem.tag.element):
        return ANON_PLACEHOLDER
    return value


def is_phi(keyword: str) -> bool:
    """True if *keyword* maps to a tag in the active profile. Kept for
    keyword-based callers (study tree / titles)."""
    from pydicom.datadict import tag_for_keyword
    t = tag_for_keyword(keyword) if keyword else None
    if t is None:
        return False
    return should_anonymize((t >> 16) & 0xFFFF, t & 0xFFFF)


def mask_text(keyword: str, value: str, anonymized: bool) -> str:
    """Keyword-based masking (profile-aware) for callers without the element
    in hand. Non-profile fields and the un-anonymized state pass through."""
    if anonymized and is_phi(keyword):
        return ANON_PLACEHOLDER
    return value


# ----------------------------------------------------------------- export
_BYTES_VRS = frozenset({"OB", "OW", "OF", "OD", "OL", "OV", "UN"})
_NUMERIC_VRS = frozenset(
    {"US", "SS", "UL", "SL", "FL", "FD", "AT", "UV", "SV", "US or SS"}
)


def _emptify_elem(elem) -> None:
    """Clear *elem*'s value to a zero-length value appropriate for its VR
    (the tag stays present). Best-effort; never raises."""
    vr = str(getattr(elem, "VR", ""))
    try:
        if vr == "SQ":
            elem.value = []
        elif vr in _BYTES_VRS:
            elem.value = b""
        elif vr in _NUMERIC_VRS:
            elem.value = None
        else:
            elem.value = ""
    except Exception:
        try:
            elem.value = None
        except Exception:
            pass


def deidentify_dataset(ds) -> None:
    """Emptify (in place) every profile tag's value and, when enabled, every
    private element's value. The tags stay present; UIDs, the pixel module
    and the transfer syntax are untouched, so the file still decodes.

    Group 0002 tags are handled in ``ds.file_meta``."""
    tags, emptify_private = get_anon_profile()
    fmeta = getattr(ds, "file_meta", None)
    for (g, e) in tags:
        target = fmeta if (g == 0x0002 and fmeta is not None) else ds
        if target is None:
            continue
        t = Tag(g, e)
        if t in target:
            _emptify_elem(target[t])
    if emptify_private:
        for elem in ds:
            if _is_private(elem.tag.group):
                _emptify_elem(elem)
