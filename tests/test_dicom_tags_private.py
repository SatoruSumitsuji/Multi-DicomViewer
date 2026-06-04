"""Private-tag selection must survive the per-file block reassignment.

A private DATA element's (group,element) literal is NOT stable across
series: the same private creator can be granted a different block number
in another file, moving e.g. (7005,1005) to (7005,1105). Selecting by the
raw literal therefore silently drops the overlay on the next series. These
tests pin the fix: identify private tags by (group, creator, offset).
"""
from pydicom.dataset import Dataset

from multi_dicomviewer.core import dicom_tags as dt


def _ds_with_private(block: int, value: str) -> Dataset:
    """A dataset whose private creator 'ACME PRIVATE' occupies *block*,
    carrying one data element at offset 0x05."""
    ds = Dataset()
    ds.PatientName = "Test"
    group = 0x7005
    ds.add_new((group << 16) | block, "LO", "ACME PRIVATE")       # creator
    ds.add_new((group << 16) | (block << 8) | 0x05, "LO", value)  # data elem
    return ds


def _private_row(ds):
    return next(r for r in dt.iter_tag_rows(ds)
               if r.ident.startswith('(7005,"'))


def test_private_ident_is_block_independent():
    r10 = _private_row(_ds_with_private(0x10, "block10"))
    r11 = _private_row(_ds_with_private(0x11, "block11"))
    # Same creator + offset → identical stable identifier despite the
    # raw literal differing ((7005,1005) vs (7005,1105)).
    assert r10.ident == r11.ident == '(7005,"ACME PRIVATE",05)'
    assert r10.tag != r11.tag


def test_selection_resolves_across_blocks():
    ident = _private_row(_ds_with_private(0x10, "block10")).ident
    # The SAME selection must resolve in a series where the creator moved
    # to a different block.
    ds2 = _ds_with_private(0x11, "moved")
    elem = dt._lookup(ds2, ident)
    assert elem is not None
    assert elem.value == "moved"
    lines = dt.overlay_lines(ds2, [ident])
    assert any("moved" in ln for ln in lines)


def test_absent_creator_resolves_to_none():
    ident = '(7005,"NOT PRESENT",05)'
    ds = _ds_with_private(0x10, "x")
    assert dt._lookup(ds, ident) is None
    assert dt.overlay_lines(ds, [ident]) == []


def test_legacy_raw_literal_still_resolves_same_block():
    # Selections saved before the fix used the raw literal; they must keep
    # working for a series whose block matches.
    ds = _ds_with_private(0x10, "kept")
    elem = dt._lookup(ds, "(7005,1005)")
    assert elem is not None and elem.value == "kept"


def test_standard_keyword_unchanged():
    ds = Dataset()
    ds.PatientName = "Yamada"
    row = next(r for r in dt.iter_tag_rows(ds) if r.keyword == "PatientName")
    assert row.ident == "PatientName"
    assert dt._lookup(ds, "PatientName").value == "Yamada"


def test_upgrade_private_literal_self_heals():
    # A legacy raw literal saved from a block-0x10 series upgrades to the
    # stable key the moment that series is shown again.
    ds = _ds_with_private(0x10, "v")
    assert dt.upgrade_private_literal(ds, "(7005,1005)") == \
        '(7005,"ACME PRIVATE",05)'


def test_upgrade_passthrough_for_keyword_and_stable_and_unresolvable():
    ds = _ds_with_private(0x10, "v")
    # pydicom keyword — unchanged.
    assert dt.upgrade_private_literal(ds, "PatientName") == "PatientName"
    # already stable — unchanged.
    stable = '(7005,"ACME PRIVATE",05)'
    assert dt.upgrade_private_literal(ds, stable) == stable
    # literal whose block isn't present in this ds — left as-is (can't
    # resolve here; will upgrade when the matching series is shown).
    assert dt.upgrade_private_literal(ds, "(7005,1105)") == "(7005,1105)"
