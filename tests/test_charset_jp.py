"""Japanese PatientName / description mojibake repair.

Japanese XA / US / CT units routinely write patient names as raw Shift-JIS
(cp932) bytes while declaring an ISO-2022 character set in (0008,0005) — often
even the malformed defined term ``ISO 2022 IR87`` (no space). The bytes carry no
ISO-2022 ESC shifts, so pydicom keeps the ASCII G0 set and mangles the kanji
("文字化け"). These tests pin the two-pronged fix in :mod:`dicom_io`:

* ``_normalize_charset`` repairs the malformed defined term so genuinely
  escape-coded values still decode through pydicom; and
* ``decode_text`` decodes the original element bytes as cp932 when they are raw
  Shift-JIS mislabelled as ISO-2022.
"""
from pydicom.dataset import Dataset
from pydicom.dataelem import RawDataElement
from pydicom.tag import Tag

from multi_dicomviewer.core.dicom_io import (
    _decode_jp_bytes,
    _normalize_charset,
    decode_text,
    repair_dataset_text,
)

_SCS = Tag(0x00080005)
_PN = Tag(0x00100010)
_DESC = Tag(0x0008103E)  # SeriesDescription (LO)


def _ds_raw(vr: str, tag: Tag, raw: bytes, charset) -> Dataset:
    """A metadata dataset carrying *raw* undecoded bytes for *tag*, declaring
    *charset* in (0008,0005) — mimics a freshly-read pydicom dataset."""
    ds = Dataset()
    ds.add_new(_SCS, "CS", charset)
    ds[tag] = RawDataElement(tag, vr, len(raw), raw, 0, True, True)
    return ds


def test_raw_shift_jis_patient_name_recovered():
    # b'\x83V...' == cp932 'シモヤマダ　フタミ'; declared as the typo'd term.
    raw = b"\x83V\x83\x82\x83\x84\x83}\x83_\x81@\x83t\x83^\x83~"
    ds = _ds_raw("PN", _PN, raw,
                 ["ISO 2022 IR 6", "ISO 2022 IR 13", "ISO 2022 IR87"])
    _normalize_charset(ds)
    assert decode_text(ds, "PatientName") == "シモヤマダ　フタミ"


def test_pn_prefers_ideographic_group_over_garbled_alphabetic():
    # alphabetic=ideographic; alphabetic component is corrupt source bytes, the
    # ideographic (cp932) component is the readable kanji name.
    raw = b"GARBLE\x04!\\=" + "マエダ".encode("cp932")
    ds = _ds_raw("PN", _PN, raw, ["ISO 2022 IR 6", "ISO 2022 IR87"])
    assert decode_text(ds, "PatientName") == "マエダ"


def test_normalize_fixes_typo_for_escape_coded_value():
    # A GENUINE ISO-2022-JP value (has ESC shifts) declared with the typo'd
    # term: the cp932 fallback must NOT trigger; normalization lets pydicom
    # decode it correctly.
    raw = "山田^太郎".encode("iso2022_jp")
    assert b"\x1b" in raw  # sanity: real escape sequences present
    ds = _ds_raw("PN", _PN, raw, ["ISO 2022 IR 6", "ISO 2022 IR87"])
    _normalize_charset(ds)
    assert ds.SpecificCharacterSet == ["ISO 2022 IR 6", "ISO 2022 IR 87"]
    assert decode_text(ds, "PatientName") == "山田^太郎"


def test_ascii_value_is_untouched():
    raw = b"CAG Bip 15 fps Medium"
    ds = _ds_raw("LO", _DESC, raw, "ISO 2022 IR 6")
    assert decode_text(ds, "SeriesDescription") == "CAG Bip 15 fps Medium"


def test_shift_jis_description_recovered():
    raw = "左冠動脈造影".encode("cp932")
    ds = _ds_raw("LO", _DESC, raw, ["ISO 2022 IR 6", "ISO 2022 IR87"])
    _normalize_charset(ds)
    assert decode_text(ds, "SeriesDescription") == "左冠動脈造影"


def test_default_returned_for_absent_tag():
    ds = Dataset()
    ds.add_new(_SCS, "CS", "ISO 2022 IR 6")
    assert decode_text(ds, "PatientName", "Anonymous") == "Anonymous"


def test_normalize_is_noop_for_valid_charset():
    ds = Dataset()
    ds.add_new(_SCS, "CS", ["ISO 2022 IR 6", "ISO 2022 IR 87"])
    _normalize_charset(ds)
    assert ds.SpecificCharacterSet == ["ISO 2022 IR 6", "ISO 2022 IR 87"]


# --- _decode_jp_bytes: self-validating codec chain (Opt A) -------------------

def test_codec_chain_decodes_shift_jis():
    assert _decode_jp_bytes("マエダ".encode("cp932")) == "マエダ"


def test_codec_chain_decodes_utf8():
    # Genuine UTF-8 must be taken as UTF-8, not mangled through cp932.
    assert _decode_jp_bytes("マエダ".encode("utf-8")) == "マエダ"


def test_codec_chain_decodes_euc_jp():
    # EUC-JP is not valid UTF-8 and (here) not valid cp932, so the chain
    # reaches euc_jp.
    assert _decode_jp_bytes("漢字".encode("euc_jp")) == "漢字"


def test_codec_chain_shift_jis_is_not_misread_as_utf8():
    # The safety property: a Shift-JIS byte stream must NOT decode cleanly as
    # UTF-8 (which would silently produce wrong characters).
    raw = "シモヤマダ".encode("cp932")
    try:
        raw.decode("utf-8")
        misread = True
    except UnicodeDecodeError:
        misread = False
    assert misread is False
    assert _decode_jp_bytes(raw) == "シモヤマダ"


def test_codec_chain_truncated_bytes_degrade_gracefully():
    raw = "マエダ".encode("cp932")[:-1]  # drop last byte -> mid-character cut
    out = _decode_jp_bytes(raw)
    assert out.startswith("マエ")  # partial name still shown, no exception


# --- repair_dataset_text: dataset-wide repair (Opt B) ------------------------

def test_repair_dataset_fixes_all_text_tags_in_place():
    ds = Dataset()
    ds.add_new(_SCS, "CS", ["ISO 2022 IR 6", "ISO 2022 IR87"])
    pn = "シモヤマダ".encode("cp932")
    desc = "左冠動脈造影".encode("cp932")
    ds[_PN] = RawDataElement(_PN, "PN", len(pn), pn, 0, True, True)
    ds[_DESC] = RawDataElement(_DESC, "LO", len(desc), desc, 0, True, True)

    repair_dataset_text(ds)

    # str(value) — what the tag viewer renders — is now clean for every tag.
    assert str(ds.PatientName) == "シモヤマダ"
    assert str(ds.SeriesDescription) == "左冠動脈造影"


def test_repair_keeps_full_pn_value_not_just_one_component():
    # Unlike decode_text (display), the dataset repair must keep the element
    # verbatim — both PN groups, separated by '='.
    raw = "ﾔﾏﾀﾞ".encode("cp932") + b"=" + "山田".encode("cp932")
    ds = Dataset()
    ds.add_new(_SCS, "CS", ["ISO 2022 IR 6", "ISO 2022 IR87"])
    ds[_PN] = RawDataElement(_PN, "PN", len(raw), raw, 0, True, True)

    repair_dataset_text(ds)

    assert "=" in str(ds.PatientName)
    assert "山田" in str(ds.PatientName)


def test_repair_recurses_into_sequences():
    inner = Dataset()
    inner.add_new(_SCS, "CS", ["ISO 2022 IR 6", "ISO 2022 IR87"])
    name = "シモヤマダ".encode("cp932")
    inner[_PN] = RawDataElement(_PN, "PN", len(name), name, 0, True, True)
    outer = Dataset()
    outer.add_new(_SCS, "CS", ["ISO 2022 IR 6", "ISO 2022 IR87"])
    # ReferencedPatientSequence (0008,1120) is an SQ.
    outer.add_new(0x00081120, "SQ", [inner])

    repair_dataset_text(outer)

    assert str(outer[0x00081120].value[0].PatientName) == "シモヤマダ"


def test_repair_leaves_ascii_untouched():
    ds = Dataset()
    ds.add_new(_SCS, "CS", "ISO 2022 IR 6")
    raw = b"CAG Bip 15 fps Medium"
    ds[_DESC] = RawDataElement(_DESC, "LO", len(raw), raw, 0, True, True)
    repair_dataset_text(ds)
    assert str(ds.SeriesDescription) == "CAG Bip 15 fps Medium"
