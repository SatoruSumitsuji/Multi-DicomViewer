"""DICOM folder scanning, study-tree building, and pixel access.

Indexing reads metadata only (stop_before_pixels) so a large folder loads
fast; pixel data is pulled lazily when a series is actually opened.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pydicom
from pydicom.encaps import get_frame
from pydicom.errors import InvalidDicomError
from pydicom.pixels import apply_color_lut, convert_color_space, pixel_array

from .study_model import Modality, Patient, Series, Study

# Japanese modalities often declare the malformed SpecificCharacterSet defined
# term "ISO 2022 IR87" (missing the space before the number). pydicom warns
# about it from inside dcmread — before we get a chance to repair it — for every
# such file. We deliberately handle this case (see _normalize_charset /
# decode_text below), so silence just that one warning to keep logs clean.
warnings.filterwarnings(
    "ignore",
    message=r"Unknown encoding 'ISO 2022 IR\d+'",
    category=UserWarning,
    module="pydicom.charset",
)


def _warn(msg: str) -> None:
    """Best-effort stderr log. Under pythonw / PyInstaller --windowed
    the process has no console attached and ``sys.stderr`` is None, so
    a direct ``sys.stderr.write(...)`` raises ``'NoneType' has no
    attribute 'write'``. Guard against that and swallow any I/O error
    — diagnostics must never break a load."""
    s = sys.stderr
    if s is None:
        return
    try:
        s.write(msg)
        if not msg.endswith("\n"):
            s.write("\n")
    except Exception:
        pass


def _safe(ds, tag, default=""):
    return getattr(ds, tag, default) or default


# --- Japanese character-set repair -------------------------------------------
# Many Japanese XA / US / CT units write patient names and descriptions as raw
# Shift-JIS (cp932) bytes while declaring an ISO-2022 character set in (0008,0005)
# SpecificCharacterSet — sometimes even the mistyped defined term "ISO 2022 IR87"
# (missing the space). Those bytes carry NO ISO-2022 ESC (0x1B) shifts, so
# pydicom keeps the 7-bit ASCII G0 set and mangles every kanji byte ("文字化け").
# Repair is three-fold: normalise the defined term so genuinely escape-coded
# fields still decode (and pydicom stops warning); decode the original element
# bytes ourselves with a self-validating codec chain (so UTF-8 / EUC-JP files are
# also covered, not just Shift-JIS); and apply that to the WHOLE dataset on read
# so every consumer (study tree, tag viewer, overlay, export filenames) sees
# clean text, not just the patient tree.

_IR_TERM_RE = re.compile(r"^ISO 2022 IR\s*(\d+)$")

#: DICOM string VRs that can carry free-text / person-name Japanese. Numeric,
#: date/time, UID and binary VRs are never re-decoded.
_JP_TEXT_VRS = frozenset({"PN", "LO", "LT", "SH", "ST", "UT", "UC"})


def _decode_jp_bytes(raw: bytes) -> str:
    """Decode mis-encoded Japanese DICOM bytes with a self-validating codec
    chain: UTF-8, then Shift-JIS (cp932), then EUC-JP — returning the first that
    decodes WITHOUT error.

    Order matters and makes this safe: UTF-8 is self-validating, so a Shift-JIS
    byte stream (0x80–0x9F lead bytes appear where UTF-8 expects 0xC0+ leads)
    almost never decodes cleanly as UTF-8 and falls through to cp932; conversely
    genuine UTF-8 is accepted up front. Only when every strict decode fails
    (truncated / corrupt source bytes) do we fall back to a lossy cp932 decode so
    a partial name is still shown rather than nothing."""
    for enc in ("utf-8", "cp932", "euc_jp"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("cp932", errors="replace")


def _normalize_charset(ds) -> None:
    """Fix a malformed (0008,0005) defined term in place — e.g.
    'ISO 2022 IR87' -> 'ISO 2022 IR 87'. Must run before any text element is
    decoded so pydicom uses the corrected term (no-op for valid datasets)."""
    try:
        scs = ds.get("SpecificCharacterSet")
    except Exception:
        return
    if not scs:
        return
    items = [scs] if isinstance(scs, str) else list(scs)
    fixed, changed = [], False
    for term in items:
        s = str(term).strip()
        m = _IR_TERM_RE.match(s)
        ns = f"ISO 2022 IR {m.group(1)}" if m else s
        changed = changed or (ns != s)
        fixed.append(ns)
    if changed:
        ds.SpecificCharacterSet = fixed if len(fixed) > 1 else fixed[0]


def _is_raw_sjis(raw: bytes) -> bool:
    """True when *raw* carries 8-bit (kanji) bytes but none of the ISO-2022 ESC
    (0x1B) shifts a real 'ISO 2022 IR 87/159' value must use — the fingerprint
    of raw Shift-JIS mislabelled as an ISO-2022 character set."""
    return 0x1B not in raw and any(b >= 0x80 for b in raw)


def _best_pn_part(text: str) -> str:
    """A PN value is alphabetic=ideographic=phonetic. Show the group richest in
    CJK characters (the kanji/kana name a clinician reads), else the first
    non-empty group."""
    groups = [g for g in text.split("=") if g.strip()]
    if len(groups) <= 1:
        return text.replace("=", " ").strip()
    cjk = lambda s: sum(ord(ch) >= 0x3000 for ch in s)  # noqa: E731
    best = max(groups, key=cjk) if any(cjk(g) for g in groups) else groups[0]
    return best.strip()


def decode_text(ds, tag, default="") -> str:
    """Read a DICOM text/PN element for DISPLAY, repairing mis-encoded Japanese
    on the original element bytes and, for PN, showing the most readable name
    component. Correctly-encoded Western names and genuinely escape-coded
    Japanese names fall through to pydicom's own decode unchanged."""
    try:
        item = ds.get_item(tag)
    except Exception:
        item = None
    raw = getattr(item, "value", None) if item is not None else None
    if isinstance(raw, (bytes, bytearray)) and _is_raw_sjis(bytes(raw)):
        txt = _decode_jp_bytes(bytes(raw)).rstrip("\x00 ").strip()
        if getattr(item, "VR", "") == "PN":
            txt = _best_pn_part(txt)
        return txt or default
    return str(_safe(ds, tag, default)) or default


def repair_dataset_text(ds) -> None:
    """Repair mis-encoded Japanese text across the WHOLE dataset, in place.

    Normalises the (0008,0005) defined term, then rewrites every text/PN element
    whose original bytes are raw Shift-JIS / UTF-8 / EUC-JP mislabelled as
    ISO-2022 (the :func:`_is_raw_sjis` fingerprint) with its correctly-decoded
    value, recursing into sequences. Applied once at read time so the tag viewer,
    overlay, export filenames and anything else reading the dataset all show
    clean text — pure-ASCII and genuinely escape-coded values are left untouched.

    Unlike :func:`decode_text` this keeps the full faithful PN value (all
    alphabetic=ideographic=phonetic groups) rather than picking one component,
    because a tag inspector should show the element verbatim."""
    _normalize_charset(ds)
    for tag in list(ds._dict):  # raw items only; get_item never converts/caches
        try:
            item = ds.get_item(tag)
        except Exception:
            continue
        vr = getattr(item, "VR", "")
        if vr == "SQ":
            try:
                for sub in ds[tag].value:
                    repair_dataset_text(sub)
            except Exception:
                pass
            continue
        if vr not in _JP_TEXT_VRS:
            continue
        raw = getattr(item, "value", None)
        if isinstance(raw, (bytes, bytearray)) and _is_raw_sjis(bytes(raw)):
            txt = _decode_jp_bytes(bytes(raw)).rstrip("\x00 ")
            try:
                ds[tag] = pydicom.DataElement(item.tag, vr, txt)
            except Exception:
                pass


def _acq_key(ds) -> str:
    """A lexically-sortable acquisition timestamp for a series. DICOM TM
    is zero-padded HHMMSS[.ffffff], so string compare is chronological."""
    dt = str(_safe(ds, "AcquisitionDateTime")).strip()
    if dt:
        return dt
    d = str(_safe(ds, "AcquisitionDate") or _safe(ds, "ContentDate")).strip()
    t = str(_safe(ds, "AcquisitionTime") or _safe(ds, "ContentTime")).strip()
    return d + t


def _to_float(value, default=None):
    """float(value) but tolerant: empty/None/garbage DICOM elements (a
    common cause of 'float() argument must be ... not NoneType') yield
    *default* instead of raising."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_num(value, default=None):
    """First numeric value of a DICOM element that may be single OR multi-
    valued (WindowCenter/WindowWidth are often a MultiValue like [40, 400])."""
    if value is None:
        return default
    if not isinstance(value, (str, bytes)) and hasattr(value, "__iter__"):
        try:
            value = next(iter(value))
        except StopIteration:
            return default
    return _to_float(value, default)


#: SOP Class UID prefix shared by all Secondary Capture variants (single- and
#: multi-frame, grayscale and colour). GE et al. store dose-report sheets,
#: "electronic film" screen captures and 3-D render snapshots as Secondary
#: Capture but tag them Modality=CT, so they would otherwise land in the CT
#: viewer. They are NOT Hounsfield data (and are often RGB), so load_series
#: routes them to the image viewer with auto W/L + colour (see
#: load_secondary_capture); load_ct keeps a defensive auto-window fallback too.
_SECONDARY_CAPTURE_PREFIX = "1.2.840.10008.5.1.4.1.1.7"


def _series_is_secondary_capture(series) -> bool:
    """True if *series*'s first instance is a Secondary Capture object (read
    from the header only — cheap)."""
    files = getattr(series, "files", None) or []
    if not files:
        return False
    ds = _read_header(files[0])
    return ds is not None and str(
        getattr(ds, "SOPClassUID", "")).startswith(_SECONDARY_CAPTURE_PREFIX)


def _is_broken_pseudo_color(rgb: np.ndarray) -> bool:
    """True if an RGB frame is a BROKEN pseudo-colour capture: some vendors
    (e.g. GE) store YBR / degenerate chroma under PI=RGB, so every decoder
    renders the image with a saturated green/magenta cast — those should be
    shown as grayscale. A GENUINE colour render (e.g. Fujifilm Synapse 3-D VR)
    has a near-neutral background, so it stays in colour.

    Discriminator: the border ring is almost always background. A real render's
    border is black/neutral (low saturation); a broken capture's border is
    highly saturated (the green cast). Colour-agnostic (catches green & magenta)."""
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return False
    a = rgb[..., :3].astype(np.float32)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]], axis=0)
    mx = border.max(1)
    mn = border.min(1)
    sat = np.where(mx > 1.0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    return float(np.median(sat)) > 0.4


def _to_gray2d(arr: np.ndarray) -> np.ndarray:
    """Reduce a decoded frame to a 2-D grayscale array. RGB/RGBA screen
    captures (e.g. an XA radiation-dose summary page) collapse to
    luminance so the grayscale canvas can still show them."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):  # (H, W, C)
        return arr[..., :3].mean(axis=-1)
    if arr.ndim == 3:                               # unexpected (1, H, W)
        return arr.reshape(arr.shape[-2], arr.shape[-1])
    if arr.ndim == 4:                               # (frames, H, W, C)
        return arr[0, ..., :3].mean(axis=-1)
    return arr


def _read_header(path: str):
    """Metadata-only dataset of *path* (no pixel data), or None on failure.

    Japanese text is repaired dataset-wide here so every consumer of the header
    (viewer overlay, tag dialog, export) sees clean text, not just the tree."""
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        repair_dataset_text(ds)
        return ds
    except Exception:
        return None


def scan_folder(
    root: str, progress: Optional[Callable[[int, int], None]] = None
) -> dict[str, Patient]:
    """Walk *root* recursively and build a Patient/Study/Series tree.

    *progress*, if given, is called as progress(done, total) while the
    files are read (total known up-front from a fast directory walk).
    """
    all_files = [
        os.path.join(dp, fn)
        for dp, _d, files in os.walk(root)
        for fn in files
    ]
    return _build_tree(all_files, progress)


def index_files(
    paths: list[str], progress: Optional[Callable[[int, int], None]] = None
) -> dict[str, Patient]:
    """Build a Patient/Study/Series tree from EXACTLY the given files.

    Unlike :func:`scan_folder`, a file's parent directory is NOT expanded —
    only the listed files are indexed. Used when the user drags individual
    DICOM file(s) onto the app (vs dropping a folder, which loads everything
    in it). Directories in *paths* are ignored (a folder drop takes the
    scan_folder path instead).
    """
    all_files = [p for p in paths if os.path.isfile(p)]
    return _build_tree(all_files, progress)


def _build_tree(
    all_files: list[str],
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Patient]:
    """Read *all_files* and assemble the Patient/Study/Series tree. Shared by
    scan_folder (recursive walk) and index_files (explicit file list)."""
    patients: dict[str, Patient] = {}
    frames_by_path: dict[str, int] = {}   # NumberOfFrames per file, for counts

    total = len(all_files)
    if progress is not None:
        progress(0, total)

    for idx, path in enumerate(all_files, 1):
        if progress is not None and (idx % 25 == 0 or idx == total):
            progress(idx, total)
        if True:
            try:
                ds = pydicom.dcmread(
                    path, stop_before_pixels=True, force=False
                )
            except (InvalidDicomError, IsADirectoryError, PermissionError):
                continue
            except Exception:
                # Not a DICOM file — skip silently during a folder scan.
                continue

            if not hasattr(ds, "SOPInstanceUID"):
                continue

            _normalize_charset(ds)
            pid = str(_safe(ds, "PatientID", "UNKNOWN"))
            pname = decode_text(ds, "PatientName", "Anonymous")
            patient = patients.setdefault(pid, Patient(pid, pname))

            st_uid = str(_safe(ds, "StudyInstanceUID", "NO_STUDY"))
            study = patient.studies.setdefault(
                st_uid,
                Study(
                    study_uid=st_uid,
                    description=decode_text(ds, "StudyDescription"),
                    date=str(_safe(ds, "StudyDate")),
                ),
            )

            se_uid = str(_safe(ds, "SeriesInstanceUID", "NO_SERIES"))
            series = study.series.get(se_uid)
            if series is None:
                num = _safe(ds, "SeriesNumber", None)
                anum = _safe(ds, "AcquisitionNumber", None)
                ino = _safe(ds, "InstanceNumber", None)
                raw_mod = str(_safe(ds, "Modality")).strip().upper()
                series = Series(
                    series_uid=se_uid,
                    modality=Modality.from_dicom(raw_mod),
                    description=decode_text(ds, "SeriesDescription"),
                    number=int(num) if str(num).strip().isdigit() else None,
                    acq_number=(
                        int(anum) if str(anum).strip().lstrip("-").isdigit()
                        else None
                    ),
                    instance_number=(
                        int(ino) if str(ino).strip().lstrip("-").isdigit()
                        else None
                    ),
                    dicom_modality=raw_mod,
                    acq_time=_acq_key(ds),
                )
                study.series[se_uid] = series
            series.files.append(path)
            frames_by_path[path] = int(
                _to_float(getattr(ds, "NumberOfFrames", 1), 1) or 1
            )

    _merge_studyuid_duplicate_patients(patients)
    _split_packed_xa_series(patients)
    _merge_cross_uid_biplane(patients)
    # Tree "N img" = total frames, so a single-file multi-frame series (NM/US/XA
    # cine) reports its frame count instead of "1 img". Done after split/merge
    # so each restructured series sums the frames of the files it ended up with;
    # the per-file frame counts were captured during the scan above (no extra
    # header reads).
    for patient in patients.values():
        for study in patient.studies.values():
            for se in study.series.values():
                se.n_images = sum(frames_by_path.get(f, 1) for f in se.files)
    return patients


def _patient_file_count(pat: Patient) -> int:
    return sum(len(se.files) for st in pat.studies.values()
               for se in st.series.values())


def _merge_studyuid_duplicate_patients(patients: dict[str, Patient]) -> None:
    """Fuse patient nodes that share a StudyInstanceUID into one.

    A DICOM StudyInstanceUID is globally unique to one study of one patient, so
    two Patient nodes carrying the SAME study UID are the same person — the
    split is a data error. The usual cause is a single file in a series whose
    PatientID / PatientName bytes were truncated or mangled by the modality
    (e.g. an Iwaki XA unit that wrote a 9-digit PatientID and a cut-off cp932
    name on its first cine while the rest of the study has the correct 10-digit
    ID). Without this, that one clip would hang off a separate, garbled patient
    node even though it is a real acquisition of the same study.

    The merge is loss-free: every series/file is moved onto the surviving node;
    nothing is dropped. The surviving identity is the one with the cleanest name
    (fewest U+FFFD decode-failure marks), then the most files, then the longest
    PatientID — i.e. the intact header wins over the truncated one.
    """
    # Union pids that are connected through any shared study UID.
    parent = {pid: pid for pid in patients}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    study_pids: dict[str, list[str]] = {}
    for pid, pat in patients.items():
        for su in pat.studies:
            study_pids.setdefault(su, []).append(pid)
    for pids in study_pids.values():
        for other in pids[1:]:
            parent[find(other)] = find(pids[0])

    groups: dict[str, list[str]] = {}
    for pid in list(patients):
        groups.setdefault(find(pid), []).append(pid)

    for members in groups.values():
        if len(members) < 2:
            continue
        canon = max(
            members,
            key=lambda p: (
                -patients[p].name.count("�"),   # cleanest name first
                _patient_file_count(patients[p]),    # then most complete
                len(p),                              # then longest PatientID
            ),
        )
        target = patients[canon]
        for pid in members:
            if pid == canon:
                continue
            src = patients.pop(pid)
            for su, study in src.studies.items():
                tgt_study = target.studies.get(su)
                if tgt_study is None:
                    target.studies[su] = study
                    continue
                for se_uid, se in study.series.items():
                    existing = tgt_study.series.get(se_uid)
                    if existing is None:
                        tgt_study.series[se_uid] = se
                    else:  # same series split across the two nodes — merge files
                        for f in se.files:
                            if f not in existing.files:
                                existing.files.append(f)


def _split_packed_xa_series(patients: dict[str, Patient]) -> None:
    """Some vendors (notably Philips Allura/Azurion) store every cine of a
    study under a single SeriesInstanceUID and only distinguish them via
    InstanceNumber — and ultrasound routinely packs ALL its clips (often
    dozens) into one US series the same way. Other DICOM viewers show one row
    per instance, so do we, by splitting such series into one row per file.

    Threshold differs by modality: XA keeps 1-file (single) and 2-file
    (biplane) series intact and only splits ≥ 3 files (packed); US / OTHER /
    IVUS have no biplane concept (each file is an independent cine), so they
    split as soon as there are ≥ 2 files. Without this, load_xa would treat
    the N files as N "planes" of one acquisition — the viewer then shows only
    one and sticks on "Buffering…".

    Each split row gets the file's own ``InstanceNumber`` promoted to
    ``Series.number`` (so the tree column shows distinct numbers) and
    its own AcquisitionDateTime/SeriesDescription read fresh from that
    file. The synthesized SeriesInstanceUID is ``"<orig>#<n>"`` so
    downstream code (multisync state, remove_node, dedup) keeps working
    with a single string key.
    """
    for patient in patients.values():
        for study in patient.studies.values():
            new_series: dict[str, Series] = {}
            for uid, se in study.series.items():
                # XA: 2 files = biplane (keep), ≥3 = packed (split). US/OTHER/
                # IVUS have no biplane concept, so split at ≥2 files. CT and
                # anything else are left intact. MR/NM are EXCLUDED from the
                # split: their multi-file series are a single cine / stack (one
                # frame per file) and must stay together so they play as a 2-D
                # cine (load_series routes them to load_secondary_capture), not
                # as N separate single-image rows.
                _kind = (se.dicom_modality or "").upper()
                _splittable = (
                    se.modality in (Modality.XA, Modality.IVUS, Modality.OTHER)
                    and _kind not in ("MR", "NM")
                )
                _min_files = 3 if se.modality == Modality.XA else 2
                if not _splittable or len(se.files) < _min_files:
                    new_series[uid] = se
                    continue
                # Read each packed file's header ONCE to split into per-file
                # rows. Costly only for these unusual series; common
                # (biplane / single) series skip this entirely.
                headers = [(path, _read_header(path)) for path in se.files]
                # Tree "No" column for the split rows: promote each file's
                # InstanceNumber ONLY when those are DISTINCT across the
                # group — the vendor's per-instance numbering (Philips Allura
                # / US pack many cines under one SeriesUID, numbered per
                # instance). When they are NOT distinct (several cines all
                # tagged the same InstanceNumber, e.g. all "#1", or none at
                # all) promoting it would label every row "#1" and HIDE the
                # real SeriesNumber, so keep the real SeriesNumber instead.
                def _ino_of(ds):
                    if ds is None:
                        return None
                    v = getattr(ds, "InstanceNumber", None)
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        return None
                distinct_ino = len(
                    {_ino_of(ds) for _p, ds in headers if _ino_of(ds) is not None}
                ) > 1
                for idx, (path, ds) in enumerate(headers):
                    if ds is None:
                        # Unreadable: keep the original group key so the
                        # file isn't silently dropped from the tree.
                        new_series[uid] = se
                        continue
                    ino_int: Optional[int] = _ino_of(ds)
                    anum = getattr(ds, "AcquisitionNumber", None)
                    try:
                        anum_int: Optional[int] = (
                            int(anum) if anum is not None else None
                        )
                    except (TypeError, ValueError):
                        anum_int = None
                    sub_key = ino_int if ino_int is not None else (idx + 1)
                    new_uid = f"{uid}#{sub_key}"
                    # Collision guard (two files with the same
                    # InstanceNumber under the same series): append the
                    # path so neither is lost.
                    if new_uid in new_series:
                        new_uid = f"{new_uid}@{idx}"
                    desc = decode_text(ds, "SeriesDescription", "") or se.description
                    new_series[new_uid] = Series(
                        series_uid=new_uid,
                        modality=se.modality,
                        description=desc,
                        files=[path],
                        # Promote InstanceNumber into the Series No column
                        # (how other viewers label packed series) when it
                        # actually discriminates the rows; otherwise fall
                        # back to the real SeriesNumber so the tree matches
                        # the number burned into the image.
                        number=(ino_int if (distinct_ino and ino_int is not None)
                                else se.number),
                        acq_number=anum_int,
                        instance_number=ino_int,
                        dicom_modality=se.dicom_modality,
                        acq_time=_acq_key(ds),
                    )
            study.series = new_series


def _merge_cross_uid_biplane(patients: dict[str, Patient]) -> None:
    """Fuse XA biplane pairs that the vendor split across **different**
    SeriesInstanceUIDs.

    Standard biplane (e.g. the 20260513 dataset) shares one SeriesUID
    between the two planes, so the existing per-SeriesUID grouping
    already produces ``len(files) == 2`` → the viewer treats it as
    biplane. Some other vendors (e.g. the 20260401 dataset) instead
    give EACH plane its own SeriesUID; index_folder ends up with two
    single-file XA series that the viewer would show as singles.

    This pass detects those split pairs by:
      * same StudyInstanceUID (per-study scoping below)
      * both series are XA modality with exactly 1 file each
      * same AcquisitionDateTime (millisecond precision)
      * same InstanceNumber
      * resulting group is exactly 2 series (never 3+)
      * PositionerPrimaryAngle differs between the two files
        (true biplane = two genuinely different views)
      * Rows × Columns match exactly (biplane records both planes at
        the same detector geometry; this catches and rejects the
        "cine + stitched panorama" pseudo-pairs vendors sometimes
        emit, where the stitch's image dimensions differ wildly)

    NOTE: a NumberOfFrames-equality check was tried first but real
    biplane often has small differences (one plane stopping a frame
    or two earlier), so size-equality is used instead — it is just
    as discriminating against false pairs and never rejects a true
    biplane.

    When all checks pass the second file is appended to the first
    Series' file list and the second Series entry is dropped — the
    surviving Series now has 2 files and load_xa renders it as biplane
    (Front / Lateral) automatically.

    Conservative by design: any check that fails leaves the two
    series untouched, so the worst case is "still shown as singles"
    (the pre-fix behaviour), not "wrong files glued together".
    """
    for patient in patients.values():
        for study in patient.studies.values():
            # Bucket XA single-file series by their identity key. Skip
            # any series that lacks the discriminator fields.
            buckets: dict[tuple, list[tuple[str, Series]]] = {}
            for uid, se in study.series.items():
                if (se.modality != Modality.XA
                        or len(se.files) != 1
                        or not se.acq_time
                        or se.instance_number is None):
                    continue
                key = (se.acq_time, int(se.instance_number))
                buckets.setdefault(key, []).append((uid, se))

            for _key, members in buckets.items():
                if len(members) != 2:
                    # Singletons stay singles; ≥3 with the same key are
                    # suspicious data, never auto-merged.
                    continue
                (uid_a, se_a), (uid_b, se_b) = members
                ds_a = _read_header(se_a.files[0])
                ds_b = _read_header(se_b.files[0])
                if ds_a is None or ds_b is None:
                    continue
                pa_a = _to_float(
                    getattr(ds_a, "PositionerPrimaryAngle", None)
                )
                pa_b = _to_float(
                    getattr(ds_b, "PositionerPrimaryAngle", None)
                )
                if pa_a is None or pa_b is None:
                    continue
                # 1° is well above noise yet small enough not to reject
                # genuine narrow-separation biplanes.
                if abs(pa_a - pa_b) < 1.0:
                    continue
                # Image-size equality is the strongest signal we have
                # that the two files were shot on the same biplane
                # gantry rather than being a cine + a vendor-built
                # stitched panorama (which differs by a factor of ~10
                # in both rows and columns).
                rows_a = int(_to_float(getattr(ds_a, "Rows", 0), 0))
                cols_a = int(_to_float(getattr(ds_a, "Columns", 0), 0))
                rows_b = int(_to_float(getattr(ds_b, "Rows", 0), 0))
                cols_b = int(_to_float(getattr(ds_b, "Columns", 0), 0))
                if rows_a == 0 or rows_b == 0 or cols_a == 0 or cols_b == 0:
                    continue
                if rows_a != rows_b or cols_a != cols_b:
                    continue

                # All guards passed — merge B's file into A's series.
                # load_xa re-sorts by |PositionerPrimaryAngle| so the
                # order at this point doesn't decide Front vs Lateral.
                se_a.files.append(se_b.files[0])
                del study.series[uid_b]


def merge_patients(
    dst: dict[str, Patient], src: dict[str, Patient]
) -> None:
    """Merge a freshly scanned tree *src* into the accumulated tree *dst*
    in place, so opening another folder adds its studies instead of
    discarding the ones already loaded.

    Identity is by UID at each level (PatientID / StudyInstanceUID /
    SeriesInstanceUID). Existing Patient/Study/Series objects are kept
    (the browser holds references to them); only missing nodes are added.
    A series already present is left untouched, so re-dropping the same
    folder does not duplicate its files.
    """
    for pid, sp in src.items():
        dp = dst.get(pid)
        if dp is None:
            dst[pid] = sp
            continue
        for suid, sst in sp.studies.items():
            dst_st = dp.studies.get(suid)
            if dst_st is None:
                dp.studies[suid] = sst
                continue
            for seuid, sse in sst.series.items():
                if seuid not in dst_st.series:
                    dst_st.series[seuid] = sse


def remove_node(
    patients: dict[str, Patient], kind: str, key: str
) -> set[str]:
    """Drop a patient / study / series from the accumulated tree so it is
    no longer listed or viewable. *kind* is "patient"|"study"|"series";
    *key* is the PatientID / StudyInstanceUID / SeriesInstanceUID.

    Returns the set of SeriesInstanceUIDs removed (so the shell can blank
    any pane currently showing one). Studies/patients left empty by the
    removal are pruned too.
    """
    removed: set[str] = set()
    if kind == "patient":
        p = patients.pop(key, None)
        if p:
            for st in p.studies.values():
                removed.update(st.series.keys())
    elif kind == "study":
        # key is "StudyInstanceUID" or "StudyInstanceUID\x1fKIND".
        # The browser splits a study per modality kind, so a delete is
        # scoped to that kind; an unscoped key drops the whole study.
        suid, _, mod = key.partition("\x1f")
        for p in patients.values():
            st = p.studies.get(suid)
            if st is None:
                continue
            if mod:
                drop = [
                    u for u, s in st.series.items() if s.kind == mod
                ]
                for u in drop:
                    del st.series[u]
                    removed.add(u)
            else:
                removed.update(st.series.keys())
                del p.studies[suid]
            break
    elif kind == "series":
        for p in patients.values():
            for st in p.studies.values():
                if st.series.pop(key, None) is not None:
                    removed.add(key)
                    break
            if removed:
                break

    for pid in list(patients):
        p = patients[pid]
        for suid in [s for s, st in p.studies.items() if not st.series]:
            del p.studies[suid]
        if not p.studies:
            del patients[pid]
    return removed


class XAPlane:
    """One projection of an XA acquisition (a biplane series has two).

    Frames decode lazily: frame 0 is filled at load time so the still
    appears immediately; any other frame is decoded on first access and
    cached into ``volume``. A background prefetch (prefetch_planes) warms
    the rest so continuous cine playback is smooth.
    """

    def __init__(self, name: str, path: str, ds, total_frames: int,
                 volume: np.ndarray, is_color: bool = False,
                 frame_files: Optional[list[str]] = None,
                 force_color: bool = False):
        self.name = name
        self.path = path
        self._ds = ds                 # shared by UI seek + background prefetch
        #: When True, decode this plane in colour even if the default
        #: decision (e.g. Modality=IVUS) would force grayscale. Toggled by
        #: the IVUS viewer's manual "colour display" button via
        #: :func:`apply_color_mode_to_planes`. Read by :func:`_plane_decode`.
        self.force_color = force_color
        #: For a multi-FILE stack (Secondary Capture: one DICOM file per frame)
        #: this is the per-frame file list — frame i is decoded from
        #: frame_files[i] lazily/in the background, so the first image shows
        #: immediately instead of waiting for the whole stack. None for a normal
        #: single-dataset multi-frame cine (decoded from _ds).
        self.frame_files = frame_files
        #: serialises decodes off the one shared dataset so a (rare) manual
        #: seek on the UI thread and the prefetch thread never decode from
        #: it concurrently. Lets the prefetch reuse the dataset load_xa
        #: already read instead of re-opening the file (a second full read
        #: of a large biplane clip was the bulk of "biplane is slow to
        #: first show").
        self._lock = threading.Lock()
        self.total_frames = total_frames
        #: (F, H, W) float32 grayscale, or (F, H, W, 3) uint8 RGB
        self.volume = volume
        self.is_color = is_color
        self._ready = np.zeros(total_frames, dtype=bool)
        self._ready[0] = True

    def frame(self, i: int) -> np.ndarray:
        """Return frame *i* (float32 gray or uint8 RGB), decoding it on
        first access and caching into the shared volume."""
        i = max(0, min(int(i), self.total_frames - 1))
        if not self._ready[i]:
            with self._lock:
                if not self._ready[i]:          # prefetch may have won the race
                    self.volume[i] = _plane_decode(self, i)
                    self._ready[i] = True
        return self.volume[i]

    def is_ready(self, i: int) -> bool:
        """True if frame *i* is already decoded/cached. Lets the cine loop
        decide to *hold* rather than decode on the UI thread (a compressed
        frame is ~10-15 ms and the JPEG codec holds the GIL, so a synchronous
        decode there stalls timer + repaint = the 'first-loop stutter')."""
        i = max(0, min(int(i), self.total_frames - 1))
        return bool(self._ready[i])


@dataclass
class LoadedSeries:
    """Pixel data for one opened series.

    XA: volume shape is (frames, H, W), spacing_mm is in-plane pixel size.
        xa_planes lists every projection; for a single-plane cine it has one
        entry, for a biplane acquisition two (Front then Lateral). volume
        mirrors xa_planes[0].volume for callers that ignore biplane.
    CT: volume shape is (slices, H, W) in Hounsfield units, spacing_mm is
        (row, col) in-plane; slice_thickness handled by the viewer.
    """
    modality: Modality
    volume: np.ndarray
    spacing_mm: Optional[tuple[float, float]]
    cine_fps: Optional[float]
    window: Optional[float]
    level: Optional[float]
    xa_planes: Optional[list[XAPlane]] = None
    #: True when frames are RGB (volume is (F,H,W,3) uint8, no window/level).
    is_color: bool = False
    #: CT inter-slice spacing in mm (z), for undistorted oblique MPR/MIP.
    slice_mm: Optional[float] = None
    #: Metadata-only dataset of the first instance, kept so the viewer can
    #: list/overlay DICOM tags without re-reading pixel data.
    header: Optional[pydicom.Dataset] = None
    #: CT only: 3x3 rotation whose columns are the patient-LPS unit
    #: directions of the volume's voxel axes (x=cols, y=rows, z=slices),
    #: from ImageOrientationPatient + the slice progression. Lets the CT
    #: viewer report the oblique view's C-arm angle (LAO/RAO·CRA/CAU).
    #: None -> assume standard axial supine head-first (identity LPS).
    patient_basis: Optional[np.ndarray] = None
    #: The SHELL's series UID for this loaded data — the same value the
    #: tree shows. This is normally identical to the DICOM file's
    #: SeriesInstanceUID, but differs for series the scanner split out
    #: of a packed XA SeriesUID (where every file would otherwise share
    #: the same DICOM UID). Viewers MUST use this, not the header's
    #: UID, as the cache/dedup key, or different split rows collapse to
    #: a single "same series, skip reload" decision.
    series_uid: str = ""


def _imager_spacing(ds) -> Optional[tuple[float, float]]:
    # XA calibration: ImagerPixelSpacing is at the detector; PixelSpacing
    # is preferred when present (already corrected).
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        val = getattr(ds, tag, None)
        if val:
            r, c = _to_float(val[0]), _to_float(val[1])
            if r is not None and c is not None:
                return r, c
    return None


def series_spacing_mm(series: Series) -> Optional[tuple[float, float]]:
    """(row_mm, col_mm) pixel spacing of *series* read from its first
    file's PixelSpacing / ImagerPixelSpacing, or None when uncalibrated.
    Used to hand a DICOM calibration to external tools."""
    files = list(getattr(series, "files", []) or [])
    if not files:
        return None
    try:
        ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
    except Exception:
        return None
    return _imager_spacing(ds)


def _cine_fps(ds) -> Optional[float]:
    cr = _to_float(getattr(ds, "CineRate", None))
    if cr:
        return cr
    rd = _to_float(getattr(ds, "RecommendedDisplayFrameRate", None))
    if rd:
        return rd
    ft = _to_float(getattr(ds, "FrameTime", None))
    if ft and ft > 0:
        return 1000.0 / ft
    return None


_COLOR_PI = {
    "RGB", "YBR_FULL", "YBR_FULL_422", "YBR_PARTIAL_420",
    "YBR_PARTIAL_422", "YBR_ICT", "YBR_RCT", "PALETTE COLOR",
}


def _is_color_capable(ds) -> bool:
    """Header-only: COULD this dataset carry genuine colour? True when it is
    stored multi-sample (SamplesPerPixel>=3) or with a colour Photometric-
    Interpretation. Unlike :func:`_is_color_ds` this does NOT force IVUS to
    grayscale — it answers "is there any colour to recover if the user asks
    for it?", which gates the IVUS viewer's manual colour toggle."""
    if int(getattr(ds, "SamplesPerPixel", 1) or 1) >= 3:
        return True
    return str(getattr(ds, "PhotometricInterpretation", "")) in _COLOR_PI


def _is_color_ds(ds) -> bool:
    # IVUS is fundamentally a grayscale modality. Some scanners export
    # it as YBR_FULL_422 / SamplesPerPixel=3 for JPEG-baseline storage
    # efficiency — the chroma channels are near-neutral noise and the Y
    # channel carries the real signal. Treating those as colour gives
    # the image a faint tint, which the user sees as a bug. So: force
    # IVUS to the grayscale path regardless of PhotometricInterpretation.
    # The IVUS viewer can override this per-series via a manual "colour
    # display" toggle (force_color → apply_color_mode_to_planes) for the
    # rare genuinely-colour IVUS (e.g. NIRS chemogram, VH tissue maps).
    if str(getattr(ds, "Modality", "")).upper() == "IVUS":
        return False
    return _is_color_capable(ds)


_IMAGECODECS = None  # module: lazy-imported once; False if unavailable

#: encapsulated transfer syntax -> (imagecodecs decoder, max |Δ| vs the
#: reference decoder we will tolerate). Decoding a JPEG-LS / JPEG2000 /
#: lossless-JPEG codestream is deterministic, so those must be bit-exact
#: (tol 0). Only DCT JPEG (Baseline/Extended) has standard-permitted IDCT
#: rounding, so ±1 LSB is allowed there (user-approved, clinically nil).
_FAST_TS = {
    "1.2.840.10008.1.2.4.50": ("jpeg8_decode", 1),    # JPEG Baseline
    "1.2.840.10008.1.2.4.51": ("jpeg_decode", 1),     # JPEG Extended
    "1.2.840.10008.1.2.4.57": ("jpegsof3_decode", 0),  # JPEG Lossless
    "1.2.840.10008.1.2.4.70": ("jpegsof3_decode", 0),  # JPEG Lossless SV1
    "1.2.840.10008.1.2.4.80": ("jpegls_decode", 0),    # JPEG-LS lossless
    "1.2.840.10008.1.2.4.81": ("jpegls_decode", 0),    # JPEG-LS near-loss.
    "1.2.840.10008.1.2.4.90": ("jpeg2k_decode", 0),    # JPEG2000 lossless
    "1.2.840.10008.1.2.4.91": ("jpeg2k_decode", 0),    # JPEG2000
}


# Transfer syntaxes whose decoder performs the colour-space transform ITSELF,
# so pixel_array() already returns RGB even though the file's
# PhotometricInterpretation still advertises the stored YBR space. Converting
# YBR->RGB again double-applies the transform and turns black (0,0,0) into
# green (0,135,0) — the classic "green ultrasound" bug. (JPEG baseline/extended
# apply the JFIF/APP14 YCbCr<->RGB transform; JPEG 2000 reverses its ICT/RCT
# multi-component transform on decode.) For everything else (uncompressed, RLE,
# lossless JPEG, JPEG-LS) the YBR is preserved and DOES need converting.
_DECODES_TO_RGB = frozenset({
    "1.2.840.10008.1.2.4.50",   # JPEG Baseline (Process 1)
    "1.2.840.10008.1.2.4.51",   # JPEG Extended (Process 2 & 4)
    "1.2.840.10008.1.2.4.90",   # JPEG 2000 Lossless (RCT)
    "1.2.840.10008.1.2.4.91",   # JPEG 2000 (ICT)
})

#: Fast-path self-check tolerance (max |Δ| vs the reference decoder) for COLOUR
#: lossy JPEG. Two valid JPEG decoders round YCbCr->RGB + 4:2:2 chroma
#: upsampling differently, so colour differs more than grayscale's ≤1 LSB — yet
#: a genuinely wrong decode (channel swap / wrong colour space) still differs by
#: ~255 and is caught. User-approved for the ~13× colour-cine decode speed-up.
_COLOR_FAST_TOL = 32


def _imagecodecs():
    global _IMAGECODECS
    if _IMAGECODECS is None:
        try:
            import imagecodecs as _m
            _IMAGECODECS = _m
        except Exception:
            _IMAGECODECS = False
    return _IMAGECODECS or None


def _fast_raw_frame(ds, index: int):
    """imagecodecs (libjpeg-turbo / SIMD) decode of ONE grayscale OR colour
    encapsulated frame — several times faster than pydicom's reference handler
    (~13× on colour JPEG ultrasound cines). Returns the ndarray, or None
    meaning 'use pydicom'.

    Safety: the first decoded frame of each dataset is self-checked against the
    reference decoder (exact for lossless transfer syntaxes, ≤1 LSB for lossy
    grayscale DCT JPEG, ≤_COLOR_FAST_TOL for lossy colour where two valid
    decoders round YCbCr->RGB / chroma upsampling differently). Any mismatch
    beyond tolerance, unsupported syntax, or error permanently disables the
    fast path for that dataset, so a wrong pixel is never shown. Calls are
    serialised per plane (plane._lock), so the per-dataset flag writes are
    race-free.
    """
    if getattr(ds, "_mdv_nofast", False):
        return None
    ic = _imagecodecs()
    if ic is None:
        ds._mdv_nofast = True
        return None
    try:
        fn = getattr(ds, "_mdv_fast_fn", None)
        if fn is None:
            ts = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
            spec = _FAST_TS.get(ts)
            if spec is None:
                ds._mdv_nofast = True
                return None
            fn = getattr(ic, spec[0], None)
            if fn is None:
                ds._mdv_nofast = True
                return None
            ds._mdv_fast_fn = fn
            # Colour lossy JPEG (e.g. YBR_FULL_422 ultrasound) decodes ~13×
            # faster here too; allow the larger colour decoder variation (vs
            # the ≤1 LSB grayscale bound) while still catching a wrong decode.
            tol = spec[1]
            if int(getattr(ds, "SamplesPerPixel", 1) or 1) >= 3 and tol > 0:
                tol = _COLOR_FAST_TOL
            ds._mdv_fast_tol = tol
        nf = int(_to_float(getattr(ds, "NumberOfFrames", 1), 1)) or 1
        fb = get_frame(ds.PixelData, index, number_of_frames=nf)
        arr = np.squeeze(np.asarray(fn(fb)))
        if not getattr(ds, "_mdv_fast_ok", False):
            ref = np.squeeze(np.asarray(pixel_array(ds, index=index)))
            bad = ref.shape != arr.shape or int(
                np.abs(arr.astype(np.int64) - ref.astype(np.int64)).max()
            ) > ds._mdv_fast_tol
            if bad:
                ds._mdv_nofast = True
                return ref            # already have the correct frame
            ds._mdv_fast_ok = True
        return arr
    except Exception:
        ds._mdv_nofast = True
        return None


def _jfif_rgb_override(ds, index: int):
    """Correct a JPEG colour frame that is MIS-TAGGED PhotometricInterpretation
    = RGB while its codestream is actually JFIF-YCbCr (seen on GE volume-render
    "Processed Images" — field report: CT Series #302/#303, a green cast over a
    pink heart on black). The lossless-JPEG / pydicom decoders return the raw
    YCbCr samples un-converted, and because the tag says RGB no YBR->RGB step
    runs, so the image shows a green/teal cast. imagecodecs.jpeg_decode honours
    the JPEG's own JFIF marker and returns the true display RGB. Returns that
    ndarray, or None when this case doesn't apply (caller keeps the normal
    decode)."""
    if str(getattr(ds, "PhotometricInterpretation", "")) != "RGB":
        return None
    ts = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", ""))
    if ts not in _FAST_TS:                       # encapsulated JPEG-family only
        return None
    ic = _imagecodecs()
    if ic is None or not hasattr(ic, "jpeg_decode"):
        return None
    try:
        nf = int(_to_float(getattr(ds, "NumberOfFrames", 1), 1)) or 1
        fb = get_frame(ds.PixelData, index, number_of_frames=nf)
        if b"JFIF" not in fb[:64]:               # codestream isn't JFIF-YCbCr
            return None
        rgb = np.asarray(ic.jpeg_decode(fb))
        if rgb.ndim == 3 and rgb.shape[-1] >= 3:
            return np.ascontiguousarray(rgb[..., :3])
    except Exception:
        return None
    return None


def _raw_frame(ds, index: int) -> np.ndarray:
    """Decoded pixels for one frame, color space untouched.

    Fastest path: imagecodecs (see _fast_raw_frame) for grayscale
    encapsulated frames — several x quicker than pydicom's handler.

    Fast path: ``pixel_array(ds, index=index)`` decodes just that frame
    (works for uncompressed and codecs with a usable per-frame offset).

    Fallback: some encapsulated multiframe codecs can't seek to one frame
    (no Basic Offset Table, or a handler that ignores ``index=``). The old
    code re-decoded the WHOLE clip on every frame access -> O(N) per frame
    -> O(N^2) for a cine, and the background prefetch did it too, so long
    or compressed angio crawled. Now we decode the whole stack ONCE, cache
    it on the dataset, and index that thereafter (O(N) total). The slow
    indexed attempt is also disabled after its first failure so we don't
    keep paying the exception per frame.
    """
    cached = getattr(ds, "_mdv_full", None)
    if cached is not None:
        if getattr(ds, "_mdv_is_stack", False):
            return cached[min(index, len(cached) - 1)]
        return cached
    fast = _fast_raw_frame(ds, index)
    if fast is not None:
        return fast
    if not getattr(ds, "_mdv_no_index", False):
        try:
            return np.asarray(pixel_array(ds, index=index))
        except Exception:
            ds._mdv_no_index = True
    arr = np.asarray(ds.pixel_array)
    is_stack = arr.ndim == 4 or (arr.ndim == 3 and arr.shape[-1] not in (3, 4))
    ds._mdv_full = arr
    ds._mdv_is_stack = is_stack
    return arr[min(index, len(arr) - 1)] if is_stack else arr


def _to_u8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    hi = float(a.max()) if a.size else 0.0
    if hi > 0:
        a = a / hi * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def _decode_frame(ds, index: int, force_color: bool = False) -> np.ndarray:
    """One frame ready for display: grayscale -> 2-D float (window/level
    applied later by the viewer); color -> (H, W, 3) uint8 RGB, already
    in display color space (YBR converted, palette LUT applied).

    *force_color* overrides the default grayscale decision for a colour-
    capable dataset (used by the IVUS viewer's manual colour toggle, which
    recovers the rare genuinely-colour IVUS that :func:`_is_color_ds`
    otherwise forces to grayscale). It has no effect on a dataset that is
    not colour-capable — that still decodes to 2-D grayscale."""
    arr = _raw_frame(ds, index)
    color = _is_color_ds(ds) or (force_color and _is_color_capable(ds))
    if not color:
        # IVUS encoded as YBR has the luminance in channel 0; pull THAT
        # rather than averaging Y+Cb+Cr (which would mix in the
        # near-neutral chroma and produce a faint tint after windowing).
        # All other grayscale paths fall through to _to_gray2d.
        pi = str(getattr(ds, "PhotometricInterpretation", ""))
        if (str(getattr(ds, "Modality", "")).upper() == "IVUS"
                and pi.startswith("YBR")
                and arr.ndim == 3 and arr.shape[-1] >= 3):
            return np.ascontiguousarray(arr[..., 0])
        return _to_gray2d(arr)

    pi = str(getattr(ds, "PhotometricInterpretation", "RGB"))
    if pi == "PALETTE COLOR":
        rgb = np.asarray(apply_color_lut(arr, ds))
    elif pi.startswith("YBR"):
        ts = str(getattr(getattr(ds, "file_meta", None),
                         "TransferSyntaxUID", ""))
        if ts in _DECODES_TO_RGB:
            # The JPEG/JPEG2000 decoder already returned RGB; the YBR tag
            # describes the STORED space, not the decoded array. Converting
            # again double-applies YBR->RGB and greens the image.
            rgb = arr
        else:
            try:
                rgb = np.asarray(convert_color_space(arr, pi, "RGB"))
            except Exception:
                rgb = arr
    else:  # RGB
        # Some encapsulated JPEG colour frames are mis-tagged PI=RGB but carry a
        # JFIF-YCbCr codestream; decode those honouring the JFIF marker so they
        # don't show a green cast (see _jfif_rgb_override). Everything else keeps
        # the already-decoded array.
        override = _jfif_rgb_override(ds, index)
        rgb = override if override is not None else arr
    if rgb.ndim == 2:                       # safety: not actually color
        return rgb.astype(np.float32)
    rgb = rgb[..., :3]
    return np.ascontiguousarray(_to_u8(rgb))


def _fit_frame(fr: np.ndarray, target: tuple) -> np.ndarray:
    """Crop/pad *fr* to *target* (the volume's per-frame shape) — defensive for
    a multi-file stack whose pages aren't all the same size."""
    target = tuple(int(t) for t in target)
    if fr.shape == target:
        return fr
    out = np.zeros(target, dtype=fr.dtype)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(fr.shape, target))
    out[sl] = fr[sl]
    return out


def _plane_decode(plane: "XAPlane", i: int) -> np.ndarray:
    """Decode frame *i* of a plane. For a normal cine (frame_files is None) this
    is the single-dataset path. For a multi-FILE Secondary-Capture stack it
    reads frame i's own file and matches the volume's colour/grayscale format
    (so the whole stack doesn't have to be decoded up-front)."""
    fc = bool(getattr(plane, "force_color", False))
    ff = getattr(plane, "frame_files", None)
    if ff is None:
        return _decode_frame(plane._ds, i, force_color=fc)
    ds = pydicom.dcmread(ff[i], force=True)
    fr = _decode_frame(ds, 0, force_color=fc)
    if plane.volume.ndim == 4:                 # (N,H,W,3) colour volume
        if fr.ndim == 2:
            g = np.clip(fr, 0, 255).astype(np.uint8)
            fr = np.repeat(g[..., None], 3, axis=2)
        else:
            fr = np.ascontiguousarray(fr[..., :3]).astype(np.uint8)
    else:                                      # (N,H,W) grayscale volume
        if fr.ndim == 3:
            fr = (0.299 * fr[..., 0] + 0.587 * fr[..., 1]
                  + 0.114 * fr[..., 2])
        fr = np.asarray(fr, np.float32)
    return _fit_frame(fr, plane.volume.shape[1:])


def apply_color_mode_to_planes(planes: list["XAPlane"], color: bool) -> bool:
    """Re-decode every plane in *planes* into grayscale or colour and reshape
    its volume so the lazy frame decode / background prefetch fill in the same
    format. This is how the IVUS viewer's manual "colour display" toggle
    overrides the default decision (IVUS always grayscale).

    Frame 0 is decoded eagerly to learn the new shape; the rest reset to
    not-ready so they decode on demand / via prefetch in the new mode. The
    caller MUST stop any running prefetch BEFORE calling this (the volume
    arrays are replaced, so a concurrent prefetch write would target a stale
    buffer) and restart it afterwards.

    Returns the ACHIEVED colour flag: False when no plane could produce colour
    (not colour-capable, or frame 0 still decodes to 2-D even when forced) — the
    caller should then leave the toggle off and tell the user there is no colour
    to show. True when at least one plane became colour."""
    achieved = False
    for plane in planes:
        plane.force_color = bool(color)
        # Decode frame 0 in the requested mode to learn its new shape/dtype.
        # force_color is read inside _plane_decode, so this honours the toggle.
        f0 = _plane_decode(plane, 0)
        is_col = f0.ndim == 3
        if is_col:
            f0 = np.ascontiguousarray(f0[..., :3]).astype(np.uint8)
            dt = np.uint8
        elif np.issubdtype(f0.dtype, np.floating):
            f0 = np.asarray(f0, np.float32)
            dt = np.float32
        else:
            dt = f0.dtype
        # Swap the volume/_ready arrays under the plane lock so a (mis-stopped)
        # background prefetch can't write a frame into the half-replaced state.
        # The caller is still expected to stop the prefetch first; this is
        # defence-in-depth against the 80 ms-wait race.
        with plane._lock:
            plane.volume = np.zeros(
                (plane.total_frames,) + f0.shape, dtype=dt)
            plane.volume[0] = f0
            plane.is_color = is_col
            plane._ready = np.zeros(plane.total_frames, dtype=bool)
            plane._ready[0] = True
        achieved = achieved or is_col
    return achieved


def prefetch_planes(
    planes: list["XAPlane"],
    should_stop: Callable[[], bool],
    is_playing: Callable[[], bool] = lambda: True,
) -> None:
    """Warm the cine in the background, FRAME-MAJOR across all planes.

    Reuses the dataset load_xa already read (plane._ds) instead of
    re-opening each file — a second full read of a large biplane clip on
    startup was the bulk of "biplane is slow to first show". plane._lock
    serialises this against the (rare) UI-thread manual-seek decode so the
    one shared dataset is never decoded from two threads at once; during
    cine the UI never decodes (it holds on un-ready frames), so there is
    no contention on the hot path.

    Order matters for biplane: the dual view shows frame *i* of BOTH
    planes together, so it can't advance to *i* until *i* is ready on
    every plane. Warming plane 0 fully then plane 1 (plane-major) froze
    playback at frame 0 for the whole of plane 0's decode = "biplane is
    super slow". Frame-major (frame *i* of every plane, then *i+1*) keeps
    the planes in lockstep so the play head advances exactly like the
    single-plane case.

    Pacing is *adaptive* because the codec holds the GIL ~10-15 ms/frame
    and does NOT release it (threading can't parallelise this), so any
    decode in flight delays the cine timer by up to one frame's worth:

    * NOT playing (still frame 0 / the "Buffering..." gate): decode
      flat-out. GIL starvation of a non-animated UI is invisible, and
      this is exactly when we want the clip warmed so playback — started
      only once a lead is buffered — then runs straight from cache.
    * Playing: yield the GIL generously between frames so the timer keeps
      time. The viewer only starts the timer once a lead exists, so the
      prefetch is finishing the tail far ahead and never catches up.
    """
    planes = [p for p in planes if p.total_frames >= 1]
    if not planes or all(p.total_frames <= 1 for p in planes):
        return
    maxf = max(p.total_frames for p in planes)
    # Inter-frame yield when NOT playing. The JPEG codec holds the GIL for the
    # whole ~10-15 ms decode and does not release it, so warming "flat out"
    # (a sleep(0)) lets this thread re-grab the GIL immediately after each
    # frame — starving the UI thread. On a short clip the starvation is
    # invisible, but a long pull-back (e.g. a 4000-frame colour IVUS) warms for
    # tens of seconds and the app looks frozen right after a heavy 2x3 loads.
    # A small real sleep hands the event loop a reliable slice every frame: the
    # UI stays responsive while warming continues (only marginally slower).
    idle_sleep = 0.003
    for i in range(maxf):
        for plane in planes:
            if should_stop():
                return
            if i >= plane.total_frames or plane._ready[i]:
                continue
            with plane._lock:
                if plane._ready[i]:             # UI seek decoded it first
                    continue
                plane.volume[i] = _plane_decode(plane, i)
                plane._ready[i] = True
            # 4 ms hands the cine timer a slice on every frame while playing;
            # idle_sleep keeps the UI responsive while warming a long clip.
            time.sleep(0.004 if is_playing() else idle_sleep)


def load_xa(
    series: Series,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> LoadedSeries:
    """Open an XA series and decode only frame 0 of each plane.

    One file -> a single-plane multiframe cine. More than one file -> a
    biplane acquisition: each instance is one projection. Planes are ordered
    by |PositionerPrimaryAngle| (the near-AP plane first = Front, the steep
    plane = Lateral); an explicit ViewName overrides the heuristic.

    *progress* — optional ``(phase, done, total)`` callback. Fires per-
    file so the shell can keep a real progress bar visible during the
    biggest delay: pydicom.dcmread of a large compressed cine clip
    (100-300 MB encapsulated for a long IVUS pull-back is normal) and
    the first-frame decode.

    The full cine is NOT read here — remaining frames decode lazily/in the
    background (see XAPlane.frame / prefetch_planes), so the viewer can show
    and start playing immediately regardless of clip length or compression.
    """
    raw: list[tuple[float, Optional[str], str, object, int, np.ndarray]] = []
    fps: Optional[float] = None
    spacing: Optional[tuple[float, float]] = None

    n_files = max(1, len(series.files))
    if progress is not None:
        progress("Reading DICOM file…", 0, n_files * 2)

    for idx, path in enumerate(series.files):
        if progress is not None:
            progress(
                f"Reading file {idx + 1}/{n_files}…",
                idx * 2, n_files * 2,
            )
        ds = pydicom.dcmread(path, force=True)
        nf = max(1, int(_to_float(getattr(ds, "NumberOfFrames", 1), 1)))
        if progress is not None:
            progress(
                f"Decoding first frame ({idx + 1}/{n_files})…",
                idx * 2 + 1, n_files * 2,
            )
        f0 = _decode_frame(ds, 0)  # (H,W) float gray or (H,W,3) uint8 RGB

        pa = _to_float(getattr(ds, "PositionerPrimaryAngle", None))
        sort_key = abs(pa) if pa is not None else float(idx)
        vn = getattr(ds, "ViewName", None)
        raw.append((sort_key, str(vn) if vn else None, path, ds, nf, f0))

        if fps is None:
            fps = _cine_fps(ds)
        if spacing is None:
            spacing = _imager_spacing(ds)
    if progress is not None:
        progress("Finalising…", n_files * 2, n_files * 2)

    raw.sort(key=lambda t: t[0])  # frontal (small angle) first
    n = len(raw)
    planes: list[XAPlane] = []
    for i, (_key, view_name, path, ds, nf, f0) in enumerate(raw):
        if view_name:
            name = view_name
        elif n == 1:
            name = "Single"
        elif n == 2:
            name = "Front" if i == 0 else "Lateral"
        else:
            name = f"Plane {i + 1}"
        color = f0.ndim == 3
        # Keep the cine in its compact native dtype (XA/IVUS are 8- or
        # 16-bit). Pre-allocating the whole clip as float32 quadrupled
        # memory and could fail to allocate; window/level math upcasts a
        # single frame at a time, so int storage is fine.
        if color:
            dt = np.uint8
        elif np.issubdtype(f0.dtype, np.floating):
            dt = np.float32
        else:
            dt = f0.dtype
        vol = np.zeros((nf,) + f0.shape, dtype=dt)
        vol[0] = f0
        planes.append(XAPlane(name, path, ds, nf, vol, is_color=color))

    is_color = any(p.is_color for p in planes)
    if is_color:
        window, level = 255.0, 127.0  # unused for color, kept non-None
    else:
        f0s = np.concatenate([p.volume[0].ravel() for p in planes])
        lo, hi = float(f0s.min()), float(f0s.max())
        window = max(hi - lo, 1.0)
        level = (hi + lo) / 2.0

    return LoadedSeries(
        # Generic multi-frame cine loader: XA and IVUS share this path.
        # Keep the series' own modality so the shell routes it to the
        # matching viewer (XA → XAViewer, IVUS → IVUSViewer).
        modality=series.modality,
        volume=planes[0].volume,
        spacing_mm=spacing,
        cine_fps=fps,
        window=window,
        level=level,
        xa_planes=planes,
        is_color=is_color,
        header=_read_header(planes[0].path),
        series_uid=series.series_uid,
    )


def thumbnail(series: Series, max_px: int = 144) -> np.ndarray:
    """Small uint8 preview of *series* (one frame / mid slice). Returns a
    2-D grayscale array, or (H, W, 3) uint8 for color series."""
    # Secondary Capture (reports / film / snapshots) take the auto-window path
    # below, not the HU window — they are not Hounsfield data — and their
    # PI=RGB pseudo-colour is degenerate, so they're shown as grayscale.
    is_sc = _series_is_secondary_capture(series)
    if series.modality == Modality.CT and not is_sc:
        files = sorted(series.files)
        ds = pydicom.dcmread(files[len(files) // 2], force=True)
        px = _to_gray2d(ds.pixel_array).astype(np.float32)
        px = px * _to_float(getattr(ds, "RescaleSlope", 1.0), 1.0) + (
            _to_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
        )
        lo, hi = -100.0, 700.0  # generic soft-tissue/contrast window
    else:
        ds = pydicom.dcmread(series.files[0], force=True)
        px = _decode_frame(ds, 0)
        if px.ndim == 3:
            # Genuine colour (XA / IVUS, or a real colour SC render) stays
            # colour; a BROKEN pseudo-RGB SC capture → ITU-601 luminance.
            if is_sc and _is_broken_pseudo_color(px):
                px = (0.299 * px[..., 0] + 0.587 * px[..., 1]
                      + 0.114 * px[..., 2]).astype(np.float32)
            else:       # downsample as-is
                step = max(1, -(-max(px.shape[:2]) // max_px))
                return np.ascontiguousarray(
                    px[::step, ::step, :3].astype(np.uint8)
                )
        px = px.astype(np.float32)
        lo, hi = float(px.min()), float(px.max())

    out = np.clip((px - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0
    step = max(1, -(-max(out.shape) // max_px))  # ceil division
    return out[::step, ::step].astype(np.uint8)


def load_ct(
    series: Series,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> LoadedSeries:
    """Load and stack a CT series into a HU volume sorted along the patient
    axis.

    *progress*, if given, is called as progress(phase, done, total) during
    the two slow O(slices) loops (file read, then HU build) so the caller
    can keep a real progress bar visible — this is what the user sees as
    "the wait after the scan bar hits 100%".
    """
    files = list(series.files)
    n_files = len(files)
    slices = []
    for i, path in enumerate(files):
        ds = pydicom.dcmread(path, force=True)
        if "PixelData" in ds:
            slices.append(ds)
        if progress is not None and (i % 8 == 0 or i + 1 == n_files):
            progress("Reading CT slices…", i + 1, n_files)

    if not slices:
        raise ValueError("CT series has no pixel data")

    def sort_key(d):
        ipp = getattr(d, "ImagePositionPatient", None)
        z = _to_float(ipp[2]) if ipp is not None else None
        if z is not None:
            return z
        return _to_float(getattr(d, "InstanceNumber", 0), 0.0)

    slices.sort(key=sort_key)

    # Defensive: a CT series occasionally contains a scout/localizer
    # slice (or an accidentally-packed second acquisition) with a
    # different (Rows, Columns) — typically the same image rotated
    # 90° so e.g. (334, 552) and (552, 334) end up under one series
    # UID. The 3-D volume needs a uniform shape, so we keep only the
    # DOMINANT shape and drop the rest. Other viewers handle this
    # silently; we mirror that and log a stderr note for traceability.
    shape_counts: dict[tuple[int, int], int] = {}
    for d in slices:
        key = (int(d.Rows), int(d.Columns))
        shape_counts[key] = shape_counts.get(key, 0) + 1
    dom_shape = max(shape_counts.items(), key=lambda kv: kv[1])[0]
    rows, cols = dom_shape
    n_pre = len(slices)
    slices = [d for d in slices
              if (int(d.Rows), int(d.Columns)) == dom_shape]
    # Re-establish ``first`` (was previously ``slices[0]`` before this
    # filter existed) — downstream PixelSpacing / SliceThickness /
    # ImageOrientationPatient reads all use ``first.<tag>``.
    first = slices[0]
    n_dropped = n_pre - len(slices)
    if n_dropped:
        _warn(
            f"[load_ct] '{series.description or series.series_uid}': "
            f"dropped {n_dropped}/{n_pre} slice(s) with non-dominant "
            f"shape (kept shape={rows}x{cols}, "
            f"other shapes={[s for s in shape_counts if s != dom_shape]})\n"
        )

    vol = np.empty((len(slices), rows, cols), dtype=np.float32)

    n_sl = len(slices)
    for i, d in enumerate(slices):
        px = _to_gray2d(d.pixel_array).astype(np.float32)
        # Last-resort guard: if pydicom returns the array transposed
        # versus the header's (Rows, Columns) — rare, vendor-specific —
        # transpose it back so the assignment doesn't blow up.
        if px.shape != (rows, cols) and px.shape == (cols, rows):
            px = px.T
        if px.shape != (rows, cols):
            # Still mismatched (truly anomalous slice) — skip with note;
            # the corresponding vol[i] stays uninitialised, so blank it.
            _warn(
                f"[load_ct] slice {i}: shape mismatch "
                f"{px.shape} vs {(rows, cols)}; filling with zeros\n"
            )
            vol[i] = 0.0
            continue
        slope = _to_float(getattr(d, "RescaleSlope", 1.0), 1.0)
        intercept = _to_float(getattr(d, "RescaleIntercept", 0.0), 0.0)
        vol[i] = px * slope + intercept  # -> Hounsfield units
        if progress is not None and (i % 8 == 0 or i + 1 == n_sl):
            progress("Building CT volume…", i + 1, n_sl)

    spacing = None
    ps = getattr(first, "PixelSpacing", None)
    if ps:
        r, c = _to_float(ps[0]), _to_float(ps[1])
        if r is not None and c is not None:
            spacing = (r, c)

    # Inter-slice spacing (z): median |Δ ImagePositionPatient_z|, else
    # SliceThickness, else 1.0. Needed so oblique MPR/MIP is not distorted.
    def _zpos(d):
        ipp = getattr(d, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            return _to_float(ipp[2])
        return None

    zvals = [z for z in (_zpos(d) for d in slices) if z is not None]
    slice_mm = None
    if len(zvals) >= 2:
        diffs = np.diff(np.sort(np.asarray(zvals, dtype=np.float64)))
        # Ignore sub-micron gaps: floating-point noise in the z positions of a
        # NON-spatial stack (e.g. 20 reformat/MIP frames stored at one location)
        # would otherwise yield a microscopic spacing (~1e-5 mm), collapsing the
        # volume to near-zero depth — its slices then can't be paged/resliced
        # (they read as air = black). No real CT slice spacing is < 1 micron.
        diffs = diffs[diffs > 1e-3]
        if diffs.size:
            slice_mm = float(np.median(diffs))
        else:
            # ≥2 frames but all at (nearly) the same position — a NON-spatial
            # stack (rotation/MIP frames stored at one location). There is no
            # real inter-frame spacing; use a small compact value so they page
            # like a stack. (SliceThickness here is the slab thickness, often
            # tens of mm, which would inflate the pseudo-volume and break the
            # reslice FOV — see #305-type series.)
            slice_mm = 1.0
    if not slice_mm:                     # a single slice
        slice_mm = _to_float(
            getattr(first, "SliceThickness", None), 1.0
        ) or 1.0

    # Voxel-axis -> patient-LPS directions so the viewer can report the
    # oblique view's C-arm angle. Volume x=columns, y=rows, z=slices.
    # ImageOrientationPatient = (row-cosine | column-cosine).
    pbasis = None
    iop = getattr(first, "ImageOrientationPatient", None)
    if iop is not None and len(iop) >= 6:
        cc = [_to_float(iop[k]) for k in range(6)]
        if None not in cc:
            rx = np.asarray(cc[0:3], dtype=np.float64)   # +x (columns)
            cy = np.asarray(cc[3:6], dtype=np.float64)   # +y (rows)
            nrx, ncy = np.linalg.norm(rx), np.linalg.norm(cy)
            if nrx > 1e-6 and ncy > 1e-6:
                rx, cy = rx / nrx, cy / ncy
                sdir = None
                ip0 = getattr(slices[0], "ImagePositionPatient", None)
                ip1 = getattr(slices[-1], "ImagePositionPatient", None)
                if ip0 is not None and ip1 is not None and len(slices) >= 2:
                    dd = [_to_float(ip1[k]) - _to_float(ip0[k])
                          if _to_float(ip1[k]) is not None
                          and _to_float(ip0[k]) is not None else None
                          for k in range(3)]
                    if None not in dd:
                        d = np.asarray(dd, dtype=np.float64)
                        if np.linalg.norm(d) > 1e-6:
                            sdir = d / np.linalg.norm(d)
                if sdir is None:                          # +z (slices)
                    sdir = np.cross(rx, cy)
                pbasis = np.column_stack([rx, cy, sdir])

    # Default W/L. True CT keeps the coronary/angio default (800/200); a
    # Secondary-Capture series (dose report, electronic film, 3-D snapshot —
    # tagged Modality=CT but NOT Hounsfield data) is instead windowed from its
    # OWN embedded Window Center/Width, or, lacking that, from the actual value
    # range — so it shows up by default instead of being blacked out by the HU
    # window. WC/WW are in the rescaled output domain, matching `vol`.
    window, level = 800.0, 200.0
    if str(getattr(first, "SOPClassUID", "")).startswith(
            _SECONDARY_CAPTURE_PREFIX):
        wc = _first_num(getattr(first, "WindowCenter", None))
        ww = _first_num(getattr(first, "WindowWidth", None))
        if wc is not None and ww is not None and ww > 0:
            level, window = float(wc), float(ww)
        else:
            lo, hi = float(np.nanmin(vol)), float(np.nanmax(vol))
            window = max(hi - lo, 1.0)
            level = (hi + lo) / 2.0

    return LoadedSeries(
        modality=Modality.CT,
        volume=vol,
        spacing_mm=spacing,
        cine_fps=None,
        window=window,
        level=level,
        slice_mm=slice_mm,
        header=_read_header(getattr(first, "filename", "") or ""),
        patient_basis=pbasis,
        series_uid=series.series_uid,
    )


def load_secondary_capture(
    series: Series,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> LoadedSeries:
    """Load a Secondary-Capture series (dose report, "electronic film" screen
    capture, 3-D render snapshot — tagged Modality=CT but NOT Hounsfield data)
    as an image stack for the generic image viewer:

    * colour is preserved (RGB stays RGB — the CT MPR viewer is grayscale-only),
    * the pages/frames scroll like a cine,
    * the window is auto-fit (embedded Window Center/Width, else value range) so
      a CT preset can't black it out.

    Returns ``modality=OTHER`` so the shell routes it to the XA/image viewer,
    while the Study tree still groups it under CT (the Series keeps its CT
    modality, so it isn't re-grouped or split per-file)."""
    # Order pages by InstanceNumber (header-only reads — no pixel decode, so
    # this stays fast even for a big stack).
    order: list[tuple[float, str]] = []
    for idx, path in enumerate(series.files):
        ds = _read_header(path)
        ino = (_to_float(getattr(ds, "InstanceNumber", None))
               if ds is not None else None)
        order.append((ino if ino is not None else float(idx), path))
    order.sort(key=lambda t: t[0])
    sorted_files = [p for _, p in order]
    n = max(1, len(sorted_files))

    # Decode ONLY frame 0 so the first image appears immediately; the remaining
    # pages decode lazily on access / in the XA viewer's background prefetch
    # (XAPlane.frame_files → _plane_decode). This is the big win for a 40+ page
    # 1024² colour 3-D-VR stack that used to block on decoding every page.
    if progress is not None:
        progress("Reading first image…", 0, n)
    ds0 = None
    f0 = None
    start = 0
    for k, path in enumerate(sorted_files):
        try:
            ds0 = pydicom.dcmread(path, force=True)
            f0 = _decode_frame(ds0, 0)   # (H,W) float gray | (H,W,3) uint8 RGB
            start = k
            break
        except Exception:
            ds0 = None
    if ds0 is None or f0 is None:
        raise ValueError("Secondary-Capture series has no decodable image")
    sorted_files = sorted_files[start:]        # drop any unreadable leading pages

    # GENUINE colour (e.g. Fujifilm Synapse 3-D VR) is kept in colour; BROKEN
    # pseudo-colour (GE "electronic film" — PI=RGB with degenerate chroma that
    # every decoder renders green/magenta) falls back to ITU-601 grayscale (what
    # it actually is). Decided from frame 0's border (background) saturation.
    color = (f0.ndim == 3) and not _is_broken_pseudo_color(f0)
    if color:
        if f0.ndim == 2:
            g = np.clip(f0, 0, 255).astype(np.uint8)
            frame0 = np.repeat(g[..., None], 3, axis=2)
        else:
            frame0 = np.ascontiguousarray(f0[..., :3]).astype(np.uint8)
        window = level = None
    else:
        if f0.ndim == 3:
            f0 = (0.299 * f0[..., 0] + 0.587 * f0[..., 1]
                  + 0.114 * f0[..., 2])
        frame0 = np.asarray(f0, np.float32)
        lo, hi = float(np.nanmin(frame0)), float(np.nanmax(frame0))
        window = max(hi - lo, 1.0)            # W/L from frame 0 (good enough)
        level = (hi + lo) / 2.0

    nframes = len(sorted_files)
    vol = np.zeros((nframes,) + frame0.shape, dtype=frame0.dtype)
    vol[0] = frame0
    plane = XAPlane(series.description or "Secondary Capture",
                    sorted_files[0], ds0, nframes, vol,
                    is_color=color, frame_files=sorted_files)
    # only frame 0 is decoded; the rest fill in via _plane_decode (lazy/prefetch)
    return LoadedSeries(
        modality=Modality.OTHER,
        volume=vol,
        spacing_mm=_imager_spacing(ds0),
        cine_fps=_cine_fps(ds0),
        window=window,
        level=level,
        xa_planes=[plane],
        is_color=color,
        header=_read_header(sorted_files[0]),
        series_uid=series.series_uid,
    )


def load_series(
    series: Series,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> LoadedSeries:
    if series.modality == Modality.CT:
        # Secondary Capture (reports / electronic film / 3-D snapshots) are
        # tagged CT but are not HU data — show them in the image viewer with
        # colour + auto window instead of the grayscale CT MPR.
        if _series_is_secondary_capture(series):
            return load_secondary_capture(series, progress)
        return load_ct(series, progress)
    # MR / NM multi-image series are stored one single-frame file per frame, so
    # stack them into a 2-D cine (one page per file, like a Secondary-Capture
    # stack) instead of treating each file as an independent "plane" — load_xa
    # would otherwise show only one file and stall on "Buffering…".
    if (series.dicom_modality or "").upper() in ("MR", "NM") \
            and len(series.files) > 1:
        return load_secondary_capture(series, progress)
    return load_xa(series, progress)
