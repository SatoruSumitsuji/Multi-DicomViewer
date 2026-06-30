"""DICOM (verbatim copy) and MP4 (rendered) export for selected series.

Two entry points:

* :func:`export_dicom` — copies the original .dcm files into one subfolder
  per series. Lossless — same Transfer Syntax, same pixels, same DICOM
  tags. Per-file names are built from the user-checked series-level
  fields; duplicates within a series are auto-suffixed `(2), (3), …`.

* :func:`export_mp4` — renders each series to a single .mp4 in the
  chosen folder. XA / IVUS use the existing cine decoding path and apply
  the series' default window/level; CT renders slices in z-order using
  the series' default WL (800/200 — the angio preset).

Both helpers take a ``progress`` callback ``(current, total, message)``
so the caller can show a QProgressDialog without coupling the core to
Qt.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from typing import Callable, Iterable, Optional

import numpy as np
import pydicom

from .dicom_io import (
    _decode_frame,
    _is_color_ds,
    _to_float,
    _to_gray2d,
    load_ct,
    load_xa,
)
from .study_model import Modality, Series

ProgressCB = Optional[Callable[[int, int, str], None]]


# --------------------------------------------------------------- filename
_ILLEGAL = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')

#: Max chars for a single DICOM-tag value used as a filename component, and
#: for the whole joined filename — keeps output paths valid even when a tag
#: holds a huge value.
_MAX_FIELD_LEN = 64
_MAX_NAME_LEN = 120


def _safe_name(s: str) -> str:
    """Strip characters Windows refuses in filenames; collapse whitespace
    and trailing dots/spaces (also reserved on Windows)."""
    s = _ILLEGAL.sub("_", str(s)).strip()
    s = re.sub(r"\s+", "_", s)
    s = s.rstrip(". ")
    return s or "_"


def _fmt_date(acq: str) -> str:
    """Date part of an acquisition timestamp → 'YYYYMMDD'. Accepts
    either 'YYYYMMDDHHMMSS[.fff]' or just 'YYYYMMDD'. Empty input → ''."""
    s = str(acq or "").strip()
    if not s:
        return ""
    s = s.split(".", 1)[0]
    if len(s) >= 8 and s[:8].isdigit():
        return s[:8]
    return _safe_name(s)


def _fmt_time(acq: str) -> str:
    """Time part of an acquisition timestamp → 'HHMMSS', or '' when the
    timestamp has no time component (CT slice headers often don't)."""
    s = str(acq or "").strip()
    if not s:
        return ""
    s = s.split(".", 1)[0]
    if len(s) >= 14 and s[:14].isdigit() and s[8:14].isdigit():
        return s[8:14]
    return ""


def _fmt_int3(value) -> str:
    """Three-digit zero-padded integer (e.g. 3 → '003', 47 → '047').
    None / non-numeric → '' so the field is silently skipped on series
    that lack the tag."""
    if value is None:
        return ""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return ""
    return f"{abs(n):03d}"


def _fmt_primary(value) -> str:
    """C-arm primary angle as LAO/RAO + 3-digit absolute value
    (DICOM convention: + = LAO, - = RAO). 0 → 'LAO000'."""
    v = _to_float(value, None)
    if v is None:
        return ""
    n = int(round(v))
    tag = "RAO" if n < 0 else "LAO"
    return f"{tag}{abs(n):03d}"


def _fmt_secondary(value) -> str:
    """C-arm secondary angle as CRA/CAU + 3-digit absolute value
    (DICOM convention: + = CRA, - = CAU). 0 → 'CRA000'."""
    v = _to_float(value, None)
    if v is None:
        return ""
    n = int(round(v))
    tag = "CAU" if n < 0 else "CRA"
    return f"{tag}{abs(n):03d}"


def _series_fields(series: Series, ds: pydicom.Dataset) -> dict[str, str]:
    """Series-level field values (same across every instance in a series).
    Reads from the cached ``Series`` where possible, then from the first
    file's metadata for the per-frame DICOM tags."""
    n_frames = 0
    if "NumberOfFrames" in ds:
        n_frames = int(_to_float(ds.NumberOfFrames, 0) or 0)
    if not n_frames:                                # single-frame CT slice
        n_frames = len(series.files)
    out = {
        "date":        _fmt_date(series.acq_time),
        "time":        _fmt_time(series.acq_time),
        "series_no":   _fmt_int3(series.number),
        "instance_no": _fmt_int3(series.instance_number),
        "type":        _safe_name(series.kind),
        "description": _safe_name(series.description or ""),
        "images":      f"{n_frames}img",
        "primary":     _fmt_primary(
            getattr(ds, "PositionerPrimaryAngle", None)
        ),
        "secondary":   _fmt_secondary(
            getattr(ds, "PositionerSecondaryAngle", None)
        ),
    }
    return out


def _dicom_tag_value(ds: pydicom.Dataset, identifier: str) -> str:
    """Filename-safe string for a DICOM tag's value, addressed by either
    a pydicom keyword (``PatientName``) or a tag-string literal
    (``(0019,1099)``). Returns ``""`` when the tag is absent or holds
    nothing useful, so it drops out of the joined filename."""
    from .dicom_tags import _lookup  # local import — avoid load-time cycle
    elem = _lookup(ds, identifier)
    if elem is None:
        return ""
    val = elem.value
    # A tag holding a huge value (e.g. text stuffed into an OB element) must
    # never become a megabyte-long filename component — slice the raw value
    # before stringifying so we don't even build the giant string.
    if isinstance(val, (bytes, bytearray, memoryview)):
        val = bytes(val[:128])
    try:
        text = str(val).strip()
    except Exception:
        return ""
    if not text:
        return ""
    return _safe_name(text[:_MAX_FIELD_LEN])


def build_filename(fields: Iterable[str],
                   series: Series,
                   ds: pydicom.Dataset) -> str:
    """Join the user-chosen *fields* (in the order given) with '_' for one
    output file. Empty / missing components are dropped. Series- and
    acquisition-level fields are populated from ``series`` / ``ds``;
    DICOM-tag identifiers not in the predefined table (anything that
    looks like a pydicom keyword or a ``(group,element)`` literal) are
    resolved against the header so the user can mix tag values in
    alongside the formatted fields."""
    vals = _series_fields(series, ds)
    parts = [
        vals[k] if k in vals else _dicom_tag_value(ds, k)
        for k in fields
    ]
    name = "_".join(p for p in parts if p)
    # Cap the whole name so the output path stays within filesystem limits
    # (Windows component limit is 255; keep well under, leaving room for the
    # extension and any " (2)" dedup suffix).
    return _safe_name(name)[:_MAX_NAME_LEN] or "export"


def build_series_folder(fields: Iterable[str],
                        series: Series,
                        ds: pydicom.Dataset) -> str:
    """Subfolder name for a series in DICOM export. Same fields as the
    per-file name — every chosen field is series-level now."""
    return build_filename(fields, series, ds)


def _unique_path(path: str) -> str:
    """If *path* already exists append ' (2)', ' (3)', … to the stem so
    nothing is overwritten."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while True:
        cand = f"{base} ({i}){ext}"
        if not os.path.exists(cand):
            return cand
        i += 1


# --------------------------------------------------------------- DICOM
def export_dicom(series_list: list[Series],
                 out_dir: str,
                 fields: Iterable[str],
                 progress: ProgressCB = None) -> list[str]:
    """Copy original .dcm files to *out_dir*/<series-folder>/<name>.dcm.

    Returns the list of folders written (one per series). Filenames are
    deduped per folder; bad characters are stripped. The InstanceNumber
    in each output filename comes from THAT file, so e.g. a 300-slice CT
    series yields 300 files each with its own '#42' suffix.
    """
    fields = tuple(fields)
    written: list[str] = []
    total_files = sum(len(s.files) for s in series_list)
    done = 0
    if progress:
        progress(0, total_files, "Preparing…")

    for si, series in enumerate(series_list):
        if not series.files:
            continue
        # Read the first file's header once to drive the folder name.
        from . import dicom_io  # local import — avoid load-time cycle
        try:
            first_ds = pydicom.dcmread(
                series.files[0], stop_before_pixels=True, force=True
            )
            dicom_io.repair_dataset_text(first_ds)  # clean JP in folder names
        except Exception:
            first_ds = pydicom.Dataset()
        folder = build_series_folder(fields, series, first_ds) or (
            f"series_{si + 1}"
        )
        sub = _unique_path(os.path.join(out_dir, folder))
        os.makedirs(sub, exist_ok=True)
        written.append(sub)

        for path in series.files:
            try:
                ds = pydicom.dcmread(
                    path, stop_before_pixels=True, force=True
                )
                dicom_io.repair_dataset_text(ds)  # clean JP in filenames
            except Exception:
                ds = first_ds
            base = build_filename(fields, series, ds)
            target = _unique_path(os.path.join(sub, base + ".dcm"))
            try:
                shutil.copy2(path, target)
            except Exception as e:
                if progress:
                    progress(done, total_files,
                             f"Failed: {os.path.basename(path)} ({e})")
            done += 1
            if progress and (done % 4 == 0 or done == total_files):
                progress(done, total_files,
                         f"Copying [{si + 1}/{len(series_list)}] "
                         f"{os.path.basename(path)}")
    if progress:
        progress(total_files, total_files, "Done")
    return written


# ----------------------------------------------------------- Anon DICOM
def export_anon_dicom(series_list: list[Series],
                      out_dir: str,
                      fields: Iterable[str],
                      progress: ProgressCB = None) -> list[str]:
    """Like :func:`export_dicom`, but writes a DE-IDENTIFIED copy of each
    file: the active anonymization profile (see
    :func:`core.anonymize.deidentify_dataset`) emptifies the configured tags'
    values and every private element's value. Pixels, UIDs and the transfer
    syntax are preserved, so the output still decodes/displays.

    One subfolder per series; per-file names from the checked *fields*.
    Returns the list of folders written."""
    from .anonymize import deidentify_dataset

    fields = tuple(fields)
    written: list[str] = []
    total_files = sum(len(s.files) for s in series_list)
    done = 0
    if progress:
        progress(0, total_files, "Preparing…")

    for si, series in enumerate(series_list):
        if not series.files:
            continue
        from . import dicom_io  # local import — avoid load-time cycle
        try:
            first_ds = pydicom.dcmread(
                series.files[0], stop_before_pixels=True, force=True
            )
            dicom_io.repair_dataset_text(first_ds)  # clean JP in folder names
        except Exception:
            first_ds = pydicom.Dataset()
        folder = build_series_folder(fields, series, first_ds) or (
            f"series_{si + 1}"
        )
        # Suffix so an anonymized export never overwrites a plain DICOM export
        # sitting in the same chosen folder.
        sub = _unique_path(os.path.join(out_dir, folder + "_anon"))
        os.makedirs(sub, exist_ok=True)
        written.append(sub)

        for path in series.files:
            try:
                ds = pydicom.dcmread(path, force=True)   # full read (pixels)
                dicom_io.repair_dataset_text(ds)  # clean JP before filename build
                deidentify_dataset(ds)
                base = build_filename(fields, series, ds)
                target = _unique_path(os.path.join(sub, base + ".dcm"))
                ds.save_as(target, enforce_file_format=True)
            except Exception as e:
                if progress:
                    progress(done, total_files,
                             f"Failed: {os.path.basename(path)} ({e})")
            done += 1
            if progress and (done % 4 == 0 or done == total_files):
                progress(done, total_files,
                         f"Anonymizing [{si + 1}/{len(series_list)}] "
                         f"{os.path.basename(path)}")
    if progress:
        progress(total_files, total_files, "Done")
    return written


# --------------------------------------------------------------- CSV
def _decode_binary_text(raw: bytes) -> Optional[str]:
    """Best-effort text from a binary (OB/UN/…) value. Returns the full
    decoded string when the bytes are predominantly printable text — e.g. an
    XML/JSON report or other text payload stuffed into an OB element — or
    None when they look like genuine binary (image/waveform) that should
    stay summarised.

    Tries UTF-8, then CP932 (Shift-JIS, common in Japanese DICOM), then
    Latin-1 as a never-fails 1:1 byte fallback. A trailing NUL (DICOM
    even-length padding) is dropped."""
    if not raw:
        return ""
    raw = raw.rstrip(b"\x00")           # drop DICOM even-length NUL padding
    if not raw:
        return ""
    text = None
    for enc in ("utf-8", "cp932", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:                    # latin-1 never raises → unreachable
        return None
    # Reject mostly-unprintable payloads (real binary, not text-in-OB).
    printable = sum(1 for c in text if c.isprintable() or c in "\r\n\t")
    if printable / len(text) < 0.85:
        return None
    return text


def _csv_value(elem, ident: str, anonymized: bool) -> str:
    """FULL display string for *elem*'s value for a CSV cell — unlike the
    overlay/table formatters, this never truncates and never collapses
    whitespace, so very large text elements land intact (csv quoting keeps
    embedded newlines/commas safe). Binary VRs (OB/UN/…) that actually hold
    text are decoded and written in full; genuinely binary ones (and
    sequences) fall back to a short descriptor. PHI is masked to match the
    on-screen Anonymize state."""
    from .anonymize import mask_value
    from .dicom_tags import _BINARY_VRS

    vr = str(elem.VR)
    if vr == "SQ":
        try:
            return f"<sequence: {len(elem.value)} item(s)>"
        except TypeError:
            return "<sequence>"
    if vr in _BINARY_VRS:
        raw = elem.value
        if isinstance(raw, (bytes, bytearray, memoryview)):
            decoded = _decode_binary_text(bytes(raw))
            if decoded is not None:
                return mask_value(elem, decoded, anonymized)
            return f"<binary: {len(raw)} bytes>"
        # Non-bytes binary value (rare) — fall through to the str() path.
    try:
        text = str(elem.value)
    except Exception:
        return "<unreadable>"
    return mask_value(elem, text, anonymized)


def export_csv(series_list: list[Series],
               out_dir: str,
               fields: Iterable[str],
               tag_identifiers: list[list[str]],
               anonymized: bool = False,
               progress: ProgressCB = None) -> list[str]:
    """Write one .csv per series into *out_dir*, listing the DICOM-tag-overlay
    tags currently shown for that series as ``Tag Name, Tag Number, Value``.

    ``tag_identifiers`` is a per-series list aligned with ``series_list`` —
    each entry holds the overlay tag identifiers (pydicom keywords, private
    keys, or ``(group,element)`` literals) chosen for that series' modality.
    Only tags actually present in the header are written (matching what the
    overlay shows). Values are written in FULL (no truncation). Filenames are
    built from the same checked *fields* as the other exporters and deduped.
    Returns the list of files written."""
    import csv as _csv
    from .dicom_tags import _lookup

    fields = tuple(fields)
    written: list[str] = []
    n = len(series_list)
    if progress:
        progress(0, n, "Preparing…")

    for si, series in enumerate(series_list):
        if progress:
            progress(si, n,
                     f"Writing CSV [{si + 1}/{n}] {series.kind} "
                     f"#{series.number or '?'}")
        if not series.files:
            continue

        # One value column per plane: a biplane XA series (2 files) yields
        # Lt + Rt, anything else (single XA / IVUS / CT / a single-plane
        # synthetic series from the image right-click) yields one "Value".
        planes = _csv_plane_datasets(series)          # [(label, ds), ...]
        col_labels = [lab for lab, _ in planes]

        idents = tag_identifiers[si] if si < len(tag_identifiers) else []
        rows: list[list[str]] = []
        for ident in idents:
            name = number = None
            values: list[str] = []
            for _lab, ds in planes:
                elem = _lookup(ds, ident)
                if elem is not None and name is None:
                    name = elem.name or ident
                    number = (f"({elem.tag.group:04X},"
                              f"{elem.tag.element:04X})")
                values.append(
                    _csv_value(elem, ident, anonymized)
                    if elem is not None else ""
                )
            if name is None:                          # absent in every plane
                continue
            rows.append([name, number] + values)

        base = build_filename(fields, series, planes[0][1])
        target = _unique_path(os.path.join(out_dir, base + ".csv"))
        try:
            # utf-8-sig so Excel on Windows reads Japanese tag values
            # correctly; newline="" per the csv module's contract.
            with open(target, "w", encoding="utf-8-sig", newline="") as fh:
                w = _csv.writer(fh)
                w.writerow(["Tag Name", "Tag Number"] + col_labels)
                w.writerows(rows)
        except Exception as e:
            if progress:
                progress(si, n, f"Failed: {os.path.basename(target)} ({e})")
            continue
        written.append(target)
    if progress:
        progress(n, n, "Done")
    return written


def _csv_plane_datasets(series: Series):
    """Datasets to write as CSV value columns for *series*.

    A biplane XA series (exactly 2 files) → ``[("Lt", ds), ("Rt", ds)]``,
    ordered by ``|PositionerPrimaryAngle|`` exactly as ``load_xa`` does
    (frontal/near-AP plane = Lt, the steep plane = Rt) so the columns match
    the on-screen left/right. Everything else (single-plane XA, IVUS, CT, or
    a single-file synthetic series from the image right-click) → a single
    ``[("Value", ds)]``. Unreadable files fall back to an empty Dataset."""
    from .study_model import Modality

    def _read(path):
        try:
            return pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            return pydicom.Dataset()

    files = series.files or []
    if series.modality == Modality.XA and len(files) == 2:
        scored = []
        for i, p in enumerate(files):
            ds = _read(p)
            pa = _to_float(getattr(ds, "PositionerPrimaryAngle", None), None)
            scored.append((abs(pa) if pa is not None else float(i), ds))
        scored.sort(key=lambda t: t[0])
        return [("Lt", scored[0][1]), ("Rt", scored[1][1])]
    return [("Value", _read(files[0]) if files else pydicom.Dataset())]


# --------------------------------------------------------------- MP4
def _to_rgb_u8(frame: np.ndarray,
               window: Optional[float],
               level: Optional[float]) -> np.ndarray:
    """Any decoded frame → (H, W, 3) uint8 RGB suitable for H.264.

    Grayscale floats are window/level'd; uint8 grayscale is mapped to its
    full range; (H, W, 3) RGB is returned as-is."""
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return np.ascontiguousarray(arr[..., :3].astype(np.uint8))

    arr = _to_gray2d(arr)
    if window and window > 0 and level is not None:
        lo = float(level) - float(window) / 2.0
        out = (arr.astype(np.float32) - lo) / max(float(window), 1e-6)
        u8 = np.clip(out, 0.0, 1.0) * 255.0
    else:
        a = arr.astype(np.float32)
        lo = float(a.min()) if a.size else 0.0
        hi = float(a.max()) if a.size else 1.0
        u8 = (a - lo) / max(hi - lo, 1e-6) * 255.0
    u8 = u8.astype(np.uint8)
    return np.repeat(u8[:, :, None], 3, axis=2)


def _pad_to_even(rgb: np.ndarray) -> np.ndarray:
    """libx264 / yuv420p needs even W and H. Pad with the last row/col so
    no resampling artefact appears."""
    h, w = rgb.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if not (pad_h or pad_w):
        return rgb
    return np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _make_writer(path: str, fps: float, bitrate_mbps: int,
                 crf: Optional[int]):
    """imageio-ffmpeg writer; libx264 + yuv420p so QuickTime / Windows
    Media Player / browsers all play it.

    Two quality modes:
      * crf is None → explicit target bitrate ('Mbps' means Mbps).
      * crf set → x264 constant-quality (-crf). The encoder spends bits
        only where needed, so flat/black areas (IVUS composites) cost
        almost nothing and files are far smaller at equal visual quality.
    """
    import imageio.v2 as imageio   # lazy: avoids cost when user never exports

    kwargs = dict(
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,           # we already pad to even ourselves
        quality=None,                 # don't let imageio add its own -qscale
        ffmpeg_log_level="error",
    )
    if crf is not None:
        kwargs["output_params"] = [
            "-crf", str(int(crf)), "-preset", "medium"
        ]
    else:
        kwargs["bitrate"] = f"{int(bitrate_mbps)}M"
    return imageio.get_writer(path, **kwargs)


def _move_over(src: str, dst: str) -> None:
    """Move *src* → *dst*, overwriting. Python's file ops handle Unicode
    destinations fine even when the bundled ffmpeg can't write there."""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    try:
        os.replace(src, dst)               # atomic on the same filesystem
    except OSError:
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)


class _Mp4Stream:
    """Streaming MP4 encoder: frames are piped to ffmpeg one at a time so
    memory stays flat (a 4-up 1024² composite × ~1400 frames would be
    >4 GB if buffered). When the destination path isn't pure ASCII the
    bundled Windows ffmpeg can't open it (fails with '[Errno 22] Invalid
    argument'), so we encode to an ASCII temp file and move it into place
    on close()."""

    def __init__(self, path: str, fps: float, bitrate_mbps: int,
                 crf: Optional[int]):
        self._final = path
        self._tmp: Optional[str] = None
        out = path
        if not path.isascii():
            fd, self._tmp = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            out = self._tmp
        self._writer = _make_writer(out, fps, bitrate_mbps, crf)

    def add(self, frame: np.ndarray) -> None:
        self._writer.append_data(_pad_to_even(frame))

    def close(self) -> None:
        self._writer.close()
        if self._tmp:
            _move_over(self._tmp, self._final)

    def abort(self) -> None:
        try:
            self._writer.close()
        except Exception:
            pass
        if self._tmp and os.path.exists(self._tmp):
            try:
                os.remove(self._tmp)
            except OSError:
                pass


def open_mp4_stream(path: str, fps: float, bitrate_mbps: int,
                    crf: Optional[int] = None) -> _Mp4Stream:
    """Open a streaming MP4 encoder. Call ``.add(frame_rgb)`` per frame,
    then ``.close()`` (or ``.abort()`` on cancel). Bounds memory and works
    with non-ASCII output paths — used by the MultiSync composite export."""
    return _Mp4Stream(path, fps, bitrate_mbps, crf)


def write_mp4(path: str, frames: list[np.ndarray],
              fps: float, bitrate_mbps: int,
              crf: Optional[int] = None) -> None:
    """Public alias for callers that already have all rendered RGB frames.
    When *crf* is given, encode at constant quality instead of a fixed
    bitrate."""
    _write_mp4(path, frames, fps, bitrate_mbps, crf)


def _write_mp4(path: str, frames: list[np.ndarray],
               fps: float, bitrate_mbps: int,
               crf: Optional[int] = None) -> None:
    stream = open_mp4_stream(path, fps, bitrate_mbps, crf)
    try:
        for f in frames:
            stream.add(f)
    except BaseException:
        stream.abort()
        raise
    stream.close()


def _render_xa_series(series: Series) -> tuple[list[np.ndarray], float]:
    """Decoded RGB frames for an XA / IVUS series, with the source FPS
    (None → caller's default). For biplane (2 planes), the two views are
    laid out side-by-side per frame so the output is one MP4."""
    loaded = load_xa(series)
    planes = loaded.xa_planes or []
    if not planes:
        return [], 0.0
    n = max(p.total_frames for p in planes)
    fps = loaded.cine_fps or 0.0

    out: list[np.ndarray] = []
    for i in range(n):
        tiles = []
        h_max = 0
        for p in planes:
            j = min(i, p.total_frames - 1)
            tile = _to_rgb_u8(p.frame(j), loaded.window, loaded.level)
            tiles.append(tile)
            h_max = max(h_max, tile.shape[0])
        # Pad each tile to the tallest height, then hstack.
        padded = []
        for t in tiles:
            if t.shape[0] < h_max:
                pad = h_max - t.shape[0]
                t = np.pad(t, ((0, pad), (0, 0), (0, 0)), mode="constant")
            padded.append(t)
        out.append(np.hstack(padded) if len(padded) > 1 else padded[0])
    return out, fps


def _render_ct_series(series: Series) -> tuple[list[np.ndarray], float]:
    """Decoded RGB frames (axial slice scroll) for a CT series, with the
    default 15 fps (the dialog can override). The series' default WL
    (800/200 — angio preset, see load_ct) is applied."""
    loaded = load_ct(series)
    vol = loaded.volume
    out: list[np.ndarray] = []
    for i in range(vol.shape[0]):
        out.append(_to_rgb_u8(vol[i], loaded.window, loaded.level))
    return out, 15.0


def render_series_for_mp4(series: Series
                          ) -> tuple[list[np.ndarray], float]:
    """Public helper: returns (RGB-frames, default-fps) for one series.
    Dispatches by modality. The caller decides the final fps (the source
    rate is the default; the dialog can override)."""
    if series.modality == Modality.CT:
        return _render_ct_series(series)
    return _render_xa_series(series)


def _clip_to_range(frames: list[np.ndarray],
                   rng: Optional[tuple[int, int]]) -> list[np.ndarray]:
    """Slice *frames* to the inclusive [start, end] Play range (0-based).
    None, or a range covering the whole clip, returns *frames* unchanged.
    Bounds are clamped defensively so a stale range can never raise."""
    if not rng or not frames:
        return frames
    start, end = int(rng[0]), int(rng[1])
    last = len(frames) - 1
    start = max(0, min(start, last))
    end = max(start, min(end, last))
    if start == 0 and end == last:
        return frames
    return frames[start:end + 1]


def export_mp4(series_list: list[Series],
               out_dir: str,
               fields: Iterable[str],
               bitrate_mbps: int,
               fps_override: Optional[float],
               crf: Optional[int] = None,
               frame_ranges: Optional[
                   list[Optional[tuple[int, int]]]] = None,
               progress: ProgressCB = None) -> list[str]:
    """Write one .mp4 per series into *out_dir*. Returns the list of
    files written. ``fps_override`` (from the dialog) wins over the
    source cine rate when non-None. When ``crf`` is set, encode at
    constant quality instead of the target bitrate.

    ``frame_ranges``, when given, is a per-series list aligned with
    ``series_list``: each entry is an inclusive ``(start, end)`` 0-based
    frame range set on the viewer's Play seek bar, or ``None`` to export
    every frame. CT (slice scroll) is never range-clipped."""
    fields = tuple(fields)
    written: list[str] = []
    n = len(series_list)
    if progress:
        progress(0, n, "Preparing…")

    for si, series in enumerate(series_list):
        if progress:
            progress(si, n,
                     f"Rendering [{si + 1}/{n}] {series.kind} "
                     f"#{series.number or '?'}")
        try:
            frames, src_fps = render_series_for_mp4(series)
        except Exception as e:
            if progress:
                progress(si, n, f"Failed: {e}")
            continue
        if not frames:
            continue
        # Honour the Play-range markers (cine modalities only — CT is a
        # slice scroll, not a timed cine, so it always exports in full).
        if frame_ranges is not None and series.modality != Modality.CT:
            rng = frame_ranges[si] if si < len(frame_ranges) else None
            frames = _clip_to_range(frames, rng)
            if not frames:
                continue
        fps = float(fps_override) if fps_override and fps_override > 0 \
            else (src_fps or 15.0)

        # Filename: pull C-arm angle values from the first file's
        # metadata (one MP4 per series, so all series-level fields
        # apply uniformly).
        try:
            ds0 = pydicom.dcmread(
                series.files[0], stop_before_pixels=True, force=True
            )
        except Exception:
            ds0 = pydicom.Dataset()
        base = build_filename(fields, series, ds0)
        target = _unique_path(os.path.join(out_dir, base + ".mp4"))

        if progress:
            qual = f"CRF {crf}" if crf is not None else f"{bitrate_mbps} Mbps"
            progress(si, n,
                     f"Encoding [{si + 1}/{n}] {os.path.basename(target)} "
                     f"({len(frames)} frames @ {fps:.1f} fps, {qual})")
        _write_mp4(target, frames, fps, bitrate_mbps, crf)
        written.append(target)
    if progress:
        progress(n, n, "Done")
    return written
