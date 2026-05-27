"""DICOM folder scanning, study-tree building, and pixel access.

Indexing reads metadata only (stop_before_pixels) so a large folder loads
fast; pixel data is pulled lazily when a series is actually opened.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pydicom
from pydicom.encaps import get_frame
from pydicom.errors import InvalidDicomError
from pydicom.pixels import apply_color_lut, convert_color_space, pixel_array

from .study_model import Modality, Patient, Series, Study


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
    """Metadata-only dataset of *path* (no pixel data), or None on failure."""
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return None


def scan_folder(
    root: str, progress: Optional[Callable[[int, int], None]] = None
) -> dict[str, Patient]:
    """Walk *root* recursively and build a Patient/Study/Series tree.

    *progress*, if given, is called as progress(done, total) while the
    files are read (total known up-front from a fast directory walk).
    """
    patients: dict[str, Patient] = {}

    all_files = [
        os.path.join(dp, fn)
        for dp, _d, files in os.walk(root)
        for fn in files
    ]
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

            pid = str(_safe(ds, "PatientID", "UNKNOWN"))
            pname = str(_safe(ds, "PatientName", "Anonymous"))
            patient = patients.setdefault(pid, Patient(pid, pname))

            st_uid = str(_safe(ds, "StudyInstanceUID", "NO_STUDY"))
            study = patient.studies.setdefault(
                st_uid,
                Study(
                    study_uid=st_uid,
                    description=str(_safe(ds, "StudyDescription")),
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
                    description=str(_safe(ds, "SeriesDescription")),
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

    _split_packed_xa_series(patients)
    return patients


def _split_packed_xa_series(patients: dict[str, Patient]) -> None:
    """Some vendors (notably Philips Allura/Azurion) store every XA cine
    of a study under a single SeriesInstanceUID and only distinguish
    them via InstanceNumber. Other DICOM viewers show one row per
    instance — so do we, by splitting any XA series with ≥ 3 files into
    one per file. Single (1 file) and biplane (2 files) series are
    left intact.

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
                if se.modality != Modality.XA or len(se.files) < 3:
                    new_series[uid] = se
                    continue
                # Read each packed file's header to split into per-file
                # rows. Costly only for these unusual series; common
                # (biplane / single) series skip this entirely.
                for idx, path in enumerate(se.files):
                    ds = _read_header(path)
                    if ds is None:
                        # Unreadable: keep the original group key so the
                        # file isn't silently dropped from the tree.
                        new_series[uid] = se
                        continue
                    ino = getattr(ds, "InstanceNumber", None)
                    try:
                        ino_int: Optional[int] = int(ino)
                    except (TypeError, ValueError):
                        ino_int = None
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
                    desc = str(
                        _safe(ds, "SeriesDescription", "") or se.description
                    )
                    new_series[new_uid] = Series(
                        series_uid=new_uid,
                        modality=se.modality,
                        description=desc,
                        files=[path],
                        # The real SeriesNumber is identical for all
                        # packed files (it provides no discriminator),
                        # so promote InstanceNumber into the Series No
                        # column — that is how other viewers label these.
                        number=ino_int if ino_int is not None else se.number,
                        acq_number=anum_int,
                        instance_number=ino_int,
                        dicom_modality=se.dicom_modality,
                        acq_time=_acq_key(ds),
                    )
            study.series = new_series


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
                 volume: np.ndarray, is_color: bool = False):
        self.name = name
        self.path = path
        self._ds = ds                 # shared by UI seek + background prefetch
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
                    self.volume[i] = _decode_frame(self._ds, i)
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


def _is_color_ds(ds) -> bool:
    # IVUS is fundamentally a grayscale modality. Some scanners export
    # it as YBR_FULL_422 / SamplesPerPixel=3 for JPEG-baseline storage
    # efficiency — the chroma channels are near-neutral noise and the Y
    # channel carries the real signal. Treating those as colour gives
    # the image a faint tint, which the user sees as a bug. So: force
    # IVUS to the grayscale path regardless of PhotometricInterpretation.
    if str(getattr(ds, "Modality", "")).upper() == "IVUS":
        return False
    if int(getattr(ds, "SamplesPerPixel", 1) or 1) >= 3:
        return True
    return str(getattr(ds, "PhotometricInterpretation", "")) in _COLOR_PI


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
    """imagecodecs (libjpeg-turbo / SIMD) decode of ONE grayscale
    encapsulated frame — several times faster than pydicom's reference
    handler. Returns the ndarray, or None meaning 'use pydicom'.

    Safety: the first decoded frame of each dataset is self-checked against
    the reference decoder (exact for lossless transfer syntaxes, ≤1 LSB for
    lossy DCT JPEG). Any mismatch, unsupported syntax, colour data, or error
    permanently disables the fast path for that dataset, so a wrong pixel is
    never shown. Calls are serialised per plane (plane._lock), so the
    per-dataset flag writes are race-free.
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
            if spec is None or int(
                getattr(ds, "SamplesPerPixel", 1) or 1
            ) != 1:
                ds._mdv_nofast = True
                return None
            fn = getattr(ic, spec[0], None)
            if fn is None:
                ds._mdv_nofast = True
                return None
            ds._mdv_fast_fn = fn
            ds._mdv_fast_tol = spec[1]
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


def _decode_frame(ds, index: int) -> np.ndarray:
    """One frame ready for display: grayscale -> 2-D float (window/level
    applied later by the viewer); color -> (H, W, 3) uint8 RGB, already
    in display color space (YBR converted, palette LUT applied)."""
    arr = _raw_frame(ds, index)
    if not _is_color_ds(ds):
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
        try:
            rgb = np.asarray(convert_color_space(arr, pi, "RGB"))
        except Exception:
            rgb = arr
    else:  # RGB
        rgb = arr
    if rgb.ndim == 2:                       # safety: not actually color
        return rgb.astype(np.float32)
    rgb = rgb[..., :3]
    return np.ascontiguousarray(_to_u8(rgb))


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
    for i in range(maxf):
        for plane in planes:
            if should_stop():
                return
            if i >= plane.total_frames or plane._ready[i]:
                continue
            with plane._lock:
                if plane._ready[i]:             # UI seek decoded it first
                    continue
                plane.volume[i] = _decode_frame(plane._ds, i)
                plane._ready[i] = True
            # sleep(0) still yields the GIL briefly (responsive stop,
            # warm rate ~= raw decode speed); 4 ms hands the cine timer
            # a slice on every frame while playing.
            time.sleep(0.004 if is_playing() else 0)


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
        progress("Reading cine file…", 0, n_files * 2)

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
    if series.modality == Modality.CT:
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
        if px.ndim == 3:  # color: downsample as-is, no window/level
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
        diffs = diffs[diffs > 1e-6]
        if diffs.size:
            slice_mm = float(np.median(diffs))
    if not slice_mm:
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

    return LoadedSeries(
        modality=Modality.CT,
        volume=vol,
        spacing_mm=spacing,
        cine_fps=None,
        window=800.0,   # coronary/angio default
        level=200.0,
        slice_mm=slice_mm,
        header=_read_header(getattr(first, "filename", "") or ""),
        patient_basis=pbasis,
        series_uid=series.series_uid,
    )


def load_series(
    series: Series,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> LoadedSeries:
    if series.modality == Modality.CT:
        return load_ct(series, progress)
    return load_xa(series, progress)
