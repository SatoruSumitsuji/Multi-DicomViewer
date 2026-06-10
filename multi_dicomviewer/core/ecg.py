"""Extract an ECG (or other physiological) waveform trace from an XA/angio
DICOM dataset, for the cine viewer's ECG strip.

Two carriers are tried, in priority order:

1. **Waveform Sequence** (5400,0100) — the modern carrier. Decoded via
   pydicom's :meth:`Dataset.waveform_array` (which already applies channel
   sensitivity / baseline). The channel whose label looks like an ECG lead
   is taken; otherwise the first channel.

2. **Legacy Curve Data** (50xx,3000) — older cath-lab XA. Parsed by hand:
   the 50xx group whose *Type of Data* is ``ECG`` (or, failing that, the
   first curve group) supplies the samples; *Data Value Representation*
   (50xx,0103) gives the sample dtype and *Curve Dimensions* (50xx,0005)
   whether the data interleaves an axis we must drop.

:func:`read_ecg` returns an :class:`ECGTrace` or ``None`` when neither
carrier is present / decodable. The reader is deliberately defensive — any
malformed tag yields ``None`` rather than raising, so a viewer can call it
on every series without guarding each field itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Data Value Representation (50xx,0103) → numpy little-endian dtype.
#: Per the (retired) DICOM Curve module enumerated values.
_CURVE_DTYPE = {
    0: "<u2",   # US — unsigned short
    1: "<i2",   # SS — signed short
    2: "<f4",   # FL — single float
    3: "<f8",   # FD — double float
    4: "<i4",   # SL — signed long
}


@dataclass
class ECGTrace:
    """One physiological trace ready to plot.

    ``samples`` is a 1-D float array (amplitude over time). ``fs`` is the
    sampling rate in Hz, or 0 when unknown (legacy curves rarely state it —
    the viewer then maps frames onto the trace proportionally instead of by
    absolute time). ``label`` is a human channel name for the strip caption.
    """

    samples: np.ndarray
    fs: float
    label: str

    @property
    def n(self) -> int:
        return int(self.samples.shape[0]) if self.samples is not None else 0

    @property
    def duration_s(self) -> float:
        return self.n / self.fs if self.fs > 0 else 0.0


def read_ecg(ds) -> ECGTrace | None:
    """Best-effort ECG trace from *ds* (a pydicom Dataset). None if absent."""
    if ds is None:
        return None
    trace = _from_waveform(ds)
    if trace is not None and trace.n >= 2:
        return trace
    trace = _from_curve(ds)
    if trace is not None and trace.n >= 2:
        return trace
    return None


# --------------------------------------------------------------- waveform
def _looks_like_ecg(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ("ecg", "ekg", "lead", "electrocardiogr"))


def _waveform_channel_labels(item) -> list[str]:
    """Channel captions from an item's ChannelDefinitionSequence, aligned
    with the waveform_array column order. Missing names become ''."""
    out: list[str] = []
    chs = getattr(item, "ChannelDefinitionSequence", None) or []
    for c in chs:
        label = ""
        src = getattr(c, "ChannelSourceSequence", None)
        if src and len(src):
            label = str(getattr(src[0], "CodeMeaning", "") or "")
        if not label:
            label = str(getattr(c, "ChannelLabel", "") or "")
        out.append(label)
    return out


def _from_waveform(ds) -> ECGTrace | None:
    wf = getattr(ds, "WaveformSequence", None)
    if not wf:
        return None
    for idx, item in enumerate(wf):
        try:
            arr = ds.waveform_array(idx)
        except Exception:
            continue
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2 or arr.size == 0:
            continue
        labels = _waveform_channel_labels(item)
        # Prefer a channel that names itself an ECG lead; else first.
        col = 0
        for ci, lab in enumerate(labels):
            if ci < arr.shape[1] and _looks_like_ecg(lab):
                col = ci
                break
        samples = np.asarray(arr[:, col], dtype=np.float32)
        fs = 0.0
        try:
            fs = float(getattr(item, "SamplingFrequency", 0) or 0)
        except (TypeError, ValueError):
            fs = 0.0
        label = labels[col] if col < len(labels) and labels[col] else "ECG"
        return ECGTrace(samples, fs, label)
    return None


# ------------------------------------------------------------ legacy curve
def _from_curve(ds) -> ECGTrace | None:
    groups = sorted({
        el.tag.group for el in ds if 0x5000 <= el.tag.group <= 0x50FF
    })
    if not groups:
        return None

    def _type_of_data(g):
        el = ds.get((g, 0x0020), None)
        return str(getattr(el, "value", "") or "")

    # Prefer the curve explicitly typed as ECG; else the first one present.
    ordered = sorted(
        groups, key=lambda g: 0 if _looks_like_ecg(_type_of_data(g)) else 1
    )
    for g in ordered:
        data_el = ds.get((g, 0x3000), None)        # Curve Data
        if data_el is None:
            continue
        raw = data_el.value
        if not isinstance(raw, (bytes, bytearray)) or len(raw) < 2:
            continue
        rep_el = ds.get((g, 0x0103), None)         # Data Value Representation
        try:
            rep = int(getattr(rep_el, "value", 0) or 0)
        except (TypeError, ValueError):
            rep = 0
        dtype = _CURVE_DTYPE.get(rep, "<i2")
        try:
            vals = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        except Exception:
            continue
        if vals.size < 2:
            continue
        # Curve Dimensions: when >1 the data interleaves axes (e.g. an
        # explicit time axis + amplitude). Take the LAST column as the
        # dependent (amplitude) axis.
        dims = 1
        dim_el = ds.get((g, 0x0005), None)
        try:
            dims = max(1, int(getattr(dim_el, "value", 1) or 1))
        except (TypeError, ValueError):
            dims = 1
        if dims > 1 and vals.size % dims == 0:
            vals = vals.reshape(-1, dims)[:, -1]
        label = _type_of_data(g) or "ECG"
        return ECGTrace(np.ascontiguousarray(vals), 0.0, label)
    return None
