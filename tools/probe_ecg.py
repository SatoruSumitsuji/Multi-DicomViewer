"""Probe an XA / angio DICOM for embedded ECG / waveform / cardiac-timing
data — to see what's available for Virtual Bi-XA cardiac-phase sync.

Usage:
    python tools/probe_ecg.py "C:\\path\\to\\angio.dcm"
    python tools/probe_ecg.py "C:\\path\\to\\case_folder"   # scans *.dcm

Read-only. Reports, per file:
  * modality / manufacturer / model
  * frame timing (cine rate, frame time, frame time vector, #frames)
  * heart rate + trigger tags
  * legacy Curve Data (groups 5000-50FF) — common ECG carrier in cath-lab XA
  * Waveform Sequence (5400,0100) + (003A,*) channels / sampling rate
"""
from __future__ import annotations

import os
import sys

import pydicom

# Japanese-locale Windows defaults stdout to cp932, which can't encode some
# characters; force UTF-8 so the report never crashes on output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def _g(ds, kw, default="-"):
    return getattr(ds, kw, default)


def _probe(path: str) -> None:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not read: {e}")
        return

    print(f"  Modality      : {_g(ds, 'Modality')}")
    print(f"  Manufacturer  : {_g(ds, 'Manufacturer')} / "
          f"{_g(ds, 'ManufacturerModelName')}")
    print(f"  Frames        : {_g(ds, 'NumberOfFrames')}")
    print(f"  Cine Rate     : {_g(ds, 'CineRate')}  "
          f"(RecommendedDisplayFrameRate={_g(ds, 'RecommendedDisplayFrameRate')})")
    print(f"  Frame Time    : {_g(ds, 'FrameTime')} ms")
    ftv = getattr(ds, "FrameTimeVector", None)
    if ftv is not None:
        ftv = list(ftv)
        print(f"  FrameTimeVec  : {len(ftv)} entries, first few = {ftv[:5]}")
    print(f"  Heart Rate    : {_g(ds, 'HeartRate')}")
    for kw in ("TriggerTime", "NominalInterval", "LowRRValue",
               "HighRRValue", "IntervalsAcquired", "FrameReferenceTime"):
        if kw in ds:
            print(f"  {kw:<14}: {ds.data_element(kw).value}")

    # ---- legacy Curve Data (groups 0x5000-0x50FF) ----
    curve_groups = sorted({el.tag.group for el in ds
                           if 0x5000 <= el.tag.group <= 0x50FF})
    if curve_groups:
        print(f"  >> Curve Data present in {len(curve_groups)} group(s):")
        for grp in curve_groups:
            desc = ds.get((grp, 0x0022), None)   # Curve Description
            typ = ds.get((grp, 0x0020), None)    # Type of Data
            ndata = ds.get((grp, 0x0010), None)  # Number of Points
            print(f"       group {grp:#06x}: "
                  f"type={getattr(typ, 'value', '?')} "
                  f"desc={getattr(desc, 'value', '?')} "
                  f"points={getattr(ndata, 'value', '?')}")
        print("       -> likely the ECG trace (read (50xx,3000) Curve Data).")
    else:
        print("  Curve Data    : none")

    # ---- modern Waveform Sequence (5400,0100) ----
    wf = getattr(ds, "WaveformSequence", None)
    if wf:
        print(f"  >> Waveform Sequence: {len(wf)} item(s)")
        for i, item in enumerate(wf):
            nch = _g(item, "NumberOfWaveformChannels")
            ns = _g(item, "NumberOfWaveformSamples")
            fs = _g(item, "SamplingFrequency")
            print(f"       [{i}] channels={nch} samples={ns} "
                  f"sampling={fs} Hz")
            chs = getattr(item, "ChannelDefinitionSequence", None)
            if chs:
                for c in chs:
                    src = getattr(c, "ChannelSourceSequence", None)
                    label = "?"
                    if src and len(src):
                        label = _g(src[0], "CodeMeaning", "?")
                    print(f"           channel: {label}")
    else:
        print("  WaveformSeq   : none")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    target = argv[1]
    if os.path.isdir(target):
        files = []
        for root, _d, names in os.walk(target):
            files += [os.path.join(root, n) for n in names
                      if n.lower().endswith(".dcm")]
        files.sort()
    else:
        files = [target]
    if not files:
        print("No .dcm files found.")
        return 1
    for p in files:
        print(f"\n=== {p} ===")
        _probe(p)
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
