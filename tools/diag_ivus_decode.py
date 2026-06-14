"""Diagnose IVUS decode performance — grayscale vs colour — on this machine.

Usage:
    python tools/diag_ivus_decode.py <path-to-IVUS-folder-or-file> [num_frames]

Reports the transfer syntax / colour storage, whether the imagecodecs fast
decode path is available and stays enabled, and the average per-frame decode
time in BOTH grayscale (default IVUS) and forced-colour modes. This isolates
the Mac "playback stutter / seek freeze in colour" report: if colour ms/frame
is much higher than grayscale, the slowdown is the colour decode/transform; if
the fast path falls back ("nofast"), imagecodecs isn't decoding this codec here.
"""
import os
import sys
import time

import numpy as np
import pydicom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core import dicom_io


def _first_dicom(path):
    if os.path.isfile(path):
        return path
    cands = []
    for dp, _d, files in os.walk(path):
        for fn in files:
            cands.append(os.path.join(dp, fn))
    for p in sorted(cands):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            if "PixelData" in ds or getattr(ds, "Rows", 0):
                return p
        except Exception:
            continue
    raise SystemExit(f"no DICOM found under {path}")


def _make_plane(ds):
    nf = max(1, int(dicom_io._to_float(getattr(ds, "NumberOfFrames", 1), 1)))
    f0 = dicom_io._decode_frame(ds, 0)
    vol = np.zeros((nf,) + f0.shape, dtype=f0.dtype)
    vol[0] = f0
    return dicom_io.XAPlane("IVUS", "diag", ds, nf, vol,
                            is_color=(f0.ndim == 3))


def _time_all(plane, n):
    # Reset readiness so every frame is actually decoded (not a cache hit).
    plane._ready = np.zeros(plane.total_frames, dtype=bool)
    plane._ready[0] = True
    t0 = time.perf_counter()
    decoded = 0
    for i in range(1, n):
        plane.frame(i)
        decoded += 1
    dt = time.perf_counter() - t0
    return dt, decoded


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    f = _first_dicom(path)
    ds = pydicom.dcmread(f, force=True)
    ts = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", ""))
    nf = max(1, int(dicom_io._to_float(getattr(ds, "NumberOfFrames", 1), 1)))
    n = min(nmax, nf)

    print("file               :", f)
    print("Modality           :", getattr(ds, "Modality", "?"))
    print("TransferSyntaxUID  :", ts, "(", getattr(ds.file_meta,
          "TransferSyntaxUID", ""), ")" if hasattr(ds, "file_meta") else "")
    print("SamplesPerPixel    :", getattr(ds, "SamplesPerPixel", "?"))
    print("PhotometricInterp  :", getattr(ds, "PhotometricInterpretation", "?"))
    print("NumberOfFrames     :", nf, f"(timing first {n})")
    print("imagecodecs avail  :", dicom_io._imagecodecs() is not None)
    print("colour-capable     :", dicom_io._is_color_capable(ds))
    print("is_color_ds(IVUS)  :", dicom_io._is_color_ds(ds))
    print("-" * 60)

    # GRAYSCALE (default IVUS path)
    gp = _make_plane(ds)
    g_dt, g_n = _time_all(gp, n)
    print(f"GRAY  : {g_dt*1000/max(g_n,1):6.1f} ms/frame  "
          f"({g_n} frames, {g_dt:.2f}s total)  "
          f"shape={gp.volume.shape[1:]} dtype={gp.volume.dtype}")
    print("        fast-path:",
          "DISABLED (pydicom fallback)" if getattr(ds, "_mdv_nofast", False)
          else ("OK (imagecodecs)" if getattr(ds, "_mdv_fast_ok", False)
                else "unknown"),
          "| full-stack cached:" , getattr(ds, "_mdv_full", None) is not None)

    # COLOUR (forced) — fresh dataset so the fast-path flags re-evaluate cleanly
    ds2 = pydicom.dcmread(f, force=True)
    cp = _make_plane(ds2)
    dicom_io.apply_color_mode_to_planes([cp], True)
    c_dt, c_n = _time_all(cp, n)
    print(f"COLOR : {c_dt*1000/max(c_n,1):6.1f} ms/frame  "
          f"({c_n} frames, {c_dt:.2f}s total)  "
          f"shape={cp.volume.shape[1:]} dtype={cp.volume.dtype}")
    print("        fast-path:",
          "DISABLED (pydicom fallback)" if getattr(ds2, "_mdv_nofast", False)
          else ("OK (imagecodecs)" if getattr(ds2, "_mdv_fast_ok", False)
                else "unknown"),
          "| full-stack cached:", getattr(ds2, "_mdv_full", None) is not None)
    print("-" * 60)
    ratio = (c_dt / max(g_dt, 1e-9))
    print(f"colour / gray decode time ratio : {ratio:.2f}x")
    if ratio > 2.0:
        print(">>> Colour decode is the bottleneck (see fast-path lines above).")
    else:
        print(">>> Decode is comparable; stutter likely elsewhere (render/timer).")


if __name__ == "__main__":
    main()
