# Multi-DICOMviewer

Research multi-modality DICOM viewer. **XA angiography (cath cine)** and
**CT (3D MPR)** today, with **IVUS, OCT/OFDI, NM and MRI** on the roadmap.

> **Terminology.** The CT viewer is a general 3D MPR tool (not heart-only), so
> we say just **"CT"**. Use **"cardiac CT"** only when the myocardium / valves
> are specifically in view, and **"coronary CTA"** only for coronary-vessel-only
> imaging — never the narrower term for the broader content.

> Research / educational use only. Not a medical device. Not for clinical diagnosis.

## Why one app

Angio and cardiac CT are normally separate tools. They are combined here so the
**same patient's CT and invasive angiogram load side by side** (CT in the left
pane, XA in the right), which is the correlation researchers actually want. The
DICOM core, study browser, and measurement model are fully shared; only the two
rendering modules are modality-specific.

## Architecture

```
multi_dicomviewer/
  app.py                  QApplication bootstrap
  config.py               HU window/level presets, cine defaults
  core/                   shared, GUI-free
    dicom_io.py           folder scan, study tree, lazy pixel load
    study_model.py        Patient > Study > Series, modality detection
    measurements.py       modality-agnostic distance/angle model
  ui/
    main_window.py        shell: browser dock + CT|XA compare layout
    study_browser.py      Patient/Study/Series tree (⚹ = has CT+XA)
    viewer_base.py        AbstractViewer contract
  viewers/
    image_canvas.py       fit-to-window 2D canvas + measurement overlay
    xa_viewer.py          cine playback, W/L, calibrated tools
    ct_viewer.py          VTK orthogonal MPR (axial/coronal/sagittal)
tools/
  make_test_data.py       synthetic DICOM (1 patient, CT + XA)
run.py                    entry point
```

## Setup

```powershell
python -m pip install -r requirements.txt
```

Python 3.13. **Windows/Linux** render CT with VTK (`requirements.txt`); **macOS**
renders CT with pygfx/wgpu→Metal (`requirements-mac.txt`) because VTK's
OpenGL→Metal path hangs. If the CT backend fails to import the app still runs and
the XA side stays fully usable (the CT pane shows why).

## Run

```powershell
# 1. make sample data (no patient data needed)
python tools/make_test_data.py sample_data

# 2. launch, pointing at any DICOM folder
python run.py sample_data
```

Or launch with no argument and use **File ▸ Open DICOM folder…**.

## macOS build — first launch

**Apple Silicon only.** The prebuilt macOS app is an **arm64** build, so it runs
on Apple Silicon Macs (M1 and later). **Intel Macs are not supported.**

**No workaround needed (v1.5.0 and later).** The prebuilt macOS app
(`Multi-DicomViewer.app`, shipped in `Multi-DicomViewer-macos.zip`) is now
**code-signed with a Developer ID, notarized by Apple, and stapled**, so
Gatekeeper passes it on a normal double-click — even offline.

1. **Unzip** the download to get `Multi-DicomViewer.app` and move it into your
   **Applications** folder.
2. **Double-click** to launch (or open it from Launchpad). That's it.

> **Old unsigned builds only (before v1.5.0).** If you are running a much older
> download that Gatekeeper still blocks, clear its quarantine flag once with the
> command below, then open it normally. **You do not need this for v1.5.0+
> (including v2.0.11).**
>
> ```bash
> { find /Applications ~/Applications ~/Downloads -maxdepth 2 -name "Multi-DicomViewer.app" -print; mdfind -name "Multi-DicomViewer.app"; } | sort -u | while read -r p; do xattr -dr com.apple.quarantine "$p"; done
> ```

## Using it

- The browser groups files into Patient ▸ Study ▸ Series. Patients that have
  both a CT and an XA series are flagged with `⚹ CT+XA`.
- Click a **CT** series → loads into the left MPR pane. Click an **XA** series →
  loads into the right cine pane. Pick one of each to compare the same patient.
- **XA pane:** Play/Pause, frame slider, fps, window/level, Distance/Angle tools
  (millimetres when the cine carries `ImagerPixelSpacing`, pixels otherwise).
- **CT pane:** axial/coronal/sagittal MPR, per-plane slice sliders, shared HU
  window/level with coronary/mediastinum/lung/bone presets.

> **macOS only — right-click to sharpen a Slab view.** With a Slab (thick) MPR,
> the image is rendered at reduced quality *while* you pan/rotate/wheel-page and
> snaps back to full quality when you stop, so paging stays smooth on low-memory
> Macs. If a coarse image ever lingers after you stop, **right-click the image**
> to force it back to full quality immediately. (No effect on the Windows build.)

### macOS only — the "HQ-Img" toggle (3D CT)

**HQ-Img** keeps the MPR at **full resolution even while you drag / zoom /
rotate** (it turns the coarse interactive preview OFF). It looks sharper, but
re-reconstructs every frame, so it is heavier on GPU and unified memory. The
button sits at the far left of the Plane row; the blue state means ON. Default
is OFF.

> **Apple Silicon only.** The macOS build is arm64 — **Intel Macs are not
> supported** (the app will not launch on them). The table below is therefore
> all Apple Silicon.

**Recommended: Apple Silicon with 16 GB or more.** On 8 GB Macs (mostly older
M1 / M2), large CT data (thin slices, many images) may stutter or briefly
freeze — turn HQ-Img **OFF** in that case.

| Class | Hardware (machine only) | HQ-Img | Behaviour (varies with data size) |
|---|---|---|---|
| 🟢 Safe | Apple Silicon **Pro / Max**, or **24 GB+** | Use freely | Smooth even on large data (600+ slices, thin) |
| 🟢 OK | Apple Silicon **16 GB** (M1–M5) | OK for normal data | Comfortable on typical CT (~hundreds of slices); watch only very large data + Slab together |
| 🟡 Conditional | Apple Silicon **8 GB** (mainly older M1 / M2; M4/M5 ship with 16 GB+) | Off recommended | Heavier as data grows; large data may stutter / briefly freeze |

> Figures are guidance derived from the processing involved, **not measured
> benchmarks**. (No effect on the Windows build.) A bilingual notice and a
> shareable image are in [`docs/`](docs/HQ-IMG_notice.md).

## Scope / next steps

Implemented: DICOM I/O, study browser, linked layout, XA cine + measurement,
CT orthogonal MPR.

Natural extensions (not built): curved MPR / centerline for coronaries, volume
rendering, QCA stenosis %, measurement persistence per study, compressed-syntax
coverage beyond the bundled `pylibjpeg` codecs.

## License

Copyright (C) 2025–2026 Satoru Sumitsuji.

Multi-DICOMviewer is free software, licensed under the **GNU General Public
License v3.0** — see [`LICENSE`](LICENSE). This choice is required because the
app links **PyQt6**, which is GPL-licensed. You may use, study, share, and
modify it under the GPL's terms; the complete corresponding source is this
repository.

Bundled third-party components and their licenses are listed in
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).

> **Not a medical device.** Research / educational use only; not for clinical
> diagnosis (see the notice at the top of this file).
