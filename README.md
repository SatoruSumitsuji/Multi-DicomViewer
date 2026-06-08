# Multi-DICOMviewer

Research multi-modality DICOM viewer. **XA angiography (cath cine)** and
**cardiac CT (CCTA)** today, with **IVUS, OCT/OFDI, NM and MRI** on the roadmap.

> Research / educational use only. Not a medical device. Not for clinical diagnosis.

## Why one app

Angio and cardiac CT are normally separate tools. They are combined here so the
**same patient's CCTA and invasive angiogram load side by side** (CT in the left
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

Python 3.13, Windows. The CT viewer needs VTK; if VTK fails to import the app
still runs and the XA side stays fully usable (the CT pane shows why).

## Run

```powershell
# 1. make sample data (no patient data needed)
python tools/make_test_data.py sample_data

# 2. launch, pointing at any DICOM folder
python run.py sample_data
```

Or launch with no argument and use **File ▸ Open DICOM folder…**.

## macOS build — first launch

The prebuilt macOS app (`Multi-DicomViewer.app`, shipped in
`Multi-DicomViewer-macos.zip`) is **not code-signed**, so macOS Gatekeeper
blocks it the first time. Clear the quarantine flag once and it launches
normally afterwards.

1. **Unzip** the download to get `Multi-DicomViewer.app`, and move it into your
   **Applications** folder.
2. Open the **Terminal** app, then **copy & paste this single line** and press
   Return:

   ```bash
   { find /Applications ~/Applications ~/Downloads -maxdepth 2 -name "Multi-DicomViewer.app" -print; mdfind -name "Multi-DicomViewer.app"; } | sort -u | while read -r p; do xattr -dr com.apple.quarantine "$p"; done
   ```

   You don't need to `cd` anywhere — it searches by name, so your current
   directory doesn't matter, and it's safe to run even if more than one copy
   exists (e.g. both `/Applications` and `~/Applications`).
3. Open the app normally (double-click, or from Launchpad).

What the command does: `find` checks the usual spots, `mdfind` uses Spotlight as
a disk-wide fallback, `sort -u` de-duplicates, and `xattr -dr
com.apple.quarantine` removes the Gatekeeper flag from every copy found.

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
