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

## Using it

- The browser groups files into Patient ▸ Study ▸ Series. Patients that have
  both a CT and an XA series are flagged with `⚹ CT+XA`.
- Click a **CT** series → loads into the left MPR pane. Click an **XA** series →
  loads into the right cine pane. Pick one of each to compare the same patient.
- **XA pane:** Play/Pause, frame slider, fps, window/level, Distance/Angle tools
  (millimetres when the cine carries `ImagerPixelSpacing`, pixels otherwise).
- **CT pane:** axial/coronal/sagittal MPR, per-plane slice sliders, shared HU
  window/level with coronary/mediastinum/lung/bone presets.

## Scope / next steps

Implemented: DICOM I/O, study browser, linked layout, XA cine + measurement,
CT orthogonal MPR.

Natural extensions (not built): curved MPR / centerline for coronaries, volume
rendering, QCA stenosis %, measurement persistence per study, compressed-syntax
coverage beyond the bundled `pylibjpeg` codecs.
