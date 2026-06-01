# Mac CT port — handoff & plan (pygfx)

This file is the cross-machine handoff for porting the CT viewer to Mac.
It is committed to the repo (not in any one machine's local Claude memory)
so a Claude Code session on the Mac mini can pick up exactly here.

## Why this exists

The CT viewer (`multi_dicomviewer/viewers/ct_viewer.py`) renders oblique
MPR with **VTK 9.x**, which depends on **OpenGL**. macOS only emulates
OpenGL on top of Metal, and that emulation **hangs** on CT MPR — confirmed
on an M1 Mac via a `sample` stack trace: ~14.7 GB memory, stuck in
`vtkOrderIndependentTranslucentPass::Render` → `glBlitFramebuffer` → Apple's
OpenGL-on-Metal layer.

So the macOS build currently ships as **Multi-DicomViewer-NoCT**: it refuses
to load CT (gate `BLOCK_CT = sys.platform == "darwin"` in
`multi_dicomviewer/config.py`; `_open_series` shows a Japanese popup before
any VTK code runs). XA / IVUS / NM / MRI all route to XAViewer/IVUSViewer,
which are pure Qt+numpy (no VTK) and should run fine on Mac.

The fix is to replace VTK with **pygfx** (wgpu → Metal native on Mac, DX12
on Windows), which avoids OpenGL entirely.

## Status (as of commit 2c922af, 2026-06-01)

- Windows dev machine: full app with VTK CT works.
- Mac mini M4 16GB: purchased and set up. Homebrew, Python 3.13 (`python3.13`),
  git all installed. Repo cloned to `~/Multi-DicomViewer`. A venv `.venv`
  was created and `pip install -r requirements-mac.txt` succeeded
  (PyQt6 + pygfx/wgpu/rendercanvas + the 2-D stack, no vtk).
- **Next action: run Phase 0 (the pygfx spike) and read its result.** This
  has NOT been run on the Mac yet.

## Phase 0 — feasibility spike (DO THIS FIRST on the Mac)

```
cd ~/Multi-DicomViewer
source .venv/bin/activate         # prompt shows (.venv)
python tools/pygfx_spike.py
```

`tools/pygfx_spike.py` builds a synthetic 256³ CT-like volume and renders an
oblique MPR slice with pygfx, printing the GPU backend and showing live fps.

**Pass criteria (all must hold):**
1. A window opens (no crash).
2. Terminal `[spike] adapter: ... backend=...` and the on-window overlay say
   **Metal** (NOT opengl) — this is the whole point: we escaped the OpenGL
   hang.
3. fps >= 15 for the 256³ volume.
4. Left-drag rotates the slice plane; right-drag = W/L; wheel = zoom;
   `R` reset; `Esc` quit.

If Phase 0 passes, pygfx is the confirmed VTK replacement and we proceed to
Phases 1-6 (build full feature parity with the current VTK CT viewer, then
flip `BLOCK_CT` off on Mac and rename back to plain Multi-DicomViewer).

If Phase 0 fails, capture the full terminal output + any traceback before
changing anything — the failure mode (install error vs. backend vs. perf)
decides the next step.

## What the current VTK CT viewer does (the parity target for Phases 1-6)

`ct_viewer.py` is an SSMview-style **dual-pane linked oblique MPR**
(vtkImageReslice → vtkImageMapToColors → vtkImageActor). Key behaviours the
pygfx port must reproduce:
- Two panes with independent per-pane frames `_frame={"A":(u,v,n),"B":...}`;
  a CrossLine drag relinks them (MOVE near the intersection, ROTATE on a
  line); the other pane reslices through the crosshair centre.
- Tools: ZOOM / MOVE / ROTATE / SPIN / PAGING / THICK (slab-MIP) / W-L,
  keys R/T/W/S/G/Z/V/C, 2-stage Reset, W/L presets (Coronary 800/200).
- HU ColorMap: `_ColorMapDialog` bands (colour + HU min/max + opacity);
  LUT via `_band_lut`; defaults `_DEFAULT_BANDS`.
- Angio angle readout: per-pane "LAOn CRAn" / "RAOn CAUn" of the pane normal
  (`_angio_angle`, `LoadedSeries.patient_basis`).
- Measure tools (Line/Polyline/Ellipse/Polygon) shared with XA/IVUS via
  `core/measure_geom.py`; on CT they also report HU stats.

CT loader provides `LoadedSeries.slice_mm` (z spacing) and `patient_basis`
(voxel→LPS). Reuse these; the geometry is already solved — only the renderer
changes (VTK → pygfx).

## Plan after Phase 0 (rough)

VTK stays for Windows initially; pygfx replaces it on Mac. Build the pygfx
MPR renderer behind the same viewer interface, port the tools/colormap/
readout, verify on `sample_data/ct`, then once pygfx is stable on Windows
too, retire VTK for a single codebase. Estimated 6-8 weeks.
