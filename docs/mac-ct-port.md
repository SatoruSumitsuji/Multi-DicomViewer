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
- **Phase 0 PASSED on the Mac mini M4 (2026-06-01).** See result below.
  pygfx is the confirmed VTK replacement → proceed to Phases 1-8.
- **Phases 1-8 IMPLEMENTED (2026-06-02), `multi_dicomviewer/viewers/ct_viewer_pygfx.py`.**
  Dual oblique MPR, all tools (zoom/move/rotate/spin/paging/W-L), QPainter
  overlays (crosshair/▲/slab guides/info/angio), HU colormap, measurements
  (line/polyline/ellipse/polygon/angle) with HU stats (numpy trilinear, no
  scipy), CPU slab-MIP, and the per-OS factory + `BLOCK_CT=False` are done and
  committed. Logic verified headless on the Mac (M4); GPU/QPainter DISPLAY and
  interactive picking still need a visual pass on the real CT_Sample series.
  Dual-pane linkage tuned to the product spec (Rotate relinks, Paging slides
  the other pane's centreline / keeps its image, SPIN sign +1) — NOT VTK parity.

## Phase 0 RESULT — PASSED (Mac mini M4 16GB, 2026-06-01)

Ran `python tools/pygfx_spike.py` on the Mac mini. All four pass criteria met:

1. Window opens, no crash.
2. `backend=Metal`, `device=Apple M4` (NOT opengl) — confirmed both via
   `wgpu.gpu.enumerate_adapters_sync()` and the on-window overlay. The whole
   point: we escaped the M1/Apple OpenGL-on-Metal hang that kills VTK CT.
3. **~29.5 fps** for the 256³ volume (continuous render mode, ~30 fps
   vsync-capped) — comfortably above the 15 fps bar.
4. Left-drag rotate, right-drag W/L, wheel zoom, `R` reset all work; the
   oblique slice (grey sphere disk + central cylinder cross-section)
   renders and updates live.

Versions on the Mac: wgpu 0.31.0, pygfx 0.16.0, Python 3.13.13, PyQt6.

### Gotcha found & fixed in the spike (IMPORTANT for Phases 1-6)

The spike initially rendered a black window: the overlay/fps updated but no
slice was visible. Root cause was a **coordinate-system bug**, not pygfx:

- pygfx places a `Volume`'s grid at **voxel coordinates** — the box spans
  `-0.5 .. N-0.5` in world space (verified via `get_world_bounding_box()` →
  `[-0.5,-0.5,-0.5]..[255.5,255.5,255.5]`), NOT the `-1..1` unit cube the
  original spike assumed.
- So the camera (`width=2.2`, looking at origin) framed an empty corner, and
  the slice plane `(0,0,1,0)` (z=0) cut the air face, not the centre.
- `VolumeSliceMaterial.plane = (a,b,c,d)` is evaluated in **world space**
  (see `renderers/wgpu/wgsl/volume_slice.wgsl`: box corners are multiplied by
  `u_wobject.world_transform` before the `a·x+b·y+c·z+d` test). With an
  identity object transform, world == voxel coords.

Fix applied to `tools/pygfx_spike.py`: frame the camera on the real volume box
(`cam.show_object(mesh, view_dir=(0,0,-1), up=(0,1,0), scale=1.2)`) and put the
slice plane through the volume centre (`d = -n·centre`, centre = `(N-1)/2`).
**The Phase 1-6 pygfx MPR renderer must likewise work in voxel/world coords
and pivot the reslice plane about the crosshair centre — not assume a
normalised cube.** The CT loader's `patient_basis` (voxel→LPS) is what maps
these voxel coords to physical space.

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

## De-risk spikes before Phase 1 (Mac mini M4, 2026-06-01)

Two unknowns not covered by Phase 0 were spiked before committing to the build.
Full phased plan: the approved plan file (parallel module `viewers/ct_viewer_pygfx.py`,
GPU `VolumeSliceMaterial` per pane with a face-on ortho camera, QPainter overlays,
numpy trilinear HU sampling, per-OS factory in `main_window.py`).

- **Overlay compositing — PASS** (`tools/overlay_spike.py`). A transparent
  `QPainter` QWidget composites correctly OVER the rendercanvas wgpu/Metal
  surface (crosshair + translucent panel visible), and pointer events route to
  the canvas handlers (drag moves the painted crosshair). → Crosshair, measures,
  text and angio readout will be drawn with QPainter on a per-pane overlay, NOT
  pygfx scene primitives. Reuses the existing 2-D world↔screen math directly.

- **Slab-MIP — approach B REJECTED, use CPU (approach C)** (`tools/slab_spike.py`).
  `VolumeMipMaterial` full-thickness MIP renders fine with the ortho camera, but
  its `clipping_planes` clip only the box-SURFACE fragment, not the ray
  integration: clip ON → entirely black at every thickness (box faces lie
  outside the slab). So a slab cannot be bounded with clipping planes. The THICK
  tool will instead CPU-resample N oblique planes within ±t/2 along N (reusing
  the Phase 6 numpy trilinear sampler), max-composite to a 2-D array, and display
  it (`gfx.Image`, same colormap). Correct, reuses code, exact HU. (Full-thickness
  GPU MIP works if ever wanted as a separate mode.)

## Plan after Phase 0 (approved phased plan)

VTK stays for Windows; pygfx replaces it on Mac via a NEW parallel module
`multi_dicomviewer/viewers/ct_viewer_pygfx.py` reusing the VTK file's pure-numpy
state machine, dialogs and measure logic verbatim. Phases: 1 minimal end-to-end
(render + W/L + paging) → 2 tools → 3 QPainter overlays → 4 colormap → 5
measurements → 6 HU stats (numpy trilinear) → 7 slab-MIP (CPU) → 8 integration
(per-OS factory, flip `BLOCK_CT` off on darwin, restore `APP_NAME`). Verify each
phase on the synthetic harness; final phase on real CT through the app on the M4.
Then, once stable on Windows too, retire VTK for a single codebase. ~6-8 weeks.
