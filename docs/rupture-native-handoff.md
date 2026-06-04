# Rupture-Predictor native port — Mac handoff & parity checklist

This file is the cross-machine handoff for the **browser-independent (native)
Rupture-Predictor**. It is committed to the repo (not in any one machine's
local Claude memory, which does NOT sync across machines) so a Claude Code
session on the Mac mini can pick up exactly here.

## TL;DR for the Mac session

The native port is **already implemented and on `main`** — you do **not** need
to re-do it. `git pull` and the browser-independent Rupture-Predictor is there
as the default. What remains on Mac is **Phase 6: real-hardware run + numeric
parity check against the legacy HTML**, then confirm it in the packaged Zip.

```
cd ~/Multi-DicomViewer
git pull origin main          # brings in the native port + later tweaks
source .venv/bin/activate
python run.py sample_data     # or open a real IVUS/XA folder
```

## Why this exists

The original Rupture-Predictor was a separate **browser/HTML tool**
(`multi_dicomviewer/resources/Rupture-Predictor.html`, JavaScript). It was
ported to a **self-contained PyQt6 window** so the app no longer hands off to
an external browser. Being pure **Qt + numpy (no VTK / no OpenGL)**, the native
tool is expected to run fine on the macOS **NoCT** build — unlike the CT viewer
(see `docs/mac-ct-port.md`).

## Status (as of commit 454827f, 2026-06-04)

- Implemented on Windows, committed, pushed to `origin/main`:
  - `5e01d9c` — Native Rupture-Predictor: PyQt6 port of the browser tool.
  - `6d77258` — enlarge (×2.5) + colour-code the Stretch Ratio result.
- **Mac: NOT yet verified on real hardware.** Phases 1–5 done on Windows;
  this doc is Phase 6 (Mac run + parity + packaged-Zip check).

## What was done on Windows (the parity target)

- **`multi_dicomviewer/core/rupture_math.py`** — Qt-free math that mirrors what
  the HTML did in JavaScript (calibration, arc length, ∠A1-C-A2, virtual
  balloon, stretched adventitia length, stretch ratio, per-diameter and
  target-rate tables). This is the single source of truth for the numbers.
- **`multi_dicomviewer/ui/rupture_predictor_window.py`** — the native window:
  a point-picking image canvas (CH/CV/A1/A2/AC/B) + a side panel with the
  Stretch Ratio result, calibration readout, balloon-diameter input, and the
  two result tables. Renders the IVUS frame stepper or the XA displayed image.
- **`multi_dicomviewer/ui/main_window.py`**
  - `_open_rupture_predictor` — opens the **native** window (default). When
    DICOM pixel spacing is present it passes `calib` so the manual **CH/CV
    calibration steps are skipped** and the workflow starts at A1.
  - `_open_rupture_predictor_browser` — the **legacy HTML** path, kept behind
    the env flag `MDV_RUPTURE_BROWSER=1` for A/B comparison. It writes a
    self-contained session HTML to a temp file and opens it in the **system
    default browser** (Safari on Mac) — no QtWebEngine dependency.

## Legacy A/B comparison (how to check parity)

```
# native (default):
python run.py <dicom_folder>

# legacy HTML in Safari, same data, for side-by-side numbers:
MDV_RUPTURE_BROWSER=1 python run.py <dicom_folder>
```

Open Tools ▸ Rupture-Predictor in each, pick the SAME points, and compare the
Stretch Ratio (and the per-diameter table). They should agree to ~0.01–0.02.

## Phase 6 — Mac TODO (do these, in order)

1. `git pull origin main`; `pip install -r requirements-mac.txt` if anything
   changed.
2. Launch, open an **IVUS** series (and separately an **XA** still), Tools ▸
   Rupture-Predictor. Confirm the native window opens (no crash) and
   point-picking + zoom work.
3. With DICOM spacing present, confirm the **CH/CV calibration step is skipped**
   (workflow starts at A1) and the Calibration readout shows px/mm.
4. **Parity:** rerun with `MDV_RUPTURE_BROWSER=1` (opens Safari), repeat the
   same picks, confirm the Stretch Ratio matches the native value.
5. Verify the **result label** renders correctly on macOS: the number is ~2.5×
   the caption (16→40 pt) and sits on the right colour chip —
   **<1.5 blue / 1.5–1.8 yellow / ≥1.8 red** (font metrics differ on macOS, so
   eyeball the chip + number layout).
6. Confirm it all works in the **packaged Zip** from the GitHub Actions Mac
   build, not just the dev venv.

## Mac-specific gotchas to watch

- **Fonts / rich text:** the result label uses HTML rich text (`<span>` with
  `font-size` + `background-color`). Verify the chip + 40 pt number look right
  with macOS's default font; adjust pt size only if it clips.
- **Packaging:** if you want the legacy fallback available in the Mac Zip, the
  PyInstaller spec must bundle `resources/Rupture-Predictor.html`. The native
  default path does **not** need it. (`Multi-DicomViewer.spec`.)
- **IVUS on NoCT build:** IVUS/XA route to the pure-Qt XAViewer/IVUSViewer, so
  the native Rupture tool should be fully functional on the Mac NoCT build.
  Confirm the IVUS frame stepper hands frames over correctly on Mac.

## Verification checklist

- [ ] Native window opens on Mac; point-picking + zoom work.
- [ ] DICOM-calibrated path skips CH/CV (starts at A1).
- [ ] Stretch Ratio matches the legacy HTML within ~0.01–0.02.
- [ ] Result label: ×2.5 number + correct colour band (blue/yellow/red).
- [ ] Both result tables (径ごと / 目標伸展率→必要直径) populate.
- [ ] Works in the packaged Mac Zip, not only `python run.py`.

## Pointers

- Commits: `5e01d9c` (native port), `6d77258` (result display). Read with
  `git log -p 5e01d9c -- multi_dicomviewer/ui/rupture_predictor_window.py`.
- Related handoff: `docs/mac-ct-port.md` (the separate CT/pygfx effort).
