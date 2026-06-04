# Merging `mac-ct-pygfx` → `main` — conflict resolution guide

Cross-machine guide for folding the pygfx/Metal CT viewer branch into `main`,
so one codebase builds both platforms (CT on Mac via pygfx, CT on Windows via
VTK, everything else shared). Written from a **trial merge run on Windows**
(`git merge --abort`-ed afterwards) so the conflicts below are the real ones.

## Pre-state (verified)

- Merge base: `d1f0961`. Both sides advanced from there in parallel, so the
  same files were edited on each side — most overlaps are the *same feature
  built twice* and resolve trivially.
- `main` is already prepped: commit **`df6a55f`** made `measure_geom` a
  **superset** (added `ellipse_axes` + `ellipse_drag`; XA/IVUS + VTK-CT now
  delegate to the shared `ellipse_drag`). So `ct_viewer_pygfx.py` imports
  (`ellipse_axes/ellipse_cab/ellipse_drag/ellipse_from_major/ellipse_outline`)
  are all satisfied by main's `measure_geom` **unchanged**.
- ⚠️ **Windows pauses all pushes to `main` until this merge is pushed**, or the
  final fast-forward breaks.

## Start the merge (on the Mac)

```bash
cd ~/Multi-DicomViewer
git fetch origin
git checkout mac-ct-pygfx
git merge origin/main          # brings df6a55f in; conflicts listed below
```

Trial merge produced **5 conflicted files** (`config.py` and `main_window.py`
auto-merged cleanly — see the post-merge checks for those).

## Per-file resolution

| File | Conflicts | Nature | Action |
|------|-----------|--------|--------|
| `core/measure_geom.py` | 4 | All semantically identical; **main is a strict superset** | **Take main's whole file** |
| `viewers/image_canvas.py` | 7 | **All identical logic, comments only** | Keep either side per hunk (use main's) |
| `viewers/ivus_viewer.py` | 4 | branch `count()-3` vs main helper | **Take main's side** |
| `viewers/xa_viewer.py` | 5 | Same UI features built twice | **Take main's side** (+ keep branch W/L styling, see note) |
| `ui/study_browser.py` | 2 | **Genuine behaviour difference** | **DECISION — see below** |

### `core/measure_geom.py` — take main wholesale
main has every ellipse helper the branch defined (`ellipse_cab`, `ellipse_axes`,
`ellipse_outline`, `ellipse_from_major`, `ellipse_drag`, ellipse `major_minor`)
**plus** `ellipse_params`. The bodies are equivalent; main is the canonical
superset. During the merge (`origin/main` = "theirs"):
```bash
git checkout --theirs multi_dicomviewer/core/measure_geom.py
git add multi_dicomviewer/core/measure_geom.py
# sanity: every symbol ct_viewer_pygfx imports must exist
python -c "from multi_dicomviewer.core import measure_geom as g; \
[getattr(g,n) for n in ('ellipse_axes','ellipse_cab','ellipse_drag','ellipse_from_major','ellipse_outline')]; print('OK')"
```

### `viewers/image_canvas.py` — cosmetic only
All 7 hunks are the *same code* with different comments (oblique-ellipse
outline, `angle_at(pts[1],pts[0],pts[2])`, vertex anchor `pts[1]`,
ellipse→polygon `[maj0,min0,maj1,min1]`, `ellipse_drag`, `ellipse_from_major`
draft + preview). **Do NOT `checkout --theirs`** here (it would drop the
branch's non-conflicting edits). Open the file, delete the `<<<<`/`====`/`>>>>`
markers keeping main's block in each of the 7 spots, `git add`.

### `viewers/ivus_viewer.py` — take main's helper
4 hunks: replace `self._series_nav_row.insertWidget(count()-3, btn)` with
main's `self._insert_series_nav_widget(btn)`. Consistent with taking main's
`xa_viewer` (which defines that helper + `_series_nav_right_anchor`).
```bash
git checkout --theirs multi_dicomviewer/viewers/ivus_viewer.py && git add -- multi_dicomviewer/viewers/ivus_viewer.py
```
(safe here — main's ivus_viewer changes are a superset of the branch's nav tweaks).

### `viewers/xa_viewer.py` — take main, keep one extra
5 hunks (toolbar anchor, Play size, seek-slider style, comments). Take **main's**
side for all:
- Play button = main's **×1.3** (the size the user explicitly tuned this session).
- Seek slider = main's inline white-disc/blue-dot QSS **with `setMinimumHeight(24)`**
  — this is the user-verified fix for the "handle clipped top/bottom" report;
  the branch's `_SEEK_SLIDER_QSS` lacks the height reserve, so do NOT use it.
- Toolbar = main's `_insert_series_nav_widget` + `_series_nav_right_anchor`.

⚠️ **Keep the branch's W/L slider styling**: the non-conflicting lines
`self.win_slider.setStyleSheet(_SLIDER_QSS)` / `lvl_slider` survive the merge and
reference the branch's module constant `_SLIDER_QSS` (a nice extra — leave it).
After resolving, the branch constant `_SEEK_SLIDER_QSS` is unused — delete its
definition or leave it. Then verify the module imports with no `NameError`:
```bash
python -c "import multi_dicomviewer.viewers.xa_viewer; print('xa OK')"
```

### `ui/study_browser.py` — ✅ DECIDED: take main's version
Both sides added the **same menu item** "Close (close series list)" but with
different behaviour. **User decision (2026-06-05): take main's contextual
version** — `_close_series_list(self, item)` collapses **only the study of the
right-clicked item** (series/study → that study; patient → all its studies).
Discard the branch's global "collapse-all" variant.

Resolve by taking main's side for **both** hunks — the method body AND its menu
line `act_close.triggered.connect(lambda: self._close_series_list(item))`. Since
main's study_browser change is the chosen one and a superset here:
```bash
git checkout --theirs multi_dicomviewer/ui/study_browser.py && git add -- multi_dicomviewer/ui/study_browser.py
```

## Post-merge checks (auto-merged files — verify semantics, not text)

`config.py` and `main_window.py` merged without conflict, but **confirm the CT
enablement actually survived**:
1. `config.py`: `BLOCK_CT` must be **False on darwin** (branch flipped it) and
   the Mac `APP_NAME` is whatever you want shipped (see `-NoCT` below).
2. `main_window.py`: the viewer factory must select **`ct_viewer_pygfx.CTViewer`
   on darwin** (Phase 8 wiring) AND main's features (native Rupture, Virtual
   BiXA menu, etc.) must still be present.
3. New branch-only files (`ct_viewer_pygfx.py`, `__main__.py`, `tools/*spike*`)
   carry over additively — no action.

## `-NoCT` naming
Once CT renders on the merged build, drop the suffix so the Mac build is just
`Multi-DicomViewer`: `config.py` (`APP_NAME`, `BLOCK_CT`), `Multi-DicomViewer.spec`
(`name`, `bundle_identifier`). Do this only **after** real-hardware CT is confirmed.

## Verification checklist (Mac, before finalizing)
- [ ] `measure_geom` superset imports OK; oblique ellipse draws/drags in XA/IVUS **and** CT.
- [ ] CT series **renders on Mac** (pygfx/Metal) — no BLOCK_CT popup.
- [ ] Merged main-side features intact: native Rupture (incl. per-diameter click + 2.5× colour label), angle click-order end1→vertex→end2, CT dashed draft, CT Measure History, Measure History/DICOM Tags top-right.
- [ ] `study_browser` "Close" behaves per the chosen design.
- [ ] App boots on Windows too (VTK CT path unaffected) — or at least imports clean.

## Finalize
```bash
# on mac-ct-pygfx, after resolving + verifying:
git commit                       # completes the merge
git push origin mac-ct-pygfx
# fold into main (fast-forwards: main is an ancestor of the merge):
git checkout main && git merge mac-ct-pygfx && git push origin main
```
Then tell Windows to `git pull` — and we're back to one trunk, one app, both
platforms. Windows may resume normal feature work on `main`.
