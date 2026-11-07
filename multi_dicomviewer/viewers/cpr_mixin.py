"""Backend-independent CPR / short-axis / CoSync logic shared by both CT
viewers (Windows VTK ``ct_viewer.py`` and Mac pygfx ``ct_viewer_pygfx.py``).

The short-axis (Curved-Planar-Reformation) feature is pure 3-D geometry over
the HU volume plus a small state machine — none of it needs a rendering
backend. Only the LIVE on-screen slice (VTK ``vtkImageReslice`` vs pygfx
``VolumeSliceMaterial.plane``) and the overlay actors/painter are backend
specific; those stay in each viewer. Everything here operates on the shared
``self._cpr`` state dict and the host attributes both viewers already expose
(``self._vol``, ``self._dims``, ``self._win``, ``self._lvl``, ``self._cpr``,
``self._measures``), so a viewer gains the whole shared surface by inheriting
this mixin.

``self._cpr`` keys (created by each viewer's ``_enter_cpr``):
    cl        CenterLine (core.centerline) — points / arclen / tangents
    u, v      (M,3) DISPLAY short-axis axes per sample (after T + rot)
    u0, v0    (M,3) base axes (pre-transform), from cl.frames, u negated
    idx       int   current REAL centreline sample index
    T         2x2   cumulative Rt90/Lt90/Flip display transform
    rot       float continuous in-plane rotation about the tangent (°)
    reversed  bool  scroll distal->proximal (IVUS pull-back direction)
    half      float cross-section half-FOV (mm)
    src, src_mi  the map pane and its trace-measure index
    ref_up    (3,)  RMF seed normal, kept for rebuilds

This module is GUI-free (only numpy + i18n) and headless-testable against a
tiny fake host that provides ``_vol`` / ``_dims`` / ``_cpr`` etc.
"""
from __future__ import annotations

import math

import numpy as np

from multi_dicomviewer.i18n import t
from multi_dicomviewer.core.centerline import CenterLine


class CPRMixin:
    """Shared short-axis geometry, CoSync query interface and numpy renderer.

    A host viewer inherits this and additionally provides the backend-specific
    pieces: ``_enter_cpr`` (build the state + enter the mode), ``_refresh`` /
    ``_cpr_sync_bar`` (redraw the panes + scrubber) and the live slice + overlay
    rendering. The methods here never touch a rendering backend."""

    # ---- scroll-direction mapping (IVUS pull-back parity) ----
    def _cpr_disp(self, idx):
        """Map a real centreline sample index <-> the DISPLAYED scrub position
        (reversed = distal->proximal, to match an IVUS pull-back). Involution."""
        c = self._cpr
        n = c["cl"].n
        return (n - 1 - int(idx)) if c.get("reversed") else int(idx)

    # ---- CoSync interface (short-axis as a synchronisable scrub source) ----
    def cpr_active(self) -> bool:
        return self._cpr is not None

    def cpr_sync_state(self):
        """(display index, count, rotation deg) of the short-axis scrub, or None
        when inactive. The DISPLAY index honours the reverse flag so CoSync
        frame numbers run distal->proximal like an IVUS."""
        c = self._cpr
        if c is None:
            return None
        return (self._cpr_disp(c["idx"]), int(c["cl"].n), float(c.get("rot", 0.0)))

    # ---- 3-D geometry of the current cross-section ----
    def _cpr_ctrl_pts3d(self):
        """The trace's 3-D control points (pseudo-centre points), or None."""
        c = self._cpr
        if c is None:
            return None
        src, mi = c.get("src"), c.get("src_mi")
        if src is None or mi is None or not (0 <= mi < len(self._measures[src])):
            return None
        return self._measures[src][mi].get("pts3d")

    def get_current_cpr(self):
        """(ctrl, points) of the current coronary centreline as lists of [x,y,z]
        world-mm, or None. Works from the built short-axis (self._cpr) OR, when
        the short-axis hasn't been built yet, straight from the DRAWN coronary
        polyline — the Coronary Tree only needs the centreline geometry, not the
        cross-section view. Shared by both viewers (VTK + pygfx)."""
        if self._cpr is not None:
            ctrl = self._cpr_ctrl_pts3d()
            if not ctrl or len(ctrl) < 2:
                return None
            pts = np.asarray(self._cpr["cl"].points, float)
        else:
            ctrl = self._coronary_trace_ctrl()
            if ctrl is None or len(ctrl) < 2:
                return None
            from multi_dicomviewer.core.centerline import CenterLine
            step = max(1e-3, min(self._dims))
            cl = CenterLine.from_points(np.asarray(ctrl, float), step_mm=step)
            if cl.n < 2:
                return None
            pts = np.asarray(cl.points, float)
        return ([list(map(float, np.asarray(P, float))) for P in ctrl],
                pts.round(4).tolist())

    def _cpr_register_to_tree(self) -> bool:
        """「保存登録」button: SAVE the current short-axis to a .cpr.json first
        (the user names the file), THEN register that CPR into the Coronary Tree —
        routed to its source CT by series UID, with the file name as the vessel
        name (保存後登録 in one click). Cancelling the save cancels the registration.
        Shared by both viewers (VTK + pygfx). Returns True when the file was saved
        (so Save&Exit can follow through), False when cancelled / nothing to save."""
        import os
        from PyQt6.QtWidgets import QMessageBox
        path = self._cpr_save()                  # Save As prompt → path, or None
        if not path:                             # cancelled / nothing to save
            return False
        data = self._cpr_build_data()            # includes series / src_dir / ctrl
        if data is None:
            return False
        name = os.path.basename(path)            # file name → vessel name
        for suf in (".cpr.json", ".json"):
            if name.lower().endswith(suf):
                name = name[:-len(suf)]
                break
        shell = self.window()
        if shell is None or not hasattr(shell, "coronary_register_cpr"):
            QMessageBox.information(
                self.window(), t("Coronary Tree"),
                t("保存しました。Coronary Tree パネルを開けませんでした。"))
            return True                          # file saved (registration skipped)
        try:
            shell.coronary_register_cpr(data, name or "vessel", path)
        except Exception as exc:                 # noqa: BLE001
            QMessageBox.warning(self.window(), t("Coronary Tree"),
                                t("登録に失敗しました: {e}", e=str(exc)))
        return True                              # file saved (register may have failed)

    def _cpr_save_and_exit(self) -> None:
        """「Save&Exit」button: save + register the short-axis, then — only if the
        save completed (not cancelled) — leave coronary MPR / short-axis mode
        automatically, so Exit need not be pressed separately. Shared by both
        viewers; _coronary_exit is each viewer's own."""
        if self._cpr_register_to_tree():
            self._coronary_exit()

    def _coronary_trace_ctrl(self):
        """3-D control points of the drawn coronary centreline polyline when no
        short-axis is built yet — the most recent polyline trace with pts3d
        (prefer one tagged as a CPR source). None if there is no such trace."""
        best = None
        for k in ("A", "B"):
            for m in (self._measures.get(k, []) if isinstance(self._measures, dict)
                      else []):
                if m.get("type") == "polyline":
                    p3 = m.get("pts3d")
                    if p3 and len(p3) >= 2:
                        best = p3
                        if m.get("_cpr_src"):
                            return p3        # the canonical CPR-source trace
        return best

    # ---- Auto-tracking (core.cpr_autotrack) ------------------------------
    def _cpr_auto_on(self) -> bool:
        """Auto toggle (Draw row). Default ON: the clicked trace points become
        start / via / end of a lumen-following centreline."""
        return bool(getattr(self, "_cpr_auto", True))

    _CPR_AUTO_STAGE = {
        "enhance": "血管を強調中",
        "path": "経路を探索中",
        "center": "中心を補正中",
        "ctrl": "編集点を配置中",
    }

    def _cpr_finish_trace(self, pane, m) -> None:
        """A Draw trace was finished: (Auto) trace it through the lumen in the
        background, then build the short-axis from it. Shared by both viewers'
        _commit_draft. With Auto off this runs synchronously, as before."""
        def build():
            lst = self._measures.get(pane, [])
            mi = next((i for i, mm in enumerate(lst) if mm is m), None)
            if mi is None:                     # trace deleted / Exit meanwhile
                return
            self._enter_cpr(pane, mi)
            if self._cpr is not None:
                # Fit (fixed HU 200–700 disc, 2.5 mm) can pull a THIN auto-traced
                # point onto a neighbouring vein; the auto trace is already
                # centred sub-voxel, so only a hand-clicked trace is auto-Fitted.
                if not m.get("_cpr_auto"):
                    self._cpr_fit()
                self._cpr_jump_ctrl_end(False)   # most proximal edit point (|◀)
                self._cpr_auto_report()          # flag low-contrast stretches
        self._cpr_auto_run(m, build)

    def _cpr_auto_run(self, m, on_done) -> None:
        """Run the auto trace for trace measure *m* in a worker thread with a
        small non-modal "Auto-Trace中…" progress window (stage + bar + Cancel),
        then apply it to *m* and call *on_done* on the GUI thread. Calls
        *on_done* right away when Auto is off / not applicable. Non-modal on
        purpose: a modal shown from the finishing click swallows its release on
        macOS (the dead-mouse bug)."""
        self._cpr_auto_gaps = []
        vol = getattr(self, "_vol", None)
        p3 = m.get("pts3d")
        if (not self._cpr_auto_on() or vol is None or not p3 or len(p3) < 2):
            on_done()
            return
        if getattr(self, "_cpr_auto_busy", False):
            return                                   # one trace at a time
        import threading
        from PyQt6.QtCore import QTimer, Qt
        from PyQt6.QtWidgets import QProgressDialog
        from multi_dicomviewer.core.cpr_autotrack import (TraceCancelled,
                                                          trace_vessel)
        self._cpr_auto_busy = True
        st = {"stage": "enhance", "i": 1, "n": max(1, len(p3) - 1),
              "frac": 0.0, "res": None, "done": False, "cancel": False}

        def prog(stage, i, n, frac):
            st.update(stage=stage, i=i, n=n, frac=frac)
            return not st["cancel"]

        dims = tuple(float(d) for d in self._dims)
        clicks = [np.asarray(q, float) for q in p3]

        def work():
            try:
                st["res"] = trace_vessel(vol, dims, clicks, progress=prog)
            except TraceCancelled:
                st["res"] = None
            except Exception as e:                           # noqa: BLE001
                import traceback
                st["res"] = None
                st["error"] = "".join(traceback.format_exception(e))
            finally:
                st["done"] = True
        dlg = QProgressDialog(self)
        dlg.setWindowTitle(t("Auto-Trace"))
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        dlg.setRange(0, 100)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False); dlg.setAutoReset(False)
        dlg.setCancelButtonText(t("Cancel"))
        dlg.setMinimumWidth(320)

        def label():
            stage = t(self._CPR_AUTO_STAGE.get(st["stage"], ""))
            seg = (t("区間 {i}/{n}").format(i=st["i"], n=st["n"]) + "　"
                   if st["n"] > 1 and st["stage"] in ("enhance", "path") else "")
            return t("Auto-Trace中…") + "\n" + seg + stage
        dlg.setLabelText(label())
        dlg.canceled.connect(lambda: st.update(cancel=True))
        dlg.show()
        threading.Thread(target=work, daemon=True).start()
        timer = QTimer(self)

        def poll():
            if not st["done"] and dlg.wasCanceled():
                st["cancel"] = True              # worker stops at its next stage
            if not st["done"]:
                dlg.setValue(int(round(100 * st["frac"])))
                dlg.setLabelText(label())
                return
            timer.stop()
            cancelled = bool(st["cancel"])       # read BEFORE closing: closing a
            try:                                 # QProgressDialog emits canceled()
                dlg.canceled.disconnect()
            except TypeError:
                pass
            dlg.close(); dlg.deleteLater()
            st["cancel"] = cancelled
            self._cpr_auto_busy = False
            res = st["res"]
            if st.get("error"):
                import sys
                print("Auto-Trace failed:", st["error"], file=sys.stderr)
            if st["cancel"]:
                return                  # cancelled: leave the clicked trace as is
            if res is not None and res.ok and len(res.ctrl_mm) >= 2:
                self._cpr_auto_store(m, res)
            on_done()
        timer.timeout.connect(poll)
        timer.start(60)

    def _cpr_auto_apply(self, m) -> None:
        """Synchronous variant of the auto trace (headless tests / scripts):
        replace *m*'s clicked ``pts3d`` with the lumen-following centreline.
        Leaves *m* unchanged when Auto is off, there is no volume, or tracking
        fails. Stores the low-contrast gaps in ``self._cpr_auto_gaps``."""
        self._cpr_auto_gaps = []
        if not self._cpr_auto_on():
            return
        vol = getattr(self, "_vol", None)
        p3 = m.get("pts3d")
        if vol is None or not p3 or len(p3) < 2:
            return
        from multi_dicomviewer.core.cpr_autotrack import trace_vessel
        try:
            res = trace_vessel(vol, tuple(float(d) for d in self._dims),
                               [np.asarray(q, float) for q in p3])
        except Exception:                                    # noqa: BLE001
            res = None
        if res is None or not res.ok or len(res.ctrl_mm) < 2:
            return
        self._cpr_auto_store(m, res)

    def _cpr_auto_store(self, m, res) -> None:
        """Write a successful TraceResult into trace measure *m*."""
        src = None
        for k in ("A", "B"):
            if any(mm is m for mm in self._measures.get(k, [])):
                src = k
        ctrl = np.asarray(res.ctrl_mm, float)
        # 自動CPR編集点距離 (Settings): thin the auto edit points to that spacing,
        # always keeping the hand-clicked start / via / end (m's pts3d are still
        # the clicks here).
        try:
            from multi_dicomviewer.core import settings
            from multi_dicomviewer.core.cpr_autotrack import thin_ctrl
            sp = float(settings.load_coronary_params().get("cpr_ctrl_mm", 3))
            clicks = [np.asarray(q, float) for q in (m.get("pts3d") or [])]
            th = thin_ctrl(res.path_mm, ctrl, clicks, sp)
            if len(th) >= 2:
                ctrl = th
        except Exception:                        # noqa: BLE001
            pass
        m["pts3d"] = [list(map(float, q)) for q in ctrl]
        if src is not None and hasattr(self, "_world3d_to_out"):
            m["pts"] = [self._world3d_to_out(src, q) for q in ctrl]
        m["_cpr_auto"] = True
        self._cpr_auto_gaps = list(res.gaps)

    def _cpr_auto_report(self) -> None:
        """After an auto-tracked CPR is built: tell the user about stretches
        where the trace crossed low contrast (stenosis / occlusion / calcium) so
        they check them. Deferred — a modal shown inside the finishing click
        swallows its release on macOS."""
        gaps = list(getattr(self, "_cpr_auto_gaps", []) or [])
        if not gaps:
            return
        txt = ", ".join(f"{a:.0f}–{b:.0f} mm" for a, b in gaps)

        def show():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, t("Short-axis"),
                t("自動トレースが造影の弱い区間を通過しました（始点からの距離）: "
                  "{gaps}\n狭窄・閉塞・石灰化の可能性があります。短軸で確認し、"
                  "必要なら編集点を修正してください。").format(gaps=txt))
        defer = getattr(self, "_after_mouse_release", None)
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, (lambda: defer(show)) if defer else show)

    #: Fit lumen-snap parameters (contrast CTA): a control point is moved to the
    #: HU-weighted centroid of voxels in [LO, HI] HU within a RADIUS-mm disc of the
    #: vessel's SHORT-AXIS cross-section; if there is NO in-range voxel within the
    #: disc the point is left where it is.
    _CPR_FIT_HU_LO = 200.0
    _CPR_FIT_HU_HI = 700.0
    _CPR_FIT_RADIUS_MM = 2.5

    def _cpr_fit_tangent(self, p3, i):
        """Local vessel tangent at control point i (finite difference of the
        neighbours; the adjacent segment at the ends)."""
        n = len(p3)
        if n < 2:
            return np.array([0.0, 0.0, 1.0])
        if i <= 0:
            d = np.asarray(p3[1], float) - np.asarray(p3[0], float)
        elif i >= n - 1:
            d = np.asarray(p3[-1], float) - np.asarray(p3[-2], float)
        else:
            d = np.asarray(p3[i + 1], float) - np.asarray(p3[i - 1], float)
        nd = float(np.linalg.norm(d))
        return d / nd if nd > 1e-9 else np.array([0.0, 0.0, 1.0])

    def _cpr_snap_ctrl_to_lumen(self, P, tangent):
        """Move control point *P* (world mm) to the HU-weighted centroid of the
        contrast lumen in its SHORT-AXIS cross-section (⟂ *tangent*): sample a
        RADIUS-mm disc, keep voxels in [HU_LO, HU_HI], return their HU-weighted
        centroid — or P unchanged when no in-range voxel is within the disc."""
        vol = getattr(self, "_vol", None)
        if vol is None:
            return np.asarray(P, float)
        tn = np.asarray(tangent, float)
        tl = float(np.linalg.norm(tn))
        if tl < 1e-9:
            return np.asarray(P, float)
        tn = tn / tl
        # Two orthonormal in-plane axes ⟂ the tangent.
        ref = np.array([0.0, 0.0, 1.0]) if abs(tn[2]) < 0.9 \
            else np.array([1.0, 0.0, 0.0])
        u = np.cross(tn, ref)
        u /= (np.linalg.norm(u) or 1.0)
        v = np.cross(tn, u)
        sx, sy, sz = self._dims
        zmax, ymax, xmax = vol.shape
        r = float(self._CPR_FIT_RADIUS_MM)
        step = max(0.25, min(sx, sy, sz) * 0.5)
        lo, hi = self._CPR_FIT_HU_LO, self._CPR_FIT_HU_HI
        P0 = np.asarray(P, float)
        acc = np.zeros(3)
        wsum = 0.0
        a = -r
        while a <= r + 1e-9:
            b = -r
            while b <= r + 1e-9:
                if a * a + b * b <= r * r:
                    Q = P0 + a * u + b * v
                    vx = int(round(Q[0] / sx))
                    vy = int(round(Q[1] / sy))
                    vz = int(round(Q[2] / sz))
                    if 0 <= vz < zmax and 0 <= vy < ymax and 0 <= vx < xmax:
                        huv = float(vol[vz, vy, vx])
                        if lo <= huv <= hi:
                            acc += huv * Q
                            wsum += huv
                b += step
            a += step
        if wsum <= 1e-9:                 # no contrast lumen within the disc
            return P0
        return acc / wsum

    def _cpr_fit(self):
        """Fit: snap EVERY editable control point to the contrast-lumen centre in
        its short-axis cross-section (HU 200–700, 2.5 mm radius; a point with no
        in-range voxel is left alone), then rebuild the short-axis centreline from
        the snapped points KEEPING the display state (rotation / flip / reverse /
        FOV / arc-length position). Shared by both viewers."""
        from PyQt6.QtWidgets import QMessageBox
        c = self._cpr
        if c is None:
            QMessageBox.information(self, t("Short-axis"),
                                   t("短軸(CPR)を作成してからFitしてください。"))
            return
        src, mi = c.get("src"), c.get("src_mi")
        if (src not in ("A", "B") or mi is None
                or not (0 <= mi < len(self._measures.get(src, [])))):
            QMessageBox.information(self, t("Short-axis"),
                                   t("中心線トレースが見つかりません。"))
            return
        m = self._measures[src][mi]
        p3 = m.get("pts3d")
        if not p3 or len(p3) < 2:
            QMessageBox.information(self, t("Short-axis"),
                                   t("中心線の点が不足しています。"))
            return
        # Snap every control point to the lumen centre in its cross-section.
        p3f = [np.asarray(q, float) for q in p3]
        snapped = [self._cpr_snap_ctrl_to_lumen(P, self._cpr_fit_tangent(p3f, i))
                   for i, P in enumerate(p3f)]
        m["pts3d"] = [list(map(float, q)) for q in snapped]
        m["pts"] = [self._world3d_to_out(src, q) for q in snapped]
        old_n = c["cl"].n
        frac = (c["idx"] / (old_n - 1)) if old_n > 1 else 0.0
        saved = {"T": np.asarray(c["T"], float).copy(),
                 "rot": float(c.get("rot", 0.0)),
                 "reversed": bool(c.get("reversed", False)),
                 "half": float(c.get("half", 25.0)),
                 "ref_up": np.asarray(c.get("ref_up", (0.0, 0.0, 1.0)), float)}
        # Rebuild from the edited trace (resets the display state), then restore.
        self._enter_cpr(src, mi, ref_up=saved["ref_up"])
        c2 = self._cpr
        if c2 is None:
            return
        c2["T"] = saved["T"]
        c2["rot"] = saved["rot"]
        c2["reversed"] = saved["reversed"]
        c2["half"] = saved["half"]
        n2 = c2["cl"].n
        c2["idx"] = int(min(max(round(frac * (n2 - 1)), 0), n2 - 1))
        self._cpr_apply_xform()          # rebuild u,v from u0,v0 + T + rot
        self._cpr_sync_bar()
        self._refresh(reset_cam=True)

    def _cpr_frame(self):
        """(origin, u, v, tangent) of the current cross-section."""
        c = self._cpr
        i = c["idx"]
        return (np.asarray(c["cl"].points[i], float), c["u"][i], c["v"][i],
                np.asarray(c["cl"].tangents[i], float))

    # ---- edit-point (control-point) navigation & editing -----------------
    def _cpr_ctrl_indices(self):
        """Dense-centreline index of EACH control point, aligned to the trace's
        pts3d. Cached on _cpr; invalidated on rebuild/Fit so it always matches
        the current centreline. Empty when there's no short-axis."""
        c = self._cpr
        if c is None:
            return []
        p3 = self._cpr_ctrl_pts3d()
        n = len(p3) if p3 else 0
        cached = c.get("_ctrl_idx")
        # Use the cache only when it still matches the control-point COUNT — an
        # edit/move that changed pts3d without invalidating it would otherwise
        # leave a too-long cache (→ IndexError on p3[k]).
        if cached is not None and len(cached) == n:
            return cached
        out = []
        if p3:
            pts = np.asarray(c["cl"].points, float)
            for P in p3:
                out.append(int(np.argmin(
                    np.linalg.norm(pts - np.asarray(P, float), axis=1))))
        c["_ctrl_idx"] = out
        return out

    def _cpr_at_ctrl(self):
        """Index (into pts3d) of the control point the CURRENT cross-section sits
        exactly on, or None between control points (an interpolated section)."""
        c = self._cpr
        if c is None:
            return None
        cur = c["idx"]
        for k, di in enumerate(self._cpr_ctrl_indices()):
            if di == cur:
                return k
        return None

    def _cpr_jump_ctrl(self, direction: int) -> None:
        """Alt+F / Alt+A in the short-axis: move the cross-section to the NEXT /
        PREVIOUS control (edit) point so it can be adjusted."""
        c = self._cpr
        if c is None:
            return
        idxs = sorted(set(self._cpr_ctrl_indices()))
        if not idxs:
            return
        cur = c["idx"]
        if direction > 0:
            nxt = next((i for i in idxs if i > cur), idxs[-1])
        else:
            nxt = next((i for i in reversed(idxs) if i < cur), idxs[0])
        c["idx"] = int(nxt)
        self._cpr_sync_bar()
        self._cpr_center_map_on_current()        # map pane follows the edit point
        self._refresh()

    def _cpr_jump_ctrl_end(self, to_last: bool) -> None:
        """|◀ / ▶| : jump the cross-section to the FIRST (一番手前 / proximal) or
        LAST (一番奥 / distal) control (edit) point in one click."""
        c = self._cpr
        if c is None:
            return
        idxs = sorted(set(self._cpr_ctrl_indices()))
        if not idxs:
            return
        c["idx"] = int(idxs[-1] if to_last else idxs[0])
        self._cpr_sync_bar()
        self._cpr_center_map_on_current()        # map pane follows the edit point
        self._refresh()

    def cpr_key_nav(self, where: str) -> bool:
        """Shell nav keys while a CPR short-axis is open: A / F step the PREVIOUS /
        NEXT edit point, Shift+A / Shift+F jump to the FIRST / LAST — instead of
        changing the series. *where* ∈ prev/next/first/last. Returns True if claimed
        (i.e. a CPR is open), so the shell skips series navigation."""
        if self._cpr is None:
            return False
        if where == "prev":
            self._cpr_jump_ctrl(-1)
        elif where == "next":
            self._cpr_jump_ctrl(+1)
        elif where == "first":
            self._cpr_jump_ctrl_end(False)
        elif where == "last":
            self._cpr_jump_ctrl_end(True)
        else:
            return False
        return True

    def _cpr_center_map_on_current(self) -> None:
        """Pan the MAP (long-axis) pane so the CURRENT control/scroll point sits at
        the view centre — so stepping edit points moves BOTH panes (short-axis +
        map), making the selected point obvious. Per-viewer camera recentre is done
        by _cpr_recenter_map_pane (VTK / pygfx); no-op if the map pane is unknown."""
        c = self._cpr
        if c is None:
            return
        mapkey = c.get("src")
        if mapkey not in ("A", "B") or mapkey == "A":
            return                                # map is pane B; A = cross-section
        pts = getattr(c.get("cl"), "points", None)
        if pts is None or len(pts) == 0:
            return
        i = int(min(max(int(c["idx"]), 0), len(pts) - 1))
        P = np.asarray(pts[i], float)
        if hasattr(self, "_cpr_recenter_map_pane"):
            self._cpr_recenter_map_pane(mapkey, P)

    def _style_measure_btn(self, on: bool) -> None:
        """Make the Measure toolbar button unmistakable while ON — the interaction
        model changes (view ops then need Alt), so plain blue isn't enough. ON =
        bold white on amber with a border and a label that spells out the Alt
        change; OFF restores the plain look. Shared by both viewers (VTK + pygfx)."""
        b = getattr(self, "_meas_btn", None)
        if b is None:
            return
        if on:
            b.setText(t("📏 計測中（視点操作は Alt）"))
            b.setStyleSheet(
                "QPushButton{background:#ff8c00;color:white;font-weight:bold;"
                "border:2px solid #b35900;border-radius:4px;padding:2px 10px;}")
        else:
            b.setText("📏 Measure")
            b.setStyleSheet(
                "QPushButton:checked{background:#1f77b4;color:white;}")

    def _cpr_set_ctrl(self, m, src, p3) -> None:
        """Write a new control-point list back onto the trace measure (both the
        3-D points and their 2-D projection on the src/map pane)."""
        m["pts3d"] = [list(map(float, np.asarray(q, float))) for q in p3]
        m["pts"] = [self._world3d_to_out(src, np.asarray(q, float)) for q in p3]

    def _cpr_edit_rebuild(self) -> None:
        """Rebuild after an Add/Delete, re-mapping the scroll index by arc-length
        fraction so the view stays at ~the same place along the vessel (P2)."""
        c = self._cpr
        old_n = c["cl"].n
        frac = (c["idx"] / (old_n - 1)) if old_n > 1 else 0.0
        self._cpr_rebuild()
        n2 = self._cpr["cl"].n
        self._cpr["idx"] = int(min(max(round(frac * (n2 - 1)), 0), n2 - 1))
        self._cpr_sync_bar()
        self._refresh()

    def _cpr_add_ctrl_at(self, P) -> None:
        """Insert a new control (edit) point at 3-D world point *P*, keeping the
        proximal→distal order, then rebuild. Used by the cross-section right-click
        Add Point (long-axis edits are made by adding points)."""
        c = self._cpr
        if c is None:
            return
        src, mi = c.get("src"), c.get("src_mi")
        if src not in ("A", "B") or mi is None \
                or not (0 <= mi < len(self._measures.get(src, []))):
            return
        m = self._measures[src][mi]
        p3 = [np.asarray(q, float) for q in (m.get("pts3d") or [])]
        if len(p3) < 2:
            return
        # Insert just after the control points that precede the current section.
        ci = self._cpr_ctrl_indices()
        k = int(min(max(sum(1 for di in ci if di <= c["idx"]), 1), len(p3)))
        p3.insert(k, np.asarray(P, float))
        self._cpr_set_ctrl(m, src, p3)
        self._cpr_rebuild()
        idxs = self._cpr_ctrl_indices()            # land ON the new point (edit)
        if 0 <= k < len(idxs):
            self._cpr["idx"] = int(idxs[k])
            self._cpr_sync_bar()
            self._refresh()

    def _cpr_delete_ctrl_near(self) -> None:
        """Delete the control (edit) point nearest the current cross-section,
        keeping AT LEAST TWO (a centreline needs two), then rebuild."""
        from PyQt6.QtWidgets import QMessageBox
        c = self._cpr
        if c is None:
            return
        src, mi = c.get("src"), c.get("src_mi")
        if src not in ("A", "B") or mi is None \
                or not (0 <= mi < len(self._measures.get(src, []))):
            return
        m = self._measures[src][mi]
        p3 = [np.asarray(q, float) for q in (m.get("pts3d") or [])]
        if len(p3) <= 2:
            QMessageBox.information(
                self, t("Short-axis"),
                t("At least 2 edit points are required "
                  "(cannot delete any more)."))
            return
        ci = self._cpr_ctrl_indices()
        k = int(np.argmin([abs(di - c["idx"]) for di in ci])) if ci else 0
        del p3[k]
        self._cpr_set_ctrl(m, src, p3)
        self._cpr_edit_rebuild()

    # ---- MAP-pane edit points: click → show its section, Shift+drag select,
    #      bulk Delete / Fit / Smooth (shared by both viewers) ----------------
    def _cpr_src_measure(self):
        """The active CPR's source trace measure on the map pane, or None."""
        c = self._cpr
        if c is None:
            return None
        src, mi = c.get("src"), c.get("src_mi")
        if src not in ("A", "B") or mi is None \
                or not (0 <= mi < len(self._measures.get(src, []))):
            return None
        return self._measures[src][mi]

    def _cpr_is_src(self, which, mi) -> bool:
        c = self._cpr
        return (c is not None and which == c.get("src")
                and mi == c.get("src_mi"))

    def _cpr_map_edit_ok(self, which) -> bool:
        """Map-pane bulk editing is live: a CPR is open, *which* is its map
        (long-axis) pane, Measure is ON (handles shown) with no draw type."""
        c = self._cpr
        return (c is not None and which == c.get("src") and which != "A"
                and bool(getattr(self, "_meas_on", False))
                and not getattr(self, "_meas_type", None)
                and getattr(self, "_mode", "3D") == "3D"
                and self._cpr_src_measure() is not None)

    def _cpr_goto_ctrl(self, k) -> None:
        """Show control point *k*'s short-axis section in the left pane (a
        click on that edit point on the map pane — no ◀ ▶ stepping needed).
        The map is NOT panned, so the point stays under the cursor."""
        c = self._cpr
        if c is None:
            return
        idxs = self._cpr_ctrl_indices()
        if not (0 <= k < len(idxs)):
            return
        if c["idx"] != int(idxs[k]):
            c["idx"] = int(idxs[k])
            self._cpr_sync_bar()
            self._refresh()
            try:
                self.cpr_index_changed.emit(self._cpr_disp(c["idx"]))
            except Exception:                            # noqa: BLE001
                pass

    def _cpr_map_point_moved(self, k) -> None:
        """A map-pane drag of edit point *k* finished: rebuild the centreline
        from the edited points and land the short-axis ON that point."""
        if self._cpr is None:
            return
        self._cpr_rebuild()
        self._cpr_goto_ctrl(k)

    def _cpr_map_added_point(self, which, sx, sy) -> None:
        """A map-pane 'Add point' on the CPR line: rebuild, then show the new
        point's section (the edit point nearest the click)."""
        if self._cpr is None:
            return
        self._cpr_rebuild()
        m = self._cpr_src_measure()
        if m is None:
            return
        best, bk = float("inf"), None
        for k, q in enumerate(m.get("pts") or []):
            qx, qy = self._cpr_screen_xy(which, q)
            d = math.hypot(qx - sx, qy - sy)
            if d < best:
                best, bk = d, k
        if bk is not None:
            self._cpr_goto_ctrl(bk)

    def _cpr_screen_xy(self, which, q):
        """Screen px of output-plane point *q* on pane *which* (either viewer)."""
        f = getattr(self, "_world_to_screen", None) or self._world_to_qt
        return f(which, q[0], q[1])

    def _cpr_sel_get(self):
        """Selected map edit-point indices, dropped if the point count changed
        since they were picked (an Add/Delete/rebuild elsewhere)."""
        sel = getattr(self, "_cpr_sel", None) or set()
        m = self._cpr_src_measure()
        n = len(m.get("pts3d") or []) if m is not None else 0
        if not sel or n != getattr(self, "_cpr_sel_n", -1):
            self._cpr_sel = set()
            return set()
        return {k for k in sel if 0 <= k < n}

    def _cpr_sel_set(self, sel) -> None:
        m = self._cpr_src_measure()
        self._cpr_sel = set(sel)
        self._cpr_sel_n = len(m.get("pts3d") or []) if m is not None else 0

    def _cpr_sel_clear(self) -> bool:
        """Clear the selection; True if there was one (caller redraws)."""
        had = bool(getattr(self, "_cpr_sel", None))
        self._cpr_sel = set()
        if had and self._cpr is not None:
            self._redraw_geom(self._cpr.get("src"))
        return had

    def _cpr_band_start(self, which, sx, sy) -> None:
        self._cpr_band = [which, float(sx), float(sy), float(sx), float(sy)]

    def _cpr_band_move(self, sx, sy) -> None:
        b = getattr(self, "_cpr_band", None)
        if b is None:
            return
        b[3], b[4] = float(sx), float(sy)
        self._redraw_geom(b[0])

    def _cpr_band_end(self) -> None:
        """Shift+drag released: select the edit points inside the rectangle
        (replacing the selection). A Shift+CLICK (no real drag) toggles the
        single point under the cursor instead."""
        b = getattr(self, "_cpr_band", None)
        self._cpr_band = None
        if b is None:
            return
        which, x0, y0, x1, y1 = b
        m = self._cpr_src_measure()
        if m is None:
            self._redraw_geom(which)
            return
        pts = m.get("pts") or []
        sel = set(self._cpr_sel_get())
        if abs(x1 - x0) < 4 and abs(y1 - y0) < 4:
            best, bk = 12.0, None
            for k, q in enumerate(pts):
                qx, qy = self._cpr_screen_xy(which, q)
                d = math.hypot(qx - x1, qy - y1)
                if d < best:
                    best, bk = d, k
            if bk is not None:
                sel ^= {bk}
        else:
            lx, hx = min(x0, x1), max(x0, x1)
            ly, hy = min(y0, y1), max(y0, y1)
            sel = set()
            for k, q in enumerate(pts):
                qx, qy = self._cpr_screen_xy(which, q)
                if lx <= qx <= hx and ly <= qy <= hy:
                    sel.add(k)
        self._cpr_sel_set(sel)
        self._redraw_geom(which)

    def _cpr_band_world_rect(self, key):
        """The rubber-band rectangle as a closed output-plane polyline on pane
        *key* (for drawing), or None."""
        b = getattr(self, "_cpr_band", None)
        if b is None or b[0] != key:
            return None
        _, x0, y0, x1, y1 = b
        cs = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        return [tuple(self._disp_to_world(key, x, y)) for x, y in cs]

    def _cpr_sel_world_pts(self, key):
        """Output-plane positions of the selected edit points on pane *key*."""
        c = self._cpr
        if c is None or key != c.get("src"):
            return []
        m = self._cpr_src_measure()
        sel = self._cpr_sel_get()
        if m is None or not sel:
            return []
        pts = m.get("pts") or []
        return [pts[k] for k in sorted(sel) if k < len(pts)]

    @staticmethod
    def _cpr_smooth_pts(p3, sel, iters: int = 3, lam: float = 0.5):
        """Smooth the selected control points: each moves *lam* of the way to
        its neighbours' chord (at its own arc fraction, so points slide toward
        a straighter course rather than along it). Unselected points and the
        two vessel ends stay fixed, so the section blends into its neighbours."""
        P = np.array([np.asarray(q, float) for q in p3])
        n = len(P)
        idx = sorted(k for k in sel if 0 < k < n - 1)
        for _ in range(int(iters)):
            Q = P.copy()
            for k in idx:
                a, b = P[k - 1], P[k + 1]
                d1 = float(np.linalg.norm(P[k] - a))
                d2 = float(np.linalg.norm(b - P[k]))
                f = d1 / (d1 + d2) if (d1 + d2) > 1e-9 else 0.5
                Q[k] = P[k] + lam * ((a + f * (b - a)) - P[k])
            P = Q
        return [P[k] for k in range(n)]

    def _cpr_sel_apply(self, op: str) -> None:
        """Bulk edit of the selected map edit points — 'delete' / 'fit' (snap
        to the contrast-lumen centre) / 'smooth' — as ONE Ctrl+Z step, then
        rebuild the short-axis (display state kept)."""
        from PyQt6.QtWidgets import QMessageBox
        c = self._cpr
        m = self._cpr_src_measure()
        sel = self._cpr_sel_get()
        if c is None or m is None or not sel:
            return
        src = c["src"]
        p3 = [np.asarray(q, float) for q in (m.get("pts3d") or [])]
        n = len(p3)
        before = self._meas_pane_snap(src)
        keep_sel = True
        if op == "delete":
            new = [q for k, q in enumerate(p3) if k not in sel]
            if len(new) < 2:
                QMessageBox.information(
                    self, t("Short-axis"),
                    t("At least 2 edit points are required "
                      "(cannot delete any more)."))
                return
            keep_sel = False
        elif op == "fit":
            new = list(p3)
            for k in sel:
                new[k] = self._cpr_snap_ctrl_to_lumen(
                    p3[k], self._cpr_fit_tangent(p3, k))
        elif op == "smooth":
            new = self._cpr_smooth_pts(p3, sel)
        else:
            return
        self._cpr_set_ctrl(m, src, new)
        self._cpr_edit_rebuild()
        if keep_sel and len(new) == n:
            self._cpr_sel_set(sel)          # keep it: Smooth / Fit can repeat
        else:
            self._cpr_sel = set()
        self._meas_record(src, before)
        self._redraw_meas(src)

    def _cpr_sel_menu(self, which, sx, sy) -> None:
        """Right-click on the map pane while edit points are selected."""
        from PyQt6.QtCore import QPoint
        from PyQt6.QtWidgets import QMenu
        sel = self._cpr_sel_get()
        if not sel:
            return
        menu = QMenu(self)
        a_del = menu.addAction(t("選択した編集点を削除 ({n}点)", n=len(sel)))
        a_fit = menu.addAction(t("選択した編集点を内腔中心へ補正 (Fit)"))
        a_smo = menu.addAction(t("選択区間をスムージング"))
        menu.addSeparator()
        a_clr = menu.addAction(t("選択解除"))
        chosen = menu.exec(self.pane[which].canvas.mapToGlobal(
            QPoint(int(sx), int(sy))))
        if hasattr(self, "_reset_pointer_state"):
            self._reset_pointer_state()
        if chosen is a_del:
            self._cpr_sel_apply("delete")
        elif chosen is a_fit:
            self._cpr_sel_apply("fit")
        elif chosen is a_smo:
            self._cpr_sel_apply("smooth")
        elif chosen is a_clr:
            self._cpr_sel_clear()

    # ---- VR始点: pick the vessel's PROXIMAL start on a VR pane -------------
    # The left pane A shows a heart VR while Draw is armed; a click (no drag)
    # on it picks the vessel point under the cursor as the trace's FIRST point,
    # the map pane B jumps there (Slab MIP through it) and the user clicks the
    # distal points on B as usual. The trace drawn so far is overlaid on the VR
    # so it stays clear which vessel is being followed. Backends provide
    # _cpr_vr_ray(key, sx, sy, dvr) and the VR line drawing.

    #: Same CTA opacity ramp as the VR transfer function (HU, alpha), so the
    #: pick lands on the surface the user actually SEES.
    _CPR_VR_OP_HU = np.array([-1000.0, 80.0, 150.0, 250.0, 400.0, 600.0,
                              1000.0, 2000.0])
    _CPR_VR_OP_A = np.array([0.0, 0.0, 0.03, 0.12, 0.35, 0.60, 0.85, 0.95])
    #: Heart crop radius (mm) around the map-pane centre for the start VR —
    #: keeps the aortic root / ostia / AV grooves, drops most of the chest wall.
    _CPR_VR_CROP_MM = 70.0

    _CPR_VR_PICK = False

    def _cpr_vr_active(self) -> bool:
        return bool(getattr(self, "_cpr_vr_on", False))

    def _cpr_vr_pick_ok(self, key) -> bool:
        """A plain click on VR pane *key* picks the trace start point — OFF:
        VR表示 is for checking the CPR line's position only; the line is set on
        the map pane B (set _CPR_VR_PICK = True to bring the pick back)."""
        return (self._CPR_VR_PICK and self._cpr_vr_active() and key == "A"
                and bool(getattr(self, "_vr_on", {}).get(key))
                and self._cpr is None)

    def _cpr_vr_crop_mask(self):
        """Crop for the start VR. With the Epi model loaded: only the band near
        the Epi boundary (_cpr_vr_epi_band — where the coronaries run), so the
        chest wall / lungs / far chambers don't hide the vessels. Without Epi:
        a sphere of _CPR_VR_CROP_MM around the start-VR centre (the map pane's
        centre when VR始点 was pressed)."""
        vol = getattr(self, "_vol", None)
        if vol is None:
            return None
        band = self._cpr_vr_epi_band()
        if band is not None and band.any():
            return band
        return self._cpr_vr_sphere_mask()

    #: Epi band for the start VR: from _CPR_VR_EPI_IN_MM inside the Epi
    #: boundary to _CPR_VR_EPI_OUT_MM outside it (epicardial fat + coronaries).
    _CPR_VR_EPI_IN_MM = 3.0
    _CPR_VR_EPI_OUT_MM = 8.0

    def _cpr_vr_epi_band(self):
        """Bool [z,y,x] band around the loaded Epi boundary (in/out mm above);
        None when no Epi region / no scipy."""
        f = getattr(self, "_vr_epi_full_mask", None)
        epi = f() if callable(f) else None
        if epi is None:
            return None
        epi = np.asarray(epi, bool)
        zs, ys, xs = np.where(epi)
        if len(zs) == 0:
            return None
        try:
            from scipy.ndimage import distance_transform_edt
        except Exception:                        # noqa: BLE001
            return None
        sx, sy, sz = (float(d) for d in self._dims)
        out_mm = float(self._CPR_VR_EPI_OUT_MM)
        in_mm = float(self._CPR_VR_EPI_IN_MM)
        mz = int(np.ceil(out_mm / sz)) + 2
        my = int(np.ceil(out_mm / sy)) + 2
        mx = int(np.ceil(out_mm / sx)) + 2
        z0, z1 = max(0, zs.min() - mz), min(epi.shape[0], zs.max() + mz + 1)
        y0, y1 = max(0, ys.min() - my), min(epi.shape[1], ys.max() + my + 1)
        x0, x1 = max(0, xs.min() - mx), min(epi.shape[2], xs.max() + mx + 1)
        sub = epi[z0:z1, y0:y1, x0:x1]
        smp = (sz, sy, sx)
        d_out = distance_transform_edt(~sub, sampling=smp)
        d_in = distance_transform_edt(sub, sampling=smp)
        band = ((~sub) & (d_out <= out_mm)) | (sub & (d_in <= in_mm))
        full = np.zeros(epi.shape, bool)
        full[z0:z1, y0:y1, x0:x1] = band
        return full

    def _cpr_vr_sphere_mask(self):
        """Bool [z,y,x] sphere of _CPR_VR_CROP_MM around the start-VR centre."""
        vol = getattr(self, "_vol", None)
        if vol is None:
            return None
        c = getattr(self, "_cpr_vr_center", None)
        if c is None:
            return None
        sx, sy, sz = (float(d) for d in self._dims)
        R = float(self._CPR_VR_CROP_MM)
        nz, ny, nx = vol.shape
        ci = (float(c[2]) / sz, float(c[1]) / sy, float(c[0]) / sx)
        lo = [max(0, int(math.floor(ci[0] - R / sz))),
              max(0, int(math.floor(ci[1] - R / sy))),
              max(0, int(math.floor(ci[2] - R / sx)))]
        hi = [min(nz, int(math.ceil(ci[0] + R / sz)) + 1),
              min(ny, int(math.ceil(ci[1] + R / sy)) + 1),
              min(nx, int(math.ceil(ci[2] + R / sx)) + 1)]
        mask = np.zeros(vol.shape, bool)
        if any(h <= l for l, h in zip(lo, hi)):
            return mask
        zz, yy, xx = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d2 = (((zz * sz) - float(c[2])) ** 2 + ((yy * sy) - float(c[1])) ** 2
              + ((xx * sx) - float(c[0])) ** 2)
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = d2 <= R * R
        return mask

    def _cpr_vr_first_hit(self, O, D, mask=None, shift=0.0, step=0.5):
        """March the ray O + t·D (volume mm) front-to-back through the HU volume
        with the VR opacity ramp (slid by *shift* HU, as the VR W/L does) and
        return the point where the accumulated opacity first reaches 0.25 —
        the visible surface under the cursor — or None if the ray hits nothing.
        Voxels outside *mask* (the VR crop) are transparent."""
        vol = getattr(self, "_vol", None)
        if vol is None:
            return None
        O = np.asarray(O, float)
        D = np.asarray(D, float)
        nd = float(np.linalg.norm(D))
        if nd < 1e-9:
            return None
        D = D / nd
        dims = np.asarray(self._dims, float)
        nz, ny, nx = vol.shape
        hi = np.array([nx - 1, ny - 1, nz - 1], float) * dims
        t0, t1 = 0.0, np.inf
        for i in range(3):                       # clip the ray to the volume box
            if abs(D[i]) < 1e-12:
                if O[i] < 0.0 or O[i] > hi[i]:
                    return None
                continue
            ta = (0.0 - O[i]) / D[i]
            tb = (hi[i] - O[i]) / D[i]
            t0 = max(t0, min(ta, tb))
            t1 = min(t1, max(ta, tb))
        if not np.isfinite(t1) or t1 <= t0:
            return None
        ts = np.arange(t0, t1, float(step))
        if len(ts) == 0:
            return None
        P = O[None, :] + ts[:, None] * D[None, :]
        ix = np.clip(np.rint(P[:, 0] / dims[0]).astype(np.intp), 0, nx - 1)
        iy = np.clip(np.rint(P[:, 1] / dims[1]).astype(np.intp), 0, ny - 1)
        iz = np.clip(np.rint(P[:, 2] / dims[2]).astype(np.intp), 0, nz - 1)
        hu = vol[iz, iy, ix].astype(float)
        if mask is not None:
            m = np.asarray(mask)
            if m.shape == vol.shape:
                hu = np.where(m[iz, iy, ix] > 0, hu, -1000.0)
        a = np.interp(hu, self._CPR_VR_OP_HU + float(shift), self._CPR_VR_OP_A)
        a = np.clip(a * (float(step) / 0.8), 0.0, 1.0)
        acc = 1.0 - np.cumprod(1.0 - a)
        k = int(np.argmax(acc >= 0.25))
        if acc[k] < 0.25:
            return None
        return P[k]

    def _cpr_vr_pick_start(self, key, sx, sy, dvr=None) -> bool:
        """VR click → the vessel point under (sx, sy) becomes the trace start.
        The visible surface hit is snapped along the view ray to the centre of
        the vessel-sized lumen there (a chamber-sized run is ignored, so a click
        on a chamber wall stays on the surface for the user to check)."""
        ray = self._cpr_vr_ray(key, sx, sy, dvr)
        if ray is None:
            return False
        O, D, mask, shift = ray
        P0 = self._cpr_vr_first_hit(O, D, mask=mask, shift=shift)
        if P0 is None:
            return False
        P = self._cpr_vr_snap(P0, D)
        self._cpr_vr_set_start(P)
        return True

    def _cpr_vr_snap(self, P0, D, max_lumen_mm=6.0, floor_hu=150.0):
        """Visible-surface hit *P0* → the vessel's lumen centre. First along the
        view ray *D*: the bright run just behind the surface, if it is VESSEL-
        sized (≤ *max_lumen_mm* deep — a chamber / aorta is deeper, then *P0*
        is kept so a chamber click is not dragged inside). Then a local 3-D
        HU-weighted centroid (r 2.5 mm) pulls a click on the vessel's flank
        onto its axis. (Not the MPR _snap_to_lumen: its in-plane cavity guard
        reads a vessel lying ACROSS the view as a cavity.)"""
        P0 = np.asarray(P0, float)
        D = np.asarray(D, float)
        D = D / max(1e-9, float(np.linalg.norm(D)))
        step = 0.25
        ds = np.arange(-1.0, 10.0 + step, step)
        hu = self._hu_along(P0, D, ds)
        if hu is None:
            return P0
        peak = float(hu.max())
        if peak < floor_hu:
            return P0
        thr = max(floor_hu, 0.5 * peak)
        bright = hu >= thr
        i0 = int(np.argmax(ds >= 0.0))
        idx = np.where(bright[i0:])[0]
        if len(idx) == 0:
            return P0
        i = i0 + int(idx[0])
        while i > 0 and bright[i - 1]:
            i -= 1
        j = i
        while j < len(ds) and bright[j]:
            j += 1
        if float(ds[j - 1] - ds[i]) > max_lumen_mm or j >= len(ds):
            return P0                             # chamber-deep → keep the surface
        w = hu[i:j] - thr + 1e-3
        P = P0 + float(np.dot(w, ds[i:j]) / float(w.sum())) * D
        return self._cpr_vr_centroid(P, thr)

    def _cpr_vr_centroid(self, P, thr, r_mm=2.5, iters=2):
        """HU-weighted centroid of the ≥*thr* voxels within *r_mm* of *P*
        (twice) — re-centres a point on a tube's axis. Skipped where the
        neighbourhood is mostly bright (a cavity, not a vessel)."""
        vol = getattr(self, "_vol", None)
        if vol is None:
            return np.asarray(P, float)
        dims = np.asarray(self._dims, float)
        nz, ny, nx = vol.shape
        P = np.asarray(P, float)
        for _ in range(int(iters)):
            c = P / dims
            rv = np.ceil(r_mm / dims).astype(int)
            x0, x1 = max(0, int(c[0]) - rv[0]), min(nx, int(c[0]) + rv[0] + 2)
            y0, y1 = max(0, int(c[1]) - rv[1]), min(ny, int(c[1]) + rv[1] + 2)
            z0, z1 = max(0, int(c[2]) - rv[2]), min(nz, int(c[2]) + rv[2] + 2)
            if x1 <= x0 or y1 <= y0 or z1 <= z0:
                break
            sub = vol[z0:z1, y0:y1, x0:x1].astype(float)
            zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
            X = xx * dims[0]
            Y = yy * dims[1]
            Z = zz * dims[2]
            inside = ((X - P[0]) ** 2 + (Y - P[1]) ** 2 + (Z - P[2]) ** 2
                      <= r_mm * r_mm)
            br = inside & (sub >= thr)
            n_in = int(inside.sum())
            if n_in == 0 or not br.any() or br.sum() > 0.75 * n_in:
                break
            w = (sub - thr + 1.0)[br]
            P = np.array([float((X[br] * w).sum() / w.sum()),
                          float((Y[br] * w).sum() / w.sum()),
                          float((Z[br] * w).sum() / w.sum())])
        return P

    def _cpr_vr_set_start(self, P) -> None:
        """Make *P* the first point of the trace on the map pane B: B's plane is
        centred on it (Slab MIP through the start) and the draft polyline starts
        there. A trace already in progress keeps its later points and only gets
        its start replaced (re-pick the start without losing the distal clicks)."""
        P = np.asarray(P, float)
        self._cpr_vr_start = P.copy()
        self._cpr_recenter_map_pane("B", P)
        d = getattr(self, "_draft", None)
        if (d is not None and d.get("pane") == "B"
                and d.get("type") == "polyline" and d.get("pts3d")
                and len(d["pts3d"]) == len(d.get("pts", [])) >= 2):
            d["pts3d"][0] = P
        else:
            d = {"type": "polyline", "pane": "B", "pts": [(0.0, 0.0)],
                 "pts3d": [P]}
            self._draft = d
            self._draft_redo = []
        d["pts"] = [self._world3d_to_out("B", Q) for Q in d["pts3d"]]
        self._refresh(reset_cam=False)
        self._redraw_geom("B")
        self._cpr_vr_line_update()
        if hasattr(self, "_set_active"):
            self._set_active("B")
        if hasattr(self, "_flash_active_frame"):
            self._flash_active_frame()

    def _cpr_vr_line_pts(self):
        """(points, start) to overlay on the start VR: the trace being drawn on
        B (3-D clicks) and the picked start point; ([], None) when off."""
        if not self._cpr_vr_active():
            return [], None
        d = getattr(self, "_draft", None)
        pts = []
        if (d is not None and d.get("pane") == "B"
                and d.get("type") == "polyline" and d.get("pts3d")):
            pts = [np.asarray(q, float) for q in d["pts3d"]]
        start = getattr(self, "_cpr_vr_start", None)
        if pts:
            start = pts[0]
        return pts, start

    def _cpr_vr_toggle(self) -> None:
        """VR始点 button: show (or hide) the heart VR on the left pane for
        picking the vessel's proximal start. Arms Draw if needed."""
        if self._cpr_vr_active():
            self._cpr_vr_stop()
            return
        if getattr(self, "_vol", None) is None or self._mode != "3D":
            self._cpr_vr_sync_btn()
            return
        if not getattr(self, "_coronary_mpr_pending", False):
            self._coronary_draw()                # arms Measure + Polyline
        if not getattr(self, "_coronary_mpr_pending", False):
            self._cpr_vr_sync_btn()
            return
        self._cpr_vr_on = True
        self._cpr_vr_start = None
        # Centre the VR on the centreline crossing (as seen on the map pane B),
        # fresh every time: drop the last VR camera pose / pan of pane A.
        ctr = getattr(self, "_center", None)
        if ctr is None or not np.all(np.isfinite(np.asarray(ctr, float))):
            ctr = self._pc["B"]
        self._cpr_vr_center = np.asarray(ctr, float).copy()
        getattr(self, "_vr_view_saved", {}).pop("A", None)
        pan = getattr(self, "_vr_pan", None)
        if isinstance(pan, dict) and "A" in pan:
            pan["A"] = type(pan["A"])((0.0, 0.0)) if isinstance(
                pan["A"], (list, tuple)) else (0.0, 0.0)
        if hasattr(self, "set_side"):
            self.set_side("Bi")
        try:
            self._vr_set("A", True)              # crop: _cpr_vr_crop_mask
        except Exception:                        # noqa: BLE001
            self._cpr_vr_on = False
            raise
        finally:
            self._cpr_vr_sync_btn()
        self._cpr_vr_line_update()

    def _cpr_vr_stop(self) -> None:
        """Leave the start VR (CPR built / Draw disarmed / Exit / button)."""
        if not self._cpr_vr_active():
            return
        self._cpr_vr_on = False
        self._cpr_vr_start = None
        self._cpr_vr_line_update()               # clears the VR overlay line
        if getattr(self, "_vr_on", {}).get("A"):
            self._vr_set("A", False)
        self._cpr_vr_sync_btn()

    def _cpr_vr_sync_btn(self) -> None:
        b = getattr(self, "_cpr_vr_btn", None)
        if b is None:
            return
        b.blockSignals(True)
        b.setChecked(self._cpr_vr_active())
        b.blockSignals(False)

    def _cpr_apply_xform(self):
        """Rebuild the CPR display axes u, v from the base frame (u0, v0), the
        cumulative Rt90/Flip transform T, and the continuous rotation ``rot``
        (applied last, in-plane about the tangent) — the IVUS-style rotation
        used by the CoSync 按分."""
        c = self._cpr
        T = c["T"]
        bu = T[0, 0] * c["u0"] + T[0, 1] * c["v0"]
        bv = T[1, 0] * c["u0"] + T[1, 1] * c["v0"]
        th = math.radians(c.get("rot", 0.0))
        ct, st = math.cos(th), math.sin(th)
        c["u"] = ct * bu + st * bv
        c["v"] = -st * bu + ct * bv

    # ---- numpy volume sampling (live-independent; used by CoSync export) ----
    def _sample_vol_grid(self, P):
        """Trilinear-sample the HU volume at world points *P* (…,3 array).
        Out-of-volume samples read -1000 HU (air). Shared by the CoSync stack
        renderer."""
        sx, sy, sz = self._dims
        fx = P[..., 0] / sx
        fy = P[..., 1] / sy
        fz = P[..., 2] / sz
        nz, ny, nx = self._vol.shape
        inb = ((fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
               & (fz >= 0) & (fz <= nz - 1))
        x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
        z0 = np.clip(np.floor(fz).astype(int), 0, nz - 2)
        tx, ty, tz = fx - x0, fy - y0, fz - z0
        V = self._vol
        out = np.zeros(P.shape[:-1], np.float32)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = ((tx if dx else 1 - tx) * (ty if dy else 1 - ty)
                         * (tz if dz else 1 - tz))
                    out += w * V[z0 + dz, y0 + dy, x0 + dx]
        out[~inb] = -1000.0
        return out

    def cpr_cosync_spec(self, px: int = 96):
        """Render the short-axis stack as a synthetic 'pull-back' for CoSync.

        Returns a dict the shell hands to the CoSync window so the Stretch-MPR
        joins the multi-pane landmark grid exactly like an IVUS pull-back:
          frames  : (n, px, px) float32 HU cross-sections along the vessel
          window/level, spacing_mm (per display pixel), start (current index),
          rotation deg and label.
        None when the short-axis isn't active."""
        c = self._cpr
        if c is None or self._vol is None:
            return None
        half = float(c["half"])
        n = int(c["cl"].n)
        # Match the CT pane's on-screen orientation: u runs left->right (columns
        # -half->+half); v runs bottom->top in the VTK pane, so as image ROWS
        # (top-down in a QImage) it must go +half->-half — otherwise the CoSync
        # image comes out flipped vs the pane.
        gs_u = np.linspace(-half, half, px)
        gs_v = np.linspace(half, -half, px)
        gu, gv = np.meshgrid(gs_u, gs_v)
        # Bake the Rt90/Flip (T) orientation into the stack, but NOT the
        # continuous rotation — that is carried as the CoSync free-rotation
        # (below) so the rotation 按分 can drive it and it isn't applied twice.
        T = c["T"]
        bu = T[0, 0] * c["u0"] + T[0, 1] * c["v0"]
        bv = T[1, 0] * c["u0"] + T[1, 1] * c["v0"]
        frames = np.empty((n, px, px), np.float32)
        for i in range(n):
            o = np.asarray(c["cl"].points[i], float)
            u = bu[i]
            vv = bv[i]
            P = (o[None, None, :] + gu[..., None] * u[None, None, :]
                 + gv[..., None] * vv[None, None, :])
            frames[i] = self._sample_vol_grid(P)
        if c.get("reversed"):                     # distal->proximal, like IVUS
            frames = frames[::-1].copy()
        return {
            "kind": "ct_cpr",
            "frames": frames,
            "window": float(self._win),
            "level": float(self._lvl),
            "spacing_mm": (2.0 * half) / max(1, px - 1),
            "start": self._cpr_disp(c["idx"]),
            "rotation": float(c.get("rot", 0.0)),
            "label": t("CT short-axis"),
        }

    # ==================================================================
    # Control logic — scrub / rotate / reverse / paging / centreline rebuild.
    # These drive the shared state, then call two backend hooks the concrete
    # viewer provides: ``_refresh()`` (redraw both panes) and ``_cpr_sync_bar()``
    # (update the Qt scrubber slider + label). Both viewers already implement
    # those, so the whole control surface is shared.
    # ==================================================================
    def _cpr_set_index(self, d):
        """Scroll to DISPLAY position *d* (from the scrubber); maps to the real
        centreline index via the reverse flag."""
        c = self._cpr
        if c is None:
            return
        d = int(min(max(int(d), 0), c["cl"].n - 1))
        idx = self._cpr_disp(d)                   # display -> real index
        changed = (idx != c["idx"])
        c["idx"] = idx
        self._cpr_sync_bar()
        self._refresh()
        if changed:
            self.cpr_index_changed.emit(d)        # CoSync: broadcast display pos

    def set_cpr_index(self, d: int, *, silent: bool = False) -> None:
        """CoSync driver: move to DISPLAY position *d* (mapped via the reverse
        flag) without echoing the signal back (silent) to avoid feedback."""
        c = self._cpr
        if c is None:
            return
        d = int(min(max(int(d), 0), c["cl"].n - 1))
        idx = self._cpr_disp(d)
        if idx == c["idx"]:
            return
        c["idx"] = idx
        self._cpr_sync_bar()
        self._refresh()
        if not silent:
            self.cpr_index_changed.emit(d)

    def set_cpr_rotation(self, deg: float, *, silent: bool = False) -> None:
        """Rotate the short-axis cross-section in-plane to *deg* (the CoSync
        rotation 按分 drives this); rebuilds the display frame."""
        c = self._cpr
        if c is None:
            return
        deg = float(deg)
        if abs(deg - c.get("rot", 0.0)) < 1e-6:
            return
        c["rot"] = deg
        self._cpr_apply_xform()
        self._refresh()
        if not silent:
            self.cpr_rotation_changed.emit(deg)

    def _cpr_toggle_reverse(self):
        """Reverse the short-axis scroll direction (distal->proximal, to match
        an IVUS pull-back). Only the traversal order flips — each cross-section's
        content is unchanged."""
        if self._cpr is None:
            return
        self._cpr["reversed"] = self._cpr_rev_btn.isChecked()
        self._cpr_sync_bar()
        # broadcast the new display position so a linked CoSync stays in step
        self.cpr_index_changed.emit(self._cpr_disp(self._cpr["idx"]))

    def _cpr_rebuild(self):
        """Recompute the centreline / short-axis frames from the (edited)
        control points, keeping the current scroll index."""
        c = self._cpr
        p3 = self._cpr_ctrl_pts3d()
        if not p3 or len(p3) < 2:
            return
        cl = CenterLine.from_points([np.asarray(P, float) for P in p3],
                                    step_mm=max(1e-3, min(self._dims)))
        if cl.n < 2:
            return
        fu, fv = cl.frames(ref_up=c["ref_up"])
        fu = -fu                                  # view first->last (un-mirror)
        c["cl"], c["u0"], c["v0"] = cl, fu, fv
        c["idx"] = min(c["idx"], cl.n - 1)
        c.pop("_ctrl_idx", None)                  # control-point indices changed
        self._cpr_apply_xform()                   # re-apply Rt90/Flip -> u, v
        self._cpr_sync_bar()
        self._refresh()

    # ---- marker-drag release (grab / move stay per-backend: coordinate + actor
    #      coupled) ----
    def _cpr_drag_end(self):
        """Release: rebuild the centreline from the adjusted control points, then
        re-snap the view onto the dragged edit point (its dense index shifts a
        little as arc-length changes) so it stays FILLED / editable."""
        if self._cpr_drag is None:
            return
        ci = self._cpr_drag
        self._cpr_drag = None
        self._cpr_rebuild()
        idxs = self._cpr_ctrl_indices()
        if 0 <= ci < len(idxs):
            self._cpr["idx"] = int(idxs[ci])
            self._cpr_sync_bar()
            self._refresh()

    # ---- manual short-axis rotation (drag the section like a dial) ----
    def _cpr_cursor_angle(self, sx, sy) -> float:
        c = self.pane["A"].canvas
        return math.atan2(sy - c.height() / 2.0, sx - c.width() / 2.0)

    def _cpr_rot_start(self, sx, sy):
        self._cpr_rot_prev = self._cpr_cursor_angle(sx, sy)

    def _cpr_rot_move(self, sx, sy):
        if self._cpr is None or self._cpr_rot_prev is None:
            return
        ang = self._cpr_cursor_angle(sx, sy)
        d = math.degrees(ang - self._cpr_rot_prev)
        self._cpr_rot_prev = ang
        self.set_cpr_rotation(self._cpr.get("rot", 0.0) + d)

    def _cpr_rot_end(self):
        self._cpr_rot_prev = None

    def _cpr_page_drag(self, dy):
        """Paging tool on the short-axis: drag up = advance the pull-back
        (~6 px per cross-section), like the 2-D paging drag."""
        if self._cpr is None:
            return
        self._cpr_page_accum = getattr(self, "_cpr_page_accum", 0.0) - dy
        step = 6.0
        d = self._cpr_disp(self._cpr["idx"])     # current display position
        while self._cpr_page_accum >= step:
            self._cpr_page_accum -= step
            d += 1
        while self._cpr_page_accum <= -step:
            self._cpr_page_accum += step
            d -= 1
        self._cpr_set_index(d)
