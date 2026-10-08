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

    def _cpr_register_to_tree(self) -> None:
        """「保存登録」button: SAVE the current short-axis to a .cpr.json first
        (the user names the file), THEN register that CPR into the Coronary Tree —
        routed to its source CT by series UID, with the file name as the vessel
        name (保存後登録 in one click). Cancelling the save cancels the registration.
        Shared by both viewers (VTK + pygfx)."""
        import os
        from PyQt6.QtWidgets import QMessageBox
        path = self._cpr_save()                  # Save As prompt → path, or None
        if not path:                             # cancelled / nothing to save
            return
        data = self._cpr_build_data()            # includes series / src_dir / ctrl
        if data is None:
            return
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
            return
        try:
            shell.coronary_register_cpr(data, name or "vessel", path)
        except Exception as exc:                 # noqa: BLE001
            QMessageBox.warning(self.window(), t("Coronary Tree"),
                                t("登録に失敗しました: {e}", e=str(exc)))

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
