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

    def _cpr_frame(self):
        """(origin, u, v, tangent) of the current cross-section."""
        c = self._cpr
        i = c["idx"]
        return (np.asarray(c["cl"].points[i], float), c["u"][i], c["v"][i],
                np.asarray(c["cl"].tangents[i], float))

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
        self._cpr_apply_xform()                   # re-apply Rt90/Flip -> u, v
        self._cpr_sync_bar()
        self._refresh()

    # ---- marker-drag release (grab / move stay per-backend: coordinate + actor
    #      coupled) ----
    def _cpr_drag_end(self):
        """Release: rebuild the centreline from the adjusted control points."""
        if self._cpr_drag is None:
            return
        self._cpr_drag = None
        self._cpr_rebuild()

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
