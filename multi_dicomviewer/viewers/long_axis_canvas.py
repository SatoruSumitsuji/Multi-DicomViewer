"""IVUS longitudinal (long-axis) cut canvas.

The IVUS pull-back is treated as a 3-D stack: frame index = pull-back
distance, the (row, col) of each frame is the cross-section. A long-
axis cut is one column of that stack per frame, sampled along a line
through the frame's rotation centre at angle ``θ``. With all centres
at the image centre and θ = 0, the cut is the classic horizontal slice
of every frame stitched side-by-side.

The widget:

* paints a uint8 (H_lateral, N_frames) image scaled to fit;
* draws a yellow vertical line at the current frame index;
* draws a horizontal yellow line where the rotation axis projects (the
  middle row by construction — every centre lies on that line);
* **plain horizontal drag** stretches / compresses the strip's WIDTH only
  (each frame column gets wider or narrower). The height is NOT scaled —
  it always fills the display frame, so no vertical scrollbar appears;
* **plain vertical drag** rotates the cut around the catheter axis (no
  modifier key). The owning axis is locked on the first move, so a drag is
  purely rotate OR purely width-zoom, never both;
* a left click WITHOUT drag jumps to the clicked frame.

The actual long-axis pixel array is built by ``build_long_axis`` —
kept module-level so a (rare) headless test can call it without
constructing a Qt widget.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor, QImage, QPainter, QPen, QPixmap, QPolygon,
)
from PyQt6.QtWidgets import QMenu, QWidget

from multi_dicomviewer.core.image_export import pick_export_format


def build_long_axis(
    frames: Sequence[np.ndarray],
    centers: Sequence[tuple[float, float]],
    angle_rad: float,
    lateral_samples: int,
    apply_u8: callable,
    step: float = 1.0,
) -> np.ndarray:
    """Render one (lateral_samples, n_frames) uint8 image.

    *frames* is a sequence of (H, W) or (H, W, 3) ndarrays, one per
    pull-back position (caller decides whether to pull from cache or
    block-decode). *centers* is parallel — (cx, cy) in image pixels for
    each frame. *apply_u8* converts the raw sampled pixels (whatever
    dtype the frames have) to uint8 for display — typically a closure
    around the viewer's W/L LUT so the strip honours the user's
    current window/level instantly. Out-of-image samples are 0.

    *step* is the spacing (in source pixels) between the lateral samples.
    Default 1.0 samples every pixel across the cut. A larger step covers the
    SAME physical extent with fewer samples (``lateral_samples``) — used for
    a coarse, fast low-LOD preview that still spans the full vessel depth.
    """
    n = len(frames)
    half = (lateral_samples - 1) / 2.0
    ux, uy = math.cos(angle_rad), math.sin(angle_rad)
    # Row 0 (strip TOP) samples the +(ux,uy) end of the cut line, the last
    # row (strip BOTTOM) the −(ux,uy) end — so the strip's top lines up with
    # the cross-section's cyan projection triangle (at +(ux,uy)) and the
    # cyan-topped / red-bottomed long-axis cursor.
    r = (half - np.arange(lateral_samples, dtype=np.float32)) * float(step)
    dxs = r * ux
    dys = r * uy

    # Fast path: when every frame is present and the stack is a single
    # contiguous (n, H, W[, 3]) ndarray (the common case — `frames` is the
    # plane's volume, see IVUSViewer._frames_for_long_axis), gather all
    # columns at once instead of one Python iteration per frame. The per-frame
    # loop over a long colour pull-back was the bulk of the UI-thread freeze on
    # a long-axis rotate/rebuild; this vectorised gather is the same maths and
    # is bit-identical to the loop below (see tools/test_long_axis.py).
    vol = frames if isinstance(frames, np.ndarray) else None
    if vol is not None and vol.ndim in (3, 4) and n > 0:
        is_color = vol.ndim == 4
        h, w = vol.shape[1], vol.shape[2]
        cx = np.asarray([c[0] for c in centers], dtype=np.float64)
        cy = np.asarray([c[1] for c in centers], dtype=np.float64)
        # (n, lateral) integer sample coordinates.
        xs = np.rint(cx[:, None] + dxs[None, :]).astype(np.int64)
        ys = np.rint(cy[:, None] + dys[None, :]).astype(np.int64)
        mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        fi = np.arange(n)[:, None]                      # (n, 1) frame index
        gathered = vol[fi, np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)]
        gathered[~mask] = 0                             # out-of-image -> 0
        # (n, lateral[, 3]) -> (lateral, n[, 3]); apply_u8 is element-wise so
        # running it once over the whole strip == once per column.
        sampled = (gathered.transpose(1, 0, 2) if is_color
                   else gathered.T)
        return np.ascontiguousarray(apply_u8(sampled), dtype=np.uint8)

    # Fallback: a list that may contain None / mixed shapes (e.g. a partly
    # decoded pull-back) — sample frame by frame.
    is_color = any(f is not None and f.ndim == 3 for f in frames)
    out = np.zeros(
        (lateral_samples, n, 3) if is_color else (lateral_samples, n),
        dtype=np.uint8,
    )
    if n == 0:
        return out
    for i in range(n):
        f = frames[i]
        if f is None:
            continue
        h, w = f.shape[:2]
        cx, cy = centers[i]
        xs = np.rint(cx + dxs).astype(np.int32)
        ys = np.rint(cy + dys).astype(np.int32)
        mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if is_color:
            # Sample RGB. A stray grayscale frame in a colour strip is
            # broadcast to all three channels so the assignment still fits.
            col = np.zeros((lateral_samples, 3), dtype=f.dtype)
            if mask.any():
                if f.ndim == 3:
                    col[mask] = f[ys[mask], xs[mask], :3]
                else:
                    col[mask] = f[ys[mask], xs[mask]][:, None]
            out[:, i] = apply_u8(col)
        else:
            col = np.zeros(lateral_samples, dtype=f.dtype)
            if mask.any():
                col[mask] = f[ys[mask], xs[mask]]
            out[:, i] = apply_u8(col)
    return out


class LongAxisCanvas(QWidget):
    """Widget showing the long-axis cut + a frame cursor + an axis
    indicator. Emits rotation deltas while the user drags, and a frame
    jump on a simple click."""

    rotated = pyqtSignal(float)        # incremental ΔΘ in radians
    #: emitted on mouse-release after a rotation drag, so the host can
    #: rebuild the strip at full resolution (rotation drags preview at a
    #: reduced LOD for speed).
    rotation_finished = pyqtSignal()
    frame_picked = pyqtSignal(int)     # frame index (0-based)
    #: Right-click on a keyframe (manual-centre) ▼/▲ marker → carries the
    #: scope ('frame' = remove just that frame's manual centre, 'all' =
    #: remove every manual centre) and the clicked keyframe's frame index.
    keyframe_remove = pyqtSignal(str, int)
    #: Emitted whenever a new long-axis image is pushed in; the host
    #: scroll area listens so it can re-size the canvas to keep the
    #: source aspect ratio at the current viewport height.
    image_changed = pyqtSignal()
    #: Horizontal-zoom drag: emits (multiplier, anchor_content_x). The
    #: host scroll area applies the multiplier to ``_h_zoom`` via
    #: ``multiply_h_zoom``, re-sizes the canvas, and adjusts the
    #: horizontal scrollbar so ``anchor_content_x`` (in source-pixel
    #: coords, i.e. frame index in the long axis) stays under the drag.
    h_zoom_changed = pyqtSignal(float, float)
    #: Right-click export: carries the chosen format key ('png'/'jpeg'/
    #: 'tiff'); the host (IVUSViewer) captures this canvas and saves it.
    export_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.setMouseTracking(True)
        # Black background; the long-axis image is grayscale.
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor("#000"))
        self.setPalette(pal)

        self._qimg: Optional[QImage] = None
        #: Cached down-scaled pixmap of the strip at the current widget size.
        #: A long pull-back's strip QImage is thousands of columns wide; re-
        #: scaling it on every frame-cursor move (set_current_frame → update)
        #: stutters/freezes seek + playback on slower GPUs (Mac). We scale
        #: ONCE per (image, size) and just blit this pixmap on cursor moves;
        #: only the thin cursor/axis overlay is redrawn each frame.
        self._scaled_pix: Optional[QPixmap] = None
        self._scaled_key: Optional[tuple] = None
        #: Top-left widget point at which ``_scaled_pix`` (the visible-band
        #: pixmap) should be blitted.
        self._scaled_origin: tuple = (0, 0)
        self._n_frames = 0
        self._cur_frame = 0
        #: Frame indices at which the user has set a manual rotation
        #: centre (keyframes). Drawn as small cyan downward-triangle
        #: markers at the top of the strip; a click near one snaps to
        #: that frame via the existing _x_to_frame mapping (no special
        #: hit-test needed since each marker sits on its frame's x).
        self._keyframes: list[int] = []
        self._press_pos: Optional[QPoint] = None
        self._dragged = False
        self._press_was_left = False
        #: Which axis "owns" the current drag, locked on first move past the
        #: dead-band: "rot" (vertical → rotate) or "zoom" (horizontal →
        #: width stretch). None between drags.
        self._drag_axis: Optional[str] = None
        #: Horizontal stretch factor — multiplied into the natural
        #: aspect-preserving width. >1 makes each frame column wider
        #: (zooms in); <1 narrower. Reset to 1.0 on a new series load.
        self._h_zoom: float = 1.0
        #: Drag state (resize-invariant: tracked in screen-global coords
        #: so a mid-drag canvas resize doesn't bend the trajectory).
        self._press_global: Optional[QPoint] = None
        self._last_global: Optional[QPoint] = None
        self._press_anchor_content_x: float = 0.0

    # ----------------------------------------------------------- public
    def set_image(self, arr: np.ndarray) -> None:
        """Set the long-axis image and repaint. Accepts (H_lat, N_frames)
        uint8 grayscale, or (H_lat, N_frames, 3) uint8 RGB for a colour
        IVUS strip (the cut-line colour sampled as-is)."""
        arr = np.ascontiguousarray(arr)
        if arr.dtype != np.uint8:
            return
        if arr.ndim == 2:
            h, w = arr.shape
            self._n_frames = w
            self._qimg = QImage(
                arr.data, w, h, w, QImage.Format.Format_Grayscale8
            ).copy()
        elif arr.ndim == 3 and arr.shape[2] == 3:
            h, w = arr.shape[:2]
            self._n_frames = w
            self._qimg = QImage(
                arr.data, w, h, 3 * w, QImage.Format.Format_RGB888
            ).copy()
        else:
            return
        self._scaled_pix = None          # new image → rescale on next paint
        self._scaled_key = None
        self.image_changed.emit()
        self.update()

    def clear(self) -> None:
        self._qimg = None
        self._scaled_pix = None
        self._scaled_key = None
        self._n_frames = 0
        self._cur_frame = 0
        self._keyframes = []
        self.image_changed.emit()
        self.update()

    def preferred_width_for_height(self, h: int) -> int:
        """Native width the canvas wants at viewport height *h*. At
        zoom = 1 this is the source aspect at *h* tall (each frame is
        a column); the ``_h_zoom`` factor multiplies it uniformly with
        :py:meth:`preferred_height_for_viewport_height` so the canvas
        scales as a whole and the initial aspect ratio survives any
        left/right zoom drag. Horizontal scrollbar appears when the
        result exceeds the viewport width."""
        h = max(60, int(h))
        if self._qimg is None or self._qimg.height() <= 0:
            return max(60, h)
        natural = self._qimg.width() * h / self._qimg.height()
        return max(60, int(round(natural * self._h_zoom)))

    def preferred_height_for_viewport_height(self, h: int) -> int:
        """Canvas height for viewport height *h*. The long-axis strip's
        height ALWAYS fills the display frame — the drag-zoom stretches
        only the width (``preferred_width_for_height``), never the
        height — so this just returns the viewport height and no vertical
        scrollbar ever appears."""
        return max(60, int(h))

    def source_width(self) -> int:
        """Width of the underlying long-axis image in source pixels
        (= number of frames). 0 when no image is loaded."""
        return self._qimg.width() if self._qimg is not None else 0

    def multiply_h_zoom(self, factor: float) -> None:
        """Multiply ``_h_zoom`` by *factor* and clamp into a wide range.
        Upper limit is intentionally loose so each frame column can be
        zoomed many times the canvas height — useful when inspecting a
        single landmark across two or three frames. Lower limit keeps
        the strip from collapsing to a hairline."""
        new_zoom = float(self._h_zoom) * float(factor)
        self._h_zoom = max(0.05, min(2000.0, new_zoom))

    def reset_h_zoom(self) -> None:
        """Restore the natural horizontal scale. Called by IVUSViewer
        on series change so each new pull-back starts unstretched."""
        self._h_zoom = 1.0

    def set_current_frame(self, idx: int) -> None:
        idx = max(0, int(idx))
        if idx != self._cur_frame:
            self._cur_frame = idx
            self.update()

    def set_keyframes(self, frame_indices) -> None:
        """Mark which frames have a manual rotation centre. Drawn as
        small cyan triangle markers at the top edge of the strip so
        the user can see at a glance where centres are set and use
        the Center ◀/▶ buttons (or click near a marker) to jump."""
        clean = sorted({int(i) for i in (frame_indices or [])})
        if clean != self._keyframes:
            self._keyframes = clean
            self.update()

    def _ensure_scaled(self, full_r: QRect, vis: QRect) -> None:
        """(Re)build the cached strip pixmap for the VISIBLE band *vis* of the
        full canvas rect *full_r*. Only the columns that intersect *vis* are
        scaled — at high horizontal zoom the full strip is tens of thousands of
        px wide, and scaling all of it to a pixmap on every paint (the previous
        behaviour) allocated a huge image and froze the viewer; a frame-cursor
        move or a zoom drag now costs only the viewport width. Stored with its
        blit origin in ``_scaled_origin``. FastTransformation matches the old
        nearest scaling, so the look is identical."""
        if self._qimg is None or vis.width() <= 0 or vis.height() <= 0:
            self._scaled_pix = None
            self._scaled_key = None
            return
        src_w = self._qimg.width()
        src_h = self._qimg.height()
        fw = max(1, full_r.width())
        # Source columns covering the visible band (clamped to the image).
        sx0 = int(math.floor((vis.x() - full_r.x()) / fw * src_w))
        sx1 = int(math.ceil((vis.x() + vis.width() - full_r.x()) / fw * src_w))
        sx0 = max(0, min(sx0, src_w - 1))
        sx1 = max(sx0 + 1, min(sx1, src_w))
        key = (self._qimg.cacheKey(), sx0, sx1, vis.width(), vis.height())
        self._scaled_origin = (vis.x(), vis.y())
        if self._scaled_pix is not None and self._scaled_key == key:
            return
        sub = self._qimg.copy(sx0, 0, sx1 - sx0, src_h)
        scaled = sub.scaled(
            int(vis.width()), int(vis.height()),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._scaled_pix = QPixmap.fromImage(scaled)
        self._scaled_key = key

    # ----------------------------------------------------- coord transforms
    def _draw_rect(self) -> QRect:
        if self._qimg is None or self._n_frames == 0:
            return QRect()
        # Fill the canvas widget exactly. The host scroll area sizes
        # this canvas via preferred_width_for_height(), which already
        # bakes in the user's horizontal-zoom factor — so an aspect-
        # preserving min(sx, sy) "fit" here would just pillarbox the
        # strip and the h-drag zoom would have no visible effect (the
        # widget would grow but the image stayed at natural aspect).
        # Filling lets the per-frame columns stretch with _h_zoom as
        # designed; vertical drag continues to rotate, not resize.
        return QRect(0, 0, max(1, self.width()), max(1, self.height()))

    def _x_to_frame(self, x: float) -> int:
        r = self._draw_rect()
        if not r.isValid() or self._n_frames == 0:
            return 0
        f = (x - r.x()) / max(r.width(), 1) * self._n_frames
        return int(max(0, min(self._n_frames - 1, f)))

    def _keyframe_tri_size(self, r: QRect) -> tuple[int, int]:
        """(tri_h, tri_w) for the keyframe ▼/▲ markers at draw-rect *r* —
        0.8× the original size, aspect ratio kept. Shared by paint and the
        click hit-test so the clickable area always matches what's drawn."""
        base_h = max(8, min(14, r.height() // 8))
        base_w = max(5, base_h - 2)
        return max(3, int(round(base_h * 0.8))), max(2, int(round(base_w * 0.8)))

    def _keyframe_at_widget(self, x: float, y: float) -> Optional[int]:
        """Keyframe index whose marker is under (x, y) widget px, or None.
        The grab area is a ±tri_w column (the triangle width) spanning the
        FULL strip height — so a click anywhere near the vertical line (not
        only on the top/bottom triangles) counts. When two adjacent markers'
        columns overlap, the one whose line is nearest the click wins."""
        if not self._keyframes or self._n_frames == 0:
            return None
        r = self._draw_rect()
        if not r.isValid():
            return None
        _tri_h, tri_w = self._keyframe_tri_size(r)
        top_y, bot_y = r.y(), r.y() + r.height()
        if not (top_y - 2 <= y <= bot_y + 2):
            return None
        best, best_dx = None, None
        for f in self._keyframes:
            if not (0 <= f < self._n_frames):
                continue
            mx = r.x() + int((f + 0.5) / self._n_frames * r.width())
            dx = abs(x - mx)
            if dx <= tri_w and (best_dx is None or dx < best_dx):
                best, best_dx = f, dx
        return best

    def _keyframe_menu(self, frame: int, global_pos: QPoint) -> None:
        """Remove this point / Remove all points for the keyframe ▼/▲ at
        *frame* — mirrors the cross-section red-centre menu but targets the
        clicked keyframe explicitly (it need not be the current frame)."""
        menu = QMenu(self)
        a_one = menu.addAction("Remove this point")
        a_all = menu.addAction("Remove all points")
        chosen = menu.exec(global_pos)
        if chosen is a_one:
            self.keyframe_remove.emit("frame", int(frame))
        elif chosen is a_all:
            self.keyframe_remove.emit("all", int(frame))

    # ----------------------------------------------------- mouse
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            if self._qimg is None:
                return
            # Right-click ON a keyframe ▼/▲ (within the triangle width) →
            # the same Remove this point / Remove all points menu as the
            # cross-section's red centre marker, acting on THAT keyframe.
            pos = e.position().toPoint()
            kf = self._keyframe_at_widget(pos.x(), pos.y())
            if kf is not None:
                self._keyframe_menu(kf, e.globalPosition().toPoint())
                return
            # Otherwise the strip offers a still export.
            key = pick_export_format(self, e.globalPosition().toPoint())
            if key:
                self.export_requested.emit(key)
            return
        if e.button() == Qt.MouseButton.LeftButton:
            self._press_pos = e.position().toPoint()
            self._press_global = e.globalPosition().toPoint()
            self._last_global = self._press_global
            # Anchor (in source pixels / frame-index space) under the
            # press point — stays valid through canvas resizes so the
            # horizontal zoom keeps that frame under the cursor.
            self._press_anchor_content_x = self._canvas_x_to_content_x(
                float(self._press_pos.x())
            )
            self._dragged = False
            self._drag_axis = None
            self._press_was_left = True

    def _canvas_x_to_content_x(self, canvas_x: float) -> float:
        """Map a widget-x to a source-pixel x (= frame index, fractional).
        Returns 0 when there is no image."""
        if self._qimg is None:
            return 0.0
        src_w = self._qimg.width()
        canvas_w = max(1, self.width())
        return float(canvas_x) / canvas_w * src_w

    def mouseMoveEvent(self, e):
        if self._press_global is None or not self._press_was_left:
            return
        p_global = e.globalPosition().toPoint()
        # Cumulative drag (from press) decides which axis "owns" this
        # gesture and is also the dead-band for click-vs-drag.
        total_dx = p_global.x() - self._press_global.x()
        total_dy = p_global.y() - self._press_global.y()
        # 6 px dead-band: small hand jitter on a click should NOT be
        # treated as a drag (which would suppress the frame-jump on
        # release).
        if abs(total_dx) + abs(total_dy) < 6:
            return
        d_x = p_global.x() - self._last_global.x()
        d_y = p_global.y() - self._last_global.y()
        self._last_global = p_global
        self._dragged = True
        # Lock the gesture to one axis on the first move past the dead-band so
        # it doesn't flip between rotate/zoom mid-drag. Vertical-dominant →
        # rotate; horizontal-dominant → width stretch. No modifier needed.
        if self._drag_axis is None:
            self._drag_axis = "rot" if abs(total_dy) >= abs(total_dx) else "zoom"
        if self._drag_axis == "rot":
            # Plain vertical drag → rotate around the catheter axis.
            # 1 px of y-drag = ~0.25° (full sweep ≈ 150° per ~600 px).
            self.rotated.emit(math.radians(0.25 * d_y))
        else:
            # Plain horizontal drag → stretch / compress the WIDTH only
            # (the height always fills the display frame). Exponential
            # mapping: 100 px right ≈ 1.65×, 100 px left ≈ 0.61×. The
            # press-time frame stays under the cursor.
            factor = math.exp(0.005 * d_x)
            self.h_zoom_changed.emit(factor, self._press_anchor_content_x)

    def mouseReleaseEvent(self, e):
        if (e.button() == Qt.MouseButton.LeftButton
                and self._press_was_left
                and not self._dragged):
            # Pure click → frame jump. A click landing ON a keyframe ▼/▲
            # snaps to that exact manual-centre frame (precise even when
            # neighbours are close); anywhere else seeks to the clicked x.
            pt = e.position().toPoint()
            kf = self._keyframe_at_widget(pt.x(), pt.y())
            self.frame_picked.emit(
                kf if kf is not None else self._x_to_frame(pt.x())
            )
        elif (e.button() == Qt.MouseButton.LeftButton
                and self._drag_axis == "rot" and self._dragged):
            # End of a rotation drag → ask the host to rebuild at full res
            # (the drag itself previewed at a reduced LOD).
            self.rotation_finished.emit()
        self._press_pos = None
        self._press_global = None
        self._last_global = None
        self._press_was_left = False
        self._dragged = False
        self._drag_axis = None

    # ----------------------------------------------------- paint
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0a0a0a"))
        if self._qimg is None or self._n_frames == 0:
            p.setPen(QColor("#888"))
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Long-axis: no data yet (load an IVUS series)"
            )
            return
        r = self._draw_rect()
        # Only scale the columns Qt is actually asking us to repaint (the
        # exposed viewport band), spanning the FULL strip height. At high
        # horizontal zoom r is tens of thousands of px wide; scaling the whole
        # of it per paint froze the viewer. The thin overlays below still draw
        # against the full rect r (Qt clips them to the exposed region).
        vis = r.intersected(e.rect())
        vis = QRect(vis.x(), r.y(), max(1, vis.width()), r.height())
        self._ensure_scaled(r, vis)
        if self._scaled_pix is not None:
            p.drawPixmap(self._scaled_origin[0], self._scaled_origin[1],
                         self._scaled_pix)
        # Horizontal axis (where all rotation centres project — by
        # construction the strip's vertical mid-row).
        p.setPen(QPen(QColor(255, 220, 80, 110), 1, Qt.PenStyle.DashLine))
        mid_y = r.y() + r.height() // 2
        p.drawLine(r.x(), mid_y, r.x() + r.width(), mid_y)
        # Keyframe markers — drawn UNDER the current-frame cursor so
        # the yellow cursor stays dominant when both overlap. Red ▼
        # at the top edge and ▲ at the bottom edge, plus a thin red
        # tick line between them, so each keyed frame is unmistakable
        # against dark IVUS. Red matches the "fixed = red" colour
        # convention used by the cross-section centre marker.
        if self._keyframes and self._n_frames > 0:
            # Manual-centre (keyframe) markers — YELLOW ▼/▲ at the top and
            # bottom edges with a faint connecting tick. Yellow keeps them
            # distinct from the current-frame cursor's cyan/red arrows (which
            # previously shared the red). Triangles are 0.8× the old size.
            # Clicking ON a triangle jumps to that exact keyframe (see
            # _keyframe_at_widget) so dense neighbours don't interfere.
            tri_h, tri_w = self._keyframe_tri_size(r)
            top_y = r.y()
            bot_y = r.y() + r.height()
            for f in self._keyframes:
                if not (0 <= f < self._n_frames):
                    continue
                mx = r.x() + int(
                    (f + 0.5) / self._n_frames * r.width()
                )
                # The marker for the CURRENT frame is drawn fully opaque so
                # "you are on a manual-centre frame" stands out at a glance;
                # every other keyframe marker is faded to 50% alpha.
                fa = 1.0 if f == self._cur_frame else 0.5
                tick = QColor(255, 220, 0, int(130 * fa))
                fill = QColor(255, 220, 0, int(235 * fa))
                outline = QColor(0, 0, 0, int(220 * fa))
                # Thin vertical tick connecting the two triangles —
                # easy to spot even on a very wide zoomed-in strip.
                p.setPen(QPen(tick, 1))
                p.drawLine(mx, top_y + tri_h, mx, bot_y - tri_h)
                # Top ▼ (apex pointing down into the strip)
                p.setPen(QPen(outline, 1))
                p.setBrush(fill)
                p.drawConvexPolygon(QPolygon([
                    QPoint(mx, top_y + tri_h),
                    QPoint(mx - tri_w, top_y),
                    QPoint(mx + tri_w, top_y),
                ]))
                # Bottom ▲ (apex pointing up into the strip)
                p.drawConvexPolygon(QPolygon([
                    QPoint(mx, bot_y - tri_h),
                    QPoint(mx - tri_w, bot_y),
                    QPoint(mx + tri_w, bot_y),
                ]))
            p.setBrush(Qt.BrushStyle.NoBrush)
        # Current frame cursor — the vertical bar standing for the cross-
        # section shown above. Top half CYAN, bottom half RED to match the
        # strip's up/down. Each half carries a LEFT-pointing arrowhead on the
        # RIGHT side of the line (apex touching the line): cyan at the top
        # end, red at the bottom end — so up/down maps to the cyan/red
        # projection triangles on the cross-section. Antialias the heads.
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx = r.x() + int(
            (self._cur_frame + 0.5) / max(self._n_frames, 1) * r.width()
        )
        top_y = r.y()
        bot_y = r.y() + r.height()
        mid_y = r.y() + r.height() // 2
        cyan = QColor(0, 160, 210, 235)
        red = QColor(215, 45, 45, 235)
        p.setPen(QPen(cyan, 2))
        p.drawLine(cx, top_y, cx, mid_y)
        p.setPen(QPen(red, 2))
        p.drawLine(cx, mid_y, cx, bot_y)
        # Left-pointing arrowheads, apex on the line, body to the right.
        aw, ah = 11, 7
        p.setPen(Qt.PenStyle.NoPen)
        cy_y = top_y + ah + 1          # cyan ◄ hugging the top end
        p.setBrush(cyan)
        p.drawConvexPolygon(QPolygon([
            QPoint(cx, cy_y),
            QPoint(cx + aw, cy_y - ah),
            QPoint(cx + aw, cy_y + ah),
        ]))
        rd_y = bot_y - ah - 1          # red ◄ hugging the bottom end
        p.setBrush(red)
        p.drawConvexPolygon(QPolygon([
            QPoint(cx, rd_y),
            QPoint(cx + aw, rd_y - ah),
            QPoint(cx + aw, rd_y + ah),
        ]))
        p.setBrush(Qt.BrushStyle.NoBrush)
        # Small label.
        p.setPen(QColor("#cfe8ff"))
        p.drawText(
            r.x() + 6, r.y() + 16,
            f"Long-axis · frame {self._cur_frame + 1}/{self._n_frames}"
        )
