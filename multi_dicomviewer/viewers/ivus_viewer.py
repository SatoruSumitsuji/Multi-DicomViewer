"""IVUS pull-back viewer.

IVUS is a single-plane multi-frame grayscale cine — exactly what the XA
viewer already does — so this reuses XAViewer's machinery (lazy frame
decode + background prefetch, cine transport, W/L, measure tools, DICOM
tag overlay, and the First/Prev/Next/Last cross-series transport) and
adds the IVUS-specific long-axis (longitudinal) view: a horizontal
strip below the cross-section, built by stacking one line per frame
through that frame's rotation centre at the user-selected angle.

handles_modality = "IVUS" lets the shell tell an IVUS pane apart from
an XA one even though they share most of the implementation; the shell
navigates within whichever modality the active cine pane shows.
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSplitter,
)

from multi_dicomviewer.core.dicom_io import apply_color_mode_to_planes
from multi_dicomviewer.core.image_export import export_image_as, safe_basename
from multi_dicomviewer.core.settings import load_ivus_color, save_ivus_color
from multi_dicomviewer.viewers.long_axis_canvas import (
    LongAxisCanvas, build_long_axis,
)
from multi_dicomviewer.viewers.xa_viewer import (
    XAViewer, _Prefetcher, apply_window,
)


class _LongAxisScroll(QScrollArea):
    """Wraps a LongAxisCanvas so its size follows the source aspect at
    the current viewport height. Horizontal drag-zoom scales BOTH
    canvas dimensions by ``_h_zoom`` (initial aspect preserved), so:

    * a horizontal scrollbar appears when the canvas width exceeds the
      viewport (long pull-back, or zoomed in);
    * a vertical scrollbar appears when zoom > 1 makes the canvas
      taller than the viewport — after each zoom we pin the rotation-
      axis row (the strip's vertical midline) at the viewport's
      vertical centre, so the anatomically-meaningful row stays in
      view as the user zooms in or out.

    When the canvas is smaller than the viewport in either direction
    AlignCenter keeps it visually centred ("長軸画面の中央ぞろえ").

    Horizontal-zoom drags on the canvas are forwarded here via
    ``h_zoom_changed(multiplier, anchor_content_x)`` so the horizontal
    scroll position can be adjusted to keep the user's press-frame
    anchored under the cursor through the resize."""

    def __init__(self, canvas: LongAxisCanvas, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self.setWidget(canvas)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Vertical scrollbar appears once zoom > 1 grows the canvas
        # past the splitter height; below that the canvas fits
        # vertically and AlignCenter keeps it centred.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        canvas.image_changed.connect(self._resize_canvas)
        canvas.h_zoom_changed.connect(self._on_h_zoom)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._resize_canvas()

    def _resize_canvas(self):
        vp_h = max(60, self.viewport().height())
        w = self._canvas.preferred_width_for_height(vp_h)
        h = self._canvas.preferred_height_for_viewport_height(vp_h)
        self._canvas.resize(w, h)

    def _on_h_zoom(self, factor: float, anchor_content_x: float) -> None:
        """Apply the per-event horizontal-zoom delta from a drag,
        re-size the canvas (BOTH dimensions, so the initial aspect
        ratio is preserved through zoom), then:

        * move the horizontal scrollbar so the press-time frame
          (anchor_content_x, in source-pixel coords) stays under the
          cursor through the resize;
        * move the vertical scrollbar so the strip's mid-row (the
          rotation-axis line — the visually anchored row) stays at
          the viewport's vertical centre.
        """
        src_w = self._canvas.source_width()
        if src_w <= 0:
            return
        # 1. Where, in widget-x BEFORE the resize, was the horizontal anchor?
        cw_before = max(1, self._canvas.width())
        anchor_cx_before = anchor_content_x / src_w * cw_before
        scroll_before = self.horizontalScrollBar().value()
        # When the canvas is narrower than the viewport AlignCenter
        # leaves blank gutters on both sides; account for them so the
        # anchor's viewport-x reflects what the user actually sees.
        vp_w = self.viewport().width()
        gutter_before = max(0, (vp_w - cw_before) // 2)
        viewport_anchor_x = (anchor_cx_before + gutter_before) - scroll_before

        # 2. Apply the zoom delta and re-fit the canvas to the new
        #    preferred size at the same viewport height. Width AND
        #    height both scale with _h_zoom so the strip's aspect
        #    stays constant.
        self._canvas.multiply_h_zoom(factor)
        self._resize_canvas()

        # 3. Adjust the horizontal scrollbar so the anchor lands on
        #    the same viewport-x as it did before the resize.
        cw_after = max(1, self._canvas.width())
        anchor_cx_after = anchor_content_x / src_w * cw_after
        gutter_after = max(0, vp_w - cw_after) // 2
        new_scroll = int(round(
            (anchor_cx_after + gutter_after) - viewport_anchor_x
        ))
        sb = self.horizontalScrollBar()
        sb.setValue(max(sb.minimum(), min(sb.maximum(), new_scroll)))

        # 4. Vertical: keep the canvas centred — at zoom <= 1 it fits
        #    the viewport and AlignCenter handles it; at zoom > 1 we
        #    nudge the vertical scrollbar so the strip's mid-row (the
        #    rotation-axis line) sits at the viewport's mid-line.
        vp_h = self.viewport().height()
        ch_after = max(1, self._canvas.height())
        v_sb = self.verticalScrollBar()
        if ch_after > vp_h:
            v_target = max(0, (ch_after - vp_h) // 2)
            v_sb.setValue(
                max(v_sb.minimum(), min(v_sb.maximum(), v_target))
            )


class IVUSViewer(XAViewer):
    handles_modality = "IVUS"

    def __init__(self, parent=None):
        super().__init__(parent)
        # IVUS enables the %PlaqueArea readout on each canvas: when two
        # Ellipse/Polygon measures exist they are auto-paired; with 3+
        # the user picks 2 via the outline right-click menu.
        for c in (self.canvas, self.canvas2):
            c.is_ivus = True

        # Long-axis state ----------------------------------------------
        # Per-plane rotation-centre array, lazily populated on series
        # load. Each entry is a (n_frames, 2) float array of (cx, cy)
        # image-pixel coordinates; the default for every frame is the
        # image centre (set in load_series).
        self._la_centers: list[np.ndarray] = []
        # Parallel boolean array per plane — True for frames the user
        # explicitly dragged a centre on ("keyframes"). Non-keyed frame
        # centres are linearly interpolated between adjacent keys (with
        # a clamp to the nearest key beyond the ends). With no keys at
        # all, every frame falls back to the image centre and the long-
        # axis matches the original "straight catheter" view.
        self._la_center_keyed: list[np.ndarray] = []
        #: Rotation angle of the long-axis cut, radians; 0 = sample
        #: along the +X (column) axis of each frame.
        self._la_angle: float = 0.0
        #: Toggled by V (or _ivus_toggle_long_axis from MainWindow).
        self._la_visible: bool = False
        #: Built lazily — kept so frame changes only update the cursor,
        #: not the full strip.
        self._la_img: np.ndarray | None = None
        #: Decoded frames cached for the duration of a rotation drag (the
        #: angle changes per move, the frames don't), so each preview only
        #: re-composites instead of re-decoding. Set on the drag's first
        #: build, cleared by the full-resolution rebuild on release.
        self._la_drag_frames: list | None = None

        self.long_axis = LongAxisCanvas()
        # Long-axis lives inside a horizontally-scrollable area so its
        # width can grow with the frame count (1 px per frame at the
        # current splitter height) while the viewport stays bounded.
        self._la_scroll = _LongAxisScroll(self.long_axis)

        # Vertical splitter: cross-section above, long-axis strip below.
        # Dragging the handle gives the long-axis as much vertical room
        # as the user wants; collapsed start so the strip is invisible
        # until V toggles it on.
        layout = self.layout()
        idx = layout.indexOf(self._canvas_area)
        layout.removeWidget(self._canvas_area)

        self._la_splitter = QSplitter(Qt.Orientation.Vertical)
        self._la_splitter.setChildrenCollapsible(False)
        # Wider handle + hover tint matches the QMainWindow dock
        # separators so every drag handle in the app feels the same.
        self._la_splitter.setHandleWidth(10)
        self._la_splitter.setStyleSheet(
            "QSplitter::handle { background:#a8a8a8; }"
            "QSplitter::handle:hover { background:#4a90d9; }"
        )
        self._la_splitter.addWidget(self._canvas_area)
        self._la_splitter.addWidget(self._la_scroll)
        # Initial sizes: cross-section keeps most of the height; the
        # long-axis area gets a sensible default of ~180 px (drag the
        # handle up to grow). Hidden by default — toggle_long_axis()
        # shows the strip.
        self._la_scroll.hide()
        self._la_splitter.setStretchFactor(0, 5)
        self._la_splitter.setStretchFactor(1, 2)
        self._la_splitter.setSizes([600, 180])

        layout.insertWidget(idx, self._la_splitter, 1)

        self.long_axis.rotated.connect(self._on_la_rotated)
        self.long_axis.rotation_finished.connect(self._on_la_rotation_finished)
        self.long_axis.frame_picked.connect(self._on_la_frame_picked)
        self.long_axis.export_requested.connect(self._on_export_long_axis)
        # Right-click on a long-axis keyframe ▼/▲ → remove that manual centre.
        self.long_axis.keyframe_remove.connect(self._on_la_keyframe_remove)
        # Hook the cross-section canvases for the rotation-centre marker.
        for c in (self.canvas, self.canvas2):
            c.ivus_center_changed.connect(self._on_center_dragged)
            c.ivus_center_reset.connect(self._on_center_reset)
            # Dragging the yellow cut line on the cross-section re-angles
            # the long-axis plane (live draft, then full rebuild on release).
            c.ivus_angle_changed.connect(self._on_la_angle_set)
            c.ivus_angle_finished.connect(self._on_la_rotation_finished)

        # The inherited (XA) series-nav row already runs Series…Flip-H/Flip-V
        # and would overflow with the IVUS controls added on the end. So the
        # IVUS-specific cluster (Long View + the centre-keyframe buttons) goes
        # on a SECOND toolbar row, inserted right below the first one, so the
        # two rows never overlap on a narrow IVUS pane.
        ivus_row = QHBoxLayout()
        ivus_row.setContentsMargins(4, 0, 4, 0)

        # "Long View" toggle. Mirrors the V shortcut so the check state always
        # matches _la_visible.
        self._long_view_btn = QPushButton("Long View")
        self._long_view_btn.setCheckable(True)
        self._long_view_btn.setMinimumWidth(110)
        self._long_view_btn.setStyleSheet(
            "QPushButton { font-weight: bold; }"
            "QPushButton:checked { background:#1f77b4; color:white; }"
        )
        self._long_view_btn.setToolTip(
            "Show/hide the IVUS long-axis (longitudinal) view — shortcut: V"
        )
        self._long_view_btn.clicked.connect(self.toggle_long_axis)
        ivus_row.addWidget(self._long_view_btn)

        # Centre-keyframe controls — right of Long View so the whole "long-axis"
        # cluster reads left-to-right. Prev/Next cycle through the keyed frames
        # on the active plane (wraps); Clear removes every manual centre.
        self._prev_key_btn = QPushButton("◀ Center")
        self._prev_key_btn.setToolTip(
            "Jump to the previous frame with a manual rotation centre"
        )
        self._prev_key_btn.clicked.connect(
            lambda: self._jump_to_keyframe(-1)
        )
        ivus_row.addWidget(self._prev_key_btn)
        self._next_key_btn = QPushButton("Center ▶")
        self._next_key_btn.setToolTip(
            "Jump to the next frame with a manual rotation centre"
        )
        self._next_key_btn.clicked.connect(
            lambda: self._jump_to_keyframe(+1)
        )
        ivus_row.addWidget(self._next_key_btn)
        self._clear_centers_btn = QPushButton("Clear Centers")
        self._clear_centers_btn.setToolTip(
            "Remove every manual rotation centre on the active plane "
            "(same as right-click ▸ Reset all on the marker)"
        )
        self._clear_centers_btn.clicked.connect(self._clear_all_centers)
        ivus_row.addWidget(self._clear_centers_btn)
        ivus_row.addStretch(1)

        # "カラー表示" toggle — IVUS decodes grayscale by default (the chroma
        # of a YBR-stored grayscale IVUS is just compression noise, and showing
        # it tints the image). The rare genuinely-colour IVUS (NIRS chemogram,
        # VH tissue map) is opted into per-series here: click to recover colour,
        # click again to return to grayscale. Shown on every IVUS for a
        # consistent toolbar; on a truly monochrome series the click is a no-op
        # and reports "no colour" (see _on_color_toggle). The choice persists
        # per SeriesInstanceUID (core.settings.save_ivus_color).
        self._color_btn = QPushButton("カラー表示")
        self._color_btn.setCheckable(True)
        self._color_btn.setToolTip(
            "IVUS をカラー表示に切り替える(NIRS ケモグラム等の色情報を復元)。\n"
            "通常の IVUS はグレイのまま。もう一度押すとグレイに戻ります。"
        )
        self._color_btn.setStyleSheet(
            "QPushButton:checked { background:#c0392b; color:white; }"
        )
        self._color_btn.clicked.connect(self._on_color_toggle)
        ivus_row.addWidget(self._color_btn)
        # Insert as the second item of the viewer's main column (index 1),
        # directly under the inherited series-nav row (index 0).
        self.layout().insertLayout(1, ivus_row)
        # Buttons start disabled — they enable once a series is loaded
        # with at least one keyframe (see _refresh_keyframe_markers).
        for b in (self._prev_key_btn, self._next_key_btn,
                  self._clear_centers_btn):
            b.setEnabled(False)

    # ============================================================ public
    def toggle_long_axis(self) -> None:
        """V shortcut / Long View button entry point. Shows/hides the
        strip and the per-frame rotation-centre marker on the cross-
        section canvas. Keeps the Long View button's check state in
        sync with the visibility so V and the button never disagree."""
        self._la_visible = not self._la_visible
        self._la_scroll.setVisible(self._la_visible)
        if hasattr(self, "_long_view_btn"):
            self._long_view_btn.setChecked(self._la_visible)
        for c in (self.canvas, self.canvas2):
            c.ivus_show_center = self._la_visible
        if self._la_visible:
            self._refresh_center_marker()
            self._rebuild_long_axis()
        else:
            self.long_axis.clear()
            for c in (self.canvas, self.canvas2):
                c.update()

    # =========================================================== plumbing
    def load_series(self, loaded, title: str) -> None:
        super().load_series(loaded, title)
        # Build per-plane center arrays — default = image centre, which
        # is the catheter centre on a calibrated IVUS frame.
        self._la_centers = []
        self._la_center_keyed = []
        for plane in self._planes:
            n = plane.total_frames
            f0 = plane.volume[0]
            h, w = f0.shape[:2]
            arr = np.tile(
                np.array([w / 2.0, h / 2.0], dtype=np.float32),
                (n, 1),
            )
            self._la_centers.append(arr)
            self._la_center_keyed.append(
                np.zeros(n, dtype=bool)
            )
        # Reset the long-axis angle and horizontal zoom when switching
        # series — different acquisitions have no shared anatomical
        # "horizontal" and the previous zoom is unlikely to fit.
        self._la_angle = 0.0
        self._la_img = None
        # New series → drop the long-axis rotation-drag frame cache.
        self._la_drag_frames = None
        self.long_axis.reset_h_zoom()
        # New series → no keyframes yet; clear markers & disable jump
        # buttons until the user keys at least one centre.
        self._refresh_keyframe_markers()
        if self._la_visible:
            self._refresh_center_marker()
            self._rebuild_long_axis()
        # Restore this series' remembered colour choice (default grayscale).
        self._sync_color_toggle()

    # ======================================================= colour display
    def _sync_color_toggle(self) -> None:
        """Set the カラー表示 button to this series' remembered choice and
        apply it. super().load_series has just (re)built every plane in the
        default grayscale mode, so we only need to flip to colour when the
        user previously chose it for this SeriesInstanceUID."""
        uid = getattr(self, "_loaded_uid", "")
        want = load_ivus_color(uid) if uid else False
        achieved = self._set_color_mode(True) if want else False
        self._color_btn.blockSignals(True)
        self._color_btn.setChecked(bool(achieved))
        self._color_btn.blockSignals(False)

    def _on_color_toggle(self) -> None:
        """カラー表示 button click. Switch the whole series between grayscale
        and colour, persist the choice, and — if the user asked for colour on
        a series that has none — revert and say so."""
        want = self._color_btn.isChecked()
        achieved = self._set_color_mode(want)
        if want and not achieved:
            self._color_btn.blockSignals(True)
            self._color_btn.setChecked(False)
            self._color_btn.blockSignals(False)
            self.readout.setText(
                "このIVUSにカラー情報はありません(グレイ表示のまま)。"
            )
            return
        uid = getattr(self, "_loaded_uid", "")
        if uid:
            save_ivus_color(uid, achieved)
        self.readout.setText(
            "カラー表示に切り替えました。" if achieved
            else "グレイスケール表示に戻しました。"
        )

    def _set_color_mode(self, color: bool) -> bool:
        """Re-decode every plane grayscale↔colour and refresh the viewer to
        match. Returns the achieved colour flag (False if no colour exists to
        show). Stops cine + prefetch first because the volume arrays are
        replaced, then restarts the prefetch in the new mode."""
        if not self._planes:
            return False
        self.stop()
        # FULLY stop the prefetch before apply_color_mode_to_planes swaps each
        # plane's volume/_ready arrays. The shared _stop_prefetch only waits
        # 80 ms; a slow colour decode still in flight would then keep writing
        # plane.volume[i] while the array is replaced underneath it — a shape
        # race (gray 2-D frame into a colour 3-D slot) that throws in the
        # prefetch thread, which PyQt6 turns into a hard abort. Wait unbounded
        # so no decode is mid-flight during the swap.
        if self._prefetch is not None:
            self._prefetch.stop()
            self._prefetch.wait()
            self._prefetch = None
        achieved = apply_color_mode_to_planes(self._planes, color)
        self._is_color = achieved
        # The planes were re-decoded in the new mode — drop the long-axis
        # rotation-drag frame cache so it rebuilds against the new pixels.
        self._la_drag_frames = None

        # Mirror load_series' W/L-slider handling for the new mode: colour has
        # no window/level, grayscale re-derives the span from frame 0.
        for s in (self.win_slider, self.lvl_slider):
            s.blockSignals(True)
        if achieved:
            self.win_slider.setEnabled(False)
            self.lvl_slider.setEnabled(False)
        else:
            self.win_slider.setEnabled(True)
            self.lvl_slider.setEnabled(True)
            stacked = np.concatenate(
                [p.volume[0].ravel() for p in self._planes]
            )
            vmin, vmax = float(stacked.min()), float(stacked.max())
            span = max(vmax - vmin, 1.0)
            self._window, self._level = span, (vmax + vmin) / 2.0
            # New grayscale baseline → make it the W/L-popup Reset target too.
            self._window_init, self._level_init = self._window, self._level
            self.win_slider.setRange(1, int(span * 2))
            self.win_slider.setValue(int(self._window))
            self.lvl_slider.setRange(int(vmin - span), int(vmax + span))
            self.lvl_slider.setValue(int(self._level))
        for s in (self.win_slider, self.lvl_slider):
            s.blockSignals(False)
        self._refresh_wl_lut()

        # Reflect the new tone in the title bar (… | color | … vs grayscale).
        self._update_color_title()

        # Warm the rest of the cine in the new mode, then repaint.
        self._prefetch = _Prefetcher(
            self._planes, lambda: self._timer.isActive(), self
        )
        self._prefetch.start()
        self._render()
        if self._la_visible:
            self._rebuild_long_axis()
        return achieved

    def _update_color_title(self) -> None:
        """Swap the grayscale/color token in the (already-built) title bar so
        it tracks a runtime colour toggle without rebuilding the whole string."""
        t = self.title_label.text()
        new_tone = "color" if self._is_color else "grayscale"
        for old in ("grayscale", "color"):
            marker = f"|   {old}   |"
            if marker in t:
                self.title_label.setText(t.replace(marker, f"|   {new_tone}   |"))
                return

    def _render(self):
        super()._render()
        # Cursor on the long-axis follows the cine; the strip itself
        # only rebuilds on rotation / centre / W-L changes.
        if self._la_visible:
            self.long_axis.set_current_frame(self._frame)
            self._refresh_center_marker()

    def _wl_changed(self):
        super()._wl_changed()
        if self._la_visible:
            self._rebuild_long_axis()

    # ======================================================== long-axis
    def _active_plane_idx(self) -> int:
        return min(self._active, max(0, len(self._planes) - 1))

    def _refresh_center_marker(self) -> None:
        """Sync the rotation-centre marker shown on the cross-section
        canvas with the current frame's stored centre — both position
        (``ivus_center_image``) and ``ivus_center_keyed`` (red when
        this frame is keyed/fixed, blue when interpolated/movable)."""
        if not self._planes or not self._la_centers:
            return
        pi = self._active_plane_idx()
        if pi >= len(self._la_centers):
            return
        centers = self._la_centers[pi]
        if self._frame >= len(centers):
            return
        cx, cy = float(centers[self._frame, 0]), float(centers[self._frame, 1])
        keyed = bool(self._la_center_keyed[pi][self._frame])
        for c in (self.canvas, self.canvas2):
            c.ivus_center_image = (cx, cy)
            c.ivus_center_keyed = keyed
            c.ivus_la_angle = self._la_angle
            c.update()

    def _frames_for_long_axis(
        self, plane, progress=None
    ) -> list[np.ndarray]:
        """All frames of *plane*. Forces a decode of any not-yet-ready
        frame (the prefetch usually has them; for a long pull-back this
        may briefly block the UI). Returns None placeholders for any
        frame that fails to decode. *progress*, if given, is called as
        ``progress(done, total)`` after every frame so the caller can
        drive a QProgressDialog."""
        out: list[np.ndarray | None] = []
        n = plane.total_frames
        for i in range(n):
            try:
                out.append(plane.frame(i))
            except Exception:
                out.append(None)
            if progress is not None:
                progress(i + 1, n)
        return out

    #: Resolution divisor for the live-rotation preview (both axes), so the
    #: composite is ~LOD² cheaper while a drag is in progress. The full-res
    #: strip is rebuilt on mouse release.
    _LA_DRAFT_LOD = 3

    def _rebuild_long_axis(self, draft: bool = False) -> None:
        if not (self._la_visible and self._planes and self._la_centers):
            return
        # Re-entrancy guard: the slow path pumps the event loop
        # (processEvents) while decoding, which can deliver an event that
        # re-triggers a rebuild — re-entering with a half-built strip /
        # dialog. Coalesce to the in-flight build instead.
        if getattr(self, "_la_building", False):
            return
        self._la_building = True
        try:
            self._rebuild_long_axis_impl(draft)
        finally:
            self._la_building = False

    def _rebuild_long_axis_impl(self, draft: bool = False) -> None:
        pi = self._active_plane_idx()
        plane = self._planes[pi]
        centers = self._la_centers[pi]
        f0 = plane.volume[0]
        h, w = f0.shape[:2]
        # Lateral samples = the larger of the two cross-section
        # dimensions so the cut goes fully edge-to-edge in any rotation.
        lateral = int(round(math.sqrt(h * h + w * w)))
        # Closure that maps a sampled line (raw dtype) to uint8 honouring
        # the current W/L — same path the cross-section uses, so the
        # strip's brightness/contrast tracks the W/L sliders live.
        wl_lut, wl_off = self._wl_lut, self._wl_off
        window, level = self._window, self._level

        def to_u8(col: np.ndarray) -> np.ndarray:
            if col.dtype == np.uint8:
                return col
            if (wl_lut is not None
                    and np.issubdtype(col.dtype, np.integer)):
                return wl_lut[col + wl_off] if wl_off else wl_lut[col]
            return apply_window(col, window, level)

        center_pairs = [(float(c[0]), float(c[1])) for c in centers]

        # Fast path: live rotation preview. The frames were decoded on the
        # drag's first build and cached, so we only re-composite (at reduced
        # LOD) — no decode, no dialog.
        if draft and self._la_drag_frames is not None:
            self._set_la_image(self._la_drag_frames, center_pairs, lateral,
                               to_u8, draft=True)
            return

        # Slow path: force every frame to decode (mostly cache hits once the
        # prefetch finished; several seconds on a fresh pull-back), then
        # composite. minimumDuration=400 ms keeps W/L tweaks and the first
        # rotation build (cache hits, well under that) from flashing a dialog.
        n = plane.total_frames
        dlg = QProgressDialog(
            "Decoding frames for long-axis view…",
            None, 0, n, self,
        )
        dlg.setWindowTitle("Building Long View")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(400)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)

        def _cb(done: int, total: int) -> None:
            if dlg.maximum() != total:
                dlg.setMaximum(total)
            dlg.setValue(done)
            # Only pump the event loop every few frames so the cache-hit
            # fast path (a few ms per "decode") doesn't spend most of
            # its time in processEvents() overhead.
            if done % 8 == 0 or done == total:
                QApplication.processEvents()

        try:
            frames = self._frames_for_long_axis(plane, progress=_cb)
            # Cache for the rest of a rotation drag; a full rebuild drops it.
            # (Grayscale only — colour takes the persistent-luma branch above.)
            self._la_drag_frames = frames if draft else None
            dlg.setLabelText("Compositing long-axis image…")
            dlg.setMaximum(0)            # 0,0 -> indeterminate busy bar
            dlg.setValue(0)
            QApplication.processEvents()
            self._set_la_image(frames, center_pairs, lateral, to_u8,
                               draft=draft)
        finally:
            dlg.close()

    def _set_la_image(self, frames, center_pairs, lateral, to_u8,
                      draft: bool) -> None:
        """Composite *frames* into the long-axis strip and push it to the
        canvas. In *draft* mode both axes are sampled at 1/LOD and the
        result is nearest-upscaled back to the full (lateral, n_frames) size
        — so the displayed columns / aspect ratio / frame cursor are
        unchanged (no layout jump), only the detail is coarser."""
        if draft:
            k = self._LA_DRAFT_LOD
            lat_small = max(48, -(-lateral // k))          # ceil
            # step=k so the coarse samples still span the FULL vessel depth
            # (same physical extent as the full strip), then upscale the rows
            # back to `lateral` — otherwise the preview would sample only the
            # central 1/k and look vertically zoomed-in.
            small = build_long_axis(
                frames[::k], center_pairs[::k], self._la_angle,
                lat_small, to_u8, step=k,
            )
            up = np.repeat(np.repeat(small, k, axis=0), k, axis=1)
            img = np.ascontiguousarray(up[:lateral, :len(frames)])
        else:
            img = build_long_axis(
                frames, center_pairs, self._la_angle, lateral, to_u8
            )
        self._la_img = img
        self.long_axis.set_image(img)
        self.long_axis.set_current_frame(self._frame)

    # ----------------------------- mouse callbacks (long-axis canvas)
    def _on_la_rotated(self, dtheta: float) -> None:
        self._la_angle = (self._la_angle + float(dtheta)) % (2 * math.pi)
        # Rotate the cross-section cut line + projection triangles live too.
        self._refresh_center_marker()
        # Live preview at reduced LOD; full-res rebuild fires on release.
        self._rebuild_long_axis(draft=True)

    def _on_la_rotation_finished(self) -> None:
        # Drag ended → crisp full-resolution strip.
        self._rebuild_long_axis(draft=False)

    def _on_la_angle_set(self, angle: float) -> None:
        """Cut-line drag on the cross-section set an absolute long-axis
        angle. Sync the angle, redraw the cross-section guide on both
        planes, and preview the strip at draft LOD (full rebuild fires
        from ivus_angle_finished on release)."""
        self._la_angle = float(angle) % (2 * math.pi)
        self._refresh_center_marker()
        self._rebuild_long_axis(draft=True)

    def _on_la_frame_picked(self, idx: int) -> None:
        self.frame_slider.setValue(int(idx))   # triggers _seek

    def _on_la_keyframe_remove(self, scope: str, frame: int) -> None:
        """Right-click ▸ Remove on a long-axis keyframe ▼/▲. Routes to the
        shared centre-reset on the CLICKED keyframe (not necessarily the
        current frame), so removing a point on the strip behaves exactly
        like removing it on the cross-section's red centre marker."""
        self._on_center_reset(scope, frame)

    def _on_export_long_axis(self, fmt_key) -> None:
        """Right-click export on the long-axis strip. Still image (PNG/JPEG/
        TIFF) saves the strip; CSV exports the IVUS series' DICOM tags. The
        long-axis offers no DICOM/MP4 (a reconstructed strip isn't a DICOM
        cine — that would only confuse)."""
        if fmt_key == "csv":
            self.plane_export_requested.emit(
                "csv", getattr(self, "_loaded_uid", ""), ""
            )
            return
        export_image_as(self, self.long_axis.grab(), fmt_key,
                        safe_basename(self._export_basename(), "longaxis"))

    # ----------------------------- mouse callbacks (cross-section canvas)
    def _on_center_dragged(self, cx: float, cy: float) -> None:
        """User dragged the centre marker on the cross-section. Pin
        this frame as a keyframe at (cx, cy) and re-interpolate every
        non-keyed frame between adjacent keys so the new centre joins
        the per-frame "polyline" smoothly. See the design PDF: the
        intermediate frames' centres are the weighted average of the
        two neighbouring keys."""
        if not (self._planes and self._la_centers):
            return
        pi = self._active_plane_idx()
        centers = self._la_centers[pi]
        keyed = self._la_center_keyed[pi]
        if self._frame >= len(centers):
            return
        centers[self._frame] = (cx, cy)
        keyed[self._frame] = True
        self._reinterp_centers(pi)
        self._refresh_keyframe_markers()
        # Flip the marker colour blue → red the moment the user pins
        # this frame as a key.
        self._refresh_center_marker()
        if self._la_visible:
            self._rebuild_long_axis()

    def _on_center_reset(self, scope: str, frame: int | None = None) -> None:
        """Remove-point menu on the marker (cross-section red centre or a
        long-axis keyframe ▼/▲):

        * "frame": un-key *frame* (defaults to the current frame); its centre
          is then recomputed from interpolation (or returns to the image
          centre when no keys remain on the plane).
        * "all":  un-key every frame on this plane; every centre returns
          to the image centre, restoring the original long-axis view.
        """
        if not (self._planes and self._la_centers):
            return
        pi = self._active_plane_idx()
        plane = self._planes[pi]
        f0 = plane.volume[0]
        h, w = f0.shape[:2]
        cx0, cy0 = w / 2.0, h / 2.0
        centers = self._la_centers[pi]
        keyed = self._la_center_keyed[pi]
        if scope == "frame":
            fr = self._frame if frame is None else int(frame)
            if 0 <= fr < len(centers):
                keyed[fr] = False
            self._reinterp_centers(pi)
        else:  # "all"
            keyed[:] = False
            centers[:] = (cx0, cy0)
        self._refresh_center_marker()
        self._refresh_keyframe_markers()
        if self._la_visible:
            self._rebuild_long_axis()

    # ----------------------------- keyframe (centre) navigation
    def _keyframe_indices(self) -> np.ndarray:
        """Frame indices on the active plane that the user has pinned
        as keyframes (= manual rotation centres). Empty array when no
        series is loaded or no centres have been set."""
        if not (self._planes and self._la_center_keyed):
            return np.zeros(0, dtype=np.int64)
        pi = self._active_plane_idx()
        if pi >= len(self._la_center_keyed):
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(self._la_center_keyed[pi])

    def _refresh_keyframe_markers(self) -> None:
        """Sync the long-axis strip's keyframe markers and the enabled
        state of the ◀ Center / Center ▶ / Clear Centers buttons with
        the keyed-frame set. Called whenever that set changes (series
        load, centre drag, centre reset)."""
        idxs = self._keyframe_indices()
        self.long_axis.set_keyframes(idxs.tolist())
        has = bool(idxs.size)
        for b in (self._prev_key_btn, self._next_key_btn,
                  self._clear_centers_btn):
            b.setEnabled(has)

    def _jump_to_keyframe(self, direction: int) -> None:
        """Move the cine to the previous (direction = -1) or next
        (+1) keyframe on the active plane. Wraps around the ends.
        A no-op when the plane has no keyframes (the buttons are
        disabled in that state anyway, this is defence-in-depth)."""
        idxs = self._keyframe_indices()
        if idxs.size == 0:
            return
        cur = self._frame
        if direction > 0:
            after = idxs[idxs > cur]
            target = int(after[0] if after.size else idxs[0])
        else:
            before = idxs[idxs < cur]
            target = int(before[-1] if before.size else idxs[-1])
        # Stop cine first so the jump doesn't fight the timer, then
        # seek via the slider so frame_changed fires (MultiSync mirrors
        # this jump in any peer pane) and _render() runs.
        self.stop()
        self.frame_slider.setValue(target)
        self.readout.setText(
            f"Manual centre @ frame {target + 1}  ·  "
            f"{idxs.size} centre(s) on this plane"
        )

    def _clear_all_centers(self) -> None:
        """'Clear Centers' button. Removes every manual rotation
        centre on the active plane and restores the straight-catheter
        long-axis baseline — same as the marker's right-click ▸
        'Reset all', exposed as a button for discoverability."""
        if self._keyframe_indices().size == 0:
            return
        self._on_center_reset("all")
        self.readout.setText("Cleared all manual rotation centres on this plane.")

    def _reinterp_centers(self, pi: int) -> None:
        """Fill every non-keyed frame's centre with a linear interpolation
        between the nearest keyed neighbours. Frames before the first key
        (or after the last) clamp to that nearest key. With zero keys, all
        centres fall back to the plane's image centre — the "straight
        catheter" baseline."""
        centers = self._la_centers[pi]
        keyed = self._la_center_keyed[pi]
        n = len(centers)
        idxs = np.flatnonzero(keyed)
        if idxs.size == 0:
            plane = self._planes[pi]
            f0 = plane.volume[0]
            h, w = f0.shape[:2]
            centers[:] = (w / 2.0, h / 2.0)
            return
        first, last = int(idxs[0]), int(idxs[-1])
        # Clamp the ends to the nearest keyed value.
        centers[:first] = centers[first]
        centers[last + 1:] = centers[last]
        # Linear-interpolate every gap between consecutive keys.
        for a, b in zip(idxs[:-1], idxs[1:]):
            a, b = int(a), int(b)
            if b == a + 1:
                continue            # no gap to fill
            ca = centers[a].astype(np.float32)
            cb = centers[b].astype(np.float32)
            ts = (np.arange(a + 1, b, dtype=np.float32) - a) / float(b - a)
            centers[a + 1:b, 0] = ca[0] + ts * (cb[0] - ca[0])
            centers[a + 1:b, 1] = ca[1] + ts * (cb[1] - ca[1])
