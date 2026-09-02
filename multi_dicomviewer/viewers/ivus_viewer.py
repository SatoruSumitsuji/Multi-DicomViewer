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

import json
import math

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSplitter,
)

from multi_dicomviewer.core import coreg
from multi_dicomviewer.core.dicom_io import apply_color_mode_to_planes
from multi_dicomviewer.core.image_export import export_image_as, safe_basename
from multi_dicomviewer.core.settings import load_ivus_color, save_ivus_color
from multi_dicomviewer.i18n import t
from multi_dicomviewer.viewers.long_axis_canvas import (
    LongAxisCanvas, build_long_axis,
)
from multi_dicomviewer.viewers.xa_viewer import (
    XAViewer, _AutoHideLabel, _Prefetcher, apply_window,
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

    #: Export the angle keyframes — (series_uid, payload dict). The shell
    #: reuses the normal export filename-tag picker to name the file.
    angle_export_requested = pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        # %PlaqueArea is now done through the shared "Compare" flow (pick two
        # Ellipse/Polygon, choose %PA / Thickness) — same as the CT viewers —
        # so there is no longer an IVUS-specific auto-%PA mode to enable here.

        # Per-frame image-rotation keyframes (CoSync-style). {frame: angle°}
        # per series (keyed by SeriesInstanceUID so switching series back
        # restores them). When set, each shown frame's angle is interpolated
        # (shortest ≤180° path, held constant before the first / after the last
        # keyframe) instead of one angle for the whole pull-back.
        self._angle_kf_by_uid: dict[str, dict[int, float]] = {}
        self._angle_kf: dict[int, float] = {}

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
        #: Long-axis is only allowed in the 1x1 (full-screen) layout — in a
        #: small multi-pane cell the strip is uselessly tiny and its rebuild is
        #: the heaviest op (the main freeze source). The shell flips this via
        #: set_long_axis_allowed() on every layout change.
        self._la_allowed: bool = True
        #: Built lazily — kept so frame changes only update the cursor,
        #: not the full strip.
        self._la_img: np.ndarray | None = None
        #: Decoded frames cached for the duration of a rotation drag (the
        #: angle changes per move, the frames don't), so each preview only
        #: re-composites instead of re-decoding. Set on the drag's first
        #: build, cleared by the full-resolution rebuild on release.
        self._la_drag_frames: list | None = None
        #: Debounce for rebuilding the long-axis strip after a W/L drag. Each
        #: W/L tick previews the strip at draft LOD (cheap); this fires ~160 ms
        #: after the user stops to rebuild it crisp. Without it every W/L step
        #: ran the full-resolution (decode-gated, modal) rebuild and froze the
        #: viewer — badly in a 2×3 grid.
        self._la_wl_timer = QTimer(self)
        self._la_wl_timer.setSingleShot(True)
        self._la_wl_timer.setInterval(160)
        self._la_wl_timer.timeout.connect(self._on_la_wl_settle)
        #: While the background prefetch is still warming this pull-back, the
        #: long-axis is built from whatever frames are ready (cold frames are
        #: black columns) — it NEVER block-decodes on the UI thread. This timer
        #: re-composites every ~1 s so the strip fills in as frames warm, then
        #: stops (and rebuilds crisp) once every frame is ready. This is what
        #: stopped a long colour pull-back freezing / aborting the UI on a
        #: long-axis rotate.
        self._la_warm_timer = QTimer(self)
        self._la_warm_timer.setInterval(1000)
        self._la_warm_timer.timeout.connect(self._on_la_warm_tick)

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
        # Also enable CoSync-style free drag-rotation of the cross-section: an
        # empty left-drag on the image body spins it about the centre (lowest
        # priority — never steals the centre-marker/cut-line or a measurement;
        # suppressed while the long-axis centre marker is shown). "Reset" (in
        # the orientation toolbar row) undoes it along with Rt90/Lt90/Flip.
        for c in (self.canvas, self.canvas2):
            c.set_free_rotation_enabled(True)
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

        # "Angle Set" (left of Long View): rotate the cross-section to the angle
        # you want at THIS frame, then click to key it. The seek bar marks each
        # keyframe and the angle is interpolated between them (CoSync-style).
        self._angle_set_btn = QPushButton(t("Multi-Rot Angle Set"))
        self._angle_set_btn.setToolTip(
            t("Key the current frame's rotation angle. Angles are interpolated "
              "between keyframes (shortest path); the first/last keys are held "
              "to frame 1 / the last frame. Right-click to clear all keys."))
        self._angle_set_btn.clicked.connect(self._on_angle_set)
        self._angle_set_btn.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._angle_set_btn.customContextMenuRequested.connect(
            lambda _p: self._clear_angle_keys())
        ivus_row.addWidget(self._angle_set_btn)

        # "Reset" (right of Angle Set, normal gap): drop every angle keyframe
        # and return the cross-section rotation to 0°.
        self._angle_reset_btn = QPushButton(t("Reset"))
        self._angle_reset_btn.setToolTip(
            t("Clear all angle keyframes and reset the rotation to 0°."))
        self._angle_reset_btn.clicked.connect(self._on_angle_reset)
        ivus_row.addWidget(self._angle_reset_btn)

        # "Export" / "Import" the angle keyframes. Export routes through the
        # shell so the filename reuses the same DICOM-tag picker as the other
        # exports; Import reads a saved file straight back.
        self._angle_export_btn = QPushButton(t("Export"))
        self._angle_export_btn.setToolTip(
            t("Save the angle keyframes to a file (filename tags chosen like "
              "the other exports)."))
        self._angle_export_btn.clicked.connect(self._on_angle_export_clicked)
        ivus_row.addWidget(self._angle_export_btn)
        self._angle_import_btn = QPushButton(t("Import"))
        self._angle_import_btn.setToolTip(
            t("Load angle keyframes from a previously exported file."))
        self._angle_import_btn.clicked.connect(self._on_angle_import)
        ivus_row.addWidget(self._angle_import_btn)

        # 3× the default button gap between the Angle-Set group and Long View
        # (measured: gap = layout spacing + addSpacing, so 2× spacing → 3× total).
        _gap = ivus_row.spacing()
        if _gap < 0:
            _gap = 6
        ivus_row.addSpacing(2 * _gap)

        # "Long View" toggle. Mirrors the V shortcut so the check state always
        # matches _la_visible. Width follows the label (no fixed minimum).
        self._long_view_btn = QPushButton(t("Long View"))
        self._long_view_btn.setCheckable(True)
        # Tight horizontal padding so the button hugs the "Long View" label
        # (its width is otherwise bounded only by the bold text).
        self._long_view_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding:2px 6px; }"
            "QPushButton:checked { background:#1f77b4; color:white; }"
        )
        self._long_view_btn.setToolTip(
            t("Show/hide the IVUS long-axis (longitudinal) view — shortcut: V")
        )
        self._long_view_btn.clicked.connect(self.toggle_long_axis)
        ivus_row.addWidget(self._long_view_btn)

        # Centre-keyframe controls — right of Long View so the whole "long-axis"
        # cluster reads left-to-right. Prev/Next cycle through the keyed frames
        # on the active plane (wraps); Clear removes every manual centre.
        self._prev_key_btn = QPushButton(t("◀ Center"))
        self._prev_key_btn.setToolTip(
            t("Jump to the previous frame with a manual rotation centre")
        )
        self._prev_key_btn.clicked.connect(
            lambda: self._jump_to_keyframe(-1)
        )
        ivus_row.addWidget(self._prev_key_btn)
        self._next_key_btn = QPushButton(t("Center ▶"))
        self._next_key_btn.setToolTip(
            t("Jump to the next frame with a manual rotation centre")
        )
        self._next_key_btn.clicked.connect(
            lambda: self._jump_to_keyframe(+1)
        )
        ivus_row.addWidget(self._next_key_btn)
        self._clear_centers_btn = QPushButton(t("Clear Centers"))
        self._clear_centers_btn.setToolTip(
            t("Remove every manual rotation centre on the active plane "
              "(same as right-click ▸ Reset all on the marker)")
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
        self._color_btn = QPushButton(t("Color"))
        self._color_btn.setCheckable(True)
        self._color_btn.setToolTip(
            t("IVUS: Gray to Color (NIRS chemograms, etc.)\n"
              "Return to Gray")
        )
        self._color_btn.setStyleSheet(
            "QPushButton:checked { background:#c0392b; color:white; }"
        )
        self._color_btn.clicked.connect(self._on_color_toggle)
        ivus_row.addWidget(self._color_btn)
        # Insert as the second item of the viewer's main column (index 1),
        # directly under the inherited series-nav row (index 0).
        self.layout().insertLayout(1, ivus_row)
        # Angle-keyframe summary, shown BELOW the seek bar in the same teal as
        # the readout (auto-hides when there are no keys). Placed just above the
        # readout so both sit under the transport strip.
        self._angle_info = _AutoHideLabel("")
        self._angle_info.setStyleSheet("color:#27e0c0; padding:2px 6px;")
        self._angle_info.setVisible(False)
        self.layout().insertWidget(
            self.layout().indexOf(self.readout), self._angle_info)
        # Seek-bar angle markers are interactive: click → jump to that key's
        # frame (then rotate + Angle Set to update it); right-click → Delete.
        if hasattr(self, "_range_marks"):
            self._range_marks.angle_mark_clicked.connect(
                self._on_angle_mark_clicked)
            self._range_marks.angle_mark_delete.connect(
                self._on_angle_mark_delete)
        # Buttons start disabled — they enable once a series is loaded
        # with at least one keyframe (see _refresh_keyframe_markers).
        for b in (self._prev_key_btn, self._next_key_btn,
                  self._clear_centers_btn):
            b.setEnabled(False)

    # ============================================================ public
    def set_long_axis_allowed(self, allowed: bool) -> None:
        """Enable/disable the long-axis feature for this viewer. The shell
        calls this on every layout change: long-axis is allowed ONLY in the
        1x1 (full-screen) layout. In a multi-pane grid the strip is uselessly
        small and its rebuild is the heaviest op (the main freeze source), so
        it is disabled there. Disabling force-hides a currently-shown strip and
        greys out the Long View button; the V shortcut becomes a no-op."""
        self._la_allowed = bool(allowed)
        if hasattr(self, "_long_view_btn"):
            self._long_view_btn.setEnabled(self._la_allowed)
            self._long_view_btn.setToolTip(
                t("Long-axis (longitudinal) view — available only in "
                  "single-pane (1x1) layout") if not self._la_allowed
                else t("Show/hide the long-axis (longitudinal) view  [V]")
            )
        if not self._la_allowed and self._la_visible:
            self.toggle_long_axis()      # hide it (stops warm timer, clears)

    def retranslate_ui(self) -> None:
        """Re-apply every persistent, user-facing string this viewer builds so
        a runtime language switch takes effect without a restart. Mirrors the
        constructor's t() calls exactly; two-state labels/tooltips are
        re-derived from the CURRENT state (never flipped). Safe to call with or
        without a series loaded (every ref is guarded)."""
        # Let the inherited (XA) viewer re-apply its own persistent strings
        # first — the series-nav row, the DICOM-tag font control built by
        # build_tag_font_control, "Measure History", etc. all live there.
        sup = getattr(super(), "retranslate_ui", None)
        if callable(sup):
            sup()

        # "Angle Set" button.
        btn = getattr(self, "_angle_set_btn", None)
        if btn is not None:
            btn.setText(t("Multi-Rot Angle Set"))
            btn.setToolTip(
                t("Key the current frame's rotation angle. Angles are "
                  "interpolated between keyframes (shortest path); the "
                  "first/last keys are held to frame 1 / the last frame. "
                  "Right-click to clear all keys."))
        btn = getattr(self, "_angle_reset_btn", None)
        if btn is not None:
            btn.setText(t("Reset"))
            btn.setToolTip(
                t("Clear all angle keyframes and reset the rotation to 0°."))
        btn = getattr(self, "_angle_export_btn", None)
        if btn is not None:
            btn.setText(t("Export"))
            btn.setToolTip(
                t("Save the angle keyframes to a file (filename tags chosen "
                  "like the other exports)."))
        btn = getattr(self, "_angle_import_btn", None)
        if btn is not None:
            btn.setText(t("Import"))
            btn.setToolTip(
                t("Load angle keyframes from a previously exported file."))
        # Re-render the angle-keyframe summary label in the new language.
        if getattr(self, "_angle_kf", None) is not None:
            self._refresh_angle_marks()

        # "Long View" toggle. Text is state-independent; the tooltip depends on
        # whether the long-axis is currently allowed (see set_long_axis_allowed).
        btn = getattr(self, "_long_view_btn", None)
        if btn is not None:
            btn.setText(t("Long View"))
            if getattr(self, "_la_allowed", True):
                btn.setToolTip(
                    t("Show/hide the IVUS long-axis (longitudinal) view "
                      "— shortcut: V")
                )
            else:
                btn.setToolTip(
                    t("Long-axis (longitudinal) view — available only in "
                      "single-pane (1x1) layout")
                )

        # Centre-keyframe navigation cluster.
        btn = getattr(self, "_prev_key_btn", None)
        if btn is not None:
            btn.setText(t("◀ Center"))
            btn.setToolTip(
                t("Jump to the previous frame with a manual rotation centre")
            )
        btn = getattr(self, "_next_key_btn", None)
        if btn is not None:
            btn.setText(t("Center ▶"))
            btn.setToolTip(
                t("Jump to the next frame with a manual rotation centre")
            )
        btn = getattr(self, "_clear_centers_btn", None)
        if btn is not None:
            btn.setText(t("Clear Centers"))
            btn.setToolTip(
                t("Remove every manual rotation centre on the active plane "
                  "(same as right-click ▸ Reset all on the marker)")
            )

        # "Color" (カラー表示) toggle. Text is state-independent; the tooltip
        # names both directions of the toggle, so it is re-applied verbatim.
        btn = getattr(self, "_color_btn", None)
        if btn is not None:
            btn.setText(t("Color"))
            btn.setToolTip(
                t("IVUS: Gray to Color (NIRS chemograms, etc.)\n"
                  "Return to Gray")
            )

        # Cascade to the child canvases so any on-image overlay / longitudinal
        # strip re-translates and repaints. Call each canvas' own
        # retranslate_ui when present, then force a repaint regardless.
        for c in (getattr(self, "canvas", None),
                  getattr(self, "canvas2", None),
                  getattr(self, "long_axis", None)):
            if c is None:
                continue
            rt = getattr(c, "retranslate_ui", None)
            if callable(rt):
                rt()
            c.update()

        self.update()

    def toggle_long_axis(self) -> None:
        """V shortcut / Long View button entry point. Shows/hides the
        strip and the per-frame rotation-centre marker on the cross-
        section canvas. Keeps the Long View button's check state in
        sync with the visibility so V and the button never disagree."""
        self._on_user_interaction()
        # Disallowed in multi-pane (see set_long_axis_allowed): never turn ON.
        if not self._la_visible and not self._la_allowed:
            return
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
            self._la_warm_timer.stop()
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
        # Restore this series' per-frame angle keyframes (empty for a series
        # never keyed) and apply the interpolated angle to the shown frame.
        uid = getattr(self, "_loaded_uid", "")
        self._angle_kf = self._angle_kf_by_uid.setdefault(uid, {}) if uid \
            else {}
        self._refresh_angle_marks()
        self._apply_frame_angle()
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
                t("This IVUS has no color information")
            )
            return
        uid = getattr(self, "_loaded_uid", "")
        if uid:
            save_ivus_color(uid, achieved)
        self.readout.setText(
            t("Switched to color.") if achieved
            else t("Reverted to grayscale.")
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
        self._sync_wl_enabled()      # grey out right-click Change W/L in colour
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
        # Drive the per-frame rotation BEFORE the base render so the shown
        # frame is already at its interpolated angle.
        self._apply_frame_angle()
        super()._render()
        # Cursor on the long-axis follows the cine; the strip itself
        # only rebuilds on rotation / centre / W-L changes.
        if self._la_visible:
            self.long_axis.set_current_frame(self._frame)
            self._refresh_center_marker()

    # ==================================================== per-frame rotation
    def _apply_frame_angle(self) -> None:
        """If angle keyframes exist, set the cross-section's rotation to the
        value interpolated at the current frame (shortest ≤180° path, held
        constant outside the keyed range). No keyframes → leave rotation as-is
        (a single manual angle then still applies to every frame, as before)."""
        if not self._angle_kf:
            return
        pairs = list(self._angle_kf.items())
        ang = coreg.map_rotation(pairs, self._frame)
        if ang is not None:
            self.canvas.set_free_rotation(ang)

    def _on_angle_set(self) -> None:
        """"Angle Set": key the current frame at the cross-section's current
        rotation angle (overwrites any existing key on that frame)."""
        if not self._planes:
            return
        self._angle_kf[int(self._frame)] = float(self.canvas.free_rotation())
        self._refresh_angle_marks()
        self._apply_frame_angle()

    def _clear_angle_keys(self) -> None:
        """Right-click on "Angle Set" → drop every angle keyframe for this
        series (rotation stays at its current value)."""
        if not self._angle_kf:
            return
        self._angle_kf.clear()
        self._refresh_angle_marks()

    def _on_angle_reset(self) -> None:
        """"Reset" (next to Angle Set): clear every angle keyframe and return
        the cross-section rotation to 0° (does not touch the 90°/flip
        orientation — that has its own Reset in the toolbar)."""
        self._angle_kf.clear()
        self._refresh_angle_marks()
        for c in (self.canvas, self.canvas2):
            c.set_free_rotation(0.0)

    def _on_angle_export_clicked(self) -> None:
        """"Export": hand the angle keyframes to the shell, which shows the
        usual filename-tag picker and writes the file."""
        if not self._angle_kf:
            self.readout.setText(t("No angle keyframes to export."))
            return
        uid = getattr(self, "_loaded_uid", "") or ""
        payload = {
            "format": "MDV-IVUS-Angles",
            "version": 1,
            "series_uid": uid,
            "keyframes": {str(f): self._angle_kf[f]
                          for f in sorted(self._angle_kf)},
        }
        self.angle_export_requested.emit(uid, payload)

    def _on_angle_import(self) -> None:
        """"Import": load angle keyframes from a previously exported file and
        apply them to the current series."""
        path, _ = QFileDialog.getOpenFileName(
            self, t("Import Angle Set"), "",
            t("IVUS Angle Set (*.json);;All files (*)"))
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, t("Import Angle Set"), str(exc))
            return
        if not isinstance(data, dict) or data.get("format") != "MDV-IVUS-Angles":
            QMessageBox.warning(
                self, t("Import Angle Set"),
                t("This is not an IVUS Angle Set file."))
            return
        kf: dict[int, float] = {}
        for k, v in (data.get("keyframes") or {}).items():
            try:
                kf[int(k)] = float(v)
            except (TypeError, ValueError):
                continue
        self._angle_kf.clear()
        self._angle_kf.update(kf)
        self._refresh_angle_marks()
        self._apply_frame_angle()
        self.readout.setText(t("Imported {n} angle keyframe(s).", n=len(kf)))

    def _on_angle_mark_clicked(self, frame: int) -> None:
        """Click a seek-bar angle marker → jump to that keyframe's frame so the
        user can rotate and re-key it (Angle Set overwrites that frame)."""
        self.frame_slider.setValue(int(frame))

    def _on_angle_mark_delete(self, frame: int) -> None:
        """Right-click ▸ Delete on a seek-bar angle marker → drop that key and
        re-apply the (now re-interpolated) angle to the shown frame."""
        if int(frame) in self._angle_kf:
            del self._angle_kf[int(frame)]
            self._refresh_angle_marks()
            self._apply_frame_angle()

    def _refresh_angle_marks(self) -> None:
        """Sync the seek-bar keyframe triangles and the below-seekbar summary
        label to the current angle keyframes."""
        frames = sorted(self._angle_kf)
        last = max(self.frame_slider.maximum(), 0)
        if hasattr(self, "_range_marks"):
            self._range_marks.set_angle_marks(frames, 0, last)
        if frames:
            parts = ", ".join(
                f"F{f + 1}: {round(self._angle_kf[f])}°" for f in frames)
            self._angle_info.setText(t("Angle keyframes — {list}", list=parts))
        else:
            self._angle_info.setText("")

    def _reset_transform(self) -> None:
        """Reset also drops the per-frame angle keyframes (back to a single,
        un-rotated view)."""
        self._angle_kf.clear()
        self._refresh_angle_marks()
        super()._reset_transform()

    def _on_user_interaction(self) -> None:
        """Dismiss the transient 'no colour information' readout on any
        transport / measure / zoom / orientation action (see XAViewer)."""
        if self.readout.text() == t("This IVUS has no color information"):
            self.readout.setText("")

    def _wl_changed(self):
        super()._wl_changed()
        if self._la_visible:
            # Live preview at draft LOD (re-composites cached frames — cheap),
            # then a crisp full-res rebuild once the slider settles. The first
            # draft decodes+caches the frames (slow path); subsequent ticks hit
            # the fast re-composite path, so dragging stays responsive.
            self._rebuild_long_axis(draft=True)
            self._la_wl_timer.start()

    def _on_la_wl_settle(self) -> None:
        """W/L slider stopped → rebuild the long-axis strip at full resolution."""
        if self._la_visible:
            self._rebuild_long_axis(draft=False)

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

    def _frames_for_long_axis(self, plane, progress=None):
        """All frames of *plane*. Forces a decode of any not-yet-ready
        frame (the prefetch usually has them; for a long pull-back this
        may briefly block the UI). *progress*, if given, is called as
        ``progress(done, total)`` after every frame so the caller can
        drive a QProgressDialog.

        Returns the plane's contiguous ``(n, H, W[, 3])`` volume ndarray
        when EVERY frame decoded — so ``build_long_axis`` can gather all
        columns in one vectorised pass instead of looping per frame (the
        per-frame loop was the bulk of the long-axis rebuild freeze). If any
        frame failed to decode, returns a list with ``None`` placeholders so
        the caller's per-frame fallback can skip them."""
        out: list[np.ndarray | None] = []
        n = plane.total_frames
        ok = True
        for i in range(n):
            try:
                out.append(plane.frame(i))
            except Exception:
                out.append(None)
                ok = False
            if progress is not None:
                progress(i + 1, n)
        vol = getattr(plane, "volume", None)
        if ok and isinstance(vol, np.ndarray) and vol.shape[0] == n:
            return vol
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
            # Apply W/L the SAME way the cross-section does (XAViewer._frame_of)
            # so the strip tracks the W/L sliders. The LUT is built only for
            # grayscale integer series (None in colour mode), so it is tried
            # FIRST — otherwise an 8-bit IVUS (uint8) short-circuited here and
            # the strip ignored W/L. The uint8 pass-through then only catches
            # already-display-ready pixels (e.g. a colour RGB strip).
            if (wl_lut is not None
                    and np.issubdtype(col.dtype, np.integer)):
                # Wide-int index so a signed dtype (wl_off>0) can't overflow
                # when the offset is added (NumPy 2 / NEP50 raises instead of
                # upcasting). Mirrors XAViewer._frame_of.
                if wl_off:
                    return wl_lut[col.astype(np.intp) + wl_off]
                return wl_lut[col]
            if col.dtype == np.uint8:
                return col
            return apply_window(col, window, level)

        center_pairs = [(float(c[0]), float(c[1])) for c in centers]

        # Build the strip straight from the plane's VOLUME — NEVER block-decode
        # on the UI thread. Frames the background prefetch hasn't warmed yet are
        # zero (black) columns; the warm-timer re-composites as they fill in and
        # finalises once all are ready. Forcing a synchronous full decode here
        # (esp. a multi-GB colour pull-back) was what froze — and sometimes
        # aborted — the UI on a long-axis rotate, and no W/L draft path avoided
        # it because the very FIRST build had to decode everything.
        frames = plane.volume
        ready = getattr(plane, "_ready", None)
        all_ready = bool(ready.all()) if ready is not None else True

        self._set_la_image(frames, center_pairs, lateral, to_u8, draft=draft)

        # Keep filling the strip as the prefetch warms the rest; finalise &
        # stop once every frame is ready.
        if all_ready:
            self._la_warm_timer.stop()
        elif not self._la_warm_timer.isActive():
            self._la_warm_timer.start()

    def _on_la_warm_tick(self) -> None:
        """Periodic re-composite while the pull-back warms (frames decode in
        the background). Re-draws at draft LOD so the refresh itself stays
        cheap; once every frame is ready it rebuilds crisp and stops."""
        if not (self._la_visible and self._planes and self._la_centers):
            self._la_warm_timer.stop()
            return
        plane = self._planes[self._active_plane_idx()]
        ready = getattr(plane, "_ready", None)
        if ready is None or bool(ready.all()):
            self._la_warm_timer.stop()
            self._rebuild_long_axis(draft=False)      # final crisp strip
        else:
            self._rebuild_long_axis(draft=True)       # cheap progressive fill

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

    # ---- Case Presentation: extend the 2-D state with the long-axis angle
    def capture_view_state(self) -> dict:
        st = super().capture_view_state()
        st["kind"] = "ivus"
        st["la_angle"] = float(getattr(self, "_la_angle", 0.0))
        return st

    def restore_view_state(self, st: dict) -> None:
        super().restore_view_state(st)
        if not isinstance(st, dict) or st.get("la_angle") is None:
            return
        try:
            self._la_angle = float(st["la_angle"]) % (2 * math.pi)
            if hasattr(self, "_refresh_center_marker"):
                self._refresh_center_marker()
            self._rebuild_long_axis(draft=False)
        except Exception:                                # noqa: BLE001
            pass

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
            t("Manual centre @ frame {frame}  ·  "
              "{count} centre(s) on this plane",
              frame=target + 1, count=idxs.size)
        )

    def _clear_all_centers(self) -> None:
        """'Clear Centers' button. Removes every manual rotation
        centre on the active plane and restores the straight-catheter
        long-axis baseline — same as the marker's right-click ▸
        'Reset all', exposed as a button for discoverability."""
        if self._keyframe_indices().size == 0:
            return
        self._on_center_reset("all")
        self.readout.setText(t("Cleared all manual rotation centres on this plane."))

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
