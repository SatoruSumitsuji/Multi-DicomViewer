"""IVUS–XA co-registration window.

Shows one representative angiographic view (or up to a few) alongside one
or more IVUS pull-backs of the SAME vessel. The user traces a smooth
*guide* curve along the vessel on each angio, then pins *CoReg points*
(anatomical landmarks): for a landmark, scrub an IVUS to the frame that
shows it and slide the on-guide circle on each angio to where that
cross-section sits. Once ≥2 landmarks are pinned, scrubbing the active
IVUS drives a marker that rides each angio's guide, showing which
cross-section the current IVUS frame corresponds to.

Data model (kept here; the pure math is in :mod:`core.coreg`):

* **guide** — per angio pane: the user's control vertices + the smoothed
  curve, both in image-pixel coords. One vessel ⇒ one guide per view.
* **landmark** — a shared anatomical point: ``frames`` maps an IVUS
  pane-index → the frame that shows it (skippable per IVUS), ``fracs``
  maps an angio pane-index → the arc-length fraction on that view's guide
  (skippable per view). Both sides can be skipped where the landmark is
  not identifiable.

Driving is by a single *active* IVUS (master); switch it to review
pre/post/FU at the same anatomy. The panes are deliberately lightweight
(bare ImageCanvas + a scrub slider) so the window stays uncluttered.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.core import coreg, dicom_io
from multi_dicomviewer.core.study_model import Modality, Series
from multi_dicomviewer.viewers.image_canvas import ImageCanvas
from multi_dicomviewer.viewers.xa_viewer import apply_window


def _grid_dims(n: int) -> int:
    """Number of columns for *n* panes: 1 for a single pane, 2 up to four,
    3 for five or six."""
    if n <= 1:
        return 1
    if n <= 4:
        return 2
    return 3


class _CoregPane:
    """One pane: an ImageCanvas + a scrub slider bound to a series.

    Role follows the series modality: an IVUS series is a scrubbable
    master candidate; any other (XA/angio) is a still view that carries
    the guide + marker overlay and its editing."""

    def __init__(self, index: int, owner: "CoregWindow"):
        self.index = index
        self.owner = owner
        self.series: Series | None = None
        self.plane = None
        self.total = 0
        self.cur = 0
        self.window = 1.0
        self.level = 0.5
        self.is_color = False
        self.is_ivus = False

        self.frame = QFrame()
        self.frame.setFrameShape(QFrame.Shape.Box)
        col = QVBoxLayout(self.frame)
        col.setContentsMargins(3, 3, 3, 3)
        col.setSpacing(2)

        self.title = QLabel("—")
        self.title.setStyleSheet("QLabel { color:#cfcfcf; }")
        col.addWidget(self.title)

        self.canvas = ImageCanvas()
        col.addWidget(self.canvas, 1)

        srow = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.lbl = QLabel("—")
        self.lbl.setMinimumWidth(90)
        srow.addWidget(self.slider, 1)
        srow.addWidget(self.lbl)
        col.addLayout(srow)

        # IVUS-only row: Master radio + a placement "include" checkbox.
        self.nav = QHBoxLayout()
        self.master_radio = QRadioButton("Master")
        self.master_radio.setToolTip(
            "Make this the active IVUS — its frame drives the markers")
        self.master_radio.toggled.connect(self._on_master_toggled)
        self.include_cb = QCheckBox("この点に含める")
        self.include_cb.setChecked(True)
        self.include_cb.setToolTip(
            "While adding a CoReg point, include this IVUS's current frame")
        self.include_cb.setVisible(False)
        self.nav.addWidget(self.master_radio)
        self.nav.addStretch(1)
        self.nav.addWidget(self.include_cb)
        col.addLayout(self.nav)

    # ---- loading ----
    def load(self, series: Series | None, plane_index: int = 0,
             start_frame: int = 0) -> None:
        """Load *series* into this pane. *plane_index* selects which biplane
        projection to show (0 = Lt / left, 1 = Rt / right) so the pane
        mirrors what the source pane displayed; ignored for single-plane.
        *start_frame* opens on the frame the source pane was showing (so an
        angio still keeps the frame the user picked, not frame 0)."""
        self.series = series
        if series is None:
            self.plane = None
            self.total = 0
            self.title.setText("—")
            self.slider.setEnabled(False)
            self.lbl.setText("—")
            self.canvas.set_frame(np.zeros((8, 8), np.uint8))
            return
        loaded = dicom_io.load_xa(series)
        planes = loaded.xa_planes or [None]
        pidx = plane_index if 0 <= plane_index < len(planes) else 0
        self.plane = planes[pidx]
        self.total = self.plane.total_frames if self.plane else 0
        self.is_color = bool(loaded.is_color
                             or getattr(self.plane, "is_color", False))
        self.is_ivus = (series.modality == Modality.IVUS)
        self._init_wl(loaded)
        self.canvas.spacing_mm = dicom_io.series_spacing_mm(series)
        self.master_radio.setVisible(self.is_ivus)
        # Angio panes carry the CoReg overlay; IVUS panes do not but get
        # MultiSync-style drag-to-rotate of the cross-section.
        self.canvas.set_coreg_visible(not self.is_ivus)
        self.canvas.set_free_rotation_enabled(self.is_ivus)
        mod = series.modality.value if hasattr(series.modality, "value") \
            else str(series.modality)
        side = f" {'Lt' if pidx == 0 else 'Rt'}" if len(planes) > 1 else ""
        self.title.setText(f"{mod}{side} — {getattr(series, 'label', '')}")
        start = (max(0, min(int(start_frame), self.total - 1))
                 if self.total > 0 else 0)
        self.cur = start
        self.slider.blockSignals(True)
        self.slider.setEnabled(self.total > 1)
        self.slider.setRange(0, max(0, self.total - 1))
        self.slider.setValue(start)
        self.slider.blockSignals(False)
        self.show_frame(start)

    def _init_wl(self, loaded) -> None:
        """Default window/level: the loaded values when present, else the
        first frame's value range (mirrors what the IVUS viewer does)."""
        self.window = float(loaded.window or 0.0)
        self.level = float(loaded.level or 0.0)
        if self.window <= 0 and self.plane is not None and not self.is_color:
            f = self.plane.frame(0)
            vmin, vmax = float(np.min(f)), float(np.max(f))
            self.window = max(vmax - vmin, 1.0)
            self.level = (vmax + vmin) / 2.0

    def _display_frame(self, idx: int) -> np.ndarray:
        f = self.plane.frame(idx)
        if self.is_color:
            return f
        return apply_window(f, self.window, self.level)

    def show_frame(self, idx: int) -> None:
        if self.plane is None:
            return
        idx = max(0, min(int(idx), self.total - 1))
        self.cur = idx
        self.canvas.set_frame(self._display_frame(idx))
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.lbl.setText(f"F {idx + 1} / {self.total}")

    # ---- events ----
    def _on_slider(self, value: int) -> None:
        if self.plane is None:
            return
        self.show_frame(int(value))
        if self.is_ivus:
            self.owner._on_ivus_scrub(self.index)

    def _on_master_toggled(self, on: bool) -> None:
        if on:
            self.owner._set_active_ivus(self.index)

    def _on_slider_pressed(self) -> None:
        # Grabbing an IVUS seekbar means the user is positioning for a NEW
        # CoReg point → drop any landmark selected for modify.
        if self.is_ivus:
            self.owner._on_ivus_grab()


class CoregWindow(QMainWindow):
    """Standalone IVUS–XA co-registration window."""

    def __init__(self, pane_specs, parent=None):
        """*pane_specs*: list of ``(Series, plane_index)`` — one per pane to
        create, mirroring the source panes on screen. plane_index picks the
        biplane projection (0=Lt, 1=Rt); a biplane pane shown as "Bi"
        contributes two specs (both planes)."""
        super().__init__(parent)
        self.setWindowTitle("IVUS-XA CoReg")
        self.resize(1240, 900)

        self.panes: list[_CoregPane] = []
        #: pane_idx -> {"vertices": [...], "curve": [...]} for angio panes.
        self._guides: dict[int, dict] = {}
        #: landmarks: {"frames": {ivus_idx: frame}, "fracs": {xa_idx: s}}.
        self._landmarks: list[dict] = []
        self._active_ivus: int = -1
        self._tracing: int = -1          # angio pane-idx being traced, or -1
        self._placing: bool = False      # adding/modifying a CoReg point
        self._pending_fracs: dict[int, float] = {}   # xa_idx -> provisional s
        self._selected_lm: int = -1      # CoReg point selected (blue) / modify
        self._syncing: bool = False      # re-entrancy guard for IVUS↔IVUS sync

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._grid_host, 1)
        root.addWidget(self._build_panel())

        self._load_series(pane_specs)

    # ------------------------------------------------------------- build
    def _build_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(300)
        v = QVBoxLayout(panel)

        # Guide section — one Trace/Clear row per angio pane (filled later).
        gbox = QGroupBox("ガイド線 (血管トレース)")
        self._guide_box = QVBoxLayout(gbox)
        v.addWidget(gbox)

        # CoReg points section.
        cbox = QGroupBox("CoReg点")
        cv = QVBoxLayout(cbox)
        # Mode toggle: white = click to enter create/modify (goes blue); blue
        # = click to commit (確定). Modifies the selected CoReg point, or
        # creates a new one when none is selected.
        self._add_btn = QPushButton("CoReg点の追加・修正・確定")
        self._add_btn.clicked.connect(self._on_mode_button)
        cv.addWidget(self._add_btn)
        self._sort_btn = QPushButton("CoReg点をソート")
        self._sort_btn.setToolTip("アクティブIVUSのフレーム順に並べ替え")
        self._sort_btn.clicked.connect(self._sort_landmarks)
        cv.addWidget(self._sort_btn)
        self._clear_all_btn = QPushButton("CoReg点の一括削除")
        self._clear_all_btn.clicked.connect(self._clear_all_landmarks)
        cv.addWidget(self._clear_all_btn)
        self._cancel_btn = QPushButton("キャンセル")
        self._cancel_btn.clicked.connect(self._cancel_placing)
        self._cancel_btn.setVisible(False)
        cv.addWidget(self._cancel_btn)
        # Commit a CoReg point with NO XA position — for landmarks that are
        # only identifiable on IVUS (used for IVUS↔IVUS sync only).
        self._exclude_xa_cb = QCheckBox("XA除外（IVUS同期のみ）")
        self._exclude_xa_cb.setToolTip(
            "XA上で位置を決めず、このCoReg点をIVUS同期にだけ使う")
        self._exclude_xa_cb.setVisible(False)
        cv.addWidget(self._exclude_xa_cb)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("QLabel { color:#88ccff; }")
        cv.addWidget(self._status)
        # One row per CoReg point: [CoReg-N (jump)] [✕ (delete)].
        self._lm_host = QWidget()
        self._lm_box = QVBoxLayout(self._lm_host)
        self._lm_box.setContentsMargins(0, 0, 0, 0)
        self._lm_box.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._lm_host)
        cv.addWidget(scroll, 1)
        v.addWidget(cbox, 1)

        # Marker style toggles.
        sbox = QGroupBox("マーカー表示")
        sv = QVBoxLayout(sbox)
        self._style_cbs: dict[str, QCheckBox] = {}
        for key, text, on in (("guide", "ガイド線", True),
                              ("point", "点", True),
                              ("cutline", "断面線", False),
                              ("arrow", "方向矢印", False)):
            cb = QCheckBox(text)
            cb.setChecked(on)
            cb.toggled.connect(self._apply_style)
            sv.addWidget(cb)
            self._style_cbs[key] = cb
        v.addWidget(sbox)
        return panel

    def _load_series(self, pane_specs) -> None:
        specs = [(s, p, f) for (s, p, f) in (pane_specs or [])
                 if s is not None][:6]
        cols = _grid_dims(len(specs))
        for i, (se, pidx, start) in enumerate(specs):
            pane = _CoregPane(i, self)
            self.panes.append(pane)
            self._grid.addWidget(pane.frame, i // cols, i % cols)
            pane.load(se, pidx, start)
            if not pane.is_ivus:
                self._wire_angio(pane)
                self._guides[i] = {"vertices": [], "curve": [], "frame": None}
                self._add_guide_row(pane)
        # Pick the first IVUS pane as the active master.
        for pane in self.panes:
            if pane.is_ivus:
                pane.master_radio.setChecked(True)
                break
        self._apply_style()
        self._refresh_list()

    def _wire_angio(self, pane: _CoregPane) -> None:
        c = pane.canvas
        c.set_coreg_visible(True)
        c.coreg_guide_changed.connect(
            lambda verts, pi=pane.index: self._on_guide_changed(pi, verts))
        c.coreg_guide_finished.connect(
            lambda pi=pane.index: self._on_guide_finished(pi))
        c.coreg_slider_moved.connect(
            lambda s, pi=pane.index: self._on_slider_moved(pi, s))
        c.coreg_anchor_clicked.connect(
            lambda idx, pi=pane.index: self._on_anchor_clicked(pi, idx))
        c.coreg_marker_dragged.connect(
            lambda s, pi=pane.index: self._on_marker_dragged(pi, s))

    def _add_guide_row(self, pane: _CoregPane) -> None:
        row = QHBoxLayout()
        # Number by angio order (#1, #2 …) so it matches the per-guide ✓
        # marks shown on each CoReg-point row. This pane is already appended.
        angio_no = sum(1 for p in self.panes if not p.is_ivus)
        mod = pane.series.modality.value if pane.series else "?"
        lbl = QLabel(f"#{angio_no} {mod}")
        trace = QPushButton("トレース")
        trace.setCheckable(True)
        trace.toggled.connect(
            lambda on, pi=pane.index: self._toggle_trace(pi, on))
        clr = QPushButton("クリア")
        clr.clicked.connect(lambda _=False, pi=pane.index: self._clear_guide(pi))
        row.addWidget(lbl)
        row.addStretch(1)
        row.addWidget(trace)
        row.addWidget(clr)
        self._guide_box.addLayout(row)
        pane._trace_btn = trace          # keep for programmatic untoggle

    # ------------------------------------------------- guide editing
    def _toggle_trace(self, pane_idx: int, on: bool) -> None:
        pane = self.panes[pane_idx]
        if on:
            # Only one pane traces at a time.
            if self._tracing >= 0 and self._tracing != pane_idx:
                self.panes[self._tracing]._trace_btn.setChecked(False)
            self._tracing = pane_idx
            # If a guide already exists, jump back to the frame it was drawn
            # on so the edit happens on the same image.
            g = self._guides.get(pane_idx, {})
            if g.get("vertices") and g.get("frame") is not None:
                pane.show_frame(int(g["frame"]))
            pane.canvas.set_coreg_edit(True)
            pane.canvas.set_coreg_mode("trace")
            self._status.setText(
                "クリックで頂点追加、ダブルクリックで確定。頂点は赤くなったら"
                "ドラッグで移動／右クリック→点の削除。線上で右クリック→点の追加。")
        else:
            if self._tracing == pane_idx:
                self._tracing = -1
            pane.canvas.set_coreg_mode("")
            pane.canvas.set_coreg_edit(False)
            self._status.setText("")
        self._update_coreg_controls_enabled()

    def _on_guide_changed(self, pane_idx: int, vertices) -> None:
        verts = [tuple(v) for v in vertices]
        curve = coreg.smooth_curve(verts)
        # Remember the angio frame the trace is drawn on, so re-entering
        # trace later jumps back to the same image.
        self._guides[pane_idx] = {
            "vertices": verts, "curve": curve,
            "frame": self.panes[pane_idx].cur,
        }
        self.panes[pane_idx].canvas.set_coreg_guide(verts, curve)
        self._drive_markers()

    def _on_guide_finished(self, pane_idx: int) -> None:
        btn = getattr(self.panes[pane_idx], "_trace_btn", None)
        if btn is not None:
            btn.setChecked(False)            # triggers _toggle_trace(off)

    def _clear_guide(self, pane_idx: int) -> None:
        self._guides[pane_idx] = {"vertices": [], "curve": [], "frame": None}
        self.panes[pane_idx].canvas.set_coreg_guide([], [])
        self.panes[pane_idx].canvas.set_coreg_marker(None)
        self.panes[pane_idx].canvas.set_coreg_anchors([])

    # ------------------------------------------------- CoReg placement
    def _style_blue(self, btn, on: bool) -> None:
        btn.setStyleSheet(
            "QPushButton { background:#2d7ff0; color:white; font-weight:bold; }"
            if on else "")

    def _on_mode_button(self) -> None:
        """Top toggle: white → enter create/modify (blue); blue → commit."""
        if self._placing:
            self._commit_placing()
        else:
            self._start_placing()

    def _start_placing(self) -> None:
        if self._active_ivus < 0:
            QMessageBox.information(self, "CoReg", "IVUSがありません。")
            return
        # XA is optional: with no angio panes this is a pure multi-IVUS sync
        # point. (An untraced XA is handled at commit — the marker just has
        # no guide, or the user turns on XA除外.)
        has_angio = any(not p.is_ivus for p in self.panes)
        self._placing = True
        modifying = 0 <= self._selected_lm < len(self._landmarks)
        # Modify: seed from the point's existing angio positions. New: empty.
        self._pending_fracs = (
            dict(self._landmarks[self._selected_lm]["fracs"])
            if modifying else {})
        # XA-exclude toggle: only meaningful when angio panes exist; default
        # on when re-editing a point that has no XA position.
        self._exclude_xa_cb.setChecked(
            modifying and not self._landmarks[self._selected_lm]["fracs"])
        self._exclude_xa_cb.setVisible(has_angio)
        for p in self.panes:
            if p.is_ivus:
                # Frame(s) were chosen before entering — lock the seekbars.
                p.slider.setEnabled(False)
                p.include_cb.setVisible(not modifying)
            else:
                p.canvas.set_coreg_edit(True)
                p.canvas.set_coreg_mode("anchor")
                # Keep the 按分 (interpolated) green point as the starting
                # position for a NEW point so it doesn't vanish — adjust from
                # there. Modify: start from the point's existing position.
                s = self._pending_fracs.get(p.index)
                if s is None and not modifying:
                    s = self._interp_frac(p.index)
                    if s is not None:
                        self._pending_fracs[p.index] = s
                curve = self._guides.get(p.index, {}).get("curve", [])
                p.canvas.set_coreg_marker(
                    coreg.point_at_fraction(curve, s)
                    if (s is not None and curve) else None)
        self._style_blue(self._add_btn, True)
        self._cancel_btn.setVisible(True)
        self._sort_btn.setVisible(False)
        self._clear_all_btn.setVisible(False)
        self._update_coreg_controls_enabled()
        if not has_angio:
            self._status.setText(
                "各IVUSを目印フレーム・回転へ合わせ、もう一度ボタンで確定"
                "（IVUS同期点）。")
        elif modifying:
            self._status.setText(
                f"CoReg-{self._selected_lm + 1} を修正中: アンギオのガイド上の"
                "丸を目的位置へ移動 → もう一度ボタンで確定。")
        else:
            self._status.setText(
                "新規CoReg点: 同定できるアンギオでガイド上の丸を目的位置へ"
                "移動 → もう一度ボタンで確定。")

    def _on_slider_moved(self, pane_idx: int, s: float) -> None:
        # Provisional anchor for this angio while placing a CoReg point.
        if self._placing:
            self._pending_fracs[pane_idx] = float(s)

    def _commit_placing(self) -> None:
        if not self._placing:
            return
        has_angio = any(not p.is_ivus for p in self.panes)
        exclude_xa = self._exclude_xa_cb.isChecked() or not has_angio
        if not exclude_xa and not self._pending_fracs:
            QMessageBox.information(
                self, "CoReg",
                "少なくとも1つのアンギオで丸を配置してください"
                "（XAで決められない場合は『XA除外』をON）。")
            return
        fracs = {} if exclude_xa else dict(self._pending_fracs)
        if 0 <= self._selected_lm < len(self._landmarks):
            lm = self._landmarks[self._selected_lm]
            lm["fracs"] = fracs
            lm["rots"] = {i: self.panes[i].canvas.free_rotation()
                          for i in lm["frames"]}
        else:
            frames = {p.index: p.cur for p in self.panes
                      if p.is_ivus and p.include_cb.isChecked()}
            if not frames:
                QMessageBox.information(
                    self, "CoReg", "含めるIVUSを1つ以上選んでください。")
                return
            rots = {i: self.panes[i].canvas.free_rotation() for i in frames}
            self._landmarks.append(
                {"frames": frames, "rots": rots, "fracs": fracs})
        self._end_placing()
        self._refresh_list()
        self._drive_markers()

    def _cancel_placing(self) -> None:
        self._end_placing()
        self._refresh_list()
        self._drive_markers()

    def _end_placing(self) -> None:
        self._placing = False
        self._pending_fracs = {}
        self._style_blue(self._add_btn, False)
        self._cancel_btn.setVisible(False)
        self._exclude_xa_cb.setVisible(False)
        self._sort_btn.setVisible(True)
        self._clear_all_btn.setVisible(True)
        self._status.setText("")
        for p in self.panes:
            if p.is_ivus:
                p.slider.setEnabled(p.total > 1)
                p.include_cb.setVisible(False)
            else:
                p.canvas.set_coreg_mode("")
                p.canvas.set_coreg_edit(False)
        self._update_coreg_controls_enabled()

    # ------------------------------------------------- landmark list
    def _landmark_tip(self, lm) -> str:
        ivus_idx = [p.index for p in self.panes if p.is_ivus]
        xa_idx = [p.index for p in self.panes if not p.is_ivus]
        fr = " ".join(
            f"I{i + 1}:{lm['frames'][idx] + 1}" if idx in lm["frames"]
            else f"I{i + 1}:–" for i, idx in enumerate(ivus_idx))
        xa = " ".join(
            f"A{i + 1}:{'✓' if idx in lm['fracs'] else '–'}"
            for i, idx in enumerate(xa_idx))
        return f"{fr}   {xa}"

    def _landmark_info(self, lm) -> str:
        """Two-line summary shown right of the CoReg button: line 1 = each
        IVUS's frame/angle (all IVUS, like MultiSync); line 2 = a ✓ / · per
        angio guide (#1, #2 …) showing on which traces this point exists.
        Two lines so up to 3 IVUS × 3 XA still fit at the button font size."""
        ivus = [p for p in self.panes if p.is_ivus]
        xa = [p for p in self.panes if not p.is_ivus]
        toks = []
        for p in ivus:
            if p.index in lm["frames"]:
                fr = lm["frames"][p.index] + 1
                rot = lm.get("rots", {}).get(p.index)
                toks.append(f"{fr}/{round(rot)}°" if rot is not None
                            else str(fr))
            else:
                toks.append("–")
        guides = " ".join(
            f"#{k + 1}{'✓' if p.index in lm['fracs'] else '·'}"
            for k, p in enumerate(xa))
        return f"F {'  '.join(toks)}\n{guides}"

    def _refresh_list(self) -> None:
        while self._lm_box.count():
            it = self._lm_box.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        for n, lm in enumerate(self._landmarks, 1):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(3)
            x = QPushButton("✕")          # delete, moved to the left
            x.setFixedWidth(26)
            x.setToolTip("このCoReg点を削除")
            x.clicked.connect(lambda _c, i=n - 1: self._delete_landmark(i))
            b = QPushButton(lm.get("name") or f"CoReg-{n}")   # jump / rename
            b.setFixedWidth(78)
            b.setToolTip(self._landmark_tip(lm) + "\n(右クリックで名称変更)")
            b.clicked.connect(lambda _c, i=n - 1: self._goto_landmark(i))
            b.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            b.customContextMenuRequested.connect(
                lambda _pos, i=n - 1: self._rename_landmark(i))
            self._style_blue(b, (n - 1) == self._selected_lm)
            info = QLabel(self._landmark_info(lm))   # black, button-size text
            info.setStyleSheet("QLabel { color:black; }")
            h.addWidget(x)
            h.addWidget(b)
            h.addWidget(info, 1)
            self._lm_box.addWidget(row)

    def _goto_landmark(self, idx: int) -> None:
        """Clicking CoReg-N selects it (blue) and jumps every IVUS that
        recorded this landmark to its frame; the active IVUS then drives the
        angio markers onto it. Ignored mid-placement."""
        if self._placing or not (0 <= idx < len(self._landmarks)):
            return
        self._selected_lm = idx
        lm = self._landmarks[idx]
        self._syncing = True
        try:
            for pane in self.panes:
                if pane.is_ivus and pane.index in lm["frames"]:
                    pane.show_frame(int(lm["frames"][pane.index]))
        finally:
            self._syncing = False
        for pane in self.panes:
            self._apply_rotation(pane)
        self._drive_markers()
        self._refresh_list()

    def _rename_landmark(self, idx: int) -> None:
        """Right-click a CoReg button → give it a short name (≤8 chars) like
        R1os / LCX / LAD / D9a instead of CoReg-N. Blank clears it."""
        if not (0 <= idx < len(self._landmarks)):
            return
        cur = self._landmarks[idx].get("name", "")
        text, ok = QInputDialog.getText(
            self, "名称変更", "名称（8文字まで）:", text=cur)
        if ok:
            self._landmarks[idx]["name"] = text.strip()[:8]
            self._refresh_list()

    def _delete_landmark(self, idx: int) -> None:
        if not (0 <= idx < len(self._landmarks)):
            return
        del self._landmarks[idx]
        if self._selected_lm == idx:
            self._selected_lm = -1
        elif self._selected_lm > idx:
            self._selected_lm -= 1
        self._refresh_list()
        self._drive_markers()

    def _sort_landmarks(self) -> None:
        """Order CoReg points along the active IVUS's pull-back (by its
        frame); points without an active-IVUS frame sort by their smallest
        frame and fall after."""
        R = self._active_ivus

        def key(lm):
            if R in lm["frames"]:
                return (0, lm["frames"][R])
            return (1, min(lm["frames"].values()) if lm["frames"] else 0)

        self._landmarks.sort(key=key)
        self._selected_lm = -1
        self._refresh_list()

    def _clear_all_landmarks(self) -> None:
        if not self._landmarks:
            return
        if QMessageBox.question(
                self, "CoReg", "すべてのCoReg点を削除しますか？"
        ) == QMessageBox.StandardButton.Yes:
            self._landmarks.clear()
            self._selected_lm = -1
            self._refresh_list()
            self._drive_markers()

    def _on_ivus_grab(self) -> None:
        # User grabbed an IVUS seekbar (not during placement) → they are
        # positioning for a NEW point, so drop the modify selection.
        if self._placing:
            return
        if self._selected_lm != -1:
            self._selected_lm = -1
            self._refresh_list()

    def _update_coreg_controls_enabled(self) -> None:
        """Trace and CoReg-point editing are mutually exclusive: while a
        guide trace is active the CoReg-point controls are disabled. During
        placement the mode (確定) button stays enabled; sort / clear are
        hidden then, and the landmark rows are left enabled but their clicks
        are ignored (see _goto_landmark)."""
        tracing = self._tracing >= 0
        self._add_btn.setEnabled(not tracing)
        for b in (self._sort_btn, self._clear_all_btn):
            b.setEnabled(not tracing and not self._placing)
        self._lm_host.setEnabled(not tracing)

    # ------------------------------------------------- driving
    def _set_active_ivus(self, pane_idx: int) -> None:
        self._active_ivus = pane_idx
        self._drive_markers()

    def _on_ivus_scrub(self, pane_idx: int) -> None:
        """Scrubbing the MASTER IVUS moves the others in lockstep (once ≥2
        CoReg points exist) via the frames they share on the landmarks, then
        drives the angio markers. A non-master IVUS scrubs independently —
        it moves only itself and does not touch the markers."""
        if self._placing or self._syncing:
            return
        # Every scrubbed IVUS (master or not) updates its own 按分 rotation.
        self._apply_rotation(self.panes[pane_idx])
        if pane_idx != self._active_ivus:
            return
        self._syncing = True
        try:
            self._sync_ivus_from(pane_idx)
        finally:
            self._syncing = False
        for p in self.panes:
            self._apply_rotation(p)
        self._drive_markers()

    def _apply_rotation(self, pane) -> None:
        """Set *pane*'s IVUS display rotation to the angle interpolated from
        the CoReg landmarks at its current frame (按分). No-op when no
        rotation has been recorded (leaves the user's manual rotation)."""
        if not pane.is_ivus:
            return
        pairs = [(lm["frames"][pane.index], lm["rots"][pane.index])
                 for lm in self._landmarks
                 if pane.index in lm.get("frames", {})
                 and pane.index in lm.get("rots", {})]
        ang = coreg.map_rotation(pairs, pane.cur)
        if ang is not None:
            pane.canvas.set_free_rotation(ang)

    def _sync_ivus_from(self, src_idx: int) -> None:
        src = self.panes[src_idx]
        if not src.is_ivus:
            return
        for q in self.panes:
            if q is src or not q.is_ivus:
                continue
            pairs = [(lm["frames"][src_idx], lm["frames"][q.index])
                     for lm in self._landmarks
                     if src_idx in lm["frames"] and q.index in lm["frames"]]
            f = coreg.map_between(pairs, src.cur)
            if f is not None:
                q.show_frame(max(0, min(int(round(f)), q.total - 1)))

    def _interp_frac(self, pane_idx: int):
        """The 按分 (interpolated) guide fraction for angio *pane_idx* at the
        active IVUS's current frame, or None (too few anchors). Used to seed
        a new CoReg point's marker so it starts on the interpolated spot."""
        R = self._active_ivus
        if R < 0:
            return None
        anchors = [(lm["frames"][R], lm["fracs"][pane_idx])
                   for lm in self._landmarks
                   if R in lm["frames"] and pane_idx in lm["fracs"]]
        return coreg.map_frame_to_fraction(anchors, self.panes[R].cur)

    def _drive_markers(self) -> None:
        """For every angio pane, map the active IVUS's current frame to a
        point on that view's guide (from the landmarks common to the pair)
        and push the marker + the anchor foot-points to its canvas."""
        if self._active_ivus < 0 or self._placing:
            return
        R = self._active_ivus
        fR = self.panes[R].cur
        for pane in self.panes:
            if pane.is_ivus:
                continue
            curve = self._guides.get(pane.index, {}).get("curve", [])
            # Anchor foot-points on this guide (for the amber dots).
            feet = [coreg.point_at_fraction(curve, lm["fracs"][pane.index])
                    for lm in self._landmarks
                    if pane.index in lm["fracs"] and curve]
            pane.canvas.set_coreg_anchors([f for f in feet if f is not None])
            anchors = [(lm["frames"][R], lm["fracs"][pane.index])
                       for lm in self._landmarks
                       if R in lm["frames"] and pane.index in lm["fracs"]]
            s = coreg.map_frame_to_fraction(anchors, fR)
            pane.canvas.set_coreg_marker(
                None if (s is None or not curve)
                else coreg.point_at_fraction(curve, s))

    def _on_anchor_clicked(self, pane_idx: int, idx: int) -> None:
        """Review: clicking this view's idx-th amber anchor jumps the active
        IVUS to that landmark's frame, so the green marker lands on the dot."""
        if self._active_ivus < 0:
            return
        R = self._active_ivus
        lms = [lm for lm in self._landmarks if pane_idx in lm["fracs"]]
        if 0 <= idx < len(lms) and R in lms[idx]["frames"]:
            self.panes[R].slider.setValue(int(lms[idx]["frames"][R]))

    def _on_marker_dragged(self, pane_idx: int, s: float) -> None:
        """Review: dragging the green marker maps the guide fraction back to
        an IVUS frame and scrubs the active IVUS (reverse driving)."""
        if self._active_ivus < 0 or self._placing:
            return
        R = self._active_ivus
        anchors = [(lm["frames"][R], lm["fracs"][pane_idx])
                   for lm in self._landmarks
                   if R in lm["frames"] and pane_idx in lm["fracs"]]
        f = coreg.map_fraction_to_frame(anchors, s)
        if f is None:
            return
        total = self.panes[R].total
        self.panes[R].slider.setValue(max(0, min(int(round(f)), total - 1)))

    def _apply_style(self, *_a) -> None:
        style = {k: cb.isChecked() for k, cb in self._style_cbs.items()}
        for pane in self.panes:
            if not pane.is_ivus:
                pane.canvas.set_coreg_style(style)
