"""MultiSync IVUS window — up to 4 IVUS pull-backs shown together and
played back synchronised through user-set sync points.

Ported from the standalone MultiSync-Viewer.html, generalised from its
fixed A/B pair to up to four slots in a 1×2 / 2×2 grid. One slot is the
user-chosen *master*; the others follow it through the piecewise-linear
mapping in :mod:`multi_dicomviewer.core.multisync`. Sync points are set
per slot by frame + rotation; an edit that would make a slot's frames
non-monotonic with the master's is rejected with a "矛盾しています"
warning so the mapping can never fold back on itself.
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QRect, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.core import dicom_io, multisync
from multi_dicomviewer.core.study_model import Series
from multi_dicomviewer.viewers.image_canvas import to_qimage

_N_SLOTS = 4
_SLOT_COLORS = ["#33e6ff", "#7cfc00", "#ffd400", "#ff80c0"]


def _to_u8(arr: np.ndarray) -> np.ndarray:
    """Display-ready uint8 (IVUS is normally 8-bit MONOCHROME2 already)."""
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    hi = float(a.max()) if a.size else 0.0
    if hi > 0:
        a = a / hi * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


class _SyncCanvas(QWidget):
    """Fit-to-window IVUS frame display with drag-to-rotate."""

    rotated = pyqtSignal(float)            # emits the new absolute rotation

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(170, 170)
        self._qimg = None
        self._rotation = 0.0
        self._drag_ang = None
        # Yellow border drawn on the image area when this slot is at a
        # frame pinned by a sync point.
        self._at_sync = False

    def set_at_sync(self, on: bool) -> None:
        if self._at_sync == bool(on):
            return
        self._at_sync = bool(on)
        self.update()

    def set_frame(self, frame8) -> None:
        self._qimg = to_qimage(_to_u8(frame8)) if frame8 is not None else None
        self.update()

    def set_rotation(self, deg: float) -> None:
        self._rotation = float(deg)
        self.update()

    # ---- drag to rotate ----
    def _angle(self, pos) -> float:
        cx, cy = self.width() / 2.0, self.height() / 2.0
        return math.degrees(math.atan2(pos.y() - cy, pos.x() - cx))

    def mousePressEvent(self, e):
        self._drag_ang = self._angle(e.position())

    def mouseMoveEvent(self, e):
        if self._drag_ang is None:
            return
        now = self._angle(e.position())
        self._rotation = (self._rotation + (now - self._drag_ang)) % 360.0
        self._drag_ang = now
        self.update()
        self.rotated.emit(self._rotation)

    def mouseReleaseEvent(self, _e):
        self._drag_ang = None

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0a0a0a"))
        if self._qimg is not None:
            w, h = self._qimg.width(), self._qimg.height()
            if w > 0 and h > 0:
                scale = min(self.width() / w, self.height() / h)
                dw, dh = w * scale, h * scale
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                p.save()
                p.translate(self.width() / 2.0, self.height() / 2.0)
                p.rotate(self._rotation)
                p.translate(-dw / 2.0, -dh / 2.0)
                p.drawImage(QRect(0, 0, int(dw), int(dh)), self._qimg)
                p.restore()
        # Yellow frame around the image area when this slot's current
        # frame is one of the pinned sync-point frames.
        if self._at_sync:
            from PyQt6.QtGui import QPen
            p.setPen(QPen(QColor("#ffd000"), 4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            r = self.rect().adjusted(2, 2, -2, -2)
            p.drawRect(r)


class _Prefetch(QThread):
    """Warms a plane's frames off the UI thread for smooth playback."""

    def __init__(self, plane, parent=None):
        super().__init__(parent)
        self._plane = plane
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        dicom_io.prefetch_planes(
            [self._plane], lambda: self._stop, lambda: True
        )


class _Slot:
    """One IVUS pull-back slot: its widgets + loaded pull-back state."""

    def __init__(self, index: int, owner: "MultiSyncWindow"):
        self.index = index
        self.owner = owner
        self.series: Series | None = None
        self.plane = None
        self.total = 0
        self.fps = 30.0
        self.cur = 0
        self.rotation = 0.0
        self._prefetch: _Prefetch | None = None
        # Set by MultiSyncWindow before triggering combo.setCurrentIndex(),
        # so the slot opens on the same frame the main pane is showing
        # instead of resetting to 0. Consumed (and cleared) by load().
        self._pending_start_frame = 0

        self.frame = QFrame()
        self.frame.setFrameShape(QFrame.Shape.Box)
        col = QVBoxLayout(self.frame)
        col.setContentsMargins(3, 3, 3, 3)
        col.setSpacing(2)

        head = QHBoxLayout()
        self.master_radio = QRadioButton(f"Slot {index + 1}")
        self.master_radio.setToolTip("Make this the playback master")
        self.combo = QComboBox()
        self.combo.setToolTip("Choose the IVUS series for this slot")
        head.addWidget(self.master_radio)
        head.addWidget(self.combo, 1)
        col.addLayout(head)

        self.canvas = _SyncCanvas()
        self.canvas.rotated.connect(self._on_rotated)
        col.addWidget(self.canvas, 1)

        srow = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.lbl = QLabel("—")
        self.lbl.setMinimumWidth(86)
        srow.addWidget(self.slider, 1)
        srow.addWidget(self.lbl)
        col.addLayout(srow)
        self._tint(False)

    def _tint(self, is_master: bool) -> None:
        c = _SLOT_COLORS[self.index]
        self.frame.setStyleSheet(
            "QFrame { border:%s; }" %
            (f"3px solid {c}" if is_master else "1px solid #444")
        )

    # ---- loading ----
    def load(self, series: Series | None) -> None:
        if self._prefetch is not None:
            self._prefetch.stop()
            self._prefetch.wait(2000)
            self._prefetch = None
        self.series = series
        if series is None:
            self.plane = None
            self.total = 0
            self.slider.setEnabled(False)
            self.lbl.setText("—")
            self.canvas.set_frame(None)
            return
        loaded = dicom_io.load_xa(series)
        self.plane = (loaded.xa_planes or [None])[0]
        self.total = self.plane.total_frames if self.plane else 0
        self.fps = float(loaded.cine_fps or 30.0)
        start = max(0, min(int(self._pending_start_frame), self.total - 1)) \
            if self.total > 0 else 0
        self._pending_start_frame = 0
        self.cur = start
        self.rotation = 0.0
        self.slider.blockSignals(True)
        self.slider.setEnabled(self.total > 1)
        self.slider.setRange(0, max(0, self.total - 1))
        self.slider.setValue(start)
        self.slider.blockSignals(False)
        self.show_frame(start)
        if self.plane is not None and self.total > 1:
            self._prefetch = _Prefetch(self.plane, self.owner)
            self._prefetch.start()

    def show_frame(self, idx: int, rotation: float | None = None) -> None:
        if self.plane is None:
            return
        idx = max(0, min(int(idx), self.total - 1))
        changed = (idx != self.cur)
        self.cur = idx
        if rotation is not None:
            self.rotation = rotation
        self.canvas.set_frame(self.plane.frame(idx))
        self.canvas.set_rotation(self.rotation)
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.lbl.setText(f"F {idx + 1} / {self.total}")
        # Push the new frame back into the linked main-window pane, so
        # MultiSync scrub / playback / sync-mapping all keep the pane in
        # lockstep with the slot. The owner's echo guard prevents this
        # from bouncing back via the pane's frame_changed signal.
        if changed:
            self.owner._slot_to_pane(self.index, idx)
        # Yellow border whenever this slot's current frame is pinned by
        # any sync point — the visual cue the user asked for.
        matched = any(
            sp["frames"][self.index] == self.cur
            for sp in self.owner.sync_points
        )
        self.canvas.set_at_sync(matched)

    # ---- slot events ----
    def _on_slider(self, value: int) -> None:
        self.owner._slot_scrubbed(self.index, int(value))

    def _on_rotated(self, deg: float) -> None:
        self.rotation = deg

    def stop_prefetch(self) -> None:
        if self._prefetch is not None:
            self._prefetch.stop()
            self._prefetch.wait(2000)
            self._prefetch = None


class MultiSyncWindow(QMainWindow):
    """Standalone MultiSync IVUS window."""

    def __init__(self, ivus_series: list[Series], layout_count: int = 4,
                 preset: list | None = None,
                 preset_frames: list | None = None,
                 preset_viewers: list | None = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("MultiSync — IVUS synchronised viewer")
        self.resize(1180, 880)

        self._series = list(ivus_series)
        #: each sync point: {"frames":[f|None]*4, "rots":[deg]*4}
        self.sync_points: list[dict] = []
        self._master = 0
        self._sync_on = True
        self._playing = False
        self._speed = 1.0
        self._loop = True
        self._accum = 0.0
        # Live link with the main window's panes (per slot, parallel to
        # self.slots). Each entry is the source viewer object or None.
        # While linked, slot frame changes drive the pane and vice versa;
        # _link_echo_guard breaks the would-be A→B→A bounce.
        self._link_viewers: list = [None] * _N_SLOTS
        self._link_echo_guard = False

        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        # --- slot grid ---
        self.slots = [_Slot(i, self) for i in range(_N_SLOTS)]
        self._master_group = QButtonGroup(self)
        for s in self.slots:
            self._master_group.addButton(s.master_radio, s.index)
            s.combo.addItem("(none)", None)
            for se in self._series:
                s.combo.addItem(se.label, se)
            s.combo.currentIndexChanged.connect(
                lambda _i, sl=s: self._slot_series_changed(sl)
            )
        self.slots[0].master_radio.setChecked(True)
        self._master_group.idClicked.connect(self._set_master)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._grid_host, 1)

        # --- layout 1×2 / 2×2 toggle ---
        lay_row = QHBoxLayout()
        lay_row.addWidget(QLabel("Layout:"))
        self._btn_1x2 = QPushButton("1×2")
        self._btn_2x2 = QPushButton("2×2")
        for b in (self._btn_1x2, self._btn_2x2):
            b.setCheckable(True)
        self._btn_1x2.clicked.connect(lambda: self._apply_layout(2))
        self._btn_2x2.clicked.connect(lambda: self._apply_layout(4))
        lay_row.addWidget(self._btn_1x2)
        lay_row.addWidget(self._btn_2x2)
        lay_row.addStretch(1)
        root.addLayout(lay_row)

        root.addWidget(self._build_control_bar())
        root.addWidget(self._build_sync_editor())

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)

        self._apply_layout(2 if layout_count == 2 else 4)
        self._refresh_master_tint()
        self._rebuild_sync_editor()

        # Inherit the series shown in the main window's panes — plus the
        # exact frame each pane was on, so MultiSync opens displaying the
        # same images the user is already looking at.
        preset_frames = list(preset_frames or [])
        preset_viewers = list(preset_viewers or [])
        if preset:
            for i, se in enumerate(preset[:_N_SLOTS]):
                if se is None:
                    continue
                idx = self.slots[i].combo.findData(se)
                if idx >= 0:
                    # Mark the slot's pending start frame BEFORE the
                    # currentIndexChanged signal fires; _Slot.load reads
                    # this so it opens on the pane's current frame rather
                    # than frame 0.
                    f0 = preset_frames[i] if i < len(preset_frames) else 0
                    self.slots[i]._pending_start_frame = int(f0)
                    self.slots[i].combo.setCurrentIndex(idx)
                # Hold the source viewer for the bidirectional live link
                # set up below.
                if i < len(preset_viewers):
                    self._link_viewers[i] = preset_viewers[i]
        self._connect_live_link()

    # ============================================== layout
    def _apply_layout(self, count: int) -> None:
        self._layout_count = count
        cols = 2
        for s in self.slots:
            self._grid.removeWidget(s.frame)
            s.frame.setVisible(False)
        for i in range(count):
            r, c = divmod(i, cols)
            self._grid.addWidget(self.slots[i].frame, r, c)
            self.slots[i].frame.setVisible(True)
        for r in range(2):
            self._grid.setRowStretch(r, 1 if r < (count + 1) // cols else 0)
        for c in range(cols):
            self._grid.setColumnStretch(c, 1)
        self._btn_1x2.setChecked(count == 2)
        self._btn_2x2.setChecked(count == 4)
        # A hidden slot can't be master.
        if self._master >= count:
            self._set_master(0)
            self.slots[0].master_radio.setChecked(True)

    # ============================================== control bar
    def _build_control_bar(self) -> QWidget:
        bar = QFrame()
        bar.setFrameShape(QFrame.Shape.StyledPanel)
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 4, 6, 4)

        def _b(text, fn, tip=""):
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(fn)
            row.addWidget(btn)
            return btn

        _b("⏮", lambda: self._nav("first"), "First")
        _b("«", lambda: self._nav("-10"), "-10")
        _b("◀", lambda: self._nav("-1"), "-1")
        self._play_btn = _b("▶", self._toggle_play, "Play / Pause")
        _b("▶", lambda: self._nav("+1"), "+1")
        _b("»", lambda: self._nav("+10"), "+10")
        _b("⏭", lambda: self._nav("last"), "Last")

        row.addSpacing(12)
        row.addWidget(QLabel("Speed:"))
        self._speed_combo = QComboBox()
        for s in ("0.25", "0.5", "1", "2", "4"):
            self._speed_combo.addItem(f"{s}×", float(s))
        self._speed_combo.setCurrentText("1×")
        self._speed_combo.currentIndexChanged.connect(
            lambda _i: setattr(
                self, "_speed", self._speed_combo.currentData()
            )
        )
        row.addWidget(self._speed_combo)

        row.addSpacing(12)
        self._loop_btn = QPushButton("↺ Loop ON")
        self._loop_btn.setCheckable(True)
        self._loop_btn.setChecked(True)
        self._loop_btn.toggled.connect(self._on_loop)
        row.addWidget(self._loop_btn)

        self._sync_btn = QPushButton("🔗 Sync ON")
        self._sync_btn.setCheckable(True)
        self._sync_btn.setChecked(True)
        self._sync_btn.toggled.connect(self._on_sync_toggle)
        row.addWidget(self._sync_btn)

        row.addStretch(1)
        row.addWidget(QLabel("Master = the slot whose radio is checked"))
        return bar

    def _on_loop(self, on: bool) -> None:
        self._loop = on
        self._loop_btn.setText("↺ Loop ON" if on else "↺ Loop OFF")

    def _on_sync_toggle(self, on: bool) -> None:
        self._sync_on = on
        self._sync_btn.setText("🔗 Sync ON" if on else "🔗 Sync OFF")
        if on:
            self._drive_followers()

    # ============================================== sync editor
    def _build_sync_editor(self) -> QWidget:
        box = QGroupBox("Sync Points  (frame + angle)")
        outer = QVBoxLayout(box)
        top = QHBoxLayout()

        def _tinted(text: str, bg: str) -> QPushButton:
            """Bold button with a light tinted background so the toolbar
            row is unmistakable (plain push buttons disappeared into the
            dock chrome)."""
            b = QPushButton(text)
            b.setStyleSheet(
                "QPushButton{"
                f"background:{bg};color:#1c1c1c;font-weight:bold;"
                "border:1px solid #888;border-radius:5px;padding:5px 12px;"
                "}"
                "QPushButton:hover{background:#ffffff;}"
                "QPushButton:pressed{background:#cccccc;}"
            )
            return b

        add = _tinted("+ Add Sync Point", "#d4efdf")    # light green
        add.clicked.connect(self._add_sync_point)
        top.addWidget(add)
        save = _tinted("Save Sync…", "#fcf3cf")          # light yellow
        save.clicked.connect(self._save_sync)
        top.addWidget(save)
        load = _tinted("Load Sync…", "#d6eaf8")          # light blue
        load.clicked.connect(self._load_sync)
        top.addWidget(load)
        mp4 = _tinted("Export MP4…", "#fadbd8")          # light pink
        mp4.setToolTip("Render the synchronised composite to an MP4")
        mp4.clicked.connect(self._export_mp4)
        top.addWidget(mp4)
        self._sync_status = QLabel("")
        self._sync_status.setStyleSheet("color:#c0392b; font-weight:bold;")
        top.addWidget(self._sync_status, 1)
        outer.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(170)
        self._sync_rows_host = QWidget()
        self._sync_rows = QVBoxLayout(self._sync_rows_host)
        self._sync_rows.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._sync_rows_host)
        outer.addWidget(scroll)
        return box

    def _add_sync_point(self) -> None:
        """Capture the CURRENT frame + rotation of every loaded slot as
        one sync point — "the images shown right now are synchronised".
        Rejected (矛盾) if those frames would fold the mapping back on
        itself relative to the existing sync points."""
        frames: list = [None] * _N_SLOTS
        rots: list = [0.0] * _N_SLOTS
        loaded = []
        for i in range(self._layout_count):
            sl = self.slots[i]
            if sl.plane is not None:
                frames[i] = sl.cur
                rots[i] = sl.rotation
                loaded.append(i)
        if len(loaded) < 2:
            self._sync_status.setStyleSheet(
                "color:#c0392b;font-weight:bold;"
            )
            self._sync_status.setText(
                "Sync点には2つ以上のスロットにIVUSが必要です。"
            )
            return
        cand = self.sync_points + [{"frames": frames, "rots": rots}]
        if multisync.has_conflict(cand, self._master, _N_SLOTS):
            self._sync_status.setStyleSheet(
                "color:#c0392b;font-weight:bold;"
            )
            self._sync_status.setText(
                "矛盾しています — 現在の各スロットのフレームは既存の"
                "Sync点と前後関係が逆転します。フレームを調整してください。"
            )
            return
        self.sync_points.append({"frames": frames, "rots": rots})
        self._rebuild_sync_editor()
        self._sync_status.setStyleSheet("color:#1e8449;font-weight:bold;")
        self._sync_status.setText(
            "Sync-%d 追加: %s" % (
                len(self.sync_points),
                "  ".join(f"S{i + 1} F{frames[i] + 1}" for i in loaded),
            )
        )

    def _rebuild_sync_editor(self) -> None:
        while self._sync_rows.count():
            it = self._sync_rows.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        for pi, sp in enumerate(self.sync_points):
            self._sync_rows.addWidget(self._sync_row(pi, sp))
        self._refresh_sync_marks()

    def _refresh_sync_marks(self) -> None:
        """Re-evaluate the yellow image-area border for every slot —
        ON iff its current frame is pinned by some sync point."""
        for sl in self.slots:
            if sl.plane is None:
                sl.canvas.set_at_sync(False)
                continue
            matched = any(
                sp["frames"][sl.index] == sl.cur
                for sp in self.sync_points
            )
            sl.canvas.set_at_sync(matched)

    def _sync_row(self, pi: int, sp: dict) -> QWidget:
        row = QFrame()
        h = QHBoxLayout(row)
        h.setContentsMargins(4, 2, 4, 2)
        lbl = QLabel(f"Sync-{pi + 1}")
        lbl.setMinimumWidth(56)
        h.addWidget(lbl)
        for slot in range(self._layout_count):
            f = sp["frames"][slot]
            r = sp["rots"][slot] or 0.0
            if f is not None:
                # Frame + rotation; A0 when the slot wasn't rotated.
                text = f"S{slot + 1}: F{f + 1} A{int(round(r))}"
            else:
                text = f"S{slot + 1}: —"
            btn = QPushButton(text)
            btn.setStyleSheet(
                f"color:{_SLOT_COLORS[slot]};"
                + ("font-weight:bold;" if f is not None else "")
            )
            btn.setToolTip(
                "Set this slot's frame + rotation to its current view"
            )
            btn.clicked.connect(
                lambda _c, p=pi, s=slot: self._set_point_slot(p, s)
            )
            h.addWidget(btn)
        jump = QPushButton("Jump")
        jump.clicked.connect(lambda _c, p=pi: self._jump_to_point(p))
        h.addWidget(jump)
        dele = QPushButton("✕")
        dele.setFixedWidth(28)
        dele.clicked.connect(lambda _c, p=pi: self._del_point(p))
        h.addWidget(dele)
        h.addStretch(1)
        return row

    def _set_point_slot(self, pi: int, slot: int) -> None:
        sl = self.slots[slot]
        if sl.plane is None:
            self._sync_status.setText(
                f"Slot {slot + 1} has no IVUS loaded."
            )
            return
        value = sl.cur
        if multisync.would_conflict(
            self.sync_points, self._master, _N_SLOTS, pi, slot, value
        ):
            self._sync_status.setText(
                f"矛盾しています — Slot {slot + 1} のフレーム "
                f"{value + 1} は他のSync点と前後関係が逆転します。"
            )
            return
        self._sync_status.setText("")
        self.sync_points[pi]["frames"][slot] = value
        self.sync_points[pi]["rots"][slot] = sl.rotation
        self._rebuild_sync_editor()
        if self._sync_on:
            self._drive_followers()

    def _del_point(self, pi: int) -> None:
        if 0 <= pi < len(self.sync_points):
            del self.sync_points[pi]
            self._sync_status.setText("")
            self._rebuild_sync_editor()
            if self._sync_on:
                self._drive_followers()

    def _jump_to_point(self, pi: int) -> None:
        sp = self.sync_points[pi]
        for slot in range(self._layout_count):
            f = sp["frames"][slot]
            if f is not None and self.slots[slot].plane is not None:
                self.slots[slot].show_frame(f, sp["rots"][slot])

    # ============================================== slot events
    def _slot_series_changed(self, sl: _Slot) -> None:
        self._stop_play()
        # The user picked a different (or no) series from this slot's
        # combo — the live link to whichever pane originally seeded it
        # no longer makes sense, so drop it.
        if self._link_viewers[sl.index] is not None:
            self._disconnect_link_for(sl.index)
        sl.load(sl.combo.currentData())

    def _set_master(self, idx: int) -> None:
        self._master = idx
        self._refresh_master_tint()
        if self._sync_on:
            self._drive_followers()

    def _refresh_master_tint(self) -> None:
        for s in self.slots:
            s._tint(s.index == self._master)

    def _slot_scrubbed(self, idx: int, value: int) -> None:
        sl = self.slots[idx]
        if sl.plane is None:
            return
        sl.show_frame(value)
        # Scrubbing the master drives the followers (sync ON); scrubbing
        # a follower only moves that slot (needed to set sync points).
        if idx == self._master and self._sync_on:
            self._drive_followers()

    def _drive_followers(self) -> None:
        """Map the master's current frame onto every other loaded slot."""
        m = self.slots[self._master]
        if m.plane is None:
            return
        fm = m.cur
        for sl in self.slots:
            if sl is m or sl.plane is None:
                continue
            fb = multisync.map_frame(
                self.sync_points, self._master, sl.index, fm,
                m.fps, sl.fps,
            )
            rot = multisync.map_rotation(
                self.sync_points, self._master, sl.index, fm
            )
            sl.show_frame(int(round(fb)), rot)

    # ============================================== playback
    def _toggle_play(self) -> None:
        if self._playing:
            self._stop_play()
        else:
            m = self.slots[self._master]
            if m.plane is None or m.total < 2:
                return
            self._playing = True
            self._play_btn.setText("⏸")
            self._accum = 0.0
            self._last_ms = None
            self._timer.start(int(1000.0 / max(1.0, m.fps)))

    def _stop_play(self) -> None:
        self._playing = False
        self._play_btn.setText("▶")
        self._timer.stop()

    def _tick(self) -> None:
        m = self.slots[self._master]
        if m.plane is None:
            self._stop_play()
            return
        self._accum += self._speed
        adv = int(self._accum)
        if adv < 1:
            return
        self._accum -= adv
        nf = m.cur + adv
        if nf >= m.total:
            if self._loop:
                nf %= m.total
            else:
                nf = m.total - 1
                self._stop_play()
        m.show_frame(nf)
        if self._sync_on:
            self._drive_followers()

    def _nav(self, where: str) -> None:
        m = self.slots[self._master]
        if m.plane is None:
            return
        cur = m.cur
        nf = {
            "first": 0, "last": m.total - 1,
            "-1": cur - 1, "+1": cur + 1,
            "-10": cur - 10, "+10": cur + 10,
        }[where]
        m.show_frame(nf)
        if self._sync_on:
            self._drive_followers()

    # ============================================== sync file I/O
    def _save_sync(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Sync settings", "", "Sync settings (*.txt)"
        )
        if not path:
            return
        lines = ["# MultiSync settings v1",
                 f"master={self._master + 1}", "[Slots]"]
        for i, s in enumerate(self.slots):
            label = s.series.label if s.series is not None else ""
            lines.append(f"slot{i + 1}={label}")
            lines.append(f"fps{i + 1}={s.fps:.3f}")
        lines.append("[SyncPoints]")
        lines.append("# label\tframe1..4 (- = unset)\trot1..4 (deg)")
        for pi, sp in enumerate(self.sync_points):
            fr = "\t".join(str(f) if f is not None else "-"
                           for f in sp["frames"])
            ro = "\t".join(f"{r:.1f}" for r in sp["rots"])
            lines.append(f"Sync-{pi + 1}\t{fr}\t{ro}")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError as exc:
            QMessageBox.warning(self, "Save Sync", str(exc))
            return
        self._sync_status.setStyleSheet("color:#1e8449;font-weight:bold;")
        self._sync_status.setText("Saved.")

    def _load_sync(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Sync settings", "", "Sync settings (*.txt)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            QMessageBox.warning(self, "Load Sync", str(exc))
            return
        master = 0
        slot_labels: dict[int, str] = {}
        sps: list[dict] = []
        section = None
        for raw in text.splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("[") and s.endswith("]"):
                section = s[1:-1]
                continue
            if section is None and s.startswith("master="):
                master = int(s.split("=", 1)[1]) - 1
            elif section == "Slots" and s.startswith("slot"):
                k, v = s.split("=", 1)
                slot_labels[int(k[4:]) - 1] = v
            elif section == "SyncPoints":
                parts = raw.split("\t")
                if len(parts) >= 9:
                    frames = [None if p == "-" else int(p)
                              for p in parts[1:5]]
                    rots = [float(p) for p in parts[5:9]]
                    sps.append({"frames": frames, "rots": rots})
        # Match the saved slot series to this window's combo items.
        for i in range(_N_SLOTS):
            lbl = slot_labels.get(i, "")
            if not lbl:
                continue
            for ci in range(self.slots[i].combo.count()):
                se = self.slots[i].combo.itemData(ci)
                if se is not None and se.label == lbl:
                    self.slots[i].combo.setCurrentIndex(ci)
                    break
        self.sync_points = sps
        self._master = max(0, min(master, _N_SLOTS - 1))
        self.slots[self._master].master_radio.setChecked(True)
        self._refresh_master_tint()
        self._rebuild_sync_editor()
        self._sync_status.setStyleSheet("color:#1e8449;font-weight:bold;")
        self._sync_status.setText(
            f"Loaded {len(sps)} sync point(s)."
        )
        if self._sync_on:
            self._drive_followers()

    # ============================================== MP4 export
    def _render_cell(self, slot: _Slot, frame_idx: int,
                     rotation: float, size: int):
        """One slot rendered as an (size, size, 3) RGB uint8 array,
        fitted + rotated — the per-slot tile of the MP4 composite. RGB
        (not BGR) so it can be fed straight to imageio's libx264 writer.
        Fitting is done with numpy slicing so OpenCV isn't a dependency."""
        cell = np.zeros((size, size, 3), np.uint8)
        if slot.plane is None:
            return cell
        arr = _to_u8(slot.plane.frame(
            max(0, min(frame_idx, slot.total - 1))
        ))
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.shape[2] != 3:
            arr = arr[..., :3]
        h, w = arr.shape[:2]
        sc = min(size / w, size / h)
        dw, dh = max(1, int(w * sc)), max(1, int(h * sc))
        # Nearest-neighbour downscale: 480×480 cell vs ~512–1024 source is
        # close enough to a 1:1 mapping that an exact resampler isn't
        # worth a heavy dependency. (For sharper output, sample on a
        # regular grid.)
        ys = (np.linspace(0, h - 1, dh)).astype(np.int32)
        xs = (np.linspace(0, w - 1, dw)).astype(np.int32)
        small = arr[ys[:, None], xs[None, :]]
        ox, oy = (size - dw) // 2, (size - dh) // 2
        cell[oy:oy + dh, ox:ox + dw] = small
        if abs(rotation) > 1e-3:
            cell = self._rotate_rgb(cell, -rotation)
        return cell

    @staticmethod
    def _rotate_rgb(img: np.ndarray, deg: float) -> np.ndarray:
        """Affine rotate (H, W, 3) uint8 around the centre by *deg*. Uses
        nearest-neighbour sampling via numpy; output is the same shape as
        the input (black where pixels rotate off the canvas)."""
        h, w = img.shape[:2]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        a = math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        # Inverse-map each output pixel back to a source pixel.
        yy, xx = np.indices((h, w), dtype=np.float32)
        dx, dy = xx - cx, yy - cy
        sx = ca * dx + sa * dy + cx
        sy = -sa * dx + ca * dy + cy
        sxi = np.rint(sx).astype(np.int32)
        syi = np.rint(sy).astype(np.int32)
        ok = (sxi >= 0) & (sxi < w) & (syi >= 0) & (syi < h)
        out = np.zeros_like(img)
        out[ok] = img[np.clip(syi, 0, h - 1),
                      np.clip(sxi, 0, w - 1)][ok]
        return out

    def _export_mp4(self) -> None:
        m = self.slots[self._master]
        if m.plane is None or m.total < 1:
            QMessageBox.information(
                self, "Export MP4", "Load the master slot first."
            )
            return

        # Bitrate / FPS dialog — reuse the same dialog the series-browser
        # uses, with the per-file-name checkboxes hidden (the user already
        # picks the path in the Save dialog).
        from multi_dicomviewer.ui.export_dialog import ExportDialog
        dlg_cfg = ExportDialog(
            "mp4", 1,
            default_fps=float(m.fps) if m.fps else None,
            show_filename_fields=False,
            title_override="Export composite MP4",
            parent=self,
        )
        if dlg_cfg.exec() != dlg_cfg.DialogCode.Accepted:
            return
        cfg = dlg_cfg.result_settings()

        path, _ = QFileDialog.getSaveFileName(
            self, "Export composite MP4", "", "MP4 video (*.mp4)"
        )
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"

        self._stop_play()

        # Layout is whichever the user has selected on screen: 1×2 (count
        # 2) → cols=2,rows=1; 2×2 (count 4) → cols=2,rows=2. Hidden slots
        # render as black tiles so an unfilled 2×2 still keeps its shape
        # instead of collapsing to 1×N.
        cols = 2
        rows = (self._layout_count + cols - 1) // cols
        cell = 480
        W, H = cell * cols, cell * rows

        dlg = QProgressDialog(
            "Exporting composite MP4…", "Cancel", 0, m.total, self
        )
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(0)

        frames_rgb: list[np.ndarray] = []
        for fm in range(m.total):
            if dlg.wasCanceled():
                break
            canvas = np.zeros((H, W, 3), np.uint8)
            for k in range(self._layout_count):
                sl = self.slots[k]
                if sl.plane is None:
                    continue
                if sl is m:
                    fb = fm
                else:
                    fb = int(round(multisync.map_frame(
                        self.sync_points, self._master, sl.index, fm,
                        m.fps, sl.fps,
                    )))
                rot = multisync.map_rotation(
                    self.sync_points, self._master, sl.index, fm
                )
                tile = self._render_cell(sl, fb, rot, cell)
                r, c = divmod(k, cols)
                canvas[r * cell:(r + 1) * cell,
                       c * cell:(c + 1) * cell] = tile
            frames_rgb.append(canvas)
            if fm % 8 == 0:
                dlg.setValue(fm)
                QApplication.processEvents()
        dlg.close()

        if not frames_rgb:
            return
        try:
            from multi_dicomviewer.core import export as exporter
            exporter.write_mp4(
                path, frames_rgb,
                fps=cfg.fps,
                bitrate_mbps=cfg.bitrate_mbps,
            )
        except Exception as e:
            QMessageBox.critical(self, "Export MP4", f"Encoding failed:\n{e}")
            return
        QMessageBox.information(
            self, "Export MP4",
            f"Saved {len(frames_rgb)} frames @ {cfg.fps:.1f} fps, "
            f"{cfg.bitrate_mbps} Mbps:\n{path}"
        )

    # ============================================== live link with panes
    def _connect_live_link(self) -> None:
        """Subscribe to each linked pane's frame_changed so MultiSync
        mirrors scrubs / cine playback happening in the main window."""
        for i, v in enumerate(self._link_viewers):
            if v is None or not hasattr(v, "frame_changed"):
                continue
            v.frame_changed.connect(
                lambda idx, slot=i: self._pane_to_slot(slot, idx)
            )

    def _disconnect_link_for(self, slot_index: int) -> None:
        v = self._link_viewers[slot_index]
        if v is None:
            return
        try:
            v.frame_changed.disconnect()
        except (TypeError, RuntimeError):
            # No connections, or the viewer was already torn down — both
            # are benign; we just want the reference gone.
            pass
        self._link_viewers[slot_index] = None

    def _pane_to_slot(self, slot_index: int, idx: int) -> None:
        """Pane changed frame → mirror it on the slot.
        Guarded only around the immediate mirror so we don't push the
        same frame straight back to the pane. The guard is released
        BEFORE _drive_followers so each follower slot's resulting
        show_frame can legitimately push its own (sync-mapped) frame
        into the follower pane."""
        if self._link_echo_guard:
            return
        sl = self.slots[slot_index]
        if sl.plane is None or idx == sl.cur:
            return
        self._link_echo_guard = True
        try:
            sl.show_frame(idx)
        finally:
            self._link_echo_guard = False
        # The master moving must still drive the synced followers, just
        # like a user scrub on the master would.
        if slot_index == self._master and self._sync_on:
            self._drive_followers()

    def _slot_to_pane(self, slot_index: int, idx: int) -> None:
        """Slot changed frame → push it onto the linked pane's viewer.
        Guarded the same way as the reverse direction."""
        if self._link_echo_guard:
            return
        v = self._link_viewers[slot_index]
        if v is None or not hasattr(v, "goto_frame"):
            return
        self._link_echo_guard = True
        try:
            v.goto_frame(int(idx))
        finally:
            self._link_echo_guard = False

    def closeEvent(self, e):
        self._stop_play()
        for i in range(_N_SLOTS):
            self._disconnect_link_for(i)
        for s in self.slots:
            s.stop_prefetch()
        super().closeEvent(e)
