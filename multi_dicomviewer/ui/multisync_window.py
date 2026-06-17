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
from PyQt6.QtCore import QPoint, QRect, QRectF, QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter
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
    QLayout,
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


class _FlowLayout(QLayout):
    """Minimal wrapping horizontal layout: lays children left-to-right
    and overflows onto a new line when the available width runs out.
    Used by the Sync Point list so all sync points stay visible without
    a vertical scroll bar when there are many of them."""

    def __init__(self, parent=None, hspacing: int = 6, vspacing: int = 4):
        super().__init__(parent)
        self._h = hspacing
        self._v = vspacing
        self._items: list = []

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        m = self.contentsMargins()
        effective = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = effective.x()
        y = effective.y()
        line_h = 0
        for item in self._items:
            sh = item.sizeHint()
            next_x = x + sh.width() + self._h
            if next_x - self._h > effective.right() and line_h > 0:
                x = effective.x()
                y = y + line_h + self._v
                next_x = x + sh.width() + self._h
                line_h = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), sh))
            x = next_x
            line_h = max(line_h, sh.height())
        return y + line_h - rect.y() + m.bottom()


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


class _SliderMarks(QWidget):
    """Thin strip drawn directly above a QSlider that paints a small
    down-triangle (▽) at each marked frame, horizontally aligned with the
    slider's handle position for that value. Used to flag, on every slot's
    seek bar, the frames pinned by sync points."""

    def __init__(self, slider: QSlider, parent=None):
        super().__init__(parent)
        self._slider = slider
        self._marks: list[int] = []
        self.setFixedHeight(9)

    def set_marks(self, marks) -> None:
        new = sorted({int(m) for m in marks})
        if new == self._marks:
            return
        self._marks = new
        self.update()

    def paintEvent(self, _e):
        if not self._marks:
            return
        from PyQt6.QtWidgets import QStyle, QStyleOptionSlider
        from PyQt6.QtGui import QPolygon

        sl = self._slider
        mn, mx = sl.minimum(), sl.maximum()
        if mx <= mn:
            return
        opt = QStyleOptionSlider()
        sl.initStyleOption(opt)
        style = sl.style()
        groove = style.subControlRect(
            QStyle.ComplexControl.CC_Slider, opt,
            QStyle.SubControl.SC_SliderGroove, sl,
        )
        handle = style.subControlRect(
            QStyle.ComplexControl.CC_Slider, opt,
            QStyle.SubControl.SC_SliderHandle, sl,
        )
        span = groove.width() - handle.width()
        # The marks strip and its slider share the same x / width (stacked
        # in one column), so the slider's groove x maps directly onto this
        # widget's coordinates.
        base = groove.x() + handle.width() / 2.0
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor("#ffd000"))
        p.setPen(Qt.PenStyle.NoPen)
        h = self.height()
        for m in self._marks:
            pos = QStyle.sliderPositionFromValue(mn, mx, int(m), span)
            cx = base + pos
            tri = QPolygon([
                QPoint(int(round(cx - 4)), 0),
                QPoint(int(round(cx + 4)), 0),
                QPoint(int(round(cx)), h - 1),
            ])
            p.drawPolygon(tri)


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

        # Top of the slot: just the IVUS-series picker. The Master
        # radio used to live here but moved to the per-slot nav row
        # below so all the playhead controls are grouped together.
        self.combo = QComboBox()
        self.combo.setToolTip("Choose the IVUS series for this slot")
        col.addWidget(self.combo)

        self.canvas = _SyncCanvas()
        self.canvas.rotated.connect(self._on_rotated)
        col.addWidget(self.canvas, 1)

        srow = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        # Down-triangle markers sit in their own thin strip stacked
        # directly above the slider so they line up with the handle's
        # travel; the strip + slider together take the row's stretch and
        # the frame label sits beside them.
        self.marks = _SliderMarks(self.slider)
        slider_col = QVBoxLayout()
        slider_col.setContentsMargins(0, 0, 0, 0)
        slider_col.setSpacing(0)
        slider_col.addWidget(self.marks)
        slider_col.addWidget(self.slider)
        self.lbl = QLabel("—")
        self.lbl.setMinimumWidth(86)
        srow.addLayout(slider_col, 1)
        srow.addWidget(self.lbl)
        col.addLayout(srow)

        # Per-slot navigation row: [Prev] [Next] [Master radio].
        # When Sync ON, scrubbing or pressing Prev/Next on the master
        # slot drives the other slots via the sync mapping; on a
        # non-master slot the controls only move that slot.
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.setToolTip("Previous frame")
        self.btn_prev.clicked.connect(self._on_prev_frame)
        nav.addWidget(self.btn_prev)
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.setToolTip("Next frame")
        self.btn_next.clicked.connect(self._on_next_frame)
        nav.addWidget(self.btn_next)
        nav.addStretch(1)
        self.master_radio = QRadioButton("Master")
        self.master_radio.setToolTip(
            "Make this the master — when Sync ON, only the master's "
            "controls drive the other slots via the sync mapping"
        )
        nav.addWidget(self.master_radio)
        col.addLayout(nav)
        self._tint(False)

    def _tint(self, is_master: bool) -> None:
        # Same 3px width whether master or not, so switching the master
        # never reflows the image / slider (the old 1px → 3px swap made
        # the non-master tiles visibly shrink). The master frame is always
        # the same cyan whichever slot it is (not the slot's own colour),
        # and non-master frames are a near-white light grey. The per-slot
        # _SLOT_COLORS are still used for the sync-point row labels.
        self.frame.setStyleSheet(
            "QFrame { border:%s; }" %
            ("3px solid #33e6ff" if is_master else "3px solid #dcdcdc")
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

    def _on_prev_frame(self) -> None:
        if self.plane is None or self.total < 1:
            return
        self.owner._slot_scrubbed(self.index, max(0, self.cur - 1))

    def _on_next_frame(self) -> None:
        if self.plane is None or self.total < 1:
            return
        self.owner._slot_scrubbed(
            self.index, min(self.total - 1, self.cur + 1)
        )

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
        #: each sync point: {"frames":[f|None]*4, "rots":[deg]*4}.
        #: A None frame on a *loaded* slot is shown as a grey provisional
        #: value computed from the slot's other sync points; Edit + ○
        #: commits it (and the others) to a real value (black).
        self.sync_points: list[dict] = []
        self._master = 0
        self._sync_on = True
        #: Sync row the user last acted on (Jump / Edit / ○) — its row is
        #: tinted so it's clear which point is being worked on.
        self._active_point: int | None = None
        #: Sync point currently in Edit mode, or None. Informational only
        #: (drives the status hint / highlight) — it does NOT change how
        #: scrubbing behaves; that depends solely on the Sync ON/OFF state.
        self._editing: int | None = None
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

        # --- layout 1×2 / 2×2 toggle + Sync ON ---
        # Everything else (play / nav / speed / loop) moved into the
        # per-slot rows so each slot has its own ◀ Prev / Next ▶ /
        # Master controls. This bar only sets the grid shape and the
        # global Sync-link toggle.
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
        self._sync_btn = QPushButton("🔗 Sync ON")
        self._sync_btn.setCheckable(True)
        self._sync_btn.setChecked(True)
        self._sync_btn.toggled.connect(self._on_sync_toggle)
        lay_row.addWidget(self._sync_btn)
        lay_row.addStretch(1)
        root.addLayout(lay_row)

        root.addWidget(self._build_sync_editor())

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
        # Showing / hiding slots changes which pull-backs feed the synced
        # span, so recompute the global timeline.
        self._refresh_global_seek()

    # ============================================== sync toggle
    def _on_sync_toggle(self, on: bool) -> None:
        self._sync_on = on
        self._sync_btn.setText("🔗 Sync ON" if on else "🔗 Sync OFF")
        self._refresh_global_seek()
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
        srt = _tinted("Sort Sync…", "#e8daef")           # light purple
        srt.setToolTip("Sort Sync-Points by frame number")
        srt.clicked.connect(self._sort_sync)
        top.addWidget(srt)
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

        # Global synced timeline, full width of the Sync Points box.
        # It runs over *virtual master frames* spanning the union of every
        # loaded slot mapped into master-frame coordinates, so scrubbing it
        # shows the synchronised result from the most distal frame of any
        # pull-back to the most proximal — not just the master's own range.
        # Out-of-master-range positions freeze the master at its first /
        # last frame while the followers keep moving (and vice versa).
        grow = QHBoxLayout()
        gcap = QLabel("Synced:")
        gcap.setStyleSheet("font-weight:bold;")
        grow.addWidget(gcap)
        self._global_slider = QSlider(Qt.Orientation.Horizontal)
        self._global_slider.setEnabled(False)
        self._global_slider.setToolTip(
            "Synchronised timeline across every pull-back "
            "(most distal → most proximal frame)"
        )
        self._global_slider.valueChanged.connect(self._on_global_seek)
        self._global_lbl = QLabel("—")
        self._global_lbl.setMinimumWidth(110)
        grow.addWidget(self._global_slider, 1)
        grow.addWidget(self._global_lbl)
        outer.addLayout(grow)
        # Current virtual-frame span [lo, hi] the global slider covers, and
        # a guard so global-driven slot moves don't echo back as a scrub.
        self._global_lo = 0
        self._global_hi = 0
        self._global_echo = False

        # Wrapping list: sync rows pack left-to-right and flow onto
        # additional lines as the window widens / shrinks, so as many
        # sync points as possible stay visible at once instead of being
        # hidden below a vertical scroll bar.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(80)
        scroll.setMaximumHeight(260)
        self._sync_rows_host = QWidget()
        self._sync_rows = _FlowLayout(self._sync_rows_host,
                                       hspacing=6, vspacing=4)
        self._sync_rows_host.setLayout(self._sync_rows)
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
        # Capture EVERY loaded slot, not just the ones the current layout
        # happens to show — otherwise a 1×2 layout over 3–4 loaded runs
        # would silently create partial (None) sync points.
        for i in range(_N_SLOTS):
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
                "Two or more IVUS data are needed for Sync"
            )
            return
        cand = self.sync_points + [{"frames": frames, "rots": rots}]
        if multisync.has_conflict(cand, self._master, _N_SLOTS):
            self._sync_status.setStyleSheet(
                "color:#c0392b;font-weight:bold;"
            )
            self._sync_status.setText(
                "Inconsistent: Frames are out of order. Check and adjust "
                "Sync settings."
            )
            return
        self.sync_points.append({"frames": frames, "rots": rots})
        self._rebuild_sync_editor()
        self._sync_status.setStyleSheet("color:#1e8449;font-weight:bold;")
        self._sync_status.setText(
            "Sync-%d added: %s" % (
                len(self.sync_points),
                "  ".join(f"S{i + 1} F{frames[i] + 1}" for i in loaded),
            )
        )

    def _sort_sync(self) -> None:
        """Reorder the sync points by frame, smallest → largest. Sorted on
        the master's frame (None → first set frame); since a consistent
        set is monotonic across every slot, this orders them by every
        slot's frame at once. Renumbers Sync-1, Sync-2, …"""
        if len(self.sync_points) < 2:
            return

        def _key(sp):
            f = sp["frames"][self._master]
            if f is None:
                f = next((x for x in sp["frames"] if x is not None), 0)
            return f

        self.sync_points.sort(key=_key)
        # Indices changed, so any selection / edit target is stale.
        self._active_point = None
        self._editing = None
        self._sync_status.setStyleSheet("color:#1e8449;font-weight:bold;")
        self._sync_status.setText("Sorted Sync points by frame number")
        self._rebuild_sync_editor()

    def _rebuild_sync_editor(self) -> None:
        while self._sync_rows.count():
            it = self._sync_rows.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        self._sync_row_frames = []
        for pi, sp in enumerate(self.sync_points):
            row = self._sync_row(pi, sp)
            self._sync_row_frames.append(row)
            self._sync_rows.addWidget(row)
        self._refresh_sync_marks()
        self._refresh_slot_marks()
        self._refresh_global_seek()

    def _apply_row_highlight(self) -> None:
        """Tint only the active row (Edit/Jump selection) light red; the
        rest stay default. Done without rebuilding so it is safe to call
        on every scrub / Synced-bar move without disturbing the global
        slider's value."""
        for pi, row in enumerate(getattr(self, "_sync_row_frames", [])):
            row.setStyleSheet(
                "QFrame { background:#f7d4d4; }"
                if pi == self._active_point else ""
            )

    def _active_matches(self) -> bool:
        """True iff every loaded slot's current frame equals the active
        sync point's committed frame for that slot (provisional / unset
        slots are ignored)."""
        pi = self._active_point
        if pi is None or not (0 <= pi < len(self.sync_points)):
            return False
        sp = self.sync_points[pi]
        for slot in range(_N_SLOTS):
            sl = self.slots[slot]
            f = sp["frames"][slot]
            if sl.plane is None or f is None:
                continue
            if sl.cur != f:
                return False
        return True

    def _update_active_highlight(self) -> None:
        """Drop the red selection the moment any image / the Synced bar
        moves even one frame off the selected sync point."""
        if self._active_point is not None and not self._active_matches():
            self._active_point = None
            self._apply_row_highlight()

    def _refresh_slot_marks(self) -> None:
        """Update each slot's down-triangle (▽) markers to the frames that
        slot is pinned at by the current sync points."""
        for sl in self.slots:
            if sl.plane is None:
                sl.marks.set_marks([])
                continue
            marks = [sp["frames"][sl.index] for sp in self.sync_points
                     if sp["frames"][sl.index] is not None]
            sl.marks.set_marks(marks)

    # ============================================== global synced timeline
    def _virtual_range(self) -> tuple[int, int] | None:
        """[lo, hi] in master-frame coordinates spanning every loaded
        slot — the master's own [0, total-1] plus wherever the sync
        mapping places each follower's first / last frame. Returns None
        when the master has no pull-back loaded."""
        m = self.slots[self._master]
        if m.plane is None or m.total < 1:
            return None
        v_min = 0.0
        v_max = float(m.total - 1)
        # Every loaded slot feeds the synced span (the global bar drives
        # them all), not just the ones the current layout shows.
        for k in range(_N_SLOTS):
            sl = self.slots[k]
            if sl is m or sl.plane is None or sl.total < 1:
                continue
            v_first = multisync.map_frame(
                self.sync_points, sl.index, self._master, 0,
                sl.fps, m.fps,
            )
            v_last = multisync.map_frame(
                self.sync_points, sl.index, self._master, sl.total - 1,
                sl.fps, m.fps,
            )
            v_min = min(v_min, v_first, v_last)
            v_max = max(v_max, v_first, v_last)
        return int(math.floor(v_min)), int(math.ceil(v_max))

    def _refresh_global_seek(self) -> None:
        rng = self._virtual_range()
        if rng is None:
            self._global_lo = self._global_hi = 0
            self._global_slider.blockSignals(True)
            self._global_slider.setRange(0, 0)
            self._global_slider.setEnabled(False)
            self._global_slider.blockSignals(False)
            self._global_lbl.setText("—")
            return
        lo, hi = rng
        self._global_lo, self._global_hi = lo, hi
        cur = self.slots[self._master].cur
        self._global_slider.blockSignals(True)
        self._global_slider.setRange(lo, hi)
        self._global_slider.setValue(max(lo, min(cur, hi)))
        # The Synced bar is the synchronised-view control in its own right
        # and always drives every image together — independent of the
        # Sync ON/OFF toggle (which only governs the Master button / per-
        # slot sliders). So it is enabled whenever the span is non-empty.
        self._global_slider.setEnabled(hi > lo)
        self._global_slider.blockSignals(False)
        self._update_global_lbl(self._global_slider.value())

    def _update_global_lbl(self, fm: int) -> None:
        span = self._global_hi - self._global_lo + 1
        if span <= 0:
            self._global_lbl.setText("—")
            return
        self._global_lbl.setText(f"{fm - self._global_lo + 1} / {span}")

    def _set_global_value(self, fm: int) -> None:
        """Move the global slider to virtual frame *fm* without re-driving
        the slots (used to keep it in step with a master scrub)."""
        fm = max(self._global_lo, min(int(fm), self._global_hi))
        self._global_slider.blockSignals(True)
        self._global_slider.setValue(fm)
        self._global_slider.blockSignals(False)
        self._update_global_lbl(fm)

    def _on_global_seek(self, value: int) -> None:
        self._drive_to_virtual(int(value))

    def _drive_to_virtual(self, fm: int) -> None:
        """Drive the master (clamped) and every follower from virtual
        master frame *fm*, allowing the extended pre/post range where one
        video keeps playing while the other is frozen at an end frame."""
        m = self.slots[self._master]
        if m.plane is None:
            return
        self._global_echo = True
        try:
            m.show_frame(max(0, min(fm, m.total - 1)), m.rotation)
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
                sl.show_frame(
                    max(0, min(int(round(fb)), sl.total - 1)), rot
                )
        finally:
            self._global_echo = False
        self._update_global_lbl(fm)
        self._update_active_highlight()

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

    # ---- provisional (grey) values for unset cells ----
    def _provisional(self, pi: int, slot: int):
        """Computed (frame, rotation) for a *loaded* slot whose frame is
        None at sync point *pi*, derived from that slot's OTHER sync points
        via the same mapping engine. The reference run is the master (or,
        if the master has no frame at this point, the first slot that
        does). Returns None when nothing usable can anchor the estimate."""
        sl = self.slots[slot]
        if sl.plane is None:
            return None
        sp = self.sync_points[pi]
        ref = self._master
        if sp["frames"][ref] is None:
            ref = next(
                (j for j in range(_N_SLOTS)
                 if j != slot and sp["frames"][j] is not None
                 and self.slots[j].plane is not None),
                None,
            )
        if ref is None or sp["frames"][ref] is None:
            return None
        ref_frame = sp["frames"][ref]
        fb = multisync.map_frame(
            self.sync_points, ref, slot, ref_frame,
            self.slots[ref].fps, sl.fps,
        )
        rot = multisync.map_rotation(self.sync_points, ref, slot, ref_frame)
        return max(0, min(int(round(fb)), sl.total - 1)), rot

    def _sync_row(self, pi: int, sp: dict) -> QWidget:
        # Wrapped by _FlowLayout, so the row needs to report its
        # natural width — NO addStretch at the end (that would make
        # each row claim full available width and force one-per-line).
        row = QFrame()
        row.setFrameShape(QFrame.Shape.StyledPanel)
        # Tint the row the user is currently working on (Jump / Edit / ○).
        if pi == self._active_point:
            row.setStyleSheet("QFrame { background:#f7d4d4; }")
        h = QHBoxLayout(row)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(3)
        # Sync-N label styled as a light-blue chip (boxed like the
        # buttons). Same blue for every point.
        lbl = QLabel(f"Sync-{pi + 1}")
        lbl.setStyleSheet(
            "QLabel { background:#d6eaf8; border:1px solid #8aa9c0;"
            " border-radius:4px; padding:2px 6px;"
            " font-weight:bold; color:#1c1c1c; }"
        )
        h.addWidget(lbl)
        # Each S-cell is a boxed chip on a white background; only the text
        # colour conveys committed (black) vs provisional (grey) vs unset.
        cell_box = (
            " background:#ffffff; border:1px solid #c0c0c0;"
            " border-radius:3px; padding:1px 5px;"
        )
        for slot in range(_N_SLOTS):
            if self.slots[slot].plane is None:
                # Run not loaded in this slot — nothing to pin.
                if slot >= self._layout_count:
                    continue
                cell = QLabel(f"S{slot + 1}: —")
                cell.setStyleSheet("QLabel { color:#999999;" + cell_box + "}")
                h.addWidget(cell)
                continue
            f = sp["frames"][slot]
            if f is not None:
                # Committed: black + bold.
                r = sp["rots"][slot] or 0.0
                cell = QLabel(f"S{slot + 1}: F{f + 1} A{int(round(r))}")
                cell.setStyleSheet(
                    "QLabel { color:#000000; font-weight:bold;"
                    + cell_box + "}"
                )
            else:
                # Provisional: grey computed estimate (or — if no anchor).
                prov = self._provisional(pi, slot)
                if prov is None:
                    cell = QLabel(f"S{slot + 1}: —")
                    cell.setStyleSheet(
                        "QLabel { color:#999999;" + cell_box + "}"
                    )
                else:
                    pf, pr = prov
                    cell = QLabel(f"S{slot + 1}: F{pf + 1} A{int(round(pr))}")
                    cell.setStyleSheet(
                        "QLabel { color:#8a8a8a;" + cell_box + "}"
                    )
                    cell.setToolTip("Provisional value — Edit → ○ to confirm")
            h.addWidget(cell)
        # Tighter horizontal padding so Jump / Edit aren't so wide
        # (roughly half the default side spacing).
        nav_css = "QPushButton { padding:3px 8px; }"
        jump = QPushButton("Jump")
        jump.setStyleSheet(nav_css)
        jump.setToolTip("Move all images to this Sync point")
        jump.clicked.connect(lambda _c, p=pi: self._jump_to_point(p))
        h.addWidget(jump)
        edit = QPushButton("Edit")
        edit.setStyleSheet(nav_css)
        edit.setToolTip("Move to this Sync point and fine-tune → ○ to confirm")
        edit.clicked.connect(lambda _c, p=pi: self._edit_point(p))
        h.addWidget(edit)
        ok = QPushButton("○")
        ok.setFixedWidth(28)
        ok.setToolTip("Confirm each image's current frame as this Sync point")
        ok.clicked.connect(lambda _c, p=pi: self._confirm_point(p))
        h.addWidget(ok)
        dele = QPushButton("✕")
        dele.setFixedWidth(28)
        dele.clicked.connect(lambda _c, p=pi: self._del_point(p))
        h.addWidget(dele)
        return row

    def _del_point(self, pi: int) -> None:
        if 0 <= pi < len(self.sync_points):
            del self.sync_points[pi]
            if self._editing == pi:
                self._editing = None
            self._active_point = None
            self._sync_status.setText("")
            self._rebuild_sync_editor()
            if self._sync_on:
                self._drive_followers()

    def _goto_point(self, pi: int) -> None:
        """Move every loaded slot to this sync point — to its committed
        frame, or to the grey provisional estimate when not yet set."""
        sp = self.sync_points[pi]
        for slot in range(_N_SLOTS):
            sl = self.slots[slot]
            if sl.plane is None:
                continue
            f = sp["frames"][slot]
            if f is not None:
                sl.show_frame(f, sp["rots"][slot])
            else:
                prov = self._provisional(pi, slot)
                if prov is not None:
                    sl.show_frame(prov[0], prov[1])

    def _jump_to_point(self, pi: int) -> None:
        # Jump = view only: not an edit, so leave edit mode.
        self._editing = None
        self._active_point = pi
        self._goto_point(pi)
        self._rebuild_sync_editor()

    def _edit_point(self, pi: int) -> None:
        # Edit = jump there and flag the point as being edited (highlight +
        # status hint). To adjust images independently, turn Sync OFF; with
        # Sync ON the master scrub keeps driving the followers as usual.
        self._editing = pi
        self._active_point = pi
        self._goto_point(pi)
        self._sync_status.setStyleSheet("color:#c0392b;font-weight:bold;")
        self._sync_status.setText(
            f"Editing Sync-{pi + 1} — fine-tune each image (turn Sync OFF "
            "for independent adjustment), then ○ to confirm."
        )
        self._rebuild_sync_editor()

    def _confirm_point(self, pi: int) -> None:
        """Capture every loaded slot's current frame + rotation as this
        sync point (committing any grey provisional cells), rejected if it
        would fold the mapping back on itself."""
        frames = list(self.sync_points[pi]["frames"])
        rots = list(self.sync_points[pi]["rots"])
        loaded = []
        for slot in range(_N_SLOTS):
            sl = self.slots[slot]
            if sl.plane is not None:
                frames[slot] = sl.cur
                rots[slot] = sl.rotation
                loaded.append(slot)
        if len(loaded) < 2:
            self._sync_status.setStyleSheet("color:#c0392b;font-weight:bold;")
            self._sync_status.setText(
                "Two or more IVUS data is needed to confirm"
            )
            return
        cand = [
            {"frames": list(p["frames"]), "rots": list(p["rots"])}
            for p in self.sync_points
        ]
        cand[pi] = {"frames": frames, "rots": rots}
        if multisync.has_conflict(cand, self._master, _N_SLOTS):
            self._sync_status.setStyleSheet("color:#c0392b;font-weight:bold;")
            self._sync_status.setText(
                "Inconsistent: Frames are out of order. Check and adjust "
                "Sync settings."
            )
            return
        self.sync_points[pi]["frames"] = frames
        self.sync_points[pi]["rots"] = rots
        self._editing = None
        self._active_point = pi
        self._sync_status.setStyleSheet("color:#1e8449;font-weight:bold;")
        self._sync_status.setText(f"Sync-{pi + 1} confirmed.")
        self._rebuild_sync_editor()
        if self._sync_on:
            self._drive_followers()

    # ============================================== slot events
    def _slot_series_changed(self, sl: _Slot) -> None:
        # The user picked a different (or no) series from this slot's
        # combo — the live link to whichever pane originally seeded it
        # no longer makes sense, so drop it.
        if self._link_viewers[sl.index] is not None:
            self._disconnect_link_for(sl.index)
        sl.load(sl.combo.currentData())
        # A new (or cleared) pull-back changes the slider range and the
        # synced span, so refresh this slot's ▽ marks and the global bar.
        self._refresh_slot_marks()
        self._refresh_global_seek()

    def _set_master(self, idx: int) -> None:
        # Changing the master only changes which run is the reference /
        # driver — it must NOT move any image. (It used to call
        # _drive_followers(), which re-mapped every follower onto the new
        # master's current frame and nudged carefully-positioned images
        # out from under the user mid-setup / mid-Edit.) Re-mapping happens
        # on the next master scrub, as usual.
        self._master = idx
        self._refresh_master_tint()
        # Provisional (grey) estimates are computed relative to the master,
        # so re-render the rows (and the global span) when it changes.
        self._rebuild_sync_editor()

    def _refresh_master_tint(self) -> None:
        for s in self.slots:
            s._tint(s.index == self._master)

    def _slot_scrubbed(self, idx: int, value: int) -> None:
        sl = self.slots[idx]
        if sl.plane is None:
            return
        sl.show_frame(value)
        # Behaviour depends ONLY on the Sync toggle (not on whether a point
        # is being edited): Sync ON → the master drives every follower;
        # Sync OFF → the master loses its role and each image (master
        # included) moves on its own. Scrubbing a follower always moves
        # just that slot, which is how a point is fine-tuned under Sync OFF.
        if idx == self._master and self._sync_on:
            self._drive_followers()
        self._update_active_highlight()

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
        # Mirror the master's frame onto the global synced slider, unless
        # the global slider is itself the driver (then it owns its value).
        if not self._global_echo:
            self._set_global_value(fm)

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
    @staticmethod
    def _qimage_to_rgb(img: QImage) -> np.ndarray:
        """QImage → owned, contiguous (H, W, 3) uint8 RGB for the MP4
        writer, honouring the row stride (RGB888 rows are 4-byte aligned).

        The returned array MUST be a real copy, not a view: np.frombuffer
        aliases the QImage's pixel buffer, and once this local QImage is
        garbage-collected that memory is freed. Returning a view left the
        encoder reading freed memory — intermittently corrupting frames or
        breaking the ffmpeg pipe with '[Errno 22] Invalid argument'."""
        img = img.convertToFormat(QImage.Format.Format_RGB888)
        w, h = img.width(), img.height()
        bpl = img.bytesPerLine()
        ptr = img.constBits()
        ptr.setsize(h * bpl)
        arr = np.frombuffer(ptr, np.uint8).reshape(h, bpl)
        return np.array(arr[:, :w * 3].reshape(h, w, 3))   # copy, not view

    def _compose_frame_qt(self, states: list, W: int, H: int,
                          cell: int, cols: int) -> np.ndarray:
        """Render one composite frame with QPainter — the SAME optimised
        smooth scale + rotate the live _SyncCanvas uses — so the export
        pixel-matches the on-screen view and is ~50× faster than the old
        numpy bilinear path. *states* is a list of (slot_index, _Slot,
        frame_idx, rotation_deg) for the loaded slots."""
        img = QImage(W, H, QImage.Format.Format_RGB888)
        img.fill(QColor("#000000"))
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        try:
            for (k, sl, fb, rot) in states:
                frame = sl.plane.frame(max(0, min(fb, sl.total - 1)))
                if frame is None:
                    continue
                qimg = to_qimage(_to_u8(frame))
                w, h = qimg.width(), qimg.height()
                if w <= 0 or h <= 0:
                    continue
                r, c = divmod(k, cols)
                tx, ty = c * cell, r * cell
                scale = min(cell / w, cell / h)
                dw, dh = w * scale, h * scale
                p.save()
                p.setClipRect(tx, ty, cell, cell)
                p.translate(tx + cell / 2.0, ty + cell / 2.0)
                p.rotate(rot)
                p.translate(-dw / 2.0, -dh / 2.0)
                p.drawImage(QRectF(0.0, 0.0, dw, dh), qimg)
                p.restore()
        finally:
            p.end()
        return self._qimage_to_rgb(img)

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
        # Defaults tuned for IVUS review composites: CRF 10 (near-lossless,
        # speckle preserved) at 15 fps. 40 Mbps is the fallback only if the
        # user switches to Bitrate mode. All overridable in the dialog.
        dlg_cfg = ExportDialog(
            "mp4", 1,
            default_fps=15.0,
            default_bitrate=40,
            default_crf=10,
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

        # Layout is whichever the user has selected on screen: 1×2 (count
        # 2) → cols=2,rows=1; 2×2 (count 4) → cols=2,rows=2. Hidden slots
        # render as black tiles so an unfilled 2×2 still keeps its shape
        # instead of collapsing to 1×N.
        cols = 2
        rows = (self._layout_count + cols - 1) // cols
        # Render each tile at (close to) the native source resolution so no
        # detail is thrown away the way the old fixed 480 px tile did —
        # clamped to [480, 1024] to keep the file / encode time bounded and
        # forced even for yuv420p.
        src_max = 0
        for k in range(self._layout_count):
            sl = self.slots[k]
            if sl.plane is None or sl.total < 1:
                continue
            fr = sl.plane.frame(max(0, min(sl.cur, sl.total - 1)))
            if fr is not None:
                src_max = max(src_max, fr.shape[0], fr.shape[1])
        cell = max(480, min(src_max or 480, 1024))
        cell -= cell % 2
        W, H = cell * cols, cell * rows

        # Synced range: span from the earliest virtual master frame at
        # which any slot's first frame appears, to the latest at which
        # any slot's last frame appears. The master is the natural
        # timeline so its [0, total-1] is always included; each follower
        # extends the range when the sync mapping places its 0th / last
        # frame outside the master's window. Out-of-master-range frames
        # render the master frozen at its first/last frame while the
        # follower keeps playing — and vice versa — so the export shows
        # everything either video has to offer in synced time.
        v_min = 0.0
        v_max = float(m.total - 1)
        for k in range(self._layout_count):
            sl = self.slots[k]
            if sl is m or sl.plane is None or sl.total < 1:
                continue
            v_first = multisync.map_frame(
                self.sync_points, sl.index, self._master, 0,
                sl.fps, m.fps,
            )
            v_last = multisync.map_frame(
                self.sync_points, sl.index, self._master, sl.total - 1,
                sl.fps, m.fps,
            )
            v_min = min(v_min, v_first, v_last)
            v_max = max(v_max, v_first, v_last)
        fm_start = int(math.floor(v_min))
        fm_end = int(math.ceil(v_max))
        total_export = fm_end - fm_start + 1

        dlg = QProgressDialog(
            "Exporting composite MP4…", "Cancel", 0, total_export, self
        )
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(0)

        # Stream each composite frame straight into the encoder instead of
        # buffering them all (a 4-up 1024² × ~1400-frame export would be
        # >4 GB in RAM). The encoder also writes via an ASCII temp file so
        # non-ASCII output paths (Japanese folders) work.
        from multi_dicomviewer.core import export as exporter
        try:
            stream = exporter.open_mp4_stream(
                path, fps=cfg.fps,
                bitrate_mbps=cfg.bitrate_mbps, crf=cfg.crf,
            )
        except Exception as e:
            dlg.close()
            QMessageBox.critical(self, "Export MP4", f"Encoding failed:\n{e}")
            return

        written = 0
        cancelled = False
        try:
            for i, fm in enumerate(range(fm_start, fm_end + 1)):
                if dlg.wasCanceled():
                    cancelled = True
                    break
                states: list = []
                for k in range(self._layout_count):
                    sl = self.slots[k]
                    if sl.plane is None:
                        continue
                    if sl is m:
                        # Master plays through its own timeline; clamping
                        # holds the first / last frame as a still while a
                        # follower is in its extended pre/post range. The
                        # master's rotation is whatever the user dragged it
                        # to in the live view (constant across all fm).
                        fb = max(0, min(fm, m.total - 1))
                        rot = m.rotation
                    else:
                        # Sync-map the follower; clamping holds its first /
                        # last frame as a still when the master is the one
                        # still playing past this follower's range.
                        fb_float = multisync.map_frame(
                            self.sync_points, self._master, sl.index, fm,
                            m.fps, sl.fps,
                        )
                        fb = max(0, min(int(round(fb_float)), sl.total - 1))
                        # Per-fm follower rotation = the same map_rotation
                        # the live _drive_followers() applies; fall back to
                        # the slot's drag rotation when sync points captured
                        # none so the export still keeps its orientation.
                        rot = multisync.map_rotation(
                            self.sync_points, self._master, sl.index, fm
                        )
                        if abs(rot) < 1e-3 and abs(sl.rotation) > 1e-3:
                            rot = sl.rotation
                    states.append((k, sl, fb, rot))
                stream.add(self._compose_frame_qt(states, W, H, cell, cols))
                written += 1
                if i % 8 == 0:
                    dlg.setValue(i)
                    QApplication.processEvents()
        except Exception as e:
            stream.abort()
            dlg.close()
            QMessageBox.critical(self, "Export MP4", f"Encoding failed:\n{e}")
            return
        dlg.close()

        if cancelled or written == 0:
            stream.abort()
            return
        try:
            stream.close()
        except Exception as e:
            QMessageBox.critical(self, "Export MP4", f"Encoding failed:\n{e}")
            return
        qual = f"CRF {cfg.crf}" if cfg.crf is not None \
            else f"{cfg.bitrate_mbps} Mbps"
        QMessageBox.information(
            self, "Export MP4",
            f"Saved {written} frames @ {cfg.fps:.1f} fps, "
            f"{qual}:\n{path}"
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
        self._update_active_highlight()

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
        for i in range(_N_SLOTS):
            self._disconnect_link_for(i)
        for s in self.slots:
            s.stop_prefetch()
        super().closeEvent(e)
