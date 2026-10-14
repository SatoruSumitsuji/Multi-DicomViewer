"""Virtual BiXA window — fuse an LCA cine and an RCA cine shot at the
(nearly) same C-arm projection into one bilateral-contrast cine.

Modelled on :mod:`multi_dicomviewer.ui.multisync_window`, but instead of a
grid of independent panes the star is a single **composite** view: the
two cines are min-blended (dark angiographic contrast from either side
survives) after the moving image (RCA) is registered onto the reference
(LCA) by a 2-landmark similarity transform. Temporal alignment reuses the
piecewise-linear frame mapping in :mod:`multi_dicomviewer.core.multisync`
(LCA = master, RCA = follower); the composite math lives in
:mod:`multi_dicomviewer.core.bixa`.
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.core import bixa, dicom_io, multisync
from multi_dicomviewer.core.study_model import Series
from multi_dicomviewer.viewers.image_canvas import to_qimage
from multi_dicomviewer.viewers.xa_viewer import apply_window

_N_SLOTS = 2                     # LCA (master, slot 0) + RCA (follower, slot 1)


# --------------------------------------------------------------- display helper
def _disp_u8(plane, idx: int, window, level) -> np.ndarray:
    """One plane frame as a display-ready 2-D uint8 grayscale image.

    Grayscale planes are window/level'd (the same ``apply_window`` the XA
    viewer uses); color planes are reduced to luminance so the min-fusion
    has a single channel to work on."""
    f = plane.frame(max(0, min(int(idx), plane.total_frames - 1)))
    if getattr(plane, "is_color", False):
        g = np.asarray(f)[..., :3].mean(axis=2)
        return np.clip(g, 0, 255).astype(np.uint8)
    return apply_window(f, float(window or 1.0), float(level or 0.0))


def _plane_view_angles(plane):
    """(PositionerPrimary, Secondary) of a plane's dataset, or None when
    either tag is missing — mirrors XAViewer._plane_angles."""
    ds = getattr(plane, "_ds", None)
    if ds is None:
        return None

    def _f(name):
        try:
            v = float(getattr(ds, name, float("nan")))
            return None if math.isnan(v) else v
        except (TypeError, ValueError):
            return None

    b = _f("PositionerPrimaryAngle")
    a = _f("PositionerSecondaryAngle")
    return (b, a) if (b is not None and a is not None) else None


def _fmt_view(view) -> str:
    """(primary, secondary) -> 'LAO30/CRA20' style label, or '角度不明'."""
    if view is None:
        return "角度不明"
    b, a = view
    prim = f"{'RAO' if b < 0 else 'LAO'}{abs(int(round(b)))}"
    sec = f"{'CAU' if a < 0 else 'CRA'}{abs(int(round(a)))}"
    return f"{prim}/{sec}"


# --------------------------------------------------------------------- canvas
class _BiXACanvas(QWidget):
    """Fit-to-window uint8 image display. In ``landmark`` mode it captures
    up to two click points (in image-pixel coordinates) and draws numbered
    markers; in ``composite`` mode it just shows the blended image."""

    landmarks_changed = pyqtSignal(object)        # emits list[(x, y)]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self._qimg = None
        self._img_wh = (0, 0)             # source image (w, h)
        self._mode = "composite"
        self._landmarks: list[tuple[float, float]] = []
        # Fit transform from the last paint, for click -> image mapping.
        self._scale = 1.0
        self._ox = 0.0
        self._oy = 0.0

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.update()

    def set_image(self, u8) -> None:
        if u8 is None:
            self._qimg = None
            self._img_wh = (0, 0)
        else:
            u8 = np.ascontiguousarray(u8)
            h, w = u8.shape[:2]
            self._img_wh = (w, h)
            self._qimg = to_qimage(u8)
        self.update()

    def set_landmarks(self, pts) -> None:
        self._landmarks = [(float(x), float(y)) for x, y in (pts or [])]
        self.update()

    def landmarks(self) -> list[tuple[float, float]]:
        return list(self._landmarks)

    def clear_landmarks(self) -> None:
        self._landmarks = []
        self.update()
        self.landmarks_changed.emit(self.landmarks())

    # ---- click to set landmarks ----
    def mousePressEvent(self, e):
        if self._mode != "landmark" or self._qimg is None:
            return
        w, h = self._img_wh
        ix = (e.position().x() - self._ox) / max(self._scale, 1e-9)
        iy = (e.position().y() - self._oy) / max(self._scale, 1e-9)
        if not (0 <= ix < w and 0 <= iy < h):
            return
        if len(self._landmarks) >= 2:
            self._landmarks = []          # third click restarts the pair
        self._landmarks.append((ix, iy))
        self.update()
        self.landmarks_changed.emit(self.landmarks())

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0a0a0a"))
        if self._qimg is None:
            return
        w, h = self._qimg.width(), self._qimg.height()
        if w <= 0 or h <= 0:
            return
        scale = min(self.width() / w, self.height() / h)
        dw, dh = w * scale, h * scale
        ox, oy = (self.width() - dw) / 2.0, (self.height() - dh) / 2.0
        self._scale, self._ox, self._oy = scale, ox, oy
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(QRect(int(ox), int(oy), int(dw), int(dh)), self._qimg)
        # numbered landmark markers
        if self._mode == "landmark":
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            f = QFont()
            f.setBold(True)
            p.setFont(f)
            for i, (lx, ly) in enumerate(self._landmarks):
                cx = ox + lx * scale
                cy = oy + ly * scale
                p.setPen(QPen(QColor("#ff3b3b"), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPoint(int(cx), int(cy)), 8, 8)
                p.drawLine(int(cx - 11), int(cy), int(cx + 11), int(cy))
                p.drawLine(int(cx), int(cy - 11), int(cx), int(cy + 11))
                p.setPen(QPen(QColor("#ffd000"), 2))
                p.drawText(int(cx + 10), int(cy - 8), str(i + 1))


# ------------------------------------------------------------------ prefetch
class _Prefetch(QThread):
    """Warms a plane's frames off the UI thread for smooth playback."""

    def __init__(self, plane, parent=None):
        super().__init__(parent)
        self._plane = plane
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        if self._plane is not None:
            dicom_io.prefetch_planes(
                [self._plane], lambda: self._stop, lambda: True
            )


# -------------------------------------------------------------------- window
class BiXAWindow(QMainWindow):
    """Virtual BiXA — bilateral-contrast composite of an LCA + RCA cine."""

    def __init__(self, xa_series: list[Series],
                 preset: list | None = None,
                 preset_frames: list | None = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Virtual BiXA — 両側造影コンポジット")
        self.resize(1080, 900)

        self._series = list(xa_series)
        # loaded planes + per-series display window/level + fps
        self._lca_plane = None
        self._rca_plane = None
        self._lca_wl = (1.0, 0.0)
        self._rca_wl = (1.0, 0.0)
        self._lca_fps = 30.0
        self._rca_fps = 30.0
        self._lca_pending = 0
        self._rca_pending = 0
        # play state
        self._fL = 0
        self._fR = 0
        self._sync_on = True
        # spatial registration (scale, rot_rad, tx, ty) or None until set
        self._transform = None
        # temporal sync points: {"frames":[fL,fR], "rots":[0,0]}
        self.sync_points: list[dict] = []
        self._prefetch_l: _Prefetch | None = None
        self._prefetch_r: _Prefetch | None = None

        from PyQt6.QtCore import QTimer
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)

        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        root.addLayout(self._build_picker_row())
        root.addWidget(self._build_view_stack(), 1)
        root.addLayout(self._build_transport())
        root.addWidget(self._build_sync_editor())

        # Inherit panes' series + frames.
        preset = list(preset or [])
        preset_frames = list(preset_frames or [])
        for combo, slot in ((self._lca_combo, 0), (self._rca_combo, 1)):
            se = preset[slot] if slot < len(preset) else None
            if se is not None:
                idx = combo.findData(se)
                if idx >= 0:
                    f0 = preset_frames[slot] if slot < len(preset_frames) else 0
                    if slot == 0:
                        self._lca_pending = int(f0)
                    else:
                        self._rca_pending = int(f0)
                    combo.setCurrentIndex(idx)
        self._refresh_angle_badge()

    # ============================================== UI builders
    def _build_picker_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("LCA (基準):"))
        self._lca_combo = QComboBox()
        self._rca_combo = QComboBox()
        for combo, color in ((self._lca_combo, "#33e6ff"),
                             (self._rca_combo, "#ffd400")):
            combo.addItem("(none)", None)
            for se in self._series:
                combo.addItem(se.label, se)
            combo.setStyleSheet(f"QComboBox {{ color:{color}; }}")
        self._lca_combo.currentIndexChanged.connect(
            lambda _i: self._series_changed(0)
        )
        self._rca_combo.currentIndexChanged.connect(
            lambda _i: self._series_changed(1)
        )
        row.addWidget(self._lca_combo, 1)
        row.addWidget(QLabel("RCA (重畳):"))
        row.addWidget(self._rca_combo, 1)
        self._angle_badge = QLabel("—")
        self._angle_badge.setStyleSheet("padding:2px 8px; font-weight:bold;")
        row.addWidget(self._angle_badge)
        return row

    def _build_view_stack(self) -> QWidget:
        self._stack = QStackedWidget()
        # page 0: composite preview
        self._comp_canvas = _BiXACanvas()
        self._stack.addWidget(self._comp_canvas)
        # page 1: landmark alignment (two source canvases side by side)
        align = QWidget()
        av = QVBoxLayout(align)
        av.setContentsMargins(0, 0, 0, 0)
        instr = QLabel(
            "位置合わせ: 両画像で対応する同じ解剖点を2点ずつクリック "
            "(例: 椎体角・横隔膜・カテ先)。順番を合わせること。"
        )
        instr.setStyleSheet("color:#ccc; padding:2px;")
        av.addWidget(instr)
        crow = QHBoxLayout()
        self._lca_canvas = _BiXACanvas()
        self._rca_canvas = _BiXACanvas()
        self._lca_canvas.set_mode("landmark")
        self._rca_canvas.set_mode("landmark")
        self._lca_canvas.landmarks_changed.connect(self._landmarks_edited)
        self._rca_canvas.landmarks_changed.connect(self._landmarks_edited)
        for cap, cv in (("LCA (基準)", self._lca_canvas),
                        ("RCA (重畳)", self._rca_canvas)):
            box = QVBoxLayout()
            lab = QLabel(cap)
            lab.setStyleSheet("color:#aaa;")
            box.addWidget(lab)
            box.addWidget(cv, 1)
            crow.addLayout(box, 1)
        av.addLayout(crow, 1)
        arow = QHBoxLayout()
        self._apply_btn = QPushButton("✓ この2点で位置合わせ")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply_alignment)
        arow.addWidget(self._apply_btn)
        clr = QPushButton("ランドマーク消去")
        clr.clicked.connect(self._clear_landmarks)
        arow.addWidget(clr)
        self._align_status = QLabel("")
        self._align_status.setStyleSheet("color:#27e0c0;")
        arow.addWidget(self._align_status, 1)
        av.addLayout(arow)
        self._stack.addWidget(align)
        return self._stack

    def _build_transport(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._align_toggle = QPushButton("🎯 位置合わせ")
        self._align_toggle.setCheckable(True)
        self._align_toggle.setToolTip(
            "ランドマークで2画像を空間的に重ね合わせる画面に切替"
        )
        self._align_toggle.toggled.connect(self._toggle_align)
        row.addWidget(self._align_toggle)

        self._play_btn = QPushButton("▶ Play")
        self._play_btn.setCheckable(True)
        self._play_btn.toggled.connect(self._toggle_play)
        row.addWidget(self._play_btn)

        row.addWidget(QLabel("LCA:"))
        self._lca_slider = QSlider(Qt.Orientation.Horizontal)
        self._lca_slider.setEnabled(False)
        self._lca_slider.valueChanged.connect(self._lca_scrubbed)
        row.addWidget(self._lca_slider, 1)
        self._lca_lbl = QLabel("0/0")
        self._lca_lbl.setMinimumWidth(64)
        row.addWidget(self._lca_lbl)

        row.addWidget(QLabel("RCA:"))
        self._rca_slider = QSlider(Qt.Orientation.Horizontal)
        self._rca_slider.setEnabled(False)
        self._rca_slider.valueChanged.connect(self._rca_scrubbed)
        row.addWidget(self._rca_slider, 1)
        self._rca_lbl = QLabel("0/0")
        self._rca_lbl.setMinimumWidth(64)
        row.addWidget(self._rca_lbl)

        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(1.0, 60.0)
        self._fps_spin.setValue(15.0)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.valueChanged.connect(self._set_fps)
        row.addWidget(self._fps_spin)

        self._sync_btn = QPushButton("🔗 Sync ON")
        self._sync_btn.setCheckable(True)
        self._sync_btn.setChecked(True)
        self._sync_btn.toggled.connect(self._toggle_sync)
        row.addWidget(self._sync_btn)
        return row

    def _build_sync_editor(self) -> QWidget:
        box = QGroupBox("時間同期 (LCAフレーム ↔ RCAフレーム)")
        outer = QVBoxLayout(box)
        top = QHBoxLayout()

        def _tinted(text, bg):
            b = QPushButton(text)
            b.setStyleSheet(
                "QPushButton{background:%s;color:#1c1c1c;font-weight:bold;"
                "border:1px solid #888;border-radius:5px;padding:5px 12px;}"
                "QPushButton:hover{background:#fff;}" % bg
            )
            return b

        add = _tinted("+ Sync点を追加 (現在の2フレームを対応付け)", "#d4efdf")
        add.clicked.connect(self._add_sync_point)
        top.addWidget(add)
        save = _tinted("設定を保存…", "#fcf3cf")
        save.clicked.connect(self._save_settings)
        top.addWidget(save)
        load = _tinted("設定を読込…", "#d6eaf8")
        load.clicked.connect(self._load_settings)
        top.addWidget(load)
        mp4 = _tinted("MP4書き出し…", "#fadbd8")
        mp4.clicked.connect(self._export_mp4)
        top.addWidget(mp4)
        self._sync_status = QLabel("")
        self._sync_status.setStyleSheet("color:#c0392b; font-weight:bold;")
        top.addWidget(self._sync_status, 1)
        outer.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(54)
        scroll.setMaximumHeight(120)
        self._rows_host = QWidget()
        self._rows = QHBoxLayout(self._rows_host)
        self._rows.setContentsMargins(4, 2, 4, 2)
        self._rows.addStretch(1)
        scroll.setWidget(self._rows_host)
        outer.addWidget(scroll)
        return box

    # ============================================== series loading
    def _series_changed(self, slot: int) -> None:
        combo = self._lca_combo if slot == 0 else self._rca_combo
        se = combo.currentData()
        plane, wl, fps = self._load_plane(se)
        pending = self._lca_pending if slot == 0 else self._rca_pending
        if slot == 0:
            self._stop_prefetch(0)
            self._lca_plane, self._lca_wl, self._lca_fps = plane, wl, fps
            self._lca_pending = 0
            n = plane.total_frames if plane else 0
            self._fL = max(0, min(pending, n - 1)) if n else 0
            self._lca_slider.setEnabled(n > 1)
            self._lca_slider.blockSignals(True)
            self._lca_slider.setRange(0, max(0, n - 1))
            self._lca_slider.setValue(self._fL)
            self._lca_slider.blockSignals(False)
            self._prefetch_l = self._start_prefetch(plane)
        else:
            self._stop_prefetch(1)
            self._rca_plane, self._rca_wl, self._rca_fps = plane, wl, fps
            self._rca_pending = 0
            n = plane.total_frames if plane else 0
            self._fR = max(0, min(pending, n - 1)) if n else 0
            self._rca_slider.setEnabled(n > 1)
            self._rca_slider.blockSignals(True)
            self._rca_slider.setRange(0, max(0, n - 1))
            self._rca_slider.setValue(self._fR)
            self._rca_slider.blockSignals(False)
            self._prefetch_r = self._start_prefetch(plane)
        # Re-deriving RCA from the mapping keeps a sensible default when the
        # user has set sync points already.
        if slot == 0 and self._sync_on:
            self._derive_rca()
        self._refresh_angle_badge()
        self._refresh_all()

    @staticmethod
    def _load_plane(se):
        if se is None:
            return None, (1.0, 0.0), 30.0
        loaded = dicom_io.load_xa(se)
        plane = (loaded.xa_planes or [None])[0]
        wl = (loaded.window or 1.0, loaded.level or 0.0)
        fps = float(loaded.cine_fps or 30.0)
        return plane, wl, fps

    def _start_prefetch(self, plane):
        if plane is None or plane.total_frames <= 1:
            return None
        pf = _Prefetch(plane, self)
        pf.start()
        return pf

    def _stop_prefetch(self, slot: int) -> None:
        pf = self._prefetch_l if slot == 0 else self._prefetch_r
        if pf is not None:
            pf.stop()
            pf.wait(2000)
        if slot == 0:
            self._prefetch_l = None
        else:
            self._prefetch_r = None

    # ============================================== composite / display
    def _warp_rca(self, rca_u8, out_shape):
        s, r, tx, ty = self._transform or (1.0, 0.0, 0.0, 0.0)
        return bixa.apply_similarity(
            rca_u8, s, r, tx, ty, out_shape=out_shape, fill=255
        )

    def _compose(self):
        if self._lca_plane is None or self._rca_plane is None:
            return None
        lca = _disp_u8(self._lca_plane, self._fL, *self._lca_wl)
        rca = _disp_u8(self._rca_plane, self._fR, *self._rca_wl)
        rca_w = self._warp_rca(rca, lca.shape)
        return bixa.blend_min(lca, rca_w)

    def _refresh_all(self) -> None:
        nL = self._lca_plane.total_frames if self._lca_plane else 0
        nR = self._rca_plane.total_frames if self._rca_plane else 0
        self._lca_lbl.setText(f"{self._fL + 1}/{nL}" if nL else "0/0")
        self._rca_lbl.setText(f"{self._fR + 1}/{nR}" if nR else "0/0")
        if self._align_toggle.isChecked():
            if self._lca_plane is not None:
                self._lca_canvas.set_image(
                    _disp_u8(self._lca_plane, self._fL, *self._lca_wl)
                )
            if self._rca_plane is not None:
                self._rca_canvas.set_image(
                    _disp_u8(self._rca_plane, self._fR, *self._rca_wl)
                )
        else:
            self._comp_canvas.set_image(self._compose())

    def _refresh_angle_badge(self) -> None:
        va = _plane_view_angles(self._lca_plane) if self._lca_plane else None
        vb = _plane_view_angles(self._rca_plane) if self._rca_plane else None
        if va is None or vb is None:
            self._angle_badge.setText(
                f"LCA {_fmt_view(va)}  /  RCA {_fmt_view(vb)}"
            )
            self._angle_badge.setStyleSheet(
                "padding:2px 8px; font-weight:bold; color:#aaa;"
            )
            return
        sep = bixa.view_separation_deg(va, vb)
        color = "#1e8449" if sep <= 5 else ("#b9770e" if sep <= 15 else "#c0392b")
        self._angle_badge.setText(
            f"LCA {_fmt_view(va)}  /  RCA {_fmt_view(vb)}   差 {sep:.1f}°"
        )
        self._angle_badge.setStyleSheet(
            f"padding:2px 8px; font-weight:bold; color:white; "
            f"background:{color}; border-radius:5px;"
        )

    # ============================================== alignment workflow
    def _toggle_align(self, on: bool) -> None:
        self._stack.setCurrentIndex(1 if on else 0)
        if on:
            self._lca_canvas.set_landmarks(
                self._lca_canvas.landmarks()
            )
            self._rca_canvas.set_landmarks(
                self._rca_canvas.landmarks()
            )
        self._refresh_all()

    def _landmarks_edited(self, _pts) -> None:
        ready = (len(self._lca_canvas.landmarks()) == 2
                 and len(self._rca_canvas.landmarks()) == 2)
        self._apply_btn.setEnabled(ready)
        self._align_status.setText(
            "2点ずつ揃いました → 「位置合わせ」を押す" if ready
            else "LCA・RCA それぞれ対応2点をクリック"
        )

    def _apply_alignment(self) -> None:
        src = self._rca_canvas.landmarks()       # moving = RCA
        dst = self._lca_canvas.landmarks()        # reference = LCA
        if len(src) != 2 or len(dst) != 2:
            return
        try:
            self._transform = bixa.similarity_from_landmarks(src, dst)
        except ValueError as e:
            self._align_status.setText(f"位置合わせ失敗: {e}")
            return
        s, r, _tx, _ty = self._transform
        self._align_status.setText(
            f"位置合わせ完了 (倍率 {s:.2f}, 回転 {math.degrees(r):.1f}°)"
        )
        self._align_toggle.setChecked(False)     # back to composite
        self._refresh_all()

    def _clear_landmarks(self) -> None:
        self._lca_canvas.clear_landmarks()
        self._rca_canvas.clear_landmarks()
        self._transform = None
        self._align_status.setText("ランドマーク・位置合わせを消去しました")
        self._refresh_all()

    # ============================================== transport
    def _lca_scrubbed(self, value: int) -> None:
        self._fL = int(value)
        if self._sync_on:
            self._derive_rca()
        self._refresh_all()

    def _rca_scrubbed(self, value: int) -> None:
        # Independent RCA nudge — used to line up a phase before pinning a
        # sync point. Does NOT re-derive from the mapping.
        self._fR = int(value)
        self._refresh_all()

    def _derive_rca(self) -> None:
        if self._rca_plane is None:
            return
        fr = multisync.map_frame(
            self.sync_points, 0, 1, self._fL, self._lca_fps, self._rca_fps
        )
        self._fR = max(0, min(int(round(fr)), self._rca_plane.total_frames - 1))
        self._rca_slider.blockSignals(True)
        self._rca_slider.setValue(self._fR)
        self._rca_slider.blockSignals(False)

    def _toggle_sync(self, on: bool) -> None:
        self._sync_on = on
        self._sync_btn.setText("🔗 Sync ON" if on else "🔗 Sync OFF")
        if on:
            self._derive_rca()
            self._refresh_all()

    def _set_fps(self, _v) -> None:
        if self._timer.isActive():
            self._timer.start(int(1000.0 / max(self._fps_spin.value(), 1e-3)))

    def _toggle_play(self, on: bool) -> None:
        if on and (self._lca_plane is None or self._lca_plane.total_frames < 2):
            self._play_btn.setChecked(False)
            return
        self._play_btn.setText("⏸ Pause" if on else "▶ Play")
        if on:
            self._timer.start(int(1000.0 / max(self._fps_spin.value(), 1e-3)))
        else:
            self._timer.stop()

    def _tick(self) -> None:
        if self._lca_plane is None:
            return
        nL = self._lca_plane.total_frames
        self._fL = (self._fL + 1) % nL
        self._lca_slider.blockSignals(True)
        self._lca_slider.setValue(self._fL)
        self._lca_slider.blockSignals(False)
        if self._sync_on:
            self._derive_rca()
        self._refresh_all()

    # ============================================== sync points
    def _add_sync_point(self) -> None:
        if self._lca_plane is None or self._rca_plane is None:
            self._sync_status.setText("LCA と RCA の両方を選んでください。")
            return
        cand = self.sync_points + [
            {"frames": [self._fL, self._fR], "rots": [0.0, 0.0]}
        ]
        if multisync.has_conflict(cand, 0, _N_SLOTS):
            self._sync_status.setStyleSheet("color:#c0392b;font-weight:bold;")
            self._sync_status.setText(
                "矛盾 — このフレーム対は既存のSync点と前後関係が逆転します。"
            )
            return
        self.sync_points.append(
            {"frames": [self._fL, self._fR], "rots": [0.0, 0.0]}
        )
        self._sync_status.setStyleSheet("color:#1e8449;font-weight:bold;")
        self._sync_status.setText(
            f"Sync-{len(self.sync_points)} 追加: "
            f"LCA F{self._fL + 1} ↔ RCA F{self._fR + 1}"
        )
        self._rebuild_rows()
        if self._sync_on:
            self._derive_rca()
            self._refresh_all()

    def _rebuild_rows(self) -> None:
        while self._rows.count():
            it = self._rows.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        for pi, sp in enumerate(self.sync_points):
            self._rows.addWidget(self._sync_chip(pi, sp))
        self._rows.addStretch(1)

    def _sync_chip(self, pi: int, sp: dict) -> QWidget:
        chip = QFrame()
        chip.setFrameShape(QFrame.Shape.StyledPanel)
        h = QHBoxLayout(chip)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(3)
        fL, fR = sp["frames"]
        lbl = QLabel(f"S{pi + 1}: L{fL + 1} ↔ R{fR + 1}")
        lbl.setStyleSheet("font-weight:bold; color:#ddd;")
        h.addWidget(lbl)
        jump = QPushButton("Jump")
        jump.clicked.connect(lambda _c, p=pi: self._jump_point(p))
        h.addWidget(jump)
        dele = QPushButton("✕")
        dele.setFixedWidth(26)
        dele.clicked.connect(lambda _c, p=pi: self._del_point(p))
        h.addWidget(dele)
        return chip

    def _jump_point(self, pi: int) -> None:
        sp = self.sync_points[pi]
        fL, fR = sp["frames"]
        self._fL, self._fR = int(fL), int(fR)
        for sl, val in ((self._lca_slider, self._fL),
                        (self._rca_slider, self._fR)):
            sl.blockSignals(True)
            sl.setValue(val)
            sl.blockSignals(False)
        self._refresh_all()

    def _del_point(self, pi: int) -> None:
        if 0 <= pi < len(self.sync_points):
            del self.sync_points[pi]
            self._sync_status.setText("")
            self._rebuild_rows()
            if self._sync_on:
                self._derive_rca()
                self._refresh_all()

    # ============================================== settings I/O
    def _save_settings(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "BiXA設定を保存", "", "BiXA settings (*.txt)"
        )
        if not path:
            return
        lca = self._lca_combo.currentData()
        rca = self._rca_combo.currentData()
        lines = [
            "# Virtual BiXA settings v1",
            f"lca={lca.label if lca else ''}",
            f"rca={rca.label if rca else ''}",
            f"lca_fps={self._lca_fps:.3f}",
            f"rca_fps={self._rca_fps:.3f}",
        ]
        if self._transform is not None:
            s, r, tx, ty = self._transform
            lines.append(f"transform={s:.6f}\t{r:.6f}\t{tx:.4f}\t{ty:.4f}")
        lines.append("[Landmarks]")
        for i, (x, y) in enumerate(self._lca_canvas.landmarks()):
            lines.append(f"lca{i + 1}={x:.2f}\t{y:.2f}")
        for i, (x, y) in enumerate(self._rca_canvas.landmarks()):
            lines.append(f"rca{i + 1}={x:.2f}\t{y:.2f}")
        lines.append("[SyncPoints]")
        for sp in self.sync_points:
            fL, fR = sp["frames"]
            lines.append(f"{fL}\t{fR}")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError as exc:
            QMessageBox.warning(self, "保存", str(exc))
            return
        self._sync_status.setStyleSheet("color:#1e8449;font-weight:bold;")
        self._sync_status.setText("保存しました。")

    def _load_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "BiXA設定を読込", "", "BiXA settings (*.txt)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            QMessageBox.warning(self, "読込", str(exc))
            return
        section = None
        lca_lbl = rca_lbl = ""
        transform = None
        lca_lm: list[tuple[float, float]] = []
        rca_lm: list[tuple[float, float]] = []
        sps: list[dict] = []
        for raw in text.splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("[") and s.endswith("]"):
                section = s[1:-1]
                continue
            if section is None:
                if s.startswith("lca="):
                    lca_lbl = s.split("=", 1)[1]
                elif s.startswith("rca="):
                    rca_lbl = s.split("=", 1)[1]
                elif s.startswith("transform="):
                    parts = s.split("=", 1)[1].split("\t")
                    if len(parts) == 4:
                        transform = tuple(float(p) for p in parts)
            elif section == "Landmarks":
                key, _, val = s.partition("=")
                xy = val.split("\t")
                if len(xy) == 2:
                    pt = (float(xy[0]), float(xy[1]))
                    (lca_lm if key.startswith("lca") else rca_lm).append(pt)
            elif section == "SyncPoints":
                parts = raw.split("\t")
                if len(parts) >= 2:
                    sps.append({"frames": [int(parts[0]), int(parts[1])],
                                "rots": [0.0, 0.0]})
        # Re-select series by label.
        for combo, lbl in ((self._lca_combo, lca_lbl), (self._rca_combo, rca_lbl)):
            if not lbl:
                continue
            for ci in range(combo.count()):
                se = combo.itemData(ci)
                if se is not None and se.label == lbl:
                    combo.setCurrentIndex(ci)
                    break
        self._lca_canvas.set_landmarks(lca_lm)
        self._rca_canvas.set_landmarks(rca_lm)
        self._transform = transform
        self.sync_points = sps
        self._rebuild_rows()
        if self._sync_on:
            self._derive_rca()
        self._refresh_all()
        self._sync_status.setStyleSheet("color:#1e8449;font-weight:bold;")
        self._sync_status.setText(
            f"読込: Sync点 {len(sps)} / 位置合わせ "
            f"{'あり' if transform else 'なし'}"
        )

    # ============================================== MP4 export
    def _export_mp4(self) -> None:
        if self._lca_plane is None or self._rca_plane is None:
            QMessageBox.information(
                self, "MP4書き出し", "LCA と RCA の両方を選んでください。"
            )
            return
        from multi_dicomviewer.ui.export_dialog import ExportDialog
        dlg_cfg = ExportDialog(
            "mp4", 1, default_fps=self._fps_spin.value(),
            show_filename_fields=False,
            title_override="Virtual BiXA — コンポジットMP4書き出し",
            parent=self,
        )
        if dlg_cfg.exec() != dlg_cfg.DialogCode.Accepted:
            return
        cfg = dlg_cfg.result_settings()
        path, _ = QFileDialog.getSaveFileName(
            self, "コンポジットMP4を保存", "", "MP4 video (*.mp4)"
        )
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"

        nL = self._lca_plane.total_frames
        dlg = QProgressDialog("BiXA MP4を書き出し中…", "Cancel", 0, nL, self)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(0)

        frames_rgb: list[np.ndarray] = []
        for i in range(nL):
            if dlg.wasCanceled():
                break
            lca = _disp_u8(self._lca_plane, i, *self._lca_wl)
            if self._sync_on:
                fr = multisync.map_frame(
                    self.sync_points, 0, 1, i, self._lca_fps, self._rca_fps
                )
                fr = max(0, min(int(round(fr)), self._rca_plane.total_frames - 1))
            else:
                fr = min(i, self._rca_plane.total_frames - 1)
            rca = _disp_u8(self._rca_plane, fr, *self._rca_wl)
            comp = bixa.blend_min(lca, self._warp_rca(rca, lca.shape))
            frames_rgb.append(np.repeat(comp[:, :, None], 3, axis=2))
            if i % 8 == 0:
                dlg.setValue(i)
                QApplication.processEvents()
        dlg.close()
        if not frames_rgb:
            return
        try:
            from multi_dicomviewer.core import export as exporter
            exporter.write_mp4(
                path, frames_rgb, fps=cfg.fps,
                bitrate_mbps=cfg.bitrate_mbps, crf=cfg.crf,
            )
        except Exception as e:                       # noqa: BLE001
            QMessageBox.critical(self, "MP4書き出し", f"エンコード失敗:\n{e}")
            return
        qual = f"CRF {cfg.crf}" if cfg.crf is not None \
            else f"{cfg.bitrate_mbps} Mbps"
        QMessageBox.information(
            self, "MP4書き出し",
            f"{len(frames_rgb)} フレーム @ {cfg.fps:.1f} fps, "
            f"{qual} を保存:\n{path}"
        )

    def closeEvent(self, e):
        self._timer.stop()
        self._stop_prefetch(0)
        self._stop_prefetch(1)
        super().closeEvent(e)
