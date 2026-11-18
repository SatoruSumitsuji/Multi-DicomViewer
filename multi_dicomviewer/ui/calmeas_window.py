"""Tools ▸ Cal-Measure — calibration-first measurement on a 2-D series
(XA angio, echo, …), separate from the ordinary Measure (user 2026-11-18).

The series shown in the active pane is handed over (every frame; for a
biplane pane the plane(s) shown). Steps: pick the frame → 1. calibrate the
pixel size (two points + their distance, a catheter of known French size,
or the DICOM value) → 2. measure (distance, polyline length, FWHM diameter
on a line across the vessel) → save as ``<name>.CalMeas.json`` (the
calibration of every plane + every measurement) or CSV.

Mouse: left = place points; right-click / double-click = finish a polyline
or the catheter polygon; wheel = zoom; middle-drag or Ctrl+left-drag = pan;
Esc = cancel the current tool. Each plane has its own calibration.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton, QSlider,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from multi_dicomviewer.core import calmeas as cm
from multi_dicomviewer.i18n import t

FORMAT = "MDV-CalMeas"
SUFFIX = ".CalMeas.json"
FR_SIZES = (4, 5, 6, 7, 8)


def _apply_window(f, window, level) -> np.ndarray:
    lo = level - window / 2.0
    out = (np.asarray(f, np.float32) - lo) / max(window, 1e-6)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


class _Canvas(QWidget):
    """Image + overlays; reports clicks in IMAGE pixel coordinates."""
    clicked = pyqtSignal(float, float)
    finished = pyqtSignal()
    moved = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._qimg = None
        self._shape = (1, 1)
        self._scale = 1.0
        self._off = QPointF(0, 0)
        self._fitted = False
        self._pan = None
        self.overlay = None                       # fn(painter, to_widget)

    # -- image / view
    def set_image(self, arr) -> None:
        a = np.ascontiguousarray(arr)
        h, w = a.shape[:2]
        if a.ndim == 2:
            q = QImage(a.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            a = np.ascontiguousarray(a[..., :3].astype(np.uint8))
            q = QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._qimg = q.copy()
        if (h, w) != self._shape:
            self._shape = (h, w)
            self._fitted = False
        if not self._fitted:
            self.fit()
        self.update()

    def fit(self) -> None:
        h, w = self._shape
        if w <= 0 or h <= 0 or self.width() <= 0:
            return
        self._scale = min(self.width() / w, self.height() / h)
        self._off = QPointF((self.width() - w * self._scale) / 2,
                            (self.height() - h * self._scale) / 2)
        self._fitted = True
        self.update()

    def to_widget(self, x, y) -> QPointF:
        return QPointF(self._off.x() + x * self._scale,
                       self._off.y() + y * self._scale)

    def to_image(self, p) -> tuple[float, float]:
        return ((p.x() - self._off.x()) / self._scale,
                (p.y() - self._off.y()) / self._scale)

    @property
    def scale(self) -> float:
        return self._scale

    # -- Qt events
    def resizeEvent(self, e):
        self._fitted = False
        self.fit()
        super().resizeEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0))
        if self._qimg is not None:
            h, w = self._shape
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(QRectF(self._off.x(), self._off.y(), w * self._scale,
                               h * self._scale), self._qimg)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.overlay is not None:
            self.overlay(p, self.to_widget)
        p.end()

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        pos = e.position()
        x, y = self.to_image(pos)
        self._scale = max(0.05, min(40.0, self._scale * f))
        self._off = QPointF(pos.x() - x * self._scale, pos.y() - y * self._scale)
        self.update()

    def mousePressEvent(self, e):
        pan = (e.button() == Qt.MouseButton.MiddleButton
               or (e.button() == Qt.MouseButton.LeftButton
                   and e.modifiers() & Qt.KeyboardModifier.ControlModifier))
        if pan:
            self._pan = (e.position(), QPointF(self._off))
            return
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(*self.to_image(e.position()))
        elif e.button() == Qt.MouseButton.RightButton:
            self.finished.emit()

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.finished.emit()

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            p0, o0 = self._pan
            d = e.position() - p0
            self._off = QPointF(o0.x() + d.x(), o0.y() + d.y())
            self.update()
            return
        self.moved.emit(*self.to_image(e.position()))

    def mouseReleaseEvent(self, e):
        self._pan = None


class CalMeasWindow(QWidget):
    """The Cal-Measure tool window. *planes*: [{"plane": XAPlane, "index":
    int, "name": str}] — the plane(s) shown in the pane."""

    TOOL_HINT = {
        None: "ツールを選んでください。ホイール=拡大縮小、中ボタン/Ctrl+ドラッグ=移動。",
        "cal2": "距離が分かっている 2 点をクリックし、距離 (mm) を入力して「確定」。",
        "calcath": "造影剤で満たされたカテーテルの区間を Polygon で囲み、右クリック/"
                   "ダブルクリックで閉じます。",
        "dist": "2 点をクリック → 直線距離。",
        "poly": "血管に沿って点をクリックし、右クリック/ダブルクリックで終了 → 長さ。",
        "diam": "血管を横切るように 2 点をクリック（線は血管の外まで伸ばす）→ 半値幅径。",
    }

    def __init__(self, planes, frame=0, window=None, level=None, title="",
                 series_uid="", modality="", parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(t("Cal-Measure") + (f" — {title}" if title else ""))
        self.resize(1250, 820)
        self._planes = list(planes)
        self._title = title
        self._uid = series_uid
        self._modality = modality
        self._window = window
        self._level = level
        self._state = [{"cal": None, "meas": []} for _ in self._planes]
        self._pi = 0
        self._frame = int(frame)
        self._tool = None
        self._pts = []
        self._mouse = None
        self._pending = None            # finished calibration awaiting 確定
        self._next_id = 1
        self._sel_id = None
        self._dirty = False
        self._path = ""
        self._build()
        self._set_plane(0)

    # ---- data access -----------------------------------------------------
    @property
    def _plane(self):
        return self._planes[self._pi]["plane"]

    def _n_frames(self) -> int:
        p = self._plane
        return int(getattr(p, "total_frames", 0) or len(p.volume))

    def _raw(self):
        return self._plane.frame(self._frame)

    def _gray(self):
        return cm.to_gray(self._raw())

    def _display(self):
        f = self._raw()
        if f.ndim == 3:
            return f
        if self._window is None or self._level is None:
            lo, hi = np.percentile(f, [0.5, 99.5])
            return _apply_window(f, max(hi - lo, 1e-6), 0.5 * (lo + hi))
        return _apply_window(f, self._window, self._level)

    def _cal(self):
        return self._state[self._pi]["cal"]

    def _mmpp(self, pi=None):
        c = self._state[self._pi if pi is None else pi]["cal"]
        return None if c is None else float(c["mm_per_px"])

    # ---- UI ------------------------------------------------------------------
    def _build(self) -> None:
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        top = QHBoxLayout()
        self._plane_cb = QComboBox()
        for p in self._planes:
            self._plane_cb.addItem(str(p.get("name") or t("Plane {n}",
                                                         n=p["index"] + 1)))
        self._plane_cb.currentIndexChanged.connect(self._set_plane)
        self._plane_cb.setVisible(len(self._planes) > 1)
        top.addWidget(self._plane_cb)
        self._prev_btn = QPushButton("◀")
        self._next_btn = QPushButton("▶")
        for b in (self._prev_btn, self._next_btn):
            b.setFixedWidth(34)
        self._prev_btn.clicked.connect(lambda: self._set_frame(self._frame - 1))
        self._next_btn.clicked.connect(lambda: self._set_frame(self._frame + 1))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.valueChanged.connect(self._set_frame)
        self._frame_spin = QSpinBox()
        self._frame_spin.valueChanged.connect(lambda v: self._set_frame(v - 1))
        top.addWidget(QLabel(t("フレーム")))
        top.addWidget(self._prev_btn)
        top.addWidget(self._slider, 1)
        top.addWidget(self._next_btn)
        top.addWidget(self._frame_spin)
        self._fit_btn = QPushButton(t("全体表示"))
        top.addWidget(self._fit_btn)
        left.addLayout(top)
        self.canvas = _Canvas()
        self.canvas.overlay = self._paint_overlay
        self.canvas.clicked.connect(self._on_click)
        self.canvas.finished.connect(self._on_finish)
        self.canvas.moved.connect(self._on_move)
        self._fit_btn.clicked.connect(self.canvas.fit)
        left.addWidget(self.canvas, 1)
        self._hint = QLabel()
        self._hint.setWordWrap(True)
        left.addWidget(self._hint)
        root.addLayout(left, 3)

        side = QVBoxLayout()
        g1 = QGroupBox(t("1. キャリブレーション"))
        v1 = QVBoxLayout(g1)
        self._cal_lbl = QLabel()
        self._cal_lbl.setWordWrap(True)
        v1.addWidget(self._cal_lbl)
        row = QHBoxLayout()
        self._cal2_btn = QPushButton(t("2点間距離"))
        self._calcath_btn = QPushButton(t("カテ径"))
        self._caldicom_btn = QPushButton(t("DICOM値"))
        for b in (self._cal2_btn, self._calcath_btn, self._caldicom_btn):
            b.setCheckable(b is not self._caldicom_btn)
            row.addWidget(b)
        self._cal2_btn.clicked.connect(lambda: self._set_tool("cal2"))
        self._calcath_btn.clicked.connect(lambda: self._set_tool("calcath"))
        self._caldicom_btn.clicked.connect(self._cal_dicom)
        v1.addLayout(row)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel(t("距離 (mm)")))
        self._dist_spin = QDoubleSpinBox()
        self._dist_spin.setRange(0.1, 1000.0)
        self._dist_spin.setDecimals(2)
        self._dist_spin.setValue(10.0)
        r2.addWidget(self._dist_spin)
        r2.addWidget(QLabel(t("カテーテル")))
        self._fr_cb = QComboBox()
        for fr in FR_SIZES:
            self._fr_cb.addItem(f"{fr} Fr ({fr * cm.FR_MM:.2f} mm)", fr)
        self._fr_cb.setCurrentIndex(FR_SIZES.index(6))
        r2.addWidget(self._fr_cb)
        v1.addLayout(r2)
        self._pend_lbl = QLabel()
        self._pend_lbl.setWordWrap(True)
        v1.addWidget(self._pend_lbl)
        self._cal_ok_btn = QPushButton(t("確定"))
        self._cal_ok_btn.clicked.connect(self._cal_confirm)
        v1.addWidget(self._cal_ok_btn)
        side.addWidget(g1)

        g2 = QGroupBox(t("2. 計測"))
        v2 = QVBoxLayout(g2)
        row = QHBoxLayout()
        self._dist_btn = QPushButton(t("距離"))
        self._poly_btn = QPushButton(t("Polyline長"))
        self._diam_btn = QPushButton(t("径（半値幅）"))
        for b, k in ((self._dist_btn, "dist"), (self._poly_btn, "poly"),
                     (self._diam_btn, "diam")):
            b.setCheckable(True)
            b.clicked.connect(lambda _c=False, k=k: self._set_tool(k))
            row.addWidget(b)
        v2.addLayout(row)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["#", t("種類"), t("値 (mm)"), t("フレーム"), t("備考")])
        hh = self._table.horizontalHeader()
        for col in range(4):
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setWordWrap(True)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._on_row)
        v2.addWidget(self._table, 1)
        self._del_btn = QPushButton(t("選択した計測を削除"))
        self._del_btn.clicked.connect(self._delete_selected)
        v2.addWidget(self._del_btn)
        side.addWidget(g2, 1)
        row = QHBoxLayout()
        self._save_btn = QPushButton(t("保存…"))
        self._load_btn = QPushButton(t("読込…"))
        self._csv_btn = QPushButton(t("CSV…"))
        self._close_btn = QPushButton(t("閉じる"))
        self._save_btn.clicked.connect(lambda: self.save())
        self._load_btn.clicked.connect(lambda: self.load())
        self._csv_btn.clicked.connect(lambda: self.export_csv())
        self._close_btn.clicked.connect(self.close)
        for b in (self._save_btn, self._load_btn, self._csv_btn,
                  self._close_btn):
            row.addWidget(b)
        side.addLayout(row)
        root.addLayout(side, 2)

    # ---- plane / frame -------------------------------------------------------
    def _set_plane(self, i) -> None:
        self._pi = max(0, min(int(i), len(self._planes) - 1))
        n = self._n_frames()
        for w in (self._slider, self._frame_spin):
            w.blockSignals(True)
        self._slider.setRange(0, max(0, n - 1))
        self._frame_spin.setRange(1, max(1, n))
        for w in (self._slider, self._frame_spin):
            w.blockSignals(False)
        self._set_tool(None)
        self._pending = None
        self._set_frame(min(self._frame, max(0, n - 1)), force=True)
        self._refresh()

    def _set_frame(self, f, force=False) -> None:
        f = max(0, min(int(f), self._n_frames() - 1))
        if f == self._frame and not force:
            return
        self._frame = f
        for w, v in ((self._slider, f), (self._frame_spin, f + 1)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self._pts = []
        self.canvas.set_image(self._display())

    # ---- tools -----------------------------------------------------------------
    def _set_tool(self, key) -> None:
        if key is not None and key == self._tool:
            key = None                               # toggle off
        if key in ("dist", "poly", "diam") and self._mmpp() is None:
            QMessageBox.information(self, t("Cal-Measure"),
                                    t("先にキャリブレーションをしてください。"))
            key = None
        self._tool = key
        self._pts = []
        for b, k in ((self._cal2_btn, "cal2"), (self._calcath_btn, "calcath"),
                     (self._dist_btn, "dist"), (self._poly_btn, "poly"),
                     (self._diam_btn, "diam")):
            b.setChecked(k == key)
        if key in ("cal2", "calcath"):
            self._pending = None
        self._refresh()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self._set_tool(None)
            return
        if e.key() == Qt.Key.Key_Delete:
            self._delete_selected()
            return
        super().keyPressEvent(e)

    def _on_move(self, x, y) -> None:
        self._mouse = (x, y)
        if self._tool and self._pts:
            self.canvas.update()

    def _on_click(self, x, y) -> None:
        k = self._tool
        if k is None:
            return
        self._pts.append((float(x), float(y)))
        if k == "cal2" and len(self._pts) == 2:
            d = cm.path_length(self._pts)
            self._pending = {"method": "two_point", "frame": self._frame,
                             "p0": list(self._pts[0]), "p1": list(self._pts[1]),
                             "dist_px": d}
            self._tool_done()
        elif k == "dist" and len(self._pts) == 2:
            self._add_meas("distance", self._pts)
            self._pts = []
        elif k == "diam" and len(self._pts) == 2:
            self._add_diameter(self._pts)
            self._pts = []
        self._refresh()

    def _on_finish(self) -> None:
        k = self._tool
        if k == "poly" and len(self._pts) >= 2:
            self._add_meas("polyline", self._pts)
            self._pts = []
        elif k == "calcath" and len(self._pts) >= 3:
            self._catheter(self._pts)
            self._tool_done()
        self._refresh()

    def _tool_done(self) -> None:
        self._tool = None
        self._pts = []
        for b in (self._cal2_btn, self._calcath_btn):
            b.setChecked(False)

    # ---- calibration -------------------------------------------------------------
    def _catheter(self, poly) -> None:
        r = cm.catheter_width(self._gray(), poly)
        if not r.ok:
            why = {"polygon": t("Polygon が小さすぎます。"),
                   "shape": t("細長い物体（カテーテル）が見つかりません。"),
                   "short": t("カテーテルの区間が短すぎます。"),
                   "profiles": t("断面の半値幅を測れませんでした。"),
                   "dip": t("断面が二峰性です（壁だけが写っている = 造影剤で満たされて"
                            "いない区間?）。満たされた区間を囲み直してください。")}
            self._pending = None
            QMessageBox.warning(self, t("Cal-Measure"),
                                why.get(r.reason, t("抽出できませんでした。")))
            return
        self._pending = {"method": "catheter", "frame": self._frame,
                         "polygon": [list(p) for p in poly],
                         "width_px": r.width_px, "iqr_px": r.iqr_px,
                         "n_profiles": r.n_profiles,
                         "centerline": [list(p) for p in r.centerline],
                         "warnings": list(r.warnings)}

    def _cal_dicom(self) -> None:
        ds = getattr(self._plane, "_ds", None)
        r = cm.dicom_mm_per_px(ds)
        if r is None:
            QMessageBox.information(self, t("Cal-Measure"),
                                    t("DICOM に画素サイズの情報がありません。"))
            return
        self._tool_done()
        self._pending = {"method": "dicom", "frame": self._frame,
                         "mm_per_px": r[0], "source": r[1]}
        self._refresh()

    def _cal_confirm(self) -> None:
        pd = self._pending
        if pd is None:
            return
        c = dict(pd)
        if c["method"] == "two_point":
            c["dist_mm"] = float(self._dist_spin.value())
            if c["dist_px"] < 2:
                QMessageBox.warning(self, t("Cal-Measure"),
                                    t("2 点が近すぎます。"))
                return
            c["mm_per_px"] = c["dist_mm"] / c["dist_px"]
        elif c["method"] == "catheter":
            fr = int(self._fr_cb.currentData())
            c["fr"] = fr
            c["diam_mm"] = fr * cm.FR_MM
            c["mm_per_px"] = c["diam_mm"] / c["width_px"]
        c["at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._state[self._pi]["cal"] = c
        self._pending = None
        self._dirty = True
        self._refresh()

    @staticmethod
    def cal_text(c) -> str:
        if c is None:
            return t("未較正（計測できません）")
        m = c["method"]
        if m == "two_point":
            s = t("2点間距離 {d:.2f} mm / {p:.1f} px", d=c["dist_mm"],
                  p=c["dist_px"])
        elif m == "catheter":
            s = t("カテ径 {fr} Fr ({d:.2f} mm) / 幅 {w:.2f} px", fr=c["fr"],
                  d=c["diam_mm"], w=c["width_px"])
        else:
            s = t("DICOM値 ({src})", src=c.get("source", ""))
        return t("較正: {s} → {m:.4f} mm/px（フレーム {f}）", s=s,
                 m=c["mm_per_px"], f=int(c.get("frame", 0)) + 1)

    # ---- measurements -------------------------------------------------------------
    def _add_meas(self, kind, pts) -> dict:
        rec = {"id": self._next_id, "type": kind, "frame": self._frame,
               "points": [list(map(float, p)) for p in pts],
               "length_px": cm.path_length(pts)}
        self._next_id += 1
        self._state[self._pi]["meas"].append(rec)
        self._dirty = True
        self._sel_id = rec["id"]
        return rec

    def _add_diameter(self, pts) -> dict | None:
        d = cm.measure_diameter(self._gray(), pts[0], pts[1])
        if not d.fwhm.ok:
            why = {"edge": t("線を血管の外まで伸ばしてください（端が内腔の中です）。"),
                   "contrast": t("内腔のコントラストが足りません。"),
                   "short": t("線が短すぎます。")}
            self._hint.setText(why.get(d.fwhm.reason, t("半値幅を測れませんでした。")))
            return None
        rec = self._add_meas("diameter", pts)
        rec["fwhm_px"] = d.fwhm.width_px
        rec["edges"] = [list(e) for e in d.edges]
        rec["dark"] = bool(d.fwhm.dark)
        rec["direction_ok"] = bool(d.direction_ok)
        rec["direction_reason"] = d.direction_reason
        rec["angle_deg"] = None if math.isnan(d.angle_deg) else d.angle_deg
        if d.corrected is not None:
            rec["corrected_px"] = d.corrected.width_px
            rec["corrected_line"] = [list(p) for p in d.corrected_line]
            rec["corrected_edges"] = [list(p) for p in d.corrected_edges]
        return rec

    def _value_mm(self, rec, pi=None):
        s = self._mmpp(pi)
        if s is None:
            return None
        px = rec["fwhm_px"] if rec["type"] == "diameter" else rec["length_px"]
        return px * s

    def note_text(self, rec, pi=None) -> str:
        if rec["type"] != "diameter":
            return ""
        if not rec.get("direction_ok"):
            return t("方向推定不可（狭窄・分岐・重なりなど）")
        a = rec.get("angle_deg") or 0.0
        txt = t("線の傾き約 {a:.0f}°（+{p:.0f}%）", a=a,
                p=(1 / math.cos(math.radians(a)) - 1) * 100)
        if rec.get("corrected_px") is not None and self._mmpp(pi) is not None:
            txt += " / " + t("参考(直交) {v:.2f} mm",
                             v=rec["corrected_px"] * self._mmpp(pi))
        return txt

    KIND = {"distance": "距離", "polyline": "Polyline長", "diameter": "径（半値幅）"}

    def _delete_selected(self) -> None:
        if self._sel_id is None:
            return
        st = self._state[self._pi]
        st["meas"] = [m for m in st["meas"] if m["id"] != self._sel_id]
        self._sel_id = None
        self._dirty = True
        self._refresh()

    def _on_row(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        it = self._table.item(rows[0].row(), 0)
        if it is None:
            return
        mid = int(it.data(Qt.ItemDataRole.UserRole))
        self._sel_id = mid
        rec = next((m for m in self._state[self._pi]["meas"]
                    if m["id"] == mid), None)
        if rec is not None and rec["frame"] != self._frame:
            self._set_frame(rec["frame"])
        self.canvas.update()

    # ---- refresh -----------------------------------------------------------------
    def _refresh(self) -> None:
        c = self._cal()
        self._cal_lbl.setText(self.cal_text(c))
        has_dicom = cm.dicom_mm_per_px(getattr(self._plane, "_ds", None))
        self._caldicom_btn.setEnabled(has_dicom is not None)
        pd = self._pending
        if pd is None:
            self._pend_lbl.setText("")
        elif pd["method"] == "two_point":
            self._pend_lbl.setText(t("2 点間 {p:.1f} px — 距離を入力して「確定」。",
                                     p=pd["dist_px"]))
        elif pd["method"] == "catheter":
            w = {"dip": t("断面が二峰性（造影剤で満たされていない?）"),
                 "spread": t("幅のばらつきが大きい"),
                 "few": t("断面が少ない")}
            warn = "、".join(w[x] for x in pd["warnings"] if x in w)
            self._pend_lbl.setText(
                t("カテーテル幅 {w:.2f} px（{n} 断面、IQR {q:.2f} px）。"
                  "Fr を選んで「確定」。", w=pd["width_px"], n=pd["n_profiles"],
                  q=pd["iqr_px"]) + (("\n⚠ " + warn) if warn else ""))
        else:
            note = t("（XA の検出器面の値: 拡大率は未補正）") \
                if pd["source"] == "ImagerPixelSpacing" else ""
            self._pend_lbl.setText(t("DICOM値 {m:.4f} mm/px（{src}）{n}。「確定」で使用。",
                                     m=pd["mm_per_px"], src=pd["source"], n=note))
        self._cal_ok_btn.setEnabled(pd is not None)
        ok = c is not None
        for b in (self._dist_btn, self._poly_btn, self._diam_btn):
            b.setEnabled(ok)
        self._hint.setText(t(self.TOOL_HINT.get(self._tool, "")))
        self._fill_table()
        self.canvas.update()

    def _fill_table(self) -> None:
        ms = self._state[self._pi]["meas"]
        self._table.blockSignals(True)
        self._table.setRowCount(len(ms))
        sel_row = None
        for r, m in enumerate(ms):
            v = self._value_mm(m)
            cells = [str(m["id"]), t(self.KIND[m["type"]]),
                     "—" if v is None else f"{v:.2f}", str(m["frame"] + 1),
                     self.note_text(m)]
            for col, txt in enumerate(cells):
                it = QTableWidgetItem(txt)
                if col == 0:
                    it.setData(Qt.ItemDataRole.UserRole, m["id"])
                self._table.setItem(r, col, it)
            self._table.item(r, 4).setToolTip(cells[4])
            if m["id"] == self._sel_id:
                sel_row = r
        self._table.resizeRowsToContents()
        if sel_row is not None:
            self._table.selectRow(sel_row)
        self._table.blockSignals(False)

    # ---- overlay -----------------------------------------------------------------
    def _paint_overlay(self, p, tw) -> None:
        def pen(col, w=2.0, style=Qt.PenStyle.SolidLine):
            q = QPen(QColor(col))
            q.setWidthF(w)
            q.setStyle(style)
            p.setPen(q)

        def poly(pts, close=False):
            for a, b in zip(pts[:-1], pts[1:]):
                p.drawLine(tw(*a), tw(*b))
            if close and len(pts) >= 3:
                p.drawLine(tw(*pts[-1]), tw(*pts[0]))

        def dot(q, r=3.5):
            p.drawEllipse(tw(*q), r, r)

        def label(q, txt, col="#ffff00"):
            f = QFont()
            f.setPointSize(10)
            f.setBold(True)
            p.setFont(f)
            w = tw(*q)
            p.setPen(QColor(0, 0, 0))
            p.drawText(QPointF(w.x() + 7, w.y() - 5), txt)
            p.setPen(QColor(col))
            p.drawText(QPointF(w.x() + 6, w.y() - 6), txt)

        # calibration (on its frame), pending first
        for c, col in ((self._pending, "#ff9900"), (self._cal(), "#33ccff")):
            if c is None or c.get("frame") != self._frame:
                continue
            pen(col)
            if c["method"] == "two_point":
                poly([c["p0"], c["p1"]])
                dot(c["p0"])
                dot(c["p1"])
            elif c["method"] == "catheter":
                pen(col, 1.5, Qt.PenStyle.DashLine)
                poly(c["polygon"], close=True)
                pen(col, 1.5)
                poly(c["centerline"])
        # measurements of this frame
        for m in self._state[self._pi]["meas"]:
            if m["frame"] != self._frame:
                continue
            sel = m["id"] == self._sel_id
            col = "#00ff66" if not sel else "#ff3399"
            pen(col, 2.5 if sel else 1.8)
            pts = m["points"]
            poly(pts)
            for q in pts:
                dot(q, 2.5)
            if m["type"] == "diameter":
                pen("#ffff00", 2.5)
                for e in m.get("edges") or []:
                    dot(e, 3.0)
                if m.get("corrected_line"):
                    pen("#ff9900", 1.2, Qt.PenStyle.DashLine)
                    poly(m["corrected_line"])
            v = self._value_mm(m)
            if v is not None:
                label(pts[-1], f"#{m['id']} {v:.2f} mm")
        # in progress
        if self._tool and self._pts:
            pen("#ff9900", 1.5)
            poly(self._pts, close=False)
            for q in self._pts:
                dot(q)
            if self._mouse is not None:
                pen("#ff9900", 1.0, Qt.PenStyle.DashLine)
                p.drawLine(tw(*self._pts[-1]), tw(*self._mouse))

    # ---- save / load ---------------------------------------------------------------
    def to_dict(self) -> dict:
        planes = []
        for i, (p, st) in enumerate(zip(self._planes, self._state)):
            meas = []
            for m in st["meas"]:
                r = dict(m)
                r["value_mm"] = self._value_mm(m, i)
                r["note"] = self.note_text(m, i)
                if m.get("corrected_px") is not None and self._mmpp(i):
                    r["corrected_mm"] = m["corrected_px"] * self._mmpp(i)
                meas.append(r)
            planes.append({"index": int(p["index"]), "name": str(p.get("name", "")),
                           "path": str(getattr(p["plane"], "path", "") or ""),
                           "calibration": st["cal"], "measurements": meas})
        return {"format": FORMAT, "version": 1,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "series": {"series_uid": self._uid, "title": self._title,
                           "modality": self._modality},
                "planes": planes}

    def default_path(self) -> str:
        d = ""
        for p in self._planes:
            pp = getattr(p["plane"], "path", "") or ""
            if pp:
                d = os.path.dirname(pp)
                break
        return os.path.join(d, SUFFIX) if d else SUFFIX

    @staticmethod
    def with_suffix(path: str) -> str:
        if path.lower().endswith(SUFFIX.lower()):
            return path
        if path.lower().endswith(".json"):
            path = path[:-5]
        return path + SUFFIX

    def save(self, path=None) -> str:
        if not any(st["cal"] or st["meas"] for st in self._state):
            QMessageBox.information(self, t("Cal-Measure"),
                                    t("保存する内容がありません。"))
            return ""
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, t("Cal-Measure を保存"), self._path or self.default_path(),
                t("Cal-Measure (*.CalMeas.json)"))
            if not path:
                return ""
        path = self.with_suffix(path)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        except OSError as exc:
            QMessageBox.warning(self, t("Cal-Measure"),
                                t("保存に失敗しました: {e}", e=str(exc)))
            return ""
        self._path = path
        self._dirty = False
        self._hint.setText(t("保存しました: {p}", p=os.path.basename(path)))
        return path

    def load(self, path=None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, t("Cal-Measure を開く"), os.path.dirname(self.default_path()),
                t("Cal-Measure (*.CalMeas.json)"))
            if not path:
                return False
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, t("Cal-Measure"),
                                t("読み込めませんでした: {e}", e=str(exc)))
            return False
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            QMessageBox.warning(self, t("Cal-Measure"),
                                t("Cal-Measure の形式ではありません。"))
            return False
        uid = (data.get("series") or {}).get("series_uid", "")
        if uid and self._uid and uid != self._uid:
            if QMessageBox.question(
                    self, t("Cal-Measure"),
                    t("別のシリーズで保存された結果です。読み込みますか？")) \
                    != QMessageBox.StandardButton.Yes:
                return False
        by_index = {int(p["index"]): i for i, p in enumerate(self._planes)}
        n = 0
        for pl in data.get("planes") or []:
            i = by_index.get(int(pl.get("index", -1)))
            if i is None:
                continue
            ms = []
            for m in pl.get("measurements") or []:
                m = {k: v for k, v in m.items()
                     if k not in ("value_mm", "note", "corrected_mm")}
                ms.append(m)
                self._next_id = max(self._next_id, int(m.get("id", 0)) + 1)
            self._state[i] = {"cal": pl.get("calibration"), "meas": ms}
            n += 1
        self._path = path
        self._dirty = False
        self._refresh()
        self._hint.setText(t("読み込みました: {p}", p=os.path.basename(path)))
        return n > 0

    def export_csv(self, path=None) -> str:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, t("CSV を保存"),
                self.with_suffix(self._path or self.default_path())[:-5] + ".csv",
                t("CSV (*.csv)"))
            if not path:
                return ""
        rows = [["plane", "id", "type", "value_mm", "frame", "calibration",
                 "mm_per_px", "note"]]
        for i, (p, st) in enumerate(zip(self._planes, self._state)):
            for m in st["meas"]:
                v = self._value_mm(m, i)
                rows.append([p.get("name", ""), m["id"], m["type"],
                             "" if v is None else f"{v:.3f}", m["frame"] + 1,
                             (st["cal"] or {}).get("method", ""),
                             "" if self._mmpp(i) is None else
                             f"{self._mmpp(i):.5f}", self.note_text(m, i)])
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerows(rows)
        except OSError as exc:
            QMessageBox.warning(self, t("Cal-Measure"),
                                t("保存に失敗しました: {e}", e=str(exc)))
            return ""
        return path

    def closeEvent(self, e):
        if self._dirty:
            r = QMessageBox.question(
                self, t("Cal-Measure"),
                t("保存していない結果があります。保存しますか？"),
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel)
            if r == QMessageBox.StandardButton.Cancel:
                e.ignore()
                return
            if r == QMessageBox.StandardButton.Save and not self.save():
                e.ignore()
                return
        super().closeEvent(e)
