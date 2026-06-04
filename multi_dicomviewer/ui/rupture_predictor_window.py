"""Native Rupture-Predictor — adventitia stretch-ratio tool (PyQt6).

Replaces the browser-based ``resources/Rupture-Predictor.html`` with a
fully in-app window so the viewer is self-contained (no external browser,
clean Mac build, immune to browser changes). The clinical math lives in
the Qt-free :mod:`multi_dicomviewer.core.rupture_math`; this module is the
UI: an interactive canvas (image + point picking + overlay) plus a side
panel of results, mirroring the HTML workflow and Japanese prompts.

Two entry shapes, matching the old hand-off:
- IVUS: constructed with an ``XAPlane`` + current frame index, giving a
  frame stepper that decodes frames lazily across the whole pull-back.
- XA / snapshot: constructed with a single ``QImage``.

When DICOM pixel calibration is supplied (``calib``), the CH/CV manual
calibration steps are skipped and the workflow starts at A1.
"""
from __future__ import annotations

import math

from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QImage, QPainter, QPainterPath, QPen,
                         QPolygonF)
from PyQt6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QHBoxLayout,
                             QInputDialog, QLabel, QMainWindow, QMenu,
                             QMessageBox, QPushButton, QSlider, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from multi_dicomviewer.core import rupture_math as rm
from multi_dicomviewer.viewers.image_canvas import to_qimage

# Point colours, matching the HTML ``pointColors``.
_POINT_COLORS = {
    "CH1": "#33E6FF", "CH2": "#33E6FF", "CV1": "#33E6FF", "CV2": "#33E6FF",
    "A1": "#FFD700", "A2": "#FFD700", "AC": "#ADFF2F", "B": "#FF4040",
}
_CALIB_POINTS = ("CH1", "CH2", "CV1", "CV2")
_CLINICAL_POINTS = ("A1", "A2", "AC", "B")
_ALL_POINTS = _CALIB_POINTS + _CLINICAL_POINTS

# Workflow steps (verbatim Japanese strings from HTML 3379-3390).
_STEP_TAIL = ("このウインドウで点指定がしづらい場合はウインドウを動かすことが"
              "できます。点の位置は後で調整可能です。")
_WORKFLOW = [
    ("CH1", f"Horizontal Calibration (横のキャリブレーション) の1つ目の点 "
            f"CH1を指定してください。{_STEP_TAIL}", "point"),
    ("CH2", f"Horizontal Calibration (横のキャリブレーション) の2つ目の点 "
            f"CH2を指定してください。{_STEP_TAIL}", "point"),
    ("CH_INPUT", "CH (横のキャリブレーション) の実際の距離を入力してください",
     "input"),
    ("CV1", f"Vertical Calibration (縦のキャリブレーション) の1つ目の点 "
            f"CV1を指定してください。{_STEP_TAIL}", "point"),
    ("CV2", f"Vertical Calibration (縦のキャリブレーション) の2つ目の点 "
            f"CV2を指定してください。{_STEP_TAIL}", "point"),
    ("CV_INPUT", "CV (縦のキャリブレーション) の実際の距離を入力してください",
     "input"),
    ("A1", f"進展する外膜の断端の1つを指定してください。{_STEP_TAIL}", "point"),
    ("A2", f"進展する外膜のもう1つの断端を指定してください。{_STEP_TAIL}", "point"),
    ("AC", f"進展する外膜の真ん中の点（円弧の中心）ACを指定してください。"
           f"{_STEP_TAIL}", "point"),
    ("B", f"バルーンが拡張する一番血管側の点 Bを指定してください。{_STEP_TAIL}",
     "point"),
]

# Standard semi-compliant balloon sizes for the quick-pick combo (mm).
_COMMON_DIAMETERS = [round(0.75 + i * 0.25, 2) for i in range(22)]


class RuptureCanvas(QWidget):
    """Image canvas with named-point picking and the stretch-ratio overlay.

    Transform / zoom / pan are ported from
    :class:`multi_dicomviewer.viewers.image_canvas.ImageCanvas` (the
    debugged fit-to-window + zoom-to-cursor core); the interaction model is
    bespoke (ordered, named points) so we don't subclass.
    """

    point_placed = pyqtSignal(str)   # a workflow point was just placed
    point_moved = pyqtSignal(str)    # an existing point was dragged / nudged

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)

        self._qimg: QImage | None = None
        self._img_size = (0, 0)

        self.points: dict[str, tuple[float, float] | None] = {
            n: None for n in _ALL_POINTS}
        self.skip_calib = False          # DICOM mode hides CH/CV points
        self.viz: rm.BalloonResult | None = None
        self.workflow_active: str | None = None
        self.hint_text: str = ""

        self._hover: str | None = None
        self._drag: str | None = None
        self._selected: str | None = None

        # Zoom / pan (ported from ImageCanvas).
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self._panning = False
        self._pan_anchor: tuple[float, float] | None = None

    # ---------------------------------------------------------------- image
    def set_qimage(self, qimg: QImage | None) -> None:
        self._qimg = qimg
        if qimg is not None and not qimg.isNull():
            self._img_size = (qimg.width(), qimg.height())
        self.update()

    # ----------------------------------------------------- coord transforms
    def _draw_rect(self) -> QRect:
        w, h = self._img_size
        if w == 0 or h == 0:
            return QRect()
        cw, ch = self.width(), self.height()
        scale = min(cw / w, ch / h) * self._zoom
        dw, dh = int(w * scale), int(h * scale)
        x = (cw - dw) // 2 + int(round(self._pan[0]))
        y = (ch - dh) // 2 + int(round(self._pan[1]))
        return QRect(x, y, dw, dh)

    def _widget_to_image(self, p: QPoint) -> tuple[float, float] | None:
        r = self._draw_rect()
        if not r.isValid() or not r.contains(p):
            return None
        w, h = self._img_size
        return ((p.x() - r.x()) / r.width() * w,
                (p.y() - r.y()) / r.height() * h)

    def _widget_to_image_f(self, x: float, y: float) -> tuple[float, float]:
        r = self._draw_rect()
        w, h = self._img_size
        if not r.isValid():
            return (0.0, 0.0)
        return ((x - r.x()) / r.width() * w, (y - r.y()) / r.height() * h)

    def _image_to_widget_f(self, pt: tuple[float, float]) -> tuple[float, float]:
        r = self._draw_rect()
        w, h = self._img_size
        if not r.isValid():
            return (0.0, 0.0)
        return (r.x() + pt[0] / w * r.width(), r.y() + pt[1] / h * r.height())

    def _image_to_widget(self, pt: tuple[float, float]) -> QPoint:
        wf = self._image_to_widget_f(pt)
        return QPoint(int(wf[0]), int(wf[1]))

    def _img_len_to_widget(self, length_px: float) -> float:
        """Image-pixel length → widget pixels (uniform scale)."""
        r = self._draw_rect()
        w, _ = self._img_size
        if not r.isValid() or w == 0:
            return length_px
        return length_px * r.width() / w

    # ----------------------------------------------------------- zoom / pan
    def _apply_zoom(self, factor: float, ax: float, ay: float) -> None:
        new_zoom = max(0.25, min(8.0, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-9:
            return
        anchor_img = self._widget_to_image_f(ax, ay)
        self._zoom = new_zoom
        after = self._image_to_widget_f(anchor_img)
        self._pan[0] += ax - after[0]
        self._pan[1] += ay - after[1]
        self._clamp_pan()
        self.update()

    def zoom_in(self) -> None:
        self._apply_zoom(1.25, self.width() / 2.0, self.height() / 2.0)

    def zoom_out(self) -> None:
        self._apply_zoom(1 / 1.25, self.width() / 2.0, self.height() / 2.0)

    def reset_zoom(self) -> None:
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self.update()

    def wheelEvent(self, e):
        pos = e.position()
        factor = 1.25 if e.angleDelta().y() > 0 else 1 / 1.25
        self._apply_zoom(factor, pos.x(), pos.y())

    def _clamp_pan(self) -> None:
        w, h = self._img_size
        if w == 0 or h == 0:
            return
        cw, ch = self.width(), self.height()
        base = min(cw / w, ch / h) * self._zoom
        dw, dh = w * base, h * base
        ox, oy = (cw - dw) / 2.0, (ch - dh) / 2.0
        self._pan[0] = max(cw / 2.0 - dw - ox, min(cw / 2.0 - ox, self._pan[0]))
        self._pan[1] = max(ch / 2.0 - dh - oy, min(ch / 2.0 - oy, self._pan[1]))

    # ------------------------------------------------------- hit testing
    def _visible_points(self):
        for name in _ALL_POINTS:
            if self.skip_calib and name in _CALIB_POINTS:
                continue
            p = self.points[name]
            if p is not None:
                yield name, p

    def _pick_point(self, sx: float, sy: float) -> str | None:
        """Name of the placed point within ~12 px of (sx, sy), or None."""
        for name, p in self._visible_points():
            w = self._image_to_widget(p)
            if math.hypot(w.x() - sx, w.y() - sy) < 12.0:
                return name
        return None

    # ------------------------------------------------------- mouse events
    def mousePressEvent(self, e):
        pos = e.position()
        sx, sy = pos.x(), pos.y()
        is_ctrl_pan = (
            e.button() == Qt.MouseButton.LeftButton
            and bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier))
        if e.button() == Qt.MouseButton.MiddleButton or is_ctrl_pan:
            self._panning = True
            self._pan_anchor = (sx, sy)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if e.button() == Qt.MouseButton.RightButton:
            self._context_menu(sx, sy)
            return

        if e.button() != Qt.MouseButton.LeftButton:
            return

        # Placing the current workflow point.
        if self.workflow_active is not None:
            pt = self._widget_to_image(QPoint(int(sx), int(sy)))
            if pt is None:
                return
            self.points[self.workflow_active] = pt
            self._selected = self.workflow_active
            name = self.workflow_active
            self.workflow_active = None
            self.update()
            self.point_placed.emit(name)
            return

        # Otherwise: start dragging an existing point.
        hit = self._pick_point(sx, sy)
        if hit is not None:
            self._drag = hit
            self._selected = hit
            self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        sx, sy = pos.x(), pos.y()
        if self._panning and self._pan_anchor is not None:
            self._pan[0] += sx - self._pan_anchor[0]
            self._pan[1] += sy - self._pan_anchor[1]
            self._pan_anchor = (sx, sy)
            self._clamp_pan()
            self.update()
            return
        if self._drag is not None:
            pt = self._widget_to_image(QPoint(int(sx), int(sy)))
            if pt is not None:
                self.points[self._drag] = pt
                self.viz = None            # geometry changed → clear overlay
                self.update()
                self.point_moved.emit(self._drag)
            return
        # Hover highlight.
        new_hover = self._pick_point(sx, sy)
        if new_hover != self._hover:
            self._hover = new_hover
            self.setCursor(Qt.CursorShape.OpenHandCursor if new_hover
                           else Qt.CursorShape.CrossCursor)
            self.update()

    def mouseReleaseEvent(self, _e):
        if self._panning:
            self._panning = False
            self._pan_anchor = None
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if self._drag is not None:
            self._drag = None
            self.update()

    def keyPressEvent(self, e):
        if self._selected is None or self.points.get(self._selected) is None:
            super().keyPressEvent(e)
            return
        step = 10.0 if (e.modifiers() & Qt.KeyboardModifier.ShiftModifier) else 1.0
        dx = dy = 0.0
        k = e.key()
        if k == Qt.Key.Key_Left:
            dx = -step
        elif k == Qt.Key.Key_Right:
            dx = step
        elif k == Qt.Key.Key_Up:
            dy = -step
        elif k == Qt.Key.Key_Down:
            dy = step
        else:
            super().keyPressEvent(e)
            return
        x, y = self.points[self._selected]
        self.points[self._selected] = (x + dx, y + dy)
        self.viz = None
        self.update()
        self.point_moved.emit(self._selected)

    def _context_menu(self, sx: float, sy: float) -> None:
        menu = QMenu(self)
        a_reset = menu.addAction("全ての点をリセット")
        a_restart = menu.addAction("A1 から指定し直す")
        chosen = menu.exec(self.mapToGlobal(QPoint(int(sx), int(sy))))
        win = self.window()
        if chosen == a_reset and hasattr(win, "reset_all"):
            win.reset_all()
        elif chosen == a_restart and hasattr(win, "restart_from_a1"):
            win.restart_from_a1()

    # ------------------------------------------------------------- arc util
    @staticmethod
    def _arc_image_points(center, radius, a_start, a_delta, n=64):
        pts = []
        for i in range(n + 1):
            a = a_start + a_delta * (i / n)
            pts.append((center[0] + radius * math.cos(a),
                        center[1] + radius * math.sin(a)))
        return pts

    def _polyline_widget(self, img_pts):
        poly = QPolygonF()
        for p in img_pts:
            wf = self._image_to_widget_f(p)
            poly.append(QPointF(wf[0], wf[1]))
        return poly

    # ------------------------------------------------------------- painting
    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0a0a0a"))
        if self._qimg is None or self._qimg.isNull():
            return
        r = self._draw_rect()
        p.drawImage(r, self._qimg)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        self._draw_calibration_lines(p)
        if self.viz is not None:
            self._draw_overlay(p, self.viz)
        self._draw_points(p)
        self._draw_hint(p)

    def _draw_calibration_lines(self, p: QPainter) -> None:
        if self.skip_calib:
            return
        pen = QPen(QColor("#33E6FF"), 1.5, Qt.PenStyle.DashLine)
        p.setPen(pen)
        for a, b in (("CH1", "CH2"), ("CV1", "CV2")):
            pa, pb = self.points[a], self.points[b]
            if pa is not None and pb is not None:
                p.drawLine(self._image_to_widget(pa), self._image_to_widget(pb))

    def _draw_overlay(self, p: QPainter, v: rm.BalloonResult) -> None:
        a1, a2 = self.points["A1"], self.points["A2"]
        ac, b = self.points["AC"], self.points["B"]
        if None in (a1, a2, ac, b):
            return
        w = self._image_to_widget

        def wpt(pt):
            wf = self._image_to_widget_f(pt)
            return QPointF(wf[0], wf[1])

        # Orange A1-A2 line.
        p.setPen(QPen(QColor("#FFA500"), 2))
        p.drawLine(w(a1), w(a2))

        # Black dashed B -> foot.
        p.setPen(QPen(QColor("#000000"), 1.5, Qt.PenStyle.DashLine))
        p.drawLine(w(b), w(v.foot_point))

        # Black solid B -> C.
        p.setPen(QPen(QColor("#000000"), 3))
        p.drawLine(w(b), w(v.virtual_center))

        # Light-gray dashed parallel-through-C line.
        ux = (a2[0] - a1[0])
        uy = (a2[1] - a1[1])
        ln = math.hypot(ux, uy) or 1.0
        ux, uy = ux / ln, uy / ln
        ext = max(self._img_size) * 1.0
        cpar1 = (v.virtual_center[0] - ux * ext, v.virtual_center[1] - uy * ext)
        cpar2 = (v.virtual_center[0] + ux * ext, v.virtual_center[1] + uy * ext)
        p.setPen(QPen(QColor(150, 150, 150, 160), 1, Qt.PenStyle.DashLine))
        p.drawLine(w(cpar1), w(cpar2))

        # Virtual balloon circle: dashed, 50%-transparent border so the
        # region reads as a subtle guide rather than a hard outline.
        cw = wpt(v.virtual_center)
        rad = self._img_len_to_widget(v.virtual_radius_px)
        p.setPen(QPen(QColor(255, 0, 0, 128), 1.5, Qt.PenStyle.DashLine))
        p.setBrush(QColor(255, 0, 0, 64))
        p.drawEllipse(cw, rad, rad)
        p.setBrush(Qt.BrushStyle.NoBrush)

        # Yellow dashed original arc A1 -> AC -> A2 (short paths each side).
        c = v.original_circle.center
        rr = v.original_circle.r
        ang_a1 = math.atan2(a1[1] - c[1], a1[0] - c[0])
        ang_ac = math.atan2(ac[1] - c[1], ac[0] - c[0])
        ang_a2 = math.atan2(a2[1] - c[1], a2[0] - c[0])
        d1 = rm._normalize_pi(ang_ac - ang_a1)
        d2 = rm._normalize_pi(ang_a2 - ang_ac)
        arc_pts = (self._arc_image_points(c, rr, ang_a1, d1, 48)
                   + self._arc_image_points(c, rr, ang_ac, d2, 48))
        p.setPen(QPen(QColor("#FFD700"), 3))
        p.drawPolyline(self._polyline_widget(arc_pts))

        # Red (or black) stretched path: A1->B1 + arc(B1->B2) + B2->A2.
        # B1/B2 are diametrically opposite, so the two B1->B2 arcs are equal
        # semicircles (the math kernel's length is the same either way). For
        # DRAWING, pick the semicircle that wraps the balloon on the side
        # AWAY from B — the stretched adventitia goes A1-B1-B2-A2 over the
        # top of the balloon, it must not bulge back through B.
        b1, b2 = v.b1, v.b2
        ab1 = math.atan2(b1[1] - v.virtual_center[1], b1[0] - v.virtual_center[0])
        ab2 = math.atan2(b2[1] - v.virtual_center[1], b2[0] - v.virtual_center[0])
        db = rm._normalize_pi(ab2 - ab1)
        alt = db - 2 * math.pi if db > 0 else db + 2 * math.pi

        def _arc_mid_dist_from_b(delta):
            am = ab1 + delta / 2.0
            mp = (v.virtual_center[0] + v.virtual_radius_px * math.cos(am),
                  v.virtual_center[1] + v.virtual_radius_px * math.sin(am))
            return rm.distance(mp, b)

        if _arc_mid_dist_from_b(alt) > _arc_mid_dist_from_b(db):
            db = alt
        arc_b = self._arc_image_points(v.virtual_center, v.virtual_radius_px,
                                       ab1, db, 48)
        # Arc is always red.
        p.setPen(QPen(QColor("#FF0000"), 3))
        p.drawPolyline(self._polyline_widget(arc_b))
        seg_color = "#FF0000" if v.include_a1b1a2b2 else "#000000"
        p.setPen(QPen(QColor(seg_color), 3))
        p.drawLine(w(a1), w(b1))
        p.drawLine(w(b2), w(a2))

        # foot (magenta), C (lime), B1/B2 (orange) dots + labels.
        self._dot(p, v.foot_point, "#FF40FF")
        self._dot(p, v.virtual_center, "#7CFC00")
        self._dot(p, b1, "#FFA500", "B1")
        self._dot(p, b2, "#FFA500", "B2")

    def _dot(self, p: QPainter, pt, color: str, label: str = "") -> None:
        w = self._image_to_widget(pt)
        p.setPen(QPen(QColor("#000000"), 1))
        p.setBrush(QColor(color))
        p.drawEllipse(w, 4, 4)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if label:
            self._draw_label(p, label, w + QPoint(7, -7))

    def _draw_points(self, p: QPainter) -> None:
        for name, pt in self._visible_points():
            w = self._image_to_widget(pt)
            color = QColor(_POINT_COLORS[name])
            base_r = 6
            r = int(base_r * 1.3) if name == self._hover else base_r
            hollow = name in _CALIB_POINTS
            if name == self._selected:
                p.setPen(QPen(QColor("#00FF00"), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(w, r + 4, r + 4)
                p.setPen(QPen(QColor("#FFFFFF"), 1.5))
                p.drawEllipse(w, r + 2, r + 2)
            if name == self._hover:
                p.setPen(QPen(QColor("#FFFF00"), 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(w, r + 3, r + 3)
            if hollow:
                p.setPen(QPen(color, 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(w, r, r)
                p.setPen(QPen(QColor("#FFFFFF"), 1))
                p.drawEllipse(w, max(1, r - 3), max(1, r - 3))
            else:
                p.setPen(QPen(QColor("#000000"), 1))
                p.setBrush(color)
                p.drawEllipse(w, r, r)
                p.setBrush(Qt.BrushStyle.NoBrush)
            self._draw_label(p, name, w + QPoint(8, -8))

    def _draw_label(self, p: QPainter, text: str, at: QPoint) -> None:
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        p.setFont(font)
        path = QPainterPath()
        path.addText(QPointF(at.x(), at.y()), font, text)
        p.setPen(QPen(QColor("#000000"), 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.fillPath(path, QColor("#FFFFFF"))

    def _draw_hint(self, p: QPainter) -> None:
        if not self.hint_text:
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        p.setFont(font)
        metrics = p.fontMetrics()
        tw = metrics.horizontalAdvance(self.hint_text)
        pad = 10
        bw = tw + pad * 2
        bh = metrics.height() + pad
        x = (self.width() - bw) // 2
        y = 8
        p.fillRect(QRect(x, y, bw, bh), QColor(0, 0, 0, 180))
        p.setPen(QColor("#FFD700"))
        p.drawText(QRect(x, y, bw, bh), Qt.AlignmentFlag.AlignCenter,
                   self.hint_text)


class _StepDialog(QDialog):
    """Modeless instruction dialog shown while the user clicks a point.

    Non-modal so the canvas stays interactive (pan/zoom between picks),
    matching the HTML's draggable dialog. Carries a Back button."""

    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Tool
                            | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("Rupture-Predictor")
        lay = QVBoxLayout(self)
        self._msg = QLabel("")
        self._msg.setWordWrap(True)
        self._msg.setMinimumWidth(360)
        lay.addWidget(self._msg)
        row = QHBoxLayout()
        row.addStretch(1)
        self._back = QPushButton("戻る")
        self._back.clicked.connect(self.back_requested.emit)
        row.addWidget(self._back)
        lay.addLayout(row)

    def show_message(self, text: str, can_back: bool) -> None:
        self._msg.setText(text)
        self._back.setEnabled(can_back)
        self.adjustSize()


class RupturePredictorWindow(QMainWindow):
    """Top-level Rupture-Predictor tool window.

    Parameters
    ----------
    plane : XAPlane | None
        For IVUS — gives the frame stepper its lazily-decoded frames.
    frame_index : int
        The frame the user was viewing (stepper start).
    qimage : QImage | None
        For XA / snapshot — a single displayed image.
    calib : tuple[float, float] | None
        ``(hpxmm, vpxmm)`` DICOM pixels-per-mm. When present the CH/CV
        manual calibration steps are skipped.
    """

    def __init__(self, *, plane=None, frame_index: int = 0,
                 qimage: QImage | None = None, calib=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rupture-Predictor")
        self.resize(1200, 800)

        self._plane = plane
        self._frame_index = int(frame_index)
        self._lock_index: int | None = None

        self._calib: rm.Calibration | None = None
        self._ch_mm: float | None = None
        self._cv_mm: float | None = None
        if calib is not None:
            hpxmm, vpxmm = calib
            self._calib = rm.calibration_dicom(hpxmm, vpxmm)

        # ----- canvas -----
        self.canvas = RuptureCanvas(self)
        self.canvas.skip_calib = self._calib is not None
        self.canvas.point_placed.connect(self._on_point_placed)
        self.canvas.point_moved.connect(self._on_point_moved)

        central = QWidget(self)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)

        left = QVBoxLayout()
        left.addWidget(self.canvas, 1)
        self._stepper = self._build_stepper() if plane is not None else None
        if self._stepper is not None:
            left.addWidget(self._stepper)
        left_w = QWidget()
        left_w.setLayout(left)
        outer.addWidget(left_w, 1)
        outer.addWidget(self._build_side_panel())
        self.setCentralWidget(central)

        # Workflow state.
        self._step_dialog = _StepDialog(self)
        self._step_dialog.back_requested.connect(self._on_back)
        self._wf_index = 0
        self._steps = self._build_steps()

        # Initial image.
        if plane is not None:
            self._show_frame(self._frame_index)
        elif qimage is not None:
            self.canvas.set_qimage(qimage)

        QTimer.singleShot(0, self._start_workflow)

    # --------------------------------------------------------- workflow defs
    def _build_steps(self):
        if self._calib is not None:
            return [s for s in _WORKFLOW if s[0] in _CLINICAL_POINTS]
        return list(_WORKFLOW)

    def _start_workflow(self) -> None:
        self._wf_index = 0
        self._run_step()

    def _run_step(self) -> None:
        if self._wf_index >= len(self._steps):
            self._finish_workflow()
            return
        name, message, kind = self._steps[self._wf_index]
        if kind == "input":
            self._run_input_step(name, message)
            return
        # point step
        self.canvas.workflow_active = name
        self.canvas.hint_text = f"{name} をクリックして指定"
        self.canvas.update()
        self._step_dialog.show_message(message, can_back=self._wf_index > 0)
        if not self._step_dialog.isVisible():
            self._position_step_dialog()
            self._step_dialog.show()

    def _run_input_step(self, name: str, message: str) -> None:
        val, ok = QInputDialog.getDouble(
            self, "Rupture-Predictor", message, 1.0, 0.01, 1000.0, 2)
        if not ok:
            # Treat cancel as Back.
            self._on_back()
            return
        if name == "CH_INPUT":
            self._ch_mm = val
        elif name == "CV_INPUT":
            self._cv_mm = val
        self._wf_index += 1
        self._run_step()

    def _on_back(self) -> None:
        if self._wf_index <= 0:
            return
        self._wf_index -= 1
        # Clear the point we are going back to re-pick (if any).
        name = self._steps[self._wf_index][0]
        if name in self.canvas.points:
            self.canvas.points[name] = None
            self.canvas.viz = None
            self.canvas.update()
        self._run_step()

    def _finish_workflow(self) -> None:
        self.canvas.workflow_active = None
        self.canvas.hint_text = ""
        self._step_dialog.hide()
        self.canvas.update()
        self._ensure_calibration()
        self._recompute()

    def _ensure_calibration(self) -> None:
        if self._calib is not None:
            return
        pts = self.canvas.points
        if (None not in (pts["CH1"], pts["CH2"], pts["CV1"], pts["CV2"])
                and self._ch_mm and self._cv_mm):
            self._calib = rm.calibration_manual(
                pts["CH1"], pts["CH2"], self._ch_mm,
                pts["CV1"], pts["CV2"], self._cv_mm)

    def _position_step_dialog(self) -> None:
        g = self.geometry()
        self._step_dialog.move(g.x() + 80, g.y() + 80)

    # ------------------------------------------------------------- recompute
    def _on_point_moved(self, _name: str) -> None:
        # Live recompute after a drag/nudge once everything is in place.
        if self._all_clinical_placed() and self._calib is not None:
            self._recompute()

    def _all_clinical_placed(self) -> bool:
        return all(self.canvas.points[n] is not None for n in _CLINICAL_POINTS)

    def _recompute(self) -> None:
        if self._calib is None or not self._all_clinical_placed():
            return
        pts = self.canvas.points
        avg = self._calib.avg
        diameter = self._diam_spin.value()
        res = rm.calculate_for_balloon_diameter(
            pts["A1"], pts["A2"], pts["AC"], pts["B"], diameter, avg)
        if res is None:
            QMessageBox.warning(self, "Rupture-Predictor",
                                "A1, AC, A2の点から円を計算できませんでした")
            return
        self.canvas.viz = res
        self.canvas.update()
        self._update_panel(res)

    # ------------------------------------------------------------- side panel
    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(340)
        lay = QVBoxLayout(panel)

        self._result_label = QLabel("Stretch Ratio: —")
        f = QFont()
        f.setPointSize(16)
        f.setBold(True)
        self._result_label.setFont(f)
        lay.addWidget(self._result_label)

        self._calib_label = QLabel("Calibration: —")
        lay.addWidget(self._calib_label)

        # Balloon diameter input + common-size combo.
        row = QHBoxLayout()
        row.addWidget(QLabel("バルーン直径(mm):"))
        self._diam_spin = QDoubleSpinBox()
        self._diam_spin.setRange(0.50, 20.00)
        self._diam_spin.setSingleStep(0.25)
        self._diam_spin.setValue(3.00)
        self._diam_spin.valueChanged.connect(self._on_diameter_changed)
        row.addWidget(self._diam_spin)
        self._diam_combo = QComboBox()
        self._diam_combo.addItem("選択", 0.0)
        for d in _COMMON_DIAMETERS:
            self._diam_combo.addItem(f"{d:.2f}", d)
        self._diam_combo.currentIndexChanged.connect(self._on_combo_changed)
        row.addWidget(self._diam_combo)
        lay.addLayout(row)

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        lay.addWidget(self._info_label)

        lay.addWidget(QLabel("径ごとの結果 (直径クリックで反映)"))
        self._results_table = QTableWidget(0, 3)
        self._results_table.setHorizontalHeaderLabels(
            ["直径(mm)", "進展外膜長(mm)", "Stretch Ratio"])
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.cellClicked.connect(self._on_result_clicked)
        lay.addWidget(self._results_table, 1)

        lay.addWidget(QLabel("目標伸展率→必要直径 (クリックで反映)"))
        self._rate_table = QTableWidget(0, 2)
        self._rate_table.setHorizontalHeaderLabels(["Stretch Ratio", "直径(mm)"])
        self._rate_table.verticalHeader().setVisible(False)
        self._rate_table.cellClicked.connect(self._on_rate_clicked)
        self._rate_table.setMaximumHeight(130)
        lay.addWidget(self._rate_table)

        btns = QHBoxLayout()
        b_reset = QPushButton("リセット")
        b_reset.clicked.connect(self.reset_all)
        b_zin = QPushButton("＋")
        b_zin.clicked.connect(self.canvas.zoom_in)
        b_zout = QPushButton("－")
        b_zout.clicked.connect(self.canvas.zoom_out)
        for b in (b_reset, b_zin, b_zout):
            btns.addWidget(b)
        lay.addLayout(btns)

        return panel

    @staticmethod
    def _ratio_colors(v: float) -> tuple[str, str]:
        """(text, background) chip colours for a stretch ratio — risk rises
        with the value: <1.5 blue, 1.5–1.8 yellow, ≥1.8 red."""
        if v < 1.5:
            return "#ffffff", "#1565c0"      # blue
        if v < 1.8:
            return "#000000", "#f5d100"      # yellow (dark text for contrast)
        return "#ffffff", "#c62828"          # red

    def _update_panel(self, res: rm.BalloonResult) -> None:
        # Make the number ~2.5× the label font (16 → 40 pt) and colour-code it
        # so the result reads at a glance from across the room.
        fg, bg = self._ratio_colors(res.stretch_ratio)
        self._result_label.setStyleSheet("")
        self._result_label.setText(
            "Stretch Ratio: "
            f"<span style='font-size:40pt; font-weight:bold; color:{fg}; "
            f"background-color:{bg};'>&nbsp;{res.stretch_ratio:.2f}&nbsp;</span>"
        )
        if self._calib is not None:
            self._calib_label.setText(
                f"H: {self._calib.hpxmm:.2f} px/mm   "
                f"V: {self._calib.vpxmm:.2f} px/mm")
        self._info_label.setText(
            f"元の弧長 (A1-AC-A2): {res.original_arc_len_mm:.2f} mm\n"
            f"角度 ∠A1-C-A2: {res.angle_a1ca2_deg:.1f}°\n"
            f"仮想バルーン直径: {res.balloon_diameter_mm:.2f} mm\n"
            f"伸展外膜長: {res.stretched_adventitia_len_mm:.2f} mm")
        self._fill_results_table()
        self._fill_rate_table()

    def _fill_results_table(self) -> None:
        pts = self.canvas.points
        avg = self._calib.avg
        rows = rm.results_table(pts["A1"], pts["A2"], pts["AC"], pts["B"], avg)
        cur = round(self._diam_spin.value(), 2)
        self._results_table.setRowCount(len(rows))
        for i, res in enumerate(rows):
            vals = [f"{res.balloon_diameter_mm:.2f}",
                    f"{res.stretched_adventitia_len_mm:.2f}",
                    f"{res.stretch_ratio:.2f}"]
            highlight = abs(res.balloon_diameter_mm - cur) < 1e-6
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                if highlight:
                    item.setBackground(QColor("#d4edda"))
                if c == 0:
                    # Diameter cell: clickable to load that diameter (blue,
                    # underlined link look); carry the exact value as data.
                    item.setData(Qt.ItemDataRole.UserRole,
                                 res.balloon_diameter_mm)
                    item.setForeground(QColor("#1565c0"))
                    fnt = item.font()
                    fnt.setUnderline(True)
                    item.setFont(fnt)
                    item.setToolTip("クリックでこの直径を反映")
                self._results_table.setItem(i, c, item)

    def _fill_rate_table(self) -> None:
        pts = self.canvas.points
        avg = self._calib.avg
        table = rm.expansion_rate_table(
            pts["A1"], pts["A2"], pts["AC"], pts["B"], avg)
        self._rate_table.setRowCount(len(table))
        for i, (rate, diam) in enumerate(table):
            self._rate_table.setItem(i, 0, QTableWidgetItem(f"{rate:.1f}"))
            txt = f"{diam:.2f}" if diam is not None else "範囲外"
            self._rate_table.setItem(i, 1, QTableWidgetItem(txt))

    def _on_diameter_changed(self, _v: float) -> None:
        self._recompute()

    def _on_combo_changed(self, _i: int) -> None:
        d = self._diam_combo.currentData()
        if d:
            self._diam_spin.setValue(float(d))

    def _on_result_clicked(self, row: int, _col: int) -> None:
        # Clicking any cell in a per-diameter row loads that row's diameter
        # into the balloon-diameter input; the spin's valueChanged then
        # recomputes the stretch ratio label and redraws the overlay figure.
        item = self._results_table.item(row, 0)
        if item is None:
            return
        d = item.data(Qt.ItemDataRole.UserRole)
        if d is None:
            try:
                d = float(item.text())
            except ValueError:
                return
        self._diam_spin.setValue(float(d))

    def _on_rate_clicked(self, row: int, _col: int) -> None:
        item = self._rate_table.item(row, 1)
        if item is None:
            return
        try:
            self._diam_spin.setValue(float(item.text()))
        except ValueError:
            pass

    # ----------------------------------------------------- public actions
    def reset_all(self) -> None:
        for n in _ALL_POINTS:
            self.canvas.points[n] = None
        self.canvas.viz = None
        self.canvas._selected = None
        self._lock_index = None
        self._update_lock_label()
        self.canvas.update()
        self._start_workflow()

    def restart_from_a1(self) -> None:
        for n in _CLINICAL_POINTS:
            self.canvas.points[n] = None
        self.canvas.viz = None
        self._lock_index = None
        self._update_lock_label()
        self.canvas.update()
        # Jump workflow to A1.
        for i, (name, _m, _k) in enumerate(self._steps):
            if name == "A1":
                self._wf_index = i
                break
        self._run_step()

    # ----------------------------------------------------- IVUS frame stepper
    def _build_stepper(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(4, 2, 4, 2)
        total = int(getattr(self._plane, "total_frames", 1))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, max(0, total - 1))
        self._slider.setValue(self._frame_index)
        self._slider.valueChanged.connect(self._show_frame)
        b_first = QPushButton("⏮")
        b_prev = QPushButton("◀")
        b_home = QPushButton("○")
        b_next = QPushButton("▶")
        b_last = QPushButton("⏭")
        b_first.clicked.connect(lambda: self._show_frame(0))
        b_prev.clicked.connect(lambda: self._show_frame(self._frame_index - 1))
        b_home.clicked.connect(self._goto_home_frame)
        b_next.clicked.connect(lambda: self._show_frame(self._frame_index + 1))
        b_last.clicked.connect(lambda: self._show_frame(total - 1))
        for b in (b_first, b_prev):
            lay.addWidget(b)
        lay.addWidget(self._slider, 1)
        for b in (b_next, b_last, b_home):
            lay.addWidget(b)
        self._frame_label = QLabel("")
        lay.addWidget(self._frame_label)
        self._update_lock_label()
        return bar

    def _goto_home_frame(self) -> None:
        self._show_frame(self._lock_index if self._lock_index is not None
                         else self._frame_index)

    def _show_frame(self, idx: int) -> None:
        if self._plane is None:
            return
        total = int(getattr(self._plane, "total_frames", 1))
        idx = max(0, min(int(idx), total - 1))
        self._frame_index = idx
        try:
            arr = self._plane.frame(idx)
        except Exception:
            return
        self.canvas.set_qimage(to_qimage(arr))
        if self._slider.value() != idx:
            self._slider.blockSignals(True)
            self._slider.setValue(idx)
            self._slider.blockSignals(False)
        self._update_lock_label()

    def _update_lock_label(self) -> None:
        if self._plane is None or not hasattr(self, "_frame_label"):
            return
        total = int(getattr(self._plane, "total_frames", 1))
        txt = f"{self._frame_index + 1}/{total}"
        if self._lock_index is not None:
            txt += f"  (Locked {self._lock_index + 1})"
            color = ("#1a7f37" if self._frame_index == self._lock_index
                     else "#c0392b")
            self._frame_label.setStyleSheet(f"color: {color};")
        else:
            self._frame_label.setStyleSheet("")
        self._frame_label.setText(txt)

    def _toast(self, text: str) -> None:
        self.statusBar().showMessage(text, 2200)

    def _on_point_placed(self, name: str) -> None:
        # Frame-lock check for IVUS: picks lock to the first point's frame.
        if self._plane is not None:
            if self._lock_index is None:
                self._lock_index = self._frame_index
                self._update_lock_label()
            elif self._frame_index != self._lock_index:
                # Reject: undo the placement and warn.
                self.canvas.points[name] = None
                self.canvas.workflow_active = name
                self.canvas.update()
                self._toast(
                    f"点は フレーム {self._lock_index + 1} にロックされています"
                    "。そのフレームに戻して指定してください。")
                return
        if (self._wf_index < len(self._steps)
                and self._steps[self._wf_index][0] == name):
            self._wf_index += 1
            self.canvas.hint_text = ""
            self._run_step()
