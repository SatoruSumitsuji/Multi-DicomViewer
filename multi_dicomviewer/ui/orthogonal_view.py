"""Orthogonal-View tool.

Clones the active XA pane's currently-shown frame into its own window
and lets the user pick two points on the image. The tool then reports
the two C-arm angles (LAO/RAO + CRA/CAU) whose viewing direction is
orthogonal to the picked vector — i.e. the two angiographic projections
that would see the vessel the user just outlined in plane.

Only meaningful when the active pane is XA in 1×1 layout, so the shell
gates the menu item on that.

Math
----
PositionerPrimaryAngle β (LAO+ / RAO-) is rotation about the patient
cranio-caudal axis (z in LPS); PositionerSecondaryAngle α (CRA+ / CAU-)
is rotation about the L-R axis (x). The X-ray beam direction (source →
detector) in LPS is

    beam(β, α) = (-sin β · cos α,  cos β · cos α,  -sin α).

Image axes in LPS (u = image-right per mm, v = image-down per mm):

    u_right(β, α) = ( cos β,           sin β,           0    )
    v_down (β, α) = ( sin β · sin α,  -cos β · sin α,  -cos α)

A picked image-space vector V = (du, dv) corresponds to a 3-D direction
V_3d = du·u + dv·v. Its two perpendiculars in the image plane,
(-dv, du) and (dv, -du), map to 3-D directions n_red and n_blue (the
on-screen red and blue arrows respectively). To compute the C-arm
angle whose BEAM gives the "view from the red side", we use n_blue —
the source sits on the red side, so the beam (source → detector) goes
toward the blue side. Likewise the blue-label angle is from n_red.
Each beam direction n is recovered as

    sin α = -nz                         → α = asin(-nz)
    tan β = -nx / ny  (when cos α ≠ 0)  → β = atan2(-nx, ny)
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _image_axes_lps(beta_deg: float, alpha_deg: float):
    """(u_right, v_down) unit vectors in patient LPS for a C-arm at the
    given primary / secondary angles. See module docstring."""
    b = math.radians(beta_deg)
    a = math.radians(alpha_deg)
    cb, sb = math.cos(b), math.sin(b)
    ca, sa = math.cos(a), math.sin(a)
    u_right = np.array([cb, sb, 0.0])
    v_down = np.array([sb * sa, -cb * sa, -ca])
    return u_right, v_down


def _beam_angles_from(n) -> tuple[float, float]:
    """C-arm (β, α) whose beam direction equals unit vector *n* in LPS."""
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    sa = max(-1.0, min(1.0, -nz))
    alpha = math.degrees(math.asin(sa))
    if abs(math.cos(math.radians(alpha))) < 1e-6:
        beta = 0.0                              # gimbal-lock: β undefined
    else:
        beta = math.degrees(math.atan2(-nx, ny))
    return beta, alpha


def _format_angle(beta: float, alpha: float) -> str:
    pi_ = int(round(beta))
    si_ = int(round(alpha))
    lao = f"LAO {pi_}" if pi_ >= 0 else f"RAO {-pi_}"
    cra = f"CRA {si_}" if si_ >= 0 else f"CAU {-si_}"
    return f"{lao} / {cra}"


class _OrthoCanvas(QWidget):
    """Fit-to-window canvas showing the cloned XA frame, two pickable
    points and the two orthogonal-view arrows."""

    point_added = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._qimg: QImage | None = None
        self._img_size = (0, 0)
        self.spacing_mm = None
        self.points: list[tuple[float, float]] = []
        self.angle1: tuple[float, float] | None = None  # red arrow
        self.angle2: tuple[float, float] | None = None  # blue arrow
        self.cur_angle_text = ""
        # View-only canvases (panes whose source DICOM lacks C-arm
        # positioner angles) ignore clicks so the user can't drop
        # picks where angles can't be computed.
        self._pickable = True

    def set_pickable(self, on: bool) -> None:
        self._pickable = bool(on)
        self.setCursor(
            Qt.CursorShape.CrossCursor if on
            else Qt.CursorShape.ArrowCursor
        )

    def set_frame(self, qimg: QImage | None, spacing_mm) -> None:
        self._qimg = qimg.copy() if qimg is not None else None
        self._img_size = (
            (qimg.width(), qimg.height()) if qimg is not None else (0, 0)
        )
        self.spacing_mm = spacing_mm
        self.update()

    def set_current_angle_text(self, text: str) -> None:
        self.cur_angle_text = text
        self.update()

    def clear_picks(self) -> None:
        self.points.clear()
        self.angle1 = None
        self.angle2 = None
        self.update()

    # ----------------------------------------------------- coord helpers
    def _draw_rect(self) -> QRect:
        w, h = self._img_size
        if w == 0 or h == 0:
            return QRect()
        cw, ch = self.width(), self.height()
        scale = min(cw / w, ch / h)
        dw, dh = int(w * scale), int(h * scale)
        return QRect((cw - dw) // 2, (ch - dh) // 2, dw, dh)

    def _widget_to_image(self, p: QPoint):
        r = self._draw_rect()
        if not r.isValid() or not r.contains(p):
            return None
        w, h = self._img_size
        return (
            (p.x() - r.x()) / r.width() * w,
            (p.y() - r.y()) / r.height() * h,
        )

    def _image_to_widget(self, pt) -> QPoint:
        r = self._draw_rect()
        w, h = self._img_size
        return QPoint(
            int(r.x() + pt[0] / w * r.width()),
            int(r.y() + pt[1] / h * r.height()),
        )

    # ----------------------------------------------------- input
    def mousePressEvent(self, e):
        if not self._pickable:
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        pt = self._widget_to_image(e.position().toPoint())
        if pt is None:
            return
        if len(self.points) >= 2:
            # Third left-click restarts the pick.
            self.points = [pt]
            self.angle1 = None
            self.angle2 = None
        else:
            self.points.append(pt)
        self.point_added.emit()
        self.update()

    # ----------------------------------------------------- painting
    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0a0a0a"))
        if self._qimg is None:
            return
        r = self._draw_rect()
        p.drawImage(r, self._qimg)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        yellow = QColor(255, 217, 0)

        # Picked points + connecting dashed segment.
        p.setBrush(yellow)
        p.setPen(QPen(yellow, 1))
        for pt in self.points:
            wp = self._image_to_widget(pt)
            p.drawEllipse(wp, 6, 6)
        if len(self.points) >= 2:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255), 2, Qt.PenStyle.DashLine))
            a = self._image_to_widget(self.points[0])
            b = self._image_to_widget(self.points[1])
            p.drawLine(a, b)
            self._paint_perpendiculars(p, a, b)

        # Current image's own C-arm angle, big yellow at the bottom.
        if self.cur_angle_text:
            font = QFont()
            font.setPointSize(20)
            font.setBold(True)
            p.setFont(font)
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(self.cur_angle_text)
            x = (self.width() - tw) // 2
            y = self.height() - 20
            p.setPen(yellow)
            p.drawText(x, y, self.cur_angle_text)

    def _paint_perpendiculars(self, p: QPainter, a: QPoint, b: QPoint) -> None:
        ax, ay = a.x(), a.y()
        bx, by = b.x(), b.y()
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy)
        if L < 1.0:
            return
        nx1, ny1 = -dy / L, dx / L           # 90° CCW in widget space
        nx2, ny2 = dy / L, -dx / L           # opposite
        arrow_len = max(60.0, min(120.0, L))
        # Tip sits just off the picked line so the head doesn't overlap
        # the dashed segment itself.
        tip_gap = 4.0
        red = QColor(220, 50, 50)
        blue = QColor(70, 150, 230)
        # Arrows point TOWARD the picked vector: tail outside, head near
        # the line.
        self._paint_arrow(
            p,
            mx + nx1 * arrow_len, my + ny1 * arrow_len,
            mx + nx1 * tip_gap,   my + ny1 * tip_gap, red,
        )
        self._paint_arrow(
            p,
            mx + nx2 * arrow_len, my + ny2 * arrow_len,
            mx + nx2 * tip_gap,   my + ny2 * tip_gap, blue,
        )

        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        p.setFont(font)
        fm = p.fontMetrics()
        pad = 14
        # Clamp the label rect to stay inside the canvas. Without this,
        # a pick near the canvas edge places the label outside the
        # widget; Qt clips at the widget boundary so the label looks
        # "hidden by the next panel" in the biplane window.
        def _clamp(tx, ty, tw):
            margin = 6
            tx = max(margin, min(self.width() - tw - margin, tx))
            ty = max(fm.ascent() + margin,
                     min(self.height() - margin, ty))
            return tx, ty
        if self.angle1 is not None:
            txt = _format_angle(*self.angle1)
            tw = fm.horizontalAdvance(txt)
            tx = mx + nx1 * (arrow_len + pad) - tw / 2.0
            ty = my + ny1 * (arrow_len + pad) + fm.ascent() / 2.0
            tx, ty = _clamp(tx, ty, tw)
            self._paint_text_with_bg(p, int(tx), int(ty), txt, red)
        if self.angle2 is not None:
            txt = _format_angle(*self.angle2)
            tw = fm.horizontalAdvance(txt)
            tx = mx + nx2 * (arrow_len + pad) - tw / 2.0
            ty = my + ny2 * (arrow_len + pad) + fm.ascent() / 2.0
            tx, ty = _clamp(tx, ty, tw)
            self._paint_text_with_bg(p, int(tx), int(ty), txt, blue)

    def _paint_arrow(self, p: QPainter, x0, y0, x1, y1, color) -> None:
        p.setPen(QPen(color, 4))
        p.setBrush(color)
        p.drawLine(int(x0), int(y0), int(x1), int(y1))
        ang = math.atan2(y1 - y0, x1 - x0)
        s = 14
        poly = QPolygonF([
            QPointF(x1, y1),
            QPointF(x1 - s * math.cos(ang - 0.45),
                    y1 - s * math.sin(ang - 0.45)),
            QPointF(x1 - s * math.cos(ang + 0.45),
                    y1 - s * math.sin(ang + 0.45)),
        ])
        p.drawPolygon(poly)

    def _paint_text_with_bg(self, p: QPainter, x: int, y: int,
                            text: str, color: QColor) -> None:
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text)
        th = fm.height()
        bg = QRect(x - 4, y - fm.ascent() - 2, tw + 8, th + 4)
        p.fillRect(bg, QColor(0, 0, 0, 180))
        p.setPen(color)
        p.drawText(x, y, text)


class _OrthoPanel(QWidget):
    """One Orthogonal-View pane: canvas + its own current-angle line and
    red / blue result lines. Owns its own pick state so each plane of a
    biplane study computes its angles independently."""

    def __init__(self, qimg: QImage, spacing_mm, beta, alpha,
                 subtitle: str, pickable: bool = True, parent=None):
        super().__init__(parent)
        self._beta = beta
        self._alpha = alpha
        # A pane whose source has no PositionerPrimaryAngle/Secondary
        # tags shows up as VIEW-ONLY: the image is cloned so the user
        # still sees it side-by-side with the angio picks on the other
        # panes, but clicks are inert and the red/blue arrows stay "—".
        self._pickable = bool(pickable)

        col = QVBoxLayout(self)
        col.setContentsMargins(2, 2, 2, 2)
        col.setSpacing(2)

        # Header row: subtitle on the left, per-panel Clear on the right.
        # Each panel owning its own Clear lets the biplane window clear
        # Frontal / Lateral independently, which is what the user
        # actually wants — one side's picks shouldn't be wiped when the
        # other side needs a redo. (Single-panel mode gets the same
        # header for consistency.)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        sub = QLabel(subtitle)
        sub.setStyleSheet(
            "color:#cfe8ff; padding:2px 8px;"
            " font-size:12pt; font-weight:bold;"
        )
        head.addWidget(sub, 1)
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Clear picks and angles on THIS image only")
        clear_btn.clicked.connect(self.clear)
        clear_btn.setEnabled(self._pickable)
        head.addWidget(clear_btn)
        col.addLayout(head)

        self.canvas = _OrthoCanvas()
        self.canvas.set_pickable(self._pickable)
        self.canvas.point_added.connect(self._on_pick)
        col.addWidget(self.canvas, 1)

        if not self._pickable:
            cur = ("View only — no C-arm positioner angles in this "
                    "image's DICOM header")
        elif beta is not None and alpha is not None:
            cur = f"Current image: {_format_angle(beta, alpha)}"
        else:
            cur = ("Current image: (no Positioner angles in DICOM "
                    "header — results are relative to image axes only)")
        self._cur_lbl = QLabel(cur)
        self._cur_lbl.setStyleSheet(
            "color:#ffd700; padding:2px 8px;"
            " font-size:12pt; font-weight:bold;"
        )
        self._red_lbl = QLabel("→ —")
        self._red_lbl.setStyleSheet(
            "color:#ff4040; padding:2px 8px;"
            " font-size:13pt; font-weight:bold;"
        )
        self._blue_lbl = QLabel("→ —")
        self._blue_lbl.setStyleSheet(
            "color:#3aa0ff; padding:2px 8px;"
            " font-size:13pt; font-weight:bold;"
        )
        col.addWidget(self._cur_lbl)
        col.addWidget(self._red_lbl)
        col.addWidget(self._blue_lbl)

        self.canvas.set_frame(qimg, spacing_mm)
        if self._pickable and beta is not None and alpha is not None:
            self.canvas.set_current_angle_text(_format_angle(beta, alpha))

    def clear(self) -> None:
        self.canvas.clear_picks()
        self._red_lbl.setText("→ —")
        self._blue_lbl.setText("→ —")

    def _on_pick(self) -> None:
        pts = self.canvas.points
        if len(pts) < 2:
            self._red_lbl.setText("→ —")
            self._blue_lbl.setText("→ —")
            return
        sp = self.canvas.spacing_mm
        sy, sx = (float(sp[0]), float(sp[1])) if sp is not None else (1.0, 1.0)
        du = (pts[1][0] - pts[0][0]) * sx
        dv = (pts[1][1] - pts[0][1]) * sy
        L = math.hypot(du, dv)
        if L < 1e-6:
            return
        du /= L
        dv /= L
        if self._beta is None or self._alpha is None:
            self._red_lbl.setText(
                "→ (no Positioner angle in DICOM header)"
            )
            self._blue_lbl.setText("")
            return
        u, v = _image_axes_lps(self._beta, self._alpha)
        # The red arrow points TO one side of the picked vessel; the
        # "view from the red side" places the X-ray source on the red
        # side, so the beam (source → detector) is in the OPPOSITE
        # direction — toward the blue side. Same logic for blue.
        # Hence the red label's beam direction equals the blue arrow's
        # 3-D direction, and vice versa.
        n_red_arrow = -dv * u + du * v        # arrow tip direction (red)
        n_blue_arrow = dv * u - du * v        # arrow tip direction (blue)
        self.canvas.angle1 = _beam_angles_from(n_blue_arrow)   # red label
        self.canvas.angle2 = _beam_angles_from(n_red_arrow)    # blue label
        self.canvas.update()
        self._red_lbl.setText("→ " + _format_angle(*self.canvas.angle1))
        self._blue_lbl.setText("→ " + _format_angle(*self.canvas.angle2))


class OrthogonalViewWindow(QMainWindow):
    """Tool window for the Orthogonal-View workflow.

    Accepts one or two panels. With one (single-plane XA) the layout is
    identical to before; with two (biplane XA) the panels sit side by
    side and each computes its own red/blue C-arm angles independently
    from picks on its own image."""

    def __init__(self, panels, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Orthogonal-View — {title}")
        n = len(panels)
        # 4-pane case needs more horizontal room than 1-/2-pane;
        # height roughly matches the shell viewer.
        if n >= 3:
            self.resize(1600, 1000)
        elif n == 2:
            self.resize(1400, 820)
        else:
            self.resize(960, 820)

        central = QWidget()
        col = QVBoxLayout(central)
        col.setContentsMargins(2, 2, 2, 2)
        col.setSpacing(4)

        info = QLabel(
            "Click two points on the image to define a vector. "
            "The red and blue arrows mark the two C-arm angles whose "
            "view is orthogonal to that vector."
            + ("  (Multi-pane: each pane computes its own orthogonal "
               "angles independently — use each pane's own Clear "
               "button to clear just that side. Panes without C-arm "
               "angles in their DICOM header are view-only.)"
               if n > 1 else "")
        )
        info.setStyleSheet("color:#ccc; padding:4px;")
        info.setWordWrap(True)
        col.addWidget(info)

        self._panels: list[_OrthoPanel] = []
        # Up to 2 panes → single row; 3+ panes (2×2 shell layout) → 2×N
        # grid so the panels stay roughly square instead of slim columns.
        if n <= 2:
            pane_row = QHBoxLayout()
            pane_row.setSpacing(6)
            for spec in panels:
                panel = _OrthoPanel(
                    spec["qimg"], spec["spacing"],
                    spec["beta"], spec["alpha"],
                    spec.get("subtitle", ""),
                    pickable=spec.get("pickable", True),
                )
                self._panels.append(panel)
                pane_row.addWidget(panel, 1)
            col.addLayout(pane_row, 1)
        else:
            grid = QGridLayout()
            grid.setSpacing(6)
            cols = 2
            for i, spec in enumerate(panels):
                panel = _OrthoPanel(
                    spec["qimg"], spec["spacing"],
                    spec["beta"], spec["alpha"],
                    spec.get("subtitle", ""),
                    pickable=spec.get("pickable", True),
                )
                self._panels.append(panel)
                r, c = divmod(i, cols)
                grid.addWidget(panel, r, c)
            for c in range(cols):
                grid.setColumnStretch(c, 1)
            for r in range((n + cols - 1) // cols):
                grid.setRowStretch(r, 1)
            col.addLayout(grid, 1)

        self.setCentralWidget(central)
