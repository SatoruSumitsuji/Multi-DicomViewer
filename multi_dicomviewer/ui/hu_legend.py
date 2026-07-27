"""HU colour-map legend strip for the ColorMap Setting dialog.

Three stacked bands across the HU axis (default −1000…2000) so the user can
verify the mapping at a glance:

  1. the enabled colour *groups* (each band drawn as its own colour segment,
     labelled with its HU range),
  2. the windowed **grayscale** strip (what the current W/L shows), and
  3. the resulting **colour** strip (grayscale with the band colours blended in,
     smoothed) — i.e. exactly what a colour-mode pane paints.

The colour strip is produced by a caller-supplied ``color_lut_fn(bands,
opacity, win, lvl) -> (N, 3) float`` so the VTK and pygfx viewers each feed
their own (identical) band LUT.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class HuLegend(QWidget):
    def __init__(self, color_lut_fn, hu_lo=-1000.0, hu_hi=2000.0, parent=None):
        super().__init__(parent)
        self._lut_fn = color_lut_fn
        self._lo, self._hi = float(hu_lo), float(hu_hi)
        self._bands: list = []
        self._opacity = 0.25
        self._win, self._lvl = 400.0, 40.0
        self.setMinimumHeight(112)
        self.setMinimumWidth(360)

    def set_params(self, bands, opacity, win, lvl) -> None:
        self._bands = [dict(b) for b in bands]
        self._opacity = float(opacity)
        self._win, self._lvl = float(win), float(lvl)
        self.update()

    # ------------------------------------------------------------ paint
    def _x(self, hu: float, w: int) -> int:
        return int(round((hu - self._lo) / (self._hi - self._lo) * (w - 1)))

    def _strip_image(self, rgb: np.ndarray) -> QImage:
        """(N,3) float 0..1 → a 1-px-tall RGB QImage (scaled to the row on
        draw)."""
        arr = np.ascontiguousarray(
            np.clip(rgb, 0, 1) * 255.0).astype(np.uint8)[None, :, :]
        n = arr.shape[1]
        return QImage(arr.data, n, 1, 3 * n, QImage.Format.Format_RGB888).copy()

    def paintEvent(self, _e):                        # noqa: N802 (Qt override)
        w = self.width()
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#202020"))
        n = 512
        hu = self._lo + (self._hi - self._lo) * np.arange(n) / (n - 1)

        # ---- row geometry ------------------------------------------------
        y = 2
        grp_h, strip_h, gap = 22, 26, 3
        # 1) colour GROUP segments + range labels
        p.setPen(QPen(QColor("#ddd")))
        for b in self._bands:
            if not b.get("on"):
                continue
            x0 = self._x(b["lo"], w)
            x1 = self._x(b["hi"], w)
            c = b["rgb"]
            p.fillRect(QRect(x0, y, max(1, x1 - x0), grp_h),
                       QColor.fromRgbF(c[0], c[1], c[2]))
        p.setPen(QPen(QColor("#000")))
        p.drawRect(QRect(0, y, w - 1, grp_h))
        # small range text (only where the segment is wide enough)
        p.setPen(QPen(QColor("#111")))
        f = p.font(); f.setPointSize(7); p.setFont(f)
        for b in self._bands:
            if not b.get("on"):
                continue
            x0 = self._x(b["lo"], w); x1 = self._x(b["hi"], w)
            if x1 - x0 > 34:
                p.drawText(QRect(x0 + 2, y, x1 - x0 - 2, grp_h),
                           Qt.AlignmentFlag.AlignCenter,
                           f"{b['lo']}–{b['hi']}")

        # 2) windowed grayscale strip
        y += grp_h + gap
        glo = self._lvl - self._win / 2.0
        span = max(1e-6, self._win)
        g = np.clip((hu - glo) / span, 0.0, 1.0)
        gimg = self._strip_image(np.stack([g, g, g], axis=1))
        p.drawImage(QRect(0, y, w, strip_h), gimg)
        p.setPen(QPen(QColor("#000"))); p.drawRect(QRect(0, y, w - 1, strip_h))

        # 3) colour result strip
        y += strip_h + gap
        try:
            rgb = np.asarray(self._lut_fn(self._bands, self._opacity,
                                          self._win, self._lvl), float)[:, :3]
        except Exception:
            rgb = np.stack([g, g, g], axis=1)
        p.drawImage(QRect(0, y, w, strip_h), self._strip_image(rgb))
        p.setPen(QPen(QColor("#000"))); p.drawRect(QRect(0, y, w - 1, strip_h))

        # ---- HU tick labels along the bottom -----------------------------
        y += strip_h + 1
        p.setPen(QPen(QColor("#ccc")))
        f2 = p.font(); f2.setPointSize(7); p.setFont(f2)
        ticks = sorted({-1000, 0, 50, 250, 350, 700, 850, 2000}
                       | {b["lo"] for b in self._bands if b.get("on")}
                       | {b["hi"] for b in self._bands if b.get("on")})
        for hv in ticks:
            if hv < self._lo or hv > self._hi:
                continue
            x = self._x(hv, w)
            p.drawLine(x, y - strip_h - gap - 1, x, y)
            r = QRect(x - 24, y, 48, 12)
            p.drawText(r, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                       str(int(hv)))
        p.end()
