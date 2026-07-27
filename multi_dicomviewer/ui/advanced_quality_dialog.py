"""Advanced image-quality tuning for the 2-D (XA / IVUS) viewer.

Fine-grained sliders — bilateral denoise strength, unsharp sharpening and CLAHE
local-contrast — beyond the simple S-Cine / S-Zoom / Denoise toggles. A live
before/after preview on a built-in sample shows the effect as you drag. CT is
NOT affected (its look is driven by HU window / colour map instead)."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.core import image_quality, settings
from multi_dicomviewer.i18n import t
from multi_dicomviewer.viewers.image_canvas import to_qimage


def _sample_image() -> np.ndarray:
    """A deterministic angio-like grayscale: gently varying background + two
    wavy dark 'vessels', with added noise — so denoise (calms noise), sharpen
    (crisps edges) and CLAHE (lifts local contrast) are all visible."""
    h = w = 240
    yy, xx = np.mgrid[0:h, 0:w]
    img = (120.0 + 35.0 * np.sin(xx / 34.0) + 20.0 * np.cos(yy / 60.0))
    for x0, amp, freq, wdt, dark in ((70, 26, 0.045, 3.2, 95),
                                     (150, 16, 0.075, 2.2, 80),
                                     (110, 40, 0.03, 1.6, 70)):
        cx = x0 + amp * np.sin(yy * freq)
        img -= dark * np.exp(-((xx - cx) ** 2) / (2 * wdt ** 2))
    rng = np.random.default_rng(0)
    img = img + rng.normal(0.0, 13.0, (h, w))
    return np.clip(img, 0, 255).astype(np.uint8)


class AdvancedQualityDialog(QDialog):
    """Sliders for denoise / sharpen / CLAHE with a live before/after preview.
    *params* seeds the sliders; values() returns the chosen params."""

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Advanced image quality (XA / IVUS)"))
        self._sample = _sample_image()
        root = QVBoxLayout(self)

        have = image_quality.available()
        note = QLabel(
            t("Fine-tune the angio / IVUS enhancement. Applies only while "
              "'Denoise' is ON in Settings. CT is not affected.")
            + ("" if have else t("  (OpenCV unavailable — preview disabled)")))
        note.setWordWrap(True)
        root.addWidget(note)

        # ---- live before/after preview -----------------------------------
        prow = QGridLayout()
        self._before = QLabel()
        self._after = QLabel()
        for lbl, w_ in ((t("Original"), self._before), (t("Adjusted"), self._after)):
            w_.setFixedSize(240, 240)
            w_.setAlignment(Qt.AlignmentFlag.AlignCenter)
            w_.setStyleSheet("background:#000;")
        prow.addWidget(QLabel(t("Original")), 0, 0, Qt.AlignmentFlag.AlignCenter)
        prow.addWidget(QLabel(t("Adjusted")), 0, 1, Qt.AlignmentFlag.AlignCenter)
        prow.addWidget(self._before, 1, 0)
        prow.addWidget(self._after, 1, 1)
        root.addLayout(prow)
        self._before.setPixmap(QPixmap.fromImage(to_qimage(self._sample)))

        # ---- sliders -------------------------------------------------------
        grid = QGridLayout()
        self._sliders: dict[str, QSlider] = {}
        self._vals: dict[str, QLabel] = {}
        # (key, label, slider min, slider max, scale so value = slider/scale)
        rows = (
            ("denoise", t("Denoise strength"), 0, 150, 1.0),
            ("sharpen", t("Sharpen"), 0, 200, 1.0),
            ("clahe", t("Local contrast (CLAHE)"), 0, 40, 10.0),
        )
        self._scale = {k: sc for k, _l, _lo, _hi, sc in rows}
        for r, (key, label, lo, hi, scale) in enumerate(rows):
            grid.addWidget(QLabel(label), r, 0)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(int(round(float(params.get(key, 0)) * scale)))
            sl.setEnabled(have)
            sl.valueChanged.connect(self._refresh_preview)
            self._sliders[key] = sl
            grid.addWidget(sl, r, 1)
            vl = QLabel("")
            vl.setMinimumWidth(44)
            self._vals[key] = vl
            grid.addWidget(vl, r, 2)
        root.addLayout(grid)

        rrow = QVBoxLayout()
        reset = QPushButton(t("Reset (classic Denoise)"))
        reset.clicked.connect(self._reset)
        rrow.addWidget(reset)
        root.addLayout(rrow)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self._refresh_preview()

    def _reset(self) -> None:
        # Classic single-step Denoise: sigma 50, no sharpen, no CLAHE.
        for key, val in (("denoise", 50), ("sharpen", 0), ("clahe", 0)):
            self._sliders[key].setValue(int(round(val * self._scale[key])))

    def values(self) -> dict:
        return {k: sl.value() / self._scale[k]
                for k, sl in self._sliders.items()}

    def _refresh_preview(self) -> None:
        v = self.values()
        self._vals["denoise"].setText(f"{v['denoise']:.0f}")
        self._vals["sharpen"].setText(f"{v['sharpen']:.0f}%")
        self._vals["clahe"].setText(f"{v['clahe']:.1f}")
        if not image_quality.available():
            return
        out = image_quality.enhance(
            self._sample, v["denoise"], v["sharpen"], v["clahe"])
        if out is not None:
            self._after.setPixmap(QPixmap.fromImage(to_qimage(out)))
