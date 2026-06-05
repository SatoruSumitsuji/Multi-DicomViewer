"""Shared 'DICOM tag overlay text size' control.

A single point-size value drives the DICOM-tag overlay font in EVERY viewer
(XA / IVUS / US via ImageCanvas, CT via VTK or pygfx) so the overlay reads the
same size regardless of modality. Each viewer shows a small slider stacked
ABOVE its "DICOM Tags…" button; the shell (MainWindow) keeps them in sync by
broadcasting the value to all viewers.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton, QSlider, QVBoxLayout, QWidget

#: DICOM-tag overlay font, in points. One default for every modality.
TAG_FONT_PT_DEFAULT = 11
TAG_FONT_PT_MIN = 7
TAG_FONT_PT_MAX = 24
#: Slider width — wide enough to operate comfortably (the reason it sits ABOVE
#: the button rather than cramped beside it).
_SLIDER_W = 120


def build_tag_font_control(pt: int = TAG_FONT_PT_DEFAULT):
    """Build the stacked [size-slider / "DICOM Tags…" button] widget.

    Returns ``(container, slider, button)``. The caller wires the slider's
    ``valueChanged`` (to broadcast the new size) and the button's ``clicked``
    (to open the tag picker)."""
    container = QWidget()
    col = QVBoxLayout(container)
    col.setContentsMargins(0, 0, 0, 0)
    col.setSpacing(1)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(TAG_FONT_PT_MIN, TAG_FONT_PT_MAX)
    slider.setValue(int(pt))
    slider.setFixedWidth(_SLIDER_W)
    slider.setToolTip("DICOM tag text size")

    button = QPushButton("DICOM Tags…")
    button.setToolTip("Choose DICOM tags to overlay on the image")

    col.addWidget(slider)
    col.addWidget(button)
    return container, slider, button
