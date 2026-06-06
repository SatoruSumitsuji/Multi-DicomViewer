"""Shared 'DICOM tag overlay text size' control.

A single point-size value drives the DICOM-tag overlay font in EVERY viewer
(XA / IVUS / US via ImageCanvas, CT via VTK or pygfx) so the overlay reads the
same size regardless of modality. Each viewer shows a small slider stacked
ABOVE its "DICOM Tags…" button; the shell (MainWindow) keeps them in sync by
broadcasting the value to all viewers.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QPushButton, QSlider, QVBoxLayout, QWidget

#: DICOM-tag overlay font, in points. One default for every modality.
TAG_FONT_PT_DEFAULT = 11
TAG_FONT_PT_MIN = 7
TAG_FONT_PT_MAX = 16
#: Slider width — wide enough to operate comfortably (the reason it sits ABOVE
#: the button rather than cramped beside it).
_SLIDER_W = 120

#: DICOM-tag overlay typeface. Japanese-capable (tags can contain Japanese) and
#: shared by every viewer so the overlay reads the same. Yu Gothic Medium where
#: the font file is present. VTK (the Windows CT viewer) loads the SAME file via
#: SetFontFile + SetFontFamily(_VTK_FONT_FILE); the Qt viewers load it through
#: addApplicationFont so the family resolves regardless of font enumeration.
# Meiryo, not Yu Gothic: VTK's FreeType renders Yu Gothic's .ttc blank but
# loads Meiryo's fine. Meiryo is Japanese-capable and was the user's accepted
# alternative; the Qt viewers load the SAME file so every viewer matches.
TAG_FONT_FILE = r"C:\Windows\Fonts\meiryo.ttc"   # Meiryo
#: vtkTextProperty.SetFontFamily() value meaning "use SetFontFile" (VTK_FONT_FILE).
VTK_FONT_FILE = 4

_resolved_family: str | None = None


def overlay_font_family() -> str:
    """Resolved family name for the DICOM-tag overlay — Yu Gothic Medium when
    its font file is present, else a name Qt substitutes (still rendering
    Japanese via its glyph fallback). Resolved once."""
    global _resolved_family
    if _resolved_family is None:
        fam = ""
        if os.path.exists(TAG_FONT_FILE):
            fid = QFontDatabase.addApplicationFont(TAG_FONT_FILE)
            if fid >= 0:
                names = QFontDatabase.applicationFontFamilies(fid)
                if names:
                    fam = names[0]
        if not fam:
            # No Meiryo file (macOS/Linux): pick an installed Japanese-capable
            # family so tags still render deliberately (incl. Japanese).
            available = set(QFontDatabase.families())
            for cand in ("Meiryo", "Hiragino Sans",
                         "Hiragino Kaku Gothic ProN", "Yu Gothic",
                         "Noto Sans CJK JP"):
                if cand in available:
                    fam = cand
                    break
        _resolved_family = fam or "sans-serif"
    return _resolved_family


def overlay_qfont(pt, bold: bool = False) -> QFont:
    """QFont for the DICOM-tag overlay at *pt* points (Meiryo, Japanese-capable
    — the same font the VTK CT viewer loads from its file)."""
    f = QFont(overlay_font_family())
    f.setPointSize(int(pt))
    f.setBold(bool(bold))
    return f


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
