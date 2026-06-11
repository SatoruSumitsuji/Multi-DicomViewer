"""Single-frame still-image export (JPEG / PNG / TIFF).

Used by the right-click export action in every viewer (non-CT ImageCanvas,
IVUS long-axis, Windows VTK CT, Mac pygfx CT). Each viewer captures its own
on-screen view as a QImage (WYSIWYG — measurements, crosshair, tag text
included) and hands it here; this module owns the format menu, the save
dialog, the remembered output folder, filename sanitising, and the
per-format quality.

The right-click menu lists each format explicitly so the choice is made up
front (no second format dropdown in the dialog):
  * Export PNG (lossless)  — DEFAULT. Lossless, so the burned-in "Frame N"
        text and the thin measurement/crosshair lines stay crisp.
  * Export JPEG (95%)      — near-lossless; lower values mosquito-ring lines.
  * Export TIFF (lossless) — lossless; for publication / print use.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog, QMenu, QWidget

#: Remembered between calls so consecutive exports default to the last
#: folder the user chose (process-lifetime only).
_last_dir: str = ""

#: JPEG quality on Qt's 0-100 scale. 95 = near-lossless; keeps thin lines
#: and small text clean while staying far smaller than PNG/TIFF.
_JPEG_QUALITY = 95

_ILLEGAL = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')

#: Still-image formats, always offered (these save the clicked canvas).
_STILL_FORMATS = [
    ("Export PNG (lossless)", "png"),
    ("Export JPEG (95%)", "jpeg"),
    ("Export TIFF (lossless)", "tiff"),
]

# key -> (Qt format, default extension, save-quality, accepted extensions,
#         dialog filter)
_FORMATS = {
    "png": ("PNG", ".png", -1, (".png",), "PNG (*.png)"),
    "jpeg": ("JPEG", ".jpg", _JPEG_QUALITY, (".jpg", ".jpeg"),
             "JPEG (*.jpg *.jpeg)"),
    "tiff": ("TIFF", ".tif", -1, (".tif", ".tiff"), "TIFF (*.tif *.tiff)"),
}


def safe_basename(*parts: object) -> str:
    """Join non-empty parts with '_' into a Windows-safe filename stem."""
    chunks = []
    for p in parts:
        s = _ILLEGAL.sub("_", str(p or "")).strip()
        s = re.sub(r"\s+", "_", s).strip("._")
        if s:
            chunks.append(s)
    stem = "_".join(chunks)
    return stem or "image"


def pick_export_format(
    parent: Optional[QWidget],
    global_point: QPoint,
    include_dicom: bool = False,
    include_mp4: bool = False,
) -> Optional[str]:
    """Show the right-click export menu at *global_point*; return the chosen
    format key, or None if dismissed.

    Order: PNG, JPEG, TIFF, [DICOM], [MP4], CSV. PNG/JPEG/TIFF save the
    clicked canvas. DICOM (lossless original copy), MP4 (rendered cine) and
    CSV (DICOM-tag dump) are series/plane exports the host routes through the
    shell. DICOM/MP4 are only listed when the caller opts in — CT has no MP4,
    and the IVUS long-axis strip offers neither (still + CSV only)."""
    items = list(_STILL_FORMATS)
    if include_dicom:
        items.append(("Export DICOM (lossless)", "dicom"))
    if include_mp4:
        items.append(("Export MP4", "mp4"))
    items.append(("Export CSV (DICOM tags)", "csv"))
    menu = QMenu(parent)
    acts = [(menu.addAction(label), key) for label, key in items]
    chosen = menu.exec(global_point)
    for act, key in acts:
        if act is chosen:
            return key
    return None


def export_image_as(
    parent: Optional[QWidget],
    image: "QImage | QPixmap",
    fmt_key: str,
    default_basename: str = "image",
) -> Optional[str]:
    """Save *image* in the format named by *fmt_key* via a Save-As dialog
    pre-filtered to that one format. Returns the saved path, or None if the
    user cancelled or the write failed. *image* may be a QImage or QPixmap.
    """
    global _last_dir
    if isinstance(image, QPixmap):
        image = image.toImage()
    if image is None or image.isNull():
        return None
    fmt, ext, quality, accepted, filt = _FORMATS[fmt_key]

    start_dir = _last_dir or os.path.expanduser("~")
    start = os.path.join(start_dir, f"{safe_basename(default_basename)}{ext}")

    path, _ = QFileDialog.getSaveFileName(
        parent, f"Export {fmt}", start, filt
    )
    if not path:
        return None
    if os.path.splitext(path)[1].lower() not in accepted:
        path += ext

    _last_dir = os.path.dirname(path) or _last_dir
    return path if image.save(path, fmt, quality) else None
