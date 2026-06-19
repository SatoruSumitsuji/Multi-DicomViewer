"""Export a binary DICOM tag (OB/OW/UN/…) to text in three representations.

A binary VR element (e.g. a vendor-private tag like (0029,1007)) that the tag
viewer / CSV export can only summarise as ``<binary: N bytes>`` can be written
out in full here, picking one or more of:

  * <stem>.<GGGG_EEEE>.hex.txt     — space-separated uppercase hex ("AB CD …")
  * <stem>.<GGGG_EEEE>.base64.txt  — Base64 (text-safe, fully reversible)
  * <stem>.<GGGG_EEEE>.latin1.txt  — bytes decoded 1:1 as Latin-1 (UTF-8 file;
                                     fishes out any embedded text fragments)

App-integrated version of ``tools/dump_binary_tag.py``: same three formats,
launched from Tools ▸ "Export binary DICOM tag…".
"""
from __future__ import annotations

import base64
import os

import pydicom
from pydicom.tag import Tag
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

#: (format key, file-suffix, checkbox label) for the three representations.
_FORMATS = (
    ("hex", "hex.txt", "Hex dump (inspect byte values)"),
    ("base64", "base64.txt", "Base64 (exact, reversible)"),
    ("latin1", "latin1.txt", "Latin-1 1:1 (read embedded text)"),
)


def parse_tag(s: str) -> Tag:
    """'0029,1007' / '00291007' / '(0029,1007)' → Tag(0x0029, 0x1007)."""
    t = s.strip().lstrip("(").rstrip(")").replace(",", "").replace(" ", "")
    if len(t) != 8:
        raise ValueError(f"Tag must be 8 hex digits (group+element): {s!r}")
    return Tag(int(t[:4], 16), int(t[4:], 16))


def read_tag_bytes(file_path: str, tag: Tag) -> tuple[str, bytes]:
    """(VR, raw bytes) for *tag* in *file_path*. Reads without pixels first
    (fast for the usual private-tag case); re-reads in full only if the tag
    isn't found that way. Raises KeyError if the tag is absent."""
    ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
    if tag not in ds:
        ds = pydicom.dcmread(file_path, force=True)   # e.g. tag at/after pixels
    if tag not in ds:
        raise KeyError(f"{tag} is not present in this file")
    elem = ds[tag]
    raw = elem.value
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raw = str(raw).encode("utf-8", "replace")
    return str(elem.VR), bytes(raw)


def format_text(raw: bytes, fmt: str) -> str:
    if fmt == "hex":
        return " ".join(f"{b:02X}" for b in raw)
    if fmt == "base64":
        return base64.b64encode(raw).decode("ascii")
    if fmt == "latin1":
        return raw.decode("latin-1")
    raise ValueError(f"unknown format: {fmt!r}")


def export_tag(file_path: str, tag: Tag, raw: bytes,
               formats, out_dir: str) -> list[str]:
    """Write the chosen *formats* of *raw* next to ``<stem>`` in *out_dir*.
    Returns the written file paths."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(file_path))[0]
    label = f"{tag.group:04X}_{tag.element:04X}"
    written: list[str] = []
    suffix_by_key = {k: s for k, s, _ in _FORMATS}
    for fmt in formats:
        path = os.path.join(out_dir, f"{stem}.{label}.{suffix_by_key[fmt]}")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(format_text(raw, fmt))
        written.append(path)
    return written


class BinaryTagExportDialog(QDialog):
    """Pick a DICOM file + a tag, read its raw bytes, and export the chosen
    text representation(s)."""

    def __init__(self, start_dir: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export binary DICOM tag")
        self.resize(560, 300)
        self._vr = None
        self._raw: bytes | None = None
        self._start_dir = start_dir or ""

        v = QVBoxLayout(self)

        # --- DICOM file
        frow = QHBoxLayout()
        frow.addWidget(QLabel("DICOM file:"))
        self._file = QLineEdit()
        frow.addWidget(self._file, 1)
        fb = QPushButton("Browse…")
        fb.clicked.connect(self._browse_file)
        frow.addWidget(fb)
        v.addLayout(frow)

        # --- tag + read
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Tag (group,element):"))
        self._tag = QLineEdit()
        self._tag.setPlaceholderText("e.g. 0029,1007")
        trow.addWidget(self._tag, 1)
        rb = QPushButton("Read")
        rb.clicked.connect(self._read)
        trow.addWidget(rb)
        v.addLayout(trow)

        self._info = QLabel("Choose a file and tag, then Read.")
        self._info.setWordWrap(True)
        v.addWidget(self._info)

        # --- formats
        box = QGroupBox("Formats to export")
        bl = QVBoxLayout(box)
        self._checks: dict[str, QCheckBox] = {}
        for key, _suffix, label in _FORMATS:
            cb = QCheckBox(label)
            cb.setChecked(True)
            self._checks[key] = cb
            bl.addWidget(cb)
        v.addWidget(box)

        # --- output folder
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Output folder:"))
        self._out = QLineEdit()
        orow.addWidget(self._out, 1)
        ob = QPushButton("Browse…")
        ob.clicked.connect(self._browse_out)
        orow.addWidget(ob)
        v.addLayout(orow)

        # --- actions
        brow = QHBoxLayout()
        brow.addStretch(1)
        self._export_btn = QPushButton("Export")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export)
        brow.addWidget(self._export_btn)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        brow.addWidget(close)
        v.addLayout(brow)

    # ----------------------------------------------------------- helpers
    def _browse_file(self) -> None:
        start = self._file.text() or ""
        start_dir = os.path.dirname(start) if start else self._start_dir
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose DICOM file", start_dir,
            "All files (*);;DICOM files (*.dcm *.ima *.dicom)",
        )
        if path:
            self._file.setText(path)
            if not self._out.text():
                self._out.setText(os.path.dirname(path))

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder",
                                             self._out.text() or "")
        if d:
            self._out.setText(d)

    def _read(self) -> None:
        self._raw = None
        self._export_btn.setEnabled(False)
        path = self._file.text().strip()
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Export binary DICOM tag",
                                "Please choose a valid DICOM file.")
            return
        try:
            tag = parse_tag(self._tag.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Export binary DICOM tag", str(exc))
            return
        try:
            vr, raw = read_tag_bytes(path, tag)
        except Exception as exc:
            QMessageBox.warning(self, "Export binary DICOM tag",
                                f"Could not read {tag}:\n{exc}")
            return
        self._vr = vr
        self._raw = raw
        printable = sum(1 for b in raw if 32 <= b < 127 or b in (9, 10, 13))
        pct = printable / max(len(raw), 1)
        self._info.setText(
            f"Tag {tag}  VR={vr}  {len(raw):,} bytes  "
            f"printable {printable:,}/{len(raw):,} ({pct:.1%})"
        )
        if not self._out.text():
            self._out.setText(os.path.dirname(os.path.abspath(path)))
        self._export_btn.setEnabled(True)

    def _export(self) -> None:
        if self._raw is None:
            return
        formats = [k for k, cb in self._checks.items() if cb.isChecked()]
        if not formats:
            QMessageBox.warning(self, "Export binary DICOM tag",
                                "Select at least one format.")
            return
        out_dir = self._out.text().strip() or os.path.dirname(
            os.path.abspath(self._file.text()))
        try:
            tag = parse_tag(self._tag.text())
            written = export_tag(self._file.text(), tag, self._raw,
                                 formats, out_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Export binary DICOM tag",
                                 f"Export failed:\n{exc}")
            return
        QMessageBox.information(
            self, "Export binary DICOM tag",
            "Wrote:\n" + "\n".join(written))
