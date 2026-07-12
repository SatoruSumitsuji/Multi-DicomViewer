"""DicomFolder — scan a source folder for DICOM files, group them by study
date / modality / study UID / (modality + date), and copy or move them into
tidy per-group sub-folders of a target folder.

Native PyQt6 rewrite of the former standalone Electron ``DicomFolder`` app,
folded into the Tools menu so the viewer stays self-contained (no Node /
Chromium runtime). Tag reading uses pydicom — already a viewer dependency.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pydicom
from pydicom.fileset import FileSet
from pydicom.filebase import DicomBytesIO
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from multi_dicomviewer.core import settings
from multi_dicomviewer.core.dicom_io import (
    _decode_frame,
    _normalize_charset,
    _to_float,
    decode_text,
)
from multi_dicomviewer.i18n import t
from multi_dicomviewer.viewers.image_canvas import to_qimage
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QHeaderView,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# group-by keys
_BY_DATE = "studyDate"
_BY_MOD = "modality"
_BY_UID = "studyInstanceUID"
_BY_COMBINED = "combined"

#: Only XA gets the "@STILL" single-frame split: an XA spot film (1 frame) is a
#: genuine still, distinct from a cine run. Every other modality keeps its
#: 1-frame files in the regular "<MOD>;<date>" group (a single-frame CT/MR/NM/US
#: image is normal data, not a "still").
_STILL_MODALITY = "XA"


def _is_dicom(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


def _human(n: float) -> str:
    if n < 1024:
        return f"{int(n)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024.0
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PB"


def _fmt_time(raw: str) -> str:
    """DICOM TM (``HHMMSS[.ffffff]``) → ``HH:MM:SS`` for display. Fixed width,
    so a plain string sort of this column is also chronological. Unparseable
    values are shown as-is."""
    digits = (raw or "").strip().split(".")[0]
    if len(digits) >= 6 and digits[:6].isdigit():
        return f"{digits[0:2]}:{digits[2:4]}:{digits[4:6]}"
    return (raw or "").strip()


#: FIGURE SPACE (U+2007) — same advance width as a digit in most UI fonts, so
#: right-justifying the number with it lines the unit up column-for-column even
#: in a proportional font (a normal space is narrower and would drift).
_FIGSP = " "


def _size_aligned(n: float) -> str:
    """Human size with the UNIT start aligned across rows, e.g. ``"  12.3 MB"``
    / ``" 512.0 KB"``. The numeric part (always ``N.N``) is figure-space padded
    to a fixed width so the space+unit begin at the same character position."""
    unit = "B"
    val = float(n)
    for u in ("B", "KB", "MB", "GB", "TB", "PB"):
        unit = u
        if val < 1024 or u == "PB":
            break
        val /= 1024.0
    return f"{f'{val:.1f}'.rjust(7, _FIGSP)} {unit}"


def _preview_qimage(path: str):
    """Decode a DICOM file's first frame to a display QImage. Returns
    ``(QImage|None, frame_count)``. Grayscale is auto-windowed (CT uses a
    soft-tissue HU window; everything else min–max), colour is shown as-is."""
    try:
        ds = pydicom.dcmread(path, force=True)
        px = _decode_frame(ds, 0)
    except Exception:
        return None, 1
    try:
        nframes = int(getattr(ds, "NumberOfFrames", 1) or 1)
    except (TypeError, ValueError):
        nframes = 1
    if px.ndim == 3:                                 # already display RGB uint8
        return to_qimage(np.ascontiguousarray(px[..., :3].astype(np.uint8))), \
            nframes
    px = px.astype(np.float32)
    if str(getattr(ds, "Modality", "")).upper() == "CT":
        px = px * _to_float(getattr(ds, "RescaleSlope", 1.0), 1.0) + \
            _to_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
        lo, hi = -100.0, 700.0
    else:
        lo, hi = float(px.min()), float(px.max())
    out = np.clip((px - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        out = 255.0 - out                            # MONOCHROME1: min = white
    return to_qimage(out.astype(np.uint8)), nframes


class _RightPadDelegate(QStyledItemDelegate):
    """Adds a fixed right-hand margin to a column's cells so right-aligned
    numbers don't butt against the next column (Files / Acq # / Acq Time /
    Series #). Purely visual — the underlying value (and its numeric sort) is
    untouched."""

    def __init__(self, pad: int, parent=None):
        super().__init__(parent)
        self._pad = int(pad)

    def paint(self, painter, option, index):         # noqa: N802 (Qt override)
        option.rect = option.rect.adjusted(0, 0, -self._pad, 0)
        super().paint(painter, option, index)

    def sizeHint(self, option, index):               # noqa: N802 (Qt override)
        s = super().sizeHint(option, index)
        s.setWidth(s.width() + self._pad)
        return s


class _FilePreviewDialog(QDialog):
    """Popup image viewer for the DicomFolder table: shows one file's first
    frame with First / Prev / Next / Last buttons to step through the other
    files in the SAME folder (in the table's current display order)."""

    def __init__(self, paths: list, labels: list, start: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Image preview"))
        self.resize(720, 660)
        self._paths = paths
        self._labels = labels
        self._i = max(0, min(start, len(paths) - 1))
        self._pix: QPixmap | None = None

        v = QVBoxLayout(self)
        self._title = QLabel("")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._title)
        self._img = QLabel("")
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setMinimumSize(360, 360)
        self._img.setStyleSheet("background:#000; color:#aaa;")
        v.addWidget(self._img, 1)

        nav = QHBoxLayout()
        self._first_btn = QPushButton(t("⏮ First"))
        self._prev_btn = QPushButton(t("◀ Prev"))
        self._next_btn = QPushButton(t("Next ▶"))
        self._last_btn = QPushButton(t("Last ⏭"))
        self._first_btn.clicked.connect(lambda: self._go(0))
        self._prev_btn.clicked.connect(lambda: self._go(self._i - 1))
        self._next_btn.clicked.connect(lambda: self._go(self._i + 1))
        self._last_btn.clicked.connect(lambda: self._go(len(self._paths) - 1))
        for b in (self._first_btn, self._prev_btn,
                  self._next_btn, self._last_btn):
            nav.addWidget(b)
        v.addLayout(nav)
        self._load()

    def _go(self, i: int) -> None:
        i = max(0, min(i, len(self._paths) - 1))
        if i != self._i:
            self._i = i
            self._load()

    def _load(self) -> None:
        qimg, nframes = _preview_qimage(self._paths[self._i])
        if qimg is None:
            self._pix = None
            self._img.setText(t("(cannot display this file)"))
        else:
            self._pix = QPixmap.fromImage(qimg)
            self._render()
        extra = t("  ·  {n} frames (showing #1)", n=nframes) if nframes > 1 else ""
        self._title.setText(
            f"{self._labels[self._i]}   "
            f"({self._i + 1}/{len(self._paths)}){extra}")
        self._first_btn.setEnabled(self._i > 0)
        self._prev_btn.setEnabled(self._i > 0)
        self._next_btn.setEnabled(self._i < len(self._paths) - 1)
        self._last_btn.setEnabled(self._i < len(self._paths) - 1)

    def _render(self) -> None:
        if self._pix is None:
            return
        self._img.setPixmap(self._pix.scaled(
            self._img.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):                         # noqa: N802 (Qt override)
        super().resizeEvent(e)
        self._render()


#: SOP Class UID of a DICOMDIR (Media Storage Directory Storage). Any file with
#: this class is an index, not image data — always ignored during the sort,
#: whatever its filename (covers renamed copies like "..._DICOMDIR" too).
_DICOMDIR_SOP_CLASS = "1.2.840.10008.1.3.10"


def _read_tags(path: str) -> dict | None:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        _normalize_charset(ds)
    except Exception:
        return None

    # Ignore DICOMDIR index files entirely, detected by content so a renamed
    # copy is caught as well as a file literally named "DICOMDIR".
    meta = getattr(ds, "file_meta", None)
    if (meta is not None
            and str(getattr(meta, "MediaStorageSOPClassUID", ""))
            == _DICOMDIR_SOP_CLASS) or "DirectoryRecordSequence" in ds:
        return None

    def s(name: str, default: str = "Unknown") -> str:
        v = getattr(ds, name, None)
        v = "" if v is None else str(v).strip()
        return v or default

    raw_date = s("StudyDate", "Unknown")
    if len(raw_date) == 8 and raw_date != "Unknown":
        disp_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
    else:
        disp_date = raw_date
    try:
        nframes = int(getattr(ds, "NumberOfFrames", 1) or 1)
    except (TypeError, ValueError):
        nframes = 1

    def num(name: str):
        """Integer tag value, or None when absent / non-numeric (so the file
        row sorts numerically and blanks group together)."""
        v = getattr(ds, name, None)
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return {
        "raw_date": raw_date if raw_date != "Unknown" else "Unknown",
        "studyDate": disp_date,
        "modality": s("Modality"),
        "studyInstanceUID": s("StudyInstanceUID"),
        "patientName": decode_text(ds, "PatientName", "") or "Unknown",
        "numberOfFrames": nframes,
        "seriesNumber": num("SeriesNumber"),
        "acquisitionNumber": num("AcquisitionNumber"),
        "acquisitionTime": s("AcquisitionTime", ""),
    }


def _safe(name: str) -> str:
    """Strip characters that are illegal in Windows folder names."""
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name.strip() or "Unknown"


def _group_of(f: dict, group_by: str, separate_single: bool) -> tuple[str, str]:
    """(group key, default sub-folder name) for a scanned file *f*."""
    raw = f["raw_date"] if f["raw_date"] != "Unknown" else "Unknown"
    if group_by == _BY_DATE:
        return f["studyDate"], _safe(raw)
    if group_by == _BY_MOD:
        return f["modality"], _safe(f["modality"])
    if group_by == _BY_UID:
        uid = f["studyInstanceUID"]
        return uid, _safe(f"Study_{uid[:20]}")
    # combined
    # Only XA single-frame files are split off as "<MOD>@STILL;<date>" (a spot
    # film vs a cine run). Every other modality keeps its 1-frame files in the
    # plain "<MOD>;<date>" group.
    still = (separate_single and f["numberOfFrames"] == 1
             and f["modality"].upper() == _STILL_MODALITY)
    if still:
        return (f"{f['studyDate']}_{f['modality']}_STILL",
                _safe(f"{f['modality']}@STILL;{raw}"))
    return (f"{f['studyDate']}_{f['modality']}",
            _safe(f"{f['modality']};{raw}"))


def _flat_name(dest_dir: str, modality: str, counters: dict) -> str:
    """A simple, short destination file name ``<MODALITY>_<6-digit>`` (e.g.
    ``XA_000001``), numbered per output folder + modality. Kept short on purpose
    so long path-derived names don't hit file-name / DICOMDIR length limits.
    Skips any number already present on disk so re-runs don't collide."""
    mod = _safe(str(modality) or "XX")
    key = (dest_dir, mod)
    while True:
        counters[key] = counters.get(key, 0) + 1
        cand = f"{mod}_{counters[key]:06d}"
        if not os.path.exists(os.path.join(dest_dir, cand)):
            return cand


def _write_folder_dicomdir(folder: str, flat_names: list) -> bool:
    """Build a fresh DICOMDIR inside *folder* indexing the flat-named DICOM
    files *flat_names* already present there. Records are built from headers
    only (``stop_before_pixels`` — large cine files aren't fully read); each
    IMAGE record references its file by the flat single-component name so the
    files stay put. Returns True if a DICOMDIR was written.

    The record hierarchy (PATIENT/STUDY/SERIES/IMAGE) is built by pydicom's
    FileSet, whose ``_write_dicomdir`` recomputes all inter-record byte offsets
    from the encoded record sizes — so overriding ReferencedFileID before the
    encode yields a correctly-offset DICOMDIR that points at the flat files."""
    fs = FileSet()
    uid2flat: dict[str, str] = {}
    for flat in flat_names:
        try:
            ds = pydicom.dcmread(os.path.join(folder, flat),
                                 stop_before_pixels=True, force=True)
            uid = getattr(ds, "SOPInstanceUID", None)
            if uid is None:
                continue
            fs.add(ds)
        except Exception:
            continue
        uid2flat[str(uid)] = flat
    if not uid2flat:
        return False
    for node in fs._tree:
        rec = node._record
        uid = rec.get("ReferencedSOPInstanceUIDInFile", None)
        if uid is not None and str(uid) in uid2flat:
            rec.ReferencedFileID = uid2flat[str(uid)]
    fp = DicomBytesIO()
    fp.is_little_endian = True
    fp.is_implicit_VR = False
    fs._write_dicomdir(fp)
    with open(os.path.join(folder, "DICOMDIR"), "wb") as fh:
        fh.write(fp.getvalue())
    return True


class _ScanWorker(QThread):
    counting = pyqtSignal()
    progress = pyqtSignal(int, int)
    done = pyqtSignal(list)                         # [file dicts]

    def __init__(self, root: str):
        super().__init__()
        self._root = root
        self._abort = False

    def stop(self) -> None:
        self._abort = True

    def run(self) -> None:
        self.counting.emit()
        all_files: list[str] = []
        for dirpath, _dirs, files in os.walk(self._root):
            if self._abort:
                return
            for fn in files:
                all_files.append(os.path.join(dirpath, fn))
        total = len(all_files)
        out: list[dict] = []
        for i, fp in enumerate(all_files):
            if self._abort:
                return
            # Existing DICOMDIR index files are ignored entirely — they are not
            # sorted, counted or displayed. When "With DICOMDIR" is on a fresh
            # DICOMDIR is generated per output folder instead.
            if os.path.basename(fp).upper() == "DICOMDIR":
                continue
            if _is_dicom(fp):
                tags = _read_tags(fp)
                if tags is not None:
                    try:
                        size = os.path.getsize(fp)
                    except OSError:
                        size = 0
                    tags.update(
                        path=fp,
                        name=os.path.basename(fp),
                        relpath=os.path.relpath(fp, self._root),
                        size=size,
                    )
                    out.append(tags)
            if i % 25 == 0 or i == total - 1:
                self.progress.emit(i + 1, total)
        if not self._abort:
            self.done.emit(out)


class _OrganizeWorker(QThread):
    progress = pyqtSignal(int, int)
    stage = pyqtSignal(str)                          # transient status text
    done = pyqtSignal(int, int, int, str)           # ok, fail, dicomdirs, error

    def __init__(self, assignments, target, move, with_dicomdir=False):
        super().__init__()
        # assignments: list of (file_dict, destination-subfolder-name). The
        # subfolder already reflects any manual folder rename / drag-move done
        # in the table, so the worker no longer recomputes the grouping.
        self._assignments = assignments
        self._target = target
        self._move = move
        self._with_dicomdir = with_dicomdir

    def run(self) -> None:
        ok = fail = 0
        err = ""
        # Existing DICOMDIRs were already dropped at scan time, so every file
        # here is real image data. Each lands in its group folder under a short
        # "<MODALITY>_<6-digit>" name; a per-(folder, modality) counter keeps
        # them numbered and collision-free.
        counters: dict = {}
        folder_files: dict = {}                     # dest_dir -> [flat names]
        total = len(self._assignments)
        for i, (f, sub) in enumerate(self._assignments):
            try:
                dest_dir = os.path.join(self._target, _safe(sub))
                os.makedirs(dest_dir, exist_ok=True)
                name = _flat_name(dest_dir, f["modality"], counters)
                dest = os.path.join(dest_dir, name)
                if self._move:
                    shutil.move(f["path"], dest)
                else:
                    shutil.copy2(f["path"], dest)
                folder_files.setdefault(dest_dir, []).append(name)
                ok += 1
            except Exception as exc:                # keep going on per-file error
                fail += 1
                if not err:
                    err = str(exc)
            if i % 5 == 0 or i == total - 1:
                self.progress.emit(i + 1, total)

        # Generate one fresh DICOMDIR per output folder (only when requested).
        ndd = 0
        if self._with_dicomdir:
            for n, (dest_dir, names) in enumerate(folder_files.items(), 1):
                self.stage.emit(
                    t("Building DICOMDIR… {n}/{total}",
                      n=n, total=len(folder_files)))
                try:
                    if _write_folder_dicomdir(dest_dir, names):
                        ndd += 1
                except Exception as exc:
                    fail += 1
                    if not err:
                        err = str(exc)
        self.done.emit(ok, fail, ndd, err)


class _GroupTree(QTreeWidget):
    """Group tree that lets the user drag file rows onto another group row to
    reassign which output folder they land in. Qt's own row-move is suppressed
    (we never call ``super().dropEvent``); instead ``files_dropped`` fires with
    the target group key + the dragged files' indices and the window re-renders
    from its own model. External folder drops (source ≠ this tree) are passed
    through so the window's whole-window drop handler still catches them."""

    files_dropped = pyqtSignal(str, list)            # (target key, [file idx])

    def dragEnterEvent(self, ev):                    # noqa: N802 (Qt override)
        if ev.source() is self:
            ev.acceptProposedAction()
        else:
            ev.ignore()                              # let the window handle it

    def dragMoveEvent(self, ev):                     # noqa: N802 (Qt override)
        if ev.source() is self:
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev):                         # noqa: N802 (Qt override)
        if ev.source() is not self:
            ev.ignore()
            return
        target = self.itemAt(ev.position().toPoint())
        if target is None:
            ev.ignore()
            return
        # A drop anywhere on a group row (or on one of its file rows) targets
        # that group.
        grp = target if target.parent() is None else target.parent()
        key = grp.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(key, str):
            ev.ignore()
            return
        idxs = [it.data(0, Qt.ItemDataRole.UserRole)
                for it in self.selectedItems()
                if isinstance(it.data(0, Qt.ItemDataRole.UserRole), int)]
        if idxs:
            self.files_dropped.emit(key, idxs)
            ev.acceptProposedAction()
        else:
            ev.ignore()


class DicomFolderWindow(QMainWindow):
    def __init__(self, start_dir: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("DicomFolder — organize DICOM files"))
        self.resize(950, 720)
        self.setAcceptDrops(True)                  # drop a folder = source
        self._files: list[dict] = []
        self._source: str | None = start_dir
        self._target: str | None = None
        # Output-folder model (editable in the table before sorting):
        #   _groups:     group key -> {"name": output folder, "manual": bool,
        #                              "order": int} — includes both the auto
        #                groups AND user-created ("New Folder") ones.
        #   _file_group: file index (into _files) -> its current group key.
        # Drag-drop reassigns _file_group; "New Folder" adds to _groups; the
        # editable name column renames _groups[key]["name"].
        self._groups: dict[str, dict] = {}
        self._file_group: dict[int, str] = {}
        self._next_new = 1                           # id for new-folder keys
        self._item_by_key: dict[str, object] = {}    # key -> its group row
        self._worker: QThread | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- source / target pickers
        srow = QHBoxLayout()
        self._src_btn = QPushButton(t("Select source folder…"))
        self._src_btn.clicked.connect(self._pick_source)
        srow.addWidget(self._src_btn)
        self._src_lbl = QLabel(t("(none)"))
        self._src_lbl.setStyleSheet("color:#555;")
        srow.addWidget(self._src_lbl, 1)
        root.addLayout(srow)

        trow = QHBoxLayout()
        self._tgt_btn = QPushButton(t("Select output folder…"))
        self._tgt_btn.clicked.connect(self._pick_target)
        trow.addWidget(self._tgt_btn)
        self._same_btn = QPushButton(t("Same as source"))
        self._same_btn.clicked.connect(
            lambda: self._set_target(self._source) if self._source else None)
        trow.addWidget(self._same_btn)
        self._parent_btn = QPushButton(t("Parent of source"))
        self._parent_btn.clicked.connect(self._target_parent)
        trow.addWidget(self._parent_btn)
        self._tgt_lbl = QLabel(t("(none)"))
        self._tgt_lbl.setStyleSheet("color:#555;")
        trow.addWidget(self._tgt_lbl, 1)
        root.addLayout(trow)

        # ---- group-by + mode options
        orow = QHBoxLayout()
        orow.addWidget(QLabel(t("Group by:")))
        self._group_grp = QButtonGroup(self)
        self._radios: dict[str, QRadioButton] = {}
        for key, text in (
            (_BY_COMBINED, t("Modality + date")),
            (_BY_DATE, t("Study date")),
            (_BY_MOD, t("Modality")),
            (_BY_UID, t("Study UID")),
        ):
            rb = QRadioButton(text)
            self._group_grp.addButton(rb)
            self._radios[key] = rb
            orow.addWidget(rb)
        self._radios[_BY_COMBINED].setChecked(True)
        # Existing DICOMDIR index files are always ignored. "With DICOMDIR" on
        # -> generate a fresh DICOMDIR inside each output folder (indexing that
        # folder's files). Off (default) -> no DICOMDIR is created.
        self._with_dicomdir_cb = QCheckBox(t("With DICOMDIR"))
        self._with_dicomdir_cb.setToolTip(
            t("On: create a new DICOMDIR in each output folder.\n"
              "Off: don't create any DICOMDIR.\n"
              "(Existing DICOMDIR files in the source are always ignored.)"))
        orow.addWidget(self._with_dicomdir_cb)
        self._sep_cb = QCheckBox(t("Separate XA single-frame (XA@STILL)"))
        self._sep_cb.setChecked(True)
        orow.addWidget(self._sep_cb)
        orow.addStretch(1)
        root.addLayout(orow)
        for rb in self._radios.values():
            rb.toggled.connect(self._regroup)
        self._sep_cb.toggled.connect(self._regroup)

        mrow = QHBoxLayout()
        mrow.addWidget(QLabel(t("Action:")))
        self._mode_grp = QButtonGroup(self)
        self._copy_rb = QRadioButton(t("Copy"))
        self._move_rb = QRadioButton(t("Move"))
        self._copy_rb.setChecked(True)
        for rb in (self._copy_rb, self._move_rb):
            self._mode_grp.addButton(rb)
            mrow.addWidget(rb)
        mrow.addStretch(1)
        root.addLayout(mrow)

        # ---- folder-editing row: create a new (empty) output folder, plus a
        # hint that files can be dragged between folders in the table.
        erow = QHBoxLayout()
        self._new_folder_btn = QPushButton(t("New Folder"))
        self._new_folder_btn.setToolTip(
            t("Create a new (empty) output folder, then rename it and drag "
              "files onto it."))
        self._new_folder_btn.clicked.connect(self._new_folder)
        self._new_folder_btn.setEnabled(False)
        erow.addWidget(self._new_folder_btn)
        _hint = QLabel(t("Tip: drag file rows onto a folder to move them "
                         "between output folders."))
        _hint.setStyleSheet("color:#555;")
        erow.addWidget(_hint)
        # "Show the file": preview the selected file's image in a popup, with
        # First/Prev/Next/Last to page through the other files in that folder.
        self._show_btn = QPushButton(t("Show the file"))
        self._show_btn.setToolTip(
            t("Preview the selected file's image; step through the folder's "
              "files with First/Prev/Next/Last."))
        self._show_btn.clicked.connect(self._show_file)
        self._show_btn.setEnabled(False)
        erow.addSpacing(12)
        erow.addWidget(self._show_btn)
        erow.addStretch(1)
        root.addLayout(erow)

        # ---- group tree (folder-name column editable). Expanding a group lists
        # its files with per-file Acq #, Acq Time and Series # columns; every
        # column is sortable (click a header) — files sort within their group.
        # File rows can be dragged onto another folder row to reassign them.
        self._tree = _GroupTree()
        self._tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self._tree.setDragEnabled(True)
        self._tree.setAcceptDrops(True)
        self._tree.viewport().setAcceptDrops(True)
        self._tree.setDropIndicatorShown(True)
        self._tree.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self._tree.files_dropped.connect(self._on_files_dropped)
        self._tree.setHeaderLabels(
            [t("Group"), t("Files"), t("Folder name (editable) / Size"),
             t("Acq #"), t("Acq Time"), t("Series #")])
        # A uniform right margin on the right-aligned numeric columns (Files,
        # Acq #, Acq Time, Series #) so their values don't butt against the next
        # column. ~1.5× the old 3-space Files margin.
        self._pad_delegate = _RightPadDelegate(18, self._tree)
        for _c in (1, 3, 4, 5):
            self._tree.setItemDelegateForColumn(_c, self._pad_delegate)
        self._tree.setColumnWidth(0, 380)
        self._tree.setColumnWidth(1, 60)
        self._tree.setColumnWidth(2, 200)
        self._tree.setColumnWidth(3, 70)
        self._tree.setColumnWidth(4, 90)
        self._tree.setColumnWidth(5, 80)
        # Make the (wide, elidable) Group column absorb window resizing instead
        # of the last section. Otherwise the default stretch-last-section shrinks
        # the Series # column as the window narrows and its numbers vanish while
        # the roomy Group column keeps empty space. Now Group flexes and the
        # fixed-width Acq #/Acq Time/Series # columns always show their data.
        hdr = self._tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setMinimumSectionSize(40)
        self._tree.setSortingEnabled(True)
        # Restore the sort column + order from the last session (defaults to the
        # Group column, ascending) and persist any change the user makes.
        _col, _order = settings.load_dicomfolder_sort()
        self._sort = (_col, _order)
        self._tree.sortByColumn(_col, Qt.SortOrder(_order))
        self._tree.header().sortIndicatorChanged.connect(self._on_sort_changed)
        self._tree.itemChanged.connect(self._on_name_edited)
        root.addWidget(self._tree, 1)

        self._stat_lbl = QLabel("")
        root.addWidget(self._stat_lbl)
        self._bar = QProgressBar()
        self._bar.setVisible(False)
        root.addWidget(self._bar)

        brow = QHBoxLayout()
        brow.addStretch(1)
        # Red warning shown (left of the button) while no output folder is set.
        # Kept at the button's ORIGINAL text size (per request), so capture that
        # before enlarging the button.
        self._go_btn = QPushButton(t("Sort Files"))
        _base_font = QFont(self._go_btn.font())
        self._no_out_lbl = QLabel(t("Output folder is not selected"))
        self._no_out_lbl.setFont(_base_font)
        self._no_out_lbl.setStyleSheet("color:#d00;")
        brow.addWidget(self._no_out_lbl)
        # Enlarge the action button ×1.5 to make it stand out (was "Organize").
        _go_font = QFont(_base_font)
        _go_font.setPointSizeF(_base_font.pointSizeF() * 1.5)
        self._go_btn.setFont(_go_font)
        self._go_btn.setEnabled(False)
        self._go_btn.clicked.connect(self._organize)
        brow.addWidget(self._go_btn)
        # "Clear Selection" (right of Sort Files): reset source/output folders
        # and the scanned groups back to the initial empty state.
        self._clear_btn = QPushButton(t("Clear Selection"))
        self._clear_btn.setFont(_go_font)
        self._clear_btn.clicked.connect(self._clear)
        brow.addWidget(self._clear_btn)
        root.addLayout(brow)

        # Initial label/button states (no source, no target yet).
        self._update_tgt_lbl()
        self._update_go()

    # ----------------------------------------------------------- helpers
    def _group_by(self) -> str:
        for key, rb in self._radios.items():
            if rb.isChecked():
                return key
        return _BY_COMBINED

    def _set_busy(self, busy: bool) -> None:
        for w in (self._src_btn, self._tgt_btn, self._same_btn,
                  self._parent_btn, self._go_btn, self._clear_btn,
                  self._new_folder_btn, self._show_btn):
            w.setEnabled(not busy)

    def _clear(self) -> None:
        """Clear ALL selections — source/output folders and the scanned groups —
        returning the window to its initial empty state."""
        if self._worker is not None and self._worker.isRunning():
            return
        self._source = None
        self._target = None
        self._files = []
        self._groups = {}
        self._file_group = {}
        self._item_by_key = {}
        self._next_new = 1
        self._tree.clear()
        self._src_lbl.setText(t("(none)"))
        self._src_lbl.setStyleSheet("color:#555;")
        self._stat_lbl.setText("")
        self._bar.setVisible(False)
        self._update_tgt_lbl()                      # output (none) back to grey
        self._update_go()                           # disable Sort Files, hide warning

    # ---------------------------------------------------------- drag&drop
    @staticmethod
    def _dropped_dir(ev) -> str | None:
        md = ev.mimeData()
        if not md.hasUrls():
            return None
        for u in md.urls():
            p = u.toLocalFile()
            if p and os.path.isdir(p):
                return p
        return None

    def dragEnterEvent(self, ev):                  # noqa: N802 (Qt override)
        busy = self._worker is not None and self._worker.isRunning()
        if not busy and self._dropped_dir(ev) is not None:
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):                   # noqa: N802 (Qt override)
        self.dragEnterEvent(ev)

    def dropEvent(self, ev):                        # noqa: N802 (Qt override)
        d = self._dropped_dir(ev)
        if d:
            ev.acceptProposedAction()
            self._set_source_and_scan(d)            # a dropped folder = source

    # ----------------------------------------------------------- source
    def _pick_source(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, t("Select source folder"), self._source or "")
        if d:
            self._set_source_and_scan(d)

    def _set_source_and_scan(self, d: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._source = d
        self._src_lbl.setText(d)
        self._update_tgt_lbl()                      # (none) turns red once a source exists
        self._update_go()                           # show the red warning now too
        self._scan(d)

    def _scan(self, d: str) -> None:
        self._files = []
        self._tree.clear()
        self._set_busy(True)
        self._bar.setVisible(True)
        self._bar.setRange(0, 0)
        self._worker = _ScanWorker(d)
        self._worker.counting.connect(
            lambda: self._stat_lbl.setText(t("Counting files…")))
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_scan_done)
        self._worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        self._bar.setRange(0, max(1, total))
        self._bar.setValue(done)
        self._stat_lbl.setText(t("Scanning… {done}/{total}",
                                 done=done, total=total))

    def _on_scan_done(self, files: list[dict]) -> None:
        self._bar.setVisible(False)
        self._set_busy(False)
        self._files = files
        if not files:
            self._stat_lbl.setText(t("No DICOM files found."))
            self._update_go()
            return
        self._regroup()
        self._update_go()

    # ----------------------------------------------------------- target
    def _pick_target(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, t("Select output folder"), self._source or "")
        if d:
            self._set_target(d)

    def _set_target(self, d: str) -> None:
        self._target = d
        self._update_tgt_lbl()
        self._update_go()

    def _update_tgt_lbl(self) -> None:
        """Output-folder label state:
          * output set            -> the folder path, grey
          * no source yet         -> "(none)", grey
          * source set, no output -> "(none)", BOLD RED (it's now the thing the
                                     user must still pick before sorting)."""
        if self._target:
            self._tgt_lbl.setText(self._target)
            self._tgt_lbl.setStyleSheet("color:#555;")
        elif self._source:
            self._tgt_lbl.setText(t("(none)"))
            self._tgt_lbl.setStyleSheet("color:#d00; font-weight:bold;")
        else:
            self._tgt_lbl.setText(t("(none)"))
            self._tgt_lbl.setStyleSheet("color:#555;")

    def _target_parent(self) -> None:
        if not self._source:
            return
        parent = os.path.dirname(self._source.rstrip("/\\"))
        if parent and parent != self._source:
            self._set_target(parent)

    def _update_go(self) -> None:
        has_target = bool(self._target)
        self._go_btn.setEnabled(bool(self._files) and has_target)
        # "Sort Files" is greyed out until an output folder is chosen. The red
        # "Output folder is not selected" note appears only ONCE A SOURCE EXISTS
        # (before that there's nothing to sort yet, so it would be noise) and an
        # output folder still isn't set.
        self._no_out_lbl.setVisible(bool(self._source) and not has_target)
        # Nothing to clear until a source folder is picked.
        self._clear_btn.setEnabled(bool(self._source))
        # New Folder / Show the file need a scanned set to act on.
        self._new_folder_btn.setEnabled(bool(self._files))
        self._show_btn.setEnabled(bool(self._files))

    # ----------------------------------------------------------- grouping
    def _regroup(self) -> None:
        """Recompute the auto-grouping from the current basis and render it.
        This resets any manual folder edits (new folders / drag-moves), same as
        the folder-name map used to reset — changing the grouping basis is a
        fresh start."""
        if not self._files:
            return
        group_by = self._group_by()
        separate = self._sep_cb.isChecked()
        self._sep_cb.setEnabled(group_by == _BY_COMBINED)
        # Existing DICOMDIRs were dropped at scan time, so every file here is
        # real image data. Build key→folder-name (auto) + file→key assignment.
        self._groups = {}
        self._file_group = {}
        self._next_new = 1
        order = 0
        for idx, f in enumerate(self._files):
            key, default_sub = _group_of(f, group_by, separate)
            if key not in self._groups:
                self._groups[key] = {"name": default_sub, "manual": False,
                                     "order": order}
                order += 1
            self._file_group[idx] = key
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        """Render _groups + _file_group into the tree WITHOUT recomputing the
        assignment — called after the auto-regroup and after every manual edit
        (drag-move, new folder, rename)."""
        gfiles: dict[str, list[int]] = {k: [] for k in self._groups}
        for idx, key in self._file_group.items():
            gfiles.setdefault(key, []).append(idx)

        # Populate with sorting OFF (fast, order preserved) then switch it back
        # on so the header stays click-sortable.
        self._tree.setSortingEnabled(False)
        self._tree.blockSignals(True)
        self._tree.clear()
        self._item_by_key = {}
        for key in sorted(self._groups, key=lambda k: self._groups[k]["order"]):
            g = self._groups[key]
            idxs = gfiles.get(key, [])
            files = [self._files[i] for i in idxs]
            mods = ", ".join(sorted({f["modality"] for f in files})) or "—"
            npt = len({f["patientName"] for f in files})
            item = QTreeWidgetItem(self._tree)
            # Auto groups show their descriptive key; a user "New Folder" shows
            # its (editable) name — the key is just an internal id there.
            head = g["name"] if g["manual"] else key
            item.setText(0, f"{head}   [{mods} | {npt} pt]")
            # File count — numeric (sorts right) + right-aligned; the right
            # margin comes from _RightPadDelegate now, not trailing spaces.
            item.setData(1, Qt.ItemDataRole.DisplayRole, len(files))
            item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight)
            item.setText(2, g["name"])
            # Group rows: editable name + drop target, but not draggable.
            item.setFlags((item.flags() | Qt.ItemFlag.ItemIsEditable
                           | Qt.ItemFlag.ItemIsDropEnabled)
                          & ~Qt.ItemFlag.ItemIsDragEnabled)
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            self._item_by_key[key] = item
            # File rows — every file, so a header-click sort covers the whole
            # group (no truncation). Acq #/Series # carry their INTEGER value so
            # the column sorts numerically (10 after 2, not before); Acq Time is
            # fixed-width HH:MM:SS so a string sort is chronological too.
            for idx in idxs:
                f = self._files[idx]
                leaf = QTreeWidgetItem(item)
                leaf.setText(0, f["name"])
                # Under "… / Size": the file size, unit-aligned across rows.
                leaf.setText(2, _size_aligned(f["size"]))
                if f.get("acquisitionNumber") is not None:
                    leaf.setData(3, Qt.ItemDataRole.DisplayRole,
                                 f["acquisitionNumber"])
                leaf.setTextAlignment(3, Qt.AlignmentFlag.AlignRight)
                leaf.setText(4, _fmt_time(f.get("acquisitionTime", "")))
                leaf.setTextAlignment(4, Qt.AlignmentFlag.AlignRight)
                if f.get("seriesNumber") is not None:
                    leaf.setData(5, Qt.ItemDataRole.DisplayRole,
                                 f["seriesNumber"])
                leaf.setTextAlignment(5, Qt.AlignmentFlag.AlignRight)
                # File rows: draggable, not a drop target, not editable. UserRole
                # carries the file index so a drop knows what moved.
                leaf.setFlags((leaf.flags() | Qt.ItemFlag.ItemIsDragEnabled)
                              & ~(Qt.ItemFlag.ItemIsDropEnabled
                                  | Qt.ItemFlag.ItemIsEditable))
                leaf.setData(0, Qt.ItemDataRole.UserRole, idx)
        self._tree.blockSignals(False)
        self._tree.setSortingEnabled(True)
        # Re-apply the remembered sort so files show in the user's chosen order.
        self._tree.sortByColumn(self._sort[0], Qt.SortOrder(self._sort[1]))
        self._stat_lbl.setText(
            t("{files} DICOM file(s) in {groups} group(s).",
              files=len(self._files), groups=len(self._groups)))

    def _on_files_dropped(self, key: str, idxs: list) -> None:
        """Drag-drop: reassign the dropped files to group *key* and re-render."""
        if key not in self._groups:
            return
        changed = False
        for i in idxs:
            if 0 <= i < len(self._files) and self._file_group.get(i) != key:
                self._file_group[i] = key
                changed = True
        if changed:
            self._rebuild_tree()

    def _new_folder(self) -> None:
        """"New Folder": add an empty output folder and start renaming it. Files
        can then be dragged onto it."""
        if not self._files:
            return
        key = f"__new_{self._next_new}"
        self._next_new += 1
        order = 1 + max((g["order"] for g in self._groups.values()), default=-1)
        self._groups[key] = {"name": t("New Folder"), "manual": True,
                             "order": order}
        self._rebuild_tree()
        item = self._item_by_key.get(key)
        if item is not None:
            self._tree.setCurrentItem(item)
            self._tree.editItem(item, 2)             # inline-rename the folder

    def _show_file(self) -> None:
        """"Show the file": preview the selected file's image in a popup that
        can page through the other files in the SAME folder (in the table's
        current display order). If a folder row is selected, start at its first
        file; with nothing selected, use the first folder."""
        if not self._files:
            return
        cur = self._tree.currentItem()
        start_leaf = None
        if cur is None:
            grp = self._tree.topLevelItem(0)
        elif cur.parent() is None:                   # a folder row
            grp = cur
        else:                                        # a file row
            grp = cur.parent()
            start_leaf = cur
        if grp is None:
            return
        paths, labels, start = [], [], 0
        for r in range(grp.childCount()):
            leaf = grp.child(r)
            idx = leaf.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(idx, int):
                continue
            if leaf is start_leaf:
                start = len(paths)
            paths.append(self._files[idx]["path"])
            labels.append(self._files[idx]["name"])
        if not paths:
            QMessageBox.information(
                self, t("Show the file"),
                t("This folder has no files to preview."))
            return
        _FilePreviewDialog(paths, labels, start, self).exec()

    def _on_sort_changed(self, column: int, order) -> None:
        """Header clicked → remember the sort so a fresh (re)group keeps it and
        it survives an app restart. *order* arrives as a ``Qt.SortOrder`` enum
        (not int) from the signal, so read its ``.value`` (0=Asc, 1=Desc)."""
        order_int = getattr(order, "value", order)
        self._sort = (int(column), int(order_int))
        settings.save_dicomfolder_sort(*self._sort)

    def _on_name_edited(self, item, col) -> None:
        if col != 2:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        # Only a top-level group row (str key) carries an editable folder name;
        # file rows store an int index and are not editable.
        if not isinstance(key, str) or key not in self._groups:
            return
        val = item.text(2).strip()
        if val:
            self._groups[key]["name"] = val
            # A user folder shows its name in the Group column too — keep that in
            # sync (auto groups show their key there, which renaming doesn't
            # affect). setText(0) re-emits itemChanged for col 0, ignored above.
            if self._groups[key]["manual"]:
                _head, sep, rest = item.text(0).partition("   [")
                item.setText(0, f"{val}{sep}{rest}")

    # ----------------------------------------------------------- organize
    def _organize(self) -> None:
        if not self._files or not self._target:
            return
        move = self._move_rb.isChecked()
        verb = t("Move") if move else t("Copy")
        if QMessageBox.question(
            self, t("Organize"),
            t("{verb} {n} file(s) into sub-folders of\n{path}?",
              verb=verb, n=len(self._files), path=self._target),
        ) != QMessageBox.StandardButton.Yes:
            return
        self._set_busy(True)
        self._bar.setVisible(True)
        self._bar.setRange(0, len(self._files))
        # Each file's destination = its assigned group's (possibly renamed /
        # newly-created) folder name, reflecting every drag-move done in the
        # table.
        assignments = []
        for idx, f in enumerate(self._files):
            key = self._file_group.get(idx)
            g = self._groups.get(key)
            sub = g["name"] if g else _group_of(
                f, self._group_by(), self._sep_cb.isChecked())[1]
            assignments.append((f, sub))
        self._worker = _OrganizeWorker(
            assignments, self._target, move,
            self._with_dicomdir_cb.isChecked())
        self._worker.progress.connect(
            lambda d, tot: (self._bar.setValue(d),
                            self._stat_lbl.setText(
                                t("{verb} … {done}/{total}",
                                  verb=verb, done=d, total=tot))))
        self._worker.stage.connect(self._stat_lbl.setText)
        self._worker.done.connect(lambda ok, fail, ndd, err:
                                  self._on_organized(ok, fail, ndd, err, move))
        self._worker.start()

    def _on_organized(self, ok: int, fail: int, ndd: int, err: str,
                      moved: bool) -> None:
        self._bar.setVisible(False)
        self._set_busy(False)
        msg = (t("Moved {n} file(s).", n=ok) if moved
               else t("Copied {n} file(s).", n=ok))
        if ndd:                                      # fresh DICOMDIRs generated
            msg += t("\nCreated {n} DICOMDIR(s).", n=ndd)
        if fail:
            msg += t("\n{n} failed: {err}", n=fail, err=err)
        QMessageBox.information(self, t("DicomFolder"), msg)
        if moved:                                    # sources are gone — rescan
            if self._source:
                self._scan(self._source)

    def closeEvent(self, ev):                        # noqa: N802 (Qt override)
        if self._worker is not None and self._worker.isRunning():
            if hasattr(self._worker, "stop"):
                self._worker.stop()
            self._worker.wait(2000)
        super().closeEvent(ev)
