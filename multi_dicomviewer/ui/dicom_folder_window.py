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

import pydicom
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
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

#: Modalities that are inherently single-frame (cross-sectional / static), so a
#: 1-frame file is NOT a "still" and must not get the "@STILL" suffix.
_STILL_EXEMPT = {"CT", "MR", "NM"}


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


def _read_tags(path: str) -> dict | None:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
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
    return {
        "raw_date": raw_date if raw_date != "Unknown" else "Unknown",
        "studyDate": disp_date,
        "modality": s("Modality"),
        "studyInstanceUID": s("StudyInstanceUID"),
        "patientName": s("PatientName"),
        "numberOfFrames": nframes,
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
    # CT/MR/NM are inherently single-frame, so a 1-frame file there is NOT a
    # "still" — keep it as plain "<MOD>;<date>" rather than "<MOD>@STILL;<date>".
    # The STILL split is meant for genuinely single-shot images (e.g. an XA spot
    # film vs a cine).
    still = (separate_single and f["numberOfFrames"] == 1
             and f["modality"].upper() not in _STILL_EXEMPT)
    if still:
        return (f"{f['studyDate']}_{f['modality']}_STILL",
                _safe(f"{f['modality']}@STILL;{raw}"))
    return (f"{f['studyDate']}_{f['modality']}",
            _safe(f"{f['modality']};{raw}"))


def _unique_name(target_dir: str, name: str, rel_path: str) -> str:
    """File name within *target_dir* that won't collide — prefixed with the
    source sub-folder path (parts joined by '_') and a (n) suffix on clash,
    mirroring the original tool."""
    parts = os.path.dirname(rel_path).split(os.sep)
    prefix = "_".join(p for p in parts if p)
    base = f"{prefix}_{name}" if prefix else name
    if not os.path.exists(os.path.join(target_dir, base)):
        return base
    stem, ext = os.path.splitext(base)
    n = 1
    while True:
        cand = f"{stem}({n}){ext}"
        if not os.path.exists(os.path.join(target_dir, cand)):
            return cand
        n += 1


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
    done = pyqtSignal(int, int, str)                # ok, fail, error

    def __init__(self, files, target, move, group_by, separate, name_map):
        super().__init__()
        self._files = files
        self._target = target
        self._move = move
        self._group_by = group_by
        self._separate = separate
        self._name_map = name_map

    def run(self) -> None:
        ok = fail = 0
        err = ""
        total = len(self._files)
        for i, f in enumerate(self._files):
            try:
                key, default_sub = _group_of(f, self._group_by, self._separate)
                sub = _safe(self._name_map.get(key, default_sub))
                dest_dir = os.path.join(self._target, sub)
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(
                    dest_dir, _unique_name(dest_dir, f["name"], f["relpath"]))
                if self._move:
                    shutil.move(f["path"], dest)
                else:
                    shutil.copy2(f["path"], dest)
                ok += 1
            except Exception as exc:                # keep going on per-file error
                fail += 1
                if not err:
                    err = str(exc)
            if i % 5 == 0 or i == total - 1:
                self.progress.emit(i + 1, total)
        self.done.emit(ok, fail, err)


class DicomFolderWindow(QMainWindow):
    def __init__(self, start_dir: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DicomFolder — organize DICOM files")
        self.resize(950, 720)
        self.setAcceptDrops(True)                  # drop a folder = source
        self._files: list[dict] = []
        self._source: str | None = start_dir
        self._target: str | None = None
        self._name_map: dict[str, str] = {}
        self._worker: QThread | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- source / target pickers
        srow = QHBoxLayout()
        self._src_btn = QPushButton("Select source folder…")
        self._src_btn.clicked.connect(self._pick_source)
        srow.addWidget(self._src_btn)
        self._src_lbl = QLabel("(none)")
        self._src_lbl.setStyleSheet("color:#555;")
        srow.addWidget(self._src_lbl, 1)
        root.addLayout(srow)

        trow = QHBoxLayout()
        self._tgt_btn = QPushButton("Select output folder…")
        self._tgt_btn.clicked.connect(self._pick_target)
        trow.addWidget(self._tgt_btn)
        self._same_btn = QPushButton("Same as source")
        self._same_btn.clicked.connect(
            lambda: self._set_target(self._source) if self._source else None)
        trow.addWidget(self._same_btn)
        self._parent_btn = QPushButton("Parent of source")
        self._parent_btn.clicked.connect(self._target_parent)
        trow.addWidget(self._parent_btn)
        self._tgt_lbl = QLabel("(none)")
        self._tgt_lbl.setStyleSheet("color:#555;")
        trow.addWidget(self._tgt_lbl, 1)
        root.addLayout(trow)

        # ---- group-by + mode options
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Group by:"))
        self._group_grp = QButtonGroup(self)
        self._radios: dict[str, QRadioButton] = {}
        for key, text in (
            (_BY_COMBINED, "Modality + date"),
            (_BY_DATE, "Study date"),
            (_BY_MOD, "Modality"),
            (_BY_UID, "Study UID"),
        ):
            rb = QRadioButton(text)
            self._group_grp.addButton(rb)
            self._radios[key] = rb
            orow.addWidget(rb)
        self._radios[_BY_COMBINED].setChecked(True)
        self._sep_cb = QCheckBox("Separate single-frame (STILL)")
        self._sep_cb.setChecked(True)
        orow.addWidget(self._sep_cb)
        orow.addStretch(1)
        root.addLayout(orow)
        for rb in self._radios.values():
            rb.toggled.connect(self._regroup)
        self._sep_cb.toggled.connect(self._regroup)

        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Action:"))
        self._mode_grp = QButtonGroup(self)
        self._copy_rb = QRadioButton("Copy")
        self._move_rb = QRadioButton("Move")
        self._copy_rb.setChecked(True)
        for rb in (self._copy_rb, self._move_rb):
            self._mode_grp.addButton(rb)
            mrow.addWidget(rb)
        mrow.addStretch(1)
        root.addLayout(mrow)

        # ---- group tree (folder-name column editable)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Group", "Files", "Folder name (editable)"])
        self._tree.setColumnWidth(0, 420)
        self._tree.setColumnWidth(1, 70)
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
        self._go_btn = QPushButton("Sort Files")
        _base_font = QFont(self._go_btn.font())
        self._no_out_lbl = QLabel("Output folder is not selected")
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
                  self._parent_btn, self._go_btn):
            w.setEnabled(not busy)

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
            self, "Select source folder", self._source or "")
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
            lambda: self._stat_lbl.setText("Counting files…"))
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_scan_done)
        self._worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        self._bar.setRange(0, max(1, total))
        self._bar.setValue(done)
        self._stat_lbl.setText(f"Scanning… {done}/{total}")

    def _on_scan_done(self, files: list[dict]) -> None:
        self._bar.setVisible(False)
        self._set_busy(False)
        self._files = files
        if not files:
            self._stat_lbl.setText("No DICOM files found.")
            return
        self._regroup()
        self._update_go()

    # ----------------------------------------------------------- target
    def _pick_target(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select output folder", self._source or "")
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
            self._tgt_lbl.setText("(none)")
            self._tgt_lbl.setStyleSheet("color:#d00; font-weight:bold;")
        else:
            self._tgt_lbl.setText("(none)")
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

    # ----------------------------------------------------------- grouping
    def _regroup(self) -> None:
        if not self._files:
            return
        self._name_map = {}                          # reset on basis change
        group_by = self._group_by()
        separate = self._sep_cb.isChecked()
        self._sep_cb.setEnabled(group_by == _BY_COMBINED)

        groups: dict[str, dict] = {}
        for f in self._files:
            key, default_sub = _group_of(f, group_by, separate)
            g = groups.setdefault(
                key, {"default": default_sub, "files": [],
                      "patients": set(), "mods": set()})
            g["files"].append(f)
            g["patients"].add(f["patientName"])
            g["mods"].add(f["modality"])

        self._tree.blockSignals(True)
        self._tree.clear()
        for key in sorted(groups):
            g = groups[key]
            item = QTreeWidgetItem(self._tree)
            mods = ", ".join(sorted(g["mods"]))
            item.setText(0, f"{key}   [{mods} | {len(g['patients'])} pt]")
            item.setText(1, str(len(g["files"])))
            item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight)
            item.setText(2, g["default"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            for f in g["files"][:50]:
                leaf = QTreeWidgetItem(item)
                leaf.setText(0, f["name"])
                leaf.setText(1, "")
                leaf.setText(2, f"{f['patientName']} · {_human(f['size'])}")
            if len(g["files"]) > 50:
                more = QTreeWidgetItem(item)
                more.setText(0, f"… {len(g['files']) - 50} more")
        self._tree.blockSignals(False)
        self._stat_lbl.setText(
            f"{len(self._files)} DICOM file(s) in {len(groups)} group(s).")

    def _on_name_edited(self, item, col) -> None:
        if col != 2:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key is None:                              # only top-level group rows
            return
        val = item.text(2).strip()
        if val:
            self._name_map[key] = val

    # ----------------------------------------------------------- organize
    def _organize(self) -> None:
        if not self._files or not self._target:
            return
        move = self._move_rb.isChecked()
        verb = "Move" if move else "Copy"
        if QMessageBox.question(
            self, "Organize",
            f"{verb} {len(self._files)} file(s) into sub-folders of\n"
            f"{self._target}?",
        ) != QMessageBox.StandardButton.Yes:
            return
        self._set_busy(True)
        self._bar.setVisible(True)
        self._bar.setRange(0, len(self._files))
        self._worker = _OrganizeWorker(
            list(self._files), self._target, move,
            self._group_by(), self._sep_cb.isChecked(), dict(self._name_map))
        self._worker.progress.connect(
            lambda d, t: (self._bar.setValue(d),
                          self._stat_lbl.setText(f"{verb} … {d}/{t}")))
        self._worker.done.connect(lambda ok, fail, err:
                                  self._on_organized(ok, fail, err, move))
        self._worker.start()

    def _on_organized(self, ok: int, fail: int, err: str, moved: bool) -> None:
        self._bar.setVisible(False)
        self._set_busy(False)
        msg = f"{'Moved' if moved else 'Copied'} {ok} file(s)."
        if fail:
            msg += f"\n{fail} failed: {err}"
        QMessageBox.information(self, "DicomFolder", msg)
        if moved:                                    # sources are gone — rescan
            if self._source:
                self._scan(self._source)

    def closeEvent(self, ev):                        # noqa: N802 (Qt override)
        if self._worker is not None and self._worker.isRunning():
            if hasattr(self._worker, "stop"):
                self._worker.stop()
            self._worker.wait(2000)
        super().closeEvent(ev)
