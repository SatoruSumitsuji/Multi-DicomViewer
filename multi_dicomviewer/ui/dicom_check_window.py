"""DicomCheck — scan a folder, list every non-DICOM file (and optionally
``DICOMDIR``) plus the empty folders left behind, and delete the ones the
user ticks.

Native PyQt6 rewrite of the former standalone ``DicomChecker.html`` browser
tool, folded into the Tools menu so the viewer stays self-contained (no
external browser / Chromium runtime). DICOM detection matches the original:
a file is DICOM iff bytes 128..132 equal ``DICM``.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

_FULLPATH = Qt.ItemDataRole.UserRole          # leaf items: absolute path
_KIND = Qt.ItemDataRole.UserRole + 1          # "file" | "folder"


def _is_dicom(path: str) -> bool:
    """True iff *path* has the standard 128-byte preamble + ``DICM`` magic."""
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


class _ScanWorker(QThread):
    """Walk *root* off the UI thread, collecting (relpath, fullpath, size)
    for every file that should be offered for deletion."""

    counting = pyqtSignal()
    progress = pyqtSignal(int, int)               # done, total
    done = pyqtSignal(list)                        # [(relpath, full, size)]

    def __init__(self, root: str, include_dicomdir: bool):
        super().__init__()
        self._root = root
        self._inc = include_dicomdir
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
        out: list[tuple[str, str, int]] = []
        for i, fp in enumerate(all_files):
            if self._abort:
                return
            name = os.path.basename(fp)
            is_dcm = _is_dicom(fp)
            is_dicomdir = name.upper() == "DICOMDIR"
            if (not is_dcm) or (is_dicomdir and self._inc):
                try:
                    size = os.path.getsize(fp)
                except OSError:
                    size = 0
                out.append((os.path.relpath(fp, self._root), fp, size))
            if i % 50 == 0 or i == total - 1:
                self.progress.emit(i + 1, total)
        if not self._abort:
            self.done.emit(out)


class DicomCheckWindow(QMainWindow):
    def __init__(self, start_dir: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DicomCheck — remove non-DICOM files")
        self.resize(900, 700)
        self.setAcceptDrops(True)                  # drop a folder anywhere
        self._root: str | None = None
        self._worker: _ScanWorker | None = None
        self._guard = False                        # re-entrancy for itemChanged

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        root.addWidget(QLabel(
            "Keep DICOM files only — find non-DICOM files and empty folders "
            "and delete the ones you select."
        ))

        row = QHBoxLayout()
        self._pick_btn = QPushButton("Select folder…")
        self._pick_btn.clicked.connect(self._pick)
        row.addWidget(self._pick_btn)
        self._inc_cb = QCheckBox("Also offer DICOMDIR for deletion")
        self._inc_cb.setChecked(True)
        row.addWidget(self._inc_cb)
        row.addStretch(1)
        root.addLayout(row)

        self._path_lbl = QLabel("No folder selected.")
        self._path_lbl.setStyleSheet("color:#555;")
        root.addWidget(self._path_lbl)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Name", "Size"])
        self._tree.setColumnWidth(0, 600)
        self._tree.itemChanged.connect(self._on_item_changed)
        root.addWidget(self._tree, 1)

        self._stat_lbl = QLabel("")
        root.addWidget(self._stat_lbl)

        self._bar = QProgressBar()
        self._bar.setVisible(False)
        root.addWidget(self._bar)

        btns = QHBoxLayout()
        self._all_btn = QPushButton("Select all")
        self._all_btn.clicked.connect(lambda: self._check_all(True))
        self._none_btn = QPushButton("Deselect all")
        self._none_btn.clicked.connect(lambda: self._check_all(False))
        self._del_btn = QPushButton("Delete selected files + empty folders")
        self._del_btn.clicked.connect(self._delete)
        for b in (self._all_btn, self._none_btn):
            b.setEnabled(False)
            btns.addWidget(b)
        btns.addStretch(1)
        self._del_btn.setEnabled(False)
        btns.addWidget(self._del_btn)
        # Enlarge the bottom action buttons ×1.5 for readability (user request).
        for b in (self._all_btn, self._none_btn, self._del_btn):
            f = b.font()
            f.setPointSizeF(f.pointSizeF() * 1.5)
            b.setFont(f)
        root.addLayout(btns)

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
            self._start_scan(d)

    # --------------------------------------------------------------- scan
    def _pick(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select folder", self._root or "")
        if d:
            self._start_scan(d)

    def _start_scan(self, d: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._root = d
        self._path_lbl.setText(f"Folder: {d}")
        self._tree.clear()
        self._stat_lbl.setText("")
        self._set_busy(True)
        self._bar.setVisible(True)
        self._bar.setRange(0, 0)                   # indeterminate while counting
        self._worker = _ScanWorker(d, self._inc_cb.isChecked())
        self._worker.counting.connect(
            lambda: self._stat_lbl.setText("Counting files…"))
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_scan_done)
        self._worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        self._bar.setRange(0, max(1, total))
        self._bar.setValue(done)
        self._stat_lbl.setText(f"Scanning… {done}/{total}")

    def _on_scan_done(self, items: list[tuple[str, str, int]]) -> None:
        self._bar.setVisible(False)
        self._set_busy(False)
        self._populate(items)
        has = bool(items)
        self._all_btn.setEnabled(has)
        self._none_btn.setEnabled(has)
        if not has:
            self._stat_lbl.setText("No non-DICOM files found. ✅")
        else:
            total_bytes = sum(s for _r, _f, s in items)
            self._stat_lbl.setText(
                f"{len(items)} non-DICOM file(s) found  "
                f"({_human(total_bytes)}). Selected: 0")

    def _set_busy(self, busy: bool) -> None:
        self._pick_btn.setEnabled(not busy)
        self._inc_cb.setEnabled(not busy)

    # --------------------------------------------------------------- tree
    def _populate(self, items: list[tuple[str, str, int]]) -> None:
        self._guard = True
        self._tree.clear()
        folder_nodes: dict[str, QTreeWidgetItem] = {}

        def folder_for(rel_dir: str) -> QTreeWidgetItem | None:
            if rel_dir in ("", "."):
                return None
            if rel_dir in folder_nodes:
                return folder_nodes[rel_dir]
            parent_dir = os.path.dirname(rel_dir)
            parent = folder_for(parent_dir)
            node = (QTreeWidgetItem(parent) if parent is not None
                    else QTreeWidgetItem(self._tree))
            node.setText(0, os.path.basename(rel_dir))
            node.setFlags(node.flags()
                          | Qt.ItemFlag.ItemIsUserCheckable
                          | Qt.ItemFlag.ItemIsAutoTristate)
            node.setCheckState(0, Qt.CheckState.Unchecked)
            node.setData(0, _KIND, "folder")
            folder_nodes[rel_dir] = node
            return node

        for rel, full, size in sorted(items, key=lambda t: t[0].lower()):
            parent = folder_for(os.path.dirname(rel))
            leaf = (QTreeWidgetItem(parent) if parent is not None
                    else QTreeWidgetItem(self._tree))
            leaf.setText(0, os.path.basename(rel))
            leaf.setText(1, _human(size))
            leaf.setTextAlignment(1, Qt.AlignmentFlag.AlignRight)
            leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            leaf.setCheckState(0, Qt.CheckState.Unchecked)
            leaf.setData(0, _FULLPATH, full)
            leaf.setData(0, _KIND, "file")
        self._tree.expandToDepth(0)
        self._guard = False

    def _on_item_changed(self, _item, _col) -> None:
        if self._guard:
            return
        self._update_selected_count()

    def _iter_leaves(self):
        it = self._tree.invisibleRootItem()

        def walk(node):
            for i in range(node.childCount()):
                c = node.child(i)
                if c.data(0, _KIND) == "file":
                    yield c
                yield from walk(c)   # recurse into folders (nested non-DICOM files)
        yield from walk(it)

    def _check_all(self, on: bool) -> None:
        self._guard = True
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        root = self._tree.invisibleRootItem()
        for i in range(root.childCount()):
            self._set_state_recursive(root.child(i), state)
        self._guard = False
        self._update_selected_count()

    def _set_state_recursive(self, node, state) -> None:
        if node.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            node.setCheckState(0, state)
        for i in range(node.childCount()):
            self._set_state_recursive(node.child(i), state)

    def _checked_files(self) -> list[str]:
        return [c.data(0, _FULLPATH) for c in self._iter_leaves()
                if c.checkState(0) == Qt.CheckState.Checked]

    def _update_selected_count(self) -> None:
        n = len(self._checked_files())
        self._del_btn.setEnabled(n > 0)
        txt = self._stat_lbl.text()
        base = txt.split("  Selected:")[0].split(". Selected:")[0]
        sep = "  Selected:" if "non-DICOM" in base else "Selected:"
        self._stat_lbl.setText(f"{base}{sep} {n}")

    # ------------------------------------------------------------- delete
    def _delete(self) -> None:
        targets = self._checked_files()
        if not targets:
            return
        if QMessageBox.question(
            self, "Delete",
            f"Delete {len(targets)} file(s)? This cannot be undone.\n"
            "Empty folders left behind will also be removed.",
        ) != QMessageBox.StandardButton.Yes:
            return

        self._bar.setVisible(True)
        self._bar.setRange(0, len(targets))
        ok = 0
        fail = 0
        for i, fp in enumerate(targets):
            try:
                os.remove(fp)
                ok += 1
            except OSError:
                fail += 1
            self._bar.setValue(i + 1)

        removed_dirs = self._remove_empty_dirs()
        self._bar.setVisible(False)

        msg = f"Deleted {ok} file(s)"
        if removed_dirs:
            msg += f" and {removed_dirs} empty folder(s)"
        if fail:
            msg += f"  ({fail} could not be deleted)"
        QMessageBox.information(self, "DicomCheck", msg + ".")
        # Re-scan so the tree reflects the new state.
        if self._root:
            self._start_scan(self._root)

    def _remove_empty_dirs(self) -> int:
        if not self._root:
            return 0
        removed = 0
        for dirpath, _dirs, _files in os.walk(self._root, topdown=False):
            if os.path.abspath(dirpath) == os.path.abspath(self._root):
                continue
            try:
                if not os.listdir(dirpath):
                    os.rmdir(dirpath)
                    removed += 1
            except OSError:
                pass
        return removed

    def closeEvent(self, ev):                       # noqa: N802 (Qt override)
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        super().closeEvent(ev)
