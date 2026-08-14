"""Application shell: study browser + a configurable viewer grid.

The screen is split into 1×1, 1×2, 2×1 or 2×2 panes. Each pane is modality
agnostic: drag a series from the left info panel onto a pane and the pane
spins up the viewer that matches the series' modality (XA / CT today;
IVUS, OCT/OFDI, NM slot in via _VIEWER_FACTORY as they land). A patient
with both a CCTA and an invasive angiogram can therefore be correlated by
dropping each study into its own pane.
"""
from __future__ import annotations

import math
import os
import sys
import traceback

from PyQt6.QtCore import (
    QEvent, QMimeData, QObject, QRect, QSize, Qt, QThread, QTimer, pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QDrag,
    QDragEnterEvent,
    QDropEvent,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from multi_dicomviewer.config import (
    APP_NAME,
    APP_VERSION,
    BLOCK_CT,
    BLOCK_CT_MESSAGE,
    build_string,
)
from multi_dicomviewer import i18n
from multi_dicomviewer.i18n import t
from multi_dicomviewer.core import anonymize, dicom_io, settings
from multi_dicomviewer.core.dicom_tags import (
    default_overlay_keywords,
    upgrade_private_literal,
)
from multi_dicomviewer.core.study_model import Modality, Series
from multi_dicomviewer.ui.history_dialog import MeasureHistoryDialog
from multi_dicomviewer.ui.study_browser import (
    SERIES_MIME,
    STUDY_MIME,
    FitButton,
    StudyPanel,
)
from multi_dicomviewer.ui.tag_dialog import TagSelectionDialog
from multi_dicomviewer.ui.tag_font import (
    TAG_FONT_PT_DEFAULT, TAG_FONT_PT_MAX, TAG_FONT_PT_MIN,
)
from multi_dicomviewer.viewers.ivus_viewer import IVUSViewer
from multi_dicomviewer.viewers.sr_viewer import SRViewer
from multi_dicomviewer.viewers.xa_viewer import XAViewer

# CT's render backend is heavy and slow to import (VTK on Windows/Linux —
# hundreds of DLLs; pygfx/wgpu on macOS). Importing it at startup was most of
# the perceived launch delay (worst on Windows, where a freshly-downloaded,
# unsigned build also gets a one-time Defender scan of those DLLs). Import it
# LAZILY — only when the first CT study is opened — so the main window appears
# immediately; users who never open a CT never pay the cost. Cached after the
# first call.
_CTViewer = None            # the CTViewer class, imported on first CT open
_CT_IMPORT_ERROR = ""       # set if the backend can't be imported


def _ct_viewer():
    global _CTViewer, _CT_IMPORT_ERROR
    if _CTViewer is None and not _CT_IMPORT_ERROR:
        try:
            # macOS renders CT with pygfx (wgpu→Metal): VTK's OpenGL→Metal path
            # hangs. Windows/Linux keep the proven VTK viewer. Both expose the
            # same CTViewer interface, so only the import target differs.
            if sys.platform == "darwin":
                from multi_dicomviewer.viewers.ct_viewer_pygfx import CTViewer
            else:
                from multi_dicomviewer.viewers.ct_viewer import CTViewer
            _CTViewer = CTViewer
        except Exception as exc:  # backend missing/broken — keep app usable for XA.
            _CT_IMPORT_ERROR = str(exc) or "unknown import error"
    if _CTViewer is None:
        hint = ("pip install -r requirements-mac.txt" if sys.platform == "darwin"
                else "pip install vtk")
        raise RuntimeError(
            t("CT viewer unavailable — its render backend failed to import:\n"
              "{err}\n\n{hint}", err=_CT_IMPORT_ERROR, hint=hint)
        )
    return _CTViewer()


#: modality -> zero-arg factory building its viewer. Add OCT / NM here as
#: those viewer modules land; nothing else needs to change. OTHER (dose /
#: exposure summary pages, secondary captures, and not-yet-special-cased
#: kinds) falls back to the generic grayscale cine/image viewer so it is
#: still viewable rather than "unsupported".
_VIEWER_FACTORY = {
    Modality.XA: XAViewer,
    Modality.CT: _ct_viewer,
    Modality.IVUS: IVUSViewer,
    Modality.SR: SRViewer,      # structured reports (dose report etc.)
    Modality.OTHER: XAViewer,
}

#: Max grid the visual layout picker offers (rows × cols). 6×6 = up to 36 panes.
_MAX_GRID_ROWS = 6
_MAX_GRID_COLS = 6
#: layout key "RxC" -> (rows, cols, pane-count). Every R∈1..6, C∈1..6 combo is a
#: valid layout, chosen from the visual grid picker (no more 1×2-vs-2×1 button
#: confusion). Panes fill left-to-right, top-to-bottom.
_LAYOUTS = {
    f"{r}x{c}": (r, c, r * c)
    for r in range(1, _MAX_GRID_ROWS + 1)
    for c in range(1, _MAX_GRID_COLS + 1)
}
#: The full _MAX_GRID_ROWS×_MAX_GRID_COLS grid is the MASTER arrangement. Every
#: other (non-1×1) layout shows the TOP-LEFT R×C sub-block of it, identified by
#: the canonical cell indices (row*_MAX_GRID_COLS + col, reading order). Because
#: the canonical column count is fixed, a given pane always sits in the same
#: on-screen spot no matter which layout you reached it from. 1×1 is special
#: (shows the active pane).
_LAYOUT_CELLS = {
    f"{r}x{c}": [rr * _MAX_GRID_COLS + cc
                 for rr in range(r) for cc in range(c)]
    for r in range(1, _MAX_GRID_ROWS + 1)
    for c in range(1, _MAX_GRID_COLS + 1)
}
#: Multi-pane layouts (used to gate MultiSync, which needs 2+ panes).
_MULTI_PANE = tuple(k for k, (_r, _c, n) in _LAYOUTS.items() if n >= 2)
#: Max cine viewers (panes) allowed to play at once. Starting a play beyond this
#: is refused with a message — guards against multi-pane decode/render overload.
_PLAY_CAP = 4
#: Separator between fields in a pane's top-band title (kept short — a long
#: em-dash read too wide). e.g. "● Pane 1 - YAMADA TARO - 20260615 - 3/12".
_PANE_SEP = " - "
#: Enough panes for the largest layout (6×6 = 36).
_MAX_PANES = max(c for _r, _c, c in _LAYOUTS.values())
#: Grid dimensions of the largest layout — used to reset stretch on every
#: row/col when shrinking back to a smaller grid.
_GRID_MAX_ROWS = max(r for r, _c, _cnt in _LAYOUTS.values())
_GRID_MAX_COLS = max(c for _r, c, _cnt in _LAYOUTS.values())

#: drag payload (source pane index) for swapping pane positions
PANE_MIME = "application/x-mdv-pane"


class LayoutGridPicker(QWidget):
    """Office-"insert table"-style visual layout chooser over a ROWS×COLS grid.
    DRAG to select ANY contiguous rectangle of cells (not just the top-left
    block): press a corner, drag to the opposite corner, release to pick. A
    single click picks that one cell. The "R×C" caption tracks the selection.
    Cells with a loaded pane are shaded brighter so you can see where to drag."""

    #: emitted with the 0-based inclusive rectangle (r0, c0, r1, c1) picked
    picked = pyqtSignal(int, int, int, int)

    _CELL = 22
    _GAP = 4
    _MARGIN = 7
    _CAPTION_H = 18

    def __init__(self, rows: int, cols: int, parent=None):
        super().__init__(parent)
        self._rows = rows
        self._cols = cols
        #: 0-based (row, col) drag anchor (None = not dragging) and the cell the
        #: cursor is over (None = not hovering). Both feed the highlight rect.
        self._anchor: tuple | None = None
        self._cur: tuple | None = None
        #: The layout currently in effect, as a 0-based inclusive rectangle —
        #: highlighted while the picker rests (no hover/drag) so you can see the
        #: on-screen arrangement, including a 1×1's specific pane cell.
        self._current_rect = (0, 0, 0, 0)
        #: 0-based (row, col) cells whose backing pane currently holds data.
        self._occupied: set = set()
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QSize:
        w = 2 * self._MARGIN + self._cols * self._CELL + (self._cols - 1) * self._GAP
        h = (2 * self._MARGIN + self._rows * self._CELL
             + (self._rows - 1) * self._GAP + self._CAPTION_H)
        return QSize(w, h)

    def set_current_rect(self, r0: int, c0: int, r1: int, c1: int) -> None:
        """The applied layout's 0-based inclusive master-grid rectangle."""
        self._current_rect = (r0, c0, r1, c1)
        self.update()

    def set_occupancy(self, occupied) -> None:
        """*occupied* = iterable of 0-based (row, col) cells that hold data."""
        self._occupied = set(occupied)
        self.update()

    def _cell_rect(self, r: int, c: int) -> QRect:
        x = self._MARGIN + c * (self._CELL + self._GAP)
        y = self._MARGIN + r * (self._CELL + self._GAP)
        return QRect(x, y, self._CELL, self._CELL)

    def _cell_at(self, x: float, y: float):
        """0-based (row, col) under (x, y), or None."""
        for r in range(self._rows):
            for c in range(self._cols):
                if self._cell_rect(r, c).adjusted(
                        -self._GAP // 2, -self._GAP // 2,
                        self._GAP // 2, self._GAP // 2).contains(int(x), int(y)):
                    return r, c
        return None

    @staticmethod
    def _norm(a, b):
        """Two 0-based cells → inclusive rectangle (r0, c0, r1, c1)."""
        return (min(a[0], b[0]), min(a[1], b[1]),
                max(a[0], b[0]), max(a[1], b[1]))

    def _hi_rect(self):
        """The rectangle to highlight right now: the live drag, else the hovered
        single cell, else the applied layout."""
        if self._anchor is not None and self._cur is not None:
            return self._norm(self._anchor, self._cur)
        if self._cur is not None:
            return (self._cur[0], self._cur[1], self._cur[0], self._cur[1])
        return self._current_rect

    def mousePressEvent(self, e):
        hit = self._cell_at(e.position().x(), e.position().y())
        if hit is not None:
            self._anchor = self._cur = hit
            self.update()

    def mouseMoveEvent(self, e):
        hit = self._cell_at(e.position().x(), e.position().y())
        if hit is not None and hit != self._cur:
            self._cur = hit
            self.update()

    def mouseReleaseEvent(self, e):
        if self._anchor is None:
            return
        hit = self._cell_at(e.position().x(), e.position().y()) or self._cur \
            or self._anchor
        r0, c0, r1, c1 = self._norm(self._anchor, hit)
        self._anchor = None
        self.update()
        self.picked.emit(r0, c0, r1, c1)

    def leaveEvent(self, _e):
        if self._anchor is None:                    # keep the drag rect while held
            self._cur = None
            self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#2b2b2b"))
        r0, c0, r1, c1 = self._hi_rect()
        on = QColor("#4a90d9")          # inside the selection (would apply)
        off_data = QColor("#8a8a8a")    # outside, pane HAS data
        off_empty = QColor("#3a3a3a")   # outside, pane is empty
        edge = QColor("#1f1f1f")
        for r in range(self._rows):
            for c in range(self._cols):
                sel = r0 <= r <= r1 and c0 <= c <= c1
                brush = on if sel else (
                    off_data if (r, c) in self._occupied else off_empty)
                p.setPen(QPen(edge, 1))
                p.setBrush(brush)
                p.drawRoundedRect(self._cell_rect(r, c), 3, 3)
        # caption "R×C"
        p.setPen(QColor("#e0e0e0"))
        p.drawText(
            QRect(0, self.height() - self._CAPTION_H,
                  self.width(), self._CAPTION_H),
            Qt.AlignmentFlag.AlignCenter,
            f"{r1 - r0 + 1}×{c1 - c0 + 1}",
        )


def _viewer_chrome_widgets(viewer) -> list:
    """Every toolbar/control widget of *viewer* — i.e. all leaf widgets in
    its top-level layout EXCEPT the image area. The image area is the one
    top-level item carrying a stretch factor (`addWidget(canvas, 1)` /
    `addLayout(imgrow, 1)`), which every viewer (XA, IVUS, CT-VTK, CT-pygfx)
    uses, so it is kept while the rest is hidden for "Max Image". Bare
    sub-layouts (XA's series-nav / transport rows added via addLayout) are
    walked recursively so their buttons toggle too."""
    lay = viewer.layout()
    out: list = []
    if lay is None:
        return out

    def walk(layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            w = item.widget()
            if w is not None:
                if not getattr(w, "_mdv_keep_on_max", False):
                    out.append(w)
            elif item.layout() is not None:
                walk(item.layout())

    for i in range(lay.count()):
        if lay.stretch(i) > 0:
            continue                       # the image area — keep it visible
        item = lay.itemAt(i)
        w = item.widget()
        if w is not None:
            # Widgets flagged _mdv_keep_on_max (e.g. the Play/seek transport)
            # stay visible in Max Image so playback is still usable.
            if not getattr(w, "_mdv_keep_on_max", False):
                out.append(w)
        elif item.layout() is not None:
            walk(item.layout())
    return out


def set_viewer_chrome_visible(viewer, visible: bool) -> None:
    """Hide (False) / restore (True) a viewer's toolbars for "Max Image".
    On hide, each chrome widget's own shown/hidden state is remembered (via
    isHidden(), which ignores ancestor visibility) so restoring brings back
    exactly what was showing — e.g. a collapsed Measure bar stays collapsed.
    Restoring is a no-op when the viewer was never hidden, so it can be called
    unconditionally (e.g. on every layout change) without forcing default-
    hidden rows (Measure bar, ECG strip) back on."""
    if not visible:
        widgets = _viewer_chrome_widgets(viewer)
        if not getattr(viewer, "_mdv_chrome_hidden", False):
            viewer._mdv_chrome_saved = {
                id(w): (not w.isHidden()) for w in widgets
            }
            viewer._mdv_chrome_hidden = True
        for w in widgets:
            w.setVisible(False)
    else:
        if not getattr(viewer, "_mdv_chrome_hidden", False):
            return                       # never hidden — leave widgets as-is
        saved = getattr(viewer, "_mdv_chrome_saved", None) or {}
        for w in _viewer_chrome_widgets(viewer):
            w.setVisible(saved.get(id(w), True))
        viewer._mdv_chrome_hidden = False
        viewer._mdv_chrome_saved = None


class _DragTitle(QLabel):
    """Pane title bar. Click activates the pane; dragging it onto another
    pane asks the shell to swap the two panes' grid positions."""

    def __init__(self, pane: "ViewerPane"):
        super().__init__(pane._idle_title())
        self._pane = pane
        self._press = None

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._press = e.position().toPoint()
        self._pane.activated.emit(self._pane)

    def mouseMoveEvent(self, e) -> None:
        if self._press is None:
            return
        if (
            e.position().toPoint() - self._press
        ).manhattanLength() < 12:
            return
        self._press = None
        md = QMimeData()
        md.setData(PANE_MIME, str(self._pane.index).encode("ascii"))
        drag = QDrag(self._pane)
        drag.setMimeData(md)
        drag.exec(Qt.DropAction.MoveAction)

    def mouseReleaseEvent(self, e) -> None:
        self._press = None


def _is_xa(viewer) -> bool:
    """True only for the XA viewer. IVUSViewer subclasses XAViewer, so an
    isinstance() check would wrongly match IVUS — gate on the modality the
    viewer declares instead."""
    return getattr(viewer, "handles_modality", "") == "XA"


def _is_cine(viewer) -> bool:
    """True for the cine viewers (XA and IVUS) that share the
    First/Prev/Next/Last transport and the F/A/Home/End/D/S keys."""
    return getattr(viewer, "handles_modality", "") in ("XA", "IVUS")


def _is_ct(viewer) -> bool:
    """True for the CT viewer — it has its own Series First/Prev/Next/Last
    row and shares the F/A/Home/End navigation keys with the cine viewers
    (but none of the cine transport keys, which CT uses for its tools)."""
    return getattr(viewer, "handles_modality", "") == "CT"


class _Placeholder(QWidget):
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        self._label = QLabel(text)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setStyleSheet("color:#888; font-size:13px;")
        lay.addWidget(self._label)

    def set_text(self, text: str) -> None:
        self._label.setText(text)


class _SeriesLoadWorker(QObject):
    """Reads + decodes a series off the GUI thread.

    The heavy part of opening a study — reading every slice and decoding the
    pixel data into a numpy HU volume (tens of seconds for a 600+ slice cardiac
    CT) — is pure CPU/IO with no Qt objects, so it runs here in a worker thread
    while the main window stays responsive. Only the returned LoadedSeries
    (numpy volume + header) crosses back; the GPU/VTK pipeline is built by the
    main thread in the ``done`` slot (Qt/GL objects must live on the GUI
    thread). The progress callback fires on THIS thread, so it emits a queued
    signal rather than touching the dialog directly."""

    progress = pyqtSignal(str, int, int)     # phase, done, total
    done = pyqtSignal(object)                 # LoadedSeries
    failed = pyqtSignal(str)                  # formatted error text

    def __init__(self, series):
        super().__init__()
        self._series = series

    def run(self) -> None:
        try:
            loaded = dicom_io.load_series(
                self._series,
                progress=lambda phase, d, tot: self.progress.emit(phase, d, tot),
            )
        except Exception as exc:                          # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(exc))
            return
        self.done.emit(loaded)


class _FolderScanWorker(QObject):
    """Scans / indexes a DICOM folder off the GUI thread.

    Reading a large study's per-file metadata (``scan_folder`` /
    ``index_files``) is pure pydicom + dataclass work with no Qt objects, so
    it runs here while the shell stays responsive — the user can pick another
    folder, drag series, etc. while a big CT study is still being indexed
    (the scan used to run on the GUI thread behind an app-modal dialog, which
    froze the whole window, including the native folder picker). The built
    Patient tree crosses back via ``done``; it's merged on the GUI thread."""

    progress = pyqtSignal(int, int)          # done, total
    done = pyqtSignal(object)                 # patients dict
    failed = pyqtSignal(str)

    def __init__(self, scan_fn):
        super().__init__()
        self._scan_fn = scan_fn

    def run(self) -> None:
        try:
            patients = self._scan_fn(
                lambda done, total: self.progress.emit(done, total))
        except Exception as exc:                          # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(exc))
            return
        self.done.emit(patients)


class _StillLabel(QLabel):
    """A demoted pane's frozen last image.

    When a pane is put to sleep to free memory (its heavy CT volume / XA clip
    is released) the last image it showed is kept here as a plain pixmap.
    Selecting the pane (a click) both activates it AND reloads its series — the
    user asked that picking a pane always make it the live/active one, even for
    CT (where only one is kept live at a time, so the previous CT sleeps). A
    mouse-wheel does the same. Double-click is left to the pane's own 1×1
    maximise handler."""

    def __init__(self, pane):
        super().__init__(pane)
        self._pane = pane
        self._src = None                       # unscaled source pixmap
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#000;")

    def set_image(self, pixmap) -> None:
        self._src = pixmap
        self._rescale()

    def _rescale(self) -> None:
        if self._src is None or self._src.isNull():
            return
        super().setPixmap(self._src.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):                    # noqa: N802 (Qt override)
        super().resizeEvent(e)
        self._rescale()

    def mousePressEvent(self, ev):               # noqa: N802 (Qt override)
        # Selecting a still pane makes it active AND reloads it (promote is a
        # no-op if it's mid-load). activated fires first so it's the active
        # pane before the load starts.
        self._pane.activated.emit(self._pane)
        self._pane.promote_requested.emit(self._pane)

    def wheelEvent(self, ev):                    # noqa: N802 (Qt override)
        self._pane.promote_requested.emit(self._pane)


class ViewerPane(QFrame):
    """One grid cell. Modality agnostic: it caches one viewer per modality
    and shows whichever the dropped/loaded series needs."""

    activated = pyqtSignal(object)            # this pane was clicked/used
    series_dropped = pyqtSignal(object, str)  # (pane, series_uid)
    study_dropped = pyqtSignal(object, str, str)  # (pane, study_uid, kind)
    paths_dropped = pyqtSignal(list)          # DICOM folder(s) and/or file(s)
    viewer_ready = pyqtSignal(object)         # a viewer was just created
    pane_move_requested = pyqtSignal(int, int)  # (src index, dest index)
    pane_cleared = pyqtSignal(object)         # the ✕ emptied this pane
    maximize_requested = pyqtSignal(object)   # 1×1 button / double-click → 1×1
    promote_requested = pyqtSignal(object)    # a demoted still pane was used

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self._viewers: dict[Modality, object] = {}
        self._cur_viewer = None
        self._shown_uid: str | None = None  # series currently displayed
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.Box)
        # Let the grid divide width / height EQUALLY in 1×2 / 2×2 even
        # when a loaded viewer's toolbar would otherwise demand a large
        # minimum (series-nav row + Measure + DICOM Tags ≈ 700 px).
        # With this tiny pane minimum the grid honours the stretch=1
        # split; viewer toolbars stay one row and just clip horizontally
        # if the pane is narrower than the toolbar's natural width —
        # acceptable, and unblocks the Studies dock to expand wide
        # enough for 10 thumbnails in a row.
        self.setMinimumSize(80, 80)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Title BAR = the dark grey band: a draggable title label (left) and
        # a close ✕ (right). The ✕ empties just THIS pane (keeps the grid
        # layout) so a new series can be dropped in.
        self._title = _DragTitle(self)
        self._title.setToolTip(
            t("Drag and drop onto another pane to swap their positions")
        )
        self._title.setStyleSheet(
            "padding:2px 6px; color:#ccc; background:transparent;"
        )
        # (The per-pane "1×1" button was removed — double-clicking the pane
        # still maximises it, and "Layout 1×1" is in the top bar.)
        self._close_btn = QPushButton("✕")
        self._close_btn.setToolTip(
            t("Close this pane's image (keeps the layout; returns to drop-waiting state)")
        )
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.setFixedWidth(26)
        self._close_btn.setFlat(True)
        self._close_btn.setStyleSheet(
            "QPushButton{color:#ccc; border:none; font-weight:bold;}"
            "QPushButton:hover{color:#fff; background:#c0392b;}"
        )
        self._close_btn.clicked.connect(self._on_close_clicked)
        self._title_bar = QWidget()
        self._title_bar.setStyleSheet("background:#2a2a2a;")
        _tb = QHBoxLayout(self._title_bar)
        _tb.setContentsMargins(0, 0, 0, 0)
        _tb.setSpacing(0)
        _tb.addWidget(self._title, 1)
        _tb.addWidget(self._close_btn)

        self._idle = _Placeholder(
            t("Pane {n}\n\n"
              "Drag & drop a series from the Info panel,\n"
              "or a DICOM folder, here", n=index + 1)
        )
        self._stack = QStackedWidget()
        self._stack.addWidget(self._idle)
        # "Still" page: a memory-light snapshot of a demoted (slept) pane. It
        # holds the last image the pane showed as a plain pixmap while the heavy
        # viewer (CT volume / XA clip) is freed; a data-needing interaction
        # reloads the real series (see promote_requested / _promote_pane).
        self._still = _StillLabel(self)
        self._stack.addWidget(self._still)
        self._still_series = None      # Series to reload on promote (None → live)
        self._still_state = None       # best-effort view snapshot (Phase 3)
        self._series_ref = None        # Series currently shown (for LRU/demote)
        self._volume_bytes = 0         # est. live volume size (bytes), mem budget

        self._box = QVBoxLayout(self)
        self._box.setContentsMargins(1, 1, 1, 1)
        self._box.setSpacing(0)
        self._box.addWidget(self._title_bar)
        self._box.addWidget(self._stack, 1)
        self._active_on = False
        self._full_bleed = False
        #: "Hide Buttons (Max Image)" state for this pane — hides the current
        #: viewer's toolbars (the title bar stays). Stored so it re-applies
        #: when the pane swaps to another modality's cached viewer.
        self._chrome_visible = True
        #: Compact transport (half-height) — on in multi-row layouts.
        self._compact = False
        self._refresh_border()
        # Catch drops over the title bar / placeholder too (viewers added
        # later are covered in _viewer_for).
        self._install_dnd(self)

    # ---------------------------------------------------------- appearance
    def _idle_title(self) -> str:
        return "● " + t("Pane {n}", n=self.index + 1) + _PANE_SEP + t("empty")

    def _set_pane_title(self, body: str) -> None:
        """Set the top-band title to "● Pane N - <body>" (or just "● Pane N"
        when body is empty)."""
        body = (body or "").strip()
        head = "● " + t("Pane {n}", n=self.index + 1)
        self._title.setText(
            f"{head}{_PANE_SEP}{body}" if body else head
        )

    def retranslate_ui(self) -> None:
        """Re-apply this pane's persistent strings after a live language
        change, and cascade to every cached viewer."""
        self._title.setToolTip(
            t("Drag and drop onto another pane to swap their positions")
        )
        self._close_btn.setToolTip(
            t("Close this pane's image (keeps the layout; returns to "
              "drop-waiting state)")
        )
        self._idle.set_text(
            t("Pane {n}\n\n"
              "Drag & drop a series from the Info panel,\n"
              "or a DICOM folder, here", n=self.index + 1)
        )
        if self._cur_viewer is None:
            self._title.setText(self._idle_title())
        for v in self._viewers.values():
            if v is not None and hasattr(v, "retranslate_ui"):
                v.retranslate_ui()

    def set_active(self, on: bool) -> None:
        self._active_on = on
        self._refresh_border()

    def set_full_bleed(self, on: bool) -> None:
        """1×1 mode: hide the title bar and drop the border/margins so the
        viewer fills the entire central frame."""
        self._full_bleed = on
        # Title bar hidden only in 1×1 full-bleed ("Max Image" keeps it).
        self._title_bar.setVisible(not on)
        m = 0 if on else 1
        self._box.setContentsMargins(m, m, m, m)
        self._refresh_border()

    def set_chrome_visible(self, visible: bool) -> None:
        """Show / hide the current viewer's toolbars for the shell-wide
        "Max Image" toggle. The title bar (top band) stays visible so the
        pane number / patient name remain readable."""
        self._chrome_visible = visible
        v = self._cur_viewer
        if v is not None:
            set_viewer_chrome_visible(v, visible)

    def _on_close_clicked(self) -> None:
        """✕ on the title bar — empty THIS pane (keep its grid slot) so a new
        series can be dropped. The pane's cached viewers are released; the
        shell re-syncs nav/MultiSync via pane_cleared."""
        self.activated.emit(self)        # make this the active pane first
        self.reset()
        self.pane_cleared.emit(self)

    def set_compact(self, on: bool) -> None:
        """Half-height transport for multi-row layouts (viewers that support
        it expose set_compact; others are left unchanged)."""
        self._compact = on
        v = self._cur_viewer
        if v is not None and hasattr(v, "set_compact"):
            v.set_compact(on)

    def _apply_pane_state(self, viewer) -> None:
        """Re-apply this pane's Max-Image / compact state to *viewer* — used
        when a (possibly cached) viewer becomes the shown one."""
        if viewer is None:
            return
        set_viewer_chrome_visible(viewer, self._chrome_visible)
        if hasattr(viewer, "set_compact"):
            viewer.set_compact(self._compact)

    def _refresh_border(self) -> None:
        if self._full_bleed:
            self.setStyleSheet("ViewerPane { border:0; }")
        else:
            self.setStyleSheet(
                "ViewerPane { border:2px solid %s; }"
                % ("#3b82f6" if self._active_on else "#444")
            )

    # ------------------------------------------------------------- viewers
    def _viewer_for(self, modality: Modality):
        """Return the cached viewer for *modality*, building it on first
        use. Returns None (and the pane shows the error) if unsupported."""
        if modality in self._viewers:
            return self._viewers[modality]
        factory = _VIEWER_FACTORY.get(modality)
        if factory is None:
            return None
        try:
            viewer = factory()
        except Exception as exc:
            self._show_message(str(exc))
            self._viewers[modality] = None
            return None
        self._viewers[modality] = viewer
        self._stack.addWidget(viewer)
        self._install_dnd(viewer)
        self.viewer_ready.emit(viewer)
        return viewer

    def show_series(self, loaded, title: str,
                    pane_label: str | None = None) -> None:
        """Load *loaded* into this pane. *title* is the viewer's own header
        (kept descriptive); *pane_label* is the top-band body
        ("Name - YYYYMMDD - SeriesNo/InstanceNo"), defaulting to *title*."""
        viewer = self._viewer_for(loaded.modality)
        if viewer is None:
            mod = loaded.modality.value
            self._show_message(
                t("{mod} viewer is not implemented.\n"
                  "(Supported: XA, CT, IVUS. OCT/OFDI/NM planned.)", mod=mod)
            )
            self._cur_viewer = None
            self._set_pane_title(t("{mod} (unsupported)", mod=mod))
            return
        # Make the viewer the current (visible) stack page BEFORE loading, so
        # every setVisible() inside load_series()/_relayout() runs against an
        # on-screen parent. On a freshly built viewer the page is otherwise
        # still hidden, and macOS (Cocoa) can drop the show of a child toggled
        # visible under a hidden QStackedWidget page — that's why a biplane XA's
        # Bi/Lt/Rt bar failed to appear on first load until a re-drop forced a
        # second _relayout. Already-current viewers make this a no-op.
        self._stack.setCurrentWidget(viewer)
        viewer.load_series(loaded, title)
        self._cur_viewer = viewer
        self._clear_still()          # a live load supersedes any frozen snapshot
        self._set_pane_title(pane_label if pane_label is not None else title)
        # A freshly built or re-shown viewer must match this pane's current
        # Max-Image / compact state.
        self._apply_pane_state(viewer)

    def _show_message(self, text: str) -> None:
        # Reuse the idle widget's label to surface load/availability errors.
        self._idle.findChild(QLabel).setText(text)
        self._stack.setCurrentWidget(self._idle)

    def current_viewer(self):
        return self._cur_viewer

    def has_data(self) -> bool:
        """True if this pane currently shows a loaded series (drives the
        layout-picker's bright-vs-dark cell shading). A demoted "still" pane
        still counts as having data — its series is just frozen to save
        memory, not gone."""
        return self._cur_viewer is not None or self._still_series is not None

    def is_still(self) -> bool:
        """True while this pane is a memory-freed frozen snapshot."""
        return self._still_series is not None

    def demote_to_still(self, series, state=None) -> bool:
        """Sleep this pane: keep its last image on screen as a static pixmap
        but free the live viewer's volume/clip. Returns False if there was
        nothing live to demote (e.g. still loading). *series* is what to
        reload on promote; *state* is an optional view snapshot for restore."""
        v = self._cur_viewer
        if v is None or series is None:
            return False
        # Prefer a viewer-provided snapshot() (needed where a GL/GPU surface —
        # VTK on Windows, wgpu on Mac — doesn't render into QWidget.grab());
        # fall back to grabbing the widget for plain-Qt viewers (XA canvas).
        if hasattr(v, "snapshot"):
            try:
                pix = v.snapshot()
            except Exception:
                pix = v.grab()
        else:
            pix = v.grab()
        if pix is None or pix.isNull():
            return False
        self._still.set_image(pix)
        self._still_series = series
        self._still_state = state
        # Release the heavy data. CT's VTK clear() frees the volume but leaves
        # _loaded_uid set, which would make _open_series' fast-path show an
        # empty viewer on reload — force a real reload by clearing it. (XA's
        # clear() already resets _loaded_uid.)
        v.clear()
        try:
            v._loaded_uid = ""
        except Exception:
            pass
        self._cur_viewer = None
        self._stack.setCurrentWidget(self._still)
        return True

    def _clear_still(self) -> None:
        self._still_series = None
        self._still_state = None
        self._still.set_image(QPixmap())

    def is_loaded(self, modality, series_uid: str) -> bool:
        """True if this pane already has *series_uid* loaded into the
        cached viewer for *modality* — lets the shell skip the whole
        disk-read + decode + viewer-rebuild pipeline when the user
        returns to a series they were already viewing."""
        v = self._viewers.get(modality)
        return v is not None and getattr(v, "_loaded_uid", "") == series_uid

    def switch_to_loaded(self, modality, pane_label: str) -> None:
        """Bring the cached viewer for *modality* to the front without
        calling load_series. Caller must verify with is_loaded() first.
        *pane_label* is the top-band body (see show_series)."""
        viewer = self._viewers.get(modality)
        if viewer is None:
            return
        self._stack.setCurrentWidget(viewer)
        self._cur_viewer = viewer
        self._clear_still()          # returning to a live viewer ends still mode
        self._set_pane_title(pane_label)

    def set_shown_series(self, uid: str | None) -> None:
        self._shown_uid = uid

    def shown_series_uid(self) -> str | None:
        return self._shown_uid

    def all_viewers(self) -> list:
        return [v for v in self._viewers.values() if v is not None]

    def reset(self) -> None:
        for v in self.all_viewers():
            v.clear()
        self._cur_viewer = None
        self._shown_uid = None
        self._series_ref = None
        self._clear_still()
        self._idle.findChild(QLabel).setText(
            t("Pane {n}\n\n"
              "Drag & drop a series from the Info panel,\n"
              "or a DICOM folder, here", n=self.index + 1)
        )
        self._stack.setCurrentWidget(self._idle)
        self._title.setText(self._idle_title())

    # ------------------------------------------------------ click / drop
    def mousePressEvent(self, _event) -> None:
        self.activated.emit(self)

    def mouseDoubleClickEvent(self, _event) -> None:
        # Double-click landing directly on the pane frame → 1×1. (Child
        # widgets — title band, image — are handled in eventFilter.) Only
        # outside 1×1, so a double-click that commits a measurement in the
        # maximised view isn't hijacked.
        if not self._full_bleed:
            self.maximize_requested.emit(self)

    def _is_titlebar_button(self, obj) -> bool:
        """True only for the title-bar ✕ button — it keeps its own click
        action. Everything else, including the dark title band and its
        draggable label, double-clicks to 1×1 (window-title metaphor)."""
        return obj is self._close_btn

    def _install_dnd(self, widget) -> None:
        """Make *widget* and every descendant forward drags to this pane.

        Once a viewer is loaded its child widgets (image canvas, sliders…)
        sit on top of the pane; they don't accept drops, and Qt does not
        reliably bubble a drop up to the pane through them. Enabling
        acceptDrops + an event filter on the whole subtree guarantees a
        folder/series dropped anywhere on the pane is still caught here,
        so a 2nd/3rd study can be dropped while one is already showing."""
        widget.setAcceptDrops(True)
        widget.installEventFilter(self)
        for child in widget.findChildren(QWidget):
            child.setAcceptDrops(True)
            child.installEventFilter(self)

    def _dispatch_mime(self, md) -> bool:
        """Route a dropped pane/series/folder to the shell. True if used."""
        if md.hasFormat(PANE_MIME):
            src = int(bytes(md.data(PANE_MIME)).decode("ascii"))
            self.pane_move_requested.emit(src, self.index)
            return True
        if md.hasFormat(SERIES_MIME):
            uid = bytes(md.data(SERIES_MIME)).decode("utf-8")
            self.activated.emit(self)
            self.series_dropped.emit(self, uid)
            return True
        if md.hasFormat(STUDY_MIME):
            study_uid, _, kind = bytes(
                md.data(STUDY_MIME)).decode("utf-8").partition("\x1f")
            self.activated.emit(self)
            self.study_dropped.emit(self, study_uid, kind)
            return True
        # DICOM folder(s)/file(s) dropped onto the pane. A FOLDER drop loads
        # the whole folder; a FILE drop loads ONLY the dropped file(s) — not
        # the rest of their containing folder. Dropping SEVERAL folders spreads
        # them across panes from here on (see MainWindow._spread_panes).
        paths = [u.toLocalFile() for u in md.urls()]
        paths = [p for p in paths if p]
        if paths:
            # Make this the active pane first so the first series auto-opens
            # here, not in some other pane.
            self.activated.emit(self)
            self.paths_dropped.emit(paths)
            return True
        return False

    def _wants(self, md) -> bool:
        return (
            md.hasFormat(PANE_MIME)
            or md.hasFormat(SERIES_MIME)
            or md.hasFormat(STUDY_MIME)
            or md.hasUrls()
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._wants(event.mimeData()):
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragEnterEvent) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        if self._dispatch_mime(event.mimeData()):
            event.acceptProposedAction()

    def eventFilter(self, obj, event) -> bool:
        """Catch drags/drops AND mouse clicks on the loaded viewer's
        child widgets and treat them as if they landed on the pane
        itself — so clicking anywhere on the image (not just the
        control strip) makes this the active pane."""
        t = event.type()
        if t in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            if self._wants(event.mimeData()):
                event.acceptProposedAction()
                return True
        elif t == QEvent.Type.Drop:
            if self._dispatch_mime(event.mimeData()):
                event.acceptProposedAction()
                return True
        elif t == QEvent.Type.MouseButtonPress:
            # Activate the pane, but DON'T consume the event — the click
            # must still reach the viewer (measure, crosshair, etc.).
            self.activated.emit(self)
        elif t == QEvent.Type.MouseButtonDblClick:
            # Double-click anywhere on the pane (the dark title band, the
            # image, the placeholder) maximises to 1×1 — only the 1×1 / ✕
            # buttons are exempt. Suppressed in 1×1 so viewer double-click
            # actions (e.g. committing a polygon measurement) still work.
            if not self._full_bleed and not self._is_titlebar_button(obj):
                self.maximize_requested.emit(self)
                return True
        return super().eventFilter(obj, event)


class MainWindow(QMainWindow):
    def __init__(self, initial_folder: str | None = None):
        super().__init__()
        # Resolve the UI language FIRST — before any dock / layout-bar / pane /
        # menu is constructed — because each string is translated when its
        # widget is created. (A language change still applies on next launch.)
        i18n.set_language(settings.load_language())
        self.setWindowTitle(f"{APP_NAME}  {build_string()}")
        self.resize(1500, 950)
        self.setAcceptDrops(True)  # drop a DICOM folder anywhere on the window
        self._patients = {}
        self._series_by_uid: dict[str, Series] = {}
        self._cur_xa: Series | None = None
        # Last opened series per modality, so each modality's series-nav
        # (First / Prev / Next / Last) resumes from where the user left
        # it after switching panes/modalities.
        self._last_by_modality: dict[str, Series] = {}
        # Last opened series per study NODE, so clicking a Study row (not a
        # specific series) restores the LAST series the user was on in that
        # study instead of always jumping back to series #1. Keyed by
        # (study_uid, kind) — NOT study_uid alone — because one study_uid
        # can appear as two nodes (e.g. XA + OT acquired the same date);
        # keying by uid only let an XA-node click resume the sibling OT
        # series (the "clicking XA jumps to OT" bug).
        self._last_by_study: dict[tuple[str, str], Series] = {}
        self._anon = False                  # Anonymize toggle (display only)
        # DICOM tags overlaid on images — restored from the previous run
        # and remembered per modality so XA / IVUS / CT each keep their
        # own preferred tag list (the same list also seeds the
        # Export-DICOM dialog's filename fields).
        self._tag_keywords_by_modality: dict[str, list[str]] = (
            settings.load_tag_keywords_by_modality()
        )
        # Per-modality Export-DICOM dialog field selection, also
        # remembered across sessions.
        self._export_fields_by_modality: dict[str, list[str]] = (
            settings.load_export_fields_by_modality()
        )
        # Anonymization profile (which tags the Anonymize toggle / Export
        # (Anon DICOM) blank). Load the saved one, else the built-in default,
        # and push it into the anonymize module so display + export agree.
        _anon_saved = settings.load_anon_profile()
        if _anon_saved is not None:
            self._anon_tags, self._anon_emptify_private = _anon_saved
        else:
            self._anon_tags = list(anonymize.DEFAULT_ANON_TAGS)
            self._anon_emptify_private = anonymize.DEFAULT_EMPTIFY_PRIVATE
        anonymize.set_anon_profile(self._anon_tags, self._anon_emptify_private)
        self._overlay_hidden = False        # hide the on-image tag text
        # Measurement log, keyed by StudyInstanceUID, kept until the app
        # exits (survives folder reloads / series switches).
        self._measure_history: dict[str, list] = {}
        self._study_by_series_uid: dict[str, str] = {}
        self._cur_study_uid: str | None = None
        self._hist_dialog: MeasureHistoryDialog | None = None
        # Background series loads in flight: pane -> (thread, worker, dialog).
        # A series load runs off the GUI thread (see _open_series) so the app
        # stays usable while a big CT reads; this tracks the live jobs so a
        # pane isn't double-loaded and the threads aren't GC'd mid-run.
        self._loads: dict[object, tuple] = {}
        # Background folder-scan/index jobs (worker -> (thread, dlg, open_in_pane,
        # spread_roots)) so the scan runs off the GUI thread and its threads
        # aren't GC'd mid-run.
        self._scans: dict[object, tuple] = {}
        # Per-series MP4 export range [start, end] (0-based frame indices),
        # reported by a cine viewer's Play-range markers. A series only
        # appears here while its range is narrower than the full clip; a
        # full range is removed so the export defaults to every frame.
        self._mp4_ranges: dict[str, tuple[int, int]] = {}

        # --- study browser dock ---
        self.browser = StudyPanel()
        self.browser.series_chosen.connect(self._on_series_chosen)
        self.browser.paths_dropped.connect(self._on_paths_dropped)
        self.browser.study_clicked.connect(self._on_study_clicked)
        self.browser.delete_requested.connect(self._delete_node)
        self.browser.delete_all_requested.connect(self._delete_all_nodes)
        self.browser.export_requested.connect(self._on_export_requested)
        dock = QDockWidget(t("Studies"), self)
        dock.setWidget(self.browser)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self._studies_dock = dock

        # Dock separator (the drag handle between the Studies dock and the
        # central viewer grid). Qt's default 4 px is hard to grab; 8 px is a
        # thin line that still reads as inert until the cursor hovers (then
        # blue). The separator's width is BOTH its look and its hit zone in
        # QMainWindow, so this stays thin to avoid eating content width.
        self.setStyleSheet(
            "QMainWindow::separator {"
            " background:#a8a8a8; width:8px; height:8px;"
            "}"
            "QMainWindow::separator:hover {"
            " background:#4a90d9;"
            "}"
            # Rounded "Layout 1×3" dropdown (and any other top-bar combo) so it
            # matches the rounded buttons across the panes.
            "QComboBox {"
            " border:1px solid #c8c8c8; border-radius:6px;"
            " padding:2px 8px; background:#ededed; color:#101010;"
            "}"
        )

        # --- configurable viewer grid ---
        self._layout_key = "1x1"
        #: Master-grid cell indices currently shown, in reading order — the
        #: selected (possibly non-top-left) rectangle. Set by _apply_layout;
        #: unused in the special 1×1 layout (which shows the active pane).
        self._layout_cells = [0]
        #: "Max Image" (toolbars hidden) state, shared across every pane.
        self._chrome_hidden = False
        # Bi / Lt / Rt is per-pane now (each viewer stores its own plane
        # choice); the toolbar buttons mirror whichever pane is active.
        #: Shared DICOM-tag overlay text size (pt) for every viewer/modality.
        self._tag_font_pt = TAG_FONT_PT_DEFAULT
        #: Debounce for the Tag-size slider. Applying a new size touches EVERY
        #: viewer in EVERY pane (each XA/IVUS canvas repaints; a CT pane does a
        #: full VTK render). Doing that on every valueChanged froze a 2x3 grid
        #: while dragging — so we collapse a drag into a single apply ~160 ms
        #: after the last change (and on release).
        self._tag_font_timer = QTimer(self)
        self._tag_font_timer.setSingleShot(True)
        self._tag_font_timer.setInterval(160)
        self._tag_font_timer.timeout.connect(self._apply_tag_font_pt)
        self._panes: list[ViewerPane] = []
        for i in range(_MAX_PANES):
            pane = ViewerPane(i)
            pane.activated.connect(self._set_active_pane)
            pane.series_dropped.connect(self._on_series_dropped)
            pane.study_dropped.connect(self._on_study_dropped)
            # Folder/file dropped ON a pane → import AND open into THAT pane
            # (several folders → one per pane, from this one on).
            pane.paths_dropped.connect(
                lambda paths, p=pane: self._load_paths(paths, p))
            pane.viewer_ready.connect(self._wire_viewer)
            pane.pane_move_requested.connect(self._swap_panes)
            pane.pane_cleared.connect(self._on_pane_cleared)
            pane.maximize_requested.connect(self._maximize_pane)
            pane.promote_requested.connect(self._promote_pane)
            self._panes.append(pane)
        # Panes in grid-slot order (drag a title onto another to swap).
        self._order: list[ViewerPane] = list(self._panes)
        self._active = self._panes[0]
        # LRU bookkeeping for the memory cap: a monotonic "use clock" stamped on
        # each pane whenever it's loaded or activated. When more than the cap of
        # a modality is live at once, the least-recently-used one is demoted to a
        # frozen still (its volume/clip freed) so many series can stay on screen
        # without exhausting memory. See _touch_pane / _enforce_live_caps.
        self._use_seq = 0
        self._pane_touch: dict[ViewerPane, int] = {}
        self._live_cap = self._load_live_caps()   # {Modality: live-at-once cap}
        # Per-modality RAW-volume memory budget (bytes) for LIVE panes. The
        # count cap alone treats a 200-slice CT the same as a 640-slice
        # 0.25 mm one (~0.7 GB) — so a high cap (e.g. CT=4) lets several huge
        # volumes go live at once and the GPU/VTK build of the newest exhausts
        # memory → black CT. This budget demotes the oldest live volume(s) when
        # the live total (incl. the incoming one) would exceed it, so small
        # series still use the full count cap while big ones can't pile up.
        # Actual footprint ≈ 2-3× the raw bytes (numpy + VTK float copy).
        self._live_bytes_budget = {
            Modality.CT: 1_100_000_000, Modality.XA: 2_000_000_000}

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(2)

        central = QWidget()
        col = QVBoxLayout(central)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        # Wrap the top layout bar in a width-decoupled scroll container so it
        # does NOT pin the central area's minimum width. Otherwise the bar's
        # full-width minimum capped how wide the Studies dock could grow (worst
        # at small logical resolutions, e.g. 1920×1080 at 150% scaling). The
        # bar's buttons elide as the area narrows; only at extreme narrowness
        # does a horizontal scrollbar appear. Height stays one row.
        _bar = self._build_layout_bar()
        _bar_scroll = QScrollArea()
        _bar_scroll.setWidget(_bar)
        _bar_scroll.setWidgetResizable(True)
        _bar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        _bar_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        _bar_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        _bar_scroll.setMinimumWidth(0)       # don't pin the central min width
        _bar_scroll.setFixedHeight(_bar.sizeHint().height())
        col.addWidget(_bar_scroll)
        col.addWidget(self._grid_host, 1)
        self.setCentralWidget(central)
        # Studies dock: minimum width = roughly one minimum-size
        # thumbnail (_THUMB_MIN_PX 80 + item padding + spacing +
        # scrollbar + frames). The tree just gets its own horizontal
        # scrollbar when this narrow; default opens at a comfy 380.
        self._studies_dock.setMinimumWidth(140)
        self.resizeDocks(
            [self._studies_dock], [380], Qt.Orientation.Horizontal
        )

        self.browser.anonymize_toggled.connect(self._set_anonymized)
        self.browser.anon_settings_requested.connect(self._open_anon_settings)
        self.browser.fit_dock_width_requested.connect(
            self._fit_studies_dock_width
        )
        self.browser.resize_dock_step_requested.connect(
            self._step_studies_dock_width
        )
        self.browser.dicom_info_toggled.connect(self._on_dicom_info_btn)
        self.browser.dicom_tags_requested.connect(self._open_tag_dialog_active)

        self._build_menu()
        self._build_shortcuts()
        self._apply_layout(self._layout_key)
        self.statusBar().showMessage(t("Open a DICOM folder to begin."))

        if initial_folder and os.path.isdir(initial_folder):
            self._load_folder(initial_folder)

    # ------------------------------------------------------------------ menu
    def _build_language_menu(self) -> None:
        """Language menu: pick the UI language. Items are shown in each
        language's own native name; the current one is checked. The change is
        persisted and applied LIVE (see retranslate_ui) — no restart."""
        lm = self.menuBar().addMenu(t("&Language"))
        group = QActionGroup(self)
        group.setExclusive(True)
        cur = i18n.get_language()
        for code, name in i18n.enabled_languages().items():
            act = QAction(name, self, checkable=True)
            act.setChecked(code == cur)
            act.triggered.connect(lambda _c=False, cc=code: self._set_language(cc))
            group.addAction(act)
            lm.addAction(act)

    def _set_language(self, code: str) -> None:
        if code == i18n.get_language():
            return
        settings.save_language(code)
        i18n.set_language(code)
        # Live switch: re-apply every persistent string in place (dialogs and
        # right-click menus are rebuilt on open, so they pick up the new
        # language for free; on-image overlays re-evaluate t() on repaint).
        self.retranslate_ui()

    def _rebuild_menu(self) -> None:
        """Tear the menu bar down and build it fresh in the current language.
        The menu's checkable actions carry state (Anonymize, Hide overlay), so
        it is rebuilt rather than text-patched; that state is restored after."""
        mb = self.menuBar()
        for top in mb.actions():
            menu = top.menu()
            if menu is None:
                continue
            for a in menu.actions():
                # Drop shortcuts BEFORE the replacements are created so Qt
                # never sees two actions claiming the same key.
                a.setShortcut(QKeySequence())
                a.setShortcuts([])
                a.setParent(None)
                a.deleteLater()
            menu.setParent(None)
            menu.deleteLater()
        mb.clear()
        self._build_menu()
        # Restore checkable state without re-firing the toggles.
        for act, state in (
            (self._anon_act, self._anon),
            (self._hide_overlay_act, self._overlay_hidden),
        ):
            act.blockSignals(True)
            act.setChecked(state)
            act.blockSignals(False)

    def retranslate_ui(self) -> None:
        """Apply the active language to every PERSISTENT widget in place, so a
        language change takes effect immediately (no restart). On-demand
        dialogs / windows and right-click menus are constructed fresh each time
        they open, so they need nothing here; on-image overlays re-evaluate
        t() on their next repaint (forced below)."""
        self._rebuild_menu()
        self.setWindowTitle(f"{APP_NAME}  {build_string()}")
        self._studies_dock.setWindowTitle(t("Studies"))

        # Top layout bar.
        self._info_btn.setText(
            t("◀ Hide Studies") if self._studies_dock.isVisible()
            else t("Show Studies ▶")
        )
        self._info_btn.setHelpToolTip(
            t("Show/hide the left Info window (study tree)")
        )
        self._layout_btn.setText(self._layout_btn_text())
        self._layout_btn.setToolTip(t("Drag over the grid to show any block of panes (not just the top-left)."))
        self._pane_step_btn.setText(t("Pane →"))
        self._pane_step_btn.setHelpToolTip(
            t("Flip the fullscreen (1×1) view to the next loaded pane; wraps "
              "around from the last back to the first. Available only in the "
              "1×1 layout.")
        )
        self._clear_all_btn.setText(t("✕ Clear All"))
        self._clear_all_btn.setHelpToolTip(
            t("Clear the image from every pane (same as each pane's ✕). "
              "The layout and the studies list are kept.")
        )
        self._hide_btns_btn.setText(t("Hide Buttons (Max Image)"))
        self._hide_btns_btn.setHelpToolTip(
            t("Hide every pane's toolbars/title to maximise the image "
              "(for presentation). Use Show Buttons to bring them back.")
        )
        self._show_btns_btn.setText(t("Show Buttons"))
        self._show_btns_btn.setHelpToolTip(t("Restore every pane's toolbars."))
        self._tags_btn.setText(t("DICOM Info"))
        self._tags_btn.setHelpToolTip(
            t("Left-click: show/hide DICOM info on the image\n"
              "Right-click: choose which tag data to show")
        )
        self._tagsz_lbl.setText(t("Tag size:"))
        self._tag_font_slider.setToolTip(t("DICOM tag text size (all panes)"))
        self._settings_btn.setText(t("Settings"))
        self._settings_btn.setToolTip(
            t("Display count, angio image quality, CT colour map"))

        # Panes (placeholders, tooltips) + every cached viewer.
        for pane in self._panes:
            pane.retranslate_ui()
        # Re-derive the "● Pane N — <series>" titles of loaded panes.
        self._refresh_pane_titles()

        # Study tree / thumbnail browser.
        if hasattr(self.browser, "retranslate_ui"):
            self.browser.retranslate_ui()

        # Open non-modal tool windows follow the language live too (modal
        # dialogs can't be reached while the language menu is open, so they
        # just pick it up next open). Only those that implement retranslate_ui.
        for attr in ("_coreg_win", "_rupture_win", "_ortho_win",
                     "_dicomcheck_win", "_dicomfolder_win", "_hist_dialog"):
            win = getattr(self, attr, None)
            if (win is not None and hasattr(win, "retranslate_ui")
                    and win.isVisible()):
                win.retranslate_ui()

        self.update()

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu(t("&File"))

        open_act = QAction(t("&Open DICOM folder…"), self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._choose_folder)
        m.addAction(open_act)

        open_file_act = QAction(t("Open DICOM &file…"), self)
        open_file_act.setShortcut("Ctrl+Shift+O")
        open_file_act.setToolTip(
            t("Open one or more individual DICOM files (not the whole folder)")
        )
        open_file_act.triggered.connect(self._choose_files)
        m.addAction(open_file_act)

        clear_act = QAction(t("&Clear viewers"), self)
        clear_act.triggered.connect(self._clear_all)
        m.addAction(clear_act)

        m.addSeparator()
        quit_act = QAction(t("&Quit"), self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        m.addAction(quit_act)

        self._build_language_menu()

        vm = self.menuBar().addMenu(t("&View"))
        self._anon_act = QAction(t("&Anonymize"), self)
        self._anon_act.setCheckable(True)
        self._anon_act.setShortcut("Ctrl+Shift+A")
        self._anon_act.setToolTip(t(
            "Mask patient/case info on all on-screen displays "
            "(files unchanged)"
        ))
        self._anon_act.toggled.connect(self._set_anonymized)
        vm.addAction(self._anon_act)

        tags_act = QAction(t("DICOM tag overlay items…"), self)
        tags_act.triggered.connect(self._tags_for_active_pane)
        vm.addAction(tags_act)

        self._hide_overlay_act = QAction(t("Hide DICOM overlay"), self)
        self._hide_overlay_act.setCheckable(True)
        # Q is the quick single-key toggle (shown in the menu); Ctrl+H
        # kept so the old shortcut still works. (D used to do this — it
        # is now unbound on CT and serves the cine play/2x toggle on
        # XA/IVUS panes via _build_shortcuts below.)
        self._hide_overlay_act.setShortcuts(
            [QKeySequence("Q"), QKeySequence("Ctrl+H")]
        )
        self._hide_overlay_act.setToolTip(t(
            "Show/hide the DICOM tag text drawn on the image (Q)"
        ))
        self._hide_overlay_act.toggled.connect(self._toggle_overlay_hidden)
        vm.addAction(self._hide_overlay_act)

        vm.addSeparator()
        exp_act = QAction(t("Export DICOM tag overlay settings…"), self)
        exp_act.triggered.connect(self._export_tag_conditions)
        vm.addAction(exp_act)

        imp_act = QAction(t("Import DICOM tag overlay settings…"), self)
        imp_act.triggered.connect(self._import_tag_conditions)
        vm.addAction(imp_act)

        tm = self.menuBar().addMenu(t("&Tools"))
        # MultiSync-Viewer retired 2026-07-09: its capabilities (multi-IVUS
        # frame+rotation sync, save/load, MP4 export) are now the CoSync
        # tool below (a strict superset). The old window (multisync_window.py)
        # and _open_multisync are kept on disk but no longer reachable.
        self._rupture_act = QAction(t("Rupture-Predictor…"), self)
        self._rupture_act.triggered.connect(self._open_rupture_predictor)
        tm.addAction(self._rupture_act)
        self._ortho_act = QAction(t("Orthogonal-View…"), self)
        self._ortho_act.setToolTip(t(
            "Pick a vector on the active XA image and get the two C-arm "
            "angles whose view is orthogonal to it "
            "(available whenever the active pane shows an XA series "
            "with C-arm positioner angles)"
        ))
        self._ortho_act.triggered.connect(self._open_orthogonal_view)
        tm.addAction(self._ortho_act)
        self._coaxial_act = QAction(t("Coaxial Eval…"), self)
        self._coaxial_act.setToolTip(t(
            "Draw a labelled Line (GC / proxLAD / …) on the same vessel in "
            "2+ angio views, then compute the 3-D GC-to-vessel angle "
            "(available whenever a visible pane shows an XA series with "
            "C-arm positioner angles)"
        ))
        self._coaxial_act.triggered.connect(self._open_coaxial_eval)
        tm.addAction(self._coaxial_act)
        self._coreg_act = QAction(t("CoSync…"), self)
        self._coreg_act.setToolTip(t(
            "Co-register IVUS pull-back frames to positions on the angio "
            "vessel: trace a guide, pin CoSync landmarks, then scrubbing "
            "the IVUS drives a marker along the angio "
            "(needs at least one IVUS and one XA series loaded)"
        ))
        self._coreg_act.triggered.connect(self._open_coreg)
        tm.addAction(self._coreg_act)

        tm.addSeparator()
        self._dicomcheck_act = QAction(t("DicomCheck…"), self)
        self._dicomcheck_act.setToolTip(t(
            "Scan a folder and delete non-DICOM files (and empty folders)"
        ))
        self._dicomcheck_act.triggered.connect(self._open_dicom_check)
        tm.addAction(self._dicomcheck_act)
        self._dicomfolder_act = QAction(t("DicomFolder…"), self)
        self._dicomfolder_act.setToolTip(t(
            "Organize DICOM files into sub-folders by date / modality / study"
        ))
        self._dicomfolder_act.triggered.connect(self._open_dicom_folder)
        tm.addAction(self._dicomfolder_act)
        self._bintag_act = QAction(t("Export binary DICOM tag…"), self)
        self._bintag_act.setToolTip(t(
            "Write a binary VR tag (OB/OW/UN — shown as <binary>) to text "
            "as hex / Base64 / Latin-1"
        ))
        self._bintag_act.triggered.connect(self._open_bintag_export)
        tm.addAction(self._bintag_act)
        self._sync_layout_gate()

    def _sync_layout_gate(self) -> None:
        """Gate Tools menu items by current layout / contents:
        Orthogonal-View needs at least one VISIBLE pane to hold an XA series
        with C-arm positioner angles (it works on every visible pane — angio
        panes are pickable, the rest open view-only)."""
        if hasattr(self, "_ortho_act"):
            self._ortho_act.setEnabled(self._any_visible_pane_has_angles())
        if hasattr(self, "_coaxial_act"):
            self._coaxial_act.setEnabled(self._any_visible_pane_has_angles())

    @staticmethod
    def _viewer_has_positioner_angles(v) -> bool:
        """True iff *v* is an XA viewer whose first plane has both
        PositionerPrimaryAngle and PositionerSecondaryAngle present."""
        if not _is_xa(v):
            return False
        planes = getattr(v, "_planes", []) or []
        for p in planes:
            ds = getattr(p, "_ds", None)
            if ds is None:
                continue
            if (getattr(ds, "PositionerPrimaryAngle", None) is not None
                    and getattr(ds, "PositionerSecondaryAngle",
                                  None) is not None):
                return True
        return False

    def _any_visible_pane_has_angles(self) -> bool:
        """True iff at least one currently visible pane shows an XA
        series with C-arm angles — the gate for Orthogonal-View."""
        if self._layout_key == "1x1":
            v = (self._active.current_viewer()
                 if self._active is not None else None)
            return self._viewer_has_positioner_angles(v)
        for pane in self._shown_panes():
            if not pane.isVisible():
                continue
            if self._viewer_has_positioner_angles(pane.current_viewer()):
                return True
        return False

    def _open_multisync(self) -> None:
        """Launch the MultiSync IVUS window, carrying over the current
        1×2 / 2×2 layout and the IVUS series shown in the panes."""
        from multi_dicomviewer.ui.multisync_window import MultiSyncWindow
        if self._layout_key not in _MULTI_PANE:
            return
        ivus = [
            se
            for p in self._patients.values()
            for st in p.studies.values()
            for se in st.series.values()
            if se.modality == Modality.IVUS
        ]
        if not ivus:
            QMessageBox.information(
                self, t("MultiSync"),
                t("No IVUS series are loaded. Open a folder with IVUS "
                  "pull-backs first."),
            )
            return
        # Pre-fill each slot from the matching pane (in display order);
        # non-IVUS panes leave their slot empty. We also hand over the
        # pane's current frame index AND its viewer instance so MultiSync
        # can (a) open the slot on the same frame the pane is showing and
        # (b) live-mirror frame changes both ways while the window is up.
        preset: list = []
        preset_frames: list = []
        preset_viewers: list = []
        for pane in self._shown_panes():
            se = self._series_by_uid.get(pane.shown_series_uid())
            if se is not None and se.modality == Modality.IVUS:
                v = pane.current_viewer()
                preset.append(se)
                preset_frames.append(int(getattr(v, "_frame", 0)))
                preset_viewers.append(v)
            else:
                preset.append(None)
                preset_frames.append(0)
                preset_viewers.append(None)
        # MultiSync uses 2 slots for a 2-pane main layout (1×2 / 2×1) and 4
        # otherwise. len(preset) == the current layout's pane count (preset has
        # one entry per shown pane).
        self._multisync = MultiSyncWindow(
            ivus,
            layout_count=len(preset),
            preset=preset,
            preset_frames=preset_frames,
            preset_viewers=preset_viewers,
        )
        self._as_taskbar_window(self._multisync)
        # Open maximized so the synchronised grid + sync editor have
        # room without the user resizing first.
        self._multisync.showMaximized()
        self._multisync.raise_()

    def _open_coreg(self) -> None:
        """Launch the CoSync window, seeded with the IVUS and XA
        series currently shown in the panes (IVUS pull-backs + the
        representative angio view[s]). Needs ≥1 IVUS and ≥1 XA."""
        from multi_dicomviewer.ui.coreg_window import CoregWindow
        # One CoSync pane per shown pane holding an IVUS/XA series, mirroring
        # the on-screen arrangement 1:1 (no series_uid de-dup, which dropped
        # a pane when two shown panes shared a UID, e.g. biplane). Each spec
        # is (series, plane_index): for a biplane pane we send exactly the
        # projection(s) actually displayed — Lt→plane 0, Rt→plane 1, and Bi
        # (both shown) → both planes as two CoSync views.
        specs: list = []
        for pane in self._shown_panes():
            se = self._series_by_uid.get(pane.shown_series_uid())
            if se is None or se.modality not in (Modality.IVUS, Modality.XA):
                continue
            v = pane.current_viewer()
            fr = int(getattr(v, "_frame", 0))    # keep the frame on screen now
            # Carry the source IVUS's display ROTATION and play-order (Reverse)
            # so the CoSync pane opens matching what the user set up — the same
            # way a CT short-axis already carries its rotation (angio panes have
            # neither: rotation 0, not reversed).
            cv = getattr(v, "canvas", None)
            rot = (float(cv.free_rotation())
                   if cv is not None and hasattr(cv, "free_rotation") else 0.0)
            rev = bool(getattr(v, "_reversed", False))
            side = v.current_side() if hasattr(v, "current_side") else None
            if side == "Bi":
                specs.append((se, 0, fr, rot, rev))
                specs.append((se, 1, fr, rot, rev))
            elif side == "Rt":
                specs.append((se, 1, fr, rot, rev))
            else:                                # "Lt", single-plane, or none
                specs.append((se, 0, fr, rot, rev))
        # Include any CT pane currently in short-axis (Stretch MPR) mode as a
        # synthetic pull-back — it joins the CoSync grid like an IVUS and can
        # be landmark-synced against the real IVUS / other short-axes.
        cpr_specs = []
        for pane in self._shown_panes():
            v = pane.current_viewer()
            if getattr(v, "handles_modality", "") == "CT" \
                    and hasattr(v, "cpr_active") and v.cpr_active():
                spec = v.cpr_cosync_spec()
                if spec is not None:
                    cpr_specs.append(spec)
        specs = specs[:6]
        # A CT short-axis is a driver too (is_ivus in the CoSync window), so it
        # satisfies the "need a pull-back" requirement.
        has_ivus = (any(s.modality == Modality.IVUS for s, *_ in specs)
                    or bool(cpr_specs))
        specs = (specs + cpr_specs)[:6]
        if not has_ivus:
            QMessageBox.information(
                self, t("CoSync"),
                t("Show at least one IVUS pull-back in the panes first. "
                  "(XA angio is optional — with no XA this behaves like a "
                  "multi-IVUS sync viewer.)"),
            )
            return
        self._coreg_win = CoregWindow(specs)
        self._as_taskbar_window(self._coreg_win)
        self._coreg_win.showMaximized()
        self._coreg_win.raise_()

    def _open_dicom_check(self) -> None:
        """Launch the DicomCheck tool (delete non-DICOM files / empty dirs)."""
        from multi_dicomviewer.ui.dicom_check_window import DicomCheckWindow
        self._dicomcheck_win = DicomCheckWindow(
            start_dir=self._last_open_dir())
        self._as_taskbar_window(self._dicomcheck_win)
        self._dicomcheck_win.show()
        self._dicomcheck_win.raise_()

    def _open_bintag_export(self) -> None:
        """Launch the binary-tag export dialog (hex / Base64 / Latin-1). The
        source file is chosen from a currently-open pane (4×3 grid)."""
        from multi_dicomviewer.ui.binary_tag_export import BinaryTagExportDialog
        entries = []
        for i, pane in enumerate(self._order):       # on-screen grid-slot order
            uid = pane.shown_series_uid()
            se = self._series_by_uid.get(uid) if uid else None
            path = se.files[0] if (se and se.files) else None
            entries.append({"slot": i + 1,
                            "label": (se.label if se else ""),
                            "path": path})
        BinaryTagExportDialog(entries, self).exec()

    def _open_dicom_folder(self) -> None:
        """Launch the DicomFolder tool (organize DICOM files by tag)."""
        from multi_dicomviewer.ui.dicom_folder_window import DicomFolderWindow
        self._dicomfolder_win = DicomFolderWindow(
            start_dir=self._last_open_dir())
        self._as_taskbar_window(self._dicomfolder_win)
        self._dicomfolder_win.show()
        self._dicomfolder_win.raise_()

    def _last_open_dir(self):
        """Best-effort starting folder for the file tools — the last folder
        the user opened, or None."""
        d = getattr(self, "_last_dir", None)
        return d if isinstance(d, str) and d else None

    def _active_display_image(self):
        """QImage of the frame currently shown in the active pane's
        XA/IVUS viewer (the windowed canvas image), or None."""
        v = (self._active.current_viewer()
             if self._active is not None else None)
        if v is None:
            return None
        canvas = getattr(v, "canvas", None)
        qimg = getattr(canvas, "_qimg", None) if canvas is not None else None
        if qimg is not None and not qimg.isNull():
            return qimg
        return None

    @staticmethod
    def _launch_browser_maximized(uri: str) -> None:
        """Open *uri* maximised. Prefer Chrome/Edge with --new-window
        + an explicit window-position/window-size at the primary
        screen's available geometry (–-start-maximized alone is often
        ignored when the user already has a Chrome session running).
        Falls back to the OS default browser if neither is found."""
        import shutil
        import subprocess
        import webbrowser
        # Primary-screen geometry so the new window covers the full
        # work area regardless of the existing browser session's
        # state. availableGeometry excludes the taskbar, so the
        # result is effectively a maximised window.
        from PyQt6.QtGui import QGuiApplication
        scr = QGuiApplication.primaryScreen()
        geom = scr.availableGeometry() if scr is not None else None
        if geom is not None:
            pos_arg = f"--window-position={geom.x()},{geom.y()}"
            size_arg = f"--window-size={geom.width()},{geom.height()}"
        else:
            pos_arg = None
            size_arg = None
        for exe in ("msedge", "chrome"):
            p = shutil.which(exe)
            if p is None:
                # Common Windows install paths as a second chance.
                for guess in (
                    fr"C:\Program Files (x86)\Microsoft\Edge\Application\{exe}.exe",
                    fr"C:\Program Files\Microsoft\Edge\Application\{exe}.exe",
                    fr"C:\Program Files\Google\Chrome\Application\{exe}.exe",
                    fr"C:\Program Files (x86)\Google\Chrome\Application\{exe}.exe",
                ):
                    if os.path.exists(guess):
                        p = guess
                        break
            if p:
                cmd = [p, "--new-window", "--start-maximized"]
                if pos_arg and size_arg:
                    cmd.extend([pos_arg, size_arg])
                cmd.append(uri)
                try:
                    subprocess.Popen(cmd)
                    return
                except OSError:
                    pass
        webbrowser.open(uri)

    #: Rupture-Predictor hand-off encoding params. 2× upscale + JPEG
    #: q92 keeps picks at ~100 px/mm (still well above the user's
    #: clipboard-screenshot reference of 88 px/mm) while shrinking the
    #: per-frame payload roughly 5–10× vs the previous 3×-PNG, so the
    #: encoding loop and the browser parse of the inline data URLs are
    #: both noticeably faster.
    _RP_HANDOFF_UPSCALE = 2
    _RP_HANDOFF_JPEG_QUALITY = 92

    @staticmethod
    def _qimage_to_data_url(qimg, fmt: str = "PNG",
                             quality: int = -1) -> str:
        """QImage -> 'data:image/...;base64,...' for HTML hand-off.

        Pass ``fmt='JPEG'`` and a quality 0–100 for compact lossy
        encodings (used by the Rupture-Predictor hand-off); the default
        PNG path is kept for any caller that needs lossless."""
        from PyQt6.QtCore import QBuffer, QByteArray
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QBuffer.OpenModeFlag.WriteOnly)
        qimg.save(buf, fmt, quality)
        buf.close()
        mime = ("image/jpeg" if fmt.upper() in ("JPG", "JPEG")
                else "image/png")
        return (f"data:{mime};base64,"
                + bytes(ba.toBase64()).decode("ascii"))

    @classmethod
    def _rp_upscaled_data_url(cls, qimg) -> str:
        """SmoothTransformation upscale (factor = _RP_HANDOFF_UPSCALE)
        + JPEG (quality = _RP_HANDOFF_JPEG_QUALITY) base64. The single
        encoding helper for both the IVUS frame burst and the XA
        single-image fallback so any future tweak (factor / format /
        quality) lands in one place."""
        s = cls._RP_HANDOFF_UPSCALE
        if s != 1:
            qimg = qimg.scaled(
                qimg.width() * s, qimg.height() * s,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        return cls._qimage_to_data_url(
            qimg, "JPEG", cls._RP_HANDOFF_JPEG_QUALITY,
        )

    def _ivus_frame_burst(self, viewer, progress=None):
        """For an IVUS viewer, render a burst of frames around the
        currently displayed one as upscaled JPEG data URLs (see
        :data:`_RP_HANDOFF_UPSCALE` / :data:`_RP_HANDOFF_JPEG_QUALITY`).

        Returns ``(frames, current_index)`` — ``frames`` is a list of
        ``data:image/jpeg;base64,...`` strings, ``current_index`` is
        the position in that list corresponding to the frame the user
        was looking at. Returns ``(None, None)`` if the burst can't be
        built (no plane / no frames).

        *progress* — optional ``(done, total, message)`` callback fired
        each frame so the shell can update a QProgressDialog (encoding
        101 frames at 3× upscale takes a few seconds)."""
        from multi_dicomviewer.viewers.image_canvas import to_qimage
        planes = getattr(viewer, "_planes", []) or []
        if not planes:
            return None, None
        plane = planes[getattr(viewer, "_active", 0)]
        total = getattr(plane, "total_frames", 0)
        if total <= 0:
            return None, None
        cur = max(0, min(int(getattr(viewer, "_frame", 0)), total - 1))
        # ± _RP_FRAME_HALF_BURST frames around cur, clamped. 50 each
        # side → 101 frames @ 30fps ≈ ±1.7 s, enough to span at least
        # one full cardiac cycle for boundary verification. Payload at
        # 2×-upscaled JPEG q92 (see _rp_upscaled_data_url) runs only
        # a few MB embedded inline so the browser parse is fast.
        _RP_FRAME_HALF_BURST = 50
        start = max(0, cur - _RP_FRAME_HALF_BURST)
        end = min(total - 1, cur + _RP_FRAME_HALF_BURST)
        burst_total = end - start + 1
        frames = []
        for k, idx in enumerate(range(start, end + 1)):
            if progress is not None:
                progress(
                    k, burst_total,
                    t("Encoding IVUS frame {k} / {total}…",
                      k=k + 1, total=burst_total)
                )
                if getattr(progress, "cancelled", False):
                    return None, None
            try:
                arr = plane.frame(idx)
            except Exception:
                continue
            qimg = to_qimage(arr)
            if qimg.isNull():
                continue
            frames.append(self._rp_upscaled_data_url(qimg))
        if progress is not None:
            progress(burst_total, burst_total, t("Frames ready."))
        if not frames:
            return None, None
        return frames, cur - start

    @staticmethod
    def _read_positioner_angles(ds):
        """Extract (primary, secondary) angle from a DICOM dataset, with
        NaN / missing collapsed to None so downstream code can branch on
        'we have angles or we don't' with a single test."""
        try:
            b = float(getattr(ds, "PositionerPrimaryAngle", float("nan")))
            if math.isnan(b):
                b = None
        except (TypeError, ValueError):
            b = None
        try:
            a = float(getattr(ds, "PositionerSecondaryAngle", float("nan")))
            if math.isnan(a):
                a = None
        except (TypeError, ValueError):
            a = None
        return b, a

    def _ortho_panels_from_xa_viewer(self, v, pane_label: str = "") -> list:
        """Build OrthogonalView panel specs for an XA viewer's currently-
        shown plane(s). A pane on "Bi" exports both planes (Front+Lateral);
        a pane on Lt/Rt exports just that single active plane — regardless
        of the grid layout, since Bi/Lt/Rt is now per-pane."""
        planes = getattr(v, "_planes", []) or []
        if not planes:
            return []
        # Mirror the viewer's own canvas usage: both planes only when the
        # second canvas is actually shown (this pane is on "Bi").
        is_dual = (
            len(planes) >= 2
            and getattr(v, "canvas2", None) is not None
            and not getattr(v, "canvas2").isHidden()
        )
        canvases = [getattr(v, "canvas", None)]
        plane_idxs = [getattr(v, "_active", 0)]
        if is_dual:
            canvases.append(getattr(v, "canvas2", None))
            plane_idxs = [0, 1]

        out = []
        for canvas, pi in zip(canvases, plane_idxs):
            if canvas is None or pi >= len(planes):
                continue
            qimg = getattr(canvas, "_qimg", None)
            if qimg is None or qimg.isNull():
                continue
            beta, alpha = self._read_positioner_angles(planes[pi]._ds)
            # Subtitle: in 1×1-biplane keep the plane name (Front /
            # Lateral); in multi-pane layouts label by source pane.
            if is_dual:
                subtitle = planes[pi].name
            elif pane_label:
                subtitle = pane_label
            else:
                subtitle = ""
            out.append({
                "qimg": qimg,
                "spacing": getattr(canvas, "spacing_mm", None),
                "beta": beta,
                "alpha": alpha,
                "subtitle": subtitle,
                "pickable": beta is not None and alpha is not None,
            })
        return out

    def _ortho_panel_view_only(self, v, pane_label: str) -> list:
        """Clone a non-XA / no-angle pane's current frame as a view-only
        panel (image shown so the user can refer to it, but clicks are
        inert). Falls back to a widget-grab when the viewer doesn't
        expose a canvas QImage."""
        qimg = None
        spacing = None
        canvas = getattr(v, "canvas", None)
        if canvas is not None:
            qimg = getattr(canvas, "_qimg", None)
            spacing = getattr(canvas, "spacing_mm", None)
        if qimg is None or qimg.isNull():
            return []  # nothing safe to show
        return [{
            "qimg": qimg,
            "spacing": spacing,
            "beta": None,
            "alpha": None,
            "subtitle": pane_label,
            "pickable": False,
        }]

    def _open_orthogonal_view(self) -> None:
        """Tools ▸ Orthogonal-View — clone every currently visible pane
        into the Orthogonal-View tool.

        * Panes with an XA series whose header carries the C-arm
          positioner angles are PICKABLE: clicking two points on the
          image returns the two orthogonal C-arm angles.
        * Panes without angles (CT, IVUS, XA missing the tags, etc.)
          are shown VIEW-ONLY for reference but cannot be clicked.

        The gate is enforced by _sync_layout_gate: the menu item is
        only enabled when at least one visible pane has angles. The
        checks below are guard-rails for the manual entry points
        (keyboard shortcut, script-triggered)."""
        from multi_dicomviewer.ui.orthogonal_view import OrthogonalViewWindow

        # Enumerate visible panes in display order.
        visible_panes = [p for p in self._shown_panes() if p.isVisible()]

        panels = []
        for pane in visible_panes:
            v = pane.current_viewer()
            if v is None:
                continue
            pane_label = t("Pane {n}", n=self._panes.index(pane) + 1)
            if _is_xa(v):
                panels.extend(self._ortho_panels_from_xa_viewer(
                    v, pane_label
                ))
            else:
                panels.extend(self._ortho_panel_view_only(v, pane_label))

        # Need at least one pickable panel — otherwise the tool is
        # pointless. The layout-bar gate normally already prevents this.
        if not any(p["pickable"] for p in panels):
            QMessageBox.information(
                self, t("Orthogonal-View"),
                t("No visible pane shows an XA series with C-arm "
                  "positioner angles."),
            )
            return

        # Title from the active pane's series if available.
        title = "—"
        if self._active is not None:
            uid = self._active.shown_series_uid()
            se = self._series_by_uid.get(uid) if uid else None
            if se is not None:
                title = se.label
        self._ortho_win = OrthogonalViewWindow(panels, title)
        self._as_taskbar_window(self._ortho_win)
        self._ortho_win.showMaximized()
        self._ortho_win.raise_()

    def _coaxial_lines_from_xa_viewer(self, v) -> list:
        """Collect vessel-labelled Line measures from an XA viewer's
        canvases, one entry per labelled line ready for
        core.coaxial.compute_coaxial_angles.

        Each line carries the C-arm angles STAMPED on it when it was drawn
        (canvas.view_angles → m['beta']/m['alpha'] in _commit_draft). This
        is correct for biplane shown single-plane too, where one canvas
        displays different planes — and hence different angles — over time,
        so the live plane angle can no longer be trusted at collection time.
        Both canvas and canvas2 are always scanned; the hidden one is simply
        empty. Older lines without a stamp fall back to the canvas's current
        view angle."""
        lines = []
        for canvas in (getattr(v, "canvas", None), getattr(v, "canvas2", None)):
            if canvas is None:
                continue
            spacing = getattr(canvas, "spacing_mm", None) or (1.0, 1.0)
            live = getattr(canvas, "view_angles", None)
            for m in getattr(canvas, "measures", []):
                if m.get("type") != "line" or not m.get("vessel"):
                    continue
                pts = m.get("pts", [])
                if len(pts) < 2:
                    continue
                beta = m.get("beta")
                alpha = m.get("alpha")
                if beta is None or alpha is None:
                    if live is None:
                        continue
                    beta, alpha = live
                lines.append({
                    "label": m["vessel"],
                    "beta": beta,
                    "alpha": alpha,
                    "line_2d": (tuple(pts[0]), tuple(pts[1])),
                    "spacing": spacing,
                })
        return lines

    def _open_coaxial_eval(self) -> None:
        """Tools ▸ Coaxial Eval — gather every vessel-labelled Line drawn on
        the visible XA panes, reconstruct each vessel's 3-D direction from
        its 2+ views and report the GC-to-vessel angles."""
        from multi_dicomviewer.core import coaxial
        from multi_dicomviewer.ui.coaxial_dialog import CoaxialResultDialog

        # Scan EVERY pane currently on screen (in reading order). Using
        # _shown_panes() — not self._order[:count] — is essential: the shown
        # panes are the selected layout rectangle (e.g. a 2×1 shows cells 0 & 3,
        # not 0 & 1), so the old slice missed the lower pane and only the top
        # biplane's two images were collected. Each biplane pane contributes
        # both its canvases (frontal + lateral).
        visible_panes = self._shown_panes()

        all_lines = []
        for pane in visible_panes:
            v = pane.current_viewer()
            if v is None or not _is_xa(v):
                continue
            all_lines.extend(self._coaxial_lines_from_xa_viewer(v))

        if not all_lines:
            QMessageBox.information(
                self, t("Coaxial Eval"),
                t("No vessel-labelled lines found.\n\n"
                  "Draw a Line on the same vessel in 2+ angio views, then "
                  "right-click each line ▸ Vessel type ▸ pick GC / proxLAD / "
                  "etc. before running Coaxial Eval."),
            )
            return

        view_counts: dict[str, int] = {}
        for ln in all_lines:
            view_counts[ln["label"]] = view_counts.get(ln["label"], 0) + 1

        result = coaxial.compute_coaxial_angles(all_lines)
        dlg = CoaxialResultDialog(result, view_counts, parent=self)
        dlg.exec()

    @staticmethod
    def _bundled_resource(rel: str) -> str:
        """Resolve a file shipped under ``multi_dicomviewer/resources``
        in both modes:

        * dev (``python -m multi_dicomviewer``): the file sits next to
          this source tree, so we resolve via ``__file__``.
        * PyInstaller one-dir (the .exe / .app build): the spec copies
          the resources into the bundle preserving the layout
          ``multi_dicomviewer/resources/<file>`` next to the binary's
          internal modules, so the SAME relative resolution works.

        Returns an absolute filesystem path."""
        import pathlib as _pl
        here = _pl.Path(__file__).resolve().parent  # …/multi_dicomviewer/ui
        return str(here.parent / "resources" / rel)

    def _open_rupture_predictor(self) -> None:
        """Open the NATIVE Rupture-Predictor on the active pane's
        displayed image + its DICOM pixel calibration (so the tool can
        skip the manual CH/CV calibration entirely).

        Replaces the old browser hand-off so the app is self-contained.
        Set ``MDV_RUPTURE_BROWSER=1`` to fall back to the legacy HTML in
        an external browser — kept for A/B numeric comparison until the
        native port is signed off (see :meth:`_open_rupture_predictor_browser`).
        """
        if os.environ.get("MDV_RUPTURE_BROWSER"):
            self._open_rupture_predictor_browser()
            return
        from multi_dicomviewer.ui.rupture_predictor_window import (
            RupturePredictorWindow,
        )
        se = (self._series_by_uid.get(self._active.shown_series_uid())
              if self._active is not None else None)
        spacing = dicom_io.series_spacing_mm(se) if se is not None else None
        calib = None
        if spacing is not None:
            row_mm, col_mm = spacing
            if row_mm and row_mm > 0 and col_mm and col_mm > 0:
                # px/mm = 1 / (mm/px); PixelSpacing is (row, col). No 2×
                # upscale here — that was only a JPEG data-URL granularity
                # hack; the native canvas reaches sub-pixel picks via zoom.
                calib = (1.0 / col_mm, 1.0 / row_mm)   # (hpxmm, vpxmm)
        v = (self._active.current_viewer()
             if self._active is not None else None)
        is_ivus = (v is not None
                   and getattr(v, "handles_modality", "") == "IVUS")
        plane = None
        frame_index = 0
        if is_ivus:
            planes = getattr(v, "_planes", []) or []
            if planes:
                plane = planes[getattr(v, "_active", 0)]
                frame_index = int(getattr(v, "_frame", 0))
        # Hand over the plane + current frame for IVUS (the stepper decodes
        # frames lazily across the whole pull-back) or the single displayed
        # QImage for XA. Keep a reference so the window isn't GC'd.
        if plane is None:
            qimg = self._active_display_image()
            if qimg is None:
                QMessageBox.warning(
                    self, t("Rupture-Predictor"),
                    t("The active pane has no image displayed."))
                return
            self._rupture_win = RupturePredictorWindow(
                qimage=qimg, calib=calib)
        else:
            self._rupture_win = RupturePredictorWindow(
                plane=plane, frame_index=frame_index, calib=calib)
        self._as_taskbar_window(self._rupture_win)
        self._rupture_win.showMaximized()
        self._rupture_win.raise_()
        parts = [t("IVUS frame stepper") if plane is not None
                 else t("displayed image")]
        if calib is not None:
            parts.append(t("DICOM calibration (CH/CV step skipped)"))
        self.statusBar().showMessage(
            t("Rupture-Predictor opened with {parts}.",
              parts=" + ".join(parts)))

    def _open_rupture_predictor_browser(self) -> None:
        """Legacy browser Rupture-Predictor (HTML hand-off). Retained as a
        fallback behind ``MDV_RUPTURE_BROWSER`` for A/B comparison against
        the native port; slated for removal once parity is confirmed.

        The hand-off rides in a generated, self-contained session HTML
        (window.MDV_HANDOFF) — a data URL is too big for a query string
        and a file:// page can't fetch a sibling image."""
        import json
        import pathlib
        import tempfile
        # The DICOM-aware variant of Rupture-Predictor is bundled with
        # Multi-DicomViewer (multi_dicomviewer/resources/). The original
        # standalone HTML at C:\CC_Product\Rupture-Predictor\ stays put
        # as the MP4 / clipboard variant — outside this app's purview.
        src = self._bundled_resource("Rupture-Predictor.html")
        if not os.path.exists(src):
            QMessageBox.warning(
                self, t("Rupture-Predictor"), t("Not found:\n{src}", src=src),
            )
            return
        handoff: dict = {}
        se = (self._series_by_uid.get(self._active.shown_series_uid())
              if self._active is not None else None)
        spacing = dicom_io.series_spacing_mm(se) if se is not None else None
        if spacing is not None:
            row_mm, col_mm = spacing
            if row_mm and row_mm > 0 and col_mm and col_mm > 0:
                # px/mm = 1 / (mm/px); PixelSpacing is (row, col).
                handoff["hpxmm"] = round(1.0 / col_mm, 5)
                handoff["vpxmm"] = round(1.0 / row_mm, 5)
        # Scale the DICOM-derived calibration to match the upscaled
        # image that goes into the data URL — see _rp_upscaled_data_url
        # for why we upscale at all (finer pick granularity than the
        # native 512² DICOM gives).
        if "hpxmm" in handoff:
            handoff["hpxmm"] = round(
                handoff["hpxmm"] * self._RP_HANDOFF_UPSCALE, 5
            )
        if "vpxmm" in handoff:
            handoff["vpxmm"] = round(
                handoff["vpxmm"] * self._RP_HANDOFF_UPSCALE, 5
            )

        # IVUS gets the full FRAME BURST around the current frame so
        # the user can step ±50 frames in Rupture-Predictor to verify
        # the rupture / adventitia boundary. Picks lock to the frame
        # of the first point so the four picks always reference the
        # same anatomy. Non-IVUS (XA snapshot) still hands over a
        # single image since stepping doesn't apply.
        v = (self._active.current_viewer()
             if self._active is not None else None)
        is_ivus = (v is not None
                   and getattr(v, "handles_modality", "") == "IVUS")
        if is_ivus:
            # Encoding ~101 frames at 3× upscale takes a couple of
            # seconds — show a progress dialog so the user knows we
            # didn't just freeze the app.
            prog = QProgressDialog(
                t("Preparing IVUS frames for Rupture-Predictor…"),
                t("Cancel"), 0, 1, self,
            )
            prog.setWindowModality(Qt.WindowModality.ApplicationModal)
            prog.setMinimumDuration(0)
            prog.setAutoClose(False)
            prog.setAutoReset(False)
            prog.setValue(0)
            prog.show()
            QApplication.processEvents()

            def _cb(done, total, msg):
                if prog.wasCanceled():
                    _cb.cancelled = True
                    return
                if total and prog.maximum() != total:
                    prog.setMaximum(total)
                prog.setValue(done)
                if msg:
                    prog.setLabelText(msg)
                QApplication.processEvents()
            _cb.cancelled = False

            frames, cur_idx = self._ivus_frame_burst(v, progress=_cb)
            if _cb.cancelled:
                prog.close()
                return
            prog.setLabelText(t("Writing Rupture-Predictor session HTML…"))
            QApplication.processEvents()
            # We hold prog open until after the HTML write below.
            self._rupture_progress = prog
            if frames:
                handoff["frames"] = frames
                handoff["currentFrame"] = cur_idx
        if "frames" not in handoff:
            img = self._active_display_image()
            if img is not None:
                # Same upscale + JPEG path the IVUS burst uses, so
                # XA snapshots and IVUS frames behave identically.
                handoff["image"] = self._rp_upscaled_data_url(img)
        if not handoff:
            self._launch_browser_maximized(pathlib.Path(src).as_uri())
            return
        try:
            html = open(src, encoding="utf-8").read()
            inject = ("<script>window.MDV_HANDOFF="
                      + json.dumps(handoff) + ";</script>\n")
            if "</head>" in html:
                html = html.replace("</head>", inject + "</head>", 1)
            else:
                html = inject + html
            tmp = os.path.join(
                tempfile.gettempdir(),
                "Rupture-Predictor_mdv_session.html",
            )
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(html)
        except OSError as exc:
            prog = getattr(self, "_rupture_progress", None)
            if prog is not None:
                prog.close()
                self._rupture_progress = None
            QMessageBox.warning(self, t("Rupture-Predictor"), str(exc))
            self._launch_browser_maximized(pathlib.Path(src).as_uri())
            return
        self._launch_browser_maximized(pathlib.Path(tmp).as_uri())
        # Tear down the encoding-progress dialog (if we showed one).
        prog = getattr(self, "_rupture_progress", None)
        if prog is not None:
            prog.close()
            self._rupture_progress = None
        parts = []
        if "frames" in handoff:
            parts.append(t("IVUS frame burst ({n} frames)",
                           n=len(handoff['frames'])))
        elif "image" in handoff:
            parts.append(t("displayed image"))
        if "hpxmm" in handoff:
            parts.append(t("DICOM calibration (CH/CV step skipped)"))
        self.statusBar().showMessage(
            t("Rupture-Predictor opened with {parts}.",
              parts=" + ".join(parts))
        )

    # --------------------------------------------------------- screen layout
    def _build_layout_bar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 3, 6, 3)

        self._info_btn = FitButton(t("◀ Hide Studies"))
        self._info_btn.setHelpToolTip(
            t("Show/hide the left Info window (study tree)")
        )
        self._info_btn.clicked.connect(self._toggle_info)
        row.addWidget(self._info_btn)
        row.addSpacing(12)

        # Visual layout picker (Office "insert table" style): a "Layout ▾"
        # button opens a popup grid you hover/drag to choose ROWS×COLS — no more
        # mixing up 1×2 vs 2×1.
        self._layout_btn = QPushButton(self._layout_btn_text())
        self._layout_btn.setToolTip(t("Drag over the grid to show any block of panes (not just the top-left)."))
        # This menu-button renders square on macOS; give it the same light-grey
        # rounded border as the pane buttons so the top bar matches.
        self._layout_btn.setStyleSheet(
            "QPushButton {"
            " border:1px solid #c8c8c8; border-radius:6px;"
            " padding:3px 10px 3px 8px; background:#ededed; color:#101010; }")
        self._layout_menu = QMenu(self._layout_btn)
        self._layout_picker = LayoutGridPicker(_GRID_MAX_ROWS, _GRID_MAX_COLS)
        self._layout_picker.picked.connect(self._on_layout_picked)
        _wa = QWidgetAction(self._layout_menu)
        _wa.setDefaultWidget(self._layout_picker)
        self._layout_menu.addAction(_wa)
        self._layout_btn.setMenu(self._layout_menu)
        self._layout_menu.aboutToShow.connect(self._refresh_layout_picker)
        row.addWidget(self._layout_btn)

        # "Pane →": flip the fullscreen 1×1 view to the next loaded pane, in
        # reading order (left→right, then the next row's left), wrapping from the
        # last back to the first — a keyboard-free way to page through the
        # loaded images one at a time. Only meaningful in 1×1, so it is greyed
        # out in every multi-pane layout (see _apply_layout).
        self._pane_step_btn = FitButton(t("Pane →"))
        self._pane_step_btn.setHelpToolTip(
            t("Flip the fullscreen (1×1) view to the next loaded pane; wraps "
              "around from the last back to the first. Available only in the "
              "1×1 layout.")
        )
        self._pane_step_btn.clicked.connect(self._cycle_active_pane)
        self._pane_step_btn.setEnabled(getattr(self, "_layout_key", "1x1")
                                       == "1x1")
        row.addSpacing(8)
        row.addWidget(self._pane_step_btn)

        # "Clear All": empty every pane at once (same as each pane's ✕). The
        # layout is kept; the studies list is untouched. The leading ✕ matches
        # the per-pane close glyph and, because FitButton elides from the right,
        # stays visible (Clear's intent still readable) when the bar is narrow.
        self._clear_all_btn = FitButton(t("✕ Clear All"))
        self._clear_all_btn.setHelpToolTip(
            t("Clear the image from every pane (same as each pane's ✕). "
              "The layout and the studies list are kept.")
        )
        self._clear_all_btn.clicked.connect(self._clear_all)
        row.addSpacing(8)
        row.addWidget(self._clear_all_btn)

        row.addSpacing(12)
        # "Max Image" toggle pair: Hide Buttons strips every pane down to just
        # the image (and the title bar) for presentation; Show Buttons brings
        # the toolbars back. Applies to ALL panes at once.
        self._hide_btns_btn = FitButton(t("Hide Buttons (Max Image)"))
        self._hide_btns_btn.setHelpToolTip(
            t("Hide every pane's toolbars/title to maximise the image "
              "(for presentation). Use Show Buttons to bring them back.")
        )
        self._hide_btns_btn.clicked.connect(lambda: self._set_chrome_hidden(True))
        row.addWidget(self._hide_btns_btn)
        self._show_btns_btn = FitButton(t("Show Buttons"))
        self._show_btns_btn.setHelpToolTip(t("Restore every pane's toolbars."))
        self._show_btns_btn.clicked.connect(
            lambda: self._set_chrome_hidden(False)
        )
        row.addWidget(self._show_btns_btn)
        self._sync_chrome_buttons()

        row.addSpacing(12)
        # Global DICOM-overlay controls (apply to every pane), so they live in
        # the shell's top row instead of stacked in each viewer's toolbar.
        # "DICOM Info" unifies the old tree "DICOM Info" + image "DICOM Tags":
        #   left-click  → show/hide the on-image DICOM text;
        #   right-click → choose which tags to overlay (active pane's modality).
        # The tag-size slider sits to its right.
        self._tags_btn = FitButton(t("DICOM Info"))
        self._tags_btn.setCheckable(True)
        self._tags_btn.setChecked(not self._overlay_hidden)
        self._tags_btn.setHelpToolTip(
            t("Left-click: show/hide DICOM info on the image\n"
              "Right-click: choose which tag data to show")
        )
        # Connect AFTER setChecked so the initial state doesn't fire the toggle.
        self._tags_btn.toggled.connect(self._set_overlay_shown)
        self._tags_btn.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._tags_btn.customContextMenuRequested.connect(
            lambda _p: self._open_tag_dialog_active()
        )
        row.addWidget(self._tags_btn)
        self._tagsz_lbl = QLabel(t("Tag size:"))
        self._tagsz_lbl.setMinimumWidth(0)   # may clip when the bar is squeezed
        row.addWidget(self._tagsz_lbl)
        self._tag_font_slider = QSlider(Qt.Orientation.Horizontal)
        self._tag_font_slider.setRange(TAG_FONT_PT_MIN, TAG_FONT_PT_MAX)
        self._tag_font_slider.setValue(int(self._tag_font_pt))
        # Prefer ~110 px but allow shrinking, so the top bar can get narrow
        # enough (e.g. at 1920×1080 / 150% scaling) to free the Studies dock.
        self._tag_font_slider.setMinimumWidth(40)
        self._tag_font_slider.setMaximumWidth(110)
        self._tag_font_slider.setToolTip(t("DICOM tag text size (all panes)"))
        self._tag_font_slider.valueChanged.connect(self._set_tag_font_pt)
        row.addWidget(self._tag_font_slider)

        # "Settings" — a popup gathering the app-wide display preferences:
        # display count (memory cap), angio image quality (S-Cine/S-Zoom/
        # Denoise) and the CT HU colour map.
        self._settings_btn = FitButton(t("Settings"))
        self._settings_btn.setToolTip(
            t("Display count, angio image quality, CT colour map"))
        # Deep-red background (darker than the CT active-pane frame red) with
        # white text, so this settings entry clearly stands out in the top bar.
        self._settings_btn.setStyleSheet(
            "QPushButton { background:#b3121b; color:#ffffff; font-weight:bold;"
            " border:1px solid #7d0d13; border-radius:6px; padding:3px 8px; }"
            "QPushButton:hover { background:#c8202a; }")
        self._settings_btn.clicked.connect(self._open_settings)
        row.addWidget(self._settings_btn)

        # Bi/Lt/Rt lives inside each viewer's own "Plane:" bar now (per-pane),
        # so there is no global plane switch in this top bar anymore.
        # (FitButton itself is now shrinkable — Preferred policy + an icon/first-
        # char minimumSizeHint — so this bar no longer forces a large minimum
        # width on the central area, which had blocked the Studies dock from
        # widening at small logical resolutions e.g. 1920×1080 at 150% scaling.)
        row.addStretch(1)
        return bar

    def _set_chrome_hidden(self, hidden: bool) -> None:
        """Apply "Max Image" (hide toolbars) to every pane, and remember it so
        new/swapped viewers and layout changes keep the same state."""
        self._chrome_hidden = hidden
        for pane in self._panes:
            pane.set_chrome_visible(not hidden)
        self._sync_chrome_buttons()

    def _sync_chrome_buttons(self) -> None:
        """Enable only the button that does something in the current state."""
        hidden = getattr(self, "_chrome_hidden", False)
        if hasattr(self, "_hide_btns_btn"):
            self._hide_btns_btn.setEnabled(not hidden)
            self._show_btns_btn.setEnabled(hidden)

    def _toggle_info(self, *_a) -> None:
        vis = not self._studies_dock.isVisible()
        self._studies_dock.setVisible(vis)
        self._info_btn.setText(
            t("◀ Hide Studies") if vis else t("Show Studies ▶")
        )

    #: Non-modal tool windows opened as their OWN owner-less taskbar window.
    #: Attribute names on self; closed together with the main window below.
    _TOOL_WINDOW_ATTRS = (
        "_coreg_win", "_multisync", "_ortho_win", "_rupture_win",
        "_dicomcheck_win", "_dicomfolder_win",
    )

    def _as_taskbar_window(self, win):
        """Mark a freshly-created non-modal tool window as an independent
        top-level window. It is built owner-less (no ``parent=self``) so Windows
        gives it its OWN taskbar button + hover thumbnail — an *owned* window is
        hidden from the taskbar, so only the main window would show. The process
        AppUserModelID (set in app.main) groups them all under one app button.
        Closing the main window also closes it (see closeEvent)."""
        icon = self.windowIcon()
        if not icon.isNull():
            win.setWindowIcon(icon)
        return win

    def closeEvent(self, e):  # noqa: N802 (Qt override)
        # Finish any background series loads first: a QThread destroyed while
        # still running aborts the process, so stop + wait each one (the decode
        # can't be interrupted, but a load in flight at quit is rare).
        for pane in list(self._loads):
            self._cleanup_load(pane)
        # Owner-less tool windows (CoSync etc.) don't close automatically with
        # us — close them explicitly so the app actually quits (else the last
        # open one keeps the process alive) and nothing is orphaned.
        for attr in self._TOOL_WINDOW_ATTRS:
            win = getattr(self, attr, None)
            if win is not None:
                try:
                    win.close()
                except RuntimeError:
                    pass          # already destroyed by Qt — fine
        # Finalize each VTK CT render window while its native window is still
        # valid, so VTK releases its GL context cleanly instead of flooding the
        # terminal with "wglMakeCurrent failed ... invalid handle (code 6)" when
        # Qt destroys the HWNDs first during teardown. Duck-typed: only the VTK
        # CT viewer defines finalize_gl (pygfx / QPainter viewers have none).
        for pane in self._panes:
            for viewer in pane.all_viewers():
                fin = getattr(viewer, "finalize_gl", None)
                if callable(fin):
                    fin()
        super().closeEvent(e)

    def _shown_panes(self) -> list:
        """Panes currently on screen, in display (reading) order. 1×1 shows the
        active pane; every other layout shows the selected master-grid rectangle
        (``self._layout_cells``), so each pane keeps its on-screen position no
        matter which layout you switch from."""
        if self._layout_key == "1x1":
            a = self._active if self._active is not None else self._order[0]
            return [a]
        return [self._order[i] for i in self._layout_cells]

    def _current_layout_rect(self):
        """The applied layout as a 0-based inclusive master-grid rectangle. For
        1×1 it is the active pane's cell; otherwise the bounding rect of the
        shown cells."""
        if self._layout_key == "1x1":
            if self._active in self._order:
                r, c = divmod(self._order.index(self._active), _MAX_GRID_COLS)
            else:
                r = c = 0
            return (r, c, r, c)
        rc = [divmod(i, _MAX_GRID_COLS) for i in self._layout_cells]
        rows = [x[0] for x in rc]
        cols = [x[1] for x in rc]
        return (min(rows), min(cols), max(rows), max(cols))

    def _refresh_layout_picker(self) -> None:
        """Before the picker pops up: highlight the current layout rectangle and
        shade each cell by whether its pane holds data."""
        self._layout_picker.set_occupancy(self._pane_occupancy())
        self._layout_picker.set_current_rect(*self._current_layout_rect())

    def _layout_btn_text(self) -> str:
        # setMenu() adds the native dropdown arrow, so the text itself stays
        # plain: "Layout  2×3".
        return f"{t('Layout')}  {self._layout_key.replace('x', '×')}"

    def _on_layout_picked(self, r0: int, c0: int, r1: int, c1: int) -> None:
        """A rectangle was dragged in the visual grid picker → show exactly that
        (possibly non-top-left) block of master cells. A single cell → 1×1 of
        that pane."""
        self._layout_menu.hide()
        rows, cols = r1 - r0 + 1, c1 - c0 + 1
        if rows == 1 and cols == 1:
            cell = r0 * _MAX_GRID_COLS + c0
            if 0 <= cell < len(self._order):
                self._set_active_pane(self._order[cell])
            self._apply_layout("1x1")
            return
        cells = [rr * _MAX_GRID_COLS + cc
                 for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)]
        self._apply_layout(f"{rows}x{cols}", cells)

    def _maximize_pane(self, pane: ViewerPane) -> None:
        """1×1 button / double-click on a pane → show only that pane (1×1).
        Make it the active pane first so _apply_layout('1x1') shows it."""
        self._set_active_pane(pane)
        self._apply_layout("1x1")

    def _pane_occupancy(self) -> set:
        """0-based (row, col) master-grid cells whose backing pane holds data —
        fed to the layout picker so loaded panes shade bright, empty ones dark.
        Cell (r, c) maps to the same canonical pane _apply_layout places there
        (self._order[r * _MAX_GRID_COLS + c])."""
        occ = set()
        for idx, pane in enumerate(self._order):
            if pane.has_data():
                occ.add(divmod(idx, _MAX_GRID_COLS))
        return occ

    def _apply_layout(self, key: str, cells=None) -> None:
        self._layout_key = key
        rows, cols, count = _LAYOUTS[key]
        # Which master cells to show: the passed rectangle, or the top-left
        # R×C block (default / backward-compatible).
        self._layout_cells = list(cells) if cells is not None \
            else list(_LAYOUT_CELLS[key])
        if hasattr(self, "_layout_btn"):
            self._layout_btn.setText(self._layout_btn_text())

        # Detach every pane, then re-add the visible subset to the grid.
        for pane in self._panes:
            self._grid.removeWidget(pane)
            pane.setVisible(False)
        full = key == "1x1"
        self._grid.setSpacing(0 if full else 2)
        if full:
            # 1×1 shows the active pane WITHOUT disturbing self._order, so
            # returning to a grid restores the canonical arrangement.
            shown = self._active if self._active is not None else self._order[0]
            self._grid.addWidget(shown, 0, 0)
            shown.setVisible(True)
            shown.set_full_bleed(True)
        else:
            # Place the selected master cells (the chosen rectangle) into a
            # rows×cols grid, preserving their reading-order positions.
            for i, cell_idx in enumerate(self._layout_cells):
                r, c = divmod(i, cols)
                pane = self._order[cell_idx]
                self._grid.addWidget(pane, r, c)
                pane.setVisible(True)
                pane.set_full_bleed(False)
        # Reset stretch on ALL grid lines (up to the largest layout, 2×3): a
        # leftover stretch on a now-empty row/col from a bigger layout would
        # otherwise still reserve space, shrinking the panes on the way back.
        for r in range(_GRID_MAX_ROWS):
            self._grid.setRowStretch(r, 1 if r < rows else 0)
        for c in range(_GRID_MAX_COLS):
            self._grid.setColumnStretch(c, 1 if c < cols else 0)

        if full:
            # 1×1: the shown pane is the only one visible — keep it active.
            shown = self._active if self._active is not None else self._order[0]
            self._set_active_pane(shown)
        else:
            shown = self._shown_panes()
            if self._active not in shown:
                self._set_active_pane(shown[0])
            else:
                self._set_active_pane(self._active)
        # Compact transport (half-height) on every multi-row layout so the
        # Play/seek strip doesn't eat the smaller panes; re-apply the shared
        # Max-Image state too (panes were detached/re-shown above).
        compact = rows >= 2
        for pane in self._panes:
            pane.set_compact(compact)
            pane.set_chrome_visible(not self._chrome_hidden)
        # IVUS long-axis is only usable (and only safe) in 1x1 — in a small
        # multi-pane cell the strip is uselessly tiny and its rebuild is the
        # heaviest op / main freeze source. Disable it everywhere but 1x1
        # (force-hides any open strip).
        self._sync_long_axis_gate()
        # "Pane →" flips the fullscreen 1×1 view to the next loaded pane, so it
        # only makes sense in 1×1 — grey it out in every multi-pane layout.
        if hasattr(self, "_pane_step_btn"):
            self._pane_step_btn.setEnabled(self._layout_key == "1x1")
        # Bi/Lt/Rt is per-pane (each viewer's own "Plane:" bar), so a grid
        # change never overrides any pane's plane choice — a pane left on
        # "Bi" keeps showing both planes here too (just smaller).
        self._sync_layout_gate()      # MultiSync menu = only in multi-pane
        # A layout change re-shows each pane → every visible CT viewer re-runs
        # its oblique-MPR reslice on the UI thread via showEvent (deferred one
        # event-loop turn). With 2 live CT that is up to 4 reslices of ~0.7 GB
        # volumes back-to-back, so the window looks frozen for a beat. Show a
        # busy cursor that SPANS those deferred reslices — the restore is queued
        # AFTER each showEvent's singleShot refresh (FIFO), so it clears once
        # they finish — so the pause reads as "working", not hung.
        if any(_is_ct(p.current_viewer()) for p in self._shown_panes()):
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QTimer.singleShot(0, QApplication.restoreOverrideCursor)

    def _sync_long_axis_gate(self) -> None:
        """Allow the IVUS long-axis only in the 1x1 layout; disable it in every
        multi-pane layout on every IVUS viewer in every pane."""
        allowed = self._layout_key == "1x1"
        for pane in self._panes:
            for v in pane.all_viewers():
                if hasattr(v, "set_long_axis_allowed"):
                    v.set_long_axis_allowed(allowed)

    def _swap_panes(self, src_index: int, dest_index: int) -> None:
        """Swap two panes' grid slots (drag a pane title onto another)."""
        if src_index == dest_index:
            return
        src = self._panes[src_index]
        dest = self._panes[dest_index]
        i, j = self._order.index(src), self._order.index(dest)
        self._order[i], self._order[j] = self._order[j], self._order[i]
        self._apply_layout(self._layout_key, self._layout_cells)

    def _cycle_active_pane(self) -> None:
        """"Pane →" button: advance to the next pane, wrapping the last back to
        the first (endless loop).

        In a multi-pane grid it moves the active-pane highlight through the
        on-screen cells in reading order (left→right, then the next row's left),
        cycling over _shown_panes() (already in reading order).

        In the single-pane 1×1 layout there is only one cell, so instead it
        flips the FULLSCREEN view to the next pane that holds an image (a
        "next image" flipper) — otherwise the button would do nothing there."""
        if self._layout_key == "1x1":
            loaded = [p for p in self._order if p.has_data()]
            if len(loaded) <= 1:
                return
            try:
                i = loaded.index(self._active)
            except ValueError:
                i = -1
            self._set_active_pane(loaded[(i + 1) % len(loaded)])
            self._apply_layout("1x1")     # re-render so the new pane fills 1×1
            return
        shown = self._shown_panes()
        if len(shown) <= 1:
            return
        try:
            i = shown.index(self._active)
        except ValueError:
            i = -1
        self._set_active_pane(shown[(i + 1) % len(shown)])

    def _set_active_pane(self, pane: ViewerPane) -> None:
        self._active = pane
        self._touch_pane(pane)
        for p in self._panes:
            p.set_active(p is pane and p.isVisible())
        self._sync_xa_shortcuts()
        self._follow_active_pane()

    # ------------------------------------------------- memory cap (LRU sleep)
    def _load_live_caps(self) -> dict:
        """Per-modality 'live at once' caps as {Modality: n}, from user settings.
        Beyond a cap the least-recently-used pane of that modality is frozen to
        a still and its memory freed — so opening many large CTs / angios can't
        exhaust RAM (the cause of the hard crash when 3 CT + 11 XA were open)."""
        caps = settings.load_live_caps()
        # Historically CT was HARD-CAPPED to 1 live here: a 2nd/3rd live CT pane
        # can exhaust the GL context/GPU (earlier CT panes go BLACK) or the RAM
        # (force-quit). LVEF needs two phases (ED/ES) side by side, so the cap is
        # now the user's own "Display count → CT panes" setting (default 1),
        # gated behind an explicit warning in the Settings dialog when they raise
        # it to ≥2. Over-cap panes still fall back to a memory-light still, and
        # _free_live_for_incoming + the memory budget still protect against huge
        # volumes piling up. XA (plain-Qt canvas) is unchanged.
        return {Modality.CT: caps["CT"], Modality.XA: caps["XA"]}

    def _touch_pane(self, pane: ViewerPane) -> None:
        """Stamp *pane* as most-recently-used on the LRU clock."""
        self._use_seq += 1
        self._pane_touch[pane] = self._use_seq

    def _pane_live_modality(self, pane: ViewerPane):
        """The modality whose data *pane* currently holds live (not a still),
        or None if the pane is empty / frozen / still loading."""
        if pane.is_still() or pane.current_viewer() is None:
            return None
        s = getattr(pane, "_series_ref", None)
        return s.modality if s is not None else None

    def _pane_lv_busy(self, p) -> bool:
        """True if the pane's live viewer is in an active LV analysis session.
        Such a pane must NOT be auto-demoted to a still: the LV state (mode,
        long axis, traced borders, computed volume) isn't captured by the
        snapshot, so demoting garbles the frozen image and loses the whole
        session (reported when a 3rd CT pushed an LV pane over the live cap)."""
        v = getattr(p, "_cur_viewer", None)
        try:
            return bool(v is not None and hasattr(v, "lv_active")
                        and v.lv_active())
        except Exception:
            return False

    def _enforce_live_caps(self, keep: ViewerPane) -> None:
        """Demote least-recently-used live panes to frozen stills until each
        modality is within its _LIVE_CAP. *keep* (the just-loaded/active pane)
        is never demoted, nor is a pane mid-LV-analysis. A pane with a load in
        flight is left alone."""
        for mod, cap in self._live_cap.items():
            live = [p for p in self._panes
                    if self._pane_live_modality(p) == mod
                    and p not in self._loads]
            if len(live) <= cap:
                continue
            # Oldest first; never touch the pane we want to keep. Iterate a copy
            # so removing from `live` mid-loop doesn't skip the next candidate.
            live.sort(key=lambda p: self._pane_touch.get(p, 0))
            for p in list(live):
                if len(live) <= cap:
                    break
                if p is keep or self._pane_lv_busy(p):
                    continue                      # protect the active LV session
                if p.demote_to_still(getattr(p, "_series_ref", None)):
                    live.remove(p)

    def _free_live_for_incoming(self, incoming_pane: ViewerPane,
                                modality, incoming_bytes: int = 0) -> None:
        """Demote the least-recently-used OTHER live panes of *modality* to
        stills — freeing their volume + GPU memory — BEFORE the incoming pane
        builds its own (possibly large) volume.

        Without this, loading a 2nd big CT builds its ~0.7 GB VTK volume while
        the outgoing CT is still live, so the peak holds BOTH and the GPU/RAM
        exhausts → the new pane renders black and the only recovery was an app
        restart (which loses the whole session). Freeing first means the build
        has the memory it needs.

        Two limits are enforced (oldest demoted first):
          * the COUNT cap — keep at most cap-1 others (the incoming makes cap);
          * a raw-volume MEMORY budget — so a high count cap (e.g. CT=4) can't
            let several huge volumes pile up: small series still use the whole
            cap, but big ones auto-free the oldest until the live total
            (incl. the incoming one) fits the budget.
        The post-build _enforce_live_caps still runs for the general case."""
        others = [p for p in self._panes
                  if p is not incoming_pane
                  and self._pane_live_modality(p) == modality
                  and p not in self._loads
                  and not self._pane_lv_busy(p)]    # never demote an LV session
        others.sort(key=lambda p: self._pane_touch.get(p, 0))   # oldest first

        def _demote_oldest() -> bool:
            if not others:
                return False
            p = others.pop(0)
            p.demote_to_still(getattr(p, "_series_ref", None))
            return True

        cap = self._live_cap.get(modality)
        if cap:                                  # count cap: leave room for one
            while len(others) > cap - 1 and _demote_oldest():
                pass
        budget = self._live_bytes_budget.get(modality)
        if budget:                               # memory budget (raw volume)
            def _live_total() -> int:
                return (int(incoming_bytes)
                        + sum(int(getattr(p, "_volume_bytes", 0))
                              for p in others))
            while others and _live_total() > budget and _demote_oldest():
                pass

    def _promote_pane(self, pane: ViewerPane) -> None:
        """A frozen still pane was interacted with → reload its series (its
        view rebuilds; the LRU cap then frees whatever is now oldest). The
        still state is cleared by show_series only once the reload SUCCEEDS, so
        a failed load leaves the frozen image in place and the user can retry."""
        if not pane.is_still() or pane in self._loads:
            return
        series = pane._still_series
        if series is None:
            return
        self._set_active_pane(pane)
        self._open_series(series, pane)

    def _follow_active_pane(self) -> None:
        """Make the browser reflect whatever the ACTIVE pane shows: highlight
        its series in the tree + thumbnail grid, or blank the browser when the
        pane is empty. Silent (no series_chosen) so the pane is never reloaded.

        Called whenever the target pane changes OR its contents change (load /
        ✕ clear), so the Studies dock always mirrors the targeted pane."""
        uid = self._active.shown_series_uid()
        se = self._series_by_uid.get(uid) if uid else None
        if se is not None:
            self.browser.sync_to_series(se)
        else:
            # Empty pane targeted → blank the browser (don't leave the
            # previously-targeted pane's thumbnail showing).
            self.browser.show_empty()

    def _on_pane_cleared(self, pane: ViewerPane) -> None:
        """A pane's ✕ emptied it (layout kept). Re-sync the features that
        depend on pane contents: the cine shortcuts and the MultiSync gate."""
        self._sync_xa_shortcuts()
        self._sync_layout_gate()
        # ✕ emits `activated` (so the cleared pane becomes active) BEFORE it
        # resets, so the browser was last followed while the old series still
        # showed. Now that the active pane is empty, blank the browser.
        if pane is self._active:
            self._follow_active_pane()

    # --------------------------------------------------- series navigation
    def _build_shortcuts(self) -> None:
        # Cine (XA/IVUS) keys. They are app-wide QShortcuts, so they
        # would otherwise swallow S / T / R / D / W / V before a focused
        # CT pane sees them — keep them enabled only while the active
        # pane is showing a cine viewer (see _sync_xa_shortcuts).
        # Series-navigation keys work on BOTH cine and CT panes (gated
        # separately in _sync_xa_shortcuts; _nav_active routes to the
        # active pane's kind). F and A are free on CT (its tool keys are
        # Z/V/R/S/G/T/W + C).
        self._nav_shortcuts = []
        for key, fn in (
            ("F", lambda: self._nav_active("next")),
            ("A", lambda: self._nav_active("prev")),
            ("Home", lambda: self._nav_active("first")),
            ("End", lambda: self._nav_active("last")),
        ):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(fn)
            self._nav_shortcuts.append(sc)
        self._xa_shortcuts = []
        for key, fn in (
            # Cine transport layout (user spec):
            #   T, R = step +1 frame   (R duplicates T for two-handed use)
            #   E    = step -1 frame
            #   D    = play / 2× toggle (cycles 1×→2×→1× on the second
            #          press, starts at 1× when stopped)
            #   S    = stop
            #   W    = Window/Level (open the W/L popup)
            #   V    = ECG show/hide on XA · IVUS long-axis show/hide on IVUS
            #          (context-split by the active pane's modality — the two
            #          never collide, and long-axis exists only on IVUS)
            ("T", lambda: self._xa_step(+1)),
            ("R", lambda: self._xa_step(+1)),
            ("E", lambda: self._xa_step(-1)),
            ("D", lambda: self._xa_play_speed_toggle()),
            ("S", lambda: self._xa_stop()),
            ("W", lambda: self._xa_wl()),
            ("V", lambda: self._xa_v_key()),
            ("Z", lambda: self._xa_zoom(True)),
            ("Shift+Z", lambda: self._xa_zoom(False)),
        ):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(fn)
            self._xa_shortcuts.append(sc)
        self._sync_xa_shortcuts()

    def _sync_xa_shortcuts(self) -> None:
        v = self._active.current_viewer()
        on = _is_cine(v)
        for sc in getattr(self, "_xa_shortcuts", []):
            sc.setEnabled(on)
        # F/A/Home/End also navigate CT panes (CT never sees these letters
        # for tools, so enabling them app-wide is safe there too).
        for sc in getattr(self, "_nav_shortcuts", []):
            sc.setEnabled(on or _is_ct(v))
        # Orthogonal-View depends on the active viewer being XA — refresh
        # its gate whenever the active pane / viewer changes.
        self._sync_layout_gate()

    def _xa_pane(self) -> ViewerPane | None:
        """Pane currently showing a cine (XA or IVUS) — the active one if
        it is, else the first that is."""
        if _is_cine(self._active.current_viewer()):
            return self._active
        for p in self._panes:
            if p.isVisible() and _is_cine(p.current_viewer()):
                return p
        return None

    def _ct_pane(self) -> ViewerPane | None:
        """Pane currently showing a CT — the active one if it is, else the
        first visible one that is (mirrors _xa_pane)."""
        if _is_ct(self._active.current_viewer()):
            return self._active
        for p in self._panes:
            if p.isVisible() and _is_ct(p.current_viewer()):
                return p
        return None

    def _xa(self) -> XAViewer | None:
        p = self._xa_pane()
        v = p.current_viewer() if p else None
        return v if _is_cine(v) else None

    def _xa_play(self, on: bool) -> None:
        v = self._xa()
        if v is not None:
            (v.play if on else v.stop)()

    def _xa_play_toggle(self) -> None:
        v = self._xa()
        if v is None:
            return
        # The play_btn is checkable; flipping it triggers _toggle_play.
        v.play_btn.setChecked(not v.play_btn.isChecked())

    def _xa_step(self, delta: int) -> None:
        """T / R = step one frame forward / back. Also stops playback so
        the user can scrub past the end of a cine without it wrapping."""
        v = self._xa()
        if v is None:
            return
        v.step_frame(int(delta))

    def _xa_play_speed_toggle(self) -> None:
        """D = play / 2× toggle. Stopped → start at 1×. Playing 1× →
        switch to 2×. Playing 2× → switch back to 1× (stays playing).
        Use S to stop."""
        v = self._xa()
        if v is None:
            return
        v.toggle_play_speed()

    def _xa_stop(self) -> None:
        """S = stop. (Pure stop — restart with D.)"""
        v = self._xa()
        if v is None:
            return
        v.stop()
        v.play_btn.setChecked(False)

    def _xa_wl(self) -> None:
        """W = open the Window/Level popup for the active cine pane (same
        popup as the image right-click ▸ Change W/L)."""
        v = self._xa()
        if v is not None and hasattr(v, "show_wl_dialog"):
            v.show_wl_dialog()

    def _xa_v_key(self) -> None:
        """V = ECG strip on an XA pane, IVUS long-axis on an IVUS pane.

        The two features never overlap — long-axis exists only in the IVUS
        viewer and ECG is an XA feature — so one key can serve both, split by
        the active cine pane's modality (handles_modality). This keeps V a
        left-hand key for both and frees us from a right-hand key."""
        v = self._xa()
        if v is None:
            return
        if getattr(v, "handles_modality", "") == "IVUS":
            self._ivus_toggle_long_axis()
        else:
            self._xa_toggle_ecg()

    def _xa_toggle_ecg(self) -> None:
        """V (on an XA pane) = toggle the ECG waveform strip. The strip is built
        per series from the DICOM WaveformSequence when present; on a
        series without ECG the button just stays hidden with a status
        message."""
        v = self._xa()
        if v is None:
            return
        v.toggle_ecg()

    def _ivus_toggle_long_axis(self) -> None:
        """V (on an IVUS pane) = toggle the IVUS long-axis (longitudinal)
        view. No-op on XA — the long-axis only makes sense for an IVUS
        pull-back (V does ECG there instead, via _xa_v_key)."""
        v = self._xa()
        if v is None or getattr(v, "handles_modality", "") != "IVUS":
            self.statusBar().showMessage(
                t("Long-axis view is IVUS-only.")
            )
            return
        if not getattr(v, "_la_allowed", True):
            self.statusBar().showMessage(
                t("Long-axis view is available only in single-pane (1x1) layout.")
            )
            return
        v.toggle_long_axis()

    def _xa_zoom(self, zoom_in: bool) -> None:
        v = self._xa()
        if v is None:
            return
        (v.zoom_in if zoom_in else v.zoom_out)()

    def _nav_active(self, where: str) -> None:
        """F/A/Home/End: navigate by the ACTIVE pane's kind — a CT pane
        steps through the study's CT series, otherwise the cine list.

        Exception: while a CT pane is LV-tracing, A/F (prev/next) step the
        long-axis plane instead of the series list — the viewer's lv_nav_key
        claims the key and we stop here."""
        v = self._active.current_viewer()
        if _is_ct(v):
            if hasattr(v, "lv_nav_key") and v.lv_nav_key(where):
                return
            self._nav_ct(where)
        else:
            self._nav_xa(where)

    def _nav_xa(self, where: str) -> None:
        self._nav_pane_series(self._xa_pane(), where)

    def _nav_ct(self, where: str) -> None:
        self._nav_pane_series(self._ct_pane(), where)

    def _nav_pane_series(self, xp, where: str) -> None:
        # Step within the SAME modality as the pane (XA among XA, IVUS
        # among IVUS, CT among CT, NM among NM, …), inside that pane.
        # The navigation key comes from the SERIES currently shown — not
        # the viewer's handles_modality, because non-canonical modalities
        # (NM/OCT/OFD) all fall back to the XAViewer with
        # handles_modality="XA" and would otherwise step through XA series
        # instead of their own kind.
        if xp is None:
            return
        cur_series = self._series_by_uid.get(xp.shown_series_uid())
        if cur_series is not None:
            mod = cur_series.kind
        else:
            mod = getattr(xp.current_viewer(), "handles_modality", "")
        lst = self.browser.ordered_series(mod)
        if not lst:
            return
        # Keep navigation within the CURRENT study so First/Prev/Next/Last
        # never crosses into another patient's or another date's study —
        # the user has to click that study's row in the tree to switch.
        cur_study = self._study_by_series_uid.get(
            cur_series.series_uid if cur_series is not None else ""
        )
        if cur_study:
            scoped = [
                s for s in lst
                if self._study_by_series_uid.get(s.series_uid) == cur_study
            ]
            if scoped:
                lst = scoped
        # Navigate relative to the series ACTUALLY shown in the (active) cine
        # pane. Using the app-wide "last opened of this modality" instead made
        # First/Prev/Next/Last jump to an unrelated image in multi-pane layouts
        # (another pane's series could be the more-recent open). Fall back to
        # the last-opened only when this pane has no series of the kind.
        cur = (cur_series or self._last_by_modality.get(mod))
        try:
            idx = lst.index(cur)
        except ValueError:
            idx = -1
        # Skip hidden series: First/Last land on the nearest VISIBLE series;
        # Prev/Next step to the nearest visible one in that direction.
        visible = [s for s in lst if not getattr(s, "hidden", False)]
        if not visible:
            return                                   # nothing visible to show
        if where == "first":
            tgt = visible[0]
        elif where == "last":
            tgt = visible[-1]
        elif where == "next":
            tgt = visible[-1]
            for i in range(idx + 1, len(lst)):
                if not getattr(lst[i], "hidden", False):
                    tgt = lst[i]
                    break
        else:  # prev
            tgt = visible[0]
            for i in range(idx - 1, -1, -1):
                if not getattr(lst[i], "hidden", False):
                    tgt = lst[i]
                    break
        self._set_active_pane(xp)
        self.browser.select_series(tgt)

    def _cine_series_scope(self, pane) -> tuple:
        """(visible_series_list, current_series) for the same-modality,
        same-study cine list that First/Prev/Next/Last steps through in
        *pane* — the basis for the series-position counter. Mirrors the
        scoping used by _nav_xa."""
        cur_series = self._series_by_uid.get(pane.shown_series_uid())
        if cur_series is not None:
            mod = cur_series.kind
        else:
            mod = getattr(pane.current_viewer(), "handles_modality", "")
        lst = self.browser.ordered_series(mod)
        cur_study = self._study_by_series_uid.get(
            cur_series.series_uid if cur_series is not None else ""
        )
        if cur_study:
            scoped = [
                s for s in lst
                if self._study_by_series_uid.get(s.series_uid) == cur_study
            ]
            if scoped:
                lst = scoped
        visible = [s for s in lst if not getattr(s, "hidden", False)]
        return visible, cur_series

    def _update_cine_series_pos(self, pane) -> None:
        """Push '<pos>/<count>' of the series shown in *pane* (within its
        study's same-modality cine list) into the viewer's series counter.
        No-op for non-cine viewers (which lack set_series_position)."""
        v = pane.current_viewer()
        if v is None or not hasattr(v, "set_series_position"):
            return
        visible, cur = self._cine_series_scope(pane)
        try:
            idx = visible.index(cur)
        except ValueError:
            idx = -1
        v.set_series_position(idx, len(visible))

    # ------------------------------------------------------------- data load
    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, t("Open DICOM folder"))
        if folder:
            self._load_folder(folder)

    def _choose_files(self) -> None:
        # "All files" is the default filter because DICOM files very often have
        # no extension (e.g. "STUDY.1"); the .dcm convenience filter is second.
        paths, _ = QFileDialog.getOpenFileNames(
            self, t("Open DICOM file(s)"), getattr(self, "_last_dir", "") or "",
            "All files (*);;DICOM files (*.dcm *.DCM *.ima *.IMA *.dicom)",
        )
        paths = [p for p in paths if p]
        if paths:
            self._last_dir = os.path.dirname(paths[0])
            self._load_files(paths)

    # ---------------------------------------------------- drag & drop a folder
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        # A FOLDER drop loads the whole folder; a FILE drop loads ONLY the
        # dropped file(s) — not the rest of their containing folder. Dropping
        # SEVERAL folders imports all of them.
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        paths = [p for p in paths if p]
        if not paths:
            return
        event.acceptProposedAction()
        self._load_paths(paths)

    def _load_folder(self, folder: str, open_in_pane=None) -> None:
        self._last_dir = folder              # seeds the DicomCheck/Folder tools
        self._load_index(
            t("Scanning {folder} …", folder=folder),
            lambda prog: dicom_io.scan_folder(folder, prog),
            open_in_pane=open_in_pane,
        )

    def _load_files(self, paths: list[str], open_in_pane=None) -> None:
        n = len(paths)
        self._load_index(
            t("Loading {n} file(s) …", n=n),
            lambda prog: dicom_io.index_files(paths, prog),
            open_in_pane=open_in_pane,
        )

    def _load_paths(self, paths: list[str], open_in_pane=None) -> None:
        """Import a drop of ANY mix of folders and files — several folders at
        once included. Folders expand recursively; plain files do not."""
        paths = [p for p in (paths or []) if p]
        dirs = [p for p in paths if os.path.isdir(p)]
        files = [p for p in paths if os.path.isfile(p)]
        if not dirs and not files:
            return
        if dirs:
            # Seeds the DicomCheck/Folder tools; the first dropped folder is
            # the best guess at "where the user is working".
            self._last_dir = dirs[0]
        if dirs and not files:
            msg = (t("Scanning {folder} …", folder=dirs[0]) if len(dirs) == 1
                   else t("Scanning {n} folders …", n=len(dirs)))
        elif files and not dirs:
            msg = t("Loading {n} file(s) …", n=len(files))
        else:
            msg = t("Scanning {n} folders …", n=len(dirs))
        self._load_index(
            msg,
            lambda prog: dicom_io.index_paths(dirs + files, prog),
            open_in_pane=open_in_pane,
            spread_roots=dirs if len(dirs) > 1 else None,
        )

    def _on_paths_dropped(self, paths) -> None:
        """Folder/file dropped ONTO the tree → import into the tree only; no
        pane is opened (drag a series onto a pane to display it)."""
        self._load_paths(paths)

    def _load_index(self, status_msg: str, scan_fn, open_in_pane=None,
                    spread_roots=None) -> None:
        """Run *scan_fn(progress)* under a modal progress dialog, then merge
        the resulting patients into the tree. *open_in_pane* opens the freshly
        imported series into that pane (folder/file dropped ON a pane); when
        None the import just populates the tree — nothing is loaded into a
        pane (File▸Open, window/tree drops). Shared by folder drops
        (scan_folder) and file drops (index_files)."""
        self.statusBar().showMessage(status_msg)
        # Bring the app to the front (drop often comes from Explorer).
        if self.isMinimized():
            self.showNormal()
        self.raise_()
        self.activateWindow()

        # NON-modal + off the GUI thread: a big CT study's per-file metadata
        # scan used to run here on the GUI thread behind an app-modal dialog,
        # freezing the whole window (you couldn't even open the folder picker).
        # Now a worker does the scan while the shell stays live; the post-scan
        # merge/auto-open runs in _on_scan_done on the GUI thread.
        dlg = QProgressDialog(t("Loading DICOM…"), None, 0, 0, self)
        dlg.setWindowTitle(t("Scanning"))
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        dlg.setMinimumDuration(300)   # don't flash for tiny folders
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)

        thread = QThread(self)
        worker = _FolderScanWorker(scan_fn)
        worker.moveToThread(thread)
        self._scans[worker] = (thread, dlg, open_in_pane, spread_roots)
        worker.progress.connect(self._on_scan_progress)
        worker.done.connect(self._on_scan_done)
        worker.failed.connect(self._on_scan_failed)
        thread.started.connect(worker.run)
        thread.start()

    def _scan_entry_for_sender(self):
        """(worker, entry) for the scan worker that emitted the running signal,
        or (None, None) if already cleaned up."""
        worker = self.sender()
        entry = self._scans.get(worker)
        return (worker, entry) if entry is not None else (None, None)

    def _cleanup_scan(self, worker) -> None:
        entry = self._scans.pop(worker, None)
        if entry is None:
            return
        thread, dlg = entry[0], entry[1]
        dlg.reset()
        dlg.close()
        thread.quit()
        thread.wait()
        worker.deleteLater()
        thread.deleteLater()

    def _on_scan_progress(self, done: int, total: int) -> None:
        _worker, entry = self._scan_entry_for_sender()
        if entry is None:
            return
        dlg = entry[1]
        if total and dlg.maximum() != total:
            dlg.setMaximum(total)
        dlg.setValue(done)

    def _on_scan_failed(self, msg: str) -> None:
        worker, entry = self._scan_entry_for_sender()
        if entry is None:
            return
        self._cleanup_scan(worker)
        QMessageBox.critical(self, t("Scan failed"), msg)

    def _on_scan_done(self, new_patients) -> None:
        worker, entry = self._scan_entry_for_sender()
        if entry is None:
            return
        open_in_pane, spread_roots = entry[2], entry[3]
        self._cleanup_scan(worker)
        # Series UIDs contributed by THIS folder — used to auto-open the
        # newly dropped study (not whatever sorts first overall).
        new_uids = {
            se.series_uid
            for p in new_patients.values()
            for st in p.studies.values()
            for se in st.series.values()
        }
        # Nothing displayable was dropped (empty folder, or no DICOM files):
        # say so plainly and stop — don't leave a spinner up or re-open some
        # unrelated already-loaded series.
        if not new_uids:
            self.statusBar().showMessage(
                t("No displayable DICOM files were found."))
            QMessageBox.information(
                self, t("No DICOM files"),
                t("The dropped folder/file(s) contained no displayable DICOM."),
            )
            return
        # Accumulate: a new folder adds its studies; previously loaded
        # ones stay in the info panel.
        dicom_io.merge_patients(self._patients, new_patients)

        self.browser.populate(self._patients)
        n_ser = self._reindex_series_maps()
        n_pat = len(self._patients)
        self.statusBar().showMessage(
            t("{n_pat} patient(s), {n_ser} series (total). "
              "⚹ marks patients with both CT and XA. "
              "Drag & drop a series onto a pane to display.",
              n_pat=n_pat, n_ser=n_ser)
        )
        # Auto-open a series from the just-loaded folder into the active pane.
        # When the folder has CT, prefer the "main" CT series (see
        # _initial_ct_target); otherwise fall back to the first series in the
        # browser's display order.
        # Only open into a pane when the drop landed ON a pane; otherwise the
        # import just populates the tree (drag a series onto a pane to show).
        if open_in_pane is None:
            return
        ordered = self.browser.ordered_series()
        new_series = [se for se in ordered if se.series_uid in new_uids]
        # SEVERAL folders dropped on a pane → one folder per pane, laid out
        # from the dropped pane on (the drop itself says "show these side by
        # side"). A single folder keeps the classic one-pane behaviour.
        if spread_roots and len(spread_roots) > 1:
            self._spread_folders(spread_roots, new_series, open_in_pane)
            return
        target = self._initial_ct_target(new_series)
        if target is None:
            target = (self._initial_noncT_target(new_series)
                      or (ordered[0] if ordered else None))
        if target is not None:
            self._open_series(target, open_in_pane)

    # ------------------------------------------- multi-folder drop onto a pane
    @staticmethod
    def _series_root(series, roots_norm: list[str]) -> "int | None":
        """Index of the dropped root *series* came from — the LONGEST matching
        path prefix, so dropping a parent AND its child assigns each series to
        the most specific one. None when it belongs to no dropped root."""
        files = getattr(series, "files", None) or []
        if not files:
            return None
        f = os.path.normcase(os.path.abspath(files[0]))
        best = None
        for i, root in enumerate(roots_norm):
            if f == root or f.startswith(root + os.sep):
                if best is None or len(root) > len(roots_norm[best]):
                    best = i
        return best

    def _fit_panes(self, n: int, drop_pane) -> list:
        """*n* panes to fill, starting at *drop_pane* in reading order, growing
        the layout when the current one is too small.

        Preference order:
          1. the layout as-is, when it already has n panes at/after the drop;
          2. the smallest master block that CONTAINS the dropped pane and still
             has n cells at/after it (squarer beats longer: 4 → 2×2, not 4×1);
          3. the smallest block that fits n at all, filled from the top-left —
             the fallback when the drop landed too late in the grid (e.g. the
             bottom-right pane) to fit n folders after it.
        """
        shown = self._shown_panes()
        try:
            start = shown.index(drop_pane)
        except ValueError:
            start = 0
        if len(shown) - start >= n:
            return shown[start:start + n]

        # Squarest-then-widest layout that fits, smallest first.
        keys = sorted(_LAYOUTS, key=lambda k: (_LAYOUTS[k][2],
                                               abs(_LAYOUTS[k][0] - _LAYOUTS[k][1]),
                                               _LAYOUTS[k][0]))
        drop_cell = (self._order.index(drop_pane)
                     if drop_pane in self._order else 0)
        for key in keys:
            cells = _LAYOUT_CELLS[key]
            if drop_cell not in cells:
                continue
            i = cells.index(drop_cell)
            if len(cells) - i < n:
                continue
            self._apply_layout(key, list(cells))
            return [self._order[c] for c in cells[i:i + n]]
        for key in keys:
            cells = _LAYOUT_CELLS[key]
            if len(cells) >= n:
                self._apply_layout(key, list(cells))
                return [self._order[c] for c in cells[:n]]
        return shown[start:start + n]

    def _spread_folders(self, roots: list[str], new_series: list,
                        drop_pane) -> None:
        """Open ONE series per dropped folder, into consecutive panes from
        *drop_pane*. Folders keep their drop order; every folder is in the tree
        regardless, so any that don't fit on screen are still one drag away."""
        roots_norm = [os.path.normcase(os.path.abspath(r)) for r in roots]
        by_root: dict[int, list] = {}
        for se in new_series:                      # already in display order
            i = self._series_root(se, roots_norm)
            if i is not None:
                by_root.setdefault(i, []).append(se)

        targets = []
        for i in range(len(roots)):
            cand = by_root.get(i) or []
            tgt = (self._initial_ct_target(cand)
                   or self._initial_noncT_target(cand))
            if tgt is not None:
                targets.append(tgt)
        if not targets:
            return

        dropped = len(targets)
        targets = targets[:len(self._order)]       # never exceed the master grid
        panes = self._fit_panes(len(targets), drop_pane)
        for series, pane in zip(targets, panes):
            self._open_series(series, pane)
        if panes:
            self._set_active_pane(panes[0])

        shown_n = min(len(targets), len(panes))
        if dropped > shown_n:
            self.statusBar().showMessage(
                t("Showing {shown} of {total} folders — the rest are in the "
                  "list on the left; drag one onto a pane to display it.",
                  shown=shown_n, total=dropped)
            )

    @staticmethod
    def _initial_noncT_target(candidates: list) -> "Series | None":
        """Which non-CT series to auto-open from *candidates* (display order).

        Prefer a real, playable acquisition over a static Secondary Capture
        report/snapshot. An XA study often carries an SC summary page (a
        multi-panel still) that sorts FIRST, so "just open the first series"
        made a folder drop show that still instead of the angio cine. Order:
          1) first multi-frame cine that is NOT a Secondary Capture
          2) first series that is NOT a Secondary Capture
          3) first candidate (all SC, or nothing else to choose)
        Returns None only when *candidates* is empty."""
        if not candidates:
            return None
        def _is_sc(se):
            return dicom_io._series_is_secondary_capture(se)
        cine = next(
            (se for se in candidates if se.image_count > 1 and not _is_sc(se)),
            None,
        )
        if cine is not None:
            return cine
        primary = next((se for se in candidates if not _is_sc(se)), None)
        return primary if primary is not None else candidates[0]

    @staticmethod
    def _initial_ct_target(candidates: list) -> "Series | None":
        """Which CT series to auto-open from *candidates*.

        Among CT series with ≥ 200 images (a full recon, not a scout/preview),
        pick the LARGEST series number. If none reach 200 images, pick the
        SMALLEST series number instead. Returns None when there are no CT
        series, so the caller keeps its non-CT default."""
        ct = [se for se in candidates if se.modality == Modality.CT]
        if not ct:
            return None
        big = [se for se in ct if se.image_count >= 200]
        if big:
            return max(big, key=lambda se: (se.number
                                            if se.number is not None else -1))
        return min(ct, key=lambda se: (se.number
                                       if se.number is not None else 1 << 30))

    def _reindex_series_maps(self) -> int:
        """Rebuild the uid lookup tables from the current patient tree.
        Returns the total series count. Call after any add/remove."""
        all_series = [
            se
            for p in self._patients.values()
            for st in p.studies.values()
            for se in st.series.values()
        ]
        self._series_by_uid = {se.series_uid: se for se in all_series}
        # SeriesInstanceUID -> StudyInstanceUID, so a completed measurement
        # can be filed under the right study's history.
        self._study_by_series_uid = {
            se.series_uid: st.study_uid
            for p in self._patients.values()
            for st in p.studies.values()
            for se in st.series.values()
        }
        return len(all_series)

    @staticmethod
    def _first_cine_fps_of(series) -> float | None:
        """Best-guess cine rate for a series, read from the first file's
        CineRate / RecommendedDisplayFrameRate / FrameTime. None when the
        series has none of those — caller falls back to a 15 fps default."""
        try:
            import pydicom
            ds = pydicom.dcmread(
                series.files[0], stop_before_pixels=True, force=True
            )
        except Exception:
            return None
        return dicom_io._cine_fps(ds)

    @staticmethod
    def _series_modality_key(series) -> str:
        """Bucket *series* into one of settings.TAG_MODALITIES so the
        per-modality tag/export-field memory keys consistently."""
        mod = getattr(series, "modality", None)
        val = getattr(mod, "value", None) or str(mod or "")
        return val if val in settings.TAG_MODALITIES else "OTHER"

    @staticmethod
    def _label_dicom_tags(series, identifiers: list) -> list:
        """Resolve ``[(identifier, label)]`` for the Export dialog.

        Reads the first file's header so private tags can show their
        DICOM element name (e.g. "(0019,1099) Private Field") instead
        of just the hex literal."""
        if not identifiers:
            return []
        try:
            import pydicom
            ds = pydicom.dcmread(
                series.files[0], stop_before_pixels=True, force=True
            )
        except Exception:
            ds = None
        from multi_dicomviewer.core.dicom_tags import _lookup
        out = []
        for ident in identifiers:
            elem = _lookup(ds, ident) if ds is not None else None
            if elem is not None:
                label = f"{ident}  ({elem.name})" if elem.name else ident
            else:
                label = ident
            out.append((ident, label))
        return out

    def _on_plane_export(self, fmt: str, uid: str, plane_path: str) -> None:
        """Image right-click ▸ Export DICOM / MP4 / CSV → run the same export
        as the Studies-list right-click, but scoped to the RIGHT-CLICKED
        image. *plane_path* (when non-empty) is the clicked biplane plane's
        own .dcm file: we build a single-file synthetic Series so the export
        targets just that plane (lossless DICOM copy, that plane's cine /
        tags). Empty *plane_path* exports the whole series (single-plane XA,
        IVUS, CT)."""
        series = self._series_by_uid.get(uid) if uid else None
        if series is None:
            self.statusBar().showMessage(
                t("Export: could not identify the displayed series.")
            )
            return
        if plane_path:
            import dataclasses
            target = dataclasses.replace(series, files=[plane_path])
        else:
            target = series
        # Image right-click: an MP4 should match what's on screen.
        self._on_export_requested(fmt, [target], use_display_transform=True)

    def _display_transform_for_uid(self, uid: str):
        """On-screen state of the viewer currently showing *uid*, as
        {"flip", "rot90", "free_rot", "window", "level"} — so an image
        right-click MP4 export bakes in what's shown (orientation, free rotation
        and current W/L). None when the series isn't shown or the viewer has no
        image canvas (e.g. the CT viewer), so those fall back to the original."""
        if not uid:
            return None
        for pane in self._shown_panes():
            if pane.shown_series_uid() != uid:
                continue
            v = pane.current_viewer()
            cv = getattr(v, "canvas", None)
            if cv is None or not hasattr(cv, "orient_state"):
                return None
            flip, rot90 = cv.orient_state()
            fr = (float(cv.free_rotation())
                  if hasattr(cv, "free_rotation") else 0.0)
            win = getattr(v, "_window", None)
            lvl = getattr(v, "_level", None)
            # Colour (RGB) series have no meaningful W/L — don't force one.
            if getattr(v, "_is_color", False):
                win = lvl = None
            return {"flip": flip, "rot90": rot90, "free_rot": fr,
                    "window": win, "level": lvl}
        return None

    def _on_angle_export(self, uid: str, payload: object) -> None:
        """IVUS "Export" of angle keyframes: reuse the export filename-tag
        picker (same dialog + per-modality memory as image export) to name the
        file, then write *payload* (a small JSON dict) to a user-chosen path."""
        import json
        import pydicom
        from multi_dicomviewer.core import export as exporter
        from multi_dicomviewer.ui.export_dialog import (
            DEFAULT_FIELDS,
            ExportDialog,
        )
        series = self._series_by_uid.get(uid) if uid else None
        if series is None:
            self.statusBar().showMessage(
                t("Export: could not identify the displayed series."))
            return
        export_mod = self._series_modality_key(series)
        tag_idents = list(self._tag_keywords_by_modality.get(export_mod, []))
        dicom_tags = self._label_dicom_tags(series, tag_idents)
        initial = list(
            self._export_fields_by_modality.get(export_mod) or []
        ) or list(DEFAULT_FIELDS)
        dlg = ExportDialog(
            "csv", 1, dicom_tags=dicom_tags, initial_fields=initial,
            title_override=t("Export Angle Set"), parent=self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        cfg = dlg.result_settings()
        self._export_fields_by_modality[export_mod] = list(cfg.fields)
        try:
            settings.save_export_fields_by_modality(
                self._export_fields_by_modality)
        except Exception:
            pass
        # Default filename from the picked tags (same builder as image export).
        base = "angle"
        try:
            ds = pydicom.dcmread(series.files[0], stop_before_pixels=True,
                                 force=True)
            base = exporter.build_filename(cfg.fields, series, ds) or base
        except Exception:
            pass
        path, _ = QFileDialog.getSaveFileName(
            self, t("Export Angle Set"), f"{base}.ivangle.json",
            t("IVUS Angle Set (*.ivangle.json)"))
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".ivangle.json"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1)
        except OSError as exc:
            self.statusBar().showMessage(str(exc))
            return
        self.statusBar().showMessage(t("Angle Set exported."))

    def _on_export_requested(self, fmt: str, series_list: list,
                             use_display_transform: bool = False) -> None:
        """Right-click ▸ Export (DICOM)/(MP4)/(CSV): show the filename-
        fields dialog, ask for an output folder, run the export with a
        live progress bar. Runs on the UI thread (simpler; MP4 of a few
        hundred frames is still seconds, not minutes). CSV writes one
        file per series listing the displayed DICOM-tag-overlay tags.

        *use_display_transform* is True only for the IMAGE right-click path:
        an MP4 then bakes in what the viewer shows (orientation + free rotation
        + current W/L). The Studies-list export leaves it False, so it always
        writes the ORIGINAL image regardless of on-screen state. DICOM (verbatim
        copy) and CSV (tag data) ignore it either way."""
        if not series_list or fmt not in (
            "dicom", "mp4", "csv", "anon-dicom"
        ):
            return
        from multi_dicomviewer.core import export as exporter
        from multi_dicomviewer.ui.export_dialog import (
            DEFAULT_FIELDS,
            ExportDialog,
        )

        # FPS default for MP4 = the source cine rate of the FIRST series
        # that has one; falls back to None (the dialog uses 15 fps).
        default_fps = None
        if fmt == "mp4":
            for s in series_list:
                fps = self._first_cine_fps_of(s)
                if fps:
                    default_fps = fps
                    break

        # Modality of the first picked series drives both:
        #   - the DICOM-tag list we offer in the filename-fields box
        #     (the same list the user has chosen in DICOM-Tag overlay)
        #   - per-modality memory of which fields were ticked last time
        export_mod = self._series_modality_key(series_list[0])
        tag_idents = list(
            self._tag_keywords_by_modality.get(export_mod, [])
        )
        dicom_tags = self._label_dicom_tags(series_list[0], tag_idents)
        initial = list(
            self._export_fields_by_modality.get(export_mod) or []
        ) or list(DEFAULT_FIELDS)

        dlg = ExportDialog(
            fmt, len(series_list), default_fps=default_fps,
            dicom_tags=dicom_tags, initial_fields=initial,
            parent=self,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        cfg = dlg.result_settings()
        # Persist this modality's selection for next time (best-effort).
        self._export_fields_by_modality[export_mod] = list(cfg.fields)
        try:
            settings.save_export_fields_by_modality(
                self._export_fields_by_modality
            )
        except Exception:
            pass
        # CSV content is the displayed tags regardless of filename fields, so
        # an empty selection only means a fallback ("export") filename — never
        # a reason to cancel. DICOM/MP4 still need at least one field.
        if not cfg.fields and fmt != "csv":
            self.statusBar().showMessage(
                t("Export cancelled — no filename fields were ticked.")
            )
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, t("Choose output folder for export")
        )
        if not out_dir:
            return

        title = {
            "dicom": t("Exporting DICOM…"),
            "mp4": t("Exporting MP4…"),
            "csv": t("Exporting CSV…"),
            "anon-dicom": t("Exporting Anon DICOM…"),
        }[fmt]
        prog = QProgressDialog(title, t("Cancel"), 0, 1, self)
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        prog.setAutoClose(False)
        prog.setAutoReset(False)
        prog.setValue(0)
        prog.show()
        QApplication.processEvents()

        cancelled = {"yes": False}

        def _cb(done: int, total: int, msg: str) -> None:
            if prog.wasCanceled():
                cancelled["yes"] = True
                raise RuntimeError("cancelled")
            if total and prog.maximum() != total:
                prog.setMaximum(total)
            prog.setValue(done)
            # Defensive: never let a long message (e.g. an error echoing a
            # huge filename) blow the dialog's width up — Qt sizes it to the
            # label and a megabyte-long line crashes the backing store.
            if msg:
                msg = msg if len(msg) <= 200 else msg[:200] + "…"
                if prog.labelText() != msg:
                    prog.setLabelText(msg)
            QApplication.processEvents()

        try:
            if fmt == "dicom":
                written = exporter.export_dicom(
                    series_list, out_dir, cfg.fields, progress=_cb
                )
            elif fmt == "anon-dicom":
                # De-identified copy using the active anonymization profile
                # (same tags the Anonymize toggle blanks on screen).
                written = exporter.export_anon_dicom(
                    series_list, out_dir, cfg.fields, progress=_cb
                )
            elif fmt == "csv":
                # Each series' CSV lists the overlay tags chosen for ITS own
                # modality (the selection differs per modality), matching the
                # tags actually shown on that series.
                per_series_idents = [
                    list(self._tag_keywords_by_modality.get(
                        self._series_modality_key(s), []))
                    for s in series_list
                ]
                written = exporter.export_csv(
                    series_list, out_dir, cfg.fields, per_series_idents,
                    anonymized=self._anon, progress=_cb,
                )
            else:
                # Per-series Play range (from the seek-bar markers), aligned
                # with series_list; None = export every frame.
                frame_ranges = [
                    self._mp4_ranges.get(s.series_uid) for s in series_list
                ]
                # Image right-click → bake each series' on-screen state
                # (orientation + free rotation + W/L) so it exports as shown.
                # Studies-list export leaves this None → original image.
                transforms = (
                    [self._display_transform_for_uid(s.series_uid)
                     for s in series_list]
                    if use_display_transform else None
                )
                written = exporter.export_mp4(
                    series_list, out_dir, cfg.fields,
                    bitrate_mbps=cfg.bitrate_mbps,
                    fps_override=cfg.fps,
                    crf=cfg.crf,
                    frame_ranges=frame_ranges,
                    transforms=transforms,
                    progress=_cb,
                )
        except RuntimeError as e:
            if cancelled["yes"]:
                self.statusBar().showMessage(t("Export cancelled."))
                prog.close()
                return
            prog.close()
            QMessageBox.critical(self, t("Export failed"), str(e))
            return
        except Exception as e:
            prog.close()
            QMessageBox.critical(self, t("Export failed"), str(e))
            return
        prog.close()
        n = len(written)
        if fmt in ("dicom", "anon-dicom"):
            msg = t("Exported {n} folder(s) to {dir}", n=n, dir=out_dir)
        else:
            msg = t("Exported {n} file(s) to {dir}", n=n, dir=out_dir)
        self.statusBar().showMessage(msg)

    def _delete_node(self, kind: str, key: str, label: str) -> None:
        """Right-click ▸ delete: drop a patient/study/series from the list
        (and blank any pane showing it). Files are not touched.

        No confirmation dialog — the menu item itself already says
        "remove from the list", the action is non-destructive (files are
        untouched; reloading the folder restores it), and the dialog only
        repeated that same wording."""
        removed = dicom_io.remove_node(self._patients, kind, key)
        if not removed:
            return
        self._reindex_series_maps()
        for pane in self._panes:
            if pane.shown_series_uid() in removed:
                pane.reset()
        if self._cur_xa is not None and self._cur_xa.series_uid in removed:
            self._cur_xa = None
        self.browser.populate(self._patients)  # keeps expand state
        self.statusBar().showMessage(
            t("Deleted ({n} series removed from the list)", n=len(removed))
        )

    def _delete_all_nodes(self) -> None:
        """"Delete All" button → remove EVERY study/series from the list (and
        empty every pane). Files on disk are untouched — reloading the folder
        restores them — but it clears everything, so confirm first."""
        if not self._patients:
            self.statusBar().showMessage(
                t("Nothing to delete — the list is empty."))
            return
        n_series = sum(
            len(st.series)
            for p in self._patients.values()
            for st in p.studies.values()
        )
        reply = QMessageBox.question(
            self, t("Delete All"),
            t("Remove all {n} series from the list?\n\n"
              "The image files on disk are NOT deleted (reloading the folder "
              "restores them).", n=n_series),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        for pane in self._panes:
            pane.reset()
        self._patients.clear()
        self._cur_xa = None
        self._cur_study_uid = None
        self._last_by_study.clear()
        self._last_by_modality.clear()
        self._reindex_series_maps()
        self.browser.populate(self._patients)
        self.browser.show_empty()          # blank the tree highlight + thumbs
        self._sync_xa_shortcuts()
        self._sync_layout_gate()
        self.statusBar().showMessage(
            t("Deleted all ({n} series removed from the list)", n=n_series)
        )

    def _on_series_chosen(self, series: Series) -> None:
        """Click / keyboard nav in the browser → load into the active pane."""
        self._open_series(series, self._active)

    def _on_study_clicked(self, study_uid: str, kind: str) -> None:
        """Row click on a Study header in the browser tree: resume the
        last-viewed series of THAT study node (so XA/IVUS don't reset to
        series #1 after the user moves to another study and back).

        Scoped to (study_uid, kind): a study_uid shared by two nodes (XA +
        OT on the same date) resolves to the clicked node's own kind, never
        the sibling's — otherwise clicking the XA node jumped to OT."""
        tgt = self._study_target_series(study_uid, kind)
        if tgt is not None:
            self.browser.select_series(tgt)

    def _study_target_series(self, study_uid: str, kind: str) -> "Series | None":
        """The series a click / drop on Study node (study_uid, kind) should
        open: the last-viewed series of that node if still present & visible,
        else its first VISIBLE series in display order. None if it has none."""
        last = self._last_by_study.get((study_uid, kind))
        if (last is not None and last.series_uid in self._series_by_uid
                and not getattr(last, "hidden", False)):
            return last
        for se in self.browser.ordered_series():
            if (self._study_by_series_uid.get(se.series_uid) == study_uid
                    and se.kind == kind
                    and not getattr(se, "hidden", False)):
                return se
        return None

    def _on_study_dropped(self, pane: ViewerPane, study_uid: str,
                          kind: str) -> None:
        """A Study node was dragged onto *pane* → open that study's resume /
        first series directly into it (the drag made *pane* active first)."""
        tgt = self._study_target_series(study_uid, kind)
        if tgt is not None:
            self._open_series(tgt, pane)

    def _on_series_dropped(self, pane: ViewerPane, uid: str) -> None:
        series = self._series_by_uid.get(uid)
        if series is not None:
            self._open_series(series, pane)

    @staticmethod
    def _fmt_date(raw: str) -> str:
        """Normalise a DICOM date to the raw "YYYYMMDD" 8-digit form (kept as-is,
        no separators). Returns "" for an empty/unrecognised value."""
        raw = (raw or "").strip()
        return raw[:8] if len(raw) >= 8 and raw[:8].isdigit() else ""

    @staticmethod
    def _clean_name(raw) -> str:
        """DICOM PersonName → a short alphabetic display name.

        A PersonName can carry up to three "=" separated representations
        (Alphabetic=Ideographic=Phonetic, i.e. romaji=漢字=カナ). Keep ONLY the
        first (alphabetic) one so the pane title stays short, then turn the
        "Family^Given^…" component carets into spaces and collapse whitespace.
        """
        alpha = str(raw or "").split("=", 1)[0]
        return " ".join(alpha.replace("^", " ").split())

    def _series_patient_study(self, series: Series):
        """The (Patient, Study) *series* belongs to, or (None, None)."""
        study_uid = self._study_by_series_uid.get(series.series_uid, "")
        for p in self._patients.values():
            st = p.studies.get(study_uid)
            if st is not None:
                return p, st
        return None, None

    def _pane_bar(self, series: Series) -> str:
        """Pane top-band body: "Name - YYYYMMDD - SeriesNo/InstanceNo".
        Missing fields are dropped; a missing series/instance number shows "?".
        (The pane prefixes "● Pane N - " itself.)

        When Anonymize is ON the real patient name and date are MASKED — the
        literal placeholders "Name" and "Date" are shown instead (the
        series/instance numbers are not PHI and stay)."""
        if self._anon:
            name, date = t("Name"), t("Date")
        else:
            patient, study = self._series_patient_study(series)
            name = self._clean_name(patient.name) if patient is not None else ""
            date = self._fmt_date(study.date) if (study and study.date) \
                else self._fmt_date((getattr(series, "acq_time", "") or "")[:8])
        sn = series.number if series.number is not None else "?"
        inst = series.instance_number if series.instance_number is not None \
            else "?"
        parts = [p for p in (name, date, f"{sn}/{inst}") if p]
        return _PANE_SEP.join(parts)

    def _open_series(self, series: Series, pane: ViewerPane) -> None:
        # Mac build cannot render CT (VTK's OpenGL→Metal path hangs). Tell
        # the user explicitly and abort the load before any disk read /
        # viewer construction touches VTK.
        if BLOCK_CT and series.modality == Modality.CT:
            QMessageBox.information(self, t("Unsupported data"), BLOCK_CT_MESSAGE)
            self.statusBar().showMessage(
                t("CT data cannot be loaded in this build.")
            )
            return
        # Fast-path: this exact series is already loaded in this pane's
        # cached viewer (XAViewer / IVUSViewer / CTViewer track
        # _loaded_uid). Skip the whole disk read + decode + viewer
        # rebuild pipeline — and therefore skip the CT progress dialog
        # too — so returning to a series is instant and its frame /
        # camera / W-L / measurements all stay put.
        if pane.is_loaded(series.modality, series.series_uid):
            pane.switch_to_loaded(series.modality, self._pane_bar(series))
            pane.set_shown_series(series.series_uid)
            pane._series_ref = series
            self._touch_pane(pane)
            self._cur_study_uid = self._study_by_series_uid.get(
                series.series_uid
            )
            v = pane.current_viewer()
            if v is not None and hasattr(v, "set_tag_keywords"):
                v.set_anonymized(self._anon)
                self._migrate_tag_idents(v)
                v.set_tag_keywords(self._effective_kw(v))
            if series.modality == Modality.XA:
                self._cur_xa = series
            # Resumed viewer keeps the Plane (Bi/Lt/Rt) side it was left on
            # (stored in the viewer's own "Plane:" bar) — nothing to do here.
            # Key by raw DICOM Modality (series.kind) so NM/OCT/etc.
            # in the OTHER bucket each get their own resume slot.
            self._last_by_modality[series.kind] = series
            study_uid = self._study_by_series_uid.get(series.series_uid, "")
            if study_uid:
                self._last_by_study[(study_uid, series.kind)] = series
            self._sync_xa_shortcuts()
            self._update_cine_series_pos(pane)
            if pane is self._active:
                self._follow_active_pane()
            self._enforce_live_caps(keep=pane)
            self.statusBar().showMessage(t("Resumed {label}", label=series.label))
            return
        # A load already running for this pane? Ignore the repeat request so
        # the same series isn't decoded twice into one pane (loading a
        # DIFFERENT pane meanwhile is fine — it gets its own worker).
        if pane in self._loads:
            self.statusBar().showMessage(
                t("Still loading {label} …", label=series.label))
            return
        self.statusBar().showMessage(t("Loading {label} …", label=series.label))
        # Cut the OUTGOING series's background prefetch BEFORE reading
        # any new files so disk / CPU bandwidth is freed for the new
        # load instead of contending with the old decode. Only on the
        # slow path (the fast-path above didn't reload anything, so its
        # prefetch should keep warming for the same series).
        cur_v = pane.current_viewer()
        if cur_v is not None and hasattr(cur_v, "_stop_prefetch"):
            cur_v._stop_prefetch()
        # CT load (read every slice + build the HU volume) is the multi-second
        # wait the user saw after the scan bar hit 100%. It now runs on a
        # WORKER THREAD so the rest of the app stays usable while it reads;
        # a NON-modal phased progress dialog tracks it (was app-modal, which
        # froze everything). XA/IVUS goes the same route — a 100-300 MB cine
        # clip can also take seconds to read + decode frame 0. Only the final
        # GPU/VTK build runs back on the GUI thread (see _finish_open_series).
        is_ct = series.modality == Modality.CT
        title = (t("Loading CT") if is_ct
                 else t("Loading {kind}", kind=series.kind or 'DICOM'))
        initial_msg = (t("Reading CT slices…") if is_ct
                       else t("Reading DICOM file…"))
        dlg = QProgressDialog(initial_msg, None, 0, 0, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        dlg.show()

        thread = QThread(self)
        worker = _SeriesLoadWorker(series)
        worker.moveToThread(thread)
        self._loads[pane] = (thread, worker, dlg, series)

        # Connect to BOUND METHODS of self (not lambdas): self lives on the GUI
        # thread, so Qt auto-uses a queued connection and the slot runs on the
        # GUI thread — safe to touch the dialog / build the view. A lambda has
        # no QObject affinity and would run in the WORKER thread instead. The
        # pane/series context is recovered from the registry via sender().
        worker.progress.connect(self._on_load_progress)
        worker.done.connect(self._on_load_done)
        worker.failed.connect(self._on_load_failed)
        thread.started.connect(worker.run)
        thread.start()

    def _load_entry_for_sender(self):
        """(pane, entry) for the worker that emitted the signal now running, or
        (None, None) if it's already been cleaned up."""
        worker = self.sender()
        for pane, entry in self._loads.items():
            if entry[1] is worker:
                return pane, entry
        return None, None

    def _on_load_progress(self, phase: str, done: int, total: int) -> None:
        _pane, entry = self._load_entry_for_sender()
        if entry is None:
            return
        dlg = entry[2]
        if dlg.labelText() != phase:
            dlg.setLabelText(phase)
        if total and dlg.maximum() != total:
            dlg.setMaximum(total)
        dlg.setValue(done)

    def _on_load_done(self, loaded) -> None:
        pane, entry = self._load_entry_for_sender()
        if pane is None:
            return
        self._finish_open_series(pane, entry[3], loaded)

    def _on_load_failed(self, msg: str) -> None:
        pane, entry = self._load_entry_for_sender()
        if pane is None:
            return
        self._fail_open_series(pane, entry[3], msg)

    def _cleanup_load(self, pane) -> None:
        """Tear down a finished background load: stop its thread and drop the
        registry entry (also closes the progress dialog)."""
        entry = self._loads.pop(pane, None)
        if entry is None:
            return
        thread, worker, dlg = entry[0], entry[1], entry[2]
        dlg.close()
        thread.quit()
        thread.wait()
        worker.deleteLater()
        thread.deleteLater()

    def _fail_open_series(self, pane, series, msg: str) -> None:
        """Background load raised — surface it exactly as the old inline path."""
        self._cleanup_load(pane)
        QMessageBox.critical(
            self, t("Load failed"), f"{series.label}\n\n{msg}")
        self.statusBar().showMessage(t("Load failed."))

    def _finish_open_series(self, pane, series, loaded) -> None:
        """Runs on the GUI thread once the worker has the decoded series.
        Builds the GPU/VTK view (Qt objects must live here) and does all the
        post-load bookkeeping the old synchronous path did."""
        entry = self._loads.get(pane)
        dlg = entry[2] if entry else None
        if dlg is not None:
            # The VTK pipeline build has no fine-grained progress, so show
            # an indeterminate "constructing" bar for that last phase.
            dlg.setLabelText(t("Constructing 3D view…"))
            dlg.setMaximum(0)            # 0,0 -> busy/indeterminate bar
            dlg.setValue(0)
            QApplication.processEvents()

        self._cur_study_uid = self._study_by_series_uid.get(series.series_uid)
        # Free the least-recently-used live panes of THIS modality BEFORE
        # building the (possibly large) new volume, so the build never
        # transiently holds both its and the outgoing volume's GPU/RAM — the
        # exhaustion that showed a black CT recoverable only by an app restart.
        incoming_bytes = int(getattr(getattr(loaded, "volume", None),
                                     "nbytes", 0) or 0)
        self._free_live_for_incoming(pane, loaded.modality, incoming_bytes)
        try:
            pane.show_series(loaded, series.label, self._pane_bar(series))
        except Exception as exc:                          # noqa: BLE001
            # Safety net: if the GPU/VTK build still fails (e.g. a single
            # volume larger than available graphics memory), surface it instead
            # of leaving a silently-black pane that forces a restart.
            traceback.print_exc()
            self._cleanup_load(pane)
            pane.reset()
            QMessageBox.critical(
                self, t("Load failed"),
                t("Could not display {label}.\nGraphics/RAM may be exhausted "
                  "by several large CT series open at once — close another CT "
                  "pane and try again.", label=series.label))
            self.statusBar().showMessage(t("Load failed."))
            return
        pane._series_ref = series          # remembered for LRU demote / promote
        pane._volume_bytes = incoming_bytes    # for the live-memory budget
        self._touch_pane(pane)             # freshly loaded = most-recently-used
        self._cleanup_load(pane)
        # Carry the current anonymize + tag-overlay choices onto the
        # (possibly newly built) viewer.
        v = pane.current_viewer()
        pane.set_shown_series(series.series_uid if v is not None else None)
        if v is not None and hasattr(v, "set_tag_keywords"):
            v.set_anonymized(self._anon)
            self._migrate_tag_idents(v)
            v.set_tag_keywords(self._effective_kw(v))
        if loaded.modality == Modality.XA:
            self._cur_xa = series
        # The viewer itself defaults a freshly loaded biplane/dual series to
        # "Bi" (both planes) in its own "Plane:" bar; the user switches THIS
        # pane to Lt/Rt independently afterwards.
        # Remember the last series per modality and per study so the
        # cine viewers' First/Prev/Next/Last nav and the Study-row
        # click both resume from here after the user moves away and
        # comes back. Key by raw DICOM Modality (series.kind) so the
        # OTHER bucket (NM/OCT/OFD/…) keeps each kind's own resume.
        self._last_by_modality[series.kind] = series
        study_uid_log = self._study_by_series_uid.get(series.series_uid, "")
        if study_uid_log:
            self._last_by_study[(study_uid_log, series.kind)] = series
        self._sync_xa_shortcuts()  # XA keys only when active pane is XA
        self._update_cine_series_pos(pane)
        # A drop / folder-load makes the pane active BEFORE the series is in
        # it, so the browser was blanked on the way in — re-follow now that
        # the active pane has content (also keeps tree+thumb in step for any
        # load path).
        if pane is self._active:
            self._follow_active_pane()
        # Keep an open history window pointed at the now-current study.
        if self._hist_dialog is not None and self._hist_dialog.isVisible():
            self._refresh_history_dialog()
        # Now that this pane is fully live, sleep the least-recently-used panes
        # of the same modality if we're over the memory cap — freeing their
        # volume/clip while keeping their last image on screen.
        self._enforce_live_caps(keep=pane)
        self.statusBar().showMessage(t("Loaded {label}", label=series.label))

    def _clear_all(self) -> None:
        for pane in self._panes:
            pane.reset()
        # The active pane is now empty → blank the browser highlight/thumbnail
        # and re-sync the cine shortcuts / layout gate (panes reset directly,
        # bypassing _set_active_pane).
        self._follow_active_pane()
        self._sync_xa_shortcuts()
        self._sync_layout_gate()
        self.statusBar().showMessage(t("Viewers cleared."))

    # ------------------------------------- per-viewer signal/option wiring
    def _wire_viewer(self, viewer) -> None:
        """Connect a freshly built viewer's signals and push current
        display options onto it."""
        # First/Prev/Next/Last cross-series nav: cine viewers step the cine
        # list, the CT viewer steps the study's CT series.
        if hasattr(viewer, "series_nav"):
            viewer.series_nav.connect(
                self._nav_xa if _is_cine(viewer) else self._nav_ct)
        # Measuring / history are modality-agnostic (XA and IVUS both).
        if hasattr(viewer, "measurement_added"):
            viewer.measurement_added.connect(self._record_measurement)
        if hasattr(viewer, "measurement_removed"):
            viewer.measurement_removed.connect(self._remove_measurement)
        if hasattr(viewer, "history_requested"):
            viewer.history_requested.connect(self._show_history)
        # CT HU colour map is global: when it's edited in one CT pane, mirror it
        # onto every other CT pane (persistence is done in the viewer).
        if hasattr(viewer, "colormap_changed"):
            viewer.colormap_changed.connect(
                lambda bands, op, smooth, src=viewer:
                self._propagate_ct_colormap(src, bands, op, smooth))
        if hasattr(viewer, "tags_requested"):
            viewer.tags_requested.connect(
                lambda vv=viewer: self._open_tag_dialog(vv)
            )
        # Right-click ▸ Export DICOM/MP4/CSV on the image → reuse the same
        # series export the Studies-list right-click uses, scoped to the
        # clicked plane.
        if hasattr(viewer, "plane_export_requested"):
            viewer.plane_export_requested.connect(self._on_plane_export)
        # IVUS "Export" of angle keyframes → reuse the export filename-tag
        # picker, then write the small JSON file.
        if hasattr(viewer, "angle_export_requested"):
            viewer.angle_export_requested.connect(self._on_angle_export)
        if hasattr(viewer, "set_tag_keywords"):
            viewer.set_anonymized(self._anon)
            viewer.set_tag_keywords(self._effective_kw(viewer))
        # Play-range markers → remember each series' MP4 export range.
        if hasattr(viewer, "play_range_changed"):
            viewer.play_range_changed.connect(self._on_play_range_changed)
        # DICOM-tag overlay text size: one slider per viewer, all kept in sync.
        if hasattr(viewer, "overlay_font_changed"):
            viewer.overlay_font_changed.connect(self._set_tag_font_pt)
        if hasattr(viewer, "set_overlay_font_pt"):
            viewer.set_overlay_font_pt(self._tag_font_pt)
        # IVUS long-axis allowed only in 1x1 — apply the current layout's gate
        # to this (possibly just-created) viewer.
        if hasattr(viewer, "set_long_axis_allowed"):
            viewer.set_long_axis_allowed(self._layout_key == "1x1")
        # Cap how many panes may play a cine at once (avoids decode/render
        # overload across panes).
        if hasattr(viewer, "set_play_gate"):
            viewer.set_play_gate(self._play_gate_check)

    def _play_gate_check(self, viewer) -> bool:
        """Veto STARTING playback when _PLAY_CAP panes already play. Counts the
        cine viewers currently shown in OTHER panes; if that already reaches the
        cap, tell the user and refuse."""
        others = 0
        for pane in self._panes:
            v = pane.current_viewer()
            if v is None or v is viewer:
                continue
            if getattr(v, "is_playing", None) and v.is_playing():
                others += 1
        if others >= _PLAY_CAP:
            QMessageBox.information(
                self, t("Playback limit"),
                t("At most {n} data can play at once.\n"
                  "Stop one of the playing data to play another.", n=_PLAY_CAP),
            )
            return False
        return True

    def _on_play_range_changed(
        self, uid: str, start: int, end: int, total: int
    ) -> None:
        """A cine viewer's Play-range markers moved (or a series loaded).
        Remember the range for the MP4 export, dropping it when it spans the
        whole clip so the default stays "export every frame"."""
        if not uid:
            return
        if start <= 0 and end >= total - 1:
            self._mp4_ranges.pop(uid, None)
        else:
            self._mp4_ranges[uid] = (int(start), int(end))

    def _set_tag_font_pt(self, pt: int) -> None:
        """Record the new DICOM-tag overlay text size and keep the shell's
        global Tag-size slider in step (no recursion: blocked). The actual
        broadcast to every viewer is debounced via ``_tag_font_timer`` so a
        slider drag doesn't re-render all panes on every tick (see
        :py:meth:`_apply_tag_font_pt`)."""
        pt = int(pt)
        self._tag_font_pt = pt
        sl = getattr(self, "_tag_font_slider", None)
        if sl is not None and sl.value() != pt:
            sl.blockSignals(True)
            sl.setValue(pt)
            sl.blockSignals(False)
        # Coalesce a burst of valueChanged into one apply shortly after the
        # user stops dragging. (restart() resets the countdown each tick.)
        self._tag_font_timer.start()

    def _apply_tag_font_pt(self) -> None:
        """Broadcast the latest DICOM-tag overlay text size to every viewer in
        every pane so the size stays uniform across modalities. Debounced from
        :py:meth:`_set_tag_font_pt` so the (per-pane repaint / CT VTK render)
        cost is paid once per drag, not once per slider step."""
        pt = int(self._tag_font_pt)
        for pane in self._panes:
            for v in pane.all_viewers():
                if hasattr(v, "set_overlay_font_pt"):
                    v.set_overlay_font_pt(pt)

    def _open_tag_dialog_active(self) -> None:
        """Global "DICOM Tags" button (top row): open the overlay-tag picker
        for the ACTIVE pane's viewer (its modality's selection applies to
        every pane of that modality)."""
        v = self._active.current_viewer() if self._active is not None else None
        if v is None:
            QMessageBox.information(
                self, t("DICOM Tags"),
                t("Load a series first, then choose overlay items."),
            )
            return
        self._open_tag_dialog(v)

    # ------------------------------------------- anonymize / DICOM-tag overlay
    @staticmethod
    def _modality_of(viewer) -> str:
        """Classify *viewer* into one of the persisted-list buckets."""
        m = getattr(viewer, "handles_modality", "") or "OTHER"
        return m if m in settings.TAG_MODALITIES else "OTHER"

    def _migrate_tag_idents(self, viewer) -> None:
        """Self-heal legacy raw private-tag literals in *viewer*'s modality
        list to the stable private-creator key, resolved against the shown
        header. A selection saved before the private-creator change thus
        upgrades automatically just by viewing the series it was made on —
        no DICOM Tags re-confirm needed. No-op when nothing legacy resolves."""
        if viewer is None or not hasattr(viewer, "current_header"):
            return
        header = viewer.current_header()
        if header is None:
            return
        mod = self._modality_of(viewer)
        kws = self._tag_keywords_by_modality.get(mod, [])
        if not kws:
            return
        upgraded: list[str] = []
        seen: set[str] = set()
        changed = False
        for k in kws:
            nk = upgrade_private_literal(header, k)
            if nk != k:
                changed = True
            if nk not in seen:
                seen.add(nk)
                upgraded.append(nk)
        if changed:
            self._tag_keywords_by_modality[mod] = upgraded
            settings.save_tag_keywords_by_modality(
                self._tag_keywords_by_modality
            )

    def _effective_kw(self, viewer=None) -> list:
        """Keywords actually pushed to *viewer*: none while the overlay
        is hidden, otherwise the user's selection FOR THAT VIEWER's
        modality. Falls back to the active pane's modality when no
        viewer is supplied."""
        if self._overlay_hidden:
            return []
        if viewer is None and self._active is not None:
            viewer = self._active.current_viewer()
        mod = self._modality_of(viewer) if viewer is not None else "OTHER"
        return list(self._tag_keywords_by_modality.get(mod, []))

    def _apply_overlay_hidden(self, hidden: bool) -> None:
        """Core: push the (possibly empty) overlay to every viewer.
        Each viewer gets its OWN modality's list."""
        self._overlay_hidden = bool(hidden)
        for v in self._tag_viewers():
            v.set_tag_keywords(self._effective_kw(v))
            # The pygfx CT viewer shows default tags when none are selected, so
            # an empty keyword list can't hide it — toggle its overlay directly.
            if hasattr(v, "set_overlay_hidden"):
                v.set_overlay_hidden(self._overlay_hidden)
        self.statusBar().showMessage(
            t("DICOM overlay: hidden") if hidden else t("DICOM overlay: shown")
        )

    def _set_overlay_shown(self, show: bool) -> None:
        """Single source of truth for the DICOM-info overlay show/hide. Keeps
        every control in step (all blocked to avoid re-entrancy): the tree's
        DICOM Info button, the top-row DICOM Info button and the View-menu
        'Hide DICOM overlay' action. All three toggle paths funnel here."""
        show = bool(show)
        self._apply_overlay_hidden(not show)
        self.browser.set_dicom_info_shown(show)        # already blocks signals
        if hasattr(self, "_hide_overlay_act"):
            self._hide_overlay_act.blockSignals(True)
            self._hide_overlay_act.setChecked(not show)
            self._hide_overlay_act.blockSignals(False)
        if hasattr(self, "_tags_btn"):
            self._tags_btn.blockSignals(True)
            self._tags_btn.setChecked(show)
            self._tags_btn.blockSignals(False)

    def _toggle_overlay_hidden(self, on: bool) -> None:
        """View ▸ Hide DICOM overlay (checked = hidden)."""
        self._set_overlay_shown(not on)

    def _on_dicom_info_btn(self, show: bool) -> None:
        """Tree 'DICOM Info' button (checked = show)."""
        self._set_overlay_shown(show)

    def _tag_viewers(self) -> list:
        out = []
        for pane in self._panes:
            for v in pane.all_viewers():
                if hasattr(v, "set_tag_keywords"):
                    out.append(v)
        return out

    def _set_anonymized(self, on: bool) -> None:
        on = bool(on)
        if on == self._anon:
            return
        self._anon = on
        # Keep the menu action and the browser's "Anonymous" button in
        # lock-step without re-entering this slot.
        self._anon_act.blockSignals(True)
        self._anon_act.setChecked(on)
        self._anon_act.blockSignals(False)
        self.browser.set_anonymized(on)  # also reflects its own button
        for v in self._tag_viewers():
            v.set_anonymized(on)
        # Mask/unmask the Name + Date in every pane's top-band title now, so the
        # change is immediate (the title is otherwise only built on open).
        self._refresh_pane_titles()
        self.statusBar().showMessage(
            t("Anonymize: ON (case info masked on screen)")
            if on
            else t("Anonymize: OFF")
        )

    def _refresh_pane_titles(self) -> None:
        """Re-apply each loaded pane's top-band title (e.g. after the Anonymize
        toggle, so Name/Date mask or unmask without reopening the series)."""
        for pane in self._panes:
            if pane.current_viewer() is None:
                continue
            uid = pane.shown_series_uid()
            se = self._series_by_uid.get(uid) if uid else None
            if se is not None:
                pane._set_pane_title(self._pane_bar(se))

    def _open_anon_settings(self) -> None:
        """Right-click on the Anonymous button → choose which tags the
        Anonymize toggle AND Export (Anon DICOM) blank. Saves the profile and
        redraws overlays so a change takes effect immediately."""
        from multi_dicomviewer.ui.anon_dialog import AnonSettingsDialog
        dlg = AnonSettingsDialog(
            self._anon_tags, self._anon_emptify_private, self
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        tags, emptify_private = dlg.result_profile()
        self._anon_tags = tags
        self._anon_emptify_private = emptify_private
        anonymize.set_anon_profile(tags, emptify_private)
        settings.save_anon_profile(tags, emptify_private)
        # Redraw overlays so the new masking shows at once (when anon is on).
        for v in self._tag_viewers():
            if hasattr(v, "set_tag_keywords"):
                v.set_tag_keywords(self._effective_kw(v))
        self.statusBar().showMessage(
            t("Anonymize settings updated: {n} tag(s)", n=len(tags))
            + (t(" + private") if emptify_private else "")
        )

    def _open_settings(self) -> None:
        """Top-bar Settings button: display count + angio image quality + CT
        colour map, in one popup. Saved and applied live on OK."""
        from multi_dicomviewer.ui.settings_dialog import SettingsDialog
        caps = {"CT": self._live_cap[Modality.CT],
                "XA": self._live_cap[Modality.XA]}
        quality = settings.load_display_quality()
        dlg = SettingsDialog(caps, quality, self._open_ct_color,
                             on_advanced=self._open_advanced_quality,
                             parent=self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        # Display count: persist + apply (over-cap panes sleep now; keep=active
        # so the focused pane is never the one demoted).
        vals = dlg.caps()
        settings.save_live_caps(vals)
        self._live_cap = {Modality.CT: vals["CT"], Modality.XA: vals["XA"]}
        self._enforce_live_caps(keep=self._active)
        # Angio quality: merge into the persisted prefs, then push to every
        # loaded XA viewer (updates its canvases + S-Cine/S-Zoom/Denoise
        # buttons) so the change takes effect without reopening the series.
        q = settings.load_display_quality()
        q.update(dlg.quality())
        settings.save_display_quality(q)
        for v in self._all_loaded_viewers():
            if hasattr(v, "reload_display_quality"):
                v.reload_display_quality()
        self.statusBar().showMessage(
            t("Settings saved (CT {ct} / Angio {xa} live)",
              ct=vals["CT"], xa=vals["XA"]))

    def _all_loaded_viewers(self) -> list:
        """Every cached viewer across all panes (deduplicated)."""
        seen, out = set(), []
        for pane in self._panes:
            for v in pane.all_viewers():
                if id(v) not in seen:
                    seen.add(id(v))
                    out.append(v)
        return out

    def _open_advanced_quality(self) -> None:
        """Settings ▸ Angio image quality ▸ Advanced… : fine denoise / sharpen /
        CLAHE with a live preview (XA + IVUS only). Saved + applied on OK."""
        from multi_dicomviewer.ui.advanced_quality_dialog import (
            AdvancedQualityDialog)
        dlg = AdvancedQualityDialog(settings.load_advanced_quality(), self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        settings.save_advanced_quality(dlg.values())
        for v in self._all_loaded_viewers():
            if hasattr(v, "reload_display_quality"):
                v.reload_display_quality()

    def _propagate_ct_colormap(self, source, bands, opacity, smooth) -> None:
        """Mirror a colour-map edit from *source* onto every other CT viewer so
        the HU colour map is shared app-wide (each already persisted it)."""
        for v in self._all_loaded_viewers():
            if v is not source and hasattr(v, "apply_global_colormap"):
                v.apply_global_colormap(bands, opacity, smooth)

    def _open_ct_color(self, parent=None) -> None:
        """Open the HU colour-map editor for a CT pane, modal ON TOP of the
        Settings popup (*parent*). Prefers the active pane's CT viewer; else the
        first loaded CT viewer. Tells the user to load a CT series if none open."""
        cands = []
        av = self._active.current_viewer() if self._active else None
        if av is not None:
            cands.append(av)
        cands += [v for v in self._all_loaded_viewers() if v is not av]
        for v in cands:
            if hasattr(v, "_open_setting") and getattr(v, "_vol", None) \
                    is not None:
                v._open_setting(parent=parent, modal=True)
                return
        QMessageBox.information(
            parent or self, t("CT colour"),
            t("Load a CT series first, then edit its HU colour map."))

    def _fit_studies_dock_width(self, width: int) -> None:
        """Thumbnail "Fit: min × 10 across" → widen the Studies dock toward
        *width* px (capped so the viewers keep room). Best-effort: the grid
        wraps to whatever actually fits."""
        dock = self._studies_dock
        if not dock.isVisible():
            dock.setVisible(True)
            self._info_btn.setText(t("◀ Hide Studies"))
        # Leave only a small slice for the panes so the Tree can get large;
        # resizeDocks still won't shrink the central area below its own minimum.
        cap = max(300, self.width() - 120)
        w = max(200, min(int(width), cap))
        self.resizeDocks([dock], [w], Qt.Orientation.Horizontal)

    def _step_studies_dock_width(self, delta: int) -> None:
        """◀ / ▶ toolbar buttons → widen/narrow the Studies dock by *delta* px.
        An always-available alternative to dragging the thin separator. Clamped
        to the dock minimum and to leaving the central panes usable room."""
        dock = self._studies_dock
        if not dock.isVisible():
            dock.setVisible(True)
            self._info_btn.setText(t("◀ Hide Studies"))
        # Leave only a small slice for the panes so the Tree can get large;
        # resizeDocks still clamps to the central area's own minimum width.
        cap = max(dock.minimumWidth(), self.width() - 120)
        w = max(dock.minimumWidth(), min(dock.width() + delta, cap))
        self.resizeDocks([dock], [w], Qt.Orientation.Horizontal)

    def _open_tag_dialog(self, viewer) -> None:
        header = (
            viewer.current_header()
            if hasattr(viewer, "current_header")
            else None
        )
        if header is None:
            QMessageBox.information(
                self,
                t("DICOM Tags"),
                t("Load a series first, then choose overlay items."),
            )
            return
        mod = self._modality_of(viewer)
        current = self._tag_keywords_by_modality.get(mod, [])
        seed = current or default_overlay_keywords(header)
        dlg = TagSelectionDialog(header, seed, self._anon, self)
        if dlg.exec():
            self._apply_tag_keywords(dlg.selected_keywords(), modality=mod)

    def _apply_tag_keywords(
        self,
        keywords,
        modality: str = "XA",
        persist: bool = True,
    ) -> None:
        """Set the overlay selection FOR *modality*, push it to that
        modality's viewers, and (by default) persist it so it survives
        the next app launch."""
        self._tag_keywords_by_modality[modality] = list(keywords)
        for v in self._tag_viewers():
            if self._modality_of(v) == modality:
                v.set_tag_keywords(self._effective_kw(v))
        if persist:
            settings.save_tag_keywords_by_modality(
                self._tag_keywords_by_modality
            )

    def _export_tag_conditions(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            t("Export DICOM tag overlay settings"),
            "tag_conditions.json",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            settings.export_tag_keywords(
                path, self._tag_keywords_by_modality
            )
        except OSError as exc:
            QMessageBox.critical(self, t("Export failed"), str(exc))
            return
        total = sum(
            len(v) for v in self._tag_keywords_by_modality.values()
        )
        self.statusBar().showMessage(
            t("Exported overlay settings ({n} items across modalities)",
              n=total)
        )

    def _import_tag_conditions(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            t("Import DICOM tag overlay settings"),
            "",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            imported = settings.import_tag_keywords(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, t("Import failed"), str(exc))
            return
        # Imported conditions become the new persisted defaults — accept
        # either a legacy single list (applied to every modality) or a
        # v2 per-modality dict.
        if isinstance(imported, list):
            by_mod = {mod: list(imported)
                       for mod in settings.TAG_MODALITIES}
        else:
            by_mod = imported
        self._tag_keywords_by_modality = by_mod
        for v in self._tag_viewers():
            v.set_tag_keywords(self._effective_kw(v))
        settings.save_tag_keywords_by_modality(
            self._tag_keywords_by_modality
        )
        kws = by_mod.get("XA", [])
        self.statusBar().showMessage(
            t("Imported overlay settings ({n} items)", n=len(kws))
        )

    def _tags_for_active_pane(self) -> None:
        ordered = [self._active] + [
            p for p in self._panes if p is not self._active
        ]
        for pane in ordered:
            v = pane.current_viewer()
            if (
                v is not None
                and hasattr(v, "current_header")
                and v.current_header() is not None
            ):
                self._open_tag_dialog(v)
                return
        QMessageBox.information(
            self,
            t("DICOM Tags"),
            t("Load a series first, then choose overlay items."),
        )

    # ----------------------------------------------- measurement history
    def _record_measurement(self, m) -> None:
        uid = self._cur_study_uid or "—"
        self._measure_history.setdefault(uid, []).append(m)
        if self._hist_dialog is not None and self._hist_dialog.isVisible():
            self._refresh_history_dialog()

    def _remove_measurement(self, mid: int) -> None:
        """Drop the most-recent history entry for source-measure *mid* — used
        when a trace is 'resumed' (un-committed to keep extending it) so the
        eventual re-commit doesn't leave a duplicate. Scans newest-first and
        removes one match across the current study (falls back to any study)."""
        uid = self._cur_study_uid or "—"
        for key in (uid, *[k for k in self._measure_history if k != uid]):
            hist = self._measure_history.get(key, [])
            for i in range(len(hist) - 1, -1, -1):
                if getattr(hist[i], "mid", None) == mid:
                    del hist[i]
                    if (self._hist_dialog is not None
                            and self._hist_dialog.isVisible()):
                        self._refresh_history_dialog()
                    return

    def _show_history(self) -> None:
        if self._hist_dialog is None:
            self._hist_dialog = MeasureHistoryDialog(self)
        self._refresh_history_dialog()
        self._hist_dialog.show()
        self._hist_dialog.raise_()
        self._hist_dialog.activateWindow()

    def _refresh_history_dialog(self) -> None:
        uid = self._cur_study_uid or "—"
        hist = self._measure_history.get(uid, [])
        self._hist_dialog.set_entries(t("Study Measurement History"), hist)
