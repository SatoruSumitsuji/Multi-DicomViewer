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

from PyQt6.QtCore import QEvent, QMimeData, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QDrag,
    QDragEnterEvent,
    QDropEvent,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.config import (
    APP_NAME,
    APP_VERSION,
    BLOCK_CT,
    BLOCK_CT_MESSAGE,
)
from multi_dicomviewer.core import anonymize, dicom_io, settings
from multi_dicomviewer.core.dicom_tags import (
    default_overlay_keywords,
    upgrade_private_literal,
)
from multi_dicomviewer.core.study_model import Modality, Series
from multi_dicomviewer.ui.history_dialog import MeasureHistoryDialog
from multi_dicomviewer.ui.study_browser import (
    SERIES_MIME,
    FitButton,
    StudyPanel,
)
from multi_dicomviewer.ui.tag_dialog import TagSelectionDialog
from multi_dicomviewer.ui.tag_font import TAG_FONT_PT_DEFAULT
from multi_dicomviewer.viewers.ivus_viewer import IVUSViewer
from multi_dicomviewer.viewers.xa_viewer import XAViewer

try:
    # macOS renders CT with pygfx (wgpu→Metal): VTK's OpenGL→Metal path hangs.
    # Windows/Linux keep the proven VTK viewer. Both expose the same CTViewer
    # interface, so only the import target differs.
    if sys.platform == "darwin":
        from multi_dicomviewer.viewers.ct_viewer_pygfx import CTViewer
    else:
        from multi_dicomviewer.viewers.ct_viewer import CTViewer

    _CT_IMPORT_ERROR = ""
except Exception as exc:  # backend missing / broken — keep app usable for XA.
    CTViewer = None
    _CT_IMPORT_ERROR = str(exc)


def _ct_viewer():
    if CTViewer is None:
        hint = ("pip install -r requirements-mac.txt" if sys.platform == "darwin"
                else "pip install vtk")
        raise RuntimeError(
            "CT viewer unavailable — its render backend failed to import:\n"
            f"{_CT_IMPORT_ERROR}\n\n{hint}"
        )
    return CTViewer()


#: modality -> zero-arg factory building its viewer. Add OCT / NM here as
#: those viewer modules land; nothing else needs to change. OTHER (dose /
#: exposure summary pages, secondary captures, and not-yet-special-cased
#: kinds) falls back to the generic grayscale cine/image viewer so it is
#: still viewable rather than "unsupported".
_VIEWER_FACTORY = {
    Modality.XA: XAViewer,
    Modality.CT: _ct_viewer,
    Modality.IVUS: IVUSViewer,
    Modality.OTHER: XAViewer,
}

#: layout key -> (rows, cols, pane-count). Panes 0..count-1 are shown,
#: filling left-to-right, top-to-bottom.
_LAYOUTS = {
    "1x1": (1, 1, 1),
    "1x2": (1, 2, 2),
    "2x1": (2, 1, 2),
    "2x2": (2, 2, 4),
}
#: Multi-pane layouts (used to gate MultiSync, which needs 2+ panes).
_MULTI_PANE = ("1x2", "2x1", "2x2")
_MAX_PANES = 4

#: drag payload (source pane index) for swapping pane positions
PANE_MIME = "application/x-mdv-pane"


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


class _Placeholder(QWidget):
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet("color:#888; font-size:13px;")
        lay.addWidget(label)


class ViewerPane(QFrame):
    """One grid cell. Modality agnostic: it caches one viewer per modality
    and shows whichever the dropped/loaded series needs."""

    activated = pyqtSignal(object)            # this pane was clicked/used
    series_dropped = pyqtSignal(object, str)  # (pane, series_uid)
    folder_dropped = pyqtSignal(str)          # a DICOM folder was dropped
    files_dropped = pyqtSignal(list)          # individual DICOM file(s) dropped
    viewer_ready = pyqtSignal(object)         # a viewer was just created
    pane_move_requested = pyqtSignal(int, int)  # (src index, dest index)

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

        self._title = _DragTitle(self)
        self._title.setToolTip(
            "Drag and drop onto another pane to swap their positions"
        )
        self._title.setStyleSheet(
            "padding:2px 6px; color:#ccc; background:#2a2a2a;"
        )

        self._idle = _Placeholder(
            f"Pane {index + 1}\n\n"
            "Drag & drop a series from the Info panel,\n"
            "or a DICOM folder, here"
        )
        self._stack = QStackedWidget()
        self._stack.addWidget(self._idle)

        self._box = QVBoxLayout(self)
        self._box.setContentsMargins(1, 1, 1, 1)
        self._box.setSpacing(0)
        self._box.addWidget(self._title)
        self._box.addWidget(self._stack, 1)
        self._active_on = False
        self._full_bleed = False
        self._refresh_border()
        # Catch drops over the title bar / placeholder too (viewers added
        # later are covered in _viewer_for).
        self._install_dnd(self)

    # ---------------------------------------------------------- appearance
    def _idle_title(self) -> str:
        return f"● Pane {self.index + 1} — empty"

    def set_active(self, on: bool) -> None:
        self._active_on = on
        self._refresh_border()

    def set_full_bleed(self, on: bool) -> None:
        """1×1 mode: hide the title bar and drop the border/margins so the
        viewer fills the entire central frame."""
        self._full_bleed = on
        self._title.setVisible(not on)
        m = 0 if on else 1
        self._box.setContentsMargins(m, m, m, m)
        self._refresh_border()

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

    def show_series(self, loaded, title: str) -> None:
        viewer = self._viewer_for(loaded.modality)
        if viewer is None:
            mod = loaded.modality.value
            self._show_message(
                f"{mod} viewer is not implemented.\n"
                "(Supported: XA, CT, IVUS. OCT/OFDI/NM planned.)"
            )
            self._cur_viewer = None
            self._title.setText(
                f"● Pane {self.index + 1} — {mod} (unsupported)"
            )
            return
        viewer.load_series(loaded, title)
        self._stack.setCurrentWidget(viewer)
        self._cur_viewer = viewer
        self._title.setText(f"● Pane {self.index + 1} — {title}")

    def _show_message(self, text: str) -> None:
        # Reuse the idle widget's label to surface load/availability errors.
        self._idle.findChild(QLabel).setText(text)
        self._stack.setCurrentWidget(self._idle)

    def current_viewer(self):
        return self._cur_viewer

    def is_loaded(self, modality, series_uid: str) -> bool:
        """True if this pane already has *series_uid* loaded into the
        cached viewer for *modality* — lets the shell skip the whole
        disk-read + decode + viewer-rebuild pipeline when the user
        returns to a series they were already viewing."""
        v = self._viewers.get(modality)
        return v is not None and getattr(v, "_loaded_uid", "") == series_uid

    def switch_to_loaded(self, modality, title: str) -> None:
        """Bring the cached viewer for *modality* to the front without
        calling load_series. Caller must verify with is_loaded() first."""
        viewer = self._viewers.get(modality)
        if viewer is None:
            return
        self._stack.setCurrentWidget(viewer)
        self._cur_viewer = viewer
        self._title.setText(f"● Pane {self.index + 1} — {title}")

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
        self._idle.findChild(QLabel).setText(
            f"Pane {self.index + 1}\n\n"
            "Drag & drop a series from the Info panel,\n"
            "or a DICOM folder, here"
        )
        self._stack.setCurrentWidget(self._idle)
        self._title.setText(self._idle_title())

    # ------------------------------------------------------ click / drop
    def mousePressEvent(self, _event) -> None:
        self.activated.emit(self)

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
        # DICOM folder(s)/file(s) dropped onto the pane. A FOLDER drop loads
        # the whole folder; a FILE drop loads ONLY the dropped file(s) — not
        # the rest of their containing folder.
        paths = [u.toLocalFile() for u in md.urls()]
        paths = [p for p in paths if p]
        if paths:
            # Make this the active pane first so the first series auto-opens
            # here, not in some other pane.
            self.activated.emit(self)
            dirs = [p for p in paths if os.path.isdir(p)]
            if dirs:
                self.folder_dropped.emit(dirs[0])
            else:
                self.files_dropped.emit(paths)
            return True
        return False

    def _wants(self, md) -> bool:
        return (
            md.hasFormat(PANE_MIME)
            or md.hasFormat(SERIES_MIME)
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
        return super().eventFilter(obj, event)


class MainWindow(QMainWindow):
    def __init__(self, initial_folder: str | None = None):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}  v{APP_VERSION}")
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
        # Per-series MP4 export range [start, end] (0-based frame indices),
        # reported by a cine viewer's Play-range markers. A series only
        # appears here while its range is narrower than the full clip; a
        # full range is removed so the export defaults to every frame.
        self._mp4_ranges: dict[str, tuple[int, int]] = {}

        # --- study browser dock ---
        self.browser = StudyPanel()
        self.browser.series_chosen.connect(self._on_series_chosen)
        self.browser.study_clicked.connect(self._on_study_clicked)
        self.browser.delete_requested.connect(self._delete_node)
        self.browser.export_requested.connect(self._on_export_requested)
        dock = QDockWidget("Studies", self)
        dock.setWidget(self.browser)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self._studies_dock = dock

        # Widen the dock separator (the drag handle between the Studies
        # dock and the central viewer grid) — Qt's default is 4 px which
        # is hard to grab. Same rule for any future horizontally-docked
        # widget. Background tone matches the chrome so the bar still
        # reads as inert until the cursor hovers.
        self.setStyleSheet(
            "QMainWindow::separator {"
            " background:#a8a8a8; width:8px; height:8px;"
            "}"
            "QMainWindow::separator:hover {"
            " background:#4a90d9;"
            "}"
        )

        # --- configurable viewer grid ---
        self._layout_key = "1x1"
        # Bi / Lt / Rt is per-pane now (each viewer stores its own plane
        # choice); the toolbar buttons mirror whichever pane is active.
        #: Shared DICOM-tag overlay text size (pt) for every viewer/modality.
        self._tag_font_pt = TAG_FONT_PT_DEFAULT
        self._panes: list[ViewerPane] = []
        for i in range(_MAX_PANES):
            pane = ViewerPane(i)
            pane.activated.connect(self._set_active_pane)
            pane.series_dropped.connect(self._on_series_dropped)
            pane.folder_dropped.connect(self._load_folder)
            pane.files_dropped.connect(self._load_files)
            pane.viewer_ready.connect(self._wire_viewer)
            pane.pane_move_requested.connect(self._swap_panes)
            self._panes.append(pane)
        # Panes in grid-slot order (drag a title onto another to swap).
        self._order: list[ViewerPane] = list(self._panes)
        self._active = self._panes[0]

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(2)

        central = QWidget()
        col = QVBoxLayout(central)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(self._build_layout_bar())
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
        self.browser.dicom_info_toggled.connect(self._on_dicom_info_btn)

        self._build_menu()
        self._build_shortcuts()
        self._apply_layout(self._layout_key)
        self.statusBar().showMessage("Open a DICOM folder to begin.")

        if initial_folder and os.path.isdir(initial_folder):
            self._load_folder(initial_folder)

    # ------------------------------------------------------------------ menu
    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("&File")

        open_act = QAction("&Open DICOM folder…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._choose_folder)
        m.addAction(open_act)

        clear_act = QAction("&Clear viewers", self)
        clear_act.triggered.connect(self._clear_all)
        m.addAction(clear_act)

        m.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        m.addAction(quit_act)

        vm = self.menuBar().addMenu("&View")
        self._anon_act = QAction("&Anonymize", self)
        self._anon_act.setCheckable(True)
        self._anon_act.setShortcut("Ctrl+Shift+A")
        self._anon_act.setToolTip(
            "Mask patient/case info on all on-screen displays "
            "(files unchanged)"
        )
        self._anon_act.toggled.connect(self._set_anonymized)
        vm.addAction(self._anon_act)

        tags_act = QAction("DICOM tag overlay items…", self)
        tags_act.triggered.connect(self._tags_for_active_pane)
        vm.addAction(tags_act)

        self._hide_overlay_act = QAction("Hide DICOM overlay", self)
        self._hide_overlay_act.setCheckable(True)
        # Q is the quick single-key toggle (shown in the menu); Ctrl+H
        # kept so the old shortcut still works. (D used to do this — it
        # is now unbound on CT and serves the cine play/2x toggle on
        # XA/IVUS panes via _build_shortcuts below.)
        self._hide_overlay_act.setShortcuts(
            [QKeySequence("Q"), QKeySequence("Ctrl+H")]
        )
        self._hide_overlay_act.setToolTip(
            "Show/hide the DICOM tag text drawn on the image (Q)"
        )
        self._hide_overlay_act.toggled.connect(self._toggle_overlay_hidden)
        vm.addAction(self._hide_overlay_act)

        vm.addSeparator()
        exp_act = QAction("Export DICOM tag overlay settings…", self)
        exp_act.triggered.connect(self._export_tag_conditions)
        vm.addAction(exp_act)

        imp_act = QAction("Import DICOM tag overlay settings…", self)
        imp_act.triggered.connect(self._import_tag_conditions)
        vm.addAction(imp_act)

        tm = self.menuBar().addMenu("&Tools")
        self._sync_act = QAction("MultiSync IVUS viewer…", self)
        self._sync_act.setToolTip(
            "Open the panes' IVUS series in a synchronised viewer "
            "(only available in the 1×2 / 2×1 / 2×2 layout)"
        )
        self._sync_act.triggered.connect(self._open_multisync)
        tm.addAction(self._sync_act)
        self._rupture_act = QAction("Rupture-Predictor…", self)
        self._rupture_act.triggered.connect(self._open_rupture_predictor)
        tm.addAction(self._rupture_act)
        self._ortho_act = QAction("Orthogonal-View…", self)
        self._ortho_act.setToolTip(
            "Pick a vector on the active XA image and get the two C-arm "
            "angles whose view is orthogonal to it "
            "(available whenever the active pane shows an XA series "
            "with C-arm positioner angles)"
        )
        self._ortho_act.triggered.connect(self._open_orthogonal_view)
        tm.addAction(self._ortho_act)
        self._coaxial_act = QAction("Coaxial Eval…", self)
        self._coaxial_act.setToolTip(
            "Draw a labelled Line (GC / proxLAD / …) on the same vessel in "
            "2+ angio views, then compute the 3-D GC-to-vessel angle "
            "(available whenever a visible pane shows an XA series with "
            "C-arm positioner angles)"
        )
        self._coaxial_act.triggered.connect(self._open_coaxial_eval)
        tm.addAction(self._coaxial_act)
        self._sync_layout_gate()

    def _sync_layout_gate(self) -> None:
        """Gate Tools menu items by current layout / contents:
        MultiSync needs 1×2 or 2×2; Orthogonal-View needs at least one
        VISIBLE pane to hold an XA series with C-arm positioner angles
        (it works on every visible pane — angio panes are pickable,
        the rest open view-only)."""
        if hasattr(self, "_sync_act"):
            self._sync_act.setEnabled(self._layout_key in _MULTI_PANE)
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
        _, _, count = _LAYOUTS[self._layout_key]
        for pane in self._order[:count]:
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
                self, "MultiSync",
                "No IVUS series are loaded. Open a folder with IVUS "
                "pull-backs first.",
            )
            return
        count = _LAYOUTS[self._layout_key][2]
        # Pre-fill each slot from the matching pane (in display order);
        # non-IVUS panes leave their slot empty. We also hand over the
        # pane's current frame index AND its viewer instance so MultiSync
        # can (a) open the slot on the same frame the pane is showing and
        # (b) live-mirror frame changes both ways while the window is up.
        preset: list = []
        preset_frames: list = []
        preset_viewers: list = []
        for i in range(count):
            pane = self._order[i]
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
        self._multisync = MultiSyncWindow(
            ivus,
            layout_count=count,
            preset=preset,
            preset_frames=preset_frames,
            preset_viewers=preset_viewers,
            parent=self,
        )
        # Open maximized so the synchronised grid + sync editor have
        # room without the user resizing first.
        self._multisync.showMaximized()
        self._multisync.raise_()

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
                    f"Encoding IVUS frame {k + 1} / {burst_total}…"
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
            progress(burst_total, burst_total, "Frames ready.")
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
        if self._layout_key == "1x1":
            visible_panes = (
                [self._active] if self._active is not None else []
            )
        else:
            _, _, count = _LAYOUTS[self._layout_key]
            visible_panes = [p for p in self._order[:count] if p.isVisible()]

        panels = []
        for pane in visible_panes:
            v = pane.current_viewer()
            if v is None:
                continue
            pane_label = f"Pane {self._panes.index(pane) + 1}"
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
                self, "Orthogonal-View",
                "No visible pane shows an XA series with C-arm "
                "positioner angles.",
            )
            return

        # Title from the active pane's series if available.
        title = "—"
        if self._active is not None:
            uid = self._active.shown_series_uid()
            se = self._series_by_uid.get(uid) if uid else None
            if se is not None:
                title = se.label
        self._ortho_win = OrthogonalViewWindow(panels, title, parent=self)
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

        if self._layout_key == "1x1":
            visible_panes = (
                [self._active] if self._active is not None else []
            )
        else:
            _, _, count = _LAYOUTS[self._layout_key]
            visible_panes = [p for p in self._order[:count] if p.isVisible()]

        all_lines = []
        for pane in visible_panes:
            v = pane.current_viewer()
            if v is None or not _is_xa(v):
                continue
            all_lines.extend(self._coaxial_lines_from_xa_viewer(v))

        if not all_lines:
            QMessageBox.information(
                self, "Coaxial Eval",
                "No vessel-labelled lines found.\n\n"
                "Draw a Line on the same vessel in 2+ angio views, then "
                "right-click each line ▸ Vessel type ▸ pick GC / proxLAD / "
                "etc. before running Coaxial Eval.",
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
                    self, "Rupture-Predictor",
                    "アクティブなペインに表示中の画像がありません。")
                return
            self._rupture_win = RupturePredictorWindow(
                qimage=qimg, calib=calib, parent=self)
        else:
            self._rupture_win = RupturePredictorWindow(
                plane=plane, frame_index=frame_index, calib=calib, parent=self)
        self._rupture_win.showMaximized()
        self._rupture_win.raise_()
        parts = ["IVUS frame stepper" if plane is not None
                 else "displayed image"]
        if calib is not None:
            parts.append("DICOM calibration (CH/CV step skipped)")
        self.statusBar().showMessage(
            "Rupture-Predictor opened with " + " + ".join(parts) + ".")

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
                self, "Rupture-Predictor", f"Not found:\n{src}",
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
                "Preparing IVUS frames for Rupture-Predictor…",
                "Cancel", 0, 1, self,
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
            prog.setLabelText("Writing Rupture-Predictor session HTML…")
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
            QMessageBox.warning(self, "Rupture-Predictor", str(exc))
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
            parts.append(f"IVUS frame burst ({len(handoff['frames'])} frames)")
        elif "image" in handoff:
            parts.append("displayed image")
        if "hpxmm" in handoff:
            parts.append("DICOM calibration (CH/CV step skipped)")
        self.statusBar().showMessage(
            "Rupture-Predictor opened with " + " + ".join(parts) + "."
        )

    # --------------------------------------------------------- screen layout
    def _build_layout_bar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 3, 6, 3)

        self._info_btn = FitButton("◀ Info shown")
        self._info_btn.setHelpToolTip(
            "Show/hide the left Info window (study tree)"
        )
        self._info_btn.clicked.connect(self._toggle_info)
        row.addWidget(self._info_btn)
        row.addSpacing(12)

        row.addWidget(QLabel("Layout:"))

        self._layout_group = QButtonGroup(self)
        self._layout_group.setExclusive(True)
        for label, key in (
            ("1×1", "1x1"),
            ("1×2", "1x2"),
            ("2×1", "2x1"),
            ("2×2", "2x2"),
        ):
            btn = FitButton(label)
            btn.setCheckable(True)
            btn.setChecked(key == self._layout_key)
            btn.clicked.connect(lambda _c, k=key: self._apply_layout(k))
            self._layout_group.addButton(btn)
            row.addWidget(btn)

        # Bi/Lt/Rt lives inside each viewer's own "Plane:" bar now (per-pane),
        # so there is no global plane switch in this top bar anymore.
        row.addStretch(1)
        return bar

    def _toggle_info(self, *_a) -> None:
        vis = not self._studies_dock.isVisible()
        self._studies_dock.setVisible(vis)
        self._info_btn.setText("◀ Info shown" if vis else "Info hidden ▶")

    def _apply_layout(self, key: str) -> None:
        self._layout_key = key
        rows, cols, count = _LAYOUTS[key]

        # Detach every pane, then re-add the visible subset to the grid.
        for pane in self._panes:
            self._grid.removeWidget(pane)
            pane.setVisible(False)
        full = key == "1x1"
        self._grid.setSpacing(0 if full else 2)
        if full:
            # 1×1 shows the active pane WITHOUT disturbing self._order, so
            # returning to 1×2 / 2×2 restores the original Pane 1, 2, 3, 4
            # arrangement.
            shown = self._active if self._active is not None else self._order[0]
            self._grid.addWidget(shown, 0, 0)
            shown.setVisible(True)
            shown.set_full_bleed(True)
        else:
            for i in range(count):
                r, c = divmod(i, cols)
                self._grid.addWidget(self._order[i], r, c)
                self._order[i].setVisible(True)
                self._order[i].set_full_bleed(False)
        # Reset stretch on ALL grid lines (max 2×2): a leftover stretch on
        # an now-empty row/col from a bigger layout would otherwise still
        # reserve space, shrinking the panes when going back to 1×1/1×2.
        max_r, max_c = _LAYOUTS["2x2"][0], _LAYOUTS["2x2"][1]
        for r in range(max_r):
            self._grid.setRowStretch(r, 1 if r < rows else 0)
        for c in range(max_c):
            self._grid.setColumnStretch(c, 1 if c < cols else 0)

        if full:
            # 1×1: the shown pane is the only one visible — keep it active.
            shown = self._active if self._active is not None else self._order[0]
            self._set_active_pane(shown)
        elif self._active not in self._order[:count]:
            self._set_active_pane(self._order[0])
        else:
            self._set_active_pane(self._active)
        # Bi/Lt/Rt is per-pane (each viewer's own "Plane:" bar), so a grid
        # change never overrides any pane's plane choice — a pane left on
        # "Bi" keeps showing both planes here too (just smaller).
        self._sync_layout_gate()      # MultiSync menu = only in 1×2 / 2×2

    def _swap_panes(self, src_index: int, dest_index: int) -> None:
        """Swap two panes' grid slots (drag a pane title onto another)."""
        if src_index == dest_index:
            return
        src = self._panes[src_index]
        dest = self._panes[dest_index]
        i, j = self._order.index(src), self._order.index(dest)
        self._order[i], self._order[j] = self._order[j], self._order[i]
        self._apply_layout(self._layout_key)

    def _set_active_pane(self, pane: ViewerPane) -> None:
        self._active = pane
        for p in self._panes:
            p.set_active(p is pane and p.isVisible())
        self._sync_xa_shortcuts()

    # --------------------------------------------------- series navigation
    def _build_shortcuts(self) -> None:
        # Cine (XA/IVUS) keys. They are app-wide QShortcuts, so they
        # would otherwise swallow S / T / R / D / W / V before a focused
        # CT pane sees them — keep them enabled only while the active
        # pane is showing a cine viewer (see _sync_xa_shortcuts).
        self._xa_shortcuts = []
        for key, fn in (
            ("F", lambda: self._nav_xa("next")),
            ("A", lambda: self._nav_xa("prev")),
            ("Home", lambda: self._nav_xa("first")),
            ("End", lambda: self._nav_xa("last")),
            # Cine transport layout (user spec):
            #   T, R = step +1 frame   (R duplicates T for two-handed use)
            #   E    = step -1 frame
            #   D    = play / 2× toggle (cycles 1×→2×→1× on the second
            #          press, starts at 1× when stopped)
            #   S    = stop
            #   W    = ECG waveform show/hide
            #   V    = IVUS long-axis show/hide (IVUS only — no-op on XA)
            ("T", lambda: self._xa_step(+1)),
            ("R", lambda: self._xa_step(+1)),
            ("E", lambda: self._xa_step(-1)),
            ("D", lambda: self._xa_play_speed_toggle()),
            ("S", lambda: self._xa_stop()),
            ("W", lambda: self._xa_toggle_ecg()),
            ("V", lambda: self._ivus_toggle_long_axis()),
            ("Z", lambda: self._xa_zoom(True)),
            ("Shift+Z", lambda: self._xa_zoom(False)),
        ):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(fn)
            self._xa_shortcuts.append(sc)
        self._sync_xa_shortcuts()

    def _sync_xa_shortcuts(self) -> None:
        on = _is_cine(self._active.current_viewer())
        for sc in getattr(self, "_xa_shortcuts", []):
            sc.setEnabled(on)
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

    def _xa_toggle_ecg(self) -> None:
        """W = toggle the ECG waveform strip. The strip itself is built
        per series from the DICOM WaveformSequence when present; on a
        series without ECG the button just stays hidden with a status
        message."""
        v = self._xa()
        if v is None:
            return
        v.toggle_ecg()

    def _ivus_toggle_long_axis(self) -> None:
        """V = toggle the IVUS long-axis (longitudinal) view. No-op on
        XA — the long-axis only makes sense for an IVUS pull-back."""
        v = self._xa()
        if v is None or getattr(v, "handles_modality", "") != "IVUS":
            self.statusBar().showMessage(
                "Long-axis view is IVUS-only."
            )
            return
        v.toggle_long_axis()

    def _xa_zoom(self, zoom_in: bool) -> None:
        v = self._xa()
        if v is None:
            return
        (v.zoom_in if zoom_in else v.zoom_out)()

    def _nav_xa(self, where: str) -> None:
        # Step within the SAME modality as the cine pane (XA among XA,
        # IVUS among IVUS, NM among NM, …), inside that pane (don't
        # hijack a CT pane). The navigation key comes from the SERIES
        # currently shown — not the viewer's handles_modality, because
        # non-canonical modalities (NM/OCT/OFD) all fall back to the
        # XAViewer with handles_modality="XA" and would otherwise step
        # through XA series instead of their own kind.
        xp = self._xa_pane()
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
        # Prefer the per-kind last-opened series (so IVUS resumes from
        # its last view even if the pane has shown CT/XA since, and
        # NM does not lump together with other OTHER-bucket modalities).
        cur = (self._last_by_modality.get(mod) or cur_series)
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

    # ------------------------------------------------------------- data load
    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open DICOM folder")
        if folder:
            self._load_folder(folder)

    # ---------------------------------------------------- drag & drop a folder
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        # A FOLDER drop loads the whole folder; a FILE drop loads ONLY the
        # dropped file(s) — not the rest of their containing folder.
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        paths = [p for p in paths if p]
        if not paths:
            return
        dirs = [p for p in paths if os.path.isdir(p)]
        files = [p for p in paths if os.path.isfile(p)]
        if dirs:
            event.acceptProposedAction()
            self._load_folder(dirs[0])
        elif files:
            event.acceptProposedAction()
            self._load_files(files)

    def _load_folder(self, folder: str) -> None:
        self._load_index(
            f"Scanning {folder} …",
            lambda prog: dicom_io.scan_folder(folder, prog),
        )

    def _load_files(self, paths: list[str]) -> None:
        n = len(paths)
        self._load_index(
            f"Loading {n} file(s) …",
            lambda prog: dicom_io.index_files(paths, prog),
        )

    def _load_index(self, status_msg: str, scan_fn) -> None:
        """Run *scan_fn(progress)* under a modal progress dialog, then merge
        the resulting patients into the tree and auto-open a series. Shared by
        folder drops (scan_folder) and file drops (index_files)."""
        self.statusBar().showMessage(status_msg)
        # Bring the app to the front (drop often comes from Explorer).
        if self.isMinimized():
            self.showNormal()
        self.raise_()
        self.activateWindow()

        dlg = QProgressDialog("Loading DICOM…", None, 0, 0, self)
        dlg.setWindowTitle("Scanning")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(300)   # don't flash for tiny folders
        dlg.setAutoClose(True)
        dlg.setAutoReset(True)

        def _progress(done, total):
            if total and dlg.maximum() != total:
                dlg.setMaximum(total)
            dlg.setValue(done)
            QApplication.processEvents()

        try:
            new_patients = scan_fn(_progress)
        except Exception as exc:
            dlg.close()
            QMessageBox.critical(self, "Scan failed", str(exc))
            return
        dlg.setValue(dlg.maximum())   # ensure it closes
        dlg.close()
        # Series UIDs contributed by THIS folder — used to auto-open the
        # newly dropped study (not whatever sorts first overall).
        new_uids = {
            se.series_uid
            for p in new_patients.values()
            for st in p.studies.values()
            for se in st.series.values()
        }
        # Accumulate: a new folder adds its studies; previously loaded
        # ones stay in the info panel.
        dicom_io.merge_patients(self._patients, new_patients)

        self.browser.populate(self._patients)
        n_ser = self._reindex_series_maps()
        n_pat = len(self._patients)
        self.statusBar().showMessage(
            f"{n_pat} patient(s), {n_ser} series (total). "
            "⚹ marks patients with both CT and XA. "
            "Drag & drop a series onto a pane to display."
        )
        # Auto-open a series from the just-loaded folder into the active pane.
        # When the folder has CT, prefer the "main" CT series (see
        # _initial_ct_target); otherwise fall back to the first series in the
        # browser's display order.
        ordered = self.browser.ordered_series()
        new_series = [se for se in ordered if se.series_uid in new_uids]
        target = self._initial_ct_target(new_series)
        if target is None:
            target = next(
                (se for se in ordered if se.series_uid in new_uids),
                ordered[0] if ordered else None,
            )
        if target is not None:
            self.browser.select_series(target)

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
                "Export: could not identify the displayed series."
            )
            return
        if plane_path:
            import dataclasses
            target = dataclasses.replace(series, files=[plane_path])
        else:
            target = series
        self._on_export_requested(fmt, [target])

    def _on_export_requested(self, fmt: str, series_list: list) -> None:
        """Right-click ▸ Export (DICOM)/(MP4)/(CSV): show the filename-
        fields dialog, ask for an output folder, run the export with a
        live progress bar. Runs on the UI thread (simpler; MP4 of a few
        hundred frames is still seconds, not minutes). CSV writes one
        file per series listing the displayed DICOM-tag-overlay tags."""
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
                "Export cancelled — no filename fields were ticked."
            )
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, "Choose output folder for export"
        )
        if not out_dir:
            return

        title = {
            "dicom": "Exporting DICOM…",
            "mp4": "Exporting MP4…",
            "csv": "Exporting CSV…",
            "anon-dicom": "Exporting Anon DICOM…",
        }[fmt]
        prog = QProgressDialog(title, "Cancel", 0, 1, self)
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
                written = exporter.export_mp4(
                    series_list, out_dir, cfg.fields,
                    bitrate_mbps=cfg.bitrate_mbps,
                    fps_override=cfg.fps,
                    crf=cfg.crf,
                    frame_ranges=frame_ranges,
                    progress=_cb,
                )
        except RuntimeError as e:
            if cancelled["yes"]:
                self.statusBar().showMessage("Export cancelled.")
                prog.close()
                return
            prog.close()
            QMessageBox.critical(self, "Export failed", str(e))
            return
        except Exception as e:
            prog.close()
            QMessageBox.critical(self, "Export failed", str(e))
            return
        prog.close()
        unit = "folder" if fmt in ("dicom", "anon-dicom") else "file"
        self.statusBar().showMessage(
            f"Exported {len(written)} {unit}"
            f"{'s' if len(written) != 1 else ''} to {out_dir}"
        )

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
            f"Deleted ({len(removed)} series removed from the list)"
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
        last = self._last_by_study.get((study_uid, kind))
        if (last is not None and last.series_uid in self._series_by_uid
                and not getattr(last, "hidden", False)):
            self.browser.select_series(last)
            return
        # Never visited (or last-viewed is now hidden) → fall back to the
        # first VISIBLE series of this study node, in display order.
        for se in self.browser.ordered_series():
            if (self._study_by_series_uid.get(se.series_uid) == study_uid
                    and se.kind == kind
                    and not getattr(se, "hidden", False)):
                self.browser.select_series(se)
                return

    def _on_series_dropped(self, pane: ViewerPane, uid: str) -> None:
        series = self._series_by_uid.get(uid)
        if series is not None:
            self._open_series(series, pane)

    def _open_series(self, series: Series, pane: ViewerPane) -> None:
        # Mac build cannot render CT (VTK's OpenGL→Metal path hangs). Tell
        # the user explicitly and abort the load before any disk read /
        # viewer construction touches VTK.
        if BLOCK_CT and series.modality == Modality.CT:
            QMessageBox.information(self, "未対応のデータ", BLOCK_CT_MESSAGE)
            self.statusBar().showMessage(
                "CTデータはこのビルドでは読み込めません。"
            )
            return
        # Fast-path: this exact series is already loaded in this pane's
        # cached viewer (XAViewer / IVUSViewer / CTViewer track
        # _loaded_uid). Skip the whole disk read + decode + viewer
        # rebuild pipeline — and therefore skip the CT progress dialog
        # too — so returning to a series is instant and its frame /
        # camera / W-L / measurements all stay put.
        if pane.is_loaded(series.modality, series.series_uid):
            pane.switch_to_loaded(series.modality, series.label)
            pane.set_shown_series(series.series_uid)
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
            self.statusBar().showMessage(f"Resumed {series.label}")
            return
        self.statusBar().showMessage(f"Loading {series.label} …")
        # Cut the OUTGOING series's background prefetch BEFORE reading
        # any new files so disk / CPU bandwidth is freed for the new
        # load instead of contending with the old decode. Only on the
        # slow path (the fast-path above didn't reload anything, so its
        # prefetch should keep warming for the same series).
        cur_v = pane.current_viewer()
        if cur_v is not None and hasattr(cur_v, "_stop_prefetch"):
            cur_v._stop_prefetch()
        # CT load (read every slice + build the HU volume + VTK pipeline)
        # is the multi-second wait the user saw *after* the scan bar hit
        # 100%. Keep a real, phased progress dialog up through all of it
        # so the last stretch is never a frozen UI. XA/IVUS also gets a
        # dialog now — a 100-300 MB compressed cine clip can take
        # several seconds to read off disk and decode frame 0, and the
        # user previously had no way to tell the UI from a freeze.
        is_ct = series.modality == Modality.CT
        title = ("Loading CT" if is_ct
                 else f"Loading {series.kind or 'DICOM'}")
        initial_msg = ("Reading CT slices…" if is_ct
                       else "Reading DICOM file…")
        dlg = QProgressDialog(initial_msg, None, 0, 0, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        dlg.show()
        QApplication.processEvents()

        def _cb(phase: str, done: int, total: int) -> None:
            if dlg is None:
                return
            if dlg.labelText() != phase:
                dlg.setLabelText(phase)
            if total and dlg.maximum() != total:
                dlg.setMaximum(total)
            dlg.setValue(done)
            QApplication.processEvents()

        try:
            loaded = dicom_io.load_series(series, progress=_cb)
        except Exception as exc:
            if dlg is not None:
                dlg.close()
            traceback.print_exc()
            QMessageBox.critical(
                self, "Load failed", f"{series.label}\n\n{exc}"
            )
            self.statusBar().showMessage("Load failed.")
            return

        if dlg is not None:
            # The VTK pipeline build has no fine-grained progress, so show
            # an indeterminate "constructing" bar for that last phase.
            dlg.setLabelText("Constructing 3D view…")
            dlg.setMaximum(0)            # 0,0 -> busy/indeterminate bar
            dlg.setValue(0)
            QApplication.processEvents()

        self._cur_study_uid = self._study_by_series_uid.get(series.series_uid)
        pane.show_series(loaded, series.label)
        if dlg is not None:
            dlg.close()
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
        # Keep an open history window pointed at the now-current study.
        if self._hist_dialog is not None and self._hist_dialog.isVisible():
            self._refresh_history_dialog()
        self.statusBar().showMessage(f"Loaded {series.label}")

    def _clear_all(self) -> None:
        for pane in self._panes:
            pane.reset()
        self.statusBar().showMessage("Viewers cleared.")

    # ------------------------------------- per-viewer signal/option wiring
    def _wire_viewer(self, viewer) -> None:
        """Connect a freshly built viewer's signals and push current
        display options onto it."""
        # First/Prev/Next/Last cross-series nav: XA and IVUS cine viewers.
        if _is_cine(viewer) and hasattr(viewer, "series_nav"):
            viewer.series_nav.connect(self._nav_xa)
        # Measuring / history are modality-agnostic (XA and IVUS both).
        if hasattr(viewer, "measurement_added"):
            viewer.measurement_added.connect(self._record_measurement)
        if hasattr(viewer, "history_requested"):
            viewer.history_requested.connect(self._show_history)
        if hasattr(viewer, "tags_requested"):
            viewer.tags_requested.connect(
                lambda vv=viewer: self._open_tag_dialog(vv)
            )
        # Right-click ▸ Export DICOM/MP4/CSV on the image → reuse the same
        # series export the Studies-list right-click uses, scoped to the
        # clicked plane.
        if hasattr(viewer, "plane_export_requested"):
            viewer.plane_export_requested.connect(self._on_plane_export)
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
        """Broadcast the DICOM-tag overlay text size to every viewer in every
        pane so the size stays uniform across modalities."""
        pt = int(pt)
        self._tag_font_pt = pt
        for pane in self._panes:
            for v in pane.all_viewers():
                if hasattr(v, "set_overlay_font_pt"):
                    v.set_overlay_font_pt(pt)

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
            "DICOM overlay: hidden" if hidden else "DICOM overlay: shown"
        )

    def _toggle_overlay_hidden(self, on: bool) -> None:
        """View ▸ Hide DICOM overlay — also sync the toolbar button."""
        self._apply_overlay_hidden(on)
        self.browser.set_dicom_info_shown(not on)

    def _on_dicom_info_btn(self, show: bool) -> None:
        """'DICOM Info' button (checked = show) — also sync the menu."""
        self._apply_overlay_hidden(not show)
        self._hide_overlay_act.blockSignals(True)
        self._hide_overlay_act.setChecked(not show)
        self._hide_overlay_act.blockSignals(False)

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
        self.statusBar().showMessage(
            "Anonymize: ON (case info masked on screen)"
            if on
            else "Anonymize: OFF"
        )

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
            f"Anonymize settings updated: {len(tags)} tag(s)"
            + (" + private" if emptify_private else "")
        )

    def _fit_studies_dock_width(self, width: int) -> None:
        """Thumbnail "Fit: min × 10 across" → widen the Studies dock toward
        *width* px (capped so the viewers keep room). Best-effort: the grid
        wraps to whatever actually fits."""
        dock = self._studies_dock
        if not dock.isVisible():
            dock.setVisible(True)
            self._info_btn.setText("◀ Info shown")
        cap = max(300, self.width() - 360)   # leave ≥360px for the panes
        w = max(200, min(int(width), cap))
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
                "DICOM Tags",
                "Load a series first, then choose overlay items.",
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
            "Export DICOM tag overlay settings",
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
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        total = sum(
            len(v) for v in self._tag_keywords_by_modality.values()
        )
        self.statusBar().showMessage(
            f"Exported overlay settings ({total} items across modalities)"
        )

    def _import_tag_conditions(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import DICOM tag overlay settings",
            "",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            imported = settings.import_tag_keywords(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
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
            f"Imported overlay settings ({len(kws)} items)"
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
            "DICOM Tags",
            "Load a series first, then choose overlay items.",
        )

    # ----------------------------------------------- measurement history
    def _record_measurement(self, m) -> None:
        uid = self._cur_study_uid or "—"
        self._measure_history.setdefault(uid, []).append(m)
        if self._hist_dialog is not None and self._hist_dialog.isVisible():
            self._refresh_history_dialog()

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
        self._hist_dialog.set_entries("Study Measurement History", hist)
