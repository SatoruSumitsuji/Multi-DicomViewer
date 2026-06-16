"""Study browser: a Patient/Study/Series tree and a thumbnail grid, toggled.

Both views list the same series in the same order and emit series_chosen
on activation, so the shell does not care which one is showing.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QMimeData, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QDrag,
    QFont,
    QIcon,
    QImage,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QListView,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QHeaderView,
    QMenu,
    QLayout,
    QPushButton,
    QSlider,
    QStackedWidget,
    QStyle,
    QStyleOptionButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.core import dicom_io
from multi_dicomviewer.core.anonymize import ANON_PLACEHOLDER
from multi_dicomviewer.core.study_model import Patient, Series

_ROLE = Qt.ItemDataRole.UserRole
#: stable key on patient/study items so expand state survives a repopulate
_ID_ROLE = Qt.ItemDataRole.UserRole + 1

#: thumbnail size-slider bounds; previews are decoded at _THUMB_GEN_PX so
#: the grid stays sharp anywhere in this range.
_THUMB_MIN_PX = 60   # was 80; ~75% so more thumbnails fit for gap-scanning
_THUMB_MAX_PX = 280
_THUMB_GEN_PX = 280

#: Drag payload carrying a series_uid, dropped onto a viewer pane.
SERIES_MIME = "application/x-mdv-series-uid"


class FitButton(QPushButton):
    """A push button that stays readable when the toolbar is dragged narrow.

    When the full label fits, it is shown in full (centred, as normal) and
    the tooltip falls back to the optional help text. When the label is too
    wide, it is elided from the right — so the *start* of the label is always
    legible — and the tooltip exposes the complete text. Eliding (rather than
    setting ``text-align: left`` via a stylesheet) is used deliberately: a
    stylesheet would override the native checked/pressed appearance these
    buttons rely on, whereas re-eliding the text leaves the look untouched.
    """

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._full_text = text
        self._help_tip = ""

    def setHelpToolTip(self, tip: str) -> None:
        """Tooltip shown when the label fits (and appended after the full
        label when it is elided)."""
        self._help_tip = tip
        self._relayout_text()

    def setText(self, text: str) -> None:  # noqa: N802 (Qt override)
        self._full_text = text
        super().setText(text)
        self._relayout_text()

    def resizeEvent(self, e) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(e)
        self._relayout_text()

    def _relayout_text(self) -> None:
        fm = self.fontMetrics()
        # Width actually available for the LABEL — taken from the style's
        # content rect, NOT a guessed constant. This is correct per platform:
        # macOS native buttons pad/round wider than Windows, and a guessed
        # reserve was either too small (the "fitted" text overflowed and the
        # centred label clipped its leftmost char) or too big (over-elided to
        # just the icon at any width). The style rect makes the elided text
        # truly fit, so a wide button shows the full label and a narrow one
        # shows as many leading chars as fit.
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        content = self.style().subElementRect(
            QStyle.SubElement.SE_PushButtonContents, opt, self
        )
        avail = content.width() - 2          # tiny safety margin
        if avail <= 0:
            return
        if fm.horizontalAdvance(self._full_text) <= avail:
            if super().text() != self._full_text:
                super().setText(self._full_text)
            super().setToolTip(self._help_tip)
        else:
            elided = fm.elidedText(
                self._full_text, Qt.TextElideMode.ElideRight, avail
            )
            # At very narrow widths elidedText can collapse to just "…"/empty,
            # hiding exactly the leftmost char the user needs — force the first
            # character to remain.
            if self._full_text and elided.strip("… ") == "":
                elided = self._full_text[0]
            if super().text() != elided:
                super().setText(elided)
            tip = self._full_text
            if self._help_tip:
                tip = f"{self._full_text}\n\n{self._help_tip}"
            super().setToolTip(tip)


def _start_series_drag(source: QWidget, series: Series) -> None:
    """Begin dragging *series* out of the info panel onto a viewer pane.
    A hidden (greyed) series can still be dragged: like a direct mouse
    click, a drag is an explicit request to view it. Only the on-image
    First/Prev/Next/Last buttons and the keyboard shortcuts skip hidden
    series."""
    md = QMimeData()
    md.setData(SERIES_MIME, series.series_uid.encode("utf-8"))
    md.setText(series.label)
    drag = QDrag(source)
    drag.setMimeData(md)
    drag.exec(Qt.DropAction.CopyAction)


def _patient_label(patient: Patient, anon: bool) -> str:
    if anon:  # name + ID are case-identifying
        return f"{ANON_PLACEHOLDER}  ({ANON_PLACEHOLDER})"
    return patient.label


def _study_node_label(study, kind: str, anon: bool) -> str:
    """Label for one (study, modality-kind) node. A study with both XA
    and IVUS on the same date is shown as two separate nodes — one per
    kind — because they are clinically different acquisitions."""
    if anon:  # study date is case-identifying; modality kind is not
        return f"{ANON_PLACEHOLDER}  {kind}"
    date = study.date or "????-??-??"
    desc = study.description
    base = f"{date}  {kind}"
    return f"{base}  —  {desc}" if desc else base


def _thumb_label(se: Series, use_instance: bool) -> str:
    """Thumbnail caption — same shape as ``Series.label`` (two lines
    after the ``"  " -> "\\n"`` substitution) but the leading "#NUM"
    follows the tree's current sort. Sort by Series No → show Series
    Number; sort by Instance No → show DICOM InstanceNumber. Falls
    back to Series Number when instance_number is missing."""
    if use_instance and se.instance_number is not None:
        n = f"#{se.instance_number} "
    elif se.number is not None:
        n = f"#{se.number} "
    else:
        n = ""
    desc = f" — {se.description}" if se.description else ""
    label = f"{n}{se.kind}{desc}  [{se.image_count} img]"
    return label.replace("  ", "\n", 1)


def _series_sort_key(mode: str):
    if mode == "acq":  # Date/Time (empty sorts last; number breaks ties)
        return lambda s: (s.acq_time == "", s.acq_time, s.number or 0)
    if mode == "images":  # number of images (summed frames, matches display)
        return lambda s: (s.image_count, s.number or 0)
    if mode == "instance":  # DICOM InstanceNumber (None sorts last)
        return lambda s: (s.instance_number is None, s.instance_number or 0)
    return lambda s: (s.number is None, s.number or 0)  # Series Number


def _fmt_acq(acq: str) -> str:
    """'20260227114526.26' -> '2026-02-27 11:45:26'. Time-only inputs are
    shown as HH:MM:SS; empty -> '—'."""
    if not acq:
        return "—"
    s = str(acq)
    date = ""
    if len(s) >= 12 and s[:8].isdigit() and "19" <= s[:2] <= "21":
        date = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        s = s[8:]
    t = s
    if len(s) >= 6 and s[:6].isdigit():
        t = f"{s[:2]}:{s[2:4]}:{s[4:6]}"
    return f"{date} {t}".strip()


def iter_study_groups(patients: dict[str, Patient], sort_modes=None):
    """Yield (patient, study, kind, [Series…]) in browser display order:
    patient by name, study by date, modality kind alphabetical. Series
    within a (study, kind) node are ordered by *sort_modes*
    [(study_uid, kind)] = (mode, asc) — applied to THAT study only;
    others stay Series-Number ascending. Single source of truth so the
    tree and the thumbnail grid order series identically."""
    sort_modes = sort_modes or {}
    for patient in sorted(patients.values(), key=lambda p: p.name):
        for study in sorted(
            patient.studies.values(), key=lambda s: s.date
        ):
            by_kind: dict[str, list[Series]] = {}
            for se in study.series.values():
                by_kind.setdefault(se.kind, []).append(se)
            for kind in sorted(by_kind):
                mode, asc = sort_modes.get(
                    (study.study_uid, kind), ("number", True)
                )
                yield patient, study, kind, sorted(
                    by_kind[kind], key=_series_sort_key(mode),
                    reverse=not asc,
                )


def _group_header(patient: Patient, study, kind: str, anon: bool) -> str:
    # Two lines: patient (name / ID) on top, then date + kind only
    # (no study description) so the line stays short and readable.
    if anon:
        line2 = f"{ANON_PLACEHOLDER}  {kind}"
    else:
        line2 = f"{study.date or '????-??-??'}  {kind}"
    return f"{_patient_label(patient, anon)}\n{line2}"


class StudyBrowser(QTreeWidget):
    series_chosen = pyqtSignal(object)  # emits Series
    study_clicked = pyqtSignal(str, str)  # emits (study_uid, kind) — row
                                        # click on a Study header. The kind
                                        # MUST travel with the uid: one
                                        # study_uid can have two nodes (e.g.
                                        # XA + OT on the same date), and the
                                        # shell resolves to the LAST-viewed
                                        # series of THAT node, not series #1
                                        # and not the sibling kind's node.
    #: (kind, key, label) — kind is "patient"|"study"|"series";
    #: key is PatientID / StudyInstanceUID / SeriesInstanceUID.
    delete_requested = pyqtSignal(str, str, str)
    #: (fmt, [Series, …]) — fmt is "dicom" or "mp4". The shell pops up
    #: the export dialog (filename fields, MP4 bitrate/fps), asks for
    #: an output folder, and runs the export off the UI thread.
    export_requested = pyqtSignal(str, list)
    #: "Hide" → grey + skip these series ([Series]).
    hide_requested = pyqtSignal(list)
    #: Study "UnHide (unhide and show all)" → un-hide every hidden series in
    #: a study node (study_uid, kind).
    unhide_study_requested = pyqtSignal(str, str)
    #: Series "UnHide (show this series)" → un-hide just the given series
    #: ([Series]).
    unhide_series_requested = pyqtSignal(list)

    #: Selected-row highlight: bright blue for a normal series (stays
    #: bright even when the tree loses focus — Qt would otherwise dim it
    #: to a hard-to-read pale grey), grey for a hidden series so the row
    #: still reads as "greyed/skip" while selected.
    _SEL_BG_NORMAL = "#1f6feb"
    _SEL_BG_HIDDEN = "#a8a8a8"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(7)
        self.setHeaderLabels([
            "Date/Time",
            "Series No",
            "Instance No",
            "Type",
            "Images",
            "Description",
            "File Path",
        ])
        hdr = self.header()
        hdr.setStretchLastSection(False)
        # Draggable section dividers; widths persist across repopulate.
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setMinimumSectionSize(56)
        # Col 0 is the tree column (holds the Patient/Study hierarchy too),
        # so it needs to be wide enough not to crowd "Series No".
        for col, w in enumerate((210, 90, 70, 70, 80, 240, 600)):
            self.setColumnWidth(col, w)
        # "Type" (col 3 = series.kind) is hidden: it's redundant with the
        # modality kind already shown on each Study group row, and not useful
        # per-series in the list. Column data is still populated (kept simple),
        # just not displayed.
        self.setColumnHidden(3, True)
        # Click Date/Time / Series No / Instance No / Images headers to
        # sort (a ▲/▼ indicator shows the active column). Type /
        # Description / File Path are not sortable. We rebuild the
        # (grouped) tree ourselves, so Qt's own item sorting stays off.
        hdr.setSortIndicatorShown(True)
        hdr.setSectionsClickable(True)
        hdr.sectionClicked.connect(self._on_header_clicked)
        self._sort_cols = {
            0: "acq", 1: "number", 2: "instance", 4: "images",
        }
        # Show full text and let a horizontal scrollbar reach the right.
        self.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        # Extended selection enables Ctrl+Click toggle, Shift+Click range,
        # Ctrl+A select-all — the standard Windows multi-select gestures.
        self.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.customContextMenuRequested.connect(self._context_menu)
        self.itemActivated.connect(self._on_activated)
        self.itemClicked.connect(self._on_clicked)
        # The selected-row highlight follows the current series' hidden
        # state: blue for a normal series, grey for a hidden one (so a
        # hidden row still reads as "greyed" even while selected, instead
        # of the strong blue swamping the grey text).
        self.currentItemChanged.connect(
            lambda *_: self._refresh_selection_color()
        )
        self._refresh_selection_color()
        self._ordered: list[Series] = []
        self._items: dict[int, QTreeWidgetItem] = {}
        #: (study_uid, kind) -> study QTreeWidgetItem
        self._study_items: dict[tuple, QTreeWidgetItem] = {}
        self._patients: dict[str, Patient] = {}
        self._anon = False
        #: per (study_uid, kind) -> (mode, asc). Sorting affects only the
        #: currently-selected Study; others stay Series-No ascending.
        self._sort_modes: dict[tuple, tuple] = {}
        hdr.setSortIndicator(1, Qt.SortOrder.AscendingOrder)

    def _on_header_clicked(self, col: int) -> None:
        key = getattr(self, "_sort_cols", {}).get(col)
        if key is None:                       # Type/Description/Path
            return
        cur = self.current_study_key()        # the selected Study only
        if cur is None:
            return
        mode, asc = self._sort_modes.get(cur, ("number", True))
        asc = not asc if key == mode else True
        self._sort_modes[cur] = (key, asc)
        # Emits sortIndicatorChanged -> StudyPanel rebuilds tree+thumbs
        # and re-selects this Study to keep context.
        self.header().setSortIndicator(
            col,
            Qt.SortOrder.AscendingOrder if asc
            else Qt.SortOrder.DescendingOrder,
        )

    def set_anonymized(self, on: bool) -> None:
        """Mask/unmask patient & study identifiers; keeps the same tree."""
        on = bool(on)
        if on == self._anon:
            return
        self._anon = on
        if self._patients:
            self.populate(self._patients)

    def _expansion_snapshot(self) -> dict:
        """Map each patient/study item's stable key -> expanded?, so a
        repopulate (anonymize toggle, or a newly dropped folder being
        merged in) can keep the user's open/closed state instead of
        forcing everything open."""
        snap: dict = {}
        for i in range(self.topLevelItemCount()):
            p = self.topLevelItem(i)
            kp = p.data(0, _ID_ROLE)
            if kp is not None:
                snap[kp] = p.isExpanded()
            for j in range(p.childCount()):
                s = p.child(j)
                ks = s.data(0, _ID_ROLE)
                if ks is not None:
                    snap[ks] = s.isExpanded()
        return snap

    def populate(self, patients: dict[str, Patient]) -> None:
        prev = self._expansion_snapshot()
        # Default depth = Study level: patients expanded (studies visible),
        # studies collapsed (series hidden until the user clicks the ▸
        # expander). The user's own open/closed choices are preserved
        # across a repopulate via the snapshot.
        self.clear()
        self._patients = patients
        self._ordered = []
        self._items = {}
        self._study_items = {}
        cur_patient = None
        p_item = None
        # One study node per (study, modality kind): XA and IVUS on the
        # same date appear as separate Study entries.
        for patient, study, kind, series_list in iter_study_groups(
            patients, self._sort_modes
        ):
            if patient is not cur_patient:
                cur_patient = patient
                tag = "  ⚹ CT+XA" if patient.has_both_modalities() else ""
                p_item = QTreeWidgetItem(
                    [_patient_label(patient, self._anon) + tag]
                )
                pk = ("P", patient.patient_id)
                p_item.setData(0, _ID_ROLE, pk)
                self.addTopLevelItem(p_item)
                p_item.setExpanded(prev.get(pk, True))
            s_item = QTreeWidgetItem(
                [_study_node_label(study, kind, self._anon)]
            )
            sk = ("S", study.study_uid, kind)
            s_item.setData(0, _ID_ROLE, sk)
            p_item.addChild(s_item)
            self._study_items[(study.study_uid, kind)] = s_item
            for series in series_list:
                acq = (
                    ANON_PLACEHOLDER  # date/time is case-identifying
                    if self._anon
                    else _fmt_acq(series.acq_time)
                )
                no = f"#{series.number}" if series.number is not None else ""
                ino = (
                    f"#{series.instance_number}"
                    if series.instance_number is not None else ""
                )
                se_item = QTreeWidgetItem([
                    acq,
                    no,
                    ino,
                    series.kind,
                    f"{series.image_count} img",
                    series.description or "",
                    " ; ".join(series.files),
                ])
                se_item.setData(0, _ROLE, series)
                s_item.addChild(se_item)
                self._ordered.append(series)
                self._items[id(series)] = se_item
            s_item.setExpanded(prev.get(sk, False))
        self._apply_hidden_style()      # grey any series marked hidden

    def _apply_hidden_style(self) -> None:
        """Grey every hidden series row (and restore normal colour for the
        rest). Re-applied after each populate so the state survives the
        anonymize-toggle / re-sort repopulates."""
        grey = QColor("#888")
        cols = self.columnCount()
        for s in self._ordered:
            it = self._items.get(id(s))
            if it is None:
                continue
            brush = QBrush(grey) if getattr(s, "hidden", False) else QBrush()
            for c in range(cols):
                it.setForeground(c, brush)
        # The current row may have just changed hidden state (Hide/UnHide).
        self._refresh_selection_color()

    def _refresh_selection_color(self) -> None:
        """Set the selected-row highlight colour from the current item:
        grey when it is a hidden series, blue otherwise. Clicking an
        un-hidden series gives a blue background, a hidden series a grey
        one; the moment a selected series is Hidden, its highlight flips
        to grey so the greyed text is no longer swamped by the blue."""
        it = self.currentItem()
        data = it.data(0, _ROLE) if it is not None else None
        hidden = isinstance(data, Series) and getattr(data, "hidden", False)
        bg = self._SEL_BG_HIDDEN if hidden else self._SEL_BG_NORMAL
        self.setStyleSheet(
            "QTreeWidget::item:selected,"
            " QTreeWidget::item:selected:!active {"
            f"   background:{bg}; color:white;"
            " }"
        )

    def series_in_study(self, study_uid: str, kind: str) -> list[Series]:
        """All series under the (study_uid, kind) node, in display order."""
        out: list[Series] = []
        for s in self._ordered:
            it = self._items.get(id(s))
            par = it.parent() if it is not None else None
            idk = par.data(0, _ID_ROLE) if par is not None else None
            if idk and idk[0] == "S" and idk[1] == study_uid and idk[2] == kind:
                out.append(s)
        return out

    def set_series_hidden(self, series_list, hidden: bool) -> None:
        """Mark *series_list* hidden / shown and re-grey the tree."""
        for s in series_list:
            s.hidden = bool(hidden)
        self._apply_hidden_style()

    def _set_study_rows_visible(
        self, study_uid: str, kind: str, *, unhidden_only: bool
    ) -> None:
        """Show / fold series ROWS under a study node. With *unhidden_only*,
        hidden (greyed) series rows are folded out and only the un-hidden
        ones remain visible; otherwise every series row is shown. This only
        changes which rows appear in the tree — it does NOT touch the
        series' hide/unhide flag, and resets to "show all" on the next
        repopulate (anonymize toggle / re-sort)."""
        s_item = self._study_items.get((study_uid, kind))
        if s_item is None:
            return
        for se in self.series_in_study(study_uid, kind):
            it = self._items.get(id(se))
            if it is None:
                continue
            it.setHidden(unhidden_only and getattr(se, "hidden", False))
        s_item.setExpanded(True)
        self.setCurrentItem(s_item)

    def _unhide_study_show_all(self, study_uid: str, kind: str) -> None:
        """Study "UnHide": clear every hidden flag in the node (routed
        through the panel so the thumbnail grid re-greys too), then show
        every series row and expand the node."""
        self.unhide_study_requested.emit(study_uid, kind)
        self._set_study_rows_visible(study_uid, kind, unhidden_only=False)

    def ordered_series(self, modality: str | None = None) -> list[Series]:
        """All series in tree-display order. ``modality`` filters by the
        raw DICOM Modality string (``series.kind`` — "XA"/"CT"/"IVUS"/
        "NM"/…) so non-canonical modalities (NM/OCT/OFD/…) that fall
        into the Modality.OTHER bucket still navigate within their own
        kind instead of pooling together as "OTHER"."""
        if modality is None:
            return list(self._ordered)
        return [s for s in self._ordered if s.kind == modality]

    def current_study_key(self) -> tuple | None:
        """(study_uid, kind) of the selected node — a Study node itself,
        or the Study a selected Series belongs to. None otherwise."""
        it = self.currentItem()
        if it is None:
            return None
        if isinstance(it.data(0, _ROLE), Series):
            it = it.parent()
        idk = it.data(0, _ID_ROLE) if it is not None else None
        if idk and idk[0] == "S":
            return (idk[1], idk[2])
        return None

    def select_study_key(self, key: tuple) -> None:
        """Re-select a Study node by (study_uid, kind) without loading
        anything (used to keep context after a re-sort)."""
        it = self._study_items.get(key)
        if it is not None:
            self.setCurrentItem(it)

    def reselect_series_after_sort(self, series: Series) -> None:
        """Re-highlight *series* after a re-sort repopulate, keeping the
        study list OPEN and the row in view. Used so a header-sort doesn't
        drop the user's series selection onto the parent Study node (which
        looked like the list collapsing). setCurrentItem fires the panel's
        selection handler — which fixes the thumbnail grid — but NOT
        series_chosen (that only fires on a real click), so no reload."""
        item = self._items.get(id(series))
        if item is None:
            return
        study_item = item.parent()
        patient_item = (
            study_item.parent() if study_item is not None else None
        )
        if patient_item is not None:
            patient_item.setExpanded(True)
        if study_item is not None:
            study_item.setExpanded(True)        # keep the series list open
        self.setCurrentItem(item)
        self.scrollToItem(item)

    def select_series(self, series: Series) -> None:
        item = self._items.get(id(series))
        if item is not None:
            # Keep the default Study-level view: make the patient (and
            # thus the Study node) visible, but DON'T auto-expand the
            # study — series stay hidden until the user clicks ▸.
            study_item = item.parent()
            patient_item = (
                study_item.parent() if study_item is not None else None
            )
            if patient_item is not None:
                patient_item.setExpanded(True)
            # Setting the current item on a series under a collapsed study
            # makes the view auto-scroll and expand the study (revealing
            # all series). Keep the Study-level view: when collapsed,
            # highlight the Study node instead and leave series hidden.
            if study_item is not None and not study_item.isExpanded():
                self.setCurrentItem(study_item)
            else:
                self.setCurrentItem(item)
                self.scrollToItem(item)
            self.series_chosen.emit(series)

    def highlight_series(self, series: Series) -> None:
        """Visually select *series* in the tree WITHOUT emitting the
        viewer-load signal — used by Tree↔Thumbnail toggle sync where
        we want the highlight to follow but the displayed viewer to
        stay put."""
        item = self._items.get(id(series))
        if item is None:
            return
        study_item = item.parent()
        patient_item = (
            study_item.parent() if study_item is not None else None
        )
        if patient_item is not None:
            patient_item.setExpanded(True)
        if study_item is not None and not study_item.isExpanded():
            target = study_item
        else:
            target = item
        # Block both itemSelectionChanged + series_chosen so the
        # silent highlight doesn't trigger thumb rebuilds / viewer
        # loads.
        self.blockSignals(True)
        try:
            self.clearSelection()
            target.setSelected(True)
            self.setCurrentItem(target)
            if target is item:
                self.scrollToItem(item)
        finally:
            self.blockSignals(False)

    def _on_clicked(self, item: QTreeWidgetItem, _col: int = 0) -> None:
        """Mouse click in the tree. A hidden (greyed) series IS loaded on a
        direct mouse click — a one-off view: once shown, Play and the cine
        seek-bar work on it normally. The on-image First/Prev/Next/Last
        buttons and the keyboard shortcuts still skip it, and keyboard
        activation (Enter, via :meth:`_on_activated`) skips it too."""
        data = item.data(0, _ROLE)
        if isinstance(data, Series):
            self.series_chosen.emit(data)
            return
        self._emit_study_click(item)

    def _on_activated(self, item: QTreeWidgetItem, _col: int = 0) -> None:
        """Keyboard activation (Enter). Hidden series are skipped here — only
        a direct mouse click (see :meth:`_on_clicked`) loads a hidden one."""
        data = item.data(0, _ROLE)
        if isinstance(data, Series):
            if getattr(data, "hidden", False):
                return            # hidden series: keyboard does not load
            self.series_chosen.emit(data)
            return
        self._emit_study_click(item)

    def _emit_study_click(self, item: QTreeWidgetItem) -> None:
        # Clicking a Study row: tell the shell which study was picked
        # (by uid) so it can resume the LAST-viewed series of that study
        # instead of always jumping to series #1. The shell falls back
        # to the first series in browser order if nothing is remembered.
        idk = item.data(0, _ID_ROLE)
        if idk and idk[0] == "S" and item.childCount() > 0:
            study_uid, kind = idk[1], idk[2]
            self.study_clicked.emit(study_uid, kind)

    def _selected_series(self) -> list[Series]:
        """All Series across the current multi-selection, deduped by UID,
        in tree order. Empty when nothing series-like is selected."""
        out: list[Series] = []
        seen: set[str] = set()
        for it in self.selectedItems():
            data = it.data(0, _ROLE)
            if isinstance(data, Series) and data.series_uid not in seen:
                out.append(data)
                seen.add(data.series_uid)
        return out

    def _context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is None:
            return
        # If the right-click landed on an unselected item, include it in
        # the action so a quick right-click "just works" without a prior
        # left-click. This matches standard Windows Explorer behaviour.
        if not item.isSelected():
            self.clearSelection()
            item.setSelected(True)
            self.setCurrentItem(item)
        data = item.data(0, _ROLE)
        idk = item.data(0, _ID_ROLE)
        if isinstance(data, Series):
            kind, key, label = "series", data.series_uid, data.label
        elif idk and idk[0] == "P":
            kind, key, label = "patient", idk[1], item.text(0)
        elif idk and idk[0] == "S":
            # Scope deletion to this (study, modality-kind) node only, so
            # deleting the IVUS node leaves the same study's XA node.
            kind, key, label = "study", f"{idk[1]}\x1f{idk[2]}", item.text(0)
        else:
            return
        menu = QMenu(self)
        sel_series = self._selected_series()
        # Export actions: visible whenever at least one Series is in the
        # selection (right-click on a series row, or on any row while
        # several series are multi-selected).
        if sel_series:
            n = len(sel_series)
            suffix = f" ({n} series)" if n > 1 else ""
            act_dcm = QAction(f"Export (DICOM){suffix}", self)
            act_dcm.setToolTip(
                "Copy the original .dcm files to a chosen folder; one "
                "subfolder per series; per-file names from the checked "
                "filename fields. Lossless — no re-encode."
            )
            act_dcm.triggered.connect(
                lambda: self.export_requested.emit("dicom", list(sel_series))
            )
            menu.addAction(act_dcm)
            act_anon = QAction(f"Export (Anon DICOM){suffix}", self)
            act_anon.setToolTip(
                "Like Export (DICOM) but de-identified: the configured tags' "
                "values and all private tags are emptied (right-click the "
                "Anonymous button to choose). Pixels/UIDs kept."
            )
            act_anon.triggered.connect(
                lambda: self.export_requested.emit(
                    "anon-dicom", list(sel_series))
            )
            menu.addAction(act_anon)
            act_mp4 = QAction(f"Export (MP4){suffix}", self)
            act_mp4.setToolTip(
                "Render each selected series to an .mp4 in a chosen "
                "folder. Bitrate (Mbps) and FPS configurable."
            )
            act_mp4.triggered.connect(
                lambda: self.export_requested.emit("mp4", list(sel_series))
            )
            menu.addAction(act_mp4)
            act_csv = QAction(f"Export (CSV){suffix}", self)
            act_csv.setToolTip(
                "Write the DICOM-Tag-overlay tags shown for each series to "
                "a .csv (Tag Name, Tag Number, Value); one file per series. "
                "Full values — not truncated."
            )
            act_csv.triggered.connect(
                lambda: self.export_requested.emit("csv", list(sel_series))
            )
            menu.addAction(act_csv)
            # Hide / UnHide — between the Export group and Close.
            menu.addSeparator()
            act_hide = QAction(f"Hide (hide this series){suffix}", self)
            act_hide.setToolTip(
                "Grey out this series — kept in the list. First/Prev/Next/"
                "Last and the keyboard shortcuts skip it; a direct mouse "
                "click still shows it (Play / seek then work normally)."
            )
            act_hide.triggered.connect(
                lambda: self.hide_requested.emit(list(sel_series))
            )
            menu.addAction(act_hide)
            act_unhide = QAction(f"UnHide (show this series){suffix}", self)
            act_unhide.setToolTip(
                "Un-hide just the selected series (the rest of the study "
                "keeps its current hide/unhide state)."
            )
            act_unhide.triggered.connect(
                lambda: self.unhide_series_requested.emit(list(sel_series))
            )
            menu.addAction(act_unhide)
        menu.addSeparator()
        if kind == "study":
            su, sk_ = idk[1], idk[2]
            # Select — show only the un-hidden series rows. Hidden (greyed)
            # series are folded out of the list; their hide state is kept.
            act_select = QAction("Select (show unhide only)", self)
            act_select.setToolTip(
                "Show only the un-hidden series in this study; hidden "
                "(greyed) series rows are folded out of the list. Nothing "
                "is removed or un-hidden."
            )
            act_select.triggered.connect(
                lambda: self._set_study_rows_visible(su, sk_, unhidden_only=True)
            )
            menu.addAction(act_select)
            # Show — show every series row, keeping each series' hide state.
            act_show = QAction("Show (show all with hide/unhide)", self)
            act_show.setToolTip(
                "Show every series in this study, keeping each series' "
                "current hide/unhide (greyed) state."
            )
            act_show.triggered.connect(
                lambda: self._set_study_rows_visible(
                    su, sk_, unhidden_only=False)
            )
            menu.addAction(act_show)
            # UnHide — clear every hidden flag and show all series.
            act_unhide = QAction("UnHide (unhide and show all)", self)
            act_unhide.setToolTip(
                "Un-hide every hidden series in this study and show them all."
            )
            act_unhide.triggered.connect(
                lambda: self._unhide_study_show_all(su, sk_)
            )
            menu.addAction(act_unhide)
            # Close — collapse the series list (hide/unhide state kept).
            act_close = QAction("Close (close list)", self)
            act_close.setToolTip(
                "Collapse this study's series list. Hide/unhide state is "
                "kept; nothing is removed."
            )
            act_close.triggered.connect(lambda: self._close_series_list(item))
            menu.addAction(act_close)
            # Delete — remove the whole study node from the list.
            act_del = QAction("Delete (remove)", self)
            act_del.setToolTip(
                "Remove this study (all its series) from the list. Files "
                "are not deleted; reload the folder to restore."
            )
            act_del.triggered.connect(
                lambda: self.delete_requested.emit(kind, key, label)
            )
            menu.addAction(act_del)
        else:
            act_close = QAction("Close (close series list)", self)
            act_close.setToolTip(
                "Collapse this study's series list so the Study list is "
                "easier to browse. Nothing is removed."
            )
            act_close.triggered.connect(lambda: self._close_series_list(item))
            menu.addAction(act_close)
            menu.addSeparator()
            act = QAction("Delete (remove from list)", self)
            act.setToolTip(
                "Files are not deleted; just removed from the list so they "
                "can no longer be viewed."
            )
            act.triggered.connect(
                lambda: self.delete_requested.emit(kind, key, label)
            )
            menu.addAction(act)
        menu.exec(self.viewport().mapToGlobal(pos))

    def _close_series_list(self, item: QTreeWidgetItem) -> None:
        """Collapse the study node whose series list contains *item*,
        folding the series away so the Study-level list is easy to browse.
        Right-click on a series (or its study) collapses that study; on a
        patient, all its studies. Nothing is removed from the list."""
        if item is None:
            return
        data = item.data(0, _ROLE)
        idk = item.data(0, _ID_ROLE)
        if isinstance(data, Series):
            s_node = item.parent()
            if s_node is not None:
                s_node.setExpanded(False)
                self.setCurrentItem(s_node)
        elif idk and idk[0] == "S":
            item.setExpanded(False)
            self.setCurrentItem(item)
        elif idk and idk[0] == "P":
            for j in range(item.childCount()):
                item.child(j).setExpanded(False)

    def startDrag(self, _supported) -> None:  # noqa: N802 (Qt override)
        items = self.selectedItems() or (
            [self.currentItem()] if self.currentItem() else []
        )
        for it in items:
            se = it.data(0, _ROLE)
            if isinstance(se, Series):
                _start_series_drag(self, se)
                return


class _ThumbWorker(QThread):
    """Decodes one preview per series off the UI thread."""

    ready = pyqtSignal(int, object)  # (row, uint8 HxW ndarray)

    def __init__(self, series: list[Series], parent=None):
        super().__init__(parent)
        self._series = series
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        for row, se in enumerate(self._series):
            if self._stop:
                return
            try:
                self.ready.emit(
                    row, dicom_io.thumbnail(se, _THUMB_GEN_PX)
                )
            except Exception:
                continue  # unreadable series just keeps its placeholder


class _ThumbList(QListWidget):
    """Thumbnail grid that groups series under full-width section headers
    (patient / study-date / kind) and is a drag source for viewer panes.

    Right-clicking a thumbnail exposes the same Export (DICOM) / Export
    (MP4) / Delete-from-list actions the Tree view offers; the signals
    are forwarded by :class:`StudyPanel` so the shell handles both views
    identically."""

    _HEADER_H = 44  # two lines: patient, then study info

    #: ("dicom"|"mp4", [Series])
    export_requested = pyqtSignal(str, list)
    #: (kind, key, label) — same shape the Tree's signal uses.
    delete_requested = pyqtSignal(str, str, str)
    #: "Hide" the given series ([Series]).
    hide_requested = pyqtSignal(list)
    #: "UnHide (show this series)" — un-hide just the given series ([Series]).
    unhide_requested = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._header_items: list[QListWidgetItem] = []
        self.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.customContextMenuRequested.connect(self._context_menu)

    def add_header(self, text: str) -> QListWidgetItem:
        it = QListWidgetItem(text)
        it.setFlags(Qt.ItemFlag.NoItemFlags)  # not selectable / draggable
        it.setBackground(QColor("#2a2a2a"))
        it.setForeground(QColor("#cfe8ff"))
        f = QFont()
        f.setBold(True)
        it.setFont(f)
        it.setTextAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        it.setSizeHint(QSize(self._row_width(), self._HEADER_H))
        self.addItem(it)
        self._header_items.append(it)
        return it

    def clear(self) -> None:  # noqa: D102 (Qt override)
        self._header_items = []
        super().clear()

    def _row_width(self) -> int:
        return max(60, self.viewport().width() - 4)

    def _fit_headers(self) -> None:
        w = self._row_width()
        for it in self._header_items:
            it.setSizeHint(QSize(w, self._HEADER_H))

    def resizeEvent(self, e) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(e)
        self._fit_headers()

    def startDrag(self, _supported) -> None:  # noqa: N802 (Qt override)
        it = (self.selectedItems() or [self.currentItem()])[0] \
            if (self.selectedItems() or self.currentItem()) else None
        if it is None:
            return
        se = it.data(_ROLE)
        if isinstance(se, Series):
            _start_series_drag(self, se)

    # ------------------------------------------------ context menu
    def _selected_series(self) -> list[Series]:
        """All Series in the current multi-selection, deduped by UID,
        in display order. Header items are skipped."""
        out: list[Series] = []
        seen: set[str] = set()
        for it in self.selectedItems():
            se = it.data(_ROLE)
            if isinstance(se, Series) and se.series_uid not in seen:
                out.append(se)
                seen.add(se.series_uid)
        return out

    def _context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        se = item.data(_ROLE) if item is not None else None
        on_series = isinstance(se, Series)
        menu = QMenu(self)

        if on_series:
            # Mirror the Tree's behaviour: a right-click on an unselected
            # thumbnail picks JUST that one so the action targets the row
            # the user just pointed at (Windows-Explorer style).
            if not item.isSelected():
                self.clearSelection()
                item.setSelected(True)
                self.setCurrentItem(item)
            sel = self._selected_series() or [se]
            n = len(sel)
            suffix = f" ({n} series)" if n > 1 else ""
            act_dcm = QAction(f"Export (DICOM){suffix}", self)
            act_dcm.setToolTip(
                "Copy the original .dcm files to a chosen folder; one "
                "subfolder per series; per-file names from the checked "
                "filename fields. Lossless — no re-encode."
            )
            act_dcm.triggered.connect(
                lambda: self.export_requested.emit("dicom", list(sel))
            )
            menu.addAction(act_dcm)
            act_anon = QAction(f"Export (Anon DICOM){suffix}", self)
            act_anon.setToolTip(
                "Like Export (DICOM) but de-identified: the configured tags' "
                "values and all private tags are emptied (right-click the "
                "Anonymous button to choose). Pixels/UIDs kept."
            )
            act_anon.triggered.connect(
                lambda: self.export_requested.emit("anon-dicom", list(sel))
            )
            menu.addAction(act_anon)
            act_mp4 = QAction(f"Export (MP4){suffix}", self)
            act_mp4.setToolTip(
                "Render each selected series to an .mp4 in a chosen "
                "folder. Bitrate (Mbps) and FPS configurable."
            )
            act_mp4.triggered.connect(
                lambda: self.export_requested.emit("mp4", list(sel))
            )
            menu.addAction(act_mp4)
            act_csv = QAction(f"Export (CSV){suffix}", self)
            act_csv.setToolTip(
                "Write the DICOM-Tag-overlay tags shown for each series to a "
                ".csv (Tag Name, Tag Number, Value); one file per series. "
                "Full values — not truncated."
            )
            act_csv.triggered.connect(
                lambda: self.export_requested.emit("csv", list(sel))
            )
            menu.addAction(act_csv)
            # Hide / UnHide.
            menu.addSeparator()
            act_hide = QAction(f"Hide (hide this series){suffix}", self)
            act_hide.setToolTip(
                "Grey out this series — First/Prev/Next/Last and the "
                "keyboard shortcuts skip it; a direct mouse click still "
                "shows it (Play / seek then work normally)."
            )
            act_hide.triggered.connect(
                lambda: self.hide_requested.emit(list(sel))
            )
            menu.addAction(act_hide)
            act_unhide = QAction(f"UnHide (show this series){suffix}", self)
            act_unhide.setToolTip(
                "Un-hide just the selected series (the rest of the study "
                "keeps its current hide/unhide state)."
            )
            act_unhide.triggered.connect(
                lambda: self.unhide_requested.emit(list(sel))
            )
            menu.addAction(act_unhide)
            # Delete (one signal per series — same contract as the Tree).
            menu.addSeparator()
            label = ("Delete (remove from list)" if n == 1
                     else f"Delete {n} series (remove from list)")
            act_del = QAction(label, self)
            act_del.setToolTip(
                "Files are not deleted; just removed from the list so they "
                "can no longer be viewed."
            )

            def _emit_deletes(targets: list[Series]) -> None:
                for s in targets:
                    self.delete_requested.emit(
                        "series", s.series_uid, s.label
                    )

            act_del.triggered.connect(lambda: _emit_deletes(list(sel)))
            menu.addAction(act_del)
        else:
            return            # header / empty area → no menu
        menu.exec(self.viewport().mapToGlobal(pos))


class StudyPanel(QWidget):
    """Tree view + thumbnail grid with a toggle. Public API mirrors the
    parts of StudyBrowser the shell uses."""

    series_chosen = pyqtSignal(object)
    study_clicked = pyqtSignal(str, str)     # study row click (uid, kind)
    anonymize_toggled = pyqtSignal(bool)     # the "Anonymous" button
    #: right-click on the Anonymous button → open the anonymize-settings dialog
    anon_settings_requested = pyqtSignal()
    dicom_info_toggled = pyqtSignal(bool)    # "DICOM Info" (True = show)
    #: right-click on the DICOM Info button → choose which tags to overlay
    dicom_tags_requested = pyqtSignal()
    delete_requested = pyqtSignal(str, str, str)  # (kind, key, label)
    #: ("dicom"|"mp4"|"csv"|"anon-dicom", [Series])
    export_requested = pyqtSignal(str, list)
    #: "Fit: min × 10 across" → ask the shell to widen the Studies dock to
    #: roughly this pixel width (the dock is shell-owned).
    fit_dock_width_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._series_by_row: list[Series] = []
        self._worker: _ThumbWorker | None = None

        self._patients_cache: dict[str, Patient] = {}

        self.tree = StudyBrowser()
        # The selected-row highlight is managed by StudyBrowser itself
        # (blue for a normal series, grey for a hidden one) — see
        # StudyBrowser._refresh_selection_color.
        self.tree.series_chosen.connect(self.series_chosen)
        self.tree.study_clicked.connect(self.study_clicked)
        self.tree.delete_requested.connect(self.delete_requested)
        self.tree.export_requested.connect(self.export_requested)
        self.tree.hide_requested.connect(self._on_hide_series)
        self.tree.unhide_study_requested.connect(self._on_unhide_study)
        self.tree.unhide_series_requested.connect(self._on_unhide_series)
        # Re-sort the thumbnail grid whenever the tree's header sort
        # changes (keep the two views in lock-step).
        self.tree.header().sortIndicatorChanged.connect(
            lambda *_: self._resort_thumbs()
        )

        #: thumbnail icon edge in px (driven by the size slider).
        #: Open at the slider's minimum so more series fit on screen by
        #: default; the user can drag the slider to enlarge as needed.
        self._thumb_px = _THUMB_MIN_PX
        #: (study_uid, kind) currently shown in the thumbnail grid —
        #: thumbnails show ONLY the selected study, not every study.
        self._cur_study_key: tuple | None = None
        #: guard so the tree's selection churn during populate() doesn't
        #: trigger a thumbnail rebuild per intermediate selection.
        self._populating = False

        self.thumbs = _ThumbList()
        self.thumbs.setViewMode(QListView.ViewMode.IconMode)
        self.thumbs.setMovement(QListView.Movement.Static)
        self.thumbs.setResizeMode(QListView.ResizeMode.Adjust)
        # Make the selected thumbnail unmistakable: a thick gold border
        # + blue fill, kept bright even when the grid is not focused.
        self.thumbs.setStyleSheet(
            "QListWidget::item:selected,"
            " QListWidget::item:selected:!active {"
            "   background:#1f6feb; color:white;"
            "   border:3px solid #ffd400;"
            " }"
        )
        self.thumbs.setIconSize(QSize(self._thumb_px, self._thumb_px))
        self.thumbs.setUniformItemSizes(False)  # headers span a full row
        self.thumbs.setSpacing(6)
        self.thumbs.setWordWrap(True)
        self.thumbs.setDragEnabled(True)
        self.thumbs.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        # Multi-select so right-click Export / Delete can target the
        # full Ctrl/Shift-built selection just like the Tree view does.
        self.thumbs.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.thumbs.itemClicked.connect(self._thumb_clicked)
        # Right-click menu on a thumbnail forwards the same shell-level
        # signals the Tree view emits, so Export (DICOM/MP4) and Delete-
        # from-list are reachable from either view.
        self.thumbs.export_requested.connect(self.export_requested)
        self.thumbs.delete_requested.connect(self.delete_requested)
        self.thumbs.hide_requested.connect(self._on_hide_series)
        self.thumbs.unhide_requested.connect(self._on_unhide_series)
        #: (header_item, patient, study, kind) — for anonymize relabel
        self._thumb_headers: list[tuple] = []
        #: series-order list of thumbnail items, aligned with the worker
        self._thumb_items: list[QListWidgetItem] = []
        self._item_by_series: dict[int, QListWidgetItem] = {}

        # Thumbnail size slider — rescales the grid live. Not fixed-width
        # (it must be able to shrink so the whole dock can get narrow).
        self.thumb_size = QSlider(Qt.Orientation.Horizontal)
        self.thumb_size.setRange(_THUMB_MIN_PX, _THUMB_MAX_PX)
        self.thumb_size.setValue(self._thumb_px)
        self.thumb_size.setMinimumWidth(36)
        self.thumb_size.setMaximumWidth(140)
        self.thumb_size.setToolTip("Thumbnail size")
        self.thumb_size.valueChanged.connect(self._set_thumb_size)

        # Rebuild the thumbnail grid whenever the selected study changes.
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)

        self._stack = QStackedWidget()
        self._stack.addWidget(self.tree)
        self._stack.addWidget(self.thumbs)

        self.btn_info = FitButton("📋 Tree")
        self.btn_thumb = FitButton("🖼 Thumbnails")
        for b in (self.btn_info, self.btn_thumb):
            b.setCheckable(True)
        self.btn_info.setChecked(True)
        self.btn_info.clicked.connect(lambda: self._show(0))
        self.btn_thumb.clicked.connect(lambda: self._show(1))
        # Right-click the Thumbnail button → "min size × 10 across" layout
        # (re-applicable any time; the first switch to thumbnails does it
        # automatically too).
        self.btn_thumb.setToolTip(
            "Left-click: thumbnail view (starts at min size × 10 across).\n"
            "Right-click: re-apply the min size × 10 across layout."
        )
        self.btn_thumb.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.btn_thumb.customContextMenuRequested.connect(
            lambda _p: self._fit_thumbs_10across()
        )
        #: first switch to the thumbnail view auto-fits min × 10 across
        self._thumb_fit_done = False

        self.btn_anon = FitButton("Anonymous")
        self.btn_anon.setCheckable(True)
        self.btn_anon.setHelpToolTip(
            "Left-click: mask patient/case info on all on-screen displays "
            "(files unchanged).\nRight-click: choose which tags to "
            "anonymize (also used by Export (Anon DICOM))."
        )
        self.btn_anon.toggled.connect(self.anonymize_toggled)
        # Right-click opens the anonymize-settings dialog (which tags to blank).
        self.btn_anon.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.btn_anon.customContextMenuRequested.connect(
            lambda _p: self.anon_settings_requested.emit()
        )

        self.btn_dicom = FitButton("DICOM Info")
        self.btn_dicom.setCheckable(True)
        self.btn_dicom.setChecked(True)        # overlay shown by default
        self.btn_dicom.setHelpToolTip(
            "左クリック: 画像上のDICOM情報の表示/非表示\n"
            "右クリック: 表示するタグ項目を選択"
        )
        self.btn_dicom.toggled.connect(self.dicom_info_toggled)
        # Right-click → choose overlay tags (mirrors the top-row DICOM Info).
        self.btn_dicom.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.btn_dicom.customContextMenuRequested.connect(
            lambda _p: self.dicom_tags_requested.emit()
        )

        # Let every toolbar button shrink (text clips) so the whole dock
        # can be dragged down to roughly one minimum thumbnail wide.
        for b in (self.btn_info, self.btn_thumb,
                  self.btn_anon, self.btn_dicom):
            b.setMinimumWidth(0)

        top = QHBoxLayout()
        top.setContentsMargins(2, 2, 2, 2)
        top.addWidget(self.btn_info)
        top.addWidget(self.btn_thumb)
        top.addWidget(self.btn_anon)
        top.addWidget(self.btn_dicom)
        top.addStretch(1)
        top.addWidget(QLabel("🖼"))
        top.addWidget(self.thumb_size)

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.addLayout(top)
        col.addWidget(self._stack, 1)
        # Don't let the layout force a wide minimum onto the panel — the
        # toolbar would otherwise pin the whole Studies dock to ~430 px.
        # With no constraint the toolbar just clips when the dock is
        # dragged narrow (see minimumSizeHint below).
        col.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

    def minimumSizeHint(self) -> QSize:
        """Allow the Studies dock to shrink to about one minimum-size
        thumbnail wide; the toolbar buttons clip gracefully."""
        return QSize(_THUMB_MIN_PX + 30,
                     super().minimumSizeHint().height())

    # ----------------------------------------------------------- public API
    def populate(self, patients: dict[str, Patient]) -> None:
        self._patients_cache = patients
        self._populating = True
        self.tree.populate(patients)
        self._populating = False
        self._cur_study_key = self.tree.current_study_key()
        self._rebuild_thumbs()

    def _rebuild_thumbs(self) -> None:
        """Rebuild the thumbnail grid for the CURRENTLY SELECTED study
        only (one section header + that study's series)."""
        self._stop_worker()
        self.thumbs.clear()
        self._thumb_headers = []
        self._thumb_items = []
        self._item_by_series = {}
        self._series_by_row = []
        if not self._patients_cache:
            return
        anon = self.tree._anon
        key = self.tree.current_study_key()
        groups = list(iter_study_groups(
            self._patients_cache, self.tree._sort_modes
        ))
        target = None
        for g in groups:
            _patient, study, kind, _series = g
            if key is not None and (study.study_uid, kind) == key:
                target = g
                break
        if target is None and groups:           # nothing selected yet
            target = groups[0]
        if target is None:
            return
        patient, study, kind, series_list = target
        # Keep state consistent with what is actually on screen (matters
        # when nothing was selected yet and we fell back to groups[0]).
        self._cur_study_key = (study.study_uid, kind)
        # Mirror the tree's current sort: under "instance" mode the
        # caption shows the DICOM InstanceNumber rather than Series No
        # so the user's grouping cue (which # they sorted by) matches
        # the grid.
        mode, _asc = self.tree._sort_modes.get(
            (study.study_uid, kind), ("number", True)
        )
        use_instance = mode == "instance"
        px = self._thumb_px
        hdr = self.thumbs.add_header(
            _group_header(patient, study, kind, anon)
        )
        self._thumb_headers.append((hdr, patient, study, kind))
        for se in series_list:
            it = QListWidgetItem(_thumb_label(se, use_instance))
            it.setData(_ROLE, se)
            it.setSizeHint(QSize(px + 14, px + 40))
            it.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            self.thumbs.addItem(it)
            self._series_by_row.append(se)
            self._thumb_items.append(it)
            self._item_by_series[id(se)] = it
        self.thumbs.setIconSize(QSize(px, px))
        self.thumbs._fit_headers()
        self._apply_thumb_hidden_style()
        if self._series_by_row:
            self._worker = _ThumbWorker(self._series_by_row, self)
            self._worker.ready.connect(self._set_thumb)
            self._worker.start()

    def _apply_thumb_hidden_style(self) -> None:
        """Grey hidden series in the thumbnail grid (mirrors the tree)."""
        grey = QColor("#888")
        for se in self._series_by_row:
            it = self._item_by_series.get(id(se))
            if it is not None:
                it.setForeground(
                    QBrush(grey) if getattr(se, "hidden", False)
                    else QBrush()
                )

    # ------------------------------------------------- hide / show series
    def _on_hide_series(self, series_list) -> None:
        self.tree.set_series_hidden(series_list, True)
        self._apply_thumb_hidden_style()

    def _on_unhide_study(self, study_uid: str, kind: str) -> None:
        series = self.tree.series_in_study(study_uid, kind)
        self.tree.set_series_hidden(series, False)
        self._apply_thumb_hidden_style()

    def _on_unhide_series(self, series_list) -> None:
        """UnHide just the given series (Tree or Thumbnail "UnHide (show
        this series)")."""
        self.tree.set_series_hidden(series_list, False)
        self._apply_thumb_hidden_style()

    def _apply_thumb_fit(self) -> None:
        """Set the thumbnail size to minimum and ask the shell to widen the
        Studies dock so ~10 fit across. No view switch here, so it is safe to
        call from _show without recursion."""
        self.thumb_size.setValue(_THUMB_MIN_PX)    # triggers _set_thumb_size
        # Column ≈ icon edge + item padding + grid spacing; pad for the
        # vertical scrollbar and the panel/dock frame so 10 really fit.
        col = _THUMB_MIN_PX + 20
        self.fit_dock_width_requested.emit(10 * col + 36)

    def _fit_thumbs_10across(self) -> None:
        """Thumbnail-button right-click: switch to the thumbnail view and
        apply the min × 10-across layout (gap-scanning)."""
        self._show(1)                              # ensure thumbnail view
        self._apply_thumb_fit()

    def _set_thumb_size(self, px: int) -> None:
        """Live-rescale the thumbnail grid from the size slider."""
        self._thumb_px = int(px)
        self.thumbs.setIconSize(QSize(self._thumb_px, self._thumb_px))
        for it in self._thumb_items:
            it.setSizeHint(
                QSize(self._thumb_px + 14, self._thumb_px + 40)
            )

    def _on_tree_selection(self) -> None:
        """Switch the thumbnail grid when the selected study changes.

        The rebuild is **deferred** by one event-loop tick so a right-
        click — which selects the clicked item before exec'ing the
        context menu — can pop the menu immediately. Without this the
        menu appeared only AFTER the thumbnail grid had rebuilt, which
        was visible as a 100-500 ms hang on CT (slow mid-slice decode).
        Inside ``menu.exec()`` Qt is still spinning a nested event
        loop, so the singleShot fires and the rebuild proceeds in the
        background of the menu being open."""
        if self._populating:
            return
        key = self.tree.current_study_key()
        if key is not None and key != self._cur_study_key:
            self._cur_study_key = key
            QTimer.singleShot(0, self._rebuild_thumbs)

    def set_anonymized(self, on: bool) -> None:
        """Anonymize the left info area (patient/study identifiers).

        Series labels carry no case info, so the decoded previews are
        kept; only the group headers (patient/date) are re-masked."""
        self.btn_anon.blockSignals(True)
        self.btn_anon.setChecked(bool(on))
        self.btn_anon.blockSignals(False)
        self.tree.set_anonymized(on)
        for hdr, patient, study, kind in self._thumb_headers:
            hdr.setText(_group_header(patient, study, kind, bool(on)))

    def set_dicom_info_shown(self, on: bool) -> None:
        """Sync the DICOM Info button without re-emitting (called when
        the equivalent View-menu action is toggled)."""
        self.btn_dicom.blockSignals(True)
        self.btn_dicom.setChecked(bool(on))
        self.btn_dicom.blockSignals(False)

    def ordered_series(self, modality: str | None = None) -> list[Series]:
        return self.tree.ordered_series(modality)

    def select_series(self, series: Series) -> None:
        self.tree.select_series(series)
        # Make sure the thumbnail grid is on this series' study (the
        # tree-selection signal usually does this; this is the safety
        # net for any path that bypasses it).
        key = self.tree.current_study_key()
        if key is not None and key != self._cur_study_key:
            self._cur_study_key = key
            self._rebuild_thumbs()
        item = self._item_by_series.get(id(series))
        if item is not None:
            self.thumbs.setCurrentItem(item)
            self.thumbs.scrollToItem(item)

    # ------------------------------------------------------------- internals
    def _current_tree_series(self) -> Series | None:
        """Series currently selected in the Tree view, or None."""
        items = self.tree.selectedItems()
        it = items[0] if items else self.tree.currentItem()
        if it is None:
            return None
        data = it.data(0, _ROLE)
        return data if isinstance(data, Series) else None

    def _current_thumb_series(self) -> Series | None:
        """Series currently selected in the Thumbnail grid, or None."""
        items = self.thumbs.selectedItems()
        it = items[0] if items else self.thumbs.currentItem()
        if it is None:
            return None
        data = it.data(_ROLE)
        return data if isinstance(data, Series) else None

    def _highlight_thumb_series(self, series: Series) -> None:
        """Mark *series* as the thumb-grid's current item without
        firing itemClicked (which would re-load the viewer)."""
        item = self._item_by_series.get(id(series))
        if item is None:
            return
        self.thumbs.blockSignals(True)
        try:
            self.thumbs.clearSelection()
            item.setSelected(True)
            self.thumbs.setCurrentItem(item)
            self.thumbs.scrollToItem(item)
        finally:
            self.thumbs.blockSignals(False)

    def _show(self, idx: int) -> None:
        # Tree ↔ Thumbnail selection sync: when the user toggles views,
        # the highlighted series in the outgoing view becomes the
        # highlighted series in the incoming view, so it stays obvious
        # which file they were just looking at. We do NOT emit any
        # load-the-viewer signal — the toggle is a view switch, not a
        # series choice.
        cur = self._stack.currentIndex()
        if cur != idx:
            if cur == 0 and idx == 1:
                se = self._current_tree_series()
                if se is not None:
                    self._highlight_thumb_series(se)
            elif cur == 1 and idx == 0:
                se = self._current_thumb_series()
                if se is not None:
                    self.tree.highlight_series(se)
        self._stack.setCurrentIndex(idx)
        self.btn_info.setChecked(idx == 0)
        self.btn_thumb.setChecked(idx == 1)
        # First time we ever show the thumbnail view: start at min × 10 across
        # (the user's "scan for missing series" default).
        if idx == 1 and not self._thumb_fit_done:
            self._thumb_fit_done = True
            self._apply_thumb_fit()

    def _resort_thumbs(self) -> None:
        """Rebuild tree + thumbnails after a header-sort change and keep the
        user's selection: a selected SERIES stays selected (list open, row in
        view) instead of collapsing onto its Study node; otherwise the Study
        node itself is re-selected. Sort targets the selected Study only."""
        if not self._patients_cache:
            return
        series = self._current_tree_series()   # selected series (if any)
        cur = self.tree.current_study_key()    # its study (fallback)
        self.populate(self._patients_cache)
        if series is not None:
            self.tree.reselect_series_after_sort(series)
        elif cur is not None:
            self.tree.select_study_key(cur)

    def _thumb_clicked(self, item: QListWidgetItem) -> None:
        # A hidden (greyed) series IS loaded on a direct mouse click — a
        # one-off view; Play / seek then work on it. Only the on-image
        # nav buttons and keyboard shortcuts skip hidden series.
        se = item.data(_ROLE)
        if isinstance(se, Series):
            self.series_chosen.emit(se)

    def _set_thumb(self, row: int, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr)
        if arr.ndim == 3:  # (H, W, 3) color preview
            h, w = arr.shape[:2]
            img = QImage(
                arr.data, w, h, 3 * w, QImage.Format.Format_RGB888
            )
        else:
            h, w = arr.shape
            img = QImage(
                arr.data, w, h, w, QImage.Format.Format_Grayscale8
            )
        if 0 <= row < len(self._thumb_items):
            self._thumb_items[row].setIcon(
                QIcon(QPixmap.fromImage(img.copy()))
            )

    def _stop_worker(self) -> None:
        """Detach the running thumbnail worker WITHOUT blocking the UI.

        The worker checks ``self._stop`` between items, so it exits at
        the next iteration boundary on its own — we don't wait for that
        to happen, which used to freeze the UI for up to 3 s when a
        right-click landed mid-decode of a slow CT mid-slice thumb.

        Two safeties make the detach OK:
          1. We disconnect ``ready`` so any in-flight emit from the old
             worker doesn't paint a stale thumbnail into the *new*
             grid's wrong row.
          2. The worker is QObject-parented to this StudyPanel, so Qt
             reaps it when the panel dies even if it's still running.
        """
        if self._worker is not None:
            self._worker.stop()
            try:
                self._worker.ready.disconnect(self._set_thumb)
            except (TypeError, RuntimeError):
                pass
            self._worker = None
