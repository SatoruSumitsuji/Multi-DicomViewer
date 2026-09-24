"""Case Presentation tool (Tools ▸ Case Presentation).

A time-ordered table of the series currently open in the viewer. Each row is
captured from a shown pane (種別 / シリーズ番号 / 時間 are filled automatically,
the doctor types a コメント). The row's 表示 button re-displays that exact image
— same pane, same frame/slice, same zoom / W-L / MPR camera.

Because XA and IVUS/CT machine clocks drift, one modality is the reference
(XA by default) and the others get a constant offset (from an anchor pair or
typed by hand); the 統合時間 column is every row's time on the reference clock,
and 「統合時間で並べ替え」 orders the rows into true procedure order (a non-XA
event within 10 s of an XA event is placed just after it — see core logic).
"""
from __future__ import annotations

import copy
import json
import os
import sys
from datetime import datetime, timedelta

from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QFrame,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.core.case_presentation import (
    json_safe, modified_sort_order, offset_from_anchor, parse_dcm_dt,
    unified_time)
from multi_dicomviewer.i18n import t
from multi_dicomviewer.ui.snap_dock import SnapDock

# Column layout — 表示 / 更新 / 削除 sit between 統合時間 and コメント (easier to
# reach than the far right edge or the top toolbar), so the comment column is
# last (and stretches).
# C_DEL = 除外 (drop the ROW from the presentation, files untouched);
# C_ERASE = 削除 (MOVE the series' files to a CasePresentation-Erase trash folder
# beside the image folder, then drop the row).
(C_NO, C_MOD, C_SER, C_FRAMES, C_SIZE, C_TIME, C_UNI, C_SHOW, C_UPD, C_DEL,
 C_ERASE, C_COMMENT) = range(12)
_HEADERS = ["No", "種別", "Ser", "Frame", "サイズ", "時間", "統合時間", "表示",
            "更新", "除外", "削除", "コメント"]
_SNAP_TOL_S = 10.0            # ±seconds: snap a non-ref event just after an XA


def _accel(*parts: str) -> str:
    """Platform-aware shortcut label for a button caption. Pass modifier names
    ('ctrl' / 'alt' / 'shift') then the key, e.g. _accel('alt', 'shift', 'A'):
    'Alt+Shift+A' on Windows/Linux, '⌥⇧A' on macOS (Qt maps Ctrl→⌘, Alt→⌥)."""
    *mods, key = parts
    if sys.platform == "darwin":
        sym = {"ctrl": "⌘", "alt": "⌥", "shift": "⇧"}
        return "".join(sym[m] for m in mods) + key
    name = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift"}
    return "+".join([name[m] for m in mods] + [key])


def _cap_depth(obj, limit: int = 60, _d: int = 0):
    """Return a copy of a JSON-ish view_state with nesting capped at *limit*
    levels (deeper branches become None). A corrupt, pathologically deep
    view_state would otherwise blow copy.deepcopy's recursion limit and crash
    the app (seen when a saved .json carried a runaway-nested state). Real view
    states are only a few levels deep, so the cap never touches valid data; it
    also returns fresh dict/list objects, so it doubles as a safe copy."""
    if _d >= limit:
        return None
    if isinstance(obj, dict):
        return {k: _cap_depth(v, limit, _d + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cap_depth(v, limit, _d + 1) for v in obj]
    return obj


class _LeftBarDelegate(QStyledItemDelegate):
    """Paints a thin left 縦棒 on a data cell. Set only on the data columns via
    setItemDelegateForColumn, so the bar never bleeds onto the action-button
    columns (whose 縦棒 is a per-button state indicator, or absent). Selection
    highlighting is left to the default painter, so selected cells stay
    readable (the QSS-border approach broke that)."""

    _COL = QColor("#dcdcdc")

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        painter.save()
        painter.setPen(self._COL)
        x = option.rect.left()
        painter.drawLine(x, option.rect.top(), x, option.rect.bottom())
        painter.restore()


def _fmt_raw_time(tm: str) -> str:
    """DICOM TM 'HHMMSS[.ffffff]' → 'HH:MM:SS' (best effort)."""
    tm = (tm or "").strip().replace(":", "")
    if len(tm) < 2:
        return ""
    hh = tm[0:2]
    mm = tm[2:4] if len(tm) >= 4 else "00"
    ss = tm[4:6] if len(tm) >= 6 else "00"
    return f"{hh}:{mm}:{ss}"


def _fmt_offset(sec) -> str:
    """Signed seconds → '＋1時間2分3秒' style, for the applied-offset hint."""
    try:
        total = float(sec)
    except (TypeError, ValueError):
        return "0秒"
    sign = "−" if total < 0 else "＋"
    s = abs(total)
    d = int(s // 86400); s -= d * 86400
    h = int(s // 3600);  s -= h * 3600
    m = int(s // 60);    s -= m * 60
    out = ""
    if d:
        out += f"{d}日"
    if h:
        out += f"{h}時間"
    if m:
        out += f"{m}分"
    out += f"{int(round(s))}秒"
    return sign + out


def _fmt_secs(sec) -> str:
    """Seconds-since-epoch (from parse_dcm_dt) → 'HH:MM:SS'."""
    if sec is None:
        return ""
    try:
        dt = datetime(1970, 1, 1) + timedelta(seconds=float(sec))
        return dt.strftime("%H:%M:%S")
    except (ValueError, OverflowError):
        return ""


class _DHMSEntry(QWidget):
    """Signed duration entry as ＋/− 時間 分 秒. Value is signed seconds."""

    def __init__(self, seconds: float = 0.0, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        # Sign as two always-visible one-click radio buttons (a dropdown was
        # easy to miss / hard to change): ＋ = clock behind ref, − = ahead.
        self._plus = QRadioButton("＋")
        self._minus = QRadioButton("−")
        self._plus.setChecked(True)
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        grp.addButton(self._plus)
        grp.addButton(self._minus)
        lay.addWidget(self._plus)
        lay.addWidget(self._minus)
        self._h = QSpinBox(); self._h.setRange(0, 9999)
        self._m = QSpinBox(); self._m.setRange(0, 59)
        self._s = QDoubleSpinBox()
        self._s.setRange(0.0, 59.0); self._s.setDecimals(0); self._s.setSingleStep(1.0)
        for sb, suf in ((self._h, t("時間")),
                        (self._m, t("分")), (self._s, t("秒"))):
            sb.setSuffix(" " + suf)
            lay.addWidget(sb)
        lay.addStretch(1)
        self.set_seconds(seconds)

    def set_seconds(self, total: float) -> None:
        (self._minus if total < 0 else self._plus).setChecked(True)
        s = abs(float(total))
        h = int(s // 3600);  s -= h * 3600
        m = int(s // 60);    s -= m * 60
        self._h.setValue(h); self._m.setValue(m); self._s.setValue(round(s))

    def seconds(self) -> float:
        mag = (self._h.value() * 3600
               + self._m.value() * 60 + self._s.value())
        return -mag if self._minus.isChecked() else mag


class _OffsetDialog(QDialog):
    """Manual per-modality offset entry (時間 / 分 / 秒; + = that
    modality's clock is behind the reference)."""

    def __init__(self, modalities, offsets, reference, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("時刻オフセット (手入力)"))
        root = QVBoxLayout(self)
        root.addWidget(QLabel(t(
            "各検査の検査時間に下記のオフセット時間を計算した時間が"
            "統合時間に表示されます。\n"
            "基準「{ref}」に対する各モダリティの時刻ズレ (時間・分・秒)。\n"
            "＋ = そのモダリティの時計が基準より遅れている。", ref=reference)))
        form = QFormLayout()
        self._entries = {}
        for mod in modalities:
            e = _DHMSEntry(float(offsets.get(mod, 0.0)))
            self._entries[mod] = e
            form.addRow(mod, e)
        root.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def values(self) -> dict:
        return {m: e.seconds() for m, e in self._entries.items()}


class _DnDTable(QTableWidget):
    """QTableWidget with single-row internal drag & drop. Rather than let Qt
    shuffle the QTableWidgetItems (which would desync from the owner's row
    model), it emits rowMoved(src, final_dst) for the owner to reorder its list
    and rebuild."""

    rowMoved = pyqtSignal(int, int)
    navRow = pyqtSignal(int)          # F/A → +1 / -1 (when NOT editing a cell)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDropIndicatorShown(True)
        self.setDragDropOverwriteMode(False)      # insert BETWEEN rows, not over
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

    def keyPressEvent(self, e) -> None:            # noqa: N802 (Qt override)
        # F = next row, A = previous row — but only when a cell is NOT being
        # edited (so typing 'f'/'a' into a コメント still works) and no modifier
        # is held.
        if (self.state() != QAbstractItemView.State.EditingState
                and e.modifiers() == Qt.KeyboardModifier.NoModifier
                and e.key() in (Qt.Key.Key_F, Qt.Key.Key_A)):
            self.navRow.emit(1 if e.key() == Qt.Key.Key_F else -1)
            e.accept()
            return
        super().keyPressEvent(e)

    def mousePressEvent(self, e) -> None:          # noqa: N802 (Qt override)
        # Select the pressed row FIRST so a press-and-drag starts a drag right
        # away (otherwise the first press only sets the selection and the drag
        # won't begin until a second press-drag). But ONLY on a plain click —
        # with Ctrl (toggle discontiguous rows) or Shift (extend a range) held,
        # defer to the default ExtendedSelection so multi-row picks survive.
        mods = e.modifiers()
        multi = bool(mods & (Qt.KeyboardModifier.ControlModifier
                             | Qt.KeyboardModifier.ShiftModifier))
        if e.button() == Qt.MouseButton.LeftButton and not multi:
            idx = self.indexAt(e.position().toPoint())
            if idx.isValid():
                self.selectRow(idx.row())
        super().mousePressEvent(e)

    def dropEvent(self, e) -> None:            # noqa: N802 (Qt override)
        if e.source() is not self:
            super().dropEvent(e)
            return
        src = self.currentRow()
        pos = e.position().toPoint()
        idx = self.indexAt(pos)
        if idx.isValid():
            dst = idx.row()
            rect = self.visualRect(idx)
            if pos.y() > rect.center().y():    # dropped on the lower half → below
                dst += 1
        else:
            dst = self.rowCount()
        e.setDropAction(Qt.DropAction.IgnoreAction)   # model handles the move
        e.accept()
        if src < 0:
            return
        if dst > src:                          # "insert before dst" → final index
            dst -= 1
        self.rowMoved.emit(src, dst)


class CasePresentationWindow(SnapDock):
    """Dockable Case-Presentation panel. Starts as a floating window; drag it
    onto the Studies dock to tab it there, drag it back out to float again
    (Studies reappears). Floating gestures (double-click maximize / edge snap)
    come from SnapDock. One instance is kept by the shell."""

    def __init__(self, shell, parent=None):
        super().__init__(t("Case Presentation"), parent)
        self.setObjectName("CasePresentationDock")
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self._shell = shell
        self._rows: list[dict] = []
        self._offsets: dict[str, float] = {}       # modality → seconds
        self._reference = "XA"
        self._last_path: str | None = None         # for 上書き保存 (overwrite)
        self._last_dir: str = ""                    # remembered file-dialog folder
        self._dirty = False                         # unsaved changes → close warns
        self._undo: list = []                       # Ctrl+Z snapshots (pre-change)
        self._redo: list = []                       # Ctrl+Y snapshots
        self._displayed_uid: str | None = None      # row shown now → persistent tint

        central = QWidget()
        self.setWidget(central)
        # Allow a narrow panel (the table + toolbars can scroll/shrink).
        central.setMinimumWidth(180)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(3)                    # tight rows (no wasted band)

        # -- toolbar row 1: capture / reference / sort + (right) offset ----
        bar1 = QHBoxLayout()
        b_add = QPushButton(t("この表示を追加"))
        b_add.setToolTip(t("アクティブなペインの表示を1行として取り込む"))
        b_add.clicked.connect(self._add_active)
        bar1.addWidget(b_add)
        b_add_study = QPushButton(t("全シリーズを追加"))
        b_add_study.setToolTip(t(
            "選択中の検査の全シリーズを種別/Ser/時間つきで取り込む "
            "(表示は各シリーズの自動フレーム)"))
        b_add_study.clicked.connect(self._add_all_series)
        bar1.addWidget(b_add_study)
        bar1.addSpacing(12)
        bar1.addWidget(QLabel(t("基準:")))
        self._ref_combo = QComboBox()
        self._ref_combo.setToolTip(t("統合時間の基準モダリティ (通常 XA)"))
        self._ref_combo.currentTextChanged.connect(self._on_ref_changed)
        bar1.addWidget(self._ref_combo)
        b_sort = QPushButton(t("統合時間で並べ替え"))
        b_sort.setToolTip(t(
            "統合時間で手技順に整列 (基準の±10秒以内の他モダリティは"
            "その基準の直後に配置)"))
        b_sort.clicked.connect(self._sort_rows)
        bar1.addWidget(b_sort)
        # Column visibility — toggle optional columns to save width. No/種別/Ser/
        # 表示/コメント stay always-on. Listed in the table's column (title-row)
        # order so the menu matches the header.
        b_cols = QPushButton(t("列表示"))
        b_cols.setToolTip(t("列の表示/非表示を切り替え"))
        col_menu = QMenu(b_cols)
        self._col_actions = {}
        for col, label in ((C_FRAMES, t("Frame")), (C_SIZE, t("サイズ")),
                           (C_TIME, t("時間")), (C_UNI, t("統合時間")),
                           (C_UPD, t("更新")), (C_DEL, t("除外")),
                           (C_ERASE, t("削除"))):
            a = col_menu.addAction(label)
            a.setCheckable(True)
            a.setChecked(True)
            a.toggled.connect(lambda on, c=col: self._set_col_visible(c, on))
            self._col_actions[col] = a
        b_cols.setMenu(col_menu)
        bar1.addWidget(b_cols)
        # Time-offset tools sit just right of 列表示 (left-packed, not floated
        # to the far right where they were hard to find).
        b_anchor = QPushButton(t("アンカーで揃える"))
        b_anchor.setToolTip(t(
            "同一時点とみなす基準行と他モダリティ行を1行ずつ選択 → その"
            "モダリティの時刻オフセットを自動計算"))
        b_anchor.clicked.connect(self._anchor_align)
        bar1.addWidget(b_anchor)
        b_off = QPushButton(t("オフセット手入力…"))
        b_off.clicked.connect(self._edit_offsets)
        bar1.addWidget(b_off)
        bar1.addStretch(1)
        # Reliable window control (floating): maximize ⇄ restore, and the way to
        # UNDO an edge-snap even if the title bar is hard to reach.
        b_max = QPushButton("⤢")
        b_max.setToolTip(t("最大化 / 元のサイズに戻す (フロート時)。"
                           "上下スナップの解除にも使えます"))
        b_max.setFixedWidth(34)
        b_max.clicked.connect(self.toggle_maximize)
        bar1.addWidget(b_max)
        outer.addWidget(self._wrap_bar(bar1))

        # -- toolbar row 2: reorder / update-delete / file (left-packed) ---
        bar2 = QHBoxLayout()
        # "リスト変更": reorder the SELECTED row within the list (最初 / 10上 /
        # 一つ上 / 一つ下 / 10下 / 最後; drag & drop also works). Vertical symbols
        # (bar+triangle = edge, double = 10, single = 1) are deliberately kept
        # DIFFERENT from row 3's horizontal arrows, which move which series is
        # DISPLAYED. 状態更新・除外・削除 were removed here — each row now has its
        # own buttons (and the right-click menu), so they were redundant.
        bar2.addWidget(QLabel(t("リスト変更:")))
        for sym, tip, fn in (
                ("⤒", t("選択行を最初へ"), lambda: self._move_edge(True)),
                ("⏫", t("選択行を10上へ"), lambda: self._move(-10)),
                ("▲", t("選択行を一つ上へ"), lambda: self._move(-1)),
                ("▼", t("選択行を一つ下へ"), lambda: self._move(+1)),
                ("⏬", t("選択行を10下へ"), lambda: self._move(+10)),
                ("⤓", t("選択行を最後へ"), lambda: self._move_edge(False))):
            b = QPushButton(sym)
            b.setToolTip(tip)
            b.setFixedWidth(34)
            b.clicked.connect(fn)
            bar2.addWidget(b)
        bar2.addSpacing(12)
        # Batch remove of the CURRENT multi-selection (Shift/Ctrl-select rows,
        # or right-click them). The per-row 除外/削除 buttons act on one row;
        # these act on every selected row at once. Same handlers as the
        # right-click menu — nothing happens if the selection is empty.
        b_bdel = QPushButton(t("一括除外"))
        b_bdel.setToolTip(t("選択中の複数シリーズをまとめてプレゼンから除外"
                            "（元ファイルは残す）"))
        b_bdel.clicked.connect(self._delete_selected)
        bar2.addWidget(b_bdel)
        b_berase = QPushButton(t("一括削除"))
        b_berase.setToolTip(t(
            "選択中の複数シリーズの元ファイルをまとめて CasePresentation-Erase "
            "フォルダへ移動（元に戻せます）"))
        b_berase.clicked.connect(self._erase_selected)
        bar2.addWidget(b_berase)
        bar2.addSpacing(12)
        b_overwrite = QPushButton(t("上書き保存"))
        b_overwrite.setToolTip(t("直前に保存/読込したファイルへ上書き保存"))
        b_overwrite.clicked.connect(self._save_overwrite)
        bar2.addWidget(b_overwrite)
        self._b_overwrite = b_overwrite
        self._update_overwrite_style()      # reflect the current saved/dirty state
        b_save = QPushButton(t("名前を付けて保存…"))
        b_save.clicked.connect(self._save)
        bar2.addWidget(b_save)
        b_load = QPushButton(t("読込…"))
        b_load.clicked.connect(self._load)
        bar2.addWidget(b_load)
        b_reloc = QPushButton(t("フォルダ再指定…"))
        b_reloc.setToolTip(t(
            "画像フォルダを移動/改名して「見当たりません」になった時に、新しい"
            "フォルダを再指定して再スキャン。SeriesUIDで自動的に再結合します"))
        b_reloc.clicked.connect(self._relocate_folders)
        bar2.addWidget(b_reloc)
        b_clear = QPushButton(t("全消去"))
        b_clear.clicked.connect(self._clear_all)
        bar2.addWidget(b_clear)
        bar2.addStretch(1)
        outer.addWidget(self._wrap_bar(bar2))

        # -- toolbar row 3: which SERIES is displayed (select + display) ---
        # Click-based navigation that always works regardless of keyboard focus
        # / active window (Alt+F/A = 次/前, Alt+Shift+F/A = 最後/最初 mirror the
        # buttons). HORIZONTAL arrows only (back = left-based, forward =
        # right-based) so this row can't be confused with row 2's vertical
        # reorder arrows. Each button with a shortcut shows it in the platform's
        # keys (10前/10後 have none).
        _altA, _altF = _accel("alt", "A"), _accel("alt", "F")
        _asA, _asF = _accel("alt", "shift", "A"), _accel("alt", "shift", "F")
        bar3 = QHBoxLayout()
        for label, tip, fn in (
                (f"|◀ {t('最初')} {_asA}",
                 f"{t('一番最初の行へ移動して表示')}  ({_asA})",
                 self._nav_first),
                (f"◀◀ {t('10前')}", t("10行前へ移動して表示"),
                 lambda: self._nav_row(-10)),
                (f"◀ {t('前')} {_altA}",
                 f"{t('前の行へ移動して表示')}  ({_altA})",
                 lambda: self._nav_row(-1)),
                (f"{t('次')} {_altF} ▶",
                 f"{t('次の行へ移動して表示')}  ({_altF})",
                 lambda: self._nav_row(+1)),
                (f"{t('10後')} ▶▶", t("10行後へ移動して表示"),
                 lambda: self._nav_row(+10)),
                (f"{t('最後')} {_asF} ▶|",
                 f"{t('一番最後の行へ移動して表示')}  ({_asF})",
                 self._nav_last)):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            bar3.addWidget(b)
        bar3.addStretch(1)
        outer.addWidget(self._wrap_bar(bar3))

        # -- table (drag & drop reorders rows) ----------------------------
        self._table = _DnDTable(0, len(_HEADERS))
        self._table.rowMoved.connect(self._on_row_dragged)
        self._table.navRow.connect(self._nav_row)
        self._table.setHorizontalHeaderLabels([t(h) for h in _HEADERS])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        hh = self._table.horizontalHeader()
        # Every data column is USER-RESIZABLE (drag the header borders) instead of
        # locked to its contents; the comment column stretches to fill the rest.
        # Sensible initial widths are set below and persist across rebuilds.
        for c in (C_NO, C_MOD, C_SER, C_FRAMES, C_SIZE, C_TIME, C_UNI, C_SHOW,
                  C_UPD, C_DEL, C_ERASE):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(C_COMMENT, QHeaderView.ResizeMode.Stretch)
        for c, w in ((C_NO, 44), (C_MOD, 70), (C_SER, 56), (C_FRAMES, 56),
                     (C_SIZE, 68), (C_TIME, 92), (C_UNI, 92), (C_SHOW, 64),
                     (C_UPD, 56), (C_DEL, 56), (C_ERASE, 56)):
            self._table.setColumnWidth(c, w)
        # No global grid: the left "縦棒" is drawn per column so it's fully
        # controllable. DATA columns get a light bar via a delegate (never
        # bleeding onto the button columns); the padding keeps text off that bar.
        # The four action buttons paint their OWN conditional bar (see
        # _cell_btn_css / _rebuild) — 除外・削除 none, 表示/更新 state-dependent.
        # Only padding + an explicit selection colour go through QSS: a QSS
        # *border* here silently broke the selected-row highlight (cells went
        # blank), so borders are the delegate's job, not the stylesheet's.
        self._table.setShowGrid(False)
        self._table.setStyleSheet(
            "QTableWidget::item{padding-left:6px;}"
            "QTableWidget::item:selected{background:#cfe4ff;color:#000;}")
        self._bar_delegate = _LeftBarDelegate(self._table)
        for c in (C_NO, C_MOD, C_SER, C_FRAMES, C_SIZE, C_TIME, C_UNI,
                  C_COMMENT):
            self._table.setItemDelegateForColumn(c, self._bar_delegate)
        self._table.cellChanged.connect(self._on_cell_changed)
        # Row right-click menu: 状態更新 / 削除.
        self._table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._row_menu)
        outer.addWidget(self._table, 1)

        self._hint = QLabel("")
        self._hint.setStyleSheet("color:#888;")
        outer.addWidget(self._hint)

        self._refresh_ref_combo()
        self._rebuild()

        # Keyboard: Ctrl+S = 上書き保存, Ctrl+Z / Ctrl+Y = Undo / Redo.
        # Save is APPLICATION-wide (gated to the panel being visible) so Ctrl+S
        # works even while an image pane / the main window has focus — you don't
        # have to click into the panel first. (No other Ctrl+S exists in the app.)
        from PyQt6.QtGui import QKeySequence, QShortcut
        sc_save = QShortcut(QKeySequence.StandardKey.Save, self)
        sc_save.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_save.activated.connect(self._save_shortcut)
        sc_undo = QShortcut(QKeySequence.StandardKey.Undo, self)
        sc_undo.activated.connect(self._undo_action)
        sc_redo = QShortcut(QKeySequence.StandardKey.Redo, self)
        sc_redo.activated.connect(self._redo_action)
        sc_redo2 = QShortcut(QKeySequence("Ctrl+Y"), self)
        sc_redo2.activated.connect(self._redo_action)

        # Row navigation shortcuts. Plain F/A stay with the main viewer (its
        # per-image stepping); the panel uses Alt+F / Alt+A = next / prev SERIES,
        # application-wide but gated to when the panel is visible and a text field
        # isn't being edited (see _nav_shortcut). Alt (not Shift) for next/prev:
        # the main window already binds app-wide Shift+F/Shift+A to the active
        # pane's last/first image, and two app-wide shortcuts on one sequence go
        # AMBIGUOUS in Qt so neither fires. Alt+F needs the &File menu mnemonic
        # freed (done: it's Alt+E now). 最初/最後 use Alt+Shift+A / Alt+Shift+F
        # (not Ctrl+A/Ctrl+F, which clashed with the conventional select-all /
        # find and Studies' Ctrl+A select-all).
        for seq, fn in (("Alt+F", lambda: self._nav_shortcut(+1)),
                        ("Alt+A", lambda: self._nav_shortcut(-1)),
                        ("Alt+Shift+F", lambda: self._nav_shortcut("last")),
                        ("Alt+Shift+A", lambda: self._nav_shortcut("first"))):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(fn)

        # (SnapDock installs the app-wide event filter for the title double-click
        # / floating maximize / edge-snap.)

        # Whenever this panel becomes DOCKED, force it to TAB with the Studies
        # dock (never a side-by-side split) so the docked layout is predictable:
        # Studies + Case Presentation share one area as tabs.
        self.topLevelChanged.connect(self._on_top_level_changed)

    def _on_top_level_changed(self, floating: bool) -> None:
        if not floating:
            # Defer so it runs AFTER Qt finishes placing the drop.
            QTimer.singleShot(0, self._tabify_with_studies)

    def _tabify_with_studies(self) -> None:
        studies = getattr(self._shell, "_studies_dock", None)
        if studies is None or studies is self or self.isFloating():
            return
        try:
            self._shell.tabifyDockWidget(studies, self)
            self.raise_()                     # show this panel's tab on top
        except Exception:                      # noqa: BLE001
            pass

    def _wrap_bar(self, lay) -> QScrollArea:
        """Put a toolbar row in a horizontally-scrollable strip so the panel can
        be dragged narrow without the buttons forcing a wide minimum."""
        lay.setContentsMargins(0, 0, 0, 0)
        w = QWidget()
        w.setLayout(lay)
        sc = QScrollArea()
        sc.setWidget(w)
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        sc.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Reserve exactly the h-scrollbar height (needed only when dragged
        # narrow) — no bigger, so there's no empty band at normal width.
        sbh = sc.horizontalScrollBar().sizeHint().height()
        sc.setFixedHeight(max(26, w.sizeHint().height()) + max(2, sbh))
        return sc

    def _set_col_visible(self, col: int, on: bool) -> None:
        self._table.setColumnHidden(col, not on)

    def changeEvent(self, e):  # noqa: N802 (Qt override)
        super().changeEvent(e)
        # When focus returns here after the doctor adjusted the image in the
        # main viewer, silently update the selected row's key image to that
        # live view (the "各操作をしたとき" background 更新).
        if e.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            if not getattr(self, "_suppress_capture", False):
                self._capture_view_into_selected()
            # If activating left no focused child (e.g. clicked the title bar),
            # focus the table so keyboard use lands somewhere sensible.
            if self.focusWidget() is None:
                self._table.setFocus(Qt.FocusReason.OtherFocusReason)

    # (Plain F/A are intentionally NOT intercepted — they belong to the main
    # viewer. Row navigation uses the 前/次 buttons and Alt+F/A · Ctrl+F/A
    # shortcuts set up in __init__. Title-bar double-click is handled by the
    # inherited SnapDock.eventFilter.)

    # ------------------------------------------------------------- undo/redo
    def _state_snapshot(self) -> dict:
        """Deep copy of the editable state (rows / offsets / reference), plus the
        save target so undoing a 読込 also restores where 上書き保存 points."""
        return {
            "rows": copy.deepcopy(self._rows),
            "offsets": copy.deepcopy(self._offsets),
            "reference": self._reference,
            "last_path": self._last_path,
        }

    def _restore_state(self, snap: dict) -> None:
        self._rows = copy.deepcopy(snap.get("rows", []))
        self._offsets = copy.deepcopy(snap.get("offsets", {}))
        self._reference = snap.get("reference", "XA")
        self._last_path = snap.get("last_path", self._last_path)
        self._dirty = True
        self._refresh_ref_combo()
        self._rebuild()

    def _record_undo(self) -> None:
        """Snapshot the CURRENT state onto the undo stack (call BEFORE a change);
        a new change forks the redo history."""
        try:
            snap = self._state_snapshot()
        except RecursionError:
            # Safety net: a pathologically deep row (corrupt view_state) could
            # blow deepcopy's limit. Don't crash — skip this undo point; the edit
            # still applies. (view_state is depth-capped on entry, so this should
            # not normally happen.)
            return
        self._undo.append(snap)
        if len(self._undo) > 100:
            self._undo.pop(0)
        self._redo.clear()

    def _undo_action(self) -> None:
        if not self._undo:
            return
        snap = self._undo.pop()
        cur = self._state_snapshot()
        fm = snap.pop("file_moves", None)
        restored = None
        if fm:                                   # 削除 was undone → restore files
            try:
                restored = self._shell.case_restore_erased(fm)
            except Exception:                    # noqa: BLE001
                restored = None
            cur["file_moves"] = fm               # let Redo re-erase them
        self._redo.append(cur)
        self._restore_state(snap)                # rebuilds (resets the hint)
        if restored is not None:
            self._hint.setText(t(
                "{n} 個のファイルを元の場所に戻しました"
                "（読込完了後に表示できます）。", n=restored))

    def _redo_action(self) -> None:
        if not self._redo:
            return
        snap = self._redo.pop()
        cur = self._state_snapshot()
        fm = snap.pop("file_moves", None)
        if fm:                                   # redo the 削除 → move files out
            try:
                self._shell.case_reerase(fm)
            except Exception:                    # noqa: BLE001
                pass
            cur["file_moves"] = fm               # let Undo restore them again
        self._undo.append(cur)
        self._restore_state(snap)

    # ---------------------------------------------------------------- add
    def _add_active(self) -> None:
        row = self._shell.case_capture_active()
        if row is None:
            self._warn(t("表示中のシリーズがありません。"))
            return
        row["view_state"] = _cap_depth(row.get("view_state", {}))
        self._record_undo()
        self._rows.append(row)
        self._after_rows_changed(select_last=True)

    def _add_all(self) -> None:
        rows = self._shell.case_capture_all()
        if not rows:
            self._warn(t("表示中のシリーズがありません。"))
            return
        for r in rows:
            r["view_state"] = _cap_depth(r.get("view_state", {}))
        self._record_undo()
        self._rows.extend(rows)
        self._after_rows_changed(select_last=True)

    def _add_all_series(self) -> None:
        """全シリーズを追加: one row per series of the selected study (種別/Ser/
        時間 auto). Skips series already present so re-pressing doesn't duplicate."""
        rows = self._shell.case_capture_all_series()
        if not rows:
            self._warn(t("シリーズが見つかりません (検査を表示してから押してください)。"))
            return
        self._record_undo()
        have = {r.get("series_uid") for r in self._rows if r.get("series_uid")}
        added = 0
        for r in rows:
            if r.get("series_uid") and r["series_uid"] in have:
                continue
            r["view_state"] = _cap_depth(r.get("view_state", {}))
            self._rows.append(r)
            added += 1
        if added == 0:
            self._warn(t("追加できる新しいシリーズがありません。"))
            return
        self._after_rows_changed(select_last=True)

    # ------------------------------------------------------------- offsets
    def _present_modalities(self) -> list:
        seen = []
        for r in self._rows:
            m = r.get("modality", "")
            if m and m not in seen:
                seen.append(m)
        return seen

    def _non_ref_modalities(self) -> list:
        return [m for m in self._present_modalities() if m != self._reference]

    def _edit_offsets(self) -> None:
        mods = self._non_ref_modalities()
        if not mods:
            self._warn(t("基準以外のモダリティがありません。"))
            return
        dlg = _OffsetDialog(mods, self._offsets, self._reference, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._record_undo()
            self._offsets.update(dlg.values())
            self._dirty = True
            self._rebuild()
            parts = [f"{m}: {_fmt_offset(self._offsets.get(m, 0.0))}"
                     for m in mods]
            self._hint.setText(t("オフセットを適用しました — ") + " / ".join(parts))

    def _anchor_align(self) -> None:
        sel = self._selected_row_indices()
        if len(sel) != 2:
            self._warn(t("基準行と他モダリティ行をちょうど2行選択してください。"))
            return
        a, b = self._rows[sel[0]], self._rows[sel[1]]
        # identify which is the reference
        if a.get("modality") == self._reference \
                and b.get("modality") != self._reference:
            ref_row, oth = a, b
        elif b.get("modality") == self._reference \
                and a.get("modality") != self._reference:
            ref_row, oth = b, a
        else:
            self._warn(t("基準「{ref}」の行と、それ以外のモダリティの行を"
                         "1行ずつ選んでください。", ref=self._reference))
            return
        ref_dt = parse_dcm_dt(ref_row.get("date", ""), ref_row.get("time", ""))
        oth_dt = parse_dcm_dt(oth.get("date", ""), oth.get("time", ""))
        if ref_dt is None or oth_dt is None:
            self._warn(t("選択した行に有効な時刻がありません。"))
            return
        off = offset_from_anchor(ref_dt, oth_dt)
        self._record_undo()
        self._offsets[oth["modality"]] = off
        self._hint.setText(t(
            "{mod} のオフセットを {sec:+.1f} 秒に設定しました。",
            mod=oth["modality"], sec=off))
        self._dirty = True
        self._rebuild()

    # -------------------------------------------------------------- sort
    def _unified_for(self, row) -> float | None:
        # CT is EXCLUDED from the unified-time calc (no meaningful acquisition
        # instant vs the XA/IVUS procedure timeline): its 統合時間 stays blank and
        # it sorts to the end (keeping order), per request.
        if (row.get("modality", "") or "").upper() == "CT":
            return None
        dt = parse_dcm_dt(row.get("date", ""), row.get("time", ""))
        return unified_time(dt, row.get("modality", ""), self._reference,
                            self._offsets)

    @staticmethod
    def _is_ct(row) -> bool:
        return (row.get("modality", "") or "").upper() == "CT"

    def _sort_rows(self) -> None:
        items = [{"dt": self._unified_for(r),
                  "is_ref": r.get("modality") == self._reference}
                 for r in self._rows]
        order = modified_sort_order(items, tol=_SNAP_TOL_S)
        # CT is outside the timeline → put CT rows at the TOP by default (keeping
        # their relative order); the time-sorted rest follow.
        ct = [i for i in order if self._is_ct(self._rows[i])]
        rest = [i for i in order if not self._is_ct(self._rows[i])]
        order = ct + rest
        self._record_undo()
        self._rows = [self._rows[i] for i in order]
        self._dirty = True
        self._rebuild()

    # ------------------------------------------------------------ reorder
    def _relocate(self, i: int, j_final: int) -> None:
        """MOVE (not swap) row *i* to final position *j_final* (clamped)."""
        n = len(self._rows)
        if not (0 <= i < n):
            return
        j_final = max(0, min(n - 1, j_final))
        if i == j_final:
            return
        self._record_undo()
        r = self._rows.pop(i)
        self._rows.insert(j_final, r)          # after pop, insert clamps to end
        self._dirty = True
        self._rebuild(select=j_final)

    def _move(self, delta: int) -> None:
        """Move the selected row by *delta* (±1 / ±10; relocate, not swap)."""
        sel = self._selected_row_indices()
        if len(sel) != 1:
            return
        self._relocate(sel[0], sel[0] + delta)

    def _move_edge(self, first: bool) -> None:
        """Move the selected row to the very first / last position."""
        sel = self._selected_row_indices()
        if len(sel) != 1:
            return
        self._relocate(sel[0], 0 if first else len(self._rows) - 1)

    def _on_row_dragged(self, src: int, dst: int) -> None:
        """Drag & drop reorder from the table (src row dropped at dst)."""
        self._relocate(src, dst)

    def _delete_selected(self) -> None:
        sel = set(self._selected_row_indices())
        if not sel:
            return
        self._record_undo()
        target = min(sel)
        self._rows = [r for i, r in enumerate(self._rows) if i not in sel]
        self._after_rows_changed()
        self._show_at_index(target)

    def _delete_row(self, row) -> None:
        """除外: drop the one row backing a per-row 除外 button (files untouched)."""
        try:
            i = self._rows.index(row)
        except ValueError:
            return
        self._record_undo()
        del self._rows[i]
        self._after_rows_changed()
        self._show_at_index(i)

    # ---- 削除 (erase): MOVE the series' files to a trash folder + drop the row
    def _erase_row(self, row) -> None:
        self._erase_rows([row])

    def _erase_selected(self) -> None:
        sel = sorted(self._selected_row_indices())
        rows = [self._rows[i] for i in sel if 0 <= i < len(self._rows)]
        self._erase_rows(rows)

    def _erase_rows(self, rows) -> None:
        """削除: MOVE each row's series files to a CasePresentation-Erase folder
        beside the image folder (reversible), then drop the row. Confirms first
        (destructive). Rows whose series can't be resolved to files (not loaded /
        ambiguous) are KEPT and reported."""
        rows = [r for r in rows if r]
        if not rows:
            return
        if QMessageBox.question(
                self, t("削除（ファイル移動）"),
                t("選択した {n} 件のシリーズの元ファイルを、画像フォルダの親にある "
                  "CasePresentation-Erase フォルダへ移動します。\n"
                  "（削除ではなく移動なので、後で手動で元に戻せます。）\n"
                  "よろしいですか?", n=len(rows))) \
                != QMessageBox.StandardButton.Yes:
            return
        self._record_undo()
        moved, errs, kept = 0, [], 0
        done = []
        all_moves, all_keys, all_dirs = [], [], []
        for r in rows:
            uid = r.get("series_uid", "")
            res = (self._shell.case_erase_series(uid) if uid
                   else {"no_files": True})
            if res.get("no_files"):
                kept += 1
                continue
            moved += int(res.get("moved", 0))
            errs.extend(res.get("errors", []))
            all_moves.extend(res.get("moves", []))
            all_keys.extend(res.get("keys", []))
            all_dirs.extend(res.get("dirs", []))
            done.append(r)
        # Attach the file-move record to the undo snapshot we just pushed, so
        # Ctrl+Z can move the files back out of CasePresentation-Erase (and
        # Ctrl+Y re-erase them) — see _undo_action / _redo_action.
        if all_moves and self._undo:
            self._undo[-1]["file_moves"] = {
                "moves": all_moves, "keys": all_keys, "dirs": all_dirs}
        target = min((i for i, r in enumerate(self._rows) if r in done),
                     default=None)
        self._rows = [r for r in self._rows if r not in done]
        self._after_rows_changed()
        self._show_at_index(target)
        msg = t("{m} 個のファイルを CasePresentation-Erase へ移動しました。", m=moved)
        if kept:
            msg += t(" 未読込/特定不可で残した行: {k} 件（「状態更新」後に再実行）。",
                     k=kept)
        if errs:
            msg += t(" 失敗: {e} 件。", e=len(errs))
        self._hint.setText(msg)
        if errs:
            self._warn("\n".join(errs[:8]))

    def _refresh_row(self, row) -> None:
        """Per-row 更新: capture the row's CURRENT on-screen view as its key
        image (frame/zoom/W-L/…), then re-check load state (keeping this row
        selected). This is what makes 'adjust to the best frame → 更新 → 保存'
        actually persist that frame."""
        before = row.get("view_state")
        self._capture_view_into(row)
        captured = row.get("view_state") is not before
        if captured:
            row["_refreshed"] = True     # move the 縦棒 status bar to 更新
        try:
            i = self._rows.index(row)
        except ValueError:
            i = None
        self._rebuild(select=i)
        if captured:
            self._hint.setText(t("この行のキー画像を現在の表示で更新しました。"))
        else:
            self._hint.setText(t(
                "現在このシリーズが表示されていないため、キー画像は"
                "更新されませんでした (先に「表示」してください)。"))

    def _refresh_state(self) -> None:
        """Toolbar 状態更新: capture the selected rows' current on-screen views
        as their key images, then re-check load state."""
        sel = self._selected_row_indices()
        for i in sel:
            if 0 <= i < len(self._rows):
                r = self._rows[i]
                before = r.get("view_state")
                self._capture_view_into(r)
                if r.get("view_state") is not before:
                    r["_refreshed"] = True   # move the 縦棒 status bar to 更新
        self._rebuild(select=sel[0] if sel else None)

    def _nav_goto(self, index: int) -> None:
        """Select row *index* (clamped) and display it.

        The row SELECTION happens now (cheap); the actual display is DEFERRED to
        the next event-loop turn. Displaying a series can spin up a CT/VTK GL
        context and rebuild panes — doing that synchronously inside a key-event
        dispatch can hard-crash Qt/VTK. singleShot(0) runs it after the event
        unwinds."""
        n = len(self._rows)
        if n == 0:
            return
        index = max(0, min(n - 1, index))
        self._table.selectRow(index)
        self._table.setCurrentCell(index, C_COMMENT)
        row = self._rows[index]
        QTimer.singleShot(0, lambda r=row: self._display_row_deferred(r))

    def _displayed_index(self) -> int:
        """Row index of the currently-displayed series (_displayed_uid), or -1."""
        uid = self._displayed_uid
        if not uid:
            return -1
        for i, r in enumerate(self._rows):
            if r.get("series_uid") == uid:
                return i
        return -1

    def _nav_row(self, step: int) -> None:
        """Move the selection by ``step`` (±1 / ±10) and display it. When the
        table has no current row (e.g. right after a 読込 / a select=None
        rebuild), step from the CURRENTLY-DISPLAYED row instead of collapsing to
        the first/last edge — otherwise Alt+F ("next") would jump to row 0 and
        look exactly like Alt+Shift+A ("first"). With NO reference row at all
        (nothing selected and nothing displayed) a relative move does nothing —
        use 最初/最後 (Alt+Shift+A/F) or 表示 to establish a starting point."""
        if not self._rows:
            return
        cur = self._table.currentRow()
        if cur < 0:
            cur = self._displayed_index()
        if cur < 0:                       # no reference row → do nothing
            return
        self._nav_goto(cur + step)

    def _nav_first(self) -> None:
        self._nav_goto(0)

    def _nav_last(self) -> None:
        self._nav_goto(len(self._rows) - 1)

    def _nav_shortcut(self, target) -> None:
        """Shift/Ctrl row-nav shortcut, gated so it only fires when the panel is
        visible and a text field / modal isn't taking input."""
        if not self.isVisible():
            return
        app = QApplication.instance()
        if app is not None:
            if app.activeModalWidget() is not None:
                return
            if isinstance(app.focusWidget(), (QLineEdit, QAbstractSpinBox)):
                return
        if target == "first":
            self._nav_first()
        elif target == "last":
            self._nav_last()
        else:
            self._nav_row(int(target))

    def _save_shortcut(self) -> None:
        """App-wide Ctrl+S → 上書き保存, gated to when the panel is visible and no
        modal is up (so it never hijacks Ctrl+S for an unrelated window). If a
        コメント cell is mid-edit its text is committed first so the save
        includes it."""
        if not self.isVisible():
            return
        app = QApplication.instance()
        if app is not None and app.activeModalWidget() is not None:
            return
        # Flush any open cell editor so the current comment is persisted.
        if self._table.state() == QAbstractItemView.State.EditingState:
            self._table.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self._save_overwrite()

    def _display_row_deferred(self, row) -> None:
        # Stay silent on unloaded rows so rapid F/A stepping isn't interrupted
        # by a modal warning.
        try:
            self._shell.case_redisplay(row)
        except Exception:                            # noqa: BLE001
            pass
        self._displayed_uid = row.get("series_uid")
        self._update_displayed_highlight()
        self._return_focus()

    def _current_row_dict(self):
        i = self._table.currentRow()
        if 0 <= i < len(self._rows):
            return self._rows[i]
        return None

    def _capture_view_into(self, row) -> None:
        """Silently refresh a row's stored key image to the CURRENT live view of
        its series (if that series is on screen). This is the background "更新"
        the doctor wants: the image they're looking at while writing findings
        becomes the one 表示 restores later."""
        if not row:
            return
        try:
            vs = self._shell.case_current_view_state(row.get("series_uid", ""))
        except Exception:                                # noqa: BLE001
            vs = None
        if vs:
            row["view_state"] = _cap_depth(vs)   # guard against deep-nested state
            self._dirty = True

    def _capture_view_into_selected(self) -> None:
        self._capture_view_into(self._current_row_dict())

    def _return_focus(self) -> None:
        """Bring focus back to this window after a display (which activates the
        main viewer window). Keeping THIS window active is what makes F/A keep
        stepping rows — see eventFilter(). Deferred so it wins any activation
        the display path performs on the next event-loop turn."""
        # Suppress the activation-driven capture through the display: the view
        # is (re)stored to the row's OWN saved state — for a not-yet-loaded
        # series that restore is deferred ~450 ms — so capturing now would grab
        # a pre-restore view and clobber the row's key image.
        self._suppress_capture = True
        QTimer.singleShot(0, self._do_return_focus)
        QTimer.singleShot(
            800, lambda: setattr(self, "_suppress_capture", False))

    def _do_return_focus(self) -> None:
        self.activateWindow()
        self.raise_()
        # Give the TABLE keyboard focus: activating a window alone may leave no
        # focused child, so F/A key presses wouldn't be dispatched anywhere and
        # row navigation would appear dead. With the table focused, F/A flow to
        # its keyPressEvent (and the app filter) reliably.
        self._table.setFocus(Qt.FocusReason.OtherFocusReason)

    def _row_menu(self, pos) -> None:
        """Row right-click menu: 表示 / move (最初・10上・一つ上・一つ下・10下・
        最後) / 状態更新 / 削除."""
        idx = self._table.indexAt(pos)
        row = idx.row() if idx.isValid() else -1
        if row >= 0 and row not in set(self._selected_row_indices()):
            self._table.selectRow(row)        # right-click selects the row
        menu = QMenu(self)
        a_show = menu.addAction(t("表示"))
        a_show.setEnabled(row >= 0)
        menu.addSeparator()
        mv = menu.addMenu(t("移動"))
        a_first = mv.addAction(t("最初へ"))
        a_up10 = mv.addAction(t("10上へ"))
        a_up1 = mv.addAction(t("一つ上へ"))
        a_dn1 = mv.addAction(t("一つ下へ"))
        a_dn10 = mv.addAction(t("10下へ"))
        a_last = mv.addAction(t("最後へ"))
        menu.addSeparator()
        a_ref = menu.addAction(t("状態更新"))
        nsel = len(self._selected_row_indices())
        a_del = menu.addAction(t("一括除外 ({n}件)", n=nsel) if nsel > 1
                               else t("除外"))
        a_erase = menu.addAction(
            t("一括削除・ファイル移動 ({n}件)", n=nsel) if nsel > 1
            else t("削除（ファイル移動）"))
        chosen = menu.exec(self._table.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is a_show:
            if 0 <= row < len(self._rows):
                self._show_row(self._rows[row])
        elif chosen is a_first:
            self._move_edge(True)
        elif chosen is a_up10:
            self._move(-10)
        elif chosen is a_up1:
            self._move(-1)
        elif chosen is a_dn1:
            self._move(+1)
        elif chosen is a_dn10:
            self._move(+10)
        elif chosen is a_last:
            self._move_edge(False)
        elif chosen is a_ref:
            self._rebuild()
        elif chosen is a_del:
            self._delete_selected()
        elif chosen is a_erase:
            self._erase_selected()

    def _clear_all(self) -> None:
        if not self._rows:
            return
        if QMessageBox.question(
                self, t("全消去"),
                t("全ての行を消去しますか?")) == QMessageBox.StandardButton.Yes:
            self._record_undo()
            self._rows = []
            self._displayed_uid = None
            self._after_rows_changed()

    # ---------------------------------------------------------- display
    def _show_row(self, row) -> None:
        # Make the clicked row current so F/A continue from it (clicking the 表示
        # cell-button doesn't go through the table's row selection).
        try:
            i = self._rows.index(row)
            self._table.selectRow(i)
            self._table.setCurrentCell(i, C_COMMENT)
        except ValueError:
            pass
        ok = self._shell.case_redisplay(row)
        if not ok:
            self._warn(t(
                "このシリーズは現在読み込まれていません "
                "(閉じられた可能性があります)。元のフォルダを開き直してください。"))
        else:
            self._displayed_uid = row.get("series_uid")
            self._update_displayed_highlight()
            self._return_focus()

    # ------------------------------------------------------------ file
    def _default_dir(self) -> str:
        """File-dialog start folder: the PARENT of the currently displayed image
        folder; if nothing is displayed/selected, the last folder a dialog used.
        Falls back to a row's image folder's parent, then "" (cwd)."""
        img = ""
        try:
            if hasattr(self._shell, "case_image_dir"):
                img = self._shell.case_image_dir() or ""
        except Exception:                            # noqa: BLE001
            img = ""
        if not img:                                  # nothing shown → a row's folder
            for r in self._rows:
                for d in (r.get("src_dirs") or []):
                    if d and os.path.isdir(d):
                        img = d
                        break
                if img:
                    break
        if img:
            parent = os.path.dirname(os.path.normpath(img))
            if parent and os.path.isdir(parent):
                return parent
        return self._last_dir or ""

    def _write_to(self, path: str) -> None:
        """Serialise the current presentation to *path* (JSON)."""
        data = {
            "version": 1,
            "reference": self._reference,
            "offsets": self._offsets,
            "rows": [{
                "series_uid": r.get("series_uid", ""),
                "modality": r.get("modality", ""),
                "number": r.get("number"),
                "frames": r.get("frames"),
                "size_mb": r.get("size_mb"),
                "pane_index": r.get("pane_index", 0),
                "date": r.get("date", ""),
                "time": r.get("time", ""),
                "comment": r.get("comment", ""),
                "label": r.get("label", ""),
                "src_dirs": r.get("src_dirs", []),
                "refreshed": bool(r.get("_refreshed")),
                "view_state": json_safe(r.get("view_state", {})),
            } for r in self._rows],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._last_path = path
            self._last_dir = os.path.dirname(path) or self._last_dir
            self._dirty = False
            self._hint.setText(t("保存しました: {p}", p=path))
        except OSError as exc:
            self._warn(t("保存に失敗しました: {e}", e=str(exc)))

    def _save(self) -> None:
        """名前を付けて保存 — opens in the parent of the displayed image folder
        (else the last-used folder)."""
        if not self._rows:
            self._warn(t("保存する行がありません。"))
            return
        # Reuse the same file if one is known, else a new file in the default dir.
        if self._last_path:
            default = self._last_path
        else:
            d = self._default_dir()
            default = os.path.join(d, "CasePresentation.json") if d \
                else "CasePresentation.json"
        path, _ = QFileDialog.getSaveFileName(
            self, t("Case Presentation を保存"), default, t("JSON (*.json)"))
        if not path:
            return
        self._write_to(path)

    def _save_overwrite(self) -> None:
        """上書き保存 — write straight to the last saved/loaded file (no dialog);
        falls back to 名前を付けて保存 when there is no such file yet."""
        if not self._rows:
            self._warn(t("保存する行がありません。"))
            return
        import os
        if self._last_path and os.path.isdir(os.path.dirname(self._last_path)
                                             or "."):
            self._write_to(self._last_path)
        else:
            self._save()

    # ------------------------------------------------------------ close
    def closeEvent(self, e) -> None:
        """Warn on unsaved changes before closing: 保存 / 終了 / キャンセル.

        This is a dock: closing HIDES it (the instance and its data persist, so
        reopening restores everything). The app-wide F/A filter is left
        installed for the panel's lifetime (it no-ops while hidden because focus
        can't be inside a hidden widget) so F/A still work after a reopen."""
        if not self._dirty or not self._rows:
            e.accept()
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(t("Case Presentation"))
        box.setText(t("このデータは未保存ですが、そのまま終了していいですか?"))
        b_save = box.addButton(t("保存"), QMessageBox.ButtonRole.AcceptRole)
        b_exit = box.addButton(t("終了"),
                               QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(t("キャンセル"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(b_save)
        box.exec()
        c = box.clickedButton()
        if c is b_save:
            self._save_overwrite()
            # If the Save-As dialog was cancelled, _dirty stays True → keep open.
            if self._dirty:
                e.ignore()
            else:
                e.accept()
        elif c is b_exit:
            e.accept()
        else:
            e.ignore()

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, t("Case Presentation を読込"), self._default_dir(),
            t("JSON (*.json)"))
        if not path:
            return
        self.load_file(path)

    def load_file(self, path: str) -> bool:
        """Load a Case Presentation *.json from *path* (also used by a drag&drop
        of the file onto the shell): populate the rows and offer to open the
        related folders in the background so the 表示 buttons work."""
        self._last_dir = os.path.dirname(path) or self._last_dir
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            self._warn(t("読込に失敗しました: {e}", e=str(exc)))
            return False
        # A 読込 REPLACES the whole list. Snapshot the current state first so
        # Ctrl+Z can bring back the (possibly unsaved) work that was here — but
        # only when there IS existing content to lose (a load into an empty
        # panel needs no undo point). A failed parse above returns before this,
        # so a bad file never disturbs the current list or the undo history.
        if self._rows:
            self._record_undo()
        self._reference = data.get("reference", "XA")
        self._offsets = {k: float(v) for k, v in
                         (data.get("offsets") or {}).items()}
        self._displayed_uid = None          # nothing shown yet after a load
        self._rows = []
        for r in data.get("rows", []):
            if not isinstance(r, dict):
                continue
            self._rows.append({
                "series_uid": r.get("series_uid", ""),
                "modality": r.get("modality", ""),
                "number": r.get("number"),
                "frames": r.get("frames"),
                "size_mb": r.get("size_mb"),
                "pane_index": r.get("pane_index", 0),
                "date": r.get("date", ""),
                "time": r.get("time", ""),
                "comment": r.get("comment", ""),
                "label": r.get("label", ""),
                "src_dirs": r.get("src_dirs", []),
                "_refreshed": bool(r.get("refreshed")),
                "view_state": _cap_depth(r.get("view_state", {})),
            })
        self._last_path = path          # 上書き保存 targets the loaded file
        self._dirty = False             # freshly loaded = matches the file
        self._refresh_ref_combo()
        self._rebuild()
        self._hint.setText(t("読込みました: {p}", p=path))
        self._offer_open_missing()
        return True

    def _offer_open_missing(self) -> None:
        """After a load, offer to re-scan the folders of any series that aren't
        currently loaded, so their 表示 buttons work."""
        missing = [r for r in self._rows
                   if not self._shell.case_series_loaded(r.get("series_uid", ""))]
        if not missing:
            return
        dirs = []
        for r in missing:
            for d in r.get("src_dirs", []):
                if d and d not in dirs:
                    dirs.append(d)
        if not dirs:
            self._warn(t(
                "未読込のシリーズが {n} 件ありますが、保存に元フォルダ情報が"
                "無いため自動で開けません。元のDICOMフォルダを開いてください。",
                n=len(missing)))
            return
        ans = QMessageBox.question(
            self, t("Case Presentation"),
            t("未読込のシリーズが {n} 件あります。関連フォルダ {m} 個を開いて"
              "読み込みますか?", n=len(missing), m=len(dirs)))
        if ans == QMessageBox.StandardButton.Yes:
            opened = self._shell.case_open_folders(dirs)
            self._hint.setText(t(
                "{m} 個のフォルダを読み込み中… 完了後に「状態更新」を押すと"
                "[表示]が有効になります。", m=opened))

    def _relocate_folders(self) -> None:
        """Re-point the presentation at moved / renamed image folders. Pick a
        folder; it is re-scanned recursively and the series RE-BIND by their
        SeriesInstanceUID (path-independent), so "見当たりません" rows become
        displayable again. The chosen folder is recorded on the still-missing
        rows so a re-save keeps the new location."""
        from PyQt6.QtWidgets import QFileDialog
        start = os.path.dirname(self._last_path) if self._last_path else \
            self._last_dir
        d = QFileDialog.getExistingDirectory(
            self, t("画像フォルダを再指定（配下を再帰的に探索）"), start or "")
        if not d:
            return
        missing = [r for r in self._rows
                   if not self._shell.case_series_loaded(r.get("series_uid", ""))]
        opened = self._shell.case_open_folders([d])
        if not opened:
            self._warn(t("そのフォルダには読み込める画像がありませんでした。"))
            return
        # Record the new folder on the rows that were missing (best-effort), so a
        # re-save points at the new location next time.
        for r in missing:
            sd = r.setdefault("src_dirs", [])
            if d not in sd:
                sd.append(d)
        self._dirty = True
        self._hint.setText(t(
            "フォルダを再指定して読込中… 完了後に右クリック →「状態更新」で "
            "[表示] が有効になります（SeriesUID で自動再結合、{n} 件対象）。",
            n=len(missing)))

    # ----------------------------------------------------------- helpers
    def _on_ref_changed(self, text: str) -> None:
        if text and text != self._reference:
            self._record_undo()
            self._reference = text
            self._dirty = True
            self._rebuild()

    def _refresh_ref_combo(self) -> None:
        mods = self._present_modalities()
        if self._reference not in mods:
            mods = ([self._reference] + mods) if self._reference else mods
        self._ref_combo.blockSignals(True)
        self._ref_combo.clear()
        self._ref_combo.addItems(mods or ["XA"])
        if self._reference in mods:
            self._ref_combo.setCurrentText(self._reference)
        self._ref_combo.blockSignals(False)

    def _after_rows_changed(self, select_last: bool = False) -> None:
        self._dirty = True
        self._refresh_ref_combo()
        self._rebuild(select=(len(self._rows) - 1) if select_last else None)

    def _selected_row_indices(self) -> list:
        return sorted({idx.row() for idx in self._table.selectionModel()
                       .selectedRows()}) if self._table.selectionModel() else []

    def _show_at_index(self, idx: int | None) -> None:
        """After a 除外/削除 removes rows, show the series that is now at *idx* —
        i.e. the one just BELOW the removed row. If the removed row was the last,
        *idx* clamps to the new last row (one above). No-op on an empty list."""
        if idx is None or not self._rows:
            return
        idx = max(0, min(idx, len(self._rows) - 1))
        self._show_row(self._rows[idx])

    def _warn(self, msg: str) -> None:
        QMessageBox.information(self, t("Case Presentation"), msg)

    # ---- dirty flag (drives the 上書き保存 "saved" cue) --------------------
    @property
    def _dirty(self) -> bool:
        return getattr(self, "_dirty_flag", False)

    @_dirty.setter
    def _dirty(self, val: bool) -> None:
        self._dirty_flag = bool(val)
        self._update_overwrite_style()

    def _update_overwrite_style(self) -> None:
        """Give 上書き保存 a green background while the presentation matches its
        file on disk (saved, no unsaved edits), so 'this is already saved' is
        obvious at a glance. Any edit clears it back to the normal button."""
        b = getattr(self, "_b_overwrite", None)
        if b is None:                       # called before the button exists
            return
        if not self._dirty_flag and self._last_path:
            b.setStyleSheet("background-color:#cdeccd;")
            b.setToolTip(t("保存済み（未変更）— 直前のファイルへ上書き保存"))
        else:
            b.setStyleSheet("")
            b.setToolTip(t("直前に保存/読込したファイルへ上書き保存"))

    @staticmethod
    def _cell_btn_css(bar: bool = False, grey: bool = False,
                      bg: bool = False) -> str:
        """Stylesheet for a per-row action button. *bar* draws the left 縦棒
        status indicator; *bg* fills a green "done" background (used on 更新 once
        the row has been updated); *grey* dims the label. No flags → empty string
        = native button (no bar/background), used for 除外・削除 always."""
        rules = ""
        if grey:
            rules += "color:#999;"
        if bg:
            rules += "background-color:#d7f0d7;"
        if bar:
            rules += "border-left:4px solid #1e6fd0;"
        return ("QPushButton{" + rules + "}") if rules else ""

    # ------------------------------------------------------------ render
    def _rebuild(self, select: int | None = None) -> None:
        self._building = True
        tb = self._table
        tb.blockSignals(True)
        tb.setRowCount(0)
        tb.setRowCount(len(self._rows))
        for i, r in enumerate(self._rows):
            uni = self._unified_for(r)
            is_ref = r.get("modality") == self._reference
            ro = QTableWidgetItem(str(i + 1))
            ro.setFlags(ro.flags() & ~Qt.ItemFlag.ItemIsEditable)
            tb.setItem(i, C_NO, ro)
            frames = r.get("frames")
            size_mb = r.get("size_mb")
            for col, val in ((C_MOD, r.get("modality", "")),
                             (C_SER, "" if r.get("number") is None
                              else str(r.get("number"))),
                             (C_FRAMES, "" if not frames else str(int(frames))),
                             (C_SIZE, "" if size_mb in (None, 0)
                              else f"{float(size_mb):.2f}"),
                             (C_TIME, _fmt_raw_time(r.get("time", ""))),
                             (C_UNI, _fmt_secs(uni))):
                it = QTableWidgetItem(str(val))
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col == C_UNI and is_ref:
                    it.setForeground(QColor("#1e6fd0"))
                tb.setItem(i, col, it)
            cm = QTableWidgetItem(r.get("comment", ""))
            if not r.get("comment", "").strip():
                cm.setBackground(QColor(255, 235, 235))    # empty = must fill
            tb.setItem(i, C_COMMENT, cm)
            # Status 縦棒: 表示 carries it until the row has been 更新'd, then it
            # moves to 更新 — so at a glance a bar on 表示 = "still to review",
            # a bar on 更新 = "done". 除外・削除 never get one.
            refreshed = bool(r.get("_refreshed"))
            loaded = self._shell.case_series_loaded(r.get("series_uid", ""))
            btn = QPushButton(t("表示"))
            # 表示 text stays black even for a not-yet-loaded series (loading is
            # backgrounded on a CasePresentation.json drop, so the row can render
            # before the series is indexed). The unloaded state is conveyed by the
            # tooltip only, not by greying the label.
            btn.setStyleSheet(self._cell_btn_css(bar=not refreshed))
            if not loaded:
                btn.setToolTip(t(
                    "未読込 — 「読込」時に自動で開くか、元フォルダを開いて"
                    "から「状態更新」を押してください"))
            btn.clicked.connect(lambda _c, row=r: self._show_row(row))
            tb.setCellWidget(i, C_SHOW, btn)
            # Per-row 更新 (re-check load state) / 削除 (remove this row) — saves
            # reaching the top toolbar or the right-click menu.
            b_upd = QPushButton(t("更新"))
            b_upd.setStyleSheet(self._cell_btn_css(bar=refreshed, bg=refreshed))
            b_upd.setToolTip(t("この行の読込状態を再確認"))
            b_upd.clicked.connect(lambda _c, row=r: self._refresh_row(row))
            tb.setCellWidget(i, C_UPD, b_upd)
            b_del = QPushButton(t("除外"))
            b_del.setToolTip(t("この行をプレゼンから除外（元ファイルは残す）"))
            b_del.clicked.connect(lambda _c, row=r: self._delete_row(row))
            tb.setCellWidget(i, C_DEL, b_del)
            b_erase = QPushButton(t("削除"))
            b_erase.setToolTip(t(
                "このシリーズの元ファイルを、画像フォルダの親にある "
                "CasePresentation-Erase フォルダへ移動（元に戻せます）"))
            b_erase.clicked.connect(lambda _c, row=r: self._erase_row(row))
            tb.setCellWidget(i, C_ERASE, b_erase)
        tb.blockSignals(False)
        self._building = False
        if select is not None and 0 <= select < len(self._rows):
            tb.selectRow(select)
        n = len(self._rows)
        empties = sum(1 for r in self._rows if not r.get("comment", "").strip())
        self._hint.setText(t(
            "{n} 行 / コメント未入力 {e} 行 (赤い欄にコメントを入力)。"
            "  基準: {ref}", n=n, e=empties, ref=self._reference))
        self._update_displayed_highlight()

    def _update_displayed_highlight(self) -> None:
        """Persistently tint the CURRENTLY-DISPLAYED series' row light blue,
        independent of the table's selection highlight. The selection blue is
        focus-dependent — it fades when you click into the viewer / scrub the
        seekbar — whereas this per-item background stays put, so "which series is
        on screen" remains visible during image operations."""
        disp = self._displayed_uid
        blue = QColor("#cfe4ff")
        clear = QBrush()                      # NoBrush → view default
        red = QColor(255, 235, 235)
        tb = self._table
        was = tb.blockSignals(True)           # setBackground emits cellChanged —
        try:                                  # block it so _on_cell_changed (undo
            for i, r in enumerate(self._rows):  # snapshot, dirty…) isn't triggered
                is_disp = bool(disp) and r.get("series_uid") == disp
                for c in (C_NO, C_MOD, C_SER, C_FRAMES, C_SIZE, C_TIME, C_UNI):
                    it = tb.item(i, c)
                    if it is not None:
                        it.setBackground(blue if is_disp else clear)
                cm = tb.item(i, C_COMMENT)
                if cm is not None:
                    if not (r.get("comment", "") or "").strip():
                        cm.setBackground(red)  # empty-comment warning wins
                    else:
                        cm.setBackground(blue if is_disp else clear)
                # 更新 only makes sense for the series whose image is on screen —
                # capturing the "current view" of a series that isn't displayed
                # does nothing. Enable it ONLY on the displayed row.
                upd = tb.cellWidget(i, C_UPD)
                if upd is not None:
                    upd.setEnabled(is_disp)
                    upd.setToolTip(
                        t("この行の読込状態を再確認") if is_disp
                        else t("表示中のシリーズのみ更新できます（先に「表示」）"))
        finally:
            tb.blockSignals(was)

    def _on_cell_changed(self, row: int, col: int) -> None:
        if getattr(self, "_building", False) or col != C_COMMENT:
            return
        if 0 <= row < len(self._rows):
            item = self._table.item(row, col)
            self._record_undo()
            self._rows[row]["comment"] = item.text() if item else ""
            self._dirty = True
            # Writing findings for this row → its live image IS the key image;
            # capture it in the background so 表示 restores it later.
            self._capture_view_into(self._rows[row])
            # update the empty-highlight + counters without full rebuild churn
            if item is not None:
                item.setBackground(QColor(255, 255, 255)
                                   if self._rows[row]["comment"].strip()
                                   else QColor(255, 235, 235))
            # keep the displayed-row tint correct after editing its comment
            self._update_displayed_highlight()
