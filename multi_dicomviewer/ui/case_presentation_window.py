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

import json
import os
from datetime import datetime, timedelta

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.core.case_presentation import (
    json_safe, modified_sort_order, offset_from_anchor, parse_dcm_dt,
    unified_time)
from multi_dicomviewer.i18n import t

# Column layout
C_NO, C_MOD, C_SER, C_TIME, C_UNI, C_COMMENT, C_SHOW = range(7)
_HEADERS = ["No", "種別", "Ser", "時間", "統合時間", "コメント", ""]
_SNAP_TOL_S = 10.0            # ±seconds: snap a non-ref event just after an XA


def _fmt_raw_time(tm: str) -> str:
    """DICOM TM 'HHMMSS[.ffffff]' → 'HH:MM:SS' (best effort)."""
    tm = (tm or "").strip().replace(":", "")
    if len(tm) < 2:
        return ""
    hh = tm[0:2]
    mm = tm[2:4] if len(tm) >= 4 else "00"
    ss = tm[4:6] if len(tm) >= 6 else "00"
    return f"{hh}:{mm}:{ss}"


def _fmt_secs(sec) -> str:
    """Seconds-since-epoch (from parse_dcm_dt) → 'HH:MM:SS'."""
    if sec is None:
        return ""
    try:
        dt = datetime(1970, 1, 1) + timedelta(seconds=float(sec))
        return dt.strftime("%H:%M:%S")
    except (ValueError, OverflowError):
        return ""


class _OffsetDialog(QDialog):
    """Manual per-modality offset entry (seconds; + = that modality's clock is
    behind the reference)."""

    def __init__(self, modalities, offsets, reference, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("時刻オフセット (手入力)"))
        root = QVBoxLayout(self)
        root.addWidget(QLabel(t(
            "基準「{ref}」に対する各モダリティの時刻ズレ (秒)。\n"
            "＋ = そのモダリティの時計が基準より遅れている。", ref=reference)))
        form = QFormLayout()
        self._spins = {}
        for mod in modalities:
            sb = QDoubleSpinBox()
            sb.setRange(-86400.0, 86400.0)
            sb.setDecimals(1)
            sb.setSingleStep(1.0)
            sb.setSuffix(t(" 秒"))
            sb.setValue(float(offsets.get(mod, 0.0)))
            self._spins[mod] = sb
            form.addRow(mod, sb)
        root.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def values(self) -> dict:
        return {m: sb.value() for m, sb in self._spins.items()}


class CasePresentationWindow(QMainWindow):
    """Non-modal tool window; one instance kept by the shell."""

    def __init__(self, shell, parent=None):
        super().__init__(parent)
        self._shell = shell
        self._rows: list[dict] = []
        self._offsets: dict[str, float] = {}       # modality → seconds
        self._reference = "XA"
        self._last_path: str | None = None         # for 上書き保存 (overwrite)
        self._dirty = False                         # unsaved changes → close warns
        self.setWindowTitle(t("Case Presentation"))
        self.resize(900, 520)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # -- toolbar row 1: capture / reference / sort --------------------
        bar1 = QHBoxLayout()
        b_add = QPushButton(t("この表示を追加"))
        b_add.setToolTip(t("アクティブなペインの表示を1行として取り込む"))
        b_add.clicked.connect(self._add_active)
        bar1.addWidget(b_add)
        b_add_all = QPushButton(t("全ペインを追加"))
        b_add_all.setToolTip(t("表示中の全ペインをそれぞれ1行として取り込む"))
        b_add_all.clicked.connect(self._add_all)
        bar1.addWidget(b_add_all)
        b_add_study = QPushButton(t("全シリーズを追加"))
        b_add_study.setToolTip(t(
            "選択中の検査の全シリーズを種別/Ser/時間つきで取り込む "
            "(表示は各シリーズの自動フレーム)"))
        b_add_study.clicked.connect(self._add_all_series)
        bar1.addWidget(b_add_study)
        bar1.addSpacing(16)
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
        bar1.addStretch(1)
        outer.addLayout(bar1)

        # -- toolbar row 2: offset / reorder / file -----------------------
        bar2 = QHBoxLayout()
        b_anchor = QPushButton(t("アンカーで揃える"))
        b_anchor.setToolTip(t(
            "同一時点とみなす基準行と他モダリティ行を1行ずつ選択 → その"
            "モダリティの時刻オフセットを自動計算"))
        b_anchor.clicked.connect(self._anchor_align)
        bar2.addWidget(b_anchor)
        b_off = QPushButton(t("オフセット手入力…"))
        b_off.clicked.connect(self._edit_offsets)
        bar2.addWidget(b_off)
        bar2.addSpacing(16)
        b_up = QPushButton("▲")
        b_up.setToolTip(t("選択行を上へ"))
        b_up.clicked.connect(lambda: self._move(-1))
        bar2.addWidget(b_up)
        b_down = QPushButton("▼")
        b_down.setToolTip(t("選択行を下へ"))
        b_down.clicked.connect(lambda: self._move(+1))
        bar2.addWidget(b_down)
        b_refresh = QPushButton(t("状態更新"))
        b_refresh.setToolTip(t("各行の読込状態を再確認 (フォルダ読込完了後に押す)"))
        b_refresh.clicked.connect(lambda: self._rebuild())
        bar2.addWidget(b_refresh)
        b_del = QPushButton(t("削除"))
        b_del.clicked.connect(self._delete_selected)
        bar2.addWidget(b_del)
        bar2.addStretch(1)
        b_overwrite = QPushButton(t("上書き保存"))
        b_overwrite.setToolTip(t("直前に保存/読込したファイルへ上書き保存"))
        b_overwrite.clicked.connect(self._save_overwrite)
        bar2.addWidget(b_overwrite)
        b_save = QPushButton(t("名前を付けて保存…"))
        b_save.clicked.connect(self._save)
        bar2.addWidget(b_save)
        b_load = QPushButton(t("読込…"))
        b_load.clicked.connect(self._load)
        bar2.addWidget(b_load)
        b_clear = QPushButton(t("全消去"))
        b_clear.clicked.connect(self._clear_all)
        bar2.addWidget(b_clear)
        outer.addLayout(bar2)

        # -- table --------------------------------------------------------
        self._table = QTableWidget(0, len(_HEADERS))
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
        for c in (C_NO, C_MOD, C_SER, C_TIME, C_UNI, C_SHOW):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(C_COMMENT, QHeaderView.ResizeMode.Stretch)
        for c, w in ((C_NO, 44), (C_MOD, 70), (C_SER, 56), (C_TIME, 92),
                     (C_UNI, 92), (C_SHOW, 64)):
            self._table.setColumnWidth(c, w)
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

    # ---------------------------------------------------------------- add
    def _add_active(self) -> None:
        row = self._shell.case_capture_active()
        if row is None:
            self._warn(t("表示中のシリーズがありません。"))
            return
        self._rows.append(row)
        self._after_rows_changed(select_last=True)

    def _add_all(self) -> None:
        rows = self._shell.case_capture_all()
        if not rows:
            self._warn(t("表示中のシリーズがありません。"))
            return
        self._rows.extend(rows)
        self._after_rows_changed(select_last=True)

    def _add_all_series(self) -> None:
        """全シリーズを追加: one row per series of the selected study (種別/Ser/
        時間 auto). Skips series already present so re-pressing doesn't duplicate."""
        rows = self._shell.case_capture_all_series()
        if not rows:
            self._warn(t("シリーズが見つかりません (検査を表示してから押してください)。"))
            return
        have = {r.get("series_uid") for r in self._rows if r.get("series_uid")}
        added = 0
        for r in rows:
            if r.get("series_uid") and r["series_uid"] in have:
                continue
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
            self._offsets.update(dlg.values())
            self._dirty = True
            self._rebuild()

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
        self._offsets[oth["modality"]] = off
        self._hint.setText(t(
            "{mod} のオフセットを {sec:+.1f} 秒に設定しました。",
            mod=oth["modality"], sec=off))
        self._dirty = True
        self._rebuild()

    # -------------------------------------------------------------- sort
    def _unified_for(self, row) -> float | None:
        dt = parse_dcm_dt(row.get("date", ""), row.get("time", ""))
        return unified_time(dt, row.get("modality", ""), self._reference,
                            self._offsets)

    def _sort_rows(self) -> None:
        items = [{"dt": self._unified_for(r),
                  "is_ref": r.get("modality") == self._reference}
                 for r in self._rows]
        order = modified_sort_order(items, tol=_SNAP_TOL_S)
        self._rows = [self._rows[i] for i in order]
        self._dirty = True
        self._rebuild()

    # ------------------------------------------------------------ reorder
    def _move(self, delta: int) -> None:
        sel = self._selected_row_indices()
        if len(sel) != 1:
            return
        i = sel[0]
        j = i + delta
        if not (0 <= j < len(self._rows)):
            return
        self._rows[i], self._rows[j] = self._rows[j], self._rows[i]
        self._dirty = True
        self._rebuild(select=j)

    def _delete_selected(self) -> None:
        sel = set(self._selected_row_indices())
        if not sel:
            return
        self._rows = [r for i, r in enumerate(self._rows) if i not in sel]
        self._after_rows_changed()

    def _row_menu(self, pos) -> None:
        """Row right-click menu: 状態更新 / 削除."""
        idx = self._table.indexAt(pos)
        if idx.isValid():
            r = idx.row()
            if r not in set(self._selected_row_indices()):
                self._table.selectRow(r)      # right-click selects the row
        menu = QMenu(self)
        a_ref = menu.addAction(t("状態更新"))
        a_del = menu.addAction(t("削除"))
        chosen = menu.exec(self._table.viewport().mapToGlobal(pos))
        if chosen is a_ref:
            self._rebuild()
        elif chosen is a_del:
            self._delete_selected()

    def _clear_all(self) -> None:
        if not self._rows:
            return
        if QMessageBox.question(
                self, t("全消去"),
                t("全ての行を消去しますか?")) == QMessageBox.StandardButton.Yes:
            self._rows = []
            self._after_rows_changed()

    # ---------------------------------------------------------- display
    def _show_row(self, row) -> None:
        ok = self._shell.case_redisplay(row)
        if not ok:
            self._warn(t(
                "このシリーズは現在読み込まれていません "
                "(閉じられた可能性があります)。元のフォルダを開き直してください。"))

    # ------------------------------------------------------------ file
    def _default_save_dir(self) -> str:
        """Default save folder = the folder the image data lives in (the first
        existing source folder recorded on any row)."""
        import os
        for r in self._rows:
            for d in (r.get("src_dirs") or []):
                if d and os.path.isdir(d):
                    return d
        return ""

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
                "pane_index": r.get("pane_index", 0),
                "date": r.get("date", ""),
                "time": r.get("time", ""),
                "comment": r.get("comment", ""),
                "label": r.get("label", ""),
                "src_dirs": r.get("src_dirs", []),
                "view_state": json_safe(r.get("view_state", {})),
            } for r in self._rows],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._last_path = path
            self._dirty = False
            self._hint.setText(t("保存しました: {p}", p=path))
        except OSError as exc:
            self._warn(t("保存に失敗しました: {e}", e=str(exc)))

    def _save(self) -> None:
        """名前を付けて保存 — defaults to the image-data folder."""
        import os
        if not self._rows:
            self._warn(t("保存する行がありません。"))
            return
        # Default: overwrite the same file if one is known, else a new file in
        # the image-data folder.
        if self._last_path:
            default = self._last_path
        else:
            d = self._default_save_dir()
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
        """Warn on unsaved changes before closing: 保存 / 終了 / キャンセル."""
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
            self, t("Case Presentation を読込"), "", t("JSON (*.json)"))
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            self._warn(t("読込に失敗しました: {e}", e=str(exc)))
            return
        self._reference = data.get("reference", "XA")
        self._offsets = {k: float(v) for k, v in
                         (data.get("offsets") or {}).items()}
        self._rows = []
        for r in data.get("rows", []):
            if not isinstance(r, dict):
                continue
            self._rows.append({
                "series_uid": r.get("series_uid", ""),
                "modality": r.get("modality", ""),
                "number": r.get("number"),
                "pane_index": r.get("pane_index", 0),
                "date": r.get("date", ""),
                "time": r.get("time", ""),
                "comment": r.get("comment", ""),
                "label": r.get("label", ""),
                "src_dirs": r.get("src_dirs", []),
                "view_state": r.get("view_state", {}),
            })
        self._last_path = path          # 上書き保存 targets the loaded file
        self._dirty = False             # freshly loaded = matches the file
        self._refresh_ref_combo()
        self._rebuild()
        self._hint.setText(t("読込みました: {p}", p=path))
        self._offer_open_missing()

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

    # ----------------------------------------------------------- helpers
    def _on_ref_changed(self, text: str) -> None:
        if text and text != self._reference:
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

    def _warn(self, msg: str) -> None:
        QMessageBox.information(self, t("Case Presentation"), msg)

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
            for col, val in ((C_MOD, r.get("modality", "")),
                             (C_SER, "" if r.get("number") is None
                              else str(r.get("number"))),
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
            btn = QPushButton(t("表示"))
            if not self._shell.case_series_loaded(r.get("series_uid", "")):
                btn.setStyleSheet("color:#999;")
                btn.setToolTip(t(
                    "未読込 — 「読込」時に自動で開くか、元フォルダを開いて"
                    "から「状態更新」を押してください"))
            btn.clicked.connect(lambda _c, row=r: self._show_row(row))
            tb.setCellWidget(i, C_SHOW, btn)
        tb.blockSignals(False)
        self._building = False
        if select is not None and 0 <= select < len(self._rows):
            tb.selectRow(select)
        n = len(self._rows)
        empties = sum(1 for r in self._rows if not r.get("comment", "").strip())
        self._hint.setText(t(
            "{n} 行 / コメント未入力 {e} 行 (赤い欄にコメントを入力)。"
            "  基準: {ref}", n=n, e=empties, ref=self._reference))

    def _on_cell_changed(self, row: int, col: int) -> None:
        if getattr(self, "_building", False) or col != C_COMMENT:
            return
        if 0 <= row < len(self._rows):
            item = self._table.item(row, col)
            self._rows[row]["comment"] = item.text() if item else ""
            self._dirty = True
            # update the empty-highlight + counters without full rebuild churn
            if item is not None:
                item.setBackground(QColor(255, 255, 255)
                                   if self._rows[row]["comment"].strip()
                                   else QColor(255, 235, 235))
