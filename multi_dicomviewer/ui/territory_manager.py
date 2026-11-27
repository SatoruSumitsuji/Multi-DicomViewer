# -*- coding: utf-8 -*-
"""Territory Manage — a SEPARATE window for the Territory targets (user
2026-11-27): the CoronaryTree tab's target list is small and easy to miss.

One row per target: show/hide (checkbox), No., 名称 (editable — the name then
replaces 「Target-N」 in the right pane's bottom-right result), 心筋%, 灌流域 mL,
元血管名, ターゲット位置 (from the vessel's proximal end, % and mm). Below:
show / hide every target, delete the selected one. Opened from row 3's
「Territory Manage」 or a right-click on the target result of the VR pane.
The data stay in the CoronaryTreeWindow (*owner*); this window only edits
them through its methods and is refreshed whenever its target table is.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QDialog, QHBoxLayout,
                             QHeaderView, QLabel, QPushButton, QTableWidget,
                             QTableWidgetItem, QVBoxLayout)

from multi_dicomviewer.i18n import t

_C_SHOW, _C_NO, _C_NAME, _C_PCT, _C_ML, _C_VES, _C_POS = range(7)


class TerritoryManagerDialog(QDialog):
    def __init__(self, owner, parent=None):
        super().__init__(parent or owner)
        self._owner = owner
        self._filling = False
        self.setWindowTitle(t("Territory Manage"))
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setModal(False)
        self.resize(760, 300)
        lay = QVBoxLayout(self)
        self._msg = QLabel("")
        self._msg.setStyleSheet("color:#a05000;")
        lay.addWidget(self._msg)
        self._tbl = QTableWidget(0, 7, self)
        self._tbl.setHorizontalHeaderLabels([
            t("表示"), t("No."), t("名称"), t("心筋%"), t("灌流域mL"),
            t("元血管名"), t("ターゲット位置")])
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._tbl.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._tbl.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed)
        f = self._tbl.font()
        f.setPointSizeF(max(f.pointSizeF(), 11.0))
        self._tbl.setFont(f)
        # every column width can be dragged (user 2026-11-27); sized to the
        # contents once, then left to the user
        hh = self._tbl.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setStretchLastSection(True)
        hh.setMinimumSectionSize(36)
        self._sized = False
        self._tbl.itemChanged.connect(self._item_changed)
        lay.addWidget(self._tbl, 1)
        tip = QLabel(t("名称をダブルクリックで入力すると、右ペイン右下の「Target-N」"
                       "がその名前で表示されます。"))
        tip.setStyleSheet("color:#555;")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        row = QHBoxLayout()
        self._all_btn = QPushButton("")
        self._all_btn.clicked.connect(self._toggle_all)
        row.addWidget(self._all_btn)
        self._del_btn = QPushButton(t("選択したターゲットを削除"))
        self._del_btn.clicked.connect(self._delete_selected)
        row.addWidget(self._del_btn)
        row.addStretch(1)
        close = QPushButton(t("閉じる"))
        close.clicked.connect(self.close)
        row.addWidget(close)
        lay.addLayout(row)
        self.refresh()

    # ------------------------------------------------------------- content
    @staticmethod
    def _pos_text(tree, vid, idx: int) -> str:
        """Target position from the vessel's PROXIMAL end (the connected end,
        whatever the drawn direction): % of its length and mm."""
        v = tree.vessels.get(vid) if tree is not None else None
        if v is None or v.n < 2:
            return "—"
        pts = np.asarray(v.points, float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        idx = max(0, min(int(idx), len(pts) - 1))
        try:
            prox = int(tree.proximal_idx(vid))
        except Exception:                                # noqa: BLE001
            prox = 0
        if prox == 0:
            mm, k = float(seg[:idx].sum()), idx
        else:
            mm, k = float(seg[idx:].sum()), len(pts) - 1 - idx
        pct = 100.0 * k / (len(pts) - 1)
        return f"{pct:.0f}%  ({mm:.1f} mm)"

    def refresh(self) -> None:
        o = self._owner
        targets = list(getattr(o, "_targets", []) or [])
        calc = getattr(o, "_territory", None) is not None
        tree = getattr(o, "_tree", None)
        if not targets:
            self._msg.setText(t(
                "ターゲットはまだありません（Territory 計算後、右ペインの冠動脈を"
                "右クリックして設定）。") if calc else t(
                "Territory が計算されていません（3行目の「Territory Calc」）。"))
        else:
            self._msg.setText("" if calc else t(
                "Territory 未計算のため 心筋% / 灌流域 は「—」です。"))
        self._filling = True
        try:
            self._tbl.setRowCount(len(targets))
            for r, tg in enumerate(targets):
                v = tree.vessels.get(tg.get("vid")) if tree is not None else None
                hidden = bool(tg.get("hidden", False))
                show = QTableWidgetItem("")
                show.setFlags(Qt.ItemFlag.ItemIsUserCheckable
                              | Qt.ItemFlag.ItemIsEnabled
                              | Qt.ItemFlag.ItemIsSelectable)
                show.setCheckState(Qt.CheckState.Unchecked if hidden
                                   else Qt.CheckState.Checked)
                cells = [show,
                         QTableWidgetItem(str(r + 1)),
                         QTableWidgetItem(str(tg.get("name", "") or "")),
                         QTableWidgetItem(f"{tg.get('pct', 0.0):.1f}%"
                                          if calc else "—"),
                         QTableWidgetItem(f"{tg.get('ml', 0.0):.1f}"
                                          if calc else "—"),
                         QTableWidgetItem(v.name if v is not None
                                          else str(tg.get("vid", ""))),
                         QTableWidgetItem(self._pos_text(tree, tg.get("vid"), tg.get("idx", 0)))]
                for c, it in enumerate(cells):
                    if c not in (_C_SHOW, _C_NAME):
                        it.setFlags(Qt.ItemFlag.ItemIsEnabled
                                    | Qt.ItemFlag.ItemIsSelectable)
                    if c in (_C_NO, _C_PCT, _C_ML):
                        it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if c == _C_NAME:
                        it.setToolTip(t("ダブルクリックで名称を入力"))
                    if hidden:
                        it.setForeground(QColor("#999999"))
                    self._tbl.setItem(r, c, it)
                no = self._tbl.item(r, _C_NO)
                try:
                    no.setForeground(QColor(o._target_hex(r + 1)))
                    fb = no.font(); fb.setBold(True); no.setFont(fb)
                except Exception:                        # noqa: BLE001
                    pass
        finally:
            self._filling = False
        if targets and not self._sized:          # first fill: fit, then free
            self._tbl.resizeColumnsToContents()
            if self._tbl.columnWidth(_C_NAME) < 160:
                self._tbl.setColumnWidth(_C_NAME, 160)
            self._sized = True
        # the button follows the per-target flags (what is drawn)
        shown = bool(getattr(o, "_targets_shown", True)) and any(
            not tg.get("hidden", False) for tg in targets)
        self._all_btn.setText(t("全ターゲットを非表示") if shown
                              else t("全ターゲットを表示"))
        self._all_btn.setEnabled(bool(targets))
        self._del_btn.setEnabled(bool(targets))

    # ------------------------------------------------------------- edits
    def _item_changed(self, it) -> None:
        if self._filling:
            return
        r, c = it.row(), it.column()
        o = self._owner
        if c == _C_SHOW:
            targets = getattr(o, "_targets", []) or []
            if 0 <= r < len(targets):
                want_hidden = it.checkState() != Qt.CheckState.Checked
                if bool(targets[r].get("hidden", False)) != want_hidden:
                    from PyQt6.QtCore import QTimer
                    QTimer.singleShot(0, lambda n=r + 1: o.toggle_target_n(n))
        elif c == _C_NAME:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda n=r + 1, s=it.text():
                              o.set_target_name(n, s))

    def _toggle_all(self) -> None:
        self._owner._toggle_targets_shown()
        self.refresh()

    def _delete_selected(self) -> None:
        r = self._tbl.currentRow()
        if r >= 0:
            self._owner.delete_target_n(r + 1)
