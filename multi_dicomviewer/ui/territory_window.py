"""Tools ▸ Territory — CT coronary perfusion-territory analysis.

Two inputs: a ``.corotree.json`` (the coronary tree — centrelines + roles) and a
``.FullLv.json`` (the Epi + Endo/Blood bundle → myocardium). Drop them on this
panel or pick them with the buttons, press 解析, and it assigns every myocardial
voxel (Epi ∧ ¬Endo) to its nearest coronary centreline and reports the
perfusion-territory volumes — per root vessel (whole subtree), the LEFT-system
sum (LM+LAD+LCX), and the per-vessel breakdown. Headless集計 for now; a coloured
CT overlay can build on the same TerritoryEngine later.
"""
from __future__ import annotations

import json
import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QDockWidget, QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from multi_dicomviewer.i18n import t
from multi_dicomviewer.core import full_lv
from multi_dicomviewer.core.coronary_territory import (
    CoronaryTree, format_territory_report)
from multi_dicomviewer.core.lv_function import LVFunction


class TerritoryWindow(QDockWidget):
    """Dockable panel that runs the two-file CT Territory report."""

    def __init__(self, shell):
        super().__init__(t("CT Territory"), shell)
        self._shell = shell
        self._corotree_path = ""
        self._full_path = ""
        self.setObjectName("TerritoryWindow")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        v.addWidget(QLabel(
            t("corotree.json と FullLv.json を指定（またはドラッグ＆ドロップ）"
              "して「解析」を押してください。")))

        # --- corotree row ---
        r1 = QHBoxLayout()
        self._corotree_lbl = QLabel(t("冠動脈ツリー: 未選択"))
        self._corotree_lbl.setWordWrap(False)
        b1 = QPushButton(t("corotree 選択…"))
        b1.clicked.connect(self._pick_corotree)
        r1.addWidget(self._corotree_lbl, 1)
        r1.addWidget(b1)
        v.addLayout(r1)

        # --- FullLv row ---
        r2 = QHBoxLayout()
        self._full_lbl = QLabel(t("LV (FullLv): 未選択"))
        b2 = QPushButton(t("FullLv 選択…"))
        b2.clicked.connect(self._pick_full)
        r2.addWidget(self._full_lbl, 1)
        r2.addWidget(b2)
        v.addLayout(r2)

        # --- max-dist cap ---
        r3 = QHBoxLayout()
        self._cap_chk = QCheckBox(t("割当上限距離"))
        self._cap_chk.setToolTip(t(
            "オンにすると、どの中心線からもこの距離を超える心筋voxelは"
            "「未割当」として集計します（オフ＝全voxelを最寄り血管へ）。"))
        self._cap_spin = QDoubleSpinBox()
        self._cap_spin.setRange(1.0, 100.0)
        self._cap_spin.setValue(10.0)
        self._cap_spin.setSingleStep(1.0)
        self._cap_spin.setSuffix(" mm")
        self._cap_spin.setEnabled(False)
        self._cap_chk.toggled.connect(self._cap_spin.setEnabled)
        r3.addWidget(self._cap_chk)
        r3.addWidget(self._cap_spin)
        r3.addStretch(1)
        self._run_btn = QPushButton(t("解析"))
        self._run_btn.clicked.connect(self._run)
        r3.addWidget(self._run_btn)
        v.addLayout(r3)

        # --- report ---
        self._out = QPlainTextEdit()
        self._out.setReadOnly(True)
        self._out.setFont(QFont("Consolas", 10))
        self._out.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._out.setPlaceholderText(
            t("2ファイルを指定して「解析」を押すと結果がここに表示されます。"))
        v.addWidget(self._out, 1)

        self.setWidget(body)
        self.setAcceptDrops(True)
        self._update_labels()

    # ------------------------------------------------------- file pickers
    def _pick_corotree(self):
        from PyQt6.QtWidgets import QFileDialog
        p, _ = QFileDialog.getOpenFileName(
            self, t("Select coronary tree"), self._start_dir(),
            "Coronary tree (*.corotree.json);;JSON (*.json)")
        if p:
            self._corotree_path = p
            self._update_labels()

    def _pick_full(self):
        from PyQt6.QtWidgets import QFileDialog
        p, _ = QFileDialog.getOpenFileName(
            self, t("Select FullLv"), self._start_dir(),
            "Full LV (*.FullLv.json);;JSON (*.json)")
        if p:
            self._full_path = p
            self._update_labels()

    def _start_dir(self) -> str:
        for p in (self._corotree_path, self._full_path):
            if p and os.path.isdir(os.path.dirname(p)):
                return os.path.dirname(p)
        return ""

    def _update_labels(self):
        self._corotree_lbl.setText(
            t("冠動脈ツリー: {n}", n=os.path.basename(self._corotree_path))
            if self._corotree_path else t("冠動脈ツリー: 未選択"))
        self._full_lbl.setText(
            t("LV (FullLv): {n}", n=os.path.basename(self._full_path))
            if self._full_path else t("LV (FullLv): 未選択"))
        self._run_btn.setEnabled(bool(self._corotree_path and self._full_path))

    # ------------------------------------------------------- drag & drop
    def _route_drop(self, path: str) -> bool:
        """Assign a dropped JSON to the corotree or FullLv slot by its content.
        Returns True if it matched one of the two."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:                                    # noqa: BLE001
            return False
        if full_lv.is_full_lv(data):
            self._full_path = path
            return True
        if isinstance(data, dict) and data.get("format") == "MDV-CoroTree":
            self._corotree_path = path
            return True
        return False

    def dragEnterEvent(self, e):                              # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):                               # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):                                   # noqa: N802
        matched = 0
        for u in e.mimeData().urls():
            f = u.toLocalFile()
            if f and os.path.isfile(f) and self._route_drop(f):
                matched += 1
        if matched:
            e.acceptProposedAction()
            self._update_labels()
            if self._corotree_path and self._full_path:
                self._run()                       # both present → analyse now
        else:
            QMessageBox.information(
                self, t("CT Territory"),
                t("corotree.json か FullLv.json を落としてください。"))

    # ------------------------------------------------------- run
    def _run(self):
        if not (self._corotree_path and self._full_path):
            return
        try:
            with open(self._corotree_path, encoding="utf-8") as f:
                tree_data = json.load(f)
            with open(self._full_path, encoding="utf-8") as f:
                full_data = json.load(f)
        except Exception as exc:                             # noqa: BLE001
            QMessageBox.warning(self, t("CT Territory"),
                                t("読込失敗: {err}", err=str(exc)))
            return
        if not full_lv.is_full_lv(full_data):
            QMessageBox.warning(
                self, t("CT Territory"),
                t("FullLv ファイルではありません（format が MDV-FullLV でない）。"))
            return
        # Same-CT guard: warn (don't block) if the two files name different
        # source series — a corotree + FullLv from DIFFERENT CTs give a
        # silently wrong territory.
        tser = ((tree_data.get("series") or {}).get("series_uid") or "")
        fser = full_lv.series_uid(full_data)
        if tser and fser and tser != fser:
            if QMessageBox.warning(
                    self, t("CT Territory"),
                    t("冠動脈ツリーと FullLv の元CT（SeriesUID）が一致しません。"
                      "このまま解析すると結果が正しくない可能性があります。続行しますか？"),
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
        tree = CoronaryTree.from_json(tree_data)
        lvf = LVFunction.from_full(full_data)
        if lvf is None:
            QMessageBox.warning(
                self, t("CT Territory"),
                t("心筋を構築できません（FullLv の Epi region か Endo が不足）。"))
            return
        if not tree.vessels:
            QMessageBox.warning(self, t("CT Territory"),
                                t("冠動脈ツリーに血管がありません。"))
            return
        cap = self._cap_spin.value() if self._cap_chk.isChecked() else None
        try:
            lines, _eng = format_territory_report(tree, lvf, max_dist_mm=cap)
        except Exception as exc:                             # noqa: BLE001
            QMessageBox.warning(self, t("CT Territory"),
                                t("解析失敗: {err}", err=str(exc)))
            return
        head = [
            "=" * 60,
            "CT Territory report",
            f"  tree : {os.path.basename(self._corotree_path)} "
            f"({len(tree.vessels)} vessels)",
            f"  LV   : {os.path.basename(self._full_path)}",
            "-" * 60,
        ]
        self._out.setPlainText("\n".join(head + lines + ["=" * 60]))
