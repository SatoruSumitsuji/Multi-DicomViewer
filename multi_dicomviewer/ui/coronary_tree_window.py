"""Coronary Tree panel — a dockable list of the coronary centrelines that make
up a CT Territory analysis, shown as their parent/child branch tree.

Like the Case Presentation panel it is a :class:`SnapDock`, so it can FLOAT or
dock into the left Studies frame (tabbed with Studies / Case Presentation). One
instance is kept by the shell. The panel owns the ``CoronaryTree`` model; the CT
viewer registers vessels into it (roots LM-LAD / LM-LCX / RCA, or snapped
branches) and reads back the selection / visibility to draw the overlays. This
first increment covers the model, the tree view, edit actions (rename /
re-parent / delete) and ``.corotree.json`` save / load; the viewer wiring and
the on-image overlay follow.
"""
from __future__ import annotations

import json
import os

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.core.coronary_territory import (
    ROOT_ROLES, CoronaryTree)
from multi_dicomviewer.i18n import t
from multi_dicomviewer.ui.snap_dock import SnapDock

#: One colour per root trunk; branches inherit their root's colour.
ROOT_COLORS = {"LM-LAD": "#d62728", "LM-LCX": "#1f77b4", "RCA": "#2ca02c"}
_UID_ROLE = Qt.ItemDataRole.UserRole


class CoronaryTreeWindow(SnapDock):
    """Dockable coronary-tree list. Float or dock into the left Studies frame."""

    #: a vessel row was selected (vessel id, or "" when cleared)
    vesselSelected = pyqtSignal(str)
    #: a vessel's centreline visibility was toggled (vessel id, on)
    visibilityChanged = pyqtSignal(str, bool)
    #: the tree changed (add / delete / re-parent / load / clear)
    treeChanged = pyqtSignal()
    #: "ツリーに追加" was pressed — register the active CT viewer's current CPR
    #: with this role (LM-LAD / LM-LCX / RCA / Branch). The shell handles it.
    addCprRequested = pyqtSignal(str)

    def __init__(self, shell):
        super().__init__(t("Coronary Tree"))
        self._shell = shell
        self._tree = CoronaryTree()
        self._last_path: str | None = None
        self._building = False

        central = QWidget()
        self.setWidget(central)
        central.setMinimumWidth(200)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(3)

        bar = QHBoxLayout()
        for label, tip, fn in (
                (t("読込…"), t("冠動脈ツリー (.corotree.json) を読込"), self._load),
                (t("上書き保存"), t("直前のファイルへ上書き保存"), self._save_overwrite),
                (t("名前を付けて保存…"), t("冠動脈ツリーを保存"), self._save_as),
                (t("全消去"), t("全ての血管を消去 (元CPRは残る)"), self._clear_all)):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch(1)
        outer.addLayout(bar)

        # Role picker + "ツリーに追加" — the CT Territory workflow lives HERE (not
        # on the image's plain CPR row) so the two stay clearly separate.
        role_row = QHBoxLayout()
        role_row.addWidget(QLabel(t("役割:")))
        self._role_combo = QComboBox()
        self._role_combo.addItems(["LM-LAD", "LM-LCX", "RCA", "Branch"])
        self._role_combo.setToolTip(t(
            "ルート3種は入口を第1点に。Branchは既存血管の上から描き始める "
            "(枝名は追加後に指定)"))
        role_row.addWidget(self._role_combo)
        self._add_btn = QPushButton(t("ツリーに追加"))
        self._add_btn.setToolTip(t(
            "いま描いたCPR(アクティブなCT)をこの役割でツリーに追加。"
            "Branchは最近接血管に吸着、3mm超なら近づけて再度追加"))
        self._add_btn.clicked.connect(
            lambda: self.addCprRequested.emit(self._role_combo.currentText()))
        role_row.addWidget(self._add_btn)
        role_row.addStretch(1)
        outer.addLayout(role_row)

        self._tree_w = QTreeWidget()
        self._tree_w.setColumnCount(2)
        self._tree_w.setHeaderLabels([t("血管"), t("分岐")])
        self._tree_w.setColumnWidth(0, 150)
        self._tree_w.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree_w.customContextMenuRequested.connect(self._menu)
        self._tree_w.itemChanged.connect(self._on_item_changed)
        self._tree_w.itemSelectionChanged.connect(self._on_selection)
        self._tree_w.itemDoubleClicked.connect(lambda *_: self._rename())
        outer.addWidget(self._tree_w, 1)

        self._hint = QLabel("")
        self._hint.setStyleSheet("color:#888;")
        outer.addWidget(self._hint)
        self._refresh_hint()

    # ------------------------------------------------------------- model
    @property
    def tree(self) -> CoronaryTree:
        return self._tree

    def add_root(self, vid, name, role, points, ctrl=None):
        """Register an explicit root trunk (viewer entry point)."""
        v = self._tree.add_root(vid, name, role, points)
        if ctrl is not None:
            v.ctrl = np.asarray(ctrl, float).reshape(-1, 3)
        self._populate()
        self.treeChanged.emit()
        return v

    def add_branch(self, vid, name, points, ctrl=None, snap_tol_mm=3.0):
        """Register a branch (viewer entry point). Returns the snap result; the
        vessel is only added when ``ok`` is True (within snap_tol_mm)."""
        res = self._tree.add_branch(vid, name, points, snap_tol_mm=snap_tol_mm)
        if res.get("ok"):
            if ctrl is not None:
                self._tree.vessels[vid].ctrl = np.asarray(
                    ctrl, float).reshape(-1, 3)
            self._populate()
            self.treeChanged.emit()
        return res

    def selected_vid(self) -> str | None:
        it = self._tree_w.currentItem()
        return None if it is None else it.data(0, _UID_ROLE)

    def rename_vessel(self, vid: str, name: str) -> None:
        """Public: set a vessel's name (used by the shell after the post-draw
        name prompt for a branch)."""
        v = self._tree.vessels.get(vid)
        if v is not None and name and name.strip():
            v.name = name.strip()
            self._populate()
            self.treeChanged.emit()

    # ----------------------------------------------------------- display
    def _root_role(self, vid: str) -> str:
        v = self._tree.vessels.get(vid)
        while v is not None and v.parent is not None:
            v = self._tree.vessels.get(v.parent)
        return v.role if v is not None else ""

    def _junction_text(self, vid: str) -> str:
        v = self._tree.vessels[vid]
        if v.parent is None:
            return t("root")
        p = self._tree.vessels.get(v.parent)
        pname = p.name if p is not None else v.parent
        if p is None or p.n < 2 or v.junction is None:
            return f"@{pname}"
        # Junction position along the parent as BOTH an arc-length fraction (0% =
        # proximal/ostium, 100% = distal) and mm from the parent's proximal end;
        # points are uniform arc-length samples, so idx maps linearly to length.
        frac = v.junction / (p.n - 1)
        total_mm = float(np.linalg.norm(np.diff(p.points, axis=0), axis=1).sum())
        return f"@{pname} {round(100 * frac)}% / {frac * total_mm:.0f}mm"

    def _populate(self):
        self._building = True
        self._tree_w.blockSignals(True)
        self._tree_w.clear()
        items: dict[str, QTreeWidgetItem] = {}

        def add_item(vid):
            v = self._tree.vessels[vid]
            it = QTreeWidgetItem([v.name, self._junction_text(vid)])
            it.setData(0, _UID_ROLE, vid)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(0, Qt.CheckState.Checked)      # visible by default
            it.setForeground(0, QColor(ROOT_COLORS.get(
                self._root_role(vid), "#333333")))
            parent = self._tree.vessels[vid].parent
            if parent in items:
                items[parent].addChild(it)
            else:
                self._tree_w.addTopLevelItem(it)
            items[vid] = it

        # Insertion order guarantees parents precede children.
        for vid in self._tree.vessels:
            add_item(vid)
        self._tree_w.expandAll()
        self._tree_w.blockSignals(False)
        self._building = False
        self._refresh_hint()

    def _refresh_hint(self):
        n = len(self._tree.vessels)
        roots = len(self._tree.roots())
        self._hint.setText(t("{n} 本 / ルート {r} 本", n=n, r=roots))

    # -------------------------------------------------------- edit slots
    def _on_selection(self):
        self.vesselSelected.emit(self.selected_vid() or "")

    def _on_item_changed(self, item, _col):
        if self._building:
            return
        vid = item.data(0, _UID_ROLE)
        if vid:
            on = item.checkState(0) == Qt.CheckState.Checked
            self.visibilityChanged.emit(vid, on)

    def _menu(self, pos):
        it = self._tree_w.itemAt(pos)
        menu = QMenu(self)
        a_ren = menu.addAction(t("名前を変更"))
        a_par = menu.addAction(t("親を変更…"))
        a_del = menu.addAction(t("削除 (枝ごと)"))
        for a in (a_ren, a_par, a_del):
            a.setEnabled(it is not None)
        chosen = menu.exec(self._tree_w.viewport().mapToGlobal(pos))
        if chosen is a_ren:
            self._rename()
        elif chosen is a_par:
            self._reparent()
        elif chosen is a_del:
            self._delete()

    def _rename(self):
        vid = self.selected_vid()
        if not vid:
            return
        cur = self._tree.vessels[vid].name
        name, ok = QInputDialog.getText(self, t("名前を変更"), t("血管名:"),
                                        text=cur)
        if ok and name.strip():
            self._tree.vessels[vid].name = name.strip()
            self._populate()
            self.treeChanged.emit()

    def _reparent(self):
        vid = self.selected_vid()
        if not vid:
            return
        v = self._tree.vessels[vid]
        if v.parent is None:
            self._warn(t("ルート血管の親は変更できません。"))
            return
        # Candidate parents = every OTHER vessel that is not a descendant of vid.
        banned = {vid, *self._tree.descendants(vid)}
        cands = [(k, self._tree.vessels[k].name)
                 for k in self._tree.vessels if k not in banned]
        if not cands:
            return
        labels = [f"{name} ({k})" for k, name in cands]
        cur = labels[[k for k, _ in cands].index(v.parent)] \
            if v.parent in [k for k, _ in cands] else labels[0]
        pick, ok = QInputDialog.getItem(self, t("親を変更"), t("新しい親:"),
                                        labels, labels.index(cur), False)
        if ok and pick:
            new_parent = cands[labels.index(pick)][0]
            self._tree.set_parent(vid, new_parent)   # junction = nearest sample
            self._populate()
            self.treeChanged.emit()

    def _delete(self):
        vid = self.selected_vid()
        if not vid:
            return
        victims = [vid, *self._tree.descendants(vid)]
        name = self._tree.vessels[vid].name
        if QMessageBox.question(
                self, t("削除"),
                t("{name} とその下流の枝 {k} 本を削除しますか? "
                  "(元のCPRファイルは残ります)", name=name,
                  k=len(victims) - 1)) != QMessageBox.StandardButton.Yes:
            return
        for k in victims:
            self._tree.vessels.pop(k, None)
        self._populate()
        self.treeChanged.emit()

    def _clear_all(self):
        if not self._tree.vessels:
            return
        if QMessageBox.question(
                self, t("全消去"),
                t("全ての血管を消去しますか?")) == QMessageBox.StandardButton.Yes:
            self._tree = CoronaryTree()
            self._populate()
            self.treeChanged.emit()

    # ------------------------------------------------------------- files
    def _warn(self, msg):
        QMessageBox.information(self, t("Coronary Tree"), msg)

    def _save_as(self):
        if not self._tree.vessels:
            self._warn(t("保存する血管がありません。"))
            return
        d = os.path.dirname(self._last_path) if self._last_path else ""
        default = os.path.join(d, "coronary.corotree.json") if d \
            else "coronary.corotree.json"
        path, _ = QFileDialog.getSaveFileName(
            self, t("冠動脈ツリーを保存"), default, t("CoroTree (*.corotree.json)"))
        if path:
            self._write_to(path)

    def _save_overwrite(self):
        if self._last_path:
            self._write_to(self._last_path)
        else:
            self._save_as()

    def _write_to(self, path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._tree.to_json(), f, ensure_ascii=False, indent=2)
            self._last_path = path
            self._hint.setText(t("保存しました: {p}", p=path))
        except OSError as exc:
            self._warn(t("保存に失敗しました: {e}", e=str(exc)))

    def load_file(self, path) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            self._warn(t("読込に失敗しました: {e}", e=str(exc)))
            return False
        if data.get("format") != "MDV-CoroTree":
            self._warn(t("冠動脈ツリー形式のファイルではありません。"))
            return False
        self._tree = CoronaryTree.from_json(data)
        self._last_path = path
        self._populate()
        self.treeChanged.emit()
        self._hint.setText(t("読込みました: {p}", p=path))
        return True

    def _load(self):
        d = os.path.dirname(self._last_path) if self._last_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, t("冠動脈ツリーを読込"), d, t("CoroTree (*.corotree.json)"))
        if path:
            self.load_file(path)
