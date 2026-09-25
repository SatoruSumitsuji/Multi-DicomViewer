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

from multi_dicomviewer.core.centerline import CenterLine
from multi_dicomviewer.core.coronary_territory import (
    ROOT_ROLES, CoronaryTree)
from multi_dicomviewer.i18n import t
from multi_dicomviewer.ui.snap_dock import SnapDock

#: One colour per root trunk; branches inherit their root's colour.
ROOT_COLORS = {"LM-LAD": "#d62728", "LM-LCX": "#1f77b4", "RCA": "#2ca02c"}
_UID_ROLE = Qt.ItemDataRole.UserRole
#: Centreline resample step (mm) when rebuilding a .cpr.json's control points —
#: dense enough for a coronary vessel; territory granularity, not correctness.
_CPR_STEP_MM = 0.5


class CoronaryTreeWindow(SnapDock):
    """Dockable coronary-tree list. Float or dock into the left Studies frame."""

    #: a vessel row was selected (vessel id, or "" when cleared)
    vesselSelected = pyqtSignal(str)
    #: a vessel's centreline visibility was toggled (vessel id, on)
    visibilityChanged = pyqtSignal(str, bool)
    #: the tree changed (add / delete / re-parent / load / clear / connect)
    treeChanged = pyqtSignal()

    def __init__(self, shell):
        super().__init__(t("Coronary Tree"))
        self._shell = shell
        self._tree = CoronaryTree()
        self._last_path: str | None = None
        self._building = False
        self._hidden: set[str] = set()      # vids the user unchecked (overlay off)
        self._ct_uid: str = ""              # source-CT series UID (overlay target)
        self._ct_dir: str = ""              # source-CT folder (re-open fallback)
        self.setAcceptDrops(True)           # drag .cpr.json onto the panel

        central = QWidget()
        self.setWidget(central)
        central.setMinimumWidth(200)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(3)

        # Batch workflow: load per-vessel .cpr.json files, set each role
        # (right-click ▸ 役割), then 接続 grows the tree by nearest endpoint.
        bar = QHBoxLayout()
        for label, tip, fn in (
                (t("CPR読込…"),
                 t("枝ごとの .cpr.json を複数選択で読み込む（未接続で追加）"),
                 self._load_cpr),
                (t("接続"),
                 t("読み込んだ枝を最近接端点でツリーに接続 (3mm以内)。"
                   "後から追加読込→再度接続も可"), self._connect),
                (t("画像表示"),
                 t("元の3DCTをデフォルト表示し、冠動脈ツリーを重畳表示"),
                 self._show_image),
                (t("ツリー保存…"), t("冠動脈ツリーを .corotree.json に保存"),
                 self._save_as),
                (t("ツリー読込…"),
                 t("保存した冠動脈ツリー (.corotree.json) を読込"), self._load),
                (t("全消去"), t("全ての血管を消去 (.cpr.json は残る)"),
                 self._clear_all)):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch(1)
        outer.addLayout(bar)

        self._tree_w = QTreeWidget()
        self._tree_w.setColumnCount(3)
        self._tree_w.setHeaderLabels([t("血管"), t("役割"), t("分岐")])
        self._tree_w.setColumnWidth(0, 130)
        self._tree_w.setColumnWidth(1, 70)
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

        # Push the on-image overlay whenever the tree, selection or a vessel's
        # visibility changes (the shell fans it out to every CT viewer).
        self.treeChanged.connect(self._push_overlay)
        self.vesselSelected.connect(lambda *_: self._push_overlay())

    # ------------------------------------------------------------- model
    @property
    def tree(self) -> CoronaryTree:
        return self._tree

    def _unique_vid(self) -> str:
        n = len(self._tree.vessels) + 1
        vid = f"v{n}"
        while vid in self._tree.vessels:
            n += 1
            vid = f"v{n}"
        return vid

    def _load_cpr(self):
        """Load one or more per-vessel .cpr.json files as UNCONNECTED vessels
        (role defaults to branch; set roots via right-click ▸ 役割). Press 接続
        afterwards to attach them by nearest endpoint. Files can also be dragged
        and dropped onto the panel (see dropEvent)."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, t("CPR (.cpr.json) を読込"),
            os.path.dirname(self._last_path) if self._last_path else "",
            t("CPR (*.cpr.json)"))
        if paths:
            self._load_cpr_paths(paths)

    def _load_cpr_paths(self, paths):
        """Load the given .cpr.json paths (shared by the file dialog and drag &
        drop) as unconnected vessels, then bring the source CT into view."""
        added, errs = 0, []
        ct_uid, ct_dir = "", ""              # source 3-D CT of the first vessel
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("format") != "MDV-CPR":
                    errs.append(f"{os.path.basename(p)}: " + t("CPR形式でない"))
                    continue
                ctrl = data.get("ctrl")
                if not ctrl or len(ctrl) < 2:
                    errs.append(f"{os.path.basename(p)}: " + t("中心点が不足"))
                    continue
                ctrl = np.asarray(ctrl, float)
                cl = CenterLine.from_points(ctrl, step_mm=_CPR_STEP_MM)
                # strip ".cpr.json" → vessel name
                name = os.path.splitext(
                    os.path.splitext(os.path.basename(p))[0])[0]
                self._tree.add_vessel(self._unique_vid(), name or "vessel",
                                      "branch", cl.points, ctrl=ctrl)
                added += 1
                if not ct_uid:               # remember the CT to open the overlay on
                    ct_uid = (data.get("series") or {}).get("series_uid", "")
                    ct_dir = data.get("src_dir", "") or ""
            except (OSError, ValueError) as exc:            # noqa: BLE001
                errs.append(f"{os.path.basename(p)}: {exc}")
        if paths:
            self._last_path = paths[0]
        if ct_uid:                           # overlay target for the shell
            self._ct_uid = ct_uid
        if ct_dir:
            self._ct_dir = ct_dir
        self._populate()
        self.treeChanged.emit()
        msg = t("{n} 本を読込（役割を設定して「接続」）。", n=added)
        if errs:
            msg += " " + t("失敗 {e} 件。", e=len(errs))
        # Bring the source 3-D CT into view so the tree overlay has its volume.
        # prompt=False: a plain .cpr.json load / drop must never pop a native
        # folder dialog (use 画像表示 to pick the folder when it can't be found).
        if added and (self._ct_uid or self._ct_dir):
            st = self._show_source_ct(prompt=False)
            if st:
                msg += " " + t("3DCT: {s}", s=st)
        self._hint.setText(msg)
        if errs:
            self._warn("\n".join(errs[:8]))

    def _show_image(self):
        """画像表示 button: (re)open the source 3-D CT at its default view and
        overlay the coronary tree on it."""
        if not self._tree.vessels:
            self._warn(t("先に CPR を読み込んでください。"))
            return
        st = self._show_source_ct()
        self._hint.setText(t("画像表示: {s}", s=st) if st
                           else t("元の3DCTが特定できません。CPRを読み込み直すか"
                                  "フォルダを選択してください。"))

    def _show_source_ct(self, prompt: bool = True) -> str:
        """Ask the shell to show the tree's source CT (by UID / saved folder / the
        tree-file's own folder / optional prompt) and refresh the overlay.
        *prompt* False = never scan or pop a dialog (the automatic CPR-load path).
        Returns a short status string."""
        if self._shell is None or not hasattr(self._shell, "coronary_show_ct"):
            return ""
        # The folder the .cpr.json / .corotree.json came from — the CT usually
        # lives here or in a subfolder, so a deliberate 画像表示 can find it
        # without asking.
        cpr_dir = os.path.dirname(self._last_path) if self._last_path else ""
        try:
            return self._shell.coronary_show_ct(
                self._ct_uid, self._ct_dir, cpr_dir=cpr_dir, prompt=prompt) or ""
        except Exception:                                   # noqa: BLE001
            return ""

    # --------------------------------------------------- drag & drop
    def dragEnterEvent(self, e):
        """Accept a drag that carries at least one .cpr.json file."""
        md = e.mimeData()
        if md is not None and md.hasUrls() and any(
                u.toLocalFile().lower().endswith(".cpr.json")
                for u in md.urls()):
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        md = e.mimeData()
        if md is not None and md.hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        """Load every .cpr.json dropped onto the panel."""
        md = e.mimeData()
        paths = []
        if md is not None and md.hasUrls():
            for u in md.urls():
                f = u.toLocalFile()
                if f.lower().endswith(".cpr.json"):
                    paths.append(f)
        if paths:
            e.acceptProposedAction()
            self._load_cpr_paths(paths)
        else:
            super().dropEvent(e)

    def _connect(self):
        """Grow the tree: attach every loose branch to the nearest connected
        vessel by its nearest endpoint (3 mm). Roots must be set first."""
        if not self._tree.vessels:
            self._warn(t("先に CPR を読み込んでください。"))
            return
        if not self._tree.roots():
            self._warn(t("ルート (LM-LAD / LM-LCX / RCA) を1本以上設定して"
                         "ください。血管を右クリック →「役割」で設定できます。"))
            return
        res = self._tree.connect_all(snap_tol_mm=3.0)
        self._populate()
        self.treeChanged.emit()
        msg = t("{c} 本を接続しました。", c=len(res["connected"]))
        if res["unconnected"]:
            msg += t(" 未接続 {u} 本（3mm以内に幹/枝がありません。近い枝を"
                     "先に接続するか、右クリックで親を指定）。",
                     u=len(res["unconnected"]))
        self._hint.setText(msg)

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
            # A true trunk shows "root"; a loose branch (parent None but NOT a
            # root role) shows "未接続" so it is never mistaken for a trunk.
            return t("root") if v.role in ROOT_ROLES else t("未接続")
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

    def _make_item(self, vid: str) -> QTreeWidgetItem:
        v = self._tree.vessels[vid]
        role_txt = v.role if v.role in ROOT_ROLES else t("枝")
        it = QTreeWidgetItem([v.name, role_txt, self._junction_text(vid)])
        it.setData(0, _UID_ROLE, vid)
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        it.setCheckState(0, Qt.CheckState.Checked)          # visible by default
        it.setForeground(0, QColor(ROOT_COLORS.get(
            self._root_role(vid), "#333333")))
        return it

    def _populate(self):
        self._building = True
        self._tree_w.blockSignals(True)
        self._tree_w.clear()
        loose = set(self._tree.unconnected())

        # Connected tree — traverse from the roots so nesting is correct
        # regardless of the order vessels were loaded.
        def add_recursive(vid, parent_item):
            it = self._make_item(vid)
            if parent_item is None:
                self._tree_w.addTopLevelItem(it)
            else:
                parent_item.addChild(it)
            for c in self._tree.children(vid):
                add_recursive(c, it)

        for r in self._tree.roots():
            add_recursive(r, None)
        # Unconnected branches → a red "（未接続）" group.
        if loose:
            grp = QTreeWidgetItem([t("（未接続 {n}）", n=len(loose)), "", ""])
            grp.setForeground(0, QColor("#b00000"))
            self._tree_w.addTopLevelItem(grp)
            for vid in self._tree.vessels:
                if vid in loose:
                    grp.addChild(self._make_item(vid))
        self._tree_w.expandAll()
        self._tree_w.blockSignals(False)
        self._building = False
        self._refresh_hint()

    def _refresh_hint(self):
        n = len(self._tree.vessels)
        roots = len(self._tree.roots())
        loose = len(self._tree.unconnected())
        self._hint.setText(
            t("{n} 本 / ルート {r} / 未接続 {u}", n=n, r=roots, u=loose))

    # -------------------------------------------------------- edit slots
    def _on_selection(self):
        self.vesselSelected.emit(self.selected_vid() or "")

    def _on_item_changed(self, item, _col):
        if self._building:
            return
        vid = item.data(0, _UID_ROLE)
        if vid:
            on = item.checkState(0) == Qt.CheckState.Checked
            if on:
                self._hidden.discard(vid)
            else:
                self._hidden.add(vid)
            self.visibilityChanged.emit(vid, on)
            self._push_overlay()

    # --------------------------------------------------------- overlay
    def overlay_spec(self) -> list:
        """The on-image overlay: one entry per VISIBLE vessel with 2+ points —
        its world-mm centreline, root colour and name — for the CT viewers to
        reproject onto their MPR planes. The currently-selected vessel is
        flagged so the viewer can highlight it."""
        sel = self.selected_vid()
        out = []
        for vid, v in self._tree.vessels.items():
            if vid in self._hidden:
                continue
            pts = np.asarray(v.points, float)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            out.append({
                "vid": vid,
                "name": v.name,
                "points": pts.tolist(),
                "color": ROOT_COLORS.get(self._root_role(vid), "#888888"),
                "selected": (vid == sel),
            })
        return out

    def _push_overlay(self):
        """Ask the shell to fan the current overlay out to every CT viewer."""
        if self._shell is not None \
                and hasattr(self._shell, "coronary_overlay_refresh"):
            try:
                self._shell.coronary_overlay_refresh()
            except Exception:                            # noqa: BLE001
                pass

    def _menu(self, pos):
        it = self._tree_w.itemAt(pos)
        vid = it.data(0, _UID_ROLE) if it is not None else None
        menu = QMenu(self)
        role_menu = menu.addMenu(t("役割"))
        role_acts = {}
        for r in (*ROOT_ROLES, "branch"):
            a = role_menu.addAction(t("枝") if r == "branch" else r)
            role_acts[a] = r
        a_rev = menu.addAction(t("向きを反転 (近位↔遠位)"))
        a_ren = menu.addAction(t("名前を変更"))
        a_par = menu.addAction(t("親を変更…"))
        a_del = menu.addAction(t("削除 (枝ごと)"))
        for a in (a_rev, a_ren, a_par, a_del):
            a.setEnabled(vid is not None)
        role_menu.setEnabled(vid is not None)
        chosen = menu.exec(self._tree_w.viewport().mapToGlobal(pos))
        if chosen in role_acts and vid is not None:
            self._tree.set_role(vid, role_acts[chosen])
            self._populate()
            self.treeChanged.emit()
        elif chosen is a_rev and vid is not None:
            self._reverse(vid)
        elif chosen is a_ren:
            self._rename()
        elif chosen is a_par:
            self._reparent()
        elif chosen is a_del:
            self._delete()

    def _reverse(self, vid):
        """Flip a vessel's proximal↔distal direction. A connected branch is also
        detached so 接続 re-attaches it with the corrected direction; a root just
        flips (fix an ostium drawn at the wrong end)."""
        self._tree.reverse_vessel(vid)
        v = self._tree.vessels.get(vid)
        if v is not None and v.role not in ROOT_ROLES:
            v.parent = None                 # re-attach on next 接続
            v.junction = None
        self._populate()
        self.treeChanged.emit()

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
        if v.role in ROOT_ROLES:                 # a true trunk has no parent
            self._warn(t("ルート(LM-LAD/LM-LCX/RCA)の親は変更できません。"
                         "役割を「枝」に変えてから接続してください。"))
            return
        # Candidate parents = ROOT vessels only (role LM-LAD/LM-LCX/RCA), never a
        # descendant of vid. Per user request the picker lists trunks only, so a
        # loose branch is attached to a root; further branch-to-branch nesting is
        # done via 接続 (nearest endpoint).
        banned = {vid, *self._tree.descendants(vid)}
        cands = [(k, self._tree.vessels[k].name, self._tree.vessels[k].role)
                 for k, vv in self._tree.vessels.items()
                 if k not in banned and vv.role in ROOT_ROLES]
        if not cands:
            self._warn(t("親にできるルート(LM-LAD/LM-LCX/RCA)がありません。"
                         "先にルートを設定してください。"))
            return
        # Show the ROLE (LM-LAD / LM-LCX / RCA) first, then the vessel name only
        # when it differs from the role — so the picker reads as roots, not file
        # names.
        labels = [role if name == role else f"{role}（{name}）"
                  for _k, name, role in cands]
        keys = [k for k, _n, _r in cands]
        cur = labels[keys.index(v.parent)] if v.parent in keys else labels[0]
        pick, ok = QInputDialog.getItem(self, t("親を変更"), t("新しい親:"),
                                        labels, labels.index(cur), False)
        if ok and pick:
            new_parent = keys[labels.index(pick)]
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
            # Embed the source-CT identity so a later load (or a drag&drop of
            # the .corotree.json onto the shell) can re-open the same 3-D CT.
            series = {"series_uid": self._ct_uid, "src_dir": self._ct_dir}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._tree.to_json(series=series), f,
                          ensure_ascii=False, indent=2)
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
        self._hidden.clear()
        ser = data.get("series") or {}
        self._ct_uid = ser.get("series_uid", "") or ""
        self._ct_dir = ser.get("src_dir", "") or ""
        self._last_path = path
        self._populate()
        self.treeChanged.emit()
        self._hint.setText(t("読込みました: {p}", p=path))
        return True

    def _load(self):
        d = os.path.dirname(self._last_path) if self._last_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, t("冠動脈ツリーを読込"), d, t("CoroTree (*.corotree.json)"))
        if path and self.load_file(path):
            # ツリー読込 → also bring up the source CT with the overlay drawn.
            self._show_image()
