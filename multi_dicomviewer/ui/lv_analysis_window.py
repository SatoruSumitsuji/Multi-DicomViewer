"""LV Analysis panel — manage / edit / save the LV side of a CT Territory
analysis (the ``.FullLv.json`` bundle) as a FIRST-CLASS dataset, on equal
footing with the Coronary Tree.

``corotree.json`` + ``FullLv.json`` are the two inputs to CT Territory. The
Coronary Tree has long had its own tab to build / edit / save the vessel side;
the LV side had no such route — once a FullLv was created there was no way to
re-open it, see what it contains, edit a component (MV / AoV / Apex / Epi /
Blood-Endo) and save it back. This panel is that route.

Like the Coronary Tree panel it is a :class:`SnapDock`, so it can FLOAT or dock
into the left Studies frame (tabbed with Studies / Case Presentation / Coronary
Tree). One instance is kept by the shell. Datasets are grouped per source CT and
labelled 日付 / 症例名 / Se番号 (never a bare UID) so a glance says which data a
dataset derives from. Editing opens the SOURCE 3-D CT and loads the FullLv into
the viewer's editable LV state (the actual drawing / tracing uses the in-viewer
LV bar); Save writes the FullLv.json AND the 5 component files.
"""
from __future__ import annotations

import json
import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.core import full_lv as full_lv_mod
from multi_dicomviewer.i18n import t
from multi_dicomviewer.ui.snap_dock import SnapDock

#: Tree role: the source-CT UID a top-level group stands for.
_GRP_UID = Qt.ItemDataRole.UserRole
#: Tree role: component key on a child row ("mv"/"aov"/"apex"/"epi"/"blood").
_COMP_KEY = Qt.ItemDataRole.UserRole + 1

#: The five LV components, in workflow order, with their display captions.
_COMPONENTS = (
    ("mv", "MV"),
    ("aov", "AoV"),
    ("apex", "Apex"),
    ("epi", "Epi"),
    ("blood", "Blood/Endo"),
)


def _human_ml(v) -> str:
    try:
        return f"{float(v):.1f} mL"
    except (TypeError, ValueError):
        return ""


def _component_status(data: dict) -> dict:
    """Presence + a short summary for each of the 5 components, read straight
    from a parsed FullLv dict (no heavy mask build)."""
    fd = data or {}
    epi = fd.get("epi") or {}
    bld = fd.get("bld") or {}
    mv_ok = isinstance(bld.get("mitral"), dict)
    av_ok = isinstance(bld.get("aortic"), dict)
    apex_ok = bld.get("apex") is not None
    epi_ok = isinstance(epi.get("region"), dict)
    endo_ok = isinstance(bld.get("endo"), dict)
    blood_ok = endo_ok or (bld.get("volume_ml") is not None)
    hu_lo, hu_hi = bld.get("hu_lo"), bld.get("hu_hi")
    hu = ""
    if hu_lo is not None and hu_hi is not None:
        try:
            hu = f"HU {int(hu_lo)}–{int(hu_hi)}"
        except (TypeError, ValueError):
            hu = ""
    blood_summary = " ".join(
        s for s in (_human_ml(bld.get("volume_ml")), hu) if s)
    return {
        "mv": (mv_ok, ""),
        "aov": (av_ok, ""),
        "apex": (apex_ok, ""),
        "epi": (epi_ok, t("領域マスクあり") if epi_ok else t("領域マスクなし")),
        "blood": (blood_ok, blood_summary),
    }


class LVAnalysisWindow(SnapDock):
    """Dockable LV-dataset manager. Signals mirror the Coronary Tree panel so the
    shell can coordinate (open the source CT, re-run territory, …)."""

    #: Emitted when the user asks to edit a dataset — (uid, src_dir, full_data).
    edit_requested = pyqtSignal(str, str, dict)

    def __init__(self, shell):
        super().__init__(t("LV Analysis"))
        self._shell = shell
        # uid -> {"data": full_dict, "path": str|"", "meta": label_meta dict}
        self._by_uid: dict[str, dict] = {}
        self.setAcceptDrops(True)                   # drop .FullLv.json onto it

        central = QWidget()
        self.setWidget(central)
        central.setMinimumWidth(220)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # File row: load / clear.
        frow = QHBoxLayout()
        frow.setSpacing(3)
        flbl = QLabel(t("LV データ："))
        flbl.setStyleSheet("font-weight:bold;")
        frow.addWidget(flbl)
        load_btn = QPushButton(t("読込"))
        load_btn.setToolTip(t(".FullLv.json を読み込む（ドラッグ＆ドロップも可）"))
        load_btn.clicked.connect(self._load_dialog)
        frow.addWidget(load_btn)
        clear_btn = QPushButton(t("全消去"))
        clear_btn.setToolTip(t("一覧から全ての LV データを消去（ファイルは残る）"))
        clear_btn.clicked.connect(self._clear_all)
        frow.addWidget(clear_btn)
        frow.addStretch(1)
        outer.addLayout(frow)

        # Dataset tree: top-level = source CT (日付/症例名/Se#), children = the 5
        # components with a ✓ / — status and a short summary.
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels([t("データ / 構成要素"), t("状態")])
        self._tree.setColumnWidth(0, 240)
        self._tree.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.setRootIsDecorated(True)
        self._tree.itemSelectionChanged.connect(self._sync_buttons)
        outer.addWidget(self._tree, 1)

        # Action row for the selected dataset.
        arow = QHBoxLayout()
        arow.setSpacing(3)
        albl = QLabel(t("操作："))
        albl.setStyleSheet("font-weight:bold;")
        arow.addWidget(albl)
        self._edit_btn = QPushButton(t("元CTで編集"))
        self._edit_btn.setToolTip(t(
            "選択した LV データの元 3DCT を開き、ビューアの LV モードに読み込んで "
            "MV / AoV / Apex / Epi / Blood-Endo を編集できる状態にする"))
        self._edit_btn.clicked.connect(self._edit_selected)
        arow.addWidget(self._edit_btn)
        self._save_btn = QPushButton(t("保存"))
        self._save_btn.setToolTip(t(
            "編集中（アクティブな CT）の LV 状態を FullLv.json と 5 つの個別ファイル "
            "(MV/AoV/Apex/Epi/Blood-Endo) に保存する"))
        self._save_btn.clicked.connect(self._save_active)
        arow.addWidget(self._save_btn)
        arow.addStretch(1)
        outer.addLayout(arow)

        self._hint = QLabel(t(
            "FullLv.json を読み込むか、ここにドロップしてください。"))
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color:#555;")
        outer.addWidget(self._hint)

        self._sync_buttons()

    # -------------------------------------------------- public (shell) API
    def load_full_lv(self, data_or_path) -> bool:
        """Register a FullLv (parsed dict OR a .FullLv.json path) in the list,
        keyed by its source-CT UID. Public — the shell calls it on a drop."""
        data = data_or_path
        path = ""
        if isinstance(data_or_path, str):
            path = data_or_path
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError) as exc:
                self._warn(t("読込に失敗しました: {e}", e=str(exc)))
                return False
        if not full_lv_mod.is_full_lv(data):
            self._warn(t("FullLv (.FullLv.json) 形式ではありません。"))
            return False
        meta = full_lv_mod.label_meta(data)
        uid = meta.get("series_uid", "") or path or f"lv{len(self._by_uid)}"
        self._by_uid[uid] = {"data": data, "path": path, "meta": meta}
        self._populate()
        self.show()
        self.raise_()
        self._hint.setText(t("読込みました: {p}",
                             p=os.path.basename(path) if path
                             else full_lv_mod.format_label(meta, "LV")))
        return True

    def has_data(self) -> bool:
        return bool(self._by_uid)

    def cleanup(self) -> None:                       # symmetry with other panels
        pass

    # -------------------------------------------------------------- list UI
    def _populate(self) -> None:
        self._tree.blockSignals(True)
        self._tree.clear()
        for uid, b in self._by_uid.items():
            data = b["data"]
            meta = b.get("meta") or full_lv_mod.label_meta(data)
            # Enrich the label from the LIVE loaded CT when available (always
            # correct once the source CT is up).
            if self._shell is not None \
                    and hasattr(self._shell, "coronary_series_meta"):
                try:
                    live = self._shell.coronary_series_meta(uid) or {}
                except Exception:                    # noqa: BLE001
                    live = {}
                for k in ("patient", "date", "series_number"):
                    if not meta.get(k) and live.get(k):
                        meta[k] = live[k]
            grp = QTreeWidgetItem([full_lv_mod.format_label(meta, "LV"), ""])
            grp.setData(0, _GRP_UID, uid)
            gf = grp.font(0); gf.setBold(True); grp.setFont(0, gf)
            grp.setForeground(0, QColor("#6b1a3a"))
            self._tree.addTopLevelItem(grp)
            status = _component_status(data)
            for key, cap in _COMPONENTS:
                ok, summary = status[key]
                mark = "✓" if ok else "—"
                it = QTreeWidgetItem([f"{mark}  {cap}", summary])
                it.setData(0, _COMP_KEY, key)
                it.setData(0, _GRP_UID, uid)
                it.setForeground(
                    0, QColor("#1a6b2a") if ok else QColor("#999999"))
                grp.addChild(it)
            grp.setExpanded(True)
        self._tree.blockSignals(False)
        self._sync_buttons()

    def _selected_uid(self) -> str:
        items = self._tree.selectedItems()
        if not items:
            return ""
        return items[0].data(0, _GRP_UID) or ""

    def _sync_buttons(self) -> None:
        uid = self._selected_uid()
        self._edit_btn.setEnabled(bool(uid))
        # Save acts on the ACTIVE viewer's live LV state, not the list selection.
        self._save_btn.setEnabled(True)

    # ----------------------------------------------------------- actions
    def _load_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, t("FullLv を読み込む"), "",
            t("Full LV (*.FullLv.json);;JSON (*.json)"))
        for p in paths:
            self.load_full_lv(p)

    def _clear_all(self) -> None:
        if not self._by_uid:
            return
        self._by_uid.clear()
        self._populate()
        self._hint.setText(t("一覧を消去しました（ファイルは残っています）。"))

    def _edit_selected(self) -> None:
        uid = self._selected_uid()
        if not uid:
            return
        b = self._by_uid.get(uid)
        if not b:
            return
        data = b["data"]
        src_dir = full_lv_mod.src_dir(data)
        # The shell opens the source CT and loads the FullLv editably; the actual
        # editing then happens in the in-viewer LV bar.
        if self._shell is not None \
                and hasattr(self._shell, "lv_open_full_for_edit"):
            try:
                self._shell.lv_open_full_for_edit(data, src_dir)
                self._hint.setText(t(
                    "元CTを開いて LV モードに読み込みました。ビューア下の LV バーで "
                    "MV / AoV / Apex / Epi / Blood-Endo を編集し、「保存」で書き戻せます。"))
                return
            except Exception as exc:                 # noqa: BLE001
                self._warn(t("編集用に開けませんでした: {e}", e=str(exc)))
                return
        self._warn(t("この版では編集起動がまだ利用できません。"))

    def _save_active(self) -> None:
        if self._shell is None or not hasattr(self._shell, "lv_save_active_full"):
            self._warn(t("この版では保存がまだ利用できません。"))
            return
        try:
            saved = self._shell.lv_save_active_full()
        except Exception as exc:                     # noqa: BLE001
            self._warn(t("保存に失敗しました: {e}", e=str(exc)))
            return
        if saved:
            # Refresh the list from the just-saved bundle (if the shell returned
            # the data) so the status rows reflect the new state.
            if isinstance(saved, dict):
                self.load_full_lv(saved)
            self._hint.setText(t("保存しました（FullLv ＋ 個別ファイル）。"))
        else:
            self._hint.setText(t(
                "保存できませんでした（アクティブな CT に LV データがありません）。"))

    # ---------------------------------------------------------- drag&drop
    @staticmethod
    def _dropped_full_lv(ev) -> list:
        md = ev.mimeData()
        if not md.hasUrls():
            return []
        out = []
        for u in md.urls():
            p = u.toLocalFile()
            if p and p.lower().endswith(".json"):
                out.append(p)
        return out

    def dragEnterEvent(self, ev):                    # noqa: N802 (Qt override)
        if self._dropped_full_lv(ev):
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):                     # noqa: N802 (Qt override)
        self.dragEnterEvent(ev)

    def dropEvent(self, ev):                          # noqa: N802 (Qt override)
        paths = self._dropped_full_lv(ev)
        if paths:
            ev.acceptProposedAction()
            for p in paths:
                self.load_full_lv(p)

    # --------------------------------------------------------------- utils
    def _warn(self, msg: str) -> None:
        self._hint.setText(msg)
        QMessageBox.warning(self, t("LV Analysis"), msg)
