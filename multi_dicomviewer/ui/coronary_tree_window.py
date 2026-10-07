"""Coronary Tree panel — a dockable list of the coronary centrelines that make
up a CT Territory analysis, shown as their parent/child branch tree.

Like the Case Presentation panel it is a :class:`SnapDock`, so it can FLOAT or
dock into the left Studies frame (tabbed with Studies / Case Presentation). One
instance is kept by the shell. The panel owns the ``CoronaryTree`` model; the CT
viewer registers vessels into it (roots LM / LAD / LCX / RCA, or snapped
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
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.core import full_lv as full_lv_mod
from multi_dicomviewer.core import settings as settings_mod
from multi_dicomviewer.core.centerline import CenterLine
from multi_dicomviewer.core.coronary_territory import (
    ROOT_ROLES, CoronaryTree, format_territory_report, short_vessel_name)
from multi_dicomviewer.core.lv_function import LVFunction
from multi_dicomviewer.i18n import t
from multi_dicomviewer.ui.snap_dock import SnapDock

#: One colour per root trunk; branches inherit their root's colour. Soft/pale
#: tones so the lines read on a grey CT: LAD pale blue, LCX pale yellow, RCA pale
#: green, LM pale purple (the left-main stub). Legacy LM-LAD/LM-LCX keys alias to
#: LAD/LCX so an old tree still colours before it is normalised.
ROOT_COLORS = {
    "LM":  "#b4a7d6",       # pale purple
    "LAD": "#6fa8dc",       # pale blue
    "LCX": "#bf9000",       # dark(er) yellow — readable on a light panel
    "RCA": "#93c47d",       # pale green
    "LM-LAD": "#6fa8dc", "LM-LCX": "#bf9000",   # legacy aliases
}
#: Perfusion-territory fill colours (RGBA 0-1) matching the tree colours but PALE
#: (a translucent wash over the myocardium): LAD blue, LCX yellow, RCA green, and
#: a set Target's distal territory in red on top. Labels 1..4 in that order.
TERRITORY_FILLS = [
    (0.36, 0.58, 0.94, 0.30),    # 1 LAD  — blue (light wash so coronaries show)
    (1.00, 0.85, 0.40, 0.28),    # 2 LCX  — yellow
    (0.58, 0.77, 0.49, 0.28),    # 3 RCA  — green
    (0.918, 0.50, 0.50, 0.42),   # 4 Target territory — red
]
#: Clinical display order for root trunks — LM first (not alphabetical).
_ROOT_ORDER = {"LM": 0, "LAD": 1, "LCX": 2, "RCA": 3}
_ROLE_TERR_LABEL = {"LAD": 1, "LCX": 2, "RCA": 3}
#: Target-territory fill alpha (kept when the user changes the colour).
_TARGET_ALPHA = 0.42
#: A small palette for the Target-territory colour — distinct from the base
#: LAD-blue / LCX-yellow / RCA-green so a target still stands out. "その他…" opens
#: a full colour picker.
TARGET_COLOR_CHOICES = [
    ("赤", "#e06666"),
    ("橙", "#f6b26b"),
    ("桃", "#e78ac3"),
    ("紫", "#9b59b6"),
    ("白", "#f2f2f2"),
]
#: Reserved for the (future-phase) perfusion-territory overlay — pale red.
TARGET_COLOR = "#ea9999"
_UID_ROLE = Qt.ItemDataRole.UserRole          # vessel vid on an item (None on a group)
_CTUID_ROLE = Qt.ItemDataRole.UserRole + 1    # source-CT UID on every item
_GRP_ROLE = Qt.ItemDataRole.UserRole + 2      # CT-group header → its UID
#: Centreline resample step (mm) when rebuilding a .cpr.json's control points —
#: dense enough for a coronary vessel; territory granularity, not correctness.
_CPR_STEP_MM = 0.5
#: 接続 (auto-connect) snap tolerance: a branch endpoint within this distance of
#: an existing vessel attaches to it. Was 3 mm; widened to 5 mm per user request.
_CONNECT_TOL_MM = 5.0


class CoronaryTreeWindow(SnapDock):
    """Dockable coronary-tree list. Float or dock into the left Studies frame."""

    #: a vessel row was selected (vessel id, or "" when cleared)
    vesselSelected = pyqtSignal(str)
    #: a vessel's centreline visibility was toggled (vessel id, on)
    visibilityChanged = pyqtSignal(str, bool)
    #: the tree changed (add / delete / re-parent / load / clear / connect)
    treeChanged = pyqtSignal()

    # ---- per-source-CT bundle plumbing --------------------------------------
    def _bundle(self, uid=None):
        """State bundle for a CT UID (the ACTIVE one by default); created lazily."""
        uid = self._active_uid if uid is None else (uid or "")
        b = self._by_uid.get(uid)
        if b is None:
            b = {"tree": CoronaryTree(), "hidden": set(), "targets": [],
                 "full_lv": None, "territory": None, "myo_ml": None,
                 "ct_dir": "", "last_path": None, "label": "", "visible": True,
                 "series": {}, "paths": {}}   # paths: vid → its .cpr.json file
            self._by_uid[uid] = b
        return b

    def _use_uid(self, uid, ct_dir=None, label=None, series=None):
        """Make *uid* the ACTIVE CT (its bundle then backs _tree/_hidden/…).
        *series* = the .cpr.json's embedded meta (patient/date/series_number) kept
        for the group label when the CT isn't loaded for a live lookup."""
        self._active_uid = uid or ""
        b = self._bundle()
        if ct_dir:
            b["ct_dir"] = ct_dir
        if label:
            b["label"] = label
        if series:
            # keep any non-empty fields (don't let a later meta-less load wipe them)
            for k, v in series.items():
                if v not in (None, ""):
                    b["series"][k] = v
        return b

    def _uid_label(self, uid, series=None):
        """CT-group label = 症例 / 検査日(8桁) / シリーズ番号 (patient / study date /
        Se#). The LIVE loaded-CT metadata is preferred (always correct once the CT
        is up), then the .cpr.json's embedded *series*. If ALL three are present →
        just those; if ANY is missing → append the source UID after whatever
        exists; if none → the UID alone."""
        emb = series or {}
        live = {}
        if self._shell is not None and hasattr(self._shell, "coronary_series_meta"):
            try:
                live = self._shell.coronary_series_meta(uid) or {}
            except Exception:                               # noqa: BLE001
                live = {}

        def pick(*keys):
            for src in (live, emb):                         # live wins
                for k in keys:
                    v = src.get(k)
                    if v not in (None, ""):
                        return str(v).strip()
            return ""

        name = pick("patient", "patient_id")
        date = pick("date", "study_date")                   # 8-digit, no dashes
        sn = pick("series_number")
        se = f"Se{sn}" if sn else ""
        parts = [p for p in (name, date, se) if p]
        short = (uid[:16] + "…") if uid and len(uid) > 16 else uid
        if len(parts) == 3:                                 # all present → no ID
            return "CT: " + " ".join(parts)
        if parts:                                           # some missing → + ID
            return "CT: " + " ".join(parts) + " / " \
                + (short or t("(ID不明)"))
        return "CT: " + (short or t("(未指定)"))             # nothing → ID only

    # The rest of the panel keeps its single-tree code; these properties point it
    # at the ACTIVE CT's bundle. _populate / overlay iterate self._by_uid directly.
    @property
    def _tree(self): return self._bundle()["tree"]
    @_tree.setter
    def _tree(self, v): self._bundle()["tree"] = v

    @property
    def _hidden(self): return self._bundle()["hidden"]
    @_hidden.setter
    def _hidden(self, v): self._bundle()["hidden"] = v

    @property
    def _targets(self): return self._bundle()["targets"]
    @_targets.setter
    def _targets(self, v): self._bundle()["targets"] = v

    @property
    def _full_lv_data(self): return self._bundle()["full_lv"]
    @_full_lv_data.setter
    def _full_lv_data(self, v): self._bundle()["full_lv"] = v

    @property
    def _territory(self): return self._bundle()["territory"]
    @_territory.setter
    def _territory(self, v): self._bundle()["territory"] = v

    @property
    def _myo_ml(self): return self._bundle()["myo_ml"]
    @_myo_ml.setter
    def _myo_ml(self, v): self._bundle()["myo_ml"] = v

    @property
    def _ct_dir(self): return self._bundle()["ct_dir"]
    @_ct_dir.setter
    def _ct_dir(self, v): self._bundle()["ct_dir"] = v

    @property
    def _last_path(self): return self._bundle()["last_path"]
    @_last_path.setter
    def _last_path(self, v): self._bundle()["last_path"] = v

    @property
    def _ct_uid(self): return self._active_uid
    @_ct_uid.setter
    def _ct_uid(self, v): self._use_uid(v)

    def __init__(self, shell):
        super().__init__(t("Coronary Tree"))
        self._shell = shell
        # Per-source-CT state: each 3-D CT gets its OWN CoronaryTree + Territory,
        # keyed by its series UID, so two CTs' CPRs never merge. `_tree`, `_hidden`,
        # `_targets`, `_full_lv_data`, `_territory`, `_myo_ml`, `_ct_dir`,
        # `_last_path` are PROPERTIES onto the ACTIVE CT's bundle (see below).
        self._by_uid: dict[str, dict] = {}
        self._active_uid: str = ""          # the CT currently being worked on
        self._building = False
        self._overlay_on: bool = False      # ツリー表示 toggle (off until pressed)
        self._overlay_btn = None            # the ツリー表示/非表示 toggle button
        self._targets_shown = True          # global show/hide of all Target overlays
        self._target_mode = False           # click-a-vessel-to-set-a-target toggle
        self._target_btn = None
        # Appearance params (line / territory / target colours) from Settings ▸
        # Coronary Tree / Territory. Target colour is now PER target number (1..6).
        self._coro_params = settings_mod.load_coronary_params()
        self.setAcceptDrops(True)           # drag .cpr.json onto the panel

        central = QWidget()
        self.setWidget(central)
        central.setMinimumWidth(200)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(3)

        # Compact, docked-friendly rows: a fixed prefix label + short buttons, so
        # the text never elides in a narrow left-dock.
        def _row(prefix, specs):
            row = QHBoxLayout()
            row.setSpacing(3)
            lbl = QLabel(prefix)
            lbl.setStyleSheet("font-weight:bold;")
            row.addWidget(lbl)
            made = []
            for label, tip, fn, checkable in specs:
                b = QPushButton(label)
                b.setToolTip(tip)
                b.setCheckable(checkable)
                if checkable:
                    b.clicked.connect(fn)
                else:
                    b.clicked.connect(fn)
                b.setStyleSheet("padding: 2px 6px;")
                row.addWidget(b)
                made.append(b)
            row.addStretch(1)
            outer.addLayout(row)
            return made

        # CPR row: build the tree from per-vessel .cpr.json files.
        _row(t("CPR："), [
            (t("読込"), t("枝ごとの .cpr.json を複数選択で読込（未接続で追加）"),
             self._load_cpr, False),
            (t("接続"), t("読み込んだ枝を最近接端点で接続 (5mm以内)"),
             self._connect, False),
            (t("消去"),
             t("下の血管リストで選択した血管（下流の枝を含む）を一覧から消去。"
               "Ctrlで複数選択・Shiftで範囲選択（元のCPRファイルは残る）"),
             self._delete_selected, False),
        ])
        # Tree row: overlay toggle / save / load / clear.
        tree_btns = _row(t("ツリー："), [
            # Default state is OFF (overlay hidden), so the button shows the ACTION
            # it performs: 表示 (= click to show). _set_overlay flips it to 非表示.
            (t("表示"),
             t("冠動脈ツリーを3DCTに重畳 (再クリックで非表示)。未読込なら自動読込"),
             self._toggle_overlay, False),
            (t("保存"), t("冠動脈ツリーを .corotree.json に保存"),
             self._save_as, False),
            (t("読込み"), t("保存した .corotree.json を読込"), self._load, False),
            (t("全消去"), t("全ての血管を消去 (.cpr.json は残る)"),
             self._clear_all, False),
        ])
        self._overlay_btn = tree_btns[0]     # relabelled 表示/非表示
        # VR row (built by hand — it carries a spin box, not just buttons): show /
        # hide the right-pane VR, switch 内腔 (lumen) ⇔ シェル (Epi surface + shell),
        # and set the outward shell thickness (0–20 mm) when シェル is selected.
        self._vr_shown = True                # matches apply_full_lv's auto-on VR
        self._vr_shell_mode = False          # default 内腔VR (True = シェルVR)
        vr_row = QHBoxLayout()
        vr_row.setSpacing(3)
        vr_lbl = QLabel(t("VR："))
        vr_lbl.setStyleSheet("font-weight:bold;")
        vr_row.addWidget(vr_lbl)
        self._vr_show_btn = QPushButton(t("表示/非表示"))
        self._vr_show_btn.setToolTip(t("右画面のVR (立体表示) を表示/非表示"))
        self._vr_show_btn.setStyleSheet("padding: 2px 6px;")
        self._vr_show_btn.clicked.connect(self._toggle_vr)
        vr_row.addWidget(self._vr_show_btn)
        self._vr_shell_btn = QPushButton(t("シェルVR"))
        self._vr_shell_btn.setToolTip(
            t("VRの範囲を切替: 内腔VR (心内腔) ⇔ シェルVR (Epi表面+外側の殻)。"
              "シェルVR時は右のシェルVR範囲で殻の厚みを調整"))
        self._vr_shell_btn.setStyleSheet("padding: 2px 6px;")
        self._vr_shell_btn.clicked.connect(self._toggle_vr_shell)
        vr_row.addWidget(self._vr_shell_btn)
        self._vr_shell_lbl = QLabel(t("シェルVR範囲"))
        vr_row.addWidget(self._vr_shell_lbl)
        self._vr_shell_spin = QSpinBox()
        self._vr_shell_spin.setRange(0, 20)
        self._vr_shell_spin.setSingleStep(1)
        self._vr_shell_spin.setSuffix(" mm")
        self._vr_shell_spin.setValue(10)
        self._vr_shell_spin.setToolTip(t("Epi表面から外側の殻の厚み (0–20mm)"))
        self._vr_shell_spin.valueChanged.connect(self._on_vr_shell_mm)
        vr_row.addWidget(self._vr_shell_spin)
        vr_row.addStretch(1)
        outer.addLayout(vr_row)
        self._sync_vr_row()
        # Target row: set / show-hide / delete-last / delete-all.
        tgt_btns = _row(t("ターゲット："), [
            (t("設定"),
             t("ON中、CT (VR/短軸) 上の血管をクリックで Target 設定。遠位の"
               "灌流域 (mL・心筋%) を表と画像に表示"),
             self._toggle_target_mode, True),
            (t("表示/非表示"), t("設定した Target/灌流域の表示を切替",),
             self._toggle_targets_shown, False),
            (t("最後の1つ削除"), t("最後に設定した Target を削除"),
             self._delete_last_target, False),
            (t("全削除"), t("設定した Target を全て削除"),
             self._clear_targets, False),
        ])
        self._target_btn = tgt_btns[0]
        self._targets_vis_btn = tgt_btns[1]      # relabelled 表示/非表示
        self._targets_vis_btn.setText(
            t("非表示") if self._targets_shown else t("表示"))

        split = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(split, 1)

        self._tree_w = QTreeWidget()
        self._tree_w.setColumnCount(4)
        self._tree_w.setHeaderLabels(
            [t("血管"), t("役割"), t("分岐"), t("心筋量")])
        self._tree_w.setColumnWidth(0, 130)
        self._tree_w.setColumnWidth(1, 70)
        self._tree_w.setColumnWidth(2, 70)
        # Ctrl = arbitrary multi-select, Shift = contiguous range (for 消去).
        self._tree_w.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self._tree_w.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree_w.customContextMenuRequested.connect(self._menu)
        self._tree_w.itemChanged.connect(self._on_item_changed)
        self._tree_w.itemSelectionChanged.connect(self._on_selection)
        self._tree_w.itemDoubleClicked.connect(lambda *_: self._rename())
        split.addWidget(self._tree_w)

        # ---- Territory results (populated when a FullLv is loaded) ----
        terr = QWidget()
        tv = QVBoxLayout(terr)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(2)
        self._terr_lbl = QLabel(t("Territory: FullLv 未読込"))
        self._terr_lbl.setStyleSheet("font-weight:bold;")
        tv.addWidget(self._terr_lbl)
        self._targets_w = QTreeWidget()
        # Column 0 = show/hide CHECKBOX (its own column), 1 = Target number, then
        # 血管 / 位置 / 心筋% / 灌流域mL.
        self._targets_w.setColumnCount(6)
        self._targets_w.setHeaderLabels(
            ["", t("Target"), t("血管"), t("位置"), t("心筋 %"), t("灌流域 mL")])
        self._targets_w.setColumnWidth(0, 28)
        self._targets_w.setColumnWidth(1, 52)
        self._targets_w.setColumnWidth(2, 96)
        self._targets_w.setColumnWidth(3, 52)
        self._targets_w.setColumnWidth(4, 56)
        self._targets_w.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._targets_w.customContextMenuRequested.connect(self._targets_menu)
        self._targets_w.itemSelectionChanged.connect(self._on_target_selection)
        # Column-0 checkbox = per-target show/hide (checked = shown).
        self._targets_w.itemChanged.connect(self._on_target_item_changed)
        tv.addWidget(self._targets_w, 1)
        self._terr_out = QPlainTextEdit()
        self._terr_out.setReadOnly(True)
        self._terr_out.setFont(QFont("Consolas", 9))
        self._terr_out.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._terr_out.setMaximumHeight(150)
        tv.addWidget(self._terr_out)
        split.addWidget(terr)
        split.setSizes([320, 260])

        self._hint = QLabel("")
        self._hint.setStyleSheet("color:#888;")
        outer.addWidget(self._hint)
        self._refresh_hint()

        # Push the on-image overlay whenever the tree, selection or a vessel's
        # visibility changes (the shell fans it out to every CT viewer).
        self.treeChanged.connect(self._push_overlay)
        self.treeChanged.connect(self._compute_territory)   # tree edit → re-assign
        self.vesselSelected.connect(lambda *_: self._push_overlay())

    # ------------------------------------------------------------- model
    @property
    def tree(self) -> CoronaryTree:
        return self._tree

    def _unique_vid(self) -> str:
        # Unique across EVERY loaded CT (案A), not just the active tree, so a
        # vessel id resolves to one CT when a viewer reports it (VR hide / target).
        used = set()
        for b in self._by_uid.values():
            used.update(b["tree"].vessels)
        n = len(used) + 1
        vid = f"v{n}"
        while vid in used:
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
        """Load the given .cpr.json paths (file dialog or drag & drop) as
        unconnected vessels — each routed to the tree of ITS OWN source CT (by
        series UID), so CPRs from different 3-D CTs stay in separate CoronaryTrees
        rather than merging. The LAST CT loaded becomes the active one."""
        added, errs, skipped = 0, [], 0
        first_uid = None
        last_uid = None
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
                ser = data.get("series") or {}
                uid = ser.get("series_uid", "") or ""
                src_dir = data.get("src_dir", "") or ""
                # Route to THIS CT's tree (create it on first sight).
                self._use_uid(uid, ct_dir=src_dir, series=ser)
                last_uid = uid
                if first_uid is None:
                    first_uid = uid
                # strip ".cpr.json" → vessel name
                name = os.path.splitext(
                    os.path.splitext(os.path.basename(p))[0])[0]
                # Skip a name already in THIS CT's tree (dup drop / corotree + cpr).
                if name in {v.name for v in self._tree.vessels.values()}:
                    skipped += 1
                    continue
                ctrl = np.asarray(ctrl, float)
                cl = CenterLine.from_points(ctrl, step_mm=_CPR_STEP_MM)
                new_vid = self._unique_vid()
                self._tree.add_vessel(new_vid, name or "vessel",
                                      "branch", cl.points, ctrl=ctrl)
                self._bundle()["paths"][new_vid] = p   # for in-place overwrite
                added += 1
                self._last_path = p
            except (OSError, ValueError) as exc:            # noqa: BLE001
                errs.append(f"{os.path.basename(p)}: {exc}")
        if last_uid is not None:             # active = the last CT loaded
            self._use_uid(last_uid)
        self._populate()
        self.treeChanged.emit()
        # CPR読込 just loads the vessels; the 3-D CT + overlay come up only when
        # the user presses ツリー表示 (which auto-loads the CT if needed). If the
        # overlay is already ON, refresh it so the newly-loaded vessels appear.
        if self._overlay_on:
            self._push_overlay()
        msg = t("{n} 本を読込（役割を設定して「接続」→「ツリー表示」）。", n=added)
        if errs:
            msg += " " + t("失敗 {e} 件。", e=len(errs))
        self._hint.setText(msg)
        if errs:
            self._warn("\n".join(errs[:8]))

    def register_cpr(self, data, name=None, path=None) -> bool:
        """Register an in-memory CPR (a viewer's 登録 button) as an unconnected
        vessel — same routing as loading its .cpr.json (to the source-CT bundle by
        series UID) but without a file. A duplicate name is auto-uniquified rather
        than skipped, since 登録 is a deliberate single action. *path* = the .cpr.json
        it was saved to (kept for in-place overwrite). Returns True on add."""
        if not isinstance(data, dict) or data.get("format") != "MDV-CPR":
            self._warn(t("CPR形式ではありません。"))
            return False
        ctrl = data.get("ctrl")
        if not ctrl or len(ctrl) < 2:
            self._warn(t("中心点が不足しています。"))
            return False
        ser = data.get("series") or {}
        uid = ser.get("series_uid", "") or ""
        src_dir = data.get("src_dir", "") or ""
        self._use_uid(uid, ct_dir=src_dir, series=ser)
        nm = (str(name).strip() if name else "") or "vessel"
        existing = {v.name for v in self._tree.vessels.values()}
        if nm in existing:                       # uniquify: "LAD" → "LAD (2)"
            base, k = nm, 2
            while f"{base} ({k})" in existing:
                k += 1
            nm = f"{base} ({k})"
        ctrl_arr = np.asarray(ctrl, float)
        cl = CenterLine.from_points(ctrl_arr, step_mm=_CPR_STEP_MM)
        new_vid = self._unique_vid()
        self._tree.add_vessel(new_vid, nm, "branch", cl.points, ctrl=ctrl_arr)
        if path:
            self._bundle()["paths"][new_vid] = path   # enable in-place overwrite
        self._populate()
        self.treeChanged.emit()
        # Always push: even with the 2-D ツリー表示 toggle OFF, the VR coronary tubes
        # are drawn (force=True), so a just-Saved CPR shows on the VR immediately.
        self._push_overlay()
        self._hint.setText(t("「{n}」を登録しました（役割を設定して「接続」→"
                             "「ツリー表示」）。", n=nm))
        # Surface the panel so the just-added vessel is visible.
        self.show()
        self.raise_()
        return True

    # ---- edit an already-saved CPR line (Resume / overwrite) --------------
    def cpr_resume_data(self, vid: str) -> dict | None:
        """The data needed to RE-EDIT vessel *vid*: its control points, source-CT
        UID and the .cpr.json path (if known). None if the vid isn't found."""
        uid = self._ct_of_vessel(vid)
        if not uid:
            return None
        b = self._by_uid[uid]
        v = b["tree"].vessels.get(vid)
        if v is None:
            return None
        ctrl = getattr(v, "ctrl", None)
        if ctrl is None or len(ctrl) < 2:        # fall back to the sampled line
            ctrl = v.points
        return {
            "uid": uid,
            "name": v.name,
            "ctrl": [list(map(float, np.asarray(q, float))) for q in ctrl],
            "path": b["paths"].get(vid, "") or "",
            "series": dict(b.get("series") or {}),
            "src_dir": b.get("ct_dir", "") or "",
        }

    def update_vessel_cpr(self, vid: str, ctrl, path: str = "") -> bool:
        """Overwrite vessel *vid*'s centreline IN PLACE from edited control points
        (keeps its name / role / parent), refresh the overlay, and remember the
        saved .cpr.json *path*. Used by the CPR-line right-click ▸ 上書き保存."""
        uid = self._ct_of_vessel(vid)
        if not uid:
            return False
        b = self._by_uid[uid]
        v = b["tree"].vessels.get(vid)
        if v is None or ctrl is None or len(ctrl) < 2:
            return False
        ctrl_arr = np.asarray(ctrl, float)
        cl = CenterLine.from_points(ctrl_arr, step_mm=_CPR_STEP_MM)
        v.points = cl.points
        v.ctrl = ctrl_arr
        if path:
            b["paths"][vid] = path
        self._populate()
        self.treeChanged.emit()
        self._push_overlay()
        self._hint.setText(t("「{n}」を上書き保存しました。", n=v.name))
        return True

    def _toggle_overlay(self):
        """ツリー表示 / ツリー非表示 button: toggle the whole coronary overlay on
        the 3-D CT. Turning it ON also (re)opens the source CT if it isn't
        loaded, so one click gives a CT with the tree drawn on it."""
        if not self._tree.vessels:
            self._warn(t("先に CPR を読み込んでください。"))
            return
        self._set_overlay(not self._overlay_on)

    def _set_overlay(self, on: bool):
        """Set the overlay on/off, update the button label, (re)show the source
        CT when turning on, and push the overlay (draw when on, clear when off).
        Used by the toggle button and by ツリー読込 / .corotree.json drop (on)."""
        self._overlay_on = bool(on)
        if self._overlay_btn is not None:
            self._overlay_btn.setText(t("非表示") if on else t("表示"))
        if on:
            # overlay_spec now returns the vessels; showing the CT triggers the
            # overlay refresh (and again once the volume finishes decoding).
            st = self._show_source_ct(prompt=True)
            self._hint.setText(
                t("ツリー表示: {s}", s=st) if st
                else t("元の3DCTが特定できません。CPRを読み込み直すか"
                       "フォルダを選択してください。"))
        else:
            self._push_overlay()             # overlay_spec is now empty → clear
            self._hint.setText(t("ツリー非表示"))
        self._push_territory()               # tint follows the overlay toggle

    # ------------------------------------------------------------- VR row
    def _sync_vr_row(self) -> None:
        """Update the VR row's labels / enabled state to the current mode: the
        VR show/hide button shows the ACTION for the current state (shown → 非表示),
        the シェル/内腔 button shows the ACTIVE mode, and the シェルVR範囲 spin is
        enabled only while シェルVR is selected."""
        if getattr(self, "_vr_shell_btn", None) is None:
            return
        if getattr(self, "_vr_show_btn", None) is not None:
            self._vr_show_btn.setText(
                t("非表示") if getattr(self, "_vr_shown", True) else t("表示"))
        shell = bool(getattr(self, "_vr_shell_mode", True))
        self._vr_shell_btn.setText(t("シェルVR") if shell else t("内腔VR"))
        self._vr_shell_lbl.setEnabled(shell)
        self._vr_shell_spin.setEnabled(shell)

    def _toggle_vr(self) -> None:
        """VR ▸ 表示/非表示 — show or hide the right-pane Volume Rendering."""
        self._vr_shown = not bool(getattr(self, "_vr_shown", True))
        self._sync_vr_row()                  # relabel 表示 ⇄ 非表示
        if self._shell is not None and hasattr(self._shell, "coronary_vr_visible"):
            self._shell.coronary_vr_visible(self._vr_shown)

    def _toggle_vr_shell(self) -> None:
        """VR ▸ 内腔VR/シェルVR — switch the VR crop between the LV lumen and the
        Epi surface + outward shell."""
        self._vr_shell_mode = not bool(getattr(self, "_vr_shell_mode", True))
        self._sync_vr_row()
        if self._shell is not None and hasattr(self._shell, "coronary_vr_shell"):
            self._shell.coronary_vr_shell(self._vr_shell_mode)

    def _on_vr_shell_mm(self, mm: int) -> None:
        """VR ▸ シェルVR範囲 — outward shell thickness (mm) changed."""
        if self._shell is not None and hasattr(self._shell, "coronary_vr_shell_mm"):
            self._shell.coronary_vr_shell_mm(float(mm))

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
    @staticmethod
    def _collect_cpr_paths(urls) -> list:
        """Every .cpr.json among the dropped URLs — files directly, and (for a
        dropped FOLDER) every .cpr.json found under it recursively. Sorted for a
        stable load order."""
        out = []
        for u in urls or []:
            f = u.toLocalFile()
            if not f:
                continue
            if os.path.isdir(f):
                for root, _dirs, names in os.walk(f):
                    for nm in names:
                        if nm.lower().endswith(".cpr.json"):
                            out.append(os.path.join(root, nm))
            elif f.lower().endswith(".cpr.json"):
                out.append(f)
        # De-dup on a normalised key (a folder + a file inside it can overlap,
        # and QUrl vs os.walk spell separators differently) but keep one path.
        seen = {}
        for p in out:
            seen.setdefault(os.path.normcase(os.path.normpath(p)), p)
        return sorted(seen.values())

    def dragEnterEvent(self, e):
        """Accept a drag that carries at least one .cpr.json file OR a folder
        (which may contain .cpr.json files)."""
        md = e.mimeData()
        if md is not None and md.hasUrls() and any(
                (u.toLocalFile() and (os.path.isdir(u.toLocalFile())
                 or u.toLocalFile().lower().endswith(".cpr.json")))
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
        """Load every .cpr.json dropped onto the panel — individual files and/or
        all .cpr.json inside any dropped folder."""
        md = e.mimeData()
        paths = self._collect_cpr_paths(md.urls()) \
            if (md is not None and md.hasUrls()) else []
        if paths:
            e.acceptProposedAction()
            self._load_cpr_paths(paths)
        else:
            super().dropEvent(e)

    def _connect(self):
        """Grow the tree: attach every loose branch to the nearest connected
        vessel by its nearest endpoint (5 mm). Roots must be set first."""
        if not self._tree.vessels:
            self._warn(t("先に CPR を読み込んでください。"))
            return
        if not self._tree.roots():
            self._warn(t("ルート (LM / LAD / LCX / RCA) を1本以上設定して"
                         "ください。血管を右クリック →「役割」で設定できます。"))
            return
        res = self._tree.connect_all(snap_tol_mm=_CONNECT_TOL_MM)
        self._populate()
        self.treeChanged.emit()
        msg = t("{c} 本を接続しました。", c=len(res["connected"]))
        if res["unconnected"]:
            msg += t(" 未接続 {u} 本（5mm以内に幹/枝がありません。近い枝を"
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
    # --- configurable colours (Settings ▸ Coronary Tree / Territory) ---------
    def _root_color(self, role: str) -> str:
        """Line / tube colour for a root system — user-set in Settings, falling
        back to the shipped ROOT_COLORS (and to grey for an unknown role)."""
        lc = (self._coro_params or {}).get("line_colors") or {}
        base = role.split("-")[-1] if role else role     # LM-LAD → LAD (legacy)
        return lc.get(role) or lc.get(base) \
            or ROOT_COLORS.get(role) or ROOT_COLORS.get(base) or "#888888"

    def _target_hex(self, n: int) -> str:
        """Territory/marker colour for Target number *n* (1-based), by 6-slot
        palette from Settings (wraps past 6)."""
        tc = (self._coro_params or {}).get("target_colors") \
            or settings_mod.CORONARY_PARAMS_DEFAULT["target_colors"]
        try:
            return tc[(int(n) - 1) % 6]
        except (TypeError, ValueError, IndexError):
            return tc[0]

    def _target_rgba(self, n: int):
        """Target *n* fill as an (r,g,b,a) 0-1 tuple (alpha = _TARGET_ALPHA)."""
        c = QColor(self._target_hex(n))
        if not c.isValid():
            c = QColor("#e06666")
        return (c.redF(), c.greenF(), c.blueF(), _TARGET_ALPHA)

    def coronary_params_refresh(self) -> None:
        """Re-read the appearance params (after a Settings change) and redraw the
        tree + overlay + territory with the new colours / sizes."""
        self._coro_params = settings_mod.load_coronary_params()
        if not self._building:
            self._populate()                     # line colours in the list
        self._push_overlay()                     # line colours on MPR/VR
        self._push_territory()                    # territory + target colours

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
        it = QTreeWidgetItem([v.name, role_txt, self._junction_text(vid),
                              self._myo_text(vid)])
        it.setData(0, _UID_ROLE, vid)
        it.setData(0, _CTUID_ROLE, self._active_uid)   # which CT this vessel is in
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        it.setCheckState(0, Qt.CheckState.Unchecked
                         if vid in self._hidden else Qt.CheckState.Checked)
        it.setForeground(0, QColor(self._root_color(self._root_role(vid))))
        # Monospace 心筋量 column so the padded % / mL line up vertically.
        mf = QFont("Consolas")
        mf.setStyleHint(QFont.StyleHint.Monospace)
        it.setFont(3, mf)
        return it

    def _populate(self):
        self._building = True
        self._tree_w.blockSignals(True)
        self._tree_w.clear()
        saved_active = self._active_uid
        # One top-level group per source CT — collapsible (▲▼) with a show/hide-all
        # checkbox and a 症例/検査日/Se# label — shown even for a SINGLE CT so the
        # source 3-D CT is always identified. Only CTs that actually hold vessels
        # get a header (no empty placeholder when nothing is loaded).
        uids = [u for u, b in self._by_uid.items() if b["tree"].vessels]
        for uid in uids:
            self._active_uid = uid           # route self._tree/_hidden/… to this CT
            b = self._bundle()
            tree = b["tree"]
            lbl = self._uid_label(uid, b.get("series"))   # live meta + embedded
            b["label"] = lbl                               # cache for relabel diff
            grp = QTreeWidgetItem([lbl, "", "", ""])
            grp.setData(0, _GRP_ROLE, uid)
            grp.setData(0, _CTUID_ROLE, uid)
            grp.setFlags(grp.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            grp.setCheckState(0, Qt.CheckState.Checked if b.get("visible", True)
                              else Qt.CheckState.Unchecked)
            gf = grp.font(0); gf.setBold(True); grp.setFont(0, gf)
            grp.setForeground(0, QColor("#1a3a6b"))
            self._tree_w.addTopLevelItem(grp)
            parent = grp
            loose = set(tree.unconnected())

            def add_recursive(vid, parent_item, _tree=tree):
                it = self._make_item(vid)
                if parent_item is None:
                    self._tree_w.addTopLevelItem(it)
                else:
                    parent_item.addChild(it)
                for c in _tree.children(vid):
                    add_recursive(c, it, _tree)

            # Clinical order LM → LAD → LCX → RCA (LM on top), not alphabetical;
            # any other root role sorts after, then by name for stability.
            def _root_key(r):
                role = tree.vessels[r].role if r in tree.vessels else ""
                return (_ROOT_ORDER.get(role, 99), tree.vessels[r].name
                        if r in tree.vessels else "")

            for r in sorted(tree.roots(), key=_root_key):
                add_recursive(r, parent)
            if loose:
                lg = QTreeWidgetItem([t("（未接続 {n}）", n=len(loose)), "", "", ""])
                lg.setData(0, _CTUID_ROLE, uid)
                lg.setForeground(0, QColor("#b00000"))
                if parent is None:
                    self._tree_w.addTopLevelItem(lg)
                else:
                    parent.addChild(lg)
                for vid in tree.vessels:
                    if vid in loose:
                        lg.addChild(self._make_item(vid))
        self._active_uid = saved_active
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
    def _activate_item_ct(self, item) -> None:
        """Switch the active CT bundle to the one the given tree item belongs to
        (案A: each item carries its source-CT UID). No-op for a None item or one
        whose CT is already active."""
        if item is None:
            return
        uid = item.data(0, _CTUID_ROLE)
        if uid and uid != self._active_uid and uid in self._by_uid:
            self._active_uid = uid

    def _on_selection(self):
        self._activate_item_ct(self._tree_w.currentItem())
        self.vesselSelected.emit(self.selected_vid() or "")

    def _ct_of_vessel(self, vid: str) -> str:
        """Which CT bundle owns a vid (first match); "" if none."""
        for uid, b in self._by_uid.items():
            if vid in b["tree"].vessels:
                return uid
        return ""

    def hide_vessel(self, vid: str) -> None:
        """Hide one vessel's centreline (VR right-click ▸ この血管を非表示). Unchecks
        it in the tree; the MPR overlay + VR tube then drop it."""
        uid = self._ct_of_vessel(vid)
        if not uid:
            return
        b = self._by_uid[uid]
        if vid in b["hidden"]:
            return
        b["hidden"].add(vid)
        self._populate()                     # reflect the unchecked state
        self._push_overlay()

    def _on_item_changed(self, item, _col):
        if self._building:
            return
        grp_uid = item.data(0, _GRP_ROLE)
        if grp_uid and grp_uid in self._by_uid:
            # CT group header checkbox → show/hide ALL of that CT's coronaries.
            on = item.checkState(0) == Qt.CheckState.Checked
            self._by_uid[grp_uid]["visible"] = on
            self._push_overlay()
            return
        vid = item.data(0, _UID_ROLE)
        if vid:
            self._activate_item_ct(item)
            on = item.checkState(0) == Qt.CheckState.Checked
            if on:
                self._hidden.discard(vid)
            else:
                self._hidden.add(vid)
            self.visibilityChanged.emit(vid, on)
            self._push_overlay()

    # --------------------------------------------------------- overlay
    def uid_for_series(self, series_uid: str) -> str:
        """Which loaded-CT bundle a pane's shown SeriesUID belongs to (base match,
        tolerating the "#…" suffix); "" if none. Lets the shell push each pane the
        overlay of the CT it is actually displaying (per-CT trees, 案A)."""
        if not series_uid:
            return ""
        base = series_uid.split("#", 1)[0]
        for uid in self._by_uid:
            if uid == series_uid or uid.split("#", 1)[0] == base:
                return uid
        return ""

    def overlay_spec(self, force: bool = False, uid: str | None = None) -> list:
        """The on-image overlay: one entry per VISIBLE vessel with 2+ points —
        its world-mm centreline, root colour and name — for the CT viewers to
        reproject onto their MPR planes. The currently-selected vessel is
        flagged so the viewer can highlight it. Empty while the ツリー表示 toggle
        is OFF (so the overlay only shows on demand) — unless *force* (the VR
        pane always shows the colour-coded coronaries). *uid* selects which
        source-CT's tree to emit (default: the active one); a hidden CT group
        (visible=False) emits nothing."""
        if not self._overlay_on and not force:
            return []
        if uid is None:
            uid = self._active_uid
        b = self._by_uid.get(uid)
        if b is None or not b.get("visible", True):
            return []
        saved = self._active_uid
        self._active_uid = uid
        try:
            sel = self.selected_vid()
            out = []
            for vid, v in b["tree"].vessels.items():
                if vid in b["hidden"]:
                    continue
                pts = np.asarray(v.points, float)
                if pts.ndim != 2 or pts.shape[0] < 2:
                    continue
                out.append({
                    "vid": vid,
                    "name": v.name,
                    "points": pts.tolist(),
                    "root": self._root_role(vid),   # root system "LM"/"LAD"/…
                    "role": v.role,                 # OWN role (LM trunk ×1.5)
                    "color": self._root_color(self._root_role(vid)),
                    "selected": (vid == sel),
                })
            return out
        finally:
            self._active_uid = saved

    def refresh_ct_labels(self) -> None:
        """Re-derive each CT group's label from the now-loaded CT's live metadata
        (the shell calls this after a source CT is shown); repopulate only if a
        label actually changed, so it doesn't fight the user's interaction."""
        changed = False
        for uid, b in self._by_uid.items():
            if not b["tree"].vessels:
                continue
            new = self._uid_label(uid, b.get("series"))
            if new != b.get("label"):
                b["label"] = new
                changed = True
        if changed and not self._building:
            self._populate()

    def _push_overlay(self):
        """Ask the shell to fan the current overlay out to every CT viewer."""
        if self._shell is not None \
                and hasattr(self._shell, "coronary_overlay_refresh"):
            try:
                self._shell.coronary_overlay_refresh()
            except Exception:                            # noqa: BLE001
                pass

    # -------------------------------------------------------- territory
    def load_full_lv(self, data_or_path) -> bool:
        """Load a FullLv (parsed dict OR a .FullLv.json path) as the LV side of
        the territory analysis, then compute + show the report. Public — the
        shell calls it on a .FullLv.json drop."""
        data = data_or_path
        if isinstance(data_or_path, str):
            try:
                with open(data_or_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError) as exc:            # noqa: BLE001
                self._warn(t("FullLv 読込失敗: {e}", e=str(exc)))
                return False
        if not full_lv_mod.is_full_lv(data):
            self._warn(t("FullLv (.FullLv.json) 形式ではありません。"))
            return False
        fser = full_lv_mod.series_uid(data)
        # 案A: attach the FullLv to the CT bundle it belongs to (by series UID).
        # If that CT is loaded, switch to it FIRST so territory lands on the right
        # tree; otherwise keep the active CT and warn on a UID mismatch.
        match_uid = self.uid_for_series(fser) if fser else ""
        if match_uid:
            self._use_uid(match_uid)
        elif self._active_uid and fser \
                and self._active_uid.split("#", 1)[0] != fser.split("#", 1)[0]:
            if QMessageBox.warning(
                    self, t("Coronary Tree"),
                    t("冠動脈ツリーと FullLv の元CT(SeriesUID)が一致しません。"
                      "結果が正しくない可能性があります。続行しますか？"),
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) \
                    != QMessageBox.StandardButton.Yes:
                return False
        if not self._active_uid and fser:
            self._use_uid(fser)
        self._full_lv_data = data
        if not self._ct_dir:
            self._ct_dir = ((data.get("src") or {}).get("src_dir")) or ""
        self._targets = []
        self._compute_territory()
        return True

    def _compute_territory(self) -> None:
        """(Re)build the TerritoryEngine from the current tree + loaded FullLv and
        refresh the report. Called on FullLv load and whenever the tree changes
        (assignment depends on the centrelines)."""
        self._territory = None
        self._myo_ml = None
        if self._full_lv_data is None:
            self._terr_lbl.setText(t("Territory: FullLv 未読込"))
            self._terr_out.setPlainText("")
            self._refresh_targets_table()
            return
        if not self._tree.vessels:
            self._terr_lbl.setText(t("Territory: 血管なし"))
            self._terr_out.setPlainText("")
            return
        lvf = LVFunction.from_full(self._full_lv_data)
        if lvf is None:
            self._terr_lbl.setText(t("Territory: 心筋を構築できません"))
            self._terr_out.setPlainText("")
            return
        # The engine build (nearest-centreline assignment of every myocardial
        # voxel) + the tree/overlay refresh is the slow part — run it behind a busy
        # window so the tree display doesn't look stuck.
        def _work():
            lines, eng = format_territory_report(self._tree, lvf)
            self._territory = eng
            self._myo_ml = float(lvf.myocardial_volume_ml())
            self._terr_lbl.setText(
                t("Territory: 心筋(緻密層) {v:.1f} mL", v=self._myo_ml))
            self._terr_out.setPlainText("\n".join(lines))
            self._populate()             # fill the 心筋量 column (engine now exists)
            self._recompute_targets()
            self._push_overlay()
            self._push_territory()
        try:
            self._run_busy(t("Territory を計算中…"), _work)
        except Exception as exc:                            # noqa: BLE001
            self._terr_out.setPlainText(t("解析失敗: {e}", e=str(exc)))
            return

    def _myo_text(self, vid: str) -> str:
        """Territory at the vessel's PROXIMAL point = its WHOLE subtree from the
        ostium, as 'ml mL / pct%'. Empty when no engine. A parent includes its
        children, so values nest and the sum can exceed 100% (expected)."""
        eng = self._territory
        if eng is None or vid not in self._tree.vessels:
            return ""
        try:
            _mask, ml = eng.territory(vid, 0)
        except Exception:                                # noqa: BLE001
            return ""
        pct = (100.0 * ml / self._myo_ml) if self._myo_ml else 0.0
        # Integer % first (right-aligned 3 wide so the % sign lines up), then the
        # mL starts right after "/ " (monospace column → the mL start lines up too,
        # with only a single space after the slash).
        return f"{round(pct):>3d}% / {ml:.1f}mL"

    def _recompute_targets(self) -> None:
        """Recompute each target's mL / % against the current engine, dropping any
        whose vessel no longer exists; then refill the table."""
        eng = self._territory
        good = []
        for tg in self._targets:
            vid = tg["vid"]
            if eng is None or vid not in self._tree.vessels:
                continue
            n = self._tree.vessels[vid].n
            idx = max(0, min(int(tg["idx"]), n - 1))
            _mask, ml = eng.territory(vid, idx)
            pct = (100.0 * ml / self._myo_ml) if self._myo_ml else 0.0
            good.append({"vid": vid, "idx": idx, "ml": ml, "pct": pct,
                         "hidden": tg.get("hidden", False)})
        self._targets = good
        self._refresh_targets_table()

    def _refresh_targets_table(self) -> None:
        self._targets_w.blockSignals(True)      # our own setCheckState re-entrancy
        self._targets_w.clear()
        for i, tg in enumerate(self._targets, 1):
            v = self._tree.vessels.get(tg["vid"])
            # Show the vessel's OWN name exactly as the tree does (e.g. S@L… / D@L…)
            # — short_vessel_name mangles already-short names (S@LAD → LAD) and the
            # root-role prefix was misleading for a branch.
            label = v.name if v is not None else tg["vid"]
            # 位置 = arc-length from the vessel's PROXIMAL end as a % of its length
            # (points are uniform arc-length samples, so idx maps linearly).
            n = v.n if v is not None else 1
            pos = (100.0 * int(tg["idx"]) / (n - 1)) if n > 1 else 0.0
            # Col 0 = checkbox only (empty text); col 1 = centred BLACK Target
            # number; then 血管 / 位置 / 心筋% / 灌流域mL.
            it = QTreeWidgetItem(["", str(i), label, f"{pos:.0f}%",
                                  f"{tg['pct']:.1f}%", f"{tg['ml']:.1f}"])
            it.setData(0, _UID_ROLE, i - 1)     # row → index into _targets
            it.setTextAlignment(1, Qt.AlignmentFlag.AlignHCenter
                                | Qt.AlignmentFlag.AlignVCenter)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            hidden = tg.get("hidden", False)
            it.setCheckState(0, Qt.CheckState.Unchecked if hidden
                             else Qt.CheckState.Checked)
            # Target number black (grey when hidden).
            it.setForeground(1, QColor("#999999" if hidden else "#000000"))
            if hidden:
                for c in range(self._targets_w.columnCount()):
                    it.setForeground(c, QColor("#999999"))
            self._targets_w.addTopLevelItem(it)
        self._targets_w.blockSignals(False)

    def _on_target_item_changed(self, it, col):
        """Column-0 checkbox toggled → set that target's hidden flag and refresh."""
        if col != 0:
            return
        ti = it.data(0, _UID_ROLE)
        if not (isinstance(ti, int) and 0 <= ti < len(self._targets)):
            return
        hidden = (it.checkState(0) != Qt.CheckState.Checked)
        if self._targets[ti].get("hidden", False) == hidden:
            return                               # no real change (avoid churn)
        self._targets[ti]["hidden"] = hidden
        self._refresh_targets_table()
        self._push_overlay()
        self._push_territory()

    def add_target(self, vid: str, idx: int) -> None:
        """Public (shell): set a Target at (vid, idx) — a click on the CT vessel.
        Adds it, computes its distal territory, refreshes the table + overlay."""
        if self._territory is None or vid not in self._tree.vessels:
            return
        n = self._tree.vessels[vid].n
        idx = max(0, min(int(idx), n - 1))
        _mask, ml = self._territory.territory(vid, idx)
        pct = (100.0 * ml / self._myo_ml) if self._myo_ml else 0.0
        self._targets.append({"vid": vid, "idx": idx, "ml": ml, "pct": pct,
                              "hidden": False})
        self._refresh_targets_table()
        # The red dot is instant, but the territory colour-fill + VR surfaces take
        # a moment — show a busy window so it doesn't look stuck.
        self._run_busy(t("灌流域を計算中…"),
                       lambda: (self._push_overlay(), self._push_territory()))

    def _run_busy(self, msg, fn):
        """Run *fn* behind a modal busy window (the territory colour-fill / VR
        rebuild is a short but visible wait)."""
        from PyQt6.QtWidgets import QApplication, QProgressDialog
        dlg = QProgressDialog(msg, "", 0, 0, self)
        dlg.setWindowTitle(t("Territory"))
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setValue(0)
        dlg.show()
        QApplication.processEvents()
        try:
            return fn()
        finally:
            dlg.reset()
            dlg.deleteLater()

    def _toggle_targets_shown(self):
        self._targets_shown = not self._targets_shown
        if getattr(self, "_targets_vis_btn", None) is not None:
            self._targets_vis_btn.setText(
                t("非表示") if self._targets_shown else t("表示"))
        self._push_overlay()
        self._push_territory()

    def _delete_last_target(self):
        if self._targets:
            self._targets.pop()
            self._refresh_targets_table()
            self._push_overlay()
            self._push_territory()

    def _territory_full_mask(self):
        """Full-volume 0/1 [z,y,x] mask of the SELECTED target's distal territory
        (or the union of all targets when none is selected), for the CT tint.
        None when there is nothing to draw."""
        eng = self._territory
        if eng is None or not self._targets or not self._targets_shown:
            return None
        fd = self._full_lv_data or {}
        bld = fd.get("bld") or {}
        epi = fd.get("epi") or {}
        sp = bld.get("spacing") or epi.get("spacing")
        vs = ((bld.get("endo") or {}).get("vol_shape")
              or (epi.get("region") or {}).get("vol_shape"))
        if not sp or not vs:
            return None
        sx, sy, sz = (float(s) for s in sp)
        vs = [int(s) for s in vs]
        sel = None
        it = self._targets_w.currentItem()
        if it is not None:
            sel = it.data(0, _UID_ROLE)
        if isinstance(sel, int) and 0 <= sel < len(self._targets) \
                and not self._targets[sel].get("hidden"):
            tgs = [self._targets[sel]]
        else:
            tgs = [tg for tg in self._targets if not tg.get("hidden")]
        full = np.zeros(vs, bool)
        for tg in tgs:
            mask_v, _ml = eng.territory(tg["vid"], tg["idx"])
            c = eng.centers[mask_v]
            if len(c) == 0:
                continue
            fx = np.clip(np.round(c[:, 0] / sx).astype(int), 0, vs[2] - 1)
            fy = np.clip(np.round(c[:, 1] / sy).astype(int), 0, vs[1] - 1)
            fz = np.clip(np.round(c[:, 2] / sz).astype(int), 0, vs[0] - 1)
            full[fz, fy, fx] = True
        return full

    def _territory_systems_full(self):
        """Full-volume int-label [z,y,x] map of the perfusion territories for the
        CT colour overlay: 1=LAD, 2=LCX, 3=RCA (the whole myocardium partitioned by
        nearest-coronary system), and 4=a set Target's distal territory ON TOP of
        the base. Returns (label_mask uint8, TERRITORY_FILLS) or (None, None)."""
        eng = self._territory
        if eng is None or not self._overlay_on:
            return None, None
        fd = self._full_lv_data or {}
        bld = fd.get("bld") or {}
        epi = fd.get("epi") or {}
        sp = bld.get("spacing") or epi.get("spacing")
        vs = ((bld.get("endo") or {}).get("vol_shape")
              or (epi.get("region") or {}).get("vol_shape"))
        if not sp or not vs:
            return None, None
        sx, sy, sz = (float(s) for s in sp)
        vs = [int(s) for s in vs]
        c = eng.centers
        if len(c) == 0:
            return None, None
        fx = np.clip(np.round(c[:, 0] / sx).astype(int), 0, vs[2] - 1)
        fy = np.clip(np.round(c[:, 1] / sy).astype(int), 0, vs[1] - 1)
        fz = np.clip(np.round(c[:, 2] / sz).astype(int), 0, vs[0] - 1)
        code = np.asarray(eng.assignment["code"])
        code_to_vid = eng.assignment["code_to_vid"]
        # per-vessel-code → LAD/LCX/RCA label (1/2/3); 0 = LM stub / unassigned.
        clabel = np.zeros(len(code_to_vid), np.uint8)
        for ci, vid in enumerate(code_to_vid):
            clabel[ci] = _ROLE_TERR_LABEL.get(self._root_role(vid), 0)
        safe = np.clip(code, 0, max(0, len(code_to_vid) - 1))
        labels_v = np.where(code >= 0, clabel[safe], 0).astype(np.uint8)
        full = np.zeros(vs, np.uint8)
        full[fz, fy, fx] = labels_v
        # Target territories → labels 4..9 by target NUMBER (Target1..6), each its
        # own colour, ON TOP of the base (honours per-target hidden + selection).
        if self._targets_shown and self._targets:
            sel = None
            it = self._targets_w.currentItem()
            if it is not None:
                sel = it.data(0, _UID_ROLE)
            if isinstance(sel, int) and 0 <= sel < len(self._targets) \
                    and not self._targets[sel].get("hidden"):
                picks = [sel]
            else:
                picks = [i for i in range(len(self._targets))
                         if not self._targets[i].get("hidden")]
            for i in picks:
                tg = self._targets[i]
                mv, _ml = eng.territory(tg["vid"], tg["idx"])
                if mv.any():
                    full[fz[mv], fy[mv], fx[mv]] = 4 + (i % 6)   # Target(i+1)
        # Colours: LAD/LCX/RCA systems (configurable hue, shipped alpha) then the
        # six Target fills (labels 4..9), so index = label-1.
        line = (self._coro_params or {}).get("line_colors") or {}

        def _fill(role, alpha):
            c = QColor(line.get(role) or ROOT_COLORS.get(role) or "#888888")
            return (c.redF(), c.greenF(), c.blueF(), alpha)

        colors = [_fill("LAD", TERRITORY_FILLS[0][3]),
                  _fill("LCX", TERRITORY_FILLS[1][3]),
                  _fill("RCA", TERRITORY_FILLS[2][3])]
        colors += [self._target_rgba(n) for n in range(1, 7)]   # T1..T6
        return full, colors

    def _territory_summary_text(self) -> str:
        """Compact per-system summary (myocardium + LM/LAD/LCX/RCA %/mL), one line
        each, for the VR overlay. Empty when nothing is computed."""
        eng = self._territory
        if eng is None or self._myo_ml is None:
            return ""
        total = eng.myocardium_ml
        by_role: dict = {}
        for vid in self._tree.roots():
            v = self._tree.vessels[vid]
            _m, ml = eng.territory(vid, 0)
            by_role[v.role] = by_role.get(v.role, 0.0) + ml
        lines = [f"Myocardium {self._myo_ml:.1f}mL"]
        for role in ("LM", "LAD", "LCX", "RCA"):
            if role in by_role:
                ml = by_role[role]
                pct = (100.0 * ml / total) if total > 0 else 0.0
                lines.append(f"{role} {pct:.1f}%/{ml:.1f}mL")
        return "\n".join(lines)

    def _target_summary_text(self) -> str:
        """Per-target list for the VR bottom-left overlay: 'Target-N pct%/mL' for
        each VISIBLE target, same %/mL style as the per-system summary. Empty when
        no targets are shown."""
        if not self._targets_shown or not self._targets:
            return ""
        lines = []
        for i, tg in enumerate(self._targets, 1):
            if tg.get("hidden"):
                continue
            lines.append(f"Target-{i} {tg.get('pct', 0.0):.1f}%/"
                         f"{tg.get('ml', 0.0):.1f}mL")
        return "\n".join(lines)

    def _push_territory(self):
        """Compute the perfusion-territory colour map (LAD/LCX/RCA + Target) and
        ask the shell to overlay it on the CT (cleared when the overlay is off);
        also push the per-system summary text for the VR overlay."""
        if self._shell is None \
                or not hasattr(self._shell, "coronary_territory_refresh"):
            return
        if self._overlay_on:
            mask, colors = self._territory_systems_full()
        else:
            mask, colors = None, None
        try:
            self._shell.coronary_territory_refresh(mask, colors)
        except Exception:                                # noqa: BLE001
            pass
        if hasattr(self._shell, "coronary_territory_summary"):
            txt = self._territory_summary_text() if self._overlay_on else ""
            try:
                self._shell.coronary_territory_summary(txt)
            except Exception:                            # noqa: BLE001
                pass
        if hasattr(self._shell, "coronary_target_summary"):
            ttxt = self._target_summary_text() if self._overlay_on else ""
            try:
                self._shell.coronary_target_summary(ttxt)
            except Exception:                            # noqa: BLE001
                pass

    def _toggle_target_mode(self):
        self._target_mode = self._target_btn.isChecked()
        if self._target_mode and self._territory is None:
            self._warn(t("先に FullLv を読み込んでください（Territory 未計算）。"))
            self._target_btn.setChecked(False)
            self._target_mode = False
            return
        if self._shell is not None \
                and hasattr(self._shell, "coronary_target_mode"):
            self._shell.coronary_target_mode(self._target_mode)

    def _on_target_selection(self):
        self._push_overlay()          # re-highlight the selected Target marker
        self._push_territory()        # re-tint the selected Target's territory

    def _clear_targets(self):
        self._targets = []
        self._refresh_targets_table()
        self._push_overlay()
        self._push_territory()

    def toggle_target_n(self, n: int) -> None:
        """Show/hide the n-th (1-based) target's territory — used by the VR pane's
        right-click on a target marker."""
        i = int(n) - 1
        if 0 <= i < len(self._targets):
            self._targets[i]["hidden"] = not self._targets[i].get("hidden", False)
            self._refresh_targets_table()
            self._push_overlay()
            self._push_territory()

    def delete_target_n(self, n: int) -> None:
        """Delete the n-th (1-based) target — used by the VR pane's right-click on
        a target marker."""
        i = int(n) - 1
        if 0 <= i < len(self._targets):
            del self._targets[i]
            self._refresh_targets_table()
            self._push_overlay()
            self._push_territory()

    def move_target_n(self, n: int, vid: str, idx: int) -> None:
        """Move the n-th (1-based) target to (vid, idx) — a VR marker drag. Recompute
        its territory and refresh the table + overlays."""
        i = int(n) - 1
        if not (0 <= i < len(self._targets)):
            return
        if self._territory is None or vid not in self._tree.vessels:
            return
        nn = self._tree.vessels[vid].n
        idx = max(0, min(int(idx), nn - 1))
        _mask, ml = self._territory.territory(vid, idx)
        pct = (100.0 * ml / self._myo_ml) if self._myo_ml else 0.0
        self._targets[i].update({"vid": vid, "idx": idx, "ml": ml, "pct": pct})
        self._refresh_targets_table()
        self._run_busy(t("灌流域を再計算中…"),
                       lambda: (self._push_overlay(), self._push_territory()))

    def _targets_menu(self, pos):
        it = self._targets_w.itemAt(pos)
        if it is None:
            return
        ti = it.data(0, _UID_ROLE)
        if not (isinstance(ti, int) and 0 <= ti < len(self._targets)):
            return
        hidden = self._targets[ti].get("hidden", False)
        menu = QMenu(self)
        a_vis = menu.addAction(t("表示") if hidden else t("非表示"))
        a_del = menu.addAction(t("削除"))
        col_menu = menu.addMenu(t("領域の色変更"))
        col_acts = {}
        for name, hexv in TARGET_COLOR_CHOICES:
            col_acts[col_menu.addAction(t(name))] = hexv
        a_custom = col_menu.addAction(t("その他…"))
        ch = menu.exec(self._targets_w.viewport().mapToGlobal(pos))
        if ch is a_vis:
            self._targets[ti]["hidden"] = not hidden
            self._refresh_targets_table()
            self._push_overlay()
            self._push_territory()
        elif ch is a_del:
            del self._targets[ti]
            self._refresh_targets_table()
            self._push_overlay()
            self._push_territory()
        elif ch in col_acts:
            self.set_target_color(col_acts[ch], ti + 1)
        elif ch is a_custom:
            self._pick_target_color(ti + 1)

    def _pick_target_color(self, n: int = 1) -> None:
        """Open a full colour picker for Target *n*'s territory/marker colour."""
        from PyQt6.QtWidgets import QColorDialog
        cur = QColor(self._target_hex(n))
        c = QColorDialog.getColor(cur, self, t("ターゲット{n}の色", n=n))
        if c.isValid():
            self.set_target_color(c.name(), n)

    def set_target_color(self, hex_or_color, n: int = 1) -> None:
        """Set Target *n*'s colour (its 6-slot palette entry, persisted to
        Settings so it stays), then refresh. Accepts a '#rrggbb' string. *n* comes
        from the panel's target table or the VR right-click (both 1-based)."""
        c = QColor(hex_or_color)
        if not c.isValid():
            return
        slot = (int(n) - 1) % 6
        tc = list((self._coro_params or {}).get("target_colors")
                  or settings_mod.CORONARY_PARAMS_DEFAULT["target_colors"])
        while len(tc) < 6:
            tc.append(settings_mod.CORONARY_PARAMS_DEFAULT["target_colors"][len(tc)])
        tc[slot] = c.name()
        self._coro_params["target_colors"] = tc
        settings_mod.save_coronary_params(self._coro_params)   # persist the slot
        self._run_busy(t("灌流域を再描画中…"),
                       lambda: (self._refresh_targets_table(),
                                self._push_overlay(), self._push_territory()))

    def target_specs(self, uid: str | None = None) -> list:
        """Target markers/territories for the CT overlay: each = the target 3-D
        point (on its vessel) + its distal territory volume, for the viewer to
        draw a marker (and, later, a colour fill). Empty when the overlay is off
        or nothing is set. Territory/Target analysis is single-CT (the active
        one), so a pane showing a DIFFERENT source CT gets nothing (*uid*)."""
        if uid is not None and uid != self._active_uid:
            return []
        if not self._overlay_on or not self._targets or not self._targets_shown:
            return []
        sel = None
        it = self._targets_w.currentItem()
        if it is not None:
            sel = it.data(0, _UID_ROLE)
        out = []
        for i, tg in enumerate(self._targets):
            if tg.get("hidden"):
                continue
            v = self._tree.vessels.get(tg["vid"])
            if v is None:
                continue
            pts = np.asarray(v.points, float)
            idx = max(0, min(int(tg["idx"]), len(pts) - 1))
            out.append({
                "n": i + 1,
                "vid": tg["vid"],
                "idx": idx,
                "point": pts[idx].tolist(),
                "ml": tg["ml"],
                "pct": tg["pct"],
                "selected": (i == sel),
                "color": self._target_hex(i + 1),   # per-target colour (1..6)
            })
        return out

    def _menu(self, pos):
        it = self._tree_w.itemAt(pos)
        self._activate_item_ct(it)      # 案A: operate on the clicked item's CT
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
            self._warn(t("ルート(LM/LAD/LCX/RCA)の親は変更できません。"
                         "役割を「枝」に変えてから接続してください。"))
            return
        # Candidate parents = ROOT vessels only (role LM/LAD/LCX/RCA), never a
        # descendant of vid. Per user request the picker lists trunks only, so a
        # loose branch is attached to a root; further branch-to-branch nesting is
        # done via 接続 (nearest endpoint).
        banned = {vid, *self._tree.descendants(vid)}
        cands = [(k, self._tree.vessels[k].name, self._tree.vessels[k].role)
                 for k, vv in self._tree.vessels.items()
                 if k not in banned and vv.role in ROOT_ROLES]
        if not cands:
            self._warn(t("親にできるルート(LM/LAD/LCX/RCA)がありません。"
                         "先にルートを設定してください。"))
            return
        # Show the ROLE (LM / LAD / LCX / RCA) first, then the vessel name only
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

    def _delete_selected(self):
        """消去 button — remove the vessels SELECTED in the list (each + its
        downstream branches) from the tree. Multi-selection via Ctrl / Shift."""
        # 案A: a selection can span CTs — group the chosen vids by their source
        # CT so each is removed from its OWN tree.
        by_ct: dict[str, list] = {}
        for it in self._tree_w.selectedItems():
            vid = it.data(0, _UID_ROLE)
            uid = it.data(0, _CTUID_ROLE)
            if not vid or not uid or uid not in self._by_uid:
                continue
            if vid in self._by_uid[uid]["tree"].vessels \
                    and vid not in by_ct.get(uid, ()):
                by_ct.setdefault(uid, []).append(vid)
        if not by_ct:
            self._warn(t("血管リストで消去する血管を選択してください。"))
            return
        victims: dict[str, set] = {}
        total_sel = total_vic = 0
        for uid, vids in by_ct.items():
            tree = self._by_uid[uid]["tree"]
            vic = set()
            for vid in vids:
                vic.add(vid)
                vic.update(tree.descendants(vid))
            victims[uid] = vic
            total_sel += len(vids)
            total_vic += len(vic)
        if QMessageBox.question(
                self, t("消去"),
                t("選択した {n} 本（下流の枝を含め計 {k} 本）を消去しますか? "
                  "(元のCPRファイルは残ります)", n=total_sel, k=total_vic)) \
                != QMessageBox.StandardButton.Yes:
            return
        for uid, vic in victims.items():
            b = self._by_uid[uid]
            for v in vic:
                b["tree"].vessels.pop(v, None)
                b["hidden"].discard(v)
        self._populate()
        self.treeChanged.emit()

    def _clear_all(self):
        # 案A: clears EVERY loaded CT's vessels.
        if not any(b["tree"].vessels for b in self._by_uid.values()):
            return
        if QMessageBox.question(
                self, t("全消去"),
                t("全ての血管を消去しますか?")) == QMessageBox.StandardButton.Yes:
            for b in self._by_uid.values():
                b["tree"] = CoronaryTree()
                b["hidden"] = set()
            self._populate()
            self.treeChanged.emit()

    # ------------------------------------------------------------- files
    def _warn(self, msg):
        QMessageBox.information(self, t("Coronary Tree"), msg)

    def _tree_save_dir(self) -> str:
        """Default folder for ツリー保存 / 読込 dialogs: the folder of the LAST tree
        saved/loaded this session for THIS CT (直前の保存先), else the SOURCE 3-D CT's
        own folder, else a registered vessel's .cpr.json folder; '' if unknown."""
        b = self._bundle()
        lp = b.get("last_path")
        if lp:
            d = os.path.dirname(lp)
            if d and os.path.isdir(d):
                return d
        cd = b.get("ct_dir") or ""
        if cd and os.path.isdir(cd):
            return cd
        for p in (b.get("paths") or {}).values():
            d = os.path.dirname(p) if p else ""
            if d and os.path.isdir(d):
                return d
        return ""

    def _save_as(self):
        if not self._tree.vessels:
            self._warn(t("保存する血管がありません。"))
            return
        # Default to the source 3-D CT's folder (then the last-saved folder once a
        # save/load has happened this session for this dataset).
        d = self._tree_save_dir()
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

    def _dedupe_tree(self) -> int:
        """Drop UNCONNECTED branch vessels whose name duplicates a CONNECTED vessel
        (a root or a parented branch) — the "（未接続 N）" ghosts left when the raw
        .cpr.json files were loaded alongside a corotree. Returns the count removed."""
        connected_names = {v.name for v in self._tree.vessels.values()
                           if v.role in ROOT_ROLES or v.parent is not None}
        drop = [vid for vid, v in self._tree.vessels.items()
                if v.role not in ROOT_ROLES and v.parent is None
                and v.name in connected_names]
        for vid in drop:
            self._tree.vessels.pop(vid, None)
            self._hidden.discard(vid)
        return len(drop)

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
        # 案A: select (or create) THIS tree's source-CT bundle FIRST, so the
        # loaded vessels land in the right CT and don't overwrite another's.
        ser = data.get("series") or {}
        uid = ser.get("series_uid", "") or ""
        self._use_uid(uid, ct_dir=ser.get("src_dir", "") or "", series=ser)
        self._tree = CoronaryTree.from_json(data)
        self._dedupe_tree()          # drop unconnected ghosts of connected vessels
        self._hidden.clear()
        self._last_path = path
        self._populate()
        self.treeChanged.emit()
        self._hint.setText(t("読込みました: {p}", p=path))
        return True

    def _load(self):
        d = self._tree_save_dir()            # source CT folder, else last-used
        path, _ = QFileDialog.getOpenFileName(
            self, t("冠動脈ツリーを読込"), d, t("CoroTree (*.corotree.json)"))
        if path and self.load_file(path):
            # ツリー読込 → turn the overlay ON and bring up the source CT.
            self._set_overlay(True)
