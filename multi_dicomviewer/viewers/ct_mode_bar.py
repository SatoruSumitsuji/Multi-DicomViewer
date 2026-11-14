"""CT work-mode switch shared by both CT viewers (Windows VTK ``ct_viewer.py``
and Mac pygfx ``ct_viewer_pygfx.py``), so the two behave identically.

Four mode buttons sit right of 3D/2D on the Plane bar (row 1):

    標準モード          plain CT; every result kept, only hidden  — always
    LVモード            Epi / Endo / Blood + LV values; the coronary tree drawn
                        over them (view only, ツリー表示 on/off)  — LV measured
                                                               or FullLv loaded
    CoronaryTreeモード  the coronary tree only (LV hidden), CoronaryTree tab in
                        front, ツリー表示 ON                    — a tree exists
    Territoryモード     perfusion territory + VR review          — a Territory
                                                               result exists

Every switch is IN PLACE (the CT is never re-opened): 標準 / LV / CoronaryTree
hide / restore overlays (data kept); the Territory review lays the panes on the
LV axis + VR and leaving it restores the previous view. 「Territory計算」 (shown
in CoronaryTreeモード only) computes the territory from the in-session LV (else
the docked FullLv) + the tree via the shell (``shell.ct_territory_compute``) and
enters the review. Availability comes from the viewer's own in-session LV data
plus what the Coronary Tree dock holds (``shell.ct_mode_state``). Auto: an LV
edit step → LVモード; the CPR button / ツリー登録 / editing a tree line →
CoronaryTreeモード. A row-3 readout names the FullLv / corotree (or 計測済み /
未保存).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from multi_dicomviewer.i18n import t
from multi_dicomviewer.ui.study_browser import FitButton

#: (key, label, tooltip) in button order.
_CT_MODES = (
    ("standard", "標準モード",
     "標準モード：LV・冠動脈ツリーの表示を消した通常の 3DCT 表示（計測結果は"
     "すべて保持され、モードを切り替えるとすぐ表示されます）"),
    ("lv", "LVモード",
     "LVモード：Epi / Endo / Blood と LV 計測値を表示（冠動脈ツリーがあれば"
     "重ねて表示、ツリー表示で ON/OFF）。LV を計測済み、または .FullLv.json "
     "読込時に有効"),
    ("coronary", "CoronaryTreeモード",
     "CoronaryTreeモード：LV の表示を消して冠動脈ツリー（CPR 線）だけを表示・"
     "編集。左のタブを CoronaryTree に切り替えます。CPR ボタンかツリー登録で"
     "自動的に入ります"),
    ("territory", "Territoryモード",
     "Territoryモード：灌流領域 + VR の表示。Territory を計算済みのとき有効"
     "（計算は CoronaryTreeモードの「Territory計算」）"),
)

_TERR_BTN_TIP = (
    "Territory計算：LV の計測結果（保存していなくても可。無ければ読み込んだ "
    ".FullLv.json）と冠動脈ツリーから灌流領域を計算し、Territoryモードに入ります")
_TERR_BTN_CSS = (
    "QPushButton { background:#fff4d6; color:black; border:1px solid #c89b2c; }"
    "QPushButton:disabled { background:#f0f0f0; color:#aaa;"
    " border:1px solid #ddd; }"
)

# Checked = the current mode (teal, so it reads apart from the yellow 3D/2D).
_MODE_CSS = (
    "QPushButton { background:white; color:black; }"
    "QPushButton:checked { background:#2a7f8f; color:white; }"
    "QPushButton:disabled { background:#f0f0f0; color:#aaa; }"
)


class CTModeBarMixin:
    """Mode buttons + loaded-file readout. Host provides ``_territory_display``,
    ``_bottom_ctx`` and ``window()`` / ``parent()`` (a QWidget)."""

    # ------------------------------------------------------------- build
    def _build_ct_mode_buttons(self, row: QHBoxLayout) -> None:
        """Append the four mode buttons to the Plane bar *row* (after 3D/2D)."""
        row.addSpacing(16)
        self._ct_mode_btns: dict = {}
        for key, label, tip in _CT_MODES:
            b = FitButton(t(label))
            b.setCheckable(True)
            b.setChecked(key == "standard")
            b.setEnabled(key == "standard")
            b.setHelpToolTip(t(tip))
            b.setStyleSheet(_MODE_CSS)
            b.clicked.connect(lambda _c, k=key: self._ct_mode_clicked(k))
            self._ct_mode_btns[key] = b
            row.addWidget(b)
        # 「Territory計算」 is not a mode: it sits at the right end of the
        # Coronary / VR / LV row (the viewers place it there).
        self._ct_terr_button()

    def _ct_terr_button(self):
        """The 「Territory計算」 (Calc Territory) button, made once: CoronaryTree
        モード only (the tree is the prerequisite; an LVモード button would invite
        mistakes). Placed by the viewer at the right end of its Coronary row."""
        tb = getattr(self, "_ct_terr_btn", None)
        if tb is None:
            tb = FitButton(t("Territory計算"))
            tb.setHelpToolTip(t(_TERR_BTN_TIP))
            tb.setStyleSheet(_TERR_BTN_CSS)
            tb.clicked.connect(
                lambda _c=False: self._ct_territory_compute_clicked())
            tb.setVisible(False)
            self._ct_terr_btn = tb
        return tb

    def _build_ct_files_line(self) -> QWidget:
        """The row-3 readout of the loaded .FullLv.json / .corotree.json."""
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(8, 0, 8, 0)
        self._ct_files_lbl = QLabel("")
        self._ct_files_lbl.setStyleSheet("color:#1a5fb4; font-weight:bold;")
        # Take the row's full width (the text arrives after the bar is laid out,
        # so a Preferred label stayed at its empty-text width and clipped).
        self._ct_files_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Preferred)
        self._ct_files_lbl.setMinimumWidth(0)
        row.addWidget(self._ct_files_lbl, 1)
        self._ct_files_wrap = w
        return w

    # ------------------------------------------------------------- state
    def _ct_mode_reset(self) -> None:
        """A fresh CT / series: no explicit mode, nothing hidden."""
        self._ct_mode_sel = None          # explicit choice (None = derived)
        self._ct_lv_hidden = False        # LV overlays hidden (data kept)
        self._ct_lv_snap = None           # what to restore on LVモード
        self._ct_tree_suppressed = False  # tree lines hidden here (標準モード)
        self._ct_tree_on_lv = True        # ツリー表示 remembered for LVモード
        self._ct_terr_snap = None         # the view before the Territory review
        self._ct_terr_lvd = None          # LVD shown before the review
        self._full_lv_saved_path = ""     # this CT's last saved .FullLv.json

    def _ct_lv_data(self) -> bool:
        """Any LV analysis on this CT — measured in this session or loaded."""
        if (getattr(self, "_lv", None) is not None
                or getattr(self, "_lvv", None) is not None
                or getattr(self, "_lvv_epi_surf", None) is not None
                or getattr(self, "_lv_apex", None) is not None):
            return True
        vs = getattr(self, "_lv_valves", None) or {}
        return any(v is not None for v in vs.values())

    def _ct_lv_ready(self) -> bool:
        """The in-session LV is complete enough to build a FullLv (the FullLV
        button's own readiness rule)."""
        f = getattr(self, "_full_lv_missing", None)
        if f is None:
            return False
        try:
            return not f()
        except Exception:                                # noqa: BLE001
            return False

    def _ct_current_mode(self) -> str:
        if getattr(self, "_territory_display", False):
            return "territory"
        sel = getattr(self, "_ct_mode_sel", None)
        if sel in ("standard", "lv", "coronary"):
            return sel
        ctx = getattr(self, "_bottom_ctx", "both")
        if ctx == "coronary" or getattr(self, "_coronary_mode", False):
            return "coronary"
        if ctx == "lv" or self._ct_lv_data():
            return "lv"
        return "standard"

    def _ct_mode_shell(self):
        """The MainWindow shell (walk up the parents: self.window() can be an
        intermediate widget when the pane nests this viewer)."""
        w = self
        for _ in range(16):
            if w is None:
                break
            if hasattr(w, "ct_mode_state"):
                return w
            try:
                w = w.parent()
            except Exception:                            # noqa: BLE001
                break
        win = self.window()
        return win if (win is not None and hasattr(win, "ct_mode_state")) else None

    def refresh_ct_mode_bar(self) -> None:
        """Re-read which files are loaded for this CT and update the buttons
        (enabled / current) and the file readout. Cheap; safe to call often."""
        btns = getattr(self, "_ct_mode_btns", None)
        if not btns:
            return
        shell = self._ct_mode_shell()
        state = {}
        if shell is not None:
            try:
                state = shell.ct_mode_state(self) or {}
            except Exception:                            # noqa: BLE001
                state = {}
        full = state.get("full", "") or ""
        tree = state.get("tree", "") or ""
        lv_mem = self._ct_lv_data() or bool(state.get("full_mem"))
        avail = {"standard": True,
                 "lv": bool(full) or lv_mem,
                 "coronary": (bool(tree) or bool(state.get("tree_mem"))
                              or getattr(self, "_coronary_mode", False)),
                 "territory": bool(state.get("terr"))}
        cur = self._ct_current_mode()
        for key, b in btns.items():
            # The CURRENT mode stays enabled even if its file just went away, so
            # the highlight never turns into an unreadable greyed-checked button.
            b.setEnabled(avail[key] or key == cur)
            if b.isChecked() != (key == cur):
                b.setChecked(key == cur)
        tb = getattr(self, "_ct_terr_btn", None)
        if tb is not None:
            tb.setVisible(cur == "coronary")
            tb.setEnabled(avail["coronary"] and bool(
                state.get("tree") or state.get("tree_mem"))
                and (self._ct_lv_ready() or bool(full)
                     or bool(state.get("full_mem"))))
        lbl = getattr(self, "_ct_files_lbl", None)
        if lbl is not None:
            none = t("なし")
            if not full and lv_mem:
                import os
                sp = getattr(self, "_full_lv_saved_path", "") or ""
                if getattr(self, "_full_lv_saved", False) and sp:
                    full = os.path.basename(sp)
                else:
                    full = (t("計測済み（保存済）")
                            if getattr(self, "_full_lv_saved", False)
                            else t("計測済み（未保存）"))
            if not tree and state.get("tree_mem"):
                tree = t("ツリーあり（未保存）")
            lbl.setText(t("読込ファイル") + ":  FullLv: " + (full or none)
                        + "  ／  corotree: " + (tree or none))

    def _ct_mode_clicked(self, key: str) -> None:
        cur = self._ct_current_mode()
        if key == cur:                         # already there — keep it checked
            self.refresh_ct_mode_bar()
            return
        # Every switch is IN PLACE (the CT is never re-opened).
        shell = self._ct_mode_shell()
        if cur == "territory":
            self._ct_territory_leave()
            self._ct_switch_inplace(key, auto=True)
            return
        if key == "territory":
            if (hasattr(self, "_lv_mode_active") and self._lv_mode_active()):
                self._ct_lv_editing_msg()
                self.refresh_ct_mode_bar()
                return
            if shell is not None and hasattr(shell, "ct_territory_show"):
                try:
                    shell.ct_territory_show(self)
                except Exception:                        # noqa: BLE001
                    import traceback
                    traceback.print_exc()
            self.refresh_ct_mode_bar()
            return
        self._ct_switch_inplace(key)

    # ------------------------------------------------- in-place switching
    # 標準モード: plain CT (LV + tree kept, hidden). LVモード: the LV overlays and
    # results; the coronary tree drawn over them (view only; ツリー表示 on/off is
    # remembered for LVモード) with the CoronaryTree tab in front once a tree
    # exists. CoronaryTreeモード: the tree only (LV hidden), the tab in front,
    # ツリー表示 ON. Auto: an LV edit step → LVモード; CPR / ツリー登録 →
    # CoronaryTreeモード.
    def _ct_mode_auto(self, key: str) -> None:
        """Enter *key* because the user started its work (no-op if already
        there or in the Territory review)."""
        cur = self._ct_current_mode()
        if cur == key or cur == "territory":
            return
        self._ct_switch_inplace(key, auto=True)

    def _ct_switch_inplace(self, key: str, auto: bool = False) -> None:
        if getattr(self, "_ct_switching", False):
            return
        cur = self._ct_current_mode()
        if (not auto and key != "lv" and hasattr(self, "_lv_mode_active")
                and self._lv_mode_active()):
            self._ct_lv_editing_msg()
            self.refresh_ct_mode_bar()
            return
        shell = self._ct_mode_shell()
        self._ct_switching = True
        try:
            if cur == "lv" and shell is not None and hasattr(
                    shell, "ct_tree_overlay_on"):
                on = shell.ct_tree_overlay_on(self)
                if on is not None:
                    self._ct_tree_on_lv = bool(on)
            # Leaving CoronaryTreeモード ends an open CPR session first.
            if (key != "coronary" and getattr(self, "_coronary_mode", False)
                    and hasattr(self, "_coronary_exit")):
                try:
                    self._coronary_exit()
                except Exception:                        # noqa: BLE001
                    pass
            self._ct_mode_sel = key
            self._ct_show_lv(key == "lv")
            self._ct_tree_suppressed = (key == "standard")
            self._ct_tree_apply()
            if shell is not None and hasattr(shell, "ct_tree_show"):
                if key == "coronary":
                    shell.ct_tree_show(self, True, raise_tab=True)
                elif key == "lv":
                    shell.ct_tree_show(self, self._ct_tree_on_lv,
                                       raise_tab=True)
        finally:
            self._ct_switching = False
        for fn in ("_apply_file_context", "_lv_update_submode_ui"):
            try:
                getattr(self, fn)()
            except Exception:                            # noqa: BLE001
                pass
        self.refresh_ct_mode_bar()
        self._ct_render()

    def _ct_tree_apply(self) -> None:
        """Re-apply the tree lines from the last requested spec (hidden while
        _ct_tree_suppressed)."""
        if hasattr(self, "set_coronary_overlay"):
            try:
                self.set_coronary_overlay(getattr(self, "_coro_overlay_req",
                                                  None))
            except Exception:                            # noqa: BLE001
                pass

    def _ct_render(self) -> None:
        for k in ("A", "B"):
            try:
                if hasattr(self, "_redraw_meas"):
                    self._redraw_meas(k)
                ov = getattr(self, "_overlay", None)
                if isinstance(ov, dict) and ov.get(k) is not None:
                    ov[k].update()
                pane = getattr(self, "pane", None)
                if isinstance(pane, dict) and pane.get(k) is not None \
                        and hasattr(pane[k], "render"):
                    pane[k].render()
            except Exception:                            # noqa: BLE001
                pass

    def _ct_show_lv(self, on: bool) -> None:
        """Hide every LV overlay remembering what was shown (data kept), or
        restore exactly that."""
        def g(a, d=False):
            return bool(getattr(self, a, d))
        if not on and not getattr(self, "_ct_lv_hidden", False):
            lvv = getattr(self, "_lvv", None)
            snap = {
                "border": g("_lv_border_show"),
                "blood_border": g("_lvv_epi_show"),
                "area": lvv is None and g("_lvv_mask_on"),
                "endo": g("_lvv_endo_show"),
                "lvd": g("_lvv_lvd_shown"),
                "blood": (None if lvv is None else
                          ("lv" if g("_lvv_mask_on") else
                           ("all" if g("_lvv_hl_on") else None))),
                "valves": dict(getattr(self, "_lv_valve_shown", {}) or {}),
                "apex": g("_lv_apex_shown", True),
                "lvv_apex": g("_lvv_apex_shown", True),
            }
            self._ct_lv_snap = snap
            self._ct_lv_hidden = True
            self._ct_lv_vis(snap, False)
        elif on and getattr(self, "_ct_lv_hidden", False):
            self._ct_lv_hidden = False
            snap = getattr(self, "_ct_lv_snap", None) or {}
            self._ct_lv_snap = None
            self._ct_lv_vis(snap, True)

    def _ct_lv_vis(self, snap: dict, on: bool) -> None:
        def _do(fn, *a):
            f = getattr(self, fn, None)
            if f is not None:
                try:
                    f(*a)
                except Exception:                        # noqa: BLE001
                    pass
        if snap.get("border"):
            _do("set_lv_border_visible", on)
        if snap.get("blood_border"):
            _do("_lv_set_epi_blood_border", on)
        if snap.get("area"):
            _do("_lv_set_epi_area", on)
        if snap.get("endo"):
            _do("_lv_set_auto_endo", on)
        if snap.get("lvd"):
            _do("_lv_quick_set_lvd", on)
        if snap.get("blood"):
            _do("_lv_set_blood", snap["blood"] if on else None)
        shown = getattr(self, "_lv_valve_shown", None)
        valves = getattr(self, "_lv_valves", None) or {}
        if isinstance(shown, dict):
            for w, was in (snap.get("valves") or {}).items():
                if not was or valves.get(w) is None:
                    continue
                shown[w] = bool(on)
                # flip the ring (and its Auto trace) in place: no re-draw
                for k in ("A", "B"):
                    for mm in (getattr(self, "_measures", {}) or {}).get(k, []):
                        if mm.get("_lv_valve") == w:
                            mm["hidden"] = not on
        if snap.get("apex", True):
            self._lv_apex_shown = bool(on)
            if getattr(self, "_lv_apex", None) is not None:
                _do("_lv_apex_marker_draw")
        if snap.get("lvv_apex", True):
            self._lvv_apex_shown = bool(on)
            for k in ("A", "B"):
                for mm in (getattr(self, "_measures", {}) or {}).get(k, []):
                    if mm.get("_lvv") == "apex":
                        mm["hidden"] = not on
        _do("_lv_style_selectors")

    def _ct_lv_editing_msg(self) -> None:
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self.window(), t("モード"),
                t("LV の編集中です。編集を終了してから切り替えてください。"))
        except Exception:                                # noqa: BLE001
            pass

    # ------------------------------------------------- Territory (in place)
    def _ct_territory_compute_clicked(self) -> None:
        """「Territory計算」 (CoronaryTreeモード): the shell computes the territory
        from this CT's LV + tree and calls back _ct_territory_enter."""
        shell = self._ct_mode_shell()
        if shell is None or not hasattr(shell, "ct_territory_compute"):
            return
        try:
            shell.ct_territory_compute(self)
        except Exception:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
        self.refresh_ct_mode_bar()

    def _ct_full_lv_build(self):
        """A FullLv dict built IN MEMORY from the in-session LV (no file), or
        None when the LV isn't complete / the build fails."""
        if not self._ct_lv_ready():
            return None
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication
        from multi_dicomviewer.core import full_lv
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            bld = self._full_lv_bld_for_save()
            epi = self._full_lv_epi_dict()
            if bld is None or epi is None:
                return None
            full, _err = full_lv.build(epi, bld)
            return full
        except Exception:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            return None
        finally:
            QApplication.restoreOverrideCursor()

    _CT_VIEW_KEYS = ("_frame", "_pc", "_cross_ang", "_roll", "_thick")

    def _ct_view_snap(self) -> dict:
        """The MPR view (frames / centres / slab / mode) to restore after the
        Territory review re-laid the panes on the LV axis."""
        import copy
        snap = {"mode": getattr(self, "_mode", "3D")}
        for a in self._CT_VIEW_KEYS:
            if hasattr(self, a):
                snap[a] = copy.deepcopy(getattr(self, a))
        c = getattr(self, "_center", None)
        snap["_center"] = None if c is None else np.asarray(c, float).copy()
        return snap

    def _ct_view_restore(self, snap) -> None:
        if snap:
            for a in self._CT_VIEW_KEYS:
                if a in snap:
                    setattr(self, a, snap[a])
            if snap.get("_center") is not None:
                self._center = snap["_center"]
        elif hasattr(self, "_init_frames"):
            self._init_frames()                  # no snapshot → default MPR
        self._view_initial = True
        if hasattr(self, "_refresh"):
            self._refresh(reset_cam=True)        # fit both panes on the restored view

    def _ct_territory_enter(self, full) -> bool:
        """Enter the Territory review IN PLACE. With a complete in-session LV the
        LV state is kept as is (only the layout changes: LV-axis panes + VR);
        otherwise the FullLv is applied first (apply_full_lv, display-only)."""
        if getattr(self, "_territory_display", False):
            return True
        if (getattr(self, "_coronary_mode", False)
                and hasattr(self, "_coronary_exit")):
            try:
                self._coronary_exit()            # end an open CPR session first
            except Exception:                    # noqa: BLE001
                pass
        # The review shows the LV (+ tree): undo the in-place hiding first.
        self._ct_show_lv(True)
        self._ct_tree_suppressed = False
        self._ct_tree_apply()
        self._ct_terr_snap = self._ct_view_snap()
        self._ct_terr_lvd = bool(getattr(self, "_lvv_lvd_shown", False))
        light = (self._ct_lv_ready()
                 and getattr(self, "_lvv", None) is not None
                 and getattr(self, "_lvv_epi_surf", None) is not None)
        if not light:
            ok = bool(hasattr(self, "apply_full_lv") and self.apply_full_lv(full))
            if not ok:
                self._ct_terr_snap = None
            self.refresh_ct_mode_bar()
            return ok
        from PyQt6.QtCore import QTimer
        if getattr(self, "_mode", "3D") != "3D" and hasattr(self, "_set_mode"):
            self._set_mode("3D")
        self._lvv_lvd_shown = False              # LVD is clutter in the review
        try:
            self._lvv_setup_axis_views()         # A = LV short axis, B = long axis
        except Exception:                        # noqa: BLE001
            pass
        lvv = getattr(self, "_lvv", None)
        if lvv is not None and not lvv.get("observe"):
            try:
                self._lvv_observe_exit()
            except Exception:                    # noqa: BLE001
                pass
        self._refresh(reset_cam=False)
        self._territory_display = True
        for fn in ("_cpr_mark_full_lv", "_refresh_tool_availability",
                   "_terr_bar_sync"):
            try:
                getattr(self, fn)()
            except Exception:                    # noqa: BLE001
                pass
        self._ct_render()
        QTimer.singleShot(0, lambda: self._vr_set("B", True))
        return True

    def _ct_territory_leave(self) -> None:
        """Leave the Territory review IN PLACE: VR off on every pane, the view
        from before the review back, the LV / Coronary bars back."""
        vr_on = getattr(self, "_vr_on", {}) or {}
        for k in ("A", "B"):
            try:
                if vr_on.get(k):
                    self._vr_set(k, False)
            except Exception:                    # noqa: BLE001
                pass
        self._territory_display = False
        snap = getattr(self, "_ct_terr_snap", None)
        self._ct_terr_snap = None
        try:
            self._ct_view_restore(snap)
        except Exception:                        # noqa: BLE001
            import traceback
            traceback.print_exc()
        lvd = getattr(self, "_ct_terr_lvd", None)
        if lvd is not None:
            self._lvv_lvd_shown = bool(lvd)
            self._ct_terr_lvd = None
        for fn in ("_refresh_tool_availability", "_terr_bar_sync"):
            try:
                getattr(self, fn)()
            except Exception:                    # noqa: BLE001
                pass
        shell = self._ct_mode_shell()
        if shell is not None and hasattr(shell, "ct_territory_left"):
            try:
                shell.ct_territory_left(self)
            except Exception:                    # noqa: BLE001
                pass
