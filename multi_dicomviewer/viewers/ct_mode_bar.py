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
LV axis + VR and leaving it restores the previous view. 「Territory Calc」 sits
in row 3 between its inputs and its result — 読込ファイル: FullLv ＋ corotree ▶
[Territory Calc] territory [Load] [Save] (user 2026-11-27) — in every mode once
an LV and a tree exist; it computes the territory from the in-session LV (else
the docked FullLv) + the tree via the shell (``shell.ct_territory_compute``) and
enters the review (in the review = recompute). Availability comes from the viewer's own in-session LV data
plus what the Coronary Tree dock holds (``shell.ct_mode_state``). Auto: an LV
edit step → LVモード; the CPR button / ツリー登録 / editing a tree line →
CoronaryTreeモード. A row-3 readout names the FullLv / corotree (or 計測済み /
未保存).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QSizePolicy, QSlider,
                             QWidget)
from PyQt6.QtCore import Qt

from multi_dicomviewer.i18n import t
from multi_dicomviewer.ui.study_browser import FitButton

#: (key, label, tooltip) in button order.
_CT_MODES = (
    ("standard", "Simple View",
     "Simple View：LV・冠動脈ツリーの表示を消した通常の 3DCT 表示（計測結果は"
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
     "（計算は 3行目「読込ファイル」の「Territory Calc」）"),
)

_TERR_BTN_TIP = (
    "Territory計算：LV の計測結果（保存していなくても可。無ければ読み込んだ "
    ".FullLv.json）と冠動脈ツリーから灌流領域を計算し、Territoryモードに入ります")
_TERR_LOAD_TIP = (
    "Territory を読込：.territory.json を開き、リンク先の corotree.json と "
    "FullLv.json を読み込んで Territory を再計算し、すぐ Territoryモードで表示します")
_TERR_SAVE_TIP = (
    "Territory を保存：計算した Territory を XXXX.territory.json に保存します"
    "（corotree.json と FullLv.json へのリンク、ターゲット、結果を記録。"
    "先に両方を保存しておく必要があります）")
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


#: 結果文字サイズ slider range (% of the default result-text size)
RESULT_FONT_PCT_MIN = 50
RESULT_FONT_PCT_MAX = 150


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
        # 「Territory Calc / Load / Save」 are not modes: they sit in row 3
        # (読込ファイル) between the FullLv + corotree and the territory.
        self._ct_terr_button()
        # 結果文字サイズ (user 2026-11-25): ONE size for every result text on
        # BOTH panes — WW/WL, the angle, the pane kind (A | MPR), measure /
        # LV / Territory results, legends, the hover HU. The DICOM-tag size
        # stays on the top-row slider.
        row.addSpacing(16)
        cap = QLabel(t("結果文字サイズ："))
        row.addWidget(cap)
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(RESULT_FONT_PCT_MIN, RESULT_FONT_PCT_MAX)
        sl.setSingleStep(5)
        sl.setPageStep(10)
        sl.setFixedWidth(110)
        sl.setValue(self._result_font_pct_now())
        sl.setToolTip(t(
            "結果の文字サイズ（{a}〜{b}%）：WW/WL・角度・計測 / LV / Territory "
            "の結果などを左右両ペインで一括して変更します。DICOM タグは上の"
            "スライダーで変更します", a=RESULT_FONT_PCT_MIN,
            b=RESULT_FONT_PCT_MAX))
        sl.valueChanged.connect(self._result_font_slider_moved)
        row.addWidget(sl)
        self._result_font_slider = sl
        self._result_font_cap = cap
        if self._result_font_pct_now() != 100:     # the saved size, once built
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self.set_result_font_pct(
                self._result_font_pct_now()))

    # ---- 結果文字サイズ ------------------------------------------------------
    def _result_font_pct_now(self) -> int:
        """The result-text size (% of the default), from the saved setting
        the first time."""
        pct = getattr(self, "_result_font_pct", None)
        if pct is None:
            try:
                from multi_dicomviewer.core import settings as _st
                pct = int(_st.load_ui_state("ct_result_font_pct", 100) or 100)
            except Exception:                            # noqa: BLE001
                pct = 100
            pct = max(RESULT_FONT_PCT_MIN, min(RESULT_FONT_PCT_MAX, pct))
            self._result_font_pct = pct
        return int(pct)

    def _rfs(self, size) -> int:
        """*size* (a default font size, px or pt) at the result-text size."""
        return max(6, int(round(float(size) * self._result_font_pct_now()
                                / 100.0)))

    def _result_font_slider_moved(self, pct) -> None:
        """Slider: every CT viewer in the shell takes the size (both panes,
        all CT panes) and it is saved; without a shell just this viewer."""
        shell = self._ct_mode_shell() if hasattr(self, "_ct_mode_shell") \
            else None
        if shell is not None and hasattr(shell, "ct_result_font_set"):
            shell.ct_result_font_set(int(pct))
        else:
            self.set_result_font_pct(int(pct))

    def set_result_font_pct(self, pct) -> None:
        """Public (shell): apply the result-text size (%) to this viewer."""
        pct = max(RESULT_FONT_PCT_MIN, min(RESULT_FONT_PCT_MAX, int(pct)))
        self._result_font_pct = pct
        sl = getattr(self, "_result_font_slider", None)
        if sl is not None and sl.value() != pct:
            sl.blockSignals(True)
            sl.setValue(pct)
            sl.blockSignals(False)
        try:
            self._apply_result_font()
        except Exception:                                # noqa: BLE001
            import traceback
            traceback.print_exc()

    # ---- Territory: Blood / 壁厚3D / 壁厚短軸 (the review bar) ---------------
    def _terr_toggle_blood(self) -> None:
        """Territory: Blood — its OWN on/off toggle: ON = LV-Blood (水色, the
        FullLv's blood, instant), OFF = no blood. Independent of 壁厚 (a 壁厚
        on/off never changes it); never switches to the All-Blood tint."""
        btn = getattr(self, "_terr_blood_btn", None)
        want = bool(btn.isChecked()) if btn is not None else True
        self._lv_set_blood("lv" if want else None)
        self._terr_btn_states()

    def _terr_toggle_thick(self, mode) -> None:
        """Territory: 壁厚3D / 壁厚短軸 on/off. The LV handler hides the blood fill
        when a heat map comes on (LV mode); here the Blood state is kept."""
        blood = bool(getattr(self, "_lvv_mask_on", False))
        tint = bool(getattr(self, "_lvv_hl_on", False))
        self._lvv_set_thick_mode(mode)
        if blood and not getattr(self, "_lvv_mask_on", False):
            self._lv_set_blood("lv")
        elif tint and not getattr(self, "_lvv_hl_on", False):
            self._lv_set_blood("all")
        self._terr_btn_states()

    def _terr_btn_states(self) -> None:
        """Territory bar buttons: checked + coloured background while ON (Blood
        水色, 壁厚3D yellow, 壁厚短軸 magenta — the LV bar's colours); 壁厚3D ⇔
        壁厚短軸 are exclusive — while one is on the other is greyed out."""
        self._ctlv_sync()
        if not getattr(self, "_territory_display", False):
            return
        style = getattr(self, "_lvv_style_toggle", None)
        b = getattr(self, "_terr_blood_btn", None)
        if b is not None:
            on = (bool(getattr(self, "_lvv_mask_on", False))
                  or bool(getattr(self, "_lvv_hl_on", False)))
            b.blockSignals(True)
            b.setChecked(on)
            b.blockSignals(False)
            if style is not None:
                style(b, "#40e0ff", "black")
        tm = getattr(self, "_lvv_thick_mode", None)
        modes = getattr(self, "_THICK_MODES", {}) or {}
        for m, attr in (("3d", "_terr_thick3d_btn"), ("sax", "_terr_thicksax_btn")):
            btn = getattr(self, attr, None)
            if btn is None:
                continue
            btn.blockSignals(True)
            btn.setChecked(tm == m)
            btn.blockSignals(False)
            btn.setEnabled(tm is None or tm == m)
            if style is not None and m in modes:
                style(btn, modes[m][1], "black")

    def _terr_vr_rotate(self, tries: int = 0) -> None:
        """Territory review start: VR on the right pane, then make it the
        active pane with the ROTATE tool — the first thing done there is
        turning the VR (it used to start on Paging). Retries briefly while
        the VR is still being built (Rotate is greyed on a non-VR pane)."""
        from PyQt6.QtCore import QTimer
        if not getattr(self, "_territory_display", False):
            return
        vr_on = getattr(self, "_vr_on", {}) or {}
        if tries == 0 and not vr_on.get("B"):
            self._vr_set("B", True)
            vr_on = getattr(self, "_vr_on", {}) or {}
        if not vr_on.get("B"):
            if tries < 30:
                QTimer.singleShot(100, lambda: self._terr_vr_rotate(tries + 1))
            return
        try:
            self._set_active("B")
            self._set_tool("ROTATE")
        except Exception:                        # noqa: BLE001
            pass

    # ---- LV: 壁厚3D / 壁厚短軸 / CoronaryTree (LVモード, row 2) --------------
    def _ct_targets_visible(self) -> bool:
        """Territory Target points (dots + numbers, MPR and VR) belong to the
        Territory review only — hidden in 標準 / LV / CoronaryTree (kept)."""
        return bool(getattr(self, "_territory_display", False))

    def _ctlv_tree_available(self) -> bool:
        shell = self._ct_mode_shell()
        if shell is None or not hasattr(shell, "ct_tree_overlay_on"):
            return False
        try:
            return shell.ct_tree_overlay_on(self) is not None
        except Exception:                        # noqa: BLE001
            return False

    def _ctlv_tree_toggle(self) -> None:
        """LV: CoronaryTree — draw the coronary tree over the LV (view only) or
        hide it again. Remembered while the CT stays open (LVモード re-entry)."""
        btn = getattr(self, "_ctlv_tree_btn", None)
        on = bool(btn.isChecked()) if btn is not None else False
        if on and not self._ctlv_tree_available():
            on = False
        self._ct_lv_tree_on = on
        if self._ct_current_mode() == "lv":
            shell = self._ct_mode_shell()
            if on and shell is not None and hasattr(shell, "ct_tree_show"):
                try:
                    shell.ct_tree_show(self, True)   # ツリー表示 ON in the dock
                except Exception:                    # noqa: BLE001
                    pass
            self._ct_tree_suppressed = not on
            self._ct_tree_apply()
            self._ct_render()
        self._ctlv_sync()

    # ---- LV: Epi境界 / Endo境界 / 血流 — plain show/hide toggles (LVモード) ----
    # (Epi / Blood/Endo open their detail rows; these only switch the overlay.)
    _CTLV_SHOW = (("epi", "Epi境界", "#50dc50",
                   "Epi 境界の表示 / 非表示（データは保持）"),
                  ("endo", "Endo境界", "#ff8c28",
                   "Endo 境界（自動 Endo）の表示 / 非表示（データは保持）"),
                  ("blood", "血流", "#40e0ff",
                   "血流（LV-Blood）の表示 / 非表示（データは保持）"))

    def _ctlv_build_show_toggles(self, row) -> None:
        """Add the Epi境界 / Endo境界 / 血流 toggles to *row* (LVモード view
        tools; shown with 壁厚3D / 壁厚短軸 / CoronaryTree)."""
        self._ctlv_show_btns = {}
        for key, label, _c, tip in self._CTLV_SHOW:
            b = FitButton(t(label))
            b.setCheckable(True)
            b.setHelpToolTip(t(tip))
            b.clicked.connect(lambda _=False, k=key: self._ctlv_show_toggle(k))
            b.setVisible(False)
            row.addWidget(b)
            self._ctlv_show_btns[key] = b

    def _ctlv_show_state(self, key) -> tuple:
        """(available, shown) of one LV overlay."""
        g = lambda a: bool(getattr(self, a, False))      # noqa: E731
        lvv = getattr(self, "_lvv", None)
        if key == "epi":
            m = (getattr(self, "_lv", None) or {}).get("model") \
                if isinstance(getattr(self, "_lv", None), dict) else None
            avail = (getattr(self, "_lvv_epi_surf", None) is not None
                     or (m is not None
                         and len(getattr(m, "epi_contours", []) or []) >= 3))
            return avail, g("_lvv_epi_show") or g("_lv_border_show")
        if key == "endo":
            b = getattr(self, "_lvv_auto_endo_btn", None)
            avail = lvv is not None and b is not None and b.isEnabled()
            return avail, g("_lvv_endo_show")
        avail = (lvv is not None
                 and getattr(self, "_lvv_blood_comp", None) is not None)
        return avail, g("_lvv_mask_on") or g("_lvv_hl_on")

    def _ctlv_show_toggle(self, key) -> None:
        btn = (getattr(self, "_ctlv_show_btns", None) or {}).get(key)
        want = bool(btn.isChecked()) if btn is not None else True
        try:
            if key == "epi":
                if getattr(self, "_lvv", None) is not None \
                        or getattr(self, "_lvv_epi_surf", None) is not None:
                    self._lv_set_epi_blood_border(want)
                if not want or getattr(self, "_lvv", None) is None:
                    if hasattr(self, "set_lv_border_visible"):
                        self.set_lv_border_visible(want)
            elif key == "endo":
                self._lv_set_auto_endo(want)
            else:
                self._lv_set_blood("lv" if want else None)
        except Exception:                        # noqa: BLE001
            import traceback
            traceback.print_exc()
        f = getattr(self, "_lv_style_selectors", None)
        if callable(f):
            try:
                f()
            except Exception:                    # noqa: BLE001
                pass
        self._ctlv_sync()
        self._ct_render()
        self._diag_b(f"show_toggle {key} -> {want}")   # diag (MDV_DIAG_BLOOD)

    def _ctlv_sync(self) -> None:
        """Show the LV group in LVモード only (3-D, no LV edit step open), with
        coloured backgrounds while ON and 壁厚3D ⇔ 壁厚短軸 exclusive."""
        grp = getattr(self, "_ctlv_group", None)
        if not grp:
            return
        editing = False
        f = getattr(self, "_lv_mode_active", None)
        if callable(f):
            try:
                editing = bool(f())
            except Exception:                    # noqa: BLE001
                editing = False
        show = (self._ct_current_mode() == "lv"
                and getattr(self, "_mode", "3D") == "3D" and not editing)
        for w in grp:
            if w.isHidden() == show:
                w.setVisible(show)
        if not show:
            return
        style = getattr(self, "_lvv_style_toggle", None)
        tm = getattr(self, "_lvv_thick_mode", None)
        modes = getattr(self, "_THICK_MODES", {}) or {}
        for m, attr in (("3d", "_ctlv_thick3d_btn"), ("sax", "_ctlv_thicksax_btn")):
            btn = getattr(self, attr, None)
            if btn is None:
                continue
            btn.blockSignals(True)
            btn.setChecked(tm == m)
            btn.blockSignals(False)
            btn.setEnabled(tm is None or tm == m)
            if style is not None and m in modes:
                style(btn, modes[m][1], "black")
        for key, _l, color, _tip in self._CTLV_SHOW:
            btn = (getattr(self, "_ctlv_show_btns", None) or {}).get(key)
            if btn is None:
                continue
            avail, on = self._ctlv_show_state(key)
            btn.blockSignals(True)
            btn.setChecked(bool(on and avail))
            btn.blockSignals(False)
            btn.setEnabled(avail)
            if style is not None:
                style(btn, color, "black")
        tb = getattr(self, "_ctlv_tree_btn", None)
        if tb is not None:
            avail = self._ctlv_tree_available()
            on = bool(getattr(self, "_ct_lv_tree_on", False)) and avail
            tb.blockSignals(True)
            tb.setChecked(on)
            tb.blockSignals(False)
            tb.setEnabled(avail)
            if style is not None:
                style(tb, "#b8860b", "white")

    # ---- row order + per-mode visibility (user 2026-11-17) ------------------
    def _ct_bar_reorder(self, row) -> None:
        """Re-lay the LV bar's row *row* in the row-1 mode order — LV →
        CoronaryTree → Territory — with ONE 「LV:」 caption: the LVモード view
        tools (壁厚3D / 壁厚短軸 / CoronaryTree) follow the LV steps
        (MV-AoV+Apex … FullLV). Built order was [LV view tools] [CoronaryTree]
        [LV steps + edit groups] [Territory]."""
        from PyQt6.QtWidgets import QSpacerItem, QSizePolicy as _SP
        items = []
        while row.count():
            items.append(row.takeAt(0))

        def idx(w, seq):
            for i, it in enumerate(seq):
                if it is not None and it.widget() is w:
                    return i
            return -1
        i_cor = idx(getattr(self, "_coro_cap", None), items)
        i_sp = idx(getattr(self, "_coro_spacer_w", None), items)
        _tw = getattr(self, "_ct_terr_wrap", None)
        i_terr = idx(_tw, items) if _tw is not None else -1
        if i_terr < 0:                           # Territory moved to row 3:
            i_terr = len(items)                  # the trailing stretch stays last
            while i_terr > i_sp + 1 and items[i_terr - 1].widget() is None:
                i_terr -= 1
        if min(i_cor, i_sp) < 0 or not (i_cor < i_sp < i_terr + 1):
            for it in items:                     # unexpected layout → unchanged
                row.addItem(it)
            return
        view = [it for it in items[:i_cor]
                if it.widget() is not None
                and it.widget() is not getattr(self, "_ctlv_cap", None)]
        coro = items[i_cor:i_sp]
        spacer = items[i_sp]
        t_start = i_terr - 1 if (i_terr > i_sp + 1 and i_terr < len(items)
                                 and items[i_terr - 1].widget() is None
                                 and items[i_terr].widget() is not None) \
            else i_terr
        lv = items[i_sp + 1:t_start]
        terr = items[t_start:]
        at = idx(getattr(self, "_lv_file_label", None), lv)
        if at < 0:
            at = len(lv) - 1
        at += 1
        if at < len(lv) and lv[at].widget() is getattr(self, "_lv_endo_btn", None):
            at += 1
        gap = QSpacerItem(8, 0, _SP.Policy.Fixed, _SP.Policy.Minimum)
        order = lv[:at] + [gap] + view + lv[at:] + [spacer] + coro + terr
        for it in order:
            row.addItem(it)
        cap = getattr(self, "_ctlv_cap", None)
        if cap is not None:                      # the LV steps' 「LV:」 is the one
            cap.setVisible(False)
            grp = getattr(self, "_ctlv_group", None)
            if grp and cap in grp:
                grp.remove(cap)

    def _ct_bar_ctx(self) -> str:
        """The bar half shown for the current mode: Simple View = the 道しるべ
        (entry steps of every mode, "both"); LV / CoronaryTree = that mode's
        tools only (no crossing into another mode's sub-menus)."""
        m = self._ct_current_mode()
        return m if m in ("lv", "coronary") else "both"

    def _ct_terr_group_sync(self) -> None:
        """Territory Calc / Load / Save live in row 3 and show in every mode
        (Calc / Save enabled by the data, see refresh_ct_mode_bar)."""
        for nm in ("_ct_terr_btn", "_ct_terr_mgr_btn", "_ct_terr_load_btn",
                   "_ct_terr_save_btn"):
            b = getattr(self, nm, None)
            if b is not None and b.isHidden():
                b.setVisible(True)

    def _ct_terr_button(self):
        """Make the 「Territory Calc」 / 「Load」 / 「Save」 buttons once (placed
        in row 3 by _build_ct_files_line: FullLv ＋ corotree ▶ [Territory
        Calc] territory [Load] [Save]). Calc needs an LV (in-session or a
        FullLv) and a tree; Load opens a .territory.json; Save writes one
        (needs a computed Territory). Returns the Calc button
        (``_ct_terr_btn``)."""
        tb = getattr(self, "_ct_terr_btn", None)
        if tb is None:
            tb = FitButton(t("Territory Calc"))
            tb.setHelpToolTip(t(_TERR_BTN_TIP))
            tb.setStyleSheet(_TERR_BTN_CSS)
            tb.clicked.connect(
                lambda _c=False: self._ct_territory_compute_clicked())
            lb = FitButton(t("Load"))
            lb.setHelpToolTip(t(_TERR_LOAD_TIP))
            lb.clicked.connect(lambda _c=False: self._ct_territory_load_clicked())
            sb = FitButton(t("Save"))
            sb.setHelpToolTip(t(_TERR_SAVE_TIP))
            sb.clicked.connect(lambda _c=False: self._ct_territory_save_clicked())
            # Territory Manage (user 2026-11-27): the target list in its own
            # window (the CoronaryTree tab's list is small / easy to miss)
            mb = FitButton(t("Territory Manage"))
            mb.setHelpToolTip(t("Territory Manage：ターゲットの一覧を別ウインドウで表示します（表示・非表示、名称、心筋%、灌流域mL、元血管名、位置）。右ペイン右下のターゲット結果を右クリックしても開きます"))
            mb.setStyleSheet(_TERR_BTN_CSS)
            mb.clicked.connect(lambda _c=False: self._ct_territory_manage())
            self._ct_terr_mgr_btn = mb
            self._ct_terr_btn = tb
            self._ct_terr_load_btn = lb
            self._ct_terr_save_btn = sb
            self._ct_terr_wrap = None            # no row-1 group any more
        return tb

    def _build_ct_files_line(self) -> QWidget:
        """The row-3 readout of the loaded .FullLv.json / .corotree.json /
        .territory.json — each name followed by an Unload (red ⊗) button
        (user 2026-11-17) — read as inputs → Calc → result (user 2026-11-27):
        FullLv ⊗ ＋ corotree ⊗ ▶ [Territory Calc] territory ⊗ [Load] [Save]."""
        from PyQt6.QtWidgets import QToolButton
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(8, 0, 8, 0)
        row.setSpacing(2)
        sty = "color:#1a5fb4; font-weight:bold;"
        self._ct_files_lbl = QLabel("")          # the 「Loaded Files:」 caption
        self._ct_files_lbl.setStyleSheet(sty)
        row.addWidget(self._ct_files_lbl)
        self._ct_file_lbls = {}
        self._ct_unload_btns = {}
        self._ct_file_off = {}                   # kind → Unloaded (↻ shown)
        icon = self._ct_unload_icon()
        self._ct_terr_button()                   # Calc / Load / Save
        for i, kind in enumerate(("full", "tree", "terr")):
            if i:
                sep = QLabel("&nbsp;＋&nbsp;" if i == 1 else "&nbsp;▶&nbsp;")
                sep.setTextFormat(Qt.TextFormat.RichText)
                sep.setStyleSheet(sty)
                row.addWidget(sep)
            if kind == "terr":                   # inputs ▶ Calc ▶ result
                row.addWidget(self._ct_terr_btn)
                row.addWidget(self._ct_terr_mgr_btn)
                row.addSpacing(6)
            lb = QLabel("")
            lb.setStyleSheet(sty)
            lb.setTextFormat(Qt.TextFormat.RichText)
            row.addWidget(lb)
            b = QToolButton()
            b.setIcon(icon)
            b.setAutoRaise(True)
            b.setFixedSize(18, 18)
            b.setToolTip(t("このファイルをUnload（表示しない）"))
            b.setVisible(False)
            b.clicked.connect(lambda _=False, k=kind: self._ct_unload_clicked(k))
            row.addWidget(b)
            self._ct_file_lbls[kind] = lb
            self._ct_unload_btns[kind] = b
        row.addSpacing(6)                        # the territory file's own
        row.addWidget(self._ct_terr_load_btn)    # Load / Save
        row.addWidget(self._ct_terr_save_btn)
        row.addStretch(1)
        self._ct_files_wrap = w
        return w

    @staticmethod
    def _ct_unload_icon(reload: bool = False):
        """The Unload button: a red disc with a white ✕. Its pair, the reload
        button of an Unloaded file: a green disc with a white ↻ (same size and
        stroke, so the two read as one on / off control)."""
        from PyQt6.QtCore import QPointF, QRectF
        from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
        pm = QPixmap(32, 32)
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#2e8b57" if reload else "#d32f2f"))
        p.drawEllipse(2, 2, 28, 28)
        pen = QPen(QColor("white"))
        pen.setWidth(4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        if reload:
            p.setBrush(Qt.BrushStyle.NoBrush)
            # ↻: a 300° arc open at the top right, arrow head at its end
            p.drawArc(QRectF(9, 9, 14, 14), 60 * 16, 290 * 16)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("white"))
            p.drawPolygon(QPolygonF([QPointF(19.5, 5.5), QPointF(26.5, 10.5),
                                     QPointF(18.5, 14.0)]))
        else:
            p.drawLine(10, 10, 22, 22)
            p.drawLine(22, 10, 10, 22)
        p.end()
        return QIcon(pm)

    def _ct_files_text(self) -> str:
        """The whole readout as one string (caption + the three parts)."""
        lbls = getattr(self, "_ct_file_lbls", None) or {}
        parts = [lbls[k].text() for k in ("full", "tree", "terr") if k in lbls]
        cap = getattr(self, "_ct_files_lbl", None)
        return ((cap.text() if cap is not None else "")
                + " &nbsp;／&nbsp; ".join(parts))

    def _ct_unload_clicked(self, kind: str) -> None:
        """⊗ next to a loaded file: Unload it — its data is kept but nothing of
        it is shown (the name greys, the button turns ↻). ↻: load it again
        (shown at once from the kept data)."""
        shell = self._ct_mode_shell()
        off = bool((getattr(self, "_ct_file_off", None) or {}).get(kind))
        fn = "ct_reload_file" if off else "ct_unload_file"
        if shell is None or not hasattr(shell, fn):
            return
        try:
            getattr(shell, fn)(self, kind)
        except Exception:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
        self.refresh_ct_mode_bar()

    def _ct_lv_unload(self) -> None:
        """FullLv Unload on this viewer: leave LVモード, hide every LV overlay
        and treat the LV as not loaded (data kept; a FullLv load / an LV step
        brings it back)."""
        cur = self._ct_current_mode()
        if cur == "territory":
            self._ct_territory_leave()
            self._ct_switch_inplace("standard", auto=True)
        elif cur == "lv":
            self._ct_switch_inplace("standard", auto=True)
        self._ct_show_lv(False)
        self._ct_lv_unloaded = True
        self._ct_render()

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
        self._full_lv_sig_base = None     # 変更あり baseline (loaded / saved)
        self._ct_lv_unloaded = False      # FullLv ⊗ Unload: LV kept, not shown

    def _ct_lv_data(self) -> bool:
        """Any LV analysis on this CT — measured in this session or loaded
        (not after a FullLv Unload: then it is kept but counts as absent)."""
        if getattr(self, "_ct_lv_unloaded", False):
            return False
        if (getattr(self, "_lv", None) is not None
                or getattr(self, "_lvv", None) is not None
                or getattr(self, "_lvv_epi_surf", None) is not None
                or getattr(self, "_lv_apex", None) is not None):
            return True
        vs = getattr(self, "_lv_valves", None) or {}
        return any(v is not None for v in vs.values())

    def _ct_lv_ready(self) -> bool:
        """The in-session LV is complete enough to build a FullLv (the FullLV
        button's own readiness rule). Not after a FullLv Unload."""
        if getattr(self, "_ct_lv_unloaded", False):
            return False
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
            # Calc: any mode (in the review = recompute) once a tree and an
            # LV / FullLv exist; the tooltip names what is missing.
            has_tree = bool(state.get("tree") or state.get("tree_mem"))
            has_lv = (self._ct_lv_ready() or bool(full)
                      or bool(state.get("full_mem")))
            tb.setEnabled(has_tree and has_lv)
            mb = getattr(self, "_ct_terr_mgr_btn", None)
            if mb is not None:
                mb.setEnabled(has_tree)          # the window says what is missing
            miss = [x for x, ok in ((t("FullLv（または LV 計測）"), has_lv),
                                    (t("corotree（冠動脈ツリー）"), has_tree))
                    if not ok]
            tip = t(_TERR_BTN_TIP)
            if miss:
                tip += "\n\n" + t("足りないもの: {m}", m=t("・").join(miss))
            if tb.property("ct_tip") != tip:
                tb.setProperty("ct_tip", tip)
                tb.setHelpToolTip(tip)
        for nm in ("_ct_terr_save_btn", "_terr_save_btn2"):
            b = getattr(self, nm, None)
            if b is not None:
                b.setEnabled(bool(state.get("terr")))     # a computed Territory
        self._ct_terr_group_sync()
        lbl = getattr(self, "_ct_files_lbl", None)
        if lbl is not None:
            import html
            import os
            none = t("なし")
            full_file = bool(full)
            if not full and lv_mem:
                sp = getattr(self, "_full_lv_saved_path", "") or ""
                if getattr(self, "_full_lv_saved", False) and sp:
                    full = os.path.basename(sp)
                    full_file = True
                else:
                    full = (t("計測済み（保存済）")
                            if getattr(self, "_full_lv_saved", False)
                            else t("計測済み（未保存）"))
            tree_file = bool(tree)
            if not tree and state.get("tree_mem"):
                tree = t("ツリーあり（未保存）")
            terr = state.get("terr_file", "") or ""
            terr_file = bool(terr)
            if not terr and state.get("terr"):
                terr = t("計算済み（未保存）")
            # 「変更あり・未保存」: a named file whose content was changed after
            # it was loaded / saved (user 2026-11-17).
            full_mod = full_file and self._full_lv_modified()
            tree_mod = tree_file and bool(state.get("tree_dirty"))
            terr_mod = terr_file and (bool(state.get("terr_changed"))
                                      or tree_mod or full_mod)
            mod = html.escape(t("変更あり・未保存"))

            # ⊗ Unloaded files: the name stays, greyed, with ↻ (reload).
            off = {}
            if getattr(self, "_ct_lv_unloaded", False):
                nm = state.get("full_off") or ""
                sp = getattr(self, "_full_lv_saved_path", "") or ""
                if not nm and sp and getattr(self, "_full_lv_saved", False):
                    nm = os.path.basename(sp)
                if nm:
                    off["full"] = nm
            elif not full_file and state.get("full_off"):
                off["full"] = state["full_off"]
            if not tree_file and state.get("tree_off"):
                off["tree"] = state["tree_off"]
            if not terr_file and state.get("terr_off"):
                off["terr"] = state["terr_off"]
            for k_, v_ in list(off.items()):
                if v_ == "*":
                    off[k_] = t("計算済み（未保存）")

            def short(s_, n=34):
                s_ = str(s_ or "")
                return s_ if len(s_) <= n else (s_[:n // 2 - 1] + "…"
                                                + s_[-(n - n // 2):])

            def part(name, text, modified, grey=None):
                if grey:
                    return ("<span style='color:#a0a0a0;'>" + html.escape(name)
                            + ": " + html.escape(short(grey)) + "</span>")
                s_ = html.escape(short(text or none))
                if modified:
                    s_ += ("<span style='color:#d2691e;'>（" + mod
                           + "）</span>")
                return html.escape(name) + ": " + s_
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setText(html.escape(t("読込ファイル")) + ":&nbsp; ")
            lbls = getattr(self, "_ct_file_lbls", None) or {}
            btns = getattr(self, "_ct_unload_btns", None) or {}
            for kind, name, text, modified, is_file in (
                    ("full", "FullLv", full, full_mod, full_file),
                    ("tree", "corotree", tree, tree_mod, tree_file),
                    ("terr", "territory", terr, terr_mod, terr_file)):
                grey = off.get(kind) if not is_file else None
                if kind in lbls:
                    lbls[kind].setText(part(name, text, modified, grey))
                    full_nm = str(grey or text or "")
                    if lbls[kind].toolTip() != full_nm:
                        lbls[kind].setToolTip(full_nm)   # the whole name
                if kind in btns:
                    # ⊗ next to a loaded FILE; ↻ next to an Unloaded one;
                    # nothing for 未保存 / なし.
                    b_ = btns[kind]
                    fo = self.__dict__.setdefault("_ct_file_off", {})
                    was = bool(fo.get(kind))
                    now = bool(grey)
                    fo[kind] = now
                    if was != now or b_.property("ct_icon_set") is None:
                        b_.setIcon(self._ct_unload_icon(reload=now))
                        b_.setToolTip(
                            t("再読込（もう一度表示）") if now
                            else t("このファイルをUnload（表示しない）"))
                        b_.setProperty("ct_icon_set", True)
                    b_.setVisible(bool(is_file) or now)
        self._ctlv_sync()

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
            if key == "lv":
                self._ct_lv_unloaded = False     # an LV step: the LV is back
            # What each mode shows (data is always kept, only hidden):
            #   標準         the 3DCT only
            #   LV           Epi / Blood / Endo only (no tree, no 領域色)
            #   CoronaryTree the tree only (no LV, no 領域色)
            #   Territory    everything (_ct_territory_enter)
            self._ct_show_lv(key == "lv")
            self._ct_tree_suppressed = not (
                key == "coronary"
                or (key == "lv" and getattr(self, "_ct_lv_tree_on", False)))
            self._ct_tree_apply()
            if shell is not None and hasattr(shell, "ct_territory_tint_off"):
                try:
                    shell.ct_territory_tint_off(self)     # 領域色 off (kept)
                except Exception:                        # noqa: BLE001
                    pass
            if shell is not None and hasattr(shell, "ct_tree_show"):
                if key == "coronary":
                    shell.ct_tree_show(self, True, raise_tab=True)
        finally:
            self._ct_switching = False
        for fn in ("_apply_file_context", "_lv_update_submode_ui"):
            try:
                getattr(self, fn)()
            except Exception:                            # noqa: BLE001
                pass
        self.refresh_ct_mode_bar()
        self._ct_render()
        self._diag_b(f"switch_inplace -> {key}")   # diag (MDV_DIAG_BLOOD)

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
                # 壁厚3D / 壁厚短軸 heat map (user 2026-11-27: gone in Simple
                # View whatever the view — 3-D or short axis; back in LVモード)
                "thick": (None if getattr(self, "_lvv_thick_mode", None) is None
                          else {"mode": self._lvv_thick_mode,
                                "vol": getattr(self, "_lvv_thick_vol", None),
                                "stats": getattr(self, "_lvv_thick_stats", None),
                                "thi": getattr(self, "_lvv_thick_thi", 20.0)}),
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
        th = snap.get("thick")
        if th:
            self._lvv_thick_mode = th["mode"] if on else None
            self._lvv_thick_vol = th["vol"] if on else None
            self._lvv_thick_stats = th["stats"] if on else None
            if on:
                self._lvv_thick_thi = th.get("thi", 20.0)
            _do("_lv_update_text")
            _do("_lvv_thick_refresh_display")
            _do("_lvv_thick_sync_buttons")
        _do("_lv_style_selectors")
        # the LV pane overlay (base-cut line, apex markers, SAX lines …) is
        # rebuilt now: hidden in Simple View / CoronaryTree, back in LVモード
        if hasattr(self, "_redraw_lv"):
            for k in ("A", "B"):
                _do("_redraw_lv", k)

    def _ct_lv_editing_msg(self) -> None:
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self.window(), t("モード"),
                t("LV の編集中です。編集を終了してから切り替えてください。"))
        except Exception:                                # noqa: BLE001
            pass

    # ------------------------------------------------- Territory (in place)
    def _ct_territory_manage(self) -> None:
        """「Territory Manage」: the shell opens the target window."""
        shell = self._ct_mode_shell()
        if shell is not None and hasattr(shell, "coronary_target_action"):
            try:
                shell.coronary_target_action(0, "manage")
            except Exception:                            # noqa: BLE001
                pass

    def _ct_territory_compute_clicked(self) -> None:
        """「Territory Calc」 (row 3, any mode): the shell computes the territory
        from this CT's LV + tree and calls back _ct_territory_enter. In the
        Territory review it is a recompute (the review is left first); while
        an LV step is being edited it asks to finish that first."""
        shell = self._ct_mode_shell()
        if shell is None or not hasattr(shell, "ct_territory_compute"):
            return
        if hasattr(self, "_lv_mode_active") and self._lv_mode_active():
            self._ct_lv_editing_msg()
            return
        if self._ct_current_mode() == "territory":
            try:
                self._ct_territory_leave()
            except Exception:                            # noqa: BLE001
                import traceback
                traceback.print_exc()
        try:
            shell.ct_territory_compute(self)
        except Exception:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
        self.refresh_ct_mode_bar()

    def _ct_territory_load_clicked(self) -> None:
        """Territory ▸ Load (.territory.json) via the shell."""
        shell = self._ct_mode_shell()
        if shell is None or not hasattr(shell, "ct_territory_load"):
            return
        try:
            shell.ct_territory_load(self)
        except Exception:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
        self.refresh_ct_mode_bar()

    def _ct_territory_save_clicked(self) -> None:
        """Territory ▸ Save (XXXX.territory.json) via the shell."""
        shell = self._ct_mode_shell()
        if shell is None or not hasattr(shell, "ct_territory_save"):
            return
        try:
            shell.ct_territory_save(self)
        except Exception:                                # noqa: BLE001
            import traceback
            traceback.print_exc()

    def _ct_terr_bar_file_buttons(self, row) -> None:
        """Load / Save on the Territory bar too (Territoryモード hides the row
        that holds the Calc / Load / Save group)."""
        lb = FitButton(t("Load"))
        lb.setHelpToolTip(t(_TERR_LOAD_TIP))
        lb.clicked.connect(lambda _c=False: self._ct_territory_load_clicked())
        sb = FitButton(t("Save"))
        sb.setHelpToolTip(t(_TERR_SAVE_TIP))
        sb.clicked.connect(lambda _c=False: self._ct_territory_save_clicked())
        row.addWidget(lb)
        row.addWidget(sb)
        self._terr_load_btn2 = lb
        self._terr_save_btn2 = sb

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

    # ---- FullLv 「変更あり・未保存」 (user 2026-11-17) ------------------------
    def _full_lv_sig(self):
        """A cheap fingerprint of what a .FullLv.json holds — the three LV
        steps: MV-AoV + Apex, Epi, Blood/Endo. Results are identified by their
        objects (a re-trace / re-Calc makes new ones), the small values by
        value. No mask is copied or hashed (cheap on every bar refresh)."""
        def r(a):
            if a is None:
                return None
            try:
                return tuple(np.round(np.asarray(a, float).ravel(), 3).tolist())
            except Exception:                    # noqa: BLE001
                return repr(a)
        vs = getattr(self, "_lv_valves", None) or {}
        valves = tuple(
            (k, None if v is None else tuple(r(x) for x in v))
            for k, v in sorted(vs.items()))
        mvaov = getattr(self, "_lv_mvaov", None)
        try:
            import json
            mvaov = json.dumps(mvaov, sort_keys=True, default=str)
        except Exception:                        # noqa: BLE001
            mvaov = repr(mvaov)
        def spin(n):
            w = getattr(self, n, None)
            try:
                return float(w.value()) if w is not None else None
            except Exception:                    # noqa: BLE001
                return None
        return (
            valves, r(getattr(self, "_lv_apex", None)), mvaov,           # step 1
            id(getattr(self, "_lvv_epi_surf", None)),                    # step 2
            id(getattr(self, "_lvv_epi_model_dict", None)),
            bool(getattr(self, "_lv_dirty", False)),
            id(getattr(self, "_lvv_blood_comp", None)),                  # step 3
            id(getattr(self, "_lv_endo_mask_comp", None)),
            spin("_lvv_lo_spin"), spin("_lvv_hi_spin"),
            getattr(self, "_lvv_lvd_level", None),
            r(getattr(self, "_lvv_blood_apex", None)),
        )

    def _full_lv_mark_clean(self) -> None:
        """The LV now equals a .FullLv.json (just loaded / saved)."""
        try:
            self._full_lv_sig_base = self._full_lv_sig()
        except Exception:                        # noqa: BLE001
            self._full_lv_sig_base = None

    def _full_lv_modified(self) -> bool:
        base = getattr(self, "_full_lv_sig_base", None)
        if base is None:
            return False
        try:
            return self._full_lv_sig() != base
        except Exception:                        # noqa: BLE001
            return False

    def _diag_b(self, tag: str) -> None:
        """Opt-in diagnostics for the right-pane blood (2026-11-17): with the
        environment variable MDV_DIAG_BLOOD=1, append the blood-overlay state
        of both panes to %TEMP%/mdv_bdiag.txt. No patient data — only actor /
        LUT / reslice / camera state. Off (a no-op) otherwise."""
        import os as _os
        if _os.environ.get("MDV_DIAG_BLOOD", "") != "1":
            return
        try:
            import time
            import tempfile as _tf
            path = _os.path.join(_tf.gettempdir(), "mdv_bdiag.txt")
            mv = getattr(self, "_lvv_mask_vol", None)
            lines = [f"--- {time.strftime('%H:%M:%S')} {tag}  mode="
                     f"{self._ct_current_mode()} terr="
                     f"{getattr(self, '_territory_display', None)} "
                     f"lvv={getattr(self, '_lvv', None) is not None} "
                     f"mask_on={getattr(self, '_lvv_mask_on', None)} "
                     f"hl_on={getattr(self, '_lvv_hl_on', None)} "
                     f"mask_vol={None if mv is None else mv.GetDimensions()} "
                     f"lv_hidden={getattr(self, '_ct_lv_hidden', None)} "
                     f"vr_on={getattr(self, '_vr_on', None)}"]
            for k in ("A", "B"):
                p = self.pane[k]
                rm = p.reslice_mask
                inp = rm.GetInput()
                rm.Update()
                out = rm.GetOutput()
                rng = out.GetScalarRange() if out is not None else None
                lut = p.colors_mask.GetLookupTable()
                lv = lut.GetTableValue(1) if lut is not None and \
                    lut.GetNumberOfTableValues() > 1 else None
                cam = p.ren.GetActiveCamera()
                hidden = len((getattr(self, "_vr_hidden", {}) or {}).get(k, []))
                lines.append(
                    f"  {k}: vis actor={p.actor.GetVisibility()} hl="
                    f"{p.actor_hl.GetVisibility()} mask="
                    f"{p.actor_mask.GetVisibility()} | mask_in="
                    f"{None if inp is None else inp.GetDimensions()} "
                    f"same_vol={inp is mv} out_ext={out.GetExtent()} "
                    f"out_range={rng} lut1={lv} | img_ext="
                    f"{p.reslice.GetOutput().GetExtent()} | cam pp="
                    f"{cam.GetParallelProjection()} pos="
                    f"{tuple(round(c, 1) for c in cam.GetPosition())} fp="
                    f"{tuple(round(c, 1) for c in cam.GetFocalPoint())} "
                    f"clip={tuple(round(c, 1) for c in cam.GetClippingRange())} "
                    f"| mask_pos={p.actor_mask.GetPosition()} "
                    f"vr_hidden={hidden} | mask_bounds="
                    f"{tuple(round(c, 1) for c in p.actor_mask.GetBounds())} "
                    f"img_bounds="
                    f"{tuple(round(c, 1) for c in p.actor.GetBounds())} "
                    f"mask_um={p.actor_mask.GetUserMatrix() is not None} "
                    f"cam_vup={tuple(round(c, 2) for c in cam.GetViewUp())} "
                    f"peel={p.ren.GetUseDepthPeeling()}")
                props = p.ren.GetViewProps()
                props.InitTraversal()
                vis = []
                while True:
                    a = props.GetNextProp()
                    if a is None:
                        break
                    if a.GetVisibility():
                        nm = a.GetClassName()
                        for an in ("actor", "actor_hl", "actor_mask",
                                   "actor_terr", "actor_thick", "vr_volume"):
                            if getattr(p, an, None) is a:
                                nm = an
                        try:
                            bb = tuple(round(c, 0) for c in a.GetBounds())
                        except Exception:            # noqa: BLE001
                            bb = None
                        vis.append(f"{nm}{bb if bb and bb[0] < 1e30 else ''}")
                lines.append(f"  {k} visible props ({len(vis)}): "
                             + ", ".join(vis))
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as exc:                     # noqa: BLE001
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(f"diag error {tag}: {exc!r}\n")
            except Exception:                        # noqa: BLE001
                pass

    def _ct_restore_blood_panes(self) -> None:
        """Leaving the Territory review: the RIGHT pane was the VR there (no
        blood drawn on it). Put the blood overlays back on BOTH MPR panes — the
        tint / LV-Blood layers visible, the LV-Blood volume wired to both mask
        reslices, the LUTs re-applied — so LV / Simple View / CoronaryTree show
        the blood on the right pane again (user 2026-11-17: it stayed off there
        after a .territory.json load). VTK panes only; no-op elsewhere."""
        panes = getattr(self, "pane", None)
        if not isinstance(panes, dict):
            return
        mv = getattr(self, "_lvv_mask_vol", None)
        vr_on = getattr(self, "_vr_on", {}) or {}
        for k in ("A", "B"):
            p = panes.get(k)
            if p is None or vr_on.get(k):
                continue
            for n in ("actor", "actor_hl", "actor_mask"):
                a = getattr(p, n, None)
                if a is not None and hasattr(a, "SetVisibility"):
                    a.SetVisibility(True)
            rm = getattr(p, "reslice_mask", None)
            if mv is not None and rm is not None and rm.GetInput() is not mv:
                rm.SetInputData(mv)
            f = getattr(self, "_vr_depth_peeling", None)
            if callable(f):
                f(p, False)                      # VR-only (it hid the blood)
        try:
            if hasattr(self, "_refresh"):
                self._refresh(reset_cam=False)   # reslice the mask on each plane
            for fn in ("_lvv_update_mask", "_lvv_update_highlight"):
                f = getattr(self, fn, None)
                if callable(f):
                    f()
        except Exception:                        # noqa: BLE001
            import traceback
            traceback.print_exc()

    def _ct_lv_load_editable(self, full) -> bool:
        """Apply a FullLv bundle as the EDITABLE LV (MV-AoV / Apex / Epi /
        Blood-Endo — apply_lv_analysis, as a lone .FullLv.json drop does), not
        the display-only review apply. It came from a saved file → FullLV reads
        保存済. Returns True if anything was applied."""
        shell = self._ct_mode_shell()
        conv = getattr(shell, "_full_lv_to_files", None) if shell else None
        if conv is None or not hasattr(self, "apply_lv_analysis"):
            return False
        try:
            files = conv(full)
            ok = bool(files) and bool(self.apply_lv_analysis(files))
        except Exception:                        # noqa: BLE001
            import traceback
            traceback.print_exc()
            return False
        if ok:
            self._full_lv_saved = True
            self._full_lv_mark_clean()
            w = getattr(shell, "_corotree_win", None)
            p = getattr(w, "_full_lv_path", "") or "" if w is not None else ""
            if p:
                self._full_lv_saved_path = p
            f = getattr(self, "_full_lv_refresh_btn", None)
            if callable(f):
                try:
                    f()
                except Exception:                # noqa: BLE001
                    pass
        return ok

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
        def _light():
            return (self._ct_lv_ready()
                    and getattr(self, "_lvv", None) is not None
                    and getattr(self, "_lvv_epi_surf", None) is not None)
        light = _light()
        # No in-session LV: load the FullLv EDITABLY first (the FullLv-drop
        # loader), so LVモード afterwards continues it without recomputing
        # Epi / Blood / Endo (user 2026-11-17) — then enter the light path.
        if not light and full and self._ct_lv_load_editable(full):
            light = _light()
        if not light:
            ok = bool(hasattr(self, "apply_full_lv") and self.apply_full_lv(full))
            if ok:
                self._full_lv_mark_clean()
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
                   "_terr_bar_sync", "_update_active_frames"):
            try:
                getattr(self, fn)()
            except Exception:                    # noqa: BLE001
                pass
        self._ct_render()
        QTimer.singleShot(0, self._terr_vr_rotate)   # VR on + Rotate
        self._diag_b("territory_enter(light)")   # diag (MDV_DIAG_BLOOD)
        QTimer.singleShot(4000, lambda: self._diag_b("territory_enter+4s"))   # diag (MDV_DIAG_BLOOD)
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
        for fn in ("_refresh_tool_availability", "_terr_bar_sync",
                   "_update_active_frames"):
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
        self._ct_restore_blood_panes()
        self._diag_b("territory_leave")          # diag (MDV_DIAG_BLOOD)
