"""CT work-mode switch shared by both CT viewers (Windows VTK ``ct_viewer.py``
and Mac pygfx ``ct_viewer_pygfx.py``), so the two behave identically.

Four mode buttons sit right of 3D/2D on the Plane bar (row 1):

    標準モード          plain CT (no LV / coronary overlay)      — always enabled
    LVモード            the editable LV from the .FullLv.json    — FullLv loaded
    CoronaryTreeモード  only the coronary CPR lines              — corotree loaded
    Territoryモード     perfusion territory + VR review          — BOTH loaded

A one-line readout under the LV / Territory bar (row 3) names the .FullLv.json and
.corotree.json the dock currently holds for THIS CT; that same state decides
which buttons are enabled. The data lives in the shell / Coronary Tree dock,
so the viewer only asks (``shell.ct_mode_state``) and requests a switch
(``shell.ct_set_mode``) — the shell re-opens the CT through the same paths a
file drop uses. The CURRENT mode is read from the viewer's own state
(``_territory_display`` / ``_bottom_ctx``), so it always matches the screen.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from multi_dicomviewer.i18n import t
from multi_dicomviewer.ui.study_browser import FitButton

#: (key, label, tooltip) in button order.
_CT_MODES = (
    ("standard", "標準モード",
     "標準モード：LV / 冠動脈ツリーの表示を外した通常の 3DCT 表示（読み込んだ "
     "ファイルはそのまま保持され、他のモードへいつでも戻れます）"),
    ("lv", "LVモード",
     "LV 編集モードへ移動：MV / AoV / Apex / Epi / Blood-Endo を編集できる画面"
     "（.FullLv.json 読込時のみ有効）"),
    ("coronary", "CoronaryTreeモード",
     "CoronaryTree 編集モードへ移動：領域色・Epi・Blood・Endo・LVD を消し、冠動脈 "
     "CPR 線だけを表示して CPR 線を編集できる画面（.corotree.json 読込時のみ有効）"),
    ("territory", "Territoryモード",
     "Territory（灌流領域 + VR）表示へ移動（.FullLv.json と .corotree.json の"
     "両方を読み込んだときのみ有効）"),
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
    def _ct_current_mode(self) -> str:
        if getattr(self, "_territory_display", False):
            return "territory"
        ctx = getattr(self, "_bottom_ctx", "both")
        if ctx == "lv":
            return "lv"
        if ctx == "coronary":
            return "coronary"
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
        avail = {"standard": True, "lv": bool(full), "coronary": bool(tree),
                 "territory": bool(full and tree)}
        cur = self._ct_current_mode()
        for key, b in btns.items():
            # The CURRENT mode stays enabled even if its file just went away, so
            # the highlight never turns into an unreadable greyed-checked button.
            b.setEnabled(avail[key] or key == cur)
            if b.isChecked() != (key == cur):
                b.setChecked(key == cur)
        lbl = getattr(self, "_ct_files_lbl", None)
        if lbl is not None:
            none = t("なし")
            lbl.setText(t("読込ファイル") + ":  FullLv: " + (full or none)
                        + "  ／  corotree: " + (tree or none))

    def _ct_mode_clicked(self, key: str) -> None:
        cur = self._ct_current_mode()
        if key == cur:                         # already there — keep it checked
            self.refresh_ct_mode_bar()
            return
        shell = self._ct_mode_shell()
        if shell is not None and hasattr(shell, "ct_set_mode"):
            try:
                shell.ct_set_mode(key, self)
            except Exception:                            # noqa: BLE001
                import traceback
                traceback.print_exc()
        # The switch re-opens the CT asynchronously (a NEW viewer); until then
        # this one keeps showing the mode it is actually in.
        self.refresh_ct_mode_bar()
