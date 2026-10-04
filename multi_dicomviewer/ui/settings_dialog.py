"""Unified Settings popup (top-bar "Settings" button).

Gathers the app-wide display preferences in one place:

  * Display count — how many CT / angio panes stay fully loaded before older
    ones become memory-saving stills (see settings.load_live_caps).
  * Angio image quality — S-Cine / S-Zoom / Denoise (settings.display_quality).
  * CT colour — a button that opens the HU colour-map editor for the active
    CT pane (that editor applies live and is per-viewer).
"""
from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from multi_dicomviewer.core import image_quality, settings
from multi_dicomviewer.i18n import t


class _ColorButton(QPushButton):
    """A small swatch button that opens a colour picker and remembers the pick
    as a '#rrggbb' string (`.hex`)."""

    def __init__(self, hex_color: str, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self.hex = hex_color
        self.setFixedSize(46, 22)
        self.clicked.connect(self._pick)
        self._apply()

    def _apply(self):
        c = QColor(self.hex)
        txt = "#000000" if c.lightnessF() > 0.6 else "#ffffff"
        self.setStyleSheet(
            f"background:{self.hex}; color:{txt}; border:1px solid #888;")
        self.setText(self.hex)

    def _pick(self):
        c = QColorDialog.getColor(QColor(self.hex), self, self._title)
        if c.isValid():
            self.hex = c.name()
            self._apply()


class SettingsDialog(QDialog):
    """App-wide display settings. *caps* is {"CT": n, "XA": m}; *quality* is the
    display-quality dict; *on_ct_color* opens the CT colour-map editor."""

    def __init__(self, caps: dict, quality: dict, on_ct_color,
                 on_advanced=None, parent=None, lv_endo=None, lv_wall=None,
                 coronary=None):
        super().__init__(parent)
        self.setWindowTitle(t("Settings"))
        self._on_ct_color = on_ct_color
        self._on_advanced = on_advanced
        # Wider + scrollable: the sections stack vertically and there are now many,
        # so the content scrolls inside a fixed-size, comfortably wide dialog.
        self.setMinimumWidth(560)
        outer = QVBoxLayout(self)
        _scroll = QScrollArea()
        _scroll.setWidgetResizable(True)
        _content = QWidget()
        _scroll.setWidget(_content)
        outer.addWidget(_scroll, 1)
        root = QVBoxLayout(_content)

        # ---- Display count -------------------------------------------------
        gb_count = QGroupBox(t("Display count"))
        cform = QFormLayout(gb_count)
        cnote = QLabel(
            t("How many images stay fully loaded at once. Older panes beyond "
              "this become a still snapshot to save memory; click one to "
              "reload it. Higher = more interactive at once, but more memory."))
        cnote.setWordWrap(True)
        cform.addRow(cnote)
        self._spins: dict[str, QSpinBox] = {}
        for key, label in (("CT", t("CT panes")), ("XA", t("Angio panes"))):
            sb = QSpinBox()
            sb.setRange(settings.LIVE_CAPS_MIN[key], settings.LIVE_CAPS_MAX[key])
            sb.setValue(int(caps.get(key, settings.LIVE_CAPS_DEFAULT[key])))
            sb.setSuffix(t("  (max {n})", n=settings.LIVE_CAPS_MAX[key]))
            self._spins[key] = sb
            cform.addRow(label, sb)
        root.addWidget(gb_count)
        # Warn the FIRST time the user RAISES CT panes to ≥2 (several live CT
        # volumes can exhaust GPU/RAM → force-quit on modest machines). Connected
        # AFTER the initial setValue above, so opening the dialog on an already-≥2
        # setting doesn't nag. Cancel reverts to 1; OK acknowledges for this
        # dialog session (raising 2→3→4 won't re-prompt).
        self._ct_multi_ack = int(caps.get("CT", 1)) >= 2
        self._spins["CT"].valueChanged.connect(self._on_ct_count_changed)

        # ---- Angio image quality ------------------------------------------
        gb_q = QGroupBox(t("Angio image quality"))
        qlay = QVBoxLayout(gb_q)
        have_cv2 = image_quality.available()
        self._q_boxes: dict[str, QCheckBox] = {}
        for key, label, tip, needs_cv2 in (
            ("xa_hq_cine", t("S-Cine"),
             t("Smooth (bilinear) frames even during cine playback. "
               "Default off (fast). Affects motion only."), False),
            ("xa_smooth", t("S-Zoom"),
             t("High-quality (Lanczos) upscaling — sharper when enlarged. "
               "Default off. For fast machines."), True),
            ("xa_denoise", t("Denoise"),
             t("Edge-preserving noise reduction — calms speckle while keeping "
               "vessel/catheter edges crisp. Default off."), True),
        ):
            cb = QCheckBox(label)
            cb.setChecked(bool(quality.get(key)))
            cb.setToolTip(tip)
            if needs_cv2 and not have_cv2:
                cb.setEnabled(False)
                cb.setToolTip(tip + t("  (OpenCV not available)"))
            self._q_boxes[key] = cb
            qlay.addWidget(cb)
        # Advanced… → fine denoise/sharpen/CLAHE with a live preview.
        adv_btn = QPushButton(t("Advanced…"))
        adv_btn.setToolTip(
            t("Fine-tune denoise strength, sharpening and local contrast "
              "(applies while Denoise is on; live preview)"))
        adv_btn.setEnabled(callable(self._on_advanced) and have_cv2)
        adv_btn.clicked.connect(self._open_advanced)
        qlay.addWidget(adv_btn)
        root.addWidget(gb_q)

        # ---- CT colour -----------------------------------------------------
        gb_c = QGroupBox(t("CT colour"))
        clay = QHBoxLayout(gb_c)
        clay.addWidget(QLabel(t("HU-value colour map:")))
        color_btn = QPushButton(t("Color setting…"))
        color_btn.setToolTip(
            t("Edit the HU-value colour bands for the active CT pane"))
        color_btn.clicked.connect(self._open_color)
        clay.addWidget(color_btn)
        clay.addStretch(1)
        root.addWidget(gb_c)

        # ---- CT image quality (Mac 3DCT only) -----------------------------
        gb_ctq = QGroupBox(t("CT Image Quality (Only Mac)"))
        ctqlay = QVBoxLayout(gb_ctq)
        self._ctq_radios: dict[str, QRadioButton] = {}
        mode = quality.get("ct_quality_mode", "adaptive")
        if mode not in ("high", "adaptive", "low"):
            mode = "adaptive"
        for val, label, tip in (
            ("high", t("Always high quality"),
             t("Keep 3DCT MPR sharp even while dragging / zooming / rotating. "
               "Smoother on a fast Mac; heavier on a slow one.")),
            ("adaptive", t("High when still, low while moving"),
             t("Sharp static image; a coarse preview only while you "
               "drag / zoom / rotate. Default.")),
            ("low", t("Always low quality"),
             t("Always the coarse preview — fastest, for slow machines.")),
        ):
            rb = QRadioButton(label)
            rb.setToolTip(tip)
            rb.setChecked(val == mode)
            self._ctq_radios[val] = rb
            ctqlay.addWidget(rb)
        root.addWidget(gb_ctq)

        # ---- LV Auto-Endo (advanced) --------------------------------------
        # The everyday 肉柱 knob lives on the Blood/Endo bar; these are the rarely
        # touched per-method shape/resolution values. Defaults are good.
        lv = dict(settings._LV_ENDO_DEFAULTS)
        if isinstance(lv_endo, dict):
            lv.update({k: lv_endo[k] for k in lv if k in lv_endo})
        gb_lv = QGroupBox(t("LV Auto-Endo (advanced)"))
        lvform = QFormLayout(gb_lv)
        lvnote = QLabel(
            t("Advanced Auto-Endo shape/resolution. Normally leave these — the "
              "everyday 肉柱 knob is on the Blood/Endo bar. Smaller resolution "
              "values are finer but slower."))
        lvnote.setWordWrap(True)
        lvform.addRow(lvnote)
        self._lv_spins: dict[str, QDoubleSpinBox | QSpinBox] = {}
        # (key, label, tip, is_int, lo, hi, step, decimals, suffix)
        lv_fields = (
            ("min_chord_mm", t("凸包滑: 凹み判定 (弦長)"),
             t("Hull edges longer than this (mm) are treated as concavities and "
               "bulged; shorter = convex wall, kept straight."),
             False, 1.0, 30.0, 0.5, 1, t(" mm")),
            ("n_meridians", t("放射: 本数"),
             t("Number of radial directions used by the 放射 method. Higher = "
               "finer angular detail, a little slower."),
             True, 60, 720, 20, 0, ""),
            ("grid_mm", t("マスク解像度"),
             t("Auto-Endo mask sampling pitch (mm). Smaller = finer surface, "
               "slower to compute."),
             False, 0.3, 2.0, 0.1, 1, t(" mm")),
            ("step_mm", t("輪郭解像度"),
             t("Displayed Endo outline sampling pitch (mm). Smaller = smoother "
               "line, a little slower to draw."),
             False, 0.3, 1.5, 0.05, 2, t(" mm")),
        )
        for key, label, tip, is_int, lo, hi, step, dec, suf in lv_fields:
            sb = QSpinBox() if is_int else QDoubleSpinBox()
            if not is_int:
                sb.setDecimals(dec)
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setValue(int(lv[key]) if is_int else float(lv[key]))
            if suf:
                sb.setSuffix(suf)
            sb.setToolTip(tip)
            self._lv_spins[key] = sb
            lvform.addRow(label, sb)
        root.addWidget(gb_lv)

        # ---- LV wall-thickness colour bands (壁厚) -------------------------
        gb_wall = QGroupBox(t("LV wall thickness colour (壁厚)"))
        wform = QFormLayout(gb_wall)
        wnote = QLabel(t(
            "Colour thresholds in mm for the 壁厚3D / 壁厚短軸 heat maps, thin→"
            "thick = red→green. Enter any number of ASCENDING values separated "
            "by commas (N values = N+1 colour bands). Default 5, 7, 9."))
        wnote.setWordWrap(True)
        wform.addRow(wnote)
        thr = (lv_wall or {}).get("thresholds") or [5.0, 7.0, 9.0]
        self._wall_edit = QLineEdit(", ".join(f"{float(x):g}" for x in thr))
        self._wall_edit.setToolTip(t("e.g. '4, 6, 8, 10' for 5 bands, or '6' "
                                     "for a single thin/thick split at 6 mm"))
        wform.addRow(t("しきい値 (mm)"), self._wall_edit)
        root.addWidget(gb_wall)

        # ---- Coronary Tree / Territory ------------------------------------
        cp = coronary or settings.load_coronary_params()
        gb_coro = QGroupBox(t("Coronary Tree / Territory"))
        coform = QFormLayout(gb_coro)
        cnote2 = QLabel(t(
            "冠動脈の見た目を設定します（アプリ修正不要）。太さ・LM比・境界(ハロー)の"
            "色/透過度・各系統/ターゲットの色。VRの太さ/ハローはWindowsのみ有効。"))
        cnote2.setWordWrap(True)
        coform.addRow(cnote2)
        # 太さ (直径)
        self._coro_dia = QDoubleSpinBox()
        self._coro_dia.setRange(settings.CORONARY_TUBE_MIN,
                                settings.CORONARY_TUBE_MAX)
        self._coro_dia.setSingleStep(settings.CORONARY_TUBE_STEP)
        self._coro_dia.setDecimals(1)
        self._coro_dia.setSuffix(" mm")
        self._coro_dia.setValue(float(cp.get("tube_diameter_mm", 1.0)))
        coform.addRow(t("冠動脈の太さ（直径）"), self._coro_dia)
        # LM 太さ比
        self._coro_lm = QDoubleSpinBox()
        self._coro_lm.setRange(settings.CORONARY_LM_MIN, settings.CORONARY_LM_MAX)
        self._coro_lm.setSingleStep(settings.CORONARY_LM_STEP)
        self._coro_lm.setDecimals(1)
        self._coro_lm.setSuffix(" ×")
        self._coro_lm.setValue(float(cp.get("lm_ratio", 1.5)))
        coform.addRow(t("LMの太さ比（他の枝に対して）"), self._coro_lm)
        # 境界(ハロー)色 + 透過度
        halo_row = QHBoxLayout()
        self._coro_halo_btn = _ColorButton(
            cp.get("halo_color", "#000000"), t("冠動脈境界(ハロー)の色"))
        halo_row.addWidget(self._coro_halo_btn)
        halo_row.addWidget(QLabel(t("透過度")))
        self._coro_halo_op = QComboBox()
        for pct in range(10, 91, 10):
            self._coro_halo_op.addItem(f"{pct}%", pct / 100.0)
        cur_op = int(round(float(cp.get("halo_opacity", 0.5)) * 10)) * 10
        idx = max(0, min(8, (cur_op - 10) // 10))
        self._coro_halo_op.setCurrentIndex(idx)
        halo_row.addWidget(self._coro_halo_op)
        halo_row.addStretch(1)
        _halo_w = QWidget()
        _halo_w.setLayout(halo_row)
        coform.addRow(t("冠動脈境界（ハロー）"), _halo_w)
        # 領域色: LM / LAD / LCX / RCA + Target1..6
        lc = cp.get("line_colors") or {}
        grid = QGridLayout()
        self._coro_line_btns: dict[str, _ColorButton] = {}
        sys_defaults = settings.CORONARY_PARAMS_DEFAULT["line_colors"]
        for i, role in enumerate(("LM", "LAD", "LCX", "RCA")):
            grid.addWidget(QLabel(role), 0, i * 2)
            b = _ColorButton(lc.get(role) or sys_defaults[role],
                             t("{r} の色", r=role))
            self._coro_line_btns[role] = b
            grid.addWidget(b, 0, i * 2 + 1)
        tcs = cp.get("target_colors") \
            or settings.CORONARY_PARAMS_DEFAULT["target_colors"]
        self._coro_target_btns: list[_ColorButton] = []
        for i in range(6):
            r, cstart = divmod(i, 3)
            grid.addWidget(QLabel(f"Target{i + 1}"), 1 + r, cstart * 2)
            hexv = tcs[i] if i < len(tcs) \
                else settings.CORONARY_PARAMS_DEFAULT["target_colors"][i]
            b = _ColorButton(hexv, t("Target{n} の色", n=i + 1))
            self._coro_target_btns.append(b)
            grid.addWidget(b, 1 + r, cstart * 2 + 1)
        _grid_w = QWidget()
        _grid_w.setLayout(grid)
        coform.addRow(t("領域色（系統 / ターゲット）"), _grid_w)
        root.addWidget(gb_coro)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    def _on_ct_count_changed(self, val: int) -> None:
        """Confirm before enabling multiple live CT panes — the memory/GPU load
        can force-quit the app on some machines. OK enables it; Cancel snaps the
        count back to 1."""
        if val < 2 or self._ct_multi_ack:
            return
        ans = QMessageBox.warning(
            self, t("CT panes"),
            t("Allowing multiple CT panes to display at once may force-quit the "
              "app depending on your PC's specs and state. If a problem occurs, "
              "set the CT display count back to 1."),
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if ans == QMessageBox.StandardButton.Ok:
            self._ct_multi_ack = True
        else:
            sb = self._spins["CT"]
            sb.blockSignals(True)
            sb.setValue(1)
            sb.blockSignals(False)

    def _open_color(self) -> None:
        # Pass THIS dialog as the parent so the colour editor opens modal ON TOP
        # of Settings (operable), and Settings resumes once it's closed.
        if callable(self._on_ct_color):
            self._on_ct_color(self)

    def _open_advanced(self) -> None:
        if callable(self._on_advanced):
            self._on_advanced()

    def caps(self) -> dict:
        """Chosen live-pane caps as {"CT": n, "XA": m}."""
        return {k: sb.value() for k, sb in self._spins.items()}

    def quality(self) -> dict:
        """Chosen image-quality prefs: the angio {key: bool} toggles plus the
        Mac CT quality mode ('high' | 'adaptive' | 'low')."""
        out = {k: cb.isChecked() for k, cb in self._q_boxes.items()}
        for val, rb in self._ctq_radios.items():
            if rb.isChecked():
                out["ct_quality_mode"] = val
                break
        return out

    def lv_endo(self) -> dict:
        """Chosen advanced Auto-Endo params (min_chord_mm / n_meridians /
        grid_mm / step_mm)."""
        out = {}
        for k, sb in self._lv_spins.items():
            v = sb.value()
            out[k] = int(v) if isinstance(sb, QSpinBox) else float(v)
        return out

    def coronary(self) -> dict:
        """Chosen Coronary Tree / Territory appearance params (sanitising/clamping
        happens in settings.save_coronary_params)."""
        return {
            "tube_diameter_mm": float(self._coro_dia.value()),
            "lm_ratio": float(self._coro_lm.value()),
            "halo_color": self._coro_halo_btn.hex,
            "halo_opacity": float(self._coro_halo_op.currentData()),
            "line_colors": {r: b.hex for r, b in self._coro_line_btns.items()},
            "target_colors": [b.hex for b in self._coro_target_btns],
        }

    def lv_wall(self) -> dict:
        """Chosen wall-thickness colour thresholds (mm), parsed from the text —
        any count; sanitising (sort / dedupe / clamp) happens in settings."""
        thr = []
        for tok in self._wall_edit.text().replace("、", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                thr.append(float(tok))
            except ValueError:
                continue
        return {"thresholds": thr}
