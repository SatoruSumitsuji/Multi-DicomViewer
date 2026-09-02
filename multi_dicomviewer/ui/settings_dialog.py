"""Unified Settings popup (top-bar "Settings" button).

Gathers the app-wide display preferences in one place:

  * Display count — how many CT / angio panes stay fully loaded before older
    ones become memory-saving stills (see settings.load_live_caps).
  * Angio image quality — S-Cine / S-Zoom / Denoise (settings.display_quality).
  * CT colour — a button that opens the HU colour-map editor for the active
    CT pane (that editor applies live and is per-viewer).
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from multi_dicomviewer.core import image_quality, settings
from multi_dicomviewer.i18n import t


class SettingsDialog(QDialog):
    """App-wide display settings. *caps* is {"CT": n, "XA": m}; *quality* is the
    display-quality dict; *on_ct_color* opens the CT colour-map editor."""

    def __init__(self, caps: dict, quality: dict, on_ct_color,
                 on_advanced=None, parent=None, lv_endo=None):
        super().__init__(parent)
        self.setWindowTitle(t("Settings"))
        self._on_ct_color = on_ct_color
        self._on_advanced = on_advanced
        root = QVBoxLayout(self)

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

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

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
