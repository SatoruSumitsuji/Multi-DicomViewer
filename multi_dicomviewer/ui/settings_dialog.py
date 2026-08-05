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
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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
                 on_advanced=None, parent=None):
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

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

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
