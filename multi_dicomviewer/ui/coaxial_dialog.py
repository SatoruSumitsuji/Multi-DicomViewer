"""Coaxial-evaluation result dialog.

Shows the GC-to-vessel angles computed by core.coaxial.compute_coaxial_angles
plus any warnings (too few views, views too close together, no GC, ...).
Pure presentation — all the geometry happens in core.coaxial.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from multi_dicomviewer.i18n import t


def _angle_color(deg: float) -> str:
    """Green when near-coaxial, amber mid, red when far off-axis. Purely a
    visual cue — the cut-offs are not a clinical threshold. Dark, saturated
    tones so they stay legible on a light (white / light-gray) background."""
    if deg <= 20.0:
        return "#2b8a3e"
    if deg <= 45.0:
        return "#e8590c"
    return "#c92a2a"


def _fmt_vec(v) -> str:
    """A 3-D unit vector as compact signed LPS components."""
    return "(" + ", ".join(f"{float(x):+.2f}" for x in v) + ")"


def _fmt_views(views) -> str:
    """The C-arm angles used, as 'β/α°' pairs."""
    return ", ".join(f"{b:g}/{a:g}°" for (b, a) in views)


def _fmt_proj(beta: float, alpha: float) -> str:
    """A C-arm angulation in clinical notation: primary as LAO/RAO, secondary
    as CRA/CAU. e.g. (30, -20) -> 'LAO 30 / CAU 20'."""
    lr = f"LAO {beta:.0f}" if beta >= 0 else f"RAO {-beta:.0f}"
    cc = f"CRA {alpha:.0f}" if alpha >= 0 else f"CAU {-alpha:.0f}"
    return f"{lr} / {cc}"


def _subset_text(label: str, det: dict) -> str | None:
    """Per-image-combination angle table. Numbers each image by its C-arm
    angulation (LAO/RAO · CRA/CAU), then lists the GC↔vessel angle for every
    combination of 2+ images, grouped by combination size. Combinations whose
    views are too parallel to reconstruct show "N/A". The all-image row is the
    headline (final) angle."""
    sub = det.get("subsets") if det else None
    if not sub or not sub.get("views") or not sub.get("combos"):
        return None
    views = sub["views"]
    lines = [t("Images:")]
    for i, (b, a) in enumerate(views, 1):
        lines.append(f"  {i}) {_fmt_proj(b, a)}")
    lines.append("")
    lines.append(
        t("GC ↔ {label} angle by image combination:").format(label=label))
    last_size = None
    row: list[str] = []
    for c in sub["combos"]:
        if c["size"] != last_size and row:
            lines.append("  " + "    ".join(row))
            row = []
        last_size = c["size"]
        key = "-".join(str(i) for i in c["idx"])
        val = t("N/A") if c.get("theta") is None else f"{c['theta']:.0f}°"
        row.append(f"{key}: {val}")
    if row:
        lines.append("  " + "    ".join(row))
    n = len(views)
    if n >= 2:
        final_key = "-".join(str(i) for i in range(1, n + 1))
        lines.append(
            t("  ({key} = all images = final result)").format(key=final_key))
    return "\n".join(lines)


def _fmt_kappa(k) -> str:
    """Conditioning number for display: '∞' when the views collapse to one
    direction, '—' when unavailable, else 1-decimal."""
    if k is None:
        return "—"
    if k == float("inf"):
        return "∞"
    return f"{k:.1f}"


def _confidence_summary(det: dict) -> dict | None:
    """Plain-language reliability verdict for one GC-vessel angle, so the
    reader doesn't have to interpret spread / kappa / leave-one-out by hand.

    Combines two cues already computed in core.coaxial:
      * spread  — how much the angle changes between view-pairs (deg). The
                  practical "how much could this be off" number.
      * kappa   — geometric conditioning; large means the views are too
                  alike for a stable reconstruction.
    Plus whether there is any cross-check at all (>= 2 view-pairs).

    Returns {level, color, headline, detail, band_deg} or None when there is
    nothing to judge from. ``band_deg`` is a rough +/- uncertainty (half the
    spread), or None when no spread is available (e.g. only one view-pair).
    """
    conf = det.get("confidence") or {}
    n_pairs = len(conf.get("pairwise") or [])
    spread = conf.get("pairwise_spread")   # only set when n_pairs >= 2
    kappas = [k for k in (conf.get("cond_gc"), conf.get("cond_vessel"))
              if k is not None]
    kappa = max(kappas) if kappas else None
    have_crosscheck = n_pairs >= 2

    if spread is None and kappa is None and n_pairs == 0:
        return None

    # Decision tree, in plain terms:
    #  - With >= 2 reconstruction pairs we can CROSS-CHECK the answer, so the
    #    measured spread between pairs is the trustworthy evidence and drives
    #    the verdict (a tiny spread can be called "高い").
    #  - With only 1 pair (exactly 2 views) there is NOTHING to cross-check
    #    against, so we can never say "高い"; cap at "中程度" and let a poor
    #    geometry (large kappa) drop it to "低い".
    def _score_spread(s):
        if s <= 5.0:
            return 0
        if s <= 15.0:
            return 1
        return 2

    if have_crosscheck and spread is not None:
        level = _score_spread(spread)
        # A truly collapsed geometry (kappa = inf, or very large with the
        # spread not already tiny) caps the claim at "中程度". A near-zero
        # measured spread is strong direct evidence and is trusted as-is.
        if spread > 2.0 and (kappa == float("inf")
                             or (kappa is not None and kappa > 60.0)):
            level = max(level, 1)
        band = round(spread / 2.0)
    else:
        # One pair (or none): no cross-check possible.
        level = 1
        if kappa == float("inf") or (kappa is not None and kappa > 10.0):
            level = 2
        band = None

    band_txt = f"±{band}°" if band is not None else t("unknown")

    if level == 0:
        return {
            "level": 0, "color": "#2b8a3e",
            "headline": t("Confidence: High  ✓"),
            "band_deg": band,
            "detail": (
                t("The estimates from the different projection angles agree "
                  "well (approximate angular uncertainty: {band_txt}).")
                .format(band_txt=band_txt)
            ),
        }
    if level == 1:
        if not have_crosscheck:
            detail = t(
                "Computed from only two projections, so the result could not "
                "be cross-checked against another direction. The estimate "
                "becomes unstable when the projection directions are similar. "
                "If possible, add one more view at a well-separated angle."
            )
        else:
            detail = t(
                "The estimate varies by about {band_txt} across projection "
                "angles. Strongly foreshortened views or vessel curvature may "
                "be contributing. Treat the value as a reference only."
            ).format(band_txt=band_txt)
        return {
            "level": 1, "color": "#e8590c",
            "headline": t("Confidence: Moderate"),
            "band_deg": band, "detail": detail,
        }
    return {
        "level": 2, "color": "#c92a2a",
        "headline": t("Confidence: Low  ⚠"),
        "band_deg": band,
        "detail": (
            t("The estimate varies widely (about {band_txt}) across projection "
              "angles. This value may be misleading. We recommend re-deriving "
              "it using views where the vessel appears long and straight and "
              "the angles are well separated from one another.")
            .format(band_txt=band_txt)
        ),
    }


def _steps_text(label: str, det: dict, gc_det: dict) -> str:
    """A short 3-line trace explaining how the GC↔vessel angle was found:
    reconstruct each vessel's 3-D direction, then take the angle between
    them. Patient LPS: x=Left, y=Posterior, z=Head."""
    lines = []
    if gc_det:
        lines.append(
            f"GC dir from views {_fmt_views(gc_det['views'])}  →  "
            f"{_fmt_vec(gc_det['direction'])}"
        )
    lines.append(
        f"{label} dir from views {_fmt_views(det['views'])}  →  "
        f"{_fmt_vec(det['direction'])}"
    )
    cos = det.get("cos_to_gc")
    if cos is not None:
        lines.append(
            f"θ = arccos | GC · {label} | = arccos({cos:.3f})"
        )
    # Per-view 2-D GC↔vessel angles (foreshortened) as a confidence cue:
    # views that agree → reconstruction well constrained; wide spread →
    # treat θ with more caution. NOT used to compute θ.
    per_view = det.get("per_view_2d") or []
    if per_view:
        shown = "  ".join(
            f"{p['beta']:g}/{p['alpha']:g}°:{p['angle_2d']:.0f}°"
            for p in per_view
        )
        lines.append(t("per-view 2-D ∠ (as seen): {shown}").format(shown=shown))
        spread = det.get("spread_2d")
        if spread is not None:
            lines.append(
                t("  spread {spread:.0f}°  (small = views agree)")
                .format(spread=spread)
            )

    # Reconstruction-stability cues (confidence only — θ above is unchanged):
    #  • pairwise 2-view θ: what each PAIR of views alone would have given,
    #    so a 2-view-vs-3-view discrepancy is visible directly.
    #  • leave-one-out: how far θ moves when any single view is dropped
    #    (shown only with ≥4 views; at 3 views it equals the pairwise set).
    #  • conditioning κ: geometric diversity of the views (≈1 ideal, large =
    #    views too alike / weakly constrained — catches same-tilt pairs the
    #    separation check misses).
    conf = det.get("confidence") or {}
    pw = conf.get("pairwise") or []
    if len(pw) >= 2:
        shown = "  ".join(
            f"{a[0]:g}/{a[1]:g}+{b[0]:g}/{b[1]:g}:{p['theta']:.0f}°"
            for p in pw for (a, b) in [p["views"]]
        )
        lines.append(t("2-view θ per pair: {shown}").format(shown=shown))
        ps = conf.get("pairwise_spread")
        if ps is not None:
            lines.append(
                t("  pair spread {ps:.0f}°  (small = robust)").format(ps=ps)
            )
    loo = conf.get("leave_one_out") or []
    shared = conf.get("shared_views") or []
    # Only worth showing when it differs from the pairwise set (>=4 views).
    if loo and len(shared) >= 4 and conf.get("loo_swing") is not None:
        lines.append(
            t("leave-one-out swing {swing:.0f}°  "
              "(θ change when any one view is dropped)")
            .format(swing=conf['loo_swing'])
        )
    cg, cv = conf.get("cond_gc"), conf.get("cond_vessel")
    if cg is not None or cv is not None:
        lines.append(
            t("conditioning κ: {cg} (GC) / "
              "{cv} ({label})  — ≈1 ideal, large = views alike")
            .format(cg=_fmt_kappa(cg), cv=_fmt_kappa(cv), label=label)
        )
    # One-line glossary so the technical cues above are self-explanatory.
    if per_view or pw or cg is not None or cv is not None:
        lines.append(
            t("  * spread = disagreement of the estimate across projection "
              "angles (smaller is more reliable) / κ = geometric diversity of "
              "the projection directions (closer to 1 is better; large means "
              "the directions are too similar)")
        )
    return "\n".join(lines)


class CoaxialResultDialog(QDialog):
    def __init__(self, result: dict, view_counts: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Coaxial Evaluation"))
        self.setMinimumWidth(420)
        # Pin a light background so the dark text below is always legible,
        # regardless of the OS/Qt palette (the previous light-on-dark colours
        # were unreadable on the default light-gray dialog background).
        self.setStyleSheet("QDialog { background:#f5f5f5; }")

        col = QVBoxLayout(self)
        col.setContentsMargins(16, 14, 16, 14)
        col.setSpacing(8)

        angles = result.get("angles", {})
        directions = result.get("directions", {})
        warnings = result.get("warnings", [])
        # The "black" technical breakdown (per-image combination table + the
        # 3-D reconstruction steps) is hidden by default and revealed by the
        # "Detail" button; every such widget is collected here.
        self._detail_widgets: list = []

        head = QLabel(t("GC ↔ coronary-proximal angle"))
        head.setStyleSheet("font-size:14pt; font-weight:bold; color:#111111;")
        col.addWidget(head)

        if "GC" in directions:
            sub = QLabel(
                t("Guiding Catheter reconstructed from {n} views. "
                  "0° = perfectly coaxial, 90° = perpendicular.")
                .format(n=view_counts.get('GC', 0))
            )
            sub.setStyleSheet("color:#333333; font-size:10pt;")
            sub.setWordWrap(True)
            col.addWidget(sub)

        if angles:
            # Stable, clinically-grouped order; fall back to whatever is there.
            order = ["LM", "proxLAD", "proxLCX", "proxRCA"]
            for label in order:
                if label not in angles:
                    continue
                deg = angles[label]
                row = QHBoxLayout()
                name = QLabel(f"GC ↔ {label}")
                name.setStyleSheet("font-size:13pt; color:#1a1a1a;")
                val = QLabel(f"{deg:.0f}°")
                val.setStyleSheet(
                    f"font-size:16pt; font-weight:bold; "
                    f"color:{_angle_color(deg)};"
                )
                val.setAlignment(Qt.AlignmentFlag.AlignRight
                                 | Qt.AlignmentFlag.AlignVCenter)
                cnt = QLabel(
                    t("({n} views)").format(n=view_counts.get(label, 0))
                )
                cnt.setStyleSheet("color:#555555; font-size:9pt;")
                row.addWidget(name)
                row.addStretch(1)
                row.addWidget(val)
                row.addWidget(cnt)
                col.addLayout(row)

                det = directions and result.get("details", {}).get(label)

                # Plain-language reliability verdict, right under the angle so
                # the reader sees "how much can I trust this" before the
                # technical spread / kappa lines further down.
                summary = _confidence_summary(det) if det else None
                if summary:
                    band = summary.get("band_deg")
                    band_line = ""
                    if band is not None:
                        band_line = (
                            t("\n    Expected uncertainty: about ±{band}°"
                              " ({deg:.0f}° → roughly "
                              "{lo:.0f}–{hi:.0f}°)")
                            .format(band=band, deg=deg,
                                    lo=max(0, deg - band),
                                    hi=min(90, deg + band))
                        )
                    rel = QLabel(
                        f"{summary['headline']}\n    {summary['detail']}"
                        + band_line
                    )
                    rel.setStyleSheet(
                        f"color:{summary['color']}; font-size:10pt; "
                        f"font-weight:bold; padding:0 0 2px 14px;"
                    )
                    rel.setWordWrap(True)
                    col.addWidget(rel)

                # GC-Target評価最適角度 — the two opposite C-arm angulations
                # whose projection shows this angle without foreshortening.
                opt = det.get("optimal_view") if det else None
                if opt:
                    (b1, a1), (b2, a2) = opt
                    ov = QLabel(
                        t("Optimal GC-target assessment angle:\n"
                          "    {v1}\n"
                          "    {v2}  (opposite direction)\n"
                          "    (this projection shows the target lesion "
                          "without shortening)")
                        .format(v1=_fmt_proj(b1, a1), v2=_fmt_proj(b2, a2))
                    )
                    ov.setStyleSheet(
                        "color:#0b3d91; font-size:10pt; font-weight:bold; "
                        "padding:0 0 2px 14px;"
                    )
                    col.addWidget(ov)
                elif det and "optimal_view" in det:
                    ov = QLabel(
                        t("Optimal GC-target assessment angle: — "
                          "(GC and vessel are nearly coaxial, so it is not "
                          "uniquely defined)")
                    )
                    ov.setStyleSheet(
                        "color:#555555; font-size:9pt; padding:0 0 2px 14px;"
                    )
                    col.addWidget(ov)

                # ---- "Detail" (hidden by default) — the black technical
                # breakdown: the per-image-combination angle table, then the
                # 3-D reconstruction steps.
                combo_txt = _subset_text(label, det) if det else None
                if combo_txt:
                    combo = QLabel(combo_txt)
                    combo.setStyleSheet(
                        "color:#111111; font-size:9pt; "
                        "font-family:'Consolas','Menlo',monospace; "
                        "padding:2px 0 4px 14px;"
                    )
                    combo.setVisible(False)
                    self._detail_widgets.append(combo)
                    col.addWidget(combo)

                if det:
                    steps = QLabel(
                        _steps_text(label, det, result.get("details", {})
                                    .get("GC", {}))
                    )
                    steps.setStyleSheet(
                        "color:#333333; font-size:9pt; "
                        "font-family:'Consolas','Menlo',monospace; "
                        "padding:0 0 4px 14px;"
                    )
                    steps.setWordWrap(True)
                    steps.setVisible(False)
                    self._detail_widgets.append(steps)
                    col.addWidget(steps)
        else:
            none_lbl = QLabel(t("No GC-to-vessel angle could be computed."))
            none_lbl.setStyleSheet("color:#b35900; font-size:11pt;")
            none_lbl.setWordWrap(True)
            col.addWidget(none_lbl)

        if warnings:
            wsep = QLabel(t("⚠  Notes"))
            wsep.setStyleSheet(
                "color:#b35900; font-weight:bold; font-size:11pt; "
                "padding-top:6px;"
            )
            col.addWidget(wsep)
            for w in warnings:
                wl = QLabel("• " + w)
                wl.setStyleSheet("color:#9c4221; font-size:10pt;")
                wl.setWordWrap(True)
                col.addWidget(wl)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        # "Detail" reveals the hidden black breakdown (combination table +
        # reconstruction steps). Absent when there is nothing to reveal.
        if self._detail_widgets:
            self._detail_btn = QPushButton(t("Detail"))
            self._detail_btn.setCheckable(True)
            self._detail_btn.toggled.connect(self._toggle_detail)
            btn_row.addWidget(self._detail_btn)
        ok = QPushButton(t("Close"))
        ok.clicked.connect(self.accept)
        btn_row.addWidget(ok)
        col.addLayout(btn_row)

    def _toggle_detail(self, on: bool) -> None:
        """Show/hide the black technical breakdown; resize the dialog to fit."""
        for w in self._detail_widgets:
            w.setVisible(on)
        self._detail_btn.setText(t("Hide Detail") if on else t("Detail"))
        self.adjustSize()
