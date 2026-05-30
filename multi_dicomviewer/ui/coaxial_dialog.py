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


def _angle_color(deg: float) -> str:
    """Green when near-coaxial, amber mid, red when far off-axis. Purely a
    visual cue — the cut-offs are not a clinical threshold."""
    if deg <= 20.0:
        return "#37b24d"
    if deg <= 45.0:
        return "#f59f00"
    return "#e03131"


def _fmt_vec(v) -> str:
    """A 3-D unit vector as compact signed LPS components."""
    return "(" + ", ".join(f"{float(x):+.2f}" for x in v) + ")"


def _fmt_views(views) -> str:
    """The C-arm angles used, as 'β/α°' pairs."""
    return ", ".join(f"{b:g}/{a:g}°" for (b, a) in views)


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
    return "\n".join(lines)


class CoaxialResultDialog(QDialog):
    def __init__(self, result: dict, view_counts: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Coaxial Evaluation")
        self.setMinimumWidth(420)

        col = QVBoxLayout(self)
        col.setContentsMargins(16, 14, 16, 14)
        col.setSpacing(8)

        angles = result.get("angles", {})
        directions = result.get("directions", {})
        warnings = result.get("warnings", [])

        head = QLabel("GC ↔ coronary-proximal angle")
        head.setStyleSheet("font-size:14pt; font-weight:bold; color:#dee2e6;")
        col.addWidget(head)

        if "GC" in directions:
            sub = QLabel(
                f"Guiding Catheter reconstructed from "
                f"{view_counts.get('GC', 0)} views. "
                f"0° = perfectly coaxial, 90° = perpendicular."
            )
            sub.setStyleSheet("color:#adb5bd; font-size:10pt;")
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
                name.setStyleSheet("font-size:13pt; color:#e9ecef;")
                val = QLabel(f"{deg:.0f}°")
                val.setStyleSheet(
                    f"font-size:16pt; font-weight:bold; "
                    f"color:{_angle_color(deg)};"
                )
                val.setAlignment(Qt.AlignmentFlag.AlignRight
                                 | Qt.AlignmentFlag.AlignVCenter)
                cnt = QLabel(f"({view_counts.get(label, 0)} views)")
                cnt.setStyleSheet("color:#868e96; font-size:9pt;")
                row.addWidget(name)
                row.addStretch(1)
                row.addWidget(val)
                row.addWidget(cnt)
                col.addLayout(row)

                det = directions and result.get("details", {}).get(label)
                if det:
                    steps = QLabel(
                        _steps_text(label, det, result.get("details", {})
                                    .get("GC", {}))
                    )
                    steps.setStyleSheet(
                        "color:#868e96; font-size:9pt; "
                        "font-family:'Consolas','Menlo',monospace; "
                        "padding:0 0 4px 14px;"
                    )
                    steps.setWordWrap(True)
                    col.addWidget(steps)
        else:
            none_lbl = QLabel("No GC-to-vessel angle could be computed.")
            none_lbl.setStyleSheet("color:#ffa94d; font-size:11pt;")
            none_lbl.setWordWrap(True)
            col.addWidget(none_lbl)

        if warnings:
            wsep = QLabel("⚠  Notes")
            wsep.setStyleSheet(
                "color:#ffd43b; font-weight:bold; font-size:11pt; "
                "padding-top:6px;"
            )
            col.addWidget(wsep)
            for w in warnings:
                wl = QLabel("• " + w)
                wl.setStyleSheet("color:#ffd43b; font-size:10pt;")
                wl.setWordWrap(True)
                col.addWidget(wl)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        ok = QPushButton("Close")
        ok.clicked.connect(self.accept)
        btn_row.addWidget(ok)
        col.addLayout(btn_row)
