"""Shared right-click submenu builders for measurement / comparison styling:
a "Change Color" swatch submenu and a "Change Transparency" 0–100 (step 5)
submenu. Used by every viewer (CT VTK + pygfx, Angio/IVUS canvas) so the
styling menus behave identically everywhere.

No viewer imports here (avoids cycles); the caller passes the colour list.
"""
from __future__ import annotations

from PyQt6.QtGui import QColor, QIcon, QPixmap

from multi_dicomviewer.i18n import t


def add_color_submenu(menu, color_choices):
    """Add a 'Change Color' submenu of swatches. Returns [(QAction, hex)]."""
    sub = menu.addMenu(t("Change Color"))
    actions = []
    for name, hexcol in color_choices:
        a = sub.addAction(name)
        pix = QPixmap(16, 16)
        pix.fill(QColor(hexcol))
        a.setIcon(QIcon(pix))
        actions.append((a, hexcol))
    return actions


def add_transparency_submenu(menu, current=0):
    """Add a 'Change Transparency' submenu (0–100 %, step 5; 0 = opaque). The
    current value is ticked. Returns [(QAction, int_percent)]."""
    sub = menu.addMenu(t("Change Transparency"))
    actions = []
    cur = int(round(current / 5.0)) * 5
    for v in range(0, 101, 5):
        a = sub.addAction(f"{v} %")
        a.setCheckable(True)
        a.setChecked(v == cur)
        actions.append((a, v))
    return actions


def transp_to_alpha(transp) -> int:
    """0–100 transparency % → 0–255 alpha (0 % = 255 opaque)."""
    t = max(0, min(100, int(transp)))
    return int(round((100 - t) / 100.0 * 255))
