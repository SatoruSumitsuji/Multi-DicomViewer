"""ImageCanvas drag-Zoom (MultiReg panes, XA / IVUS Zoom drag): drag DOWN =
zoom IN (bigger), drag UP = zoom OUT (smaller) — like the CT viewer's Zoom."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.viewers.image_canvas import ImageCanvas  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


def drag(c, dy):
    z0 = c._zoom
    QTest.mousePress(c, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(150, 150))
    for k in range(1, 11):
        QTest.mouseMove(c, QPoint(150, 150 + dy * k // 10))
    QTest.mouseRelease(c, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, QPoint(150, 150 + dy))
    return z0, c._zoom


c = ImageCanvas()
c.resize(300, 300)
c.set_frame(np.full((64, 64), 120, np.uint8))
c.set_drag_mode("zoom")
c.show()
app.processEvents()
z0, z1 = drag(c, 60)
check("drag DOWN zooms IN", z1 > z0, f"{z0:.3f}→{z1:.3f}")
z0, z1 = drag(c, -60)
check("drag UP zooms OUT", z1 < z0, f"{z0:.3f}→{z1:.3f}")

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
