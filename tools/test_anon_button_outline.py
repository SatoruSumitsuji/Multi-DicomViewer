# -*- coding: utf-8 -*-
"""Top-row 「匿名化」 button (user 2026-11-28): framed while NOT anonymized,
the native checked look (no frame) once anonymized.

    python tools/test_anon_button_outline.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtGui import QColor  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.ui.study_browser import OutlineWhenOffButton  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


def edge_hits(b):
    img = b.grab().toImage()
    want = QColor(OutlineWhenOffButton.OUTLINE)
    w, h = img.width(), img.height()
    pts = [(x, 1) for x in range(4, w - 4, 3)] + \
        [(1, y) for y in range(4, h - 4, 3)]
    n = 0
    for x, y in pts:
        c = img.pixelColor(x, y)
        if abs(c.red() - want.red()) < 40 and abs(c.green() - want.green()) < 40 \
                and abs(c.blue() - want.blue()) < 40:
            n += 1
    return n / max(1, len(pts))


b = OutlineWhenOffButton("Anonymous")
b.setCheckable(True)
b.resize(120, 28)
check("not anonymized → framed", edge_hits(b) > 0.8, edge_hits(b))
b.setChecked(True)
check("anonymized → no frame (as before)", edge_hits(b) < 0.2, edge_hits(b))
b.setChecked(False)
check("back to not anonymized → framed again", edge_hits(b) > 0.8)
mw = (ROOT / "multi_dicomviewer/ui/main_window.py").read_text(encoding="utf-8")
check("the top-row anon button uses it",
      'self._anon_btn = OutlineWhenOffButton(t("Anonymous"))' in mw)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(0 if not fails else 1)
