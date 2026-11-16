"""Settings ▸ Coronary Tree / Territory ▸ Territory colours: each name (LM,
LAD, LCX, RCA, Target1..6) stays right next to ITS swatch however wide the
dialog is — never drifting toward the next name (2026-11-16).

    python tools/test_settings_territory_colors.py
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

from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings as st  # noqa: E402

assert str(st.SETTINGS_DIR).startswith(HOME), st.SETTINGS_DIR
from multi_dicomviewer.ui.settings_dialog import SettingsDialog  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


dlg = SettingsDialog({"CT": 4, "XA": 4}, st.load_display_quality(), None,
                     coronary=st.load_coronary_params())
pairs = [(r, b) for r, b in dlg._coro_line_btns.items()] + \
        [(f"Target{i + 1}", b) for i, b in enumerate(dlg._coro_target_btns)]


def label_for(btn, name):
    for lab in btn.parentWidget().findChildren(QLabel):
        if lab.text() == name:
            return lab
    return None


for width in (700, 1600):
    dlg.resize(width, 800)
    dlg.show()
    app.processEvents()
    gaps, worst = [], 0
    for name, btn in pairs:
        lab = label_for(btn, name)
        if lab is None:
            gaps.append(None)
            continue
        lr = lab.mapTo(dlg, lab.rect().topRight()).x()
        bl = btn.mapTo(dlg, btn.rect().topLeft()).x()
        gaps.append(bl - lr)
    check(f"w={width}: every name shares a cell with its swatch",
          None not in gaps)
    ok = None not in gaps and all(0 <= g <= 12 for g in gaps)
    check(f"w={width}: swatch right next to its name (≤12 px)", ok, str(gaps))
    # the swatch is nearer its own name than the next name on the row
    row = [label_for(b, n) for n, b in pairs[:4]]
    btns = [b for _n, b in pairs[:4]]
    near = all(
        btns[i].mapTo(dlg, btns[i].rect().topLeft()).x()
        - row[i].mapTo(dlg, row[i].rect().topRight()).x()
        < row[i + 1].mapTo(dlg, row[i + 1].rect().topLeft()).x()
        - btns[i].mapTo(dlg, btns[i].rect().topRight()).x()
        for i in range(3))
    check(f"w={width}: swatch closer to its own name than to the next", near)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
