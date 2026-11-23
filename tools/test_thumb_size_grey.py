"""Studies panel: the 🖼 thumbnail-size slider and its icon are only usable in
the Thumbnail view — greyed out (disabled, grey icon) in the Tree view
(user 2026-11-23)."""
import os
import sys
import tempfile

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PyQt6.QtCore import QSize  # noqa: E402
from PyQt6.QtGui import QIcon  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.core import settings  # noqa: E402
from multi_dicomviewer.ui.study_browser import StudyPanel  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR

FAILS = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        FAILS.append(name)


def grey(img):
    """True when every visible pixel is (nearly) colourless."""
    for y in range(img.height()):
        for x in range(img.width()):
            c = img.pixelColor(x, y)
            if c.alpha() > 40 and max(c.red(), c.green(), c.blue()) - min(
                    c.red(), c.green(), c.blue()) > 12:
                return False
    return True


p = StudyPanel()
p.resize(420, 300)
p.show()
app.processEvents()
lab = p._thumb_size_icon


def icon_img():
    return lab.pixmap().toImage()


check("start in Tree view: slider disabled", not p.thumb_size.isEnabled())
check("start in Tree view: 🖼 icon grey", grey(icon_img()))
p._show(1)
app.processEvents()
check("Thumbnail view: slider enabled", p.thumb_size.isEnabled())
check("Thumbnail view: 🖼 icon in colour", not grey(icon_img()))
v0 = p.thumb_size.value()
p.btn_info.click()
app.processEvents()
check("back to Tree (button click): slider disabled again",
      not p.thumb_size.isEnabled() and grey(icon_img()))
check("the size value is kept", p.thumb_size.value() == v0)
p.btn_thumb.click()
app.processEvents()
check("Thumbnail (button click): enabled again",
      p.thumb_size.isEnabled() and not grey(icon_img()))
check("the icon keeps its tooltip",
      bool(lab.toolTip()) and lab.toolTip() == p.thumb_size.toolTip())
if os.environ.get("MDV_SHOT"):                 # widget-only grabs for review
    p._show(0)
    app.processEvents()
    p.grab().save(os.path.join(os.environ["MDV_SHOT"], "tree.png"))
    p._show(1)
    app.processEvents()
    p.grab().save(os.path.join(os.environ["MDV_SHOT"], "thumb.png"))
_ = (QIcon, QSize)
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
sys.stdout.flush()
os._exit(1 if FAILS else 0)
