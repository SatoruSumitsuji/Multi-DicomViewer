"""MultiReg: file dialogs start in the PARENT of the source DICOM folder, and
Export MP4 can add the pane title (above) and/or the frame info (below) each
image — or neither (images only, as before); the last choice is the default.
Synthetic data only; a fake HOME keeps ui_state.json away from the real one.
"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_home = tempfile.mkdtemp(prefix="mdv_home_")
os.environ["HOME"] = os.environ["USERPROFILE"] = _home
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer import i18n  # noqa: E402
from multi_dicomviewer.core import settings  # noqa: E402
from multi_dicomviewer.ui import coreg_window as cw  # noqa: E402
from multi_dicomviewer.ui.export_dialog import ExportDialog  # noqa: E402
from multi_dicomviewer.viewers.image_canvas import ImageCanvas  # noqa: E402

i18n.set_language("ja")
settings.UI_STATE_PATH = Path(_home) / "ui_state.json"
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


W = cw.CoregWindow

# --- default folder = parent of the series folder ---------------------------
root = tempfile.mkdtemp(prefix="mdv_cr_")
s1 = os.path.join(root, "study", "ser1")
s2 = os.path.join(root, "study", "ser2")
for d in (s1, s2):
    os.makedirs(d)
fake = SimpleNamespace(
    panes=[SimpleNamespace(series=None),
           SimpleNamespace(series=SimpleNamespace(files=[os.path.join(s1, "a.dcm")])),
           SimpleNamespace(series=SimpleNamespace(files=[os.path.join(s2, "b.dcm")]))],
    _active_ivus=2)
study = os.path.normcase(os.path.join(root, "study"))
check("default dir = parent of the master's DICOM folder",
      os.path.normcase(W._default_dir(fake)) == study, W._default_dir(fake))
fake._active_ivus = -1
check("no master → first pane with files", os.path.normcase(
    W._default_dir(fake)) == study)
empty = SimpleNamespace(panes=[SimpleNamespace(series=None)], _active_ivus=-1)
check("no files anywhere → '' (or the last folder)", W._default_dir(empty) == "")
W._remember_dir(empty, os.path.join(s1, "x.cosync.json"))
check("…then the last folder a dialog used", W._default_dir(empty) == s1)
src = (ROOT / "multi_dicomviewer/ui/coreg_window.py").read_text(encoding="utf-8")
check("save / load / MP4 dialogs all start at _default_dir()",
      src.count("self._default_dir(),") == 3)

# --- the dialog: two remembered checkboxes ---------------------------------
dlg = ExportDialog("mp4", 1, show_filename_fields=False,
                   extra_checks=[("title", "T", True), ("frame", "F", False)],
                   extra_title="X")
check("ExportDialog extra checkboxes", dlg.extra_checked()
      == {"title": True, "frame": False})
dlg.close()
plain = ExportDialog("mp4", 1, show_filename_fields=False)
check("no extra_checks → no extra boxes", plain.extra_checked() == {})
plain.close()
check("i18n ja labels", i18n.t("Frame info below each image") == "画像下のフレーム情報")


# --- composition: bands only when asked ------------------------------------
class Pane:
    def __init__(self, title, lbl):
        self.canvas = ImageCanvas()
        self.canvas.resize(300, 300)
        self.canvas.set_frame(np.full((64, 64), 128, np.uint8))
        self.title, self.lbl = QLabel(title), QLabel(lbl)


host = SimpleNamespace(panes=[Pane("IVUS L — Ser 3", "F 5 / 900"),
                              Pane("XA — Ser 7", "F 1 / 60")],
                       _qimage_to_rgb=W._qimage_to_rgb,
                       _draw_band_text=W._draw_band_text)
for p in host.panes:
    p.canvas.show()
app.processEvents()
cell = 480
plain_img = W._compose_coreg_frame(host, cell, 2)
both = W._compose_coreg_frame(host, cell, 2, title=True, frame_info=True)
top_only = W._compose_coreg_frame(host, cell, 2, title=True)
band = max(18, cell // 22) & ~1
check("images only: unchanged size", plain_img.shape[0] == cell)
check("title + frame: one band above and one below",
      both.shape[0] == cell + 2 * band and top_only.shape[0] == cell + band,
      str((plain_img.shape, both.shape, top_only.shape)))
check("even dimensions (yuv420p)", both.shape[0] % 2 == 0
      and both.shape[1] % 2 == 0 and top_only.shape[0] % 2 == 0)
check("title band has white text", both[:band].max() > 200
      and top_only[:band].max() > 200)
check("frame band has white text", both[-band:].max() > 200)
check("no text when not asked (band area absent)", plain_img[:4].max() < 200)

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
