"""MultiReg save / load: new files are named *.multireg.json and store every
pane's frame; loading puts each ANGIO pane back on its frame (the guide's
frame first, else the stored pane frame) and shows the top MultiReg point; older .cosync.json files still load.
Synthetic data only; the file dialog is mocked.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QFileDialog  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer import i18n  # noqa: E402
from multi_dicomviewer.ui import coreg_window as cw  # noqa: E402

i18n.set_language("ja")
W = cw.CoregWindow
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


class Canvas:
    def set_coreg_guide(self, *a):
        pass

    set_coreg_marker = set_coreg_anchors = set_coreg_guide


class Pane:
    def __init__(self, index, ivus, total, cur):
        self.index, self.is_ivus, self.total, self.cur = index, ivus, total, cur
        self.series = SimpleNamespace(label=f"S{index}", files=[])
        self.canvas = Canvas()

    def show_frame(self, f):
        self.cur = max(0, min(int(f), self.total - 1))


class Status:
    def setText(self, s):
        self.text = s


def make():
    o = SimpleNamespace(
        panes=[Pane(0, True, 900, 100), Pane(1, False, 60, 23),
               Pane(2, False, 40, 7)],
        _landmarks=[{"frames": {0: 100}, "rots": {}, "fracs": {1: 0.5},
                     "name": "A"}],
        _guides={1: {"vertices": [(1.0, 2.0), (5.0, 6.0)], "curve": [],
                     "frame": 23}},
        _active_ivus=0, _status=Status(), _selected_lm=-1, _renaming=-1,
        _placing=False, _syncing=False)
    for name in ("_default_dir", "_remember_dir", "_save_coreg", "_load_coreg",
                 "_parse_coreg_json", "_parse_pane_frames", "_apply_loaded",
                 "_parse_multisync_txt", "_goto_landmark"):
        fn = W.__dict__[name]
        setattr(o, name, fn.__func__ if isinstance(fn, staticmethod)
                else fn.__get__(o))
    o._clear_guide = lambda i: o._guides.__setitem__(
        i, {"vertices": [], "curve": [], "frame": None})
    o._refresh_list = o._drive_markers = lambda: None
    o._apply_rotation = lambda pane: None
    return o


tmp = tempfile.mkdtemp(prefix="mdv_mr_")
real_save, real_open = QFileDialog.getSaveFileName, QFileDialog.getOpenFileName
try:
    o = make()
    base = os.path.join(tmp, "case")
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (base, ""))
    o._save_coreg()
    path = base + ".multireg.json"
    check("saved as .multireg.json", os.path.isfile(path), path)
    data = json.load(open(path, encoding="utf-8"))
    check("every pane's frame stored",
          [p["frame"] for p in data["panes"]] == [100, 23, 7], str(data["panes"]))
    check("guide keeps its frame", data["guides"]["1"]["frame"] == 23)

    o2 = make()
    for p in o2.panes:
        p.cur = 0
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (path, ""))
    o2._load_coreg()
    check("load: angio with a guide → the guide's frame", o2.panes[1].cur == 23)
    check("load: angio without a guide → its stored frame", o2.panes[2].cur == 7)
    check("load: the IVUS shows the TOP MultiReg point (not its pane frame)",
          o2.panes[0].cur == 100)
    check("load: that point is shown but not left selected",
          o2._selected_lm == -1)

    # An older .cosync.json: no pane frames, a guide frame only.
    old = os.path.join(tmp, "old.cosync.json")
    d = dict(data)
    d["format"] = "MDV-CoSync"
    d["panes"] = [{k: v for k, v in p.items() if k != "frame"}
                  for p in data["panes"]]
    json.dump(d, open(old, "w", encoding="utf-8"))
    o3 = make()
    for p in o3.panes:
        p.cur = 0
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (old, ""))
    o3._load_coreg()
    check("old .cosync.json still loads", len(o3._landmarks) == 1)
    check("old file: guide's angio frame restored", o3.panes[1].cur == 23)
    check("old file: no frame stored → that angio untouched", o3.panes[2].cur == 0)
finally:
    QFileDialog.getSaveFileName, QFileDialog.getOpenFileName = real_save, real_open

check("dialog filter names *.multireg.json",
      i18n.t("MultiReg file (*.multireg.json)") == "MultiRegファイル (*.multireg.json)")

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
