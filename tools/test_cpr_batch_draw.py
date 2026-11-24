# -*- coding: utf-8 -*-
"""「一括描出」 (user 2026-11-25): every unconnected branch line of the tree
(a few points at a branch's distal end) is traced PROXIMALLY to the tree and
connected there — FinTrace ◯ when it reached its branch point, × when it
stopped (the next line still runs). The AutoTrace column is gone; the button
sits between Draw and Auto on both viewers. Synthetic tubes, fake HOME,
dialogs patched.

    python tools/test_cpr_batch_draw.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(
    lambda *a, **k: QMessageBox.StandardButton.Yes)
from multi_dicomviewer.core.centerline import CenterLine  # noqa: E402
from multi_dicomviewer.i18n import t  # noqa: E402
from multi_dicomviewer.ui import coronary_tree_window as ctw  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        fails.append(name)


# ---- synthetic CT: LAD along x, branch 1 reaches it, branch 2 occluded ------
h = 0.4
n = 150
z, y, x = (np.mgrid[0:n // 3, 0:n, 0:n].astype(np.float32) * h)
dims = (h, h, h)
ZC = 9.0
vol = np.full(x.shape, -60.0, np.float32)


def seg(a, b, r, hu):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    tt = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]
                  + (z - a[2]) * ab[2]) / float(ab @ ab), 0, 1)
    d2 = ((x - a[0] - tt * ab[0]) ** 2 + (y - a[1] - tt * ab[1]) ** 2
          + (z - a[2] - tt * ab[2]) ** 2)
    rr = r[0] + (r[1] - r[0]) * tt if isinstance(r, tuple) else r
    vol[d2 <= rr * rr] = hu


LAD = np.c_[np.linspace(4, 56, 18), np.full(18, 10.0), np.full(18, ZC)]
seg(LAD[0], LAD[-1], 1.8, 420.0)
J1, E1 = np.array([18.0, 10.0, ZC]), np.array([34.0, 52.0, ZC])
seg(J1, E1, (1.4, 0.8), 400.0)
J2, E2 = np.array([46.0, 10.0, ZC]), np.array([50.0, 54.0, ZC])
u2 = (E2 - J2) / np.linalg.norm(E2 - J2)
M2 = J2 + 0.5 * (E2 - J2)
seg(J2, M2 - 3.0 * u2, 1.3, 400.0)
seg(M2 + 3.0 * u2, E2, (1.1, 0.8), 400.0)            # occluded midway
vol = ndimage.gaussian_filter(vol, 0.5 / h)
vol += np.random.default_rng(3).normal(0, 15, vol.shape).astype(np.float32)


def tip(J, E):
    u = (E - J) / np.linalg.norm(E - J)
    return np.array([E - 7.0 * u, E - 3.5 * u, E - 0.5 * u])


D = tempfile.mkdtemp(dir=HOME)
UID = "bd-ct"
w = ctw.CoronaryTreeWindow(None)
w._use_uid(UID)
b = w._bundle()
tree = b["tree"]


def add(vid, name, ctrl, role="branch"):
    p = os.path.join(D, f"{name}.cpr.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"format": "MDV-CPR", "version": 1, "type": "cpr",
                   "series": {"series_uid": UID}, "src_dir": "",
                   "ctrl": np.asarray(ctrl).tolist(),
                   "state": {"idx": 0}}, f)
    pts = CenterLine.from_points(ctrl, step_mm=0.5).points
    if role == "branch":
        tree.add_vessel(vid, name, "branch", pts, ctrl=np.asarray(ctrl))
    else:
        tree.add_root(vid, name, role, pts)
        tree.vessels[vid].ctrl = np.asarray(ctrl)
    b["paths"][vid] = p
    return p


add("lad", "LAD", LAD, role="LAD")
p1 = add("d1", "D1", tip(J1, E1)[::-1])          # drawn distal → proximal
p2 = add("d2", "D2", tip(J2, E2))
w._populate()

check("targets = the unconnected branch lines",
      sorted(w.batch_targets_to_tree()) == ["d1", "d2"])
seen = []
out = w.batch_extend(vol, dims, progress=lambda *a: seen.append(a))
d1, d2 = tree.vessels["d1"], tree.vessels["d2"]
check("reached: connected to LAD near its branch point, FinTrace ◯",
      d1.parent == "lad" and d1.fin_traced
      and abs(tree.vessels["lad"].points[d1.junction][0] - J1[0]) < 4.0,
      (d1.parent, d1.junction, d1.fin_traced))
check("reached: the line runs proximal → distal (start at the branch)",
      np.linalg.norm(d1.points[0] - J1) < 5.0
      and np.linalg.norm(d1.points[-1] - E1) < 2.0,
      (d1.points[0], d1.points[-1]))
data = json.load(open(p1, encoding="utf-8"))
check("reached: its .cpr.json rewritten with the whole line",
      len(data["ctrl"]) > 8 and np.allclose(data["ctrl"], d1.ctrl))
check("occluded: stopped, left unconnected, FinTrace ×",
      d2.parent is None and not d2.fin_traced
      and np.linalg.norm(d2.points[0] - M2) < 6.0, (d2.points[0],))
check("progress: 'line k of n' with the line's name",
      any(a[0] == "batch" and a[3] == 1 and a[4] == 2 for a in seen)
      and any(a[0] == "batch" and a[3] == 2 for a in seen), seen[:2])
txt = ctw.CoronaryTreeWindow.batch_extend_text(out)
check("summary names ◯ and × lines",
      "◯ " in txt and "× " in txt and "D1" in txt and "D2" in txt, txt)


def fin_cell(vid):
    it = w._item_for_vid(vid)
    return it.text(ctw._C_FIN) if it is not None else None


check("FinTrace column shows ◯ / ×",
      fin_cell("d1") == ctw._MARK_YES and fin_cell("d2") == ctw._MARK_NO,
      (fin_cell("d1"), fin_cell("d2")))
hdr = [w._tree_w.headerItem().text(c) for c in range(ctw._N_COLS)]
check("AutoTrace column gone; 分岐 kept",
      ctw._N_COLS == 6 and t("AutoTrace") not in hdr and t("分岐") in hdr,
      hdr)

# a second run: only the still-unconnected line is tried again
seen.clear()
w.batch_extend(vol, dims, progress=lambda *a: seen.append(a))
check("run again: only the × line is retried",
      {a[5] for a in seen if a[0] == "batch"} <= {"D2"}
      and tree.vessels["d1"].parent == "lad")

# cancel at once: nothing traced, reported
tree.vessels["d2"].parent = None
out = w.batch_extend(vol, dims, cancel=lambda: True)
check("stop: cancelled, reported", out["cancelled"]
      and t("（停止しました。残りの線は未処理です）")
      in ctw.CoronaryTreeWindow.batch_extend_text(out))

# ---- the button between Draw and Auto, on both viewers ---------------------
for f in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    src = (ROOT / "multi_dicomviewer" / "viewers" / f).read_text(
        encoding="utf-8")
    i_d = src.find("row.addWidget(self._cpr_draw_btn)")
    i_b = src.find("row.addWidget(self._cpr_batch_btn)")
    i_a = src.find("row.addWidget(self._cpr_auto_btn)")
    check(f"{f}: 一括描出 between Draw and Auto",
          0 < i_d < i_b < i_a and 'FitButton(t("一括描出"))' in src)
mw = (ROOT / "multi_dicomviewer" / "ui" / "main_window.py").read_text(
    encoding="utf-8")
check("shell: coronary_batch_extend runs it under the stop button",
      "def coronary_batch_extend" in mw and "w.batch_extend(" in mw)
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
os._exit(1 if fails else 0)
