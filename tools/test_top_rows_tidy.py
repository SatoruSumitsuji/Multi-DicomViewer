# -*- coding: utf-8 -*-
"""Top rows tidied (user 2026-11-22):
- Image top row: one pane-buttons toggle whose text says what a click does
  (「ペイン内ボタンを隠す」 / 「ペイン内ボタンを表示」, was Hide / Show
  Buttons); the left-most button reads 「◀ 情報タブを隠す」 / 「情報タブを表示 ▶」, 「✕ すべての画像をクリア」, and 匿名化 moved here between that
  toggle and DICOM情報.
- Studies frame row: ツリー / サムネイル / 「✕ 全症例・検査削除」 only
  (匿名化, DICOM情報, ◀ / ▶ gone) + the 🖼 thumbnail-size slider
  (hover: サムネイルアイコンのサイズ変更).

    python tools/test_top_rows_tidy.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer import i18n  # noqa: E402
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
settings.save_language("ja")    # MainWindow reads it at start-up
i18n.set_language("ja")
from multi_dicomviewer.ui.main_window import MainWindow  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


w = MainWindow()


def row_widgets(widget):
    lay = widget.layout()
    out = []
    for i in range(lay.count()):
        it = lay.itemAt(i)
        if it.widget() is not None:
            out.append(it.widget())
    return out


bar = w._hide_btns_btn.parentWidget()
ws = row_widgets(bar)
check("no Show Buttons button any more", not hasattr(w, "_show_btns_btn"))
check("toggle text while shown", w._hide_btns_btn._full_text == "ペイン内ボタンを隠す",
      w._hide_btns_btn._full_text)
check("info-tabs button text", w._info_btn._full_text in (
    "◀ 情報タブを隠す", "情報タブを表示 ▶"), w._info_btn._full_text)
check("Clear All text", w._clear_all_btn._full_text == "✕ すべての画像をクリア",
      w._clear_all_btn._full_text)
i_h, i_a, i_d = (ws.index(w._hide_btns_btn), ws.index(w._anon_btn),
                 ws.index(w._tags_btn))
check("order: toggle → 匿名化 → DICOM情報", i_h < i_a < i_d and i_a == i_h + 1
      and i_d == i_a + 1, (i_h, i_a, i_d))
check("「ペイン →」 not shown", not w._pane_step_btn.isVisible())
check("no 「ペイン →」 button in the row", w._pane_step_btn not in ws)
check("匿名化 text", w._anon_btn._full_text == "匿名化", w._anon_btn._full_text)

# toggle hides / shows the pane chrome
w._hide_btns_btn.click()
check("click → panes' buttons hidden, text 「ペイン内ボタンを表示」",
      w._chrome_hidden is True
      and w._hide_btns_btn._full_text == "ペイン内ボタンを表示")
w._hide_btns_btn.click()
check("click again → shown, text 「ペイン内ボタンを隠す」",
      w._chrome_hidden is False
      and w._hide_btns_btn._full_text == "ペイン内ボタンを隠す")
w._set_chrome_hidden(True)
check("state set elsewhere is reflected on the toggle",
      w._hide_btns_btn._full_text == "ペイン内ボタンを表示")
w._set_chrome_hidden(False)

# Anonymous: button ↔ menu action ↔ state
w._anon_btn.click()
check("匿名化 ON → anonymized + menu action checked",
      w._anon and w._anon_act.isChecked())
w._anon_act.trigger()
check("menu OFF → button unchecked", not w._anon and not w._anon_btn.isChecked())

# Studies frame row
sp = w.browser
vis = [x for x in row_widgets(sp) if isinstance(x, QWidget)]
top = sp.layout().itemAt(0).layout()
top_ws = [top.itemAt(i).widget() for i in range(top.count())
          if top.itemAt(i).widget() is not None]
check("studies row = ツリー / サムネイル / 削除 + 🖼 size slider",
      top_ws == [sp.btn_info, sp.btn_thumb, sp.btn_delete_all,
                 sp._thumb_size_icon, sp.thumb_size],
      [b.text() for b in top_ws if hasattr(b, "text")])
check("Delete All text", sp.btn_delete_all._full_text == "✕ 全症例・検査削除",
      sp.btn_delete_all._full_text)
w.show()
app.processEvents()
check("old studies-row widgets not shown",
      not any(x.isVisible() for x in (sp.btn_anon, sp.btn_dicom,
                                      sp.btn_dock_narrow, sp.btn_dock_widen)))
check("thumbnail-size slider shown", sp.thumb_size.isVisible()
      and sp._thumb_size_icon.isVisible())
check("hover text on 🖼 and the slider",
      sp.thumb_size.toolTip() == "サムネイルアイコンのサイズ変更"
      and sp._thumb_size_icon.toolTip() == "サムネイルアイコンのサイズ変更",
      sp.thumb_size.toolTip())
w.hide()

# first display: the labels are shown in full and the hover texts exist
# without touching the Studies frame (a label elided once used to stay
# elided / the hover text only appeared after resizing the frame).
from PyQt6.QtCore import QTimer  # noqa: E402
from multi_dicomviewer.ui.study_browser import FitButton  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402
w.resize(1600, 900)
w.show()
for _ in range(5):
    app.processEvents()
# room for all three labels (offscreen fonts are wider than Windows')
need = sum(b.sizeHint().width() for b in (sp.btn_info, sp.btn_thumb,
                                          sp.btn_delete_all)) + 140
w.resizeDocks([w._studies_dock], [need], Qt.Orientation.Horizontal)
for _ in range(5):
    app.processEvents()
for b in (sp.btn_info, sp.btn_thumb, sp.btn_delete_all):
    check(f"first display: {b._full_text} shown in full",
          b.text() == b._full_text, (b.width(), b.text()))
    check(f"first display: {b._full_text} has a hover text", bool(b.toolTip()))
fb = FitButton("📋 ツリーツリー")
fb.setHelpToolTip("help")
fb.resize(30, 24)
fb.show()
app.processEvents()
squeezed = fb.text()
fb.resize(fb.sizeHint().width(), 24)
app.processEvents()
check("an elided label grows back once the button is wide enough",
      squeezed != fb._full_text and fb.text() == fb._full_text,
      (squeezed, fb.text()))
fb.setToolTip("plain tooltip")
fb.resize(fb.sizeHint().width() + 10, 24)
app.processEvents()
check("a plain setToolTip survives a re-fit", fb.toolTip() == "plain tooltip",
      fb.toolTip())
fb.hide()
w.hide()

# live language switch keeps the new labels
w._set_language("en")
check("en: toggle / Clear / Anonymous / Delete labels",
      (w._hide_btns_btn._full_text, w._clear_all_btn._full_text, w._anon_btn._full_text,
       sp.btn_delete_all._full_text) == ("Hide Pane Buttons",
                                     "✕ Clear All Images", "Anonymous",
                                     "✕ Delete All Studies"),
      (w._hide_btns_btn._full_text, w._clear_all_btn._full_text, w._anon_btn._full_text,
       sp.btn_delete_all._full_text))

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.stdout.flush()
os._exit(1 if fails else 0)
