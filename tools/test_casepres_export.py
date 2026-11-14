"""Case Presentation ▸ 書き出し… (Export): the table as shown → .xlsx / PDF
(no JSON, no images); row 2 without the Reorder buttons; header titles
left-aligned; Show uses the SELECTED pane (main_window.case_redisplay).
"""
import ast
import os
import sys
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import (QApplication, QFileDialog, QHeaderView,  # noqa: E402
                             QMainWindow, QPushButton)

app = QApplication.instance() or QApplication([])

from multi_dicomviewer import i18n  # noqa: E402
from multi_dicomviewer.core import table_export  # noqa: E402
from multi_dicomviewer.ui import case_presentation_window as cpw  # noqa: E402

i18n.set_language("ja")
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


class Shell(QMainWindow):
    def case_series_loaded(self, uid):
        return True


w = cpw.CasePresentationWindow(Shell())
w._rows = [  # synthetic rows only
    {"series_uid": "1.2.3.1", "modality": "XA", "number": 3, "frames": 60,
     "size_mb": 12.5, "time": "091200", "comment": "LAD 造影"},
    {"series_uid": "1.2.3.2", "modality": "IVUS", "number": 7, "frames": 900,
     "size_mb": 80.25, "time": "091500", "comment": "pullback <a&b>"},
]
w._rebuild()

texts = [b.text() for b in w.findChildren(QPushButton)]
check("row 2: no Reorder buttons any more",
      not any(x in texts for x in ("⤒", "⏫", "▲", "▼", "⏬", "⤓")))
check("row 2: 書き出し… right of 名前を付けて保存…",
      "書き出し…" in texts
      and texts.index("書き出し…") == texts.index("名前を付けて保存…") + 1)
i18n.set_language("en")
check("en: Exclude / Delete (no 'selected')",
      i18n.t("一括除外") == "Exclude" and i18n.t("一括削除") == "Delete")
i18n.set_language("ja")

w._hide_col(cpw._HEADERS.index("サイズ"))
headers, rows = w._export_table()
check("export columns = shown, no button columns, hidden one left out",
      headers == ["No", "種別", "Ser", "フレーム", "時間", "統合時間", "コメント"],
      str(headers))
check("export rows = the table text", rows[0][1] == "XA" and rows[1][-1]
      == "pullback <a&b>" and len(rows) == 2, str(rows))

tmp = tempfile.mkdtemp(prefix="mdv_cpx_")
real = QFileDialog.getSaveFileName
try:
    out = os.path.join(tmp, "case")
    QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (out, "Excel (*.xlsx)"))
    w._export()
    x = out + ".xlsx"
    ok = os.path.isfile(x)
    sheet = ""
    if ok:
        with zipfile.ZipFile(x) as z:
            sheet = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
            names = z.namelist()
    check("xlsx written (extension added from the chosen type)",
          ok and "[Content_Types].xml" in names, x)
    check("xlsx: header + text + numbers + escaping",
          "コメント" in sheet and "LAD 造影" in sheet and "<v>900</v>" in sheet
          and "pullback &lt;a&amp;b&gt;" in sheet)
    try:
        import openpyxl                       # optional cross-check
        wb = openpyxl.load_workbook(x)
        ws = wb.active
        check("openpyxl reads it", ws["A1"].value == "No"
              and ws.cell(3, 7).value == "pullback <a&b>")
    except ImportError:
        print("SKIP openpyxl not installed")
    QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (os.path.join(tmp, "case.pdf"), "PDF (*.pdf)"))
    w._export()
    pdf = os.path.join(tmp, "case.pdf")
    check("pdf written", os.path.isfile(pdf)
          and open(pdf, "rb").read(5) == b"%PDF-"
          and os.path.getsize(pdf) > 1000)
finally:
    QFileDialog.getSaveFileName = real
check("table_export column names", table_export._col_name(0) == "A"
      and table_export._col_name(26) == "AA")

# Header: left-aligned titles drawn by our own path (no fallback).
fallback = []
orig = QHeaderView.paintSection


def spy(self, *a):
    fallback.append(1)
    return orig(self, *a)


QHeaderView.paintSection = spy
try:
    w.setFloating(True)
    w.resize(1200, 400)
    w.show()
    app.processEvents()
    w._table.horizontalHeader().grab()
finally:
    QHeaderView.paintSection = orig
check("header titles painted left-aligned (no fallback)", not fallback)
w.close()

# Show → the selected pane first (the captured pane only as a fallback).
src = (ROOT / "multi_dicomviewer/ui/main_window.py").read_text(encoding="utf-8")
fn = next(n for n in ast.walk(ast.parse(src))
          if isinstance(n, ast.FunctionDef) and n.name == "case_redisplay")
body = ast.get_source_segment(src, fn)
check("case_redisplay: the active pane wins over the captured pane_index",
      body.index("if self._active in shown:")
      < body.index("pane = self._panes[idx]"))

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
