"""Write a simple text table (a header row + rows of strings) to .xlsx or PDF.

No extra dependency: the .xlsx is a minimal Office Open XML workbook built with
zipfile (one sheet, bold frozen header, numeric-looking cells stored as
numbers); the PDF is an HTML table laid out by QTextDocument and printed with
QPdfWriter (A4 landscape, small font, the table fits the page width).
Used by Case Presentation ▸ Export.
"""

from __future__ import annotations

import re
import zipfile
from xml.sax.saxutils import escape

_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def _col_name(i: int) -> str:
    """0 → A, 25 → Z, 26 → AA."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _xml_text(v) -> str:
    # Drop the control characters XML 1.0 forbids (keep tab / newline).
    s = "" if v is None else str(v)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    return escape(s)


def write_xlsx(path: str, headers: list, rows: list,
               sheet: str = "Sheet1") -> None:
    """Write *headers* + *rows* (lists of str) as a one-sheet .xlsx."""
    ncol = max([len(headers)] + [len(r) for r in rows]) if (headers or rows) else 0
    widths = [8.0] * ncol
    for r in [headers] + list(rows):
        for c, v in enumerate(r):
            n = len(str(v or ""))
            n += sum(1 for ch in str(v or "") if ord(ch) > 0x2E7F)   # wide chars
            widths[c] = min(60.0, max(widths[c], n + 2.0))

    def cell(ri, ci, v, bold=False):
        ref = f"{_col_name(ci)}{ri}"
        st = ' s="1"' if bold else ""
        s = "" if v is None else str(v)
        if not bold and _NUM.match(s.strip()):
            return f'<c r="{ref}"{st}><v>{s.strip()}</v></c>'
        return (f'<c r="{ref}" t="inlineStr"{st}><is><t xml:space="preserve">'
                f"{_xml_text(s)}</t></is></c>")

    lines = []
    lines.append('<row r="1">' + "".join(cell(1, c, h, True)
                                         for c, h in enumerate(headers)) + "</row>")
    for i, r in enumerate(rows, start=2):
        lines.append(f'<row r="{i}">' + "".join(cell(i, c, v)
                                                for c, v in enumerate(r)) + "</row>")
    cols = "".join(f'<col min="{c + 1}" max="{c + 1}" width="{w:.1f}" '
                   'customWidth="1"/>' for c, w in enumerate(widths))
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2"'
        ' activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        + (f"<cols>{cols}</cols>" if cols else "")
        + "<sheetData>" + "".join(lines) + "</sheetData></worksheet>")
    sname = _xml_text(re.sub(r'[\[\]:*?/\\]', "_", sheet)[:31] or "Sheet1")
    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/'
            'vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            "</Types>"),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
            'relationships"><Relationship Id="rId1" Type="http://schemas.'
            'openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
            ' Target="xl/workbook.xml"/></Relationships>'),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/'
            'main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/'
            f'relationships"><sheets><sheet name="{sname}" sheetId="1" r:id="rId1"/>'
            "</sheets></workbook>"),
        "xl/_rels/workbook.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
            'relationships"><Relationship Id="rId1" Type="http://schemas.'
            'openxmlformats.org/officeDocument/2006/relationships/worksheet"'
            ' Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http:'
            '//schemas.openxmlformats.org/officeDocument/2006/relationships/styles"'
            ' Target="styles.xml"/></Relationships>'),
        "xl/styles.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/'
            '2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/>'
            '</font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill><fill>'
            '<patternFill patternType="gray125"/></fill></fills><borders count="1">'
            "<border><left/><right/><top/><bottom/><diagonal/></border></borders>"
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" '
            'borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" '
            'fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" '
            'fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
            "</styleSheet>"),
        "xl/worksheets/sheet1.xml": sheet_xml,
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)


def table_html(headers: list, rows: list, title: str = "") -> str:
    """The table as HTML (QTextDocument subset) for the PDF."""
    th = "".join(f"<th align='left'>{escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(v or ''))}</td>" for v in r) + "</tr>"
        for r in rows)
    head = f"<h3>{escape(title)}</h3>" if title else ""
    return ("<html><body style='font-size:8pt'>" + head
            + "<table border='1' cellspacing='0' cellpadding='3' width='100%'"
            " style='border-collapse:collapse'>"
            f"<thead><tr style='background:#e8e8e8'>{th}</tr></thead>"
            f"<tbody>{body}</tbody></table></body></html>")


def write_pdf(path: str, headers: list, rows: list, title: str = "") -> None:
    """Write the table as a PDF (A4 landscape; needs a QGuiApplication)."""
    from PyQt6.QtCore import QMarginsF, QSizeF
    from PyQt6.QtGui import QPageLayout, QPageSize, QPdfWriter, QTextDocument
    w = QPdfWriter(path)
    w.setPageLayout(QPageLayout(QPageSize(QPageSize.PageSizeId.A4),
                                QPageLayout.Orientation.Landscape,
                                QMarginsF(10, 10, 10, 10),
                                QPageLayout.Unit.Millimeter))
    w.setResolution(96)
    w.setTitle(title or "Table")
    doc = QTextDocument()
    doc.setHtml(table_html(headers, rows, title))
    r = w.pageLayout().paintRectPixels(w.resolution())
    doc.setPageSize(QSizeF(r.width(), r.height()))
    pr = getattr(doc, "print", None) or getattr(doc, "print_")
    pr(w)
