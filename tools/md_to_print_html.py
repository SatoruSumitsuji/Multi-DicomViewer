"""Convert a Markdown document to a self-contained, print-optimised HTML
file (A4, CJK-safe fonts, page-break rules) suitable for "Print to PDF".

Supports the Markdown subset used by the project docs: ATX headings,
fenced code blocks, GitHub-style tables, ordered / unordered lists,
blockquotes, horizontal rules, and inline bold / code / links. Heading
anchors use a GitHub-like slug so in-document TOC links keep working.

Usage:
    python tools/md_to_print_html.py docs/USER_GUIDE_ja.md docs/USER_GUIDE_ja.html
"""
from __future__ import annotations

import html
import re
import sys


def slug(text: str) -> str:
    """GitHub-like heading slug: lowercase, strip punctuation (keep word
    chars, spaces, hyphens, and any non-ASCII letters such as kana/kanji),
    then spaces → hyphens."""
    s = text.strip().lower()
    s = re.sub(r"[^\w\s\-ぁ-んァ-ヶ一-龠ー]", "", s, flags=re.UNICODE)
    s = s.replace(" ", "-")
    return s


def inline(text: str) -> str:
    """Inline Markdown → HTML on an already-plain string."""
    out = html.escape(text)
    # inline code first so ** inside code isn't bolded
    out = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        out,
    )
    return out


def convert(md: str) -> str:
    lines = md.split("\n")
    htmlout: list[str] = []
    i = 0
    n = len(lines)

    def close_list(stack):
        while stack:
            htmlout.append(f"</{stack.pop()}>")

    list_stack: list[str] = []

    while i < n:
        line = lines[i]

        # fenced code block
        if line.startswith("```"):
            close_list(list_stack)
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            i += 1  # skip closing fence
            htmlout.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue

        # table: a header row followed by a |---|---| separator
        if (line.lstrip().startswith("|")
                and i + 1 < n
                and re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", lines[i + 1])
                and "-" in lines[i + 1]):
            close_list(list_stack)

            def cells(row):
                row = row.strip()
                if row.startswith("|"):
                    row = row[1:]
                if row.endswith("|"):
                    row = row[:-1]
                return [c.strip() for c in row.split("|")]

            header = cells(line)
            i += 2  # skip header + separator
            htmlout.append("<table><thead><tr>")
            for c in header:
                htmlout.append(f"<th>{inline(c)}</th>")
            htmlout.append("</tr></thead><tbody>")
            while i < n and lines[i].lstrip().startswith("|"):
                htmlout.append("<tr>")
                for c in cells(lines[i]):
                    htmlout.append(f"<td>{inline(c)}</td>")
                htmlout.append("</tr>")
                i += 1
            htmlout.append("</tbody></table>")
            continue

        # horizontal rule
        if re.match(r"^\s*---+\s*$", line):
            close_list(list_stack)
            htmlout.append("<hr>")
            i += 1
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            close_list(list_stack)
            level = len(m.group(1))
            text = m.group(2).strip()
            htmlout.append(
                f'<h{level} id="{slug(text)}">{inline(text)}</h{level}>'
            )
            i += 1
            continue

        # blockquote
        if line.lstrip().startswith(">"):
            close_list(list_stack)
            buf = []
            while i < n and lines[i].lstrip().startswith(">"):
                buf.append(inline(re.sub(r"^\s*>\s?", "", lines[i])))
                i += 1
            htmlout.append("<blockquote>" + "<br>".join(buf) + "</blockquote>")
            continue

        # unordered list
        m = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        if m:
            if "ul" not in list_stack:
                close_list(list_stack)
                htmlout.append("<ul>")
                list_stack.append("ul")
            htmlout.append(f"<li>{inline(m.group(2))}</li>")
            i += 1
            continue

        # ordered list
        m = re.match(r"^(\s*)\d+\.\s+(.*)$", line)
        if m:
            if "ol" not in list_stack:
                close_list(list_stack)
                htmlout.append("<ol>")
                list_stack.append("ol")
            htmlout.append(f"<li>{inline(m.group(2))}</li>")
            i += 1
            continue

        # blank line ends a list / paragraph
        if not line.strip():
            close_list(list_stack)
            i += 1
            continue

        # paragraph
        close_list(list_stack)
        htmlout.append(f"<p>{inline(line)}</p>")
        i += 1

    close_list(list_stack)
    return "\n".join(htmlout)


_CSS = """
@page { size: A4; margin: 18mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: "Yu Gothic", "Meiryo", "Hiragino Kaku Gothic ProN",
               "Noto Sans CJK JP", sans-serif;
  font-size: 10.5pt; line-height: 1.7; color: #1a1a1a;
  max-width: 860px; margin: 0 auto; padding: 24px;
}
h1 { font-size: 20pt; border-bottom: 3px solid #1f6feb; padding-bottom: 6px;
     margin: 0 0 14px; }
h2 { font-size: 15pt; border-bottom: 1px solid #c9c9c9; padding-bottom: 4px;
     margin: 22px 0 10px; page-break-after: avoid; }
h3 { font-size: 12.5pt; margin: 16px 0 8px; page-break-after: avoid; }
p { margin: 6px 0; }
ul, ol { margin: 6px 0 6px 0; padding-left: 1.6em; }
li { margin: 3px 0; }
code { font-family: "Consolas", "Yu Gothic", monospace;
       background: #f0f2f4; padding: 1px 4px; border-radius: 3px;
       font-size: 9.5pt; }
pre { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px;
      padding: 10px 12px; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 9pt; line-height: 1.45; }
blockquote { border-left: 4px solid #e3a008; background: #fff8e6;
             margin: 10px 0; padding: 8px 14px; color: #5a4a1a;
             page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 10px 0;
        page-break-inside: avoid; font-size: 10pt; }
th, td { border: 1px solid #c9c9c9; padding: 5px 9px; text-align: left;
         vertical-align: top; }
th { background: #eef3fb; }
tr:nth-child(even) td { background: #fafbfc; }
a { color: #0b56c4; text-decoration: none; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 18px 0; }
h2 { string-set: none; }
"""


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    with open(argv[1], encoding="utf-8") as f:
        md = f.read()
    body = convert(md)
    doc = (
        "<!DOCTYPE html>\n<html lang=\"ja\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<title>Multi-DICOMviewer 使い方説明書</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )
    with open(argv[2], "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"wrote {argv[2]} ({len(doc)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
