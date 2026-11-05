# -*- coding: utf-8 -*-
"""i18n coverage: every UI string the code passes through t() has an entry in
each language table, placeholders survive translation, and no Japanese is
left on an English / Chinese / Spanish UI.

Keys = the string-constant first argument of every t(...) / _t(...) call, plus
the label tables that reach t() through a variable (see CONTAINERS / EXTRA),
plus any other Japanese string literal in the package (a Japanese literal that
is shown without t() would leak into a non-Japanese UI).

Run: python tools/test_i18n_coverage.py
"""
import ast
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "multi_dicomviewer"
sys.path.insert(0, str(ROOT.parent))
JA = re.compile(r"[぀-ヿ㐀-鿿]")
KANA = re.compile(r"[぀-ヿ]")
PH = re.compile(r"\{[^{}]*\}")
SKIP = {"bixa.py", "bixa_window.py", "i18n.py"}
CONTAINERS = {"_FORMATS", "_MARKER_LABELS", "FIELD_LABELS", "_LABELS", "_CT_MODES",
              "_HEADERS", "_THICK_MODES", "_LVV_KNOB_ROLE", "_CPR_AUTO_STAGE",
              "TARGET_COLOR_CHOICES"}
EXTRA = ["Export DICOM", "Export MP4", "Export CSV", "Export Anon DICOM", "Export",
         "Bi", "Lt", "Rt", "Zoom", "Move", "Rotate", "Spin", "Paging", "Thick",
         "WL", "Rt90°", "Lt90°", "Flip-H", "Flip-V", "Point"]


def collect() -> dict:
    keys: dict = {}

    def add(k, where):
        if isinstance(k, str) and re.search(r"[A-Za-z぀-ヿ㐀-鿿]", k):
            keys.setdefault(k, where)

    for p in sorted(ROOT.rglob("*.py")):
        if p.name in SKIP:
            continue
        rel = p.relative_to(ROOT)
        tree = ast.parse(p.read_text(encoding="utf-8"))
        par = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in ("t", "_t") and n.args
                    and isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[0].value, str)):
                add(n.args[0].value, f"{rel}:{n.lineno}")
            if isinstance(n, ast.Assign):
                names = {getattr(tg, "id", None) or getattr(tg, "attr", None)
                         for tg in n.targets}
                if names & CONTAINERS:
                    for c in ast.walk(n.value):
                        if isinstance(c, ast.Constant) and isinstance(c.value, str):
                            s = c.value
                            if s.startswith(("#", "_")) or re.fullmatch(r"[a-z0-9_]+", s):
                                continue
                            add(s, f"{rel}:{c.lineno}")
            if (isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and JA.search(n.value)):
                pa = par.get(n)
                if isinstance(pa, (ast.Expr, ast.JoinedStr)):
                    continue                     # docstring / f-string part
                if "satoru@" in n.value:
                    continue
                add(n.value, f"{rel}:{n.lineno}")
    for k in EXTRA:
        add(k, "extra")
    return keys


def main() -> int:
    keys = collect()
    tbl = {c: json.loads((ROOT / "resources" / "i18n" / f"{c}.json")
                         .read_text(encoding="utf-8"))
           for c in ("en", "ja", "zh", "es")}
    fails = []

    def fail(msg):
        fails.append(msg)

    for k, where in keys.items():
        ja_key = bool(JA.search(k))
        if ja_key and k not in tbl["en"]:
            fail(f"en missing: {where} {k[:60]!r}")
        for c in ("zh", "es"):
            if k not in tbl[c]:
                fail(f"{c} missing: {where} {k[:60]!r}")
        if not ja_key and k not in tbl["ja"]:
            fail(f"ja missing: {where} {k[:60]!r}")
    for c, tb in tbl.items():
        for k, v in tb.items():
            if sorted(PH.findall(k)) != sorted(PH.findall(v)):
                fail(f"{c} placeholder mismatch: {k[:50]!r} -> {v[:50]!r}")
            if c in ("en", "es") and JA.search(v):
                fail(f"{c} value has Japanese: {k[:40]!r} -> {v[:40]!r}")
            if c == "zh" and KANA.search(v):
                fail(f"zh value has kana: {k[:40]!r} -> {v[:40]!r}")
    # The runtime lookup: an English UI never returns a Japanese key.
    from multi_dicomviewer import i18n
    for c in ("en", "zh", "es"):
        i18n.set_language(c)
        leak = [k for k in keys if JA.search(k)
                and (KANA if c == "zh" else JA).search(i18n.t(k))]
        if leak:
            fail(f"{c}: {len(leak)} keys still Japanese at runtime, e.g. {leak[0][:50]!r}")
    i18n.set_language("en")

    for m in fails[:80]:
        print("FAIL", m)
    print(f"{len(keys)} keys checked;",
          "ALL PASS" if not fails else f"{len(fails)} FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
