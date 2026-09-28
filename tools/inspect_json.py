#!/usr/bin/env python3
"""Summarise a Multi-DicomViewer sidecar JSON without dumping its huge blobs.

Prints the top-level keys, the format/type marker, the source-CT series, and —
for each embedded mask (region / endo / blood / thick) — whether it's present
and its shape / vol_shape. Packed base64 strings are shown only as a length, so
the output stays readable. Works for EpiLv / EndoLv / BldLv / MV / AoV / Apex /
FullLv / corotree / cpr files (and any JSON, really).

Usage:
    python tools/inspect_json.py <file.json> [<file2.json> ...]
"""
from __future__ import annotations

import json
import os
import sys

_MASK_KEYS = ("region", "endo", "blood", "thick")


def _mask_line(name, m):
    if not isinstance(m, dict):
        return f"    {name:8}: (none)"
    plen = len(m["packed"]) if isinstance(m.get("packed"), str) else 0
    return (f"    {name:8}: shape={m.get('shape')} "
            f"vol_shape={m.get('vol_shape')} packed={plen}B")


def _brief(v):
    """One-line rendering of a value, truncating long strings / big lists."""
    if isinstance(v, str):
        return f'"{v[:60]}…"' if len(v) > 60 else f'"{v}"'
    if isinstance(v, list):
        return f"[list x{len(v)}]" if len(v) > 8 else str(v)
    if isinstance(v, dict):
        return "{" + ", ".join(sorted(v.keys())) + "}"
    return str(v)


def inspect(path):
    print("=" * 72)
    print(os.path.basename(path))
    if not os.path.exists(path):
        print("  !! not found")
        return
    print(f"  size: {os.path.getsize(path):,} bytes")
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as exc:                              # noqa: BLE001
        print(f"  !! parse error: {exc}")
        return
    if not isinstance(d, dict):
        print(f"  (top-level is {type(d).__name__}, not an object)")
        return
    print("  keys   :", ", ".join(sorted(d.keys())))
    for k in ("format", "type", "kind", "valve", "version"):
        if k in d:
            print(f"  {k:7}:", d[k])
    ser = d.get("series") or (d.get("src") or {})
    if isinstance(ser, dict) and ser:
        print("  series :", {kk: ser[kk] for kk in ("series_uid", "patient",
              "date", "series_number") if kk in ser})
    if d.get("src_dir"):
        print("  src_dir:", d["src_dir"])
    if d.get("spacing") is not None:
        print("  spacing:", d["spacing"])
    # masks (direct, or nested under a FullLv 'bld'/'epi')
    holders = [("", d)]
    if isinstance(d.get("bld"), dict):
        holders.append(("bld.", d["bld"]))
    if isinstance(d.get("epi"), dict):
        holders.append(("epi.", d["epi"]))
    any_mask = False
    for prefix, h in holders:
        for mk in _MASK_KEYS:
            if mk in h:
                any_mask = True
                print(_mask_line(prefix + mk, h[mk]))
    if not any_mask:
        print("    (no embedded masks)")
    # coronary tree vessels
    if isinstance(d.get("vessels"), list):
        print(f"  vessels: {len(d['vessels'])}")
        for v in d["vessels"][:20]:
            print(f"    - {v.get('role','?'):5} {v.get('name','')} "
                  f"(parent={v.get('parent')})")


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    for p in args:
        inspect(p)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
