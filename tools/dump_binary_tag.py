"""Binary DICOM tag → 3-pattern text dump (manual helper, not part of the app).

Reads one DICOM tag's raw bytes from a file (or the first file under a folder
that contains the tag) and writes its value in three representations, so a
binary element (VR OB/OW/UN/…) that the viewer/CSV only summarises as
``<binary: N bytes>`` can still be exported in full when needed:

  * <stem>.<tag>.hex.txt     — space-separated uppercase hex ("AB CD …")
  * <stem>.<tag>.base64.txt  — Base64 (text-safe, fully reversible)
  * <stem>.<tag>.latin1.txt  — bytes decoded 1:1 as Latin-1 (UTF-8 file;
                               mostly mojibake unless the bytes are text)

Pick by use: Base64 to preserve/restore the exact bytes elsewhere, hex to
inspect byte values, latin1 to fish out any embedded text fragments.

Usage:
    python tools/dump_binary_tag.py <path to .dcm or folder> <tag> [out_dir]

    <tag>   group,element in hex, e.g.  0029,1007   (comma optional: 00291007)
    out_dir optional; defaults to the source file's folder.

Examples:
    python tools/dump_binary_tag.py "C:\\data\\TVC00101" 0029,1007
    python tools/dump_binary_tag.py case.dcm 7FE0,0010 C:\\out
"""
from __future__ import annotations

import base64
import os
import sys

import pydicom
from pydicom.tag import Tag


def _parse_tag(s: str) -> Tag:
    """'0029,1007' / '00291007' / '(0029,1007)' → Tag(0x0029, 0x1007)."""
    t = s.strip().lstrip("(").rstrip(")").replace(",", "").replace(" ", "")
    if len(t) != 8:
        raise ValueError(f"tag must be 8 hex digits (group+element): {s!r}")
    return Tag(int(t[:4], 16), int(t[4:], 16))


def _find_file(path: str, tag: Tag) -> str | None:
    """*path* if it is a file, else the first DICOM under it holding *tag*."""
    if os.path.isfile(path):
        return path
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            except Exception:
                continue
            if tag in ds:
                print(f"Found {tag} in: {fp}")
                return fp
    return None


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    target = sys.argv[1]
    tag = _parse_tag(sys.argv[2])
    out_dir = sys.argv[3] if len(sys.argv) > 3 else None

    fp = _find_file(target, tag)
    if fp is None:
        print(f"No DICOM containing {tag} found under: {target}")
        return 1

    ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
    if tag not in ds:
        print(f"{tag} not present in {fp}")
        return 1
    elem = ds[tag]
    raw = elem.value
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raw = str(raw).encode("utf-8", "replace")
    raw = bytes(raw)

    printable = sum(1 for b in raw if 32 <= b < 127 or b in (9, 10, 13))
    print(f"Tag {tag} VR={elem.VR}  bytes={len(raw)}  "
          f"printable={printable}/{len(raw)} "
          f"({printable / max(len(raw), 1):.1%})")

    base_dir = out_dir or os.path.dirname(os.path.abspath(fp))
    os.makedirs(base_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(fp))[0]
    tag_label = f"{tag.group:04X}_{tag.element:04X}"

    written = []
    for suffix, text in (
        (f"{tag_label}.hex.txt", " ".join(f"{b:02X}" for b in raw)),
        (f"{tag_label}.base64.txt", base64.b64encode(raw).decode("ascii")),
        (f"{tag_label}.latin1.txt", raw.decode("latin-1")),
    ):
        path = os.path.join(base_dir, f"{stem}.{suffix}")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        written.append((path, len(text)))

    print("Wrote:")
    for path, n in written:
        print(f"  {path}  ({n} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
