#!/usr/bin/env python3
"""Headless CT Territory report.

Loads a coronary tree (.corotree.json) + an LV Epi region (EpiLv.json) + an LV
Endo mask (BldLv.json), builds the compact-layer myocardium (Epi ∩ ~Endo),
assigns every myocardial voxel to its nearest coronary centreline (Voronoi),
and prints the perfusion-territory volumes — per root vessel (whole subtree),
the LEFT-SYSTEM sum (LM+LAD+LCX), and the per-vessel breakdown. No GUI.

Usage:
    python tools/territory_report.py <coronary.corotree.json> \
        <*.BldLv.json> <*.EpiLv.json> [--max-dist MM]

--max-dist caps how far a voxel may be from a centreline to be assigned; voxels
beyond it are reported as "unassigned" instead of being attributed to the
nearest (however far) vessel. Omit to assign every voxel (no cap).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core.coronary_territory import (   # noqa: E402
    ROOT_ROLES, CoronaryTree, TerritoryEngine)
from multi_dicomviewer.core.lv_function import LVFunction   # noqa: E402

#: Roots that make up the LEFT coronary system (summed for LM-lesion reporting).
_LEFT_ROLES = ("LM", "LAD", "LCX")


def _short(name: str) -> str:
    """A readable vessel label from a long CPR filename stem, e.g.
    'ARIFIN;20260629_Se006@LM-LAD@202609231139' → 'LM-LAD'."""
    toks = name.split("@")
    if len(toks) < 2:
        return name
    keep = [x for x in toks[1:] if not re.fullmatch(r"[0-9;]+", x)]
    return "@".join(keep) if keep else toks[1]


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Headless CT Territory report.")
    ap.add_argument("corotree", help="coronary tree .corotree.json")
    ap.add_argument("bldlv", help="LV Endo file (*.BldLv.json)")
    ap.add_argument("epilv", help="LV Epi region file (*.EpiLv.json)")
    ap.add_argument("--max-dist", type=float, default=None,
                    help="max voxel→centreline distance (mm); else assign all")
    args = ap.parse_args(argv)

    tree = CoronaryTree.from_json(_load_json(args.corotree))
    lvf = LVFunction.from_files(_load_json(args.epilv), _load_json(args.bldlv))
    if lvf is None:
        print("ERROR: could not build the myocardium — is the Epi region "
              "(EpiLv.json) or Endo mask (BldLv.json) missing?", file=sys.stderr)
        return 2

    eng = TerritoryEngine(tree, lvf, max_dist_mm=args.max_dist)
    total = eng.myocardium_ml
    myo_only = lvf.myocardial_volume_ml()

    def pct(ml):
        return f"{100.0 * ml / total:5.1f}%" if total > 0 else "  n/a"

    print("=" * 60)
    print("CT Territory report")
    print("  tree :", os.path.basename(args.corotree),
          f"({len(tree.vessels)} vessels)")
    print("  LV   : Epi", os.path.basename(args.epilv),
          "| Endo", os.path.basename(args.bldlv))
    if args.max_dist is not None:
        print(f"  assignment cap: {args.max_dist:g} mm")
    print("-" * 60)
    print(f"Myocardium (Epi & ~Endo) : {myo_only:8.1f} mL")
    print(f"  assigned voxels total  : {total:8.1f} mL")
    print("-" * 60)

    # Per-ROOT territory = the whole subtree from that root's ostium (idx 0).
    roots = tree.roots()
    by_role = {}
    print("Per-root territory (whole subtree, from ostium):")
    for vid in roots:
        v = tree.vessels[vid]
        _mask, ml = eng.territory(vid, 0)
        by_role[v.role] = by_role.get(v.role, 0.0) + ml
        print(f"  {v.role:<4} {_short(v.name):<16} : {ml:8.1f} mL  {pct(ml)}")
    if not roots:
        print("  (no roots set — assign LM/LAD/LCX/RCA roles first)")

    # Left-system sum (LM + LAD + LCX) for LM-lesion reporting.
    left = sum(by_role.get(r, 0.0) for r in _LEFT_ROLES)
    if left > 0:
        print("-" * 60)
        present = [r for r in _LEFT_ROLES if by_role.get(r)]
        print(f"Left system ({'+'.join(present)}) : {left:8.1f} mL  {pct(left)}")

    # Unassigned (only possible with --max-dist).
    code = eng.assignment["code"]
    unassigned_ml = float(int((code < 0).sum())) * eng.voxel_ml
    if unassigned_ml > 0:
        print("-" * 60)
        print(f"Unassigned (beyond cap)  : {unassigned_ml:8.1f} mL  "
              f"{pct(unassigned_ml)}")

    # Per-vessel own share (nearest-voxel count per vessel, not subtree).
    print("-" * 60)
    print("Per-vessel own share (nearest voxels), largest first:")
    c2v = eng.assignment["code_to_vid"]
    import numpy as np
    codes = eng.assignment["code"]
    uniq, cnt = np.unique(codes[codes >= 0], return_counts=True)
    rows = sorted(zip(uniq, cnt), key=lambda x: -x[1])
    for u, c in rows:
        vid = c2v[int(u)]
        ml = float(c) * eng.voxel_ml
        print(f"  {_short(tree.vessels[vid].name):<18} : {ml:8.1f} mL  {pct(ml)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
