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
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core.coronary_territory import (   # noqa: E402
    CoronaryTree, format_territory_report)
from multi_dicomviewer.core.lv_function import LVFunction   # noqa: E402


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

    lines, _eng = format_territory_report(tree, lvf, max_dist_mm=args.max_dist)
    print("=" * 60)
    print("CT Territory report")
    print("  tree :", os.path.basename(args.corotree),
          f"({len(tree.vessels)} vessels)")
    print("  LV   : Epi", os.path.basename(args.epilv),
          "| Endo", os.path.basename(args.bldlv))
    print("-" * 60)
    for ln in lines:
        print(ln)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
