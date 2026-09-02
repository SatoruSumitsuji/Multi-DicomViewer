"""Pure logic for the Case Presentation tool (Tools ▸ Case Presentation).

No Qt / no I/O here so it stays unit-testable. The UI (ui/case_presentation_
window.py) owns the table; the shell (main_window) owns capture / re-display.

Two jobs live here:

* Time alignment — XA / IVUS / CT clocks drift between machines. One modality
  is the REFERENCE (XA by default); every other modality gets a constant
  offset (seconds) so its acquisition times map onto the reference clock. The
  offset is found from a user-picked anchor pair (one reference row + one row
  of the other modality taken to be the same real moment) or typed by hand.
  ``unified_time`` = the row's own time on the reference clock.

* Modified chronological sort — after alignment there is still a few-second
  residual jitter, so a non-reference event whose unified time lands within a
  tolerance (default 10 s) of a reference event is snapped to sit *immediately
  after* that reference event rather than a hair before it.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

_EPOCH = datetime(1970, 1, 1)


def parse_dcm_dt(date_str: str, time_str: str) -> Optional[float]:
    """DICOM DA (``YYYYMMDD``) + TM (``HHMMSS[.ffffff]``) → seconds since a
    fixed naive epoch (only *differences* are ever used, so the epoch and the
    absence of a timezone don't matter). None if unparseable / empty."""
    date_str = (date_str or "").strip()
    time_str = (time_str or "").strip().replace(":", "")
    if len(date_str) < 8:
        return None
    try:
        y, mo, d = int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:8])
    except ValueError:
        return None
    hh = mm = ss = us = 0
    if time_str:
        try:
            if len(time_str) >= 2:
                hh = int(time_str[0:2])
            if len(time_str) >= 4:
                mm = int(time_str[2:4])
            if len(time_str) >= 6:
                sec = float(time_str[4:])          # may carry ".ffffff"
                ss = int(sec)
                us = int(round((sec - ss) * 1e6))
        except ValueError:
            hh = mm = ss = us = 0
    try:
        dt = datetime(y, mo, d, min(hh, 23), min(mm, 59),
                      min(ss, 59), min(us, 999999))
    except ValueError:
        return None
    return (dt - _EPOCH).total_seconds()


def offset_from_anchor(ref_dt: float, other_dt: float) -> float:
    """Seconds to add to the *other* modality's times to land on the reference
    clock, given an anchor pair taken to be the same real moment."""
    return float(ref_dt) - float(other_dt)


def unified_time(row_dt: Optional[float], modality: str, reference: str,
                 offsets: dict) -> Optional[float]:
    """The row's time expressed on the reference clock (None if it has no
    time). Reference-modality rows are returned unchanged."""
    if row_dt is None:
        return None
    if modality == reference:
        return float(row_dt)
    return float(row_dt) + float(offsets.get(modality, 0.0))


def modified_sort_order(items: list, tol: float = 10.0) -> list:
    """Return row indices in presentation order.

    *items* is a list of dicts, each with:
      ``dt``     — unified time in seconds, or None (no time known)
      ``is_ref`` — True for a reference-modality (e.g. XA) row

    Rule: a non-reference row whose unified time is within *tol* seconds of a
    reference row is placed immediately AFTER the nearest such reference row.
    Rows with no time sort to the end, keeping their original order.
    """
    refs = [(i, it["dt"]) for i, it in enumerate(items)
            if it.get("is_ref") and it.get("dt") is not None]
    keys = []
    for i, it in enumerate(items):
        dt = it.get("dt")
        if dt is None:
            keys.append((1, 0.0, 0, 0.0, i))         # no time → end, stable
            continue
        if it.get("is_ref"):
            keys.append((0, dt, 0, dt, i))           # reference: rank 0
            continue
        # nearest reference within tolerance → snap after it
        best_dt = None
        best_d = None
        for _ri, rdt in refs:
            d = abs(dt - rdt)
            if d <= tol and (best_d is None or d < best_d):
                best_d, best_dt = d, rdt
        primary = best_dt if best_dt is not None else dt
        keys.append((0, primary, 1, dt, i))          # non-ref: rank 1, own dt
    return sorted(range(len(items)), key=lambda i: keys[i])


def json_safe(obj):
    """Recursively convert a captured view-state dict to JSON-serialisable
    types (numpy scalars/arrays → Python / lists) so it can be saved."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
