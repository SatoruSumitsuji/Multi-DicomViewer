"""Headless tests for core.case_presentation (time align + modified sort)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core.case_presentation import (  # noqa: E402
    json_safe, modified_sort_order, offset_from_anchor, parse_dcm_dt,
    unified_time)


def test_parse():
    a = parse_dcm_dt("20260826", "091200")
    b = parse_dcm_dt("20260826", "091210")
    assert a is not None and b is not None
    assert abs((b - a) - 10.0) < 1e-6
    # fractional seconds
    c = parse_dcm_dt("20260826", "091200.500000")
    assert abs((c - a) - 0.5) < 1e-6
    # midnight crossing across days
    d0 = parse_dcm_dt("20260826", "235959")
    d1 = parse_dcm_dt("20260827", "000009")
    assert abs((d1 - d0) - 10.0) < 1e-6
    # empty / bad
    assert parse_dcm_dt("", "091200") is None
    assert parse_dcm_dt("2026", "") is None
    print("parse OK")


def test_offset_unified():
    xa = parse_dcm_dt("20260826", "091200")     # ref moment
    iv = parse_dcm_dt("20260826", "090500")     # IVUS clock 7 min behind
    off = offset_from_anchor(xa, iv)
    assert abs(off - 420.0) < 1e-6
    offsets = {"IVUS": off}
    # an IVUS row 3 min after its anchor → unified = xa_anchor + 180
    iv2 = parse_dcm_dt("20260826", "090800")
    u = unified_time(iv2, "IVUS", "XA", offsets)
    assert abs(u - (xa + 180.0)) < 1e-6
    # XA row unchanged
    assert unified_time(xa, "XA", "XA", offsets) == xa
    # no time → None
    assert unified_time(None, "IVUS", "XA", offsets) is None
    print("offset/unified OK")


def _order_labels(rows):
    items = [{"dt": r["dt"], "is_ref": r["mod"] == "XA"} for r in rows]
    order = modified_sort_order(items, tol=10.0)
    return [rows[i]["label"] for i in order]


def test_modified_sort():
    base = parse_dcm_dt("20260826", "091200")
    rows = [
        {"label": "XA3",   "mod": "XA",   "dt": base + 0},        # 09:12:00
        {"label": "IVUS5", "mod": "IVUS", "dt": base - 4},        # 4s BEFORE XA3
        {"label": "IVUS9", "mod": "IVUS", "dt": base + 810},      # 13.5 min later
        {"label": "XA8",   "mod": "XA",   "dt": base + 800},
    ]
    labels = _order_labels(rows)
    # IVUS5 is within 10s of XA3 → snapped AFTER XA3 (not before it)
    assert labels.index("XA3") < labels.index("IVUS5"), labels
    # IVUS9 (far) stays in true chronological place, after XA8
    assert labels.index("XA8") < labels.index("IVUS9"), labels
    assert labels == ["XA3", "IVUS5", "XA8", "IVUS9"], labels

    # two non-ref snapped to the same XA → ordered by their own time
    rows2 = [
        {"label": "XA",  "mod": "XA",   "dt": base},
        {"label": "Bp5", "mod": "IVUS", "dt": base + 5},
        {"label": "Am3", "mod": "US",   "dt": base - 3},
    ]
    labels2 = _order_labels(rows2)
    assert labels2 == ["XA", "Am3", "Bp5"], labels2

    # rows with no time go to the end, original order preserved
    rows3 = [
        {"label": "N1", "mod": "CT", "dt": None},
        {"label": "X",  "mod": "XA", "dt": base},
        {"label": "N2", "mod": "CT", "dt": None},
    ]
    labels3 = _order_labels(rows3)
    assert labels3 == ["X", "N1", "N2"], labels3
    print("modified_sort OK")


def test_json_safe():
    import numpy as np
    st = {"slice": np.int64(12), "cam": np.array([1.0, 2.0, 3.0]),
          "nested": {"w": np.float32(0.5)}, "tup": (1, 2)}
    out = json_safe(st)
    import json
    json.dumps(out)                       # must not raise
    assert out["slice"] == 12 and out["cam"] == [1.0, 2.0, 3.0]
    assert out["nested"]["w"] == 0.5 and out["tup"] == [1, 2]
    print("json_safe OK")


if __name__ == "__main__":
    test_parse()
    test_offset_unified()
    test_modified_sort()
    test_json_safe()
    print("ALL OK")
