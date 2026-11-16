"""lv_epi_convert — turn an OLD EpiLv.json into the current MV-AoV tent form.

Old EpiLv files (before the MV-AoV point method) hold the Epi border traced on
the long-axis planes about the OLD axis (apex → MV centre), each wall ending at
the MV plane. The current method traces about the axis common apex → MV-AoV
point, and both basal ends of every border run INTO the MV-AoV point (the
"tent", see lv_measure.tent_terminal_ok), so the Epi base is closed with no MV /
AoV plane cut.

Conversion (pure numpy, no GUI):

1. rebuild the old Epi surface (its deepest traced extent, on its own axis);
2. cut that surface with the NEW axis' long-axis planes (the current plane
   count) — the same ring/plane reslice ``LVModel._mv_parallel_surface`` uses —
   giving the two wall profiles (along, radius) of every plane;
3. keep each wall apical of the MV-AoV point's level, resample it to a few
   editable points, and join  point → wall A → apex → wall B → point;
4. the apex vertex is the NEW (common) apex; the old apex is not used.

The result is an EpiLv dict ("epi_tent": true) the viewer applies like a
loaded file. It is NOT saved — the user reviews / edits it and saves as usual.
Endo data in the file (if any) is kept as is.
"""
from __future__ import annotations

import numpy as np

from .lv_axis import LVAxis
from .lv_measure import LV_PLANE_COUNTS, LVModel

#: A wall point must stay this far apical of the MV-AoV point's level (mm) —
#: more than tent_terminal_ok's 0.5 mm, so no converted point is dropped.
TENT_GAP_MM = 1.0
#: Editable points per converted wall (base → apex), excluding the two ends.
WALL_POINTS = 10

#: Keys of the old file that no longer match the converted Epi (they are
#: recomputed by Calc Vol / HU補正 / Save).
_STALE_KEYS = ("region", "volume", "epi_sax_levels", "epi_sax_extra",
               "epi_hu_done", "axis_def", "saved_at")


def is_old_epi(data) -> bool:
    """True if *data* is an EpiLv dict with an Epi border made BEFORE the
    MV-AoV tent method (no ``epi_tent`` marker)."""
    return (isinstance(data, dict) and data.get("kind") == "mdv-lvef"
            and bool(data.get("epi_planes")) and not data.get("epi_tent"))


def _plane_walls(rings, ax: LVAxis, phi: float):
    """(pos, neg) wall samples [(along, radius)] where the ring stack *rings*
    crosses the long-axis plane *phi* of *ax* (pos = the +e_s side)."""
    _o, e_s, _e_t, nrm = ax.long_axis_basis(phi)
    apex = np.asarray(ax.apex, float)
    pos, neg = [], []
    for ring in rings:
        b = (ring - apex) @ nrm                     # signed distance to the plane
        m = len(ring)
        for k in range(m):
            j = (k + 1) % m
            b0, b1 = float(b[k]), float(b[j])
            if b0 == b1:
                continue
            if (b0 <= 0.0 <= b1) or (b1 <= 0.0 <= b0):
                t = b0 / (b0 - b1)
                P = ring[k] + t * (ring[j] - ring[k])
                dp = P - apex
                s = float(dp @ e_s)
                (pos if s >= 0.0 else neg).append((float(dp @ ax.axis), abs(s)))
    return pos, neg


def _resample(samples, top: float):
    """Sorted, de-duplicated (along, radius) samples between the apex and *top*,
    resampled to WALL_POINTS levels. None if the wall is too short."""
    if len(samples) < 2:
        return None
    arr = np.array(sorted(samples), float)
    keep = np.concatenate(([True], np.diff(arr[:, 0]) > 1e-6))
    arr = arr[keep]
    arr = arr[(arr[:, 0] > 0.5) & (arr[:, 0] < top)]
    if len(arr) < 2:
        return None
    lv = np.linspace(float(arr[0, 0]), float(arr[-1, 0]), WALL_POINTS)
    return lv, np.interp(lv, arr[:, 0], arr[:, 1])


def convert(data, apex, axis_dir, radial0, point, n_planes):
    """Convert the old EpiLv dict *data* to the tent form on the axis
    (*apex*, *axis_dir*, *radial0*) closing at the MV-AoV *point*.

    Returns ``(new_dict, "")``, or ``(None, reason)`` when it can't be done."""
    try:
        old = LVModel.from_dict(data)
    except Exception as exc:                            # noqa: BLE001
        return None, f"unreadable EpiLv ({exc})"
    surf = old._full_surface("epi")
    if surf is None:
        return None, "the file has no Epi border to convert"
    apex = np.asarray(apex, float).reshape(3)
    point = np.asarray(point, float).reshape(3)
    ax = LVAxis.from_frame(apex, axis_dir, radial0)
    top_level = float((point - apex) @ ax.axis)
    if top_level <= 10.0:
        return None, "the MV-AoV point is too close to the apex"
    top = top_level - TENT_GAP_MM
    n = int(n_planes) if int(n_planes) in LV_PLANE_COUNTS else old.n_planes
    if data.get("endo_planes"):
        n = old.n_planes                            # keep the Endo planes valid
    rings = [np.asarray(surf.ring_world(k), float) for k in range(len(surf.along))]
    m = LVModel(n_planes=n)
    m.epi_axis = ax
    m.axis = ax
    m.epi_tent = True
    m.set_apex_point("epi", apex)
    for phi in m.plane_angles():
        e_s = np.asarray(ax.meridian_dir(phi), float)
        walls = []
        for samples, sign in zip(_plane_walls(rings, ax, phi), (1.0, -1.0)):
            res = _resample(samples, top)
            if res is None:
                break
            lv, r = res
            walls.append(apex + lv[:, None] * ax.axis + sign * r[:, None] * e_s)
        if len(walls) != 2:
            continue
        poly = np.vstack([point[None], walls[0][::-1], apex[None], walls[1],
                          point[None]])
        m.set_long_axis_contour(phi, poly, which="epi")
    if len(m.epi_contours) < 3:
        return None, "the old Epi border does not reach the new long-axis planes"
    out = {k: v for k, v in data.items() if k not in _STALE_KEYS}
    nd = m.to_dict()
    for k in ("n_planes", "epi_axis", "epi_apex", "epi_planes", "epi_tent"):
        out[k] = nd[k]
    out["converted_from"] = {"axis_def": data.get("axis_def", "") or "",
                             "saved_at": data.get("saved_at", "") or ""}
    return out, ""
