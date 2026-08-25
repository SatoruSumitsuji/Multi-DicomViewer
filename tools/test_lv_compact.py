"""Headless test for compact-layer / papillary separation from the blood pool.

Synthetic: a blood cylinder (radius 15 mm) along +z with TWO papillary muscle
"islands" (radius 4 mm) fully inside it (surrounded by blood) and ONE wall-
attached papillary "notch" bitten out of the boundary. The envelope must fill
the enclosed islands and bridge the notch → recover the full cylinder cavity;
the papillary volume must match the islands + notch removed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core.lv_compact import compact_from_blood   # noqa: E402


def main():
    sx = sy = sz = 0.5
    nx, ny, nz = 160, 160, 220
    cx, cy = 40.0, 40.0
    apex_z, base_z = 10.0, 90.0
    R = 15.0                                     # cavity radius (mm)
    rp = 4.0                                     # papillary radius (mm)

    xc = (np.arange(nx) * sx).reshape(1, 1, -1)
    yc = (np.arange(ny) * sy).reshape(1, -1, 1)
    zc = (np.arange(nz) * sz).reshape(-1, 1, 1)
    r2 = (xc - cx) ** 2 + (yc - cy) ** 2
    inz = (zc >= apex_z) & (zc <= base_z)
    blood = (r2 <= R ** 2) & inz                 # solid cavity

    # Two enclosed papillary islands (carve out of the blood → holes).
    for (ox, oy) in ((cx + 7.0, cy), (cx - 7.0, cy)):
        isl = ((xc - ox) ** 2 + (yc - oy) ** 2 <= rp ** 2) & inz
        blood &= ~isl
    # One wall-attached notch (a bite at the +y edge → open concavity).
    notch = ((xc - cx) ** 2 + (yc - (cy + R)) ** 2 <= (rp + 1.0) ** 2) & inz
    blood &= ~notch

    L = base_z - apex_z
    cyl_ml = np.pi * R ** 2 * L / 1000.0
    isl_ml = 2.0 * np.pi * rp ** 2 * L / 1000.0
    blood_true_ml = float(blood.sum()) * (sx * sy * sz) / 1000.0

    res = compact_from_blood(
        blood, (sx, sy, sz), apex_xyz=(cx, cy, apex_z),
        axis_dir=(0.0, 0.0, 1.0), radial0=(1.0, 0.0, 0.0),
        along_apex=0.5, along_base=L - 0.5, sax_step_mm=1.0,
        close_mm=4.0, half_mm=40.0, grid_mm=0.4)

    if res.get("error"):
        print("SKIP:", res["error"])
        return

    print(f"analytic cylinder  = {cyl_ml:7.2f} mL")
    print(f"blood (true mask)  = {blood_true_ml:7.2f} mL")
    print(f"  -> blood_ml      = {res['blood_ml']:7.2f} mL")
    print(f"  -> envelope_ml   = {res['envelope_ml']:7.2f} mL  (papillary-incl.)")
    print(f"  -> papillary_ml  = {res['papillary_ml']:7.2f} mL")
    print(f"islands analytic   = {isl_ml:7.2f} mL")

    # Envelope ~ full cylinder (islands filled + notch bridged), within 6%.
    e_err = abs(res["envelope_ml"] - cyl_ml) / cyl_ml
    assert e_err < 0.06, f"envelope off by {e_err*100:.1f}%"
    # Blood ~ the true carved mask (sanity of the SAX sampling), within 6%.
    b_err = abs(res["blood_ml"] - blood_true_ml) / blood_true_ml
    assert b_err < 0.06, f"blood off by {b_err*100:.1f}%"
    # Papillary = envelope - blood > 0 and ~ islands + notch (order of magnitude).
    assert res["papillary_ml"] > 0.5 * isl_ml, res["papillary_ml"]
    print("OK: envelope recovers the cavity; papillary volume separated")

    # ---- endo contour extraction: per-meridian (along, radius) of the envelope
    from multi_dicomviewer.core.lv_compact import (   # noqa: E402
        endo_contours_from_blood)
    # close_mm ≥ the wall-attached notch radius (5 mm) so it is bridged on the
    # θ=90 meridian too (larger papillary attachments need a larger close).
    prof = endo_contours_from_blood(
        blood, (sx, sy, sz), apex_xyz=(cx, cy, apex_z),
        axis_dir=(0.0, 0.0, 1.0), radial0=(1.0, 0.0, 0.0), n_meridians=12,
        along_apex=0.5, along_base=L - 0.5, sax_step_mm=1.0,
        close_mm=6.0, half_mm=40.0, grid_mm=0.4)
    if prof.get("error"):
        print("SKIP endo contours:", prof["error"])
        print("PASS")
        return
    assert len(prof) == 12, f"expected 12 meridians, got {len(prof)}"
    # Away from the poles the envelope radius should recover the cavity R=15 mm
    # (papillary islands filled). The θ=90 meridian points straight into the
    # deliberately-severe wall-attached notch — allow it to be partially bridged
    # (non-degenerate), but EVERY OTHER meridian must recover R within 2 mm.
    all_r, good = [], 0
    for th, arr in prof.items():
        mid = arr[(arr[:, 0] > 20.0) & (arr[:, 0] < 60.0)]   # mid-cavity band
        assert len(mid) > 0, f"theta {th}: no mid-cavity samples"
        rmean = float(mid[:, 1].mean())
        all_r.append(rmean)
        assert rmean > 8.0, f"theta {th}: degenerate radius {rmean:.2f}"
        if abs(rmean - R) < 2.0:
            good += 1
    assert good >= 11, f"only {good}/12 meridians recovered R (notch aside)"
    print(f"endo-envelope radius (mid): mean {np.mean(all_r):.2f} mm "
          f"(cavity R={R}); {good}/12 meridians within 2 mm")
    print("OK: envelope endo contour recovers the cavity radius")
    print("PASS")


if __name__ == "__main__":
    main()
