"""Headless test: compute LV function metrics from a BldLv-style dict alone."""
import base64
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_dicomviewer.core.lv_function import LVFunction  # noqa: E402


def _pack(comp, bbox, vol_shape):
    comp = np.ascontiguousarray(comp, bool)
    return {"bbox": [int(x) for x in bbox],
            "shape": [int(s) for s in comp.shape],
            "vol_shape": [int(s) for s in vol_shape],
            "packed": base64.b64encode(
                zlib.compress(np.packbits(comp).tobytes(), 6)).decode("ascii")}


def test_function_from_file():
    # Concentric spheres: endo r=20, epi r=30 (1 mm voxels).
    N = 80
    c = N / 2.0
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
    r = np.sqrt((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2)
    endo = r <= 20.0
    epi = r <= 30.0
    vshape = (N, N, N)
    # bbox = tight box of each mask (full volume here for simplicity)
    bb = (0, N, 0, N, 0, N)
    data = {
        "spacing": [1.0, 1.0, 1.0],           # (sx, sy, sz)
        "endo": _pack(endo, bb, vshape),
        "epi": _pack(epi, bb, vshape),
        "axis": {"apex": [c, c, 5.0], "dir": [0.0, 0.0, 1.0],
                 "radial0": [1.0, 0.0, 0.0]},
    }
    fn = LVFunction.from_json(data)
    assert fn is not None
    s = fn.summary()
    # true cavity = 4/3 π 20³ ≈ 33510 mL(voxels); myo = 4/3π(30³−20³) ≈ 79587
    cav_true = 4.0 / 3.0 * np.pi * 20 ** 3 / 1000.0
    myo_true = 4.0 / 3.0 * np.pi * (30 ** 3 - 20 ** 3) / 1000.0
    print(f"cavity={s['cavity_ml']:.1f} (true {cav_true:.1f})  "
          f"myo={s['myo_ml']:.1f} (true {myo_true:.1f})  "
          f"mass={s['myo_mass_g']:.1f} g")
    assert abs(s["cavity_ml"] - cav_true) / cav_true < 0.02
    assert abs(s["myo_ml"] - myo_true) / myo_true < 0.02
    assert abs(s["myo_mass_g"] - myo_true * 1.05) < 1.0
    # 3-D nearest-distance wall ≈ 10 mm on a sphere. (The radial/短軸 method is
    # for LV-like walls parallel to the axis — on a SPHERE it over-reads near the
    # poles, which is expected; its value is validated on a cylinder in
    # test_wall_thickness. Here we only confirm the file→metric pipeline runs.)
    print(f"wall_3d mean={s['wall_3d']['mean']:.2f} (true 10)  "
          f"wall_sax mean={s['wall_sax']['mean']:.2f} (sphere → higher)")
    assert abs(s["wall_3d"]["mean"] - 10.0) <= 1.5
    assert s["wall_sax"] is not None and s["wall_sax"]["mean"] > 0.0
    # missing axis → radial returns None, 3-D still works
    data2 = dict(data)
    data2.pop("axis")
    fn2 = LVFunction.from_json(data2)
    assert fn2.wall_thickness("sax") is None
    assert fn2.wall_thickness("3d") is not None
    print("LV FUNCTION OK")


if __name__ == "__main__":
    test_function_from_file()
