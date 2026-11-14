"""export.open_mp4_stream(preset=…): the x264 speed preset is passed through
(MultiReg uses "veryfast", ≈2× "medium" at the same CRF); other exports keep
"medium". Synthetic frames only.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from multi_dicomviewer.core import export  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


seen = []
real = export._make_writer


def spy(path, fps, br, crf, preset="medium"):
    seen.append(preset)
    return real(path, fps, br, crf, preset)


export._make_writer = spy
tmp = tempfile.mkdtemp(prefix="mdv_mp4p_")
rng = np.random.default_rng(1)
frames = [rng.integers(0, 255, (120, 160, 3), np.uint8) for _ in range(12)]
try:
    path = os.path.join(tmp, "v.mp4")
    s = export.open_mp4_stream(path, fps=15, bitrate_mbps=10, crf=18,
                               preset="veryfast")
    for f in frames:
        s.add(f)
    s.close()
    check("veryfast passed to the writer", seen == ["veryfast"], str(seen))
    check("valid MP4 written", os.path.isfile(path)
          and open(path, "rb").read(12)[4:8] == b"ftyp")
    try:
        import imageio.v2 as imageio
        r = imageio.get_reader(path)
        n = sum(1 for _ in r)
        r.close()
        check("every frame is in the file", n == len(frames), str(n))
    except Exception as e:                              # noqa: BLE001
        print("SKIP frame count:", e)
    p2 = export.open_mp4_stream(os.path.join(tmp, "m.mp4"), 15, 10, crf=18)
    p2.abort()
    check("default preset stays medium (other exports unchanged)",
          seen[-1] == "medium")
finally:
    export._make_writer = real

src = (ROOT / "multi_dicomviewer/ui/coreg_window.py").read_text(encoding="utf-8")
check("MultiReg MP4 uses veryfast",
      '_MP4_PRESET = "veryfast"' in src and "preset=self._MP4_PRESET" in src)

print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
