"""MP4 export diagnostic — reproduces the MultiSync composite encode with
synthetic frames using the SAME code path the app uses, so we can see the
real error if 'Export MP4' fails. Run it with the same Python you launch
the app with:

    python tools/diag_mp4.py

It writes a few hundred 1024x1024 frames to C:\\Temp via the project's
encoder and prints PASS / FAIL with the full traceback.
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

# Make the repo importable when run from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.core import export                      # noqa: E402
from multi_dicomviewer.ui.multisync_window import MultiSyncWindow  # noqa: E402

print("python   :", sys.executable)
import imageio, imageio_ffmpeg                                  # noqa: E402
print("imageio  :", imageio.__version__)
print("ffmpeg   :", imageio_ffmpeg.__version__,
      imageio_ffmpeg.get_ffmpeg_exe())

out_dir = r"C:\Temp"
os.makedirs(out_dir, exist_ok=True)

w = MultiSyncWindow([], layout_count=4)
rng = np.random.default_rng(0)


class _FakePlane:
    total_frames = 2000

    def frame(self, i):
        return rng.integers(0, 255, (512, 512)).astype(np.uint8)


class _S:
    pass


def _state(k):
    sl = _S()
    sl.plane = _FakePlane()
    sl.total = 2000
    return (k, sl, 0, 20.0 * k)


states = [_state(k) for k in range(4)]
cell, cols, W, H = 512, 2, 1024, 1024

# Verify the composite array is an owned copy (the bug was returning a view
# aliasing a freed QImage buffer).
rgb = w._compose_frame_qt(states, W, H, cell, cols)
print("frame shape/dtype:", rgb.shape, rgb.dtype,
      "| OWNDATA:", rgb.flags["OWNDATA"], "| base None:", rgb.base is None)

N = 400
for crf, name in [(18, "crf18"), (None, "bitrate40")]:
    out = os.path.join(out_dir, f"diag_{name}.mp4")
    try:
        s = export.open_mp4_stream(out, fps=15, bitrate_mbps=40, crf=crf)
        for i in range(N):
            s.add(w._compose_frame_qt(states, W, H, cell, cols))
            if i % 8 == 0:
                app.processEvents()
        s.close()
        print(f"PASS {name}: {N} frames -> {os.path.getsize(out)//1024} KB"
              f"  ({out})")
    except Exception:
        print(f"FAIL {name}:")
        traceback.print_exc()

print("done.")
