"""Static config and clinical window/level presets."""

import sys  # noqa: F401  (kept for platform-specific config if needed)

# macOS now renders CT via the pygfx (wgpu→Metal) viewer, so the Mac build
# ships full CT again — same app name and no CT block on every platform.
APP_NAME = "Multi-DICOMviewer"
APP_VERSION = "1.8.5"

# CT loads on all platforms now (Windows/Linux: VTK; macOS: pygfx). Kept as a
# constant so the UI gate below stays in place should a build ever need it.
BLOCK_CT = False
BLOCK_CT_MESSAGE = (
    "This build does not support CT data.\n"
    "(The Mac build has no CT viewer.)"
)

# Hounsfield-unit window/level presets for the CT viewer (W, L).
CT_WL_PRESETS = {
    "Coronary / Angio": (800, 200),
    "Mediastinum": (400, 40),
    "Lung": (1500, -600),
    "Bone": (2000, 500),
    "Soft tissue": (350, 50),
}

# Default cine frame rate (frames/sec) when DICOM does not specify one.
DEFAULT_CINE_FPS = 15.0
