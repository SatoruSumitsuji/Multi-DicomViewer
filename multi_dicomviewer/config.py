"""Static config and clinical window/level presets."""

import sys

# Mac build does not ship the CT viewer (VTK's OpenGL→Metal path hangs on
# macOS — see the Mac mini M4 / pygfx-port plan). The Mac variant is
# renamed so users can tell at a glance it does not include CT.
if sys.platform == "darwin":
    APP_NAME = "Multi-DicomViewer-NoCT"
else:
    APP_NAME = "Multi-DICOMviewer"
APP_VERSION = "1.0"

# True when CT loading should be blocked at the UI level (Mac build).
BLOCK_CT = sys.platform == "darwin"
BLOCK_CT_MESSAGE = (
    "このアプリはCTデータに対応していません。\n"
    "（Mac版はCTビューワーを搭載していません）"
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
