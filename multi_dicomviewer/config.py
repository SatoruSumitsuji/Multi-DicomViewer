"""Static config and clinical window/level presets."""

APP_NAME = "Multi-DICOMviewer"
APP_VERSION = "1.0"

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
