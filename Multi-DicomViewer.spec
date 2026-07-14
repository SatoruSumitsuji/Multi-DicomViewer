# PyInstaller spec for Multi-DicomViewer (Windows + macOS).
#
#   Windows:  pyinstaller --noconfirm Multi-DicomViewer.spec
#   macOS:    pyinstaller --noconfirm Multi-DicomViewer.spec
#
# Outputs to dist/Multi-DicomViewer/  (onedir bundle).
# On macOS the same spec also produces dist/Multi-DicomViewer.app.

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)
import pathlib
import re
import sys

# --- Version for the macOS bundle's Info.plist. Read from the single source
#     of truth (config.APP_VERSION) rather than hard-coding it here, so a
#     release bump can't leave the .app's "Get Info" version stale. ---
def _app_version(default="0.0.0"):
    try:
        src = pathlib.Path("multi_dicomviewer/config.py").read_text(encoding="utf-8")
        m = re.search(r'^APP_VERSION\s*=\s*"([^"]+)"', src, re.M)
        if m:
            return m.group(1)
    except Exception:
        pass
    return default


APP_VERSION = _app_version()

# --- CT render backend differs per OS: VTK on Windows/Linux, pygfx + wgpu
#     (Metal) on macOS. Collect whichever is actually installed for THIS
#     build so the packaged app can import its CTViewer. wgpu/pygfx ship
#     native libs + shader (.wgsl) data that PyInstaller won't auto-detect,
#     so collect_all() is required. ---
if sys.platform == "darwin":
    ct_datas, ct_binaries, ct_hidden = [], [], []
    for _pkg in ("pygfx", "wgpu", "rendercanvas"):
        _d, _b, _h = collect_all(_pkg)
        ct_datas += _d
        ct_binaries += _b
        ct_hidden += _h
else:
    ct_datas = collect_data_files("vtkmodules")
    ct_binaries = collect_dynamic_libs("vtkmodules")
    ct_hidden = collect_submodules("vtkmodules")

# --- pylibjpeg: entry-point plugin discovery ---
plj_datas, plj_binaries, plj_hidden = collect_all("pylibjpeg")
pllj_datas, pllj_binaries, pllj_hidden = collect_all("pylibjpeg_libjpeg")
plop_datas, plop_binaries, plop_hidden = collect_all("pylibjpeg_openjpeg")

# --- imagecodecs: many C extensions, prefer the kitchen-sink helper ---
ic_datas, ic_binaries, ic_hidden = collect_all("imagecodecs")

# --- OpenCV (Angio/IVUS Smooth/Denoise): native libs PyInstaller won't fully
#     auto-detect. Optional — guarded so a build environment without cv2 still
#     packages (the app then just disables the quality toggles at runtime). ---
try:
    cv2_datas, cv2_binaries, cv2_hidden = collect_all("cv2")
except Exception:
    cv2_datas, cv2_binaries, cv2_hidden = [], [], []

# --- pydicom encoders/handlers (plugin discovery) ---
pyd_datas, pyd_binaries, pyd_hidden = collect_all("pydicom")

# --- imageio + imageio-ffmpeg (MP4 export). The bundled ffmpeg binary
#     lives inside the imageio_ffmpeg package; collect_all picks it up.
iio_datas, iio_binaries, iio_hidden = collect_all("imageio")
iiof_datas, iiof_binaries, iiof_hidden = collect_all("imageio_ffmpeg")

datas = (
    ct_datas + plj_datas + pllj_datas + plop_datas + ic_datas + cv2_datas
    + pyd_datas + iio_datas + iiof_datas
    # Resources shipped with the app — the DICOM-aware Rupture-Predictor
    # HTML is launched from Tools and must be present in the bundle.
    # Preserve the source-tree path so the dev and bundle layouts match.
    + [(
        "multi_dicomviewer/resources/Rupture-Predictor.html",
        "multi_dicomviewer/resources",
    )]
    # i18n tables. i18n.enabled_languages() offers a language ONLY if its
    # table file exists, so leaving these out of the bundle silently
    # degrades the packaged app to English-only. Glob, so a new
    # resources/i18n/<code>.json ships without touching this spec.
    + [(
        "multi_dicomviewer/resources/i18n/*.json",
        "multi_dicomviewer/resources/i18n",
    )]
)
binaries = (
    ct_binaries + plj_binaries + pllj_binaries + plop_binaries
    + ic_binaries + cv2_binaries + pyd_binaries + iio_binaries + iiof_binaries
)
hiddenimports = (
    ct_hidden
    + plj_hidden
    + pllj_hidden
    + plop_hidden
    + ic_hidden
    + cv2_hidden
    + pyd_hidden
    + iio_hidden
    + iiof_hidden
    + ["multi_dicomviewer"]
)

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Trim unused heavy deps if present in the environment
        "tkinter",
        "matplotlib",
        "PySide6",
        "PyQt5",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

# Native splash screen (Windows/Linux): the PyInstaller bootloader shows this
# image the moment it starts — BEFORE Python and PyQt6 load — so the user sees
# the app within a blink instead of staring at 2-3 blank seconds while the big
# Qt/DICOM DLLs cold-load and Defender scans them. macOS doesn't support the
# PyInstaller splash (and already launches instantly), so it's skipped there;
# app.py shows a Qt splash as the fallback when this native one isn't present.
if sys.platform != "darwin":
    splash = Splash(
        "packaging/splash.png",
        binaries=a.binaries,
        datas=a.datas,
        text_pos=None,
    )
    _exe_extra = [splash]
    _coll_extra = [splash.binaries]
else:
    _exe_extra = []
    _coll_extra = []

exe = EXE(
    pyz,
    a.scripts,
    *_exe_extra,
    [],
    exclude_binaries=True,
    name="Multi-DicomViewer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,        # windowed (no console)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    *_coll_extra,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Multi-DicomViewer",
)

# macOS: also produce a .app bundle from the same Analysis. CT renders
# on macOS via the pygfx (wgpu→Metal) viewer, so this ships the FULL app
# (Windows uses the VTK CT viewer, macOS the pygfx one) — no longer "-NoCT".
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Multi-DicomViewer.app",
        icon=None,
        bundle_identifier="org.research.multi-dicomviewer",
        info_plist={
            "NSHighResolutionCapable": "True",
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION,
        },
    )
