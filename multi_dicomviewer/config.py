"""Static config and clinical window/level presets."""

import functools
import os
import subprocess
import sys  # noqa: F401  (kept for platform-specific config if needed)

# macOS now renders CT via the pygfx (wgpu→Metal) viewer, so the Mac build
# ships full CT again — same app name and no CT block on every platform.
APP_NAME = "Multi-DICOMviewer"
APP_VERSION = "2.0.6"


def _run_git(root, *args, timeout=5):
    """Return the stripped stdout of `git -C root <args>`, or None on any
    failure (git missing, not a repo, timeout)."""
    try:
        out = subprocess.run(
            ["git", "-C", root, *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:                                    # noqa: BLE001
        return None


def _git_build_info():
    """(short_hash, branch, dirty) from the source checkout, or None when
    not running from a git repo (e.g. a frozen build). *dirty* is None when
    it couldn't be determined (e.g. `git status` timed out) — distinct from
    False (confirmed clean)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = _run_git(root, "rev-parse", "--short", "HEAD")
    if not h:
        return None
    branch = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD") or ""
    status = _run_git(root, "status", "--porcelain")
    dirty = None if status is None else bool(status)
    return h, branch, dirty


def _baked_build_info():
    """(short_hash, branch, dirty) from a build stamp CI writes at package
    time (multi_dicomviewer/_build.py), or None when absent."""
    try:
        from multi_dicomviewer import _build   # written by CI; absent in dev
    except Exception:                                    # noqa: BLE001
        return None
    h = getattr(_build, "BUILD_HASH", "")
    if not h:
        return None
    return h, getattr(_build, "BUILD_BRANCH", ""), \
        bool(getattr(_build, "BUILD_DIRTY", False))


def _source_mtime():
    """Newest .py modification time under the package, formatted local
    ``MM-DD HH:MM:SS`` — a monotone 'source generation' stamp so repeated
    uncommitted edits are distinguishable (a hash+``*`` alone can't tell
    edit round 1 from round 5). '' if it can't be read."""
    import time
    root = os.path.dirname(os.path.abspath(__file__))
    newest = 0.0
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".py"):
                try:
                    m = os.path.getmtime(os.path.join(dirpath, fn))
                except OSError:
                    continue
                if m > newest:
                    newest = m
    return time.strftime("%m-%d %H:%M:%S", time.localtime(newest)) \
        if newest > 0 else ""


@functools.lru_cache(maxsize=1)
def build_string():
    """Self-identifying build id: version + git short hash + branch, plus
    ``*`` and a source-modification timestamp when the working tree is
    dirty, e.g. ``v1.11.4 (feature/x @ a1b2c3d* src 07-06 15:42:03)``. The
    timestamp advances with every edit, so you can tell whether a relaunch
    picked up the latest changes — which the hash+``*`` alone cannot show.
    Shown in the window title. Falls back to a baked CI stamp (frozen
    builds), then the bare version. Cached — computed once per process."""
    base = f"v{APP_VERSION}"
    info = _git_build_info() or _baked_build_info()
    src = _source_mtime()
    if not info:
        # No git and no baked stamp — the source mtime is the only build id
        # we have, so always show it (this is the git-not-on-PATH case).
        return f"{base} src {src}" if src else base
    h, branch, dirty = info
    star = "*" if dirty else ""
    # Show the source time whenever the tree is dirty OR its state is unknown
    # (git status missing/timed out) — exactly when the hash alone can't tell
    # you whether you're running the latest edits. A confirmed-clean checkout
    # (dirty is False) is fully identified by the hash, so src is omitted.
    gen = f" src {src}" if (dirty in (True, None) and src) else ""
    branchpart = f"{branch} @ " if (branch and branch != "HEAD") else ""
    return f"{base} ({branchpart}{h}{star}){gen}"

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
