#!/bin/sh
# ============================================================================
#  Multi-DICOMviewer — one-click recovery (macOS)
#
#  Double-click this when the viewer has frozen. It force-closes the hung
#  viewer and reopens it, with NO Activity Monitor / Force Quit dialog.
#
#  How it works: launches the app with "--restart", which runs as a small
#  separate process — so it works even while the main window is completely
#  frozen. That helper kills the recorded viewer PID and relaunches it.
#
#  First-time setup:
#    1) Make it executable (once):   chmod +x restart_viewer.command
#    2) For a Desktop/Dock icon: drag this file to the Desktop (or Dock), or
#       right-click > Make Alias and move the alias to the Desktop.
#       The alias persists; you only set it up once.
#
#  NOTE: this dev launcher uses `python3 run.py`. A packaged .app should
#  instead relaunch via the bundle (e.g. `open -n MultiDicomViewer.app
#  --args --restart`).
# ============================================================================
cd "$(dirname "$0")"
# Prefer python3; fall back to python.
if command -v python3 >/dev/null 2>&1; then
    python3 run.py --restart
else
    python run.py --restart
fi
