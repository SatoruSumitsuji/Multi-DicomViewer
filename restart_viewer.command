#!/bin/sh
# ============================================================================
#  Multi-DICOMviewer — one-click recovery (macOS)
#
#  Double-click this when the viewer has frozen. It force-closes the hung
#  viewer and reopens it, with NO Activity Monitor / Force Quit dialog.
#
#  Works both in a downloaded release (next to Multi-DicomViewer.app) and in a
#  source checkout (next to run.py): it launches the app with "--restart",
#  which runs as a small separate process and so works even while the main
#  window is completely frozen — it kills the recorded viewer PID and
#  relaunches it.
#
#  First-time setup for a Desktop/Dock icon: drag this file to the Desktop (or
#  Dock), or right-click > Make Alias and move the alias to the Desktop. The
#  alias persists; you only set it up once. (macOS may ask to confirm opening
#  it the first time: right-click > Open.)
# ============================================================================
cd "$(dirname "$0")"
APP="$(/bin/ls -d *.app 2>/dev/null | head -n 1)"
if [ -n "$APP" ]; then
    open -n "./$APP" --args --restart
elif command -v python3 >/dev/null 2>&1; then
    python3 run.py --restart
else
    python run.py --restart
fi
