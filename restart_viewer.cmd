@echo off
REM ============================================================================
REM  Multi-DICOMviewer — one-click recovery
REM
REM  Double-click this (or a desktop SHORTCUT to it) when the viewer has frozen.
REM  It force-closes the hung viewer and reopens it, with NO Task Manager.
REM
REM  How it works: launches the app with "--restart", which runs as a small
REM  separate process — so it works even while the main window is completely
REM  frozen. That helper kills the recorded viewer PID and relaunches it.
REM
REM  To make a desktop icon: right-click this file > Send to > Desktop
REM  (create shortcut). Optionally rename the shortcut to "ビューア再起動".
REM
REM  NOTE: this dev launcher uses `python run.py`. The packaged release should
REM  instead make the shortcut target  "Multi-DicomViewer.exe --restart"
REM  (same logic, no Python needed).
REM ============================================================================
cd /d "%~dp0"
start "" python run.py --restart
exit
