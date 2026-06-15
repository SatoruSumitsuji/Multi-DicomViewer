@echo off
REM ============================================================================
REM  Multi-DICOMviewer — one-click recovery (Windows)
REM
REM  Double-click this (or a desktop SHORTCUT to it) when the viewer has frozen.
REM  It force-closes the hung viewer and reopens it, with NO Task Manager.
REM
REM  Works both in a downloaded release (next to Multi-DicomViewer.exe) and in
REM  a source checkout (next to run.py): it launches the app with "--restart",
REM  which runs as a small separate process and so works even while the main
REM  window is completely frozen — it kills the recorded viewer PID and
REM  relaunches it.
REM
REM  To make a desktop icon: right-click this file > Send to > Desktop
REM  (create shortcut). Optionally rename the shortcut to "ビューア再起動".
REM ============================================================================
cd /d "%~dp0"
if exist "Multi-DicomViewer.exe" (
    start "" "Multi-DicomViewer.exe" --restart
) else (
    start "" python run.py --restart
)
exit
