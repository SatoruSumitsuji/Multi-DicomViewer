"""Multi-DICOMviewer entry point.

Usage:
    python run.py [path-to-dicom-folder]
    python run.py --version      # print the current build id and exit
"""
import sys

if __name__ == "__main__":
    if any(a in ("--version", "-V") for a in sys.argv[1:]):
        # Print the SAME build string the window title shows, computed from
        # the source on disk right now — the authoritative "latest" to
        # compare a running window's title against (are they the same? then
        # the open app IS the latest; older? relaunch).
        from multi_dicomviewer.config import APP_NAME, build_string
        print(f"{APP_NAME}  {build_string()}")
        sys.exit(0)
    from multi_dicomviewer.app import main
    sys.exit(main(sys.argv))
