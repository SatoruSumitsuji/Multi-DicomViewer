"""Multi-DICOMviewer entry point.

Usage:
    python run.py [path-to-dicom-folder]
"""
import sys

from multi_dicomviewer.app import main

if __name__ == "__main__":
    sys.exit(main(sys.argv))
