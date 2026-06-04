"""Launch the app:  python -m multi_dicomviewer [initial_folder]"""
import sys

from multi_dicomviewer.app import main

raise SystemExit(main(sys.argv))
