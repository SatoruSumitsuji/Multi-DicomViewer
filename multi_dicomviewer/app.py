"""QApplication bootstrap."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from multi_dicomviewer.config import APP_NAME
from multi_dicomviewer.ui.main_window import MainWindow


def main(argv: list[str]) -> int:
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)

    initial = argv[1] if len(argv) > 1 else None
    win = MainWindow(initial_folder=initial)
    win.showMaximized()
    return app.exec()
