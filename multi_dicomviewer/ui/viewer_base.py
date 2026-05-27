"""Abstract viewer interface.

Both the XA and CT modules implement this so the shell can route a series
to whichever viewer fits its modality without knowing the internals.
"""
from __future__ import annotations

from abc import abstractmethod

from PyQt6.QtWidgets import QWidget

from multi_dicomviewer.core.dicom_io import LoadedSeries


class AbstractViewer(QWidget):
    #: Modality string this viewer handles ("XA" or "CT").
    handles_modality: str = ""

    @abstractmethod
    def load_series(self, loaded: LoadedSeries, title: str) -> None:
        """Display a freshly loaded series."""

    @abstractmethod
    def clear(self) -> None:
        """Release the current series and blank the view."""
