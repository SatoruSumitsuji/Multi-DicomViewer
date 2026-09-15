"""Abstract viewer interface.

Both the XA and CT modules implement this so the shell can route a series
to whichever viewer fits its modality without knowing the internals.

It also carries a small shared mechanism (:class:`ImageFloorMixin`) that keeps
at least half of a viewer's height for the image in multi-pane (compact) mode:
a viewer puts its bulky control bars into a height-cappable scroll area, and
the mixin caps that area on resize so the image never drops below the floor;
overflow scrolls instead of crushing the picture.
"""
from __future__ import annotations

from abc import abstractmethod

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QFrame, QScrollArea, QWidget

from multi_dicomviewer.core.dicom_io import LoadedSeries

#: Qt's QWIDGETSIZE_MAX — "no maximum" sentinel for setMaximumHeight.
_QWIDGETSIZE_MAX = 16777215


class _ChromeScrollArea(QScrollArea):
    """A scroll area whose sizeHint tracks its content's height, so the parent
    layout gives it exactly the content height until an explicit maximumHeight
    caps it — then it scrolls vertically. Flagged ``_mdv_chrome_scroll`` so the
    shell's "Max Image" walk recurses into it (toggling the inner bars) rather
    than hiding the whole container."""
    _mdv_chrome_scroll = True

    def _wants_hbar(self) -> bool:
        """The content is WIDER than the viewport → a horizontal scrollbar will
        appear (e.g. a wide control bar in a narrow multi-pane column)."""
        w = self.widget()
        if w is None:
            return False
        vw = self.viewport().width()
        return vw > 0 and w.sizeHint().width() > vw

    def sizeHint(self) -> QSize:
        base = super().sizeHint()
        w = self.widget()
        if w is None:
            return base
        h = w.sizeHint().height() + 2 * self.frameWidth()
        # When a horizontal scrollbar is needed it is drawn INSIDE the viewport
        # and steals from the bottom, hiding the last control row (the reported
        # "3rd row clipped only in a narrow 1×2 column"). Reserve its height so
        # the parent layout gives us enough room to show every row above it.
        if self._wants_hbar():
            h += self.horizontalScrollBar().sizeHint().height()
        return QSize(base.width(), h)

    def minimumSizeHint(self) -> QSize:
        s = super().minimumSizeHint()
        return QSize(s.width(), 0)          # allow shrinking below content

    def resizeEvent(self, e):  # noqa: N802 (Qt override)
        super().resizeEvent(e)
        # A width change can flip whether the horizontal scrollbar is needed,
        # which changes our height sizeHint — re-query it (only on a flip, to
        # avoid a relayout loop).
        over = self._wants_hbar()
        if over != getattr(self, "_mdv_over", None):
            self._mdv_over = over
            self.updateGeometry()


class ImageFloorMixin:
    """Reserve at least :attr:`IMAGE_FLOOR` of the widget height for the image
    in compact (multi-pane) mode.

    Opt in from a viewer by:
      * ``self._below_scroll = self._make_chrome_scroll(inner)`` and add it to
        the top-level layout in place of the bulky control bars;
      * set ``self._mdv_compact`` in ``set_compact`` and call
        ``self._apply_image_floor()``.
    Viewers that don't set ``_below_scroll`` are unaffected.
    """
    IMAGE_FLOOR = 0.5

    def _make_chrome_scroll(self, inner: QWidget) -> QScrollArea:
        sc = _ChromeScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        sc.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sc.setWidget(inner)
        return sc

    def _apply_image_floor(self) -> None:
        scroll = getattr(self, "_below_scroll", None)
        if scroll is None:
            return
        if not getattr(self, "_mdv_compact", False):
            scroll.setMaximumHeight(_QWIDGETSIZE_MAX)     # no cap in 1×1
            return
        lay = self.layout()
        if lay is None:
            return
        h = self.height()
        # Height taken by every OTHER top-level chrome item (toolbar, title,
        # readout …) that isn't the image (the stretch item) or our scroll.
        other = 0
        for i in range(lay.count()):
            if lay.stretch(i) > 0:
                continue                                  # the image area
            it = lay.itemAt(i)
            w = it.widget()
            if w is None or w is scroll or not w.isVisible():
                continue
            other += w.sizeHint().height()
        # image ≥ floor·h  ⇔  other + below ≤ (1-floor)·h reserved for chrome…
        # but we anchor to the image: below may use whatever is left under the
        # floor line, i.e. floor·h − other. (Floor 0.5 → image keeps ≥ half.)
        cap = int(h * self.IMAGE_FLOOR) - other
        scroll.setMaximumHeight(max(24, cap))

    def resizeEvent(self, e):  # noqa: N802 (Qt override)
        super().resizeEvent(e)
        self._apply_image_floor()


class AbstractViewer(ImageFloorMixin, QWidget):
    #: Modality string this viewer handles ("XA" or "CT").
    handles_modality: str = ""

    @abstractmethod
    def load_series(self, loaded: LoadedSeries, title: str) -> None:
        """Display a freshly loaded series."""

    @abstractmethod
    def clear(self) -> None:
        """Release the current series and blank the view."""
