"""Modality-agnostic measurement model.

Shared so the same distance/angle tools work in both viewers and so a
measurement can later be serialized per study regardless of which viewer
created it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Measurement:
    #: One of "distance" / "angle" (legacy 2-pt / 3-pt tools), or any of
    #: the richer kinds the ImageCanvas emits now ("Line" / "Polyline" /
    #: "Ellipse" / "Polygon"). The history dialog renders `kind: label`,
    #: so any string works.
    kind: str
    points: list[tuple[float, float]] = field(default_factory=list)
    spacing_mm: Optional[tuple[float, float]] = None  # (row, col) mm/px
    #: Optional source-measure id (the viewer's per-result sequence number).
    #: Lets the history store drop a stale entry when a trace is "resumed"
    #: (un-committed to keep extending it), so re-committing doesn't duplicate.
    mid: Optional[int] = None
    #: Pre-formatted label (used when the emitter already built the full
    #: metrics string — e.g. polygon Area + Dmax/Dmin). label() returns
    #: this verbatim when set, otherwise it falls back to the legacy
    #: distance/angle formatters below.
    text: Optional[str] = None

    def value(self) -> Optional[float]:
        if self.kind == "distance" and len(self.points) >= 2:
            return self._distance(self.points[0], self.points[1])
        if self.kind == "angle" and len(self.points) >= 3:
            return self._angle(*self.points[:3])
        return None

    def label(self) -> str:
        if self.text:
            return self.text
        v = self.value()
        if v is None:
            return ""
        if self.kind == "distance":
            if self.spacing_mm is not None:
                return f"{v:.2f} mm"
            return f"{v:.1f} px"
        return f"{v:.1f}°"

    def _distance(self, a, b) -> float:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        if self.spacing_mm is not None:
            sy, sx = self.spacing_mm
            return math.hypot(dx * sx, dy * sy)
        return math.hypot(dx, dy)

    @staticmethod
    def _angle(a, b, c) -> float:
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cos = max(-1.0, min(1.0, cos))
        return math.degrees(math.acos(cos))
