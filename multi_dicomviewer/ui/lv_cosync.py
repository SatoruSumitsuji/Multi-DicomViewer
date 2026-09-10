"""LV-CoSync: a live link between TWO CT viewers (typically Diastole + Systole)
so their LV short-axis views page together and share one scale.

This is the CT-主体 branch of Tools ▸ CoSync (the IVUS/XA branch is the separate
CoregWindow). It is a LIVE link on the main panes — not a snapshot window: when
one CT is paged along its LV long axis, the peer follows.

Phase 1 links: same scale (VTK ParallelScale) + along-LV-axis paging. By default
the level is mirrored in absolute mm from the apex (so a DIFFERENCE in LV length
between phases stays visible — a real finding); with 左室長補正 (LV-length
correction) ON, the level is mirrored as an apex→base FRACTION so the two align
by relative depth.

The viewers expose the contract (see CTViewer.lv_cosync_*):
  set_lv_cosync_on(bool), lv_cosync_available() -> bool,
  lv_cosync_level_changed(float mm)  [signal],
  lv_cosync_set_level_mm(mm, silent=), lv_cosync_level_mm(),
  lv_cosync_level_fraction(), lv_cosync_set_level_fraction(f, silent=),
  lv_cosync_get_scale(), lv_cosync_set_scale(ps).
"""
from __future__ import annotations

from PyQt6.QtCore import QObject


class LvCosyncLink(QObject):
    """Bidirectional level+scale link between two CT viewers. A re-entrancy guard
    (``_syncing``) stops the mirror from echoing back into a feedback loop."""

    def __init__(self, viewer_a, viewer_b, parent=None):
        super().__init__(parent)
        self._v = [viewer_a, viewer_b]
        self._syncing = False
        self._length_correct = False          # 左室長補正: False = mm, True = fraction
        for i, v in enumerate(self._v):
            v.set_lv_cosync_on(True)
            # default-arg binds the source index at connect time.
            v.lv_cosync_level_changed.connect(
                lambda mm, src=i: self._on_level(src, mm))
        self._lock_scale()

    # -- state ------------------------------------------------------------
    @property
    def length_correct(self) -> bool:
        return self._length_correct

    def set_length_correct(self, on: bool) -> None:
        """Toggle 左室長補正 (mm ↔ apex→base fraction) and re-apply from viewer 0
        so the peer immediately reflects the new mapping."""
        self._length_correct = bool(on)
        try:
            lvl = self._v[0].lv_cosync_level_mm()
        except Exception:                                # noqa: BLE001
            lvl = None
        if lvl is not None:
            self._on_level(0, float(lvl))

    # -- linkage ----------------------------------------------------------
    def _on_level(self, src: int, mm: float) -> None:
        if self._syncing:
            return
        dst = 1 - src
        vs, vd = self._v[src], self._v[dst]
        self._syncing = True
        try:
            if self._length_correct:
                frac = vs.lv_cosync_level_fraction()
                if frac is not None:
                    vd.lv_cosync_set_level_fraction(float(frac), silent=True)
            else:
                vd.lv_cosync_set_level_mm(float(mm), silent=True)
        except Exception:                                # noqa: BLE001
            pass
        finally:
            self._syncing = False

    def _lock_scale(self) -> None:
        """Force the peer to the driver's short-axis zoom so both are 1:1 scale."""
        try:
            ps = self._v[0].lv_cosync_get_scale()
            if ps:
                self._v[1].lv_cosync_set_scale(ps)
        except Exception:                                # noqa: BLE001
            pass

    def relock_scale(self) -> None:
        """Re-apply the same-scale lock (e.g. after a zoom on the driver)."""
        self._lock_scale()

    def viewers(self) -> list:
        return list(self._v)

    def teardown(self) -> None:
        """Drop the link: disconnect and leave CoSync on both viewers."""
        for v in self._v:
            try:
                v.lv_cosync_level_changed.disconnect()
            except Exception:                            # noqa: BLE001
                pass
            try:
                v.set_lv_cosync_on(False)
            except Exception:                            # noqa: BLE001
                pass
        self._v = []
