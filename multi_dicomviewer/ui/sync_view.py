"""SyncView: a live link between TWO 3-D CT viewers shown side by side so that
mouse / keyboard operations in EITHER are mirrored to the other, for comparing
two CTs under identical manipulation (e.g. the same case at MidDiastole vs
EndSystole).

Two layers of mirroring:

1. VIEW OPERATIONS (always) — pan / rotate / zoom / window-level / paging /
   thickness are echoed as an incremental DELTA (the two scans are unregistered,
   so an absolute copy is meaningless). The user aligns both views manually,
   then turns SyncView on; from then on every gesture is echoed to the peer.
   On entry the shell offers to match the zoom so the deltas keep the pair at
   1:1 scale.

2. LV SHORT-AXIS LEVEL (when BOTH viewers are in LV-volume analysis with a valid
   long axis) — paging the short-axis level along the apex→MV-centre axis is
   mirrored, either as an absolute mm-from-apex ("mm" mode) or as an apex→base
   FRACTION ("按分" mode, so the two phases align by RELATIVE depth despite
   different LV lengths — the clinically meaningful comparison). This reuses the
   viewers' lv_cosync_* contract (built for the retired LV-CoSync).

Both layers are re-entrancy guarded so a mirrored change never echoes back.
"""
from __future__ import annotations

from PyQt6.QtCore import QObject


class SyncViewLink(QObject):
    def __init__(self, viewer_a, viewer_b, parent=None):
        super().__init__(parent)
        self._v = [viewer_a, viewer_b]
        self._syncing = False
        self._level_fraction = False            # False = mm, True = 按分 (fraction)
        # --- view-operation mirror ---
        for i, v in enumerate(self._v):
            v.set_sync_view_on(True)
            v.sync_view_op.connect(
                lambda kind, params, src=i: self._on_op(src, kind, params))
        # --- LV short-axis level link (only if BOTH are LV-volume with an axis) ---
        self._level_on = all(
            hasattr(v, "lv_cosync_available") and v.lv_cosync_available()
            for v in self._v)
        if self._level_on:
            for i, v in enumerate(self._v):
                v.set_lv_cosync_on(True)
                v.lv_cosync_level_changed.connect(
                    lambda mm, src=i: self._on_level(src, mm))

    # -- view-operation mirror -------------------------------------------
    def _on_op(self, src: int, kind: str, params) -> None:
        if self._syncing:
            return
        vd = self._v[1 - src]
        self._syncing = True
        try:
            vd.apply_sync_op(kind, dict(params))
        except Exception:                                # noqa: BLE001
            pass
        finally:
            self._syncing = False

    # -- LV short-axis level link ----------------------------------------
    def level_link_active(self) -> bool:
        return self._level_on

    @property
    def level_fraction(self) -> bool:
        return self._level_fraction

    def set_level_fraction(self, on: bool) -> None:
        """Switch the level mirror between mm (False) and apex→base fraction /
        按分 (True), and re-apply from viewer 0 so the peer reflects it at once."""
        self._level_fraction = bool(on)
        if not self._level_on:
            return
        try:
            mm = self._v[0].lv_cosync_level_mm()
        except Exception:                                # noqa: BLE001
            mm = None
        if mm is not None:
            self._on_level(0, float(mm))

    def _on_level(self, src: int, mm: float) -> None:
        if self._syncing or not self._level_on:
            return
        vs, vd = self._v[src], self._v[1 - src]
        self._syncing = True
        try:
            if self._level_fraction:
                frac = vs.lv_cosync_level_fraction()
                if frac is not None:
                    vd.lv_cosync_set_level_fraction(float(frac), silent=True)
            else:
                vd.lv_cosync_set_level_mm(float(mm), silent=True)
        except Exception:                                # noqa: BLE001
            pass
        finally:
            self._syncing = False

    # -- same-scale option -----------------------------------------------
    def match_scale(self) -> None:
        """Force viewer B to viewer A's zoom so both are at 1:1 scale."""
        try:
            ps = self._v[0].sync_view_get_scale()
            if ps:
                self._v[1].sync_view_set_scale(ps)
        except Exception:                                # noqa: BLE001
            pass

    def viewers(self) -> list:
        return list(self._v)

    def teardown(self) -> None:
        """Drop the link: disconnect both layers and leave SyncView / the level
        link on both viewers."""
        for v in self._v:
            for sig in ("sync_view_op", "lv_cosync_level_changed"):
                try:
                    getattr(v, sig).disconnect()
                except Exception:                        # noqa: BLE001
                    pass
            try:
                v.set_sync_view_on(False)
            except Exception:                            # noqa: BLE001
                pass
            try:
                v.set_lv_cosync_on(False)
            except Exception:                            # noqa: BLE001
                pass
        self._v = []
