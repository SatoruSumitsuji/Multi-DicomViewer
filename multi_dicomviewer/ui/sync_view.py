"""SyncView: a live link between TWO 3-D CT viewers shown side by side so that
mouse / keyboard view operations in EITHER are mirrored to the other, for
"同一操作で2つの3DCTを比較" (compare two CTs under identical manipulation).

The two CTs are DIFFERENT scans (different orientation, spacing, size, origin),
so operations are mirrored as an incremental DELTA — pan by the same screen
gesture, rotate by the same degrees, zoom by the same factor, page by the same
slice count, window/level by the same step — NOT as an absolute state copy
(which would be meaningless across unregistered volumes). The user first aligns
the two views manually, then turns SyncView on; from then on every gesture is
echoed to the peer.

On entry the shell offers "同じ縮尺にしますか？" — if accepted, both viewers are
forced to a common zoom (VTK ParallelScale) so the delta mirroring keeps them at
1:1 scale.

Viewer contract (see CTViewer.sync_view_* / apply_sync_op):
  set_sync_view_on(bool), sync_view_available() -> bool,
  sync_view_op(kind, params)  [signal],  apply_sync_op(kind, params),
  sync_view_get_scale(), sync_view_set_scale(ps).
"""
from __future__ import annotations

from PyQt6.QtCore import QObject


class SyncViewLink(QObject):
    """Bidirectional view-operation mirror between two CT viewers. A re-entrancy
    guard (``_syncing``) plus the viewers' own ``_sync_view_applying`` flag stop
    the mirror from echoing back into a feedback loop."""

    def __init__(self, viewer_a, viewer_b, parent=None):
        super().__init__(parent)
        self._v = [viewer_a, viewer_b]
        self._syncing = False
        for i, v in enumerate(self._v):
            v.set_sync_view_on(True)
            # default-arg binds the source index at connect time.
            v.sync_view_op.connect(
                lambda kind, params, src=i: self._on_op(src, kind, params))

    # -- linkage ----------------------------------------------------------
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

    # -- same-scale option ------------------------------------------------
    def match_scale(self) -> None:
        """Force viewer B to viewer A's zoom so both start at 1:1 scale (the
        "同じ縮尺にしますか？ → はい" path)."""
        try:
            ps = self._v[0].sync_view_get_scale()
            if ps:
                self._v[1].sync_view_set_scale(ps)
        except Exception:                                # noqa: BLE001
            pass

    def viewers(self) -> list:
        return list(self._v)

    def teardown(self) -> None:
        """Drop the link: disconnect and leave SyncView on both viewers."""
        for v in self._v:
            try:
                v.sync_view_op.disconnect()
            except Exception:                            # noqa: BLE001
                pass
            try:
                v.set_sync_view_on(False)
            except Exception:                            # noqa: BLE001
                pass
        self._v = []
