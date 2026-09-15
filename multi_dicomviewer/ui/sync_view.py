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

from PyQt6.QtCore import QObject, QTimer


class SyncViewLink(QObject):
    def __init__(self, viewer_a, viewer_b, parent=None):
        super().__init__(parent)
        self._v = [viewer_a, viewer_b]
        self._syncing = False
        self._level_fraction = False            # False = mm, True = 按分 (fraction)
        # Paired-snapshot undo/redo (method A): the shell's Undo/Redo revert the
        # last SYNCED gesture on BOTH panes together. A short debounce coalesces
        # the many per-increment sync_view_op emits of one drag into ONE entry.
        self._undo = []                         # [(snapA, snapB), …] settled states
        self._redo = []
        self._on_undo_changed = None            # shell callback to refresh buttons
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(250)
        self._settle.timeout.connect(self._push_snapshot)
        self._restoring = False
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
        self._push_snapshot(force=True)         # base state (before any gesture)

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
        if not self._restoring:                 # coalesce this gesture's emits
            self._settle.start()

    # -- paired-snapshot undo / redo (method A) --------------------------
    def _snap_pair(self):
        try:
            return (self._v[0].capture_view_state(),
                    self._v[1].capture_view_state())
        except Exception:                                # noqa: BLE001
            return None

    def _push_snapshot(self, force: bool = False) -> None:
        """Record the settled state of BOTH panes as one undo entry."""
        pair = self._snap_pair()
        if pair is None:
            return
        self._undo.append(pair)
        if not force:
            self._redo.clear()
        if self._on_undo_changed:
            self._on_undo_changed()

    def _restore_pair(self, pair) -> None:
        self._restoring = True
        self._settle.stop()
        try:
            self._v[0].restore_view_state(dict(pair[0]))
            self._v[1].restore_view_state(dict(pair[1]))
        except Exception:                                # noqa: BLE001
            pass
        finally:
            self._restoring = False

    def can_undo(self) -> bool:
        return len(self._undo) >= 2

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> None:
        if len(self._undo) < 2:
            return
        self._redo.append(self._undo.pop())     # current → redo
        self._restore_pair(self._undo[-1])       # revert to the previous state
        if self._on_undo_changed:
            self._on_undo_changed()

    def redo(self) -> None:
        if not self._redo:
            return
        pair = self._redo.pop()
        self._undo.append(pair)
        self._restore_pair(pair)
        if self._on_undo_changed:
            self._on_undo_changed()

    def set_undo_callback(self, cb) -> None:
        self._on_undo_changed = cb

    def note_view_change(self) -> None:
        """A shared-toolbar action (transform / side / centreline …) changed the
        view directly (not via a drag) — schedule an undo snapshot for it too."""
        if not self._restoring:
            self._settle.start()

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
        if not self._restoring:                 # level paging is undoable too
            self._settle.start()

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
        self._settle.stop()
        self._undo.clear()
        self._redo.clear()
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
