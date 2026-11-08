# -*- coding: utf-8 -*-
"""MV / AoV plane — Auto step (shared by the VTK and pygfx CT viewers).

The valve-edit row's Auto button reads the rough position from the centreline
CROSSING (self._center, as Apex → Set does) and fits the plane with
core.valve_auto:

* AoV: two presses — (1) Auto → "put the crossing near the AoV centre",
  (2) Auto → detect (crossing in / near the aortic root).
* MV:  three presses, each prompted — (1) Auto → "put the crossing near the
  MV centre", (2) Auto → MV centre recorded, "put the crossing near the LV
  apex", (3) Auto → detect. The rough apex seeds the first rotation axis only
  (not stored as the Apex — the Epi step still sets that against the MV
  plane).

The result goes through _lv_apply_valve_data (same ring / caches as Load) and
is marked unsaved. Manual correction = the existing Draw → Set.
"""
from __future__ import annotations

import numpy as np

from multi_dicomviewer.i18n import t


class ValveAutoMixin:
    _lv_valve_auto_seed = None               # (which, "mv"|"apex", MV-centre mm)

    def _lv_valve_auto(self) -> None:
        which = getattr(self, "_lv_valve_edit", None) or "mitral"
        vol = getattr(self, "_vol", None)
        P = getattr(self, "_center", None)
        if vol is None or P is None:
            return
        P = np.asarray(P, float).copy()
        vname = "MV" if which == "mitral" else "AoV"
        from multi_dicomviewer.core import valve_auto as va
        dims = tuple(float(d) for d in self._dims)
        if which == "aortic":
            # Two presses: 1st asks for the crossing at the AoV centre, 2nd runs.
            seed = self._lv_valve_auto_seed
            if seed is None or seed[0] != which:
                self._lv_valve_auto_seed = (which, "aov", None)
                self._lvv_prompt(t(
                    "AoV中心近傍にセンターライン交点を合わせてAutoを押してください。"))
                return
            self._lv_valve_auto_seed = None
            mv = (self._lv_valves.get("mitral") if hasattr(self, "_lv_valves")
                  else None)
            hint = np.asarray(mv[0], float) if mv is not None else None
            res = self._run_busy(t("Auto"), t("{v}面を検出中…").format(v=vname),
                                 lambda: va.detect_aov(vol, dims, P, lv_hint_mm=hint))
        else:
            # Three presses, each step prompted:
            #   1st: ask for the crossing at the MV centre
            #   2nd: record the MV centre, ask for the crossing at the LV apex
            #   3rd: record the apex (rough; not stored as the Apex) → detect.
            seed = self._lv_valve_auto_seed
            if seed is None or seed[0] != which:
                self._lv_valve_auto_seed = (which, "mv", None)
                self._lvv_prompt(t(
                    "MV中心近傍にセンターライン交点を合わせてAutoを押してください。"))
                return
            if seed[1] == "mv":
                self._lv_valve_auto_seed = (which, "apex", P)
                self._lvv_prompt(t(
                    "LV心尖近傍にセンターラインの交点を合わせてAutoを押してください。"))
                return
            mv_p, apex_p = np.asarray(seed[2], float), P
            if float(np.linalg.norm(mv_p - apex_p)) < 20.0:
                self._lvv_prompt(t(
                    "心尖部がMV中心に近すぎます。LV心尖近傍にセンターラインの交点を"
                    "合わせてAutoを押してください。"))
                return
            self._lv_valve_auto_seed = None
            res = self._run_busy(t("Auto"), t("{v}面を検出中…").format(v=vname),
                                 lambda: va.detect_mv(vol, dims, mv_p, apex_p))
        if res is None or not res.ok:
            self._lvv_prompt(t(
                "{v}面を自動で検出できませんでした。十字の交点を{v}の近くに置いて"
                "やり直すか、Draw → Set で手動設定してください。").format(v=vname))
            return
        self._lv_apply_valve_data(which, {
            "c": [float(x) for x in res.center],
            "n": [float(x) for x in res.normal],
            "r": float(res.radius)})
        self._lv_valve_saved_sig[which] = None          # auto result: unsaved
        self._lv_valve_auto_contour(which, res)
        self._lv_update_submode_ui()
        self._lvv_prompt(t(
            "{v}面を自動設定しました（半径 {r:.1f} mm）。確認して、ずれていれば "
            "Draw → Set で修正してください。").format(v=vname, r=float(res.radius)))

    def _lv_apex_auto(self) -> None:
        """Apex row → Auto: the epicardial LV apex from the set MV plane (the
        farthest epicardial point from the MV centre, near the MV normal), set
        as the COMMON apex just like Set at the crossing."""
        vol = getattr(self, "_vol", None)
        if vol is None:
            return
        mv = self._lv_valves.get("mitral") if hasattr(self, "_lv_valves") else None
        if mv is None:
            self._lvv_prompt(t("心尖部の自動設定には MV 面が必要です。先に MV を設定して"
                               "ください。"))
            return
        from multi_dicomviewer.core import valve_auto as va
        dims = tuple(float(d) for d in self._dims)
        c, n = np.asarray(mv[0], float), np.asarray(mv[1], float)
        res = self._run_busy(t("Auto"), t("心尖部を検出中…"),
                             lambda: va.detect_apex(vol, dims, c, n))
        if res is None or not res.ok:
            self._lvv_prompt(t(
                "心尖部を自動で検出できませんでした。センターラインの交点を心尖部に"
                "合わせて Set で設定してください。"))
            return
        self._lv_apex = np.asarray(res.apex, float).copy()
        self._lv_apex_shown = True
        self._lv_apex_marker_draw()
        self._lv_update_submode_ui()
        self._lvv_prompt(t(
            "心尖部（心外膜側）を自動設定しました（MV 中心から {d:.0f} mm、心尖部の"
            "壁厚 {w:.1f} mm）。確認して、ずれていれば交点を合わせて Set で修正して"
            "ください。").format(
                d=float(np.linalg.norm(res.apex - c)), w=float(res.wall_mm)))

    #: Detected annulus points (the saddle contour the plane was fitted to),
    #: drawn as a thin outline next to the fitted circle so a wrong hinge shows.
    _LV_VALVE_AUTO_COLOR = "#ff66ff"

    def _lv_valve_auto_contour(self, which, res) -> None:
        P = np.asarray(getattr(res, "points", []), float).reshape(-1, 3)
        if len(P) < 3:
            return
        c = np.asarray(res.center, float)
        n = np.asarray(res.normal, float)
        n = n / (np.linalg.norm(n) or 1.0)
        ref = (np.array([1.0, 0.0, 0.0]) if abs(float(n[0])) < 0.9
               else np.array([0.0, 1.0, 0.0]))
        u = np.cross(n, ref)
        u = u / (np.linalg.norm(u) or 1.0)
        w = np.cross(n, u)
        Q = P - c
        P = P[np.argsort(np.arctan2(Q @ w, Q @ u))]
        pts3d = [tuple(map(float, q)) for q in P]
        hidden = not self._lv_valve_shown.get(which, True)
        # Tagged with the valve, so Clear / Hide / a new Set or Load (which
        # rebuild that valve's measures) remove or hide it with the ring.
        for k in ("A", "B"):
            self._meas_seq += 1
            self._measures[k].append({
                "id": self._meas_seq, "type": "polygon",
                "pts": [self._world3d_to_out(k, q) for q in pts3d],
                "pts3d": list(pts3d), "color": self._LV_VALVE_AUTO_COLOR,
                "_lv_valve": which, "_lv_valve_auto": True,
                "transp": 30, "hidden": hidden})
            self._redraw_meas(k)
