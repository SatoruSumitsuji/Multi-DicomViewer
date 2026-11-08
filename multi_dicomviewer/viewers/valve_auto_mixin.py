# -*- coding: utf-8 -*-
"""MV / AoV plane — Auto step (shared by the VTK and pygfx CT viewers).

The valve-edit row's Auto button reads the rough position from the centreline
CROSSING (self._center, as Apex → Set does) and fits the plane with
core.valve_auto:

* AoV: one press — crossing in / near the aortic root.
* MV:  crossing near the MV centre. The rough apex seeds the first rotation
  axis only: the set Apex is used when there is one; otherwise the first press
  stores the MV position and asks for the crossing to be moved to the apex
  roughly (not stored as the Apex — the Epi step still sets that against the
  MV plane) and Auto pressed again.

The result goes through _lv_apply_valve_data (same ring / caches as Load) and
is marked unsaved. Manual correction = the existing Draw → Set.
"""
from __future__ import annotations

import numpy as np

from multi_dicomviewer.i18n import t


class ValveAutoMixin:
    _lv_valve_auto_seed = None               # (which, MV-centre mm) between presses

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
            mv = (self._lv_valves.get("mitral") if hasattr(self, "_lv_valves")
                  else None)
            hint = np.asarray(mv[0], float) if mv is not None else None
            res = self._run_busy(t("Auto"), t("{v}面を検出中…").format(v=vname),
                                 lambda: va.detect_aov(vol, dims, P, lv_hint_mm=hint))
        else:
            apex = getattr(self, "_lv_apex", None)
            seed = self._lv_valve_auto_seed
            if apex is not None:
                mv_p, apex_p = P, np.asarray(apex, float)
            elif seed is not None and seed[0] == which:
                mv_p, apex_p = np.asarray(seed[1], float), P
                if float(np.linalg.norm(mv_p - apex_p)) < 20.0:
                    self._lvv_prompt(t(
                        "心尖部がMV中心に近すぎます。十字の交点を心尖部付近へ移動して、"
                        "もう一度 Auto を押してください。"))
                    return
            else:
                self._lv_valve_auto_seed = (which, P)
                self._lvv_prompt(t(
                    "MV中心の位置を記録しました。次に十字の交点を心尖部付近"
                    "（だいたいで可）へ移動して、もう一度 Auto を押してください。"))
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
