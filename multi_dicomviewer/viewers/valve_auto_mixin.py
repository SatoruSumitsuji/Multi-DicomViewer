# -*- coding: utf-8 -*-
"""MV+Apex / AoV — Auto step (shared by the VTK and pygfx CT viewers).

Auto asks 画像設定 / 点設定 / キャンセル: 画像設定 keeps the normal image
controls; Auto again brings the same choice back (adjust again, or 点設定).
点設定 arms a CLICK pick on the image (a click = press + release without a
drag; drags still run the selected tool). The MV-centre click shows an orange
+, the apex-vicinity click a green + (both kept a few seconds after the
result):

* MV+Apex: (1) click near the MV centre on a long-axis view, (2) click near
  the LV apex → detect_mv_refined (MV → apex → MV … until it settles; the apex
  click only gives the LV side) sets the MV plane AND the common apex at once.
  Between the two clicks Rotate / Paging / the centreline drag are blocked so
  both points sit on the same image.
* AoV: one click near the AoV centre (short-axis view) → detect_aov.

While a procedure runs (waiting for a click, or detecting) the Auto button has
an orange frame; Esc or Auto again cancels. Results go through
_lv_apply_valve_data (same ring / caches as Load) and are marked unsaved.
Manual correction = Draw → Set (MV / AoV) and Apex Set (crossing).

MV+Apex Save / Load / Clear / Hide act on both: an .MVLv.json and an
.ApxLv.json of the same name in one folder.
"""
from __future__ import annotations

import numpy as np

from multi_dicomviewer.i18n import t


#: The Auto button while its procedure runs (waiting for the next press, or
#: detecting): an orange frame, so "an Auto procedure is in progress" shows.
_AUTO_RUN_STY = ("QPushButton{background:#fff3e0;border:2px solid #f57c00;"
                 "border-radius:3px;padding:1px 6px;}")


class _AutoRunning:
    def __init__(self, host, btn):
        self.host, self.btn = host, btn

    def __enter__(self):
        self.host._lv_auto_mark(self.btn, True)
        return self

    def __exit__(self, *exc):
        self.host._lv_auto_mark(self.btn, self.host._lv_valve_auto_seed_v
                                is not None and self.btn is getattr(
                                    self.host, "_lv_ve_auto_btn", None))
        return False


class ValveAutoMixin:
    _lv_valve_auto_seed_v = None             # (which, "mv"|"apex"|"aov", MV mm)

    @property
    def _lv_valve_auto_seed(self):
        return self._lv_valve_auto_seed_v

    @_lv_valve_auto_seed.setter
    def _lv_valve_auto_seed(self, v):
        # Every step change (start, next press, done, valve entered / left)
        # passes here, so the Auto button's "running" frame follows it.
        self._lv_valve_auto_seed_v = v
        self._lv_auto_mark(getattr(self, "_lv_ve_auto_btn", None), v is not None)

    def _lv_auto_running(self, btn):
        return _AutoRunning(self, btn)

    def _lv_auto_mark(self, btn, on) -> None:
        if btn is None:
            return
        dis = getattr(self, "_BTN_DIS", "")
        btn.setStyleSheet((_AUTO_RUN_STY + dis) if on else dis)
        try:
            btn.repaint()
        except Exception:
            pass

    # ------------------------------------------------------------ pick
    _LV_PICK_COLOR = "#ff9800"               # MV-centre click
    _LV_PICK_APEX_COLOR = "#33dd33"          # apex-vicinity click (green +)
    _LV_PICK_SHOW_MS = 3000                  # click markers kept after the result

    def _lv_valve_auto(self) -> None:
        """Auto asks 画像設定 / 点設定 / キャンセル — on the first press AND on
        every press during 画像設定, so the image can be adjusted as often as
        needed. 画像設定 keeps the normal image controls; 点設定 arms the click
        pick; キャンセル during 画像設定 ends the procedure. Auto while the pick
        is armed cancels."""
        which = getattr(self, "_lv_valve_edit", None) or "mitral"
        if getattr(self, "_vol", None) is None:
            return
        seed = self._lv_valve_auto_seed if self._lv_auto_pending() else None
        if seed is not None and seed[1] != "prep":
            self._lv_pick_cancel()
            return
        choice = self._lv_auto_choice(which)
        if choice is None:
            if seed is not None:                 # キャンセル during 画像設定
                self._lv_pick_cancel()
            return
        if choice == "prep":
            self._lv_valve_auto_seed = (which, "prep", None)
            self._lvv_prompt(t(
                "画像を調整してから、もう一度 Auto を押してください。選択画面で "
                "{p} を選ぶと点の設定に進みます。（Esc で中止）").format(
                    p=t("MV点設定") if which == "mitral" else t("AoV点設定")))
            return
        self._lv_pick_arm(which)

    def _lv_auto_choice(self, which):
        """'prep' (adjust the image first), 'pick' (set the points now) or
        None (cancel)."""
        from PyQt6.QtWidgets import QMessageBox
        pick_lbl = t("MV点設定") if which == "mitral" else t("AoV点設定")
        box = QMessageBox(self.window())
        box.setWindowTitle(t("Auto"))
        box.setIcon(QMessageBox.Icon.Question)
        if which == "mitral":
            box.setText(t(
                "MV+Apex の Auto を始めます。\n\n画像設定：先に観察する左室長軸像を"
                "調整します（Rotate / Paging などが普段どおり使えます）。調整後に"
                "もう一度 Auto を押すと、この選択画面に戻ります。\n\nMV点設定：すぐに "
                "MV 中心近傍 → 心尖近傍 の順にクリックします。"))
        else:
            box.setText(t(
                "AoV の Auto を始めます。\n\n画像設定：先に AoV の短軸像を調整します"
                "（Rotate / Paging などが普段どおり使えます）。調整後にもう一度 Auto "
                "を押すと、この選択画面に戻ります。\n\nAoV点設定：すぐに AoV 中心近傍を"
                "クリックします。"))
        b_prep = box.addButton(t("画像設定"), QMessageBox.ButtonRole.ActionRole)
        b_pick = box.addButton(pick_lbl, QMessageBox.ButtonRole.AcceptRole)
        b_cancel = box.addButton(t("キャンセル"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(b_prep)
        box.setEscapeButton(b_cancel)
        box.exec()
        c = box.clickedButton()
        return "prep" if c is b_prep else "pick" if c is b_pick else None

    def _lv_pick_arm(self, which) -> None:
        # Measure would take the clicks as drawing → off while picking.
        if getattr(self, "_meas_on", False) and hasattr(self, "_meas_btn"):
            self._meas_btn.setChecked(False)
            self._toggle_measure()
        self._lv_pick_xy = None
        self._lv_pick_marker_clear()
        if which == "aortic":
            self._lv_valve_auto_seed = (which, "aov", None)
            self._lvv_prompt(t("AoVの短軸を表示し、その中心近傍をクリックしてください。"))
        else:
            self._lv_valve_auto_seed = (which, "mv", None)
            self._lvv_prompt(t(
                "左室長軸の観察画像で、まずMV中心近傍（MVの楕円の中央付近）をクリック"
                "し、次に心尖近傍をクリックしてください。"))

    def _lv_auto_pending(self) -> bool:
        """An Auto procedure runs for the valve being edited (画像設定 or pick)."""
        seed = self._lv_valve_auto_seed
        return (seed is not None
                and seed[0] == getattr(self, "_lv_valve_edit", None))

    def _lv_pick_active(self) -> bool:
        """Clicks on the image are picks (not during 画像設定)."""
        return self._lv_auto_pending() and self._lv_valve_auto_seed[1] != "prep"

    def _lv_pick_locked(self) -> bool:
        """Between the MV and the apex click: the image must stay the same, so
        Rotate / Paging / the centreline drag are blocked (Zoom / Move / WL
        still work)."""
        return self._lv_pick_active() and self._lv_valve_auto_seed[1] == "apex"

    def _lv_pick_blocks_tool(self, tool) -> bool:
        return self._lv_pick_locked() and tool in ("ROTATE", "PAGING")

    def _lv_pick_cancel(self, prompt=True) -> None:
        self._lv_valve_auto_seed = None
        self._lv_pick_xy = None
        self._lv_pick_marker_clear()
        if prompt:
            self._lvv_prompt(t("Auto を中止しました。"))

    def _lv_pick_escape(self) -> bool:
        """Esc while an Auto procedure runs → cancel. True when consumed."""
        if self._lv_auto_pending():
            self._lv_pick_cancel()
            return True
        return False

    def _lv_pick_press(self, pane, x, y) -> None:
        self._lv_pick_xy = ((pane, float(x), float(y))
                            if self._lv_pick_active() else None)

    def _lv_pick_release(self, pane, x, y) -> bool:
        """A left release that did not drag (≤ 4 px) since its press = a pick."""
        p = getattr(self, "_lv_pick_xy", None)
        self._lv_pick_xy = None
        if (p is None or p[0] != pane or not self._lv_pick_active()
                or abs(float(x) - p[1]) + abs(float(y) - p[2]) > 4.0):
            return False
        wx, wy = self._disp_to_world(pane, x, y)
        P = np.asarray(self._out_to_world3d(pane, wx, wy), float)
        self._lv_pick_point(P)
        return True

    def _lv_pick_point(self, P) -> None:
        seed = self._lv_valve_auto_seed
        if seed is None:
            return
        which, stage = seed[0], seed[1]
        P = np.asarray(P, float)
        if stage == "aov":
            self._lv_run_aov_auto(P)
        elif stage == "mv":
            self._lv_valve_auto_seed = (which, "apex", P.copy())
            self._lv_pick_marker(P)
            self._lvv_prompt(t("心尖近傍をクリックしてください。"))
        else:
            mv_p = np.asarray(seed[2], float)
            if float(np.linalg.norm(mv_p - P)) < 20.0:
                self._lvv_prompt(t(
                    "心尖部がMV中心に近すぎます。心尖近傍をクリックしてください。"))
                return
            # Show the apex-vicinity click (green +) BEFORE the detection runs,
            # so it is clear the click was taken.
            self._lv_pick_marker(P, self._LV_PICK_APEX_COLOR, keep=True)
            self._lv_pick_repaint()
            self._lv_run_mv_apex_auto(mv_p, P)

    def _lv_pick_marker(self, P, color=None, keep=False) -> None:
        """'+' marker at a picked point on both panes (keep=True adds it next
        to the markers already there)."""
        if not keep:
            self._lv_pick_marker_clear()
        q = tuple(map(float, P))
        for k in ("A", "B"):
            self._meas_seq += 1
            self._measures[k].append({
                "id": self._meas_seq, "type": "point",
                "pts": [self._world3d_to_out(k, q)], "pts3d": [q],
                "color": color or self._LV_PICK_COLOR, "_lv_pick": True})
            self._redraw_meas(k)

    def _lv_pick_repaint(self) -> None:
        """Paint the panes now (the detection that follows blocks the loop)."""
        for k in ("A", "B"):
            try:
                pn = self.pane[k]
                if hasattr(pn, "render"):
                    pn.render()
                if hasattr(self, "_overlay"):
                    self._overlay[k].repaint()
            except Exception:                            # noqa: BLE001
                pass
        try:
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception:                                # noqa: BLE001
            pass

    def _lv_pick_marker_clear_later(self) -> None:
        """Keep the click markers a moment after the result, then remove them
        (unless a new Auto has started meanwhile)."""
        from PyQt6.QtCore import QTimer
        tok = object()
        self._lv_pick_clear_tok = tok

        def fire():
            if (getattr(self, "_lv_pick_clear_tok", None) is tok
                    and not self._lv_auto_pending()):
                self._lv_pick_marker_clear()
        QTimer.singleShot(self._LV_PICK_SHOW_MS, fire)

    def _lv_pick_marker_clear(self) -> None:
        meas = getattr(self, "_measures", None)
        if not meas:
            return
        for k in ("A", "B"):
            ms = meas.get(k, [])
            if any(m.get("_lv_pick") for m in ms):
                meas[k] = [m for m in ms if not m.get("_lv_pick")]
                self._redraw_meas(k)

    # ------------------------------------------------------------ detect
    def _lv_run_aov_auto(self, P) -> None:
        from multi_dicomviewer.core import valve_auto as va
        vol = self._vol
        dims = tuple(float(d) for d in self._dims)
        mv = self._lv_valves.get("mitral") if hasattr(self, "_lv_valves") else None
        hint = np.asarray(mv[0], float) if mv is not None else None
        with self._lv_auto_running(self._lv_ve_auto_btn):
            res = self._run_busy(
                t("Auto"), t("{v}面を検出中…").format(v="AoV"),
                lambda: va.detect_aov(vol, dims, P, lv_hint_mm=hint))
        self._lv_valve_auto_seed = None
        if res is None or not res.ok:
            self._lvv_prompt(t(
                "AoV面を自動で検出できませんでした。Auto をもう一度押してAoV中心を"
                "クリックし直すか、Draw → Set で手動設定してください。"))
            return
        self._lv_set_valve_result("aortic", res)
        self._lvv_prompt(t(
            "AoV面を自動設定しました（半径 {r:.1f} mm）。確認して、ずれていれば "
            "Draw → Set で修正してください。").format(r=float(res.radius)))

    def _lv_run_mv_apex_auto(self, mv_p, apex_p) -> None:
        from multi_dicomviewer.core import valve_auto as va
        vol = self._vol
        dims = tuple(float(d) for d in self._dims)

        def work():
            res, apex = va.detect_mv_refined(vol, dims, mv_p, apex_p)
            if res.ok and apex is None:          # refinement stopped early
                a = va.detect_apex(vol, dims, res.center, res.normal)
                apex = a if a.ok else None
            return res, apex

        with self._lv_auto_running(self._lv_ve_auto_btn):
            out = self._run_busy(t("Auto"), t("MV面と心尖部を検出中…"), work)
        self._lv_valve_auto_seed = None
        self._lv_pick_marker_clear_later()
        res, apex = out if out is not None else (None, None)
        if res is None or not res.ok:
            self._lvv_prompt(t(
                "MV面を自動で検出できませんでした。Auto をもう一度押してクリックし"
                "直すか、Draw → Set で手動設定してください。"))
            return
        self._lv_set_valve_result("mitral", res)
        if apex is not None:
            self._lv_apex = np.asarray(apex.apex, float).copy()
            self._lv_apex_shown = True
        # Check view: MV face-on (left) and edge-on along the long axis (right).
        if hasattr(self, "_lv_view_mv_perpendicular"):
            self._lv_view_mv_perpendicular(reset_cam=False)
            for k in ("A", "B"):
                self._redraw_meas(k)
        self._lv_apex_marker_draw()
        self._lv_update_submode_ui()
        if apex is None:
            self._lvv_prompt(t(
                "MV面を自動設定しました（半径 {r:.1f} mm）が、心尖部は検出できません"
                "でした。交点を心尖部に合わせて Apex Set で設定してください。").format(
                    r=float(res.radius)))
            return
        self._lvv_prompt(t(
            "MV面（半径 {r:.1f} mm）と心尖部（MV 中心から {d:.0f} mm、壁厚 "
            "{w:.1f} mm）を自動設定しました。確認して、ずれていれば MV は Draw → "
            "Set、心尖部は交点を合わせて Apex Set で修正してください。").format(
                r=float(res.radius), d=float(np.linalg.norm(apex.apex - res.center)),
                w=float(apex.wall_mm)))

    def _lv_set_valve_result(self, which, res) -> None:
        self._lv_apply_valve_data(which, {
            "c": [float(x) for x in res.center],
            "n": [float(x) for x in res.normal],
            "r": float(res.radius)})
        self._lv_valve_saved_sig[which] = None          # auto result: unsaved
        self._lv_valve_auto_contour(which, res)
        self._lv_update_submode_ui()

    # --------------------------------------------- MV+Apex Save/Load/Clear/Hide
    def _lv_apex_payload(self) -> dict:
        return {"type": "apex",
                "series": (self._lv_series_meta()
                           if hasattr(self, "_lv_series_meta") else {}),
                "src_dir": (self._lv_series_dir()
                            if hasattr(self, "_lv_series_dir") else ""),
                "apex": list(map(float, np.asarray(self._lv_apex, float)))}

    def _lv_mv_payload(self) -> dict:
        c, n, r = self._lv_valves["mitral"]
        return {"type": "valve", "valve": "mitral",
                "series": (self._lv_series_meta()
                           if hasattr(self, "_lv_series_meta") else {}),
                "src_dir": (self._lv_series_dir()
                            if hasattr(self, "_lv_series_dir") else ""),
                "c": list(map(float, c)), "n": list(map(float, n)),
                "r": float(r)}

    @staticmethod
    def _lv_mvapex_stem(path) -> str:
        """'…/case.MVLv.json' → '…/case' (any case of the suffix)."""
        low = path.lower()
        for suf in (".mvlv.json", ".apxlv.json", ".json"):
            if low.endswith(suf):
                return path[: len(path) - len(suf)]
        return path

    def _lv_mvapex_save(self) -> None:
        """Save MV + apex: one dialog, two files of the same name side by side
        (.MVLv.json and .ApxLv.json). Only the MV is written when no apex."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import json
        import os
        if self._lv_valves.get("mitral") is None:
            QMessageBox.information(self.window(), t("LV"),
                                    t("Set the {w} plane first.").format(w="MV"))
            return
        d = self._lv_save_dir() if hasattr(self, "_lv_save_dir") else ""
        stem = (self._lv_default_stem() if hasattr(self, "_lv_default_stem")
                else "valve")
        default = os.path.join(d, stem + ".MVLv.json") if d else stem + ".MVLv.json"
        path, _ = QFileDialog.getSaveFileName(
            self.window(), t("Save MV + Apex"), default,
            "MV plane (*.MVLv.json);;JSON (*.json)")
        if not path:
            return
        base = self._lv_mvapex_stem(path)
        files = [(base + ".MVLv.json", self._lv_mv_payload())]
        if getattr(self, "_lv_apex", None) is not None:
            files.append((base + ".ApxLv.json", self._lv_apex_payload()))
        try:
            for fp, data in files:
                if hasattr(self, "_unlink_case_variant"):
                    self._unlink_case_variant(fp)
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
        except Exception as exc:                        # noqa: BLE001
            QMessageBox.warning(self.window(), t("LV"),
                                t("Save failed: {err}", err=str(exc)))
            return
        self._lv_remember_dir(files[0][0])
        self._lv_valve_saved_sig["mitral"] = self._lv_valve_sig("mitral")
        names = "\n".join(os.path.basename(fp) for fp, _d in files)
        msg = t("Saved: {p}", p=names)
        if len(files) == 1:
            msg += "\n\n" + t("心尖部が未設定のため、MV だけ保存しました。")
        QMessageBox.information(self.window(), t("LV"), msg)

    def _lv_mvapex_load(self) -> None:
        """Load an .MVLv.json (or .ApxLv.json) and its same-name partner."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import json
        import os
        if getattr(self, "_vol", None) is None:
            return
        d = self._lv_save_dir() if hasattr(self, "_lv_save_dir") else ""
        path, _ = QFileDialog.getOpenFileName(
            self.window(), t("Load MV + Apex"), d,
            "MV / Apex (*.MVLv.json *.ApxLv.json);;JSON (*.json)")
        if not path:
            return
        base = self._lv_mvapex_stem(path)
        folder = os.path.dirname(base) or "."
        name = os.path.basename(base).lower()

        def find(suffix):
            try:
                for f in os.listdir(folder):
                    if f.lower() == name + suffix:
                        return os.path.join(folder, f)
            except OSError:
                pass
            return None

        mv_fp, ax_fp = find(".mvlv.json"), find(".apxlv.json")
        if path.lower().endswith(".json") and not (mv_fp or ax_fp):
            mv_fp = path                                 # a plain .json: as MV
        try:
            loaded = {}
            for kind, fp in (("mv", mv_fp), ("apex", ax_fp)):
                if fp:
                    with open(fp, encoding="utf-8") as f:
                        loaded[kind] = json.load(f)
            cur = (self._lv_series_meta().get("series_uid", "")
                   if hasattr(self, "_lv_series_meta") else "")
            if any((dd.get("series") or {}).get("series_uid", "") not in ("", cur)
                   for dd in loaded.values()) and cur:
                if QMessageBox.question(
                        self.window(), t("LV"),
                        t("This valve file was saved for a DIFFERENT series — it "
                          "may not line up. Load anyway?")) \
                        != QMessageBox.StandardButton.Yes:
                    return
            if "mv" in loaded:
                self._lv_apply_valve_data("mitral", loaded["mv"])
            if "apex" in loaded:
                self._lv_apply_apex_data(loaded["apex"])
            self._lv_remember_dir(path)
        except Exception as exc:                        # noqa: BLE001
            QMessageBox.warning(self.window(), t("LV"),
                                t("Load failed: {err}", err=str(exc)))
            return
        got = [os.path.basename(fp) for fp in (mv_fp, ax_fp) if fp]
        msg = t("Loaded: {p}", p="\n".join(got))
        if "apex" not in loaded:
            msg += "\n\n" + t("同じ名前の .ApxLv.json がないため、心尖部は読み込んでいません。")
        if "mv" not in loaded:
            msg += "\n\n" + t("同じ名前の .MVLv.json がないため、MV は読み込んでいません。")
        QMessageBox.information(self.window(), t("LV"), msg)

    def _lv_mvapex_clear(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        has_mv = self._lv_valves.get("mitral") is not None
        has_ax = getattr(self, "_lv_apex", None) is not None
        if not (has_mv or has_ax):
            self._lvv_prompt(t("No {v} plane on the image to clear.").format(
                v="MV"))
            return
        if QMessageBox.question(
                self.window(), t("LV"),
                t("MV 面と心尖部を画像から消しますか？保存したファイルは残ります。")) \
                != QMessageBox.StandardButton.Yes:
            return
        if has_mv:
            self._lv_valves["mitral"] = None
            self._lv_valve_shown["mitral"] = True
            self._lv_valve_saved_sig["mitral"] = None
            for k in ("A", "B"):
                self._measures[k] = [m for m in self._measures.get(k, [])
                                     if m.get("_lv_valve") != "mitral"]
                self._redraw_meas(k)
            self._lv_on_valve_changed("mitral")
        if has_ax:
            self._lv_apex = None
            self._lv_apex_marker_draw()
        self._lv_update_valve_buttons()
        self._lv_update_submode_ui()

    def _lv_mvapex_toggle(self) -> None:
        """Hide / Show MV + apex together (the apex follows the MV ring)."""
        if self._lv_valves.get("mitral") is not None:
            self._lv_toggle_valve_visibility("mitral")
            shown = self._lv_valve_shown.get("mitral", True)
        else:
            shown = not getattr(self, "_lv_apex_shown", True)
        if getattr(self, "_lv_apex", None) is not None:
            self._lv_apex_shown = shown
            self._lv_apex_marker_draw()
        self._lv_update_valve_buttons()
        self._lv_update_submode_ui()

    def _lv_mvapex_rclick(self) -> None:
        if getattr(self, "_vol", None) is None or (
                self._lv_valves.get("mitral") is None
                and getattr(self, "_lv_apex", None) is None):
            self._lvv_prompt(t("Set the {v} plane first.").format(v="MV"))
            return
        self._lv_mvapex_toggle()

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
        with self._lv_auto_running(getattr(self, "_lv_ax_auto_btn", None)):
            res = self._run_busy(t("Auto"), t("心尖部を検出中…"),
                                 lambda: va.detect_apex(vol, dims, c, n))
        if res is None or not res.ok:
            self._lvv_prompt(t(
                "心尖部を自動で検出できませんでした。センターラインの交点を心尖部に"
                "合わせて Set で設定してください。"))
            return
        self._lv_apex = np.asarray(res.apex, float).copy()
        self._lv_apex_shown = True
        # Move the crossing onto the apex so both panes pass through it and the
        # marker shows right away (the result is checked without hunting).
        self._lv_goto_point(self._lv_apex)
        self._lv_apex_marker_draw()
        self._lv_update_submode_ui()
        self._lvv_prompt(t(
            "心尖部（心外膜側）を自動設定しました（MV 中心から {d:.0f} mm、心尖部の"
            "壁厚 {w:.1f} mm）。確認して、ずれていれば交点を合わせて Set で修正して"
            "ください。").format(
                d=float(np.linalg.norm(res.apex - c)), w=float(res.wall_mm)))

    def _lv_goto_point(self, P) -> None:
        """Move the centreline crossing to the 3-D point P (pane orientations
        kept), like a click-recentre: both panes reslice through P and centre on
        it. One Ctrl+Z step."""
        if getattr(self, "_center", None) is None:
            return
        before = self._view_snapshot() if hasattr(self, "_view_snapshot") else None
        self._center = np.asarray(P, float).copy()
        self._clamp_center()
        self._pc = {"A": self._center.copy(), "B": self._center.copy()}
        self._view_initial = False
        if hasattr(self, "_center_camera"):                  # VTK viewer
            for k in ("A", "B"):
                self._center_camera(k)
        elif isinstance(getattr(self, "_pan", None), dict):  # pygfx viewer
            self._pan = {"A": np.zeros(2), "B": np.zeros(2)}
        self._refresh()
        for k in ("A", "B"):
            self._redraw_meas(k)
        if before is not None and hasattr(self, "_undo_view"):
            self._undo_view(before, self._view_snapshot())

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
