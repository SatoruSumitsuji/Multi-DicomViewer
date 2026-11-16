"""LV mode MV-AoV point UI (2026-11-16) — source checks of both viewers.

MV+Apex → MV-AoV+Apex (the point + apex at the crossing), the AoV button hidden,
the gate accepts the MV-AoV point, the Epi plane count from Settings, the
tent snap of the Epi basal ends, Fit = re-converge onto the point."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
fails = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


for f in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    s = (ROOT / "multi_dicomviewer" / "viewers" / f).read_text(encoding="utf-8")
    check(f"{f}: button MV-AoV+Apex", 'FitButton(t("MV-AoV+Apex"))' in s
          and 'FitButton(t("MV+Apex"))' not in s)
    check(f"{f}: AoV button hidden", "self._lv_aov_btn.setVisible(False)" in s)
    check(f"{f}: gate accepts MV-AoV", "or self._lv_mvaov_on()))" in s)
    check(f"{f}: plane count from Settings",
          "LVModel(n_planes=lv_epi_n_planes())" in s)
    check(f"{f}: tent snap", "tent_terminal_ok(ax, P, c)" in s
          and "m.epi_tent = bool(tent)" in s)
    check(f"{f}: Fit → tent refit", "self._lv_tent_refit()" in s)
    check(f"{f}: Set → point at the crossing", "self._lv_mvaov_set_ask()" in s)
    check(f"{f}: apex moves re-aim the point", s.count("self._lv_mvaov_sync()") >= 2)
mx = (ROOT / "multi_dicomviewer" / "viewers" / "valve_auto_mixin.py").read_text(
    encoding="utf-8")
check("mixin: MV-AoV+Apex has no Auto",
      'is None or which == "mitral":' in mx)
check("mixin: undo snapshot has mvaov", '"mvaov"' in mx)
dlg = (ROOT / "multi_dicomviewer" / "ui" / "settings_dialog.py").read_text(
    encoding="utf-8")
check("Settings: Epi境界設定断面数", "Epi境界設定断面数" in dlg and "epi_n_planes" in dlg)
print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
