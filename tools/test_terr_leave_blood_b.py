"""Leaving the Territory review brings the blood back on the RIGHT pane
(2026-11-17): after a .territory.json load the right pane (the VR in the
review) stayed without blood in LV / other modes. On leave both MPR panes get
their blood layers visible, the LV-Blood volume wired to both mask reslices,
the LUTs re-applied.

    python tools/test_terr_leave_blood_b.py
"""
import os
import sys
import tempfile
from pathlib import Path

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
os.environ["QT_QPA_PLATFORM"] = ("windows" if sys.platform == "win32"
                                 else os.environ.get("QT_QPA_PLATFORM", "offscreen"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from multi_dicomviewer.core import settings  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
from multi_dicomviewer.core.dicom_io import LoadedSeries  # noqa: E402
from multi_dicomviewer.core.study_model import Modality  # noqa: E402
from multi_dicomviewer.viewers import ct_viewer as ctv  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


vol = np.full((40, 48, 48), -100.0, np.float32)
s = LoadedSeries(modality=Modality.CT, volume=vol, spacing_mm=(0.7, 0.7),
                 cine_fps=None, window=800.0, level=200.0, slice_mm=1.5,
                 patient_basis=None, series_uid="tlb-ct")
v = ctv.CTViewer()
v.resize(1400, 800)
v.show()
v.load_series(s, "tlb")
v._set_mode("3D")
app.processEvents()

full = np.zeros(vol.shape, np.float32)
full[10:30, 15:35, 15:35] = 1.0
sx, sy, sz = v._dims
v._lvv = {"observe": True}
v._lvv_mask_vol = ctv.numpy_to_vtk_image(full, sx, sy, sz)
v._lvv_mask_on = True
v._lvv_hl_on = False
pB = v.pane["B"]
# The broken state seen after the review: right-pane blood layer hidden and
# its mask reslice not wired to the LV-Blood volume.
pB.actor_mask.SetVisibility(False)
pB.actor_hl.SetVisibility(False)
pB.reslice_mask.SetInputData(ctv._placeholder_image())
v._territory_display = True
v._ct_terr_snap = v._ct_view_snap()
v._ct_territory_leave()
app.processEvents()
check("right pane: blood layers visible again",
      pB.actor_mask.GetVisibility() and pB.actor_hl.GetVisibility())
check("right pane: mask reslice wired to the LV-Blood volume",
      pB.reslice_mask.GetInput() is v._lvv_mask_vol)
pB.reslice_mask.Update()
rng = pB.reslice_mask.GetOutput().GetScalarRange()
check("right pane: the LV-Blood region is on its plane", rng[1] >= 1.0, rng)
lut = pB.colors_mask.GetLookupTable()
check("right pane: LV-Blood LUT on (opaque for the region)",
      lut is not None and lut.GetTableValue(lut.GetNumberOfTableValues() - 1)[3] > 0)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.stdout.flush()
os._exit(1 if fails else 0)
