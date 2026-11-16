"""CoronaryTree tools use ONE Epi by fixed priority (2026-11-16):
FullLv.json > EpiLv.json > an Epi made in LV mode; none → the VR's 70 mm sphere.
The same pick drives the VR crop, the short-axis Epi line (CPR → Epi) and the
gap-point snap. No Set Epi-Data / no question at Open VR.

    python tools/test_coro_epi_priority.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from multi_dicomviewer.viewers.cpr_mixin import CPRMixin  # noqa: E402

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


class Host(CPRMixin):
    def __init__(self):
        self._vol = np.zeros((4, 4, 4), np.int16)
        self._lvv_epi_model_dict = None
        self._lvv_epi_surf = None
        self._lv = None
        self._lv_file_name = ""
        self.synced = 0

    def _cpr_vr_group_sync(self):
        self.synced += 1


h = Host()
check("no Epi → None (sphere)", h._coro_epi() == (None, None, None))
check("no Epi → no source", h._cpr_vr_epi_source() is None)

# an Epi made in LV mode only
lv_surf = object()


class M:
    epi = lv_surf


h._lv = {"model": M()}
check("LV-mode Epi → 'lv'", h._coro_epi() == ("lv", None, lv_surf))

# a loaded EpiLv (through _lvv_apply_epi_data → tagged as a file)
epilv = {"kind": "mdv-lvef", "region": {"r": 1}}
h._lvv_epi_model_dict = epilv
h._lvv_epi_file_dict = epilv
h._lvv_epi_surf = s_epi = object()
check("EpiLv wins over the LV-mode Epi", h._coro_epi() == ("epilv", epilv, s_epi))
check("VR source = the EpiLv's packed region", h._cpr_vr_epi_source() is epilv)
check("short-axis Epi line / gap snap = the EpiLv surface",
      h._cpr_epi_surface() is s_epi)

# a stash of the LV trace (not a file) in the slot → 'lv'
h._lvv_epi_model_dict = {"kind": "mdv-lvef"}
check("an untagged Epi in the slot reads as the LV-mode Epi",
      h._coro_epi()[0] == "lv")

# Territory entry with an in-session LV (no file): not taken for a FullLv
h._lvv_epi_model_dict = {"kind": "mdv-lvef"}
h._lvv_epi_surf = object()
h._cpr_mark_full_lv()
check("Territory light entry: an LV-mode Epi stays 'lv'", h._coro_epi()[0] == "lv")

# an Epi loaded in LV mode (Epi pass Load), then the pass left unchanged:
# the stash tags it → EpiLv; with unsaved edits it stays the LV-mode Epi
from multi_dicomviewer.viewers.ct_viewer import CTViewer  # noqa: E402


class Model:
    def __init__(self):
        self.epi_axis = type("A", (), {"apex": [0.0, 0.0, 0.0]})()
        self.epi_contours = [1, 2, 3]
        self.epi = object()

    def build(self):
        pass

    def to_dict(self):
        return {"kind": "mdv-lvef"}


for dirty, want in ((False, "epilv"), (True, "lv")):
    h._lv_dirty = dirty
    h._lvv_epi_file_dict = None
    try:
        CTViewer._lv_stash_epi_for_blood(h, Model())
    except Exception:                            # noqa: BLE001
        pass
    check(f"LV-mode Load + leave, dirty={dirty} → {want!r}",
          h._coro_epi()[0] == want, str(h._coro_epi()[0]))
h._lv = {"model": M()}

# FullLv applied: its Epi is kept aside
full_epi = {"kind": "mdv-lvef", "region": {"f": 1}}
h._lvv_epi_model_dict = full_epi
h._lvv_epi_file_dict = full_epi
h._lvv_epi_surf = s_full = object()
h._lv_file_name = "case.FullLv.json"
h._cpr_mark_full_lv()
check("FullLv → 'fulllv'", h._coro_epi() == ("fulllv", full_epi, s_full))
# …then an EpiLv is loaded: the FullLv still wins
h._lvv_epi_model_dict = epilv
h._lvv_epi_file_dict = epilv
h._lvv_epi_surf = s_epi
check("FullLv still wins after an EpiLv load", h._coro_epi() == ("fulllv", full_epi, s_full))
check("VR source = the FullLv's region", h._cpr_vr_epi_source() is full_epi)
check("short-axis Epi line / gap snap = the FullLv surface",
      h._cpr_epi_surface() is s_full)
check("_cpr_full_lv_on follows the pick", h._cpr_full_lv_on())

# the FullLv region is unpacked from ITS dict (not the live EpiLv slot)
import base64  # noqa: E402
import zlib  # noqa: E402
comp = np.ones((2, 2, 2), bool)
packed = base64.b64encode(zlib.compress(np.packbits(comp.ravel()).tobytes())).decode()
full_epi["region"] = {"vol_shape": [4, 4, 4], "shape": [2, 2, 2],
                      "bbox": [1, 3, 1, 3, 1, 3], "packed": packed}
h._vr_epi_full_mask = lambda: (_ for _ in ()).throw(AssertionError("live slot used"))
try:
    reg = h._cpr_vr_epi_region()
    ok = reg is not None and int(reg.sum()) == 8 and bool(reg[1, 1, 1])
except AssertionError:
    ok = False
check("VR crop region = the FullLv's packed region", ok)

# a new 3DCT (another volume): the kept FullLv no longer applies
h._vol = np.zeros((4, 4, 4), np.int16)
check("another 3DCT → the FullLv pick lapses", h._coro_epi()[0] == "epilv")

src = (ROOT / "multi_dicomviewer/viewers/cpr_mixin.py").read_text(encoding="utf-8")
check("no Set Epi-Data / Epi question left",
      "Set Epi-Data" not in src and "_cpr_vr_epi_choice" not in src)
check("Open VR: Epi → band, none → sphere",
      "self._cpr_vr_no_epi = self._coro_epi()[0] is None" in src)
check("no Epi status label", "_cpr_epi_lbl" not in src)
for f in ("ct_viewer.py", "ct_viewer_pygfx.py"):
    v = (ROOT / "multi_dicomviewer/viewers" / f).read_text(encoding="utf-8")
    check(f"{f}: a loaded Epi is tagged as a file",
          "self._lvv_epi_file_dict = data" in v)
    check(f"{f}: an Epi loaded in LV mode, unchanged → EpiLv on stash",
          'if not getattr(self, "_lv_dirty", True):' in v
          and "self._lvv_epi_file_dict = d\n" in v)

print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
