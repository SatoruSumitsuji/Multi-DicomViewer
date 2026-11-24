"""IVUS long axis follows the DISPLAYED cross-section (user 2026-11-24): with
the image turned ~180° by Multi-Rot, the side branch seen on the blue side
showed on the red side of the strip. The strip now samples, frame by frame, the
section the yellow cut line marks on screen — through the Multi-Rot / free
rotation AND the flip / 90° turns — and the centre marker rides the anatomy.
Synthetic data only."""
import math
import os
import sys
import tempfile

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.environ["USERPROFILE"] = HOME
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from multi_dicomviewer.core import dicom_io, settings, view_xform  # noqa: E402
from multi_dicomviewer.viewers.ivus_viewer import IVUSViewer  # noqa: E402

assert str(settings.SETTINGS_DIR).startswith(HOME), settings.SETTINGS_DIR
FAILS = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info if not ok else "")
    if not ok:
        FAILS.append(name)


# --- pure geometry round trips
rng = np.random.default_rng(1)
ok = True
for _ in range(200):
    w, h = float(rng.integers(20, 90)), float(rng.integers(20, 90))
    flip, rot, deg = bool(rng.integers(2)), int(rng.integers(4)), \
        float(rng.uniform(0, 360))
    x, y = float(rng.uniform(0, w)), float(rng.uniform(0, h))
    vx, vy = view_xform.raw_to_view(x, y, w, h, flip, rot, deg)
    bx, by = view_xform.view_to_raw(vx, vy, w, h, flip, rot, deg)
    ok &= math.hypot(bx - x, by - y) < 1e-6
    # a raw step along the returned direction is seen along the screen dir
    a = float(rng.uniform(0, 2 * math.pi))
    d = view_xform.view_dirs_to_raw(math.cos(a), math.sin(a), flip, rot,
                                    [deg])[0]
    px, py = view_xform.raw_to_view(x + d[0], y + d[1], w, h, flip, rot, deg)
    ok &= (abs(px - vx - math.cos(a)) < 1e-6
           and abs(py - vy - math.sin(a)) < 1e-6)
check("geometry: raw↔view round trip and direction mapping", ok)
check("orient_point matches the canvas pixel orientation (Rt90 of a 4×2)",
      view_xform.orient_point(3.5, 0.5, 4, 2, False, 1) == (1.5, 3.5))

# --- a 64×64 IVUS: bright "side branch" blob RIGHT of the centre (raw)
N, S = 30, 64
vol = np.full((N, S, S), 20, dtype=np.uint8)
vol[:, 30:34, 50:56] = 250                    # raw +x side
d = tempfile.mkdtemp()
p = os.path.join(d, "ivus.dcm")
meta = FileMetaDataset()
meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
meta.MediaStorageSOPInstanceUID = generate_uid()
meta.TransferSyntaxUID = ExplicitVRLittleEndian
ds = FileDataset(p, {}, file_meta=meta, preamble=b"\0" * 128)
ds.SOPClassUID = meta.MediaStorageSOPClassUID
ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
ds.StudyInstanceUID = generate_uid()
ds.SeriesInstanceUID = generate_uid()
ds.PatientID = "TEST"
ds.PatientName = "Test^Synthetic"
ds.Modality = "IVUS"
ds.NumberOfFrames = N
ds.Rows = ds.Columns = S
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"
ds.BitsAllocated = ds.BitsStored = 8
ds.HighBit = 7
ds.PixelRepresentation = 0
ds.PixelData = vol.tobytes()
ds.save_as(p, enforce_file_format=True)
pats = dicom_io.index_files([p])
se = [s for pt in pats.values() for st in pt.studies.values()
      for s in st.series.values()][0]

v = IVUSViewer()
v.resize(1000, 800)
v.show()
v.load_series(dicom_io.load_xa(se, None), "t")
app.processEvents()
for _i in range(N):                            # decode every frame now
    v._planes[0].frame(_i)
v.toggle_long_axis()
app.processEvents()
check("long view on", v._la_visible and v._la_img is not None)


def settle():
    for _ in range(5):
        app.processEvents()
    v._rebuild_long_axis(draft=False)       # the settled, crisp strip
    app.processEvents()


def blob_side(frame=None):
    """'top' (cyan, +cut direction) / 'bottom' (red) — where the blob is in
    the strip column of *frame*."""
    img = v._la_img
    f = v._frame if frame is None else frame
    col = img[:, f].astype(float)
    mid = (len(col) - 1) / 2.0
    rows = np.flatnonzero(col > 200)
    if rows.size == 0:
        return None
    return "top" if rows.mean() < mid else "bottom"


def seen_side():
    """Where the blob is SEEN relative to the cut line: 'top' when it lies on
    the +cut direction side (the cyan triangle) of the screen."""
    c = v.canvas
    bx, by = c.raw_to_view_pt(53.0, 32.0)
    cx, cy = c._ivus_center_view()
    a = v._la_angle
    return "top" if (bx - cx) * math.cos(a) + (by - cy) * math.sin(a) > 0 \
        else "bottom"


def reset_all():
    v._angle_kf.clear()
    v._refresh_angle_marks()
    for c in (v.canvas, v.canvas2):
        c.reset_orient()
        c.set_free_rotation(0.0)
    v._la_angle = 0.0
    settle()


reset_all()
check("no rotation: blob is right = cyan side, strip top",
      seen_side() == "top" and blob_side() == "top",
      (seen_side(), blob_side()))

# Multi-Rot ~180°: the image turns, the screen cut line stays
v._angle_kf[0] = 182.0
v._angle_kf[N - 1] = 182.0
v._refresh_angle_marks()
v._apply_frame_angle()
v.canvas.update()
settle()
check("Multi-Rot 182°: blob is now seen on the red side …",
      seen_side() == "bottom", seen_side())
check("… and the strip shows it at the bottom (red) too",
      blob_side() == "bottom", blob_side())

# per-frame rotation: 0° at frame 0 → 180° at the last frame
v._angle_kf.clear()
v._angle_kf[0] = 0.0
v._angle_kf[N - 1] = 180.0
v._refresh_angle_marks()
settle()
check("per-frame Multi-Rot: frame 0 (0°) top, last frame (180°) bottom",
      blob_side(0) == "top" and blob_side(N - 1) == "bottom",
      (blob_side(0), blob_side(N - 1)))

# flips / 90° turns, with a vertical cut too
for kinds, ang in ((["fliph"], 0.0), (["flipv"], 0.0), (["rt90"], math.pi / 2),
                   (["lt90"], math.pi / 2), (["rt90", "fliph"], math.pi / 2)):
    reset_all()
    for k in kinds:
        v._image_transform(k)
    v._la_angle = ang
    v._refresh_center_marker()
    settle()
    check(f"{'+'.join(kinds)}, cut {round(math.degrees(ang))}°: strip side "
          f"= seen side", blob_side() == seen_side(),
          (blob_side(), seen_side()))

# the strip is rebuilt by itself when the view turns (no manual rebuild)
reset_all()
v._image_transform("fliph")
for _ in range(20):
    app.processEvents()
v._la_wl_timer.stop()
v._on_la_wl_settle()
check("Flip-H re-syncs the strip on its own (blob now bottom)",
      blob_side() == "bottom", blob_side())

# the centre marker rides the anatomy (stored RAW, drawn where seen)
reset_all()
v._on_center_dragged(40.0, 20.0)               # raw centre
v.canvas.set_free_rotation(90.0)
vx, vy = v.canvas._ivus_center_view()
ex, ey = view_xform.raw_to_view(40.0, 20.0, S, S, False, 0, 90.0)
check("centre marker drawn where its raw point is seen (90° turn)",
      abs(vx - ex) < 1e-6 and abs(vy - ey) < 1e-6, ((vx, vy), (ex, ey)))
# dragging on screen stores the RAW point under the cursor
c = v.canvas
c._ivus_dragging_center = True
tx, ty = view_xform.raw_to_view(25.0, 35.0, S, S, False, 0, 90.0)
wx, wy = c._image_to_widget_f((tx, ty))


class _E:
    def __init__(self, x, y):
        from PyQt6.QtCore import QPointF
        self._p = QPointF(x, y)

    def position(self):
        return self._p

    def buttons(self):
        from PyQt6.QtCore import Qt
        return Qt.MouseButton.LeftButton

    def modifiers(self):
        from PyQt6.QtCore import Qt
        return Qt.KeyboardModifier.NoModifier


got = []
c.ivus_center_changed.connect(lambda x, y: got.append((x, y)))
c.mouseMoveEvent(_E(wx, wy))
c._ivus_dragging_center = False
check("dragging the centre on a turned image stores the raw point",
      got and abs(got[-1][0] - 25.0) < 1.5 and abs(got[-1][1] - 35.0) < 1.5,
      got[-1:] if got else None)
v.close()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
sys.stdout.flush()
os._exit(1 if FAILS else 0)
