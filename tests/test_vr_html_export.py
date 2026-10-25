"""Unit tests for core.vr_html_export — the self-contained VR→HTML/GLB exporter.

Qt-free, headless. The marching-cubes path needs scikit-image, so those parts
skip when it is absent; the GLB writer, tube mesh, Taubin smoother and HTML
wrapper are tested unconditionally with hand-built geometry.
"""
import base64
import json
import struct

import numpy as np
import pytest

from multi_dicomviewer.core import vr_html_export as vx

_HAS_SKIMAGE = True
try:
    import skimage  # noqa: F401
except Exception:                                        # noqa: BLE001
    _HAS_SKIMAGE = False


def _tetra():
    """A trivial closed mesh (tetrahedron) for format-layer tests."""
    verts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], int)
    return verts, faces


def test_vertex_normals_unit_length():
    v, f = _tetra()
    n = vx.vertex_normals(v, f)
    assert n.shape == v.shape
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)


def test_tube_mesh_basic():
    line = np.array([[0, 0, z] for z in range(0, 20, 2)], float)
    verts, faces = vx.tube_mesh(line, radius=1.5, sides=10)
    assert verts is not None and len(verts) > 0
    assert faces.max() < len(verts)                      # indices in range
    # Points on the tube body lie ~radius from the axis (x=y=0).
    body = verts[: 10 * len(line)]
    rad = np.hypot(body[:, 0], body[:, 1])
    assert np.allclose(rad, 1.5, atol=1e-6)


def test_build_glb_valid_container():
    v, f = _tetra()
    glb = vx.build_glb([{"positions": v, "indices": f,
                         "color": (1, 0, 0, 1), "name": "t"}])
    magic, ver, total = struct.unpack("<III", glb[:12])
    assert magic == 0x46546C67 and ver == 2 and total == len(glb)
    jlen, jtype = struct.unpack("<II", glb[12:20])
    assert jtype == 0x4E4F534A                           # 'JSON'
    gltf = json.loads(glb[20:20 + jlen].decode("utf-8"))
    assert gltf["asset"]["version"] == "2.0"
    assert len(gltf["meshes"]) == 1 and len(gltf["materials"]) == 1
    # POSITION accessor must carry min/max (glTF spec requirement).
    pos_acc = gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
    assert "min" in gltf["accessors"][pos_acc]
    assert "max" in gltf["accessors"][pos_acc]


def test_self_contained_html_offline():
    v, f = _tetra()
    glb = vx.build_glb([{"positions": v, "indices": f,
                         "color": (0, 1, 0, 1), "name": "t"}])
    html = vx.self_contained_html(glb, title="x", subtitle="y")
    # Everything inlined: three.js present, GLB embedded, no external loaders.
    assert "THREE" in html and "GLB_B64" in html
    assert 'src="http' not in html and "src='http" not in html
    assert 'href="http' not in html and "href='http" not in html
    # The embedded base64 round-trips to the original GLB bytes.
    import re
    m = re.search(r'GLB_B64 = "([^"]+)"', html)
    assert m and base64.b64decode(m.group(1)) == glb


@pytest.mark.skipif(not _HAS_SKIMAGE, reason="scikit-image not installed")
def test_iso_surface_sphere():
    n = 40
    zc = yc = xc = n / 2
    Z, Y, X = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    mask = ((X - xc) ** 2 + (Y - yc) ** 2 + (Z - zc) ** 2) <= 14 ** 2
    verts, faces = vx.iso_surface(mask, (1.0, 1.0, 1.0))
    assert verts is not None and len(verts) > 100
    # Vertices lie ~14 mm from the centre (smoothed sphere).
    r = np.linalg.norm(verts - np.array([xc, yc, zc]), axis=1)
    assert 11.0 < r.mean() < 16.0


@pytest.mark.skipif(not _HAS_SKIMAGE, reason="scikit-image not installed")
def test_build_vr_glb_end_to_end(tmp_path):
    n = 36
    c = n / 2
    Z, Y, X = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    rr = (X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2
    shell = (rr <= 15 ** 2) & (rr >= 9 ** 2)
    lab = np.zeros((n, n, n), np.int32)
    lab[shell & (X >= c)] = 1
    lab[shell & (X < c)] = 2
    colors = [(1, 0.8, 0.2, 1), (0.3, 0.7, 1, 1)]
    vessels = [{"points": np.array([[c, c, z] for z in range(4, 32)], float),
                "color": (1, 0, 0, 1), "radius": 1.0, "name": "v"}]
    out = tmp_path / "scene.html"
    path, stats = vx.export_vr_html(str(out), lab, colors, (1., 1., 1.), vessels)
    assert path == str(out) and out.exists()
    assert stats["territories"] == 2 and stats["vessels"] == 1
    assert out.stat().st_size > 500_000              # three.js inlined
