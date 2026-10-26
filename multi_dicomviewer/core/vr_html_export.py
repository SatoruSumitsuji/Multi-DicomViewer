"""Self-contained HTML export of the CT VR model (perfusion-territory surfaces
+ coronary tubes).

This is a **shared, OS-agnostic** exporter: it takes plain numpy data (a
territory label volume, per-label colours, voxel spacing, and coronary
centrelines) and produces a single ``.html`` file that opens in any browser and
lets the viewer spin the 3-D heart with the mouse — no app, no internet, no
plug-in. It is meant to replace the VTK-only ``.glb`` export so the Mac build
(which has no VTK) can export too, and (once validated) so Windows can share the
same path.

Pipeline (all numpy / scipy, plus scikit-image for marching cubes):

  label volume ──► per-label smooth iso-surface (marching cubes on a Gaussian-
                   smoothed float mask) ──► Taubin mesh smoothing ──► coloured
  coronary centrelines ──► swept tube meshes
  all primitives ──► one in-memory **GLB** (pure-numpy writer)
  GLB ──► base64, embedded in an HTML that inlines a vendored three.js +
          OrbitControls + GLTFLoader (offline) with a simple orbit viewer.

Smoothness (the user's priority) is governed here, not by the file format:
  (a) the binary mask is blurred to a float field before marching (marching a
      hard 0/1 mask stair-steps); (b) the mesh is Taubin-smoothed; (c) smooth
      vertex normals are written so the viewer shades it smoothly.

No Qt, no VTK, no GUI — safe to unit-test headless.
"""

from __future__ import annotations

import base64
import json
import os
import struct

import numpy as np

# --- tuning defaults -------------------------------------------------------
_MASK_BLUR_SIGMA_VOX = 0.8      # Gaussian blur (voxels) before marching cubes
_TAUBIN_ITERS = 12             # mesh smoothing passes
_TAUBIN_LAMBDA = 0.5
_TAUBIN_MU = -0.53             # |mu| slightly > lambda = volume-preserving
_TARGET_MAX_VERTS = 120_000    # per-surface vertex budget (downsample volume)
_TUBE_SIDES = 12               # cross-section polygon sides for coronary tubes


# ===========================================================================
# Mesh generation
# ===========================================================================
def _downsample_factor(mask, target_max_verts):
    """A crude zoom factor (<=1) that keeps a surface's vertex count near the
    budget: iso-surface area ~ (#boundary voxels) scales ~ n^2 in a linear
    downsample, so shrink by sqrt of the overshoot. Returns 1.0 when fine."""
    try:
        import scipy.ndimage as ndi  # noqa: F401
    except Exception:                                    # noqa: BLE001
        return 1.0
    nb = int(np.count_nonzero(mask))
    if nb <= 0:
        return 1.0
    # ~boundary voxels as a rough vert proxy; assume ~ nb**(2/3) surface voxels.
    approx_verts = max(1.0, nb ** (2.0 / 3.0) * 2.0)
    if approx_verts <= target_max_verts:
        return 1.0
    f = (target_max_verts / approx_verts) ** 0.5
    return float(max(0.25, min(1.0, f)))


def iso_surface(mask, spacing_xyz, *, blur_sigma_vox=_MASK_BLUR_SIGMA_VOX,
                taubin_iters=_TAUBIN_ITERS,
                target_max_verts=_TARGET_MAX_VERTS):
    """Smooth iso-surface of a boolean/int *mask* (`[z, y, x]`).

    *spacing_xyz* = (sx, sy, sz) mm per voxel. Returns ``(verts, faces)`` with
    ``verts`` in **world mm** ordered (X, Y, Z) = (x·sx, y·sy, z·sz) — the same
    frame the live VR overlay uses — and ``faces`` an (K,3) int array wound
    counter-clockwise (outward normals). ``(None, None)`` if the mask is empty
    or scikit-image is unavailable."""
    try:
        from skimage import measure
    except Exception:                                    # noqa: BLE001
        return None, None
    import scipy.ndimage as ndi

    m = np.asarray(mask)
    if m.dtype != bool:
        m = m != 0
    if not m.any():
        return None, None

    sx, sy, sz = (float(s) for s in spacing_xyz)

    # Crop to the mask's bounding box (+2 voxel pad so the surface closes).
    zz, yy, xx = np.where(m)
    z0, z1 = max(0, zz.min() - 2), min(m.shape[0], zz.max() + 3)
    y0, y1 = max(0, yy.min() - 2), min(m.shape[1], yy.max() + 3)
    x0, x1 = max(0, xx.min() - 2), min(m.shape[2], xx.max() + 3)
    sub = m[z0:z1, y0:y1, x0:x1].astype(np.float32)

    # Optional isotropic downsample to hold the vertex budget (also smooths).
    f = _downsample_factor(sub, target_max_verts)
    if f < 0.999:
        sub = ndi.zoom(sub, f, order=1)
        szf, syf, sxf = sz / f, sy / f, sx / f
    else:
        szf, syf, sxf = sz, sy, sx

    # Blur the hard mask to a float field so marching cubes doesn't stair-step.
    field = ndi.gaussian_filter(sub, sigma=blur_sigma_vox)
    if field.max() <= 0.5 or field.min() >= 0.5:
        # Degenerate after blur (tiny structure) — fall back to the raw mask.
        field = sub
    try:
        # spacing is per-axis of the array = (z, y, x).
        verts, faces, _normals, _vals = measure.marching_cubes(
            field, level=0.5, spacing=(szf, syf, sxf))
    except Exception:                                    # noqa: BLE001
        return None, None
    if verts is None or len(verts) == 0 or len(faces) == 0:
        return None, None

    # marching_cubes verts columns are (z, y, x)·spacing; shift by the crop
    # origin (in mm) and reorder to (X, Y, Z) = (x, y, z).
    verts = np.asarray(verts, np.float64)
    verts[:, 0] += z0 * sz
    verts[:, 1] += y0 * sy
    verts[:, 2] += x0 * sx
    verts = verts[:, [2, 1, 0]]                 # (z,y,x) -> (x,y,z)
    faces = np.asarray(faces, np.int64)[:, [0, 2, 1]]   # reverse winding (handedness flip)

    if taubin_iters > 0:
        verts = taubin_smooth(verts, faces, iters=taubin_iters)
    return verts.astype(np.float32), faces.astype(np.int64)


def taubin_smooth(verts, faces, *, iters=_TAUBIN_ITERS,
                  lamb=_TAUBIN_LAMBDA, mu=_TAUBIN_MU):
    """Taubin λ|μ smoothing (low-pass, approximately volume-preserving).

    Uniform umbrella Laplacian: each vertex moves toward the average of its
    neighbours by +λ then −|μ|, which smooths without the shrink a plain
    Laplacian causes. Pure numpy; builds the neighbour-sum via scatter-add."""
    V = np.asarray(verts, np.float64)
    F = np.asarray(faces, np.int64)
    n = len(V)
    # Undirected edges from triangle sides.
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    e = np.concatenate([e, e[:, ::-1]], axis=0)          # both directions
    i, j = e[:, 0], e[:, 1]
    deg = np.bincount(i, minlength=n).astype(np.float64)
    deg[deg == 0] = 1.0

    def _umbrella(P):
        s = np.zeros_like(P)
        np.add.at(s, i, P[j])
        return s / deg[:, None] - P

    for _ in range(int(iters)):
        V = V + lamb * _umbrella(V)
        V = V + mu * _umbrella(V)
    return V


def vertex_normals(verts, faces):
    """Area-weighted smooth vertex normals (unit length)."""
    V = np.asarray(verts, np.float64)
    F = np.asarray(faces, np.int64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nrm = np.zeros_like(V)
    for k in range(3):
        np.add.at(nrm, F[:, k], fn)
    ln = np.linalg.norm(nrm, axis=1)
    ln[ln == 0] = 1.0
    return (nrm / ln[:, None]).astype(np.float32)


def tube_mesh(polyline, radius, *, sides=_TUBE_SIDES):
    """A swept circular tube of *radius* mm around *polyline* ((N,3) world mm).

    Rotation-minimising (parallel-transport) frames avoid twist. Returns
    ``(verts, faces)`` or ``(None, None)`` for a degenerate line. End caps are
    added so the tube reads solid."""
    P = np.asarray(polyline, np.float64).reshape(-1, 3)
    # Drop consecutive duplicates.
    keep = np.concatenate([[True], (np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6)])
    P = P[keep]
    if len(P) < 2:
        return None, None
    N = len(P)
    # Per-vertex tangents (central differences).
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    tl = np.linalg.norm(T, axis=1)
    tl[tl == 0] = 1.0
    T = T / tl[:, None]
    # Parallel-transport an initial normal along the curve.
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, T[0])) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    nrm0 = np.cross(T[0], ref)
    nrm0 /= (np.linalg.norm(nrm0) or 1.0)
    normals = [nrm0]
    for k in range(1, N):
        v = np.cross(T[k - 1], T[k])
        nv = np.linalg.norm(v)
        nprev = normals[-1]
        if nv < 1e-8:
            nk = nprev
        else:
            axis = v / nv
            ang = np.arctan2(nv, np.dot(T[k - 1], T[k]))
            nk = (nprev * np.cos(ang)
                  + np.cross(axis, nprev) * np.sin(ang)
                  + axis * np.dot(axis, nprev) * (1 - np.cos(ang)))
        nk = nk - np.dot(nk, T[k]) * T[k]
        nk /= (np.linalg.norm(nk) or 1.0)
        normals.append(nk)
    normals = np.asarray(normals)
    binorm = np.cross(T, normals)

    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    ring = (normals[:, None, :] * ca[None, :, None]
            + binorm[:, None, :] * sa[None, :, None])      # (N, sides, 3)
    verts = (P[:, None, :] + radius * ring).reshape(-1, 3)

    faces = []
    for k in range(N - 1):
        a0 = k * sides
        a1 = (k + 1) * sides
        for s in range(sides):
            s2 = (s + 1) % sides
            faces.append([a0 + s, a1 + s, a1 + s2])
            faces.append([a0 + s, a1 + s2, a0 + s2])
    # Flat end caps (fan).
    c0 = len(verts)
    c1 = c0 + 1
    verts = np.vstack([verts, P[0], P[-1]])
    for s in range(sides):
        s2 = (s + 1) % sides
        faces.append([c0, s2, s])                          # start cap
        base = (N - 1) * sides
        faces.append([c1, base + s, base + s2])            # end cap
    return verts.astype(np.float32), np.asarray(faces, np.int64)


def sphere_mesh(center, radius, *, n_lat=12, n_lon=18):
    """A small UV sphere (for a Territory target marker). Returns (verts, faces)
    in world mm, centred on *center*."""
    cx, cy, cz = (float(c) for c in center)
    r = float(radius)
    lat = np.linspace(0.0, np.pi, n_lat + 1)
    lon = np.linspace(0.0, 2 * np.pi, n_lon, endpoint=False)
    TH, LO = np.meshgrid(lat, lon, indexing="ij")        # (n_lat+1, n_lon)
    X = cx + r * np.sin(TH) * np.cos(LO)
    Y = cy + r * np.sin(TH) * np.sin(LO)
    Z = cz + r * np.cos(TH)
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            a = i * n_lon + j
            b = i * n_lon + j2
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + j2
            faces.append([a, c, d])
            faces.append([a, d, b])
    return verts, np.asarray(faces, np.int64)


# ===========================================================================
# GLB writer (pure numpy)
# ===========================================================================
_CT_FLOAT = 5126
_CT_UINT = 5125
_TGT_ARRAY = 34962
_TGT_ELEMENT = 34963


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    r = (4 - len(b) % 4) % 4
    return b + fill * r


def build_glb(primitives, *, generator="Multi-DICOMviewer", extras=None) -> bytes:
    """Pack *primitives* into a single binary ``.glb`` (glTF 2.0).

    Each primitive is a dict:
      ``positions`` (M,3) float, ``indices`` (K,3) int, ``normals`` (M,3) float
      (optional, computed if absent), ``color`` (r,g,b,a) 0..1, ``name`` str,
      ``blend`` bool (translucent), ``double_sided`` bool,
      ``metallic``/``roughness`` floats.
    Returns the GLB bytes."""
    bin_parts = []
    offset = 0
    buffer_views = []
    accessors = []
    meshes = []
    nodes = []
    materials = []

    def _add_view(data_bytes, target):
        nonlocal offset
        data_bytes = _pad4(data_bytes)
        bv = {"buffer": 0, "byteOffset": offset, "byteLength": len(data_bytes)}
        if target is not None:
            bv["target"] = target
        buffer_views.append(bv)
        bin_parts.append(data_bytes)
        offset += len(data_bytes)
        return len(buffer_views) - 1

    for prim in primitives:
        pos = np.ascontiguousarray(np.asarray(prim["positions"], np.float32))
        idx_tri = np.asarray(prim["indices"], np.int64).reshape(-1, 3)
        idx = np.ascontiguousarray(idx_tri.reshape(-1).astype(np.uint32))
        nrm = prim.get("normals")
        if nrm is None:
            nrm = vertex_normals(pos, idx_tri)
        nrm = np.ascontiguousarray(np.asarray(nrm, np.float32))

        # bufferViews
        pos_bv = _add_view(pos.tobytes(), _TGT_ARRAY)
        nrm_bv = _add_view(nrm.tobytes(), _TGT_ARRAY)
        idx_bv = _add_view(idx.tobytes(), _TGT_ELEMENT)

        pmin = pos.min(axis=0).tolist()
        pmax = pos.max(axis=0).tolist()
        pos_acc = len(accessors)
        accessors.append({"bufferView": pos_bv, "componentType": _CT_FLOAT,
                          "count": int(len(pos)), "type": "VEC3",
                          "min": pmin, "max": pmax})
        nrm_acc = len(accessors)
        accessors.append({"bufferView": nrm_bv, "componentType": _CT_FLOAT,
                          "count": int(len(nrm)), "type": "VEC3"})
        idx_acc = len(accessors)
        accessors.append({"bufferView": idx_bv, "componentType": _CT_UINT,
                          "count": int(len(idx)), "type": "SCALAR"})

        col = [float(c) for c in prim.get("color", (0.8, 0.8, 0.8, 1.0))]
        if len(col) == 3:
            col = col + [1.0]
        blend = bool(prim.get("blend", col[3] < 0.999))
        mat = {
            "name": str(prim.get("name", "mat")),
            "pbrMetallicRoughness": {
                "baseColorFactor": col,
                "metallicFactor": float(prim.get("metallic", 0.0)),
                "roughnessFactor": float(prim.get("roughness", 0.75)),
            },
            "doubleSided": bool(prim.get("double_sided", True)),
        }
        if blend:
            mat["alphaMode"] = "BLEND"
        mat_idx = len(materials)
        materials.append(mat)

        mesh_idx = len(meshes)
        meshes.append({"name": str(prim.get("name", "mesh")),
                       "primitives": [{"attributes": {"POSITION": pos_acc,
                                                       "NORMAL": nrm_acc},
                                       "indices": idx_acc,
                                       "material": mat_idx}]})
        nodes.append({"name": str(prim.get("name", "node")), "mesh": mesh_idx})

    bin_blob = b"".join(bin_parts)
    gltf = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_blob)}],
    }
    if extras:
        gltf["asset"]["extras"] = extras

    json_bytes = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"),
                       fill=b" ")
    bin_bytes = _pad4(bin_blob)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, total)          # 'glTF', v2, len
    out += struct.pack("<II", len(json_bytes), 0x4E4F534A)    # JSON chunk
    out += json_bytes
    out += struct.pack("<II", len(bin_bytes), 0x004E4942)     # BIN chunk
    out += bin_bytes
    return bytes(out)


# ===========================================================================
# Self-contained HTML
# ===========================================================================
def _webgl_asset(name: str) -> str:
    """Read a vendored three.js asset from multi_dicomviewer/resources/webgl."""
    here = os.path.dirname(os.path.abspath(__file__))        # …/core
    root = os.path.dirname(here)                             # …/multi_dicomviewer
    path = os.path.join(root, "resources", "webgl", name)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def self_contained_html(glb_bytes: bytes, *, title="Multi-DICOMviewer VR",
                        subtitle="", bg="#0b0d10", summary_tr="",
                        summary_br="", targets=(), pbasis=None) -> str:
    """Wrap *glb_bytes* in one offline HTML with an orbit viewer.

    Reproduces the in-app VR corner readouts: *summary_tr* (top-right, the
    per-system perfusion summary), *summary_br* (bottom-right, the per-target
    list), the Territory *targets* as number labels that track their 3-D marker,
    and — when *pbasis* (3×3 volume→patient) is given — a LIVE LAO/RAO · CRA/CAU
    readout that updates as the model turns. Controls: drag=rotate, wheel=zoom,
    double-click / 視点リセット=reset, 自動回転 toggle, 背景 cycle."""
    three_js = _webgl_asset("three.min.js")
    orbit_js = _webgl_asset("OrbitControls.js")
    gltf_js = _webgl_asset("GLTFLoader.js")
    glb_b64 = base64.b64encode(glb_bytes).decode("ascii")
    sub = _esc(subtitle)
    ttl = _esc(title)
    tr_html = _esc(summary_tr)
    br_html = _esc(summary_br)
    tg_json = json.dumps([
        {"x": float(t["point"][0]), "y": float(t["point"][1]),
         "z": float(t["point"][2]), "n": str(t.get("n", "")),
         "color": t.get("color_hex", "#ea9999")}
        for t in (targets or []) if t.get("point") is not None])
    pb_json = json.dumps([[float(x) for x in row] for row in pbasis]) \
        if pbasis is not None else "null"

    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{ttl}</title>
<style>
  html,body{{margin:0;height:100%;background:{bg};color:#e8e8e8;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;overflow:hidden}}
  #c{{position:fixed;inset:0}}
  .hud{{position:fixed;text-shadow:0 1px 3px #000,0 0 2px #000;pointer-events:none;
    white-space:pre;font-size:19.5px;line-height:1.45}}
  #tl{{top:10px;left:12px;max-width:52vw}}
  #tl b{{font-size:22.5px}}
  #tr{{top:10px;right:12px;text-align:left;color:#ffd900;font-family:monospace;
    font-weight:bold;max-width:52vw}}
  #br{{bottom:40px;right:12px;text-align:left;color:#ffd900;font-family:monospace;
    font-weight:bold;max-width:52vw}}
  #angio{{bottom:12px;left:50%;transform:translateX(-50%);color:#ffd900;
    font-family:monospace;font-size:25.5px;font-weight:bold}}
  #labels{{position:fixed;inset:0;pointer-events:none;overflow:hidden}}
  #labels span{{position:absolute;font-family:monospace;font-weight:bold;
    font-size:22.5px;text-shadow:0 1px 3px #000,0 0 2px #000,1px 0 2px #000;
    transform:translate(8px,-20px);white-space:nowrap}}
  #bar{{position:fixed;bottom:12px;left:12px;display:flex;gap:8px}}
  #bar button{{background:#1f2630;color:#e8e8e8;border:1px solid #39434f;
    border-radius:7px;padding:7px 13px;font-size:19.5px;cursor:pointer}}
  #bar button:hover{{background:#2b3440}}
  #bar button.on{{background:#2d5fa8;border-color:#3f7ad0}}
  #tip{{position:fixed;top:10px;left:50%;transform:translateX(-50%);font-size:18px;
    color:#9aa4b0;text-shadow:0 1px 3px #000}}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="labels"></div>
<div class="hud" id="tl"><b>{ttl}</b>{(chr(10) + sub) if sub else ""}</div>
<div class="hud" id="tr">{tr_html}</div>
<div class="hud" id="br">{br_html}</div>
<div class="hud" id="angio"></div>
<div id="tip">ドラッグ=回転 ・ ホイール=拡大縮小 ・ 右ドラッグ=平行移動</div>
<div id="bar">
  <button id="spin" class="on">自動回転: ON</button>
  <button id="reset">視点リセット</button>
  <button id="bgbtn">背景</button>
</div>
<script>{three_js}</script>
<script>{orbit_js}</script>
<script>{gltf_js}</script>
<script>
const GLB_B64 = "{glb_b64}";
const TARGETS = {tg_json};
const PBASIS = {pb_json};
function b64ToArrayBuffer(b64){{
  const bin = atob(b64); const len = bin.length;
  const bytes = new Uint8Array(len);
  for(let i=0;i<len;i++) bytes[i]=bin.charCodeAt(i);
  return bytes.buffer;
}}
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
renderer.setPixelRatio(Math.min(window.devicePixelRatio||1, 2));
const bgColors = ["{bg}", "#ffffff", "#303338"];
let bgIdx = 0;
const scene = new THREE.Scene();
scene.background = new THREE.Color(bgColors[0]);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100000);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 2.2;

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 0.85); key.position.set(1,1,1);
const fill = new THREE.DirectionalLight(0xffffff, 0.45); fill.position.set(-1,-0.5,-1);
const rim = new THREE.DirectionalLight(0xffffff, 0.35); rim.position.set(0,1,-1);
scene.add(key); scene.add(fill); scene.add(rim);

const home = new THREE.Vector3(100,60,120);
const modelCenter = new THREE.Vector3();   // model centre offset (subtracted)
let modelRadius = 100;

// Target number labels (DOM, track their 3-D marker).
const labelBox = document.getElementById('labels');
const labelEls = TARGETS.map(t=>{{
  const el = document.createElement('span');
  el.textContent = t.n; el.style.color = t.color;
  labelBox.appendChild(el); return el;
}});

const loader = new THREE.GLTFLoader();
loader.parse(b64ToArrayBuffer(GLB_B64), '', (gltf)=>{{
  const obj = gltf.scene;
  const box = new THREE.Box3().setFromObject(obj);
  box.getCenter(modelCenter);
  const size = box.getSize(new THREE.Vector3());
  obj.position.sub(modelCenter);
  scene.add(obj);
  modelRadius = 0.5*Math.max(size.x,size.y,size.z) || 100;
  const dist = modelRadius / Math.sin((45*Math.PI/180)/2);
  camera.position.set(dist*0.7, dist*0.4, dist*0.9);
  camera.near = modelRadius/100; camera.far = modelRadius*100;
  camera.updateProjectionMatrix();
  controls.target.set(0,0,0); controls.update();
  home.copy(camera.position);
}}, (err)=>{{
  document.getElementById('tl').innerHTML += '\\n<span style="color:#ff8">読み込みエラー</span>';
  console.error(err);
}});

function resize(){{
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix();
}}
window.addEventListener('resize', resize); resize();

document.getElementById('spin').onclick = (e)=>{{
  controls.autoRotate = !controls.autoRotate;
  e.target.classList.toggle('on', controls.autoRotate);
  e.target.textContent = '自動回転: ' + (controls.autoRotate?'ON':'OFF');
}};
function resetView(){{ camera.position.copy(home); controls.target.set(0,0,0); controls.update(); }}
document.getElementById('reset').onclick = resetView;
renderer.domElement.addEventListener('dblclick', resetView);
document.getElementById('bgbtn').onclick = ()=>{{
  bgIdx = (bgIdx+1)%bgColors.length;
  scene.background = new THREE.Color(bgColors[bgIdx]);
}};

// --- C-arm LAO/RAO · CRA/CAU readout (same math as the app) ----------------
const angioEl = document.getElementById('angio');
function updateAngio(){{
  if(!PBASIS){{ return; }}
  // viewing normal = toward the camera (world) = app's "back"
  const d = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
  const nx = PBASIS[0][0]*d.x + PBASIS[0][1]*d.y + PBASIS[0][2]*d.z;
  const ny = PBASIS[1][0]*d.x + PBASIS[1][1]*d.y + PBASIS[1][2]*d.z;
  const nz = PBASIS[2][0]*d.x + PBASIS[2][1]*d.y + PBASIS[2][2]*d.z;
  const axial = Math.hypot(nx, ny);
  const prim = axial<1e-9 ? 0 : Math.round(Math.atan2(nx, -ny)*180/Math.PI);
  const sec = Math.round(Math.atan2(nz, axial)*180/Math.PI);
  const lao = prim>=0 ? ('LAO'+prim) : ('RAO'+(-prim));
  const cra = sec>=0 ? ('CRA'+sec) : ('CAU'+(-sec));
  angioEl.textContent = lao+' '+cra;
}}

// --- target number labels --------------------------------------------------
const _p = new THREE.Vector3();
function updateLabels(){{
  const w = window.innerWidth, h = window.innerHeight;
  const viewDir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
  for(let i=0;i<TARGETS.length;i++){{
    const t = TARGETS[i];
    _p.set(t.x-modelCenter.x, t.y-modelCenter.y, t.z-modelCenter.z);
    // Hide a marker on the far side of the heart (approx: its outward dir
    // from the model centre points away from the camera).
    const facing = _p.x*viewDir.x + _p.y*viewDir.y + _p.z*viewDir.z;
    const proj = _p.clone().project(camera);
    const onscreen = proj.z<1 && Math.abs(proj.x)<1.05 && Math.abs(proj.y)<1.05;
    if(facing < -0.15*_p.length() || !onscreen){{ labelEls[i].style.display='none'; continue; }}
    labelEls[i].style.display='block';
    labelEls[i].style.left = ((proj.x*0.5+0.5)*w)+'px';
    labelEls[i].style.top  = ((-proj.y*0.5+0.5)*h)+'px';
  }}
}}

(function loop(){{
  requestAnimationFrame(loop);
  controls.update();
  updateAngio();
  updateLabels();
  renderer.render(scene, camera);
}})();
</script>
</body>
</html>
"""


# ===========================================================================
# High-level orchestration (shared by both viewers)
# ===========================================================================
def build_vr_glb(label_vol, colors, spacing_xyz, vessels=(), *,
                 targets=(), territory_alpha=1.0, generator="Multi-DICOMviewer",
                 extras=None):
    """Build GLB bytes from a territory label volume + coronary centrelines.

    *label_vol*  : `[z,y,x]` int labels (0 = background, 1..K = territories).
    *colors*     : list of (r,g,b[,a]) 0..1, one per label 1..K.
    *spacing_xyz*: (sx, sy, sz) mm/voxel.
    *vessels*    : iterable of dicts ``{"points":(N,3) world mm, "color":rgba,
                   "radius":mm, "name":str}`` for coronary tubes.
    *targets*    : iterable of dicts ``{"point":(x,y,z) world mm, "color":rgba,
                   "radius":mm, "n":str}`` → a coloured marker sphere each.
    Returns ``(glb_bytes, stats)`` where stats lists what was built."""
    prims = []
    stats = {"territories": 0, "vessels": 0, "targets": 0, "verts": 0, "faces": 0}
    lab = np.rint(np.asarray(label_vol)).astype(np.int32) if label_vol is not None else None
    if lab is not None and colors:
        for i, c in enumerate(colors, 1):
            mask = (lab == i)
            if not mask.any():
                continue
            verts, faces = iso_surface(mask, spacing_xyz)
            if verts is None:
                continue
            # Territory surfaces are drawn OPAQUE by default: they are the outer
            # epicardial shell partitioned by vessel, so there is nothing useful
            # behind them to show through, and opaque avoids transparency-sort
            # artifacts while spinning. (The per-label alpha in *colors* — ~0.33
            # for the live point-cloud wash — is intentionally overridden.)
            rgba = [float(c[0]), float(c[1]), float(c[2]), float(territory_alpha)]
            prims.append({"positions": verts, "indices": faces,
                          "color": rgba, "name": f"territory_{i}",
                          "blend": rgba[3] < 0.999, "double_sided": True,
                          "roughness": 0.8})
            stats["territories"] += 1
            stats["verts"] += len(verts)
            stats["faces"] += len(faces)
    for v in (vessels or []):
        pts = v.get("points")
        if pts is None or len(pts) < 2:
            continue
        r = float(v.get("radius", 1.2))
        verts, faces = tube_mesh(pts, r)
        if verts is None:
            continue
        col = v.get("color", (0.9, 0.2, 0.2, 1.0))
        col = [float(col[0]), float(col[1]), float(col[2]),
               float(col[3]) if len(col) > 3 else 1.0]
        prims.append({"positions": verts, "indices": faces, "color": col,
                      "name": v.get("name", "vessel"), "blend": False,
                      "double_sided": True, "roughness": 0.5, "metallic": 0.1})
        stats["vessels"] += 1
        stats["verts"] += len(verts)
        stats["faces"] += len(faces)
    for tg in (targets or []):
        P = tg.get("point")
        if P is None:
            continue
        r = float(tg.get("radius", 2.5))
        verts, faces = sphere_mesh(np.asarray(P, float).reshape(3), r)
        col = tg.get("color", (0.92, 0.6, 0.6, 1.0))
        col = [float(col[0]), float(col[1]), float(col[2]),
               float(col[3]) if len(col) > 3 else 1.0]
        prims.append({"positions": verts, "indices": faces, "color": col,
                      "name": "target_" + str(tg.get("n", "")), "blend": False,
                      "double_sided": True, "roughness": 0.4, "metallic": 0.0})
        stats["targets"] += 1
        stats["verts"] += len(verts)
        stats["faces"] += len(faces)
    if not prims:
        return None, stats
    return build_glb(prims, generator=generator, extras=extras), stats


def export_vr_html(path, label_vol, colors, spacing_xyz, vessels=(), *,
                   targets=(), summary_tr="", summary_br="", pbasis=None,
                   title="Multi-DICOMviewer VR", subtitle="", extras=None):
    """Full pipeline → write a self-contained ``.html`` (or ``.glb`` if *path*
    ends in .glb).

    *targets* items carry both the 3-D marker (``point``, ``color`` rgba,
    ``radius``) and its label (``n``, ``color_hex``). *summary_tr*/*summary_br*
    are the top-right / bottom-right readout texts; *pbasis* (3×3 volume→patient)
    drives the live LAO/CRA readout. Returns ``(written_path, stats)`` or
    ``(None, stats)`` when there was nothing to export."""
    glb, stats = build_vr_glb(label_vol, colors, spacing_xyz, vessels,
                              targets=targets, extras=extras)
    if glb is None:
        return None, stats
    if str(path).lower().endswith(".glb"):
        with open(path, "wb") as f:
            f.write(glb)
        return path, stats
    html = self_contained_html(glb, title=title, subtitle=subtitle,
                               summary_tr=summary_tr, summary_br=summary_br,
                               targets=targets, pbasis=pbasis)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path, stats
