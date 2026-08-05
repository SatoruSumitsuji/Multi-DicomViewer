"""Binary STL writer for the LV reconstructed surfaces.

The LV endo/epi surfaces already expose a triangulated closed mesh
(``LVSurface.to_mesh()`` → vertices (V,3), faces (F,3)); this just serialises
one or more such meshes into a single binary STL. Coordinates are passed through
unchanged — the LV meshes are in volume millimetres, so the STL is real-scale.
Backend-independent (pure numpy), shared by the VTK and pygfx CT viewers.
"""
from __future__ import annotations

import struct

import numpy as np


def write_stl(path: str, meshes, header: str = "Multi-DicomViewer LV") -> int:
    """Write *meshes* to a binary STL at *path*; return the triangle count.

    *meshes* is an iterable of ``(verts, faces)`` pairs (each verts (V,3) float,
    faces (F,3) int) — pass one for a single surface, or several to merge them
    into one file (e.g. Endo + Epi together). Per-face normals are computed from
    the vertex winding.
    """
    tris = []
    for verts, faces in meshes:
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if len(faces):
            tris.append(verts[faces])          # (F, 3, 3)
    T = (np.concatenate(tris, axis=0) if tris
         else np.zeros((0, 3, 3), dtype=np.float64))
    n = int(len(T))

    # Face normals from the triangle winding (unit; zero-area → zero normal).
    if n:
        nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        ln[ln == 0.0] = 1.0
        nrm = nrm / ln
    else:
        nrm = np.zeros((0, 3), dtype=np.float64)

    rec = np.zeros(n, dtype=np.dtype([
        ("n", "<f4", 3),
        ("v0", "<f4", 3), ("v1", "<f4", 3), ("v2", "<f4", 3),
        ("attr", "<u2"),
    ]))
    if n:
        rec["n"] = nrm.astype("<f4")
        rec["v0"] = T[:, 0].astype("<f4")
        rec["v1"] = T[:, 1].astype("<f4")
        rec["v2"] = T[:, 2].astype("<f4")

    hdr = header.encode("ascii", "replace")[:80]
    hdr = hdr + b" " * (80 - len(hdr))
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(struct.pack("<I", n))
        f.write(rec.tobytes())
    return n
