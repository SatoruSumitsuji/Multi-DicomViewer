"""Coronary perfusion-territory core (headless, GUI-free).

Model
-----
A set of coronary centrelines organised into up to three trees whose ROOTS are
the explicit vessels ``LM-LAD`` / ``LM-LCX`` / ``RCA`` (LM-LCX also covers the
LM-absent "LCX from aorta" case).  Every other vessel is a *branch* that
attaches to an already-drawn vessel by snapping its PROXIMAL end (its first
sample) to the nearest sample on any existing vessel — roots *or* branches, so
multi-level sub-branches (D9→D9a, X12→X12a, R4→R16a…) are supported.

Direction convention: a vessel's ``points`` run PROXIMAL → DISTAL, so index 0
is proximal (an ostium for a root, the junction on the parent for a branch) and
the last index is the distal free end.

Territory
---------
Given a myocardial voxel cloud, each voxel is assigned to its nearest centreline
sample (a KD-tree nearest-neighbour = the Voronoi allocation).  The perfusion
territory *distal to* a chosen centreline point (vessel ``vid``, sample ``idx``)
is every voxel whose assigned sample is distal to that point along the tree:
the rest of that vessel (samples ≥ idx) plus every downstream sub-branch in
full, recursively.

Coordinates match the rest of the app: world-mm = voxel_index · spacing, origin
0, arrays indexed ``[z, y, x]``; there is no affine.  Centreline ``points`` are
(M, 3) float world-mm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:                                    # cKDTree is the fast path (scipy is a dep)
    from scipy.spatial import cKDTree as _KDTree
except Exception:                       # pragma: no cover - brute-force fallback
    _KDTree = None

#: The three roles that start a NEW tree (an explicit ostial trunk).
ROOT_ROLES = ("LM-LAD", "LM-LCX", "RCA")


@dataclass
class Vessel:
    """One coronary centreline. ``points`` (M,3) world-mm run proximal→distal."""

    vid: str
    name: str
    role: str                           # a ROOT_ROLES value, or "branch"
    points: np.ndarray
    parent: str | None = None           # parent vessel id (None for a root)
    junction: int | None = None         # sample index ON THE PARENT at the branch

    def __post_init__(self):
        self.points = np.asarray(self.points, float).reshape(-1, 3)

    @property
    def n(self) -> int:
        return int(len(self.points))


class CoronaryTree:
    """A collection of coronary vessels forming up to three trees."""

    def __init__(self):
        # Insertion order == draw order; a branch may only attach to a vessel
        # already in the dict (parents are always drawn before their children).
        self.vessels: dict[str, Vessel] = {}

    # ------------------------------------------------------------ build
    def add_root(self, vid: str, name: str, role: str, points) -> Vessel:
        """Add an explicit root trunk (LM-LAD / LM-LCX / RCA). First point is the
        ostium (proximal)."""
        if role not in ROOT_ROLES:
            raise ValueError(f"root role must be one of {ROOT_ROLES!r}, got {role!r}")
        if vid in self.vessels:
            raise ValueError(f"duplicate vessel id {vid!r}")
        v = Vessel(vid=vid, name=name, role=role, points=points)
        self.vessels[vid] = v
        return v

    def add_branch(self, vid: str, name: str, points,
                   snap_tol_mm: float = 3.0) -> dict:
        """Attach a branch by snapping its FIRST point to the nearest sample
        among all existing vessels → that vessel becomes the parent and the
        nearest sample index the junction.

        Returns a dict ``{ok, parent, junction, dist_mm, reason}``. When the
        nearest existing sample is farther than ``snap_tol_mm`` the vessel is
        NOT added and ``ok`` is False with ``reason='too-far'`` (so the UI can
        ask the user to drag the start onto a vessel and retry)."""
        pts = np.asarray(points, float).reshape(-1, 3)
        if pts.size == 0:
            return {"ok": False, "reason": "empty"}
        if not self.vessels:
            return {"ok": False, "reason": "no-parent"}
        if vid in self.vessels:
            raise ValueError(f"duplicate vessel id {vid!r}")
        parent, jidx, dist = self._nearest_sample(pts[0])
        if dist > snap_tol_mm:
            return {"ok": False, "reason": "too-far", "dist_mm": float(dist),
                    "parent": parent, "junction": int(jidx)}
        self.vessels[vid] = Vessel(vid=vid, name=name, role="branch",
                                   points=pts, parent=parent, junction=int(jidx))
        return {"ok": True, "parent": parent, "junction": int(jidx),
                "dist_mm": float(dist)}

    def set_parent(self, vid: str, parent: str, junction: int | None = None):
        """Manual override of a branch's parent (fallback when auto-snap picked
        the wrong vessel). ``junction`` defaults to the nearest sample on the
        new parent to this vessel's proximal end."""
        v = self.vessels[vid]
        p = self.vessels[parent]
        if junction is None:
            d = np.linalg.norm(p.points - v.points[0], axis=1)
            junction = int(np.argmin(d))
        v.parent = parent
        v.junction = int(junction)
        v.role = "branch"

    def _nearest_sample(self, xyz):
        """(vid, sample_index, dist_mm) of the nearest sample across all vessels."""
        xyz = np.asarray(xyz, float)
        best = (None, -1, np.inf)
        for vid, v in self.vessels.items():
            d = np.linalg.norm(v.points - xyz, axis=1)
            j = int(np.argmin(d))
            if d[j] < best[2]:
                best = (vid, j, float(d[j]))
        return best

    # --------------------------------------------------------- topology
    def roots(self) -> list[str]:
        return [vid for vid, v in self.vessels.items() if v.parent is None]

    def children(self, vid: str) -> list[str]:
        return [c for c, v in self.vessels.items() if v.parent == vid]

    def descendants(self, vid: str) -> list[str]:
        """All vessels downstream of ``vid`` (its whole sub-tree, exclusive)."""
        out, stack = [], list(self.children(vid))
        while stack:
            c = stack.pop()
            out.append(c)
            stack.extend(self.children(c))
        return out

    # ------------------------------------------------------- assignment
    def _stack(self):
        """Concatenate every vessel's samples into one point cloud for the
        KD-tree. Returns (pts (P,3), owner_code (P,) int, owner_idx (P,) int,
        code_to_vid list) where owner_code indexes code_to_vid."""
        pts, code, idx, code_to_vid = [], [], [], []
        for c, (vid, v) in enumerate(self.vessels.items()):
            pts.append(v.points)
            code.append(np.full(v.n, c, int))
            idx.append(np.arange(v.n))
            code_to_vid.append(vid)
        if not pts:
            return (np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int), [])
        return (np.vstack(pts), np.concatenate(code),
                np.concatenate(idx), code_to_vid)

    def assign(self, voxel_xyz, max_dist_mm: float | None = None) -> dict:
        """Nearest-centreline assignment for a voxel cloud.

        ``voxel_xyz`` is (V,3) world-mm. Returns a dict with:
          - ``code`` (V,) int — vessel code (index into ``code_to_vid``), or -1
            when unassigned (no vessels, or farther than ``max_dist_mm``).
          - ``idx``  (V,) int — nearest sample index on that vessel (-1 if none).
          - ``dist`` (V,) float — distance in mm.
          - ``code_to_vid`` list — code → vessel id.
        """
        V = np.asarray(voxel_xyz, float).reshape(-1, 3)
        P, owner_code, owner_idx, code_to_vid = self._stack()
        if len(P) == 0 or len(V) == 0:
            return {"code": np.full(len(V), -1, int),
                    "idx": np.full(len(V), -1, int),
                    "dist": np.full(len(V), np.inf),
                    "code_to_vid": code_to_vid}
        if _KDTree is not None:
            dist, nn = _KDTree(P).query(V, k=1)
        else:                            # pragma: no cover - fallback
            d2 = ((V[:, None, :] - P[None, :, :]) ** 2).sum(-1)
            nn = d2.argmin(1)
            dist = np.sqrt(d2[np.arange(len(V)), nn])
        code = owner_code[nn].copy()
        idx = owner_idx[nn].copy()
        dist = np.asarray(dist, float)
        if max_dist_mm is not None:
            far = dist > max_dist_mm
            code[far] = -1
            idx[far] = -1
        return {"code": code, "idx": idx, "dist": dist,
                "code_to_vid": code_to_vid}

    # -------------------------------------------------- distal territory
    def distal_set(self, vid: str, idx: int) -> tuple[str, int, set[str]]:
        """For a chosen point (vessel ``vid``, sample ``idx``) return
        ``(vid, start_idx, full)``: on ``vid`` samples ≥ ``start_idx`` are
        distal, and every vessel id in ``full`` is entirely distal (a downstream
        sub-branch). A voxel is in the territory iff its assigned sample is on
        ``vid`` at index ≥ start_idx, or on a vessel in ``full``."""
        start = int(idx)
        # A child is fully distal iff it branches off THIS vessel at or after the
        # chosen point; every vessel below such a child is fully distal too.
        full: set[str] = set()
        queue = [c for c in self.children(vid)
                 if self.vessels[c].junction is not None
                 and self.vessels[c].junction >= start]
        while queue:
            f = queue.pop()
            if f in full:
                continue
            full.add(f)
            queue.extend(self.children(f))     # descendants of a distal vessel
        return vid, start, full

    def territory_mask(self, assign_result: dict, vid: str, idx: int) -> np.ndarray:
        """Boolean array (V,) over the assigned voxels: True where the voxel's
        assigned centreline sample is distal to the chosen (vid, idx)."""
        code = assign_result["code"]
        a_idx = assign_result["idx"]
        code_to_vid = assign_result["code_to_vid"]
        vid_to_code = {v: c for c, v in enumerate(code_to_vid)}
        _, start, full = self.distal_set(vid, idx)
        out = np.zeros(len(code), bool)
        # chosen vessel, distal portion
        if vid in vid_to_code:
            cc = vid_to_code[vid]
            out |= (code == cc) & (a_idx >= start)
        # fully-distal downstream sub-branches
        full_codes = [vid_to_code[v] for v in full if v in vid_to_code]
        if full_codes:
            out |= np.isin(code, full_codes)
        return out


# -------------------------------------------------------- mask <-> world
def voxel_centers_from_mask(mask_zyx, bbox, spacing_xyz):
    """Voxel-centre world-mm coords of every True voxel in a sub-volume mask.

    ``mask_zyx`` is a bool array indexed [z,y,x]; ``bbox`` = (z0,z1,y0,y1,x0,x1)
    is its offset into the full volume; ``spacing_xyz`` = (sx,sy,sz) mm. Returns
    ``(centers (V,3) world-mm, zyx (V,3) int)`` where zyx are FULL-volume indices
    — so a computed territory boolean can be scattered straight back into the
    volume grid. World convention: world = voxel_index · spacing (origin 0)."""
    mask = np.asarray(mask_zyx, bool)
    z0, _z1, y0, _y1, x0, _x1 = bbox
    sx, sy, sz = spacing_xyz
    zl, yl, xl = np.nonzero(mask)
    z = zl + z0
    y = yl + y0
    x = xl + x0
    centers = np.stack([x * sx, y * sy, z * sz], axis=1).astype(float)
    zyx = np.stack([z, y, x], axis=1).astype(int)
    return centers, zyx
