"""Headless tests for the CT Territory geometry core (core/coronary_territory).

Validates the Phase-0 engine on a synthetic LAD + diagonal-branch tree with a
known topology: role-based roots, first-point snap attachment (incl. a
multi-level sub-branch and the 3 mm too-far guard), nearest-centreline voxel
assignment, and distal-territory extraction across the branch tree. Pure numpy
(scipy optional) - no Qt / VTK - so it runs anywhere:

    python tools/test_coronary_territory.py
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, r"C:\CC_Product\Multi-DicomViewer")

import numpy as np  # noqa: E402

from multi_dicomviewer.core.coronary_territory import (   # noqa: E402
    CoronaryTree, TerritoryEngine, tree_from_specs, voxel_centers_from_mask)


def _line(p0, p1, step_mm=1.0):
    """Straight polyline from p0 to p1 sampled ~every step_mm (proximal→distal)."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    n = max(2, int(round(np.linalg.norm(p1 - p0) / step_mm)) + 1)
    t = np.linspace(0.0, 1.0, n)
    return p0[None, :] + t[:, None] * (p1 - p0)[None, :]


def _build_tree():
    """LAD trunk (x 0→60) with D1@x10, D2@x40, and D1a off D1@y12."""
    tree = CoronaryTree()
    tree.add_root("lad", "LM-LAD", "LM-LAD", _line((0, 0, 0), (60, 0, 0)))
    r1 = tree.add_branch("d1", "D1", _line((10, 0, 0), (10, 25, 0)))
    r2 = tree.add_branch("d2", "D2", _line((40, 0, 0), (40, 25, 0)))
    r3 = tree.add_branch("d1a", "D1a", _line((10, 12, 0), (25, 12, 0)))
    return tree, r1, r2, r3


def test_topology_and_snap():
    tree, r1, r2, r3 = _build_tree()
    assert tree.roots() == ["lad"], tree.roots()
    assert set(tree.children("lad")) == {"d1", "d2"}, tree.children("lad")
    assert tree.children("d1") == ["d1a"], tree.children("d1")
    # branch attachment: parent + junction index on the parent
    assert r1["ok"] and r1["parent"] == "lad" and r1["junction"] == 10, r1
    assert r2["ok"] and r2["parent"] == "lad" and r2["junction"] == 40, r2
    assert r3["ok"] and r3["parent"] == "d1" and r3["junction"] == 12, r3
    assert set(tree.descendants("lad")) == {"d1", "d2", "d1a"}
    print("OK topology + snap (parents/junctions correct, multi-level)")


def test_too_far_guard():
    tree, *_ = _build_tree()
    # Proximal end 5 mm off any vessel (z=5 lifts it clear of the z=0 tree) →
    # rejected (>3 mm) and NOT added.
    res = tree.add_branch("x", "X", _line((10, 10, 5), (10, 30, 5)), snap_tol_mm=3.0)
    assert not res["ok"] and res["reason"] == "too-far", res
    assert "x" not in tree.vessels
    assert abs(res["dist_mm"] - 5.0) < 1e-6, res
    # Nudged within tolerance (z=2) it snaps (to D1, the nearest vessel).
    ok = tree.add_branch("x", "X", _line((10, 10, 2), (10, 30, 2)), snap_tol_mm=3.0)
    assert ok["ok"] and ok["parent"] == "d1", ok
    print("OK 3 mm too-far guard (reject >3 mm, accept within)")


# Probe voxels near known vessels (each maps deterministically to one vessel).
_PROBES = {
    "lad50": (50.0, 0.5, 0.0),    # LAD, distal   (idx ~50)
    "lad20": (20.0, 0.5, 0.0),    # LAD, proximal (idx ~20)
    "d1":    (10.0, 15.0, 0.0),   # on D1
    "d2":    (40.0, 15.0, 0.0),   # on D2
    "d1a":   (18.0, 12.0, 0.0),   # on D1a
}


def _assign_probes(tree):
    names = list(_PROBES)
    pts = np.array([_PROBES[k] for k in names], float)
    a = tree.assign(pts, max_dist_mm=5.0)
    c2v = a["code_to_vid"]
    got = {names[i]: (c2v[a["code"][i]] if a["code"][i] >= 0 else None)
           for i in range(len(names))}
    return a, names, got


def test_assignment():
    tree, *_ = _build_tree()
    _a, _names, got = _assign_probes(tree)
    assert got["lad50"] == "lad" and got["lad20"] == "lad", got
    assert got["d1"] == "d1" and got["d2"] == "d2" and got["d1a"] == "d1a", got
    print("OK nearest-centreline assignment (each probe -> correct vessel)")


def _territory(tree, a, names, vid, idx):
    mask = tree.territory_mask(a, vid, idx)
    return {names[i]: bool(mask[i]) for i in range(len(names))}


def test_distal_territory():
    tree, *_ = _build_tree()
    a, names, _ = _assign_probes(tree)

    # P on LAD at x=30: distal = LAD[>=30] + D2 (junction 40≥30). D1/D1a excluded.
    t = _territory(tree, a, names, "lad", 30)
    assert t == {"lad50": True, "lad20": False, "d1": False,
                 "d2": True, "d1a": False}, t

    # P on LAD at x=5: D1 (10≥5) and thus D1a distal; D2 distal; LAD[>=5].
    t = _territory(tree, a, names, "lad", 5)
    assert t == {"lad50": True, "lad20": True, "d1": True,
                 "d2": True, "d1a": True}, t

    # P at the very start of D1 (idx 0): D1 distal + D1a (its sub-branch); the
    # parent LAD and the sibling D2 are NOT in D1's distal territory.
    t = _territory(tree, a, names, "d1", 0)
    assert t == {"lad50": False, "lad20": False, "d1": True,
                 "d2": False, "d1a": True}, t
    print("OK distal territory (tree walk: partial vessel + downstream branches)")


def test_voxel_centers_from_mask():
    # 2×2×2 sub-volume at offset (z0,y0,x0)=(1,2,3), spacing (0.5,0.5,1.0) mm.
    mask = np.zeros((2, 2, 2), bool)
    mask[0, 0, 0] = True          # full-vol index z=1,y=2,x=3
    mask[1, 1, 1] = True          # full-vol index z=2,y=3,x=4
    centers, zyx = voxel_centers_from_mask(
        mask, bbox=(1, 3, 2, 4, 3, 5), spacing_xyz=(0.5, 0.5, 1.0))
    order = np.lexsort((zyx[:, 0], zyx[:, 1], zyx[:, 2]))
    zyx, centers = zyx[order], centers[order]
    assert zyx.tolist() == [[1, 2, 3], [2, 3, 4]], zyx.tolist()
    # world = voxel_index · spacing (x·sx, y·sy, z·sz); spacing = (0.5,0.5,1.0)
    assert np.allclose(centers[0], [3 * 0.5, 2 * 0.5, 1 * 1.0]), centers[0]
    assert np.allclose(centers[1], [4 * 0.5, 3 * 0.5, 2 * 1.0]), centers[1]
    print("OK voxel_centers_from_mask (world = index*spacing, full-vol zyx)")


def _myo_lvf():
    """A synthetic LVFunction-like myocardium: 10 compact-layer voxels (Epi &
    ~Endo) at controlled world positions near the LAD/D1/D2 tree. spacing
    (1,1,1) mm, origin 0 → world (x,y,z) = local (x,y,z)."""
    epi = np.zeros((1, 30, 70), bool)
    # LAD voxels at y=1 (1 mm off the y=0 trunk), avoiding the branch x's 10/40.
    for x in (15, 20, 25, 30, 35, 45, 50, 55):
        epi[0, 1, x] = True
    epi[0, 15, 10] = True      # on D1
    epi[0, 15, 40] = True      # on D2
    endo = np.zeros_like(epi)
    return SimpleNamespace(epi=epi, endo=endo,
                           spacing_zyx=(1.0, 1.0, 1.0),
                           origin=np.array([0.0, 0.0, 0.0]))


def _engine():
    specs = [
        {"vid": "lad", "name": "LM-LAD", "role": "LM-LAD",
         "points": _line((0, 0, 0), (60, 0, 0))},
        {"vid": "d1", "name": "D1", "role": "branch",
         "points": _line((10, 0, 0), (10, 25, 0))},
        {"vid": "d2", "name": "D2", "role": "branch",
         "points": _line((40, 0, 0), (40, 25, 0))},
    ]
    tree, results = tree_from_specs(specs)
    assert results[0] is None and results[1]["ok"] and results[2]["ok"], results
    return TerritoryEngine(tree, _myo_lvf())


def test_engine_volume():
    eng = _engine()
    vml = 1.0 / 1000.0                       # 1 mm³ voxel
    assert abs(eng.myocardium_ml - 10 * vml) < 1e-12, eng.myocardium_ml
    # LAD @ idx30: LAD voxels x∈{30,35,45,50,55}=5 + D2 voxel(1) = 6.
    m, vol = eng.territory("lad", 30)
    assert int(m.sum()) == 6, int(m.sum())
    assert abs(vol - 6 * vml) < 1e-12, vol
    # LAD @ idx5: everything distal (both branches + all LAD voxels) = 10.
    _m2, vol2 = eng.territory("lad", 5)
    assert abs(vol2 - 10 * vml) < 1e-12, vol2
    # D1 @ idx0: only the D1 voxel.
    _m3, vol3 = eng.territory("d1", 0)
    assert abs(vol3 - 1 * vml) < 1e-12, vol3
    print("OK TerritoryEngine volumes (myocardium + distal territory mL)")


def test_engine_grids():
    eng = _engine()
    grid = eng.territory_grid("lad", 30)
    assert grid.shape == (1, 30, 70) and int(grid.sum()) == 6, int(grid.sum())
    assert grid[0, 1, 55] and not grid[0, 1, 15]        # distal in, proximal out
    assert grid[0, 15, 40] and not grid[0, 15, 10]      # D2 in, D1 out
    lab, code_to_vid = eng.assigned_grid()
    assert int((lab > 0).sum()) == 10, int((lab > 0).sum())   # all myo assigned
    # the LAD voxel at x=55 is labelled with LAD's code (+1)
    assert code_to_vid[lab[0, 1, 55] - 1] == "lad", lab[0, 1, 55]
    assert code_to_vid[lab[0, 15, 40] - 1] == "d2"
    print("OK TerritoryEngine grids (territory scatter + int-label assignment)")


def test_json_roundtrip():
    import json
    tree, *_ = _build_tree()
    # attach a control-point set to one vessel to exercise the ctrl round-trip
    tree.vessels["d1"].ctrl = np.array([[10, 0, 0], [10, 25, 0]], float)
    data = tree.to_json(series={"series_uid": "1.2.3"}, snap_tol_mm=3.0)
    # must survive a real JSON encode/decode (no numpy types leaking through)
    back = CoronaryTree.from_json(json.loads(json.dumps(data)))
    assert data["format"] == "MDV-CoroTree" and data["series"]["series_uid"] == "1.2.3"
    assert list(back.vessels) == list(tree.vessels), list(back.vessels)
    for vid, v in tree.vessels.items():
        b = back.vessels[vid]
        assert b.role == v.role and b.parent == v.parent, vid
        assert b.junction == v.junction, (vid, b.junction, v.junction)
        assert np.allclose(b.points, v.points), vid
    assert np.allclose(back.vessels["d1"].ctrl, tree.vessels["d1"].ctrl)
    # territory is identical after the round-trip
    a0, names, _ = _assign_probes(tree)
    a1, _, _ = _assign_probes(back)
    t0 = {names[i]: bool(tree.territory_mask(a0, "lad", 30)[i]) for i in range(len(names))}
    t1 = {names[i]: bool(back.territory_mask(a1, "lad", 30)[i]) for i in range(len(names))}
    assert t0 == t1, (t0, t1)
    print("OK .corotree.json round-trip (topology/points/junctions/territory)")


def test_batch_connect():
    """Batch workflow: add vessels unconnected (any draw direction), then
    connect_all grows the tree by nearest endpoint, reversing as needed."""
    tree = CoronaryTree()
    tree.add_vessel("lad", "LM-LAD", "LM-LAD", _line((0, 0, 0), (60, 0, 0)))
    # D9 drawn REVERSED (distal→proximal): its LAST point sits on the LAD.
    tree.add_vessel("d9", "D9", "branch", _line((45, 22, 0), (30, 0, 0)))
    # D9a off D9 (normal direction), first point on D9.
    tree.add_vessel("d9a", "D9a", "branch", _line((37, 11, 0), (50, 16, 0)))
    # A stray branch far from everything → must stay unconnected.
    tree.add_vessel("x", "X", "branch", _line((200, 200, 0), (210, 210, 0)))

    assert tree.roots() == ["lad"], tree.roots()
    assert set(tree.unconnected()) == {"d9", "d9a", "x"}, tree.unconnected()

    res = tree.connect_all(snap_tol_mm=3.0)
    assert set(res["connected"]) == {"d9", "d9a"}, res
    assert res["unconnected"] == ["x"], res
    # D9 was reversed so its proximal end (points[0]) is now the junction on LAD.
    assert tree.vessels["d9"].parent == "lad", tree.vessels["d9"].parent
    assert np.allclose(tree.vessels["d9"].points[0], [30, 0, 0]), \
        tree.vessels["d9"].points[0]
    assert tree.vessels["d9"].junction == 30, tree.vessels["d9"].junction
    # D9a attached to D9 (multi-level, resolved after D9 connected).
    assert tree.vessels["d9a"].parent == "d9", tree.vessels["d9a"].parent
    assert tree.children("lad") == ["d9"] and tree.children("d9") == ["d9a"]

    # Incremental: add another branch near D9 and reconnect — prior links kept.
    tree.add_vessel("d9b", "D9b", "branch", _line((45, 14, 0), (55, 18, 0)))
    res2 = tree.connect_all(snap_tol_mm=3.0)
    assert "d9b" in res2["connected"], res2
    assert tree.vessels["d9"].parent == "lad"          # unchanged
    assert tree.vessels["d9b"].parent in ("d9a", "d9"), tree.vessels["d9b"].parent
    print("OK batch connect_all (nearest-endpoint, auto-reverse, multi-level, "
          "incremental, far branch left loose)")


def main():
    test_batch_connect()
    test_json_roundtrip()
    test_topology_and_snap()
    test_too_far_guard()
    test_assignment()
    test_distal_territory()
    test_voxel_centers_from_mask()
    test_engine_volume()
    test_engine_grids()
    print("\nAll CT Territory core tests passed.")


if __name__ == "__main__":
    main()
