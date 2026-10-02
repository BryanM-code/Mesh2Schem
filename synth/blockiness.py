#!/usr/bin/env python3
"""Detect sources that are already blocky: Minecraft exports, voxel art, box-massing models.

Usage:  python -m synth.blockiness [--root data/synthetic/pilot-v1]

The dataset must teach real architecture -> blocks; a source that is ALREADY made of blocks
defeats the purpose, and text filters only catch the ones whose caption admits it. In the
pilot a Minecraft-built "Cathedral" passed every text rule. Geometry gives it away:

* `lattice16` -- take the most common edge component length as the block size u; the share
  of vertices lying on a 1/16-of-u lattice (Minecraft's own sub-block unit) in all three
  axes. The lattice phase is estimated per axis from the histogram of fractional positions,
  so the tiny inset offsets exporters add against z-fighting do not break it (anchoring on
  the minimum vertex did).
* `axis_area` -- share of surface area facing exactly along +-X, +-Y or +-Z.

Measured on the 200 pilot sources: ordinary models score lattice16 <= 0.023 at the 97th
percentile; the four flagged at >= 0.5 were a Minecraft cathedral, a voxel-art ruin, a
voxelised curved tower and a box-massing model. `axis_area` alone is useless (real
apartment blocks reach 1.0), so it is recorded but not used to decide.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BLOCKY_LATTICE = 0.5


def _lattice_share(v, step, tol=0.06):
    ok = np.ones(len(v), dtype=bool)
    for ax in range(3):
        frac = np.mod(v[:, ax] / step, 1.0)
        hist, edges = np.histogram(frac, bins=50, range=(0.0, 1.0))
        i = int(np.argmax(hist))
        phase = (edges[i] + edges[i + 1]) / 2.0
        ok &= np.abs(np.mod(frac - phase + 0.5, 1.0) - 0.5) < tol
    return float(ok.mean())


def blockiness(verts, faces):
    """dict(axis_area, lattice16, blocks_across, blocky) for a mesh in any units."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a = np.linalg.norm(n, axis=1)
    ok = a > 0
    out = {"axis_area": 0.0, "lattice16": 0.0, "blocks_across": 0.0, "blocky": False}
    if not ok.any():
        return out
    out["axis_area"] = round(float(a[ok][np.abs(n[ok] / a[ok, None]).max(axis=1) > 0.9995].sum() / a[ok].sum()), 4)
    scale = float(np.ptp(v, axis=0).max())
    e = np.abs(v[f[:, [1, 2, 0]]] - v[f]).reshape(-1)
    e = e[e > 1e-3 * scale]
    if len(e) == 0:
        return out
    vals, cnt = np.unique(np.round(e / scale, 4), return_counts=True)
    u = float(vals[np.argmax(cnt)] * scale)
    out["lattice16"] = round(_lattice_share(v, u / 16.0), 4)
    out["blocks_across"] = round(scale / u, 1)
    out["blocky"] = out["lattice16"] >= BLOCKY_LATTICE
    return out


def main(argv=None):
    from mcvoxel.scene import load_scene

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data/synthetic/pilot-v1")
    args = ap.parse_args(argv)
    root = Path(args.root)
    sel = {r["uid"]: r for r in csv.DictReader(open(root / "selection.csv", encoding="utf-8"))}
    rows = []
    for glb in sorted((root / "glb").glob("*.glb")):
        try:
            sm = load_scene(glb)
        except Exception:
            continue
        r = {"uid": glb.stem, "name": sel.get(glb.stem, {}).get("name"), "bucket": sel.get(glb.stem, {}).get("pilot_bucket")}
        r.update(blockiness(sm.verts, sm.faces))
        rows.append(r)
    rows.sort(key=lambda r: -r["lattice16"])
    (root / "blockiness.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    lat = np.array([r["lattice16"] for r in rows])
    print("sources {}  flagged blocky {}  lattice16 p50 {:.3f} p97 {:.3f}".format(
        len(rows), sum(r["blocky"] for r in rows), np.percentile(lat, 50), np.percentile(lat, 97)))
    for r in rows[:8]:
        print("  {:5s} lattice16 {:.3f} axis {:.3f} blocks {:6.1f}  {}".format(
            "FLAG" if r["blocky"] else "", r["lattice16"], r["axis_area"], r["blocks_across"], r["bucket"]))


if __name__ == "__main__":
    main()
