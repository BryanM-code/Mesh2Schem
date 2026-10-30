#!/usr/bin/env python3
"""How blocky a mesh is, for checking that the inputs are no blockier than TRELLIS output.

* `lattice16` -- take the most common edge component length as the block size u; the share
  of vertices lying on a 1/16-of-u lattice (Minecraft's own sub-block unit) in all three
  axes. The lattice phase is estimated per axis from the histogram of fractional positions,
  so the tiny inset offsets exporters add against z-fighting do not break it.
* `axis_area` -- share of surface area facing exactly along +-X, +-Y or +-Z.

A mesh is `blocky` when lattice16 >= 0.5: ordinary meshes score at most a few hundredths.
`axis_area` alone cannot tell (real apartment blocks reach 1.0), so it is recorded only.
"""
from __future__ import annotations

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
