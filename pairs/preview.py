"""Picture helpers for the galleries and the test sets: `render_rgb` draws a voxel grid in
its own colours (an input, as the model sees it) with the block renderer
(`minecraft.render`), and `sample_surface` samples coloured points off a mesh's surface.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from minecraft import block_colors
from minecraft import render
from minecraft.schem_voxels import VoxelGrid

SIZE = 360
YAW, PITCH = 40.0, 30.0


@contextmanager
def _colour_names(lut):
    original = block_colors.color_for
    block_colors.color_for = lambda s, _o=original: lut.get(s) or _o(s)
    try:
        yield
    finally:
        block_colors.color_for = original


def _orbit(grid_yzx, names, yaw, pitch, size):
    vg = VoxelGrid(np.asarray(grid_yzx, dtype=np.uint16), list(names))
    return render._render_orbit(vg, yaw, pitch, None, size, 300_000)


def render_rgb(occ_yzx, rgb_yzx, yaw=YAW, pitch=PITCH, size=SIZE):
    """Solid voxels drawn in their own colour (quantised to 16 levels a channel)."""
    occ = np.asarray(occ_yzx, dtype=bool)
    if not occ.any():
        return None
    ys, zs, xs = np.nonzero(occ)
    cols = (np.asarray(rgb_yzx)[ys, zs, xs] // 16 * 16 + 8).clip(0, 255)
    keys, inv = np.unique(cols, axis=0, return_inverse=True)
    names = ["minecraft:air"] + ["rgb:%d,%d,%d" % tuple(k) for k in keys]
    grid = np.zeros(occ.shape, dtype=np.uint16)
    grid[ys, zs, xs] = inv.ravel() + 1
    lut = {n: tuple(int(c) for c in k) for n, k in zip(names[1:], keys)}
    with _colour_names(lut):
        return _orbit(grid, names, yaw, pitch, size)


def sample_surface(verts, faces, point_colors, n=500_000, seed=0):
    """(points, colours 0..255, face normals) sampled uniformly by area off a mesh."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    ok = area > 0
    if not ok.any():
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(ok)
    pick = idx[rng.choice(len(idx), size=n, p=area[idx] / area[idx].sum())]
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    bary = np.stack([1 - s, s * (1 - r2), s * r2], axis=1)
    pts = (tri[pick] * bary[:, :, None]).sum(axis=1)
    cols = point_colors(pick, bary) if point_colors is not None else np.full((n, 3), 190.0)
    normals = cross[pick] / area[pick, None]
    return pts, np.clip(np.nan_to_num(cols, nan=190.0), 0, 255), normals
