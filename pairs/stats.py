#!/usr/bin/env python3
"""Voxel-level statistics of an input, the same for the pairs' inputs and for real TRELLIS
output (`pairs.calibrate` compares the two). Both go through the same `voxelize_scene`:

    kept_fraction      solid kept by keep-largest-component (1 - floater share)
    fill_gain          filled voxels per surface voxel (how enclosed the mesh is)
    rough3             share of shell voxels with 3+ faces open to air (blobby surfaces)
    lum, chroma        mean L* and C* of shell colours (TRELLIS bakes dark)
    nbr_de             mean colour difference between face-adjacent shell voxels (noise)
"""
from __future__ import annotations

import numpy as np

from voxel.block_palette import srgb_to_lab


def voxel_stats(solid, shell, rgb):
    pad = np.pad(solid, 1)
    open_faces = sum((~pad[tuple(slice(1 + (d if a == ax else 0), pad.shape[a] - 1 + (d if a == ax else 0))
                                 for a in range(3))]).astype(np.int8)
                     for ax in range(3) for d in (-1, 1))
    rough3 = float((open_faces[shell] >= 3).mean()) if shell.any() else 0.0
    cols = rgb[shell].astype(np.float64)
    lab = srgb_to_lab(cols) if len(cols) else np.zeros((0, 3))
    lum = float(lab[:, 0].mean()) if len(lab) else 0.0
    chroma = float(np.hypot(lab[:, 1], lab[:, 2]).mean()) if len(lab) else 0.0
    # neighbour colour difference along +x, +y, +z where both voxels are shell
    full = np.zeros(solid.shape + (3,))
    full[shell] = lab
    des = []
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax], b[ax] = slice(0, -1), slice(1, None)
        both = shell[tuple(a)] & shell[tuple(b)]
        if both.any():
            des.append(np.linalg.norm(full[tuple(a)][both] - full[tuple(b)][both], axis=1))
    nbr = float(np.concatenate(des).mean()) if des else 0.0
    return {"rough3": rough3, "lum": lum, "chroma": chroma, "nbr_de": nbr}
