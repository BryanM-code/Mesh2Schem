"""Checks every pair must pass before it counts as a training example.

`check_pair` runs on the arrays a pair is about to be written with and returns a list of
failure strings (empty = valid). A pair that fails is recorded as rejected with reason
`validation_failed`, never silently kept. The checks encode the dataset's contract:

* labels in range: shape in the vocabulary or IGNORE_SHAPE, material likewise;
* air is air in both maps, ignore is ignore in both maps;
* ignored voxels are interior -- nothing a camera could see is left unsupervised;
* the outer layer (sides and top) is air, so the margin of supervised air exists;
* every supervised (shape, material) pair is one the vocabulary holds (`vocab.VALID`):
  stairs only on materials with stairs, slabs on materials with slabs, and so on;
* the input grid has exactly the target's shape (same frame, voxel for voxel);
* input and target overlap (occupancy IoU above a floor), so a pair is not misregistered;
* the written .schem reads back block for block (checked where it is written).
"""
from __future__ import annotations

import numpy as np

from mcvoxel import vocab

MIN_IOU = 0.2

_STAIR_LABELS = [i for i, s in enumerate(vocab.SHAPES) if s.startswith("stairs_")]
_SLAB_LABELS = [vocab.SHAPE_SLAB_BOTTOM, vocab.SHAPE_SLAB_TOP]


def check_target(shape, material, label=""):
    """Failures of one target's (shape, material) grids, both (X, Y, Z)."""
    f = []
    p = (label + ": ") if label else ""
    if shape.shape != material.shape:
        return [p + "shape and material grids differ in size"]
    ok_shape = (shape < len(vocab.SHAPES)) | (shape == vocab.IGNORE_SHAPE)
    if not ok_shape.all():
        f.append(p + "{} shape labels out of range".format(int((~ok_shape).sum())))
    ok_mat = (material < len(vocab.MATERIALS)) | (material == vocab.IGNORE_MATERIAL)
    if not ok_mat.all():
        f.append(p + "{} material labels out of range".format(int((~ok_mat).sum())))
        return f
    if ((shape == 0) != (material == 0)).any():
        f.append(p + "air disagrees between shape and material")
    ign = shape == vocab.IGNORE_SHAPE
    if (ign != (material == vocab.IGNORE_MATERIAL)).any():
        f.append(p + "ignore disagrees between shape and material")
    air = shape == 0
    pad = np.pad(air, 1, constant_values=True)
    pad[:, 0, :] = False                                   # below the grid is ground, not air
    touches_air = (pad[2:, 1:-1, 1:-1] | pad[:-2, 1:-1, 1:-1] | pad[1:-1, 2:, 1:-1] |
                   pad[1:-1, :-2, 1:-1] | pad[1:-1, 1:-1, 2:] | pad[1:-1, 1:-1, :-2])
    exposed_ignored = ign & touches_air
    if exposed_ignored.any():
        f.append(p + "{} ignored voxels are visible from outside".format(int(exposed_ignored.sum())))
    outer = np.zeros(shape.shape, dtype=bool)
    outer[0], outer[-1], outer[:, :, 0], outer[:, :, -1], outer[:, -1] = True, True, True, True, True
    if (~air & outer).any():
        f.append(p + "the outer layer is not all air (margin missing)")
    solid = ~air & ~ign & ok_shape
    sh = np.where(solid, shape, 0).astype(np.int64)
    mat = np.where(solid, material, 0).astype(np.int64)
    invalid = solid & ~vocab.VALID[sh, mat]
    stairs = invalid & np.isin(shape, _STAIR_LABELS)
    if stairs.any():
        f.append(p + "{} stairs on materials without stairs".format(int(stairs.sum())))
    slabs = invalid & np.isin(shape, _SLAB_LABELS)
    if slabs.any():
        f.append(p + "{} slabs on materials without slabs".format(int(slabs.sum())))
    other = invalid & ~stairs & ~slabs
    for label in np.unique(shape[other]).tolist():
        f.append(p + "{} {} labels on incompatible materials".format(
            int((other & (shape == label)).sum()), vocab.SHAPES[label]))
    if not solid.any():
        f.append(p + "no supervised solid voxel")
    return f


def check_pair(targets: dict, input_occ, iou: float):
    """All failures of a pair. `targets` maps a label to (shape, material) grids (X, Y, Z)."""
    failures = []
    for label, (shape, material) in targets.items():
        failures += check_target(shape, material, label)
        if input_occ.shape != shape.shape:
            failures.append("{}: input grid {} does not match target grid {}".format(
                label, input_occ.shape, shape.shape))
    if iou < MIN_IOU:
        failures.append("input/target occupancy IoU {:.2f} below {}".format(iou, MIN_IOU))
    return failures
