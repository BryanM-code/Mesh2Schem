"""Checks every pair must pass before it counts as a training example.

`check_pair` runs on the arrays a pair is about to be written with and returns a list of
failure strings (empty = valid). A pair that fails is recorded as rejected with reason
`validation_failed`, never silently kept. The checks encode the dataset's contract:

* labels in range: shape in the vocabulary or IGNORE_SHAPE, material likewise;
* air is air in both maps, ignore is ignore in both maps;
* ignored voxels are interior -- nothing a camera could see is left unsupervised;
* the outer layer (sides and top) is air, so the margin of supervised air exists;
* stairs only on stair-capable materials, slabs only on slab-capable ones, glass only full;
* the input grid has exactly the target's shape (same frame, voxel for voxel);
* input and target overlap (occupancy IoU above a floor), so a pair is not misregistered;
* the written .schem reads back block for block (checked where it is written).
"""
from __future__ import annotations

import numpy as np

from mcvoxel import block_shapes as bs
from mcvoxel import vocab
from mcvoxel.block_palette import AUX_PALETTE

MIN_IOU = 0.2

_STAIRS = np.zeros(len(vocab.MATERIALS), dtype=bool)
_SLABS = np.zeros(len(vocab.MATERIALS), dtype=bool)
_GLASS = np.zeros(len(vocab.MATERIALS), dtype=bool)
for _i, _m in enumerate(vocab.MATERIALS):
    _slab, _stairs = bs.BLOCK_VARIANTS.get(_m, (None, None))
    _STAIRS[_i] = _stairs is not None
    _SLABS[_i] = _slab is not None
    _GLASS[_i] = _m in AUX_PALETTE


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
    solid = ~air & ~ign
    mat = np.where(solid, material, 0).astype(np.int64)
    stairs = solid & (shape >= 4) & (shape <= 7)
    if (stairs & ~_STAIRS[mat]).any():
        f.append(p + "{} stairs on materials without stairs".format(int((stairs & ~_STAIRS[mat]).sum())))
    slabs = solid & ((shape == 2) | (shape == 3))
    if (slabs & ~_SLABS[mat]).any():
        f.append(p + "{} slabs on materials without slabs".format(int((slabs & ~_SLABS[mat]).sum())))
    for sh, family in ((vocab.SHAPE_FENCE, bs.FENCE_VARIANTS), (vocab.SHAPE_WALL, bs.WALL_VARIANTS)):
        compatible = np.array([m in family for m in vocab.MATERIALS])
        bad = solid & (shape == sh) & ~compatible[mat]
        if bad.any():
            f.append(p + "{} {} labels on incompatible materials".format(int(bad.sum()), vocab.SHAPES[sh]))
    glass_partial = solid & _GLASS[mat] & (shape != 1)
    if glass_partial.any():
        f.append(p + "{} glass voxels are not full blocks".format(int(glass_partial.sum())))
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
