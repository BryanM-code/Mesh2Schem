"""Shell / exterior / interior masks: what a model is supervised on and what it ignores.

    exterior  air reachable from outside            -> supervised as air
    shell     blocks touching exterior air           -> supervised (shape + material)
    interior  everything else (rooms, wall cores)    -> ignored (IGNORE_SHAPE / _MATERIAL)

Interior blocks are masked, never deleted. Every non-air block counts as solid for the
flood (a pane, a door, a fence all close an opening); the grid arrives padded with air by
the crop, so the outside is one connected region. The ground the crop removed still counts:
everything at or below the ground level is solid for the flood, so a house whose floor was
the grass plate does not flood from below, and below-ground air is ignored.

Open doorways and glassless windows let the first flood (6-connected air from the grid's
boundary) leak into rooms. The second pass is `mcvoxel.grid.occluded_interior`: air
covered from above and blocked on at least four of the other five axis directions (+-X,
+-Z, below) counts as inside -- rooms behind an open door pass, porches and courtyards do
not. Exterior air is then re-flooded from the boundary through what is left, so a room
cut off by a doorway it filled is not exterior either.

`interior_leak_suspect`: exterior air under a roof and more than `DEEP` blocks (in its own
layer) from open sky -- eaves, porches and window recesses are shallow, a leaked room is
deep -- is at least `LEAK_MIN_DEEP` blocks and over `LEAK_SHARE` of itself plus the
interior air. Real for canopies, verandas, pilotis and unfinished floors; for an ordinary
house it means the rooms leaked through a hole the heuristic could not close.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from mcvoxel.grid import occluded_interior

SIX = ndimage.generate_binary_structure(3, 1)
DEEP = 3                     # blocks from open sky before roofed exterior air is "deep"
LEAK_MIN_DEEP = 100          # deep roofed exterior air below this: too little to matter
LEAK_SHARE = 0.2             # ...or below this share of (deep + interior air)


@dataclass
class Masks:
    shell: np.ndarray
    exterior: np.ndarray
    interior: np.ndarray
    stats: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def _outside(air: np.ndarray) -> np.ndarray:
    """Air 6-connected to the grid's boundary."""
    labels, _ = ndimage.label(air, structure=SIX)
    edge = np.concatenate([labels[0].ravel(), labels[-1].ravel(), labels[:, 0].ravel(),
                           labels[:, -1].ravel(), labels[:, :, 0].ravel(),
                           labels[:, :, -1].ravel()])
    ids = np.unique(edge[edge > 0])
    return np.isin(labels, ids)


def _touches(mask: np.ndarray) -> np.ndarray:
    """Voxels with a 6-neighbour in `mask`."""
    return ndimage.binary_dilation(mask, structure=SIX)


def compute_masks(blocks: np.ndarray, ground_y: int | None = None) -> Masks:
    """`ground_y`: the ground level in this grid's coordinates (the crop removed the ground).

    Everything at or below it counts as solid ground for the flood, so a house whose floor
    was the grass plate does not flood from below, and below-ground air is ignored rather
    than supervised. None: the ground is taken to be just below the lowest block.
    """
    solid = blocks != 0
    if ground_y is None:
        layers = np.flatnonzero(solid.any(axis=(0, 2)))
        ground_y = int(layers[0]) - 1 if len(layers) else -1
    ground = np.zeros(solid.shape[1], dtype=bool)
    ground[:max(ground_y + 1, 0)] = True
    blocked = solid | ground[None, :, None]
    air = ~blocked
    first = _outside(air)
    leaked = first & occluded_interior(blocked)
    exterior = _outside(first & ~leaked)
    shell = solid & _touches(exterior)
    interior = ~(exterior | shell)

    roofed = np.maximum.accumulate(solid[:, ::-1, :], axis=1)[:, ::-1, :]
    covered = air & roofed                     # above-ground air with a block above it
    covered_n = int(covered.sum())
    interior_air = int((interior & air).sum())          # rooms, above the ground
    covered_interior = int((covered & interior).sum())
    # Exterior air under a roof and more than DEEP blocks (horizontally, same layer) from
    # open sky: eaves, porches and window recesses are shallow; a leaked room is deep.
    open_sky = exterior & ~roofed
    near_open = ndimage.binary_dilation(open_sky, structure=np.ones((3, 1, 3), bool),
                                        iterations=DEEP) if open_sky.any() else open_sky
    deep = int((exterior & roofed & ~near_open).sum())
    stats = {
        "ground_y": int(ground_y),
        "solid": int(solid.sum()),
        "shell": int(shell.sum()),
        "interior": int(interior.sum()),
        "interior_air": interior_air,
        "below_ground_air": int((~solid & ground[None, :, None]).sum()),
        "interior_solid": int((interior & solid).sum()),
        "exterior_air": int(exterior.sum()),
        "leaked_rooms": int(leaked.sum()),
        "covered_air": covered_n,
        "covered_interior_share": round(covered_interior / covered_n, 4) if covered_n else None,
        "deep_covered_exterior": deep,
        "interior_fraction": round(int(interior.sum()) / max(int(interior.sum())
                                                              + int(shell.sum()), 1), 4),
        "interior_solid_fraction": round(int((interior & solid).sum())
                                         / max(int(solid.sum()), 1), 4),
    }
    flags = []
    if deep >= LEAK_MIN_DEEP and deep > LEAK_SHARE * (deep + interior_air):
        flags.append("interior_leak_suspect")
    return Masks(shell, exterior, interior, stats, flags)
