"""Shell / exterior / interior masks: what a model is supervised on and what it ignores.

    exterior  air the outside reaches                 -> supervised as air
    shell     blocks touching exterior air            -> supervised (shape + material)
    interior  everything else: enclosed rooms, wall   -> ignored (IGNORE_SHAPE / _MATERIAL)
              cores, buried blocks, the ground

Exterior is decided by the size of the openings, never by what is overhead (D22): being
covered from above does not, by itself, make air interior. The outside reaches through
any opening at least `OPENING` blocks wide and `OPENING_HEIGHT` tall -- the air is eroded
by an `OPENING` x `OPENING_HEIGHT` x `OPENING` box, flooded from the grid's boundary and
grown back by the same box. Porches, recessed porches, verandas, balconies (recessed ones
too), loggias, arcades, colonnades, carports, canopies, eaves and soffits, gable
undersides and the space under a jettied floor are open that wide, so they are exterior
and the walls behind them are shell: they are part of the building and trained on. Air
within `REVEAL` block of that exterior is exterior too, through openings of any size:
window and door reveals, thresholds. A room behind a doorway or a window narrower than
`OPENING` stays interior, glass or no glass. Last, a leftover pocket of air in which at
least `GAP_SIGHT` of the voxels see the outside along a straight horizontal line is an open
gap, not a room -- the strip between a garden wall or a hedge and the house under the
eaves, a narrow passage open at both ends -- and is exterior too. Through a doorway or a
window only a narrow tube of a room has that line of sight.

Ground (from the crop): voxels at or below the ground height of their column count as
solid for the flood, so a house whose floor was the grass plate does not flood from below,
and blocks buried in the ground are not shell. Interior blocks are masked, never deleted.

`large_opening`: more than `LARGE_OPENING` of the deep covered air is exterior -- an open
hall, a garage with its door up, a wide arch into a courtyard building. Kept as exterior,
flagged for review. *Deep* covered air has a block somewhere above it and is more than
`DEEP` blocks (in its own layer) from open-sky air: eaves, soffits and ordinary porches are
shallower than that, and would otherwise put most houses over the line.

`exposure` audits any set of masks the same way, whatever rule made them: a block is
*outside-visible* when it touches exterior air or *open covered air* (covered air with a
straight horizontal line of sight to the outside); visible blocks that are not shell are
visible but ignored. Blocks below the ground height of their column that are not shell
are below-ground ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

SIX = ndimage.generate_binary_structure(3, 1)
OPENING = 3                  # an opening this wide (and OPENING_HEIGHT tall) lets the outside in
OPENING_HEIGHT = 2
REVEAL = 1                   # then this many blocks further, through any opening
GAP_SIGHT = 0.75             # a pocket this much in sight of the outside is an open gap
LARGE_OPENING = 0.30         # deep covered air that ends up exterior, above this share: flagged
DEEP = 3                     # covered air further than this from open sky is "deep"
LARGE_MIN = 50               # ...and at least this much of it is exterior


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


def ground_mask(shape, height) -> np.ndarray:
    """Voxels at or below the ground height of their column. `height`: int (X, Z) or a
    scalar ground level; -1 / None: no ground."""
    X, Y, Z = shape
    if height is None:
        height = -1
    h = np.broadcast_to(np.asarray(height, dtype=np.int64), (X, Z))
    return np.arange(Y)[None, :, None] <= h[:, None, :]


def sightline(air: np.ndarray) -> np.ndarray:
    """Air with a straight horizontal (+-X or +-Z) line of air to the grid's boundary."""
    sight = np.zeros_like(air)
    for axis in (0, 2):
        for flip in (False, True):
            a = np.flip(air, axis) if flip else air
            s = np.logical_and.accumulate(a, axis=axis)
            sight |= np.flip(s, axis) if flip else s
    return sight


def exterior_air(air: np.ndarray) -> np.ndarray:
    """Air the outside reaches: through openings at least OPENING x OPENING_HEIGHT, then
    REVEAL blocks further through any opening, then open gaps (see the module docstring)."""
    box = np.ones((OPENING, OPENING_HEIGHT, OPENING), bool)
    core = ndimage.binary_erosion(air, structure=box, border_value=1)
    reached = _outside(core) & core
    big = ndimage.binary_dilation(reached, structure=box) & air
    if REVEAL:
        big |= ndimage.binary_dilation(big, structure=SIX, iterations=REVEAL) & air
    rest = air & ~big
    if rest.any():
        labels, n = ndimage.label(rest, structure=SIX)
        seen = sightline(air) & rest
        size = np.bincount(labels.ravel(), minlength=n + 1)
        sighted = np.bincount(labels.ravel(), weights=seen.ravel(), minlength=n + 1)
        gap = sighted >= GAP_SIGHT * np.maximum(size, 1)
        gap[0] = False
        big |= gap[labels]
    return big


def exposure(solid, shell, exterior, height) -> tuple[dict, dict]:
    """Rule-independent audit: (stats, masks) -- see the module docstring."""
    ground = ground_mask(solid.shape, height)
    above_air = ~solid & ~ground
    roofed = np.maximum.accumulate(solid[:, ::-1, :], axis=1)[:, ::-1, :]
    covered = above_air & roofed
    open_covered = covered & sightline(above_air)
    visible = solid & _touches(exterior | open_covered)
    visible_ignored = visible & ~shell
    Y = solid.shape[1]
    h = np.broadcast_to(np.asarray(-1 if height is None else height, dtype=np.int64),
                        (solid.shape[0], solid.shape[2]))
    below = solid & ~shell & (np.arange(Y)[None, :, None] < h[:, None, :])
    n_solid, n_visible = int(solid.sum()), int(visible.sum())
    stats = {
        "outside_visible": n_visible,
        "visible_ignored": int(visible_ignored.sum()),
        "visible_ignored_fraction": round(int(visible_ignored.sum()) / n_visible, 4)
        if n_visible else 0.0,
        "below_ground_ignored": int(below.sum()),
        "below_ground_fraction": round(int(below.sum()) / n_solid, 4) if n_solid else 0.0,
        "open_covered_air": int(open_covered.sum()),
    }
    return stats, {"visible_ignored": visible_ignored, "below_ground": below}


def compute_masks(blocks: np.ndarray, height=None) -> Masks:
    """`height`: the crop's ground height per column (int (X, Z), -1 = none), or a scalar
    ground level, or None for no ground."""
    solid = blocks != 0
    ground = ground_mask(blocks.shape, height)
    air = ~(solid | ground)
    exterior = exterior_air(air)
    shell = solid & _touches(exterior)
    interior = ~(exterior | shell)

    roofed = np.maximum.accumulate(solid[:, ::-1, :], axis=1)[:, ::-1, :]
    covered = air & roofed                     # above-ground air with a block above it
    covered_n = int(covered.sum())
    covered_ext = int((covered & exterior).sum())
    open_sky = air & ~roofed
    near_sky = ndimage.binary_dilation(open_sky, structure=np.ones((3, 1, 3), bool),
                                       iterations=DEEP) if open_sky.any() else open_sky
    deep = covered & ~near_sky
    deep_n, deep_ext = int(deep.sum()), int((deep & exterior).sum())
    h = np.asarray(-1 if height is None else height)
    rep = int(np.median(h[h >= 0])) if (h >= 0).any() else -1
    stats = {
        "ground_y": rep,
        "solid": int(solid.sum()),
        "shell": int(shell.sum()),
        "interior": int(interior.sum()),
        "interior_air": int((interior & air).sum()),          # rooms, above the ground
        "below_ground_air": int((~solid & ground).sum()),
        "interior_solid": int((interior & solid).sum()),
        "exterior_air": int(exterior.sum()),
        "covered_air": covered_n,
        "covered_exterior": covered_ext,
        "covered_exterior_share": round(covered_ext / covered_n, 4) if covered_n else None,
        "deep_covered_air": deep_n,
        "deep_covered_exterior": deep_ext,
        "deep_exterior_share": round(deep_ext / deep_n, 4) if deep_n else None,
        "interior_fraction": round(int(interior.sum()) / max(int(interior.sum())
                                                              + int(shell.sum()), 1), 4),
        "interior_solid_fraction": round(int((interior & solid).sum())
                                         / max(int(solid.sum()), 1), 4),
    }
    audit, _ = exposure(solid, shell, exterior, height)
    stats.update(audit)
    flags = []
    if deep_ext >= LARGE_MIN and deep_ext > LARGE_OPENING * deep_n:
        flags.append("large_opening")
    return Masks(shell, exterior, interior, stats, flags)
