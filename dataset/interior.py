"""Shell / exterior / interior masks: what a model is supervised on and what it ignores.

    exterior  air the outside reaches                 -> supervised as air
    shell     blocks touching exterior air            -> supervised (shape + material)
    interior  everything else: enclosed rooms, wall   -> ignored (IGNORE_SHAPE / _MATERIAL)
              cores, buried blocks, the ground

Exterior is decided by the size of the openings, never by what is overhead: being
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

Depth under cover: covered exterior air is supervised only near the outside.
After the steps above, each exterior air voxel's distance is measured -- 6-connected steps
through exterior air only -- from *open* exterior air (`open_air`). Exterior air more than
`COVER_DEPTH` steps away becomes interior, and the shell is taken afterwards, so blocks
that touch only that deep air are ignored. Porches, verandas, loggias, eaves and jetties
are shallower than that and stay exterior; the inside of an open hall, a hollow facade
shell or an open ground floor beyond `COVER_DEPTH` does not. Being covered from above
still never, on its own, makes air interior -- only depth under cover does.
`COVER_DEPTH` = 6 is the smallest depth that keeps every porch, veranda and loggia of a
set of porch builds checked by eye.

Open air is uncovered air (no block anywhere above it in its column; air on the grid's
boundary counts too) and covered air that is open to the side and to
the sky: it has a straight horizontal line to the grid's boundary (`sightline`), and it
either sees the sky along a steep path (`sky_path`: rising at least `SKY_SLOPE` blocks per
block across) or lies above such air in its column with that line too (the soffit band
under the eave). That is the air under a deep, tall eave or overhang: the walls behind a
9-block eave 45 blocks up (Hotel Piran) are facade, not the inside of a hall, though the
eave covers them. The inside of a hall is walled in at its own height, so no sideways line
reaches it, or it sees the sky only near a tall opening (within about half the opening's
height), so the cap still holds there.

Ground (from the crop): voxels at or below the ground height of their column count as
solid for the flood, so a house whose floor was the grass plate does not flood from below,
and blocks buried in the ground are not shell. Interior blocks are masked, never deleted.

`large_opening`: more than `LARGE_OPENING` of the deep covered air is exterior -- an open
hall, a garage with its door up, a wide arch into a courtyard building. Kept as exterior,
flagged for review. *Deep* covered air has a block somewhere above it and is more than
`DEEP` blocks (in its own layer) from open-sky air: eaves, soffits and ordinary porches are
shallower than that, and would otherwise put most houses over the line. Measured after the
depth cap. `deep_cover_capped`: the cap turned more than `CAPPED_FLAG` air voxels interior
(`deep_cover_capped` in the stats is the count).

`exposure` audits any set of masks the same way, whatever rule made them: a block is
*outside-visible* when it touches exterior air or *open covered air* -- covered air with a
straight horizontal line of sight to the outside, within `COVER_DEPTH` steps (through
above-ground air) of open air, the same depth the rule uses; visible blocks that are
not shell are visible but ignored. Blocks below the ground height of their column that are not shell
are below-ground ignored. Because that audit uses the rule's own depth, it cannot see a
facade the depth rule itself hides, so `compute_masks` also records it with no depth limit
(`visible_ignored_any_depth`): every block a straight horizontal line from outside reaches.
That figure also counts rooms seen through doorways and halls past the cap, so it is a
review signal (`deep_visible_ignored` above `DEEP_VISIBLE_FLAG`), not a rule.
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
COVER_DEPTH = 6              # covered exterior air further than this from open air: interior
SKY_SLOPE = 2                # covered air seeing the sky along a path this steep is open
CAPPED_FLAG = 200            # the cap turned more air than this interior: flagged
DEEP_VISIBLE_FLAG = 0.15     # visible-but-ignored with no depth limit, above this: flagged


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


def _boundary(shape) -> np.ndarray:
    edge = np.zeros(shape, bool)
    edge[0], edge[-1], edge[:, 0], edge[:, -1], edge[:, :, 0], edge[:, :, -1] = (True,) * 6
    return edge


def _shift(layer: np.ndarray, dx: int, dz: int) -> np.ndarray:
    """out[x, z] = layer[x + dx, z + dz]; outside the grid counts as open (True)."""
    out = np.ones_like(layer)
    X, Z = layer.shape
    out[max(-dx, 0):X - max(dx, 0), max(-dz, 0):Z - max(dz, 0)] = \
        layer[max(dx, 0):X - max(-dx, 0), max(dz, 0):Z - max(-dz, 0)]
    return out


def sky_path(air: np.ndarray, open_sky: np.ndarray, slope: int = SKY_SLOPE) -> np.ndarray:
    """Voxels of `air` with a path through `air` to `open_sky` (or out of the grid) that
    climbs at least `slope` blocks per block across, in one horizontal direction: moves of
    one block up, or of `slope - 1` up, one across and one up. The moves are 6-connected, so
    a path never squeezes between two blocks that meet at an edge."""
    X, Y, Z = air.shape
    lit = np.zeros_like(air)
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        path = np.zeros_like(air)
        for y in range(Y - 1, -1, -1):
            up = path[:, y + 1, :] if y + 1 < Y else np.ones((X, Z), bool)
            step = np.ones((X, Z), bool)
            for k in range(1, min(slope, Y - y)):
                step &= air[:, y + k, :]
            if y + slope < Y:
                step &= _shift(air[:, y + slope - 1, :], dx, dz) \
                    & _shift(path[:, y + slope, :], dx, dz)
            path[:, y, :] = air[:, y, :] & (open_sky[:, y, :] | up | step)
        lit |= path
    return lit


def open_air(air: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Air that depth under cover is measured from (see the module docstring): uncovered air
    and air on the grid's boundary, plus covered air with a straight horizontal line to the
    boundary that sees the sky along a steep path, or lies above such air in its column."""
    roofed = np.maximum.accumulate(solid[:, ::-1, :], axis=1)[:, ::-1, :]
    seeds = air & (~roofed | _boundary(air.shape))
    side = air & sightline(air)
    if not side.any():
        return seeds
    lit = sky_path(air, air & ~roofed) & side
    for y in range(1, air.shape[1]):                 # up the column, while the line holds
        lit[:, y, :] |= lit[:, y - 1, :] & side[:, y, :]
    return seeds | lit


def near_open(air: np.ndarray, solid: np.ndarray, depth: int) -> np.ndarray:
    """Voxels of `air` within `depth` 6-connected steps, through `air` only, of open air
    (`open_air`)."""
    seeds = open_air(air, solid)
    if not seeds.any() or depth == 0:
        return seeds
    # depth < 0: no limit (every voxel of `air` connected to open air)
    return ndimage.binary_dilation(seeds, structure=SIX, iterations=max(depth, 0), mask=air)


def exposure(solid, shell, exterior, height, cover_depth=None) -> tuple[dict, dict]:
    """Rule-independent audit: (stats, masks) -- see the module docstring."""
    depth = COVER_DEPTH if cover_depth is None else cover_depth
    ground = ground_mask(solid.shape, height)
    above_air = ~solid & ~ground
    roofed = np.maximum.accumulate(solid[:, ::-1, :], axis=1)[:, ::-1, :]
    covered = above_air & roofed
    open_covered = covered & sightline(above_air) & near_open(above_air, solid, depth)
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
    reached = exterior_air(air)
    exterior = near_open(reached, solid, COVER_DEPTH)            # the depth cap
    capped = int((reached & ~exterior).sum())
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
        "cover_depth": COVER_DEPTH,
        "deep_cover_capped": capped,
        "interior_fraction": round(int(interior.sum()) / max(int(interior.sum())
                                                              + int(shell.sum()), 1), 4),
        "interior_solid_fraction": round(int((interior & solid).sum())
                                         / max(int(solid.sum()), 1), 4),
    }
    audit, _ = exposure(solid, shell, exterior, height)
    stats.update(audit)
    any_depth, _ = exposure(solid, shell, exterior, height, cover_depth=-1)
    stats["visible_ignored_any_depth"] = any_depth["visible_ignored"]
    stats["visible_ignored_any_depth_fraction"] = any_depth["visible_ignored_fraction"]
    flags = []
    if deep_ext >= LARGE_MIN and deep_ext > LARGE_OPENING * deep_n:
        flags.append("large_opening")
    if capped > CAPPED_FLAG:
        flags.append("deep_cover_capped")
    if any_depth["visible_ignored_fraction"] > DEEP_VISIBLE_FLAG:
        flags.append("deep_visible_ignored")
    return Masks(shell, exterior, interior, stats, flags)
