"""Sub-voxel shape classification: full cube vs slab vs stairs ("tetris fit").

Instead of every occupied voxel becoming a full cube, look at the mesh geometry inside
it (via a 4x-finer occupancy grid = 8 octants per coarse voxel) and pick the closest
Minecraft shape: full cube, bottom/top slab, or stairs (4 horizontal facings, the low
step always at the bottom).  The chosen shape rides on the block id as a state string
(`minecraft:oak_stairs[facing=east,half=bottom]`), which schematic.py already exports
verbatim into Sponge v2.

Binary octant occupancy (each octant solid at >= OCTANT_MIN sub-cells) is matched to the
7 shape templates, but any SURFACE (shell) voxel that fits no slab/stair template well is
still made a SLAB rather than a full cube.  That combination keeps clean stair detection
on real slopes (the exterior flood inflates surface fills, so a plain fill-fraction test
reads a 45 deg ramp as near-full — the binary 6/8 threshold does not) while ensuring a
shallow roof reads as a continuous run of slabs/stairs instead of full blocks with a
scattered partial here and there.  Only mask 0xFF and interior voxels stay full.

Octant convention (shared with the viewer geometry):
    octant index = ox + 2*oz + 4*oy,  o* in {0,1} = upper half along that axis, oy=1 =
    TOP half.  Block axes: +X = east, +Z = south (matches schematic.py grid[y, z, x]),
    Y = height.
"""

import numpy as np

# ── Shape codes (index into _SHAPE_BOXES) ─────────────────────────────────────
(SHAPE_FULL, SHAPE_STAIR_E, SHAPE_STAIR_W, SHAPE_STAIR_N, SHAPE_STAIR_S,
 SHAPE_SLAB_BOTTOM, SHAPE_SLAB_TOP) = range(7)


# ── Tunables ──────────────────────────────────────────────────────────────────
OCTANT_MIN = {4: 6, 2: 1}     # min solid sub-cells (of (scale/2)^3) to call an octant solid


def _decide_shape(mask):
    """Shape for one 8-bit octant mask (octant bit = ox + 2*oz + 4*oy, oy=1 = top).

    Slabs and stairs are HORIZONTAL cuts (material bottom-heavy or top-heavy); Minecraft
    has no vertical half-block, so a VERTICAL cut (a wall/beam: equal material in the top
    and bottom halves) must stay a full cube.  So the decision keys on the bottom vs top
    octant balance, not raw Hamming distance to a template (which can't tell a vertical-
    half wall from a stair).
    """
    b = [(mask >> i) & 1 for i in range(8)]
    bot = b[0] + b[1] + b[2] + b[3]          # bottom octants (bits 0-3)
    top = b[4] + b[5] + b[6] + b[7]          # top octants (bits 4-7)
    if bot == 0 and top == 0:
        return SHAPE_SLAB_BOTTOM             # empty/thin surface voxel -> half block
    if bot >= 3 and top >= 3:
        return SHAPE_FULL                    # both halves mostly full -> cube
    if bot >= 3 and 1 <= top <= 2:
        # Full bottom + a partial top cleanly on ONE horizontal side -> a stair.
        top_e, top_w = b[5] + b[7], b[4] + b[6]
        top_n, top_s = b[4] + b[5], b[6] + b[7]
        dx, dz = top_e - top_w, top_s - top_n
        if abs(dx) >= abs(dz) and abs(dx) >= 1 and min(top_e, top_w) == 0:
            return SHAPE_STAIR_E if dx > 0 else SHAPE_STAIR_W
        if abs(dz) >= 1 and min(top_n, top_s) == 0:
            return SHAPE_STAIR_S if dz > 0 else SHAPE_STAIR_N
        # not one-sided -> falls through to the both-halves test below
    if bot >= 2 and top >= 2:
        return SHAPE_FULL                    # substantial material in BOTH halves ->
                                             # vertical/thick (wall/beam, even leaning) -> cube
    if bot > top:
        return SHAPE_SLAB_BOTTOM             # bottom-heavy horizontal cut (roof)
    if top > bot:
        return SHAPE_SLAB_TOP                # top-heavy (overhang underside)
    return SHAPE_FULL                        # bot == top (both 1): vertical sliver -> cube


# Precomputed mask -> shape code (256 entries) so classify is a single table lookup.
_MASK_TO_SHAPE = np.array([_decide_shape(m) for m in range(256)], dtype=np.int8)
                             # variant-capable block, so a slope voxel is never left a full cube


# ── Minecraft blocks that HAVE slab/stairs variants (MC 1.20.1, DataVersion 3465) ──
# base palette id -> (slab_id, stairs_id).  stairs_id None = slab exists but no stairs.
BLOCK_VARIANTS = {
    "minecraft:stone":             ("minecraft:stone_slab", "minecraft:stone_stairs"),
    "minecraft:cobblestone":       ("minecraft:cobblestone_slab", "minecraft:cobblestone_stairs"),
    "minecraft:stone_bricks":      ("minecraft:stone_brick_slab", "minecraft:stone_brick_stairs"),
    "minecraft:andesite":          ("minecraft:andesite_slab", "minecraft:andesite_stairs"),
    "minecraft:polished_andesite": ("minecraft:polished_andesite_slab",
                                     "minecraft:polished_andesite_stairs"),
    "minecraft:smooth_stone":      ("minecraft:smooth_stone_slab", None),
    "minecraft:granite":           ("minecraft:granite_slab", "minecraft:granite_stairs"),
    "minecraft:polished_granite":  ("minecraft:polished_granite_slab",
                                    "minecraft:polished_granite_stairs"),
    "minecraft:diorite":           ("minecraft:diorite_slab", "minecraft:diorite_stairs"),
    "minecraft:polished_diorite":  ("minecraft:polished_diorite_slab",
                                    "minecraft:polished_diorite_stairs"),
    "minecraft:deepslate":         ("minecraft:cobbled_deepslate_slab",
                                    "minecraft:cobbled_deepslate_stairs"),
    "minecraft:polished_deepslate": ("minecraft:polished_deepslate_slab",
                                     "minecraft:polished_deepslate_stairs"),
    "minecraft:blackstone":        ("minecraft:blackstone_slab", "minecraft:blackstone_stairs"),
    "minecraft:polished_blackstone": ("minecraft:polished_blackstone_slab",
                                      "minecraft:polished_blackstone_stairs"),
    "minecraft:sandstone":         ("minecraft:sandstone_slab", "minecraft:sandstone_stairs"),
    "minecraft:smooth_sandstone":  ("minecraft:smooth_sandstone_slab",
                                    "minecraft:smooth_sandstone_stairs"),
    "minecraft:red_sandstone":     ("minecraft:red_sandstone_slab",
                                    "minecraft:red_sandstone_stairs"),
    "minecraft:mud_bricks":        ("minecraft:mud_brick_slab", "minecraft:mud_brick_stairs"),
    "minecraft:bricks":            ("minecraft:brick_slab", "minecraft:brick_stairs"),
    "minecraft:nether_bricks":     ("minecraft:nether_brick_slab", "minecraft:nether_brick_stairs"),
    "minecraft:birch_planks":      ("minecraft:birch_slab", "minecraft:birch_stairs"),
    "minecraft:oak_planks":        ("minecraft:oak_slab", "minecraft:oak_stairs"),
    "minecraft:jungle_planks":     ("minecraft:jungle_slab", "minecraft:jungle_stairs"),
    "minecraft:acacia_planks":     ("minecraft:acacia_slab", "minecraft:acacia_stairs"),
    "minecraft:spruce_planks":     ("minecraft:spruce_slab", "minecraft:spruce_stairs"),
    "minecraft:dark_oak_planks":   ("minecraft:dark_oak_slab", "minecraft:dark_oak_stairs"),
    "minecraft:mangrove_planks":   ("minecraft:mangrove_slab", "minecraft:mangrove_stairs"),
    "minecraft:cherry_planks":     ("minecraft:cherry_slab", "minecraft:cherry_stairs"),
    "minecraft:crimson_planks":    ("minecraft:crimson_slab", "minecraft:crimson_stairs"),
    "minecraft:warped_planks":     ("minecraft:warped_slab", "minecraft:warped_stairs"),
    "minecraft:copper_block":      ("minecraft:cut_copper_slab", "minecraft:cut_copper_stairs"),
    "minecraft:exposed_copper":    ("minecraft:exposed_cut_copper_slab",
                                    "minecraft:exposed_cut_copper_stairs"),
    "minecraft:weathered_copper":  ("minecraft:weathered_cut_copper_slab",
                                    "minecraft:weathered_cut_copper_stairs"),
    "minecraft:oxidized_copper":   ("minecraft:oxidized_cut_copper_slab",
                                    "minecraft:oxidized_cut_copper_stairs"),
    "minecraft:prismarine":        ("minecraft:prismarine_slab", "minecraft:prismarine_stairs"),
    "minecraft:dark_prismarine":   ("minecraft:dark_prismarine_slab",
                                    "minecraft:dark_prismarine_stairs"),
    "minecraft:purpur_block":      ("minecraft:purpur_slab", "minecraft:purpur_stairs"),
    "minecraft:quartz_block":      ("minecraft:quartz_slab", "minecraft:quartz_stairs"),
}

# variant id (slab OR stairs) -> base palette id, for tile lookup / diagnostic colors.
_VARIANT_TO_BASE = {}
for _base, (_slab, _stairs) in BLOCK_VARIANTS.items():
    if _slab:
        _VARIANT_TO_BASE[_slab] = _base
    if _stairs:
        _VARIANT_TO_BASE[_stairs] = _base

# 1.20.1 structural families. Keep these separate from slab/stair capability.
FENCE_VARIANTS = {b: b.replace("_planks", "_fence") for b in BLOCK_VARIANTS if b.endswith("_planks")}
WALL_VARIANTS = {"minecraft:" + base: "minecraft:" + wall for base, wall in (
    ("cobblestone", "cobblestone_wall"), ("stone_bricks", "stone_brick_wall"),
    ("bricks", "brick_wall"), ("andesite", "andesite_wall"), ("diorite", "diorite_wall"),
    ("granite", "granite_wall"), ("sandstone", "sandstone_wall"),
    ("red_sandstone", "red_sandstone_wall"), ("mud_bricks", "mud_brick_wall"),
    ("deepslate", "cobbled_deepslate_wall"), ("polished_deepslate", "polished_deepslate_wall"),
    ("blackstone", "blackstone_wall"), ("polished_blackstone", "polished_blackstone_wall"),
    ("nether_bricks", "nether_brick_wall"), ("prismarine", "prismarine_wall"))}
for _family in (FENCE_VARIANTS, WALL_VARIANTS):
    _VARIANT_TO_BASE.update({v: k for k, v in _family.items()})

_SLAB_CAPABLE = [b for b, (s, _) in BLOCK_VARIANTS.items() if s]
_STAIRS_CAPABLE = [b for b, (_, s) in BLOCK_VARIANTS.items() if s]


def octant_counts(fine_solid, scale):
    """(X,Y,Z,8) uint16 solid sub-cell counts per octant of each coarse voxel.

    `fine_solid` must be coarse-aligned: shape == coarse shape * scale. Octant index is
    ox + 2*oz + 4*oy with o = 1 for the upper half along that axis (oy = height) -- the
    same convention `_decide_shape` reads.
    """
    X, Y, Z = (d // scale for d in fine_solid.shape)
    h = scale // 2
    c = fine_solid.reshape(X, 2, h, Y, 2, h, Z, 2, h).sum(axis=(2, 5, 8), dtype=np.uint16)
    # axes now (X, ox, Y, oy, Z, oz) -> (X, Y, Z, oy, oz, ox) so the flat index is 4oy+2oz+ox
    return np.ascontiguousarray(c.transpose(0, 2, 4, 3, 5, 1)).reshape(X, Y, Z, 8)


def shapes_from_counts(counts, scale):
    """Shape code per voxel from its octant counts (no interior/leak logic)."""
    masks = ((counts >= OCTANT_MIN[scale]).astype(np.uint8)
             << np.arange(8, dtype=np.uint8)).sum(axis=-1).astype(np.uint8)
    return _MASK_TO_SHAPE[masks]
