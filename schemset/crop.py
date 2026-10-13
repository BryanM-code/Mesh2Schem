"""Crop a build to the building: remove ground, landscape and site clutter, pad with air.

Builders build with anything -- gravel and suspicious gravel for concrete panels, snow and
ice blocks, coral blocks and ores for their colour -- so a block's name alone cannot say
"landscape". The crop combines a per-palette *role* with where the block is:

    COVER  plants, leaves, flowers, crops, snow layers, carpets of moss: lies on things.
           Removed unless it touches what is kept and does not stand on open ground (ivy,
           a window box, snow on a roof stay; a hedge or a lawn goes).
    TURF   grass, dirt and their family: ground at or below the ground level; above it,
           kept only where it touches the building (a green roof). The only block that
           marks a hill.
    FLUID  water, lava: a ground surface (a lake, the sea); above the ground, like COVER.
    HARD   stone, sand, gravel, clay, snow and ice blocks, ores, coral blocks, prismarine:
           ground at or below the ground level, building material above it.
    WOODY  logs and wood: a tree (removed) when it touches leaves and is trunk-shaped or
           canopy-heavy, building otherwise (log cabins, timber framing). See `_trees`.
    BUILT  everything else.
    INVISIBLE  light, structure_void, barrier: nothing to see, so they become air.

Steps (every voxel count is recorded):

1. **Ground level** G: over "open" columns (no building in them), the most common top of
   the ground run (TURF, HARD, FLUID) that starts at the column's lowest block -- COVER on
   top is not ground, so a snow layer or a flower does not lift the surface. Columns whose
   surface is natural (TURF, FLUID) are preferred; then any open column; then every column
   standing on ground. No ground under anything -> no ground level.
2. **Ground**: everything but BUILT and building WOODY at or below G goes. Those stay
   inside the footprint (basements, sunken floors).
3. **Footprint** (the rule of `mcvoxel.grid.remove_site`, measured from G instead of the
   lowest block): columns whose building material (BUILT, building WOODY, HARD outside
   hills) rises at least `SITE_HEIGHT` above the ground, kept only in masses at least
   `CORE_WIDTH` columns wide, grown by `MARGIN` columns. A hill column is one whose top
   block (cover and trees aside) is TURF with no BUILT or WOODY above the ground. Outside the
   footprint everything goes: roads, plazas, lawns, garden walls, fences, cars, lamp posts,
   trees, hills. A build with no such core keeps every building column (`core: "none"`).
4. Inside the footprint, above the ground: BUILT, building WOODY and HARD stay; COVER,
   TURF and FLUID stay only within `SOFT_REACH` of what stays and only in clumps that do
   not stand on open ground (a hedge or a lawn does; a green roof or ivy does not);
   trees go.
5. Crop to the kept blocks' box plus `PAD` blocks of air on every side.

Flags: `multi_building` when more than one large separate mass remains (26-connected after
bridging one-block gaps; large = at least `LARGE_SHARE` of the biggest and `LARGE_MIN`
blocks); `crop_aggressive` when more than half of the non-air blocks were removed;
`crop_kept_landscape` when COVER/TURF/FLUID are over `KEPT_LANDSCAPE` of what is kept;
`crop_no_core` when step 3 found no core.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

import block_roles

from .read import AIR, Build, split_state

AIR_R, COVER, TURF, FLUID, HARD, WOODY, BUILT = 0, 1, 2, 3, 4, 5, 6
TREE = 7                                      # WOODY found to be part of a tree
ROLE_NAMES = {AIR_R: "air", COVER: "cover", TURF: "turf", FLUID: "fluid", HARD: "hard",
              WOODY: "woody", BUILT: "built", TREE: "tree"}

SITE_HEIGHT = 3          # a column must rise this far above the ground (4+ blocks tall)
CORE_WIDTH = 3
MARGIN = 2
SOFT_REACH = 1
PAD = 2
KEPT_LANDSCAPE = 0.10
TREE_LEAF_SHARE = 0.2
TREE_SPAN = 7
TREE_CANOPY = 2.0
LARGE_SHARE = 0.15
LARGE_MIN = 500
AGGRESSIVE = 0.5

INVISIBLE = {"light", "structure_void", "barrier"}
TURF_BASES = {"grass_block", "dirt", "coarse_dirt", "podzol", "mycelium", "rooted_dirt", "mud",
              "farmland", "dirt_path", "grass_path", "moss_block", "pale_moss_block",
              "muddy_mangrove_roots", "crimson_nylium", "warped_nylium"}
FLUID_BASES = {"water", "lava", "bubble_column"}
COVER_BASES = {
    "snow", "moss_carpet", "pale_moss_carpet", "kelp", "kelp_plant", "seagrass",
    "tall_seagrass", "sea_pickle", "glow_lichen", "hanging_roots", "big_dripleaf",
    "big_dripleaf_stem", "small_dripleaf", "spore_blossom", "nether_sprouts", "crimson_roots",
    "warped_roots", "crimson_fungus", "warped_fungus", "weeping_vines", "weeping_vines_plant",
    "twisting_vines", "twisting_vines_plant", "cave_vines", "cave_vines_plant",
    "brown_mushroom", "red_mushroom", "melon_stem", "pumpkin_stem", "attached_melon_stem",
    "attached_pumpkin_stem", "cocoa", "nether_wart", "torchflower_crop", "pitcher_crop",
    "pitcher_plant", "bamboo_sapling", "frogspawn", "mangrove_propagule", "short_dry_grass",
    "tall_dry_grass", "bush", "firefly_bush", "leaf_litter", "wildflowers", "cactus_flower",
    "sculk_vein", "resin_clump", "closed_eyeblossom", "open_eyeblossom",
} | set(block_roles.GROUND_COVER)
HARD_BASES = {
    "stone", "andesite", "diorite", "granite", "deepslate", "tuff", "calcite", "cobblestone",
    "mossy_cobblestone", "sandstone", "red_sandstone", "sand", "red_sand", "gravel",
    "suspicious_sand", "suspicious_gravel", "clay", "snow_block", "powder_snow", "packed_ice",
    "ice", "blue_ice", "bedrock", "netherrack", "soul_sand", "soul_soil", "obsidian",
    "crying_obsidian", "magma_block", "dripstone_block", "pointed_dripstone", "amethyst_block",
    "budding_amethyst", "sculk", "prismarine", "sea_lantern", "sponge", "wet_sponge",
}
WOODY_NAMES = {"crimson_stem", "warped_stem", "stripped_crimson_stem", "stripped_warped_stem",
               "bamboo_block", "stripped_bamboo_block"}


def role_of(state: str) -> int:
    """Crop role of one palette entry (see the module docstring)."""
    if state == AIR:
        return AIR_R
    name = split_state(state)[0]
    namespace, _, base = name.rpartition(":")
    if namespace not in ("", "minecraft"):
        return BUILT                          # unknown legacy ids: solid, keep the shape
    if base in INVISIBLE:
        return AIR_R
    if base in TURF_BASES:
        return TURF
    if base in FLUID_BASES:
        return FLUID
    if (base in COVER_BASES or base.endswith(("_leaves", "_sapling"))
            or ("coral" in base and not base.endswith("_block"))):
        return COVER
    if (base in HARD_BASES or base.endswith("_ore") or base.startswith("infested_")
            or base.endswith("coral_block")):
        return HARD
    if base.endswith(("_log", "_wood", "_hyphae")) or base in WOODY_NAMES:
        return WOODY
    return BUILT


def is_leaves(state: str) -> bool:
    return split_state(state)[0].endswith("_leaves")


@dataclass
class CropResult:
    blocks: np.ndarray                  # uint16 (X, Y, Z), cropped and padded
    palette: list[str]
    offset: tuple[int, int, int]        # grid origin in the source grid's coordinates
    stats: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def _bbox(mask: np.ndarray):
    if not mask.any():
        return None
    out = []
    for axis in range(3):
        hit = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        out.append((int(hit[0]), int(hit[-1])))
    return [[lo for lo, _ in out], [hi for _, hi in out]]


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])          # ties: the lowest level


def _top_index(mask: np.ndarray) -> np.ndarray:
    """Per (x, z) column: the highest y where mask is set (-1 where it never is)."""
    Y = mask.shape[1]
    return np.where(mask.any(axis=1), Y - 1 - np.argmax(mask[:, ::-1, :], axis=1), -1)


def ground_level(role: np.ndarray) -> tuple[int | None, str, int]:
    """(G, method, columns used). See step 1 of the module docstring."""
    solid = (role != AIR_R) & (role != COVER)
    has = solid.any(axis=1)
    ground = (role == TURF) | (role == HARD) | (role == FLUID)
    bottom = np.argmax(solid, axis=1)
    on_ground = has & np.take_along_axis(ground, bottom[:, None, :], axis=1)[:, 0, :]
    if not on_ground.any():
        return None, "none", 0
    Y = role.shape[1]
    above_bottom = np.arange(Y)[None, :, None] >= bottom[:, None, :]
    breaks = ~ground & above_bottom
    first_break = np.where(breaks.any(axis=1), np.argmax(breaks, axis=1), Y)
    top = first_break - 1
    top_role = np.take_along_axis(role, np.clip(top, 0, Y - 1)[:, None, :], axis=1)[:, 0, :]
    natural = (top_role == TURF) | (top_role == FLUID)
    building = ((role == BUILT) | (role == WOODY)).any(axis=1)
    open_cols = on_ground & ~building
    for cols, method in ((open_cols & natural, "open_natural"), (open_cols, "open"),
                         (on_ground & natural, "under_natural"), (on_ground, "under")):
        if cols.sum() >= max(4, 0.02 * has.sum()):
            return _mode(top[cols]), method, int(cols.sum())
    return None, "none", 0


def _trees(role: np.ndarray, leaves: np.ndarray) -> np.ndarray:
    """WOODY voxels that belong to trees.

    A connected group of logs is a tree when it touches leaves and either at least
    `TREE_LEAF_SHARE` of it touches leaves (a trunk under a canopy), or it stays within a
    `TREE_SPAN` x `TREE_SPAN` column (a palm: a long bare trunk with leaves on top), or its
    box (grown by 2) holds `TREE_CANOPY` x as many leaves as it has logs (a big custom tree
    with a thick trunk and branches). Log walls, beams and timber frames are wider than a
    trunk and carry few leaves.
    """
    woody = role == WOODY
    if not woody.any() or not leaves.any():
        return np.zeros_like(woody)
    near_leaf = ndimage.binary_dilation(leaves, structure=np.ones((3, 3, 3), bool))
    labels, n = ndimage.label(woody, structure=np.ones((3, 3, 3), bool))
    size = np.bincount(labels.ravel(), minlength=n + 1)
    touching = np.bincount(labels.ravel(), weights=(near_leaf & woody).ravel(), minlength=n + 1)
    share = np.divide(touching, size, out=np.zeros(n + 1), where=size > 0)
    tree_label = share >= TREE_LEAF_SHARE
    for k, box in enumerate(ndimage.find_objects(labels), 1):
        if box is None or touching[k] == 0 or tree_label[k]:
            continue
        if box[0].stop - box[0].start <= TREE_SPAN and box[2].stop - box[2].start <= TREE_SPAN:
            tree_label[k] = True
            continue
        grown = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in box)
        tree_label[k] = int(leaves[grown].sum()) >= TREE_CANOPY * size[k]
    tree_label[0] = False
    return tree_label[labels]


def footprint_of(core: np.ndarray, ground: int | None) -> tuple[np.ndarray, str]:
    """2D (X, Z) footprint of the building: tall, broad building columns + margin."""
    has = core.any(axis=1)
    if not has.any():
        return np.zeros(has.shape, bool), "empty"
    base = (ground + 1) if ground is not None else int(np.flatnonzero(core.any(axis=(0, 2)))[0])
    tall = _top_index(core) >= base + SITE_HEIGHT
    broad = ndimage.binary_opening(tall, structure=np.ones((CORE_WIDTH, CORE_WIDTH), bool))
    square = np.ones((3, 3), bool)
    if broad.any():
        return ndimage.binary_dilation(broad, structure=square, iterations=MARGIN), "ok"
    return ndimage.binary_dilation(has, structure=square, iterations=MARGIN), "none"


def masses(solid: np.ndarray) -> list[int]:
    """Sizes of separate masses, largest first (26-connected after bridging 1-block gaps)."""
    if not solid.any():
        return []
    bridged = ndimage.binary_dilation(solid, structure=np.ones((3, 3, 3), bool))
    labels, n = ndimage.label(bridged, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(labels[solid], minlength=n + 1)[1:]
    return sorted((int(s) for s in sizes if s), reverse=True)


def crop_build(build: Build) -> CropResult:
    roles = np.array([role_of(s) for s in build.palette], dtype=np.uint8)
    leaf_pal = np.array([is_leaves(s) for s in build.palette], dtype=bool)
    invisible_pal = np.array([s != AIR and roles[i] == AIR_R
                              for i, s in enumerate(build.palette)], dtype=bool)
    role = roles[build.blocks]
    role[_trees(role, leaf_pal[build.blocks])] = TREE
    nonair_orig = build.blocks != 0

    G, ground_method, ground_columns = ground_level(role)
    Y = role.shape[1]
    above = (np.arange(Y) > G)[None, :, None] if G is not None else np.ones((1, Y, 1), bool)
    built = (role == BUILT) | (role == WOODY)
    soft = (role == COVER) | (role == TURF) | (role == FLUID)

    # A hill: its top block, cover and trees aside, is turf, and nothing built stands in it.
    solid_no_cover = (role != AIR_R) & (role != COVER) & (role != TREE)
    top_y = _top_index(solid_no_cover)
    top_role = np.take_along_axis(role, np.clip(top_y, 0, Y - 1)[:, None, :], axis=1)[:, 0, :]
    hill = (top_y >= 0) & (top_role == TURF) & ~(built & above).any(axis=1)
    material = (built | ((role == HARD) & ~hill[:, None, :])) & above
    foot, core_status = footprint_of(material, G)
    in_foot = foot[:, None, :]

    keep_hard = (built | (role == HARD)) & above & in_foot
    keep_below = built & ~above & in_foot
    anchor = keep_hard | keep_below
    reach = ndimage.binary_dilation(anchor, structure=np.ones((3, 3, 3), bool),
                                    iterations=SOFT_REACH) if anchor.any() else anchor
    # A clump of cover/turf/fluid standing on open ground -- snow, lawn, a hedge or bush or
    # a vine rooted against the wall -- is landscape. Clumps that do not reach the ground,
    # or reach it only under the building (a planter under a porch roof), sit on the
    # building: green roofs, window boxes, ivy, snow on a roof.
    roofed = np.maximum.accumulate(anchor[:, ::-1, :], axis=1)[:, ::-1, :]
    candidate = soft & above & in_foot
    keep_soft = candidate & reach
    if G is not None and candidate.any():
        labels, n = ndimage.label(candidate, structure=np.ones((3, 3, 3), bool))
        rooted = np.unique(labels[:, G + 1, :][~roofed[:, G + 1, :] & candidate[:, G + 1, :]])
        keep_soft &= ~np.isin(labels, rooted[rooted > 0])
    kept = keep_hard | keep_below | keep_soft

    removed = nonair_orig & ~kept
    before = int(nonair_orig.sum())
    by_role = {ROLE_NAMES[code]: int((removed & (role == code)).sum())
               for code in (COVER, TURF, FLUID, HARD, WOODY, BUILT, TREE)}
    by_role["invisible"] = int((removed & invisible_pal[build.blocks]).sum())
    removed_n = int(removed.sum())
    fraction = removed_n / before if before else 0.0

    box = _bbox(kept)
    stats = {
        "ground_level": G, "ground_method": ground_method, "ground_columns": ground_columns,
        "core": core_status, "footprint_columns": int(foot.sum()), "hill_columns": int(hill.sum()),
        "bbox_before": _bbox(nonair_orig), "bbox_after": box,
        "non_air_before": before, "non_air_after": int(kept.sum()),
        "removed": removed_n, "removed_fraction": round(fraction, 4),
        "removed_by_role": by_role,
        "removed_built_above_ground": int((removed & built & above).sum()),
        "kept_below_ground": int(keep_below.sum()),
        "kept_landscape": int(keep_soft.sum()),
    }
    if box is None:
        return CropResult(np.zeros((1, 1, 1), np.uint16), [AIR], (0, 0, 0),
                          {**stats, "masses": []}, ["crop_empty", "crop_aggressive"])

    lo = [box[0][a] - PAD for a in range(3)]
    hi = [box[1][a] + PAD + 1 for a in range(3)]
    shape = [hi[a] - lo[a] for a in range(3)]
    src = tuple(slice(max(lo[a], 0), min(hi[a], build.blocks.shape[a])) for a in range(3))
    dst = tuple(slice(s.start - lo[a], s.stop - lo[a]) for a, s in enumerate(src))
    cropped = np.zeros(shape, dtype=build.blocks.dtype)
    cropped[dst] = np.where(kept[src], build.blocks[src], 0)

    used = np.flatnonzero(np.bincount(cropped.ravel(), minlength=len(build.palette)))
    palette = [AIR] + sorted(build.palette[i] for i in used if i != 0)
    slot = {s: k for k, s in enumerate(palette)}
    lookup = np.zeros(len(build.palette), dtype=np.uint16)
    for i in used:
        lookup[i] = slot[build.palette[i]]
    cropped = lookup[cropped]

    new_roles = np.array([role_of(s) for s in palette], dtype=np.uint8)
    sizes = masses(np.isin(new_roles[cropped], (HARD, WOODY, BUILT)))
    large = [s for s in sizes if s >= LARGE_MIN and s >= LARGE_SHARE * sizes[0]] if sizes else []
    stats["masses"] = sizes[:8]
    stats["large_masses"] = len(large)
    stats["dims_after"] = shape
    flags = []
    if len(large) > 1:
        flags.append("multi_building")
    if fraction > AGGRESSIVE:
        flags.append("crop_aggressive")
    if stats["kept_landscape"] > KEPT_LANDSCAPE * stats["non_air_after"]:
        flags.append("crop_kept_landscape")
    if core_status != "ok":
        flags.append("crop_no_core")
    return CropResult(cropped, palette, tuple(lo), stats, flags)
