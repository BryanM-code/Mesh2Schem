"""Crop a build to the building: remove landscape and site clutter, pad with air.

Block names are only a hint; geometry decides (D23). Builders build with anything --
stone, gravel and sandstone for walls and roofs, snow blocks, ores and coral for colour --
so a natural-sounding name never makes a block landscape on its own. Each palette entry
gets a *role* from its name:

    COVER    plants, leaves, flowers, crops, snow layers, moss carpet: lies on things.
    TURF     grass, dirt and their family.
    FLUID    water, lava, and the plants that grow in water.
    NATURAL  natural minerals: the stone family, sand, gravel, clay, snow and ice blocks,
             ores, bedrock, netherrack, end stone. Terrain only where they belong to the
             natural ground mass; building material everywhere else.
    WOODY    logs and wood: a tree (removed) or building (log cabins, timber frames).
    BUILT    everything else (cobblestone, bricks, planks, concrete, glass, ...).
    INVISIBLE  light, structure_void, barrier: nothing to see, so they become air.

Steps (every voxel count is recorded):

1. **Natural terrain** (`terrain`): in each column, the *ground run* is the unbroken run of
   TURF / FLUID / NATURAL blocks that starts at the bottom of the capture (within
   `BOTTOM_BAND` layers of the lowest ground layer). Terrain is layered and broad, so each
   layer of the runs is opened with a `TERRAIN_WIDTH` square: walls, pillars and chimneys
   standing on the ground are thinner than that and drop out, hills and plates stay. The
   columns left form regions; a region is natural terrain only if it reaches the edge of
   the build and shows natural evidence -- at least `SOFT_SHARE` of its surface is soft
   ground (grass, dirt, sand, gravel, snow, water, ...) or it is a rock mass at least
   `ROCK_DEPTH` deep. A paved plaza or a stone platform is neither, so it is not terrain.
2. **Ground height** per column (`ground_height`): the terrain surface in *open* terrain
   columns (no building material above them), carried into every other column from its
   nearest open terrain (the lowest one where several are equally near). Under a building
   the terrain never rises above that height, so a stone or sand floor laid on the ground,
   or a natural-stone podium, is building, not terrain. Where every terrain column has
   building above it (the capture is the building on a layer of ground), only soft ground
   (dirt, grass, sand, water, ...) under it is terrain; rock there stays as building. No natural
   terrain: no ground, and nothing is below ground (D23). The cropped grid then gets a
   *virtual* ground plane just under its lowest kept block -- no block is below it -- so
   that a building whose floor was the world's ground does not flood from below.
3. **Trees** (`_trees`): groups of logs that touch leaves *and stand on the ground*, with
   their leaves. Logs inside a building that happen to touch leaves (a timber frame
   behind an ivy bush) are building.
4. **Footprint** (the idea of `mcvoxel.grid.remove_site`, measured from the local ground):
   columns whose building material rises at least `SITE_HEIGHT` above the ground, in masses
   at least `CORE_WIDTH` columns wide. A core mass no wider than `LINEAR_HALF_WIDTH` from
   its centre line, at least `LINEAR_LENGTH` long and no taller than `WALL_HEIGHT` above
   the ground (90th percentile of its columns) is a wall or fence line -- a cemetery wall
   with its coping -- not a building; a taller narrow mass is a building wing and stays. The building cores, grown by `MARGIN` columns,
   are the footprint; every building material there stays. Building material attached to
   it stays up to `REACH` columns from the cores (steps, stoops, porches, chimneys, short
   garden walls); beyond that, site structures go -- boundary walls, fences, paths, roads,
   plazas, cars, lamp posts. Every separate building core is kept (compounds).
5. Soft blocks (COVER, TURF, FLUID that are not terrain) stay only next to what is kept
   (`SOFT_REACH`), and only in clumps that do not stand on open ground -- terrain, the
   ground, removed site material or the bottom of the capture, with nothing kept overhead:
   a green roof, a window box, ivy or a pool stay; a lawn, a hedge or a bush go.
6. **Below ground**: kept blocks below the ground height of their column are buried
   (masked later, never deleted). If more than `BELOW_GROUND_LIMIT` of the kept blocks
   would be buried, the ground is not trusted: there is no ground (`ground_fallback`).
7. Crop to the kept blocks' box plus `PAD` blocks of air on every side.
8. **Terrain floors** (`floor_under_rooms`, after the masks): where an enclosed room stands
   directly on the terrain -- the builder used the world's ground as its floor, as in a gym
   hall on the grass -- the top terrain block under that room is put back as its floor.
   It is interior (ignored) like any floor; the grid just has no hole at the bottom.

Flags: `multi_building` when more than one large separate mass remains (26-connected after
bridging one-block gaps; large = at least `LARGE_SHARE` of the biggest and `LARGE_MIN`
blocks); `crop_aggressive` when more than half of the non-air blocks were removed;
`crop_kept_landscape` when soft blocks are over `KEPT_LANDSCAPE` of what is kept;
`crop_no_core` when step 4 found no core; `site_linear_removed` when a wall or fence line
was cut back to `REACH`; `ground_fallback` (step 6).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

import block_roles

from .read import AIR, Build, split_state

AIR_R, COVER, TURF, FLUID, NATURAL, WOODY, BUILT = 0, 1, 2, 3, 4, 5, 6
TREE = 7                                      # WOODY found to be part of a tree
ROLE_NAMES = {AIR_R: "air", COVER: "cover", TURF: "turf", FLUID: "fluid",
              NATURAL: "natural", WOODY: "woody", BUILT: "built", TREE: "tree"}

BOTTOM_BAND = 2
TERRAIN_WIDTH = 3
SOFT_SHARE = 0.3
ROCK_DEPTH = 3
SITE_HEIGHT = 3          # a column must rise this far above the ground (4+ blocks tall)
CORE_WIDTH = 3
MARGIN = 2
REACH = 6
LINEAR_HALF_WIDTH = 2.0  # max distance to the edge of the mass: <= 4 columns wide
LINEAR_LENGTH = 10
WALL_HEIGHT = 6          # a wall or fence line is no taller than this above the ground
SOFT_REACH = 1
PAD = 2
BELOW_GROUND_LIMIT = 0.15
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
FLUID_BASES = {"water", "lava", "bubble_column", "kelp", "kelp_plant", "seagrass",
               "tall_seagrass", "sea_pickle"}
COVER_BASES = {
    "snow", "moss_carpet", "pale_moss_carpet", "glow_lichen", "hanging_roots", "big_dripleaf",
    "big_dripleaf_stem", "small_dripleaf", "spore_blossom", "nether_sprouts", "crimson_roots",
    "warped_roots", "crimson_fungus", "warped_fungus", "weeping_vines", "weeping_vines_plant",
    "twisting_vines", "twisting_vines_plant", "cave_vines", "cave_vines_plant",
    "brown_mushroom", "red_mushroom", "melon_stem", "pumpkin_stem", "attached_melon_stem",
    "attached_pumpkin_stem", "cocoa", "nether_wart", "torchflower_crop", "pitcher_crop",
    "pitcher_plant", "bamboo_sapling", "frogspawn", "mangrove_propagule", "short_dry_grass",
    "tall_dry_grass", "bush", "firefly_bush", "leaf_litter", "wildflowers", "cactus_flower",
    "sculk_vein", "resin_clump", "closed_eyeblossom", "open_eyeblossom",
} | (set(block_roles.GROUND_COVER) - FLUID_BASES)
# Soft ground: a terrain surface made of these is natural evidence (with TURF and FLUID).
SOFT_NATURAL = {"sand", "red_sand", "gravel", "clay", "snow_block", "powder_snow", "ice",
                "packed_ice", "blue_ice", "soul_sand", "soul_soil", "suspicious_sand",
                "suspicious_gravel", "netherrack", "end_stone"}
NATURAL_BASES = SOFT_NATURAL | {
    "stone", "andesite", "diorite", "granite", "deepslate", "tuff", "calcite", "sandstone",
    "red_sandstone", "bedrock", "magma_block", "dripstone_block", "pointed_dripstone",
    "amethyst_block", "budding_amethyst", "sculk", "basalt", "smooth_basalt", "blackstone",
}
WOODY_NAMES = {"crimson_stem", "warped_stem", "stripped_crimson_stem", "stripped_warped_stem",
               "bamboo_block", "stripped_bamboo_block"}


def _base(state: str) -> tuple[str, str]:
    name = split_state(state)[0]
    namespace, _, base = name.rpartition(":")
    return namespace, base


def role_of(state: str) -> int:
    """Crop role of one palette entry (see the module docstring)."""
    if state == AIR:
        return AIR_R
    namespace, base = _base(state)
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
    if (base in NATURAL_BASES or base.endswith("_ore") or base.startswith("infested_")):
        return NATURAL
    if base.endswith(("_log", "_wood", "_hyphae")) or base in WOODY_NAMES:
        return WOODY
    return BUILT


def is_leaves(state: str) -> bool:
    return split_state(state)[0].endswith("_leaves")


def is_soft_ground(state: str) -> bool:
    """A terrain surface of this block is natural evidence (grass, dirt, sand, water, ...)."""
    role = role_of(state)
    return role in (TURF, FLUID) or (role == NATURAL and _base(state)[1] in SOFT_NATURAL)


@dataclass
class CropResult:
    blocks: np.ndarray                  # uint16 (X, Y, Z), cropped and padded
    palette: list[str]
    offset: tuple[int, int, int]        # grid origin in the source grid's coordinates
    ground: np.ndarray | None = None    # int16 (X, Z): ground height in this grid, -1 none
    stats: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    # Source-grid masks for review renders and floors (not saved): what went, and why.
    removed_landscape: np.ndarray | None = None
    removed_site: np.ndarray | None = None
    terrain: np.ndarray | None = None


def _bbox(mask: np.ndarray):
    if not mask.any():
        return None
    out = []
    for axis in range(3):
        hit = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        out.append((int(hit[0]), int(hit[-1])))
    return [[lo for lo, _ in out], [hi for _, hi in out]]


def _top_index(mask: np.ndarray) -> np.ndarray:
    """Per (x, z) column: the highest y where mask is set (-1 where it never is)."""
    Y = mask.shape[1]
    return np.where(mask.any(axis=1), Y - 1 - np.argmax(mask[:, ::-1, :], axis=1), -1)


# --------------------------------------------------------------------------- terrain

def ground_runs(role: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(low, top) per column of the ground run that starts at the bottom of the capture.

    The run is unbroken TURF / FLUID / NATURAL from the column's lowest non-cover block,
    which must lie within BOTTOM_BAND of the lowest ground layer. -1 where none.
    """
    Y = role.shape[1]
    cand = (role == TURF) | (role == FLUID) | (role == NATURAL)
    base = (role != AIR_R) & (role != COVER)
    if not cand.any():
        none = np.full(role.shape[::2], -1, dtype=np.int32)
        return none, none
    floor = int(np.flatnonzero(cand.any(axis=(0, 2)))[0])
    has = base.any(axis=1)
    low = np.argmax(base, axis=1)
    at_low = np.take_along_axis(cand, low[:, None, :], axis=1)[:, 0, :]
    ok = has & at_low & (low <= floor + BOTTOM_BAND)
    ys = np.arange(Y)[None, :, None]
    brk = ~cand & (ys >= low[:, None, :])
    first = np.where(brk.any(axis=1), np.argmax(brk, axis=1), Y)
    top = np.where(ok, first - 1, -1).astype(np.int32)
    low = np.where(ok, low, -1).astype(np.int32)
    return low, top


def terrain(role: np.ndarray, soft: np.ndarray) -> tuple[np.ndarray, dict]:
    """Natural terrain voxels and the per-column terrain surface (see step 1)."""
    X, Y, Z = role.shape
    low, top = ground_runs(role)
    stats = {"terrain_regions": 0, "terrain_rejected": 0, "terrain_columns": 0}
    surface = np.full((X, Z), -1, dtype=np.int32)
    if (top < 0).all():
        return np.zeros(role.shape, bool), {**stats, "surface": surface}
    # Each layer of the runs must be broad: the grey opening of the run tops is the
    # per-layer binary opening, so thin walls and pillars on the ground drop out.
    opened = ndimage.grey_opening(np.where(top >= 0, top, -1), size=(TERRAIN_WIDTH,) * 2,
                                  mode="nearest")
    opened = np.where((top >= 0) & (opened >= low), np.minimum(opened, top), -1)
    cols = opened >= 0
    labels, n = ndimage.label(cols, structure=np.ones((3, 3), bool))
    solid = role != AIR_R
    box = _bbox(solid)
    (x0, _, z0), (x1, _, z1) = box
    edge = np.zeros((X, Z), bool)
    edge[x0, :] = edge[x1, :] = edge[:, z0] = edge[:, z1] = True
    # Natural evidence is read off the real surface (the run's top), not the opened one.
    surf_soft = np.take_along_axis(soft, np.clip(top, 0, Y - 1)[:, None, :], axis=1)[:, 0, :]
    keep = np.zeros(n + 1, bool)
    for k in range(1, n + 1):
        region = labels == k
        if not (region & edge).any():
            stats["terrain_rejected"] += 1
            continue
        share = float(surf_soft[region].mean())
        depth = float(np.median(opened[region] - low[region] + 1))
        if share >= SOFT_SHARE or depth >= ROCK_DEPTH:
            keep[k] = True
            stats["terrain_regions"] += 1
        else:
            stats["terrain_rejected"] += 1
    tcols = keep[labels]
    surface = np.where(tcols, opened, -1).astype(np.int32)
    ys = np.arange(Y)[None, :, None]
    mask = tcols[:, None, :] & (ys >= low[:, None, :]) & (ys <= surface[:, None, :])
    stats["terrain_columns"] = int(tcols.sum())
    return mask, {**stats, "surface": surface}


def ground_height(surface: np.ndarray, open_cols: np.ndarray) -> np.ndarray:
    """Terrain surface in open terrain columns, carried into every other column from its
    nearest open terrain (the lowest one where several are equally near). -1: none."""
    if not open_cols.any():
        return np.full(surface.shape, -1, dtype=np.int32)
    big = np.iinfo(np.int32).max
    h = np.where(open_cols, surface, big).astype(np.int64)
    known = open_cols.copy()
    while not known.all():
        nb = ndimage.grey_erosion(h, size=(3, 3), mode="nearest")
        grow = ~known & (nb < big)
        if not grow.any():
            break
        h[grow] = nb[grow]
        known |= grow
    return np.where(known, h, -1).astype(np.int32)


def _trees(role: np.ndarray, leaves: np.ndarray, grounded_at: np.ndarray) -> np.ndarray:
    """WOODY voxels (and their leaves) that belong to trees.

    A connected group of logs is a tree when it touches leaves, stands on the ground (its
    lowest log is on the terrain, at the ground height, or at the bottom of the capture),
    and either at least `TREE_LEAF_SHARE` of it touches leaves (a trunk under a canopy), or
    it stays within a `TREE_SPAN` x `TREE_SPAN` column (a palm), or its box (grown by 2)
    holds `TREE_CANOPY` x as many leaves as it has logs (a big custom tree). Leaves joined
    to a tree's logs go with it.
    """
    woody = role == WOODY
    out = np.zeros_like(woody)
    if not woody.any() or not leaves.any():
        return out
    near_leaf = ndimage.binary_dilation(leaves, structure=np.ones((3, 3, 3), bool))
    labels, n = ndimage.label(woody, structure=np.ones((3, 3, 3), bool))
    size = np.bincount(labels.ravel(), minlength=n + 1)
    touching = np.bincount(labels.ravel(), weights=(near_leaf & woody).ravel(), minlength=n + 1)
    share = np.divide(touching, size, out=np.zeros(n + 1), where=size > 0)
    tree_label = np.zeros(n + 1, bool)
    for k, box in enumerate(ndimage.find_objects(labels), 1):
        if box is None or touching[k] == 0:
            continue
        part = labels[box] == k
        ys = np.nonzero(part.any(axis=(0, 2)))[0]
        y_lo = box[1].start + int(ys[0])
        foot = part[:, ys[0], :]
        if not grounded_at[box[0], y_lo, box[2]][foot].any():
            continue                                  # floating: part of a building
        if share[k] >= TREE_LEAF_SHARE:
            tree_label[k] = True
        elif box[0].stop - box[0].start <= TREE_SPAN and box[2].stop - box[2].start <= TREE_SPAN:
            tree_label[k] = True
        else:
            grown = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in box)
            tree_label[k] = int(leaves[grown].sum()) >= TREE_CANOPY * size[k]
    trunks = tree_label[labels]
    if trunks.any():
        lab, _ = ndimage.label(leaves | trunks, structure=np.ones((3, 3, 3), bool))
        ids = np.unique(lab[trunks])
        out = np.isin(lab, ids[ids > 0])
    return out


# --------------------------------------------------------------------------- footprint

def _edt_max(labels: np.ndarray, n: int) -> np.ndarray:
    """Per label: the largest distance from inside it to its edge (half its width)."""
    if not n:
        return np.zeros(0)
    dist = ndimage.distance_transform_edt(np.pad(labels > 0, 1))[1:-1, 1:-1]
    return np.asarray(ndimage.maximum(dist, labels, index=np.arange(1, n + 1)))


def footprint(material: np.ndarray, base: np.ndarray) -> dict:
    """Building cores, wall/fence lines, footprint and reach (2D masks). See step 4."""
    X, Y, Z = material.shape
    top = _top_index(material)
    tall = (top >= 0) & (top >= base + SITE_HEIGHT)
    broad = ndimage.binary_opening(tall, structure=np.ones((CORE_WIDTH,) * 2, bool))
    labels, n = ndimage.label(broad, structure=np.ones((3, 3), bool))
    half = _edt_max(labels, n)
    rise = top - base + 1
    linear = np.zeros(n + 1, bool)
    for k, box in enumerate(ndimage.find_objects(labels), 1):
        if box is None:
            continue
        length = max(box[0].stop - box[0].start, box[1].stop - box[1].start)
        low = np.percentile(rise[box][labels[box] == k], 90) <= WALL_HEIGHT
        linear[k] = half[k - 1] <= LINEAR_HALF_WIDTH and length >= LINEAR_LENGTH and low
    building = (labels > 0) & ~linear[labels]
    status = "ok"
    if not building.any():
        building = labels > 0                       # only lines: keep them, no cutting
        status = "lines_only" if n else "none"
    if not building.any():
        building = material.any(axis=1)              # nothing tall and broad: keep all
    square = np.ones((3, 3), bool)
    near = ndimage.binary_dilation(building, structure=square, iterations=MARGIN)
    reach = ndimage.binary_dilation(building, structure=square, iterations=REACH)
    return {"core": building, "near": near, "reach": reach, "lines": (labels > 0) & linear[labels],
            "status": status, "components": int(n), "linear_components": int(linear.sum())}


def masses(solid: np.ndarray) -> list[int]:
    """Sizes of separate masses, largest first (26-connected after bridging 1-block gaps)."""
    if not solid.any():
        return []
    bridged = ndimage.binary_dilation(solid, structure=np.ones((3, 3, 3), bool))
    labels, n = ndimage.label(bridged, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(labels[solid], minlength=n + 1)[1:]
    return sorted((int(s) for s in sizes if s), reverse=True)


def _keep(role, material, soft, terrain_mask, height):
    """Footprint, attached structures and soft blocks for one ground height map."""
    Y = role.shape[1]
    ys = np.arange(Y)[None, :, None]
    if material.any():
        lowest = int(np.flatnonzero(material.any(axis=(0, 2)))[0])
    else:
        lowest = 0
    base = np.where(height >= 0, height + 1, lowest)
    fp = footprint(material, base)
    near3 = fp["near"][:, None, :]
    reach3 = fp["reach"][:, None, :]
    keep_mat = material & near3
    zone = material & reach3
    if zone.any():
        lab, _ = ndimage.label(zone, structure=np.ones((3, 3, 3), bool))
        ids = np.unique(lab[keep_mat & zone])
        keep_mat |= np.isin(lab, ids[ids > 0])
    keep_soft = np.zeros_like(soft)
    cand = soft & reach3
    if cand.any() and keep_mat.any():
        near_kept = ndimage.binary_dilation(keep_mat, structure=np.ones((3, 3, 3), bool),
                                            iterations=SOFT_REACH)
        roofed = np.maximum.accumulate(keep_mat[:, ::-1, :], axis=1)[:, ::-1, :]
        support = terrain_mask | (material & ~keep_mat)          # ground or removed site
        on_ground = np.zeros_like(cand)
        on_ground[:, 1:, :] = support[:, :-1, :]
        on_ground[:, 0, :] = True                                # the bottom of the capture
        on_ground |= (height[:, None, :] >= 0) & (ys <= height[:, None, :] + 1)
        lab, n = ndimage.label(cand, structure=np.ones((3, 3, 3), bool))
        rooted = np.unique(lab[cand & on_ground & ~roofed])
        keep_soft = cand & near_kept & ~np.isin(lab, rooted[rooted > 0])
    return keep_mat, keep_soft, fp


def crop_build(build: Build) -> CropResult:
    roles = np.array([role_of(s) for s in build.palette], dtype=np.uint8)
    leaf_pal = np.array([is_leaves(s) for s in build.palette], dtype=bool)
    soft_pal = np.array([is_soft_ground(s) for s in build.palette], dtype=bool)
    invisible_pal = np.array([s != AIR and roles[i] == AIR_R
                              for i, s in enumerate(build.palette)], dtype=bool)
    role = roles[build.blocks]
    nonair_orig = build.blocks != 0
    Y = role.shape[1]
    ys = np.arange(Y)[None, :, None]

    terrain_mask, tstats = terrain(role, soft_pal[build.blocks])
    surface = tstats.pop("surface")
    leaves = leaf_pal[build.blocks]
    grounded = np.zeros_like(terrain_mask)
    grounded[:, 1:, :] = terrain_mask[:, :-1, :]
    floor = int(np.flatnonzero((role != AIR_R).any(axis=(0, 2)))[0]) if nonair_orig.any() else 0
    grounded[:, :floor + 2, :] = True
    tree = _trees(role, leaves & ~terrain_mask, grounded | (ys <= surface[:, None, :] + 1))
    role[tree & (role == WOODY)] = TREE

    material = ((role == BUILT) | (role == WOODY) | (role == NATURAL)) & ~terrain_mask & ~tree
    open_cols = (surface >= 0) & ~(material & (ys > surface[:, None, :])).any(axis=1)
    height = ground_height(surface, open_cols)
    if open_cols.any():
        capped = np.where(open_cols, surface, np.minimum(surface, height))
        terrain_mask &= ys <= capped[:, None, :]
        surface = capped
    else:                       # terrain only under the building: soft ground alone is terrain
        terrain_mask &= soft_pal[build.blocks]
    material = ((role == BUILT) | (role == WOODY) | (role == NATURAL)) & ~terrain_mask & ~tree
    soft = ((role == COVER) | (role == TURF) | (role == FLUID)) & ~terrain_mask & ~tree
    G = int(np.median(surface[open_cols])) if open_cols.any() else None
    method = "terrain" if G is not None else "none"

    keep_mat, keep_soft, fp = _keep(role, material, soft, terrain_mask, height)
    kept = keep_mat | keep_soft
    below = kept & (ys < height[:, None, :])
    if kept.any() and below.sum() > BELOW_GROUND_LIMIT * kept.sum():
        height = np.full_like(height, -1)
        method = "fallback"
        keep_mat, keep_soft, fp = _keep(role, material, soft, terrain_mask, height)
        kept = keep_mat | keep_soft
        below = kept & (ys < height[:, None, :])

    removed = nonair_orig & ~kept
    landscape = removed & (terrain_mask | tree | soft | invisible_pal[build.blocks])
    site = removed & ~landscape
    before = int(nonair_orig.sum())
    by_role = {ROLE_NAMES[code]: int((removed & (role == code)).sum())
               for code in (COVER, TURF, FLUID, NATURAL, WOODY, BUILT, TREE)}
    by_role["invisible"] = int((removed & invisible_pal[build.blocks]).sum())
    removed_n = int(removed.sum())
    fraction = removed_n / before if before else 0.0
    lines_removed = int((site & fp["lines"][:, None, :]).sum())

    box = _bbox(kept)
    stats = {
        "ground_level": G, "ground_method": method, "ground_columns": int(open_cols.sum()),
        **tstats, "terrain_voxels": int(terrain_mask.sum()),
        "core": fp["status"], "core_components": fp["components"],
        "linear_components": fp["linear_components"],
        "footprint_columns": int(fp["near"].sum()),
        "bbox_before": _bbox(nonair_orig), "bbox_after": box,
        "non_air_before": before, "non_air_after": int(kept.sum()),
        "removed": removed_n, "removed_fraction": round(fraction, 4),
        "removed_by_role": by_role,
        "removed_landscape": int(landscape.sum()), "removed_site": int(site.sum()),
        "removed_lines": lines_removed,
        "removed_built_above_ground": int((site & material & (ys > height[:, None, :])).sum()),
        "kept_below_ground": int(below.sum()),
        "kept_landscape": int(keep_soft.sum()),
    }
    if box is None:
        return CropResult(np.zeros((1, 1, 1), np.uint16), [AIR], (0, 0, 0),
                          np.full((1, 1), -1, np.int16), {**stats, "masses": []},
                          ["crop_empty", "crop_aggressive"], landscape, site, terrain_mask)

    lo = [box[0][a] - PAD for a in range(3)]
    hi = [box[1][a] + PAD + 1 for a in range(3)]
    shape = [hi[a] - lo[a] for a in range(3)]
    src = tuple(slice(max(lo[a], 0), min(hi[a], build.blocks.shape[a])) for a in range(3))
    dst = tuple(slice(s.start - lo[a], s.stop - lo[a]) for a, s in enumerate(src))
    cropped = np.zeros(shape, dtype=build.blocks.dtype)
    cropped[dst] = np.where(kept[src], build.blocks[src], 0)

    # Ground height in the cropped grid; padding columns take their nearest column's.
    # No ground: a virtual plane under the lowest kept block (see step 2).
    xi = np.clip(np.arange(lo[0], hi[0]), 0, build.blocks.shape[0] - 1)
    zi = np.clip(np.arange(lo[2], hi[2]), 0, build.blocks.shape[2] - 1)
    h = height[np.ix_(xi, zi)]
    if (h < 0).all():
        ground = np.full(h.shape, PAD - 1, dtype=np.int16)
        stats["virtual_ground"] = True
    else:
        ground = np.where(h >= 0, np.maximum(h - lo[1], -1), -1).astype(np.int16)
        stats["virtual_ground"] = False

    used = np.flatnonzero(np.bincount(cropped.ravel(), minlength=len(build.palette)))
    palette = [AIR] + sorted(build.palette[i] for i in used if i != 0)
    slot = {s: k for k, s in enumerate(palette)}
    lookup = np.zeros(len(build.palette), dtype=np.uint16)
    for i in used:
        lookup[i] = slot[build.palette[i]]
    cropped = lookup[cropped]

    new_roles = np.array([role_of(s) for s in palette], dtype=np.uint8)
    sizes = masses(np.isin(new_roles[cropped], (NATURAL, WOODY, BUILT)))
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
    if fp["status"] != "ok":
        flags.append("crop_no_core")
    if lines_removed:
        flags.append("site_linear_removed")
    if method == "fallback":
        flags.append("ground_fallback")
    return CropResult(cropped, palette, tuple(lo), ground, stats, flags, landscape, site,
                      terrain_mask)


def floor_under_rooms(build: Build, crop: CropResult, interior: np.ndarray) -> int:
    """Step 8: put back the top terrain block under interior air that stands on the ground.

    `interior` is the masks' interior (cropped grid). Updates `crop.blocks` / `.palette` in
    place and returns the number of blocks restored."""
    crop.stats.setdefault("terrain_floor", 0)
    if crop.terrain is None or crop.ground is None:
        return 0
    X, Y, Z = crop.blocks.shape
    g = crop.ground.astype(np.int64)
    xs, zs = np.nonzero((g >= 0) & (g + 1 < Y))
    if not len(xs):
        return 0
    gy = g[xs, zs]
    room = interior[xs, gy + 1, zs] & (crop.blocks[xs, gy + 1, zs] == 0)
    sx, sy, sz = xs + crop.offset[0], gy + crop.offset[1], zs + crop.offset[2]
    inside = ((sx >= 0) & (sx < build.blocks.shape[0]) & (sy >= 0) & (sy < build.blocks.shape[1])
              & (sz >= 0) & (sz < build.blocks.shape[2]))
    pick = room & inside
    pick[pick] = crop.terrain[sx[pick], sy[pick], sz[pick]] & (crop.blocks[xs[pick], gy[pick], zs[pick]] == 0)
    if not pick.any():
        return 0
    states = [build.palette[i] for i in build.blocks[sx[pick], sy[pick], sz[pick]]]
    palette = sorted(set(crop.palette[1:]) | set(states))
    palette = [AIR] + palette
    slot = {s: k for k, s in enumerate(palette)}
    remap = np.array([slot[s] for s in crop.palette], dtype=np.uint16)
    blocks = remap[crop.blocks]
    blocks[xs[pick], gy[pick], zs[pick]] = [slot[s] for s in states]
    crop.blocks, crop.palette = blocks, palette
    n = int(pick.sum())
    s = crop.stats
    s["terrain_floor"] = n
    s["non_air_after"] += n
    s["removed"] -= n
    s["removed_landscape"] -= n
    s["removed_fraction"] = round(s["removed"] / s["non_air_before"], 4) if s["non_air_before"] else 0.0
    return n
