"""What a block is for: its role in a build.

The crop (`dataset.crop`), the storey estimate and the vocabulary coverage need the same
answers about every block: is it a full cube, is it terrain, is it ground cover, what is its
modern name. The tables live here once.

Builds stored in the legacy format carry pre-1.13 names (`stonebrick`, `planks`, `leaves`,
`concretePowder`, `stained_hardened_clay`): `normalise` folds them onto their modern
spellings (`LEGACY_ALIASES`) so they do not read as unrelated materials.
"""
from __future__ import annotations


# Plants that sit *on* the ground. Counted with terrain for pad detection (a grass-covered pad
# reads as ground even where the top voxel is a flower), but tracked separately.
GROUND_COVER = frozenset({
    "short_grass", "tallgrass", "grass", "tall_grass", "fern", "large_fern", "dead_bush",
    "poppy", "dandelion", "blue_orchid", "allium", "azure_bluet", "oxeye_daisy",
    "cornflower", "lily_of_the_valley", "wither_rose", "sunflower", "lilac", "rose_bush",
    "peony", "red_tulip", "orange_tulip", "white_tulip", "pink_tulip", "torchflower",
    "moss_carpet", "azalea", "flowering_azalea", "seagrass", "lily_pad", "sugar_cane",
    "wheat", "carrots", "potatoes", "beetroots", "cactus", "bamboo", "sweet_berry_bush",
    "pink_petals", "spore_blossom", "hanging_roots", "glow_lichen", "vine",
})

VEGETATION_SUFFIXES = ("_leaves", "_sapling", "_log", "_wood", "_mushroom")

# ── Shapes ───────────────────────────────────────────────────────────────────
# A "cube" is a block that fills its voxel, which is what makes it usable as wall or roof
# material. Everything below is something else -- a slope, a plate, a post, a fitting.
NON_CUBE_SUFFIXES = (
    "_stairs", "_slab", "_pane", "_fence", "_fence_gate", "_wall", "_door", "_trapdoor",
    "_button", "_pressure_plate", "_sign", "_wall_sign", "_hanging_sign", "_carpet", "_bed",
    "_torch", "_lantern", "_banner", "_wall_banner", "_ladder", "_chain", "_rail", "_bars",
    "_pot", "_head", "_skull", "_candle", "_shelf", "_table", "_pane",
)

NON_CUBE_NAMES = frozenset({
    "torch", "soul_torch", "redstone_torch", "lantern", "soul_lantern", "ladder", "chain",
    "vine", "flower_pot", "iron_bars", "glass_pane", "end_rod", "lever", "tripwire_hook",
    "chest", "trapped_chest", "ender_chest", "barrel", "furnace", "blast_furnace", "smoker",
    "crafting_table", "cartography_table", "fletching_table", "smithing_table", "loom",
    "stonecutter", "grindstone", "anvil", "chipped_anvil", "damaged_anvil", "enchanting_table",
    "brewing_stand", "cauldron", "water_cauldron", "lava_cauldron", "composter", "bell",
    "campfire", "soul_campfire", "scaffolding", "conduit", "beacon", "lectern", "jukebox",
    "note_block", "daylight_detector", "hopper", "dispenser", "dropper", "observer",
    "piston", "sticky_piston", "piston_head", "redstone_wire", "repeater", "comparator",
    "cake", "snow", "water", "lava", "fire", "soul_fire", "cobweb", "turtle_egg",
    "flower_pot", "decorated_pot", "lightning_rod", "amethyst_cluster", "sculk_vein",
    "item_frame", "glow_item_frame", "painting", "armor_stand", "spawner",
})

# ── Pre-flattening names ─────────────────────────────────────────────────────
# Seen in builds that came through the legacy MCEdit path. Verified present in
# `49__pmc-suburban-house`; the analyser reports any name it still cannot place so this table
# can grow from evidence rather than guesswork.
LEGACY_ALIASES = {
    "stonebrick": "stone_bricks",
    "smooth_brick": "stone_bricks",
    "planks": "oak_planks",
    "wood": "oak_planks",
    "log": "oak_log",
    "log2": "acacia_log",
    "leaves": "oak_leaves",
    "leaves2": "acacia_leaves",
    "stained_hardened_clay": "terracotta",
    "hardened_clay": "terracotta",
    "concretePowder": "white_concrete_powder",
    "concrete_powder": "white_concrete_powder",
    "stained_glass": "white_stained_glass",
    "stained_glass_pane": "white_stained_glass_pane",
    "wool": "white_wool",
    "double_stone_slab": "smooth_stone",
    "brick_block": "bricks",
    "quartz_ore": "nether_quartz_ore",
    "tallgrass": "short_grass",
    "grass": "short_grass",
    "yellow_flower": "dandelion",
    "red_flower": "poppy",
    "double_plant": "sunflower",
    "waterlily": "lily_pad",
    "fence": "oak_fence",
    "wooden_door": "oak_door",
    "trapdoor": "oak_trapdoor",
    "wooden_slab": "oak_slab",
    "wooden_button": "oak_button",
    "wooden_pressure_plate": "oak_pressure_plate",
    "noteblock": "note_block",
    "mob_spawner": "spawner",
    "snow_layer": "snow",
    "web": "cobweb",
    "iron_fence": "iron_bars",
    "thin_glass": "glass_pane",
}


# ── Stair / slab variants ────────────────────────────────────────────────────
# Minecraft's naming is irregular enough that this has to be a table. A material absent from
# it has no stair variant *in the game* -- concrete, wool and terracotta are the big ones.
_STAIRS = {
    "bricks": "brick_stairs",
    "stone_bricks": "stone_brick_stairs",
    "mossy_stone_bricks": "mossy_stone_brick_stairs",
    "stone": "stone_stairs",
    "smooth_stone": "stone_stairs",
    "cobblestone": "cobblestone_stairs",
    "mossy_cobblestone": "mossy_cobblestone_stairs",
    "andesite": "andesite_stairs",
    "polished_andesite": "polished_andesite_stairs",
    "diorite": "diorite_stairs",
    "polished_diorite": "polished_diorite_stairs",
    "granite": "granite_stairs",
    "polished_granite": "polished_granite_stairs",
    "deepslate_tiles": "deepslate_tile_stairs",
    "deepslate_bricks": "deepslate_brick_stairs",
    "polished_deepslate": "polished_deepslate_stairs",
    "cobbled_deepslate": "cobbled_deepslate_stairs",
    "blackstone": "blackstone_stairs",
    "polished_blackstone": "polished_blackstone_stairs",
    "polished_blackstone_bricks": "polished_blackstone_brick_stairs",
    "sandstone": "sandstone_stairs",
    "smooth_sandstone": "smooth_sandstone_stairs",
    "cut_sandstone": "sandstone_stairs",
    "red_sandstone": "red_sandstone_stairs",
    "smooth_red_sandstone": "smooth_red_sandstone_stairs",
    "quartz_block": "quartz_stairs",
    "quartz_pillar": "quartz_stairs",
    "chiseled_quartz_block": "quartz_stairs",
    "smooth_quartz": "smooth_quartz_stairs",
    "nether_bricks": "nether_brick_stairs",
    "red_nether_bricks": "red_nether_brick_stairs",
    "prismarine": "prismarine_stairs",
    "prismarine_bricks": "prismarine_brick_stairs",
    "dark_prismarine": "dark_prismarine_stairs",
    "purpur_block": "purpur_stairs",
    "end_stone_bricks": "end_stone_brick_stairs",
    "tuff": "tuff_stairs",
    "polished_tuff": "polished_tuff_stairs",
    "tuff_bricks": "tuff_brick_stairs",
    "mud_bricks": "mud_brick_stairs",
    "copper_block": "cut_copper_stairs",
    "cut_copper": "cut_copper_stairs",
    "exposed_cut_copper": "exposed_cut_copper_stairs",
    "weathered_cut_copper": "weathered_cut_copper_stairs",
    "oxidized_cut_copper": "oxidized_cut_copper_stairs",
}
# Every wood works the same way, so generate rather than list.
_WOODS = ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove",
          "cherry", "bamboo", "crimson", "warped", "pale_oak")
for _w in _WOODS:
    _STAIRS[_w + "_planks"] = _w + "_stairs"
_STAIRS["bamboo_mosaic"] = "bamboo_mosaic_stairs"

# Slabs follow the stair naming exactly in every case above.
_SLABS = {base: stair[:-len("stairs")] + "slab" for base, stair in _STAIRS.items()}
# ...with two exceptions where the stair name is irregular but the slab is not.
_SLABS["stone"] = "stone_slab"
_SLABS["smooth_stone"] = "smooth_stone_slab"

# Reverse lookup, first base wins: several materials share one stair variant (quartz_block,
# quartz_pillar and chiseled_quartz_block all make quartz_stairs), so this picks a plausible
# representative rather than whichever happened to be defined last.
_STAIRS_TO_BASE: dict[str, str] = {}
for _base, _stair in _STAIRS.items():
    _STAIRS_TO_BASE.setdefault(_stair, _base)


def normalise(blockstate: str) -> str:
    """`"minecraft:oak_stairs[facing=east]"` -> `"oak_stairs"`, legacy spellings folded in.

    Strips the property suffix, then the namespace, then maps pre-flattening names onto their
    modern equivalents. Returns a bare base name with no namespace -- every other predicate in
    this module expects that form.
    """
    name = str(blockstate or "")
    cut = name.find("[")
    if cut >= 0:
        name = name[:cut]
    cut = name.find(":")
    if cut >= 0:
        name = name[cut + 1:]
    name = name.strip()
    return LEGACY_ALIASES.get(name, name)


def is_cube(base: str) -> bool:
    """True if the block fills its whole voxel, i.e. is usable as wall or roof material."""
    if not base or base in NON_CUBE_NAMES:
        return False
    if base.endswith(NON_CUBE_SUFFIXES):
        return False
    # `glass_pane` is caught by suffix, but plain `glass` is a cube -- and a cube we still do
    # not want counted as wall material, which `is_glass` handles separately.
    return True


def is_plant(base: str) -> bool:
    """Vegetation and ground cover: excluded from structure, tracked as landscaping."""
    return base in GROUND_COVER or base.endswith(VEGETATION_SUFFIXES)
