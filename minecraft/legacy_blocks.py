"""Legacy (pre-1.13) numeric block id + data value -> modern blockstate strings.

Minecraft 1.13's "flattening" replaced numeric block ids with namespaced blockstate
strings. An MCEdit/WorldEdit `.schematic` stores the old `id:data` pairs, so converting
one into the Sponge `.schem` format is, at heart, running this table.

Two deliberate choices:

* **1.13 spellings.** `minecraft:sign`, not `minecraft:oak_sign`; `minecraft:grass_path`,
  not `minecraft:dirt_path`. Converted files record ``DataVersion`` 1631 (1.13.2), so
  Minecraft's own data fixers rename these again when a newer version loads the file.
  Writing 1.13 names is the only spelling that stays correct as the game moves on.

* **Only properties the data value actually encodes** -- facing, half, axis, colour,
  age, layers and so on. Properties a legacy file never stored (fence/wall/pane
  connections, stair ``shape``, redstone wire directions) are left off entirely so the
  palette entry falls back to the block's default state rather than to a guess.

Blocks whose real identity lived in a tile entity rather than the data value (banners,
flower pots, skulls, beds) resolve to a sensible placeholder here; `schem_convert`
refines those in a second pass using the TileEntities list.

Entry point: `resolve(key)`, where `key` comes from `key()` or `key_with_below()`.
"""
from __future__ import annotations

# ── Key encoding ──────────────────────────────────────────────────────────────
# Most blocks resolve from (id, data) alone. A few -- door and double-plant upper
# halves -- stored nothing useful in their own data value and have to be read together
# with the block underneath, so they get a wider key with a flag bit above it.
_EXT_FLAG = 1 << 20


def key(block_id: int, data: int) -> int:
    """Palette key for a block that resolves from its own id + data alone."""
    return (block_id << 4) | (data & 0x0F)


# Ids whose *upper* half (data bit 0x8) must be read together with the block below.
NEEDS_BLOCK_BELOW = frozenset({64, 71, 193, 194, 195, 196, 197, 175})


def describe(block_id: int, data: int) -> str:
    """Human-readable legacy id for warnings, e.g. "35:14"."""
    return "{}:{}".format(block_id, data)


# ── Blockstate formatting ─────────────────────────────────────────────────────

def blockstate(name: str, **props) -> str:
    """Build a namespaced blockstate string with properties in canonical order."""
    if not props:
        return "minecraft:" + name
    body = ",".join("{}={}".format(k, props[k]) for k in sorted(props))
    return "minecraft:{}[{}]".format(name, body)


_bs = blockstate


def _b(value) -> str:
    return "true" if value else "false"


# ── Shared vocabularies ───────────────────────────────────────────────────────

_WOODS = ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak")
_COLORS = ("white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
           "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black")

_FACING_6 = ("down", "up", "north", "south", "west", "east")      # EnumFacing.getFront
_FACING_H = ("south", "west", "north", "east")                    # EnumFacing.getHorizontal
_AXES = ("y", "x", "z")

_RAIL_SHAPES = ("north_south", "east_west", "ascending_east", "ascending_west",
                "ascending_north", "ascending_south",
                "south_east", "south_west", "north_west", "north_east")

_DOUBLE_PLANTS = ("sunflower", "lilac", "tall_grass", "large_fern", "rose_bush", "peony")

_DOORS = {64: "oak_door", 71: "iron_door", 193: "spruce_door", 194: "birch_door",
          195: "jungle_door", 196: "acacia_door", 197: "dark_oak_door"}


def _pick(names, d):
    """Legacy variant lookup; out-of-range metadata falls back to variant 0, as MC did."""
    return names[d] if 0 <= d < len(names) else names[0]


def _facing6(d):
    return _pick(_FACING_6, d & 7)


def _facingh(d):
    return _FACING_H[d & 3]


def _wall_facing(d):
    """Chests, furnaces, ladders, wall signs: data 2-5 only."""
    return _FACING_6[d & 7] if 2 <= (d & 7) <= 5 else "north"


# ── Per-block resolvers ───────────────────────────────────────────────────────

def _const(name):
    return lambda d: _bs(name)


def _variants(*names):
    return lambda d: _bs(_pick(names, d))


def _suffixed(prefix_list, suffix):
    return lambda d: _bs(_pick(prefix_list, d) + suffix)


def _stairs(name):
    def resolve_stairs(d):
        return _bs(name, facing=("east", "west", "south", "north")[d & 3],
                   half="top" if d & 4 else "bottom")
    return resolve_stairs


def _slab(*variants):
    def resolve_slab(d):
        return _bs(_pick(variants, d & 7), type="top" if d & 8 else "bottom")
    return resolve_slab


def _log(*species):
    def resolve_log(d):
        wood = _pick(species, d & 3)
        orientation = (d >> 2) & 3
        if orientation == 3:
            return _bs(wood + "_wood")           # bark on all six sides
        return _bs(wood + "_log", axis=_AXES[orientation])
    return resolve_log


def _leaves(*species):
    def resolve_leaves(d):
        return _bs(_pick(species, d & 3) + "_leaves", persistent=_b(d & 4))
    return resolve_leaves


def _pillar(name):
    return lambda d: _bs(name, axis=_pick(_AXES, (d >> 2) & 3))


def _rail(name, powered_bit=True):
    def resolve_rail(d):
        if not powered_bit:
            return _bs(name, shape=_pick(_RAIL_SHAPES, d))
        return _bs(name, shape=_pick(_RAIL_SHAPES[:6], d & 7), powered=_b(d & 8))
    return resolve_rail


def _piston(name):
    return lambda d: _bs(name, facing=_facing6(d), extended=_b(d & 8))


def _door_lower(name):
    """Only the lower half; the upper half needs the block below (see resolve_ext)."""
    def resolve_door(d):
        if d & 8:  # upper half without context -- resolve_ext normally handles these
            return _bs(name, half="upper", hinge="right" if d & 1 else "left",
                       powered=_b(d & 2))
        return _bs(name, half="lower", facing=("east", "south", "west", "north")[d & 3],
                   open=_b(d & 4))
    return resolve_door


def _trapdoor(name):
    def resolve_trapdoor(d):
        return _bs(name, facing=("north", "south", "west", "east")[d & 3],
                   half="top" if d & 8 else "bottom", open=_b(d & 4))
    return resolve_trapdoor


def _fence_gate(name):
    return lambda d: _bs(name, facing=_facingh(d), open=_b(d & 4), powered=_b(d & 8))


def _torch(standing, wall, **extra):
    def resolve_torch(d):
        if 1 <= (d & 7) <= 4:
            return _bs(wall, facing=("east", "west", "south", "north")[(d & 7) - 1], **extra)
        return _bs(standing, **extra)
    return resolve_torch


def _button(name):
    def resolve_button(d):
        face = d & 7
        if face == 0:
            return _bs(name, face="ceiling", facing="north", powered=_b(d & 8))
        if face == 5:
            return _bs(name, face="floor", facing="north", powered=_b(d & 8))
        return _bs(name, face="wall", facing=_pick(("east", "west", "south", "north"), face - 1),
                   powered=_b(d & 8))
    return resolve_button


def _shulker(color):
    return lambda d: _bs(color + "_shulker_box", facing=_facing6(d))


def _glazed(color):
    return lambda d: _bs(color + "_glazed_terracotta", facing=_facingh(d))


def _command(name):
    return lambda d: _bs(name, facing=_facing6(d), conditional=_b(d & 8))


def _repeater(powered):
    def resolve_repeater(d):
        return _bs("repeater", facing=_facingh(d), delay=((d >> 2) & 3) + 1,
                   powered=_b(powered))
    return resolve_repeater


def _comparator(powered):
    def resolve_comparator(d):
        return _bs("comparator", facing=_facingh(d),
                   mode="subtract" if d & 4 else "compare", powered=_b(powered))
    return resolve_comparator


# Mushroom blocks encode which of the six faces show cap texture rather than pores.
_MUSHROOM_FACES = {
    0: (),
    1: ("up", "west", "north"), 2: ("up", "north"), 3: ("up", "north", "east"),
    4: ("up", "west"), 5: ("up",), 6: ("up", "east"),
    7: ("up", "west", "south"), 8: ("up", "south"), 9: ("up", "east", "south"),
    10: ("north", "south", "east", "west"),                       # stem, sides only
    14: ("up", "down", "north", "south", "east", "west"),          # all cap
    15: ("up", "down", "north", "south", "east", "west"),          # all stem
}
_SIDES = ("down", "east", "north", "south", "up", "west")


def _mushroom(cap_name):
    def resolve_mushroom(d):
        faces = _MUSHROOM_FACES.get(d, _MUSHROOM_FACES[14])
        name = "mushroom_stem" if d in (10, 15) else cap_name
        return _bs(name, **{side: _b(side in faces) for side in _SIDES})
    return resolve_mushroom


def _vine(d):
    return _bs("vine", south=_b(d & 1), west=_b(d & 2), north=_b(d & 4), east=_b(d & 8))


def _lever(d):
    # BlockLever.EnumOrientation ordinals: DOWN_X, EAST, WEST, SOUTH, NORTH, UP_Z, UP_X, DOWN_Z
    face, facing = (("ceiling", "west"), ("wall", "east"), ("wall", "west"),
                    ("wall", "south"), ("wall", "north"), ("floor", "north"),
                    ("floor", "west"), ("ceiling", "north"))[d & 7]
    return _bs("lever", face=face, facing=facing, powered=_b(d & 8))


def _bed(d):
    # Colour lived in the tile entity; schem_convert refines it in a second pass.
    return _bs("red_bed", facing=("south", "west", "north", "east")[d & 3],
               part="head" if d & 8 else "foot", occupied=_b(d & 4))


def _pumpkin(carved):
    def resolve_pumpkin(d):
        if (d & 7) > 3:
            return _bs("pumpkin")           # 1.13 gained a genuinely faceless pumpkin
        return _bs(carved, facing=_facingh(d))
    return resolve_pumpkin


def _hopper(d):
    facing = _FACING_6[d & 7] if (d & 7) in (0, 2, 3, 4, 5) else "down"
    return _bs("hopper", facing=facing, enabled=_b(not d & 8))


# ── The table ─────────────────────────────────────────────────────────────────

_RESOLVERS = {
    0: _const("air"),
    1: _variants("stone", "granite", "polished_granite", "diorite", "polished_diorite",
                 "andesite", "polished_andesite"),
    2: _const("grass_block"),
    3: _variants("dirt", "coarse_dirt", "podzol"),
    4: _const("cobblestone"),
    5: _suffixed(_WOODS, "_planks"),
    6: lambda d: _bs(_pick(_WOODS, d & 7) + "_sapling", stage=(d >> 3) & 1),
    7: _const("bedrock"),
    8: lambda d: _bs("water", level=d),
    9: lambda d: _bs("water", level=d),
    10: lambda d: _bs("lava", level=d),
    11: lambda d: _bs("lava", level=d),
    12: _variants("sand", "red_sand"),
    13: _const("gravel"),
    14: _const("gold_ore"),
    15: _const("iron_ore"),
    16: _const("coal_ore"),
    17: _log("oak", "spruce", "birch", "jungle"),
    18: _leaves("oak", "spruce", "birch", "jungle"),
    19: _variants("sponge", "wet_sponge"),
    20: _const("glass"),
    21: _const("lapis_ore"),
    22: _const("lapis_block"),
    23: lambda d: _bs("dispenser", facing=_facing6(d), triggered=_b(d & 8)),
    24: _variants("sandstone", "chiseled_sandstone", "cut_sandstone"),
    25: _const("note_block"),
    26: _bed,
    27: _rail("powered_rail"),
    28: _rail("detector_rail"),
    29: _piston("sticky_piston"),
    30: _const("cobweb"),
    31: _variants("dead_bush", "grass", "fern"),
    32: _const("dead_bush"),
    33: _piston("piston"),
    34: lambda d: _bs("piston_head", facing=_facing6(d),
                      type="sticky" if d & 8 else "normal"),
    35: _suffixed(_COLORS, "_wool"),
    36: lambda d: _bs("moving_piston", facing=_facing6(d),
                      type="sticky" if d & 8 else "normal"),
    37: _const("dandelion"),
    38: _variants("poppy", "blue_orchid", "allium", "azure_bluet", "red_tulip",
                  "orange_tulip", "white_tulip", "pink_tulip", "oxeye_daisy"),
    39: _const("brown_mushroom"),
    40: _const("red_mushroom"),
    41: _const("gold_block"),
    42: _const("iron_block"),
    43: _variants("smooth_stone", "sandstone", "oak_planks", "cobblestone", "bricks",
                  "stone_bricks", "nether_bricks", "quartz_block",
                  "smooth_sandstone", "smooth_quartz"),
    44: _slab("stone_slab", "sandstone_slab", "petrified_oak_slab", "cobblestone_slab",
              "brick_slab", "stone_brick_slab", "nether_brick_slab", "quartz_slab"),
    45: _const("bricks"),
    46: lambda d: _bs("tnt", unstable=_b(d & 1)),
    47: _const("bookshelf"),
    48: _const("mossy_cobblestone"),
    49: _const("obsidian"),
    50: _torch("torch", "wall_torch"),
    51: lambda d: _bs("fire", age=d),
    52: _const("spawner"),
    53: _stairs("oak_stairs"),
    54: lambda d: _bs("chest", facing=_wall_facing(d)),
    55: lambda d: _bs("redstone_wire", power=d),
    56: _const("diamond_ore"),
    57: _const("diamond_block"),
    58: _const("crafting_table"),
    59: lambda d: _bs("wheat", age=d & 7),
    60: lambda d: _bs("farmland", moisture=d & 7),
    61: lambda d: _bs("furnace", facing=_wall_facing(d)),
    62: lambda d: _bs("furnace", facing=_wall_facing(d), lit="true"),
    63: lambda d: _bs("sign", rotation=d),
    64: _door_lower("oak_door"),
    65: lambda d: _bs("ladder", facing=_wall_facing(d)),
    66: _rail("rail", powered_bit=False),
    67: _stairs("cobblestone_stairs"),
    68: lambda d: _bs("wall_sign", facing=_wall_facing(d)),
    69: _lever,
    70: lambda d: _bs("stone_pressure_plate", powered=_b(d & 1)),
    71: _door_lower("iron_door"),
    72: lambda d: _bs("oak_pressure_plate", powered=_b(d & 1)),
    73: _const("redstone_ore"),
    74: lambda d: _bs("redstone_ore", lit="true"),
    75: _torch("redstone_torch", "redstone_wall_torch", lit="false"),
    76: _torch("redstone_torch", "redstone_wall_torch", lit="true"),
    77: _button("stone_button"),
    78: lambda d: _bs("snow", layers=(d & 7) + 1),
    79: _const("ice"),
    80: _const("snow_block"),
    81: lambda d: _bs("cactus", age=d),
    82: _const("clay"),
    83: lambda d: _bs("sugar_cane", age=d),
    84: lambda d: _bs("jukebox", has_record=_b(d & 1)),
    85: _const("oak_fence"),
    86: _pumpkin("carved_pumpkin"),
    87: _const("netherrack"),
    88: _const("soul_sand"),
    89: _const("glowstone"),
    90: lambda d: _bs("nether_portal", axis="z" if (d & 3) == 2 else "x"),
    91: _pumpkin("jack_o_lantern"),
    92: lambda d: _bs("cake", bites=d & 7),
    93: _repeater(False),
    94: _repeater(True),
    95: _suffixed(_COLORS, "_stained_glass"),
    96: _trapdoor("oak_trapdoor"),
    97: _variants("infested_stone", "infested_cobblestone", "infested_stone_bricks",
                  "infested_mossy_stone_bricks", "infested_cracked_stone_bricks",
                  "infested_chiseled_stone_bricks"),
    98: _variants("stone_bricks", "mossy_stone_bricks", "cracked_stone_bricks",
                  "chiseled_stone_bricks"),
    99: _mushroom("brown_mushroom_block"),
    100: _mushroom("red_mushroom_block"),
    101: _const("iron_bars"),
    102: _const("glass_pane"),
    103: _const("melon"),
    104: lambda d: _bs("pumpkin_stem", age=d & 7),
    105: lambda d: _bs("melon_stem", age=d & 7),
    106: _vine,
    107: _fence_gate("oak_fence_gate"),
    108: _stairs("brick_stairs"),
    109: _stairs("stone_brick_stairs"),
    110: _const("mycelium"),
    111: _const("lily_pad"),
    112: _const("nether_bricks"),
    113: _const("nether_brick_fence"),
    114: _stairs("nether_brick_stairs"),
    115: lambda d: _bs("nether_wart", age=d & 3),
    116: _const("enchanting_table"),
    117: lambda d: _bs("brewing_stand", has_bottle_0=_b(d & 1), has_bottle_1=_b(d & 2),
                       has_bottle_2=_b(d & 4)),
    118: lambda d: _bs("cauldron", level=d & 3),
    119: _const("end_portal"),
    120: lambda d: _bs("end_portal_frame", facing=_facingh(d), eye=_b(d & 4)),
    121: _const("end_stone"),
    122: _const("dragon_egg"),
    123: _const("redstone_lamp"),
    124: lambda d: _bs("redstone_lamp", lit="true"),
    125: _suffixed(_WOODS, "_planks"),
    126: lambda d: _bs(_pick(_WOODS, d & 7) + "_slab", type="top" if d & 8 else "bottom"),
    127: lambda d: _bs("cocoa", facing=_facingh(d), age=(d >> 2) & 3),
    128: _stairs("sandstone_stairs"),
    129: _const("emerald_ore"),
    130: lambda d: _bs("ender_chest", facing=_wall_facing(d)),
    131: lambda d: _bs("tripwire_hook", facing=_facingh(d), attached=_b(d & 4),
                       powered=_b(d & 8)),
    132: lambda d: _bs("tripwire", powered=_b(d & 1), attached=_b(d & 4),
                       disarmed=_b(d & 8)),
    133: _const("emerald_block"),
    134: _stairs("spruce_stairs"),
    135: _stairs("birch_stairs"),
    136: _stairs("jungle_stairs"),
    137: _command("command_block"),
    138: _const("beacon"),
    139: _variants("cobblestone_wall", "mossy_cobblestone_wall"),
    140: _const("flower_pot"),          # contents come from the tile entity (potted_state)
    141: lambda d: _bs("carrots", age=d & 7),
    142: lambda d: _bs("potatoes", age=d & 7),
    143: _button("oak_button"),
    144: lambda d: skull_state(d),      # type + rotation come from the tile entity
    145: lambda d: _bs(_pick(("anvil", "chipped_anvil", "damaged_anvil"), (d >> 2) & 3),
                       facing=_facingh(d)),
    146: lambda d: _bs("trapped_chest", facing=_wall_facing(d)),
    147: lambda d: _bs("light_weighted_pressure_plate", power=d),
    148: lambda d: _bs("heavy_weighted_pressure_plate", power=d),
    149: _comparator(False),
    150: _comparator(True),
    151: lambda d: _bs("daylight_detector", power=d),
    152: _const("redstone_block"),
    153: _const("nether_quartz_ore"),
    154: _hopper,
    155: lambda d: _bs("quartz_pillar", axis=_AXES[d - 2]) if 2 <= d <= 4
         else _bs(_pick(("quartz_block", "chiseled_quartz_block"), d)),
    156: _stairs("quartz_stairs"),
    157: _rail("activator_rail"),
    158: lambda d: _bs("dropper", facing=_facing6(d), triggered=_b(d & 8)),
    159: _suffixed(_COLORS, "_terracotta"),
    160: _suffixed(_COLORS, "_stained_glass_pane"),
    161: _leaves("acacia", "dark_oak"),
    162: _log("acacia", "dark_oak"),
    163: _stairs("acacia_stairs"),
    164: _stairs("dark_oak_stairs"),
    165: _const("slime_block"),
    166: _const("barrier"),
    167: _trapdoor("iron_trapdoor"),
    168: _variants("prismarine", "prismarine_bricks", "dark_prismarine"),
    169: _const("sea_lantern"),
    170: _pillar("hay_block"),
    171: _suffixed(_COLORS, "_carpet"),
    172: _const("terracotta"),
    173: _const("coal_block"),
    174: _const("packed_ice"),
    175: lambda d: _bs(_pick(_DOUBLE_PLANTS, d & 7), half="upper" if d & 8 else "lower"),
    176: lambda d: _bs("white_banner", rotation=d),     # colour from the tile entity
    177: lambda d: _bs("white_wall_banner", facing=_wall_facing(d)),
    178: lambda d: _bs("daylight_detector", inverted="true", power=d),
    179: _variants("red_sandstone", "chiseled_red_sandstone", "cut_red_sandstone"),
    180: _stairs("red_sandstone_stairs"),
    181: lambda d: _bs("smooth_red_sandstone" if d & 8 else "red_sandstone"),
    182: _slab("red_sandstone_slab"),
    183: _fence_gate("spruce_fence_gate"),
    184: _fence_gate("birch_fence_gate"),
    185: _fence_gate("jungle_fence_gate"),
    186: _fence_gate("dark_oak_fence_gate"),
    187: _fence_gate("acacia_fence_gate"),
    188: _const("spruce_fence"),
    189: _const("birch_fence"),
    190: _const("jungle_fence"),
    191: _const("dark_oak_fence"),
    192: _const("acacia_fence"),
    193: _door_lower("spruce_door"),
    194: _door_lower("birch_door"),
    195: _door_lower("jungle_door"),
    196: _door_lower("acacia_door"),
    197: _door_lower("dark_oak_door"),
    198: lambda d: _bs("end_rod", facing=_facing6(d)),
    199: _const("chorus_plant"),
    200: lambda d: _bs("chorus_flower", age=d & 7),
    201: _const("purpur_block"),
    202: _pillar("purpur_pillar"),
    203: _stairs("purpur_stairs"),
    204: _const("purpur_block"),
    205: _slab("purpur_slab"),
    206: _const("end_stone_bricks"),
    207: lambda d: _bs("beetroots", age=d & 3),
    208: _const("grass_path"),
    209: _const("end_gateway"),
    210: _command("repeating_command_block"),
    211: _command("chain_command_block"),
    212: lambda d: _bs("frosted_ice", age=d & 3),
    213: _const("magma_block"),
    214: _const("nether_wart_block"),
    215: _const("red_nether_bricks"),
    216: _pillar("bone_block"),
    217: _const("structure_void"),
    218: lambda d: _bs("observer", facing=_facing6(d), powered=_b(d & 8)),
    251: _suffixed(_COLORS, "_concrete"),
    252: _suffixed(_COLORS, "_concrete_powder"),
    255: lambda d: _bs("structure_block", mode=_pick(("save", "load", "corner", "data"), d)),
}

# Shulker boxes (219-234) and glazed terracotta (235-250) are one colour per id.
for _i, _color in enumerate(_COLORS):
    _RESOLVERS[219 + _i] = _shulker(_color)
    _RESOLVERS[235 + _i] = _glazed(_color)
del _i, _color


# ── Blocks whose real identity lived in a tile entity ─────────────────────────
# Banners, beds, skulls and flower pots all render as one thing in the block array and
# store what they actually *are* in the matching TileEntity. `resolve` returns the
# neutral placeholder; the converter calls these once it has the tile entity in hand.

_SKULL_TYPES = ("skeleton", "wither_skeleton", "zombie", "player", "creeper", "dragon")

# Flower pot contents, keyed by the legacy item id in the tile entity.
_POTTED = {
    "sapling": ("potted_{}_sapling", _WOODS),
    "red_flower": ("potted_{}", ("poppy", "blue_orchid", "allium", "azure_bluet",
                                 "red_tulip", "orange_tulip", "white_tulip",
                                 "pink_tulip", "oxeye_daisy")),
    "tallgrass": ("potted_{}", ("dead_bush", "fern", "fern")),
    "yellow_flower": ("potted_dandelion", None),
    "red_mushroom": ("potted_red_mushroom", None),
    "brown_mushroom": ("potted_brown_mushroom", None),
    "cactus": ("potted_cactus", None),
    "deadbush": ("potted_dead_bush", None),
    "sapling_oak": ("potted_oak_sapling", None),
}


def skull_state(data: int, skull_type: int = 0, rotation: int = 0) -> str:
    """Skulls: floor-vs-wall from the data value, species and rotation from the entity."""
    kind = _pick(_SKULL_TYPES, skull_type)
    noun = "skull" if kind in ("skeleton", "wither_skeleton") else "head"
    face = data & 7
    if 2 <= face <= 5:
        return _bs("{}_wall_{}".format(kind, noun), facing=_FACING_6[face])
    return _bs("{}_{}".format(kind, noun), rotation=rotation & 0x0F)


def banner_state(data: int, base: int = 15, wall: bool = False) -> str:
    """Banners: the tile entity's `Base` is a dye index, the inverse of the wool order."""
    color = _COLORS[15 - base] if 0 <= base <= 15 else "white"
    if wall:
        return _bs(color + "_wall_banner", facing=_wall_facing(data))
    return _bs(color + "_banner", rotation=data & 0x0F)


def bed_state(data: int, color: int = 14) -> str:
    """Beds: 1.12 kept the dyed colour in the tile entity; 14 (red) is the default."""
    return _bs(_pick(_COLORS, color) + "_bed",
               facing=("south", "west", "north", "east")[data & 3],
               part="head" if data & 8 else "foot", occupied=_b(data & 4))


def potted_state(item, damage: int = 0):
    """Flower pots: returns None when the contents aren't recognised, so the caller keeps
    the plain empty pot rather than inventing a plant."""
    name = str(item or "").strip()
    if not name:
        return None
    name = name.split(":")[-1].lower()
    if name in ("air", "0", ""):
        return _bs("flower_pot")
    entry = _POTTED.get(name)
    if entry is None:
        return None
    template, variants = entry
    if variants is None:
        return _bs(template)
    return _bs(template.format(_pick(variants, damage)))


# ── Resolution ────────────────────────────────────────────────────────────────

def resolve(packed_key: int):
    """Blockstate string for a packed key, or None if the legacy id is unknown.

    Unknown ids are almost always modded blocks carried in a schematic's AddBlocks
    array; the caller decides what to substitute and reports the count.
    """
    if packed_key & _EXT_FLAG:
        block_id = (packed_key >> 8) & 0xFFF
        data = (packed_key >> 4) & 0x0F
        below = packed_key & 0x0F
        return _resolve_upper_half(block_id, data, below)

    block_id = packed_key >> 4
    resolver = _RESOLVERS.get(block_id)
    return resolver(packed_key & 0x0F) if resolver else None


def _resolve_upper_half(block_id: int, data: int, below_data: int):
    """Doors and double plants: the upper half only makes sense with the block below."""
    name = _DOORS.get(block_id)
    if name is not None:
        return _bs(name, half="upper",
                   hinge="right" if data & 1 else "left", powered=_b(data & 2),
                   facing=("east", "south", "west", "north")[below_data & 3],
                   open=_b(below_data & 4))
    if block_id == 175:
        return _bs(_pick(_DOUBLE_PLANTS, below_data & 7), half="upper")
    return resolve(key(block_id, data))
