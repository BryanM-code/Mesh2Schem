"""Turn a region of a Bedrock world into a Sponge `.schem` the library can store.

`bedrock_world` gets the blocks out; this decides what they are called in Java's
vocabulary and writes the file. The writing itself is `schem_convert`'s -- the same
deterministic gzip the legacy `.schematic` path uses, so a region converted twice produces
identical bytes and the library's content-hash dedup still works.

**Block states are deliberately dropped.** Bedrock spells them differently from Java
(`weirdo_direction` for stair facing, `upside_down_bit` for slab halves), and translating
them faithfully is a table the size of `legacy_blocks`. Every block therefore lands in its
default state: shapes, materials and colours are right, but a staircase may face the wrong
way. That is what the thumbnails and TRELLIS need, and the conversion says so in its
warnings rather than pretending otherwise.
"""
from __future__ import annotations

import schem_convert
from bedrock_world import WorldError

# ── Bedrock -> Java block names ───────────────────────────────────────────────
# Modern Bedrock uses the same `minecraft:<name>` vocabulary as Java for almost everything,
# so this is an exceptions table rather than a full mapping. Every entry here was found by
# comparing a real world's palette against Java's names.
NAME_ALIASES = {
    # Straightforward renames.
    "minecraft:noteblock": "minecraft:note_block",
    "minecraft:darkoak_wall_sign": "minecraft:dark_oak_wall_sign",
    "minecraft:darkoak_standing_sign": "minecraft:dark_oak_sign",
    "minecraft:spruce_wall_sign": "minecraft:spruce_wall_sign",
    "minecraft:azalea_leaves_flowered": "minecraft:flowering_azalea_leaves",
    "minecraft:frame": "minecraft:item_frame",
    "minecraft:glow_frame": "minecraft:glow_item_frame",
    "minecraft:mob_spawner": "minecraft:spawner",
    "minecraft:invisible_bedrock": "minecraft:barrier",
    "minecraft:grass": "minecraft:short_grass",
    "minecraft:tallgrass": "minecraft:short_grass",
    "minecraft:waterlily": "minecraft:lily_pad",
    "minecraft:flowing_water": "minecraft:water",
    "minecraft:flowing_lava": "minecraft:lava",
    "minecraft:torchflower_crop": "minecraft:torchflower",

    # Bedrock keeps the pre-flattening "wooden" names for the oak variants.
    "minecraft:wooden_door": "minecraft:oak_door",
    "minecraft:wooden_button": "minecraft:oak_button",
    "minecraft:wooden_pressure_plate": "minecraft:oak_pressure_plate",
    "minecraft:trapdoor": "minecraft:oak_trapdoor",
    "minecraft:wooden_slab": "minecraft:oak_slab",
    "minecraft:oak_stairs": "minecraft:oak_stairs",

    # Colour lives in the block entity, which this conversion drops. A sensible default
    # keeps the shape and roughly the right tone rather than falling back to grey.
    "minecraft:bed": "minecraft:red_bed",
    "minecraft:standing_banner": "minecraft:white_banner",
    "minecraft:wall_banner": "minecraft:white_wall_banner",
    "minecraft:skull": "minecraft:skeleton_skull",
    "minecraft:flower_pot": "minecraft:flower_pot",
}

__all__ = ["NAME_ALIASES", "java_name", "region_to_schem"]


def java_name(bedrock: str) -> str:
    """The Java blockstate name for a Bedrock block name.

    Identity for the overwhelming majority; `NAME_ALIASES` covers the rest. Anything
    unknown is passed through unchanged, which is the useful failure: an unrecognised
    block still occupies space and still renders (as neutral grey) rather than vanishing.
    """
    name = str(bedrock or "").strip()
    if not name:
        return "minecraft:air"
    if ":" not in name:
        name = "minecraft:" + name
    return NAME_ALIASES.get(name, name)


def region_to_schem(grid, names, dest_path) -> dict:
    """Write a region as a Sponge v2 `.schem`. Returns dims plus conversion warnings.

    `grid` is (height, length, width) of palette indices -- Sponge's own YZX order, so it
    flattens straight into BlockData with no transpose.
    """
    import numpy as np
    import nbtlib
    from nbtlib import ByteArray, Compound, Int, IntArray, List, Short

    height, length, width = grid.shape
    if max(width, height, length) > schem_convert.MAX_AXIS:
        raise WorldError(
            "Selected region is {}x{}x{}; no side may exceed {}.".format(
                width, height, length, schem_convert.MAX_AXIS),
            code="axis_too_large", stage="region")
    volume = width * height * length
    if volume > schem_convert.MAX_VOLUME:
        raise WorldError(
            "Selected region is {}x{}x{} = {:,} blocks; the limit is {:,}. "
            "Choose a smaller area.".format(width, height, length, volume,
                                            schem_convert.MAX_VOLUME),
            code="volume_too_large", stage="region")

    # Bedrock names collapse onto Java ones, so several source indices can share a
    # destination slot. Build the Java palette first, then remap the grid through it in one
    # vectorised pass rather than per block.
    #
    # The palette is ordered by *name*, not by the order blocks were encountered, so the
    # output depends only on which blocks are where -- not on how the world happened to be
    # laid out in its database. `schem_convert` orders its legacy palette the same way and
    # for the same reason: the same region converted twice must produce identical bytes,
    # or the library's content-hash dedup would miss the repeat.
    states = sorted({java_name(name) for name in names} | {"minecraft:air"})
    palette = {state: slot for slot, state in enumerate(states)}
    remap = np.array([palette[java_name(name)] for name in names], dtype=np.uint16)

    indices = remap[grid].reshape(-1)

    warnings = ["Block states were not carried over: every block is in its default state, "
                "so stairs, slabs and logs may face the wrong way. Shapes and materials "
                "are unaffected."]
    renamed = sorted({n for n in names if java_name(n) != n})
    if renamed:
        shown = ", ".join(renamed[:5])
        more = "" if len(renamed) <= 5 else " (+{} more)".format(len(renamed) - 5)
        warnings.append("{} Bedrock block name(s) were translated to their Java "
                        "spelling: {}{}.".format(len(renamed), shown, more))

    schematic = Compound({
        "Version": Int(2),
        "DataVersion": Int(schem_convert.SPONGE_DATA_VERSION),
        "Metadata": Compound(),
        "Width": Short(width), "Height": Short(height), "Length": Short(length),
        "Offset": IntArray([0, 0, 0]),
        "PaletteMax": Int(len(palette)),
        "Palette": Compound({state: Int(slot) for state, slot in palette.items()}),
        "BlockData": ByteArray(
            schem_convert._encode_varints(indices, len(palette), np)),
        "BlockEntities": List[Compound]([]),
    })
    schem_convert._write_sponge(nbtlib.File(schematic), dest_path)

    return {"width": width, "height": height, "length": length,
            "volume": volume, "palette_size": len(palette), "warnings": warnings}
