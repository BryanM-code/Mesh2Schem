"""A palette-indexed voxel grid (`VoxelGrid`), and decoding a Sponge `.schem`'s blocks.

`BlockData` is a stream of unsigned LEB128 varints, one per block, in YZX order. Palettes
routinely run past 128 entries, so multi-byte varints are the normal case, and the decode is
vectorised, since a build can hold millions of blocks. v1 and v2 keep `Palette` and
`BlockData` at the root; v3 nests them under `Schematic.Blocks` (`_blocks_source`).
"""
from __future__ import annotations

import numpy as np

# Everything that reads as empty space. Anything else is a solid voxel.
AIR_NAMES = frozenset({"minecraft:air", "minecraft:cave_air", "minecraft:void_air"})


class VoxelGrid:
    """A decoded schematic: palette indices in YZX order, plus the palette's names.

    `grid` is uint16 with shape (height, length, width) -- the axis order Sponge itself
    stores, kept as-is so no transpose is needed to index it. `names[i]` is the full
    blockstate string for palette index i, e.g. "minecraft:oak_stairs[facing=east]".
    """

    __slots__ = ("grid", "names", "width", "height", "length")

    def __init__(self, grid, names: list[str]):
        self.grid = grid
        self.names = names
        self.height, self.length, self.width = grid.shape

    @property
    def dims(self) -> tuple[int, int, int]:
        return self.width, self.height, self.length

    def solid_mask(self):
        """Boolean array, True wherever the voxel is not one of the air blocks."""
        air = np.zeros(len(self.names), dtype=bool)
        for index, name in enumerate(self.names):
            if base_name(name) in AIR_NAMES:
                air[index] = True
        return ~air[self.grid]


def base_name(blockstate: str) -> str:
    """Strip the [property=value,...] suffix: "minecraft:oak_stairs[facing=east]" -> base."""
    cut = blockstate.find("[")
    return blockstate if cut < 0 else blockstate[:cut]


def _blocks_source(root):
    """Return (palette_compound, block_data_tag, width, height, length) for any Sponge version.

    v3 moved the payload into a nested `Schematic.Blocks` compound and renamed BlockData to
    Data; v1/v2 keep both at the root. `schem_convert._identify` makes the same distinction
    when it validates the file -- this is the read-side twin of that branch.
    """
    if "Schematic" in root and "Width" not in root:
        root = root["Schematic"]
        blocks = root.get("Blocks")
        if not isinstance(blocks, dict):
            raise ValueError("Sponge v3 schematic has no Blocks compound.")
        palette, data = blocks.get("Palette"), blocks.get("Data")
    else:
        palette, data = root.get("Palette"), root.get("BlockData")

    if not isinstance(palette, dict) or not palette or data is None:
        raise ValueError("Schematic has no usable Palette/BlockData.")

    dims = tuple(int(root[k]) & 0xFFFF for k in ("Width", "Height", "Length"))
    return (palette, data) + dims


def _decode_varints(raw, expected: int):
    """Decode a packed LEB128 varint stream into one integer per block.

    Vectorised: a terminator byte (high bit clear) ends a value, so the terminator flags
    shifted by one mark where each value *starts*, a cumulative sum labels every byte with
    the value it belongs to, and a scatter-add reassembles the 7-bit groups in place. This
    beats a Python loop by orders of magnitude on the millions of blocks a large build has.
    """
    if len(raw) == 0:
        raise ValueError("Schematic block data is empty.")

    # Fast path: a palette of 128 or fewer states encodes every block as a single byte, so
    # the stream already *is* the value array. This is the common case even for very large
    # builds, and it skips several block-sized int64 temporaries the general decode needs.
    if len(raw) == expected and not (raw & 0x80).any():
        return raw

    starts = np.empty(len(raw), dtype=bool)
    starts[0] = True
    starts[1:] = raw[:-1] < 0x80             # a value starts right after a terminator
    group = np.cumsum(starts) - 1            # which value each byte belongs to
    within = np.arange(len(raw), dtype=np.int64) - np.flatnonzero(starts)[group]

    values = np.zeros(int(group[-1]) + 1, dtype=np.int64)
    np.add.at(values, group, (raw & 0x7F).astype(np.int64) << (7 * within))

    if len(values) != expected:
        raise ValueError(
            "Schematic block data does not match its dimensions "
            "({:,} blocks encoded, {:,} expected).".format(len(values), expected))
    return values
