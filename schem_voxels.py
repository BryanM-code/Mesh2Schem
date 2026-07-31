"""Read a Sponge .schem into a dense voxel grid, for rendering thumbnails.

`schem_convert` only ever needs to *count* blocks to validate a file; this module actually
decodes them. It reads Sponge only -- v1/v2 keep `Palette`/`BlockData` at the root, v3
nests them under `Schematic.Blocks` -- which is enough, because every file the library
stores on disk has already been through `ensure_sponge` on the way in.

Loading goes through `nbt_safe.load` like everything else: these files come off the
internet, and nothing gets handed to nbtlib until the strict tree walk has cleared it.

`BlockData` is a stream of unsigned LEB128 varints, one per block, in YZX order. Palettes
routinely run past 128 entries (a single ordinary house here uses 136), so multi-byte
varints are the normal case, not an edge case -- and the decode is vectorised rather than
looped, since a build can hold millions of blocks.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

import nbt_safe
from nbt_safe import NbtError

# Everything that reads as empty space. Anything else is a solid voxel.
AIR_NAMES = frozenset({"minecraft:air", "minecraft:cave_air", "minecraft:void_air"})

__all__ = ["VoxelGrid", "AIR_NAMES", "load_voxels"]


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


def load_voxels(path) -> Optional[VoxelGrid]:
    """Decode a Sponge .schem at `path`, or return None if it cannot be read.

    Returns None rather than raising: the only caller is thumbnail rendering, where an
    unreadable file means "no picture", not an error worth failing a request over. Files
    that reach here have already been validated once on the way into the library.
    """
    try:
        root = nbt_safe.load(path)
    except (NbtError, OSError):
        return None
    if "" in root:  # some writers wrap the payload in an explicitly-unnamed compound
        root = root[""]

    try:
        palette, block_data, width, height, length = _blocks_source(root)
        volume = width * height * length
        if volume <= 0:
            return None

        raw = np.asarray(block_data).view(np.uint8)
        indices = _decode_varints(raw, volume)

        names = [""] * (max(int(v) for v in palette.values()) + 1)
        for state, index in palette.items():
            names[int(index)] = str(state)
        if int(indices.max()) >= len(names):
            return None  # index past the end of the palette: not a file we can trust

        grid = indices.astype(np.uint16).reshape(height, length, width)
    except (ValueError, TypeError, KeyError, AttributeError, MemoryError):
        return None
    return VoxelGrid(grid, names)
