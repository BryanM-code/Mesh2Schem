"""A world's blocks, however they were stored, and the geometry the importer needs.

Bedrock and Java keep chunks in completely different containers -- a LevelDB of
bit-packed subchunks versus Anvil region files of NBT sections -- but once decoded they
are the same thing: 16x16x16 cubes of palette indices. This holds that, so everything
downstream (finding the built area, cutting a region out, drawing the map) is written once
and works for both.

A dropped `.schem` is folded into the same shape by `schem_source`, which is what lets a
schematic be cropped with the same picker a world is.
"""
from __future__ import annotations

from typing import Optional

BLOCKS_PER_CHUNK = 4096

# Blocks that mean "nothing here".
AIR_BLOCKS = frozenset({"minecraft:air", "minecraft:cave_air", "minecraft:void_air"})

# What counts as "not part of the build" when locating it automatically. A creative build
# sits on a flat plate of these; including them would blow past the schematic volume limit
# and bury the thing you actually want.
GROUND_BLOCKS = AIR_BLOCKS | frozenset({
    "minecraft:bedrock", "minecraft:dirt", "minecraft:grass_block",
    "minecraft:water", "minecraft:flowing_water", "minecraft:lava",
})

__all__ = ["VoxelWorld", "AIR_BLOCKS", "GROUND_BLOCKS", "BLOCKS_PER_CHUNK"]


class VoxelWorld:
    """Decoded chunks keyed by (chunk x, chunk z, section y).

    Each value is `(palette names, indices)` where `indices` is 4096 palette indices in
    `(x << 8) | (z << 4) | y` order -- the order both games happen to use inside a chunk.
    """

    __slots__ = ("subchunks", "level_name", "source_label")

    def __init__(self, subchunks: dict, level_name: str = "", source_label: str = "world"):
        self.subchunks = subchunks
        self.level_name = level_name
        self.source_label = source_label

    # ── Extent ────────────────────────────────────────────────────────────────

    def world_bounds(self) -> dict:
        """Block-space extent of every chunk present, built or not."""
        if not self.subchunks:
            raise ValueError("world has no block data")
        xs = [cx for cx, _cz, _sy in self.subchunks]
        zs = [cz for _cx, cz, _sy in self.subchunks]
        ys = [sy for _cx, _cz, sy in self.subchunks]
        return {
            "min_x": min(xs) * 16, "max_x": max(xs) * 16 + 15,
            "min_y": min(ys) * 16, "max_y": max(ys) * 16 + 15,
            "min_z": min(zs) * 16, "max_z": max(zs) * 16 + 15,
        }

    def croppable_extent(self) -> dict:
        """How much of this source is worth keeping so a crop can be redrawn later.

        For a world that is the built area: keeping every chunk of empty terrain would be
        enormous and useless. A schematic overrides this with its whole self -- it is
        already small, and trimming it would mean you could never get back to exactly the
        file you dropped.
        """
        return self.auto_bbox() or self.world_bounds()

    def _keep_mask(self, names, indices, skip):
        """Boolean mask over a chunk's 4096 blocks: True where the name isn't skipped.

        Tested per *palette entry* and then indexed, not per block -- a palette is tens of
        names where the chunk is 4096 blocks.
        """
        import numpy as np

        keep = np.array([name not in skip for name in names], dtype=bool)
        in_range = indices < len(names)
        safe = np.where(in_range, indices, 0)
        return keep[safe] & in_range

    def auto_bbox(self, skip=GROUND_BLOCKS) -> Optional[dict]:
        """Bounding box of the built area: everything that isn't air or ground plate.

        This is what makes a world importable at all. A flat creative world is mostly a
        plate of bedrock/dirt/grass whose full extent runs past the schematic volume limit;
        the part worth keeping is a fraction of it.
        """
        import numpy as np

        positions = np.arange(BLOCKS_PER_CHUNK)
        local = ((positions >> 8) & 15, positions & 15, (positions >> 4) & 15)

        lo = [None, None, None]
        hi = [None, None, None]
        count = 0
        for (cx, cz, sy), (names, indices) in self.subchunks.items():
            mask = self._keep_mask(names, indices, skip)
            if not mask.any():
                continue
            count += int(mask.sum())
            for axis, base in enumerate((cx * 16, sy * 16, cz * 16)):
                present = local[axis][mask]
                low, high = base + int(present.min()), base + int(present.max())
                if lo[axis] is None or low < lo[axis]:
                    lo[axis] = low
                if hi[axis] is None or high > hi[axis]:
                    hi[axis] = high
        if count == 0:
            return None
        return {"min_x": lo[0], "max_x": hi[0],
                "min_y": lo[1], "max_y": hi[1],
                "min_z": lo[2], "max_z": hi[2], "blocks": count}

    # ── Extraction ────────────────────────────────────────────────────────────

    def extract_region(self, bbox: dict):
        """Return (grid, names) for a region, in Sponge's YZX order.

        `grid` is a numpy array of palette indices shaped (height, length, width), which is
        exactly what the Sponge writer wants -- no transpose needed downstream.
        """
        import numpy as np

        x0, x1 = sorted((int(bbox["min_x"]), int(bbox["max_x"])))
        y0, y1 = sorted((int(bbox["min_y"]), int(bbox["max_y"])))
        z0, z1 = sorted((int(bbox["min_z"]), int(bbox["max_z"])))
        width, height, length = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1

        palette = {"minecraft:air": 0}
        names = ["minecraft:air"]
        grid = np.zeros((height, length, width), dtype=np.uint16)

        for (cx, cz, sy), (chunk_names, indices) in self.subchunks.items():
            if cx * 16 > x1 or cx * 16 + 15 < x0:
                continue
            if cz * 16 > z1 or cz * 16 + 15 < z0:
                continue
            if sy * 16 > y1 or sy * 16 + 15 < y0:
                continue

            mask = self._keep_mask(chunk_names, indices, AIR_BLOCKS)
            if not mask.any():
                continue
            positions = np.nonzero(mask)[0]
            xs = cx * 16 + ((positions >> 8) & 15)
            ys = sy * 16 + (positions & 15)
            zs = cz * 16 + ((positions >> 4) & 15)
            inside = ((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
                      & (zs >= z0) & (zs <= z1))
            if not inside.any():
                continue
            positions, xs, ys, zs = positions[inside], xs[inside], ys[inside], zs[inside]

            # Only the palette entries actually used by blocks inside the region are
            # carried over -- a chunk usually straddles the boundary, and adopting its
            # whole palette would pad the schematic with blocks the region doesn't hold.
            used = indices[positions]
            local = np.zeros(len(chunk_names), dtype=np.uint16)
            for index in np.unique(used):
                name = chunk_names[int(index)]
                slot = palette.get(name)
                if slot is None:
                    slot = len(names)
                    palette[name] = slot
                    names.append(name)
                local[index] = slot
            grid[ys - y0, zs - z0, xs - x0] = local[used]
        return grid, names

    # ── Plan view ─────────────────────────────────────────────────────────────

    def surface(self, bounds: dict):
        """Topmost non-air block per (x, z) column, for the map you pick a region on.

        Returns `(top_y array indexed [z][x], {(z, x): name})`. One pass over the chunks
        rather than a lookup per column: a world is sparse, and walking what exists is far
        cheaper than probing every coordinate in its bounding box.
        """
        import numpy as np

        width = bounds["max_x"] - bounds["min_x"] + 1
        length = bounds["max_z"] - bounds["min_z"] + 1
        top_y = np.full((length, width), -(1 << 30), dtype=np.int32)
        top_name = {}

        for (cx, cz, sy), (names, indices) in self.subchunks.items():
            mask = self._keep_mask(names, indices, AIR_BLOCKS)
            if not mask.any():
                continue
            positions = np.nonzero(mask)[0]
            xs = cx * 16 + ((positions >> 8) & 15) - bounds["min_x"]
            ys = sy * 16 + (positions & 15)
            zs = cz * 16 + ((positions >> 4) & 15) - bounds["min_z"]
            inside = (xs >= 0) & (xs < width) & (zs >= 0) & (zs < length)
            for position, x, y, z in zip(positions[inside], xs[inside], ys[inside], zs[inside]):
                if y > top_y[z, x]:
                    top_y[z, x] = y
                    top_name[(int(z), int(x))] = names[int(indices[position])]
        return top_y, top_name
