"""Present a Sponge `.schem` as a `VoxelWorld`, so it can be cropped like a world.

A schematic is already just a build, but it is often a build plus a lot of extra: the
ground it stood on, a neighbour's fence, empty air where the selection was drawn loosely.
Turning it into the same container a world reads into means the same map-and-drag picker
works on it, and the crop is one code path rather than two.

The conversion is cheap: a schematic is small by definition, and re-chunking it is a
handful of array writes.
"""
from __future__ import annotations

from bedrock_world import WorldError
from voxel_world import VoxelWorld

__all__ = ["SchematicSource", "load_schem_source"]


class SchematicSource(VoxelWorld):
    """A schematic, chunked up so the region picker can work on it."""

    def __init__(self, subchunks: dict, level_name: str = "", dims=(0, 0, 0)):
        super().__init__(subchunks, level_name, source_label="schematic")
        self.dims = dims

    def croppable_extent(self) -> dict:
        """The whole schematic, not its trimmed contents.

        A world keeps only its built area, but a schematic is small and *is* the thing you
        dropped -- trimming it would make "put it back how it was" impossible.
        """
        return self.world_bounds()

    def world_bounds(self) -> dict:
        """The schematic's own extent, which is exactly its dimensions.

        Overridden because the chunked form rounds up to 16-block boundaries; a schematic
        45 blocks wide would otherwise report 48 and let a crop run off the end of it.
        """
        width, height, length = self.dims
        return {"min_x": 0, "max_x": width - 1,
                "min_y": 0, "max_y": height - 1,
                "min_z": 0, "max_z": length - 1}


def load_schem_source(path) -> SchematicSource:
    """Read a `.schem` into the shared chunked form."""
    import numpy as np
    import schem_voxels

    voxels = schem_voxels.load_voxels(path)
    if voxels is None:
        raise WorldError("Could not read that schematic.", code="unreadable_schem",
                         stage="schematic")

    grid = voxels.grid                       # (height, length, width), palette indices
    names = list(voxels.names)
    height, length, width = grid.shape

    # Chunk it exactly as a world is chunked, in the shared (x << 8) | (z << 4) | y order.
    positions = np.arange(4096)
    local_x = (positions >> 8) & 15
    local_y = positions & 15
    local_z = (positions >> 4) & 15

    subchunks = {}
    for cx in range((width + 15) // 16):
        for cz in range((length + 15) // 16):
            for sy in range((height + 15) // 16):
                xs = cx * 16 + local_x
                ys = sy * 16 + local_y
                zs = cz * 16 + local_z
                inside = (xs < width) & (ys < height) & (zs < length)
                if not inside.any():
                    continue
                indices = np.zeros(4096, dtype=np.uint16)
                indices[inside] = grid[ys[inside], zs[inside], xs[inside]]
                if not indices.any():
                    continue                 # nothing but palette slot 0 in this chunk
                subchunks[(cx, cz, sy)] = (names, indices)

    if not subchunks:
        raise WorldError("That schematic is empty.", code="empty_schem", stage="schematic")
    return SchematicSource(subchunks, dims=(width, height, length))
