"""Write a palette grid as a Sponge v2 `.schem`.

The palette is sorted by name and the gzip is written with `mtime=0`, so the same grid
always gives the same bytes. Block states are checked against the target version's block
table before writing (`voxel.vocab.validate_state`), and `Metadata` can carry provenance.
"""
from __future__ import annotations

import json

from minecraft import schem_convert


MAX_AXIS = schem_convert.MAX_AXIS
MAX_VOLUME = schem_convert.MAX_VOLUME

AIR = "minecraft:air"


class WriteError(ValueError):
    """The grid cannot be written as a valid schematic."""


def write_schem(grid, names, dest_path, metadata: dict | None = None,
                data_version: int | None = None, validate: bool | None = None) -> dict:
    """Write `grid` (height, length, width) of palette indices to a Sponge v2 `.schem`.

    `data_version` defaults to `schem_convert.SPONGE_DATA_VERSION` (1.13.2). The model's
    predictions are written with `voxel.vocab.TARGET_DATA_VERSION` (4671, Minecraft 1.21.11).

    `validate` checks every placed blockstate against that version's block table
    (`voxel.vocab.validate_state`) before writing, and raises `WriteError` naming the bad
    states. It defaults to on when `data_version` is the target version, off otherwise (there
    is no table for other versions).

    `names[i]` is the full blockstate for index i, properties included. Indices that no voxel
    uses are dropped and the palette re-densified: the importer requires indices dense from
    zero.

    Returns dims, volume and palette size.
    """
    import numpy as np
    import nbtlib
    from nbtlib import ByteArray, Compound, Int, IntArray, List, Short, String

    if grid.ndim != 3:
        raise WriteError("Grid must be 3-D (height, length, width); got {}.".format(grid.shape))
    height, length, width = (int(v) for v in grid.shape)
    if min(width, height, length) <= 0:
        raise WriteError("Grid has a zero-length axis: {}x{}x{}.".format(width, height, length))
    if max(width, height, length) > MAX_AXIS:
        raise WriteError(
            "Grid is {}x{}x{}; no side may exceed {}.".format(width, height, length, MAX_AXIS))
    volume = width * height * length
    if volume > MAX_VOLUME:
        raise WriteError(
            "Grid is {}x{}x{} = {:,} blocks; the limit is {:,}.".format(
                width, height, length, volume, MAX_VOLUME))

    flat = np.asarray(grid).reshape(-1)
    if len(names) == 0:
        raise WriteError("Palette is empty.")
    if int(flat.max()) >= len(names):
        raise WriteError("Grid references palette index {} but only {} names were given."
                         .format(int(flat.max()), len(names)))

    # Only the states actually placed reach the file, plus air -- which must always exist,
    # because an all-solid build still needs a slot for the importer's air checks and because
    # a palette without it reads oddly to every other tool.
    used = sorted({str(names[i]) for i in np.unique(flat)} | {AIR})
    _check_states(used, data_version, validate)
    palette = {state: slot for slot, state in enumerate(used)}
    remap = np.zeros(len(names), dtype=np.uint16)
    for i, name in enumerate(names):
        remap[i] = palette.get(str(name), palette[AIR])
    indices = remap[flat]

    schematic = Compound({
        "Version": Int(2),
        "DataVersion": Int(schem_convert.SPONGE_DATA_VERSION if data_version is None
                           else int(data_version)),
        "Metadata": _metadata(metadata, Compound, String, Int),
        "Width": Short(width), "Height": Short(height), "Length": Short(length),
        "Offset": IntArray([0, 0, 0]),
        "PaletteMax": Int(len(palette)),
        "Palette": Compound({state: Int(slot) for state, slot in palette.items()}),
        "BlockData": ByteArray(schem_convert._encode_varints(indices, len(palette), np)),
        "BlockEntities": List[Compound]([]),
    })
    schem_convert._write_sponge(nbtlib.File(schematic), dest_path)

    return {"width": width, "height": height, "length": length, "volume": volume,
            "palette_size": len(palette)}


def _check_states(states, data_version, validate) -> None:
    """Raise WriteError if any state is not a real block of the version being written."""
    from voxel.vocab import TARGET_DATA_VERSION as target
    if not validate and (validate is False or data_version != target):
        return
    from voxel.vocab import validate_state
    version = target if data_version is None else int(data_version)
    bad = []
    for state in states:
        try:
            validate_state(state, version)
        except ValueError as exc:
            bad.append(str(exc))
    if bad:
        raise WriteError("{} blockstate(s) are not valid for DataVersion {}: {}".format(
            len(bad), version, "; ".join(bad[:5])))


def _metadata(metadata, Compound, String, Int):
    """Provenance tags, kept to scalars so any NBT reader can display them."""
    if not metadata:
        return Compound()
    out = {}
    for key, value in metadata.items():
        if isinstance(value, bool):
            out[str(key)] = String("true" if value else "false")
        elif isinstance(value, int):
            out[str(key)] = Int(int(value))
        elif isinstance(value, (dict, list, tuple)):
            out[str(key)] = String(json.dumps(value, sort_keys=True, default=str))
        else:
            out[str(key)] = String(str(value))
    return Compound(out)
