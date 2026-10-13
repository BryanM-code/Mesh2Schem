"""Write a voxel grid out as a Sponge `.schem` the library will accept.

`mcworld_convert.region_to_schem` already does something very close to this, and the NBT
construction here is deliberately its twin -- same tags, same sorted palette, same
deterministic gzip through `schem_convert._write_sponge`. It is a separate function for three
reasons, all of which matter for generated builds specifically:

*   It does not run names through `mcworld_convert.java_name`. That is a *Bedrock* alias
    table; pointing it at a palette we chose ourselves is a coupling with no upside and one
    concrete downside -- it rewrites `minecraft:grass` to `minecraft:short_grass` and several
    `wooden_*` spellings, silently, on blocks the generator deliberately picked.
*   It does not claim block states were dropped. `region_to_schem` appends that warning
    unconditionally because Bedrock states cannot be translated; ours are authored with
    `facing` and `half` set on purpose, and that warning would end up in `meta.json`'s
    `source_note` misdescribing the dataset.
*   It stamps provenance into `Metadata`. `ensure_sponge` returns Sponge files untouched
    (`schem_convert.py:234`), so a Metadata compound written here survives import byte for
    byte -- which means every synthetic build in the library carries the seed and parameters
    that produced it, and can be found, filtered or ablated later.

The palette is sorted by name and the gzip is written with `mtime=0`, so the same grid always
produces the same bytes. That is not a nicety: the library dedups on content sha256, so a
generator that produced different bytes for the same house would quietly fill the library
with duplicates.
"""
from __future__ import annotations

import json

import schem_convert

__all__ = ["WriteError", "write_schem"]

MAX_AXIS = schem_convert.MAX_AXIS
MAX_VOLUME = schem_convert.MAX_VOLUME

AIR = "minecraft:air"


class WriteError(ValueError):
    """The grid cannot be written as a schematic the library would accept."""


def write_schem(grid, names, dest_path, metadata: dict | None = None,
                data_version: int | None = None, validate: bool | None = None) -> dict:
    """Write `grid` (height, length, width) of palette indices to a Sponge v2 `.schem`.

    `data_version` defaults to `schem_convert.SPONGE_DATA_VERSION` (1.13.2), right for the
    1.13 names the house generator uses. Pass a later version when the palette holds later
    blocks: the parked Objaverse dataset writes 3465 (1.20.1); training targets in the
    mcvox-v3 vocabulary write `mcvoxel.vocab.TARGET_DATA_VERSION` (4671, 1.21.11, D21).

    `validate` checks every placed blockstate against that version's block table
    (`mcvoxel.vocab.validate_state`) before writing, and raises `WriteError` naming the bad
    states. It defaults to on when `data_version` is the target version, off otherwise (the
    1.13 house generator has no table to check against).

    `names[i]` is the full blockstate for index i, properties included. Indices that no voxel
    uses are dropped and the palette re-densified: the importer requires indices dense from
    zero, and a generator that samples materials routinely leaves holes.

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
    target = None
    if validate is not False:
        try:                                     # mcvoxel ships with requirements-synth.txt
            from mcvoxel.vocab import TARGET_DATA_VERSION as target
        except ImportError:
            if validate:
                raise WriteError("Validation needs mcvoxel (requirements-synth.txt).")
            return
    if not validate and (validate is False or data_version != target):
        return
    from mcvoxel.vocab import validate_state
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
