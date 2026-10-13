"""Read any schematic in the dataset into one shape: a palette-indexed (X, Y, Z) grid.

    build = load_build(path)
    build.blocks        uint16 (X, Y, Z), Y up, +X east, +Z south (mcvoxel's convention)
    build.palette       modern blockstate strings; index 0 is always "minecraft:air"

Three formats, all decided by content, never by extension:

* **Sponge** `.schem` v1 / v2 (palette and varint `BlockData` at the root) and v3 (under
  `Schematic.Blocks`). Decoded with `schem_voxels`' varint reader.
* **Legacy MCEdit** `.schematic` (numeric id:data). Converted in memory with
  `schem_convert`'s own steps -- the same id:data table, door and plant halves, and the
  tile-entity refinements for banners, beds, skulls and pots -- without writing a file.
  The conversion's file entry points cap a build at 16M blocks; one dataset build is
  26.7M, so the steps are called directly. An id the table does not know is kept as a
  solid `legacy:unknown_<id>` block and counted, never turned into air (or into stone,
  which is what the converter substitutes for files it writes).
* **Litematica** `.litematic`, read here on nbtlib (litemapy is GPL; this repository is
  MIT). Regions are placed by `Position` into one grid; a negative `Size` means the region
  extends from `Position` towards smaller coordinates; `BlockStates` packs
  max(2, ceil(log2(palette size))) bits per block, and entries may span two longs.

Every file goes through `nbt_safe.load` first: these files come off the internet.

Palette strings are modernised so one block has one spelling: legacy aliases folded with
`block_roles.normalise` (keeping the properties), renamed blocks given their current name
(`sign` -> `oak_sign`, `grass_path` -> `dirt_path`, `chain` -> `iron_chain`, and
`stone_slab` -> `smooth_stone_slab` for files older than DataVersion 1901, when 1.14 gave
the old name to a new block), cave and void air folded into air, and properties sorted.
Non-minecraft namespaces are kept as they are.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import block_roles
import legacy_blocks
import nbt_safe
import schem_convert
import schem_voxels
from nbt_safe import FileRejected

AIR = "minecraft:air"
AIR_BASES = {"air", "cave_air", "void_air"}
STONE_SLAB_RENAMED = 1901          # 18w43a: 1.13's stone_slab became smooth_stone_slab
# Old names that no later version uses for anything else (chain: iron_chain since 1.21.9).
ALWAYS_RENAMED = {"sign": "oak_sign", "wall_sign": "oak_wall_sign", "grass_path": "dirt_path",
                  "chain": "iron_chain"}


class ReadError(RuntimeError):
    pass


@dataclass
class Build:
    blocks: np.ndarray                 # uint16 (X, Y, Z)
    palette: list[str]                 # modern blockstates; palette[0] == AIR
    data_version: int | None
    source_format: str                 # sponge_v1/v2/v3, mcedit, litematic_v<N>
    raw_non_air: int                   # non-air voxels as stored, before any renaming
    regions: int = 1
    unknown_legacy: dict[str, int] = field(default_factory=dict)   # "id:data" -> voxels
    non_vanilla: dict[str, int] = field(default_factory=dict)      # namespaced name -> voxels
    warnings: list[str] = field(default_factory=list)

    @property
    def dims(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.blocks.shape)

    def non_air(self) -> int:
        return int(np.count_nonzero(self.blocks))


# --------------------------------------------------------------------------- blockstates

def split_state(state: str) -> tuple[str, dict[str, str]]:
    """"minecraft:oak_stairs[facing=east,half=top]" -> ("minecraft:oak_stairs", {...})."""
    state = str(state).strip()
    cut = state.find("[")
    if cut < 0:
        return state, {}
    props = {}
    for part in state[cut + 1:state.rfind("]")].split(","):
        key, sep, value = part.partition("=")
        if sep:
            props[key.strip()] = value.strip()
    return state[:cut], props


def join_state(name: str, props: dict[str, str]) -> str:
    if not props:
        return name
    return "{}[{}]".format(name, ",".join(f"{k}={props[k]}" for k in sorted(props)))


def modern_state(raw: str, data_version: int | None) -> str:
    """One canonical spelling per block (see the module docstring)."""
    name, props = split_state(raw)
    namespace, sep, base = name.rpartition(":")
    if sep and namespace not in ("", "minecraft"):
        return join_state(name, props)
    base = block_roles.normalise(base)
    base = ALWAYS_RENAMED.get(base, base)
    if base == "stone_slab" and data_version is not None and data_version < STONE_SLAB_RENAMED:
        base = "smooth_stone_slab"
    if base in AIR_BASES:
        return AIR
    return join_state("minecraft:" + base, props)


def _is_raw_air(raw: str) -> bool:
    name = split_state(raw)[0]
    return name.rpartition(":")[2] in AIR_BASES and name.rpartition(":")[0] in ("", "minecraft")


def _finish(indices_xyz: np.ndarray, raw_names: list[str], data_version, source_format,
            **extra) -> Build:
    """Compact the palette to what is used, modernised and sorted, air first."""
    counts = np.bincount(indices_xyz.ravel(), minlength=len(raw_names))
    used = np.flatnonzero(counts)
    if len(used) and int(used[-1]) >= len(raw_names):
        raise ReadError(f"block index {int(used[-1])} is past the palette "
                        f"({len(raw_names)} entries)")
    raw_used = {int(i): raw_names[int(i)] for i in used}
    if any(n is None or n == "" for n in raw_used.values()):
        raise ReadError("a used palette index has no blockstate")
    modern = {i: modern_state(n, data_version) for i, n in raw_used.items()}
    palette = [AIR] + sorted(set(modern.values()) - {AIR})
    slot = {s: k for k, s in enumerate(palette)}
    lookup = np.zeros(max(raw_used) + 1 if raw_used else 1, dtype=np.uint16)
    raw_air = np.zeros_like(lookup, dtype=bool)
    for i, s in modern.items():
        lookup[i] = slot[s]
        raw_air[i] = _is_raw_air(raw_used[i])
    blocks = lookup[indices_xyz]
    non_vanilla: dict[str, int] = {}
    for i, s in modern.items():
        if not s.startswith(("minecraft:", "legacy:")):       # legacy: see unknown_legacy
            non_vanilla[s] = non_vanilla.get(s, 0) + int(counts[i])
    return Build(blocks=blocks, palette=palette, data_version=data_version,
                 source_format=source_format,
                 raw_non_air=int(sum(int(counts[i]) for i in raw_used if not raw_air[i])),
                 non_vanilla=dict(sorted(non_vanilla.items())), **extra)


# --------------------------------------------------------------------------- Sponge

def _read_sponge(root) -> Build:
    if "Schematic" in root and "Width" not in root:
        container, version = root["Schematic"], 3
    else:
        container, version = root, int(root.get("Version", 1))
    data_version = int(container["DataVersion"]) if "DataVersion" in container else None
    try:
        palette, block_data, w, h, length = schem_voxels._blocks_source(root)
        raw = np.asarray(block_data).view(np.uint8)
        values = schem_voxels._decode_varints(raw, w * h * length)
    except (ValueError, TypeError, KeyError) as exc:
        raise ReadError(f"Sponge v{version}: {exc}") from exc
    names = [None] * (max(int(v) for v in palette.values()) + 1)
    for state, index in palette.items():
        names[int(index)] = str(state)
    yzx = np.asarray(values).reshape(h, length, w)
    build = _finish(yzx.transpose(2, 0, 1), names, data_version, f"sponge_v{version}")
    if data_version is None:
        build.warnings.append("no DataVersion; 1.13-era renames applied only where safe")
    return build


# --------------------------------------------------------------------------- legacy MCEdit

def _read_legacy(root) -> Build:
    w, h, length = (schem_convert._dim(root, k) for k in ("Width", "Height", "Length"))
    volume = w * h * length
    info = schem_convert.SchematicInfo("mcedit", w, h, length)
    ids = schem_convert._block_ids(root, volume, np)
    data = schem_convert._block_data(root, volume, np)
    keys = schem_convert._palette_keys(ids, data, w, length, np)
    palette, indices, _ = schem_convert._build_palette(keys, np)
    _, warnings = schem_convert._convert_tile_entities(root, info, ids, data, palette, indices)

    unknown: dict[str, int] = {}
    for key in np.unique(keys).tolist():
        if legacy_blocks.resolve(key) is not None:
            continue
        block_id = (key >> 8) & 0xFFF if key & (1 << 20) else key >> 4
        where = keys == key
        label = legacy_blocks.describe(block_id, key & 0x0F if not key & (1 << 20)
                                       else (key >> 4) & 0x0F)
        unknown[label] = unknown.get(label, 0) + int(np.count_nonzero(where))
        indices[where] = palette.setdefault(f"legacy:unknown_{block_id}", len(palette))
    entities = root.get("Entities")
    if entities is not None and len(entities):
        warnings.append(f"{len(entities)} entities not read (only blocks are)")
    names = [None] * (max(palette.values()) + 1)
    for state, index in palette.items():
        names[index] = state
    yzx = indices.reshape(h, length, w)
    return _finish(yzx.transpose(2, 0, 1), names, schem_convert.SPONGE_DATA_VERSION, "mcedit",
                   unknown_legacy=dict(sorted(unknown.items())), warnings=list(warnings))


# --------------------------------------------------------------------------- Litematica

def litematic_bits(palette_size: int) -> int:
    return max(2, (max(palette_size, 1) - 1).bit_length())


def unpack_litematic(longs, bits: int, count: int) -> np.ndarray:
    """Litematica's bit array: `bits` per entry, little-end first, entries span longs."""
    words = np.asarray(longs).astype(np.int64).view(np.uint64)
    need = (count * bits + 63) // 64
    if len(words) < need:
        raise ReadError(f"BlockStates holds {len(words)} longs; {need} needed")
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    start = np.arange(count, dtype=np.int64) * bits
    first, offset = start >> 6, (start & 63).astype(np.uint64)
    last = (start + bits - 1) >> 6
    value = words[first] >> offset
    spans = np.flatnonzero(last != first)
    if len(spans):
        value[spans] |= words[last[spans]] << (np.uint64(64) - offset[spans])
    return (value & np.uint64((1 << bits) - 1)).astype(np.int64)


def _region_extent(region) -> tuple[list[int], list[int]]:
    pos = [int(region["Position"][a]) for a in "xyz"]
    size = [int(region["Size"][a]) for a in "xyz"]
    low = [p + s + 1 if s < 0 else p for p, s in zip(pos, size)]
    return low, [abs(s) for s in size]


def _read_litematic(root) -> Build:
    version = int(root.get("Version", 0))
    data_version = int(root["MinecraftDataVersion"]) if "MinecraftDataVersion" in root else None
    regions = root.get("Regions")
    if not isinstance(regions, dict) or not regions:
        raise ReadError("litematic has no Regions")
    placed = []
    for name, region in regions.items():
        low, size = _region_extent(region)
        if 0 in size:
            continue
        entries = region.get("BlockStatePalette") or []
        names = []
        for entry in entries:
            props = {str(k): str(v) for k, v in (entry.get("Properties") or {}).items()}
            names.append(join_state(str(entry["Name"]), props))
        if not names:
            raise ReadError(f"region {name!r} has an empty palette")
        sx, sy, sz = size
        values = unpack_litematic(region["BlockStates"], litematic_bits(len(names)),
                                  sx * sy * sz)
        if values.max(initial=0) >= len(names):
            raise ReadError(f"region {name!r}: block index past its palette")
        placed.append((low, values.reshape(sy, sz, sx).transpose(2, 0, 1), names))
    if not placed:
        raise ReadError("litematic regions are all empty")

    lo = [min(p[0][a] for p in placed) for a in range(3)]
    hi = [max(p[0][a] + p[1].shape[a] for p in placed) for a in range(3)]
    merged: dict[str, int] = {AIR: 0}
    grid = np.zeros([hi[a] - lo[a] for a in range(3)], dtype=np.int64)
    for low, values, names in placed:            # later regions paste over earlier ones
        remap = np.array([merged.setdefault(n, len(merged)) for n in names], dtype=np.int64)
        air_ids = [i for i, n in enumerate(names) if _is_raw_air(n)]
        solid = ~np.isin(values, air_ids)
        target = grid[tuple(slice(low[a] - lo[a], low[a] - lo[a] + values.shape[a])
                            for a in range(3))]   # a view: assigning writes into grid
        target[solid] = remap[values][solid]
    build = _finish(grid, sorted(merged, key=merged.get), data_version,
                    f"litematic_v{version}", regions=len(placed))
    meta = root.get("Metadata") or {}
    enclosing = meta.get("EnclosingSize")
    if enclosing is not None:
        expected = [int(enclosing[a]) for a in "xyz"]
        if expected != list(build.dims):
            build.warnings.append(f"EnclosingSize {expected} differs from the regions' "
                                  f"extent {list(build.dims)}")
    return build


# --------------------------------------------------------------------------- entry point

def load_build(path) -> Build:
    """Read a schematic of any supported format. Raises ReadError (or FileRejected)."""
    path = Path(path)
    try:
        root = nbt_safe.load(path)
    except FileRejected as exc:
        raise ReadError(f"{exc.stage}: {exc}") from exc
    if "" in root and len(root) == 1:
        root = root[""]
    try:
        if "Regions" in root:
            return _read_litematic(root)
        if ("Schematic" in root and "Width" not in root) or ("Palette" in root
                                                            and "BlockData" in root):
            return _read_sponge(root)
        if "Blocks" in root and "Width" in root:
            return _read_legacy(root)
    except FileRejected as exc:
        raise ReadError(f"{exc.stage}: {exc}") from exc
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ReadError(f"{type(exc).__name__}: {exc}") from exc
    raise ReadError(f"unrecognised schematic layout (root keys: {sorted(root)[:8]})")


def validate(build: Build, row: dict) -> dict:
    """Dimensions and non-air count against the index row (from the collector's manifest)."""
    want = [int(v) for v in row["dimensions_xyz_blocks"]]
    got = list(build.dims)
    if got == want:
        dims = "ok"
    elif [got[2], got[1], got[0]] == want:
        dims = "xz_swapped"
    else:
        dims = "mismatch"
    expected = row.get("non_air_blocks_from_metadata")
    if expected is None:
        non_air = "n/a"
    else:
        non_air = "ok" if build.raw_non_air == int(expected) else "mismatch"
    return {"dims": dims, "dims_read": got, "dims_index": want,
            "non_air": non_air, "non_air_read": build.raw_non_air,
            "non_air_index": expected}
