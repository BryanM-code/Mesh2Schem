"""Identify a Minecraft schematic and, if it is the legacy format, convert it to .schem.

Two formats turn up when downloading builds:

* **Sponge** (`.schem`) -- gzipped NBT with a blockstate ``Palette`` and a varint
  ``BlockData`` array. Versions 1/2 keep those at the root; version 3 nests them under
  ``Schematic.Blocks``. Nothing to do: these pass through byte-for-byte, which keeps the
  library's sha256 dedup stable.

* **MCEdit / WorldEdit** (`.schematic`) -- the pre-1.13 format, with numeric block ids
  in ``Blocks``, nibble metadata in ``Data``, and optionally high id bits in
  ``AddBlocks``. These are rewritten as Sponge v2 using the table in `legacy_blocks`.

Format is decided by *content*, never by the file extension -- a `.schem` that is really
an old MCEdit file gets converted, and a `.schematic` that is really Sponge does not.

Conversion is deterministic: the palette is ordered by legacy key, properties are
sorted, and the gzip header is written with mtime=0. The same input always produces
byte-identical output, so re-adding a file still trips the duplicate check.

What is deliberately *not* carried over is recorded in `ConversionResult.warnings`
rather than dropped silently -- see `convert_legacy`.
"""
from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, field
from pathlib import Path

import legacy_blocks
import nbt_safe
from nbt_safe import FileRejected, NbtError

# 1.13.2. Converted files claim this so Minecraft's data fixers upgrade the 1.13 block
# names in `legacy_blocks` for us whenever a newer version loads the schematic.
SPONGE_DATA_VERSION = 1631

MAX_AXIS = 4096
# Roughly 250^3. Well past any single build, and the ceiling on how much numpy work one
# upload can ask for -- the conversion is O(volume) and holds a few arrays that size.
MAX_VOLUME = 16_000_000

FORMAT_LABELS = {
    "sponge_v1": "Sponge .schem (v1)",
    "sponge_v2": "Sponge .schem (v2)",
    "sponge_v3": "Sponge .schem (v3)",
    "mcedit": "legacy MCEdit .schematic",
}


class SchematicError(FileRejected):
    """Raised when a file is not a usable Minecraft schematic.

    Carries the same structure as every other rejection -- see nbt_safe.FileRejected --
    so a caller can tell *what* was wrong (`code`) and *where* (`path`, `offset`) without
    parsing the message.
    """

    def __init__(self, message, *, code, stage="schematic", offset=None, path=None):
        super().__init__(message, code=code, stage=stage, offset=offset, path=path)


def _reject(exc: NbtError) -> SchematicError:
    """Re-raise a low-level NBT rejection as a schematic one, keeping its diagnostics."""
    return SchematicError(exc.message, code=exc.code, stage=exc.stage,
                          offset=exc.offset, path=exc.path)


@dataclass
class SchematicInfo:
    format: str
    width: int
    height: int
    length: int

    @property
    def label(self) -> str:
        return FORMAT_LABELS.get(self.format, self.format)

    @property
    def dims(self) -> tuple[int, int, int]:
        return self.width, self.height, self.length

    @property
    def is_legacy(self) -> bool:
        return self.format == "mcedit"


@dataclass
class ConversionResult:
    path: Path
    info: SchematicInfo
    converted: bool
    warnings: list[str] = field(default_factory=list)


# ── Reading + identification ──────────────────────────────────────────────────

def _dim(root, name: str) -> int:
    try:
        value = int(root[name]) & 0xFFFF  # Short is signed; recover the unsigned value
    except KeyError as exc:
        raise SchematicError("Not a schematic: no {} tag in the NBT data.".format(name),
                             code="missing_dimension", path=name) from exc
    except (TypeError, ValueError) as exc:
        raise SchematicError("Schematic {} is not a number.".format(name),
                             code="bad_dimension", path=name) from exc
    if value <= 0:
        raise SchematicError("Schematic has a non-positive {} ({}).".format(name, value),
                             code="non_positive_dimension", path=name)
    if value > MAX_AXIS:
        raise SchematicError(
            "Schematic {} is {} blocks; the limit is {}.".format(name, value, MAX_AXIS),
            code="axis_too_large", path=name)
    return value


def _volume(w: int, h: int, length: int) -> int:
    volume = w * h * length
    if volume > MAX_VOLUME:
        raise SchematicError(
            "Schematic is {}x{}x{} = {:,} blocks; the limit is {:,}.".format(
                w, h, length, volume, MAX_VOLUME),
            code="volume_too_large")
    return volume


def _identify(root) -> SchematicInfo:
    """Work out which schematic format `root` is, and validate its block payload."""
    # Sponge v3 nests everything under a "Schematic" compound.
    if "Schematic" in root and "Width" not in root:
        root = root["Schematic"]
        if not isinstance(root, dict):
            raise SchematicError("Not a schematic: the Schematic tag is not a compound.",
                                 code="bad_v3_container", path="Schematic")
        blocks = root.get("Blocks")
        if not isinstance(blocks, dict) or "Palette" not in blocks or "Data" not in blocks:
            raise SchematicError("Not a schematic: v3 container has no Blocks/Palette/Data.",
                                 code="bad_v3_container", path="Schematic.Blocks")
        w, h, length = _dim(root, "Width"), _dim(root, "Height"), _dim(root, "Length")
        _check_block_data(blocks["Data"], blocks["Palette"], _volume(w, h, length),
                          "Schematic.Blocks.")
        return SchematicInfo("sponge_v3", w, h, length)

    w, h, length = _dim(root, "Width"), _dim(root, "Height"), _dim(root, "Length")
    volume = _volume(w, h, length)

    if "Palette" in root and "BlockData" in root:
        version = int(root.get("Version", 2))
        _check_block_data(root["BlockData"], root["Palette"], volume)
        return SchematicInfo("sponge_v1" if version == 1 else "sponge_v2", w, h, length)

    blocks = root.get("Blocks")
    if _is_byte_array(blocks):
        if len(blocks) < volume:
            raise SchematicError(
                "Schematic is truncated: {}x{}x{} needs {:,} block ids but only {:,} "
                "are present.".format(w, h, length, volume, len(blocks)),
                code="truncated_blocks", path="Blocks")
        return SchematicInfo("mcedit", w, h, length)

    raise SchematicError(
        "Not a schematic: the NBT has dimensions but no block data "
        "(expected Palette+BlockData, or Blocks for the legacy format).",
        code="no_block_data")


def _is_byte_array(tag) -> bool:
    """True for an NBT byte array. nbtlib models those as 1-byte numpy arrays, so this
    also rules out an Int_Array or a String sitting where block data should be."""
    dtype = getattr(tag, "dtype", None)
    return dtype is not None and dtype.itemsize == 1


def _byte_view(root, name: str, np):
    """Return an NBT byte array as unsigned bytes, or raise if it isn't one."""
    tag = root.get(name)
    if not _is_byte_array(tag):
        raise SchematicError(
            "Schematic's {} tag is missing or is not a byte array.".format(name),
            code="bad_byte_array", path=name)
    return np.asarray(tag).view(np.uint8)


# Maps every varint terminator byte (high bit clear) to 0 and every continuation byte
# to 1, so a single bytes.count() tallies the encoded values at C speed.
_VARINT_TERMINATORS = bytes(0 if b < 0x80 else 1 for b in range(256))


def _check_block_data(block_data, palette, volume: int, prefix: str = "") -> None:
    """Sponge block data is a stream of LEB128 varints -- one per block, in order."""
    if not isinstance(palette, dict) or not palette:
        raise SchematicError("Schematic has an empty or unreadable block palette.",
                             code="bad_palette", path=prefix + "Palette")
    raw = block_data.tobytes() if hasattr(block_data, "tobytes") else bytes(block_data)
    count = raw.translate(_VARINT_TERMINATORS).count(0)
    if count != volume:
        raise SchematicError(
            "Schematic block data does not match its dimensions "
            "({:,} blocks encoded, {:,} expected).".format(count, volume),
            code="block_count_mismatch",
            path=prefix + ("Data" if prefix else "BlockData"))


def inspect(path) -> SchematicInfo:
    """Validate `path` as a schematic and report its format and dimensions."""
    try:
        root = nbt_safe.load(path)
    except NbtError as exc:
        raise _reject(exc) from exc
    if "" in root:  # some writers wrap the payload in an explicitly-unnamed compound
        root = root[""]
    return _identify(root)


# ── Conversion ────────────────────────────────────────────────────────────────

def ensure_sponge(src_path, dest_path) -> ConversionResult:
    """Return a Sponge .schem for `src_path`, converting into `dest_path` if needed.

    Sponge files are left exactly where they are (``converted=False``, ``path=src_path``)
    so their bytes -- and therefore their content hash -- never change.
    """
    src_path = Path(src_path)
    try:
        root = nbt_safe.load(src_path)
    except NbtError as exc:
        raise _reject(exc) from exc
    if "" in root:
        root = root[""]

    info = _identify(root)
    if not info.is_legacy:
        return ConversionResult(path=src_path, info=info, converted=False)

    schematic, warnings = convert_legacy(root, info)
    _write_sponge(schematic, dest_path)
    return ConversionResult(path=Path(dest_path), info=info, converted=True,
                            warnings=warnings)


def convert_legacy(root, info: SchematicInfo):
    """Build a Sponge v2 NBT document from a parsed legacy MCEdit schematic.

    Returns ``(nbtlib.File, warnings)``. Warnings name everything the legacy format
    carried that this conversion could not: unmapped (usually modded) block ids, and
    entities, which have no faithful 1.13 spelling and are not part of a build's shape.
    """
    import nbtlib
    import numpy as np
    from nbtlib import ByteArray, Compound, Int, IntArray, List, Short

    w, h, length = info.dims
    volume = w * h * length
    warnings: list[str] = []

    ids = _block_ids(root, volume, np)
    data = _block_data(root, volume, np)
    keys = _palette_keys(ids, data, w, length, np)

    palette, indices, unmapped = _build_palette(keys, np)
    if unmapped:
        shown = ", ".join(sorted(unmapped)[:6])
        more = "" if len(unmapped) <= 6 else " (+{} more)".format(len(unmapped) - 6)
        warnings.append(
            "{} legacy block id(s) have no 1.13 equivalent and were kept as stone so the "
            "shape survives: {}{}. These are usually modded blocks.".format(
                len(unmapped), shown, more))

    block_entities, be_warnings = _convert_tile_entities(
        root, info, ids, data, palette, indices)
    warnings.extend(be_warnings)

    entities = root.get("Entities")
    if entities is not None and len(entities) > 0:
        warnings.append(
            "{} entity/entities (mobs, item frames, armour stands) were not carried "
            "over; only blocks are converted.".format(len(entities)))

    metadata = Compound()
    offset = [0, 0, 0]
    for axis, tag in enumerate(("WEOffsetX", "WEOffsetY", "WEOffsetZ")):
        if tag in root:
            try:
                offset[axis] = int(root[tag])
                metadata[tag] = Int(offset[axis])
            except (TypeError, ValueError):
                pass

    schematic = Compound({
        "Version": Int(2),
        "DataVersion": Int(SPONGE_DATA_VERSION),
        "Metadata": metadata,
        "Width": Short(w), "Height": Short(h), "Length": Short(length),
        "Offset": IntArray(offset),
        "PaletteMax": Int(len(palette)),
        "Palette": Compound({state: Int(index) for state, index in palette.items()}),
        "BlockData": ByteArray(_encode_varints(indices, len(palette), np)),
        "BlockEntities": List[Compound](block_entities),
    })
    return nbtlib.File(schematic), warnings


# ── Legacy block arrays ───────────────────────────────────────────────────────

def _block_ids(root, volume: int, np):
    """Combine `Blocks` with the optional high-bit nibbles from `AddBlocks`/`Add`."""
    blocks = _byte_view(root, "Blocks", np)[:volume].astype(np.uint16)
    if len(blocks) < volume:
        raise SchematicError("Schematic is missing block ids.",
                             code="truncated_blocks", path="Blocks")

    # Two spellings in the wild for the same thing: ids above 255, four bits per block.
    field = "AddBlocks" if "AddBlocks" in root else ("Add" if "Add" in root else None)
    if field is None:
        return blocks

    add = _byte_view(root, field, np)
    if len(add) * 2 < volume:
        raise SchematicError(
            "Schematic's {} array holds {:,} nibbles for {:,} blocks.".format(
                field, len(add) * 2, volume), code="add_array_too_short", path=field)
    # Nibble-packed, low nibble first: even block index -> low, odd -> high.
    high = np.empty(len(add) * 2, dtype=np.uint16)
    high[0::2] = add & 0x0F
    high[1::2] = add >> 4
    return blocks | (high[:volume] << 8)


def _block_data(root, volume: int, np):
    if "Data" not in root:
        return np.zeros(volume, dtype=np.uint8)  # no metadata at all: every block default
    values = _byte_view(root, "Data", np)
    if len(values) < volume:
        raise SchematicError(
            "Schematic is truncated: {:,} metadata values for {:,} blocks.".format(
                len(values), volume), code="truncated_data", path="Data")
    return values[:volume] & 0x0F


def _palette_keys(ids, data, w: int, length: int, np):
    """Pack (id, data) into one lookup key per block, widened for upper halves."""
    keys = (ids.astype(np.uint32) << 4) | data
    layer = w * length

    # Door and double-plant tops stored their variant only in the block *below* them,
    # so those get a wider key carrying both halves' metadata.
    needs_context = np.zeros(len(ids), dtype=bool)
    for block_id in legacy_blocks.NEEDS_BLOCK_BELOW:
        needs_context |= ids == block_id
    needs_context &= (data & 0x08) != 0
    needs_context[:layer] = False  # nothing below the bottom layer to read
    where = np.flatnonzero(needs_context)
    if len(where):
        below = data[where - layer]
        keys[where] = ((1 << 20) | (ids[where].astype(np.uint32) << 8)
                       | (data[where].astype(np.uint32) << 4) | below)
    return keys


def _build_palette(keys, np):
    """Map every distinct legacy key to a blockstate, then to a palette index."""
    distinct = np.unique(keys)
    if len(distinct) > 0xFFFF:
        raise SchematicError("Schematic uses more than 65,535 distinct block states.",
                             code="palette_too_large", path="Blocks")

    palette: dict[str, int] = {}
    remap = np.empty(len(distinct), dtype=np.uint16)
    unmapped: set[str] = set()
    for slot, packed in enumerate(distinct.tolist()):
        state = legacy_blocks.resolve(packed)
        if state is None:
            block_id = (packed >> 8) & 0xFFF if packed & (1 << 20) else packed >> 4
            unmapped.add(legacy_blocks.describe(block_id, packed & 0x0F))
            state = "minecraft:stone"
        remap[slot] = palette.setdefault(state, len(palette))

    # A lookup table beats np.unique(return_inverse=True) here: keys top out at 2^21,
    # so the table is tiny and the block array never needs an int64 companion.
    lookup = np.zeros(int(distinct[-1]) + 1, dtype=np.uint16)
    lookup[distinct] = remap
    return palette, lookup[keys], unmapped


def _encode_varints(indices, palette_size: int, np):
    """Sponge BlockData: one unsigned LEB128 varint per block, in YZX order."""
    if palette_size <= 0x80:
        raw = indices.astype(np.uint8).tobytes()  # every index fits in a single byte
    else:
        encoded = [_varint(i) for i in range(palette_size)]
        raw = b"".join([encoded[i] for i in indices.tolist()])
    # NBT byte arrays are signed, so hand nbtlib int8 rather than letting it guess.
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int8)


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        out.append(chunk | 0x80 if value else chunk)
        if not value:
            return bytes(out)


# ── Tile entities ─────────────────────────────────────────────────────────────

# Pre-1.11 files used short tile entity names; 1.11+ already writes namespaced ids.
_TILE_ENTITY_IDS = {
    "Airportal": "end_portal", "Banner": "banner", "Beacon": "beacon", "Bed": "bed",
    "Cauldron": "brewing_stand", "Chest": "chest", "Comparator": "comparator",
    "Control": "command_block", "DLDetector": "daylight_detector", "Dropper": "dropper",
    "EnchantTable": "enchanting_table", "EndGateway": "end_gateway",
    "EnderChest": "ender_chest", "EndPortal": "end_portal", "FlowerPot": "flower_pot",
    "Furnace": "furnace", "Hopper": "hopper", "MobSpawner": "mob_spawner",
    "Music": "noteblock", "Piston": "piston", "RecordPlayer": "jukebox",
    "Sign": "sign", "Skull": "skull", "Structure": "structure_block",
    "Trap": "dispenser",
}


def _convert_tile_entities(root, info, ids, data, palette, indices):
    """Rewrite TileEntities as Sponge BlockEntities, refining the blocks they describe.

    Banners, beds, skulls and flower pots all look identical in the legacy block array
    and stored what they actually were in the matching tile entity, so this pass patches
    those positions in the block index array as it walks the list.
    """
    from nbtlib import Compound, IntArray, String

    tile_entities = root.get("TileEntities")
    if not tile_entities:
        return [], []

    w, h, length = info.dims
    converted, warnings, out_of_bounds = [], [], 0

    for entry in tile_entities:
        if not isinstance(entry, dict):
            continue
        try:
            x, y, z = int(entry["x"]), int(entry["y"]), int(entry["z"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= x < w and 0 <= y < h and 0 <= z < length):
            out_of_bounds += 1
            continue

        position = (y * length + z) * w + x
        refined = _refine_block(int(ids[position]), int(data[position]), entry)
        if refined is not None:
            indices[position] = palette.setdefault(refined, len(palette))

        payload = Compound({k: v for k, v in entry.items()
                            if k not in ("x", "y", "z", "id")})
        payload["Id"] = String(_normalize_tile_entity_id(entry.get("id")))
        payload["Pos"] = IntArray([x, y, z])
        converted.append(payload)

    if out_of_bounds:
        warnings.append(
            "{} tile entity/entities sat outside the schematic bounds and were "
            "dropped.".format(out_of_bounds))
    return converted, warnings


def _normalize_tile_entity_id(raw) -> str:
    name = str(raw or "").strip()
    if not name:
        return "minecraft:air"
    if ":" in name:
        return name
    return "minecraft:" + _TILE_ENTITY_IDS.get(name, name.lower())


def _refine_block(block_id: int, data: int, entry):
    """Blockstate for the four families whose identity lived in the tile entity."""
    if block_id == 176 or block_id == 177:
        base = _int_tag(entry, "Base")
        return legacy_blocks.banner_state(data, 15 if base is None else base,
                                          wall=(block_id == 177))
    if block_id == 26:
        color = _int_tag(entry, "color")
        return legacy_blocks.bed_state(data, 14 if color is None else color)
    if block_id == 144:
        return legacy_blocks.skull_state(data, _int_tag(entry, "SkullType") or 0,
                                         _int_tag(entry, "Rot") or 0)
    if block_id == 140:
        return legacy_blocks.potted_state(entry.get("Item"), _int_tag(entry, "Data") or 0)
    return None


def _int_tag(entry, name):
    try:
        return int(entry[name])
    except (KeyError, TypeError, ValueError):
        return None


# ── Writing ───────────────────────────────────────────────────────────────────

def _write_sponge(schematic, dest_path) -> None:
    """Write gzipped NBT deterministically: same input bytes in, same bytes out.

    Both `filename=""` and `mtime=0` matter. Left to itself GzipFile stamps the header
    with the output file's name and the current time, which would make the content hash
    of a converted schematic depend on which temp file it happened to land in.
    """
    payload = io.BytesIO()
    schematic.write(payload)
    with open(dest_path, "wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=fh, mtime=0) as gz:
            gz.write(payload.getvalue())
