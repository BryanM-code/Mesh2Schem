"""Read a Minecraft Bedrock world export (`.mcworld`) far enough to cut a build out of it.

A `.mcworld` is a ZIP holding `level.dat` and a LevelDB database of chunks. Nothing in the
library can use that directly -- builds are stored as Sponge `.schem` -- so this module
reads the world and `mcworld_convert` turns a chosen region into one.

Everything here is stdlib (`zipfile`, `struct`, `zlib`). No LevelDB library is involved:
the on-disk SST format is simple enough to read directly, and Mojang's fork uses a
compression type that most bindings reject anyway.

Three formats stack up, outermost first:

1. **SST (LevelDB table).** A footer at the end points at an index block, whose entries
   point at data blocks, each holding prefix-compressed key/value pairs.
2. **Chunk keys.** `<cx int32 LE><cz int32 LE>[<dimension int32 LE>]<tag>[<subY>]`, where
   tag 0x2F is a subchunk of 16x16x16 blocks.
3. **Subchunk payload.** A bit-packed array of palette indices, then the palette itself as
   *little-endian* NBT -- Bedrock's byte order, not Java's.

Everything is bounded. These files come off the internet, so the module refuses anything
oversized before allocating for it, and never joins an archive's own names onto a path.
"""
from __future__ import annotations

import struct
import zipfile
import zlib
from pathlib import Path
from typing import Iterator, Optional

# ── Limits ────────────────────────────────────────────────────────────────────
# Sized to admit a real world comfortably while refusing anything that looks like an
# attempt to exhaust memory. A 672x624 world -- the one this was built against -- uses
# 1,780 subchunks and 368 KB uncompressed.
MAX_ZIP_ENTRIES = 10_000
MAX_ZIP_UNCOMPRESSED = 2 * 1024 * 1024 * 1024      # 2 GiB of world data
MAX_LDB_BYTES = 512 * 1024 * 1024
MAX_BLOCK_BYTES = 64 * 1024 * 1024                 # one decompressed SST block
MAX_RECORDS = 5_000_000
MAX_PALETTE = 65_536
MAX_NBT_DEPTH = 32

SST_MAGIC = 0xDB4775248B80FB57
TAG_SUBCHUNK = 0x2F
BLOCKS_PER_SUBCHUNK = 4096

# Bedrock packs indices at one of these widths; anything else is a malformed subchunk
# rather than a format we don't know, because the set has never changed.
VALID_BITS = (1, 2, 3, 4, 5, 6, 8, 16)

# What counts as "not part of the build" when locating it automatically. A creative build
# sits on a flat plate of these; including them would blow past the schematic volume limit
# and bury the thing you actually want.
GROUND_BLOCKS = frozenset({
    "minecraft:air", "minecraft:cave_air", "minecraft:void_air",
    "minecraft:bedrock", "minecraft:dirt", "minecraft:grass_block",
    "minecraft:water", "minecraft:flowing_water", "minecraft:lava",
})
AIR_BLOCKS = frozenset({"minecraft:air", "minecraft:cave_air", "minecraft:void_air"})

__all__ = ["WorldError", "BedrockWorld", "open_world", "GROUND_BLOCKS", "AIR_BLOCKS"]


class WorldError(Exception):
    """A `.mcworld` we refuse to read.

    Carries the same shape as `schem_convert.SchematicError` -- a stable `code` plus the
    `stage` that rejected it -- so the UI's existing error rendering works unchanged and
    tests can assert on the code rather than the wording.
    """

    def __init__(self, message, *, code, stage="mcworld"):
        super().__init__(message)
        self.message = message
        self.code = code
        self.stage = stage

    def details(self) -> dict:
        return {"error": self.message, "message": self.message,
                "code": self.code, "stage": self.stage, "offset": None, "path": None}


# ── Varints and little-endian NBT ─────────────────────────────────────────────

def _varint(buf: bytes, i: int) -> tuple[int, int]:
    """LEB128, as LevelDB writes lengths and offsets."""
    result = shift = 0
    while True:
        if i >= len(buf):
            raise WorldError("Truncated varint in the world database.",
                             code="truncated_varint", stage="leveldb")
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7
        if shift > 63:
            raise WorldError("Malformed varint in the world database.",
                             code="bad_varint", stage="leveldb")


def _read_nbt_le(buf: bytes, i: int, depth: int = 0):
    """Read one little-endian NBT tag. Bedrock's palettes are stored this way.

    The project's `nbt_safe`/`nbtlib` path reads Java's big-endian NBT and cannot be
    pointed at this, so palettes get their own small reader. Only the tags Bedrock
    actually uses in a block palette are implemented; anything else is a rejection rather
    than a guess.
    """
    if depth > MAX_NBT_DEPTH:
        raise WorldError("Block palette is nested too deeply.",
                         code="nbt_too_deep", stage="palette")

    def take(n):
        nonlocal i
        if i + n > len(buf):
            raise WorldError("Truncated block palette.", code="truncated_palette",
                             stage="palette")
        chunk = buf[i:i + n]
        i += n
        return chunk

    def string():
        length = struct.unpack("<H", take(2))[0]
        return take(length).decode("utf-8", "replace")

    def payload(tag):
        nonlocal i
        if tag == 1:
            return take(1)[0]
        if tag == 2:
            return struct.unpack("<h", take(2))[0]
        if tag == 3:
            return struct.unpack("<i", take(4))[0]
        if tag == 4:
            return struct.unpack("<q", take(8))[0]
        if tag == 5:
            return struct.unpack("<f", take(4))[0]
        if tag == 6:
            return struct.unpack("<d", take(8))[0]
        if tag == 8:
            return string()
        if tag == 7:
            return take(struct.unpack("<i", take(4))[0])
        if tag == 9:
            element = take(1)[0]
            count = struct.unpack("<i", take(4))[0]
            if count < 0 or count > MAX_PALETTE:
                raise WorldError("Block palette declares an implausible list.",
                                 code="bad_list_length", stage="palette")
            return [payload(element) for _ in range(count)]
        if tag == 10:
            compound = {}
            while True:
                inner = take(1)[0]
                if inner == 0:
                    return compound
                key = string()
                compound[key] = payload(inner)
        raise WorldError("Unsupported NBT tag {} in block palette.".format(tag),
                         code="bad_nbt_tag", stage="palette")

    tag = buf[i]
    i += 1
    if tag == 0:
        return None, i
    string()                                     # root name, always empty here
    return payload(tag), i


# ── LevelDB SST ───────────────────────────────────────────────────────────────

def _decompress_block(raw: bytes, kind: int) -> bytes:
    """SST blocks carry a compression byte. Type 4 is Mojang's raw-zlib addition."""
    try:
        if kind == 0:
            data = raw
        elif kind == 2:
            data = zlib.decompress(raw)
        elif kind == 4:
            data = zlib.decompress(raw, -15)
        else:
            raise WorldError("World database uses compression type {}.".format(kind),
                             code="unsupported_compression", stage="leveldb")
    except zlib.error as exc:
        raise WorldError("Could not decompress the world database: {}".format(exc),
                         code="bad_compression", stage="leveldb") from exc
    if len(data) > MAX_BLOCK_BYTES:
        raise WorldError("World database block expands to {:,} bytes.".format(len(data)),
                         code="block_too_large", stage="leveldb")
    return data


def _block_entries(block: bytes) -> Iterator[tuple[bytes, bytes]]:
    """Walk one SST block's key/value pairs.

    Keys are prefix-compressed against the previous key. The tail of the block is a restart
    array -- `n` offsets plus a uint32 count -- which is not entry data.

    The returned key has its last 8 bytes removed. LevelDB stores *internal* keys: the user
    key followed by a sequence number and value type. Leaving those on makes every key 8
    bytes too long and nothing matches the chunk-key layout at all.
    """
    if len(block) < 4:
        raise WorldError("Truncated block in the world database.",
                         code="truncated_block", stage="leveldb")
    restarts = struct.unpack("<I", block[-4:])[0]
    end = len(block) - 4 - 4 * restarts
    if end < 0:
        raise WorldError("Corrupt restart array in the world database.",
                         code="bad_restarts", stage="leveldb")
    i, key = 0, b""
    while i < end:
        shared, i = _varint(block, i)
        unshared, i = _varint(block, i)
        value_len, i = _varint(block, i)
        if shared > len(key) or i + unshared + value_len > len(block):
            raise WorldError("Corrupt entry in the world database.",
                             code="bad_entry", stage="leveldb")
        key = key[:shared] + block[i:i + unshared]
        i += unshared
        value = block[i:i + value_len]
        i += value_len
        if len(key) > 8:
            yield key[:-8], value


def _read_table(data: bytes) -> Iterator[tuple[bytes, bytes]]:
    """Every key/value pair in one .ldb table file."""
    if len(data) < 48:
        raise WorldError("World database file is too small to be a LevelDB table.",
                         code="ldb_too_small", stage="leveldb")
    magic = struct.unpack("<Q", data[-8:])[0]
    if magic != SST_MAGIC:
        raise WorldError("World database file is not a LevelDB table.",
                         code="bad_ldb_magic", stage="leveldb")

    footer = data[-48:-8]
    i = 0
    _, i = _varint(footer, i)                    # metaindex handle: unused
    _, i = _varint(footer, i)
    index_offset, i = _varint(footer, i)
    index_size, i = _varint(footer, i)

    def block_at(offset, size):
        if offset + size + 1 > len(data):
            raise WorldError("World database points past the end of the file.",
                             code="handle_out_of_range", stage="leveldb")
        return _decompress_block(data[offset:offset + size], data[offset + size])

    handles = []
    for _key, value in _block_entries(block_at(index_offset, index_size)):
        j = 0
        offset, j = _varint(value, j)
        size, j = _varint(value, j)
        handles.append((offset, size))

    seen = 0
    for offset, size in handles:
        for key, value in _block_entries(block_at(offset, size)):
            seen += 1
            if seen > MAX_RECORDS:
                raise WorldError("World database holds more than {:,} records.".format(
                    MAX_RECORDS), code="too_many_records", stage="leveldb")
            yield key, value


# ── Subchunks ─────────────────────────────────────────────────────────────────

def _decode_subchunk(payload: bytes) -> Optional[tuple[list[str], list[int]]]:
    """Decode one subchunk into (palette names, 4096 indices), or None if it holds no blocks.

    Only the first storage layer is read. The second, when present, is waterlogging state
    -- which this pipeline drops along with every other block state.
    """
    if len(payload) < 2:
        return None
    i = 0
    version = payload[i]; i += 1
    storages = payload[i]; i += 1
    if version == 9:
        i += 1                                   # subchunk Y, already known from the key
    if storages < 1:
        return None

    header = payload[i]; i += 1
    bits = header >> 1
    if bits == 0:
        return None
    if bits not in VALID_BITS:
        raise WorldError("Subchunk uses {} bits per block.".format(bits),
                         code="bad_bits_per_block", stage="subchunk")

    per_word = 32 // bits
    words = -(-BLOCKS_PER_SUBCHUNK // per_word)
    if i + words * 4 + 4 > len(payload):
        raise WorldError("Truncated subchunk block data.", code="truncated_subchunk",
                         stage="subchunk")

    # Unpacked with numpy rather than a Python loop: a world is thousands of subchunks of
    # 4096 blocks each, and as Python ints that is both slow to build and hundreds of
    # megabytes to hold. As a uint16 array a whole world is a few tens of MB, which is
    # what makes it practical to keep one parsed while the region is chosen.
    import numpy as np

    packed = np.frombuffer(payload, dtype="<u4", count=words, offset=i)
    shifts = (np.arange(per_word, dtype=np.uint32) * bits)
    values = (packed[:, None] >> shifts[None, :]) & ((1 << bits) - 1)
    indices = values.reshape(-1)[:BLOCKS_PER_SUBCHUNK].astype(np.uint16)
    i += words * 4

    count = struct.unpack_from("<i", payload, i)[0]; i += 4
    if count < 1 or count > MAX_PALETTE:
        raise WorldError("Subchunk palette declares {} entries.".format(count),
                         code="bad_palette_size", stage="subchunk")
    names = []
    for _ in range(count):
        entry, i = _read_nbt_le(payload, i)
        name = entry.get("name") if isinstance(entry, dict) else None
        names.append(str(name) if name else "minecraft:air")
    return names, indices


# ── The world ─────────────────────────────────────────────────────────────────

class BedrockWorld:
    """A parsed Bedrock world, held as decoded subchunks keyed by (cx, cz, subY)."""

    def __init__(self, subchunks: dict, level_name: str = ""):
        self.subchunks = subchunks
        self.level_name = level_name

    # ── Extent ────────────────────────────────────────────────────────────────

    def world_bounds(self) -> dict:
        """Block-space extent of every subchunk present, built or not."""
        if not self.subchunks:
            raise WorldError("This world has no block data.", code="empty_world")
        xs = [cx for cx, _cz, _sy in self.subchunks]
        zs = [cz for _cx, cz, _sy in self.subchunks]
        ys = [sy for _cx, _cz, sy in self.subchunks]
        return {
            "min_x": min(xs) * 16, "max_x": max(xs) * 16 + 15,
            "min_y": min(ys) * 16, "max_y": max(ys) * 16 + 15,
            "min_z": min(zs) * 16, "max_z": max(zs) * 16 + 15,
        }

    def _interesting_mask(self, names, indices, skip):
        """Boolean mask over a subchunk's 4096 blocks: True where the name isn't skipped.

        Tested per *palette entry* and then indexed, not per block -- a palette is tens of
        names where the subchunk is 4096 blocks.
        """
        import numpy as np

        keep = np.array([name not in skip for name in names], dtype=bool)
        safe = np.where(indices < len(names), indices, 0)
        return keep[safe] & (indices < len(names))

    def auto_bbox(self, skip=GROUND_BLOCKS) -> Optional[dict]:
        """Bounding box of the built area: everything that isn't air or ground plate.

        This is what makes a world importable at all. A flat creative world is mostly a
        plate of bedrock/dirt/grass whose full extent runs past the schematic volume limit;
        the part worth keeping is a fraction of it.
        """
        import numpy as np

        # Block position within a subchunk is packed as (x << 8) | (z << 4) | y.
        positions = np.arange(BLOCKS_PER_SUBCHUNK)
        local_x = (positions >> 8) & 15
        local_y = positions & 15
        local_z = (positions >> 4) & 15

        lo = [None, None, None]
        hi = [None, None, None]
        count = 0
        for (cx, cz, sy), (names, indices) in self.subchunks.items():
            mask = self._interesting_mask(names, indices, skip)
            if not mask.any():
                continue
            count += int(mask.sum())
            for axis, (local, base) in enumerate(
                    ((local_x, cx * 16), (local_y, sy * 16), (local_z, cz * 16))):
                present = local[mask]
                low = base + int(present.min())
                high = base + int(present.max())
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

        for (cx, cz, sy), (sub_names, indices) in self.subchunks.items():
            # Skip whole subchunks that cannot overlap the region.
            if cx * 16 > x1 or cx * 16 + 15 < x0:
                continue
            if cz * 16 > z1 or cz * 16 + 15 < z0:
                continue
            if sy * 16 > y1 or sy * 16 + 15 < y0:
                continue
            base_x, base_y, base_z = cx * 16, sy * 16, cz * 16
            mask = self._interesting_mask(sub_names, indices, AIR_BLOCKS)
            if not mask.any():
                continue

            positions = np.nonzero(mask)[0]
            xs = base_x + ((positions >> 8) & 15)
            ys = base_y + (positions & 15)
            zs = base_z + ((positions >> 4) & 15)
            inside = ((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
                      & (zs >= z0) & (zs <= z1))
            if not inside.any():
                continue
            positions, xs, ys, zs = positions[inside], xs[inside], ys[inside], zs[inside]

            # Map this subchunk's palette onto the region's shared one, then write every
            # block in one indexed assignment. Only the entries actually used by blocks
            # inside the region are carried over -- a subchunk usually straddles the
            # boundary, and adopting its whole palette would pad the schematic with blocks
            # the region does not contain.
            local_indices = indices[positions]
            local = np.zeros(len(sub_names), dtype=np.uint16)
            for index in np.unique(local_indices):
                name = sub_names[int(index)]
                slot = palette.get(name)
                if slot is None:
                    slot = len(names)
                    palette[name] = slot
                    names.append(name)
                local[index] = slot
            grid[ys - y0, zs - z0, xs - x0] = local[local_indices]
        return grid, names


def open_world(path) -> BedrockWorld:
    """Read a `.mcworld` into memory, refusing anything oversized or malformed."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise WorldError("Not a .mcworld file (it isn't a zip archive).",
                         code="not_a_zip", stage="read")

    subchunks = {}
    level_name = ""
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ZIP_ENTRIES:
                raise WorldError("World archive holds {:,} entries.".format(len(infos)),
                                 code="too_many_entries", stage="read")
            total = sum(info.file_size for info in infos)
            if total > MAX_ZIP_UNCOMPRESSED:
                raise WorldError("World archive expands to {:,} bytes.".format(total),
                                 code="archive_too_large", stage="read")

            # Names are matched, never joined onto a path: an entry called
            # "../../etc/passwd" is simply not one of the names we look for.
            tables = sorted(
                info for info in infos
                if info.filename.startswith("db/") and info.filename.endswith(".ldb"))
            if not tables:
                raise WorldError(
                    "No world database inside this .mcworld (expected db/*.ldb).",
                    code="no_leveldb", stage="read")

            for info in tables:
                if info.file_size > MAX_LDB_BYTES:
                    raise WorldError("World database file is {:,} bytes.".format(
                        info.file_size), code="ldb_too_large", stage="read")
                data = archive.read(info.filename)
                for key, value in _read_table(data):
                    if len(key) != 10 or key[8] != TAG_SUBCHUNK:
                        continue                 # overworld subchunks only
                    decoded = _decode_subchunk(value)
                    if decoded is None:
                        continue
                    cx, cz = struct.unpack("<ii", key[0:8])
                    sy = struct.unpack("<b", key[9:10])[0]
                    subchunks[(cx, cz, sy)] = decoded

            for info in infos:
                if info.filename == "levelname.txt" and info.file_size < 4096:
                    level_name = archive.read(info.filename).decode(
                        "utf-8", "replace").strip()
                    break
    except zipfile.BadZipFile as exc:
        raise WorldError("Damaged .mcworld archive: {}".format(exc),
                         code="bad_zip", stage="read") from exc

    if not subchunks:
        raise WorldError("This world contains no readable block data.",
                         code="no_subchunks", stage="leveldb")
    return BedrockWorld(subchunks, level_name)
