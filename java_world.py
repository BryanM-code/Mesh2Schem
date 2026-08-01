"""Read a Minecraft Java Edition world (Anvil) far enough to cut a build out of it.

Builds downloaded from sites like PlanetMinecraft often arrive as a zipped *world* rather
than a schematic -- a folder of `level.dat` and `region/r.<x>.<z>.mca` files, with no
single file inside that is the build. This reads one, so the same map-and-drag picker used
for Bedrock exports works on them too.

Java's format is friendlier than Bedrock's in one important way: its NBT is big-endian, so
`nbtlib` reads it directly and there's no second NBT reader to maintain.

Two layers:

1. **Region file (`.mca`).** An 8 KiB header -- 1024 chunk locations (3-byte sector offset
   plus 1-byte sector count) then 1024 timestamps -- followed by chunks at those offsets,
   each a 4-byte big-endian length, a compression byte, and compressed NBT.
2. **Chunk NBT.** Sections of 16x16x16 blocks, each a palette plus a bit-packed index
   array. Both the container names and the packing changed over the years; see below.

As everywhere else here, a downloaded file is bounded before it is trusted.
"""
from __future__ import annotations

import gzip
import io
import re
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Iterator, Optional

from bedrock_world import WorldError
from voxel_world import VoxelWorld

# ── Limits ────────────────────────────────────────────────────────────────────
MAX_ZIP_ENTRIES = 20_000
MAX_ZIP_UNCOMPRESSED = 4 * 1024 * 1024 * 1024
MAX_REGION_BYTES = 256 * 1024 * 1024
MAX_CHUNK_BYTES = 32 * 1024 * 1024
MAX_REGIONS = 512
MAX_PALETTE = 65_536

# 1.16 (DataVersion 2527) stopped letting a block index straddle two longs: from then on
# each long is padded rather than packed tight. Reading one as the other yields blocks that
# are subtly, silently wrong, so the world's DataVersion decides which way to unpack.
PADDED_SINCE = 2527

# 1.18 (DataVersion 2825) moved sections from `Level.Sections` to a top-level `sections`,
# and the palette from `Palette`/`BlockStates` into `block_states.palette`/`.data`.
MODERN_SINCE = 2825

REGION_RE = re.compile(r"^r\.(-?\d+)\.(-?\d+)\.mca$")

__all__ = ["JavaWorld", "open_world", "find_world_root", "looks_like_java_world"]


class JavaWorld(VoxelWorld):
    """A parsed Java world. The geometry lives in VoxelWorld, shared with Bedrock."""

    def __init__(self, subchunks: dict, level_name: str = ""):
        super().__init__(subchunks, level_name, source_label="Java world")


# ── Chunk decoding ────────────────────────────────────────────────────────────

def _decompress_chunk(payload: bytes, kind: int) -> bytes:
    try:
        if kind == 1:
            data = gzip.decompress(payload)
        elif kind == 2:
            data = zlib.decompress(payload)
        elif kind == 3:
            data = payload
        else:
            raise WorldError(
                "Chunk uses compression type {} (LZ4 or custom), which isn't "
                "supported.".format(kind),
                code="unsupported_chunk_compression", stage="region")
    except (OSError, zlib.error) as exc:
        raise WorldError("Could not decompress a chunk: {}".format(exc),
                         code="bad_chunk_compression", stage="region") from exc
    if len(data) > MAX_CHUNK_BYTES:
        raise WorldError("A chunk expands to {:,} bytes.".format(len(data)),
                         code="chunk_too_large", stage="region")
    return data


def _unpack_indices(longs, bits: int, padded: bool):
    """Expand a section's packed palette indices into 4096 values.

    Pre-1.16 an index could span the boundary between two longs; from 1.16 each long is
    padded so it can't. The two layouts are indistinguishable from the data alone, which is
    why the world's DataVersion has to pick.
    """
    import numpy as np

    words = np.asarray(longs, dtype=">i8").astype(np.uint64)
    mask = np.uint64((1 << bits) - 1)

    if padded:
        per_word = 64 // bits
        shifts = (np.arange(per_word, dtype=np.uint64) * np.uint64(bits))
        values = (words[:, None] >> shifts[None, :]) & mask
        return values.reshape(-1)[:4096].astype(np.uint16)

    # Spanning layout: treat the longs as one continuous little-endian bit stream.
    out = np.zeros(4096, dtype=np.uint16)
    buffer = 0
    held = 0
    produced = 0
    for word in words.tolist():
        buffer |= int(word) << held
        held += 64
        while held >= bits and produced < 4096:
            out[produced] = buffer & int(mask)
            buffer >>= bits
            held -= bits
            produced += 1
        if produced >= 4096:
            break
    return out


def _java_to_shared_order():
    """Permutation from Java's block order to the one `VoxelWorld` uses.

    Java indexes a section YZX (`y<<8 | z<<4 | x`); Bedrock indexes XZY
    (`x<<8 | z<<4 | y`). They look interchangeable and are not -- decoding one as the other
    swaps X and Y, which renders as a build sliced into stripes one chunk apart. Converting
    once here means everything downstream has a single convention to think about.
    """
    import numpy as np

    source = np.arange(4096)
    x = source & 15
    z = (source >> 4) & 15
    y = (source >> 8) & 15
    return (x << 8) | (z << 4) | y


_TO_SHARED_ORDER = None


def _section_blocks(section, data_version: int):
    """(palette names, 4096 indices) for one chunk section, or None if it has no blocks."""
    import numpy as np

    global _TO_SHARED_ORDER
    if _TO_SHARED_ORDER is None:
        _TO_SHARED_ORDER = _java_to_shared_order()

    if "block_states" in section:                     # 1.18+
        states = section["block_states"]
        palette = states.get("palette")
        packed = states.get("data")
    else:                                             # 1.13 - 1.17
        palette = section.get("Palette")
        packed = section.get("BlockStates")
    if not palette:
        return None
    if len(palette) > MAX_PALETTE:
        raise WorldError("A chunk section declares {} palette entries.".format(len(palette)),
                         code="bad_palette_size", stage="section")

    names = []
    for entry in palette:
        name = entry.get("Name") if hasattr(entry, "get") else None
        names.append(str(name) if name else "minecraft:air")

    if packed is None or len(packed) == 0:
        # A section with one palette entry and no data is uniformly that block, and a
        # uniform section needs no reordering.
        return names, np.zeros(4096, dtype=np.uint16)

    bits = max(4, (len(names) - 1).bit_length())
    values = _unpack_indices(packed, bits, data_version >= PADDED_SINCE)

    reordered = np.zeros(4096, dtype=np.uint16)
    reordered[_TO_SHARED_ORDER] = values
    return names, reordered


def _iter_region_chunks(data: bytes, region_x: int, region_z: int) -> Iterator[tuple]:
    """Yield (chunk x, chunk z, root NBT) for every chunk present in a region file."""
    if len(data) < 8192:
        return
    for index in range(1024):
        offset = int.from_bytes(data[index * 4:index * 4 + 3], "big")
        sectors = data[index * 4 + 3]
        if not offset or not sectors:
            continue
        start = offset * 4096
        if start + 5 > len(data):
            continue
        length = int.from_bytes(data[start:start + 4], "big")
        if length < 1 or start + 4 + length > len(data):
            continue
        blob = _decompress_chunk(data[start + 5:start + 4 + length], data[start + 4])
        try:
            import nbtlib
            parsed = nbtlib.File.parse(io.BytesIO(blob))
        except Exception as exc:
            raise WorldError("Could not read a chunk: {}".format(exc),
                             code="bad_chunk_nbt", stage="region") from exc
        root = parsed[""] if "" in parsed else parsed
        yield (region_x * 32 + (index % 32), region_z * 32 + (index // 32), root)


# ── Opening a world ───────────────────────────────────────────────────────────

def looks_like_java_world(names) -> bool:
    """True when a list of archive entry names contains a Java world."""
    has_level = any(n.endswith("level.dat") for n in names)
    has_region = any(REGION_RE.match(n.rsplit("/", 1)[-1]) for n in names)
    return has_level and has_region


def find_world_root(names) -> Optional[str]:
    """The prefix inside an archive that holds the world, e.g. 'Suburban House/'.

    Worlds are usually zipped with their folder, but sometimes from inside it. Anchoring on
    `region/r.*.mca` finds either.
    """
    for name in names:
        parts = name.split("/")
        if len(parts) >= 2 and parts[-2] == "region" and REGION_RE.match(parts[-1]):
            return "/".join(parts[:-2])
    return None


def _read_level_name(payload: bytes) -> str:
    try:
        import nbtlib
        parsed = nbtlib.File.parse(io.BytesIO(gzip.decompress(payload)))
        root = parsed[""] if "" in parsed else parsed
        data = root.get("Data", root)
        return str(data.get("LevelName", "") or "")
    except Exception:
        return ""


def _data_version(payload: bytes) -> int:
    try:
        import nbtlib
        parsed = nbtlib.File.parse(io.BytesIO(gzip.decompress(payload)))
        root = parsed[""] if "" in parsed else parsed
        data = root.get("Data", root)
        return int(data.get("DataVersion", 0))
    except Exception:
        return 0


def open_world(path) -> JavaWorld:
    """Read a zipped Java world into memory, refusing anything oversized or malformed."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise WorldError("Not a zipped Minecraft world (it isn't a zip archive).",
                         code="not_a_zip", stage="read")

    subchunks = {}
    level_name = ""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise WorldError("Archive holds {:,} entries.".format(len(infos)),
                             code="too_many_entries", stage="read")
        total = sum(info.file_size for info in infos)
        if total > MAX_ZIP_UNCOMPRESSED:
            raise WorldError("Archive expands to {:,} bytes.".format(total),
                             code="archive_too_large", stage="read")

        names = [info.filename for info in infos]
        root = find_world_root(names)
        if root is None:
            raise WorldError("No Minecraft world inside this zip (expected region/*.mca).",
                             code="no_region_files", stage="read")
        prefix = (root + "/") if root else ""

        # Names are matched, never joined onto a path: an entry called "../../etc/passwd"
        # is simply not one of the names looked for.
        level_entry = prefix + "level.dat"
        data_version = 0
        for info in infos:
            if info.filename == level_entry and info.file_size < 8 * 1024 * 1024:
                payload = archive.read(info.filename)
                level_name = _read_level_name(payload)
                data_version = _data_version(payload)
                break

        regions = []
        for info in infos:
            parts = info.filename.split("/")
            if len(parts) < 2 or parts[-2] != "region":
                continue
            match = REGION_RE.match(parts[-1])
            if not match or "/".join(parts[:-2]) != root:
                continue
            if info.file_size > MAX_REGION_BYTES:
                raise WorldError("Region file is {:,} bytes.".format(info.file_size),
                                 code="region_too_large", stage="read")
            regions.append((int(match.group(1)), int(match.group(2)), info.filename))
        if not regions:
            raise WorldError("This world has no region files.",
                             code="no_region_files", stage="read")
        if len(regions) > MAX_REGIONS:
            raise WorldError("World has {:,} region files.".format(len(regions)),
                             code="too_many_regions", stage="read")

        for region_x, region_z, name in sorted(regions):
            data = archive.read(name)
            for chunk_x, chunk_z, root_nbt in _iter_region_chunks(data, region_x, region_z):
                # DataVersion is per chunk in Java, and a world can hold chunks written by
                # different versions; prefer the chunk's own, fall back to the level's.
                version = int(root_nbt.get("DataVersion", 0)) or data_version
                level = root_nbt.get("Level", root_nbt)
                sections = root_nbt.get("sections")
                if sections is None:
                    sections = level.get("Sections") or []
                for section in sections:
                    y = section.get("Y")
                    if y is None:
                        continue
                    decoded = _section_blocks(section, version)
                    if decoded is None:
                        continue
                    names_, indices = decoded
                    if len(names_) == 1 and names_[0] in ("minecraft:air",):
                        continue
                    subchunks[(chunk_x, chunk_z, int(y))] = (names_, indices)

    if not subchunks:
        raise WorldError("This world contains no readable block data.",
                         code="no_chunks", stage="region")
    return JavaWorld(subchunks, level_name)
