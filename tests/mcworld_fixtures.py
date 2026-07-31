"""Build synthetic Bedrock `.mcworld` files in code, so no binaries are checked in.

Same principle as `schem_fixtures`: the tests need real files in a real format, and
generating them is both more honest and more maintainable than committing a sample world.
Writing one means writing a LevelDB SST table by hand -- which is also the clearest
statement of what `bedrock_world` claims the format is.
"""
from __future__ import annotations

import io
import struct
import zipfile
import zlib

SST_MAGIC = 0xDB4775248B80FB57
TAG_SUBCHUNK = 0x2F

__all__ = ["encode_subchunk", "build_ldb", "write_mcworld", "simple_world", "flat_world"]


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        out.append(chunk | 0x80 if value else chunk)
        if not value:
            return bytes(out)


def _nbt_string_le(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<H", len(raw)) + raw


def _nbt_block_entry(name: str) -> bytes:
    """One palette entry: a little-endian NBT compound of {name, states, version}."""
    out = bytearray()
    out.append(10)                       # TAG_Compound
    out += _nbt_string_le("")            # root name
    out.append(8)                        # TAG_String
    out += _nbt_string_le("name")
    out += _nbt_string_le(name)
    out.append(10)                       # TAG_Compound "states", left empty
    out += _nbt_string_le("states")
    out.append(0)
    out.append(3)                        # TAG_Int "version"
    out += _nbt_string_le("version")
    out += struct.pack("<i", 18090528)
    out.append(0)                        # end of root compound
    return bytes(out)


def encode_subchunk(palette, indices, version=9, sub_y=0, bits=None) -> bytes:
    """Pack a subchunk the way Bedrock stores one.

    `indices` is 4096 palette indices in (x << 8) | (z << 4) | y order.
    """
    assert len(indices) == 4096, "a subchunk is 4096 blocks"
    if bits is None:
        bits = 1
        while (1 << bits) < len(palette) and bits < 16:
            bits = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 8: 16}[bits]

    out = bytearray()
    out.append(version)
    out.append(1)                        # one storage layer
    if version == 9:
        out += struct.pack("<b", sub_y)
    out.append((bits << 1) | 0)          # persistent (NBT) palette, not runtime ids

    per_word = 32 // bits
    words = -(-4096 // per_word)
    for w in range(words):
        word = 0
        for slot in range(per_word):
            position = w * per_word + slot
            if position >= 4096:
                break
            word |= (indices[position] & ((1 << bits) - 1)) << (slot * bits)
        out += struct.pack("<I", word)

    out += struct.pack("<i", len(palette))
    for name in palette:
        out += _nbt_block_entry(name)
    return bytes(out)


def build_ldb(records, compression=0) -> bytes:
    """Write a minimal single-block LevelDB SST table.

    Real tables split into many blocks; one is enough to exercise the reader, and the
    footer/index/block layout is identical either way. Keys are written as LevelDB's
    *internal* keys -- the user key plus an 8-byte sequence/type trailer -- because that is
    what the reader has to strip.
    """
    def make_block(entries):
        body = bytearray()
        previous = b""
        restarts = []
        for key, value in entries:
            restarts.append(len(body))
            shared = 0
            if restarts and previous:
                limit = min(len(previous), len(key))
                while shared < limit and previous[shared] == key[shared]:
                    shared += 1
            shared = 0                    # a restart every entry: simple and legal
            body += _varint(shared) + _varint(len(key) - shared) + _varint(len(value))
            body += key[shared:] + value
            previous = key
        for offset in restarts:
            body += struct.pack("<I", offset)
        body += struct.pack("<I", len(restarts))
        return bytes(body)

    def emit(block):
        if compression == 0:
            payload = block
        elif compression == 2:
            payload = zlib.compress(block)
        elif compression == 4:
            deflate = zlib.compressobj(9, zlib.DEFLATED, -15)
            payload = deflate.compress(block) + deflate.flush()
        else:
            # Stored uncompressed but labelled with a type the reader doesn't know, so a
            # test can check that an unrecognised compression is refused rather than
            # guessed at.
            payload = block
        return payload + bytes([compression]) + struct.pack("<I", 0)

    internal = [(key + struct.pack("<Q", (index + 1) << 8 | 1), value)
                for index, (key, value) in enumerate(sorted(records))]

    out = bytearray()
    data_offset = 0
    data_block = emit(make_block(internal))
    out += data_block
    data_size = len(data_block) - 5

    last_key = internal[-1][0] if internal else b"\xff"
    index_entries = [(last_key, _varint(data_offset) + _varint(data_size))]
    index_offset = len(out)
    index_block = emit(make_block(index_entries))
    out += index_block
    index_size = len(index_block) - 5

    footer = bytearray()
    footer += _varint(0) + _varint(0)                       # metaindex handle (unused)
    footer += _varint(index_offset) + _varint(index_size)
    footer += bytes(40 - len(footer))
    footer += struct.pack("<Q", SST_MAGIC)
    out += footer
    return bytes(out)


def write_mcworld(path, records, extra=None, compression=0) -> str:
    """Zip a LevelDB table into a `.mcworld`."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("level.dat", b"\x00" * 64)
        archive.writestr("db/CURRENT", b"MANIFEST-000001\n")
        archive.writestr("db/MANIFEST-000001", b"\x00" * 32)
        archive.writestr("db/000005.ldb", build_ldb(records, compression))
        for name, payload in (extra or {}).items():
            archive.writestr(name, payload)
    return str(path)


def subchunk_key(cx, cz, sub_y) -> bytes:
    return struct.pack("<ii", cx, cz) + bytes([TAG_SUBCHUNK]) + struct.pack("<b", sub_y)


def simple_world(path, block="minecraft:stone", version=9, compression=0,
                 bits=None, cx=0, cz=0, sub_y=0):
    """One subchunk: air everywhere except a 2x2x2 cube of `block` at its origin."""
    palette = ["minecraft:air", block]
    indices = [0] * 4096
    for x in range(2):
        for y in range(2):
            for z in range(2):
                indices[(x << 8) | (z << 4) | y] = 1
    payload = encode_subchunk(palette, indices, version=version, sub_y=sub_y, bits=bits)
    return write_mcworld(path, [(subchunk_key(cx, cz, sub_y), payload)],
                         compression=compression)


def flat_world(path, chunks=2):
    """Ground plate with a small tower on it, so auto_bbox has something to find.

    Layer y=0 is bedrock and y=1 grass across every chunk; a 3-block quartz column stands
    at (0, 2..4, 0). Only the column should survive `auto_bbox`.
    """
    records = []
    for cx in range(chunks):
        for cz in range(chunks):
            palette = ["minecraft:air", "minecraft:bedrock", "minecraft:grass_block",
                       "minecraft:quartz_block"]
            indices = [0] * 4096
            for x in range(16):
                for z in range(16):
                    indices[(x << 8) | (z << 4) | 0] = 1
                    indices[(x << 8) | (z << 4) | 1] = 2
            if cx == 0 and cz == 0:
                for y in (2, 3, 4):
                    indices[(0 << 8) | (0 << 4) | y] = 3
            records.append((subchunk_key(cx, cz, 0),
                            encode_subchunk(palette, indices, sub_y=0)))
    return write_mcworld(path, records)
