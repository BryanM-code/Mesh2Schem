"""Builders for synthetic schematic files used by the tests.

Fixtures are generated rather than committed so the tests stay readable: every block in
them is placed by a line you can point at, which is what makes an assertion about the
converted output meaningful.
"""
from __future__ import annotations

import gzip
import io
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import nbtlib
from nbtlib import ByteArray, Compound, Int, List, Short, String


def write_nbt(root: Compound, path, root_name: str = "") -> str:
    """Gzip a compound to `path` under the given NBT root name."""
    payload = io.BytesIO()
    handle = nbtlib.File(root)
    handle.root_name = root_name
    handle.write(payload)
    with gzip.open(str(path), "wb") as fh:
        fh.write(payload.getvalue())
    return str(path)


def index(x: int, y: int, z: int, w: int, length: int) -> int:
    """Schematic block order is YZX, shared by both formats."""
    return (y * length + z) * w + x


def legacy_house(path) -> str:
    """A 5x4x5 MCEdit schematic covering every conversion path worth testing.

    Contains: plain blocks, an axis-rotated log, upside-down stairs, a door (both
    halves), a double plant (both halves), a chest, a wall banner and a flower pot
    (both tile-entity backed), a modded block id carried in AddBlocks, and an entity.
    """
    w, h, length = 5, 4, 5
    volume = w * h * length
    blocks = np.zeros(volume, dtype=np.uint16)
    data = np.zeros(volume, dtype=np.uint8)

    def put(x, y, z, block_id, meta=0):
        i = index(x, y, z, w, length)
        blocks[i] = block_id
        data[i] = meta

    for x in range(w):
        for z in range(length):
            put(x, 0, z, 4)                                  # cobblestone floor
    for y in (1, 2):
        for x in range(w):
            for z in range(length):
                if x in (0, w - 1) or z in (0, length - 1):
                    put(x, y, z, 5)                          # oak plank walls

    put(2, 1, 0, 64, 3)              # oak door, lower half, facing north
    put(2, 2, 0, 64, 8 | 1)          # oak door, upper half, hinge right
    put(1, 3, 1, 17, 1 | 4)          # spruce log laid along X
    put(1, 1, 1, 54, 5)              # chest facing east
    put(3, 2, 1, 177, 3)             # wall banner facing south
    put(3, 1, 1, 140)                # flower pot
    put(2, 1, 2, 175, 0)             # sunflower, lower half
    put(2, 2, 2, 175, 8)             # sunflower, upper half
    put(1, 1, 3, 53, 1 | 4)          # oak stairs facing west, upside down
    put(3, 3, 3, 700)                # a modded id, only reachable via AddBlocks

    add = np.zeros((volume + 1) // 2, dtype=np.uint8)
    for i in range(volume):
        high = (blocks[i] >> 8) & 0x0F
        if high:
            add[i // 2] |= high if i % 2 == 0 else high << 4

    root = Compound({
        "Width": Short(w), "Height": Short(h), "Length": Short(length),
        "Materials": String("Alpha"),
        "Blocks": ByteArray((blocks & 0xFF).astype(np.int8)),
        "Data": ByteArray(data.astype(np.int8)),
        "AddBlocks": ByteArray(add.astype(np.int8)),
        "Entities": List[Compound]([Compound({"id": String("ArmorStand")})]),
        "TileEntities": List[Compound]([
            Compound({"id": String("Chest"), "x": Int(1), "y": Int(1), "z": Int(1),
                      "Items": List[Compound]([])}),
            Compound({"id": String("Banner"), "x": Int(3), "y": Int(2), "z": Int(1),
                      "Base": Int(11)}),          # dye 11 -> yellow
            Compound({"id": String("FlowerPot"), "x": Int(3), "y": Int(1), "z": Int(1),
                      "Item": String("minecraft:red_flower"), "Data": Int(4)}),
        ]),
        "WEOffsetX": Int(-2), "WEOffsetY": Int(0), "WEOffsetZ": Int(-2),
    })
    return write_nbt(root, path, "Schematic")


def sponge_v2(path, w: int = 2, h: int = 2, length: int = 2) -> str:
    """A minimal, already-modern .schem: alternating air and stone."""
    volume = w * h * length
    block_data = bytes((i % 2) for i in range(volume))
    root = Compound({
        "Version": Int(2), "DataVersion": Int(1631),
        "Width": Short(w), "Height": Short(h), "Length": Short(length),
        "Palette": Compound({"minecraft:air": Int(0), "minecraft:stone": Int(1)}),
        "PaletteMax": Int(2),
        "BlockData": ByteArray(np.frombuffer(block_data, dtype=np.uint8).astype(np.int8)),
        "BlockEntities": List[Compound]([]),
        "Metadata": Compound({}),
    })
    return write_nbt(root, path, "")


def decode_sponge(path):
    """Read a .schem back as {(x, y, z): blockstate} plus its root compound."""
    import nbt_safe

    root = nbt_safe.load(path)
    w, h, length = int(root["Width"]), int(root["Height"]), int(root["Length"])
    names = {int(v): str(k) for k, v in root["Palette"].items()}

    values, current, shift = [], 0, 0
    for byte in root["BlockData"].tobytes():
        current |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
        else:
            values.append(current)
            current, shift = 0, 0

    blocks = {}
    for y in range(h):
        for z in range(length):
            for x in range(w):
                blocks[(x, y, z)] = names[values[index(x, y, z, w, length)]]
    return blocks, root


# ── Hostile files ─────────────────────────────────────────────────────────────

def gzipped(path, payload: bytes) -> str:
    with gzip.open(str(path), "wb") as fh:
        fh.write(payload)
    return str(path)


def raw(path, payload: bytes) -> str:
    with open(str(path), "wb") as fh:
        fh.write(payload)
    return str(path)


def gzip_bomb(path, megabytes: int = 400) -> str:
    """A small file that expands to `megabytes` of zeros."""
    with gzip.open(str(path), "wb") as fh:
        chunk = b"\x00" * (1 << 20)
        for _ in range(megabytes):
            fh.write(chunk)
    return str(path)


def huge_list_nbt() -> bytes:
    """NBT declaring a two-billion-entry list it does not contain.

    This is the shape that sends nbtlib into an unbounded loop: its reader returns 0 at
    EOF rather than raising, so it happily iterates the declared count.
    """
    return (b"\x0a\x00\x00" + b"\x09" + b"\x00\x04" + b"list"
            + b"\x01" + struct.pack(">i", 2_000_000_000))


def deep_nbt(levels: int = 10_000) -> bytes:
    """Compounds nested `levels` deep."""
    return b"\x0a\x00\x00" + (b"\x0a\x00\x01x" * levels) + b"\x00" * (levels + 1)


def huge_array_nbt() -> bytes:
    """A byte array claiming two billion bytes it does not have."""
    return (b"\x0a\x00\x00" + b"\x07" + b"\x00\x06Blocks"
            + struct.pack(">i", 2_000_000_000) + b"\x01\x02")
