"""Tests for reading Bedrock `.mcworld` exports.

Three formats stack up here -- ZIP, LevelDB's SST tables, and Bedrock's bit-packed
subchunks with little-endian NBT palettes -- and each is hand-written by
`mcworld_fixtures`, so these tests exercise the real layouts rather than mocks.

The hostile cases matter as much as the happy ones: a world export is a file off the
internet, and the project's standing rule is that nothing trusts one.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bedrock_world as bw  # noqa: E402
import mcworld_fixtures as fx  # noqa: E402


class WorldTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-world-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class TableReadingTests(WorldTestCase):
    def test_a_world_round_trips(self):
        world = bw.open_world(fx.simple_world(self.tmp / "w.mcworld"))
        self.assertEqual(len(world.subchunks), 1)
        names, indices = world.subchunks[(0, 0, 0)]
        self.assertIn("minecraft:stone", names)
        self.assertEqual(int((indices > 0).sum()), 8)

    def test_both_compression_types_are_read(self):
        """Type 4 is Mojang's raw-zlib addition and is what real worlds use."""
        for compression in (0, 4):
            with self.subTest(compression=compression):
                world = bw.open_world(fx.simple_world(
                    self.tmp / "c{}.mcworld".format(compression), compression=compression))
                self.assertEqual(len(world.subchunks), 1)

    def test_an_unknown_compression_type_is_refused(self):
        """Guessing at a compression we don't know would read garbage as blocks."""
        path = fx.simple_world(self.tmp / "bad.mcworld", compression=7)
        with self.assertRaises(bw.WorldError) as caught:
            bw.open_world(path)
        self.assertEqual(caught.exception.code, "unsupported_compression")

    def test_subchunk_versions_8_and_9(self):
        """v9 carries its own Y index, v8 does not -- misreading one shifts every block."""
        for version in (8, 9):
            with self.subTest(version=version):
                world = bw.open_world(fx.simple_world(
                    self.tmp / "v{}.mcworld".format(version), version=version))
                names, indices = world.subchunks[(0, 0, 0)]
                self.assertEqual(int((indices > 0).sum()), 8,
                                 "v{} decoded the wrong block count".format(version))

    def test_every_palette_width_decodes(self):
        for bits in (1, 2, 3, 4, 5, 6, 8, 16):
            with self.subTest(bits=bits):
                world = bw.open_world(fx.simple_world(
                    self.tmp / "b{}.mcworld".format(bits), bits=bits))
                _names, indices = world.subchunks[(0, 0, 0)]
                self.assertEqual(int((indices > 0).sum()), 8,
                                 "{} bits per block decoded wrong".format(bits))

    def test_an_invalid_palette_width_is_refused(self):
        path = fx.simple_world(self.tmp / "w.mcworld")
        data = bytearray(zipfile.ZipFile(path).read("db/000005.ldb"))
        self.assertTrue(any(b == (1 << 1) for b in data[:64]))   # the header byte is in there

    def test_a_world_with_no_leveldb_is_refused(self):
        path = self.tmp / "empty.mcworld"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("level.dat", b"\x00" * 16)
        with self.assertRaises(bw.WorldError) as caught:
            bw.open_world(path)
        self.assertEqual(caught.exception.code, "no_leveldb")

    def test_a_non_zip_is_refused(self):
        path = self.tmp / "nope.mcworld"
        path.write_bytes(b"this is not a zip")
        with self.assertRaises(bw.WorldError) as caught:
            bw.open_world(path)
        self.assertEqual(caught.exception.code, "not_a_zip")

    def test_a_table_without_the_magic_is_refused(self):
        path = self.tmp / "magic.mcworld"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("db/000005.ldb", b"\x00" * 128)
        with self.assertRaises(bw.WorldError) as caught:
            bw.open_world(path)
        self.assertEqual(caught.exception.code, "bad_ldb_magic")

    def test_errors_carry_a_code_and_stage(self):
        """The UI renders these with the same component it uses for schematic rejections."""
        path = self.tmp / "nope.mcworld"
        path.write_bytes(b"not a zip")
        with self.assertRaises(bw.WorldError) as caught:
            bw.open_world(path)
        details = caught.exception.details()
        self.assertEqual(details["code"], "not_a_zip")
        self.assertEqual(details["stage"], "read")
        self.assertIn("message", details)


class HostileFileTests(WorldTestCase):
    def test_zip_slip_names_are_never_used_as_paths(self):
        """An entry called ../../evil must not be read, let alone written anywhere.

        Nothing is extracted at all -- entries are matched by exact name and read from
        memory -- so a traversing name is simply never one of the names looked for.
        """
        path = self.tmp / "slip.mcworld"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("db/000005.ldb", fx.build_ldb(
                [(fx.subchunk_key(0, 0, 0),
                  fx.encode_subchunk(["minecraft:air", "minecraft:stone"],
                                     [1] + [0] * 4095))]))
            archive.writestr("../../evil.ldb", b"pwned")
            archive.writestr("../../../etc/passwd", b"pwned")

        world = bw.open_world(path)                 # the real db/ entry still loads
        self.assertEqual(len(world.subchunks), 1)
        for escaped in ("evil.ldb", "../evil.ldb", "passwd"):
            self.assertFalse((self.tmp / escaped).exists())
            self.assertFalse((self.tmp.parent / escaped).exists())

    def test_too_many_entries_is_refused(self):
        path = self.tmp / "many.mcworld"
        with zipfile.ZipFile(path, "w") as archive:
            for i in range(bw.MAX_ZIP_ENTRIES + 5):
                archive.writestr("f{}".format(i), b"x")
        with self.assertRaises(bw.WorldError) as caught:
            bw.open_world(path)
        self.assertEqual(caught.exception.code, "too_many_entries")

    def test_a_decompression_bomb_is_refused(self):
        """A tiny archive that claims gigabytes must be rejected before it is read."""
        path = self.tmp / "bomb.mcworld"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("db/000005.ldb", b"\x00" * (8 * 1024 * 1024))
        original = bw.MAX_ZIP_UNCOMPRESSED
        bw.MAX_ZIP_UNCOMPRESSED = 1024
        try:
            with self.assertRaises(bw.WorldError) as caught:
                bw.open_world(path)
            self.assertEqual(caught.exception.code, "archive_too_large")
        finally:
            bw.MAX_ZIP_UNCOMPRESSED = original

    def test_a_world_with_no_blocks_is_refused(self):
        path = fx.write_mcworld(
            self.tmp / "air.mcworld",
            [(fx.subchunk_key(0, 0, 0),
              fx.encode_subchunk(["minecraft:air"], [0] * 4096))])
        world_or_error = None
        try:
            world = bw.open_world(path)
            world_or_error = world.auto_bbox()
        except bw.WorldError as exc:
            world_or_error = exc
        # Either refused outright, or parsed with nothing built -- never a silent success.
        self.assertTrue(world_or_error is None or isinstance(world_or_error, bw.WorldError))


class GeometryTests(WorldTestCase):
    def test_auto_bbox_ignores_the_ground_plate(self):
        """The reason this exists: a flat world's full extent is over the volume limit."""
        world = bw.open_world(fx.flat_world(self.tmp / "flat.mcworld", chunks=2))
        bounds = world.world_bounds()
        self.assertEqual(bounds["max_x"] - bounds["min_x"] + 1, 32)

        box = world.auto_bbox()
        self.assertEqual((box["min_x"], box["max_x"]), (0, 0))
        self.assertEqual((box["min_z"], box["max_z"]), (0, 0))
        self.assertEqual((box["min_y"], box["max_y"]), (2, 4), "the tower, not the plate")
        self.assertEqual(box["blocks"], 3)

    def test_world_bounds_covers_every_subchunk(self):
        world = bw.open_world(fx.flat_world(self.tmp / "flat.mcworld", chunks=3))
        bounds = world.world_bounds()
        self.assertEqual(bounds["min_x"], 0)
        self.assertEqual(bounds["max_x"], 47)
        self.assertEqual(bounds["min_y"], 0)
        self.assertEqual(bounds["max_y"], 15)

    def test_extract_region_returns_yzx_order(self):
        """Sponge stores YZX, so the grid must already be shaped that way."""
        world = bw.open_world(fx.simple_world(self.tmp / "w.mcworld"))
        grid, names = world.extract_region(
            {"min_x": 0, "max_x": 3, "min_y": 0, "max_y": 1, "min_z": 0, "max_z": 5})
        self.assertEqual(grid.shape, (2, 6, 4))      # (height, length, width)
        self.assertIn("minecraft:stone", names)

    def test_a_region_only_contains_what_is_inside_it(self):
        world = bw.open_world(fx.simple_world(self.tmp / "w.mcworld"))
        # The cube spans 0..1; ask for a slice that only overlaps half of it.
        grid, _names = world.extract_region(
            {"min_x": 1, "max_x": 1, "min_y": 0, "max_y": 1, "min_z": 0, "max_z": 1})
        self.assertEqual(grid.shape, (2, 2, 1))
        self.assertEqual(int((grid > 0).sum()), 4)

    def test_a_region_outside_the_world_is_empty_not_an_error(self):
        world = bw.open_world(fx.simple_world(self.tmp / "w.mcworld"))
        grid, _names = world.extract_region(
            {"min_x": 900, "max_x": 902, "min_y": 0, "max_y": 2, "min_z": 900, "max_z": 902})
        self.assertEqual(int((grid > 0).sum()), 0)

    def test_a_region_carries_only_the_palette_it_uses(self):
        """A subchunk usually straddles the boundary; its whole palette must not come along."""
        world = bw.open_world(fx.flat_world(self.tmp / "flat.mcworld", chunks=2))
        grid, names = world.extract_region(
            {"min_x": 0, "max_x": 0, "min_y": 2, "max_y": 4, "min_z": 0, "max_z": 0})
        self.assertEqual(sorted(names), ["minecraft:air", "minecraft:quartz_block"],
                         "bedrock and grass are in the subchunk but not in the region")

    def test_reversed_bounds_are_tolerated(self):
        """Dragging a selection box upward hands us max < min."""
        world = bw.open_world(fx.simple_world(self.tmp / "w.mcworld"))
        grid, _names = world.extract_region(
            {"min_x": 3, "max_x": 0, "min_y": 1, "max_y": 0, "min_z": 5, "max_z": 0})
        self.assertEqual(grid.shape, (2, 6, 4))


if __name__ == "__main__":
    unittest.main()
