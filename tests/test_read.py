"""Reading schematics: format detection, the legacy (pre-1.13) format, and files that must be
rejected before they are parsed."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import schem_fixtures as fx  # noqa: E402
from dataset.read import ReadError, load_build  # noqa: E402
from minecraft import legacy_blocks, nbt_safe  # noqa: E402
from minecraft.nbt_safe import NbtError  # noqa: E402


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="read-tests-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class FormatTests(TempDirTest):
    def test_detects_legacy_schematic(self):
        build = load_build(fx.legacy_house(self.tmp / "house.schematic"))
        self.assertEqual(build.source_format, "mcedit")
        self.assertEqual(build.dims, (5, 4, 5))

    def test_detects_sponge_schem(self):
        build = load_build(fx.sponge_v2(self.tmp / "small.schem", 3, 2, 4))
        self.assertEqual(build.source_format, "sponge_v2")
        self.assertEqual(build.dims, (3, 2, 4))

    def test_format_comes_from_contents_not_extension(self):
        """A legacy file named .schem is still legacy, and vice versa."""
        self.assertEqual(load_build(fx.legacy_house(self.tmp / "legacy.schem")).source_format, "mcedit")
        self.assertEqual(load_build(fx.sponge_v2(self.tmp / "sponge.schematic")).source_format, "sponge_v2")

    def test_valid_nbt_that_is_not_a_schematic_is_rejected(self):
        from nbtlib import Compound, Int, String
        path = fx.write_nbt(Compound({"Health": Int(20), "Name": String("Steve")}), self.tmp / "player.schem")
        with self.assertRaises(ReadError):
            load_build(path)


class LegacyReadingTests(TempDirTest):
    """The legacy format stores numeric ids; they are read as modern block states."""

    def block(self, x, y, z):
        build = load_build(fx.legacy_house(self.tmp / "house.schematic"))
        return build.palette[build.blocks[x, y, z]]

    def test_plain_blocks(self):
        self.assertEqual(self.block(0, 0, 0), "minecraft:cobblestone")
        self.assertEqual(self.block(0, 1, 1), "minecraft:oak_planks")
        self.assertEqual(self.block(2, 3, 2), "minecraft:air")
        self.assertEqual(self.block(1, 3, 1), "minecraft:spruce_log[axis=x]")
        self.assertEqual(self.block(1, 1, 3), "minecraft:oak_stairs[facing=west,half=top]")
        self.assertEqual(self.block(1, 1, 1), "minecraft:chest[facing=east]")

    def test_door_upper_half_inherits_facing_from_the_half_below(self):
        """Legacy files store facing only on the lower half; a default would face doors wrong."""
        self.assertEqual(self.block(2, 1, 0), "minecraft:oak_door[facing=north,half=lower,open=false]")
        self.assertEqual(self.block(2, 2, 0),
                         "minecraft:oak_door[facing=north,half=upper,hinge=right,open=false,powered=false]")

    def test_double_plant_upper_half_inherits_its_species(self):
        self.assertEqual(self.block(2, 1, 2), "minecraft:sunflower[half=lower]")
        self.assertEqual(self.block(2, 2, 2), "minecraft:sunflower[half=upper]")

    def test_tile_entities_refine_the_blocks_they_describe(self):
        """Banner colour and pot contents live in the tile entity, not the block data."""
        self.assertEqual(self.block(3, 2, 1), "minecraft:yellow_wall_banner[facing=south]")
        self.assertEqual(self.block(3, 1, 1), "minecraft:potted_red_tulip")

    def test_unknown_ids_keep_their_volume_and_are_reported(self):
        build = load_build(fx.legacy_house(self.tmp / "house.schematic"))
        self.assertEqual(build.palette[build.blocks[3, 3, 3]], "legacy:unknown_700")
        self.assertEqual(build.unknown_legacy, {"700:0": 1})

    def test_entities_are_reported(self):
        build = load_build(fx.legacy_house(self.tmp / "house.schematic"))
        self.assertTrue(any("entities" in w for w in build.warnings))


class HostileInputTests(TempDirTest):
    """Schematics come off the internet: every one of these must fail before parsing.

    Assertions are on `code`, not on the message: prose gets reworded, codes don't.
    """

    def assertRejected(self, path, code: str, *, stage=None, nbt_path=None, offset=None):
        with self.assertRaises(NbtError) as caught:
            nbt_safe.load(path)
        error = caught.exception
        self.assertEqual(error.code, code, msg=str(error))
        if stage is not None:
            self.assertEqual(error.stage, stage, msg=str(error))
        if nbt_path is not None:
            self.assertEqual(error.path, nbt_path, msg=str(error))
        if offset is not None:
            self.assertEqual(error.offset, offset, msg=str(error))
        return error

    def test_rejects_files_that_are_not_nbt(self):
        for name, payload in [("exe", b"MZ\x90\x00" + b"\x00" * 512),
                              ("html", b"<!DOCTYPE html><h1>404</h1>"),
                              ("zip", b"PK\x03\x04" + b"\x00" * 64)]:
            with self.subTest(name):
                self.assertRejected(fx.raw(self.tmp / (name + ".schem"), payload),
                                    "bad_magic", stage="decompress", offset=0)

    def test_rejects_empty_file(self):
        self.assertRejected(fx.raw(self.tmp / "d.schem", b""), "empty_file", stage="read")

    def test_rejects_decompression_bomb(self):
        self.assertRejected(fx.gzip_bomb(self.tmp / "bomb.schem"), "decompression_bomb", stage="decompress")

    def test_rejects_oversized_list_without_hanging(self):
        """The specific shape that makes nbtlib loop forever instead of raising."""
        self.assertRejected(fx.gzipped(self.tmp / "list.schem", fx.huge_list_nbt()),
                            "list_too_long", stage="structure", nbt_path="list")

    def test_rejects_deeply_nested_nbt(self):
        error = self.assertRejected(fx.gzipped(self.tmp / "deep.schem", fx.deep_nbt()), "too_deep",
                                    stage="structure")
        self.assertIn("...", error.path)               # a deep path is elided, not dumped
        self.assertLess(len(error.path), 60)

    def test_rejects_array_longer_than_the_file(self):
        self.assertRejected(fx.gzipped(self.tmp / "arr.schem", fx.huge_array_nbt()),
                            "truncated", stage="structure", nbt_path="Blocks")

    def test_rejects_a_bad_tag_id_naming_the_tag(self):
        payload = bytearray(b"\x0a\x00\x00")
        payload += b"\x2a" + b"\x00\x05Width" + b"\x00\x04"   # tag id 42 does not exist
        self.assertRejected(fx.gzipped(self.tmp / "tag.schem", bytes(payload)),
                            "unknown_tag", stage="structure", nbt_path="Width")

    def test_the_reader_reports_a_rejected_file_as_a_read_error(self):
        with self.assertRaises(ReadError):
            load_build(fx.raw(self.tmp / "e.schem", b""))


class RejectionLocationTests(TempDirTest):
    """A rejection has to say *where*, or a bad download can't be diagnosed."""

    def corrupted_house(self):
        """A valid schematic with one byte inside TileEntities[2].Items[1] broken."""
        import io
        import nbtlib
        from nbtlib import ByteArray, Compound, Int, List, Short, String
        root = Compound({
            "Width": Short(2), "Height": Short(2), "Length": Short(2),
            "Blocks": ByteArray([1] * 8), "Data": ByteArray([0] * 8),
            "TileEntities": List[Compound]([
                Compound({"id": String("Chest"), "x": Int(0), "y": Int(0), "z": Int(0)}),
                Compound({"id": String("Chest"), "x": Int(1), "y": Int(0), "z": Int(0)}),
                Compound({"id": String("Chest"), "x": Int(1), "y": Int(1), "z": Int(0),
                          "Items": List[Compound]([
                              Compound({"Slot": Int(0)}),
                              Compound({"Slot": Int(1), "Count": Int(64)}),
                          ])}),
            ]),
        })
        buf = io.BytesIO()
        handle = nbtlib.File(root)
        handle.root_name = "Schematic"
        handle.write(buf)
        data = bytearray(buf.getvalue())
        broken_at = data.index(b"\x05Count") - 2      # the tag id byte of "Count"
        data[broken_at] = 0x2a                        # not a real NBT tag id
        return fx.gzipped(self.tmp / "corrupt.schematic", bytes(data)), broken_at

    def rejection(self):
        path, broken_at = self.corrupted_house()
        with self.assertRaises(NbtError) as caught:
            nbt_safe.load(path)
        return caught.exception, broken_at

    def test_points_at_the_exact_tag_that_is_broken(self):
        error, broken_at = self.rejection()
        self.assertEqual(error.code, "unknown_tag")
        self.assertEqual(error.path, "TileEntities[2].Items[1].Count")
        self.assertGreaterEqual(error.offset, broken_at)

    def test_str_includes_the_location_but_message_stays_clean(self):
        error, _ = self.rejection()
        self.assertIn("TileEntities[2].Items[1].Count", str(error))
        self.assertTrue(str(error).startswith(error.message))
        self.assertNotIn("(at ", error.message)

    def test_details_are_json_friendly(self):
        details = self.rejection()[0].details()
        self.assertEqual(set(details), {"error", "message", "code", "stage", "offset", "path"})
        self.assertEqual(json.loads(json.dumps(details)), details)


class LegacyBlockTableTests(unittest.TestCase):
    def test_every_legacy_id_and_data_pair_resolves_or_returns_none(self):
        """No id/data combination may raise: schematics carry junk in unused bits."""
        for block_id in range(4096):
            for data in range(16):
                state = legacy_blocks.resolve(legacy_blocks.key(block_id, data))
                self.assertTrue(state is None or state.startswith("minecraft:"))

    def test_properties_are_emitted_in_a_stable_order(self):
        self.assertEqual(legacy_blocks.blockstate("x", b=1, a=2), "minecraft:x[a=2,b=1]")

    def test_banner_base_is_the_inverse_of_the_wool_order(self):
        self.assertEqual(legacy_blocks.banner_state(0, 15), "minecraft:white_banner[rotation=0]")
        self.assertEqual(legacy_blocks.banner_state(0, 0), "minecraft:black_banner[rotation=0]")

    def test_unrecognised_pot_contents_leave_the_pot_alone(self):
        self.assertIsNone(legacy_blocks.potted_state("minecraft:diamond_block"))
        self.assertEqual(legacy_blocks.potted_state("minecraft:cactus"), "minecraft:potted_cactus")


if __name__ == "__main__":
    unittest.main()
