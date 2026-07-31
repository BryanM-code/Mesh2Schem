"""Format detection, legacy conversion, and hostile-input handling."""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import legacy_blocks
import schem_convert
import schem_fixtures as fx
from schem_convert import SchematicError


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-tests-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


# ── Format detection ──────────────────────────────────────────────────────────

class FormatDetectionTests(TempDirTest):
    def test_detects_legacy_schematic(self):
        info = schem_convert.inspect(fx.legacy_house(self.tmp / "house.schematic"))
        self.assertEqual(info.format, "mcedit")
        self.assertTrue(info.is_legacy)
        self.assertEqual(info.dims, (5, 4, 5))

    def test_detects_sponge_schem(self):
        info = schem_convert.inspect(fx.sponge_v2(self.tmp / "small.schem", 3, 2, 4))
        self.assertEqual(info.format, "sponge_v2")
        self.assertFalse(info.is_legacy)
        self.assertEqual(info.dims, (3, 2, 4))

    def test_format_comes_from_contents_not_extension(self):
        """A legacy file named .schem is still legacy, and vice versa."""
        mislabelled = fx.legacy_house(self.tmp / "actually_legacy.schem")
        self.assertEqual(schem_convert.inspect(mislabelled).format, "mcedit")
        modern = fx.sponge_v2(self.tmp / "actually_sponge.schematic")
        self.assertEqual(schem_convert.inspect(modern).format, "sponge_v2")


# ── Conversion ────────────────────────────────────────────────────────────────

class ConversionTests(TempDirTest):
    def convert(self, **kwargs):
        src = fx.legacy_house(self.tmp / "house.schematic")
        result = schem_convert.ensure_sponge(src, self.tmp / "out.schem")
        return result, fx.decode_sponge(result.path)

    def test_sponge_files_pass_through_untouched(self):
        """Modern files must keep their exact bytes, or the library's dedup hash moves."""
        src = Path(fx.sponge_v2(self.tmp / "small.schem"))
        before = src.read_bytes()
        result = schem_convert.ensure_sponge(src, self.tmp / "out.schem")
        self.assertFalse(result.converted)
        self.assertEqual(result.path, src)
        self.assertEqual(src.read_bytes(), before)
        self.assertFalse((self.tmp / "out.schem").exists())

    def test_converted_file_is_a_valid_sponge_schem(self):
        result, (_, root) = self.convert()
        self.assertTrue(result.converted)
        self.assertEqual(int(root["Version"]), 2)
        self.assertEqual(int(root["DataVersion"]), schem_convert.SPONGE_DATA_VERSION)
        self.assertEqual(schem_convert.inspect(result.path).format, "sponge_v2")

    def test_conversion_is_deterministic(self):
        """Same input, same bytes -- otherwise re-adding a file would dodge the dedup."""
        src = fx.legacy_house(self.tmp / "house.schematic")
        digests = set()
        for name in ("first.schem", "second.schem"):
            out = schem_convert.ensure_sponge(src, self.tmp / name).path
            digests.add(hashlib.sha256(out.read_bytes()).hexdigest())
        self.assertEqual(len(digests), 1)

    def test_plain_blocks_and_metadata_survive(self):
        _, (blocks, root) = self.convert()
        self.assertEqual(blocks[(0, 0, 0)], "minecraft:cobblestone")
        self.assertEqual(blocks[(0, 1, 1)], "minecraft:oak_planks")
        self.assertEqual(blocks[(2, 3, 2)], "minecraft:air")
        self.assertEqual(blocks[(1, 3, 1)], "minecraft:spruce_log[axis=x]")
        self.assertEqual(blocks[(1, 1, 3)], "minecraft:oak_stairs[facing=west,half=top]")
        self.assertEqual(blocks[(1, 1, 1)], "minecraft:chest[facing=east]")
        self.assertEqual([int(v) for v in root["Offset"]], [-2, 0, -2])
        self.assertEqual(int(root["Metadata"]["WEOffsetX"]), -2)

    def test_door_upper_half_inherits_facing_from_the_half_below(self):
        """Legacy stored facing only on the lower half; a default here faces doors wrong."""
        _, (blocks, _) = self.convert()
        self.assertEqual(blocks[(2, 1, 0)],
                         "minecraft:oak_door[facing=north,half=lower,open=false]")
        self.assertEqual(
            blocks[(2, 2, 0)],
            "minecraft:oak_door[facing=north,half=upper,hinge=right,open=false,powered=false]")

    def test_double_plant_upper_half_inherits_its_species(self):
        _, (blocks, _) = self.convert()
        self.assertEqual(blocks[(2, 1, 2)], "minecraft:sunflower[half=lower]")
        self.assertEqual(blocks[(2, 2, 2)], "minecraft:sunflower[half=upper]")

    def test_tile_entities_refine_the_blocks_they_describe(self):
        """Banner colour and pot contents live in the tile entity, not the block data."""
        _, (blocks, _) = self.convert()
        self.assertEqual(blocks[(3, 2, 1)], "minecraft:yellow_wall_banner[facing=south]")
        self.assertEqual(blocks[(3, 1, 1)], "minecraft:potted_red_tulip")

    def test_tile_entities_are_rewritten_in_sponge_form(self):
        _, (_, root) = self.convert()
        entities = {str(be["Id"]): [int(v) for v in be["Pos"]] for be in root["BlockEntities"]}
        self.assertEqual(entities["minecraft:chest"], [1, 1, 1])
        self.assertEqual(entities["minecraft:banner"], [3, 2, 1])
        for be in root["BlockEntities"]:
            self.assertNotIn("x", be)  # legacy x/y/z are replaced by Pos

    def test_unmapped_ids_keep_their_volume_and_are_reported(self):
        result, (blocks, _) = self.convert()
        self.assertEqual(blocks[(3, 3, 3)], "minecraft:stone")
        self.assertTrue(any("700:0" in w for w in result.warnings))

    def test_dropped_entities_are_reported(self):
        result, _ = self.convert()
        self.assertTrue(any("entity" in w for w in result.warnings))


# ── Hostile input ─────────────────────────────────────────────────────────────

class HostileInputTests(TempDirTest):
    """These files come off the internet, so every one of these must fail closed.

    Assertions are on `code`, not on the message: prose gets reworded, codes don't.
    """

    def assertRejected(self, path, code: str, *, stage=None, nbt_path=None, offset=None):
        with self.assertRaises(SchematicError) as caught:
            schem_convert.inspect(path)
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
        self.assertRejected(fx.gzip_bomb(self.tmp / "bomb.schem"),
                            "decompression_bomb", stage="decompress")

    def test_rejects_oversized_list_without_hanging(self):
        """The specific shape that makes nbtlib loop forever instead of raising."""
        self.assertRejected(fx.gzipped(self.tmp / "list.schem", fx.huge_list_nbt()),
                            "list_too_long", stage="structure", nbt_path="list")

    def test_rejects_deeply_nested_nbt(self):
        self.assertRejected(fx.gzipped(self.tmp / "deep.schem", fx.deep_nbt()),
                            "too_deep", stage="structure")

    def test_rejects_array_longer_than_the_file(self):
        self.assertRejected(fx.gzipped(self.tmp / "arr.schem", fx.huge_array_nbt()),
                            "truncated", stage="structure", nbt_path="Blocks")

    def test_rejects_valid_nbt_that_is_not_a_schematic(self):
        from nbtlib import Compound, Int, String
        path = fx.write_nbt(Compound({"Health": Int(20), "Name": String("Steve")}),
                            self.tmp / "player.schem")
        self.assertRejected(path, "missing_dimension", stage="schematic", nbt_path="Width")

    def test_rejects_dimensions_beyond_the_volume_cap(self):
        from nbtlib import ByteArray, Compound, Short
        path = fx.write_nbt(Compound({
            "Width": Short(4000), "Height": Short(4000), "Length": Short(4000),
            "Blocks": ByteArray([0, 1, 2])}), self.tmp / "huge.schem")
        self.assertRejected(path, "volume_too_large", stage="schematic")

    def test_rejects_a_non_positive_dimension_naming_the_axis(self):
        from nbtlib import ByteArray, Compound, Short
        path = fx.write_nbt(Compound({
            "Width": Short(4), "Height": Short(0), "Length": Short(4),
            "Blocks": ByteArray([0])}), self.tmp / "flat.schem")
        self.assertRejected(path, "non_positive_dimension", nbt_path="Height")

    def test_rejects_block_payload_that_does_not_match_the_dimensions(self):
        from nbtlib import ByteArray, Compound, Short
        path = fx.write_nbt(Compound({
            "Width": Short(10), "Height": Short(10), "Length": Short(10),
            "Blocks": ByteArray([0, 1, 2])}), self.tmp / "short.schem")
        self.assertRejected(path, "truncated_blocks", nbt_path="Blocks")

    def test_rejects_sponge_file_with_the_wrong_number_of_blocks(self):
        from nbtlib import ByteArray, Compound, Int, Short
        path = fx.write_nbt(Compound({
            "Version": Int(2), "Width": Short(4), "Height": Short(4), "Length": Short(4),
            "Palette": Compound({"minecraft:stone": Int(0)}),
            "BlockData": ByteArray([0, 0, 0])}), self.tmp / "mismatch.schem")
        self.assertRejected(path, "block_count_mismatch", nbt_path="BlockData")

    def test_rejects_a_bad_tag_id_naming_the_tag(self):
        payload = bytearray(b"\x0a\x00\x00")
        payload += b"\x2a" + b"\x00\x05Width" + b"\x00\x04"   # tag id 42 does not exist
        self.assertRejected(fx.gzipped(self.tmp / "tag.schem", bytes(payload)),
                            "unknown_tag", stage="structure", nbt_path="Width")


class RejectionLocationTests(TempDirTest):
    """A rejection has to say *where*, or a bad download can't be diagnosed."""

    def corrupted_house(self):
        """A valid schematic with one byte inside TileEntities[2].Items[1] broken."""
        from nbtlib import Compound, Int, List, ByteArray, Short, String
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
        import io as _io
        import nbtlib
        buf = _io.BytesIO()
        handle = nbtlib.File(root)
        handle.root_name = "Schematic"
        handle.write(buf)
        data = bytearray(buf.getvalue())
        broken_at = data.index(b"\x05Count") - 2      # the tag id byte of "Count"
        data[broken_at] = 0x2a                        # not a real NBT tag id
        return fx.gzipped(self.tmp / "corrupt.schematic", bytes(data)), broken_at

    def test_points_at_the_exact_tag_that_is_broken(self):
        path, broken_at = self.corrupted_house()
        with self.assertRaises(SchematicError) as caught:
            schem_convert.inspect(path)
        error = caught.exception
        self.assertEqual(error.code, "unknown_tag")
        self.assertEqual(error.path, "TileEntities[2].Items[1].Count")
        self.assertGreaterEqual(error.offset, broken_at)

    def test_str_includes_the_location_but_message_stays_clean(self):
        path, _ = self.corrupted_house()
        with self.assertRaises(SchematicError) as caught:
            schem_convert.inspect(path)
        error = caught.exception
        self.assertIn("TileEntities[2].Items[1].Count", str(error))
        self.assertTrue(str(error).startswith(error.message))
        self.assertNotIn("(at ", error.message)

    def test_details_are_json_friendly(self):
        import json
        path, _ = self.corrupted_house()
        with self.assertRaises(SchematicError) as caught:
            schem_convert.inspect(path)
        details = caught.exception.details()
        self.assertEqual(set(details), {"error", "message", "code", "stage", "offset", "path"})
        self.assertEqual(json.loads(json.dumps(details)), details)

    def test_deep_paths_are_elided_rather_than_dumped(self):
        with self.assertRaises(SchematicError) as caught:
            schem_convert.inspect(fx.gzipped(self.tmp / "deep.schem", fx.deep_nbt()))
        self.assertIn("...", caught.exception.path)
        self.assertLess(len(caught.exception.path), 60)


# ── The mapping table ─────────────────────────────────────────────────────────

class LegacyBlockTableTests(unittest.TestCase):
    def test_every_legacy_id_and_data_pair_resolves_or_returns_none(self):
        """No id/data combination may raise -- schematics carry junk in unused bits."""
        for block_id in range(4096):
            for data in range(16):
                state = legacy_blocks.resolve(legacy_blocks.key(block_id, data))
                self.assertTrue(state is None or state.startswith("minecraft:"))

    def test_vanilla_ids_are_all_covered(self):
        missing = [i for i in range(219) if not legacy_blocks.is_known(i)]
        self.assertEqual(missing, [])

    def test_upper_half_keys_resolve_for_every_context(self):
        for block_id in legacy_blocks.NEEDS_BLOCK_BELOW:
            for data in range(8, 16):
                for below in range(16):
                    state = legacy_blocks.resolve(
                        legacy_blocks.key_with_below(block_id, data, below))
                    self.assertTrue(state.startswith("minecraft:"))

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
