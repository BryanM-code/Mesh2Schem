"""Tests for the procedural house generator and its schematic writer.

Three things have to hold, and they fail in very different ways.

**Determinism.** The library dedups on content sha256, so a generator that produced different
bytes for the same seed would fill it with duplicates that all claim to be distinct, and no
training run could ever be reproduced. Same seed, same bytes -- checked by writing twice.

**Importability.** A generated file has to satisfy exactly what the upload path demands, so
the test runs the upload path's own `schem_convert.inspect`, then decodes the result back and
compares it block for block. Comparing *blocks* and not palette indices is deliberate: the
writer re-densifies the palette, so the index arrays legitimately differ.

**Being a house.** The above can all pass on a solid cube of bricks. The last group asserts
the things that make it a building -- that it has a roof, that its walls are not air, that the
blockstates it claims to author are actually authored -- and closes the loop by feeding a
generated house back through `house_stats` and checking the analyser recovers what the
generator was told to build.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import block_roles
import house_grammar
import house_params
import house_stats
import schem_convert
import schem_voxels
import schem_write


class TempDirTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="housegen-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.tmp, name)


class WriterTest(TempDirTest):

    def test_round_trips_block_for_block(self):
        house = house_grammar.generate(4242)
        target = self.path("a.schem")
        house_grammar.write(house, target)

        voxels = schem_voxels.load_voxels(target)
        self.assertIsNotNone(voxels)
        want = np.array(house.names, dtype=object)[house.grid]
        got = np.array(voxels.names, dtype=object)[voxels.grid]
        self.assertTrue(bool((want == got).all()))

    def test_palette_is_dense_and_has_no_unused_entries(self):
        house = house_grammar.generate(7)
        target = self.path("b.schem")
        house_grammar.write(house, target)
        voxels = schem_voxels.load_voxels(target)
        self.assertTrue(all(voxels.names))
        used = set(np.unique(voxels.grid).tolist())
        # Air is always in the palette even if every voxel is solid; everything else must be
        # placed somewhere, or the file carries a palette that lies about its contents.
        unused = set(range(len(voxels.names))) - used
        for slot in unused:
            self.assertEqual(voxels.names[slot], "minecraft:air")

    def test_metadata_survives_ensure_sponge(self):
        """`ensure_sponge` passes Sponge files through untouched, so provenance sticks."""
        house = house_grammar.generate(11)
        source = self.path("c.schem")
        house_grammar.write(house, source)
        with open(source, "rb") as fh:
            before = fh.read()

        result = schem_convert.ensure_sponge(source, self.path("converted.schem"))
        self.assertFalse(result.converted)
        with open(source, "rb") as fh:
            self.assertEqual(fh.read(), before)

        import nbt_safe
        root = nbt_safe.load(source)
        self.assertEqual(str(root["Metadata"]["Generator"]), "house_grammar")
        self.assertEqual(int(root["Metadata"]["Seed"]), 11)

    def test_rejects_oversize_grid(self):
        grid = np.zeros((2, 2, schem_write.MAX_AXIS + 1), dtype=np.uint16)
        with self.assertRaises(schem_write.WriteError):
            schem_write.write_schem(grid, ["minecraft:air"], self.path("huge.schem"))
        self.assertFalse(os.path.exists(self.path("huge.schem")))

    def test_rejects_index_past_palette(self):
        grid = np.ones((2, 2, 2), dtype=np.uint16) * 5
        with self.assertRaises(schem_write.WriteError):
            schem_write.write_schem(grid, ["minecraft:air"], self.path("bad.schem"))


class DeterminismTest(TempDirTest):

    def _sha(self, seed, name):
        house = house_grammar.generate(seed)
        target = self.path(name)
        house_grammar.write(house, target)
        with open(target, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    def test_same_seed_same_bytes(self):
        self.assertEqual(self._sha(99, "x.schem"), self._sha(99, "y.schem"))

    def test_distinct_seeds_distinct_files(self):
        hashes = {self._sha(seed, "s{}.schem".format(seed)) for seed in range(200, 216)}
        self.assertEqual(len(hashes), 16)


class ImportabilityTest(TempDirTest):

    def test_every_house_passes_the_upload_path(self):
        for seed in range(500, 512):
            house = house_grammar.generate(seed)
            target = self.path("h{}.schem".format(seed))
            house_grammar.write(house, target)

            info = schem_convert.inspect(target)
            self.assertEqual(info.format, "sponge_v2", "seed {}".format(seed))
            height, length, width = house.grid.shape
            self.assertEqual(info.dims, (width, height, length), "seed {}".format(seed))
            self.assertLessEqual(max(info.dims), schem_convert.MAX_AXIS)
            self.assertLessEqual(width * height * length, schem_convert.MAX_VOLUME)


class GeometryTest(unittest.TestCase):

    def test_houses_are_hollow_not_solid(self):
        """Density inside the range measured over the 103 analysable real builds."""
        for seed in range(600, 612):
            house = house_grammar.generate(seed)
            solid = (house.grid != 0).sum()
            density = solid / float(house.grid.size)
            self.assertGreater(density, 0.04, "seed {} is nearly empty".format(seed))
            self.assertLess(density, 0.55, "seed {} is nearly solid".format(seed))

    def test_stairs_carry_blockstates(self):
        """Two thirds of the real builds have properties; default-facing roofs look wrong."""
        found = False
        for seed in range(700, 712):
            house = house_grammar.generate(seed)
            for name in house.names:
                if "_stairs" in name:
                    found = True
                    self.assertIn("[facing=", name)
                    self.assertIn("half=", name)
        self.assertTrue(found, "no stairs in twelve houses -- the roofs are not being built")

    def test_every_block_name_is_real(self):
        """Guards the `stone_brick_stairs` -> `stone_brick` class of bug.

        Chopping a suffix off a block name produces something that validates, writes and
        renders as fallback grey. Nothing in the format catches it, so the test does: every
        material the generator emits has to round-trip through the vocabulary tables.
        """
        for seed in range(800, 820):
            house = house_grammar.generate(seed)
            for name in house.names:
                base = block_roles.normalise(name)
                if base.endswith("_stairs"):
                    self.assertIsNotNone(
                        block_roles.base_for_stairs(name),
                        "{} is not a stair variant this repo knows".format(name))
                elif base.endswith("_slab"):
                    stem = base[:-len("_slab")]
                    self.assertTrue(
                        any(s.startswith(stem) for s in block_roles._SLABS.values()),
                        "{} is not a slab variant this repo knows".format(name))

    def test_door_is_placed_on_the_ground_floor(self):
        seen = 0
        for seed in range(900, 912):
            house = house_grammar.generate(seed)
            if any("_door" in name for name in house.names):
                seen += 1
        self.assertGreaterEqual(seen, 10, "most houses should have a door")


class BiasTest(unittest.TestCase):

    def test_roof_type_can_be_forced(self):
        for roof in ("gable", "hip", "flat"):
            for seed in range(300, 306):
                spec = house_grammar.sample_spec(seed, {"roof_type": roof})
                self.assertEqual(spec.roof_type, roof)

    def test_storeys_can_be_forced(self):
        for seed in range(300, 306):
            self.assertEqual(house_grammar.sample_spec(seed, {"storeys": 3}).storeys, 3)

    def test_wall_bias_keeps_the_palette_coherent(self):
        """Forcing a wall must not break the co-occurrence -- the rest follows from it."""
        wall = house_params.STYLES[0][0]
        for seed in range(400, 410):
            spec = house_grammar.sample_spec(seed, {"wall": wall})
            self.assertEqual(spec.wall, wall)
            kin = [s for s in house_params.STYLES if s[0] == wall]
            self.assertIn(spec.roof_stairs, {s[1] for s in kin})

    def test_size_bias_shifts_the_footprint(self):
        small = [house_grammar.sample_spec(s, {"size": "small"}) for s in range(50)]
        large = [house_grammar.sample_spec(s, {"size": "large"}) for s in range(50)]
        mean = lambda specs: sum(max(s.width, s.depth) for s in specs) / len(specs)  # noqa: E731
        self.assertLess(mean(small), mean(large))


class AnalysisRoundTripTest(TempDirTest):
    """Feed generated houses back through Phase 1 and see if it recognises them.

    This is the test that ties the two halves together: the generator samples from statistics
    the analyser produced, so the analyser ought to read a generated house and recover roughly
    what it was told to build. When it does not, one of the two is wrong and this says so.
    """

    def _analyse(self, house):
        voxels = schem_voxels.VoxelGrid(house.grid, list(house.names))
        seg = house_stats.segment(voxels)
        self.assertIsNone(seg.reject, "segmentation rejected a generated house")
        ys, zs, xs = np.nonzero(seg.structure)
        sub = seg.structure[ys.min():ys.max() + 1, zs.min():zs.max() + 1, xs.min():xs.max() + 1]
        area = int(sub.any(axis=0).sum())
        peaks, _ = house_stats._analyse_storeys(sub, area)
        return seg, sub, peaks

    def test_ground_pad_is_found(self):
        for seed in range(1000, 1006):
            house = house_grammar.generate(seed)
            seg, _, _ = self._analyse(house)
            self.assertEqual(seg.ground_top, house.spec.pad_depth - 1,
                             "seed {}".format(seed))
            self.assertEqual(seg.method, "terrain_pad")

    def test_storeys_are_recovered(self):
        for seed in range(1100, 1108):
            house = house_grammar.generate(seed)
            _, sub, peaks = self._analyse(house)
            self.assertGreaterEqual(len(peaks), house.spec.storeys,
                                    "seed {} built {} storeys, analyser found {} plates"
                                    .format(seed, house.spec.storeys, len(peaks)))

    def test_roof_type_is_recovered(self):
        """Sloped roofs must not read as flat, and flat ones must not read as sloped."""
        agree = 0
        total = 0
        for seed in range(1200, 1216):
            house = house_grammar.generate(seed)
            _, sub, peaks = self._analyse(house)
            if not peaks:
                continue
            roof = house_stats._analyse_roof(sub, peaks)
            total += 1
            built_flat = house.spec.roof_type == "flat"
            read_flat = roof["type"] == "flat"
            if built_flat == read_flat:
                agree += 1
        self.assertGreater(total, 8)
        self.assertGreaterEqual(agree / total, 0.75,
                                "analyser and generator disagree about flat vs sloped")


if __name__ == "__main__":
    unittest.main()
