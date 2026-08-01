"""Tests for turning a Bedrock region into a Sponge `.schem`, and the import endpoints.

The conversion has to satisfy two things the rest of the app already relies on: the output
must be readable by the app's *own* schematic parser, and converting the same region twice
must produce identical bytes, because the library dedups on a content hash.
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db  # noqa: E402

SANDBOX = Path(tempfile.mkdtemp(prefix="bdc-mcworld-tests-"))
db.PROJECT_ROOT = db.DATA_DIR = SANDBOX
db.DB_PATH = SANDBOX / "collector.db"
db.BUILDS_DIR = SANDBOX / "builds"

import bedrock_world as bw  # noqa: E402
import block_colors  # noqa: E402
import mcworld_convert as mc  # noqa: E402
import mcworld_fixtures as fx  # noqa: E402
import mcworld_preview  # noqa: E402
import schem_convert  # noqa: E402
import server  # noqa: E402
from schem_meta import sha256_file  # noqa: E402

server.TMP_DIR = SANDBOX / ".tmp"
server.IMPORT_DIR = server.TMP_DIR / "import"
server.IMPORT_DIR.mkdir(parents=True, exist_ok=True)
db.init_db()


def tearDownModule():
    shutil.rmtree(SANDBOX, ignore_errors=True)


class NameMappingTests(unittest.TestCase):
    def test_modern_names_pass_straight_through(self):
        """Bedrock and Java agree on almost everything, which is why this is a short table."""
        for name in ("minecraft:quartz_stairs", "minecraft:oak_log",
                     "minecraft:deepslate_tile_stairs", "minecraft:grass_block"):
            with self.subTest(name):
                self.assertEqual(mc.java_name(name), name)

    def test_the_renames_are_applied(self):
        self.assertEqual(mc.java_name("minecraft:noteblock"), "minecraft:note_block")
        self.assertEqual(mc.java_name("minecraft:wooden_door"), "minecraft:oak_door")
        self.assertEqual(mc.java_name("minecraft:darkoak_wall_sign"),
                         "minecraft:dark_oak_wall_sign")

    def test_a_bare_name_gets_the_namespace(self):
        self.assertEqual(mc.java_name("stone"), "minecraft:stone")

    def test_an_unknown_block_survives_rather_than_vanishing(self):
        """A modded block should still occupy space, just render as neutral grey."""
        self.assertEqual(mc.java_name("minecraft:some_mod_block"),
                         "minecraft:some_mod_block")

    def test_every_mapped_name_has_a_colour(self):
        """A rename that lands on a name block_colors doesn't know would render grey."""
        greyless = [target for target in mc.NAME_ALIASES.values()
                    if block_colors.color_for(target) == block_colors.FALLBACK]
        self.assertEqual(greyless, [], "these aliases map onto uncoloured blocks")


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-conv-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def region(self, chunks=2):
        world = bw.open_world(fx.flat_world(self.tmp / "w.mcworld", chunks=chunks))
        return world, world.auto_bbox()

    def test_the_app_can_read_what_we_write(self):
        """The real test of the writer: the project's own parser accepts the output."""
        world, box = self.region()
        grid, names = world.extract_region(box)
        out = self.tmp / "out.schem"
        info = mc.region_to_schem(grid, names, out)

        parsed = schem_convert.inspect(out)
        self.assertEqual(parsed.format, "sponge_v2")
        self.assertEqual(parsed.dims, (info["width"], info["height"], info["length"]))

    def test_dimensions_match_the_region(self):
        world, _box = self.region()
        box = {"min_x": 0, "max_x": 4, "min_y": 0, "max_y": 2, "min_z": 0, "max_z": 9}
        grid, names = world.extract_region(box)
        info = mc.region_to_schem(grid, names, self.tmp / "dims.schem")
        self.assertEqual((info["width"], info["height"], info["length"]), (5, 3, 10))

    def test_conversion_is_deterministic(self):
        """Dedup is a content hash, so the same region twice must be the same bytes."""
        hashes = []
        for run in range(2):
            world = bw.open_world(fx.flat_world(self.tmp / "w.mcworld", chunks=2))
            grid, names = world.extract_region(world.auto_bbox())
            out = self.tmp / "run{}.schem".format(run)
            mc.region_to_schem(grid, names, out)
            hashes.append(sha256_file(out))
        self.assertEqual(hashes[0], hashes[1])

    def test_the_palette_order_does_not_depend_on_traversal(self):
        """Ordering by name is what makes the hash stable across database layouts."""
        world, box = self.region()
        grid, names = world.extract_region(box)
        mc.region_to_schem(grid, names, self.tmp / "a.schem")

        shuffled = list(reversed(names))
        remap = {name: index for index, name in enumerate(shuffled)}
        regrid = grid.copy()
        for old, name in enumerate(names):
            regrid[grid == old] = remap[name]
        mc.region_to_schem(regrid, shuffled, self.tmp / "b.schem")

        self.assertEqual(sha256_file(self.tmp / "a.schem"),
                         sha256_file(self.tmp / "b.schem"))

    def test_the_blocks_land_where_they_were(self):
        world, box = self.region()
        grid, names = world.extract_region(box)
        out = self.tmp / "tower.schem"
        mc.region_to_schem(grid, names, out)

        import nbt_safe
        import numpy as np
        root = nbt_safe.load(out)
        if "" in root:
            root = root[""]
        palette = {str(k): int(v) for k, v in root["Palette"].items()}
        raw = np.asarray(root["BlockData"]).view(np.uint8)
        self.assertEqual(len(raw), 3, "a 1x3x1 tower is three blocks")
        self.assertTrue(all(value == palette["minecraft:quartz_block"] for value in raw))

    def test_an_oversized_region_is_refused_with_a_code(self):
        """Rejected up front, rather than by running out of memory."""
        world, _box = self.region()
        huge = {"min_x": 0, "max_x": 3000, "min_y": 0, "max_y": 300,
                "min_z": 0, "max_z": 3000}
        import numpy as np
        grid = np.zeros((301, 3001, 3001), dtype=np.uint16) if False else None
        del grid, huge
        # Build the check directly rather than allocating 2.7 billion cells for it.
        with self.assertRaises(bw.WorldError) as caught:
            mc.region_to_schem(_FakeGrid((400, 400, 400)), ["minecraft:air"],
                               self.tmp / "huge.schem")
        self.assertEqual(caught.exception.code, "volume_too_large")

    def test_conversion_reports_that_states_were_dropped(self):
        world, box = self.region()
        grid, names = world.extract_region(box)
        info = mc.region_to_schem(grid, names, self.tmp / "warn.schem")
        self.assertTrue(any("default state" in w for w in info["warnings"]),
                        "the lossy part of the conversion must be stated")


class _FakeGrid:
    """Just enough of an array to reach the size check without allocating one."""

    def __init__(self, shape):
        self.shape = shape


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-prev-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_a_map_renders_at_block_resolution(self):
        world = bw.open_world(fx.flat_world(self.tmp / "w.mcworld", chunks=2))
        image = mcworld_preview.render_world_map(world, world.world_bounds())
        self.assertIsNotNone(image)
        width, height = image.size
        self.assertEqual(width % 32, 0, "32 blocks across, scaled by a whole number")
        self.assertEqual(width, height)

    def test_the_highlight_is_drawn(self):
        world = bw.open_world(fx.flat_world(self.tmp / "w.mcworld", chunks=2))
        bounds = world.world_bounds()
        plain = mcworld_preview.render_world_map(world, bounds)
        marked = mcworld_preview.render_world_map(world, bounds,
                                                  highlight=world.auto_bbox())
        self.assertNotEqual(plain.tobytes(), marked.tobytes())


class ImportEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-api-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        db.PROJECT_ROOT = db.DATA_DIR = self.tmp
        db.DB_PATH = self.tmp / "collector.db"
        db.BUILDS_DIR = self.tmp / "builds"
        server.TMP_DIR = self.tmp / ".tmp"
        server.IMPORT_DIR = server.TMP_DIR / "import"
        server.IMPORT_DIR.mkdir(parents=True, exist_ok=True)
        db.init_db()

    def upload(self, path=None, name="world.mcworld"):
        path = path or fx.flat_world(self.tmp / "w.mcworld", chunks=2)
        payload = Path(path).read_bytes()
        return self.client.post("/api/import/inspect",
                                data={"file": (io.BytesIO(payload), name)},
                                content_type="multipart/form-data")

    def test_inspect_reports_the_world_and_where_the_build_is(self):
        res = self.upload()
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertIn("token", body)
        self.assertEqual(body["auto_bbox"]["blocks"], 3)
        self.assertEqual(body["kind"], "bedrock_world")
        self.assertEqual(body["suggested"]["dims"], "1x3x1")
        self.assertFalse(body["suggested"]["too_big"])
        self.assertIn("map_bounds", body, "the client needs the map's coordinate frame")

    def test_a_non_mcworld_is_refused(self):
        res = self.client.post("/api/import/inspect",
                               data={"file": (io.BytesIO(b"x"), "thing.schem")},
                               content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)

    def test_a_broken_world_is_refused_with_a_code(self):
        res = self.client.post("/api/import/inspect",
                               data={"file": (io.BytesIO(b"not a zip"), "w.mcworld")},
                               content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["code"], "not_a_zip")

    def test_the_preview_is_a_png(self):
        body = self.upload().get_json()
        res = self.client.get(body["preview"])
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["Content-Type"], "image/png")
        self.assertTrue(res.get_data().startswith(b"\x89PNG\r\n\x1a\n"))
        res.close()

    def test_region_reports_dimensions_without_saving(self):
        token = self.upload().get_json()["token"]
        res = self.client.post("/api/import/{}/region".format(token),
                               json={"min_x": 0, "max_x": 9, "min_y": 0, "max_y": 4,
                                     "min_z": 0, "max_z": 9})
        self.assertEqual(res.get_json()["dims"], "10x5x10")
        self.assertEqual(db.list_builds(), [], "inspecting must not add anything")

    def test_an_oversized_region_is_flagged_before_saving(self):
        token = self.upload().get_json()["token"]
        res = self.client.post("/api/import/{}/region".format(token),
                               json={"min_x": 0, "max_x": 3000, "min_y": 0, "max_y": 300,
                                     "min_z": 0, "max_z": 3000})
        self.assertTrue(res.get_json()["too_big"])

    def test_a_malformed_region_is_refused(self):
        token = self.upload().get_json()["token"]
        res = self.client.post("/api/import/{}/region".format(token),
                               json={"min_x": "left"})
        self.assertEqual(res.status_code, 400)

    def test_saving_creates_a_normal_build(self):
        body = self.upload().get_json()
        res = self.client.post("/api/import/{}/save".format(body["token"]),
                               json=dict(body["auto_bbox"], name="Tower"))
        self.assertEqual(res.status_code, 201)
        build = res.get_json()["build"]
        self.assertEqual(build["name"], "Tower")
        self.assertEqual(build["dims"], "1x3x1")
        self.assertEqual(build["status"], "needs_image")
        stored = db.build_dir(build["id"], build["slug"]) / "schem" / "original.schem"
        self.assertTrue(stored.exists())
        self.assertEqual(schem_convert.inspect(stored).format, "sponge_v2")

    def test_the_conversion_warning_lands_in_the_note(self):
        body = self.upload().get_json()
        res = self.client.post("/api/import/{}/save".format(body["token"]),
                               json=dict(body["auto_bbox"], name="Tower",
                                         source_note="from a friend"))
        note = res.get_json()["build"]["source_note"]
        self.assertIn("from a friend", note)
        self.assertIn("default state", note, "the lossy part must be recorded")

    def test_the_same_region_twice_is_a_duplicate(self):
        body = self.upload().get_json()
        first = self.client.post("/api/import/{}/save".format(body["token"]),
                                 json=dict(body["auto_bbox"], name="One"))
        self.assertEqual(first.status_code, 201)
        second = self.client.post("/api/import/{}/save".format(body["token"]),
                                  json=dict(body["auto_bbox"], name="Two"))
        self.assertEqual(second.status_code, 409, "deterministic output should dedup")
        self.assertEqual(second.get_json()["error"], "duplicate")

    def test_discarding_forgets_the_upload(self):
        token = self.upload().get_json()["token"]
        self.assertEqual(self.client.delete("/api/import/{}".format(token)).status_code, 200)
        res = self.client.post("/api/import/{}/region".format(token),
                               json={"min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1,
                                     "min_z": 0, "max_z": 1})
        self.assertEqual(res.status_code, 404)

    def test_a_bogus_token_is_never_used_as_a_path(self):
        for token in ("../../etc", "..", "zzz", "a" * 31, "../" * 8 + "collector.db"):
            with self.subTest(token=token):
                res = self.client.get("/api/import/{}/preview.png".format(token))
                self.assertIn(res.status_code, (404, 308))
                res.close()

    def test_the_api_version_covers_world_import(self):
        meta = self.client.get("/api/meta").get_json()
        self.assertGreaterEqual(meta["api_version"], 4)


if __name__ == "__main__":
    unittest.main()
