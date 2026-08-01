"""Tests for Java Edition worlds, the zip dispatcher, and cropping.

Builds downloaded from build sites usually arrive as a zipped *world*, with no single file
inside that is the build -- which is why a zip used to be a dead end. These cover reading
one, telling the three kinds of zip apart, and cutting a region out of any source.

Worlds are written in code rather than checked in, same as `schem_fixtures`. Writing an
Anvil region by hand is also the clearest statement of what `java_world` claims the format
is.
"""
from __future__ import annotations

import io
import gzip
import os
import shutil
import struct
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db  # noqa: E402

SANDBOX = Path(tempfile.mkdtemp(prefix="bdc-java-tests-"))
db.PROJECT_ROOT = db.DATA_DIR = SANDBOX
db.DB_PATH = SANDBOX / "collector.db"
db.BUILDS_DIR = SANDBOX / "builds"

import import_source  # noqa: E402
import java_world as jw  # noqa: E402
import mcworld_fixtures as mfx  # noqa: E402
import schem_source  # noqa: E402
import server  # noqa: E402
from bedrock_world import WorldError  # noqa: E402

server.TMP_DIR = SANDBOX / ".tmp"
server.IMPORT_DIR = server.TMP_DIR / "import"
server.IMPORT_DIR.mkdir(parents=True, exist_ok=True)
db.init_db()


def tearDownModule():
    shutil.rmtree(SANDBOX, ignore_errors=True)


# ── Building a Java world by hand ─────────────────────────────────────────────

def _nbt_bytes(compound_writer) -> bytes:
    """Serialise a root compound with nbtlib (Java NBT: big-endian)."""
    import nbtlib
    buffer = io.BytesIO()
    nbtlib.File(compound_writer).write(buffer)
    return buffer.getvalue()


def make_section(palette, indices, data_version, y=0):
    """One chunk section: a palette plus bit-packed indices in Java's YZX order."""
    from nbtlib import Byte, Compound, List, LongArray, String

    bits = max(4, (len(palette) - 1).bit_length())
    padded = data_version >= jw.PADDED_SINCE

    longs = []
    if padded:
        per_word = 64 // bits
        for start in range(0, 4096, per_word):
            word = 0
            for slot, position in enumerate(range(start, min(start + per_word, 4096))):
                word |= (indices[position] & ((1 << bits) - 1)) << (slot * bits)
            longs.append(word)
    else:
        buffer = held = 0
        for value in indices:
            buffer |= (value & ((1 << bits) - 1)) << held
            held += bits
            while held >= 64:
                longs.append(buffer & ((1 << 64) - 1))
                buffer >>= 64
                held -= 64
        if held:
            longs.append(buffer & ((1 << 64) - 1))

    signed = [value - (1 << 64) if value >= (1 << 63) else value for value in longs]
    return Compound({
        "Y": Byte(y),
        "Palette": List[Compound]([Compound({"Name": String(name)}) for name in palette]),
        "BlockStates": LongArray(signed),
    })


def make_region(chunks) -> bytes:
    """An Anvil region file: 8 KiB of header, then zlib-compressed chunk NBT."""
    from nbtlib import Compound, Int, List, String

    locations = bytearray(4096)
    timestamps = bytearray(4096)
    body = bytearray()
    sector = 2                                    # after the two header sectors

    for (chunk_x, chunk_z, sections, data_version) in chunks:
        root = Compound({
            "DataVersion": Int(data_version),
            "Level": Compound({
                "xPos": Int(chunk_x), "zPos": Int(chunk_z),
                "Status": String("full"),
                "Sections": List[Compound](sections),
            }),
        })
        payload = zlib.compress(_nbt_bytes(root))
        frame = struct.pack(">I", len(payload) + 1) + bytes([2]) + payload
        pad = (-len(frame)) % 4096
        frame += b"\x00" * pad
        count = len(frame) // 4096

        index = (chunk_z & 31) * 32 + (chunk_x & 31)
        locations[index * 4:index * 4 + 3] = sector.to_bytes(3, "big")
        locations[index * 4 + 3] = count
        body += frame
        sector += count

    return bytes(locations) + bytes(timestamps) + bytes(body)


def write_java_world(path, chunks, data_version=2730, root="MyWorld", level_name="MyWorld"):
    """Zip a Java world: level.dat plus one region file."""
    from nbtlib import Compound, Int, String

    level = Compound({"Data": Compound({
        "LevelName": String(level_name), "DataVersion": Int(data_version)})})
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(root + "/level.dat", gzip.compress(_nbt_bytes(level)))
        archive.writestr(root + "/region/r.0.0.mca", make_region(chunks))
    return str(path)


def tower_world(path, data_version=2730, root="MyWorld"):
    """Ground plate across one chunk with a 1x3x1 quartz column standing on it."""
    palette = ["minecraft:air", "minecraft:bedrock", "minecraft:grass_block",
               "minecraft:quartz_block"]
    indices = [0] * 4096
    java = lambda x, y, z: (y << 8) | (z << 4) | x     # noqa: E731 - Java's own order
    for x in range(16):
        for z in range(16):
            indices[java(x, 0, z)] = 1
            indices[java(x, 1, z)] = 2
    for y in (2, 3, 4):
        indices[java(0, y, 0)] = 3
    return write_java_world(path, [(0, 0, [make_section(palette, indices, data_version)],
                                    data_version)],
                            data_version=data_version, root=root)


class JavaWorldTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-jw-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_a_world_round_trips(self):
        world = jw.open_world(tower_world(self.tmp / "w.zip"))
        self.assertEqual(world.level_name, "MyWorld")
        self.assertEqual(len(world.subchunks), 1)

    def test_both_bit_packings_decode(self):
        """1.16 stopped letting an index span two longs; reading one as the other is wrong
        in a way that looks almost right, so both are exercised."""
        for data_version, label in ((1631, "1.13 spanning"), (2730, "1.17 padded")):
            with self.subTest(label):
                world = jw.open_world(tower_world(
                    self.tmp / "v{}.zip".format(data_version), data_version=data_version))
                box = world.auto_bbox()
                self.assertEqual(box["blocks"], 3, "should find just the 3-block tower")
                self.assertEqual((box["min_y"], box["max_y"]), (2, 4))

    def test_block_order_is_converted_from_java(self):
        """Java indexes YZX and Bedrock XZY. Decoding one as the other swaps X and Y and
        renders a build sliced into chunk-wide stripes."""
        world = jw.open_world(tower_world(self.tmp / "w.zip"))
        box = world.auto_bbox()
        self.assertEqual((box["min_x"], box["max_x"]), (0, 0), "the tower is one column")
        self.assertEqual((box["min_z"], box["max_z"]), (0, 0))
        self.assertEqual((box["min_y"], box["max_y"]), (2, 4), "and three blocks tall")

    def test_a_world_zipped_without_its_folder_still_opens(self):
        world = jw.open_world(tower_world(self.tmp / "bare.zip", root=""))
        self.assertEqual(len(world.subchunks), 1)

    def test_a_zip_with_no_region_files_is_refused(self):
        path = self.tmp / "empty.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("MyWorld/level.dat", b"\x00" * 8)
        with self.assertRaises(WorldError) as caught:
            jw.open_world(path)
        self.assertEqual(caught.exception.code, "no_region_files")

    def test_a_non_zip_is_refused(self):
        path = self.tmp / "nope.zip"
        path.write_bytes(b"not a zip at all")
        with self.assertRaises(WorldError) as caught:
            jw.open_world(path)
        self.assertEqual(caught.exception.code, "not_a_zip")

    def test_extracting_a_region_keeps_only_what_is_inside(self):
        world = jw.open_world(tower_world(self.tmp / "w.zip"))
        grid, names = world.extract_region(world.auto_bbox())
        self.assertEqual(grid.shape, (3, 1, 1))
        self.assertEqual(sorted(names), ["minecraft:air", "minecraft:quartz_block"])


class ZipDispatchTests(unittest.TestCase):
    """Telling the three kinds of zip apart is the fix for 'which part of this is useful'."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-zip-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_a_zipped_java_world_is_recognised(self):
        path = tower_world(self.tmp / "world.zip", root="Suburban House")
        described = import_source.describe_upload(path, "world.zip")
        self.assertEqual(described["kind"], "java_world")
        self.assertEqual(described["world_root"], "Suburban House")
        self.assertEqual(described["region_files"], 1)

    def test_a_zip_of_schematics_is_recognised(self):
        import schem_fixtures as sfx
        path = self.tmp / "schems.zip"
        one = sfx.sponge_v2(self.tmp / "a.schem", 2, 2, 2)
        with zipfile.ZipFile(path, "w") as archive:
            archive.write(one, "pack/a.schem")
            archive.write(one, "pack/b.schem")
            archive.writestr("pack/readme.txt", b"hello")
        described = import_source.describe_upload(path, "schems.zip")
        self.assertEqual(described["kind"], "zip_schematics")
        self.assertEqual(len(described["schematics"]), 2)

    def test_an_unrelated_zip_says_what_is_in_it(self):
        """'Not supported' is not something anyone can act on."""
        path = self.tmp / "junk.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("notes.txt", b"x")
            archive.writestr("photo.png", b"x")
        described = import_source.describe_upload(path, "junk.zip")
        self.assertEqual(described["kind"], "unknown")
        self.assertIn("notes.txt", described["contents"])
        self.assertIn("no schematic", described["reason"])

    def test_a_zipped_bedrock_world_is_pointed_at(self):
        path = self.tmp / "holder.zip"
        inner = mfx.simple_world(self.tmp / "inner.mcworld")
        with zipfile.ZipFile(path, "w") as archive:
            archive.write(inner, "worlds/inner.mcworld")
        described = import_source.describe_upload(path, "holder.zip")
        self.assertEqual(described["kind"], "zip_worlds")
        self.assertEqual(described["worlds"], ["worlds/inner.mcworld"])

    def test_the_plain_kinds_are_still_recognised(self):
        import schem_fixtures as sfx
        schem = sfx.sponge_v2(self.tmp / "a.schem", 2, 2, 2)
        self.assertEqual(import_source.describe_upload(schem, "a.schem")["kind"], "schematic")
        world = mfx.simple_world(self.tmp / "b.mcworld")
        self.assertEqual(import_source.describe_upload(world, "b.mcworld")["kind"],
                         "bedrock_world")

    def test_zip_slip_names_are_never_written_out(self):
        import schem_fixtures as sfx
        path = self.tmp / "evil.zip"
        one = sfx.sponge_v2(self.tmp / "a.schem", 2, 2, 2)
        with zipfile.ZipFile(path, "w") as archive:
            archive.write(one, "../../escaped.schem")
        described = import_source.describe_upload(path, "evil.zip")
        self.assertEqual(described["kind"], "zip_schematics")

        out = self.tmp / "extracted"
        written = import_source.extract_zip_schematics(path, described["schematics"], out)
        self.assertEqual(len(written), 1)
        for entry in written:
            self.assertTrue(Path(entry["path"]).resolve().is_relative_to(out.resolve()),
                            "extraction must stay inside the folder it was given")
        self.assertFalse((self.tmp.parent / "escaped.schem").exists())


class SchematicAsSourceTests(unittest.TestCase):
    """A dropped schematic is croppable too, which is what makes the picker universal."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-ss-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def schem(self, w=6, h=5, l=7):
        import schem_fixtures as sfx
        return Path(sfx.sponge_v2(self.tmp / "s.schem", w, h, l))

    def test_bounds_are_the_schematics_own_dimensions(self):
        """Chunking rounds up to 16; reporting that would let a crop run off the end."""
        source = schem_source.load_schem_source(self.schem(6, 5, 7))
        self.assertEqual(source.world_bounds(),
                         {"min_x": 0, "max_x": 5, "min_y": 0, "max_y": 4,
                          "min_z": 0, "max_z": 6})

    def test_a_region_can_be_cut_out_of_it(self):
        source = schem_source.load_schem_source(self.schem(6, 5, 7))
        grid, _names = source.extract_region(
            {"min_x": 1, "max_x": 3, "min_y": 0, "max_y": 2, "min_z": 2, "max_z": 4})
        self.assertEqual(grid.shape, (3, 3, 3))

    def test_the_whole_thing_round_trips_unchanged(self):
        """Cropping to the full extent must not quietly alter the build."""
        import mcworld_convert as mc
        import schem_convert
        original = self.schem(6, 5, 7)
        source = schem_source.load_schem_source(original)
        grid, names = source.extract_region(source.world_bounds())
        out = self.tmp / "round.schem"
        mc.region_to_schem(grid, names, out)
        self.assertEqual(schem_convert.inspect(out).dims, (6, 5, 7))


class CropAndRecropTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-crop-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        db.PROJECT_ROOT = db.DATA_DIR = self.tmp
        db.DB_PATH = self.tmp / "collector.db"
        db.BUILDS_DIR = self.tmp / "builds"
        server.TMP_DIR = self.tmp / ".tmp"
        server.IMPORT_DIR = server.TMP_DIR / "import"
        server.IMPORT_DIR.mkdir(parents=True, exist_ok=True)
        db.init_db()

    def drop(self, path, name):
        payload = Path(path).read_bytes()
        return self.client.post("/api/import/inspect",
                                data={"file": (io.BytesIO(payload), name)},
                                content_type="multipart/form-data")

    def imported_schematic(self, w=8, h=6, l=9):
        import schem_fixtures as sfx
        path = sfx.sponge_v2(self.tmp / "s.schem", w, h, l)
        body = self.drop(path, "s.schem").get_json()
        return body

    def test_a_dropped_schematic_opens_the_picker(self):
        body = self.imported_schematic()
        self.assertEqual(body["kind"], "schematic")
        self.assertEqual(body["kind_label"], "schematic")
        self.assertIn("map_bounds", body)
        self.assertIn("preview", body)

    def test_saving_a_crop_keeps_the_uncropped_source(self):
        body = self.imported_schematic()
        crop = {"min_x": 1, "max_x": 4, "min_y": 0, "max_y": 3, "min_z": 2, "max_z": 5}
        res = self.client.post("/api/import/{}/save".format(body["token"]),
                               json=dict(crop, name="Cropped"))
        self.assertEqual(res.status_code, 201)
        build = res.get_json()["build"]
        self.assertEqual(build["dims"], "4x4x4")
        self.assertTrue(build["can_recrop"])

        folder = db.build_dir(build["id"], build["slug"]) / "schem"
        self.assertTrue((folder / "original.schem").exists())
        self.assertTrue((folder / "source.schem").exists(),
                        "without the source a crop could never be redrawn")

    def test_a_crop_can_be_widened_afterwards(self):
        """The whole reason the source is kept: a box drawn too tight is recoverable."""
        body = self.imported_schematic()
        tight = {"min_x": 2, "max_x": 3, "min_y": 0, "max_y": 1, "min_z": 2, "max_z": 3}
        build = self.client.post("/api/import/{}/save".format(body["token"]),
                                 json=dict(tight, name="Tight")).get_json()["build"]
        self.assertEqual(build["dims"], "2x2x2")

        bounds = self.client.get(
            "/api/builds/{}/source-bounds".format(build["id"])).get_json()
        res = self.client.post("/api/builds/{}/recrop".format(build["id"]),
                               json=bounds["bounds"])
        self.assertEqual(res.status_code, 200)
        widened = res.get_json()["build"]
        self.assertEqual(widened["dims"], "8x6x9", "back to the whole source")
        self.assertEqual(widened["name"], "Tight", "everything else is untouched")

    def test_recropping_changes_the_content_hash(self):
        body = self.imported_schematic()
        build = self.client.post(
            "/api/import/{}/save".format(body["token"]),
            json=dict(body["bounds"], name="Whole")).get_json()["build"]
        res = self.client.post("/api/builds/{}/recrop".format(build["id"]),
                               json={"min_x": 0, "max_x": 3, "min_y": 0, "max_y": 3,
                                     "min_z": 0, "max_z": 3})
        self.assertNotEqual(res.get_json()["build"]["schem_hash"], build["schem_hash"])

    def test_recrop_needs_a_source(self):
        """Builds added before crops were kept say so rather than failing obscurely."""
        import schem_fixtures as sfx
        path = sfx.sponge_v2(self.tmp / "plain.schem", 3, 3, 3)
        res = self.client.post("/api/builds", data={
            "schem": (io.BytesIO(Path(path).read_bytes()), "plain.schem"), "name": "Plain"},
            content_type="multipart/form-data")
        build = res.get_json()["build"]
        self.assertFalse(build["can_recrop"])
        again = self.client.post("/api/builds/{}/recrop".format(build["id"]),
                                 json={"min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1,
                                       "min_z": 0, "max_z": 1})
        self.assertEqual(again.status_code, 409)
        self.assertEqual(again.get_json()["error"], "no_source")

    def test_an_oversized_recrop_is_refused(self):
        body = self.imported_schematic()
        build = self.client.post(
            "/api/import/{}/save".format(body["token"]),
            json=dict(body["bounds"], name="Whole")).get_json()["build"]
        res = self.client.post("/api/builds/{}/recrop".format(build["id"]),
                               json={"min_x": 0, "max_x": 3000, "min_y": 0, "max_y": 300,
                                     "min_z": 0, "max_z": 3000})
        self.assertEqual(res.status_code, 400)

    def test_a_java_world_imports_end_to_end(self):
        body = self.drop(tower_world(self.tmp / "w.zip"), "w.zip").get_json()
        self.assertEqual(body["kind"], "java_world")
        res = self.client.post("/api/import/{}/save".format(body["token"]),
                               json=dict(body["auto_bbox"], name="Tower"))
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.get_json()["build"]["dims"], "1x3x1")

    def test_an_unimportable_zip_reports_its_contents(self):
        path = self.tmp / "junk.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("notes.txt", b"x")
        res = self.drop(path, "junk.zip")
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertEqual(body["kind"], "unknown")
        self.assertIn("notes.txt", body["contents"])

    def test_a_zip_of_schematics_adds_them_as_a_batch(self):
        import schem_fixtures as sfx
        path = self.tmp / "pack.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.write(sfx.sponge_v2(self.tmp / "a.schem", 2, 2, 2), "a.schem")
            archive.write(sfx.sponge_v2(self.tmp / "b.schem", 3, 3, 3), "b.schem")
        body = self.drop(path, "pack.zip").get_json()
        self.assertEqual(body["kind"], "zip_schematics")

        res = self.client.post("/api/import/{}/save-schematics".format(body["token"]),
                               json={})
        self.assertEqual(res.status_code, 201)
        self.assertEqual(len(res.get_json()["added"]), 2)
        self.assertEqual(len(db.list_builds()), 2)

    def test_the_crop_travels_in_the_sidecar(self):
        body = self.imported_schematic()
        crop = {"min_x": 1, "max_x": 4, "min_y": 0, "max_y": 3, "min_z": 2, "max_z": 5}
        build = self.client.post("/api/import/{}/save".format(body["token"]),
                                 json=dict(crop, name="Cropped")).get_json()["build"]
        meta = db.read_meta(db.build_dir(build["id"], build["slug"]))
        self.assertIsNotNone(meta.get("crop"), "a clone needs to know how it was cropped")


if __name__ == "__main__":
    unittest.main()
