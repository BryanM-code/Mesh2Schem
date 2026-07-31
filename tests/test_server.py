"""API-level tests for the upload endpoints.

The data layer is pointed at a throwaway directory *before* server is imported, since
importing it runs db.init_db(); the real library under data/ is never touched.
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

import db

SANDBOX = Path(tempfile.mkdtemp(prefix="bdc-server-tests-"))
db.PROJECT_ROOT = SANDBOX
db.DATA_DIR = SANDBOX
db.DB_PATH = SANDBOX / "collector.db"
db.BUILDS_DIR = SANDBOX / "builds"

import schem_convert  # noqa: E402
import schem_fixtures as fx  # noqa: E402
import server  # noqa: E402

server.TMP_DIR = SANDBOX / ".tmp"
server.TMP_DIR.mkdir(parents=True, exist_ok=True)
db.init_db()


def tearDownModule():
    shutil.rmtree(SANDBOX, ignore_errors=True)


class SchematicFilenameTests(unittest.TestCase):
    def test_accepts_schematic_extension(self):
        self.assertTrue(server._is_schematic_filename("suburbanhouse310695835.schematic"))
        self.assertTrue(server._is_schematic_filename("house.schem"))
        self.assertTrue(server._is_schematic_filename("HOUSE.SCHEMATIC"))

    def test_rejects_non_schematic_extensions(self):
        self.assertFalse(server._is_schematic_filename("house.glb"))
        self.assertFalse(server._is_schematic_filename("house.txt"))
        self.assertFalse(server._is_schematic_filename(None))


class UploadEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-upload-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        # A fresh library per test. Conversion is deterministic, so two tests saving the
        # same fixture into one database would (correctly) collide on the dedup hash.
        db.PROJECT_ROOT = db.DATA_DIR = self.tmp
        db.DB_PATH = self.tmp / "collector.db"
        db.BUILDS_DIR = self.tmp / "builds"
        server.TMP_DIR = self.tmp / ".tmp"
        server.TMP_DIR.mkdir(parents=True, exist_ok=True)
        db.init_db()

    def post(self, url, path, field="schem", **form):
        payload = Path(path).read_bytes()
        data = {field: (io.BytesIO(payload), os.path.basename(path))}
        data.update(form)
        return self.client.post(url, data=data, content_type="multipart/form-data")

    # ── inspect ───────────────────────────────────────────────────────────────

    def test_inspect_converts_a_legacy_schematic(self):
        res = self.post("/api/inspect", fx.legacy_house(self.tmp / "house.schematic"))
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body["converted"])
        self.assertEqual(body["source_format"], "mcedit")
        self.assertEqual(body["dims"], "5x4x5")
        self.assertTrue(body["warnings"])

    def test_inspect_leaves_a_modern_schem_alone(self):
        res = self.post("/api/inspect", fx.sponge_v2(self.tmp / "small.schem", 3, 2, 4))
        body = res.get_json()
        self.assertFalse(body["converted"])
        self.assertEqual(body["source_format"], "sponge_v2")
        self.assertEqual(body["dims"], "3x2x4")

    def test_inspect_reports_the_hash_of_what_would_be_saved(self):
        """The hash must describe the converted .schem, not the legacy upload."""
        src = fx.legacy_house(self.tmp / "house.schematic")
        reported = self.post("/api/inspect", src).get_json()["hash"]
        converted = schem_convert.ensure_sponge(src, self.tmp / "out.schem").path
        from schem_meta import sha256_file
        self.assertEqual(reported, sha256_file(converted))

    def test_inspect_rejects_a_non_schematic_extension(self):
        res = self.client.post("/api/inspect",
                               data={"schem": (io.BytesIO(b"x"), "model.glb")},
                               content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        self.assertIn(".schematic", res.get_json()["error"])

    def test_inspect_rejects_a_file_that_only_looks_like_a_schematic(self):
        bad = fx.raw(self.tmp / "trojan.schem", b"MZ\x90\x00" + b"\x00" * 512)
        res = self.post("/api/inspect", bad)
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["code"], "bad_magic")

    def test_inspect_rejects_a_decompression_bomb(self):
        res = self.post("/api/inspect", fx.gzip_bomb(self.tmp / "bomb.schem", 250))
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["code"], "decompression_bomb")

    # ── rejection reporting ───────────────────────────────────────────────────

    def test_a_rejection_reports_what_and_where(self):
        """The whole point: a bad download is diagnosable from the response alone."""
        bad = fx.gzipped(self.tmp / "list.schematic", fx.huge_list_nbt())
        res = self.post("/api/inspect", bad)
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertEqual(body["code"], "list_too_long")
        self.assertEqual(body["stage"], "structure")
        self.assertEqual(body["path"], "list")
        self.assertIsInstance(body["offset"], int)
        self.assertEqual(body["filename"], "list.schematic")
        # `error` is what the UI shows: the message plus an appended location.
        # `message` is the bare text, with no location baked into it.
        self.assertTrue(body["error"].startswith(body["message"]))
        self.assertTrue(body["error"].endswith(
            "(at {}, byte {})".format(body["path"], body["offset"])))
        self.assertNotIn("(at ", body["message"])

    def test_rejections_from_every_stage_carry_a_code(self):
        from nbtlib import ByteArray, Compound, Short
        cases = {
            "empty_file": fx.raw(self.tmp / "a.schem", b""),
            "bad_magic": fx.raw(self.tmp / "b.schem", b"<html>"),
            "too_deep": fx.gzipped(self.tmp / "c.schem", fx.deep_nbt()),
            "truncated_blocks": fx.write_nbt(
                Compound({"Width": Short(9), "Height": Short(9), "Length": Short(9),
                          "Blocks": ByteArray([0])}), self.tmp / "d.schem"),
        }
        for expected, path in cases.items():
            with self.subTest(expected):
                res = self.post("/api/inspect", path)
                self.assertEqual(res.status_code, 400)
                self.assertEqual(res.get_json()["code"], expected)

    def test_saving_reports_the_same_rejection_as_inspecting(self):
        bad = fx.raw(self.tmp / "bad.schematic", b"PK\x03\x04not a schematic")
        inspected = self.post("/api/inspect", bad).get_json()
        saved = self.post("/api/builds", bad, name="Nope")
        self.assertEqual(saved.status_code, 400)
        self.assertEqual(saved.get_json()["code"], inspected["code"])

    # ── create ────────────────────────────────────────────────────────────────

    def test_saving_a_legacy_schematic_stores_a_real_schem(self):
        res = self.post("/api/builds", fx.legacy_house(self.tmp / "house.schematic"),
                        name="Legacy House")
        self.assertEqual(res.status_code, 201)
        body = res.get_json()
        self.assertTrue(body["converted"])

        build = body["build"]
        stored = db.build_dir(build["id"], build["slug"]) / "schem" / "original.schem"
        self.assertTrue(stored.exists())
        self.assertEqual(schem_convert.inspect(stored).format, "sponge_v2")
        self.assertEqual(build["dims"], "5x4x5")

    def test_re_adding_the_same_legacy_file_is_caught_as_a_duplicate(self):
        """Conversion is deterministic, so the second upload hashes to the same bytes."""
        src = fx.legacy_house(self.tmp / "dupe.schematic")
        self.assertEqual(self.post("/api/builds", src, name="First").status_code, 201)
        res = self.post("/api/builds", src, name="Second")
        self.assertEqual(res.status_code, 409)
        self.assertEqual(res.get_json()["error"], "duplicate")

    def test_uploads_leave_no_temp_files_behind(self):
        self.post("/api/inspect", fx.legacy_house(self.tmp / "a.schematic"))
        self.post("/api/inspect", fx.raw(self.tmp / "b.schem", b"nope"))
        self.assertEqual(list(server.TMP_DIR.iterdir()), [])


class PairedLibraryTests(unittest.TestCase):
    """Shared setup for the two features that act on an existing, paired build."""

    def setUp(self):
        self.client = server.app.test_client()
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-paired-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        db.PROJECT_ROOT = db.DATA_DIR = self.tmp
        db.DB_PATH = self.tmp / "collector.db"
        db.BUILDS_DIR = self.tmp / "builds"
        server.TMP_DIR = self.tmp / ".tmp"
        server.TMP_DIR.mkdir(parents=True, exist_ok=True)
        db.init_db()

    def add_build(self, name="House"):
        path = Path(fx.sponge_v2(self.tmp / (name + ".schem"), 6, 5, 6))
        res = self.client.post(
            "/api/builds",
            data={"schem": (io.BytesIO(path.read_bytes()), path.name), "name": name},
            content_type="multipart/form-data")
        self.assertEqual(res.status_code, 201)
        return res.get_json()["build"]

    def glb_bytes(self):
        """A minimal but genuinely loadable mesh -- validate_glb insists on real faces."""
        import trimesh
        out = self.tmp / "mesh.glb"
        trimesh.creation.box(extents=[1, 1, 1]).export(out)
        return out.read_bytes()

    def image_bytes(self, size=(240, 180), fmt="PNG", name="ref.png"):
        from PIL import Image
        out = self.tmp / name
        Image.new("RGB", size, (90, 120, 160)).save(out, fmt)
        return out.read_bytes()

    def pair(self, build_id, **form):
        data = {"glb": (io.BytesIO(self.glb_bytes()), "mesh.glb")}
        data.update(form)
        return self.client.post("/api/builds/{}/pair".format(build_id), data=data,
                                content_type="multipart/form-data")

    def attach_image(self, build_id, payload=None, name="ref.png", **form):
        data = {"image": (io.BytesIO(payload or self.image_bytes()), name)}
        data.update(form)
        return self.client.post("/api/builds/{}/image".format(build_id), data=data,
                                content_type="multipart/form-data")

    def files_of(self, build):
        folder = db.build_dir(build["id"], build["slug"])
        return (folder / "trellis" / "output.glb", folder / "thumb" / "mesh.png",
                folder / "thumb" / "schem.png")

    def status_of(self, build_id):
        return self.client.get("/api/builds/{}".format(build_id)).get_json()["build"]["status"]

    def fetch(self, url):
        """GET a file response, buffering the body and releasing the handle.

        Routes served by send_file hold an open file until the response is closed. The
        real server closes it when the response is written; the test client does not, and
        on Windows a still-open handle would also block the very deletions these tests go
        on to assert.
        """
        res = self.client.get(url)
        res.get_data()
        res.close()
        return res


class ThumbnailEndpointTests(PairedLibraryTests):
    def test_a_schematic_thumbnail_is_rendered_and_cached_on_disk(self):
        build = self.add_build()
        res = self.fetch("/api/builds/{}/thumb/schem.png".format(build["id"]))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["Content-Type"], "image/png")
        self.assertTrue(res.data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(res.data), 500)
        self.assertTrue(self.files_of(build)[2].exists())

    def test_a_second_request_reuses_the_cached_render(self):
        """Thumbnails are a cache, not a per-request render -- the file must not be rewritten."""
        build = self.add_build()
        url = "/api/builds/{}/thumb/schem.png".format(build["id"])
        self.fetch(url)
        cached = self.files_of(build)[2]
        stamp = cached.stat().st_mtime_ns
        self.assertEqual(self.fetch(url).status_code, 200)
        self.assertEqual(cached.stat().st_mtime_ns, stamp)

    def test_an_unpaired_build_has_no_mesh_thumbnail(self):
        build = self.add_build()
        res = self.fetch("/api/builds/{}/thumb/mesh.png".format(build["id"]))
        self.assertEqual(res.status_code, 404)

    def test_pairing_produces_a_mesh_thumbnail(self):
        build = self.add_build()
        self.assertEqual(self.pair(build["id"]).status_code, 200)
        res = self.fetch("/api/builds/{}/thumb/mesh.png".format(build["id"]))
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.data.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_an_unknown_kind_is_not_a_path(self):
        """`kind` selects a renderer from a whitelist; it never reaches the filesystem."""
        build = self.add_build()
        for kind in ("evil", "../../collector.db", "schem/../../.."):
            with self.subTest(kind):
                res = self.fetch(
                    "/api/builds/{}/thumb/{}.png".format(build["id"], kind))
                self.assertEqual(res.status_code, 404)

    def test_an_unknown_build_has_no_thumbnail(self):
        self.assertEqual(self.fetch("/api/builds/9999/thumb/schem.png").status_code, 404)


class UnpairEndpointTests(PairedLibraryTests):
    def test_removing_a_pair_clears_the_row_and_deletes_the_mesh(self):
        build = self.add_build()
        self.attach_image(build["id"])       # so the build steps back to needs_trellis
        self.pair(build["id"])
        glb, mesh_png, _ = self.files_of(build)
        self.fetch("/api/builds/{}/thumb/mesh.png".format(build["id"]))
        self.assertTrue(glb.exists() and mesh_png.exists())

        res = self.client.delete("/api/builds/{}/pair?confirm=true".format(build["id"]))
        self.assertEqual(res.status_code, 200)
        after = res.get_json()["build"]
        self.assertEqual(after["status"], "needs_trellis")
        self.assertIsNone(after["glb_path"])
        self.assertIsNone(after["paired_at"])
        self.assertFalse(glb.exists())
        self.assertFalse(mesh_png.exists(), "the stale mesh picture must go with the mesh")

    def test_removing_a_pair_keeps_the_build_and_its_schematic(self):
        """The point of unpairing rather than deleting: only the .glb was wrong."""
        build = self.add_build(name="Keeper")
        self.pair(build["id"])
        self.client.delete("/api/builds/{}/pair?confirm=true".format(build["id"]))

        still = self.client.get("/api/builds/{}".format(build["id"])).get_json()["build"]
        self.assertEqual(still["name"], "Keeper")
        self.assertEqual(still["schem_hash"], build["schem_hash"])
        self.assertTrue((db.build_dir(build["id"], build["slug"]) / "schem" /
                         "original.schem").exists())

    def test_removing_a_pair_requires_confirmation(self):
        build = self.add_build()
        self.pair(build["id"])
        res = self.client.delete("/api/builds/{}/pair".format(build["id"]))
        self.assertEqual(res.status_code, 400)
        self.assertTrue((db.build_dir(build["id"], build["slug"]) / "trellis" /
                         "output.glb").exists())

    def test_removing_a_pair_that_isnt_there_is_a_conflict(self):
        build = self.add_build()
        res = self.client.delete("/api/builds/{}/pair?confirm=true".format(build["id"]))
        self.assertEqual(res.status_code, 409)
        self.assertEqual(res.get_json()["error"], "not_paired")

    def test_removing_a_pair_from_an_unknown_build_is_a_404(self):
        self.assertEqual(
            self.client.delete("/api/builds/9999/pair?confirm=true").status_code, 404)

    def test_rebuild_from_disk_does_not_resurrect_a_removed_pair(self):
        """The reason the .glb is deleted outright: rebuild_from_disk reads paired-ness
        straight off the filesystem, so anything left behind would silently re-pair."""
        build = self.add_build()
        self.attach_image(build["id"])
        self.pair(build["id"])
        self.client.delete("/api/builds/{}/pair?confirm=true".format(build["id"]))

        self.assertEqual(self.client.post("/api/admin/rebuild").status_code, 200)
        after = self.client.get("/api/builds/{}".format(build["id"])).get_json()["build"]
        self.assertEqual(after["status"], "needs_trellis")
        self.assertIsNone(after["glb_path"])

    def test_a_build_can_be_paired_again_after_unpairing(self):
        """The whole workflow: wrong file dropped, removed, right one dropped."""
        build = self.add_build()
        self.attach_image(build["id"])
        self.pair(build["id"])
        self.client.delete("/api/builds/{}/pair?confirm=true".format(build["id"]))
        res = self.pair(build["id"])
        self.assertEqual(res.status_code, 200, "re-pairing must not need replace=true")
        self.assertEqual(res.get_json()["build"]["status"], "complete")


class PipelineStatusTests(PairedLibraryTests):
    """Status is derived from which files exist, and names the first one missing."""

    def test_a_new_build_needs_an_image_first(self):
        build = self.add_build()
        self.assertEqual(build["status"], "needs_image")
        self.assertFalse(build["has_image"])
        self.assertFalse(build["has_glb"])

    def test_the_three_stages_advance_in_order(self):
        build = self.add_build()
        self.assertEqual(self.status_of(build["id"]), "needs_image")
        self.attach_image(build["id"])
        self.assertEqual(self.status_of(build["id"]), "needs_trellis")
        self.pair(build["id"])
        self.assertEqual(self.status_of(build["id"]), "complete")

    def test_removing_an_artefact_steps_the_status_back(self):
        build = self.add_build()
        self.attach_image(build["id"])
        self.pair(build["id"])
        self.client.delete("/api/builds/{}/pair?confirm=true".format(build["id"]))
        self.assertEqual(self.status_of(build["id"]), "needs_trellis")
        self.client.delete("/api/builds/{}/image?confirm=true".format(build["id"]))
        self.assertEqual(self.status_of(build["id"]), "needs_image")

    def test_a_mesh_without_an_image_still_reads_as_needing_the_image(self):
        """Status names what's missing, so a glb alone doesn't make a build complete."""
        build = self.add_build()
        self.pair(build["id"])
        self.assertEqual(self.status_of(build["id"]), "needs_image")
        self.assertTrue(self.client.get(
            "/api/builds/{}".format(build["id"])).get_json()["build"]["has_glb"])

    def test_rebuild_from_disk_derives_all_three_stages(self):
        build = self.add_build()
        self.attach_image(build["id"])
        self.pair(build["id"])
        self.assertEqual(self.client.post("/api/admin/rebuild").status_code, 200)
        self.assertEqual(self.status_of(build["id"]), "complete")

    def test_the_status_filter_offers_all_three(self):
        a, b, c = self.add_build("A"), self.add_build("B"), self.add_build("C")
        self.attach_image(b["id"])
        self.attach_image(c["id"])
        self.pair(c["id"])
        for status, expected in (("needs_image", a["id"]), ("needs_trellis", b["id"]),
                                 ("complete", c["id"])):
            with self.subTest(status):
                got = self.client.get("/api/builds?status=" + status).get_json()["builds"]
                self.assertEqual([x["id"] for x in got], [expected])


class GeneratedImageTests(PairedLibraryTests):
    def test_an_image_is_stored_as_a_canonical_png(self):
        """Whatever is dropped, the library holds one predictable generated.png."""
        build = self.add_build()
        res = self.attach_image(build["id"], self.image_bytes(fmt="JPEG", name="r.jpg"),
                                name="photo.jpg")
        self.assertEqual(res.status_code, 200)
        stored = db.build_dir(build["id"], build["slug"]) / "image" / "generated.png"
        self.assertTrue(stored.exists())
        self.assertTrue(stored.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(res.get_json()["build"]["image_path"],
                         "builds/{}__{}/image/generated.png".format(build["id"], build["slug"]))

    def test_the_file_decides_not_the_extension(self):
        build = self.add_build()
        res = self.attach_image(build["id"], b"this is not an image", name="lie.png")
        self.assertEqual(res.status_code, 400)

    def test_a_too_small_image_is_rejected(self):
        build = self.add_build()
        res = self.attach_image(build["id"], self.image_bytes(size=(8, 8), name="s.png"))
        self.assertEqual(res.status_code, 400)
        self.assertIn("minimum", res.get_json()["error"])

    def test_replacing_an_image_needs_confirmation(self):
        build = self.add_build()
        self.attach_image(build["id"])
        self.assertEqual(self.attach_image(build["id"]).status_code, 409)
        self.assertEqual(self.attach_image(build["id"], replace="true").status_code, 200)

    def test_removing_an_image_deletes_the_file_and_its_thumbnail(self):
        build = self.add_build()
        self.attach_image(build["id"])
        self.fetch("/api/builds/{}/thumb/image.png".format(build["id"]))
        folder = db.build_dir(build["id"], build["slug"])
        self.assertTrue((folder / "image" / "generated.png").exists())
        self.assertTrue((folder / "thumb" / "image.png").exists())

        self.assertEqual(self.client.delete(
            "/api/builds/{}/image?confirm=true".format(build["id"])).status_code, 200)
        self.assertFalse((folder / "image" / "generated.png").exists())
        self.assertFalse((folder / "thumb" / "image.png").exists())

    def test_removing_an_image_requires_confirmation(self):
        build = self.add_build()
        self.attach_image(build["id"])
        self.assertEqual(self.client.delete(
            "/api/builds/{}/image".format(build["id"])).status_code, 400)

    def test_an_image_thumbnail_is_served(self):
        build = self.add_build()
        self.attach_image(build["id"])
        res = self.fetch("/api/builds/{}/thumb/image.png".format(build["id"]))
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.data.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_the_schematic_can_be_downloaded_for_screenshotting(self):
        """The fallback when there's no source page: load it in Minecraft yourself."""
        build = self.add_build(name="Tower")
        res = self.fetch("/api/builds/{}/schem".format(build["id"]))
        self.assertEqual(res.status_code, 200)
        self.assertIn("tower.schem", res.headers["Content-Disposition"])


class ReferenceScreenshotTests(PairedLibraryTests):
    """Up to MAX_SHOTS raw captures per build, optional and status-neutral."""

    def add_shots(self, build_id, count=1, payload=None, name="shot.png"):
        data = {"images": [(io.BytesIO(payload or self.image_bytes()), name)
                           for _ in range(count)]}
        return self.client.post("/api/builds/{}/shots".format(build_id), data=data,
                                content_type="multipart/form-data")

    def slots_of(self, build_id):
        build = self.client.get("/api/builds/{}".format(build_id)).get_json()["build"]
        return [s["slot"] for s in build["shots"]]

    def test_screenshots_fill_slots_in_order(self):
        build = self.add_build()
        res = self.add_shots(build["id"], 2)
        self.assertEqual(res.status_code, 200)
        self.assertEqual([a["slot"] for a in res.get_json()["added"]], [1, 2])
        self.assertEqual(self.slots_of(build["id"]), [1, 2])

    def test_screenshots_never_change_the_status(self):
        """They're working material, not a stage the pipeline waits on."""
        build = self.add_build()
        self.add_shots(build["id"], 3)
        self.assertEqual(self.status_of(build["id"]), "needs_image")

    def test_the_cap_is_enforced_and_the_overflow_is_reported(self):
        build = self.add_build()
        self.add_shots(build["id"], db.MAX_SHOTS)
        res = self.add_shots(build["id"], 1)
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertEqual(body["added"], [])
        self.assertEqual(len(body["rejected"]), 1)
        self.assertIn("at most", body["rejected"][0]["reason"])
        self.assertEqual(self.slots_of(build["id"]), [1, 2, 3])

    def test_a_partial_drop_keeps_what_fits(self):
        """Dropping four when one slot is free adds one and says why the rest didn't."""
        build = self.add_build()
        self.add_shots(build["id"], 2)
        res = self.add_shots(build["id"], 3)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.get_json()["added"]), 1)
        self.assertEqual(len(res.get_json()["rejected"]), 2)
        self.assertEqual(self.slots_of(build["id"]), [1, 2, 3])

    def test_a_removed_slot_is_reused(self):
        """Slots are stable ids, so removing the middle one leaves a gap to refill."""
        build = self.add_build()
        self.add_shots(build["id"], 3)
        self.client.delete("/api/builds/{}/shots/2?confirm=true".format(build["id"]))
        self.assertEqual(self.slots_of(build["id"]), [1, 3])

        res = self.add_shots(build["id"], 1)
        self.assertEqual([a["slot"] for a in res.get_json()["added"]], [2],
                         "the reported slot must be the gap, not the highest")
        self.assertEqual(self.slots_of(build["id"]), [1, 2, 3])

    def test_removing_a_screenshot_deletes_its_file_and_thumbnail(self):
        build = self.add_build()
        self.add_shots(build["id"], 1)
        self.fetch("/api/builds/{}/thumb/shot-1.png".format(build["id"]))
        folder = db.build_dir(build["id"], build["slug"])
        self.assertTrue((folder / "shots" / "shot-1.png").exists())
        self.assertTrue((folder / "thumb" / "shot-1.png").exists())

        res = self.client.delete("/api/builds/{}/shots/1?confirm=true".format(build["id"]))
        self.assertEqual(res.status_code, 200)
        self.assertFalse((folder / "shots" / "shot-1.png").exists())
        self.assertFalse((folder / "thumb" / "shot-1.png").exists())

    def test_removing_a_screenshot_requires_confirmation(self):
        build = self.add_build()
        self.add_shots(build["id"], 1)
        self.assertEqual(self.client.delete(
            "/api/builds/{}/shots/1".format(build["id"])).status_code, 400)
        self.assertEqual(self.slots_of(build["id"]), [1])

    def test_removing_an_empty_slot_is_a_404(self):
        build = self.add_build()
        self.assertEqual(self.client.delete(
            "/api/builds/{}/shots/2?confirm=true".format(build["id"])).status_code, 404)

    def test_a_bad_image_among_good_ones_is_reported_individually(self):
        build = self.add_build()
        data = {"images": [
            (io.BytesIO(self.image_bytes()), "good.png"),
            (io.BytesIO(b"not an image"), "bad.png"),
        ]}
        res = self.client.post("/api/builds/{}/shots".format(build["id"]), data=data,
                               content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.get_json()["added"]), 1)
        self.assertEqual(res.get_json()["rejected"][0]["filename"], "bad.png")

    def test_each_screenshot_gets_its_own_thumbnail(self):
        build = self.add_build()
        self.add_shots(build["id"], 2)
        for slot in (1, 2):
            with self.subTest(slot=slot):
                res = self.fetch("/api/builds/{}/thumb/shot-{}.png".format(build["id"], slot))
                self.assertEqual(res.status_code, 200)
                self.assertTrue(res.data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(self.fetch(
            "/api/builds/{}/thumb/shot-3.png".format(build["id"])).status_code, 404)

    def test_a_slot_outside_the_cap_is_not_a_thumbnail_kind(self):
        build = self.add_build()
        self.assertEqual(self.fetch(
            "/api/builds/{}/thumb/shot-4.png".format(build["id"])).status_code, 404)

    def test_screenshots_survive_a_rebuild_from_disk(self):
        build = self.add_build()
        self.add_shots(build["id"], 2)
        self.assertEqual(self.client.post("/api/admin/rebuild").status_code, 200)
        self.assertEqual(self.slots_of(build["id"]), [1, 2])

    def test_rebuild_from_disk_notices_a_screenshot_deleted_by_hand(self):
        build = self.add_build()
        self.add_shots(build["id"], 2)
        (db.build_dir(build["id"], build["slug"]) / "shots" / "shot-1.png").unlink()
        self.client.post("/api/admin/rebuild")
        self.assertEqual(self.slots_of(build["id"]), [2])

    def test_deleting_a_build_takes_its_screenshots_with_it(self):
        build = self.add_build()
        self.add_shots(build["id"], 2)
        self.client.delete("/api/builds/{}?confirm=true".format(build["id"]))
        conn = db.connect()
        try:
            left = conn.execute("SELECT COUNT(*) FROM shots WHERE build_id = ?",
                                (build["id"],)).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(left, 0, "the cascade should clear the shot rows")

    def test_the_build_reports_how_many_slots_are_free(self):
        build = self.add_build()
        got = self.client.get("/api/builds/{}".format(build["id"])).get_json()["build"]
        self.assertEqual(got["shot_count"], 0)
        self.assertEqual(got["shot_slots_free"], db.MAX_SHOTS)
        self.assertEqual(got["max_shots"], db.MAX_SHOTS)

        self.add_shots(build["id"], 1)
        got = self.client.get("/api/builds/{}".format(build["id"])).get_json()["build"]
        self.assertEqual(got["shot_count"], 1)
        self.assertEqual(got["shot_slots_free"], db.MAX_SHOTS - 1)

    def test_screenshots_appear_in_the_library_listing(self):
        build = self.add_build()
        self.add_shots(build["id"], 2)
        listed = self.client.get("/api/builds").get_json()["builds"][0]
        self.assertEqual(listed["shot_count"], 2)
        self.assertEqual([s["slot"] for s in listed["shots"]], [1, 2])


class StoredFileTests(PairedLibraryTests):
    """The stored bytes at full resolution -- the other half of the thumbnail route.

    A screenshot is only worth keeping if you can get the real file back out later to feed
    it to something else, so this must serve the original, not a 384px render of it.
    """

    def add_shot(self, build_id, size=(640, 480)):
        data = {"images": (io.BytesIO(self.image_bytes(size=size, name="s.png")), "s.png")}
        return self.client.post("/api/builds/{}/shots".format(build_id), data=data,
                                content_type="multipart/form-data")

    def test_a_screenshot_comes_back_at_its_stored_size(self):
        from PIL import Image
        build = self.add_build()
        self.add_shot(build["id"], size=(640, 480))

        full = self.fetch("/api/builds/{}/file/shot-1".format(build["id"]))
        thumb = self.fetch("/api/builds/{}/thumb/shot-1.png".format(build["id"]))
        self.assertEqual(full.status_code, 200)
        self.assertEqual(full.headers["Content-Type"], "image/png")
        self.assertEqual(Image.open(io.BytesIO(full.data)).size, (640, 480))
        self.assertNotEqual(Image.open(io.BytesIO(thumb.data)).size, (640, 480),
                            "the thumbnail must not be what the file route serves")

    def test_every_artefact_is_retrievable(self):
        build = self.add_build()
        self.add_shot(build["id"])
        self.attach_image(build["id"])
        self.pair(build["id"])
        for kind in ("schem", "shot-1", "image", "mesh"):
            with self.subTest(kind=kind):
                res = self.fetch("/api/builds/{}/file/{}".format(build["id"], kind))
                self.assertEqual(res.status_code, 200)
                self.assertGreater(len(res.data), 0)

    def test_files_are_named_after_the_build(self):
        build = self.add_build(name="Stone Tower")
        self.add_shot(build["id"])
        res = self.fetch("/api/builds/{}/file/shot-1?download=1".format(build["id"]))
        self.assertIn("stone-tower-shot-1.png", res.headers["Content-Disposition"])
        self.assertIn("attachment", res.headers["Content-Disposition"])

    def test_without_the_download_flag_it_opens_in_the_browser(self):
        build = self.add_build()
        self.add_shot(build["id"])
        res = self.fetch("/api/builds/{}/file/shot-1".format(build["id"]))
        self.assertNotIn("attachment", res.headers.get("Content-Disposition", ""))

    def test_an_absent_artefact_is_a_404_not_a_crash(self):
        build = self.add_build()
        for kind in ("shot-1", "image", "mesh"):
            with self.subTest(kind=kind):
                self.assertEqual(self.fetch(
                    "/api/builds/{}/file/{}".format(build["id"], kind)).status_code, 404)

    def test_the_kind_is_a_whitelist_not_a_path(self):
        build = self.add_build()
        for kind in ("evil", "shot-0", "shot-9", "shot-99", "collector.db", "..%2F..%2Fcollector.db"):
            with self.subTest(kind=kind):
                self.assertEqual(self.fetch(
                    "/api/builds/{}/file/{}".format(build["id"], kind)).status_code, 404)

    def test_an_unknown_build_has_no_files(self):
        self.assertEqual(self.fetch("/api/builds/9999/file/schem").status_code, 404)

    def test_a_pasted_image_is_accepted(self):
        """A clipboard paste arrives as a blob, so the client supplies the filename."""
        build = self.add_build()
        data = {"images": (io.BytesIO(self.image_bytes()), "pasted-1.png")}
        res = self.client.post("/api/builds/{}/shots".format(build["id"]), data=data,
                               content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        self.assertEqual([a["slot"] for a in res.get_json()["added"]], [1])


class ServerFreshnessTests(PairedLibraryTests):
    """The frontend is served from disk; the Python behind it is not.

    A server left running while the code changes serves a new app.js against an old API,
    and controls whose data never arrives just silently don't appear. /api/meta is how the
    page notices, so its contract matters.
    """

    def test_meta_reports_the_api_version(self):
        meta = self.client.get("/api/meta").get_json()
        self.assertEqual(meta["api_version"], server.API_VERSION)
        self.assertIsInstance(meta["api_version"], int)

    def test_meta_reports_the_limits_the_frontend_relies_on(self):
        meta = self.client.get("/api/meta").get_json()
        self.assertEqual(meta["max_shots"], db.MAX_SHOTS)
        self.assertEqual(meta["schema_version"], db.SCHEMA_VERSION)
        self.assertEqual(meta["statuses"], list(db.ALL_STATUSES))

    def test_the_api_version_covers_the_shot_fields(self):
        """Bumping API_VERSION is what tells an old page it's out of date, so the fields
        the current frontend needs must actually be present at this version."""
        build = self.add_build()
        payload = self.client.get("/api/builds/{}".format(build["id"])).get_json()["build"]
        for field in ("shots", "shot_count", "shot_slots_free", "max_shots"):
            with self.subTest(field=field):
                self.assertIn(field, payload)


class MetadataTests(PairedLibraryTests):
    def test_source_url_and_notes_can_be_edited(self):
        build = self.add_build()
        res = self.client.patch("/api/builds/{}".format(build["id"]),
                                json={"source_url": "https://example.com/build/1",
                                      "source_note": "CC-BY"})
        self.assertEqual(res.status_code, 200)
        after = res.get_json()["build"]
        self.assertEqual(after["source_url"], "https://example.com/build/1")
        self.assertEqual(after["source_note"], "CC-BY")

    def test_derived_fields_are_not_editable(self):
        """Only user-entered fields; everything else comes from the files."""
        build = self.add_build()
        self.client.patch("/api/builds/{}".format(build["id"]),
                          json={"status": "complete", "schem_hash": "forged",
                                "name": "Renamed"})
        after = self.client.get("/api/builds/{}".format(build["id"])).get_json()["build"]
        self.assertEqual(after["name"], "Renamed")
        self.assertEqual(after["status"], "needs_image")
        self.assertEqual(after["schem_hash"], build["schem_hash"])

    def test_search_covers_the_source_url(self):
        build = self.add_build()
        self.client.patch("/api/builds/{}".format(build["id"]),
                          json={"source_url": "https://planetminecraft.com/x"})
        found = self.client.get("/api/builds?q=planetminecraft").get_json()["builds"]
        self.assertEqual([b["id"] for b in found], [build["id"]])


if __name__ == "__main__":
    unittest.main()
