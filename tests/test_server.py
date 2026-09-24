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
import thumbs  # noqa: E402

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

    def add_build(self, name="House", dims=(6, 5, 6)):
        path = Path(fx.sponge_v2(self.tmp / (name + ".schem"), *dims))
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


class BuildNumberTests(PairedLibraryTests):
    """The number shown beside every build's name, which is its id.

    It is what the folder on disk is named after and what the CSV reports, so it has to be
    reachable from the search box and usable as a sort -- otherwise "build #7" is a label
    you can read but not act on.
    """

    def test_a_number_finds_exactly_that_build(self):
        first, second = self.add_build("One"), self.add_build("Two")
        found = self.client.get(
            "/api/builds?q={}".format(second["id"])).get_json()["builds"]
        self.assertEqual([b["id"] for b in found], [second["id"]])
        self.assertNotIn(first["id"], [b["id"] for b in found])

    def test_the_hash_it_is_displayed_with_is_accepted_too(self):
        build = self.add_build("One")
        found = self.client.get(
            "/api/builds?q=%23{}".format(build["id"])).get_json()["builds"]
        self.assertEqual([b["id"] for b in found], [build["id"]])

    def test_a_number_still_matches_text_containing_it(self):
        """The id match is an extra, not a special case: a name with the digits still hits."""
        named = self.add_build("House 2 by the lake")
        found = self.client.get("/api/builds?q=2").get_json()["builds"]
        self.assertIn(named["id"], [b["id"] for b in found])

    def test_the_library_can_be_ordered_by_number(self):
        ids = [self.add_build("B{}".format(n))["id"] for n in range(3)]
        ascending = self.client.get("/api/builds?sort=id&dir=asc").get_json()["builds"]
        self.assertEqual([b["id"] for b in ascending], sorted(ids))
        descending = self.client.get("/api/builds?sort=id&dir=desc").get_json()["builds"]
        self.assertEqual([b["id"] for b in descending], sorted(ids, reverse=True))

    def test_no_two_builds_share_a_number(self):
        """The point of the number: it names one build, including among saved duplicates,
        which are the one case where the name, the dimensions and the hash all match."""
        path = Path(fx.sponge_v2(self.tmp / "same.schem", 6, 5, 6))
        ids = []
        for _ in range(4):
            res = self.client.post(
                "/api/builds",
                data={"schem": (io.BytesIO(path.read_bytes()), path.name),
                      "name": "Same", "allow_duplicate": "true"},
                content_type="multipart/form-data")
            self.assertEqual(res.status_code, 201)
            ids.append(res.get_json()["build"]["id"])
        self.assertEqual(len(set(ids)), len(ids))


class BlockShapeTests(unittest.TestCase):
    """The inspect view draws a block's real shape, not a cube for everything.

    Every roof in the library is stairs and slabs, so drawing them as cubes makes a roof
    unreadable -- and drawing them *wrongly* is worse, because it looks like the schematic
    is broken when it is not.
    """

    def setUp(self):
        self.axes = thumbs._view_axes(35, 30)
        self.cell = 40

    def sprite(self, base, state):
        right, down, near = self.axes
        boxes = thumbs._shape_boxes(base, state)
        shade, mask, off_x, off_y = thumbs._orbit_sprite(
            self.cell, right, down, near, boxes, grow=False)
        return shade, mask, off_x, off_y

    def shade_at(self, shade, off_x, off_y, point):
        """The shade the stencil carries at one point of the unit cube."""
        right, down, _ = self.axes
        px = sum(v * a for v, a in zip(point, right)) * self.cell + off_x
        py = sum(v * a for v, a in zip(point, down)) * self.cell + off_y
        return float(shade[int(round(py)), int(round(px))])

    def test_a_slab_fills_half_the_block(self):
        _, cube, _, _ = self.sprite("stone", {})
        _, bottom, _, _ = self.sprite("stone_slab", {"type": "bottom"})
        _, double, _, _ = self.sprite("stone_slab", {"type": "double"})
        self.assertLess(bottom.sum(), cube.sum())
        self.assertEqual(double.sum(), cube.sum(), "a double slab is a full block")

    def test_a_stairs_tall_half_is_on_the_side_it_faces(self):
        """Measured off the library's own roofs: 79% of roof stairs face uphill, which is
        only geometrically possible if `facing` names the full-height half."""
        for facing, inside in (("north", (0.5, 0.9, 0.25)), ("south", (0.5, 0.9, 0.75)),
                               ("west", (0.25, 0.9, 0.5)), ("east", (0.75, 0.9, 0.5))):
            with self.subTest(facing):
                boxes = thumbs._shape_boxes("oak_stairs", {"facing": facing,
                                                           "half": "bottom"})
                tall = [b for b in boxes if b[1] >= 0.5][0]
                x, y, z = inside
                self.assertTrue(b_contains(tall, x, y, z),
                                "the tall half should cover the {} side".format(facing))

    def test_a_stairs_riser_is_not_painted_over_by_its_own_tread(self):
        """A stair is two boxes that overlap on screen, so they have to be drawn back to
        front. Drawn the other way round the tread covers the riser, and a roof of stairs
        comes out looking like it has bites taken out of every course -- which reads as a
        corrupt schematic rather than as a rendering bug."""
        shade, _, off_x, off_y = self.sprite("oak_stairs", {"facing": "north",
                                                            "half": "bottom"})
        riser = self.shade_at(shade, off_x, off_y, (0.5, 0.75, 0.5))
        self.assertAlmostEqual(riser, thumbs._face_shade((0, 0, 1)), places=3,
                               msg="the riser is showing the tread's shade")
        tread = self.shade_at(shade, off_x, off_y, (0.5, 0.5, 0.8))
        self.assertAlmostEqual(tread, thumbs.SHADE_TOP, places=3)

    def test_shapes_that_tile_are_grown_and_thin_ones_are_not(self):
        """Blocks that butt against their neighbours get a pixel of overlap so rounding
        cannot open a seam between them; a fence post has space around it and must not."""
        self.assertTrue(thumbs._tiles("stone"))
        self.assertTrue(thumbs._tiles("stone_brick_stairs"))
        self.assertTrue(thumbs._tiles("oak_slab"))
        self.assertFalse(thumbs._tiles("glass_pane"))
        self.assertFalse(thumbs._tiles("oak_fence"))

    def test_an_unknown_block_is_drawn_as_a_cube(self):
        self.assertEqual(thumbs._shape_boxes("some_future_block", {}), thumbs._CUBE_BOX)


def b_contains(box, x, y, z):
    x0, y0, z0, x1, y1, z1 = box
    return x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1


class InspectViewTests(PairedLibraryTests):
    """The big render behind Inspect: a camera at any angle, and a cutaway height."""

    DEFAULT = {"yaw": thumbs.VIEW_DEFAULT_YAW, "pitch": thumbs.VIEW_DEFAULT_PITCH}

    def view(self, build, **params):
        query = "&".join("{}={}".format(k, v) for k, v in params.items())
        return self.fetch("/api/builds/{}/view.png{}".format(
            build["id"], ("?" + query) if query else ""))

    def image_of(self, response):
        from PIL import Image
        return Image.open(io.BytesIO(response.data)).convert("RGBA")

    def view_file(self, build):
        return db.build_dir(build["id"], build["slug"]) / "thumb" / "view.png"

    def test_the_view_is_rendered_bigger_than_a_thumbnail(self):
        build = self.add_build()
        res = self.view(build, **self.DEFAULT)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["Content-Type"], "image/png")
        self.assertEqual(self.image_of(res).size, (thumbs.VIEW_SIZE, thumbs.VIEW_SIZE))
        self.assertGreater(thumbs.VIEW_SIZE, thumbs.SIZE)

    def test_the_bare_url_is_the_opening_angle(self):
        """Both angles default to where the viewer opens, so the plain URL is that picture."""
        build = self.add_build()
        self.assertEqual(self.view(build).data, self.view(build, **self.DEFAULT).data)

    def test_only_the_opening_angle_is_cached_on_disk(self):
        """One angle is worth a file; a camera you can drag is not a cache."""
        build = self.add_build()
        self.view(build, **self.DEFAULT)
        self.assertTrue(self.view_file(build).exists())

        self.view(build, yaw=110, pitch=20)
        self.view(build, **dict(self.DEFAULT, layer=1))
        self.assertEqual(sorted(p.name for p in self.view_file(build).parent.glob("view*.png")),
                         ["view.png"])

    def test_any_angle_renders_not_just_the_quarter_turns(self):
        """The point of the free camera: 17 degrees is as legal a frame as 90."""
        seen = set()
        build = self.add_build()
        for yaw, pitch in ((0, 35), (17, 35), (90, 35), (203, 5), (45, 80)):
            res = self.view(build, yaw=yaw, pitch=pitch)
            self.assertEqual(res.status_code, 200, "{}/{}".format(yaw, pitch))
            self.assertEqual(self.image_of(res).size, (thumbs.VIEW_SIZE, thumbs.VIEW_SIZE))
            seen.add(res.data)
        self.assertEqual(len(seen), 5, "every angle should be a different picture")

    def test_yaw_wraps_and_pitch_is_clamped(self):
        """Yaw is a bearing, so it comes round; pitch stops, because past vertical there is
        nothing to see but the build upside down."""
        build = self.add_build()
        self.assertEqual(self.view(build, yaw=370, pitch=35).data,
                         self.view(build, yaw=10, pitch=35).data)
        self.assertEqual(self.view(build, yaw=0, pitch=500).data,
                         self.view(build, yaw=0, pitch=thumbs.VIEW_PITCH_MAX).data)
        self.assertEqual(self.view(build, yaw=0, pitch=-500).data,
                         self.view(build, yaw=0, pitch=thumbs.VIEW_PITCH_MIN).data)

    def test_a_preview_is_smaller_than_the_frame_it_stands_in_for(self):
        """What a drag in progress asks for: the next angle matters more than this one's
        detail, so it is rendered small and replaced when the camera stops."""
        build = self.add_build()
        preview = self.view(build, yaw=70, pitch=30, preview=1)
        self.assertEqual(self.image_of(preview).size,
                         (thumbs.VIEW_PREVIEW_SIZE, thumbs.VIEW_PREVIEW_SIZE))
        self.assertLess(thumbs.VIEW_PREVIEW_SIZE, thumbs.VIEW_SIZE)

    def test_a_cutaway_is_a_different_picture(self):
        build = self.add_build()
        whole = self.view(build, **self.DEFAULT)
        cut = self.view(build, **dict(self.DEFAULT, layer=1))
        self.assertEqual(cut.status_code, 200)
        self.assertNotEqual(cut.data, whole.data)

    def test_a_cutaway_keeps_the_frame_of_the_build_it_was_cut_from(self):
        """Same scale, same place. A slider that re-centred the build at every height would
        make the one comparison it exists for -- this floor against the one below -- impossible."""
        build = self.add_build()
        whole = self.image_of(self.view(build, **self.DEFAULT)).getbbox()
        cut = self.image_of(self.view(build, **dict(self.DEFAULT, layer=1))).getbbox()
        self.assertEqual((whole[0], whole[2]), (cut[0], cut[2]))   # same width, same place
        self.assertGreater(cut[1], whole[1])       # the roof is gone, so content starts lower

    def test_turning_keeps_the_build_at_one_scale(self):
        """The camera is fitted against the footprint's diagonal rather than against the angle
        it is currently at, so spinning it doesn't pump the build in and out of the frame.

        A silhouette does legitimately narrow and widen as a rectangle turns, so the invariant
        is checked on quarter-turns of a square-footprint build, which are congruent: at one
        scale they measure the same, at a per-angle scale they would not."""
        build = self.add_build("Square", dims=(5, 5, 5))
        widths, heights = [], []
        for yaw in (0, 90, 180, 270):
            box = self.image_of(self.view(build, yaw=yaw, pitch=35)).getbbox()
            widths.append(box[2] - box[0])
            heights.append(box[3] - box[1])
        self.assertLessEqual(max(widths) - min(widths), 2, "quarter-turns measure the same")
        self.assertLessEqual(max(heights) - min(heights), 2)

        # And no angle -- including the widest, across the diagonal -- overflows the frame.
        for yaw in (15, 45, 135):
            box = self.image_of(self.view(build, yaw=yaw, pitch=35)).getbbox()
            self.assertLess(box[0], box[2])
            self.assertLessEqual(box[2], thumbs.VIEW_SIZE)
            self.assertLessEqual(box[3], thumbs.VIEW_SIZE)

    def test_the_top_layer_is_served_as_the_whole_build(self):
        """A slider at the top asks for the picture already on disk, not a re-render."""
        build = self.add_build()
        whole = self.view(build, **self.DEFAULT)
        self.assertEqual(self.view(build, **dict(self.DEFAULT, layer=build["h"] - 1)).data,
                         whole.data)
        self.assertEqual(self.view(build, **dict(self.DEFAULT, layer=9999)).data, whole.data)

    def test_a_cut_below_the_build_lands_on_its_bottom_layer(self):
        """The slider cannot go below 1, so a lower number is a bad URL, not a new view."""
        build = self.add_build()
        self.assertEqual(self.view(build, **dict(self.DEFAULT, layer=-5)).data,
                         self.view(build, **dict(self.DEFAULT, layer=0)).data)

    def test_an_unknown_build_has_no_view(self):
        self.assertEqual(self.fetch("/api/builds/9999/view.png").status_code, 404)

    def test_re_cropping_throws_the_cached_view_away(self):
        """A view that outlived its schematic is a picture of a build that isn't there."""
        build = self.add_build()
        self.view(build, **self.DEFAULT)
        self.assertTrue(self.view_file(build).exists())

        res = self.client.post("/api/builds/{}/recrop".format(build["id"]),
                               json={"min_x": 0, "max_x": 3, "min_y": 0, "max_y": 3,
                                     "min_z": 0, "max_z": 3})
        self.assertEqual(res.status_code, 200)
        self.assertFalse(self.view_file(build).exists())


if __name__ == "__main__":
    unittest.main()
