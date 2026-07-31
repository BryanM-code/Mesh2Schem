"""Tests for sharing one library between two machines via a git repo.

The repo carries `data/builds/`; the database is not tracked, because it is binary
(git cannot merge it) and derivable (it does not need to be). Three things have to hold
for that to work:

* what the user typed has to survive the trip, which is what meta.json is for;
* two machines must never mint the same build id, which is what the id blocks are for;
* starting the app after a clone or a pull has to reconcile the database with the folders
  that arrived, including ones that were deleted elsewhere.

The interesting case is a genuine merge, so that is simulated directly: two libraries are
built independently and their folders are combined, exactly as git would.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db  # noqa: E402
import schem_fixtures as fx  # noqa: E402


class MachineTestCase(unittest.TestCase):
    """Points db at a throwaway folder and lets a test pretend to be either machine."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-multi-")
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(self.become, 0)
        self.use(self.root / "pc")

    def use(self, data_dir: Path):
        """Point the data layer at one machine's copy of the library."""
        data_dir.mkdir(parents=True, exist_ok=True)
        db.PROJECT_ROOT = db.DATA_DIR = data_dir
        db.DB_PATH = data_dir / "collector.db"
        db.BUILDS_DIR = data_dir / "builds"
        db.init_db()

    def become(self, machine_id: int):
        """Pretend to be machine N, which is what decides the id block."""
        db.MACHINE_ID = machine_id

    def add(self, name, w=3, h=2, l=3):
        path = Path(fx.sponge_v2(self.root / (name + ".schem"), w, h, l))
        return db.create_build(name, "", path, __import__("schem_meta").sha256_file(path),
                               (w, h, l), allow_duplicate=True)


class MachineIdTests(MachineTestCase):
    def test_the_first_machine_numbers_from_one(self):
        self.become(0)
        self.assertEqual(self.add("first")["id"], 1)
        self.assertEqual(self.add("second", 4)["id"], 2)

    def test_a_second_machine_numbers_from_its_own_block(self):
        self.become(1)
        self.assertEqual(self.add("mac-first")["id"], 1_000_001)
        self.assertEqual(self.add("mac-second", 4)["id"], 1_000_002)

    def test_the_blocks_cannot_overlap(self):
        """The whole point: ids minted on one machine are never reachable by the other."""
        self.become(0)
        pc_ids = [self.add("a")["id"], self.add("b", 4)["id"]]
        self.become(1)
        mac_ids = [self.add("c", 5)["id"], self.add("d", 6)["id"]]
        self.assertFalse(set(pc_ids) & set(mac_ids))
        self.assertTrue(all(i < db.ID_BLOCK for i in pc_ids))
        self.assertTrue(all(i > db.ID_BLOCK for i in mac_ids))

    def test_a_machine_ignores_the_other_block_when_allocating(self):
        """Machine 0 must not jump to 1,000,003 just because machine 1's rows are there."""
        self.become(1)
        self.add("mac", 7)
        self.become(0)
        self.assertEqual(self.add("pc", 8)["id"], 1)


class MetaSidecarTests(MachineTestCase):
    def test_a_sidecar_is_written_beside_the_build(self):
        build = self.add("Suburban House")
        meta = json.loads((db.build_dir(build["id"], build["slug"]) / "meta.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(meta["name"], "Suburban House")
        self.assertEqual(meta["variant_no"], 0)
        self.assertEqual(meta["added_at"], build["added_at"])

    def test_editing_metadata_updates_the_sidecar(self):
        build = self.add("Before")
        db.update_build(build["id"], name="After", source_note="CC-BY",
                        source_url="https://example.com/x")
        meta = db.read_meta(db.build_dir(build["id"], "before"))
        self.assertEqual(meta["name"], "After")
        self.assertEqual(meta["source_note"], "CC-BY")
        self.assertEqual(meta["source_url"], "https://example.com/x")

    def test_a_rebuild_recovers_what_the_user_typed(self):
        """Without the sidecar this is the data that would be lost on another machine."""
        build = self.add("Grand Library")
        db.update_build(build["id"], source_note="from planetminecraft",
                        source_url="https://www.planetminecraft.com/")
        db.DB_PATH.unlink()                      # as if freshly cloned: no database at all
        db.init_db()
        db.rebuild_from_disk()

        recovered = db.get_build(build["id"])
        self.assertEqual(recovered["name"], "Grand Library")
        self.assertEqual(recovered["source_note"], "from planetminecraft")
        self.assertEqual(recovered["source_url"], "https://www.planetminecraft.com/")
        self.assertEqual(recovered["added_at"], build["added_at"])

    def test_a_folder_with_no_sidecar_still_rebuilds(self):
        build = self.add("Nameless")
        (db.build_dir(build["id"], build["slug"]) / "meta.json").unlink()
        db.DB_PATH.unlink()
        db.init_db()
        db.rebuild_from_disk()
        self.assertEqual(db.get_build(build["id"])["name"], "nameless")  # falls back to slug

    def test_a_rebuild_writes_a_missing_sidecar_back(self):
        """Libraries that predate this get one, so their metadata travels from now on."""
        build = self.add("Older Build")
        sidecar = db.build_dir(build["id"], build["slug"]) / "meta.json"
        sidecar.unlink()
        db.rebuild_from_disk()
        self.assertTrue(sidecar.exists())
        self.assertEqual(db.read_meta(sidecar.parent)["name"], "Older Build")

    def test_a_corrupt_sidecar_does_not_break_the_rebuild(self):
        build = self.add("Resilient")
        (db.build_dir(build["id"], build["slug"]) / "meta.json").write_text("{ not json",
                                                                           encoding="utf-8")
        db.DB_PATH.unlink()
        db.init_db()
        db.rebuild_from_disk()
        self.assertIsNotNone(db.get_build(build["id"]))


class MergeTests(MachineTestCase):
    """What git actually produces: one folder tree holding both machines' builds."""

    def build_both_machines(self):
        self.become(0)
        self.use(self.root / "pc")
        pc = self.add("PC House")
        db.update_build(pc["id"], source_url="https://example.com/pc")

        self.become(1)
        self.use(self.root / "mac")
        mac = self.add("Mac Tower", 5, 4, 5)
        db.update_build(mac["id"], source_note="added on the laptop")
        return pc, mac

    def merge_into(self, target: Path):
        """Copy both machines' build folders together, as a pull would."""
        merged = target / "builds"
        merged.mkdir(parents=True, exist_ok=True)
        for machine in ("pc", "mac"):
            source = self.root / machine / "builds"
            if not source.exists():
                continue
            for folder in source.iterdir():
                if folder.is_dir() and not (merged / folder.name).exists():
                    shutil.copytree(folder, merged / folder.name)

    def test_both_machines_builds_survive_a_merge(self):
        pc, mac = self.build_both_machines()
        self.assertNotEqual(pc["id"], mac["id"], "ids must differ before we even start")

        self.merge_into(self.root / "merged")
        self.become(0)
        self.use(self.root / "merged")
        (self.root / "merged" / "collector.db").unlink()   # fresh clone: no database
        db.init_db()
        summary = db.rebuild_from_disk()

        self.assertEqual(summary["added"], 2, "both builds should arrive")
        names = sorted(b["name"] for b in db.list_builds())
        self.assertEqual(names, ["Mac Tower", "PC House"])

    def test_a_merge_keeps_each_machines_typed_metadata(self):
        self.build_both_machines()
        self.merge_into(self.root / "merged")
        self.use(self.root / "merged")
        (self.root / "merged" / "collector.db").unlink()
        db.init_db()
        db.rebuild_from_disk()

        by_name = {b["name"]: b for b in db.list_builds()}
        self.assertEqual(by_name["PC House"]["source_url"], "https://example.com/pc")
        self.assertEqual(by_name["Mac Tower"]["source_note"], "added on the laptop")

    def test_the_same_schematic_on_both_machines_becomes_two_variants(self):
        """Identical content, added independently -- the UNIQUE constraint must not trip."""
        self.become(0)
        self.use(self.root / "pc")
        self.add("Shared", 3, 3, 3)
        self.become(1)
        self.use(self.root / "mac")
        self.add("Shared", 3, 3, 3)

        self.merge_into(self.root / "merged")
        self.use(self.root / "merged")
        (self.root / "merged" / "collector.db").unlink()
        db.init_db()
        summary = db.rebuild_from_disk()

        self.assertEqual(summary["added"], 2, "summary: {}".format(summary))
        variants = sorted(b["variant_no"] for b in db.list_builds())
        self.assertEqual(variants, [0, 1], "the second copy should become variant 1")


class StartupSyncTests(MachineTestCase):
    def test_nothing_happens_when_the_database_already_agrees(self):
        self.add("Settled")
        self.assertIsNone(db.sync_from_disk_if_needed())

    def test_a_fresh_clone_is_rebuilt(self):
        build = self.add("Cloned")
        db.DB_PATH.unlink()
        db.init_db()
        self.assertEqual(db.list_builds(), [])

        summary = db.sync_from_disk_if_needed()
        self.assertIsNotNone(summary)
        self.assertEqual(summary["added"], 1)
        self.assertEqual(db.get_build(build["id"])["name"], "Cloned")

    def test_a_folder_arriving_from_a_pull_is_picked_up(self):
        self.add("Existing")
        self.become(1)
        incoming = db.BUILDS_DIR / "1000001__pulled"
        (incoming / "schem").mkdir(parents=True)
        fx.sponge_v2(incoming / "schem" / "original.schem", 4, 4, 4)

        summary = db.sync_from_disk_if_needed()
        self.assertEqual(summary["added"], 1)
        self.assertEqual(len(db.list_builds()), 2)

    def test_a_build_deleted_elsewhere_is_pruned(self):
        keep = self.add("Keep")
        gone = self.add("Gone", 4)
        shutil.rmtree(db.build_dir(gone["id"], gone["slug"]))

        summary = db.sync_from_disk_if_needed()
        self.assertEqual(summary["pruned"], 1)
        self.assertEqual([b["id"] for b in db.list_builds()], [keep["id"]])

    def test_pruning_is_off_by_default(self):
        """A missing folder normally means something is wrong, not that it was deleted."""
        build = self.add("Kept")
        shutil.rmtree(db.build_dir(build["id"], build["slug"]))
        summary = db.rebuild_from_disk()
        self.assertEqual(summary["pruned"], 0)
        self.assertIsNotNone(db.get_build(build["id"]))

    def test_an_empty_builds_folder_never_wipes_the_database(self):
        """The dangerous case: a drive not mounted must not read as 'delete everything'."""
        build = self.add("Precious")
        shutil.rmtree(db.BUILDS_DIR)
        summary = db.rebuild_from_disk(prune_missing=True)
        self.assertEqual(summary["pruned"], 0, "nothing scanned -> nothing pruned")
        self.assertIsNotNone(db.get_build(build["id"]))


if __name__ == "__main__":
    unittest.main()
