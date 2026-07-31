"""Tests for the schema migrations, run against real old databases rather than mocks.

v1 tracked a (schem, glb) pair. v2 added the image TRELLIS actually reconstructs from,
which meant changing a CHECK constraint -- and SQLite can only do that by rebuilding the
table, so v1 -> v2 genuinely rewrites an existing database. v3 then split that single
image into the raw reference screenshots and the photo-real image generated from them,
which renames a file on disk.

These are the only places in the project where existing data is rewritten, so they are
tested end to end: real old schemas in, real files on disk, ids and constraints checked.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db  # noqa: E402

V1_SCHEMA = """
CREATE TABLE builds (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, slug TEXT NOT NULL,
  source_note TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'needs_trellis'
    CHECK (status IN ('needs_trellis','paired')),
  schem_hash TEXT NOT NULL, variant_no INTEGER NOT NULL DEFAULT 0,
  w INTEGER NOT NULL, h INTEGER NOT NULL, l INTEGER NOT NULL,
  schem_path TEXT NOT NULL, glb_path TEXT,
  added_at TEXT NOT NULL, paired_at TEXT,
  UNIQUE (schem_hash, variant_no)
);
CREATE INDEX idx_builds_hash ON builds(schem_hash);
CREATE TABLE schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version VALUES (1);
"""


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-migrate-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        db.PROJECT_ROOT = db.DATA_DIR = self.tmp
        db.DB_PATH = self.tmp / "collector.db"
        db.BUILDS_DIR = self.tmp / "builds"

    def make_v1(self, rows=()):
        conn = sqlite3.connect(str(db.DB_PATH))
        conn.executescript(V1_SCHEMA)
        for row in rows:
            conn.execute(
                "INSERT INTO builds (id, name, slug, source_note, status, schem_hash,"
                " variant_no, w, h, l, schem_path, glb_path, added_at, paired_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
        conn.commit()
        conn.close()

    def read_all(self):
        conn = sqlite3.connect(str(db.DB_PATH))
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM builds ORDER BY id")]
        finally:
            conn.close()

    def version(self):
        conn = sqlite3.connect(str(db.DB_PATH))
        try:
            return conn.execute("SELECT version FROM schema_version").fetchone()[0]
        finally:
            conn.close()

    # ── The migration itself ──────────────────────────────────────────────────

    def test_a_v1_database_is_migrated_in_place(self):
        self.make_v1([(1, "House", "house", "note", "needs_trellis", "h1", 0,
                       4, 4, 4, "s.schem", None, "2026-01-01T00:00:00+00:00", None)])
        db.init_db()
        self.assertEqual(self.version(), db.SCHEMA_VERSION)
        rows = self.read_all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "House")
        self.assertEqual(rows[0]["source_note"], "note")
        self.assertIn("source_url", rows[0])
        self.assertIn("image_path", rows[0])

    def test_a_previously_paired_build_keeps_its_mesh(self):
        """It becomes needs_image -- truthfully, since it has no reference image -- but
        nothing about the mesh it already had is lost."""
        self.make_v1([(1, "Paired", "paired", "", "paired", "h1", 0, 4, 4, 4,
                       "s.schem", "g.glb", "2026-01-01T00:00:00+00:00",
                       "2026-01-02T00:00:00+00:00")])
        db.init_db()
        row = self.read_all()[0]
        self.assertEqual(row["status"], "needs_image")
        self.assertEqual(row["glb_path"], "g.glb")
        self.assertEqual(row["paired_at"], "2026-01-02T00:00:00+00:00")

    def test_ids_and_the_autoincrement_counter_survive(self):
        """Folders on disk are named by id, so an id shift would orphan every one."""
        self.make_v1([
            (3, "A", "a", "", "needs_trellis", "h1", 0, 1, 1, 1, "a.schem", None, "t", None),
            (9, "B", "b", "", "paired", "h2", 0, 1, 1, 1, "b.schem", "b.glb", "t", "t"),
        ])
        db.init_db()
        self.assertEqual([r["id"] for r in self.read_all()], [3, 9])

        conn = sqlite3.connect(str(db.DB_PATH))
        try:
            seq = conn.execute(
                "SELECT seq FROM sqlite_sequence WHERE name='builds'").fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(seq, "AUTOINCREMENT state must carry over")
        self.assertGreaterEqual(seq[0], 9)

    def test_the_duplicate_constraint_survives(self):
        self.make_v1([(1, "A", "a", "", "needs_trellis", "same", 0, 1, 1, 1,
                       "a.schem", None, "t", None)])
        db.init_db()
        conn = sqlite3.connect(str(db.DB_PATH))
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO builds (name, slug, status, schem_hash, variant_no,"
                    " w, h, l, schem_path, added_at)"
                    " VALUES ('B','b','needs_image','same',0,1,1,1,'b.schem','t')")
                conn.commit()
        finally:
            conn.close()

    def test_the_new_status_values_are_enforced(self):
        self.make_v1()
        db.init_db()
        conn = sqlite3.connect(str(db.DB_PATH))
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO builds (name, slug, status, schem_hash, variant_no,"
                    " w, h, l, schem_path, added_at)"
                    " VALUES ('X','x','paired','h',0,1,1,1,'x.schem','t')")
                conn.commit()
        finally:
            conn.close()

    def test_migrating_twice_changes_nothing(self):
        self.make_v1([(1, "A", "a", "", "paired", "h1", 0, 1, 1, 1,
                       "a.schem", "a.glb", "t", "t")])
        db.init_db()
        first = self.read_all()
        db.init_db()
        db.init_db()
        self.assertEqual(self.read_all(), first)
        self.assertEqual(self.version(), db.SCHEMA_VERSION)

    def test_a_fresh_database_starts_at_the_current_version(self):
        db.init_db()
        self.assertEqual(self.version(), db.SCHEMA_VERSION)
        self.assertEqual(self.read_all(), [])


V2_SCHEMA = """
CREATE TABLE builds (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, slug TEXT NOT NULL,
  source_note TEXT NOT NULL DEFAULT '', source_url TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'needs_image'
    CHECK (status IN ('needs_image','needs_trellis','complete')),
  schem_hash TEXT NOT NULL, variant_no INTEGER NOT NULL DEFAULT 0,
  w INTEGER NOT NULL, h INTEGER NOT NULL, l INTEGER NOT NULL,
  schem_path TEXT NOT NULL, image_path TEXT, glb_path TEXT,
  added_at TEXT NOT NULL, image_at TEXT, paired_at TEXT,
  UNIQUE (schem_hash, variant_no)
);
CREATE TABLE schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version VALUES (2);
"""


class V2ToV3MigrationTests(unittest.TestCase):
    """v3 splits v2's single image into screenshots + the generated image.

    v2's `reference.png` was described as "the picture TRELLIS reconstructs from", which
    is exactly the generated one -- so it is renamed rather than discarded.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-migrate3-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        db.PROJECT_ROOT = db.DATA_DIR = self.tmp
        db.DB_PATH = self.tmp / "collector.db"
        db.BUILDS_DIR = self.tmp / "builds"

    def make_v2(self, image_path=None, status="needs_image"):
        conn = sqlite3.connect(str(db.DB_PATH))
        conn.executescript(V2_SCHEMA)
        conn.execute(
            "INSERT INTO builds (id, name, slug, source_note, source_url, status,"
            " schem_hash, variant_no, w, h, l, schem_path, image_path, glb_path,"
            " added_at, image_at, paired_at)"
            " VALUES (1,'House','house','','', ?, 'h1',0,4,4,4,'s.schem', ?, NULL,"
            " '2026-01-01T00:00:00+00:00', ?, NULL)",
            (status, image_path, "2026-01-02T00:00:00+00:00" if image_path else None))
        conn.commit()
        conn.close()

    def image_dir(self):
        folder = db.build_dir(1, "house") / "image"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def row(self):
        conn = sqlite3.connect(str(db.DB_PATH))
        conn.row_factory = sqlite3.Row
        try:
            return dict(conn.execute("SELECT * FROM builds WHERE id = 1").fetchone())
        finally:
            conn.close()

    def test_the_v2_image_is_renamed_to_say_it_is_the_generated_one(self):
        self.make_v2("builds/1__house/image/reference.png", "needs_trellis")
        old = self.image_dir() / "reference.png"
        old.write_bytes(b"the generated image")

        db.init_db()

        new = self.image_dir() / "generated.png"
        self.assertFalse(old.exists(), "the old name must not linger")
        self.assertTrue(new.exists())
        self.assertEqual(new.read_bytes(), b"the generated image", "content must survive")
        self.assertEqual(self.row()["image_path"], "builds/1__house/image/generated.png")
        self.assertEqual(self.row()["status"], "needs_trellis", "status is unchanged by v3")

    def test_a_build_with_no_image_migrates_untouched(self):
        self.make_v2(None)
        db.init_db()
        self.assertIsNone(self.row()["image_path"])
        self.assertEqual(self.row()["status"], "needs_image")

    def test_a_missing_file_leaves_the_row_pointing_at_reality(self):
        """The row follows what's actually on disk, so the two can never disagree."""
        self.make_v2("builds/1__house/image/reference.png", "needs_trellis")
        db.init_db()   # no file was ever written
        self.assertEqual(self.row()["image_path"], "builds/1__house/image/reference.png")

    def test_the_shots_table_is_created(self):
        self.make_v2(None)
        db.init_db()
        conn = sqlite3.connect(str(db.DB_PATH))
        try:
            self.assertIsNotNone(conn.execute(
                "SELECT name FROM sqlite_master WHERE name='shots'").fetchone())
        finally:
            conn.close()

    def test_migrating_twice_does_not_rename_again(self):
        self.make_v2("builds/1__house/image/reference.png", "needs_trellis")
        (self.image_dir() / "reference.png").write_bytes(b"x")
        db.init_db()
        first = self.row()
        db.init_db()
        self.assertEqual(self.row(), first)
        self.assertTrue((self.image_dir() / "generated.png").exists())

    def test_a_v1_database_migrates_all_the_way_to_v3(self):
        """The steps chain: an untouched v1 database ends up fully current."""
        conn = sqlite3.connect(str(db.DB_PATH))
        conn.executescript(V1_SCHEMA)
        conn.execute(
            "INSERT INTO builds (id, name, slug, source_note, status, schem_hash,"
            " variant_no, w, h, l, schem_path, glb_path, added_at, paired_at)"
            " VALUES (1,'Old','old','','paired','h1',0,2,2,2,'s.schem','g.glb','t','t')")
        conn.commit()
        conn.close()

        db.init_db()

        row = self.row()
        self.assertEqual(row["status"], "needs_image")
        self.assertEqual(row["glb_path"], "g.glb")
        self.assertIn("source_url", row)
        conn = sqlite3.connect(str(db.DB_PATH))
        try:
            self.assertEqual(
                conn.execute("SELECT version FROM schema_version").fetchone()[0],
                db.SCHEMA_VERSION)
            self.assertIsNotNone(conn.execute(
                "SELECT name FROM sqlite_master WHERE name='shots'").fetchone())
        finally:
            conn.close()


class StatusDerivationTests(unittest.TestCase):
    """Status is a function of which files exist -- never set independently."""

    def test_it_names_the_first_missing_artefact(self):
        self.assertEqual(db.derive_status(False, False), db.STATUS_NEEDS_IMAGE)
        self.assertEqual(db.derive_status(True, False), db.STATUS_NEEDS_TRELLIS)
        self.assertEqual(db.derive_status(True, True), db.STATUS_COMPLETE)

    def test_a_mesh_without_an_image_is_still_incomplete(self):
        self.assertEqual(db.derive_status(False, True), db.STATUS_NEEDS_IMAGE)


if __name__ == "__main__":
    unittest.main()
