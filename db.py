"""SQLite data layer for BuildDataCollector.

One table (`builds`) holds all metadata; the actual .schem/.glb bytes live on disk under
data/builds/<id>__<slug>/ and are referenced by relative path only. A tiny
`schema_version` table is the whole migration story (forward-only, one seam).

Design notes:
- A fresh connection is opened per operation so the Flask dev server's worker threads
  never share a handle.
- Deliberate duplicates are allowed: the UNIQUE constraint is the *composite*
  (schem_hash, variant_no). A normal add is variant 0; "save anyway" bumps variant_no.
- Nothing here ever moves or deletes the user's ORIGINAL files -- uploads are always
  copied in. The only deletion path is delete_build(), which removes only the copy the
  tool created under data/builds/.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from slugify import slugify
from schem_meta import parse_header, sha256_file

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "collector.db"
BUILDS_DIR = DATA_DIR / "builds"

SCHEMA_VERSION = 3

# A build is collected in four steps, because TRELLIS takes a photo-real 2D image rather
# than a schematic:
#
#   schem  ->  reference screenshots  ->  AI-generated image  ->  TRELLIS mesh
#              (optional, up to 3)        (what TRELLIS eats)
#
# The screenshots are working material -- raw captures of the build from whatever angles
# are worth keeping, used later to generate the photo-real image -- so they are optional
# and never gate the status. Status is derived from the artefacts that do gate it (see
# derive_status), and names the first one missing.
STATUS_NEEDS_IMAGE = "needs_image"
STATUS_NEEDS_TRELLIS = "needs_trellis"
STATUS_COMPLETE = "complete"
ALL_STATUSES = (STATUS_NEEDS_IMAGE, STATUS_NEEDS_TRELLIS, STATUS_COMPLETE)

STATUS_LABELS = {
    STATUS_NEEDS_IMAGE: "Needs AI image",
    STATUS_NEEDS_TRELLIS: "Needs TRELLIS",
    STATUS_COMPLETE: "Complete",
}

# How many reference screenshots one build may keep. Enough for a few angles to choose
# between later; small enough that they stay a fixed set of slots rather than a gallery.
MAX_SHOTS = 3

_DIR_RE = re.compile(r"^(\d+)__(.+)$")

# ── Sharing a library between machines ────────────────────────────────────────
# Ids come from a counter and are baked into folder names, so two machines writing to
# their own copies would both mint id 1 and collide the moment the folders met. Each
# machine therefore allocates from its own block: machine 0 gets 1..999_999, machine 1
# gets 1_000_000.., and so on. Set BDC_MACHINE_ID on every machine after the first.
ID_BLOCK = 1_000_000
MACHINE_ID = int(os.environ.get("BDC_MACHINE_ID", "0") or 0)

# Everything the user typed rather than the files implying. It lives in the database, so
# without a copy on disk it would not survive being cloned onto another machine -- the
# rebuild would name every build after its folder and drop the notes and source URL. This
# file is that copy, one per build, so git merges two machines' edits per-build instead of
# conflicting over one binary database.
META_FILENAME = "meta.json"
META_FIELDS = ("name", "source_note", "source_url", "variant_no",
               "added_at", "image_at", "paired_at")


def derive_status(has_image: bool, has_glb: bool) -> str:
    """The stage a build is at, given what it has. Names the next thing that's missing."""
    if not has_image:
        return STATUS_NEEDS_IMAGE
    if not has_glb:
        return STATUS_NEEDS_TRELLIS
    return STATUS_COMPLETE

# Columns a client may sort the Library by -> the SQL expression used for it.
_SORT_COLUMNS = {
    "name": "name COLLATE NOCASE",
    "added_at": "added_at",
    "paired_at": "paired_at",
    "status": "status",
    "image_at": "image_at",
    "longest": "MAX(w, h, l)",
    "w": "w",
    "h": "h",
    "l": "l",
}


class DuplicateBuildError(Exception):
    """Raised by create_build when the schem_hash already exists and override is off."""

    def __init__(self, existing):
        super().__init__("A build with identical content already exists.")
        self.existing = existing  # list[dict]


class BuildNotFoundError(Exception):
    """Raised when an operation targets a build id that isn't in the DB."""


class GlbAlreadyPairedError(Exception):
    """Raised when pairing would overwrite an existing output.glb without replace=True."""


class NotPairedError(Exception):
    """Raised when unpair_build targets a build that has no TRELLIS mesh to remove."""


class ImageAlreadySetError(Exception):
    """Raised when setting a reference image would overwrite one without replace=True."""


class NoImageError(Exception):
    """Raised when clearing the generated image on a build that has none."""


class TooManyShotsError(Exception):
    """Raised when a build already holds MAX_SHOTS reference screenshots."""


class ShotNotFoundError(Exception):
    """Raised when a screenshot slot is empty or out of range."""


# ── Connection & schema ───────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS builds (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT    NOT NULL,
  slug        TEXT    NOT NULL,
  source_note TEXT    NOT NULL DEFAULT '',
  source_url  TEXT    NOT NULL DEFAULT '',
  status      TEXT    NOT NULL DEFAULT 'needs_image'
                CHECK (status IN ('needs_image','needs_trellis','complete')),
  schem_hash  TEXT    NOT NULL,
  variant_no  INTEGER NOT NULL DEFAULT 0,
  w INTEGER NOT NULL, h INTEGER NOT NULL, l INTEGER NOT NULL,
  schem_path  TEXT    NOT NULL,
  image_path  TEXT,
  glb_path    TEXT,
  added_at    TEXT    NOT NULL,
  image_at    TEXT,
  paired_at   TEXT,
  UNIQUE (schem_hash, variant_no)
);
CREATE INDEX IF NOT EXISTS idx_builds_hash   ON builds(schem_hash);
CREATE INDEX IF NOT EXISTS idx_builds_status ON builds(status);
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

-- Reference screenshots: up to MAX_SHOTS per build, each pinned to a numbered slot so
-- the file on disk (shots/shot-<slot>.png) and the row always name each other. Removing
-- the middle one leaves a gap rather than renumbering, so a slot's picture never
-- silently becomes a different picture.
CREATE TABLE IF NOT EXISTS shots (
  build_id INTEGER NOT NULL REFERENCES builds(id) ON DELETE CASCADE,
  slot     INTEGER NOT NULL CHECK (slot BETWEEN 1 AND 3),
  path     TEXT    NOT NULL,
  added_at TEXT    NOT NULL,
  PRIMARY KEY (build_id, slot)
);
CREATE INDEX IF NOT EXISTS idx_shots_build ON shots(build_id);
"""


def connect() -> sqlite3.Connection:
    """Open a new connection with Row access and sane pragmas."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create folders + tables on first run and record the schema version."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BUILDS_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect()
    try:
        conn.executescript(_SCHEMA_SQL)
        row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        else:
            _migrate(conn, int(row["version"]))
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection, from_version: int) -> None:
    """Forward-only migration. Steps run in order; each is safe to re-run."""
    if from_version >= SCHEMA_VERSION:
        return
    if from_version < 2:
        _migrate_v1_to_v2(conn)
    if from_version < 3:
        _migrate_v2_to_v3(conn)
    conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """v1 (schem, glb) -> v2 (schem, image, glb) with a three-stage status.

    The status CHECK constraint has to change and SQLite cannot alter one in place, so
    this is the standard rebuild: create the new table, copy the rows through, swap the
    names. Everything happens in one transaction -- a crash mid-migration rolls back to
    the v1 table rather than leaving a half-built one.

    Old rows are remapped by the same rule the rest of the app uses: status names the
    first missing artefact. No v1 build has a reference image, so a previously-'paired'
    build becomes 'needs_image' while keeping its glb -- which is the truth under the new
    model, not a downgrade. Adding the image it was made from completes it.
    """
    conn.executescript(
        """
        CREATE TABLE builds_v2 (
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          name        TEXT    NOT NULL,
          slug        TEXT    NOT NULL,
          source_note TEXT    NOT NULL DEFAULT '',
          source_url  TEXT    NOT NULL DEFAULT '',
          status      TEXT    NOT NULL DEFAULT 'needs_image'
                        CHECK (status IN ('needs_image','needs_trellis','complete')),
          schem_hash  TEXT    NOT NULL,
          variant_no  INTEGER NOT NULL DEFAULT 0,
          w INTEGER NOT NULL, h INTEGER NOT NULL, l INTEGER NOT NULL,
          schem_path  TEXT    NOT NULL,
          image_path  TEXT,
          glb_path    TEXT,
          added_at    TEXT    NOT NULL,
          image_at    TEXT,
          paired_at   TEXT,
          UNIQUE (schem_hash, variant_no)
        );
        INSERT INTO builds_v2
          (id, name, slug, source_note, source_url, status, schem_hash, variant_no,
           w, h, l, schem_path, image_path, glb_path, added_at, image_at, paired_at)
        SELECT id, name, slug, source_note, '', 'needs_image', schem_hash, variant_no,
               w, h, l, schem_path, NULL, glb_path, added_at, NULL, paired_at
        FROM builds;
        DROP TABLE builds;
        ALTER TABLE builds_v2 RENAME TO builds;
        CREATE INDEX IF NOT EXISTS idx_builds_hash   ON builds(schem_hash);
        CREATE INDEX IF NOT EXISTS idx_builds_status ON builds(status);
        """
    )


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """v2 (one image) -> v3 (reference screenshots + the AI-generated image).

    v2 had a single `reference.png`: "the picture TRELLIS reconstructs from". v3 splits
    that idea in two, because they are collected differently and mean different things --
    raw screenshots of the build are working material you capture in a moment, while the
    photo-real image fed to TRELLIS is generated later from them. So the existing file is
    the *generated* one, and it is renamed on disk to say so.

    The rename is best-effort per build and the row follows whatever actually happened on
    disk. A file that cannot be moved keeps its old name and its old path in the database,
    so the two never disagree; rebuild_from_disk would report it. The shots table itself is
    created by the main schema script, which runs first.
    """
    rows = conn.execute(
        "SELECT id, slug, image_path FROM builds WHERE image_path IS NOT NULL").fetchall()
    for row in rows:
        folder = build_dir(row["id"], row["slug"])
        old, new = folder / "image" / "reference.png", folder / "image" / "generated.png"
        if not old.exists() or new.exists():
            continue
        try:
            old.rename(new)
        except OSError:
            continue                       # leave the row pointing at the file that exists
        conn.execute("UPDATE builds SET image_path = ? WHERE id = ?",
                     (_rel(new), row["id"]))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rel(path: Path) -> str:
    """Project-root-relative POSIX path for storage in the DB."""
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def build_dir(build_id: int, slug: str) -> Path:
    """Absolute path of a build's folder: data/builds/<id>__<slug>/."""
    return BUILDS_DIR / "{}__{}".format(build_id, slug)


def next_build_id(conn: sqlite3.Connection) -> int:
    """Allocate an id inside this machine's block, so two machines can never collide.

    Blocks are a million wide, which is more builds than this will ever hold, and the
    first block starts at 1 so a single-machine library numbers exactly as it always did.
    """
    base = MACHINE_ID * ID_BLOCK
    row = conn.execute(
        "SELECT MAX(id) FROM builds WHERE id > ? AND id <= ?",
        (base, base + ID_BLOCK)).fetchone()
    highest = row[0]
    return base + 1 if highest is None else int(highest) + 1


# ── Per-build metadata sidecar ────────────────────────────────────────────────

def meta_path(build_id: int, slug: str) -> Path:
    return build_dir(build_id, slug) / META_FILENAME


def write_meta(build: dict) -> None:
    """Mirror a build's user-entered fields into its folder as meta.json.

    Best-effort: the database is still the thing the app reads, and a library that never
    leaves this machine works fine without these files. They exist so a *copy* of the
    folder -- cloned, synced, restored from a backup -- carries the typing with it.
    """
    try:
        folder = build_dir(build["id"], build["slug"])
        if not folder.exists():
            return
        payload = {key: build.get(key) for key in META_FIELDS}
        target = folder / META_FILENAME
        # Written whole, then moved into place, so a reader never sees half a file.
        temp = target.with_name(target.name + ".tmp")
        with open(temp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(temp, target)
    except (OSError, TypeError, ValueError):
        pass


def read_meta(folder: Path) -> dict:
    """Read a build folder's meta.json, or {} if it has none / is unreadable."""
    try:
        with open(folder / META_FILENAME, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {key: loaded[key] for key in META_FIELDS if key in loaded}


def _sync_meta(conn: sqlite3.Connection, build_id: int) -> None:
    """Refresh the sidecar from the row. Call after anything the user typed changes."""
    row = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
    if row is not None:
        write_meta(row_to_dict(row))


def row_to_dict(row: sqlite3.Row, conn: Optional[sqlite3.Connection] = None) -> dict:
    """Serialize a DB row to a JSON-friendly dict with derived display fields.

    Pass `conn` to include the build's reference screenshots. Without one the shot fields
    come back empty rather than triggering a query per row -- callers that need them
    (list_builds, get_build) already hold a connection.
    """
    w, h, l = row["w"], row["h"], row["l"]
    status = row["status"]
    shots = list_shots(conn, row["id"]) if conn is not None else []
    return {
        "shots": shots,
        "shot_count": len(shots),
        "shot_slots_free": MAX_SHOTS - len(shots),
        "max_shots": MAX_SHOTS,
        "id": row["id"],
        "name": row["name"],
        "slug": row["slug"],
        "source_note": row["source_note"],
        "source_url": row["source_url"],
        "status": status,
        "status_label": STATUS_LABELS.get(status, status),
        "schem_hash": row["schem_hash"],
        "variant_no": row["variant_no"],
        "w": w, "h": h, "l": l,
        "longest": max(w, h, l),
        "dims": "{}x{}x{}".format(w, h, l),
        "schem_path": row["schem_path"],
        "image_path": row["image_path"],
        "glb_path": row["glb_path"],
        "has_image": bool(row["image_path"]),
        "has_glb": bool(row["glb_path"]),
        "added_at": row["added_at"],
        "image_at": row["image_at"],
        "paired_at": row["paired_at"],
    }


def find_by_hash(conn: sqlite3.Connection, schem_hash: str) -> list[dict]:
    """All existing builds sharing this content hash (used for the dedup warning)."""
    rows = conn.execute(
        "SELECT * FROM builds WHERE schem_hash = ? ORDER BY variant_no", (schem_hash,)
    ).fetchall()
    return [row_to_dict(r, conn) for r in rows]


# ── Create ────────────────────────────────────────────────────────────────────

def create_build(name: str, source_note: str, temp_schem_path,
                 schem_hash: str, dims: tuple[int, int, int],
                 allow_duplicate: bool = False, source_url: str = "") -> dict:
    """Insert a build row and COPY the schem into data/builds/<id>__<slug>/schem/.

    Raises DuplicateBuildError (with the existing rows) when the hash already exists and
    allow_duplicate is False. On any file error the inserted row is rolled back so the DB
    and disk never disagree.
    """
    name = (name or "").strip() or "Untitled build"
    source_note = (source_note or "").strip()
    source_url = (source_url or "").strip()
    w, h, l = dims
    slug = slugify(name)

    conn = connect()
    created_dir: Optional[Path] = None
    try:
        existing = find_by_hash(conn, schem_hash)
        if existing and not allow_duplicate:
            raise DuplicateBuildError(existing)
        variant_no = (existing[-1]["variant_no"] + 1) if existing else 0

        build_id = next_build_id(conn)
        conn.execute(
            """INSERT INTO builds
               (id, name, slug, source_note, source_url, status, schem_hash, variant_no,
                w, h, l, schem_path, image_path, glb_path, added_at, image_at, paired_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', NULL, NULL, ?, NULL, NULL)""",
            (build_id, name, slug, source_note, source_url, STATUS_NEEDS_IMAGE, schem_hash,
             variant_no, w, h, l, _now_iso()),
        )

        folder = build_dir(build_id, slug)
        schem_folder = folder / "schem"
        schem_folder.mkdir(parents=True, exist_ok=True)
        created_dir = folder
        dest = schem_folder / "original.schem"
        if dest.exists():
            # Never silently overwrite an existing copy.
            raise FileExistsError("A schem already exists at {}".format(dest))
        shutil.copyfile(temp_schem_path, dest)  # COPY, never move

        conn.execute("UPDATE builds SET schem_path = ? WHERE id = ?",
                     (_rel(dest), build_id))
        conn.commit()
        row = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
        write_meta(row_to_dict(row, conn))
        return row_to_dict(row, conn)
    except Exception:
        conn.rollback()
        # Clean up a just-created empty folder so a failed save leaves no orphan.
        if created_dir is not None:
            try:
                if not any((created_dir / "schem").glob("*")):
                    shutil.rmtree(created_dir, ignore_errors=True)
            except Exception:
                pass
        raise
    finally:
        conn.close()


# ── Read / list ───────────────────────────────────────────────────────────────

def list_builds(status: Optional[str] = None, q: Optional[str] = None,
                max_longest: Optional[int] = None, min_longest: Optional[int] = None,
                sort: str = "added_at", direction: str = "desc") -> list[dict]:
    """Return builds matching the given filters, sorted by a whitelisted column."""
    where, params = [], []
    if status in ALL_STATUSES:
        where.append("status = ?")
        params.append(status)
    if q:
        where.append("(name LIKE ? OR source_note LIKE ? OR source_url LIKE ?)")
        like = "%{}%".format(q)
        params.extend([like, like, like])
    if max_longest is not None:
        where.append("MAX(w, h, l) <= ?")
        params.append(max_longest)
    if min_longest is not None:
        where.append("MAX(w, h, l) >= ?")
        params.append(min_longest)

    order_col = _SORT_COLUMNS.get(sort, "added_at")
    order_dir = "ASC" if str(direction).lower() == "asc" else "DESC"
    sql = "SELECT * FROM builds"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY {} {}, id DESC".format(order_col, order_dir)

    conn = connect()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [row_to_dict(r, conn) for r in rows]
    finally:
        conn.close()


def get_build(build_id: int) -> Optional[dict]:
    """Return a single build dict, or None if the id doesn't exist."""
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
        return row_to_dict(row, conn) if row else None
    finally:
        conn.close()


# ── Update editable metadata ──────────────────────────────────────────────────

# Only these are the user's to change. Everything else about a build is derived from its
# files, so letting it be edited would just create a way for the row to start lying.
EDITABLE_FIELDS = ("name", "source_note", "source_url")


def update_build(build_id: int, **fields) -> dict:
    """Update a build's user-entered metadata. Unknown fields are ignored.

    The slug follows the name, because it names the folder on disk. Renaming does NOT
    move the folder -- the folder is keyed by id and the slug is only a human-readable
    tail, so an old folder keeps working and rebuild_from_disk re-derives the rest.
    """
    updates = {k: (fields[k] or "").strip() for k in EDITABLE_FIELDS if k in fields}
    if not updates:
        return get_build(build_id) or _raise_missing(build_id)
    if "name" in updates and not updates["name"]:
        updates["name"] = "Untitled build"

    conn = connect()
    try:
        row = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
        if row is None:
            raise BuildNotFoundError(build_id)
        assignments = ", ".join("{} = ?".format(k) for k in updates)
        params = list(updates.values())
        conn.execute("UPDATE builds SET {} WHERE id = ?".format(assignments),
                     params + [build_id])
        conn.commit()
        _sync_meta(conn, build_id)
        return row_to_dict(conn.execute("SELECT * FROM builds WHERE id = ?",
                                        (build_id,)).fetchone())
    finally:
        conn.close()


def _raise_missing(build_id: int):
    raise BuildNotFoundError(build_id)


# ── Reference screenshots ─────────────────────────────────────────────────────

def shot_path(build_id: int, slug: str, slot: int) -> Path:
    """Where screenshot `slot` lives. Slot and filename always name each other."""
    return build_dir(build_id, slug) / "shots" / "shot-{}.png".format(slot)


def list_shots(conn: sqlite3.Connection, build_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT slot, path, added_at FROM shots WHERE build_id = ? ORDER BY slot",
        (build_id,)).fetchall()
    return [{"slot": r["slot"], "path": r["path"], "added_at": r["added_at"]} for r in rows]


def add_shot(build_id: int, temp_image_path) -> tuple[dict, int]:
    """COPY a (pre-validated) screenshot into the build's shots/ folder.

    Screenshots are optional and never change the status: they are raw captures kept so a
    photo-real image can be generated from a chosen angle later. Each lands in the lowest
    free slot, so removing one and adding another reuses the gap rather than growing.

    Returns (build, slot). The slot is returned rather than left to be inferred from the
    build's shot list -- filling a gap means the new one is not necessarily the last.
    """
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
        if row is None:
            raise BuildNotFoundError(build_id)

        used = {s["slot"] for s in list_shots(conn, build_id)}
        free = next((n for n in range(1, MAX_SHOTS + 1) if n not in used), None)
        if free is None:
            raise TooManyShotsError(build_id)

        dest = shot_path(build_id, row["slug"], free)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(temp_image_path, dest)  # COPY, never move

        conn.execute(
            "INSERT INTO shots (build_id, slot, path, added_at) VALUES (?, ?, ?, ?)",
            (build_id, free, _rel(dest), _now_iso()))
        conn.commit()
        build = row_to_dict(conn.execute("SELECT * FROM builds WHERE id = ?",
                                         (build_id,)).fetchone(), conn)
        return build, free
    finally:
        conn.close()


def remove_shot(build_id: int, slot: int) -> dict:
    """Delete one screenshot. The file goes first, as with every other removal here."""
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
        if row is None:
            raise BuildNotFoundError(build_id)
        existing = conn.execute(
            "SELECT 1 FROM shots WHERE build_id = ? AND slot = ?", (build_id, slot)).fetchone()
        if existing is None:
            raise ShotNotFoundError((build_id, slot))

        path = shot_path(build_id, row["slug"], slot)
        if BUILDS_DIR in path.resolve().parents:
            path.unlink(missing_ok=True)

        conn.execute("DELETE FROM shots WHERE build_id = ? AND slot = ?", (build_id, slot))
        conn.commit()
        result = row_to_dict(conn.execute("SELECT * FROM builds WHERE id = ?",
                                          (build_id,)).fetchone(), conn)
    finally:
        conn.close()

    folder = build_dir(build_id, row["slug"])
    thumb = folder / "thumb" / "shot-{}.png".format(slot)
    try:
        if BUILDS_DIR in thumb.resolve().parents:
            thumb.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        (folder / "shots").rmdir()
    except OSError:
        pass
    return result


# ── AI-generated image ────────────────────────────────────────────────────────

def set_generated_image(build_id: int, temp_image_path, replace: bool = False) -> dict:
    """COPY a (pre-validated) generated image into the build's image/ folder.

    This is the picture TRELLIS actually consumes -- it reconstructs from a photo-real 2D
    image, not from a schematic -- so it is a first-class artefact of the build, and it is
    what gates the TRELLIS stage. The reference screenshots are the raw material it is
    generated from; this is the result.

    Refuses to overwrite an existing image unless replace=True, mirroring pair_build.
    """
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
        if row is None:
            raise BuildNotFoundError(build_id)

        folder = build_dir(build_id, row["slug"])
        image_folder = folder / "image"
        image_folder.mkdir(parents=True, exist_ok=True)
        dest = image_folder / "generated.png"
        if dest.exists() and not replace:
            raise ImageAlreadySetError(str(dest))
        shutil.copyfile(temp_image_path, dest)  # COPY, never move

        conn.execute(
            "UPDATE builds SET status = ?, image_path = ?, image_at = ? WHERE id = ?",
            (derive_status(True, bool(row["glb_path"])), _rel(dest), _now_iso(), build_id),
        )
        conn.commit()
        _sync_meta(conn, build_id)
        return row_to_dict(conn.execute("SELECT * FROM builds WHERE id = ?",
                                        (build_id,)).fetchone(), conn)
    finally:
        conn.close()


def clear_generated_image(build_id: int) -> dict:
    """Remove a build's generated image and step its status back.

    Same ordering rule as unpair_build: the file goes first, so a failure to delete it
    leaves the row honestly claiming the image is still there.
    """
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
        if row is None:
            raise BuildNotFoundError(build_id)
        if not row["image_path"]:
            raise NoImageError(build_id)

        folder = build_dir(build_id, row["slug"])
        image = folder / "image" / "generated.png"
        if BUILDS_DIR in image.resolve().parents:
            image.unlink(missing_ok=True)

        conn.execute(
            "UPDATE builds SET status = ?, image_path = NULL, image_at = NULL WHERE id = ?",
            (derive_status(False, bool(row["glb_path"])), build_id),
        )
        conn.commit()
        _sync_meta(conn, build_id)
        result = row_to_dict(conn.execute("SELECT * FROM builds WHERE id = ?",
                                          (build_id,)).fetchone(), conn)
    finally:
        conn.close()

    thumb = folder / "thumb" / "image.png"
    try:
        if BUILDS_DIR in thumb.resolve().parents:
            thumb.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        (folder / "image").rmdir()
    except OSError:
        pass
    return result


# ── Pair ──────────────────────────────────────────────────────────────────────

def pair_build(build_id: int, temp_glb_path, replace: bool = False) -> dict:
    """COPY a (pre-validated) glb into the build's trellis/ folder and mark it paired.

    Refuses to overwrite an existing output.glb unless replace=True.
    """
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
        if row is None:
            raise BuildNotFoundError(build_id)

        folder = build_dir(build_id, row["slug"])
        trellis_folder = folder / "trellis"
        trellis_folder.mkdir(parents=True, exist_ok=True)
        dest = trellis_folder / "output.glb"
        if dest.exists() and not replace:
            raise GlbAlreadyPairedError(str(dest))
        shutil.copyfile(temp_glb_path, dest)  # COPY, never move

        conn.execute(
            "UPDATE builds SET status = ?, glb_path = ?, paired_at = ? WHERE id = ?",
            (derive_status(bool(row["image_path"]), True), _rel(dest), _now_iso(), build_id),
        )
        conn.commit()
        _sync_meta(conn, build_id)
        updated = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
        return row_to_dict(updated, conn)
    finally:
        conn.close()


# ── Unpair (explicit, confirmed) ──────────────────────────────────────────────

def unpair_build(build_id: int) -> dict:
    """Remove a build's TRELLIS mesh and put it back to needs_trellis.

    The escape hatch for a `.glb` dropped onto the wrong build: it undoes pair_build
    without touching the schematic, so the build keeps its id, name, notes and history and
    is simply waiting for a mesh again.

    The copied output.glb is really deleted rather than set aside, because
    rebuild_from_disk decides paired-ness purely by whether that file exists -- anything
    left behind would quietly re-pair the build on the next rebuild. As everywhere else
    here, only the tool's own copy under data/builds/ is touched; the .glb the user
    dropped is still wherever they downloaded it.
    """
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
        if row is None:
            raise BuildNotFoundError(build_id)
        if not row["glb_path"]:
            raise NotPairedError(build_id)

        # The mesh goes first, and a failure to remove it propagates. Updating the row
        # first would risk a database that says "not paired" over a folder that still
        # holds an output.glb -- and rebuild_from_disk, reading only the folder, would put
        # the pair straight back. Deleting first means the worst case is an error the user
        # sees, never a disagreement they don't.
        folder = build_dir(build_id, row["slug"])
        glb = folder / "trellis" / "output.glb"
        if BUILDS_DIR in glb.resolve().parents:   # guard: only ever inside data/builds/
            glb.unlink(missing_ok=True)

        conn.execute(
            """UPDATE builds SET status = ?, glb_path = NULL, paired_at = NULL
               WHERE id = ?""",
            (derive_status(bool(row["image_path"]), False), build_id),
        )
        conn.commit()
        _sync_meta(conn, build_id)
        updated = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
        result = row_to_dict(updated, conn)
    finally:
        conn.close()

    # The mesh's thumbnail is only a cache, so losing the race to delete it costs nothing:
    # it is keyed to a source file that no longer exists, so it can never be served again.
    thumb = folder / "thumb" / "mesh.png"
    try:
        if BUILDS_DIR in thumb.resolve().parents:
            thumb.unlink(missing_ok=True)
    except OSError:
        pass
    # Tidy the now-empty trellis/ back out of existence, so an unpaired build's folder
    # looks exactly like one that was never paired. rmdir refuses on a non-empty
    # directory, which is the safety we want: anything else in there stays.
    try:
        (folder / "trellis").rmdir()
    except OSError:
        pass
    return result


# ── Delete (explicit, confirmed) ──────────────────────────────────────────────

def delete_build(build_id: int) -> dict:
    """Delete a build row and remove ONLY its copied folder under data/builds/.

    Never touches the user's original downloaded files. Returns the removed build dict.
    """
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
        if row is None:
            raise BuildNotFoundError(build_id)
        removed = row_to_dict(row, conn)
        conn.execute("DELETE FROM builds WHERE id = ?", (build_id,))
        conn.commit()
    finally:
        conn.close()

    folder = build_dir(build_id, row["slug"])
    # Guard: only ever delete inside data/builds/.
    if BUILDS_DIR in folder.resolve().parents and folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    return removed


# ── Startup sync ──────────────────────────────────────────────────────────────

def sync_from_disk_if_needed() -> Optional[dict]:
    """Rebuild the database when the folders on disk disagree with it.

    The database is a cache of what's in data/builds/, and it is deliberately not shared
    between machines -- it is binary, it cannot be merged, and it is derivable. So on a
    fresh clone there is no database at all, and after a pull there are folders it has
    never seen (or rows whose folder someone else deleted). Comparing the two counts is
    cheap; rebuilding only when they differ keeps startup instant in the normal case.

    Returns the rebuild summary, or None if nothing needed doing.
    """
    BUILDS_DIR.mkdir(parents=True, exist_ok=True)
    on_disk = {int(m.group(1)) for m in
               (_DIR_RE.match(p.name) for p in BUILDS_DIR.iterdir() if p.is_dir()) if m}
    conn = connect()
    try:
        in_db = {row["id"] for row in conn.execute("SELECT id FROM builds").fetchall()}
    finally:
        conn.close()
    if on_disk == in_db:
        return None
    return rebuild_from_disk(prune_missing=True)


# ── Rebuild-from-disk safety net ──────────────────────────────────────────────

def _reconcile_shots(conn: sqlite3.Connection, build_id: int, folder: Path) -> None:
    """Make the shots table agree with shots/shot-<n>.png on disk.

    Slots present on disk but not in the table are inserted; rows whose file has gone are
    deleted. As everywhere in rebuild_from_disk, the filesystem is the authority.
    """
    known = {s["slot"]: s for s in list_shots(conn, build_id)}
    for slot in range(1, MAX_SHOTS + 1):
        path = folder / "shots" / "shot-{}.png".format(slot)
        if path.exists() and slot not in known:
            conn.execute(
                "INSERT INTO shots (build_id, slot, path, added_at) VALUES (?, ?, ?, ?)",
                (build_id, slot, _rel(path),
                 datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                 .isoformat(timespec="seconds")))
        elif not path.exists() and slot in known:
            conn.execute("DELETE FROM shots WHERE build_id = ? AND slot = ?",
                         (build_id, slot))

def rebuild_from_disk(prune_missing: bool = False) -> dict:
    """Reconcile the DB with what's actually on disk under data/builds/.

    For each <id>__<slug>/ folder: re-derive dims + hash from schem/original.schem and
    derive status from which artefacts are present. What the *user* typed comes from the
    folder's meta.json when it has one -- that file is how a name, note or source URL
    survives being cloned onto another machine -- falling back to the existing row, and
    finally to the folder's slug.

    `prune_missing` deletes rows whose folder is gone. Off by default, because the folder
    being absent normally means something is wrong rather than that the build was deleted.
    The startup sync turns it on: there, a vanished folder is exactly what a pull that
    removed a build looks like.
    """
    BUILDS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"added": 0, "updated": 0, "pruned": 0, "skipped": [], "scanned": 0}
    seen_ids: set[int] = set()
    conn = connect()
    try:
        for folder in sorted(p for p in BUILDS_DIR.iterdir() if p.is_dir()):
            m = _DIR_RE.match(folder.name)
            if not m:
                summary["skipped"].append({"folder": folder.name, "reason": "name not <id>__<slug>"})
                continue
            summary["scanned"] += 1
            build_id, slug = int(m.group(1)), m.group(2)
            schem = folder / "schem" / "original.schem"
            if not schem.exists():
                summary["skipped"].append({"folder": folder.name, "reason": "no schem/original.schem"})
                continue
            try:
                w, h, l = parse_header(schem)
                schem_hash = sha256_file(schem)
            except Exception as exc:
                summary["skipped"].append({"folder": folder.name, "reason": str(exc)})
                continue

            glb = folder / "trellis" / "output.glb"
            image = folder / "image" / "generated.png"
            paired, has_image = glb.exists(), image.exists()
            status = derive_status(has_image, paired)
            glb_path = _rel(glb) if paired else None
            image_path = _rel(image) if has_image else None
            schem_path = _rel(schem)
            meta = read_meta(folder)
            seen_ids.add(build_id)

            existing = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
            if existing is not None:
                # meta.json wins where it has an opinion: it is the copy that travelled
                # with the files, so it carries edits made on the other machine.
                name = meta.get("name") or existing["name"]
                source_note = meta.get("source_note")
                source_note = existing["source_note"] if source_note is None else source_note
                source_url = meta.get("source_url")
                source_url = existing["source_url"] if source_url is None else source_url
                conn.execute(
                    "UPDATE builds SET name=?, source_note=?, source_url=? WHERE id=?",
                    (name, source_note, source_url, build_id))

                paired_at = existing["paired_at"]
                if paired and not paired_at:
                    paired_at = _now_iso()
                if not paired:
                    paired_at = None
                image_at = existing["image_at"]
                if has_image and not image_at:
                    image_at = _now_iso()
                if not has_image:
                    image_at = None
                conn.execute(
                    """UPDATE builds SET slug=?, schem_hash=?, w=?, h=?, l=?,
                       schem_path=?, image_path=?, glb_path=?, status=?,
                       image_at=?, paired_at=? WHERE id=?""",
                    (slug, schem_hash, w, h, l, schem_path, image_path, glb_path, status,
                     image_at, paired_at, build_id),
                )
                summary["updated"] += 1
            else:
                same_hash = conn.execute(
                    "SELECT COALESCE(MAX(variant_no), -1) AS mx FROM builds WHERE schem_hash = ?",
                    (schem_hash,),
                ).fetchone()
                # A build cloned from another machine arrives with its meta.json, so it
                # keeps the name and notes that were typed there. Without one there is
                # genuinely nothing to go on but the folder name.
                #
                # The variant number is only a *preference*: add the same schematic on two
                # machines and both sidecars claim variant 0, which the UNIQUE constraint
                # would refuse. Take the recorded one when it's free, otherwise the next
                # one that is -- they were, after all, added independently.
                variant_no = meta.get("variant_no")
                if variant_no is None:
                    variant_no = same_hash["mx"] + 1
                else:
                    taken = conn.execute(
                        "SELECT 1 FROM builds WHERE schem_hash = ? AND variant_no = ?",
                        (schem_hash, int(variant_no))).fetchone()
                    if taken:
                        variant_no = same_hash["mx"] + 1
                added_at = meta.get("added_at") or datetime.fromtimestamp(
                    folder.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
                paired_at = meta.get("paired_at") or (_now_iso() if paired else None)
                image_at = meta.get("image_at") or (_now_iso() if has_image else None)
                if not paired:
                    paired_at = None
                if not has_image:
                    image_at = None
                try:
                    conn.execute(
                        """INSERT INTO builds
                           (id, name, slug, source_note, source_url, status, schem_hash,
                            variant_no, w, h, l, schem_path, image_path, glb_path,
                            added_at, image_at, paired_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (build_id, meta.get("name") or slug, slug,
                         meta.get("source_note") or "", meta.get("source_url") or "",
                         status, schem_hash, int(variant_no),
                         w, h, l, schem_path, image_path, glb_path, added_at,
                         image_at, paired_at),
                    )
                except sqlite3.IntegrityError as exc:
                    # One unusable folder must not cost the rest of the library. Report it
                    # and carry on -- the alternative is a rebuild that aborts partway and
                    # leaves the database describing some of the disk but not all of it.
                    summary["skipped"].append({"folder": folder.name, "reason": str(exc)})
                    seen_ids.discard(build_id)
                    continue
                summary["added"] += 1

            _reconcile_shots(conn, build_id, folder)
            if not meta:
                # Older build, or one whose sidecar was lost: write it now so this build's
                # metadata can travel from here on.
                row = conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()
                write_meta(row_to_dict(row))

        if prune_missing:
            # A row whose folder is gone. Guarded on having scanned something: an empty
            # or unreadable builds/ directory must not be read as "everything was deleted".
            if summary["scanned"] > 0:
                for row in conn.execute("SELECT id FROM builds").fetchall():
                    if row["id"] not in seen_ids:
                        conn.execute("DELETE FROM builds WHERE id = ?", (row["id"],))
                        summary["pruned"] += 1

        # Keep AUTOINCREMENT ahead of any explicitly-inserted ids so a later normal
        # insert can't collide. (sqlite_sequence has no unique index on `name`, so we
        # SELECT-then-INSERT/UPDATE rather than upsert; the table is absent only if no
        # AUTOINCREMENT row has ever existed, which try/except covers.)
        mx = conn.execute("SELECT COALESCE(MAX(id), 0) AS mx FROM builds").fetchone()["mx"]
        try:
            seq_row = conn.execute(
                "SELECT seq FROM sqlite_sequence WHERE name = 'builds'"
            ).fetchone()
            if seq_row is None:
                conn.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('builds', ?)", (mx,))
            elif seq_row["seq"] < mx:
                conn.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = 'builds'", (mx,))
        except sqlite3.OperationalError:
            pass  # sqlite_sequence not present yet; nothing to reconcile
        conn.commit()
    finally:
        conn.close()
    return summary
