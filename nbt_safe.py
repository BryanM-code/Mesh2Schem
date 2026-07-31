"""Bounded, defensive reading of *untrusted* NBT files.

Schematics downloaded from the internet are hostile input until proven otherwise. A
file can be a decompression bomb, a truncated stub, or not NBT at all -- and nbtlib
will not protect us from any of that. Its reader returns 0 instead of raising when it
runs off the end of the buffer (`read_numeric` swallows `struct.error`), so a 30-byte
file that declares a list of two billion entries sends it into a two-billion-iteration
allocation loop.

So nothing here hands a file to nbtlib until it has been:

1. size-checked and decompressed under a hard output cap (bomb guard), and
2. walked by a strict, allocation-free structural scan that proves every declared
   length, tag id, and nesting depth is sane and fits inside the buffer.

Once the scan passes, nbtlib's parse is guaranteed to terminate on real data.

Nothing in this module executes, evaluates, imports, or resolves anything from the file
-- NBT is inert data, and the only cost of a rejected file is the capped read.
"""
from __future__ import annotations

import gzip
import io
import struct
from pathlib import Path

# Caps. A real single-build schematic is a few hundred KB compressed and a few MB
# expanded; every limit here is orders of magnitude above that and exists only so a
# hostile file fails fast instead of exhausting memory or spinning forever.
MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 192 * 1024 * 1024
MAX_DEPTH = 32
MAX_TAGS = 2_000_000
# How many segments of an NBT path to show before eliding the middle.
_PATH_SEGMENTS = 8

GZIP_MAGIC = b"\x1f\x8b"
TAG_COMPOUND = 0x0A

_U2 = struct.Struct(">H")
_I4 = struct.Struct(">i")

# Payload sizes for the fixed-width scalar tags, and item sizes for the array tags.
_FIXED_PAYLOAD = {1: 1, 2: 2, 3: 4, 4: 8, 5: 4, 6: 8}
_ARRAY_ITEM = {7: 1, 11: 4, 12: 8}


class FileRejected(Exception):
    """Base for every "we refuse to read this file" error.

    Rejections carry structure, not just prose: a stable `code`, the `stage` that threw
    it out, and where the problem is. Tests assert on the code, so rewording a message
    never breaks them, and a rejected download can be diagnosed from the API response
    alone instead of by re-reading the file by hand.

    Stages, in the order a file passes through them:
      ``read``        -- size and existence, before anything is decompressed
      ``decompress``  -- magic bytes and the bounded gzip expansion
      ``structure``   -- the strict NBT tree walk
      ``parse``       -- nbtlib, after the walk cleared it
      ``schematic``   -- schematic-level checks (dimensions, block payload)
    """

    def __init__(self, message, *, code, stage, offset=None, path=None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.stage = stage
        # Offsets are into the *decompressed* NBT, which is what the offending tag is
        # measured in; for an uncompressed file that is also the offset on disk.
        self.offset = offset
        self.path = path            # NBT path, e.g. "TileEntities[3].Items"

    @property
    def where(self) -> str:
        """Location as a phrase, or "" when the file failed before it had one."""
        bits = []
        if self.path:
            bits.append("at {}".format(self.path))
        if self.offset is not None:
            bits.append("byte {}".format(self.offset))
        return ", ".join(bits)

    def details(self) -> dict:
        """The rejection as JSON-friendly fields, ready to return from an endpoint."""
        return {"error": str(self), "message": self.message, "code": self.code,
                "stage": self.stage, "offset": self.offset, "path": self.path}

    def __str__(self):
        return "{} ({})".format(self.message, self.where) if self.where else self.message


class NbtError(FileRejected):
    """Raised when a file is not safe, well-formed, size-bounded NBT."""


# ── Stage 1: read + bounded decompress ────────────────────────────────────────

def read_nbt_bytes(path) -> bytes:
    """Return the uncompressed NBT bytes of `path`, refusing anything oversized.

    Accepts gzipped NBT (every real .schem/.schematic) or raw uncompressed NBT.
    Anything else is rejected on its magic bytes before a single byte is expanded.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise NbtError("Could not read the file: {}".format(exc),
                       code="unreadable", stage="read") from exc
    if size == 0:
        raise NbtError("The file is empty.", code="empty_file", stage="read")
    if size > MAX_COMPRESSED_BYTES:
        raise NbtError(
            "The file is {:.1f} MB; schematics are capped at {} MB.".format(
                size / (1 << 20), MAX_COMPRESSED_BYTES // (1 << 20)),
            code="file_too_large", stage="read")

    with open(path, "rb") as fh:
        magic = fh.read(2)

    if magic == GZIP_MAGIC:
        # read(n) on a GzipFile decompresses incrementally, so asking for one byte
        # past the cap is enough to detect a bomb without ever holding more than that.
        try:
            with gzip.open(str(path), "rb") as gz:
                data = gz.read(MAX_DECOMPRESSED_BYTES + 1)
        except (OSError, EOFError) as exc:
            raise NbtError("The file is not readable gzip data: {}".format(exc),
                           code="bad_gzip", stage="decompress") from exc
        if len(data) > MAX_DECOMPRESSED_BYTES:
            raise NbtError(
                "Refusing to expand this file: it unpacks to more than {} MB "
                "(possible decompression bomb).".format(MAX_DECOMPRESSED_BYTES // (1 << 20)),
                code="decompression_bomb", stage="decompress",
                offset=MAX_DECOMPRESSED_BYTES)
        if not data:
            raise NbtError("The file decompresses to nothing.",
                           code="empty_after_decompress", stage="decompress")
        return data

    if magic[:1] == bytes([TAG_COMPOUND]):
        return path.read_bytes()  # already size-capped above

    raise NbtError(
        "This is not a schematic: the file starts with {} -- expected gzip (1f 8b) "
        "or raw NBT (0a).".format(magic.hex(" ") if magic else "no data"),
        code="bad_magic", stage="decompress", offset=0)


# ── Stage 2: strict structural scan ───────────────────────────────────────────

class _Scanner:
    """Walks the NBT tree measuring tags, without allocating anything from the file.

    A trail of the containers currently open is kept as (offset, length) slices of the
    buffer rather than decoded strings, so the happy path never builds a string. It is
    only turned into a readable path like ``TileEntities[3].Items`` when a scan fails.
    """

    def __init__(self, buf: bytes):
        self.buf = buf
        self.end = len(buf)
        self.tags = 0
        self.trail: list[tuple] = []

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def where(self):
        """The NBT path of the tag being scanned, or None at the document root."""
        parts = [
            "[{}]".format(value) if at is None
            else self.buf[at:at + value].decode("utf-8", "replace")
            for at, value in self.trail
        ]
        if not parts:
            return None
        if len(parts) > _PATH_SEGMENTS:
            # Deep nesting is itself a rejection reason, so the whole chain is noise;
            # the two ends are what identify the tag.
            parts = parts[:2] + ["..."] + parts[-3:]
        rendered = ""
        for part in parts:
            rendered += part if (not rendered or part.startswith("[")) else "." + part
        return rendered

    def _reject(self, message: str, code: str, pos: int):
        raise NbtError(message, code=code, stage="structure", offset=pos, path=self.where())

    # ── The walk ──────────────────────────────────────────────────────────────

    def _take(self, pos: int, count: int) -> int:
        """Return pos+count, or reject if that would run past the end of the buffer."""
        if count < 0:
            self._reject("Malformed NBT: a tag declares a negative length ({}).".format(count),
                         "negative_length", pos)
        stop = pos + count
        if stop > self.end:
            self._reject(
                "Malformed NBT: a tag claims {:,} bytes but only {:,} remain.".format(
                    count, self.end - pos), "truncated", pos)
        return stop

    def payload(self, pos: int, tag_id: int, depth: int) -> int:
        self.tags += 1
        if self.tags > MAX_TAGS:
            self._reject("Malformed NBT: more than {:,} tags.".format(MAX_TAGS),
                         "too_many_tags", pos)
        if depth > MAX_DEPTH:
            self._reject("Malformed NBT: nested deeper than {} levels.".format(MAX_DEPTH),
                         "too_deep", pos)

        size = _FIXED_PAYLOAD.get(tag_id)
        if size is not None:
            return self._take(pos, size)

        if tag_id == 8:  # TAG_String
            return self._take(self._take(pos, 2), _U2.unpack_from(self.buf, pos)[0])

        item = _ARRAY_ITEM.get(tag_id)
        if item is not None:  # TAG_Byte_Array / TAG_Int_Array / TAG_Long_Array
            after_len = self._take(pos, 4)
            count = _I4.unpack_from(self.buf, pos)[0]
            if count < 0:
                self._reject("Malformed NBT: an array declares a negative length ({}).".format(
                    count), "negative_length", pos)
            return self._take(after_len, count * item)

        if tag_id == 9:
            return self._list(pos, depth)
        if tag_id == 10:
            return self._compound(pos, depth)

        self._reject("This is not NBT data: unknown tag id {}.".format(tag_id),
                     "unknown_tag", pos)

    def _list(self, pos: int, depth: int) -> int:
        after_head = self._take(pos, 5)
        subtype = self.buf[pos]
        count = _I4.unpack_from(self.buf, pos + 1)[0]
        if count <= 0:
            return after_head
        if subtype == 0:
            self._reject(
                "Malformed NBT: a TAG_End list declares {:,} entries.".format(count),
                "end_list_entries", pos)
        if subtype > 12:
            self._reject("This is not NBT data: a list declares unknown tag id {}.".format(
                subtype), "unknown_tag", pos)
        # Every element costs at least one byte, so a count past the end of the buffer
        # is a lie. Catching it *here* is what stops nbtlib from looping billions of
        # times over a truncated file instead of raising.
        if count > self.end - after_head:
            self._reject(
                "Malformed NBT: a list declares {:,} entries but only {:,} bytes "
                "remain.".format(count, self.end - after_head), "list_too_long", pos)
        p = after_head
        for i in range(count):
            self.trail.append((None, i))
            p = self.payload(p, subtype, depth + 1)
            self.trail.pop()
        return p

    def _compound(self, pos: int, depth: int) -> int:
        p = pos
        while True:
            p = self._take(p, 1)
            tag_id = self.buf[p - 1]
            if tag_id == 0:  # TAG_End closes the compound
                return p
            name_at = self._take(p, 2)
            name_len = _U2.unpack_from(self.buf, p)[0]
            p = self._take(name_at, name_len)
            self.trail.append((name_at, name_len))
            p = self.payload(p, tag_id, depth + 1)
            self.trail.pop()

    def name(self, pos: int) -> int:
        """Skip a length-prefixed tag name starting at `pos`."""
        after_len = self._take(pos, 2)
        return self._take(after_len, _U2.unpack_from(self.buf, pos)[0])


def scan(data: bytes) -> None:
    """Raise NbtError unless `data` is a structurally valid NBT document."""
    if not data or data[0] != TAG_COMPOUND:
        found = "0x{:02x}".format(data[0]) if data else "nothing"
        raise NbtError(
            "This is not NBT data: the file begins with {}, not a compound tag "
            "(0x0a).".format(found), code="not_compound_root", stage="structure", offset=0)
    scanner = _Scanner(data)
    scanner.payload(scanner.name(1), TAG_COMPOUND, 0)


# ── Public entry point ────────────────────────────────────────────────────────

def load(path):
    """Read `path` as NBT, validating it first. Returns an ``nbtlib.File``.

    Raises NbtError for anything that is not a bounded, well-formed NBT document.
    """
    try:
        import nbtlib
    except ImportError as exc:  # pragma: no cover - environment issue
        raise NbtError("nbtlib is not installed (pip install nbtlib).",
                       code="nbtlib_missing", stage="parse") from exc

    data = read_nbt_bytes(path)
    scan(data)
    try:
        return nbtlib.File.parse(io.BytesIO(data))
    except NbtError:
        raise
    except Exception as exc:  # nbtlib disagreeing after a clean scan is still a bad file
        raise NbtError("Could not parse the NBT data: {}".format(exc),
                       code="parse_failed", stage="parse") from exc
