"""Read-only schematic header parsing + content hashing.

Dimensions come from `schem_convert`, which validates the file before anything reads it
and understands both schematic formats (Sponge `.schem` v1/v2/v3 and the legacy MCEdit
`.schematic`). This module is the thin, stable surface the rest of the app calls:
give it a path, get back dimensions or a hash.
"""
from __future__ import annotations

import hashlib

from schem_convert import SchematicError, inspect

# Kept as an alias so existing callers importing SchemParseError keep working; the
# distinction between "unreadable" and "not a schematic" was never meaningful to them.
SchemParseError = SchematicError

__all__ = ["SchemParseError", "SchematicError", "parse_header", "sha256_file"]


def sha256_file(path, chunk_size: int = 1 << 16) -> str:
    """Return the sha256 hex digest of a file's raw bytes (streamed, memory-light)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_header(path) -> tuple[int, int, int]:
    """Return (width, height, length) for a schematic file.

    Raises SchemParseError with a human-readable message for anything that is not a
    valid, safely-sized schematic, so callers can surface it straight to the UI.
    """
    return inspect(path).dims
