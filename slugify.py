"""Turn a build name into a short, filesystem-safe slug for folder names."""
from __future__ import annotations

import re

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str, max_len: int = 40) -> str:
    """Lowercase, keep [a-z0-9], collapse the rest to single dashes, trim to max_len.

    Always returns a non-empty slug ("build" as a last resort) so folder names stay valid.
    """
    s = _SLUG_RE.sub("-", (name or "").lower()).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "build"
