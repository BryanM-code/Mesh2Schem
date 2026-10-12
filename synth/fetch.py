"""Download one Objaverse GLB from the Hugging Face mirror, verified when a hash is known.

PARKED: see docs/mesh-to-schem-plan.md v4

Streaming by design: the pilot keeps each GLB only until its pair is written (see
`synth.run`, `keep_raw`). The full dataset would be ~100-165 GB of meshes but ~5-10 GB of
pairs, and the machine has ~164 GB free.
"""
from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"


class FetchError(RuntimeError):
    pass


def fetch_glb(path_in_repo: str, dest: Path, sha256: str | None = None, retries: int = 3) -> Path:
    """Download `path_in_repo` (e.g. glbs/000-023/<uid>.glb) to `dest`; reuse if present."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        if not sha256 or _sha256(dest) == sha256:
            return dest
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(BASE + path_in_repo, timeout=120) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            if sha256 and _sha256(tmp) != sha256:
                raise FetchError("sha256 mismatch for {}".format(path_in_repo))
            tmp.replace(dest)
            return dest
        except Exception as exc:          # network hiccup or bad hash: retry with backoff
            last = exc
            time.sleep(2 ** attempt)
    raise FetchError("could not download {}: {}".format(path_in_repo, last))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
