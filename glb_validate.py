"""Read-only .glb validation via trimesh.

Reuses the pattern proven in Real2Minecraft's trellis_client: load with force="mesh"
and accept only if the result is a mesh with at least one face. We deliberately do NOT
require a baked texture here -- the spec asks for "a loadable mesh", nothing more.
"""
from __future__ import annotations

from typing import Optional, Tuple


def validate_glb(path) -> Tuple[bool, int, Optional[str]]:
    """Return (ok, face_count, error).

    ok is True only when trimesh loads `path` as a mesh with >0 faces. On failure,
    error is a short human-readable reason and face_count is 0.
    """
    try:
        import trimesh
    except ImportError:  # pragma: no cover - environment issue
        return False, 0, "trimesh is not installed (pip install trimesh)."

    try:
        mesh = trimesh.load(str(path), force="mesh")
    except Exception as exc:
        return False, 0, "Not a loadable mesh: {}".format(exc)

    faces = getattr(mesh, "faces", None)
    n = len(faces) if faces is not None else 0
    if n <= 0:
        return False, 0, "File loaded but contains no mesh faces."
    return True, n, None
