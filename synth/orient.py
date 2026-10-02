"""Stand a model up and square it to the grid.

glTF is Y-up, so a model is taken as upright; `orient` only rotates it about +Y so its
facades lie on the grid axes (`mcvoxel.yaw_align`, the same estimator Real2Minecraft uses on
TRELLIS meshes), which is what lets walls become straight rows of blocks.

A model whose "height" is under half its footprint is flagged `maybe_lying_down`: wide low
buildings trip it as well, so it is recorded for review, not acted on.

Scale lives in `synth.scale` (1:1 with the player).
"""
from __future__ import annotations

import numpy as np

from mcvoxel import yaw_align


def orient(scene):
    """Rotate about +Y so facades lie on the grid axes. Returns (scene, info)."""
    ir = yaw_align.estimate_yaw(scene.verts, scene.faces)
    info = {"yaw_deg": round(float(ir.get("theta_deg", 0.0)), 2), "yaw_applied": bool(ir.get("applied")),
            "yaw_confidence": round(float(ir.get("confidence", 0.0)), 3),
            "wall_area_frac": round(float(ir.get("wall_area_frac", 0.0)), 3)}
    if ir.get("applied"):
        scene = scene.transformed(np.eye(3), (0, 0, 0))
        scene.verts = yaw_align.apply_yaw(scene.verts, ir["theta_deg"])
    lo, hi = scene.bounds
    ext = hi - lo
    info["extent_units"] = [float(v) for v in ext]
    info["maybe_lying_down"] = bool(ext[1] < 0.5 * max(ext[0], ext[2]))
    return scene, info
