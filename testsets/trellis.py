#!/usr/bin/env python3
"""The TRELLIS server: rendering a photo of a build for it, and asking it for a mesh.

TRELLIS runs locally in its own bundle (`C:\TRELLIS\run.bat`) and serves an HTTP API on port
7960 (`TRELLIS_URL`, or the environment variable of that name). `trellis_generate` sends an
image and saves the GLB it returns; `render_photo` makes the image: a lit, photo-like render
of a build's clean surface on a transparent background.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

TRELLIS_URL = os.environ.get("TRELLIS_URL", "http://127.0.0.1:7960")
SAMPLING_STEPS = 24                   # TRELLIS's two sampling stages (its default is 12)
MESH_SIMPLIFY = 0.9
TEXTURE_SIZE = 1024
VIEW_YAW, VIEW_PITCH = 35.0, 12.0     # a street-level three-quarter view, like a photo
IMAGE_SIZE = 768


# ── a photo for TRELLIS ───────────────────────────────────────────────────────


def render_photo(scene, size=IMAGE_SIZE, yaw=VIEW_YAW, pitch=VIEW_PITCH, samples=3_000_000,
                 light=(0.55, 0.45)):
    """A photo-like RGBA render of a clean source model: textured, lit, transparent ground.
    `light` = (ambient, sun): colour x (ambient + sun * |normal . sun direction|).

    Point splatting at twice the size, then downsampled for antialiasing; the transparent
    background means the TRELLIS server does not run its own background removal.
    """
    from PIL import Image
    from minecraft import render
    from voxel.scene import scene_color_fns
    from pairs.preview import sample_surface

    _, point_colors = scene_color_fns(scene)
    pts, cols, normals = sample_surface(scene.verts, scene.faces, point_colors, samples, seed=1)
    right, down, near = (np.array(a) for a in render._view_axes(yaw, pitch))
    sx, sy, depth = pts @ right, pts @ down, pts @ near
    sun = np.array([0.35, 0.85, 0.4])
    sun /= np.linalg.norm(sun)
    lam = np.clip(np.abs(normals @ sun), 0, 1)
    rgb = np.clip(cols * (light[0] + light[1] * lam)[:, None], 0, 255).astype(np.uint8)
    big = 2 * size
    span = max(float(np.ptp(sx)), float(np.ptp(sy))) or 1.0
    scale = (big * 0.86) / span
    px = np.rint((sx - sx.min()) * scale + (big - np.ptp(sx) * scale) / 2).astype(np.int64)
    py = np.rint((sy - sy.min()) * scale + (big - np.ptp(sy) * scale) / 2).astype(np.int64)
    canvas = np.zeros((big, big, 4), dtype=np.uint8)
    order = np.argsort(depth, kind="stable")              # far first, near last
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x, y = np.clip(px[order] + dx, 0, big - 1), np.clip(py[order] + dy, 0, big - 1)
        canvas[y, x, :3] = rgb[order]
        canvas[y, x, 3] = 255
    return Image.fromarray(canvas, "RGBA").resize((size, size), Image.LANCZOS)


# ── the server ────────────────────────────────────────────────────────────────

def server_up(url=TRELLIS_URL) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/ping", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def trellis_generate(image_path: Path, out_path: Path, seed: int = 0, url=TRELLIS_URL):
    """POST the image to the TRELLIS server and save the GLB it returns."""
    base = url.rstrip("/")
    body = urllib.parse.urlencode({
        "image_base64": base64.b64encode(Path(image_path).read_bytes()).decode(), "seed": str(seed),
        "ss_sampling_steps": str(SAMPLING_STEPS), "slat_sampling_steps": str(SAMPLING_STEPS),
        "mesh_simplify_ratio": str(MESH_SIMPLIFY), "texture_size": str(TEXTURE_SIZE), "output_format": "glb",
    }).encode()
    req = urllib.request.Request(base + "/generate_no_preview", data=body, method="POST")
    with urllib.request.urlopen(req, timeout=1800) as r:
        model_url = json.loads(r.read().decode()).get("model_url") or "/download/model"
    with urllib.request.urlopen(base + model_url, timeout=180) as r:
        data = r.read()
    if len(data) < 1024:
        raise RuntimeError("TRELLIS returned {} bytes".format(len(data)))
    out_path.write_bytes(data)
    return out_path
