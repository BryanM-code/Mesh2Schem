"""Top-down map of a Bedrock world, for choosing which part of it is a build.

A world export holds a whole area -- the one this was built against is a 672x624
neighbourhood of separate houses. Importing all of it as one build would be both over the
schematic volume limit and a poor training example, so the region has to be picked. This
renders the map you pick it on.

Deliberately not the isometric renderer in `thumbs.py`: for choosing coordinates you want
a plan view where screen position maps directly to world X/Z. It shares `block_colors`, so
a roof looks the same colour here as it will in the finished thumbnail.
"""
from __future__ import annotations

from typing import Optional

import block_colors
import mcworld_convert
from bedrock_world import AIR_BLOCKS

# The map is displayed at a few hundred pixels across and used to drag a box, so more
# resolution than this buys nothing and costs render time on a large world.
MAX_EDGE = 900
MIN_EDGE = 64

__all__ = ["render_world_map", "MAX_EDGE"]


def _surface_grid(world, bounds):
    """Topmost non-air block name and its height, per (x, z) column.

    One pass over the subchunks rather than a lookup per column: a world is sparse, and
    walking what exists is far cheaper than probing every coordinate in its bounding box.
    """
    import numpy as np

    width = bounds["max_x"] - bounds["min_x"] + 1
    length = bounds["max_z"] - bounds["min_z"] + 1
    top_y = np.full((length, width), -1 << 30, dtype=np.int32)
    top_name = {}

    for (cx, cz, sy), (names, indices) in world.subchunks.items():
        base_x, base_y, base_z = cx * 16, sy * 16, cz * 16
        for position, index in enumerate(indices):
            if index >= len(names):
                continue
            name = names[index]
            if name in AIR_BLOCKS:
                continue
            y = base_y + (position & 15)
            x = base_x + ((position >> 8) & 15) - bounds["min_x"]
            z = base_z + ((position >> 4) & 15) - bounds["min_z"]
            if 0 <= x < width and 0 <= z < length and y > top_y[z, x]:
                top_y[z, x] = y
                top_name[(z, x)] = name
    return top_y, top_name


def render_world_map(world, bounds: Optional[dict] = None, highlight: Optional[dict] = None):
    """Render the world from above. Returns a PIL Image, or None if there's nothing to draw.

    `highlight` (a bbox dict) is outlined on the map -- used to show the auto-detected
    build area so there is something sensible to start from.
    """
    from PIL import Image, ImageDraw

    bounds = bounds or world.world_bounds()
    width = bounds["max_x"] - bounds["min_x"] + 1
    length = bounds["max_z"] - bounds["min_z"] + 1
    if width < 1 or length < 1:
        return None

    top_y, top_name = _surface_grid(world, bounds)
    if not top_name:
        return None

    import numpy as np
    canvas = np.zeros((length, width, 4), dtype=np.uint8)

    heights = [y for y in top_y.reshape(-1) if y > -(1 << 29)]
    low, high = (min(heights), max(heights)) if heights else (0, 1)
    span = max(1, high - low)

    for (z, x), name in top_name.items():
        red, green, blue = block_colors.color_for(mcworld_convert.java_name(name))
        # Shade by height so the relief reads -- otherwise a flat plan view of a
        # neighbourhood is a hard-to-parse mosaic of roof colours.
        shade = 0.65 + 0.35 * ((top_y[z, x] - low) / span)
        canvas[z, x] = (min(255, int(red * shade)), min(255, int(green * shade)),
                        min(255, int(blue * shade)), 255)

    image = Image.fromarray(canvas, "RGBA")

    # Nearest-neighbour on the way up: at this scale one block should be a crisp square,
    # not a smudge, because the point is to pick coordinates off it.
    scale = max(1, min(MAX_EDGE // max(width, length), 8))
    if max(width, length) < MIN_EDGE:
        scale = max(scale, MIN_EDGE // max(width, length))
    if scale > 1:
        image = image.resize((width * scale, length * scale), Image.NEAREST)

    if highlight:
        draw = ImageDraw.Draw(image)
        x0 = (highlight["min_x"] - bounds["min_x"]) * scale
        x1 = (highlight["max_x"] - bounds["min_x"] + 1) * scale - 1
        z0 = (highlight["min_z"] - bounds["min_z"]) * scale
        z1 = (highlight["max_z"] - bounds["min_z"] + 1) * scale - 1
        draw.rectangle([x0, z0, x1, z1], outline=(88, 166, 255, 255), width=2)

    return image
