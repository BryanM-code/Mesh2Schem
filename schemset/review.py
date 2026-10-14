"""Review renders: voxel images (used by `schemset.gallery`), contact sheets and cutaways.

Drawn with the Library's own voxel renderer (`thumbs._render_orbit`), so a build looks the
same here as on its Inspect page. Best effort: a build that cannot be drawn gets an empty
tile rather than stopping the run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import schem_voxels
import thumbs

TILE = 320
CAPTION = 46
INTERIOR_MARK = "minecraft:red_concrete"         # interior air, drawn so rooms show up
BUDGET = 60_000


def render(blocks: np.ndarray, palette: list[str], yaw: float = 45, pitch: float = 30,
           cut_above: int | None = None, size: int = TILE):
    """PIL image of an (X, Y, Z) palette grid, or None."""
    if not np.any(blocks):
        return None
    grid = np.ascontiguousarray(np.transpose(blocks, (1, 2, 0)))     # Sponge (Y, Z, X)
    try:
        return thumbs._render_orbit(schem_voxels.VoxelGrid(grid, list(palette)), yaw, pitch,
                                    cut_above, size, BUDGET)
    except Exception:                                                 # noqa: BLE001
        return None


def _tile(image, size: int = TILE):
    from PIL import Image
    canvas = Image.new("RGB", (size, size), (246, 246, 244))
    if image is not None:
        canvas.paste(image.convert("RGBA").resize((size, size)), (0, 0),
                     image.convert("RGBA").resize((size, size)))
    return canvas


def contact_sheet(items: list[dict], path: Path, columns: int = 2) -> None:
    """items: {"caption": str, "panels": [PIL image or None, ...]} -> one PNG."""
    from PIL import Image, ImageDraw
    if not items:
        return
    panels = max(len(it["panels"]) for it in items)
    cell_w, cell_h = panels * TILE + 8, TILE + CAPTION
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for k, it in enumerate(items):
        x0, y0 = (k % columns) * cell_w, (k // columns) * cell_h
        for p, image in enumerate(it["panels"]):
            sheet.paste(_tile(image), (x0 + p * TILE, y0))
        for line, text in enumerate(it["caption"].split("\n")[:3]):
            draw.text((x0 + 4, y0 + TILE + 2 + 14 * line), text, fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, optimize=True)


def interior_view(blocks: np.ndarray, palette: list[str], interior: np.ndarray,
                  cut_above: int, ground_y: int = -1, yaw: float = 45, pitch: float = 35):
    """Cutaway at `cut_above` with interior air (above the ground) filled red, so rooms
    the mask found show red and rooms it missed show empty."""
    mark = len(palette)               # its own entry, even if the build uses red concrete
    shown = list(palette) + [INTERIOR_MARK + "[r2m=interior]"]
    grid = blocks.astype(np.uint16).copy()
    rooms = interior & (blocks == 0)
    rooms[:, :max(ground_y + 1, 0), :] = False
    grid[rooms] = mark
    return render(grid, shown, yaw=yaw, pitch=pitch, cut_above=cut_above)
