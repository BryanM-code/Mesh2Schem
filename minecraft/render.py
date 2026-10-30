"""Pictures of Minecraft builds, for the review galleries.

Blocks are drawn as shaded cubes (stairs, slabs, fences, panes, doors and trapdoors in their
real shapes) by an orthographic camera at any yaw and pitch. Under an orthographic camera
every block projects to the same shape wherever it is, so one stencil is drawn per block
shape and then copied once per visible block, farthest first. Pictures are drawn at
SUPERSAMPLE x and scaled down, which is the antialiasing.

`draw` is the entry point: an (X, Y, Z) palette grid -> a PIL image.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from minecraft import block_colors
from minecraft import block_roles
from minecraft import schem_voxels

TILE = 320                      # default picture size, in pixels
BUDGET = 60_000                 # visible blocks drawn at most; bigger builds are drawn coarser
SUPERSAMPLE = 2
MIN_CELL_PX = 2                 # smallest cube edge, in supersampled pixels
VIEW_FRAME_MARGIN = 1.03        # room round the build at its widest angle
# Face shading: tops lit fully; sides by a fixed sun in the build's own coordinates, so a
# face keeps its brightness however the camera turns.
SHADE_TOP = 1.0
LIGHT_DIR = (-0.92, 0.0, -0.39)
SHADE_SIDE_BASE, SHADE_SIDE_RANGE = 0.635, 0.092


def draw(blocks: np.ndarray, palette: list[str], yaw: float = 45, pitch: float = 30,
         cut_above: int | None = None, size: int = TILE):
    """PIL image of an (X, Y, Z) palette grid (`palette[i]` is entry i's block state), or
    None when there is nothing to draw. `cut_above` leaves out every layer above it."""
    if not np.any(blocks):
        return None
    grid = np.ascontiguousarray(np.transpose(blocks, (1, 2, 0)))     # Sponge (Y, Z, X)
    try:
        return _render_orbit(schem_voxels.VoxelGrid(grid, list(palette)), yaw, pitch,
                             cut_above, size, BUDGET)
    except Exception:                                                 # noqa: BLE001
        return None


def _downsample(grid, solid, factor: int):
    """Shrink the voxel grid by `factor`, keeping the first solid block in each cell.

    Averaging block *ids* would be meaningless, so this picks a representative instead:
    walk the factor^3 sub-positions and take the first solid one found. factor is 2-4 in
    practice, so this is a few dozen vectorised passes, not a Python loop over voxels.
    """
    h, l, w = grid.shape
    nh, nl, nw = h // factor, l // factor, w // factor
    if nh < 1 or nl < 1 or nw < 1:
        return grid, solid

    out = np.zeros((nh, nl, nw), dtype=grid.dtype)
    out_solid = np.zeros((nh, nl, nw), dtype=bool)
    for dy in range(factor):
        for dz in range(factor):
            for dx in range(factor):
                sub = grid[dy:dy + nh * factor:factor,
                           dz:dz + nl * factor:factor,
                           dx:dx + nw * factor:factor]
                sub_solid = solid[dy:dy + nh * factor:factor,
                                  dz:dz + nl * factor:factor,
                                  dx:dx + nw * factor:factor]
                take = sub_solid & ~out_solid
                out[take] = sub[take]
                out_solid |= sub_solid
    return out, out_solid


def _surface(solid):
    """Voxels that are solid and touch empty space -- the only ones that can be seen.

    A hollowed-out build renders identically to a filled one, so for a typical house this
    throws away most of the work before any drawing happens.
    """
    padded = np.pad(solid, 1, mode="constant", constant_values=False)
    enclosed = (padded[:-2, 1:-1, 1:-1] & padded[2:, 1:-1, 1:-1] &
                padded[1:-1, :-2, 1:-1] & padded[1:-1, 2:, 1:-1] &
                padded[1:-1, 1:-1, :-2] & padded[1:-1, 1:-1, 2:])
    return solid & ~enclosed


def _fit_to_budget(voxels, max_surface: int, fit_cell):
    """Downsample until the cubes are big enough to see and few enough to draw.

    The budget is measured against the *surface* count, since that is what the draw loop
    actually iterates -- a solidly-filled build has orders of magnitude more voxels than
    visible ones, and gating on the total would render it needlessly blocky. Each attempt
    downsamples the original rather than the previous result, so the output depends only on
    the final factor and rounding never compounds.

    Returns (grid, solid, surface, factor), or None when there is nothing left to draw.
    """
    base_grid, base_solid = voxels.grid, voxels.solid_mask()
    if not base_solid.any():
        return None
    grid, solid, factor = base_grid, base_solid, 1
    while True:
        h, l, w = grid.shape
        visible = _surface(solid)
        if factor >= 16 or (fit_cell(w, h, l) >= MIN_CELL_PX
                            and int(visible.sum()) <= max_surface):
            return grid, solid, visible, factor
        factor += 1
        grid, solid = _downsample(base_grid, base_solid, factor)
        if not solid.any():
            return None


# ── The camera ───────────────────────────────────────────────────────────────
# An orthographic camera at any yaw and pitch. Every cube projects to the same shape wherever
# it is, so one stencil is drawn per frame and copied once per block; the angle only changes
# which faces of the stencil are visible and how they are shaded.

def _view_axes(yaw_deg: float, pitch_deg: float):
    """Camera basis: build coords -> (screen right, screen down, towards the viewer).

    Yaw spins the build about its vertical axis; pitch raises the camera above the horizon.
    Each returned triple dots with a build coordinate to give one component, so the whole
    projection is three dot products and no matrices.
    """
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return ((cy, 0.0, sy),                      # screen x
            (-sy * sp, -cp, cy * sp),           # screen y, growing downwards
            (-sy * cp, sp, cy * cp))            # depth, growing towards the viewer


def _fit_orbit_cell(w: int, h: int, l: int, pitch_deg: float, canvas: int) -> int:
    """Cube edge at which the build fits from *any* yaw, so spinning never resizes it.

    Fitted against the diagonal of the footprint -- the widest it ever projects, at 45
    degrees to it -- rather than against the current angle, so views of the same build from
    different sides are drawn at the same scale.

    The margin is not decoration: at the diagonal the build fills the frame to the pixel, and
    per-voxel rounding plus the stencil's grown edge would put that last pixel outside it.
    """
    pitch = math.radians(pitch_deg)
    spread = math.hypot(w, l) * VIEW_FRAME_MARGIN
    down = (h * abs(math.cos(pitch)) + spread * abs(math.sin(pitch))) * VIEW_FRAME_MARGIN
    return int(min(canvas / max(spread, 1), canvas / max(down, 1)))


def _face_shade(normal) -> float:
    """How lit one face of a cube is, under a sun fixed in the build's own coordinates."""
    if normal == (0, 1, 0):
        return SHADE_TOP
    if normal == (0, -1, 0):
        return SHADE_SIDE_BASE - 2 * SHADE_SIDE_RANGE      # the underside, seen from below
    lit = sum(a * b for a, b in zip(normal, LIGHT_DIR))
    return SHADE_SIDE_BASE + SHADE_SIDE_RANGE * lit


# ── Block shapes ──────────────────────────────────────────────────────────────
# A block is not always a cube: roofs are made of stairs and slabs, and drawn as cubes they
# would read as a terraced mass of blocks.
#
# Each shape is a list of axis-aligned boxes in unit-cube coordinates (x, y, z from 0 to 1),
# projected and filled the same way the cube is. Only shapes that change the silhouette
# enough to matter are modelled; anything unrecognised stays a cube.

_CUBE_BOX = ((0.0, 0.0, 0.0, 1.0, 1.0, 1.0),)

# Shapes that fill their block edge to edge in the horizontal plane, and so butt up against
# whatever is beside them. These get the seam-closing dilation; the thin decorative shapes
# that float in the middle of their block do not.
_TILING_SUFFIXES = ("_slab", "_stairs", "_door", "_trapdoor", "_carpet")


def _tiles(base: str) -> bool:
    return base.endswith(_TILING_SUFFIXES) or _shape_boxes(base, {}) == _CUBE_BOX


def _shape_boxes(base: str, state: dict):
    """The boxes one blockstate fills, in unit-cube coordinates. () means "draw nothing"."""
    if base.endswith("_slab"):
        kind = state.get("type", "bottom")
        if kind == "top":
            return ((0.0, 0.5, 0.0, 1.0, 1.0, 1.0),)
        if kind == "double":
            return _CUBE_BOX
        return ((0.0, 0.0, 0.0, 1.0, 0.5, 1.0),)

    if base.endswith("_stairs"):
        # `facing` names the full-height half: on real roofs most stairs face uphill, the
        # side the tall part has to be on for the steps to descend outwards.
        facing = state.get("facing", "north")
        half = state.get("half", "bottom")
        step = ((0.0, 0.0, 0.0, 1.0, 0.5, 1.0) if half == "bottom"
                else (0.0, 0.5, 0.0, 1.0, 1.0, 1.0))
        y0, y1 = (0.5, 1.0) if half == "bottom" else (0.0, 0.5)
        tall = {"north": (0.0, y0, 0.0, 1.0, y1, 0.5),
                "south": (0.0, y0, 0.5, 1.0, y1, 1.0),
                "west": (0.0, y0, 0.0, 0.5, y1, 1.0),
                "east": (0.5, y0, 0.0, 1.0, y1, 1.0)}.get(facing)
        corner = state.get("shape", "straight")
        if corner != "straight" and facing in ("east", "west", "north", "south"):
            ccw = {"north": "west", "west": "south", "south": "east", "east": "north"}
            side = ccw[facing] if corner.endswith("left") else ccw[ccw[ccw[facing]]]
            boxes = [step]
            for ix in (0, 1):
                for iz in (0, 1):
                    high = {"east": ix == 1, "west": ix == 0, "north": iz == 0, "south": iz == 1}
                    occupied = (high[facing] and high[side]) if corner.startswith("outer") else (high[facing] or high[side])
                    if occupied:
                        boxes.append((ix*.5,y0,iz*.5,(ix+1)*.5,y1,(iz+1)*.5))
            return tuple(boxes)
        return (step, tall) if tall else (step,)

    if base.endswith("_trapdoor"):
        facing, half = state.get("facing", "north"), state.get("half", "bottom")
        if state.get("open") == "true":
            t = 0.19
            return {"north": (0.0, 0.0, 0.0, 1.0, 1.0, t),
                    "south": (0.0, 0.0, 1.0 - t, 1.0, 1.0, 1.0),
                    "west": (0.0, 0.0, 0.0, t, 1.0, 1.0),
                    "east": (1.0 - t, 0.0, 0.0, 1.0, 1.0, 1.0)}.get(
                        facing, (0.0, 0.0, 0.0, 1.0, 1.0, t)),
        return (((0.0, 0.0, 0.0, 1.0, 0.19, 1.0),) if half == "bottom"
                else ((0.0, 0.81, 0.0, 1.0, 1.0, 1.0),))

    if base.endswith("_pane") or base.endswith("_bars"):
        return ((0.44, 0.0, 0.0, 0.56, 1.0, 1.0), (0.0, 0.0, 0.44, 1.0, 1.0, 0.56))
    if base.endswith("_fence"):
        boxes = [(0.375, 0.0, 0.375, 0.625, 1.0, 0.625)]
        for direction in ("east", "west", "north", "south"):
            if state.get(direction) != "true":
                continue
            for y0, y1 in ((.375, .5625), (.75, .9375)):
                boxes.append({"east": (.5, y0, .4375, 1., y1, .5625),
                              "west": (0., y0, .4375, .5, y1, .5625),
                              "north": (.4375, y0, 0., .5625, y1, .5),
                              "south": (.4375, y0, .5, .5625, y1, 1.)}[direction])
        return tuple(boxes)
    if base.endswith("_wall"):
        boxes = [(0.25, 0., 0.25, .75, 1., .75)] if state.get("up", "true") == "true" else []
        for direction in ("east", "west", "north", "south"):
            kind = state.get(direction, "none")
            if kind == "none":
                continue
            height = 1. if kind == "tall" else .875
            boxes.append({"east": (.5, 0., .3125, 1., height, .6875),
                          "west": (0., 0., .3125, .5, height, .6875),
                          "north": (.3125, 0., 0., .6875, height, .5),
                          "south": (.3125, 0., .5, .6875, height, 1.)}[direction])
        return tuple(boxes)
    if base.endswith("_fence_gate"):
        return ((0.0, 0.3, 0.44, 1.0, 1.0, 0.56),)
    if base.endswith("_door"):
        return ((0.0, 0.0, 0.0, 1.0, 1.0, 0.19),)
    if base.endswith("_carpet"):
        return ((0.0, 0.0, 0.0, 1.0, 0.0625, 1.0),)
    if base.endswith(("_pressure_plate",)):
        return ((0.06, 0.0, 0.06, 0.94, 0.0625, 0.94),)
    if base.endswith(("_button", "_torch", "_lever")):
        return ((0.35, 0.0, 0.35, 0.65, 0.6, 0.65),)
    return _CUBE_BOX


def _box_faces(box):
    """(normal, 4 corners) for each face of an axis-aligned box."""
    x0, y0, z0, x1, y1, z1 = box
    return (
        ((0, 1, 0), ((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1))),
        ((0, -1, 0), ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))),
        ((1, 0, 0), ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))),
        ((-1, 0, 0), ((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1))),
        ((0, 0, 1), ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))),
        ((0, 0, -1), ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0))),
    )


def _orbit_sprite(cell: int, right, down, near, boxes=_CUBE_BOX, grow=True):
    """One block as this camera sees it: (shade per pixel, mask, x offset, y offset).

    The offsets say where the block's (0,0,0) corner sits inside the stencil, so a voxel is
    drawn by projecting that one corner and blitting. Only the faces turned towards the
    camera are drawn -- the other three are behind them, and the painter's pass over the
    voxels themselves takes care of what is behind what.

    Every shape is drawn on the *cube's* stencil, at the cube's offsets, so a slab and a full
    block can be blitted by the same loop with no per-shape bookkeeping -- a half-height slab
    is simply a stencil with its lower half empty.
    """
    from PIL import Image, ImageDraw

    def flatten(corner):
        return (sum(c * a for c, a in zip(corner, right)) * cell,
                sum(c * a for c, a in zip(corner, down)) * cell)

    cube = [flatten(c) for _, corners in _box_faces(_CUBE_BOX[0]) for c in corners]
    xs = [p[0] for p in cube]
    ys = [p[1] for p in cube]
    # A pixel of margin all round, for the dilation below to grow into.
    off_x, off_y = 1 - math.floor(min(xs)), 1 - math.floor(min(ys))
    width = int(math.ceil(max(xs) - min(xs))) + 3
    height = int(math.ceil(max(ys) - min(ys))) + 3

    stencil = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(stencil)
    shades = [0.0]
    # Farthest box first, so the near half of a stair is drawn over the far half and not the
    # other way round. `near` grows towards the viewer, so ascending order is back to front --
    # negating it here painted every stair's tread over its own riser, which is what made a
    # roof of stairs look like it had bites taken out of it.
    ordered = sorted(boxes, key=lambda b: sum(
        ((b[i] + b[i + 3]) / 2.0) * near[i] for i in range(3)))
    for box in ordered:
        for normal, corners in _box_faces(box):
            if sum(n * a for n, a in zip(normal, near)) <= 0:
                continue                                   # facing away from the camera
            shades.append(_face_shade(normal))
            draw.polygon([(flatten(c)[0] + off_x, flatten(c)[1] + off_y) for c in corners],
                         fill=len(shades) - 1)

    labels = np.asarray(stencil)
    # Shapes that tile edge to edge are grown by a pixel. Voxels are blitted at whole pixels
    # while the lattice they sit on is not, so two neighbours can round apart and leave a
    # hairline of background between them -- visible as pinholes along every joint in a roof.
    # Shapes that stand alone in their block (a pane, a fence post) are left as they are:
    # they have space around them, and widening one would only make it the wrong size.
    if grow:
        labels = _grow_labels(labels)
    shade = np.asarray(shades, dtype=np.float32)[labels]
    return shade, labels > 0, off_x, off_y


def _grow_labels(labels):
    """Widen a stencil by one pixel, so neighbouring cubes meet instead of nearly meeting.

    Voxels are blitted at whole pixels while the lattice they sit on is not, so adjacent
    cubes can round apart and leave a hairline of background between them. Growing each
    stencil into its own margin closes that: cubes now overlap by a pixel instead, and the
    painter's pass already knows which of two overlapping cubes wins.
    """
    grown = labels.copy()
    rows, cols = labels.shape
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        target = grown[max(dy, 0):rows + min(dy, 0), max(dx, 0):cols + min(dx, 0)]
        source = labels[max(-dy, 0):rows + min(-dy, 0), max(-dx, 0):cols + min(-dx, 0)]
        take = (target == 0) & (source != 0)
        target[take] = source[take]
    return grown


def _render_orbit(voxels, yaw: float, pitch: float, cut_above: Optional[int],
                  size: int, max_surface: int) -> Optional["object"]:
    """Draw a decoded grid from an arbitrary camera onto a fixed, centred square frame.

    The frame is worked out from the build's bounding box rather than from what is actually
    drawn, and its scale from the footprint's diagonal rather than from the current yaw, so
    turning, tilting or cutting the top off leaves the build the same size in the same place.
    """
    from PIL import Image

    canvas_px = size * SUPERSAMPLE
    right, down, near = _view_axes(yaw, pitch)
    fitted = _fit_to_budget(voxels, max_surface,
                            lambda w, h, l: _fit_orbit_cell(w, h, l, pitch, canvas_px))
    if fitted is None:
        return None
    grid, solid, visible, factor = fitted

    # Framed against the blocks rather than the grid: schematics routinely carry empty air
    # around the build, and fitting to that would shrink the build to leave room for nothing.
    (y0, y1), (z0, z1), (x0, x1) = _content_box(solid)
    cell = max(MIN_CELL_PX, _fit_orbit_cell(x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1,
                                            pitch, canvas_px))
    sprites, off_x, off_y = _palette_sprites(voxels.names, cell, right, down, near)
    sprite_h, sprite_w = sprites[0][1].shape

    # Centre the build on the canvas. Centred on the *volume* the cubes fill, which runs half
    # a cube past the last voxel's own coordinate in each direction -- centring on the
    # coordinates instead leaves the build sitting half a block off in the frame.
    # The padding is a sprite's worth on each side, so a cube at the very edge is drawn whole
    # and then cropped, rather than clipped mid-blit.
    pad = max(sprite_h, sprite_w) + 2
    centre = ((x0 + x1 + 1) / 2.0, (y0 + y1 + 1) / 2.0, (z0 + z1 + 1) / 2.0)
    origin_x = sum(c * a for c, a in zip(centre, right)) * cell - canvas_px / 2.0 - pad + off_x
    origin_y = sum(c * a for c, a in zip(centre, down)) * cell - canvas_px / 2.0 - pad + off_y
    framed = canvas_px + 2 * pad
    canvas = np.zeros((framed, framed, 4), dtype=np.uint8)

    if cut_above is not None:
        # Cut the solid mask, then re-find the surface: the newly exposed floors and room
        # interiors are surface now, and drawing the pre-cut surface would lid them over.
        solid = solid.copy()
        solid[max(0, min(int(cut_above) // factor, grid.shape[0] - 1)) + 1:] = False
        visible = _surface(solid)

    ys, zs, xs = np.nonzero(visible)
    if len(ys) == 0:
        return _blank(size)                          # cut below the whole build

    fx, fy, fz = xs.astype(np.float64), ys.astype(np.float64), zs.astype(np.float64)
    project_on = lambda axis: fx * axis[0] + fy * axis[1] + fz * axis[2]   # noqa: E731
    left = np.rint(project_on(right) * cell - origin_x).astype(np.int64)
    top = np.rint(project_on(down) * cell - origin_y).astype(np.int64)
    depth = project_on(near)

    # An oversized build can be clamped to MIN_CELL_PX and overrun the frame; drop whatever
    # falls outside it rather than letting a blit run off the end of the canvas.
    inside = ((left >= 0) & (top >= 0)
              & (left + sprite_w <= framed) & (top + sprite_h <= framed))
    if not inside.all():
        keep = np.nonzero(inside)[0]
        left, top, depth = left[keep], top[keep], depth[keep]
        ys, zs, xs = ys[keep], zs[keep], xs[keep]
        if len(ys) == 0:
            return _blank(size)

    palette = grid[ys, zs, xs]
    colors = block_colors.palette_colors(voxels.names, np)[palette]

    # Painter's algorithm: farthest first, so nearer blocks overwrite what they hide. Each
    # voxel picks the stencil its own blockstate calls for, which is what lets a roof of
    # stairs read as a roof of stairs instead of a staircase of cubes.
    for i in np.argsort(depth, kind="stable"):
        shaded_rgb, mask = sprites[palette[i]]
        if shaded_rgb is None:
            continue
        y0, x0 = top[i], left[i]
        tile = canvas[y0:y0 + sprite_h, x0:x0 + sprite_w]
        tile[mask, :3] = shaded_rgb * colors[i]
        tile[mask, 3] = 255

    cropped = canvas[pad:pad + canvas_px, pad:pad + canvas_px]
    return Image.fromarray(cropped, "RGBA").resize((size, size), Image.LANCZOS)


def _palette_sprites(names, cell, right, down, near):
    """One stencil per palette entry, sharing the cube's frame. (sprites, off_x, off_y).

    Built per shape and shared between palette entries that have the same one, so a build
    whose palette holds forty blockstates still only draws a handful of stencils -- the cost
    is a function of how many *shapes* are on screen, not of how many blocks.
    """
    by_shape = {}
    sprites = []
    off_x = off_y = 0
    for name in names:
        base = block_roles.normalise(name)
        state = _blockstate(name)
        boxes = _shape_boxes(base, state)
        key = boxes
        entry = by_shape.get(key)
        if entry is None:
            shade, mask, off_x, off_y = _orbit_sprite(
                cell, right, down, near, boxes, grow=_tiles(base))
            entry = (shade[mask][:, None] if mask.any() else None, mask)
            by_shape[key] = entry
        sprites.append(entry)
    return sprites, off_x, off_y


def _blockstate(name: str) -> dict:
    """`"minecraft:oak_stairs[facing=east,half=top]"` -> {"facing": "east", "half": "top"}."""
    cut = name.find("[")
    if cut < 0 or not name.endswith("]"):
        return {}
    out = {}
    for part in name[cut + 1:-1].split(","):
        key, sep, value = part.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _content_box(solid):
    """(y, z, x) first/last indices holding a block -- the build inside its grid."""
    spans = []
    for axis in (0, 1, 2):
        along = np.any(solid, axis=tuple(a for a in (0, 1, 2) if a != axis))
        hit = np.nonzero(along)[0]
        spans.append((int(hit[0]), int(hit[-1])))
    return spans


def _blank(size: int):
    """An empty frame -- what a cutaway below the whole build honestly looks like."""
    from PIL import Image

    return Image.new("RGBA", (size, size), (0, 0, 0, 0))
