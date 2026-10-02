"""Thumbnail rendering for the Library: an isometric picture of every build.

Two renderers, one camera. A schematic is drawn as shaded voxel cubes; a TRELLIS mesh is
drawn by splatting points sampled off its surface. Both go through `project()`, so the two
images of the same build line up -- which is the entire point. A `.glb` that does not
belong to its `.schem` is obvious the moment they sit side by side, and that is what makes
a mis-dropped pair findable without opening anything.

The same voxel drawing also backs the *inspect view*: the picture you get when a thumbnail is
no longer enough. That one is several times the size, on a real orthographic camera you can
point anywhere rather than the thumbnail's one fixed angle, and it can be cut off at a height
to show the inside. Only the frame it opens at is cached; see schem_view.

Thumbnails are a *derived* cache, kept next to the files they depict:

    data/builds/<id>__<slug>/thumb/schem.png
    data/builds/<id>__<slug>/thumb/mesh.png
    data/builds/<id>__<slug>/thumb/view.png

Nothing about them is recorded in the database. A thumbnail is regenerated whenever it is
missing or older than its source, which means it self-heals after a rebuild-from-disk, and
`delete_build`'s rmtree already disposes of it. That keeps this module purely additive:
delete every PNG and the app still works, just slower for one page load.

Nothing here raises into a request. Rendering is best-effort -- an unreadable schematic or
an exotic mesh returns None and the UI shows a placeholder tile.
"""
from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

import block_colors
import block_roles
import db
import schem_voxels

# Output is a square PNG rendered at SUPERSAMPLE x and box-filtered down, which is the
# whole antialiasing story -- cheap, and enough at thumbnail size.
SIZE = 384
SUPERSAMPLE = 2
CANVAS = SIZE * SUPERSAMPLE

# The inspect view: the same voxels, big enough to read individual blocks off, and seen from
# wherever you drag the camera to. Unlike the thumbnail this is a real orthographic camera
# with a free yaw and pitch, so it is not restricted to the angles a grid transpose can give.
VIEW_SIZE = 1024
# What a drag asks for while it is still moving: a quarter of the pixels and a fraction of
# the voxels, so a turn is a stream of frames rather than one arriving after the mouse stops.
VIEW_PREVIEW_SIZE = 384
VIEW_DEFAULT_YAW, VIEW_DEFAULT_PITCH = 45, 35
# Pitch is bounded rather than wrapped. Past vertical the build would be upside down, and
# there is no reading a build from underneath, so the camera stops where the view stops
# being informative -- a little below the horizon, and just short of straight down.
VIEW_PITCH_MIN, VIEW_PITCH_MAX = -25, 89
VIEW_FRAME_MARGIN = 1.03           # breathing room around the build, at its widest angle

# Face shading for voxel cubes: top lit, then the two visible sides.
SHADE_TOP, SHADE_LEFT, SHADE_RIGHT = 1.0, 0.72, 0.55

# Where the light comes from for the free camera, which needs a shade for all four sides
# rather than just a left and a right. Chosen so the two faces visible at the default angle
# land on SHADE_LEFT and SHADE_RIGHT: the view opens looking like the thumbnail, and orbiting
# re-lights the build from a fixed sun instead of repainting whichever face turned up on the
# left. A face's brightness then never changes while you can see it.
LIGHT_DIR = (-0.92, 0.0, -0.39)
SHADE_SIDE_BASE, SHADE_SIDE_RANGE = 0.635, 0.092

# Ceilings on how much work one thumbnail may ask for. A build is downsampled until it
# fits; past that the render is simply coarser, never slower.
MAX_SURFACE_VOXELS = 120_000
MIN_CELL_PX = 2                 # smallest cube edge, in supersampled pixels

# The inspect view is allowed more, because being able to see every block is the point of
# it and it is drawn once per angle rather than once per Library card. Past this the view
# coarsens exactly as a thumbnail does, so the biggest build costs no more than the budget.
MAX_VIEW_SURFACE_VOXELS = 300_000
# A frame drawn while the camera is still moving is allowed far less. It is on screen for a
# fraction of a second and is replaced by the full render the moment the drag stops, so a
# big build turns coarse while it spins rather than turning slowly.
MAX_PREVIEW_SURFACE_VOXELS = 45_000

# Surface samples per mesh render. Sized against the CANVAS, not against the mesh's face
# count: how densely the splats have to fall is a property of how many pixels they must
# cover, and a coarse mesh needs exactly as many as a fine one to render a solid surface.
# Each sample paints 3x3 px over the ~350k a framed model occupies. Every sample also
# becomes nine entries in the depth sort, so this is the knob that sets render cost;
# measured against a closed surface, unpainted pixels inside the silhouette run 0.8% at
# 250k samples, 0.05% at 400k and 0.002% here, for about 0.8s of a once-per-mesh render.
MESH_SAMPLES = 600_000

_locks_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}

__all__ = ["SIZE", "VIEW_SIZE", "VIEW_PREVIEW_SIZE", "VIEW_DEFAULT_YAW", "VIEW_DEFAULT_PITCH",
           "VIEW_PITCH_MIN", "VIEW_PITCH_MAX", "schem_thumb", "mesh_thumb", "thumb_path",
           "clear_mesh_thumb", "schem_file", "schem_view", "schem_orbit",
           "render_schem_view"]


# ── Shared camera ─────────────────────────────────────────────────────────────

def project(x, y, z, cell: float):
    """Map build coordinates to (screen_x, screen_y, depth) in 2:1 isometric.

    The classic pixel-art projection: one cell step along x or z moves `cell` across and
    half a cell down, one step up in y moves a full cell up. Depth is x+y+z, which is
    distance along the (1,1,1) view direction, so sorting by it gives painter's order.

    Used by both renderers -- a voxel at build coordinate p and a mesh vertex at the same
    p land on the same pixel, which is what makes the two thumbnails comparable.
    """
    screen_x = (x - z) * cell
    screen_y = (x + z) * (cell * 0.5) - y * cell
    return screen_x, screen_y, x + y + z


# ── Voxel (schematic) rendering ───────────────────────────────────────────────

def _cube_sprite(cell: int):
    """A cube stencil labelled by face: 0 empty, 1 top, 2 left, 3 right.

    Drawn once per cell size with Pillow's polygon fill, so the three faces tessellate
    exactly and a wall of cubes shows no seams.
    """
    from PIL import Image, ImageDraw

    width, top_h, side_h = cell * 2, cell, cell
    sprite = Image.new("L", (width, top_h + side_h), 0)
    draw = ImageDraw.Draw(sprite)
    # Top rhombus, then the two side quads hanging off its lower edges.
    draw.polygon([(0, cell // 2), (cell, 0), (width, cell // 2), (cell, cell)], fill=1)
    draw.polygon([(0, cell // 2), (cell, cell), (cell, cell + side_h), (0, cell // 2 + side_h)], fill=2)
    draw.polygon([(cell, cell), (width, cell // 2), (width, cell // 2 + side_h), (cell, cell + side_h)], fill=3)

    labels = np.asarray(sprite)
    shade = np.zeros(labels.shape, dtype=np.float32)
    shade[labels == 1] = SHADE_TOP
    shade[labels == 2] = SHADE_LEFT
    shade[labels == 3] = SHADE_RIGHT
    return labels > 0, shade


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


def _fit_cell(w: int, h: int, l: int, canvas: int) -> int:
    """Largest integer cube edge (in supersampled px) at which the build still fits."""
    span_x = (w + l) or 1
    span_y = (w + l - 2) * 0.5 + h + 1
    return int(min(canvas / span_x, canvas / max(span_y, 1)))


def render_schem(schem_path) -> Optional["object"]:
    """Render a schematic as an isometric voxel image. Returns a PIL Image or None."""
    voxels = schem_voxels.load_voxels(schem_path)
    if voxels is None:
        return None
    return _render_voxels(voxels, SIZE, MAX_SURFACE_VOXELS)


def _render_voxels(voxels, size: int, max_surface: int) -> Optional["object"]:
    """Draw a decoded grid isometrically onto a transparent `size` x `size` frame."""
    from PIL import Image

    canvas_px = size * SUPERSAMPLE
    fitted = _fit_to_budget(voxels, max_surface,
                            lambda w, h, l: _fit_cell(w, h, l, canvas_px))
    if fitted is None:
        return None
    grid, solid, visible, factor = fitted

    ys, zs, xs = np.nonzero(visible)
    if len(ys) == 0:
        return None

    h, l, w = grid.shape
    cell = max(MIN_CELL_PX, _fit_cell(w, h, l, canvas_px))
    sprite_mask, sprite_shade = _cube_sprite(cell)
    sprite_h, sprite_w = sprite_mask.shape

    screen_x, screen_y, depth = project(xs.astype(np.float64), ys.astype(np.float64),
                                        zs.astype(np.float64), cell)
    left = np.rint(screen_x - screen_x.min()).astype(np.int64)
    top = np.rint(screen_y - screen_y.min()).astype(np.int64)

    canvas = np.zeros((int(top.max()) + sprite_h, int(left.max()) + sprite_w, 4), dtype=np.uint8)
    colors = block_colors.palette_colors(voxels.names, np)[grid[ys, zs, xs]]

    # Painter's algorithm: farthest first, so nearer cubes overwrite what they hide.
    shaded_rgb = sprite_shade[sprite_mask][:, None]
    for i in np.argsort(depth, kind="stable"):
        y0, x0 = top[i], left[i]
        tile = canvas[y0:y0 + sprite_h, x0:x0 + sprite_w]
        tile[sprite_mask, :3] = shaded_rgb * colors[i]
        tile[sprite_mask, 3] = 255

    return _finish(Image.fromarray(canvas, "RGBA"), size)


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


# ── Inspect view: a camera you can drag ───────────────────────────────────────
# The thumbnail's camera is a fixed 2:1 isometric one, and its cube sprite tessellates only
# at that one angle -- which is why turning the build used to mean transposing the grid, and
# why that only ever gave four angles.
#
# This is a real orthographic camera instead, at any yaw and pitch. The trick that makes the
# thumbnail fast still applies: under an orthographic projection every cube projects to the
# same shape wherever it is, so one stencil is drawn per *frame* and then blitted once per
# voxel. What changes with the angle is which three faces of that stencil are visible and
# how they are shaded -- not the per-voxel work, which is why an arbitrary angle costs the
# same as a quarter-turn did.

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
    degrees to it -- rather than against the current angle. A build that grew and shrank as
    it turned would be unreadable to drag.

    The margin is not decoration: at the diagonal the build fills the frame to the pixel, and
    per-voxel rounding plus the stencil's grown edge would put that last pixel outside it.
    """
    pitch = math.radians(pitch_deg)
    spread = math.hypot(w, l) * VIEW_FRAME_MARGIN
    down = (h * abs(math.cos(pitch)) + spread * abs(math.sin(pitch))) * VIEW_FRAME_MARGIN
    return int(min(canvas / max(spread, 1), canvas / max(down, 1)))


# The six faces of a unit cube: outward normal, then its corners in perimeter order.
_CUBE_FACES = (
    ((0, 1, 0), ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))),
)


def _face_shade(normal) -> float:
    """How lit one face of a cube is, under a sun fixed in the build's own coordinates."""
    if normal == (0, 1, 0):
        return SHADE_TOP
    if normal == (0, -1, 0):
        return SHADE_SIDE_BASE - 2 * SHADE_SIDE_RANGE      # the underside, seen from below
    lit = sum(a * b for a, b in zip(normal, LIGHT_DIR))
    return SHADE_SIDE_BASE + SHADE_SIDE_RANGE * lit


# ── Block shapes ──────────────────────────────────────────────────────────────
# A voxel is not always a cube, and drawing it as one hides the single most-used detail in
# Minecraft house building. Every roof in the real library is made of stairs and slabs; drawn
# as cubes they read as a terraced mass of blocks, which is exactly the wrong thing to
# conclude from a picture of a roof that is not one.
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
        # `facing` names the full-height half. Measured on the library's own roofs: 79% of
        # roof stairs face the uphill direction, which is the side the tall part has to be on
        # for the steps to descend outwards. Both the real builds and the generator agree.
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


def render_schem_view(schem_path, yaw: float = VIEW_DEFAULT_YAW,
                      pitch: float = VIEW_DEFAULT_PITCH, max_layer: Optional[int] = None,
                      size: int = VIEW_SIZE) -> Optional["object"]:
    """Render a schematic from an arbitrary angle. Returns a PIL Image or None.

    `yaw` spins the build (degrees, anticlockwise seen from above) and `pitch` lifts the
    camera above the horizon. `max_layer` is the highest layer to draw, counted from the
    ground at 0, so lowering it takes the roof and upper floors off and leaves the inside
    visible from whatever angle you are looking from.
    """
    voxels = schem_voxels.load_voxels(schem_path)
    if voxels is None:
        return None
    budget = MAX_VIEW_SURFACE_VOXELS if size >= VIEW_SIZE else MAX_PREVIEW_SURFACE_VOXELS
    return _render_orbit(voxels, yaw, pitch, max_layer, size, budget)


def _render_orbit(voxels, yaw: float, pitch: float, cut_above: Optional[int],
                  size: int, max_surface: int) -> Optional["object"]:
    """Draw a decoded grid from an arbitrary camera onto a fixed, centred square frame.

    The frame is worked out from the build's bounding box rather than from what is actually
    drawn, and its scale from the footprint's diagonal rather than from the current yaw. That
    is what makes the view draggable: spinning it, tilting it and cutting the top off all
    leave the build the same size in the same place, so every frame is comparable with the
    last instead of the build swimming around under the pointer.
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


# ── Mesh (TRELLIS glb) rendering ──────────────────────────────────────────────

def render_mesh(glb_path) -> Optional["object"]:
    """Render a .glb by splatting sampled surface points. Returns a PIL Image or None.

    Point splatting rather than triangle rasterisation: sampling the surface and scattering
    the points is fully vectorised, where rasterising a few hundred thousand triangles from
    Python is not. At thumbnail resolution the two are indistinguishable.
    """
    from PIL import Image
    import trimesh

    try:
        mesh = trimesh.load(str(glb_path), force="mesh")
    except Exception:
        return None
    faces = getattr(mesh, "faces", None)
    if faces is None or len(faces) == 0:
        return None

    try:
        points, face_index, colors = _sample_mesh(mesh, trimesh)
    except Exception:
        return None
    if points is None or len(points) == 0:
        return None

    # Normalise into a unit cube so any mesh, at any scale, fills the frame the same way.
    lo, hi = points.min(axis=0), points.max(axis=0)
    extent = float(np.max(hi - lo)) or 1.0
    unit = (points - (lo + hi) * 0.5) / extent

    try:
        normals = np.asarray(mesh.face_normals)[face_index]
    except Exception:
        normals = None
    shade = _lambert(normals, len(unit))

    # cell=1 because _splat scales the result to the frame itself; only the projection's
    # shape matters here, and that is what it shares with the voxel render.
    screen_x, screen_y, depth = project(unit[:, 0], unit[:, 1], unit[:, 2], 1.0)
    rgb = np.clip(colors.astype(np.float32) * shade[:, None], 0, 255).astype(np.uint8)

    canvas = _splat(screen_x, screen_y, depth, rgb)
    if canvas is None:
        return None
    return _finish(Image.fromarray(canvas, "RGBA"))


def _sample_mesh(mesh, trimesh):
    """Sample points + per-point colour off the mesh surface, degrading to grey."""
    try:
        points, face_index, colors = trimesh.sample.sample_surface(
            mesh, MESH_SAMPLES, sample_color=True, seed=0)
        colors = np.asarray(colors)[:, :3].astype(np.float32)
    except Exception:
        # No usable visual (or no Pillow to decode its texture): shape still reads fine.
        points, face_index = trimesh.sample.sample_surface(mesh, MESH_SAMPLES, seed=0)
        colors = np.full((len(points), 3), 190.0, dtype=np.float32)
    return np.asarray(points, dtype=np.float64), np.asarray(face_index), colors


def _lambert(normals, n: int):
    """Simple headlight-ish diffuse term, with enough ambient that nothing goes black."""
    if normals is None or len(normals) != n:
        return np.ones(n, dtype=np.float32)
    light = np.array([0.5, 0.78, 0.38], dtype=np.float64)
    light /= np.linalg.norm(light)
    lam = np.abs(normals @ light)          # abs(): TRELLIS meshes have inconsistent winding
    return (0.42 + 0.58 * lam).astype(np.float32)


def _splat(screen_x, screen_y, depth, rgb):
    """Scatter coloured points into a canvas, nearest-last (painter's algorithm).

    Each point paints a centred 3x3 block, because sampling is random and a smaller mark
    leaves pixels inside the silhouette that no sample ever hit.

    The nine offsets are expanded into the point list *before* the depth sort rather than
    looped around it. Sorting per offset instead would let the last offset's pass paint a
    far point over a near one that an earlier pass had already drawn -- every pixel would
    be a race between the surface facing you and the surface behind it, which shows up as
    heavy salt-and-pepper noise across the whole model.
    """
    span_x = float(screen_x.max() - screen_x.min()) or 1.0
    span_y = float(screen_y.max() - screen_y.min()) or 1.0
    scale = min((CANVAS - 8) / span_x, (CANVAS - 8) / span_y)

    # +1 leaves room for the block's -1 offset, so it stays centred on its point.
    px = np.rint((screen_x - screen_x.min()) * scale).astype(np.int32) + 1
    py = np.rint((screen_y - screen_y.min()) * scale).astype(np.int32) + 1
    height, width = int(py.max()) + 2, int(px.max()) + 2
    if height < 3 or width < 3:
        return None

    offsets = np.array([-1, 0, 1], dtype=np.int32)
    block_y = (py[:, None] + offsets[None, :]).repeat(3, axis=1).ravel()
    block_x = (px[:, None] + np.tile(offsets, 3)[None, :]).ravel()
    block_depth = np.repeat(depth, 9)
    block_rgb = np.repeat(rgb, 9, axis=0)

    order = np.argsort(block_depth, kind="stable")   # ascending, so nearest is written last
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    canvas[block_y[order], block_x[order], :3] = block_rgb[order]
    canvas[block_y[order], block_x[order], 3] = 255
    return canvas


# ── Reference image ───────────────────────────────────────────────────────────

def render_image(image_path) -> Optional["object"]:
    """Thumbnail for the stored reference image -- just a fitted copy.

    Nothing to render here: the reference image is already a picture. It gets the same
    square, transparent-backed treatment as the other two so all three tiles line up.
    """
    from PIL import Image

    try:
        with Image.open(image_path) as img:
            return _finish(img.convert("RGBA"))
    except Exception:
        return None


# ── Shared output ─────────────────────────────────────────────────────────────

def _finish(image, size: int = SIZE):
    """Downsample for antialiasing and centre on a transparent `size` x `size` square."""
    from PIL import Image

    width, height = image.size
    scale = min(size / width, size / height)
    target_w = max(1, min(size, int(round(width * scale))))
    target_h = max(1, min(size, int(round(height * scale))))
    resized = image.resize((target_w, target_h), Image.LANCZOS)

    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(resized, ((size - target_w) // 2, (size - target_h) // 2))
    return out


# ── Cache ─────────────────────────────────────────────────────────────────────

def thumb_path(build_id: int, slug: str, kind: str) -> Path:
    """Where a thumbnail lives. db.build_dir is read at call time, never cached --
    the test suite rebinds db.BUILDS_DIR to a sandbox, and a captured constant would
    write into the real library."""
    return db.build_dir(build_id, slug) / "thumb" / "{}.png".format(kind)


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = _locks[key] = threading.Lock()
        return lock


def _is_fresh(out_path: Path, source: Path) -> bool:
    try:
        return out_path.stat().st_mtime >= source.stat().st_mtime
    except OSError:
        return False


def _ensure(out_path: Path, source: Path, render) -> Optional[Path]:
    """Return a current thumbnail for `source`, rendering it if missing or stale.

    Written to a temp file and moved into place, so a half-written PNG is never served and
    two threads racing on the same build cannot corrupt each other's output.
    """
    if not source.exists():
        return None
    if _is_fresh(out_path, source):
        return out_path

    with _lock_for(out_path):
        if _is_fresh(out_path, source):     # another thread rendered it while we waited
            return out_path
        try:
            image = render(source)
            if image is None:
                return None
            out_path.parent.mkdir(parents=True, exist_ok=True)
            temp = out_path.with_name(out_path.name + ".{}.tmp".format(os.getpid()))
            image.save(temp, "PNG", optimize=True)
            os.replace(temp, out_path)
        except Exception:
            return None                      # a thumbnail is never worth failing a request
    return out_path


def schem_file(build_id: int, slug: str) -> Path:
    """The schematic every picture of this build is drawn from."""
    return db.build_dir(build_id, slug) / "schem" / "original.schem"


def schem_thumb(build_id: int, slug: str) -> Optional[Path]:
    """Path to this build's schematic thumbnail, rendering it on first use."""
    return _ensure(thumb_path(build_id, slug, "schem"), schem_file(build_id, slug),
                   render_schem)


def schem_view(build_id: int, slug: str) -> Optional[Path]:
    """Path to the view the inspector opens with, rendering it on first use.

    Exactly one angle is cached on disk: the one every viewing starts from. Once the camera
    is dragged there are two continuous axes of them, which is not a cache -- those are drawn
    on demand by schem_orbit and left to the browser to keep.
    """
    return _ensure(thumb_path(build_id, slug, "view"), schem_file(build_id, slug),
                   render_schem_view)


def schem_orbit(build_id: int, slug: str, yaw: float, pitch: float,
                max_layer: Optional[int] = None,
                size: int = VIEW_SIZE) -> Optional["object"]:
    """Render one frame of the inspect view and return the image, caching nothing."""
    source = schem_file(build_id, slug)
    if not source.exists():
        return None
    try:
        return render_schem_view(source, yaw=yaw, pitch=pitch, max_layer=max_layer, size=size)
    except Exception:
        return None                          # a picture is never worth failing a request


def image_thumb(build_id: int, slug: str) -> Optional[Path]:
    """Path to this build's generated-image thumbnail, or None when it has no image."""
    source = db.build_dir(build_id, slug) / "image" / "generated.png"
    return _ensure(thumb_path(build_id, slug, "image"), source, render_image)


def shot_thumb(build_id: int, slug: str, slot: int) -> Optional[Path]:
    """Path to the thumbnail for one reference screenshot, or None if that slot is empty."""
    source = db.shot_path(build_id, slug, slot)
    return _ensure(thumb_path(build_id, slug, "shot-{}".format(slot)), source, render_image)


def mesh_thumb(build_id: int, slug: str) -> Optional[Path]:
    """Path to this build's TRELLIS mesh thumbnail, or None when it isn't paired."""
    source = db.build_dir(build_id, slug) / "trellis" / "output.glb"
    return _ensure(thumb_path(build_id, slug, "mesh"), source, render_mesh)


def clear_mesh_thumb(build_id: int, slug: str) -> None:
    """Drop the mesh thumbnail. Called when a pair is removed, so the stale picture of
    the mesh that is no longer there cannot outlive it."""
    path = thumb_path(build_id, slug, "mesh")
    try:
        if db.BUILDS_DIR in path.resolve().parents:
            path.unlink(missing_ok=True)
    except OSError:
        pass
