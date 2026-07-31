"""Thumbnail rendering for the Library: an isometric picture of every build.

Two renderers, one camera. A schematic is drawn as shaded voxel cubes; a TRELLIS mesh is
drawn by splatting points sampled off its surface. Both go through `project()`, so the two
images of the same build line up -- which is the entire point. A `.glb` that does not
belong to its `.schem` is obvious the moment they sit side by side, and that is what makes
a mis-dropped pair findable without opening anything.

Thumbnails are a *derived* cache, kept next to the files they depict:

    data/builds/<id>__<slug>/thumb/schem.png
    data/builds/<id>__<slug>/thumb/mesh.png

Nothing about them is recorded in the database. A thumbnail is regenerated whenever it is
missing or older than its source, which means it self-heals after a rebuild-from-disk, and
`delete_build`'s rmtree already disposes of it. That keeps this module purely additive:
delete every PNG and the app still works, just slower for one page load.

Nothing here raises into a request. Rendering is best-effort -- an unreadable schematic or
an exotic mesh returns None and the UI shows a placeholder tile.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

import block_colors
import db
import schem_voxels

# Output is a square PNG rendered at SUPERSAMPLE x and box-filtered down, which is the
# whole antialiasing story -- cheap, and enough at thumbnail size.
SIZE = 384
SUPERSAMPLE = 2
CANVAS = SIZE * SUPERSAMPLE

# Face shading for voxel cubes: top lit, then the two visible sides.
SHADE_TOP, SHADE_LEFT, SHADE_RIGHT = 1.0, 0.72, 0.55

# Ceilings on how much work one thumbnail may ask for. A build is downsampled until it
# fits; past that the render is simply coarser, never slower.
MAX_SURFACE_VOXELS = 120_000
MIN_CELL_PX = 2                 # smallest cube edge, in supersampled pixels

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

__all__ = ["SIZE", "schem_thumb", "mesh_thumb", "thumb_path", "clear_mesh_thumb"]


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


def _fit_cell(w: int, h: int, l: int) -> int:
    """Largest integer cube edge (in supersampled px) at which the build still fits."""
    span_x = (w + l) or 1
    span_y = (w + l - 2) * 0.5 + h + 1
    return int(min(CANVAS / span_x, CANVAS / max(span_y, 1)))


def render_schem(schem_path) -> Optional["object"]:
    """Render a schematic as an isometric voxel image. Returns a PIL Image or None."""
    from PIL import Image

    voxels = schem_voxels.load_voxels(schem_path)
    if voxels is None:
        return None
    base_grid, base_solid = voxels.grid, voxels.solid_mask()
    if not base_solid.any():
        return None

    # Shrink until the cubes are big enough to see and there are few enough to draw. The
    # budget is measured against the *surface* count, since that is what the draw loop
    # actually iterates -- a solidly-filled build has orders of magnitude more voxels than
    # visible ones, and gating on the total would render it needlessly blocky. Each attempt
    # downsamples the original rather than the previous result, so the output depends only
    # on the final factor and rounding never compounds.
    grid, solid, factor = base_grid, base_solid, 1
    while True:
        h, l, w = grid.shape
        visible = _surface(solid)
        if factor >= 16 or (_fit_cell(w, h, l) >= MIN_CELL_PX
                            and int(visible.sum()) <= MAX_SURFACE_VOXELS):
            break
        factor += 1
        grid, solid = _downsample(base_grid, base_solid, factor)
        if not solid.any():
            return None

    ys, zs, xs = np.nonzero(visible)
    if len(ys) == 0:
        return None

    h, l, w = grid.shape
    cell = max(MIN_CELL_PX, _fit_cell(w, h, l))
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

    return _finish(Image.fromarray(canvas, "RGBA"))


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

def _finish(image):
    """Downsample for antialiasing and centre on a transparent SIZE x SIZE square."""
    from PIL import Image

    width, height = image.size
    scale = min(SIZE / width, SIZE / height)
    target_w = max(1, min(SIZE, int(round(width * scale))))
    target_h = max(1, min(SIZE, int(round(height * scale))))
    resized = image.resize((target_w, target_h), Image.LANCZOS)

    out = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    out.paste(resized, ((SIZE - target_w) // 2, (SIZE - target_h) // 2))
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


def schem_thumb(build_id: int, slug: str) -> Optional[Path]:
    """Path to this build's schematic thumbnail, rendering it on first use."""
    source = db.build_dir(build_id, slug) / "schem" / "original.schem"
    return _ensure(thumb_path(build_id, slug, "schem"), source, render_schem)


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
