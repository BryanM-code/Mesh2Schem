"""Review images for a pair: what went in, what the model sees, what it must produce.

Panels, all on the Library's orbit camera (`thumbs`) so they line up with each other:

    source.png            the clean source mesh, in its own material colours
    degraded.png          the TRELLIS-like degraded mesh the input was made from
    input.png             the input voxels, in their (degraded) colours -- what the model sees
    shapes.png            the target coloured by construction: full, slab, stairs by facing,
                          glass -- the quickest way to judge a roof
    target.png            the Minecraft target (primary roof style), front
    target_back.png       ... from behind (the side TRELLIS never saw)
    target_geometric.png  the other roof style, for comparison
    review.png            all of the above plus a metadata tile, one image per pair

Meshes are drawn by splatting points sampled off the surface (fully vectorised; at this size
indistinguishable from rasterising); voxels with the Library's own cube renderer.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

import block_colors
import thumbs
from schem_voxels import VoxelGrid

SIZE = 360
YAW, PITCH = 40.0, 30.0
SHAPE_COLOURS = {"full": (185, 185, 190), "glass": (110, 205, 235), "slab_bottom": (235, 195, 70),
                 "slab_top": (225, 140, 45), "east": (220, 75, 65), "west": (70, 165, 95),
                 "north": (75, 115, 225), "south": (175, 95, 205), "other": (120, 120, 120),
                 "fence": (165, 100, 45), "wall": (90, 200, 170)}


@contextmanager
def _colour_names(lut):
    original = block_colors.color_for
    block_colors.color_for = lambda s, _o=original: lut.get(s) or _o(s)
    try:
        yield
    finally:
        block_colors.color_for = original


def _orbit(grid_yzx, names, yaw, pitch, size):
    vg = VoxelGrid(np.asarray(grid_yzx, dtype=np.uint16), list(names))
    return thumbs._render_orbit(vg, yaw, pitch, None, size, 300_000)


def render_blocks(grid_yzx, names, yaw=YAW, pitch=PITCH, size=SIZE):
    # Use the same measured palette as block assignment. The Library's generic dye
    # fallback paints cyan terracotta teal and green terracotta bright green, which
    # misrepresents valid schematic materials during visual quality review.
    from mcvoxel.block_palette import AUX_PALETTE, BLOCK_PALETTE
    from mcvoxel.block_shapes import base_block

    lut = {name: BLOCK_PALETTE.get(base_block(name)) or AUX_PALETTE.get(base_block(name))
           for name in names}
    with _colour_names(lut):
        return _orbit(grid_yzx, names, yaw, pitch, size)


def render_rgb(occ_yzx, rgb_yzx, yaw=YAW, pitch=PITCH, size=SIZE):
    """Solid voxels drawn in their own colour (quantised to 16 levels a channel)."""
    occ = np.asarray(occ_yzx, dtype=bool)
    if not occ.any():
        return None
    ys, zs, xs = np.nonzero(occ)
    cols = (np.asarray(rgb_yzx)[ys, zs, xs] // 16 * 16 + 8).clip(0, 255)
    keys, inv = np.unique(cols, axis=0, return_inverse=True)
    names = ["minecraft:air"] + ["rgb:%d,%d,%d" % tuple(k) for k in keys]
    grid = np.zeros(occ.shape, dtype=np.uint16)
    grid[ys, zs, xs] = inv.ravel() + 1
    lut = {n: tuple(int(c) for c in k) for n, k in zip(names[1:], keys)}
    with _colour_names(lut):
        return _orbit(grid, names, yaw, pitch, size)


def shape_class(state: str) -> str:
    name = state.split("[", 1)[0]
    if "glass" in name:
        return "glass"
    if name.endswith("_fence"):
        return "fence"
    if name.endswith("_wall"):
        return "wall"
    if name.endswith("_stairs"):
        for f in ("east", "west", "north", "south"):
            if "facing=" + f in state:
                return f
        return "other"
    if name.endswith("_slab"):
        return "slab_top" if "type=top" in state else "slab_bottom"
    return "full"


def render_shapes(grid_yzx, names, yaw=YAW, pitch=PITCH, size=SIZE):
    """The target with every block coloured by its construction (shape and stair facing)."""
    lut = {n: SHAPE_COLOURS[shape_class(n)] for n in names if not n.startswith("minecraft:air")}
    with _colour_names(lut):
        return _orbit(grid_yzx, names, yaw, pitch, size)


def render_roof_regions(plan, size=SIZE):
    """Top-down plane map with deterministic colours and pitch/direction annotations."""
    from PIL import Image, ImageDraw
    import colorsys
    region = plan.region
    rgb = np.full(region.shape+(3,), (32,34,40), np.uint8)
    records = [row for row in plan.info.get("plane_decisions", []) if row.get("planned",True)]
    lut = {}
    for r in np.unique(region[region>0]):
        colour = tuple(int(255*c) for c in colorsys.hsv_to_rgb((int(r)*.618034)%1,.55,.85))
        lut[int(r)]=colour
        rgb[region==r]=colour
    map_im = Image.fromarray(rgb.transpose(1,0,2), 'RGB')
    map_im.thumbnail((size-16,int(size*.6)), Image.Resampling.NEAREST)
    # Enlarge small footprints for a readable map, without interpolating region ids.
    ratio=min((size-16)/region.shape[0],(size*.6)/region.shape[1])
    map_im=map_im.resize((max(1,int(region.shape[0]*ratio)),max(1,int(region.shape[1]*ratio))),Image.Resampling.NEAREST)
    out=Image.new('RGB',(size,size),(32,34,40));out.paste(map_im,(8,8))
    draw=ImageDraw.Draw(out)
    for r in np.unique(region[region>0]):
        x,z=np.median(np.argwhere(region==r),axis=0)
        draw.text((8+(x+.5)*ratio,8+(z+.5)*ratio),str(r),fill='white')
    y=int(size*.6)+16
    for row in sorted(records,key=lambda r:-r['columns'])[:max(1,(size-y)//14)]:
        r=row['region'];draw.rectangle((8,y+2,16,y+10),fill=lut[r])
        draw.text((22,y),f"{r}: {row['pattern']} {row['direction'] or ''} ({row['columns']} cols)",fill='white')
        y+=14
    return out


def sample_surface(verts, faces, point_colors, n=500_000, seed=0):
    """(points, colours 0..255, face normals) sampled uniformly by area off a mesh."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    ok = area > 0
    if not ok.any():
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(ok)
    pick = idx[rng.choice(len(idx), size=n, p=area[idx] / area[idx].sum())]
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    bary = np.stack([1 - s, s * (1 - r2), s * r2], axis=1)
    pts = (tri[pick] * bary[:, :, None]).sum(axis=1)
    cols = point_colors(pick, bary) if point_colors is not None else np.full((n, 3), 190.0)
    normals = cross[pick] / area[pick, None]
    return pts, np.clip(np.nan_to_num(cols, nan=190.0), 0, 255), normals


def render_scene(sm, verts=None, yaw=YAW, pitch=PITCH, size=SIZE, n=500_000):
    """A SceneMesh (clean source or degraded) by point splatting, in the orbit camera."""
    from PIL import Image
    from mcvoxel.scene import scene_color_fns

    verts = sm.verts if verts is None else verts
    try:
        _, point_colors = scene_color_fns(sm)
    except Exception:
        point_colors = None
    pts, cols, normals = sample_surface(verts, sm.faces, point_colors, n)
    if pts is None:
        return None
    right, down, near = thumbs._view_axes(yaw, pitch)
    sx, sy, depth = pts @ np.array(right), pts @ np.array(down), pts @ np.array(near)
    light = np.array([0.45, 0.8, 0.4])
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.abs(normals @ light)
    rgb = np.clip(cols * shade[:, None], 0, 255).astype(np.uint8)
    canvas = thumbs._splat(sx, sy, depth, rgb)
    if canvas is None:
        return None
    return thumbs._finish(Image.fromarray(canvas, "RGBA"), size)


def info_tile(lines, size=SIZE):
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (size, size), (32, 34, 40, 255))
    d = ImageDraw.Draw(im)
    y = 8
    for line in lines:
        d.text((8, y), str(line)[:58], fill=(225, 226, 230))
        y += 14
        if y > size - 14:
            break
    return im


def sheet(panels, cols=4, size=SIZE):
    """[(image or None, label)] -> one RGB image, `cols` tiles per row."""
    from PIL import Image, ImageDraw
    rows = (len(panels) + cols - 1) // cols
    out = Image.new("RGB", (size * cols, (size + 18) * rows), (22, 22, 26))
    d = ImageDraw.Draw(out)
    for i, (im, label) in enumerate(panels):
        x, y = (i % cols) * size, (i // cols) * (size + 18)
        if im is not None:
            tile = Image.new("RGBA", (size, size), (22, 22, 26, 255))
            im = im.convert("RGBA")
            im.thumbnail((size, size))
            tile.alpha_composite(im, ((size - im.width) // 2, (size - im.height) // 2))
            out.paste(tile.convert("RGB"), (x, y))
        d.text((x + 6, y + size + 3), label, fill=(205, 206, 212))
    return out
