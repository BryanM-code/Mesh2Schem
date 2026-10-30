#!/usr/bin/env python3
"""A normalized build -> the coloured surface a camera sees from outside.

A training input imitates what TRELLIS reconstructs from a photo of the building, so it is
made from the building's *outside* only. `meshify` turns one `normalized/<id>.npz` into a
`voxel.scene.SceneMesh` in the build's own grid frame -- block (x, y, z) spans
[x, x+1) x [y, y+1) x [z, z+1), +X east, +Y up, +Z south, one unit = one block, the frame of
`target_shape` / `target_material` -- which `pairs.degrade` then makes TRELLIS-like.

What is meshed
--------------
The boundary between *exterior space* and everything else, at 1/16 of a block (Minecraft's
own model unit):

* Exterior space is the npz `exterior` mask (depth-capped) plus the empty part of every
  non-cube block in the shell layer that the outside reaches: the air above a slab, beside a
  pane, in front of a door or an open trapdoor, between fence posts. That sub-block part is
  found by a flood over the empty regions of the shell layer's non-cube blocks (empty
  regions of neighbouring blocks connect where they meet on a shared face), seeded from
  exterior air. It never travels through a block that does not itself touch exterior air,
  so carpets or slabs running into a room do not carry the outside in with them.
* Undrawn decor does not stop the outside. The inside/outside flood (`dataset.interior`) treats decor as solid, so air
  enclosed only by decor -- the pockets of a smoke plume of cobwebs above a chimney --
  comes out interior. Here exterior air also flows through decor and through "interior"
  air with no drawn block anywhere above it (a real room always has a drawn roof). Roofed
  rooms, the depth-capped halls included, stop it, and decor under a roof passes it on only
  where it touches exterior air itself, so a cobweb in a doorway never opens the room.
* A face is emitted where exterior space meets a block's geometry (the block's colour) or
  air that is not exterior (an *opening cap*: the room seen through a doorway, an open
  window or beyond the depth cap of an open hall, in a dark `CAP_RGB`). Interior blocks,
  furniture behind glass and the inner side of every wall never meet exterior space, so
  they produce no faces: nothing of the interior enters the input.
* No ground plate: the ground (`ground_height`, including the virtual plane under
  builds with no terrain) is never meshed and never counts as exterior, and neither is the
  space below the grid. The underside of a block resting on the ground is not meshed.
  One exception keeps flush features: a block *in* the ground layer (at or below its
  column's ground height) that the outside reaches -- a path, a patio, a plaza paved into
  the terrain -- gets dark faces (`CAP_RGB`) where it meets the ground, so it is a closed
  slab rather than a zero-thickness sheet that degradation would erase. The target
  supervises those blocks (1.6% of all supervised blocks, over 5% in 55 builds).

Block geometry (`geometry`), in 1/16 units: full cubes (leaves too); slabs; stairs, both
halves, with their corner shapes; thin panes and iron bars (post + arms); fences (post +
two rails per arm); walls (post + low or tall arms); fence gates (posts and rails, closed or
swung open, lowered in a wall); trapdoors (closed bottom / top, open against the side
opposite their `facing`); doors (both halves, the plate on the side opposite `facing`, or
swung to the hinge side when open -- the vocabulary's rule); carpets; snow layers; fluids
as a 14/16 box. Decor is approximated by one or two boxes (beds, chests, lanterns, pots,
campfires, ...) or skipped (plants, torches, signs, banners, buttons, rails, chains, rods,
ladders, cobwebs, ...): a skipped block is empty space, so the wall behind it is meshed.
Unknown non-vocabulary blocks are cubes; `info["default_cube"]` lists them.

States a format did not store (legacy `.schematic` panes, fences and walls carry no
connections; legacy stairs no corner shape) are derived for the geometry only with the
game's own rules (`voxel.blockstates.stair_shapes` / `structural_connections`); stored
states are used as stored. Targets are never touched.

Colours are `block_colors.color_for` colours: one flat colour per block, no
texture (no Minecraft texture file is read or shipped). That is a domain gap: real photos,
and TRELLIS, show texture, weathering and lighting; `pairs.degrade` adds noise and baked
lighting on top. Glass faces get `KIND_GLASS`, door faces `KIND_DOOR`, the rest
`KIND_SURFACE`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage

MESH_VERSION = "meshify-3"                # change it whenever the meshes change
N = 16                                    # model units per block
CAP_RGB = (38, 36, 34)                    # a room seen through an opening: dark, neutral
FLUID_TOP = 14
SIX = ndimage.generate_binary_structure(3, 1)
FULL_BOX = (0, 0, 0, 16, 16, 16)

# Voxel kinds on the (padded) grid.
T_EXT, T_FULL, T_ROOM, T_GROUND, T_PART = 0, 1, 2, 3, 4
# Pixel codes on a voxel face.
P_SOLID, P_CAP, P_EXT, P_GROUND = 0, 1, 2, 3

OPPOSITE = {"east": "west", "west": "east", "north": "south", "south": "north"}
LEFT = {"east": "north", "north": "west", "west": "south", "south": "east"}
RIGHT = {v: k for k, v in LEFT.items()}
CW = {"north": "east", "east": "south", "south": "west", "west": "north"}
# The half of a voxel on one horizontal side, as (x0, x1, z0, z1).
HALF = {"east": (8, 16, 0, 16), "west": (0, 8, 0, 16), "south": (0, 16, 8, 16),
        "north": (0, 16, 0, 8)}
AIR_NAMES = {"air", "cave_air", "void_air"}


@dataclass(frozen=True)
class Geometry:
    boxes: tuple                          # ((x0, y0, z0, x1, y1, z1), ...) in 1/16
    kind: int                             # voxel.scene KIND_*
    rgb: tuple
    source: str                           # "shape" | "vocab" | "approx" | "skip" | "default"

    @property
    def full(self) -> bool:
        return self.boxes == (FULL_BOX,)


# --------------------------------------------------------------------------- geometry

def _name_props(state: str):
    from voxel.vocab import _props
    name = state.split("[", 1)[0]
    if ":" in name:
        name = name.split(":", 1)[1]
    return name, _props(state)


def _plate(side: str, thick: int):
    """A plate `thick` units deep against the voxel's `side` face."""
    return {"north": (0, 0, 0, 16, 16, thick), "south": (0, 0, 16 - thick, 16, 16, 16),
            "west": (0, 0, 0, thick, 16, 16), "east": (16 - thick, 0, 0, 16, 16, 16)}[side]


def _quadrant(a: str, b: str):
    """The quarter of the voxel's footprint where side `a` and side `b` meet."""
    xa = HALF[a] if a in ("east", "west") else HALF[b]
    za = HALF[a] if a in ("north", "south") else HALF[b]
    return xa[0], xa[1], za[2], za[3]


def _stairs(props):
    facing = props.get("facing", "east")
    top = props.get("half", "bottom") == "top"
    shape = props.get("shape", "straight")
    slab_y, step_y = ((8, 16), (0, 8)) if top else ((0, 8), (8, 16))
    parts = []
    if shape == "outer_left":
        parts.append(_quadrant(facing, LEFT[facing]))
    elif shape == "outer_right":
        parts.append(_quadrant(facing, RIGHT[facing]))
    else:
        parts.append(HALF[facing])
        if shape == "inner_left":
            parts.append(HALF[LEFT[facing]])
        elif shape == "inner_right":
            parts.append(HALF[RIGHT[facing]])
    boxes = [(0, slab_y[0], 0, 16, slab_y[1], 16)]
    boxes += [(x0, step_y[0], z0, x1, step_y[1], z1) for x0, x1, z0, z1 in parts]
    return boxes


def _arms(props, post, arm):
    """Post plus one arm per connected side; `arm(side)` -> list of boxes."""
    boxes = [post] if post else []
    for side in ("north", "south", "west", "east"):
        value = props.get(side, "false")
        if value not in ("false", "none"):
            boxes += arm(side, value)
    return boxes


def _span(side, y0, y1, lo, hi):
    """An arm from the post toward `side`, `lo..hi` wide, y0..y1 tall."""
    return {"north": (lo, y0, 0, hi, y1, 8), "south": (lo, y0, 8, hi, y1, 16),
            "west": (0, y0, lo, 8, y1, hi), "east": (8, y0, lo, 16, y1, hi)}[side]


def _pane(props):
    return _arms(props, (7, 0, 7, 9, 16, 9), lambda s, v: [_span(s, 0, 16, 7, 9)])


def _fence(props):
    return _arms(props, (6, 0, 6, 10, 16, 10),
                 lambda s, v: [_span(s, 6, 9, 7, 9), _span(s, 12, 15, 7, 9)])


def _wall(props):
    post = (4, 0, 4, 12, 16, 12) if props.get("up", "true") == "true" else None
    boxes = _arms(props, post, lambda s, v: [_span(s, 0, 16 if v == "tall" else 14, 5, 11)])
    return boxes or [(4, 0, 4, 12, 16, 12)]


def _gate(props):
    facing = props.get("facing", "north")
    drop = 3 if props.get("in_wall") == "true" else 0
    along_x = facing in ("north", "south")
    if props.get("open") == "true":
        lo, hi = (9, 16) if facing in ("south", "east") else (0, 7)
        boxes = [(0, 5, 7, 2, 16, 9), (14, 5, 7, 16, 16, 9),
                 (0, 6, lo, 2, 15, hi), (14, 6, lo, 16, 15, hi)]
    else:
        boxes = [(0, 5, 7, 2, 16, 9), (14, 5, 7, 16, 16, 9), (2, 6, 7, 14, 9, 9),
                 (2, 12, 7, 14, 15, 9), (6, 9, 7, 10, 12, 9)]
    if not along_x:                       # the gate spans Z: swap x and z
        boxes = [(z0, y0, x0, z1, y1, x1) for x0, y0, z0, x1, y1, z1 in boxes]
    return [(x0, y0 - drop, z0, x1, y1 - drop, z1) for x0, y0, z0, x1, y1, z1 in boxes]


def _trapdoor(props):
    if props.get("open") == "true":
        return [_plate(OPPOSITE[props.get("facing", "north")], 3)]
    return [(0, 13, 0, 16, 16, 16)] if props.get("half") == "top" else [(0, 0, 0, 16, 3, 16)]


def _door(props):
    facing = props.get("facing", "north")
    if props.get("open") == "true":
        side = CW[facing] if props.get("hinge") == "right" else OPPOSITE[CW[facing]]
    else:
        side = OPPOSITE[facing]
    return [_plate(side, 3)]


# Decor drawn as rough boxes (anything bigger than a few pixels that a camera would see).
APPROX = {
    "bed": [(0, 0, 0, 16, 9, 16)], "chest": [(1, 0, 1, 15, 14, 15)],
    "lantern": [(5, 0, 5, 11, 9, 11)], "pot": [(5, 0, 5, 11, 6, 11)],
    "campfire": [(0, 0, 0, 16, 7, 16)], "cake": [(1, 0, 1, 15, 8, 15)],
    "daylight_detector": [(0, 0, 0, 16, 6, 16)], "decorated_pot": [(1, 0, 1, 15, 16, 15)],
    "enchanting_table": [(0, 0, 0, 16, 12, 16)], "end_portal_frame": [(0, 0, 0, 16, 13, 16)],
    "stonecutter": [(0, 0, 0, 16, 9, 16)], "anvil": [(2, 0, 2, 14, 16, 14)],
    "grindstone": [(4, 0, 2, 12, 16, 14)], "hopper": [(0, 10, 0, 16, 16, 16), (4, 4, 4, 12, 10, 12)],
    "lectern": [(0, 0, 0, 16, 13, 16)], "cactus": [(1, 0, 1, 15, 16, 15)],
    "head": [(4, 0, 4, 12, 8, 12)], "path": [(0, 0, 0, 16, 15, 16)],
}
# Decor too small or thin for a camera (or TRELLIS) to resolve: empty space.
SKIP_NAMES = {
    "allium", "azure_bluet", "blue_orchid", "oxeye_daisy", "cornflower", "dandelion", "poppy",
    "lily_of_the_valley", "wither_rose", "sunflower", "lilac", "peony", "torchflower",
    "pitcher_plant", "spore_blossom", "glow_lichen", "hanging_roots", "vine", "cobweb", "ladder",
    "lever", "tripwire", "tripwire_hook", "redstone_wire", "repeater", "comparator", "end_rod",
    "fire", "soul_fire", "kelp", "seagrass", "tall_seagrass", "sugar_cane", "bamboo",
    "bamboo_sapling", "lily_pad", "wheat", "carrots", "potatoes", "beetroots", "nether_wart",
    "sea_pickle", "scaffolding", "moving_piston", "brewing_stand", "bell", "conduit",
    "end_portal", "leaf_litter", "wildflowers", "resin_clump", "small_dripleaf", "big_dripleaf",
    "big_dripleaf_stem", "closed_eyeblossom", "open_eyeblossom", "azalea", "flowering_azalea",
    "bush", "attached_melon_stem", "attached_pumpkin_stem", "melon_stem", "pumpkin_stem",
    "light", "barrier", "structure_void", "string", "frogspawn", "candle", "item_frame",
    "glow_item_frame", "twisting_vines", "weeping_vines", "cave_vines", "chorus_flower",
    "chorus_plant", "dragon_egg", "turtle_egg", "sniffer_egg", "nether_portal",
}
SKIP_SUFFIXES = ("_button", "_pressure_plate", "rail", "_sign", "_banner", "torch", "_candle",
                 "_sapling", "_propagule", "_tulip", "_fan", "_coral", "_mushroom", "_roots",
                 "_fungus", "_bud", "_cluster", "_vines_plant", "_chain", "lightning_rod",
                 "_wall_head", "_wall_skull", "_petals", "_grass", "fern", "_bush", "_plant",
                 "_flower", "_egg", "golem_statue", "_sprouts")


def _decor(name, props):
    """(boxes, source) for a block outside the shape families and the vocabulary."""
    if name in ("water", "lava", "bubble_column"):
        return [(0, 0, 0, 16, FLUID_TOP, 16)], "approx"
    if name == "snow":
        layers = int(props.get("layers", "1") or 1)
        return [(0, 0, 0, 16, min(2 * layers, 16), 16)], "approx"
    if name.endswith("_bed"):
        return APPROX["bed"], "approx"
    if name.endswith("chest"):
        return APPROX["chest"], "approx"
    if name.endswith("lantern"):
        if props.get("hanging") == "true":
            return [(5, 1, 5, 11, 10, 11)], "approx"
        return APPROX["lantern"], "approx"
    if name == "flower_pot" or name.startswith("potted_"):
        return APPROX["pot"], "approx"
    if name in ("campfire", "soul_campfire"):
        return APPROX["campfire"], "approx"
    if name == "cake" or name.endswith("candle_cake"):
        return APPROX["cake"], "approx"
    if name in ("anvil", "chipped_anvil", "damaged_anvil"):
        return APPROX["anvil"], "approx"
    if name in ("dirt_path", "farmland", "grass_path"):
        return APPROX["path"], "approx"
    if name.endswith("_shelf"):
        return [_plate(OPPOSITE[props.get("facing", "north")], 6)], "approx"
    if name in APPROX:
        return APPROX[name], "approx"
    if name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES):
        return [], "skip"
    if name.endswith(("_head", "_skull")):
        return APPROX["head"], "approx"
    return [FULL_BOX], "default"


@lru_cache(maxsize=None)
def geometry(state: str) -> Geometry | None:
    """The boxes, kind and colour of one blockstate, or None for air."""
    from minecraft import block_colors
    from voxel import vocab
    from voxel.scene import KIND_DOOR, KIND_GLASS, KIND_SURFACE

    name, props = _name_props(str(state))
    if name in AIR_NAMES or not name:
        return None
    kind = KIND_GLASS if "glass" in name else (KIND_DOOR if name.endswith("_door") else KIND_SURFACE)
    rgb = tuple(int(c) for c in block_colors.color_for(str(state)))
    source = "shape"
    if name.endswith("_slab"):
        t = props.get("type", "bottom")
        boxes = [FULL_BOX] if t == "double" else [(0, 8 if t == "top" else 0, 0, 16,
                                                   16 if t == "top" else 8, 16)]
    elif name.endswith("_stairs"):
        boxes = _stairs(props)
    elif name.endswith("_pane") or name == "iron_bars":
        boxes = _pane(props)
    elif name.endswith("_fence_gate"):
        boxes = _gate(props)
    elif name.endswith("_fence"):
        boxes = _fence(props)
    elif name.endswith("_wall"):
        boxes = _wall(props)
    elif name.endswith("_trapdoor"):
        boxes = _trapdoor(props)
    elif name.endswith("_door"):
        boxes = _door(props)
    elif name.endswith("_carpet"):
        boxes = [(0, 0, 0, 16, 1, 16)]
    elif name.endswith("_leaves"):
        boxes = [FULL_BOX]
    else:
        status = vocab.classify_state(str(state))["status"]
        if status in ("exact", "lossy"):
            boxes, source = [FULL_BOX], "vocab"
        else:
            boxes, source = _decor(name, props)
    boxes = tuple(sorted({tuple(int(v) for v in b) for b in boxes
                          if b[3] > b[0] and b[4] > b[1] and b[5] > b[2]}))
    return Geometry(boxes=boxes, kind=kind, rgb=rgb, source=source)


@lru_cache(maxsize=None)
def _local(boxes: tuple):
    """(occ 16^3, empty-region labels 16^3, n regions, face labels (6, 16, 16), face bits (6,))."""
    occ = np.zeros((N, N, N), dtype=bool)
    for x0, y0, z0, x1, y1, z1 in boxes:
        occ[max(x0, 0):min(x1, N), max(y0, 0):min(y1, N), max(z0, 0):min(z1, N)] = True
    labels, n = ndimage.label(~occ, structure=SIX)
    labels = labels.astype(np.int16)
    faces = np.stack([labels[0], labels[-1], labels[:, 0], labels[:, -1],
                      labels[:, :, 0], labels[:, :, -1]])
    bits = np.zeros(6, dtype=np.int64)
    for f in range(6):
        for c in np.unique(faces[f]):
            if c:
                bits[f] |= 1 << (int(c) - 1)
    return occ, labels, int(n), faces, bits


# --------------------------------------------------------------------------- derived states

_NEEDS = re.compile(r"_(pane|fence|wall)$|^iron_bars$")


def derive_states(blocks: np.ndarray, palette) -> tuple[np.ndarray, list, dict]:
    """Fill neighbour-derived properties a format did not store, for the geometry only.

    Stairs without `shape=` get the game's corner rule, panes / bars / fences / walls
    without connection properties get theirs; every other block keeps its stored state.
    Returns (blocks, palette, counts) -- a new palette (appended entries) when anything
    was derived."""
    from voxel.blockstates import stair_shapes, structural_connections

    palette = [str(s) for s in palette]
    stairs, links = set(), set()
    for i, s in enumerate(palette):
        name, props = _name_props(s)
        if name.endswith("_stairs") and "shape" not in props:
            stairs.add(i)
        elif _NEEDS.search(name) and not any(k in props for k in ("north", "east", "south", "west")):
            links.add(i)
    counts = {"stairs": 0, "connections": 0}
    if not stairs and not links:
        return blocks, palette, counts
    grid = np.ascontiguousarray(np.transpose(blocks, (1, 2, 0)))         # Sponge (Y, Z, X)
    out, names = grid, palette
    if stairs:
        out, names = stair_shapes(out, names)
    if links:
        out, names = structural_connections(out, names)
    take = np.isin(grid, sorted(stairs | links))
    counts["stairs"] = int(np.isin(grid, sorted(stairs)).sum())
    counts["connections"] = int(np.isin(grid, sorted(links)).sum())
    merged = np.where(take, out, grid)
    return np.ascontiguousarray(np.transpose(merged, (2, 0, 1))), list(names), counts


# --------------------------------------------------------------------------- the skin

STEPS = ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1))   # face f = 2 * axis + (sign > 0)
MAX_REGIONS = 20                         # empty regions per block (a cross pane has 4)


@lru_cache(maxsize=1 << 16)
def _rects_of(shape: tuple, data: bytes) -> tuple:
    """Greedy cover of a 2-D boolean mask by rectangles: ((u0, u1, v0, v1), ...)."""
    m = np.frombuffer(data, dtype=bool).reshape(shape).copy()
    out = []
    H, W = shape
    for i in range(H):
        j = 0
        while j < W:
            if not m[i, j]:
                j += 1
                continue
            j1 = j
            while j1 < W and m[i, j1]:
                j1 += 1
            i1 = i + 1
            while i1 < H and m[i1, j:j1].all():
                i1 += 1
            m[i:i1, j:j1] = False
            out.append((i, i1, j, j1))
            j = j1
    return tuple(out)


def _rects(mask: np.ndarray) -> tuple:
    mask = np.ascontiguousarray(mask, dtype=bool)
    return _rects_of(mask.shape, mask.tobytes())


def _corners(axis: int, sign: int, plane: float, u0, u1, v0, v1) -> np.ndarray:
    """(4, 3) quad corners, counter-clockwise seen from the side the normal points to.
    (u, v) run along the two other axes in order: (y, z) for X, (x, z) for Y, (x, y) for Z."""
    uv = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    if (sign < 0) != (axis == 1):
        uv = uv[::-1]
    other = [a for a in range(3) if a != axis]
    pts = np.zeros((4, 3))
    for k, (u, v) in enumerate(uv):
        pts[k, axis] = plane
        pts[k, other[0]] = u
        pts[k, other[1]] = v
    return pts


@lru_cache(maxsize=None)
def _quad_set(axis: int, sign: int, plane: int, rects: tuple) -> np.ndarray:
    """(n, 4, 3) local quads (1/16 units) for rectangles on one plane."""
    return np.stack([_corners(axis, sign, plane, *r) for r in rects]).reshape(-1, 4, 3)


def _region_mask(bits: int, n: int) -> np.ndarray:
    """bool (n + 1,): which region labels (1..n) are exterior; label 0 (solid) never."""
    out = np.zeros(n + 1, dtype=bool)
    for c in range(n):
        out[c + 1] = bool((bits >> c) & 1)
    return out


@lru_cache(maxsize=None)
def _internal(boxes: tuple, bits: int) -> np.ndarray:
    """(n, 4, 3) local quads between a block's solid cells and its exterior empty cells,
    inside the voxel (its six voxel faces are the caller's)."""
    occ, labels, n, _, _ = _local(boxes)
    ext = _region_mask(bits, n)[labels]
    out = []
    for axis in range(3):
        o = np.moveaxis(occ, axis, 0)
        e = np.moveaxis(ext, axis, 0)
        for sign, mask in ((1, o[:-1] & e[1:]), (-1, e[:-1] & o[1:])):
            for i in range(N - 1):
                if mask[i].any():
                    out.append(_quad_set(axis, sign, i + 1, _rects(mask[i])))
    return np.concatenate(out) if out else np.zeros((0, 4, 3))


@lru_cache(maxsize=None)
def _touching(boxes_a: tuple, boxes_b: tuple, axis: int) -> tuple:
    """Region pairs (ca, cb) of block a and block b = a + e_axis that meet on the shared face."""
    la = _local(boxes_a)[3][2 * axis + 1]
    lb = _local(boxes_b)[3][2 * axis]
    m = (la > 0) & (lb > 0)
    return tuple(sorted({(int(u), int(v)) for u, v in zip(la[m], lb[m])}))


def _groups(keys: np.ndarray):
    """(key, indices) for every distinct value of `keys`."""
    if not len(keys):
        return
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    ends = np.r_[starts[1:], len(k)]
    for s, e in zip(starts, ends):
        yield k[s], order[s:e]


def _shifted(arr: np.ndarray, axis: int, sign: int) -> np.ndarray:
    """`arr` (padded by one voxel) at voxel + sign * e_axis, for every core voxel."""
    sl = [slice(1, -1)] * 3
    sl[axis] = slice(1 + sign, arr.shape[axis] - 1 + sign)
    return arr[tuple(sl)]


def _exterior_regions(T, part, g_of_part, boxes_g):
    """(bitmask of exterior empty regions per non-cube block, shell flag per block).

    Nodes are the empty regions of non-cube blocks that touch exterior air (the shell
    layer); edges join regions of face-neighbouring shell blocks that meet on the shared
    face; a connected group that touches exterior air on any block face is exterior."""
    n_part = int(part.sum())
    ncomp_g = np.array([_local(b)[2] for b in boxes_g], dtype=np.int64)
    fbits_g = np.array([_local(b)[4] for b in boxes_g], dtype=np.int64).reshape(-1, 6)
    if len(ncomp_g) and ncomp_g.max() > MAX_REGIONS:
        raise ValueError("a block geometry has more than {} empty regions".format(MAX_REGIONS))
    ncomp = ncomp_g[g_of_part]
    seed_bits = np.zeros(n_part, dtype=np.int64)
    shell = np.zeros(n_part, dtype=bool)
    for f, (axis, sign) in enumerate(STEPS):
        nb = _shifted(T, axis, sign)[part] == T_EXT
        shell |= nb
        seed_bits[nb] |= fbits_g[g_of_part[nb], f]
    ext_bits = np.zeros(n_part, dtype=np.int64)
    if not shell.any():
        return ext_bits, shell
    offset = np.zeros(n_part + 1, dtype=np.int64)
    offset[1:] = np.cumsum(np.where(shell, ncomp, 0))
    n_nodes = int(offset[-1])
    sp = np.flatnonzero(shell)
    max_c = int(ncomp[sp].max()) if len(sp) else 0
    seeds = np.zeros(n_nodes, dtype=bool)
    for c in range(max_c):
        has = sp[(ncomp[sp] > c) & (((seed_bits[sp] >> c) & 1) == 1)]
        seeds[offset[has] + c] = True
    index = np.full(T.shape, -1, dtype=np.int64)
    index[1:-1, 1:-1, 1:-1][part] = np.arange(n_part)
    rows, cols = [], []
    for axis in range(3):
        b = _shifted(index, axis, 1)[part]
        a = np.flatnonzero(b >= 0)
        b = b[a]
        keep = shell[a] & shell[b]
        a, b = a[keep], b[keep]
        if not len(a):
            continue
        ga, gb = g_of_part[a], g_of_part[b]
        for _, sel in _groups(ga * (len(boxes_g) + 1) + gb):
            for c1, c2 in _touching(boxes_g[int(ga[sel[0]])], boxes_g[int(gb[sel[0]])], axis):
                rows.append(offset[a[sel]] + c1 - 1)
                cols.append(offset[b[sel]] + c2 - 1)
    if rows:
        import scipy.sparse as sps
        from scipy.sparse.csgraph import connected_components
        r, c = np.concatenate(rows), np.concatenate(cols)
        graph = sps.coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n_nodes, n_nodes))
        _, lab = connected_components(graph, directed=False)
        good = np.zeros(int(lab.max()) + 1, dtype=bool)
        good[lab[seeds]] = True
        node_ext = good[lab]
    else:
        node_ext = seeds
    for c in range(max_c):
        has = sp[ncomp[sp] > c]
        on = node_ext[offset[has] + c]
        ext_bits[has[on]] |= 1 << c
    return ext_bits, shell


class _Quads:
    """Accumulates quads: corners in block units, colour and kind per quad."""

    def __init__(self):
        self.corners, self.rgb, self.kind = [], [], []
        self.counts = {"block": 0, "cap": 0, "internal": 0, "ground": 0}
        self.voxels = {}                 # what -> [lo (3,), hi (3,)] of the voxels it came from

    def add(self, local: np.ndarray, vox: np.ndarray, rgb: np.ndarray, kind: np.ndarray, what: str):
        """`local` (n, 4, 3) quads in 1/16 units, placed at every voxel of `vox` (m, 3);
        `rgb` (m, 3) and `kind` (m,) per voxel."""
        if not len(local) or not len(vox):
            return
        corners = (vox[:, None, None, :].astype(np.float64) + local[None] / N).reshape(-1, 4, 3)
        self.corners.append(corners)
        self.rgb.append(np.repeat(np.asarray(rgb, dtype=np.float64), len(local), axis=0))
        self.kind.append(np.repeat(np.asarray(kind, dtype=np.int32), len(local)))
        self.counts[what] += len(corners)
        lo, hi = vox.min(axis=0), vox.max(axis=0) + 1
        if what in self.voxels:
            lo = np.minimum(lo, self.voxels[what][0])
            hi = np.maximum(hi, self.voxels[what][1])
        self.voxels[what] = [lo, hi]


def meshify(blocks: np.ndarray, palette, exterior: np.ndarray, ground_height=None,
            derive: bool = True):
    """(SceneMesh, info): the outside of a build -- see the module docstring.

    `blocks` (X, Y, Z) indexes `palette`; `exterior` is the npz exterior-air mask;
    `ground_height` (X, Z) int, -1 = none (voxels at or below it are ground)."""
    from voxel.scene import KIND_DOOR, KIND_GLASS, KIND_NAMES, KIND_SURFACE, Material, SceneMesh

    blocks = np.asarray(blocks)
    X, Y, Z = blocks.shape
    info = {"derived": {"stairs": 0, "connections": 0}}
    if derive:
        blocks, palette, info["derived"] = derive_states(blocks, palette)
    palette = [str(s) for s in palette]
    geos = [geometry(s) for s in palette]
    is_air = np.array([g is None for g in geos])
    is_full = np.array([g is not None and g.full for g in geos])
    pal_rgb = np.array([g.rgb if g else CAP_RGB for g in geos], dtype=np.float64)
    pal_kind = np.array([g.kind if g else KIND_SURFACE for g in geos], dtype=np.int32)
    used = np.zeros(len(palette), dtype=bool)
    used[np.unique(blocks)] = True
    sources = {}
    for i, (s, g) in enumerate(zip(palette, geos)):
        if used[i] and g is not None and g.source in ("default", "skip"):
            sources.setdefault(g.source, set()).add(_name_props(s)[0])

    # ── voxel kinds, padded by one: outside the grid is exterior, below it is ground ──
    solid = ~is_air[blocks]
    ext = np.asarray(exterior, dtype=bool) & ~solid
    if ground_height is None:
        ground = np.zeros(blocks.shape, dtype=bool)
    else:
        h = np.broadcast_to(np.asarray(ground_height, dtype=np.int64), (X, Z))
        ground = np.arange(Y)[None, :, None] <= h[:, None, :]
    # Decor is not drawn, but the inside/outside flood treated it as solid. Outside air flows on through
    # it and through "interior" air that has no drawn roof; roofed rooms stop it, and roofed
    # decor passes it only where the decor itself touches exterior air (a cobweb in a
    # doorway does not open the room behind it).
    drawn = np.array([g is not None and bool(g.boxes) for g in geos])[blocks]
    decor = solid & ~drawn
    roofed = np.maximum.accumulate(drawn[:, ::-1, :], axis=1)[:, ::-1, :]
    touching = ndimage.binary_dilation(ext, structure=SIX)
    passable = ext | (~solid & ~ground & ~roofed) | (decor & (~roofed | touching))
    reached = ndimage.binary_propagation(ext, structure=SIX, mask=passable)
    info["opened_by_decor"] = int((reached & ~ext).sum())
    ext = reached
    solid = solid & ~decor           # undrawn decor is air: outside where reached, else a room
    part = solid & ~is_full[blocks]
    T = np.full((X + 2, Y + 2, Z + 2), T_EXT, dtype=np.int8)
    T[:, 0, :] = T_GROUND
    core = T[1:-1, 1:-1, 1:-1]
    core[...] = T_ROOM
    core[~solid & ground] = T_GROUND
    core[ext] = T_EXT
    core[solid & is_full[blocks]] = T_FULL
    core[part] = T_PART
    B = np.zeros(T.shape, dtype=np.int64)          # palette index, padded
    B[1:-1, 1:-1, 1:-1] = blocks

    # ── the empty regions of non-cube blocks that the outside reaches ──
    part_pal = blocks[part]
    pal_ids = sorted({int(i) for i in np.unique(part_pal)})
    boxes_g = [geos[p].boxes for p in pal_ids]
    g_of_pal = np.full(len(palette), -1, dtype=np.int64)
    g_of_pal[pal_ids] = np.arange(len(pal_ids))
    g_of_part = g_of_pal[part_pal]
    ext_bits, shell_part = _exterior_regions(T, part, g_of_part, boxes_g)
    info["partial_blocks"] = int(part.sum())
    info["shell_partial_blocks"] = int(shell_part.sum())

    # ── state ids: 0 exterior air, 1 cube, 2 room air, 3 ground, 4+ (geometry, regions) ──
    sid = np.zeros(T.shape, dtype=np.int64)
    sid[T == T_FULL] = 1
    sid[T == T_ROOM] = 2
    sid[T == T_GROUND] = 3
    combo = g_of_part * (1 << MAX_REGIONS) + ext_bits
    ucombo, inv = np.unique(combo, return_inverse=True)
    inv = inv.ravel()
    sid[1:-1, 1:-1, 1:-1][part] = 4 + inv
    codes = [np.full((6, N, N), P_EXT, np.int8), np.full((6, N, N), P_SOLID, np.int8),
             np.full((6, N, N), P_CAP, np.int8), np.full((6, N, N), P_GROUND, np.int8)]
    combo_g = [(int(c >> MAX_REGIONS), int(c & ((1 << MAX_REGIONS) - 1))) for c in ucombo]
    for g, bits in combo_g:
        _, _, n, faces, _ = _local(boxes_g[g])
        ext_lab = _region_mask(bits, n)
        codes.append(np.where(faces == 0, P_SOLID, np.where(ext_lab[faces], P_EXT, P_CAP)).astype(np.int8))

    quads = _Quads()
    cap_rgb = np.array([CAP_RGB], dtype=np.float64)
    part_pos = np.argwhere(part)

    # internal faces of non-cube blocks toward their own exterior regions
    for k, sel in _groups(inv):
        g, bits = combo_g[int(k)]
        if bits:
            pal = part_pal[sel]
            quads.add(_internal(boxes_g[g], bits), part_pos[sel], pal_rgb[pal], pal_kind[pal], "internal")

    # voxel-boundary faces: per axis, p (lower) and q = p + e_axis (upper), on the padded grid
    whole = {(axis, sign): _quad_set(axis, sign, 0 if sign < 0 else N, ((0, N, 0, N),))
             for axis in range(3) for sign in (-1, 1)}
    for axis in range(3):
        sl_p = [slice(None)] * 3
        sl_q = [slice(None)] * 3
        sl_p[axis] = slice(0, -1)
        sl_q[axis] = slice(1, None)
        sp_, sq_ = sid[tuple(sl_p)], sid[tuple(sl_q)]
        bp, bq = B[tuple(sl_p)], B[tuple(sl_q)]
        step = np.zeros(3, dtype=np.int64)
        step[axis] = 1
        # fast path: a cube or room air facing exterior air across the whole face
        for upper in (False, True):
            emit_s, look_s = (sq_, sp_) if upper else (sp_, sq_)
            sign = -1 if upper else 1
            for kind_sid, what in ((1, "block"), (2, "cap")):
                m = (emit_s == kind_sid) & (look_s == 0)
                if not m.any():
                    continue
                vox = np.argwhere(m) + (step if upper else 0) - 1
                if kind_sid == 1:
                    pal = (bq if upper else bp)[m]
                    quads.add(whole[(axis, sign)], vox, pal_rgb[pal], pal_kind[pal], what)
                else:
                    quads.add(whole[(axis, sign)], vox, np.repeat(cap_rgb, len(vox), axis=0),
                              np.full(len(vox), KIND_SURFACE), what)
        # pixel path: a non-cube block on either side
        m = ((sp_ >= 4) | (sq_ >= 4)) & (sp_ != 3) & (sq_ != 3)
        if not m.any():
            continue
        pos = np.argwhere(m)
        kp, kq = sp_[m], sq_[m]
        pal_p, pal_q = bp[m], bq[m]
        for _, sel in _groups(kp * (len(codes) + 1) + kq):
            a_sid, b_sid = int(kp[sel[0]]), int(kq[sel[0]])
            face_p = codes[a_sid][2 * axis + 1]       # p's +axis face
            face_q = codes[b_sid][2 * axis]           # q's -axis face
            for upper in (False, True):
                e_sid = b_sid if upper else a_sid
                if e_sid in (0, 3):
                    continue
                face_e, face_o = (face_q, face_p) if upper else (face_p, face_q)
                sign, plane = (-1, 0) if upper else (1, N)
                for code, what in ((P_SOLID, "block"), (P_CAP, "cap")):
                    mask = (face_e == code) & (face_o == P_EXT)
                    if not mask.any():
                        continue
                    local = _quad_set(axis, sign, plane, _rects(mask))
                    vox = pos[sel] + (step if upper else 0) - 1
                    if code == P_SOLID:
                        pal = (pal_q if upper else pal_p)[sel]
                        quads.add(local, vox, pal_rgb[pal], pal_kind[pal], what)
                    else:
                        quads.add(local, vox, np.repeat(cap_rgb, len(vox), axis=0),
                                  np.full(len(vox), KIND_SURFACE), what)

    # ground-layer blocks the outside reaches: closed against the ground, in the cap colour
    outside = T == T_EXT                                  # exterior air, padding included
    open_parts = 4 + np.flatnonzero(np.array([b for _, b in combo_g], dtype=np.int64) != 0)
    outside[1:-1, 1:-1, 1:-1] |= np.isin(sid[1:-1, 1:-1, 1:-1], open_parts)   # non-cube blocks it enters
    near = outside[1:-1, 1:-1, 1:-1].copy()
    for axis in range(3):
        for sign in (-1, 1):
            near |= _shifted(outside, axis, sign)
    skin = np.zeros(T.shape, dtype=bool)
    skin[1:-1, 1:-1, 1:-1] = solid & ground & near
    info["ground_layer_skin_blocks"] = int(skin.sum())
    if skin.any():
        for axis in range(3):
            for sign in (-1, 1):
                nb = np.zeros(T.shape, dtype=bool)
                nb[1:-1, 1:-1, 1:-1] = _shifted(T == T_GROUND, axis, sign)
                m = skin & nb
                if not m.any():
                    continue
                pos = np.argwhere(m)
                for key, sel in _groups(sid[m]):
                    face = codes[int(key)][2 * axis + (1 if sign > 0 else 0)]
                    mask = face == P_SOLID
                    if not mask.any():
                        continue
                    local = _quad_set(axis, sign, N if sign > 0 else 0, _rects(mask))
                    vox = pos[sel] - 1
                    quads.add(local, vox, np.repeat(cap_rgb, len(vox), axis=0),
                              np.full(len(vox), KIND_SURFACE), "ground")

    # ── assemble the SceneMesh: four unshared vertices and two triangles per quad ──
    if quads.corners:
        corners = np.concatenate(quads.corners)
        rgb = np.concatenate(quads.rgb)
        kind = np.concatenate(quads.kind)
    else:
        corners, rgb, kind = np.zeros((0, 4, 3)), np.zeros((0, 3)), np.zeros(0, np.int32)
    nq = len(corners)
    verts = corners.reshape(-1, 3)
    base = np.arange(nq, dtype=np.int64)[:, None] * 4
    faces = np.concatenate([base + np.array([0, 1, 2]), base + np.array([0, 2, 3])], axis=1).reshape(-1, 3)
    materials = [Material(name=KIND_NAMES[k], base_rgba=np.ones(4), image=None, image_alpha=None,
                          alpha_mode="OPAQUE", kind=k) for k in (KIND_SURFACE, KIND_GLASS, KIND_DOOR)]
    mesh = SceneMesh(verts=verts.astype(np.float64), faces=faces, uv=np.full((len(verts), 2), np.nan),
                     vcolor=np.repeat(rgb, 4, axis=0), face_mat=np.repeat(kind, 2).astype(np.int32),
                     materials=materials)
    info.update(quads=dict(quads.counts), triangles=int(len(faces)),
                glass_quads=int((kind == KIND_GLASS).sum()), door_quads=int((kind == KIND_DOOR).sum()),
                default_cube=sorted(sources.get("default", ())), skipped=sorted(sources.get("skip", ())))
    # For the frame check (`pairs.make.frame_identity`): where the quads came from, and
    # whether every vertex is on the 1/16 lattice inside the grid.
    info["voxel_bounds"] = {k: [[int(v) for v in lo], [int(v) for v in hi]] for k, (lo, hi) in quads.voxels.items()}
    sixteenths = verts * N
    info["on_lattice"] = bool(np.abs(sixteenths - np.rint(sixteenths)).max(initial=0.0) < 1e-6)
    info["in_grid"] = bool(len(verts) == 0 or ((verts.min(axis=0) >= 0).all()
                                               and (verts.max(axis=0) <= np.array([X, Y, Z])).all()))
    return mesh, info


def meshify_npz(path) -> tuple:
    """Load one normalized npz and meshify it: (SceneMesh, info, npz arrays)."""
    z = np.load(path)
    arrays = {k: z[k] for k in z.files}
    mesh, info = meshify(arrays["blocks"], arrays["palette"], arrays["exterior"],
                         arrays["ground_height"])
    return mesh, info, arrays
