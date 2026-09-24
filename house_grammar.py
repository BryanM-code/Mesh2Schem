"""Grow a suburban house from a seed, using proportions measured off the real library.

This is the generator half of the synthetic-build tool. It is a **grammar**, not a model:
footprint, then walls, then roof, then openings, then trim, then the plot -- each stage a
small function that reads the sampled spec and writes voxels. Nothing here is learned. Every
distribution it samples from was measured by `house_stats` and frozen, reviewable, in
`house_params`.

Being rule-based is the point. The problem this whole tool exists to solve is that there are
only about a hundred real (image, schem) pairs, and training a net to make more builds would
need the same data that is missing. A grammar needs no training data -- only a description of
what the real builds look like, which is exactly what Phase 1 produced.

Three decisions worth knowing about:

*   **Materials are sampled as whole tuples, not axis by axis.** Three quarters of the time
    the (wall, roof, trim, foundation) combination is one a human actually built; the rest
    chain conditionals off the wall. Sampling the four independently is how you get lime
    concrete walls under a nether brick roof, and it is the single fastest way to make
    synthetic data look synthetic.
*   **Block states are authored, not defaulted.** Roof stairs carry real `facing` and `half`,
    doors carry `half` and `hinge`. Two thirds of the real builds have states in their
    palettes, and a roof of default-facing stairs reads as wrong at a glance.
*   **A roof is a shell of stairs, not a wedge of cubes.** Measured over the 33 suburban
    builds this generator is aimed at, stairs are a median 34% of every block in the roof
    band and slabs another 7%, and the median roof column is only 40% filled. Earlier
    versions filled the wedge solid and laid one stair course on top, so a flat-roofed house
    came out with 504 cubes and no stairs at all -- see `_place_roof` and `_closing_floor`.
*   **Colour is a constraint, not a draw.** A house whose wall and roof are two shades of the
    same grey is the commonest way this gives itself away, and brightness alone does not
    catch it -- white terracotta under cobblestone is 49 apart and still a grey box. Styles
    are drawn against a contrast *and* a saturation floor, and the wall material is chosen
    before the combination so one popular material cannot take half the batch. See
    `_sample_style`.
*   **Trim is what makes an edge read.** The builds singled out for roofs that pop put light
    material on the perimeter ring of every course -- quoins up the corners, a band at each
    storey, a thin line along the roof's rake -- and use it sparingly: a light share of 0.227
    against 0.440 for the rest. The same accent frames the front door, without which the
    entrance vanishes into a matching wall under a matching porch.
*   **Every house gets a plot.** The three most common blocks in the real library by volume
    are stone, dirt and grass; 94 of 106 builds contain grass. A bare floating shell would be
    off-distribution in exactly the way that matters, since the plot is most of what a
    render and a TRELLIS reconstruction actually see.

The grammar deliberately stops at the exterior shell plus floors. The capture pipeline
photographs the outside, so furniture would be cost without benefit -- see the fidelity note
at the end of the plan for what that does and does not give up.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

import numpy as np

import block_colors
import block_roles
import house_params as P
import schem_write

__all__ = ["HouseSpec", "House", "sample_spec", "build_voxels", "generate"]

AIR = "minecraft:air"

# north = -Z, south = +Z, west = -X, east = +X
NORTH, SOUTH, WEST, EAST = "north", "south", "west", "east"


def pick(rng, table):
    """Weighted choice from a {value: weight} table, order-stable for a given seed."""
    items = sorted(table.items(), key=lambda kv: str(kv[0]))
    return rng.choices([k for k, _ in items], [w for _, w in items], k=1)[0]


# ── the sampled description of a house ───────────────────────────────────────

@dataclass
class HouseSpec:
    seed: int
    width: int
    depth: int
    plan: str
    notch: tuple
    storeys: int
    storey_height: int
    roof_type: str
    pitch: float
    ridge_axis: str
    overhang: int
    wall: str
    roof_stairs: str
    trim: str | None
    foundation: str
    glass: str
    door: str
    window_w: int
    window_h: int
    sill: int
    spacing: int
    door_offset: float
    pad_depth: int
    pad_margin: int
    pad_surface: str
    pad_subsoil: str
    ground_cover: str
    path_block: str
    tree_log: str
    tree_leaves: str
    accent: str
    accent_stairs: str
    gable: str
    chimney_mat: str
    porch_w: int
    porch_d: int
    porch_post_mat: str
    garage_w: int
    garage_door: str
    wing_w: int
    wing_d: int
    wing_offset: float
    wing_storeys: int
    plinth: str | None
    shutter: str
    bush_leaves: str
    flower: str
    path_edge_slab: str
    porch_floor: str
    porch_post: str
    driveway_w: int
    driveway_side: str
    trees: list = field(default_factory=list)
    chimney: bool = False
    porch: bool = False
    driveway: bool = False
    walk_wide: bool = True
    rake_trim: bool = False
    corner_posts: bool = False
    band_course: bool = False
    porch_rail: bool = False
    garage: bool = False
    wing: bool = False

    def summary(self) -> str:
        return ("{}x{} {} {}-storey {} pitch {} / {} + {}".format(
            self.width, self.depth, self.plan, self.storeys, self.roof_type, self.pitch,
            self.wall.replace("minecraft:", ""),
            self.roof_stairs.replace("minecraft:", "")))


@dataclass
class House:
    grid: np.ndarray
    names: list
    spec: HouseSpec


def luminance(blockstate: str) -> float:
    """Perceived brightness of a block, 0-255. The same numbers `thumbs` renders with."""
    r, g, b = block_colors.color_for(blockstate)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def saturation(blockstate: str) -> float:
    """How much colour a block has, 0-1. Grey is 0 whatever its brightness."""
    r, g, b = block_colors.color_for(blockstate)
    top = max(r, g, b)
    return 0.0 if top == 0 else (top - min(r, g, b)) / top


def _style_score(wall, roof):
    """How far apart a wall and roof read. Higher is more clearly two materials.

    Luminance alone is not enough: white terracotta walls under a cobblestone roof are 49
    apart in brightness and still look like one grey box, because nothing in the pair has a
    hue. Saturation carries most of the weight for exactly that reason.
    """
    contrast = abs(luminance(wall) - luminance(roof))
    return contrast + 200.0 * max(saturation(wall), saturation(roof))


def _qualifies(wall, roof) -> bool:
    """Does this pair read as two materials rather than as one grey box?"""
    contrast = abs(luminance(wall) - luminance(roof))
    if contrast < P.MIN_WALL_ROOF_CONTRAST:
        return False
    return saturation(wall) >= P.MIN_WALL_SATURATION or contrast >= P.STRONG_CONTRAST


def _by_wall():
    """Qualifying styles grouped by wall material. Built once; ordered for a stable seed."""
    groups: dict[str, list] = {}
    for style in P.STYLES:
        if _qualifies(style[0], style[1]):
            groups.setdefault(style[0], []).append(tuple(style))
    return groups


_QUALIFYING = _by_wall()


def _sample_style(rng, bias):
    """Wall/roof/trim/foundation as a set, never four independent draws.

    Drawn against a contrast constraint rather than uniformly. The frozen STYLES table has a
    median wall saturation of 0.13 where the real suburban builds measure 0.43, so sampling
    it flat produces grey houses far more often than the library does.

    The wall is chosen first, from the *distinct* materials that have a qualifying partner,
    and the rest of the tuple from that wall's own group. Filtering the 79 tuples and then
    drawing from what survives is the obvious way to do it and the wrong one: seven of the
    survivors have birch walls, so birch came out on 45% of houses. Choosing the material
    first and the combination second is what keeps the batch varied.
    """
    if bias.get("wall"):
        wall = bias["wall"]
        kin = _QUALIFYING.get(wall) or [s for s in P.STYLES if s[0] == wall]
        if kin:
            return tuple(rng.choice(kin))
        # An unknown wall: keep it, and give it the roof that contrasts with it best.
        best = max(P.STYLES, key=lambda s: _style_score(wall, s[1]))
        return (wall,) + tuple(best[1:])

    # Two kinds of suburban house qualify, and they have to be balanced deliberately or the
    # bigger group swamps the other. A wall with a hue -- wood, brick, coloured terracotta --
    # is what the real median saturation of 0.43 describes; a white or pale-grey wall under a
    # dark roof carries its contrast on brightness alone. Both are real, and drawing walls
    # uniformly across all twenty gives the second group two thirds of the batch, because
    # that is how the styles table happens to be composed.
    chromatic = sorted(w for w in _QUALIFYING if saturation(w) >= P.MIN_WALL_SATURATION)
    neutral = sorted(w for w in _QUALIFYING if saturation(w) < P.MIN_WALL_SATURATION)
    group = chromatic if (chromatic and rng.random() < P.P_CHROMATIC_WALL) else neutral
    if not group:
        group = chromatic or neutral
    if not group:                        # no tuple qualifies at all; fall back to the table
        return tuple(rng.choice(P.STYLES))
    return tuple(rng.choice(_QUALIFYING[rng.choice(group)]))


def _pick_contrasting(rng, table, against):
    """A material from `table` that does not read as the same thing as `against`."""
    best = None
    for _ in range(P.STYLE_TRIES):
        choice = pick(rng, table)
        gap = abs(luminance(choice) - luminance(against))
        if gap >= P.MIN_ACCENT_CONTRAST:
            return choice
        if best is None or gap > best[0]:
            best = (gap, choice)
    return best[1]


def _pick_accent(rng, wall, roof):
    """A light trim material that contrasts with both the wall and the roof.

    Same rejection-with-fallback shape as the style draw: an accent the same brightness as
    the wall it is meant to outline is not an accent.
    """
    best = None
    for _ in range(P.STYLE_TRIES):
        accent = pick(rng, P.ACCENT)
        gap = min(abs(luminance(accent) - luminance(wall)),
                  abs(luminance(accent) - luminance(roof)))
        if gap >= P.MIN_ACCENT_CONTRAST:
            return accent
        if best is None or gap > best[0]:
            best = (gap, accent)
    return best[1]


def sample_spec(seed: int, bias: dict | None = None) -> HouseSpec:
    """Draw every parameter of one house. Same seed, same house."""
    bias = dict(bias or {})
    rng = random.Random(seed)

    width = int(bias.get("width") or pick(rng, P.FOOTPRINT_W))
    if bias.get("size") in P.SIZE_TERCILES:
        lo, hi = P.SIZE_TERCILES[bias["size"]]
        width = rng.randrange(lo, hi + 1)
    aspect = pick(rng, P.ASPECT)
    depth = int(bias.get("depth") or max(7, round(width / aspect)))
    if rng.random() < 0.5:
        width, depth = depth, width  # the long axis is not always x

    plan = bias.get("plan") or pick(rng, P.PLAN_SHAPE)
    notch = ()
    if plan in ("L", "T"):
        nx = max(3, int(width * rng.uniform(*P.NOTCH_SHARE)))
        nz = max(3, int(depth * rng.uniform(*P.NOTCH_SHARE)))
        notch = (nx, nz, rng.randrange(4))

    storeys = int(bias.get("storeys") or pick(rng, P.STOREY_COUNT))
    storey_height = int(bias.get("storey_height") or pick(rng, P.STOREY_HEIGHT))

    # Storeys and footprint are correlated in the real library and the marginals cannot say
    # so: sampling them independently puts three storeys on a 12x15 plan and produces a tower.
    # Real builds here are wide and low -- a third storey belongs on a big footprint.
    if not bias.get("storeys"):
        area = width * depth
        if area < 300:
            storeys = min(storeys, 1 if area < 200 else 2)
        elif area < 550:
            storeys = min(storeys, 2)

    # Same story for storey height: 3 storeys at 9 blocks each is 27 blocks of wall, which is
    # a tower whatever its footprint. The tall storeys in the library are double-height rooms
    # in single-storey builds, not stacked three deep.
    if not bias.get("storey_height"):
        storey_height = min(storey_height, {1: 9, 2: 7}.get(storeys, 6))

    # A window has to fit between its sill and the ceiling. Sampling sill and height against
    # storey height independently does not guarantee that -- sill 4 in a 5-block storey leaves
    # nowhere for the opening -- and `_place_openings` silently skipped the lot, so about 4%
    # of houses came out with no windows at all. Clamp here rather than there, so the spec
    # written to disk says what was actually built.
    window_h = int(bias.get("window_h") or pick(rng, P.WINDOW_H))
    sill = int(bias.get("sill") or pick(rng, P.WINDOW_SILL))
    window_h = max(1, min(window_h, storey_height - 2))
    sill = max(1, min(sill, storey_height - 1 - window_h))

    roof_type = bias.get("roof_type") or pick(rng, P.ROOF_TYPE)
    # Concrete, wool and terracotta have no stair variant. A style whose roof material cannot
    # make a slope has to be roofed flat, which is what the real builds using them do.
    pitch = float(bias.get("pitch") if bias.get("pitch") is not None
                  else pick(rng, P.ROOF_PITCH[roof_type]))

    # A hip roof rises from all four eaves, so its height is set by the short axis of the plan
    # and by nothing else -- and the plan was sampled without reference to the walls. A 21x19
    # single-storey house at pitch 1.0 draws a ten-block roof on a five-block wall, and because
    # a hip is built as concentric stair rings that reads as a stepped pyramid with a hut
    # under it, not a house. Measured over 3000 seeds, 87% of single-storey pitch-1.0 hips
    # overtopped their walls. A gable is not caught the same way: its rise is limited by
    # the span across the ridge and it reads as an A-frame when tall, which the library has.
    # Same fix as storeys and footprint above: keep the marginals, correct the pairing. Step
    # the pitch down until the rise fits on the walls, so a small plan with short walls gets
    # the 0.5 hip that the library's small hips actually have. `_one_roof` still caps the
    # built rise, but after this it rarely has to.
    if roof_type == "hip" and bias.get("pitch") is None:
        wall_height = storeys * storey_height
        hip_span = min(width, depth) // 2
        while pitch > 0.5 and pitch * hip_span > P.ROOF_RISE_MAX["hip"] * wall_height:
            pitch -= 0.5

    wall, roof_stairs, trim, foundation = _sample_style(rng, bias)
    accent = _pick_accent(rng, wall, roof_stairs)
    accent_stairs = (block_roles.stairs_for(block_roles.normalise(accent))
                     or P.ACCENT_STAIRS_FALLBACK)
    if not accent_stairs.startswith("minecraft:"):
        accent_stairs = "minecraft:" + accent_stairs
    ridge_axis = "x" if width >= depth else "z"
    if bias.get("ridge_axis"):
        ridge_axis = bias["ridge_axis"]

    return HouseSpec(
        seed=seed, width=width, depth=depth, plan=plan, notch=notch,
        storeys=storeys, storey_height=storey_height,
        roof_type=roof_type, pitch=pitch, ridge_axis=ridge_axis,
        overhang=int(pick(rng, P.OVERHANG)),
        wall=wall, roof_stairs=roof_stairs, trim=trim, foundation=foundation,
        glass=pick(rng, P.GLASS), door=pick(rng, P.DOOR),
        window_w=int(pick(rng, P.WINDOW_W)), window_h=window_h,
        sill=sill, spacing=int(pick(rng, P.WINDOW_SPACING)),
        door_offset=float(pick(rng, P.DOOR_OFFSET)),
        pad_depth=int(pick(rng, P.PAD_DEPTH)), pad_margin=int(pick(rng, P.PAD_MARGIN)),
        pad_surface=pick(rng, P.PAD_SURFACE), pad_subsoil=pick(rng, P.PAD_SUBSOIL),
        ground_cover=pick(rng, P.GROUND_COVER), path_block=pick(rng, P.PATH_BLOCK),
        tree_log=pick(rng, P.TREE_LOG), tree_leaves=pick(rng, P.TREE_LEAVES),
        accent=accent, accent_stairs=accent_stairs,
        gable=(_pick_contrasting(rng, P.GABLE, wall)
               if rng.random() < P.P_GABLE_DIFFERENT else wall),
        chimney_mat=_pick_contrasting(rng, P.CHIMNEY, wall),
        porch_w=int(pick(rng, P.PORCH_WIDTH)), porch_d=int(pick(rng, P.PORCH_DEPTH)),
        porch_post_mat=pick(rng, P.PORCH_POST_MAT),
        garage_w=int(pick(rng, P.GARAGE_WIDTH)), garage_door=pick(rng, P.GARAGE_DOOR),
        wing_w=int(pick(rng, P.WING_WIDTH)), wing_d=int(pick(rng, P.WING_DEPTH)),
        wing_offset=float(pick(rng, P.WING_OFFSET)),
        wing_storeys=int(pick(rng, P.WING_STOREYS)),
        plinth=(_pick_contrasting(rng, P.PLINTH, wall)
                if rng.random() < P.P_PLINTH else None),
        shutter=pick(rng, P.SHUTTER), bush_leaves=pick(rng, P.BUSH_LEAVES),
        flower=pick(rng, P.FLOWERS), path_edge_slab=pick(rng, P.PATH_EDGE_SLAB),
        porch_floor=pick(rng, P.PORCH_FLOOR), porch_post=pick(rng, P.PORCH_POST),
        driveway_w=int(pick(rng, P.DRIVEWAY_W)),
        driveway_side=rng.choice(("west", "east")),
        chimney=rng.random() < P.P_CHIMNEY,
        porch=rng.random() < P.P_PORCH,
        driveway=rng.random() < P.P_DRIVEWAY,
        walk_wide=rng.random() < 0.6,
        rake_trim=rng.random() < P.P_RAKE_TRIM,
        corner_posts=rng.random() < P.P_CORNER_POSTS,
        band_course=rng.random() < P.P_BAND_COURSE,
        porch_rail=rng.random() < P.P_PORCH_RAIL,
        garage=rng.random() < P.P_GARAGE,
        wing=rng.random() < P.P_FRONT_WING,
    )


# ── voxel canvas ─────────────────────────────────────────────────────────────

class Canvas:
    """A grid of palette indices in Sponge's own (height, length, width) order."""

    def __init__(self, height, length, width):
        self.grid = np.zeros((height, length, width), dtype=np.uint16)
        self.names = [AIR]
        self._index = {AIR: 0}
        self.height, self.length, self.width = height, length, width

    def idx(self, state: str) -> int:
        slot = self._index.get(state)
        if slot is None:
            slot = len(self.names)
            self._index[state] = slot
            self.names.append(state)
        return slot

    def put(self, y, z, x, state):
        if 0 <= y < self.height and 0 <= z < self.length and 0 <= x < self.width:
            self.grid[y, z, x] = self.idx(state)

    def column(self, y0, y1, z, x, state):
        for y in range(max(0, y0), min(self.height, y1 + 1)):
            self.put(y, z, x, state)

    def fill(self, y, z, x, state):
        """Like `put`, but only into air: for things that drape around a building."""
        if (0 <= y < self.height and 0 <= z < self.length and 0 <= x < self.width
                and self.grid[y, z, x] == 0):
            self.grid[y, z, x] = self.idx(state)

    def layer(self, y, mask, state):
        if 0 <= y < self.height:
            self.grid[y][mask] = self.idx(state)


# ── stage 1: footprint ───────────────────────────────────────────────────────

def _footprint(spec) -> np.ndarray:
    """The building's plan as a (depth, width) boolean mask."""
    plan = np.ones((spec.depth, spec.width), dtype=bool)
    if spec.plan == "L" and spec.notch:
        nx, nz, corner = spec.notch
        nx, nz = min(nx, spec.width - 4), min(nz, spec.depth - 4)
        if nx > 0 and nz > 0:
            zs = slice(0, nz) if corner in (0, 1) else slice(spec.depth - nz, spec.depth)
            xs = slice(0, nx) if corner in (0, 2) else slice(spec.width - nx, spec.width)
            plan[zs, xs] = False
    elif spec.plan == "T" and spec.notch:
        nx, nz, corner = spec.notch
        nx = min(max(3, nx // 2), (spec.width - 5) // 2)
        nz = min(nz, spec.depth - 4)
        if nx > 0 and nz > 0:
            zs = slice(0, nz) if corner % 2 == 0 else slice(spec.depth - nz, spec.depth)
            plan[zs, 0:nx] = False
            plan[zs, spec.width - nx:spec.width] = False
    if not plan.any():
        plan[:, :] = True
    return plan


def _with_wing(spec, plan):
    """Add a projecting front wing, and return (whole plan, just the wing).

    This is the shape that separates a house from a shed, and measuring the library made
    that unarguable: 83% of the real builds have at least one gable facing the front and 96%
    have roofs at two or more ridge levels, against 50% and a flat prism for the generator.
    A wing sticking out of the front elevation with its own gable end gives all three at
    once -- a second ridge, a gable pointed at the street, and relief in the wall, where the
    real builds average 0.25 of their wall area against our 0.09.

    The wing is grown into a taller array rather than carved out of the existing one, so it
    genuinely projects instead of eating the plan it is attached to.
    """
    if not spec.wing:
        return plan, None
    depth, width = plan.shape
    reach = spec.wing_d
    grown = np.zeros((depth + reach, width), dtype=bool)
    grown[:depth] = plan
    wing = np.zeros_like(grown)

    # As wide as asked for, centred on the offset the spec chose, and always landing on a
    # stretch of front wall that actually exists -- an L-plan's front is not the full width.
    front_x = [x for x in range(width) if plan[:, x].any()]
    if len(front_x) < spec.wing_w + 2:
        return plan, None
    span = front_x[-1] - front_x[0] + 1
    start = front_x[0] + int(round((span - spec.wing_w) * spec.wing_offset))
    start = max(front_x[0], min(start, front_x[-1] - spec.wing_w + 1))

    for x in range(start, start + spec.wing_w):
        south = np.nonzero(plan[:, x])[0]
        if len(south) == 0:
            continue
        edge = int(south.max())
        for dz in range(1, reach + 1):
            wing[edge + dz, x] = True
    if not wing.any():
        return plan, None
    return grown | wing, wing


def _corners(mask):
    """Plan cells where two walls meet -- where a house puts its quoins.

    A cell counts as a corner when it is missing a neighbour on two *perpendicular* sides,
    which finds the outside corners of a rectangle and of an L or T's wings alike, and does
    not fire along a straight run of wall.
    """
    padded = np.pad(mask, 1, constant_values=False)
    north, south = ~padded[:-2, 1:-1], ~padded[2:, 1:-1]
    west, east = ~padded[1:-1, :-2], ~padded[1:-1, 2:]
    return mask & (north | south) & (west | east)


def _ring(mask):
    """Perimeter cells of a plan: the wall line."""
    padded = np.pad(mask, 1, constant_values=False)
    inner = (padded[:-2, 1:-1] & padded[2:, 1:-1] &
             padded[1:-1, :-2] & padded[1:-1, 2:] & mask)
    return mask & ~inner


def _dilate(mask, amount):
    """Grow a mask outward *within its own array*. Only useful where there is room to grow.

    A rectangular plan is already solid to the array's edge, so dilating it in place is a
    no-op -- which is exactly the trap `_overhung` exists to avoid. Used for the plot, where
    the mask sits inside a much larger canvas and there is somewhere for it to go.
    """
    out = mask.copy()
    for _ in range(max(0, amount)):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _overhung(mask, amount):
    """The plan grown outward by `amount`, in a bigger array. Returns (grown, offset).

    The array has to grow with the mask. Dilating a rectangular footprint in place cannot
    add a single cell -- it is already True everywhere -- so a roof built that way has no
    overhang at all, and offsetting it by the overhang anyway (to account for growth that
    never happened) slides the whole roof off the walls. That was the off-centre roof: it
    hung 2 past the north and west walls and stopped 2 short of the south and east.

    `offset` is how far the grown mask's origin sits *before* the original's, so the caller
    places it at `base - offset` and the two stay concentric by construction.
    """
    amount = max(0, int(amount))
    if amount == 0:
        return mask.copy(), 0
    return _dilate(np.pad(mask, amount, constant_values=False), amount), amount


# ── stage 5: roof ────────────────────────────────────────────────────────────

FAR = 1 << 20


def _edge_distances(mask):
    """Per-cell distance to the plan's edge in each of the four compass directions.

    Measured *within the plan*, row by row and column by column, rather than from the
    bounding box's centre. That distinction is the whole reason roofs sit correctly on
    L- and T-plans: a centre-based model puts the ridge where the notch is, and the roof
    comes out visibly broken over the missing corner.
    """
    depth, width = mask.shape
    north = np.full(mask.shape, FAR)
    south = np.full(mask.shape, FAR)
    west = np.full(mask.shape, FAR)
    east = np.full(mask.shape, FAR)
    for x in range(width):
        cells = np.nonzero(mask[:, x])[0]
        if len(cells):
            north[cells, x] = cells - cells.min()
            south[cells, x] = cells.max() - cells
    for z in range(depth):
        cells = np.nonzero(mask[z, :])[0]
        if len(cells):
            west[z, cells] = cells - cells.min()
            east[z, cells] = cells.max() - cells
    return north, south, west, east


def _roof_height(spec, mask, eave, wall_height=None, parts=None):
    """Per-column roof height and the facing each slope's stairs should carry.

    A roof is the *upper envelope* of one gable per part of the plan, not one gable over the
    whole of it. That is what a cross gable is: two ridges at right angles, each sized to its
    own wing, meeting in a valley -- and taking the max of two height maps produces exactly
    that, with the valley falling out of the arithmetic rather than needing to be modelled.

    Sizing each part separately is the point. Running one gable over a plan that includes a
    projecting wing measures the wing's columns as furthest from the eave and pushes them
    *higher* than the main block, which is the opposite of what a builder does.
    """
    if not parts:
        parts = [(mask, spec.ridge_axis, eave)]
    height = np.full(mask.shape, float(eave))
    facing = np.full(mask.shape, SOUTH, dtype=object)
    for part_mask, axis, part_eave in parts:
        if not part_mask.any():
            continue
        h, f = _one_roof(spec, part_mask, part_eave, wall_height, axis)
        take = np.where(part_mask, h > height, False)
        height = np.where(take, h, height)
        facing = np.where(take, f, facing)
    return np.where(mask, height, float(eave)), facing


def _one_roof(spec, mask, eave, wall_height, ridge_axis):
    """One gable or hip over one part of the plan.

    Height rises with distance from the eave, capped at a single ridge height, so a wing
    narrower than the main block gets a lower ridge instead of a taller one -- which is
    what a builder would do, and keeps the rise proportionate to the span rather than to
    whichever axis happens to be longest.
    """
    if spec.roof_type == "flat":
        return (np.full(mask.shape, float(eave)),
                np.full(mask.shape, SOUTH, dtype=object))

    north, south, west, east = _edge_distances(mask)
    zs, xs = np.nonzero(mask)
    span_z = zs.max() - zs.min() + 1
    span_x = xs.max() - xs.min() + 1

    if spec.roof_type == "gable":
        if ridge_axis == "x":            # ridge runs along x, so height varies with z
            drop = np.minimum(north, south)
            cap_span = span_z
            facing = np.where(north <= south, SOUTH, NORTH)
        else:
            drop = np.minimum(west, east)
            cap_span = span_x
            facing = np.where(west <= east, EAST, WEST)
    else:                                 # hip: slopes away on all four sides
        drop = np.minimum(np.minimum(north, south), np.minimum(west, east))
        cap_span = min(span_z, span_x)
        nearest = drop
        facing = np.where(north == nearest, SOUTH,
                          np.where(south == nearest, NORTH,
                                   np.where(west == nearest, EAST, WEST)))

    # Keep the height fractional. A pitch of 0.5 puts every other course on a half-block, and
    # rounding it away is what turns a shallow roof into a flight of two-block terraces --
    # the "wedding cake" look. `_place_roof` puts a slab on the halves instead, which is how
    # the shallow roofs in the real library are actually built.
    ceiling = eave + spec.pitch * (cap_span // 2)
    # ...but never so tall that the house reads as a barn. Measured over the suburban subset,
    # a roof rises a median 0.57 of its wall height and 1.38 at the 90th percentile; a wide
    # single-storey plan at pitch 1.0 would otherwise put a nine-block roof on a five-block
    # wall, which is the one proportion no house in the library has. The cap is per roof
    # type -- a hip's rings must not outnumber its wall courses (see P.ROOF_RISE_MAX).
    if wall_height:
        ceiling = min(ceiling, eave + P.ROOF_RISE_MAX[spec.roof_type] * wall_height)
    exact = eave + np.where(mask, drop, 0) * spec.pitch
    exact = np.minimum(exact, ceiling)
    return np.where(mask, exact, float(eave)), facing


def _place_roof(canvas, spec, mask, eave, base_x, base_z, rng, parts=None):
    """A roof built the way the real suburban builds are: a stair shell over an open attic.

    Measured over the 33 suburban builds, a roof is 34% stairs and 7% slabs by block, and the
    median column under its surface is only 40% filled. The previous version filled the whole
    wedge with cubes and laid one course of stairs on top, which is why generated roofs read
    as solid terraced masses -- and why a palette dump of a generated house showed 504 cubes
    and no stairs at all when the pitch happened to be flat.

    So: one sloping course of stairs, nothing underneath it, and vertical faces closed only
    where they are actually exposed. `_closing_floor` is what makes that watertight without
    special-casing gable ends, hips or the inside corners of an L-plan.
    """
    height, facing = _roof_height(spec, mask, eave,
                                  wall_height=spec.storeys * spec.storey_height,
                                  parts=parts)
    stairs = spec.roof_stairs
    base = block_roles.base_for_stairs(spec.roof_stairs)
    solid = "minecraft:" + base if base else spec.wall
    roof_slab = block_roles.slab_for(base) if base else None
    wall_slab = block_roles.slab_for(block_roles.normalise(spec.wall))

    if spec.roof_type == "flat":
        return _place_flat_roof(canvas, spec, mask, eave, base_x, base_z, solid, roof_slab)

    # Each part closes down to its *own* eave: a single-storey wing against a two-storey
    # block has to stop at its own wall head, not at the main one three courses higher.
    eave_map = np.full(mask.shape, float(eave))
    for part_mask, _axis, part_eave in (parts or []):
        eave_map = np.where(part_mask, np.minimum(eave_map, part_eave), eave_map)
    floor = _closing_floor(mask, height, eave_map)
    ridge = float(np.max(height[mask])) if mask.any() else eave
    cap_ridge = roof_slab is not None and rng.random() < P.ROOF_RIDGE_SLAB
    # The overhanging ring reads as an eave when it is turned upside down: the stair's flat
    # face becomes the soffit and its step becomes the fascia. 14 of the 33 builds do this.
    underside = rng.random() < P.P_EAVE_UNDERSIDE
    perimeter = _ring(mask)
    edge = perimeter if underside else np.zeros_like(mask)
    # The light line round the outside of the roof -- fascia at the eaves, rake up the gable
    # ends. In the builds singled out for trim that pops, the light material sits on the
    # perimeter of every roof course (on-ring 0.85-1.00) and nowhere else in the roof.
    rake = perimeter if spec.rake_trim else np.zeros_like(mask)
    rake_slab = block_roles.slab_for(block_roles.normalise(spec.accent_stairs))

    depth, width = mask.shape
    for z in range(depth):
        for x in range(width):
            if not mask[z, x]:
                continue
            exact = float(height[z, x])
            top = int(np.floor(exact))
            half = (exact - top) >= 0.5
            # Close the vertical face below this column, but only down to whatever the
            # lowest neighbour leaves exposed -- so gable ends fill in and the attic does not.
            # This closure *is* the gable end, so it gets the gable material: the triangle
            # under a roof is a different colour from the wall below it in 44% of the library,
            # and it is one of the cheapest details that makes a house look designed.
            canvas.column(int(floor[z, x]), top - 1, base_z + z, base_x + x, spec.gable)

            trimmed = bool(rake[z, x])
            course_stairs = spec.accent_stairs if trimmed else stairs
            course_slab = (rake_slab if trimmed else roof_slab) or roof_slab

            at_ridge = exact >= ridge - 1e-6
            if at_ridge and cap_ridge and not half and course_slab:
                # A slab caps the ridge line: the two slopes meet under it, and a full block
                # there reads as a beam sitting proud of the roof.
                canvas.put(top, base_z + z, base_x + x, "{}[type=bottom]".format(course_slab))
            elif half and course_slab:
                # The half-course of a shallow pitch: a slab reads as a continuous slope
                # where a full stair would read as a terrace.
                canvas.put(top, base_z + z, base_x + x, spec.accent if trimmed else solid)
                canvas.put(top + 1, base_z + z, base_x + x,
                           "{}[type=bottom]".format(course_slab))
            else:
                half_state = "top" if (underside and edge[z, x] and top == eave) else "bottom"
                canvas.put(top, base_z + z, base_x + x,
                           "{}[facing={},half={},shape=straight]".format(
                               course_stairs, facing[z, x], half_state))

    if wall_slab:
        # A course of top slabs tucked under the eave: the trim board a house has where its
        # wall stops and its roof starts.
        for z, x in zip(*np.nonzero(_ring(mask))):
            canvas.put(eave - 1, base_z + z, base_x + x, "{}[type=top]".format(wall_slab))
    return height


def _closing_floor(mask, height, eave):
    """Per-column level to close a roof column down to: one above the lowest neighbour.

    A roof only needs to be solid where you could otherwise see into it. For a column in the
    middle of a slope every neighbour is within one course, so nothing is filled and the
    attic stays open; at a gable end, a hip corner or the inside of an L the neighbour is
    missing entirely, so the column fills all the way to the eave and closes the triangle.
    One rule, and it covers every plan shape the grammar can produce.
    """
    eave = np.asarray(eave, dtype=float)
    floor_level = np.floor(eave).astype(int) if eave.ndim else int(eave)
    tops = np.where(mask, np.floor(height).astype(int), floor_level - 1)
    padded = np.pad(tops, 1, constant_values=int(np.min(floor_level)) - 1)
    lowest = np.minimum.reduce([padded[:-2, 1:-1], padded[2:, 1:-1],
                                padded[1:-1, :-2], padded[1:-1, 2:]])
    return np.maximum(lowest + 1, floor_level)


def _place_flat_roof(canvas, spec, mask, eave, base_x, base_z, solid, roof_slab):
    """A flat roof: a deck, a slab parapet and a stair coping on the corners.

    Flat roofs are 14% of the suburban subset, and the old code gave them a cube deck and a
    cube parapet -- no stairs, no slabs, the one roof shape that came out entirely square.
    """
    for z, x in zip(*np.nonzero(mask)):
        canvas.put(eave, base_z + z, base_x + x, solid)
    parapet = spec.trim or spec.wall
    parapet_slab = block_roles.slab_for(block_roles.normalise(parapet)) or roof_slab
    depth, width = mask.shape
    for z, x in zip(*np.nonzero(_ring(mask))):
        canvas.put(eave + 1, base_z + z, base_x + x, parapet)
        # Coped with stairs turned outward -- the overhanging lip a parapet has, and the
        # reason a flat roof is not the one shape in the grammar made entirely of cubes.
        out = _outward_from(mask, z, x)
        if out:
            canvas.put(eave + 2, base_z + z, base_x + x,
                       "{}[facing={},half=top,shape=straight]".format(spec.roof_stairs, out))
        elif parapet_slab:
            canvas.put(eave + 2, base_z + z, base_x + x,
                       "{}[type=bottom]".format(parapet_slab))
    return np.where(mask, eave, eave).astype(float)


def _outward_from(mask, z, x):
    """Which way is out of the plan from this perimeter cell, or None if it is enclosed."""
    depth, width = mask.shape
    for face, (dz, dx) in OUTWARD.items():
        nz, nx = z + dz, x + dx
        if not (0 <= nz < depth and 0 <= nx < width) or not mask[nz, nx]:
            return face
    return None


# ── stage 6: openings ────────────────────────────────────────────────────────

def _wall_runs(mask, face):
    """Contiguous straight runs of wall on one elevation, as (fixed, [varying...]).

    Openings are placed per run rather than per bounding-box edge so that an L- or T-plan gets
    windows on the inside faces of its wings too, instead of only around a rectangle it does
    not have.
    """
    depth, width = mask.shape
    runs, current, key = [], [], None
    if face in (NORTH, SOUTH):
        for x in range(width):
            column = np.nonzero(mask[:, x])[0]
            if len(column) == 0:
                edge = None
            else:
                edge = int(column.min()) if face == NORTH else int(column.max())
            if edge != key and current:
                runs.append((key, current))
                current = []
            key = edge
            if edge is not None:
                current.append(x)
    else:
        for z in range(depth):
            row = np.nonzero(mask[z, :])[0]
            if len(row) == 0:
                edge = None
            else:
                edge = int(row.min()) if face == WEST else int(row.max())
            if edge != key and current:
                runs.append((key, current))
                current = []
            key = edge
            if edge is not None:
                current.append(z)
    if current:
        runs.append((key, current))
    return [(k, v) for k, v in runs if k is not None and len(v) >= 5]


def _place_chimney(canvas, spec, mask, roof_heights, base_x, base_z, base_y, overhang):
    """A 2x2 masonry stack rising a few blocks past the roof where it comes through.

    Sized and stopped against the roof rather than the grid: a 1x1 column run to the top of
    the canvas reads as a flagpole, not a chimney, and it inflates the build's height for
    everything downstream.

    Built of its own material rather than the house's. A chimney is masonry whatever the
    walls are made of -- bricks, stone bricks, andesite and cobblestone across the library --
    and one in the wall colour reads as a lump of the house rather than as a chimney.
    """
    cells = sorted(zip(*np.nonzero(_ring(mask))), key=lambda c: (c[0], c[1]))
    if not cells:
        return
    z, x = cells[len(cells) // 3]
    z = min(z, mask.shape[0] - 2)
    x = min(x, mask.shape[1] - 2)
    rz = min(z + overhang, roof_heights.shape[0] - 1)
    rx = min(x + overhang, roof_heights.shape[1] - 1)
    through = int(roof_heights[rz, rx])
    for dz in range(2):
        for dx in range(2):
            if not mask[min(z + dz, mask.shape[0] - 1), min(x + dx, mask.shape[1] - 1)]:
                continue
            canvas.column(base_y, through + 2, base_z + z + dz, base_x + x + dx,
                          spec.chimney_mat)


OUTWARD = {NORTH: (-1, 0), SOUTH: (1, 0), WEST: (0, -1), EAST: (0, 1)}
# The face an open trapdoor's panel covers is the one named by `facing`, so a shutter hung in
# the air outside a wall points back *at* the wall to sit flush against it.
INWARD = {NORTH: SOUTH, SOUTH: NORTH, WEST: EAST, EAST: WEST}


def _place_openings(canvas, spec, mask, base_y, base_x, base_z, eave, rng,
                    wing=None, wall_top=None, wing_top=None):
    """Windows on every elevation, shuttered, plus one door on the front.

    Windows are glass *panes* rather than blocks (panes outnumber solid glass two to one in
    the suburban builds) and are hung with trapdoor shutters, which is the commonest thing
    found immediately beside a window across the subset. Bare glass in a bare wall is the
    single cheapest tell that a generator made the house.
    """
    shutters = spec.shutter if rng.random() < P.P_SHUTTERS else None
    sill_slab = block_roles.slab_for(block_roles.normalise(spec.trim or spec.wall))

    # The ground floor of the front is laid out *first*, because a door and a garage are the
    # things a plan is arranged around and a window is what goes in the space left over.
    # Placing windows first and the door afterwards is how the entrance ended up with a
    # shutter cutting across it: the door replaced the wall, but the shutter beside the
    # window it landed on was already hanging there.
    runs = _wall_runs(mask, SOUTH)
    front = max(runs or [(None, [])], key=lambda r: len(r[1]))
    front_edge, front_cells = front
    placed_door = _place_door(canvas, spec, front_edge, front_cells, base_y, base_x, base_z)

    # The garage goes on the door's run if there is room beside the door, and otherwise on
    # any other run that faces the street -- most usefully the front of the wing, which is
    # the single-storey garage bay the real builds have. Confining it to the door's run
    # dropped four garages in ten: a front wing splits the main block's front into two
    # short runs, and one of those has to hold a door with a garage beside it.
    garage = None
    ordered = sorted(runs, key=lambda r: (r[0] != front_edge, -len(r[1])))
    for edge, cells in ordered:
        top = wing_top if (wing is not None and wing[edge, cells[0]]) else wall_top
        if top is not None and top - base_y < 5:
            continue                             # no wall tall enough for the opening
        garage = _place_garage(canvas, spec, edge, cells,
                               placed_door if edge == front_edge else None,
                               base_y, base_x, base_z, mask)
        if garage or not spec.garage:
            break

    # Cells on each front run's ground floor that already have something in them.
    taken = {}
    if placed_door:
        taken.setdefault(front_edge, set()).update(
            range(placed_door[1] - 2, placed_door[1] + 3))
    if garage:
        taken.setdefault(garage[2], set()).update(
            range(garage[0] - 2, garage[0] + garage[1] + 2))

    for face in (NORTH, SOUTH, WEST, EAST):
        dz, dx = OUTWARD[face]
        for edge, cells in _wall_runs(mask, face):
            # Spacing is capped against the length of the wall it is on. Sampled spacing runs
            # to 7, which on a 15-block elevation leaves room for a single window and a lot
            # of blank wall -- and once the door and the garage have taken the front's ground
            # floor, none at all. A wall gets at least three bays if three will fit.
            span = len(cells) - 4
            step = max(spec.window_w + 2, spec.spacing + spec.window_w)
            step = min(step, max(spec.window_w + 2, span // 3))
            # Bays are chosen once for the whole elevation and reused on every storey, so
            # the upper windows sit over the lower ones. Deciding per storey put them in
            # different places on each floor, which no house has and every eye notices.
            bays = list(range(cells[0] + 2, cells[-1] - spec.window_w, step))
            # Drop the occasional bay -- a real house has a blank one where a chimney or a
            # stair lands -- but drop the same bay on every floor.
            bays = [b for b in bays if rng.random() >= 0.15]
            for storey in range(spec.storeys):
                floor = base_y + storey * spec.storey_height
                sill = floor + max(1, spec.sill)
                head = min(sill + spec.window_h - 1, floor + spec.storey_height - 2)
                if head < sill:
                    continue
                ground_front = storey == 0 and face == SOUTH and edge in taken
                for pos in bays:
                    if ground_front and taken[edge].intersection(
                            range(pos - 1, pos + spec.window_w + 1)):
                        continue                 # the door or the garage is already here
                    for d in range(spec.window_w):
                        for y in range(sill, head + 1):
                            if face in (NORTH, SOUTH):
                                canvas.put(y, base_z + edge, base_x + pos + d, spec.glass)
                            else:
                                canvas.put(y, base_z + pos + d, base_x + edge, spec.glass)
                    # A sill under the opening, in the trim material: the line that stops a
                    # window reading as a hole punched in a flat wall.
                    if sill_slab and sill - 1 > floor:
                        for d in range(spec.window_w):
                            _, wz, wx = _face_cell(face, edge, pos + d, base_z, base_x)
                            canvas.put(sill - 1, wz, wx, "{}[type=top]".format(sill_slab))
                    if shutters:
                        for side in (-1, spec.window_w):
                            _, wz, wx = _face_cell(face, edge, pos + side, base_z, base_x)
                            for y in range(sill, head + 1):
                                canvas.put(y, wz + dz, wx + dx,
                                           "{}[facing={},half=bottom,open=true]".format(
                                               shutters, INWARD[face]))
    return placed_door, garage


def _place_door(canvas, spec, edge, cells, base_y, base_x, base_z):
    """The front door, in a surround of the accent material.

    A bare door in a matching wall under a matching porch is invisible from any distance --
    the entrance has to be the thing the eye lands on, and in the real builds it always is,
    because it is framed in the light trim.
    """
    if edge is None or len(cells) < 5:
        return None
    at = cells[0] + max(2, int(round(len(cells) * spec.door_offset)))
    at = min(at, cells[-1] - 2)
    for dx in (-1, 1):
        canvas.column(base_y + 1, base_y + 3, base_z + edge, base_x + at + dx, spec.accent)
    canvas.put(base_y + 3, base_z + edge, base_x + at, spec.accent)
    canvas.put(base_y + 1, base_z + edge, base_x + at,
               "{}[facing=south,half=lower,hinge=left,open=false]".format(spec.door))
    canvas.put(base_y + 2, base_z + edge, base_x + at,
               "{}[facing=south,half=upper,hinge=left,open=false]".format(spec.door))
    return (edge, at)


def _place_garage(canvas, spec, edge, cells, door, base_y, base_x, base_z, mask):
    """A garage: a bay cut into the house with the door set back inside its own reveal.

    The first version put a flat panel in the plane of the wall, which from any angle is a
    coloured square painted on the elevation. What makes the real ones read -- #88's double
    garage is the clearest -- is *depth*: the opening is a hole with jambs and a lintel, and
    the door sits a block behind the face of the wall, so the reveal throws a shadow line all
    the way round it. Behind the door is a room with a floor and a back wall, not a view
    straight through the house.

    Returns (x, width, edge) so the windows know not to use the same stretch of wall and the
    driveway knows exactly where to arrive.
    """
    if not spec.garage or edge is None:
        return None
    width = spec.garage_w
    if door is not None:
        # Put it at whichever end of the front is further from the door, with room to spare.
        door_x = door[1]
        left_room = door_x - 3 - cells[0]
        right_room = cells[-1] - (door_x + 3)
        if max(left_room, right_room) < width + 1:
            return None
        start = cells[-1] - width if right_room >= left_room else cells[0] + 1
    else:
        # A run of its own -- the wing's front -- so it sits in the middle of it.
        if len(cells) < width + 2:
            return None
        start = cells[0] + (len(cells) - width) // 2

    # Four courses, not three. A three-block opening is the same height as a window and
    # reads as one; a garage door is the tallest hole in the elevation.
    head = base_y + 5                      # the lintel course; the opening runs below it
    inner = edge - 1                       # the door plane, one block in from the wall face
    depth = min(4, max(2, spec.porch_d + 1))
    # The head is dark against a light house and light against a dark one, because a lintel
    # in the wall colour is a lintel nobody can see -- #88's double garage has a dark band
    # over both doors and that band is most of what makes them read as doors.
    lintel = (spec.accent if luminance(spec.wall) < 120
              else block_roles.base_for_stairs(spec.roof_stairs))
    lintel = "minecraft:" + lintel if lintel and ":" not in lintel else (lintel or spec.accent)

    for dx in range(width):
        x = start + dx
        # The hole through the wall, and the lintel over it.
        canvas.column(base_y + 1, head - 1, base_z + edge, base_x + x, AIR)
        canvas.put(head, base_z + edge, base_x + x, lintel)
        # The door, set back in the reveal and boarded: the real ones are vertical panels
        # with stiles between them, and one material over four blocks reads as a poster.
        panel = spec.garage_door
        state = ("{}[facing=south,half=bottom,open=true]".format(panel)
                 if panel.endswith("_trapdoor") else panel)
        for y in range(base_y + 1, head):
            canvas.put(y, base_z + inner, base_x + x, state)
        if dx % 2 == 1:
            canvas.column(base_y + 1, head - 1, base_z + inner, base_x + x, spec.accent)

    # Jambs: the accent returns into the reveal, so the opening has a visible thickness
    # rather than being a shape painted on the elevation.
    for dz in (edge, inner):
        for x in (start - 1, start + width):
            if 0 <= x < mask.shape[1]:
                canvas.column(base_y + 1, head, base_z + dz, base_x + x, spec.accent)

    # The room behind it: a floor, side walls and a back wall, so looking in shows a garage
    # rather than the inside of the far elevation.
    for dz in range(1, depth + 1):
        z = edge - dz
        if z < 0:
            break
        for dx in range(-1, width + 1):
            x = start + dx
            canvas.put(base_y, base_z + z, base_x + x, spec.foundation)
            if dx in (-1, width):
                canvas.column(base_y + 1, head, base_z + z, base_x + x, spec.wall)
        if dz == depth:
            for dx in range(width):
                canvas.column(base_y + 1, head, base_z + z, base_x + start + dx, spec.wall)
    return (start, width, edge)


def _face_cell(face, edge, along, base_z, base_x):
    """(y, z, x) of a wall cell on one elevation; y is the caller's business."""
    if face in (NORTH, SOUTH):
        return None, base_z + edge, base_x + along
    return None, base_z + along, base_x + edge


# ── stage 8: the plot ────────────────────────────────────────────────────────

def _place_tree(canvas, rng, spec, y, z, x):
    """A tree the size of the ones on the real plots, not a sapling.

    Trunk and canopy are sized from P.TREE_TRUNK / P.TREE_CANOPY: the real trees stand as
    tall as the house and carry twice the leaves the old fixed 4-6 trunk under a radius-2
    ball did. The canopy is laid with `fill`, into air only, so a tree can stand close to
    the house and drape round its eave the way #83's does, instead of either overwriting
    the roof or being kept three blocks off the wall.
    """
    trunk = int(pick(rng, P.TREE_TRUNK))
    radius = int(pick(rng, P.TREE_CANOPY))
    for dy in range(trunk):
        canvas.put(y + dy, z, x, spec.tree_log)
    top = y + trunk
    # Widest a course below the top of the trunk, a step narrower above and below it, and a
    # tuft on top: a rounded crown rather than a cube on a stick.
    for dy in range(-radius, 2):
        r = radius if -radius < dy < 0 else max(1, radius - 1)
        if dy == 1:
            r = 1
        for dz in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dz) == r and abs(dx) == r and r > 1:
                    continue
                if abs(dz) + abs(dx) > r + max(1, r - 1):
                    continue
                canvas.fill(top + dy, z + dz, x + dx, spec.tree_leaves)


def _place_bush(canvas, rng, spec, y, z, x):
    """A shrub: a clump, not the single floating leaf block the old plot scattered."""
    canvas.put(y, z, x, spec.bush_leaves)
    if rng.random() < 0.45:
        canvas.put(y + 1, z, x, spec.bush_leaves)
    for dz, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        if rng.random() < 0.35:
            canvas.put(y, z + dz, x + dx, spec.bush_leaves)


def _paved(canvas, spec, y, cells, edge_slab=None):
    """Lay a run of paving, optionally kerbed with slabs along its edges."""
    cells = [c for c in cells]
    paved = set(cells)
    for z, x in cells:
        canvas.put(y, z, x, spec.path_block)
    if edge_slab:
        for z, x in cells:
            for dz, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                if (z + dz, x + dx) not in paved:
                    canvas.put(y, z + dz, x + dx, "{}[type=bottom]".format(edge_slab))


def _place_porch(canvas, rng, spec, base_y, pz, px, storey_height, half_w=None):
    """A front porch: a decked bay standing on posts under a shed roof of its own.

    The earlier version hung a single course of upside-down stairs off the wall above the
    door, which reads as a shelf rather than as somewhere you stand. Looking at how the real
    porches are built -- #101, #103, #24, #17 -- a porch is a piece of *building*: a floor
    you step onto, posts at its outer corners, and a roof that slopes away from the wall.

    A shed roof of its own rather than an extension of the main one. Folding the porch bay
    into the main roof's plan is what #101 and #103 do, but it moves the ridge off the centre
    of the house and raises the whole roof to cover the extra span; a separate lower roof is
    what the smaller-porched builds do and it leaves the massing alone.
    """
    half_w = max(1, spec.porch_w // 2 if half_w is None else half_w)
    depth = spec.porch_d
    deck = [(pz + dz, px + dx)
            for dz in range(1, depth + 1) for dx in range(-half_w, half_w + 1)]
    floor_slab = spec.porch_floor
    for z, x in deck:
        canvas.put(base_y - 1, z, x, floor_slab + "[type=top]")

    # The roof: a slab deck one course below the storey top, fringed with stairs that fall
    # away from the house, so it has a front edge instead of stopping in mid-air.
    top = base_y + storey_height - 1
    fringe = block_roles.slab_for(block_roles.normalise(spec.accent_stairs))
    for dx in range(-half_w - 1, half_w + 2):
        for dz in range(1, depth + 1):
            if dz == depth:
                canvas.put(top, pz + dz, px + dx,
                           "{}[facing=north,half=bottom,shape=straight]".format(
                               spec.accent_stairs))
            elif fringe:
                canvas.put(top, pz + dz, px + dx, "{}[type=bottom]".format(fringe))
            else:
                canvas.put(top, pz + dz, px + dx, spec.accent)
    # ...and returned along the two sides, so it is a roof rather than an awning.
    for dx in (-half_w - 1, half_w + 1):
        for dz in range(1, depth):
            canvas.put(top, pz + dz, px + dx,
                       "{}[facing={},half=bottom,shape=straight]".format(
                           spec.accent_stairs, EAST if dx < 0 else WEST))

    # Posts at the outer corners, and a rail between them with the entry left open.
    for dx in (-half_w, half_w):
        canvas.column(base_y, top - 1, pz + depth, px + dx, spec.porch_post_mat)
    if spec.porch_rail:
        rail = spec.porch_post if "fence" in spec.porch_post else spec.porch_post
        for dx in range(-half_w, half_w + 1):
            if abs(dx) <= 0:
                continue                       # leave the way in
            canvas.put(base_y, pz + depth, px + dx, rail)
        for dz in range(1, depth):
            for dx in (-half_w, half_w):
                canvas.put(base_y, pz + dz, px + dx, rail)


def _place_plot(canvas, rng, spec, mask, base_x, base_z, base_y, door, garage=None):
    """Grass pad, walkway, driveway, porch, planting and trees."""
    plot = np.ones((canvas.length, canvas.width), dtype=bool)
    for y in range(0, base_y - 1):
        canvas.layer(y, plot, spec.pad_subsoil)
    canvas.layer(max(0, base_y - 1), plot, spec.pad_surface)

    house = np.zeros_like(plot)
    house[base_z:base_z + mask.shape[0], base_x:base_x + mask.shape[1]] = mask
    paved = np.zeros_like(plot)

    # ── the walkway, from the front door to the edge of the plot ─────────────
    if door:
        edge, at = door
        pz, px = base_z + edge, base_x + at
        walk = [(z, x) for z in range(pz + 1, canvas.length)
                for x in (px, px - 1 if spec.walk_wide else px)]
        _paved(canvas, spec, base_y - 1, walk, spec.path_edge_slab)
        for z, x in walk:
            paved[z, x] = True
        if spec.porch:
            # Narrowed to fit when the door is close to the wing: a porch roof that runs
            # into the side of the wing had its stairs and posts overwrite the wing's wall.
            half_w = max(1, spec.porch_w // 2)
            while half_w > 1 and any(
                    house[pz + dz, px + dx]
                    for dz in range(1, spec.porch_d + 1)
                    for dx in (-half_w - 1, half_w + 1)
                    if 0 <= pz + dz < house.shape[0] and 0 <= px + dx < house.shape[1]):
                half_w -= 1
            _place_porch(canvas, rng, spec, base_y, pz, px, spec.storey_height, half_w)

    # ── the driveway ─────────────────────────────────────────────────────────
    # It runs to the garage when there is one -- a driveway that stops at a blank wall while
    # the garage opens onto grass is worse than no driveway at all -- and otherwise down
    # whichever side of the house was chosen for it.
    if spec.driveway or garage:
        hz, hx = np.nonzero(house)
        if garage:
            # From the garage's own wall, one block wider than the opening so the apron
            # reads as an apron. Starting it at the house's south corner instead is what put
            # the driveway somewhere the garage does not open onto: on an L or a T the front
            # wall the garage is in is not the southernmost part of the plan.
            x0, width = base_x + garage[0] - 1, garage[1] + 2
            z_from = base_z + garage[2] + 1
        else:
            width = spec.driveway_w
            west_side = spec.driveway_side == "west"
            x0 = (hx.min() - 1 - width) if west_side else (hx.max() + 2)
            # Starts level with the front half of the house, not its back corner: a driveway
            # runs to the street, and paving the whole side reads as a car park.
            z_from = (hz.min() + hz.max()) // 2
        x0 = max(1, min(x0, canvas.width - width - 1))
        drive = [(z, x) for z in range(z_from, canvas.length)
                 for x in range(x0, x0 + width)]
        drive = [(z, x) for z, x in drive if not house[z, x]]
        _paved(canvas, spec, base_y - 1, drive, spec.path_edge_slab)
        for z, x in drive:
            paved[z, x] = True

    # The real plots are densely planted -- grass, flowers and shrubs over most of the open
    # ground, not a token sprig. Scenery is a large share of a collected build's volume, so
    # a sparse plot is off-distribution in the same way no plot at all would be.
    # Planting keeps off the paving *and* off the block beside it -- a tree whose trunk is
    # clear of the driveway still puts its canopy over the garage door.
    open_ground = ~_dilate(house, 1) & ~_dilate(paved, 1)
    cells = list(zip(*np.nonzero(open_ground)))
    rng.shuffle(cells)
    # Mixed, not a monoculture: one flower species carpeting an entire lawn was the old
    # plot's most conspicuous tell.
    for z, x in cells[:max(10, int(len(cells) * 0.5))]:
        canvas.put(base_y, z, x, spec.ground_cover if rng.random() < 0.94 else spec.flower)

    beds = [(z, x) for z, x in cells if _dilate(house, 2)[z, x]]
    rng.shuffle(beds)
    for z, x in beds[:max(3, len(beds) // 6)]:
        _place_bush(canvas, rng, spec, base_y, z, x)

    # Anywhere on the open ground that is a trunk's width off the house and the paving. The
    # old rule also kept trunks three blocks in from every edge of the plot, which on the
    # three- and four-block margins the pad table mostly deals leaves no legal spot at all:
    # one plot in nine had no tree, and the rest had them huddled at the back. The real
    # plots stand their trees on the boundary -- #2's are a row along the fence line -- and
    # a canopy that runs past the edge of the pad is simply clipped, as it is there.
    clear_of_paving = ~_dilate(paved, 2)
    spots = [(z, x) for z, x in cells
             if 1 <= z < canvas.length - 1 and 1 <= x < canvas.width - 1
             and not _dilate(house, 2)[z, x] and clear_of_paving[z, x]]
    rng.shuffle(spots)
    chosen = []
    want = rng.randrange(2, 5)
    for z, x in spots:
        if all(abs(z - cz) + abs(x - cx) > 6 for cz, cx in chosen):
            chosen.append((z, x))
        if len(chosen) >= want:
            break
    for z, x in chosen:
        _place_tree(canvas, rng, spec, base_y, z, x)
    spec.trees = [[int(z), int(x)] for z, x in chosen]


# ── assembly ─────────────────────────────────────────────────────────────────

def build_voxels(spec: HouseSpec):
    """Run every grammar stage and return (grid, names) ready for `schem_write`."""
    rng = random.Random(spec.seed ^ 0x5EED)
    mask, wing = _with_wing(spec, _footprint(spec))
    ring = _ring(mask)

    # Sized from the mask, not from the spec: a front wing makes the plan deeper than the
    # depth that was sampled, and a plot sized from the sampled depth would clip it.
    margin = spec.pad_margin
    plan_d, plan_w = mask.shape
    plot_w = plan_w + 2 * margin
    plot_d = plan_d + 2 * margin
    base_x = base_z = margin

    base_y = spec.pad_depth
    wall_top = base_y + spec.storeys * spec.storey_height
    eave = wall_top + 1
    # A wing is usually a storey lower than the block it is attached to -- the single-storey
    # garage or living-room bay against a two-storey house. That step in the wall head is
    # where the real builds' 0.25 of wall-band relief comes from, and a full-height wing
    # gives none of it: it projects horizontally but the elevation stays one flat height.
    wing_storeys = min(spec.storeys, spec.wing_storeys) if wing is not None else spec.storeys
    wing_top = base_y + wing_storeys * spec.storey_height
    wing_eave = wing_top + 1
    roof_rise = 0
    if spec.roof_type != "flat":
        span = (spec.depth if spec.ridge_axis == "x" else spec.width)
        if spec.roof_type == "hip":
            span = min(spec.width, spec.depth)
        roof_rise = int(round(span / 2.0 * spec.pitch))
    height = eave + roof_rise + 4

    canvas = Canvas(height, plot_d, plot_w)

    # foundation course, one below the ground floor
    for z, x in zip(*np.nonzero(mask)):
        canvas.put(base_y - 1, base_z + z, base_x + x, spec.foundation)

    in_wing = wing if wing is not None else np.zeros_like(mask)
    for storey in range(spec.storeys):
        floor = base_y + storey * spec.storey_height
        # The wing stops when it runs out of storeys; the main block carries on above it.
        raising = ~in_wing if storey >= wing_storeys else np.ones_like(mask)
        for z, x in zip(*np.nonzero(mask & raising)):
            canvas.put(floor, base_z + z, base_x + x, spec.foundation)
        for z, x in zip(*np.nonzero(ring & raising)):
            canvas.column(floor + 1, floor + spec.storey_height - 1,
                          base_z + z, base_x + x, spec.wall)
        # The plinth: the ground storey's bottom course in something heavier, which 60 of
        # the 106 real builds have. Without it a house looks dropped onto the lawn rather
        # than standing on anything.
        if storey == 0 and spec.plinth:
            for z, x in zip(*np.nonzero(ring)):
                canvas.put(floor + 1, base_z + z, base_x + x, spec.plinth)
        # The band course at each storey division. In the real builds this is where the light
        # material goes -- a horizontal line all the way round, at 10-35% of the layer.
        band = spec.accent if spec.band_course else spec.trim
        if band:
            for z, x in zip(*np.nonzero(ring & raising)):
                canvas.put(floor + spec.storey_height - 1, base_z + z, base_x + x, band)

    # Where the wing stops, the main block's wall carries on past it -- so the wing needs a
    # wall closing its own head, and the main block needs one where the two meet.
    if wing is not None and wing_storeys < spec.storeys:
        for z, x in zip(*np.nonzero(in_wing)):
            canvas.put(wing_top, base_z + z, base_x + x, spec.foundation)

    # Quoins: the light corner posts that run the full height of the wall. Measured on the
    # perimeter ring at on-ring 0.90-1.00 in every build called out for its trim.
    if spec.corner_posts:
        for z, x in zip(*np.nonzero(_corners(mask))):
            top = wing_top if in_wing[z, x] else wall_top
            canvas.column(base_y, top - 1, base_z + z, base_x + x, spec.accent)

    for z, x in zip(*np.nonzero(mask & ~(in_wing if wing_storeys < spec.storeys
                                        else np.zeros_like(mask)))):
        canvas.put(wall_top, base_z + z, base_x + x, spec.foundation)

    # An L- or T-plan's notch is only a few blocks wide, so growing the roof far enough
    # swallows it and the house gets a rectangular roof over a non-rectangular building --
    # which then reads as concentric terraces rather than as two wings.
    eaves = spec.overhang if spec.plan == "rect" else min(spec.overhang, 1)
    roof_mask, roof_off = _overhung(mask, eaves)
    # The wing gets its own gable, turned across the main one. Each part is grown by the same
    # overhang so the two roofs meet properly at the valley instead of one stopping short.
    parts = None
    if wing is not None:
        main_part, _ = _overhung(mask & ~wing, eaves)
        wing_part, _ = _overhung(wing, eaves)
        across = "z" if spec.ridge_axis == "x" else "x"
        parts = [(main_part, spec.ridge_axis, eave), (wing_part, across, wing_eave)]
    roof_heights = _place_roof(canvas, spec, roof_mask, eave,
                               base_x - roof_off, base_z - roof_off, rng, parts=parts)

    door, garage = _place_openings(canvas, spec, mask, base_y, base_x, base_z, eave, rng,
                                   wing=wing, wall_top=wall_top, wing_top=wing_top)

    if spec.chimney:
        _place_chimney(canvas, spec, mask, roof_heights, base_x, base_z, base_y, eaves)

    _place_plot(canvas, rng, spec, mask, base_x, base_z, base_y, door, garage)

    return canvas.grid, canvas.names


def generate(seed: int, bias: dict | None = None) -> House:
    """Sample a house and build it."""
    spec = sample_spec(seed, bias)
    grid, names = build_voxels(spec)
    return House(grid=grid, names=names, spec=spec)


def write(house: House, dest_path) -> dict:
    """Write a generated house, stamping its seed and parameters into `Metadata`."""
    spec = asdict(house.spec)
    info = schem_write.write_schem(
        house.grid, house.names, dest_path,
        metadata={"Generator": "house_grammar", "Seed": house.spec.seed,
                  "Params": spec})
    info["spec"] = spec
    return info
