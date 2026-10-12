"""Frozen sampling parameters for the house generator, distilled from the real library.

PARKED: see docs/mesh-to-schem-plan.md v4

`house_stats` measures; this decides what the generator does with the measurements. The two
are deliberately separate files, and this one is checked in and edited by hand.

The reason is control. `data/build_stats.json` is machine output -- rerun the analyser and it
changes, silently, and with it every house the generator would produce. A training run has to
be reproducible and attributable, so the numbers the generator actually samples from are
frozen here, where they can be read, argued with, and corrected. Every value carries the
statistic it came from, so the distance between "what the library does" and "what we chose to
imitate" stays visible.

Two places where the frozen values deliberately disagree with the raw measurements:

*   **Contaminated palettes are dropped.** The analyser reads a build's wall material off its
    perimeter, and on builds captured with a deep landscape block underneath, that occasionally
    catches scenery -- `dead_bubble_coral_wall_fan` and `infested_cobblestone` both turned up
    as "trim". Those styles are filtered out rather than reproduced.
*   **The long tails are clipped.** Real footprints run to 92 blocks and sills to 19, but those
    come from sprawling multi-wing builds and gable-end openings, not from anything a
    single-house grammar should imitate. The ranges here are p10-p90.

Run `python gen_houses.py --check-params` to see where these have drifted from a fresh
analysis run.
"""
from __future__ import annotations

__all__ = ["FOOTPRINT_W", "FOOTPRINT_D", "ASPECT", "PLAN_SHAPE", "STOREY_COUNT",
           "STOREY_HEIGHT", "ROOF_TYPE", "ROOF_PITCH", "OVERHANG", "WINDOW_W", "WINDOW_H",
           "NOTCH_SHARE", "TREE_TRUNK", "TREE_CANOPY",
           "WINDOW_SILL", "WINDOW_SPACING", "DOOR_OFFSET", "STYLES", "PAD_DEPTH",
           "PAD_MARGIN", "PAD_SURFACE", "PAD_SUBSOIL", "GROUND_COVER", "TREE_LOG",
           "TREE_LEAVES", "PATH_BLOCK", "GLASS", "DOOR", "P_OBSERVED_STYLE", "P_CHIMNEY",
           "P_PORCH", "SIZE_TERCILES",
           # suburban-subset additions: roof construction, shutters, paving, planting
           "ROOF_RIDGE_SLAB", "P_EAVE_UNDERSIDE", "ROOF_FILL_TARGET", "ROOF_RISE_MAX",
           "SHUTTER", "P_SHUTTERS", "PATH_EDGE_SLAB", "PORCH_FLOOR", "PORCH_POST",
           "BUSH_LEAVES", "FLOWERS", "P_DRIVEWAY", "DRIVEWAY_W", "SUBURBAN_IDS",
           # colour: contrast constraints and the light trim that makes an edge pop
           "MIN_WALL_ROOF_CONTRAST", "MIN_WALL_SATURATION", "STRONG_CONTRAST",
           "P_CHROMATIC_WALL", "STYLE_TRIES", "ACCENT",
           "ACCENT_STAIRS_FALLBACK", "MIN_ACCENT_CONTRAST", "P_RAKE_TRIM",
           "P_CORNER_POSTS", "P_BAND_COURSE",
           # gable, chimney, porch and garage
           "GABLE", "P_GABLE_DIFFERENT", "CHIMNEY", "PORCH_WIDTH", "PORCH_DEPTH",
           "PORCH_POST_MAT", "P_PORCH_RAIL", "P_GARAGE", "GARAGE_WIDTH", "GARAGE_DOOR",
           # massing: the projecting front wing, and the plinth course
           "P_FRONT_WING", "WING_WIDTH", "WING_DEPTH", "WING_OFFSET", "WING_STOREYS",
           "PLINTH", "P_PLINTH"]

# ── Footprint ────────────────────────────────────────────────────────────────
# Measured: width p25 18.5 / med 26 / p75 38.5, depth p25 19.5 / med 27 / p75 37. The 5-block
# histogram peaks in the 15-30 band, which is where an ordinary suburban house lives; the
# 60-90 tail is sprawling estate builds and is clipped off.
FOOTPRINT_W = {15: 3, 17: 6, 19: 9, 21: 11, 23: 12, 25: 12, 27: 11, 29: 9,
               31: 8, 33: 6, 35: 5, 37: 4, 39: 3, 41: 2}
FOOTPRINT_D = {13: 3, 15: 6, 17: 9, 19: 11, 21: 12, 23: 11, 25: 10, 27: 8,
               29: 7, 31: 5, 33: 4, 35: 3}

# Measured: aspect p10 1.06 / med 1.30 / p90 1.83. Depth is sampled as width / aspect, so
# proportions stay plausible instead of the two axes drifting independently.
ASPECT = {1.0: 8, 1.1: 14, 1.2: 16, 1.3: 15, 1.4: 12, 1.5: 9, 1.6: 6, 1.8: 4, 2.0: 2}

# Measured: fill_ratio p10 0.49 / p25 0.60 / med 0.72 / p75 0.79 / p90 0.86, rectangularity
# med 0.6 (the suburban subset reads the same: 0.54 / 0.61 / 0.73 / 0.81 / 0.85). A fill ratio
# near 1.0 is a plain rectangle; 0.6-0.8 is an L or a T. Fewer than one build in ten is
# above 0.85, so a plain rectangle is the exception, not half the library -- the old 46%
# rect weight, checked against the generator's own output over 1500 seeds, gave a median
# fill of 0.90 and a p10 of 0.81: boxier than nine tenths of the real builds. With the notch
# share below, this mix reproduces the measured curve (0.65 / 0.70 / 0.75 / 0.81 / 0.95).
PLAN_SHAPE = {"rect": 20, "L": 55, "T": 25}
# How much of each axis the L or T notch takes out. The plan's fill ratio is set almost
# entirely by this: a notch of a third of each axis removes a ninth of the box (fill 0.89),
# which is what the old 0.28-0.42 range produced for every L and T alike. Real L-plans are
# two arms of comparable size, so the notch is nearer half of each axis -- 0.40-0.65 puts an
# L at fill 0.6-0.8, which is where the measurement says it lives.
NOTCH_SHARE = (0.40, 0.65)

# ── Massing ──────────────────────────────────────────────────────────────────
# Re-measured over the 33 suburban builds only (see SUBURBAN_IDS): 2 storeys 19, 1 storey 9,
# 3 storeys 4. The old marginal came from all 106 builds, where sheds, shops and single-room
# cabins pull the average down to a coin flip -- a suburban house is a two-storey house.
STOREY_COUNT = {1: 28, 2: 59, 3: 13}

# Measured exactly: 4:5  5:19  6:19  7:8  8:4  9:6  10:2  11:1. The 9-11 tail is double-height
# rooms in a handful of builds; kept but rare.
STOREY_HEIGHT = {4: 5, 5: 19, 6: 19, 7: 8, 8: 4, 9: 3}

# ── Roof ─────────────────────────────────────────────────────────────────────
# Measured over the suburban subset: gable 19, hip 6, flat 5, complex 2. "complex" is not a
# thing the grammar builds directly, so its weight goes to gable and hip -- multi-wing plans
# produce genuinely complex rooflines on their own.
#
# Flat is set to zero rather than to its measured 14%. That IS a deliberate departure from
# the data, on review of the output: a flat roof on a rectangular plan comes out as a cube
# with a lid, which is the one thing in the batch that stops reading as a suburban house at
# all. The five real flat-roofed builds are modernist designs whose interest is in massing
# the grammar does not model. Still reachable with `--roof flat`.
ROOF_TYPE = {"gable": 72, "hip": 28, "flat": 0}

# Measured per type. 1.0 (full stair per block) and 0.5 (stair + slab) are the two roofs
# Minecraft builders actually make; 1.5 and 2.0 are steep A-frames and stay rare.
# Measured over the suburban subset: 1.0 on 16 builds, 0.5 on 9, 1.5 on 2, and 2.0 on none
# at all. The 2.0 weight came from the all-builds table and produced tent-shaped A-frames
# whose roof is most of the elevation -- there is no suburban house in the library like that.
ROOF_PITCH = {
    "gable": {1.0: 16, 0.5: 9, 1.5: 2},
    "hip": {0.5: 12, 1.0: 4},
    "flat": {0.0: 1},
}
# One block, nearly always. Two different automated measurements of this disagreed with each
# other (19 of 33 builds project past their core; only 25% of individual eave sides do) and
# both disagree with what the builds plainly look like -- the bbox comparisons are confounded
# by porches, garages and wings, which widen the thing being compared against. Reviewed by
# eye instead: a roof that stops flush with its wall is the single clearest tell that a house
# was generated, and 2+ blocks never appears. Flush is kept as a small minority.
OVERHANG = {0: 5, 1: 25}

# ── Roof construction ────────────────────────────────────────────────────────
# What a suburban roof is actually made of, block by block. Measured over the 33 builds:
# stairs appear in 32 of them and are a median 34% of every block in the roof band; slabs
# appear in 32 and are a median 7%; the median roof column is 40% filled, i.e. roofs are a
# SHELL over an open attic, not a solid wedge. The top course is slab-dominant in 22 of 33 --
# a slab ridge cap -- and 14 of 33 use upside-down (half=top) stairs, which is the eave.
#
# The generator used to fill the whole wedge with cubes and cap it with one stair course.
# These three numbers are what turn that back into something a builder would recognise.
ROOF_RIDGE_SLAB = 0.67        # ridge capped with slabs rather than a full block
P_EAVE_UNDERSIDE = 0.45       # upside-down stair course along the overhang
ROOF_FILL_TARGET = 0.40       # for reference: measured median column fill of a real roof
# A roof rises a median 0.57 of the wall height it sits on across the suburban subset, 1.00
# at p75 and 1.38 at p90. Capped at p90 rather than the median: tall roofs are real, a roof
# twice its own walls is not.
#
# Per roof type, because the same ratio does not look the same on both. A gable at 1.4 is a
# steep A-frame, which the library has. A hip is built as concentric stair rings shrinking
# toward the centre, and once there are more rings than there are courses of wall the house
# reads as a stepped pyramid with a hut underneath -- the "ziggurat" in review. The six
# suburban hip roofs rise roughly 0.35, 0.43, 0.56, 0.71, 1.4 and 1.6 of their walls; the two
# above 1.0 are the 17x13 one-chunk cottages, whose whole cap is six rings tall. On a 20+
# block plan that proportion is a dozen rings, so hips are held at 1.0: the roof may match
# the walls, never overtop them. `sample_spec` also lowers the pitch to stay under this
# before anything is built, so the cap is the backstop rather than the mechanism.
ROOF_RISE_MAX = {"gable": 1.4, "hip": 1.0, "flat": 0.0}

# ── Openings ─────────────────────────────────────────────────────────────────
# Re-measured by counting contiguous runs of glass on the suburban builds' elevations:
# 1 wide 1996 runs, 2 wide 388, 3 wide 144, 4 wide 65, 5+ 40. Suburban windows are narrow
# and repeated, not the picture windows the all-builds marginal suggested.
WINDOW_W = {1: 46, 2: 27, 3: 15, 4: 8, 5: 4}
WINDOW_H = {1: 4, 2: 15, 3: 10, 4: 4}
WINDOW_SILL = {1: 10, 2: 14, 3: 7, 4: 2}
WINDOW_SPACING = {2: 8, 3: 13, 4: 11, 5: 7, 6: 5, 7: 3}
DOOR_OFFSET = {0.3: 3, 0.4: 5, 0.5: 8, 0.6: 5, 0.7: 3}

# Shutters. The commonest thing sitting immediately beside a window in the real builds is a
# trapdoor stood on its side -- dark oak 188, iron 156, birch 124, spruce 34, oak 19, across
# ten builds. Wall banners are the other recurring one. This is the detail that most cheaply
# reads as "someone built this": bare glass in a bare wall is what a generator produces.
SHUTTER = {"minecraft:dark_oak_trapdoor": 8, "minecraft:birch_trapdoor": 6,
           "minecraft:spruce_trapdoor": 5, "minecraft:oak_trapdoor": 4,
           "minecraft:iron_trapdoor": 3, "minecraft:crimson_trapdoor": 1}
P_SHUTTERS = 0.42             # 10 of the 33 builds hang something beside their windows

# ── Materials ────────────────────────────────────────────────────────────────
# (wall, roof_stairs, trim, foundation) -- 79 combinations a human actually built, taken
# straight from the analysis with contaminated entries filtered out. The generator samples
# whole tuples most of the time, which is what stops it producing lime walls under a nether
# brick roof. Delete a line here and it stops appearing in output.
STYLES = [
    ('minecraft:bricks', 'minecraft:stone_brick_stairs', 'minecraft:quartz_pillar', 'minecraft:bricks'),
    ('minecraft:spruce_planks', 'minecraft:stone_brick_stairs', 'minecraft:quartz_pillar', 'minecraft:quartz_pillar'),
    ('minecraft:quartz_pillar', 'minecraft:cobblestone_stairs', 'minecraft:green_terracotta', 'minecraft:chiseled_quartz_block'),
    ('minecraft:light_gray_terracotta', 'minecraft:dark_oak_stairs', 'minecraft:oak_planks', 'minecraft:dark_oak_planks'),
    ('minecraft:gray_terracotta', 'minecraft:jungle_stairs', 'minecraft:jungle_planks', 'minecraft:gray_terracotta'),
    ('minecraft:bricks', 'minecraft:stone_brick_stairs', 'minecraft:quartz_block', 'minecraft:bricks'),
    ('minecraft:white_wool', 'minecraft:spruce_stairs', 'minecraft:smooth_stone', 'minecraft:black_wool'),
    ('minecraft:chiseled_quartz_block', 'minecraft:stone_brick_stairs', 'minecraft:cut_sandstone', 'minecraft:quartz_block'),
    ('minecraft:white_concrete_powder', 'minecraft:granite_stairs', None, 'minecraft:white_concrete_powder'),
    ('minecraft:quartz_block', 'minecraft:andesite_stairs', 'minecraft:white_wool', 'minecraft:smooth_sandstone'),
    ('minecraft:white_concrete_powder', 'minecraft:polished_andesite_stairs', 'minecraft:quartz_block', 'minecraft:white_concrete_powder'),
    ('minecraft:birch_planks', 'minecraft:cobblestone_stairs', 'minecraft:polished_blackstone', 'minecraft:polished_andesite'),
    ('minecraft:white_wool', 'minecraft:quartz_stairs', None, 'minecraft:bricks'),
    ('minecraft:spruce_planks', 'minecraft:stone_brick_stairs', 'minecraft:gray_terracotta', 'minecraft:spruce_planks'),
    ('minecraft:light_gray_concrete', 'minecraft:cobblestone_stairs', 'minecraft:stone_bricks', 'minecraft:quartz_pillar'),
    ('minecraft:white_concrete', 'minecraft:spruce_stairs', 'minecraft:stone_bricks', 'minecraft:light_gray_concrete'),
    ('minecraft:green_concrete', 'minecraft:stone_brick_stairs', 'minecraft:quartz_pillar', 'minecraft:smooth_sandstone'),
    ('minecraft:spruce_planks', 'minecraft:polished_blackstone_brick_stairs', 'minecraft:cracked_stone_bricks', 'minecraft:cracked_stone_bricks'),
    ('minecraft:bricks', 'minecraft:quartz_stairs', 'minecraft:terracotta', 'minecraft:bricks'),
    ('minecraft:bricks', 'minecraft:spruce_stairs', 'minecraft:cut_sandstone', 'minecraft:cut_sandstone'),
    ('minecraft:lime_terracotta', 'minecraft:nether_brick_stairs', 'minecraft:bricks', 'minecraft:bricks'),
    ('minecraft:quartz_block', 'minecraft:stone_stairs', 'minecraft:oak_planks', 'minecraft:quartz_block'),
    ('minecraft:quartz_block', 'minecraft:nether_brick_stairs', 'minecraft:spruce_planks', 'minecraft:quartz_block'),
    ('minecraft:stone_bricks', 'minecraft:deepslate_tile_stairs', None, 'minecraft:stone_bricks'),
    ('minecraft:stone_bricks', 'minecraft:polished_deepslate_stairs', 'minecraft:spruce_planks', 'minecraft:polished_andesite'),
    ('minecraft:birch_planks', 'minecraft:stone_brick_stairs', 'minecraft:quartz_pillar', 'minecraft:quartz_pillar'),
    ('minecraft:birch_planks', 'minecraft:spruce_stairs', None, 'minecraft:birch_planks'),
    ('minecraft:bricks', 'minecraft:dark_oak_stairs', 'minecraft:dark_oak_planks', 'minecraft:bricks'),
    ('minecraft:dark_oak_planks', 'minecraft:dark_oak_stairs', 'minecraft:black_concrete', 'minecraft:cobbled_deepslate'),
    ('minecraft:mud_bricks', 'minecraft:dark_oak_stairs', None, 'minecraft:spruce_planks'),
    ('minecraft:birch_planks', 'minecraft:brick_stairs', 'minecraft:smooth_sandstone', 'minecraft:birch_planks'),
    ('minecraft:polished_blackstone_bricks', 'minecraft:deepslate_brick_stairs', 'minecraft:blackstone', 'minecraft:birch_planks'),
    ('minecraft:pink_terracotta', 'minecraft:deepslate_brick_stairs', 'minecraft:oak_planks', 'minecraft:pink_terracotta'),
    ('minecraft:yellow_terracotta', 'minecraft:spruce_stairs', 'minecraft:quartz_block', 'minecraft:chiseled_quartz_block'),
    ('minecraft:spruce_planks', 'minecraft:cobbled_deepslate_stairs', 'minecraft:stone_bricks', 'minecraft:spruce_planks'),
    ('minecraft:spruce_planks', 'minecraft:spruce_stairs', 'minecraft:polished_diorite', 'minecraft:polished_diorite'),
    ('minecraft:cracked_polished_blackstone_bricks', 'minecraft:cobbled_deepslate_stairs', 'minecraft:spruce_planks', 'minecraft:tuff_bricks'),
    ('minecraft:birch_planks', 'minecraft:deepslate_brick_stairs', None, 'minecraft:birch_planks'),
    ('minecraft:quartz_pillar', 'minecraft:deepslate_brick_stairs', None, 'minecraft:white_concrete'),
    ('minecraft:pale_oak_planks', 'minecraft:polished_deepslate_stairs', 'minecraft:bricks', 'minecraft:pale_oak_planks'),
    ('minecraft:smooth_quartz', 'minecraft:deepslate_tile_stairs', None, 'minecraft:smooth_quartz'),
    ('minecraft:quartz_bricks', 'minecraft:deepslate_tile_stairs', 'minecraft:polished_deepslate', 'minecraft:quartz_bricks'),
    ('minecraft:light_gray_terracotta', 'minecraft:dark_oak_stairs', 'minecraft:quartz_pillar', 'minecraft:stone_bricks'),
    ('minecraft:light_blue_concrete', 'minecraft:stone_brick_stairs', 'minecraft:quartz_pillar', 'minecraft:smooth_quartz'),
    ('minecraft:white_concrete', 'minecraft:deepslate_tile_stairs', None, 'minecraft:white_concrete'),
    ('minecraft:polished_blackstone_bricks', 'minecraft:andesite_stairs', 'minecraft:birch_planks', 'minecraft:polished_andesite'),
    ('minecraft:spruce_planks', 'minecraft:spruce_stairs', 'minecraft:white_concrete_powder', 'minecraft:spruce_planks'),
    ('minecraft:white_concrete', 'minecraft:stone_brick_stairs', 'minecraft:quartz_pillar', 'minecraft:smooth_sandstone'),
    ('minecraft:quartz_pillar', 'minecraft:quartz_stairs', 'minecraft:white_wool', 'minecraft:bricks'),
    ('minecraft:quartz_block', 'minecraft:deepslate_tile_stairs', 'minecraft:spruce_planks', 'minecraft:quartz_block'),
    ('minecraft:white_terracotta', 'minecraft:stone_brick_stairs', 'minecraft:stone_bricks', 'minecraft:white_terracotta'),
    ('minecraft:quartz_pillar', 'minecraft:stone_brick_stairs', 'minecraft:stone_bricks', 'minecraft:quartz_pillar'),
    ('minecraft:jungle_planks', 'minecraft:stone_brick_stairs', None, 'minecraft:jungle_planks'),
    ('minecraft:quartz_pillar', 'minecraft:stone_brick_stairs', None, 'minecraft:quartz_pillar'),
    ('minecraft:stone_bricks', 'minecraft:stone_brick_stairs', 'minecraft:light_gray_concrete', 'minecraft:light_gray_concrete'),
    ('minecraft:birch_planks', 'minecraft:stone_brick_stairs', 'minecraft:terracotta', 'minecraft:birch_planks'),
    ('minecraft:cyan_terracotta', 'minecraft:deepslate_tile_stairs', 'minecraft:smooth_quartz', 'minecraft:cyan_terracotta'),
    ('minecraft:white_concrete', 'minecraft:stone_brick_stairs', 'minecraft:stone_bricks', 'minecraft:white_concrete'),
    ('minecraft:birch_planks', 'minecraft:stone_brick_stairs', None, 'minecraft:birch_planks'),
    ('minecraft:light_gray_terracotta', 'minecraft:dark_oak_stairs', 'minecraft:quartz_pillar', 'minecraft:light_gray_terracotta'),
    ('minecraft:dark_oak_planks', 'minecraft:oak_stairs', 'minecraft:stone_bricks', 'minecraft:stone_bricks'),
    ('minecraft:spruce_planks', 'minecraft:stone_brick_stairs', 'minecraft:quartz_pillar', 'minecraft:stone_bricks'),
    ('minecraft:white_terracotta', 'minecraft:cobblestone_stairs', 'minecraft:chiseled_quartz_block', 'minecraft:stone_bricks'),
    ('minecraft:bricks', 'minecraft:jungle_stairs', 'minecraft:quartz_pillar', 'minecraft:jungle_planks'),
    ('minecraft:light_blue_terracotta', 'minecraft:stone_brick_stairs', 'minecraft:quartz_pillar', 'minecraft:light_blue_terracotta'),
    ('minecraft:stone_bricks', 'minecraft:cobblestone_stairs', 'minecraft:light_gray_concrete', 'minecraft:stone_bricks'),
    ('minecraft:quartz_pillar', 'minecraft:cobblestone_stairs', 'minecraft:white_concrete', 'minecraft:quartz_pillar'),
    ('minecraft:quartz_block', 'minecraft:cobblestone_stairs', None, 'minecraft:dark_oak_planks'),
    ('minecraft:spruce_planks', 'minecraft:spruce_stairs', 'minecraft:stone_bricks', 'minecraft:spruce_planks'),
    ('minecraft:quartz_block', 'minecraft:quartz_stairs', 'minecraft:yellow_terracotta', 'minecraft:quartz_block'),
    ('minecraft:green_terracotta', 'minecraft:nether_brick_stairs', 'minecraft:nether_bricks', 'minecraft:green_terracotta'),
    ('minecraft:dark_oak_planks', 'minecraft:cobblestone_stairs', 'minecraft:gray_wool', 'minecraft:birch_planks'),
    ('minecraft:cut_sandstone', 'minecraft:stone_brick_stairs', 'minecraft:stone_bricks', 'minecraft:cut_sandstone'),
    ('minecraft:bricks', 'minecraft:brick_stairs', 'minecraft:oak_planks', 'minecraft:bricks'),
    ('minecraft:light_gray_wool', 'minecraft:dark_oak_stairs', 'minecraft:cracked_stone_bricks', 'minecraft:oak_planks'),
    ('minecraft:blue_terracotta', 'minecraft:dark_oak_stairs', 'minecraft:dark_oak_planks', 'minecraft:oak_planks'),
    ('minecraft:stone_bricks', 'minecraft:stone_brick_stairs', 'minecraft:light_gray_terracotta', 'minecraft:stone_bricks'),
    ('minecraft:white_terracotta', 'minecraft:stone_brick_stairs', 'minecraft:black_wool', 'minecraft:white_terracotta'),
    ('minecraft:quartz_pillar', 'minecraft:cobblestone_stairs', 'minecraft:dark_oak_planks', 'minecraft:dark_oak_planks'),
]

# Probability of using a whole observed tuple rather than chaining the conditionals. The
# remainder produces novel-but-plausible combinations by sampling roof/trim/foundation
# conditioned on the wall.
P_OBSERVED_STYLE = 0.75

# ── Colour ───────────────────────────────────────────────────────────────────
# A house that comes out all one grey is the commonest way this generator gives itself away,
# and luminance alone does not catch it: white_terracotta walls under a cobblestone roof are
# 49 apart in brightness and still read as a grey box, because they are the same hue.
#
# Measured on the suburban subset -- wall saturation median 0.43, roof saturation median
# 0.05, and 22 of 34 builds have a wall or roof that is genuinely chromatic. Real suburban
# houses are wood, brick or coloured terracotta *under* a grey or dark roof. The frozen
# STYLES table had drifted to a wall saturation of 0.13, i.e. mostly grey on grey.
#
# So style tuples are now drawn against these constraints rather than uniformly. Both are
# rejection thresholds with a best-effort fallback, so sampling can never fail.
# A style qualifies when its wall and roof are far enough apart in brightness AND the pair
# is not simply two greys -- either the wall has a hue of its own, or the brightness gap is
# wide enough to carry the contrast by itself. That second clause is what keeps the white
# house under a dark roof, which is a real and common suburban look (builds 5, 19, 26).
MIN_WALL_ROOF_CONTRAST = 40   # luminance apart; the measured median for real builds
MIN_WALL_SATURATION = 0.25    # ...and the wall should have a colour, not just a brightness
STRONG_CONTRAST = 80          # ...unless the brightness gap alone is this wide
STYLE_TRIES = 24              # draws before settling for the best-scoring tuple seen
# How often the wall is a material with a hue rather than a white or pale grey. Both kinds
# are real; this is the mixing weight that lands the batch on the measured wall-saturation
# median of 0.43, which drawing uniformly across the qualifying materials does not.
P_CHROMATIC_WALL = 0.62

# The light trim that makes a roof edge pop. Measured off the builds called out for it (#3,
# #15, #29, #81, #99): light material sits on the *perimeter ring* of every course --
# on-ring 0.85 to 1.00 -- at 10-35% of each layer, never as a wall colour. It is the corner
# posts, the band course at each storey, and a thin line running the roof's outer edge.
#
# What separates those builds from the rest is that they use it *sparingly*: light share of
# the whole structure is a median 0.227 for them against 0.440 for everything else. Trim
# reads as trim because most of the house is not it.
ACCENT = {"minecraft:quartz_block": 8, "minecraft:quartz_pillar": 7,
          "minecraft:smooth_quartz": 6, "minecraft:white_concrete": 6,
          "minecraft:white_terracotta": 5, "minecraft:birch_planks": 5,
          "minecraft:polished_diorite": 2, "minecraft:smooth_sandstone": 2}
# For accents with no stair variant of their own (white concrete and terracotta, the two
# commonest), the rake falls back to the nearest thing that does have one.
ACCENT_STAIRS_FALLBACK = "minecraft:quartz_stairs"
MIN_ACCENT_CONTRAST = 45      # an accent that matches the wall is not an accent

P_RAKE_TRIM = 0.60            # light line along the roof's outer edge
P_CORNER_POSTS = 0.55         # light quoins up the corners of the plan
P_BAND_COURSE = 0.70          # light band at each storey division

# The gable end -- the triangle of wall the roof encloses -- is a different material from
# the wall below it in 32 of 73 analysable builds. It is one of the cheapest ways to make a
# house look designed rather than extruded: #101 is dark brown walls under a light tan gable.
GABLE = {"minecraft:birch_planks": 6, "minecraft:smooth_sandstone": 5,
         "minecraft:white_terracotta": 5, "minecraft:oak_planks": 4,
         "minecraft:quartz_block": 4, "minecraft:stripped_oak_wood": 3,
         "minecraft:light_gray_terracotta": 3, "minecraft:stripped_birch_wood": 2}
P_GABLE_DIFFERENT = 0.44      # measured: 32/73

# A chimney is masonry whatever the house is made of -- measured across the library its
# materials are bricks, stone bricks, andesite, terracotta and cobblestone, and 12 of the
# builds with one use something other than their own wall material for it.
CHIMNEY = {"minecraft:bricks": 6, "minecraft:stone_bricks": 5,
           "minecraft:polished_andesite": 3, "minecraft:andesite": 3,
           "minecraft:cobblestone": 3, "minecraft:deepslate_bricks": 2,
           "minecraft:terracotta": 2}

# ── Porches and garages ──────────────────────────────────────────────────────
# A real porch is not a canopy bolted to a wall: it is a bay of the plan with no walls that
# the main roof carries on over, standing on posts, with a deck under it. #101, #103 and #24
# all build it that way, and it is why their fronts read as houses rather than as elevations.
PORCH_WIDTH = {3: 6, 4: 8, 5: 6, 6: 3}
PORCH_DEPTH = {2: 8, 3: 6}
PORCH_POST_MAT = {"minecraft:oak_log": 5, "minecraft:spruce_log": 4,
                  "minecraft:quartz_pillar": 4, "minecraft:oak_fence": 3,
                  "minecraft:stripped_oak_log": 3, "minecraft:cobblestone_wall": 2}
P_PORCH_RAIL = 0.55           # a fence rail between the posts

# ── Massing ──────────────────────────────────────────────────────────────────
# The front wing, and the single biggest gap between the generator and the library. Measured
# over all 106 real builds against the 16 generated ones:
#
#                                   real      generated
#   separate roof-peak groups        9            2
#   builds with a front gable       83%          50%
#   builds with 2+ ridge levels     96%         100%
#   wall-band area spread          0.25         0.09
#
# A house is not a prism with a lid. Nearly every real one has a wing projecting from the
# front with its own gable turned across the main ridge -- it is what gives a suburban house
# a front at all, and it produces the second ridge, the street-facing gable and the wall
# relief in one move. See `_with_wing` and the `parts` argument to `_roof_height`.
P_FRONT_WING = 0.62
WING_WIDTH = {5: 6, 6: 8, 7: 7, 8: 4, 9: 2}
WING_DEPTH = {2: 7, 3: 9, 4: 4}
# Where along the front it lands, 0 = hard left, 1 = hard right. Real wings sit off-centre
# far more often than not; a wing centred on the elevation reads as a church.
WING_OFFSET = {0.0: 5, 0.15: 7, 0.3: 5, 0.7: 5, 0.85: 7, 1.0: 5}
# How tall the wing is. Usually one storey against a taller block -- the single-storey
# garage or living-room bay -- because the step in the wall head is what the measurement
# calls relief. A wing the same height as the house projects but does not step.
WING_STOREYS = {1: 7, 2: 3}

# The plinth: the bottom course of wall in a heavier material, which 60 of the 106 builds
# have. It is what stops a house looking like it was dropped on the lawn.
PLINTH = {"minecraft:stone_bricks": 5, "minecraft:cobblestone": 4, "minecraft:bricks": 4,
          "minecraft:andesite": 3, "minecraft:deepslate_bricks": 2,
          "minecraft:polished_andesite": 2, "minecraft:mud_bricks": 2}
P_PLINTH = 0.57

# Garages are part of the building, not a shed beside it -- a bay of the front wall opened
# to full height with a header over it, with the driveway running to it.
P_GARAGE = 0.45
GARAGE_WIDTH = {3: 5, 4: 9, 5: 4}
GARAGE_DOOR = {"minecraft:spruce_trapdoor": 5, "minecraft:dark_oak_trapdoor": 4,
               "minecraft:iron_bars": 3, "minecraft:oak_trapdoor": 3,
               "minecraft:iron_trapdoor": 2}

# Re-counted over the suburban subset by how many builds use each: glass_pane 16,
# white_stained_glass_pane 12, black_stained_glass_pane 11, black_stained_glass 10,
# gray_stained_glass_pane 7, plain glass 6. Panes outnumber solid glass two to one, and the
# tinted ones are how a builder gets a window that reads as a window at a distance.
GLASS = {"minecraft:glass_pane": 16, "minecraft:white_stained_glass_pane": 12,
         "minecraft:black_stained_glass_pane": 11, "minecraft:gray_stained_glass_pane": 7,
         "minecraft:light_gray_stained_glass_pane": 4, "minecraft:glass": 6,
         "minecraft:black_stained_glass": 5}
# Measured: dark_oak 23, oak 14, birch 13, iron 8, spruce 5, acacia 4.
DOOR = {"minecraft:dark_oak_door": 23, "minecraft:oak_door": 14, "minecraft:birch_door": 13,
        "minecraft:spruce_door": 5, "minecraft:acacia_door": 4}

# ── Ground and landscaping ───────────────────────────────────────────────────
# Measured: 97 of 103 analysable builds sit on a terrain pad; pad depth is 1-3 layers in 54 of
# them, with a long tail of captured hillside that is landscape rather than plot. grass_block
# appears in 94 of 106 builds, short_grass in 73, oak_leaves in 64 -- the plot is not optional
# scenery, it is most of the volume of a real stored build.
PAD_DEPTH = {1: 23, 2: 15, 3: 16, 4: 7}
PAD_MARGIN = {3: 6, 4: 10, 5: 8, 6: 4}
PAD_SURFACE = {"minecraft:grass_block": 10}
PAD_SUBSOIL = {"minecraft:dirt": 10}
GROUND_COVER = {"minecraft:short_grass": 8, "minecraft:dandelion": 1, "minecraft:poppy": 1}
TREE_LOG = {"minecraft:oak_log": 6, "minecraft:birch_log": 3, "minecraft:spruce_log": 3}
TREE_LEAVES = {"minecraft:oak_leaves": 6, "minecraft:birch_leaves": 3,
               "minecraft:spruce_leaves": 3}
# How big a tree is. Measured with one metric on the real builds and the generator alike
# (leaf columns over the open plot, log columns with leaves above them): the generator's
# plots were already as *covered* as the suburban median (canopy over 0.15 of the open
# ground against a real median of 0.08, #2 at 0.17) but its trees were small -- canopy top
# 8 blocks above ground and about 60 leaves per tree, where #2's trees carry 114 leaves each
# and #83's stand 16 tall. Sparse-looking plots were a scale problem, not a count problem.
# Trunk height in blocks, and canopy radius at the widest layer.
TREE_TRUNK = {4: 3, 5: 4, 6: 4, 7: 3, 8: 2}
TREE_CANOPY = {2: 5, 3: 5}
# Walkways and driveways. Measured off the ground outside the wall line: quartz_stairs in 16
# builds, stone_brick_stairs in 14, quartz_slab in 11, smooth_quartz_slab in 10, smooth_stone
# in 9, bricks in 5, polished_andesite in 3. Suburban paving is pale and hard -- gravel and
# dirt paths were an all-builds artefact from gardens and farms, and read as rural.
PATH_BLOCK = {"minecraft:smooth_stone": 9, "minecraft:stone_bricks": 8,
              "minecraft:polished_andesite": 6, "minecraft:quartz_block": 5,
              "minecraft:bricks": 5, "minecraft:smooth_quartz": 4,
              "minecraft:gravel": 3, "minecraft:cobblestone": 2}
# The kerb a path or driveway is edged with, laid flat as a slab.
PATH_EDGE_SLAB = {"minecraft:smooth_stone_slab": 6, "minecraft:quartz_slab": 5,
                  "minecraft:stone_brick_slab": 4, "minecraft:smooth_quartz_slab": 4,
                  "minecraft:polished_andesite_slab": 2}

# Porch decking, and the posts holding its canopy up.
PORCH_FLOOR = {"minecraft:stone_slab": 5, "minecraft:quartz_slab": 4,
               "minecraft:smooth_stone_slab": 4, "minecraft:oak_slab": 3,
               "minecraft:spruce_slab": 3}
PORCH_POST = {"minecraft:oak_fence": 5, "minecraft:spruce_fence": 4,
              "minecraft:dark_oak_fence": 3, "minecraft:cobblestone_wall": 3,
              "minecraft:stone_brick_wall": 2}

# Bushes and shrubs: what the plots are planted with, by build count -- oak_leaves 20,
# spruce_leaves 13, birch_leaves 9, azalea 1.
BUSH_LEAVES = {"minecraft:oak_leaves": 20, "minecraft:spruce_leaves": 13,
               "minecraft:birch_leaves": 9, "minecraft:azalea_leaves": 3}
FLOWERS = {"minecraft:dandelion": 4, "minecraft:poppy": 4, "minecraft:azure_bluet": 6,
           "minecraft:oxeye_daisy": 3, "minecraft:cornflower": 2}

P_CHIMNEY = 0.35
P_PORCH = 0.55                # a front porch or stoop is normal on a suburban house
P_DRIVEWAY = 0.70             # ...and so is somewhere to put the car
DRIVEWAY_W = {3: 8, 4: 6, 5: 3}

# Footprint reweighting for --size. Terciles of the measured width distribution.
SIZE_TERCILES = {"small": (11, 19), "medium": (19, 27), "large": (27, 39)}

# The builds the suburban-specific numbers above were measured on. Kept here so a later
# re-measurement can be run against the same set, and so it is obvious which tables came from
# all 106 builds and which came from these 33. Anything commented "suburban subset" is these.
SUBURBAN_IDS = (3, 5, 6, 7, 8, 9, 15, 16, 17, 19, 24, 25, 26, 29, 30, 54, 59, 60, 65, 66,
                67, 68, 69, 70, 71, 72, 73, 76, 77, 79, 99, 103, 105)
