"""Phase 3 fixes on synthetic builds: D22 interior by opening size, D23 terrain and ground,
site lines, trees, terrain floors, D24 set-asides and the geometric duplicate check."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import mcvoxel  # noqa: F401
    HAVE_MCVOXEL = True
except ImportError:          # the synth requirements are not installed
    HAVE_MCVOXEL = False

from schemset import read as R  # noqa: E402

PAL = ["minecraft:air", "minecraft:grass_block[snowy=false]", "minecraft:dirt",
       "minecraft:oak_planks", "minecraft:stone", "minecraft:cobblestone",
       "minecraft:stone_bricks", "minecraft:smooth_stone_slab[type=top]",
       "minecraft:oak_log[axis=y]", "minecraft:oak_leaves[distance=1,persistent=true]",
       "minecraft:polished_andesite", "minecraft:glass", "minecraft:coal_ore",
       "minecraft:end_stone", "minecraft:sandstone", "minecraft:gold_ore"]
(AIR, GRASS, DIRT, PLANKS, STONE, COBBLE, BRICKS, SLAB, LOG, LEAVES, ANDESITE, GLASS, COAL,
 END_STONE, SANDSTONE, GOLD) = range(16)


def build_of(grid):
    return R.Build(blocks=grid.astype(np.uint16), palette=list(PAL), data_version=3465,
                   source_format="test", raw_non_air=int(np.count_nonzero(grid)))


def plate(x=40, y=24, z=40, depth=2):
    """Dirt with a grass top; the surface is at y = depth - 1."""
    g = np.zeros((x, y, z), dtype=np.uint16)
    g[:, :depth - 1, :] = DIRT
    g[:, depth - 1, :] = GRASS
    return g


def box(g, x0, x1, y0, y1, z0, z1, wall=PLANKS, floor=BRICKS, roof=BRICKS):
    """Closed hollow box: walls x0..x1 / z0..z1 (inclusive), floor y0, roof y1; inside cleared."""
    g[x0 + 1:x1, y0 + 1:y1, z0 + 1:z1] = AIR
    g[x0:x1 + 1, y0, z0:z1 + 1] = floor
    g[x0:x1 + 1, y1, z0:z1 + 1] = roof
    for y in range(y0 + 1, y1):
        g[x0:x1 + 1, y, z0] = g[x0:x1 + 1, y, z1] = wall
        g[x0, y, z0:z1 + 1] = g[x1, y, z0:z1 + 1] = wall
    return g


def run(grid):
    from schemset.crop import crop_build
    from schemset.interior import compute_masks
    crop = crop_build(build_of(grid))
    return crop, compute_masks(crop.blocks, crop.ground)


def at(crop, x, y, z):
    """Source coordinates -> index into the cropped grid."""
    ox, oy, oz = crop.offset
    return x - ox, y - oy, z - oz


@unittest.skipUnless(HAVE_MCVOXEL, "mcvoxel (requirements-synth.txt) not installed")
class InteriorByOpening(unittest.TestCase):
    """D22: openings at least 3 wide let the outside in; covered air alone is never interior."""

    def test_porch(self):
        g = plate()
        box(g, 10, 20, 2, 7, 10, 18)                      # house, floor at y=2
        g[10:21, 2, 19:24] = PLANKS                      # porch deck in front (+z)
        g[10:21, 7, 19:24] = BRICKS                      # porch roof
        for x in (10, 15, 20):                           # posts: 4-wide open bays
            g[x, 3:7, 23] = LOG
        crop, m = run(g)
        porch = at(crop, 13, 4, 21)
        self.assertTrue(m.exterior[porch])
        self.assertTrue(m.shell[at(crop, 13, 4, 18)])   # the house wall behind the porch
        self.assertFalse(m.exterior[at(crop, 15, 4, 14)])   # inside the house

    def test_jettied_upper_floor(self):
        g = plate()
        box(g, 10, 18, 2, 6, 10, 18)                      # ground floor
        box(g, 9, 19, 6, 11, 9, 19)                       # upper floor overhangs by 1
        crop, m = run(g)
        self.assertTrue(m.exterior[at(crop, 9, 4, 14)])     # under the jetty
        self.assertTrue(m.shell[at(crop, 10, 4, 14)])       # wall under the overhang
        self.assertFalse(m.exterior[at(crop, 14, 4, 14)])

    def test_recessed_balcony(self):
        g = plate()
        g[10:21, 2:13, 10:19] = BRICKS                   # a solid block of a building
        g[14:17, 7:10, 17:19] = 0                        # 3 wide, 2 deep, 3 tall recess
        crop, m = run(g)
        self.assertTrue(m.exterior[at(crop, 15, 8, 17)])
        self.assertTrue(m.shell[at(crop, 15, 8, 16)])

    def test_room_behind_small_openings_stays_interior(self):
        g = plate()
        box(g, 10, 20, 2, 7, 10, 18)
        g[12, 3:5, 10] = 0                               # 1-wide doorway
        g[15:17, 4:6, 10] = 0                            # 2-wide window, no glass
        crop, m = run(g)
        self.assertFalse(m.exterior[at(crop, 15, 4, 14)])
        self.assertTrue(m.exterior[at(crop, 12, 3, 10)])    # the doorway itself is a reveal
        self.assertNotIn("large_opening", m.flags)

    def test_window_reveal(self):
        g = plate()
        box(g, 10, 20, 2, 7, 10, 18)
        g[10:21, 3:7, 9] = BRICKS                        # a second, outer wall layer
        g[14:16, 4:6, 9] = 0                             # 2 x 2 window: a 1-deep reveal
        g[14:16, 4:6, 10] = GLASS                        # glass set in the inner layer
        crop, m = run(g)
        self.assertTrue(m.exterior[at(crop, 14, 4, 9)])
        self.assertTrue(m.shell[at(crop, 14, 4, 10)])       # the glass is trained on

    def test_open_gap_under_eaves_is_exterior(self):
        g = plate()
        box(g, 10, 20, 2, 7, 10, 18)
        g[9:22, 7, 8:21] = BRICKS                        # roof with 2-block eaves
        g[10:21, 2:4, 8] = STONE                         # a low garden wall 1 block out
        crop, m = run(g)
        self.assertTrue(m.exterior[at(crop, 15, 2, 9)])     # the strip between wall and house
        self.assertTrue(m.shell[at(crop, 15, 2, 10)])

    def test_big_door_into_a_hall(self):
        """D22 amended: the outside reaches in through a big door, but only COVER_DEPTH deep."""
        from schemset.interior import COVER_DEPTH
        g = plate()
        box(g, 10, 30, 2, 9, 10, 30)                      # a hall
        g[18:23, 3:7, 10] = 0                            # 5-wide, 4-tall door, open
        crop, m = run(g)
        self.assertTrue(m.exterior[at(crop, 20, 4, 10 + COVER_DEPTH - 1)])
        self.assertFalse(m.exterior[at(crop, 20, 4, 20)])       # deep inside: interior
        self.assertIn("deep_cover_capped", m.flags)
        self.assertGreater(m.stats["deep_cover_capped"], 200)


@unittest.skipUnless(HAVE_MCVOXEL, "mcvoxel (requirements-synth.txt) not installed")
class DepthUnderCover(unittest.TestCase):
    """D22 amended: covered exterior air is supervised only within COVER_DEPTH of open sky."""

    def test_four_deep_porch_stays_exterior(self):
        g = plate()
        box(g, 10, 22, 2, 8, 10, 18)
        g[10:23, 2, 19:23] = PLANKS                      # porch deck, 4 deep (z 19..22)
        g[10:23, 8, 19:23] = BRICKS                      # porch roof
        for x in (10, 16, 22):
            g[x, 3:8, 22] = LOG
        crop, m = run(g)
        for z in range(19, 23):
            self.assertTrue(m.exterior[at(crop, 13, 4, z)], z)
        self.assertTrue(m.shell[at(crop, 13, 4, 18)])       # the wall behind the porch
        self.assertNotIn("deep_cover_capped", m.flags)

    def test_open_fronted_hall(self):
        """Depth counts from open air (normalize-5): the uncovered air outside, and the band
        just inside the 6-tall opening that sees the sky steeply -- 6 // SKY_SLOPE deep."""
        from schemset.interior import COVER_DEPTH, SKY_SLOPE
        g = plate()
        box(g, 10, 24, 2, 9, 12, 23)                      # walls on all sides...
        g[11:24, 3:9, 23] = 0                            # ...but the +z front is open
        g[10:25, 2:10, 4:13] = BRICKS                    # solid building behind the hall
        crop, m = run(g)                                  # depth: z 23 (1) back to z 13 (11)
        for z in range(13, 24):
            k = 24 - z
            self.assertEqual(bool(m.exterior[at(crop, 17, 5, z)]),
                             k <= COVER_DEPTH + 6 // SKY_SLOPE, k)
        self.assertFalse(m.shell[at(crop, 17, 5, 12)])      # the back wall: ignored

    def test_facades_under_a_deep_tall_eave_stay_shell(self):
        """Hotel Piran (normalize-4 bug): a 9-block eave 30 blocks up hid three facades. The
        air under it is open to the side and sees the sky, so depth counts from there."""
        g = plate(x=48, y=40, z=48)
        box(g, 16, 31, 2, 32, 16, 31, roof=PLANKS)          # a 30-tall closed block...
        g[7:41, 32, 7:41] = BRICKS                       # ...under a 9-block eave all round
        crop, m = run(g)
        for x, z in ((15, 24), (32, 24), (24, 15), (24, 32)):   # air in front of each side
            for y in (4, 16, 31):
                self.assertTrue(m.exterior[at(crop, x, y, z)], (x, y, z))
        for x, z in ((16, 24), (31, 24), (24, 16), (24, 31)):   # every facade is shell
            for y in (4, 16, 31):
                self.assertTrue(m.shell[at(crop, x, y, z)], (x, y, z))
        self.assertTrue(m.shell[at(crop, 12, 32, 24)])      # and the eave's underside
        self.assertNotIn("deep_visible_ignored", m.flags)

    def test_deep_low_canopy_is_still_capped(self):
        """The cap still holds under a low cover: a 3-tall canopy 12 deep sees the sky only
        near its edge, so the wall at its back is ignored."""
        from schemset.interior import COVER_DEPTH
        g = plate(x=48, y=24, z=40)
        box(g, 10, 36, 2, 12, 8, 14)                      # a building...
        g[10:37, 5, 15:27] = BRICKS                      # ...with a canopy 12 deep, 3 up
        crop, m = run(g)
        self.assertTrue(m.exterior[at(crop, 20, 3, 26 - COVER_DEPTH + 1)])
        self.assertFalse(m.exterior[at(crop, 20, 3, 16)])
        self.assertFalse(m.shell[at(crop, 20, 3, 14)])      # the wall under the canopy

    def test_hollow_shell_with_a_beam(self):
        g = plate(x=40, y=24, z=40)
        box(g, 8, 32, 2, 18, 8, 32)                       # hollow box, roofed
        g[19:22, 3:6, 8] = 0                             # one 3-wide, 3-tall opening
        beam = [(x, 4 + (x - 14) // 2, 30 - (x - 14)) for x in range(14, 27)]
        for x, y, z in beam:                             # a diagonal beam, far from it
            g[x, y, z] = LOG
        crop, m = run(g)
        self.assertTrue(m.exterior[at(crop, 20, 4, 9)])     # just inside the opening
        for x, y, z in beam:
            self.assertFalse(m.shell[at(crop, x, y, z)], (x, y, z))

    def test_courtyard_facades_stay_shell(self):
        g = plate(x=40, y=16, z=40)
        g[8:31, 2:9, 8:31] = BRICKS                      # a block...
        g[14:25, 2:9, 14:25] = 0                         # ...around an open courtyard
        g[14:25, 2, 14:25] = PLANKS                      # courtyard paving
        crop, m = run(g)
        self.assertTrue(m.shell[at(crop, 13, 5, 19)])       # courtyard facade
        self.assertTrue(m.shell[at(crop, 19, 5, 13)])
        self.assertTrue(m.exterior[at(crop, 19, 5, 19)])
        self.assertEqual(m.stats["deep_cover_capped"], 0)

    def test_two_deep_jetty_stays_exterior(self):
        g = plate()
        box(g, 10, 18, 2, 6, 10, 18)                      # ground floor
        box(g, 8, 20, 6, 11, 8, 20)                       # upper floor overhangs by 2
        crop, m = run(g)
        self.assertTrue(m.exterior[at(crop, 8, 4, 14)])
        self.assertTrue(m.exterior[at(crop, 9, 4, 14)])
        self.assertTrue(m.shell[at(crop, 10, 4, 14)])


@unittest.skipUnless(HAVE_MCVOXEL, "mcvoxel (requirements-synth.txt) not installed")
class GroundAndCrop(unittest.TestCase):
    """D23: only natural terrain makes ground; building materials are kept."""

    def test_platform_without_terrain_has_no_ground(self):
        g = np.zeros((30, 16, 30), np.uint16)
        g[:, 0, :] = ANDESITE                            # a paved platform, no terrain
        box(g, 10, 18, 1, 6, 10, 18)
        crop, m = run(g)
        self.assertIsNone(crop.stats["ground_level"])
        self.assertEqual(crop.stats["ground_method"], "none")
        self.assertEqual(crop.stats["kept_below_ground"], 0)
        self.assertEqual(m.stats["below_ground_ignored"], 0)
        self.assertGreater(m.stats["interior_air"], 0)

    def test_stone_plate_is_not_terrain(self):
        g = np.zeros((30, 16, 30), np.uint16)
        g[:, 0, :] = STONE                               # one layer of stone: a plaza
        box(g, 10, 18, 1, 6, 10, 18)
        crop, _ = run(g)
        self.assertEqual(crop.stats["terrain_voxels"], 0)

    def test_cobblestone_foundation_and_stone_walls_on_grass(self):
        g = plate(depth=3)                               # surface y = 2
        box(g, 10, 18, 3, 9, 10, 18, wall=STONE, floor=PLANKS, roof=BRICKS)
        g[10:19, 2, 10:19] = COBBLE                      # foundation course in the plate
        crop, m = run(g)
        kept = {crop.palette[i] for i in np.unique(crop.blocks)}
        self.assertNotIn(PAL[GRASS], kept)
        self.assertNotIn(PAL[DIRT], kept)
        self.assertEqual(int((crop.blocks == crop.palette.index(PAL[STONE])).sum()),
                         int((g == STONE).sum()))       # every stone wall block kept
        self.assertEqual(int((crop.blocks == crop.palette.index(PAL[COBBLE])).sum()), 81)
        self.assertEqual(crop.stats["ground_level"], 2)

    def test_stone_walls_standing_on_grass(self):
        g = plate(depth=3)
        box(g, 10, 18, 3, 9, 10, 18, wall=STONE, floor=STONE, roof=BRICKS)
        crop, _ = run(g)
        self.assertEqual(int((crop.blocks == crop.palette.index(PAL[STONE])).sum()),
                         int((g == STONE).sum()))

    def test_boundary_wall_removed_building_kept(self):
        from schemset.crop import REACH
        g = plate(x=60, z=40)
        box(g, 10, 18, 2, 7, 10, 18)
        g[19:49, 2:5, 14] = BRICKS                       # 30 long, 1 wide, attached
        crop, _ = run(g)
        house = int((g[10:19, 2:8, 10:19] != 0).sum())
        kept_house = int((crop.blocks[at(crop, 10, 2, 10)[0]:at(crop, 19, 2, 10)[0], :, :] != 0).sum())
        self.assertGreaterEqual(kept_house, house)
        far = 18 + REACH + 2
        self.assertTrue(far - crop.offset[0] >= crop.blocks.shape[0]
                        or not crop.blocks[far - crop.offset[0]:].any())

    def test_wall_with_coping_is_a_line(self):
        g = plate(x=70, z=40)
        box(g, 10, 22, 2, 9, 10, 22)
        g[26:66, 2:6, 16] = BRICKS                       # a free-standing 4-tall wall...
        g[26:66, 6, 15:18] = SLAB                        # ...with a 3-wide coping
        crop, _ = run(g)
        self.assertIn("site_linear_removed", crop.flags)
        self.assertEqual(crop.blocks.shape[0], 13 + 2 * 2)   # the house alone

    def test_tall_narrow_wing_is_kept(self):
        g = plate(x=70, z=40)
        box(g, 10, 22, 2, 9, 10, 22)
        for y0 in range(2, 20, 4):                       # 4 wide, 20 tall, 30 long block
            box(g, 26, 55, y0, y0 + 4, 14, 17)
        crop, _ = run(g)
        self.assertNotIn("site_linear_removed", crop.flags)
        self.assertEqual(int((crop.blocks != 0).sum()), int((g >= PLANKS).sum()))

    def test_ground_fallback(self):
        g = plate(y=30, depth=12)                        # deep plate, surface y = 11
        box(g, 10, 18, 4, 16, 10, 18)                    # mostly buried house
        crop, m = run(g)
        self.assertIn("ground_fallback", crop.flags)
        self.assertEqual(crop.stats["ground_method"], "fallback")
        self.assertEqual(crop.stats["kept_below_ground"], 0)
        self.assertEqual(m.stats["below_ground_ignored"], 0)

    def test_buried_foundation_below_ground(self):
        g = plate(y=30, depth=5)                         # surface y = 4
        box(g, 10, 18, 3, 22, 10, 18)                    # floor one below the surface
        crop, m = run(g)
        self.assertEqual(crop.stats["ground_method"], "terrain")
        self.assertGreater(m.stats["below_ground_ignored"], 0)
        self.assertLess(m.stats["below_ground_fraction"], 0.15)

    def test_logs_in_a_building_are_not_a_tree(self):
        g = plate()
        box(g, 10, 18, 2, 10, 10, 18)
        g[14, 7:9, 9] = LOG                              # two frame logs high on a wall...
        g[13:16, 6:10, 8] = LEAVES                       # ...behind an ivy bush
        g[30, 2:7, 30] = LOG                             # and a real tree on the ground
        g[28:33, 7:9, 28:33] = LEAVES
        crop, _ = run(g)
        self.assertEqual(int((crop.blocks == crop.palette.index(PAL[LOG])).sum()), 2)

    def test_terrain_floor_under_a_room(self):
        from schemset.crop import floor_under_rooms
        g = plate(depth=2)                               # surface y = 1: the hall's floor
        box(g, 10, 20, 1, 7, 10, 20, floor=GRASS)
        crop, m = run(g)
        n = floor_under_rooms(build_of(g), crop, m.interior)
        self.assertEqual(n, 9 * 9)
        self.assertIn(PAL[GRASS], crop.palette)


def count(crop, block):
    return int((crop.blocks == crop.palette.index(PAL[block])).sum()) if PAL[block] in crop.palette else 0


class SiteCleanup(unittest.TestCase):
    """normalize-6, crop steps 9-11: terrain masses, flat aprons, trees, detached parts."""

    def crop(self, grid):
        from schemset.crop import crop_build
        return crop_build(build_of(grid))

    def test_building_on_a_block_of_rock(self):
        g = np.zeros((36, 30, 36), np.uint16)
        g[:, :12, :] = STONE
        g[3:33:5, 2:10:3, 3:33:5] = COAL                 # sparse ores
        g[4:9, 3:6, 4:30] = AIR                          # a cave breaks the columns' ground runs
        g[6, 3, 5:28:4] = COBBLE                         # with bits of a mine in it
        g[:, 12, :] = GRASS
        box(g, 10, 22, 13, 19, 10, 22)
        crop = self.crop(g)
        self.assertEqual(count(crop, STONE), 0)
        self.assertEqual(count(crop, COAL), 0)
        self.assertEqual(count(crop, COBBLE), 0)
        self.assertEqual(count(crop, PLANKS), int((g == PLANKS).sum()))

    def test_stone_building_with_a_solid_core(self):
        g = np.zeros((30, 24, 30), np.uint16)
        g[:, 0, :] = ANDESITE                            # a paved plate, no terrain
        g[8:20, 1:15, 8:20] = END_STONE                  # solid end stone...
        for y in range(3, 14, 3):                        # ...with windows on every face
            g[9:19:3, y, 8] = g[9:19:3, y, 19] = GLASS
            g[8, y, 9:19:3] = g[19, y, 9:19:3] = GLASS
        g[8:20, 15, 8:20] = BRICKS
        crop = self.crop(g)
        self.assertEqual(count(crop, END_STONE), int((g == END_STONE).sum()))

    def test_a_mass_of_ore_is_a_builders_material(self):
        g = np.zeros((30, 24, 30), np.uint16)
        g[:, 0, :] = ANDESITE
        g[8:20, 1:7, 8:20] = GOLD                        # a texture pack's stone, solid
        box(g, 8, 19, 7, 13, 8, 19)
        crop = self.crop(g)
        self.assertEqual(count(crop, GOLD), int((g == GOLD).sum()))

    def test_sandstone_pyramid_is_kept(self):
        # On a paved plate: standing on grass, step 1 already reads a solid natural mass as a
        # hill (D23). Step 9 leaves it: sandstone is a builder's material, not terrain fill.
        g = np.zeros((40, 16, 40), np.uint16)
        g[:, 0, :] = ANDESITE
        for k in range(8):
            g[5 + k:34 - k, 1 + k, 5 + k:34 - k] = SANDSTONE
        crop = self.crop(g)
        self.assertEqual(count(crop, SANDSTONE), int((g == SANDSTONE).sum()))

    def test_parking_lot_is_trimmed(self):
        from schemset.crop import APRON_KEEP, MARGIN, PAD
        g = np.zeros((60, 16, 40), np.uint16)
        g[:, 0, :] = ANDESITE                            # a lot all round, no terrain
        box(g, 5, 15, 1, 8, 10, 25)
        crop = self.crop(g)
        self.assertIn("apron_removed", crop.flags)
        reach = MARGIN + APRON_KEEP
        self.assertEqual(crop.blocks.shape[0], (15 - 5 + 1) + 2 * reach + 2 * PAD - (5 - reach < 0))
        self.assertEqual(count(crop, PLANKS), int((g == PLANKS).sum()))

    def test_detached_parts_go(self):
        # Step 11 on what earlier steps left: a lamp post and a rock fragment separated from
        # the house by two blocks or more go; a part one block away is bridged and stays.
        from schemset.crop import BUILT, NATURAL, clean_site
        role = np.full((50, 20, 40), BUILT, np.uint8)
        kept = np.zeros(role.shape, bool)
        kept[5:19, 1:10, 10:26] = True                   # the house
        kept[24, 1:7, 17] = True                         # a lamp post, 5 away
        kept[30:33, 1:4, 30:33] = True                   # a fragment, 27 blocks
        role[30:33, 1:4, 30:33] = NATURAL
        kept[20, 1:4, 12] = True                         # a bollard one block from the wall
        height = np.full((50, 40), -1)
        out, _, detached, stats = clean_site(kept, role, height, np.zeros_like(kept))
        self.assertFalse(out[24].any())
        self.assertFalse(out[30:33].any())
        self.assertTrue(out[20, 1:4, 12].all())
        self.assertEqual(stats["detached_parts"], 2)
        self.assertEqual(int(out.sum()), 14 * 9 * 16 + 3)

    def test_second_building_of_a_compound_stays(self):
        g = plate(x=60, z=40)
        box(g, 5, 18, 2, 9, 10, 25)
        box(g, 30, 43, 2, 9, 10, 25)
        crop = self.crop(g)
        self.assertEqual(count(crop, PLANKS), int((g == PLANKS).sum()))
        self.assertIn("multi_building", crop.flags)

    def test_tree_against_the_wall_goes_ivy_and_green_roof_stay(self):
        g = plate(x=40, z=36)                            # surface y = 1
        box(g, 10, 20, 2, 9, 10, 20)
        g[21:30, 2, 8:23] = ANDESITE                     # a plaza; the tree stands on it
        g[23, 3:6, 15] = LOG
        g[21:26, 6:9, 13:18] = LEAVES                    # its crown just touches the wall
        g[9, 3:8, 12:18] = LEAVES                        # ivy on the other wall
        g[11:20, 10, 11:20] = LEAVES                     # a green roof
        crop = self.crop(g)
        self.assertEqual(count(crop, LOG), 0)
        self.assertEqual(count(crop, LEAVES), 5 * 6 + 9 * 9)
        self.assertGreater(crop.stats["removed_trees"], 0)


class SetAsideAndDuplicates(unittest.TestCase):
    def rows(self):
        from schemset import freeze as F
        rows = []
        for i, (bid, creator, split) in enumerate((
                ("pmc-the-lighthouse-5778696", "a", "train"),
                ("pmc-traditional-house-cottage-style", "b", "train"),
                ("x1", "c", "train"), ("x2", "c", "train"), ("y1", "d", "test"))):
            rows.append({"id": bid, "split": split, "flags": [], "creator_group": creator,
                         "licence_tier": "B"})
        groups = [{"key": k, "split": s, "ids": [r["id"] for r in rows if r["creator_group"] == k]}
                  for k, s in (("a", "train"), ("b", "train"), ("c", "train"), ("d", "test"))]
        return F, rows, groups

    def test_d24_set_aside(self):
        F, rows, groups = self.rows()
        with tempfile.TemporaryDirectory() as tmp:
            out = F.apply_overrides(rows, groups, Path(tmp) / "none.json")
        by_id = {r["id"]: r for r in rows}
        for bid in ("pmc-the-lighthouse-5778696", "pmc-traditional-house-cottage-style"):
            self.assertEqual(by_id[bid]["split"], "excluded")
            self.assertIn("excluded_landscape_diorama", by_id[bid]["flags"])
        self.assertEqual(len(out["set_aside_d24"]), 2)
        self.assertEqual(out["duplicate_moves"], [])

    def test_d25_cliff_house_set_aside_and_nobody_else_moves(self):
        F, rows, groups = self.rows()
        rows.append({"id": "pmc-richelieu-contemporary", "split": "test", "flags": [],
                     "creator_group": "e", "licence_tier": "B"})
        groups.append({"key": "e", "split": "test", "ids": ["pmc-richelieu-contemporary"]})
        with tempfile.TemporaryDirectory() as tmp:
            out = F.apply_overrides(rows, groups, Path(tmp) / "none.json")
        by_id = {r["id"]: r for r in rows}
        self.assertEqual(by_id["pmc-richelieu-contemporary"]["split"], "excluded")
        self.assertIn("excluded_target_lost_terrain", by_id["pmc-richelieu-contemporary"]["flags"])
        self.assertEqual(len(out["set_aside_d24"]), 3)
        self.assertEqual([by_id[b]["split"] for b in ("x1", "x2", "y1")], ["train", "train", "test"])

    def test_duplicate_pair_moves_the_smaller_group(self):
        F, rows, groups = self.rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dups.json"
            path.write_text(json.dumps({"pairs": [{"ids": ["x1", "y1"], "verdict": "duplicate"}]}))
            out = F.apply_overrides(rows, groups, path)
        by_id = {r["id"]: r for r in rows}
        self.assertEqual(by_id["y1"]["split"], "train")          # group d (1) joins c (2)
        self.assertEqual(by_id["x2"]["split"], "train")
        self.assertEqual(out["duplicate_moves"][0]["ids"], ["y1"])
        self.assertIn("geometric_duplicate", by_id["x1"]["flags"])

    def test_best_iou_finds_rotated_mirrored_copy(self):
        from schemset.dedupe import best_iou, _tight
        rng = np.random.default_rng(3)
        a = rng.random((9, 7, 13)) > 0.6
        b = np.rot90(a[::-1], 1, axes=(0, 2))
        b = np.pad(b, ((1, 0), (0, 1), (2, 0)))
        iou, name, _ = best_iou(_tight(a), _tight(b))
        self.assertAlmostEqual(iou, 1.0)
        self.assertTrue(name.startswith("mirror"))


if __name__ == "__main__":
    unittest.main()
