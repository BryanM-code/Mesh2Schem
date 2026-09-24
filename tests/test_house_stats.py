"""Tests for the library-anatomy analysis.

Every fixture here is a house whose anatomy is known *by construction* -- the ground pad is
exactly this many layers, the floors are at exactly these heights, the roof is exactly this
gable. That is the only way to test an analyser whose whole job is inferring those facts:
asserting against a real build would just be asserting that the code still does what it did
last time, including whatever it got wrong.

The awkward cases each come from a real build that broke an earlier version of the analyser,
and the test names say which: a stray full-coverage layer far below the house
(`49__pmc-suburban-house`), fifteen solid layers of stone underneath it (`11__lastylehome`),
a pad that covers a third of the bounding box because trees inflated it (`10__westlahome`),
and a chimney poking through an otherwise obvious gable (`100__georgian-house`).
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import house_stats
import schem_voxels

AIR = 0
NAMES = [
    "minecraft:air",            # 0
    "minecraft:grass_block",    # 1
    "minecraft:dirt",           # 2
    "minecraft:bricks",         # 3  wall
    "minecraft:oak_planks",     # 4  floor
    "minecraft:stone_brick_stairs",  # 5  roof
    "minecraft:glass",          # 6
    "minecraft:chest",          # 7  furniture
    "minecraft:quartz_pillar",  # 8  trim
    "minecraft:stone",          # 9  deep terrain
]
GRASS, DIRT, WALL, FLOOR, ROOF, GLASS, CHEST, TRIM, STONE = range(1, 10)


def make_house(width=21, depth=17, storeys=2, storey_height=5, pad=2,
               roof="gable", ridge=6, margin=3, deep=0, gap=0, chimney=False,
               furniture=0, shed=False, wall_block=WALL):
    """Build a voxel grid whose anatomy is known exactly, and return a VoxelGrid.

    The house sits centred on a pad `margin` blocks larger than it on every side, so the
    footprint the analyser recovers is the *building*, not the plot.
    """
    plot_w, plot_d = width + 2 * margin, depth + 2 * margin
    base = gap + deep + pad
    wall_top = base + storeys * storey_height
    height = wall_top + ridge + 3
    grid = np.zeros((height, plot_d, plot_w), dtype=np.uint16)

    for y in range(gap, gap + deep):
        grid[y, :, :] = STONE
    for y in range(gap + deep, base):
        grid[y, :, :] = DIRT
    if pad:
        grid[base - 1, :, :] = GRASS

    x0, z0 = margin, margin
    x1, z1 = x0 + width - 1, z0 + depth - 1

    for s in range(storeys):
        fy = base + s * storey_height
        grid[fy, z0:z1 + 1, x0:x1 + 1] = FLOOR
        for y in range(fy + 1, fy + storey_height):
            grid[y, z0:z1 + 1, x0] = wall_block
            grid[y, z0:z1 + 1, x1] = wall_block
            grid[y, z0, x0:x1 + 1] = wall_block
            grid[y, z1, x0:x1 + 1] = wall_block
        # Windows, so the openings pass has something to find.
        wy = fy + 2
        for wx in range(x0 + 3, x1 - 2, 5):
            grid[wy:wy + 2, z0, wx:wx + 2] = GLASS
            grid[wy:wy + 2, z1, wx:wx + 2] = GLASS

    grid[wall_top, z0:z1 + 1, x0:x1 + 1] = FLOOR

    cz = (z0 + z1) / 2.0
    cx = (x0 + x1) / 2.0
    for z in range(z0, z1 + 1):
        for x in range(x0, x1 + 1):
            if roof == "flat":
                top = wall_top + 1
            elif roof == "gable":
                top = wall_top + max(1, int(round(ridge - abs(z - cz))))
            elif roof == "hip":
                top = wall_top + max(1, int(round(ridge - max(abs(z - cz), abs(x - cx)))))
            else:
                raise ValueError(roof)
            grid[wall_top + 1:top + 1, z, x] = ROOF

    if chimney:
        # A 2x2 stack well above the ridge: the thing that makes a naive "measure the ridge
        # at the height map's maximum" classifier report a 2x2 blob instead of a roof.
        grid[wall_top:wall_top + ridge + 3, z0 + 2:z0 + 4, x0 + 2:x0 + 4] = WALL

    for i in range(furniture):
        grid[base + 1, z0 + 2 + (i % 5), x0 + 2 + (i // 5)] = CHEST

    if shed:
        # A detached outbuilding, five blocks clear of the house.
        sz, sx = z0, x1 + 4
        if sx + 3 < plot_w:
            grid[base + 1:base + 4, sz:sz + 4, sx:sx + 3] = WALL

    return schem_voxels.VoxelGrid(grid, list(NAMES)), {
        "base": base, "wall_top": wall_top, "width": width, "depth": depth,
        "storeys": storeys, "storey_height": storey_height,
    }


def anatomy(voxels):
    """Run the same pipeline `analyse` does, on an in-memory grid."""
    seg = house_stats.segment(voxels)
    sub_info = {"seg": seg}
    if seg.reject or seg.structure is None or not seg.structure.any():
        return sub_info
    ys, zs, xs = np.nonzero(seg.structure)
    sub = seg.structure[ys.min():ys.max() + 1, zs.min():zs.max() + 1, xs.min():xs.max() + 1]
    sub_grid = voxels.grid[ys.min():ys.max() + 1, zs.min():zs.max() + 1, xs.min():xs.max() + 1]
    footprint = sub.any(axis=0)
    area = int(footprint.sum())
    peaks, _ = house_stats._analyse_storeys(sub, area)
    sub_info.update(sub=sub, sub_grid=sub_grid, footprint=footprint, area=area, peaks=peaks)
    if peaks:
        sub_info["roof"] = house_stats._analyse_roof(sub, peaks)
    return sub_info


class GroundSegmentationTest(unittest.TestCase):

    def test_pad_found(self):
        voxels, spec = make_house(pad=3)
        seg = house_stats.segment(voxels)
        self.assertEqual(seg.ground_top, spec["base"] - 1)
        self.assertEqual(seg.method, "terrain_pad")
        self.assertIsNone(seg.reject)

    def test_no_pad_at_all(self):
        """`102__traditionhouse3` has no plot: its biggest terrain layer is 7% of the bbox."""
        voxels, _ = make_house(pad=0, margin=2)
        seg = house_stats.segment(voxels)
        self.assertIsNone(seg.ground_top)
        self.assertIsNone(seg.reject)
        self.assertTrue(seg.structure.any())

    def test_deep_terrain_block(self):
        """`11__lastylehome` carries fifteen fully solid layers of stone under the house."""
        voxels, spec = make_house(deep=12, pad=1)
        seg = house_stats.segment(voxels)
        self.assertEqual(seg.ground_top, spec["base"] - 1)
        self.assertIn("deep_terrain", seg.flags)

    def test_floating_start(self):
        """`10__westlahome` starts four layers up, with nothing below the pad."""
        voxels, spec = make_house(gap=4, pad=2)
        seg = house_stats.segment(voxels)
        self.assertEqual(seg.ground_top, spec["base"] - 1)

    def test_pad_survives_inflated_bounding_box(self):
        """The `10__westlahome` failure: trees inflate the bbox so the pad reads as 30%.

        Normalising terrain coverage by the bounding box loses the pad entirely; normalising
        by the largest terrain layer finds it. The margin here makes the plot much wider than
        the pad, which is what a stand of trees does to a real build.
        """
        voxels, spec = make_house(margin=2, pad=2)
        grid = voxels.grid
        tall = np.zeros((grid.shape[0], grid.shape[1] + 20, grid.shape[2] + 20),
                        dtype=np.uint16)
        tall[:, :grid.shape[1], :grid.shape[2]] = grid
        # A lone tree far out in the corner, inflating the bbox without adding pad.
        tall[spec["base"]:spec["base"] + 6, -3, -3] = STONE
        seg = house_stats.segment(schem_voxels.VoxelGrid(tall, list(NAMES)))
        self.assertEqual(seg.ground_top, spec["base"] - 1)

    def test_floating_gap_discards_stray_layer(self):
        """`49__pmc-suburban-house`: one full-coverage layer, then the real ground above it."""
        voxels, spec = make_house(pad=2, gap=0)
        grid = voxels.grid
        stretched = np.zeros((grid.shape[0] + 11, grid.shape[1], grid.shape[2]),
                             dtype=np.uint16)
        stretched[11:] = grid
        stretched[0, :, :] = GRASS  # the stray layer, eleven below the real pad
        seg = house_stats.segment(schem_voxels.VoxelGrid(stretched, list(NAMES)))
        self.assertIn("floating_gap", seg.flags)
        self.assertEqual(seg.ground_top, spec["base"] - 1 + 11)

    def test_plot_slab_is_not_a_floor(self):
        """A build on a laid platform has no terrain material at all."""
        voxels, _ = make_house(pad=0, margin=3)
        grid = voxels.grid.copy()
        grid[0, :, :] = TRIM  # a full-coverage quartz platform
        seg = house_stats.segment(schem_voxels.VoxelGrid(grid, list(NAMES)))
        self.assertEqual(seg.method, "plot_slab")
        self.assertEqual(seg.ground_top, 0)

    def test_detached_shed_is_dropped(self):
        voxels, spec = make_house(shed=True, margin=6)
        seg = house_stats.segment(voxels)
        footprint = seg.structure.any(axis=0)
        self.assertLessEqual(int(footprint.sum()),
                             spec["width"] * spec["depth"] + 4)

    def test_tiny_build_rejected(self):
        grid = np.zeros((8, 6, 6), dtype=np.uint16)
        grid[0, :, :] = GRASS
        grid[1:3, 2, 2] = WALL
        seg = house_stats.segment(schem_voxels.VoxelGrid(grid, list(NAMES)))
        self.assertEqual(seg.reject, "too_small")


class StoreyTest(unittest.TestCase):

    def test_floor_plates_found(self):
        voxels, spec = make_house(storeys=3, storey_height=5)
        info = anatomy(voxels)
        # Three floors plus the plate under the roof, at the spacing they were built with.
        self.assertGreaterEqual(len(info["peaks"]), 3)
        gaps = {b - a for a, b in zip(info["peaks"], info["peaks"][1:])}
        self.assertEqual(gaps, {spec["storey_height"]})

    def test_single_storey(self):
        voxels, _ = make_house(storeys=1, storey_height=6)
        info = anatomy(voxels)
        self.assertTrue(info["peaks"])

    def test_uniform_block_has_no_floors(self):
        """A solid mass has no plate/room alternation, so it is not a house."""
        grid = np.zeros((30, 14, 14), dtype=np.uint16)
        grid[0, :, :] = GRASS
        grid[1:, 2:12, 2:12] = WALL
        info = anatomy(schem_voxels.VoxelGrid(grid, list(NAMES)))
        peaks = info.get("peaks", [])
        self.assertLessEqual(len(peaks), 1)


class RoofTest(unittest.TestCase):

    def test_gable(self):
        voxels, _ = make_house(roof="gable", ridge=6)
        info = anatomy(voxels)
        self.assertEqual(info["roof"]["type"], "gable")
        self.assertEqual(info["roof"]["axis"], "z")

    def test_hip(self):
        voxels, _ = make_house(roof="hip", ridge=9, width=19, depth=17)
        info = anatomy(voxels)
        self.assertEqual(info["roof"]["type"], "hip")

    def test_flat(self):
        voxels, _ = make_house(roof="flat")
        info = anatomy(voxels)
        self.assertEqual(info["roof"]["type"], "flat")

    def test_chimney_does_not_break_the_gable(self):
        """`100__georgian-house`: measuring the ridge at the height map's max finds a 2x2
        chimney covering 1% of the footprint, not a roof."""
        plain, _ = make_house(roof="gable", ridge=6, chimney=False)
        stacked, _ = make_house(roof="gable", ridge=6, chimney=True)
        self.assertEqual(anatomy(plain)["roof"]["type"], "gable")
        with_chimney = anatomy(stacked)["roof"]
        self.assertEqual(with_chimney["type"], "gable")
        self.assertEqual(with_chimney["axis"], "z")

    def test_eave_sits_above_the_top_floor(self):
        """A single-layer wall band means the eave collapsed onto the floor plate."""
        voxels, _ = make_house(storeys=2, storey_height=5, roof="gable", ridge=6)
        info = anatomy(voxels)
        self.assertGreater(info["roof"]["eave_y"], info["peaks"][-1])
        self.assertGreater(info["roof"]["rise"], 1)


class MaterialTest(unittest.TestCase):

    def _materials(self, voxels):
        info = anatomy(voxels)
        roof = info["roof"]
        peaks = [p for p in info["peaks"] if p < roof["eave_y"] - 1] or info["peaks"][:1]
        return house_stats._analyse_materials(
            info["sub_grid"], info["sub"], [_base(n) for n in voxels.names],
            info["footprint"], peaks, roof["eave_y"])

    def test_wall_material_found(self):
        voxels, _ = make_house()
        self.assertEqual(self._materials(voxels)["wall"]["name"], "minecraft:bricks")

    def test_furniture_does_not_become_the_wall(self):
        """The median real build uses 105 blockstates, most of them not structural."""
        voxels, _ = make_house(furniture=25)
        self.assertEqual(self._materials(voxels)["wall"]["name"], "minecraft:bricks")

    def test_roof_material_found(self):
        voxels, _ = make_house(roof="gable")
        self.assertEqual(self._materials(voxels)["roof_stairs"]["name"],
                         "minecraft:stone_brick_stairs")

    def test_glass_is_not_the_wall(self):
        voxels, _ = make_house()
        materials = self._materials(voxels)
        self.assertNotIn("glass", materials["wall"]["name"])
        self.assertEqual(materials["glass"]["name"], "minecraft:glass")


class OpeningTest(unittest.TestCase):

    def test_windows_counted(self):
        voxels, _ = make_house(width=21, depth=17, storeys=1)
        info = anatomy(voxels)
        roof = info["roof"]
        face = house_stats._analyse_face(
            info["sub"], info["sub_grid"], [_base(n) for n in voxels.names],
            "north", info["peaks"][0], roof["eave_y"])
        self.assertIsNotNone(face)
        self.assertGreaterEqual(face["windows"], 1)


def _base(name):
    import block_roles
    return block_roles.normalise(name)


if __name__ == "__main__":
    unittest.main()
