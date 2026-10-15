"""Phase 4 pairs: meshify geometry and exterior-only faces, the 2x input frame, the pair
validator, and determinism (same seed -> same bytes)."""
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

SKIP = "mcvoxel (requirements-synth.txt) not installed"


def single(state, derive=True):
    """One block at (2, 2, 2) in open air -> (mesh, info)."""
    from schemset.meshify import meshify
    blocks = np.zeros((5, 5, 5), dtype=np.uint16)
    blocks[2, 2, 2] = 1
    return meshify(blocks, ["minecraft:air", state], blocks == 0, None, derive=derive)


def quads(mesh):
    """(centres, normals, areas) of the mesh's quads (4 vertices, 2 triangles each)."""
    c = mesh.verts.reshape(-1, 4, 3)
    n = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
    area = np.linalg.norm(n, axis=1)
    return c.mean(axis=1), n / area[:, None], area


HOUSE_PAL = ["minecraft:air", "minecraft:stone_bricks", "minecraft:oak_planks",
             "minecraft:oak_stairs[facing=east,half=bottom,shape=straight]",
             "minecraft:oak_stairs[facing=west,half=bottom,shape=straight]",
             "minecraft:glass_pane[east=true,north=false,south=false,west=true]",
             "minecraft:oak_door[facing=south,half=lower,hinge=left,open=false]",
             "minecraft:oak_door[facing=south,half=upper,hinge=left,open=false]",
             "minecraft:chest[facing=north]", "minecraft:oak_slab[type=bottom]"]


def house():
    """A small house on flat ground: walls, a stair roof, a pane window, a door, a chest
    inside. Returns the arrays of a normalized npz (X, Y, Z)."""
    from mcvoxel import vocab
    from schemset.interior import compute_masks
    g = np.zeros((16, 14, 14), dtype=np.uint16)
    x0, x1, z0, z1, y0, y1 = 3, 12, 3, 10, 1, 5
    g[x0:x1 + 1, y0, z0:z1 + 1] = 2                       # floor
    for y in range(y0 + 1, y1 + 1):
        g[x0:x1 + 1, y, z0] = g[x0:x1 + 1, y, z1] = 1
        g[x0, y, z0:z1 + 1] = g[x1, y, z0:z1 + 1] = 1
    for k in range(4):                                     # a stair roof, solid attic
        y = y1 + 1 + k
        g[x0 + k:x1 + 1 - k, y, z0:z1 + 1] = 2
        g[x0 - 1 + k, y, z0 - 1:z1 + 2] = 3                # rises to the east
        g[x1 + 1 - k, y, z0 - 1:z1 + 2] = 4                # rises to the west
    g[x0 + 3:x1 - 2, y1 + 5, z0 - 1:z1 + 2] = 9            # slab ridge
    g[6, 3, z0] = 5                                        # a pane window in the north wall
    g[9, 2, z1], g[9, 3, z1] = 6, 7                        # a door in the south wall
    g[7, 2, 6] = 8                                         # a chest in the room
    height = np.zeros((16, 14), dtype=np.int16)            # ground: y <= 0
    masks = compute_masks(g, height)
    ts, tm = vocab.encode_target(g, HOUSE_PAL, masks.shell, masks.exterior, masks.interior)
    return {"blocks": g, "palette": np.array(HOUSE_PAL), "shell": masks.shell,
            "exterior": masks.exterior, "interior": masks.interior,
            "crop_offset": np.zeros(3, dtype=np.int32), "ground_height": height,
            "ground_y": np.int32(0), "target_shape": ts, "target_material": tm}


@unittest.skipUnless(HAVE_MCVOXEL, SKIP)
class MeshifyGeometry(unittest.TestCase):
    """Real block geometry, in the build's own frame (block (x, y, z) = [x, x+1) ...)."""

    def bounds(self, state):
        mesh, _ = single(state)
        return np.round(mesh.verts.min(axis=0), 4).tolist(), np.round(mesh.verts.max(axis=0), 4).tolist()

    def test_cube(self):
        mesh, info = single("minecraft:stone")
        self.assertEqual(self.bounds("minecraft:stone"), ([2, 2, 2], [3, 3, 3]))
        self.assertEqual(info["quads"]["block"], 6)

    def test_slabs(self):
        self.assertEqual(self.bounds("minecraft:oak_slab[type=bottom]"), ([2, 2, 2], [3, 2.5, 3]))
        self.assertEqual(self.bounds("minecraft:oak_slab[type=top]"), ([2, 2.5, 2], [3, 3, 3]))

    def test_stairs_bottom_half(self):
        mesh, _ = single("minecraft:oak_stairs[facing=east,half=bottom,shape=straight]")
        self.assertEqual(self.bounds("minecraft:oak_stairs[facing=east,half=bottom,shape=straight]"),
                         ([2, 2, 2], [3, 3, 3]))
        c, n, a = quads(mesh)
        up = n[:, 1] > 0.99
        top = up & np.isclose(c[:, 1], 3.0)                # the high step, east half
        tread = up & np.isclose(c[:, 1], 2.5)              # the low tread, west half
        self.assertAlmostEqual(float(a[top].sum()), 0.5)
        self.assertAlmostEqual(float(c[top, 0].mean()), 2.75)
        self.assertAlmostEqual(float(a[tread].sum()), 0.5)
        self.assertAlmostEqual(float(c[tread, 0].mean()), 2.25)

    def test_stairs_top_half(self):
        mesh, _ = single("minecraft:oak_stairs[facing=east,half=top,shape=straight]")
        c, n, a = quads(mesh)
        down = n[:, 1] < -0.99
        low = down & np.isclose(c[:, 1], 2.0)              # under the step: east half
        under = down & np.isclose(c[:, 1], 2.5)            # under the slab: west half
        self.assertAlmostEqual(float(a[low].sum()), 0.5)
        self.assertAlmostEqual(float(c[low, 0].mean()), 2.75)
        self.assertAlmostEqual(float(a[under].sum()), 0.5)
        self.assertAlmostEqual(float(c[under, 0].mean()), 2.25)

    def test_pane_is_thin(self):
        lo, hi = self.bounds("minecraft:glass_pane[east=false,north=true,south=true,west=false]")
        self.assertEqual((lo, hi), ([2 + 7 / 16, 2, 2], [2 + 9 / 16, 3, 3]))
        from mcvoxel.scene import KIND_GLASS
        mesh, _ = single("minecraft:glass_pane[east=false,north=true,south=true,west=false]")
        self.assertTrue((mesh.face_mat == KIND_GLASS).all())

    def test_open_and_closed_trapdoors(self):
        # open, facing south: the plate stands against the north face
        self.assertEqual(self.bounds("minecraft:oak_trapdoor[facing=south,half=bottom,open=true]"),
                         ([2, 2, 2], [3, 3, 2 + 3 / 16]))
        self.assertEqual(self.bounds("minecraft:oak_trapdoor[facing=south,half=top,open=false]"),
                         ([2, 2 + 13 / 16, 2], [3, 3, 3]))

    def test_door(self):
        from mcvoxel.scene import KIND_DOOR
        state = "minecraft:oak_door[facing=north,half=lower,hinge=left,open=false]"
        self.assertEqual(self.bounds(state), ([2, 2, 2 + 13 / 16], [3, 3, 3]))
        mesh, _ = single(state)
        self.assertTrue((mesh.face_mat == KIND_DOOR).all())

    def test_fence_wall_carpet(self):
        lo, hi = self.bounds("minecraft:oak_fence[east=false,north=false,south=false,west=false]")
        self.assertEqual((lo, hi), ([2 + 6 / 16, 2, 2 + 6 / 16], [2 + 10 / 16, 3, 2 + 10 / 16]))
        lo, hi = self.bounds("minecraft:cobblestone_wall[east=low,north=none,south=none,up=false,west=low]")
        self.assertEqual((lo, hi), ([2, 2, 2 + 5 / 16], [3, 2 + 14 / 16, 2 + 11 / 16]))
        self.assertEqual(self.bounds("minecraft:white_carpet"), ([2, 2, 2], [3, 2 + 1 / 16, 3]))

    def test_decor_is_skipped_but_leaves_are_cubes(self):
        mesh, _ = single("minecraft:torch")
        self.assertEqual(len(mesh.faces), 0)
        self.assertEqual(self.bounds("minecraft:oak_leaves[distance=1,persistent=true]"), ([2, 2, 2], [3, 3, 3]))

    def test_legacy_pane_gets_its_connections(self):
        from schemset.meshify import meshify
        g = np.zeros((7, 5, 5), dtype=np.uint16)
        g[1, 2, 2] = g[5, 2, 2] = 1
        g[2:5, 2, 2] = 2
        mesh, info = meshify(g, ["minecraft:air", "minecraft:stone", "minecraft:glass_pane"], g == 0, None)
        self.assertEqual(info["derived"]["connections"], 3)
        c, n, a = quads(mesh)
        glass = mesh.face_mat.reshape(-1, 2)[:, 0] == 1
        # the pane's faces run unbroken along x from stone to stone
        side = glass & (np.abs(n[:, 2]) > 0.99)
        self.assertAlmostEqual(float(a[side].sum()), 6.0)


@unittest.skipUnless(HAVE_MCVOXEL, SKIP)
class MeshifyOutsideOnly(unittest.TestCase):

    def test_interior_blocks_produce_no_faces(self):
        from schemset.meshify import meshify
        h = house()
        mesh, info = meshify(h["blocks"], h["palette"], h["exterior"], h["ground_height"])
        c, n, a = quads(mesh)
        # strictly inside the room (walls are x 3..12, z 3..10, floor y 1, wall top y 5)
        inside = ((c[:, 0] > 4.01) & (c[:, 0] < 11.99) & (c[:, 2] > 4.01) & (c[:, 2] < 9.99)
                  & (c[:, 1] > 2.01) & (c[:, 1] < 5.99))
        self.assertEqual(int(inside.sum()), 0)
        chest = (np.abs(c[:, 0] - 7.5) < 0.6) & (np.abs(c[:, 2] - 6.5) < 0.6) & (c[:, 1] < 3.1)
        self.assertEqual(int(chest.sum()), 0)
        self.assertGreater(info["quads"]["internal"], 0)    # the recessed pane, the stair treads

    def test_no_ground_plate(self):
        from schemset.meshify import meshify
        h = house()
        mesh, _ = meshify(h["blocks"], h["palette"], h["exterior"], h["ground_height"])
        self.assertGreaterEqual(float(mesh.verts[:, 1].min()), 1.0)   # ground is y <= 0
        c, n, a = quads(mesh)
        self.assertEqual(int(((n[:, 1] < -0.99) & np.isclose(c[:, 1], 1.0)).sum()), 0)

    def test_path_in_the_ground_layer_is_a_closed_slab(self):
        """A path paved into the terrain: its top in the block's colour, its sides and bottom
        against the ground dark -- a slab, not a sheet degradation would erase."""
        from schemset.meshify import CAP_RGB, meshify
        g = np.zeros((8, 5, 5), dtype=np.uint16)
        g[2:6, 1, 2] = 1                                   # ground is y <= 1
        height = np.ones((8, 5), dtype=np.int64)
        exterior = (g == 0) & (np.arange(5)[None, :, None] > 1)
        mesh, info = meshify(g, ["minecraft:air", "minecraft:stone"], exterior, height)
        self.assertEqual(info["quads"]["block"], 4)        # the top of each path block
        self.assertGreater(info["quads"]["ground"], 0)
        lo, hi = mesh.verts.min(axis=0), mesh.verts.max(axis=0)
        self.assertEqual((lo.tolist(), hi.tolist()), ([2, 1, 2], [6, 2, 3]))
        c, n, a = quads(mesh)
        dark = np.all(np.isclose(mesh.vcolor.reshape(-1, 4, 3)[:, 0], CAP_RGB), axis=1)
        self.assertTrue(dark[n[:, 1] < -0.99].all())       # the bottom is dark
        self.assertFalse(dark[n[:, 1] > 0.99].any())       # the top is stone

    def test_doorway_gets_a_dark_cap(self):
        from schemset.meshify import CAP_RGB, meshify
        g = np.zeros((9, 7, 9), dtype=np.uint16)
        g[2:7, 1, 2:7] = 1
        for y in range(2, 5):
            g[2:7, y, 2] = g[2:7, y, 6] = 1
            g[2, y, 2:7] = g[6, y, 2:7] = 1
        g[2:7, 5, 2:7] = 1
        g[4, 2:4, 2] = 0                                   # a 1 x 2 doorway, no door
        from schemset.interior import compute_masks
        masks = compute_masks(g, 0)
        mesh, info = meshify(g, ["minecraft:air", "minecraft:stone"], masks.exterior, np.zeros((9, 9), int))
        self.assertEqual(info["quads"]["cap"], 2)
        caps = np.all(np.isclose(mesh.vcolor.reshape(-1, 4, 3)[:, 0], CAP_RGB), axis=1)
        self.assertEqual(int(caps.sum()), 2)

    def test_cobweb_in_the_doorway_keeps_the_room_closed(self):
        """Decor is not drawn, but it must not carry the outside into a roofed room."""
        from schemset.interior import compute_masks
        from schemset.meshify import meshify
        g = np.zeros((9, 7, 9), dtype=np.uint16)
        g[2:7, 1, 2:7] = 1
        for y in range(2, 5):
            g[2:7, y, 2] = g[2:7, y, 6] = 1
            g[2, y, 2:7] = g[6, y, 2:7] = 1
        g[2:7, 5, 2:7] = 1
        g[4, 2:4, 2] = 2                                   # cobwebs fill the doorway
        g[4, 2, 3] = 2                                     # ...and one more just inside
        masks = compute_masks(g, 0)
        mesh, info = meshify(g, ["minecraft:air", "minecraft:stone", "minecraft:cobweb"],
                             masks.exterior, np.zeros((9, 9), int))
        c, n, a = quads(mesh)
        inside = ((c[:, 0] > 3.01) & (c[:, 0] < 5.99) & (c[:, 2] > 3.01) & (c[:, 2] < 5.99)
                  & (c[:, 1] > 2.01) & (c[:, 1] < 4.99))        # the room: x, z 3..6, y 2..5
        self.assertEqual(int(inside.sum()), 0)
        self.assertEqual(info["quads"]["cap"], 2)          # the room behind the doorway, dark
        self.assertEqual(info["opened_by_decor"], 2)       # the doorway's cobwebs, not the one inside

    def test_air_enclosed_by_decor_is_outside(self):
        """Cobwebs (decor, not drawn) around a pocket of air the masks call interior, with
        no roof above it: all of it is outside -- no caps floating in the sky."""
        from schemset.interior import compute_masks
        from schemset.meshify import meshify
        g = np.zeros((9, 9, 9), dtype=np.uint16)
        g[2:7, 2:7, 2:7] = 1                               # a cube of cobwebs...
        g[3:6, 3:6, 3:6] = 0                               # ...around a pocket of air
        masks = compute_masks(g, None)
        self.assertFalse(masks.exterior[4, 4, 4])          # interior, to the masks
        mesh, info = meshify(g, ["minecraft:air", "minecraft:cobweb"], masks.exterior, None)
        self.assertEqual(len(mesh.faces), 0)
        self.assertEqual(info["opened_by_decor"], 125)     # 98 cobwebs + the 27-voxel pocket


@unittest.skipUnless(HAVE_MCVOXEL, SKIP)
class InputFrame(unittest.TestCase):
    """D16: input = SCALE x the target grid, same origin: cell i spans [i/2, (i+1)/2) blocks."""

    def test_two_x_same_origin(self):
        from schemset.meshify import meshify
        from schemset.pairs import SCALE, place, voxelize_input
        g = np.zeros((6, 5, 7), dtype=np.uint16)
        g[3, 2, 4] = 1
        mesh, _ = meshify(g, ["minecraft:air", "minecraft:stone"], g == 0, None)
        for off in (0.25, -0.25):
            vm = voxelize_input(place(mesh, 0.0, (off, off, off)), g.shape, colors=False)
            self.assertEqual(vm.solid.shape, tuple(SCALE * s for s in g.shape))
            for axis, block in enumerate((3, 2, 4)):
                cells = np.flatnonzero(vm.solid.any(axis=tuple(a for a in range(3) if a != axis)))
                lo = int(np.floor(SCALE * (block + off)))
                self.assertEqual(cells.tolist(), [lo, lo + 1, lo + 2], (axis, off))

    def test_known_offset_and_yaw_within_tolerance(self):
        from schemset.meshify import meshify
        from schemset.pairs import SCALE, frame_shift, place, target_volume, voxelize_input
        h = house()
        mesh, _ = meshify(h["blocks"], h["palette"], h["exterior"], h["ground_height"])
        vol = target_volume(h["target_shape"], h["exterior"], h["ground_height"])
        vol2 = vol.repeat(2, 0).repeat(2, 1).repeat(2, 2)
        for yaw, off in ((0.0, (0.3, 0.1, -0.2)), (4.0, (-0.4, 0.15, 0.35))):
            vm = voxelize_input(place(mesh, yaw, off), h["blocks"].shape, colors=False)
            shift = frame_shift(vm.solid, vol2)
            for s, o in zip(shift, off):
                self.assertLess(abs(s - SCALE * o), 0.75, (yaw, off, shift))


@unittest.skipUnless(HAVE_MCVOXEL, SKIP)
class PairValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from schemset.meshify import meshify
        from schemset.pairs import make_input, sample_variant
        cls.h = house()
        mesh, _ = meshify(cls.h["blocks"], cls.h["palette"], cls.h["exterior"], cls.h["ground_height"])
        var = sample_variant(1234)
        vm, deg, info, cls.reference = make_input(mesh, cls.h["blocks"].shape, var)
        cls.pair = {"input_occ": vm.solid.astype(np.uint8), "input_surface": vm.surface.astype(np.uint8),
                    "input_rgb": vm.rgb.astype(np.uint8), "target_shape": cls.h["target_shape"],
                    "target_material": cls.h["target_material"],
                    "ground_height": cls.h["ground_height"].astype(np.int16)}

    def validate(self, pair, reference=None):
        from schemset.pairs import validate_pair
        return validate_pair(pair, self.h["exterior"], self.reference if reference is None else reference)

    def test_good_pair_passes(self):
        failures, m = self.validate(self.pair)
        self.assertEqual(failures, [])
        self.assertGreater(m["iou"], 0.35)

    def test_misaligned_pair_is_rejected(self):
        bad = dict(self.pair)
        bad["input_occ"] = np.roll(self.pair["input_occ"], 4, axis=0)     # two blocks east
        failures, m = self.validate(bad)
        self.assertTrue(any(f.startswith("frame_misaligned") for f in failures), failures)

    def test_lost_building_is_rejected(self):
        bad = dict(self.pair)
        occ = np.zeros_like(self.pair["input_occ"])
        occ[:8] = self.pair["input_occ"][:8]               # only the westmost 4 blocks survive
        bad["input_occ"] = occ
        failures, m = self.validate(bad)
        self.assertTrue(any(f.startswith("input_misses_building") for f in failures), failures)

    def test_wrong_resolution_is_rejected(self):
        bad = dict(self.pair)
        bad["input_occ"] = self.pair["input_occ"][::2, ::2, ::2]
        failures, _ = self.validate(bad)
        self.assertTrue(any(f.startswith("shape:input_occ") for f in failures), failures)

    def test_invalid_target_pair_is_rejected(self):
        from mcvoxel import vocab
        bad = dict(self.pair)
        ts = self.pair["target_shape"].copy()
        sup = np.argwhere((ts == vocab.SHAPE_FULL) & (self.h["blocks"] == 1))[0]   # stone bricks
        ts[tuple(sup)] = vocab.SHAPE_INDEX["door_north"]         # a stone-brick door: not VALID
        bad["target_shape"] = ts
        failures, _ = self.validate(bad)
        self.assertTrue(any(f.startswith("target_invalid_pairs") for f in failures), failures)

    def test_frame_identity_catches_a_shifted_mesh(self):
        from schemset.meshify import geometry, meshify
        from schemset.pairs import frame_identity
        h = self.h
        mesh, info = meshify(h["blocks"], h["palette"], h["exterior"], h["ground_height"])
        drawn = np.array([bool(g and g.boxes) for g in map(geometry, h["palette"])])
        shell = h["shell"] & drawn[h["blocks"]]
        self.assertEqual(frame_identity(mesh, info, shell)[0], [])
        moved = mesh.transformed(np.eye(3), (2.0, 0.0, 0.0))            # two blocks east
        self.assertTrue(any(p.startswith("frame_identity:short") for p in frame_identity(moved, info, shell)[0]))
        info_off = dict(info, on_lattice=False)                         # e.g. a 0.3-block offset
        self.assertIn("frame_identity:off_lattice", frame_identity(mesh, info_off, shell)[0])

    def test_wrong_dtype_is_rejected(self):
        bad = dict(self.pair)
        bad["input_rgb"] = self.pair["input_rgb"].astype(np.float32)
        failures, _ = self.validate(bad)
        self.assertTrue(any(f.startswith("dtype:input_rgb") for f in failures), failures)


@unittest.skipUnless(HAVE_MCVOXEL, SKIP)
class Determinism(unittest.TestCase):

    def test_same_seed_same_bytes(self):
        from schemset.normalize import save_npz
        from schemset.pairs import process_build, variant_seed
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "normalized").mkdir()
            save_npz(tmp / "normalized" / "test-house.npz", house())
            row = {"id": "test-house", "split": "train", "category_top": "residential"}
            first = process_build(row, str(tmp), variants=1, measure_mesh=False)
            self.assertEqual(first["status"], "ok", first.get("reason"))
            self.assertEqual(first["variants"][0]["status"], "ok", first["variants"][0].get("detail"))
            path = tmp / "pairs" / "synthetic" / "test-house" / "v0.npz"
            a = path.read_bytes()
            path.unlink()
            second = process_build(row, str(tmp), variants=1, measure_mesh=False)
            self.assertEqual(path.read_bytes(), a)
            self.assertEqual(first["variants"][0]["seed"], variant_seed("test-house", 0))
            self.assertEqual(json.dumps(first["variants"][0]["validation"]),
                             json.dumps(second["variants"][0]["validation"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
