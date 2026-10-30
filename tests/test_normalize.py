"""Normalizing tiny synthetic files: the readers, crop, interior masks, storey
estimate, vocabulary coverage, and an end-to-end normalize run."""
import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


import nbtlib  # noqa: E402
from nbtlib import ByteArray, Compound, Int, IntArray, List, LongArray, Short, String  # noqa: E402

from minecraft import schem_convert  # noqa: E402
import schem_fixtures  # noqa: E402
from dataset import read as R  # noqa: E402


# --------------------------------------------------------------------------- file builders

def varints(values) -> ByteArray:
    raw = b"".join(schem_convert._varint(int(v)) for v in values)
    return ByteArray(np.frombuffer(raw, dtype=np.uint8).astype(np.int8))


def sponge(path, palette: list[str], grid_xyz: np.ndarray, version: int = 2,
           data_version: int = 3465) -> str:
    """Sponge v2 or v3 from an (X, Y, Z) grid of palette indices."""
    w, h, length = grid_xyz.shape
    flat = np.transpose(grid_xyz, (1, 2, 0)).ravel()            # YZX
    pal = Compound({s: Int(i) for i, s in enumerate(palette)})
    dims = {"Width": Short(w), "Height": Short(h), "Length": Short(length)}
    if version == 3:
        root = Compound({"Schematic": Compound({
            "Version": Int(3), "DataVersion": Int(data_version), **dims,
            "Offset": IntArray([0, 0, 0]),
            "Blocks": Compound({"Palette": pal, "Data": varints(flat),
                                "BlockEntities": List[Compound]([])})})})
    else:
        root = Compound({"Version": Int(2), "DataVersion": Int(data_version), **dims,
                         "PaletteMax": Int(len(palette)), "Palette": pal,
                         "BlockData": varints(flat), "Offset": IntArray([0, 0, 0])})
    return schem_fixtures.write_nbt(root, path, "")


def pack_litematic(values, bits: int) -> LongArray:
    """Litematica's bit array: entries may span two longs (the inverse of the reader)."""
    words = [0] * ((len(values) * bits + 63) // 64)
    for i, v in enumerate(values):
        start = i * bits
        w, off = start >> 6, start & 63
        words[w] |= (int(v) << off) & ((1 << 64) - 1)
        if off + bits > 64:
            words[w + 1] |= int(v) >> (64 - off)
    return LongArray([x - (1 << 64) if x >= 1 << 63 else x for x in words])


def litematic_region(position, size, names: list[str], grid_xyz: np.ndarray) -> Compound:
    """`grid_xyz` is indexed from the region's minimum corner, shape abs(size)."""
    entries = []
    for name in names:
        base, props = R.split_state(name)
        entry = Compound({"Name": String(base)})
        if props:
            entry["Properties"] = Compound({k: String(v) for k, v in props.items()})
        entries.append(entry)
    flat = np.transpose(grid_xyz, (1, 2, 0)).ravel()            # YZX, x fastest
    xyz = lambda v: Compound({"x": Int(v[0]), "y": Int(v[1]), "z": Int(v[2])})  # noqa: E731
    return Compound({
        "Position": xyz(position), "Size": xyz(size),
        "BlockStatePalette": List[Compound](entries),
        "BlockStates": pack_litematic(flat, R.litematic_bits(len(names))),
        "TileEntities": List[Compound]([]), "Entities": List[Compound]([]),
    })


def build_of(grid: np.ndarray, palette: list[str]) -> R.Build:
    return R.Build(blocks=grid.astype(np.uint16), palette=palette, data_version=3465,
                   source_format="test", raw_non_air=int(np.count_nonzero(grid)))


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)


# --------------------------------------------------------------------------- readers

class LitematicTests(TempCase):
    def test_multi_region_negative_size_spanning_bits(self):
        # Region A: 4x2x3 = 24 entries at 3 bits -> entry 21 spans two longs.
        names_a = ["minecraft:air", "minecraft:stone", "minecraft:bricks",
                   "minecraft:oak_stairs[half=bottom,facing=east]", "minecraft:glass"]
        a = np.arange(24).reshape(4, 2, 3) % 5
        # Region B: Size (-2, 2, -3) at Position (5, 1, 3): x 4..5, y 1..2, z 1..3.
        names_b = ["minecraft:air", "minecraft:gold_block"]
        b = np.ones((2, 2, 3), dtype=int)
        root = Compound({
            "Version": Int(6), "MinecraftDataVersion": Int(3465),
            "Metadata": Compound({"EnclosingSize": Compound({"x": Int(6), "y": Int(3),
                                                             "z": Int(4)})}),
            "Regions": Compound({
                "A": litematic_region((0, 0, 0), (4, 2, 3), names_a, a),
                "B": litematic_region((5, 1, 3), (-2, 2, -3), names_b, b)}),
        })
        path = schem_fixtures.write_nbt(root, self.tmp / "t.litematic")
        build = R.load_build(path)

        self.assertEqual(build.dims, (6, 3, 4))
        self.assertEqual(build.regions, 2)
        self.assertEqual(build.source_format, "litematic_v6")
        self.assertEqual(build.warnings, [])
        state = lambda x, y, z: build.palette[build.blocks[x, y, z]]  # noqa: E731
        for (x, y, z), v in np.ndenumerate(a):                     # A: x 0..3, y 0..1, z 0..2
            self.assertEqual(state(x, y, z), R.modern_state(names_a[v], 3465), (x, y, z))
        for x in (4, 5):                                          # B: x 4..5, y 1..2, z 1..3
            for y in (1, 2):
                for z in (1, 2, 3):
                    self.assertEqual(state(x, y, z), "minecraft:gold_block")
        self.assertEqual(state(5, 0, 0), R.AIR)              # outside both regions
        self.assertEqual(state(4, 1, 0), R.AIR)
        self.assertIn("minecraft:oak_stairs[facing=east,half=bottom]", build.palette)
        self.assertEqual(build.raw_non_air, build.non_air())
        self.assertEqual(build.non_air(), int(np.count_nonzero(a)) + 12)

    def test_unpack_matches_pack(self):
        rng = np.random.default_rng(0)
        for bits in (2, 3, 5, 7, 13):
            values = rng.integers(0, 1 << bits, size=300)
            out = R.unpack_litematic(pack_litematic(values, bits), bits, len(values))
            np.testing.assert_array_equal(out, values)

    def test_bits(self):
        self.assertEqual([R.litematic_bits(n) for n in (1, 2, 4, 5, 16, 17)], [2, 2, 2, 3, 4, 5])


class LegacyTests(TempCase):
    def test_unknown_ids_are_counted_and_stay_solid(self):
        path = schem_fixtures.legacy_house(self.tmp / "house.schematic")
        build = R.load_build(path)
        self.assertEqual(build.source_format, "mcedit")
        self.assertEqual(build.unknown_legacy, {"700:0": 1})
        self.assertEqual(build.palette[build.blocks[3, 3, 3]], "legacy:unknown_700")
        self.assertNotIn("minecraft:stone", build.palette)     # not silently stone either
        self.assertEqual(build.dims, (5, 4, 5))
        self.assertEqual(build.palette[build.blocks[0, 0, 0]], "minecraft:cobblestone")
        self.assertTrue(build.palette[build.blocks[2, 1, 0]].startswith("minecraft:oak_door["))
        self.assertTrue(any("entities" in w for w in build.warnings))


class SpongeTests(TempCase):
    def test_v2_and_v3_read_the_same_grid(self):
        palette = ["minecraft:air"] + [f"minecraft:oak_stairs[facing=east,half=bottom,n={i}]"
                                       for i in range(140)]       # >128: 2-byte varints
        rng = np.random.default_rng(1)
        grid = rng.integers(0, len(palette), size=(5, 3, 4))
        v2 = R.load_build(sponge(self.tmp / "a.schem", palette, grid, 2))
        v3 = R.load_build(sponge(self.tmp / "b.schem", palette, grid, 3))
        self.assertEqual((v2.source_format, v3.source_format), ("sponge_v2", "sponge_v3"))
        for build in (v2, v3):
            self.assertEqual(build.dims, (5, 3, 4))
            decoded = np.vectorize(lambda i: build.palette[i])(build.blocks)
            want = np.vectorize(lambda i: R.modern_state(palette[i], 3465))(grid)
            np.testing.assert_array_equal(decoded, want)
        self.assertEqual(v2.palette, v3.palette)

    def test_modernised_names(self):
        self.assertEqual(R.modern_state("minecraft:stone_slab[type=bottom]", 1631),
                         "minecraft:smooth_stone_slab[type=bottom]")
        self.assertEqual(R.modern_state("minecraft:stone_slab[type=bottom]", 3465),
                         "minecraft:stone_slab[type=bottom]")
        self.assertEqual(R.modern_state("minecraft:sign[rotation=4]", 1631),
                         "minecraft:oak_sign[rotation=4]")
        self.assertEqual(R.modern_state("minecraft:cave_air", 3465), R.AIR)
        self.assertEqual(R.modern_state("minecraft:oak_stairs[half=top,facing=west]", 3465),
                         "minecraft:oak_stairs[facing=west,half=top]")
        self.assertEqual(R.modern_state("conquest:stone", 3465), "conquest:stone")

    def test_validate(self):
        palette = ["minecraft:air", "minecraft:stone"]
        grid = np.zeros((4, 2, 3), dtype=int)
        grid[1, 0, 1] = 1
        build = R.load_build(sponge(self.tmp / "c.schem", palette, grid))
        row = {"dimensions_xyz_blocks": [3, 2, 4], "non_air_blocks_from_metadata": 1}
        v = R.validate(build, row)
        self.assertEqual((v["dims"], v["non_air"]), ("xz_swapped", "ok"))
        v = R.validate(build, {"dimensions_xyz_blocks": [4, 2, 3],
                               "non_air_blocks_from_metadata": 2})
        self.assertEqual((v["dims"], v["non_air"]), ("ok", "mismatch"))


# --------------------------------------------------------------------------- crop

PALETTE = ["minecraft:air", "minecraft:grass_block[snowy=false]", "minecraft:dirt",
           "minecraft:oak_planks", "minecraft:stone_bricks", "minecraft:poppy",
           "minecraft:oak_log[axis=y]", "minecraft:oak_leaves[distance=1,persistent=true]",
           "minecraft:glass", "minecraft:white_wool"]
GRASS, DIRT, PLANKS, BRICKS, POPPY, LOG, LEAVES, GLASS, WOOL = range(1, 10)


def plate(x=30, y=14, z=30) -> np.ndarray:
    g = np.zeros((x, y, z), dtype=np.uint16)
    g[:, 0, :] = DIRT
    g[:, 1, :] = GRASS                                  # ground level 1
    return g


def house(g, x0, z0, size=7, height=6, door=False):
    """Hollow box on the ground: planks walls, brick floor and roof, glass windows."""
    x1, z1, y0, y1 = x0 + size - 1, z0 + size - 1, 2, 2 + height - 1
    g[x0:x1 + 1, y0, z0:z1 + 1] = BRICKS
    g[x0:x1 + 1, y1, z0:z1 + 1] = BRICKS
    for y in range(y0 + 1, y1):
        g[x0:x1 + 1, y, z0] = g[x0:x1 + 1, y, z1] = PLANKS
        g[x0, y, z0:z1 + 1] = g[x1, y, z0:z1 + 1] = PLANKS
    g[x0, y0 + 2, z0 + 3] = GLASS
    if door:
        g[x0 + 3, y0 + 1:y0 + 3, z0] = 0
    return g


def tree(g, x, z):
    g[x, 2:7, z] = LOG
    g[x - 2:x + 3, 6:9, z - 2:z + 3] = np.where(g[x - 2:x + 3, 6:9, z - 2:z + 3] == 0,
                                                LEAVES, g[x - 2:x + 3, 6:9, z - 2:z + 3])


class CropTests(unittest.TestCase):
    def test_removes_plate_and_landscape_keeps_house(self):
        from dataset.crop import PAD, crop_build
        g = house(plate(), 10, 10)
        g[2, 2, 2] = g[25, 2, 25] = POPPY
        tree(g, 24, 5)
        result = crop_build(build_of(g, PALETTE))
        kept = {result.palette[i] for i in np.unique(result.blocks)}
        self.assertEqual(result.stats["ground_level"], 1)
        self.assertNotIn(PALETTE[GRASS], kept)
        self.assertNotIn(PALETTE[DIRT], kept)
        self.assertNotIn(PALETTE[POPPY], kept)
        self.assertNotIn(PALETTE[LOG], kept)
        self.assertNotIn(PALETTE[LEAVES], kept)
        self.assertEqual(int((result.blocks != 0).sum()), int((g >= PLANKS).sum()
                                                             - (g == POPPY).sum()
                                                             - (g == LOG).sum()
                                                             - (g == LEAVES).sum()))
        self.assertEqual(result.blocks.shape, (7 + 2 * PAD, 6 + 2 * PAD, 7 + 2 * PAD))
        self.assertEqual(result.offset, (10 - PAD, 2 - PAD, 10 - PAD))
        self.assertNotIn("multi_building", result.flags)
        self.assertIn("crop_aggressive", result.flags)      # the plate outweighs the house

    def test_multi_building(self):
        from dataset.crop import crop_build

        def blocks(*corners):
            g = plate(x=40, z=40)
            for x0, z0 in corners:                       # solid 9x9x9 masses (729 blocks)
                g[x0:x0 + 9, 2:11, z0:z0 + 9] = WOOL
            return g

        one = crop_build(build_of(blocks((3, 3)), PALETTE))
        two = crop_build(build_of(blocks((3, 3), (25, 25)), PALETTE))
        touching = crop_build(build_of(blocks((3, 3), (12, 3)), PALETTE))
        self.assertNotIn("multi_building", one.flags)
        self.assertIn("multi_building", two.flags)
        self.assertEqual(two.stats["large_masses"], 2)
        self.assertNotIn("multi_building", touching.flags)      # one mass, side by side

    def test_gravel_and_snow_walls_are_building(self):
        """Builders texture walls with 'natural' blocks; above the ground they stay."""
        from dataset.crop import crop_build
        palette = PALETTE + ["minecraft:gravel", "minecraft:snow_block", "minecraft:snow[layers=1]"]
        gravel, snow_block, snow = len(PALETTE), len(PALETTE) + 1, len(PALETTE) + 2
        g = house(plate(), 10, 10)
        g[g == PLANKS] = gravel
        g[10:17, 7, 10:17] = snow_block
        g[10:17, 8, 10:17] = snow                            # snow lying on the roof
        g[5, 2, 5] = snow                                    # snow lying on the ground
        result = crop_build(build_of(g, palette))
        counts = {s: int((result.blocks == i).sum()) for i, s in enumerate(result.palette)}
        self.assertEqual(counts.get(palette[gravel]), int((g == gravel).sum()))
        self.assertEqual(counts.get(palette[snow]), 49)


# --------------------------------------------------------------------------- interior

class InteriorTests(unittest.TestCase):
    def masks(self, door: bool):
        from dataset.crop import crop_build
        from dataset.interior import compute_masks
        result = crop_build(build_of(house(plate(), 10, 10, door=door), PALETTE))
        return result, compute_masks(result.blocks, result.ground)

    def test_hollow_box(self):
        result, m = self.masks(door=False)
        room = 5 * 4 * 5
        self.assertEqual(m.stats["interior_air"], room)
        self.assertEqual(int((m.shell.astype(int) + m.exterior + m.interior != 1).sum()), 0)
        self.assertEqual(int((m.shell & (result.blocks == 0)).sum()), 0)
        self.assertEqual(m.flags, [])

    def test_doorway_does_not_leak_the_room(self):
        _, closed = self.masks(door=False)
        result, m = self.masks(door=True)
        # A 1-wide doorway: only the doorway itself is exterior (a reveal), the room is not.
        self.assertGreaterEqual(m.stats["interior_air"], 0.9 * closed.stats["interior_air"])
        self.assertNotIn("large_opening", m.flags)
        # interior blocks are masked, never deleted
        self.assertEqual(int((result.blocks != 0).sum()),
                         int(((result.blocks != 0) & (m.shell | m.interior)).sum()))


class StoreyTests(unittest.TestCase):
    def test_floors_every_four_blocks(self):
        from dataset.crop import crop_build
        from dataset.interior import compute_masks
        from dataset.storey import estimate
        g = plate(y=24)
        for y in (2, 6, 10, 14, 18):                     # floors and a flat roof
            g[8:17, y, 8:17] = BRICKS
        for y in range(3, 18):
            g[8, y, 8:17] = g[16, y, 8:17] = PLANKS
            g[8:17, y, 8] = g[8:17, y, 16] = PLANKS
        result = crop_build(build_of(g, PALETTE))
        m = compute_masks(result.blocks, result.ground)
        est = estimate(result.blocks, result.palette, m, m.stats["ground_y"])
        self.assertEqual(est["blocks_per_storey"], 4)
        self.assertEqual(est["confidence"], "high")


# --------------------------------------------------------------------------- coverage

class CoverageTests(unittest.TestCase):
    def test_known_states(self):
        from dataset.coverage import classify, classify_v2
        cases = {   # state: (v3 class, v3 reason or family, v2 class)
            "minecraft:oak_stairs[facing=east,half=bottom,shape=outer_left]": ("exact", "stairs", "exact"),
            "minecraft:oak_stairs[facing=east,half=top,shape=straight]": ("exact", "stairs_upside_down", "lossy"),
            "minecraft:stone_bricks": ("exact", "full", "exact"),
            "minecraft:basalt[axis=x]": ("exact", "full_sideways", "lossy"),
            "minecraft:waxed_oxidized_copper": ("lossy", "alias", "lossy"),
            "minecraft:glass_pane[east=true]": ("exact", "pane", "ignore"),
            "minecraft:oak_door[facing=north,half=lower]": ("exact", "door", "ignore"),
            "minecraft:oak_door[facing=north,half=upper]": ("derived", "door", "ignore"),
            "minecraft:iron_trapdoor[half=top]": ("exact", "trapdoor", "ignore"),
            "minecraft:nether_brick_fence": ("exact", "fence", "ignore"),
            "minecraft:smooth_quartz": ("exact", "full", "ignore"),
            "minecraft:sea_lantern": ("exact", "full", "ignore"),
            "minecraft:smooth_quartz_stairs[facing=west,half=bottom]": ("exact", "stairs", "ignore"),
            "minecraft:smooth_quartz_slab[type=double]": ("lossy", "double_slab", "ignore"),
            "minecraft:poppy": ("ignore", "decor:plants", "ignore"),
            "minecraft:chest[facing=north]": ("ignore", "decor:furniture", "ignore"),
            "minecraft:piston_head[facing=up]": ("ignore", "decor:redstone_rails", "ignore"),
            "minecraft:wall_torch[facing=north]": ("ignore", "decor:lighting", "ignore"),
            "minecraft:snow[layers=1]": ("ignore", "decor:snow_layers", "ignore"),
            "minecraft:gilded_blackstone": ("ignore", "full", "ignore"),
            "minecraft:bamboo_mosaic": ("exact", "full", "ignore"),          # appended in 0.6.1
            "minecraft:redstone_ore[lit=true]": ("lossy", "lit", "ignore"),  # appended in 0.6.1
            "legacy:unknown_700": ("ignore", None, "ignore"),
        }
        for state, (cls, detail, v2) in cases.items():
            got = classify(state)
            self.assertEqual(got["class"], cls, state)
            if detail:
                self.assertIn(detail, got["reasons"] + [got["family"]], state)
            self.assertEqual(classify_v2(state), v2, state)
        self.assertEqual(classify(R.AIR)["class"], "air")


# --------------------------------------------------------------------------- end to end

class NormalizeTests(TempCase):
    def make_dataset(self):
        ds = self.tmp / "ds"
        src = ds / "source"
        src.mkdir(parents=True)
        g = house(plate(), 10, 10)
        g[2, 2, 2] = POPPY
        sponge(src / "a.schem", PALETTE, g)
        schem_fixtures.legacy_house(src / "b.schematic")
        rows = []
        for i, (name, ftype) in enumerate((("a.schem", ".schem"), ("b.schematic", ".schematic"))):
            sha = hashlib.sha256((src / name).read_bytes()).hexdigest()
            dims = list(R.load_build(src / name).dims)
            rows.append({"id": f"b{i}", "file": name, "sha256": sha, "file_type": ftype,
                         "building_category": "residential/houses", "split": "train",
                         "licence_tier": "B", "flags": [], "dimensions_xyz_blocks": dims,
                         "non_air_blocks_from_metadata": None})
        rows.append({**rows[0], "id": "bx", "split": "excluded", "licence_tier": "X"})
        (ds / "index.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
        return ds

    def run_cli(self, ds, *extra):
        from dataset import normalize
        with contextlib.redirect_stdout(io.StringIO()) as log:
            code = normalize.main(["--dataset", str(ds), "--no-review", *extra])
        return code, log.getvalue()

    def test_end_to_end_resumable_deterministic(self):
        ds = self.make_dataset()
        before = {p: p.read_bytes() for p in (ds / "source").iterdir()}
        code, log = self.run_cli(ds)
        self.assertEqual(code, 0, log)
        index = json.loads((ds / "normalized" / "index.json").read_text("utf-8"))
        self.assertEqual([b["id"] for b in index["builds"]], ["b0", "b1"])   # bx excluded
        self.assertEqual({b["status"] for b in index["builds"]}, {"ok"})
        self.assertEqual(index["summary"]["validation"]["unknown_legacy"], {"b1": {"700:0": 1}})
        with np.load(ds / "normalized" / "b0.npz") as z:
            self.assertEqual(set(z.files), {"blocks", "palette", "shell", "exterior",
                                            "interior", "crop_offset", "ground_height",
                                            "ground_y", "target_shape", "target_material"})
            self.assertEqual(list(z["crop_offset"]), [8, 0, 8])
            from voxel import vocab
            shape, mat = z["target_shape"], z["target_material"]
            self.assertTrue((shape[z["interior"]] == vocab.IGNORE_SHAPE).all())
            self.assertTrue((shape[z["exterior"]] == 0).all())
            sup = z["shell"] & (shape != vocab.IGNORE_SHAPE)
            self.assertTrue(vocab.VALID[shape[sup], mat[sup]].all())
        cov = json.loads((ds / "vocab_coverage.json").read_text("utf-8"))
        self.assertEqual(cov["builds"], 2)
        self.assertEqual(cov["vocab_version"], "mcvox-v3")
        self.assertEqual({p: p.read_bytes() for p in (ds / "source").iterdir()}, before)

        first = (ds / "normalized" / "b0.npz").read_bytes()
        code, log = self.run_cli(ds)
        self.assertIn("0 to process", log)                      # resumable
        code, log = self.run_cli(ds, "--force")
        self.assertIn("2 to process", log)
        self.assertEqual((ds / "normalized" / "b0.npz").read_bytes(), first)   # deterministic


if __name__ == "__main__":
    unittest.main()
