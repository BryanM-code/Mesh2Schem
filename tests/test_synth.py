"""The offline synthetic-pair pipeline (synth/): filters, scale, targets, validation, runner,
review tool. Everything runs on meshes built here -- no network, no real dataset."""
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
    import skimage  # noqa: F401
    HAVE_SYNTH = True
except ImportError:            # the Flask app does not need these; see requirements-synth.txt
    HAVE_SYNTH = False


def _record(**kw):
    rec = {"uid": "u0", "name": "Test House", "license": "by", "author": "tester", "url": "https://example.invalid/m",
           "category": "house", "house_sub": "house:modern", "storeys": None, "stage": "pool", "path": "glbs/x/u0.glb",
           "faces": 20_000, "glb_bytes": 2e6, "caption": "A modern house", "opp_score": 3, "opp_style": "realistic",
           "flags": {}, "tags": [], "pool_kind": "clean", "sha256": None}
    rec.update(kw)
    return rec


def _house_glb(path, width=20.0, depth=14.0, wall_h=6.0, roof_h=7.5, turn_deg=17.0, scale=1.137):
    """Box walls + a 45-degree gable roof + a translucent pane, turned and scaled off-grid."""
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    def coloured(mesh, rgba, alpha_mode="OPAQUE", name="m"):
        mesh.visual = TextureVisuals(material=PBRMaterial(name=name, baseColorFactor=rgba, alphaMode=alpha_mode))
        return mesh

    walls = trimesh.creation.box(extents=(width, wall_h, depth))
    walls.apply_translation((width / 2, wall_h / 2, depth / 2))
    tri = np.array([[0, wall_h, -0.5], [0, wall_h, depth + 0.5], [0, wall_h + roof_h, depth / 2]])
    v = np.vstack([tri, tri + [width, 0, 0]])
    f = np.array([[0, 1, 2], [3, 5, 4], [0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2], [2, 5, 3], [2, 3, 0]])
    roof = trimesh.Trimesh(v, f)
    roof.fix_normals()
    pane = trimesh.creation.box(extents=(3, 2, 0.1))
    pane.apply_translation((width / 2, wall_h / 2, depth + 0.02))
    # Real models are neither axis-aligned nor on round numbers; a box at integer sizes is
    # exactly the primitive "blocky" model the geometry check rejects.
    turn = trimesh.transformations.rotation_matrix(np.radians(turn_deg), [0, 1, 0])
    turn[:3, :3] *= scale
    scene = trimesh.Scene()
    for mesh in (coloured(walls, [220, 210, 190, 255], name="plaster"),
                 coloured(roof, [120, 60, 40, 255], name="tiles"),
                 coloured(pane, [150, 200, 230, 90], "BLEND", name="window_glass")):
        mesh.apply_transform(turn)
        scene.add_geometry(mesh)
    scene.export(str(path))


@unittest.skipUnless(HAVE_SYNTH, "synth dependencies not installed (requirements-synth.txt)")
class Taxonomy(unittest.TestCase):
    def test_head_phrase_decides(self):
        from synth import taxonomy as T
        cases = {
            "A two-story brick house with a gable roof and a garage": "house",
            "A tall modern skyscraper with blue glass": "skyscraper",
            "A multi-story beige residential building with a mansard roof": "apartment",
            "A 12-story building with a light grey facade and balconies": "midrise_generic",
            "Cylindrical lighthouse with four levels": "tower",
            "Old red barn with a gambrel roof and hay loft door": "industrial",
            "A dark-colored four-door SUV with a boxy shape, prominent grille, bull bar": None,
            "A school desk and chair combo with light wood finish": None,
        }
        for cap, want in cases.items():
            self.assertEqual(T.classify(cap, "", [])[0], want, cap)

    def test_groups_and_civic_split(self):
        from synth import taxonomy as T
        self.assertEqual(T.group("apartment"), "residential")
        self.assertEqual(T.group("skyscraper"), "high_rise")
        sub = T.subcategory("institutional", "A red brick town hall with a clock tower")
        self.assertEqual(sub, "civic")
        self.assertEqual(T.group("institutional", sub), "civic")
        self.assertEqual(T.subcategory("institutional", "A three-storey school building"), "education")


@unittest.skipUnless(HAVE_SYNTH, "synth dependencies not installed")
class Filters(unittest.TestCase):
    def test_metadata_reasons(self):
        from synth.config import PipelineConfig
        from synth.filters import REASONS, metadata_verdict
        cfg = PipelineConfig()
        self.assertEqual(metadata_verdict(_record(), cfg), (None, ""))
        self.assertEqual(metadata_verdict(_record(license="by-nc"), cfg)[0], "licence_not_permitted")
        self.assertEqual(metadata_verdict(_record(stage="scene"), cfg)[0], "scene")
        self.assertEqual(metadata_verdict(_record(opp_style="scanned"), cfg)[0], "photogrammetry")
        self.assertEqual(metadata_verdict(_record(caption="A photogrammetry model of a church"), cfg)[0],
                         "photogrammetry")
        self.assertEqual(metadata_verdict(_record(faces=100), cfg)[0], "too_few_faces")
        self.assertEqual(metadata_verdict(_record(path=None), cfg)[0], "no_download_path")
        for rec in (_record(license="by-nc"), _record(stage="scene"), _record(opp_style="scanned")):
            self.assertIn(metadata_verdict(rec, cfg)[0], REASONS)

    def test_object_tags_and_furniture_category(self):
        from synth.config import PipelineConfig
        from synth.filters import metadata_verdict, source_warnings
        cfg = PipelineConfig()
        self.assertEqual(metadata_verdict(_record(name="Low poly cinema chair", tags=["cinema", "chair"]), cfg),
                         ("object_or_set", "chair"))
        self.assertIsNone(metadata_verdict(_record(tags=["chair", "room"]), cfg)[0])
        for name in ("Glass house", "Sports Stadium"):
            rec = _record(name=name, sketchfab_cats=["furniture-home"])
            self.assertIsNone(metadata_verdict(rec, cfg)[0])
            self.assertTrue(source_warnings(rec))
        self.assertIsNone(metadata_verdict(_record(sketchfab_cats=["architecture", "furniture-home"]), cfg)[0])
        self.assertIsNone(metadata_verdict(_record(tags=["scene", "house"]), cfg)[0])

    def test_classifier_hook_records_its_rejection(self):
        from synth import filters
        from synth.config import PipelineConfig

        def no_houses(rec):
            return rec["category"] != "house", 0.1
        filters.CLASSIFIERS.append(no_houses)
        try:
            reason, detail = filters.metadata_verdict(_record(), PipelineConfig())
        finally:
            filters.CLASSIFIERS.remove(no_houses)
        self.assertEqual(reason, "classifier_rejected")
        self.assertIn("no_houses", detail)

    def test_planarity_tells_boxes_from_noise(self):
        import trimesh
        from synth.filters import planarity
        box = trimesh.creation.box(extents=(4, 3, 2)).subdivide().subdivide()
        self.assertGreater(planarity(box.vertices, box.faces)["planar_share"], 0.95)
        blob = trimesh.creation.icosphere(subdivisions=5)
        blob.vertices += np.random.default_rng(0).normal(0, 0.02, blob.vertices.shape)
        self.assertLess(planarity(blob.vertices, blob.faces)["planar_share"], 0.2)


@unittest.skipUnless(HAVE_SYNTH, "synth dependencies not installed")
class Scale(unittest.TestCase):
    def setUp(self):
        from synth.config import PipelineConfig
        self.cfg = PipelineConfig()
        self.rng = np.random.default_rng(0)

    def test_storeys_and_units(self):
        from synth.scale import estimate_height_m
        h, info = estimate_height_m({"category": "apartment", "storeys": 8}, 2640.0, self.rng)   # cm file
        self.assertEqual((info["source"], info["unit"], info["confidence"]),
                         ("file_units_agree_with_storeys", "cm", "high"))
        self.assertAlmostEqual(h, 26.4, places=1)
        h, info = estimate_height_m({"category": "house", "storeys": 2}, 0.9, self.rng)          # unitless
        self.assertEqual((info["source"], info["confidence"]), ("caption_storeys", "medium"))
        self.assertAlmostEqual(h, 2 * 3.0 * 1.15, places=2)
        h, info = estimate_height_m({"category": "house"}, 9.5, self.rng)                         # metres
        self.assertEqual((info["source"], info["unit"], info["confidence"]), ("file_units", "m", "medium"))
        h, info = estimate_height_m({"category": "tower"}, 0.004, self.rng)                       # nothing fits
        self.assertEqual((info["source"], info["confidence"]), ("category_prior", "low"))

    def test_uncertain_scale_is_kept(self):
        from synth.scale import estimate_height_m
        # Pilot v2 read a 26 m "house" (an estate) as feet -- 8 m -- because metres fell outside
        # the house range; metres is the glTF unit, so it is kept, with low confidence.
        h, info = estimate_height_m({"category": "house"}, 26.27, self.rng)
        self.assertEqual((info["unit"], info["confidence"]), ("m", "low"))
        self.assertAlmostEqual(h, 26.27, places=2)
        # In the house range but far from a typical house (a 22 m hut): metres, low confidence.
        h, info = estimate_height_m({"category": "house"}, 21.9, self.rng)
        self.assertEqual((info["unit"], info["confidence"]), ("m", "low"))
        # Far outside the category with another unit fitting: that unit, still low confidence.
        h, info = estimate_height_m({"category": "house"}, 850.0, self.rng)
        self.assertEqual((info["unit"], info["confidence"]), ("cm", "low"))
        # A two-storey house on a round plot, 10.9 units tall: taller than 2 storeys (6.9 m)
        # by its plot and roof structure, still metres (pilot v2 shrank it to 6.9 m).
        h, info = estimate_height_m({"category": "house", "storeys": 2}, 10.89, self.rng)
        self.assertEqual((info["unit"], info["confidence"]), ("m", "medium"))
        # Twice the storeys' height is not explained by a plinth: the storey count wins.
        h, info = estimate_height_m({"category": "institutional", "storeys": 5}, 52.08, self.rng)
        self.assertEqual(info["source"], "caption_storeys")
        # Pilot v2 rejected this 9.4 m two-storey house as too small; it is a confident scale.
        h, info = estimate_height_m({"category": "house", "storeys": 2}, 9.41, self.rng)
        self.assertEqual((info["source"], info["confidence"]), ("file_units_agree_with_storeys", "high"))

    def test_only_a_broken_scale_is_rejected(self):
        from synth.scale import ScaleRejected, estimate_height_m
        for bad in (0.0, 1e-7, 3e9, float("nan"), float("inf")):
            with self.assertRaises(ScaleRejected) as ctx:
                estimate_height_m({"category": "house"}, bad, self.rng)
            self.assertEqual(ctx.exception.reason, "scale_broken", bad)

    def test_bounds(self):
        from mcvoxel.scene import SceneMesh
        from synth.scale import ScaleRejected, building_size_verdict, choose_scale

        def box_scene(w, h, d):
            v = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, d], [w, h, d], [0, 0, d]], float)
            return SceneMesh(verts=v, faces=np.array([[0, 1, 2], [3, 4, 5]]), uv=np.full((6, 2), np.nan),
                             vcolor=None, face_mat=np.zeros(2, np.int32), materials=[])
        # A 480 m tower is scaled down to the height bound and flagged, not rejected.
        bpu, info = choose_scale({"category": "skyscraper", "storeys": 110}, box_scene(60, 480, 60), self.cfg, self.rng)
        self.assertTrue(info["height_capped"] and info["scale_uncertain"])
        self.assertAlmostEqual(480 * bpu, self.cfg.max_height_blocks, places=6)
        # A 4 m kiosk keeps its real size.
        bpu, info = choose_scale({"category": "commercial"}, box_scene(6, 4, 5), self.cfg, self.rng)
        self.assertFalse(info["height_capped"] or info["scale_uncertain"])
        self.assertAlmostEqual(4 * bpu, 4.0, delta=4.0 * self.cfg.scale_jitter)
        # A model whose box is too big to voxelise quickly keeps its proportions, smaller.
        bpu, info = choose_scale({"category": "institutional"}, box_scene(640, 190, 500), self.cfg, self.rng)
        self.assertTrue(info["size_capped"] and info["scale_uncertain"])
        self.assertLessEqual(640 * 190 * 500 * bpu ** 3, self.cfg.max_grid_voxels * 1.001)
        # Only a site too large to process is refused at this stage.
        with self.assertRaises(ScaleRejected) as ctx:
            choose_scale({"category": "house"}, box_scene(2000, 8, 2000), self.cfg, self.rng)
        self.assertEqual(ctx.exception.reason, "too_large")
        # Small buildings are kept and flagged; slender towers are not small; only a target
        # too small to hold a building (or too large for a pair) is refused.
        reason, _, dims = building_size_verdict(np.ones((9, 7, 9), bool), self.cfg)
        self.assertIsNone(reason)
        self.assertTrue(dims["small_building"])
        reason, _, dims = building_size_verdict(np.ones((8, 26, 8), bool), self.cfg)
        self.assertIsNone(reason)
        self.assertFalse(dims["small_building"])
        self.assertEqual(building_size_verdict(np.ones((2, 5, 2), bool), self.cfg)[0], "degenerate_size")
        self.assertEqual(building_size_verdict(np.ones((600, 5, 10), bool), self.cfg)[0], "too_large")


@unittest.skipUnless(HAVE_SYNTH, "synth dependencies not installed")
class Validation(unittest.TestCase):
    def test_contract_violations_are_caught(self):
        from mcvoxel import vocab
        from synth.validate import check_target
        shape = np.zeros((7, 6, 7), np.uint8)
        material = np.zeros((7, 6, 7), np.uint16)
        stone = vocab.MATERIAL_INDEX["minecraft:stone_bricks"]
        shape[2:5, 0:3, 2:5], material[2:5, 0:3, 2:5] = 1, stone
        shape[3, 1, 3], material[3, 1, 3] = vocab.IGNORE_SHAPE, vocab.IGNORE_MATERIAL
        self.assertEqual(check_target(shape, material), [])
        bad = shape.copy()
        bad[3, 2, 3] = vocab.IGNORE_SHAPE                     # an ignored voxel on the roof
        self.assertTrue(any("visible" in f for f in check_target(bad, material)))
        concrete = vocab.MATERIAL_INDEX["minecraft:white_concrete"]
        bad_m = material.copy()
        shape2 = shape.copy()
        shape2[2, 2, 2], bad_m[2, 2, 2] = vocab.SHAPES.index("stairs_east"), concrete
        self.assertTrue(any("stairs on materials" in f for f in check_target(shape2, bad_m)))
        edge = shape.copy()
        edge[0, 0, 0], material[0, 0, 0] = 1, stone
        self.assertTrue(any("margin" in f for f in check_target(edge, material)))


@unittest.skipUnless(HAVE_SYNTH, "synth dependencies not installed")
class EndToEnd(unittest.TestCase):
    def setUp(self):
        from synth.config import PipelineConfig
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = PipelineConfig(run_name="t", data_root=str(self.tmp), tmp_dir=str(self.tmp / "tmp"),
                                  render_size=160)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pair_is_complete_valid_and_attributed(self):
        import schem_voxels
        from mcvoxel import vocab
        from nbt_safe import load as nbt_load
        from synth.pair import make_pair
        glb = self.tmp / "h.glb"
        _house_glb(glb)
        out = self.tmp / "sample"
        man = make_pair(_record(), glb, out, self.cfg, seed=7)
        self.assertEqual(man["status"], "ok")
        for name in ("target.schem", "target_geometric.schem", "target.npz", "input.npz", "manifest.json",
                     "review.png", "shapes.png", "source.png"):
            self.assertTrue((out / name).exists(), name)
        self.assertEqual(man["scale"]["source"], "file_units")
        self.assertGreaterEqual(man["target"]["dims"]["footprint_blocks"], 16)
        vg = schem_voxels.load_voxels(out / "target.schem")
        self.assertTrue(any("stairs" in n for n in vg.names), "a 45-degree gable must become stairs")
        self.assertTrue(any("glass" in n for n in vg.names))
        t = np.load(out / "target.npz")
        i = np.load(out / "input.npz")
        for key in ("shape", "material", "shape_geometric", "material_geometric"):
            self.assertEqual(t[key].shape, i["occ"].shape, key)              # one frame, voxel for voxel
        self.assertTrue((t["shape"] == vocab.IGNORE_SHAPE).any())            # interior ignored
        self.assertTrue((t["shape"] == 0).any())                             # outside air supervised
        builder_stairs = int((t["shape"] >= 4).sum() - (t["shape"] == vocab.IGNORE_SHAPE).sum())
        geometric_stairs = int((t["shape_geometric"] >= 4).sum() - (t["shape_geometric"] == vocab.IGNORE_SHAPE).sum())
        self.assertGreater(builder_stairs, geometric_stairs)
        self.assertEqual(man["source"]["licence"], "CC-BY-4.0")
        self.assertIn("Test House", man["source"]["attribution"])
        root = nbt_load(out / "target.schem")
        root = root[""] if "" in root else root
        self.assertEqual(int(root["DataVersion"]), 3465)
        self.assertEqual(str(root["Metadata"]["roof_style"]), "builder")
        self.assertEqual(man["validation"]["failures"], [])
        json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    def test_small_house_is_kept_and_flagged(self):
        from synth.pair import make_pair
        glb = self.tmp / "small.glb"
        _house_glb(glb, width=9.0, depth=7.0, wall_h=3.0, roof_h=3.5)
        man = make_pair(_record(), glb, self.tmp / "s", self.cfg, seed=1)
        self.assertEqual(man["status"], "ok")
        self.assertTrue(man["flags"]["small_building"])
        self.assertFalse(man["flags"]["scale_uncertain"])
        self.assertEqual(man["validation"]["failures"], [])

    def test_broken_scale_is_rejected_with_a_reason(self):
        from synth.pair import Rejected, make_pair
        glb = self.tmp / "tiny.glb"
        _house_glb(glb, scale=1e-8)
        with self.assertRaises(Rejected) as ctx:
            make_pair(_record(), glb, self.tmp / "s", self.cfg, seed=1)
        self.assertEqual(ctx.exception.reason, "scale_broken")
        self.assertTrue((self.tmp / "s" / "source.png").exists())            # the rejection can be checked

    def test_compare_runs_finds_rescued_pairs(self):
        from synth.compare_runs import compare_runs
        from synth.pair import make_pair
        glb = self.tmp / "h.glb"
        _house_glb(glb)
        run_a, run_b = self.tmp / "run-a", self.tmp / "run-b"
        (run_a / "samples" / "u0").mkdir(parents=True)
        (run_a / "samples" / "u0" / "manifest.json").write_text(json.dumps(
            {"uid": "u0", "status": "rejected", "stage": "target", "reason": "too_small", "detail": "9 blocks"}),
            encoding="utf-8")
        make_pair(_record(), glb, run_b / "samples" / "u0", self.cfg, seed=3)
        summary = compare_runs(run_a, run_b)
        self.assertEqual(summary["rescued"], 1)
        self.assertEqual(summary["rescued_from"], {"too_small": 1})
        self.assertTrue((run_b / "compare_run-a" / "rescued_01.png").exists())

    def test_same_seed_same_target(self):
        from synth.pair import make_pair
        glb = self.tmp / "h.glb"
        _house_glb(glb)
        make_pair(_record(), glb, self.tmp / "a", self.cfg, seed=3)
        make_pair(_record(), glb, self.tmp / "b", self.cfg, seed=3)
        for name in ("target.schem", "target_geometric.schem"):
            self.assertEqual((self.tmp / "a" / name).read_bytes(), (self.tmp / "b" / name).read_bytes(), name)

    def test_runner_records_and_never_deletes_a_cache(self):
        from synth import run
        glb_dir = self.tmp / "cache"
        glb_dir.mkdir()
        _house_glb(glb_dir / "u0.glb")
        self.cfg.save()
        rec = _record()
        line = run._work((rec, self.cfg.to_json(), str(glb_dir), False))
        self.assertEqual(line["status"], "ok", line)
        self.assertTrue((glb_dir / "u0.glb").exists())                       # a raw cache is read-only
        man = json.loads((self.cfg.run_dir / "samples" / "u0" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(man["status"], "ok")
        self.assertEqual(run._work((rec, self.cfg.to_json(), str(glb_dir), False))["status"], "skipped")
        (self.cfg.run_dir / "selection.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
        from synth.voxel_debug import debug_source
        audit = debug_source("u0", self.cfg.run_dir, [glb_dir], self.cfg.run_dir / "voxel_debug",
                             size=80, recorded=True)
        self.assertTrue(audit["stage_counts_match_manifest"])
        original = (self.cfg.run_dir / "samples" / "u0" / "target.schem").read_bytes()
        self.cfg.scale_jitter = 0.0
        self.assertEqual(run._work((rec, self.cfg.to_json(), str(glb_dir), False))["status"], "stale")
        self.assertEqual((self.cfg.run_dir / "samples" / "u0" / "target.schem").read_bytes(), original)

    def test_review_tool_saves_a_verdict(self):
        from synth import run
        from synth.review import create_app
        glb_dir = self.tmp / "cache"
        glb_dir.mkdir()
        _house_glb(glb_dir / "u0.glb")
        run._work((_record(), self.cfg.to_json(), str(glb_dir), False))
        client = create_app(self.cfg.run_dir).test_client()
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/sample/u0").status_code, 200)
        with client.get("/file/u0/target.png") as response:
            self.assertEqual(response.status_code, 200)
        self.assertEqual(client.get("/file/u0/manifest.json").status_code, 404)
        r = client.post("/api/verdict", json={"uid": "u0", "verdict": "flawed", "note": "roof",
                                              "tags": ["roof wrong", "not a real tag"]})
        self.assertEqual(r.status_code, 200)
        saved = json.loads((self.cfg.run_dir / "review.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["verdicts"]["u0"]["verdict"], "flawed")
        self.assertEqual(saved["verdicts"]["u0"]["tags"], ["roof wrong"])       # unknown tags dropped
        page = client.get("/sample/u0").data
        self.assertIn(b"<td>flags</td>", page)                                   # the pair's flags
        self.assertIn(b"Minecraft target, builder roofs", page)                   # both roof styles
        self.assertIn(b"Minecraft target, geometric roofs", page)
        self.assertIn(b"value='roof wrong' checked", page)                         # saved tag shown
        self.assertEqual(client.get("/?flag=height_capped").status_code, 200)
        self.assertEqual(client.get("/diagnostic/u0/stages").status_code, 404)
        debug_dir = self.cfg.run_dir / "voxel_debug"
        debug_dir.mkdir()
        shutil.copyfile(self.cfg.run_dir / "samples/u0/source.png", debug_dir / "u0.png")
        self.assertIn(b"voxel stages", client.get("/sample/u0").data)
        with client.get("/diagnostic/u0/stages") as response:
            self.assertEqual(response.status_code, 200)
        self.assertEqual(client.get("/diagnostic/u0/unknown").status_code, 404)


@unittest.skipUnless(HAVE_SYNTH, "synth dependencies not installed")
class PreviewPalette(unittest.TestCase):
    def test_terracotta_and_variants_use_target_palette(self):
        from unittest.mock import patch
        import block_colors
        from mcvoxel.block_palette import BLOCK_PALETTE
        from synth import preview

        original = block_colors.color_for
        names = ["minecraft:cyan_terracotta", "minecraft:stone_brick_stairs[facing=north,half=bottom]"]

        def check_colors(*args):
            self.assertEqual(block_colors.color_for(names[0]), BLOCK_PALETTE[names[0]])
            self.assertEqual(block_colors.color_for(names[1]), BLOCK_PALETTE["minecraft:stone_bricks"])
            return "rendered"

        with patch.object(preview, "_orbit", side_effect=check_colors):
            self.assertEqual(preview.render_blocks(np.ones((1, 1, 1)), names), "rendered")
        self.assertIs(block_colors.color_for, original)


class TrellisCompare(unittest.TestCase):
    """The real-TRELLIS experiment without the server: a stand-in 'TRELLIS' mesh (the source
    normalised to a unit cube and turned a quarter-turn, as TRELLIS returns meshes) must be
    placed back in the pair's frame and measured."""

    def setUp(self):
        from synth import run
        from synth.config import PipelineConfig
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = PipelineConfig(run_name="t", data_root=str(self.tmp), tmp_dir=str(self.tmp / "tmp"),
                                  render_size=160)
        self.cfg.save()
        cache = self.tmp / "cache"
        cache.mkdir()
        _house_glb(cache / "u0.glb")
        self.glb_cache = cache
        run._work((_record(), self.cfg.to_json(), str(cache), False))
        (self.cfg.run_dir / "selection.jsonl").write_text(json.dumps(_record()) + "\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _stand_in(self, path):
        import trimesh
        mesh = trimesh.load(str(self.glb_cache / "u0.glb"), force="mesh")
        v = mesh.vertices - mesh.bounds.mean(axis=0)
        v = v / np.ptp(mesh.vertices, axis=0).max()
        turn = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])[:3, :3]
        trimesh.Trimesh(v @ turn.T, mesh.faces, process=False).export(str(path))

    def test_prepare_place_and_compare(self):
        from synth import trellis_compare as tc
        done = tc.prepare(self.cfg.run_dir, n=5, seed=0, glb_cache=str(self.glb_cache))
        self.assertEqual(done, ["u0"])
        img = self.cfg.run_dir / "trellis_compare" / "u0" / "input.png"
        from PIL import Image
        im = Image.open(img)
        self.assertEqual(im.mode, "RGBA")
        alpha = np.asarray(im)[..., 3]
        self.assertTrue(0.05 < (alpha > 0).mean() < 0.9)         # an object on a transparent ground
        self._stand_in(self.cfg.run_dir / "trellis_compare" / "u0" / "trellis.glb")
        rep = tc.compare(self.cfg.run_dir)
        self.assertEqual(rep["pairs"], 1)
        self.assertEqual(rep["uids"], ["u0"])
        self.assertTrue((self.cfg.run_dir / "trellis_compare/u0/comparison.png").exists())
        stats = json.loads((self.cfg.run_dir / "trellis_compare" / "u0" / "stats.json").read_text(encoding="utf-8"))
        self.assertGreater(stats["trellis"]["occupancy_iou"], 0.6, stats)
        for key in ("rough3", "lum", "chroma", "nbr_de", "fill_gain", "delta_e_to_target_blocks"):
            self.assertIn(key, stats["synthetic"])


@unittest.skipUnless(HAVE_SYNTH, "synth dependencies not installed")
class Blockiness(unittest.TestCase):
    def test_voxel_build_flagged_real_model_not(self):
        import trimesh
        from mcvoxel.scene import load_scene
        from synth.blockiness import blockiness
        cubes = []
        rng = np.random.default_rng(0)
        for x, y, z in np.argwhere(rng.random((6, 5, 6)) < 0.5):
            c = trimesh.creation.box(extents=(1, 1, 1))
            c.apply_translation((x + 0.5, y + 0.5, z + 0.5))
            cubes.append(c)
        vox = trimesh.util.concatenate(cubes)
        vox.vertices += 1e-4 * rng.random(vox.vertices.shape)    # exporters' anti-z-fight insets
        self.assertTrue(blockiness(vox.vertices, vox.faces)["blocky"])
        tmp = Path(tempfile.mkdtemp())
        try:
            _house_glb(tmp / "h.glb")
            sm = load_scene(tmp / "h.glb")
            self.assertFalse(blockiness(sm.verts, sm.faces)["blocky"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
