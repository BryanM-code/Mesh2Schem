import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh
from synth.config import PipelineConfig
from synth.scale import choose_scale, ScaleRejected


class RepresentationContract(unittest.TestCase):
    def scene(self, size):
        mesh=trimesh.creation.box(extents=size)
        return SimpleNamespace(verts=mesh.vertices,faces=mesh.faces,bounds=mesh.bounds)

    def test_legacy_config_hash_is_stable(self):
        path=Path(__file__).resolve().parents[1]/'data/synthetic/pilot-v5-200-20261001/run_config.json'
        if not path.exists(): self.skipTest('local archived pilot not present')
        raw=json.loads(path.read_text())
        self.assertEqual(PipelineConfig.from_dict(raw).hash(),raw['config_hash'])

    def test_physical_size_and_architecture_enlargement_are_separate(self):
        cfg=PipelineConfig(scale_policy='architecture',architecture_enlargement=3.,max_grid_voxels=1_000_000)
        scene=self.scene((8,6,10))
        bpu,info=choose_scale({'category':'house'},scene,cfg,np.random.default_rng(13))
        self.assertEqual(info['height_m'],6)
        self.assertEqual(info['enlargement'],3)
        self.assertEqual(info['height_blocks'],18)
        self.assertFalse(info['size_capped'])
        np.testing.assert_allclose(np.ptp(scene.verts*bpu,axis=0)/np.array([8,6,10]),3)

    def test_budget_failure_records_requested_resolution_without_downscaling(self):
        cfg=PipelineConfig(scale_policy='architecture',architecture_enlargement=4.,max_grid_voxels=1000)
        with self.assertRaises(ScaleRejected) as error:
            choose_scale({'category':'house'},self.scene((8,6,10)),cfg,np.random.default_rng(0))
        self.assertEqual(error.exception.reason,'resolution_budget')
        self.assertEqual(error.exception.info['height_blocks'],24)
        self.assertFalse(error.exception.info['size_capped'])

    def test_configuration_roundtrip_preserves_opt_in(self):
        cfg=PipelineConfig(architecture_mode='coherent',scale_policy='architecture',grid_alignment='geometry')
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'config.json'; cfg.save(p)
            self.assertEqual(PipelineConfig.load(p),cfg)

    def test_coherent_pair_keeps_input_target_frame_and_schematic_contract(self):
        from test_synth import _house_glb, _record
        from synth.pair import make_pair
        cfg=PipelineConfig(architecture_mode='coherent',scale_policy='architecture',grid_alignment='geometry',
                           architecture_enlargement=1.5,renders='none',scan_min_faces=100000)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); glb=root/'house.glb'; _house_glb(glb)
            m=make_pair(_record(),glb,root/'pair',cfg,1234)
            self.assertEqual(m['status'],'ok')
            self.assertTrue(m['validation']['schematic_roundtrip'])
            self.assertEqual(m['representation_version'],'architecture-v1')
            self.assertEqual(m['architecture_diagnostics']['visual_approval'],'not_established')
            with np.load(root/'pair/input.npz') as inp, np.load(root/'pair/target.npz') as target:
                self.assertEqual(inp['occ'].shape,target['shape'].shape)
                self.assertEqual(float(target['blocks_per_unit']),m['scale']['blocks_per_unit'])


if __name__=='__main__': unittest.main()
