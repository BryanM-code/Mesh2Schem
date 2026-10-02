"""Export, preview, metric and label-contract checks for mcvox-v2 targets."""
import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import schem_voxels
import thumbs
from schem_write import write_schem
from mcvoxel import block_shapes as bs, vocab
from mcvoxel.grid import VoxelModel, shell_of, shape_shell
from synth.roof_metrics import measure_roof
from synth.target import quality
from synth.validate import check_target


class ExportDetails(unittest.TestCase):
    def test_every_fence_and_wall_connection_state_roundtrips(self):
        states=[vocab.AIR]
        for family,values,up in ((bs.FENCE_VARIANTS,('true','false'),('',)),
                                  (bs.WALL_VARIANTS,('none','low','tall'),('true','false'))):
            for block in family.values():
                for connections in itertools.product(values,repeat=4):
                    for post in up:
                        props=dict(zip(('east','north','south','west'),connections))
                        props['waterlogged']='false'
                        if post:props['up']=post
                        state=block+'['+','.join(k+'='+v for k,v in sorted(props.items()))+']'
                        vocab.validate_state(state)
                        self.assertIsNotNone(vocab.encode_state(state)[1])
                        states.append(state)
        grid=np.arange(len(states),dtype=np.uint16).reshape(1,1,-1)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'all_connections.schem'
            write_schem(grid,states,p,data_version=3465)
            back=schem_voxels.load_voxels(p)
            np.testing.assert_array_equal(np.asarray(back.names,dtype=object)[back.grid],np.asarray(states,dtype=object)[grid])

    def test_label_contract_checks_detail_material_families(self):
        shape=np.zeros((5,5,5),np.uint8);material=np.zeros(shape.shape,np.uint16)
        shape[2,0:3,2]=8;material[2,0:3,2]=vocab.MATERIAL_INDEX['minecraft:oak_planks']
        self.assertEqual(check_target(shape,material),[])
        material[2,0:3,2]=vocab.MATERIAL_INDEX['minecraft:white_concrete']
        self.assertTrue(any('incompatible' in s for s in check_target(shape,material)))

    def test_detail_preview_draws_connections_and_correct_corner_volume(self):
        fence=thumbs._shape_boxes('oak_fence',{'east':'true','north':'false'})
        self.assertTrue(any(b[3]==1. for b in fence))
        self.assertFalse(any(b[2]==0. for b in fence))
        wall=thumbs._shape_boxes('stone_brick_wall',{'up':'false','east':'low','west':'low'})
        self.assertTrue(all(b[4]==.875 for b in wall))
        for corner,volume in (('straight',.75),('inner_left',.875),('outer_right',.625)):
            boxes=thumbs._shape_boxes('oak_stairs',{'facing':'north','shape':corner})
            self.assertAlmostEqual(sum((b[3]-b[0])*(b[4]-b[1])*(b[5]-b[2]) for b in boxes),volume)

    def test_short_building_gate_uses_enclosure_not_export_visibility(self):
        solid=np.ones((6,3,8),bool);shapes=np.zeros(solid.shape,np.int8);shapes[:,2,:]=bs.SHAPE_SLAB_BOTTOM
        vm=VoxelModel(solid,solid.copy(),shell_of(solid),np.zeros(3,int),1.,shapes)
        original=quality(vm)
        vm.shell=shape_shell(solid,shapes)
        self.assertGreater(int(vm.shell.sum()),int(shell_of(solid).sum()))
        self.assertEqual(original,quality(vm))

    def test_roof_metrics_are_palette_order_independent_and_detect_open_risers(self):
        # Full at y=1 and slab at y=2 have a visible opening when backing at y=1
        # behind the higher course is omitted. Both caps are present in occupancy.
        grid=np.zeros((4,2,3),np.uint16)
        grid[1,:,0]=1;grid[2,:,1]=2;grid[2,:,2]=2
        names=[vocab.AIR,'minecraft:stone','minecraft:stone_slab[type=bottom]']
        plan=SimpleNamespace(group=np.ones((3,2),int),level=np.array([[1,1],[2,2],[2,2]]))
        before=measure_roof(grid,names,plan)
        self.assertEqual(before['missing_caps'],0);self.assertGreater(before['open_course_interfaces'],0)
        perm=np.array([2,0,1]);reordered=['minecraft:stone','minecraft:stone_slab[type=bottom]',vocab.AIR]
        self.assertEqual(before,measure_roof(perm[grid],reordered,plan))
        grid[1,:,1]=1
        self.assertEqual(measure_roof(grid,names,plan)['roof_holes'],0)


if __name__=='__main__':unittest.main()
