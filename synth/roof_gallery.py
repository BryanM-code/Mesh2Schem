"""Render a small deterministic architectural fixture gallery against saved code.

Requires the editable Real2Minecraft checkout (including its test mesh fixtures).
The baseline directory contains copies of mcvoxel/grid.py, roofs.py and assign.py
captured before the change. It is never imported under the live module names.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from . import preview
from .pair import _write_target
from .roof_compare import aligned_grid
from .roof_metrics import measure_roof


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fixtures(sf):
    for pitch, name in ((1., 'gable-1-1'), (.5, 'gable-2-1'), (1/3, 'gable-3-1'), (2., 'gable-1-2')):
        yield name, sf.gable_house(depth=10, roof_h=6*pitch, wall_h=5, overhang=1, window=False)
    yield 'hip', sf.hip_house()
    yield 'compound-valleys', sf.compound_house()
    yield 'lean-to', sf.lean_to()
    for material in ('wood', 'concrete'):
        pieces = [(sf._box((10,.4,8),(5,.2,4)),sf.WALL,sf.KIND_SURFACE),
                  (sf._box((10,.25,8),(5,4.9,4)),sf.ROOF,sf.KIND_SURFACE)]
        for x in (2.4,5.4,8.4):
            pieces.append((sf._box((.2,4.4,.2),(x,2.6,4.4)),sf.WALL,sf.KIND_SURFACE))
        scene = sf.mesh_to_scene(pieces)
        for m in scene.materials[2:]:
            m.name = material + ' support'
        yield material + '-porch', scene
    for material, detail in (('stone','parapet'), ('wood','railing')):
        scene = sf.mesh_to_scene([
            (sf._box((10,4,8),(5,2,4)),sf.WALL,sf.KIND_SURFACE),
            (sf._box((.2,1.7,6),(1.4,4.85,4)),sf.WALL,sf.KIND_SURFACE)])
        scene.materials[1].name = material + ' ' + detail
        yield detail, scene


def gallery(baseline, out, size=360):
    import mcvoxel
    from mcvoxel import grid, roofs, assign, vocab
    from mcvoxel.block_palette import BUILDER_PALETTE
    from mcvoxel.blockstates import stair_shapes

    out, baseline = Path(out), Path(baseline)
    if out.exists():
        raise ValueError('Gallery already exists; choose a new directory')
    old_grid = _module('mcvoxel._gallery_old_grid', baseline/'grid.py')
    old_roofs = _module('mcvoxel._gallery_old_roofs', baseline/'roofs.py')
    old_assign = _module('mcvoxel._gallery_old_assign', baseline/'assign.py')
    fixture_path = Path(mcvoxel.__file__).parent.parent/'tests/scene_fixtures.py'
    sf = _module('_gallery_scene_fixtures', fixture_path)
    out.mkdir(parents=True)
    offset = np.array([.21,0.,.37])
    pad = ((2,2),(0,3),(2,2))
    kw = dict(palette=list(BUILDER_PALETTE), shadow_repaint=False)
    records = []
    for name, scene in fixtures(sf):
        print(name, flush=True)
        dst = out/name
        dst.mkdir()
        vm = grid.pad_model(grid.voxelize_scene(scene,1.,offset,drop_ground=False,clean=False),pad)
        old_vm = old_grid.pad_model(old_grid.voxelize_scene(scene,1.,offset,drop_ground=False,clean=False),pad)
        vb, plan = roofs.builder_roofs(vm,scene.verts+offset-vm.origin,scene.faces,scene=scene)
        old_vb, old_plan = old_roofs.builder_roofs(old_vm,scene.verts+offset-old_vm.origin,scene.faces)
        target, geometric = assign.assign_blocks(vb,**kw), assign.assign_blocks(vm,**kw)
        old = old_assign.assign_blocks(old_vb,**kw)
        roofs.harmonize_roof_materials(target,plan,palette=list(BUILDER_PALETTE),rgb=vb.rgb)
        old_roofs.harmonize_roof_materials(old,old_plan,palette=list(BUILDER_PALETTE),rgb=old_vb.rgb)
        new_grid, names = _write_target(target,vb,dst/'target.schem',{'fixture':name})
        geo_grid, geo_names = _write_target(geometric,vm,dst/'geometric.schem',{'fixture':name})
        previous, old_names = stair_shapes(*old.sponge_grid(old_vb.solid.shape))
        previous = aligned_grid(previous,old_vm.origin,vm.solid.shape,vm.origin)
        from schem_write import write_schem
        write_schem(previous,old_names,dst/'previous.schem',data_version=3465)
        panels = [(preview.render_scene(scene,size=size),'source mesh'),
                  (preview.render_blocks(previous,old_names,size=size),'previous builder'),
                  (preview.render_blocks(new_grid,names,size=size),'new builder'),
                  (preview.render_blocks(geo_grid,geo_names,size=size),'geometric'),
                  (preview.render_blocks(previous,old_names,yaw=preview.YAW+180,size=size),'previous back'),
                  (preview.render_blocks(new_grid,names,yaw=preview.YAW+180,size=size),'new back'),
                  (preview.render_shapes(new_grid,names,size=size),'shapes / details'),
                  (preview.render_roof_regions(plan,size=size),'regions / patterns')]
        preview.sheet(panels,size=size).save(dst/'comparison.png')
        sh, counts = np.unique(target.shape,return_counts=True)
        row = {'fixture':name, 'before':measure_roof(previous,old_names,plan),
               'after':measure_roof(new_grid,names,plan), 'roofs':vb.info['roofs'],
               'shapes':{vocab.SHAPES[int(s)]:int(n) for s,n in zip(sh,counts) if s!=255},
               'states_valid':True, 'schematic_roundtrip':True}
        records.append(row)
        (dst/'manifest.json').write_text(json.dumps(row,indent=1),encoding='utf-8')
    provenance = {'fixture_sha256':hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                  'baseline_sha256':{n:hashlib.sha256((baseline/n).read_bytes()).hexdigest()
                                     for n in ('grid.py','roofs.py','assign.py')},
                  'offset':offset.tolist(), 'clean':False, 'drop_ground':False}
    (out/'summary.json').write_text(json.dumps({'provenance':provenance,'fixtures':records},indent=1),encoding='utf-8')
    lines = ['# Roof and structural detail fixtures','',
             'Same mesh, scale, offset, camera and palette. Previous uses the saved pre-change voxel/roof/assignment modules.',
             'Shared unchanged raster/colour routines and the current renderer are used on both sides. These are diagnostic fixtures, not training pairs.','',
             '| Fixture | Hole indicators before / after | Unsupported caps before / after |',
             '|---|---:|---:|']
    for r in records:
        a,b = r['before'],r['after']
        lines.append(f"| [{r['fixture']}]({r['fixture']}/comparison.png) | {a['roof_holes']} / {b['roof_holes']} | {a['unsupported_roof_caps']} / {b['unsupported_roof_caps']} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',required=True)
    ap.add_argument('--out',required=True)
    args = ap.parse_args()
    gallery(args.baseline,args.out)


if __name__ == '__main__':
    main()
