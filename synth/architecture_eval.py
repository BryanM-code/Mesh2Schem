"""Controlled architecture audit. Never writes into the pilot or fetches a source.

python -m synth.architecture_eval --split development --mode legacy --scales 0.75 1 1.5 2.5 4
Use --alignment geometry to isolate grid phase at identical recorded scale.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .architecture_inventory import BASE, OUT, dump, sha


def run(row, multiplier, args):
    from mcvoxel import roofs
    from mcvoxel.assign import assign_blocks
    from mcvoxel.blockstates import finish_states
    from mcvoxel.grid import pad_model, voxelize_scene
    from mcvoxel.scene import load_scene, split_clutter
    from . import preview
    from .config import PipelineConfig
    from .orient import orient
    from .scale import choose_scale
    from .target import palette_ids, site_height_blocks
    from .architecture_metrics import measure
    from .pair import _write_target
    from .voxel_debug import _render
    import schem_voxels

    uid = row['uid']
    tag = f'{args.mode}-{args.alignment}-{multiplier:g}x' + ('-no-scale-jitter' if args.no_scale_jitter else '') + (f'-{args.revision}' if args.revision else '')
    dest = OUT / 'runs' / uid / tag
    if (dest / 'result.json').exists():
        return json.loads((dest / 'result.json').read_text(encoding='utf-8'))
    dest.mkdir(parents=True, exist_ok=True)
    start = time.time()
    man = json.loads((BASE / 'samples' / uid / 'manifest.json').read_text(encoding='utf-8'))
    cfg = PipelineConfig.load(BASE / 'run_config.json')
    rec = next(r for r in map(json.loads, (BASE / 'selection.jsonl').read_text(encoding='utf-8').splitlines()) if r['uid'] == uid)
    path = BASE / 'raw-cache' / (uid + '.glb')
    scene, clutter = split_clutter(load_scene(path))
    scene, orientation = orient(scene)
    # Replaying RNG is essential: old manifests rounded grid phase to 3 decimal places.
    rng = np.random.default_rng(man['seed'])
    recorded_bpu, scale_info = choose_scale(rec, scene, cfg, rng)
    phase = rng.uniform(0., 1., size=3); phase[1] = 0.
    bpu = float(man['scale']['blocks_per_unit']) * multiplier
    if args.no_scale_jitter:
        # Capped examples cannot isolate jitter by division: their scale is not jitter*j.
        if man['scale'].get('height_capped') or man['scale'].get('size_capped'):
            raise ValueError('No-jitter ablation requires an uncapped baseline')
        # Reconstruct the exact draw, rather than the rounded manifest field.
        from .scale import estimate_height_m
        jrng = np.random.default_rng(man['seed'])
        estimate_height_m(rec, float((scene.bounds[1]-scene.bounds[0])[1]), jrng)
        bpu /= float(jrng.uniform(1-cfg.scale_jitter, 1+cfg.scale_jitter))
    lo, hi = scene.bounds
    offset = -lo * bpu + phase
    alignment = {'policy': 'recorded_rng', 'phase': phase.tolist()}
    if args.alignment == 'geometry':
        from mcvoxel.architecture import align_grid
        offset, alignment = align_grid(scene.verts, scene.faces, bpu)
    dims = np.ceil((hi - lo) * bpu + 8).astype(int)
    result = {'uid': uid, 'name': row['name'], 'category': row['category'], 'split': row['split'],
              'tag': tag, 'mode': args.mode, 'multiplier': multiplier, 'alignment': alignment,
              'seed': man['seed'], 'source_sha256': sha(path), 'bpu': bpu, 'offset': offset.tolist(),
              'recorded_scale_replay_error': abs(recorded_bpu - float(man['scale']['blocks_per_unit'])),
              'physical_scale': man['scale'], 'orientation': orientation, 'estimated_grid_xyz': dims.tolist()}
    result['code_snapshot'] = args.code_hashes
    # Explicit deferral, never an unrecorded resolution reduction. Fine raster is banded.
    if int(np.prod(dims)) > args.max_voxels or int(dims.max()) > args.max_dimension:
        result.update(status='resource_deferred', required_voxels=int(np.prod(dims)), budget=args.max_voxels)
        dump(dest / 'result.json', result)
        return result
    trace = {}
    scale_info = {**man['scale'], 'height_blocks': man['scale']['height_blocks'] * multiplier}
    kw = {'fine_scale': cfg.fine_scale, 'max_plate': man['scale'].get('max_plate', 3),
          'site_height': site_height_blocks(cfg, scale_info), 'site_margin': cfg.site_margin,
          'color_mode': cfg.color_mode, 'clean': cfg.close_gaps, 'trace': trace}
    if args.mode == 'coherent':
        kw['architecture'] = True
    vm = voxelize_scene(scene, bpu, offset, **kw)
    result['stages'] = {k: int(v.sum()) for k, v in trace.items() if isinstance(v, np.ndarray) and v.dtype == bool}
    result['voxel_info'] = vm.info
    if not vm.solid.any():
        result['status'] = 'empty'; dump(dest / 'result.json', result); return result
    pad = ((cfg.margin, cfg.margin), (0, cfg.margin), (cfg.margin, cfg.margin))
    vmp = pad_model(vm, pad)
    verts = scene.verts * bpu + offset - vmp.origin
    params = roofs.RoofParams(sub=cfg.fine_scale)
    if args.mode == 'coherent':
        params.coherent = True
    vb, plan = roofs.builder_roofs(vmp, verts, scene.faces, params, scene=scene)
    akw = {'palette': palette_ids(cfg), 'quantize': cfg.quantize, 'shadow_repaint': cfg.shadow_repaint}
    if args.mode == 'coherent':
        akw['coherent_materials'] = True
    geo, target = assign_blocks(vmp, **akw), assign_blocks(vb, **akw)
    result['materials'] = roofs.harmonize_roof_materials(target, plan, palette=palette_ids(cfg), rgb=vb.rgb)
    grid, names = _write_target(target, vb, dest / 'target.schem', {'audit': tag})
    gg, gn = finish_states(*geo.sponge_grid(vmp.solid.shape))
    result['metrics'] = measure(verts, scene.faces, grid, names, plan, vb)
    result['roof_info'] = vb.info.get('roofs', {})
    result['block_info'] = target.info
    result['frame_origin'] = vb.origin.tolist(); result['grid_xyz'] = list(vb.solid.shape)
    result['roundtrip'] = True
    if multiplier == 1 and args.alignment == 'recorded' and args.mode == 'legacy':
        old = schem_voxels.load_voxels(BASE / 'samples' / uid / 'target.schem')
        result['baseline_exact_states'] = bool(old.grid.shape == grid.shape and np.array_equal(np.asarray(old.names)[old.grid], np.asarray(names)[grid]))
    np.savez_compressed(dest / 'diagnostics.npz', solid=vb.solid, geometric=vmp.solid,
                        shape=target.shape, material=target.material, region=plan.region,
                        level=plan.level, roof_shape=plan.shape, roof_group=plan.group, origin=vb.origin,
                        offset=offset, bpu=bpu, **{k: v for k, v in trace.items() if isinstance(v, np.ndarray)})
    for i, layer in enumerate(getattr(plan, 'layers', [])):
        np.savez_compressed(dest / f'roof-layer-{i+1}.npz', region=layer.region, level=layer.level,
                            shape=layer.shape, group=layer.group)
    size = args.render_size
    panels = []
    for yaw in (40, 130, 220, 310):
        panels += [(preview.render_scene(scene, verts, yaw=yaw, size=size, n=200_000), f'source / yaw {yaw}'),
                   (preview.render_shapes(grid, names, yaw=yaw, size=size), 'shape geometry'),
                   (preview.render_blocks(grid, names, yaw=yaw, size=size), f'{tag} / materials')]
    preview.sheet(panels, cols=3, size=size).save(dest / 'views.png')
    stages = [(_render(trace['surface'], size=size), 'conservative raster'),
              (_render(trace['flood'], add=trace['enclosed'], cut=True, size=size), 'flood + enclosure (blue)'),
              (_render(trace['after_clean'], size=size), 'site / component cleanup'),
              (_render(vm.solid, remove=trace.get('slivers'), size=size), 'fine slivers (red)'),
              (preview.render_shapes(gg, gn, size=size), 'local geometric shapes'),
              (preview.render_shapes(grid, names, size=size), 'roof construction'),
              (preview.render_rgb(vm.shell.transpose(1,2,0), vm.rgb.transpose(1,2,0,3), size=size), 'sampled colour'),
              (preview.render_roof_regions(plan, size=size), 'roof regions')]
    preview.sheet(stages, cols=4, size=size).save(dest / 'stages.png')
    result.update(status='ok', seconds=round(time.time()-start, 2))
    dump(dest / 'result.json', result)
    return result


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--split', choices=('development', 'reserved'), default='development')
    ap.add_argument('--mode', choices=('legacy', 'coherent'), default='legacy')
    ap.add_argument('--alignment', choices=('recorded', 'geometry'), default='recorded')
    ap.add_argument('--scales', type=float, nargs='+', default=[1])
    ap.add_argument('--uid', nargs='*')
    ap.add_argument('--max-voxels', type=int, default=8_000_000)
    ap.add_argument('--max-dimension', type=int, default=600)
    ap.add_argument('--render-size', type=int, default=240)
    ap.add_argument('--revision', default='')
    ap.add_argument('--no-scale-jitter', action='store_true')
    args = ap.parse_args()
    if args.mode == 'legacy':
        sys.path.insert(0, str(OUT / 'frozen/Real2Minecraft'))
        # Alignment is a treatment only; frozen geometry/assignment remain the control.
        import mcvoxel
        mcvoxel.__path__.append(str(BASE.parents[3].parent / 'Real2Minecraft/mcvoxel'))
    import mcvoxel
    args.code_hashes = {str(p): sha(p) for p in Path(mcvoxel.__file__).parent.glob('*.py')}
    rows = json.loads((OUT / 'evaluation-set.json').read_text(encoding='utf-8'))['rows']
    for row in rows:
        if row['split'] != args.split or (args.uid and row['uid'] not in args.uid):
            continue
        for scale in args.scales:
            try:
                result = run(row, scale, args)
                print(json.dumps({k: result.get(k) for k in ('uid','category','tag','status','seconds','baseline_exact_states')}) , flush=True)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(json.dumps({'uid': row['uid'], 'scale': scale, 'error': repr(exc)}), flush=True)


if __name__ == '__main__':
    main()
