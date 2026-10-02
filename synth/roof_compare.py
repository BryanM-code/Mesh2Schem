"""Small same-source target comparison; never overwrites an existing sample run.

python -m synth.roof_compare --before data/synthetic/pilot-v4 --out NEW_DIR
                            --raw-cache data/synthetic/pilot-v1/glb --only UID_PREFIX ...
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from . import preview
from .config import PipelineConfig
from .pair import make_pair
from .roof_metrics import measure_roof


def aligned_grid(old_grid, old_origin, new_shape, new_origin, air_index=0):
    """Embed an earlier Sponge grid in the new grid's world-coordinate frame."""
    xyz=old_grid.transpose(2,0,1)
    out=np.full(new_shape,air_index,np.uint16)
    shift=np.asarray(old_origin,dtype=int)-np.asarray(new_origin,dtype=int)
    lo=np.maximum(shift,0);hi=np.minimum(shift+xyz.shape,new_shape)
    if np.all(hi>lo):
        dst=tuple(slice(a,b) for a,b in zip(lo,hi))
        src=tuple(slice(a,b) for a,b in zip(lo-shift,hi-shift))
        out[dst]=xyz[src]
    return out.transpose(1,2,0)


def compare(before, out, raw_cache, only, size=480):
    from PIL import Image
    import mcvoxel
    from mcvoxel import vocab
    from mcvoxel.roofs import RoofPlan
    import schem_voxels
    before,out,raw_cache=Path(before),Path(out),Path(raw_cache)
    if (out/'samples').exists() or (out/'run_config.json').exists():
        raise ValueError('Comparison already exists; choose a new output directory')
    records=[json.loads(line) for line in (before/'selection.jsonl').read_text(encoding='utf-8').splitlines()]
    records=[r for r in records if any(r['uid'].startswith(p) for p in only)]
    if len(records)!=len(only):
        raise ValueError('Every requested prefix must identify exactly one selected source')
    # Resolve every input before starting; no downloads and no silent source substitution.
    for r in records:
        if not (raw_cache/(r['uid']+'.glb')).is_file():raise FileNotFoundError(r['uid'])
    cfg=PipelineConfig.load(before/'run_config.json')
    cfg.run_name=out.name;cfg.data_root=str(out.parent);cfg.renders='all';cfg.render_size=size
    cfg.save()
    (out/'selection.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    (out/'vocab.json').write_text(vocab.vocab_json(),encoding='utf-8')
    code={}
    for root in (Path(__file__).parent,Path(mcvoxel.__file__).parent):
        for p in sorted(root.glob('*.py')):code[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ('thumbs.py', 'schem_write.py', 'schem_voxels.py'):
        p = Path(__file__).parent.parent / name
        code[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'source_hashes.json').write_text(json.dumps(code,indent=1),encoding='utf-8')
    summaries=[]
    for rec in records:
        uid=rec['uid'];src=before/'samples'/uid;dst=out/'samples'/uid
        previous=json.loads((src/'manifest.json').read_text(encoding='utf-8'))
        print('Building',uid,rec.get('name','').encode('ascii','replace').decode(),flush=True)
        man=make_pair(rec,raw_cache/(uid+'.glb'),dst,cfg,previous['seed'])
        old=schem_voxels.load_voxels(src/'target.schem')
        new=schem_voxels.load_voxels(dst/'target.schem')
        with np.load(src/'target.npz') as a,np.load(dst/'target.npz') as b:
            original=aligned_grid(old.grid,a['origin'],b['shape'].transpose(2,0,1).shape,b['origin'],old.names.index('minecraft:air'))
            same_frame=bool(np.array_equal(a['origin'],b['origin']) and a['shape'].shape==b['shape'].shape)
        with np.load(dst/'roof_plan.npz') as a:
            plan=RoofPlan(**{k:a[k] for k in a.files},info=man['target']['roofs'])
        stats_before=measure_roof(original,old.names,plan)
        stats_after=man['roof_topology']
        previous_image=preview.render_blocks(original,old.names,size=size)
        previous_image.save(dst/'previous_builder.png')
        oldback=preview.render_blocks(original,old.names,yaw=preview.YAW+180,size=size)
        oldback.save(dst/'previous_back.png')
        panels=[(Image.open(dst/'source.png'),'source mesh'),
                (previous_image,'previous builder'),(Image.open(dst/'target.png'),'new builder'),
                (Image.open(dst/'target_geometric.png'),'geometric target'),
                (oldback,'previous back'),(Image.open(dst/'target_back.png'),'new back'),
                (Image.open(dst/'shapes.png'),'shape / detail diagnostic'),
                (Image.open(dst/'roof_regions.png'),'roof regions / pitches')]
        preview.sheet(panels,cols=4,size=size).save(dst/'comparison.png')
        row={'uid':uid,'name':rec.get('name'),'before':stats_before,'after':stats_after,
             'same_frame_as_previous':same_frame,'input_target_frame_valid':not man['validation']['failures'],
             'validation':man['validation'],'patterns':man['target']['roofs']['pattern_distribution'],
             'shape_usage':man['blocks']['builder']['shapes'],
             'previous_shape_usage':previous['blocks']['builder']['shapes']}
        summaries.append(row)
        (out/'comparison.json').write_text(json.dumps(summaries,indent=1,ensure_ascii=False),encoding='utf-8')
        print('Topology',stats_before,'->',stats_after,flush=True)
    pattern_counts=Counter();shapes=Counter()
    for row in summaries:pattern_counts.update(row['patterns']);shapes.update(row['shape_usage'])
    totals={when:{key:sum(row[when][key] for row in summaries) for key in summaries[0][when]}
            for when in ('before','after')}
    summary={'samples':len(summaries),'topology':totals,'patterns':dict(pattern_counts),'shapes':dict(shapes),
             'validation_failures':sum(bool(r['validation']['failures']) for r in summaries),
             'metrics_note':'Roof holes = missing mesh-backed caps + open interfaces between adjacent courses; '
                            'components use emitted shapes in a shallow roof band, so separate source roofs count separately.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=1),encoding='utf-8')
    rows=['# Builder roof comparison','',f'{len(summaries)} sources from `{before}`. Same recorded seeds and scale settings.',
          '',summary['metrics_note'],'','| Source | Holes before / after | Components before / after | Unsupported caps before / after |',
          '|---|---:|---:|---:|']
    for r in summaries:
        a,b=r['before'],r['after']
        rows.append(f"| [{r['name']}](samples/{r['uid']}/comparison.png) | {a['roof_holes']} / {b['roof_holes']} | "
                    f"{a['roof_components']} / {b['roof_components']} | {a['unsupported_roof_caps']} / {b['unsupported_roof_caps']} |")
    rows+=['','All panels use the same camera and palette. Visual judgments are recorded separately after inspection.',
           '', 'Shape legend: grey full; yellow bottom slab; orange top slab; red east stair; green west stair; '
           'blue north stair; purple south stair; brown fence; turquoise wall.']
    (out/'report.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    return summary


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--raw-cache',required=True);ap.add_argument('--only',nargs='+',required=True)
    ap.add_argument('--size',type=int,default=480)
    a=ap.parse_args(argv)
    print(json.dumps(compare(a.before,a.out,a.raw_cache,a.only,a.size),indent=1))


if __name__=='__main__':main()
