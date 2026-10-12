"""Re-measure saved experimental outputs and produce a complete local review gallery.

Rendering styles are unchanged across treatments. HTML uses locally saved images;
no server, external resources or publishing is needed.
"""
from __future__ import annotations
import argparse
import html
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .architecture_inventory import BASE, OUT, ROOT, R2M, dump


def remeasure(path):
    from mcvoxel.scene import load_scene, split_clutter
    from .orient import orient
    from .architecture_metrics import measure, projected_masks
    from .roof_metrics import measure_roof
    from .validate import check_target
    import schem_voxels
    r=json.loads(path.read_text(encoding='utf-8'))
    if r['status']!='ok': return
    dest=path.parent
    if (dest/'measurements-v2.json').exists(): return
    arrays=np.load(dest/'diagnostics.npz')
    scene,_=orient(split_clutter(load_scene(BASE/'raw-cache'/(r['uid']+'.glb')))[0])
    verts=scene.verts*r['bpu']+np.array(r['offset'])-np.array(r['frame_origin'])
    plan=SimpleNamespace(region=arrays['region'],level=arrays['level'],shape=arrays['roof_shape'],
                         group=arrays['roof_group'] if 'roof_group' in arrays else (arrays['level']>=0).astype(int),
                         info=r['roof_info'])
    vg=schem_voxels.load_voxels(dest/'target.schem')
    metrics=measure(verts,scene.faces,vg.grid,vg.names,plan)
    metrics['topology']=measure_roof(vg.grid,vg.names,plan)
    metrics['label_errors']=check_target(arrays['shape'],arrays['material'])
    # Every geometry stage is compared in the same saved final block frame. No
    # per-panel recentering enters these numerical measurements.
    source_views=projected_masks(verts,scene.faces,vg.grid,vg.names)
    origin=np.asarray(r['frame_origin'])
    full_origin=arrays['frame_origin']
    shift=(full_origin-origin).astype(int)
    stage_scores={}
    for key in ('surface','flood','enclosed','after_ground','after_site','after_clean','geometric','solid'):
        if key not in arrays: continue
        mask=arrays[key]
        if key=='enclosed': mask=mask|arrays['flood']
        if mask.shape != arrays['solid'].shape:
            aligned=np.zeros(arrays['solid'].shape,bool)
            start=np.maximum(shift,0); stop=np.minimum(shift+np.array(mask.shape),aligned.shape)
            if (stop>start).all():
                dst=tuple(slice(a,b) for a,b in zip(start,stop))
                src=tuple(slice(a,b) for a,b in zip(start-shift,stop-shift))
                aligned[dst]=mask[src]
            mask=aligned
        scores=[]
        for axis,(source,_) in zip((2,0,1),source_views):
            projection=mask.any(axis=axis)
            if axis==0: projection=projection.T
            projection=np.repeat(np.repeat(projection,2,0),2,1)
            scores.append(float((projection&source).sum()/max((projection|source).sum(),1)))
        stage_scores[key]=scores
    metrics['stage_cube_silhouette_iou']=stage_scores
    dump(dest/'measurements-v2.json',metrics)


def gallery():
    evaluation=json.loads((OUT/'evaluation-set.json').read_text(encoding='utf-8'))
    results=[]
    for p in sorted((OUT/'runs').glob('*/*/result.json')):
        r=json.loads(p.read_text(encoding='utf-8'))
        if (p.parent/'measurements-v2.json').exists():
            r['metrics']=json.loads((p.parent/'measurements-v2.json').read_text(encoding='utf-8'))
        results.append(r)
    dump(OUT/'results.json',results)
    style='''body{font:15px system-ui;background:#181b20;color:#e4e7ec;margin:24px}a{color:#91caff}h2{margin-top:42px}.grid{display:flex;gap:14px;overflow-x:auto}.case{min-width:330px;width:360px}img{width:100%;background:#111}small{display:block;color:#b9c1cc}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #535a66}summary{cursor:pointer}select,input{padding:8px;margin:8px}pre{white-space:pre-wrap}'''
    body=['<!doctype html><meta charset="utf-8"><title>Architecture audit v1</title><style>'+style+'</style>',
          '<h1>Mesh → Minecraft: complete architecture audit</h1>',
          '<p>31 sources: 17 development, 14 reserved. All treatments, deferrals and failures are retained. '
          'Images use the same four camera angles and flat palette rendering. Geometry is separately colour-coded. '
          'Each view is fitted to its own bounds for visibility; numerical silhouettes use the saved common block frame. '
          'Passing export checks is not visual approval.</p>',
          '<p><a href="results.json">All measurements and provenance</a> · <a href="evaluation-set.json">Frozen selection</a> · '
          '<a href="summary.json">Aggregated comparisons</a></p>',
          '<label>Filter <input id="filter" placeholder="category, split, name"></label>',
          '<label><input id="extras" type="checkbox"> Show intermediate experiments</label>']
    for row in evaluation['rows']:
        matching=[r for r in results if r['uid']==row['uid']]
        matching.sort(key=lambda r:(r['multiplier'],r['tag']))
        text=html.escape(f"{row['category']} · {row['name']} · {row['split']}")
        body.append(f'<section data-search="{text.lower()}"><h2>{text}</h2><p>{html.escape(row["caption"] or "")}</p><div class="grid">')
        for r in matching:
            tag=r['tag']; rel=f'runs/{r["uid"]}/{tag}'
            extra=' extra' if ('no-scale-jitter' in tag or tag=='coherent-recorded-1x' or (tag.startswith('legacy-geometry') and '-v2' not in tag)) else ''
            body.append(f'<article class="case{extra}"><h3>{html.escape(tag)}</h3>')
            if r['status']=='ok':
                m=r['metrics']
                body.append(f'<small>Silhouette {m["silhouette_iou"]:.3f} · material singletons {m["material_singleton_fraction"]:.1%} · '
                            f'planned roof regions {m["roof_regions"]} · openings filled {m["opening_filled_samples"]}/{m["opening_samples"]}</small>'
                            f'<a href="{rel}/views.png"><img loading="lazy" src="{rel}/views.png"></a>'
                            f'<p><a href="{rel}/stages.png">Stage diagnostics</a> · <a href="{rel}/target.schem">Schematic</a> · '
                            f'<a href="{rel}/result.json">Provenance</a></p>')
            else:
                body.append('<p>'+html.escape(r['status'])+f'</p><a href="{rel}/result.json">Resource requirement</a>')
            body.append('</article>')
        body.append('</div></section>')
    body.append('''<script>const q=document.querySelector('#filter'), e=document.querySelector('#extras');function update(){document.querySelectorAll('section').forEach(s=>s.hidden=!s.dataset.search.includes(q.value.toLowerCase()));document.querySelectorAll('.extra').forEach(s=>s.hidden=!e.checked)}q.oninput=e.onchange=update;update()</script>''')
    (OUT/'index.html').write_text('\n'.join(body),encoding='utf-8')
    return results


def summarize(results):
    comparisons={}
    for treatment,control in [('legacy-geometry-1x','legacy-recorded-1x'),
                              ('legacy-recorded-1x-no-scale-jitter','legacy-recorded-1x'),
                              ('coherent-recorded-1x','legacy-recorded-1x'),
                              ('coherent-geometry-1x-v2','legacy-recorded-1x'),
                              ('coherent-geometry-1x-final','legacy-recorded-1x')]:
        for split in ('development','reserved'):
            treated={r['uid']:r for r in results if r['tag']==treatment and r['split']==split and r['status']=='ok'}
            controls={r['uid']:r for r in results if r['tag']==control and r['split']==split and r['status']=='ok'}
            pairs=[]
            for uid in sorted(treated.keys()&controls.keys()):
                a,b=controls[uid],treated[uid]
                fields=('silhouette_iou','unexplained_protrusion_fraction','material_singleton_fraction','roof_regions',
                        'opening_filled_samples','opening_samples','wall_boundary_jitter_blocks','repeated_openings_missing')
                pairs.append({'uid':uid,'category':a['category'],'name':a['name'],
                              'deltas':{k:b['metrics'][k]-a['metrics'][k] for k in fields
                                        if isinstance(a['metrics'].get(k),(int,float)) and isinstance(b['metrics'].get(k),(int,float))}})
            if pairs:
                deltas=[r['deltas']['silhouette_iou'] for r in pairs]
                comparisons[f'{split}/{treatment} vs {control}']={'n':len(pairs),
                     'silhouette_mean_delta':float(np.mean(deltas)), 'silhouette_min_delta':min(deltas),
                     'regressions_over_0.01':sum(v<-.01 for v in deltas),'pairs':pairs}
    scales={}
    for scale in (.75,1.,1.5,2.5,4.):
        rows=[r for r in results if r['split']=='development' and r['tag']==f'legacy-recorded-{scale:g}x']
        good=[r for r in rows if r['status']=='ok']
        scales[str(scale)]={'ok':len(good),'deferred':len(rows)-len(good),
                           'mean_silhouette':float(np.mean([r['metrics']['silhouette_iou'] for r in good])) if good else None}
    summary={'comparisons':comparisons,'legacy_scale_sweep':scales,
             'total_runs':len(results),'ok':sum(r['status']=='ok' for r in results),
             'deferred':sum(r['status']=='resource_deferred' for r in results),
             'baseline_exact_replays':[r['uid'] for r in results if r.get('baseline_exact_states') and r['tag']=='legacy-recorded-1x']}
    dump(OUT/'summary.json',summary)


def massing_check():
    sys.path.insert(0,str(R2M))
    from building.massing import build_massing
    from building.params import PARAMS
    masks={}
    box=np.zeros((32,24,32),bool);box[5:27,:18,5:27]=True;masks['rectilinear']=box
    x,z=np.meshgrid(np.arange(32)-15.5,np.arange(32)-15.5,indexing='ij')
    cylinder=(x*x+z*z<11**2); masks['curved_tower']=np.repeat(cylinder[:,None,:],24,axis=1)
    open_structure=np.zeros((32,24,32),bool)
    open_structure[4:28,18:20,4:28]=True
    for a in (5,15,25):
        for b in (5,25): open_structure[a:a+2,:18,b:b+2]=True
    masks['open_colonnade']=open_structure
    # A source-defined one-block window is intentionally small; massing closes it.
    opening=box.copy();opening[5:27,8:10,14:16]=False;masks['through_opening']=opening
    result={}
    for name,raw in masks.items():
        ir,regular=build_massing(raw,PARAMS)
        result[name]={'iou':float((raw&regular).sum()/max((raw|regular).sum(),1)),
                      'added':int((regular&~raw).sum()),'removed':int((raw&~regular).sum()),'mean_level_iou':ir['mean_iou']}
    dump(OUT/'massing-reuse.json',result)
    return result


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--remeasure',action='store_true');ap.add_argument('--massing',action='store_true')
    args=ap.parse_args()
    if args.remeasure:
        for path in sorted((OUT/'runs').glob('*/*/result.json')):
            remeasure(path)
            print(str(path.parent.relative_to(OUT)),flush=True)
    if args.massing: print(json.dumps(massing_check()))
    summarize(gallery())


if __name__=='__main__':main()
