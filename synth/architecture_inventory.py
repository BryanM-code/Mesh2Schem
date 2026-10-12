"""Freeze provenance and choose a category-stratified architecture audit (no downloads)."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R2M = ROOT.parent / 'Real2Minecraft'
BASE = ROOT / 'data/synthetic/pilot-v5-200-20261001'
OUT = ROOT / 'data/architecture-audit-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def supplement(rows, evaluation):
    """Fill architectural coverage gaps without changing the category sample.

    Added before any implementation outcomes: the taxonomy admits tools as industrial
    and temples as religious. Caption evidence supplies missing morphology coverage.
    """
    rules = [('church-nave', r'church.*(?:gable|gothic|nave)'),
             ('industrial-building', r'\bbarn\b'),
             ('open-bay', r'open front supported by posts')]
    used = {r['uid'] for r in evaluation}
    for role, pattern in rules:
        candidates = [r for r in rows if r['cached'] and re.search(pattern, r.get('caption') or '', re.I)]
        if candidates:
            chosen = sorted(candidates, key=lambda r: r['uid'])[0]
            if chosen['uid'] not in used:
                evaluation.append({**chosen, 'split': 'development', 'selection_role': role})
                used.add(chosen['uid'])
    return evaluation


def main():
    if (OUT / 'evaluation-set.json').exists():
        raise SystemExit('Frozen inventory already exists; refusing to replace it.')
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for f in sorted(BASE.rglob('*')):
        if f.is_file():
            inventory[f.relative_to(BASE).as_posix()] = {'bytes': f.stat().st_size, 'sha256': sha(f)}
    dump(OUT / 'baseline-files.json', inventory)
    for repo in (ROOT, R2M):
        snap = OUT / 'frozen' / repo.name
        for folder in ('synth', 'mcvoxel', 'building', 'tests'):
            if (repo / folder).exists():
                for f in (repo / folder).rglob('*'):
                    if f.is_file() and f.suffix in ('.py', '.json', '.md') and '__pycache__' not in f.parts:
                        target = snap / f.relative_to(repo)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, target)
        for f in repo.glob('*'):
            if f.is_file() and f.suffix in ('.py', '.toml', '.txt', '.json'):
                snap.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, snap / f.name)
        for args, name in ((['status', '--short'], 'initial-status.txt'),
                           (['diff', '--binary', 'HEAD'], 'initial-diff.patch'),
                           (['rev-parse', 'HEAD'], 'head.txt')):
            (snap / name).write_bytes(subprocess.check_output(['git', *args], cwd=repo))
    selected = {r['uid']: r for r in map(json.loads, (BASE / 'selection.jsonl').read_text(encoding='utf-8').splitlines())}
    rows = []
    for f in sorted((BASE / 'samples').glob('*/manifest.json')):
        m = json.loads(f.read_text(encoding='utf-8'))
        if m['status'] != 'ok':
            continue
        uid = m['uid']
        s, t = m['scale'], m['target']
        roof = t.get('roofs', {})
        topo = m.get('roof_topology', {})
        if 'builder' in topo:
            topo = topo['builder']
        n = max(roof.get('planned', 0), 1)
        decisions = roof.get('plane_decisions', [])
        tiny = sum(r['columns'] for r in decisions if r['columns'] < 4)
        geom = m.get('checks', {})
        source = selected[uid]
        rows.append({'uid': uid, 'name': source['name'], 'category': source['category'],
                     'height_blocks': s['height_blocks'], 'footprint_blocks': s['bbox_footprint_blocks'],
                     'confidence': s['confidence'], 'capped': s.get('size_capped') or s.get('height_capped'),
                     'faces': m['scene']['faces'], 'materials': m['scene']['materials'],
                     'planar_share': geom.get('planarity', {}).get('planar_share'),
                     'roof_regions': roof.get('regions', 0), 'tiny_roof_share': tiny / n,
                     'roof_hole_rate': topo.get('roof_holes', 0) / n,
                     'roof_unsupported_rate': topo.get('unsupported_roof_caps', 0) / n,
                     'seed': m['seed'], 'source_sha256': source.get('sha256'),
                     'cached': (BASE / 'raw-cache' / (uid + '.glb')).exists(),
                     'caption': source.get('caption'), 'source_flags': source.get('flags', {})})
    # Name-blind, deterministic stratification. Four low-fragmentation controls;
    # remaining development examples alternate median and high fragmentation.
    # Reserved examples maximize size/complexity distance from the development row.
    import numpy as np
    evaluation = []
    for ci, cat in enumerate(sorted(set(r['category'] for r in rows))):
        pool = [r for r in rows if r['category'] == cat and r['cached']]
        pool.sort(key=lambda r: (r['tiny_roof_share'] + r['roof_hole_rate'], r['uid']))
        rank = 0 if ci % 4 == 0 else (len(pool) // 2 if ci % 2 == 0 else max(0, len(pool) - 2))
        dev = pool[rank]
        features = np.array([[np.log1p(r[k]) for k in ('height_blocks', 'footprint_blocks', 'faces', 'materials')]
                             + [r['tiny_roof_share'], r['roof_hole_rate']] for r in pool])
        features = (features - features.mean(0)) / np.maximum(features.std(0), .2)
        distances = np.linalg.norm(features - features[rank], axis=1)
        distances[rank] = -1
        reserved = pool[int(np.argmax(distances))]
        evaluation.extend([{**dev, 'split': 'development', 'selection_role': 'control-candidate' if rank == 0 else 'failure-diversity'},
                           {**reserved, 'split': 'reserved', 'selection_role': 'farthest-size-complexity'}])
    evaluation = supplement(rows, evaluation)
    dump(OUT / 'pilot-inventory.json', rows)
    dump(OUT / 'evaluation-set.json', {'version': 1, 'baseline': str(BASE),
         'selection': 'Two per supported category with cached sources. Development ranks alternate low/median/high baseline roof fragmentation; reserved maximizes standardized log-size, faces, materials and fragmentation distance. Names and captions are not selection inputs. Reserved results are inspected only after implementation is frozen.',
         'rows': evaluation})
    print(json.dumps({'baseline_files': len(inventory), 'categories': dict(Counter(r['category'] for r in rows)),
                      'selected': [(r['split'], r['category'], r['name'], r['height_blocks']) for r in evaluation]}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
