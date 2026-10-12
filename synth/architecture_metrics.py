"""Source-conditioned diagnostics, not a visual-approval score.

Projection fidelity includes actual partial-block silhouettes; openings are measured
only where source geometry exposes air. Texture-painted windows are not openings.
Separate roofs, stepped curves and ornament are never scored as disconnected errors.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def projected_masks(verts, faces, grid, names, sub=2):
    from mcvoxel.roofs import top_surface
    from mcvoxel.vocab import _props
    import thumbs

    xyz = np.asarray(grid).transpose(2, 0, 1)
    dims = xyz.shape
    boxes = [thumbs._shape_boxes(n.split('[')[0], _props(n)) if n != 'minecraft:air' else [] for n in names]
    results = []
    for a, b, depth in ((0, 1, 2), (2, 1, 0), (0, 2, 1)):
        # top_surface's projection axes are 0 and 2; its height axis is 1.
        h, _ = top_surface(np.asarray(verts)[:, [a, depth, b]], faces, (dims[a], dims[b]), sub)
        source = np.isfinite(h)
        target = np.zeros_like(source)
        for state in np.unique(xyz):
            if not boxes[state]:
                continue
            coords = np.argwhere(xyz == state)
            # Project identical block shapes in batches, unioning all depth layers.
            uv = np.unique(coords[:, [a, b]], axis=0)
            for box in boxes[state]:
                for i in range(sub):
                    for j in range(sub):
                        if box[a] <= (i + .5) / sub < box[a + 3] and box[b] <= (j + .5) / sub < box[b + 3]:
                            target[uv[:, 0] * sub + i, uv[:, 1] * sub + j] = True
        results.append((source, target))
    return results


def measure(verts, faces, grid, names, plan, vm=None):
    from mcvoxel.block_shapes import base_block

    projections = projected_masks(verts, faces, grid, names)
    views = []
    wall_errors, wall_segments = [], 0
    repeated_expected = repeated_missing = 0
    repeated_errors = []
    for view_index, (source, target) in enumerate(projections):
        # Source-empty pockets inside its silhouette bound are evidence of openings.
        # Enclosed holes and open bays between supports both count; outside remains unscored.
        hull = ndimage.binary_fill_holes(source)
        along_u = np.maximum.accumulate(source, 0) & np.maximum.accumulate(source[::-1], 0)[::-1]
        along_v = np.maximum.accumulate(source, 1) & np.maximum.accumulate(source[:, ::-1], 1)[:, ::-1]
        opening = ~source & (hull | (along_u & along_v))
        # One half-block boundary uncertainty band excludes aliasing from the opening test.
        opening &= ~ndimage.binary_dilation(source)
        if view_index < 2:
            # Score only source-straight facade silhouette segments, not roof tapers,
            # curved outlines, setbacks or asymmetric architecture.
            shas, thas = source.any(0), target.any(0)
            for sign in (1, -1):
                ss = source if sign == 1 else source[::-1]
                ts = target if sign == 1 else target[::-1]
                se, te = np.argmax(ss, 0), np.argmax(ts, 0)
                cuts = np.r_[0, np.flatnonzero((np.diff(se) != 0) | ~shas[1:] | ~shas[:-1]) + 1, len(se)]
                for start, stop in zip(cuts[:-1], cuts[1:]):
                    if stop-start < 8 or not shas[start:stop].all():
                        continue
                    wall_segments += 1
                    delta = te[start:stop].astype(float)-se[start:stop]
                    delta[~thas[start:stop]] = source.shape[0]
                    wall_errors.append(float(np.mean(np.abs(delta-np.median(delta)))) / 2)
            labels, _ = ndimage.label(opening)
            components = []
            for label, box in enumerate(ndimage.find_objects(labels), 1):
                if box is None:
                    continue
                mask = labels[box] == label
                if mask.sum() >= 4:
                    cells = np.argwhere(mask) + [s.start for s in box]
                    components.append((label, cells.mean(0), len(cells), cells))
            # Only rows with three or more similarly sized source openings are repeated.
            for label, centre, area, cells in components:
                peers = [c for c in components if abs(c[1][1]-centre[1]) <= 1 and .5 <= c[2]/area <= 2]
                if len(peers) < 3:
                    continue
                repeated_expected += 1
                remaining = cells[~target[tuple(cells.T)]]
                if len(remaining) < len(cells) * .25:
                    repeated_missing += 1
                else:
                    repeated_errors.append(float(np.linalg.norm(remaining.mean(0)-centre)) / 2)
        dist = ndimage.distance_transform_edt(~source)
        views.append({'silhouette_iou': float((source & target).sum() / max((source | target).sum(), 1)),
                      'missing_source_fraction': float((source & ~target).sum() / max(source.sum(), 1)),
                      'unexplained_protrusion_fraction': float((target & (dist > 2)).sum() / max(target.sum(), 1)),
                      'opening_samples': int(opening.sum()),
                      'opening_filled_samples': int((opening & target).sum())})
    xyz = np.asarray(grid).transpose(2, 0, 1)
    lut = {b: i for i, b in enumerate(sorted(set(base_block(n) for n in names)))}
    material = np.array([lut[base_block(n)] for n in names])[xyz]
    solid = xyz != names.index('minecraft:air')
    # Surface-local material singleton: diagnostic only, intentional trim is possible.
    same = np.zeros(solid.shape, np.uint8)
    adjacent = np.zeros(solid.shape, np.uint8)
    for ax in range(3):
        l = [slice(None)] * 3; r = l.copy()
        l[ax] = slice(1, None); r[ax] = slice(None, -1)
        l, r = tuple(l), tuple(r)
        both = solid[l] & solid[r]
        eq = both & (material[l] == material[r])
        same[l] += eq; same[r] += eq
        adjacent[l] += both; adjacent[r] += both
    decisions = plan.info.get('plane_decisions', [])
    planned = [r for r in decisions if r.get('planned', True)]
    # Within-plane cross-course neighbours should share cap height and stair direction.
    eligible = inconsistent = 0
    for row in planned:
        direction = row.get('direction')
        if direction is None:
            continue
        mask = plan.region == row['region']
        ax = 1 if direction in ('east', 'west') else 0
        l = [slice(None)] * 2; r = l.copy(); l[ax] = slice(1, None); r[ax] = slice(None, -1)
        l, r = tuple(l), tuple(r)
        edge = mask[l] & mask[r]
        eligible += int(edge.sum())
        inconsistent += int((edge & ((plan.level[l] != plan.level[r]) | (plan.shape[l] != plan.shape[r]))).sum())
    return {'schema': 'architecture-diagnostics-v1', 'projection_sub': 2, 'views': views,
            'silhouette_iou': float(np.mean([v['silhouette_iou'] for v in views])),
            'unexplained_protrusion_fraction': float(np.mean([v['unexplained_protrusion_fraction'] for v in views])),
            'opening_samples': sum(v['opening_samples'] for v in views),
            'opening_filled_samples': sum(v['opening_filled_samples'] for v in views),
            'source_straight_wall_segments': wall_segments,
            'wall_boundary_jitter_blocks': float(np.mean(wall_errors)) if wall_errors else None,
            'repeated_source_openings': repeated_expected, 'repeated_openings_missing': repeated_missing,
            'repeated_opening_alignment_blocks': float(np.mean(repeated_errors)) if repeated_errors else None,
            'material_singleton_fraction': float((solid & (same == 0) & (adjacent >= 2)).sum() / max(solid.sum(), 1)),
            'roof_regions': len(planned),
            'tiny_roof_columns': sum(r['columns'] for r in planned if r['columns'] < 4),
            'roof_planned_columns': int((plan.level >= 0).sum()),
            'course_cross_edges': eligible, 'course_cross_disagreements': inconsistent,
            'limitations': ['Projection unions cannot see occluded openings or painted windows.',
                           'Material singletons are warnings, not errors; trim is intentional.',
                           'Course metric is conditional on fitted regions and must be read with coverage and silhouette.']}
