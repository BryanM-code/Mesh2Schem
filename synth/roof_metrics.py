"""Measurements of emitted roof shapes, including the voids inside partial blocks.

These measure topology, not visual quality. The mesh-derived plan defines the footprint;
source openings and glass skylights are excluded. No binary closing participates.
"""
from __future__ import annotations

import numpy as np


def measure_roof(grid, names, plan):
    from scipy import ndimage
    import thumbs

    grid = np.asarray(grid).transpose(2, 0, 1)  # XYZ
    X, Y, Z = grid.shape
    # Include broad flat roofs as well as pitched material groups. Preserved narrow
    # parapets, glass and unsampled openings have level=-1 and are outside this metric.
    roof = plan.level >= 0
    air_ids = [i for i,n in enumerate(names) if n in ("minecraft:air", "minecraft:cave_air", "minecraft:void_air")]
    has = ~np.isin(grid,air_ids)
    top = np.where(has.any(axis=1), Y - 1 - np.argmax(has[:, ::-1, :], axis=1), -1)
    masks = []
    centres = (np.arange(4) + .5) / 4
    xyz = np.stack(np.meshgrid(centres, centres, centres, indexing="ij"), axis=-1)
    for name in names:
        mask = np.zeros((4, 4, 4), bool)
        if name not in ("minecraft:air", "minecraft:cave_air", "minecraft:void_air"):
            from mcvoxel.vocab import _props
            for box in thumbs._shape_boxes(name.split('[')[0], _props(name)):
                mask |= ((xyz >= np.array(box[:3])) & (xyz < np.array(box[3:]))).all(axis=-1)
        masks.append(mask)
    masks = np.array(masks)
    cap_missing = int((roof & ((top < 0) | (np.abs(top - plan.level) > 2))).sum())
    gaps = unsupported = 0

    def at(x, y, z):
        return masks[int(grid[x, y, z])] if 0 <= x < X and 0 <= y < Y and 0 <= z < Z else np.zeros((4,4,4),bool)

    def contact(x, y, z, axis, step):
        q = [x, y, z]
        q[axis] += step
        return bool((np.take(at(x,y,z), -1 if step > 0 else 0, axis=axis) &
                     np.take(at(*q), 0 if step > 0 else -1, axis=axis)).any())

    for x, z in np.argwhere(roof):
        t = int(top[x, z])
        if t < 0:
            continue
        unsupported += int(not any(contact(x, t, z, a, d) for a in range(3) for d in (-1, 1)))
        for dx, dz, axis in ((1,0,0), (0,1,2)):
            a, b = x+dx, z+dz
            if a >= X or b >= Z or not roof[a,b] or abs(int(plan.level[x,z])-int(plan.level[a,b])) > 2:
                continue
            nt = int(top[a,b])
            if nt < 0:
                continue
            # A visible seam exists if no block face joins the courses at or above
            # the lower cap. Joining through distant walls does not hide that seam.
            gaps += int(not any(contact(x,y,z,axis,1) for y in range(max(0,min(t,nt)),max(t,nt)+1)))
    # Face-connected pieces in a shallow emitted roof band, at half-block resolution.
    band = np.zeros((X*2, Y*2, Z*2), bool)
    for x,z in np.argwhere(roof):
        t = int(top[x,z])
        for y in range(max(0,t-2),t+1):
            band[2*x:2*x+2,2*y:2*y+2,2*z:2*z+2] = masks[grid[x,y,z]][::2,::2,::2]
    _, components = ndimage.label(band)
    return {"roof_columns": int(roof.sum()), "missing_caps": cap_missing,
            "open_course_interfaces": gaps, "roof_holes": cap_missing + gaps,
            "disconnected_roof_components": max(0, int(components)-1),
            "roof_components": int(components), "unsupported_roof_caps": int(unsupported)}
