"""Source-evidenced architecture decisions shared by dataset and application callers.

No asset names, category templates or metre assumptions enter these functions.
Geometry is never normalized to a common building size. Low confidence preserves
the mesh; all distances here are in the caller's block frame unless documented.
"""
from __future__ import annotations

import numpy as np


def enclosed_cross_sections(occupied):
    """Fill only roof-covered, closed horizontal cross sections.

    A floor and three walls are insufficient evidence: open fronts, arcades and
    breezeways stay open. Closed surfaces still use the ordinary 3-D flood fill.
    """
    from scipy import ndimage
    above = np.maximum.accumulate(occupied[:, ::-1, :], axis=1)[:, ::-1, :]
    added = np.zeros_like(occupied)
    for y in range(occupied.shape[1]):
        added[:, y, :] = ndimage.binary_fill_holes(occupied[:, y, :]) & ~occupied[:, y, :] & above[:, y, :]
    return added


def coherent_wall_shapes(solid, shapes, verts, faces, origin):
    """Continuous vertical mesh surfaces use full wall courses, including curves.

    Occupancy, openings and profiles are unchanged. A cell must be backed by a
    vertical source face, continue both above and below and have no inclined or
    horizontal mesh face. Tops, window edges, ledges and stair runs stay geometric.
    """
    from .raster import rasterize_surface
    from .block_shapes import SHAPE_FULL
    tri = np.asarray(verts)[np.asarray(faces)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vertical = np.abs(n[:, 1]) <= .04 * np.linalg.norm(n, axis=1)
    walls, _ = rasterize_surface(verts, np.asarray(faces)[vertical], origin=origin, shape=solid.shape)
    other, _ = rasterize_surface(verts, np.asarray(faces)[~vertical], origin=origin, shape=solid.shape)
    supported = np.zeros_like(solid)
    supported[:, 1:-1, :] = solid[:, :-2, :] & solid[:, 2:, :]
    mask = walls & ~other & supported & solid & (shapes > SHAPE_FULL)
    out = shapes.copy(); out[mask] = SHAPE_FULL
    return out, {'vertical_surface_full_cells': int(mask.sum()), 'occupancy_changed': 0,
                 'method': 'source_vertical_faces_and_vertical_continuity'}
