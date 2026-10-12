"""The clean Minecraft target of one source model, in both roof styles.

    voxelise (mcvoxel.grid.voxelize_scene: surface, flood fill, enclosed rooms, ground plate,
              site around the building, largest part, slab / stair shapes, sheet slivers,
              colours seen from outside, glass and door kinds)
    size (synth.scale.building_size_verdict)          -> degenerate_size / too_large, else flags
    quality gates                                      -> scattered / open_model
    margins of supervised air (sides and top)
    Approach A: blocks from the geometry as it is (mcvoxel.assign.assign_blocks)
    Approach B: builder-style roofs (mcvoxel.roofs), then blocks, then one material per roof

`cfg.roof_style` names the primary target; the other is kept beside it when
`cfg.store_geometric` so the two can be compared on the same pairs (and trained on either).

The gates reject only what is not one building: a scene whose largest part is a minority
of it (`scattered`), and a model that encloses almost nothing AND has no roof over most of
its footprint (`open_model`: floor plans, interior sets, cutaways, roofless shells). An open
structure with a roof -- a pavilion, a market hall, a stadium stand -- passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mcvoxel import roofs
from mcvoxel.assign import assign_blocks
from mcvoxel.block_palette import BUILDER_PALETTE
from mcvoxel.grid import pad_model, voxelize_scene, shell_of

from .scale import building_size_verdict


class TargetRejected(Exception):
    def __init__(self, reason, detail, info):
        super().__init__("{}: {}".format(reason, detail))
        self.reason, self.detail, self.info = reason, detail, info


@dataclass
class TargetResult:
    vm_geometric: object              # padded VoxelModel, geometry as converted
    vm_builder: object                # padded VoxelModel with builder-style roof caps
    plan: object                      # mcvoxel.roofs.RoofPlan
    geometric: object                 # BlockTarget, Approach A
    builder: object                   # BlockTarget, Approach B
    info: dict = field(default_factory=dict)

    def primary(self, style):
        return (self.builder, self.vm_builder) if style == "builder" else (self.geometric, self.vm_geometric)


def site_height_blocks(cfg, scale_info=None) -> int:
    """`cfg.site_height_m` in voxels at this sample's scale (a capped scale shrinks it too)."""
    info = scale_info or {}
    hb, hm = info.get("height_blocks"), info.get("height_m")
    per_metre = float(hb) / float(hm) if hb and hm else cfg.blocks_per_metre
    return max(2, int(round(cfg.site_height_m * per_metre)))


def palette_ids(cfg):
    """The block ids targets may use, or None for the whole palette."""
    return list(BUILDER_PALETTE) if cfg.palette == "builder" else None


def quality(vm):
    """The numbers the gates read.

    interior_rel  interior share of the solid (voxels with no air neighbour) over the same
                  share for a closed box of the grid's size: ~1 for a closed building at any
                  size (a 4-block hut's box share is 0.19, a tower's 0.9), ~0 for an open one.
    roof_cover    share of the footprint's CONVEX HULL whose column reaches half the
                  building's height: a floor plan, a roofless shell, a cutaway (rooms open
                  down to an intermediate floor) or a film set of two walls covers little
                  beyond its walls. (Filling holes was not enough: an L of two walls has no
                  hole, so it came out fully "roofed".)
    """
    from skimage.morphology import convex_hull_image
    from scipy import ndimage

    solid = vm.solid
    X, Y, Z = solid.shape
    n = int(solid.sum())
    # Enclosure is a geometric property. Extra exported backing visible through slabs
    # must not make a short, closed building fail the open-model gate.
    share = int((solid & ~shell_of(solid)).sum()) / max(n, 1)
    box = max(X - 2, 0) * max(Y - 1, 0) * max(Z - 2, 0) / max(X * Y * Z, 1)
    cols = solid.any(axis=1)
    hull = convex_hull_image(cols) if cols.sum() > 2 else cols
    top = np.where(cols, Y - 1 - np.argmax(solid[:, ::-1, :], axis=1), -1)
    y0 = int(np.flatnonzero(solid.any(axis=(0, 2)))[0]) if n else 0
    roofed = hull & (top >= y0 + max(min(2, Y - 1), (Y - 1) // 2))
    return {"interior_share": round(share, 4),
            "interior_rel": round(share / box, 4) if box > 0 else 1.0,
            "roof_cover": round(int(roofed.sum()) / max(int(hull.sum()), 1), 4),
            "kept_fraction": vm.info.get("kept_fraction", 1.0)}


def make_target(scene, blocks_per_unit, offset, cfg, max_plate, scale_info=None, detail_material_hint=None) -> TargetResult:
    vm = voxelize_scene(scene, blocks_per_unit, offset, fine_scale=cfg.fine_scale, max_plate=max_plate,
                        site_height=site_height_blocks(cfg, scale_info), site_margin=cfg.site_margin,
                        color_mode=cfg.color_mode, clean=cfg.close_gaps,
                        architecture=cfg.architecture_mode == 'coherent')
    info = {"voxels": vm.info}
    if detail_material_hint in ("wood", "masonry") and vm.post_family is not None:
        vm.post_family[vm.post_mask & (vm.post_family == 0)] = 1 if detail_material_hint == "wood" else 2
        info["detail_material_hint"] = detail_material_hint
    if not vm.shell.any():
        raise TargetRejected("empty_target", "", info)
    g = cfg.gates
    if vm.info["kept_fraction"] < g["scattered_kept"]:
        raise TargetRejected("scattered", "largest part keeps {:.0%}".format(vm.info["kept_fraction"]), info)
    reason, detail, dims = building_size_verdict(vm.solid, cfg)
    info["dims"] = dims
    if reason:
        raise TargetRejected(reason, detail, info)
    if int(vm.shell.sum()) > cfg.max_shell_voxels:
        raise TargetRejected("too_many_voxels", "{:,} shell voxels".format(int(vm.shell.sum())), info)
    qa = quality(vm)
    info["qa"] = qa
    if qa["interior_rel"] < g["open_interior_rel"] and qa["roof_cover"] < g["open_roof_cover"]:
        raise TargetRejected("open_model", "interior_rel={} roof_cover={}".format(
            qa["interior_rel"], qa["roof_cover"]), info)

    m = cfg.margin
    vm = pad_model(vm, ((m, m), (0, m), (m, m)))
    verts_grid = scene.verts * blocks_per_unit + np.asarray(offset, dtype=np.float64) - vm.origin
    vm_b, plan = roofs.builder_roofs(vm, verts_grid, scene.faces,
                                   roofs.RoofParams(sub=cfg.fine_scale, coherent=cfg.architecture_mode == 'coherent'), scene=scene)
    pal = palette_ids(cfg)
    kw = {"shadow_repaint": cfg.shadow_repaint, "palette": pal, "quantize": cfg.quantize}
    kw['coherent_materials'] = cfg.architecture_mode == 'coherent'
    geometric = assign_blocks(vm, **kw)
    builder = assign_blocks(vm_b, **kw)
    info["roofs"] = {**vm_b.info.get("roofs", {}),
                     **roofs.harmonize_roof_materials(builder, plan, palette=pal, rgb=vm_b.rgb)}
    info["blocks_geometric"] = geometric.info
    info["blocks_builder"] = builder.info
    return TargetResult(vm_geometric=vm, vm_builder=vm_b, plan=plan, geometric=geometric, builder=builder,
                        info=info)
