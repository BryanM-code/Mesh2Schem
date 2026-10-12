"""How many blocks tall is this building? Real size at 1:1 with the player.

One block is one metre (Steve is 1.8 blocks tall), so the target of a 9 m house is 9 blocks
tall and the target of a 150 m tower is 150 blocks tall: proportions survive, nothing is
normalised to a common size.

Asset-library files do not state their real size reliably, and an uncertain size is NOT a
reason to throw a building away. Every estimate falls in one of three classes:

* **Confident** -- the file's units agree with the caption's storey count (high; medium when
  the file is 1.5-1.75x taller, as a plinth, plot or tall roof makes it), or the
  file read as metres (the glTF unit) gives a height typical of the category (within 2.5x of
  its typical height), or only the
  storey count is available (medium).
* **Uncertain** -- metres give a height unusual for the category (outside its range, or more
  than 2.5x from its typical height), or another unit (cm, mm,
  inches, feet) had to be assumed, or the model is unitless (normalised to about one unit)
  and gets a typical height for its category. The building is KEPT and marked
  `scale_uncertain`; how to weight such pairs is a training-time decision.
* **Broken** -- the file's numbers are not a size at all (zero, not finite, or beyond any
  reading: under 1e-5 or over 1e7 units). Only these are rejected (`scale_broken`).

A building taller than `cfg.max_height_blocks` is not rejected either: its scale is reduced
until it fits and it is marked `height_capped` (Minecraft cannot hold it at 1:1); likewise a
model whose scaled bounding box exceeds `cfg.max_grid_voxels` is marked `size_capped`. Normal
buildings are never compressed.
"""
from __future__ import annotations

import math

import numpy as np

# (typical min, typical max, typical) overall height in metres, per category: evidence for
# how CONFIDENT a size is, never a reason to reject one.
HEIGHT_M = {
    "house": (3.0, 25.0, 8.0),
    "townhouse": (5.0, 30.0, 11.0),
    "apartment": (8.0, 180.0, 25.0),
    "midrise_generic": (10.0, 70.0, 24.0),
    "skyscraper": (35.0, 900.0, 140.0),
    "office": (6.0, 400.0, 30.0),
    "commercial": (3.0, 80.0, 8.0),
    "institutional": (4.0, 100.0, 15.0),
    "religious": (5.0, 180.0, 25.0),
    "tower": (6.0, 900.0, 30.0),
    "fortification": (4.0, 120.0, 20.0),
    "industrial": (3.0, 80.0, 10.0),
    "pavilion": (2.5, 20.0, 5.0),
    "building_generic": (3.0, 200.0, 12.0),
}
BUILDING_M = (2.0, 900.0)       # any building of any kind
STOREY_M = {"house": 3.0, "townhouse": 3.0, "apartment": 3.0, "midrise_generic": 3.2,
            "skyscraper": 3.8, "office": 3.8, "commercial": 4.0, "institutional": 4.2,
            "industrial": 5.0, "religious": 5.0, "fortification": 4.5, "tower": 3.5,
            "pavilion": 3.5, "building_generic": 3.3}
ROOF_ALLOWANCE = 1.15            # storeys -> overall height with roof / parapet
STOREYS_TALLER = 1.75            # a file height up to this x the storeys' still agrees with them
TYPICAL_SPREAD = 2.5             # metres within this factor of the category's typical height: medium
UNITS = (("m", 1.0), ("cm", 0.01), ("mm", 0.001), ("in", 0.0254), ("ft", 0.3048))
BROKEN_UNITS = (1e-5, 1e7)       # a model outside this is not a size in any unit


class ScaleRejected(Exception):
    def __init__(self, reason, detail, info):
        super().__init__("{}: {}".format(reason, detail))
        self.reason, self.detail, self.info = reason, detail, info


def estimate_height_m(record: dict, height_units: float, rng: np.random.Generator):
    """(height in metres, info) for a model `height_units` tall in its own file units.

    info: source, unit, confidence (high / medium / low) and notes. Raises ScaleRejected
    (`scale_broken`) only when the file's height is not a size under any reading.
    """
    cat = record.get("category") or "building_generic"
    lo, hi, typical = HEIGHT_M.get(cat, HEIGHT_M["building_generic"])
    info = {"category": cat, "typical_m": [lo, hi], "height_units": round(float(height_units), 6), "notes": []}
    if not math.isfinite(height_units) or not BROKEN_UNITS[0] <= height_units <= BROKEN_UNITS[1]:
        raise ScaleRejected("scale_broken", "model is {:g} units tall".format(height_units), info)
    n = record.get("storeys")
    n = int(n) if n not in (None, "", "None") else None
    readings = [(u, height_units * f) for u, f in UNITS]
    building = [(u, h) for u, h in readings if BUILDING_M[0] <= h <= BUILDING_M[1]]
    typical_fit = [(u, h) for u, h in readings if lo <= h <= hi]
    metres = height_units

    def done(h, source, confidence, unit=None):
        info.update(source=source, confidence=confidence)
        if unit:
            info["unit"] = unit
        return h, info

    if n:
        storeys_h = n * STOREY_M.get(cat, 3.3) * ROOF_ALLOWANCE
        info.update(storeys=n, storeys_height_m=round(storeys_h, 2))
        # A model may be taller than its storeys say (a plinth, a plot, a tall roof or a
        # rooftop structure), not shorter: the window is wider upwards, and confidence drops
        # from high to medium beyond 1.5x.
        agree = [(u, h) for u, h in building if 1 / 1.5 <= h / storeys_h <= STOREYS_TALLER]
        if agree:
            u, h = min(agree, key=lambda uh: (abs(math.log(uh[1] / storeys_h)), uh[0] != "m"))
            return done(h, "file_units_agree_with_storeys", "high" if h / storeys_h <= 1.5 else "medium", u)
        info["notes"].append("file units disagree with the storey count")
        return done(storeys_h, "caption_storeys", "medium")
    if BUILDING_M[0] <= metres <= BUILDING_M[1]:
        if lo <= metres <= hi:
            # Inside the category's range but far from its typical height (a 22 m "hut"):
            # still metres, but not a confident size.
            if not typical / TYPICAL_SPREAD <= metres <= typical * TYPICAL_SPREAD:
                info["notes"].append("{:.1f} m is far from a typical {} ({:.0f} m)".format(metres, cat, typical))
                return done(metres, "file_units", "low", "m")
            return done(metres, "file_units", "medium", "m")
        # Metres is the glTF unit: keep it unless it is far outside the category's range
        # while another unit fits the category.
        others = [(u, h) for u, h in typical_fit if u != "m"]
        far = metres > 3 * hi or metres < lo / 3
        if others and far:
            u, h = min(others, key=lambda uh: abs(math.log(uh[1] / typical)))
            info["notes"].append("metres ({:.1f} m) far outside the category; read as {}".format(metres, u))
            return done(h, "file_units_other", "low", u)
        info["notes"].append("{:.1f} m is unusual for {}".format(metres, cat))
        return done(metres, "file_units", "low", "m")
    for pool in (typical_fit, building):
        if pool:
            u, h = min(pool, key=lambda uh: abs(math.log(uh[1] / typical)))
            info["notes"].append("metres impossible; read as {}".format(u))
            return done(h, "file_units_other", "low", u)
    # Unitless (typically normalised to about one unit): a typical size for the category.
    h = float(np.clip(typical * math.exp(rng.normal(0.0, 0.2)), lo, hi))
    info["notes"].append("no unit gives a building-sized height; category prior")
    return done(h, "category_prior", "low")


def choose_scale(record: dict, scene, cfg, rng: np.random.Generator):
    """(blocks per file unit, info) for an oriented scene. Raises ScaleRejected only for a
    broken scale (or a site too large to process at all)."""
    lo, hi = scene.bounds
    ext = hi - lo
    height_u = float(ext[1])
    height_m, info = estimate_height_m(record, height_u, rng)
    jitter = float(rng.uniform(1.0 - cfg.scale_jitter, 1.0 + cfg.scale_jitter))
    if cfg.scale_policy == 'architecture':
        from mcvoxel.architecture import resolution_evidence
        physical_bpu = height_m * cfg.blocks_per_metre / height_u
        evidence = resolution_evidence(scene.verts, scene.faces, physical_bpu, cfg.architecture_feature_blocks)
        # The pilot shows that repeating trim can masquerade as feature spacing.
        # This is advisory evidence until source/scale review establishes which
        # features matter. The requested representation multiplier is explicit.
        enlargement = float(cfg.architecture_enlargement)
        if not math.isfinite(enlargement) or enlargement < 1:
            raise ValueError('architecture_enlargement must be finite and at least 1')
        bpu = physical_bpu * enlargement
        dims = np.ceil(ext * bpu + 2 * cfg.margin + 2).astype(np.int64)
        info.update(height_m=round(height_m, 2), physical_blocks_per_unit=physical_bpu,
                    blocks_per_metre=cfg.blocks_per_metre, effective_blocks_per_metre=cfg.blocks_per_metre*enlargement,
                    blocks_per_unit=bpu, height_blocks=float(height_u*bpu),
                    bbox_footprint_blocks=float(max(ext[0], ext[2])*bpu), jitter=1.,
                    enlargement=enlargement, representation_version='architecture-v1', resolution=evidence,
                    height_capped=False, size_capped=False, scale_uncertain=info['confidence']=='low',
                    resolution_uncertain=evidence['confidence']!='high', required_grid_xyz=dims.tolist())
        exceeds = []
        if int(np.prod(dims)) > cfg.max_grid_voxels: exceeds.append('grid_voxels')
        if ext[1]*bpu > cfg.max_height_blocks: exceeds.append('height_blocks')
        if max(ext[0],ext[2])*bpu > cfg.max_site_blocks: exceeds.append('site_blocks')
        if exceeds:
            info['resource_status'] = 'deferred'; info['exceeded_limits'] = exceeds
            raise ScaleRejected('resolution_budget', 'requested architectural scale exceeds '+', '.join(exceeds)+
                                '; increase limits or explicitly revise representation scale; no automatic downscale', info)
        info['resource_status'] = 'fits'
        return bpu, info
    if cfg.scale_policy != 'legacy':
        raise ValueError('unknown scale_policy: '+cfg.scale_policy)
    bpu = height_m * cfg.blocks_per_metre * jitter / height_u
    info["height_capped"] = False
    if height_u * bpu > cfg.max_height_blocks:
        bpu = cfg.max_height_blocks / height_u
        info["height_capped"] = True
        info["notes"].append("taller than {} blocks at 1:1: scaled down to fit".format(cfg.max_height_blocks))
    # An engineering bound, not a size judgement: the bounding box must be voxelisable in
    # minutes. A larger model keeps its proportions at a smaller scale.
    info["size_capped"] = False
    box = float(np.prod(np.maximum(ext * bpu, 1.0)))
    if box > cfg.max_grid_voxels:
        bpu *= (cfg.max_grid_voxels / box) ** (1.0 / 3.0)
        info["size_capped"] = True
        info["notes"].append("bounding box over {:,} voxels at this scale: scaled down to fit".format(cfg.max_grid_voxels))
    info["scale_uncertain"] = info["confidence"] == "low" or info["height_capped"] or info["size_capped"]
    info.update(height_m=round(height_m, 2), jitter=round(jitter, 4), blocks_per_metre=cfg.blocks_per_metre,
                effective_blocks_per_metre=round(bpu * height_u / height_m, 6),
                blocks_per_unit=bpu, height_blocks=round(height_u * bpu, 1),
                bbox_footprint_blocks=round(float(max(ext[0], ext[2])) * bpu, 1))
    # An engineering bound, not a quality judgement: a site this large is not one building's
    # worth of voxels (its plot would dominate the grid).
    if info["bbox_footprint_blocks"] > cfg.max_site_blocks:
        raise ScaleRejected("too_large", "{:.0f}-block site".format(info["bbox_footprint_blocks"]), info)
    return bpu, info


def building_size_verdict(solid, cfg):
    """(reason or None, detail, dims) for the voxelised building (plot already removed).

    Only a building too small to be represented in blocks at all is rejected; a small
    building is kept and flagged `small_building`."""
    X, Y, Z = solid.shape
    fp = int(max(X, Z))
    dims = {"footprint_blocks": fp, "width_blocks": int(min(X, Z)), "height_blocks": int(Y),
            "small_building": bool(fp < cfg.small_building_blocks and Y < cfg.small_building_blocks)}
    if fp < cfg.min_representable_blocks or Y < cfg.min_representable_blocks:
        return "degenerate_size", "{} x {} blocks cannot represent a building".format(fp, Y), dims
    if fp > cfg.max_footprint_blocks:
        return "too_large", "{} > {} blocks".format(fp, cfg.max_footprint_blocks), dims
    return None, "", dims
