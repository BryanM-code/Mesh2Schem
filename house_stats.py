#!/usr/bin/env python3
"""Measure the shape of every house in the library, so a generator can imitate it.

Usage:  python house_stats.py [--out data/build_stats.json] [--only <substring>] [--quiet]

`real2minecraft` needs far more (image, schem) training pairs than can be collected by hand.
The plan is to synthesise suburban houses procedurally -- but a generator whose proportions,
materials and window spacing were *guessed* would produce builds from a different distribution
than the real ones, which is the one thing that would make the synthetic data worse than
useless. So this script measures the real library first, and the generator samples from what
it finds. Phase 1 and Phase 2 are deliberately separate programs: the numbers get reviewed by
a human before anything is generated from them.

The hard part is not the statistics, it is deciding *what to measure on*. A stored build is
not a bare house -- it is a house sitting on a chunk of landscape, and by volume the three
most common blocks in the whole library are stone, dirt and grass. Read the palette naively
and it will tell you the most popular wall material is dirt. So the first job, and most of the
subtlety here, is separating the plot from the building:

*   Terrain coverage is normalised by the **largest terrain layer**, not by the bounding box.
    `10__westlahome` has a bbox inflated by trees to 6435 columns while its pad covers ~1950;
    against the bbox the pad reads as 30% coverage and vanishes under any sane threshold.
*   That normaliser needs a floor, or a build with almost no terrain at all invents a pad out
    of its own noise -- `102__traditionhouse3`'s biggest terrain layer is 206 columns of 2703.
*   Real builds break the simple model in specific ways, each of which gets a flag rather than
    a silent wrong answer: a stray full-coverage layer with the real ground eleven layers
    above it (`49__pmc-suburban-house`), fifteen solid layers of stone under the house
    (`11__lastylehome`), a hillside that tapers instead of stopping (`12__chaparalsthome`),
    and builds that sit on a built platform with no terrain material at all.

The second subtlety is that block *role* is positional, not lexical. The median build here
uses 105 distinct blockstates, most of them furniture, so the wall material cannot be found by
counting. It is found by looking only at the perimeter columns of the wall band, and only at
blocks that are full cubes. `block_roles` owns that vocabulary, shared with the generator so
the two halves cannot drift apart.

Builds that cannot be read as a single house are **reported with a reason**, not dropped.
Knowing that eleven builds were towers or multi-building plots is part of the finding.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import block_roles
import schem_voxels

BUILDS_DIR = Path("data/builds")
DEFAULT_OUT = Path("data/build_stats.json")

FACES = ("north", "south", "west", "east")

# Segmentation thresholds. Every one of these was set against a specific build that broke the
# naive version; the docstring names them.
MIN_TERRAIN_FRACTION = 0.15   # below this share of the bbox, there is no pad to find
PAD_START_COVERAGE = 0.60     # a pad's base layer must be this dense in terrain
PAD_CONTINUE_COVERAGE = 0.50  # ...and it keeps going while layers stay this dense
MAX_AIR_GAP = 2               # empty layers between "pad" and structure before we distrust it
DEEP_TERRAIN_SHARE = 0.40     # pad taller than this share of the build is a landscape block
MIN_STRUCTURE_VOXELS = 400
MIN_COMPONENT_SHARE = 0.55    # largest building must be this much of the structure mass
MAX_TOWER_RATIO = 3.0         # height / min(width, depth) above this is a tower, not a house

FLOOR_MIN_FILL = 0.35
FLOOR_MIN_SEPARATION = 3
# Roof fitting. The absolute limit is generous and the *relative* one does the real work: a
# blocky roof with dormers, a chimney and an overhang never fits a clean geometric model
# tightly, so "is this fit good" is the wrong question. "Does a sloped model explain this roof
# substantially better than a flat one" is the right one, and it separates cleanly -- real
# gables here beat flat by 2:1 at an absolute RMSE of ~1.3, while genuinely complex roofs
# (multi-building plots, hillsides, cross-gabled townhouses) sit at 2.5-4.0.
ROOF_RMSE_LIMIT = 2.0
ROOF_RMSE_VS_FLAT = 0.75
ROOF_FLAT_PARSIMONY = 0.35
ROOF_TRIM_FRACTION = 0.12
ROOF_MIN_SLOPE_COVERAGE = 0.25
LOW_CONFIDENCE_SHARE = 0.25


# ── small numeric helpers ────────────────────────────────────────────────────

def _summary(values) -> dict | None:
    """Percentile summary of a list of numbers, or None when there is nothing to summarise."""
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return None
    arr = np.array(sorted(clean))
    pct = lambda q: float(np.percentile(arr, q))  # noqa: E731
    return {"n": len(clean), "min": float(arr[0]), "p10": pct(10), "p25": pct(25),
            "med": pct(50), "p75": pct(75), "p90": pct(90), "max": float(arr[-1]),
            "mean": float(arr.mean())}


def _counter(values) -> dict:
    """Frequency table with the most common first, as a plain JSON-friendly dict."""
    counts = collections.Counter(v for v in values if v is not None)
    return dict(counts.most_common())


def _shift_max(labels, mask):
    """One relaxation step of max-label propagation across 4-connected neighbours."""
    out = labels
    for axis in (0, 1):
        rolled = np.roll(labels, 1, axis=axis)
        if axis == 0:
            rolled[0, :] = 0
        else:
            rolled[:, 0] = 0
        out = np.maximum(out, rolled)
        rolled = np.roll(labels, -1, axis=axis)
        if axis == 0:
            rolled[-1, :] = 0
        else:
            rolled[:, -1] = 0
        out = np.maximum(out, rolled)
    return np.where(mask, out, 0)


def _largest_component(mask):
    """Boolean mask of the largest 4-connected component of a 2-D mask.

    Max-label propagation rather than a flood fill: seed every cell with its own flat index
    and relax until the labels stop moving, so the whole thing is a handful of array shifts
    per iteration instead of a Python loop over cells. `thumbs._surface` uses the same
    pad-and-shift idiom. numpy only -- scipy is not a dependency of this repo.
    """
    if not mask.any():
        return mask, 1.0
    labels = np.where(mask, np.arange(1, mask.size + 1).reshape(mask.shape), 0)
    for _ in range(4 * max(mask.shape) + 8):
        nxt = _shift_max(labels, mask)
        if np.array_equal(nxt, labels):
            break
        labels = nxt
    values, counts = np.unique(labels[mask], return_counts=True)
    best = values[int(np.argmax(counts))]
    keep = labels == best
    return keep, float(keep.sum()) / float(mask.sum())


def _largest_rectangle(mask) -> int:
    """Area of the largest axis-aligned all-True rectangle (histogram sweep, O(n))."""
    if not mask.any():
        return 0
    best = 0
    heights = np.zeros(mask.shape[1], dtype=np.int64)
    for row in mask:
        heights = np.where(row, heights + 1, 0)
        stack: list[int] = []
        for i in range(len(heights) + 1):
            cur = heights[i] if i < len(heights) else 0
            while stack and heights[stack[-1]] >= cur:
                h = heights[stack.pop()]
                w = i - (stack[-1] + 1 if stack else 0)
                best = max(best, int(h * w))
            stack.append(i)
    return best


def _peaks(values, min_value: float, min_separation: int) -> list[int]:
    """Indices of local maxima above `min_value`, thinned to `min_separation` apart.

    Boundary indices count as peaks -- `106__pharmacy`'s only floor plate is at y=1, and a
    strict interior-peak test misses it entirely.
    """
    n = len(values)
    found = []
    for i in range(n):
        if values[i] < min_value:
            continue
        left = values[i - 1] if i > 0 else -1.0
        right = values[i + 1] if i < n - 1 else -1.0
        if values[i] >= left and values[i] >= right:
            found.append(i)
    kept: list[int] = []
    for i in found:
        if kept and i - kept[-1] < min_separation:
            if values[i] > values[kept[-1]]:
                kept[-1] = i
            continue
        kept.append(i)
    return kept


# ── segmentation ─────────────────────────────────────────────────────────────

class Segmentation:
    """The plot/building split for one build, plus why it looks the way it does."""

    __slots__ = ("structure", "terrain", "plant", "ground_top", "method", "flags",
                 "coverage_at_top", "air_gap", "taper_layers", "component_share",
                 "reject", "detail")

    def __init__(self):
        self.structure = None
        self.terrain = None
        self.plant = None
        self.ground_top = None
        self.method = "none"
        self.flags: list[str] = []
        self.coverage_at_top = 0.0
        self.air_gap = 0
        self.taper_layers = 0
        self.component_share = 1.0
        self.reject = None
        self.detail = ""


def _classify_palette(names):
    """Per-palette-index boolean tables, so classification is one lookup per voxel."""
    n = len(names)
    terrain = np.zeros(n, dtype=bool)
    plant = np.zeros(n, dtype=bool)
    air = np.zeros(n, dtype=bool)
    bases = []
    for i, raw in enumerate(names):
        base = block_roles.normalise(raw)
        bases.append(base)
        if not raw or ("minecraft:" + base) in schem_voxels.AIR_NAMES or base in ("air", ""):
            air[i] = True
        elif block_roles.is_terrain(base):
            terrain[i] = True
        elif block_roles.is_plant(base):
            plant[i] = True
    return bases, air, terrain, plant


def segment(voxels) -> Segmentation:
    """Split a build into ground plot and building, adaptively and with its quirks flagged."""
    seg = Segmentation()
    bases, air, terrain_idx, plant_idx = _classify_palette(voxels.names)

    grid = voxels.grid
    solid = ~air[grid]
    terrain = terrain_idx[grid] & solid
    plant = plant_idx[grid] & solid
    height, length, width = grid.shape
    area = width * length

    # Pad detection. Coverage is measured against the biggest terrain layer, because the
    # bounding box is routinely inflated by trees and outbuildings that have nothing to do
    # with the pad -- see the module docstring.
    pad_ground = terrain | plant
    tcount = pad_ground.reshape(height, -1).sum(axis=1)
    peak = int(tcount.max()) if height else 0
    coverage = tcount / max(peak, 1)

    ground_top = None
    if peak >= MIN_TERRAIN_FRACTION * area:
        starts = np.flatnonzero(coverage >= PAD_START_COVERAGE)
        if len(starts):
            y = int(starts[0])
            while y < height and coverage[y] >= PAD_CONTINUE_COVERAGE:
                ground_top = y
                y += 1
            seg.method = "terrain_pad"
            seg.coverage_at_top = float(coverage[ground_top])

    # A build may sit on a built platform with no terrain material at all: a full-coverage
    # layer of one non-terrain cube with open space above it is a plot slab, not a floor.
    if ground_top is None:
        for y in range(min(height, 4)):
            layer = solid[y]
            if layer.mean() < 0.95:
                continue
            vals, counts = np.unique(grid[y][layer], return_counts=True)
            top_share = counts.max() / counts.sum()
            base = bases[int(vals[int(np.argmax(counts))])]
            above = solid[y + 1].mean() if y + 1 < height else 0.0
            if top_share >= 0.90 and block_roles.is_cube(base) and above < 0.75:
                ground_top = y
                seg.method = "plot_slab"
                seg.flags.append("plot_slab")
                seg.coverage_at_top = float(top_share)
                break

    structure = solid & ~terrain & ~plant
    if ground_top is not None:
        structure[:ground_top + 1] = False

        # A stray full-coverage layer far below the real build is not the ground: check for a
        # gap and, if there is one, restart the search above it.
        occupied = np.flatnonzero(structure.reshape(height, -1).any(axis=1))
        if len(occupied):
            gap = int(occupied[0]) - ground_top - 1
            if gap > MAX_AIR_GAP:
                seg.flags.append("floating_gap")
                seg.air_gap = gap
                base_y = int(occupied[0])
                # The real pad sits directly *under* the building, which is below the first
                # structure layer -- not above it. Take the highest terrain layer beneath the
                # building and treat the stray layer far below as scenery.
                rescan = np.flatnonzero(coverage[:base_y] >= PAD_CONTINUE_COVERAGE)
                if len(rescan):
                    ground_top = int(rescan[-1])
                    structure = solid & ~terrain & ~plant
                    structure[:ground_top + 1] = False
                    seg.coverage_at_top = float(coverage[ground_top])

        if ground_top + 1 >= DEEP_TERRAIN_SHARE * height:
            seg.flags.append("deep_terrain")
        taper = 0
        for y in range(ground_top + 1, min(ground_top + 5, height)):
            if coverage[y] >= 0.15:
                taper += 1
        if taper >= 3:
            seg.flags.append("terrain_taper")
        seg.taper_layers = taper

    # Stray one- and two-voxel columns are fence posts, path lamps and single leaves that
    # survived the plant filter. They distort the footprint far more than their mass suggests.
    column_depth = structure.sum(axis=0)
    structure &= (column_depth >= 3)[None, :, :]

    seg.ground_top = ground_top
    seg.terrain = terrain
    seg.plant = plant

    total = int(structure.sum())
    if total < MIN_STRUCTURE_VOXELS:
        seg.reject = "too_small"
        seg.detail = "{} structure voxels (minimum {})".format(total, MIN_STRUCTURE_VOXELS)
        seg.structure = structure
        return seg

    # Keep only the largest building: detached sheds, garden walls and trees that made it
    # through the material filters are all separate components in the footprint.
    footprint = structure.any(axis=0)
    keep, share = _largest_component(footprint)
    seg.component_share = share
    if share < MIN_COMPONENT_SHARE:
        seg.reject = "multi_building"
        seg.detail = "largest component is {:.0%} of the structure footprint".format(share)
        seg.structure = structure
        return seg
    structure &= keep[None, :, :]
    seg.structure = structure
    return seg


# ── feature extraction ───────────────────────────────────────────────────────

def _fill_holes(mask):
    """`mask` with fully-enclosed interior cells filled in.

    Essential for reading a roof. A wall band is a *ring* -- for a 30x35 house that is ~130
    cells out of ~1050 -- so measuring "how much plan does this layer cover" against the raw
    mask compares a perimeter against an area and reports the walls as smaller than the roof.
    Filling the ring gives the storey's actual plan outline, which is what the roof sits on.
    """
    if not mask.any():
        return mask
    outside = np.zeros_like(mask)
    outside[0, :] = ~mask[0, :]
    outside[-1, :] = ~mask[-1, :]
    outside[:, 0] = ~mask[:, 0]
    outside[:, -1] = ~mask[:, -1]
    free = ~mask
    for _ in range(2 * sum(mask.shape) + 8):
        grown = outside.copy()
        grown[1:, :] |= outside[:-1, :]
        grown[:-1, :] |= outside[1:, :]
        grown[:, 1:] |= outside[:, :-1]
        grown[:, :-1] |= outside[:, 1:]
        grown &= free
        if np.array_equal(grown, outside):
            break
        outside = grown
    return mask | (free & ~outside)


def _height_map(sub):
    """Highest occupied y per column, -1 where the column is empty."""
    occupied = sub.any(axis=0)
    flipped = sub[::-1]
    top = sub.shape[0] - 1 - np.argmax(flipped, axis=0)
    return np.where(occupied, top, -1), occupied


def _fit_models(top, core, eave, ridge):
    """Fit flat/gable/hip/shed roof models; return each one's trimmed RMSE and parameters.

    Trimmed, because a chimney or a pair of dormers is a handful of columns with enormous
    residuals -- enough to push an obviously-gabled roof into "complex" if they are scored.
    """
    zs, xs = np.nonzero(core)
    observed = top[core].astype(np.float64)
    n = len(observed)
    keep = max(1, int(round(n * (1.0 - ROOF_TRIM_FRACTION))))

    def score(model):
        resid = np.abs(model - observed)
        resid.sort()
        trimmed = resid[:keep]
        return float(np.sqrt((trimmed ** 2).mean()))

    results = {}

    flat_c = float(np.median(observed))
    results["flat"] = {"rmse": score(np.full(n, flat_c)), "pitch": 0.0,
                       "axis": None, "coverage": 0.0}

    slopes = (0.5, 1.0, 1.5, 2.0)
    for axis, coords in (("x", xs.astype(np.float64)), ("z", zs.astype(np.float64))):
        lo, hi = coords.min(), coords.max()
        best = None
        centres = np.arange(lo, hi + 0.5, 0.5)
        for slope in slopes:
            for centre in centres:
                model = np.maximum(ridge - slope * np.abs(coords - centre), eave)
                rmse = score(model)
                if best is None or rmse < best["rmse"]:
                    coverage = float((model > eave + 1e-9).mean())
                    best = {"rmse": rmse, "pitch": slope, "axis": axis,
                            "centre": float(centre), "coverage": coverage}
        results["gable_" + axis] = best

    # Hips are essentially always centred, so search a small window around the centroid
    # rather than the full grid -- the full search is thousands of fits per build.
    cx0, cz0 = float(xs.mean()), float(zs.mean())
    best = None
    for slope in (0.5, 1.0, 1.5):
        for dx in np.arange(-4, 4.5, 1.0):
            for dz in np.arange(-4, 4.5, 1.0):
                model = np.maximum(
                    np.minimum(ridge - slope * np.abs(xs - (cx0 + dx)),
                               ridge - slope * np.abs(zs - (cz0 + dz))), eave)
                rmse = score(model)
                if best is None or rmse < best["rmse"]:
                    coverage = float((model > eave + 1e-9).mean())
                    best = {"rmse": rmse, "pitch": slope, "axis": None,
                            "coverage": coverage}
    results["hip"] = best

    best = None
    for slope in (0.5, 1.0):
        for axis, coords in (("x", xs.astype(np.float64)), ("z", zs.astype(np.float64))):
            for sign in (1.0, -1.0):
                model = np.clip(eave + sign * slope * (coords - coords.min()), eave, ridge)
                if sign < 0:
                    model = np.clip(eave + slope * (coords.max() - coords), eave, ridge)
                rmse = score(model)
                if best is None or rmse < best["rmse"]:
                    coverage = float((model > eave + 1e-9).mean())
                    best = {"rmse": rmse, "pitch": slope, "axis": axis,
                            "coverage": coverage}
    results["shed"] = best
    return results


def _analyse_roof(sub, floor_peaks):
    """Classify the roof over the *main* mass, ignoring porches, garages and chimneys."""
    top, occupied = _height_map(sub)
    area = int(occupied.sum())
    hh = sub.shape[0]

    # Measure plan area on the *floor plates*, not on every layer. Hole-filling a wall level
    # leaks out through the windows and the doorway, so a mid-storey layer reports a fraction
    # of the plan it actually encloses -- `100__georgian` swings between 743 and 186 from one
    # layer to the next. The plates are solid, so their filled area is the real storey plan.
    plates = [(p, _fill_holes(sub[p])) for p in (floor_peaks or [0])]
    a_plan = max(int(m.sum()) for _, m in plates) or 1

    # Eave = just above the highest plate that still carries a storey's worth of plan. The
    # topmost plate is often the roof deck, which is markedly smaller and drops out here.
    # Measuring an area drop rather than a height percentile matters: a percentile is dragged
    # down by porch rails, fences and garden walls and lands below the first storey's ceiling.
    storey_plates = [(p, m) for p, m in plates if int(m.sum()) >= 0.60 * a_plan]
    if not storey_plates:
        storey_plates = plates
    top_plate, top_mask = storey_plates[-1]

    # The roof covers the top storey's plan, not every column in the footprint -- a garage or
    # porch below would otherwise pull the fit apart.
    core = top_mask & occupied
    if core.sum() < 0.15 * area:
        core = occupied
    if not core.any():
        return {"type": "unknown", "rmse": None}

    core_tops = top[core]

    # Eave = the roof's lower edge, taken as a low percentile of the height map *over the
    # core*. Restricting to the core is what makes a percentile safe here: over the full
    # footprint it is dragged down by porch rails, fences and garden walls. The alternative --
    # the first layer whose filled plan shrinks -- does not survive contact with real builds,
    # because hole-filling leaks out through the windows and the doorway, so a mid-storey
    # layer reports a fraction of the plan it actually encloses.
    eave = int(round(float(np.percentile(core_tops, 10))))
    eave = max(top_plate + 1, min(eave, hh - 1))
    ridge = float(np.percentile(core_tops, 98))
    rise = ridge - eave

    fits = _fit_models(top, core, float(eave), ridge)
    ranked = sorted(fits.items(), key=lambda kv: kv[1]["rmse"])
    best_name, best = ranked[0]
    flat_rmse = fits["flat"]["rmse"]

    explains = (best["rmse"] <= ROOF_RMSE_LIMIT
                and best["rmse"] <= ROOF_RMSE_VS_FLAT * flat_rmse
                and best["coverage"] >= ROOF_MIN_SLOPE_COVERAGE)
    if rise <= 1:
        kind, chosen = "flat", fits["flat"]
    elif flat_rmse <= best["rmse"] + ROOF_FLAT_PARSIMONY:
        kind, chosen = "flat", fits["flat"]
    elif best_name.startswith("gable") and explains:
        kind, chosen = "gable", best
    elif best_name in ("hip", "shed") and explains:
        kind, chosen = best_name, best
    else:
        kind, chosen = "complex", best

    # Columns the trim threw away, clustered, are dormers and chimneys.
    model_free = np.abs(core_tops - np.median(core_tops))
    outliers = int((model_free > max(2.0, rise * 0.6)).sum())

    overhang = int((occupied & ~core).sum())
    return {"type": kind, "best_fit": best_name, "rmse": round(float(best["rmse"]), 3),
            "flat_rmse": round(float(flat_rmse), 3),
            "pitch": chosen.get("pitch"), "axis": chosen.get("axis"),
            "coverage": round(float(chosen.get("coverage", 0.0)), 3),
            "eave_y": int(eave), "ridge_y": int(round(ridge)), "rise": int(round(rise)),
            "core_area": int(core.sum()), "overhang": overhang, "outlier_columns": outliers}


def _analyse_storeys(sub, area):
    """Floor plates show as local maxima of horizontal fill; open rooms are the troughs."""
    fill = sub.reshape(sub.shape[0], -1).sum(axis=1) / max(area, 1)
    median = float(np.median(fill[fill > 0])) if (fill > 0).any() else 0.0
    threshold = max(FLOOR_MIN_FILL, 1.5 * median)
    peaks = _peaks(list(fill), threshold, FLOOR_MIN_SEPARATION)

    # A mezzanine or a stair landing makes a peak that never returns to room-level fill in
    # between. Require a real trough between consecutive floors.
    kept: list[int] = []
    for p in peaks:
        if not kept:
            kept.append(p)
            continue
        between = fill[kept[-1] + 1:p]
        if len(between) and between.min() <= 0.70 * min(fill[kept[-1]], fill[p]):
            kept.append(p)
    return kept, [round(float(f), 3) for f in fill]


def _ring(footprint):
    """Perimeter columns: the footprint minus its 4-neighbour erosion."""
    padded = np.pad(footprint, 1, constant_values=False)
    eroded = (padded[:-2, 1:-1] & padded[2:, 1:-1] &
              padded[1:-1, :-2] & padded[1:-1, 2:] & footprint)
    return footprint & ~eroded


def _dominant(grid, mask, bases, predicate):
    """Most common base block inside `mask` that satisfies `predicate`, with its share."""
    if not mask.any():
        return None
    values, counts = np.unique(grid[mask], return_counts=True)
    tally: collections.Counter = collections.Counter()
    for value, count in zip(values, counts):
        base = bases[int(value)]
        if base and predicate(base):
            tally[base] += int(count)
    if not tally:
        return None
    total = sum(tally.values())
    ranked = tally.most_common(3)
    name, count = ranked[0]
    return {"name": "minecraft:" + name, "share": round(count / total, 3),
            "count": int(count),
            "runners_up": [["minecraft:" + n, round(c / total, 3)] for n, c in ranked[1:]]}


def _low_confidence(pick):
    """Report a weak signal as absent rather than as a confident wrong answer."""
    if pick is None:
        return None
    if pick["share"] < LOW_CONFIDENCE_SHARE:
        pick = dict(pick)
        pick["low_confidence"] = True
    return pick


def _analyse_materials(sub_grid, sub, bases, footprint, floor_peaks, eave):
    """Assign wall/roof/trim/foundation by *where* blocks sit, not by how common they are."""
    ring = _ring(footprint)
    hh = sub.shape[0]
    base_y = floor_peaks[0] if floor_peaks else 0
    wall_hi = max(base_y + 1, min(eave, hh))

    wall_band = np.zeros_like(sub)
    wall_band[base_y + 1:wall_hi] = sub[base_y + 1:wall_hi] & ring[None, :, :]

    solid_cube = lambda b: block_roles.is_cube(b) and not block_roles.is_glass(b)  # noqa: E731

    wall = _dominant(sub_grid, wall_band, bases, solid_cube)
    trim = None
    if wall:
        wall_base = wall["name"].split(":", 1)[1]
        wall_family = block_roles.family(wall_base)
        trim = _dominant(sub_grid, wall_band, bases,
                         lambda b: (solid_cube(b) and b != wall_base
                                    and block_roles.family(b) != wall_family))
        if trim and trim["share"] < 0.08:
            trim = None

    roof_band = np.zeros_like(sub)
    roof_band[min(eave, hh - 1):] = sub[min(eave, hh - 1):]
    roof_stairs = _dominant(sub_grid, roof_band, bases, lambda b: b.endswith("_stairs"))
    roof_slab = _dominant(sub_grid, roof_band, bases, lambda b: b.endswith("_slab"))
    roof_solid = _dominant(sub_grid, roof_band, bases, solid_cube)

    foundation_band = np.zeros_like(sub)
    foundation_band[base_y:base_y + 2] = sub[base_y:base_y + 2] & ring[None, :, :]
    foundation = _dominant(sub_grid, foundation_band, bases, solid_cube)

    glass = _dominant(sub_grid, sub, bases, block_roles.is_glass)

    floor = None
    if floor_peaks:
        plate = np.zeros_like(sub)
        plate[floor_peaks[0]] = sub[floor_peaks[0]]
        floor = _dominant(sub_grid, plate, bases, solid_cube)

    return {"wall": _low_confidence(wall), "trim": _low_confidence(trim),
            "roof_stairs": _low_confidence(roof_stairs),
            "roof_slab": _low_confidence(roof_slab),
            "roof_solid": _low_confidence(roof_solid),
            "foundation": _low_confidence(foundation),
            "floor": _low_confidence(floor), "glass": _low_confidence(glass)}


def _face_components(hole):
    """4-connected components of a small 2-D boolean mask, as lists of (row, col)."""
    seen = np.zeros_like(hole, dtype=bool)
    out = []
    rows, cols = hole.shape
    for r in range(rows):
        for c in range(cols):
            if not hole[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and hole[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(cells)
    return out


def _analyse_face(sub, sub_grid, bases, face, base_y, eave):
    """Count and measure the openings in one wall face.

    An opening is a hole in the *outermost* wall plane -- either missing blocks or glass. The
    outer plane has to be found per column, because a house with a porch or a bay is not a
    flat rectangle and comparing against the bounding box would read the setback as a window.
    """
    hh, ll, ww = sub.shape
    lo, hi = base_y + 1, min(eave, hh)
    if hi - lo < 2:
        return None

    if face in ("north", "south"):
        span = ww
        pick = range(ll) if face == "north" else range(ll - 1, -1, -1)
    else:
        span = ll
        pick = range(ww) if face == "west" else range(ww - 1, -1, -1)

    solid_face = np.zeros((hi - lo, span), dtype=bool)
    glass_face = np.zeros_like(solid_face)
    door_face = np.zeros_like(solid_face)
    present = np.zeros(span, dtype=bool)

    for i in range(span):
        outer = None
        for j in pick:
            column = sub[lo:hi, j, i] if face in ("north", "south") else sub[lo:hi, i, j]
            if column.any():
                outer = j
                break
        if outer is None:
            continue
        present[i] = True
        if face in ("north", "south"):
            column = sub[lo:hi, outer, i]
            values = sub_grid[lo:hi, outer, i]
        else:
            column = sub[lo:hi, i, outer]
            values = sub_grid[lo:hi, i, outer]
        solid_face[:, i] = column
        for k in range(hi - lo):
            base = bases[int(values[k])]
            if not column[k]:
                continue
            if block_roles.is_glass(base):
                glass_face[k, i] = True
            if base.endswith("_door"):
                door_face[k, i] = True

    if present.sum() < 3:
        return None

    hole = (~solid_face | glass_face) & present[None, :]
    windows, doors, large = [], [], 0
    for cells in _face_components(hole):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        w = max(cols) - min(cols) + 1
        h = max(rows) - min(rows) + 1
        if min(cols) == 0 or max(cols) == span - 1:
            continue  # runs off the end of the projection: not an opening
        has_door = any(door_face[r, c] for r, c in cells)
        # A door reaches the floor itself. Allowing one block of slack here turns every
        # low-silled window into a door, which is the difference between "this wall has four
        # windows" and "this wall has four doors".
        touches_floor = min(rows) == 0
        if has_door or (touches_floor and h >= 2 and w <= 3):
            doors.append({"w": w, "h": h, "centre": (min(cols) + max(cols)) / 2.0})
        elif len(cells) >= 2 and w <= 6 and h <= 5:
            windows.append({"w": w, "h": h, "sill": min(rows),
                            "centre": (min(cols) + max(cols)) / 2.0})
        else:
            large += 1

    centres = sorted(win["centre"] for win in windows)
    spacing = [round(b - a, 1) for a, b in zip(centres, centres[1:])]
    band_area = int(present.sum()) * (hi - lo)
    return {
        "windows": len(windows), "doors": len(doors), "large_openings": large,
        "window_w": _summary([w["w"] for w in windows]),
        "window_h": _summary([w["h"] for w in windows]),
        "sill": _summary([w["sill"] for w in windows]),
        "spacing": _summary(spacing),
        "glazing": round(sum(w["w"] * w["h"] for w in windows) / max(band_area, 1), 3),
        "door_centres": [round(d["centre"] / max(span - 1, 1), 3) for d in doors],
        "wall_span": int(present.sum()),
    }


# ── per-build entry point ────────────────────────────────────────────────────

def analyse(path, build_id: str) -> dict:
    """Full anatomy of one stored build, or a row explaining why it could not be read."""
    row = {"id": build_id, "ok": False, "reject": None, "detail": "", "flags": []}

    voxels = schem_voxels.load_voxels(path)
    if voxels is None:
        row["reject"] = "unreadable"
        row["detail"] = "load_voxels returned None"
        return row

    row["source_dims"] = {"w": voxels.width, "h": voxels.height, "l": voxels.length}
    row["palette"] = {"size": len(voxels.names)}

    bases, air, _, _ = _classify_palette(voxels.names)
    unknown = sorted({b for b, is_air in zip(bases, air)
                      if not is_air and b in block_roles.LEGACY_ALIASES.values()
                      and b in ("oak_planks", "stone_bricks", "terracotta")} & set())
    row["palette"]["distinct_bases"] = len({b for b, a in zip(bases, air) if not a})
    row["palette"]["legacy_names"] = sorted(
        {n.split(":", 1)[-1] for n in voxels.names
         if n and n.split(":", 1)[-1].split("[")[0] in block_roles.LEGACY_ALIASES})

    seg = segment(voxels)
    row["flags"] = list(seg.flags)
    row["ground"] = {"top": seg.ground_top, "method": seg.method,
                     "coverage_at_top": round(seg.coverage_at_top, 3),
                     "air_gap": seg.air_gap, "taper_layers": seg.taper_layers,
                     "component_share": round(seg.component_share, 3)}
    if seg.reject:
        row["reject"] = seg.reject
        row["detail"] = seg.detail
        return row

    structure = seg.structure
    ys, zs, xs = np.nonzero(structure)
    y0, y1 = int(ys.min()), int(ys.max())
    z0, z1 = int(zs.min()), int(zs.max())
    x0, x1 = int(xs.min()), int(xs.max())
    sub = structure[y0:y1 + 1, z0:z1 + 1, x0:x1 + 1]
    sub_grid = voxels.grid[y0:y1 + 1, z0:z1 + 1, x0:x1 + 1]
    hh, ll, ww = sub.shape

    if hh > MAX_TOWER_RATIO * min(ww, ll):
        row["reject"] = "tower"
        row["detail"] = "height {} vs footprint {}x{}".format(hh, ww, ll)
        return row

    footprint = sub.any(axis=0)
    area = int(footprint.sum())
    rect = _largest_rectangle(footprint)
    row["footprint"] = {
        "w": ww, "d": ll, "h": hh, "area": area,
        "aspect": round(max(ww, ll) / max(min(ww, ll), 1), 3),
        "fill_ratio": round(area / max(ww * ll, 1), 3),
        "rectangularity": round(rect / max(area, 1), 3),
        "solid": int(sub.sum()),
        "density": round(int(sub.sum()) / max(ww * ll * hh, 1), 3),
    }

    floor_peaks, fill_profile = _analyse_storeys(sub, area)
    if not floor_peaks:
        row["reject"] = "no_floor_peaks"
        row["detail"] = "no horizontal fill maxima above threshold"
        return row

    roof = _analyse_roof(sub, floor_peaks)
    row["roof"] = roof
    eave = roof.get("eave_y", hh - 1)

    # The topmost plate is normally the roof deck rather than a storey floor.
    storey_floors = [p for p in floor_peaks if p < eave - 1] or floor_peaks[:1]
    heights = [b - a for a, b in zip(storey_floors, storey_floors[1:])]
    row["storeys"] = {
        "count": len(storey_floors), "floor_ys": storey_floors, "heights": heights,
        "median_height": round(float(np.median(heights)), 1) if heights else None,
        "all_peaks": floor_peaks,
    }
    row["fill_profile"] = fill_profile

    row["materials"] = _analyse_materials(sub_grid, sub, bases, footprint,
                                          storey_floors, eave)
    row["openings"] = {face: _analyse_face(sub, sub_grid, bases, face,
                                           storey_floors[0], eave) for face in FACES}

    fingerprint = hashlib.sha256(
        "{}x{}x{}|{}".format(ww, hh, ll, row["footprint"]["solid"]).encode()
    ).hexdigest()[:16]
    row["fingerprint"] = fingerprint
    row["ok"] = True
    return row


# ── aggregation ──────────────────────────────────────────────────────────────

def _style_tuple(row):
    materials = row.get("materials") or {}
    pick = lambda key: (materials.get(key) or {}).get("name")  # noqa: E731
    return pick("wall"), pick("roof_stairs"), pick("trim"), pick("foundation")


def aggregate(rows) -> dict:
    """Roll per-build rows up into the distributions the generator will sample from."""
    ok = [r for r in rows if r.get("ok")]

    def field(path, source=ok):
        out = []
        for row in source:
            cur = row
            for key in path:
                cur = (cur or {}).get(key) if isinstance(cur, dict) else None
            out.append(cur)
        return out

    walls, roofs, trims, foundations = [], [], [], []
    wall_to_roof: dict = collections.defaultdict(collections.Counter)
    wall_to_trim: dict = collections.defaultdict(collections.Counter)
    wall_to_foundation: dict = collections.defaultdict(collections.Counter)
    triples: collections.Counter = collections.Counter()
    styles = []
    for row in ok:
        wall, roof, trim, foundation = _style_tuple(row)
        walls.append(wall)
        roofs.append(roof)
        trims.append(trim)
        foundations.append(foundation)
        styles.append({"id": row["id"], "wall": wall, "roof_stairs": roof,
                       "trim": trim, "foundation": foundation,
                       "roof_type": (row.get("roof") or {}).get("type"),
                       "storeys": (row.get("storeys") or {}).get("count")})
        if wall:
            if roof:
                wall_to_roof[wall][roof] += 1
            if trim:
                wall_to_trim[wall][trim] += 1
            if foundation:
                wall_to_foundation[wall][foundation] += 1
            triples[(wall, roof, trim)] += 1

    windows, sills, spacings, window_w, window_h, glazing = [], [], [], [], [], []
    for row in ok:
        for face in FACES:
            info = (row.get("openings") or {}).get(face)
            if not info:
                continue
            windows.append(info["windows"])
            glazing.append(info["glazing"])
            for key, sink in (("sill", sills), ("spacing", spacings),
                              ("window_w", window_w), ("window_h", window_h)):
                stat = info.get(key)
                if stat:
                    sink.append(stat["med"])

    fingerprints: dict = collections.defaultdict(list)
    for row in ok:
        fingerprints[row["fingerprint"]].append(row["id"])

    roof_types = field(["roof", "type"])
    return {
        "footprint_w": _summary(field(["footprint", "w"])),
        "footprint_d": _summary(field(["footprint", "d"])),
        "footprint_h": _summary(field(["footprint", "h"])),
        "aspect": _summary(field(["footprint", "aspect"])),
        "fill_ratio": _summary(field(["footprint", "fill_ratio"])),
        "rectangularity": _summary(field(["footprint", "rectangularity"])),
        "density": _summary(field(["footprint", "density"])),
        "storey_count": _counter(field(["storeys", "count"])),
        "storey_height": _counter(
            h for row in ok for h in ((row.get("storeys") or {}).get("heights") or [])),
        "roof_type": _counter(roof_types),
        "roof_pitch": _counter(field(["roof", "pitch"])),
        "roof_rise": _summary(field(["roof", "rise"])),
        "roof_rmse": _summary(field(["roof", "rmse"])),
        "windows_per_face": _summary(windows),
        "window_w": _summary(window_w),
        "window_h": _summary(window_h),
        "sill": _summary(sills),
        "spacing": _summary(spacings),
        "glazing": _summary(glazing),
        "materials": {
            "wall": _counter(walls), "roof_stairs": _counter(roofs),
            "trim": _counter(trims), "foundation": _counter(foundations),
        },
        "cooccurrence": {
            "wall_to_roof": {k: dict(v.most_common()) for k, v in wall_to_roof.items()},
            "wall_to_trim": {k: dict(v.most_common()) for k, v in wall_to_trim.items()},
            "wall_to_foundation": {k: dict(v.most_common())
                                   for k, v in wall_to_foundation.items()},
            "triples": [[list(k), v] for k, v in triples.most_common()],
        },
        "styles": styles,
        "duplicate_groups": [ids for ids in fingerprints.values() if len(ids) > 1],
    }


def analyse_library(root=BUILDS_DIR, only: str | None = None) -> dict:
    """Analyse every stored build and assemble the full report."""
    paths = sorted(Path(root).glob("*/schem/original.schem"))
    if only:
        paths = [p for p in paths if only in p.parts[-3]]
    rows = [analyse(p, p.parts[-3]) for p in paths]
    ok = [r for r in rows if r.get("ok")]
    rejected = [r for r in rows if not r.get("ok")]
    agg = aggregate(rows)
    return {
        "schema": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "counts": {"scanned": len(rows), "analysed": len(ok), "rejected": len(rejected),
                   "duplicate_groups": len(agg["duplicate_groups"])},
        "builds": rows,
        "aggregate": agg,
    }


# ── report printing ──────────────────────────────────────────────────────────

def _bar(counter: dict, width: int = 40) -> list[str]:
    if not counter:
        return ["    (none)"]
    top = max(counter.values())
    return ["    {:<34} {:>4}  {}".format(str(k), v, "#" * max(1, round(width * v / top)))
            for k, v in counter.items()]


def _fmt_summary(stat) -> str:
    if not stat:
        return "n=0"
    return ("n={n:<4} min {min:>5.1f}  p25 {p25:>5.1f}  med {med:>5.1f}  "
            "p75 {p75:>5.1f}  max {max:>5.1f}").format(**stat)


def print_report(report: dict) -> None:
    counts = report["counts"]
    agg = report["aggregate"]
    print()
    print("BUILD LIBRARY ANATOMY  --  {scanned} scanned, {analysed} analysed, "
          "{rejected} rejected, {duplicate_groups} duplicate group(s)".format(**counts))

    print("\n-- Per build " + "-" * 92)
    print("  {:<34} {:>11} {:>3} {:>4}  {:<9} {:>5}  {:<22} {:<22} {:>4}"
          .format("id", "w x d x h", "st", "hgt", "roof", "pitch", "wall", "roof-stairs", "win"))
    for row in report["builds"]:
        if not row.get("ok"):
            continue
        fp, st, roof, mats = row["footprint"], row["storeys"], row["roof"], row["materials"]
        wall = (mats.get("wall") or {}).get("name", "-").replace("minecraft:", "")
        stairs = (mats.get("roof_stairs") or {}).get("name", "-").replace("minecraft:", "")
        wins = sum((row["openings"].get(f) or {}).get("windows", 0) for f in FACES)
        flags = " ".join(row["flags"])
        print("  {:<34} {:>3}x{:>3}x{:>3} {:>3} {:>4}  {:<9} {:>5}  {:<22} {:<22} {:>4} {}"
              .format(row["id"][:34], fp["w"], fp["d"], fp["h"], st["count"],
                      st["median_height"] or 0, roof["type"], roof["pitch"] or 0,
                      wall[:22], stairs[:22], wins, flags))

    rejected = [r for r in report["builds"] if not r.get("ok")]
    if rejected:
        print("\n-- Rejected " + "-" * 93)
        for row in rejected:
            print("  {:<34} {:<18} {}".format(row["id"][:34], row["reject"], row["detail"]))

    print("\n-- Distributions " + "-" * 88)
    for label, key in (("footprint width ", "footprint_w"), ("footprint depth ", "footprint_d"),
                       ("footprint height", "footprint_h"), ("aspect ratio    ", "aspect"),
                       ("fill ratio      ", "fill_ratio"), ("rectangularity  ", "rectangularity"),
                       ("windows / face  ", "windows_per_face"), ("window width    ", "window_w"),
                       ("window height   ", "window_h"), ("sill above floor", "sill"),
                       ("window spacing  ", "spacing"), ("roof rise       ", "roof_rise"),
                       ("roof fit rmse   ", "roof_rmse")):
        print("  {}  {}".format(label, _fmt_summary(agg[key])))

    print("\n  storey count"); print("\n".join(_bar(agg["storey_count"])))
    print("\n  storey height"); print("\n".join(_bar(agg["storey_height"])))
    print("\n  roof type"); print("\n".join(_bar(agg["roof_type"])))
    print("\n  roof pitch"); print("\n".join(_bar(agg["roof_pitch"])))

    print("\n-- Materials (by build count) " + "-" * 75)
    for role in ("wall", "roof_stairs", "trim", "foundation"):
        table = agg["materials"][role]
        top = list(table.items())[:8]
        print("  {:<13} {}".format(role, "  ".join(
            "{} {}".format(str(k).replace("minecraft:", ""), v) for k, v in top) or "(none)"))

    print("\n-- Co-occurrence: wall -> roof-stairs " + "-" * 67)
    for wall, roofs in sorted(agg["cooccurrence"]["wall_to_roof"].items(),
                              key=lambda kv: -sum(kv[1].values()))[:12]:
        print("  {:<26} -> {}".format(
            wall.replace("minecraft:", "")[:26],
            ", ".join("{} {}".format(k.replace("minecraft:", ""), v)
                      for k, v in list(roofs.items())[:4])))

    legacy = collections.Counter()
    for row in report["builds"]:
        for name in (row.get("palette") or {}).get("legacy_names", []):
            legacy[name] += 1
    print("\n-- Vocabulary hygiene " + "-" * 83)
    print("  legacy / pre-flattening names: " +
          (", ".join("{}({})".format(k, v) for k, v in legacy.most_common()) or "(none)"))
    dupes = agg["duplicate_groups"]
    print("  duplicate structure groups:    " +
          (" | ".join(", ".join(g) for g in dupes) if dupes else "(none)"))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the JSON report")
    parser.add_argument("--only", default=None, help="only analyse build ids containing this")
    parser.add_argument("--quiet", action="store_true", help="write the JSON, skip the summary")
    args = parser.parse_args()

    report = analyse_library(only=args.only)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not args.quiet:
        print_report(report)
    print("wrote {}  ({:,} bytes)".format(out, out.stat().st_size))


if __name__ == "__main__":
    main()
