#!/usr/bin/env python3
"""How a build's top surface is made: which block caps each column, at each roof slope.

Usage:  python -m synth.roof_stats real               # the real library (orientation-safe builds)
        python -m synth.roof_stats run <run dir>      # a synthetic run: geometric vs builder roofs

This is the measuring stick for roof construction. For every column of a build the topmost
structural block is its CAP (full block, bottom/top slab, bottom/top-half stair, glass,
other). The local slope comes from the neighbouring caps' surface heights, so a 45-degree
roof reads 1.0, a half-pitch roof 0.5, a flat roof 0. Tabulating cap kind by slope class
shows how a roof is BUILT, independent of what it is a roof of:

    real builds       -- how Minecraft builders do it (the style target)
    geometric targets -- what a straight conversion of the geometry gives (Approach A)
    builder targets   -- what the builder-style roof pass gives (Approach B)

Stairs are also checked for orientation: a stair on a slope should face up-slope (Minecraft's
`facing` names the high side).

Only columns whose cap stands at least `MIN_CAP_HEIGHT` above the build's ground count, so
fences, paths and garden walls do not dilute the roofs. Real builds whose block states were
lost on import (every stair in its default state) are skipped: their caps cannot be read.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

KINDS = ("full", "slab_bottom", "slab_top", "stairs", "stairs_top", "glass", "other")
SLOPE_CLASSES = (("flat", 0.0, 0.25), ("low", 0.25, 0.75), ("pitched", 0.75, 1.5), ("steep", 1.5, 99.0))
MIN_CAP_HEIGHT = 3
_SURFACE = {"full": 1.0, "slab_bottom": 0.5, "slab_top": 1.0, "stairs": 0.75, "stairs_top": 1.0,
            "glass": 1.0, "other": 1.0}
_FACING = {"east": (1, 0), "west": (-1, 0), "south": (0, 1), "north": (0, -1)}   # (dx, dz)


def cap_kind(state: str) -> tuple[str, str | None]:
    """(kind, facing) for a blockstate string."""
    import block_roles

    name = state.split("[", 1)[0]
    base = block_roles.normalise(name)
    props = dict(re.findall(r"(\w+)=(\w+)", state))
    if "glass" in name and "pane" not in name:
        return "glass", None
    if name.endswith("_stairs"):
        return ("stairs_top" if props.get("half") == "top" else "stairs"), props.get("facing")
    if name.endswith("_slab"):
        t = props.get("type", "bottom")
        return ("full" if t == "double" else "slab_" + t), None
    if block_roles.is_cube(base):
        return "full", None
    return "other", None


def column_caps(grid_yzx, names, structure=None, ground_top: int = 0):
    """Per (z, x) column: cap kind index, facing, surface height; -1 kind where no cap.

    `structure` (Y,Z,X bool) limits caps to structural blocks; by default every non-air voxel.
    """
    grid = np.asarray(grid_yzx)
    air = np.array([n.split("[", 1)[0] in ("minecraft:air", "minecraft:cave_air", "minecraft:void_air")
                    for n in names])
    solid = ~air[grid] if structure is None else np.asarray(structure, dtype=bool)
    Y, Z, X = grid.shape
    has = solid.any(axis=0)
    top = np.where(has, Y - 1 - np.argmax(solid[::-1], axis=0), -1)
    kind_of = [cap_kind(n) for n in names]
    kind = np.full((Z, X), -1, dtype=np.int8)
    facing = np.full((Z, X), "", dtype=object)
    surf = np.full((Z, X), np.nan)
    zs, xs = np.nonzero(has & (top >= ground_top + MIN_CAP_HEIGHT))
    for z, x in zip(zs.tolist(), xs.tolist()):
        y = int(top[z, x])
        k, f = kind_of[int(grid[y, z, x])]
        kind[z, x] = KINDS.index(k)
        facing[z, x] = f or ""
        surf[z, x] = y + _SURFACE[k]
    return kind, facing, surf


def slopes(surf, max_step: float = 2.5, max_disagree: float = 0.6):
    """Gradient (gx, gz) of the cap surface, NaN where there is no consistent slope.

    Along each axis the step to the next column (forward) and from the previous one
    (backward) are averaged. A step larger than `max_step` is an edge (a wall, the drop at
    an eave), not roof, and is left out. Where both steps exist they must agree within
    `max_disagree`: a roof plane climbs evenly, while a single ledge (0 then 1), a parapet
    or a crenellation (+1 then -1) does not, and those columns get no slope at all.
    """
    import warnings

    def diff(axis):
        fwd = np.full(surf.shape, np.nan)
        bwd = np.full(surf.shape, np.nan)
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        d = surf[tuple(hi)] - surf[tuple(lo)]
        d[np.abs(d) > max_step] = np.nan
        fwd[tuple(lo)] = d
        bwd[tuple(hi)] = d
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN columns stay NaN
            g = np.nanmean(np.stack([fwd, bwd]), axis=0)
        uneven = np.abs(fwd - bwd) > max_disagree                # NaN compares False
        g[uneven] = np.inf
        return g

    gx, gz = diff(1), diff(0)                                   # (gx along X, gz along Z)
    bad = np.isinf(gx) | np.isinf(gz)
    gx[bad] = np.nan
    gz[bad] = np.nan
    return gx, gz


def tabulate(grid_yzx, names, structure=None, ground_top: int = 0, stats=None):
    """Add one build's caps to `stats` (a dict of Counters) and return it."""
    stats = stats if stats is not None else {"caps": {c[0]: Counter() for c in SLOPE_CLASSES},
                                             "stair_facing": Counter(), "columns": 0}
    kind, facing, surf = column_caps(grid_yzx, names, structure, ground_top)
    gx, gz = slopes(surf)
    g = np.hypot(np.nan_to_num(gx), np.nan_to_num(gz))
    known = (kind >= 0) & ~(np.isnan(gx) & np.isnan(gz))
    for name, lo, hi in SLOPE_CLASSES:
        sel = known & (g >= lo) & (g < hi)
        for k, n in zip(*np.unique(kind[sel], return_counts=True)):
            stats["caps"][name][KINDS[int(k)]] += int(n)
    stats["columns"] += int(known.sum())
    # Stairs on a real slope: do they face up-slope?
    st = known & (kind == KINDS.index("stairs")) & (g >= 0.25)
    for z, x in zip(*np.nonzero(st)):
        f = facing[z, x]
        if f not in _FACING:
            stats["stair_facing"]["no_state"] += 1
            continue
        dx, dz = _FACING[f]
        up = np.nan_to_num(gx[z, x]) * dx + np.nan_to_num(gz[z, x]) * dz
        stats["stair_facing"]["up_slope" if up > 0 else ("across" if up == 0 else "down_slope")] += 1
    return stats


def summarise(stats):
    out = {"columns": stats["columns"], "by_slope": {}, "stair_facing": dict(stats["stair_facing"])}
    for name, c in stats["caps"].items():
        n = sum(c.values())
        out["by_slope"][name] = {"columns": n, **{k: round(c[k] / n, 3) if n else 0.0 for k in KINDS}}
    return out


def format_table(title, summary):
    lines = ["{} ({:,} roof columns)".format(title, summary["columns"]),
             "  {:8s} {:>8s}  ".format("slope", "columns") + " ".join("{:>11s}".format(k) for k in KINDS)]
    for name, row in summary["by_slope"].items():
        lines.append("  {:8s} {:8,d}  ".format(name, row["columns"]) +
                     " ".join("{:>11.1%}".format(row[k]) for k in KINDS))
    sf = summary["stair_facing"]
    tot = sum(sf.values())
    if tot:
        lines.append("  stairs on slopes facing up-slope: {:.1%} of {:,} (down {:.1%}, across {:.1%})".format(
            sf.get("up_slope", 0) / tot, tot, sf.get("down_slope", 0) / tot, sf.get("across", 0) / tot))
    return "\n".join(lines)


# ── sources ───────────────────────────────────────────────────────────────────

def real_builds(root: Path = Path("data/builds"), exclude=frozenset()):
    """(build dir, VoxelGrid, Segmentation) for real builds whose block states survived import.

    `exclude` holds build ids to leave out (e.g. the held-out evaluation set).
    """
    import house_stats
    import schem_voxels

    for d in sorted(root.iterdir(), key=lambda p: int(p.name.split("__")[0])):
        if int(d.name.split("__")[0]) in exclude:
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if "synthetic" in d.name or "synthetic" in (meta.get("source_note") or "").lower():
            continue                                    # house_grammar output, not a real build
        sp = d / "schem" / "original.schem"
        vg = schem_voxels.load_voxels(sp) if sp.exists() else None
        if vg is None:
            continue
        if stateless_stairs(vg):
            continue
        seg = house_stats.segment(vg)
        if seg.structure is None or not seg.structure.any():
            continue
        yield d, vg, seg


def stateless_stairs(vg) -> bool:
    """True when the build's stairs carry no block state (lost on import)."""
    counts = np.bincount(vg.grid.ravel(), minlength=len(vg.names))
    stairs = [i for i, n in enumerate(vg.names) if "_stairs" in n and counts[i]]
    return bool(stairs) and not any("[" in vg.names[i] for i in stairs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    real = sub.add_parser("real")
    real.add_argument("--exclude-eval", action="store_true",
                      help="leave out the held-out evaluation builds (eval/real-eval-v1.json)")
    r = sub.add_parser("run")
    r.add_argument("root")
    ap.add_argument("--json", help="also write the summary here")
    args = ap.parse_args(argv)
    result = {}
    if args.cmd == "real":
        exclude = frozenset()
        if args.exclude_eval:
            from .real_eval import eval_build_ids
            exclude = frozenset(eval_build_ids())
        stats, n = None, 0
        for _d, vg, seg in real_builds(exclude=exclude):
            stats = tabulate(vg.grid, vg.names, seg.structure, int(seg.ground_top or 0), stats)
            n += 1
        result["real"] = summarise(stats)
        print(format_table("real builds with block states ({} builds)".format(n), result["real"]))
    else:
        import schem_voxels
        root = Path(args.root)
        for style, fname in (("geometric", "target_geometric.schem"), ("builder", "target.schem")):
            stats, n = None, 0
            for sp in sorted(root.glob("samples/*/" + fname)):
                vg = schem_voxels.load_voxels(sp)
                if vg is None:
                    continue
                stats = tabulate(vg.grid, vg.names, None, 0, stats)
                n += 1
            if stats:
                result[style] = summarise(stats)
                print(format_table("{} targets ({} samples)".format(style, n), result[style]))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
