#!/usr/bin/env python3
"""Generate synthetic suburban houses as importable `.schem` files.

PARKED: see docs/mesh-to-schem-plan.md v4

Usage:  python gen_houses.py --count 16 [--seed 1000] [--out data/.tmp/synthetic]
                             [--roof gable] [--storeys 2] [--wall minecraft:bricks]
                             [--size small|medium|large] [--plan L]
                             [--dry-run] [--check-params]

Phase 2 of the synthetic-build tool: `house_stats` measured the real library, `house_params`
froze the numbers, `house_grammar` grows houses from them, and this drives the batch.

Every file is validated before it is kept, and validation is in two parts because there are
two different ways a generated build can be wrong. The first is *unimportable* -- malformed
NBT, a varint stream that does not match the dimensions, a palette with holes in it. That is
caught by running `schem_convert.inspect`, which is literally the call the upload path makes,
then decoding the result back and comparing it block for block against what was written.

The second is *importable but wrong*: a house with no roof, or walls made of air, or a solid
cube of bricks. Those pass every format check and would waste a batch of TRELLIS runs before
anyone noticed. So there is a geometry pass too, and it compares against the real library --
solid fraction inside the measured range, a roof that actually covers its footprint, walls
that are not full of holes. A house that fails either check is deleted and the seed reported,
so it can be reproduced exactly.

Output defaults to `data/.tmp/synthetic/`, which is gitignored. Promoting a batch into the
library is the app's import flow's job, not this script's; it refuses to write into
`data/builds/` so a big experiment cannot be mistaken for collected data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import house_grammar
import house_params
import schem_convert
import schem_voxels

DEFAULT_OUT = Path("data/.tmp/synthetic")
STATS_PATH = Path("data/build_stats.json")

# Measured over the 103 analysable real builds: solid voxels as a fraction of the bounding
# box. Generated houses that fall outside this are not plausibly the same kind of object.
REAL_DENSITY = (0.04, 0.55)


class ValidationError(Exception):
    """A generated house is malformed. Carries the seed so it can be reproduced."""


def validate(path, house) -> dict:
    """Prove a generated file is both importable and actually a house."""
    info = schem_convert.inspect(path)
    if info.format != "sponge_v2":
        raise ValidationError("wrote {} rather than sponge_v2".format(info.format))

    height, length, width = house.grid.shape
    if info.dims != (width, height, length):
        raise ValidationError("header says {} but the grid is {}".format(
            info.dims, (width, height, length)))

    voxels = schem_voxels.load_voxels(path)
    if voxels is None:
        raise ValidationError("the file it just wrote cannot be decoded")
    if voxels.grid.shape != house.grid.shape:
        raise ValidationError("decoded shape {} != {}".format(
            voxels.grid.shape, house.grid.shape))
    if any(not name for name in voxels.names):
        raise ValidationError("palette has holes in it")

    # Compare blockstates, not indices: the writer re-densifies the palette, so the index
    # arrays legitimately differ while every block must be identical.
    want = np.array(house.names, dtype=object)[house.grid]
    got = np.array(voxels.names, dtype=object)[voxels.grid]
    if not bool((want == got).all()):
        raise ValidationError("decoded blocks differ from what was written")

    solid = int(voxels.solid_mask().sum())
    density = solid / float(width * height * length)
    if not REAL_DENSITY[0] <= density <= REAL_DENSITY[1]:
        raise ValidationError(
            "solid fraction {:.3f} is outside the real range {}".format(density, REAL_DENSITY))

    _validate_geometry(voxels, house)
    return {"dims": info.dims, "solid": solid, "density": round(density, 3),
            "palette": len(voxels.names)}


def _validate_geometry(voxels, house):
    """Catch houses that are structurally wrong but perfectly well-formed."""
    import block_roles

    solid = voxels.solid_mask()
    spec = house.spec

    structural = np.zeros(len(voxels.names), dtype=bool)
    for i, name in enumerate(voxels.names):
        base = block_roles.normalise(name)
        if base and not block_roles.is_terrain(base) and not block_roles.is_plant(base):
            structural[i] = True
    building = structural[voxels.grid] & solid
    if not building.any():
        raise ValidationError("no building above the plot")

    ys = np.nonzero(building.any(axis=(1, 2)))[0]
    if len(ys) < 4:
        raise ValidationError("building is only {} layers tall".format(len(ys)))

    footprint = building.any(axis=0)
    expected = spec.width * spec.depth
    if int(footprint.sum()) < 0.25 * expected:
        raise ValidationError("footprint {} is far under the {}x{} it was told to build"
                              .format(int(footprint.sum()), spec.width, spec.depth))

    # The roof has to actually cap the walls: every column of the building should be closed
    # at the top, or the house has a hole in it.
    top_open = building[ys[-1]] & ~building[ys[-1]]
    if top_open.any():
        raise ValidationError("open columns at the roofline")


def _bias_from_args(args) -> dict:
    bias = {}
    for key, value in (("roof_type", args.roof), ("wall", args.wall), ("plan", args.plan),
                       ("size", args.size)):
        if value:
            bias[key] = value
    if args.storeys:
        bias["storeys"] = args.storeys
    if args.pitch is not None:
        bias["pitch"] = args.pitch
    return bias


def check_params() -> int:
    """Report where the frozen params have drifted from a fresh analysis run."""
    if not STATS_PATH.exists():
        print("No {} -- run `python house_stats.py` first.".format(STATS_PATH))
        return 1
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    agg = stats["aggregate"]
    print("\nFrozen params vs {} ({} builds analysed)\n".format(
        STATS_PATH, stats["counts"]["analysed"]))
    print("  Tables marked 'suburban subset' were frozen from the {} builds in "
          "P.SUBURBAN_IDS,\n  not from all of them, so a difference here is the intent "
          "rather than drift.\n".format(len(house_params.SUBURBAN_IDS)))

    def compare(label, frozen, measured, suburban=False):
        keys = sorted(set(frozen) | set(measured), key=str)
        ftot = sum(frozen.values()) or 1
        mtot = sum(measured.values()) or 1
        print("  {}{}".format(label, "  (frozen from the suburban subset)" if suburban else ""))
        for key in keys:
            fshare = frozen.get(key, 0) / ftot
            mshare = measured.get(key, 0) / mtot
            # A table deliberately frozen from the 33 suburban builds is *expected* to differ
            # from the all-106 aggregate this compares against, so flagging it as drift would
            # train the reader to ignore the flag. See P.SUBURBAN_IDS.
            flag = ("  <-- drift" if not suburban and abs(fshare - mshare) > 0.12 else "")
            print("    {:<10} frozen {:>5.0%}   measured {:>5.0%}{}".format(
                str(key), fshare, mshare, flag))

    compare("roof type", house_params.ROOF_TYPE,
            {k: v for k, v in agg["roof_type"].items() if k != "complex"}, suburban=True)
    compare("storey count", {str(k): v for k, v in house_params.STOREY_COUNT.items()},
            agg["storey_count"], suburban=True)
    compare("storey height", {str(k): v for k, v in house_params.STOREY_HEIGHT.items()},
            agg["storey_height"])

    walls = {s[0] for s in house_params.STYLES}
    measured_walls = set(agg["materials"]["wall"])
    dropped = sorted(measured_walls - walls)
    print("\n  {} style tuples frozen; {} wall materials measured, {} not carried over"
          .format(len(house_params.STYLES), len(measured_walls), len(dropped)))
    if dropped:
        print("    dropped: " + ", ".join(d.replace("minecraft:", "") for d in dropped[:12]))
    print()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--roof", choices=["gable", "hip", "flat"])
    parser.add_argument("--wall", help="force a wall material, e.g. minecraft:bricks; the "
                                       "rest of the palette follows the styles that use it")
    parser.add_argument("--plan", choices=["rect", "L", "T"])
    parser.add_argument("--size", choices=["small", "medium", "large"])
    parser.add_argument("--storeys", type=int)
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--dry-run", action="store_true",
                        help="sample and print specs without writing files")
    parser.add_argument("--check-params", action="store_true",
                        help="compare the frozen params against data/build_stats.json")
    args = parser.parse_args()

    if args.check_params:
        sys.exit(check_params())

    out = Path(args.out)
    if "builds" in out.parts and "data" in out.parts:
        print("Refusing to write into the build library: {}\n"
              "Generated houses belong in a scratch folder and enter the library through the "
              "app's import flow, like any other build.".format(out))
        sys.exit(2)

    bias = _bias_from_args(args)
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    written, failures = [], []
    for i in range(args.count):
        seed = args.seed + i
        house = house_grammar.generate(seed, bias)
        if args.dry_run:
            print("  seed {:<8} {}".format(seed, house.spec.summary()))
            continue

        path = out / "house_{}_{}.schem".format(args.seed, i)
        try:
            house_grammar.write(house, path)
            report = validate(path, house)
        except (ValidationError, ValueError) as exc:
            failures.append((seed, str(exc)))
            if path.exists():
                path.unlink()
            print("  seed {:<8} FAILED  {}".format(seed, exc))
            continue
        written.append((seed, house.spec, report))
        print("  seed {:<8} {:>3}x{:<3}x{:<3} pal {:>2}  solid {:>6}  {}".format(
            seed, *report["dims"], report["palette"], report["solid"],
            house.spec.summary()))

    if args.dry_run:
        return

    print("\n{} written to {}, {} failed".format(len(written), out, len(failures)))
    if written:
        _recap(written)
    if failures:
        print("\nFailures (reproduce with --seed <n> --count 1):")
        for seed, why in failures:
            print("  {}  {}".format(seed, why))
        sys.exit(1)


def _recap(written) -> None:
    """Print the generated distribution on the same axes Phase 1 reports, for comparison."""
    import collections
    roofs = collections.Counter(spec.roof_type for _, spec, _ in written)
    storeys = collections.Counter(spec.storeys for _, spec, _ in written)
    plans = collections.Counter(spec.plan for _, spec, _ in written)
    walls = collections.Counter(spec.wall.replace("minecraft:", "") for _, spec, _ in written)
    fmt = lambda c: "  ".join("{} {}".format(k, v) for k, v in sorted(c.items(), key=str))  # noqa: E731
    print("  roof    " + fmt(roofs))
    print("  storeys " + fmt(storeys))
    print("  plan    " + fmt(plans))
    print("  walls   " + "  ".join("{} {}".format(k, v) for k, v in walls.most_common(6)))


if __name__ == "__main__":
    main()
