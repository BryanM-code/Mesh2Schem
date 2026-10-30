"""A model prediction -> a `.schem` file you can paste into Minecraft (WorldEdit:
`//schem load <name>`, then `//paste`).

    .venv-train\\Scripts\\python -m training.export <prediction.npz> --out build.schem

The prediction is one `training.evaluate` saved: runs/<run>/pred_<split>/<set>/<id>.npz, with
`shape` and `material` grids (X, Y, Z). Each (shape, material) pair becomes its block state;
stair corners, door halves and fence, wall and pane arms are then worked out from the
neighbours, as the game does. The file is written for Minecraft 1.21.11.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def palette_grid(shape: np.ndarray, material: np.ndarray) -> tuple:
    """(grid (Y, Z, X) of palette indices, block state names); index 0 is air."""
    from voxel.blockstates import finish_states
    from voxel.vocab import AIR, IGNORE_SHAPE, decode
    shape, material = np.asarray(shape), np.asarray(material)
    solid = (shape != 0) & (material != 0) & (shape != IGNORE_SHAPE)
    keys = shape.astype(np.int64) * 65536 + material.astype(np.int64)
    uniq, inv = np.unique(keys[solid], return_inverse=True)
    names = [AIR] + [decode(int(k // 65536), int(k % 65536)) for k in uniq]
    grid = np.zeros(shape.shape, np.int32)
    grid[solid] = inv.ravel() + 1
    return finish_states(np.ascontiguousarray(np.transpose(grid, (1, 2, 0))), names)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prediction", type=Path, help="a pred_<split>/<set>/<id>.npz from training.evaluate")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    from minecraft.schem_write import write_schem
    from voxel.vocab import TARGET_DATA_VERSION
    z = np.load(args.prediction)
    grid, names = palette_grid(z["shape"], z["material"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    info = write_schem(np.asarray(grid), names, args.out, data_version=TARGET_DATA_VERSION)
    print(f"{args.out}: {info['width']} x {info['height']} x {info['length']} blocks, "
          f"{info['palette_size']} block states")
    return 0


if __name__ == "__main__":
    sys.exit(main())
