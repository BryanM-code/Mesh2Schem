#!/usr/bin/env python3
"""Compare degraded training inputs with real TRELLIS output, at the voxel level.

Usage:  python -m synth.calibrate [--trellis ../Real2Minecraft/outputs/trellis] [--root data/synthetic/pilot-v1]

Mesh-level counts (open edges, components) are not the right yardstick: voxelisation closes
sub-voxel cracks and the cleanup drops floaters, so what the model actually sees is the
voxel grid. Both sides go through the same `voxelize_scene` and the same statistics:

    kept_fraction      solid kept by keep-largest-component (1 - floater share)
    fill_gain          filled voxels per surface voxel (how enclosed the mesh is)
    rough3             share of shell voxels with 3+ faces open to air (blobby surfaces)
    lum, chroma        mean L* and C* of shell colours (TRELLIS bakes dark)
    nbr_de             mean colour difference between face-adjacent shell voxels (noise)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mcvoxel.block_palette import srgb_to_lab
from mcvoxel.grid import voxelize_scene
from mcvoxel.scene import load_scene


def voxel_stats(solid, shell, rgb):
    pad = np.pad(solid, 1)
    open_faces = sum((~pad[tuple(slice(1 + (d if a == ax else 0), pad.shape[a] - 1 + (d if a == ax else 0))
                                 for a in range(3))]).astype(np.int8)
                     for ax in range(3) for d in (-1, 1))
    rough3 = float((open_faces[shell] >= 3).mean()) if shell.any() else 0.0
    cols = rgb[shell].astype(np.float64)
    lab = srgb_to_lab(cols) if len(cols) else np.zeros((0, 3))
    lum = float(lab[:, 0].mean()) if len(lab) else 0.0
    chroma = float(np.hypot(lab[:, 1], lab[:, 2]).mean()) if len(lab) else 0.0
    # neighbour colour difference along +x, +y, +z where both voxels are shell
    full = np.zeros(solid.shape + (3,))
    full[shell] = lab
    des = []
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax], b[ax] = slice(0, -1), slice(1, None)
        both = shell[tuple(a)] & shell[tuple(b)]
        if both.any():
            des.append(np.linalg.norm(full[tuple(a)][both] - full[tuple(b)][both], axis=1))
    nbr = float(np.concatenate(des).mean()) if des else 0.0
    return {"rough3": rough3, "lum": lum, "chroma": chroma, "nbr_de": nbr}


def trellis_side(folder: Path, blocks=(32, 48, 64)):
    rows = []
    for glb in sorted(folder.glob("*.glb")):
        try:
            sm = load_scene(glb)
        except Exception:
            continue
        lo, hi = sm.bounds
        ext = float((hi - lo).max())
        for b in blocks:
            s = b / ext
            vm = voxelize_scene(sm, s, offset=-lo * s + 0.3, with_shapes=False)
            st = voxel_stats(vm.solid, vm.shell, vm.rgb)
            st.update(kept_fraction=vm.info["kept_fraction"], fill_gain=vm.info["fill_gain"], blocks=b,
                      name=glb.name)
            rows.append(st)
    return rows


def pilot_side(root: Path):
    rows = []
    for mp in sorted((root / "samples").glob("*/manifest.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m.get("status") != "ok":
            continue
        d = np.load(mp.parent / "input.npz")
        solid = d["occ"].astype(bool).transpose(2, 0, 1)        # (Y,Z,X) -> (X,Y,Z)
        rgb = d["rgb"].transpose(2, 0, 1, 3)
        pad = np.pad(solid, 1)
        inner = (pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1] & pad[1:-1, 2:, 1:-1] &
                 pad[1:-1, :-2, 1:-1] & pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2])
        shell = solid & ~inner
        st = voxel_stats(solid, shell, rgb)
        grid = m.get("grid_xyz") or m["resolution"]["grid_xyz"]          # pilot-v1 / pipeline v2
        st.update(kept_fraction=m["input_voxels"]["kept_fraction"], fill_gain=m["input_voxels"]["fill_gain"],
                  blocks=max(grid) - 6, name=m["uid"])
        rows.append(st)
    return rows


def summarise(rows, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = [round(float(np.percentile(v, p)), 3) for p in (10, 50, 90)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trellis", default="../Real2Minecraft/outputs/trellis")
    ap.add_argument("--root", default="data/synthetic/pilot-v1")
    args = ap.parse_args(argv)
    keys = ["kept_fraction", "fill_gain", "rough3", "lum", "chroma", "nbr_de"]
    t = trellis_side(Path(args.trellis))
    p = pilot_side(Path(args.root))
    p_mid = [r for r in p if 25 <= r["blocks"] <= 70]            # sizes comparable with 32-64
    res = {"trellis": {"n": len(t), **summarise(t, keys)},
           "pilot_inputs": {"n": len(p), **summarise(p, keys)},
           "pilot_inputs_25_70_blocks": {"n": len(p_mid), **summarise(p_mid, keys)}}
    Path(args.root, "calibration.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("{:16s} {:>28s} {:>28s} {:>28s}".format("stat (p10,p50,p90)", "TRELLIS n=%d" % len(t),
                                                  "pilot inputs n=%d" % len(p), "pilot 25-70 blocks n=%d" % len(p_mid)))
    for k in keys:
        print("{:16s} {:>28s} {:>28s} {:>28s}".format(k, str(res["trellis"][k]), str(res["pilot_inputs"][k]),
                                                      str(res["pilot_inputs_25_70_blocks"][k])))


if __name__ == "__main__":
    main()
