#!/usr/bin/env python3
"""Phase 4: the real-TRELLIS evaluation set -- the same builds through the real TRELLIS server.

Usage:  python -m schemset.real_trellis prepare  --dataset data/real-schems-v1 [--n 25]
        python -m schemset.real_trellis generate --dataset data/real-schems-v1   (server on :7960)
        python -m schemset.real_trellis place    --dataset data/real-schems-v1
        python -m schemset.real_trellis status   --dataset data/real-schems-v1

The synthetic inputs imitate TRELLIS; this set checks the imitation on held-out builds and
is the evaluation set a model is scored on for real TRELLIS geometry (D17). Only val and test
builds are used -- never train.

    prepare   choose `--n` builds from val + test, round-robin over top-level categories
              (each category's builds in a fixed shuffled order), meshify each
              (`schemset.meshify`) and render a photo-like RGBA image of its outside
              (`synth.trellis_compare.render_photo`: textured with the thumbnail colours,
              lit, transparent background, a street-level three-quarter view)
    generate  send each photo to the TRELLIS server (the API Real2Minecraft uses,
              `synth.trellis_compare.trellis_generate`, fixed seed) and keep the GLB;
              resumable. Needs the server: start C:\\TRELLIS\\run.bat first
    place     fit each TRELLIS mesh into the build's target frame -- scaled to the target's
              height, stood on its lowest layer, centred on its footprint, and turned to the
              yaw (5-degree search, then 1-degree refinement) with the best block-level IoU
              against the target volume -- then voxelise it at 2 cells per block exactly like
              the synthetic inputs, validate it like a pair, and compare its statistics with
              the synthetic inputs of the same build (the gap report)

Layout under <dataset>/pairs/real_trellis/:

    selection.json            the chosen builds, seed, view
    <id>/photo.png            the image sent to TRELLIS
    <id>/trellis.glb          what TRELLIS returned
    <id>/input.npz            same format as a synthetic pair (input_* at 2x + target_*)
    <id>/stats.json           placement, validation, statistics
    index.json                every build's status and numbers, and the gap report

Caveats, recorded in index.json: the photo is a render of the build, not a real
photograph; the alignment uses the target (height, footprint, yaw), so its IoU is
optimistic; TRELLIS's own randomness is fixed by one seed per build.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

N_BUILDS = 25
SELECT_SEED = 20261014
TRELLIS_SEED = 1
SPLITS = ("val", "test")


def select(dataset: Path, n: int = N_BUILDS) -> list:
    """`n` val / test builds, round-robin over top-level categories."""
    rows = {r["id"]: r for r in json.loads((dataset / "index.json").read_text(encoding="utf-8"))["rows"]}
    norm = json.loads((dataset / "normalized" / "index.json").read_text(encoding="utf-8"))
    by_cat = {}
    for b in sorted(norm["builds"], key=lambda b: b["id"]):
        if b.get("status") == "ok" and b["split"] in SPLITS:
            by_cat.setdefault(rows[b["id"]]["category_top"], []).append(b["id"])
    rng = random.Random(SELECT_SEED)
    for ids in by_cat.values():
        rng.shuffle(ids)
    out = []
    while len(out) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(out) < n:
                out.append(by_cat[cat].pop())
    return out


def prepare(dataset: Path, n: int = N_BUILDS) -> list:
    from synth.trellis_compare import IMAGE_SIZE, VIEW_PITCH, VIEW_YAW, render_photo

    from .meshify import meshify_npz

    out = dataset / "pairs" / "real_trellis"
    out.mkdir(parents=True, exist_ok=True)
    ids = select(dataset, n)
    rows = {r["id"]: r for r in json.loads((dataset / "index.json").read_text(encoding="utf-8"))["rows"]}
    done = []
    for bid in ids:
        d = out / bid
        d.mkdir(exist_ok=True)
        if not (d / "photo.png").is_file():
            mesh, _, _ = meshify_npz(dataset / "normalized" / f"{bid}.npz")
            render_photo(mesh).save(d / "photo.png")
        done.append({"id": bid, "split": rows[bid]["split"], "category": rows[bid]["category_top"],
                     "building_category": rows[bid]["building_category"]})
        print(f"  {bid}: photo ready", flush=True)
    sel = {"builds": done, "seed": SELECT_SEED, "trellis_seed": TRELLIS_SEED,
           "view": {"yaw": VIEW_YAW, "pitch": VIEW_PITCH, "size": IMAGE_SIZE}, "splits": list(SPLITS)}
    (out / "selection.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
    return done


def generate(dataset: Path, limit: int = 0) -> int:
    from synth.trellis_compare import TRELLIS_URL, server_up, trellis_generate

    out = dataset / "pairs" / "real_trellis"
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    if not server_up():
        raise SystemExit(f"TRELLIS server not reachable at {TRELLIS_URL} -- start C:\\TRELLIS\\run.bat first")
    todo = [b["id"] for b in sel["builds"] if not (out / b["id"] / "trellis.glb").is_file()]
    if limit:
        todo = todo[:limit]
    for i, bid in enumerate(todo, 1):
        t0 = time.time()
        trellis_generate(out / bid / "photo.png", out / bid / "trellis.glb", seed=sel["trellis_seed"])
        print(f"  {i}/{len(todo)} {bid}: {time.time() - t0:.0f} s", flush=True)
    return len(todo)


# --------------------------------------------------------------------------- placement

def _fit(verts: np.ndarray, yaw: float, vol: np.ndarray):
    """verts (TRELLIS units) turned by `yaw` about +Y, scaled to the target volume's height,
    stood on its lowest layer and centred on its footprint -> (verts in block units)."""
    t = np.radians(yaw)
    rot = np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])
    v = verts @ rot.T
    ys = np.flatnonzero(vol.any(axis=(0, 2)))
    xs = np.flatnonzero(vol.any(axis=(1, 2)))
    zs = np.flatnonzero(vol.any(axis=(0, 1)))
    lo, hi = v.min(axis=0), v.max(axis=0)
    s = (ys[-1] - ys[0] + 1) / max(hi[1] - lo[1], 1e-9)
    v = v * s
    lo, hi = lo * s, hi * s
    centre = np.array([(xs[0] + xs[-1] + 1) / 2.0, 0.0, (zs[0] + zs[-1] + 1) / 2.0])
    shift = centre - np.array([(lo[0] + hi[0]) / 2.0, 0.0, (lo[2] + hi[2]) / 2.0])
    shift[1] = ys[0] - lo[1]
    return v + shift, float(s)


def _occupancy(mesh, verts, shape, scale: float):
    from mcvoxel.grid import voxelize_scene
    m = mesh.transformed(np.eye(3), np.zeros(3))
    m.verts = verts
    return voxelize_scene(m, scale, offset=(0.0, 0.0, 0.0), origin=np.zeros(3, dtype=np.int64),
                          shape=tuple(int(s * scale) for s in shape), with_shapes=False, crop=False,
                          drop_ground=False, keep_largest=False, clean=False,
                          with_colors=scale != 1.0, color_fallback="nearest")


def place_one(glb: Path, vol: np.ndarray, coarse: int = 5):
    """(VoxelModel at 2x in the target frame, placement info)."""
    from mcvoxel.scene import load_scene

    from .pairs import SCALE, iou

    sm = load_scene(glb)
    verts = sm.verts.copy()
    scores = {}

    def score(yaw):
        if yaw not in scores:
            v, _ = _fit(verts, yaw, vol)
            scores[yaw] = iou(_occupancy(sm, v, vol.shape, 1.0).solid, vol)
        return scores[yaw]

    best = max(range(0, 360, coarse), key=score)
    best = max([best + d for d in range(-coarse + 1, coarse)], key=lambda y: score(y % 360)) % 360
    v, s = _fit(verts, best, vol)
    vm = _occupancy(sm, v, vol.shape, float(SCALE))
    return vm, {"yaw_deg": int(best), "scale_blocks_per_unit": round(s, 4),
                "iou_1x_at_yaw": round(scores[best], 4), "yaws_tried": len(scores),
                "trellis_faces": int(len(sm.faces))}


def place(dataset: Path) -> dict:
    from .normalize import save_npz
    from .pairs import input_stats, normal_roughness, target_volume, validate_pair

    out = dataset / "pairs" / "real_trellis"
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    index = json.loads((dataset / "pairs" / "index.json").read_text(encoding="utf-8"))
    synthetic = {}
    for p in index["pairs"]:
        if p["status"] == "ok":
            synthetic.setdefault(p["id"], []).append(p)
    rows = []
    for b in sel["builds"]:
        bid = b["id"]
        d = out / bid
        row = dict(b)
        if not (d / "trellis.glb").is_file():
            row["status"] = "awaiting_trellis" if (d / "photo.png").is_file() else "awaiting_photo"
            rows.append(row)
            continue
        z = np.load(dataset / "normalized" / f"{bid}.npz")
        vol = target_volume(z["target_shape"], z["exterior"], z["ground_height"])
        t0 = time.time()
        vm, row["placement"] = place_one(d / "trellis.glb", vol)
        pair = {"input_occ": vm.solid.astype(np.uint8), "input_surface": vm.surface.astype(np.uint8),
                "input_rgb": np.ascontiguousarray(vm.rgb, dtype=np.uint8),
                "target_shape": z["target_shape"], "target_material": z["target_material"],
                "ground_height": z["ground_height"].astype(np.int16)}
        failures, metrics = validate_pair(pair, z["exterior"])
        row["validation"] = {"failures": failures, **metrics}
        row["stats"] = input_stats(vm)
        from mcvoxel.scene import load_scene
        tm = load_scene(d / "trellis.glb")
        tv, _ = _fit(tm.verts, row["placement"]["yaw_deg"], vol)
        row["stats"]["normal_roughness"] = normal_roughness(tv, tm.faces)
        save_npz(d / "input.npz", pair)
        row["status"] = "ok"
        row["seconds"] = round(time.time() - t0, 1)
        row["synthetic"] = [{"variant": p["variant"], "iou": p["validation"]["iou"], **p["stats"]}
                            for p in synthetic.get(bid, [])]
        (d / "stats.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
        rows.append(row)
        print(f"  {bid}: IoU {metrics.get('iou')} yaw {row['placement']['yaw_deg']}", flush=True)
    report = gap_report(rows)
    res = {"version": "real-trellis-1", "builds": rows, "gap": report,
           "caveats": ["The photo is a render of the build (thumbnail colours), not a real photograph.",
                       "Placement uses the target (height, footprint, yaw): IoU is optimistic.",
                       "One TRELLIS seed per build."]}
    (out / "index.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


GAP_KEYS = ("iou", "coverage", "excess", "fill_gain", "rough3", "lum", "chroma", "nbr_de",
            "step2_share", "normal_roughness")


def gap_report(rows: list) -> dict:
    """Per statistic: real TRELLIS p10/p50/p90, the synthetic inputs of the same builds, and
    the paired median of (synthetic - real) over builds (synthetic averaged over variants)."""
    done = [r for r in rows if r.get("status") == "ok" and r.get("synthetic")]
    out = {"builds": len(done)}
    for k in GAP_KEYS:
        real, syn, diff = [], [], []
        for r in done:
            rv = r["validation"].get(k) if k in ("iou", "coverage", "excess") else r["stats"].get(k)
            sv = [s.get(k) for s in r["synthetic"] if s.get(k) is not None]
            if k in ("coverage", "excess"):
                sv = []
            if rv is None:
                continue
            real.append(rv)
            if sv:
                syn.append(float(np.mean(sv)))
                diff.append(float(np.mean(sv)) - rv)
        out[k] = {"real_p10_p50_p90": _pct(real), "synthetic_p10_p50_p90": _pct(syn),
                  "paired_median_diff": round(float(np.median(diff)), 4) if diff else None}
    out["reading"] = "paired_median_diff > 0: the synthetic input has more of it than real TRELLIS"
    return out


def _pct(v):
    v = np.array([x for x in v if x is not None], dtype=np.float64)
    return [round(float(np.percentile(v, p)), 4) for p in (10, 50, 90)] if len(v) else None


def status(dataset: Path) -> dict:
    from synth.trellis_compare import server_up
    out = dataset / "pairs" / "real_trellis"
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8")) if (out / "selection.json").is_file() else {"builds": []}
    st = {"selected": len(sel["builds"]),
          "photos": sum((out / b["id"] / "photo.png").is_file() for b in sel["builds"]),
          "trellis_meshes": sum((out / b["id"] / "trellis.glb").is_file() for b in sel["builds"]),
          "placed": sum((out / b["id"] / "input.npz").is_file() for b in sel["builds"]),
          "server_up": server_up()}
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["prepare", "generate", "place", "status"])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--n", type=int, default=N_BUILDS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dataset = args.dataset.resolve()
    if args.cmd == "prepare":
        done = prepare(dataset, args.n)
        print(f"{len(done)} photos ready in {dataset / 'pairs' / 'real_trellis'}")
    elif args.cmd == "generate":
        print(f"generated {generate(dataset, args.limit)}")
    elif args.cmd == "place":
        res = place(dataset)
        print(json.dumps(res["gap"], indent=1))
    else:
        print(json.dumps(status(dataset), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
