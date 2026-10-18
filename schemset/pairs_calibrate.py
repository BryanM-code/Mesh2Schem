#!/usr/bin/env python3
"""Phase 4 domain-gap guard: synthetic inputs vs real TRELLIS output, same statistics.

Usage:  python -m schemset.pairs_calibrate --dataset data/real-schems-v1
                                           [--trellis <folder of .glb>] [--sizes 36 57 87]
                                           [--refresh] [--freeze-reference <folder>]

Reference: the TRELLIS meshes cached in Real2Minecraft `outputs/trellis/` (photos and AI
images of real houses and towers, plus a car and a tree, which `buildings` leaves out),
frozen once with `--freeze-reference ../Real2Minecraft/outputs/trellis` into
<dataset>/pairs/trellis_reference/ (meshes/ + manifest.json with each file's sha256). The
calibration reads that copy and stops if a mesh changed, went missing or was added, so the
targets cannot drift when the app writes new meshes to its outputs folder.
Each is yaw-aligned to its facades (`mcvoxel.yaw_align`, as building mode does), stood on
y = 0, scaled to each of `--sizes` blocks on its longest axis (default: the dataset's p25 /
p50 / p75 longest extent), and voxelised at 2 cells per block exactly as the synthetic
inputs are (`schemset.pairs.make_input`: no crop to a target, no plate removal, no
cleanup). The same statistics are then taken on both sides:

    blockiness   stair45           on 45-degree stretches of the top surface, the share of
                                   (0, 2) step pairs (block stairs) against (1, 1) (smooth)
                 normal_roughness  mean angle (deg) between the surface normal and its mean
                                   within 1 block: staircases high, smooth surfaces low
                 lattice16, axis_area   `synth.blockiness` (lattice share, axis-facing area)
    calibration  fill_gain, rough3 (blobby shell), lum (L*), chroma (C*), nbr_de
                 (neighbour colour difference) -- `synth.calibrate`

Also in the tables: the real-TRELLIS evaluation sets (`schemset.real_trellis`: smooth
photos = the main set, blocky photos = the stress test) where they have been placed, and
with `--before <index.json>` an earlier pairs version (the colour model's before / after).
`colour_acceptance` checks the colour model against TRELLIS on real photos: the synthetic
median inside the reference p10-p90 for lum, chroma and nbr_de, at least `INSIDE_MIN` of
the synthetic inputs inside that band for lum and chroma, and `pairs.COLOUR_TARGET`.

Writes <dataset>/pairs/trellis_reference.json (cached; --refresh redoes it) and
<dataset>/pairs/calibration.json (p10 / p50 / p90 side by side, and the overlap verdict
for the blockiness metrics). The synthetic side is read from pairs/index.json. Size
matters (TRELLIS's own 64^3 structure terraces a mesh more the more blocks it spans), so
each synthetic build is also compared with the reference at the nearest size: a size bin
per reference size, bounded halfway between neighbouring sizes (and +-30% at the ends).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BLOCKY = ("stair45", "normal_roughness")
COLOUR_KEYS = ("lum", "chroma", "nbr_de")
INSIDE_MIN = 0.70
EVAL_SETS = {"eval_smooth": "real_trellis_smooth", "eval_blocky": "real_trellis"}
REPORT = ("stair45", "normal_roughness", "lattice16", "axis_area", "fill_gain", "rough3",
          "lum", "chroma", "nbr_de")
NOT_BUILDINGS = ("Car_", "Tree_")
REFERENCE_DIR = "trellis_reference"          # <dataset>/pairs/: meshes/ + manifest.json


def trellis_row(glb: Path, size: int) -> dict:
    from mcvoxel.grid import voxelize_scene
    from mcvoxel.scene import load_scene
    from mcvoxel.yaw_align import apply_yaw, estimate_yaw
    from synth.blockiness import blockiness

    from .pairs import SCALE, input_stats, normal_roughness

    sm = load_scene(glb)
    yaw = estimate_yaw(sm.verts, sm.faces)
    v = apply_yaw(sm.verts, yaw["theta_deg"]) if yaw["applied"] else sm.verts.copy()
    v[:, 1] -= v[:, 1].min()
    s = size / float(np.ptp(v, axis=0).max())
    sm.verts = v * s                                       # block units
    vm = voxelize_scene(sm, float(SCALE), offset=(0.0, 0.0, 0.0), with_shapes=False,
                        drop_ground=False, keep_largest=False, clean=False, color_fallback="nearest")
    row = {"name": glb.name, "size": size, "yaw_applied": bool(yaw["applied"]),
           "building": not glb.name.startswith(NOT_BUILDINGS), **input_stats(vm)}
    row["normal_roughness"] = normal_roughness(sm.verts, sm.faces)
    b = blockiness(sm.verts, sm.faces)
    row["lattice16"], row["axis_area"] = b["lattice16"], b["axis_area"]
    return row


def _job(args):
    glb, size = args
    t0 = time.time()
    try:
        row = trellis_row(Path(glb), size)
    except Exception as exc:                                            # noqa: BLE001
        return {"name": Path(glb).name, "size": size, "skipped": f"{type(exc).__name__}: {exc}"[:200]}
    row["seconds"] = round(time.time() - t0, 2)
    return row


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_reference(src: Path, dataset: Path) -> Path:
    """Copy the reference meshes (`*.glb` in `src`) to <dataset>/pairs/trellis_reference/
    meshes/ and record each one's sha256 in manifest.json beside them, so the calibration
    targets cannot change without notice. Returns the meshes folder."""
    import shutil
    out = dataset / "pairs" / REFERENCE_DIR
    meshes = out / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    files = []
    for glb in sorted(Path(src).glob("*.glb")):
        dst = meshes / glb.name
        shutil.copy2(glb, dst)
        files.append({"name": glb.name, "sha256": _sha256(dst), "bytes": dst.stat().st_size,
                      "building": not glb.name.startswith(NOT_BUILDINGS)})
    manifest = {"source": str(Path(src).resolve()), "frozen": time.strftime("%Y-%m-%d"),
                "files": files, "set_sha256": _set_digest(files)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return meshes


def _set_digest(files) -> str:
    import hashlib
    return hashlib.sha256("\n".join(f"{f['name']} {f['sha256']}" for f in
                                    sorted(files, key=lambda f: f["name"])).encode()).hexdigest()


def check_reference(folder: Path) -> str | None:
    """The frozen reference must be exactly what its manifest lists: every mesh present
    with its checksum, and nothing else. Returns the set digest, or None for a folder
    without a manifest (an unfrozen reference); raises on any difference."""
    path = Path(folder).parent / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    listed = {f["name"]: f["sha256"] for f in manifest["files"]}
    present = {p.name for p in Path(folder).glob("*.glb")}
    problems = [f"missing: {n}" for n in sorted(set(listed) - present)]
    problems += [f"not in the manifest: {n}" for n in sorted(present - set(listed))]
    problems += [f"changed: {n}" for n in sorted(set(listed) & present)
                 if _sha256(Path(folder) / n) != listed[n]]
    if problems:
        raise SystemExit("the TRELLIS reference differs from its manifest "
                         f"({path}):\n  " + "\n  ".join(problems))
    return manifest["set_sha256"]


def trellis_reference(folder: Path, sizes, cache: Path, refresh: bool = False, workers: int = 1) -> list:
    digest = check_reference(folder)
    if cache.is_file() and not refresh:
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("sizes") == list(sizes) and data.get("folder") == str(folder) \
                and data.get("set_sha256") == digest:
            return data["rows"]
    jobs = [(str(glb), size) for glb in sorted(folder.glob("*.glb")) for size in sizes]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    rows = []
    for row in results:
        if "skipped" in row:
            print(f"  {row['name']} @ {row['size']}: skipped ({row['skipped']})", flush=True)
            continue
        rows.append(row)
    print(f"  {len(rows)} reference rows from {len({r['name'] for r in rows})} meshes", flush=True)
    cache.write_text(json.dumps({"folder": str(folder), "sizes": list(sizes), "set_sha256": digest,
                                 "rows": rows}, indent=1), encoding="utf-8")
    return rows


def pct(values, ps=(10, 50, 90)):
    v = np.array([x for x in values if x is not None], dtype=np.float64)
    return [round(float(np.percentile(v, p)), 3) for p in ps] if len(v) else None


def overlap(syn, ref) -> dict:
    """How the synthetic distribution sits against the reference: its median within the
    reference p10-p90, and the share of synthetic values inside that range."""
    s = np.array([x for x in syn if x is not None], dtype=np.float64)
    r = np.array([x for x in ref if x is not None], dtype=np.float64)
    if not len(s) or not len(r):
        return {}
    lo, hi = np.percentile(r, 10), np.percentile(r, 90)
    return {"synthetic_p50_in_reference_p10_p90": bool(lo <= np.median(s) <= hi),
            "share_of_synthetic_in_reference_p10_p90": round(float(((s >= lo) & (s <= hi)).mean()), 3),
            "median_diff": round(float(np.median(s) - np.median(r)), 3)}


def size_bins(sizes) -> list:
    """[(size, lo, hi)]: each reference size's bin of synthetic longest extents."""
    s = sorted(sizes)
    edges = [s[0] * 0.7] + [(a + b) / 2.0 for a, b in zip(s, s[1:])] + [s[-1] * 1.3]
    return [(size, edges[i], edges[i + 1]) for i, size in enumerate(s)]


def eval_rows(dataset: Path) -> dict:
    """{set: [per-build stats]} of the placed real-TRELLIS sets (the TRELLIS mesh's
    roughness and the 2x input's statistics, as for a synthetic input)."""
    out = {}
    for key, folder in EVAL_SETS.items():
        path = dataset / "pairs" / folder / "index.json"
        if path.is_file():
            rows = [b for b in json.loads(path.read_text(encoding="utf-8"))["builds"] if b.get("status") == "ok"]
            if rows:
                out[key] = [b["stats"] for b in rows]
    return out


def colour_acceptance(syn: list, ref: list) -> dict:
    """The colour model against the reference (see the module docstring)."""
    from .pairs import COLOUR_TARGET
    res = {"checks": {}}
    for k in COLOUR_KEYS:
        o = overlap([r.get(k) for r in syn], [r.get(k) for r in ref])
        res["checks"][f"{k}_p50_inside_reference_p10_p90"] = o.get("synthetic_p50_in_reference_p10_p90")
        if k in ("lum", "chroma"):
            res["checks"][f"{k}_share_inside_band"] = o.get("share_of_synthetic_in_reference_p10_p90")
            res["checks"][f"{k}_share_inside_band_ok"] = (o.get("share_of_synthetic_in_reference_p10_p90") or 0) >= INSIDE_MIN
    lum = pct([r.get("lum") for r in syn])
    chroma = pct([r.get("chroma") for r in syn])
    t = COLOUR_TARGET
    res["targets"] = {"lum_p10": [lum[0], t["lum_p10"][0] <= lum[0] <= t["lum_p10"][1]],
                      "lum_p50": [lum[1], t["lum_p50"][0] <= lum[1] <= t["lum_p50"][1]],
                      "lum_p90": [lum[2], t["lum_p90"][0] <= lum[2] <= t["lum_p90"][1]],
                      "chroma_p50": [chroma[1], t["chroma_p50"][0] <= chroma[1] <= t["chroma_p50"][1]],
                      "chroma_p90": [chroma[2], chroma[2] <= t["chroma_p90_max"]]}
    res["accepted"] = bool(all(v for k, v in res["checks"].items() if k.endswith(("_p90", "_ok")))
                           and all(v[1] for v in res["targets"].values()))
    return res


def compare(dataset: Path, trellis: Path, sizes, refresh: bool = False, label: str = "",
            workers: int = 1, pairs_rows=None, before: Path = None) -> dict:
    pairs = dataset / "pairs"
    ref = trellis_reference(trellis, sizes, pairs / "trellis_reference.json", refresh, workers)
    norm = json.loads((dataset / "normalized" / "index.json").read_text(encoding="utf-8"))
    extent = {b["id"]: max(b["crop"]["dims_after"]) for b in norm["builds"]}
    if pairs_rows is None:
        index = json.loads((pairs / "index.json").read_text(encoding="utf-8"))
        pairs_rows = [p for p in index["pairs"] if p.get("stats")]
        smooth = index["settings"].get("smooth_sigma_blocks")
    else:
        smooth = None
    ref_b = [r for r in ref if r["building"]]
    bins = size_bins(sizes)
    matched_syn, matched_ref = [], []
    by_size = {}
    for size, lo, hi in bins:
        syn = [p["stats"] for p in pairs_rows if lo <= extent.get(p["id"], 0) < hi]
        rr = [r for r in ref_b if r["size"] == size]
        matched_syn += syn
        matched_ref += rr
        by_size[str(size)] = {"extent_bin": [round(lo, 1), round(hi, 1)],
                              "synthetic": {"n": len(syn), **{m: pct([r.get(m) for r in syn]) for m in REPORT}},
                              "trellis": {"n": len(rr), **{m: pct([r.get(m) for r in rr]) for m in REPORT}},
                              "overlap": {m: overlap([r.get(m) for r in syn], [r.get(m) for r in rr])
                                          for m in BLOCKY}}
    groups = {"trellis_all": ref, "trellis_buildings": ref_b,
              "synthetic_all": [p["stats"] for p in pairs_rows],
              "synthetic_size_matched": matched_syn}
    if before and Path(before).is_file():
        prev = json.loads(Path(before).read_text(encoding="utf-8"))
        groups["synthetic_before_" + prev.get("version", "?")] = [p["stats"] for p in prev["pairs"] if p.get("stats")]
    groups.update(eval_rows(dataset))
    table = {k: {"n": len(rows), **{m: pct([r.get(m) for r in rows]) for m in REPORT}}
             for k, rows in groups.items()}
    verdict = {m: overlap([r.get(m) for r in matched_syn], [r.get(m) for r in matched_ref]) for m in REPORT}
    blockier = sorted({m for m in BLOCKY for b in [verdict] + [v["overlap"] for v in by_size.values()]
                       if b.get(m, {}).get("median_diff", 0) > 0
                       and not b.get(m, {}).get("synthetic_p50_in_reference_p10_p90", True)})
    return {"label": label, "smooth_sigma_blocks": smooth, "sizes": list(sizes), "table": table,
            "by_size": by_size, "overlap": verdict, "synthetic_blockier": blockier,
            "colour": colour_acceptance(groups["synthetic_all"], ref_b),
            "reading": ("p10 / p50 / p90. overlap: synthetic against TRELLIS buildings, size-matched "
                        "(each synthetic build against the reference at the nearest size)")}


def print_table(res: dict) -> None:
    cols = list(res["table"])
    print(f"{'stat (p10,p50,p90)':18s}" + "".join(f"{c + ' n=' + str(res['table'][c]['n']):>30s}" for c in cols))
    for m in REPORT:
        print(f"{m:18s}" + "".join(f"{str(res['table'][c][m]):>30s}" for c in cols))
    for size, b in res["by_size"].items():
        print(f"size {size} (synthetic extent {b['extent_bin']}): synthetic n={b['synthetic']['n']}, "
              f"TRELLIS n={b['trellis']['n']}")
        for m in BLOCKY:
            print(f"   {m:18s} synthetic {str(b['synthetic'][m]):>24s}   TRELLIS {str(b['trellis'][m]):>24s}")
    print("synthetic blockier than TRELLIS on:", res["synthetic_blockier"] or "none")
    c = res["colour"]
    print("colour model accepted:", c["accepted"])
    for k, v in {**c["checks"], **c["targets"]}.items():
        print(f"   {k:40s} {v}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--trellis", type=Path, default=None,
                    help="reference meshes (default: the frozen copy, pairs/trellis_reference/meshes)")
    ap.add_argument("--freeze-reference", type=Path, metavar="SRC",
                    help="copy SRC/*.glb into pairs/trellis_reference/ with checksums, then exit")
    ap.add_argument("--sizes", type=int, nargs="*", default=[36, 57, 87])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--before", type=Path, help="an earlier pairs/index.json, shown beside the current one")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dataset = args.dataset.resolve()
    if args.freeze_reference:
        meshes = freeze_reference(args.freeze_reference, dataset)
        print(f"reference frozen: {len(list(meshes.glob('*.glb')))} meshes in {meshes}")
        return 0
    trellis = args.trellis or dataset / "pairs" / REFERENCE_DIR / "meshes"
    if not trellis.is_dir():
        raise SystemExit(f"no reference meshes in {trellis} (freeze them with --freeze-reference)")
    res = compare(dataset, trellis.resolve(), args.sizes, args.refresh, args.label,
                  args.workers, before=args.before)
    path = args.dataset / "pairs" / "calibration.json"
    hist = json.loads(path.read_text(encoding="utf-8")).get("history", []) if path.is_file() else []
    hist = [h for h in hist if h.get("label") != res["label"] or not res["label"]] + [res]
    path.write_text(json.dumps({**res, "history": hist}, indent=1), encoding="utf-8")
    print_table(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
