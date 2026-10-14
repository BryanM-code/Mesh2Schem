#!/usr/bin/env python3
"""Phase 3: read, crop, mask and measure every build of the frozen schematic dataset.

Usage:  python -m schemset.normalize --dataset data/real-schems-v1
                                     [--only <id> ...] [--workers N] [--force] [--no-review]

For every index row except split "excluded" (tier X):

    read      `schemset.read`: any format -> palette grid, validated against index.json
    crop      `schemset.crop`: natural terrain, trees and site structures removed, ground
              height per column, padded with air (D23)
    masks     `schemset.interior`: shell / exterior / interior by opening size (D22;
              ignored, not deleted), plus the visibility audit
    storey    `schemset.storey`: blocks per storey + confidence (information only, D15)
    targets   `mcvoxel.vocab.encode_target`: (shape, material) training targets in the
              current vocabulary -- exterior air, shell blocks the vocabulary holds, and
              IGNORE for the interior, decor, unsupported blocks and door upper halves (D20)
    coverage  supervised (shell) block counts per blockstate, for `schemset.coverage`

Writes, all under `<dataset>/` and never inside `source/`:

    normalized/<id>.npz           blocks (uint16 X, Y, Z), palette, shell, exterior, interior
                                  (bool), crop_offset (source-grid coordinates of [0,0,0]),
                                  ground_height (int16 X, Z: the ground height of each
                                  column in this grid, at or below is ground, -1 none),
                                  ground_y (its median, -1 none), target_shape (uint8) /
                                  target_material (uint16) in the vocabulary named by the
                                  record's `vocab_version`
    normalized/records/<id>.json  everything measured for one build
    normalized/index.json         per build: status (ok / failed + reason), validation, crop
                                  stats, flags, interior fraction, storey; plus a summary
    normalized/previous_run.json  the previous pipeline version's per-build numbers, kept
                                  when VERSION changes (for "changed vs previous run")
    vocab_coverage.json           `schemset.coverage` over every ok build
    review/gallery_v2/            `schemset.gallery`: one card per build, before / after

Resumable: a build whose record has the same source sha256, pipeline VERSION, vocabulary
version and vocabulary digest (`vocab_digest`, over the tables), and whose npz exists, is
skipped (--force redoes it). Deterministic: same inputs,
same bytes (the npz zip members carry a fixed timestamp).
"""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import io
import json
import os
import sys
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

VERSION = "normalize-3"             # 3: Phase 3 fixes -- D22 opening-size interior, D23
                                    #    terrain and per-column ground, site lines, trees


# --------------------------------------------------------------------------- per build

def save_npz(path: Path, arrays: dict) -> None:
    """np.savez_compressed, but byte-identical across runs (fixed member timestamps)."""
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, value in arrays.items():
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(value), allow_pickle=False)
            zf.writestr(info, buf.getvalue())
    os.replace(tmp, path)


@functools.lru_cache(maxsize=1)
def vocab_digest() -> str:
    """Digest of the vocabulary tables (shapes, materials, VALID, aliases): an edit to them
    changes the targets' IDs, so it must re-encode the targets even within one version."""
    from mcvoxel import vocab
    return hashlib.sha256(vocab.vocab_json().encode("utf-8")).hexdigest()[:12]


def fingerprint(row: dict) -> str:
    from mcvoxel.vocab import VOCAB_VERSION
    return f"{row['sha256']}:{VERSION}:{VOCAB_VERSION}:{vocab_digest()}"


def process(row: dict, dataset: str) -> dict:
    """Normalize one build; never raises (a failure is a record with a reason)."""
    from .crop import crop_build, floor_under_rooms
    from .interior import compute_masks
    from .read import ReadError, load_build, validate
    from .storey import estimate

    dataset = Path(dataset)
    rec = {"id": row["id"], "split": row["split"], "licence_tier": row["licence_tier"],
           "file": row["file"], "file_type": row["file_type"],
           "building_category": row["building_category"],
           "fingerprint": fingerprint(row), "version": VERSION}
    try:
        try:
            build = load_build(dataset / "source" / row["file"])
        except ReadError as exc:
            return {**rec, "status": "failed", "reason": f"read: {exc}"}
        rec["read"] = {"source_format": build.source_format, "data_version": build.data_version,
                       "dims": list(build.dims), "regions": build.regions,
                       "palette_size": len(build.palette), "warnings": build.warnings,
                       "unknown_legacy": build.unknown_legacy,
                       "unknown_legacy_voxels": sum(build.unknown_legacy.values()),
                       "non_vanilla": build.non_vanilla}
        rec["validation"] = validate(build, row)

        crop = crop_build(build)
        if "crop_empty" in crop.flags:
            rec["crop"] = {**crop.stats, "offset": list(crop.offset), "flags": crop.flags}
            return {**rec, "status": "failed", "reason": "crop: no block survived the crop"}
        masks = compute_masks(crop.blocks, crop.ground)
        if floor_under_rooms(build, crop, masks.interior):      # updates crop.stats too
            masks = compute_masks(crop.blocks, crop.ground)     # same masks; stats see the floor
        rec["crop"] = {**crop.stats, "offset": list(crop.offset), "flags": crop.flags}
        rec["masks"] = {**masks.stats, "flags": masks.flags}
        rec["storey"] = estimate(crop.blocks, crop.palette, masks, masks.stats["ground_y"])

        counts = np.bincount(crop.blocks[masks.shell], minlength=len(crop.palette))
        rec["supervised"] = {crop.palette[i]: int(n) for i, n in enumerate(counts) if n}
        rec["flags"] = sorted(set(row["flags"]) | set(crop.flags) | set(masks.flags))

        from mcvoxel import vocab
        t_shape, t_mat = vocab.encode_target(crop.blocks, crop.palette, masks.shell,
                                             masks.exterior, masks.interior)
        shell_ignored = masks.shell & (t_shape == vocab.IGNORE_SHAPE)
        rec["targets"] = {"vocab_version": vocab.VOCAB_VERSION, "vocab_digest": vocab_digest(),
                          "supervised_blocks": int(masks.shell.sum()),
                          "supervised_air": int(masks.exterior.sum()),
                          "shell_ignored": int(shell_ignored.sum()),
                          "shapes": {vocab.SHAPES[int(s)]: int(n) for s, n in zip(
                              *np.unique(t_shape[masks.shell & ~shell_ignored], return_counts=True))}}
        name = f"{row['id']}.npz"
        save_npz(dataset / "normalized" / name, {
            "blocks": crop.blocks.astype(np.uint16),
            "palette": np.array(crop.palette, dtype=str),
            "shell": masks.shell, "exterior": masks.exterior, "interior": masks.interior,
            "crop_offset": np.array(crop.offset, dtype=np.int32),
            "ground_height": crop.ground.astype(np.int16),
            "ground_y": np.array(masks.stats["ground_y"], dtype=np.int32),
            "target_shape": t_shape, "target_material": t_mat,
        })
        rec["npz"] = name
        rec["status"] = "ok"
        return rec
    except Exception as exc:                                    # noqa: BLE001
        tail = traceback.format_exc().strip().splitlines()[-1]
        return {**rec, "status": "failed", "reason": f"error: {type(exc).__name__}: {tail}"}


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                   newline="\n")
    os.replace(tmp, path)


def _is_current(norm: Path, row: dict) -> bool:
    rec_path = norm / "records" / f"{row['id']}.json"
    if not rec_path.is_file():
        return False
    try:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    if rec.get("fingerprint") != fingerprint(row):
        return False
    return rec.get("status") == "failed" or (norm / f"{row['id']}.npz").is_file()


# --------------------------------------------------------------------------- summary

def _quantiles(values, qs=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)):
    if not values:
        return {}
    v = sorted(values)
    return {f"p{int(q * 100)}": v[int(round(q * (len(v) - 1)))] for q in qs}


def index_entry(rec: dict) -> dict:
    entry = {"id": rec["id"], "split": rec["split"], "licence_tier": rec["licence_tier"],
             "building_category": rec["building_category"], "status": rec.get("status", "pending")}
    if rec.get("reason"):
        entry["reason"] = rec["reason"]
    if "validation" in rec:
        v = rec["validation"]
        entry["validation"] = {"dims": v["dims"], "non_air": v["non_air"],
                               "unknown_legacy_voxels": rec["read"]["unknown_legacy_voxels"],
                               "source_format": rec["read"]["source_format"]}
    if "crop" in rec:
        c = rec["crop"]
        entry["crop"] = {k: c[k] for k in ("ground_level", "ground_method", "bbox_before",
                                           "bbox_after", "dims_after", "removed_fraction",
                                           "removed_landscape", "removed_site",
                                           "removed_lines", "removed_built_above_ground",
                                           "large_masses", "offset") if k in c}
    if "masks" in rec:
        m = rec["masks"]
        entry["interior_fraction"] = m["interior_fraction"]
        entry["exposure"] = {k: m.get(k) for k in ("shell", "visible_ignored_fraction",
                                                   "below_ground_fraction",
                                                   "covered_exterior_share")}
    if "storey" in rec:
        s = rec["storey"]
        entry["storey"] = {"blocks_per_storey": s["blocks_per_storey"],
                           "confidence": s["confidence"], "method": s["method"]}
    entry["flags"] = rec.get("flags", [])
    if rec.get("npz"):
        entry["npz"] = rec["npz"]
    return entry


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]
    failed = [r for r in records if r.get("status") == "failed"]
    read = [r for r in records if "validation" in r]
    storeys = collections.Counter()      # information only: nothing is filtered on it (D15)
    confidence = collections.Counter()
    for r in ok:
        s = r["storey"]
        storeys[f"{s['blocks_per_storey']}|{s['confidence']}"] += 1
        confidence[s["confidence"]] += 1
    targets = [r["targets"] for r in ok if "targets" in r]
    return {
        "builds": len(records), "ok": len(ok), "failed": len(failed),
        "failures": [{"id": r["id"], "reason": r["reason"]} for r in failed],
        "validation": {
            "dims": dict(collections.Counter(r["validation"]["dims"] for r in read)),
            "non_air": dict(collections.Counter(r["validation"]["non_air"] for r in read)),
            "mismatches": [{"id": r["id"], **r["validation"]} for r in read
                           if r["validation"]["dims"] != "ok"
                           or r["validation"]["non_air"] == "mismatch"],
            "unknown_legacy": {r["id"]: r["read"]["unknown_legacy"] for r in read
                               if r["read"]["unknown_legacy"]},
            "source_formats": dict(collections.Counter(r["read"]["source_format"] for r in read)),
        },
        "flags": dict(collections.Counter(f for r in ok for f in r["flags"]).most_common()),
        "crop": {"removed_fraction": _quantiles([r["crop"]["removed_fraction"] for r in ok]),
                 "ground_method": dict(collections.Counter(r["crop"]["ground_method"] for r in ok))},
        "interior_fraction": _quantiles([r["masks"]["interior_fraction"] for r in ok]),
        "visible_ignored_fraction": _quantiles([r["masks"]["visible_ignored_fraction"]
                                                for r in ok]),
        "below_ground_fraction": _quantiles([r["masks"]["below_ground_fraction"] for r in ok]),
        "storey_histogram": dict(sorted(storeys.items())),
        "storey_confidence": dict(sorted(confidence.items())),
        "targets": {"vocab_version": sorted({t["vocab_version"] for t in targets}),
                    "vocab_digest": sorted({t.get("vocab_digest") for t in targets} - {None}),
                    "supervised_blocks": sum(t["supervised_blocks"] for t in targets),
                    "shell_ignored": sum(t["shell_ignored"] for t in targets)},
    }


# --------------------------------------------------------------------------- previous run

def previous_metrics(norm: Path, rec: dict) -> dict | None:
    """The numbers a later run is compared with, from one record and its npz."""
    from .interior import exposure
    if rec.get("status") != "ok" or not (norm / rec.get("npz", "")).is_file():
        return None
    with np.load(norm / rec["npz"]) as z:
        solid = z["blocks"] != 0
        height = z["ground_height"] if "ground_height" in z.files else int(z["ground_y"])
        audit, _ = exposure(solid, z["shell"], z["exterior"], height)
        shell = int(z["shell"].sum())
    return {"split": rec.get("split"), "removed_fraction": rec["crop"]["removed_fraction"],
            "non_air_after": rec["crop"]["non_air_after"], "shell": shell,
            "interior_fraction": rec["masks"]["interior_fraction"],
            "visible_ignored_fraction": audit["visible_ignored_fraction"],
            "below_ground_fraction": audit["below_ground_fraction"], "flags": rec.get("flags", [])}


def snapshot_previous(norm: Path) -> Path | None:
    """Keep the previous pipeline version's per-build numbers before a new version rebuilds
    them (`previous_run.json`), so a review can sort by what changed."""
    index_path = norm / "index.json"
    if not index_path.is_file():
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("version") == VERSION:
        return None
    builds = {}
    for b in index["builds"]:
        path = norm / "records" / f"{b['id']}.json"
        if path.is_file():
            m = previous_metrics(norm, json.loads(path.read_text(encoding="utf-8")))
            if m:
                builds[b["id"]] = m
    out = norm / "previous_run.json"
    _write_json(out, {"version": index.get("version"), "builds": builds})
    return out


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="build ids to (re)process")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="redo builds that are current")
    ap.add_argument("--no-review", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    dataset = args.dataset.resolve()
    norm = dataset / "normalized"
    if norm.resolve().is_relative_to((dataset / "source").resolve()):
        raise SystemExit("refusing to write inside source/")
    (norm / "records").mkdir(parents=True, exist_ok=True)
    all_rows = json.loads((dataset / "index.json").read_text(encoding="utf-8"))["rows"]
    rows = [r for r in all_rows if r["split"] != "excluded"]
    previous = snapshot_previous(norm)
    if previous:
        print(f"previous version's numbers kept in {previous}")
    # Builds now excluded (D24, tier X): their derived outputs go; the source never changes.
    live = {r["id"] for r in rows}
    for r in all_rows:
        if r["id"] not in live:
            for stale in (norm / "records" / f"{r['id']}.json", norm / f"{r['id']}.npz"):
                if stale.is_file():
                    stale.unlink()
                    print(f"  removed derived output of excluded build: {stale.name}")
    if args.only:
        unknown = set(args.only) - {r["id"] for r in rows}
        if unknown:
            raise SystemExit(f"not in the index (or excluded): {sorted(unknown)}")
    todo = [r for r in rows if (args.only is None or r["id"] in args.only)
            and (args.force or not _is_current(norm, r))]
    print(f"{len(rows)} builds, {len(todo)} to process, {args.workers} worker(s)")

    def handle(rec):
        _write_json(norm / "records" / f"{rec['id']}.json", rec)
        mark = "ok" if rec["status"] == "ok" else f"FAILED {rec['reason']}"
        print(f"  {rec['id']}: {mark}", flush=True)

    if args.workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for rec in pool.map(process, todo, [str(dataset)] * len(todo)):
                handle(rec)
    else:
        for row in todo:
            handle(process(row, str(dataset)))

    records = []
    for r in rows:
        path = norm / "records" / f"{r['id']}.json"
        if not path.is_file():
            records.append({"id": r["id"], "split": r["split"], "licence_tier": r["licence_tier"],
                            "building_category": r["building_category"], "status": "pending"})
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        # Splits and dataset flags can change without the build changing (duplicate moves).
        fresh = {"split": r["split"], "licence_tier": r["licence_tier"],
                 "building_category": r["building_category"]}
        if rec.get("status") == "ok":
            fresh["flags"] = sorted(set(r["flags"]) | set(rec["crop"]["flags"])
                                    | set(rec["masks"]["flags"]))
        if any(rec.get(k) != v for k, v in fresh.items()):
            rec.update(fresh)
            _write_json(path, rec)
        records.append(rec)
    summary = summarize(records)
    _write_json(norm / "index.json", {"dataset": dataset.name, "version": VERSION,
                                      "summary": summary,
                                      "builds": [index_entry(r) for r in records]})

    from .coverage import aggregate
    coverage = aggregate([r for r in records if r.get("status") == "ok"])
    _write_json(dataset / "vocab_coverage.json", coverage)

    if not args.no_review:
        from .gallery import build_gallery
        page = build_gallery(dataset, workers=args.workers)
        print(f"review gallery: {page}")
    print(f"ok {summary['ok']}, failed {summary['failed']}, pending "
          f"{summary['builds'] - summary['ok'] - summary['failed']}")
    print(f"  {coverage['vocab_version']}: representable {coverage['v3']['representable']}% "
          f"(v2 {coverage['v2']['representable']}%)")
    for cls, v in coverage["v3"]["classes"].items():
        print(f"  {cls:8s} {v['share']:7.2f}%  builds {v['builds']}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
