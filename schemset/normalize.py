#!/usr/bin/env python3
"""Phase 3: read, crop, mask and measure every build of the frozen schematic dataset.

Usage:  python -m schemset.normalize --dataset data/real-schems-v1
                                     [--only <id> ...] [--workers N] [--force] [--no-review]

For every index row except split "excluded" (tier X):

    read      `schemset.read`: any format -> palette grid, validated against index.json
    crop      `schemset.crop`: ground, landscape and site removed, padded with air
    masks     `schemset.interior`: shell / exterior / interior (ignored, not deleted)
    storey    `schemset.storey`: blocks per storey + confidence (information only, D15)
    targets   `mcvoxel.vocab.encode_target`: (shape, material) training targets in the
              current vocabulary -- exterior air, shell blocks the vocabulary holds, and
              IGNORE for the interior, decor, unsupported blocks and door upper halves (D20)
    coverage  supervised (shell) block counts per blockstate, for `schemset.coverage`

Writes, all under `<dataset>/` and never inside `source/`:

    normalized/<id>.npz           blocks (uint16 X, Y, Z), palette, shell, exterior, interior
                                  (bool), crop_offset (source-grid coordinates of [0,0,0]),
                                  ground_y (this grid's ground level; at or below is ground),
                                  target_shape (uint8) / target_material (uint16) in the
                                  vocabulary named by the record's `vocab_version`
    normalized/records/<id>.json  everything measured for one build
    normalized/index.json         per build: status (ok / failed + reason), validation, crop
                                  stats, flags, interior fraction, storey; plus a summary
    vocab_coverage.json           `schemset.coverage` over every ok build
    review/                       before/after crop contact sheets and interior cutaways

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
import random
import sys
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

VERSION = "normalize-2"             # 2: iron_chain rename in the reader; v3 targets in the npz
SEED = 20261011
REVIEW_RANDOM = 20
REVIEW_INTERIOR = 10
SHEET_BUILDS = 8


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
    from .crop import crop_build
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
        rec["crop"] = {**crop.stats, "offset": list(crop.offset), "flags": crop.flags}
        if "crop_empty" in crop.flags:
            return {**rec, "status": "failed", "reason": "crop: no block survived the crop"}
        g = crop.stats["ground_level"]
        masks = compute_masks(crop.blocks, None if g is None else g - crop.offset[1])
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
                                           "removed_built_above_ground", "large_masses",
                                           "offset") if k in c}
    if "masks" in rec:
        entry["interior_fraction"] = rec["masks"]["interior_fraction"]
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
        "storey_histogram": dict(sorted(storeys.items())),
        "storey_confidence": dict(sorted(confidence.items())),
        "targets": {"vocab_version": sorted({t["vocab_version"] for t in targets}),
                    "vocab_digest": sorted({t.get("vocab_digest") for t in targets} - {None}),
                    "supervised_blocks": sum(t["supervised_blocks"] for t in targets),
                    "shell_ignored": sum(t["shell_ignored"] for t in targets)},
    }


# --------------------------------------------------------------------------- review

def write_review(dataset: Path, records: list[dict]) -> dict:
    from .read import load_build
    from . import review

    ok = {r["id"]: r for r in records if r.get("status") == "ok"}
    groups = {name: sorted(i for i, r in ok.items() if name in r["flags"])
              for name in ("oversized_capture", "multi_building", "crop_aggressive")}
    flagged = set().union(*groups.values())
    others = sorted(set(ok) - flagged)
    groups["random"] = sorted(random.Random(SEED).sample(others, min(REVIEW_RANDOM, len(others))))

    cache = {}

    def panels(bid):
        if bid not in cache:
            rec = ok[bid]
            build = load_build(dataset / "source" / rec["file"])
            with np.load(dataset / "normalized" / rec["npz"]) as z:
                after = review.render(z["blocks"], list(z["palette"]))
            cache[bid] = [review.render(build.blocks, build.palette), after]
        return cache[bid]

    def caption(bid):
        r = ok[bid]
        c = r["crop"]
        return (f"{bid[:60]}\nremoved {100 * c['removed_fraction']:.0f}%  ground "
                f"{c['ground_level']} ({c['ground_method']})  masses {c['masses'][:3]}\n"
                + " ".join(f for f in r["flags"] if f not in ("legacy_format",)))

    out_dir = dataset / "review"
    old = out_dir / "sheets.json"                 # this step's own previous output only
    if old.is_file():
        for name in json.loads(old.read_text(encoding="utf-8")):
            (out_dir / name).unlink(missing_ok=True)
    sheets = {}
    for name, ids in groups.items():
        prefix = name if name.startswith("crop_") else f"crop_{name}"
        for k in range(0, len(ids), SHEET_BUILDS):
            chunk = ids[k:k + SHEET_BUILDS]
            path = out_dir / f"{prefix}_{k // SHEET_BUILDS + 1:02d}.png"
            review.contact_sheet([{"caption": caption(i), "panels": panels(i)} for i in chunk],
                                 path)
            sheets[path.name] = chunk

    suspects = sorted(i for i, r in ok.items() if "interior_leak_suspect" in r["flags"])
    pick = suspects[:REVIEW_INTERIOR // 2]
    rest = sorted(set(ok) - set(pick))
    pick += random.Random(SEED + 1).sample(rest, min(REVIEW_INTERIOR - len(pick), len(rest)))
    items = []
    for bid in sorted(pick):
        with np.load(dataset / "normalized" / ok[bid]["npz"]) as z:
            blocks, palette = z["blocks"], list(z["palette"])
            interior, gy = z["interior"], int(z["ground_y"])
        ys = np.flatnonzero((blocks != 0).any(axis=(0, 2)))
        lo, hi = max(int(ys[0]), gy + 1), int(ys[-1])
        cuts = [lo + max(2, (hi - lo) // 5), lo + (hi - lo) // 2]
        m = ok[bid]["masks"]
        items.append({"caption": f"{bid[:60]}\ninterior {m['interior_fraction']:.2f}  "
                                 f"deep roofed exterior {m['deep_covered_exterior']}  cuts {cuts}\n"
                                 + " ".join(m["flags"]),
                      "panels": [review.render(blocks, palette)]
                      + [review.interior_view(blocks, palette, interior, c, gy) for c in cuts]})
    for k in range(0, len(items), 5):
        path = out_dir / f"interior_{k // 5 + 1:02d}.png"
        review.contact_sheet(items[k:k + 5], path, columns=1)
        sheets[path.name] = sorted(pick)[k:k + 5]
    _write_json(out_dir / "sheets.json", sheets)
    return sheets


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
    rows = [r for r in json.loads((dataset / "index.json").read_text(encoding="utf-8"))["rows"]
            if r["split"] != "excluded"]
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
        records.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file()
                       else {"id": r["id"], "split": r["split"], "licence_tier": r["licence_tier"],
                             "building_category": r["building_category"], "status": "pending"})
    summary = summarize(records)
    _write_json(norm / "index.json", {"dataset": dataset.name, "version": VERSION,
                                      "summary": summary,
                                      "builds": [index_entry(r) for r in records]})

    from .coverage import aggregate
    coverage = aggregate([r for r in records if r.get("status") == "ok"])
    _write_json(dataset / "vocab_coverage.json", coverage)

    if not args.no_review:
        sheets = write_review(dataset, records)
        print(f"review: {len(sheets)} sheets in {dataset / 'review'}")
    print(f"ok {summary['ok']}, failed {summary['failed']}, pending "
          f"{summary['builds'] - summary['ok'] - summary['failed']}")
    print(f"  {coverage['vocab_version']}: representable {coverage['v3']['representable']}% "
          f"(v2 {coverage['v2']['representable']}%)")
    for cls, v in coverage["v3"]["classes"].items():
        print(f"  {cls:8s} {v['share']:7.2f}%  builds {v['builds']}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
