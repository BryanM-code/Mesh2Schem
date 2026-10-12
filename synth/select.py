#!/usr/bin/env python3
"""Choose a run's assets: licence and metadata filters first, then quotas for diversity.

PARKED: see docs/mesh-to-schem-plan.md v4

Usage:  python -m synth.select --config data/synthetic/pilot-v2/run_config.json
                               [--size 200] [--prefer-cached DIR]

Reads the Objaverse index (`synth.objaverse_index`) and writes, into the run directory:

    selection.jsonl         the chosen records, each with its quota bucket
    selection_report.json   what the metadata stage rejected and why (every candidate, every
                            reason code), per-bucket shortfall, licences, download size

Quotas are per bucket so a pilot exercises every kind of building -- towers and skyscrapers
for tall grids and tiling, churches and castles for complex roofs, commercial, civic and
institutional for large flat roofs and glazing -- rather than mirroring the real Minecraft
library, which is ~80% suburban houses. Residential is 54 of 200, suburban houses 6.

Within a bucket the draw prefers objects with a render caption (its category is reliable),
then higher Objaverse++ quality, caps each author at 2 objects, and (optionally) prefers
GLBs already on disk so a re-run does not download them again.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from . import taxonomy as T
from .config import PipelineConfig
from .filters import metadata_verdict

QUOTAS = {
    "house:traditional": 8, "house:modern": 8, "house:suburban": 6, "house:villa": 4,
    "townhouse": 8, "apartment": 20,
    "midrise_generic": 16, "skyscraper": 20,
    "commercial": 16, "office": 8,
    "institutional": 12, "civic": 6,
    "religious": 18, "tower": 16, "fortification": 12, "industrial": 12,
    "pavilion": 4, "building_generic": 6,
}
PER_AUTHOR = 2


def bucket(r):
    if r["category"] == "house":
        return r.get("house_sub") or "house:unspecified"
    if r["category"] == "institutional":
        return "civic" if r.get("subcategory") == "civic" else "institutional"
    return r["category"]


def select(candidates, cfg, size=200, seed=None, cached=frozenset()):
    """(chosen records, report). `candidates` are index records (any stage)."""
    rng = random.Random(cfg.seed if seed is None else seed)
    scale = size / float(sum(QUOTAS.values()))
    quotas = {b: max(1, round(q * scale)) for b, q in QUOTAS.items()}
    reasons = collections.Counter()
    eligible = collections.defaultdict(list)
    for r in candidates:
        why, _ = metadata_verdict(r, cfg)
        if why:
            reasons[why] += 1
            continue
        r["subcategory"] = T.subcategory(r["category"], r.get("caption") or r.get("name"))
        eligible[bucket(r)].append(r)
    chosen, authors, shortfall = [], collections.Counter(), {}
    for b, quota in quotas.items():
        rs = eligible.get(b, [])
        rng.shuffle(rs)
        rs.sort(key=lambda r: (r["uid"] not in cached, not r.get("caption"), -(r.get("opp_score") or 0)))
        picked = []
        for r in rs:
            if len(picked) == quota:
                break
            if authors[r.get("author")] >= PER_AUTHOR:
                continue
            picked.append(r)
            authors[r.get("author")] += 1
        if len(picked) < quota:
            shortfall[b] = quota - len(picked)
        for r in picked:
            r["bucket"] = b
        chosen += picked
    report = {
        "config_hash": cfg.hash(), "candidates": len(candidates),
        "metadata_rejected": dict(reasons.most_common()), "eligible": sum(len(v) for v in eligible.values()),
        "eligible_by_bucket": {b: len(v) for b, v in sorted(eligible.items(), key=lambda kv: -len(kv[1]))},
        "quotas": quotas, "selected": len(chosen), "shortfall": shortfall,
        "selected_by_group": dict(collections.Counter(T.group(r["category"], r.get("subcategory")) for r in chosen)),
        "licences": dict(collections.Counter(r["license"] for r in chosen)),
        "already_on_disk": sum(1 for r in chosen if r["uid"] in cached),
        "download_gb": round(sum((r.get("glb_bytes") or 0) for r in chosen if r["uid"] not in cached) / 1e9, 3),
    }
    return chosen, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", default="data/synthetic/index/objaverse_candidates.jsonl")
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--prefer-cached", help="folder of <uid>.glb already downloaded")
    args = ap.parse_args(argv)
    cfg = PipelineConfig.load(args.config)
    cands = [json.loads(line) for line in open(args.index, encoding="utf-8")]
    cached = frozenset(p.stem for p in Path(args.prefer_cached).glob("*.glb")) if args.prefer_cached else frozenset()
    chosen, report = select(cands, cfg, args.size, cached=cached)
    run = cfg.run_dir
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "selection.jsonl", "w", encoding="utf-8") as fh:
        for r in chosen:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    report["prefer_cached"] = args.prefer_cached
    (run / "selection_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("selected {} of {} eligible ({} candidates) -> {}".format(
        report["selected"], report["eligible"], report["candidates"], run / "selection.jsonl"))
    print("metadata rejections:", report["metadata_rejected"])
    print("by group:", report["selected_by_group"])
    if report["shortfall"]:
        print("shortfall:", report["shortfall"])
    print("already on disk: {}   to download: {} GB".format(report["already_on_disk"], report["download_gb"]))


if __name__ == "__main__":
    main()
