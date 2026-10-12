#!/usr/bin/env python3
"""Index Objaverse 1.0 for buildings, from metadata only.

PARKED: see docs/mesh-to-schem-plan.md v4

Usage:  python -m synth.objaverse_index [--cache data/synthetic/cache/objaverse] [--download]

Reads the four metadata sources (nothing here downloads a mesh):

    metadata/000-*.json.gz      Objaverse 1.0: name, tags, categories, licence, author,
                                face count, GLB size          (ODC-By 1.0 wrapper)
    trellis_sketchfab.csv       TRELLIS-500K: render caption, aesthetic score, sha256 (MIT)
    opp_annotated_800k.json     Objaverse++: quality 0-3, style, scene / multi-object /
                                transparent flags            (ODC-By 1.0)
    object-paths.json.gz        uid -> GLB path in the Hugging Face mirror

and writes, under data/synthetic/index/:

    objaverse_candidates.jsonl  every object with building vocabulary, classified + flagged
    objaverse_pool.jsonl        the subset that passes every rule (`pool_rules.stage`)
    objaverse_summary.json      funnel, per-category licence / quality table
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import gzip
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

from . import taxonomy as T

HF = "https://huggingface.co/datasets/"
SOURCES = {
    "object-paths.json.gz": HF + "allenai/objaverse/resolve/main/object-paths.json.gz",
    "trellis_sketchfab.csv": HF + "JeffreyXiang/TRELLIS-500K/resolve/main/ObjaverseXL_sketchfab.csv",
    "opp_annotated_800k.json": HF + "cindyxl/ObjaversePlusPlus/resolve/main/annotated_800k.json",
}
N_SHARDS = 160

# Rules, in order: the first that applies is the object's rejection reason. Objaverse++'s
# scene / multi-object flags are NOT rejections -- on a hand-checked sample they mostly
# mark one building standing on a plot, which the voxeliser handles -- so they only mark
# the object pool_kind="plot".
HARD = ["object_head", "blocky_source", "toy_or_miniature", "interior", "furnished_interior",
        "cutaway", "part", "multi_building", "scene", "ruin_or_wreck", "not_finished"]
STYLIZED_OPP = ("arcade", "anime", "cartoon", "sci-fi")
PERMISSIVE = ("cc0", "by")


def stage(r):
    """(stage, pool_kind); stage == 'pool' means usable."""
    f = r["flags"]
    if f.get("game_rip"):
        return "game_rip", None
    for k in HARD:
        if f.get(k):
            return k, None
    if r["opp_style"] in STYLIZED_OPP or f.get("stylized"):
        return "stylized", None
    if r["animated"]:
        return "animated", None
    if (r["faces"] or 0) < 500:
        return "too_few_faces", None
    if r["opp_score"] is not None and r["opp_score"] < 2:
        return "low_quality", None
    return "pool", ("plot" if (r["opp_scene"] or r["opp_multi"]) else "clean")


def download(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if not (cache / name).exists():
            print("downloading", name)
            urllib.request.urlretrieve(url, cache / name)
    meta = cache / "metadata"
    meta.mkdir(exist_ok=True)
    for i in range(N_SHARDS):
        f = meta / "000-{:03d}.json.gz".format(i)
        if not f.exists():
            urllib.request.urlretrieve(HF + "allenai/objaverse/resolve/main/metadata/" + f.name, f)


def build(cache: Path, out_dir: Path):
    t0 = time.time()
    csv.field_size_limit(10 ** 9)
    trellis = {}
    for row in csv.DictReader(open(cache / "trellis_sketchfab.csv", encoding="utf-8")):
        uid = row["file_identifier"].rstrip("/").rsplit("/", 1)[-1]
        caps = json.loads(row["captions"]) if row["captions"] else []
        trellis[uid] = (float(row["aesthetic_score"] or 0), caps[0] if caps else "", row["sha256"])
    opp = {x["UID"]: x for x in json.load(open(cache / "opp_annotated_800k.json", encoding="utf-8"))}
    paths = json.load(gzip.open(cache / "object-paths.json.gz"))
    prefilter = re.compile(r"\b(?:" + "|".join(p for _, p in T.CATEGORIES) + r"|buildings?|architecture)\b", re.I)

    out_dir.mkdir(parents=True, exist_ok=True)
    licences_all = collections.Counter()
    funnel = collections.Counter()
    n_all = 0
    pool = []
    with open(out_dir / "objaverse_candidates.jsonl", "w", encoding="utf-8") as cand:
        for shard in sorted(glob.glob(str(cache / "metadata" / "*.json.gz"))):
            for uid, o in json.load(gzip.open(shard)).items():
                n_all += 1
                licences_all[o.get("license") or "unknown"] += 1
                name = o.get("name") or ""
                tags = [t.get("name", "") for t in (o.get("tags") or [])]
                cats = [c.get("name", "") for c in (o.get("categories") or [])]
                tr = trellis.get(uid)
                caption = tr[1] if tr else ""
                if not (prefilter.search(" ".join([caption, name, " ".join(tags)])) or "architecture" in cats):
                    continue
                cat, sub, flags, nst = T.classify(caption, name, tags, o.get("description") or "", cats)
                if cat is None:
                    continue
                glb = (o.get("archives") or {}).get("glb") or {}
                q = opp.get(uid) or {}
                r = {
                    "uid": uid, "name": name[:120], "license": o.get("license") or "unknown",
                    "author": (o.get("user") or {}).get("username"), "url": o.get("viewerUrl"),
                    "path": paths.get(uid), "category": cat, "house_sub": sub, "storeys": nst,
                    "sketchfab_cats": cats, "tags": tags[:20], "caption": caption[:500],
                    "in_trellis500k": tr is not None, "aesthetic": tr[0] if tr else None,
                    "sha256": tr[2] if tr else None,
                    "faces": o.get("faceCount"), "glb_bytes": glb.get("size"),
                    "tex_count": glb.get("textureCount"), "tex_max_res": glb.get("textureMaxResolution"),
                    "animated": (o.get("animationCount") or 0) > 0,
                    "opp_score": q.get("score"), "opp_style": q.get("style"),
                    "opp_multi": q.get("is_multi_object") == "true", "opp_scene": q.get("is_scene") == "true",
                    "opp_transparent": q.get("is_transparent") == "true",
                    "flags": flags,
                }
                s, kind = stage(r)
                r["stage"], r["pool_kind"] = s, kind
                funnel[s] += 1
                cand.write(json.dumps(r, ensure_ascii=False) + "\n")
                if s == "pool":
                    pool.append(r)
    with open(out_dir / "objaverse_pool.jsonl", "w", encoding="utf-8") as fh:
        for r in pool:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    table = {}
    for c in sorted({r["category"] for r in pool}):
        rs = [r for r in pool if r["category"] == c]
        lc = collections.Counter(r["license"] for r in rs)
        table[c] = {
            "pool": len(rs), "permissive": lc["cc0"] + lc["by"], "by_sa": lc["by-sa"],
            "nc": lc["by-nc"] + lc["by-nc-sa"],
            "permissive_captioned": sum(1 for r in rs if r["caption"] and r["license"] in PERMISSIVE),
            "plot_share": round(sum(r["pool_kind"] == "plot" for r in rs) / len(rs), 3),
            "scan_share": round(sum(1 for r in rs if r["flags"].get("scan") or r["opp_style"] == "scanned") / len(rs), 3),
            "opp3_share": round(sum(1 for r in rs if r["opp_score"] == 3) / len(rs), 3),
            "faces_p50": statistics.median([r["faces"] or 0 for r in rs]),
            "glb_mb_p50": round(statistics.median([(r["glb_bytes"] or 0) / 1e6 for r in rs]), 1),
            "authors": len({r["author"] for r in rs}),
        }
    summary = {"objects": n_all, "licences_all": licences_all.most_common(), "candidates": sum(funnel.values()),
               "funnel": funnel.most_common(), "pool": len(pool),
               "pool_licences": collections.Counter(r["license"] for r in pool).most_common(),
               "top_authors": collections.Counter(r["author"] for r in pool).most_common(10),
               "categories": table, "seconds": round(time.time() - t0, 1)}
    (out_dir / "objaverse_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default="data/synthetic/cache/objaverse")
    ap.add_argument("--out", default="data/synthetic/index")
    ap.add_argument("--download", action="store_true", help="fetch any missing metadata first")
    args = ap.parse_args(argv)
    cache = Path(args.cache)
    if args.download:
        download(cache)
    s = build(cache, Path(args.out))
    print("objects {:,}  candidates {:,}  pool {:,}  ({}s)".format(s["objects"], s["candidates"], s["pool"], s["seconds"]))
    for c, row in sorted(s["categories"].items(), key=lambda kv: -kv[1]["pool"]):
        print("  {:17s} pool {:5d}  CC0+BY {:5d}  captioned {:4d}".format(c, row["pool"], row["permissive"],
                                                                        row["permissive_captioned"]))


if __name__ == "__main__":
    sys.exit(main())
