#!/usr/bin/env python3
"""The held-out real Minecraft evaluation set: which builds, why, and a guard for training.

Usage:  python -m synth.real_eval [--out eval/real-eval-v1.json] [--target 64]

The real library answers the question the synthetic pairs cannot: "does the trained system
build the way Minecraft builders build?" So its evaluation builds must never be trained on,
and must not be 64 copies of one suburban house. The selection:

* **One build per family.** Builds from one author, pack or series (the ten "1-chunk
  suburban houses", four "paris-house" variants, a re-uploaded duplicate) would score as
  near-duplicates; `data/synthetic/real_library_families.json` groups them.
* **The most trustworthy member of each family**: block states intact (40 builds lost every
  stair's orientation on import -- they can be scored on shape and material, not
  orientation), no analyser flags (terrain taper, floating gap), then the larger build.
* **Every family**, up to `--target`: diversity before quantity. With 64 families the set is
  every family once.

It records, per build: category, whether orientation can be scored, the collector's status
(no build has its AI photo or TRELLIS mesh yet, so none is a scoreable pair today), and the
coverage gap by category -- the library has no apartment, office, tower, skyscraper, civic,
religious or industrial build.

`eval_build_ids()` / `assert_disjoint()` are the guard: any training loader that could touch
real builds must call `assert_disjoint(ids)` first. The synthetic pipeline never reads real
builds as training data; `synth.roof_stats` reads them only to measure construction style,
and `--exclude-eval` measures it on the non-eval builds only.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

DEFAULT_OUT = Path("eval/real-eval-v1.json")
FAMILIES = Path("data/synthetic/real_library_families.json")


def eval_build_ids(path=DEFAULT_OUT) -> set:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(b["build_id"]) for b in data["builds"]}


def assert_disjoint(training_build_ids, path=DEFAULT_OUT):
    """Raise if any training build is a held-out evaluation build."""
    overlap = eval_build_ids(path) & {int(i) for i in training_build_ids}
    if overlap:
        raise ValueError("evaluation builds in training data: {}".format(sorted(overlap)))


def describe(build_dir: Path):
    """Facts about one real build used to pick and label it."""
    import house_stats
    import schem_voxels
    from .roof_stats import stateless_stairs

    meta = json.loads((build_dir / "meta.json").read_text(encoding="utf-8"))
    vg = schem_voxels.load_voxels(build_dir / "schem" / "original.schem")
    row = {"build_id": int(build_dir.name.split("__")[0]), "slug": build_dir.name.split("__", 1)[1],
           "name": meta.get("name"), "status": "needs_ai_photo" if not meta.get("image_at") else
           ("needs_trellis" if not meta.get("paired_at") else "complete"),
           "source_url": meta.get("source_url") or None}
    if vg is None:
        row["readable"] = False
        return row
    seg = house_stats.segment(vg)
    struct = seg.structure
    row.update(readable=True, orientation_reliable=not stateless_stairs(vg), analyser_flags=list(seg.flags),
               structure_voxels=int(struct.sum()) if struct is not None else 0)
    if struct is not None and struct.any():
        ys = struct.any(axis=(1, 2)).nonzero()[0]
        zs = struct.any(axis=(0, 2)).nonzero()[0]
        xs = struct.any(axis=(0, 1)).nonzero()[0]
        row["dims_xyz"] = [int(xs[-1] - xs[0] + 1), int(ys[-1] - ys[0] + 1), int(zs[-1] - zs[0] + 1)]
    return row


def build(target=64, builds_root=Path("data/builds"), families_path=FAMILIES):
    fam_data = json.loads(Path(families_path).read_text(encoding="utf-8"))
    families = fam_data["families"]
    categories = fam_data.get("categories", {})
    dirs = {int(d.name.split("__")[0]): d for d in builds_root.iterdir() if (d / "meta.json").exists()}
    chosen = []
    for fam, members in sorted(families.items()):
        rows = [describe(dirs[i]) for i in members if i in dirs]
        rows = [r for r in rows if r.get("readable")]
        if not rows:
            continue
        rows.sort(key=lambda r: (not r["orientation_reliable"], len(r["analyser_flags"]), -r["structure_voxels"],
                                 r["build_id"]))
        best = dict(rows[0])
        best["family"] = fam
        best["family_size"] = len(members)
        best["category"] = categories.get(str(best["build_id"]), "unlabelled")
        chosen.append(best)
    # Diversity first: every non-detached category before detached houses, then the rest.
    chosen.sort(key=lambda r: (r["category"] == "detached house", not r["orientation_reliable"], r["build_id"]))
    chosen = chosen[:target]
    chosen.sort(key=lambda r: r["build_id"])
    cats = collections.Counter(r["category"] for r in chosen)
    wanted = ["detached house", "urban townhouse/row", "rustic/medieval/cabin", "commercial", "apartment", "office",
              "tower", "skyscraper", "civic/institutional", "religious", "industrial", "fortification"]
    return {
        "version": "real-eval-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": ("Held out: never trained on. One build per family (author / pack / series), the most "
                   "trustworthy member (block states intact, no analyser flags, larger). Categories from "
                   "the hand-labelled library survey (real_library_families.json)."),
        "builds": chosen,
        "summary": {
            "builds": len(chosen), "families_available": len(families),
            "by_category": dict(cats.most_common()),
            "orientation_reliable": sum(1 for r in chosen if r["orientation_reliable"]),
            "orientation_lost_on_import": sum(1 for r in chosen if not r["orientation_reliable"]),
            "with_analyser_flags": sum(1 for r in chosen if r["analyser_flags"]),
            "scoreable_pairs_today": sum(1 for r in chosen if r["status"] == "complete"),
            "missing_categories": [c for c in wanted if c not in cats],
            "detached_house_share": round(cats.get("detached house", 0) / max(len(chosen), 1), 3),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--target", type=int, default=64)
    args = ap.parse_args(argv)
    data = build(args.target)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    s = data["summary"]
    print("{} builds from {} families -> {}".format(s["builds"], s["families_available"], out))
    print("by category:", s["by_category"])
    print("orientation reliable {}, lost on import {}; scoreable pairs today {}".format(
        s["orientation_reliable"], s["orientation_lost_on_import"], s["scoreable_pairs_today"]))
    print("missing categories:", s["missing_categories"])


if __name__ == "__main__":
    main()
