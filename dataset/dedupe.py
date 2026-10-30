#!/usr/bin/env python3
"""Geometric duplicate check: the same build twice, after the crop.

Usage:  python -m dataset.dedupe --dataset data/real-schems-v1

Every pair of normalized builds (split other than "excluded") is compared on its kept
blocks under the 8 rotations and mirrors of the horizontal plane and offsets of up to
`OFFSET` blocks on each axis. A pair is a duplicate when

    occupancy IoU > IOU  and  palette similarity >= PALETTE

(palette similarity: weighted Jaccard over block names, properties dropped). Cheap filters
run first: height and both horizontal sizes within `DIMS_TOL` blocks (rotations allowed)
and non-air counts within `COUNT_TOL`; pairs passing them but not the duplicate test are
listed as near misses when the best IoU reaches `NEAR_IOU`.

Writes `<dataset>/geometric_duplicates.json`. `dataset.freeze.derive` reads it: a duplicate
pair split across train / val / test is put into one split by moving the smaller creator
group whole, so creator grouping stays intact (re-run `python -m dataset.freeze --out
<dataset>`, then normalize, which picks the new splits up). No build is deleted.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np

IOU = 0.9
PALETTE = 0.7
NEAR_IOU = 0.75
OFFSET = 2
DIMS_TOL = 4
COUNT_TOL = 0.15


def _tight(occ: np.ndarray) -> np.ndarray:
    idx = [np.flatnonzero(occ.any(axis=tuple(a for a in range(3) if a != ax))) for ax in range(3)]
    return occ[idx[0][0]:idx[0][-1] + 1, idx[1][0]:idx[1][-1] + 1, idx[2][0]:idx[2][-1] + 1]


def transforms(occ: np.ndarray):
    """The 8 rotations / mirrors about the vertical axis: (name, grid)."""
    for mirror in (False, True):
        g = occ[::-1] if mirror else occ
        for k in range(4):
            yield f"{'mirror+' if mirror else ''}rot{90 * k}", np.rot90(g, k, axes=(0, 2))


def best_iou(a: np.ndarray, b: np.ndarray, offset: int = OFFSET) -> tuple[float, str, tuple]:
    """Max occupancy IoU of tight grids a and b over the 8 transforms of b and shifts of up
    to `offset` blocks (b's corner against a's)."""
    na = int(a.sum())
    best = (0.0, "", (0, 0, 0))
    for name, t in transforms(b):
        if any(abs(sa - sb) > 2 * offset + DIMS_TOL for sa, sb in zip(a.shape, t.shape)):
            continue
        nb = int(t.sum())
        for d in itertools.product(range(-offset, offset + 1), repeat=3):
            sa = tuple(slice(max(0, k), min(a.shape[i], t.shape[i] + k)) for i, k in enumerate(d))
            sb = tuple(slice(max(0, -k), max(0, -k) + (s.stop - s.start)) for k, s in zip(d, sa))
            if any(s.stop <= s.start for s in sa):
                continue
            inter = int(np.count_nonzero(a[sa] & t[sb]))
            iou = inter / (na + nb - inter) if na + nb - inter else 0.0
            if iou > best[0]:
                best = (iou, name, d)
    return best


def palette_similarity(ca: dict, cb: dict) -> float:
    keys = set(ca) | set(cb)
    hi = sum(max(ca.get(k, 0), cb.get(k, 0)) for k in keys)
    return sum(min(ca.get(k, 0), cb.get(k, 0)) for k in keys) / hi if hi else 0.0


def load(dataset: Path) -> dict:
    norm = dataset / "normalized"
    index = json.loads((norm / "index.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in json.loads((dataset / "index.json").read_text(encoding="utf-8"))["rows"]}
    out = {}
    for b in index["builds"]:
        if b["status"] != "ok" or rows[b["id"]]["split"] == "excluded":
            continue
        with np.load(norm / b["npz"]) as z:
            blocks, palette = z["blocks"], [str(s) for s in z["palette"]]
        occ = blocks != 0
        names = [s.split("[", 1)[0] for s in palette]
        counts = collections.Counter()
        for i, n in enumerate(np.bincount(blocks.ravel(), minlength=len(palette))):
            if i and n:
                counts[names[i]] += int(n)
        out[b["id"]] = {"occ": _tight(occ), "count": int(occ.sum()), "palette": counts,
                        "row": rows[b["id"]]}
    return out


def _dims_close(sa, sb) -> bool:
    if abs(sa[1] - sb[1]) > DIMS_TOL:
        return False
    ha, hb = sorted((sa[0], sa[2])), sorted((sb[0], sb[2]))
    return all(abs(x - y) <= DIMS_TOL for x, y in zip(ha, hb))


def find(builds: dict) -> dict:
    ids = sorted(builds)
    checked, pairs, near = 0, [], []
    for a, b in itertools.combinations(ids, 2):
        A, B = builds[a], builds[b]
        if not _dims_close(A["occ"].shape, B["occ"].shape):
            continue
        if abs(A["count"] - B["count"]) > COUNT_TOL * max(A["count"], B["count"]):
            continue
        sim = palette_similarity(A["palette"], B["palette"])
        checked += 1
        iou, name, offset = best_iou(A["occ"], B["occ"])
        entry = {"ids": [a, b], "iou": round(iou, 4), "transform": name,
                 "offset": list(offset), "palette_similarity": round(sim, 4),
                 "splits": [A["row"]["split"], B["row"]["split"]],
                 "creators": [A["row"]["creator"], B["row"]["creator"]],
                 "creator_groups": [A["row"].get("creator_group"), B["row"].get("creator_group")]}
        if iou > IOU and sim >= PALETTE:
            pairs.append({**entry, "verdict": "duplicate"})
        elif iou >= NEAR_IOU:
            near.append({**entry, "verdict": "near miss: below the duplicate thresholds"})
    return {"checked": checked, "pairs": pairs, "near_misses": near}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    builds = load(args.dataset)
    result = find(builds)
    out = {
        "method": {
            "data": "kept blocks of every normalized build (after the crop), split != excluded",
            "duplicate": f"occupancy IoU > {IOU} under the 8 rotations / mirrors about Y and "
                         f"offsets up to {OFFSET} blocks per axis, and palette similarity "
                         f">= {PALETTE} (weighted Jaccard over block names)",
            "prefilter": f"height and sorted horizontal sizes within {DIMS_TOL} blocks, "
                         f"non-air within {COUNT_TOL:.0%}",
            "near_misses": f"prefiltered pairs with IoU >= {NEAR_IOU} that are not duplicates",
            "splits": "dataset.freeze.derive moves the smaller creator group of a duplicate "
                      "pair that straddles splits into the other's split",
        },
        "builds": len(builds), "pairs_checked": result["checked"],
        "pairs": result["pairs"], "near_misses": result["near_misses"],
    }
    path = args.dataset / "geometric_duplicates.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")
    cross = [p for p in result["pairs"] if p["splits"][0] != p["splits"][1]]
    print(f"{len(builds)} builds, {result['checked']} pairs past the filters, "
          f"{len(result['pairs'])} duplicate pair(s) ({len(cross)} across splits), "
          f"{len(result['near_misses'])} near miss(es) -> {path}")
    for p in result["pairs"]:
        print(f"  {p['ids'][0]}  ~  {p['ids'][1]}  IoU {p['iou']}  palette "
              f"{p['palette_similarity']}  {p['transform']} {p['offset']}  splits {p['splits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
