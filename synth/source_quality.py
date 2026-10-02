#!/usr/bin/env python3
"""Do metadata signals predict which sources make good targets? Hand verdicts vs metadata.

Usage:  python -m synth.source_quality [--audit data/synthetic/pilot-v1/visual_audit.json]
                                       [--order data/synthetic/pilot-v1/sheet_order.json]
                                       [--review data/synthetic/pilot-v3/review.json]

Joins per-source verdicts with their index records (data/synthetic/index) and prints, for
each signal a filter could use -- Objaverse++ style and quality, TRELLIS-500K aesthetic score,
low-poly / stylised / game / incomplete vocabulary, face count, texture count, scene flag --
the share of clean, flawed and junk targets on each side, and what excluding it would cost.

Verdicts come from either the pilot-v1 hand audit (`visual_audit.json`: one letter per source
in contact-sheet order, c / f / j) or a `synth.review` file (good / flawed / garbage, which
map to c / f / j). A signal is worth filtering on only if the side it would remove holds far
more junk than clean targets.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

GAME = re.compile(r"\b(game|gameready|game[- ]ready|asset|assets|unity|unreal|ue4|ue5|level|pack|kit|modular|"
                  r"prop|props|fortnite|roblox|sims|gta|pubg|minecraft|hello neighbor|rpg|mmo|mobile)\b", re.I)
LOW_POLY = re.compile(r"low[- ]?poly|lowpoly|\bpolygon\b|\bvoxel\b|isometric|stylized|stylised|cartoon|toon|cute|"
                      r"fantasy|hand[- ]?painted", re.I)
INCOMPLETE = re.compile(r"floor ?plan|\bplan\b|\bwip\b|work in progress|blockout|block[- ]?out|\bsection\b|cutaway|"
                        r"\betasje\b|\b1et\b|maquet|maqueta|interior|\broom\b|homework|\bhw\b|assignment|\bweek \d|"
                        r"exercise|practice|draft|sketch|massing", re.I)
MAP = {"good": "c", "flawed": "f", "garbage": "j"}
# Sketchfab tags that name an object, a set or a scene rather than a building.
OBJECT_TAGS = {"interior", "furniture", "chair", "table", "prop", "props", "diorama", "scene", "room", "vehicle",
               "character", "characters", "toy", "miniature", "lamp", "decor", "decoration"}


def _text(r):
    return " ".join([r.get("name") or "", " ".join(r.get("tags") or []), r.get("caption") or ""])


def load_verdicts(audit=None, order=None, review=None):
    verdict = {}
    if audit:
        aud = json.loads(Path(audit).read_text(encoding="utf-8"))["verdicts"]
        sheet = json.loads(Path(order).read_text(encoding="utf-8"))
        for cat, letters in aud.items():
            for uid, v in zip(sheet[cat], letters):
                verdict[uid] = v
    if review:
        for uid, v in json.loads(Path(review).read_text(encoding="utf-8")).get("verdicts", {}).items():
            if v.get("verdict") in MAP:
                verdict[uid] = MAP[v["verdict"]]
    return verdict


def load_records(uids, index_dir="data/synthetic/index"):
    recs = {}
    for name in ("objaverse_pool.jsonl", "objaverse_candidates.jsonl"):
        path = Path(index_dir) / name
        if not path.exists():
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["uid"] in uids and r["uid"] not in recs:
                recs[r["uid"]] = r
    return recs


SIGNALS = {
    "opp_style": lambda r: r.get("opp_style"),
    "opp_score": lambda r: r.get("opp_score"),
    "aesthetic": lambda r: "none" if r.get("aesthetic") is None else
    ("<4.5" if r["aesthetic"] < 4.5 else "4.5-5.5" if r["aesthetic"] < 5.5 else "5.5-6" if r["aesthetic"] < 6 else ">=6"),
    "low_poly_or_stylised_words": lambda r: bool(LOW_POLY.search(_text(r))),
    "game_words": lambda r: bool(GAME.search(_text(r))),
    "incomplete_words": lambda r: bool(INCOMPLETE.search(_text(r))),
    "index_low_poly_flag": lambda r: bool((r.get("flags") or {}).get("low_poly")),
    "textured": lambda r: (r.get("tex_count") or 0) > 0,
    "faces": lambda r: "a <5k" if (r.get("faces") or 0) < 5000 else "b 5-20k" if r["faces"] < 20000 else
    "c 20-100k" if r["faces"] < 100000 else "d >=100k",
    "opp_scene": lambda r: r.get("opp_scene"),
    "sketchfab_architecture": lambda r: "architecture" in (r.get("sketchfab_cats") or []),
    "sketchfab_furniture_home": lambda r: "furniture-home" in (r.get("sketchfab_cats") or []),
    "sketchfab_not_architecture_only": lambda r: bool(r.get("sketchfab_cats")) and
    "architecture" not in (r.get("sketchfab_cats") or []),
    "object_or_set_tags": lambda r: bool(OBJECT_TAGS & {t.lower() for t in (r.get("tags") or [])}),
    "pool_kind": lambda r: r.get("pool_kind"),
}


def table(verdict, recs):
    out = {}
    for name, key in SIGNALS.items():
        t = collections.defaultdict(collections.Counter)
        for uid, r in recs.items():
            t[str(key(r))][verdict[uid]] += 1
        out[name] = {k: {"n": sum(c.values()), "clean": round(c["c"] / sum(c.values()), 3),
                         "flawed": round(c["f"] / sum(c.values()), 3), "junk": round(c["j"] / sum(c.values()), 3)}
                     for k, c in sorted(t.items())}
    for name in ("low_poly_or_stylised_words", "game_words", "incomplete_words", "sketchfab_furniture_home",
                 "sketchfab_not_architecture_only", "object_or_set_tags"):
        removed = collections.Counter(verdict[u] for u, r in recs.items() if SIGNALS[name](r))
        out[name]["excluding_it_removes"] = {"clean": removed["c"], "flawed": removed["f"], "junk": removed["j"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit", default="data/synthetic/pilot-v1/visual_audit.json")
    ap.add_argument("--order", default="data/synthetic/pilot-v1/sheet_order.json")
    ap.add_argument("--review", help="a synth.review review.json instead of (or besides) the audit")
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args(argv)
    verdict = load_verdicts(None if args.no_audit else args.audit, args.order, args.review)
    recs = load_records(set(verdict))
    verdict = {u: v for u, v in verdict.items() if u in recs}
    print(json.dumps({"sources": len(verdict), "verdicts": dict(collections.Counter(verdict.values())),
                      "signals": table(verdict, recs)}, indent=1))


if __name__ == "__main__":
    main()
