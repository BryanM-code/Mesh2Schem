#!/usr/bin/env python3
"""Vocabulary coverage: how much of the supervised data the model vocabulary can express.

Usage:  python -m dataset.coverage --dataset data/real-schems-v1

Reads the per-build counts that `dataset.normalize` stored (supervised blocks only: the
shell -- exterior air is supervised too but always representable, so it is reported
separately and kept out of the percentages) and writes `vocab_coverage.json`, measuring the
current vocabulary (mcvox-v3, `voxel.vocab.classify_state`) against v2 side by side.

Classes (the training-target policy):

    exact     material and shape both predicted; dropped properties are derived on the
              assembled grid (stair corners, connections, a door's hinge, in_wall) or have no
              visible effect (waterlogged, powered, leaf distance, mushroom faces)
    lossy     representable, but a visible property is dropped; `reasons` name it:
              door_open, gate_open, trapdoor_facing (closed), trapdoor_half (open),
              facing (glazed terracotta, beehive), lit (lamps, bulbs), double_slab (-> the
              full block), alias (waxed copper, infested stone, petrified oak slab)
    derived   rebuilt at assembly, so IGNORE in targets: a door's upper half
    ignore    outside the vocabulary -> IGNORE in targets (never air): decor, fluids, snow
              layers, and the few structural blocks v3 leaves out

v2 classes are recomputed with v2's own rules (frozen below) so the comparison is exact:
exact, lossy, and "ignore" for everything v2 could not express (its missing-material,
missing-shape and decor classes).

Families (`family_of`) group blocks the same way for both versions -- stairs, upside-down
stairs, slabs, full cubes, sideways full cubes, panes, doors, trapdoors, fences, walls, fence
gates, carpets, leaves, and the decor kinds -- so a table can show what each version does
with the same blocks.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from minecraft import block_roles
from voxel import block_shapes as bs
from voxel import vocab

from .read import AIR, split_state

V3_CLASSES = ("exact", "lossy", "derived", "ignore")
V2_CLASSES = ("exact", "lossy", "ignore")
REPRESENTABLE = ("exact", "lossy", "derived")

# ── Decor and structural families (version-independent) ───────────────────────
_PLANTS = (set(block_roles.GROUND_COVER) - {"moss_carpet"}) | {   # moss carpet is a carpet
    "flower_pot", "brown_mushroom", "red_mushroom", "cactus", "bamboo", "sugar_cane",
    "sweet_berry_bush", "kelp", "kelp_plant", "seagrass", "tall_seagrass", "lily_pad",
    "dead_bush", "spore_blossom", "pitcher_plant", "pitcher_crop", "torchflower",
    "torchflower_crop", "cocoa", "chorus_flower", "chorus_plant", "nether_wart",
    "frogspawn", "turtle_egg", "sea_pickle", "leaf_litter", "bush", "firefly_bush",
    "wildflowers", "cactus_flower", "short_dry_grass", "tall_dry_grass", "open_eyeblossom",
    "closed_eyeblossom", "resin_clump", "crimson_roots", "warped_roots", "crimson_fungus",
    "warped_fungus", "twisting_vines", "twisting_vines_plant", "weeping_vines",
    "weeping_vines_plant", "cave_vines", "cave_vines_plant", "hanging_roots", "glow_lichen",
    "vine", "big_dripleaf", "big_dripleaf_stem", "small_dripleaf", "pumpkin",
    "carved_pumpkin", "jack_o_lantern", "melon", "wheat", "carrots", "potatoes", "beetroots",
    "melon_stem", "pumpkin_stem", "attached_melon_stem", "attached_pumpkin_stem",
    "mangrove_propagule", "pink_petals", "azalea", "flowering_azalea",
}
_FURNITURE = {
    "chest", "trapped_chest", "ender_chest", "barrel", "shulker_box", "crafting_table",
    "furnace", "blast_furnace", "smoker", "cartography_table", "fletching_table",
    "smithing_table", "loom", "stonecutter", "grindstone", "anvil", "chipped_anvil",
    "damaged_anvil", "enchanting_table", "brewing_stand", "cauldron", "water_cauldron",
    "lava_cauldron", "powder_snow_cauldron", "composter", "lectern", "jukebox", "note_block",
    "bell", "campfire", "soul_campfire", "decorated_pot", "chiseled_bookshelf", "crafter",
    "beacon", "conduit", "spawner", "trial_spawner", "vault", "end_portal_frame",
    "end_portal", "dragon_egg", "cake", "candle_cake", "respawn_anchor",
}
_REDSTONE = {
    "rail", "powered_rail", "detector_rail", "activator_rail", "redstone_wire", "repeater",
    "comparator", "lever", "tripwire", "tripwire_hook", "daylight_detector", "observer",
    "piston", "sticky_piston", "piston_head", "moving_piston", "dispenser", "dropper",
    "hopper", "target", "sculk_sensor", "calibrated_sculk_sensor",
}
_MISC = {"cobweb", "fire", "soul_fire", "pointed_dripstone", "amethyst_cluster",
         "small_amethyst_bud", "medium_amethyst_bud", "large_amethyst_bud", "structure_block",
         "jigsaw", "command_block", "chain_command_block", "repeating_command_block"}
_LIGHTING = {"torch", "wall_torch", "soul_torch", "soul_wall_torch", "redstone_torch",
             "redstone_wall_torch", "lantern", "soul_lantern", "end_rod", "candle",
             "copper_torch", "copper_wall_torch"}

# Decor kinds, checked in order: (family, test on the bare base name). The vocabulary
# leaves all of them out; they are IGNORE in supervised regions.
DECOR_FAMILIES = [
    ("plants", lambda b: b in _PLANTS or b.startswith("potted_") or b.endswith("_sapling")
     or ("coral" in b and not b.endswith("_block"))),
    ("fluids", lambda b: b in ("water", "lava", "bubble_column")),
    ("furniture", lambda b: b in _FURNITURE or b.endswith(("_shulker_box", "_shelf",
                                                           "_golem_statue", "_chest"))),
    ("beds", lambda b: b.endswith("_bed")),
    ("redstone_rails", lambda b: b in _REDSTONE),
    ("heads", lambda b: b.endswith(("_head", "_skull"))),
    ("lighting", lambda b: b in _LIGHTING or b.endswith("_candle")
     or (b.endswith("_lantern") and b not in ("jack_o_lantern", "sea_lantern"))),
    ("buttons_plates", lambda b: b.endswith(("_button", "_pressure_plate"))),
    ("signs_banners", lambda b: b.endswith(("_sign", "_banner"))),
    ("ladders_chains_scaffolding", lambda b: b in ("ladder", "scaffolding", "iron_chain", "chain")
     or b.endswith("_chain")),
    ("rods", lambda b: b.endswith("lightning_rod")),
    ("snow_layers", lambda b: b == "snow"),
    ("misc", lambda b: b in _MISC),
]


def _bare(name: str) -> str:
    return name.split(":", 1)[1] if ":" in name else name


def decor_family(base: str) -> str | None:
    for family, test in DECOR_FAMILIES:
        if test(base):
            return family
    return None


def family_of(state: str) -> str:
    """The structural family a block belongs to, independent of any vocabulary."""
    name, props = split_state(state)
    b = _bare(name)
    for prefix in ("waxed_", "infested_"):
        if b.startswith(prefix):
            b = b[len(prefix):]
    deco = decor_family(b)
    if deco:
        return "decor:" + deco
    if b.endswith("_door"):
        return "door"
    if b.endswith("_trapdoor"):
        return "trapdoor"
    if b.endswith("_fence_gate"):
        return "fence_gate"
    if b.endswith(("_pane", "_bars")):
        return "pane"
    if b.endswith("_fence"):
        return "fence"
    if b.endswith("_wall"):
        return "wall"
    if b.endswith("_carpet"):
        return "carpet"
    if b.endswith("_stairs"):
        return "stairs_upside_down" if props.get("half") == "top" else "stairs"
    if b.endswith("_slab"):
        return "slab_double" if props.get("type") == "double" else "slab"
    if b.endswith("_leaves"):
        return "leaves"
    if props.get("axis") in ("x", "z"):
        return "full_sideways"
    if block_roles.is_cube(b) or b in ("sea_lantern",):
        return "full"
    return "other"


# ── v3 (the current vocabulary) ───────────────────────────────────────────────
def classify(state: str) -> dict:
    """v3: {"class", "reasons", "family"} for one blockstate."""
    if state == AIR:
        return {"class": "air", "reasons": [], "family": "air"}
    c = vocab.classify_state(state)
    return {"class": c["status"], "reasons": c["reasons"], "family": family_of(state),
            "shape": c["shape"], "material": c["material"]}


# ── v2, frozen: its own material set and variant table ────────────────────────
_V2_MATERIALS = set(vocab.MATERIALS[:vocab.V2_MATERIAL_COUNT])
_V2_VARIANT = {}
for _base, (_slab, _stairs) in bs.BLOCK_VARIANTS.items():
    if _slab:
        _V2_VARIANT[_slab] = (_base, "slab")
    if _stairs:
        _V2_VARIANT[_stairs] = (_base, "stairs")
for _fam, _kind in ((bs.FENCE_VARIANTS, "fence"), (bs.WALL_VARIANTS, "wall")):
    _V2_VARIANT.update({v: (k, _kind) for k, v in _fam.items()})


def classify_v2(state: str) -> str:
    """v2's class for one blockstate: exact, lossy or ignore (missing or decor in v2)."""
    if state == AIR:
        return "air"
    name, props = split_state(state)
    b = _bare(name)
    if name in _V2_VARIANT:
        kind = _V2_VARIANT[name][1]
        return "lossy" if kind == "stairs" and props.get("half") == "top" else "exact"
    if name in _V2_MATERIALS:
        return "lossy" if props.get("axis", "y") != "y" else "exact"
    if b.startswith("waxed_") and ("minecraft:" + b[6:] in _V2_MATERIALS
                                   or "minecraft:" + b[6:] in _V2_VARIANT):
        return "lossy"
    if b.startswith("infested_") and "minecraft:" + b[9:] in _V2_MATERIALS:
        return "lossy"
    if b == "petrified_oak_slab":
        return "lossy"
    return "ignore"


# ── Aggregation ───────────────────────────────────────────────────────────────
def _share(n: int, total: int) -> float:
    return round(100.0 * n / total, 3) if total else 0.0


def _shares(counter: collections.Counter, classes, total) -> dict:
    return {c: _share(counter[c], total) for c in classes}


def aggregate(records: list[dict]) -> dict:
    """records: normalized per-build records with status ok and a `supervised` count map."""
    cache: dict[str, tuple] = {}

    def info(state):
        if state not in cache:
            c = classify(state)
            cache[state] = (c, classify_v2(state))
        return cache[state]

    total, exterior_air = 0, 0
    v3 = collections.Counter()
    v2 = collections.Counter()
    v3_builds = collections.defaultdict(set)
    v2_builds = collections.defaultdict(set)
    reasons = collections.Counter()
    reason_builds = collections.defaultdict(set)
    fam_total = collections.Counter()
    fam_v3 = collections.defaultdict(collections.Counter)
    fam_v2 = collections.defaultdict(collections.Counter)
    split_total = collections.Counter()
    split_v3 = collections.defaultdict(collections.Counter)
    split_v2 = collections.defaultdict(collections.Counter)
    ignored = collections.Counter()
    ignored_builds = collections.defaultdict(set)
    ignored_family = {}
    materials_used = collections.Counter()
    per_build = {}

    for rec in records:
        bid, split = rec["id"], rec.get("split")
        exterior_air += rec["masks"]["exterior_air"]
        mine = collections.Counter()
        for state, n in rec["supervised"].items():
            c3, c2 = info(state)
            if c3["class"] == "air":
                continue
            cls, fam = c3["class"], c3["family"]
            total += n
            v3[cls] += n
            v2[c2] += n
            v3_builds[cls].add(bid)
            v2_builds[c2].add(bid)
            fam_total[fam] += n
            fam_v3[fam][cls] += n
            fam_v2[fam][c2] += n
            split_total[split] += n
            split_v3[split][cls] += n
            split_v2[split][c2] += n
            mine[cls] += n
            for r in c3["reasons"]:
                reasons[r] += n
                reason_builds[r].add(bid)
            if cls == "ignore":
                base = split_state(state)[0]
                ignored[base] += n
                ignored_builds[base].add(bid)
                ignored_family[base] = fam
            if c3.get("material") not in (None, vocab.IGNORE_MATERIAL):
                materials_used[c3["material"]] += n
        per_build[bid] = {k: _share(mine[k], sum(mine.values())) for k in V3_CLASSES}

    rep3 = sum(v3[c] for c in REPRESENTABLE)
    rep2 = v2["exact"] + v2["lossy"]
    by_family = {}
    for fam, n in fam_total.most_common():
        by_family[fam] = {"blocks": n, "share": _share(n, total),
                          "v2": _shares(fam_v2[fam], V2_CLASSES, n),
                          "v3": _shares(fam_v3[fam], V3_CLASSES, n)}
    per_split = {}
    for s in sorted(split_total):
        n = split_total[s]
        per_split[s] = {"blocks": n,
                        "v2": {**_shares(split_v2[s], V2_CLASSES, n),
                               "representable": _share(split_v2[s]["exact"] + split_v2[s]["lossy"], n)},
                        "v3": {**_shares(split_v3[s], V3_CLASSES, n),
                               "representable": _share(sum(split_v3[s][c] for c in REPRESENTABLE), n)}}
    used = set(materials_used)
    return {
        "scope": "supervised blocks (shell) of every normalized build; exterior air is "
                 "supervised too and always representable, so it is counted but kept out of "
                 "the percentages",
        "vocab_version": vocab.VOCAB_VERSION,
        "shapes": len(vocab.SHAPES), "materials": len(vocab.MATERIALS),
        "valid_pairs": int(vocab.VALID.sum()),
        "target_version": vocab.TARGET_VERSION, "target_data_version": vocab.TARGET_DATA_VERSION,
        "builds": len(records), "supervised_blocks": total, "exterior_air": exterior_air,
        "v3": {"classes": {c: {"voxels": v3[c], "share": _share(v3[c], total),
                               "builds": len(v3_builds[c])} for c in V3_CLASSES},
               "representable": _share(rep3, total), "exact": _share(v3["exact"], total),
               "reasons": {r: {"voxels": n, "share": _share(n, total),     # lossy and derived
                               "builds": len(reason_builds[r])}
                           for r, n in reasons.most_common()}},
        "v2": {"classes": {c: {"voxels": v2[c], "share": _share(v2[c], total),
                               "builds": len(v2_builds[c])} for c in V2_CLASSES},
               "representable": _share(rep2, total), "exact": _share(v2["exact"], total)},
        "by_family": by_family,
        "per_split": per_split,
        "top_unrepresentable": [
            {"name": name, "family": ignored_family[name], "voxels": n, "share": _share(n, total),
             "builds": len(ignored_builds[name])}
            for name, n in ignored.most_common(30)],
        "materials_used": len(used),
        "materials_unused": [vocab.MATERIALS[i] for i in range(1, len(vocab.MATERIALS))
                             if i not in used],
        "per_build_v3": dict(sorted(per_build.items())),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    args = ap.parse_args(argv)
    norm = args.dataset / "normalized"
    index = json.loads((norm / "index.json").read_text(encoding="utf-8"))
    records = [json.loads((norm / "records" / f"{b['id']}.json").read_text(encoding="utf-8"))
               for b in index["builds"] if b["status"] == "ok"]
    out = aggregate(records)
    (args.dataset / "vocab_coverage.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"{out['builds']} builds, {out['supervised_blocks']:,} supervised blocks")
    print(f"  v2 representable {out['v2']['representable']}%, v3 representable "
          f"{out['v3']['representable']}% (exact {out['v3']['exact']}%)")
    for cls, v in out["v3"]["classes"].items():
        print(f"  v3 {cls:8s} {v['share']:7.2f}%  {v['voxels']:>10,}  builds {v['builds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
