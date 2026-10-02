#!/usr/bin/env python3
"""Statistics and contact sheets for a run.

Usage:  python -m synth.report data/synthetic/pilot-v2 [--no-sheets] [--no-roofs]

Writes into the run directory:

    report.json    every number below, machine-readable
    report.md      the same, for reading
    sheets/*.png   one contact sheet per architectural group: source | input | shapes | target

Covers: the funnel from the index to written pairs with every rejection reason (metadata,
geometry, scale, target, validation), architectural groups and categories (with the
imbalance stated, not hidden), size and height at 1:1, scale sources and their confidence,
voxel resolution and tiles per window size, materials, full / slab / stair frequency for
both roof styles, roof construction against the real builds (`synth.roof_stats`),
input-vs-target agreement, degradation parameter coverage, storage and timings, and the
human review verdicts once `synth.review` has recorded some.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def dist(values):
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if len(v) == 0:
        return {"n": 0}
    q = np.percentile(v, [10, 50, 90])
    return {"n": int(len(v)), "p10": round(float(q[0]), 3), "p50": round(float(q[1]), 3),
            "p90": round(float(q[2]), 3), "max": round(float(v.max()), 3), "mean": round(float(v.mean()), 3)}


def _shares(counter):
    tot = sum(counter.values())
    return {k: round(v / tot, 4) for k, v in counter.most_common()} if tot else {}


def load(root: Path):
    mans = []
    for p in sorted((root / "samples").glob("*/manifest.json")):
        try:
            mans.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    ledger = []
    if (root / "ledger.jsonl").exists():
        ledger = [json.loads(line) for line in open(root / "ledger.jsonl", encoding="utf-8") if line.strip()]
    sel = [json.loads(line) for line in open(root / "selection.jsonl", encoding="utf-8")] \
        if (root / "selection.jsonl").exists() else []
    sel_report = json.loads((root / "selection_report.json").read_text(encoding="utf-8")) \
        if (root / "selection_report.json").exists() else {}
    review = json.loads((root / "review.json").read_text(encoding="utf-8")).get("verdicts", {}) \
        if (root / "review.json").exists() else {}
    return mans, ledger, sel, sel_report, review


def build(root: Path, roofs: bool = True):
    from . import taxonomy as T

    mans, ledger, sel, sel_report, review = load(root)
    ok = [m for m in mans if m["status"] == "ok"]
    rej = [m for m in mans if m["status"] == "rejected"]
    err = [m for m in mans if m["status"] == "error"]
    grp = lambda m: (m.get("source") or {}).get("group") or "?"          # noqa: E731
    cat = lambda m: (m.get("source") or {}).get("category") or "?"       # noqa: E731
    sel_group = collections.Counter(T.group(r["category"], r.get("subcategory")) for r in sel)
    sel_cat = collections.Counter(r["category"] for r in sel)

    rep = {"run": root.name, "config_hash": (ok or mans or [{}])[0].get("config_hash"),
           "pipeline_version": (ok or mans or [{}])[0].get("pipeline_version")}
    by_stage = collections.defaultdict(collections.Counter)
    for m in rej:
        by_stage[m.get("stage", "?")][m.get("reason", "?")] += 1
    raws = [m.get("raw") or {} for m in mans]
    rep["funnel"] = {
        "index_candidates": sel_report.get("candidates"),
        "metadata_rejected": sel_report.get("metadata_rejected"),
        "eligible": sel_report.get("eligible"),
        "selected": len(sel),
        "downloaded": sum(1 for r in raws if r.get("downloaded")),
        "read_from_existing_cache": sum(1 for r in raws if r.get("downloaded") is False),
        "processed": len(mans), "ok": len(ok), "rejected": len(rej), "errors": len(err),
        "rejected_by_stage": {s: dict(c.most_common()) for s, c in by_stage.items()},
        "error_reasons": dict(collections.Counter(m.get("reason") for m in err)),
    }

    def breakdown(key, selected):
        out = {}
        for k in sorted(set(selected) | {key(m) for m in mans}):
            ms = [m for m in mans if key(m) == k]
            out[k] = {"selected": selected.get(k, 0), "ok": sum(1 for m in ms if m["status"] == "ok"),
                      "rejected": dict(collections.Counter(m.get("reason") for m in ms if m["status"] == "rejected"))}
        return out
    rep["by_group"] = breakdown(grp, sel_group)
    rep["by_category"] = breakdown(cat, sel_cat)
    g_ok = collections.Counter(grp(m) for m in ok)
    rep["balance"] = {"ok_share_by_group": _shares(g_ok),
                      "largest_group": g_ok.most_common(1)[0] if g_ok else None,
                      "residential_share": round(g_ok.get("residential", 0) / max(len(ok), 1), 3),
                      "detached_house_share": round(sum(1 for m in ok if cat(m) == "house") / max(len(ok), 1), 3)}

    dims = [(m["target"]["dims"], m["scale"]) for m in ok]
    rep["size"] = {"footprint_blocks": dist(d["footprint_blocks"] for d, _ in dims),
                   "height_blocks": dist(d["height_blocks"] for d, _ in dims),
                   "height_m": dist(s["height_m"] for _, s in dims),
                   "flags": {k: sum(1 for m in ok if (m.get("flags") or {}).get(k))
                             for k in ("scale_uncertain", "height_capped", "size_capped", "small_building")},
                   "per_group": {g: {"footprint": dist(m["target"]["dims"]["footprint_blocks"] for m in ok if grp(m) == g),
                                     "height": dist(m["target"]["dims"]["height_blocks"] for m in ok if grp(m) == g)}
                                 for g in sorted(g_ok)}}
    rep["scale"] = {"source": dict(collections.Counter(m["scale"]["source"] for m in ok)),
                    "confidence": dict(collections.Counter(m["scale"].get("confidence") for m in ok)),
                    "unit": dict(collections.Counter(m["scale"].get("unit") for m in ok if m["scale"].get("unit"))),
                    "rejected_by_scale": dict(collections.Counter(m.get("reason") for m in rej if m.get("stage") == "scale"))}
    res = [m["resolution"] for m in ok]
    windows = sorted({int(w) for r in res for w in r["tiles"]})
    rep["resolution"] = {
        "grid_voxels": dist(int(np.prod(r["grid_xyz"])) for r in res),
        "grid_longest_axis": dist(max(r["grid_xyz"]) for r in res),
        "tiles": {str(w): dist(r["tiles"][str(w)] for r in res) for w in windows},
        "fits_one_window": {str(w): round(sum(r["fits_one_window"][str(w)] for r in res) / max(len(res), 1), 3)
                            for w in windows},
        "total_tiles": {str(w): int(sum(r["tiles"][str(w)] for r in res)) for w in windows},
    }

    styles = sorted({s for m in ok for s in (m.get("blocks") or {})})
    shapes = {}
    for s in styles:
        tot = collections.Counter()
        per = collections.defaultdict(collections.Counter)
        for m in ok:
            c = collections.Counter(m["blocks"][s]["shapes"])
            tot.update(c)
            per[grp(m)].update(c)
        shapes[s] = {"all": _shares(_fold(tot)), "per_group": {g: _shares(_fold(c)) for g, c in sorted(per.items())}}
    rep["shapes"] = shapes
    primary = (ok[0]["target"].get("roof_style") if ok else None) or "builder"
    mats = collections.Counter()
    for m in ok:
        for name, n in m["blocks"][primary]["materials"]:
            mats[name] += n
    rep["materials"] = {"style": primary, "top": mats.most_common(25),
                        "distinct_per_sample": dist(m["blocks"][primary]["distinct_states"] for m in ok),
                        "samples_with_glass": sum(1 for m in ok if (m["target"]["blocks_" + primary].get("glass_voxels") or 0) > 0),
                        "samples_with_door_material": sum(1 for m in ok if (m["target"]["blocks_" + primary].get("door_voxels") or 0) > 0),
                        "sources_with_glass_faces": sum(1 for m in ok if m["scene"]["glass_faces"] > 0),
                        "sources_with_door_faces": sum(1 for m in ok if m["scene"]["door_faces"] > 0)}
    rep["roof_plan"] = {"columns_planned": dist(m["target"]["roofs"].get("planned") for m in ok),
                        "by_class": _sum_dicts(m["target"]["roofs"].get("by_class", {}) for m in ok),
                        "caps_restated_for_material": int(sum(m["target"]["roofs"].get("caps_restated", 0) for m in ok))}
    if roofs:
        rep["roof_construction"] = roof_construction(root, ok)
    rep["metrics"] = {k: dist(m["metrics"].get(k) for m in ok) for k in ("occupancy_iou", "shell_overlap", "shell_delta_e")}
    params = collections.defaultdict(list)
    for m in ok:
        for k, v in (m.get("degrade", {}).get("params") or {}).items():
            if isinstance(v, (int, float)) and k != "seed":
                params[k].append(v)
    rep["degradation"] = {k: dist(v) for k, v in sorted(params.items())}
    by_file = collections.Counter()
    for m in ok:
        for name, b in (m.get("files") or {}).items():
            kind = "renders (png)" if name.endswith(".png") else name
            by_file[kind] += b
    pair_bytes = [sum((m.get("files") or {}).values()) for m in ok]
    train_bytes = [sum(b for n, b in (m.get("files") or {}).items() if not n.endswith(".png")) for m in ok]
    tmp_left = []
    cfg_path = root / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        tmp_dir = Path(cfg.get("tmp_dir", "")) / cfg.get("run_name", root.name)
        tmp_left = list(tmp_dir.glob("*.glb")) if tmp_dir.exists() else []
    rep["storage"] = {"pairs_total_mb": round(sum(pair_bytes) / 1e6, 2),
                      "raw_files_left_in_temp": len(tmp_left),
                      "training_files_total_mb": round(sum(train_bytes) / 1e6, 2),
                      "per_pair_kb": dist(b / 1e3 for b in pair_bytes),
                      "per_pair_training_kb": dist(b / 1e3 for b in train_bytes),
                      "by_file_mb": {k: round(v / 1e6, 2) for k, v in by_file.most_common()},
                      "raw_downloaded_mb": round(sum(r.get("bytes") or 0 for r in raws if r.get("downloaded")) / 1e6, 1),
                      "raw_read_from_cache_mb": round(sum(r.get("bytes") or 0 for r in raws if r.get("downloaded") is False) / 1e6, 1),
                      "run_dir_mb": round(sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6, 1)}
    stages = sorted({k for m in ok for k in m.get("timings_s", {})})
    rep["timings_s"] = {k: dist(m["timings_s"].get(k) for m in ok) for k in stages}
    rep["timings_s"]["ledger_seconds_per_asset"] = dist(line.get("seconds") for line in ledger)
    if review:
        v_all = collections.Counter(v["verdict"] for v in review.values())
        per = collections.defaultdict(collections.Counter)
        by_uid = {m["uid"]: m for m in mans}
        for uid, v in review.items():
            if uid in by_uid:
                per[grp(by_uid[uid])][v["verdict"]] += 1
        tags = collections.Counter(t for v in review.values() for t in (v.get("tags") or []))
        rep["review"] = {"reviewed": sum(v_all.values()), "verdicts": dict(v_all),
                         "per_group": {g: dict(c) for g, c in sorted(per.items())},
                         "tags": dict(tags.most_common())}
    return rep


def _fold(counter):
    """Stairs in four facings -> 'stairs', slabs -> 'slab'; keeps full / air."""
    out = collections.Counter()
    for k, v in counter.items():
        out["stairs" if k.startswith("stairs") else ("slab" if k.startswith("slab") else k)] += v
    return out


def _sum_dicts(dicts):
    out = collections.Counter()
    for d in dicts:
        out.update(d)
    return dict(out)


def roof_construction(root: Path, ok):
    """synth.roof_stats over the run's targets (both styles, all and houses) + the real builds."""
    import schem_voxels
    from . import roof_stats as R

    out = {}
    for style, fname in (("builder", "target.schem"), ("geometric", "target_geometric.schem")):
        for subset in ("all", "houses"):
            stats = None
            for m in ok:
                if subset == "houses" and m["source"]["category"] not in ("house", "townhouse"):
                    continue
                p = root / "samples" / m["uid"] / fname
                vg = schem_voxels.load_voxels(p) if p.exists() else None
                if vg is not None:
                    stats = R.tabulate(vg.grid, vg.names, None, 0, stats)
            if stats:
                out["{}/{}".format(style, subset)] = R.summarise(stats)
    real = None
    for _d, vg, seg in R.real_builds():
        real = R.tabulate(vg.grid, vg.names, seg.structure, int(seg.ground_top or 0), real)
    if real:
        out["real_builds"] = R.summarise(real)
    return out


def contact_sheets(root: Path, ok, size=150, per_row=2):
    """sheets/<group>.png: source | input | shapes | target for every pair of a group."""
    from PIL import Image, ImageDraw

    out_dir = root / "sheets"
    out_dir.mkdir(exist_ok=True)
    groups = collections.defaultdict(list)
    for m in ok:
        groups[(m.get("source") or {}).get("group") or "?"].append(m)
    panels = ("source.png", "input.png", "shapes.png", "target.png")
    written = []
    for g, ms in sorted(groups.items()):
        ms.sort(key=lambda m: (m["source"]["category"], m["uid"]))
        cell_w = size * len(panels)
        rows = (len(ms) + per_row - 1) // per_row
        sheet = Image.new("RGB", (cell_w * per_row + 10 * (per_row - 1), rows * (size + 16)), (22, 22, 26))
        d = ImageDraw.Draw(sheet)
        for i, m in enumerate(ms):
            x0 = (i % per_row) * (cell_w + 10)
            y0 = (i // per_row) * (size + 16)
            name = m["source"]["name"] or m["uid"]
            name = name if name.isascii() else "(non-Latin title)"
            dims = m["target"]["dims"]
            d.text((x0 + 3, y0 + 2), "{:.8} {} {}x{} {}".format(m["uid"], m["source"]["category"],
                                                               dims["footprint_blocks"], dims["height_blocks"], name[:30]),
                   fill=(215, 215, 220))
            for j, p in enumerate(panels):
                f = root / "samples" / m["uid"] / p
                if f.exists():
                    im = Image.open(f).convert("RGBA")
                    im.thumbnail((size, size))
                    tile = Image.new("RGBA", (size, size), (22, 22, 26, 255))
                    tile.alpha_composite(im, ((size - im.width) // 2, (size - im.height) // 2))
                    sheet.paste(tile.convert("RGB"), (x0 + j * size, y0 + 14))
        path = out_dir / "{}.png".format(g)
        sheet.save(path)
        written.append(str(path))
    return written


def to_markdown(rep):
    L = ["# Run {} statistics".format(rep["run"]), "",
         "Pipeline {}, config {}.".format(rep.get("pipeline_version"), rep.get("config_hash")), ""]
    f = rep["funnel"]
    L += ["## Funnel", "",
          "| Stage | Count |", "| --- | ---: |",
          "| Objaverse index candidates | {} |".format(f.get("index_candidates") if f.get("index_candidates") is not None else "not measured for this selection"),
          "| Eligible after licence + metadata filters | {} |".format(f.get("eligible") if f.get("eligible") is not None else "not measured for this selection"),
          "| Selected | {} |".format(f["selected"]),
          "| Downloaded / read from an existing cache | {} / {} |".format(f["downloaded"], f["read_from_existing_cache"]),
          "| Processed | {} |".format(f["processed"]),
          "| **Pairs written** | **{}** |".format(f["ok"]),
          "| Rejected | {} |".format(f["rejected"]),
          "| Errors | {} |".format(f["errors"]), ""]
    L += ["Metadata-stage rejections (before download): " + ", ".join(
        "{} {}".format(k, v) for k, v in (f.get("metadata_rejected") or {}).items()), ""]
    L += ["Rejections after download:", "", "| Stage | Reason | Count |", "| --- | --- | ---: |"]
    for s, c in f["rejected_by_stage"].items():
        for r, n in c.items():
            L.append("| {} | {} | {} |".format(s, r, n))
    L.append("")
    L += ["## Architectural groups", "", "| Group | Selected | Pairs | Rejected (reasons) |", "| --- | ---: | ---: | --- |"]
    for g, row in rep["by_group"].items():
        L.append("| {} | {} | {} | {} |".format(g, row["selected"], row["ok"], ", ".join(
            "{} {}".format(k, v) for k, v in row["rejected"].items())))
    b = rep["balance"]
    L += ["", "Pairs by group: " + ", ".join("{} {:.0%}".format(k, v) for k, v in b["ok_share_by_group"].items()),
          "Residential share {:.0%}; detached houses {:.0%}.".format(b["residential_share"], b["detached_house_share"]), ""]
    L += ["## Size and estimated real-world scale", "",
          "The goal is about one block per metre. Jitter and resource caps change the effective scale; see each manifest's effective_blocks_per_metre.", "",
          "| Measure | p10 | p50 | p90 | max |", "| --- | ---: | ---: | ---: | ---: |"]
    for k in ("footprint_blocks", "height_blocks", "height_m"):
        d = rep["size"][k]
        L.append("| {} | {} | {} | {} | {} |".format(k, d.get("p10"), d.get("p50"), d.get("p90"), d.get("max")))
    fl = rep["size"]["flags"]
    L += ["", "Kept and flagged (never rejected): {} of {} pairs have an uncertain scale, {} were scaled "
              "down to fit the height bound and {} the size bound, {} are small buildings.".format(
                  fl["scale_uncertain"], f["ok"], fl["height_capped"], fl["size_capped"], fl["small_building"]),
          "Scale sources: {}.  Confidence: {}.  Units: {}.".format(rep["scale"]["source"], rep["scale"]["confidence"],
                                                                   rep["scale"]["unit"]), ""]
    r = rep["resolution"]
    L += ["## Voxel resolution", "", "Grid longest axis p50 {} / p90 {} / max {}; voxels p50 {:,.0f}.".format(
        r["grid_longest_axis"].get("p50"), r["grid_longest_axis"].get("p90"), r["grid_longest_axis"].get("max"),
        r["grid_voxels"].get("p50") or 0), "", "| Window | Pairs fitting one window | Tiles p50 | Tiles max | Tiles total |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for w in r["tiles"]:
        L.append("| {}^3 | {:.0%} | {} | {} | {} |".format(w, r["fits_one_window"][w], r["tiles"][w].get("p50"),
                                                          r["tiles"][w].get("max"), r["total_tiles"][w]))
    L += ["", "## Shapes (supervised solid voxels)", "", "| Style | full | slab | stairs |", "| --- | ---: | ---: | ---: |"]
    for s, row in rep["shapes"].items():
        a = row["all"]
        L.append("| {} | {:.1%} | {:.1%} | {:.1%} |".format(s, a.get("full", 0), a.get("slab", 0), a.get("stairs", 0)))
    if rep.get("roof_construction"):
        L += ["", "## Roof construction (caps of evenly pitched roof columns)", "",
              "| Source | columns | stairs | full | bottom slab | stairs facing up-slope |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for k, s in rep["roof_construction"].items():
            p = s["by_slope"]["pitched"]
            sf = s["stair_facing"]
            tot = sum(sf.values()) or 1
            L.append("| {} | {:,} | {:.1%} | {:.1%} | {:.1%} | {:.1%} |".format(
                k, p["columns"], p["stairs"], p["full"], p["slab_bottom"], sf.get("up_slope", 0) / tot))
    m = rep["materials"]
    L += ["", "## Materials ({} target)".format(m["style"]), "",
          "Distinct block states per pair p50 {}. Glass in {} pairs ({} sources had glass materials); door material in {} "
          "({} sources had door materials).".format(m["distinct_per_sample"].get("p50"), m["samples_with_glass"],
                                                   m["sources_with_glass_faces"], m["samples_with_door_material"],
                                                   m["sources_with_door_faces"]), "",
          "Top materials by voxels: " + ", ".join("{} {:,}".format(n, c) for n, c in m["top"][:15]), ""]
    mt = rep["metrics"]
    L += ["## Input vs target", "", "| Metric | p10 | p50 | p90 |", "| --- | ---: | ---: | ---: |"]
    for k, d in mt.items():
        L.append("| {} | {} | {} | {} |".format(k, d.get("p10"), d.get("p50"), d.get("p90")))
    s = rep["storage"]
    L += ["", "## Storage", "",
          "Pairs {} MB in total ({} MB of it training files, the rest review renders); per pair p50 {} KB "
          "(training files {} KB). Raw GLBs downloaded {} MB, read from an existing cache {} MB, none kept. "
          "Run folder {} MB.".format(s["pairs_total_mb"], s["training_files_total_mb"], s["per_pair_kb"].get("p50"),
                                     s["per_pair_training_kb"].get("p50"), s["raw_downloaded_mb"],
                                     s["raw_read_from_cache_mb"], s["run_dir_mb"]), ""]
    t = rep["timings_s"]
    L += ["## Time", "", "Seconds per asset p50 {} / p90 {}; target p50 {}, input p50 {}, renders p50 {}.".format(
        t["ledger_seconds_per_asset"].get("p50"), t["ledger_seconds_per_asset"].get("p90"),
        t.get("target", {}).get("p50"), t.get("input", {}).get("p50"), t.get("render", {}).get("p50")), ""]
    if rep.get("review"):
        rv = rep["review"]
        L += ["## Human review", "", "{} reviewed: {}.".format(rv["reviewed"], rv["verdicts"]), ""]
        for g, c in rv["per_group"].items():
            L.append("- {}: {}".format(g, c))
        if rv.get("tags"):
            L += ["", "Problem tags: " + ", ".join("{} {}".format(t, n) for t, n in rv["tags"].items()) + "."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--no-roofs", action="store_true", help="skip the (slower) roof-construction tables")
    args = ap.parse_args(argv)
    root = Path(args.root)
    rep = build(root, roofs=not args.no_roofs)
    if not args.no_sheets:
        mans = load(root)[0]
        rep["sheets"] = contact_sheets(root, [m for m in mans if m["status"] == "ok"])
    (root / "report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (root / "report.md").write_text(to_markdown(rep), encoding="utf-8")
    f = rep["funnel"]
    print("{}: {} selected, {} pairs, {} rejected, {} errors -> {}".format(
        root.name, f["selected"], f["ok"], f["rejected"], f["errors"], root / "report.md"))


if __name__ == "__main__":
    main()
