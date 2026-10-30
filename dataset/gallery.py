#!/usr/bin/env python3
"""Review gallery: one card per normalized build, before and after the crop.

Usage:  python -m dataset.gallery --dataset data/real-schems-v1 [--only <id> ...]
                                   [--workers N] [--force]

Writes `<dataset>/review/gallery_v2/` (gitignored, like all of data/): `index.html`, which
opens locally, and `img/`. Each card shows the source (BEFORE the crop) and every kept block
(AFTER; not only the shell, so a missing part shows as a hole), each from two sides, with
an overlay toggle:

    red     outside-visible but not trained on (`interior.exposure`)
    orange  below the ground and ignored
    pink    classed as landscape: removed (before) or kept soft blocks (after)
    blue    removed as a site structure (before): boundary walls, paths, plazas, clutter

Filters: split, top-level category, flag, the "fixed cases" (the builds the latest rule
change was made for, `FIXED_CASES`), the earlier review cases (`EARLIER_CASES`), and "cap
changed" (shell more than `CHANGED_SHELL` different from the previous pipeline version,
`normalized/previous_run.json`); sorts: most ignored, most removed, biggest, most changed
against the previous version; and a search over id, name and creator. Images are cached in
`images.json`; `--force` (or `--only`) draws them again.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

YAWS = (45, 225)
SIZE = 300
MARK = {"ignored": "minecraft:red_concrete[mark=ignored]",
        "below": "minecraft:orange_concrete[mark=below]",
        "landscape": "minecraft:pink_concrete[mark=landscape]",
        "site": "minecraft:light_blue_concrete[mark=site]"}

# The depth cap under cover: hollow shells and open halls it is for, and the porch set it
# must not hurt.
FIXED_CASES = {
    "hall": ["pmc-baroque-townhouse", "pmc-brick-tenement-house", "pmc-old-tenement-house",
             "pmc-adelaide-hotel-toronto", "pmc-warehouse-factory-hall",
             "pmc-aircraft-hangar-private-jet-hangar-1-5-1-scale-1-1-scale-adaptable-full-interiors",
             "ab-13526"],
    "porch": ["ab-6057", "pmc-triple-gable-tudor-townhouse-6884362",
              "pmc-cape-cod-house---furnished",
              "pmc-small-craftsman-house---exterior-amp-landscape-only",
              "pmc-victorian-queen-anne-style---carson-mansion",
              "pmc-spanish-colonial-house---philippines", "pmc-greek-revival-plantation-house"],
}
# Builds worth checking by eye: crop cases and ground cases.
EARLIER_CASES = {
    "crop": ["pmc-chapel-in-the-cemetery-download", "ab-4273",
             "pmc-east-german-kt60-gym-hall-turnhalle", "ab-11853"],
    "ground": ["pmc-european-building-12-kaiserthal-project-by-tim0fei",
               "pmc-10-early-19-century-style-rowhouses", "pmc-eunpyeong-public-library-seoul",
               "pmc-richelieu-contemporary", "pmc-19th-century---brewery", "pmc-old-warehouse"],
}
CHANGED_SHELL = 0.05             # "cap changed": shell differs from the previous run by more


def _marked(blocks: np.ndarray, palette: list[str], marks: dict) -> tuple[np.ndarray, list[str]]:
    """Paint voxels in each mask with that mark's own palette entry (later masks win)."""
    grid = blocks.astype(np.uint16).copy()
    pal = list(palette)
    for name, mask in marks.items():
        if mask is None or not mask.any():
            continue
        pal.append(MARK[name])
        grid[mask & (grid != 0)] = len(pal) - 1
    return grid, pal


def render_card(dataset: str, rec: dict, out_dir: str) -> dict:
    """Render the eight images of one card; returns what the card needs. Never raises."""
    from minecraft import render
    from .crop import crop_build, role_of, COVER, TURF, FLUID
    from .interior import exposure
    from .read import load_build

    dataset, out_dir = Path(dataset), Path(out_dir)
    bid = rec["id"]
    out = {"id": bid, "images": {}}
    try:
        build = load_build(dataset / rec.get("source_root", "source") / rec["file"])
        crop = crop_build(build)
        with np.load(dataset / "normalized" / rec["npz"]) as z:
            blocks, palette = z["blocks"], [str(s) for s in z["palette"]]
            shell, exterior = z["shell"], z["exterior"]
            height = z["ground_height"] if "ground_height" in z.files else int(z["ground_y"])
        _, m = exposure(blocks != 0, shell, exterior, height)
        soft_pal = np.array([role_of(s) in (COVER, TURF, FLUID) for s in palette])
        before_ov = _marked(build.blocks, build.palette,
                            {"landscape": crop.removed_landscape, "site": crop.removed_site})
        after_ov = _marked(blocks, palette, {"landscape": soft_pal[blocks] & (blocks != 0),
                                             "below": m["below_ground"],
                                             "ignored": m["visible_ignored"]})
        views = {"before": (build.blocks, build.palette), "before_ov": before_ov,
                 "after": (blocks, palette), "after_ov": after_ov}
        for key, (grid, pal) in views.items():
            for k, yaw in enumerate(YAWS):
                image = render.draw(grid, pal, yaw=yaw, size=SIZE)
                name = f"{bid}__{key}{k}.png"
                if image is not None:
                    image.save(out_dir / name, optimize=True)
                    out["images"][f"{key}{k}"] = name
    except Exception as exc:                                         # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _change(rec: dict, prev: dict | None) -> dict:
    if not prev:
        return {"removed": None, "shell": None, "score": 0.0}
    d_removed = rec["crop"]["removed_fraction"] - prev.get("removed_fraction", 0.0)
    old_shell = prev.get("shell") or 0
    d_shell = (rec["masks"]["shell"] - old_shell) / old_shell if old_shell else None
    score = max(abs(d_removed), abs(d_shell) if d_shell is not None else 0.0)
    return {"removed": round(d_removed, 4),
            "shell": round(d_shell, 4) if d_shell is not None else None,
            "score": round(score, 4)}


def card_data(rec: dict, row: dict, prev: dict | None, images: dict) -> dict:
    c, m = rec["crop"], rec["masks"]
    fixed = [k for k, ids in FIXED_CASES.items() if rec["id"] in ids]
    earlier = [k for k, ids in EARLIER_CASES.items() if rec["id"] in ids]
    change = _change(rec, prev)
    return {
        "id": rec["id"], "name": row.get("build_name") or "", "creator": row.get("creator") or "",
        "url": row.get("source_url") or "", "split": rec["split"], "origin": row.get("origin") or "",
        "cat": rec["building_category"], "top": rec["building_category"].split("/")[0],
        "flags": rec["flags"], "fixed": fixed, "earlier": earlier,
        "cap_changed": change["shell"] is not None and abs(change["shell"]) > CHANGED_SHELL,
        "removed": c["removed_fraction"], "ignored": m["visible_ignored_fraction"],
        "below": m["below_ground_fraction"], "interior": m["interior_fraction"],
        "shell": m["shell"], "kept": c["non_air_after"], "dims": c.get("dims_after"),
        "ground": f"{c['ground_level']} ({c['ground_method']})",
        "storey": f"{rec['storey']['blocks_per_storey']} ({rec['storey']['confidence']})",
        "change": change, "img": images,
    }


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ review</title>
<style>
:root{--bg:#f4f3ef;--card:#fff;--ink:#222;--muted:#666;--line:#ddd;--tag:#eee;--bad:#fde2e2;--fix:#dff3e4}
body{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;
 display:flex;flex-wrap:wrap;gap:8px;align-items:center;z-index:2}
header h1{font-size:16px;margin:0 10px 0 0}
select,input{font:inherit;padding:4px 6px}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:12px;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.pics{display:grid;grid-template-columns:repeat(4,1fr);background:#f6f6f3}
.pics img{width:100%;aspect-ratio:1;object-fit:contain;display:block;cursor:zoom-in}
.lab{display:grid;grid-template-columns:1fr 1fr;font-size:11px;color:var(--muted);text-align:center;
 border-bottom:1px solid var(--line)}
.info{padding:8px 10px;font-size:12px;line-height:1.5}
.info b{font-size:13px}
.tag{display:inline-block;background:var(--tag);border-radius:4px;padding:0 5px;margin:2px 3px 0 0;font-size:11px}
.tag.bad{background:var(--bad)} .tag.fix{background:var(--fix)}
.k{color:var(--muted)}
.legend{font-size:12px;color:#555}
.legend span{display:inline-block;width:10px;height:10px;margin:0 3px 0 8px;vertical-align:middle}
#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:5;align-items:center;
 justify-content:center;flex-direction:column}
#modal img{max-width:92vw;max-height:84vh;background:#f6f6f3}
#modal div{color:#fff;margin-top:8px;font-size:14px}
</style></head><body>
<header><h1>__TITLE__ after the crop (<span id="n"></span>)</h1>
<input id="q" placeholder="search id / name / creator" size="22">
<select id="split"><option value="">all splits</option><option>train</option><option>val</option><option>test</option></select>
<select id="origin"><option value="">all origins</option></select>
<select id="cat"><option value="">all categories</option></select>
<select id="flag"><option value="">any flags</option><option value="__fixed">fixed cases (depth cap: halls + porches)</option>
<option value="__cap">cap changed (shell &gt; 5% vs previous run)</option><option value="__earlier">earlier cases (crop, ground)</option></select>
<select id="sort"><option value="fixed">sort: fixed cases first</option><option value="ignored">sort: most ignored</option>
<option value="removed">sort: most removed</option><option value="size">sort: biggest</option>
<option value="changed">sort: most changed vs previous run</option><option value="name">sort: id</option></select>
<label><input type="checkbox" id="ov"> overlays</label>
<span class="legend"><span style="background:#b02e26"></span>outside-visible, not trained
<span style="background:#f9801d"></span>below ground, ignored <span style="background:#f38baa"></span>landscape
<span style="background:#3ab3da"></span>site removed</span>
</header>
<div id="grid"></div>
<div id="modal" onclick="this.style.display='none'"><img id="mimg"><div id="mtxt"></div></div>
<script>
const R = __DATA__;
const pct = v => v == null ? "–" : (100 * v).toFixed(1) + "%";
const sgn = v => v == null ? "–" : (v > 0 ? "+" : "") + (100 * v).toFixed(1);
const $ = id => document.getElementById(id);
const BAD = new Set(["ground_fallback","large_opening","multi_building","crop_aggressive","crop_no_core",
  "site_linear_removed","crop_kept_landscape"]);
const cats = [...new Set(R.map(r => r.top))].sort();
cats.forEach(c => { const o = document.createElement("option"); o.textContent = c; $("cat").appendChild(o); });
[...new Set(R.map(r => r.origin).filter(Boolean))].sort().forEach(c => {
  const o = document.createElement("option"); o.textContent = c; $("origin").appendChild(o); });
[...new Set(R.flatMap(r => r.flags))].sort().forEach(f => {
  const o = document.createElement("option"); o.textContent = f; $("flag").appendChild(o); });
function img(r, key, ov) {
  const name = r.img[ov ? key.replace(/(\d)$/, "_ov$1") : key];      // before0 -> before_ov0
  return name ? `<img src="img/${name}" data-cap="${r.id} ${key}${ov ? ' overlay' : ''}" loading="lazy">` : "<div></div>";
}
function card(r, ov) {
  const tags = r.flags.map(f => `<span class="tag${BAD.has(f) ? ' bad' : ''}">${f}</span>`).join("")
    + r.fixed.map(f => `<span class="tag fix">fixed: ${f}</span>`).join("")
    + (r.cap_changed ? `<span class="tag fix">cap changed</span>` : "");
  const ch = r.change;
  return `<div class="card"><div class="pics">${img(r,"before0",ov)}${img(r,"before1",ov)}${img(r,"after0",ov)}${img(r,"after1",ov)}</div>
  <div class="lab"><div>BEFORE crop (source)</div><div>AFTER crop (all kept blocks)</div></div>
  <div class="info"><b>${r.id}</b> ${r.url ? `<a href="${r.url}" target="_blank" rel="noopener">listing</a>` : ""}<br>
  <span class="k">${r.name} · ${r.creator}</span><br>
  <span class="k">split</span> ${r.split} · <span class="k">category</span> ${r.cat} · <span class="k">dims</span> ${r.dims ? r.dims.join("×") : "–"}<br>
  <span class="k">removed</span> ${pct(r.removed)} · <span class="k">visible but ignored</span> ${pct(r.ignored)} ·
  <span class="k">below ground</span> ${pct(r.below)} · <span class="k">interior</span> ${r.interior.toFixed(2)} ·
  <span class="k">shell</span> ${r.shell.toLocaleString()}<br>
  <span class="k">ground</span> ${r.ground} · <span class="k">storey</span> ${r.storey} ·
  <span class="k">vs previous run:</span> removed ${ch.removed == null ? "–" : sgn(ch.removed) + " pts"}, shell ${ch.shell == null ? "–" : sgn(ch.shell) + "%"}<br>${tags}</div></div>`;
}
function draw() {
  const q = $("q").value.toLowerCase(), sp = $("split").value, ca = $("cat").value, fl = $("flag").value;
  const og = $("origin").value;
  const ov = $("ov").checked, so = $("sort").value;
  let rows = R.filter(r => (!sp || r.split === sp) && (!ca || r.top === ca) && (!og || r.origin === og)
    && (!fl || (fl === "__fixed" ? r.fixed.length : fl === "__cap" ? r.cap_changed
               : fl === "__earlier" ? r.earlier.length : r.flags.includes(fl)))
    && (!q || (r.id + " " + r.name + " " + r.creator).toLowerCase().includes(q)));
  const key = {ignored: r => -r.ignored, removed: r => -r.removed, size: r => -r.kept,
               changed: r => -r.change.score, name: r => r.id,
               fixed: r => (r.fixed.length ? "0" : "1") + r.id}[so];
  rows.sort((a, b) => { const x = key(a), y = key(b); return x < y ? -1 : x > y ? 1 : 0; });
  $("n").textContent = rows.length + " of " + R.length;
  $("grid").innerHTML = rows.map(r => card(r, ov)).join("");
}
["q","split","origin","cat","flag","sort","ov"].forEach(id => $(id).addEventListener("input", draw));
$("grid").addEventListener("click", e => { if (e.target.tagName === "IMG") {
  $("mimg").src = e.target.src; $("mtxt").textContent = e.target.dataset.cap; $("modal").style.display = "flex"; } });
draw();
</script></body></html>
"""


def write_page(out_dir: Path, cards: list[dict], title: str = "real-schems-v1") -> Path:
    data = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    path = out_dir / "index.html"
    path.write_text(PAGE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data),
                    encoding="utf-8", newline="\n")
    return path


def build_gallery(dataset: Path, only=None, workers: int = 1, force: bool = False) -> Path:
    dataset = Path(dataset)
    norm = dataset / "normalized"
    out_dir = dataset / "review" / "gallery_v2"
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    rows = {r["id"]: r for r in json.loads((dataset / "index.json").read_text(encoding="utf-8"))["rows"]}
    index = json.loads((norm / "index.json").read_text(encoding="utf-8"))
    records = [json.loads((norm / "records" / f"{b['id']}.json").read_text(encoding="utf-8"))
               for b in index["builds"] if b["status"] == "ok"]
    prev_path = norm / "previous_run.json"
    prev = json.loads(prev_path.read_text(encoding="utf-8"))["builds"] if prev_path.is_file() else {}
    cache_path = out_dir / "images.json"
    images = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    todo = [r for r in records
            if force or (only is not None and r["id"] in only) or r["id"] not in images]
    args = [(str(dataset), r, str(out_dir / "img")) for r in todo]
    if workers > 1 and len(args) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(render_card, *zip(*args)))
    else:
        results = [render_card(*a) for a in args]
    for res in results:
        images[res["id"]] = res["images"]
        if res.get("error"):
            print(f"  {res['id']}: render failed: {res['error']}", flush=True)
    cache_path.write_text(json.dumps(images, indent=1) + "\n", encoding="utf-8", newline="\n")
    cards = [card_data(r, rows.get(r["id"], {}), prev.get(r["id"]), images.get(r["id"], {}))
             for r in records]
    return write_page(out_dir, cards, dataset.name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="draw every image again")
    args = ap.parse_args(argv)
    path = build_gallery(args.dataset, set(args.only) if args.only else None, args.workers,
                         args.force)
    print(f"gallery: {html.escape(str(path))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
