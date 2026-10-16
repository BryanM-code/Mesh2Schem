#!/usr/bin/env python3
"""Phase 4 review gallery: per build, the target and what the model is given for it.

Usage:  python -m schemset.pairs_gallery --dataset data/real-schems-v1 [--only <id> ...]
                                         [--workers N] [--force]

Writes `<dataset>/pairs/gallery/` (gitignored, like all of data/): `index.html`, which opens
locally, and `img/`. One card per build with pairs:

    target            the supervised blocks (decoded mcvox-v3 targets; IGNORE not drawn)
    synthetic v0, v1  the first two synthetic inputs, 2 cells per block, in their own colours
    real TRELLIS      per evaluation set (`schemset.real_trellis`: smooth photos = the main
                      set, blocky photos = the stress test), the real-TRELLIS input where it
                      exists, and the photo that was sent to TRELLIS

and per variant the block-level IoU with the target volume, the frame residual, yaw, offset
and bottleneck, plus the build's stair-step numbers. Filters: split, category, real-TRELLIS
set, rejected pairs; sorts: lowest IoU, biggest, id. Images are cached in `images.json`;
`--force` (or `--only`) draws them again.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SIZE = 300
YAW, PITCH = 45, 30
EVAL_SETS = (("smooth", "real_trellis_smooth"), ("blocky", "real_trellis"))


def _target_image(target_shape, target_material):
    from mcvoxel import vocab

    from .review import render
    sup = (target_shape != 0) & (target_shape != vocab.IGNORE_SHAPE)
    if not sup.any():
        return None
    keys = target_shape.astype(np.int64) * 65536 + target_material.astype(np.int64)
    uniq, inv = np.unique(keys[sup], return_inverse=True)
    palette = ["minecraft:air"] + [vocab.decode(int(k // 65536), int(k % 65536)) for k in uniq]
    grid = np.zeros(target_shape.shape, dtype=np.uint16)
    grid[sup] = inv.ravel() + 1
    return render(grid, palette, yaw=YAW, pitch=PITCH, size=SIZE)


def _input_image(occ, rgb):
    from mcvoxel.grid import shell_of
    from synth.preview import render_rgb
    occ = np.asarray(occ, dtype=bool)
    shell = shell_of(occ)
    if not shell.any():
        return None
    yzx = (1, 2, 0)
    return render_rgb(np.transpose(shell, yzx), np.transpose(rgb, yzx + (3,)), yaw=YAW, pitch=PITCH, size=SIZE)


def render_card(dataset: str, bid: str, files: dict, out_dir: str) -> dict:
    """Draw one build's tiles; never raises (a failed tile is missing, with the error)."""
    dataset, out_dir = Path(dataset), Path(out_dir)
    images, errors = {}, []
    jobs = []
    z = np.load(dataset / "normalized" / f"{bid}.npz")
    jobs.append(("target", lambda: _target_image(z["target_shape"], z["target_material"])))
    for key, rel in files.items():
        def job(rel=rel):
            p = np.load(dataset / "pairs" / rel)
            return _input_image(p["input_occ"], p["input_rgb"])
        jobs.append((key, job))
    for key, job in jobs:
        try:
            im = job()
            if im is not None:
                name = f"{bid}__{key}.png"
                im.save(out_dir / name, optimize=True)
                images[key] = name
        except Exception as exc:                                        # noqa: BLE001
            errors.append(f"{key}: {type(exc).__name__}: {exc}"[:200])
    return {"id": bid, "images": images, "errors": errors}


def card_data(bid, rows, row, rts, images) -> dict:
    v = sorted(rows, key=lambda r: r["variant"])
    out = {"id": bid, "name": row.get("build_name") or "", "split": v[0]["split"],
           "cat": row.get("building_category") or "", "top": (row.get("building_category") or "").split("/")[0],
           "dims": v[0].get("target_shape"), "img": images,
           "variants": [{"k": r["variant"], "status": r["status"], "reason": r.get("reason"),
                         "iou": (r.get("validation") or {}).get("iou"),
                         "res": (r.get("validation") or {}).get("shift_residual_cells"),
                         "yaw": r.get("yaw_deg"), "off": r.get("offset_blocks"),
                         "bn": (r.get("degrade") or {}).get("bottleneck"),
                         "stair": (r.get("stats") or {}).get("stair45"),
                         "rough": (r.get("stats") or {}).get("normal_roughness")} for r in v],
           "rt": {}}
    for name, rt in rts.items():
        out["rt"][name] = {"status": rt.get("status"), "iou": (rt.get("validation") or {}).get("iou"),
                           "yaw": (rt.get("placement") or {}).get("yaw_deg"),
                           "shift": (rt.get("placement") or {}).get("shift_cells"),
                           "stair": (rt.get("stats") or {}).get("stair45"),
                           "rough": (rt.get("stats") or {}).get("normal_roughness"),
                           "folder": dict(EVAL_SETS)[name]}
    out["min_iou"] = min([x["iou"] for x in out["variants"] if x["iou"] is not None] or [1.0])
    out["size"] = int(np.prod(out["dims"])) if out["dims"] else 0
    return out


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>real-schems-v1 pairs</title>
<style>
:root{--bg:#f4f3ef;--card:#fff;--ink:#222;--muted:#666;--line:#ddd;--tag:#eee;--bad:#fde2e2;--ok:#dff3e4}
body{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;
 display:flex;flex-wrap:wrap;gap:8px;align-items:center;z-index:2}
header h1{font-size:16px;margin:0 10px 0 0}
select,input{font:inherit;padding:4px 6px}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(860px,1fr));gap:12px;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.pics{display:grid;grid-template-columns:repeat(7,1fr);background:#f6f6f3}
.pics img,.pics .none{width:100%;aspect-ratio:1;object-fit:contain;display:block;cursor:zoom-in}
.pics .none{display:flex;align-items:center;justify-content:center;color:#aaa;font-size:12px;cursor:default}
.lab{display:grid;grid-template-columns:repeat(7,1fr);font-size:11px;color:var(--muted);text-align:center;
 border-bottom:1px solid var(--line)}
.info{padding:8px 10px;font-size:12px;line-height:1.5}
.info b{font-size:13px}
.tag{display:inline-block;background:var(--tag);border-radius:4px;padding:0 5px;margin:2px 3px 0 0;font-size:11px}
.tag.bad{background:var(--bad)} .tag.ok{background:var(--ok)}
.k{color:var(--muted)}
table{border-collapse:collapse;font-size:11px;margin-top:4px}
td,th{padding:1px 6px;border-bottom:1px solid #eee;text-align:right}
th{color:var(--muted);font-weight:normal}
#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:5;align-items:center;
 justify-content:center;flex-direction:column}
#modal img{max-width:92vw;max-height:84vh;background:#f6f6f3}
#modal div{color:#fff;margin-top:8px;font-size:14px}
</style></head><body>
<header><h1>real-schems-v1 pairs (<span id="n"></span>)</h1>
<input id="q" placeholder="search id / name" size="22">
<select id="split"><option value="">all splits</option><option>train</option><option>val</option><option>test</option></select>
<select id="cat"><option value="">all categories</option></select>
<select id="sel"><option value="">all builds</option><option value="rt">real-TRELLIS sets</option>
<option value="rej">with a rejected pair</option></select>
<select id="sort"><option value="iou">sort: lowest IoU</option><option value="size">sort: biggest</option>
<option value="name">sort: id</option></select>
<span class="k">tiles: target · synthetic v0 · v1 · TRELLIS input from the smooth photo (main) · from the blocky photo (stress test) · the two photos</span>
</header>
<div id="grid"></div>
<div id="modal" onclick="this.style.display='none'"><img id="mimg"><div id="mtxt"></div></div>
<script>
const R = __DATA__;
const $ = id => document.getElementById(id);
const f = (v, d) => v == null ? "–" : (+v).toFixed(d);
[...new Set(R.map(r => r.top))].sort().forEach(c => {
  const o = document.createElement("option"); o.textContent = c; $("cat").appendChild(o); });
function tile(r, key, src) {
  const name = src || (r.img[key] ? "img/" + r.img[key] : null);
  return name ? `<img src="${name}" data-cap="${r.id} ${key}" loading="lazy">` : `<div class="none">–</div>`;
}
function card(r) {
  const rows = r.variants.map(v => `<tr><td>v${v.k}</td><td>${v.status === "ok" ? "ok" : '<span class="tag bad">' + v.reason + "</span>"}</td>
    <td>${f(v.iou, 3)}</td><td>${v.res ? v.res.map(x => f(x, 2)).join(" ") : "–"}</td><td>${f(v.yaw, 1)}°</td>
    <td>${v.off ? v.off.map(x => f(x, 2)).join(" ") : "–"}</td><td>${v.bn}</td><td>${f(v.stair, 3)}</td><td>${f(v.rough, 1)}</td></tr>`).join("");
  const rt = Object.entries(r.rt).map(([n, s]) => `<span class="k">TRELLIS (${n} photo):</span> ${s.status}${s.iou != null ? " · IoU " + f(s.iou, 3) + " · yaw " + s.yaw + "° · shift " + (s.shift || []).join(",") + " cells · stair45 " + f(s.stair, 3) + " · roughness " + f(s.rough, 1) + "°" : ""}<br>`).join("");
  const photo = n => r.rt[n] ? tile(r, "photo_" + n, `../${r.rt[n].folder}/${r.id}/photo.png`) : '<div class="none">–</div>';
  return `<div class="card"><div class="pics">${tile(r, "target")}${tile(r, "v0")}${tile(r, "v1")}${tile(r, "rt_smooth")}${tile(r, "rt_blocky")}${photo("smooth")}${photo("blocky")}</div>
  <div class="lab"><div>target</div><div>synthetic v0</div><div>synthetic v1</div><div>TRELLIS · smooth</div><div>TRELLIS · blocky</div><div>photo · smooth</div><div>photo · blocky</div></div>
  <div class="info"><b>${r.id}</b> <span class="k">${r.name}</span><br>
  <span class="k">split</span> ${r.split} · <span class="k">category</span> ${r.cat} · <span class="k">target</span> ${r.dims ? r.dims.join("×") : "–"} blocks<br>${rt}
  <table><tr><th></th><th>status</th><th>IoU</th><th>frame residual (cells)</th><th>yaw</th><th>offset (blocks)</th><th>bottleneck</th><th>stair45</th><th>rough°</th></tr>${rows}</table></div></div>`;
}
function draw() {
  const q = $("q").value.toLowerCase(), sp = $("split").value, ca = $("cat").value, se = $("sel").value, so = $("sort").value;
  let rows = R.filter(r => (!sp || r.split === sp) && (!ca || r.top === ca)
    && (!se || (se === "rt" ? Object.keys(r.rt).length : r.variants.some(v => v.status !== "ok")))
    && (!q || (r.id + " " + r.name).toLowerCase().includes(q)));
  const key = {iou: r => r.min_iou, size: r => -r.size, name: r => r.id}[so];
  rows.sort((a, b) => { const x = key(a), y = key(b); return x < y ? -1 : x > y ? 1 : 0; });
  $("n").textContent = rows.length + " of " + R.length + " builds";
  $("grid").innerHTML = rows.map(card).join("");
}
["q","split","cat","sel","sort"].forEach(id => $(id).addEventListener("input", draw));
$("grid").addEventListener("click", e => { if (e.target.tagName === "IMG") {
  $("mimg").src = e.target.src; $("mtxt").textContent = e.target.dataset.cap; $("modal").style.display = "flex"; } });
draw();
</script></body></html>
"""


def build_gallery(dataset: Path, only=None, workers: int = 1, force: bool = False) -> Path:
    dataset = Path(dataset)
    out_dir = dataset / "pairs" / "gallery"
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    rows = {r["id"]: r for r in json.loads((dataset / "index.json").read_text(encoding="utf-8"))["rows"]}
    index = json.loads((dataset / "pairs" / "index.json").read_text(encoding="utf-8"))
    by_build = {}
    for p in index["pairs"]:
        by_build.setdefault(p["id"], []).append(p)
    rts = {}                                          # id -> {set: row}
    for name, folder in EVAL_SETS:
        rt_path = dataset / "pairs" / folder / "index.json"
        sel_path = dataset / "pairs" / folder / "selection.json"
        if rt_path.is_file():
            for b in json.loads(rt_path.read_text(encoding="utf-8"))["builds"]:
                rts.setdefault(b["id"], {})[name] = b
        elif sel_path.is_file():
            for b in json.loads(sel_path.read_text(encoding="utf-8"))["builds"]:
                rts.setdefault(b["id"], {})[name] = {"status": "awaiting_trellis"}
    cache_path = out_dir / "images.json"
    images = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    jobs = []
    for bid, ps in sorted(by_build.items()):
        if not (force or (only is not None and bid in only) or bid not in images):
            continue
        files = {f"v{p['variant']}": p["file"] for p in ps if p["status"] == "ok" and p["variant"] in (0, 1)}
        for name, folder in EVAL_SETS:
            if (dataset / "pairs" / folder / bid / "input.npz").is_file():
                files["rt_" + name] = f"{folder}/{bid}/input.npz"
        jobs.append((str(dataset), bid, files, str(out_dir / "img")))
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(render_card, *zip(*jobs)))
    else:
        results = [render_card(*j) for j in jobs]
    for res in results:
        images[res["id"]] = res["images"]
        for e in res["errors"]:
            print(f"  {res['id']}: {e}", flush=True)
    cache_path.write_text(json.dumps(images, indent=1) + "\n", encoding="utf-8", newline="\n")
    cards = [card_data(bid, ps, rows.get(bid, {}), rts.get(bid, {}), images.get(bid, {}))
             for bid, ps in sorted(by_build.items())]
    data = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    page = out_dir / "index.html"
    page.write_text(PAGE.replace("__DATA__", data), encoding="utf-8", newline="\n")
    return page


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="draw every image again")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    page = build_gallery(args.dataset, set(args.only) if args.only else None, args.workers, args.force)
    print(f"gallery: {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
