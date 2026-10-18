#!/usr/bin/env python3
"""Phase 4 review gallery: per build, the target and what the model is given for it.

Usage:  python -m schemset.pairs_gallery --dataset data/real-schems-v1 [--only <id> ...]
                                         [--workers N] [--force]

Writes `<dataset>/pairs/gallery/` (gitignored, like all of data/): `index.html`, which opens
locally, and `img/`. Two sections:

  real photos (`schemset.real_photo`), one card per photographed landmark:
    real photo        the photo as found (credit and licence under the card)
    TRELLIS raw       the TRELLIS mesh before voxelisation, posed as aligned
    TRELLIS input     the same mesh voxelised at 2 cells per block: what the model would get
    synthetic v1      the build's second synthetic input
    target            the supervised blocks

  pairs, one card per build with pairs:
    target            the supervised blocks (decoded mcvox-v3 targets; IGNORE not drawn)
    synthetic v0, v1  the first two synthetic inputs, 2 cells per block, in their own colours
    real TRELLIS      per evaluation set (`schemset.real_trellis`: smooth photos = the main
                      set, blocky photos = the stress test): the raw TRELLIS mesh, then its
                      voxelised input, then the two photos that were sent to TRELLIS

A raw mesh is drawn in exactly the frame of the voxelised tile beside it (`_voxel_frame`:
the voxel renderer's own bounding box, cube size and centre) and under the same light, so
the two overlay pixel for pixel: what differs is voxelisation, what is wrong in both is
TRELLIS. Per variant: block-level IoU, frame residual, yaw, offset, bottleneck, stair45,
roughness and the detail metric (`schemset.detail`). Filters: split, category, real-TRELLIS
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
RAW_SAMPLES = 1_500_000
PHOTO_THUMB = 600


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


CUT_RGB = (205, 205, 205)                 # solid cut open by the target frame: it has no colour


def _input_image(occ, rgb, mark_cut: bool = True):
    """An input's outside in its own colours. Solid exposed only because the target frame cut
    the mesh (a TRELLIS mesh bigger than the frame) was inside in the voxelisation, so it
    has no colour at all: it is drawn light grey, as black would read as a dark wall."""
    from mcvoxel.grid import shell_of
    from synth.preview import render_rgb
    occ = np.asarray(occ, dtype=bool)
    shell = shell_of(occ)
    if not shell.any():
        return None
    if mark_cut:
        rgb = np.array(rgb, copy=True)
        rgb[shell & ~np.asarray(rgb, dtype=bool).any(axis=-1)] = CUT_RGB
    yzx = (1, 2, 0)
    return render_rgb(np.transpose(shell, yzx), np.transpose(rgb, yzx + (3,)), yaw=YAW, pitch=PITCH, size=SIZE)


def _voxel_frame(occ):
    """The frame `_input_image` draws `occ` (X, Y, Z) in -- the orbit renderer's budget,
    downsample factor, content box, cube size and centre -- as a projection for points."""
    import schem_voxels
    import thumbs
    from mcvoxel.grid import shell_of
    shell = shell_of(np.asarray(occ, dtype=bool))
    grid = np.transpose(shell, (1, 2, 0)).astype(np.uint16)
    names = ["minecraft:air", "rgb:128,128,128"]
    canvas = SIZE * thumbs.SUPERSAMPLE
    right, down, near = thumbs._view_axes(YAW, PITCH)
    fitted = thumbs._fit_to_budget(schem_voxels.VoxelGrid(grid, names), 300_000,
                                   lambda w, h, l: thumbs._fit_orbit_cell(w, h, l, PITCH, canvas))
    if fitted is None:
        return None
    _, solid, _, factor = fitted
    (y0, y1), (z0, z1), (x0, x1) = thumbs._content_box(solid)
    cell = max(thumbs.MIN_CELL_PX, thumbs._fit_orbit_cell(x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1,
                                                         PITCH, canvas))
    centre = np.array([(x0 + x1 + 1) / 2.0, (y0 + y1 + 1) / 2.0, (z0 + z1 + 1) / 2.0])
    return {"factor": factor, "cell": cell, "centre": centre, "canvas": canvas,
            "right": np.array(right), "down": np.array(down), "near": np.array(near)}


def _points_image(pts_cells, rgb, normals, frame):
    """Surface points (cell units of the 2x grid) drawn in a `_voxel_frame`, shaded as the
    cubes are (`thumbs` face shading, extended to any normal)."""
    import thumbs
    from PIL import Image
    p = pts_cells / frame["factor"] - frame["centre"]
    sx = p @ frame["right"] * frame["cell"] + frame["canvas"] / 2.0
    sy = p @ frame["down"] * frame["cell"] + frame["canvas"] / 2.0
    depth = p @ frame["near"]
    n = normals * np.where((normals @ frame["near"]) < 0, -1.0, 1.0)[:, None]   # towards the viewer
    side = thumbs.SHADE_SIDE_BASE + thumbs.SHADE_SIDE_RANGE * (n @ np.array(thumbs.LIGHT_DIR))
    shade = (np.clip(n[:, 1], 0, None) * thumbs.SHADE_TOP + (1 - np.abs(n[:, 1])) * side
             + np.clip(-n[:, 1], 0, None) * (thumbs.SHADE_SIDE_BASE - 2 * thumbs.SHADE_SIDE_RANGE))
    col = np.clip(np.asarray(rgb, dtype=np.float64) * shade[:, None], 0, 255).astype(np.uint8)
    size = frame["canvas"]
    k = int(np.clip(round(frame["cell"] * 0.6), 2, 6))
    keep = (sx >= 0) & (sy >= 0) & (sx < size - k) & (sy < size - k)
    order = np.argsort(depth[keep], kind="stable")                        # far first
    x0 = np.rint(sx[keep][order]).astype(np.int64)
    y0 = np.rint(sy[keep][order]).astype(np.int64)
    c = col[keep][order]
    off = np.arange(k)
    xs = (x0[:, None, None] + off[None, None, :]).repeat(k, axis=1).ravel()
    ys = (y0[:, None, None] + off[None, :, None]).repeat(k, axis=2).ravel()
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    canvas[ys, xs, :3] = np.repeat(c, k * k, axis=0)
    canvas[ys, xs, 3] = 255
    return Image.fromarray(canvas, "RGBA").resize((SIZE, SIZE), Image.LANCZOS)


def _raw_image(dataset: Path, bid: str, glb: Path, placement: dict, occ):
    """The raw TRELLIS mesh posed as `placement` put it (yaw, scale, shift), in the frame of
    the voxelised input `occ`."""
    from mcvoxel.scene import load_scene, scene_color_fns
    from synth.preview import sample_surface

    from .pairs import SCALE, target_volume
    from .real_trellis import _fit
    z = np.load(Path(dataset) / "normalized" / f"{bid}.npz")
    vol = target_volume(z["target_shape"], z["exterior"], z["ground_height"])
    mesh = load_scene(glb)
    v, _ = _fit(mesh.verts, placement["yaw_deg"], vol, placement.get("scale_mult", 1.0),
                placement.get("aspect", 1.0))
    v = (v + np.asarray(placement["shift_cells"], dtype=np.float64) / SCALE) * SCALE
    _, point_colors = scene_color_fns(mesh)
    pts, cols, normals = sample_surface(v, mesh.faces, point_colors, RAW_SAMPLES, seed=0)
    frame = _voxel_frame(occ)
    if pts is None or frame is None:
        return None
    return _points_image(pts, cols, normals, frame)


def _photo_thumb(path: Path):
    from PIL import Image
    im = Image.open(path)
    im.thumbnail((PHOTO_THUMB, PHOTO_THUMB))
    return im.convert("RGB")


def render_card(dataset: str, bid: str, files: dict, out_dir: str, raws: dict = None, extra: dict = None) -> dict:
    """Draw one build's tiles; never raises (a failed tile is missing, with the error).
    `files`: tile -> pair npz (relative to pairs/); `raws`: tile -> (glb, placement, npz of
    the voxelised input); `extra`: tile -> photo path (drawn as a thumbnail)."""
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
    for key, (glb, placement, rel) in (raws or {}).items():
        def job(glb=glb, placement=placement, rel=rel):
            return _raw_image(dataset, bid, Path(glb), placement, np.load(dataset / "pairs" / rel)["input_occ"])
        jobs.append((key, job))
    for key, path in (extra or {}).items():
        jobs.append((key, lambda path=path: _photo_thumb(Path(path))))
    for key, job in jobs:
        try:
            im = job()
            if im is not None:
                name = f"{bid}__{key}.{'jpg' if key in (extra or {}) else 'png'}"
                im.save(out_dir / name, optimize=True)
                images[key] = name
        except Exception as exc:                                        # noqa: BLE001
            errors.append(f"{key}: {type(exc).__name__}: {exc}"[:200])
    return {"id": bid, "images": images, "errors": errors}


def _detail(stats) -> dict:
    s = stats or {}
    return {"dg": s.get("detail_geo"), "dc": s.get("detail_colour"), "ds": s.get("detail_spurious")}


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
                         "rough": (r.get("stats") or {}).get("normal_roughness"),
                         **_detail(r.get("stats"))} for r in v],
           "flags": sorted({f for r in v for f in ((r.get("validation") or {}).get("flags") or [])}),
           "rt": {}}
    for name, rt in rts.items():
        out["rt"][name] = {"status": rt.get("status"), "iou": (rt.get("validation") or {}).get("iou"),
                           "yaw": (rt.get("placement") or {}).get("yaw_deg"),
                           "shift": (rt.get("placement") or {}).get("shift_cells"),
                           "stair": (rt.get("stats") or {}).get("stair45"),
                           "rough": (rt.get("stats") or {}).get("normal_roughness"),
                           **_detail(rt.get("stats")),
                           "folder": dict(EVAL_SETS)[name]}
    out["min_iou"] = min([x["iou"] for x in out["variants"] if x["iou"] is not None] or [1.0])
    out["size"] = int(np.prod(out["dims"])) if out["dims"] else 0
    return out


def real_card_data(r: dict, images: dict) -> dict:
    s, pl, va = r.get("stats") or {}, r.get("placement") or {}, r.get("validation") or {}
    syn = r.get("synthetic_v1") or {}
    return {"id": r["id"], "subject": r.get("subject"), "split": r.get("split"), "status": r.get("status"),
            "test": r.get("in_test_metrics"), "photo": r.get("photo") or {}, "img": images,
            "cutout": (r.get("cutout") or {}).get("cutout_used"),
            "mask": (r.get("cutout") or {}).get("mask_fraction"),
            "iou": va.get("iou"), "res": va.get("shift_residual_cells"), "fail": va.get("failures"),
            "scale": pl.get("scale_mult"), "aspect": pl.get("aspect"), "yaw": pl.get("yaw_deg"),
            "stair": s.get("stair45"), "rough": s.get("normal_roughness"), "lum": s.get("lum"),
            "chroma": s.get("chroma"), "nbr": s.get("nbr_de"), **_detail(s),
            "cast": [s.get("cast_dL"), s.get("cast_da"), s.get("cast_db")],
            "syn": {"stair": syn.get("stair45"), "rough": syn.get("normal_roughness"), "lum": syn.get("lum"),
                    "chroma": syn.get("chroma"), "nbr": syn.get("nbr_de"), **_detail(syn)}}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>real-schems-v1 pairs</title>
<style>
:root{--bg:#f4f3ef;--card:#fff;--ink:#222;--muted:#666;--line:#ddd;--tag:#eee;--bad:#fde2e2;--ok:#dff3e4}
body{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;
 display:flex;flex-wrap:wrap;gap:8px;align-items:center;z-index:2}
header h1{font-size:16px;margin:0 10px 0 0}
h2{font-size:15px;margin:16px 16px 0}
select,input{font:inherit;padding:4px 6px}
#grid,#real{display:grid;grid-template-columns:repeat(auto-fill,minmax(1000px,1fr));gap:12px;padding:16px}
#real{grid-template-columns:repeat(auto-fill,minmax(720px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.pics{display:grid;grid-template-columns:repeat(9,1fr);background:#f6f6f3}
#real .pics{grid-template-columns:repeat(5,1fr)}
.pics img,.pics .none{width:100%;aspect-ratio:1;object-fit:contain;display:block;cursor:zoom-in}
.pics .none{display:flex;align-items:center;justify-content:center;color:#aaa;font-size:12px;cursor:default}
.lab{display:grid;grid-template-columns:repeat(9,1fr);font-size:11px;color:var(--muted);text-align:center;
 border-bottom:1px solid var(--line)}
#real .lab{grid-template-columns:repeat(5,1fr)}
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
<option value="rej">with a rejected pair</option><option value="low">flagged low_iou</option></select>
<select id="sort"><option value="iou">sort: lowest IoU</option><option value="size">sort: biggest</option>
<option value="name">sort: id</option></select>
<span class="k">raw = the TRELLIS mesh before voxelisation, drawn in the same frame and light as the voxelised tile to its right · light grey in an input = solid cut open by the target frame (it has no colour)</span>
</header>
<h2>Real photos, end to end (<span id="nr"></span>) <span class="k">photo &rarr; background removal &rarr; TRELLIS &rarr; voxelise &rarr; align (yaw, shift, scale)</span></h2>
<div id="real"></div>
<h2>Pairs</h2>
<div id="grid"></div>
<div id="modal" onclick="this.style.display='none'"><img id="mimg"><div id="mtxt"></div></div>
<script>
const R = __DATA__;
const P = __REAL__;
const $ = id => document.getElementById(id);
const f = (v, d) => v == null ? "–" : (+v).toFixed(d);
[...new Set(R.map(r => r.top))].sort().forEach(c => {
  const o = document.createElement("option"); o.textContent = c; $("cat").appendChild(o); });
const byId = Object.fromEntries(R.map(r => [r.id, r]));
function tile(r, key, src) {
  const name = src || (r.img[key] ? "img/" + r.img[key] : null);
  return name ? `<img src="${name}" data-cap="${r.id} ${key}" loading="lazy">` : `<div class="none">–</div>`;
}
const det = s => `detail ${f(s.dg, 2)} / colour ${f(s.dc, 2)} / spurious ${f(s.ds, 2)}`;
function card(r) {
  const rows = r.variants.map(v => `<tr><td>v${v.k}</td><td>${v.status === "ok" ? "ok" : '<span class="tag bad">' + v.reason + "</span>"}</td>
    <td>${f(v.iou, 3)}</td><td>${v.res ? v.res.map(x => f(x, 2)).join(" ") : "–"}</td><td>${f(v.yaw, 1)}°</td>
    <td>${v.off ? v.off.map(x => f(x, 2)).join(" ") : "–"}</td><td>${v.bn}</td><td>${f(v.stair, 3)}</td><td>${f(v.rough, 1)}</td>
    <td>${f(v.dg, 2)}</td><td>${f(v.dc, 2)}</td></tr>`).join("");
  const rt = Object.entries(r.rt).map(([n, s]) => `<span class="k">TRELLIS (${n} photo):</span> ${s.status}${s.iou != null ? " · IoU " + f(s.iou, 3) + " · yaw " + s.yaw + "° · shift " + (s.shift || []).join(",") + " cells · stair45 " + f(s.stair, 3) + " · roughness " + f(s.rough, 1) + "° · " + det(s) : ""}<br>`).join("");
  const photo = n => r.rt[n] ? tile(r, "photo_" + n, `../${r.rt[n].folder}/${r.id}/photo.png`) : '<div class="none">–</div>';
  const flags = r.flags.map(x => `<span class="tag bad">${x}</span>`).join("");
  return `<div class="card"><div class="pics">${tile(r, "target")}${tile(r, "v0")}${tile(r, "v1")}${tile(r, "raw_smooth")}${tile(r, "rt_smooth")}${tile(r, "raw_blocky")}${tile(r, "rt_blocky")}${photo("smooth")}${photo("blocky")}</div>
  <div class="lab"><div>target</div><div>synthetic v0</div><div>synthetic v1</div><div>TRELLIS raw · smooth</div><div>TRELLIS input · smooth</div><div>TRELLIS raw · blocky</div><div>TRELLIS input · blocky</div><div>photo · smooth</div><div>photo · blocky</div></div>
  <div class="info"><b>${r.id}</b> <span class="k">${r.name}</span> ${flags}<br>
  <span class="k">split</span> ${r.split} · <span class="k">category</span> ${r.cat} · <span class="k">target</span> ${r.dims ? r.dims.join("×") : "–"} blocks<br>${rt}
  <table><tr><th></th><th>status</th><th>IoU</th><th>frame residual (cells)</th><th>yaw</th><th>offset (blocks)</th><th>bottleneck</th><th>stair45</th><th>rough°</th><th>detail</th><th>colour</th></tr>${rows}</table></div></div>`;
}
function realCard(p) {
  const b = byId[p.id] || {img: {}};
  const credit = p.photo.page ? `<a href="${p.photo.page}">${p.photo.licence}, ${p.photo.author}</a>` : (p.photo.source || "");
  const why = p.status === "ok" ? `IoU ${f(p.iou, 3)} · scale ×${f(p.scale, 2)} · aspect ×${f(p.aspect, 2)} · yaw ${p.yaw}° · residual ${(p.res || []).map(x => f(x, 2)).join(" ")} ${p.fail && p.fail.length ? '<span class="tag bad">' + p.fail.join(", ") + "</span>" : ""}` : p.status;
  return `<div class="card"><div class="pics">${tile(p, "realphoto")}${tile(p, "raw")}${tile(p, "input")}${tile(b, "v1")}${tile(b, "target")}</div>
  <div class="lab"><div>real photo</div><div>TRELLIS raw</div><div>TRELLIS input</div><div>synthetic v1</div><div>target</div></div>
  <div class="info"><b>${p.subject}</b> <span class="k">${p.id}</span> <span class="tag ${p.test ? "ok" : ""}">${p.split}${p.test ? "" : " · not in test metrics"}</span>
  <span class="tag ${p.cutout ? "ok" : "bad"}">${p.cutout ? "cutout" : "cutout rejected: raw photo"} (mask ${f(p.mask, 2)})</span><br>
  ${why}<br>
  <table><tr><th></th><th>stair45</th><th>rough°</th><th>L*</th><th>C*</th><th>nbr ΔE</th><th>detail</th><th>colour</th><th>spurious</th></tr>
  <tr><td>real-photo TRELLIS</td><td>${f(p.stair, 2)}</td><td>${f(p.rough, 1)}</td><td>${f(p.lum, 1)}</td><td>${f(p.chroma, 1)}</td><td>${f(p.nbr, 2)}</td><td>${f(p.dg, 2)}</td><td>${f(p.dc, 2)}</td><td>${f(p.ds, 2)}</td></tr>
  <tr><td>synthetic v1</td><td>${f(p.syn.stair, 2)}</td><td>${f(p.syn.rough, 1)}</td><td>${f(p.syn.lum, 1)}</td><td>${f(p.syn.chroma, 1)}</td><td>${f(p.syn.nbr, 2)}</td><td>${f(p.syn.dg, 2)}</td><td>${f(p.syn.dc, 2)}</td><td>${f(p.syn.ds, 2)}</td></tr></table>
  <span class="k">colour cast vs the build (ΔL* Δa* Δb*):</span> ${p.cast.map(x => f(x, 1)).join(" ")} · <span class="k">photo:</span> ${credit}</div></div>`;
}
function draw() {
  const q = $("q").value.toLowerCase(), sp = $("split").value, ca = $("cat").value, se = $("sel").value, so = $("sort").value;
  let rows = R.filter(r => (!sp || r.split === sp) && (!ca || r.top === ca)
    && (!se || (se === "rt" ? Object.keys(r.rt).length : se === "low" ? r.flags.includes("low_iou") : r.variants.some(v => v.status !== "ok")))
    && (!q || (r.id + " " + r.name).toLowerCase().includes(q)));
  const key = {iou: r => r.min_iou, size: r => -r.size, name: r => r.id}[so];
  rows.sort((a, b) => { const x = key(a), y = key(b); return x < y ? -1 : x > y ? 1 : 0; });
  $("n").textContent = rows.length + " of " + R.length + " builds";
  $("grid").innerHTML = rows.map(card).join("");
  const real = P.filter(p => (!sp || p.split === sp) && (!q || (p.id + " " + p.subject).toLowerCase().includes(q)));
  $("nr").textContent = real.length;
  $("real").innerHTML = real.map(realCard).join("");
}
["q","split","cat","sel","sort"].forEach(id => $(id).addEventListener("input", draw));
document.body.addEventListener("click", e => { if (e.target.tagName === "IMG") {
  $("mimg").src = e.target.src; $("mtxt").textContent = e.target.dataset.cap; $("modal").style.display = "flex"; } });
draw();
</script></body></html>
"""


def _real_rows(dataset: Path) -> list:
    path = dataset / "pairs" / "real_photo_index.json"
    return json.loads(path.read_text(encoding="utf-8"))["builds"] if path.is_file() else []


def build_gallery(dataset: Path, only=None, workers: int = 1, force: bool = False) -> Path:
    from .real_photo import PHOTOS, build_dir, photo_path
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
        raws = {}
        for name, folder in EVAL_SETS:
            d = dataset / "pairs" / folder / bid
            if (d / "input.npz").is_file():
                files["rt_" + name] = f"{folder}/{bid}/input.npz"
                rt = rts.get(bid, {}).get(name, {})
                if (d / "trellis.glb").is_file() and rt.get("placement"):
                    raws["raw_" + name] = (str(d / "trellis.glb"), rt["placement"], f"{folder}/{bid}/input.npz")
        jobs.append((str(dataset), bid, files, str(out_dir / "img"), raws, {}))
    real = {r["id"]: r for r in _real_rows(dataset)}
    for p in PHOTOS:
        key = "real:" + p["id"]
        r = real.get(p["id"])
        if not (force or (only is not None and p["id"] in only) or key not in images):
            continue
        d = build_dir(dataset, p)
        files, raws, extra = {}, {}, {}
        ph = photo_path(dataset, p)
        if ph:
            extra["realphoto"] = str(ph)
        if r and r.get("status") == "ok":
            rel = f"{p['folder']}/{p['id']}/input.npz"
            files["input"] = rel
            raws["raw"] = (str(d / "trellis.glb"), r["placement"], rel)
        jobs.append((str(dataset), p["id"], files, str(out_dir / "img"), raws, extra, key))
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_job, jobs))
    else:
        results = [_run_job(j) for j in jobs]
    for key, res in results:
        images[key] = res["images"]
        for e in res["errors"]:
            print(f"  {key}: {e}", flush=True)
    for key in drop_stale(images, set(by_build) | {"real:" + p["id"] for p in PHOTOS}, out_dir / "img"):
        print(f"  dropped the gallery images of {key}: no longer in the pairs", flush=True)
    cache_path.write_text(json.dumps(images, indent=1) + "\n", encoding="utf-8", newline="\n")
    cards = [card_data(bid, ps, rows.get(bid, {}), rts.get(bid, {}), images.get(bid, {}))
             for bid, ps in sorted(by_build.items())]
    real_cards = [real_card_data(real.get(p["id"], {"id": p["id"], "subject": p["subject"], "status": "awaiting_trellis",
                                                    "split": rows.get(p["id"], {}).get("split")}),
                                 images.get("real:" + p["id"], {})) for p in PHOTOS]
    data = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    real_data = json.dumps(real_cards, ensure_ascii=False).replace("</", "<\\/")
    page = out_dir / "index.html"
    page.write_text(PAGE.replace("__DATA__", data).replace("__REAL__", real_data), encoding="utf-8", newline="\n")
    return page


def drop_stale(images: dict, live: set, img_dir: Path) -> list:
    """Remove cached cards of builds no longer in the pairs (set aside by D24, D25) from
    `images`, and their image files unless a live card uses the same file. Returns the keys."""
    stale = sorted(set(images) - live)
    keep = {f for k, v in images.items() if k in live for f in v.values() if isinstance(f, str)}
    for key in stale:
        for f in images.pop(key).values():
            if isinstance(f, str) and f not in keep:
                (Path(img_dir) / f).unlink(missing_ok=True)
    return stale


def _run_job(job):
    """(cache key, render_card result) for one job tuple."""
    if len(job) == 7:
        *args, key = job
        return key, render_card(*args)
    return job[1], render_card(*job)


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
