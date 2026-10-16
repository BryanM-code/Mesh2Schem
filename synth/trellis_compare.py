#!/usr/bin/env python3
"""Real TRELLIS vs synthetic degradation, on the SAME buildings.

Usage:  python -m synth.trellis_compare prepare <run dir> [--n 400] [--seed 1]
        python -m synth.trellis_compare generate <run dir> [--limit N]   (TRELLIS server on :7960)
        python -m synth.trellis_compare compare <run dir> [--trellis-dir DIR]

`--trellis-dir` reuses meshes generated for another run of the SAME sources (TRELLIS sees
the clean source, so its output does not depend on the pipeline version): pilot v3 is
compared against the meshes made for pilot v2, and pilot v2's own inputs can be compared
against them too, which measures a change of degradation settings pair by pair.

The synthetic inputs imitate TRELLIS; this experiment checks the imitation. For each chosen
pair it renders a photo-like image of the CLEAN source model, sends it through the real
TRELLIS server (the same API Real2Minecraft uses), puts the returned mesh into the pair's
voxel frame, and measures real TRELLIS and the synthetic degradation of the same building
with the same statistics:

    occupancy IoU vs the target, shell overlap     how much of the building survives
    fill gain, blobby shell (3+ open faces)        how melted / enclosed the geometry is
    brightness L*, colourfulness C*, colour noise  how the texture bakes
    colour error vs the clean model                how wrong the colours are

`compare` writes trellis_compare/report.json, per-pair stats and side-by-side sheets
(photo sent to TRELLIS / real TRELLIS voxels / synthetic input voxels / target). Paired numbers say
which way to move each degradation range in `PipelineConfig.degrade`; the unpaired check
against the 54 cached TRELLIS meshes (`synth.calibrate`) could only compare distributions.

Layout:  <run>/trellis_compare/<uid>/input.png     the render sent to TRELLIS (RGBA)
                                     trellis.glb   what TRELLIS returned
                                     stats.json    both inputs' statistics
         <run>/trellis_compare/selection.json, report.json, sheet.png

Cost: about 1-2 GPU-minutes per building on the RTX 4060 Ti, so 300-500 buildings is
roughly 6-15 hours. `generate` is resumable and skips buildings that already have a mesh.
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import os
import random
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

TRELLIS_URL = os.environ.get("R2M_TRELLIS_URL", "http://127.0.0.1:7960")
SAMPLING_STEPS = 24                   # as Real2Minecraft sends
MESH_SIMPLIFY = 0.9
TEXTURE_SIZE = 1024
VIEW_YAW, VIEW_PITCH = 35.0, 12.0     # a street-level three-quarter view, like a photo
IMAGE_SIZE = 768


# ── prepare ───────────────────────────────────────────────────────────────────

def choose(run: Path, n: int, seed: int):
    """Up to n ok pairs, round-robin over architectural groups (diversity first)."""
    by_group = collections.defaultdict(list)
    for p in sorted((run / "samples").glob("*/manifest.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("status") == "ok":
            by_group[m["source"]["group"]].append(m["uid"])
    rng = random.Random(seed)
    for uids in by_group.values():
        rng.shuffle(uids)
    chosen = []
    while len(chosen) < n and any(by_group.values()):
        for g in sorted(by_group):
            if by_group[g] and len(chosen) < n:
                chosen.append(by_group[g].pop())
    return chosen


def render_photo(scene, size=IMAGE_SIZE, yaw=VIEW_YAW, pitch=VIEW_PITCH, samples=3_000_000,
                 light=(0.55, 0.45)):
    """A photo-like RGBA render of a clean source model: textured, lit, transparent ground.
    `light` = (ambient, sun): colour x (ambient + sun * |normal . sun direction|).

    Point splatting at twice the size, then downsampled for antialiasing; the transparent
    background means the TRELLIS server does not run its own background removal (as with
    Real2Minecraft's cutouts).
    """
    from PIL import Image
    import thumbs
    from mcvoxel.scene import scene_color_fns
    from .preview import sample_surface

    _, point_colors = scene_color_fns(scene)
    pts, cols, normals = sample_surface(scene.verts, scene.faces, point_colors, samples, seed=1)
    right, down, near = (np.array(a) for a in thumbs._view_axes(yaw, pitch))
    sx, sy, depth = pts @ right, pts @ down, pts @ near
    sun = np.array([0.35, 0.85, 0.4])
    sun /= np.linalg.norm(sun)
    lam = np.clip(np.abs(normals @ sun), 0, 1)
    rgb = np.clip(cols * (light[0] + light[1] * lam)[:, None], 0, 255).astype(np.uint8)
    big = 2 * size
    span = max(float(np.ptp(sx)), float(np.ptp(sy))) or 1.0
    scale = (big * 0.86) / span
    px = np.rint((sx - sx.min()) * scale + (big - np.ptp(sx) * scale) / 2).astype(np.int64)
    py = np.rint((sy - sy.min()) * scale + (big - np.ptp(sy) * scale) / 2).astype(np.int64)
    canvas = np.zeros((big, big, 4), dtype=np.uint8)
    order = np.argsort(depth, kind="stable")              # far first, near last
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x, y = np.clip(px[order] + dx, 0, big - 1), np.clip(py[order] + dy, 0, big - 1)
        canvas[y, x, :3] = rgb[order]
        canvas[y, x, 3] = 255
    return Image.fromarray(canvas, "RGBA").resize((size, size), Image.LANCZOS)


def prepare(run: Path, n: int, seed: int, glb_cache=None):
    from mcvoxel.scene import load_scene
    from .orient import orient

    out = run / "trellis_compare"
    out.mkdir(parents=True, exist_ok=True)
    uids = choose(run, n, seed)
    sel = {r["uid"]: r for r in (json.loads(line) for line in
           (run / "selection.jsonl").read_text(encoding="utf-8").splitlines())}
    done = []
    for uid in uids:
        d = out / uid
        if (d / "input.png").exists():
            done.append(uid)
            continue
        glb, downloaded = _find_glb(run, uid, sel.get(uid), glb_cache)
        if glb is None:
            continue
        try:
            scene, _ = orient(load_scene(glb))
            d.mkdir(exist_ok=True)
            render_photo(scene).save(d / "input.png")
            done.append(uid)
        finally:
            if downloaded:                        # a re-downloaded raw GLB is not kept
                Path(glb).unlink(missing_ok=True)
    (out / "selection.json").write_text(json.dumps({"uids": done, "seed": seed, "view": [VIEW_YAW, VIEW_PITCH]},
                                                   indent=1), encoding="utf-8")
    return done


def _find_glb(run, uid, rec, glb_cache):
    """(source GLB, downloaded?): an existing cache (never deleted), else downloaded again
    into the run's temp folder (the caller deletes it once rendered)."""
    for folder in filter(None, [glb_cache]):
        p = Path(folder) / (uid + ".glb")
        if p.exists():
            return p, False
    if rec is None:
        return None, False
    from .config import PipelineConfig
    from .fetch import fetch_glb
    cfg = PipelineConfig.load(run / "run_config.json")
    dest = Path(cfg.tmp_dir) / (cfg.run_name + "-trellis") / (uid + ".glb")
    return fetch_glb(rec["path"], dest, sha256=rec.get("sha256") or None), True


# ── generate ──────────────────────────────────────────────────────────────────

def server_up(url=TRELLIS_URL) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/ping", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def trellis_generate(image_path: Path, out_path: Path, seed: int = 0, url=TRELLIS_URL):
    """POST the image to the TRELLIS bundle's API and save the returned GLB (R2M's contract)."""
    base = url.rstrip("/")
    body = urllib.parse.urlencode({
        "image_base64": base64.b64encode(Path(image_path).read_bytes()).decode(), "seed": str(seed),
        "ss_sampling_steps": str(SAMPLING_STEPS), "slat_sampling_steps": str(SAMPLING_STEPS),
        "mesh_simplify_ratio": str(MESH_SIMPLIFY), "texture_size": str(TEXTURE_SIZE), "output_format": "glb",
    }).encode()
    req = urllib.request.Request(base + "/generate_no_preview", data=body, method="POST")
    with urllib.request.urlopen(req, timeout=1800) as r:
        model_url = json.loads(r.read().decode()).get("model_url") or "/download/model"
    with urllib.request.urlopen(base + model_url, timeout=180) as r:
        data = r.read()
    if len(data) < 1024:
        raise RuntimeError("TRELLIS returned {} bytes".format(len(data)))
    out_path.write_bytes(data)
    return out_path


def generate(run: Path, limit: int = 0):
    out = run / "trellis_compare"
    uids = json.loads((out / "selection.json").read_text(encoding="utf-8"))["uids"]
    if not server_up():
        raise SystemExit("TRELLIS server not reachable at {} -- start C:\\TRELLIS\\run.bat first".format(TRELLIS_URL))
    todo = [u for u in uids if not (out / u / "trellis.glb").exists()]
    if limit:
        todo = todo[:limit]
    for i, uid in enumerate(todo, 1):
        trellis_generate(out / uid / "input.png", out / uid / "trellis.glb", seed=i)
        print("{}/{} {}".format(i, len(todo), uid), flush=True)


# ── compare ───────────────────────────────────────────────────────────────────

def voxel_stats(solid, shell, rgb):
    """The statistics `synth.calibrate` uses, on one voxelised input."""
    from .calibrate import voxel_stats as vs
    return vs(solid, shell, rgb)


def place(tscene, target_solid, origin, bpu_frame, yaws=(0, 90, 180, 270), color_mode="exposed"):
    """Fit a TRELLIS mesh into the pair's voxel frame: scale to the target's height, centre
    on its footprint, and keep the quarter-turn about +Y with the best occupancy IoU. The
    mesh is voxelised exactly as the pair's input was (same colour mode, same cleanup)."""
    from mcvoxel.grid import voxelize_scene

    ys = np.flatnonzero(target_solid.any(axis=(0, 2)))
    xs = np.flatnonzero(target_solid.any(axis=(1, 2)))
    zs = np.flatnonzero(target_solid.any(axis=(0, 1)))
    height = ys[-1] - ys[0] + 1
    centre = np.array([(xs[0] + xs[-1] + 1) / 2.0, 0.0, (zs[0] + zs[-1] + 1) / 2.0])
    best = None
    for yaw in yaws:
        t = np.radians(yaw)
        rot = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        v = tscene.verts @ rot.T
        lo, hi = v.min(axis=0), v.max(axis=0)
        s = height / max(hi[1] - lo[1], 1e-9)
        shift = centre - np.array([(lo[0] + hi[0]) / 2 * s, lo[1] * s - ys[0], (lo[2] + hi[2]) / 2 * s])
        moved = tscene.transformed(rot * s, shift + origin)
        vm = voxelize_scene(moved, 1.0, (0, 0, 0), origin=origin, shape=target_solid.shape, with_shapes=False,
                            crop=False, color_fallback="nearest", color_mode=color_mode)
        inter = int((vm.solid & target_solid).sum())
        iou = inter / max(int((vm.solid | target_solid).sum()), 1)
        if best is None or iou > best[0]:
            best = (iou, yaw, vm)
    return best


def target_colour_error(shell, rgb, target_shape, target_material):
    """Mean CIELAB distance from each input shell voxel's colour to the colour of the target
    block at that voxel -- the colour-to-material mapping the model has to learn."""
    from mcvoxel import vocab
    from mcvoxel.block_palette import AUX_PALETTE, BLOCK_PALETTE, srgb_to_lab

    solid = (target_shape != 0) & (target_shape != vocab.IGNORE_SHAPE)
    both = shell & solid
    if not both.any():
        return None
    table = np.array([BLOCK_PALETTE.get(m) or AUX_PALETTE.get(m) or (0, 0, 0) for m in vocab.MATERIALS], float)
    want = table[target_material[both].astype(np.int64)]
    return round(float(np.linalg.norm(srgb_to_lab(rgb[both].astype(float)) - srgb_to_lab(want), axis=1).mean()), 2)


def compare(run: Path, trellis_dir=None, sheet_rows: int = 8, sheet_pages: int = 4):
    from PIL import Image

    from mcvoxel.grid import shell_of
    from mcvoxel.scene import load_scene
    from . import preview

    out = run / "trellis_compare"
    tdir = Path(trellis_dir) if trellis_dir else out
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    # Runs before synth-2.0.0 sampled colours per voxel volume; voxelise TRELLIS the same way.
    color_mode = cfg.get("color_mode", "volume")
    uids = json.loads((tdir / "selection.json").read_text(encoding="utf-8"))["uids"]
    rows, tiles = [], []
    yzx_to_xyz = (2, 0, 1)
    for uid in uids:
        glb = tdir / uid / "trellis.glb"
        s = run / "samples" / uid
        if not glb.exists() or not (s / "input.npz").exists():
            continue
        tnpz = np.load(s / "target.npz")
        inp = np.load(s / "input.npz")
        target_shape = np.transpose(tnpz["shape"], yzx_to_xyz)
        target_material = np.transpose(tnpz["material"], yzx_to_xyz)
        target_solid = target_shape != 0
        origin = tnpz["origin"]
        iou, yaw, vm = place(load_scene(glb), target_solid, origin, float(tnpz["blocks_per_unit"]),
                             color_mode=color_mode)
        syn_solid = np.transpose(inp["occ"].astype(bool), yzx_to_xyz)
        syn_rgb = np.transpose(inp["rgb"], (2, 0, 1, 3))
        syn_shell = shell_of(syn_solid)
        man = json.loads((s / "manifest.json").read_text(encoding="utf-8"))
        row = {"uid": uid, "group": man["source"]["group"], "trellis_yaw": yaw}
        fill = {"trellis": vm.info.get("fill_gain"), "synthetic": (man.get("input_voxels") or {}).get("fill_gain")}
        for name, solid, shell, rgb in (("trellis", vm.solid, vm.shell, vm.rgb),
                                        ("synthetic", syn_solid, syn_shell, syn_rgb)):
            st = voxel_stats(solid, shell, rgb)
            st["fill_gain"] = fill[name]
            inter = int((solid & target_solid).sum())
            st["occupancy_iou"] = round(inter / max(int((solid | target_solid).sum()), 1), 4)
            st["delta_e_to_target_blocks"] = target_colour_error(shell, rgb, target_shape, target_material)
            row[name] = st
        (out / uid).mkdir(exist_ok=True)
        (out / uid / "stats.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
        rows.append(row)
        photo = Image.open(tdir / uid / "input.png").convert("RGBA").resize((preview.SIZE, preview.SIZE))
        target = s / "target.png"
        pair_tiles = [(photo, uid[:8] + " photo sent to TRELLIS"),
                      (preview.render_rgb(_yzx(vm.solid), _yzx(vm.rgb)),
                       "real TRELLIS, IoU {:.2f}".format(row["trellis"]["occupancy_iou"])),
                      (preview.render_rgb(inp["occ"], inp["rgb"]),
                       "synthetic input, IoU {:.2f}".format(row["synthetic"]["occupancy_iou"])),
                      (Image.open(target) if target.exists() else None, "target")]
        preview.sheet(pair_tiles, cols=4).save(out / uid / "comparison.png")
        if len(tiles) < 4 * sheet_rows * sheet_pages:
            tiles += pair_tiles
    keys = sorted({k for r in rows for k in r["trellis"]})
    summary = {k: {"trellis_p50": _p50([r["trellis"][k] for r in rows]),
                   "synthetic_p50": _p50([r["synthetic"][k] for r in rows]),
                   "paired_median_diff": _p50([r["synthetic"][k] - r["trellis"][k] for r in rows
                                               if r["synthetic"][k] is not None and r["trellis"][k] is not None])}
               for k in keys}
    report = {"pairs": len(rows), "color_mode": color_mode, "trellis_dir": str(tdir), "summary": summary,
              "uids": [r["uid"] for r in rows],
              "limitations": ["TRELLIS inputs are source-model renders, not real photographs.",
                              "Height and quarter-turn alignment use target geometry; IoU is optimistic.",
                              "Paired calibration is diagnostic, not held-out validation or realism scoring."],
              "reading": "paired_median_diff > 0: the synthetic input has more of it than real TRELLIS"}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    per_page = 4 * sheet_rows
    for i in range(0, len(tiles), per_page):
        preview.sheet(tiles[i:i + per_page], cols=4).save(out / "sheet_{:02d}.png".format(i // per_page + 1))
    return report


def _yzx(a):
    return np.ascontiguousarray(a.transpose((1, 2, 0) + tuple(range(3, a.ndim))))


def _p50(values):
    v = [x for x in values if x is not None]
    return round(float(np.median(v)), 4) if v else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("run")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--glb-cache", help="folder of <uid>.glb already on disk")
    g = sub.add_parser("generate")
    g.add_argument("run")
    g.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("run")
    c.add_argument("--trellis-dir", help="trellis_compare folder of another run of the same sources")
    args = ap.parse_args(argv)
    run = Path(args.run)
    if args.cmd == "prepare":
        done = prepare(run, args.n, args.seed, args.glb_cache)
        print("{} images ready in {}".format(len(done), run / "trellis_compare"))
    elif args.cmd == "generate":
        generate(run, args.limit)
    else:
        rep = compare(run, args.trellis_dir)
        print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
