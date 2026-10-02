#!/usr/bin/env python3
"""Show every stage of mesh -> voxels -> blocks for one source, to see where quality is lost.

Usage:  python -m synth.voxel_debug <uid ...> [--run data/synthetic/pilot-v2] [--out DIR]
                                    [--glb-cache data/synthetic/pilot-v1/glb]

For each source it writes <out>/<uid>.png: one panel per stage, all on the same camera,

    original mesh        the GLB as loaded, its own colours
    normalised mesh      oriented to the grid and scaled to blocks (1 block = 1 m)
    surface voxels       the rasterised surface
    flood fill (cut)     solid after the exterior flood fill, cut at mid-height
    enclosed rooms (cut) + voxels the occlusion fill added (blue)
    ground plate         voxels removed as the plot / plate (red)
    cleanup              voxels the closing added (blue) and keep-largest removed (red)
    slivers              shell voxels removed as boundary slivers (red)
    shapes               the geometric target coloured by construction
    blocks, geometric    the Minecraft blocks, Approach A
    blocks, builder      the Minecraft blocks, Approach B (builder roofs)

and <out>/<uid>.json with the voxel count after every stage. The voxeliser is run exactly as
the pipeline runs it (`mcvoxel.grid.voxelize_scene(..., trace=...)`), with the scale the run
recorded for that source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEEP, ADD, REMOVE = "dbg:keep", "dbg:add", "dbg:remove"
LUT = {KEEP: (178, 178, 184), ADD: (70, 140, 235), REMOVE: (228, 64, 56)}


def _grid(keep, add=None, remove=None):
    """(Y,Z,X) label grid + names from (X,Y,Z) masks: keep grey, add blue, remove red."""
    names = ["minecraft:air", KEEP, ADD, REMOVE]
    g = np.zeros(keep.shape, dtype=np.uint16)
    g[keep] = 1
    if add is not None:
        g[add] = 2
    if remove is not None:
        g[remove & ~keep] = 3
    return np.ascontiguousarray(g.transpose(1, 2, 0)), names


def _render(keep, add=None, remove=None, cut=False, size=300, yaw=40.0, pitch=30.0):
    import thumbs
    from schem_voxels import VoxelGrid
    from .preview import _colour_names

    grid, names = _grid(keep, add, remove)
    if not grid.any():
        return None
    cut_at = None
    if cut:
        ys = np.flatnonzero(grid.any(axis=(1, 2)))
        cut_at = int((ys[0] + ys[-1]) // 2)
    with _colour_names(LUT):
        return thumbs._render_orbit(VoxelGrid(grid, names), yaw, pitch, cut_at, size, 300_000)


def find_glb(uid, caches):
    for c in caches:
        p = Path(c) / (uid + ".glb")
        if p.exists():
            return p, False
    return None, False


def debug_source(uid, run: Path, caches, out: Path, size=300, new_pipeline: bool = True,
                 recorded: bool = False):
    """Render every stage for one source. `new_pipeline` runs the current defaults
    (PipelineConfig()); otherwise the run's own run_config.json and recorded scale."""
    from mcvoxel import roofs
    from mcvoxel.assign import assign_blocks
    from mcvoxel.blockstates import finish_states
    from mcvoxel.grid import pad_model, voxelize_scene
    from mcvoxel.scene import load_scene, split_clutter
    from . import preview
    from .config import PipelineConfig
    from .orient import orient
    from .scale import choose_scale
    from .target import palette_ids, site_height_blocks

    run_cfg = run / "run_config.json"
    cfg = PipelineConfig() if (new_pipeline and not recorded) or not run_cfg.exists() else PipelineConfig.load(run_cfg)
    sel = {}
    if (run / "selection.jsonl").exists():
        sel = {r["uid"]: r for r in (json.loads(l) for l in
               (run / "selection.jsonl").read_text(encoding="utf-8").splitlines())}
    rec = sel.get(uid, {})
    glb, downloaded = find_glb(uid, caches)
    if glb is None:
        if not rec:
            raise SystemExit("no GLB for {} (give --glb-cache or a run whose selection has it)".format(uid))
        from .fetch import fetch_glb
        glb = fetch_glb(rec["path"], Path(cfg.tmp_dir) / "debug" / (uid + ".glb"))
        downloaded = True
    raw = load_scene(glb)
    building, clutter = split_clutter(raw) if (new_pipeline or recorded) and cfg.drop_clutter else (raw, {"dropped": 0})
    scene, _ = orient(building)
    man_path = run / "samples" / uid / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    sc = man.get("scale") or {}
    jitter = np.array(sc.get("offset_jitter", [0.0, 0.0, 0.0]))
    max_plate = 0.35 if rec.get("pool_kind") == "plot" else sc.get("max_plate", 3)
    if recorded:
        from .run import seed_for
        rng = np.random.default_rng(seed_for(uid, cfg.seed))
        bpu, sc_info = choose_scale(rec, scene, cfg, rng)
        jitter = rng.uniform(0.0, 1.0, size=3)
        jitter[1] = 0.0
    elif new_pipeline or not sc.get("blocks_per_unit"):
        bpu, sc_info = choose_scale(rec or {"category": (man.get("source") or {}).get("category")},
                                    scene, cfg, np.random.default_rng(0))
    else:
        bpu, sc_info = float(sc["blocks_per_unit"]), sc
    lo, _ = scene.bounds
    offset = -lo * bpu + jitter

    trace = {}
    kw = {"site_height": site_height_blocks(cfg, sc_info), "site_margin": cfg.site_margin,
          "color_mode": cfg.color_mode, "clean": cfg.close_gaps} if new_pipeline or recorded else {"color_mode": "volume"}
    vm = voxelize_scene(scene, bpu, offset, fine_scale=cfg.fine_scale, max_plate=max_plate, trace=trace, **kw)
    c_lo = trace.get("crop_lo", np.zeros(3, int))

    # Keep the full frame for stage differences. Cropping first used to hide precisely
    # the site/ground/component removals this audit is meant to reveal.
    surf, flood, enclosed = trace["surface"], trace["flood"], trace["enclosed"]
    filled = flood | enclosed
    ground = trace["after_ground"]
    sited = trace.get("after_site", trace["after_ground"])
    cleaned = trace["after_clean"]
    slivers = trace.get("slivers", np.zeros_like(vm.solid))
    counts = {"clutter_faces_dropped": clutter.get("dropped", 0), "scale": {k: sc_info.get(k) for k in (
                  "source", "unit", "height_m", "confidence", "scale_uncertain")},
              "surface": int(surf.sum()), "flood": int(flood.sum()), "enclosed_added": int(enclosed.sum()),
              "ground_removed": int((filled & ~ground).sum()), "site_removed": int((ground & ~sited).sum()),
              "clean_added": int((cleaned & ~sited).sum()), "clean_removed": int((sited & ~cleaned).sum()),
              "slivers_removed": int(slivers.sum()), "final_solid": int(vm.solid.sum()),
              "final_shell": int(vm.shell.sum()), "seen_shell": vm.info.get("seen_shell"),
              "grid_xyz": list(vm.solid.shape), "blocks_per_unit": bpu,
              "config_hash": cfg.hash(), "recorded_settings": recorded,
              "stage_counts_match_manifest": (None if not recorded or man.get("status") != "ok" else
                  all(vm.info.get(k) == man["target"]["voxels"].get(k) for k in
                      ("surface_voxels", "flood_voxels", "occlusion_filled", "slivers_removed", "shell_voxels")))}

    m = cfg.margin
    vmp = pad_model(vm, ((m, m), (0, m), (m, m)))
    vb, plan = roofs.builder_roofs(vmp, scene.verts * bpu + offset - vmp.origin, scene.faces,
                                 roofs.RoofParams(sub=cfg.fine_scale), scene=scene)
    pal = palette_ids(cfg) if new_pipeline or recorded else None
    akw = {"palette": pal, "quantize": cfg.quantize, "shadow_repaint": cfg.shadow_repaint} if new_pipeline or recorded else {}
    ta, tb = assign_blocks(vmp, **akw), assign_blocks(vb, **akw)
    roofs.harmonize_roof_materials(tb, plan, palette=pal, rgb=vb.rgb if new_pipeline or recorded else None)
    ga, na = finish_states(*ta.sponge_grid(vmp.solid.shape))
    gb, nb = finish_states(*tb.sponge_grid(vb.solid.shape))

    panels = [
        (preview.render_scene(raw, size=size), "original mesh"),
        (preview.render_scene(scene, scene.verts * bpu + offset - np.asarray(trace["frame_origin"]) - c_lo, size=size),
         "normalised ({:.3f} blocks/unit)".format(bpu)),
        (_render(surf, size=size), "surface voxels ({:,})".format(counts["surface"])),
        (_render(flood, cut=True, size=size), "flood fill, cut ({:,})".format(counts["flood"])),
        (_render(flood, add=enclosed, cut=True, size=size), "enclosed rooms +{:,} (blue)".format(counts["enclosed_added"])),
        (_render(ground, remove=filled & ~ground, size=size), "ground plate -{:,} (red)".format(counts["ground_removed"])),
        (_render(sited, remove=ground & ~sited, size=size), "site -{:,} (red)".format(counts["site_removed"])),
        (_render(cleaned & sited, add=cleaned & ~sited, remove=sited & ~cleaned, size=size),
         "cleanup +{:,} / -{:,}".format(counts["clean_added"], counts["clean_removed"])),
        (_render(vm.solid, remove=slivers, size=size), "slivers -{:,} (red)".format(counts["slivers_removed"])),
        (preview.render_rgb(vm.shell.transpose(1, 2, 0), vm.rgb.transpose(1, 2, 0, 3), size=size),
         "surface colours before block mapping"),
        (preview.render_shapes(ga, na, size=size), "shapes (geometric)"),
        (preview.render_blocks(ga, na, size=size), "blocks, geometric"),
        (preview.render_blocks(gb, nb, size=size), "blocks, builder roofs"),
        (preview.render_blocks(gb, nb, yaw=220.0, size=size), "blocks, builder (back)"),
    ]
    out.mkdir(parents=True, exist_ok=True)
    preview.sheet(panels, cols=6, size=size).save(out / (uid + ".png"))
    (out / (uid + ".json")).write_text(json.dumps(counts, indent=1), encoding="utf-8")
    if downloaded:
        Path(glb).unlink(missing_ok=True)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("uids", nargs="+")
    ap.add_argument("--run", default="data/synthetic/pilot-v2")
    ap.add_argument("--glb-cache", action="append", default=["data/synthetic/pilot-v1/glb"])
    ap.add_argument("--out", default=None)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--old", action="store_true", help="legacy settings with CURRENT voxel code; not an exact historical replay")
    mode.add_argument("--recorded", action="store_true", help="replay the run's saved settings and random seed exactly")
    args = ap.parse_args(argv)
    run = Path(args.run)
    out = Path(args.out) if args.out else run / "voxel_debug"
    full = {p.name[:len(u)]: p.name for u in args.uids for p in (run / "samples").glob(u + "*")}
    for u in args.uids:
        uid = full.get(u, u)
        counts = debug_source(uid, run, args.glb_cache, out, new_pipeline=not args.old, recorded=args.recorded)
        print(uid, counts)


if __name__ == "__main__":
    main()
