"""One source model -> one training pair on disk, or a recorded rejection.

Stages, cheapest first (metadata filtering happened at selection, before download):

    metadata      synth.filters.metadata_verdict, re-checked     (a selection made under older
                  rules is held to the current ones: object_or_set, not_architecture, ...)
    load          mcvoxel.scene.load_scene                      load_failed, too_many_faces
    clutter       mcvoxel.scene.split_clutter: vegetation, vehicles and figures dropped by
                  part / material name, from the target AND the input
    checks        synth.filters.geometry_verdict                blocky_geometry, scan_geometry
    orient        synth.orient.orient (yaw to the grid)
    scale         synth.scale.choose_scale (1 block = 1 m)      scale_broken, too_large
                  (uncertain scales are kept and flagged, see `flags` below)
    target        synth.target.make_target (both roof styles)   degenerate_size, scattered,
                                                                open_model, too_many_voxels
    degrade       synth.degrade.degrade (TRELLIS-like input)    degrade_failed
    input         the degraded mesh voxelised in the target's frame
    validate      synth.validate.check_pair                     validation_failed
    write         .schem (read back and compared), .npz, manifest
    render        synth.preview panels (a failed render never fails a pair)

    samples/<uid>/
      target.schem            primary target (builder roofs by default), Sponge v2, DataVersion
                              3465, provenance in Metadata, stair corners set by the game's rule
      target_geometric.schem  the other roof style (Approach A), when stored
      target.npz              shape (uint8) + material (uint16) grids, Sponge (Y, Z, X) order,
                              255 / 65535 = interior (ignored), 0 = air (supervised); the other
                              style as shape_geometric / material_geometric
      input.npz               occ, surf (uint8), rgb (Y, Z, X, 3 uint8): the degraded model
      manifest.json           provenance, licence, every sampled parameter, QA, metrics, and
                              `flags`: scale_uncertain / height_capped / size_capped /
                              small_building --
                              kept pairs a training run may weight or hold out
      *.png                   review panels (synth.preview)

A rejected asset gets the same folder with only manifest.json (status "rejected", stage,
reason, detail) and, when the mesh loaded, source.png -- so every rejection can be checked.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

import schem_voxels
from mcvoxel import vocab
from mcvoxel.block_palette import srgb_to_lab
from mcvoxel.blockstates import finish_states
from mcvoxel.grid import voxelize_scene
from mcvoxel.scene import KIND_DOOR, KIND_GLASS, SceneTooLarge, load_scene, split_clutter
from mcvoxel.tiling import window_starts
from schem_write import write_schem

from . import preview, taxonomy
from .config import PIPELINE_VERSION
from .degrade import degrade, sample_params
from .filters import geometry_verdict, metadata_verdict, source_warnings
from .orient import orient
from .scale import ScaleRejected, choose_scale
from .target import TargetRejected, make_target
from .validate import check_pair

TARGET_DATA_VERSION = 3465          # Minecraft 1.20.1: every block in the v1 vocabulary exists
LICENCE_NAMES = {"by": "CC-BY-4.0", "cc0": "CC0-1.0"}


class Rejected(Exception):
    """The asset cannot make a trustworthy pair. `reason` is a code from filters.REASONS."""

    def __init__(self, stage, reason, detail="", info=None):
        super().__init__("{}/{}: {}".format(stage, reason, detail))
        self.stage, self.reason, self.detail, self.info = stage, reason, detail, info or {}


def _yzx(a):
    return np.ascontiguousarray(a.transpose((1, 2, 0) + tuple(range(3, a.ndim))))


def attribution(rec):
    return '"{}" by {} ({}), {}. Voxelised and converted to Minecraft blocks.'.format(
        rec.get("name") or rec.get("uid"), rec.get("author"), rec.get("url"),
        LICENCE_NAMES.get(rec.get("license"), rec.get("license")))


def source_block(rec):
    sub = rec.get("subcategory") or taxonomy.subcategory(rec.get("category"), rec.get("caption") or rec.get("name"))
    return {"dataset": "objaverse-1.0", "uid": rec.get("uid"), "url": rec.get("url"),
            "author": rec.get("author"), "licence": LICENCE_NAMES.get(rec.get("license"), rec.get("license")),
            "name": rec.get("name"), "caption": rec.get("caption"), "sha256": rec.get("sha256"),
            "category": rec.get("category"), "house_sub": rec.get("house_sub"), "subcategory": sub,
            "group": taxonomy.group(rec.get("category"), sub), "bucket": rec.get("bucket"),
            "attribution": attribution(rec), "review_warnings": source_warnings(rec)}


def shape_counts(shape):
    """{vocab shape name: count} over supervised solid voxels of a (X,Y,Z) shape grid."""
    solid = (shape != 0) & (shape != vocab.IGNORE_SHAPE)
    idx, n = np.unique(shape[solid], return_counts=True)
    return {vocab.SHAPES[int(i)]: int(c) for i, c in zip(idx, n)}


def material_counts(shape, material, top=12):
    solid = (shape != 0) & (shape != vocab.IGNORE_SHAPE)
    idx, n = np.unique(material[solid], return_counts=True)
    order = np.argsort(-n)
    return [[vocab.MATERIALS[int(idx[i])].replace("minecraft:", ""), int(n[i])] for i in order[:top]]


def resolution(grid_xyz, windows, overlap):
    out = {"grid_xyz": [int(v) for v in grid_xyz], "tiles": {}, "fits_one_window": {}}
    for w in windows:
        n = int(np.prod([len(window_starts(int(d), int(w), int(overlap))) for d in grid_xyz]))
        out["tiles"][str(w)] = n
        out["fits_one_window"][str(w)] = bool(max(grid_xyz) <= w)
    return out


def flags(scale_info: dict, dims: dict) -> dict:
    """Kept-but-noteworthy properties of a pair (never rejections)."""
    return {"scale_uncertain": bool(scale_info.get("scale_uncertain")),
            "scale_confidence": scale_info.get("confidence"),
            "scale_source": scale_info.get("source"),
            "height_capped": bool(scale_info.get("height_capped")),
            "size_capped": bool(scale_info.get("size_capped")),
            "small_building": bool(dims.get("small_building"))}


def _write_target(tgt, vm, path, meta):
    """Sponge grid of a BlockTarget -> .schem, read back and compared. Returns (grid, names)."""
    grid, names = tgt.sponge_grid(vm.solid.shape)
    grid, names = finish_states(grid, names)
    for state in names:
        vocab.validate_state(state, TARGET_DATA_VERSION)
    write_schem(grid, names, path, metadata=meta, data_version=TARGET_DATA_VERSION)
    back = schem_voxels.load_voxels(path)
    if back is None or back.grid.shape != grid.shape:
        raise Rejected("write", "schem_roundtrip", "{} does not decode".format(Path(path).name))
    want = np.array(names, dtype=object)[grid]
    got = np.array(back.names, dtype=object)[back.grid]
    if not bool((want == got).all()):
        raise Rejected("write", "schem_roundtrip", "{} blocks differ after write".format(Path(path).name))
    return grid, names


def make_pair(rec: dict, glb_path, out_dir, cfg, seed: int) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = {}
    man = {"uid": rec["uid"], "status": "started", "pipeline_version": PIPELINE_VERSION,
           "config_hash": cfg.hash(), "vocab_version": vocab.VOCAB_VERSION, "seed": seed,
           "source": source_block(rec)}
    scene = None

    def reject(stage, reason, detail="", info=None):
        """Render what was rejected (when there is a mesh), then raise."""
        if scene is not None:
            try:
                im = preview.render_scene(scene, size=cfg.render_size)
                if im is not None:
                    im.save(out_dir / "source.png")
            except Exception:
                pass
        raise Rejected(stage, reason, detail, {**man, **(info or {})})

    # ── metadata, re-checked: a selection made under older rules meets the current ones ──
    reason, detail = metadata_verdict(rec, cfg)
    if reason:
        raise Rejected("metadata", reason, detail, man)

    # ── load and cheap geometry checks ────────────────────────────────────────
    t0 = time.time()
    try:
        scene = load_scene(glb_path, max_faces=cfg.max_load_faces)
    except SceneTooLarge as exc:
        raise Rejected("load", "too_many_faces", str(exc)[:200], man)
    except Exception as exc:
        raise Rejected("load", "load_failed", str(exc)[:200], man)
    if cfg.drop_clutter:
        scene, man["clutter"] = split_clutter(scene)
    kinds = np.bincount(scene.face_kind, minlength=3)
    man["scene"] = {"faces": int(len(scene.faces)), "materials": len(scene.materials),
                    "textured_materials": sum(m.image is not None for m in scene.materials),
                    "glass_faces": int(kinds[KIND_GLASS]), "door_faces": int(kinds[KIND_DOOR])}
    reason, detail, man["checks"] = geometry_verdict(scene, cfg)
    t["load_checks"] = time.time() - t0
    if reason:
        reject("checks", reason, detail)

    # ── orient and scale ──────────────────────────────────────────────────────
    t0 = time.time()
    scene, man["orient"] = orient(scene)
    try:
        bpu, man["scale"] = choose_scale(rec, scene, cfg, rng)
    except ScaleRejected as exc:
        man["scale"] = exc.info
        reject("scale", exc.reason, exc.detail)
    lo, _ = scene.bounds
    jitter = rng.uniform(0.0, 1.0, size=3)
    jitter[1] = 0.0                                     # ground sits on a block boundary
    offset = -lo * bpu + jitter
    man["scale"]["offset_jitter"] = [round(float(v), 3) for v in jitter]
    if cfg.grid_alignment == 'geometry':
        from mcvoxel.architecture import align_grid
        offset, man['scale']['alignment'] = align_grid(scene.verts, scene.faces, bpu)
    elif cfg.grid_alignment != 'jitter':
        raise ValueError('unknown grid_alignment: '+cfg.grid_alignment)
    # Store full precision for exact replay; offset_jitter remains legacy provenance.
    man['scale']['offset_blocks'] = [float(v) for v in offset]
    man['representation_version'] = 'architecture-v1' if cfg.architecture_mode == 'coherent' else 'legacy'
    # Objaverse++ marks models that stand on a modelled plot; only those may lose more than a
    # 3-layer base, so a real podium on any other model survives.
    max_plate = 0.35 if rec.get("pool_kind") == "plot" else 3
    man["scale"]["max_plate"] = max_plate
    t["orient_scale"] = time.time() - t0

    # ── target ────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        # An explicit source description can resolve an otherwise unnamed textured
        # material. It never supplies geometry or overrides named source materials.
        caption = rec.get("caption") or ""
        detail_hint = "wood" if re.search(r"wooden (?:shop|house|walls|posts|supports)|timber (?:posts|frame)", caption, re.I) else None
        tr = make_target(scene, bpu, offset, cfg, max_plate, scale_info=man["scale"], detail_material_hint=detail_hint)
    except TargetRejected as exc:
        man["target"] = exc.info
        reject("target", exc.reason, exc.detail)
    t["target"] = time.time() - t0
    man["target"] = {**tr.info, "roof_style": cfg.roof_style}
    man["flags"] = flags(man["scale"], tr.info["dims"])
    primary, vm = tr.primary(cfg.roof_style)
    other_style = "geometric" if cfg.roof_style == "builder" else "builder"
    other, vm_other = tr.primary(other_style)

    # ── degraded input, in the target's frame ─────────────────────────────────
    t0 = time.time()
    params = sample_params(rng, cfg.degrade)
    try:
        deg, man["degrade"] = degrade(scene, params)
    except Exception as exc:
        reject("degrade", "degrade_failed", str(exc)[:200])
    vin = voxelize_scene(deg, bpu, offset, origin=vm.origin, shape=vm.solid.shape, with_shapes=False,
                         crop=False, max_plate=max_plate, fine_scale=cfg.fine_scale,
                         architecture=cfg.architecture_mode == 'coherent')
    man["input_voxels"] = vin.info
    t["input"] = time.time() - t0

    # ── metrics and validation ────────────────────────────────────────────────
    tgt_solid = primary.shape != 0                       # includes the ignored interior
    inter, union = int((vin.solid & tgt_solid).sum()), int((vin.solid | tgt_solid).sum())
    metrics = {"occupancy_iou": round(inter / max(union, 1), 4)}
    both = vin.shell & vm.shell
    metrics["shell_overlap"] = round(int(both.sum()) / max(int(vm.shell.sum()), 1), 4)
    if both.any():
        clean = vm.rgb[both].astype(np.float64)
        noisy = vin.rgb[both].astype(np.float64)
        metrics["shell_delta_e"] = round(float(np.linalg.norm(srgb_to_lab(clean) - srgb_to_lab(noisy), axis=1).mean()), 2)
    sup = primary.shape != vocab.IGNORE_SHAPE
    metrics["supervised_voxels"] = int(sup.sum())
    metrics["supervised_solid"] = int((sup & tgt_solid).sum())
    man["metrics"] = metrics
    targets = {cfg.roof_style: (primary.shape, primary.material)}
    if cfg.store_geometric:
        targets[other_style] = (other.shape, other.material)
    failures = check_pair(targets, vin.solid, metrics["occupancy_iou"])
    man["validation"] = {"failures": failures}
    if failures:
        reject("validate", "validation_failed", "; ".join(failures)[:300])

    # ── write ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    meta = {"generator": "BuildDataCollector synth " + PIPELINE_VERSION, "source_dataset": "objaverse-1.0",
            "source_uid": rec["uid"], "source_url": rec.get("url"), "source_author": rec.get("author"),
            "licence": man["source"]["licence"], "attribution": man["source"]["attribution"],
            "category": rec.get("category"), "blocks_per_metre": cfg.blocks_per_metre,
            "blocks_per_unit": round(float(bpu), 6), "seed": seed, "config_hash": cfg.hash()}
    grid_p, names_p = _write_target(primary, vm, out_dir / "target.schem", {**meta, "roof_style": cfg.roof_style})
    from .roof_metrics import measure_roof
    man["roof_topology"] = measure_roof(grid_p, names_p, tr.plan)
    man["validation"]["schematic_roundtrip"] = True
    man["validation"]["blockstates_data_version"] = TARGET_DATA_VERSION
    np.savez_compressed(out_dir / "roof_plan.npz", **{k: getattr(tr.plan,k) for k in
                        ("cls", "facing", "level", "shape", "region", "group")})
    if cfg.architecture_mode == 'coherent':
        from .architecture_metrics import measure
        man['architecture_diagnostics'] = measure(scene.verts*bpu+offset-vm.origin, scene.faces,
                                                  grid_p, names_p, tr.plan, vm)
        man['architecture_diagnostics']['visual_approval'] = 'not_established'
        for i, layer in enumerate(tr.plan.layers):
            np.savez_compressed(out_dir / ('roof_plan_layer_%d.npz' % (i+1)), **{k: getattr(layer,k) for k in
                                ('cls','facing','level','shape','region','group')})
    arrays = {"shape": _yzx(primary.shape), "material": _yzx(primary.material)}
    grid_o = names_o = None
    if cfg.store_geometric:
        grid_o, names_o = _write_target(other, vm_other, out_dir / "target_{}.schem".format(other_style),
                                        {**meta, "roof_style": other_style})
        arrays["shape_" + other_style] = _yzx(other.shape)
        arrays["material_" + other_style] = _yzx(other.material)
    np.savez_compressed(out_dir / "target.npz", origin=vm.origin, blocks_per_unit=np.float64(bpu), **arrays)
    np.savez_compressed(out_dir / "input.npz", occ=_yzx(vin.solid).astype(np.uint8),
                        surf=_yzx(vin.surface).astype(np.uint8), rgb=_yzx(vin.rgb))
    t["write"] = time.time() - t0

    man["blocks"] = {cfg.roof_style: {"shapes": shape_counts(primary.shape),
                                      "materials": material_counts(primary.shape, primary.material),
                                      "distinct_states": len(names_p) - 1}}
    if cfg.store_geometric:
        man["blocks"][other_style] = {"shapes": shape_counts(other.shape),
                                      "materials": material_counts(other.shape, other.material),
                                      "distinct_states": len(names_o) - 1}
    man["resolution"] = resolution(vm.solid.shape, cfg.windows, cfg.window_overlap)

    # ── review panels ─────────────────────────────────────────────────────────
    t0 = time.time()
    verts_frame = scene.verts * bpu + offset - vm.origin
    deg_frame = deg.verts * bpu + offset - vm.origin
    panels = {}
    jobs = {
        "source": lambda: preview.render_scene(scene, verts_frame, size=cfg.render_size),
        "degraded": lambda: preview.render_scene(deg, deg_frame, size=cfg.render_size),
        "input": lambda: preview.render_rgb(_yzx(vin.solid), _yzx(vin.rgb), size=cfg.render_size),
        "shapes": lambda: preview.render_shapes(grid_p, names_p, size=cfg.render_size),
        "target": lambda: preview.render_blocks(grid_p, names_p, size=cfg.render_size),
        "target_back": lambda: preview.render_blocks(grid_p, names_p, yaw=preview.YAW + 180, size=cfg.render_size),
        "roof_regions": lambda: preview.render_roof_regions(tr.plan, size=cfg.render_size),
    }
    if cfg.store_geometric:
        jobs["target_" + other_style] = lambda: preview.render_blocks(grid_o, names_o, size=cfg.render_size)
    errors = {}
    if cfg.renders == "none":
        jobs = {}
    for name, job in jobs.items():
        try:
            im = job()
            if im is not None and cfg.renders == "all":
                im.save(out_dir / (name + ".png"))
            panels[name] = im
        except Exception as exc:                         # a preview never fails a pair
            errors[name] = str(exc)[:200]
            panels[name] = None
    if cfg.renders != "none":
        try:
            s = man["scale"]
            dims = man["target"]["dims"]
            title = man["source"]["name"] or rec["uid"]
            if not title.isascii():                         # the default PIL font has no CJK glyphs
                title = "(non-Latin title) " + rec["uid"][:8]
            raised = [k for k, v in man["flags"].items() if v is True]
            info = [title, "{} / {}".format(man["source"]["category"], man["source"]["group"]),
                    "{} by {}".format(man["source"]["licence"], man["source"]["author"]),
                    "", "height {:.1f} m  ({}, {})".format(s["height_m"], s["source"], s["confidence"]),
                    "flags: " + (", ".join(raised) if raised else "none"),
                    "{} x {} x {} blocks (fp x h x w)".format(dims["footprint_blocks"], dims["height_blocks"], dims["width_blocks"]),
                    "grid {}".format(" x ".join(str(v) for v in man["resolution"]["grid_xyz"])),
                    "64^3 tiles: {}".format(man["resolution"]["tiles"].get("64")),
                    "", "IoU input/target {:.2f}".format(metrics["occupancy_iou"]),
                    "roof caps planned {}".format(man["target"]["roofs"].get("planned")),
                    "shapes: " + ", ".join("{} {}".format(k, v) for k, v in
                                           sorted(man["blocks"][cfg.roof_style]["shapes"].items(), key=lambda kv: -kv[1])[:4])]
            order = [("source", "source mesh"), ("degraded", "degraded mesh (input)"),
                     ("input", "input voxels"), ("target", "target: {} roofs".format(cfg.roof_style)),
                     ("target_back", "target, back"),
                     ("target_" + other_style, "target: {} roofs".format(other_style)),
                     ("shapes", "target shapes")]
            tiles = [(panels.get(k), label) for k, label in order if k in panels]
            tiles.append((preview.info_tile(info, cfg.render_size), "metadata"))
            preview.sheet(tiles, cols=4, size=cfg.render_size).save(out_dir / "review.png")
        except Exception as exc:
            errors["review"] = str(exc)[:200]
    man["renders"] = {"mode": cfg.renders, "errors": errors}
    t["render"] = time.time() - t0

    man["files"] = {p.name: p.stat().st_size for p in sorted(out_dir.iterdir()) if p.is_file()}
    man["timings_s"] = {k: round(v, 2) for k, v in t.items()}
    man["status"] = "ok"
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=_json_default), encoding="utf-8")
    return man


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)
