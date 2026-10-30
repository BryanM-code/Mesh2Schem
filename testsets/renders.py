#!/usr/bin/env python3
"""Test sets: held-out builds through the real TRELLIS server.

Usage:  python -m testsets.renders prepare  --dataset data/real-schems-v1 --set smooth [--workers N]
        python -m testsets.renders generate --dataset data/real-schems-v1 --set smooth
        python -m testsets.renders place    --dataset data/real-schems-v1 --set smooth [--workers N] [--resume]
        python -m testsets.renders trials   --dataset data/real-schems-v1 --sigmas 0.6 0.8 1.0
        python -m testsets.renders status   --dataset data/real-schems-v1

The synthetic inputs imitate TRELLIS on photos of real buildings; these sets check the
imitation on held-out builds and are what a model is scored on for real TRELLIS geometry.
The evaluation sets use only val and test builds -- never train. Two photo sets of the
same 25 builds, and one set of training inputs:

    smooth   the main set (`real_trellis_smooth/`). The photo is rendered from a SMOOTHED
             version of the build: its outside (`pairs.meshify`, glass dark like real
             windows, `PHOTO_GLASS`) blurred by `SMOOTH_SIGMA` blocks and re-extracted
             (`smooth_surface`, the same operation as the training inputs' smoothing floor,
             without degradation), keeping its colours, lit by a strong sun with little
             ambient light (`PHOTO_LIGHT` x `PHOTO_EXPOSURE`) on a plain background. Real buildings'
             roofs are smooth; `SMOOTH_SIGMA` was chosen so TRELLIS's output on these photos
             matches TRELLIS on real photos in blockiness, and the light so it matches in
             lightness (`trials`)
    blocky   the stress test (`real_trellis/`): the photo is a render of the raw block build,
             so TRELLIS reproduces Minecraft's stair-stepped roofs
    train    training inputs (`real_trellis_train/`): the smooth photo of every
             train build through TRELLIS, so a model also trains on real TRELLIS output
             (its filled porches and courtyards, its colours) beside the synthetic pairs.
             Train builds only; never scored as an evaluation set

Steps:

    prepare   choose `--n` builds from val + test, round-robin over top-level categories
              (each category's builds in a fixed shuffled order), and render each photo
              (`testsets.trellis.render_photo`: lit, transparent background, a
              street-level three-quarter view)
    generate  send each photo to the TRELLIS server
              (`testsets.trellis.trellis_generate`, fixed seed) and keep the GLB;
              resumable. Needs the server: start C:\\TRELLIS\\run.bat first
    place     fit each TRELLIS mesh into the build's target frame (`align`): scaled to the
              target's height, stood on its lowest layer, centred on its footprint; yaw by a
              5-degree search (best three kept), and for each of those a translation search
              of +-`ALIGN_REACH` cells per axis at 2 cells per block (coarse, then fine, then
              a joint 3x3x3 step), then a 1-degree yaw refinement with that shift applied
              (and a local 2x search if the yaw moved); finally the shift moves from the IoU
              peak to the centre of the curve's top (`centre_shift`, what the validator
              measures: on real TRELLIS output the top is broad). The shift is applied
              to the stored input, which is then validated like a pair (frame residual within
              `make.SHIFT_TOL`) and compared with the synthetic inputs of the same build
    trials    for each `--sigmas` value, a smooth photo set of `--trial-ids` (5 builds) in
              `real_trellis_smooth/trials/sigma_<s>/`: prepare, generate, place, and
              TRELLIS's stair45 / roughness against the targets

Layout of a set folder under <dataset>/pairs/:

    selection.json            the chosen builds, seed, view, photo settings
    <id>/photo.png            the image sent to TRELLIS
    <id>/trellis.glb          what TRELLIS returned
    <id>/input.npz            same format as a synthetic pair (input_* at 2x + target_*)
    <id>/stats.json           placement, alignment, validation, statistics
    index.json                every build's status and numbers, and the gap report

Caveats, recorded in index.json: the photo is a render of the build, not a photograph; the
alignment uses the target (height, footprint, yaw, shift), so its IoU is optimistic;
TRELLIS's own randomness is fixed by one seed per build.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from pairs.make import REFERENCE_GLASS as PHOTO_GLASS

N_BUILDS = 25
SELECT_SEED = 20261014
TRELLIS_SEED = 1
SPLITS = ("val", "test")
SETS = {"smooth": "real_trellis_smooth", "blocky": "real_trellis", "train": "real_trellis_train"}
SMOOTH_SIGMA = 0.6                 # blocks; chosen by `trials`: TRELLIS stair45 / roughness p50 0.24 / 8.3 deg
PHOTO_LIGHT = (0.23, 0.77)         # strong sun, little ambient: ambient + sun * |n . sun| (no cast
                                   # shadows; `trials --lights`, and two full runs either side)
PHOTO_EXPOSURE = 0.66              # smooth set: a gain on PHOTO_LIGHT. TRELLIS undoes a plain gain
                                   # (0.45-1.0 all give L* 36-45); contrast is what it bakes dark
SMOOTH_CELLS_PER_BLOCK = 3         # resolution of the smoothed surface
SMOOTH_MAX_CELLS = 448
ALIGN_REACH = 16                   # cells searched each way per axis (2 cells per block)
TRIAL_IDS = ["pmc-chick-fil-a-restaurant-3831809", "pmc-california-bungalow-by-ew-stillwell-1918",
             "pmc-victorian-queen-anne-style---carson-mansion", "pmc-small-train-station",
             "pmc-small-skyscraper-download"]
TARGETS = {"stair45": 0.25, "normal_roughness": 8.0}     # TRELLIS on real photos, p50


def set_dir(dataset: Path, name: str) -> Path:
    return dataset / "pairs" / SETS[name]


def select(dataset: Path, n: int = N_BUILDS) -> list:
    """`n` val / test builds, round-robin over top-level categories."""
    rows = {r["id"]: r for r in json.loads((dataset / "index.json").read_text(encoding="utf-8"))["rows"]}
    norm = json.loads((dataset / "normalized" / "index.json").read_text(encoding="utf-8"))
    by_cat = {}
    for b in sorted(norm["builds"], key=lambda b: b["id"]):
        if b.get("status") == "ok" and b["split"] in SPLITS:
            by_cat.setdefault(rows[b["id"]]["category_top"], []).append(b["id"])
    rng = random.Random(SELECT_SEED)
    for ids in by_cat.values():
        rng.shuffle(ids)
    out = []
    while len(out) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(out) < n:
                out.append(by_cat[cat].pop())
    return out


def select_train(dataset: Path) -> list:
    """Every normalized train build, sorted (the `train` set: training inputs, never scored)."""
    norm = json.loads((dataset / "normalized" / "index.json").read_text(encoding="utf-8"))
    return sorted(b["id"] for b in norm["builds"] if b.get("status") == "ok" and b["split"] == "train")


# --------------------------------------------------------------------------- photos

def smooth_surface(mesh, sigma_blocks: float, samples: int = 2_000_000):
    """The build's outside blurred by `sigma_blocks` and re-extracted, colours kept: a
    `pairs.degrade` pass with every degradation off (no noise, holes, floaters or lighting)
    at about `SMOOTH_CELLS_PER_BLOCK` cells per block; each vertex then takes the colour of
    the nearest of `samples` points on the clean surface (crisp walls and windows, where
    degrade's voxel colours would smear windows into speckle)."""
    from scipy.spatial import cKDTree

    from voxel.scene import scene_color_fns
    from pairs.degrade import DegradeParams, degrade
    from pairs.preview import sample_surface
    lo, hi = mesh.bounds
    extent = float((hi - lo).max())
    cells = int(np.clip(round(SMOOTH_CELLS_PER_BLOCK * extent), 96, SMOOTH_MAX_CELLS))
    p = DegradeParams(bottleneck=cells, blur_sigma=0.5, surface_noise=0.0, noise_wavelength=8.0,
                      hole_fraction=0.0, hole_patch=1, floaters=0, tone_gain=1.0, ambient=1.0,
                      hue_deg=0.0, chroma=1.0, color_noise=0.0, back_blend=0.0, back_darken=1.0,
                      back_tint_deg=0.0, color_smooth=0, seed=0, min_blur_units=float(sigma_blocks))
    out, _ = degrade(mesh, p)
    _, point_colors = scene_color_fns(mesh)
    pts, cols, _ = sample_surface(mesh.verts, mesh.faces, point_colors, samples, seed=0)
    _, j = cKDTree(pts).query(out.verts)
    out.vcolor = np.asarray(cols, dtype=np.float64)[j]
    return out


def render_set_photo(mesh, name: str, sigma: float = SMOOTH_SIGMA, exposure: float = None,
                     light=None):
    """The photo of one build for set `name` (the build's meshify surface in). Smooth set:
    `light` (ambient, sun; default PHOTO_LIGHT) times `exposure` (default PHOTO_EXPOSURE)."""
    from testsets.trellis import render_photo

    from pairs.make import glass_look
    if name == "blocky":
        return render_photo(mesh)
    g = PHOTO_EXPOSURE if exposure is None else exposure
    a, s = PHOTO_LIGHT if light is None else light
    return render_photo(smooth_surface(glass_look(mesh, PHOTO_GLASS), sigma), light=(a * g, s * g))


def _render_one(job) -> str:
    """One build's photo (a process-pool job): (dataset, name, id, folder, sigma, exposure, light)."""
    from pairs.meshify import meshify_npz
    dataset, name, bid, d, sigma, exposure, light = job
    mesh, _, _ = meshify_npz(Path(dataset) / "normalized" / f"{bid}.npz")
    render_set_photo(mesh, name, sigma, exposure, light).save(Path(d) / "photo.png")
    return bid


def prepare(dataset: Path, name: str, n: int = N_BUILDS, sigma: float = SMOOTH_SIGMA,
            ids=None, out: Path = None, exposure: float = None, light=None, workers: int = 1) -> list:
    from testsets.trellis import IMAGE_SIZE, VIEW_PITCH, VIEW_YAW

    out = out or set_dir(dataset, name)
    out.mkdir(parents=True, exist_ok=True)
    ids = ids or (select_train(dataset) if name == "train" else select(dataset, n))
    rows = {r["id"]: r for r in json.loads((dataset / "index.json").read_text(encoding="utf-8"))["rows"]}
    todo = []
    for bid in ids:
        (out / bid).mkdir(exist_ok=True)
        if not (out / bid / "photo.png").is_file():
            todo.append((str(dataset), name, bid, str(out / bid), sigma, exposure, light))
    if workers > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for bid in pool.map(_render_one, todo):
                print(f"  {bid}: photo ready", flush=True)
    else:
        for job in todo:
            print(f"  {_render_one(job)}: photo ready", flush=True)
    done = [{"id": bid, "split": rows[bid]["split"], "category": rows[bid]["category_top"],
             "building_category": rows[bid]["building_category"]} for bid in ids]
    sel = {"set": name, "builds": done, "seed": SELECT_SEED, "trellis_seed": TRELLIS_SEED,
           "view": {"yaw": VIEW_YAW, "pitch": VIEW_PITCH, "size": IMAGE_SIZE},
           "splits": ["train"] if name == "train" else list(SPLITS)}
    if name in ("smooth", "train"):
        sel["photo"] = {"sigma_blocks": sigma, "light": list(PHOTO_LIGHT if light is None else light),
                        "glass": PHOTO_GLASS, "exposure": PHOTO_EXPOSURE if exposure is None else exposure,
                        "cells_per_block": SMOOTH_CELLS_PER_BLOCK}
    (out / "selection.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
    return done


def generate(dataset: Path, name: str, limit: int = 0, out: Path = None) -> int:
    from testsets.trellis import TRELLIS_URL, server_up, trellis_generate

    out = out or set_dir(dataset, name)
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    if not server_up():
        raise SystemExit(f"TRELLIS server not reachable at {TRELLIS_URL} -- start C:\\TRELLIS\\run.bat first")
    todo = [b["id"] for b in sel["builds"] if not (out / b["id"] / "trellis.glb").is_file()]
    if limit:
        todo = todo[:limit]
    failed = []
    for i, bid in enumerate(todo, 1):
        t0 = time.time()
        try:
            trellis_generate(out / bid / "photo.png", out / bid / "trellis.glb", seed=sel["trellis_seed"])
        except Exception as e:  # one bad build must not end a run of hours; a dead server does
            if not server_up():
                raise SystemExit(f"TRELLIS server went down at {bid} ({e}); run generate again to resume")
            failed.append(bid)
            print(f"  {i}/{len(todo)} {bid}: FAILED ({e})", flush=True)
            continue
        print(f"  {i}/{len(todo)} {bid}: {time.time() - t0:.0f} s", flush=True)
    if failed:
        print(f"{len(failed)} failed (run generate again to retry): {' '.join(failed)}", flush=True)
    return len(todo) - len(failed)


# --------------------------------------------------------------------------- alignment

def _fit(verts: np.ndarray, yaw: float, vol: np.ndarray, mult: float = 1.0, aspect: float = 1.0):
    """verts (TRELLIS units) turned by `yaw` about +Y, scaled to the target volume's height
    (times `mult`, and the horizontal axes times `aspect` on top), stood on its lowest layer
    and centred on its footprint -> (verts in block units, scale)."""
    t = np.radians(yaw)
    rot = np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])
    v = verts @ rot.T
    ys = np.flatnonzero(vol.any(axis=(0, 2)))
    xs = np.flatnonzero(vol.any(axis=(1, 2)))
    zs = np.flatnonzero(vol.any(axis=(0, 1)))
    lo, hi = v.min(axis=0), v.max(axis=0)
    s = (ys[-1] - ys[0] + 1) / max(hi[1] - lo[1], 1e-9) * mult
    k = np.array([s * aspect, s, s * aspect])
    v = v * k
    lo, hi = lo * k, hi * k
    centre = np.array([(xs[0] + xs[-1] + 1) / 2.0, 0.0, (zs[0] + zs[-1] + 1) / 2.0])
    shift = centre - np.array([(lo[0] + hi[0]) / 2.0, 0.0, (lo[2] + hi[2]) / 2.0])
    shift[1] = ys[0] - lo[1]
    return v + shift, float(s)


def _voxelize(mesh, verts, shape, scale: float, pad: int = 0, colors: bool = False):
    """VoxelModel of `mesh` with `verts` (block units) at `scale` cells per block, in the
    target frame grown by `pad` cells on every side (origin -pad)."""
    from voxel.grid import voxelize_scene
    m = mesh.transformed(np.eye(3), np.zeros(3))
    m.verts = verts
    return voxelize_scene(m, scale, offset=(0.0, 0.0, 0.0), origin=np.full(3, -pad, dtype=np.int64),
                          shape=tuple(int(s * scale) + 2 * pad for s in shape), with_shapes=False,
                          crop=False, drop_ground=False, keep_largest=False, clean=False,
                          with_colors=colors, color_fallback="nearest")


def _voxelize_covering(mesh, verts, shape, scale: float, margin: int, colors: bool = False):
    """(VoxelModel, lo): `mesh` voxelised at `scale` in a frame that holds the whole mesh and
    the target frame grown by `margin` cells, so nothing is cut off and the fill is computed
    on the whole mesh (a cut lets the exterior flood in). `lo` (3,) is where the target
    frame starts in it; translations are then windows of this one voxelisation, exactly
    what the search scored and the stored input gets."""
    from voxel.grid import voxelize_scene
    m = mesh.transformed(np.eye(3), np.zeros(3))
    m.verts = verts
    n = np.array([int(s * scale) for s in shape])
    v = np.asarray(verts, dtype=np.float64) * scale
    origin = np.minimum(-margin, np.floor(v.min(axis=0)).astype(np.int64) - 2)
    upper = np.maximum(n + margin, np.ceil(v.max(axis=0)).astype(np.int64) + 2)
    vm = voxelize_scene(m, scale, offset=(0.0, 0.0, 0.0), origin=origin.astype(np.int64),
                        shape=tuple(int(u - o) for u, o in zip(upper, origin)), with_shapes=False,
                        crop=False, drop_ground=False, keep_largest=False, clean=False,
                        with_colors=colors, color_fallback="nearest")
    return vm, (-origin).astype(np.int64)


def _window(a, lo, shift, shape):
    """`a`'s target-frame window when the mesh is moved by `shift` cells."""
    return a[tuple(slice(int(lo[k] - shift[k]), int(lo[k] - shift[k] + shape[k])) for k in range(3))]


def _crop(vm, lo, shift, shape):
    """The stored input: the target-frame window of a covering voxelisation at `shift`."""
    from dataclasses import replace

    def cut(a):
        return None if a is None else np.ascontiguousarray(_window(a, lo, shift, shape))

    return replace(vm, solid=cut(vm.solid), surface=cut(vm.surface), shell=cut(vm.shell), rgb=cut(vm.rgb),
                   kind_frac=cut(vm.kind_frac), origin=np.zeros(3, dtype=np.int64))


def search_shift(occ_pad: np.ndarray, vol2: np.ndarray, pad, reach: int = ALIGN_REACH,
                 coarse: int = 2, start=(0, 0, 0)) -> tuple:
    """(shift (3,) cells, IoU): the integer translation s, |s| <= reach per axis, that best
    overlays a padded occupancy on `vol2` -- occ_pad[pad - s : pad - s + N] against vol2, i.e.
    the mesh moved by s cells. Coordinate search at `coarse`-cell steps, then 1-cell steps,
    then a joint 3 x 3 x 3 step around the best."""
    pad = np.broadcast_to(np.asarray(pad, dtype=np.int64), (3,))
    reach = min(int(reach), int(pad.min()))
    N = vol2.shape
    n_vol = int(np.count_nonzero(vol2))
    cache = {}

    def score(s):
        s = tuple(int(v) for v in s)
        if s not in cache:
            if max(abs(v) for v in s) > reach:
                cache[s] = -1.0
            else:
                a = _window(occ_pad, pad, s, N)
                inter = int(np.count_nonzero(a & vol2))
                union = int(np.count_nonzero(a)) + n_vol - inter
                cache[s] = inter / union if union else 0.0
        return cache[s]

    s = list(start)
    for step in (coarse, 1):
        for _ in range(4):
            moved = False
            for axis in range(3):
                values = range(-reach, reach + 1, step) if step > 1 else (s[axis] - 1, s[axis], s[axis] + 1)
                best = max(values, key=lambda v: score(s[:axis] + [v] + s[axis + 1:]))
                if score(s[:axis] + [best] + s[axis + 1:]) > score(s):
                    s[axis] = best
                    moved = True
            if not moved:
                break
    import itertools
    best = max((tuple(s[k] + d[k] for k in range(3)) for d in itertools.product((-1, 0, 1), repeat=3)),
               key=score)
    return np.array(best), score(best)


def centre_shift(occ_pad: np.ndarray, vol2: np.ndarray, pad, start, rounds: int = 12) -> np.ndarray:
    """Move a shift from the IoU peak to where the validator sees no residual: the stored
    input is the window of the padded occupancy at the shift (what lies outside the target
    grid is cut off, as in the stored input), `make.frame_shift` measures its residual
    against the target volume (the centre of the IoU curve's top), and the shift moves by
    the rounded residual until it is under half a cell. On real TRELLIS output the top of
    the curve is broad and lopsided, and the cut-off part shifts it, so the peak can sit
    cells away from it. The shift may leave the search's reach; it is bounded only by the
    voxelised frame. Where the steps go round in a circle (a weak, flat fit: each move moves
    the plateau), the visited shift with the smallest residual is kept."""
    from pairs.make import frame_shift
    pad = np.broadcast_to(np.asarray(pad, dtype=np.int64), (3,))
    N = np.array(vol2.shape)
    s_max = pad.copy()                                        # window start lo - s >= 0
    s_min = -(np.array(occ_pad.shape) - N - pad)               # window end within the frame
    s = np.array([int(v) for v in start])
    seen = {}
    for _ in range(rounds):
        key = tuple(int(v) for v in s)
        if key in seen:
            break                                             # a cycle: stop
        window = _window(occ_pad, pad, s, N)
        r = np.array(frame_shift(window, vol2))
        seen[key] = float(np.abs(r).max())
        step = np.where(np.abs(r) > 0.5, np.rint(r), 0).astype(int)
        if not step.any():
            return s
        s = np.clip(s - step, s_min, s_max)
    return np.array(min(seen, key=seen.get))


def align(mesh, vol: np.ndarray, coarse_yaw: int = 5, keep: int = 3, fine: int = 4,
          scales=(1.0,), aspects=(1.0,)):
    """(VoxelModel at 2x with colours in the target frame, alignment info) for a TRELLIS mesh.

    Coarse to fine: every `coarse_yaw` degrees at 1 cell per block (centred on the target);
    for the best `keep` of those, the full translation search at 2 cells per block
    (`search_shift`, +-ALIGN_REACH cells); then +-`fine` degrees of yaw in 1-degree steps at
    1 cell per block with that shift applied; and, if a fine yaw wins, a local 2x search from
    the shift found. Only the few 2x voxelisations are expensive.

    With more than one of `scales` (multipliers on the height fit) or `aspects` (on the
    horizontal axes as well), those are searched first at 1 cell per block, each with a
    translation search: every scale for the best `keep` yaws, then every aspect at the best
    scale and yaw. The steps above then run at the chosen scale and aspect. Real photos use
    this (their builds are their builders' interpretations); the render sets keep the
    height fit."""
    from pairs.make import SCALE, iou
    verts = mesh.verts
    vol2 = vol.repeat(SCALE, 0).repeat(SCALE, 1).repeat(SCALE, 2)
    pad = 2 * ALIGN_REACH + 2             # frame margin: room for the centring beyond the reach
    form = {"mult": 1.0, "aspect": 1.0}

    def iou_1x(yaw, shift_cells=(0, 0, 0)):
        v, _ = _fit(verts, yaw % 360, vol, **form)
        return iou(_voxelize(mesh, v + np.asarray(shift_cells) / float(SCALE), vol.shape, 1.0).solid, vol)

    coarse = {yaw: iou_1x(yaw) for yaw in range(0, 360, coarse_yaw)}
    top = sorted(coarse, key=coarse.get, reverse=True)[:keep]
    scale_search = None
    if len(scales) > 1 or len(aspects) > 1:
        reach1 = ALIGN_REACH // SCALE

        def score_1x(yaw, mult, aspect):
            v, _ = _fit(verts, yaw % 360, vol, mult, aspect)
            vm_c, lo = _voxelize_covering(mesh, v, vol.shape, 1.0, reach1 + 2)
            return search_shift(vm_c.solid, vol, lo, reach=reach1, coarse=2)[1]

        cache = {}

        def scored(yaw, m, a):
            if (yaw, m, a) not in cache:
                cache[(yaw, m, a)] = score_1x(yaw, m, a)
            return cache[(yaw, m, a)]

        by_scale = {(yaw, m): scored(yaw, m, 1.0) for yaw in top for m in scales}
        yaw_s, mult = max(by_scale, key=by_scale.get)
        aspect = 1.0
        for _ in range(3):                    # scale and aspect in turn, until neither moves
            aspect_new = max(aspects, key=lambda a: scored(yaw_s, mult, a))
            mult_new = max(scales, key=lambda m: scored(yaw_s, m, aspect_new))
            if (mult_new, aspect_new) == (mult, aspect):
                break
            mult, aspect = mult_new, aspect_new
        scale_search = {"iou_1x_by_scale": {f"{m:g}": round(scored(yaw_s, m, aspect), 4) for m in scales},
                        "iou_1x_by_aspect": {f"{a:g}": round(scored(yaw_s, mult, a), 4) for a in aspects},
                        "evaluations_1x": len(cache), "best_1x": [float(mult), float(aspect)]}

        # A voxelised surface comes out about a cell larger than it is, a bias that is a
        # whole block at 1x: the neighbouring steps are scored again at 2x, as placed.
        def near(values, v):
            i = list(values).index(v)
            return list(values)[max(i - 1, 0):i + 2]

        by_2x = {}
        for m in near(scales, mult):
            for a in near(aspects, aspect):
                v, _ = _fit(verts, yaw_s, vol, m, a)
                vm_c, lo = _voxelize_covering(mesh, v, vol.shape, float(SCALE), pad)
                by_2x[(m, a)] = search_shift(vm_c.solid, vol2, lo)[1]
        mult, aspect = max(by_2x, key=by_2x.get)
        scale_search["iou_2x_refined"] = {f"{m:g}x{a:g}": round(s, 4) for (m, a), s in by_2x.items()}
        form.update(mult=float(mult), aspect=float(aspect))
        top = [yaw_s] + [y for y in top if y != yaw_s]
    searched = {}

    occs = {}

    def search(yaw, start=(0, 0, 0), step=2):
        yaw = yaw % 360
        if yaw not in searched:
            v, _ = _fit(verts, yaw, vol, **form)
            vm_c, lo = _voxelize_covering(mesh, v, vol.shape, float(SCALE), pad)
            occs[yaw] = (vm_c.solid, lo)
            s, score = search_shift(vm_c.solid, vol2, lo, start=start, coarse=step)
            searched[yaw] = (score, s, iou(_window(vm_c.solid, lo, (0, 0, 0), vol2.shape), vol2))
        return searched[yaw]

    for yaw in top:
        search(yaw)
    best = max(searched, key=lambda y: searched[y][0])
    shift = searched[best][1]
    fine_scores = {(best + d) % 360: iou_1x(best + d, shift) for d in range(-fine, fine + 1)}
    yaw_fine = max(fine_scores, key=fine_scores.get)
    if yaw_fine != best:
        search(yaw_fine, start=tuple(int(x) for x in shift), step=1)
        if searched[yaw_fine][0] > searched[best][0]:
            best = yaw_fine
    score, peak, iou0 = searched[best]
    occ, lo = occs[best]
    shift = centre_shift(occ, vol2, lo, peak)
    v, scale = _fit(verts, best, vol, **form)
    vm_c, lo = _voxelize_covering(mesh, v, vol.shape, float(SCALE), pad, colors=True)
    vm = _crop(vm_c, lo, shift, vol2.shape)          # exactly the window the search scored
    v = v + shift / float(SCALE)                     # cells -> blocks (for the mesh statistics)
    info = {"yaw_deg": int(best), "shift_cells": [int(x) for x in shift],
            "peak_shift_cells": [int(x) for x in peak],
            "scale_blocks_per_unit": round(scale, 4), "iou_2x_unshifted": round(iou0, 4),
            "iou_2x_shifted": round(score, 4), "yaws_coarse": len(coarse), "yaws_fine": len(fine_scores),
            "yaws_searched_2x": len(searched), "trellis_faces": int(len(mesh.faces)),
            "verts_block_units": v}
    if scale_search is not None:
        info.update(scale_mult=form["mult"], aspect=form["aspect"], scale_search=scale_search)
    return vm, info


# --------------------------------------------------------------------------- placement

def measure(dataset: Path, bid: str, glb: Path, input_path: Path, scales=(1.0,), aspects=(1.0,)) -> dict:
    """One TRELLIS mesh against its build: `align` (with `scales` / `aspects`), the 2x input
    written to `input_path` (pair format), validation, the input statistics, the detail
    metric and the colour cast against the reference (`pairs.detail`)."""
    from voxel.scene import load_scene

    from pairs.detail import colour_cast, detail_of, reference_model
    from pairs.meshify import meshify_npz
    from dataset.normalize import save_npz
    from pairs.make import (SCALE, frame_shift, input_stats, normal_roughness, target_volume,
                        validate_pair)
    t0 = time.time()
    npz = Path(dataset) / "normalized" / f"{bid}.npz"
    z = np.load(npz)
    vol = target_volume(z["target_shape"], z["exterior"], z["ground_height"])
    mesh = load_scene(glb)
    vm, placement = align(mesh, vol, scales=scales, aspects=aspects)
    verts = placement.pop("verts_block_units")
    pair = {"input_occ": vm.solid.astype(np.uint8), "input_surface": vm.surface.astype(np.uint8),
            "input_rgb": np.ascontiguousarray(vm.rgb, dtype=np.uint8),
            "target_shape": z["target_shape"], "target_material": z["target_material"],
            "ground_height": z["ground_height"].astype(np.int16)}
    failures, metrics = validate_pair(pair, z["exterior"])
    vol2 = vol.repeat(SCALE, 0).repeat(SCALE, 1).repeat(SCALE, 2)
    stats = input_stats(vm)
    stats["normal_roughness"] = normal_roughness(verts, mesh.faces)
    clean, _, _ = meshify_npz(npz)
    ref = reference_model(clean, z["target_shape"].shape)
    stats.update(detail_of(vm, ref))
    stats.update(colour_cast(vm, ref))
    save_npz(input_path, pair)
    return {"placement": placement, "validation": {"failures": failures, **metrics},
            "alignment": {"residual_wide_cells": frame_shift(vm.solid, vol2, search=ALIGN_REACH)},
            "stats": stats, "status": "ok", "seconds": round(time.time() - t0, 1)}


def _place_one(job):
    """One build's placement (a process-pool job): (dataset, id, folder) -> (id, `measure`)."""
    dataset, bid, d = job
    return bid, measure(Path(dataset), bid, Path(d) / "trellis.glb", Path(d) / "input.npz")


def place(dataset: Path, name: str, out: Path = None, previous: Path = None, only=None,
          workers: int = 1, resume: bool = False) -> dict:
    """Align, voxelise, validate and measure every generated mesh of a set (`measure`);
    `previous` (an earlier index.json of the set) supplies the residuals before the alignment
    change. `only`: re-place just these builds, keeping the others' rows from the set's
    index.json. `resume`: keep every build already placed (its stats.json and input.npz).
    `workers` > 1 places builds in parallel."""
    out = out or set_dir(dataset, name)
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    index = json.loads((dataset / "pairs" / "index.json").read_text(encoding="utf-8"))
    synthetic = {}
    for p in index["pairs"]:
        if p["status"] == "ok":
            synthetic.setdefault(p["id"], []).append(p)
    before = {}
    if previous and Path(previous).is_file():
        before = {b["id"]: b for b in json.loads(Path(previous).read_text(encoding="utf-8"))["builds"]}
    kept = {}
    if only and (out / "index.json").is_file():
        kept = {r["id"]: r for r in json.loads((out / "index.json").read_text(encoding="utf-8"))["builds"]}
    by_id, todo = {}, []
    for b in sel["builds"]:
        bid = b["id"]
        d = out / bid
        if only and bid not in only and bid in kept:
            by_id[bid] = kept[bid]
            continue
        if not (d / "trellis.glb").is_file():
            by_id[bid] = {**b, "status": "awaiting_trellis" if (d / "photo.png").is_file() else "awaiting_photo"}
            continue
        if resume and (d / "stats.json").is_file() and (d / "input.npz").is_file():
            prev = json.loads((d / "stats.json").read_text(encoding="utf-8"))
            if prev.get("status") == "ok":
                by_id[bid] = prev
                continue
        todo.append((str(dataset), bid, str(d)))

    def finish(bid, measured):
        row = dict(next(b for b in sel["builds"] if b["id"] == bid))
        row.update(measured)
        metrics, placement = row["validation"], row["placement"]
        if bid in before:
            pv = before[bid].get("validation", {})
            row["alignment"]["before"] = {"iou": pv.get("iou"), "residual_cells": pv.get("shift_residual_cells"),
                                          "failures": pv.get("failures")}
        row["synthetic"] = [{"variant": p["variant"], "iou": p["validation"]["iou"], **p["stats"]}
                            for p in synthetic.get(bid, [])]
        (out / bid / "stats.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
        by_id[bid] = row
        print(f"  {bid}: IoU {metrics.get('iou')} yaw {placement['yaw_deg']} shift {placement['shift_cells']}"
              f" residual {metrics.get('shift_residual_cells')}"
              f" {'FAIL ' + str(metrics['failures']) if metrics['failures'] else 'ok'}", flush=True)

    if workers > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for bid, measured in pool.map(_place_one, todo):
                finish(bid, measured)
    else:
        for job in todo:
            finish(*_place_one(job))
    rows = [by_id[b["id"]] for b in sel["builds"]]
    res = {"version": "real-trellis-2", "set": name, "photo": sel.get("photo"), "builds": rows,
           "gap": gap_report(rows), "alignment": alignment_summary(rows),
           "caveats": ["The photo is a render of the build (flat block colours), not a photograph.",
                       "Placement uses the target (height, footprint, yaw, shift): IoU is optimistic.",
                       "One TRELLIS seed per build."]}
    (out / "index.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def alignment_summary(rows: list) -> dict:
    done = [r for r in rows if r.get("status") == "ok"]
    after = [max(abs(x) for x in r["validation"]["shift_residual_cells"]) for r in done]
    out = {"builds": len(done),
           "misaligned_after": sum(any(f.startswith("frame_misaligned") for f in r["validation"]["failures"])
                                   for r in done),
           "residual_max_axis_after": _pct(after), "iou_after": _pct([r["validation"]["iou"] for r in done]),
           "shift_cells_abs_max": _pct([max(abs(x) for x in r["placement"]["shift_cells"]) for r in done])}
    prev = [r["alignment"]["before"] for r in done if r.get("alignment", {}).get("before")]
    if prev:
        out["misaligned_before"] = sum(any(str(f).startswith("frame_misaligned") for f in (p["failures"] or []))
                                       for p in prev)
        out["residual_max_axis_before"] = _pct([max(abs(x) for x in p["residual_cells"]) for p in prev
                                                if p.get("residual_cells")])
        out["iou_before"] = _pct([p["iou"] for p in prev if p.get("iou") is not None])
    return out


GAP_KEYS = ("iou", "coverage", "excess", "fill_gain", "rough3", "lum", "chroma", "nbr_de",
            "stair45", "normal_roughness")


def gap_report(rows: list) -> dict:
    """Per statistic: real TRELLIS p10/p50/p90, the synthetic inputs of the same builds, and
    the paired median of (synthetic - real) over builds (synthetic averaged over variants)."""
    done = [r for r in rows if r.get("status") == "ok" and r.get("synthetic")]
    out = {"builds": len(done)}
    for k in GAP_KEYS:
        real, syn, diff = [], [], []
        for r in done:
            rv = r["validation"].get(k) if k in ("iou", "coverage", "excess") else r["stats"].get(k)
            sv = [s.get(k) for s in r["synthetic"] if s.get(k) is not None]
            if k in ("coverage", "excess"):
                sv = []
            if rv is None:
                continue
            real.append(rv)
            if sv:
                syn.append(float(np.mean(sv)))
                diff.append(float(np.mean(sv)) - rv)
        out[k] = {"real_p10_p50_p90": _pct(real), "synthetic_p10_p50_p90": _pct(syn),
                  "paired_median_diff": round(float(np.median(diff)), 4) if diff else None}
    out["reading"] = "paired_median_diff > 0: the synthetic input has more of it than real TRELLIS"
    return out


def _pct(v):
    v = np.array([x for x in v if x is not None], dtype=np.float64)
    return [round(float(np.percentile(v, p)), 4) for p in (10, 50, 90)] if len(v) else None


# --------------------------------------------------------------------------- sigma trials

TRIAL_KEYS = ("stair45", "normal_roughness", "lum", "chroma", "nbr_de", "detail_geo", "detail_colour")


def trials(dataset: Path, sigmas=(), ids=None, exposures=(), lights=()) -> dict:
    """A smooth photo set of `ids` per sigma (at `PHOTO_EXPOSURE`), per exposure and per light
    ((ambient, sun, exposure), at `SMOOTH_SIGMA`): TRELLIS's statistics against TARGETS
    (stair45 / roughness) and the real-photo lightness. Results merge into trials/trials.json."""
    ids = ids or TRIAL_IDS
    base = set_dir(dataset, "smooth") / "trials"
    path = base / "trials.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    res.update(targets=TARGETS, ids=ids)
    runs = [("sigmas", f"{s:g}", base / f"sigma_{s:g}", s, None, None) for s in sigmas]
    runs += [("exposures", f"{g:g}", base / f"exposure_{g:g}", SMOOTH_SIGMA, g, None) for g in exposures]
    runs += [("lights", f"{a:g}_{s:g}_{g:g}", base / f"light_{a:g}_{s:g}_{g:g}", SMOOTH_SIGMA, g, (a, s))
             for a, s, g in lights]
    for group, key, out, sigma, exposure, light in runs:
        prepare(dataset, "smooth", sigma=sigma, ids=ids, out=out, exposure=exposure, light=light)
        generate(dataset, "smooth", out=out)
        idx = place(dataset, "smooth", out=out)
        rows = [r for r in idx["builds"] if r.get("status") == "ok"]
        res.setdefault(group, {})[key] = {
            **{k: _pct([r["stats"].get(k) for r in rows]) for k in TRIAL_KEYS},
            "iou": _pct([r["validation"]["iou"] for r in rows]),
            "per_build": {r["id"]: {**{k: r["stats"].get(k) for k in TRIAL_KEYS},
                                    "iou": r["validation"]["iou"]} for r in rows}}
        print(f"{group} {key}: {json.dumps({k: v for k, v in res[group][key].items() if k != 'per_build'})}",
              flush=True)
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def status(dataset: Path) -> dict:
    from testsets.trellis import server_up
    st = {"server_up": server_up()}
    for name in SETS:
        out = set_dir(dataset, name)
        sel = (json.loads((out / "selection.json").read_text(encoding="utf-8"))
               if (out / "selection.json").is_file() else {"builds": []})
        st[name] = {"selected": len(sel["builds"]),
                    "photos": sum((out / b["id"] / "photo.png").is_file() for b in sel["builds"]),
                    "trellis_meshes": sum((out / b["id"] / "trellis.glb").is_file() for b in sel["builds"]),
                    "placed": sum((out / b["id"] / "input.npz").is_file() for b in sel["builds"])}
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["prepare", "generate", "place", "trials", "status"])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--set", choices=list(SETS), default="smooth")
    ap.add_argument("--n", type=int, default=N_BUILDS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=SMOOTH_SIGMA)
    ap.add_argument("--sigmas", type=float, nargs="*", default=[])
    ap.add_argument("--exposures", type=float, nargs="*", default=[])
    ap.add_argument("--lights", nargs="*", default=[], metavar="AMBIENT:SUN:EXPOSURE")
    ap.add_argument("--trial-ids", nargs="*", default=None)
    ap.add_argument("--previous", type=Path, help="an earlier index.json of the set (residuals before)")
    ap.add_argument("--only", nargs="*", default=None, help="place: re-place just these builds")
    ap.add_argument("--workers", type=int, default=1, help="prepare / place: builds in parallel")
    ap.add_argument("--resume", action="store_true", help="place: keep builds already placed")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dataset = args.dataset.resolve()
    if args.cmd == "prepare":
        done = prepare(dataset, args.set, args.n, args.sigma, workers=args.workers)
        print(f"{len(done)} photos ready in {set_dir(dataset, args.set)}")
    elif args.cmd == "generate":
        print(f"generated {generate(dataset, args.set, args.limit)}")
    elif args.cmd == "place":
        res = place(dataset, args.set, previous=args.previous, only=set(args.only) if args.only else None,
                    workers=args.workers, resume=args.resume)
        print(json.dumps(res["alignment"], indent=1))
        print(json.dumps(res["gap"], indent=1))
    elif args.cmd == "trials":
        trials(dataset, args.sigmas, args.trial_ids, args.exposures,
               [tuple(float(x) for x in s.split(":")) for s in args.lights])
    else:
        print(json.dumps(status(dataset), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
