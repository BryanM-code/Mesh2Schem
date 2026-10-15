#!/usr/bin/env python3
"""Phase 4: (input, target) training pairs from real-schems-v1 (D16). No training here.

Usage:  python -m schemset.pairs --dataset data/real-schems-v1 [--only <id> ...] [--pilot N]
                                 [--variants 4] [--workers N] [--force]

For every normalized build of split train / val / test ("excluded" never):

    mesh      `schemset.meshify`: the build's outside as a coloured surface in block units,
              in the target's own frame (exterior faces only, real block geometry)
    variant   K = `VARIANTS` per build, each with its own seed (`variant_seed`): a yaw of
              up to +-`YAW_DEG` about the build's vertical centre line and an offset of up
              to +-`OFFSET_BLOCKS` block on each axis, applied to the clean surface -- the
              misregistration a real placement leaves
    degrade   `synth.degrade.degrade` with `sample_params` (the calibrated TRELLIS-like
              ranges of `synth.config.PipelineConfig.degrade`): a resolution bottleneck
              relative to the build's size, blur, surface noise, holes, floaters, baked dark
              colour, a hallucinated back side
    smooth    the bottleneck blur is at least `SMOOTH_SIGMA_BLOCKS` block (sigma; FWHM >= 1
              block): the domain-gap guard -- Minecraft's one-block stair steps are blockier
              than what TRELLIS makes of real buildings (docs/pairs-v1.md,
              `schemset.pairs_calibrate`)
    input     `mcvoxel.grid.voxelize_scene` at `SCALE` = 2 cells per block into EXACTLY the
              target frame: origin (0, 0, 0), shape 2 x the target's; no crop, no ground
              plate removal, no component cleanup (the inference voxeliser does the same)
    validate  `validate_pair`: shapes and dtypes, finite values, the target holds only
              VALID pairs or IGNORE, block-level IoU within the band, and the frame: the
              input must overlay the clean surface placed by the same yaw and offset within
              max(SHIFT_TOL, one bottleneck cell). Per build, `frame_identity` checks the
              clean surface's bounds against the shell. A failure rejects the pair with its
              reason; nothing is dropped silently
    write     one compressed npz per (build, variant), deterministic bytes

Writes, all under <dataset>/pairs/ (never inside source/):

    synthetic/<id>/v<k>.npz   input_occ, input_surface (uint8 0/1, 2X x 2Y x 2Z),
                              input_rgb (uint8 sRGB, 2X x 2Y x 2Z x 3, on the input's shell,
                              0 elsewhere), target_shape (uint8 X x Y x Z), target_material
                              (uint16), ground_height (int16 X x Z) -- unchanged from
                              normalized/<id>.npz
    records/<id>.json         per build: mesh info, and per variant seed, params, validation,
                              stats, timings
    index.json                one row per (build, variant) plus a summary

The format, the frame and the loader contract (64^3 target windows with their 128^3 input
windows, rotations and mirrors applied to both grids, IGNORE in the loss) are in
docs/pairs-v1.md. Deterministic: the same build, variant and code give the same bytes.
Resumable: a build whose record matches the current fingerprint is skipped (--force).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

VERSION = "pairs-1"
SCALE = 2                         # input cells per block (D16)
VARIANTS = 4
YAW_DEG = 5.0
OFFSET_BLOCKS = 0.5
SMOOTH_SIGMA_BLOCKS = 0.42        # blur sigma floor in blocks (FWHM 1 block): set by the guard
IOU_MIN = 0.2                     # block-level IoU with the target volume: below, rejected
COVER_MIN = 0.5                   # target volume the input covers: below, rejected (lost)
IOU_FLAG = 0.45                   # below: kept, flagged `low_iou` (blob far bigger than the build)
SHIFT_SEARCH = 4                  # cells searched each way when measuring the frame shift
SHIFT_TOL = 1.0                   # cells: frame residual allowed per axis (or one bottleneck cell)
PILOT_SEED = 20261014


def variant_seed(build_id: str, variant: int) -> int:
    """A stable 31-bit seed per (build, variant), independent of run order and machine."""
    digest = hashlib.sha256(f"real-schems-v1/{VERSION}/{build_id}/{variant}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def settings() -> dict:
    from synth.config import PipelineConfig

    from .meshify import MESH_VERSION
    return {"version": VERSION, "mesh_version": MESH_VERSION, "scale": SCALE, "variants": VARIANTS,
            "yaw_deg": YAW_DEG, "offset_blocks": OFFSET_BLOCKS, "smooth_sigma_blocks": SMOOTH_SIGMA_BLOCKS,
            "iou_min": IOU_MIN, "cover_min": COVER_MIN, "iou_flag": IOU_FLAG, "shift_tol_cells": SHIFT_TOL,
            "degrade_ranges": PipelineConfig().degrade}


def settings_hash() -> str:
    return hashlib.sha256(json.dumps(settings(), sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- one variant

def sample_variant(seed: int, smooth_sigma: float = None) -> dict:
    """Everything random about one variant, drawn from its seed in a fixed order."""
    from dataclasses import asdict
    from synth.config import PipelineConfig
    from synth.degrade import sample_params
    rng = np.random.default_rng(seed)
    yaw = float(rng.uniform(-YAW_DEG, YAW_DEG))
    offset = [float(v) for v in rng.uniform(-OFFSET_BLOCKS, OFFSET_BLOCKS, size=3)]
    params = sample_params(rng, PipelineConfig().degrade)
    params.min_blur_units = SMOOTH_SIGMA_BLOCKS if smooth_sigma is None else float(smooth_sigma)
    return {"seed": seed, "yaw_deg": yaw, "offset_blocks": offset, "degrade": asdict(params)}


def place(mesh, yaw_deg: float, offset):
    """The clean surface turned by `yaw_deg` about its vertical centre line, then shifted."""
    lo, hi = mesh.bounds
    centre = np.array([(lo[0] + hi[0]) / 2.0, 0.0, (lo[2] + hi[2]) / 2.0])
    t = np.radians(yaw_deg)
    rot = np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])
    return mesh.transformed(rot, centre - rot @ centre + np.asarray(offset, dtype=np.float64))


def voxelize_input(mesh, shape, colors: bool = True):
    """A mesh in block units -> VoxelModel at SCALE cells per block in the target frame:
    origin (0, 0, 0), shape SCALE x `shape`. Surface raster, exterior flood, enclosed fill;
    no crop, no ground-plate removal, no component cleanup."""
    from mcvoxel.grid import voxelize_scene
    return voxelize_scene(mesh, float(SCALE), offset=(0.0, 0.0, 0.0), origin=np.zeros(3, dtype=np.int64),
                          shape=tuple(int(s) * SCALE for s in shape), with_shapes=False, crop=False,
                          drop_ground=False, keep_largest=False, clean=False, with_colors=colors,
                          color_fallback="nearest")


def make_input(clean, shape, variant: dict):
    """(VoxelModel at 2x in the target frame, degraded SceneMesh, info, reference) for one
    variant. `reference` is the clean surface placed the same way (yaw, offset) and voxelised
    the same way without degradation: what the frame check compares the input with."""
    from synth.degrade import DegradeParams, degrade
    info = {}
    t0 = time.time()
    moved = place(clean, variant["yaw_deg"], variant["offset_blocks"])
    deg, info["degrade"] = degrade(moved, DegradeParams(**variant["degrade"]))
    lo, hi = moved.bounds
    info["bottleneck_cell_blocks"] = float((hi - lo).max()) / variant["degrade"]["bottleneck"]
    info["t_degrade"] = round(time.time() - t0, 2)
    t1 = time.time()
    vm = voxelize_input(deg, shape)
    info["voxelize"] = {k: v for k, v in vm.info.items() if not isinstance(v, (dict, list))}
    info["t_voxelize"] = round(time.time() - t1, 2)
    reference = voxelize_input(moved, shape, colors=False).solid
    return vm, deg, info, reference


# --------------------------------------------------------------------------- validation

def target_volume(target_shape: np.ndarray, exterior=None, ground_height=None) -> np.ndarray:
    """The building's volume in the target frame: not exterior air and above the ground.

    Without `exterior`, exterior air is read off the target (shape 0 = supervised air)."""
    vol = target_shape != 0 if exterior is None else ~np.asarray(exterior, dtype=bool)
    if ground_height is not None:
        X, Y, Z = target_shape.shape
        h = np.broadcast_to(np.asarray(ground_height, dtype=np.int64), (X, Z))
        vol = vol & ~(np.arange(Y)[None, :, None] <= h[:, None, :])
    return vol


def downsample(occ2: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """Block-level occupancy: a block is occupied when at least half its cells are."""
    X, Y, Z = (s // scale for s in occ2.shape)
    cells = occ2[:X * scale, :Y * scale, :Z * scale].reshape(X, scale, Y, scale, Z, scale)
    return cells.mean(axis=(1, 3, 5), dtype=np.float32) >= 0.5


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    return int(np.count_nonzero(a & b)) / union if union else 0.0


def _shifted_iou(a: np.ndarray, b: np.ndarray, axis: int, s: int) -> float:
    """IoU of a and b moved by s cells along `axis` (zero fill)."""
    if s == 0:
        return iou(a, b)
    n = a.shape[axis]
    sa = [slice(None)] * 3
    sb = [slice(None)] * 3
    if s > 0:
        sa[axis], sb[axis] = slice(s, n), slice(0, n - s)
    else:
        sa[axis], sb[axis] = slice(0, n + s), slice(-s, n)
    inter = int(np.count_nonzero(a[tuple(sa)] & b[tuple(sb)]))
    union = int(np.count_nonzero(a)) + int(np.count_nonzero(b)) - inter
    return inter / union if union else 0.0


def frame_shift(a: np.ndarray, b: np.ndarray, search: int = SHIFT_SEARCH, rel: float = 0.01) -> list:
    """Per axis, the shift in cells that best overlays `a` on `b` (a[i] ~ b[i - shift]): the
    centre of the IoU-vs-shift curve within `rel` of its maximum (weighted by the excess), so
    a flat top -- one shape inside a slightly bigger one -- gives its middle, not an edge."""
    out = []
    for axis in range(3):
        s = np.arange(-search, search + 1)
        scores = np.array([_shifted_iou(a, b, axis, int(k)) for k in s])
        w = np.clip(scores - scores.max() * (1 - rel), 0, None)
        out.append(round(float((s * w).sum() / w.sum()), 3) if w.sum() > 0 else 0.0)
    return out


def validate_pair(arrays: dict, exterior=None, reference=None, applied_offset=None,
                  tol: float = SHIFT_TOL) -> tuple[list, dict]:
    """(failures, metrics) for one pair's arrays -- see docs/pairs-v1.md, "Validation".

    `arrays`: input_occ / input_surface / input_rgb / target_shape / target_material /
    ground_height as written. Frame check: the shift that best overlays the input on
    `reference` (the clean surface placed by the same yaw and offset, voxelised the same
    way) must be within `tol` cells on every axis. Without a reference the input is
    compared with the target volume, expected at `applied_offset` (blocks)."""
    from mcvoxel import vocab
    failures, m = [], {}
    ts, tm = arrays["target_shape"], arrays["target_material"]
    X, Y, Z = ts.shape
    want = {"input_occ": ((X * SCALE, Y * SCALE, Z * SCALE), np.uint8),
            "input_surface": ((X * SCALE, Y * SCALE, Z * SCALE), np.uint8),
            "input_rgb": ((X * SCALE, Y * SCALE, Z * SCALE, 3), np.uint8),
            "target_shape": ((X, Y, Z), np.uint8), "target_material": ((X, Y, Z), np.uint16),
            "ground_height": ((X, Z), np.int16)}
    for k, (shape, dtype) in want.items():
        a = arrays.get(k)
        if a is None:
            failures.append(f"missing:{k}")
        elif tuple(a.shape) != shape:
            failures.append(f"shape:{k} {tuple(a.shape)} != {shape}")
        elif a.dtype != dtype:
            failures.append(f"dtype:{k} {a.dtype} != {np.dtype(dtype)}")
    if failures:
        return failures, m
    for k, a in arrays.items():
        if np.issubdtype(np.asarray(a).dtype, np.floating) and not np.isfinite(a).all():
            failures.append(f"nan_inf:{k}")
    occ = arrays["input_occ"]
    if occ.max(initial=0) > 1 or arrays["input_surface"].max(initial=0) > 1:
        failures.append("input_not_binary")
    ign = ts == vocab.IGNORE_SHAPE
    if not np.array_equal(ign, tm == vocab.IGNORE_MATERIAL):
        failures.append("target_ignore_mismatch")
    sup = ~ign
    if sup.any():
        sh, ma = ts[sup].astype(np.int64), tm[sup].astype(np.int64)
        in_range = (sh < len(vocab.SHAPES)) & (ma < len(vocab.MATERIALS))
        ok = np.zeros(len(sh), dtype=bool)
        ok[in_range] = vocab.VALID[sh[in_range], ma[in_range]]
        if not ok.all():
            failures.append(f"target_invalid_pairs:{int((~ok).sum())}")
    occ_b = occ.astype(bool)
    if not occ_b.any():
        failures.append("empty_input")
        return failures, m
    vol = target_volume(ts, exterior, arrays["ground_height"])
    down = downsample(occ_b)
    m["iou"] = round(iou(down, vol), 4)
    inter = int(np.count_nonzero(down & vol))
    m["coverage"] = round(inter / max(int(vol.sum()), 1), 4)          # target volume the input covers
    m["excess"] = round(int(np.count_nonzero(down & ~vol)) / max(int(down.sum()), 1), 4)
    if reference is not None:
        ref = np.asarray(reference, dtype=bool)
        m["reference_iou"] = round(iou(occ_b, ref), 4)
        m["shift_residual_cells"] = frame_shift(occ_b, ref)
    else:
        vol2 = vol.repeat(SCALE, 0).repeat(SCALE, 1).repeat(SCALE, 2)
        applied = [0.0, 0.0, 0.0] if applied_offset is None else [float(v) * SCALE for v in applied_offset]
        m["shift_residual_cells"] = [round(s - a, 3) for s, a in zip(frame_shift(occ_b, vol2), applied)]
    m["shift_tol_cells"] = round(float(tol), 3)
    # The band: the frame itself is checked against the reference below, so a low IoU from a
    # blob much bigger than the build (excess) is kept and flagged; an input that lost the
    # building (coverage) or barely overlaps it is rejected.
    m["flags"] = ["low_iou"] if m["iou"] < IOU_FLAG else []
    if m["iou"] < IOU_MIN:
        failures.append(f"iou_below_band:{m['iou']}")
    if m["coverage"] < COVER_MIN:
        failures.append(f"input_misses_building:{m['coverage']}")
    bad = [a for a, r in zip("xyz", m["shift_residual_cells"]) if abs(r) > tol]
    if bad:
        failures.append("frame_misaligned:" + "".join(bad))
    return failures, m


def frame_identity(mesh, info: dict, shell: np.ndarray, tol: float = 1.0) -> tuple[list, dict]:
    """Per-build check of the meshify frame (a shifted, scaled or swapped mesh fails it):

    * every vertex is on the 1/16 lattice of the grid, inside the grid;
    * the surface covers the bounding box of `shell` (the shell blocks that have geometry)
      within `tol` blocks on every side -- it may reach further, where decor opened the
      view on blocks the D22 shell does not hold, but never fall short;
    * no opening cap lies more than `tol` blocks outside the blocks that were meshed (caps
      floating in the sky were the symptom of air enclosed by decor)."""
    problems, m = [], {}
    if not len(mesh.verts) or not shell.any():
        return ["frame_identity:empty"], m
    if not info.get("on_lattice"):
        problems.append("frame_identity:off_lattice")
    if not info.get("in_grid"):
        problems.append("frame_identity:outside_grid")
    lo, hi = mesh.bounds
    idx = [np.flatnonzero(shell.any(axis=tuple(a for a in range(3) if a != ax))) for ax in range(3)]
    want_lo = np.array([i[0] for i in idx], dtype=np.float64)
    want_hi = np.array([i[-1] + 1 for i in idx], dtype=np.float64)
    short = np.concatenate([lo - want_lo, want_hi - hi])          # > 0: the mesh falls short
    m["mesh_short_of_shell"] = [round(float(v), 4) for v in short]
    if short.max() > tol:
        problems.append("frame_identity:short " + ",".join(f"{v:.2f}" for v in short))
    vb = info.get("voxel_bounds", {})
    drawn = [vb[k] for k in ("block", "internal", "ground") if k in vb]
    if "cap" in vb and drawn:
        g_lo = np.min([b[0] for b in drawn], axis=0)
        g_hi = np.max([b[1] for b in drawn], axis=0)
        out = np.concatenate([g_lo - np.array(vb["cap"][0]), np.array(vb["cap"][1]) - g_hi])
        m["caps_beyond_blocks"] = [int(v) for v in out]
        if out.max() > tol:
            problems.append("frame_identity:caps_beyond_blocks " + ",".join(str(int(v)) for v in out))
    return problems, m


# --------------------------------------------------------------------------- statistics

def _top_surface(occ2: np.ndarray) -> np.ndarray:
    """Height (cell index) of the highest occupied cell of every column, -1 where empty."""
    occ = np.asarray(occ2, dtype=bool)
    return np.where(occ.any(axis=1), occ.shape[1] - 1 - np.argmax(occ[:, ::-1, :], axis=1), -1)


def stair45(occ2: np.ndarray, min_pairs: int = 20) -> float | None:
    """Voxel-level stair-step signature at the input resolution (2 cells per block).

    Along every row and column of the top surface, take consecutive height steps (d1, d2)
    that together rise or fall 2 cells over 2 cells -- a 45-degree stretch. A Minecraft
    stair roof (one block up per block across) reads (0, 2) / (2, 0): flat tread, one-block
    riser; a smooth 45-degree slope reads (1, 1). Returns the share of (0, 2) / (2, 0)
    among them (None with fewer than `min_pairs`): ~1 for block stairs, ~0 for a smooth
    roof, whatever the roof's colour or the rest of the building."""
    top = _top_surface(occ2)
    stair = smooth = 0
    for axis in (0, 1):
        h = np.moveaxis(top, axis, 0)
        d = np.diff(h, axis=0)
        ok = (h[:-1] >= 0) & (h[1:] >= 0)
        d1, d2 = d[:-1], d[1:]
        both = ok[:-1] & ok[1:]
        for sign in (1, -1):
            a, b = sign * d1, sign * d2
            stair += int((both & (((a == 0) & (b == 2)) | ((a == 2) & (b == 0)))).sum())
            smooth += int((both & (a == 1) & (b == 1)).sum())
    n = stair + smooth
    return round(stair / n, 4) if n >= min_pairs else None


def normal_roughness(verts, faces, radius: float = 1.0, n: int = 20000, seed: int = 0) -> float | None:
    """Mesh-level stair-step metric: mean angle (degrees) between the surface normal and the
    mean normal of the surface within `radius` (mesh units, here blocks) on the same side
    (normals less than 120 degrees apart, so the far side of a thin wall does not count).
    Smooth surfaces and flat walls score low; a staircase of steps of about `radius` high."""
    from scipy.spatial import cKDTree
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if not len(f):
        return None
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    ok = area > 1e-12
    if ok.sum() < 10:
        return None
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(ok)
    pick = idx[rng.choice(len(idx), size=n, p=area[idx] / area[idx].sum())]
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    bary = np.stack([1 - s, s * (1 - r2), s * r2], axis=1)
    pts = (tri[pick] * bary[:, :, None]).sum(axis=1)
    nrm = cross[pick] / area[pick, None]
    tree = cKDTree(pts)
    nb = tree.query_ball_point(pts, r=radius)
    mean = np.zeros_like(nrm)
    for i, js in enumerate(nb):
        near = nrm[js]
        mean[i] = near[near @ nrm[i] > -0.5].sum(axis=0)
    norm = np.linalg.norm(mean, axis=1)
    good = norm > 1e-9
    cosang = np.clip((nrm[good] * mean[good]).sum(axis=1) / norm[good], -1, 1)
    return round(float(np.degrees(np.arccos(cosang)).mean()), 3)


def input_stats(vm) -> dict:
    """The calibration statistics (`synth.calibrate.voxel_stats`) plus fill and steps."""
    from synth.calibrate import voxel_stats
    st = {k: round(float(v), 4) for k, v in voxel_stats(vm.solid, vm.shell, vm.rgb).items()}
    st["fill_gain"] = vm.info.get("fill_gain")
    st["kept_fraction"] = vm.info.get("kept_fraction")
    st["stair45"] = stair45(vm.solid)
    return st


# --------------------------------------------------------------------------- one build

def _fingerprint(npz_path: Path) -> str:
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()[:16]
    return f"{digest}:{VERSION}:{settings_hash()}"


def process_build(row: dict, dataset: str, variants: int = VARIANTS, measure_mesh: bool = True) -> dict:
    """All variants of one build; never raises (a failure is a record with a reason)."""
    from .meshify import meshify
    from .normalize import save_npz

    dataset = Path(dataset)
    bid = row["id"]
    src = dataset / "normalized" / f"{bid}.npz"
    out_dir = dataset / "pairs" / "synthetic" / bid
    rec = {"id": bid, "split": row["split"], "category": row.get("category_top"),
           "building_category": row.get("building_category"), "version": VERSION,
           "settings_hash": settings_hash(), "status": "started", "variants": []}
    try:
        rec["fingerprint"] = _fingerprint(src)
        z = np.load(src)
        arrays = {k: z[k] for k in z.files}
        shape = arrays["target_shape"].shape
        rec["target_shape"] = [int(s) for s in shape]
        rec["input_shape"] = [int(s) * SCALE for s in shape]
        t0 = time.time()
        clean, minfo = meshify(arrays["blocks"], arrays["palette"], arrays["exterior"],
                               arrays["ground_height"])
        minfo["seconds"] = round(time.time() - t0, 2)
        rec["mesh"] = minfo
        from .meshify import geometry
        drawn = np.array([bool(g and g.boxes) for g in map(geometry, arrays["palette"])])
        problems, rec["frame_identity"] = frame_identity(clean, minfo, arrays["shell"] & drawn[arrays["blocks"]])
        if problems:
            raise ValueError("; ".join(problems))
        if measure_mesh:
            rec["mesh"]["normal_roughness"] = normal_roughness(clean.verts, clean.faces)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = {"target_shape": arrays["target_shape"], "target_material": arrays["target_material"],
                  "ground_height": arrays["ground_height"].astype(np.int16)}
        for k in range(variants):
            rec["variants"].append(_one_variant(bid, k, clean, shape, arrays, target, out_dir,
                                                save_npz, measure_mesh))
        rec["status"] = "ok"
    except Exception as exc:                                            # noqa: BLE001
        rec["status"] = "failed"
        rec["reason"] = f"{type(exc).__name__}: {exc}"[:300]
        rec["traceback"] = traceback.format_exc()[-2000:]
    return rec


def _one_variant(bid, k, clean, shape, arrays, target, out_dir, save_npz, measure_mesh):
    seed = variant_seed(bid, k)
    var = sample_variant(seed)
    row = {"variant": k, "seed": seed, "yaw_deg": round(var["yaw_deg"], 4),
           "offset_blocks": [round(v, 4) for v in var["offset_blocks"]], "degrade": var["degrade"],
           "file": f"synthetic/{bid}/v{k}.npz"}
    t0 = time.time()
    path = out_dir / f"v{k}.npz"
    try:
        vm, deg, info, reference = make_input(clean, shape, var)
    except Exception as exc:                                            # noqa: BLE001
        row.update(status="rejected", reason="degrade_failed", detail=f"{type(exc).__name__}: {exc}"[:200])
        path.unlink(missing_ok=True)
        return row
    row["timing"] = {"degrade": info["t_degrade"], "voxelize": info["t_voxelize"]}
    row["blur_sigma_cells"] = info["degrade"].get("blur_sigma_cells")
    row["bottleneck_cell_blocks"] = round(info["bottleneck_cell_blocks"], 4)
    row["degrade_info"] = {k2: info["degrade"].get(k2) for k2 in ("mc_faces", "faces", "holes", "floater_faces",
                                                               "back_facing_share", "open_edges", "components")}
    pair = {"input_occ": vm.solid.astype(np.uint8), "input_surface": vm.surface.astype(np.uint8),
            "input_rgb": np.ascontiguousarray(vm.rgb, dtype=np.uint8), **target}
    finite = bool(np.isfinite(deg.verts).all() and np.isfinite(deg.vcolor).all())
    # Degradation moves surfaces by up to about one bottleneck cell: the frame tolerance
    # is that, or SHIFT_TOL, whichever is larger.
    tol = max(SHIFT_TOL, SCALE * info["bottleneck_cell_blocks"])
    failures, metrics = validate_pair(pair, arrays["exterior"], reference, tol=tol)
    if not finite:
        failures.append("nan_inf:degraded_mesh")
    row["validation"] = {"failures": failures, **metrics}
    row["stats"] = input_stats(vm)
    if measure_mesh:
        from synth.blockiness import blockiness
        row["stats"]["normal_roughness"] = normal_roughness(deg.verts, deg.faces)
        b = blockiness(deg.verts, deg.faces)
        row["stats"]["lattice16"], row["stats"]["axis_area"] = b["lattice16"], b["axis_area"]
    if failures:
        row.update(status="rejected", reason=failures[0].split(":", 1)[0], detail="; ".join(failures)[:300])
        path.unlink(missing_ok=True)
    else:
        save_npz(path, pair)
        row.update(status="ok", bytes=path.stat().st_size)
    row["timing"]["total"] = round(time.time() - t0, 2)
    return row


# --------------------------------------------------------------------------- CLI

def pilot_ids(rows: list, n: int) -> list:
    """`n` builds across categories and sizes, deterministic: round-robin over categories,
    each category's builds shuffled with a fixed seed."""
    by_cat = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        by_cat.setdefault(r["category_top"], []).append(r["id"])
    rng = random.Random(PILOT_SEED)
    for ids in by_cat.values():
        rng.shuffle(ids)
    out = []
    while len(out) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(out) < n:
                out.append(by_cat[cat].pop())
    return out


def _write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _percentiles(values, ps=(10, 50, 90)):
    v = np.array([x for x in values if x is not None], dtype=np.float64)
    return [round(float(np.percentile(v, p)), 4) for p in ps] if len(v) else None


def summarize(records: list) -> dict:
    rows = [(r, v) for r in records if r.get("status") == "ok" for v in r["variants"]]
    ok = [v for _, v in rows if v["status"] == "ok"]
    rejected = [v for _, v in rows if v["status"] != "ok"]
    reasons = {}
    for v in rejected:
        reasons[v["reason"]] = reasons.get(v["reason"], 0) + 1
    val = [v["validation"] for _, v in rows if "validation" in v]
    out = {"builds": len(records), "builds_ok": sum(r.get("status") == "ok" for r in records),
           "builds_failed": [{"id": r["id"], "reason": r.get("reason")} for r in records
                             if r.get("status") == "failed"],
           "pairs": len(rows), "pairs_ok": len(ok), "pairs_rejected": len(rejected),
           "rejected_by_reason": reasons,
           "rejected": [{"id": r["id"], "variant": v["variant"], "reason": v["reason"],
                         "detail": v.get("detail")} for r, v in rows if v["status"] != "ok"],
           "by_split": {}, "bytes": sum(v.get("bytes", 0) for v in ok),
           "seconds": round(sum(v.get("timing", {}).get("total", 0) for _, v in rows)
                            + sum(r.get("mesh", {}).get("seconds", 0) for r in records), 1)}
    for r, v in rows:
        s = out["by_split"].setdefault(r["split"], {"ok": 0, "rejected": 0})
        s["ok" if v["status"] == "ok" else "rejected"] += 1
    p = (5, 10, 50, 90, 95)
    out["iou"] = dict(zip(["p5", "p10", "p50", "p90", "p95"], _percentiles([m.get("iou") for m in val], p) or []))
    out["coverage"] = dict(zip(["p5", "p10", "p50", "p90", "p95"],
                               _percentiles([m.get("coverage") for m in val], p) or []))
    res = [abs(x) for m in val for x in m.get("shift_residual_cells", [])]
    out["shift_residual_abs_cells"] = dict(zip(["p50", "p90", "p99", "max"],
                                               _percentiles(res, (50, 90, 99, 100)) or []))
    ious = sorted((m["iou"], r["id"], v["variant"]) for r, v in rows
                  for m in [v.get("validation", {})] if "iou" in m)
    out["iou_lowest"] = [{"id": b, "variant": k, "iou": i} for i, b, k in ious[:10]]
    flagged = [(r["id"], v["variant"], v["validation"]["iou"], v["validation"].get("coverage"))
               for r, v in rows if v["status"] == "ok" and "low_iou" in v.get("validation", {}).get("flags", [])]
    out["flagged_low_iou"] = {"pairs": len(flagged), "builds": sorted({f[0] for f in flagged}),
                              "rows": [{"id": b, "variant": k, "iou": i, "coverage": c} for b, k, i, c in flagged]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="build ids to (re)process")
    ap.add_argument("--pilot", type=int, default=0, help="only the first N builds of `pilot_ids`")
    ap.add_argument("--variants", type=int, default=VARIANTS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="redo builds that are current")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    dataset = args.dataset.resolve()
    out = dataset / "pairs"
    if out.resolve().is_relative_to((dataset / "source").resolve()):
        raise SystemExit("refusing to write inside source/")
    (out / "records").mkdir(parents=True, exist_ok=True)
    index_rows = {r["id"]: r for r in json.loads((dataset / "index.json").read_text(encoding="utf-8"))["rows"]}
    norm = json.loads((dataset / "normalized" / "index.json").read_text(encoding="utf-8"))
    rows = [dict(index_rows[b["id"]], split=b["split"]) for b in norm["builds"]
            if b.get("status") == "ok" and b["split"] != "excluded"]
    chosen = rows
    if args.pilot:
        keep = set(pilot_ids(rows, args.pilot))
        chosen = [r for r in rows if r["id"] in keep]
    if args.only:
        unknown = set(args.only) - {r["id"] for r in rows}
        if unknown:
            raise SystemExit(f"not normalized (or excluded): {sorted(unknown)}")
        chosen = [r for r in rows if r["id"] in args.only]

    def current(r):
        path = out / "records" / f"{r['id']}.json"
        if not path.is_file():
            return False
        rec = json.loads(path.read_text(encoding="utf-8"))
        return (rec.get("status") == "ok" and len(rec.get("variants", [])) == args.variants
                and rec.get("fingerprint") == _fingerprint(dataset / "normalized" / f"{r['id']}.npz")
                and all((out / v["file"]).is_file() for v in rec["variants"] if v["status"] == "ok"))

    todo = [r for r in chosen if args.force or not current(r)]
    # Largest first, so the long builds start early and do not finish the run alone.
    size = {b["id"]: int(np.prod(b["crop"]["dims_after"])) for b in norm["builds"]}
    todo.sort(key=lambda r: -size.get(r["id"], 0))
    print(f"{len(chosen)} builds chosen, {len(todo)} to process, {args.variants} variants, "
          f"{args.workers} worker(s); settings {settings_hash()}", flush=True)
    t0 = time.time()

    def handle(rec):
        _write_json(out / "records" / f"{rec['id']}.json", rec)
        if rec["status"] != "ok":
            print(f"  {rec['id']}: FAILED {rec.get('reason')}", flush=True)
            return
        marks = " ".join(("ok" if v["status"] == "ok" else v["reason"])
                         + ("*" if "low_iou" in v.get("validation", {}).get("flags", []) else "")
                         + "({})".format(v.get("validation", {}).get("iou")) for v in rec["variants"])
        print(f"  {rec['id']}: {marks}", flush=True)

    if args.workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for rec in pool.map(process_build, todo, [str(dataset)] * len(todo),
                                [args.variants] * len(todo)):
                handle(rec)
    else:
        for r in todo:
            handle(process_build(r, str(dataset), args.variants))
    wall = round(time.time() - t0, 1)

    records = []
    for r in rows:
        path = out / "records" / f"{r['id']}.json"
        if path.is_file():
            rec = json.loads(path.read_text(encoding="utf-8"))
            rec["split"] = r["split"]
            records.append(rec)
    summary = summarize(records)
    summary["wall_seconds_last_run"] = wall
    index = {"dataset": dataset.name, "version": VERSION, "settings": settings(), "summary": summary,
             "pairs": [{"id": rec["id"], "split": rec["split"], "category": rec.get("category"),
                        "variant": v["variant"], "seed": v["seed"], "status": v["status"],
                        "reason": v.get("reason"), "file": v["file"] if v["status"] == "ok" else None,
                        "target_shape": rec.get("target_shape"), "input_shape": rec.get("input_shape"),
                        "yaw_deg": v["yaw_deg"], "offset_blocks": v["offset_blocks"],
                        "degrade": v["degrade"],
                        "validation": v.get("validation"), "stats": v.get("stats"),
                        "bytes": v.get("bytes")}
                       for rec in records if rec.get("status") == "ok" for v in rec["variants"]],
             "builds_failed": summary["builds_failed"]}
    _write_json(out / "index.json", index)
    s = summary
    print(f"pairs ok {s['pairs_ok']}, rejected {s['pairs_rejected']} {s['rejected_by_reason']}; "
          f"builds failed {len(s['builds_failed'])}")
    print(f"IoU {s['iou']}  shift residual |cells| {s['shift_residual_abs_cells']}")
    print(f"disk {s['bytes'] / 1e6:.1f} MB; this run {wall} s wall")
    return 0 if not s["builds_failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
