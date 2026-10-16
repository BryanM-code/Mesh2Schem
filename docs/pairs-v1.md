# pairs-v1: (input, target) training pairs from real-schems-v1

Phase 4 of the schematic-first plan ([mesh-to-schem-plan.md](mesh-to-schem-plan.md), D16).
No model is trained here. This file is the contract between the pair generator
(`schemset.pairs`) and the Phase 5 loader.

Everything listed here lives under `data/real-schems-v1/pairs/` and is local only. Like all of
`data/`, it is gitignored, and under D14 it is never redistributed.

## What a pair is

| | |
| --- | --- |
| Target | `target_shape` / `target_material` of `normalized/<id>.npz`, unchanged (mcvox-v3, D20; IGNORE = 255 / 65535) |
| Input | a TRELLIS-like reconstruction of the build's **outside**, voxelised at **2 cells per block** in **exactly the target's frame** (D16) |
| Per build | `VARIANTS` = 4 inputs, each from its own seed |
| Builds | every normalized build of split train, val and test. Split "excluded" is never used |

How one input is made (`schemset.pairs`, version `pairs-2`; `pairs-1` had the old colours,
see [Colour model](#colour-model-pairs-2)):

1. **Mesh** (`schemset.meshify`). The build's outside becomes a coloured surface in block
   units, in the target's frame. Only faces that meet exterior space are emitted. Exterior
   space is the D22 exterior-air mask plus the empty part of non-cube blocks in the shell
   layer that the outside reaches.
   - Block geometry is real, to 1/16 block: slabs, stairs (both halves, with corner shapes),
     panes and bars, fences, walls, gates, trapdoors (open and closed), doors, carpets, snow
     layers.
   - Fluids are a 14/16 box. Leaves are cubes. Decor is approximated (beds, chests, lanterns,
     pots, ...) or skipped (plants, torches, signs, ...).
   - Air that is not exterior gets a dark **opening cap** where the outside meets it: a
     doorway, an open window, the far end of an open hall beyond the D22 depth cap.
   - Interiors, furniture behind glass, the inner side of walls and the ground are never
     meshed. There is no ground plate.
   - Undrawn decor does not stop the outside. The D22 flood treats decor as solid, so air
     enclosed only by decor (pockets in a plume of cobwebs) comes out interior. Here,
     exterior air also flows through decor and through "interior" air with no drawn block
     anywhere above it, since a real room always has a drawn roof. Roofed rooms (the
     depth-capped halls too) stop it. Roofed decor passes it on only where the decor itself
     touches exterior air, so a cobweb in a doorway never opens the room behind it.
     Unreached decor counts as room air.
   - One exception keeps features flush with the terrain. A block *in* the ground layer that
     the outside reaches (a path, a patio, a paved plaza) gets dark faces where it meets the
     ground, so it is a closed slab rather than a zero-thickness sheet that degradation would
     erase. The targets supervise those blocks: 1.6% of all supervised blocks, and over 5% in
     55 builds (`meshify-2`).
   - Legacy formats that stored no pane, fence or wall connections or stair corners get them
     derived by the game's rules, for the geometry only.
   - Glass is `KIND_GLASS` and doors are `KIND_DOOR`. Colours are `block_colors` thumbnail
     colours (see [Domain gaps](#domain-gaps-and-caveats)). Per variant, glass is then
     recoloured dark and slightly cool, as windows read in photos (`pairs.glass_look`,
     [Colour model](#colour-model-pairs-2)).
2. **Variant.** A yaw of ±`YAW_DEG` = 5° about the build's vertical centre line, and an
   offset of ±`OFFSET_BLOCKS` = 0.5 block on x, y and z. Both are uniform and applied to the
   clean surface. This is the misregistration a real placement leaves. The target is never
   moved.
3. **Degrade** (`synth.degrade.degrade`; geometric ranges from
   `synth.config.PipelineConfig.degrade`, colour ranges from `pairs.COLOUR`; sampled by
   `sample_params`):
   - a resolution bottleneck of 128, 160 or 192 cells across the build's longest axis;
   - blur, surface noise, holes and floaters;
   - baked colour: exposure, a directional sun, hue drift, muted chroma and noise
     ([Colour model](#colour-model-pairs-2));
   - a hallucinated back side.
4. **Smoothing floor** (the domain-gap guard, [below](#domain-gap-guard-blockiness)). The
   bottleneck blur has sigma ≥ `SMOOTH_SIGMA_BLOCKS` block, so one-block Minecraft stair steps
   do not survive into small builds' inputs.
5. **Voxelise** (`schemset.pairs.voxelize_input` = `mcvoxel.grid.voxelize_scene` at 2 cells per
   block):
   - origin (0, 0, 0), shape 2 × the target's;
   - surface raster, exterior flood and enclosed fill (`occluded_interior`);
   - no crop, no ground-plate removal, no component cleanup.

   The inference voxeliser must do the same to TRELLIS meshes.
6. **Validate** ([Validation](#validation)) and write, or reject with a reason.

## Files

```
pairs/
  synthetic/<id>/v<k>.npz      one compressed npz per (build, variant), k = 0..3
  records/<id>.json            per build: mesh info, frame check, per variant everything below
  index.json                   one row per (build, variant) + summary
  trellis_reference.json       statistics of the cached TRELLIS meshes (schemset.pairs_calibrate)
  calibration.json             synthetic vs TRELLIS, p10 / p50 / p90 (schemset.pairs_calibrate)
  real_trellis_smooth/         real-TRELLIS evaluation, smooth photos: the main set (schemset.real_trellis)
    selection.json, <id>/photo.png, <id>/trellis.glb, <id>/input.npz, <id>/stats.json, index.json
    trials/sigma_<s>/, trials.json   the smoothing trials that chose the photo sigma
  real_trellis/                real-TRELLIS evaluation, blocky photos: the stress test (same layout)
  history/pairs-1/             pairs-1's index.json, calibration.json and real-TRELLIS index (the "before")
  gallery/index.html           review gallery (schemset.pairs_gallery)
```

`index.json` rows contain:

- build `id`, `split`, `category`, `variant`, `seed`, `status` (ok / rejected), `reason`, `file`;
- `target_shape` and `input_shape`;
- `yaw_deg`, `offset_blocks`, `degrade` (every sampled parameter) and `glass` (the glass look);
- `validation` (failures, `iou`, `coverage`, `excess`, `reference_iou`,
  `shift_residual_cells`, `shift_tol_cells`);
- `stats` (calibration and blockiness, below) and `bytes`.

`summary` has counts by split and reason, distributions, disk size and timing.

The npz files are written by `schemset.normalize.save_npz`: zip members carry a fixed
timestamp, so the same build, variant and code give byte-identical files (tested).
`variant_seed(id, k)` is a hash of the dataset, the seed salt `pairs-1`, the build id and the
variant. It does not depend on run order, worker count, machine or the pairs version: a
colour-only version change (pairs-1 → pairs-2) keeps every variant's geometry voxel for
voxel. The colour ranges change the values drawn, never the number or order of draws.

## Arrays and channels

| key | dtype | shape | meaning |
| --- | --- | --- | --- |
| `input_occ` | uint8 0/1 | (2X, 2Y, 2Z) | occupied after the voxeliser's fill: the reconstruction is a solid blob, like a voxelised TRELLIS mesh |
| `input_surface` | uint8 0/1 | (2X, 2Y, 2Z) | cells the reconstructed surface passes through (conservative raster), ⊆ `input_occ` |
| `input_rgb` | uint8 | (2X, 2Y, 2Z, 3) | **sRGB** (gamma-encoded, 0–255), as seen from outside. Set on the *shell* of `input_occ` (occupied cells with an empty 6-neighbour; below the grid counts as ground, not air), 0 elsewhere |
| `target_shape` | uint8 | (X, Y, Z) | mcvox-v3 shape id; 255 = IGNORE |
| `target_material` | uint16 | (X, Y, Z) | mcvox-v3 material id; 65535 = IGNORE (exactly where the shape is IGNORE) |
| `ground_height` | int16 | (X, Z) | the ground surface of each column in this grid (≤ it is ground); −1 = none (D23) |

There is deliberately **no kind (glass / door) channel**. TRELLIS gives the app no material
kinds, so such a channel would exist only in training. There are **no normals** either. If
the model wants them, the loader derives them from `input_occ` (for example the gradient of a
Gaussian-blurred occupancy), which inference can do the same way. The shell mask is
`occ & (any 6-neighbour empty)`, with the layer below the grid counted as occupied.

## Frame convention

- Arrays are (X, Y, Z): +X east, +Y up, +Z south. This is the Sponge `.schem` (Y, Z, X) order
  transposed, as everywhere in mcvoxel.
- Target block (x, y, z) spans [x, x+1) × [y, y+1) × [z, z+1) in block units. Input cell
  (i, j, k) spans [i/2, (i+1)/2) × … in the same units, with the **same origin**. So target
  block (x, y, z) covers input cells (2x..2x+1, 2y..2y+1, 2z..2z+1). Input shape is exactly
  2 × target shape.
- The frame is exact; the *reconstruction* in it is misregistered on purpose: ±5° yaw and
  ±0.5 block offset. The model has to tolerate that.
- mcvoxel's rasteriser puts a face that lies *exactly* on a cell boundary in the cell on its
  + side (cell = floor(coordinate)). Degraded and offset surfaces never sit exactly on a
  boundary, and neither do TRELLIS meshes, so the inputs carry no such bias. A clean,
  unshifted block mesh would carry it (+0.5 cell). Keep that in mind for any debugging
  voxelisation.

## Loader contract (Phase 5)

1. **Splits.** Use `split` from `index.json`, which is the D17 creator split. Never mix splits.
   The real-TRELLIS set comes only from val and test. Pick one variant per build per epoch
   (or all four); variants of one build never go to different splits.
2. **Windows: 64³ target, 128³ input.**
   - A target window starting at block `s = (sx, sy, sz)` pairs with the input window starting
     at cell `2s`: `target[s : s+64]` ↔ `input[2s : 2s+128]` on every axis.
   - An axis shorter than 64 is padded on its high side: the target with IGNORE (255 /
     65535), never air, because nothing is known beyond the capture; the input with 0.
   - Training samples starts uniformly over `[0, max(dim − 64, 0)]` per axis. It may reject
     windows with almost no supervised solid voxels.
   - Evaluation and inference tile with `mcvoxel.tiling.window_starts` and blend with
     `tiling.ramp`, exactly as the app will.
3. **Rotations and mirrors: the same transform on both grids.**
   - Rotate by k × 90° about +Y.
     - Target: `mcvoxel.blockstates.rotate_y(target_shape, k, target_material)`. This rotates
       both arrays with `np.rot90(k, axes=(0, 2))` and remaps every oriented shape label:
       stairs (both halves), open trapdoors, doors, and the axis shapes `full_axis_x/z` and
       `fence_gate_x/z` when k is odd.
     - Input: `np.rot90(a, k, axes=(0, 2))` on `input_occ`, `input_surface` and `input_rgb`
       (the channel axis is untouched).
     - `ground_height`: `np.rot90(g, k, axes=(0, 1))`.
   - Mirror east ↔ west.
     - Target: `blockstates.mirror_x(target_shape, target_material)`.
     - Input and `ground_height`: `a[::-1]` on axis 0.
   - Rotations and the mirror give all 8 symmetries of the square. Because a 2-cell block
     maps to a 2-cell block under `rot90` and `[::-1]`, the window correspondence above is
     preserved. Transform the windows after cropping (cubes stay cubes) or the whole grids
     before; both are equivalent.
   - Material ids never change under these transforms. Never rotate by angles that are not
     multiples of 90°: the labels cannot follow, and the data already carries ±5°.
4. **IGNORE in the loss.**
   - Shape loss: cross-entropy over all voxels whose shape is not 255. This **includes
     exterior air** (shape 0), which is supervised.
   - Material loss: only where the target shape is neither air nor IGNORE.
   - With PyTorch, use `ignore_index=255` for shapes and `ignore_index=65535` for materials,
     with targets cast to int64.
   - Padded voxels are IGNORE.
   - At inference, mask invalid (shape, material) logits with `mcvoxel.vocab.VALID`.
5. **Input normalisation.**
   - `occ` and `surface` as 0/1 floats.
   - `rgb / 255` (keep it sRGB; the calibration statistics are in CIELAB from sRGB), zero off
     the shell.
   - Optionally add a has-colour channel (the shell mask). Every channel must be derivable at
     inference.

## Validation

Each pair is checked before it is written. A failure **rejects** the pair with its reason in
`index.json` and `records/` (nothing is dropped silently).

| check | rejects when | reason |
| --- | --- | --- |
| Arrays | a key missing, a shape not exactly 2× / 1×, a dtype not as above | `missing`, `shape`, `dtype` |
| Finite | NaN or inf in any float array, or in the degraded mesh | `nan_inf` |
| Binary | `input_occ` / `input_surface` not 0/1 | `input_not_binary` |
| Target | IGNORE shape and material disagree, or a supervised (shape, material) not in `VALID` | `target_ignore_mismatch`, `target_invalid_pairs` |
| Input | nothing occupied | `empty_input` |
| IoU band | block-level IoU of the input (a block is occupied when ≥ half its 8 cells are) with the target volume (not exterior air, above the ground) < `IOU_MIN` = 0.2 | `iou_below_band` |
| Coverage | the input covers < `COVER_MIN` = 50% of the target volume: the reconstruction lost the building | `input_misses_building` |
| Frame | the shift that best overlays the input on the *reference* exceeds the tolerance on any axis (see below) | `frame_misaligned` |
| Build frame | a vertex off the 1/16 lattice or outside the grid; the surface falling more than 1 block short of the shell blocks that have geometry on any side; or an opening cap more than 1 block outside the meshed blocks | build fails `frame_identity` (all its variants) |
| Degrade | `synth.degrade` raised (the model vanished at the bottleneck) | `degrade_failed` |

IoU below `IOU_FLAG` = 0.45 with coverage ≥ 50% is **kept and flagged** `low_iou` (in
`validation.flags` and `summary.flagged_low_iou`). The frame is checked precisely against
the reference, so a low IoU there means a blob much bigger than the build: a sprawling
build with courtyards and arcades at a coarse bottleneck, or a barn whose open interior
D22 trains as air. Such pairs are correctly framed and hard, not broken. The first
version rejected below 0.35. That would have dropped `pmc-colonial-corivian` (IoU
0.32–0.36, coverage 0.80–0.87, frame residual under 0.6 cells against a tolerance over 2),
so the band became flag-or-reject as above.

How the frame check works:

- The *reference* is the clean surface placed by the same yaw and offset and voxelised the
  same way, without degradation. Comparing against it removes what the augmentation does on
  purpose, leaving only what degradation does.
- The shift is the centre of the IoU-against-shift curve within 1% of its maximum, searched
  ±4 cells per axis.
- The tolerance is max(1 cell, one bottleneck cell). Degradation can move a surface by up to
  about one bottleneck cell, and on a 200-block tower that is 2–3 input cells.
- A deliberate 2-block (4-cell) shift is rejected (tested).

The build frame check runs once per build, before any variant. It caught a real meshify bug.
A smoke plume of cobwebs above the 19th-century brewery's chimney enclosed air that D22
calls interior, because its flood treats decor as solid. Meshify skips cobwebs, so that
air got opening caps floating 17 blocks above the building. Since `meshify-3`, undrawn
decor and air with no drawn roof above it pass the outside on (see "Mesh" above). A
cobweb in a doorway still does not open the room behind it (tested).

## Statistics per pair

Calibration (`synth.calibrate.voxel_stats` on the input):

- `rough3`: share of shell cells with 3+ open faces (blobby).
- `lum`, `chroma`: mean L* and C* of the shell colours.
- `nbr_de`: mean ΔE between face-adjacent shell cells.
- `fill_gain`: filled cells per surface cell.
- `kept_fraction`: always 1, as nothing is cleaned.

Blockiness:

- `stair45`: on 45° stretches of the top surface, the share of (0, 2) step pairs (a
  one-block stair: flat tread, one-block riser) against (1, 1) (a smooth slope). Block
  stairs give ~1, smooth roofs ~0.
- `normal_roughness`: the mean angle between the degraded surface's normal and its mean
  within 1 block, same side only.
- `lattice16` and `axis_area`: `synth.blockiness`, on the degraded mesh.

Frame metrics:

- `iou`, `coverage` (target volume covered) and `excess` (input outside the target volume).
- `reference_iou`: the input against the clean placed reference, at 2×.

## Domain-gap guard (blockiness)

Measured on the 20-build pilot (2 variants each) against the 53 cached TRELLIS meshes
(buildings only), size-matched: each synthetic build was compared with TRELLIS meshes scaled
to the nearest of 36 / 57 / 87 blocks. See `schemset.pairs_calibrate`.

| blur floor σ (blocks) | stair45 p50 (TRELLIS ≈ 0.25) | normal roughness p50 (TRELLIS 8.1°) | verdict |
| --- | --- | --- | --- |
| 0 (degrade as calibrated) | 0.39. At ~36 blocks: 0.92 vs TRELLIS 0.10 | 10.0°. At ~57 blocks: 12.9 vs 7.8 | blockier on both |
| **0.42 (FWHM 1 block)** | 0.22, 100% of synthetic inside TRELLIS p10–p90 | 6.2°, 96% inside | **overlaps; chosen** |
| 0.6 | 0.12 | 4.6°, 71% inside | smoother than TRELLIS |
| 0.8 | 0.23 | 3.5°, 12% inside | too smooth |

`SMOOTH_SIGMA_BLOCKS` = 0.42. Big builds are unaffected, because their bottleneck blur is
already wider than that. Real TRELLIS meshes are not stair-free either: their 64³ structure
terraces a mesh more the more blocks it spans, with stair45 0.10 at 36 blocks and 0.33 at
87. That is why the comparison must be size-matched.

An earlier metric, the share of 2-cell steps, was dropped: a smooth roof steeper than 63°
gives the same steps.

Confirmed on the full run (all 1,268 pairs, 940 of them within the reference sizes):
synthetic is **not blockier** than TRELLIS on either metric. Size-matched p10 / p50 / p90:

| | stair45 synthetic | stair45 TRELLIS | roughness synthetic | roughness TRELLIS |
| --- | --- | --- | --- | --- |
| all sizes | 0.04 / 0.21 / 0.49 | 0.02 / 0.25 / 0.70 | 5.4 / 7.8 / 10.4° | 4.2 / 8.1 / 13.6° |
| ~36 blocks (n 388 / 50) | 0.03 / 0.16 / 0.49 | 0.02 / 0.10 / 0.80 | 5.9 / 8.1 / 10.4° | 8.3 / 10.0 / 15.0° |
| ~57 blocks (n 292 / 50) | 0.06 / 0.21 / 0.42 | 0.02 / 0.22 / 0.70 | 5.3 / 7.9 / 10.2° | 6.1 / 7.8 / 12.2° |
| ~87 blocks (n 260 / 50) | 0.07 / 0.27 / 0.55 | 0.02 / 0.33 / 0.68 | 4.7 / 7.1 / 10.5° | 3.9 / 4.9 / 9.2° |

## Colour model (pairs-2)

pairs-1 used `synth.degrade`'s colour ranges, which were tuned for already shaded Objaverse
textures. On flat, unshaded thumbnail colours they made inputs about twice as bright and
1.5× as colourful as TRELLIS on real photos, and glass was light blue where real windows read
dark. pairs-2 overrides the colour ranges for the pairs only; the parked synth pipeline keeps
its defaults, and the geometric ranges are untouched.

Per variant, all drawn from its seed:

| | pairs-1 | pairs-2 (`pairs.COLOUR`) |
| --- | --- | --- |
| exposure (`tone_gain`, an sRGB gain) | uniform 0.90–1.05 | **log-uniform 0.37–0.95** |
| sun (`ambient` = the floor of the shading) | 0.85–0.98 (nearly flat) | 0.50–0.85 (directional) |
| chroma scale | 0.85–1.10 | 0.95–1.25, compressed towards C\* 8 (`chroma_gamma` 0.75: strong colours muted most, as TRELLIS does) |
| hue drift (sd) | 3° | 4° |
| colour noise (RGB sd) | 2–5 | 3–7 |
| glass (`pairs.GLASS`, CIELAB before exposure) | thumbnail light blue | L\* 20–38, a\* −3…1, b\* −10…−3, per-window jitter 2–6, 0–25% of windows catch the sky (+8–20 L\*) |

`chroma_gamma` is a new optional `synth.degrade` parameter (default 1 = off):
C' = chroma · 8 · (C / 8)^gamma.

How it was fitted:

- The sample was 60 builds spread over sizes, variant 0 each, against the TRELLIS-on-real-photo
  reference (`trellis_buildings`).
- Eight candidates were tried. Darkening alone made inputs too dark and too narrow (L\* p90 25).
  Raising the exposure ceiling alone made them too bright (L\* p50 29–31). Neither got chroma
  to 70% inside the band without compression.
- Glass, measured over input cells whose target block is glass, reads L\* 12 / 24 / 39. The
  high end is wall colour bleeding into small windows at coarse bottlenecks.

Full set, before and after (p10 / p50 / p90; "inside" = share of inputs inside the
reference p10–p90):

| | TRELLIS on real photos (150) | pairs-1 (1,268) | pairs-2 (1,268) | target |
| --- | --- | --- | --- | --- |
| L\* | 11.0 / 21.9 / 33.7 | 33.8 / 46.2 / 63.3 | **12.4 / 22.0 / 36.5** (78% inside) | 10–14 / 20–26 / 36–45 |
| C\* | 2.8 / 6.7 / 11.6 | 3.2 / 10.8 / 20.1 | **2.6 / 6.0 / 11.5** (79% inside) | p50 6–8, p90 ≤ 14 |
| nbr_de | 1.53 / 2.48 / 4.99 | 1.40 / 2.86 / 4.78 | 1.19 / 2.24 / 3.53 | p50 inside the band |
| stair45, roughness, fill_gain, rough3 | | | identical to pairs-1 (same geometry) | unchanged |

All acceptance checks pass (`pairs_calibrate.colour_acceptance`):

- The synthetic median is inside the reference p10–p90 for L\*, C\* and nbr_de.
- At least 70% of inputs are inside the band for L\* and C\*.
- `pairs.COLOUR_TARGET` is met.

## Calibration against TRELLIS

`python -m schemset.pairs_calibrate --dataset data/real-schems-v1 --before <old index.json>`
writes `pairs/calibration.json`: every statistic for the TRELLIS reference, the synthetic
inputs (now and before), the size-matched subset and the two real-TRELLIS sets, plus the
blockiness overlap and the colour acceptance. pairs-2, p10 / p50 / p90:

| statistic | TRELLIS on real photos (150) | synthetic, all (1,268) | synthetic, size-matched (940) |
| --- | --- | --- | --- |
| stair45 | 0.02 / 0.25 / 0.70 | 0.04 / 0.20 / 0.49 | 0.04 / 0.21 / 0.49 |
| normal roughness | 4.2 / 8.1 / 13.6° | 4.0 / 7.5 / 10.3° | 5.4 / 7.8 / 10.4° |
| fill_gain | 2.69 / 5.45 / 10.92 | 3.10 / 5.17 / 10.09 | 3.22 / 5.11 / 8.42 |
| rough3 (blobby shell) | 0.018 / 0.029 / 0.074 | 0.026 / 0.053 / 0.100 | 0.027 / 0.055 / 0.097 |
| L\* | 11.0 / 21.9 / 33.7 | 12.4 / 22.0 / 36.5 | 12.5 / 21.8 / 35.8 |
| C\* | 2.8 / 6.7 / 11.6 | 2.6 / 6.0 / 11.5 | 2.6 / 5.9 / 11.2 |
| nbr_de | 1.53 / 2.48 / 4.99 | 1.19 / 2.24 / 3.53 | 1.44 / 2.38 / 3.53 |
| axis_area | 0.13 / 0.46 / 0.56 | 0.10 / 0.26 / 0.48 | 0.11 / 0.26 / 0.47 |

How to read it:

- **Blockiness** matches: synthetic is not blockier on either metric, at any size
  ([guard](#domain-gap-guard-blockiness)).
- **Colour** matches since pairs-2 ([Colour model](#colour-model-pairs-2)).
- **The shell is about twice as blobby** as TRELLIS's (rough3 0.053 against 0.029); the fill
  matches.
- **Axis-facing area is lower.** The degraded surfaces carry noise and ±5° yaw, while the
  reference meshes were yaw-aligned before measuring. This is not a blockiness signal.

## Real-TRELLIS evaluation sets

Both sets hold the same 25 builds from val (12) and test (13): civic 4, commercial 4,
highrise 4, industrial 4, residential 4, hospitality 3, agricultural 2 (`select`).
`schemset.real_trellis` makes them: `prepare` (photos), `generate` (TRELLIS server, seed
1), `place` (alignment, 2× voxelisation, validation, gap report).

| set | folder | photo | role |
| --- | --- | --- | --- |
| **smooth** | `real_trellis_smooth/` | the build's outside, glass dark (L\* 30, cool), **blurred by σ = 0.6 block** and re-extracted, colours from the nearest point of the clean surface, flat-ish daylight (`PHOTO_LIGHT` 0.72 + 0.28 · sun), plain background | **the main evaluation set** |
| **blocky** | `real_trellis/` | a render of the raw block build (thumbnail colours, `render_photo`'s default light) | **the stress test**: TRELLIS reproduces Minecraft's stair-stepped roofs, which real photos never show |

### Choosing the smoothing (`trials`)

The trial photos of 5 builds (a fast-food restaurant, two bungalows, a train station, a
skyscraper) went through TRELLIS at each σ. TRELLIS's output against TRELLIS on real photos
(stair45 p50 ≈ 0.25, roughness p50 ≈ 8°), p10 / p50 / p90:

| σ (blocks) | stair45 | roughness | IoU |
| --- | --- | --- | --- |
| **0.6 (chosen)** | 0.11 / **0.24** / 0.59 | 6.6 / **8.3** / 10.3° | 0.44 / 0.70 / 0.75 |
| 0.8 | 0.07 / 0.17 / 0.47 | 5.0 / 5.9 / 8.5° | 0.42 / 0.71 / 0.73 |
| 1.0 | 0.04 / 0.18 / 0.24 | 4.0 / 5.8 / 7.0° | 0.43 / 0.73 / 0.78 |

A first version took colours from degrade's voxels; windows and dark caps smeared into the
walls as speckle. Colours now come from the nearest of 2 M points on the clean surface.

### Alignment (eval sets only; training pairs keep their ±0.5-block offsets)

`place` fits each TRELLIS mesh to the target as follows:

1. Scale it to the target's height, stand it on its lowest layer and centre it on the
   footprint.
2. Search yaw every 5° at 1 cell per block and keep the best 3.
3. For each of those, search translation ±`ALIGN_REACH` = 16 cells per axis at 2 cells per
   block (`search_shift`: 2-cell steps, then 1-cell steps, then a joint 3×3×3 step).
4. Refine yaw ±4° at 1 cell per block with that shift (a local 2× search if the yaw moves).
5. Move the shift from the IoU peak to where the validator sees no residual
   (`centre_shift`).

Three things were needed beyond widening the search:

- **The validator measures the centre of the IoU curve's top, not its peak.** On real
  TRELLIS output the top is broad and lopsided: the peak can sit several cells (up to 11) from it.
  `centre_shift` moves the shift by the residual the validator reports until it is under
  half a cell. It may leave the search's reach (the cemetery chapel needed −18 cells).
- **Voxelise once, then crop the window.** Each candidate is voxelised in a frame that
  holds the whole mesh plus a margin. A mesh cut at the frame edge lets the exterior flood
  in and loses its fill, which is what the old stored inputs did near the edges: the parish
  church went from IoU 0.60 (old placement) to 0.80. The stored input is exactly the window the search
  scored. Moving the mesh and re-voxelising it changed up to 11% of the cells.
- **Validation is unchanged:** `pairs.validate_pair`, residual within `SHIFT_TOL` = 1 cell.

Before (old placement: quarter turns, centring, no shift) and after, blocky set:

| | misaligned | residual, max axis p10 / p50 / p90 (cells) | IoU p10 / p50 / p90 |
| --- | --- | --- | --- |
| before | 16 / 25 | 0.47 / 2.24 / 4.0 (the ±4 search's limit) | 0.39 / 0.62 / 0.87 |
| after | **0 / 25** | 0.10 / 0.31 / 0.47 | 0.56 / 0.65 / 0.87 |
| smooth set, after | **0 / 25** | 0.12 / 0.37 / 0.48 | 0.40 / 0.68 / 0.83 |

Shifts applied, max axis: p50 2–3, p90 10–13, at most 12 (blocky) and 18 (smooth) cells. One build fails in both
sets for a real reason, not alignment: `ab-11538`, where TRELLIS covers only 31% (smooth)
or 47% (blocky) of the building (`input_misses_building`). Tests: an 8-cell shift is
recovered exactly, and `align` end to end validates a turned mesh.

### TRELLIS statistics per set (p10 / p50 / p90)

| statistic | TRELLIS on real photos (150) | TRELLIS on smooth photos (25) | TRELLIS on blocky photos (25) | synthetic pairs-2 (1,268) |
| --- | --- | --- | --- | --- |
| stair45 | 0.02 / 0.25 / 0.70 | 0.05 / **0.22** / 0.55 | 0.35 / **0.88** / 0.95 | 0.04 / 0.20 / 0.48 |
| normal roughness | 4.2 / 8.1 / 13.6° | 3.0 / **7.7** / 11.9° | 4.4 / **18.4** / 26.5° | 4.0 / 7.5 / 10.3° |
| fill_gain | 2.69 / 5.45 / 10.92 | 3.12 / 4.74 / 11.29 | 1.92 / 4.04 / 10.64 | 3.10 / 5.17 / 10.09 |
| rough3 | 0.018 / 0.029 / 0.074 | 0.027 / 0.068 / 0.127 | 0.013 / 0.035 / 0.086 | 0.026 / 0.053 / 0.100 |
| L\* | 11.0 / 21.9 / 33.7 | 27.7 / 43.8 / 61.5 | 24.3 / 38.9 / 49.9 | 12.4 / 22.0 / 36.5 |
| C\* | 2.8 / 6.7 / 11.6 | 3.9 / 13.5 / 24.1 | 2.1 / 14.9 / 28.2 | 2.6 / 6.0 / 11.5 |
| nbr_de | 1.53 / 2.48 / 4.99 | 1.74 / 3.01 / 4.32 | 1.82 / 3.15 / 5.59 | 1.19 / 2.24 / 3.53 |
| IoU with the target volume | – | 0.40 / 0.68 / 0.83 | 0.56 / 0.65 / 0.87 | 0.62 / 0.77 / 0.86 |

How to read it:

- **Geometry.** On smooth photos TRELLIS behaves like TRELLIS on real photos (stair45 0.22,
  roughness 7.7°), and so do the synthetic inputs. On the paired builds the gap is small:
  stair45 +0.03 and roughness +0.3° (synthetic minus real, `real_trellis_smooth/index.json`).
  Blocky photos give stair-stepped meshes (stair45 0.88, roughness 18°), as intended for a
  stress test.
- **Colour.** TRELLIS bakes both photo sets **bright** (L\* 39–44), because the photos are
  renders in flat thumbnail colours, not photographs: about 2× as bright as TRELLIS on real
  photos. The synthetic inputs follow real photos (L\* 22). So a model trained on pairs-2
  sees darker colours than these eval inputs. Score colour or material on the eval sets with
  that in mind, or normalise lightness per input, the same at inference.

### Real-photo replica set (optional): pending

Candidate builds replicating real landmarks, and free-licence Wikimedia Commons photos found
for them. Nothing is downloaded yet; the downloads wait for approval.

| build | split | Commons file | licence, author |
| --- | --- | --- | --- |
| Carson Mansion | **test** | `2014.10.12.094700 Carson Mansion Eureka California.jpg` | CC0, Hermann Luyken |
| Zhytomyr bus station | **test** | `Автовокзал Житомир.jpg` | CC BY 3.0, Leonid Andronov |
| Coleraine Methodist Church | **val** | `Coleraine Methodist church (geograph 2693191).jpg` | CC BY-SA 2.0, Albert Bridge (same church to be confirmed) |
| Woolworth Building | train | `Exterior Woolworth Building 2015.jpg` | CC BY-SA 4.0, Steven Lek |
| 122 Leadenhall Street | train | `Leadenhall Building (aka the Cheesegrater) (28571108492).jpg` | CC BY 2.0, Paul Trafford |
| Swiss Federal Palace | train | `Federal Palace of Switzerland Bundeshaus im Bern by Robbie Conceptuel.png` | CC0 |
| West Virginia State Capitol | train | `The West Virginia State Capitol building - Image03.jpg` | CC BY-SA 4.0, Daniel G Rego |
| Alabama State Capitol | train | `Alabama state capitol, Montgomery.jpg` | CC BY 2.0, Jim Bowen |
| Fairmont Royal York | train | `Fairmont Royal York Hotel, Toronto (27823283571).jpg` | CC BY 2.0, shankar s. |

- Replicas in train are qualitative checks only and must stay out of test metrics.
- The Unité d'Habitation (Rezé) still needs a clean photo.
- The two parish churches (St Elizabeth, Saint Saviour) could not be identified.
- Photos stay local under `pairs/real_photo_replicas/` (gitignored, never redistributed). The
  app's background removal (rembg) runs before TRELLIS.

## Results (full run)

`python -m schemset.pairs --dataset data/real-schems-v1 --workers 6`, settings `pairs-2` /
`meshify-3`, σ floor 0.42 block:

- **Volume:** 317 builds, **1,268 pairs generated, 0 rejected, 0 builds failed**. Train
  1,012, val 128, test 128.
- **Geometry is identical to pairs-1** (same seeds). The numbers below hold for both.
- **Block-level IoU with the target volume:** p5 0.54, p10 0.62, p50 0.77, p90 0.86, p95
  0.88.
  - Coverage of the target volume: p5 0.92, p50 0.98.
  - Excess (input outside the target volume): p10 / p50 / p90 0.13 / 0.21 / 0.36.
- **Flagged `low_iou` (kept):** 26 pairs in 9 builds.
  - The lowest are `pmc-colonial-corivian` (0.32–0.36), the 9-storey Yaroslavl building
    (0.32–0.39) and `pmc-pig-and-cow-barn` (0.38–0.43).
  - The others are `pmc-house-32`, `pmc-modern-school`, `pmc-seven-eleven`,
    `pmc-small-modern-hotel-near-marina-schematic`, `pmc-victorian-house-4275811` and
    `pmc-warehouse-row`.
  - All are blobs much bigger than the build (excess), with coverage ≥ 0.80 and small frame
    residuals.
- **Frame:**
  - Residual against the placed reference: p50 0.0, p90 0.37, p99 0.80, max 1.16 cells.
    Every pair is within its tolerance (1–5.4 cells).
  - Input against the clean reference at 2×: IoU p10 / p50 / p90 0.83 / 0.92 / 0.95.
  - Per-build frame check: all 317 pass.
- **Meshify:**
  - 192k stair corner shapes and 143k pane / fence / wall connections derived for legacy
    formats.
  - 52k ground-layer blocks closed as slabs, in 163 builds.
  - 38.5k voxels of decor and decor-enclosed air opened to the outside, in 281 builds.
- **Disk:** 279.4 MB of pair npz (1,268 files). All of `pairs/` is 0.52 GB: the eval sets
  are 200 MB (TRELLIS GLBs and inputs, with 43 MB of trials) and the gallery 45 MB.
- **Time:** pairs-2 regeneration 27.6 min wall on 6 workers, the 3 largest builds first on 3
  workers: 126 s. The eval sets took about 1 GPU-minute per TRELLIS mesh, and alignment
  10 s–2 min per build.
- **Tests:** `tests/test_schemset_pairs.py`, 31 tests:
  - meshify geometry and the outside-only rule, decor;
  - the 2× frame and the validator;
  - determinism;
  - the colour model (ranges, target band, dark glass);
  - the eval alignment (8-cell shift, end to end).

## Domain gaps and caveats

- **Colour.**
  - Thumbnail colours are one flat colour per block, with no texture, so real texture,
    weathering and dirt are missing.
  - Since pairs-2 the brightness, saturation and glass match TRELLIS on real photos ([Colour
    model](#colour-model-pairs-2)).
  - The real-TRELLIS eval photos are bright renders, so their TRELLIS inputs are about 2×
    brighter than both.
- **Blobs against open interiors.** The voxeliser fills what the surface encloses, as it
  will for a TRELLIS mesh. Where D22 supervises air inside big openings (barns, halls up to
  `COVER_DEPTH`) or between sprawling wings, the input is solid and the target is air: the
  9 `low_iou` builds.
- **Surface noise.** After the smoothing floor, small builds' surfaces are smoother at the
  1-block scale than TRELLIS's (8.1° against 10.0° at ~36 blocks), and large builds'
  rougher (7.1° against 4.9° at ~87).
- **The eval photos are renders**, not photographs. Their placement uses the target
  (height, footprint, yaw, shift), so their IoU is optimistic. One TRELLIS seed per build.
- **Decor.** Torches, signs, plants, chains, rods, ladders and cobwebs are skipped; the
  outside flows through them. Unknown non-vocabulary blocks are cubes and are listed per
  build (`mesh.default_cube` in `records/`).
