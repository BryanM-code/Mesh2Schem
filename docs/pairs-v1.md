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

How one input is made (`schemset.pairs`, version `pairs-3`, on normalize-5 targets; `pairs-1`
and `pairs-2` had older colour models, see [Colour model](#colour-model-pairs-3)):

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
     [Colour model](#colour-model-pairs-3)).
2. **Variant.** A yaw of ±`YAW_DEG` = 5° about the build's vertical centre line, and an
   offset of ±`OFFSET_BLOCKS` = 0.5 block on x, y and z. Both are uniform and applied to the
   clean surface. This is the misregistration a real placement leaves. The target is never
   moved.
3. **Degrade** (`synth.degrade.degrade`; geometric ranges from
   `synth.config.PipelineConfig.degrade`, colour ranges from `pairs.COLOUR`; sampled by
   `sample_params`):
   - a resolution bottleneck of 128, 160 or 192 cells across the build's longest axis;
   - blur, surface noise, holes and floaters;
   - baked colour: a directional sun, hue drift, muted chroma and noise
     ([Colour model](#colour-model-pairs-3));
   - a hallucinated back side, dark as TRELLIS bakes the sides it does not see, and dark roofs;
   - what TRELLIS does to colour (`pairs.ARTEFACTS`): its own exposure (a mean L\* drawn per
     variant, L\* scaled to it, chroma kept), blotches, smears, regional hue drift. They draw
     from their own random stream, so the geometry does not depend on them.
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
6. **Metrics are reported twice: with and without the `low_iou` pairs.**
   - Pairs flagged `low_iou` (block IoU under `IOU_FLAG` = 0.45: the degraded blob is much
     bigger than the build) stay in training.
   - `pairs.metric_subsets(index["pairs"])` gives, per split, `all`, `without_low_iou` and
     `low_iou` as pair keys `"<id>/v<k>"`; `index.json` → `summary.metric_subsets` holds the
     counts and the flagged keys. Every Phase 5 metric is reported on `all` and on
     `without_low_iou`, so it shows whether the flagged pairs help or hurt.
   - All flagged pairs are in train (pairs-3: 31 pairs in 11 builds, train 1,012 = 981 + 31;
     pairs-2 had 26 in 9; val and test have none), so val and test numbers are the same either
     way unless that changes. Training metrics (loss, train IoU) are where the two differ.

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

Detail (`schemset.detail`, since pairs-3; the same function scores the real-TRELLIS and
real-photo inputs). The reference is the build's clean surface, glass in `REFERENCE_GLASS`,
voxelised exactly like the input in the input's own frame (a synthetic input's yaw and
offset; a TRELLIS input's target frame):

- `detail_geo` (feature recall): the reference's small features are the cells where it
  differs from its own open-close smoothing with a 3-cell cube (protrusions and recesses
  thinner than 1.5 blocks: window reveals, trim, cornices, railings, eaves). This is the
  share of them on which the input sides with the reference rather than with the smoothed
  reference. 1 means every feature is kept; 0 means all were smoothed away.
- `detail_spurious`: near the reference surface (2 cells) but off those features, the share
  of cells where the input differs from the reference (noise, bumps, blobs, missing parts).
- `detail_colour`: correlation of band-passed lightness (L*, difference of Gaussians 1 and 3
  cells, taken on the surface only) on the reference surface. Windows, trim and banding where
  the reference has them; exposure and white balance do not change it.
- Each is the best over input shifts of up to 1 cell along one axis, so leftover alignment is
  not counted as lost detail. Large grids are processed in 128-cell tiles.
- Checks (`tests/test_schemset_pairs.py`, `DetailMetric`): the reference scores 1 / 0 / 1;
  more blur keeps less; a 1-cell shift costs nothing; exposure does not change the colour
  score; tiles give the same answer as the whole grid.
- `cast_dL`, `cast_da`, `cast_db` (`detail.colour_cast`): the input's mean L*, a*, b* minus
  the reference's, over each one's surface. It mixes darkness with hue (dark colours lose
  chroma), so a warm or cool light shows only as a difference between inputs of the same
  build.

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

## Colour model (pairs-3)

pairs-1 used `synth.degrade`'s colour ranges, which were tuned for already shaded Objaverse
textures. On flat, unshaded thumbnail colours they made inputs about twice as bright and
1.5× as colourful as TRELLIS on real photos, and glass was light blue where real windows read
dark. pairs-2 overrode the colour ranges for the pairs only and matched the reference's L\*
and C\* distribution by darkening every input evenly. pairs-3 keeps that distribution but
gives it TRELLIS's structure: on a real photo TRELLIS bakes the side it sees bright and the
sides it does not see, and the roofs, near-black (L\* about 41 lit, 13 unseen, 21 roofs, from
the [facing analysis](#choosing-the-light-trials---exposures---lights)), with lightness
varying from block to block. And it sets the exposure the way TRELLIS does, from the output,
not from the build. The parked synth pipeline keeps its defaults, and the geometric ranges
are untouched.

Per variant, all drawn from its seed (exposure and the texture artefacts from a stream of
their own, so geometry and glass are the same as in pairs-2):

| | pairs-1 | pairs-2 | pairs-3 (`pairs.COLOUR`, `pairs.ARTEFACTS`) |
| --- | --- | --- | --- |
| exposure | sRGB gain, uniform 0.90–1.05 | sRGB gain, log-uniform 0.37–0.95 | **the mean L\* it ends at** (`target_lum`): p10 12, p50 22, p90 38 (8–48); L\* scaled to it, chroma kept |
| sun (`ambient` = the floor of the shading) | 0.85–0.98 (nearly flat) | 0.50–0.85 (directional) | 0.50–0.85 |
| unseen side (`back_darken`) | 0.85–1.0 | 0.85–1.0 | **0.20–0.55** |
| roofs (`roof_darken`, × up-facing share) | – | – | **0.30–0.55 darker** |
| blotches (`blotch`: sd of a lightness factor, 1–2.5 blocks across) | – | – | **0.08–0.20** |
| smears (`warp_units`: colours read this far off, wavelength 6–16 blocks) | – | – | **half the variants, 0.5–2 blocks** |
| regional hue drift (`tint_ab`: a\*, b\* sd, 8–20 blocks across) | – | – | **0–3** |
| chroma scale | 0.85–1.10 | 0.95–1.25, compressed towards C\* 8 (`chroma_gamma` 0.75) | **1.0–1.25, `chroma_gamma` 0.8** |
| hue drift (sd), colour noise (RGB sd) | 3°, 2–5 | 4°, 3–7 | 4°, **2.5–6** |
| white-balance cast (`cast_sd`) | – | – | 0 (off: TRELLIS did not carry the sunset photo's warmth through) |
| glass (`pairs.GLASS`, CIELAB before exposure) | thumbnail light blue | L\* 20–38, a\* −3…1, b\* −10…−3, per-window jitter 2–6, 0–25% of windows catch the sky (+8–20 L\*) | the same |

The new `synth.degrade` parameters (`blotch`, `warp_units`, `tint_ab`, `cast_a/b`,
`roof_darken`, `target_lum`) are optional and 0 = off; `chroma_gamma` (pairs-2) is
C' = chroma · 8 · (C / 8)^gamma. `tone_gain` is still drawn (always 1) so the draws after it
do not move.

How pairs-3 was fitted, in three steps.

**1. The structure** (30 pilot builds, `pairs.pilot_ids`, variant 1, every setting on the same
geometry). Per setting, the inputs' L\*, C\* and nbr ΔE, the colour detail, and the degraded
mesh's facing profile (mean L\* of the brightest and darkest side, the gap between them, and
roofs). p10 / p50 / p90 and medians:

| Setting | L\* | C\* | nbr ΔE | colour detail | lit / dark / gap / roof |
| --- | --- | --- | --- | ---: | --- |
| TRELLIS on real photos (reference) | 11.0 / 21.9 / 33.7 | 2.8 / 6.7 / 11.6 | 1.53 / 2.48 / 4.99 | 0.05–0.16 | 40.7 / 12.9 / 25.7 / 21.0 |
| pairs-2 | 13.7 / 19.0 / 29.3 | 2.8 / 6.1 / 10.4 | 1.2 / 2.0 / 3.1 | 0.40 | 24.3 / 16.1 / 7.0 / 21.7 |
| exposure 0.7–1.4, unseen side 0.1–0.45 | 17.9 / 22.7 / 35.4 | 2.9 / 6.3 / 11.3 | 1.6 / 2.6 / 4.5 | 0.36 | 37.1 / 9.3 / 27.9 / 31.5 |
| + dark roofs, blotches, smears, tint; exposure 0.5–1.4 | 11.6 / 16.8 / 28.0 | 4.8 / 6.8 / 10.5 | 1.7 / 2.5 / 3.7 | 0.30 | 31.8 / 7.7 / 21.4 / 19.0 |
| the same, exposure 0.7–1.6, less tint | 16.8 / 23.1 / 36.1 | 3.7 / 7.0 / 11.8 | 2.1 / 2.9 / 4.6 | 0.31 | 38.8 / 12.2 / 26.1 / 24.0 |
| the same, exposure 0.45–2.0 | 11.8 / 19.5 / 33.9 | 3.2 / 6.4 / 10.9 | 1.8 / 3.0 / 4.5 | 0.30 | 34.2 / 9.8 / 22.9 / 22.1 |
| exposure 0.5–2.3 | 13.4 / 22.3 / 38.6 | 3.4 / 6.9 / 11.8 | 2.0 / 3.2 / 4.9 | 0.30 | 38.4 / 11.4 / 24.7 / 25.2 |

Darkening the unseen side alone (row 3) gives TRELLIS's contrast but lights the roofs; dark
roofs and a wider exposure range bring the tail and the roofs in.

**2. The first full run failed.** With the last row (an sRGB gain, log-uniform 0.5–2.3), all
1,268 pairs came out too bright and too spread: L\* 12.1 / 26.7 / 50.4 with 57% inside the
reference's p10–p90 (70% required), nbr ΔE 1.94 / 3.63 / 6.37. One variant of 30 builds had
not shown it. Refitting the gain range from the full run (per build, L\* against ln gain)
found no range that passes: at best about half the inputs land inside the band. A gain
multiplies the build's own brightness, and the builds' brightness alone is as spread as
TRELLIS's output; TRELLIS sets its own exposure, whatever the photo's. So pairs-3 sets the
exposure from the output: each variant draws the mean L\* it ends at from the reference's
distribution, and degrade scales L\* to it.

**3. Exposure and chroma** (40 pilot builds × all 4 variants = 160 inputs, the full set's seeds
for those builds):

| Setting | L\* (share inside) | C\* (share inside) | nbr ΔE | lit / dark / roof |
| --- | --- | --- | --- | --- |
| TRELLIS on real photos (reference) | 11.0 / 21.9 / 33.7 | 2.8 / 6.7 / 11.6 | 1.53 / 2.48 / 4.99 | 40.7 / 12.9 / 21.0 |
| sRGB gain to the drawn mean L\* (p50 22, p90 37), blotches 0.10–0.25 | 11.8 / 22.0 / 35.5 (0.79) | 3.2 / 6.4 / 14.1 (0.78) | 1.65 / 2.73 / 5.69 | 36.3 / 11.9 / 21.7 |
| + p50 22.5, p90 38.5, `chroma_gamma` 0.7, noise 2.5–6, blotches 0.08–0.20 | 11.8 / 22.5 / 36.9 (0.78) | 3.1 / 6.3 / 14.1 (0.78) | 1.52 / 2.51 / 5.45 | 37.2 / 12.1 / 22.2 |
| the same, `chroma_gamma` 0.6 | 11.8 / 22.5 / 36.9 (0.78) | 3.2 / 6.0 / 13.1 (0.80) | 1.51 / 2.50 / 5.36 | 37.2 / 12.1 / 22.2 |
| **L\* scaled instead of a gain** (chroma kept), chroma 0.85–1.1, `chroma_gamma` 0.7 | 11.9 / 22.5 / 36.2 (0.78) | 3.7 / 6.0 / 9.9 (0.93) | 1.50 / 2.48 / 5.14 | 37.0 / 12.0 / 22.1 |
| L\* scaled, chroma 1.0–1.25, `chroma_gamma` 0.8 | 11.9 / 22.5 / 36.2 (0.78) | 3.7 / 7.0 / 12.0 (0.84) | 1.53 / 2.61 / 5.25 | 37.0 / 12.0 / 22.1 |
| the same, p90 40 (predicted from the row above) | 11.9 / 22.6 / 37.5 (0.76) | the same | the same | |
| the same, p90 40: **the full set** (1,268 pairs) | 11.65 / 22.11 / 39.27 (0.700) | 3.45 / 7.09 / 12.04 (0.83) | 1.51 / 3.01 / 5.26 | |
| **chosen: p50 22, p90 38** (predicted from the full set) | 11.65 / 21.64 / 37.33 (0.74) | the same | | |

The sample understated the full set's top (L\* p90 37.5 predicted, 39.3 measured): the upper
quantile of 160 draws is noisy. The full set at p90 40 passed with no margin (888 of 1,268
inside the band, 0.7003). On the full set the measured mean L\* follows the drawn target closely
(measured / target 0.94–1.03, correlation 0.995), so the quantiles were re-set from the full
set itself, and the pairs regenerated once more ([Results](#results-pairs-3)).

- **A gain couples chroma to lightness.** With an sRGB gain the brightest inputs were the most
  colourful (correlation of L\* and C\* across inputs 0.53; the reference's 0.14), so C\* p90
  stayed at 14 whatever `chroma_gamma` did. Scaling L\* alone removes it (−0.11), and
  `chroma` then sets colourfulness on its own.
- **Blotches and noise** set nbr ΔE: halving the blotches brought p50 and p90 down to the
  reference; the remaining p90 excess (5.25 against 4.99) is busy facades.
- A white-balance cast is available but off. The sunset barn's TRELLIS output was darker and
  greyer than its renders, its roof blue-grey rather than golden, so there was no cast to
  imitate.
- Colour detail stays above TRELLIS's ([Results](#results-pairs-3)). TRELLIS's colours follow
  the building far less than this; going further would make the inputs' colours nearly
  useless as a signal, and the model has to cope with TRELLIS's either way. Recorded as a
  remaining gap.

Full set, before and after: [Results](#results-pairs-3).

## TRELLIS artefacts and whether the synthetic inputs have them

From the gallery (raw TRELLIS next to its voxelisation, both photo sets) and the real
photos:

| Artefact | Seen on | Synthetic inputs |
| --- | --- | --- |
| The sides TRELLIS did not see, and roofs, baked near-black | every real photo (L\* 41 lit against 13 unseen, roofs 21) | **since pairs-3** (`back_darken` 0.2–0.55, `roof_darken`); pairs-2 darkened everything evenly |
| Its own exposure: the output's lightness does not follow the photo's, nor its colourfulness its lightness | the light trials (a gain of 0.45–1.0 on the photo gives L\* 36–45); the reference (L\*–C\* correlation 0.14) | **since pairs-3** (`target_lum`: a mean L\* drawn per variant, L\* scaled to it, chroma kept) |
| Lightness varying from block to block | all TRELLIS output (nbr ΔE p90 5.0) | **since pairs-3** (`blotch`); per-vertex noise averaged out |
| Colours that only loosely follow the building (colour detail 0.05–0.16) | all TRELLIS output | partly: smears (`warp_units`), regional hue drift (`tint_ab`) and the dark unseen side bring synthetic colour detail from about 0.4 (pairs-2) to 0.31 (pairs-3, all pairs; TRELLIS 0.14 on the smooth photos) |
| Regional hue changes: a purple roof, an orange side, a blue-grey barn roof | ab-6183, ab-6057, the barn | partly: `tint_ab` (small, it also lends grey walls colour) and the unseen side's wrong tint (`back_tint`) |
| Wavy streaks on fine stripes (Meridian Tower) | ab-13253 | not TRELLIS's: the voxeliser aliasing sharp stripes. Synthetic colours are sampled at the bottleneck, too soft to alias; `warp_units` smears them instead |
| Floaters | Carson Mansion, the chapel's graves, ab-11538 | yes (`floaters`, since pairs-1) |
| Blobbing, rounded corners, lost thin parts | everywhere | yes (bottleneck, blur, σ floor) |
| Clutter kept as building: vines, hay bales, the cutout's ground fringe as dark blobs at the base | the barn, the chapel | no. It needs geometry added at the base; left out to keep stair45, roughness and the frame check as they are |
| The unseen half completed by symmetry (dormers, a lean-to, the front gable copied to the back) | the barn | no: synthetic geometry is the true back. The unseen side's colours are partly mirrored (`back_blend`) |
| Thin flat parts thickened into slabs; open terraces closed | small train station, seaside restaurant | partly: blur and fill close thin gaps (the `low_iou` blobs); slabs under platforms are not added |
| Roof shapes simplified (a sawtooth hall became one arch) | empty factory | no, beyond what the blur does |
| Partial reconstructions (a crop, a partial cutout, a steep view) | Carson Mansion, the capitols, the Bundeshaus | no: a photo problem, not a mesh one. The app shows the cutout first |

## Calibration against TRELLIS

`python -m schemset.pairs_calibrate --dataset data/real-schems-v1 --before <old index.json>`
writes `pairs/calibration.json`: every statistic for the TRELLIS reference, the synthetic
inputs (now and before), the size-matched subset and the two real-TRELLIS sets, plus the
blockiness overlap and the colour acceptance.

**The reference is frozen.** The TRELLIS meshes the app made from photos lived only in
Real2Minecraft `outputs/trellis/`, where every new app run adds one. They are copied once
(`--freeze-reference ../Real2Minecraft/outputs/trellis`, 2026-10-15) to
`pairs/trellis_reference/meshes/`, with each file's sha256 in `manifest.json` (53 meshes,
134 MB, set digest `99d9f138…`). The calibration reads that copy by default and stops if any
mesh changed, went missing or was added (`check_reference`); the cached statistics
(`pairs/trellis_reference.json`) are keyed on the set digest too.

**The reference is narrow.** Its 50 building meshes come from 11 images, each run many
times: five photos of real buildings (`House_for_M2R` alone 21 meshes, `House3` 7, two towers
5 each, `House2` 2), three AI-generated images and three screenshots. Its medians are largely
those of one house photo. The real-photo set ([Real photos](#real-photos-end-to-end-schemsetreal_photo))
is an independent check on it: its 10 photos give TRELLIS L\* 17 (the reference 22) and nbr ΔE
2.5 (2.5).

pairs-3 (`pairs/calibration.json`), p10 / p50 / p90:

| statistic | TRELLIS on real photos (150) | synthetic pairs-3, all (1,268) | synthetic pairs-3, size-matched (940) | synthetic pairs-2, all |
| --- | --- | --- | --- | --- |
| stair45 | 0.02 / 0.25 / 0.70 | 0.04 / 0.20 / 0.49 | 0.04 / 0.21 / 0.50 | 0.04 / 0.20 / 0.49 |
| normal roughness | 4.2 / 8.1 / 13.6° | 4.0 / 7.5 / 10.4° | 5.3 / 7.8 / 10.5° | 4.0 / 7.5 / 10.3° |
| fill_gain | 2.69 / 5.45 / 10.92 | 3.06 / 5.14 / 9.83 | 3.17 / 5.09 / 8.41 | 3.10 / 5.17 / 10.09 |
| rough3 (blobby shell) | 0.018 / 0.029 / 0.074 | 0.026 / 0.053 / 0.100 | 0.026 / 0.055 / 0.097 | 0.026 / 0.053 / 0.100 |
| L\* | 11.0 / 21.9 / 33.7 | **11.7 / 21.6 / 37.4** | 11.5 / 21.6 / 37.4 | 12.4 / 22.0 / 36.5 |
| C\* | 2.8 / 6.7 / 11.6 | **3.5 / 7.1 / 12.0** | 3.4 / 7.0 / 12.0 | 2.6 / 6.0 / 11.5 |
| nbr_de | 1.53 / 2.48 / 4.99 | **1.49 / 2.97 / 5.17** | 1.89 / 3.11 / 5.07 | 1.19 / 2.24 / 3.53 |
| axis_area | 0.13 / 0.46 / 0.56 | 0.10 / 0.25 / 0.48 | 0.11 / 0.25 / 0.47 | 0.10 / 0.26 / 0.48 |

Colour acceptance (`colour_acceptance`): **accepted**. L\* median inside the reference's
p10–p90, 73.5% of inputs inside it (≥ 70%); C\* median inside, 82.8% inside; nbr ΔE median
inside; `COLOUR_TARGET` met (L\* p10 11.7 in 10–14, p50 21.6 in 20–26, p90 37.4 in 36–45; C\*
p50 7.1 in 6–8, p90 12.0 ≤ 14).

How to read it:

- **Blockiness** matches: synthetic is not blockier on either metric, at any size
  ([guard](#domain-gap-guard-blockiness)). The geometry is the same as pairs-2's except where
  normalize-5 changed a build.
- **Colour** matches in lightness, colourfulness and their independence, and now in TRELLIS's
  structure (a lit side, a near-black unseen side and roofs) and block-to-block variation
  ([Colour model](#colour-model-pairs-3)). nbr ΔE's median is 0.5 above the reference's.
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
| **smooth** | `real_trellis_smooth/` | the build's outside, glass dark (L\* 30, cool), **blurred by σ = 0.6 block** and re-extracted, colours from the nearest point of the clean surface, **hard sun and deep shadow** (`PHOTO_LIGHT` 0.23 + 0.77 · sun, × `PHOTO_EXPOSURE` 0.66; until this round flat 0.72 + 0.28 · sun), plain background | **the main evaluation set** |
| **blocky** | `real_trellis/` | a render of the raw block build (thumbnail colours, `render_photo`'s default light) | **the stress test**: TRELLIS reproduces Minecraft's stair-stepped roofs, which real photos never show |

Both sets were rebuilt for normalize-5 (the 13 builds whose masks changed, Hotel Piran among
them, got new photos and meshes) and every build was placed again with the detail metric.
The previous versions are in `pairs/history/pairs-2/`.

### Choosing the light (`trials --exposures`, `--lights`)

TRELLIS on the smooth photos came out about twice as bright as TRELLIS on real photos (L\*
44 against 22). Darkening the photos does not fix that; TRELLIS undoes a plain gain until the
photo is nearly black. The same 5 trial builds, TRELLIS's output p10 / p50 / p90:

| Photo light (ambient, sun) × gain | L\* | stair45 p50 | roughness p50 | nbr ΔE | IoU p50 |
| --- | --- | ---: | ---: | --- | ---: |
| flat (0.72, 0.28) × 1.0 (pairs-2; the 25 builds) | 27.7 / 43.8 / 61.5 | 0.22 | 7.7° | 1.74 / 3.01 / 4.32 | 0.68 |
| flat × 0.65 | 36.9 / 45.1 / 48.1 | 0.28 | 7.7° | | 0.69 |
| flat × 0.55 | 29.3 / 41.5 / 44.3 | 0.26 | 6.9° | | 0.69 |
| flat × 0.45 | 28.3 / 36.4 / 42.5 | 0.21 | 7.6° | 2.20 / 2.77 / 3.74 | 0.71 |
| flat × 0.25 (photo nearly black) | 12.8 / 18.6 / 19.0 | 0.23 | 7.9° | 1.08 / 1.55 / 3.21 | 0.67 |
| sun (0.3, 0.7) × 0.8 | 22.3 / 31.8 / 33.3 | 0.21 | 8.3° | 2.28 / 3.20 / 5.53 | 0.69 |
| sun (0.2, 0.8) × 0.6 | 17.9 / 24.5 / 30.3 | 0.17 | 7.6° | 1.62 / 3.34 / 5.34 | 0.62 |
| TRELLIS on real photos (reference) | 11.0 / 21.9 / 33.7 | 0.25 | 8.1° | 1.53 / 2.48 / 4.99 | |

The 5 trial builds read brighter than the whole set. Three full runs (all 25 builds):

| Photo light × gain (25 builds) | L\* | C\* | nbr ΔE | stair45 p50 | roughness p50 | IoU p10 / p50 / p90 | failed |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| flat (0.72, 0.28) × 1.0 (pairs-2) | 27.7 / 43.8 / 61.5 | 3.9 / 13.5 / 24.1 | 1.74 / 3.01 / 4.32 | 0.22 | 7.7° | 0.40 / 0.68 / 0.83 | ab-11538 |
| sun (0.2, 0.8) × 0.6 | 7.0 / 17.7 / 29.4 | 1.6 / 6.0 / 11.0 | 0.97 / 1.80 / 3.07 | 0.17 | 7.4° | 0.47 / 0.66 / 0.85 | ab-11538 |
| sun (0.26, 0.74) × 0.72 | 10.1 / 28.5 / 35.9 | 1.4 / 7.6 / 13.9 | 1.30 / 2.37 / 3.61 | 0.18 | 7.3° | 0.47 / 0.67 / 0.85 | ab-11538, the empty factory (frame) |
| **sun (0.23, 0.77) × 0.66 (chosen)** | **7.4 / 23.7 / 33.2** | **1.6 / 6.0 / 12.6** | **1.13 / 2.20 / 3.45** | **0.16** | **7.2°** | **0.47 / 0.66 / 0.86** | ab-11538 |
| TRELLIS on real photos (reference) | 11.0 / 21.9 / 33.7 | 2.8 / 6.7 / 11.6 | 1.53 / 2.48 / 4.99 | 0.25 | 8.1° | | |

The chosen light puts the median at the reference's (23.7 against 21.9) and the p90 with it.
The darkest builds come out darker than the reference's p10 (7.4 against 11.0): TRELLIS bakes four
of them nearly black (L\* 4–9: ab-6183, ab-2586, the barn ab-5731 and the dormer bungalow), as it
did one real photo (122 Leadenhall Street, 3.6). nbr ΔE stays below the reference
(2.2 against 2.5): the smooth photos have flat colours, real photos have texture. Earlier
versions of every set are in `real_trellis_smooth/trials/` (`full_light_*`) and
`pairs/history/pairs-2/`.

Why: TRELLIS bakes the side it sees about as bright whatever the photo's exposure, and on
real photos it bakes the sides it does not see, and the roofs, nearly black. Mean L\* by which
way the surface faces (azimuth bins, TRELLIS's own frame):

| TRELLIS on | lit side | darkest side | gap p10 / p50 / p90 | roof | all |
| --- | ---: | ---: | --- | ---: | ---: |
| real photos (the 50 reference meshes) | 40.7 | 12.9 | 11.5 / 25.7 / 38.6 | 21.0 | 19.4 |
| real photos (the 10 replicas and the barn) | 28.6 | 8.4 | 2.9 / 14.5 / 24.7 | 16.0 | 14.8 |
| smooth photos, flat light (pairs-2) | 46.6 | 38.0 | 1.7 / 6.4 / 13.4 | 38.1 | 43.1 |
| blocky photos | 39.2 | 30.9 | 0.8 / 6.2 / 13.7 | 37.0 | 36.2 |

A photo with hard shadows makes TRELLIS bake more of the building dark, and the spread
(p90 30, nbr ΔE p90 5.3) is that of real photos. A nearly black flat photo reaches the same
median but everything is uniformly dark (p90 19, nbr ΔE 1.6). Geometry holds (stair45 0.17,
roughness 7.6°); IoU p50 dips from 0.69 to 0.62 on these 5 builds (one seed each).

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
| blocky set, normalize-5 (this round) | **0 / 25** | 0.11 / 0.31 / 0.44 | 0.51 / 0.65 / 0.86 |
| smooth set, flat light (pairs-2) | **0 / 25** | 0.12 / 0.37 / 0.48 | 0.40 / 0.68 / 0.83 |
| smooth set, sun (this round) | **0 / 25** | 0.20 / 0.34 / 0.47 | 0.47 / 0.66 / 0.86 |

Shifts applied this round, max axis: p50 3 (smooth) and 2 (blocky), p90 about 9.5, at most 17
(smooth) and 14 (blocky) cells. One build fails in both sets for a real reason, not
alignment: `ab-11538`, where TRELLIS covers only 32% (smooth) or 47% (blocky) of the building
(`input_misses_building`). Tests: an 8-cell shift is recovered exactly, and `align` end to
end validates a turned mesh.

`centre_shift` keeps the best shift it visits. On one trial run the empty factory's residual
flipped between two shifts a cell apart and the loop stopped on the worse one (frame
misaligned, residual 1.76 cells); it now stops on a repeated shift and returns the visited
shift with the smallest residual.

### Low IoU on the smooth set (under 0.5)

Three builds, the same three under each of the three sun lights (the 0.26 / 0.74 run also had
the empty factory at 0.497, its frame misaligned before the `centre_shift` fix). Under pairs-2's
flat light the cemetery chapel (0.36) and ab-6183 (0.44) were under too. None of the three is
an alignment problem: each residual is under half a cell, so the IoU curve is centred. Photo · raw TRELLIS · voxelised ·
target, drawn in one frame, sorted out what each is:

| Build | IoU | coverage / excess | Cause |
| --- | ---: | --- | --- |
| `ab-11538` | 0.30 | 0.32 / 0.20 | **TRELLIS.** A tall curved corner block on a terraced plaza becomes a squat dome. The build's floating awning comes out far above the body, and the height fit (TRELLIS's top to the target's) shrinks the body further. Fails `input_misses_building` in both sets |
| `pmc-small-train-station` | 0.41 | 0.92 / 0.58 | **The build, then TRELLIS.** Low, thin and open: a platform under a canopy on thin posts, 43% of its columns at most 2 blocks high, mean column 3.2 blocks. TRELLIS makes the platform a thick slab and closes the space under the canopy; the voxeliser fills it. Any blob of this build has a large excess |
| `pmc-seaside-restaurant` | 0.45 | 0.99 / 0.55 | **The smooth photo, then TRELLIS.** The open terrace of tables inside the curved building is half closed in the σ 0.6 photo already, and TRELLIS closes it into a shell, which the voxeliser fills. The whole build is there (coverage 0.99); the terrace's air is not |

Next up: ab-6183 (0.51) and the cemetery chapel (0.53, its graves are floaters).

### Raw TRELLIS next to its voxelisation (the gallery)

`pairs/gallery/index.html` shows, for each evaluation build, TRELLIS's mesh before
voxelisation next to the voxelised input, for both photo sets: target · synthetic v0 · v1 ·
raw (smooth) · input (smooth) · raw (blocky) · input (blocky) · the two photos. The raw mesh
is posed as `align` placed it and drawn in exactly the voxelised tile's frame
(`pairs_gallery._voxel_frame`: the voxel renderer's own content box, cube size and centre,
under the same light), so the two overlay pixel for pixel (`GalleryRawTile` test). Since
pairs-3 a "real photos" section comes first: real photo · raw TRELLIS · TRELLIS input ·
synthetic v1 · target, with the numbers and the photo's credit.

What the side-by-side showed:
- **Black walls in the inputs were not TRELLIS's.** Where a TRELLIS mesh is bigger than the
  target frame, the stored input is cut at the frame edge. The cut exposes solid that was
  inside in the voxelisation and has no colour (rgb 0), and the gallery's renderer drew it
  black. Every such voxel lies on the frame boundary. They carry occupancy without surface
  or colour in the input, like any window edge, and the colour statistics never read them
  (they use the voxelisation's own shell). The gallery now draws them light grey.
- **The wavy streaks on ab-13253 (Meridian Tower) come from voxelising.** The raw mesh has
  fine, straight vertical stripes. The voxeliser samples them on a grid turned a few degrees,
  and they alias into curved moiré bands. The app's voxeliser does the same at inference,
  so this input property is real, but it is not TRELLIS's.
- **TRELLIS's own artefacts:** blobbing (the smooth photos of small builds are already blobs,
  see the caveats), regional hue changes (a purple roof on ab-6183, an orange side on
  ab-6057, a blue-grey barn roof), simplified roofs (the empty factory's sawtooth became
  one arch), floaters (Carson Mansion, the cemetery chapel's graves), dark unseen sides, and
  a slab under thin platforms.

### TRELLIS statistics per set (p10 / p50 / p90)

| statistic | TRELLIS on real photos (reference, 150) | TRELLIS on the 10 real photos of builds | TRELLIS on smooth photos (25) | TRELLIS on blocky photos (25) | synthetic pairs-3 (1,268) |
| --- | --- | --- | --- | --- | --- |
| stair45 | 0.02 / 0.25 / 0.70 | 0.17 / 0.42 / 0.62 | 0.05 / **0.16** / 0.49 | 0.35 / **0.87** / 0.96 | 0.04 / 0.20 / 0.49 |
| normal roughness | 4.2 / 8.1 / 13.6° | 0.8 / 6.6 / 19.7° | 2.6 / **7.2** / 12.3° | 4.3 / **18.4** / 26.5° | 4.0 / 7.5 / 10.4° |
| fill_gain | 2.69 / 5.45 / 10.92 | – | 2.68 / 4.65 / 11.42 | 1.85 / 4.10 / 10.86 | 3.06 / 5.14 / 9.83 |
| rough3 | 0.018 / 0.029 / 0.074 | – | 0.019 / 0.060 / 0.145 | 0.015 / 0.034 / 0.088 | 0.026 / 0.053 / 0.100 |
| L\* | 11.0 / 21.9 / 33.7 | 7.3 / 17.4 / 32.2 | 7.4 / **23.7** / 33.2 | 26.1 / 39.3 / 51.2 | 11.7 / 21.6 / 37.4 |
| C\* | 2.8 / 6.7 / 11.6 | 2.4 / 3.2 / 5.9 | 1.6 / 6.0 / 12.6 | 2.4 / 13.9 / 27.5 | 3.5 / 7.1 / 12.0 |
| nbr_de | 1.53 / 2.48 / 4.99 | 0.92 / 2.49 / 4.24 | 1.13 / 2.20 / 3.45 | 1.76 / 3.20 / 5.80 | 1.49 / 2.97 / 5.17 |
| feature recall (`detail_geo`) | – | 0.18 / 0.33 / 0.39 | 0.23 / 0.39 / 0.60 | 0.22 / 0.37 / 0.57 | 0.33 / 0.57 / 0.83 |
| spurious | – | 0.28 / 0.45 / 0.54 | 0.26 / 0.34 / 0.40 | 0.23 / 0.34 / 0.44 | 0.08 / 0.13 / 0.25 |
| colour detail | – | −0.02 / 0.05 / 0.15 | 0.07 / 0.14 / 0.25 | 0.04 / 0.12 / 0.41 | 0.15 / 0.31 / 0.51 |
| IoU with the target volume | – | 0.17 / 0.58 / 0.77 | 0.47 / 0.66 / 0.86 | 0.51 / 0.65 / 0.86 | 0.61 / 0.77 / 0.86 |

Until this round the smooth set was lit flat: L\* 27.7 / 43.8 / 61.5, C\* 3.9 / 13.5 / 24.1, nbr ΔE
1.74 / 3.01 / 4.32, stair45 0.22, roughness 7.7°, IoU 0.40 / 0.68 / 0.83 (`pairs/history/pairs-2/`).

How to read it:

- **Geometry.** On smooth photos TRELLIS behaves like TRELLIS on real photos (stair45 0.16,
  roughness 7.2°), and so do the synthetic inputs (0.20, 7.5°). On the paired builds the gap
  is small: synthetic v1 minus smooth TRELLIS, median stair45 +0.03 and roughness +0.8°.
  Blocky photos give stair-stepped meshes (stair45 0.87, roughness 18°), as intended for a
  stress test. The real photos of builds are rougher and blockier than both, from clutter
  and partial or street reconstructions ([Real photos](#real-photos-end-to-end-schemsetreal_photo)).
- **Colour.** The smooth set now matches TRELLIS on real photos in lightness (L\* p50 23.7
  against 21.9) and colourfulness (C\* 6.0 against 6.7); its nbr ΔE is lower (2.2 against 2.5),
  the photos' colours being flat. The paired lightness gap is +1.6 L\* (synthetic v1 minus
  smooth TRELLIS). The blocky set stays bright (L\* 39), so score colour or material on it with
  that in mind. The real photos of builds are darker and greyer still (L\* 17, C\* 3.2).
- **Detail.** TRELLIS keeps less of the build's detail than the synthetic inputs (feature
  recall 0.39 against 0.57; paired, synthetic v1 +0.14) and makes up more (spurious 0.34
  against 0.13); real photos the least and the most (0.33, 0.45). Synthetic colours follow
  the build more closely too (0.31 against 0.14; real photos 0.05).

## Real photos, end to end (`schemset.real_photo`)

The product pipeline on photos of the real buildings that some builds replicate: the app's
background removal (rembg `isnet-general-use`, a cutout kept only if it covers 5–97% of the
frame, otherwise the raw photo goes to TRELLIS, as in the app), TRELLIS (seed 1, the app's
settings), voxelisation at 2 cells per block, then `real_trellis.align` with a scale search,
because each build is its builder's interpretation. The scale search runs over ×0.6–1.5 of
the height fit and ×0.7–1.4 horizontally at 1 cell per block, in turns until neither moves,
then re-scores the neighbouring steps at 2 cells per block. The measurements are the same
as for the render sets (`real_trellis.measure`).

Photos: one supplied with the request (ab-5731, Barn by Keralis; the file in
`pairs/real_photo_tests/` was a screenshot of the described barn) and nine free-licence
Wikimedia Commons originals in `pairs/real_photo_replicas/<id>/` (URL, author, licence and
checksums in each `meta.json`; local only, never redistributed). Replicas in train are
qualitative checks only and stay out of test metrics (`in_test_metrics`).

| Photo | Split | To TRELLIS (mask) | IoU | Scale / aspect | stair45 | rough. | L\* | C\* | nbr ΔE | detail | colour |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Barn by Keralis | val | cutout (0.34) | 0.76 | ×1.00 / ×1.00 | 0.48 | 20° | 12.6 | 9.6 | 4.23 | 0.28 | 0.13 |
| Carson Mansion | test | raw photo (0.01) | 0.66 | ×0.60 / ×1.05 | 0.43 | 15° | 15.7 | 3.1 | 4.12 | 0.36 | 0.07 |
| Zhytomyr bus station | test | raw photo (0.02) | 0.18 | ×1.50 / ×1.40 | 0.70 | 20° | 17.1 | 2.5 | 4.10 | 0.08 | 0.18 |
| Coleraine Methodist Church | val | cutout (0.45) | 0.84 | ×1.00 / ×0.95 | 0.58 | 8° | 18.5 | 3.9 | 2.98 | 0.29 | 0.15 |
| Woolworth Building | train | cutout (0.74) | 0.53 | ×0.65 / ×1.05 | 0.61 | 5° | 47.1 | 5.5 | 4.28 | 0.38 | 0.02 |
| 122 Leadenhall Street | train | cutout (0.31) | 0.67 | ×0.80 / ×0.75 | 0.41 | 1° | 3.6 | 1.6 | 0.63 | 0.39 | 0.03 |
| Swiss Federal Palace | train | cutout (0.21) | 0.62 | ×0.95 / ×1.40 | 0.15 | 5° | 17.7 | 3.2 | 1.67 | 0.20 | 0.06 |
| West Virginia State Capitol | train | cutout (0.07) | 0.08 | ×1.50 / ×1.40 | 0.17 | 1° | 7.7 | 3.0 | 0.95 | 0.38 | −0.01 |
| Alabama State Capitol | train | cutout (0.31) | 0.33 | ×1.25 / ×1.40 | 0.21 | 9° | 23.1 | 3.1 | 2.01 | 0.41 | −0.08 |
| Fairmont Royal York | train | raw photo (0.02) | 0.31 | ×1.20 / ×1.00 | 0.31 | 1° | 30.6 | 4.1 | 1.77 | 0.26 | 0.03 |

What went in decides most of it:
- **Clean cutouts of whole buildings work.** Coleraine (0.84) and the barn (0.76) fit at
  the height scale.
- **The cutout fails on 3 of 9 Commons photos** (mask under 5%: a car, a sign or a lamp is
  segmented instead). The raw photo then goes in, and TRELLIS reconstructs the street:
  Zhytomyr's cars and trees (0.18), Royal York's neighbours (0.31). The app shows the
  cutout, so a user would see this and crop.
- **Partial cutouts or views** give partial meshes. The West Virginia cutout kept only the
  dome (0.08); the Bundeshaus and Alabama cutouts kept the central block of wide palaces,
  and their aspect search hits ×1.4. Carson Mansion is a close-up of the upper floors (scale
  ×0.6), Woolworth a steep worm's-eye view.
- A scale at a bound (×0.6, ×1.5, aspect ×1.4) means the mesh is a different extent of the
  building, not a different scale: the number is meaningless there.

**(a) Can TRELLIS handle a real photo of the barn?** Mostly.
- It gets the massing (the long barn), a clear gambrel profile at both gables, the dormers,
  the rusty lean-to roof, the double door and the hay loft.
- It completes the unseen half by symmetry: four dormers on each long side (the photo shows
  three), a second lean-to along the back, and the front gable copied onto the far end.
- It keeps the clutter as building: vines and flowers wrap the corners, hay bales and the
  cutout's ground fringe become dark blobs skirting the base, and the lanterns become yellow
  spots.
- The golden-hour roof comes out blue-grey, the walls brown, and the sides it did not see
  near-black.
- IoU 0.76 at the height scale (yaw 265°, residual under half a cell). The surface is rough
  (20° against 8° for the renders) because the clutter and shingles become bumps.

**(b) Does synthetic v1 look like it?** Closer than before, not the same.
- **pairs-2** was a uniformly dark, smooth, faithful copy of the build: L\* 11.8 (the real-photo
  output 12.6), but no lit and unseen side, little block-to-block variation (nbr ΔE 2.3
  against 4.2) and colours that follow the build closely (colour detail 0.63 against 0.13).
- **pairs-3** has TRELLIS's structure: a lit side, a near-black unseen side and roofs, blotches
  and regional hue drift. On this build its v1 is darker than the real-photo output (L\* 8.5
  against 12.6) and a little more colourful (C\* 11.5 against 9.6); its colours follow the build
  less (colour detail 0.44, from 0.63; TRELLIS 0.13), and block-to-block variation is still
  lower (nbr ΔE 2.2 against 4.2: the clutter and shingles make most of TRELLIS's).
- What it still lacks is TRELLIS's geometry habits: the clutter kept as building, the
  unseen half completed by symmetry, the roughness (synthetic 8.6° against 20°) and the
  spurious parts (0.09 against 0.28).
- The render sets of the same build: smooth TRELLIS is now dark (L\* 6.6) and a smooth blob
  (the photo of a small build is already one); blocky TRELLIS keeps the block steps
  (stair45 0.65) and stays bright (L\* 28).

**The sunset.** On the barn, TRELLIS's colours do not carry the warm light the way the photo
does. The output is darker and less colourful than the renders of the same build (C\* 9.6
against 40 and 26), and its roof turns blue-grey rather than golden. The real-photo colour
statistics are therefore not "warm" so much as dark, with the hue drifting by region. The
lighting skews L\* and C\* of this one build far more than its hue: treat its colour metrics as
one sample of TRELLIS's darkening, not as a cast to correct.

**Across the ten photos, against synthetic v1 of the same builds** (p50):

| | real-photo TRELLIS | synthetic v1, pairs-2 | synthetic v1, pairs-3 |
| --- | ---: | ---: | ---: |
| IoU | 0.58 | 0.80 | 0.80 |
| L\* | 17.4 | 31.7 | **17.5** |
| C\* | 3.2 | 6.5 | 7.4 |
| nbr ΔE | 2.49 | 1.30 | **1.79** |
| feature recall | 0.33 | 0.65 | 0.65 |
| spurious | 0.45 | 0.18 | 0.18 |
| colour detail | 0.05 | 0.23 | **0.15** |

Real-photo TRELLIS is darker, greyer, more varied from block to block, less faithful in colour
and noisier in shape than pairs-2 was. pairs-3 closes the lightness gap on these builds and
about 40% of the block-to-block and colour-detail gaps. It is more colourful than these ten (C\* 7.4
against 3.2): the reference it is fitted to is too (6.7), and ten photos, several of grey
stone or concrete, are a thin base for moving it. The geometric gaps are inflated by the
photos' problems and the builders' interpretations, and pairs-3 does not change geometry.

**Replace the render sets as the main Phase 5 eval, or sit alongside?** Alongside. With ten
photos, two of them in test, a build that is an interpretation of the building, and a third
of the inputs broken by the cutout or the framing, the real-photo numbers are too few and too
confounded to rank models. They are the end-to-end smoke test and the check on realism
(darkness, unseen sides, clutter, scale), reported separately and never pooled. The smooth
render set stays the main eval (exact correspondence, 25 val/test builds; since this round
it matches real-photo TRELLIS in blockiness and lightness), with the blocky set as the
stress test. Growing the real-photo set (more replicas in val and test, photos checked by
eye before they go in) would let it carry more weight.

Not found: a clean photo of the Unité d'Habitation (Rezé), and the two parish churches
(St Elizabeth, Saint Saviour) could not be identified. More real photos of builds would need
new downloads, which wait for approval.

## Results (pairs-3)

`python -m schemset.pairs --dataset data/real-schems-v1 --workers 5` (the 3 largest builds
first on 3 workers), settings `pairs-3` (`d11b2e631b41`) / `meshify-3`, σ floor 0.42 block, on
the normalize-5 targets:

- **Volume:** 317 builds, **1,268 pairs generated, 0 rejected, 0 builds failed**. Train
  1,012, val 128, test 128.
- **Geometry** is pairs-2's wherever normalize-5 left a build's masks alone (same seeds; the
  colour model draws from its own stream). The 153 builds whose masks changed in normalize-5
  have a different outside, so different inputs.
- **Colour model accepted** (`pairs/calibration.json`): every check passes, the narrowest
  being 73.5% of inputs inside the reference's L\* band (70% required).

All 1,268 pairs, p10 / p50 / p90, before and after:

| statistic | pairs-2 | pairs-3 | TRELLIS on real photos (reference, 150) |
| --- | --- | --- | --- |
| L\* | 12.4 / 22.0 / 36.5 | **11.7 / 21.6 / 37.4** | 11.0 / 21.9 / 33.7 |
| share of L\* in the reference's p10–p90 | 0.784 | **0.735** | |
| C\* | 2.65 / 6.04 / 11.51 | **3.45 / 7.09 / 12.04** | 2.79 / 6.69 / 11.60 |
| share of C\* in the reference's p10–p90 | 0.790 | **0.828** | |
| correlation of L\* and C\* across inputs | 0.19 | **0.02** | 0.14 |
| nbr ΔE | 1.19 / 2.24 / 3.53 | **1.49 / 2.97 / 5.17** | 1.53 / 2.48 / 4.99 |
| colour detail (`detail_colour`) | 0.16 / 0.38 / 0.63 (v1 of the 33 eval and real-photo builds; the metric is new) | **0.15 / 0.31 / 0.51** | 0.07 / 0.14 / 0.25 (TRELLIS on the smooth photos) |
| feature recall (`detail_geo`) | the same geometry | 0.33 / 0.57 / 0.83 | 0.23 / 0.39 / 0.60 (smooth photos) |
| spurious (`detail_spurious`) | the same geometry | 0.08 / 0.13 / 0.25 | 0.26 / 0.34 / 0.40 (smooth photos) |
| stair45 | 0.04 / 0.20 / 0.48 | 0.04 / 0.20 / 0.49 | 0.02 / 0.25 / 0.70 |
| normal roughness | 3.95 / 7.47 / 10.34° | 3.98 / 7.49 / 10.44° | 4.2 / 8.1 / 13.6° |
| fill_gain | 3.10 / 5.17 / 10.09 | 3.06 / 5.14 / 9.83 | 2.69 / 5.45 / 10.92 |
| rough3 | 0.026 / 0.053 / 0.100 | 0.026 / 0.053 / 0.100 | 0.018 / 0.029 / 0.074 |
| block IoU with the target volume | 0.62 / 0.77 / 0.86 | 0.61 / 0.77 / 0.86 | |

- **L\* and C\*** follow the reference; the top tail is brighter (p90 37.4 against 33.7), which
  `COLOUR_TARGET` asks for (p90 36–45). Lightness no longer drives colourfulness.
- **nbr ΔE**: the p90 gap is closed (5.17 against 4.99; pairs-2 3.53). The median is now above
  the reference's (2.97 against 2.48) while p10 matches: block-to-block variation is a little
  stronger than TRELLIS's in the middle of the range. One scale cannot match all three
  percentiles; lowering blotches or noise would drop p10 below the reference.
- **Geometry and blockiness are unchanged**: stair45 0.20 against 0.25, roughness 7.5°
  against 8.1°, not blockier at any size (size-matched: 0.16 / 0.21 / 0.27 at 36 / 57 / 87
  blocks against TRELLIS's 0.10 / 0.22 / 0.33).
- **Synthetic inputs keep more detail than TRELLIS** (feature recall 0.57 against 0.39, colour
  detail 0.31 against 0.14) and make far less up (spurious 0.13 against 0.34). See the same
  builds below.

Synthetic v1 against TRELLIS on the same 25 evaluation builds (p10 / p50 / p90):

| statistic | synthetic v1 (pairs-3) | TRELLIS, smooth photos | TRELLIS, blocky photos |
| --- | --- | --- | --- |
| L\* | 15.0 / 22.0 / 37.5 | 7.4 / 23.7 / 33.2 | 26.1 / 39.3 / 51.2 |
| C\* | 3.0 / 7.1 / 10.4 | 1.6 / 6.0 / 12.6 | 2.4 / 13.9 / 27.5 |
| nbr ΔE | 1.58 / 2.87 / 5.09 | 1.13 / 2.20 / 3.45 | 1.76 / 3.20 / 5.80 |
| colour detail | 0.17 / 0.28 / 0.46 | 0.07 / 0.14 / 0.25 | 0.04 / 0.12 / 0.41 |
| feature recall | 0.38 / 0.55 / 0.83 | 0.23 / 0.39 / 0.60 | 0.22 / 0.37 / 0.57 |
| spurious | 0.07 / 0.14 / 0.19 | 0.26 / 0.34 / 0.40 | 0.23 / 0.34 / 0.44 |
| stair45 | 0.08 / 0.26 / 0.53 | 0.05 / 0.16 / 0.49 | 0.35 / 0.87 / 0.96 |
| roughness | 3.6 / 8.2 / 10.2° | 2.6 / 7.2 / 12.3° | 4.4 / 18.4 / 26.5° |
| IoU | 0.67 / 0.79 / 0.84 | 0.47 / 0.66 / 0.86 | 0.51 / 0.65 / 0.86 |

- **Flagged `low_iou` (kept):** 31 pairs in 11 builds (pairs-2: 26 in 9), all in train.
  - The lowest are `pmc-colonial-corivian` (0.30–0.35), the 9-storey Yaroslavl building
    (0.32–0.39) and `pmc-pig-and-cow-barn` (0.34–0.39).
  - The others are `pmc-forest-fire-lookout-tower` and `pmc-modern-fuel-station-gas-station`
    (new: normalize-5 now supervises the open air under the lookout's platform and the
    canopy, which the blob fills), `pmc-house-32`, `pmc-modern-school`, `pmc-seven-eleven`,
    `pmc-small-modern-hotel-near-marina-schematic`, `pmc-victorian-house-4275811` and
    `pmc-warehouse-row`.
  - All are blobs much bigger than the build (excess), with coverage ≥ 0.78 and small frame
    residuals. Metrics with and without them: `summary.metric_subsets` (train 981 + 31).
- **Block-level IoU:** p5 0.52, p10 0.61, p50 0.77, p90 0.86, p95 0.87. Coverage p5 0.92, p50
  0.98; excess p10 / p50 / p90 0.13 / 0.22 / 0.38.
- **Frame:** residual against the placed reference p50 0.0, p90 0.37, p99 0.84, max 2.08
  cells (ab-11663 v3, a normalize-5 build, against its tolerance of 2.25). Every pair is
  within its tolerance. Per-build frame check: all 317 pass.
- **Meshify:** 192k stair corner shapes and 143k pane / fence / wall connections derived for
  legacy formats; 54k ground-layer blocks closed as slabs, in 163 builds; 39k voxels of
  decor and decor-enclosed air opened to the outside, in 283 builds.
- **Disk:** 298.9 MB of pair npz (1,268 files). All of `pairs/` is 1.20 GB: the eval sets with
  their trials, the frozen TRELLIS reference (134 MB), `history/pairs-2/`, the real photos and
  the gallery.
- **Time:** 35 min wall on 5 workers (the 3 largest builds first on 3 workers: 266 s). The
  eval sets took about 1 GPU-minute per TRELLIS mesh, and alignment 10 s–2 min per build.
- **Tests:** `tests/test_schemset_pairs.py`, 46 tests (15 new this round): the detail metric,
  the texture artefacts and the exposure (`target_lum`), the gallery's raw tile, the
  real-photo input, the frozen reference, the metric subsets, the scale search; and
  `tests/test_schemset_crop_rules.py`, 28 (2 new: the Hotel Piran regression and its
  counterpart, a deep low canopy that stays capped). The BDC suite: 459 tests, all pass.

## Domain gaps and caveats

- **Colour.**
  - Thumbnail colours are one flat colour per block, with no texture, so real texture,
    weathering and dirt are missing.
  - Since pairs-3 the lightness, colourfulness, glass, the lit / unseen / roof structure and
    block-to-block variation match TRELLIS on real photos ([Colour model](#colour-model-pairs-3)).
    nbr ΔE's median is 0.5 above the reference's.
  - Colours still follow the build more closely than TRELLIS's (colour detail 0.31 against
    0.14 on smooth photos, 0.05 on the real photos of builds).
  - The blocky eval set stays bright (L\* 39); the smooth set matches real photos since this
    round.
- **Geometry TRELLIS adds that the inputs lack**: clutter kept as building (vines, hay, the
  cutout's ground fringe), the unseen half completed by symmetry, slabs under thin
  platforms, simplified roofs. TRELLIS makes up far more than the synthetic inputs (spurious
  0.34 against 0.13) and keeps less (feature recall 0.39 against 0.57).
- **Blobs against open interiors.** The voxeliser fills what the surface encloses, as it
  will for a TRELLIS mesh. Where D22 supervises air inside big openings (barns, halls up to
  `COVER_DEPTH`), under canopies, or between sprawling wings, the input is solid and the
  target is air: the 11 `low_iou` builds, kept in training and reported both ways.
- **Surface noise.** After the smoothing floor, small builds' surfaces are smoother at the
  1-block scale than TRELLIS's (8.1° against 10.0° at ~36 blocks), and large builds'
  rougher (7.2° against 4.9° at ~87).
- **The eval photos are renders**, not photographs. Their placement uses the target
  (height, footprint, yaw, shift), so their IoU is optimistic. One TRELLIS seed per build.
  The smooth photo of a small build is already a blob (the barn, the seaside restaurant's
  terrace), so TRELLIS cannot recover what the photo lost.
- **The calibration reference is narrow**: 11 images, one house photo making 21 of its 50
  building meshes. The real photos of builds (10) are the independent check; they are
  darker and greyer than the reference (L\* 17, C\* 3.2).
- **Real photos of builds are few and confounded**: 10 photos, 3 of 9 Commons cutouts
  rejected (the street goes to TRELLIS), partial views, and each build is its builder's
  interpretation. The barn photo supplied is a screenshot and may be a render or AI image.
- **The voxeliser aliases fine stripes** into curved moiré bands on turned meshes (Meridian
  Tower); the app's voxeliser does the same at inference.
- **Decor.** Torches, signs, plants, chains, rods, ladders and cobwebs are skipped; the
  outside flows through them. Unknown non-vocabulary blocks are cubes and are listed per
  build (`mesh.default_cube` in `records/`).
