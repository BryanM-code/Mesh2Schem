# Phase 5: training and the baseline (started 2026-10-17)

The plan is [mesh-to-schem-plan.md, Phase 5](mesh-to-schem-plan.md#phase-5-revised-2026-10-17).
The code is Real2Minecraft `training/` (its README has the commands); the pair format and the
loader contract are [pairs-v1.md](pairs-v1.md). Runs live in `data/real-schems-v1/runs/`
(local only: weights trained on tier B / C builds are never redistributed, D14).

## How everything is scored

`training/score.py`, the same for the model and the baseline, over the **supervised voxels**
only (the visible shell and the exterior air around it; interiors and other IGNORE voxels
are never scored):

| Metric | What |
| --- | --- |
| IoU | solid (any shape but air) against the target's solid |
| F1 within 1 block | a solid voxel counts as matched if the other side has solid within one block |
| shape | exact shape label (stairs and their facing, slabs, panes, doors...) on voxels solid in both |
| material | exact material on voxels solid in both |
| ΔE | median CIELAB difference of the two materials' colours (`material_colors.json`): a near-miss costs less than a wrong colour |
| exact blocks | share of the target's solid voxels predicted exactly (shape and material) |

Builds are averaged (each build counts once). The selection score for checkpoints is the
mean over synthetic val builds of (IoU + exact blocks) / 2. Test is scored once, at the end.

## The baseline (D18)

`training/baseline.py` runs the app's own path on the real-TRELLIS evaluation sets and the
real photos: `normalize_mesh`, then building mode (B1–B5) or the sculpture voxelizer, at the
scale the model's input has (the longest side of the mesh as the evaluation placed it). Its
output has its own frame (the app normalises, turns and scales), so it is put into the
target's by `score.align`: the quarter turn and the integer shift that best overlap the two
solids. This favours the baseline slightly (a free best alignment; its blocks outside the
target frame, median 2.3% of them, go unscored). The synthetic pairs keep no mesh, so the
baseline has no synthetic score.

- 120 runs (25 + 25 builds × 2 modes, 10 photos × 2 modes), 37 min, 0 errors. Building mode
  refused 4 meshes ("does not read as a building"); they score as empty builds.
- Handedness checked: the mirror image of the app's output fits better by more than 5% in
  only 4.3% of builds (all near-symmetric), so the app's axes and the targets agree.

| Set (val / test) | Mode | IoU | F1 within 1 block | shape | material | ΔE | exact blocks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smooth photos (12 / 13) | building | 0.204 / 0.226 | 0.651 / 0.693 | 0.520 / 0.636 | 0.017 / 0.077 | 36.4 / 33.3 | 0.004 / 0.019 |
| | sculpture | 0.320 / 0.312 | 0.787 / 0.784 | 0.495 / 0.534 | 0.056 / 0.000 | 25.9 / 31.0 | 0.010 / 0.000 |
| blocky photos (12 / 13) | building | 0.186 / 0.197 | 0.619 / 0.676 | 0.586 / 0.648 | 0.054 / 0.028 | 31.5 / 32.3 | 0.007 / 0.007 |
| | sculpture | 0.307 / 0.336 | 0.772 / 0.815 | 0.516 / 0.539 | 0.020 / 0.033 | 24.4 / 28.3 | 0.010 / 0.014 |
| real photos (2 / 2) | building | 0.350 / 0.085 | 0.876 / 0.380 | 0.458 / 0.673 | 0 / 0 | 28.7 / 18.8 | 0 / 0 |
| | sculpture | 0.423 / 0.140 | 0.908 / 0.449 | 0.431 / 0.584 | 0 / 0.023 | 12.0 / 29.5 | 0 / 0.014 |

Both modes get the geometry roughly right within a block (F1 0.65–0.82) but almost never the
exact block: under 2% of the target's blocks.

## The model

A dense 3D residual U-Net (`training/model.py`): the 2× input packed into channels
(space-to-depth, so a block's sub-block detail reaches the first convolution), four levels
(32 / 64 / 128 / 256 channels), 7.2 M parameters, a shape head at every block and a
material head only at the voxels that need one. 64³ target windows (128³ input), batch 8,
bf16: 0.44 s a step on the RTX 4060 Ti. Inference tiles whole builds with
`mcvoxel.tiling.assemble`, as the app will.

- **Smoke test:** overfitting two small builds reproduces them exactly (IoU, shape, material
  and exact blocks all 1.0) after a few hundred steps. The whole pipeline learns.

### r1 (2026-10-17): the first run

20,000 steps, plain windows (the contract's 8 symmetries only), shape weights frequency^-0.5.

| Step | synthetic val IoU | synthetic val exact blocks | synthetic val material | smooth val IoU |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 0.517 | 0.076 | 0.142 | 0.302 |
| **4,000 (best)** | **0.527** | **0.080** | **0.145** | **0.312** |
| 8,000 | 0.479 | 0.052 | 0.098 | 0.317 |
| 12,000 | 0.506 | 0.045 | 0.091 | 0.311 |
| 20,000 | 0.509 | 0.049 | 0.099 | 0.307 |

The best checkpoint (step 4,000) against the baseline, same builds (val):

| Set | | model | building | sculpture |
| --- | --- | ---: | ---: | ---: |
| smooth photos (12) | IoU | 0.312 | 0.204 | 0.320 |
| | F1 within 1 block | **0.794** | 0.651 | 0.787 |
| | shape | 0.445 | **0.520** | 0.495 |
| | material | **0.079** | 0.017 | 0.056 |
| | ΔE | 28.3 | 36.4 | **25.9** |
| | exact blocks | **0.030** | 0.004 | 0.010 |
| blocky photos (12) | IoU | 0.301 | 0.186 | **0.307** |
| | exact blocks | **0.043** | 0.007 | 0.010 |
| real photos (2) | IoU | **0.444** | 0.349 | 0.423 |
| | material | **0.179** | 0 | 0 |
| synthetic (32) | IoU / exact blocks / material | 0.528 / 0.077 / 0.145 | – | – |

What r1 shows (the bottlenecks):

1. **It overfits after 4,000 steps.** The training losses keep falling (material
   cross-entropy 3.6 → 1.0) while val falls from step 4,000 on: 253 builds with four fixed
   degraded inputs each are memorised.
2. **It over-fills.** On smooth val it predicts 213k `full` voxels where the targets have
   111k (precision 0.30, recall 0.58), and twice the bottom slabs; even on synthetic input
   1.25 M against 0.80 M. The shape weights (a 277× spread from air to fence gates) make a
   missed block cost 5.6× a block placed in air.
3. **The real-TRELLIS gap is large.** IoU 0.53 on synthetic input against 0.31 on TRELLIS
   output of the same builds, and smooth val barely moves while synthetic val changes:
   what the model learns from synthetic inputs transfers only partly.
4. **Materials are the weakest part** (0.08–0.15 exact). The model sees only TRELLIS's dark
   voxel colours; building mode reads the photo's colours directly and gets some roofs right
   that the model does not (the restaurant's yellow thatch). A photo colour descriptor as an
   input (section 8's "global colour descriptor") is the obvious next lever.
5. Rare shapes (trapdoors, walls, panes, doors) are almost never predicted.

### r2 (2026-10-18): augmentation and softer shape weights

TRELLIS-like input augmentation on the fly (`training/data.py`, `AUGMENT`: colour gain and
noise, grey wash, spurious blobs, dents, one-cell closing, a one-cell shift), softer shape
weights (frequency^-0.25, capped at 10× the smallest: 0.9 for air, 2.1 for full, about 9 at
the cap), weight decay 0.02. Otherwise as r1. Best at step 6,000 (selection 0.291, r1 0.304).

| val | r1 best | r2 best | building | sculpture |
| --- | ---: | ---: | ---: | ---: |
| synthetic IoU | **0.528** | 0.522 | – | – |
| synthetic shape | 0.562 | **0.643** | – | – |
| synthetic exact blocks | **0.077** | 0.059 | – | – |
| smooth IoU | 0.312 | 0.283 | 0.204 | **0.320** |
| smooth F1 within 1 block | **0.794** | 0.767 | 0.651 | 0.787 |
| smooth shape | 0.445 | **0.550** | 0.520 | 0.495 |
| smooth material | **0.079** | 0.075 | 0.017 | 0.056 |
| smooth ΔE | 28.3 | 27.2 | 36.4 | **25.9** |
| smooth exact blocks | **0.030** | 0.020 | 0.004 | 0.010 |
| blocky IoU | 0.301 | 0.255 | 0.186 | **0.307** |
| photos IoU (2) | **0.444** | 0.328 | 0.349 | 0.423 |

Not better overall: the softer weights traded geometry and exact blocks for shape accuracy,
and rare shapes all but disappeared (walls 18 predicted against 43k on synthetic val, panes
7k against 56k). Material accuracy still peaks early (0.155 at 4,000 steps) and decays to
0.074: the augmentation does not stop the materials being memorised.

### Where the false blocks are

The over-filling is **not** mainly the shape weights. Almost every false solid voxel lies
where the input itself is solid: the voxelised mesh is solid over porches, covered terraces,
courtyards and the gaps between wings, which the targets supervise as open air (D22). The
model has to carve that away, and the input says little about where. `training.evaluate`
reports this as `input_excess` for every set (val, all variants):

| val | input-filled supervised air (× the true solid) | run | false solid (inside the input) | input-filled air carved away | missed solid |
| --- | --- | --- | --- | ---: | ---: |
| synthetic | 1.32 M (1.17×) | r1 | 807k (94%) | 43% | 211k |
| | | r2 | 530k (98%) | 60% | 374k |
| smooth (real TRELLIS) | 223k (**1.43×**) | r1 | 167k (98%) | 27% | 55k |
| | | r2 | 128k (99%) | 43% | 63k |
| blocky | 191k (1.23×) | r1 | 151k (97%) | 23% | 75k |
| | | r2 | 97k (98%) | 51% | 94k |

TRELLIS fills more open space than the synthetic inputs (1.43× the true solid against
1.17×), so a model that learned to carve on synthetic input leaves more behind on TRELLIS
output: a large part of the synthetic-to-TRELLIS gap (IoU 0.53 against 0.28–0.31). The
shape weights only move the trade-off: r2's softer weights carve more (43% against 27% on
smooth) but miss more real solid. A one-cell closing cannot imitate TRELLIS's filling: even
three iterations add 2% solid, against whole filled porches and courtyards.

### Bottlenecks after r2

1. **The input's filled open space** (above): 98–99% of the false blocks on real TRELLIS input lie
   inside the input's own solid; TRELLIS fills 1.43× the true solid volume with supervised air.
2. **Too few builds for materials**: material accuracy peaks at about 4,000 steps and decays
   in both runs; 253 builds with 313 materials are memorised, augmented or not.
3. **The synthetic-to-TRELLIS gap**, much of it 1.
4. **Rare shapes** (walls, panes, trapdoors, doors) trade off against geometry through the
   shape weights.
5. **Colours**: the model sees TRELLIS's dark voxel colours; sculpture mode's photo-guided
   colours are closer (ΔE 25.9 against 27–28).

The levers named then: real TRELLIS inputs for the train builds (r3), a photo colour
descriptor as a model input, more builds.

### r3 (2026-10-18): real TRELLIS output of the train builds

TRELLIS on the smooth photo of every train build (`real_trellis_train/`, see
[pairs-v1.md](pairs-v1.md#training-inputs-real_trellis_train-phase-5)): 244 valid inputs,
added to the cache as each build's fifth input. A sample takes the build's real TRELLIS
input with probability 0.5 (`--trellis-share 0.5`), otherwise one of its four synthetic
inputs. r1's settings otherwise (plain windows, shape weights frequency^-0.5); 12,000 steps
(r1 and r2 peaked by 6,000), validation every 1,000. `best_trellis.pt` keeps the checkpoint
with the best real-TRELLIS val score as well. 1.5 h on the 4060 Ti.

Val, same builds; "input" scores the input's own occupancy as the prediction:

| val | r1 best (4k) | r3 best (4k) | r3 best on TRELLIS val (3k)¹ | building | sculpture | input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smooth IoU | 0.312 | 0.317 | 0.325 | 0.204 | 0.320 | 0.320 |
| smooth F1 within 1 block | 0.794 | 0.797 | 0.799 | 0.651 | 0.787 | – |
| smooth shape | 0.445 | 0.491 | 0.485 | 0.520 | 0.495 | – |
| smooth material | 0.079 | 0.080 | 0.091 | 0.017 | 0.056 | – |
| smooth ΔE | 28.3 | 26.0 | 26.2 | 36.4 | 25.9 | – |
| smooth exact blocks | 0.030 | 0.018 | 0.038 | 0.004 | 0.010 | – |
| blocky IoU | 0.301 | 0.299 | 0.305 | 0.186 | 0.307 | 0.326 |
| blocky material | 0.114 | 0.099 | 0.094 | 0.054 | 0.020 | – |
| blocky exact blocks | 0.043 | 0.033 | 0.033 | 0.007 | 0.010 | – |
| photos IoU (2) | 0.444 | 0.470 | 0.439 | 0.349 | 0.423 | 0.448 |
| photos exact blocks (2) | 0.047 | 0.060 | 0.013 | 0 | 0 | – |
| synthetic IoU | 0.528 | 0.502 | 0.501 | – | – | 0.504² |
| synthetic exact blocks | 0.077 | 0.082 | 0.072 | – | – | – |

¹ Chosen on smooth val, so its smooth column is optimistic. ² Variant 0 only.

Paired bootstrap over the val builds (r3 minus r1, mean and 95% interval):

| val | IoU | shape | material | exact blocks |
| --- | --- | --- | --- | --- |
| smooth (12) | +0.005 [−0.015, +0.027] | +0.047 [+0.002, +0.088] | +0.001 [−0.045, +0.053] | −0.012 [−0.038, +0.008] |
| blocky (12) | −0.002 [−0.031, +0.023] | +0.032 [−0.054, +0.107] | −0.015 [−0.054, +0.024] | −0.010 [−0.027, +0.006] |
| synthetic (32) | −0.026 [−0.032, −0.020] | +0.024 [−0.000, +0.046] | +0.006 [−0.018, +0.029] | +0.005 [−0.011, +0.020] |

What r3 shows:

1. **Not better.** On real TRELLIS input r3 equals r1 within the noise; only shape accuracy
   rises (+0.05, barely clear of zero). Synthetic IoU drops 0.026: half the windows are no
   longer synthetic. The checkpoint chosen on smooth val is best on smooth val and no
   better than r1 on the blocky inputs of the same builds: selection noise.
2. **It learned what TRELLIS loses, not what it fills.** TRELLIS's input covers 71% of a
   train build's solid blocks (60% of its bottom layer), the synthetic inputs 89% at every
   height. r3 puts blocks back: it misses less of the true solid on smooth val (45k voxels
   against r1's 55k) but carves less of the input-filled air (20% against 27%), and its
   false solid outside the input rises from 3.7k to 13k voxels, 2.7k of them on the bottom
   layer: a ground plate under builds that have none (the barn).
3. **The model's geometry is its input's.** The input's own occupancy, taken as the
   prediction, scores IoU 0.320 on smooth val, 0.326 on blocky, 0.504 on synthetic: as much
   as every run and sculpture mode (0.28–0.33 on real TRELLIS input, 0.50–0.53 on
   synthetic). The synthetic-to-TRELLIS gap in IoU (0.53 against 0.31) is the inputs' own
   gap (0.50 against 0.32), not a failure to transfer.
4. **No operating point helps.** Scaling the blended air probability by k before the argmax:
   on smooth val, r1's IoU goes 0.316 / 0.312 / 0.305 / 0.300 / 0.284 for k = 0.5 / 1 / 1.5 /
   2 / 3, and r3's 0.303 / 0.317 / 0.316 / 0.302 / 0.251; blocky the same. Inside the input's
   solid the model's air probability barely separates filled open space from real walls:
   removing more removes true solid at the same rate. (On synthetic input there is a little
   signal: r1 0.534 at k = 1.5 against 0.527.)
5. **Twelve val builds cannot rank these runs.** The 95% intervals (±0.02 IoU, ±0.03 exact
   blocks, ±0.05 material) are wider than every difference between r1, r2 and r3 on real
   TRELLIS input.

Rare shapes go further: r3 predicts 3k pane voxels on smooth val (r1 22k, target 4.7k) and
gets no wall right; bottom slabs come up (recall 0.08 → 0.23).

### Bottlenecks after r3, in order

1. **The geometry is the input's.** Neither the model nor the rules beat the input's own
   occupancy, on synthetic or real TRELLIS input. Where TRELLIS fills a porch or a courtyard,
   or loses a wing, the voxels do not say so, and 253 builds do not teach the priors to
   guess it. This is what "over-filling" and "the synthetic-to-TRELLIS gap" were.
2. **Evaluation noise**: 12 real-TRELLIS val builds; every recent difference is inside it.
3. **Memorisation**: every run peaks between 1,000 and 6,000 steps; a fifth input per build
   (r3) does not move it. Materials and rare shapes suffer most.
4. **The synthetic inputs keep too much of the build** (89% of its solid against TRELLIS's
   71%; feature recall 0.58 against 0.40): they teach the model to trust the input more than
   TRELLIS deserves.
5. **Colours**: sculpture mode's photo-guided colours stay as close as the model's (ΔE 25.9
   against 26–28).

Next levers, cheapest first:

1. **Real TRELLIS for every val and test build** (smooth set: 32 + 31 builds instead of
   12 + 13): about 35 min of TRELLIS, 10 min of placement, 25 min of baseline. Intervals
   about 40% narrower; needed before the next comparison means anything.
2. **TRELLIS's losses in the synthetic inputs**: drop parts of the build (more at the base)
   until the synthetic inputs cover the build as TRELLIS does.
3. **The photo as a model input.** The photo shows what TRELLIS fills and loses (open
   porches, arcades, the ground plate) and the true colours. Every train build now has one
   (the smooth renders), with known cameras. First a global colour descriptor (materials;
   plan section 8), then the photo projected onto the grid (geometry; for a real photo the
   camera has to be estimated).
4. **More builds**, the only fix for memorisation.

## Photo-first pipeline, tests A and B (2026-10-18/19)

After r3 the question was whether a pipeline without a 3D generator would do better. Code:
Real2Minecraft `photofirst/` (its README has the commands), separate from the TRELLIS
pipeline. It reuses only the U-Net, the tiling and the scorer. Data:
`data/real-schems-v1/photofirst/`.

```
photo -> front end: MoGe-2 depth + the building mask -> the blocks the photo shows
      -> completion model (the U-Net, trained on what one viewpoint shows -> the build) -> blocks
```

### What one photo shows

The view of the evaluation photos (yaw 35, pitch 12) was ray-cast through every target
(`photofirst.visibility`, which also counts every block face with a clear line to the
camera).

- **One photo shows about 45% of a build's outer blocks** (median over the 63 val and test
  builds; p10–p90 39–54%), two photos from opposite corners 81%, four 92%.
- **A perfect copy of only the visible blocks scores IoU 0.43** on the smooth val builds,
  more than TRELLIS, the r1/r3 models or sculpture mode (0.31–0.32).
- **About half of TRELLIS's errors contradict the photo itself** (48% with the rays alone,
  before roofs seen at a grazing angle counted): false solid on air the camera sees, visible
  blocks it loses. The rest are in the part no photo shows.
- Mirroring the visible side is a weak guess for the rest (right about a third of the
  time); only 36% of footprints fill 90% of their rectangle.

### Test A: completing exactly what the photo shows

The completion model is training's U-Net, unchanged, on a new input: seen solid, seen air
(where the camera sees past) and the seen blocks' colours (material colours under random
exposure). Each train build is seen from 16 viewpoints (pitch 5–30). Run pf-a1: 12,000
steps, best at 7,000; unlike the TRELLIS runs it was still improving past 4,000 steps.

| val, the 12 real-TRELLIS builds, the photo TRELLIS got | photo-first, exact visible blocks | visible blocks alone | r1 | r3 | sculpture | building |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IoU | **0.522** | 0.432 | 0.312 | 0.317 | 0.320 | 0.204 |
| F1 within 1 block | **0.855** | 0.723 | 0.794 | 0.797 | 0.787 | 0.651 |
| shape | 0.461 | – | 0.445 | 0.491 | 0.495 | 0.520 |
| material | **0.177** | – | 0.079 | 0.080 | 0.056 | 0.017 |
| ΔE | **16.7** | – | 28.3 | 26.0 | 25.9 | 36.4 |
| exact blocks | **0.068** | – | 0.030 | 0.018 | 0.010 | 0.004 |

On all 32 val builds and from each corner the same: IoU 0.53 (the visible blocks alone
0.44–0.46), exact blocks 0.09, material 0.18–0.19. **The ceiling is far above the TRELLIS
pipeline.**

### Test B: what the front end recovers

The front end runs MoGe-2 (`Ruicheng/moge-2-vitl-normal`) on the photo, keeps the
building's pixels, and turns the points into the build's frame:
- the frame is given, or estimated from the point map's normals (pitch only, walls at
  right angles; up direction off by 1.5° median, 3.9° p90);
- the points are scaled to the target's height (as TRELLIS's placement is);
- a block is seen solid where points fall half a block behind the surface;
- the air in front of the points and along the rays through background pixels is seen air.

The result is placed in the target's frame by the best integer shift (and, for an
estimated frame, quarter turn). Scored against what that photo truly shows:

| 25 evaluation builds | recall, exact / within 1 | precision, exact / within 1 | solid on air the photo shows |
| --- | --- | --- | ---: |
| orthographic renders (the photos TRELLIS got), camera known, depth stretched | 0.44 / 0.80 | 0.45 / 0.81 | 3.4% |
| **perspective renders (`photofirst.render`, 50° lens), camera known** | 0.52 / **0.85** | 0.51 / **0.86** | **3.1%** |
| perspective renders, camera estimated from the photo | 0.38 / 0.68 | 0.52 / 0.85 | 2.3% |
| TRELLIS's input (its placement used the target: height, yaw, shift) | **0.74** / **0.88** | 0.38 / 0.76 | 10.6% |

- **On orthographic renders MoGe flattens the depth** (relief 0.76 of the truth, median):
  it reads depth from perspective. Perspective renders fix that; they also need no depth
  stretch (it lowers the scores).
- **The front end is much cleaner than TRELLIS** (3% against 11% solid on visibly empty
  air) and as complete within a block (0.85 against 0.88). But it places blocks exactly
  less often (0.52 against 0.74): of the truly seen blocks it has 39% exactly, 37% one
  block off, 23% further. Snapping walls and flat roofs to planes adds 0.02.
- **Estimating the camera costs 0.17 of recall within a block** (0.85 to 0.68).
- **Real photos** (the 10 with targets, all automatic): 0.51 of its blocks within a block of
  the target (TRELLIS 0.47), but it covers 24% of the target's solid (TRELLIS 0.57). Large
  landmarks (capitols, towers) fail on scale and coverage.

### End to end: not better yet

| val (12 builds) | IoU | F1 within 1 | material | exact blocks |
| --- | ---: | ---: | ---: | ---: |
| photo-first, perspective photo, camera known (pf-a1) | 0.258 | 0.695 | 0.042 | 0.009 |
| photo-first, perspective photo, camera estimated (pf-a1) | 0.253 | 0.677 | 0.046 | 0.012 |
| photo-first, the orthographic photo TRELLIS got (pf-a1) | 0.238 | 0.678 | 0.041 | 0.011 |
| photo-first, real photos (the barn and Coleraine Church, pf-a1) | 0.232 | 0.711 | 0.060 | 0.015 |
| TRELLIS pipeline, r1 | **0.312** | **0.794** | **0.079** | **0.030** |

The completion model, given the front end's blocks, falls from IoU 0.52 to 0.25.
Swapping one part of its input at a time for the truth (perspective, camera known, pf-a1;
before the background rays were added):

| input | IoU | precision | recall |
| --- | ---: | ---: | ---: |
| all from the front end | 0.176 | 0.24 | 0.57 |
| front-end blocks, true seen air | 0.318 | 0.44 | 0.58 |
| true blocks, front-end seen air | 0.300 | 0.37 | 0.70 |
| true blocks, true seen air, the photo's colours | 0.503 | 0.67 | 0.69 |
| all true | 0.497 | 0.64 | 0.72 |

1. **Seen air matters most.** The model reads the air the camera sees as "keep empty". The
   first front end marked only the air in front of the building's own pixels; rays through
   background pixels fixed most of it (0.176 to 0.258).
2. **Exact placement of the visible blocks is the binding limit.** With true seen air but
   the front end's blocks the model reaches 0.318; each block a front-end wall is off costs
   IoU twice (a false block, a missed one), and no completion can know where the wall
   really is. End to end it misses 47% of the truly seen solid.
3. **The photo's colours cost materials, not geometry** (0.503 IoU with them, material 0.058
   against 0.177).
4. A completion model trained on the front end's kind of errors (pf-a2: surface patches
   moved one or two blocks, patches lost, spurious blocks, 70% of the seen air, photo-like
   colours; validated on the front end's own val inputs every 1,000 steps) does not
   close the gap: IoU 0.25 at step 1,000, then 0.22–0.24 to step 6,000, where it was
   stopped. It became more forgiving within a block on true inputs (F1 within 1 block up
   to 0.885) but cannot recover exact positions the input does not hold.

### Verdict

The completion half works and has a far higher ceiling than the TRELLIS pipeline; the
front-end half is cleaner than TRELLIS but places surfaces less exactly, and end to end the
photo-first pipeline scores 0.25 against TRELLIS's 0.31. Not worth switching to yet; worth
keeping, in these directions:

1. **A hybrid**: TRELLIS's geometry, the photo's seen air carving away what TRELLIS fills
   (its fills on visibly empty air are 10.6%, the front end's seen air is wrong on solid
   1.6% of the time), and the completion model on top.
2. **Better visible geometry**: a stronger depth model (MoGe-3 is out), several photos (two
   opposite corners see 81% of the outer blocks).
3. **Training the completion model on real front-end outputs** (perspective renders of the
   train builds through the front end), as r3 did for TRELLIS.

### Hybrid, MoGe-3 and several photos (2026-10-19)

Three follow-ups to the verdict above, on the same 12 val builds. Each build was rendered
from its four corners (`photofirst.render --views 0 1 2 3`: yaw 35 + 90 k, the perspective
look). The cameras are known when photos are fused; real photos would need them estimated.

**MoGe-3** (`Ruicheng/moge-3-vitl`, August 2026; needs FlexGEMM on `triton-windows` 3.6,
pinned in `photofirst/requirements.txt`) is barely better than MoGe-2 here:

| front end, perspective photo | recall, exact / within 1 | precision, exact / within 1 | end to end IoU |
| --- | --- | --- | ---: |
| MoGe-2, camera known | 0.515 / 0.849 | 0.509 / 0.856 | 0.258 |
| MoGe-3, camera known | 0.517 / 0.861 | 0.530 / 0.884 | 0.261 |
| MoGe-2, camera estimated | 0.385 / 0.682 | 0.519 / 0.847 | 0.253 |
| MoGe-3, camera estimated | 0.395 / 0.689 | 0.515 / 0.855 | 0.254 |

**The hybrid** (`photofirst/hybrid.py`): TRELLIS's placed input (from the orthographic
photo it was given) minus every block the photos show as empty space (the front end's
seen air), through r1:

| seen air from | carved input alone | r1 on it | F1 within 1 (r1) |
| --- | ---: | ---: | ---: |
| none (TRELLIS as it is) | 0.343 | 0.333 | 0.841 |
| 1 photo | 0.380 | 0.358 | 0.854 |
| 2 photos, opposite corners | **0.441** | 0.396 | **0.889** |
| 4 photos | **0.441** | 0.400 | **0.889** |

The carved input alone is better than r1 on it: r1 learned TRELLIS's filled input and puts
some of the fill back. A 1-block margin (carving only air away from the photo's surfaces)
keeps more true solid, but loses a little IoU.

**Several photos, photo-first** (`photofirst/multiview.py`): each photo's front-end blocks in
the target's frame, fused. A plain union adds every photo's misplaced surfaces. Space carving
drops a block one photo saw solid where another saw clear through it, more than a block from
that photo's own surface. pf-a1 completes the result:

| photos | front end alone | photo-first end to end | F1 within 1 | true visible blocks alone | completion of the true blocks (ceiling) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.180 | 0.258 | 0.695 | 0.415 | 0.497 |
| 2 adjacent (union) | 0.264 | 0.318 | 0.783 | 0.622 | 0.686 |
| 2 opposite (union) | 0.300 | 0.397 | 0.865 | 0.764 | 0.873 |
| 2 opposite (carved) | 0.314 | 0.413 | 0.885 | | |
| 4 (union) | 0.349 | 0.387 | 0.856 | 0.891 | 0.938 |
| **4 (carved)** | **0.406** | **0.450** | **0.907** | | |
| TRELLIS pipeline, 1 photo (11 builds: one failed placement) | 0.343 | 0.333 | 0.841 | | |

**Two photos from opposite corners show 81% of a build's outer blocks, four 92%; with
them photo-first beats the TRELLIS pipeline (0.41–0.45 against 0.33).** With exact geometry
the completion model nearly rebuilds the human build from two opposite photos (0.87) or four
(0.94).

**A completion model for several photos (pf-a3)**: every training sample fuses 1–4 random
photos of the build, each with the front end's kind of errors (`data.FRONT_NOISE`), with
space carving, and photo-like colours (`--views-per-sample 1 4 --front-noise
--photo-colour`). Validated on the four-photo front-end inputs; best at step 2,000; stopped
at 8,000 as it declined. On the front end's input (carved):

| photos | IoU pf-a3 / pf-a1 | F1 within 1 | material pf-a3 / pf-a1 | exact blocks pf-a3 / pf-a1 |
| --- | --- | ---: | --- | --- |
| 1 | 0.223 / **0.258** | 0.629 | **0.091** / 0.042 | **0.026** / 0.009 |
| 2 adjacent | 0.325 / **0.334** | 0.767 | **0.098** / 0.034 | **0.027** / 0.008 |
| 2 opposite | **0.425** / 0.413 | 0.878 | **0.102** / 0.041 | **0.034** / 0.009 |
| 4 | **0.464** / 0.450 | 0.903 | **0.104** / 0.033 | **0.033** / 0.008 |

It adds a little geometry with several photos and roughly triples material and exact-block
accuracy; with one photo it is worse than pf-a1. The four-photo row is slightly optimistic
(the checkpoint was chosen on it). Four photos with pf-a3 against the TRELLIS pipeline (r1,
one photo): IoU 0.464 against 0.333, F1 within 1 0.903 against 0.841, material 0.104 against
0.084, exact blocks 0.033 against 0.033.

### How errors pass through the chain

Every stage is measured on the same builds, so the losses can be read off:

| photo-first | 1 photo | 2 opposite | 4 photos |
| --- | ---: | ---: | ---: |
| ceiling: exact visible blocks + completion | 0.497 | 0.873 | 0.938 |
| lost to what the photos do not show (1 − ceiling) | 0.50 | 0.13 | 0.06 |
| lost to the front end's geometry (ceiling − end to end) | 0.24 | 0.46 | 0.49 |
| end to end | 0.258 | 0.413 | 0.450 |

1. **Errors pass through the chain about one for one; they are not multiplied.** The
   completion model adds about +0.08 IoU whatever its input (0.415 → 0.497 on true visible
   blocks, 0.180 → 0.258 on the front end's). A front-end error becomes an output error of the
   same size.
2. **A stage trained on another stage's errors can make a better input worse.** r1, trained on
   TRELLIS's filled input, lowers the carved input from 0.441 to 0.40. Every learned stage
   has to be trained on the input it will get (the lesson of r3 too).
3. **Independent sources fused well cancel errors instead of adding them.** Carving TRELLIS
   with the photos' seen air (+0.10 on the input), carving one photo with another (+0.06),
   and more photos (+0.19 from 1 to 4) all lower the error.
4. **With more photos the bottleneck moves.** With one photo, half the score is lost to what
   it does not show. With four, almost none is, and the loss is the front end's geometry: the
   exact placement of each photo's surfaces and the photos' registration.
5. **Materials lose most at the colour step.** On true visible blocks material accuracy is
   0.18; with the photo's colours, 0.04–0.06. The photos are much darker than the materials.
