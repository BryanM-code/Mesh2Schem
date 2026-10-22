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

### Several photos registered by a model: MapAnything and Depth Anything 3 (2026-10-19/20)

The four-photo results above assumed the cameras known. Here the photos alone go through a
multi-view model that estimates the cameras and a 3D point per pixel together
(`photofirst/front_mv.py`): **MapAnything** (`facebook/map-anything-apache`, Apache 2.0, in
`r2m_train`) and **Depth Anything 3** (`depth-anything/DA3-LARGE-1.1`, in its own `r2m_da3`
venv: it needs numpy < 2; its Hugging Face card says Apache 2.0, the DA3 repository's table
CC BY-NC 4.0). Both can be given camera hints (intrinsics and poses at any scale). The 25
evaluation builds (12 val, 13 test), four corner photos each; the reconstruction is placed in
the target's frame from its cameras and refined against the target (turn ±6°, scale ±8%,
shift: the placement training pairs use), so the scores are the reconstruction's shape, as
for every input so far. End to end: pf-a3 on the 12 val builds.

Hints: **rough** (each photo's direction within ±10°, its tilt within ±5°, the true focal
length), **coarse** (direction within ±25°, tilt taken as level, focal within ±10%: about what
"which side is this photo from" and EXIF give), **true** (the renders' cameras).

| front end, hints | recall exact / within 1 | precision exact / within 1 | camera error | 90° flips | end to end IoU |
| --- | --- | --- | ---: | ---: | ---: |
| MapAnything, none | 0.282 / 0.545 | 0.445 / 0.763 | 24.3° | 6 / 25 | 0.268 |
| MapAnything, rough | 0.534 / **0.907** | 0.467 / **0.839** | 5.3° | 0 | **0.348** |
| MapAnything, coarse | 0.343 / 0.725 | 0.275 / 0.605 | 17.7° | 0 | 0.223 |
| MapAnything, true | 0.613 / 0.940 | 0.545 / 0.915 | 0.3° | 0 | 0.436 |
| DA3, none | 0.293 / 0.625 | 0.394 / 0.716 | 8.1° | 1 | 0.206 |
| DA3, rough | 0.486 / 0.866 | 0.433 / 0.777 | 5.4° | 0 | 0.295 |
| DA3, coarse | 0.342 / 0.701 | 0.283 / 0.588 | 17.9° | 0 | 0.198 |
| DA3, true | 0.595 / 0.936 | 0.560 / 0.916 | 0.0° | 0 | 0.417 |
| MapAnything, side labels + tilt from the photo (direction ±25°, tilt ±3°) | 0.460 / 0.827 | 0.398 / 0.748 | 10.8° | 0 | 0.272 |
| for reference: TRELLIS pipeline (r1), 1 photo | | | | | 0.333 |
| for reference: MoGe-2 per photo, cameras known, carved | | | | | 0.464 |

1. **MapAnything is the better front end at every hint level.** DA3 flips cameras less often
   on its own, but its depth is less consistent with its cameras (the placement's scale
   search ends at its limit on most builds).
2. **Rough hints are what make several photos work**: no flips, 91% of the visible blocks
   within one block, and end to end above the TRELLIS pipeline (0.348 against 0.333).
3. **Coarse hints are worse than none.** The models follow the hints rather than correct
   them, so a 25° error in the hint becomes an 18° camera error. Using a coarse hint only to
   break ties (keep the photos-only result unless its cameras contradict the user's sides:
   6 of 25 builds switch) gives recall within 1 of 0.71 against 0.55, still far below rough
   hints. Side labels with the tilt measured from the photo (±3°) are better than coarse
   hints but still far from rough ones (0.827 against 0.907 within 1, end to end 0.272 against
   0.348): the direction error is what hurts. **A real multi-photo app needs camera hints good
   to about 10°**: a phone's compass
   heading (EXIF `GPSImgDirection`) for the direction and the photo itself for the tilt
   (a depth model's walls give it to about 2°, `front.frame_from_normals`).
4. **The largest builds still fail** (recall within 1 of 0.60–0.71 on four of them): at the
   model's 518-pixel input a 150-block build gets about three pixels a block.
5. **Space carving across the photos** (`to_grid(..., carve=True)`, as `data.fuse_views`)
   raises precision but, where the photos do not line up exactly, deletes true walls one
   photo saw. On DA3 with rough hints: precision within 1 0.78 → 0.91, recall within 1
   0.87 → 0.68, end to end 0.295 → 0.255. Not used: the registration is not exact enough.

### Why the outputs look like "block vomit"

The block scores count exact voxels; they do not say why the outputs look wrong. Three
measures, on the same builds, with the human builds as the reference:

**Does it look like the human build?** (`photofirst/looks.py`.) Output and human build
rendered the same way from two corners and compared with DINOv2 image features (ViT-L/14;
`global`: whole image, `patch`: local structure). The reference rows place the scale:

| | global | patch |
| --- | ---: | ---: |
| a different human build | 0.553 | 0.507 |
| rule-based building mode (D18 baseline) | 0.626 | 0.552 |
| photo-first, 4 photos (pf-a3, MoGe, cameras known) | 0.690 | 0.641 |
| TRELLIS pipeline (r1) | 0.696 | 0.642 |
| the human build's geometry in one material | 0.854 | 0.797 |
| the human build with every block a full cube | 0.942 | 0.892 |

The outputs are a third of the way from "some other building" to the human build. Full cubes
instead of stairs and slabs cost little; one material instead of the build's palette costs
more; the outputs' geometry costs most: **roof form, openings and silhouette first, then
materials**.

**Not too much noise, too little structure.** Statistics without a target
(`training/quality.py`) over the visible blocks: the outputs are *more* uniform than the human
builds (neighbouring blocks of one material 0.79 against 0.72; materials making up 90% of
the blocks 6.5 against 10.2; single odd blocks 3.6% against 7.8%), and use stairs and slabs
and glass about as often. What is wrong is where things are: within one block of the human
build's own, stairs are found with F1 0.44–0.49 and face the right way half the time, thin
parts (fences, panes, trapdoors) are found a quarter of the time, glass with F1 0.22–0.48.
Even given the exact visible blocks with their true colours, the completion model's shapes
are right as often as "every block a full cube" (0.48–0.54 against 0.51) and its materials
14–22% of the time.

**Colour: the photos hold little material information.** Every truly visible block's most
frontal face sampled in the photos through the *true* cameras (no reconstruction error), the
brightest photo kept, then the nearest material by colour:

| colour as | material exact | colour difference (ΔE, median) |
| --- | ---: | ---: |
| photographed | 0.009 | 34.2 |
| one exposure gain per build (fitted to the truth) | 0.097 | 15.1 |
| one gain per face direction (fitted to the truth) | 0.124 | 12.6 |
| hue only (chromaticity) | 0.214 | 12.1 |
| the materials' own colours (no photo) | 0.836 | 0 |

The light and exposure of the photos, not the reconstruction, lose most of it; the models'
0.09–0.10 is about what plain matching gets. On the front end's own blocks it is worse: a
colour classifier on 300,000 seen blocks of the real training pairs (nearest centroid,
fitted on half the builds) gets 2–3% of the materials, below always naming the most common
one (11%); a block's colour mixes its neighbours' (its points are placed to about a block).
Materials have to come from colour averaged over whole surfaces and from priors learned on
the human builds (what is used for roofs, walls, trim).

A rule from the human builds that did not help: 96% of their roof stairs on a slope face up
the slope; turning the outputs' roof stairs that way changed facing from 0.64 to 0.62.

### Real front-end pairs and iterative completion (2026-10-20)

**Real training pairs** (`photofirst/mv_pairs.py`). Every training build was rendered from
8 directions round it (`render --train-views 8`: its own 16 viewpoints thinned, the
evaluation photos' look, a random sun direction and ±25% exposure per build; seen sets not
cast, the target is the whole build). Six photo sets per build (two of four photos, two of
three, two of two: `front_mv.view_subsets`, about as people walk round a building) go through
MapAnything with rough hints; the reconstruction is placed on the target (cameras, then the
target refinement, fast: a coarse then fine turn and scale) and turned into blocks at two
cells per block (`to_grid(res=2)`: what the points show below a block: slopes, steps,
slabs). That is the model's own input grid, no repetition. 1,518 pairs, 253 builds.
Training mixes them (70%) with the simulated inputs (`--real-pairs ma_rough`).

**Iterative completion** (`photofirst/masked.py`, MaskGIT on block grids). One pass takes
each block's most likely shape and material separately, so where the photos allow several
good answers every block takes the safest one, and uncertain details (posts, frame timbers,
window rhythms) come out as scattered single blocks: the outputs' streaks and smears are the
completion model's, not the front end's (its blocks are compact). The masked model also sees
blocks already decided (embeddings of shape and material added after the stem); in
training a random share of the target's supervised blocks is given (masked share cos(πu/2),
u uniform) and the loss is on the rest; decoding runs several passes, sampling the
undecided blocks' shapes (Gumbel, temperature falling to 0) and fixing the surest each pass
(cosine schedule), so later blocks follow earlier ones. Inside blocks are never fed back
(training never gives them).

**pf-m1: the masked model on the simulated inputs** (pf-a3's training data and settings,
started from pf-a3, stopped at 2,000 steps; best on the front-end inputs at 1,000). On the
four-photo MapAnything inputs (rough hints), the 12 val builds:

| | IoU | F1 within 1 | material | stairs F1 | thin F1 | glass F1 | stair facing | lone blocks | looks (global / patch) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pf-a3, one pass | **0.348** | 0.844 | 0.102 | 0.44 | 0.29 | 0.32 | **0.64** | 0.1% | 0.641 / 0.619 |
| pf-m1, one pass | 0.333 | 0.842 | 0.105 | 0.42 | 0.35 | 0.34 | 0.54 | 0.1% | 0.650 / 0.625 |
| pf-m1, 8 passes, sampled (τ = 1) | 0.334 | **0.845** | 0.106 | 0.44 | **0.38** | 0.34 | 0.49 | 1.2% | 0.638 / 0.620 |
| pf-m1, 8 passes, greedy (τ = 0) | 0.334 | 0.843 | **0.106** | 0.43 | 0.34 | 0.34 | 0.51 | 0.1% | **0.653 / 0.627** |

The block scores barely move; what moves is how the build reads. Decoded greedily, pf-m1's
roofs come out as stepped stairs and slabs where pf-a3 left a grey cap, and walls show a
regular frame or window rhythm where pf-a3 smeared materials (ab-6183, the paradise hotel).
Sampling (τ = 1) adds lone blocks and looks worse. The posts under eaves remain.

**pf-b1: trained on the real pairs** (70% of samples from them, 30% simulated as pf-a3;
started from pf-a3; 6,000 steps of 6; best on the four-photo half-block inputs at 4,000).
On those inputs (`ma_rough_res2`, the 12 val builds; pf-a3 never saw half-block inputs: on
block inputs it gets 0.348):

| | IoU | F1 within 1 | shape | material | ΔE | exact | stairs F1 | glass F1 | facing | palette90 | stairs + slabs | looks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pf-a3 | 0.343 | 0.817 | 0.467 | 0.086 | 27.6 | 0.014 | 0.44 | 0.23 | **0.62** | 7.4 | 0.42 | **0.633** / 0.602 |
| pf-b1 | **0.364** | **0.865** | **0.502** | **0.123** | **24.5** | **0.043** | 0.45 | **0.34** | 0.51 | 9.0 | 0.27 | 0.624 / 0.600 |
| human builds | | | | | | | | | | 10.2 | 0.45 | |

Training on the front end's own output is the clearest gain of the round: materials +43%,
exact blocks ×3, windows placed much better, a palette closer to the human builds', half
the stray posts. It leans on full cubes (stairs and slabs 0.27 of the visible blocks against
0.45), and the overall look does not move: some builds read far better (the paradise hotel's
white walls and glass bands), others take the wrong palette (the newspaper turns cream).

With two photos (opposite corners) pf-b1 gets IoU 0.340 (pf-a3 0.297), F1 within 1 0.829,
material 0.107: two photos with pf-b1 match the TRELLIS pipeline (0.333 from one photo).

**pf-bm1: the masked model on the real pairs** (pf-b1's data, `--masked`, from pf-a3)
**failed**: on the four-photo inputs one pass gives IoU 0.300 (pf-b1 0.364), eight passes
0.250 sampled and 0.239 greedy. It over-fills: precision 0.338 → 0.257 from one pass to
eight while recall rises 0.756 → 0.833. Decided solid blocks pull their neighbours solid
pass after pass (thick walls, filled volumes), and sampling leaves 5% lone blocks and 7%
posts: the "block vomit" made worse. Training only ever shows the model correct decided
blocks; decoding shows it its own mistakes, more of them on the noisier real input
(exposure bias). pf-m1 on the simulated input went the other way (precision 0.460 →
0.479, recall 0.585 → 0.519).

**pf-bm2: the masked model trained on its kind of mistakes**: from pf-b1, `--masked
--wrong-tokens 0.15` (15% of the decided blocks given in training are wrong: solid for air,
air for solid, another material), validated with greedy decoding; best at 1,000 of 4,000
steps (its validation then drifted down). Four photos:

| | IoU | F1 within 1 | precision | material | exact | thin F1 | glass F1 | looks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pf-b1, one pass | **0.364** | **0.865** | **0.413** | **0.123** | 0.043 | 0.35 | 0.34 | **0.624** / 0.600 |
| pf-bm1, 8 passes greedy | 0.239 | 0.681 | 0.257 | 0.082 | 0.031 | 0.24 | 0.28 | 0.564 / 0.538 |
| pf-bm2, one pass | 0.314 | 0.827 | 0.342 | 0.121 | **0.048** | **0.39** | 0.35 | 0.601 / 0.592 |
| pf-bm2, 8 passes greedy | 0.316 | 0.819 | 0.345 | 0.120 | 0.047 | **0.39** | **0.36** | 0.597 / 0.589 |

Wrong decided blocks in training stop the decoding from over-filling (precision steady over
the passes), and pf-bm2 places thin parts and windows best of all the models, but it adds
more solid than pf-b1 and stays below it on geometry and looks; with two photos 0.289–0.296
against pf-b1's 0.340. **pf-b1 is the best model of the round.** The masked objective on
the real inputs needs the model's own decisions as the decided blocks (on-policy), not only
corrupted true ones.

### Real photos: no target to place on (2026-10-20)

Everything above placed the reconstruction on the human build (`refine_to_target`): the
cameras' similarity to the renders' true cameras, then the turn, scale and shift that best
overlap the target. A real photo has neither. `photofirst/photos_to_schem.py` runs the
pipeline on photo files alone:

1. each photo upright (EXIF orientation), its compass heading and 35 mm focal from EXIF
   (`real.exif_info`), the building cut out by the app's own background removal
   (`trellis_client._prepare_cutout`, run in `r2m_app`)
2. hints: the compass heading (or `--yaws`), the camera's tilt measured from the photo
   (`real.tilt`: MoGe-2's point map, its normals, the up the walls and roofs imply; on the 100
   evaluation renders 1.8 degrees median error, 5.6 at p90; searching the roll too is worse,
   8.7 at p90), the focal
3. MapAnything on all photos with the hints; placed without a target (`place.free_placement`):
   the up perpendicular to every camera's right axis (a hand-held camera is level), the walls'
   directions turned onto the grid, metric scale (from a second, unhinted MapAnything pass:
   the hinted one takes the hints' unit-free scale) or `--height`, ground at y = 0
4. the completion model, `build.schem` (Sponge v2, 1.21.11 states, neighbour states finished,
   hollowed to the shell: the model's inside is never supervised), previews

**The up direction.** On the 25 evaluation builds' four photos (DA3's reconstructions with
rough hints, against the true cameras' up): the cameras alone 1.5 degrees off (median, 2.4 at
worst); levelling on the reconstruction's surfaces (walls perpendicular, roofs along) is worse,
1.8 median but 7.3 at p90 and 22 at worst (noisy surfaces, slopes); the hints' own frame 11.
With MapAnything and compass hints: 0.5 degrees median, 3.1 at p90.

**What each part costs** (MapAnything, four photos, pf-b1, half-block inputs, the 12 val
builds; for scoring, a target-free result is put on the target by a quarter turn and a
whole-block shift, `training.score.align`):

| hints | placement | scale | IoU | F1 within 1 | precision | recall |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| rough (yaw ±10°, pitch ±5°) | on the target | refined on the target | 0.364 | 0.865 | 0.413 | 0.745 |
| compass (yaw ±12°, tilt from the photo) | on the target | refined on the target | 0.349 | 0.865 | 0.398 | 0.740 |
| compass | free, the input moved onto the target before the model | the true cameras' | 0.330 | 0.836 | | |
| compass | **free, as on real photos** (aligned after, by the input's alignment) | the true cameras' | **0.299** | 0.797 | 0.340 | 0.712 |
| compass | free, as on real photos (aligned after, by the prediction's) | the true cameras' | 0.283 | 0.790 | 0.307 | 0.780 |
| compass | free, as on real photos (the prediction's alignment, no room below) | the target's height | 0.235 | 0.715 | 0.262 | 0.751 |

A phone's compass is as good a hint as the rough one (0.349 against 0.364). Placed without
the target, the front end's blocks are nearly as good (within a block, recall 0.83 against
0.84 placed on the target), and moved onto the target before the model they give 0.330: the
placement itself costs 0.02. What costs more is where the build sits in its grid: the model
reads absolute height, and the photos rarely show a build's base (the human builds have a
ground plate or foundation a few blocks below what the photos show), so a free grid with the
lowest visible point at y = 0 leaves no room for it. Room below (`free_placement(below=...)`:
2 blocks, the median offset) helps (0.274 → 0.283 by the prediction's alignment); a real-photo
run with a phone's compass scores about 0.30 (0.299 by the input's own alignment). Choosing
the scale by the build's height costs a further 0.04 against the true scale (spires, ground
plates and unseen tops make height a poor ruler); on real photos the user chooses the scale.

**Pairs placed as real photos will be** (`mv_pairs --placement free --hints compass_sim`:
the free placement at the true cameras' scale, then turned and shifted onto the target by a
quarter turn and whole blocks, `front_mv.onto_target`, so the placement's own tilt and yaw
errors stay in the input; 1,518 pairs) did not help: pf-b3 (from pf-b1, 4,000 steps) 0.327
against pf-b1's 0.330 on the free inputs moved onto the target, 0.351 against 0.364 on the
inputs placed on it. pf-b1 stays the model.

**Real photos.** On a real photo (single views from `pairs/real_photo_replicas`, a human
replica each) the whole path runs (orientation from EXIF, the cutout, MapAnything, the free
placement levelled 1.3 degrees, the model, the `.schem`), and shows what the evaluation renders
do not: a close-up that crops the building (the Carson mansion) defeats the cutout (the
segmentation keeps 0.5% of the frame and is rejected: the whole photo goes in), and one photo
leaves the back to the model (the Coleraine church comes out a speckled box open at the back).
2-4 photos each showing the whole building are the input this pipeline needs.
