# The model

A 3D U-Net that reads a voxelized TRELLIS mesh (2 cells per block) and predicts, for every
block, its shape and its material in the block vocabulary (`voxel/vocab.py`: 30 shapes x 313
materials). Code in `training/`; runs go to `data/real-schems-v1/runs/<run>/`.

## Running it

```
.venv-train\Scripts\python -m training.data cache --dataset data/real-schems-v1      (once per pairs version: 15 GB, about 6 minutes)
.venv-train\Scripts\python -m training.data trellis --dataset data/real-schems-v1    (adds the real TRELLIS train inputs)
.venv-train\Scripts\python -m training.train --dataset data/real-schems-v1 --run r1 --steps 20000
.venv-train\Scripts\python -m training.evaluate --dataset data/real-schems-v1 --run r1 [--checkpoint best.pt] [--split val|test]
.venv-train\Scripts\python -m training.gallery --dataset data/real-schems-v1 --run r1
.venv-train\Scripts\python -m training.export data\real-schems-v1\runs\r1\pred_val\smooth\<id>.npz --out build.schem
```

`training.train --smoke` overfits two small builds as a quick check that everything learns.
`training.export` writes a prediction as a `.schem` (Minecraft 1.21.11) you can paste into a
world.

## What it is

| Part | Module | What |
| --- | --- | --- |
| Data | `data.py` | the pairs unpacked into one memory-mapped cache; training samples 64-block windows (the input's 128 cells) with enough solid in them, in all 8 turns and mirrors (stairs and doors turn with the grid) |
| Network | `model.py` | each block's 8 input cells become channels (so sub-block detail reaches the first layer), then a 4-level residual U-Net (32 / 64 / 128 / 256 channels, 7.2 M parameters). A shape head at every block, a material head only where a block is solid; materials a shape cannot have are masked. A height channel says how far above the grid's floor a block is |
| Whole builds | `predict.py` | a build bigger than one window is predicted in overlapping windows and blended (`voxel/tiling.py`) |
| Scoring | `score.py` | the same scores for every run (below) |
| Training | `train.py` | weighted cross-entropy (rare shapes count more), bf16, about 0.44 s a step of 8 windows on an RTX 4060 Ti; validates on whole val builds and keeps `best.pt` (synthetic val) and `best_trellis.pt` (real-TRELLIS val) |
| Evaluation | `evaluate.py`, `gallery.py` | scores on the val or test builds, the predictions, a picture gallery |

## How it is scored

Only the supervised blocks count: the build's outside shell and the outside air around it,
never its inside.

| Score | What |
| --- | --- |
| IoU | the predicted solid blocks against the build's (shape overlap) |
| F1 within 1 block | a solid block counts if the other side has one within a block |
| shape | the exact shape (stairs and their direction, slabs, panes...) where both are solid |
| material | the exact material where both are solid |
| ΔE | how far apart the two materials' colours are (a near miss costs less than a wrong colour) |
| exact blocks | the build's blocks predicted exactly, shape and material |

Builds count equally. Checkpoints are chosen on (IoU + exact blocks) / 2 over the synthetic
val builds.

## The runs

| Run | What changed | Result |
| --- | --- | --- |
| `smoke` | two builds, to check it learns | reproduces them exactly |
| `r1` | 20,000 steps on the synthetic pairs | **the model**: best at step 4,000 |
| `r2` | input augmentation (colour, noise, blobs, dents), softer weights for rare shapes | not better: more right shapes, fewer right blocks |
| `r3` | half the samples from real TRELLIS output of the train builds (`real_trellis_train`) | the same as r1, within the noise |

`r1`'s best checkpoint on the val builds:

| Input | Builds | IoU | F1 within 1 block | Shape | Material | Exact blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic | 32 | 0.53 | 0.97 | 0.56 | 0.15 | 0.08 |
| real TRELLIS, smooth renders | 12 | 0.31 | 0.79 | 0.45 | 0.08 | 0.03 |
| real TRELLIS, blocky renders | 12 | 0.30 | 0.78 | | 0.11 | 0.04 |
| real TRELLIS, real photos | 2 | 0.44 | 0.94 | | 0.18 | 0.05 |

The test split has not been scored yet: score it once, at the end, on the checkpoint chosen.

## What limits it

1. **Its shape is the input's shape.** Taking the input's own solid as the prediction scores
   the same IoU on real TRELLIS input (0.32) as every run. TRELLIS fills porches, courtyards
   and the gaps between wings, and 98% of the model's false blocks lie inside that filled
   space: the input does not say where to carve.
2. **Too few builds for 313 materials.** Material accuracy peaks around step 4,000 and then
   falls: 253 train builds are memorised. The colours it sees are TRELLIS's dark baked ones.
3. **Synthetic and real inputs differ.** The same builds score 0.53 IoU from synthetic input and
   0.31 from real TRELLIS: TRELLIS fills more open space (1.4 times the true solid, against
   1.2) and loses more of the build.
4. **Rare shapes** (walls, panes, trapdoors, doors) are almost never predicted.
5. **The scores are noisy**: 12 real-TRELLIS val builds give about ±0.02 IoU.

More builds (`real-schems-v2` has 1,012, but pairs exist only for v1's 321) and the photo's
own colours as an input are the clearest next steps.
