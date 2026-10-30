# Training pairs

A pair is what the model learns from: an **input** that looks like what TRELLIS would make of
a building, and the **target**, the human build itself. Real TRELLIS output for 1,000 builds
would need 1,000 photos of buildings nobody has photographed, so the input is made from the
target instead, and made to look like TRELLIS output.

```
.venv\Scripts\python -m pairs.make --dataset data/real-schems-v1 --workers 6 [--only ID ...]
```

Each build gets 4 inputs (variants), each from its own seed. From v1: 316 builds, 1,264
pairs (train 1,012 / val 128 / test 124). The run is resumable, and the same build, variant
and code always give a byte-identical file.

## How an input is made (`pairs/make.py`)

1. **Mesh the outside** (`meshify.py`). The build's outside becomes a coloured surface: only
   faces the outside air touches. Block shapes are real to 1/16 of a block (slabs, stairs and
   their corners, panes, fences, walls, doors, trapdoors). Doorways and open windows get a
   dark cap where the outside meets the inside; insides and the ground are never meshed.
2. **Misplace it slightly.** A turn of up to 5 degrees and a shift of up to half a block,
   like the error of placing a real TRELLIS mesh. The target never moves.
3. **Degrade it** (`degrade.py`), like TRELLIS:
   - a coarse resolution (128, 160 or 192 cells across the build), blurred, so one-block
     details melt and stair-stepped roofs become smooth slopes;
   - surface noise, small holes and floating fragments;
   - TRELLIS's colours: lit from one side, a near-black back side and dark roofs, its own
     overall brightness, blotches, smears and hue drift, dark windows.
4. **Voxelize** at 2 cells per block, in exactly the target's frame: target block (x, y, z)
   covers input cells (2x..2x+1, 2y..2y+1, 2z..2z+1). The solid inside the surface is filled.
5. **Check** (`make.py`, validation): the input must overlap the target and line up with it
   to within a block. An input far bigger than its build (open barns, a sprawling complex) is
   kept and flagged `low_iou`.

## Do the inputs look like TRELLIS? (`calibrate.py`)

```
.venv\Scripts\python -m pairs.calibrate --dataset data/real-schems-v1
```

The degradation ranges were tuned so the inputs match real TRELLIS output on the same
statistics. The reference is TRELLIS meshes of real photos (`pairs/trellis_reference/`):

| Statistic | Inputs (p10 / p50 / p90) | TRELLIS on real photos |
| --- | --- | --- |
| lightness (L\*) | 11.7 / 21.6 / 37.4 | 11.0 / 21.9 / 33.7 |
| colourfulness (C\*) | 3.5 / 7.1 / 12.0 | 2.8 / 6.7 / 11.6 |
| blockiness (stair45, median) | 0.20 | about 0.25 |

What the inputs still lack: TRELLIS also keeps clutter (vines, hay, a fringe of ground),
completes the hidden side by symmetry and invents geometry; the inputs keep more of the build
than TRELLIS does (89% of its solid, against 71%).

## The files (`data/real-schems-v1/pairs/`)

| Path | What |
| --- | --- |
| `synthetic/<id>/v<k>.npz` | one pair per build and variant, k = 0..3 |
| `records/<id>.json`, `index.json` | how each pair was made: every random parameter, its checks and statistics |
| `calibration.json`, `trellis_reference.json` | the comparison with TRELLIS |
| `gallery/index.html` | `pairs.gallery`: each build beside its inputs |

The arrays in a pair (X, Y, Z: +X east, Y up, +Z south):

| Key | Type | Shape | What |
| --- | --- | --- | --- |
| `input_occ` | uint8 0/1 | (2X, 2Y, 2Z) | solid cells: the reconstruction as a filled blob |
| `input_surface` | uint8 0/1 | (2X, 2Y, 2Z) | cells the surface passes through |
| `input_rgb` | uint8 | (2X, 2Y, 2Z, 3) | the colour seen from outside (sRGB), on the blob's outer layer |
| `target_shape` | uint8 | (X, Y, Z) | the vocabulary's shape id; 255 = IGNORE |
| `target_material` | uint16 | (X, Y, Z) | the vocabulary's material id; 65535 = IGNORE |
| `ground_height` | int16 | (X, Z) | the ground in each column; -1 = none |

There is no glass, door or normal channel: TRELLIS gives none, so the model must not depend
on one.
