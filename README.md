# Mesh2Schem

A model that turns a TRELLIS mesh of a building (a 3D model made from one photo) into a
Minecraft build, trained on about 1,000 buildings that people built in Minecraft.

TRELLIS gives a blobby, coloured 3D shape. The model's job is to turn that shape, voxelized,
into real Minecraft blocks: stairs on the roofs, slabs, panes, doors, and a material for every
block. It learns from human builds: each human build is the answer (the target), and the
model's input is made from it to look like what TRELLIS would give for that building.

## The pipeline

```
human schematics -> dataset -> normalize -> pairs -> train -> evaluate on the test sets
```

| Step | Package | What it does | Doc |
| --- | --- | --- | --- |
| Dataset | `dataset/` | freezes the collected schematics (a read-only copy, licences, train/val/test split by creator) | [dataset.md](docs/dataset.md) |
| Normalize | `dataset/` | reads every file format, crops each build to the building, finds the ground, marks inside and outside, and encodes the outside as the model's targets | [dataset.md](docs/dataset.md) |
| Pairs | `pairs/` | makes the model's inputs: the build's outside as a mesh, degraded to look like TRELLIS output, voxelized at 2 cells per block | [pairs.md](docs/pairs.md) |
| Test sets | `testsets/` | real TRELLIS run on renders of held-out builds, and on real photos of buildings that builds copy | [test-sets.md](docs/test-sets.md) |
| Model | `training/` | a 3D U-Net, its training, scoring, evaluation, and export of its predictions as `.schem` | [model.md](docs/model.md) |

Shared code: `voxel/` (meshes to voxels, the block vocabulary the model predicts),
`minecraft/` (reading and writing Minecraft files, block colours, the renderer).

## Run it

From this folder. The data steps run in `.venv`, the model in `.venv-train`
(see [Setup](#setup)). `D` stands for `data/real-schems-v1`.

```
.venv\Scripts\python -m dataset.normalize --dataset D --workers 4
.venv\Scripts\python -m pairs.make --dataset D --workers 6
.venv\Scripts\python -m testsets.renders prepare --dataset D --set smooth      (then generate, place)
.venv-train\Scripts\python -m training.data cache --dataset D
.venv-train\Scripts\python -m training.train --dataset D --run r1 --steps 20000
.venv-train\Scripts\python -m training.evaluate --dataset D --run r1
.venv-train\Scripts\python -m training.export D\runs\r1\pred_val\smooth\<id>.npz --out build.schem
```

Every step is resumable: a build that is already done is skipped. The test sets need the
TRELLIS server running locally (`C:\TRELLIS\run.bat`, port 7960).

## The data

`data/` holds the datasets and everything made from them, and is never committed:

| Folder | What |
| --- | --- |
| `data/real-schems-v2/` | the full dataset: 1,012 builds, normalized |
| `data/real-schems-v1/` | its first 321 builds, on which the pairs, the test sets and the model were made |
| `data/real-schems-v1/pairs/` | the training pairs and the test sets |
| `data/real-schems-v1/pairs_cache/` | the pairs unpacked for training (15 GB, rebuilt by `training.data cache`) |
| `data/real-schems-v1/runs/` | the training runs: checkpoints, logs, evaluations, predictions |

## How well it works

The best model (run `r1`, its step-4,000 checkpoint), on the validation builds:

| Input | Builds | Shape overlap (IoU) | Right within 1 block | Right material | Exactly right blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| synthetic (its own kind of input) | 32 | 0.53 | 0.97 | 0.15 | 0.08 |
| real TRELLIS, smooth renders | 12 | 0.31 | 0.79 | 0.08 | 0.03 |
| real TRELLIS, blocky renders | 12 | 0.30 | 0.78 | 0.11 | 0.04 |
| real TRELLIS, real photos | 2 | 0.44 | 0.94 | 0.18 | 0.05 |

It works to some extent: it places blocks where the TRELLIS mesh is, with stairs and slabs,
but its shape is the input's shape, and it picks the right material for only about one block
in ten. What limits it is in [model.md](docs/model.md).

## Setup

Windows, an NVIDIA GPU, Python 3.10. Two environments in this folder, left out of git:

| Folder | What runs in it | How to make it |
| --- | --- | --- |
| `.venv` | the dataset, the pairs and the test sets | `requirements.txt` |
| `.venv-train` | the model: caching, training, evaluation, export | `requirements-train.txt` |

## Tests

```
.venv\Scripts\python -m unittest discover -s tests
.venv-train\Scripts\python -m unittest discover -s tests -p "test_training*.py"
```

## Licences

The builds belong to their creators. They are used to train a model privately; the
schematics, the pairs made from them, renders of them and the trained weights are never
shared. This repository holds code and docs only. Details: [dataset.md](docs/dataset.md#licences).
