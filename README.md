# Mesh2Schem

A neural network that turns the rough 3D model [TRELLIS](https://github.com/microsoft/TRELLIS)
makes from a photo of a building into a Minecraft build: for every block a shape (full block,
stairs, slab, pane, door...) and a material. It learns from buildings that people built in
Minecraft: the dataset has 1,012, and the current network was trained on 253 of them
([why not all of them](#why-not-all-1012-builds)).

## How it learns

Each human build is an answer, and the question is made from it: the build becomes a mesh,
roughed up to look like TRELLIS output, then voxelized at 2 cells per block. The network
learns to turn that back into the build.

| Step | Folder | What it does |
| --- | --- | --- |
| Dataset | `dataset/` | copies the builds with their licences, splits them by creator, and cleans each one: crop to the building, find the ground, mark inside and outside ([dataset.md](docs/dataset.md)) |
| Pairs | `pairs/` | makes each build's TRELLIS-like input ([pairs.md](docs/pairs.md)) |
| Test sets | `testsets/` | real TRELLIS output, from renders of held-out builds and from real photos ([test-sets.md](docs/test-sets.md)) |
| Model | `training/` | the 3D U-Net: training, scoring, export to `.schem` ([model.md](docs/model.md)) |

`voxel/` (meshes to voxels, the block vocabulary) and `minecraft/` (Minecraft files, block
colours, the renderer) are shared by all the steps.

## Setup

Windows, an NVIDIA GPU and Python 3.10. Two environments in this folder (git ignores them):

```
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

py -3.10 -m venv .venv-train
.venv-train\Scripts\python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv-train\Scripts\python -m pip install -r requirements-train.txt
```

`.venv` runs the dataset, pairs and test sets; `.venv-train` runs the network. The test sets
also need a TRELLIS server running on this computer (port 7960). The builds are not in this
repository, as their licences don't allow sharing them, so the steps need your own collection
of schematics.

## Run it

From this folder, with `D` the dataset folder (for example `data/real-schems-v1`). Every step
can be stopped and started again: builds already done are skipped.

```
.venv\Scripts\python -m dataset.freeze --source <collection folder> --out D
.venv\Scripts\python -m dataset.normalize --dataset D --workers 4
.venv\Scripts\python -m pairs.make --dataset D --workers 6
.venv\Scripts\python -m testsets.renders prepare --dataset D --set smooth      (then generate, place)
.venv-train\Scripts\python -m training.data cache --dataset D
.venv-train\Scripts\python -m training.train --dataset D --run r1 --steps 20000
.venv-train\Scripts\python -m training.evaluate --dataset D --run r1
.venv-train\Scripts\python -m training.export D\runs\r1\pred_val\smooth\<id>.npz --out build.schem
```

## How well it works

The best network (run `r1`, step 4,000), on validation builds it never trained on:

| Input | Builds | Shape overlap (IoU) | Right within 1 block | Right material | Exactly right blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| its own kind of input (made from the builds) | 32 | 0.53 | 0.97 | 0.15 | 0.08 |
| real TRELLIS, from smooth renders | 12 | 0.31 | 0.79 | 0.08 | 0.03 |
| real TRELLIS, from blocky renders | 12 | 0.30 | 0.78 | 0.11 | 0.04 |
| real TRELLIS, from real photos | 2 | 0.44 | 0.94 | 0.18 | 0.05 |

It puts blocks where TRELLIS's model is, using stairs and slabs, but it can't do better than
that model. The limits are below, and [model.md](docs/model.md) has the details.

## Limitations

- **Its shape is TRELLIS's shape.** TRELLIS fills in porches, courtyards and the gaps between
  wings, and the network keeps them filled: nothing in its input says where to carve. Fine
  detail is lost too.
- **Most materials are wrong:** about one block in ten is right, so builds look patchy. The
  colours it sees are TRELLIS's, which are dark, with the lighting baked in.
- **Rare block shapes** (walls, panes, trapdoors, doors) are almost never placed.
- **It does best on tall, simple buildings** like towers, and worst on low buildings with
  porches and wings, where TRELLIS fills in the most.
- **Real input is harder than training input.** The network trains on input made from the
  builds, which is cleaner than real TRELLIS output: the same builds score 0.53 shape overlap
  from one and 0.31 from the other.
- **You need your own builds and a TRELLIS server.** The training builds and the trained
  network can't be shared.

## Why not all 1,012 builds?

The 691 builds added later are cleaned, but the network was not retrained on them. Tests on
the current network showed they wouldn't fix what holds it back, so retraining (about half a
day of computing and 45 GB of temporary disk space) would be a small gain at best:

- **More builds can't fix the shape.** Using TRELLIS's model unchanged as the answer scores the
  same shape overlap (0.32) as every trained network, and 98% of the network's wrong blocks
  sit inside the space TRELLIS filled in. The input doesn't show where to carve, and more
  examples don't change that.
- **Changing the training data already made no difference.** Networks trained on more varied
  input, and partly on real TRELLIS output, did no better than the first.
- **Materials would improve only a little.** More builds would delay the point where the
  network starts memorising them (step 4,000 now), but it would still see TRELLIS's dark
  colours.

What would help more is a better input than TRELLIS's: one that keeps the photo's open spaces
and its true colours.

## Data and licences

`data/` (ignored by git) holds `real-schems-v2`, all 1,012 builds, and `real-schems-v1`, the
first 321 with their pairs and test sets (`pairs/`), the 15 GB training cache (`pairs_cache/`)
and the runs (`runs/`).

The builds belong to their creators and are only used to train privately: schematics, pairs,
renders of builds and trained weights are never shared. This repository holds only code and
docs. More in [dataset.md](docs/dataset.md#licences).

## Tests

```
.venv\Scripts\python -m unittest discover -s tests
.venv-train\Scripts\python -m unittest discover -s tests -p "test_training*.py"
```
