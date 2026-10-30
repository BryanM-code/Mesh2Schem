# Test sets

The pairs' inputs only imitate TRELLIS. To know how the model does on real TRELLIS output,
real TRELLIS is run on pictures of held-out builds, and on real photos.

All of them need the TRELLIS server running locally: `C:\TRELLIS\run.bat` (port 7960;
`http://127.0.0.1:7960/ping` answers when it is up).

## TRELLIS on renders of the builds (`testsets/renders.py`)

```
.venv\Scripts\python -m testsets.renders prepare  --dataset data/real-schems-v1 --set smooth
.venv\Scripts\python -m testsets.renders generate --dataset data/real-schems-v1 --set smooth
.venv\Scripts\python -m testsets.renders place    --dataset data/real-schems-v1 --set smooth
```

25 builds from val and test (spread over the categories) are rendered as a photo from a
street-level corner view, each photo goes through TRELLIS, and the mesh TRELLIS returns is
placed in the build's frame and voxelized exactly like a pair's input, so the model can be
scored on it against the real target.

| Set | Folder (in `pairs/`) | The photo |
| --- | --- | --- |
| `smooth` | `real_trellis_smooth/` | **the main set.** The build's outside, smoothed so its roofs are slopes rather than stair steps, lit by a strong sun: TRELLIS's output on these matches its output on real photos in blockiness and brightness |
| `blocky` | `real_trellis/` | the raw blocks: a stress test, where TRELLIS copies Minecraft's stepped roofs |
| `train` | `real_trellis_train/` | the smooth photo of every train build (244 usable): real TRELLIS inputs for training, never scored |

`prepare` picks the builds and renders the photos, `generate` sends them to TRELLIS (resumable),
and `place` fits each mesh to its build: scaled to the build's height, turned and shifted to
the best overlap, then voxelized and checked like a pair. `trials` tests other smoothing
amounts on 5 builds; `status` reports progress.

Each build's folder holds `photo.png` (what TRELLIS saw), `trellis.glb` (what it returned),
`input.npz` (the pair: TRELLIS's input and the build's target) and `stats.json`.

## TRELLIS on real photos (`testsets/photos.py`)

Free-licence photos (Wikimedia Commons) of real buildings that some builds copy, and one
photo of a real barn. The build is its builder's version of the building, so the comparison
is rough (proportions differ), but it is the real use: a real photo through TRELLIS.

```
.venv\Scripts\python -m testsets.photos fetch    --dataset data/real-schems-v1
<python with rembg> -m testsets.photos cutout --dataset data/real-schems-v1
.venv\Scripts\python -m testsets.photos generate --dataset data/real-schems-v1
.venv\Scripts\python -m testsets.photos place    --dataset data/real-schems-v1
```

`fetch` downloads each photo once with its licence, author and URL; `cutout` removes the
background (rembg, `isnet-general-use`: it needs an environment with rembg installed); `generate`
runs TRELLIS; `place` fits the mesh to the build, here also searching the scale and the
proportions. The photos stay in `data/` and are never shared. A photo of a train build is
only a visual check; only those of val and test builds are scored.

## How the model is scored on them

`training.evaluate --sets smooth blocky photos` (see [model.md](model.md)). Only val builds
are used while developing; test is scored once, at the end.
