# The dataset

About 1,000 realistic buildings that people built in Minecraft, downloaded with their
licences from Planet Minecraft and Abfielder. Each build is the answer the model learns to
give: the target.

| Dataset | Builds | Train / val / test / set aside |
| --- | --- | --- |
| `data/real-schems-v2` | 1,012 | 908 / 48 / 38 / 18 |
| `data/real-schems-v1` | the first 321 (Planet Minecraft 235, Abfielder 86) | 253 / 32 / 31 / 5 |

v2 contains v1 unchanged (v1's rows, splits and files) plus 691 more Planet Minecraft builds.
The pairs, the test sets and the trained model are made from v1.

## Freezing a collection (`dataset.freeze`)

```
.venv\Scripts\python -m dataset.freeze --source <collection folder> --out data/real-schems-v1
.venv\Scripts\python -m dataset.freeze --source <second collection> --out data/real-schems-v2 --base data/real-schems-v1
```

Freezing copies the collection byte for byte into `source/` (read-only, checked against its
checksums) and writes, beside it:

| File | What |
| --- | --- |
| `index.json` | one row per build: id, file, format, category, creator, source URL, licence and its tier, flags, split |
| `splits.json` | the split, how it was made and every build set aside |
| `near_duplicates.json` | builds whose metadata looks alike (same creator and size) |
| `geometric_duplicates.json` | `dataset.dedupe`: builds that are the same blocks turned, mirrored or cropped |
| `freeze_checksums.sha256`, `FREEZE.md` | the copy's checksums and when and from where it was made |

**Splits.** Builds are split about 80 / 10 / 10 by creator: all of a creator's builds go to
the same split, so the model is never tested on a builder whose style it trained on. Creators
linked by credits, similar names or duplicate builds count as one. The split is seeded and
reproducible. A second collection keeps the first's split and joins new builds to the split
of a creator they share, otherwise to train.

**Set aside** (split "excluded"): builds with an excluded licence, landscape dioramas where the
building is a small part of the scene, a house dug into a cliff whose rooms lose their back
walls when the rock is cropped, display stands, a capture of two unrelated builds, and an
unfinished build.

## Licences

Every build keeps its source and licence. Licences are sorted into tiers:

| Tier | Meaning | Used |
| --- | --- | --- |
| A | the creator allows reuse | trained on, privately |
| B | the licence is unclear | trained on, privately |
| C | personal or non-commercial use | trained on, privately |
| X | no derivatives, or no redistribution | never |

The schematics, the pairs made from them, renders of them and the trained weights are never
shared. Releasing or selling a model would need retraining on tier A only, or checking the
licences again first.

## Normalizing (`dataset.normalize`)

```
.venv\Scripts\python -m dataset.normalize --dataset data/real-schems-v1 [--workers 4] [--only ID ...]
```

Each build goes through four steps. A build already done with the same source and code is
skipped, and the output files are byte-identical on a re-run.

**1. Read** (`read.py`). Every format the collections use: Sponge `.schem` v1, v2 and v3,
the legacy MCEdit `.schematic` (numeric block ids before Minecraft 1.13), and Litematica
`.litematic`. Every file is checked for size, nesting and corruption before it is parsed
(`minecraft/nbt_safe.py`). Block names are brought up to date (old names, legacy ids,
`chain` -> `iron_chain`), and only blocks are read: entities such as paintings are not.

**2. Crop to the building** (`crop.py`). Captures often include landscape. What is natural
terrain (layered ground connected to the bottom and edges of the capture), trees, streets and
car parks well away from the building, and detached bits are removed. Building materials are
never treated as landscape because of their name; when unsure, a block is kept. Each column
gets a ground height; only blocks truly buried below it (a foundation, a basement) are
marked below ground.

**3. Inside and outside** (`interior.py`). The outside is everything the outside air reaches.
Air gets in through openings at least 3 blocks wide and 2 tall (doorways, archways, open
sides), and one block deep through smaller ones (window reveals). So porches, balconies,
arcades, eaves and recessed entrances count as outside, and rooms behind doors and windows
as inside. Covered outside air more than 6 steps from the open air counts as inside, so the
middle of a big hall does not. The **shell** is the layer of blocks the outside touches.

**4. Targets.** The shell and the outside air become the training target in the model's
block vocabulary (`voxel/vocab.py`): every block is a shape (full, slab, stairs facing east,
fence, pane, open trapdoor...: 30 shapes) and a material (oak planks, bricks, white
concrete...: 313 materials), with 1,232 shape-material pairs that exist in the game. Outside
air is air. Everything else is IGNORE and never scored: the inside, decoration, fluids, and
the few blocks the vocabulary cannot hold (98.6% of shell blocks fit). A storey-height
estimate is recorded too, as information only: builds are never rescaled.

| Output | What |
| --- | --- |
| `normalized/<id>.npz` | `blocks` and `palette` (the cropped build), `shell`, `exterior`, `interior` masks, `ground_height` per column, `target_shape` and `target_material` (X, Y, Z: Y up, +X east, +Z south) |
| `normalized/records/<id>.json` | everything measured for one build: format, crop, masks, storeys, flags |
| `normalized/index.json` | one line per build and a summary |

## Reviewing

```
.venv\Scripts\python -m dataset.gallery --dataset data/real-schems-v1     (review/gallery_v2/index.html)
.venv\Scripts\python -m dataset.coverage --dataset data/real-schems-v1    (how much the vocabulary covers)
.venv\Scripts\python -m dataset.dedupe --dataset data/real-schems-v2      (geometric duplicates)
```

The gallery shows each build before and after the crop and a cut-away of its inside.
