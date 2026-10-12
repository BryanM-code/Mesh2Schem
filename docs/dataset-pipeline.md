# Mesh-to-Schem dataset pipeline

## Status (2026-10-11)

Schematic-first pivot ([plan, version 4](mesh-to-schem-plan.md#version-4-2026-10-11-schematic-first-pivot)):
human-built schematics are the training targets and inputs are derived from them. The rest
of this document describes the version 3 Objaverse pipeline; the parts marked ACTIVE carry
over to the new pairs.

| Part | Files | Status |
| --- | --- | --- |
| Objaverse synth path | `synth/select.py`, `synth/fetch.py`, `synth/objaverse_index.py`, `synth/run.py` | **PARKED** (D13): kept, possible pretraining source |
| Grammar houses | `house_grammar.py`, `gen_houses.py`, `house_params.py` | **PARKED** |
| The 16 synthetic-house builds in `data/builds/` | — | **EXCLUDED** from any real-build dataset |
| TRELLIS-like degradation, real-TRELLIS comparison, real evaluation | `synth/degrade.py`, `synth/trellis_compare.py`, `synth/real_eval.py` | **ACTIVE**, reused |
| Shared voxel core | Real2Minecraft `mcvoxel/` | **ACTIVE**, reused |
| Rule-based building mode | Real2Minecraft `building/` | **ACTIVE** as the baseline (D18) |

*Version 3 status, superseded by the table above:* Current phase: dataset quality review only. No model training or large dataset generation.
The resumed audit and 40-source follow-up are documented in
[dataset-quality-audit.md](dataset-quality-audit.md). Historical pilots are preserved;
`pilot-v3` stopped at 193/200 and predates the final crop/size fixes. Use `pilot-v4` for
the current review.

How a clean 3D model of a building becomes a training pair: a TRELLIS-like messy input and
the Minecraft build a player would make of it. Code: `synth/`, on top of the shared `mcvoxel`
package (in the Real2Minecraft repo). Decisions behind it: `docs/mesh-to-schem-plan.md`;
licences: `docs/dataset-licensing.md`.

```
source model (Objaverse 1.0, CC0 / CC BY only)
  -> metadata filters: licence, text rules, photogrammetry, size    (before any download)
  -> [classifier hook: none in V1 -- TODO]
  -> download to a temporary folder
  -> clutter: vegetation, vehicles, figures dropped by part / material name
  -> geometry checks: already blocky, scan-like mesh                 (cheap, no voxels)
  -> orient (facades on the grid axes)
  -> scale at 1:1 with the player (1 block = 1 m): confident / uncertain (kept, flagged) /
     broken (rejected); taller than 300 blocks -> scaled to fit, flagged
  -> clean target: voxels, ground plate and site removed, largest part, surface repeats
     removed, colours seen from outside, builder palette, both roof styles
     gates: degenerate_size, scattered, open_model (floor plans, sets, cutaways)
  -> TRELLIS-like degradation -> input voxels in the target's frame
  -> validation (labels, frame, margins, .schem round trip)
  -> pair on disk + review renders + flags; raw GLB deleted
```

Every rejection is a reason code (`synth/filters.py: REASONS`) recorded in the asset's
`manifest.json` (`{"status": "rejected", "stage": ..., "reason": ..., "detail": ...}`) and in
the run's `ledger.jsonl`. **Size and scale are not reasons to reject a building** (since
synth-2.0.0, 2026-09-27): a small building, an uncertain scale and a building too tall for
the world are kept, and the pair's `flags` say so (`scale_uncertain`, `height_capped`,
`small_building`) so a training run can weight or hold them out. Only what is not one
building is rejected.

## Running it

```powershell
.venv\Scripts\python -m pip install -r requirements-synth.txt       # mcvoxel, scikit-image

# once: index Objaverse from metadata (no meshes are downloaded)
.venv\Scripts\python -m synth.objaverse_index --download

# review the small follow-up (selection and configuration already saved)
.venv\Scripts\python -m synth.report data/synthetic/pilot-v4
.venv\Scripts\python -m synth.review data/synthetic/pilot-v4 --compare data/synthetic/pilot-v2   # :5057

# what a pipeline change did, source by source (same selection in both runs)
.venv\Scripts\python -m synth.compare_runs data/synthetic/pilot-v2 data/synthetic/pilot-v4

# replay the saved settings and seed; counts include removed geometry outside the crop
.venv\Scripts\python -m synth.voxel_debug <uid> --run data/synthetic/pilot-v4 --recorded

# the held-out real evaluation set, and roof construction measured on real builds
.venv\Scripts\python -m synth.real_eval
.venv\Scripts\python -m synth.roof_stats real [--exclude-eval]

# reuse actual TRELLIS outputs already on disk; no server or new inference needed
.venv\Scripts\python -m synth.trellis_compare compare data/synthetic/pilot-v4 --trellis-dir data/synthetic/pilot-v2/trellis_compare
```

`synth.run` is resumable (finished assets are skipped; `--force` redoes them) and streams:
each raw GLB lives in `tmp_dir/<run>/` only while it is processed. `--raw-cache DIR` reads
GLBs already on disk instead of downloading (those are never deleted).

## Configuration

Every knob is a field of `synth/config.py: PipelineConfig`, saved as `run_config.json` in the
run folder; every manifest carries the config's hash. V1 defaults:

| Setting | Default | Why |
| --- | --- | --- |
| `licences` | CC0, CC BY | publishable and forkable; see the licence register |
| `exclude_photogrammetry` | on | scans are a different kind of mess from TRELLIS output |
| `blocks_per_metre` | 1.0 (+-5% jitter) | 1:1 with Steve; proportions survive |
| `max_height_blocks` | 300 | taller buildings are scaled down to fit and flagged `height_capped` |
| `small_building_blocks` | 16 | footprint and height both below: flagged `small_building`, kept |
| `min_representable_blocks` | 3 | below this wide or tall nothing of a building is left (`degenerate_size`) |
| `max_footprint_blocks`, `max_site_blocks` | 512, 1200 | engineering bounds on one pair (`too_large`) |
| `drop_clutter` | on | vegetation, vehicles, figures by part / material name (`mcvoxel.scene.split_clutter`) |
| `site_height_m`, `site_margin` | 2.2 m, 2 | columns lower than a storey outside the building's mass are site |
| `close_gaps` | off | binary closing filled column gaps and porches on clean models; inputs keep it |
| `color_mode` | exposed | block colour as seen from outside (`mcvoxel.surface_color`) |
| `palette` | builder | 72 building materials: no wool, soil, snow, ore or nether-wood blocks |
| `quantize` | k 32, merge 8, absorb 10 | gentler region merging than the TRELLIS defaults: clean models have real detail |
| `gates` | scattered 0.6; open 0.2 / 0.6 | see "Target" below |
| `roof_style` | builder | primary target; the geometric one is stored beside it |
| `shadow_repaint` | off | the TRELLIS shadow fix recolours real dark materials on clean models |
| `tmp_dir`, `keep_raw` | `data/synthetic/tmp`, off | raw downloads are deleted after processing |
| `degrade` | conservative (2026-09-27) | the input must still clearly be the same building; checked against real TRELLIS |
| `windows` | 64, 96, 128 | reported per pair; the grids themselves are stored at natural size |

## Stages

**Metadata filters** (`synth/select.py`, `synth/filters.py: metadata_verdict`). The Objaverse
index classifies every candidate from its render caption into a category (house, apartment,
skyscraper, religious, tower, ... 14 of them, plus institutional subcategories that split civic
buildings out) and flags objects, interiors, scenes, toys, ruins, game rips, stylised and
animated assets. Selection applies the licence, those flags, photogrammetry (Objaverse++
"scanned", scan vocabulary) and size limits, then fills per-bucket quotas so no category
dominates, capping each author at two assets.

**Classifier hook** (`filters.CLASSIFIERS`). Empty in V1. A building/non-building classifier
(for example zero-shot on the Sketchfab thumbnail) registers a callable `(record) -> (keep,
score)`; its rejections are recorded as `classifier_rejected`. TODO.

**Geometry checks** (`filters.geometry_verdict`). On the loaded mesh, before voxelising:
`blocky_geometry` (vertices on a 1/16-block lattice: a Minecraft export or voxel art) and
`scan_geometry` (a dense mesh whose area is not concentrated on a few flat directions;
calibrated on the 200 pilot-v1 models).

**Scale** (`synth/scale.py`). Real height in metres, in three classes:

* *confident* -- the file's units (m, cm, mm, in, ft) agree with the caption's storey count
  (high; medium when the file is 1.5-1.75x taller, which a plinth, plot or tall roof explains),
  or metres give a height typical of the category, or only the storey count is usable (medium);
* *uncertain* -- metres are unusual for the category, another unit had to be assumed, or the
  model is unitless and gets its category's typical height (low): kept, `scale_uncertain`;
* *broken* -- no reading of the file's numbers is a size (zero, infinite, under 1e-5 or over
  1e7 units): `scale_broken`, the only scale rejection.

Category height ranges are evidence for confidence, never bounds: pilot v2 read a 26 m estate
as feet because 26 m exceeded the "house" range, and rejected a 9.4 m two-storey house as
too small. A building taller than `max_height_blocks` is scaled down to fit (`height_capped`).

**Target** (`synth/target.py`, `mcvoxel.grid`, `mcvoxel.roofs`, `mcvoxel.assign`), stage by
stage (`synth.voxel_debug` draws each one):

1. conservative surface voxels (gap-free), exterior flood fill, enclosed rooms (so
   open-bottomed and glassless models still have an inside);
2. the ground plate (lawn, map tile) removed, then the **site**: every column outside the
   building's mass (columns at least a storey tall, in masses at least 3 wide) plus a 2-block
   margin -- lawns, pools, paths, boundary walls, fences, lamp posts, parked cars;
3. the largest part kept (no closing: on clean models it filled the gaps between columns);
4. slab / stair shapes from a 4x finer grid, and **surface repeats** removed: a voxel whose
   content lies against the face of a neighbour that already represents the surface (a wall
   straddling a voxel boundary would otherwise be two voxels thick); thin elements that
   stand clear -- parapets, porch posts, balcony slabs, fins -- are kept;
5. colours **as seen from outside** (rays through each air-facing face, facades before roofs)
   and glass / door kinds from the same hits; the builder palette; builder roofs coloured by
   the roof's median colour.

Gates reject only what is not one building: `degenerate_size` (under 3 blocks), `scattered`
(the largest part keeps under 60% of the model: a scene), and `open_model` (interior share
under 0.2 of a closed box's AND a roof over under 60% of the footprint: floor plans, interior
sets, cutaways, roofless shells; a pavilion or market hall has a roof and passes). Supervision
is the visible shell and the outside air; the interior is labelled "ignore" (Minecraft builds
are hollow and a photo never shows the inside). Three blocks of supervised air surround the
build.

**Roofs: two styles, kept side by side.** *Geometric* (Approach A) classifies each voxel from
its sub-voxel occupancy. *Builder* (Approach B, primary) reads the roof off the mesh, classifies
each column's slope and rebuilds the caps the way players do, measured on the real library:
up-slope stair courses on pitched roofs (97-99% stairs in real builds), slab roofs on low
pitches, one level of full blocks on flat roofs, one material per roof. Both are written
(`target.schem` / `target_geometric.schem`, and both label sets in `target.npz`) so models can
be trained and evaluated on either.

**Degradation** (`synth/degrade.py`). The clean model through TRELLIS's own bottleneck
(voxelised at 128-192 cells, blurred, re-meshed), light surface noise, few small holes, few
floaters, baked darker colour with a random sun, slight hue drift, and a partly hallucinated
back side. Toned down on 2026-09-27: pilot-v2 inputs were often barely recognisable.
Parameters are drawn per pair from `cfg.degrade` and recorded. The input is voxelised in the
target's frame (with closing: it is TRELLIS-like, noisy).

**Validation** (`synth/validate.py`). Labels in range; air and ignore consistent between shape
and material; no ignored voxel visible from outside; the outer layer is air; stairs and slabs
only on materials that have them; glass only as full blocks; input and target in one frame;
input/target overlap above a floor; the `.schem` reads back block for block.

## Outputs

```
data/synthetic/<run>/
  run_config.json          the configuration (+ pipeline version, config hash)
  selection.jsonl          chosen assets;   selection_report.json  metadata-stage funnel
  ledger.jsonl             one line per finished asset: status, stage, reason, bytes, seconds
  samples/<uid>/
    target.schem           primary target, Sponge v2, DataVersion 3465, provenance in Metadata
    target_geometric.schem the other roof style
    target.npz             shape (uint8) / material (uint16), Sponge (Y, Z, X);
                           255 / 65535 = interior (ignored), 0 = air; *_geometric for style A
    input.npz              occ, surf (uint8), rgb (Y, Z, X, 3)
    manifest.json          provenance, licence, attribution, every parameter, QA, metrics
    *.png, review.png      review panels
  report.json, report.md   statistics (synth.report);  sheets/<group>.png  contact sheets
  review.json              human verdicts (synth.review)
  trellis_compare/         the real-TRELLIS experiment
```

## Source quality

**Why unrealistic, stylised, game-like and incomplete models get through.** Objaverse 1.0 is
a crawl of Sketchfab: its architecture is mostly hobbyist, student and game-asset uploads,
and the metadata cannot tell them from real architecture. Measured on the 195 hand-graded
pilot-v1 sources (c = clean target, f = flawed, j = junk; `python -m
synth.source_quality`, 2026-09-27):

| Signal | Split | Clean | Junk |
| --- | --- | ---: | ---: |
| Objaverse++ style | "realistic" (181 of 195) | 53% | 23% |
| | "scanned" (11) | 27% | 55% |
| Objaverse++ quality | 2 (10) / 3 (185) | 60% / 51% | 30% / 25% |
| TRELLIS-500K aesthetic | all 195 are >= 5.5 | -- | -- |
| low-poly / stylised words | present (26) / absent (169) | 65% / 50% | 23% / 25% |
| game / asset words | present (33) / absent (162) | 55% / 51% | 33% / 23% |
| faces | < 5k / >= 100k | 67% / 30% | 19% / 30% |
| Objaverse++ scene flag | yes (127) / no (68) | 47% / 60% | 28% / 19% |

The labels are too coarse (93% "realistic", every source quality 2-3), and vocabulary points
the wrong way: low-poly models make CLEANER targets (simple, closed geometry). Excluding
low-poly, stylised and game vocabulary (with the index's low-poly flag) would have dropped 34
clean targets (and 8 flawed) to remove 17 junk ones, and
it would reject buildings like "Hello Neighbor Players House" that are fine at 1:1. So no
metadata realism filter was added; photogrammetry stays excluded (scans: 55% junk).

**What does work** is judging the geometry after voxelisation: `open_model` (floor plans,
interior sets, cutaways), `scattered` (scenes), site and clutter removal (plots, gardens,
cars, trees), and a human verdict with problem tags (`synth.review`: "unrealistic source" is
the only realism label we have; pilot v3 is the first run reviewed with tags).

**Recommendation: supplement and curate; do not scale the unreviewed pool.** Objaverse is
a useful collection of textured building candidates with per-model licences, and it has
the diversity (churches, towers, stations, pavilions, castles). What it lacks -- realistic
proportions and a trustworthy real size -- is exactly what public 3D city models have, all
real buildings at survey accuracy (licences checked 2026-09-27, before any download):

| Source | Licence | What it is | Use |
| --- | --- | --- | --- |
| 3DBAG (Netherlands) | CC BY 4.0, credit "© 3DBAG by tudelft3d and 3DGI" | all ~10M Dutch buildings, LoD1.2-2.2 (roof shapes), untextured | shape-only pairs: real massing and roofs at true scale |
| PLATEAU (Japan, MLIT) | per-dataset open licence; see the licence register | city models with multiple levels of detail, a textured subset | massing, roofs, some facade texture |
| Helsinki 3D | CC BY 4.0 | LoD2 CityGML, textured (the reality mesh is photogrammetry: excluded in V1) | textured LoD2 |
| swissBUILDINGS3D 3.0 beta | Swiss open government data, "open use, must provide the source" (confirm on swisstopo before use) | whole country, roofs with overhangs, untextured | shape-only pairs, roofs |

Their limits: LoD2 is walls and roofs only (no windows, balconies, doors), heavy on houses and
blocks, and textures (where present) are aerial imagery. They would make the model's massing,
roof and scale realistic; facade detail still comes from Objaverse. The next lever on
Objaverse itself is an image-based realism classifier (for example zero-shot CLIP on the
source render) in the `CLASSIFIERS` hook, validated against the pilot-v3 review tags rather
than against metadata. Neither is part of V1; both are small, separable pilots.

## Adding a data source

A source plugs in as index records with the same fields (`uid`, `license`, `category`,
`path`...) and a fetcher. The Berlin LoD2 city model is the planned second source for
apartments, offices and high-rises at real scale: untextured, so its pairs would train the
shape head only (material loss masked). Not a V1 dependency; the CityGML reader is TODO.

## Known limitations (V1)

* Glass is only detected from material names and alpha, so curtain walls with opaque dark
  textures become dark stone.
* At 1 block = 1 m, porch posts about 2 m apart leave one-voxel gaps that the enclosed-room
  fill closes; 3 m spacing keeps the porch open. Detail under ~0.5 m rounds away.
* Saturated colours have no stair / slab block of the same hue (vanilla has no bright red
  stairs): a bright red roof becomes the nearest stair material (orange-red).
* Doors are recorded as a material kind and built from planks; door blocks are TODO.
* Scale from a category prior is plausible, not true (a landmark tower got 180 m).
* Paired TRELLIS comparisons now exist (120 historical pairs and the current pilot subset).
  Their input images are source-model renders, not photographs. Target-informed height and
  rotation alignment make their IoU optimistic. These are calibration diagnostics, not a
  held-out validation of the eventual photo pipeline; degradation settings remain provisional.


## Builder roof and detail update (synth-2.2.0 / mcvox-v2)

See [the builder roof quality report](builder-roof-quality.md) for the September 30
comparison, exact commands, test results and remaining limitations. Builder targets
now use connected plane lattices at canonical pitches, explicit seams and exported
backing. The geometric target remains available. Narrow anchored posts and explicitly
named railings/parapets can become fences or walls when their materials agree.

Vocabulary v2 appends fence (8) and wall (9); old IDs and ignore/air contracts are
unchanged. New targets require 10 shape outputs. Connections and stair corners are
derived after assembling the whole grid. Writers validate states for DataVersion 3465
and read schematics back. Manifests record region pitch, pattern, direction, phase,
error, repairs, shape usage and topology indicators; `roof_regions.png` is available
in the review UI with fence/wall legend entries.

The new comparison uses the earlier seeds, source files and scale settings. Earlier
pilots are preserved. Frame validation applies to each input/target pair; an earlier
run may have a different crop origin, so comparison grids are aligned by their saved
origins rather than by assuming array indices match.

The older V1 limitation about enclosed-room fill now has a targeted correction:
isolated columns do not count as broad walls for horizontal rays. Attached and
ambiguous structures can still fool that heuristic. Thin textured facade features,
site podiums and material ambiguities remain audit items, not invented architecture.
