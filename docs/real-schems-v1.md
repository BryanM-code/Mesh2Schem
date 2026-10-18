# real-schems-v1: the frozen schematic dataset

The training targets of [plan version 4](mesh-to-schem-plan.md#version-4-2026-10-11-schematic-first-pivot):
321 human-built realistic Minecraft buildings from Planet Minecraft (235) and Abfielder (86),
frozen on 2026-10-11 (Phase 2). Licensing: D14 in
[dataset-licensing.md](dataset-licensing.md#schematic-dataset-2026-10). The data is local
only (`data/` is gitignored; the repository is public).

```powershell
.venv\Scripts\python -m schemset.freeze --source "<collection folder>" --out data/real-schems-v1
```

Re-running without `--force` never touches `source/`: it re-verifies the frozen copy against
`freeze_checksums.sha256` and regenerates the derived files (byte-identical for the same seed).

## Folder layout

```
data/real-schems-v1/
  source/                   verbatim copy of the collection folder, read-only (attrib +R). Never edit.
    manifest.json           authoritative list, 321 entries
    checksums.sha256        the collector's hashes of the 321 structure files
    <category>/<sub>/<id>/  structure file, thumbnail, source preview, listing evidence
    safety_holdout/, legacy_holdout/   12 command-block and 4 original holdouts: not in the manifest, not indexed
    REPORT.md, verification.json, rejection_log.json, deferred_candidates.json, ...
  freeze_checksums.sha256   sha256 of all 1,792 files under source/
  FREEZE.md                 freeze record: date, source path, counts, verification
  index.json                one row per manifest entry
  near_duplicates.json      metadata near-duplicate candidates and review lists
  splits.json               seed, method, counts, creator links and groups, id -> split
  normalized/               Phase 3: <id>.npz per build, records/<id>.json, index.json
  vocab_coverage.json       Phase 3: vocabulary coverage (docs/vocab-coverage.md)
  review/                   Phase 3: before/after crop sheets, interior cutaways, sheets.json
```

The original collection folder is
`C:\Users\mendo\Documents\Codex\2026-10-05\find-and-download-30-high-quality-2\outputs\Real2Minecraft_Dataset\`;
nothing is ever written there.

**Freeze verification (2026-10-11):** all 321 lines of the collector's `checksums.sha256`,
all 321 manifest `sha256` values, and all 1,792 copied files (byte-identical to their
originals) matched. 169,162,848 bytes.

## index.json

`{"dataset", "rows_count", "tier_mapping", "flags", "fields", "rows": [...]}`, rows in
manifest order.

| Field | Meaning |
| --- | --- |
| `id`, `file`, `sha256`, `file_type`, `building_category`, `creator`, `source_website`, `license_status`, `dimensions_xyz_blocks`, `non_air_blocks_from_metadata`, `listing_completion_percent`, `source_url`, `build_name` | passed through from the manifest (`file` is relative to `source/`) |
| `category_top` | first part of `building_category` (residential, civic, ...) |
| `creator_norm` | `creator` lowercased, whitespace and punctuation removed |
| `licence_tier` | A / B / C / X from `license_status` (below) |
| `flags` | list, see below; nothing is dropped in Phase 2 |
| `split` | `train` / `val` / `test` / `excluded` |
| `creator_group` | the union of creators that must share a split (below) |

## Licence tiers (D14)

| `license_status` (exact) | Tier | Builds |
| --- | --- | ---: |
| stated reuse permission; scope limited | A | 43 |
| license unclear | B | 265 |
| restricted personal-use license | C | 6 |
| explicit noncommercial reuse permission | C | 3 |
| explicit license: CC BY-NC-SA 4.0 | C | 2 |
| explicit license: CC BY-NC-ND 4.0 (Costanera) | X | 1 |
| explicit terms: use with credit, no redistribution (Xorph barn) | X | 1 |

Any other value stops the run (`UnknownLicence`).

## Flags

| Flag | Rule | Builds |
| --- | --- | ---: |
| `tier_x` | licence tier X | 2 |
| `incomplete_listing` | `listing_completion_percent` present and < 100 (60, 90 or 95) | 8 |
| `oversized_capture` | X or Z > 200 blocks; crop in Phase 3 | 12 |
| `legacy_format` | `.schematic` (pre-1.13 block IDs) | 142 |
| `near_dup_candidate` | a near-duplicate candidate pair (below) | 0 |

## Near-duplicates (metadata only)

The collector found 0 duplicate file hashes and 0 duplicate same-format block payloads. It
did not test rotations, crops or cross-format copies, and schematics are not parsed until
Phase 3, so this pass uses manifest metadata:

| Rule | Result |
| --- | --- |
| Same `creator_norm`, same dims allowing a 90° turn (sorted (X, Z) and Y), non-air within 3% | **0** pairs |
| The same test across different creators (re-uploads) | **0** pairs |
| Different creators, normalised `build_name` ratio ≥ 0.9, dims within 10% | **0** pairs |
| Different creators, name ratio ≥ 0.9, dims differ (listed, not flagged) | 16 pairs, all generic names: "Warehouse" ×4, "Red Barn" ×3, "Office Building", "Modern House", "Semi-detached house", "Small Victorian House", "Victorian House" / "Victorian House 5", "Farmhouse" / "Farmhouse 2", "Huge White House" / "The White House" |
| Same creator, dims and non-air within 10% (listed for a Phase 3 geometry check) | 2 pairs: yamantaca9's Mine County "Upscale Suburban Home" / "Dormer Windowed Bungalow" (64×32×41 vs 64×31×41, non-air 0.3% apart; both 60% listings), DownTheBlockHomes' "Medium Modern House" / "Modern Duplex House" (21×14×23 vs 21×15×21, one non-air count missing) |

141 entries have no `non_air_blocks_from_metadata`; for them the non-air test cannot rule a
pair out. The geometric duplicate check (rotation, crop, cross-format) moves to Phase 3.

## Splits (D17)

Tier X is `excluded`. Every other build is grouped by `creator_norm`, and groups are joined
(union-find) by creator links and near-duplicate candidates. Groups are placed largest
first (ties in a seeded shuffle, seed **20261011**) into the split that most reduces the
squared shortfall against 80/10/10 of the total and of each top-level category, each term
divided by its target. Deterministic; the method and seed are recorded in `splits.json`.

Creator links (each joins two `creator_norm` values into one group):

| Link | Creators |
| --- | --- |
| text before `;` is the creator, the rest are credits | TheAvatar, Aeroteam / ElOttomanMarshal, CC4142, MC_Nightmare, Wedgenet, 3D (each with its "; credit ..." variant) |
| near-identical names (ratio 0.95) | "CM Creation Manufactory" / "Creation Manufactory" |
| shared `nbt_author_or_exporter` | none (the Abfielder exporter "abfielder" is ignored) |

No raw creator names collapse to the same `creator_norm`. 33 builds record an NBT author
that differs from the creator (handles like MsBean / xMsBean, MaximV12 / MaximeV, or the
builder behind a publisher such as SchematicWorld), and 13 more were exported as
"abfielder". No NBT author appears under two creators, so none joins two groups. Not linked: filodema01's hangar credits Aeroteam for its aircraft.

319 builds form 216 creator groups (167 single builds; largest 7).

| Split | Builds | Creator groups | agricultural | civic | commercial | highrise | hospitality | industrial | residential |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 255 | 177 | 8 | 47 | 39 | 13 | 12 | 17 | 119 |
| val | 32 | 23 | 1 | 6 | 5 | 2 | 2 | 2 | 14 |
| test | 32 | 23 | 1 | 6 | 5 | 2 | 1 | 2 | 15 |
| excluded | 2 | 2 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |

| Split | A | B | C | X |
| --- | ---: | ---: | ---: | ---: |
| train | 32 | 216 | 7 | 0 |
| val | 9 | 19 | 4 | 0 |
| test | 2 | 30 | 0 | 0 |
| excluded | 0 | 0 | 0 | 2 |

Tiers are not stratified: test has no tier C build. All 12 `oversized_capture` builds are
in train; 5 of the 8 `incomplete_listing` builds are in test.

The tables are the seeded split. Two decisions are applied after it (`splits.json`,
`overrides`), so no other build moves: D24 set the two landscape dioramas aside (from
train), D25 the Richelieu cliff house (from test). Current: **train 253 / val 32 / test 31 /
excluded 5**.

The old `eval/real-eval-v1.json` stays on disk for its tests but is superseded by this
split (D17, D19).

## Normalization (Phase 3)

```powershell
.venv\Scripts\python -m schemset.normalize --dataset data/real-schems-v1 [--only <id> ...] [--workers 4] [--force] [--no-review]
```

Every non-excluded build is read, cropped to the building, masked, measured and encoded as
training targets: 319 in Phase 3, 317 since D24 set two landscape dioramas aside, 316 since
D25 set the Richelieu cliff house aside (2026-10-17; it was test) (Phase 3
fixes, 2026-10-12). The run is resumable and deterministic:

* A build is skipped when its record matches its source sha256, the pipeline version and the
  vocabulary (version and table digest).
* The npz files are byte-identical on a re-run.
* It never writes inside `source/`.
* A full run takes about 2 minutes with 4 workers.

Code: `schemset/read.py`, `crop.py`, `interior.py`, `storey.py`, `coverage.py`, `normalize.py`;
the duplicate check is `dedupe.py`, the review gallery `gallery.py` (with `review.py`).
Vocabulary coverage has its own page: [vocab-coverage.md](vocab-coverage.md).

### Outputs

| File | Content |
| --- | --- |
| `normalized/<id>.npz` | `blocks` (uint16, X Y Z, Y up, +X east, +Z south), `palette` (blockstate strings, `palette[0]` = air), `shell` / `exterior` / `interior` (bool, an exact partition of the grid), `crop_offset` (source-grid coordinates of [0,0,0]), `ground_height` (int16 X Z: each column's ground height in this grid, at or below it is ground, -1 none), `ground_y` (its median), `target_shape` (uint8) / `target_material` (uint16): the training targets in `mcvox-v3` |
| `normalized/records/<id>.json` | everything measured for one build: read, validation, crop, masks, storey, supervised block counts per blockstate, `targets` (vocabulary version and digest, supervised and ignored counts, shape counts) |
| `normalized/index.json` | per build: status (ok / failed + reason), validation, crop stats, flags, interior fraction, exposure, storey; plus a dataset summary |
| `normalized/previous_run.json` | the previous pipeline version's per-build numbers, kept when the version changes |
| `geometric_duplicates.json` | `schemset.dedupe`: duplicate pairs and near misses (read by `freeze.derive`) |
| `review/gallery_v2/` | `schemset.gallery`: the review gallery (`index.html`, opens locally) |

**Training targets** come from `mcvoxel.vocab.encode_target` (D20):

* Exterior air is air.
* Shell blocks the vocabulary holds get their (shape, material).
* Everything else is IGNORE (shape 255, material 65535), never air: the interior, decor,
  fluids, snow layers, unsupported blocks and door upper halves.

1.71% of shell blocks are IGNORE (46,691): 40,659 outside the vocabulary and 6,032 door upper
halves. Every non-IGNORE target pair is in `VALID`.

### Reading and validation

| | Result |
| --- | --- |
| Parsed | **319 / 319**, 0 failures |
| Formats | legacy MCEdit 140, Sponge v2 83, Sponge v3 29, litematic v5 1 / v6 14 / v7 52 (all single-region in this dataset; multi-region is supported and tested) |
| Dimensions vs index.json | 319 match exactly; 0 X/Z swaps; 0 mismatches |
| Non-air count vs `non_air_blocks_from_metadata` | 178 match exactly; 141 have no metadata count; 0 mismatches |
| Unknown legacy ids | 0 (they would be kept as solid `legacy:unknown_<id>` blocks and counted, never air) |
| Non-minecraft namespaces | 0 |

Reading details:
- Litematics are read by our own nbtlib reader, because litemapy is GPL.
- Legacy files are converted in memory with `schem_convert`'s steps.
- Palettes are modernised:
  - legacy aliases are folded
  - 1.13 names are updated (`sign` → `oak_sign`, `grass_path` → `dirt_path`)
  - `chain` → `iron_chain`, its 1.21.9 name (D21)
  - `stone_slab` → `smooth_stone_slab` for files older than DataVersion 1901
  - cave and void air become air
- One Sponge v2 file has no DataVersion.
- Only blocks are read; entities (paintings, item frames, armour stands) are not. The legacy
  reader counts them: 58 legacy builds carry some.

### Crop (D23)

`schemset/crop.py` keeps the building and removes landscape and site clutter. Block names are
only a hint and geometry decides (D23). Builders use stone, cobblestone, sandstone, andesite,
gravel, clay, snow blocks, tuff and ores for walls, roofs and foundations, so a name never
makes a block landscape. When a rule is unsure, the block stays: a kept stray block is
cheaper than a hole in a building. The module docstring has every rule and threshold.

1. **Natural terrain.** In each column, the *ground run* is the unbroken run of grass, dirt,
   sand, gravel, stone, water and the like that starts at the bottom of the capture.
   - Each layer of the runs is opened with a 3×3 square, so walls, pillars and chimneys
     standing on the ground drop out while hills and plates stay.
   - A region is terrain only if it reaches the edge of the build and shows natural
     evidence: at least 30% of its surface is soft ground (grass, dirt, sand, gravel, snow,
     water), or it is a rock mass at least 3 deep.
   - A paved plaza or a stone platform is neither, so it is not terrain.
2. **Ground height, per column.**
   - Open terrain columns (no building above) give the terrain surface, which is carried
     under the building from the nearest open terrain.
   - Under a building the terrain never rises above that height, so a stone floor or a
     natural-stone podium on the ground is building.
   - If every terrain column has building above it, only soft ground counts as terrain.
   - With no natural terrain around the building there is no ground and nothing is below
     ground. For the flood only, a virtual plane under the lowest kept block stands in, so a
     building whose floor was the world's ground does not flood from below.
3. **Trees** are groups of logs that touch leaves and stand on the ground; their leaves go
   with them. Logs inside a building that touch an ivy bush are building.
4. **Footprint.**
   - Building cores are columns whose building material rises at least 4 blocks above the
     local ground, in masses at least 3 wide.
   - A core mass at most 4 wide, at least 10 long and at most 6 tall is a wall or fence line
     (a cemetery wall with its coping), not a building. A taller narrow mass is a building
     wing and stays.
   - Every building material within 2 columns of a building core stays.
   - Material attached to it stays up to 6 columns out: steps, stoops, porches, chimneys,
     short garden walls.
   - Beyond that, site structures go: boundary walls, fences, paths, roads, plazas, cars,
     lamp posts.
   - Every separate building of a compound stays.
5. **Soft blocks** (plants, turf, water that are not terrain) stay next to the building
   unless they stand on open ground: terrain, the ground, removed site material or the
   bottom of the capture, with nothing kept overhead. Green roofs, window boxes, ivy and
   pools stay; lawns, hedges and bushes go.
6. **Below ground.** Kept blocks below their column's ground height are buried: masked
   later, never deleted. If more than 15% of the kept blocks would be buried, the ground is
   not trusted and there is none (`ground_fallback`).
7. **Terrain floors.** Where an enclosed room stands directly on the terrain (the builder
   used the world's ground as its floor), the top terrain block under it is kept as the
   room's floor. It is interior, so it is ignored like any floor.

| | Result (317 builds) |
| --- | --- |
| Non-air blocks removed | 4.61 M of 13.39 M (34.5%). By role (before 35k blocks were restored as terrain floors): turf 2.15 M, natural minerals 1.61 M, plants and snow cover 0.51 M, built (site) 0.27 M, trees 0.04 M, water 0.04 M, invisible 0.02 M |
| Removed per build | median 17%; p25 1%, p75 45%, p90 65%, max 97% |
| Ground | natural terrain in 185 builds; none (virtual plane) in 128; `ground_fallback` in 4 |
| Terrain floors restored | 136 builds, 78,802 blocks (normalize-4) |
| Kept blocks below the ground | in 76 builds; below-ground ignored per build: p90 1.3%, max 14.4% |
| Largest side after the crop | median 57 blocks, p75 87, max 350 |

### Interior mask (D22)

`schemset/interior.py` decides exterior by the size of the openings. Being covered from above
never makes air interior on its own.

1. Voxels at or below each column's ground height count as solid for the flood.
2. The air is eroded by a 3 × 2 × 3 box (3 wide, 2 tall), flooded from the grid's boundary
   and grown back. The outside therefore reaches through every opening at least 3 wide and
   2 tall: porches, recessed porches, verandas, balconies, loggias, arcades, colonnades,
   carports, canopies, eaves, soffits, gable undersides, jettied overhangs.
3. Air within 1 block of that exterior is exterior too, through any opening: window and door
   reveals, thresholds.
4. A leftover pocket in which at least 75% of the voxels see the outside along a straight
   horizontal line is an open gap, not a room, and is exterior. An example is the strip
   between a garden wall and the house under the eaves.
5. **Depth under cover** (D22 amended, 2026-10-14; measured from open air since normalize-5,
   2026-10-15). Exterior air more than 6 steps from *open* exterior air becomes interior.
   Steps are 6-connected and go through exterior air only. Open air is uncovered air (no
   block anywhere above it in its column; the grid's boundary counts too) and covered air
   that is open to the side and to the sky, such as the strip under a deep, tall eave. See
   [Depth cap under cover](#depth-cap-under-cover-d22-amended-2026-10-14) and
   [Depth from open air](#depth-from-open-air-normalize-5-2026-10-15-the-hotel-piran-fix).
6. Shell is every block touching exterior air; interior is the rest, which is ignored and
   never deleted.

A room behind a doorway or window up to 2 wide stays interior, with or without glass. Being
covered still never makes air interior on its own; only depth under cover does.

`large_opening` flags builds where more than 30% of the *deep* covered air ends up exterior.
Deep covered air means air more than 3 blocks in from open sky; eaves and porches would
otherwise flag most houses. `deep_cover_capped` flags builds where the depth cap turned more
than 200 air voxels interior; the count is `deep_cover_capped` in the record's mask stats.

| | Result (normalize-5) |
| --- | --- |
| Interior fraction (interior / (interior + shell)) | median 0.83, p10 0.74, p90 0.90, min 0.30 |
| Outside-visible but ignored (see below) | median 0.1%, p90 2.4%, max 15.4% |
| ... with no depth limit (`visible_ignored_any_depth`) | median 0.6%, p90 6.4%, max 24.3% |
| `large_opening` | 21 builds |
| `deep_cover_capped` | 92 builds |
| `deep_visible_ignored` | 5 builds |

### Phase 3 fixes: before and after (2026-10-12)

The review of the first run found three problems:
- porches, overhangs and balconies were ignored;
- the ground level was wrong on several builds;
- the crop decided "landscape" partly by block name.

These measures compare that run with the fixed one on the same 317 builds. *Outside-visible
but ignored* counts blocks not in the shell that touch exterior air or open covered air
(covered air with a straight horizontal line of sight out; `interior.exposure`), as a share
of all such blocks. *Below ground* counts blocks below their column's ground height that are
ignored, as a share of kept blocks.

| Measure | Before: median / p90 | After: median / p90 |
| --- | ---: | ---: |
| Crop removed | 22.1% / 69.5% | 17.2% / 65.5% |
| Interior fraction | 0.86 / 0.92 | 0.81 / 0.88 |
| Outside-visible but ignored | 11.5% / 25.5% | 0.2% / 3.2% |
| Below ground | 0.0% / 3.5% | 0.0% / 1.3% |

* Outside-visible but ignored above 25%: 34 builds before, 0 after. Above 10%: 180 before, 6
  after. What remains is mostly rooms, furniture and restored terrain floors seen through a
  doorway or window, which the audit's line of sight counts and the opening rule keeps
  interior on purpose.
* Below ground above 15%: 15 builds before, 0 after. Above 5%: 29 before, 16 after.
* Supervised (shell) blocks: 2.16 M → 3.27 M (+52%).
  - The 248 builds without `large_opening` gained +32% (median +24% per build): recovered
    porches, overhangs and eaves, and the fixed ground bugs.
  - The 69 `large_opening` builds nearly doubled (+95%), because their hall interiors are now
    exterior.
* Vocabulary coverage of supervised blocks: 98.44% representable (96.09% exact), against
  98.51% before. The shell now holds more porch and eave detail.

Per-build tables, for the 17 builds named in the review and every build whose removed share
moved by more than 10 points or whose shell changed by more than 10% (256 builds), are in
[phase3-fixes.md](phase3-fixes.md).

### Depth cap under cover (D22 amended, 2026-10-14)

The opening rule overshot on open halls, hollow facade shells and open ground floors: the
outside flooded all the way in, so internal blocks were trained as facade. In the baroque
townhouse, for example, the diagonal support beams inside the shell were shell. The cap
keeps covered exterior air only within `COVER_DEPTH` steps of uncovered exterior air.

The audit (`interior.exposure`) uses the same cap: open covered air counts as outside-visible
only within `COVER_DEPTH` steps (through above-ground air) of uncovered air. Otherwise deep
halls would inflate visible-but-ignored.

**Choosing the depth.** Each depth was run on the whole dataset (crop unchanged):

| | No cap (normalize-3) | Depth 3 | Depth 4 | Depth 5 | **Depth 6** | Depth 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised (shell) blocks | 3,270,234 | 2,608,035 | 2,673,361 | 2,709,740 | **2,734,544** | 2,775,708 |
| Interior fraction p0 / p10 / p50 | 0.30 / 0.66 / 0.81 | 0.53 / 0.76 / 0.84 | 0.36 / 0.75 / 0.83 | 0.30 / 0.74 / 0.83 | **0.30 / 0.74 / 0.83** | 0.30 / 0.73 / 0.83 |
| `large_opening` | 69 | 0 | 3 | 6 | **11** | 19 |
| Visible but ignored p50 / p90 / max | 0.2 / 3.2 / 18.6% | 0.0 / 1.2 / 11.8% | 0.1 / 1.7 / 11.5% | 0.1 / 1.9 / 13.0% | **0.1 / 2.1 / 14.2%** | 0.1 / 2.5 / 16.5% |
| Builds over 200 voxels capped | – | 153 | 129 | 118 | **108** | 94 |
| Air turned interior by the cap | – | 3.47 M | 3.32 M | 3.21 M | **3.12 M** | 2.97 M |

Porch set, visible but ignored (%), with the capped audit:

| Build | No cap | 3 | 4 | 5 | **6** | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ab-6057 | 0.1 | 0.0 | 0.0 | 0.0 | **0.1** | 0.1 |
| pmc-triple-gable-tudor-townhouse-6884362 | 0.1 | 0.1 | 0.3 | 0.3 | **0.3** | 0.1 |
| pmc-cape-cod-house---furnished | 13.2 | 2.8 | 3.6 | 5.1 | **6.7** | 9.0 |
| pmc-small-craftsman-house---exterior-amp-landscape-only | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 |
| pmc-victorian-queen-anne-style---carson-mansion | 0.1 | 0.0 | 0.0 | 0.0 | **0.0** | 0.1 |
| pmc-spanish-colonial-house---philippines | 0.2 | 0.1 | 0.2 | 0.2 | **0.2** | 0.2 |
| pmc-greek-revival-plantation-house | 0.7 | 0.0 | 0.0 | 0.1 | **0.1** | 0.2 |

The same, with the audit left uncapped (what a straight line of sight from outside reaches,
whatever the depth):

| Build | No cap | 3 | 4 | 5 | **6** | 8 | What the cap removes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ab-6057 | 0.1 | 5.2 | 4.6 | 4.1 | **3.6** | 2.7 | a 2-tall crawl space under the whole hotel |
| pmc-triple-gable-tudor-townhouse-6884362 | 0.1 | 2.4 | 1.7 | 1.5 | **1.2** | 0.1 | an open ground-floor hall behind the courtyard arcade, 7–9 steps in |
| pmc-cape-cod-house---furnished | 13.2 | 17.6 | 13.4 | 13.2 | **13.2** | 13.2 | nothing from depth 5 on |
| pmc-small-craftsman-house---exterior-amp-landscape-only | 0.0 | 11.7 | 11.2 | 10.6 | **10.0** | 8.9 | 1 × 1 channels under the raised floor |
| pmc-victorian-queen-anne-style---carson-mansion | 0.1 | 3.2 | 1.7 | 1.2 | **0.7** | 0.1 | a ground-level veranda (16 × 17) at depth 4; small corners at 6 |
| pmc-spanish-colonial-house---philippines | 0.2 | 1.1 | 0.7 | 0.2 | **0.2** | 0.2 | nothing from depth 5 on |
| pmc-greek-revival-plantation-house | 0.7 | 21.8 | 18.8 | 15.7 | **12.4** | 7.4 | halls behind big openings (it was `large_opening`) |

Hollow / hall set, interior fraction:

| Build | No cap | 3 | 4 | 5 | **6** | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pmc-baroque-townhouse | 0.31 | 0.86 | 0.85 | 0.85 | **0.85** | 0.85 |
| pmc-brick-tenement-house | 0.30 | 0.56 | 0.36 | 0.30 | **0.30** | 0.30 |
| pmc-old-tenement-house | 0.39 | 0.80 | 0.80 | 0.80 | **0.79** | 0.78 |
| pmc-adelaide-hotel-toronto | 0.43 | 0.89 | 0.89 | 0.89 | **0.89** | 0.88 |
| pmc-warehouse-factory-hall | 0.44 | 0.88 | 0.87 | 0.87 | **0.87** | 0.87 |
| pmc-aircraft-hangar-private-jet-hangar-1-5-1-scale-1-1-scale-adaptable-full-interiors | 0.45 | 0.90 | 0.89 | 0.89 | **0.89** | 0.89 |
| ab-13526 | 0.48 | 0.88 | 0.87 | 0.86 | **0.85** | 0.83 |

**Choice: `COVER_DEPTH` = 6.**
- Taken literally, the rule "the smallest depth that keeps every porch-set build within
  +1 point of its visible-but-ignored %" picks 3: every porch build passes in the capped
  table.
- But the capped audit cannot see what the cap removes. Air the cap turns interior stops
  counting as visible in the same step, so the criterion cannot fail at any depth.
- The uncapped audit shows what each depth actually takes. Every loss was located:
  - Crawl spaces under the hotel and the craftsman house, the Tudor's open hall and the
    Greek Revival's halls are not porches; capping them is the point.
  - The one real porch loss is the Victorian's ground-level veranda. It is within +1 point
    only from depth 6 (+0.6; depth 5 is +1.1).
- The Spanish colonial and the Cape Cod lose nothing from depth 5.
- Depth 6 is therefore the smallest depth that keeps every porch, veranda and loggia in the
  set. The hall set is at 0.79–0.89 interior from depth 3 to 8 (except the brick tenement,
  below), so a deeper cap would buy nothing there.

**Before → after** (normalize-3 → normalize-4, 317 builds):

| | normalize-3 | normalize-4 |
| --- | ---: | ---: |
| Supervised (shell) blocks | 3,270,234 | 2,734,555 (-16%) |
| Interior fraction p0 / p10 / p50 | 0.30 / 0.66 / 0.81 | 0.30 / 0.74 / 0.83 |
| Visible but ignored p50 / p90 / max | 0.2 / 3.2 / 18.6% | 0.1 / 2.1 / 14.2% |
| Builds visible-but-ignored over 10% | 6 | 4 |
| `large_opening` | 69 | 11 |
| `deep_cover_capped` | – | 108 |
| Builds whose shell changed by more than 5% | | 100 |
| Terrain floors restored (step 7 of the crop: more hall air is interior now) | 130 builds, 35,385 blocks | 136 builds, 78,802 blocks |
| v3 coverage, representable | 98.44% | 98.51% |

- The hall set: baroque townhouse 0.31 → 0.85, old tenement 0.39 → 0.79, Adelaide hotel
  0.43 → 0.89, factory hall 0.44 → 0.87, hangar 0.45 → 0.89, ab-13526 0.48 → 0.85.
- The brick tenement stays at 0.30. The hollow behind its facade has no roof, so it is open
  sky and the cap rightly does not apply.
- The porch set: visible but ignored stays within +0.2 points (the Cape Cod falls from 13.2%
  to 6.7%, as deep sightlines into its rooms no longer count).
- Shell changes on the porch set: -0% (Spanish colonial, Cape Cod), -1.8% (Victorian),
  -3.6% (ab-6057), -6.9% (Tudor), -10% (craftsman), -22% (Greek Revival).
- The largest capped builds are halls and atria: the French fire station (397k voxels), the
  brewery, ab-11663, the luxury villa, the West Virginia capitol.
- Tall hotels and towers that the big-opening flood reached are capped too, though they were
  never `large_opening`. That is why 108 builds are flagged and only 69 were before.

### Depth from open air (normalize-5, 2026-10-15): the Hotel Piran fix

**Symptom.** Hotel Piran (`pmc-hotel-piran-piran-slovenia`, val) had lost three of its four
facades in its target, its synthetic inputs and both real-TRELLIS photos.

**Where the walls disappear**, stage by stage:

| Stage | Hotel Piran's facades |
| --- | --- |
| `source/` (Sponge v2, one region, 56 × 64 × 79) | present |
| `normalized/` blocks (the crop removed 5%, all landscape) | present |
| `normalized/` masks and target | **ignored**: the air in front of three facades is interior, so the walls are not shell (shell 8,998 blocks) |
| target render | missing |
| synthetic inputs (meshify, then degrade) | replaced by dark opening caps 3 blocks in front of the facades |
| smooth and blocky photos | the same caps |
| TRELLIS raw, voxelised, aligned | TRELLIS rebuilds the caps |

**Root cause.** The depth cap (normalize-4) measured depth under cover from *uncovered* air.
Hotel Piran's roof overhangs its facades by 9 blocks, 45 blocks up. The air in front of the
walls is covered (by the eave far above it) and 7 to 9 steps from uncovered air, so the cap
made it interior and the walls lost their shell. Depth from uncovered air cannot tell the
strip under a deep, tall overhang from the inside of a hall. The audit (`exposure`) used the
same depth, so it reported 0.02% visible-but-ignored for Hotel Piran and could not see the
loss.

**Not the cause** (checked on all 317 builds):
- Loading: every build has one region (no multi-region litematic). Dimensions match the
  metadata for all 317 and block counts wherever the metadata has them (177). There are no
  unknown legacy or non-vanilla blocks; the only warnings are "entities not read".
- Crop: Hotel Piran's bounding box is unchanged and the crop removed only landscape.
- Meshify: the blocks it does not draw are decor (grass, rails, buttons, torches, signs,
  vines, banners): 0.9% of all shell blocks, and no wall.

**Fix.** Depth is measured from *open* air (`interior.open_air`): uncovered air, plus
covered air that has a straight horizontal line out to the grid's boundary and either sees
the sky along a steep path (`sky_path`: rising at least `SKY_SLOPE` = 2 blocks per block
across) or lies above such air in its column. The sideways line separates an eave strip from
a hall, which is walled in at its own height. The steep path separates both from crawl
spaces and low canopies, which see the sky only at their edge. Next to a tall opening the
sky is now visible about half the opening's height in, and the cap counts from there.

Alternatives measured on the whole dataset before choosing:

| Open air also includes | Hotel Piran, visible but ignored (any depth) | Hall set | Verdict |
| --- | ---: | --- | --- |
| nothing (normalize-4) | 38.1% | 0.79–0.89 | the bug |
| air seeing the sky steeply (45° or 63°) | 3.0–5.7% | the cathedral ab-13526 0.85 → 0.55 | reopens tall halls lit through big arches |
| air with a horizontal line out | 0.6% | hangar 0.89 → 0.75, old tenement 0.79 → 0.62 | reopens every hall behind a big door |
| both, as whole vertical runs (half lit) | 3.6% | 0.79–0.88 | misses eaves less than 4× as tall as deep |
| both, lit air only (no 6 steps from it) | 6.1–12.5% | unchanged | leaves recesses behind the eave strip ignored |
| **both, up the column (chosen)** | **2.8%** | **0.77–0.88** | |

Before and after (normalize-4 → normalize-5, 317 builds):

| | normalize-4 | normalize-5 |
| --- | ---: | ---: |
| Supervised (shell) blocks | 2,734,555 | 2,807,900 (+2.7%) |
| Builds whose shell grew by more than 1% / 5% / 10% / 20% | | 99 / 34 / 14 / 6 |
| Builds whose shell shrank | | 0 |
| Visible but ignored, no depth limit: p50 / p90 / max | 1.25 / 9.0 / 38.1% | 0.6 / 6.4 / 24.3% |
| Interior fraction p10 / p50 / p90 | 0.74 / 0.83 / 0.90 | 0.74 / 0.83 / 0.90 |
| `deep_cover_capped` / `large_opening` | 108 / 11 | 92 / 21 |
| Terrain floors restored under rooms | 136 builds | 37 builds keep a little less (air under eaves is no longer a room) |

Hall set (interior fraction): baroque townhouse 0.85 → 0.85, brick tenement 0.30 → 0.30, old
tenement 0.79 → 0.77, Adelaide hotel 0.89 → 0.86, factory hall 0.87 → 0.86, hangar 0.89 →
0.88, cathedral ab-13526 0.85 → 0.77. The cathedral's outer walls under its projecting ring
were hidden the same way as Hotel Piran's (now shell), and the bottom of its 17-block base
arch is lit. Porch set: unchanged or better (Tudor any-depth visible-but-ignored 1.2 → 0.1%,
Greek Revival 13.0 → 3.9%).

Builds most affected (shell gain; visible but ignored with no depth limit, before → after):

| Build | Split | Shell | Visible but ignored |
| --- | --- | --- | --- |
| pmc-hotel-piran-piran-slovenia | val | 8,998 → 14,635 (+63%) | 38.1 → 2.8% |
| ab-11663 (a tower split by a covered cleft) | train | 42,167 → 60,940 (+45%) | 30.5 → 1.3% |
| pmc-french-embassy-in-korea-seoul (walls under a roof plate) | train | 4,774 → 6,302 (+32%) | 33.8 → 8.9% |
| ab-13526 (cathedral) | train | 11,639 → 14,055 (+21%) | 16.5 → 3.3% |
| ab-3677 | train | 11,009 → 13,263 (+20%) | 16.9 → 0.0% |
| pmc-villa-durango | train | 5,705 → 6,850 (+20%) | 13.9 → 7.0% |
| ab-13254 | train | 13,425 → 15,982 (+19%) | 9.7 → 3.1% |
| pmc-richelieu-contemporary | test | 3,136 → 3,708 (+18%) | 17.5 → 2.3% |
| pmc-forest-fire-lookout-tower (the open stair frame) | train | 1,492 → 1,731 (+16%) | 9.0 → 0.5% |
| pmc-greek-revival-plantation-house | train | 14,994 → 17,288 (+15%) | 13.0 → 3.9% |
| pmc-red-clay-barn | train | 2,166 → 2,445 (+13%) | 5.9 → 3.7% |
| pmc-warehouse-5752980 | train | 2,162 → 2,412 (+12%) | 3.3 → 2.0% |
| pmc-old-warehouse | train | 7,107 → 7,914 (+11%) | 18.2 → 9.7% |
| pmc-japanese-minimalistic-condo | train | 4,954 → 5,505 (+11%) | 9.4 → 1.0% |

The 20 more with 5–10%: pmc-city-hall-council-building-of-kaiserthal-kaiserthal-project-,
ab-8717, pmc-villa-shodhan-ahmedabad-by-le-corbusier, pmc-office-building-4034242,
pmc-oxford-place, pmc-modern-white-beachlake-house, ab-5911, ab-4273, pmc-japanese-dusan,
pmc-adelaide-hotel-toronto, pmc-traditional-mosque, pmc-modern-fuel-station-gas-station,
ab-13627, pmc-warehouses-6539779, pmc-unite-d-habitation-nantes-reze-by-le-corbusier,
pmc-late-1800s-brick-house, pmc-french-country-chateautmawok,
pmc-national-museum-of-western-art-tokyo-by-le-corbusier-5946126,
pmc-modern-american-craftsman-style-house, ab-3640. 65 more grew by 1–5%; 54 changed by
less than 1%. Of the 25 real-TRELLIS evaluation builds, 13 changed (Hotel Piran +63%, the
rest 0–5%); their photos were rendered again.

**The audit can now see this.** `compute_masks` also records visible-but-ignored with no
depth limit (`visible_ignored_any_depth`, every block a straight horizontal line from outside
reaches) and flags `deep_visible_ignored` above 15%. It would have flagged Hotel Piran at
38%. Five builds carry it now (ab-13627, pmc-japanese-dusan, pmc-farmhouse-free-download,
pmc-modern-police-department-station-2, the Larsen-Nielsen panel house). Their remaining
blocks sit 0–2 blocks above the ground (gravel, dirt, grass and planks under raised floors
and platforms): crawl spaces, which the cap ignores by design (as ab-6057's).

**Regression tests** (`tests/test_schemset_crop_rules.py`):
- `test_facades_under_a_deep_tall_eave_stay_shell` has a 30-tall block under a 9-block eave.
  Normalize-4 kept 0 of its 12 test facade cells as shell; normalize-5 keeps all 12.
- `test_deep_low_canopy_is_still_capped` checks that the wall under a 3-tall, 12-deep canopy
  stays ignored.
- `test_open_fronted_hall` now counts depth from the lit band inside the opening.

### Duplicates

`schemset.dedupe` compares every pair of normalized builds on their kept blocks:
- under the 8 rotations and mirrors about Y, with offsets up to 2 blocks;
- a duplicate needs occupancy IoU above 0.9 and palette similarity of at least 0.7;
- height and horizontal sizes must be within 4 blocks and block counts within 15% first.

127 of the 50,086 pairs passed those filters. None is a duplicate, and none reaches IoU
0.75. The closest are two Cape Cod houses by the same creator (IoU 0.60), variants rather
than copies. No split moves were needed. `freeze.derive` would move the smaller creator group
of any cross-split duplicate pair (`geometric_duplicates.json`).

### Storey height (information only, D15)

Nothing is filtered or rescaled on this (D15, revised 2026-10-12): buildings legitimately
vary in scale, sometimes on purpose. `schemset/storey.py` measures floor-to-floor height:
- Primary: floor planes inside the rooms. The ground counts as the first floor when the build
  has none of its own.
- Fallback: window-row spacing on the facade.
- Each estimate carries a confidence; "unknown" when nothing plausible (3–12 blocks) is found.

| Blocks per storey | high | medium | low | total |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 4 | 2 | 5 | 11 |
| 4 | 27 | 9 | 8 | 44 |
| 5 | 33 | 9 | 17 | 59 |
| 6 | 20 | 10 | 19 | 49 |
| 7 | 10 | 2 | 20 | 32 |
| 8 | 2 | 3 | 19 | 24 |
| 9 | 3 | 1 | 13 | 17 |
| 10 | 2 | 3 | 7 | 12 |
| 11 | 1 | 1 | 7 | 9 |
| 12 | 0 | 0 | 4 | 4 |
| unknown | | | | 58 |

Confidence: 115 high, 38 medium, 111 low (mostly one storey, measured floor to roof), 53
unknown (317 builds, after the Phase 3 fixes; the table above is from the first run).

### Flags (normalize-5, 316 builds since D25)

| Flag | Meaning | Builds |
| --- | --- | ---: |
| `deep_cover_capped` | the depth cap turned more than 200 air voxels interior: halls, hollow shells, open ground floors, atria; review | 91 |
| `large_opening` | more than 30% of the deep covered air is still exterior after the cap: canopies, station halls, barns, the roofless brick tenement shell, and since normalize-5 the strips under deep, tall overhangs (ab-11663, the French embassy, the cathedral ab-13526) | 20 |
| `deep_visible_ignored` | more than 15% of the blocks a straight horizontal line from outside reaches are ignored, at any depth: crawl spaces under raised floors | 5 |
| `crop_aggressive` | the crop removed more than 50% of non-air blocks, usually a big terrain capture | 64 |
| `multi_building` | more than one large separate mass remains (at least 15% of the biggest and 500 blocks): compounds, rows, campuses; all are kept | 10 |
| `crop_kept_landscape` | soft blocks (plants, turf, water) are over 10% of what the crop kept | 8 |
| `ground_fallback` | more than 15% of the kept blocks would be below ground, so there is no ground: houses built into a slope (the Richelieu cliff house, the fourth, is set aside: D25) | 3 |
| `site_linear_removed` | a wall or fence line was cut back to 6 columns from the building | 3 |
| `excluded_landscape_diorama` | D24: set aside (split "excluded"), not normalized | 2 |
| `excluded_target_lost_terrain` | D25: set aside (split "excluded"), not normalized: the Richelieu cliff house, whose rooms lose the rock that formed their back walls | 1 |
| `geometric_duplicate` | `schemset.dedupe` found the build in a duplicate pair | 0 |

`interior_leak_suspect` and `crop_no_core` are retired: the opening rule replaced the first,
and every build now has a building core.

The review gallery is `review/gallery_v2/index.html` (`python -m schemset.gallery`). Each build
gets one card:
- the source BEFORE the crop and every kept block AFTER it, each from two sides;
- an overlay toggle: red = outside-visible but not trained on, orange = below ground and
  ignored, pink = landscape (removed before, kept soft blocks after), blue = removed site
  structures;
- filters by split, category, flag, the fixed cases of the latest change (the hall and
  porch sets), the earlier review cases, and "cap changed" (shell more than 5% different
  from the previous run);
- sorts by most ignored, most removed, biggest and most changed against the previous run;
- search.

### Limitations after the Phase 3 fixes

- **Covered exterior air is supervised only 6 steps deep** (D22 amended), counted from open
  air (normalize-5). The middle of a very large, low canopy, a long covered passage or a deep
  veranda beyond that is ignored, like a hall interior. An eave strip counts as open only
  where it sees the sky along a path at least 2 up per 1 across, so a deep eave less than
  about twice as tall as it is deep is still capped at 6.
- **`large_opening` builds (20; 11 under normalize-4)** still have much exterior air under
  cover after the cap: canopies, station halls, barns, a facade shell whose hollow has no
  roof, and since normalize-5 the strips under deep, tall overhangs.
- **`deep_visible_ignored` builds (5)** have ignored blocks a horizontal line from outside
  reaches, at any depth: crawl spaces under raised floors, left as they are.
- **`ground_fallback` builds (3)** are houses built into a slope. With no ground, their
  buried parts are supervised. The fourth, the Richelieu cliff house, lost the rock that
  formed its rooms' back walls and is set aside (D25).
- **A wall with a coping attached to the building** merges into the building core and is
  kept. Only free-standing lines are cut.
- **A rock layer under the whole build** (no open terrain anywhere) is kept as building.
- **Invisible blocks** (`light` 24.7k, `barrier`, `structure_void`) become air.
- Vocabulary v3 (D20) followed Phase 3; the v2 → v3 comparison is in
  [vocab-coverage.md](vocab-coverage.md).

## Known limitations

From the collector (`source/REPORT.md`, `source/verification.json`) and this pass:

- **No legal ML clearance.** No entry is marked cleared for training; 0 explicit ML
  permissions, 0 unrestricted permissive licences. Personal or noncommercial intent does
  not establish ML permission. Hence D14: private training only.
- **Not executed.** Files passed bounded NBT parsing, dimension/block-array, vanilla-block
  and command-block/entity checks, and were never opened in Minecraft. Not a malware
  guarantee.
- **Previews are the authors' own** screenshots or renders (shaders, texture packs, text,
  rain, occlusion), not renders of the downloaded files. One barn preview is 128×72.
- **Dimensions measure the selection**, not the building: terrain, water, roads, vehicles
  and props are often included (the lighthouse needs cropping; the hangar holds aircraft).
  Cropping and segmentation are Phase 3.
- **Scale varies.** The hangar's creator describes it as 1.5:1. D15 has no scale rule:
  nothing is rescaled or filtered, and blocks-per-storey is recorded as information only.
- **Formats.** 142 legacy `.schematic` files need ID conversion and 67 `.litematic` files
  need a reader (Phase 3).
- **Geometric duplicates**: checked after the crop (Phase 3 fixes): none found
  ([Duplicates](#duplicates)).
- **Authorship and completeness** come from creator attribution and listings, not
  independent checks. Some builds are empty shells. 8 listings are marked 60–95% complete.
- **Residential dominates** (148 of 321).
