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

The old `eval/real-eval-v1.json` stays on disk for its tests but is superseded by this
split (D17, D19).

## Normalization (Phase 3)

```powershell
.venv\Scripts\python -m schemset.normalize --dataset data/real-schems-v1 [--only <id> ...] [--workers 4] [--force] [--no-review]
```

Every non-excluded build is read, cropped to the building, masked, measured and encoded as
training targets: 319 in Phase 3, 317 since D24 set two landscape dioramas aside (Phase 3
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

1.75% of shell blocks are IGNORE (57,316): 50,935 outside the vocabulary and 6,381 door upper
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
| Terrain floors restored | 130 builds, 35,385 blocks |
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
5. Shell is every block touching exterior air; interior is the rest, which is ignored and
   never deleted.

A room behind a doorway or window up to 2 wide stays interior, with or without glass.
`large_opening` flags builds where more than 30% of the *deep* covered air ends up exterior:
open halls, garages, wide arches, hollow facade shells. Deep covered air means air more than
3 blocks in from open sky; eaves and porches would otherwise flag most houses.

| | Result |
| --- | --- |
| Interior fraction (interior / (interior + shell)) | median 0.81, p10 0.66, p90 0.88, min 0.30 |
| Outside-visible but ignored (see below) | median 0.2%, p90 3.2%, max 18.6% |
| `large_opening` | 69 builds |

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

### Flags (after the Phase 3 fixes)

| Flag | Meaning | Builds |
| --- | --- | ---: |
| `large_opening` | more than 30% of the deep covered air is exterior: an open hall or garage, a wide arch, a hollow facade shell. Kept as exterior; review | 69 |
| `crop_aggressive` | the crop removed more than 50% of non-air blocks, usually a big terrain capture | 65 |
| `multi_building` | more than one large separate mass remains (at least 15% of the biggest and 500 blocks): compounds, rows, campuses; all are kept | 10 |
| `crop_kept_landscape` | soft blocks (plants, turf, water) are over 10% of what the crop kept | 8 |
| `ground_fallback` | more than 15% of the kept blocks would be below ground, so there is no ground: houses built into a slope | 4 |
| `site_linear_removed` | a wall or fence line was cut back to 6 columns from the building | 3 |
| `excluded_landscape_diorama` | D24: set aside (split "excluded"), not normalized | 2 |
| `geometric_duplicate` | `schemset.dedupe` found the build in a duplicate pair | 0 |

`interior_leak_suspect` and `crop_no_core` are retired: the opening rule replaced the first,
and every build now has a building core.

The review gallery is `review/gallery_v2/index.html` (`python -m schemset.gallery`). Each build
gets one card:
- the source BEFORE the crop and every kept block AFTER it, each from two sides;
- an overlay toggle: red = outside-visible but not trained on, orange = below ground and
  ignored, pink = landscape (removed before, kept soft blocks after), blue = removed site
  structures;
- filters by split, category, flag and the fixed cases of the review;
- sorts by most ignored, most removed, biggest and most changed against the previous run;
- search.

### Limitations after the Phase 3 fixes

- **`large_opening` builds (69)** are mostly halls, garages, open ground floors and hollow
  facade shells. Their inner walls are now shell, which is right for what is open but may
  not be what a photo shows.
- **`ground_fallback` builds (4)** are houses built into a slope. With no ground, their
  buried parts are supervised, and rooms dug into the removed hill open up (the Richelieu
  cliff house).
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
