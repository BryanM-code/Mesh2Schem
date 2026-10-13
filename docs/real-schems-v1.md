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

Every non-excluded build (319) is read, cropped to the building, masked, measured and encoded
as training targets. The run is resumable and deterministic:

* A build is skipped when its record matches its source sha256, the pipeline version and the
  vocabulary (version and table digest).
* The npz files are byte-identical on a re-run. It never writes
inside `source/`. A full run takes about 50 s with 4 workers.
Code: `schemset/read.py`, `crop.py`, `interior.py`, `storey.py`, `coverage.py`, `review.py`,
`normalize.py`. Vocabulary coverage has its own page: [vocab-coverage.md](vocab-coverage.md).

### Outputs

| File | Content |
| --- | --- |
| `normalized/<id>.npz` | `blocks` (uint16, X Y Z, Y up, +X east, +Z south), `palette` (blockstate strings, `palette[0]` = air), `shell` / `exterior` / `interior` (bool, an exact partition of the grid), `crop_offset` (source-grid coordinates of [0,0,0]), `ground_y` (ground level in this grid; at or below it is ground), `target_shape` (uint8) / `target_material` (uint16): the training targets in `mcvox-v3` |
| `normalized/records/<id>.json` | everything measured for one build: read, validation, crop, masks, storey, supervised block counts per blockstate, `targets` (vocabulary version and digest, supervised and ignored counts, shape counts) |
| `normalized/index.json` | per build: status (ok / failed + reason), validation, crop stats, flags, interior fraction, storey; plus a dataset summary |

**Training targets** come from `mcvoxel.vocab.encode_target` (D20):

* Exterior air is air.
* Shell blocks the vocabulary holds get their (shape, material).
* Everything else is IGNORE (shape 255, material 65535), never air: the interior, decor,
  fluids, snow layers, unsupported blocks and door upper halves.

1.68% of shell blocks are IGNORE (38,173): 33,977 outside the vocabulary and 4,196 door upper
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

### Crop

`schemset/crop.py` keeps the building and removes ground, landscape and site clutter. The
approach differs from "largest connected mass" because the evidence said so:
- Builders texture walls with gravel, suspicious gravel, snow blocks, tuff, coral blocks and
  ores, so a block's name cannot say "landscape".
- Paved roads and plazas are connected to the building, so connectivity alone would keep them.

So roles are combined with position:
1. Find the ground level from the open columns.
2. Remove everything at or below it, except built blocks inside the footprint.
3. Take the footprint with the rule of `mcvoxel.grid.remove_site`, measured from the ground:
   columns of building material that rise at least 4 blocks, in masses at least 3 wide, plus
   a 2-column margin.
4. Outside the footprint, everything goes.
5. Inside it, plants, turf and water stay only where they sit on the building (green roofs,
   ivy, snow on a roof), not where they stand on open ground (hedges, lawns).
6. Trees are recognised from logs touching leaves and are removed.

The module docstring has every rule and threshold.

| | Result |
| --- | --- |
| Non-air blocks removed, whole dataset | 8.0 M of 16.5 M (48%): rock 3.94 M, turf 2.62 M, plants/snow cover 0.62 M, built 0.59 M, water 0.14 M, trees 0.06 M, invisible 0.02 M |
| Removed per build | median 22%; p25 7%, p75 50%, p90 70%, max 97% |
| Ground level found | 282 builds (202 from natural open ground, 36 open ground, 44 under the building); 37 have no ground |
| Built blocks kept below the ground | 253 builds (mostly floors set into the ground; also basements, foundations). Below-ground blocks are masked as ignored |
| Largest side after the crop | median 54 blocks, p75 85, max 427 |

### Interior mask

`schemset/interior.py`:
1. The ground level counts as solid, so a house whose floor was the grass plate does not flood
   from below.
2. Air is flooded from outside.
3. `mcvoxel.grid.occluded_interior` (air covered from above and blocked on 4 of the other 5
   axis directions) closes rooms leaked through open doorways.
4. Exterior air is re-flooded through what remains.
5. Shell = blocks touching exterior air; interior = the rest, which is ignored. Interior
   blocks are never deleted.

Interior fraction, i.e. interior / (interior + shell): median **0.86**, p10 0.78, p90 0.92,
min 0.64. The 8 `incomplete_listing` (exterior-only) builds need no special handling: their
interiors are ignored anyway (fractions 0.87–0.96).

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

Confidence: 102 high, 40 medium, 119 low (mostly one storey, measured floor to roof), 58
unknown.

### Flags added in Phase 3

| Flag | Meaning | Builds |
| --- | --- | ---: |
| `crop_aggressive` | the crop removed more than 50% of non-air blocks. Usually a big terrain capture (11 of the 12 `oversized_capture` builds); review the sheets | 85 |
| `multi_building` | more than one large separate mass remains (at least 15% of the biggest and 500 blocks): compounds, rows, campuses; all are kept | 18 |
| `interior_leak_suspect` | exterior air under a roof and more than 3 blocks from open sky is at least 100 blocks and over 20% of (itself + interior air): open structures (canopies, verandas, an unfinished tower) or rooms that leaked | 17 |
| `crop_kept_landscape` | plants, turf or water are over 10% of what the crop kept | 3 |
| `crop_no_core` | no building column rises 4 blocks above the ground in a 3-wide mass; every building column was kept instead | 2 |

Review renders (`review/sheets.json` lists what is where):
- Before/after crop sheets: every `oversized_capture`, `multi_building` and `crop_aggressive`
  build, plus 20 random others.
- Interior cutaways (interior air drawn red): 5 leak suspects and 5 random builds.

### Phase 3 limitations

- **Landscape scenes keep landscape.** A cottage in a forest diorama (trees with plank
  trunks), a lighthouse on a rock island, and houses on terraced slopes keep the rock under
  their footprint. These are flagged and in the review sheets.
- **Ambiguous interior.** Recessed porches and inner corners under eaves count as interior
  (ignored); open-top towers count as exterior. This follows the axis-ray heuristic.
- **Invisible blocks** (`light` 24.7k, `barrier`, `structure_void`) become air.
- Phase 3 only measured vocabulary coverage. Vocabulary v3 (D20) followed; the v2 → v3
  comparison is in [vocab-coverage.md](vocab-coverage.md).

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
- **Geometric duplicates untested**: rotations, crops, cross-format copies and minor
  variants (above).
- **Authorship and completeness** come from creator attribution and listings, not
  independent checks. Some builds are empty shells. 8 listings are marked 60–95% complete.
- **Residential dominates** (148 of 321).
