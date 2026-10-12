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
- **Scale varies.** The hangar's creator describes it as 1.5:1; D15 records
  blocks-per-storey per build rather than rescaling.
- **Formats.** 142 legacy `.schematic` files need ID conversion and 67 `.litematic` files
  need a reader (Phase 3).
- **Geometric duplicates untested**: rotations, crops, cross-format copies and minor
  variants (above).
- **Authorship and completeness** come from creator attribution and listings, not
  independent checks. Some builds are empty shells. 8 listings are marked 60–95% complete.
- **Residential dominates** (148 of 321).
