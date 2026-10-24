# real-schems-v2: v1 plus the second collection

real-schems-v2 adds a second collection of human-built schematics to
[real-schems-v1](real-schems-v1.md) without changing anything of v1. Decision D26 in
[mesh-to-schem-plan.md](mesh-to-schem-plan.md).

| | |
| --- | --- |
| Builds | 1,012: v1's 321 + 691 new (Planet Minecraft), frozen 2026-10-21 |
| Splits | train 908 / val 48 / test 38 / excluded 18 (v1: 253 / 32 / 31 / 5, unchanged) |
| Location | `data/real-schems-v2/` (local only, gitignored with all of `data/`) |
| Collected at | `C:\Users\mendo\Documents\Codex\real-schems-collect-v2\` (never written to) |
| Normalization | normalize-6, all 994 non-excluded builds ok |

## Folder layout

```
data/real-schems-v2/
  source/                    the v2 collection, verbatim and read-only (16,377 files, 3.2 GB)
  freeze_checksums.sha256    sha256 of every file under source/
  FREEZE.md                  when, from where, verification
  index.json                 1,012 rows: v1's rows as v1 has them, then the new ones
  splits.json                the combined split, how it was made, every link and decision
  near_duplicates.json       metadata near-duplicates over both collections
  geometric_duplicates.json  schemset.dedupe over all normalized builds
  normalized/                one npz + record per build, both collections (normalize-6)
  vocab_coverage.json
  review/gallery_v2/         the review gallery (index.html)
  photofirst/                the photo pipeline's targets, renders, pairs and runs
```

v1's files are not copied: each v1 row has `source_root` `../real-schems-v1/source` and
`origin` `real-schems-v1`, new rows `source` and `real-schems-v2`. `schemset.normalize` and
`schemset.gallery` read each build from its row's `source_root`.

## How it was made

```powershell
.venv\Scripts\python -m schemset.freeze --source "C:\Users\mendo\Documents\Codex\real-schems-collect-v2" --out data/real-schems-v2 --base data/real-schems-v1
.venv\Scripts\python -m schemset.normalize --dataset data/real-schems-v2 --workers 5 --no-review
.venv\Scripts\python -m schemset.dedupe --dataset data/real-schems-v2
.venv\Scripts\python -m schemset.freeze --out data/real-schems-v2 --base data/real-schems-v1   # set-asides, duplicates
.venv\Scripts\python -m schemset.normalize --dataset data/real-schems-v2 --workers 5 --no-review
.venv\Scripts\python -m schemset.gallery --dataset data/real-schems-v2 --workers 5
```

**Freeze.** Like v1's: a verbatim copy to `source.partial/`, checked three ways (the
collection's own `checksums.sha256`, 691 lines; every manifest sha256; every file
byte-identical), then renamed and set read-only. The whole folder is kept, including the
collector's review material, so the provenance stays with the data.

**Format of the new manifest.** The same fields as v1's plus more. Two differ in form and
are read as follows: `listing_completion_percent` may be text ("100% complete",
"95% overall; ...": its leading number is taken), and `non_air_blocks_from_metadata` is empty
for 681 entries, so the collector's own count (`geometry_metrics.non_air_blocks_counted`,
from the parsed NBT) fills it (`non_air_basis`). Every count then matched the reader's.

**Licence tiers (D14).** The collector wrote one licence string per listing: 54 distinct
strings, each tiered by hand in `TIER_BY_LICENCE` (any new string still stops the run). A: a
stated general reuse permission, credit requested or not; B: no clear permission; C:
non-commercial or personal-use terms, and permissions only for playing the build (servers,
maps, videos, single-player). No new build is X: the collector excluded no-derivatives,
no-redistribution and no-ML terms. New builds: A 37, B 623, C 31.

**Splits.** v1's assignment is fixed. A new build joins the split of any v1 creator group it
is linked to (same creator, creator links, metadata near-duplicate candidates, union-find),
so a creator never straddles train and test; otherwise it goes to train. A new build whose
links reach v1 groups of two splits, or that duplicates a v1 build of another split, would be
excluded (none did). For the metadata near-duplicate test, a v1 build without a metadata
count uses the count v1's normalize read: v1's Coleraine church (5,915 blocks) and a new
house (3,878) share their dims and were otherwise linked. New builds: train 655, val 16
(7 creators), test 7 (4 creators), excluded 13 (below).

**Duplicates.** Geometric check after the crop over all 994 builds: 1,284 pairs past the
cheap filters, **0 duplicates**, no near misses. The one metadata candidate (a synagogue and
a tea shop with the same dims) is not a geometric duplicate.

## Normalization (normalize-6)

All 994 builds read and validated: formats MCEdit 619, Sponge v1 / v2 / v3 2 / 216 / 80,
Litematica v5 / v6 / v7 1 / 22 / 54; dims all match; non-air counts 854 match, 140 v1
builds have none; 0 unknown legacy ids. Vocabulary mcvox-v3: **98.13%** of supervised blocks
representable (96.55% exact). 8.97 M supervised blocks.

The second collection has many showcase plots: a building on a block of terrain, an island,
a hillside, a parking lot, a garden with trees. D23's crop left most of that standing
(rock under the building, aprons, trees, lamp posts), so normalize-6 adds three steps,
applied to both collections (`schemset/crop.py` docstring has every threshold):

9. **Terrain masses**: natural material in masses at least 5 blocks thick (an opening with a
   ball, measured with the terrain step 1 already took, so a cave or a torch that broke a
   column's ground run does not leave strips of rock), mostly terrain fill (stone, dirt,
   gravel, snow...; not sandstone, basalt, blackstone, nor ores, which a texture pack makes
   a builder's stone), go where they lie below the building's lowest built block within 2
   columns, or beside it, counting built structures at least 15% of the largest (not a
   mine's torches or a display stand's frame), and only when most of the mass is outside the
   building (a solid end-stone hotel behind its windows stays whole).
10. **Flat aprons, then trees**: material at most 2 blocks above the ground, more than one
    column from the footprint's near zone (2 columns) and from anything raised, goes:
    streets, parking lots, plazas, patios. Then trees standing against the building go: a
    leafy cluster with at least 30 leaves (judged by its whole crown), rooted near the
    ground, under a quarter of whose leaves touch the building, with far fewer logs than
    leaves. Green roofs, ivy, hedges and vertical gardens stay.
11. **Detached parts**: separate masses smaller than 15% of the main building (and 500
    blocks), and separate trees, go: lamp posts, cars, benches, fragments.

| Removed by the new steps | v1 (316 builds) | new (678 builds) |
| --- | ---: | ---: |
| Terrain masses | 30,288 blocks in 23 builds | 247,566 in 91 |
| Flat aprons | 67,240 in 178 | 135,578 in 418 |
| Trees against the building | 5,715 in 16 | 17,385 in 43 |
| Detached parts | 34,343 in 47 | 62,635 in 124 |

v1 builds keep a median 99.9% of their normalize-5 blocks (p90 94.4%); 40 lose over 5%, 4
over 20%, all checked: rings of trees round two houses, a garden, a parking lot and a kiosk,
a car by the cemetery chapel (val), small outbuildings below the size rule (a signal box, a
garage). No building lost walls. The eval builds' targets changed slightly (the chapel lost
the car), so earlier results are re-scored on these targets before comparing.

**Set aside (D26, `SET_ASIDE_NEW`)** after reviewing the 240 most suspicious new builds
(ranked by kept natural material, aprons, detached parts, site area, ground fallback, kept
landscape) before and after the crop: what remains of these is not a clean target of one
building.

| Build | Flag | Why |
| --- | --- | --- |
| pmc-beautiful-modern-cottage | excluded_landscape_diorama | a house in a forest |
| pmc-regional-airport-3328707 | excluded_landscape_diorama | an airfield; parked planes |
| pmc-tamas-manor | excluded_landscape_diorama | a terrain cube's debris remains |
| pmc-lighthouse-timelapse-schematic-1-17-1 | excluded_landscape_diorama | rock arches |
| pmc-the-bates-motel-from-psycho | excluded_landscape_diorama | two buildings on a hill |
| pmc-modern-house-4-3331081 | excluded_landscape_diorama | a whole walled garden plot |
| pmc-alpine-chalet | excluded_landscape_diorama | snow plate and spruces |
| pmc-single-story-barn-conversion-modern-house | excluded_landscape_diorama | in a wood |
| pmc-simple-house-2990730 | excluded_landscape_diorama | a garden of follies |
| pmc-hotel-de-angeli | excluded_display_stand | on a geological cross-section |
| pmc-modern-house-2196729 | excluded_display_stand | on a stone pedestal |
| pmc-clayleaf-nearly-fully-furnished-brick-house | excluded_two_builds | an unrelated second house |
| pmc-graumans-chinese-theatre---hollywood-landmark | excluded_incomplete_build | the hall is an empty box |

| Flag (994 builds) | Builds |
| --- | ---: |
| `crop_aggressive` | 250 |
| `apron_removed` | 202 |
| `large_opening` | 73 |
| `detached_removed` | 51 |
| `terrain_massif_removed` | 39 |
| `deep_visible_ignored` | 36 |
| `crop_kept_landscape` | 26 |
| `ground_fallback` | 26 |
| `multi_building` | 23 |
| `trees_removed` | 13 |
| `site_linear_removed` | 8 |

Ground: natural terrain in 597 builds, none (virtual plane) in 371, fallback in 26.

## Review gallery

`review/gallery_v2/index.html` (open locally): one card per build, the source before the
crop and every kept block after it, from two sides, with overlays (red: outside-visible but
not trained on; orange: below ground; pink: landscape removed or kept soft blocks; blue: site
removed). Filters by split, **origin** (real-schems-v1 / real-schems-v2), category and flag
(`terrain_massif_removed`, `apron_removed`, `trees_removed`, `detached_removed`...), sorts
by most removed and by most changed against normalize-5.

## Known limitations

- **Thick plot bases stay**: a plate of built blocks 3+ high round a building (a plinth, a
  raised plaza, a road built on a plate) is not flat, so step 10 keeps it.
- **Thin rock stays**: rock formations under 5 blocks thick (arches, crags) and a cliff a
  house is built into keep what lies within the building's envelope.
- **A shaft hanging from a building** (a mine reached by its stairs) stays: it is built and
  connected.
- **Small outbuildings** under 15% of the main building go with the lamp posts.
- **Resource-pack builds**: the collector flags several previews made with texture packs;
  blocks are vanilla, but some (ores, pistons) are used for their texture-pack look.
- Everything in [real-schems-v1.md, Known limitations](real-schems-v1.md#known-limitations)
  applies to the new collection too: no legal ML clearance (D14: private training only), not
  executed, scale varies (D15), authorship from listings.
