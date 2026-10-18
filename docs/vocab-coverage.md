# Vocabulary coverage on real-schems-v1: v2 → v3

How much of the human-built training targets the model vocabulary can express. **`mcvox-v3`**
(D20, with the 0.6.1 append of 2026-10-17) has 30 shapes × 313 materials and 1,232 valid
pairs; it is compared with **`mcvox-v2`** (10 × 120) on the same blocks. The vocabulary is
specified in Real2Minecraft `mcvoxel/README.md`, section "Vocabulary v3".

```powershell
.venv\Scripts\python -m schemset.normalize --dataset data/real-schems-v1   # targets + vocab_coverage.json
.venv\Scripts\python -m schemset.coverage --dataset data/real-schems-v1    # re-aggregate only
```

**Scope.** All 316 normalized builds (normalize-5 targets, 2026-10-17; the Richelieu cliff
house set aside by D25), counting **supervised blocks only**: the shell (blocks touching
exterior air), 2,804,192 blocks. Interior blocks are IGNORE in the targets and not counted.
Exterior air is supervised too (55.9 M voxels) but always representable, so it is kept out of
the percentages. Full data: `data/real-schems-v1/vocab_coverage.json`.

Earlier measurements: the first v3 run (319 builds, 2,273,717 blocks) gave 98.51% against
75.89% for v2; after the Phase 3 fixes (317 builds, 3,270,234 blocks) 98.44%. The depth cap
(normalize-4) and normalize-5 changed the shell since. On the same 316 builds, the 0.6.1
append raises representable from **98.51% to 98.58%** (train 98.43 → 98.51%, val 98.66 →
98.66%, test 99.26% unchanged).

v2 is recomputed with v2's own rules, frozen in `schemset/coverage.py`, so both columns count
the same blocks.

## Headline

| Class | v2 | v3 | v3 blocks | v3 builds |
| --- | ---: | ---: | ---: | ---: |
| exact | 72.28% | **96.37%** | 2,702,331 | 316 |
| lossy | 3.12% | 1.99% | 55,790 | 240 |
| derived (door upper halves) | — | 0.22% | 6,105 | 228 |
| ignore | 24.60% | 1.43% | 39,966 | 283 |
| **representable** | **75.40%** | **98.58%** | 2,764,226 | |

The ≥95% target is met.

v3's ignore share breaks down as follows:

* Decor, fluids and snow layers: 1.414%, kept out of the vocabulary by
  design (D20).
* Structural blocks the vocabulary does not hold: 299 blocks (0.011%),
  listed [below](#structural-blocks-still-outside-the-vocabulary).

"Builds" counts builds with at least one supervised block of that class.

### Class rules (`schemset/coverage.py`, `mcvoxel.vocab.classify_state`)

- **exact**: material and shape both predicted. Dropped properties either come back at
  assembly (stair corners, fence / wall / pane connections, a door's hinge, a gate's
  `in_wall`) or have no visible effect (waterlogged, powered, leaf distance, mushroom faces).
- **lossy**: encodable, but a visible property is dropped.
- **derived**: rebuilt at assembly, so IGNORE in the targets. This covers only a door's upper
  half.
- **ignore**: IGNORE in the targets, never air. This covers decor, fluids, snow layers and
  blocks outside the vocabulary.

| Reason | Class | What decode gives back | Share | Blocks | Builds |
| --- | --- | --- | ---: | ---: | ---: |
| alias | lossy | Applied at read time. Waxed copper becomes the identical-looking unwaxed block; most of these blocks are waxed copper. Infested stone becomes its host and the petrified oak slab becomes the oak slab | 0.69% | 19,345 | 49 |
| double_slab | lossy | The full block. Identical for most materials; smooth stone loses its side seam | 0.55% | 15,537 | 117 |
| trapdoor_facing | lossy | A closed trapdoor facing north: only the hinge side moves | 0.50% | 13,953 | 122 |
| door_upper_half | derived | Rebuilt from the lower half (hinge by the game's rule) | 0.22% | 6,105 | 228 |
| trapdoor_half | lossy | An open trapdoor with `half=bottom`: the same face is covered, with the hinge low | 0.13% | 3,655 | 120 |
| facing | lossy | Facing north: glazed terracotta loses its pattern rotation; the beehive and bee nest, their front | 0.10% | 2,832 | 16 |
| gate_open | lossy | A closed gate across the same opening | 0.01% | 323 | 32 |
| door_open | lossy | A closed door on the side its plate occupies | <0.01% | 188 | 62 |
| lit | lossy | Unlit (redstone lamp, copper bulb, redstone ore) | <0.01% | 68 | 13 |

## By family (v2 → v3)

Families group blocks the same way for both versions (`coverage.family_of`). The table gives
shares within each family.

| Family | Share | Blocks | v2 exact | v2 lossy | v2 ignore | v3 exact | v3 lossy | v3 derived | v3 ignore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 65.77% | 1,844,190 | 83.89% | 0.52% | 15.59% | 99.13% | 0.85% | 0 | 0.02% |
| slab | 9.87% | 276,622 | 88.34% | 0.66% | 11.01% | 99.34% | 0.66% | 0 | 0 |
| pane | 6.81% | 190,868 | 0 | 0 | 100.00% | 99.95% | 0.05% | 0 | 0 |
| stairs | 5.37% | 150,501 | 89.12% | 2.74% | 8.14% | 97.26% | 2.74% | 0 | 0 |
| wall | 2.94% | 82,342 | 83.75% | 0 | 16.25% | 100.00% | 0 | 0 | 0 |
| stairs_upside_down | 2.77% | 77,783 | 0 | 92.12% | 7.88% | 99.84% | 0.16% | 0 | 0 |
| trapdoor | 1.85% | 51,880 | 0 | 0 | 100.00% | 65.88% | 34.12% | 0 | 0 |
| fence | 0.84% | 23,695 | 79.81% | 0 | 20.19% | 99.97% | 0 | 0 | 0.03% |
| slab_double | 0.55% | 15,537 | 86.23% | 0.18% | 13.59% | 0 | 100.00% | 0 | 0 |
| leaves | 0.55% | 15,428 | 0 | 0 | 100.00% | 100.00% | 0 | 0 | 0 |
| door | 0.48% | 13,547 | 0 | 0 | 100.00% | 52.56% | 2.38% | 45.06% | 0 |
| full_sideways | 0.39% | 10,804 | 0 | 2.68% | 97.32% | 100.00% | 0 | 0 | 0 |
| carpet | 0.34% | 9,673 | 0 | 0 | 100.00% | 100.00% | 0 | 0 | 0 |
| decor:plants | 0.25% | 6,974 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:lighting | 0.21% | 5,908 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:redstone_rails | 0.20% | 5,494 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:buttons_plates | 0.19% | 5,300 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:furniture | 0.17% | 4,765 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:signs_banners | 0.12% | 3,493 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:snow_layers | 0.12% | 3,301 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| fence_gate | 0.06% | 1,655 | 0 | 0 | 100.00% | 80.48% | 19.52% | 0 | 0 |
| decor:ladders_chains_scaffolding | 0.06% | 1,551 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:fluids | 0.04% | 1,271 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:misc | 0.04% | 1,205 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:heads | 0.01% | 295 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:rods | <0.01% | 90 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:beds | <0.01% | 20 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |

The trapdoor family's lossy share is almost all closed-trapdoor facing and open-trapdoor half:
the shape is right and only the hinge position is lost. Door "derived" is the upper halves.
v2's upside-down stairs were lossy (top slabs); v3 predicts them.

## By split (v2 → v3)

| Split | Supervised blocks | v2 exact | v2 lossy | v2 ignore | **v2 representable** | v3 exact | v3 lossy | v3 derived | v3 ignore | **v3 representable** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2,323,492 | 72.33% | 3.19% | 24.48% | **75.52%** | 96.17% | 2.10% | 0.24% | 1.49% | **98.51%** |
| val | 285,440 | 76.11% | 1.27% | 22.62% | **77.38%** | 96.88% | 1.69% | 0.10% | 1.34% | **98.66%** |
| test | 195,260 | 66.09% | 4.97% | 28.94% | **71.06%** | 98.04% | 1.14% | 0.08% | 0.74% | **99.26%** |

The test split's v2 gap came from a few pane-heavy builds: light-blue and cyan stained-glass
curtain walls. Panes are a v3 shape, so test is now the best-covered split.

## Top 30 remaining unrepresentable names

Most are decor or snow layers, IGNORE by design (D20).

| # | Name | Family | Blocks | Share | Builds |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | short_grass | decor:plants | 3,595 | 0.13% | 61 |
| 2 | snow | decor:snow_layers | 3,301 | 0.12% | 14 |
| 3 | rail | decor:redstone_rails | 2,368 | 0.08% | 49 |
| 4 | stone_button | decor:buttons_plates | 1,907 | 0.07% | 84 |
| 5 | beacon | decor:furniture | 1,708 | 0.06% | 15 |
| 6 | lantern | decor:lighting | 1,582 | 0.06% | 60 |
| 7 | wall_torch | decor:lighting | 1,481 | 0.05% | 47 |
| 8 | end_rod | decor:lighting | 1,372 | 0.05% | 33 |
| 9 | water | decor:fluids | 1,271 | 0.04% | 18 |
| 10 | cobweb | decor:misc | 1,203 | 0.04% | 14 |
| 11 | stone_pressure_plate | decor:buttons_plates | 1,094 | 0.04% | 42 |
| 12 | torch | decor:lighting | 1,008 | 0.04% | 21 |
| 13 | anvil | decor:furniture | 955 | 0.03% | 41 |
| 14 | iron_chain | decor:ladders_chains_scaffolding | 938 | 0.03% | 30 |
| 15 | daylight_detector | decor:redstone_rails | 833 | 0.03% | 21 |
| 16 | spruce_wall_sign | decor:signs_banners | 724 | 0.03% | 21 |
| 17 | oak_button | decor:buttons_plates | 604 | 0.02% | 25 |
| 18 | mangrove_shelf | decor:furniture | 597 | 0.02% | 2 |
| 19 | birch_wall_sign | decor:signs_banners | 584 | 0.02% | 20 |
| 20 | vine | decor:plants | 583 | 0.02% | 10 |
| 21 | oak_wall_sign | decor:signs_banners | 578 | 0.02% | 71 |
| 22 | birch_button | decor:buttons_plates | 489 | 0.02% | 5 |
| 23 | ladder | decor:ladders_chains_scaffolding | 459 | 0.02% | 42 |
| 24 | lever | decor:redstone_rails | 434 | 0.01% | 36 |
| 25 | dark_oak_wall_sign | decor:signs_banners | 422 | 0.01% | 11 |
| 26 | oak_pressure_plate | decor:buttons_plates | 373 | 0.01% | 12 |
| 27 | tall_grass | decor:plants | 330 | 0.01% | 20 |
| 28 | tripwire | decor:redstone_rails | 328 | 0.01% | 13 |
| 29 | azure_bluet | decor:plants | 307 | 0.01% | 18 |
| 30 | piston | decor:redstone_rails | 277 | 0.01% | 8 |

### Structural blocks still outside the vocabulary

There are 299 blocks of these (0.011%), each name in a single build.

The vocabulary was chosen on the first Phase 3 crop: everything in at least 2 builds or 200
supervised blocks. The Phase 3 fixes kept building material the first crop had called
terrain by name (ore, gravel and sand facades, blackstone trim), and 12 more names met that
rule; they were appended on 2026-10-17 (D20 amended, mcvoxel 0.6.1). On normalize-5's
targets they hold 1,870 supervised blocks: gold ore 1,216, suspicious gravel 273, emerald ore
235, rooted dirt 75, chiseled polished blackstone 24, red sand 19, emerald block 11, redstone
ore 7, bee nest 4, bamboo mosaic 4, honey block 2, shroomlight 0. Redstone ore, honey block
(one build each) and shroomlight (none) no longer meet the rule; they stay, like the other
unused family members.

| Name | Blocks | Builds | Note |
| --- | ---: | ---: | --- |
| stripped_poplar_wood | 81 | 1 | not in any 1.21.x (D21) |
| dirt_path | 73 | 1 | not a full cube |
| cracked_nether_bricks | 38 | 1 |  |
| honeycomb_block | 26 | 1 |  |
| soul_soil | 19 | 1 |  |
| cracked_deepslate_tiles | 18 | 1 |  |
| tnt | 15 | 1 |  |
| gilded_blackstone | 8 | 1 |  |
| warped_nylium | 6 | 1 |  |
| poplar_fence | 6 | 1 | not in any 1.21.x (D21) |
| warped_wart_block | 4 | 1 |  |
| crimson_nylium | 3 | 1 |  |
| slime_block | 2 | 1 |  |

### Vocabulary materials no supervised block uses

There are 32, mostly from family completion, plus shroomlight from the 0.6.1 append: amethyst_block, magenta_concrete, purple_concrete, cherry_log, cherry_wood, pale_oak_log, crimson_stem, stripped_crimson_stem, crimson_hyphae, stripped_crimson_hyphae, warped_stem, warped_hyphae, pale_oak_leaves, weathered_chiseled_copper, oxidized_chiseled_copper, exposed_copper_grate, weathered_copper_grate, oxidized_copper_grate, oxidized_copper_bulb, magenta_concrete_powder, light_blue_concrete_powder, yellow_concrete_powder, lime_concrete_powder, pink_concrete_powder, blue_concrete_powder, red_concrete_powder, magenta_glazed_terracotta, lime_glazed_terracotta, pink_glazed_terracotta, black_glazed_terracotta, red_mushroom_block, shroomlight.

## v2 baseline (Phase 3, 2026-10-12)

Phase 3 measured v2 alone with finer "missing" classes:

| Class | Share | Blocks | Builds |
| --- | ---: | ---: | ---: |
| exact | 72.71% | 1,653,162 | 319 |
| lossy (top stairs → top slabs 2.46%, aliases 0.71%, sideways pillars upright 0.01%) | 3.18% | 72,386 | 250 |
| missing material, full cube | 10.17% | 231,265 | 260 |
| missing material, stair/slab | 1.98% | 44,952 | 100 |
| missing shape (panes 6.46%, trapdoors 1.94%, leaves, carpets, doors, fences/walls of unsupported materials, ...) | 11.22% | 255,066 | 309 |
| decor / clutter | 0.74% | 16,886 | 223 |

Its estimate was 94.67% for the top 20 materials plus the pane, door, trapdoor and
upside-down-stair classes. It recommended these steps:

* panes first
* about 20 materials
* upside-down stairs and trapdoors
* doors as a derived two-block object
* decor and the aliases kept out of the vocabulary

D20 took all of these and completed the families. On top of them, v3 adds horizontal
pillars, fence gates, carpets and leaves.

Not covered by this measurement: interior blocks (ignored) and anything the crop removed.
