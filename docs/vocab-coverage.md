# Vocabulary coverage on real-schems-v1: v2 → v3

How much of the human-built training targets the model vocabulary can express. **`mcvox-v3`**
(D20) has 30 shapes × 301 materials and 1,210 valid pairs; it is compared with **`mcvox-v2`**
(10 × 120) on the same blocks. The vocabulary is specified in Real2Minecraft
`mcvoxel/README.md`, section "Vocabulary v3".

```powershell
.venv\Scripts\python -m schemset.normalize --dataset data/real-schems-v1   # targets + vocab_coverage.json
.venv\Scripts\python -m schemset.coverage --dataset data/real-schems-v1    # re-aggregate only
```

**Scope.** All 317 non-excluded builds after the crop (Phase 3 fixes, 2026-10-12),
counting **supervised blocks only**: the shell (blocks touching exterior air), 3,270,234 blocks.
Interior blocks are IGNORE in the targets and not counted. Exterior air is supervised too
(58.6 M voxels) but always representable, so it is kept out of the
percentages. Full data: `data/real-schems-v1/vocab_coverage.json`.

The Phase 3 fixes grew the shell by half (porches, overhangs and balconies are now
supervised) and set two dioramas aside. The numbers below are for that shell. The first v3
measurement (319 builds, 2,273,717 blocks) gave 98.51% representable against 75.89% for v2.

v2 is recomputed with v2's own rules, frozen in `schemset/coverage.py`, so both columns count
the same blocks.

## Headline

| Class | v2 | v3 | v3 blocks | v3 builds |
| --- | ---: | ---: | ---: | ---: |
| exact | 72.82% | **96.09%** | 3,142,279 | 317 |
| lossy | 3.15% | 2.16% | 70,639 | 242 |
| derived (door upper halves) | — | 0.20% | 6,381 | 237 |
| ignore | 24.03% | 1.56% | 50,935 | 286 |
| **representable** | **75.97%** | **98.44%** | 3,219,299 | |

The ≥95% target is met.

v3's ignore share breaks down as follows:

* Decor, fluids and snow layers: 1.478%, kept out of the vocabulary by
  design (D20).
* Structural blocks the vocabulary does not hold: 2,602 blocks (0.080%),
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
| alias | lossy | Applied at read time. Waxed copper becomes the identical-looking unwaxed block; most of these blocks are waxed copper. Infested stone becomes its host and the petrified oak slab becomes the oak slab | 0.86% | 28,072 | 51 |
| double_slab | lossy | The full block. Identical for most materials; smooth stone loses its side seam | 0.54% | 17,599 | 118 |
| trapdoor_facing | lossy | A closed trapdoor facing north: only the hinge side moves | 0.47% | 15,230 | 129 |
| door_upper_half | derived | Rebuilt from the lower half (hinge by the game's rule) | 0.20% | 6,381 | 237 |
| trapdoor_half | lossy | An open trapdoor with `half=bottom`: the same face is covered, with the hinge low | 0.14% | 4,604 | 124 |
| facing | lossy | Facing north: glazed terracotta loses its pattern rotation; the beehive, its front | 0.13% | 4,397 | 17 |
| gate_open | lossy | A closed gate across the same opening | 0.01% | 362 | 35 |
| door_open | lossy | A closed door on the side its plate occupies | <0.01% | 234 | 67 |
| lit | lossy | Unlit (redstone lamp, copper bulb) | <0.01% | 275 | 15 |

## By family (v2 → v3)

Families group blocks the same way for both versions (`coverage.family_of`). The table gives
shares within each family.

| Family | Share | Blocks | v2 exact | v2 lossy | v2 ignore | v3 exact | v3 lossy | v3 derived | v3 ignore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 67.58% | 2,210,094 | 83.95% | 0.59% | 15.46% | 98.93% | 0.95% | 0 | 0.12% |
| slab | 9.80% | 320,475 | 86.33% | 2.19% | 11.48% | 97.81% | 2.19% | 0 | <0.01% |
| pane | 6.12% | 200,191 | 0 | 0 | 100.00% | 99.96% | 0.04% | 0 | 0 |
| stairs | 4.78% | 156,374 | 89.35% | 2.64% | 8.01% | 97.36% | 2.64% | 0 | 0 |
| wall | 2.71% | 88,504 | 82.21% | 0 | 17.79% | 100.00% | 0 | 0 | 0 |
| stairs_upside_down | 2.60% | 85,110 | 0 | 92.26% | 7.74% | 99.86% | 0.14% | 0 | 0 |
| trapdoor | 1.70% | 55,679 | 0 | 0 | 100.00% | 64.21% | 35.79% | 0 | 0 |
| fence | 0.83% | 27,233 | 79.76% | 0 | 20.24% | 99.98% | 0 | 0 | 0.02% |
| full_sideways | 0.55% | 17,844 | 0 | 1.93% | 98.07% | 100.00% | 0 | 0 | 0 |
| slab_double | 0.54% | 17,599 | 85.72% | 0.19% | 14.09% | 0 | 100.00% | 0 | 0 |
| leaves | 0.48% | 15,755 | 0 | 0 | 100.00% | 100.00% | 0 | 0 | 0 |
| door | 0.44% | 14,273 | 0 | 0 | 100.00% | 52.67% | 2.62% | 44.71% | 0 |
| carpet | 0.33% | 10,899 | 0 | 0 | 100.00% | 100.00% | 0 | 0 | 0 |
| fence_gate | 0.06% | 1,871 | 0 | 0 | 100.00% | 80.65% | 19.35% | 0 | 0 |
| decor:lighting | 0.23% | 7,580 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:plants | 0.23% | 7,525 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:furniture | 0.23% | 7,430 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:redstone_rails | 0.19% | 6,239 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:buttons_plates | 0.19% | 6,113 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:signs_banners | 0.13% | 4,129 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:snow_layers | 0.10% | 3,425 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:ladders_chains_scaffolding | 0.07% | 2,158 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:fluids | 0.06% | 1,979 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:misc | 0.04% | 1,259 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:heads | 0.01% | 383 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:rods | <0.01% | 91 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:beds | <0.01% | 22 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |

The trapdoor family's lossy share is almost all closed-trapdoor facing and open-trapdoor half:
the shape is right and only the hinge position is lost. Door "derived" is the upper halves.
v2's upside-down stairs were lossy (top slabs); v3 predicts them.

## By split (v2 → v3)

| Split | Supervised blocks | v2 exact | v2 lossy | v2 ignore | **v2 representable** | v3 exact | v3 lossy | v3 derived | v3 ignore | **v3 representable** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2,752,364 | 72.92% | 3.25% | 23.83% | **76.17%** | 95.87% | 2.27% | 0.21% | 1.64% | **98.36%** |
| val | 304,131 | 75.64% | 1.24% | 23.12% | **76.88%** | 97.00% | 1.60% | 0.10% | 1.30% | **98.70%** |
| test | 213,739 | 67.46% | 4.56% | 27.98% | **72.02%** | 97.58% | 1.49% | 0.07% | 0.86% | **99.14%** |

The test split's v2 gap came from a few pane-heavy builds: light-blue and cyan stained-glass
curtain walls. Panes are a v3 shape, so test is now the best-covered split.

## Top 30 remaining unrepresentable names

Most are decor or snow layers, IGNORE by design (D20).

| # | Name | Family | Blocks | Share | Builds |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | short_grass | decor:plants | 3,743 | 0.11% | 62 |
| 2 | snow | decor:snow_layers | 3,425 | 0.10% | 18 |
| 3 | beacon | decor:furniture | 2,499 | 0.08% | 16 |
| 4 | rail | decor:redstone_rails | 2,375 | 0.07% | 49 |
| 5 | stone_button | decor:buttons_plates | 2,328 | 0.07% | 89 |
| 6 | wall_torch | decor:lighting | 2,100 | 0.06% | 48 |
| 7 | lantern | decor:lighting | 2,097 | 0.06% | 63 |
| 8 | water | decor:fluids | 1,979 | 0.06% | 22 |
| 9 | end_rod | decor:lighting | 1,801 | 0.06% | 35 |
| 10 | iron_chain | decor:ladders_chains_scaffolding | 1,344 | 0.04% | 34 |
| 11 | stone_pressure_plate | decor:buttons_plates | 1,309 | 0.04% | 46 |
| 12 | cobweb | decor:misc | 1,251 | 0.04% | 14 |
| 13 | gold_ore | full | 1,221 | 0.04% | 3 |
| 14 | anvil | decor:furniture | 1,179 | 0.04% | 42 |
| 15 | torch | decor:lighting | 1,092 | 0.03% | 25 |
| 16 | daylight_detector | decor:redstone_rails | 888 | 0.03% | 22 |
| 17 | oak_wall_sign | decor:signs_banners | 878 | 0.03% | 83 |
| 18 | spruce_wall_sign | decor:signs_banners | 807 | 0.03% | 25 |
| 19 | oak_button | decor:buttons_plates | 645 | 0.02% | 27 |
| 20 | ladder | decor:ladders_chains_scaffolding | 628 | 0.02% | 48 |
| 21 | loom | decor:furniture | 628 | 0.02% | 5 |
| 22 | birch_wall_sign | decor:signs_banners | 622 | 0.02% | 21 |
| 23 | vine | decor:plants | 612 | 0.02% | 11 |
| 24 | mangrove_shelf | decor:furniture | 597 | 0.02% | 2 |
| 25 | lever | decor:redstone_rails | 570 | 0.02% | 43 |
| 26 | birch_button | decor:buttons_plates | 496 | 0.01% | 6 |
| 27 | dark_oak_wall_sign | decor:signs_banners | 446 | 0.01% | 12 |
| 28 | tripwire | decor:redstone_rails | 434 | 0.01% | 16 |
| 29 | oak_pressure_plate | decor:buttons_plates | 408 | 0.01% | 14 |
| 30 | tall_grass | decor:plants | 396 | 0.01% | 20 |

### Structural blocks still outside the vocabulary

There are 2,602 blocks of these (0.080%).

The vocabulary was chosen on the first Phase 3 crop: everything in at least 2 builds or 200
supervised blocks. Since the Phase 3 fixes the crop keeps building material that it used to
call terrain by name (ore, gravel and sand facades, blackstone trim). As a result,
12 of these names now meet that data rule: bamboo_mosaic_slab, bee_nest, chiseled_polished_blackstone, emerald_block, emerald_ore, gold_ore, honey_block, red_sand, redstone_ore, rooted_dirt, shroomlight, suspicious_gravel.

They are not added yet: the Phase 3 fixes left the vocabulary unchanged, and adding them is
a v3 append for you to decide.

| Name | Blocks | Builds | Note |
| --- | ---: | ---: | --- |
| gold_ore | 1,221 | 3 | meets the D20 data rule now |
| chiseled_polished_blackstone | 337 | 4 | meets the D20 data rule now |
| suspicious_gravel | 273 | 2 | meets the D20 data rule now |
| emerald_ore | 241 | 3 | meets the D20 data rule now |
| stripped_poplar_wood | 81 | 1 | not in any 1.21.x (D21) |
| rooted_dirt | 75 | 3 | meets the D20 data rule now |
| dirt_path | 73 | 1 | not a full cube |
| gilded_blackstone | 55 | 1 |  |
| cracked_nether_bricks | 38 | 1 |  |
| soul_soil | 37 | 1 |  |
| redstone_ore | 26 | 2 | meets the D20 data rule now |
| honeycomb_block | 26 | 1 |  |
| red_sand | 19 | 2 | meets the D20 data rule now |
| cracked_deepslate_tiles | 18 | 1 |  |
| emerald_block | 17 | 3 | meets the D20 data rule now |
| tnt | 15 | 1 |  |
| shroomlight | 13 | 2 | meets the D20 data rule now |
| honey_block | 8 | 2 | meets the D20 data rule now |
| poplar_fence | 6 | 1 | not in any 1.21.x (D21) |
| warped_nylium | 6 | 1 |  |
| bamboo_mosaic_slab | 4 | 2 | meets the D20 data rule now |
| bee_nest | 4 | 2 | meets the D20 data rule now |
| warped_wart_block | 4 | 1 |  |
| crimson_nylium | 3 | 1 |  |
| slime_block | 2 | 1 |  |

### Vocabulary materials no supervised block uses

There are 31, mostly from family completion: amethyst_block, magenta_concrete, purple_concrete, cherry_log, cherry_wood, pale_oak_log, crimson_stem, stripped_crimson_stem, crimson_hyphae, stripped_crimson_hyphae, warped_stem, warped_hyphae, pale_oak_leaves, weathered_chiseled_copper, oxidized_chiseled_copper, exposed_copper_grate, weathered_copper_grate, oxidized_copper_grate, oxidized_copper_bulb, magenta_concrete_powder, light_blue_concrete_powder, yellow_concrete_powder, lime_concrete_powder, pink_concrete_powder, blue_concrete_powder, red_concrete_powder, magenta_glazed_terracotta, lime_glazed_terracotta, pink_glazed_terracotta, black_glazed_terracotta, red_mushroom_block.

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
