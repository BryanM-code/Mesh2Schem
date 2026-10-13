# Vocabulary coverage on real-schems-v1: v2 → v3

How much of the human-built training targets the model vocabulary can express. **`mcvox-v3`**
(D20) has 30 shapes × 301 materials and 1,210 valid pairs; it is compared with **`mcvox-v2`**
(10 × 120) on the same blocks. The vocabulary is specified in Real2Minecraft
`mcvoxel/README.md`, section "Vocabulary v3".

```powershell
.venv\Scripts\python -m schemset.normalize --dataset data/real-schems-v1   # targets + vocab_coverage.json
.venv\Scripts\python -m schemset.coverage --dataset data/real-schems-v1    # re-aggregate only
```

**Scope.** All 319 non-excluded builds after the Phase 3 crop, counting **supervised blocks
only**: the shell (blocks touching exterior air), 2,273,717 blocks. Interior blocks are
IGNORE in the targets and not counted. Exterior air is supervised too (61.9 M voxels) but
always representable, so it is kept out of the percentages. Full data:
`data/real-schems-v1/vocab_coverage.json`.

v2 is recomputed with v2's own rules, frozen in `schemset/coverage.py`, so both columns count
the same blocks. Its representable share (75.89%) matches the Phase 3 measurement exactly.

## Headline

| Class | v2 | v3 | v3 blocks | v3 builds |
| --- | ---: | ---: | ---: | ---: |
| exact | 72.71% | **96.15%** | 2,186,181 | 319 |
| lossy | 3.18% | 2.17% | 49,363 | 222 |
| derived (door upper halves) | — | 0.18% | 4,196 | 160 |
| ignore | 24.11% | 1.49% | 33,977 | 277 |
| **representable** | **75.89%** | **98.51%** | 2,239,740 | |

The ≥95% target is met.

v3's ignore share breaks down as follows:

* Decor, fluids and snow layers: 1.479%, kept out of the vocabulary by design (D20).
* Structural blocks the vocabulary does not hold: 341 blocks (0.015%), listed
  [below](#structural-blocks-still-outside-the-vocabulary).

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
| alias | lossy | Applied at read time. Waxed copper becomes the identical-looking unwaxed block; most of these blocks are waxed copper. Infested stone becomes its host and the petrified oak slab becomes the oak slab | 0.85% | 19,369 | 42 |
| double_slab | lossy | The full block. Identical for most materials; smooth stone (3,876 of these blocks) loses its side seam | 0.55% | 12,476 | 116 |
| trapdoor_facing | lossy | A closed trapdoor facing north: only the hinge side moves | 0.53% | 11,996 | 117 |
| door_upper_half | derived | Rebuilt from the lower half (hinge by the game's rule) | 0.18% | 4,196 | 160 |
| trapdoor_half | lossy | An open trapdoor with `half=bottom`: the same face is covered, with the hinge low | 0.14% | 3,234 | 100 |
| facing | lossy | Facing north: glazed terracotta loses its pattern rotation (2,010 blocks), the beehive its front (48) | 0.09% | 2,058 | 11 |
| gate_open | lossy | A closed gate across the same opening | 0.01% | 256 | 25 |
| door_open | lossy | A closed door on the side its plate occupies | <0.01% | 55 | 25 |
| lit | lossy | Unlit (redstone lamp, copper bulb) | <0.01% | 36 | 4 |

## By family (v2 → v3)

Families group blocks the same way for both versions (`coverage.family_of`). The table gives
shares within each family.

| Family | Share | Blocks | v2 exact | v2 lossy | v2 ignore | v3 exact | v3 lossy | v3 derived | v3 ignore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 64.83% | 1,474,085 | 84.47% | 0.65% | 14.88% | 98.99% | 0.98% | 0 | 0.02% |
| slab | 10.33% | 234,834 | 86.73% | 1.15% | 12.12% | 98.85% | 1.15% | 0 | 0 |
| pane | 6.46% | 146,923 | 0 | 0 | 100.00% | 99.97% | 0.03% | 0 | 0 |
| stairs | 6.05% | 137,623 | 88.88% | 2.71% | 8.41% | 97.29% | 2.71% | 0 | 0 |
| wall | 2.97% | 67,510 | 84.06% | 0 | 15.94% | 100.00% | 0 | 0 | 0 |
| stairs_upside_down | 2.69% | 61,054 | 0 | 91.95% | 8.05% | 99.73% | 0.27% | 0 | 0 |
| trapdoor | 1.94% | 44,076 | 0 | 0 | 100.00% | 65.23% | 34.77% | 0 | 0 |
| fence | 0.84% | 19,125 | 76.30% | 0 | 23.70% | 99.97% | 0 | 0 | 0.03% |
| slab_double | 0.55% | 12,476 | 85.40% | 0.22% | 14.37% | 0 | 100.00% | 0 | 0 |
| leaves | 0.54% | 12,269 | 0 | 0 | 100.00% | 100.00% | 0 | 0 | 0 |
| full_sideways | 0.46% | 10,352 | 0 | 1.83% | 98.17% | 100.00% | 0 | 0 | 0 |
| carpet | 0.41% | 9,347 | 0 | 0 | 100.00% | 100.00% | 0 | 0 | 0 |
| door | 0.39% | 8,877 | 0 | 0 | 100.00% | 50.94% | 1.79% | 47.27% | 0 |
| fence_gate | 0.07% | 1,530 | 0 | 0 | 100.00% | 83.27% | 16.73% | 0 | 0 |
| decor:plants | 0.25% | 5,665 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:lighting | 0.22% | 5,063 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:redstone_rails | 0.22% | 5,006 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:buttons_plates | 0.18% | 4,121 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:furniture | 0.15% | 3,501 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:snow_layers | 0.13% | 3,057 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:signs_banners | 0.13% | 2,963 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:ladders_chains_scaffolding | 0.07% | 1,502 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:fluids | 0.06% | 1,266 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:misc | 0.05% | 1,154 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:heads | 0.01% | 255 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:rods | <0.01% | 71 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |
| decor:beds | <0.01% | 12 | 0 | 0 | 100.00% | 0 | 0 | 0 | 100.00% |

The trapdoor family's lossy share is almost all closed-trapdoor facing and open-trapdoor half:
the shape is right and only the hinge position is lost. Door "derived" is the upper halves.
v2's upside-down stairs were lossy (top slabs); v3 predicts them.

## By split (v2 → v3)

| Split | Supervised blocks | v2 exact | v2 lossy | v2 ignore | **v2 representable** | v3 exact | v3 lossy | v3 derived | v3 ignore | **v3 representable** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1,882,844 | 73.30% | 3.26% | 23.45% | **76.55%** | 95.92% | 2.31% | 0.21% | 1.56% | **98.44%** |
| val | 227,683 | 73.55% | 1.31% | 25.14% | **74.86%** | 96.67% | 1.80% | 0.10% | 1.43% | **98.57%** |
| test | 163,190 | 64.76% | 4.97% | 30.27% | **69.73%** | 98.11% | 1.09% | 0.03% | 0.77% | **99.23%** |

The test split's v2 gap came from a few pane-heavy builds: light-blue and cyan stained-glass
curtain walls. Panes are a v3 shape, so test is now the best-covered split.

## Top 30 remaining unrepresentable names

All 30 are decor or snow layers, IGNORE by design (D20).

| # | Name | Family | Blocks | Share | Builds |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | snow | decor:snow_layers | 3,057 | 0.13% | 12 |
| 2 | short_grass | decor:plants | 2,881 | 0.13% | 60 |
| 3 | rail | decor:redstone_rails | 2,164 | 0.10% | 50 |
| 4 | stone_button | decor:buttons_plates | 1,583 | 0.07% | 76 |
| 5 | wall_torch | decor:lighting | 1,442 | 0.06% | 40 |
| 6 | beacon | decor:furniture | 1,323 | 0.06% | 11 |
| 7 | water | decor:fluids | 1,266 | 0.06% | 11 |
| 8 | lantern | decor:lighting | 1,255 | 0.06% | 57 |
| 9 | cobweb | decor:misc | 1,151 | 0.05% | 13 |
| 10 | end_rod | decor:lighting | 991 | 0.04% | 29 |
| 11 | torch | decor:lighting | 979 | 0.04% | 17 |
| 12 | iron_chain | decor:ladders_chains_scaffolding | 904 | 0.04% | 27 |
| 13 | daylight_detector | decor:redstone_rails | 833 | 0.04% | 18 |
| 14 | stone_pressure_plate | decor:buttons_plates | 823 | 0.04% | 35 |
| 15 | spruce_wall_sign | decor:signs_banners | 694 | 0.03% | 20 |
| 16 | vine | decor:plants | 606 | 0.03% | 11 |
| 17 | anvil | decor:furniture | 575 | 0.03% | 31 |
| 18 | mangrove_shelf | decor:furniture | 536 | 0.02% | 2 |
| 19 | ladder | decor:ladders_chains_scaffolding | 500 | 0.02% | 40 |
| 20 | birch_wall_sign | decor:signs_banners | 497 | 0.02% | 17 |
| 21 | birch_button | decor:buttons_plates | 483 | 0.02% | 5 |
| 22 | oak_button | decor:buttons_plates | 480 | 0.02% | 22 |
| 23 | oak_wall_sign | decor:signs_banners | 462 | 0.02% | 60 |
| 24 | tripwire | decor:redstone_rails | 433 | 0.02% | 12 |
| 25 | lever | decor:redstone_rails | 403 | 0.02% | 32 |
| 26 | dark_oak_wall_sign | decor:signs_banners | 354 | 0.02% | 11 |
| 27 | tall_grass | decor:plants | 250 | 0.01% | 19 |
| 28 | flower_pot | decor:plants | 226 | 0.01% | 32 |
| 29 | observer | decor:redstone_rails | 213 | <0.01% | 3 |
| 30 | hopper | decor:redstone_rails | 211 | <0.01% | 26 |

### Structural blocks still outside the vocabulary

There are 341 blocks of these (0.015%). Each name appears in one build, with fewer than 200
blocks, so all are below the data rule (at least 2 builds or 200 supervised blocks).

Two are arguably family members v3 leaves out:

* Cracked deepslate tiles. D20's deepslate list names cobbled, polished, bricks, tiles and
  chiseled.
* The bamboo mosaic slab. Bamboo mosaic is a second bamboo planks texture.

| Name | Blocks | Note |
| --- | ---: | --- |
| rooted_dirt | 62 | |
| stripped_poplar_wood | 60 | not in any 1.21.x: `pmc-house-32` is DataVersion 5023 (D21) |
| suspicious_gravel | 58 | |
| dirt_path | 35 | not a full cube |
| honeycomb_block | 26 | |
| copper_ore | 22 | |
| cracked_deepslate_tiles | 18 | |
| red_sand | 17 | |
| soul_soil | 16 | |
| chiseled_polished_blackstone | 8 | |
| poplar_fence | 6 | not in any 1.21.x (D21) |
| warped_wart_block, shroomlight, slime_block, emerald_block, bamboo_mosaic_slab, bee_nest | 13 | |

### Vocabulary materials no supervised block uses

There are 33. Family completion accounts for 29 of them:

* Woods: cherry, pale oak, crimson and warped logs, stems and wood; pale oak leaves (12).
* Chiseled deepslate.
* Copper: oxidized chiseled copper, three copper grates and the oxidized copper bulb (5).
* Colours: 7 concrete powders and 3 glazed terracottas.
* The red mushroom block.

The other 4 are v2 materials no build uses: amethyst, magenta wool, purple concrete and the
raw iron block.

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
