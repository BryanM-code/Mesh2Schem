# Phase 3 fixes (2026-10-12): root causes and before / after

The first Phase 3 run had three problems:
- porches, overhangs and balconies were ignored;
- the ground was detected too high on several builds;
- the crop called building materials landscape by name.

Decisions D22 (exterior by opening size), D23 (only natural terrain makes ground) and D24
(two dioramas set aside) are in [mesh-to-schem-plan.md](mesh-to-schem-plan.md). The rules
are in [real-schems-v1.md, Normalization](real-schems-v1.md#normalization-phase-3). Every
build can be checked in the review gallery, `data/real-schems-v1/review/gallery_v2/` (the
"fixed cases" filter shows these 17 first).

## Root causes

**Crop cases**

| Build | What was wrong | Fix |
| --- | --- | --- |
| `pmc-chapel-in-the-cemetery-download` | The cemetery wall has slab copings and iron bars, which make its top 3 columns wide. It passed the 3×3 building-core test as a 907-column core, larger than the chapel's 500. A single ground level (y 0) also made the wall, standing on a hill, look 4+ blocks tall. | The ground height is now per column, so measured from the hill the wall is not tall enough to be a core. Site material more than 6 columns from the building goes. A free-standing wall line with a coping is also caught by the new line rule on flat ground (tested). 6,224 site blocks removed; the chapel and a small tomb stay. |
| `ab-4273` | Two dark-oak logs of the timber frame, high on a wall (y 9–10), touched a decorative leaf bush, were taken for a tree trunk and removed: the hole. The bakery's ground floor is also an open shop front, so before the fix its ignored interior showed through that side. | A tree must stand on the ground; floating logs are building. Both logs are kept. The open front (5 wide) makes the shop exterior, so it is trained on (`large_opening`). |
| `pmc-east-german-kt60-gym-hall-turnhalle` | The hall has no floor of its own: the builder used the world's grass. The crop removed it as terrain, so the kept grid had no bottom and you saw through the windows into nothing. The 24 stone blocks removed at ground level were a path. | Terrain floors: the top terrain block under an enclosed room standing on the ground is kept as its floor, and ignored like any floor (398 blocks here). |
| `ab-11853` | Nothing above the ground was lost: the old crop removed 6 stray blocks. The source is the whole house: the listing shows a one-storey house with a roof terrace, pool and pergolas, which is what the schematic holds. | General fixes only (visible but ignored 0.9% → 0%). |

**Ground bugs.** The old ground level was the most common top of a "ground run" (any stone,
sand, gravel, snow or water by name) in columns without built blocks. Those runs could start
wherever the column's lowest block was.

| Build | What was wrong | Fix |
| --- | --- | --- |
| `pmc-european-building-12-kaiserthal-project-by-tim0fei` | Cobblestone pilasters run from the bottom to the roof with no "built" block in their columns, so the ground went to the roof (y 27): 95% below ground. | Terrain needs a run that starts at the bottom of the capture, survives a 3×3 opening per layer, reaches the edge of the build and shows natural evidence (grass, dirt, sand, gravel, snow, water, or a rock mass 3+ deep). Cobblestone is not natural. No terrain, so no ground: 0% below ground, nothing removed. |
| `pmc-10-early-19-century-style-rowhouses` | Snow blocks and obsidian on the roofs started ground runs at roof height (y 15): 74% below ground. | Same: runs must start at the bottom. 0% below ground. |
| `pmc-eunpyeong-public-library-seoul` | Its stone walls (14k stone blocks) counted as ground (y 21): 69% below ground, 81% removed. | Same; thin walls drop out of the per-layer opening. 0% below ground, 0.2% removed. |
| `pmc-19th-century---brewery` | Cobblestone cornice columns at y 59 outnumbered the canal's water columns: 72% below ground, 62% removed. | Same. 0% below ground, 0.3% removed. |
| `pmc-richelieu-contemporary` | A cliff house built into a stone hill. One global ground level (the hilltop) put most of the house below ground. | Ground per column. The house is genuinely embedded (over 15% of it would be buried), so the safety net applies: no ground (`ground_fallback`), 0% below ground. Rooms dug into the removed hill are now open (`large_opening`). Review: the hill was their wall. |
| `pmc-old-warehouse` | Warehouse on a plate beside a lower quay. One global level (the plate) put the quay-side facade below ground. | Ground per column, then the safety net (`ground_fallback`): its foundation sits in the plate. 0% below ground. |

**Porches and overhangs.** `mcvoxel.grid.occluded_interior` made air interior when it was
covered from above and blocked on 4 of the other 5 axis directions. Porches, recessed porches
and balconies, jetties, verandas and eaves all qualify.
- Fix (D22): exterior is any air the outside reaches through openings at least 3 wide and 2
  tall, plus 1 block of reveal through smaller ones, plus open gaps.
- Remaining "visible but ignored" on the Cape Cod house (13.2%) is its restored terrain floor
  and its furniture, seen through glassless windows up to 2 wide. The rule keeps those rooms
  interior on purpose; the audit's line of sight counts them.

## Before / after per build

Same 317 builds, previous pipeline version (`normalize-2`) → this one (`normalize-3`). The
measures are defined in [real-schems-v1.md](real-schems-v1.md#phase-3-fixes-before-and-after-2026-10-12).
The shell is the set of supervised blocks.

### crop

| Build | removed % | interior | visible but ignored % | below ground % | shell |
| --- | ---: | ---: | ---: | ---: | ---: |
| pmc-chapel-in-the-cemetery-download | 87.6 → 93.1 | 0.93 → 0.85 | 6.8 → 0.3 | 0.0 → 0.1 | 4,018 → 2,488 (-38%) |
| ab-4273 | 20.9 → 19.7 | 0.79 → 0.65 | 20.3 → 0.0 | 0.0 → 0.0 | 546 → 791 (+45%) |
| pmc-east-german-kt60-gym-hall-turnhalle | 38.8 → 18.8 | 0.81 → 0.80 | 2.9 → 0.0 | 0.0 → 0.0 | 1,022 → 1,054 (+3%) |
| ab-11853 | 17.5 → 17.3 | 0.82 → 0.79 | 0.9 → 0.0 | 0.0 → 0.0 | 1,345 → 1,492 (+11%) |

### ground

| Build | removed % | interior | visible but ignored % | below ground % | shell |
| --- | ---: | ---: | ---: | ---: | ---: |
| pmc-european-building-12-kaiserthal-project-by-tim0fei | 48.5 → 0.0 | 0.99 → 0.84 | 0.0 → 0.1 | 95.1 → 0.0 | 164 → 3,858 (+2252%) |
| pmc-10-early-19-century-style-rowhouses | 17.3 → 0.0 | 0.96 → 0.78 | 32.3 → 2.9 | 73.7 → 0.0 | 799 → 3,194 (+300%) |
| pmc-eunpyeong-public-library-seoul | 81.5 → 0.2 | 0.96 → 0.83 | 0.0 → 0.4 | 69.3 → 0.0 | 1,638 → 10,356 (+532%) |
| pmc-richelieu-contemporary | 94.3 → 84.4 | 0.99 → 0.58 | 0.0 → 1.1 | 67.4 → 0.0 | 354 → 4,162 (+1076%) |
| pmc-19th-century---brewery | 62.3 → 0.3 | 0.98 → 0.66 | 34.6 → 0.1 | 71.9 → 0.0 | 5,985 → 94,787 (+1484%) |
| pmc-old-warehouse | 50.4 → 47.9 | 0.94 → 0.84 | 27.5 → 0.0 | 26.8 → 0.0 | 3,805 → 8,871 (+133%) |

### porch

| Build | removed % | interior | visible but ignored % | below ground % | shell |
| --- | ---: | ---: | ---: | ---: | ---: |
| ab-6057 | 1.3 → 1.1 | 0.90 → 0.75 | 49.3 → 0.1 | 0.5 → 0.0 | 2,785 → 5,982 (+115%) |
| pmc-triple-gable-tudor-townhouse-6884362 | 15.1 → 14.6 | 0.84 → 0.74 | 25.7 → 0.1 | 0.0 → 0.0 | 1,350 → 1,998 (+48%) |
| pmc-cape-cod-house---furnished | 58.0 → 53.2 | 0.88 → 0.83 | 18.9 → 13.2 | 0.0 → 0.0 | 1,326 → 1,877 (+42%) |
| pmc-small-craftsman-house---exterior-amp-landscape-only | 83.2 → 78.4 | 0.88 → 0.75 | 22.3 → 0.0 | 0.0 → 0.0 | 1,318 → 2,908 (+121%) |
| pmc-victorian-queen-anne-style---carson-mansion | 0.0 → 0.0 | 0.86 → 0.83 | 16.1 → 0.1 | 0.0 → 0.0 | 4,202 → 5,153 (+23%) |
| pmc-spanish-colonial-house---philippines | 39.6 → 37.7 | 0.88 → 0.84 | 27.7 → 0.2 | 0.0 → 0.0 | 1,144 → 1,714 (+50%) |
| pmc-greek-revival-plantation-house | 66.8 → 66.2 | 0.92 → 0.91 | 17.8 → 0.7 | 0.0 → 0.3 | 13,039 → 19,274 (+48%) |

### Every build whose removed share moved by more than 10 points or whose shell changed by more than 10% (256)

| Build | removed % | interior | visible but ignored % | below ground % | shell |
| --- | ---: | ---: | ---: | ---: | ---: |
| ab-11338 | 2.3 → 0.5 | 0.81 → 0.77 | 15.0 → 0.0 | 0.0 → 0.0 | 824 → 1,004 (+22%) |
| ab-11479 | 24.4 → 9.4 | 0.87 → 0.65 | 28.7 → 0.0 | 0.0 → 0.0 | 2,684 → 6,285 (+134%) |
| ab-11538 | 12.4 → 19.8 | 0.89 → 0.87 | 22.3 → 4.2 | 0.0 → 7.8 | 6,430 → 7,532 (+17%) |
| ab-11663 | 0.0 → 0.0 | 0.92 → 0.90 | 8.3 → 0.7 | 0.0 → 0.0 | 56,546 → 68,091 (+20%) |
| ab-11707 | 22.1 → 0.4 | 0.71 → 0.66 | 10.8 → 2.4 | 0.0 → 0.0 | 1,358 → 1,718 (+27%) |
| ab-11853 | 17.5 → 17.3 | 0.82 → 0.79 | 0.9 → 0.0 | 0.0 → 0.0 | 1,345 → 1,492 (+11%) |
| ab-12498 | 68.3 → 67.4 | 0.85 → 0.88 | 5.7 → 0.0 | 0.0 → 0.1 | 6,910 → 7,870 (+14%) |
| ab-12687 | 30.9 → 27.1 | 0.93 → 0.86 | 49.6 → 0.0 | 0.1 → 0.2 | 2,774 → 5,731 (+107%) |
| ab-12933 | 19.4 → 17.2 | 0.80 → 0.79 | 6.7 → 0.0 | 0.0 → 0.0 | 471 → 527 (+12%) |
| ab-13041 | 1.2 → 0.0 | 0.86 → 0.77 | 27.2 → 0.0 | 0.0 → 0.0 | 1,067 → 1,539 (+44%) |
| ab-13061 | 14.3 → 12.0 | 0.88 → 0.83 | 23.5 → 0.8 | 0.0 → 0.0 | 3,953 → 5,298 (+34%) |
| ab-13068 | 34.5 → 33.2 | 0.80 → 0.65 | 15.6 → 3.4 | 0.0 → 0.0 | 1,215 → 1,867 (+54%) |
| ab-13082 | 20.3 → 18.5 | 0.85 → 0.82 | 9.1 → 1.7 | 0.0 → 0.0 | 1,636 → 1,977 (+21%) |
| ab-13217 | 0.1 → 0.0 | 0.84 → 0.82 | 10.3 → 0.1 | 0.0 → 0.0 | 3,343 → 3,726 (+11%) |
| ab-13253 | 2.7 → 0.1 | 0.97 → 0.92 | 55.2 → 0.0 | 0.0 → 0.0 | 25,304 → 70,811 (+180%) |
| ab-13254 | 9.9 → 5.1 | 0.83 → 0.57 | 15.5 → 0.5 | 0.0 → 0.0 | 10,636 → 18,932 (+78%) |
| ab-13317 | 2.5 → 0.0 | 0.84 → 0.78 | 17.2 → 0.0 | 0.0 → 0.0 | 1,046 → 1,297 (+24%) |
| ab-13320 | 0.7 → 0.1 | 0.85 → 0.81 | 13.0 → 0.0 | 0.0 → 0.0 | 2,387 → 2,826 (+18%) |
| ab-13526 | 8.3 → 7.9 | 0.86 → 0.48 | 25.5 → 2.7 | 0.0 → 0.0 | 9,962 → 15,901 (+60%) |
| ab-13710 | 36.3 → 38.3 | 0.86 → 0.80 | 17.2 → 0.4 | 0.0 → 0.0 | 2,063 → 2,603 (+26%) |
| ab-13714 | 4.2 → 3.0 | 0.90 → 0.87 | 20.8 → 1.8 | 5.1 → 0.7 | 17,439 → 22,322 (+28%) |
| ab-13926 | 1.3 → 0.0 | 0.80 → 0.77 | 8.3 → 0.0 | 0.0 → 0.0 | 643 → 732 (+14%) |
| ab-13971 | 9.9 → 9.9 | 0.81 → 0.77 | 13.6 → 0.4 | 0.0 → 0.0 | 996 → 1,155 (+16%) |
| ab-13985 | 13.1 → 5.0 | 0.85 → 0.79 | 8.6 → 0.7 | 0.0 → 0.0 | 3,579 → 5,087 (+42%) |
| ab-14102 | 40.5 → 19.3 | 0.81 → 0.78 | 6.1 → 1.8 | 0.0 → 0.0 | 848 → 1,332 (+57%) |
| ab-2014 | 3.1 → 1.4 | 0.90 → 0.86 | 17.6 → 0.4 | 0.0 → 0.0 | 22,565 → 30,870 (+37%) |
| ab-2278 | 27.4 → 25.5 | 0.89 → 0.85 | 20.8 → 0.1 | 0.0 → 0.0 | 5,904 → 7,818 (+32%) |
| ab-2279 | 19.5 → 0.4 | 0.89 → 0.62 | 9.0 → 0.0 | 0.0 → 0.0 | 9,748 → 17,696 (+82%) |
| ab-2280 | 0.4 → 0.0 | 0.86 → 0.76 | 25.6 → 0.9 | 0.0 → 0.0 | 4,582 → 6,491 (+42%) |
| ab-2282 | 42.6 → 34.8 | 0.85 → 0.78 | 14.4 → 3.6 | 1.0 → 1.8 | 7,055 → 9,868 (+40%) |
| ab-2283 | 40.1 → 30.8 | 0.89 → 0.84 | 20.8 → 0.0 | 0.5 → 1.4 | 4,985 → 6,904 (+38%) |
| ab-2286 | 17.7 → 10.8 | 0.88 → 0.82 | 24.4 → 0.0 | 0.0 → 0.0 | 4,538 → 6,909 (+52%) |
| ab-2425 | 40.7 → 35.2 | 0.81 → 0.79 | 11.6 → 0.3 | 0.0 → 0.0 | 547 → 700 (+28%) |
| ab-2586 | 18.6 → 12.6 | 0.80 → 0.76 | 15.5 → 1.3 | 0.0 → 0.0 | 757 → 888 (+17%) |
| ab-2988 | 22.5 → 20.9 | 0.91 → 0.77 | 7.1 → 0.1 | 0.2 → 1.5 | 30,435 → 55,921 (+84%) |
| ab-3052 | 63.2 → 60.4 | 0.84 → 0.81 | 22.7 → 0.0 | 0.1 → 1.7 | 1,115 → 1,551 (+39%) |
| ab-3059 | 0.0 → 0.0 | 0.84 → 0.81 | 11.0 → 1.1 | 0.0 → 0.0 | 5,448 → 6,181 (+13%) |
| ab-3640 | 31.3 → 30.8 | 0.84 → 0.64 | 13.8 → 1.5 | 0.0 → 0.0 | 1,357 → 2,093 (+54%) |
| ab-3674 | 10.9 → 4.0 | 0.80 → 0.65 | 18.8 → 1.3 | 0.0 → 0.0 | 750 → 1,099 (+47%) |
| ab-3677 | 19.4 → 18.9 | 0.91 → 0.87 | 20.9 → 0.0 | 0.4 → 0.5 | 10,399 → 13,263 (+28%) |
| ab-3832 | 14.5 → 9.8 | 0.77 → 0.74 | 5.2 → 2.2 | 0.0 → 0.0 | 1,012 → 1,121 (+11%) |
| ab-3837 | 19.1 → 18.3 | 0.82 → 0.77 | 2.9 → 0.0 | 0.0 → 0.0 | 2,221 → 2,772 (+25%) |
| ab-3846 | 7.3 → 6.2 | 0.78 → 0.68 | 11.2 → 0.0 | 0.0 → 0.0 | 1,317 → 1,672 (+27%) |
| ab-3876 | 30.5 → 27.3 | 0.90 → 0.81 | 20.5 → 0.2 | 17.1 → 7.0 | 877 → 1,400 (+60%) |
| ab-4038 | 44.4 → 38.8 | 0.84 → 0.81 | 7.0 → 1.8 | 0.0 → 0.0 | 3,518 → 4,312 (+23%) |
| ab-4039 | 44.8 → 40.5 | 0.82 → 0.80 | 1.5 → 0.0 | 0.0 → 0.0 | 2,436 → 2,830 (+16%) |
| ab-4273 | 20.9 → 19.7 | 0.79 → 0.65 | 20.3 → 0.0 | 0.0 → 0.0 | 546 → 791 (+45%) |
| ab-4396 | 2.6 → 0.1 | 0.84 → 0.71 | 36.8 → 0.0 | 0.0 → 0.0 | 1,124 → 1,816 (+62%) |
| ab-5717 | 30.8 → 30.8 | 0.80 → 0.52 | 17.7 → 0.0 | 0.0 → 0.0 | 833 → 1,366 (+64%) |
| ab-5731 | 39.2 → 29.4 | 0.88 → 0.85 | 20.1 → 10.5 | 0.0 → 0.0 | 1,003 → 1,199 (+20%) |
| ab-5911 | 50.2 → 48.0 | 0.83 → 0.68 | 15.2 → 0.1 | 0.0 → 0.0 | 1,162 → 1,874 (+61%) |
| ab-6056 | 0.3 → 0.1 | 0.88 → 0.85 | 3.1 → 0.0 | 0.0 → 0.0 | 18,845 → 20,921 (+11%) |
| ab-6057 | 1.3 → 1.1 | 0.90 → 0.75 | 49.3 → 0.1 | 0.5 → 0.0 | 2,785 → 5,982 (+115%) |
| ab-6183 | 15.6 → 15.6 | 0.83 → 0.78 | 20.8 → 0.0 | 0.0 → 0.0 | 701 → 888 (+27%) |
| ab-6262 | 55.0 → 51.6 | 0.90 → 0.82 | 16.9 → 0.3 | 1.7 → 0.1 | 1,627 → 2,694 (+66%) |
| ab-7183 | 12.8 → 10.7 | 0.91 → 0.59 | 21.7 → 0.1 | 0.0 → 0.0 | 23,535 → 47,108 (+100%) |
| ab-7986 | 2.1 → 0.0 | 0.85 → 0.75 | 26.6 → 0.4 | 0.0 → 0.0 | 1,174 → 1,706 (+45%) |
| ab-8456 | 1.2 → 0.0 | 0.76 → 0.67 | 0.9 → 0.0 | 0.0 → 0.0 | 571 → 676 (+18%) |
| ab-8571 | 15.9 → 11.5 | 0.77 → 0.74 | 9.7 → 0.0 | 0.0 → 0.0 | 764 → 846 (+11%) |
| ab-8585 | 2.0 → 0.0 | 0.81 → 0.75 | 1.0 → 0.0 | 16.6 → 0.0 | 724 → 841 (+16%) |
| ab-8717 | 34.1 → 32.0 | 0.89 → 0.73 | 18.1 → 0.4 | 0.6 → 12.1 | 8,729 → 17,124 (+96%) |
| ab-8920 | 9.7 → 3.1 | 0.77 → 0.72 | 12.9 → 0.0 | 0.0 → 0.0 | 671 → 845 (+26%) |
| ab-9267 | 2.2 → 0.2 | 0.90 → 0.88 | 13.9 → 0.1 | 0.0 → 0.0 | 6,789 → 8,392 (+24%) |
| ab-9485 | 37.2 → 31.7 | 0.88 → 0.79 | 27.7 → 3.3 | 6.1 → 7.1 | 2,953 → 4,897 (+66%) |
| ab-9745 | 28.9 → 0.0 | 0.91 → 0.67 | 12.5 → 2.1 | 18.7 → 0.0 | 265 → 763 (+188%) |
| pmc-1-20-1-woolworth-building-new-york-city | 9.7 → 8.9 | 0.90 → 0.85 | 28.5 → 0.1 | 0.0 → 0.0 | 19,494 → 28,127 (+44%) |
| pmc-1-464-is-a-typical-5-story-building-khrushchevka-1-464-5 | 29.7 → 23.3 | 0.91 → 0.86 | 31.5 → 0.0 | 0.0 → 0.0 | 4,989 → 7,396 (+48%) |
| pmc-10-early-19-century-style-rowhouses | 17.3 → 0.0 | 0.96 → 0.78 | 32.3 → 2.9 | 73.7 → 0.0 | 799 → 3,194 (+300%) |
| pmc-10-storey-russian-panel-house-khrushchyovka | 4.9 → 0.4 | 0.91 → 0.89 | 10.9 → 0.0 | 1.5 → 0.0 | 18,178 → 21,926 (+21%) |
| pmc-122-leadenhall-street-3162614 | 0.6 → 0.1 | 0.90 → 0.88 | 1.7 → 0.0 | 0.0 → 0.0 | 49,414 → 55,326 (+12%) |
| pmc-1890-train-station | 73.0 → 69.2 | 0.91 → 0.84 | 16.5 → 0.1 | 6.1 → 9.0 | 4,700 → 8,271 (+76%) |
| pmc-19-century-buildings-free-download | 95.9 → 95.5 | 0.89 → 0.87 | 9.5 → 0.0 | 0.0 → 0.0 | 4,698 → 5,230 (+11%) |
| pmc-1920s-brick-duplex-with-interior | 50.6 → 47.3 | 0.85 → 0.82 | 11.1 → 2.5 | 8.6 → 3.8 | 1,295 → 1,699 (+31%) |
| pmc-1930s-british-suburban-houses | 43.3 → 36.2 | 0.91 → 0.85 | 21.4 → 0.4 | 0.0 → 0.0 | 3,478 → 6,092 (+75%) |
| pmc-19th-century---brewery | 62.3 → 0.3 | 0.98 → 0.66 | 34.6 → 0.1 | 71.9 → 0.0 | 5,985 → 94,787 (+1484%) |
| pmc-20-s-cinema-minecraft-town-house | 33.2 → 30.9 | 0.89 → 0.82 | 21.5 → 0.7 | 6.5 → 4.8 | 3,332 → 5,264 (+58%) |
| pmc-3-level-1970s-split-level-house | 76.6 → 71.2 | 0.89 → 0.86 | 14.5 → 2.5 | 10.8 → 10.9 | 755 → 1,048 (+39%) |
| pmc-9-story-atypical-brick-building-yaroslavl-9 | 54.6 → 49.6 | 0.78 → 0.53 | 29.1 → 7.2 | 0.0 → 0.0 | 9,164 → 16,562 (+81%) |
| pmc-adelaide-hotel-toronto | 2.4 → 2.5 | 0.89 → 0.43 | 12.2 → 0.2 | 0.0 → 1.0 | 26,415 → 46,763 (+77%) |
| pmc-aircraft-hangar-private-jet-hangar-1-5-1-scale-1-1-scale-adaptable-full-interiors | 5.7 → 0.0 | 0.91 → 0.45 | 17.7 → 0.0 | 0.0 → 0.0 | 7,553 → 19,551 (+159%) |
| pmc-alabama-state-capitol-building | 47.9 → 46.8 | 0.89 → 0.87 | 15.5 → 0.5 | 0.0 → 0.0 | 9,344 → 11,698 (+25%) |
| pmc-anatolian-traditional-house-1 | 9.0 → 5.4 | 0.78 → 0.69 | 11.8 → 0.0 | 0.0 → 0.0 | 586 → 757 (+29%) |
| pmc-apartments-5646243 | 12.0 → 10.9 | 0.88 → 0.84 | 23.2 → 0.0 | 0.0 → 0.0 | 3,448 → 4,664 (+35%) |
| pmc-art-deco-multi-use-building | 6.7 → 4.1 | 0.88 → 0.84 | 12.6 → 0.0 | 0.0 → 0.0 | 5,550 → 6,938 (+25%) |
| pmc-art-deco-skyscraper-no-interior | 13.2 → 0.0 | 0.87 → 0.73 | 12.7 → 1.0 | 14.1 → 0.0 | 1,936 → 3,053 (+58%) |
| pmc-atat-rk-s-house-thessaloniki | 53.5 → 46.7 | 0.80 → 0.79 | 3.6 → 0.0 | 0.0 → 0.0 | 942 → 1,072 (+14%) |
| pmc-australian-queenslander-house | 34.6 → 19.4 | 0.86 → 0.84 | 9.2 → 3.2 | 0.0 → 0.0 | 4,022 → 4,894 (+22%) |
| pmc-austrian-house-schematic | 21.9 → 21.9 | 0.88 → 0.85 | 20.1 → 0.2 | 0.0 → 0.0 | 1,879 → 2,365 (+26%) |
| pmc-autozone | 67.9 → 49.1 | 0.85 → 0.84 | 5.1 → 0.1 | 0.0 → 0.0 | 4,061 → 5,579 (+37%) |
| pmc-autumn-cafe | 39.6 → 10.5 | 0.80 → 0.70 | 10.0 → 4.2 | 2.7 → 0.0 | 2,826 → 4,093 (+45%) |
| pmc-banco-de-mexico | 20.9 → 20.8 | 0.90 → 0.80 | 20.4 → 0.0 | 3.9 → 0.0 | 3,290 → 5,216 (+59%) |
| pmc-baroque-townhouse | 0.0 → 0.0 | 0.87 → 0.31 | 16.4 → 0.4 | 0.0 → 0.0 | 4,999 → 11,043 (+121%) |
| pmc-bioshop-small-modern-supermarket | 54.6 → 16.1 | 0.90 → 0.66 | 4.3 → 0.1 | 7.8 → 0.0 | 2,247 → 6,160 (+174%) |
| pmc-brick-tenement-house | 9.3 → 0.0 | 0.86 → 0.30 | 36.2 → 0.0 | 8.6 → 0.0 | 736 → 1,405 (+91%) |
| pmc-british-georgian-family-houses | 34.4 → 31.1 | 0.85 → 0.84 | 8.7 → 0.0 | 0.0 → 0.0 | 4,542 → 5,275 (+16%) |
| pmc-british-terraced-houses | 10.5 → 5.8 | 0.86 → 0.83 | 10.5 → 0.0 | 0.0 → 0.0 | 1,444 → 1,753 (+21%) |
| pmc-california-bungalow-by-ew-stillwell-1918 | 6.9 → 0.0 | 0.85 → 0.83 | 0.3 → 0.0 | 0.0 → 0.0 | 1,334 → 1,487 (+11%) |
| pmc-cape-cod-house---furnished | 58.0 → 53.2 | 0.88 → 0.83 | 18.9 → 13.2 | 0.0 → 0.0 | 1,326 → 1,877 (+42%) |
| pmc-cape-cod-house-2 | 47.9 → 40.3 | 0.86 → 0.82 | 11.6 → 6.2 | 0.0 → 0.0 | 1,405 → 1,923 (+37%) |
| pmc-casa-da-estremadura-estremadura-house | 41.3 → 26.5 | 0.80 → 0.81 | 0.0 → 0.0 | 0.0 → 0.0 | 396 → 491 (+24%) |
| pmc-chapel-in-the-cemetery-download | 87.6 → 93.1 | 0.93 → 0.85 | 6.8 → 0.3 | 0.0 → 0.1 | 4,018 → 2,488 (-38%) |
| pmc-chick-fil-a-restaurant-3831809 | 21.6 → 6.2 | 0.85 → 0.82 | 8.7 → 2.6 | 0.0 → 0.0 | 2,157 → 2,742 (+27%) |
| pmc-city-hall-council-building-of-kaiserthal-kaiserthal-project- | 18.2 → 9.7 | 0.88 → 0.75 | 21.7 → 1.8 | 0.0 → 0.0 | 9,263 → 16,816 (+82%) |
| pmc-clinic | 50.1 → 44.2 | 0.86 → 0.70 | 16.5 → 1.9 | 18.9 → 0.0 | 708 → 1,129 (+59%) |
| pmc-coleraine-church | 11.9 → 11.9 | 0.88 → 0.83 | 21.3 → 0.0 | 0.0 → 0.0 | 1,615 → 2,129 (+32%) |
| pmc-colonial-corivian | 86.2 → 76.1 | 0.88 → 0.88 | 6.2 → 1.8 | 0.0 → 0.0 | 8,436 → 22,526 (+167%) |
| pmc-colonial-goldeneye | 81.0 → 76.4 | 0.88 → 0.87 | 10.4 → 3.6 | 0.0 → 0.1 | 2,975 → 4,388 (+47%) |
| pmc-colonial-split-level | 65.0 → 58.1 | 0.83 → 0.81 | 5.2 → 4.8 | 0.1 → 0.0 | 1,113 → 1,396 (+25%) |
| pmc-concrete-hotel | 4.1 → 3.5 | 0.85 → 0.83 | 21.2 → 7.3 | 0.0 → 0.0 | 2,735 → 3,115 (+14%) |
| pmc-corner-hardwarwe-store-with-interior-and-download | 32.2 → 0.0 | 0.87 → 0.55 | 10.8 → 0.0 | 0.0 → 0.0 | 5,786 → 12,408 (+114%) |
| pmc-country-house-4011631 | 72.4 → 69.8 | 0.89 → 0.81 | 11.9 → 0.2 | 0.5 → 1.2 | 3,834 → 6,515 (+70%) |
| pmc-craftbucks-coffee---modern-build-ft-ena_ | 83.3 → 80.7 | 0.84 → 0.83 | 5.9 → 0.9 | 0.0 → 0.0 | 526 → 701 (+33%) |
| pmc-dormer-windowed-bungalow-mine-county-houses | 90.5 → 88.3 | 0.88 → 0.84 | 15.2 → 2.3 | 0.0 → 0.0 | 1,830 → 2,423 (+32%) |
| pmc-dutch-office | 9.6 → 6.7 | 0.87 → 0.85 | 0.7 → 0.2 | 0.0 → 0.0 | 11,508 → 13,490 (+17%) |
| pmc-dutch-style-houses-type-1 | 14.1 → 2.1 | 0.91 → 0.88 | 0.6 → 0.0 | 0.0 → 0.0 | 83,468 → 114,152 (+37%) |
| pmc-east-german-kt60-gym-hall-turnhalle | 38.8 → 18.8 | 0.81 → 0.80 | 2.9 → 0.0 | 0.0 → 0.0 | 1,022 → 1,054 (+3%) |
| pmc-empty-factory---schematic | 16.6 → 5.5 | 0.90 → 0.63 | 4.7 → 0.0 | 0.0 → 0.0 | 14,195 → 25,417 (+79%) |
| pmc-eunpyeong-public-library-seoul | 81.5 → 0.2 | 0.96 → 0.83 | 0.0 → 0.4 | 69.3 → 0.0 | 1,638 → 10,356 (+532%) |
| pmc-european-building-12-kaiserthal-project-by-tim0fei | 48.5 → 0.0 | 0.99 → 0.84 | 0.0 → 0.1 | 95.1 → 0.0 | 164 → 3,858 (+2252%) |
| pmc-european-fire-station-thevisual-play | 62.6 → 40.9 | 0.92 → 0.59 | 15.9 → 0.1 | 8.8 → 0.0 | 9,853 → 31,303 (+218%) |
| pmc-european-office-building-15-kaiserthal-project-by-tim0fei | 3.0 → 0.4 | 0.90 → 0.88 | 5.5 → 0.6 | 0.0 → 0.0 | 9,511 → 10,909 (+15%) |
| pmc-european-office-building-24-kaiserthal-project-by-tim0fei-in | 8.7 → 2.6 | 0.88 → 0.83 | 24.4 → 1.2 | 0.0 → 0.0 | 2,236 → 3,013 (+35%) |
| pmc-farmhouse-2 | 46.0 → 38.2 | 0.78 → 0.76 | 10.3 → 0.0 | 0.0 → 0.0 | 834 → 990 (+19%) |
| pmc-farmhouse-free-download | 37.7 → 37.6 | 0.83 → 0.80 | 12.1 → 17.3 | 0.1 → 0.0 | 918 → 1,026 (+12%) |
| pmc-fire-station---american-style-thevisual_play | 88.7 → 88.5 | 0.93 → 0.75 | 20.6 → 0.0 | 0.2 → 0.2 | 16,863 → 37,120 (+120%) |
| pmc-forest-fire-lookout-tower | 0.0 → 0.0 | 0.66 → 0.49 | 9.1 → 0.5 | 0.0 → 0.0 | 1,441 → 1,733 (+20%) |
| pmc-french-country-chateautmawok | 52.7 → 46.8 | 0.94 → 0.83 | 24.7 → 0.4 | 1.2 → 0.0 | 5,502 → 13,585 (+147%) |
| pmc-french-embassy-in-korea-seoul | 62.0 → 61.5 | 0.76 → 0.74 | 12.1 → 1.8 | 0.0 → 0.7 | 5,753 → 6,810 (+18%) |
| pmc-french-fire-station-thevisual_play | 79.6 → 70.5 | 0.96 → 0.87 | 18.2 → 0.2 | 3.2 → 14.4 | 65,255 → 188,447 (+189%) |
| pmc-greek-revival-plantation-house | 66.8 → 66.2 | 0.92 → 0.91 | 17.8 → 0.7 | 0.0 → 0.3 | 13,039 → 19,274 (+48%) |
| pmc-greenhouse-free-download | 24.6 → 8.1 | 0.85 → 0.80 | 19.5 → 7.4 | 0.0 → 0.0 | 322 → 435 (+35%) |
| pmc-grocery-store-3007139 | 85.7 → 83.4 | 0.93 → 0.92 | 4.0 → 0.0 | 1.1 → 1.3 | 17,979 → 23,521 (+31%) |
| pmc-hamburg-building-8-by-tim0fei | 5.4 → 0.0 | 0.93 → 0.90 | 24.8 → 2.5 | 0.0 → 0.0 | 7,898 → 10,609 (+34%) |
| pmc-hanok-traditional-korean-house | 58.1 → 52.7 | 0.90 → 0.85 | 7.0 → 0.8 | 0.1 → 0.1 | 3,100 → 4,387 (+42%) |
| pmc-herne-bay-townhouses | 3.8 → 0.0 | 0.90 → 0.82 | 8.8 → 2.1 | 22.7 → 0.0 | 1,455 → 2,174 (+49%) |
| pmc-high-rise-building-schematic | 0.0 → 0.0 | 0.95 → 0.90 | 44.4 → 0.5 | 0.0 → 0.0 | 10,550 → 20,114 (+91%) |
| pmc-hospital-2168539 | 0.2 → 0.0 | 0.97 → 0.90 | 60.0 → 0.2 | 0.0 → 0.0 | 18,420 → 53,919 (+193%) |
| pmc-hotel-berni | 61.1 → 60.6 | 0.80 → 0.76 | 1.8 → 0.2 | 0.4 → 1.0 | 9,735 → 11,260 (+16%) |
| pmc-hotel-piran-piran-slovenia | 7.8 → 6.2 | 0.90 → 0.87 | 14.7 → 0.6 | 0.0 → 0.0 | 12,581 → 15,142 (+20%) |
| pmc-house-32 | 32.1 → 23.8 | 0.80 → 0.76 | 9.7 → 0.9 | 0.0 → 0.0 | 373 → 457 (+23%) |
| pmc-japanese-minimalistic-condo | 8.3 → 7.3 | 0.78 → 0.63 | 21.2 → 0.2 | 0.0 → 0.0 | 4,061 → 5,578 (+37%) |
| pmc-kaiserthal-hotel-hamburg-building-10-kaiserthal-project-by-t | 8.5 → 0.1 | 0.91 → 0.90 | 7.5 → 4.5 | 0.0 → 0.0 | 7,054 → 8,212 (+16%) |
| pmc-kfc | 67.5 → 61.6 | 0.86 → 0.86 | 1.9 → 0.0 | 1.4 → 8.5 | 1,127 → 1,323 (+17%) |
| pmc-kimbell-art-museum-fort-worth-by-louis-kahn | 43.4 → 43.4 | 0.72 → 0.82 | 1.1 → 0.0 | 0.0 → 0.0 | 12,259 → 9,048 (-26%) |
| pmc-late-1800s-brick-house | 32.8 → 30.1 | 0.89 → 0.84 | 14.7 → 0.1 | 0.0 → 0.0 | 4,086 → 5,417 (+33%) |
| pmc-library-bibliothek | 27.3 → 25.6 | 0.92 → 0.86 | 30.9 → 0.0 | 0.0 → 0.0 | 6,167 → 10,434 (+69%) |
| pmc-lidl-supermarket | 39.5 → 27.6 | 0.87 → 0.85 | 8.0 → 1.0 | 0.0 → 0.0 | 4,918 → 5,812 (+18%) |
| pmc-lisbon-courtyardptio-de-lisboa | 50.5 → 0.1 | 0.84 → 0.81 | 4.0 → 0.4 | 0.2 → 0.0 | 3,791 → 6,032 (+59%) |
| pmc-low-cost-motel | 15.0 → 11.6 | 0.86 → 0.74 | 25.2 → 0.3 | 0.0 → 0.0 | 3,080 → 4,971 (+61%) |
| pmc-luxury-apartment | 1.2 → 0.0 | 0.94 → 0.91 | 21.0 → 0.0 | 0.0 → 0.0 | 46,939 → 64,978 (+38%) |
| pmc-luxury-imhydrox | 0.0 → 0.0 | 0.92 → 0.70 | 13.9 → 0.0 | 0.0 → 0.0 | 40,357 → 57,710 (+43%) |
| pmc-mediterranean-house-schematic | 21.7 → 21.7 | 0.88 → 0.86 | 16.1 → 5.3 | 0.0 → 0.0 | 4,069 → 4,721 (+16%) |
| pmc-medium-warehouse | 26.3 → 26.1 | 0.91 → 0.57 | 21.9 → 0.0 | 0.0 → 0.0 | 7,172 → 15,812 (+120%) |
| pmc-mitsu---office-building-thevisual_play | 46.8 → 45.4 | 0.93 → 0.90 | 30.4 → 0.3 | 0.0 → 0.0 | 14,166 → 21,093 (+49%) |
| pmc-modern-american-craftsman-style-house | 38.0 → 32.2 | 0.84 → 0.77 | 18.1 → 0.0 | 0.0 → 0.0 | 1,355 → 2,074 (+53%) |
| pmc-modern-apartments-building | 63.1 → 61.1 | 0.78 → 0.77 | 10.3 → 3.8 | 0.0 → 0.0 | 3,333 → 3,907 (+17%) |
| pmc-modern-city-building-with-baker-pharmacy-and-surgery | 26.2 → 1.9 | 0.88 → 0.86 | 6.7 → 0.4 | 0.0 → 0.0 | 4,433 → 5,821 (+31%) |
| pmc-modern-fuel-station-gas-station | 42.9 → 28.9 | 0.80 → 0.75 | 10.7 → 1.4 | 0.0 → 0.0 | 3,661 → 5,204 (+42%) |
| pmc-modern-lake-house-interiors-download | 51.5 → 48.5 | 0.89 → 0.82 | 24.5 → 6.4 | 8.1 → 5.3 | 3,234 → 4,771 (+48%) |
| pmc-modern-office-highrise | 0.0 → 0.0 | 0.84 → 0.80 | 4.9 → 0.0 | 0.0 → 0.0 | 6,149 → 7,370 (+20%) |
| pmc-modern-police-department-station-2 | 60.9 → 59.0 | 0.88 → 0.86 | 11.6 → 0.0 | 0.5 → 0.7 | 10,769 → 13,984 (+30%) |
| pmc-modern-school | 84.8 → 80.4 | 0.91 → 0.89 | 5.4 → 0.2 | 0.0 → 0.6 | 15,571 → 22,810 (+46%) |
| pmc-modern-warehouse-2128641 | 66.1 → 65.5 | 0.91 → 0.73 | 3.9 → 0.0 | 0.0 → 0.0 | 9,480 → 20,288 (+114%) |
| pmc-modern-white-beachlake-house | 35.8 → 29.9 | 0.87 → 0.83 | 15.7 → 0.2 | 0.0 → 0.6 | 3,614 → 4,760 (+32%) |
| pmc-moscow-block-of-flats-marino-district-pererva-st | 4.3 → 4.1 | 0.93 → 0.91 | 19.5 → 0.4 | 0.0 → 0.0 | 14,729 → 18,419 (+25%) |
| pmc-new-york-historical-building-2 | 0.1 → 0.0 | 0.85 → 0.81 | 15.0 → 0.1 | 0.0 → 0.0 | 7,060 → 8,372 (+19%) |
| pmc-office-and-apartment-building | 10.8 → 8.0 | 0.88 → 0.79 | 12.1 → 0.1 | 0.0 → 0.0 | 6,178 → 10,027 (+62%) |
| pmc-office-building-4034242 | 4.8 → 3.2 | 0.80 → 0.74 | 15.0 → 0.0 | 0.0 → 0.0 | 15,972 → 19,537 (+22%) |
| pmc-old-state-house | 18.3 → 8.1 | 0.80 → 0.81 | 4.4 → 0.1 | 0.0 → 0.0 | 1,647 → 1,724 (+5%) |
| pmc-old-tenement-house | 0.0 → 0.0 | 0.84 → 0.39 | 18.4 → 0.0 | 0.0 → 0.0 | 4,711 → 7,768 (+65%) |
| pmc-old-townhall-2 | 20.3 → 20.6 | 0.88 → 0.82 | 27.6 → 0.2 | 0.0 → 0.0 | 4,361 → 6,227 (+43%) |
| pmc-old-warehouse | 50.4 → 47.9 | 0.94 → 0.84 | 27.5 → 0.0 | 26.8 → 0.0 | 3,805 → 8,871 (+133%) |
| pmc-oldnew-synagogue | 14.9 → 1.0 | 0.83 → 0.60 | 7.0 → 0.1 | 0.0 → 0.0 | 1,131 → 1,974 (+75%) |
| pmc-orona-hq---traditional-skyscraper | 37.2 → 32.6 | 0.92 → 0.89 | 12.0 → 1.9 | 4.5 → 1.3 | 19,469 → 28,191 (+45%) |
| pmc-oxford-place | 1.5 → 0.2 | 0.88 → 0.82 | 13.5 → 0.9 | 0.8 → 0.0 | 22,884 → 29,034 (+27%) |
| pmc-parish-church-of-saint-saviour | 5.2 → 0.4 | 0.85 → 0.75 | 28.3 → 0.0 | 0.0 → 0.0 | 1,444 → 2,117 (+47%) |
| pmc-parish-church-of-st-elizabeth-ko-ci-farny-w-el-biety | 0.0 → 0.0 | 0.93 → 0.91 | 20.1 → 0.1 | 0.0 → 0.0 | 39,799 → 51,063 (+28%) |
| pmc-pine-river-grocery | 47.7 → 35.9 | 0.84 → 0.78 | 2.9 → 0.9 | 0.0 → 0.0 | 1,687 → 2,550 (+51%) |
| pmc-plattenbau-kk-kg-72-144-no-1 | 36.2 → 25.7 | 0.84 → 0.82 | 9.1 → 1.4 | 0.0 → 0.0 | 3,780 → 4,305 (+14%) |
| pmc-post-office-19th-century-australia | 17.8 → 0.5 | 0.84 → 0.81 | 10.5 → 0.5 | 0.0 → 0.0 | 3,644 → 4,333 (+19%) |
| pmc-realistic-american-style-suburban-house-2 | 70.1 → 66.2 | 0.95 → 0.91 | 14.8 → 0.3 | 16.4 → 2.9 | 2,138 → 3,616 (+69%) |
| pmc-red-clay-barn | 34.4 → 25.8 | 0.85 → 0.60 | 9.8 → 0.0 | 0.0 → 0.0 | 1,880 → 3,964 (+111%) |
| pmc-red-horse-barn | 48.9 → 48.6 | 0.89 → 0.69 | 12.4 → 0.0 | 0.0 → 0.0 | 9,699 → 19,591 (+102%) |
| pmc-replica-post-office | 97.2 → 96.7 | 0.88 → 0.87 | 0.1 → 0.0 | 0.0 → 0.0 | 6,363 → 7,212 (+13%) |
| pmc-richelieu-contemporary | 94.3 → 84.4 | 0.99 → 0.58 | 0.0 → 1.1 | 67.4 → 0.0 | 354 → 4,162 (+1076%) |
| pmc-rural-motel-l-shaped | 81.6 → 79.8 | 0.92 → 0.90 | 15.2 → 2.3 | 0.0 → 0.0 | 3,526 → 4,997 (+42%) |
| pmc-schule | 51.2 → 46.9 | 0.92 → 0.87 | 34.5 → 2.6 | 1.0 → 0.6 | 2,400 → 3,988 (+66%) |
| pmc-seaside-restaurant | 40.1 → 44.2 | 0.77 → 0.65 | 6.1 → 1.0 | 0.0 → 0.0 | 1,351 → 1,814 (+34%) |
| pmc-semi-detached-house | 69.5 → 67.8 | 0.83 → 0.81 | 10.5 → 0.0 | 0.0 → 0.0 | 1,086 → 1,308 (+20%) |
| pmc-semi-detached-house-3790771 | 89.6 → 88.5 | 0.88 → 0.83 | 26.3 → 0.7 | 0.0 → 0.1 | 1,715 → 3,094 (+80%) |
| pmc-seven-eleven | 64.7 → 35.6 | 0.87 → 0.79 | 4.2 → 0.3 | 0.0 → 0.0 | 1,819 → 3,284 (+81%) |
| pmc-seviat-city---ashton-hotel | 20.4 → 16.4 | 0.93 → 0.90 | 10.1 → 0.1 | 0.9 → 3.0 | 38,726 → 56,020 (+45%) |
| pmc-simple-brick-corner-apartment-and-pet-store | 0.0 → 0.0 | 0.84 → 0.81 | 14.0 → 0.0 | 0.0 → 0.0 | 2,110 → 2,450 (+16%) |
| pmc-sinaia-monastery-romania | 2.2 → 0.0 | 0.85 → 0.81 | 10.2 → 0.0 | 0.0 → 0.0 | 4,211 → 4,929 (+17%) |
| pmc-skyscraper---candler-building-nyc | 6.8 → 0.7 | 0.91 → 0.82 | 31.3 → 0.0 | 0.2 → 0.0 | 9,272 → 16,196 (+75%) |
| pmc-small-craftsman-house---exterior-amp-landscape-only | 83.2 → 78.4 | 0.88 → 0.75 | 22.3 → 0.0 | 0.0 → 0.0 | 1,318 → 2,908 (+121%) |
| pmc-small-modern-hotel-near-marina-schematic | 38.8 → 34.4 | 0.77 → 0.76 | 10.9 → 7.1 | 0.0 → 7.3 | 4,354 → 4,947 (+14%) |
| pmc-small-office | 16.4 → 14.4 | 0.80 → 0.77 | 10.8 → 0.0 | 0.3 → 0.5 | 657 → 738 (+12%) |
| pmc-small-skyscraper-download | 7.8 → 4.6 | 0.87 → 0.84 | 15.3 → 0.7 | 0.0 → 0.0 | 12,312 → 15,226 (+24%) |
| pmc-small-suburban-house-furnished | 69.7 → 65.3 | 0.89 → 0.87 | 15.9 → 2.7 | 0.0 → 0.5 | 1,300 → 1,809 (+39%) |
| pmc-small-train-station | 74.5 → 62.7 | 0.76 → 0.72 | 6.8 → 0.0 | 0.0 → 0.0 | 1,008 → 1,682 (+67%) |
| pmc-solar-da-famlia-montalvomanor-house-of-montalvo-family | 58.7 → 56.4 | 0.91 → 0.87 | 8.7 → 0.0 | 0.1 → 0.1 | 4,178 → 6,015 (+44%) |
| pmc-soviet-16-story-panel-building-i-491a-16-491 | 1.3 → 0.2 | 0.92 → 0.90 | 14.6 → 3.3 | 0.0 → 0.0 | 25,889 → 30,385 (+17%) |
| pmc-soviet-building-stalinka-stalin-s-apartments | 27.1 → 10.4 | 0.93 → 0.91 | 9.4 → 0.0 | 0.0 → 0.0 | 15,326 → 19,303 (+26%) |
| pmc-spanish-colonial-house---philippines | 39.6 → 37.7 | 0.88 → 0.84 | 27.7 → 0.2 | 0.0 → 0.0 | 1,144 → 1,714 (+50%) |
| pmc-spanish-villa | 49.0 → 47.8 | 0.91 → 0.85 | 22.5 → 0.0 | 0.0 → 0.0 | 5,709 → 9,154 (+60%) |
| pmc-stone-brick-bungalow-mine-county-houses | 90.6 → 88.8 | 0.92 → 0.91 | 21.8 → 1.4 | 0.8 → 9.6 | 1,356 → 1,941 (+43%) |
| pmc-suburban-auto-repair-garage | 70.7 → 62.6 | 0.88 → 0.82 | 4.8 → 2.6 | 9.7 → 0.0 | 1,123 → 1,738 (+55%) |
| pmc-suburban-gas-station | 52.0 → 40.8 | 0.81 → 0.76 | 0.3 → 0.0 | 18.9 → 0.0 | 1,638 → 2,251 (+37%) |
| pmc-suburban-house-lune | 32.4 → 29.2 | 0.83 → 0.80 | 12.8 → 2.0 | 0.0 → 0.1 | 1,513 → 1,861 (+23%) |
| pmc-suburban-pharmacy | 78.8 → 73.0 | 0.82 → 0.82 | 3.2 → 0.1 | 20.4 → 9.9 | 2,287 → 2,951 (+29%) |
| pmc-suburban-strip-mall-with-grocery-store | 69.5 → 61.6 | 0.89 → 0.86 | 5.8 → 0.0 | 1.4 → 2.4 | 8,935 → 11,851 (+33%) |
| pmc-swiss-federal-palace-bundeshaus | 5.8 → 5.0 | 0.92 → 0.67 | 19.4 → 0.1 | 0.0 → 0.0 | 25,033 → 65,071 (+160%) |
| pmc-tenement-classic-apartment-building | 9.6 → 0.0 | 0.85 → 0.80 | 22.3 → 0.0 | 0.0 → 0.0 | 2,201 → 3,007 (+37%) |
| pmc-thatched-cottage | 27.6 → 27.0 | 0.81 → 0.81 | 16.5 → 0.0 | 0.0 → 0.0 | 462 → 560 (+21%) |
| pmc-the-meadowbrook-50-s-styled-ranch-house | 46.2 → 1.4 | 0.84 → 0.77 | 7.7 → 0.0 | 0.0 → 0.0 | 421 → 644 (+53%) |
| pmc-the-office-building | 40.0 → 27.1 | 0.89 → 0.88 | 4.2 → 0.0 | 0.0 → 0.0 | 7,142 → 7,627 (+7%) |
| pmc-the-oldmeldrum-distillery | 50.1 → 32.7 | 0.87 → 0.79 | 10.4 → 1.2 | 0.0 → 0.0 | 11,404 → 18,096 (+59%) |
| pmc-three-bay-vietnamese-traditional-house-northern-delta-red-ri | 90.3 → 84.0 | 0.84 → 0.87 | 5.9 → 2.0 | 0.0 → 0.5 | 1,359 → 2,204 (+62%) |
| pmc-three-little-german-townhouses-with-stores-schematic-downloa | 55.2 → 47.9 | 0.84 → 0.83 | 3.5 → 0.1 | 0.2 → 2.5 | 1,728 → 2,150 (+24%) |
| pmc-toronto-fairmont-royal-york-java-1-17-1 | 2.2 → 1.5 | 0.92 → 0.89 | 22.3 → 0.0 | 0.0 → 0.0 | 36,199 → 47,877 (+32%) |
| pmc-toscana-house | 57.0 → 53.9 | 0.83 → 0.74 | 34.0 → 0.0 | 0.0 → 0.0 | 252 → 421 (+67%) |
| pmc-traditional-bungalow-3158437 | 84.6 → 82.3 | 0.94 → 0.90 | 12.1 → 0.6 | 0.0 → 0.0 | 5,654 → 7,995 (+41%) |
| pmc-traditional-house-victorian-suburban-style-4049624 | 39.8 → 39.6 | 0.89 → 0.87 | 11.5 → 0.4 | 0.0 → 0.0 | 4,941 → 5,829 (+18%) |
| pmc-traditional-mosque | 16.8 → 7.8 | 0.86 → 0.63 | 9.5 → 0.6 | 0.0 → 0.0 | 23,235 → 43,640 (+88%) |
| pmc-traditional-tbs-app-craftsman-house | 35.8 → 35.3 | 0.82 → 0.71 | 3.3 → 1.3 | 0.0 → 0.0 | 757 → 1,041 (+38%) |
| pmc-triple-gable-tudor-townhouse-6884362 | 15.1 → 14.6 | 0.84 → 0.74 | 25.7 → 0.1 | 0.0 → 0.0 | 1,350 → 1,998 (+48%) |
| pmc-tropico-5-creamery | 42.0 → 30.6 | 0.89 → 0.84 | 13.9 → 0.3 | 0.0 → 0.0 | 4,694 → 6,444 (+37%) |
| pmc-tropico-5-newspaper | 31.0 → 26.8 | 0.88 → 0.85 | 11.8 → 0.0 | 0.2 → 0.1 | 3,455 → 4,323 (+25%) |
| pmc-tropico-5-police-station | 35.3 → 29.7 | 0.88 → 0.81 | 21.2 → 0.1 | 0.0 → 0.0 | 2,712 → 4,152 (+53%) |
| pmc-tropico-5-restaurant | 36.6 → 27.8 | 0.81 → 0.69 | 5.8 → 1.5 | 0.0 → 0.0 | 1,655 → 2,591 (+57%) |
| pmc-tudor-style-house-with-floor-plans | 52.9 → 85.9 | 0.84 → 0.93 | 14.4 → 1.4 | 0.0 → 6.8 | 4,508 → 2,008 (-55%) |
| pmc-tudor-townhouse-industrial-england | 29.0 → 7.8 | 0.86 → 0.74 | 1.5 → 0.0 | 15.5 → 0.0 | 788 → 1,196 (+52%) |
| pmc-unite-d-habitation-nantes-reze-by-le-corbusier | 5.6 → 0.1 | 0.91 → 0.83 | 26.3 → 10.0 | 3.5 → 0.0 | 11,801 → 19,180 (+63%) |
| pmc-universit-de-montr-al-campus-at-laval | 47.7 → 42.2 | 0.89 → 0.86 | 4.2 → 0.1 | 1.4 → 0.8 | 20,478 → 25,510 (+25%) |
| pmc-upscale-suburban-home-mine-county-houses | 88.8 → 87.8 | 0.87 → 0.86 | 13.8 → 1.3 | 0.0 → 0.4 | 2,101 → 2,587 (+23%) |
| pmc-victorian-courthouse | 0.6 → 0.4 | 0.92 → 0.89 | 26.9 → 0.6 | 0.0 → 0.0 | 7,832 → 10,783 (+38%) |
| pmc-victorian-four-square-duplex | 21.0 → 20.6 | 0.89 → 0.87 | 15.6 → 0.0 | 0.0 → 0.0 | 2,301 → 2,801 (+22%) |
| pmc-victorian-house-4275811 | 97.3 → 96.0 | 0.82 → 0.94 | 12.8 → 0.5 | 0.0 → 0.0 | 2,048 → 4,647 (+127%) |
| pmc-victorian-house-5 | 15.8 → 15.0 | 0.87 → 0.84 | 14.0 → 1.2 | 0.0 → 0.0 | 3,296 → 3,881 (+18%) |
| pmc-victorian-necrodusk | 0.1 → 0.0 | 0.80 → 0.72 | 14.0 → 0.7 | 0.0 → 0.0 | 1,067 → 1,427 (+34%) |
| pmc-victorian-queen-anne-style---carson-mansion | 0.0 → 0.0 | 0.86 → 0.83 | 16.1 → 0.1 | 0.0 → 0.0 | 4,202 → 5,153 (+23%) |
| pmc-villa-durango | 54.8 → 52.5 | 0.87 → 0.64 | 18.1 → 0.1 | 0.0 → 0.6 | 4,976 → 10,340 (+108%) |
| pmc-villa-plinius-historically-correct-peristylium-styled-house | 72.2 → 67.0 | 0.92 → 0.90 | 20.4 → 4.4 | 2.4 → 2.0 | 1,738 → 2,560 (+47%) |
| pmc-villa-shodhan-ahmedabad-by-le-corbusier | 4.6 → 0.7 | 0.84 → 0.72 | 24.1 → 0.9 | 0.0 → 0.0 | 3,059 → 4,534 (+48%) |
| pmc-warehouse-4154583 | 5.7 → 0.0 | 0.88 → 0.49 | 32.6 → 0.0 | 0.0 → 0.0 | 1,384 → 3,637 (+163%) |
| pmc-warehouse-5752980 | 16.0 → 5.9 | 0.70 → 0.66 | 6.2 → 0.7 | 0.0 → 0.0 | 2,149 → 2,668 (+24%) |
| pmc-warehouse-factory-hall | 8.0 → 7.7 | 0.91 → 0.44 | 39.3 → 0.0 | 0.0 → 0.0 | 5,808 → 15,764 (+171%) |
| pmc-warehouse-row | 82.5 → 81.1 | 0.92 → 0.90 | 19.1 → 1.4 | 0.6 → 3.2 | 4,064 → 5,133 (+26%) |
| pmc-warehouse-two-versions | 6.9 → 4.4 | 0.84 → 0.72 | 16.0 → 0.0 | 0.0 → 0.0 | 3,670 → 4,803 (+31%) |
| pmc-warehouses-6539779 | 46.4 → 0.7 | 0.85 → 0.54 | 7.8 → 0.0 | 0.0 → 0.0 | 3,002 → 6,127 (+104%) |
| pmc-west-virginia-state-capitol-replica | 52.9 → 52.0 | 0.92 → 0.71 | 14.3 → 0.0 | 0.0 → 0.0 | 36,368 → 58,019 (+60%) |
| pmc-white-mid-rise-residential-building | 36.7 → 31.6 | 0.89 → 0.84 | 21.1 → 0.0 | 0.4 → 0.0 | 6,039 → 9,151 (+52%) |
| pmc-wilhelminian-style-building-with-rental-apartments-and-backy | 59.1 → 53.0 | 0.90 → 0.84 | 19.1 → 0.2 | 3.7 → 5.1 | 4,207 → 6,903 (+64%) |
| pmc-xix-century-factory | 0.0 → 0.0 | 0.89 → 0.86 | 14.7 → 0.2 | 0.0 → 0.0 | 6,794 → 8,134 (+20%) |
| pmc-zhytomyr-bus-station | 0.5 → 0.0 | 0.87 → 0.85 | 9.1 → 0.4 | 0.0 → 0.0 | 3,008 → 3,432 (+14%) |

## Flagged for review

All three flags can be filtered in the gallery.

**`ground_fallback`** (4): more than 15% of the kept blocks would be buried, so there is no ground.

`pmc-clinic`, `pmc-european-fire-station-thevisual-play`, `pmc-old-warehouse`, `pmc-richelieu-contemporary`

**`multi_building`** (10): more than one large separate mass; every building is kept in one target.

`ab-12498`, `pmc-9-story-atypical-brick-building-yaroslavl-9`, `pmc-bj-rk-ngen-semi-detached-houses`, `pmc-colonial-corivian`, `pmc-greek-revival-plantation-house`, `pmc-grocery-store-3007139`, `pmc-larsen-nielsen-prefab-system-hungary-panel-house-czech-versi`, `pmc-modern-school`, `pmc-seven-eleven`, `pmc-warehouse-row`

**`large_opening`** (69): more than 30% of the deep covered air is exterior: halls, garages, open ground floors, hollow facade shells.

`ab-11479`, `ab-11663`, `ab-13068`, `ab-13254`, `ab-13526`, `ab-13864`, `ab-2279`, `ab-2282`, `ab-2988`, `ab-3640`, `ab-3674`, `ab-4273`, `ab-5717`, `ab-5911`, `ab-6262`, `ab-7183`, `ab-8717`, `ab-9745`, `pmc-1890-train-station`, `pmc-19th-century---brewery`, `pmc-adelaide-hotel-toronto`, `pmc-aircraft-hangar-private-jet-hangar-1-5-1-scale-1-1-scale-adaptable-full-interiors`, `pmc-baroque-townhouse`, `pmc-bioshop-small-modern-supermarket`, `pmc-brick-tenement-house`, `pmc-chapel-in-the-cemetery-download`, `pmc-city-hall-council-building-of-kaiserthal-kaiserthal-project-`, `pmc-colonial-corivian`, `pmc-corner-hardwarwe-store-with-interior-and-download`, `pmc-empty-factory---schematic`, `pmc-european-fire-station-thevisual-play`, `pmc-fire-station---american-style-thevisual_play`, `pmc-forbes-tower-mall-and-office-building-full-interior`, `pmc-forest-fire-lookout-tower`, `pmc-french-country-chateautmawok`, `pmc-french-embassy-in-korea-seoul`, `pmc-french-fire-station-thevisual_play`, `pmc-greek-revival-plantation-house`, `pmc-hotel-berni`, `pmc-hotel-piran-piran-slovenia`, `pmc-japanese-dusan`, `pmc-luxury-imhydrox`, `pmc-medium-warehouse`, `pmc-modern-fuel-station-gas-station`, `pmc-modern-school`, `pmc-modern-warehouse-2128641`, `pmc-old-tenement-house`, `pmc-old-warehouse`, `pmc-oldnew-synagogue`, `pmc-oxford-place`, `pmc-pig-and-cow-barn`, `pmc-red-clay-barn`, `pmc-red-horse-barn`, `pmc-richelieu-contemporary`, `pmc-seaside-restaurant`, `pmc-small-modern-hotel-near-marina-schematic`, `pmc-small-train-station`, `pmc-suburban-gas-station`, `pmc-swiss-federal-palace-bundeshaus`, `pmc-traditional-mosque`, `pmc-tropico-5-restaurant`, `pmc-villa-durango`, `pmc-villa-shodhan-ahmedabad-by-le-corbusier`, `pmc-warehouse-4154583`, `pmc-warehouse-5752980`, `pmc-warehouse-factory-hall`, `pmc-warehouse-two-versions`, `pmc-warehouses-6539779`, `pmc-west-virginia-state-capitol-replica`
