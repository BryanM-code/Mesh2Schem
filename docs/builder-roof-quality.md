# Builder roof quality — September 30, 2026

Builder targets now use connected roof-plane patterns, retain the backing exposed
through partial blocks, and preserve narrow supported posts as fences or walls when
the source provides enough evidence. Versions: **mcvoxel 0.4.0**, **synth-2.2.0**,
**mcvox-v2**, Minecraft **1.20.1 / DataVersion 3465**.

The eight-source comparison has 78 → 0 missing caps, 8,599 → 0 open course interfaces,
and 5 → 0 unsupported caps in the measured roof footprint. These are topology
indicators, not a claim that all architectural details are correct. Mansion has a
material regression on small caps; windows, doors and some site features remain poor.

All implementation changes remain uncommitted. No commit, push, tag, publication,
training run or large dataset generation was performed. Earlier pilots were preserved.

## Inspect the evidence

- [Eight-source comparison](../data/synthetic/builder-roof-v5-20260930/report.md)
- [Tools Shop front/back and diagnostics](../data/synthetic/builder-roof-v5-20260930/samples/21f6babf30ed4298ad9134dbcaae4940/comparison.png)
- [Temple of Rome front/back and diagnostics](../data/synthetic/builder-roof-v5-20260930/samples/aa8a074fae9a49f0a1b26b87ac826802/comparison.png)
- [11-fixture gallery](../data/synthetic/builder-roof-v5-20260930/fixtures/report.md)
- [Per-source numerical results](../data/synthetic/builder-roof-v5-20260930/comparison.json)
- [Aggregate counts](../data/synthetic/builder-roof-v5-20260930/summary.json)
- [Visual review](../data/synthetic/builder-roof-v5-20260930/visual-review.md)

Each comparison contains source mesh, previous builder, new builder, geometric target,
previous/new back views, shape colours and a roof-region/pitch map. All target views
use the same camera and measured Minecraft palette. Source meshes retain their own
materials. Region details and repairs are in each sample's `manifest.json`; the arrays
are in `roof_plan.npz`. The review UI also exposes the region map and fence/wall legend.

## Pipeline and causes of the previous holes

The shared voxel core performs conservative triangle rasterisation, exterior fill and
an axis-ray enclosure heuristic, site/component cleanup, fine-resolution octant shape
classification, surface-repeat removal, exposed colour sampling and block assignment.
Builder mode then reconstructs top caps and harmonises roof materials. Geometric mode
remains the alternate target before the roof reconstruction.

Three separate mechanisms created the visible strips and gaps:

1. The previous roof planner required enough mesh hits inside an already surviving
   top voxel. Missing or displaced caps could therefore disappear from its footprint.
   Coarse slope classes and a circular phase did not impose a common canonical lattice.
2. Immediate-below support was insufficient between courses at different heights.
   More subtly, cube-only shell culling omitted full backing cells visible through
   slabs and stairs. A solid voxel volume could export a schematic with visible gaps.
3. Enclosure rays mistook separated columns for walls, filling porches/colonnades.
   Surface-repeat cleanup could delete thin posts, including duplicate cells created
   when one sub-block post crossed a coarse grid boundary.

Material sampling at eaves also mixed roof and facade colours. Whole-roof colour
groups could merge different planes or towers. During this change, image review caught
two further regressions: mixed slope magnitudes were mistaken for ridge seams on the
Temple, and thin parapets were blended into lower flat roofs or cut down when rays
missed their narrow tops. Both now have targeted guards and regression coverage.

## Roof construction rules

Upward mesh rays measure source height and triangle gradient at four samples per block
edge. Heights are extrapolated to block centres, so clipped eave samples do not change
the phase. Connected regions require matching slope class, cardinal direction,
nearest angular pitch and local height continuity. Distinct flat heights separate
terraces. Opposing/cardinal slope mixtures explicitly identify seams; changing only
the magnitude on one slope is not a ridge.

| Horizontal : vertical | Pitch | Repeated construction |
|---|---:|---|
| 1:1 | 45° | Stair every column, one-block rise, full backing |
| 2:1 | 26.57° | Repeated bottom-slab/full courses on a half-block lattice |
| 3:1 | 18.43° | Repeated slab/full half-steps over three columns |
| 1:2 | 63.43° | Stair every column, two-block rise, full riser backing |

A single region intercept sets the phase; upstream distance is measured from the
region's low end. This makes decisions deterministic under rotation and mirroring.
When fitting a canonical pitch would deviate by more than 1.5 blocks over a long plane,
a rational approximation with denominator at most eight follows the source pitch.
The Temple uses an 8:5 horizontal:vertical fit. The manifest records this explicitly.

Mesh-covered missing caps can be restored near a continuous anchored roof. No source
samples means no invented roof. Glass-dominated caps stay geometric. Only a shallow
band, at most two cells below a raised cap, is filled; adjacent courses receive backing
through their lower cap height. This does not create floor-to-roof pillars under eaves.
Backing is retained during shell export and padding. There is no new whole-building
binary closing; the comparison preserves the clean-source `close_gaps=False` setting.

Ridge/hip/valley seam cells receive full caps; stair corners are resolved on the complete
grid. Narrow flat strips retain their geometric thickness. The highest flat hit protects
a raised parapet sharing a coarse cell with a lower roof; existing tall vertical geometry
also vetoes a lower-roof rewrite when the upward sampler misses its top.

Roof colours are sampled from upward source hits. Harmonisation is scoped to a connected
plane and source material ID, with the nearest compatible slab/stair material. Separate
colours baked into one texture/material can still be confused, especially on tiny planes.

## Contextual details and representation

Posts require narrow fine-grid X/Z width, multiple vertically continuous cells,
separation from broad wall mass, a base and an overhead support. A post straddling a
coarse boundary is consolidated only if its measured fine cross-section is still narrow
and the surviving run passes the support rules. Broad walls, isolated fragments,
floaters and explicitly named metal/glass/vegetation fail the conversion.

Named wood or masonry chooses the corresponding block family. Unknown material names
need a close palette-colour match; explicit incompatible names take precedence. An
explicit source caption can resolve unnamed wooden construction without supplying
geometry. Tools Shop uses that hint and emits 17 fence cells. Thin raised sheets named
as railings, balustrades or parapets can qualify without overhead support. Generic
roof edges do not become walls. Existing fine occupancy continues to select slabs
for thin canopies, ledges, balconies and cornices.

The enclosure fix removes isolated compact column cross-sections and narrow tangential
spans from horizontal wall blockers. It does not remove columns from occupancy.
Broad enclosed rooms still fill. The short-building quality gate now measures geometric
enclosure independently of the larger shell needed to export partial-block backing.

Vocabulary v2 appends **fence = 8** and **wall = 9**. Shape IDs 0–7, all 120 material
IDs, air/ignore values, XYZ/YZX conventions and input/target frame rules remain unchanged.
There are ten wooden fence variants and fifteen masonry wall variants. Geometry-only
`block_shapes` codes stay unchanged; contextual assignment creates the new target labels.

`finish_states` derives stair corners and fence/wall connections after full-grid assembly,
including across a 64-cell window boundary. Straight wall runs omit their central post
when appropriate; overhead coverage controls arm height and continuing posts. States
are validated before schematic write and readback. Every decoded shape/material
combination is checked against a separate 1.20.1 block/property fixture, and all 2,590
emitted fence/wall connection combinations round-trip through `.schem`.

Old targets remain readable. Consumers of new targets must use the saved v2 vocabulary
and ten shape classes; an old eight-class checkpoint cannot predict fences/walls without
adaptation. No model was trained or migrated. Trapdoors, panes, actual door blocks and
upside-down stairs are deferred until evidence and a complete representation justify them.

## Numerical and visual results

The previous targets are the existing `pilot-v4` schematics, re-rendered with the same
renderer as the new targets. Source records, recorded seeds, cached GLBs and scale
settings are reused. All eight old/new frames also happened to match exactly; the
comparison code aligns origins when they do not. Every new input/target pair passed
the unchanged label/frame contract and schematic roundtrip.

| Indicator | Previous | New |
|---|---:|---:|
| Measured roof columns | 16,510 | 16,510 |
| Missing cap indicators | 78 | 0 |
| Open adjacent-course interfaces | 8,599 | 0 |
| Sum of the two hole indicators | 8,677 | 0 |
| Roof-band connected components | 80 | 46 |
| Sum of max(components − 1, 0) per sample | 72 | 38 |
| Caps with no face-contacting neighbour | 5 | 0 |

The footprint includes planned pitched/seam and broad flat surfaces, excluding preserved
narrow details, skylights and unsampled areas. Contacts use actual emitted block shapes
at quarter-block resolution. Components use a shallow three-block band at half-block
resolution. Separate source roofs and terraces legitimately produce multiple components.
“Hole” counts are missing-column/interface indicators, not counts of distinct openings;
zero does not establish facade correctness, structural engineering validity or complete
connectivity outside the measured roof band.

Selected region patterns: 1:1 **19**, 2:1 **12**, 3:1 **12**, 1:2 **8**, bounded 8:5 **2**,
seams **19**, broad flats **18**. Preserved geometric strips are recorded separately.
New target usage: **93,884 full**, **14,011 slabs** (11,869 bottom; 2,142 top), **1,338
stairs**, **17 fences**, **0 walls**. Wall conversion is intentionally conservative;
the fixture gallery demonstrates masonry supports and parapets with positive evidence.
The fixtures contain **15 fence** and **15 wall** cells and reduce 574 hole indicators
and seven unsupported caps to zero. All eleven new fixture schematics round-trip.

Front/back inspection found:

| Source | Judgment | What changed / what remains |
|---|---|---|
| Tools Shop | Improved | Continuous shallow courses, connected backing and surviving porch fences. Thin sign/ridge detail and the red-brown texture remain coarsely represented. |
| Temple of Rome | Improved | Roof courses connect and the colonnade opens substantially. Site removal still loses the broad front podium/stairs. Columns are too broad for fence/wall conversion. |
| Church | Improved roof | Continuous grey courses better match the shingles. Door, windows, entrance rails and tower ornament remain coarse/noisy. |
| Hello Neighbor | Improved roof | Main/extension courses are coherent. Small windows, porch shapes and material transitions remain inaccurate. |
| Apartment | Mostly unchanged | Parapet continuity survives; balcony slabs remain. Several windows merge/disappear and balcony colour remains wrong. |
| Bank | Mostly unchanged | A valid closed low target still passes its gate. The recorded scale is too small for windows, door and sign details. |
| Skyscraper | Mostly unchanged visually | Broad flat caps connect; parapet and horizontal slab bands survive. Facade colour/aliasing and small upper-roof specks remain. |
| Mansion | Mixed, with a material regression | Roof courses and tower cap colours improve, but small roof/support caps acquire brown patches. Source detail remains severely undersampled. Do not treat it as a clean training target. |

The deterministic fixture roofs have coherent courses, ridges and compound joins. Small
corner/eave aliases remain outside the sampled plane footprint, particularly on shallow
roofs. The support and railing fixtures demonstrate evidence-based shape choices, not
that every real source contains enough information for those conversions.

Lower-confidence follow-ups are texture-aware window/door segmentation, source-part-aware
colour boundaries on tiny roofs, explicit podium versus terrain separation, and refinement
of sub-sample eave/ornament fragments. Blanket filling, enlarging ambiguous windows,
reconstructing the Temple podium from a guess, and random detail were not used.

## Reproduce and validate

Commands below ran from BuildDataCollector unless noted. PowerShell redirection saved
their logs beneath `data/synthetic/builder-roof-v5-20260930`.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_roof_quality.py -q
# 5 passed
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
# 350 passed in 143.147s
```

From Real2Minecraft:

```powershell
.\r2m_app\Scripts\python.exe -m unittest discover -s tests -p test_builder_regions.py -q
# 18 passed; includes 384 canonical/near-pitch direction/offset/width subcases
.\r2m_app\Scripts\python.exe -m unittest discover -s tests -q
# 73 passed in 150.375s
.\r2m_app\Scripts\python.exe dev_building.py roof_quality_0930 Easy_Tower_for_M2R_20260706-162025 Hard_Tower_For_M2R_20260706-162335 House_for_M2R_20260706-163500
.\r2m_app\Scripts\python.exe dev_building.py --golden check roof_quality_0930
# Freshly rebuilt references: GOLDEN CLEAN (45 files, pipeline bldg-0.2.0)
```

The existing golden file was not updated. Baseline suites before implementation were
345 and 55 passing tests. Additional array-level checks of mirrored/rotated hip,
compound and lean-to plans found zero shape-label differences.

```powershell
.\.venv\Scripts\python.exe -m synth.roof_compare --before data/synthetic/pilot-v4 --out data/synthetic/builder-roof-v5-20260930 --raw-cache data/synthetic/pilot-v1/glb --only 21f6b 2ac7f 3d232 3c46e 53984 76511 aa8a0 17375
.\.venv\Scripts\python.exe -m synth.roof_gallery --baseline data/synthetic/builder-roof-v5-20260930/baseline/mcvoxel --out data/synthetic/builder-roof-v5-20260930/fixtures
```

Both generation commands refuse an existing result; use a new `--out` when repeating.
The gallery loads saved pre-change grid/roof/assignment modules under separate names;
unchanged shared raster/colour routines and the current renderer are common to both
sides. It uses the mesh helpers from the editable Real2Minecraft test checkout. The
real-source comparison is directly against the archived pilot outputs. Source hashes,
configurations, seeds and fixture provenance are saved with the results.

Logs: [BuildDataCollector tests](../data/synthetic/builder-roof-v5-20260930/tests-bdc-final.txt),
[Real2Minecraft tests](../data/synthetic/builder-roof-v5-20260930/tests-r2m-final.txt),
[golden check](../data/synthetic/builder-roof-v5-20260930/golden-check.txt).
Final repository status and index-preservation checks are recorded in
[git-status-final.txt](../data/synthetic/builder-roof-v5-20260930/git-status-final.txt).
