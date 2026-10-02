# Dataset quality audit — resumed 2026-09-29

This phase prepares and reviews training data. It does not train a model or authorize a
large dataset. The new diagnostic run is `data/synthetic/pilot-v4`, a targeted 40-source
subset of the existing 200 selected assets. Its configuration and selection rationale are
saved in that directory. No new source collection was downloaded.

## Where the interrupted work stopped

BuildDataCollector has two separate data paths. The Flask/SQLite collector stores real
schematics, screenshots, generated images and attached TRELLIS meshes under `data/builds`.
The offline `synth` pipeline selects architectural meshes, prepares clean Minecraft targets
and degraded inputs, validates their common coordinate frame, and writes review artifacts.
Both projects use the installable `mcvoxel` package in Real2Minecraft. The model/training
scaffolding is outside this task.

The interrupted agent had implemented most of the target-quality rework and generated
193 of 200 pilot-v3 results: 164 passed automated checks and 29 were rejected. Seven had
no final manifest. The crop fix, scale resource cap and additional metadata filters were
edited after that run, so its results did not represent the final working code. Its paired
TRELLIS comparison was already available (110 shared successful pairs), despite a stale
documentation note saying that experiment had not run.

Historical pilots are preserved. The current follow-up includes the seven unfinished
sources, the explicitly requested Hello Neighbor house, former height/size rejections,
known problem examples and every one of the 14 architecture categories. This is a
diagnostic sample, not an estimate of the whole pool's quality or category prevalence.

## Filtering and scale

The old pipeline conflated representability and preferred dimensions with source quality.
Pilot v2 rejected 44 sources as `too_small`, two as `too_tall`, and one as `too_large`.
Category height expectations could also select a different unit solely to fit a typical
house, shrinking a legitimate building before the footprint gate rejected it.

The inherited scale rework keeps sub-16-block buildings, retains uncertain units with
`scale_uncertain`, and marks resource reductions with `height_capped` / `size_capped`.
Only a broken numerical scale is a scale-quality rejection. There are still explicit
engineering bounds and a minimum representable grid; these are not claims that the source
architecture is garbage. Caption/storey agreement is heuristic evidence, not a measured
real-world dimension.

On the 193 completed historical v3 sources, **54 previous rejections became successful
pairs**: 40 `too_small`, two `too_tall`, one `too_large`, five `thin_shell`, three
`fragmented`, two `flat_site`, one `scattered`. These are historical comparisons, not
quality endorsements. The fresh 40-source comparison is in `pilot-v4/compare_pilot-v2`.

A further false-rejection rule was found during this continuation: `furniture-home`
categorization without an `architecture` tag, or an isolated `room`/`chair`/`prop` tag,
was treated as proof that a source was not a building. Cached source renders show that
“Glass house” and “Sports Stadium” are counterexamples. Such weak evidence now becomes
`source.review_warnings`. An object noun corroborated by the title and tag still rejects
an actual object (e.g. “Low poly cinema chair”). Existing explicit subject, licence,
scene and geometry checks remain.

The normal goal remains approximately one block per metre. For this small diagnostic run,
the bounding-volume budget is 2,000,000 voxels instead of the 8,000,000 default. The very
large stadium cases were taking several minutes apiece even at the default cap. Capped
examples retain their proportions and uncertainty flags. Every new manifest and review
page explicitly records `effective_blocks_per_metre`, including jitter and caps; a capped
example must not be mistaken for a reliable 1:1 example.

## Source quality

The original visual audit contains 195 ratings: 101 clean, 45 flawed and 49 junk. Among
181 sources labelled “realistic” in Objaverse++ metadata, only about 53% received a clean
target rating and about 23% were junk. The available quality/aesthetic labels are too
coarse to certify suitability. Low-poly vocabulary sometimes predicts easier voxelization,
but that does not establish realistic architecture. The historical ratings also conflate
source quality and conversion quality; they are not a dedicated realism ground truth.

Game assets, dioramas, incomplete models and misleading captions remain in the candidate
pool. A successful export is not a curation approval. Keep source realism, completeness,
scale confidence, and target fidelity as separate review decisions. Do not replace house
bias with fantasy-building bias or reject every game-associated building by name.

**Recommendation: retain individually reviewed Objaverse examples and supplement the
pool with real city/building data before any bulk expansion.** The official sources checked
for this audit are:

| Candidate | Useful contribution | Limitation / source |
| --- | --- | --- |
| 3DBAG | Actual building massing and roofs, geographical dimensions | LoD geometry is not facade-detail supervision. [Official data and attribution terms](https://docs.3dbag.nl/en/copyright/) |
| Helsinki semantic city model | Real building geometry, including textured CityGML | Prefer the semantic building model for a structured pilot; the reality mesh is a different product. [City of Helsinki](https://www.hel.fi/en/decision-making/information-on-helsinki/maps-and-geospatial-data/helsinki-3d) |
| PLATEAU | Diverse real urban buildings, multiple levels of detail | Check the actual dataset's level of detail, textures and licence. [MLIT start guide](https://www.mlit.go.jp/plateau/start-guide/) |

No source is a drop-in solution for windows, balconies, materials and all architectural
types. The run reused 38 cached meshes and refetched two cache misses from the same
existing selection (about 0.7 MB). No new source collection was added.

## Mesh → voxel findings and changes

| Stage | Finding | Implementation / evidence |
| --- | --- | --- |
| Normalization and scale | The former category/size rules could shrink or reject usable architecture. | Preserve unusual sizes with confidence flags; saved seed/settings make the stage replayable. |
| Triangle coverage | Conservative triangle/box rasterization is intentionally over-covering, not simply missing random triangles. | Keep the SAT rasterizer; inspect its surface output before interpreting later holes. Existing oracle tests alone are not a visual-quality proof. |
| Solid filling | Flood fill leaks through genuine open bottoms/windows; the enclosed-room heuristic repairs some interiors but also mistakes colonnades for rooms. | Temple of Rome shows 85,569 extra occlusion-filled voxels, including open spaces between columns. This occurs even at high resolution, so it is not only a one-block gap limitation. The stage sheet isolates it before cleanup. |
| Cleanup and shape conversion | Closing clean meshes fills gaps; the earlier sliver rule removed thin architectural details. | Clean targets disable closing. Fine-layer evidence distinguishes repeated surface coverage from isolated parapets/posts. |
| Connectivity | The interrupted bridge repair failed its own regression by restoring harmless removals on a detached wing, and stopped after eight rounds. | Restore only candidate links touching distinct kept pieces, stop once connected, and allow long links. Regression covers harmless removals, a 16-voxel bridge and an originally separate component. Whole-component restoration was tested and discarded because it thickened entire roofs. |
| Cropping | The origin was assigned before checking whether the frame was caller-specified, so automatic cropping never happened. | The inherited fix is retained and tested; explicitly fixed input frames stay fixed. |
| Site removal | A low bank's roof fell below the site-height threshold after ground removal. Only a rooftop sign remained tall, so cleanup cut away the building and the quality gate called it an open model. | If no broad tall core exists, preserve the source instead of using a few tall columns as the entire footprint. A low-building/sign regression protects this case. |
| Material assignment | Pooling all surfaces inside a voxel mixed facade fronts with roofs, interiors and floors. | The inherited exposed-face ray sampler is retained, followed by the builder palette and roof material harmonization. |
| Small material details | TRELLIS-oriented quantization erased isolated one-block regions unconditionally. | Clean targets set `absorb_always_size=0`; genuinely distinct small material regions are no longer automatically noise. |
| Block export | Schematic and target arrays use different axis conventions. | Existing per-sample label/frame checks and block-for-block schematic readback remain required. |
| Review rendering | The generic Library renderer used dye-name approximations for terracotta: valid gray cyan-terracotta blocks appeared teal, and green terracotta appeared bright green. | Synthetic target previews now use the shared Minecraft matching palette, including slab/stair base materials. A palette regression protects the review. Historical PNGs retain their old renderer and must not be treated as a controlled color comparison. |

This is substantial rework of the target construction stages, not a replacement of the
triangle rasterizer. The shared implementation stays in `mcvoxel`; no duplicate voxelizer
was introduced. Fresh Real2Minecraft building runs passed the existing golden check
(45 reference files, `bldg-0.2.0`). The shared package test suite passes all 55 tests.

The stage debugger now replays a run's actual seed/settings with `--recorded`, checks
stage counts against its manifest, includes raw surface colors before block mapping,
and counts removed ground/site geometry in the original frame. Previously cropping the
trace first hid removals outside the final building box. `--old` uses current code with
legacy settings and is explicitly not a historical algorithm replay; saved pilot images
provide the actual before examples.

## Degradation and real TRELLIS

The interrupted rework reduced blur/noise, holes (0–1% of faces in patches of 5–25),
floaters (0–8), color smoothing, back-side replacement and hue destruction. Those
conservative geometry ranges are retained. The paired v3 comparison nevertheless found
synthetic brightness about **11.4 L\*** below real TRELLIS on the paired median difference.
The older v2 difference was about **19.9 L\***. Strong tone darkening and ambient shading
were being multiplied onto already shaded source textures.

This continuation changes tone gain from 0.6–0.9 to 0.9–1.05 and ambient from 0.6–0.85
to 0.85–0.98, then repeats the paired experiment on the fresh pilot. These settings remain
provisional. The experiment reuses actual `trellis.glb` outputs already generated from
the same source images; it does not substitute synthetic meshes for TRELLIS.

Interpretation limits: the input pictures are source-model renders, not real photos;
placement uses target height and the best quarter-turn; retained site geometry can affect
alignment; target crops and scales changed between runs. Use the common-source tables
alongside the images. IoU alone does not establish TRELLIS likeness or pair suitability.

## Reviewing the result

Start the review with:

```powershell
.venv\Scripts\python -m synth.review data/synthetic/pilot-v4 --compare data/synthetic/pilot-v2
```

The page shows source, degraded mesh, Minecraft target, builder/geometric roofs, back view,
input voxels, flags and attribution. Diagnostic links open stage sheets and paired TRELLIS
comparisons when present. User verdicts and tags belong in `review.json`; the assistant's
visual assessment is recorded separately and must not be presented as human approval.

**Do not scale yet.** In particular, the Hello Neighbor source now survives and remains
recognizable, but window colors/details are still lost. Its stage sheet shows that loss
already in surface-color sampling at block resolution, before final palette mapping.

## Final small pilot and visual decision

The final run is **synth-2.1.1 / mcvoxel 0.3.2**, configuration `9475a3d6b801`.
All **40 sources produced pairs**, with zero processing errors or automated rejections.
This includes all seven sources left unfinished in v3. Relative to v2, nine formerly
rejected sources survive (six `too_small`, two `too_tall`, one `too_large`); all 31
previously successful sources in this selection still export. Twenty pairs have uncertain
scale, three have a height cap, seven have a size cap, and ten are small buildings. These
flags overlap. Export success is not an approval to train on all 40.

The earlier same-sample run is preserved as `pilot-v4-before-site-fix`. It generated
39 pairs and falsely rejected the bank. Replaying that source showed 78 legitimate
voxels removed as site. The final run removes zero site voxels there, retains the complete
building footprint, and passes the target gate. Its three-block height still loses facade
detail, so its pair remains visually flawed.

All 40 rows were visually inspected with source, degraded mesh, builder target,
geometric target and rear view. The assistant's independent assessment is:

| Assessment | Count | Meaning |
| --- | ---: | --- |
| Candidate for human review | 13 | Worth reviewing; no training approval implied |
| Flawed pair | 19 | Recognizable but with meaningful geometry, detail, material or site problems |
| Unsuitable current pair | 8 | Incomplete/object source, or source identity destroyed at this grid scale |

Source assessment separately found 20 plausibly complete real-world buildings, 14
stylized complete buildings, and six incomplete models/objects. This is a deliberately
difficult diagnostic sample, not a random estimate of Objaverse quality. The greenhouse
is an example of a useful source with an unsuitable current target, not garbage source
data. Both the roofless Mansion and a milling apparatus pass automated conversion while
remaining unsuitable sources for this task. Metadata alone does not settle realism or
completeness.

The assessments are in `pilot-v4/assistant_review.json`; no human verdicts were written
to `review.json`. The 13 candidates include stylized sources and still need a human
realism/completeness decision. The pipeline is improved, but this result does **not**
justify a 1,000–2,000-source run.

### Before/after evidence

`pilot-v4/visual_audit/before_after.png` renders actual v2 and v4 schematics with the
same current palette, alongside source and both final roof styles. This avoids mistaking
the old dye-color approximation for a change in the underlying schematic.

| Example | Result |
| --- | --- |
| Hello Neighbor (`17375cb9`) | Previously `too_small`; now exported and recognizable, but window loss still blocks approval. |
| Bank (`3c46e912`) | New site-removal regression fixed; complete footprint survives. Small-grid facade loss remains. |
| Church (`2ac7f39a`) | Roof and steeple remain readable; entrance and windows still need improvement. |
| Apartment (`3d232f22`) | Balcony geometry survives; too many windows merge into blank walls. |
| Masonry Mansion (`5398492c`) | Thin roof features remain; teal appearance was a preview bug. Actual gray materials now display correctly. |
| Skyscrapper (`76511fd7`) | Massing and repeated facade bands survive well enough to be a review candidate. |
| Temple of Rome (`aa8a074f`) | Stage audit isolates incorrect colonnade filling before cleanup; neither roof style repairs it. |

Stage sheets and count traces are saved for eight sources in `pilot-v4/voxel_debug`.
All eight recorded-setting replays agree with their production manifest counts.
Preview colors are representative palette colors, not textured in-game screenshots.

### Validation and saved evidence

BuildDataCollector: **345 tests passed**. Real2Minecraft/shared package: **55 tests passed**.
The final Real2Minecraft `resume0929` building outputs pass the unchanged golden check:
**45 reference files, GOLDEN CLEAN**. All generated pairs pass their label/frame validation
and schematic readback checks. Automated tests do not certify visual quality.

The run contains selection rationale, configuration, per-source attribution and manifests,
source-code fingerprints, automated report, assistant review, contact sheets and stage
traces. The review page is available via the command above; source/degraded/builder/
geometric/back panels and diagnostic links are present.

Remaining work before another quality pilot: curate complete realistic sources separately
from conversion fidelity; redesign interior inference to preserve colonnades/porches;
preserve distinct facade features at Minecraft scale; review site/field removal for
stadiums and courtyards; and validate degradation on held-out real-photo TRELLIS outputs.
Neither model training nor large-dataset generation was run.
