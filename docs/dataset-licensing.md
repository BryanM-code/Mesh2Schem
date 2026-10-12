# Dataset licence register

*Living document. Started 2026-09-24, before any 3D mesh was downloaded for the synthetic
dataset. Every source the mesh → `.schem` pipeline touches is listed here with its exact
licence and what that licence allows for each of the five uses the project needs. Nothing
is downloaded in bulk from a source until its row here says it may be.*

This is an engineering summary, not legal advice. Where a licence's application to machine
learning is unsettled, the row says so and the policy takes the conservative reading.

## The five uses

| Code | Use | What it means here |
| --- | --- | --- |
| **T** | Training | Downloading the source, deriving voxel pairs locally, training on them. Nothing leaves the machine. |
| **D** | Derived data | Publishing the voxelised targets (`.schem`) or input grids made from a source object. |
| **C** | Code | Publishing this repository's code, which contains no source assets. |
| **E** | Generated examples | Publishing images or `.schem` files of model output for a specific source object (a demo, a README figure, a portfolio page). |
| **M** | Trained model | Publishing the trained weights. |

## Summary

| Source | Licence | T | D | C | E | M | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Objaverse 1.0 objects under **CC0 1.0** | CC0 | yes | yes | yes | yes | yes | **Use** |
| Objaverse 1.0 objects under **CC BY 4.0** | CC BY 4.0 | yes | yes, attribute | yes | yes, attribute | yes, attribution list | **Use** |
| Objaverse 1.0 objects under **CC BY-SA 4.0** | CC BY-SA 4.0 | yes | only as BY-SA | yes | only as BY-SA | conservatively BY-SA | Excluded by default, flagged |
| Objaverse 1.0 objects under **CC BY-NC 4.0 / BY-NC-SA 4.0** | NC (+SA) | non-commercial only | NC (+SA) only | yes | NC (+SA) only | NC (+SA) only | Excluded by default, flagged |
| Objaverse collection wrapper | ODC-By 1.0 | yes | notice required | yes | notice | notice | Notice obligation |
| TRELLIS-500K metadata CSVs (captions, aesthetic scores) | MIT | yes | n/a | include notice if redistributed | n/a | n/a | Use for filtering |
| Objaverse++ annotations | ODC-By 1.0 | yes | n/a | notice if redistributed | n/a | n/a | Use for filtering |
| Berlin LoD2 building models | Datenlizenz Deutschland – Zero – 2.0 | yes | yes | yes | yes | yes | **Use** (geometry only, untextured) |
| 3D BAG (Netherlands) | CC BY 4.0 | yes | yes, attribute | yes | yes, attribute | yes, attribute | Allowed, not used yet |
| Helsinki 3D city model | CC BY 4.0 | yes | yes, attribute | yes | yes, attribute | yes, attribute | Allowed, not used yet |
| PLATEAU (Japan) | Public Data License 1.0 (CC BY 4.0 compatible) + Surveying Act caveat | per dataset | per dataset | yes | per dataset | per dataset | Needs per-dataset check |
| BuildingNet | not stated; sourced from 3D Warehouse | unclear | unclear | yes | unclear | unclear | **Avoid** |
| Objaverse-XL GitHub subset | no per-object licence | unclear | unclear | yes | unclear | unclear | **Avoid** |
| Building3D (Estonia) | CC BY-NC-SA 4.0 | NC only | NC-SA | yes | NC-SA | NC-SA | Avoid unless NC accepted |
| ArchiSet, BuildingWorld | not located / not released | — | — | — | — | — | Not available |
| Real Minecraft builds in `data/builds/` | creator's copyright; licence unknown for 105 of 106 | local eval only | **no** | yes | only with permission | see below | **Evaluation only** |
| Human-built schematic dataset (2026-10, ~317 builds) | per build: tiers A / B / C / X | A+B+C, privately | **no** | yes | **no** | tier A retrain or re-audit first | **Train privately** ([below](#schematic-dataset-2026-10)) |
| TRELLIS (image → mesh) | MIT (code and weights) | yes | yes | yes | yes | yes | Use |
| Minecraft block names / game | Mojang usage guidelines | yes | yes | yes | yes, with disclaimer | yes | See notes |

## Schematic dataset (2026-10)

Decision **D14** ([plan, version 4](mesh-to-schem-plan.md#version-4-2026-10-11-schematic-first-pivot)),
which supersedes D6 for this dataset. The Objaverse policy below is unchanged and still
governs the parked Objaverse path.

The dataset is ~317 human-built realistic schematics from Planet Minecraft and Abfielder,
collected at `C:\Users\mendo\Documents\Codex\2026-10-05\find-and-download-30-high-quality-2\outputs\Real2Minecraft_Dataset\`.
Status: **personal research prototype**.

| Tier | Meaning | Trained on |
| --- | --- | --- |
| A | explicit reuse permission | yes, privately |
| B | licence unclear | yes, privately |
| C | personal-use / noncommercial | yes, privately |
| X | excluded | no |

Tier X now:

| Build | Why excluded |
| --- | --- |
| Costanera | CC BY-NC-ND: no derivatives |
| Xorph barn | no redistribution |

Per use:

| Use | Rule |
| --- | --- |
| **T** training | tiers A+B+C, privately |
| **D** derived data | never: no schematics or derived pairs are redistributed |
| **E** generated examples | never: no renders are redistributed |
| **C** code | unaffected: both repositories are public and hold code and docs only. No schematics, screenshots, pairs or weights are committed (`.gitignore` blocks them; history scrubbed 2026-10-11) |
| **M** trained model | **release rule:** any public release or commercial use of model weights requires retraining on tier A only, or re-auditing licences first |

Every build keeps its provenance and licence fields.

## Objaverse 1.0 in detail

Objaverse 1.0 is 798,759 Sketchfab models. Each carries the licence its uploader chose;
the collection as a database is ODC-By 1.0. Measured over all 160 metadata shards:

| Licence | Objects | Share |
| --- | --- | --- |
| CC BY 4.0 | 704,846 | 88.2% |
| CC BY-NC-SA 4.0 | 50,723 | 6.4% |
| CC BY-NC 4.0 | 24,326 | 3.0% |
| CC BY-SA 4.0 | 15,399 | 1.9% |
| CC0 1.0 | 3,465 | 0.4% |
| NoDerivatives (any) | 0 | 0% |

Among the 21,584 objects that pass the architecture filters (see
`docs/mesh-to-schem-plan.md`, section 1), the split is 14,787 CC BY, 17 CC0, 162 BY-SA,
651 BY-NC and 5,967 BY-NC-SA. **5,458 of the BY-NC-SA objects come from one uploader**
(`homedesign3d`), almost all houses on landscaped plots. Excluding NC therefore costs
mostly one author's houses, which is also a diversity problem in its own right.

### What each Creative Commons condition means for the five uses

Creative Commons' own guidance
([Using CC-licensed Works for AI Training](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/))
is the reference used here. Its relevant statements:

- CC licences "apply only when copyright permission is required. If exceptions or
  limitations apply, then the CC license terms don't apply." Whether training needs
  permission at all depends on jurisdiction; this register assumes it does, which is the
  conservative reading.
- **Attribution (BY).** "For AI model training, attribution could be a simple link to the
  source of the dataset used to train the model." For a specific output, "providing
  attribution to the CC-licensed work tied to the particular model output with a link to
  the source is ideal."
- **ShareAlike (SA).** "If AI models or outputs are based on ShareAlike content and they
  will be shared publicly, following the ShareAlike condition would require AI developers
  to use the same CC license." Whether weights are an adaptation is unsettled; the
  conservative reading is that a published model trained on BY-SA objects must be BY-SA.
- **NonCommercial (NC).** "Following the NC restriction would require that all stages,
  from copying the data during training to sharing the trained model, must not be for
  commercial gain."
- **NoDerivatives (ND).** Would forbid training use. Objaverse 1.0 contains no ND objects.

Per use, for the licences actually present:

| | CC0 | CC BY | CC BY-SA | CC BY-NC(-SA) |
| --- | --- | --- | --- | --- |
| **T** training | allowed | allowed | allowed | allowed only if the project is non-commercial at every stage |
| **D** derived voxel data | allowed | allowed with per-object attribution: author, title, URL, licence, "voxelised" | must itself be BY-SA | must be NC (and SA) |
| **C** code | unaffected | unaffected | unaffected | unaffected |
| **E** generated examples | allowed | allowed; credit the source object shown | must be BY-SA | NC (and SA) |
| **M** trained model | allowed | allowed; publish the training attribution list | conservatively BY-SA | NC (and SA); blocks any later commercial use |

**The one place the licences diverge is SA and NC.** SA forces the licence of derived data,
of examples, and, on the conservative reading, of the model. NC makes the whole chain
non-commercial, including anything you later build on the weights. A portfolio project is
non-commercial today; NC would make it permanently so.

### Policy adopted for the pilot

**CC0 + CC BY 4.0 only** (14,804 architecture objects). This keeps all five uses open,
with attribution as the only obligation. BY-SA and NC objects are recorded in the index
with their licence but not downloaded. Changing this is Decision L1 in the plan.

Attribution is carried mechanically so it never has to be reconstructed:

- every generated `.schem` stores `source_dataset`, `uid`, `author`, `source_url` and
  `licence` in its Sponge `Metadata` (via `schem_write`);
- every sample's `manifest.json` repeats them;
- a script can emit `ATTRIBUTION.md` for any published subset or trained model from the
  index, plus the ODC-By notice: "Contains information from Objaverse, which is made
  available under the ODC Attribution License."

### Risks that a licence field cannot remove

- **Uploader-asserted licences.** Sketchfab licences are chosen by the uploader. Rips of
  commercial game assets appear under CC BY. The index rejects names, tags or descriptions
  naming games, studios or "ripped" (594 objects rejected on that rule).
- **Trademarks and real buildings.** Some models are branded buildings or famous
  landmarks. The CC licence covers the uploader's model, not a trademark shown on it.
  Avoid branded buildings in published examples.
- **Photogrammetry scans** of real places are the uploader's work under CC BY, but may
  show people, signage or private property. Prefer non-scan objects for published examples.

## City models (LoD2)

- **Berlin LoD2** (`gdi.berlin.de/data/a_lod2/atom`): about 925 CityGML tiles of 1×1 km,
  licensed under Datenlizenz Deutschland – Zero – Version 2.0, which permits any use
  without conditions. Verified on one tile (`LoD2_392_5820`): 732 buildings, cadastral
  building-function codes (`bldg:function`, e.g. 31001_1010 residential), roof-type codes
  and measured heights in metres, **no textures**. Suitable for geometry-only samples with
  real scale and real building-use labels.
- **3D BAG**: CC BY 4.0, about 10 million buildings, LoD2.2, untextured, no building-use
  attribute.
- **Helsinki**: CC BY 4.0, textured LoD2 CityGML.
- **PLATEAU**: Public Data License 1.0, compatible with CC BY 4.0; modifications must be
  marked and not presented as the ministry's. The site notes that some 3D city model data
  also falls under Japan's Surveying Act. That must be checked per dataset before use.

## The real Minecraft builds in this library

*For the 2026-10 schematic dataset see [Schematic dataset (2026-10)](#schematic-dataset-2026-10); the evaluation-only split here was D9, superseded by D17.*

105 of the 106 real builds have no recorded source URL; the one that does records only
PlanetMinecraft's front page. Their licences are therefore unknown, and build creators
retain copyright by default.

- **T** Evaluating a model on them locally is low-risk. Training or fine-tuning a model
  that is then published is unclear and is avoided by the split recommended in the plan
  (evaluation only).
- **D** Do not redistribute the `.schem` files or voxel grids made from them.
- **E** Do not publish renders or side-by-side comparisons of a creator's build without
  their permission. Aggregate metrics (numbers) are fine.

## Minecraft itself

A free tool that outputs `.schem` files is permitted by Mojang's usage guidelines as long
as it does not look official. A public demo should carry "NOT AN OFFICIAL MINECRAFT
PRODUCT. NOT APPROVED BY OR ASSOCIATED WITH MOJANG OR MICROSOFT." Renders in this project
use flat block colours (`block_colors.py`), so no Mojang texture files are redistributed;
keep it that way if a resource pack is ever added to Real2Minecraft's `textures/blocks/`.

## Other inputs

- **AI-generated photos** for the evaluation pairs come from whichever image model you use.
  Check that model's terms before publishing those photos or meshes made from them. If the
  real pairs stay evaluation-only, this never matters.
- **Captions** in TRELLIS-500K were generated by a GPT-4 model per the TRELLIS dataset card.
  They are used here only to filter and categorise, never as training targets.

## Sources

- Creative Commons, [Using CC-licensed Works for AI Training](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)
- Creative Commons, [Understanding CC Licenses and Generative AI](https://creativecommons.org/2023/08/18/understanding-cc-licenses-and-generative-ai/)
- [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) (sections 2.4, 4.2, 4.3)
- Objaverse on [Hugging Face](https://huggingface.co/datasets/allenai/objaverse) (ODC-By; per-object licences in `metadata/*.json.gz`)
- Objaverse-XL paper, [arXiv 2307.05663](https://arxiv.org/abs/2307.05663): "The individual objects are subject to the licenses that they are released under"
- TRELLIS-500K on [Hugging Face](https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K) (MIT)
- Objaverse++ on [Hugging Face](https://huggingface.co/datasets/cindyxl/ObjaversePlusPlus) (ODC-By 1.0)
- Berlin LoD2 [ATOM feed](https://gdi.berlin.de/data/a_lod2/atom) and [Berlin Open Data](https://daten.berlin.de/datensaetze/3d-gebaudemodelle-im-level-of-detail-2-lod-2-3c7c49af)
- 3D BAG [attributes and licence](https://docs.3dbag.nl/en/schema/attributes/)
- Helsinki [3D models](https://www.hel.fi/en/decision-making/information-on-helsinki/maps-and-geospatial-data/helsinki-3d)
- PLATEAU [site policy](https://www.mlit.go.jp/plateau/site-policy/)
- BuildingNet [paper](https://arxiv.org/abs/2110.04955) and 3D Warehouse [terms FAQ](https://help.sketchup.com/en/3d-warehouse/3d-warehouse-terms-use-faq)
- Minecraft [usage guidelines](https://www.minecraft.net/en-us/usage-guidelines)
