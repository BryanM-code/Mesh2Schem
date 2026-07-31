# BuildDataCollector

A small **local** web app for collecting training-data pairs for a future Minecraft-build
ML model. You drop in real Minecraft `.schem` build files now, and later attach the matching
**TRELLIS** `.glb` mesh reconstruction for each one. Every `(schem, glb)` pair becomes one
training example.

TRELLIS reconstructs from a **photo-real 2D image**, not from a schematic, so each build is
collected in four steps:

```
  .schem     ──▶  reference shots  ──▶  AI-generated image  ──▶   .glb
  the build       up to 3 raw            photo-real, made          the mesh
                  captures, any          from one of the
                  angle you like         shots
                  ── optional ──         (what TRELLIS eats)

  Needs AI image ─────────────────▶  Needs TRELLIS  ─────────▶  Complete
```

The screenshots are **optional working material** — grab a few angles whenever it's
convenient, decide later which one to generate from. They never gate the status; only the
schematic, the generated image and the mesh do.

The parts are collected at different times, so progress is tracked **on disk** (SQLite +
named folders) and survives across sessions. Status is always *derived* from which files
exist, and names the first one still missing.

- **Add New Build** — drop one schematic (detailed review) or many at once (batch list).
  `.schem`, the older `.schematic`, and Bedrock `.mcworld` world exports are all accepted;
  legacy files are **converted to `.schem`** on the way in and a world opens a map you cut
  a build out of. Dimensions are read instantly; content is sha256-hashed and
  dedup-checked. Your original files are **copied in, never moved or deleted**.
- **Library** — every build as a **thumbnail grid** or a searchable / filterable / sortable
  table (your choice is remembered), with stage badges, progress pips and a CSV export.
- **Reference screenshots** — up to 3 quick captures per build. **Paste straight from the
  clipboard** (`Win+Shift+S` then `Ctrl+V`), drop files, or browse. Optional, and kept at
  full resolution so you can open one later and pick an angle to generate from.
- **AI-generated image** — the photo-real picture TRELLIS actually consumes. Attaching it
  is what moves a build to *Needs TRELLIS*.
- **Pair TRELLIS Output** — drop the `.glb` (validated as a real loadable mesh via
  `trimesh`) and the build is *Complete*.
- **Remove any attachment** — dropped the wrong file? Detach a screenshot, the generated
  image, or the mesh; the build steps back and keeps everything else.
- **Find sources** — recover the URL each build was downloaded from, from Windows' record
  of the download.

## Requirements

- **Python 3.10+** (developed against 3.10).
- Dependencies (installed into a project-local venv below): **Flask**, **nbtlib**, **numpy**,
  **Pillow**, **trimesh**.

## Install

From this folder (`BuildDataCollector/`):

```powershell
# 1. Create an isolated virtual environment (its own venv — do not reuse another project's)
python -m venv .venv

# 2. Install dependencies
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

## Run

```powershell
.venv\Scripts\python server.py
```

This starts a local server and opens `http://127.0.0.1:5000` in your default browser.
The SQLite database (`data/collector.db`) and the `data/builds/` folder are created
automatically on first run. Press `Ctrl+C` to stop.

| Variable | Default | What it does |
| --- | --- | --- |
| `BDC_PORT` | `5000` | Port to listen on. |
| `BDC_HOST` | `127.0.0.1` | Bind address. `tailscale` binds to this machine's Tailscale address; `0.0.0.0` listens everywhere. |
| `BDC_MACHINE_ID` | `0` | Which block of build ids this machine allocates from. Set to `1` on the second machine — see [Working from two machines](#working-from-two-machines). |

```powershell
$env:BDC_PORT = "5050"        # different port
$env:BDC_MACHINE_ID = "1"     # on the second machine only
```

**Changing any Python file needs a restart** — see
[Restart after changing Python code](#restart-after-changing-python-code).

## How it stores things

Metadata lives in SQLite; the actual files live on disk in human-browsable folders. The
database only ever stores **paths and hashes**, never file bytes.

```
BuildDataCollector/
  server.py                # Flask app (JSON API + serves the UI)
  db.py                    # SQLite schema, CRUD, rebuild-from-disk
  nbt_safe.py              # bounded, defensive reading of untrusted NBT
  schem_convert.py         # format detection + legacy .schematic -> .schem conversion
  legacy_blocks.py         # pre-1.13 block id:data -> modern blockstate table
  schem_meta.py            # reads schematic Width/Height/Length + sha256
  bedrock_world.py         # reads a Bedrock .mcworld (zip + LevelDB + subchunks)
  mcworld_convert.py       # a region of a world -> Sponge .schem
  mcworld_preview.py       # top-down map of a world, for picking that region
  schem_voxels.py          # decodes a .schem into a voxel grid (for thumbnails)
  block_colors.py          # blockstate -> approximate RGB (for thumbnails)
  thumbs.py                # isometric renders of a schem and a glb, cached on disk
  glb_validate.py          # validates a .glb loads as a mesh (trimesh)
  image_validate.py        # validates + normalises a dropped reference image
  origin_scan.py           # recovers download URLs from Windows' Mark of the Web
  rebuild_from_disk.py     # CLI safety net (see below)
  templates/ , static/     # the web UI (no build step)
  tests/                   # unittest suite (see Tests)
  data/
    collector.db
    builds/
      <build_id>__<slug>/
        schem/original.schem      # always Sponge .schem, whatever was dropped in
        shots/shot-1.png          # reference screenshots, up to 3, always PNG
        shots/shot-2.png
        image/generated.png       # the AI-generated image TRELLIS consumes
        trellis/output.glb        # absent until paired
        thumb/schem.png           # derived; rendered on first view
        thumb/shot-1.png          # derived; one per screenshot
        thumb/image.png           # derived; absent until an image is attached
        thumb/mesh.png            # derived; absent until paired
```

Every artefact has a canonical filename, so a build's folder can be read at a glance and
`rebuild_from_disk` can re-derive everything from it. Screenshot slots are stable ids, not
positions: removing `shot-2.png` leaves a gap that the next screenshot fills, so a slot's
picture never silently becomes a different picture.

Everything under `thumb/` is a **cache**, not data: delete it and the app re-renders it on
the next page load. Nothing about thumbnails is recorded in the database, so they need no
schema and can't fall out of step with it.

## Schematic formats

Builds downloaded from the internet come in two formats, and both are accepted:

| Dropped in | What happens |
| --- | --- |
| **Sponge `.schem`** (v1 / v2 / v3) | Stored byte-for-byte as-is. |
| **MCEdit `.schematic`** (pre-1.13) | Converted to Sponge **v2** and stored as `.schem`. |
| **Bedrock `.mcworld`** | A whole world — you pick a region on a map, which is converted to Sponge **v2**. See [Bedrock world exports](#bedrock-world-exports-mcworld). |

The format is decided by reading the file, never by its extension — a `.schem` that is
really an old MCEdit file still gets converted, and a `.schematic` that is really Sponge
does not. Either way the library only ever contains `.schem`, and **your original
download is never modified, moved, or deleted.**

### What conversion does

`.schematic` predates Minecraft 1.13's "flattening", so it stores numeric block ids and
4-bit metadata (`35:14`) instead of blockstate strings (`minecraft:red_wool`). Converting
means running that mapping — all 254 vanilla ids, in `legacy_blocks.py`.

- Converted files record **DataVersion 1631 (1.13.2)** and use 1.13 block names, so
  Minecraft's own data fixers rename them again for whatever version later loads the file.
- Only properties the legacy data value actually encoded are written (facing, half, axis,
  colour, age…). Properties a `.schematic` never stored — fence and wall connections,
  stair `shape`, redstone wire directions — are left at their default rather than guessed,
  since Minecraft recalculates them on paste anyway.
- Door and double-plant **upper halves** are read together with the block below them,
  which is the only place their facing and species were recorded.
- Banners, beds, skulls and flower pots kept their real identity in a tile entity rather
  than in the block data; those are read and applied too, so a yellow banner stays yellow
  and a potted tulip stays a tulip.
- Conversion is **deterministic** — the same input always produces byte-identical output —
  so the sha256 dedup below still catches a file you have already added.

Anything that could not be carried across is reported in the UI rather than dropped
silently: **unmapped block ids** (almost always modded blocks — they become stone so the
build's shape survives) and **entities** (mobs, item frames, armour stands), which are not
part of a build's geometry. Item contents inside chests are copied across untouched but
are not themselves flattened.

## Bedrock world exports (`.mcworld`)

A `.mcworld` is a **Bedrock Edition world export** — a ZIP holding `level.dat` and a LevelDB
database of chunks. It is not a schematic, so it can't just be stored: a world holds a whole
area, often several separate builds, and the library's unit is one build.

Drop one on the Add page and you get a map of it instead:

```
Dropped: suburbia-karma-2025-09-23-4-56-51-am.mcworld
  1,780 subchunks · 111,670 built blocks

  ┌──────────────────────────────────────┐
  │   ▩▩   trees    ▩▩▩        ▩▩       │   top-down render, coloured
  │  ▩▩▩  ┌───────┐  road  ▩▩▩▩▩   ▩    │   by block. drag a box over
  │   ▩▩  │ house  │ ▩▩▩       ▩▩  ▩▩▩  │   one house.
  │  ▩▩▩  └───────┘  ▩▩▩▩   ▩▩▩▩▩       │
  └──────────────────────────────────────┘
  selection 51x34x51 · 88,434 blocks
  [ Save this region as a build ]
```

The blue box is drawn where the world **looked built** — everything that isn't air or the
flat ground plate of bedrock/dirt/grass. That detection isn't a convenience, it's what
makes the file importable at all: the example world above is 672 × 48 × 624 =
**20,127,744 blocks**, past the 16,000,000 limit, while the built area is 210 × 34 × 205 =
**1,463,700**. Narrow it to one house from there.

Once saved it's an ordinary build — a Sponge `.schem` like any other — so thumbnails,
screenshots, the AI image and TRELLIS pairing all work unchanged.

### What survives the conversion, and what doesn't

**Block names come across.** Modern Bedrock uses the same `minecraft:<name>` vocabulary as
Java — `quartz_stairs`, `oak_log`, `deepslate_tile_stairs` are identical — so
`mcworld_convert.NAME_ALIASES` is a short exceptions table rather than a second
`legacy_blocks`. It covers the genuine differences (`noteblock` → `note_block`,
`wooden_door` → `oak_door`, `darkoak_wall_sign` → `dark_oak_wall_sign`).

**Block states do not.** Bedrock spells them differently (`weirdo_direction` for stair
facing, `upside_down_bit` for slab halves), and translating them faithfully is a table the
size of `legacy_blocks`. Every block therefore lands in its **default state**: shapes,
materials and colours are right, but a staircase may face the wrong way. That's what the
thumbnails and TRELLIS need, and the conversion records it in the build's notes rather than
pretending otherwise. Block entities — chest contents, sign text, banner colours — are
dropped for the same reason.

### How it's read

No LevelDB library is involved; the whole path is stdlib `zipfile` + `struct` + `zlib`,
in `bedrock_world.py`. Three formats stack up:

| Layer | What it is |
| --- | --- |
| **SST** | LevelDB's table format: a footer points at an index block, which points at data blocks of prefix-compressed key/value pairs. Mojang adds compression type 4, raw zlib. |
| **Chunk keys** | `<cx int32 LE><cz int32 LE>[<dim>]<tag>[<subY>]`; tag `0x2F` is a 16×16×16 subchunk. |
| **Subchunk** | Bit-packed palette indices, then the palette as *little-endian* NBT — Bedrock's byte order, so `nbt_safe`'s Java reader can't be pointed at it. |

One detail worth knowing if you ever touch this: SST stores **internal** keys — the user key
plus an 8-byte sequence/type trailer. Forget to strip it and every key is 8 bytes too long
and nothing matches the chunk layout at all.

World files get the same distrust as everything else here: entry counts, uncompressed size,
decompressed block size, record counts and palette lengths are all capped before anything is
allocated, and archive entry names are matched exactly rather than joined onto a path, so a
`../../` entry is simply never a name it looks for.

## Safety of downloaded files

These files come off the internet, so nothing here trusts one until it has been checked.
A schematic is inert data — it is never executed — but a hostile file can still try to
exhaust memory or hang the process, so every upload passes through `nbt_safe.py` before
anything parses it:

- **Magic bytes** must be gzip (`1f 8b`) or raw NBT (`0a`). A renamed `.exe`, a saved HTML
  error page, or a zip is rejected before a single byte is decompressed.
- **Decompression is capped** (64 MB in, 192 MB out), so a gzip bomb fails instead of
  filling RAM.
- **A strict structural scan** walks the NBT tree without allocating anything from it,
  enforcing sane tag ids, nesting depth, tag count, and that every declared length fits
  inside the file. This matters specifically because nbtlib returns `0` instead of raising
  when it reads past the end of a buffer — so a 30-byte file that declares a two-billion
  entry list would otherwise send it into a two-billion iteration loop. That file is now
  rejected in single-digit milliseconds.
- **Then the schematic itself is validated**: dimensions must be positive and within
  4096 per axis and 16,000,000 blocks total, and the block payload must actually match
  those dimensions.

### Why a file was rejected, and where

A rejection is never just a message. Every one carries a stable **`code`**, the **`stage`**
that threw it out, and — when the failure is somewhere inside the file — the **NBT `path`**
and the byte **`offset`** in the decompressed data. Both upload endpoints return all of it
with a `400`:

```json
{
  "filename": "castle.schematic",
  "code":     "list_too_long",
  "stage":    "structure",
  "path":     "TileEntities[2].Items[1]",
  "offset":   296,
  "message":  "Malformed NBT: a list declares 2,000,000,000 entries but only 12 bytes remain.",
  "error":    "Malformed NBT: a list declares 2,000,000,000 entries but only 12 bytes remain. (at TileEntities[2].Items[1], byte 296)"
}
```

`message` is the bare sentence; `error` is that plus the location, and is what the UI
shows (with `code · stage · path · byte` underneath it). **Assert on `code` in tests** —
messages get reworded, codes don't. In Python the same fields are on the exception:

```python
try:
    schem_convert.inspect(path)
except schem_convert.SchematicError as exc:
    exc.code, exc.stage, exc.path, exc.offset, exc.message, exc.details()
```

| Stage | Codes |
| --- | --- |
| `read` | `unreadable`, `empty_file`, `file_too_large` |
| `decompress` | `bad_magic`, `bad_gzip`, `decompression_bomb`, `empty_after_decompress` |
| `structure` | `not_compound_root`, `unknown_tag`, `truncated`, `negative_length`, `list_too_long`, `end_list_entries`, `too_deep`, `too_many_tags` |
| `parse` | `parse_failed`, `nbtlib_missing` |
| `schematic` | `missing_dimension`, `bad_dimension`, `non_positive_dimension`, `axis_too_large`, `volume_too_large`, `no_block_data`, `bad_v3_container`, `bad_palette`, `block_count_mismatch`, `truncated_blocks`, `truncated_data`, `bad_byte_array`, `add_array_too_short`, `palette_too_large` |

Dimension and payload problems name the tag they are about, so `non_positive_dimension`
on a flat schematic comes back with `path: "Height"` rather than leaving you to guess
which axis.

### Duplicates & variants

The dedup check is the file's sha256 content hash. If you add a `.schem` whose bytes match
an existing build, you're warned and can either skip it or **save it anyway as a variant**
(the database keeps a composite `UNIQUE(schem_hash, variant_no)`, so deliberate duplicates
get `variant_no` 1, 2, …).

## Reference screenshots

Raw captures of a build, up to **3 per build**. They're optional and never change a build's
status — the point is to grab a few angles while it's convenient, then come back later, look
at what you kept, and pick one to generate the photo-real image from. No revisiting the
source site, no hunting for the file again.

**Adding one — three ways, on the build page:**

| | |
| --- | --- |
| **Paste** | Snip with `Win+Shift+S`, then `Ctrl+V` anywhere on the build page. Nothing to save or name — this is the fast path. |
| **Drop** | Drag image files onto the dropzone; several at once is fine. |
| **Browse** | Click the dropzone. |

**Getting one back out.** Click any screenshot to open it **at full resolution**, or use
**save** to download it named after the build (`stone-tower-shot-1.png`). This matters: the
thumbnails are 384px renders for recognising a build at a glance, while the file route
serves the original bytes — which is what you actually feed to an image model.

Every stored artefact is retrievable the same way:

```
GET /api/builds/<id>/file/<kind>              # view it
GET /api/builds/<id>/file/<kind>?download=1   # save it
```

where `<kind>` is `schem`, `shot-1` … `shot-3`, `image`, or `mesh`.

**Where to capture from:**

1. **The source page.** Most build sites show a render of the build. Use the **Source** link
   to go back and screenshot the site's own thumbnail. See
   [Where a build came from](#where-a-build-came-from) for how that link gets filled in.
2. **Minecraft.** Use **download the .schem**, load it with WorldEdit (`//schem load <name>`,
   then `//paste`), and screenshot it in-game. This needs a WorldEdit-capable install —
   vanilla Minecraft cannot open a `.schem`.

Slots are stable: removing the middle screenshot leaves slot 2 free, and the next one you
add fills it rather than renumbering the others. Dropping more files than there are free
slots adds what fits and tells you which ones didn't, rather than failing the whole drop.

## The AI-generated image

The photo-real picture TRELLIS actually reconstructs from, generated from one of the
screenshots above. Drop it on the build page — `.png`, `.jpg` or `.webp` — and it's
validated as a real image and stored as a canonical `generated.png`. Attaching it is what
moves a build from *Needs AI image* to *Needs TRELLIS*.

**Generating it is a manual step, by design.** Nothing here calls an image model; you run
whatever you like and drop the result in. What the app does is keep that image attached to
the build, so a mesh can always be traced back to the exact picture it came from — and so
the screenshots it was generated from are still sitting right next to it.

## Where a build came from

A `.schem` carries **no** provenance — the Sponge format has no field for it, and files
downloaded from build sites arrive with an empty `Metadata` compound. Nothing in the file
says where it came from.

Windows does record it, though. Files downloaded by a browser get an NTFS alternate data
stream called `Zone.Identifier` holding the URL. **Find sources** on the Library page reads
that stream off your *original* downloads and fills in each build's Source URL:

```
[ZoneTransfer]
ZoneId=3
ReferrerUrl=https://www.planetminecraft.com/
HostUrl=https://s3.amazonaws.com/.../suburbanhouse310695835.schematic?X-Amz-...
```

Matching is by **content hash**, not filename, so renaming a build doesn't break it. Legacy
`.schematic` files are converted first — deterministically, the same way the add path
converts them — so their hash is comparable with what the library stored. It runs as a dry
run first, only ever fills in blanks, and never overwrites a URL you set yourself.

Two honest limitations:

- **It only works on the original download.** `shutil.copyfile` doesn't carry alternate data
  streams across, and a browser never gives a web app a file's real path — so this has to
  scan the folder you downloaded into, and it finds nothing once those files are gone.
- **Some sites only record their own front page.** Where a download is handed off to a CDN
  (PlanetMinecraft does exactly this), the browser records the *site*, not the build's page.
  The UI says so when that happens and offers a site search for the build's name instead of
  pretending the link goes somewhere it doesn't. A signed CDN link is never stored as a
  source: the signature expires, and it points at bytes the library already has.

You can always paste or correct the URL by hand under **Edit name, source URL and notes**.

## Thumbnails

Every artefact of a build has a picture:

| Picture | From | When |
| --- | --- | --- |
| **Schematic** | the `.schem`'s own voxels, coloured from its block palette | always |
| **Screenshots** | each stored `shot-<n>.png` | one per screenshot |
| **AI image** | the stored `generated.png` | once an image is attached |
| **TRELLIS mesh** | points sampled off the `.glb` surface | once paired |

The two *rendered* ones — schematic and mesh — use the same isometric camera, so a
correctly-paired build shows the same shape from the same angle in both. That is what makes
a **mis-dropped file visible at a glance**: on a Library card the AI image and the mesh sit
inset over the schematic, and if they disagree, they belong to different builds. (Cards
show those two; screenshots are conveyed by the progress pips, and the build page shows all
four side by side.)

Rendering is server-side (numpy + Pillow, no 3D libraries in the browser) and cached under
each build's `thumb/` folder. A thumbnail is redrawn whenever it's missing or older than
the file it depicts, so it survives a rebuild-from-disk and costs nothing after the first
view. Builds added before this existed get theirs the first time you look at them.

Thumbnails are 384px and exist only to be *looked at*. To get the real bytes back — the
full-resolution screenshot, the generated image, the `.glb` — use the file route described
under [Reference screenshots](#reference-screenshots).

## Stepping a build back

Dropped the wrong file? Any attachment can be detached on its own, without touching the
build: **Unpair** on a Library card, or **Remove TRELLIS pair** / **Remove generated image**
/ the **✕** on a screenshot, on the build page. All of them ask to confirm first.

Removing an artefact deletes this build's *copy* of it and steps the status back if that
artefact gated a stage (removing a screenshot never does). The build keeps its id, name,
notes, source URL, schematic and history, and can be re-attached immediately. The file you
originally dropped is untouched, wherever it came from — as everywhere else here, only the
tool's own copy is ever deleted.

The copy really is deleted rather than archived, and that's deliberate: rebuild-from-disk
derives status purely from which files exist, so anything left lying around would silently
restore the stage on the next rebuild.

## Rebuild from disk (safety net)

If the database and the folders ever disagree — e.g. `collector.db` was deleted, or build
folders were copied in by hand — re-derive the rows from what's actually on disk:

```powershell
.venv\Scripts\python rebuild_from_disk.py
```

It scans every `data/builds/<id>__<slug>/` folder, re-reads dimensions + hash from the
schem, reconciles the screenshot slots against `shots/`, and derives status from which of
`image/generated.png` and `trellis/output.glb` are present. Existing rows keep their
user-entered name, notes and source URL; it **never deletes** any row or file. The same
action is available in the app via the **Rebuild from disk** button on the Library page.

## Working from two machines

The repo carries the library. Clone it on the other machine, run the app, and you have the
same builds — add, edit and delete from either, commit, and pull on the other side.

```
PC                              Mac
──                              ───
git pull                        git clone <your repo>
…add builds…                    python3 -m venv .venv && pip install -r requirements.txt
git add -A                      export BDC_MACHINE_ID=1     ← once, see below
git commit -m "…"               .venv/bin/python server.py
git push                        …add builds…
                                git add -A && git commit && git push
```

### Setting up the Mac (once)

```bash
git clone <your repo> BuildDataCollector
cd BuildDataCollector
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

echo 'export BDC_MACHINE_ID=1' >> ~/.zshrc   # so it sticks; see "machine ids" below
export BDC_MACHINE_ID=1
.venv/bin/python server.py
```

There is no database in the repo, so the first run builds one from the folders it cloned —
you'll see `synced from data/builds/: N added` in the startup banner.

### What is and isn't in the repo

| | Tracked? | Why |
| --- | --- | --- |
| `data/builds/**` | **yes** | the library itself: schematics, screenshots, images, meshes |
| `data/builds/*/meta.json` | **yes** | the names, notes and source URLs you typed |
| `data/collector.db` | no | binary — git cannot merge two machines' changes to it |
| `data/builds/*/thumb/` | no | a derived cache, re-rendered on demand |
| `.venv/` | no | per-machine, and platform-specific |

**The database is derived, so it doesn't need syncing.** It is rebuilt from the folders
whenever the two disagree — on a fresh clone, and after any pull that brought builds in or
took them away. That happens automatically at startup; the **Rebuild from disk** button
does the same thing on demand.

### `meta.json`: why it exists

`name`, `source_note`, `source_url`, `variant_no` and `added_at` are things *you typed* —
nothing about the files implies them. They used to live only in the database, which meant a
clone would rebuild every build as `name=<folder slug>` with blank notes and no source URL.

So each build folder carries a small `meta.json`:

```json
{
  "added_at": "2026-07-27T22:33:00+00:00",
  "name": "suburbanhouse310695835",
  "source_note": "Converted from legacy MCEdit .schematic.",
  "source_url": "https://www.planetminecraft.com/",
  "variant_no": 0
}
```

It is written whenever you change any of those fields, and read back when the database is
rebuilt. Being one small text file *per build* is the point: two machines editing different
builds merge cleanly, where one shared binary database could only conflict.

### Machine ids: why the Mac needs `BDC_MACHINE_ID=1`

Build ids are baked into folder names (`data/builds/<id>__<slug>`). Two machines each
counting from 1 would both create `1__something`, and merging those folders means two
different builds claiming the same id — which used to resolve by silently overwriting one
with the other.

Each machine therefore allocates from its own block of a million:

| `BDC_MACHINE_ID` | Ids it creates |
| --- | --- |
| unset / `0` | 1, 2, 3, … — **the PC, unchanged** |
| `1` | 1000001, 1000002, … |
| `2` | 2000001, … |

Set it once on every machine after the first. The startup banner prints it whenever it
isn't 0, so you can see at a glance which machine you're on. Forgetting it is the one way
to reintroduce collisions, which is why it's in the banner rather than buried in a config
file.

### Merge conflicts

Because builds are separate folders and metadata is one small file each, the normal case —
two machines adding *different* builds — merges with no conflict at all. Conflicts are
possible if you edit *the same build* on both machines without pulling in between; git will
say so, and for a binary file you pick a side:

```bash
git checkout --theirs data/builds/5__house/image/generated.png   # take the incoming one
git checkout --ours   data/builds/5__house/image/generated.png   # keep yours
git add data/builds/5__house/image/generated.png
```

The habit that avoids it: **pull before you start, push when you stop.**

### Repo size

Schematics are a few KB and screenshots a few hundred KB, but TRELLIS meshes run to tens of
MB, and git keeps every version of every file forever. That's fine for a while; if the repo
gets uncomfortable, the usual answer is Git LFS. GitHub also rejects any single file over
100 MB, which a `.glb` could eventually hit.

### Line endings

`.gitattributes` marks every data file `binary`. This matters more than it looks: cloning
between Windows and macOS, a `.schem` treated as text would have its line endings rewritten
on checkout and be silently corrupted — and the damage would only surface later as an
unreadable build. Verified: a schematic's sha256 is identical before and after a round trip
through clone.

### If you'd rather not sync at all

One machine can serve the app to the other instead, with no repo and nothing to reconcile:
set `BDC_HOST` (see [Run](#run)) so it listens beyond loopback, and open it from the Mac's
browser. **The app has no login**, so the interface it listens on is its access control:
`0.0.0.0` means everyone on whatever network you're on. `BDC_HOST=tailscale` binds to a
[Tailscale](https://tailscale.com) address, reachable by your own devices from anywhere and
nobody else — but that needs Tailscale installed on both machines.

## Restart after changing Python code

`static/app.js` and `style.css` are read from disk on **every request**, so a browser
refresh picks them up. The Python — `server.py`, `db.py`, everything they import — is
loaded **once, when the process starts**, and never reloads.

Leave a server running while the backend changes and you get the confusing half-state: a
new frontend talking to an old API. Controls whose data never arrives simply don't appear,
with no error anywhere.

So the app checks. `GET /api/meta` reports an `api_version`; the page compares it against
the version it was built for and, on a mismatch, shows a red banner across the top telling
you to restart. Individual sections say the same thing in place — the reference-screenshots
section, for instance, explains that the server doesn't know about screenshots yet rather
than rendering an empty area.

**The fix is always the same:** stop the server (`Ctrl+C`) and run `python server.py` again,
then reload the page. If a stray server is still holding port 5000:

```powershell
Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
  Where-Object { $_.CommandLine -like "*server.py*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

Bump `API_VERSION` in `server.py` whenever the JSON contract changes in a way the frontend
depends on, and `EXPECTED_API_VERSION` in `static/app.js` to match.

## Schema migrations

The database records a `schema_version` and migrates forward on startup. There is one
migration so far:

| Version | Change |
| --- | --- |
| **1 → 2** | Added the image stage and `source_url`. Status went from `needs_trellis`/`paired` to `needs_image`/`needs_trellis`/`complete`. |
| **2 → 3** | Split that single image into **reference screenshots** (new `shots` table, up to 3) and the **AI-generated image**. `image/reference.png` is renamed to `image/generated.png`. |

**v1 → v2** changes the status `CHECK` constraint, and SQLite can only do that by rebuilding
the table — so it is a real table rewrite, done in one transaction. A crash mid-migration
rolls back to the v1 table rather than leaving a half-built one. Ids and the `AUTOINCREMENT`
counter are preserved, because build folders on disk are named by id.

A build that was `paired` under v1 becomes `needs_image` and **keeps its mesh**: under the
newer model it genuinely is missing the image it was reconstructed from. Attaching that
image completes it.

**v2 → v3** renames a file on disk. v2's `reference.png` was described as "the picture
TRELLIS reconstructs from", which is exactly the generated one, so it is renamed rather
than discarded. The rename is per-build and the row follows what actually happened on disk:
a file that can't be moved keeps its old name *and* its old path in the database, so the two
never disagree.

Both steps are idempotent, and a v1 database migrates straight through to v3 in one go.

## Tests

```powershell
.venv\Scripts\python -m unittest discover -s tests
```

Covers format detection, the legacy conversion (including the awkward cases: door and
double-plant halves, tile-entity-backed blocks, modded ids, determinism), the whole
block-mapping table, and a set of hostile files — decompression bomb, truncated NBT
declaring a two-billion-entry list, deeply nested NBT, renamed executables. Also the
four-stage pipeline and its status derivation, the screenshot slots (the cap, partial
drops, gap reuse, cascade on delete), the generated-image slot, thumbnail caching and
path-traversal attempts against it, both migrations against real old databases, and the
origin scanner's URL-preference rules. Fixtures are generated in
`tests/schem_fixtures.py`, so no binaries are checked in, and the tests use a throwaway data
directory rather than your real library.

The alternate-data-stream tests skip themselves on filesystems that can't hold one, so the
suite still passes off Windows.

## Notes

- Single local user; no accounts, no auth. This is a personal utility. To use it from a
  second machine, clone the repo — see
  [Working from two machines](#working-from-two-machines).
- Scope is intentionally narrow: capture and pair files with clean metadata. Block-ID
  mapping exists only to convert legacy `.schematic` files into `.schem`; the rendering
  here exists only to make thumbnails. **There is no training logic, and nothing calls
  TRELLIS or any image model** — you run those yourself and drop the results in. Capturing
  screenshots and generating the photo-real image are both manual steps by design.
- `origin_scan` is Windows-specific. Everywhere else it simply finds nothing, and the rest
  of the app is unaffected.
