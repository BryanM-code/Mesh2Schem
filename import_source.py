"""Work out what a dropped file actually is, and open it as something croppable.

Four things arrive at the Add page and only one of them is a schematic:

* `.schem` / `.schematic` -- a build, possibly with slack around it
* `.mcworld` -- a Bedrock world export
* `.zip` -- most often a *Java* world folder, sometimes schematics, sometimes neither
* anything else -- say so plainly

The zip case is the one that used to be a dead end. A world downloaded from a build site
arrives as `level.dat` plus `region/*.mca`, and no single file inside it is a build --
which is impossible to work out by looking at the archive. So rather than reject it, this
identifies it and hands back a world the region picker can cut a build out of.

Everything opens as a `VoxelWorld`, so the picker, the preview map and the converter are
written once.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

import bedrock_world
import java_world
import schem_source
from bedrock_world import WorldError

SCHEMATIC_SUFFIXES = (".schem", ".schematic")
MAX_ZIP_ENTRIES = 20_000

__all__ = ["describe_upload", "open_source", "WorldError"]


def _zip_names(path) -> list:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ZIP_ENTRIES:
                raise WorldError("Archive holds {:,} entries.".format(len(infos)),
                                 code="too_many_entries", stage="read")
            return [info.filename for info in infos]
    except zipfile.BadZipFile as exc:
        raise WorldError("Damaged zip archive: {}".format(exc),
                         code="bad_zip", stage="read") from exc


def describe_upload(path, filename: str) -> dict:
    """Identify a dropped file. Returns {kind, ...} without doing the expensive parse.

    `kind` is one of:
      `schematic`     -- open it and crop it
      `bedrock_world` -- a .mcworld
      `java_world`    -- a zipped Java world
      `zip_schematics`-- a zip that holds schematics rather than a world
      `unknown`       -- with `contents` describing what was actually in there
    """
    lower = str(filename or "").lower()
    path = Path(path)

    if lower.endswith(SCHEMATIC_SUFFIXES):
        return {"kind": "schematic"}
    if lower.endswith(".mcworld"):
        return {"kind": "bedrock_world"}

    if lower.endswith(".zip") or zipfile.is_zipfile(path):
        names = _zip_names(path)
        if java_world.looks_like_java_world(names):
            root = java_world.find_world_root(names) or ""
            regions = sum(1 for n in names
                          if java_world.REGION_RE.match(n.rsplit("/", 1)[-1]))
            return {"kind": "java_world", "world_root": root, "region_files": regions}

        schematics = [n for n in names if n.lower().endswith(SCHEMATIC_SUFFIXES)]
        if schematics:
            return {"kind": "zip_schematics", "schematics": schematics}

        # Nothing usable. Say what *was* in there -- "this zip is no good" is not an
        # answer anyone can act on.
        worlds = [n for n in names if n.lower().endswith(".mcworld")]
        if worlds:
            return {"kind": "zip_worlds", "worlds": worlds}
        top = sorted({n.split("/")[0] for n in names if n})[:8]
        return {"kind": "unknown",
                "contents": top,
                "reason": "This zip has no schematic and no Minecraft world in it "
                          "(no region/*.mca, no level.dat, no .schem)."}

    return {"kind": "unknown", "contents": [],
            "reason": "Expected a .schem, .schematic, .mcworld or a zipped world."}


def open_source(path, kind: str):
    """Open an identified upload as a VoxelWorld the region picker can work on."""
    if kind == "schematic":
        return schem_source.load_schem_source(path)
    if kind == "bedrock_world":
        return bedrock_world.open_world(path)
    if kind == "java_world":
        return java_world.open_world(path)
    raise WorldError("Nothing to open for a {} upload.".format(kind),
                     code="not_openable", stage="read")


def extract_zip_schematics(path, names, destination) -> list:
    """Pull the schematics out of a zip into `destination`, ignoring everything else.

    Archive names are never joined onto a path -- each file is written under a name this
    function chooses -- so an entry called `../../evil.schem` lands as a plain filename.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(path) as archive:
        for index, name in enumerate(names):
            info = archive.getinfo(name)
            if info.file_size > 64 * 1024 * 1024:
                continue
            safe = Path(name).name or "entry"
            target = destination / "{:03d}__{}".format(index, safe)
            with archive.open(info) as source, open(target, "wb") as out:
                out.write(source.read())
            written.append({"name": safe, "path": str(target), "size": info.file_size})
    return written
