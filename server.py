#!/usr/bin/env python3
"""BuildDataCollector -- local web app for collecting (schem, TRELLIS glb) training pairs.

Run:  python server.py                    (this machine only, opens a browser)
      BDC_HOST=tailscale python server.py (also reachable from your other devices)

There is deliberately ONE database and ONE copy of the files, on whichever machine runs
the server; other machines just point a browser at it. Build ids come from AUTOINCREMENT
and are baked into folder names (data/builds/<id>__<slug>), so two machines writing to
their own copies would both mint id 1 and quietly collide. Serving instead of syncing
keeps a single writer and sidesteps that entirely -- see the README.

Backend is a thin JSON API over db.py; the frontend is plain HTML/JS in templates/ +
static/ with no build step. Uploads are streamed to a project-local temp file, inspected
(dims/hash for schem, trimesh-validity for glb), then COPIED into data/builds/ by db.py --
originals are never moved or deleted.

Schematics arrive in two formats -- modern Sponge `.schem` and the legacy MCEdit
`.schematic` that older downloads still use. Both are accepted; the library only ever
stores `.schem`, so legacy files are converted on the way in (see schem_convert). Every
upload is validated as a genuine, sanely-sized schematic before it is parsed at all --
these files come off the internet, so nothing trusts the extension or the header.
"""
from __future__ import annotations

import csv
import functools
import io
import os
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import webbrowser
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request, Response, send_file

import bedrock_world
import db
import image_validate
import mcworld_convert
import mcworld_preview
import origin_scan
import schem_convert
import thumbs
from schem_convert import SchematicError
from schem_meta import sha256_file
from glb_validate import validate_glb

SCHEMATIC_EXTENSIONS = (".schem", ".schematic")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
MCWORLD_EXTENSIONS = (".mcworld",)

# Bumped whenever the JSON contract changes in a way the frontend depends on. Python loads
# modules once at startup, but static/app.js is read from disk on every request -- so a
# server left running while the code changes serves a NEW frontend against an OLD API. The
# frontend checks this number and says so, instead of silently missing controls whose data
# never arrived.
API_VERSION = 4

# Everything a build can have a picture of. Whitelisted rather than interpolated: the
# thumbnail route is the only one that reaches into data/builds/, and the path it builds
# is assembled entirely from the DB row plus this table, never from the URL.
THUMB_KINDS = {
    "schem": thumbs.schem_thumb,
    "image": thumbs.image_thumb,
    "mesh": thumbs.mesh_thumb,
}
for _slot in range(1, db.MAX_SHOTS + 1):
    THUMB_KINDS["shot-{}".format(_slot)] = functools.partial(thumbs.shot_thumb, slot=_slot)

app = Flask(__name__, static_folder="static", template_folder="templates")
# Generous cap: TRELLIS glbs run to tens of MB; schem files are tiny.
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

TMP_DIR = db.DATA_DIR / ".tmp"

# Prepare storage on import so `flask run` works too, not just `python server.py`.
db.init_db()
TMP_DIR.mkdir(parents=True, exist_ok=True)

# The database is derived from data/builds/ and is deliberately not shared between
# machines. On a fresh clone it doesn't exist yet, and after a `git pull` it hasn't seen
# the folders that arrived -- so reconcile it with the disk before serving anything.
STARTUP_SYNC = db.sync_from_disk_if_needed()

# World imports stash their upload until a region is chosen; sweep any that were abandoned.
(TMP_DIR / "mcworld").mkdir(parents=True, exist_ok=True)


# ── Upload helpers ────────────────────────────────────────────────────────────

def _save_upload(file_storage, suffix: str) -> Path:
    """Stream an uploaded file to a fresh project-local temp path and return it."""
    fd, tmp = tempfile.mkstemp(dir=str(TMP_DIR), suffix=suffix)
    os.close(fd)
    tmp_path = Path(tmp)
    file_storage.save(str(tmp_path))
    return tmp_path


def _cleanup(path) -> None:
    """Best-effort removal of a temp upload."""
    try:
        if path is not None:
            Path(path).unlink(missing_ok=True)
    except Exception:
        pass


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _is_schematic_filename(name) -> bool:
    """True for the two schematic extensions we accept. Only a first-pass filter --
    the real decision is made on the file's contents by schem_convert."""
    return str(name or "").lower().endswith(SCHEMATIC_EXTENSIONS)


def _warm_thumb(build: dict, kind: str) -> None:
    """Render a build's thumbnail now, so the Library doesn't have to wait on it.

    Purely an optimisation, and deliberately silent: the thumbnail route renders on demand
    anyway (which is what covers every build added before thumbnails existed), so a
    failure here costs one slower page load and nothing else.
    """
    try:
        THUMB_KINDS[kind](build["id"], build["slug"])
    except Exception:
        pass


@contextmanager
def _prepared_schematic(file_storage):
    """Yield a validated Sponge .schem for an upload, converting legacy files first.

    The yielded ConversionResult's `path` is always a `.schem`: either the upload
    itself (already Sponge, left byte-for-byte identical so its content hash is stable)
    or a freshly converted copy. Both temp files are removed on the way out, and the
    user's original download is never touched.

    Raises SchematicError if the upload is not a genuine, safely-sized schematic.
    """
    suffix = ".schematic" if str(file_storage.filename).lower().endswith(".schematic") else ".schem"
    tmp = _save_upload(file_storage, suffix)
    converted = tmp.with_name(tmp.stem + ".converted.schem")
    try:
        yield schem_convert.ensure_sponge(tmp, converted)
    finally:
        _cleanup(tmp)
        _cleanup(converted)


def _schematic_fields(prepared, filename: str) -> dict:
    """The format/conversion half of an upload response, shared by both endpoints."""
    return {
        "filename": filename,
        "source_format": prepared.info.format,
        "source_format_label": prepared.info.label,
        "converted": prepared.converted,
        "warnings": prepared.warnings,
    }


# ── Page ──────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


# ── API: inspect (no save) ────────────────────────────────────────────────────

@app.post("/api/inspect")
def api_inspect():
    """Validate a dropped schematic, report its dims + hash, and flag content duplicates.

    Saves nothing -- powers the instant dimension readout and dedup warning on drop.
    Legacy `.schematic` files are converted in a temp file so the dims, hash and any
    conversion warnings shown here are exactly those of the `.schem` that would be saved.
    """
    f = request.files.get("schem")
    if f is None or not f.filename:
        return jsonify(error="No file provided."), 400
    if not _is_schematic_filename(f.filename):
        return jsonify(error="Expected a .schem or .schematic file."), 400

    try:
        with _prepared_schematic(f) as prepared:
            w, h, length = prepared.info.dims
            schem_hash = sha256_file(prepared.path)
            conn = db.connect()
            try:
                existing = db.find_by_hash(conn, schem_hash)
            finally:
                conn.close()
            return jsonify(
                ok=True,
                w=w, h=h, l=length, dims="{}x{}x{}".format(w, h, length),
                longest=max(w, h, length), hash=schem_hash,
                duplicate=bool(existing), existing=existing,
                **_schematic_fields(prepared, f.filename),
            )
    except SchematicError as exc:
        # Structured: `code` is stable to assert on, `stage`/`path`/`offset` say where
        # the file went wrong. See nbt_safe.FileRejected.
        return jsonify(filename=f.filename, **exc.details()), 400


# ── API: create build ─────────────────────────────────────────────────────────

@app.post("/api/builds")
def api_create_build():
    f = request.files.get("schem")
    if f is None or not f.filename:
        return jsonify(error="No file provided."), 400
    if not _is_schematic_filename(f.filename):
        return jsonify(error="Expected a .schem or .schematic file."), 400

    name = request.form.get("name", "")
    source_note = request.form.get("source_note", "")
    allow_duplicate = _truthy(request.form.get("allow_duplicate", "false"))

    try:
        with _prepared_schematic(f) as prepared:
            # db.create_build copies whatever it is handed, and it is always a .schem.
            schem_hash = sha256_file(prepared.path)
            try:
                row = db.create_build(name, source_note, prepared.path, schem_hash,
                                      prepared.info.dims, allow_duplicate)
            except db.DuplicateBuildError as exc:
                return jsonify(error="duplicate", existing=exc.existing), 409
            _warm_thumb(row, "schem")
            return jsonify(build=row, **_schematic_fields(prepared, f.filename)), 201
    except SchematicError as exc:
        # Structured: `code` is stable to assert on, `stage`/`path`/`offset` say where
        # the file went wrong. See nbt_safe.FileRejected.
        return jsonify(filename=f.filename, **exc.details()), 400


# ── API: list / detail ────────────────────────────────────────────────────────

@app.get("/api/builds")
def api_list_builds():
    args = request.args
    builds = db.list_builds(
        status=(args.get("status") or None),
        q=((args.get("q") or "").strip() or None),
        max_longest=_int_or_none(args.get("max_longest")),
        min_longest=_int_or_none(args.get("min_longest")),
        sort=args.get("sort", "added_at"),
        direction=args.get("dir", "desc"),
    )
    return jsonify(builds=builds, count=len(builds))


@app.get("/api/builds/<int:build_id>")
def api_get_build(build_id: int):
    build = db.get_build(build_id)
    if build is None:
        return jsonify(error="Build not found."), 404
    return jsonify(build=build)


# ── API: Bedrock world import ─────────────────────────────────────────────────
# A .mcworld holds a whole world, not a build, so importing takes two steps rather than
# one: upload and look at it, then pick the region that is actually the build. The upload
# is stashed under a token between the two so a large world isn't sent twice, and the
# parsed world is cached in memory so choosing a region stays instant.

WORLD_DIR = TMP_DIR / "mcworld"
_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
# Parsed worlds, keyed by token. A world is tens of MB parsed, and the whole point of the
# token is that you are mid-import on exactly one of them.
_world_cache: "OrderedDict[str, object]" = OrderedDict()
_WORLD_CACHE_MAX = 3
_world_lock = threading.Lock()
# How long an abandoned upload lingers before the next startup sweeps it.
WORLD_TTL_SECONDS = 24 * 3600


def _world_path(token: str) -> Path:
    """Path for a token's stashed upload. The token is validated, never joined blindly."""
    if not _TOKEN_RE.match(token or ""):
        raise ValueError("bad token")
    return WORLD_DIR / token / "world.mcworld"


def _sweep_world_uploads() -> int:
    """Delete stashed uploads left behind by imports that were never finished."""
    removed = 0
    try:
        cutoff = time.time() - WORLD_TTL_SECONDS
        for folder in WORLD_DIR.iterdir():
            if folder.is_dir() and folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
                removed += 1
    except OSError:
        pass
    return removed


def _load_world(token: str):
    """The parsed world for a token, from cache or by reading the stashed upload."""
    with _world_lock:
        cached = _world_cache.get(token)
        if cached is not None:
            _world_cache.move_to_end(token)
            return cached
    path = _world_path(token)
    if not path.exists():
        return None
    world = bedrock_world.open_world(path)
    with _world_lock:
        _world_cache[token] = world
        while len(_world_cache) > _WORLD_CACHE_MAX:
            _world_cache.popitem(last=False)
    return world


def _forget_world(token: str) -> None:
    with _world_lock:
        _world_cache.pop(token, None)
    try:
        shutil.rmtree(_world_path(token).parent, ignore_errors=True)
    except ValueError:
        pass


def _is_mcworld_filename(name) -> bool:
    return str(name or "").lower().endswith(MCWORLD_EXTENSIONS)


def _bbox_from(payload: dict) -> dict:
    """Read a bounding box out of a request body, rejecting anything non-numeric."""
    try:
        return {key: int(payload[key]) for key in
                ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")}
    except (KeyError, TypeError, ValueError):
        raise _ImageRejected("A region needs min_x/max_x, min_y/max_y and min_z/max_z.")


def _preview_bounds(world) -> dict:
    """The X/Z frame the preview map covers.

    Framed on what was built, not the world's full extent: a world is mostly empty terrain,
    and framing on all of it would shrink the part you actually need to click on. Returned
    with the inspect response so the client can turn a click into world coordinates.
    """
    padding = 12
    bounds = dict(world.world_bounds())
    auto = world.auto_bbox()
    if auto:
        bounds.update(min_x=auto["min_x"] - padding, max_x=auto["max_x"] + padding,
                      min_z=auto["min_z"] - padding, max_z=auto["max_z"] + padding)
    return bounds


def _region_dims(bbox: dict) -> dict:
    width = abs(bbox["max_x"] - bbox["min_x"]) + 1
    height = abs(bbox["max_y"] - bbox["min_y"]) + 1
    length = abs(bbox["max_z"] - bbox["min_z"]) + 1
    volume = width * height * length
    return {"w": width, "h": height, "l": length, "longest": max(width, height, length),
            "dims": "{}x{}x{}".format(width, height, length), "volume": volume,
            "max_volume": schem_convert.MAX_VOLUME, "too_big": volume > schem_convert.MAX_VOLUME}


@app.post("/api/mcworld/inspect")
def api_mcworld_inspect():
    """Take a .mcworld, parse it, and report what's in it plus where the build looks to be.

    Saves nothing to the library -- this is the world equivalent of /api/inspect.
    """
    f = request.files.get("world")
    if f is None or not f.filename:
        return jsonify(error="No file provided."), 400
    if not _is_mcworld_filename(f.filename):
        return jsonify(error="Expected a .mcworld file."), 400

    token = secrets.token_hex(16)
    folder = WORLD_DIR / token
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "world.mcworld"
    try:
        f.save(str(target))
        world = bedrock_world.open_world(target)
    except bedrock_world.WorldError as exc:
        shutil.rmtree(folder, ignore_errors=True)
        return jsonify(filename=f.filename, **exc.details()), 400
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    with _world_lock:
        _world_cache[token] = world
        while len(_world_cache) > _WORLD_CACHE_MAX:
            _world_cache.popitem(last=False)

    bounds = world.world_bounds()
    auto = world.auto_bbox()
    if auto is None:
        _forget_world(token)
        return jsonify(error="This world has no buildings in it -- only terrain.",
                       code="nothing_built", stage="mcworld"), 400

    return jsonify(
        token=token, filename=f.filename, level_name=world.level_name,
        subchunks=len(world.subchunks), bounds=bounds, auto_bbox=auto,
        map_bounds=_preview_bounds(world), suggested=_region_dims(auto),
        preview="/api/mcworld/{}/preview.png".format(token))


@app.get("/api/mcworld/<token>/preview.png")
def api_mcworld_preview(token: str):
    """Top-down map of the world, with the auto-detected build area outlined."""
    try:
        world = _load_world(token)
    except ValueError:
        return jsonify(error="Unknown import."), 404
    except bedrock_world.WorldError as exc:
        return jsonify(**exc.details()), 400
    if world is None:
        return jsonify(error="That import has expired. Drop the world again."), 404

    auto = world.auto_bbox()
    bounds = _preview_bounds(world)

    image = mcworld_preview.render_world_map(world, bounds, highlight=auto)
    if image is None:
        return jsonify(error="Nothing to preview."), 404

    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    buffer.seek(0)
    response = Response(buffer.getvalue(), mimetype="image/png")
    # The map covers a fixed area of a stashed upload, so it can't change under the token.
    response.headers["X-Map-Bounds"] = "{},{},{},{}".format(
        bounds["min_x"], bounds["min_z"], bounds["max_x"], bounds["max_z"])
    return response


@app.post("/api/mcworld/<token>/region")
def api_mcworld_region(token: str):
    """Dimensions and block count for a candidate region, without converting anything."""
    try:
        world = _load_world(token)
    except ValueError:
        return jsonify(error="Unknown import."), 404
    if world is None:
        return jsonify(error="That import has expired. Drop the world again."), 404
    try:
        bbox = _bbox_from(request.get_json(silent=True) or {})
    except _ImageRejected as exc:
        return jsonify(error=exc.reason), 400
    return jsonify(bbox=bbox, **_region_dims(bbox))


@app.post("/api/mcworld/<token>/save")
def api_mcworld_save(token: str):
    """Convert the chosen region to a .schem and add it to the library."""
    try:
        world = _load_world(token)
    except ValueError:
        return jsonify(error="Unknown import."), 404
    if world is None:
        return jsonify(error="That import has expired. Drop the world again."), 404

    payload = request.get_json(silent=True) or {}
    try:
        bbox = _bbox_from(payload)
    except _ImageRejected as exc:
        return jsonify(error=exc.reason), 400

    name = (payload.get("name") or "").strip()
    source_note = (payload.get("source_note") or "").strip()
    allow_duplicate = _truthy(payload.get("allow_duplicate", False))

    tmp = None
    try:
        grid, names = world.extract_region(bbox)
        fd, tmp_name = tempfile.mkstemp(dir=str(TMP_DIR), suffix=".schem")
        os.close(fd)
        tmp = Path(tmp_name)
        info = mcworld_convert.region_to_schem(grid, names, tmp)

        # Conversion warnings belong with the build, the way the legacy .schematic path
        # records what it could not carry across.
        note = source_note
        if info["warnings"]:
            joined = " ".join(info["warnings"])
            note = (note + "  " + joined).strip() if note else joined

        schem_hash = sha256_file(tmp)
        try:
            row = db.create_build(name, note, tmp, schem_hash,
                                  (info["width"], info["height"], info["length"]),
                                  allow_duplicate)
        except db.DuplicateBuildError as exc:
            return jsonify(error="duplicate", existing=exc.existing), 409
        _warm_thumb(row, "schem")
        return jsonify(build=row, warnings=info["warnings"],
                       palette_size=info["palette_size"]), 201
    except bedrock_world.WorldError as exc:
        return jsonify(**exc.details()), 400
    finally:
        _cleanup(tmp)


@app.delete("/api/mcworld/<token>")
def api_mcworld_discard(token: str):
    """Drop a stashed upload without importing anything from it."""
    try:
        _forget_world(token)
    except ValueError:
        return jsonify(error="Unknown import."), 404
    return jsonify(discarded=True)


# An import that was started and never finished leaves its upload behind; clear out any
# that are older than the TTL. Done at import so it covers `flask run` as well.
SWEPT_WORLDS = _sweep_world_uploads()


# ── API: edit metadata ────────────────────────────────────────────────────────

@app.patch("/api/builds/<int:build_id>")
def api_update_build(build_id: int):
    """Update the user-entered fields: name, notes, and the source URL.

    Everything else about a build is derived from its files, so only these are editable.
    """
    payload = request.get_json(silent=True) or request.form or {}
    fields = {k: payload[k] for k in db.EDITABLE_FIELDS if k in payload}
    if not fields:
        return jsonify(error="Nothing to update."), 400
    try:
        return jsonify(build=db.update_build(build_id, **fields))
    except db.BuildNotFoundError:
        return jsonify(error="Build not found."), 404


# ── API: images (reference screenshots + the generated image) ─────────────────

@contextmanager
def _prepared_image(file_storage):
    """Yield a validated, PNG-normalised copy of an uploaded image.

    Raises _ImageRejected with a ready-made response body when the upload is not a real,
    sanely-sized image. Both temp files are removed on the way out; the user's original is
    never touched. Shared by the screenshot and generated-image endpoints so the two
    cannot drift apart on what they accept.
    """
    tmp = png = None
    try:
        tmp = _save_upload(file_storage, Path(str(file_storage.filename)).suffix or ".img")
        ok, size, err = image_validate.validate_image(tmp)
        if not ok:
            raise _ImageRejected(err)
        png = tmp.with_name(tmp.stem + ".normalized.png")
        image_validate.normalize_to_png(tmp, png)
        yield png, size
    finally:
        _cleanup(tmp)
        _cleanup(png)


class _ImageRejected(Exception):
    """An upload that isn't a usable image. Carries the reason to show the user."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _is_image_filename(name) -> bool:
    """First-pass filter only -- the real decision is made by decoding the bytes."""
    return str(name or "").lower().endswith(IMAGE_EXTENSIONS)


@app.post("/api/builds/<int:build_id>/shots")
def api_add_shots(build_id: int):
    """Attach reference screenshots -- raw captures of the build, up to MAX_SHOTS.

    Optional and status-neutral: these are working material for generating the photo-real
    image later, not an artefact the pipeline waits on. Accepts several files at once so a
    few angles can be dropped in one go; each is validated independently, and the response
    reports per-file failures rather than rejecting the whole drop.
    """
    files = [f for f in request.files.getlist("images") if f and f.filename]
    if not files:
        return jsonify(error="No files provided."), 400

    added, rejected = [], []
    build = db.get_build(build_id)
    if build is None:
        return jsonify(error="Build not found."), 404

    for f in files:
        if not _is_image_filename(f.filename):
            rejected.append({"filename": f.filename,
                             "reason": "Expected an image (.png, .jpg, .webp)."})
            continue
        try:
            with _prepared_image(f) as (png, size):
                build, slot = db.add_shot(build_id, png)
                added.append({"filename": f.filename, "slot": slot,
                              "width": size[0], "height": size[1]})
        except _ImageRejected as exc:
            rejected.append({"filename": f.filename, "reason": exc.reason})
        except db.TooManyShotsError:
            rejected.append({"filename": f.filename,
                             "reason": "No free slot -- a build keeps at most {} "
                                       "screenshots.".format(db.MAX_SHOTS)})
        except db.BuildNotFoundError:
            return jsonify(error="Build not found."), 404

    for shot in added:
        _warm_thumb(build, "shot-{}".format(shot["slot"]))
    if not added and rejected:
        return jsonify(error=rejected[0]["reason"], added=[], rejected=rejected,
                       build=build), 400
    return jsonify(build=build, added=added, rejected=rejected)


@app.delete("/api/builds/<int:build_id>/shots/<int:slot>")
def api_remove_shot(build_id: int, slot: int):
    """Remove one reference screenshot, leaving its slot free for another."""
    confirm = request.args.get("confirm")
    if confirm is None:
        confirm = (request.get_json(silent=True) or {}).get("confirm")
    if not _truthy(confirm):
        return jsonify(error="Removing a screenshot requires explicit confirmation."), 400
    try:
        build = db.remove_shot(build_id, slot)
    except db.BuildNotFoundError:
        return jsonify(error="Build not found."), 404
    except db.ShotNotFoundError:
        return jsonify(error="no_shot", message="There is no screenshot in that slot."), 404
    except OSError as exc:
        return jsonify(error="Could not delete the screenshot, so it was left "
                             "in place: {}".format(exc)), 500
    return jsonify(build=build)


@app.post("/api/builds/<int:build_id>/image")
def api_set_image(build_id: int):
    """Attach the photo-real image TRELLIS reconstructs from -- the generated one.

    Stored as a canonical PNG, so the library holds one predictable file per build no
    matter what was dropped. This is the artefact that gates the TRELLIS stage; the
    reference screenshots are the raw material it comes from.
    """
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(error="No file provided."), 400
    if not _is_image_filename(f.filename):
        return jsonify(error="Expected an image (.png, .jpg, .webp)."), 400
    replace = _truthy(request.form.get("replace", "false"))

    try:
        with _prepared_image(f) as (png, size):
            try:
                row = db.set_generated_image(build_id, png, replace=replace)
            except db.BuildNotFoundError:
                return jsonify(error="Build not found."), 404
            except db.ImageAlreadySetError:
                return jsonify(error="image_exists",
                               message="This build already has a generated image. "
                                       "Confirm replace to overwrite it."), 409
            _warm_thumb(row, "image")
            return jsonify(build=row, width=size[0], height=size[1])
    except _ImageRejected as exc:
        return jsonify(error=exc.reason), 400


@app.delete("/api/builds/<int:build_id>/image")
def api_clear_image(build_id: int):
    """Remove a build's generated image, stepping it back to Needs AI image."""
    confirm = request.args.get("confirm")
    if confirm is None:
        confirm = (request.get_json(silent=True) or {}).get("confirm")
    if not _truthy(confirm):
        return jsonify(error="Removing the image requires explicit confirmation."), 400
    try:
        build = db.clear_generated_image(build_id)
    except db.BuildNotFoundError:
        return jsonify(error="Build not found."), 404
    except db.NoImageError:
        return jsonify(error="no_image",
                       message="This build has no generated image to remove."), 409
    except OSError as exc:
        return jsonify(error="Could not delete the image file, so it was left "
                             "in place: {}".format(exc)), 500
    return jsonify(build=build)


# ── API: the stored files themselves ──────────────────────────────────────────

def _stored_file(build: dict, kind: str):
    """(path, mimetype, download name) for one of a build's stored artefacts, or None.

    The thumbnail route serves 384px renders, which are for recognising a build at a
    glance. This is the other half: the actual bytes, at full resolution -- a screenshot
    you captured is only useful later if you can get the real file back out to feed it to
    something else.

    `kind` is matched against a fixed table and never joined onto a path, and the folder
    comes from db.build_dir() using the slug on the row.
    """
    folder = db.build_dir(build["id"], build["slug"])
    slug = build["slug"]
    fixed = {
        "schem": (folder / "schem" / "original.schem",
                  "application/octet-stream", "{}.schem".format(slug)),
        "image": (folder / "image" / "generated.png",
                  "image/png", "{}-generated.png".format(slug)),
        "mesh": (folder / "trellis" / "output.glb",
                 "model/gltf-binary", "{}.glb".format(slug)),
    }
    if kind in fixed:
        return fixed[kind]
    for slot in range(1, db.MAX_SHOTS + 1):
        if kind == "shot-{}".format(slot):
            return (db.shot_path(build["id"], slug, slot),
                    "image/png", "{}-shot-{}.png".format(slug, slot))
    return None


@app.get("/api/builds/<int:build_id>/file/<kind>")
def api_build_file(build_id: int, kind: str):
    """Serve a build's stored file at full resolution.

    Add `?download=1` to save it rather than view it in the browser. Files are named after
    the build, so a folder of them stays readable once several have been saved.
    """
    build = db.get_build(build_id)
    if build is None:
        return jsonify(error="Build not found."), 404
    entry = _stored_file(build, kind)
    if entry is None:
        return jsonify(error="Unknown file kind."), 404
    path, mimetype, download_name = entry
    if not path.exists():
        return jsonify(error="That file isn't there."), 404
    return send_file(path, mimetype=mimetype, conditional=True,
                     as_attachment=_truthy(request.args.get("download")),
                     download_name=download_name)


@app.get("/api/builds/<int:build_id>/schem")
def api_download_schem(build_id: int):
    """The schematic, as a download -- for loading into Minecraft (WorldEdit) to screenshot.

    Kept as its own URL because it is linked from the UI as a plain download; it is the
    same file the general route serves.
    """
    build = db.get_build(build_id)
    if build is None:
        return jsonify(error="Build not found."), 404
    path, mimetype, download_name = _stored_file(build, "schem")
    if not path.exists():
        return jsonify(error="Schematic file is missing."), 404
    return send_file(path, mimetype=mimetype, as_attachment=True,
                     download_name=download_name, conditional=True)


# ── API: pair a glb ───────────────────────────────────────────────────────────

@app.post("/api/builds/<int:build_id>/pair")
def api_pair_build(build_id: int):
    f = request.files.get("glb")
    if f is None or not f.filename:
        return jsonify(error="No file provided."), 400
    if not f.filename.lower().endswith(".glb"):
        return jsonify(error="Expected a .glb file."), 400
    replace = _truthy(request.form.get("replace", "false"))

    tmp = None
    try:
        tmp = _save_upload(f, ".glb")
        ok, faces, err = validate_glb(tmp)
        if not ok:
            return jsonify(error=err), 400
        try:
            row = db.pair_build(build_id, tmp, replace=replace)
        except db.BuildNotFoundError:
            return jsonify(error="Build not found."), 404
        except db.GlbAlreadyPairedError:
            return jsonify(error="already_paired",
                           message="This build already has a paired mesh. "
                                   "Confirm replace to overwrite it."), 409
        _warm_thumb(row, "mesh")
        return jsonify(build=row, faces=faces)
    finally:
        _cleanup(tmp)


# ── API: remove a pair (explicit, confirmed) ──────────────────────────────────

@app.delete("/api/builds/<int:build_id>/pair")
def api_unpair_build(build_id: int):
    """Detach a build's TRELLIS mesh, e.g. when the wrong .glb was dropped on it.

    Confirmation is required in the same shape as build deletion -- this destroys the
    copied mesh -- but the build itself, and the original file the user dropped, survive.
    """
    confirm = request.args.get("confirm")
    if confirm is None:
        confirm = (request.get_json(silent=True) or {}).get("confirm")
    if not _truthy(confirm):
        return jsonify(error="Removing a pair requires explicit confirmation."), 400
    try:
        build = db.unpair_build(build_id)
    except db.BuildNotFoundError:
        return jsonify(error="Build not found."), 404
    except db.NotPairedError:
        return jsonify(error="not_paired",
                       message="This build has no TRELLIS mesh to remove."), 409
    except OSError as exc:
        # The mesh could not be deleted (locked by another program, most likely), so the
        # build is deliberately left paired rather than desynced from its folder.
        return jsonify(error="Could not delete the mesh file, so the pair was left "
                             "in place: {}".format(exc)), 500
    return jsonify(build=build)


# ── API: thumbnails ───────────────────────────────────────────────────────────

@app.get("/api/builds/<int:build_id>/thumb/<kind>.png")
def api_build_thumb(build_id: int, kind: str):
    """Serve a build's rendered thumbnail, generating it on first request.

    The only route that reads out of data/builds/. It never joins anything from the URL
    onto a path: `kind` selects a renderer from a two-entry table, and the folder comes
    from db.build_dir() using the slug stored on the row.

    404 covers "no such build", "not paired yet" and "could not be rendered" alike -- the
    UI shows a placeholder tile for all three, and none is worth an error page.
    """
    render = THUMB_KINDS.get(kind)
    if render is None:
        return jsonify(error="Unknown thumbnail kind."), 404
    build = db.get_build(build_id)
    if build is None:
        return jsonify(error="Build not found."), 404

    path = render(build_id, build["slug"])
    if path is None or not path.exists():
        return jsonify(error="No thumbnail available."), 404
    # conditional=True gives ETag/Last-Modified handling, so a Library reload of 40 cards
    # costs 40 x 304 rather than 40 re-downloads.
    return send_file(path, mimetype="image/png", conditional=True)


# ── API: delete (explicit, confirmed) ─────────────────────────────────────────

@app.delete("/api/builds/<int:build_id>")
def api_delete_build(build_id: int):
    confirm = request.args.get("confirm")
    if confirm is None:
        confirm = (request.get_json(silent=True) or {}).get("confirm")
    if not _truthy(confirm):
        return jsonify(error="Deletion requires explicit confirmation."), 400
    try:
        removed = db.delete_build(build_id)
    except db.BuildNotFoundError:
        return jsonify(error="Build not found."), 404
    return jsonify(deleted=removed)


# ── API: CSV export ───────────────────────────────────────────────────────────

@app.get("/api/export.csv")
def api_export_csv():
    args = request.args
    builds = db.list_builds(
        status=(args.get("status") or None),
        q=((args.get("q") or "").strip() or None),
        max_longest=_int_or_none(args.get("max_longest")),
        min_longest=_int_or_none(args.get("min_longest")),
        sort=args.get("sort", "added_at"),
        direction=args.get("dir", "desc"),
    )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "name", "status", "width", "height", "length", "longest_axis",
        "source_note", "source_url", "added_at", "image_at", "paired_at",
        "schem_path", "shot_count", "shot_paths", "image_path", "glb_path",
    ])
    for b in builds:
        writer.writerow([
            b["id"], b["name"], b["status"], b["w"], b["h"], b["l"], b["longest"],
            b["source_note"], b["source_url"], b["added_at"], b["image_at"] or "",
            b["paired_at"] or "", b["schem_path"], b["shot_count"],
            # Several paths in one cell, pipe-separated: a build has at most MAX_SHOTS of
            # them, and fixed columns would go stale the moment that number changed.
            "|".join(s["path"] for s in b["shots"]),
            b["image_path"] or "", b["glb_path"] or "",
        ])
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=builds.csv"},
    )


# ── API: rebuild-from-disk admin ──────────────────────────────────────────────

@app.post("/api/admin/rebuild")
def api_admin_rebuild():
    summary = db.rebuild_from_disk()
    return jsonify(summary)


# ── API: recover download origins ─────────────────────────────────────────────

@app.post("/api/admin/scan-origins")
def api_admin_scan_origins():
    """Find where each build was downloaded from, by reading Windows' Mark of the Web.

    Scans a folder of original downloads, matches each file to a build by content hash,
    and reports the origin URL recorded when the browser saved it. `apply` decides whether
    to write the findings; without it this is a dry run, which is the sane default for
    something that reads a folder full of the user's files.

    Only fills in blanks -- a source URL that has already been set or edited by hand is
    never overwritten.
    """
    payload = request.get_json(silent=True) or {}
    folder = (payload.get("folder") or "").strip() or origin_scan.default_download_folder()
    apply_changes = _truthy(payload.get("apply", False))

    builds = db.list_builds()
    by_hash = {b["schem_hash"]: b for b in builds}
    report = origin_scan.scan_folder(folder, by_hash)

    applied, skipped = 0, 0
    for match in report["matched"]:
        build = match["build"]
        match["build"] = {"id": build["id"], "name": build["name"],
                          "source_url": build["source_url"]}
        if build["source_url"]:
            match["skipped"] = "already has a source URL"
            skipped += 1
            continue
        if apply_changes:
            db.update_build(build["id"], source_url=match["url"])
            applied += 1
    report.update(applied=applied, skipped=skipped, dry_run=not apply_changes)
    return jsonify(report)


@app.get("/api/admin/default-downloads")
def api_default_downloads():
    """The folder the origin scan should offer first."""
    return jsonify(folder=origin_scan.default_download_folder())


# ── API: what this server can do ──────────────────────────────────────────────

@app.get("/api/meta")
def api_meta():
    """Version + limits, so the frontend can tell whether it's talking to current code."""
    return jsonify(api_version=API_VERSION, schema_version=db.SCHEMA_VERSION,
                   max_shots=db.MAX_SHOTS, statuses=list(db.ALL_STATUSES))


# ── Error handling ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(_exc):
    return jsonify(error="File too large."), 413


# ── Entry point ───────────────────────────────────────────────────────────────

# Tailscale hands every machine a stable address in the CGNAT range 100.64.0.0/10 and
# encrypts all traffic between them (WireGuard). Binding to *that* address specifically --
# rather than 0.0.0.0 -- is what makes remote access safe here: the app has no accounts and
# no password, so the network it listens on IS the access control. On the Tailscale
# address, only your own devices can reach it; on 0.0.0.0, so can everyone on the coffee
# shop's wifi.
TAILSCALE_NETWORK = "100.64.0.0/10"


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass


def _is_tailscale_ip(address: str) -> bool:
    """True for an address inside Tailscale's 100.64.0.0/10 range."""
    try:
        octets = [int(part) for part in str(address).split(".")]
    except ValueError:
        return False
    return len(octets) == 4 and octets[0] == 100 and 64 <= octets[1] <= 127


def _is_loopback(address: str) -> bool:
    return str(address) in ("127.0.0.1", "localhost", "::1")


def find_tailscale_ip() -> Optional[str]:
    """This machine's Tailscale IPv4 address, or None if Tailscale isn't up.

    Asks the CLI first because it is authoritative and works the same on Windows and
    macOS; falls back to scanning local addresses so a machine without the CLI on PATH
    still works.
    """
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True,
                             timeout=5)
        for line in out.stdout.splitlines():
            if _is_tailscale_ip(line.strip()):
                return line.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if _is_tailscale_ip(address):
                return address
    except (OSError, socket.gaierror):
        pass
    return None


def resolve_host(requested: str):
    """Turn BDC_HOST into (bind address, note). Raises SystemExit with advice if it can't.

    "tailscale" is the interesting one: it resolves to this machine's Tailscale address, so
    the app is reachable from your other devices and nowhere else.
    """
    requested = (requested or "").strip()
    if requested.lower() == "tailscale":
        found = find_tailscale_ip()
        if found is None:
            raise SystemExit(
                "BDC_HOST=tailscale, but no Tailscale address was found.\n"
                "  Is Tailscale installed and signed in?  Check with:  tailscale ip -4\n"
                "  Or bind explicitly, e.g.  BDC_HOST=100.x.y.z python server.py")
        return found, "Tailscale — reachable from your other Tailscale devices only"
    if _is_loopback(requested) or not requested:
        return "127.0.0.1", "this machine only"
    if requested == "0.0.0.0":
        return "0.0.0.0", "EVERY network this machine is on"
    if _is_tailscale_ip(requested):
        return requested, "Tailscale — reachable from your other Tailscale devices only"
    return requested, "whatever network owns this address"


def _startup_banner(host: str, note: str, port: int) -> None:
    """Say plainly where the app is listening, and who can reach it."""
    shown = "127.0.0.1" if host == "0.0.0.0" else host
    print("BuildDataCollector  ->  http://{}:{}".format(shown, port))
    print("  listening on {}  ({})".format(host, note))
    if db.MACHINE_ID:
        print("  machine id {}  (new builds numbered from {:,})".format(
            db.MACHINE_ID, db.MACHINE_ID * db.ID_BLOCK + 1))
    if STARTUP_SYNC is not None:
        s = STARTUP_SYNC
        print("  synced from data/builds/: {} added, {} updated, {} removed".format(
            s["added"], s["updated"], s["pruned"]))
    if host == "0.0.0.0":
        tailscale = find_tailscale_ip()
        if tailscale:
            print("  also at        http://{}:{}  (Tailscale)".format(tailscale, port))
    if not _is_loopback(host):
        print("")
        print("  NOTE: this app has no login. Anyone who can reach the address above can")
        print("        add, edit and DELETE builds. That is fine on Tailscale, which only")
        print("        your own signed-in devices can use.")
        if host == "0.0.0.0":
            print("        0.0.0.0 is wider than that -- prefer BDC_HOST=tailscale.")
    print("  Ctrl+C to stop")


if __name__ == "__main__":
    port = int(os.environ.get("BDC_PORT", "5000"))
    bind_host, reach = resolve_host(os.environ.get("BDC_HOST", "127.0.0.1"))

    # Only pop a browser open when this is the machine you're sitting at; a headless
    # always-on server should not be launching windows every restart.
    if _is_loopback(bind_host):
        # debug=False -> no reloader, so the browser opens exactly once.
        threading.Timer(1.0, _open_browser,
                        args=("http://{}:{}".format(bind_host, port),)).start()
    _startup_banner(bind_host, reach, port)
    app.run(host=bind_host, port=port, debug=False, threaded=True)
