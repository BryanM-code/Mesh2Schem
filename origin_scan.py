"""Recover where a schematic was downloaded from, using Windows' Mark of the Web.

A `.schem` carries no provenance -- the Sponge format has no field for it, and the ones
in this library come back with an empty `Metadata` compound. Once a file is in the
library there is nothing in it that says which site it came from.

Windows, however, tags files that arrive from the internet with an NTFS alternate data
stream called `Zone.Identifier`, and that stream records the URL:

    [ZoneTransfer]
    ZoneId=3
    ReferrerUrl=https://www.planetminecraft.com/
    HostUrl=https://s3.amazonaws.com/.../suburbanhouse310695835.schematic?X-Amz-...

So the origin is recoverable, but only from the *original download*, not from the copy
under data/builds/ -- `shutil.copyfile` does not carry alternate data streams across, and
a browser never hands a web app a file's real path. It therefore has to work backwards:
scan the folder the files were downloaded into, and match each one to a build by content.

Matching is by hash, not by filename, because a build can be renamed after it is added.
Legacy `.schematic` files are converted first, exactly as the add path converts them, so
their hash is comparable with what the library stored.

This is Windows-specific and entirely best-effort: on any other platform, or for a file
that was never downloaded by a browser, it simply finds nothing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlsplit

import schem_convert
from schem_meta import sha256_file

SCHEMATIC_SUFFIXES = (".schem", ".schematic")

# Hosts that only ever serve the file itself, never a page worth linking to. A signed CDN
# URL expires within hours and tells you nothing you can revisit, so a referrer pointing
# at the actual site always wins over one of these.
_FILE_HOSTS = ("s3.amazonaws.com", "cloudfront.net", "googleusercontent.com",
               "backblazeb2.com", "cdn.", "media.", "static.", "files.")

__all__ = ["read_zone_identifier", "best_origin_url", "scan_folder"]


def read_zone_identifier(path) -> dict:
    """Parse a file's Zone.Identifier stream into a dict, or {} if it has none.

    NTFS exposes an alternate data stream as `<path>:<stream>`, which Python's open()
    handles directly. A file without one raises FileNotFoundError; a non-NTFS volume or a
    non-Windows OS raises OSError. Both mean the same thing here: nothing to recover.
    """
    try:
        with open("{}:Zone.Identifier".format(path), "r",
                  encoding="utf-8", errors="replace") as fh:
            raw = fh.read(8192)
    except OSError:
        return {}

    fields = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        key, sep, value = line.partition("=")
        if sep and value.strip():
            fields[key.strip()] = value.strip()
    return fields


def best_origin_url(fields: dict) -> Optional[str]:
    """Pick the most useful URL out of a Zone.Identifier record.

    Preference order, most revisitable first:

    1. A ReferrerUrl with a real path -- the page the download was started from. This is
       the one worth having, and it is what direct downloads record.
    2. A HostUrl that isn't a file/CDN host -- some sites serve the file from the page's
       own domain, so the URL still identifies the site.
    3. A bare ReferrerUrl origin -- for sites that hand off to a CDN (PlanetMinecraft
       does exactly this), the browser only records the site root. It won't take you to
       the build's page, but it does say where to go looking.

    A signed CDN link is never returned: the signature expires, and it points at bytes
    the library already has rather than at anything you could open.
    """
    referrer = fields.get("ReferrerUrl", "").strip()
    host_url = fields.get("HostUrl", "").strip()

    if referrer and _has_path(referrer):
        return referrer
    if host_url and _has_path(host_url) and not _is_file_host(host_url):
        return host_url
    if referrer:
        return referrer
    if host_url and not _is_file_host(host_url):
        return host_url
    return None


def _has_path(url: str) -> bool:
    """True when a URL points at something more specific than a site's front page."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return bool(parts.scheme) and parts.path not in ("", "/")


def _is_file_host(url: str) -> bool:
    try:
        host = urlsplit(url).netloc.lower()
    except ValueError:
        return False
    return any(marker in host for marker in _FILE_HOSTS)


def _candidates(folder: Path, recursive: bool) -> Iterator[Path]:
    walker = folder.rglob("*") if recursive else folder.glob("*")
    for path in walker:
        try:
            if path.is_file() and path.suffix.lower() in SCHEMATIC_SUFFIXES:
                yield path
        except OSError:
            continue


def library_hash(path: Path) -> Optional[str]:
    """The hash this file *would* have in the library, converting legacy files first.

    The library stores the Sponge form of every schematic, so a legacy `.schematic` on
    disk hashes to something the database has never seen. Running the same conversion the
    add path runs -- which is deterministic, by design -- makes the two comparable.
    """
    converted = path.with_name(path.stem + ".originscan.schem")
    try:
        prepared = schem_convert.ensure_sponge(path, converted)
        return sha256_file(prepared.path)
    except Exception:
        return None
    finally:
        try:
            converted.unlink(missing_ok=True)
        except OSError:
            pass


def scan_folder(folder, hash_to_build: dict, recursive: bool = True) -> dict:
    """Match downloads in `folder` to builds and report the origin URL of each.

    `hash_to_build` maps a library content hash to whatever the caller wants back (an id,
    a row -- this module doesn't care). Returns a report; it writes nothing itself, so the
    caller decides what to do with the findings.
    """
    folder = Path(folder).expanduser()
    report = {"folder": str(folder), "scanned": 0, "matched": [], "unmatched": 0,
              "no_origin": 0, "errors": []}
    if not folder.is_dir():
        report["errors"].append("Not a folder: {}".format(folder))
        return report

    for path in _candidates(folder, recursive):
        report["scanned"] += 1
        fields = read_zone_identifier(path)
        url = best_origin_url(fields) if fields else None

        digest = library_hash(path)
        if digest is None or digest not in hash_to_build:
            if url is None:
                report["no_origin"] += 1
            else:
                report["unmatched"] += 1
            continue
        if url is None:
            report["no_origin"] += 1
            continue

        report["matched"].append({
            "build": hash_to_build[digest],
            "filename": path.name,
            "path": str(path),
            "url": url,
            "referrer": fields.get("ReferrerUrl", ""),
            "host_url": fields.get("HostUrl", ""),
            # A bare origin is still worth storing, but the UI should say so rather than
            # pretending it links to the build's own page.
            "is_site_root": not _has_path(url),
        })
    return report


def default_download_folder() -> str:
    """The user's Downloads folder -- the obvious place to point the scan first."""
    return str(Path(os.path.expanduser("~")) / "Downloads")
