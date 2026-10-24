#!/usr/bin/env python3
"""Freeze the human-built schematic dataset and derive its index, tiers and creator splits.

Usage:  python -m schemset.freeze --source "<dataset folder>" --out data/real-schems-v1
                                  [--force] [--seed 20261011]

Phase 2 of docs/mesh-to-schem-plan.md (version 4). Writes, under --out:

    source/                   verbatim copy of the dataset folder, set read-only. Never edited.
    freeze_checksums.sha256   sha256 of every file under source/
    FREEZE.md                 when, from where, file counts, verification results
    index.json                one row per manifest entry: pass-through fields, creator_norm,
                              licence_tier (D14), flags, split
    near_duplicates.json      metadata-only near-duplicate candidates, each with its reason
    splits.json               creator-grouped train / val / test split (D17), seed and method,
                              and the decisions applied after it (`overrides`)

Two decisions are applied on top of the seeded split, so no other build moves:

*   D24, D25: builds set aside by decision (`SET_ASIDE`) get split "excluded" and their reason
    as a flag.
*   Geometric duplicates (`schemset.dedupe` -> `geometric_duplicates.json`, read here when
    present): a duplicate pair in two different splits is put into one by moving the smaller
    creator group, whole, into the other's split. Both builds get `geometric_duplicate`.

The copy goes to `source.partial/` first and becomes `source/` only after every check
passes: each line of the dataset's own checksums.sha256, each manifest sha256, and every
copied file against its original. Any mismatch stops the run (VerificationError).

Re-running is safe: an existing `source/` is never overwritten without --force. Without it
the frozen copy is re-verified against freeze_checksums.sha256 and only the derived files
(index, near-duplicates, splits) are regenerated -- deterministically, from `source/`.

Nothing is dropped here. Tier X builds are kept and flagged; their split is "excluded".
"""
from __future__ import annotations

import argparse
import collections
import datetime
import difflib
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import stat
import subprocess
import sys
import unicodedata
from pathlib import Path

DATASET = "real-schems-v1"
SEED = 20261011
SPLITS = ("train", "val", "test")
FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}

# D14. Exact `license_status` strings; anything else stops the run (UnknownLicence).
TIER_BY_LICENCE = {
    "stated reuse permission; scope limited": "A",
    "license unclear": "B",
    "restricted personal-use license": "C",
    "explicit noncommercial reuse permission": "C",
    "explicit license: CC BY-NC-SA 4.0": "C",
    "explicit license: CC BY-NC-ND 4.0": "X",                     # Costanera
    "explicit terms: use with credit, no redistribution": "X",    # Xorph barn
    # real-schems-v2 (2026-10-21): the v2 collection's wording (one string per listing, as its
    # collector recorded them), tiered by D14's rules. A: a stated general reuse permission,
    # credit requested or not, scope otherwise unspecified (as v1's A). B: no clear permission.
    # C: non-commercial or personal-use terms, and permissions only for playing it (servers,
    # maps, videos, single-player): a use-limited permission is not a reuse permission. None is
    # no-derivatives / no-redistribution / no-ML (the collector excluded those).
    "CC BY 4.0 (stated in listing)": "A",
    "CC BY 3.0": "A",
    "Free use with credit (creator statement); ML scope unspecified": "A",
    "informal credit-only reuse permission": "A",
    "Informal credit-only reuse permission": "A",
    "Informal reuse permission": "A",
    "informal reuse permission": "A",
    "Informal broad reuse permission with credit": "A",
    "Informal broad use permission with creator credit": "A",
    "Informal credit-and-link project reuse permission": "A",
    "Informal reuse permission with attribution; comprehensive licence unclear": "A",
    "Informal broad use invitation; comprehensive licence unclear": "A",
    "Informal community reuse invitation; comprehensive licence unclear": "A",
    "Informal credited third-party reuse permission; comprehensive licence unclear": "A",
    "Informal public-use permission with credit; comprehensive licence unclear": "A",
    "Informal use invitation with credit request; comprehensive licence unclear": "A",
    "Informal use permission with credit request; comprehensive licence unclear": "A",
    "Single-player download allowed; broader use with creator credit": "A",
    "Stated free-use permission; credit requested; scope not fully specified": "A",
    "Use and modifications permitted with credit; broader scope unspecified": "A",
    "Use permitted with creator credit; broader scope unclear": "A",
    "Use permitted with credit requested; broader scope unspecified": "A",
    "Use with credit statement; broader scope unspecified": "A",
    "permissive terms stated by builder (attribution requested)": "A",
    "permissive terms stated by builder (no formal licence)": "A",
    "stated reuse permission with credit; full scope unclear": "A",
    "license unclear (credit requested)": "B",
    "license unclear (creator requests credit for use beyond singleplayer)": "B",
    "license unclear (creator requires credit for server/video use)": "B",
    "Reuse invitation with contact and credit conditions; conditions unresolved": "B",
    "Specific reuse permission in comments; broader license unclear": "B",
    "CC BY-NC 4.0": "C",
    "CC BY-NC 4.0 (stated in listing)": "C",
    "CC BY-NC-SA 4.0": "C",
    "CC BY-NC-SA 4.0 (as stated on listing)": "C",
    "CC BY-NC-SA 4.0 (stated in listing)": "C",
    "CC BY-NC-SA4.0": "C",
    "CC BY-NC-SA 3.0 Unported": "C",
    "Custom reuse and modification permission; no sale or authorship misrepresentation": "C",
    "Custom reuse permission with attribution and profit-sharing condition": "C",
    "Personal viewing allowed; reuse licence unclear; no claiming authorship": "C",
    "personal/private use only; broader licence unclear": "C",
    "Single-player use/modification permitted; public server/video use with credit": "C",
    "World/server use permitted; video credit requested; broader scope unspecified": "C",
    "Informal map/server reuse with credit": "C",
    "Informal maps/server reuse permission with credit; comprehensive licence unclear": "C",
    "Informal project/server-use permission; broader license unclear": "C",
    "Informal server-use permission; comprehensive licence unclear": "C",
    "Server use permitted; project credit appreciated; broader licence unclear": "C",
    "informal attributed project/server-use permission": "C",
    "informal attributed server-use permission": "C",
    "informal server-use and room-edit permission": "C",
    "Video/stream use with credit; interior decoration invited; broader licence unclear": "C",
}

PASS_THROUGH = (
    "id", "file", "sha256", "file_type", "building_category", "creator", "source_website",
    "license_status", "dimensions_xyz_blocks", "non_air_blocks_from_metadata",
    "listing_completion_percent", "source_url", "build_name",
)
FLAGS = ("tier_x", "incomplete_listing", "oversized_capture", "legacy_format",
         "near_dup_candidate", "excluded_landscape_diorama", "excluded_target_lost_terrain",
         "excluded_display_stand", "excluded_two_builds", "excluded_incomplete_build",
         "geometric_duplicate", "duplicate_of_base", "base_split_crossing")

# D24 (2026-10-12): landscape dioramas set aside -- the building is a small part of a
# modelled landscape, so the crop cannot make a clean training target of it.
# D25 (2026-10-17): a cliff house whose rooms were dug into rock. The crop finds no natural
# terrain (`ground_fallback`) and removes the rock that formed the rooms' back walls, so the
# target opens them up: a wrong target (it was in test).
SET_ASIDE = {
    "pmc-traditional-house-cottage-style": "excluded_landscape_diorama",   # forest + road
    "pmc-the-lighthouse-5778696": "excluded_landscape_diorama",            # rock island
    "pmc-richelieu-contemporary": "excluded_target_lost_terrain",          # D25: cliff house
}
GEOMETRIC_DUPLICATES = "geometric_duplicates.json"
OVERSIZED_XZ = 200            # X or Z above this: a capture to crop in Phase 3

# Near-duplicate rules (metadata only; schematics are not parsed in this phase).
NON_AIR_TOL = 0.03            # same creator + same dims (90 deg rotation allowed) + this
NAME_RATIO = 0.90             # "very similar" normalised build_name, different creators
NAME_DIMS_TOL = 0.10          # ...promoted to a candidate only if the dims agree this well
NEAR_MISS_TOL = 0.10          # same creator, dims and non-air within 10%: listed for Phase 3

# Creator links (same split, reported). The site re-exports files under its own name, so a
# shared `nbt_author_or_exporter` of "abfielder" says nothing about who built them.
EXPORTERS = {"abfielder"}
CREATOR_RATIO = 0.90
CREATOR_MIN_CONTAINED = 6
UNKNOWN_CREATORS = {"", "unknown"}


class FreezeError(RuntimeError):
    pass


class SourceExists(FreezeError):
    pass


class VerificationError(FreezeError):
    pass


class UnknownLicence(FreezeError):
    pass


# --------------------------------------------------------------------------- freeze

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checksums(text: str) -> dict[str, str]:
    """`<sha256>  <path>` (or `<sha256> *<path>`) lines -> {relative path: sha256}."""
    out = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        m = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if not m:
            raise VerificationError(f"checksum line {n} is malformed: {line!r}")
        out[m.group(2)] = m.group(1).lower()
    return out


def list_files(root: Path) -> list[str]:
    """Every file under root, as sorted '/'-separated relative paths."""
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def hash_tree(root: Path) -> dict[str, str]:
    return {rel: sha256_file(root / rel) for rel in list_files(root)}


def load_manifest(root: Path) -> list[dict]:
    entries = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise FreezeError("manifest.json is not a list of entries")
    return entries


def verify_copy(src: Path, copy: Path) -> dict:
    """Check the copy three ways; raise VerificationError listing every mismatch."""
    problems = []
    copied = hash_tree(copy)
    original = hash_tree(src)
    if set(copied) != set(original):
        problems += [f"missing from copy: {p}" for p in sorted(set(original) - set(copied))]
        problems += [f"extra in copy: {p}" for p in sorted(set(copied) - set(original))]
    problems += [f"copy differs from original: {p}"
                 for p in sorted(set(copied) & set(original)) if copied[p] != original[p]]

    listed = parse_checksums((copy / "checksums.sha256").read_text(encoding="utf-8"))
    problems += [f"checksums.sha256: {p} {'missing' if p not in copied else 'mismatch'}"
                 for p, h in listed.items() if copied.get(p) != h]

    entries = load_manifest(copy)
    problems += [f"manifest {e.get('id')}: {e.get('file')} "
                 f"{'missing' if e.get('file') not in copied else 'sha256 mismatch'}"
                 for e in entries if copied.get(e.get("file")) != e.get("sha256")]
    if problems:
        raise VerificationError(f"{len(problems)} mismatch(es):\n  " + "\n  ".join(problems))
    return {"files": len(copied), "bytes": sum((copy / p).stat().st_size for p in copied),
            "checksum_lines": len(listed), "manifest_entries": len(entries),
            "hashes": copied}


def set_readonly(root: Path, readonly: bool = True) -> None:
    """attrib +R/-R /S /D on Windows; chmod elsewhere. Covers root itself."""
    if not root.exists():
        return
    if os.name == "nt":
        flag = "+R" if readonly else "-R"
        subprocess.run(["attrib", flag, str(root / "*"), "/S", "/D"], check=True,
                       capture_output=True)
        subprocess.run(["attrib", flag, str(root)], check=True, capture_output=True)
        return
    paths = [root, *root.rglob("*")]
    for p in (paths if not readonly else reversed(paths)):
        mode = p.stat().st_mode
        if readonly:
            p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        else:
            p.chmod(mode | stat.S_IWUSR)


def is_readonly(path: Path) -> bool:
    st = path.stat()
    if os.name == "nt":
        return bool(st.st_file_attributes & stat.FILE_ATTRIBUTE_READONLY)
    return not st.st_mode & stat.S_IWUSR


def remove_tree(root: Path) -> None:
    if root.exists():
        set_readonly(root, False)
        shutil.rmtree(root)


def write_checksums(path: Path, hashes: dict[str, str]) -> None:
    lines = "".join(f"{h}  {rel}\n" for rel, h in sorted(hashes.items()))
    path.write_text(lines, encoding="utf-8", newline="\n")


def freeze(src: Path, out: Path, force: bool = False, today: str | None = None) -> dict:
    """Copy src to out/source verbatim, verify, checksum, set read-only, write FREEZE.md."""
    src, out = Path(src), Path(out)
    dest, partial = out / "source", out / "source.partial"
    if dest.exists() and not force:
        raise SourceExists(f"{dest} exists; refusing to overwrite it (use --force)")
    if not (src / "manifest.json").is_file():
        raise FreezeError(f"{src} has no manifest.json")
    out.mkdir(parents=True, exist_ok=True)
    remove_tree(partial)                      # a stale partial copy was never frozen
    shutil.copytree(src, partial, copy_function=shutil.copy2)
    result = verify_copy(src, partial)        # raises; the partial copy stays for inspection
    remove_tree(dest)
    partial.rename(dest)
    write_checksums(out / "freeze_checksums.sha256", result["hashes"])
    set_readonly(dest, True)
    not_readonly = [rel for rel in result["hashes"] if not is_readonly(dest / rel)]
    if not_readonly:
        raise FreezeError(f"{len(not_readonly)} file(s) not read-only, e.g. {not_readonly[0]}")
    result["date"] = today or datetime.date.today().isoformat()
    result["source"] = str(src)
    _write_freeze_md(out / "FREEZE.md", result, load_manifest(dest))
    return result


def verify_frozen(out: Path) -> int:
    """Re-check out/source against freeze_checksums.sha256; return the file count."""
    dest = Path(out) / "source"
    listed = parse_checksums((Path(out) / "freeze_checksums.sha256").read_text(encoding="utf-8"))
    actual = hash_tree(dest)
    problems = [p for p in sorted(set(listed) | set(actual)) if listed.get(p) != actual.get(p)]
    if problems:
        raise VerificationError(f"frozen copy changed: {len(problems)} file(s), "
                                f"e.g. {problems[:5]}")
    return len(actual)


def _write_freeze_md(path: Path, r: dict, entries: list[dict]) -> None:
    by_ext = collections.Counter(Path(p).suffix.lower() or "(none)" for p in r["hashes"])
    by_type = collections.Counter(e["file_type"] for e in entries)
    ext_rows = "\n".join(f"| `{ext}` | {n} |" for ext, n in sorted(by_ext.items(),
                                                                 key=lambda kv: (-kv[1], kv[0])))
    type_rows = "\n".join(f"| `{t}` | {n} |" for t, n in sorted(by_type.items(),
                                                              key=lambda kv: (-kv[1], kv[0])))
    n = r["files"]
    path.write_text(f"""# {path.parent.name}: freeze record

**Never edit `source/`; derived files live beside it.** `source/` is a verbatim copy and is
set read-only. Regenerate derived files with `python -m schemset.freeze --out <this folder>`
(it refuses to touch `source/` without `--force`).

| | |
| --- | --- |
| Frozen | {r['date']} |
| Source | `{r['source']}` |
| Files | {n} ({r['bytes']:,} bytes) |
| Manifest entries | {r['manifest_entries']} |

## Files by extension (all of `source/`)

| Extension | Files |
| --- | ---: |
{ext_rows}

## Manifest entries by `file_type`

| Type | Entries |
| --- | ---: |
{type_rows}

## Verification

| Check | Result |
| --- | --- |
| Dataset `checksums.sha256`, every line, against the copy | {r['checksum_lines']} / {r['checksum_lines']} OK |
| Manifest `sha256`, every entry, against the copy | {r['manifest_entries']} / {r['manifest_entries']} OK |
| Every copied file byte-identical (sha256) to its original | {n} / {n} OK |
| `freeze_checksums.sha256` written | {n} lines |
| `source/` read-only | {n} / {n} files |
""", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- index

def creator_norm(creator: str | None) -> str:
    """Lowercase, whitespace and punctuation stripped (letters and digits only)."""
    s = unicodedata.normalize("NFKC", creator or "").casefold()
    return "".join(ch for ch in s if ch.isalnum())


def primary_creator_norm(creator: str | None) -> str:
    """creator_norm of the part before the first ';' (the rest are credits)."""
    return creator_norm((creator or "").split(";")[0])


def name_norm(name: str | None) -> str:
    s = unicodedata.normalize("NFKC", name or "").casefold()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)          # "(Full Interior)", "[Paradise]"
    s = re.sub(r"[^\w\s]|_", " ", s)
    return " ".join(s.split())


def licence_tier(status: str) -> str:
    try:
        return TIER_BY_LICENCE[status]
    except KeyError:
        raise UnknownLicence(f"unknown license_status: {status!r}") from None


def _percent(value) -> float | None:
    """60, "60", or the v2 collection's "60% complete" / "95% overall; ..." -> the number."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        m = re.match(r"\s*(\d+(?:\.\d+)?)", value)
        if not m:
            raise FreezeError(f"listing_completion_percent is not a number: {value!r}")
        value = m.group(1)
    return float(value)


def build_rows(entries: list[dict]) -> list[dict]:
    unknown = sorted({str(e.get("license_status")) for e in entries
                      if e.get("license_status") not in TIER_BY_LICENCE})
    if unknown:
        ids = {u: [e["id"] for e in entries if str(e.get("license_status")) == u] for u in unknown}
        raise UnknownLicence(f"unknown license_status value(s): {ids}")
    rows = []
    for e in entries:
        row = {k: e.get(k) for k in PASS_THROUGH}
        row["category_top"] = e["building_category"].split("/")[0]
        row["creator_norm"] = creator_norm(e["creator"])
        row["licence_tier"] = licence_tier(e["license_status"])
        x, _, z = e["dimensions_xyz_blocks"]
        pct = _percent(e.get("listing_completion_percent"))
        flags = []
        if row["licence_tier"] == "X":
            flags.append("tier_x")
        if pct is not None and pct < 100:
            flags.append("incomplete_listing")
        if x > OVERSIZED_XZ or z > OVERSIZED_XZ:
            flags.append("oversized_capture")
        if e["file_type"].lower() == ".schematic":
            flags.append("legacy_format")
        row["flags"] = flags
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- near-duplicates

def _dims_key(d) -> tuple:
    x, y, z = d
    return tuple(sorted((x, z))), y


def _close(a, b, tol) -> bool:
    return abs(a - b) <= tol * max(a, b)


def _dims_close(a, b, tol) -> bool:
    (xa, ya, za), (xb, yb, zb) = a, b
    return _close(ya, yb, tol) and ((_close(xa, xb, tol) and _close(za, zb, tol))
                                    or (_close(xa, zb, tol) and _close(za, xb, tol)))


def _non_air(a: dict, b: dict, tol: float) -> tuple[bool, str]:
    na, nb = a["non_air_blocks_from_metadata"], b["non_air_blocks_from_metadata"]
    if na is None or nb is None:
        return True, "non_air_blocks missing for one or both, cannot rule out"
    diff = abs(na - nb) / max(na, nb, 1)
    return diff <= tol, f"non_air {na} vs {nb} ({diff:.1%})"


def _pair(a: dict, b: dict) -> dict:
    return {"ids": [a["id"], b["id"]], "creators": [a["creator"], b["creator"]],
            "names": [a["build_name"], b["build_name"]],
            "dims": [a["dimensions_xyz_blocks"], b["dimensions_xyz_blocks"]]}


def find_near_duplicates(rows: list[dict]) -> dict:
    candidates, similar_names, near_misses = [], [], []
    for a, b in itertools.combinations(rows, 2):
        same_creator = a["creator_norm"] == b["creator_norm"]
        if _dims_key(a["dimensions_xyz_blocks"]) == _dims_key(b["dimensions_xyz_blocks"]):
            ok, why = _non_air(a, b, NON_AIR_TOL)
            if ok:
                rule = "same_creator_dims_nonair" if same_creator else "cross_creator_dims_nonair"
                candidates.append({**_pair(a, b), "rule": rule,
                                   "reason": f"same dims (90 deg rotation allowed); {why}"})
                continue
        if not same_creator:
            ratio = difflib.SequenceMatcher(None, name_norm(a["build_name"]),
                                            name_norm(b["build_name"])).ratio()
            if ratio >= NAME_RATIO:
                dims_ok = _dims_close(a["dimensions_xyz_blocks"], b["dimensions_xyz_blocks"],
                                      NAME_DIMS_TOL)
                entry = {**_pair(a, b), "name_ratio": round(ratio, 3)}
                if dims_ok:
                    candidates.append({**entry, "rule": "similar_name_dims",
                                       "reason": f"build_name ratio {ratio:.2f}, dims within "
                                                 f"{NAME_DIMS_TOL:.0%}"})
                else:
                    similar_names.append({**entry, "verdict": "listed only: dims differ by "
                                          f"more than {NAME_DIMS_TOL:.0%}, a shared generic name"})
        elif _dims_close(a["dimensions_xyz_blocks"], b["dimensions_xyz_blocks"], NEAR_MISS_TOL):
            ok, why = _non_air(a, b, NEAR_MISS_TOL)
            if ok:
                near_misses.append({**_pair(a, b), "reason": f"same creator, dims within "
                                    f"{NEAR_MISS_TOL:.0%}; {why}; check geometry in Phase 3"})
    uf = _UnionFind()
    for c in candidates:
        uf.union(*c["ids"])
    groups = sorted(sorted(g) for g in uf.groups().values())
    return {
        "method": {
            "data": "manifest metadata only; schematics are not parsed in Phase 2",
            "candidate_rules": {
                "same_creator_dims_nonair": f"same creator_norm, same dims allowing a 90 deg "
                                            f"rotation (sorted (X, Z) and Y), non_air within "
                                            f"{NON_AIR_TOL:.0%} (or missing)",
                "cross_creator_dims_nonair": "the same test across different creators "
                                             "(re-uploads)",
                "similar_name_dims": f"different creators, normalised build_name ratio >= "
                                     f"{NAME_RATIO}, dims within {NAME_DIMS_TOL:.0%}",
            },
            "similar_names": f"different creators, name ratio >= {NAME_RATIO}, dims differ: "
                             "listed for review, not flagged",
            "near_misses": f"same creator, dims and non_air within {NEAR_MISS_TOL:.0%}: listed "
                           "for a Phase 3 geometry check, not flagged",
        },
        "candidates": candidates,
        "groups": groups,
        "similar_names": similar_names,
        "near_misses": near_misses,
    }


# --------------------------------------------------------------------------- creators, splits

class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo

    def groups(self) -> dict[str, list[str]]:
        out = collections.defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return dict(out)


def creator_node(row: dict) -> str:
    """The unit that must not straddle splits. Unknown creators are one build each."""
    cn = row["creator_norm"]
    return f"unknown:{row['id']}" if cn in UNKNOWN_CREATORS else cn


def creator_links(rows: list[dict], entries_by_id: dict[str, dict]) -> list[dict]:
    """Pairs of creator nodes that look like one creator. Each link keeps its reason."""
    links = []
    by_primary = collections.defaultdict(set)
    by_nbt = collections.defaultdict(set)
    for r in rows:
        node = creator_node(r)
        if node.startswith("unknown:"):
            continue
        by_primary[primary_creator_norm(r["creator"])].add(node)
        nbt = creator_norm(entries_by_id[r["id"]].get("nbt_author_or_exporter"))
        if nbt and nbt not in EXPORTERS:
            by_nbt[nbt].add(node)
    for primary, nodes in sorted(by_primary.items()):
        for a, b in itertools.combinations(sorted(nodes), 2):
            links.append({"creators": [a, b], "reason": f"same primary creator "
                          f"'{primary}' (text before ';' is the creator, the rest credits)"})
    for nbt, nodes in sorted(by_nbt.items()):
        for a, b in itertools.combinations(sorted(nodes), 2):
            links.append({"creators": [a, b], "reason": f"same nbt_author_or_exporter '{nbt}'"})
    primaries = sorted(p for p in by_primary if p)
    for a, b in itertools.combinations(primaries, 2):
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        short, long_ = sorted((a, b), key=len)
        if ratio >= CREATOR_RATIO or (len(short) >= CREATOR_MIN_CONTAINED and short in long_):
            for na, nb in itertools.product(sorted(by_primary[a]), sorted(by_primary[b])):
                links.append({"creators": [na, nb], "reason": f"near-identical creator names "
                              f"'{a}' / '{b}' (ratio {ratio:.2f})"})
    return links


def creator_notes(rows: list[dict], entries_by_id: dict[str, dict]) -> dict:
    """Informational: raw names that collapse, nbt authors that differ from the creator."""
    raw = collections.defaultdict(set)
    for r in rows:
        raw[r["creator_norm"]].add(r["creator"])
    collapsed = {cn: sorted(names) for cn, names in sorted(raw.items()) if len(names) > 1}
    differs = []
    for r in rows:
        nbt = entries_by_id[r["id"]].get("nbt_author_or_exporter")
        if nbt and creator_norm(nbt) != r["creator_norm"]:
            differs.append({"id": r["id"], "creator": r["creator"], "nbt_author_or_exporter": nbt,
                            "exporter": creator_norm(nbt) in EXPORTERS})
    credits = [{"id": r["id"], "creator": r["creator"]} for r in rows if ";" in r["creator"]]
    return {"collapsed_by_norm": collapsed, "nbt_author_differs": differs,
            "credit_suffixes": credits}


def assign_splits(rows: list[dict], links: list[tuple[str, str]], seed: int = SEED,
                  fractions: dict[str, float] = FRACTIONS) -> tuple[dict[str, str], list[dict]]:
    """Creator groups -> splits, roughly stratified by top-level category.

    Groups (creators joined by `links`) are placed largest first, ties in a seeded shuffle.
    Each goes to the split where it most reduces the squared shortfall against the targets
    (fraction x total, and fraction x count per category), each term divided by its target
    so small splits and categories weigh as much as large ones. Tier X is "excluded".
    """
    assignment = {r["id"]: "excluded" for r in rows if r["licence_tier"] == "X"}
    eligible = [r for r in rows if r["licence_tier"] != "X"]
    uf = _UnionFind()
    for r in eligible:
        uf.add(creator_node(r))
    nodes = set(uf.parent)
    for a, b in links:
        if a in nodes and b in nodes:
            uf.union(a, b)
    members = collections.defaultdict(list)
    for r in eligible:
        members[uf.find(creator_node(r))].append(r)

    total = len(eligible)
    per_cat = collections.Counter(r["category_top"] for r in eligible)
    target = {s: fractions[s] * total for s in SPLITS}
    target_cat = {s: {c: fractions[s] * n for c, n in per_cat.items()} for s in SPLITS}
    count = {s: 0 for s in SPLITS}
    count_cat = {s: collections.Counter() for s in SPLITS}

    order = sorted(members)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda k: -len(members[k]))

    def cost_change(s: str, size: int, cats: collections.Counter) -> float:
        d = ((count[s] + size - target[s]) ** 2 - (count[s] - target[s]) ** 2) / target[s]
        for c, k in cats.items():
            t, n = target_cat[s][c], count_cat[s][c]
            d += ((n + k - t) ** 2 - (n - t) ** 2) / max(t, 1.0)
        return d

    groups = []
    for key in order:
        g = members[key]
        cats = collections.Counter(r["category_top"] for r in g)
        split = min(SPLITS, key=lambda s: (cost_change(s, len(g), cats), SPLITS.index(s)))
        count[split] += len(g)
        count_cat[split].update(cats)
        for r in g:
            assignment[r["id"]] = split
        groups.append({"key": key, "split": split, "builds": len(g),
                       "creators": sorted({creator_node(r) for r in g}),
                       "ids": sorted(r["id"] for r in g)})
    groups.sort(key=lambda g: g["key"])
    return assignment, groups


def split_counts(rows: list[dict]) -> dict:
    out = {}
    for s in (*SPLITS, "excluded"):
        rs = [r for r in rows if r["split"] == s]
        out[s] = {"builds": len(rs),
                  "by_category": dict(sorted(collections.Counter(
                      r["category_top"] for r in rs).items())),
                  "by_tier": dict(sorted(collections.Counter(
                      r["licence_tier"] for r in rs).items())),
                  "creators": len({creator_node(r) for r in rs})}
    return out


def derive(out: Path, seed: int = SEED, base: Path | None = None) -> dict:
    """index.json, near_duplicates.json, splits.json from out/source (deterministic).

    With `base` (an earlier frozen dataset, e.g. real-schems-v1), the result is the combined
    dataset: the base's rows come first, exactly as its index.json has them (splits, groups,
    flags), with `source_root` pointing at its own frozen copy, so nothing of it is copied or
    changed; this dataset's builds follow (`derive_combined`)."""
    out = Path(out)
    if base is not None:
        return derive_combined(out, Path(base), seed)
    entries = load_manifest(out / "source")
    entries_by_id = {e["id"]: e for e in entries}
    if len(entries_by_id) != len(entries):
        raise FreezeError("manifest ids are not unique")
    rows = build_rows(entries)

    near = find_near_duplicates(rows)
    flagged = {i for c in near["candidates"] for i in c["ids"]}
    by_id = {r["id"]: r for r in rows}
    for i in sorted(flagged):
        by_id[i]["flags"].append("near_dup_candidate")

    links = creator_links(rows, entries_by_id)
    pair_links = [tuple(l["creators"]) for l in links]
    pair_links += [(creator_node(by_id[a]), creator_node(by_id[b]))
                   for c in near["candidates"] for a, b in [c["ids"]]]
    assignment, groups = assign_splits(rows, pair_links, seed=seed)
    group_of = {i: g["key"] for g in groups for i in g["ids"]}
    for r in rows:
        r["split"] = assignment[r["id"]]
        r["creator_group"] = group_of.get(r["id"])

    overrides = apply_overrides(rows, groups, out / GEOMETRIC_DUPLICATES)

    counts = split_counts(rows)
    index = {"dataset": out.name, "rows_count": len(rows),
             "tier_mapping": TIER_BY_LICENCE, "flags": list(FLAGS),
             "fields": [*PASS_THROUGH, "category_top", "creator_norm", "licence_tier", "flags",
                        "split", "creator_group"],
             "rows": rows}
    splits = {
        "dataset": out.name, "seed": seed, "fractions": FRACTIONS,
        "method": ("Tier X -> excluded. The rest is grouped by creator_norm, joined by creator "
                   "links and near-duplicate candidates (union-find); unknown creators are one "
                   "build each. Groups are placed largest first (ties: seeded shuffle) into the "
                   "split that most reduces the target-normalised squared shortfall against "
                   "fraction x total and fraction x count per top-level building_category."),
        "counts": counts,
        "creator_links": links,
        "creator_notes": creator_notes(rows, entries_by_id),
        "groups": groups,
        "overrides": overrides,
        "assignments": {r["id"]: r["split"] for r in rows},
    }
    _write_json(out / "index.json", index)
    _write_json(out / "near_duplicates.json", near)
    _write_json(out / "splits.json", splits)
    return {"rows": rows, "near": near, "splits": splits}


# D26 (2026-10-22): builds of the new collection set aside after the normalize-6 review (crop
# steps 9-11 already applied), by id -> flag. What is left of them is not a clean target of one
# building: a diorama (D24's flag), a building on a display stand, two unrelated builds in one
# capture, an unfinished build. Their files stay frozen; they get split "excluded".
SET_ASIDE_NEW: dict[str, str] = {
    "pmc-beautiful-modern-cottage": "excluded_landscape_diorama",        # a house in a forest
    "pmc-regional-airport-3328707": "excluded_landscape_diorama",        # airfield, parked planes
    "pmc-tamas-manor": "excluded_landscape_diorama",                     # terrain cube: debris left
    "pmc-lighthouse-timelapse-schematic-1-17-1": "excluded_landscape_diorama",   # rock arches
    "pmc-the-bates-motel-from-psycho": "excluded_landscape_diorama",     # two buildings on a hill
    "pmc-modern-house-4-3331081": "excluded_landscape_diorama",          # a whole walled garden plot
    "pmc-alpine-chalet": "excluded_landscape_diorama",                   # snow plate and spruces
    "pmc-single-story-barn-conversion-modern-house": "excluded_landscape_diorama",   # in a wood
    "pmc-simple-house-2990730": "excluded_landscape_diorama",            # a garden of follies
    "pmc-hotel-de-angeli": "excluded_display_stand",                     # on a geology cross-section
    "pmc-modern-house-2196729": "excluded_display_stand",                # on a stone pedestal
    "pmc-clayleaf-nearly-fully-furnished-brick-house": "excluded_two_builds",
    "pmc-graumans-chinese-theatre---hollywood-landmark": "excluded_incomplete_build",  # hall a box
}


def derive_combined(out: Path, base: Path, seed: int = SEED) -> dict:
    """The combined dataset: `base`'s rows unchanged (its splits are fixed, so every result on
    its val / test builds stays comparable), then the new collection's (`out/source`).

    A new build joins the split of any base creator group it is linked to (same creator,
    creator links, near-duplicate candidates): a creator never straddles train and test.
    Otherwise it goes to train. A new build whose links join base groups of two splits is
    excluded (`base_split_crossing`): either split would leak the other. Tier X is excluded.
    After `schemset.dedupe`, a new build that duplicates a base build of another split is
    excluded (`duplicate_of_base`) rather than moving either: the base never moves.

    The v2 collection records its non-air counts under `geometry_metrics` (counted from the
    parsed NBT) and leaves `non_air_blocks_from_metadata` empty; that count fills it
    (`non_air_basis`), so the near-duplicate test and normalize's validation can use it.
    """
    entries = load_manifest(out / "source")
    base_index = json.loads((base / "index.json").read_text(encoding="utf-8"))
    base_entries = load_manifest(base / "source")
    entries_by_id = {e["id"]: e for e in base_entries}
    clash = sorted(e["id"] for e in entries if e["id"] in entries_by_id)
    if clash:
        raise FreezeError(f"ids in both collections: {clash[:5]}")
    entries_by_id.update({e["id"]: e for e in entries})
    root = Path(os.path.relpath(base / "source", out)).as_posix()
    base_rows = []
    for r in base_index["rows"]:
        r = dict(r)
        r["source_root"], r["origin"] = root, base.name
        base_rows.append(r)
    new_rows = build_rows(entries)
    for r in new_rows:
        r["source_root"], r["origin"] = "source", out.name
        counted = (entries_by_id[r["id"]].get("geometry_metrics") or {}).get("non_air_blocks_counted")
        if r["non_air_blocks_from_metadata"] is None and counted is not None:
            r["non_air_blocks_from_metadata"], r["non_air_basis"] = int(counted), "geometry_metrics"
    rows = base_rows + new_rows
    by_id = {r["id"]: r for r in rows}

    # A base build without a metadata count is tested with the count its own normalize read
    # (the row itself stays as the base has it): a church and a house that only share their
    # dims (the v1 Coleraine church, 5,915 blocks, against a 3,878-block house) are no pair.
    measured = {}
    for r in base_rows:
        rec = base / "normalized" / "records" / f"{r['id']}.json"
        if r["non_air_blocks_from_metadata"] is None and rec.is_file():
            n = json.loads(rec.read_text(encoding="utf-8")).get("validation", {}).get("non_air_read")
            if n is not None:
                measured[r["id"]] = int(n)
    near = find_near_duplicates([{**r, "non_air_blocks_from_metadata": measured[r["id"]]}
                                 if r["id"] in measured else r for r in rows])
    near["method"]["measured_base_counts"] = (f"{len(measured)} {base.name} builds without a "
                                              "metadata count use the count its normalize read")
    for c in near["candidates"]:
        for i in c["ids"]:
            if by_id[i]["origin"] == out.name and "near_dup_candidate" not in by_id[i]["flags"]:
                by_id[i]["flags"].append("near_dup_candidate")
    links = creator_links(rows, entries_by_id)
    pair_links = [tuple(l["creators"]) for l in links]
    pair_links += [(creator_node(by_id[a]), creator_node(by_id[b]))
                   for c in near["candidates"] for a, b in [c["ids"]]]

    uf = _UnionFind()
    for r in rows:
        if r["licence_tier"] != "X":
            uf.add(creator_node(r))
    nodes = set(uf.parent)
    for a, b in pair_links:
        if a in nodes and b in nodes:
            uf.union(a, b)
    members = collections.defaultdict(list)
    for r in rows:
        if r["licence_tier"] != "X":
            members[uf.find(creator_node(r))].append(r)
    groups, crossings = [], []
    for key in sorted(members):
        g = members[key]
        base_splits = collections.Counter(r["split"] for r in g if r["origin"] == base.name
                                          and r["split"] != "excluded")
        if len(base_splits) > 1:                  # a new build links base groups of two splits:
            crossings.append({"group": key, "base_splits": dict(base_splits),   # its builds
                              "new_excluded": sorted(r["id"] for r in g        # are excluded
                                                     if r["origin"] == out.name)})
        split = base_splits.most_common(1)[0][0] if base_splits else "train"
        crossing = len(base_splits) > 1
        for r in g:
            if r["origin"] == out.name:
                r["split"], r["creator_group"] = ("excluded" if crossing else split), key
                if crossing:
                    r["flags"].append("base_split_crossing")
        groups.append({"key": key, "split": split, "builds": len(g),
                       "creators": sorted({creator_node(r) for r in g}),
                       "ids": sorted(r["id"] for r in g),
                       "new": sum(r["origin"] == out.name for r in g)})
    for r in new_rows:
        if r["licence_tier"] == "X":
            r["split"], r["creator_group"] = "excluded", None

    set_aside, dup_excluded = [], []
    for bid, flag in sorted(SET_ASIDE_NEW.items()):
        if bid in by_id and by_id[bid]["origin"] == out.name:
            set_aside.append({"id": bid, "from": by_id[bid]["split"], "flag": flag})
            by_id[bid]["split"] = "excluded"
            if flag not in by_id[bid]["flags"]:
                by_id[bid]["flags"].append(flag)
    dup_path = out / GEOMETRIC_DUPLICATES
    if dup_path.is_file():
        for p in json.loads(dup_path.read_text(encoding="utf-8"))["pairs"]:
            if p.get("verdict") != "duplicate" or not all(i in by_id for i in p["ids"]):
                continue
            a, b = (by_id[i] for i in p["ids"])
            for r in (a, b):
                if r["origin"] == out.name and "geometric_duplicate" not in r["flags"]:
                    r["flags"].append("geometric_duplicate")
            olds = [r for r in (a, b) if r["origin"] == base.name]
            news = [r for r in (a, b) if r["origin"] == out.name]
            if olds and news and news[0]["split"] != "excluded" and olds[0]["split"] != news[0]["split"]:
                news[0]["flags"].append("duplicate_of_base")
                dup_excluded.append({"id": news[0]["id"], "duplicate_of": olds[0]["id"],
                                     "base_split": olds[0]["split"], "was": news[0]["split"]})
                news[0]["split"] = "excluded"
            elif len(news) == 2 and a["split"] != b["split"] and "excluded" not in (a["split"], b["split"]):
                small = min(news, key=lambda r: (len(members[uf.find(creator_node(r))]), r["id"]))
                other = a if small is b else b
                dup_excluded.append({"id": small["id"], "duplicate_of": other["id"], "moved_to": other["split"]})
                small["split"] = other["split"]

    counts = split_counts(rows)
    new_counts = split_counts(new_rows)
    index = {"dataset": out.name, "base": base.name, "rows_count": len(rows),
             "tier_mapping": TIER_BY_LICENCE, "flags": list(FLAGS),
             "fields": [*PASS_THROUGH, "category_top", "creator_norm", "licence_tier", "flags",
                        "split", "creator_group", "source_root", "origin"],
             "rows": rows}
    splits = {
        "dataset": out.name, "base": base.name, "seed": seed, "fractions": FRACTIONS,
        "method": (f"{base.name}'s rows and splits unchanged. A new build joins the split of "
                   "any base creator group it is linked to (creator_norm, creator links, "
                   "near-duplicate candidates; union-find); otherwise train. Tier X excluded. "
                   "A new build duplicating a base build of another split is excluded."),
        "counts": counts, "new_counts": new_counts,
        "creator_links": links, "creator_notes": creator_notes(rows, entries_by_id),
        "groups": groups, "base_split_crossings": crossings,
        "overrides": {"set_aside_new": set_aside, "duplicates": dup_excluded},
        "assignments": {r["id"]: r["split"] for r in rows},
    }
    _write_json(out / "index.json", index)
    _write_json(out / "near_duplicates.json", near)
    _write_json(out / "splits.json", splits)
    return {"rows": rows, "near": near, "splits": splits}


def apply_overrides(rows: list[dict], groups: list[dict], duplicates: Path) -> dict:
    """D24 set-asides, then geometric-duplicate moves (see the module docstring)."""
    by_id = {r["id"]: r for r in rows}
    set_aside = []
    for bid, flag in sorted(SET_ASIDE.items()):
        if bid in by_id:
            r = by_id[bid]
            set_aside.append({"id": bid, "from": r["split"], "flag": flag})
            r["split"] = "excluded"
            if flag not in r["flags"]:
                r["flags"].append(flag)
    moves = []
    if duplicates.is_file():
        pairs = [p["ids"] for p in json.loads(duplicates.read_text(encoding="utf-8"))["pairs"]
                 if p.get("verdict") == "duplicate" and all(i in by_id for i in p["ids"])]
        for a, b in pairs:
            for i in (a, b):
                if "geometric_duplicate" not in by_id[i]["flags"]:
                    by_id[i]["flags"].append("geometric_duplicate")
        members = collections.defaultdict(list)
        for r in rows:
            if r["split"] != "excluded":
                members[r["creator_group"]].append(r)
        uf = _UnionFind()
        for a, b in pairs:
            ga, gb = by_id[a]["creator_group"], by_id[b]["creator_group"]
            if ga and gb and "excluded" not in (by_id[a]["split"], by_id[b]["split"]):
                uf.union(ga, gb)
        for root, keys in sorted(uf.groups().items()):
            splits = {members[k][0]["split"] for k in keys}
            if len(keys) < 2 or len(splits) < 2:
                continue
            target_key = max(sorted(keys), key=lambda k: len(members[k]))
            target = members[target_key][0]["split"]
            for k in sorted(keys):
                if members[k][0]["split"] != target:
                    ids = sorted(r["id"] for r in members[k])
                    moves.append({"creator_group": k, "ids": ids,
                                  "from": members[k][0]["split"], "to": target,
                                  "reason": "geometric duplicate of a build in creator group "
                                            f"'{target_key}'"})
                    for r in members[k]:
                        r["split"] = target
    for g in groups:
        kept = sorted({by_id[i]["split"] for i in g["ids"]} - {"excluded"})
        g["split"] = kept[0] if kept else "excluded"
    return {"set_aside_d24": set_aside, "duplicate_moves": moves}


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")


# --------------------------------------------------------------------------- CLI

def _table(counts: dict, key: str) -> str:
    cols = sorted({c for s in counts.values() for c in s[key]})
    lines = ["split      " + " ".join(f"{c[:11]:>11s}" for c in cols) + "       total"]
    for s, v in counts.items():
        lines.append(f"{s:10s} " + " ".join(f"{v[key].get(c, 0):11d}" for c in cols)
                     + f" {v['builds']:11d}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, help="the dataset folder (read, never written)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force", action="store_true", help="re-copy over an existing source/")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--base", type=Path, default=None,
                    help="an earlier frozen dataset whose rows and splits the combined index keeps unchanged")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    dest = args.out / "source"
    if dest.exists() and not args.force:
        n = verify_frozen(args.out)
        print(f"source/ already frozen: not overwritten (--force re-copies). "
              f"{n} files match freeze_checksums.sha256.")
        if args.source and (args.source / "checksums.sha256").is_file() and \
                (args.source / "checksums.sha256").read_bytes() != \
                (dest / "checksums.sha256").read_bytes():
            print("WARNING: the source's checksums.sha256 differs from the frozen copy's.")
    else:
        if not args.source:
            ap.error("--source is required to freeze")
        r = freeze(args.source, args.out, force=args.force)
        print(f"frozen {r['files']} files ({r['bytes']:,} bytes) -> {dest}")
        print(f"verified: {r['checksum_lines']} checksum lines, {r['manifest_entries']} manifest "
              f"sha256, {r['files']} files byte-identical; source/ read-only")

    d = derive(args.out, seed=args.seed, base=args.base)
    rows = d["rows"]
    print(f"index: {len(rows)} rows")
    print("tiers:", dict(sorted(collections.Counter(r["licence_tier"] for r in rows).items())))
    print("flags:", {f: sum(f in r["flags"] for r in rows) for f in FLAGS})
    near = d["near"]
    print(f"near-duplicates: {len(near['candidates'])} candidate pair(s), "
          f"{len(near['similar_names'])} similar-name pair(s) listed, "
          f"{len(near['near_misses'])} near miss(es)")
    counts = d["splits"]["counts"]
    print(_table(counts, "by_category"))
    print(_table(counts, "by_tier"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
