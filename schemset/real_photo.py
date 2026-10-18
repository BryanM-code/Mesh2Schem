#!/usr/bin/env python3
"""Phase 4 follow-up: real-photo end-to-end tests -- photos of the real buildings that some
builds replicate, through the product pipeline: the app's background removal (a rembg
cutout) -> TRELLIS -> voxelise -> align (yaw, shift and scale) -> compare with the target.

Usage:  python -m schemset.real_photo fetch    --dataset data/real-schems-v1
        <python with rembg> -m schemset.real_photo cutout --dataset data/real-schems-v1
        python -m schemset.real_photo generate --dataset data/real-schems-v1
        python -m schemset.real_photo place    --dataset data/real-schems-v1

`PHOTOS` lists the photos: a photo the user supplied (`real_photo_tests/`) and free-licence
Wikimedia Commons photos of replicated landmarks (`real_photo_replicas/`, licence, author and
URL recorded in each folder's meta.json). Photos stay local under data/ (gitignored) and are
never redistributed. A replica in train is a qualitative check only and stays out of test
metrics (`split` is recorded on every row).

Steps:

    fetch     download each Commons photo once (the original file, via the Commons API) into
              pairs/real_photo_replicas/<id>/photo.<ext>, with meta.json
    cutout    background removal as the app does it (Real2Minecraft trellis_client
              `_prepare_cutout`: rembg `isnet-general-use`, the photo clamped to 1024 px, the
              mask kept only if it covers 5-97% of the frame, else the raw photo is sent).
              Needs rembg, which this repo's environment does not install: run this step
              with a Python that has it (the app's)
    generate  the cutout (or the photo) through TRELLIS, seed 1 (`synth.trellis_compare`,
              the settings Real2Minecraft sends); resumable
    place     `real_trellis.align` with a scale search as well (`SCALES` x the height fit,
              then `ASPECTS` horizontally): the build is its builder's interpretation, so
              proportions differ. Then validation, the input statistics, the detail metric
              (`schemset.detail`) and the colour bias against the build's own colours

Layout: pairs/<folder>/<id>/ holds cutout.png, trellis.glb, input.npz, stats.json (and the
downloaded photo for replicas); pairs/real_photo_index.json collects every row.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TRELLIS_SEED = 1
REMBG_MODEL = "isnet-general-use"            # as Real2Minecraft
MAX_SIDE = 1024
MASK_RANGE = (0.05, 0.97)
SCALES = tuple(round(0.6 + 0.05 * i, 2) for i in range(19))      # 0.6 .. 1.5 x the height fit
ASPECTS = tuple(round(0.7 + 0.05 * i, 2) for i in range(15))     # 0.7 .. 1.4 x horizontally
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "BuildDataCollector-real-photo-test/1.0 (local research; photos kept private)"

PHOTOS = [
    {"id": "ab-5731", "folder": "real_photo_tests", "file": "Screenshot 2026-10-15 181212.png",
     "subject": "Barn by Keralis (a barn in golden-hour light)",
     "source": "supplied by the user, 2026-10-15 (named ab-5731_barn.jpg in the request; the "
               "folder held this screenshot, which shows the described barn)"},
    {"id": "pmc-victorian-queen-anne-style---carson-mansion", "folder": "real_photo_replicas",
     "commons": "2014.10.12.094700 Carson Mansion Eureka California.jpg", "subject": "Carson Mansion"},
    {"id": "pmc-zhytomyr-bus-station", "folder": "real_photo_replicas",
     "commons": "\u0410\u0432\u0442\u043e\u0432\u043e\u043a\u0437\u0430\u043b \u0416\u0438\u0442\u043e\u043c\u0438\u0440.jpg",
     "subject": "Zhytomyr bus station"},
    {"id": "pmc-coleraine-church", "folder": "real_photo_replicas",
     "commons": "Coleraine Methodist church (geograph 2693191).jpg",
     "subject": "Coleraine Methodist Church (same church as the build: not confirmed)"},
    {"id": "pmc-1-20-1-woolworth-building-new-york-city", "folder": "real_photo_replicas",
     "commons": "Exterior Woolworth Building 2015.jpg", "subject": "Woolworth Building"},
    {"id": "pmc-122-leadenhall-street-3162614", "folder": "real_photo_replicas",
     "commons": "Leadenhall Building (aka the Cheesegrater) (28571108492).jpg", "subject": "122 Leadenhall Street"},
    {"id": "pmc-swiss-federal-palace-bundeshaus", "folder": "real_photo_replicas",
     "commons": "Federal Palace of Switzerland Bundeshaus im Bern by Robbie Conceptuel.png",
     "subject": "Swiss Federal Palace"},
    {"id": "pmc-west-virginia-state-capitol-replica", "folder": "real_photo_replicas",
     "commons": "The West Virginia State Capitol building - Image03.jpg", "subject": "West Virginia State Capitol"},
    {"id": "pmc-alabama-state-capitol-building", "folder": "real_photo_replicas",
     "commons": "Alabama state capitol, Montgomery.jpg", "subject": "Alabama State Capitol"},
    {"id": "pmc-toronto-fairmont-royal-york-java-1-17-1", "folder": "real_photo_replicas",
     "commons": "Fairmont Royal York Hotel, Toronto (27823283571).jpg", "subject": "Fairmont Royal York"},
]


def build_dir(dataset: Path, p: dict) -> Path:
    return Path(dataset) / "pairs" / p["folder"] / p["id"]


def photo_path(dataset: Path, p: dict) -> Path | None:
    """The photo as supplied: the user's file in its folder, or the downloaded original."""
    if "file" in p:
        path = Path(dataset) / "pairs" / p["folder"] / p["file"]
        return path if path.is_file() else None
    found = sorted(build_dir(dataset, p).glob("photo.*"))
    return found[0] if found else None


# --------------------------------------------------------------------------- fetch

def _get(url: str, timeout: int = 300, tries: int = 5) -> bytes:
    """GET with Commons' etiquette: a descriptive User-Agent, and on HTTP 429 a pause (its
    Retry-After, else 30 s doubling) before trying again."""
    import urllib.error
    import urllib.request
    wait = 30.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == tries - 1:
                raise
            pause = float(exc.headers.get("Retry-After") or wait)
            print(f"    rate-limited; waiting {pause:.0f} s", flush=True)
            time.sleep(pause)
            wait *= 2
    raise RuntimeError("unreachable")


def _commons_info(title: str) -> dict:
    import urllib.parse
    q = urllib.parse.urlencode({"action": "query", "format": "json", "formatversion": "2",
                                "prop": "imageinfo", "iiprop": "url|size|sha1|mime|extmetadata",
                                "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit",
                                "titles": "File:" + title})
    page = json.loads(_get(f"{COMMONS_API}?{q}", timeout=60))["query"]["pages"][0]
    if "imageinfo" not in page:
        raise RuntimeError(f"not on Commons: {title}")
    return page["imageinfo"][0]


def _strip_html(s: str) -> str:
    import html
    import re
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def fetch(dataset: Path) -> list:
    """Download every Commons photo not yet present (original file, sha1 checked)."""
    import hashlib
    import urllib.parse
    done = []
    for p in PHOTOS:
        if "commons" not in p:
            continue
        d = build_dir(dataset, p)
        if photo_path(dataset, p):
            done.append(p["id"])
            continue
        d.mkdir(parents=True, exist_ok=True)
        info = _commons_info(p["commons"])
        em = info.get("extmetadata", {})
        url = info["url"].split("?", 1)[0]
        ext = Path(urllib.parse.urlparse(url).path).suffix.lower() or ".jpg"
        time.sleep(5)                                   # be gentle with the file servers
        data = _get(url)
        if hashlib.sha1(data).hexdigest() != info.get("sha1"):
            raise RuntimeError(f"{p['id']}: download does not match Commons' sha1")
        (d / f"photo{ext}").write_bytes(data)
        meta = {"id": p["id"], "subject": p["subject"], "commons_title": p["commons"],
                "page": "https://commons.wikimedia.org/wiki/File:" + p["commons"].replace(" ", "_"),
                "file_url": url, "licence": _strip_html(em.get("LicenseShortName", {}).get("value")),
                "licence_url": _strip_html(em.get("LicenseUrl", {}).get("value")),
                "author": _strip_html(em.get("Artist", {}).get("value")),
                "credit": _strip_html(em.get("Credit", {}).get("value")),
                "bytes": len(data), "width": info.get("width"), "height": info.get("height"),
                "sha1": info.get("sha1"), "sha256": hashlib.sha256(data).hexdigest(),
                "retrieved": time.strftime("%Y-%m-%d"),
                "use": "local evaluation only; never redistributed"}
        (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  {p['id']}: {len(data) / 1e6:.2f} MB, {meta['licence']}, {meta['author']}", flush=True)
        done.append(p["id"])
    return done


# --------------------------------------------------------------------------- cutout

def cutout(dataset: Path) -> list:
    """Background removal as Real2Minecraft does it (see the module docstring)."""
    import rembg
    from PIL import Image
    session = rembg.new_session(REMBG_MODEL, providers=["CPUExecutionProvider"])
    rows = []
    for p in PHOTOS:
        src = photo_path(dataset, p)
        if src is None:
            print(f"  {p['id']}: no photo", flush=True)
            continue
        d = build_dir(dataset, p)
        d.mkdir(parents=True, exist_ok=True)
        img = Image.open(src).convert("RGB")
        if max(img.size) > MAX_SIDE:
            s = MAX_SIDE / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
        cut = rembg.remove(img, session=session)
        frac = float((np.asarray(cut)[:, :, 3] > 128).mean())
        ok = MASK_RANGE[0] <= frac <= MASK_RANGE[1]
        if ok:
            cut.save(d / "cutout.png")
        else:
            img.save(d / "cutout_rejected_raw.png")
        rec = {"id": p["id"], "mask_fraction": round(frac, 4), "cutout_used": ok, "model": REMBG_MODEL,
               "size": list(img.size)}
        (d / "cutout.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        rows.append(rec)
        print(f"  {p['id']}: mask {frac:.2f} {'ok' if ok else 'REJECTED: the raw photo goes to TRELLIS'}", flush=True)
    return rows


def trellis_image(dataset: Path, p: dict) -> Path | None:
    """What goes to TRELLIS: the cutout, or the photo when the cutout was rejected (as the app)."""
    c = build_dir(dataset, p) / "cutout.png"
    return c if c.is_file() else photo_path(dataset, p)


# --------------------------------------------------------------------------- generate

def generate(dataset: Path, only=None) -> int:
    from synth.trellis_compare import TRELLIS_URL, server_up, trellis_generate
    if not server_up():
        raise SystemExit(f"TRELLIS server not reachable at {TRELLIS_URL} -- start C:\\TRELLIS\\run.bat first")
    n = 0
    for p in PHOTOS:
        if only and p["id"] not in only:
            continue
        d = build_dir(dataset, p)
        if (d / "trellis.glb").is_file():
            continue
        if not (d / "cutout.json").is_file():
            print(f"  {p['id']}: run cutout first", flush=True)
            continue
        t0 = time.time()
        trellis_generate(trellis_image(dataset, p), d / "trellis.glb", seed=TRELLIS_SEED)
        n += 1
        print(f"  {p['id']}: {time.time() - t0:.0f} s", flush=True)
    return n


# --------------------------------------------------------------------------- place

def synthetic_detail(dataset: Path, bid: str, variant: int = 1) -> dict | None:
    """The detail metric and colour cast of a synthetic pair, measured as for TRELLIS: its
    stored input against the reference placed with the variant's own yaw and offset."""
    from .detail import colour_cast, detail_stats, reference_model
    from .meshify import meshify_npz
    from types import SimpleNamespace
    rec_path = Path(dataset) / "pairs" / "records" / f"{bid}.json"
    npz = Path(dataset) / "pairs" / "synthetic" / bid / f"v{variant}.npz"
    if not rec_path.is_file() or not npz.is_file():
        return None
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    var = next((v for v in rec["variants"] if v["variant"] == variant and v["status"] == "ok"), None)
    if var is None:
        return None
    clean, _, _ = meshify_npz(Path(dataset) / "normalized" / f"{bid}.npz")
    z = np.load(npz)
    ref = reference_model(clean, z["target_shape"].shape, var["yaw_deg"], var["offset_blocks"])
    vm = SimpleNamespace(solid=z["input_occ"].astype(bool), surface=z["input_surface"].astype(bool),
                         rgb=z["input_rgb"])
    out = detail_stats(vm.solid, vm.rgb, vm.surface, ref.solid, ref.rgb, ref.surface)
    out.update(colour_cast(vm, ref))
    return out


def place(dataset: Path, only=None) -> dict:
    """Every generated real-photo mesh against its build (`real_trellis.measure` with the
    scale search), next to the same build's synthetic v1 and real-TRELLIS render sets."""
    from .real_trellis import measure
    rows_meta = {r["id"]: r for r in json.loads((Path(dataset) / "index.json").read_text(encoding="utf-8"))["rows"]}
    index_path = Path(dataset) / "pairs" / "real_photo_index.json"
    kept = {}
    if only and index_path.is_file():
        kept = {r["id"]: r for r in json.loads(index_path.read_text(encoding="utf-8"))["builds"]}
    rows = []
    for p in PHOTOS:
        if only and p["id"] not in only and p["id"] in kept:
            rows.append(kept[p["id"]])
            continue
        d = build_dir(dataset, p)
        meta = rows_meta[p["id"]]
        row = {"id": p["id"], "subject": p["subject"], "folder": p["folder"], "split": meta["split"],
               "category": meta.get("category_top"), "build_name": meta.get("build_name"),
               "in_test_metrics": meta["split"] == "test"}
        if (d / "meta.json").is_file():
            m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            row["photo"] = {k: m.get(k) for k in ("page", "licence", "author", "commons_title")}
        else:
            row["photo"] = {"source": p.get("source")}
        if (d / "cutout.json").is_file():
            row["cutout"] = json.loads((d / "cutout.json").read_text(encoding="utf-8"))
        if not (d / "trellis.glb").is_file():
            row["status"] = "awaiting_trellis"
            rows.append(row)
            continue
        row.update(measure(dataset, p["id"], d / "trellis.glb", d / "input.npz", scales=SCALES, aspects=ASPECTS))
        _compare(dataset, row)
        (d / "stats.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
        pl, v = row["placement"], row["validation"]
        print(f"  {p['id']}: IoU {v.get('iou')} scale x{pl.get('scale_mult')} aspect x{pl.get('aspect')} "
              f"yaw {pl['yaw_deg']} residual {v.get('shift_residual_cells')} "
              f"detail {row['stats'].get('detail_geo')} / {row['stats'].get('detail_colour')}", flush=True)
        rows.append(row)
    res = {"version": "real-photo-1", "builds": rows, "summary": summarize(rows),
           "caveats": ["Each build is its builder's interpretation of the real building: proportions, "
                       "scale and detail differ, so IoU and detail are lower bounds on TRELLIS's quality.",
                       "Train-split replicas are qualitative checks only (in_test_metrics false).",
                       "One TRELLIS seed per photo; photos as found (angle, crop, light, clutter)."]}
    index_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def _compare(dataset: Path, row: dict) -> None:
    """The same build's synthetic v1 (its pair record's statistics, plus the detail metric
    if the record predates it) and its real-TRELLIS render sets, next to a real-photo row."""
    from .real_trellis import SETS
    bid = row["id"]
    syn = None
    rec_path = Path(dataset) / "pairs" / "records" / f"{bid}.json"
    if rec_path.is_file():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        v1 = next((v for v in rec["variants"] if v["variant"] == 1 and v["status"] == "ok"), None)
        if v1:
            syn = {"iou": v1["validation"]["iou"], "pairs_version": rec.get("version"), **v1["stats"]}
            if "detail_geo" not in syn:
                syn.update(synthetic_detail(dataset, bid, 1) or {})
    row["synthetic_v1"] = syn
    row["render_sets"] = {}
    for name, folder in SETS.items():
        sp = Path(dataset) / "pairs" / folder / bid / "stats.json"
        if sp.is_file():
            s = json.loads(sp.read_text(encoding="utf-8"))
            if s.get("status") == "ok":
                row["render_sets"][name] = {"iou": s["validation"].get("iou"), **s["stats"]}


def refresh(dataset: Path) -> dict:
    """Update the comparison columns (synthetic v1, render sets) and the summary of an
    existing real_photo_index.json without placing the meshes again."""
    index_path = Path(dataset) / "pairs" / "real_photo_index.json"
    res = json.loads(index_path.read_text(encoding="utf-8"))
    for row in res["builds"]:
        if row.get("status") == "ok":
            _compare(dataset, row)
            p = next(x for x in PHOTOS if x["id"] == row["id"])
            (build_dir(dataset, p) / "stats.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
    res["summary"] = summarize(res["builds"])
    index_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


SUMMARY_KEYS = ("iou", "stair45", "normal_roughness", "lum", "chroma", "nbr_de", "detail_geo",
                "detail_spurious", "detail_colour", "cast_dL", "cast_da", "cast_db")


def _pct(values):
    v = np.array([x for x in values if x is not None], dtype=np.float64)
    return [round(float(np.percentile(v, q)), 3) for q in (10, 50, 90)] if len(v) else None


def summarize(rows: list) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]

    def pick(r, k):
        return r["validation"].get("iou") if k == "iou" else r["stats"].get(k)

    out = {"n": len(ok), "n_test": sum(r["in_test_metrics"] for r in ok),
           "real_photo_trellis": {k: _pct([pick(r, k) for r in ok]) for k in SUMMARY_KEYS},
           "synthetic_v1": {k: _pct([(r.get("synthetic_v1") or {}).get(k) for r in ok]) for k in SUMMARY_KEYS}}
    for name in ("smooth", "blocky"):
        have = [r["render_sets"][name] for r in ok if name in r.get("render_sets", {})]
        if have:
            out[f"render_{name}"] = {"n": len(have), **{k: _pct([h.get(k) for h in have]) for k in SUMMARY_KEYS}}
    return out


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["fetch", "cutout", "generate", "place", "refresh"])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dataset = args.dataset.resolve()
    if args.cmd == "fetch":
        print(f"{len(fetch(dataset))} photos present")
    elif args.cmd == "cutout":
        cutout(dataset)
    elif args.cmd == "generate":
        print(f"{generate(dataset, args.only)} generated")
    elif args.cmd == "refresh":
        print(json.dumps(refresh(dataset)["summary"], indent=1))
    else:
        res = place(dataset, args.only)
        print(json.dumps(res.get("summary", {}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
