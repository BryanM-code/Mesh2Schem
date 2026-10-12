#!/usr/bin/env python3
"""Run a selection through the pipeline: download, process, validate, save, delete the raw file.

PARKED: see docs/mesh-to-schem-plan.md v4

Usage:  python -m synth.run --config data/synthetic/pilot-v2/run_config.json
                            [--workers 6] [--limit N] [--only UID ...] [--force]
                            [--raw-cache DIR]

Streaming by design: each raw GLB is downloaded into `cfg.tmp_dir/<run>/`, processed, and
deleted once its pair (or rejection) is recorded -- unless `cfg.keep_raw`. So at any moment
the raw data on disk is at most one GLB per worker, and only the processed pairs accumulate.
`--raw-cache DIR` reads GLBs that are already on disk (e.g. a previous pilot's) instead of
downloading them; those files are never deleted.

Resumable: an asset whose manifest already says ok or rejected is skipped (`--force` redoes
it). Every finished asset appends one line to `ledger.jsonl`: status, stage, reason, bytes,
seconds -- `synth.report` builds the statistics from the manifests and that ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import shutil
import time
import traceback
from pathlib import Path


def seed_for(uid: str, base: int) -> int:
    return int(hashlib.sha256("{}:{}".format(base, uid).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def _work(args):
    rec, cfg_json, raw_cache, force = args
    from .config import PipelineConfig
    from .fetch import FetchError, fetch_glb
    from .filters import metadata_verdict
    from .pair import PIPELINE_VERSION, Rejected, make_pair, source_block

    cfg = PipelineConfig.from_dict(json.loads(cfg_json))
    run = cfg.run_dir
    out = run / "samples" / rec["uid"]
    man_path = out / "manifest.json"
    if man_path.exists() and not force:
        try:
            previous = json.loads(man_path.read_text(encoding="utf-8"))
            status = previous.get("status")
            if status in ("ok", "rejected"):
                if (previous.get("config_hash") != cfg.hash()
                        or previous.get("pipeline_version") != PIPELINE_VERSION):
                    return {"uid": rec["uid"], "status": "stale",
                            "reason": "configuration or pipeline changed; use a new run directory"}
                return {"uid": rec["uid"], "status": "skipped"}
        except Exception:
            pass
    if out.exists():
        shutil.rmtree(out)                           # a stale or forced sample starts clean
    out.mkdir(parents=True, exist_ok=True)
    cached = Path(raw_cache) / (rec["uid"] + ".glb") if raw_cache else None
    tmp = Path(cfg.tmp_dir) / cfg.run_name / (rec["uid"] + ".glb")
    glb, downloaded = (cached, False) if cached and cached.exists() else (tmp, True)
    t0 = time.time()
    line = {"uid": rec["uid"], "bucket": rec.get("bucket"), "category": rec.get("category")}
    try:
        # The current metadata rules, before any download (a selection may predate them).
        reason, detail = metadata_verdict(rec, cfg)
        if reason:
            raise Rejected("metadata", reason, detail)
        if downloaded:
            fetch_glb(rec["path"], glb, sha256=rec.get("sha256") or None)
        line["raw_bytes"] = glb.stat().st_size
        line["downloaded"] = downloaded
        t_dl = time.time() - t0
        man = make_pair(rec, glb, out, cfg, seed_for(rec["uid"], cfg.seed))
        man["timings_s"]["download"] = round(t_dl, 2)
        man["raw"] = {"bytes": line["raw_bytes"], "downloaded": downloaded}
        man_path.write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
        line.update(status="ok", pair_bytes=sum(man["files"].values()))
    except (Rejected, FetchError) as exc:
        if isinstance(exc, FetchError):
            stage, reason, detail, info = "download", "download_failed", str(exc)[:300], {}
        else:
            stage, reason, detail, info = exc.stage, exc.reason, exc.detail, exc.info
        body = {"uid": rec["uid"], "status": "rejected", "stage": stage, "reason": reason, "detail": detail,
                "pipeline_version": PIPELINE_VERSION, "config_hash": cfg.hash(), "source": source_block(rec),
                "raw": {"bytes": line.get("raw_bytes"), "downloaded": downloaded},
                "seconds": round(time.time() - t0, 1)}
        for k in ("scene", "checks", "orient", "scale", "target"):
            if k in info:
                body[k] = info[k]
        man_path.write_text(json.dumps(body, indent=1, default=str), encoding="utf-8")
        line.update(status="rejected", stage=stage, reason=reason)
    except Exception as exc:
        man_path.write_text(json.dumps({"uid": rec["uid"], "status": "error", "reason": type(exc).__name__,
                                        "detail": str(exc)[:300], "traceback": traceback.format_exc()[-3000:],
                                        "source": source_block(rec), "seconds": round(time.time() - t0, 1)},
                                       indent=1, default=str), encoding="utf-8")
        line.update(status="error", reason=type(exc).__name__)
    finally:
        if downloaded and not cfg.keep_raw and glb.exists():
            try:
                glb.unlink()
            except OSError:
                pass
    line["seconds"] = round(time.time() - t0, 1)
    return line


def main(argv=None):
    from .config import PipelineConfig

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true", help="redo assets that already have a result")
    ap.add_argument("--raw-cache", help="folder of <uid>.glb to read instead of downloading (never deleted)")
    args = ap.parse_args(argv)

    cfg = PipelineConfig.load(args.config)
    run = cfg.run_dir
    saved = json.loads(Path(args.config).read_text(encoding="utf-8"))
    from .config import PIPELINE_VERSION
    if saved.get("config_hash") != cfg.hash() or saved.get("pipeline_version") != PIPELINE_VERSION:
        raise SystemExit("Saved configuration predates the current pipeline/defaults. "
                         "Save a new configuration in a new run directory; keep the earlier pilot for comparison.")
    recs = [json.loads(line) for line in open(run / "selection.jsonl", encoding="utf-8")]
    if args.only:
        recs = [r for r in recs if r["uid"] in set(args.only)]
    if args.limit:
        recs = recs[:args.limit]
    tmp = Path(cfg.tmp_dir) / cfg.run_name
    tmp.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(tmp).free / 1e9
    worst = args.workers * cfg.max_glb_mb / 1e3
    print("{} assets, {} workers; temp {} ({:.0f} GB free, at most {:.1f} GB raw at once)".format(
        len(recs), args.workers, tmp, free_gb, worst))
    if free_gb < worst + 1:
        raise SystemExit("not enough free space in the temp folder")
    cfg_json = cfg.to_json()
    jobs = [(r, cfg_json, args.raw_cache, args.force) for r in recs]
    t0 = time.time()
    done = 0
    with open(run / "ledger.jsonl", "a", encoding="utf-8") as ledger, \
            mp.Pool(args.workers, maxtasksperchild=4) as pool:
        for line in pool.imap_unordered(_work, jobs):
            done += 1
            if line["status"] != "skipped":
                line["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                ledger.write(json.dumps(line) + "\n")
                ledger.flush()
            print("{:4d}/{} {:8s} {} {} {}  ({:.0f}s)".format(
                done, len(jobs), line["status"], line["uid"][:12], line.get("stage", ""), line.get("reason", ""),
                time.time() - t0), flush=True)
    left = list(tmp.glob("*.glb"))
    print("done in {:.0f}s; raw GLBs left in temp: {}".format(time.time() - t0, len(left)))


if __name__ == "__main__":
    main()
