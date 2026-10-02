#!/usr/bin/env python3
"""Two runs of the same sources, side by side: what a pipeline change did to each pair.

Usage:  python -m synth.compare_runs <run A> <run B> [--out DIR] [--per-sheet 8] [--panel target]

For every source in both runs it pairs A's and B's result and writes, under
<B>/compare_<A name>/:

    changes.json        per source: status / reason in A and in B, and B's flags
    summary.json        counts: ok in both, rescued (rejected in A, ok in B), lost (ok in A,
                        rejected in B), rejected in both -- with reasons
    rescued_NN.png      sheets for sources B keeps and A rejected:  source | A's reason | B target
    lost_NN.png         sheets for sources A kept and B rejects:    source | A target | B's reason
    both_NN.png         sheets for sources both keep:               source | A target | B target

`--panel degraded` (or input, target_geometric, ...) compares that review panel instead of
the target: `degraded` shows what each run's degradation made of the same building.

The sheets are the before / after evidence for a change of filtering, voxelisation or
target rules; `synth.review --compare A` shows the same pairing one source at a time.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def _manifests(run: Path):
    out = {}
    for p in sorted((run / "samples").glob("*/manifest.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out[m["uid"]] = m
    return out


def _image(path: Path):
    from PIL import Image
    return Image.open(path) if path.exists() else None


def _reason_tile(m, size):
    from . import preview
    lines = [(m.get("source") or {}).get("name") or m["uid"], "", "rejected at {}".format(m.get("stage")),
             m.get("reason") or "", ""] + [str(m.get("detail") or "")[i:i + 50] for i in range(0, 150, 50)]
    return preview.info_tile([l for l in lines if l is not None], size)


def compare_runs(run_a: Path, run_b: Path, out: Path | None = None, per_sheet: int = 8, panel: str = "target"):
    from . import preview

    run_a, run_b = Path(run_a), Path(run_b)
    out = Path(out) if out else run_b / "compare_{}{}".format(run_a.name, "" if panel == "target" else "_" + panel)
    out.mkdir(parents=True, exist_ok=True)
    ma, mb = _manifests(run_a), _manifests(run_b)
    uids = sorted(set(ma) & set(mb))
    rows, groups = [], collections.defaultdict(list)
    for uid in uids:
        a, b = ma[uid], mb[uid]
        sa, sb = a.get("status"), b.get("status")
        kind = ("both" if sa == sb == "ok" else "rescued" if sb == "ok" else "lost" if sa == "ok" else "neither")
        rows.append({"uid": uid, "name": (b.get("source") or {}).get("name"), "kind": kind,
                     "a": {"status": sa, "reason": a.get("reason")}, "b": {"status": sb, "reason": b.get("reason")},
                     "b_flags": b.get("flags")})
        groups[kind].append(uid)
    summary = {"sources": len(uids), **{k: len(v) for k, v in groups.items()},
               "rescued_from": dict(collections.Counter(ma[u].get("reason") for u in groups["rescued"])),
               "lost_to": dict(collections.Counter(mb[u].get("reason") for u in groups["lost"])),
               "neither": dict(collections.Counter((ma[u].get("reason"), mb[u].get("reason")) for u in groups["neither"]))}
    summary["neither"] = {"{} -> {}".format(*k): n for k, n in summary["neither"].items()}
    (out / "changes.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    size = preview.SIZE
    for kind in ("rescued", "lost", "both"):
        tiles = []
        for uid in groups[kind]:
            a, b = ma[uid], mb[uid]
            fa, fb = run_a / "samples" / uid, run_b / "samples" / uid
            name = ((b.get("source") or {}).get("name") or uid)[:40]
            src = _image(fb / "source.png") or _image(fa / "source.png")
            left = _image(fa / (panel + ".png")) if a.get("status") == "ok" else _reason_tile(a, size)
            right = _image(fb / (panel + ".png")) if b.get("status") == "ok" else _reason_tile(b, size)
            flags = [k.replace("_", " ") for k, v in (b.get("flags") or {}).items() if v is True]
            tiles += [(src, name), (left, "{}: {}".format(run_a.name, a.get("reason") or panel)),
                      (right, "{}: {}{}".format(run_b.name, b.get("reason") or panel,
                                                " ({})".format(", ".join(flags)) if flags else ""))]
        for i in range(0, len(tiles), 3 * per_sheet):
            preview.sheet(tiles[i:i + 3 * per_sheet], cols=3).save(out / "{}_{:02d}.png".format(kind, i // (3 * per_sheet) + 1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--out")
    ap.add_argument("--per-sheet", type=int, default=8)
    ap.add_argument("--panel", default="target", help="review panel to compare (target, degraded, input, ...)")
    args = ap.parse_args(argv)
    print(json.dumps(compare_runs(Path(args.run_a), Path(args.run_b), args.out, args.per_sheet, args.panel), indent=1))


if __name__ == "__main__":
    main()
