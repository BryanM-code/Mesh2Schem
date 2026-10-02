#!/usr/bin/env python3
"""Review a run's pairs in the browser and record a verdict for each.

Usage:  python -m synth.review data/synthetic/pilot-v3 [--port 5057] [--compare data/synthetic/pilot-v2]
        then open http://127.0.0.1:5057

    /                 every asset as a card, filterable by status, group, flag and verdict
    /sample/<uid>     the question "is this a good training pair?" in one screen: source
                      mesh, degraded mesh (the input), the Minecraft target with builder
                      roofs and with geometric roofs, back view, input voxels, shapes; the
                      pair's flags; verdict buttons and problem tags
                      keys: 1 good, 2 flawed, 3 garbage, n / p next / previous
    /stats            the run's report (synth.report), if it has been written

`--compare RUN` adds that run's target of the same source beside this one (before / after
a pipeline change). Verdicts are saved to <run>/review.json as
{"verdicts": {uid: {"verdict", "tags", "note", "at"}}}, which `synth.report` summarises by
group and tag. The server only listens on 127.0.0.1 and only serves files that belong to
a sample folder.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import tempfile
import time
from pathlib import Path

PANELS = [("source.png", "source mesh"), ("degraded.png", "degraded mesh (the input, TRELLIS-like)"),
          ("target.png", "Minecraft target, {primary} roofs"), ("target_geometric.png", "Minecraft target, geometric roofs"),
          ("target_builder.png", "Minecraft target, builder roofs"), ("target_back.png", "target, back"),
          ("input.png", "input voxels (what the model sees)"), ("shapes.png", "target shapes"),
          ("roof_regions.png", "roof regions and selected pitches")]
VERDICTS = ("good", "flawed", "garbage")
# Why a pair is not good: each tag is a different fix (source filter, voxeliser, degradation).
TAGS = ("unrealistic source", "not a building", "incomplete / open model", "site or clutter left",
        "scale wrong", "detail lost", "holes / floating blocks", "colours / materials wrong", "roof wrong",
        "input unrecognisable")
FLAGS = ("scale_uncertain", "height_capped", "size_capped", "small_building")
SHAPE_KEY = [("full", "#b9b9be"), ("glass", "#6ecdeb"), ("bottom slab", "#ebc346"), ("top slab", "#e18c2d"),
             ("stairs east", "#dc4b41"), ("stairs west", "#46a55f"), ("stairs north", "#4b73e1"),
             ("stairs south", "#af5fcd"), ("fence", "#a5642d"), ("wall", "#5ac8aa")]

CSS = """
:root { --bg:#16171b; --panel:#202228; --ink:#e6e7ea; --muted:#9a9ea8; --rule:#30333b;
        --good:#4fb477; --flawed:#e3a84e; --garbage:#e0685f; --accent:#7aa7f0; }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
a { color:var(--accent); } header { padding:14px 20px; border-bottom:1px solid var(--rule);
  display:flex; flex-wrap:wrap; gap:8px 20px; align-items:baseline; }
header h1 { font-size:18px; margin:0; } .muted { color:var(--muted); }
form.filters { display:flex; gap:10px; flex-wrap:wrap; padding:10px 20px; border-bottom:1px solid var(--rule); }
select, button, textarea { background:var(--panel); color:var(--ink); border:1px solid var(--rule);
  border-radius:4px; padding:5px 8px; font:inherit; }
button { cursor:pointer; } button:focus-visible, a:focus-visible { outline:2px solid var(--accent); }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(210px, 1fr)); gap:12px; padding:16px 20px; }
.card { background:var(--panel); border:1px solid var(--rule); border-radius:6px; text-decoration:none;
  color:var(--ink); display:flex; flex-direction:column; }
.card img { width:100%; aspect-ratio:1; object-fit:contain; background:#1b1c20; border-radius:6px 6px 0 0; }
.card .meta { padding:6px 8px; font-size:12px; } .badge { display:inline-block; padding:0 6px;
  border-radius:3px; font-size:11px; font-weight:600; border:1px solid currentColor; }
.good { color:var(--good); } .flawed { color:var(--flawed); } .garbage { color:var(--garbage); }
.rejected { color:var(--garbage); } .error { color:var(--garbage); }
.panels { display:grid; grid-template-columns:repeat(auto-fill, minmax(330px, 1fr)); gap:12px; padding:16px 20px; }
.panels figure { margin:0; background:var(--panel); border:1px solid var(--rule); border-radius:6px; }
.panels img { width:100%; aspect-ratio:1; object-fit:contain; display:block; }
.panels figcaption { padding:4px 8px; font-size:12px; color:var(--muted); }
.side { padding:0 20px 30px; display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; }
@media (max-width: 900px) { .side { grid-template-columns:1fr; } }
table { border-collapse:collapse; width:100%; font-size:12px; } td { border-bottom:1px solid var(--rule);
  padding:3px 6px; vertical-align:top; } td:first-child { color:var(--muted); white-space:nowrap; }
pre { white-space:pre-wrap; word-break:break-word; font-size:11px; background:var(--panel); padding:8px;
  border-radius:4px; max-height:420px; overflow:auto; }
.verdict { display:flex; gap:8px; flex-wrap:wrap; align-items:center; padding:12px 20px;
  border-bottom:1px solid var(--rule); position:sticky; top:0; background:var(--bg); z-index:2; }
.verdict button.on { outline:2px solid currentColor; }
.tags { flex-basis:100%; display:flex; flex-wrap:wrap; gap:4px 14px; font-size:12px; color:var(--muted); } .key span { display:inline-block; width:10px;
  height:10px; margin:0 4px 0 10px; border-radius:2px; vertical-align:-1px; }
"""


class Run:
    def __init__(self, root):
        self.root = Path(root).resolve()          # Flask resolves relative paths from its package
        self.samples = self.root / "samples"
        self.review_path = self.root / "review.json"

    def manifests(self):
        out = []
        for p in sorted(self.samples.glob("*/manifest.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        return out

    def reviews(self):
        if self.review_path.exists():
            return json.loads(self.review_path.read_text(encoding="utf-8")).get("verdicts", {})
        return {}

    def save_review(self, uid, verdict, note, tags=()):
        data = {"verdicts": self.reviews()}
        if verdict is None:
            data["verdicts"].pop(uid, None)
        else:
            data["verdicts"][uid] = {"verdict": verdict, "tags": sorted(set(tags)), "note": note or "",
                                     "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        fd, tmp = tempfile.mkstemp(dir=str(self.root), suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(tmp, self.review_path)


def _group(m):
    return (m.get("source") or {}).get("group") or "?"


def _filtered(run, args):
    reviews = run.reviews()
    status = args.get("status", "ok")
    group = args.get("group", "")
    verdict = args.get("verdict", "")
    flag = args.get("flag", "")
    items = []
    for m in run.manifests():
        st = m.get("status")
        if status != "all" and st != status:
            continue
        if group and _group(m) != group:
            continue
        if flag and not (m.get("flags") or {}).get(flag):
            continue
        v = (reviews.get(m["uid"]) or {}).get("verdict", "")
        if verdict == "unreviewed" and v:
            continue
        if verdict and verdict != "unreviewed" and v != verdict:
            continue
        items.append((m, v))
    items.sort(key=lambda mv: (_group(mv[0]), (mv[0].get("source") or {}).get("category") or "", mv[0]["uid"]))
    return items


def create_app(root, compare_root=None):
    from flask import Flask, abort, jsonify, request, send_file

    run = Run(root)
    other = Run(compare_root) if compare_root else None
    app = Flask(__name__)

    def page(title, body):
        return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
                "content='width=device-width,initial-scale=1'><title>{}</title><style>{}</style></head>"
                "<body>{}</body></html>").format(html.escape(title), CSS, body)

    @app.get("/")
    def index():
        items = _filtered(run, request.args)
        allm = run.manifests()
        reviews = run.reviews()
        counts = {s: sum(1 for m in allm if m.get("status") == s) for s in ("ok", "rejected", "error")}
        vc = {v: sum(1 for r in reviews.values() if r.get("verdict") == v) for v in VERDICTS}
        groups = sorted({_group(m) for m in allm})

        def opt(name, value, label, current):
            sel = " selected" if current == value else ""
            return "<option value='{}'{}>{}</option>".format(html.escape(value), sel, html.escape(label))

        a = request.args
        f = ["<form class='filters' method='get'>",
             "<label>status <select name='status' onchange='this.form.submit()'>",
             "".join(opt("status", v, v, a.get("status", "ok")) for v in ("ok", "rejected", "error", "all")),
             "</select></label><label>group <select name='group' onchange='this.form.submit()'>",
             opt("group", "", "all", a.get("group", "")),
             "".join(opt("group", g, g, a.get("group", "")) for g in groups),
             "</select></label><label>flag <select name='flag' onchange='this.form.submit()'>",
             "".join(opt("flag", v, v or "any", a.get("flag", "")) for v in ("",) + FLAGS),
             "</select></label><label>verdict <select name='verdict' onchange='this.form.submit()'>",
             "".join(opt("verdict", v, v or "any", a.get("verdict", "")) for v in ("", "unreviewed") + VERDICTS),
             "</select></label></form>"]
        cards = []
        qs = request.query_string.decode()
        for m, v in items:
            st = m.get("status")
            src = m.get("source") or {}
            img = "target.png" if st == "ok" else "source.png"
            if st == "ok":
                d = (m.get("target") or {}).get("dims") or {}
                line = "{} x {} blocks".format(d.get("footprint_blocks", "?"), d.get("height_blocks", "?"))
            else:
                line = "<span class='badge {}'>{}</span> {}".format(st, html.escape(m.get("reason") or ""),
                                                                   html.escape(m.get("stage") or ""))
            badge = "<span class='badge {0}'>{0}</span>".format(v) if v else ""
            badge += "".join(" <span class='badge muted'>{}</span>".format(html.escape(f.replace("_", " ")))
                             for f in FLAGS if (m.get("flags") or {}).get(f))
            cards.append("<a class='card' href='/sample/{uid}?{qs}'><img loading='lazy' alt='' src='/file/{uid}/{img}'>"
                         "<div class='meta'>{cat} &middot; {grp} {badge}<br>{line}</div></a>".format(
                             uid=m["uid"], qs=html.escape(qs), img=img, cat=html.escape(src.get("category") or "?"),
                             grp=html.escape(_group(m)), badge=badge, line=line))
        head = ("<header><h1>{}</h1><span class='muted'>ok {} &middot; rejected {} &middot; errors {}</span>"
                "<span class='muted'>reviewed: <span class='good'>good {}</span> &middot; <span class='flawed'>flawed {}</span>"
                " &middot; <span class='garbage'>garbage {}</span></span><a href='/stats'>statistics</a>"
                "<span class='muted'>{} shown</span></header>").format(
            html.escape(run.root.name), counts["ok"], counts["rejected"], counts["error"],
            vc["good"], vc["flawed"], vc["garbage"], len(items))
        return page(run.root.name + " review", head + "".join(f) + "<div class='grid'>" + "".join(cards) + "</div>")

    @app.get("/sample/<uid>")
    def sample(uid):
        folder = run.samples / uid
        if not (folder / "manifest.json").exists():
            abort(404)
        m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        items = [x[0]["uid"] for x in _filtered(run, request.args)]
        i = items.index(uid) if uid in items else -1
        qs = request.query_string.decode()
        prev_uid = items[i - 1] if i > 0 else None
        next_uid = items[i + 1] if 0 <= i < len(items) - 1 else None
        v = (run.reviews().get(uid) or {})
        src = m.get("source") or {}
        buttons = "".join("<button class='{0}{1}' data-v='{0}'>{2} {0}</button>".format(
            name, " on" if v.get("verdict") == name else "", k) for k, name in zip("123", VERDICTS))
        on_tags = set(v.get("tags") or [])
        tag_boxes = "<div class='tags'>" + "".join(
            "<label><input type='checkbox' class='tag' value='{0}'{1}> {0}</label>".format(
                html.escape(t), " checked" if t in on_tags else "") for t in TAGS) + "</div>"
        nav = " ".join(
            "<a href='/sample/{}?{}'>{}</a>".format(u, html.escape(qs), label)
            for u, label in ((prev_uid, "&larr; previous (p)"), (next_uid, "next (n) &rarr;")) if u)
        bar = ("<div class='verdict'><a href='/?{qs}'>all</a> {nav} <strong>{name}</strong> "
               "<span class='muted'>{st}</span> {buttons}<button data-v=''>clear</button>"
               "<textarea id='note' rows='1' cols='36' placeholder='note (saved with the verdict)'>{note}</textarea>"
               "<span id='saved' class='muted'></span>{tags}</div>").format(
            qs=html.escape(qs), nav=nav, name=html.escape(src.get("name") or uid), st=html.escape(m.get("status", "")),
            buttons=buttons, note=html.escape(v.get("note", "")), tags=tag_boxes)
        diagnostics = []
        if (run.root / "voxel_debug" / (uid + ".png")).is_file():
            diagnostics.append("<a target='_blank' href='/diagnostic/{}/stages'>voxel stages</a>".format(uid))
        if (run.root / "trellis_compare" / uid / "comparison.png").is_file():
            diagnostics.append("<a target='_blank' href='/diagnostic/{}/trellis'>real TRELLIS comparison</a>".format(uid))
        if diagnostics:
            bar += "<div style='padding:12px 20px'>" + " &nbsp; ".join(diagnostics) + "</div>"
        primary = (m.get("target") or {}).get("roof_style") or "builder"
        figs = []
        for fname, label in PANELS:
            if fname == "target_{}.png".format(primary):
                continue
            if (folder / fname).exists():
                label = label.format(primary=primary)
                figs.append("<figure><img alt='{0}' src='/file/{1}/{2}'><figcaption>{0}</figcaption></figure>".format(
                    html.escape(label), uid, fname))
        if other and (other.samples / uid / "target.png").exists():
            figs.append("<figure><img alt='' src='/other/{0}/target.png'><figcaption>{1}: saved target preview (rendering palette may differ)</figcaption>"
                        "</figure>".format(uid, html.escape(other.root.name)))
        elif other and (other.samples / uid / "manifest.json").exists():
            om = json.loads((other.samples / uid / "manifest.json").read_text(encoding="utf-8"))
            figs.append("<figure><figcaption>{}: {} {} {}</figcaption></figure>".format(
                html.escape(other.root.name), html.escape(om.get("status", "")), html.escape(om.get("reason") or ""),
                html.escape(om.get("detail") or "")))
        key = "<div class='key muted'>shape colours:" + "".join(
            "<span style='background:{}'></span>{}".format(c, html.escape(n)) for n, c in SHAPE_KEY) + "</div>"
        rows = [("status", m.get("status")), ("reason", "{} / {} {}".format(m.get("stage", ""), m.get("reason", ""),
                                                                          m.get("detail", "")) if m.get("status") != "ok" else ""),
                ("name", src.get("name")), ("category", "{} / {} / {}".format(src.get("category"), src.get("subcategory") or "-", src.get("group"))),
                ("caption", src.get("caption")), ("licence", src.get("licence")), ("author", src.get("author")),
                ("source", "<a href='{0}' target='_blank' rel='noopener'>{0}</a>".format(html.escape(src.get("url") or ""))),
                ("attribution", src.get("attribution"))]
        if src.get("review_warnings"):
            rows.append(("source review warnings", "; ".join(src["review_warnings"])))
        sc = m.get("scale") or {}
        if sc:
            rows += [("scale", "{} m from {} ({}), confidence {}".format(
                sc.get("height_m"), sc.get("source"), sc.get("unit", ""), sc.get("confidence", "?")))]
            if sc.get("effective_blocks_per_metre") is not None:
                rows.append(("effective scale", "{} blocks/metre (including jitter and resource caps)".format(
                    sc["effective_blocks_per_metre"])))
        fl = m.get("flags") or {}
        if fl:
            raised = [f.replace("_", " ") for f in FLAGS if fl.get(f)]
            rows += [("flags", ", ".join(raised) if raised else "none")]
        t = m.get("target") or {}
        if t.get("dims"):
            rows += [("building", "{footprint_blocks} x {height_blocks} x {width_blocks} blocks (long x high x wide)".format(**t["dims"]))]
        if m.get("resolution"):
            r = m["resolution"]
            rows += [("grid", " x ".join(str(x) for x in r["grid_xyz"]) + "; tiles " +
                      ", ".join("{}^3: {}".format(k, n) for k, n in r["tiles"].items()))]
        if m.get("metrics"):
            rows += [("input vs target", json.dumps(m["metrics"]))]
        table = "<table>" + "".join("<tr><td>{}</td><td>{}</td></tr>".format(
            html.escape(k), v if k in ("source", "reason") else html.escape(str(v if v is not None else "")))
            for k, v in rows) + "</table>"
        detail = {k: m[k] for k in ("blocks", "target", "checks", "scene", "degrade", "timings_s", "files") if k in m}
        side = ("<div class='side'><div>{}{}</div><div><pre>{}</pre></div></div>").format(
            table, key, html.escape(json.dumps(detail, indent=1)[:60000]))
        script = """<script>
const uid = %s, nextUrl = %s, prevUrl = %s;
async function save(v) {
  const note = document.getElementById('note').value;
  const tags = [...document.querySelectorAll('input.tag:checked')].map(b => b.value);
  const r = await fetch('/api/verdict', {method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify({uid, verdict: v || null, note, tags})});
  document.getElementById('saved').textContent = r.ok ? (v ? 'saved: ' + v : 'cleared') : 'save failed';
  document.querySelectorAll('.verdict button[data-v]').forEach(b => b.classList.toggle('on', b.dataset.v === v && v !== ''));
  if (r.ok && v && nextUrl) setTimeout(() => location.href = nextUrl, 250);
}
document.querySelectorAll('.verdict button[data-v]').forEach(b => b.addEventListener('click', () => save(b.dataset.v)));
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA') return;
  if (e.key === '1') save('good'); if (e.key === '2') save('flawed'); if (e.key === '3') save('garbage');
  if (e.key === 'n' && nextUrl) location.href = nextUrl; if (e.key === 'p' && prevUrl) location.href = prevUrl;
});
</script>""" % (json.dumps(uid), json.dumps("/sample/{}?{}".format(next_uid, qs) if next_uid else None),
                json.dumps("/sample/{}?{}".format(prev_uid, qs) if prev_uid else None))
        return page(src.get("name") or uid, bar + "<div class='panels'>" + "".join(figs) + "</div>" + side + script)

    @app.post("/api/verdict")
    def verdict():
        body = request.get_json(force=True, silent=True) or {}
        uid = str(body.get("uid") or "")
        v = body.get("verdict")
        if not (run.samples / uid / "manifest.json").exists() or v not in VERDICTS + (None,):
            return jsonify({"ok": False}), 400
        tags = [t for t in (body.get("tags") or []) if t in TAGS]
        run.save_review(uid, v, str(body.get("note") or "")[:2000], tags)
        return jsonify({"ok": True})

    @app.get("/file/<uid>/<name>")
    def file(uid, name):
        allowed = {p for p, _ in PANELS} | {"review.png"}
        path = run.samples / uid / name
        if name not in allowed or not path.is_file() or path.parent.parent != run.samples:
            abort(404)
        return send_file(path, max_age=0)

    @app.get("/other/<uid>/<name>")
    def other_file(uid, name):
        if other is None or name != "target.png":
            abort(404)
        path = other.samples / uid / name
        if not path.is_file() or path.parent.parent != other.samples:
            abort(404)
        return send_file(path, max_age=0)

    @app.get("/diagnostic/<uid>/<kind>")
    def diagnostic(uid, kind):
        if not (run.samples / uid / "manifest.json").is_file():
            abort(404)
        if kind == "stages":
            path = run.root / "voxel_debug" / (uid + ".png")
        elif kind == "trellis":
            path = run.root / "trellis_compare" / uid / "comparison.png"
        else:
            abort(404)
        if not path.is_file() or not path.resolve().is_relative_to(run.root):
            abort(404)
        return send_file(path, max_age=0)

    @app.get("/stats")
    def stats():
        md = run.root / "report.md"
        if not md.exists():
            return page("statistics", "<header><h1>No report yet</h1><span class='muted'>run python -m synth.report "
                                      "{}</span></header>".format(html.escape(str(run.root))))
        return page("statistics", "<header><h1>{} statistics</h1><a href='/'>back</a></header><pre style='margin:16px 20px;"
                                  "max-height:none'>{}</pre>".format(html.escape(run.root.name),
                                                                     html.escape(md.read_text(encoding="utf-8"))))

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--port", type=int, default=5057)
    ap.add_argument("--compare", help="another run of the same sources, shown beside each pair")
    args = ap.parse_args(argv)
    app = create_app(args.root, args.compare)
    print("review {} at http://127.0.0.1:{}".format(args.root, args.port))
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
