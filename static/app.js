/* BuildDataCollector frontend — a tiny vanilla-JS SPA over the JSON API.
   Views are hash-routed: #add, #library, #build/<id>. No build step, no dependencies. */

// ── DOM helper (hyperscript) ──────────────────────────────────────────────────
function h(tag, attrs, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v === true) node.setAttribute(k, "");
    else node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return node;
}
const byId = (id) => document.getElementById(id);

// One page-wide paste handler at a time, cleared whenever a view is mounted so a stale
// one can't fire against a build you've navigated away from.
let pasteHandler = null;
function setPasteHandler(fn) {
  if (pasteHandler) document.removeEventListener("paste", pasteHandler);
  pasteHandler = fn || null;
  if (pasteHandler) document.addEventListener("paste", pasteHandler);
}

function mount(node) {
  setPasteHandler(null);
  byId("view").replaceChildren(node);
  window.scrollTo(0, 0);
}
function fromTemplate(id) { return byId(id).content.cloneNode(true).firstElementChild; }

function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString(undefined,
    { year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}
function fmtDateShort(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

let toastTimer;
function toast(msg, kind = "ok") {
  const t = byId("toast");
  t.textContent = msg;
  t.className = "toast " + (kind === "err" ? "err" : "ok");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 3800);
}

// The server may be a Windows PC while the browser is a Mac, so shortcut hints follow the
// machine you're typing on, not the one running the app.
const IS_MAC = /Mac|iPhone|iPad/i.test(
  (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || "");
const SNIP_HINT = IS_MAC ? "Ctrl+Cmd+Shift+4 then Cmd+V" : "Win+Shift+S then Ctrl+V";

function qs(obj) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(obj)) {
    if (v !== null && v !== undefined && String(v) !== "") p.set(k, v);
  }
  return p.toString();
}

// ── API layer ─────────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON */ }
  if (!res.ok) {
    const err = new Error((data && (data.error || data.message)) || res.statusText);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}
function inspectSchem(file) {
  const fd = new FormData(); fd.append("schem", file);
  return api("/api/inspect", { method: "POST", body: fd });
}
function createBuild({ file, name, note, allowDup }) {
  const fd = new FormData();
  fd.append("schem", file);
  fd.append("name", name);
  fd.append("source_note", note || "");
  fd.append("allow_duplicate", allowDup ? "true" : "false");
  return api("/api/builds", { method: "POST", body: fd });
}
const listBuilds = (params) => api("/api/builds?" + qs(params));
const getBuild = (id) => api("/api/builds/" + id);
function pairBuild(id, file, replace) {
  const fd = new FormData();
  fd.append("glb", file);
  if (replace) fd.append("replace", "true");
  return api("/api/builds/" + id + "/pair", { method: "POST", body: fd });
}
const deleteBuild = (id) => api("/api/builds/" + id + "?confirm=true", { method: "DELETE" });
const unpairBuild = (id) => api("/api/builds/" + id + "/pair?confirm=true", { method: "DELETE" });
const rebuild = () => api("/api/admin/rebuild", { method: "POST" });
function setImage(id, file, replace) {
  const fd = new FormData();
  fd.append("image", file);
  if (replace) fd.append("replace", "true");
  return api("/api/builds/" + id + "/image", { method: "POST", body: fd });
}
const clearImage = (id) => api("/api/builds/" + id + "/image?confirm=true", { method: "DELETE" });
function addShots(id, files) {
  const fd = new FormData();
  // A clipboard image arrives as a blob with no usable name, so give it one — the server
  // filters on extension before it decodes anything.
  files.forEach((f, i) => fd.append("images", f, f.name || "pasted-" + (i + 1) + ".png"));
  return api("/api/builds/" + id + "/shots", { method: "POST", body: fd });
}
const removeShot = (id, slot) =>
  api("/api/builds/" + id + "/shots/" + slot + "?confirm=true", { method: "DELETE" });
const updateBuild = (id, fields) => api("/api/builds/" + id, {
  method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(fields),
});
const scanOrigins = (folder, apply) => api("/api/admin/scan-origins", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ folder, apply }),
});
const defaultDownloads = () => api("/api/admin/default-downloads");
function inspectImport(file) {
  const fd = new FormData();
  fd.append("file", file);
  return api("/api/import/inspect", { method: "POST", body: fd });
}
const asJson = (body) => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});
const importRegion = (token, bbox) => api("/api/import/" + token + "/region", asJson(bbox));
const importSave = (token, payload) => api("/api/import/" + token + "/save", asJson(payload));
const importSaveSchematics = (token) =>
  api("/api/import/" + token + "/save-schematics", asJson({}));
const importDiscard = (token) => api("/api/import/" + token, { method: "DELETE" });
const sourceBounds = (id) => api("/api/builds/" + id + "/source-bounds");
const recropBuild = (id, bbox) => api("/api/builds/" + id + "/recrop", asJson(bbox));

// ── Server freshness ──────────────────────────────────────────────────────────
// This file is served from disk on every request, but the Python behind it is loaded once
// when the server starts. Leave a server running while the code changes and you get a new
// frontend talking to an old API — controls whose data never arrives would just quietly
// not appear. Rather than let that happen silently, check once and say so.
const EXPECTED_API_VERSION = 3;

function checkServerFreshness() {
  api("/api/meta")
    .then((meta) => {
      if (meta.api_version < EXPECTED_API_VERSION) showStaleServerBanner(meta.api_version);
    })
    .catch(() => showStaleServerBanner(null));   // no /api/meta at all: older still
}

function showStaleServerBanner(version) {
  if (byId("stale-banner")) return;
  const banner = h("div", { class: "stale-banner", id: "stale-banner" },
    h("strong", {}, "This page is newer than the server running it. "),
    "Parts of the app will be missing or fail — including dropping reference screenshots. ",
    "Stop the server (Ctrl+C in its terminal) and run ", h("code", {}, "python server.py"),
    " again.",
    h("div", { class: "stale-detail mono" },
      "page expects API v" + EXPECTED_API_VERSION +
      " · server is " + (version === null ? "older than v1 (no /api/meta)" : "v" + version)));
  document.body.insertBefore(banner, document.body.firstChild);
}

// ── Drag & drop wiring ────────────────────────────────────────────────────────
function wireDrop(zone, { accept, multiple = false, onFiles }) {
  let input = zone.querySelector('input[type="file"]');
  if (!input) {
    input = h("input", { type: "file", hidden: true });
    if (accept) input.accept = accept;
    input.multiple = multiple;
    zone.append(input);
  }
  const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
  ["dragenter", "dragover"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { stop(e); zone.classList.add("drag"); }));
  ["dragleave", "dragend"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { stop(e); if (!zone.contains(e.relatedTarget)) zone.classList.remove("drag"); }));
  zone.addEventListener("drop", (e) => {
    stop(e); zone.classList.remove("drag");
    const files = Array.from((e.dataTransfer && e.dataTransfer.files) || []);
    if (files.length) onFiles(files);
  });
  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    const files = Array.from(input.files);
    if (files.length) onFiles(files);
    input.value = "";
  });
}

// ── Schematic files ───────────────────────────────────────────────────────────
// Two formats are accepted: modern Sponge .schem and the legacy MCEdit .schematic that
// older downloads still ship. The server converts the latter, so everything the library
// stores is a .schem. These are only a first-pass filter — the server decides for real,
// by reading the file rather than trusting its name.
const SCHEMATIC_EXT = /\.(schem|schematic)$/i;
const isSchematic = (file) => SCHEMATIC_EXT.test(file.name);
const schematicBaseName = (name) => name.replace(SCHEMATIC_EXT, "");

// A .mcworld is a Bedrock world export — a whole world, not a build — so it takes a
// different path: parse it, show a map, and cut the region you actually want out of it.
const MCWORLD_EXT = /\.mcworld$/i;
const isMcworld = (file) => MCWORLD_EXT.test(file.name);

// ── Small shared bits ─────────────────────────────────────────────────────────
function chip(k, v) { return h("span", { class: "chip" }, h("span", { class: "k" }, k), h("b", {}, v)); }
function dimsChips(info) {
  return h("div", { class: "chips" },
    h("span", { class: "chip" }, h("span", { class: "dims-big" }, info.dims)),
    chip("Width", info.w), chip("Height", info.h), chip("Length", info.l), chip("Longest", info.longest),
    h("span", { class: "chip", title: info.hash },
      h("span", { class: "k" }, "sha256"), h("span", { class: "mono" }, info.hash.slice(0, 10) + "…")));
}
function dupNotice(existing, extra) {
  const ul = h("ul", { class: "existing-list" });
  existing.forEach((b) => ul.append(h("li", {},
    "#" + b.id + " “" + b.name + "” — " + b.dims + ", " + b.status_label + ", added " + fmtDateShort(b.added_at))));
  return h("div", { class: "notice notice-warn" },
    h("strong", {}, "Possible duplicate. "),
    "Identical content already exists (" + existing.length + "). " + (extra || ""), ul);
}
// ── Pipeline ──────────────────────────────────────────────────────────────────
// A build is collected in three steps, because TRELLIS reconstructs from a 2D picture
// rather than from a schematic: capture the build, get an image of it, then the mesh.
// Status always names the first thing still missing.
const STATUS_META = {
  needs_image: { label: "Needs AI image", cls: "badge-needs-image" },
  needs_trellis: { label: "Needs TRELLIS", cls: "badge-needs" },
  complete: { label: "Complete", cls: "badge-paired" },
};
// The four artefacts, in the order they're collected. Screenshots are `optional`: they're
// raw material for generating the AI image, not something the pipeline waits on, so they
// never gate the status — but they still get a slot in the strip and a pip of their own.
const STAGES = [
  { key: "schem", caption: "Schematic", has: () => true },
  { key: "shot-1", caption: "Screenshots", optional: true, has: (b) => b.shot_count > 0 },
  { key: "image", caption: "AI image", has: (b) => b.has_image },
  { key: "mesh", caption: "TRELLIS mesh", has: (b) => b.has_glb },
];

function statusBadge(status) {
  const meta = STATUS_META[status] || { label: status, cls: "badge-needs" };
  return h("span", { class: "badge " + meta.cls }, meta.label);
}

// How far along this build is, readable without parsing the badge text. The optional
// stage is drawn hollow so a build with no screenshots doesn't look unfinished.
function pipelinePips(b) {
  return h("span", { class: "pips" }, STAGES.map((s) => {
    const on = s.has(b);
    const detail = s.optional && on ? s.caption + " (" + b.shot_count + ")" : s.caption;
    return h("span", {
      class: "pip" + (on ? " on" : "") + (s.optional ? " pip-optional" : ""),
      title: detail + (s.optional ? " — optional" : ""),
    });
  }));
}

// ── Thumbnails ────────────────────────────────────────────────────────────────
// Rendered and cached server-side; the endpoint 404s for a build it cannot draw (and for
// a mesh that isn't there yet), so every tile carries its own fallback rather than letting
// a broken-image icon through. `loading="lazy"` keeps a big library from asking for every
// render at once.
function thumbTile(id, kind, emptyLabel, cls) {
  const tile = h("div", { class: cls || "thumb" });
  const img = h("img", {
    src: "/api/builds/" + id + "/thumb/" + kind + ".png",
    alt: "", loading: "lazy", decoding: "async",
  });
  // No label means the tile is optional decoration (the paired-mesh inset): drop it
  // entirely rather than leaving an empty box behind.
  img.addEventListener("error", () => {
    if (emptyLabel) tile.replaceChildren(h("div", { class: "thumb-empty" }, emptyLabel));
    else tile.remove();
  });
  tile.append(img);
  return tile;
}

function vacantTile(label) {
  return h("div", { class: "thumb thumb-vacant" }, h("div", { class: "thumb-empty" }, label));
}

// ── Inline confirm ────────────────────────────────────────────────────────────
// Two-step confirmation that appears next to the thing it is about, instead of a browser
// dialog that doesn't say which build it means. Shared by every destructive action:
// removing a build, and removing a TRELLIS pair from either the Library or the detail page.
function confirmInline(trigger, { question, confirmLabel, busyLabel, onConfirm, host }) {
  const mountPoint = host || trigger.parentNode;
  trigger.addEventListener("click", (event) => {
    event.stopPropagation();          // library cards navigate on click; this must not
    if (mountPoint.querySelector(".confirm-row")) return;
    trigger.disabled = true;

    const yes = h("button", { class: "btn btn-danger btn-sm" }, confirmLabel);
    const no = h("button", { class: "btn btn-ghost btn-sm" }, "Cancel");
    const row = h("div", { class: "confirm-row row-actions" }, h("span", {}, question), yes, no);
    row.addEventListener("click", (e) => e.stopPropagation());
    mountPoint.append(row);

    no.addEventListener("click", () => { row.remove(); trigger.disabled = false; });
    yes.addEventListener("click", () => {
      yes.disabled = no.disabled = true;
      yes.replaceChildren(h("span", { class: "spin" }), " " + (busyLabel || "Working…"));
      Promise.resolve()
        .then(onConfirm)
        .catch((err) => {
          yes.disabled = no.disabled = false;
          yes.replaceChildren(document.createTextNode(confirmLabel));
          toast(err.message || "That didn’t work.", "err");
        });
    });
  });
  return trigger;
}

// Removing a TRELLIS pair — the fix for a .glb dropped on the wrong build. Offered both
// on the Library card (where a mismatched thumbnail is what tips you off) and on the
// detail page.
function unpairButton(b, { label, className, host, onDone }) {
  const btn = h("button", {
    class: className || "btn btn-danger btn-sm",
    title: "Delete this build’s copied TRELLIS mesh and set it back to Needs TRELLIS",
  }, label || "Remove pair");
  return confirmInline(btn, {
    host,
    question: "Remove the TRELLIS mesh from “" + b.name + "”?",
    confirmLabel: "Yes, remove pair",
    busyLabel: "Removing…",
    onConfirm: () => unpairBuild(b.id).then(() => {
      toast("Removed the TRELLIS pair from “" + b.name + "”.", "ok");
      onDone();
    }),
  });
}

// Rejected uploads come back with a stable `code` plus where in the file the problem is
// (`stage`, `path`, `offset`), so a bad download can be diagnosed without reopening it.
function errorNotice(err, fallback) {
  const d = (err && err.data) || {};
  const notice = h("div", { class: "notice notice-error" },
    (err && err.message) || fallback || "Could not read this file.");
  if (d.code) {
    const bits = [d.code, d.stage];
    if (d.path) bits.push(d.path);
    if (d.offset != null) bits.push("byte " + d.offset);
    notice.append(h("div", { class: "err-detail mono" }, bits.join("  ·  ")));
  }
  return notice;
}

// What the server had to do to a legacy file, and anything it could not carry across.
function conversionNotice(info) {
  const warnings = info.warnings || [];
  if (!info.converted && warnings.length === 0) return null;
  const parts = [];
  if (info.converted) {
    parts.push(h("strong", {}, "Converted to .schem. "),
      "Read as " + info.source_format_label + " and rewritten in the modern Sponge " +
      "format — your original file is untouched.");
  }
  if (warnings.length) {
    const ul = h("ul", { class: "existing-list" });
    warnings.forEach((w) => ul.append(h("li", {}, w)));
    parts.push(ul);
  }
  return h("div", { class: "notice " + (warnings.length ? "notice-warn" : "notice-ok") }, parts);
}

// ── View: Add ─────────────────────────────────────────────────────────────────
function viewAdd() {
  const root = fromTemplate("tpl-add");
  mount(root);
  const work = root.querySelector("#add-work");
  wireDrop(root.querySelector('[data-drop="schem"]'), {
    accept: ".schem,.schematic,.mcworld", multiple: true,
    onFiles: (files) => handleAddFiles(files, work),
  });
}

function handleAddFiles(files, work) {
  // One file of any kind opens the region picker, so you can see what you are adding and
  // trim it before it lands. Several at once stays the fast batch list — being walked
  // through a picker twenty times would be worse than the slack it saves.
  if (files.length === 1) {
    renderImport(files[0], work);
    return;
  }
  const schems = files.filter(isSchematic);
  const ignored = files.length - schems.length;
  if (schems.length === 0) {
    toast("Drop one world or zip at a time, or several schematics together.", "err");
    return;
  }
  if (ignored > 0) toast("Ignored " + ignored + " file(s) that are not schematics.", "err");
  renderBatchAdd(schems, work);
}

function renderSingleAdd(file, work) {
  const card = h("div", { class: "card" }, h("div", {}, h("span", { class: "spin" }), " Checking schematic…"));
  work.replaceChildren(card);
  inspectSchem(file)
    .then((info) => renderSingleForm(card, file, schematicBaseName(file.name), info))
    .catch((err) => card.replaceChildren(errorNotice(err)));
}

function renderSingleForm(card, file, defaultName, info) {
  let dup = !!info.duplicate;
  const nameInput = h("input", { class: "input", type: "text", value: defaultName });
  const noteInput = h("textarea", { class: "input", placeholder: "e.g. source URL … / license note" });
  // Prefilled, not forced: converted files carry lossy provenance worth recording, but
  // it is the user's note and they can edit or clear it before saving.
  if (info.converted) noteInput.value = "Converted from " + info.source_format_label + ".";
  const saveBtn = h("button", { class: "btn" + (dup ? " btn-green" : "") }, dup ? "Save anyway as variant" : "Save build");
  const cancelBtn = h("button", { class: "btn btn-ghost" }, "Cancel");
  const actions = h("div", { class: "row-actions" }, saveBtn, cancelBtn);

  card.replaceChildren(dimsChips(info));
  const notice = conversionNotice(info);
  if (notice) card.append(notice);
  if (dup) card.append(dupNotice(info.existing, "Saving keeps it as a new variant."));
  card.append(
    h("div", { class: "field" }, h("label", {}, "Name"), nameInput),
    h("div", { class: "field" }, h("label", {}, "Source / notes (optional)"), noteInput),
    actions);

  cancelBtn.addEventListener("click", () => card.remove());
  saveBtn.addEventListener("click", () => {
    saveBtn.disabled = cancelBtn.disabled = true;
    saveBtn.replaceChildren(h("span", { class: "spin" }), " Saving…");
    createBuild({ file, name: nameInput.value, note: noteInput.value, allowDup: dup })
      .then((res) => {
        const b = res.build;
        card.replaceChildren(h("div", { class: "notice notice-ok" },
          "Saved “" + b.name + "” (#" + b.id + ") — " + b.dims + ".  ",
          h("a", { href: "#build/" + b.id }, "Open"), " · ", h("a", { href: "#library" }, "View library")));
        toast("Saved “" + b.name + "”", "ok");
      })
      .catch((err) => {
        saveBtn.disabled = cancelBtn.disabled = false;
        if (err.status === 409 && err.data && err.data.existing) {
          dup = true;
          saveBtn.textContent = "Save anyway as variant";
          saveBtn.classList.add("btn-green");
          card.insertBefore(dupNotice(err.data.existing, "Saving keeps it as a new variant."), actions);
        } else {
          saveBtn.textContent = "Save build";
          toast(err.message || "Save failed", "err");
        }
      });
  });
}

// ── Add: importing anything droppable ─────────────────────────────────────────
// A schematic, a Bedrock world, a zipped Java world — all three hold blocks and all three
// open the same picker, because the useful question is the same for each: which part of
// this is the build? A zip that holds neither gets described rather than merely refused.
function renderImport(file, work) {
  const card = h("div", { class: "card" },
    h("div", {}, h("span", { class: "spin" }), " Reading “" + file.name + "”… ",
      h("span", { class: "muted" }, "a world takes a moment")));
  work.replaceChildren(card);

  inspectImport(file)
    .then((info) => {
      if (info.kind === "zip_schematics") renderZipSchematics(card, file, info);
      else renderRegionPicker(card, file, info);
    })
    .catch((err) => card.replaceChildren(unimportableNotice(err, file)));
}

// A zip with no build in it is the case that used to be a dead end, so say what it does
// hold — "not supported" is not something anyone can act on.
function unimportableNotice(err, file) {
  const data = (err && err.data) || {};
  const wrap = h("div", {});
  wrap.append(errorNotice(err, "Could not import “" + file.name + "”."));
  if (data.contents && data.contents.length) {
    const list = h("ul", { class: "existing-list" });
    data.contents.forEach((name) => list.append(h("li", {}, name)));
    wrap.append(h("div", { class: "notice notice-warn" },
      h("strong", {}, "What’s actually in it: "), list));
  }
  if (data.worlds && data.worlds.length) {
    wrap.append(h("div", { class: "notice notice-warn" },
      "It holds a Bedrock world file — drop ", h("code", {}, data.worlds[0]),
      " on its own instead."));
  }
  return wrap;
}

// A zip of schematics is several builds rather than one region, so it takes the batch
// path; any of them can be cropped afterwards from its own build page.
function renderZipSchematics(card, file, info) {
  const list = h("ul", { class: "existing-list" });
  info.schematics.forEach((s) =>
    list.append(h("li", {}, s.name + "  ", h("span", { class: "muted" },
      Math.round(s.size / 1024) + " KB"))));
  const addAll = h("button", { class: "btn" },
    "Add all " + info.schematics.length + " schematics");
  const cancel = h("button", { class: "btn btn-ghost" }, "Cancel");

  cancel.addEventListener("click", () => {
    importDiscard(info.token).catch(() => {});
    card.remove();
  });
  addAll.addEventListener("click", () => {
    addAll.disabled = cancel.disabled = true;
    addAll.replaceChildren(h("span", { class: "spin" }), " Adding…");
    importSaveSchematics(info.token)
      .then((res) => {
        const parts = [h("strong", {}, "Added " + res.added.length + ". ")];
        if (res.failed && res.failed.length) {
          parts.push(res.failed.length + " skipped: "
            + res.failed.map((f) => f.name + " (" + f.reason + ")").join(", ") + ". ");
        }
        parts.push(h("a", { href: "#library" }, "View library"));
        card.replaceChildren(h("div", { class: "notice notice-ok" }, parts));
        toast("Added " + res.added.length + " schematic(s).", "ok");
      })
      .catch((err) => {
        addAll.disabled = cancel.disabled = false;
        addAll.replaceChildren(document.createTextNode("Add all"));
        card.append(errorNotice(err, "Could not add those."));
      });
  });

  card.replaceChildren(
    h("div", { class: "chips" }, chip("Zip", file.name),
      chip("Schematics", info.schematics.length)),
    h("p", { class: "panel-sub" },
      "This zip holds schematics rather than a world, so they go in as a batch. "
      + "Crop any of them afterwards from its build page."),
    list,
    h("div", { class: "row-actions" }, addAll, cancel));
}

// The picker itself: a map of the source with a draggable box over it.
function renderRegionPicker(card, file, info) {
  const bounds = info.map_bounds;
  const mapWidth = bounds.max_x - bounds.min_x + 1;
  const mapLength = bounds.max_z - bounds.min_z + 1;
  const box = Object.assign({}, info.auto_bbox);
  delete box.blocks;

  const image = h("img", { class: "world-map", src: info.preview, alt: "Map of the source" });
  const marquee = h("div", { class: "world-marquee" });
  const stage = h("div", { class: "world-stage" }, image, marquee);
  const readout = h("div", { class: "world-readout" });
  const nameInput = h("input", { class: "input", type: "text",
    value: (info.level_name || file.name.replace(/\.[^.]+$/, "")) });
  const noteInput = h("textarea", { class: "input", rows: "2",
    placeholder: "e.g. source URL … / licence note" });
  const saveBtn = h("button", { class: "btn" }, "Save this region as a build");
  const resetBtn = h("button", { class: "btn btn-ghost btn-sm" }, "Reset to detected area");
  const wholeBtn = h("button", { class: "btn btn-ghost btn-sm" }, "Use everything");
  const cancelBtn = h("button", { class: "btn btn-ghost" }, "Cancel");

  const controls = regionControls(box, () => refresh(false));
  wireMarquee(stage, image, bounds, box, (live) => refresh(live));

  function refresh(localOnly) {
    controls.sync(box);
    drawMarquee(marquee, bounds, box);
    showRegionSize(readout, saveBtn, localDims(box, info.suggested.max_volume));
    if (localOnly) return;
    controls.debounce(() => importRegion(info.token, box)
      .then((dims) => showRegionSize(readout, saveBtn, dims))
      .catch(() => { /* keep the local estimate */ }));
  }

  resetBtn.addEventListener("click", () => {
    Object.assign(box, info.auto_bbox);
    delete box.blocks;
    refresh(false);
  });
  wholeBtn.addEventListener("click", () => {
    Object.assign(box, info.bounds);
    refresh(false);
  });
  cancelBtn.addEventListener("click", () => {
    importDiscard(info.token).catch(() => {});
    card.remove();
  });

  saveBtn.addEventListener("click", () => {
    saveBtn.disabled = cancelBtn.disabled = true;
    saveBtn.replaceChildren(h("span", { class: "spin" }), " Converting…");
    importSave(info.token, Object.assign({}, box, {
      name: nameInput.value, source_note: noteInput.value,
    }))
      .then((res) => {
        const b = res.build;
        card.replaceChildren(h("div", { class: "notice notice-ok" },
          "Saved “" + b.name + "” (#" + b.id + ") — " + b.dims + ".  ",
          h("a", { href: "#build/" + b.id }, "Open"), " · ",
          h("a", { href: "#library" }, "View library")));
        toast("Imported “" + b.name + "”", "ok");
      })
      .catch((err) => {
        saveBtn.disabled = cancelBtn.disabled = false;
        saveBtn.replaceChildren(document.createTextNode("Save this region as a build"));
        if (err.status === 409 && err.data && err.data.existing) {
          card.insertBefore(dupNotice(err.data.existing,
            "This exact region is already in the library."), readout);
        } else {
          card.append(errorNotice(err, "Could not import that region."));
        }
      });
  });

  const isWorld = info.kind !== "schematic";
  card.replaceChildren(
    h("div", { class: "chips" },
      chip("Source", info.kind_label),
      chip(isWorld ? "Chunks" : "Size",
        isWorld ? info.chunks.toLocaleString()
          : (info.bounds.max_x - info.bounds.min_x + 1) + "x"
            + (info.bounds.max_y - info.bounds.min_y + 1) + "x"
            + (info.bounds.max_z - info.bounds.min_z + 1)),
      chip("Built blocks", (info.auto_bbox.blocks || 0).toLocaleString())),
    h("p", { class: "panel-sub" },
      isWorld
        ? "Drag on the map to select one build. The blue box is what looked built when the "
          + "file was read — often a whole neighbourhood, so narrow it to a single house."
        : "Drag on the map to trim this schematic. The blue box skips the ground and empty "
          + "air around the build; “Use everything” keeps it exactly as dropped."),
    stage,
    h("div", { class: "world-controls" }, controls.fields, resetBtn, wholeBtn),
    readout,
    h("div", { class: "field" }, h("label", {}, "Name"), nameInput),
    h("div", { class: "field" }, h("label", {}, "Source / notes (optional)"), noteInput),
    h("div", { class: "row-actions" }, saveBtn, cancelBtn));

  refresh(false);
}

// ── Shared region-picking pieces ──────────────────────────────────────────────
// Used by the importer and by re-cropping an existing build, so a box is drawn the same
// way in both places.

function regionControls(box, onChange) {
  const fields = {};
  const nodes = [];
  const number = (key, label) => {
    const input = h("input", { class: "input input-num", type: "number", value: box[key] });
    input.addEventListener("input", () => {
      const value = parseInt(input.value, 10);
      if (!isNaN(value)) { box[key] = value; onChange(); }
    });
    fields[key] = input;
    nodes.push(h("label", { class: "f-axis" }, label, input));
  };
  number("min_x", "X"); number("max_x", "→");
  number("min_z", "Z"); number("max_z", "→");
  number("min_y", "Y"); number("max_y", "→");

  let pending = null;
  return {
    fields: nodes,
    sync(current) {
      Object.keys(fields).forEach((key) => {
        if (document.activeElement !== fields[key]) fields[key].value = current[key];
      });
    },
    debounce(fn) { clearTimeout(pending); pending = setTimeout(fn, 150); },
  };
}

function wireMarquee(stage, image, bounds, box, onChange) {
  const mapWidth = bounds.max_x - bounds.min_x + 1;
  const mapLength = bounds.max_z - bounds.min_z + 1;
  // Screen position -> world coordinate. The map is a plan view, so this is a straight
  // linear map from the image's pixel box onto the area it covers.
  const toWorld = (event) => {
    const rect = image.getBoundingClientRect();
    const fx = Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 0.9999);
    const fz = Math.min(Math.max((event.clientY - rect.top) / rect.height, 0), 0.9999);
    return { x: bounds.min_x + Math.floor(fx * mapWidth),
             z: bounds.min_z + Math.floor(fz * mapLength) };
  };

  let dragStart = null;
  stage.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    dragStart = toWorld(e);
    stage.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  stage.addEventListener("pointermove", (e) => {
    if (!dragStart) return;
    const now = toWorld(e);
    box.min_x = Math.min(dragStart.x, now.x); box.max_x = Math.max(dragStart.x, now.x);
    box.min_z = Math.min(dragStart.z, now.z); box.max_z = Math.max(dragStart.z, now.z);
    onChange(true);
  });
  const endDrag = (e) => {
    if (!dragStart) return;
    dragStart = null;
    try { stage.releasePointerCapture(e.pointerId); } catch (_) { /* already gone */ }
    onChange(false);
  };
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", endDrag);
}

function drawMarquee(marquee, bounds, box) {
  const mapWidth = bounds.max_x - bounds.min_x + 1;
  const mapLength = bounds.max_z - bounds.min_z + 1;
  marquee.style.left = (((box.min_x - bounds.min_x) / mapWidth) * 100) + "%";
  marquee.style.top = (((box.min_z - bounds.min_z) / mapLength) * 100) + "%";
  marquee.style.width = (((box.max_x - box.min_x + 1) / mapWidth) * 100) + "%";
  marquee.style.height = (((box.max_z - box.min_z + 1) / mapLength) * 100) + "%";
}

// While dragging, the size is computed locally so the readout tracks the pointer; on
// release the server confirms it. Same arithmetic either way — the round trip validates.
function localDims(box, maxVolume) {
  const w = Math.abs(box.max_x - box.min_x) + 1;
  const hgt = Math.abs(box.max_y - box.min_y) + 1;
  const l = Math.abs(box.max_z - box.min_z) + 1;
  const volume = w * hgt * l;
  return { dims: w + "x" + hgt + "x" + l, volume: volume,
           max_volume: maxVolume, too_big: volume > maxVolume };
}

function showRegionSize(readout, saveBtn, dims) {
  const over = dims.too_big;
  readout.replaceChildren(
    h("span", { class: "world-dims" + (over ? " over" : "") }, dims.dims),
    h("span", { class: "muted" }, "  " + dims.volume.toLocaleString() + " blocks"),
    over ? h("span", { class: "world-over" },
      "  over the " + dims.max_volume.toLocaleString() + "-block limit — select a smaller area")
      : null);
  if (saveBtn) saveBtn.disabled = over;
}
function renderBatchAdd(files, work) {
  const entries = files.map((file) => ({
    file, name: schematicBaseName(file.name), note: "",
    info: null, skip: false, saved: false, saving: false, error: null, savedId: null, row: null,
  }));
  const saveAll = h("button", { class: "btn" }, "Save all");
  const head = h("div", { class: "batch-head" },
    h("div", {}, h("strong", {}, files.length + " schematics"), h("span", { class: "muted" }, " — review, then save")),
    saveAll);
  const list = h("div", {});
  work.replaceChildren(h("div", { class: "batch" }, head, list));
  entries.forEach((e) => { e.row = h("div", { class: "batch-item" }); fillBatchRow(e); list.append(e.row); });

  entries.forEach((e) => {
    inspectSchem(e.file)
      .then((info) => {
        e.info = info;
        if (info.duplicate) e.skip = true;
        if (info.converted && !e.note) e.note = "Converted from " + info.source_format_label + ".";
        fillBatchRow(e);
      })
      .catch((err) => {
        e.error = err.message || "unreadable";
        e.errorCode = (err.data && err.data.code) || null;
        e.skip = true;
        fillBatchRow(e);
      });
  });

  saveAll.addEventListener("click", async () => {
    saveAll.disabled = true;
    let ok = 0, fail = 0, skipped = 0;
    for (const e of entries) {
      if (e.saved) continue;
      if (e.skip || e.error) { skipped += e.skip ? 1 : 0; continue; }
      e.saving = true; fillBatchRow(e);
      try {
        const res = await createBuild({ file: e.file, name: e.name, note: e.note, allowDup: e.info && e.info.duplicate });
        e.saved = true; e.savedId = res.build.id; ok++;
      } catch (err) { e.error = err.message || "failed"; fail++; }
      e.saving = false; fillBatchRow(e);
    }
    saveAll.disabled = false;
    toast("Saved " + ok + (fail ? ", " + fail + " failed" : "") + (skipped ? ", " + skipped + " skipped" : "") + ".", fail ? "err" : "ok");
    if (ok > 0 && !head.querySelector(".to-lib"))
      head.append(h("a", { href: "#library", class: "btn btn-ghost btn-sm to-lib" }, "View library"));
  });
}

function fillBatchRow(e) {
  const row = e.row;
  row.classList.toggle("skip", !!e.skip);
  const fileCol = h("div", { class: "bi-file" }, h("div", { class: "bi-name", title: e.file.name }, e.file.name));
  if (e.error) fileCol.append(h("span", { class: "badge badge-dup", title: e.error },
    e.errorCode || "error"));
  else if (e.saved) fileCol.append(h("span", { class: "badge badge-paired" }, "saved #" + e.savedId));
  else if (e.saving) fileCol.append(h("span", { class: "spin" }));
  else if (e.info && e.info.duplicate) fileCol.append(h("span", { class: "badge badge-dup" }, "duplicate"));
  if (e.info && e.info.converted) {
    const warnings = e.info.warnings || [];
    fileCol.append(h("span", {
      class: "badge badge-needs",
      title: "Converted from " + e.info.source_format_label +
        (warnings.length ? "\n\n" + warnings.join("\n\n") : ""),
    }, warnings.length ? "converted ⚠" : "converted"));
  }

  const nameInput = h("input", { class: "input", type: "text", value: e.name });
  nameInput.disabled = e.saved;
  nameInput.addEventListener("input", () => { e.name = nameInput.value; });
  const noteInput = h("input", { class: "input", type: "text", placeholder: "source / notes", value: e.note });
  noteInput.disabled = e.saved;
  noteInput.addEventListener("input", () => { e.note = noteInput.value; });

  const dims = e.error ? h("span", { class: "muted", title: e.error }, "—")
    : e.info ? h("span", { class: "bi-dims" }, e.info.dims)
      : h("span", { class: "spin" });

  const skipCb = h("input", { type: "checkbox" });
  skipCb.checked = !!e.skip; skipCb.disabled = e.saved || !!e.error;
  skipCb.addEventListener("change", () => { e.skip = skipCb.checked; row.classList.toggle("skip", e.skip); });

  row.replaceChildren(fileCol, nameInput, noteInput, h("div", { class: "bi-dims" }, dims),
    h("label", { class: "batch-skip" }, skipCb, "skip"));
}

// ── View: Library ─────────────────────────────────────────────────────────────
// Filters and sort are deliberately session-only, but the grid/table choice is a standing
// preference, so it's the one piece of libState that outlives a reload.
const VIEW_KEY = "bdc.libraryView";
const libState = {
  q: "", status: "", max_longest: "", min_longest: "", sort: "added_at", dir: "desc",
  view: localStorage.getItem(VIEW_KEY) === "table" ? "table" : "grid",
};
// Last fetched page, so switching layout re-renders instead of re-fetching.
let libBuilds = [];

function viewLibrary() {
  const root = fromTemplate("tpl-library");
  mount(root);

  const q = root.querySelector("#f-q");
  const status = root.querySelector("#f-status");
  const maxl = root.querySelector("#f-maxlong");
  const minl = root.querySelector("#f-minlong");
  q.value = libState.q; status.value = libState.status;
  maxl.value = libState.max_longest; minl.value = libState.min_longest;

  const reload = () => loadLibrary(root);
  let deb;
  const debounced = (fn) => { clearTimeout(deb); deb = setTimeout(fn, 220); };
  q.addEventListener("input", () => { libState.q = q.value.trim(); debounced(reload); });
  status.addEventListener("change", () => { libState.status = status.value; reload(); });
  maxl.addEventListener("input", () => { libState.max_longest = maxl.value; debounced(reload); });
  minl.addEventListener("input", () => { libState.min_longest = minl.value; debounced(reload); });
  root.querySelector("#f-clear").addEventListener("click", () => {
    libState.q = libState.status = libState.max_longest = libState.min_longest = "";
    q.value = status.value = maxl.value = minl.value = "";
    reload();
  });

  root.querySelectorAll("th.sortable").forEach((th) => th.addEventListener("click", () => {
    const col = th.dataset.sort;
    if (libState.sort === col) libState.dir = libState.dir === "asc" ? "desc" : "asc";
    else { libState.sort = col; libState.dir = col === "name" ? "asc" : "desc"; }
    reload();
  }));

  root.querySelectorAll(".vt-btn").forEach((btn) => btn.addEventListener("click", () => {
    libState.view = btn.dataset.view;
    localStorage.setItem(VIEW_KEY, libState.view);
    applyView(root);
    renderLibrary(root, libBuilds);
  }));
  applyView(root);

  root.querySelector("#btn-scan").addEventListener("click", () => {
    const panel = root.querySelector("#scan-panel");
    panel.hidden = !panel.hidden;
    if (!panel.hidden && !panel.dataset.ready) originScanPanel(panel, () => loadLibrary(root));
  });

  root.querySelector("#btn-rebuild").addEventListener("click", async () => {
    if (!confirm("Rebuild the database rows from the data/builds/ folders on disk?\n\nThis never deletes anything — it re-derives rows from the files present.")) return;
    try {
      const s = await rebuild();
      toast("Rebuild: " + s.added + " added, " + s.updated + " updated" +
        (s.skipped && s.skipped.length ? ", " + s.skipped.length + " skipped" : "") + ".", "ok");
      reload();
    } catch (err) { toast(err.message || "Rebuild failed", "err"); }
  });

  loadLibrary(root);
}

// ── Recovering download origins ───────────────────────────────────────────────
// A .schem carries no provenance, but Windows tags downloaded files with the URL they
// came from. That tag lives on the original download, not on the library's copy, so the
// only way back to it is to scan the folder you downloaded into and match by content.
function originScanPanel(panel, onApplied) {
  panel.dataset.ready = "1";
  const folder = h("input", { class: "input", type: "text", placeholder: "folder to scan" });
  const preview = h("button", { class: "btn btn-sm" }, "Scan");
  const results = h("div", { class: "scan-results" });

  panel.replaceChildren(
    h("p", { class: "panel-sub" },
      "Windows records the URL a file was downloaded from. This reads that tag off your " +
      "original downloads and matches them to builds by content hash, so renaming a build " +
      "doesn’t break the match. It only fills in blanks — a source URL you’ve already set " +
      "is never overwritten."),
    h("div", { class: "scan-row" }, folder, preview),
    results);

  defaultDownloads().then((d) => { if (!folder.value) folder.value = d.folder; }).catch(() => {});

  const run = (apply) => {
    const btn = apply ? results.querySelector(".btn-apply") : preview;
    const label = btn.textContent;
    btn.disabled = true;
    btn.replaceChildren(h("span", { class: "spin" }), apply ? " Applying…" : " Scanning…");
    return scanOrigins(folder.value.trim(), apply)
      .then((rep) => {
        btn.disabled = false; btn.replaceChildren(document.createTextNode(label));
        showScan(rep, apply);
        if (apply && rep.applied > 0) { toast("Recorded " + rep.applied + " source URL(s).", "ok"); onApplied(); }
      })
      .catch((err) => {
        btn.disabled = false; btn.replaceChildren(document.createTextNode(label));
        results.replaceChildren(errorNotice(err, "Scan failed."));
      });
  };

  function showScan(rep, applied) {
    const found = rep.matched.filter((m) => !m.skipped);
    const summary = h("div", { class: "panel-sub" },
      rep.scanned + " file(s) scanned · " + rep.matched.length + " matched a build · " +
      found.length + " with a URL to record" +
      (rep.skipped ? " · " + rep.skipped + " already had one" : ""));
    results.replaceChildren(summary);

    if (rep.errors && rep.errors.length) {
      results.append(h("div", { class: "notice notice-error" }, rep.errors.join(" ")));
      return;
    }
    if (rep.matched.length === 0) {
      results.append(h("div", { class: "notice notice-warn" },
        "Nothing matched. The originals may have been moved, or downloaded by something " +
        "that doesn’t set the tag."));
      return;
    }

    const list = h("ul", { class: "existing-list" });
    rep.matched.forEach((m) => {
      const li = h("li", {}, "#" + m.build.id + " ", h("b", {}, m.build.name), " — ",
        h("a", { href: m.url, target: "_blank", rel: "noreferrer noopener" }, m.url));
      if (m.skipped) li.append(h("span", { class: "muted" }, " (kept your existing URL)"));
      else if (m.is_site_root) li.append(h("span", { class: "muted" }, " (site only, not the build’s page)"));
      list.append(li);
    });
    results.append(list);

    if (!applied && found.length) {
      const apply = h("button", { class: "btn btn-sm btn-apply" }, "Record these " + found.length + " URL(s)");
      apply.addEventListener("click", () => run(true));
      results.append(h("div", { class: "row-actions" }, apply));
    }
  }

  preview.addEventListener("click", () => run(false));
}

// Show whichever container the current layout uses, and light up its toggle button.
function applyView(root) {
  const grid = libState.view === "grid";
  root.querySelector("#lib-cards").hidden = !grid;
  root.querySelector(".table-wrap").hidden = grid;
  root.querySelectorAll(".vt-btn").forEach((btn) =>
    btn.classList.toggle("active", btn.dataset.view === libState.view));
}

async function loadLibrary(root) {
  const params = {
    q: libState.q, status: libState.status,
    max_longest: libState.max_longest, min_longest: libState.min_longest,
    sort: libState.sort, dir: libState.dir,
  };
  root.querySelector("#btn-export").href = "/api/export.csv?" + qs(params);
  root.querySelectorAll("th.sortable").forEach((th) => {
    const old = th.querySelector(".arrow"); if (old) old.remove();
    if (th.dataset.sort === libState.sort) th.append(h("span", { class: "arrow" }, libState.dir === "asc" ? "▲" : "▼"));
  });

  try {
    const data = await listBuilds(params);
    libBuilds = data.builds;
    renderLibrary(root, libBuilds);
  } catch (err) {
    libBuilds = [];
    root.querySelector("#lib-rows").replaceChildren();
    root.querySelector("#lib-cards").replaceChildren();
    const empty = root.querySelector("#lib-empty");
    empty.hidden = false;
    empty.textContent = err.message || "Failed to load.";
  }
}

// Draws the fetched page into the active layout. Both layouts read the same list, so
// filters, sort and the summary line behave identically in either.
function renderLibrary(root, builds) {
  const tbody = root.querySelector("#lib-rows");
  const cards = root.querySelector("#lib-cards");
  const empty = root.querySelector("#lib-empty");
  const reload = () => loadLibrary(root);

  tbody.replaceChildren();
  cards.replaceChildren();

  if (builds.length === 0) {
    empty.hidden = false;
    const filtered = libState.q || libState.status || libState.max_longest || libState.min_longest;
    empty.replaceChildren(filtered
      ? document.createTextNode("No builds match these filters.")
      : h("span", {}, "No builds yet. ", h("a", { href: "#add" }, "Add your first build"), "."));
  } else {
    empty.hidden = true;
    if (libState.view === "grid") builds.forEach((b) => cards.append(libCard(b, reload)));
    else builds.forEach((b) => tbody.append(libRow(b)));
  }

  // One count per stage, so the header says what's left to do rather than just a total.
  const at = (s) => builds.filter((b) => b.status === s).length;
  root.querySelector("#lib-summary").textContent =
    builds.length + " build" + (builds.length !== 1 ? "s" : "") +
    " · " + at("needs_image") + " need AI image" +
    " · " + at("needs_trellis") + " need TRELLIS" +
    " · " + at("complete") + " complete";
}

// A card is the schematic render, with the paired mesh tucked into the corner. When those
// two pictures disagree, the wrong .glb got dropped — and the fix is right there on the card.
function libCard(b, reload) {
  // The schematic is the hero; whatever else exists rides along in the corner. Seeing
  // the render, the reference image and the mesh together is what makes a mismatch —
  // wrong image, wrong .glb — obvious without opening anything.
  const art = h("div", { class: "card-art" }, thumbTile(b.id, "schem", "No preview"));
  const insets = h("div", { class: "thumb-insets" });
  if (b.has_image) insets.append(thumbTile(b.id, "image", null, "thumb-inset"));
  if (b.has_glb) insets.append(thumbTile(b.id, "mesh", null, "thumb-inset"));
  if (insets.children.length) art.append(insets);

  const card = h("div", { class: "lib-card" },
    art,
    h("div", { class: "card-meta" },
      h("div", { class: "card-name", title: b.name }, b.name,
        b.variant_no > 0 ? h("span", { class: "cell-variant" }, "v" + b.variant_no) : null),
      h("div", { class: "card-sub" },
        h("span", { class: "cell-dims" }, b.dims), statusBadge(b.status), pipelinePips(b))));

  const actions = h("div", { class: "card-actions" });
  if (b.has_glb) {
    actions.append(unpairButton(b, {
      label: "Unpair", className: "btn btn-danger btn-sm", host: card, onDone: reload }));
  }
  if (actions.children.length) card.append(actions);

  card.addEventListener("click", () => { location.hash = "#build/" + b.id; });
  return card;
}

function libRow(b) {
  const tr = h("tr", {},
    h("td", {}, h("span", { class: "cell-name" }, b.name),
      b.variant_no > 0 ? h("span", { class: "cell-variant" }, "v" + b.variant_no) : null),
    h("td", { class: "num" }, h("span", { class: "cell-dims" }, b.dims),
      h("span", { class: "cell-long", title: "longest axis" }, " ↔" + b.longest)),
    h("td", {}, statusBadge(b.status), pipelinePips(b)),
    h("td", {}, sourceCell(b)),
    h("td", {}, h("span", { class: "cell-date" }, fmtDateShort(b.added_at))));
  tr.addEventListener("click", () => { location.hash = "#build/" + b.id; });
  return tr;
}

// The origin link if we have one, falling back to the free-text note.
function sourceCell(b) {
  if (!b.source_url) return h("span", { class: "cell-note", title: b.source_note }, b.source_note || "—");
  const link = h("a", { href: b.source_url, target: "_blank", rel: "noreferrer noopener",
    class: "cell-note", title: b.source_url }, hostOf(b.source_url));
  link.addEventListener("click", (e) => e.stopPropagation());   // don't open the build too
  return link;
}

function hostOf(url) {
  try { return new URL(url).host.replace(/^www\./, ""); } catch (_) { return url; }
}

// ── View: Detail / Pair ───────────────────────────────────────────────────────
async function viewDetail(id) {
  const root = fromTemplate("tpl-detail");
  mount(root);
  const body = root.querySelector("#detail-body");
  try {
    const data = await getBuild(id);
    renderDetail(body, data.build);
  } catch (err) {
    body.replaceChildren(err.status === 404
      ? h("div", { class: "notice notice-error" }, "Build not found.")
      : errorNotice(err, "Failed to load."));
  }
}

function renderDetail(body, b) {
  const grid = h("div", { class: "detail-grid" });
  const add = (k, v, cls) => grid.append(h("div", {}, h("div", { class: "k" }, k), h("div", { class: "v" + (cls ? " " + cls : "") }, v)));
  add("Dimensions (W×H×L)", b.dims + "  ·  longest " + b.longest);
  add("Status", b.status_label);
  add("Added", fmtDate(b.added_at));
  add("AI image added", b.image_at ? fmtDate(b.image_at) : "—");
  add("Paired at", b.paired_at ? fmtDate(b.paired_at) : "—");
  add("Content hash", b.schem_hash, "path");
  add("Schem file", b.schem_path, "path");
  add("Screenshots", b.shot_count
    ? b.shots.map((s) => s.path).join("  ·  ") : "— (none)", "path");
  add("AI image", b.image_path || "— (not generated yet)", "path");
  add("TRELLIS mesh", b.glb_path || "— (not paired yet)", "path");

  body.replaceChildren(
    h("div", { class: "detail-head" }, h("h1", {}, b.name), statusBadge(b.status),
      b.variant_no > 0 ? h("span", { class: "cell-variant" }, "variant " + b.variant_no) : null),
    detailThumbs(b),
    originPanel(b),
    grid,
    h("div", { class: "divider" }));

  body.append(recropSection(b), shotsSection(b), h("div", { class: "divider" }),
    imageSection(b), h("div", { class: "divider" }),
    meshSection(b), dangerZone(b));
}

// The three artefacts in pipeline order, same camera for the two rendered ones — the
// comparison that makes a wrong image or a wrong .glb obvious.
function detailThumbs(b) {
  const figure = (stage) => {
    const present = stage.has(b);
    // The screenshots slot stands for up to three files, so it shows the first and says
    // how many there are rather than pretending to be a single artefact.
    const tile = present
      ? thumbTile(b.id, stage.optional ? "shot-" + b.shots[0].slot : stage.key, "No preview")
      : vacantTile(stage.optional ? "None yet" : "Not yet");
    if (present && stage.optional && b.shot_count > 1) {
      tile.append(h("span", { class: "tile-count" }, "+" + (b.shot_count - 1)));
    }
    const caption = stage.optional && present
      ? stage.caption + " (" + b.shot_count + "/" + b.max_shots + ")"
      : stage.caption;
    // Tiles whose stored file is itself a viewable image open it full size; the schematic
    // and the mesh are downloads, so those stay as plain tiles with links in their sections.
    const viewable = present && (stage.optional || stage.key === "image");
    const body = viewable
      ? h("a", {
          href: "/api/builds/" + b.id + "/file/" +
            (stage.optional ? "shot-" + b.shots[0].slot : stage.key),
          target: "_blank", rel: "noreferrer noopener", title: "Open at full size",
        }, tile)
      : tile;
    return h("figure", {}, body,
      h("figcaption", { class: present ? "" : "muted" }, caption));
  };
  return h("div", { class: "thumb-strip" }, STAGES.map(figure));
}

// ── Detail: where this build came from ────────────────────────────────────────
function originPanel(b) {
  const panel = h("div", { class: "origin-panel" });
  const head = h("div", { class: "origin-head" }, h("span", { class: "k" }, "Source"));
  panel.append(head);

  if (b.source_url) {
    head.append(h("a", { href: b.source_url, target: "_blank", rel: "noreferrer noopener" },
      b.source_url));
    // A bare origin is all Windows recorded when the site handed the download off to a
    // CDN. Say so, and offer the search that actually finds the build's own page.
    if (isSiteRoot(b.source_url)) {
      panel.append(h("div", { class: "origin-hint" },
        "This is the site, not the build’s page — the download was served from a CDN, so " +
        "only the site was recorded. ",
        h("a", {
          href: "https://duckduckgo.com/?q=" + encodeURIComponent("site:" + hostOf(b.source_url) + " " + b.name),
          target: "_blank", rel: "noreferrer noopener",
        }, "Search " + hostOf(b.source_url) + " for “" + b.name + "”")));
    }
  } else {
    head.append(h("span", { class: "muted" }, "not recorded"));
  }
  if (b.source_note) panel.append(h("div", { class: "origin-note" }, b.source_note));
  panel.append(editDetails(b));
  return panel;
}

function isSiteRoot(url) {
  try { const p = new URL(url).pathname; return p === "" || p === "/"; } catch (_) { return false; }
}

function editDetails(b) {
  const name = h("input", { class: "input", type: "text", value: b.name });
  const url = h("input", { class: "input", type: "url", value: b.source_url || "",
    placeholder: "https://… the page you downloaded it from" });
  const note = h("textarea", { class: "input", rows: "2", placeholder: "notes, licence, author…" });
  note.value = b.source_note || "";
  const save = h("button", { class: "btn btn-sm" }, "Save");

  const form = h("div", { class: "edit-form" },
    h("div", { class: "field" }, h("label", {}, "Name"), name),
    h("div", { class: "field" }, h("label", {}, "Source URL"), url),
    h("div", { class: "field" }, h("label", {}, "Notes"), note),
    h("div", { class: "row-actions" }, save));

  save.addEventListener("click", () => {
    save.disabled = true;
    save.replaceChildren(h("span", { class: "spin" }), " Saving…");
    updateBuild(b.id, { name: name.value, source_url: url.value, source_note: note.value })
      .then(() => { toast("Saved.", "ok"); viewDetail(b.id); })
      .catch((err) => {
        save.disabled = false; save.textContent = "Save";
        toast(err.message || "Save failed", "err");
      });
  });

  return h("details", { class: "edit-details" },
    h("summary", { class: "muted" }, "Edit name, source URL and notes"), form);
}

// ── Detail: re-cropping ───────────────────────────────────────────────────────
// The uncropped input is kept beside every imported build, so a box drawn too tight can be
// redrawn — widened, not only narrowed. Builds added before that was kept have no source,
// and say so rather than offering a control that can't work.
function recropSection(b) {
  const wrap = h("div", {});
  wrap.append(h("div", { class: "section-title" }, "Crop"));

  if (!b.can_recrop) {
    wrap.append(h("p", { class: "panel-sub muted" },
      "This build was added before the uncropped original was kept, so there is nothing "
      + "to re-cut from. Re-import the file to be able to crop it."));
    return wrap;
  }

  const details = h("details", { class: "recrop" },
    h("summary", { class: "muted" }, "Re-crop this build"));
  const body = h("div", { class: "recrop-body" },
    h("div", {}, h("span", { class: "spin" }), " Loading the original…"));
  details.append(body);
  wrap.append(details);

  let loaded = false;
  details.addEventListener("toggle", () => {
    if (loaded || !details.hasAttribute("open")) return;
    loaded = true;
    sourceBounds(b.id)
      .then((info) => buildRecropUI(body, b, info))
      .catch((err) => body.replaceChildren(errorNotice(err, "Could not load the original.")));
  });
  return wrap;
}

function buildRecropUI(body, b, info) {
  const bounds = info.map_bounds;
  const box = Object.assign({}, info.crop);
  const image = h("img", { class: "world-map", src: "/api/builds/" + b.id + "/source-preview.png",
    alt: "The uncropped build" });
  const marquee = h("div", { class: "world-marquee" });
  const stage = h("div", { class: "world-stage" }, image, marquee);
  const readout = h("div", { class: "world-readout" });
  const applyBtn = h("button", { class: "btn" }, "Apply new crop");
  const wholeBtn = h("button", { class: "btn btn-ghost btn-sm" }, "Use everything");

  const controls = regionControls(box, () => refresh());
  wireMarquee(stage, image, bounds, box, () => refresh());

  function refresh() {
    controls.sync(box);
    drawMarquee(marquee, bounds, box);
    showRegionSize(readout, applyBtn, localDims(box, info.current.max_volume));
  }

  wholeBtn.addEventListener("click", () => { Object.assign(box, bounds); refresh(); });
  applyBtn.addEventListener("click", () => {
    applyBtn.disabled = true;
    applyBtn.replaceChildren(h("span", { class: "spin" }), " Re-cutting…");
    recropBuild(b.id, box)
      .then((res) => {
        toast("Re-cropped to " + res.build.dims + ".", "ok");
        viewDetail(b.id);
      })
      .catch((err) => {
        applyBtn.disabled = false;
        applyBtn.replaceChildren(document.createTextNode("Apply new crop"));
        body.append(errorNotice(err, "Could not re-crop."));
      });
  });

  body.replaceChildren(
    h("p", { class: "panel-sub" },
      "This is the whole of what was imported. Drag a new box — you can widen it as well "
      + "as narrow it. Screenshots, the AI image and the mesh all stay attached."),
    stage,
    h("div", { class: "world-controls" }, controls.fields, wholeBtn),
    readout,
    h("div", { class: "row-actions" }, applyBtn));
  refresh();
}

// ── Detail: step 2 (optional), reference screenshots ──────────────────────────
// Quick captures of the build, kept so a photo-real image can be generated from a chosen
// angle later. Optional by design: grab them when it's convenient, decide later.
function shotsSection(b) {
  const wrap = h("div", {});
  // A build from a server that predates screenshots has none of these fields. Say that
  // outright — the alternative is a section that looks full and offers nothing, which is
  // exactly the kind of silent nothing that's impossible to diagnose from the UI.
  if (b.shot_slots_free === undefined || b.max_shots === undefined) {
    return h("div", {},
      h("div", { class: "section-title" }, "Reference screenshots"),
      h("div", { class: "notice notice-error" },
        h("strong", {}, "The server is running older code than this page. "),
        "It doesn’t know about reference screenshots yet, so there’s nothing to drop into. ",
        "Restart it (Ctrl+C, then ", h("code", {}, "python server.py"), ") and reload."));
  }

  wrap.append(
    h("div", { class: "section-title" },
      "Reference screenshots ",
      h("span", { class: "muted" }, "optional · " + b.shot_count + " of " + b.max_shots)),
    h("p", { class: "panel-sub" },
      "Raw captures of this build — a few angles to choose between when you generate the " +
      "photo-real image. They don’t affect the build’s status; they’re just material to " +
      "come back to."),
    // The two ways to get one, both a click away from where you'd want them.
    h("p", { class: "panel-sub" },
      "To capture one: ",
      h("a", { href: "/api/builds/" + b.id + "/schem" }, "download the .schem"),
      " and load it in Minecraft with WorldEdit",
      b.source_url
        ? h("span", {}, ", or screenshot the ",
            h("a", { href: b.source_url, target: "_blank", rel: "noreferrer noopener" },
              "source page"), ".")
        : "."));

  if (b.shot_count > 0) {
    const gallery = h("div", { class: "shot-gallery" });
    b.shots.forEach((shot) => {
      const full = "/api/builds/" + b.id + "/file/shot-" + shot.slot;
      // The tile links to the *full-resolution* file, not the thumbnail — the whole point
      // of keeping these is to open one later and feed it to an image model.
      const link = h("a", {
        href: full, target: "_blank", rel: "noreferrer noopener",
        title: "Open screenshot " + shot.slot + " at full size",
      }, thumbTile(b.id, "shot-" + shot.slot, "No preview"));
      const cell = h("figure", { class: "shot-cell" }, link);

      const remove = h("button", {
        class: "btn btn-danger btn-sm shot-remove", title: "Remove this screenshot" }, "✕");
      confirmInline(remove, {
        host: cell,
        question: "Remove screenshot " + shot.slot + "?",
        confirmLabel: "Remove",
        busyLabel: "Removing…",
        onConfirm: () => removeShot(b.id, shot.slot).then(() => {
          toast("Screenshot removed.", "ok");
          viewDetail(b.id);
        }),
      });
      cell.append(remove, h("figcaption", {},
        h("a", { href: full, target: "_blank", rel: "noreferrer noopener" },
          "Slot " + shot.slot),
        " · ",
        h("a", { href: full + "?download=1" }, "save")));
      gallery.append(cell);
    });
    wrap.append(gallery);
  }

  if (b.shot_slots_free > 0) {
    wrap.append(shotDropzone(b));
  } else {
    wrap.append(h("p", { class: "panel-sub muted" },
      "All " + b.max_shots + " slots are used. Remove one to add another."));
  }
  return wrap;
}

function shotDropzone(b) {
  const free = b.shot_slots_free;
  const zone = h("div", { class: "dropzone compact" },
    h("div", { class: "dropzone-inner" },
      h("div", { class: "dropzone-icon" }, "📷"),
      h("div", { class: "dropzone-text" }, h("strong", {},
        "Drop or paste screenshot" + (free > 1 ? "s" : "") + " here")),
      h("div", { class: "dropzone-hint" },
        SNIP_HINT + " — or drop a file, or click to browse")),
    h("div", { class: "dropzone-slots" },
      free + " of " + b.max_shots + " slot" + (b.max_shots > 1 ? "s" : "") + " free"));
  const statusLine = h("div", { style: "margin-top:10px" });

  const send = (files) => {
    statusLine.replaceChildren(h("span", { class: "spin" }),
      " Adding " + files.length + " image(s)…");
    return addShots(b.id, files)
      .then((res) => {
        // Partial success is normal here: more files than free slots, or one bad image
        // among several. Say what landed and what didn't, then reload either way.
        if (res.added.length) toast("Added " + res.added.length + " screenshot(s).", "ok");
        if (res.rejected && res.rejected.length) {
          const list = h("ul", { class: "existing-list" });
          res.rejected.forEach((r) => list.append(h("li", {}, r.filename + " — " + r.reason)));
          statusLine.replaceChildren(
            h("div", { class: "notice notice-warn" },
              res.rejected.length + " file(s) weren’t added.", list));
          setTimeout(() => viewDetail(b.id), 2500);
        } else {
          viewDetail(b.id);
        }
      })
      .catch((err) => {
        statusLine.replaceChildren(errorNotice(err, "Could not add those screenshots."));
      });
  };

  wireDrop(zone, { accept: "image/*", multiple: true, onFiles: send });

  // Paste is the fastest path by far: snip with Win+Shift+S and the image is already on
  // the clipboard, so there's no file to save, name, or find again.
  setPasteHandler((event) => {
    const items = Array.from((event.clipboardData && event.clipboardData.items) || []);
    const files = items
      .filter((i) => i.kind === "file" && String(i.type).startsWith("image/"))
      .map((i) => i.getAsFile())
      .filter(Boolean);
    if (!files.length) return;
    event.preventDefault();
    zone.classList.add("drag");
    setTimeout(() => zone.classList.remove("drag"), 400);
    send(files);
  });

  return h("div", {}, zone, statusLine);
}

// ── Detail: step 3, the AI-generated image ────────────────────────────────────
function imageSection(b) {
  const wrap = h("div", {});
  wrap.append(
    h("div", { class: "section-title" }, "AI-generated image"),
    h("p", { class: "panel-sub" },
      "The photo-real picture TRELLIS reconstructs from, generated from one of the " +
      "screenshots above. This is the artefact TRELLIS consumes, so it’s what the build " +
      "waits on — and it’s stored alongside the mesh it produced."));

  if (!b.has_image) {
    wrap.append(h("p", { class: "panel-sub" },
      b.shot_count > 0
        ? "No generated image yet. Pick an angle from the screenshots above, generate a " +
          "photo-real version of it, and drop the result here."
        : "No generated image yet — and no screenshots to work from. Capture some first, " +
          "or grab one from the source page above."));
    wrap.append(imageDropzone(b, false));
  } else {
    const replace = h("details", {},
      h("summary", { class: "muted", style: "cursor:pointer;margin:8px 0" }, "Replace the generated image"),
      imageDropzone(b, true));
    const remove = h("div", { class: "unpair-row" },
      h("p", { class: "panel-sub" },
        "Removing it deletes this build’s copied ", h("code", {}, "generated.png"),
        " and steps it back to Needs AI image. Your screenshots and your original file " +
        "both stay where they are."));
    const btn = h("button", { class: "btn btn-danger" }, "Remove generated image");
    confirmInline(btn, {
      host: remove,
      question: "Remove the generated image from “" + b.name + "”?",
      confirmLabel: "Yes, remove image",
      busyLabel: "Removing…",
      onConfirm: () => clearImage(b.id).then(() => {
        toast("Removed the generated image.", "ok");
        viewDetail(b.id);
      }),
    });
    remove.append(btn);
    const full = "/api/builds/" + b.id + "/file/image";
    wrap.append(
      h("div", { class: "notice notice-ok" }, h("strong", {}, "Image attached ✓ "),
        "Ready for TRELLIS.  ",
        h("a", { href: full, target: "_blank", rel: "noreferrer noopener" }, "Open full size"),
        " · ", h("a", { href: full + "?download=1" }, "save")),
      replace, remove);
  }
  return wrap;
}

function imageDropzone(b, replace) {
  const zone = h("div", { class: "dropzone compact" },
    h("div", { class: "dropzone-inner" },
      h("div", { class: "dropzone-icon" }, "🖼"),
      h("div", { class: "dropzone-text" }, h("strong", {}, "Drop the generated image here")),
      h("div", { class: "dropzone-hint" }, "png, jpg or webp — or click to browse")));
  const statusLine = h("div", { style: "margin-top:10px" });
  wireDrop(zone, {
    accept: "image/*", multiple: false,
    onFiles: (files) => {
      const img = files[0];
      if (!img) return;
      statusLine.replaceChildren(h("span", { class: "spin" }), " Checking “" + img.name + "”…");
      setImage(b.id, img, replace)
        .then((res) => {
          toast("Reference image saved (" + res.width + "×" + res.height + ").", "ok");
          viewDetail(b.id);
        })
        .catch((err) => {
          statusLine.replaceChildren(h("div", {
            class: err.status === 409 ? "notice notice-warn" : "notice notice-error",
          }, (err.data && err.data.message) || err.message || "Could not save that image."));
        });
    },
  });
  return h("div", {}, zone, statusLine);
}

// ── Detail: step 3, the TRELLIS mesh ──────────────────────────────────────────
function meshSection(b) {
  const wrap = h("div", {});
  if (!b.has_glb) {
    wrap.append(
      h("div", { class: "section-title" }, "Pair TRELLIS output (.glb)"),
      h("p", { class: "panel-sub" }, "Drop the TRELLIS mesh reconstruction for this build. It’s validated as a loadable mesh before saving."),
      pairDropzone(b, false));
    return wrap;
  }
  const details = h("details", {},
    h("summary", { class: "muted", style: "cursor:pointer;margin:8px 0" }, "Replace the paired mesh"),
    h("p", { class: "panel-sub" }, "Dropping a new .glb here overwrites the current output.glb for this build."),
    pairDropzone(b, true));
  // Sits with the pairing controls, not in the Danger zone: unpairing costs you a copied
  // mesh you can drop again, while removing the build costs you the build.
  const unpair = h("div", { class: "unpair-row" },
    h("p", { class: "panel-sub" },
      "Wrong mesh? Removing the pair deletes this build’s copied ",
      h("code", {}, "output.glb"), " and sets it back. The build, and the ",
      "file you originally dropped, both stay put."));
  unpair.append(unpairButton(b, {
    label: "Remove TRELLIS pair", className: "btn btn-danger",
    host: unpair, onDone: () => viewDetail(b.id),
  }));
  wrap.append(
    h("div", { class: "notice notice-ok" }, h("strong", {}, "Mesh attached ✓ "), "This build has a TRELLIS mesh."),
    details, unpair);
  return wrap;
}

function pairDropzone(b, replace) {
  const zone = h("div", { class: "dropzone compact" },
    h("div", { class: "dropzone-inner" },
      h("div", { class: "dropzone-icon" }, "⧉"),
      h("div", { class: "dropzone-text" }, h("strong", {}, "Drop the .glb here")),
      h("div", { class: "dropzone-hint" }, "or click to browse")));
  const statusLine = h("div", { style: "margin-top:10px" });
  wireDrop(zone, {
    accept: ".glb", multiple: false,
    onFiles: (files) => {
      const glb = files.find((f) => f.name.toLowerCase().endsWith(".glb"));
      if (!glb) { toast("Please drop a .glb file.", "err"); return; }
      if (replace && !confirm("Replace the existing paired mesh for this build?")) return;
      statusLine.replaceChildren(h("span", { class: "spin" }), " Validating & pairing “" + glb.name + "”…");
      pairBuild(b.id, glb, replace)
        .then((res) => {
          toast("Paired ✓ (" + (res.faces ? res.faces.toLocaleString() + " faces" : "valid mesh") + ")", "ok");
          viewDetail(b.id);
        })
        .catch((err) => {
          statusLine.replaceChildren(h("div", { class: err.status === 409 ? "notice notice-warn" : "notice notice-error" },
            (err.data && err.data.message) || err.message || "Pairing failed."));
        });
    },
  });
  return h("div", {}, zone, statusLine);
}

function dangerZone(b) {
  const btn = h("button", { class: "btn btn-danger" }, "Remove build");
  const dz = h("div", { class: "danger-zone" },
    h("div", { class: "dz-title" }, "Danger zone"),
    h("div", { class: "dz-sub" }, "Removing deletes this entry and the copied files under data/builds/. Your original downloaded files are never touched."),
    btn);
  confirmInline(btn, {
    host: dz,
    question: "Remove “" + b.name + "” (#" + b.id + ")?",
    confirmLabel: "Yes, remove",
    busyLabel: "Removing…",
    onConfirm: () => deleteBuild(b.id).then(() => {
      toast("Removed “" + b.name + "”.", "ok");
      location.hash = "#library";
    }),
  });
  return dz;
}

// ── Router ────────────────────────────────────────────────────────────────────
function setActiveNav(name) {
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.nav === name));
}
function router() {
  const hash = (location.hash || "").replace(/^#/, "") || "library";
  const [route, arg] = hash.split("/");
  if (route === "add") { setActiveNav("add"); viewAdd(); }
  else if (route === "build" && arg) { setActiveNav("library"); viewDetail(parseInt(arg, 10)); }
  else { setActiveNav("library"); viewLibrary(); }
}
window.addEventListener("hashchange", router);
window.addEventListener("DOMContentLoaded", () => { router(); checkServerFreshness(); });
