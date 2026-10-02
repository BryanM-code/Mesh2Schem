"""Reject unusable assets as early and as cheaply as possible, and always say why.

Three stages, cheapest first. Every rejection is a stable reason code (see `REASONS`) that
ends up in the run ledger and the asset's manifest as `{"status": "rejected", "stage": ...,
"reason": ...}`, so a dataset can be debugged by reason instead of by guesswork.

1. **Metadata** (`metadata_verdict`), before any download: licence, the text rules of the
   Objaverse index (`objaverse_index.stage`: objects, interiors, scenes, toys, ruins, game
   rips, stylised, animated...), photogrammetry, size limits.
2. **Classifier hook** (`CLASSIFIERS`), also before download. EMPTY in V1 -- no learned
   building/non-building classifier is run. A classifier (for example a zero-shot image model
   on the Sketchfab thumbnail) plugs in here as a callable `(record) -> (keep, score)`;
   whatever it rejects is recorded as `classifier_rejected` with its score. TODO.
3. **Geometry** (`geometry_verdict`), after download but before voxelisation, on the loaded
   mesh only: already-blocky models (Minecraft exports, voxel art) and scan-like meshes that
   the text did not admit to.

Scale and size (`scale_broken`, `degenerate_size`, `too_large`) and the target gates
(`scattered`, `open_model`) run later, in `scale` and `target`, because they need the scale
or the voxels; their reason codes are listed here too so a report can explain every code it
meets. Size is not a quality judgement: a small building, an uncertain scale and a very tall
building are kept and flagged (synth.pair.flags), never rejected. Codes of earlier pipeline
versions stay listed (`LEGACY_REASONS`) so old ledgers still explain themselves.
"""
from __future__ import annotations

import re

import numpy as np

REASONS = {
    # metadata
    "licence_not_permitted": "licence is not CC0 or CC BY 4.0",
    "no_download_path": "no GLB path in the Objaverse mirror",
    "object_head": "the caption's subject is an object, not a building",
    "blocky_source": "already made of blocks (Minecraft, voxel art, LEGO)",
    "toy_or_miniature": "toy, miniature, dollhouse, souvenir",
    "interior": "an interior, not a building",
    "furnished_interior": "mostly furniture and rooms",
    "cutaway": "cutaway, section or floor plan",
    "part": "a building part (door, window, roof...) on its own",
    "multi_building": "several buildings",
    "scene": "a city, street, village or diorama",
    "ruin_or_wreck": "ruin, wreck, damaged",
    "not_finished": "under construction, wireframe, massing or clay model",
    "game_rip": "rip of a commercial game asset (licence not the uploader's to give)",
    "stylized": "cartoon, fantasy, sci-fi or otherwise stylised",
    "animated": "animated asset",
    "too_few_faces": "too few faces to be a building",
    "low_quality": "Objaverse++ quality below 2",
    "photogrammetry": "photogrammetry scan (excluded from V1)",
    "object_or_set": "tagged as an object, prop, interior, room or diorama (Sketchfab tags)",
    "not_architecture": "filed under Sketchfab's furniture-home category and not under architecture",
    "too_many_faces": "more faces than the pipeline voxelises",
    "file_too_large": "GLB larger than the size limit",
    # classifier hook
    "classifier_rejected": "rejected by the building/non-building classifier",
    # download
    "download_failed": "download or hash check failed",
    # geometry
    "load_failed": "the GLB could not be read",
    "blocky_geometry": "vertices sit on a block lattice: already a voxel model",
    "scan_geometry": "dense mesh without planar faces: a scan",
    # scale
    "scale_broken": "the file's size is not a size in any unit (zero, infinite, microscopic, astronomical)",
    "too_large": "site or footprint over the engineering bound (one pair cannot hold it)",
    # target
    "empty_target": "nothing left after voxelisation",
    "degenerate_size": "under 3 blocks wide or tall: nothing of a building is left to represent",
    "scattered": "the largest part keeps under 60% of the model: several objects, not one building",
    "open_model": "encloses almost nothing and has no roof over most of its footprint (floor plan, set, cutaway, roofless shell)",
    "too_many_voxels": "shell larger than the per-pair bound",
    "degrade_failed": "the input degradation failed",
    "schem_roundtrip": "the written .schem does not read back identically",
    "validation_failed": "the pair failed a validation check",
}

# Codes earlier pipeline versions used (pilot v1 / v2), for reading their ledgers.
LEGACY_REASONS = {
    "too_small": "(until synth-2.0.0) under the minimum footprint / height at 1:1 scale",
    "too_tall": "(until synth-2.0.0) taller than the height bound at 1:1 scale",
    "fragmented": "(until synth-2.0.0) the model falls apart into pieces",
    "thin_shell": "(until synth-2.0.0) interior share under 0.15: paper-thin or open model",
    "flat_site": "(until synth-2.0.0) grid much wider than tall: a flat site or plaza",
}

# Index stages that are rejections (everything but "pool").
_INDEX_STAGE_REASON = {k: k for k in REASONS}

_SCAN_TEXT = re.compile(
    r"photogrammetr|\bscan\b|\bscans\b|\bscanned\b|3d[- ]?scan|reality ?capture|realitycapture|"
    r"\bdrone\b|lidar|agisoft|metashape|photoscan|photo scan|polycam|\bluma\b|scaniverse|"
    r"kiri engine|3df ?zephyr|pix4d|matterport|point cloud|captured with|made from photos",
    re.I)

# Hooks: callables (record) -> (keep: bool, score: float). None registered in V1.
CLASSIFIERS: list = []


def is_photogrammetry(rec) -> bool:
    """Objaverse++ style 'scanned', the index's scan flag, or scan vocabulary in the text."""
    if rec.get("opp_style") == "scanned" or (rec.get("flags") or {}).get("scan"):
        return True
    text = " ".join([rec.get("name") or "", " ".join(rec.get("tags") or []), rec.get("caption") or ""])
    return bool(_SCAN_TEXT.search(text))


# Tags are weak evidence: a complete building may also tag its rooms or furniture.
# Only a matching object noun in the title makes these a hard object rejection.
OBJECT_TAGS = frozenset({"interior", "furniture", "chair", "table", "prop", "props", "diorama", "room", "vehicle",
                         "character", "characters", "toy", "miniature", "lamp", "decor", "decoration"})


def source_warnings(rec):
    """Source curation hints, separate from whether a usable building is present."""
    warnings = []
    hit = OBJECT_TAGS & {str(t).lower() for t in (rec.get("tags") or [])}
    if hit:
        warnings.append("object/set tags: " + ", ".join(sorted(hit)))
    cats = rec.get("sketchfab_cats") or []
    if "furniture-home" in cats and "architecture" not in cats:
        warnings.append("filed under furniture-home without architecture")
    return warnings


def metadata_verdict(rec, cfg):
    """(reason or None, detail) for an index record, before download."""
    lic = rec.get("license")
    if lic not in cfg.licences:
        return "licence_not_permitted", str(lic)
    stage = rec.get("stage", "pool")
    if stage != "pool":
        return _INDEX_STAGE_REASON.get(stage, stage), "objaverse_index rule"
    if cfg.exclude_photogrammetry and is_photogrammetry(rec):
        return "photogrammetry", rec.get("opp_style") or "scan vocabulary"
    hit = OBJECT_TAGS & {str(t).lower() for t in (rec.get("tags") or [])}
    title = {w.lower() for w in re.findall(r"\w+", rec.get("name") or "")}
    # A house WITH furniture is architecture. A chair bearing a house tag is still a chair.
    from .taxonomy import classify
    title_text = rec.get("name") or ""
    title_category, _, title_flags, _ = classify(title_text, title_text, [])
    if hit & title and title_category is None and title_flags.get("object_head"):
        return "object_or_set", ", ".join(sorted(hit & title))
    faces = rec.get("faces") or 0
    if faces < cfg.min_faces:
        return "too_few_faces", str(faces)
    if faces > cfg.max_faces:
        return "too_many_faces", str(faces)
    if (rec.get("glb_bytes") or 0) > cfg.max_glb_mb * 1e6:
        return "file_too_large", "{:.0f} MB".format((rec.get("glb_bytes") or 0) / 1e6)
    if not rec.get("path"):
        return "no_download_path", ""
    for clf in CLASSIFIERS:
        keep, score = clf(rec)
        if not keep:
            return "classifier_rejected", "{} {:.3f}".format(getattr(clf, "__name__", "classifier"), score)
    return None, ""


def planarity(verts, faces, bin_deg: float = 3.0, top: int = 24):
    """Share of surface area whose face normal falls in the `top` most common directions.

    Architectural models are built from large flat faces sharing a handful of normals --
    four walls, a few roof planes -- so a couple of dozen 3-degree direction bins hold most
    of their area. A photogrammetry mesh is one continuous noisy surface whose normals spread
    over the whole sphere. Returns {"planar_share", "faces"}.
    """
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    area = np.linalg.norm(n, axis=1)
    ok = area > 0
    if not ok.any():
        return {"planar_share": 0.0, "faces": int(len(f))}
    n = n[ok] / area[ok, None]
    area = area[ok]
    az = np.degrees(np.arctan2(n[:, 2], n[:, 0])) + 180.0
    el = np.degrees(np.arcsin(np.clip(n[:, 1], -1.0, 1.0))) + 90.0
    n_az, n_el = int(360 / bin_deg), int(180 / bin_deg) + 1
    key = np.minimum((az / bin_deg).astype(np.int64), n_az - 1) * n_el + (el / bin_deg).astype(np.int64)
    sums = np.bincount(key, weights=area)
    share = float(np.sort(sums)[::-1][:top].sum() / area.sum())
    return {"planar_share": round(share, 4), "faces": int(len(f))}


def geometry_verdict(scene, cfg):
    """(reason or None, detail, metrics) from the loaded mesh alone."""
    from .blockiness import blockiness

    metrics = {"blockiness": blockiness(scene.verts, scene.faces),
               "planarity": planarity(scene.verts, scene.faces)}
    if metrics["blockiness"]["lattice16"] >= cfg.blocky_lattice:
        return "blocky_geometry", "lattice16 {:.2f}".format(metrics["blockiness"]["lattice16"]), metrics
    pl = metrics["planarity"]
    if cfg.exclude_photogrammetry and pl["faces"] >= cfg.scan_min_faces and pl["planar_share"] < cfg.scan_planar_share:
        return "scan_geometry", "planar share {:.2f} over {:,} faces".format(pl["planar_share"], pl["faces"]), metrics
    return None, "", metrics
