"""Every knob of the synthetic-pair pipeline, in one place, saved with every run.

A run directory holds `run_config.json` (this dataclass, serialised) and every sample's
manifest carries `config_hash`, so any pair can be traced to the exact settings that made it
and two runs can be compared knob by knob. Change a default here, not deep in a module.

Defaults (V1 decisions of 2026-09-25, revised 2026-09-27 after the pilot-v2 review):

* licences      CC0 and CC BY 4.0 only: clearly permissive, publishable, forkable.
* sources       architecture candidates, with realism assessed separately during review.
                Text metadata is not a reliable realism label (synth.source_quality).
* filtering     removes genuinely unusable assets only. Size and scale are NOT reasons to
                reject a building: a small building is kept and flagged `small_building`, an
                uncertain scale is kept and flagged `scale_uncertain`, a very tall one is
                scaled to fit and flagged `height_capped`. Only a broken scale (not a size in
                any unit) or a target too small to hold a building is rejected.
* scale         1:1 with the player: one block is one metre. Proportions survive.
* target        clutter (vegetation, vehicles, figures) dropped by name; the site (lawn,
                pool, paths, low walls) removed around the building; blocks coloured by what
                is seen from outside; a builder palette of real building materials; builder-
                style roofs (Approach B) as the primary target with the geometric conversion
                (Approach A) stored beside it.
* degradation   conservative: the input must still clearly be the same building.
* resolution    grids at their natural size; 64^3 windows the baseline, tiling the rest.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PIPELINE_VERSION = "synth-2.2.0"


@dataclass
class PipelineConfig:
    run_name: str = "pilot-v3"
    seed: int = 20260927

    # ── licensing and metadata filters (before any download) ─────────────────
    licences: tuple = ("cc0", "by")
    exclude_photogrammetry: bool = True
    # No extra "realism" filter on metadata: on the 195 hand-graded pilot-v1 sources, the
    # Objaverse++ style / quality labels, the TRELLIS-500K aesthetic score and low-poly or game
    # vocabulary did not separate junk from clean targets (low-poly words even marked cleaner
    # ones), so such a filter would drop more good buildings than bad. The index's text rules
    # (stylised, interior, cutaway, game rips, ...) and the target gates stay. See
    # docs/dataset-pipeline.md, "Source quality".
    min_faces: int = 500                     # Objaverse metadata face count
    max_faces: int = 1_500_000
    max_glb_mb: float = 60.0

    # ── cheap geometry checks (after download, before voxelisation) ──────────
    max_load_faces: int = 2_500_000
    blocky_lattice: float = 0.5              # synth.blockiness: already made of blocks
    # synth.filters.planarity below this on a dense mesh = scan-like. Calibrated on the 200
    # pilot-v1 models: flags 9 (3 labelled scans, 6 unlabelled), which the v1 hand audit
    # graded 7 junk, 1 flawed, 1 clean.
    scan_planar_share: float = 0.2
    scan_min_faces: int = 40_000

    # ── scale: 1 block = 1 metre ──────────────────────────────────────────────
    blocks_per_metre: float = 1.0
    scale_jitter: float = 0.05               # +-5%, drawn per sample, recorded
    max_height_blocks: int = 300             # taller buildings are scaled to fit (height_capped)
    small_building_blocks: int = 16          # flag (not reject) when footprint and height are both below
    min_representable_blocks: int = 3       # below this in footprint or height a target holds nothing
    max_footprint_blocks: int = 512          # engineering bound on one pair
    max_site_blocks: int = 1200              # engineering bound on the raw site before voxelising
    # Engineering bound on the model's scaled bounding box: a larger model is scaled down to
    # fit (flagged size_capped) instead of taking an hour to voxelise (pilot v3 met a 190 m
    # "stadium" whose 640 x 190 x 500 box stalled the run).
    max_grid_voxels: int = 8_000_000

    # ── target ────────────────────────────────────────────────────────────────
    margin: int = 3                          # supervised air around the building (sides, top)
    fine_scale: int = 4                      # sub-voxel grid for slab / stair shapes
    drop_clutter: bool = True                # vegetation / vehicles / figures, by part names
    # Binary closing bridges 1-voxel cracks -- right for noisy TRELLIS meshes (the input side
    # keeps it), wrong for clean models: it filled the gaps between columns and the air of
    # porches and loggias, and broke the posts (2026-09-27 audit).
    close_gaps: bool = False
    site_height_m: float = 2.2               # columns lower than this outside the footprint are site
    site_margin: int = 2                     # blocks around the footprint kept (eaves, steps)
    color_mode: str = "exposed"              # mcvoxel.surface_color: colour as seen from outside
    palette: str = "builder"                 # "builder" (building materials) or "full"
    quantize: dict = field(default_factory=lambda: {
        "k_init": 32, "merge_de": 8.0, "absorb_de": 10.0, "region_divisor": 5000,
        "absorb_always_size": 0,  # a one-block window/trim is not automatically noise
    })
    roof_style: str = "builder"              # primary target: "builder" (B) or "geometric" (A)
    store_geometric: bool = True             # keep Approach A beside the primary target
    shadow_repaint: bool = False             # mcvoxel.assign: off for albedo sources
    max_shell_voxels: int = 1_500_000
    # Gates reject what is not one building (synth.target.quality). Both open_* must fail:
    # a pavilion encloses nothing but has a roof; a floor plan has neither.
    gates: dict = field(default_factory=lambda: {
        "scattered_kept": 0.6,               # largest part keeps < 60%: a scene, not a building
        "open_interior_rel": 0.2,            # interior share / a closed box's, below this ...
        "open_roof_cover": 0.6,              # ... and roofed footprint share below this
    })

    # ── degradation (input side) ──────────────────────────────────────────────
    # Conservative since 2026-09-27 (pilot v2 inputs were often barely recognisable); tuned
    # against real TRELLIS output with synth.trellis_compare.
    degrade: dict = field(default_factory=lambda: {
        "bottleneck": [128, 160, 192],
        "blur_sigma": [0.5, 0.9],
        "surface_noise": [0.0, 0.15],
        "noise_wavelength": [8.0, 16.0],
        "hole_fraction": [0.0, 0.01],
        "hole_patch": [5, 25],
        "floaters": [0, 8],
        # Paired v3 comparison (110 real TRELLIS meshes): synthetic L* was 11.4 lower.
        # Avoid multiplying two strong darkness factors onto already shaded source textures.
        "tone_gain": [0.9, 1.05],
        "ambient": [0.85, 0.98],
        "hue_deg_sd": 3.0,
        "chroma": [0.85, 1.1],
        "color_noise": [2.0, 5.0],
        "back_blend": [0.1, 0.4],
        "back_darken": [0.85, 1.0],
        "back_tint_prob": 0.1,
        "back_tint_deg": [-30.0, 30.0],
        "color_smooth": [1, 2],
    })

    # ── storage ───────────────────────────────────────────────────────────────
    data_root: str = "data/synthetic"
    tmp_dir: str = "data/synthetic/tmp"      # raw downloads live here while processed
    keep_raw: bool = False                   # delete each raw GLB once its pair is written
    storage_budget_gb: float = 150.0

    # ── resolution report ─────────────────────────────────────────────────────
    windows: tuple = (64, 96, 128)
    window_overlap: int = 16

    # ── rendering for review ──────────────────────────────────────────────────
    # "all": every panel + review.png (pilot-v2: ~800 KB a pair against ~15 KB of training
    # files); "review": review.png only; "none": no images (a full run: 15k pairs x 800 KB
    # would be ~12 GB of pictures).
    renders: str = "all"
    render_size: int = 360

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)

    def hash(self) -> str:
        """Short content hash of every setting that can change a pair (not the run name)."""
        d = asdict(self)
        for k in ("run_name", "data_root", "tmp_dir", "keep_raw", "storage_budget_gb", "renders", "render_size"):
            d.pop(k, None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    @property
    def run_dir(self) -> Path:
        return Path(self.data_root) / self.run_name

    @classmethod
    def from_dict(cls, raw: dict) -> "PipelineConfig":
        """Build from a dict (e.g. parsed run_config.json); unknown keys are ignored."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: (tuple(v) if isinstance(v, list) and k in ("licences", "windows") else v)
                      for k, v in raw.items() if k in known})

    @classmethod
    def load(cls, path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path=None) -> Path:
        path = Path(path) if path else self.run_dir / "run_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.loads(self.to_json())
        body["pipeline_version"] = PIPELINE_VERSION
        body["config_hash"] = self.hash()
        path.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
        return path
