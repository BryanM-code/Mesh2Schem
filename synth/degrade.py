"""Turn a clean architectural model into what TRELLIS would plausibly have produced.

The model's input at inference is a TRELLIS mesh, so a training input must carry
TRELLIS's imperfections -- not arbitrary noise. Each step below reproduces one measured
or documented property:

* **Resolution bottleneck.** TRELLIS generates every object inside a 64^3 sparse structure
  normalised to the object's longest axis, and decodes a surface at a few times that. So
  detail is lost RELATIVE to object size: a 30-block house keeps sub-block detail, a
  200-block tower loses anything under ~2-3 blocks. The clean model is voxelised at
  `bottleneck` cells across its longest axis (96-160), blurred, and re-extracted with
  marching cubes: rounded corners, lost railings and mullions, merged thin walls.
  `min_blur_units` (default 0, off) sets a floor on that blur in the scene's own units:
  `schemset.pairs` uses it so Minecraft's one-block stair steps do not survive into an
  input of a small build (real buildings, and TRELLIS's reconstructions of them, have
  smooth roofs).
* **Surface noise and "melting".** Smooth low-frequency displacement along the normals
  (Real2Minecraft's notes call TRELLIS buildings "melted statues").
* **Holes and floaters.** Measured on the 54 cached TRELLIS meshes: 0% watertight, median
  11.5k open edges, median 543 connected components. Random face patches are removed and
  small detached fragments added.
* **Baked, darkened colour.** Real2Minecraft measured TRELLIS bakes ~2x darker than the
  photo, with hue drift on real surfaces and baked shadows. Colour is darkened, lit from a
  random sun direction, hue-rotated and noised.
* **Hallucinated unseen side.** Surfaces facing away from a random camera get a blend of
  the mirrored visible side, darker, and sometimes a wrong tint (the "teal walls" failure).

Every parameter is drawn per sample from `sample_params` and stored in the manifest. The
ranges were tuned once against the cached TRELLIS meshes (`synth.calibrate`, 2026-09-24):
v0.1 inputs came out too bright, too saturated and too rough.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from mcvoxel.block_palette import lab_to_srgb, srgb_to_lab
from mcvoxel.grid import voxelize_scene
from mcvoxel.scene import KIND_SURFACE, Material, SceneMesh


@dataclass
class DegradeParams:
    bottleneck: int = 128
    blur_sigma: float = 1.0          # bottleneck voxels
    surface_noise: float = 0.3       # bottleneck voxels, amplitude of normal displacement
    noise_wavelength: float = 8.0    # bottleneck voxels
    hole_fraction: float = 0.02      # share of faces removed in patches
    hole_patch: int = 40             # faces per hole
    floaters: int = 20
    tone_gain: float = 0.7
    ambient: float = 0.7
    hue_deg: float = 0.0
    chroma: float = 1.0
    color_noise: float = 4.0         # RGB std
    back_blend: float = 0.5          # how much of the unseen side is the mirrored side
    back_darken: float = 0.85
    back_tint_deg: float = 0.0       # 0 = no wrong tint
    color_smooth: int = 2            # Laplacian passes over vertex colours
    seed: int = 0
    min_blur_units: float = 0.0      # blur sigma floor in scene units (0 = off; not sampled)
    chroma_gamma: float = 1.0        # < 1 mutes strong colours more than weak ones (1 = off)
    # Optional (0 = off; not sampled here): TRELLIS texture artefacts, in scene units.
    blotch: float = 0.0              # lightness blotches: sd of a multiplicative factor
    blotch_units: float = 2.0        # ... their size
    warp_units: float = 0.0          # colours sampled this far off (wavy streaks, smears)
    warp_wavelength_units: float = 8.0
    cast_a: float = 0.0              # a white-balance cast, CIELAB a* / b* added
    cast_b: float = 0.0
    tint_ab: float = 0.0             # regional hue drift: sd of a slowly varying a* / b* shift
    tint_units: float = 12.0         # ... its size
    roof_darken: float = 0.0         # up-facing surfaces darker by up to this share (roofs)
    target_lum: float = 0.0          # > 0: scale L* so its mean is this, chroma kept (TRELLIS sets
                                     # its own exposure; the build's brightness does not)


def sample_params(rng: np.random.Generator, ranges: dict | None = None) -> DegradeParams:
    """Draw one sample's parameters from `ranges` (`PipelineConfig.degrade`).

    The defaults are the v0.2 ranges calibrated against the cached TRELLIS meshes
    (`synth.calibrate`): v0.1 inputs came out too bright, too saturated and too rough.
    """
    if ranges is None:
        from .config import PipelineConfig
        ranges = PipelineConfig().degrade
    r = ranges

    def uni(key):
        return float(rng.uniform(*r[key]))

    def integer(key):
        return int(rng.integers(r[key][0], r[key][1] + 1))

    tint = float(rng.uniform(*r["back_tint_deg"])) if rng.random() < r["back_tint_prob"] else 0.0
    return DegradeParams(
        bottleneck=int(rng.choice(r["bottleneck"])),
        blur_sigma=uni("blur_sigma"),
        surface_noise=uni("surface_noise"),
        noise_wavelength=uni("noise_wavelength"),
        hole_fraction=uni("hole_fraction"),
        hole_patch=integer("hole_patch"),
        floaters=integer("floaters"),
        tone_gain=uni("tone_gain"),
        ambient=uni("ambient"),
        hue_deg=float(rng.normal(0.0, r["hue_deg_sd"])),
        chroma=uni("chroma"),
        color_noise=uni("color_noise"),
        back_blend=uni("back_blend"),
        back_darken=uni("back_darken"),
        back_tint_deg=tint,
        color_smooth=integer("color_smooth"),
        seed=int(rng.integers(0, 2 ** 31 - 1)),
    )


def degrade(scene: SceneMesh, p: DegradeParams):
    """Return (degraded SceneMesh in the same file units and frame, info)."""
    import trimesh
    from scipy import ndimage
    from skimage.measure import marching_cubes

    rng = np.random.default_rng(p.seed)
    lo, hi = scene.bounds
    extent = float((hi - lo).max())
    s = p.bottleneck / extent
    off = -lo * s + 0.5
    vm = voxelize_scene(scene, s, offset=off, drop_ground=False, keep_largest=False, clean=False,
                        with_shapes=False, with_colors=True, color_fallback="nearest")
    sigma = max(p.blur_sigma, p.min_blur_units * s)
    pad = max(3, int(np.ceil(3 * sigma)) + 1)       # room for the blur beyond the model
    field = ndimage.gaussian_filter(np.pad(vm.solid.astype(np.float32), pad), sigma)
    if field.max() < 0.5:
        raise ValueError("model vanished at the bottleneck resolution")
    verts, faces, _, _ = marching_cubes(field, level=0.5)
    mesh = trimesh.Trimesh(verts, faces, process=False)
    if mesh.volume < 0:
        mesh.invert()
    normals = np.asarray(mesh.vertex_normals)
    v = np.asarray(mesh.vertices, dtype=np.float64)

    # Low-frequency displacement along the normal.
    if p.surface_noise > 0:
        disp = np.zeros(len(v))
        for _ in range(4):
            k = rng.normal(size=3)
            k *= (2 * np.pi / p.noise_wavelength) / np.linalg.norm(k)
            disp += np.sin(v @ k + rng.uniform(0, 2 * np.pi))
        v = v + normals * (disp / 2.0 * p.surface_noise)[:, None]

    # Colour: nearest shell voxel of the clean bottleneck model.
    shell = vm.shell
    _, idx = ndimage.distance_transform_edt(~shell, return_indices=True)

    def color_at(points):
        c = np.clip(np.rint(points - pad).astype(np.int64), 0, np.array(shell.shape) - 1)
        src = tuple(idx[a][c[:, 0], c[:, 1], c[:, 2]] for a in range(3))
        return vm.rgb[src].astype(np.float64)

    # TRELLIS texture artefacts draw from their own stream, so the draws below (and the
    # geometry) are the same with or without them.
    art = np.random.default_rng([int(p.seed), 0x7E])
    if p.warp_units > 0:
        # Colours read from a slowly wandering offset: fine stripes turn into wavy streaks
        # and edges smear, as TRELLIS's baked textures do on busy facades.
        warp = _value_noise(v, p.warp_wavelength_units * s, art, 3) * (p.warp_units * s)
        rgb = color_at(v + warp)
    else:
        rgb = color_at(v)

    # Unseen side: mirrored visible side, darker, sometimes tinted.
    az = rng.uniform(0, 2 * np.pi)
    el = rng.uniform(np.radians(5), np.radians(35))
    cam = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
    back = normals @ cam < -0.2
    if back.any():
        centre = v.mean(axis=0)
        mirrored = v[back] - 2.0 * ((v[back] - centre) @ cam)[:, None] * cam[None, :]
        blend = p.back_blend * color_at(mirrored) + (1 - p.back_blend) * rgb[back]
        blend *= p.back_darken
        if p.back_tint_deg:
            blend = _hue_rotate(blend, p.back_tint_deg, 0.9)
        rgb[back] = blend

    # Baked lighting, darkening, hue drift, noise, then TRELLIS-smooth colour.
    sun = np.array([rng.normal(), rng.uniform(0.5, 1.5), rng.normal()])
    sun /= np.linalg.norm(sun)
    shade = p.ambient + (1 - p.ambient) * np.clip(normals @ sun, 0, None)
    if p.roof_darken > 0:
        # TRELLIS on real photos bakes roofs dark (seen at a grazing angle, or not at all).
        shade = shade * (1.0 - p.roof_darken * np.clip(normals[:, 1], 0, 1))
    rgb = rgb * shade[:, None] * p.tone_gain
    rgb = _hue_rotate(rgb, p.hue_deg, p.chroma, p.chroma_gamma, p.cast_a, p.cast_b)
    if p.tint_ab > 0:
        # Whole regions drift in hue (a blue-grey roof over brown walls, a purple roof): a
        # slowly varying a* / b* shift.
        lab = srgb_to_lab(np.clip(rgb, 0, 255))
        lab[:, 1:] += p.tint_ab * _value_noise(v, p.tint_units * s, art, 2)
        rgb = lab_to_srgb(lab)
    if p.blotch > 0:
        # Lighter and darker patches about a block or two across: TRELLIS textures vary from
        # block to block far more than per-vertex noise, which averages out, makes them.
        rgb = rgb * np.clip(1.0 + p.blotch * _value_noise(v, p.blotch_units * s, art, 1), 0.2, 2.0)
    if p.target_lum > 0:
        # Exposure set by the output, not the build: L* scaled to the target mean. Chroma is
        # left alone, as in TRELLIS's output, where colourfulness does not follow lightness (an
        # sRGB gain would make the brightest inputs the most colourful).
        lab = srgb_to_lab(np.clip(rgb, 0, 255))
        lab[:, 0] = np.clip(lab[:, 0] * (p.target_lum / max(float(lab[:, 0].mean()), 1e-3)), 0.0, 100.0)
        rgb = lab_to_srgb(lab)
    rgb = rgb + rng.normal(0, p.color_noise, size=rgb.shape)
    if p.color_smooth:
        adj = mesh.vertex_adjacency_graph
        import scipy.sparse as sp
        n = len(v)
        rows, cols = zip(*adj.edges()) if adj.number_of_edges() else ((), ())
        A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        A = (A + A.T).tocsr()
        deg = np.asarray(A.sum(axis=1)).ravel() + 1.0
        for _ in range(p.color_smooth):
            rgb = (A @ rgb + rgb) / deg[:, None]
    rgb = np.clip(rgb, 0, 255)

    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_faces_mc = len(faces)
    faces, holes = _punch_holes(faces, mesh, p, rng)
    v, faces, rgb, n_float = _add_floaters(v, faces, rgb, normals, p, rng)

    # Back to file units: bottleneck index coords -> scaled frame -> file units.
    pts_scaled = v - pad + vm.origin + 0.5
    verts_units = (pts_scaled - off) / s
    out = SceneMesh(verts=verts_units, faces=faces, uv=np.full((len(verts_units), 2), np.nan),
                    vcolor=rgb, face_mat=np.zeros(len(faces), np.int32),
                    materials=[Material(name="trellis_bake", base_rgba=np.ones(4), image=None, image_alpha=None,
                                        alpha_mode="OPAQUE", kind=KIND_SURFACE)])
    info = {"params": asdict(p), "blur_sigma_cells": round(float(sigma), 3), "mc_faces": int(n_faces_mc),
            "faces": int(len(faces)),
            "holes": int(holes), "floater_faces": int(n_float), "back_facing_share": round(float(back.mean()), 3)}
    info.update(mesh_stats(verts_units, faces))
    return out, info


def mesh_stats(verts, faces):
    """The three properties measured on the cached TRELLIS meshes, for calibration."""
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=False)
    edges = np.sort(m.edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    comps = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(faces)))
    return {"open_edges": int((counts == 1).sum()), "components": int(len(comps)),
            "watertight": bool((counts == 2).all())}


CHROMA_REF = 8.0                     # C* left unchanged by `chroma_gamma`


def _hue_rotate(rgb, deg, chroma, gamma=1.0, cast_a=0.0, cast_b=0.0):
    """Rotate hue by `deg` and scale chroma by `chroma`; `gamma` < 1 also compresses it
    towards CHROMA_REF (C' = chroma * REF * (C / REF) ** gamma); then add a white-balance
    cast (`cast_a`, `cast_b` to a*, b*)."""
    lab = srgb_to_lab(np.clip(rgb, 0, 255))
    t = np.radians(deg)
    a, b = lab[..., 1], lab[..., 2]
    scale = np.full(a.shape, float(chroma))
    if gamma != 1.0:
        c = np.hypot(a, b)
        scale = scale * np.where(c > 1e-6, (np.maximum(c, 1e-6) / CHROMA_REF) ** (gamma - 1.0), 1.0)
    lab[..., 1] = (a * np.cos(t) - b * np.sin(t)) * scale + cast_a
    lab[..., 2] = (a * np.sin(t) + b * np.cos(t)) * scale + cast_b
    return lab_to_srgb(lab)


def _value_noise(points, wavelength, rng, channels=1):
    """A smooth random field at `points` (N, 3): a normal random lattice of spacing
    `wavelength` (the points' units) interpolated with cubic splines, then each channel set
    to zero mean and unit spread over the points. (N, channels)."""
    from scipy.ndimage import map_coordinates
    q = (points - points.min(axis=0)) / max(float(wavelength), 1e-6) + 1.0
    shape = tuple(int(n) for n in np.ceil(q.max(axis=0)) + 3)
    out = np.empty((len(points), channels))
    for k in range(channels):
        val = map_coordinates(rng.normal(size=shape), q.T, order=3, mode="nearest")
        out[:, k] = (val - val.mean()) / (val.std() + 1e-9)
    return out


def _punch_holes(faces, mesh, p, rng):
    """Remove `hole_fraction` of the faces as connected patches of ~`hole_patch` faces."""
    n_holes = int(p.hole_fraction * len(faces) / max(p.hole_patch, 1))
    if n_holes <= 0:
        return faces, 0
    adj = mesh.face_adjacency
    nbrs = [[] for _ in range(len(faces))]
    for a, b in adj:
        nbrs[a].append(b)
        nbrs[b].append(a)
    drop = np.zeros(len(faces), bool)
    for seed in rng.choice(len(faces), size=n_holes, replace=False):
        frontier, taken = [int(seed)], 0
        while frontier and taken < p.hole_patch:
            f = frontier.pop(0)
            if drop[f]:
                continue
            drop[f] = True
            taken += 1
            frontier.extend(nbrs[f])
    return faces[~drop], n_holes


def _add_floaters(v, faces, rgb, normals, p, rng):
    """Small detached copies of surface patches, lifted off along their normal."""
    if p.floaters <= 0 or len(faces) == 0:
        return v, faces, rgb, 0
    new_v, new_f, new_c = [v], [faces], [rgb]
    base = len(v)
    added = 0
    for _ in range(p.floaters):
        f0 = faces[rng.integers(len(faces))]
        k = int(rng.integers(1, 6))
        tri = v[f0]
        n = normals[f0].mean(axis=0)
        n /= (np.linalg.norm(n) + 1e-9)
        lift = n * rng.uniform(0.6, 2.5)
        for j in range(k):
            jitter = rng.normal(0, 0.4, size=3)
            new_v.append(tri + lift + jitter)
            new_f.append(np.array([[base, base + 1, base + 2]]))
            new_c.append(rgb[f0])
            base += 3
            added += 1
    return np.vstack(new_v), np.vstack(new_f), np.vstack(new_c), added
