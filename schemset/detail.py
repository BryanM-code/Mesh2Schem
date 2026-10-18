"""Detail retention: how much of a build's fine structure survives in an input (Phase 4
follow-up).

Measured the same way on every kind of input -- synthetic, TRELLIS on renders, TRELLIS on
real photos -- always against a *reference*: the build's clean surface (`schemset.meshify`,
glass in `pairs.REFERENCE_GLASS`) voxelised exactly like the input (`pairs.voxelize_input`,
2 cells per block, with colours) in the input's frame: for a synthetic input, placed with
its variant's yaw and offset; for a TRELLIS input, in the target frame it was aligned to.

    detail_geo       the reference's small features -- cells where it differs from its own
                     open-close smoothing with a `DETAIL_K`-cell cube: protrusions and
                     recesses thinner than that (window reveals, trim, cornices, railings,
                     eaves, dormer cheeks) -- and the share of them on which the input sides
                     with the reference rather than with the smoothed reference. 1: every
                     feature kept; 0: all smoothed away
    detail_spurious  within `DETAIL_BAND` cells of the reference surface, off those
                     features: the share of cells where the input differs from the
                     reference (noise, bumps, blobs, missing walls)
    detail_colour    correlation of band-passed lightness (L*, a difference of Gaussians of
                     `DETAIL_SIGMAS` cells, taken on the surface only) on the reference
                     surface: windows, trim and banding where the reference has them.
                     Exposure and white balance do not change it

Each is the best over input shifts of up to `DETAIL_SHIFT` cell along one axis, so what is
left of an alignment (a TRELLIS input is placed within about half a cell) is not counted as
lost detail. Grids are processed in tiles of `TILE` cells with margins, so big builds fit.
"""
from __future__ import annotations

import numpy as np

DETAIL_K = 3                   # cells (2 per block): features thinner than 1.5 blocks
DETAIL_SIGMAS = (1.0, 3.0)     # cells: the colour band, half a block to 1.5 blocks
DETAIL_SHIFT = 1               # cells, along one axis at a time
DETAIL_BAND = 2                # cells either side of the reference surface (spurious)
DENSITY_MIN = 0.05             # input surface density for a colour sample to count
TILE = 128
MARGIN = 12                    # >= 3 * the larger sigma + DETAIL_SHIFT


def _shifts():
    out = [(0, 0, 0)]
    for axis in range(3):
        for d in range(1, DETAIL_SHIFT + 1):
            for sign in (1, -1):
                s = [0, 0, 0]
                s[axis] = sign * d
                out.append(tuple(s))
    return out


def lstar(rgb: np.ndarray) -> np.ndarray:
    """CIE L* of sRGB values (uint8, last axis RGB)."""
    c = np.asarray(rgb, dtype=np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    y = lin @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.where(y > 216 / 24389, 116.0 * np.cbrt(y) - 16.0, y * 24389 / 27).astype(np.float32)


def _surface_bandpass(values, mask):
    """Band-pass of a field known only on `mask` (normalised convolution), and the mask's
    density at the finer scale."""
    from scipy.ndimage import gaussian_filter
    m = mask.astype(np.float32)
    v = values * m
    fine_w = gaussian_filter(m, DETAIL_SIGMAS[0])
    fine = gaussian_filter(v, DETAIL_SIGMAS[0]) / np.maximum(fine_w, 1e-6)
    coarse = gaussian_filter(v, DETAIL_SIGMAS[1]) / np.maximum(gaussian_filter(m, DETAIL_SIGMAS[1]), 1e-6)
    return fine - coarse, fine_w


def _tile_counts(occ, L, surf, ref_occ, ref_L, ref_surf, core, shifts, acc):
    from scipy import ndimage
    cube = np.ones((DETAIL_K,) * 3, bool)
    smooth = ndimage.binary_closing(ndimage.binary_opening(ref_occ, cube), cube)
    feat = (ref_occ != smooth) & core
    ref_shell = ref_occ & ~ndimage.binary_erosion(ref_occ)
    band = ndimage.binary_dilation(ref_shell, iterations=DETAIL_BAND) & core & ~feat
    bp_ref, _ = _surface_bandpass(ref_L, ref_surf)
    bp_in, dens = _surface_bandpass(L, surf)
    sample = ref_surf & core
    n_feat, n_band = int(feat.sum()), int(band.sum())
    acc["n_feat"] += n_feat
    acc["n_band"] += n_band
    x_ref = bp_ref[sample].astype(np.float64)
    for i, s in enumerate(shifts):
        o = np.roll(occ, s, axis=(0, 1, 2)) if any(s) else occ
        if n_feat:
            acc["agree"][i] += int((o[feat] == ref_occ[feat]).sum())
        if n_band:
            acc["disagree"][i] += int((o[band] != ref_occ[band]).sum())
        b = np.roll(bp_in, s, axis=(0, 1, 2)) if any(s) else bp_in
        d = np.roll(dens, s, axis=(0, 1, 2)) if any(s) else dens
        ok = d[sample] >= DENSITY_MIN
        if ok.any():
            x, y = x_ref[ok], b[sample][ok].astype(np.float64)
            c = acc["colour"][i]
            c[0] += len(x)
            c[1] += x.sum()
            c[2] += y.sum()
            c[3] += (x * x).sum()
            c[4] += (y * y).sum()
            c[5] += (x * y).sum()


def detail_stats(occ, rgb, surface, ref_occ, ref_rgb, ref_surface) -> dict:
    """The detail metrics of an input against its reference (same frame and shape, 2x)."""
    occ, ref_occ = np.asarray(occ, bool), np.asarray(ref_occ, bool)
    surface, ref_surface = np.asarray(surface, bool), np.asarray(ref_surface, bool)
    if occ.shape != ref_occ.shape:
        raise ValueError(f"input {occ.shape} and reference {ref_occ.shape} differ")
    L, ref_L = lstar(rgb), lstar(ref_rgb)
    shifts = _shifts()
    acc = {"n_feat": 0, "n_band": 0, "agree": np.zeros(len(shifts), np.int64),
           "disagree": np.zeros(len(shifts), np.int64), "colour": np.zeros((len(shifts), 6))}
    near = np.argwhere(ref_surface)
    if not len(near):
        return {"detail_geo": None, "detail_spurious": None, "detail_colour": None, "detail_features": 0}
    lo, hi = near.min(axis=0), near.max(axis=0) + 1
    shape = np.array(occ.shape)
    for x0 in range(lo[0], hi[0], TILE):
        for y0 in range(lo[1], hi[1], TILE):
            for z0 in range(lo[2], hi[2], TILE):
                c0 = np.array([x0, y0, z0])
                c1 = np.minimum(c0 + TILE, hi)
                p0 = np.maximum(c0 - MARGIN, 0)
                p1 = np.minimum(c1 + MARGIN, shape)
                sl = tuple(slice(a, b) for a, b in zip(p0, p1))
                if not ref_surface[sl].any():
                    continue
                core = np.zeros(tuple(p1 - p0), bool)
                core[tuple(slice(a, b) for a, b in zip(c0 - p0, c1 - p0))] = True
                _tile_counts(occ[sl], L[sl], surface[sl], ref_occ[sl], ref_L[sl], ref_surface[sl],
                             core, shifts, acc)
    geo = acc["agree"] / acc["n_feat"] if acc["n_feat"] else None
    best = int(np.argmax(geo)) if geo is not None else 0
    spur = acc["disagree"] / acc["n_band"] if acc["n_band"] else None
    rs = []
    for n, sx, sy, sxx, syy, sxy in acc["colour"]:
        if n < 50:
            rs.append(np.nan)
            continue
        cov = sxy / n - (sx / n) * (sy / n)
        vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
        rs.append(cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan)
    rs = np.array(rs)
    return {"detail_geo": None if geo is None else round(float(geo[best]), 4),
            "detail_spurious": None if spur is None else round(float(spur[best]), 4),
            "detail_colour": None if np.isnan(rs).all() else round(float(np.nanmax(rs)), 4),
            "detail_features": int(acc["n_feat"]),
            "detail_shift_cells": list(shifts[best])}


def lab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB (D65) of sRGB values (uint8, last axis RGB) -> (..., 3) float32."""
    c = np.asarray(rgb, dtype=np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]],
                 dtype=np.float32)
    xyz = (lin @ m.T) / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
                    axis=-1).astype(np.float32)


def colour_cast(vm, ref) -> dict:
    """The input's overall colour against the reference's: mean L*, a*, b* over each one's
    surface, input minus reference (exposure, and a warm or cool cast such as sunset light)."""
    a = lab(vm.rgb[np.asarray(vm.surface, bool)]).reshape(-1, 3)
    b = lab(ref.rgb[np.asarray(ref.surface, bool)]).reshape(-1, 3)
    if not len(a) or not len(b):
        return {}
    d = a.mean(axis=0) - b.mean(axis=0)
    return {"cast_dL": round(float(d[0]), 2), "cast_da": round(float(d[1]), 2), "cast_db": round(float(d[2]), 2)}


def reference_model(clean, shape, yaw_deg: float = 0.0, offset=(0.0, 0.0, 0.0)):
    """The reference VoxelModel: the clean surface (glass in `pairs.REFERENCE_GLASS`) placed
    like the input and voxelised like it."""
    from .pairs import REFERENCE_GLASS, glass_look, place, voxelize_input
    mesh = glass_look(clean, REFERENCE_GLASS)
    if yaw_deg or any(offset):
        mesh = place(mesh, yaw_deg, offset)
    return voxelize_input(mesh, shape, colors=True)


def detail_of(vm, ref) -> dict:
    """detail_stats of a VoxelModel against a reference VoxelModel."""
    return detail_stats(vm.solid, vm.rgb, vm.surface, ref.solid, ref.rgb, ref.surface)
