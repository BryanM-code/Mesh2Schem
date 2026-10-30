"""Blocks per storey: information only (builds are never rescaled or filtered on it).

Storey height here is floor-to-floor: the distance between the walking surfaces of two
consecutive floors. Two independent readings:

* **Floors**: inside the building's rooms (the columns over interior air; the footprint
  minus its outer ring when the mask found no rooms), a layer is a floor plane when
  structural blocks (full cubes and slabs) cover at least `FLOOR_COVER` of the columns
  that still have building above that height. Consecutive plane layers are one floor
  (a ceiling under a floor); its top layer is the floor's level. The roof or top ceiling
  closes the top storey. The ground level counts as the first floor when the building
  has no floor of its own within `PLAUSIBLE[0]` blocks of it (its floor was the plate).
* **Windows**: on the shell, layers with glass (blocks and panes) form window rows; the
  spacing between the bottoms of consecutive rows.

Spacings outside `PLAUSIBLE` blocks are discarded (a nave, an atrium, a mezzanine).

    high     3+ floor spacings, 75% within +-1 of their median; or 2 that agree with windows
    medium   2 floor spacings within +-1; or 1 floor spacing confirmed by windows;
             or 3+ window spacings that agree
    low      a single floor spacing (typically a one-storey building: floor and roof)
    unknown  nothing plausible -- reported as unknown rather than guessed
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from minecraft import block_roles

from .read import split_state

FLOOR_COVER = 0.6
MIN_COLUMNS = 6
PLAUSIBLE = (3, 12)
WINDOW_ROW = 0.25            # a layer is a window row at this share of the busiest layer


def _base(state: str) -> str:
    return split_state(state)[0].rpartition(":")[2]


def _structural(palette: list[str]) -> np.ndarray:
    out = np.zeros(len(palette), dtype=bool)
    for i, s in enumerate(palette[1:], 1):
        base = _base(s)
        out[i] = (block_roles.is_cube(base) and not block_roles.is_plant(base)) or \
            base.endswith("_slab")
    return out


def _glass(palette: list[str]) -> np.ndarray:
    return np.array([i > 0 and "glass" in _base(s) for i, s in enumerate(palette)], dtype=bool)


def _groups(levels: np.ndarray) -> list[tuple[int, int]]:
    """Runs of consecutive integers -> [(first, last), ...]."""
    out = []
    for y in levels.tolist():
        if out and y == out[-1][1] + 1:
            out[-1] = (out[-1][0], y)
        else:
            out.append((y, y))
    return out


def _plausible(spacings: list[int]) -> list[int]:
    return [s for s in spacings if PLAUSIBLE[0] <= s <= PLAUSIBLE[1]]


def floor_planes(blocks, palette, interior, ground_y: int) -> list[int]:
    structural = _structural(palette)[blocks]
    rooms = interior & (blocks == 0)
    rooms[:, :ground_y + 1, :] = False
    columns = rooms.any(axis=1)
    if columns.sum() < MIN_COLUMNS:
        solid_cols = structural[:, ground_y + 1:, :].any(axis=1)
        columns = ndimage.binary_erosion(solid_cols, structure=np.ones((3, 3), bool))
    if columns.sum() < MIN_COLUMNS:
        return []
    Y = blocks.shape[1]
    has = structural & columns[:, None, :]
    top = np.where(has.any(axis=1), Y - 1 - np.argmax(has[:, ::-1, :], axis=1), -1)
    planes = []
    for y in range(max(ground_y + 1, 0), Y):
        active = columns & (top >= y)
        n = int(active.sum())
        if n < max(MIN_COLUMNS, 0.1 * columns.sum()):
            continue
        if int((structural[:, y, :] & active).sum()) >= FLOOR_COVER * n:
            planes.append(y)
    levels = [last for _, last in _groups(np.array(planes, dtype=int))]
    # A building whose floor was the ground (removed by the crop) walks on ground_y.
    if ground_y >= 0 and (not levels or levels[0] - ground_y >= PLAUSIBLE[0]):
        levels.insert(0, ground_y)
    return levels


def window_rows(blocks, palette, shell) -> list[int]:
    glass = _glass(palette)[blocks] & shell
    per_y = glass.sum(axis=(0, 2))
    if per_y.max(initial=0) < 4:
        return []
    rows = np.flatnonzero(per_y >= WINDOW_ROW * per_y.max())
    return [first for first, _ in _groups(rows)]


def _median(values: list[int]) -> int:
    return int(round(float(np.median(values))))


def estimate(blocks, palette, masks, ground_y: int) -> dict:
    floors = floor_planes(blocks, palette, masks.interior, ground_y)
    windows = window_rows(blocks, palette, masks.shell)
    f_sp = _plausible(list(np.diff(floors))) if len(floors) > 1 else []
    w_sp = _plausible(list(np.diff(windows))) if len(windows) > 1 else []
    f_med = _median(f_sp) if f_sp else None
    w_med = _median(w_sp) if w_sp else None
    agree = f_med is not None and w_med is not None and abs(f_med - w_med) <= 1

    est, conf, method = None, "unknown", None
    if f_sp:
        close = sum(abs(s - f_med) <= 1 for s in f_sp)
        if len(f_sp) >= 3 and close >= 0.75 * len(f_sp):
            est, conf, method = f_med, "high", "floors"
        elif len(f_sp) >= 2 and close == len(f_sp):
            est, conf, method = f_med, "high" if agree else "medium", "floors"
        elif len(f_sp) == 1 and agree:
            est, conf, method = f_med, "medium", "floors+windows"
        elif len(f_sp) >= 2:
            est, conf, method = f_med, "low", "floors"
        else:
            est, conf, method = f_med, "low", "floors"
    if conf in ("unknown", "low") and w_sp and len(w_sp) >= 3:
        close = sum(abs(s - w_med) <= 1 for s in w_sp)
        if close >= 0.75 * len(w_sp) and (est is None or abs(est - w_med) <= 1):
            est, conf, method = w_med, "medium", "windows"
    return {"blocks_per_storey": est if est is not None else "unknown",
            "confidence": conf, "method": method,
            "floor_levels": floors, "floor_spacings": [int(s) for s in np.diff(floors)],
            "window_rows": windows[:40], "window_spacings": [int(s) for s in np.diff(windows)][:40],
            "storeys": (len(floors) - 1) if len(floors) > 1 else None}
