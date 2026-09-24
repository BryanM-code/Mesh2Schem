#!/usr/bin/env python3
"""Tile a folder of `.schem` files into one labelled PNG, for eyeballing a batch.

Usage:  python contact_sheet.py <folder> [--out sheet.png] [--cols 4] [--cell 300] [--limit 24]

The acceptance test for synthetic houses is not a number, it is whether a person looking at
twenty of them together says "those are suburban houses" and "those are twenty different
houses". Neither question survives being asked one file at a time, hence a sheet.

It goes straight to `thumbs.render_schem`, which takes a path and returns a PIL image without
touching the database or the thumbnail cache. That means the same command renders the real
library too:

    python contact_sheet.py data/builds --out real.png
    python contact_sheet.py data/.tmp/synthetic --out synthetic.png

Putting those two side by side is the actual Phase 3 review -- the generated distribution is
supposed to look like it belongs with the collected one, and the only way to know is to look.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import thumbs

PAD = 8
LABEL_H = 16


def collect(target: Path, limit: int) -> list:
    """Every schematic under `target`, whether it is a build library or a flat folder."""
    if target.is_file():
        return [target]
    paths = sorted(target.glob("*/schem/original.schem"))
    if not paths:
        paths = sorted(target.glob("*.schem"))
    if not paths:
        paths = sorted(target.rglob("*.schem"))
    return paths[:limit]


def label_for(path: Path) -> str:
    # `data/builds/<id>__<slug>/schem/original.schem` is more usefully labelled by its build
    # folder than by the filename every build shares.
    if path.name == "original.schem" and path.parent.name == "schem":
        return path.parent.parent.name
    return path.stem


def build_sheet(paths, cols: int, cell: int):
    from PIL import Image, ImageDraw

    tiles = []
    for path in paths:
        try:
            image = thumbs.render_schem(path)
        except Exception:  # a sheet should not die on one unreadable file
            image = None
        if image is None:
            print("  skipped (unreadable): {}".format(path))
            continue
        image = image.convert("RGBA")
        image.thumbnail((cell, cell))
        tiles.append((label_for(path), image))

    if not tiles:
        return None

    cols = max(1, min(cols, len(tiles)))
    rows = (len(tiles) + cols - 1) // cols
    step_x, step_y = cell + PAD, cell + PAD + LABEL_H
    sheet = Image.new("RGBA", (cols * step_x + PAD, rows * step_y + PAD), (22, 22, 26, 255))
    draw = ImageDraw.Draw(sheet)

    for i, (label, image) in enumerate(tiles):
        col, row = i % cols, i // cols
        x = PAD + col * step_x + (cell - image.width) // 2
        y = PAD + row * step_y + (cell - image.height)
        sheet.alpha_composite(image, (x, y))
        draw.text((PAD + col * step_x + 2, PAD + row * step_y + cell + 3),
                  label[:38], fill=(210, 210, 216, 255))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="a .schem, a folder of them, or a build library")
    parser.add_argument("--out", default="contact_sheet.png")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--cell", type=int, default=300)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    paths = collect(Path(args.target), args.limit)
    if not paths:
        print("No .schem files under {}".format(args.target))
        return
    sheet = build_sheet(paths, args.cols, args.cell)
    if sheet is None:
        print("Nothing could be rendered.")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.convert("RGB").save(out)
    print("{} builds -> {}  ({}x{})".format(len(paths), out, sheet.width, sheet.height))


if __name__ == "__main__":
    main()
