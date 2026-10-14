"""Tooling for the human-built schematic dataset (the version 4 training targets).

Not part of the Flask app. docs/mesh-to-schem-plan.md (version 4) is the plan;
docs/real-schems-v1.md describes the frozen dataset.

    freeze     verbatim copy + checksums, then index, licence tiers, near-duplicate
               candidates and creator splits (Phase 2); D24 set-asides and duplicate moves
    normalize  Phase 3 driver: read -> crop -> masks -> storey -> targets -> coverage
    read       any schematic format -> palette-indexed (X, Y, Z) grid, validated
    crop       natural terrain, trees and site structures removed; ground height per
               column (D23); terrain floors under rooms
    interior   shell / exterior / interior (ignored) masks by opening size (D22), and the
               visibility audit
    storey     blocks per storey + confidence (information only, D15)
    coverage   how much of the supervised data the model vocabulary can express
    dedupe     geometric duplicates after the crop (8 rotations / mirrors, offsets)
    gallery    the review gallery (review/gallery_v2/), drawn with `review`
    review     voxel renders and contact sheets
"""
