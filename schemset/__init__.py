"""Tooling for the human-built schematic dataset (the version 4 training targets).

Not part of the Flask app. docs/mesh-to-schem-plan.md (version 4) is the plan;
docs/real-schems-v1.md describes the frozen dataset.

    freeze     verbatim copy + checksums, then index, licence tiers, near-duplicate
               candidates and creator splits (Phase 2)
    normalize  Phase 3 driver: read -> crop -> masks -> storey -> targets -> coverage
    read       any schematic format -> palette-indexed (X, Y, Z) grid, validated
    crop       remove ground, landscape and site clutter around the building
    interior   shell / exterior / interior (ignored) masks
    storey     blocks per storey + confidence (information only, D15)
    coverage   how much of the supervised data the model vocabulary can express
    review     before/after crop contact sheets and interior cutaways
"""
