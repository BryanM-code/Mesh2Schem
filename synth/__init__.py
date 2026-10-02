"""Offline generator of (TRELLIS-like input voxels -> Minecraft blocks) training pairs.

Not part of the Flask app. The voxel core lives in `mcvoxel` (shared with Real2Minecraft,
installed from requirements-synth.txt); this package decides WHAT to voxelise, rejects what
cannot make a good pair (always with a reason), writes the pairs and measures them.
docs/dataset-pipeline.md describes the whole pipeline.

    config           every knob, saved with each run (PipelineConfig)
    taxonomy         architectural category / subcategory / group of an asset from its text
    objaverse_index  metadata-only index of Objaverse 1.0 buildings
    select           licence + metadata filters, then quotas for diversity
    filters          reason codes; metadata, classifier-hook and geometry checks
    fetch            streamed GLB download with checksum
    orient           yaw alignment
    scale            real size at 1:1 with the player, with plausibility bounds
    target           the clean target in both roof styles
    degrade          TRELLIS-like degradation of a clean mesh (the model's input)
    validate         the checks every pair must pass
    pair             one asset -> pair on disk, or a recorded rejection
    run              orchestration: stream, resume, ledger, delete raw
    preview          review renders
    review           local web tool to look at pairs and record verdicts
    report           statistics and contact sheets for a run
    roof_stats       roof construction measured on real builds and on targets
    real_eval        the held-out real evaluation set and its training guard
    trellis_compare  real TRELLIS vs the synthetic degradation, on the same buildings
    calibrate        unpaired comparison with the cached TRELLIS meshes
    blockiness       detector for sources that are already made of blocks

Licensing gates everything: see docs/dataset-licensing.md. Only CC0 and CC BY 4.0 objects
are selected, and every output carries its source's attribution.
"""
