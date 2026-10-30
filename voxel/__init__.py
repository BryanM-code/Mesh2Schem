"""voxel: meshes -> Minecraft blocks, and the block vocabulary the model predicts.

Modules
    raster          conservative surface voxelization, exterior flood fill, `clean_solid`
    scene           multi-material GLB loader (per-face material, glass / door kinds)
    voxel_sampling  per-voxel colour from surface samples (mode-seeking, side-aware)
    surface_color   per-voxel colour as seen from outside (rays through air-facing faces)
    grid            scene -> voxel model at any size and in any frame; visible shell,
                    enclosed rooms, ground plate and site, slab / stair shapes, surface
                    repeats (slivers), colours
    architecture    enclosed cross-sections, used by the grid's room finding
    details         thin structural posts the grid keeps
    block_palette   block -> sRGB palettes and weighted-CIELAB matching
    block_shapes    full / slab / stairs classification from sub-block occupancy, block variants
    vocab           the factored SHAPE x MATERIAL vocabulary: VALID pairs, training-target
                    encoding, 1.21.11 block state validation
    blockstates     rotation / mirror for every shape; states worked out from neighbours
                    (stair corners, connections, door halves and hinges, gate in_wall)
    tiling          windowed prediction and reassembly for large buildings
    yaw_align       rotate a mesh so its facades sit on the grid axes

Arrays are (X, Y, Z) with Y up, +X east, +Z south; `.schem` files are (Y, Z, X).
"""
