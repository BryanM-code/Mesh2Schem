"""Blockiness: how much a mesh looks like it was already made of cubes (the pairs' check that
an input is no blockier than TRELLIS output)."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _house_glb(path, width=20.0, depth=14.0, wall_h=6.0, roof_h=7.5, turn_deg=17.0, scale=1.137):
    """Box walls + a 45-degree gable roof + a translucent pane, turned and scaled off-grid."""
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    def coloured(mesh, rgba, alpha_mode="OPAQUE", name="m"):
        mesh.visual = TextureVisuals(material=PBRMaterial(name=name, baseColorFactor=rgba, alphaMode=alpha_mode))
        return mesh

    walls = trimesh.creation.box(extents=(width, wall_h, depth))
    walls.apply_translation((width / 2, wall_h / 2, depth / 2))
    tri = np.array([[0, wall_h, -0.5], [0, wall_h, depth + 0.5], [0, wall_h + roof_h, depth / 2]])
    v = np.vstack([tri, tri + [width, 0, 0]])
    f = np.array([[0, 1, 2], [3, 5, 4], [0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2], [2, 5, 3], [2, 3, 0]])
    roof = trimesh.Trimesh(v, f)
    roof.fix_normals()
    pane = trimesh.creation.box(extents=(3, 2, 0.1))
    pane.apply_translation((width / 2, wall_h / 2, depth + 0.02))
    # Real models are neither axis-aligned nor on round numbers; a box at integer sizes is
    # exactly the primitive "blocky" model the geometry check rejects.
    turn = trimesh.transformations.rotation_matrix(np.radians(turn_deg), [0, 1, 0])
    turn[:3, :3] *= scale
    scene = trimesh.Scene()
    for mesh in (coloured(walls, [220, 210, 190, 255], name="plaster"),
                 coloured(roof, [120, 60, 40, 255], name="tiles"),
                 coloured(pane, [150, 200, 230, 90], "BLEND", name="window_glass")):
        mesh.apply_transform(turn)
        scene.add_geometry(mesh)
    scene.export(str(path))


class Blockiness(unittest.TestCase):
    def test_voxel_build_flagged_real_model_not(self):
        import trimesh
        from voxel.scene import load_scene
        from pairs.blockiness import blockiness
        cubes = []
        rng = np.random.default_rng(0)
        for x, y, z in np.argwhere(rng.random((6, 5, 6)) < 0.5):
            c = trimesh.creation.box(extents=(1, 1, 1))
            c.apply_translation((x + 0.5, y + 0.5, z + 0.5))
            cubes.append(c)
        vox = trimesh.util.concatenate(cubes)
        vox.vertices += 1e-4 * rng.random(vox.vertices.shape)    # exporters' anti-z-fight insets
        self.assertTrue(blockiness(vox.vertices, vox.faces)["blocky"])
        tmp = Path(tempfile.mkdtemp())
        try:
            _house_glb(tmp / "h.glb")
            sm = load_scene(tmp / "h.glb")
            self.assertFalse(blockiness(sm.verts, sm.faces)["blocky"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
