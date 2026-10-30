"""training.score: the metrics a prediction is scored on."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voxel import vocab as V  # noqa: E402
from training import score as S  # noqa: E402

M = V.MATERIAL_INDEX
SH = V.SHAPE_INDEX


def target():
    """A 12 x 10 x 12 target: a hollow 6 x 5 x 6 brick box with a stair, air around it
    (supervised), the inside IGNORE."""
    shape = np.zeros((12, 10, 12), np.uint8)
    mat = np.zeros_like(shape, dtype=np.uint16)
    box = np.zeros_like(shape, bool)
    box[3:9, 0:5, 3:9] = True
    inner = np.zeros_like(box)
    inner[4:8, 0:4, 4:8] = True
    shell = box & ~inner
    shape[shell], mat[shell] = SH["full"], M["minecraft:bricks"]
    shape[3, 4, 5], mat[3, 4, 5] = SH["stairs_east"], M["minecraft:oak_planks"]
    shape[inner], mat[inner] = V.IGNORE_SHAPE, V.IGNORE_MATERIAL
    return shape, mat


class Score(unittest.TestCase):
    def test_a_perfect_prediction_scores_one(self):
        ts, tm = target()
        r = S.score(ts, tm, ts, tm)
        for k in ("iou", "f1", "f1_tol1", "precision", "recall", "shape_acc", "material_acc", "block_acc"):
            self.assertEqual(r[k], 1.0, k)
        self.assertEqual(r["material_de_p50"], 0.0)

    def test_ignored_voxels_are_never_scored(self):
        ts, tm = target()
        ps, pm = ts.copy(), tm.copy()
        ign = ts == S.IGNORE_SHAPE
        ps[ign], pm[ign] = SH["full"], M["minecraft:stone"]      # anything inside: unscored
        self.assertEqual(S.score(ps, pm, ts, tm)["iou"], 1.0)

    def test_a_one_block_shift_is_tolerated_only_by_f1_tol1(self):
        ts, tm = target()
        ps, pm = np.roll(ts, 1, axis=0), np.roll(tm, 1, axis=0)
        ps[ps == S.IGNORE_SHAPE] = 0
        r = S.score(ps, pm, ts, tm)
        self.assertLess(r["iou"], 0.8)
        self.assertEqual(r["f1_tol1"], 1.0)

    def test_a_near_colour_costs_less_than_a_far_one(self):
        ts, tm = target()
        solid = (ts != 0) & (ts != S.IGNORE_SHAPE)
        near, far = tm.copy(), tm.copy()
        near[solid & (tm == M["minecraft:bricks"])] = M["minecraft:granite"]
        far[solid & (tm == M["minecraft:bricks"])] = M["minecraft:light_blue_wool"]
        r_near, r_far = S.score(ts, near, ts, tm), S.score(ts, far, ts, tm)
        self.assertLess(r_near["material_acc"], 0.1)
        self.assertLess(r_near["material_de_p50"], r_far["material_de_p50"])

    def test_summary(self):
        ts, tm = target()
        rows = [S.score(ts, tm, ts, tm), S.score(np.zeros_like(ts), np.zeros_like(tm), ts, tm)]
        s = S.summarize(rows)
        self.assertEqual(s["builds"], 2)
        self.assertEqual(s["iou"]["mean"], 0.5)
        self.assertEqual(s["shape_acc"]["mean"], 1.0)      # the empty prediction has no shape to judge


class Colours(unittest.TestCase):
    def test_table_matches_the_vocabulary(self):
        lab = S.material_lab()
        self.assertEqual(len(lab), len(V.MATERIALS))
        self.assertTrue(np.isnan(lab[0]).all())
        self.assertTrue(np.isfinite(lab[1:]).all())


if __name__ == "__main__":
    unittest.main()
