"""The vocabulary on real builds: every exact block state in real-schems-v1 round-trips through
encode / decode, schem_write validates target files against 1.21.11, and the reader gives
renamed blocks their current names."""
import glob
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voxel import vocab as V

RECORDS = ROOT / "data" / "real-schems-v1" / "normalized" / "records"


def predicted(state):
    """(block id, the properties v3 predicts); derived ones are left out."""
    c = V.classify_state(state)
    name, props = state.split("[", 1)[0], V._props(state)
    kind = c["kind"]
    out = {k: props.get(k) for k in {"slab": ("type",), "stairs": ("facing", "half"),
                                     "door": ("facing", "half")}.get(kind, ())}
    if kind == "full" and name in V.AXIS_MATERIALS:
        out["axis"] = props.get("axis", "y")
    if kind == "trapdoor":
        out["open"] = props.get("open", "false")
        if out["open"] == "true":
            out["facing"] = props.get("facing", "north")
        else:
            out["half"] = props.get("half", "bottom")
    if kind == "gate":
        out["axis"] = "x" if props.get("facing", "north") in ("north", "south") else "z"
    return name, out


class DatasetRoundTrip(unittest.TestCase):
    @unittest.skipUnless(RECORDS.is_dir(), "real-schems-v1 is not normalized on this machine")
    def test_every_exact_state_in_the_inventory_round_trips(self):
        states = set()
        for path in glob.glob(str(RECORDS / "*.json")):
            states.update(json.loads(Path(path).read_text(encoding="utf-8"))["supervised"])
        exact = 0
        for state in sorted(states):
            c = V.classify_state(state)
            if c["status"] != "exact":
                continue
            exact += 1
            back = V.decode(c["shape"], c["material"], strict=True)
            V.validate_state(back, V.TARGET_DATA_VERSION)
            self.assertEqual(predicted(back), predicted(state), state)
            self.assertEqual(V.encode_state(back), (c["shape"], c["material"]), state)
        self.assertGreater(exact, 1000)

    def test_samples_without_the_dataset(self):
        for state in ("minecraft:pale_oak_trapdoor[facing=west,half=bottom,open=true,powered=false,waterlogged=false]",
                      "minecraft:deepslate_tile_stairs[facing=north,half=top,shape=inner_left,waterlogged=false]",
                      "minecraft:magenta_stained_glass_pane[east=true,north=false,south=false,waterlogged=false,west=true]",
                      "minecraft:stripped_cherry_log[axis=z]"):
            c = V.classify_state(state)
            self.assertEqual(c["status"], "exact", state)
            self.assertEqual(predicted(V.decode(c["shape"], c["material"], strict=True)), predicted(state))


class SchemWriteValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_target_version_files_are_validated(self):
        from minecraft.schem_write import WriteError, write_schem
        grid = np.array([[[0, 1, 2]]], dtype=np.uint16)          # (y, z, x)
        good = [V.AIR, V.decode(V.SHAPE_INDEX["door_north"], V.MATERIAL_INDEX["minecraft:pale_oak_planks"]),
                "minecraft:copper_bars[east=false,north=false,south=false,waterlogged=false,west=false]"]
        write_schem(grid, good, self.tmp / "ok.schem", data_version=V.TARGET_DATA_VERSION)
        bad = [V.AIR, "minecraft:chain", "minecraft:oak_stairs[facing=up]"]
        with self.assertRaises(WriteError) as ctx:
            write_schem(grid, bad, self.tmp / "bad.schem", data_version=V.TARGET_DATA_VERSION)
        self.assertIn("2 blockstate(s)", str(ctx.exception))
        self.assertFalse((self.tmp / "bad.schem").exists())
        write_schem(grid, bad, self.tmp / "legacy.schem")          # 1.13 default: not checked
        write_schem(grid, bad, self.tmp / "off.schem", data_version=V.TARGET_DATA_VERSION,
                    validate=False)


class ReaderRenames(unittest.TestCase):
    def test_current_names(self):
        from dataset.read import modern_state
        self.assertEqual(modern_state("minecraft:chain[axis=y,waterlogged=false]", 3465),
                         "minecraft:iron_chain[axis=y,waterlogged=false]")


if __name__ == "__main__":
    unittest.main()
