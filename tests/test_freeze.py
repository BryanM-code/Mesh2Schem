"""dataset.freeze on a tiny fake dataset: tiers, checksums, creator splits, determinism,
and the refusal to overwrite a frozen (read-only) source/."""
import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import freeze as F  # noqa: E402

CATEGORIES = ["residential/houses", "civic/churches", "commercial/shops", "industrial/factories"]


def _tag(i: int) -> str:
    """Unrelated-looking text, so names and creators do not look alike by accident."""
    return hashlib.sha1(str(i).encode()).hexdigest()[:8]


def _entry(i: int, creator: str, licence: str = "license unclear", **extra) -> dict:
    cat = CATEGORIES[i % len(CATEGORIES)]
    rel = f"{cat}/b{i}/build{i}.schem"
    e = {"id": f"b{i}", "file": rel, "file_type": ".schem", "building_category": cat,
         "creator": creator, "nbt_author_or_exporter": None, "source_website": "Test",
         "license_status": licence, "dimensions_xyz_blocks": [10 + i, 8, 20 + 2 * i],
         "non_air_blocks_from_metadata": 500 + 37 * i, "source_url": f"https://x/{i}",
         "build_name": f"Build {_tag(i)}"}
    e.update(extra)
    return e


def make_dataset(root: Path, entries: list[dict]) -> None:
    """Write each entry's file, a manifest with real sha256s and a checksums.sha256."""
    lines = []
    for e in entries:
        path = root / e["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        data = f"fake schematic {e['id']}".encode()
        path.write_bytes(data)
        e["sha256"] = hashlib.sha256(data).hexdigest()
        lines.append(f"{e['sha256']}  {e['file']}\r\n")
    (root / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    (root / "checksums.sha256").write_text("".join(lines), encoding="utf-8", newline="")
    (root / "README.md").write_text("fake", encoding="utf-8")


def standard_entries() -> list[dict]:
    """40 builds: a few multi-build creators, one credit-suffix alias, singletons, one X."""
    creators = (["Alpha"] * 5 + ["Beta"] * 4 + ["Gamma"] * 3 + ["Gamma; additional credit Zed"]
                + [f"Solo{_tag(100 + i)}" for i in range(27)])
    entries = [_entry(i, c) for i, c in enumerate(creators)]
    entries[1]["license_status"] = "stated reuse permission; scope limited"
    entries[2]["license_status"] = "explicit license: CC BY-NC-SA 4.0"
    entries[-1]["license_status"] = "explicit license: CC BY-NC-ND 4.0"
    return entries


class FreezeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(self._cleanup)
        self.src = self.tmp / "dataset"
        self.out = self.tmp / "out"

    def _cleanup(self):
        F.set_readonly(self.tmp, False)
        shutil.rmtree(self.tmp, ignore_errors=True)


class TierTests(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(F.licence_tier("stated reuse permission; scope limited"), "A")
        self.assertEqual(F.licence_tier("license unclear"), "B")
        for s in ("restricted personal-use license", "explicit noncommercial reuse permission",
                  "explicit license: CC BY-NC-SA 4.0"):
            self.assertEqual(F.licence_tier(s), "C")
        for s in ("explicit license: CC BY-NC-ND 4.0",
                  "explicit terms: use with credit, no redistribution"):
            self.assertEqual(F.licence_tier(s), "X")

    def test_unknown_value_raises(self):
        with self.assertRaises(F.UnknownLicence):
            F.licence_tier("CC BY 4.0")
        with self.assertRaises(F.UnknownLicence):
            F.licence_tier("License unclear")           # exact strings only
        entries = [_entry(0, "A"), _entry(1, "B", licence="public domain")]
        with self.assertRaisesRegex(F.UnknownLicence, "public domain"):
            F.build_rows(entries)

    def test_flags(self):
        entries = [_entry(0, "A", licence="explicit license: CC BY-NC-ND 4.0",
                          listing_completion_percent="60"),
                   _entry(1, "B", file_type=".schematic", listing_completion_percent="100",
                          dimensions_xyz_blocks=[201, 10, 5]),
                   _entry(2, "C", dimensions_xyz_blocks=[5, 300, 200])]
        flags = {r["id"]: r["flags"] for r in F.build_rows(entries)}
        self.assertEqual(flags["b0"], ["tier_x", "incomplete_listing"])
        self.assertEqual(flags["b1"], ["oversized_capture", "legacy_format"])
        self.assertEqual(flags["b2"], [])


class ChecksumTests(FreezeCase):
    def test_clean_freeze(self):
        make_dataset(self.src, standard_entries())
        r = F.freeze(self.src, self.out, today="2026-10-11")
        self.assertEqual(r["files"], 43)
        listed = F.parse_checksums((self.out / "freeze_checksums.sha256").read_text("utf-8"))
        self.assertEqual(len(listed), 43)
        self.assertIn("2026-10-11", (self.out / "FREEZE.md").read_text("utf-8"))
        self.assertEqual(F.verify_frozen(self.out), 43)

    def test_checksum_line_mismatch_stops(self):
        entries = standard_entries()
        make_dataset(self.src, entries)
        text = (self.src / "checksums.sha256").read_text("utf-8")
        (self.src / "checksums.sha256").write_text(
            text.replace(entries[3]["sha256"], "0" * 64), encoding="utf-8", newline="")
        with self.assertRaisesRegex(F.VerificationError, "checksums.sha256"):
            F.freeze(self.src, self.out)
        self.assertFalse((self.out / "source").exists())

    def test_manifest_sha_mismatch_stops(self):
        entries = standard_entries()
        make_dataset(self.src, entries)
        (self.src / entries[5]["file"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(F.VerificationError, "manifest b5"):
            F.freeze(self.src, self.out)
        self.assertFalse((self.out / "source").exists())

    def test_frozen_copy_change_detected(self):
        entries = standard_entries()
        make_dataset(self.src, entries)
        F.freeze(self.src, self.out)
        target = self.out / "source" / "README.md"
        F.set_readonly(self.out / "source", False)
        target.write_text("edited", encoding="utf-8")
        with self.assertRaises(F.VerificationError):
            F.verify_frozen(self.out)


class SplitTests(FreezeCase):
    def setUp(self):
        super().setUp()
        make_dataset(self.src, standard_entries())
        F.freeze(self.src, self.out)

    def test_creator_grouping(self):
        rows = F.derive(self.out)["rows"]
        by_creator = {}
        for r in rows:
            by_creator.setdefault(r["creator_norm"], set()).add(r["split"])
        self.assertTrue(all(len(s) == 1 for s in by_creator.values()), by_creator)
        # "Gamma; additional credit Zed" is a different creator_norm but the same creator.
        gamma = {r["split"] for r in rows if r["creator"].startswith("Gamma")}
        self.assertEqual(len(gamma), 1)
        self.assertEqual({r["split"] for r in rows if r["licence_tier"] == "X"}, {"excluded"})
        used = {r["split"] for r in rows if r["licence_tier"] != "X"}
        self.assertEqual(used, {"train", "val", "test"})

    def test_near_dup_candidates_share_a_split(self):
        rows = F.build_rows(standard_entries())
        a, b = rows[15], rows[25]                     # two different solo creators
        x, y, z = a["dimensions_xyz_blocks"]
        b["dimensions_xyz_blocks"] = [z, y, x]        # the same build turned 90 degrees
        b["non_air_blocks_from_metadata"] = a["non_air_blocks_from_metadata"] + 1
        near = F.find_near_duplicates(rows)
        self.assertEqual([c["ids"] for c in near["candidates"]], [[a["id"], b["id"]]])
        self.assertEqual(near["candidates"][0]["rule"], "cross_creator_dims_nonair")
        links = [(F.creator_node(a), F.creator_node(b))]
        for seed in range(5):
            assignment, _ = F.assign_splits(rows, links, seed=seed)
            self.assertEqual(assignment[a["id"]], assignment[b["id"]])

    def test_determinism(self):
        F.derive(self.out)
        first = {n: (self.out / n).read_bytes()
                 for n in ("index.json", "splits.json", "near_duplicates.json")}
        F.derive(self.out)
        for name, data in first.items():
            self.assertEqual((self.out / name).read_bytes(), data, name)
        a, _ = F.assign_splits(F.build_rows(standard_entries()), [], seed=7)
        b, _ = F.assign_splits(F.build_rows(standard_entries()), [], seed=7)
        self.assertEqual(a, b)
        recorded = json.loads((self.out / "splits.json").read_text("utf-8"))
        self.assertEqual(recorded["seed"], F.SEED)


class ReadOnlyTests(FreezeCase):
    def test_refuses_to_overwrite_frozen_source(self):
        make_dataset(self.src, standard_entries())
        F.freeze(self.src, self.out)
        manifest = self.out / "source" / "manifest.json"
        self.assertTrue(F.is_readonly(manifest))
        before = manifest.read_bytes()
        (self.src / "README.md").write_text("changed after the freeze", encoding="utf-8")

        with self.assertRaises(F.SourceExists):
            F.freeze(self.src, self.out)
        # The CLI without --force re-verifies and regenerates derived files only.
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(F.main(["--source", str(self.src), "--out", str(self.out)]), 0)
        self.assertIn("not overwritten", log.getvalue())
        self.assertEqual(manifest.read_bytes(), before)
        self.assertEqual((self.out / "source" / "README.md").read_text("utf-8"), "fake")
        self.assertTrue((self.out / "index.json").is_file())

        F.freeze(self.src, self.out, force=True)
        self.assertEqual((self.out / "source" / "README.md").read_text("utf-8"),
                         "changed after the freeze")
        self.assertTrue(F.is_readonly(self.out / "source" / "README.md"))


class CombinedTests(FreezeCase):
    """`derive(out, base=...)`: a new collection added to a frozen dataset whose splits stay."""

    def setUp(self):
        super().setUp()
        make_dataset(self.src, standard_entries())
        self.base = self.tmp / "base-v1"
        F.freeze(self.src, self.base)
        self.base_rows = {r["id"]: r for r in F.derive(self.base)["rows"]}
        self.split_of = {}
        for r in self.base_rows.values():
            self.split_of.setdefault(r["creator"], r["split"])

    def _creator_in(self, split: str) -> str:
        return next(c for c, s in sorted(self.split_of.items())
                    if s == split and not c.startswith("Gamma"))

    def _new(self, entries: list[dict]) -> Path:
        src, out = self.tmp / "new", self.tmp / "combined"
        make_dataset(src, entries)
        F.freeze(src, out)
        return out

    def test_base_unchanged_new_builds_follow_their_creator(self):
        val_creator = self._creator_in("val")
        entries = [_entry(200 + i, f"New{_tag(300 + i)}") for i in range(6)]
        entries.append(_entry(220, val_creator))                       # a base val creator
        entries.append(_entry(221, "Other", licence="explicit license: CC BY-NC-ND 4.0"))
        entries.append(_entry(222, "Lic", licence="CC BY 4.0 (stated in listing)"))
        out = self._new(entries)
        d = F.derive(out, base=self.base)
        rows = {r["id"]: r for r in d["rows"]}
        for i, r in self.base_rows.items():                          # base rows: as they were
            self.assertEqual(rows[i]["split"], r["split"], i)
            self.assertEqual(rows[i]["origin"], "base-v1")
            self.assertEqual(rows[i]["source_root"], "../base-v1/source")
        self.assertTrue((out / rows["b0"]["source_root"] / rows["b0"]["file"]).is_file())
        for i in range(6):
            self.assertEqual(rows[f"b{200 + i}"]["split"], "train")
            self.assertEqual(rows[f"b{200 + i}"]["source_root"], "source")
        self.assertEqual(rows["b220"]["split"], "val")
        self.assertEqual(rows["b221"]["split"], "excluded")
        self.assertEqual(rows["b222"]["licence_tier"], "A")
        index = json.loads((out / "index.json").read_text("utf-8"))
        self.assertEqual((index["dataset"], index["base"]), ("combined", "base-v1"))

    def test_id_clash_stops(self):
        out = self._new([_entry(3, "Somebody")])                     # b3 is a base id
        with self.assertRaisesRegex(F.FreezeError, "both collections"):
            F.derive(out, base=self.base)

    def test_bridge_between_base_splits_is_excluded(self):
        a, b = self._creator_in("train"), self._creator_in("test")
        twin = next(r for r in self.base_rows.values() if r["creator"] == b)
        e = _entry(230, f"{a}; with credits")                        # linked to a (credit suffix)
        e["dimensions_xyz_blocks"] = list(twin["dimensions_xyz_blocks"])   # ...and a near-dup of
        e["non_air_blocks_from_metadata"] = twin["non_air_blocks_from_metadata"]   # b's build
        d = F.derive(self._new([e, _entry(231, "Loner")]), base=self.base)
        rows = {r["id"]: r for r in d["rows"]}
        self.assertEqual(rows["b230"]["split"], "excluded")
        self.assertIn("base_split_crossing", rows["b230"]["flags"])
        self.assertEqual(rows["b231"]["split"], "train")
        self.assertEqual(rows[twin["id"]]["split"], "test")

    def test_duplicate_of_a_base_eval_build_is_excluded(self):
        test_row = next(r for r in self.base_rows.values() if r["split"] == "test")
        out = self._new([_entry(240, "Copier"), _entry(241, "Fine")])
        (out / F.GEOMETRIC_DUPLICATES).write_text(json.dumps({"pairs": [
            {"ids": [test_row["id"], "b240"], "verdict": "duplicate"}]}), encoding="utf-8")
        d = F.derive(out, base=self.base)
        rows = {r["id"]: r for r in d["rows"]}
        self.assertEqual(rows["b240"]["split"], "excluded")
        self.assertIn("duplicate_of_base", rows["b240"]["flags"])
        self.assertEqual(rows[test_row["id"]]["split"], "test")
        self.assertEqual(rows["b241"]["split"], "train")


if __name__ == "__main__":
    unittest.main()
