"""schemset.freeze on a tiny fake dataset: tiers, checksums, creator splits, determinism,
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

from schemset import freeze as F  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
