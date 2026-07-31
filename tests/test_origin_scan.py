"""Tests for recovering a build's download origin from Windows' Mark of the Web.

The URL-preference rules are pure logic and run everywhere. The stream-reading tests need
NTFS, so they skip on anything else rather than failing.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import origin_scan  # noqa: E402
import schem_fixtures as fx  # noqa: E402
from schem_meta import sha256_file  # noqa: E402

PMC_STREAM = (
    "[ZoneTransfer]\r\n"
    "ZoneId=3\r\n"
    "ReferrerUrl=https://www.planetminecraft.com/\r\n"
    "HostUrl=https://s3.amazonaws.com/static.planetminecraft.com/files/resource_media/"
    "schematic/1648/house.schematic?X-Amz-Signature=deadbeef\r\n"
)


def _write_stream(path: Path, contents: str) -> bool:
    """Attach a Zone.Identifier stream, or report that this filesystem can't hold one."""
    try:
        with open("{}:Zone.Identifier".format(path), "w", encoding="utf-8") as fh:
            fh.write(contents)
        return True
    except OSError:
        return False


class UrlPreferenceTests(unittest.TestCase):
    """Which of the recorded URLs is actually worth showing the user."""

    def test_a_referrer_with_a_path_wins(self):
        """The page the download started from is the one you can go back to."""
        url = origin_scan.best_origin_url({
            "ReferrerUrl": "https://www.planetminecraft.com/project/my-house/",
            "HostUrl": "https://s3.amazonaws.com/x/house.schematic?X-Amz-Signature=abc",
        })
        self.assertEqual(url, "https://www.planetminecraft.com/project/my-house/")

    def test_a_signed_cdn_link_is_never_chosen_over_nothing_useful(self):
        """An expiring signature pointing at bytes we already have is not a source."""
        url = origin_scan.best_origin_url({
            "HostUrl": "https://s3.amazonaws.com/x/house.schematic?X-Amz-Signature=abc",
        })
        self.assertIsNone(url)

    def test_a_bare_referrer_is_kept_when_that_is_all_there_is(self):
        """PlanetMinecraft hands off to a CDN, so only the site root gets recorded."""
        url = origin_scan.best_origin_url({
            "ReferrerUrl": "https://www.planetminecraft.com/",
            "HostUrl": "https://s3.amazonaws.com/x/h.schematic?X-Amz-Signature=abc",
        })
        self.assertEqual(url, "https://www.planetminecraft.com/")

    def test_a_site_root_is_flagged_as_such(self):
        self.assertFalse(origin_scan._has_path("https://www.planetminecraft.com/"))
        self.assertFalse(origin_scan._has_path("https://www.planetminecraft.com"))
        self.assertTrue(origin_scan._has_path("https://www.planetminecraft.com/project/x/"))

    def test_a_host_url_on_the_real_site_is_usable(self):
        """Some sites serve the file straight off their own domain."""
        url = origin_scan.best_origin_url({
            "HostUrl": "https://minecraftmaps.com/downloads/castle.schematic"})
        self.assertEqual(url, "https://minecraftmaps.com/downloads/castle.schematic")

    def test_an_empty_record_yields_nothing(self):
        self.assertIsNone(origin_scan.best_origin_url({}))
        self.assertIsNone(origin_scan.best_origin_url({"ZoneId": "3"}))


class ZoneIdentifierTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-ads-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_a_file_with_no_stream_reads_as_empty(self):
        plain = self.tmp / "plain.schem"
        plain.write_bytes(b"x")
        self.assertEqual(origin_scan.read_zone_identifier(plain), {})

    def test_a_stream_is_parsed_into_fields(self):
        target = self.tmp / "house.schematic"
        target.write_bytes(b"x")
        if not _write_stream(target, PMC_STREAM):
            self.skipTest("filesystem does not support alternate data streams")
        fields = origin_scan.read_zone_identifier(target)
        self.assertEqual(fields["ZoneId"], "3")
        self.assertEqual(fields["ReferrerUrl"], "https://www.planetminecraft.com/")
        self.assertTrue(fields["HostUrl"].startswith("https://s3.amazonaws.com/"))


class ScanFolderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bdc-scan-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_a_missing_folder_is_reported_not_raised(self):
        report = origin_scan.scan_folder(self.tmp / "nope", {})
        self.assertEqual(report["matched"], [])
        self.assertTrue(report["errors"])

    def test_a_modern_schem_matches_on_its_own_hash(self):
        path = Path(fx.sponge_v2(self.tmp / "modern.schem", 3, 2, 3))
        if not _write_stream(path, "[ZoneTransfer]\r\nReferrerUrl=https://example.com/build/7\r\n"):
            self.skipTest("filesystem does not support alternate data streams")

        report = origin_scan.scan_folder(self.tmp, {sha256_file(path): "build-7"})
        self.assertEqual(len(report["matched"]), 1)
        self.assertEqual(report["matched"][0]["build"], "build-7")
        self.assertEqual(report["matched"][0]["url"], "https://example.com/build/7")
        self.assertFalse(report["matched"][0]["is_site_root"])

    def test_a_legacy_schematic_matches_on_its_converted_hash(self):
        """The library stores the Sponge form, so matching has to convert first."""
        path = Path(fx.legacy_house(self.tmp / "old.schematic"))
        if not _write_stream(path, PMC_STREAM):
            self.skipTest("filesystem does not support alternate data streams")

        # The hash the library would hold for this file.
        import schem_convert
        converted = schem_convert.ensure_sponge(path, self.tmp / "conv.schem").path
        library_digest = sha256_file(converted)
        self.assertNotEqual(library_digest, sha256_file(path),
                            "a legacy file must not match on its raw bytes")

        report = origin_scan.scan_folder(self.tmp, {library_digest: "legacy-build"})
        matched = [m for m in report["matched"] if m["build"] == "legacy-build"]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["url"], "https://www.planetminecraft.com/")
        self.assertTrue(matched[0]["is_site_root"])

    def test_a_file_that_matches_nothing_is_counted_not_reported(self):
        path = Path(fx.sponge_v2(self.tmp / "orphan.schem", 2, 2, 2))
        _write_stream(path, "[ZoneTransfer]\r\nReferrerUrl=https://example.com/x/\r\n")
        report = origin_scan.scan_folder(self.tmp, {"some-other-hash": "x"})
        self.assertEqual(report["matched"], [])
        self.assertEqual(report["scanned"], 1)

    def test_the_scan_leaves_no_conversion_temp_files_behind(self):
        fx.legacy_house(self.tmp / "a.schematic")
        before = sorted(p.name for p in self.tmp.iterdir())
        origin_scan.scan_folder(self.tmp, {})
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), before)

    def test_originals_are_never_modified(self):
        path = Path(fx.legacy_house(self.tmp / "untouched.schematic"))
        digest = sha256_file(path)
        origin_scan.scan_folder(self.tmp, {})
        self.assertEqual(sha256_file(path), digest)


if __name__ == "__main__":
    unittest.main()
