"""Tests for where the server binds, and who that lets in.

This app has no accounts and no password, so the interface it listens on *is* its access
control. Binding is therefore worth testing properly: the default must stay local, the
Tailscale path must only ever produce a Tailscale address, and anything wider has to be
something you asked for explicitly.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db  # noqa: E402

SANDBOX = Path(tempfile.mkdtemp(prefix="bdc-net-tests-"))
db.PROJECT_ROOT = db.DATA_DIR = SANDBOX
db.DB_PATH = SANDBOX / "collector.db"
db.BUILDS_DIR = SANDBOX / "builds"

import server  # noqa: E402


def tearDownModule():
    shutil.rmtree(SANDBOX, ignore_errors=True)


class TailscaleRangeTests(unittest.TestCase):
    """Tailscale allocates out of 100.64.0.0/10 -- the boundaries are the interesting part."""

    def test_addresses_inside_the_range(self):
        for address in ("100.64.0.0", "100.64.0.1", "100.101.5.9", "100.127.255.255"):
            with self.subTest(address):
                self.assertTrue(server._is_tailscale_ip(address))

    def test_addresses_outside_the_range(self):
        for address in ("100.63.255.255", "100.128.0.0", "10.0.0.1", "192.168.1.5",
                        "100.64.0", "", "not-an-ip", "100.64.0.1.5"):
            with self.subTest(address):
                self.assertFalse(server._is_tailscale_ip(address))

    def test_a_public_100_address_is_not_mistaken_for_tailscale(self):
        """100.0.0.0/8 is public space; only 100.64/10 is the CGNAT block Tailscale uses."""
        self.assertFalse(server._is_tailscale_ip("100.1.2.3"))


class ApiVersionTests(unittest.TestCase):
    """The staleness banner only works if both halves agree on the number.

    They live in different files and are bumped by hand, so they drift silently: a server
    bumped without the page means `server > expected`, the comparison is false, and the
    banner never fires no matter how stale the server actually is. That is exactly what
    happened between v3 and v5, and it is what this test exists to catch.
    """

    def frontend_expectation(self) -> int:
        source = (Path(__file__).resolve().parent.parent / "static" / "app.js").read_text(
            encoding="utf-8")
        match = re.search(r"const EXPECTED_API_VERSION\s*=\s*(\d+)", source)
        self.assertIsNotNone(match, "app.js should declare EXPECTED_API_VERSION")
        return int(match.group(1))

    def test_the_page_and_the_server_agree_on_the_api_version(self):
        self.assertEqual(
            self.frontend_expectation(), server.API_VERSION,
            "static/app.js EXPECTED_API_VERSION and server.API_VERSION must match; "
            "bump both together or the stale-server banner stops working")


class PortHolderTests(unittest.TestCase):
    """Starting a second copy over a running one is this app's most misleading failure.

    Python reads modules once, so an old process serves old code forever. The API version
    banner can't catch it -- a fix behind an unchanged endpoint doesn't change the version
    -- so the page looks perfect while rejecting a file the fixed code accepts. Detecting
    the busy port up front is the only thing that makes it visible.
    """

    def test_a_free_port_reports_nobody(self):
        import socket as sk
        probe = sk.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        self.assertIsNone(server.port_holder("127.0.0.1", port))

    def test_a_busy_port_is_detected(self):
        import socket as sk
        held = sk.socket()
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        self.addCleanup(held.close)
        port = held.getsockname()[1]
        self.assertIsNotNone(server.port_holder("127.0.0.1", port))

    def test_a_stranger_on_the_port_is_not_called_our_own_server(self):
        """The advice differs -- stop the old copy, versus pick another port."""
        import socket as sk
        held = sk.socket()
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        self.addCleanup(held.close)
        holder = server.port_holder("127.0.0.1", held.getsockname()[1])
        self.assertEqual(holder, "another program")

    def test_our_own_server_is_recognised_by_its_meta_endpoint(self):
        import socket as sk
        held = sk.socket()
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        self.addCleanup(held.close)
        port = held.getsockname()[1]

        class FakeResponse:
            def read(self_inner, *a):
                return b'{"api_version": 6}'
            def __enter__(self_inner):
                return io.BytesIO(b'{"api_version": 6}')
            def __exit__(self_inner, *a):
                return False

        with mock.patch("urllib.request.urlopen", return_value=FakeResponse()):
            holder = server.port_holder("127.0.0.1", port)
        self.assertIn("older copy", holder)
        self.assertIn("6", holder)


class HostResolutionTests(unittest.TestCase):
    def test_the_default_is_this_machine_only(self):
        for requested in ("", None, "127.0.0.1", "localhost"):
            with self.subTest(requested):
                host, note = server.resolve_host(requested)
                self.assertEqual(host, "127.0.0.1")
                self.assertIn("this machine only", note)

    def test_tailscale_resolves_to_the_detected_address(self):
        with mock.patch.object(server, "find_tailscale_ip", return_value="100.101.5.9"):
            host, note = server.resolve_host("tailscale")
        self.assertEqual(host, "100.101.5.9")
        self.assertIn("Tailscale", note)

    def test_tailscale_never_silently_falls_back_to_something_wider(self):
        """The dangerous failure would be quietly binding 0.0.0.0 when Tailscale is down."""
        with mock.patch.object(server, "find_tailscale_ip", return_value=None):
            with self.assertRaises(SystemExit) as caught:
                server.resolve_host("tailscale")
        message = str(caught.exception)
        self.assertIn("tailscale ip -4", message, "should say how to diagnose it")
        self.assertNotIn("0.0.0.0", message.split("Or bind explicitly")[0])

    def test_binding_everywhere_is_possible_but_labelled(self):
        host, note = server.resolve_host("0.0.0.0")
        self.assertEqual(host, "0.0.0.0")
        self.assertIn("EVERY", note)

    def test_an_explicit_tailscale_address_is_recognised(self):
        host, note = server.resolve_host("100.101.5.9")
        self.assertEqual(host, "100.101.5.9")
        self.assertIn("Tailscale", note)

    def test_any_other_address_is_passed_through(self):
        host, _ = server.resolve_host("192.168.1.50")
        self.assertEqual(host, "192.168.1.50")

    def test_case_and_whitespace_are_tolerated(self):
        with mock.patch.object(server, "find_tailscale_ip", return_value="100.70.1.1"):
            self.assertEqual(server.resolve_host("  TailScale  ")[0], "100.70.1.1")


class TailscaleDetectionTests(unittest.TestCase):
    def test_the_cli_answer_is_used_when_it_looks_right(self):
        completed = mock.Mock(stdout="100.101.5.9\nfd7a:115c:a1e0::1\n")
        with mock.patch.object(server.subprocess, "run", return_value=completed):
            self.assertEqual(server.find_tailscale_ip(), "100.101.5.9")

    def test_a_missing_cli_falls_through_rather_than_raising(self):
        with mock.patch.object(server.subprocess, "run", side_effect=OSError):
            with mock.patch.object(server.socket, "getaddrinfo", return_value=[]):
                self.assertIsNone(server.find_tailscale_ip())

    def test_the_address_scan_is_the_fallback(self):
        addrinfo = [(None, None, None, None, ("192.168.1.5", 0)),
                    (None, None, None, None, ("100.90.1.2", 0))]
        with mock.patch.object(server.subprocess, "run", side_effect=OSError):
            with mock.patch.object(server.socket, "getaddrinfo", return_value=addrinfo):
                self.assertEqual(server.find_tailscale_ip(), "100.90.1.2")

    def test_cli_output_that_is_not_a_tailscale_address_is_ignored(self):
        """A rogue `tailscale` on PATH must not talk us into binding somewhere odd."""
        completed = mock.Mock(stdout="192.168.1.5\n")
        with mock.patch.object(server.subprocess, "run", return_value=completed):
            with mock.patch.object(server.socket, "getaddrinfo", return_value=[]):
                self.assertIsNone(server.find_tailscale_ip())


class ExposureWarningTests(unittest.TestCase):
    """The banner is the only thing that tells you the library is reachable by others."""

    def banner(self, host, note="x", port=5000):
        from io import StringIO
        buffer = StringIO()
        with mock.patch("sys.stdout", buffer):
            with mock.patch.object(server, "find_tailscale_ip", return_value=None):
                server._startup_banner(host, note, port)
        return buffer.getvalue()

    def test_a_local_bind_gets_no_warning(self):
        self.assertNotIn("no login", self.banner("127.0.0.1"))

    def test_a_tailscale_bind_warns_but_reassures(self):
        text = self.banner("100.101.5.9")
        self.assertIn("no login", text)
        self.assertIn("Tailscale", text)

    def test_binding_everywhere_says_prefer_tailscale(self):
        text = self.banner("0.0.0.0")
        self.assertIn("no login", text)
        self.assertIn("BDC_HOST=tailscale", text)

    def test_the_banner_shows_a_url_you_can_actually_click(self):
        """0.0.0.0 is not a destination, so the printed URL must not be one."""
        self.assertIn("http://127.0.0.1:5000", self.banner("0.0.0.0"))


if __name__ == "__main__":
    unittest.main()
