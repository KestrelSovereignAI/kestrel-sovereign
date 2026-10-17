"""DID and stdlib HTTPS callers use the shared SDK outbound policy."""
from __future__ import annotations

import ipaddress
import pytest
from unittest.mock import MagicMock

from kestrel_sdk.security.outbound import SSRFError, ValidatedOutboundURL
from kestrel_sovereign.security.pinned_urllib import _PinnedHTTPSConnection


class TestPinnedUrllibConnections:
    def test_unexpected_tls_error_is_not_retried_and_releases_the_socket(self):
        validated = ValidatedOutboundURL(
            url="https://example.test/did.json", scheme="https", host="example.test", port=443,
            ip_addresses=(ipaddress.ip_address("8.8.8.8"), ipaddress.ip_address("1.1.1.1")),
        )
        conn = _PinnedHTTPSConnection(validated.host, validated=validated)
        sock = MagicMock()
        conn._create_connection = MagicMock(return_value=sock)
        conn._context = MagicMock()
        conn._context.wrap_socket.side_effect = [TypeError("invalid TLS configuration"), MagicMock()]
        with pytest.raises(TypeError, match="invalid TLS configuration"):
            conn.connect()
        conn._create_connection.assert_called_once()
        sock.close.assert_called_once()
        assert conn.sock is None

    def test_all_network_failures_close_sockets_and_report_the_last_failure(self):
        validated = ValidatedOutboundURL(
            url="https://example.test/did.json", scheme="https", host="example.test", port=443,
            ip_addresses=(ipaddress.ip_address("8.8.8.8"), ipaddress.ip_address("1.1.1.1")),
        )
        conn = _PinnedHTTPSConnection(validated.host, validated=validated)
        sockets = [MagicMock(), MagicMock()]
        conn._create_connection = MagicMock(side_effect=sockets)
        conn._context = MagicMock()
        conn._context.wrap_socket.side_effect = [OSError("first unavailable"), OSError("second unavailable")]
        with pytest.raises(OSError, match="second unavailable"):
            conn.connect()
        assert conn._create_connection.call_count == 2
        for sock in sockets:
            sock.close.assert_called_once()
        assert conn.sock is None

    def test_changed_origin_is_refused_before_socket_creation(self):
        validated = ValidatedOutboundURL(
            url="https://example.test/did.json", scheme="https", host="example.test", port=443,
            ip_addresses=(ipaddress.ip_address("8.8.8.8"),),
        )
        conn = _PinnedHTTPSConnection("other.test", validated=validated)
        conn._create_connection = MagicMock()
        with pytest.raises(SSRFError, match="unexpected origin"):
            conn.connect()
        conn._create_connection.assert_not_called()

    def test_urllib_connection_dials_pinned_ip_with_original_sni(self):
        validated = ValidatedOutboundURL(
            url="https://example.test/.well-known/did.json",
            scheme="https",
            host="example.test",
            port=443,
            ip_addresses=(ipaddress.ip_address("93.184.216.34"),),
        )
        conn = _PinnedHTTPSConnection("example.test", validated=validated)
        dialed = []
        wrapped = []

        class FakeSocket:
            def setsockopt(self, *args):
                pass

            def close(self):
                pass

        class FakeContext:
            def wrap_socket(self, sock, server_hostname=None):
                wrapped.append(server_hostname)
                return sock

        def fake_create_connection(address, timeout, source_address):
            dialed.append(address)
            return FakeSocket()

        conn._create_connection = fake_create_connection
        conn._context = FakeContext()

        conn.connect()

        assert dialed == [("93.184.216.34", 443)]
        assert wrapped == ["example.test"]

    def test_urllib_connection_accepts_punycode_host_for_validated_idn(self):
        validated = ValidatedOutboundURL(
            url="https://bücher.example/.well-known/did.json",
            scheme="https",
            host="bücher.example",
            port=443,
            ip_addresses=(ipaddress.ip_address("93.184.216.34"),),
        )
        conn = _PinnedHTTPSConnection("xn--bcher-kva.example", validated=validated)
        dialed = []
        wrapped = []

        class FakeSocket:
            def setsockopt(self, *args):
                pass

            def close(self):
                pass

        class FakeContext:
            def wrap_socket(self, sock, server_hostname=None):
                wrapped.append(server_hostname)
                return sock

        def fake_create_connection(address, timeout, source_address):
            dialed.append(address)
            return FakeSocket()

        conn._create_connection = fake_create_connection
        conn._context = FakeContext()

        conn.connect()

        assert dialed == [("93.184.216.34", 443)]
        assert wrapped == ["xn--bcher-kva.example"]

    def test_urllib_connection_falls_back_to_second_validated_ip(self):
        validated = ValidatedOutboundURL(
            url="https://example.test/.well-known/did.json",
            scheme="https",
            host="example.test",
            port=443,
            ip_addresses=(
                ipaddress.ip_address("2001:4860:4860::8888"),
                ipaddress.ip_address("93.184.216.34"),
            ),
        )
        conn = _PinnedHTTPSConnection("example.test", validated=validated)
        dialed = []
        wrapped = []

        class FakeSocket:
            def setsockopt(self, *args):
                pass

            def close(self):
                pass

        class FakeContext:
            def wrap_socket(self, sock, server_hostname=None):
                wrapped.append(server_hostname)
                return sock

        def fake_create_connection(address, timeout, source_address):
            dialed.append(address)
            if address[0] == "2001:4860:4860::8888":
                raise OSError("IPv6 route unavailable")
            return FakeSocket()

        conn._create_connection = fake_create_connection
        conn._context = FakeContext()

        conn.connect()

        assert dialed == [
            ("2001:4860:4860::8888", 443),
            ("93.184.216.34", 443),
        ]
        assert wrapped == ["example.test"]


class TestDidWebSSRF:
    def test_didweb_fetcher_refuses_metadata_host(self):
        from kestrel_sovereign.identity.did_web import _default_fetcher, DidWebError
        # did:web:169.254.169.254 → https://169.254.169.254/.well-known/did.json
        with pytest.raises(DidWebError):
            _default_fetcher("https://169.254.169.254/.well-known/did.json")

    def test_didweb_fetcher_does_not_follow_redirects(self, monkeypatch):
        """#1727 codex r1: a public DID host that 30x-redirects to a private
        address must be refused, not followed."""
        from kestrel_sovereign.identity import did_web
        from urllib.error import HTTPError
        import io

        def fake_open(self, url, timeout=None):
            # Simulate a 302 → urllib (no-redirect opener) raising HTTPError.
            raise HTTPError(url, 302, "Found", {"Location": "https://169.254.169.254/x"}, io.BytesIO(b""))

        monkeypatch.setattr("urllib.request.OpenerDirector.open", fake_open)
        # Public LITERAL IP so validate_outbound_url passes WITHOUT real DNS
        # (keeps the unit test offline-safe); the redirect is what's under test.
        with pytest.raises(did_web.DidWebError, match="redirect"):
            did_web._default_fetcher("https://93.184.216.34/.well-known/did.json")
