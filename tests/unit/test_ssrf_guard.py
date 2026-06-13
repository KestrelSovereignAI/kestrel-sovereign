"""SSRF guard (#1727): block server-side fetches to non-public addresses."""
from __future__ import annotations

import pytest

from kestrel_sovereign.security.ssrf import (
    SSRFError,
    ValidatedOutboundURL,
    validate_outbound_url,
    _address_is_blocked,
    _PinnedAsyncNetworkBackend,
    _PinnedHTTPSConnection,
)
import ipaddress


class TestAddressClassification:
    @pytest.mark.parametrize("addr", [
        "127.0.0.1", "::1",            # loopback
        "169.254.169.254",            # cloud metadata (link-local)
        "10.0.0.5", "192.168.1.1", "172.16.0.1",  # RFC1918
        "0.0.0.0",                    # unspecified
        "fd00::1",                    # IPv6 ULA (private)
        "::ffff:169.254.169.254",     # IPv4-mapped metadata
    ])
    def test_blocked_addresses(self, addr):
        assert _address_is_blocked(ipaddress.ip_address(addr)) is True

    @pytest.mark.parametrize("addr", ["8.8.8.8", "1.1.1.1", "93.184.216.34"])
    def test_public_addresses_allowed(self, addr):
        assert _address_is_blocked(ipaddress.ip_address(addr)) is False


class TestValidateOutboundUrl:
    def test_literal_metadata_ip_rejected(self):
        with pytest.raises(SSRFError):
            validate_outbound_url("http://169.254.169.254/latest/meta-data/")

    def test_literal_loopback_rejected(self):
        with pytest.raises(SSRFError):
            validate_outbound_url("http://127.0.0.1:8888/api/auth/key")

    def test_literal_private_rejected(self):
        with pytest.raises(SSRFError):
            validate_outbound_url("https://10.1.2.3/internal")

    def test_public_literal_ip_allowed(self):
        validate_outbound_url("https://8.8.8.8/")  # no raise

    def test_disallowed_scheme_rejected(self):
        with pytest.raises(SSRFError):
            validate_outbound_url("file:///etc/passwd")
        with pytest.raises(SSRFError):
            validate_outbound_url("gopher://127.0.0.1/")

    def test_https_only_for_didweb(self):
        with pytest.raises(SSRFError):
            validate_outbound_url("http://example.com/x", allowed_schemes=("https",))

    def test_missing_host_rejected(self):
        with pytest.raises(SSRFError):
            validate_outbound_url("https:///nohost")

    def test_localhost_hostname_rejected(self):
        # Resolves to loopback → blocked.
        with pytest.raises(SSRFError):
            validate_outbound_url("http://localhost:9999/")

    @pytest.mark.parametrize("bad", [
        "http://example.com:abc",       # non-numeric port
        "http://[not-an-ipv6/",         # malformed IPv6 literal
    ])
    def test_malformed_url_raises_ssrferror_not_valueerror(self, bad):
        # #1727 codex r1: malformed input must surface as SSRFError (→ 400),
        # never an uncaught ValueError (→ 500).
        with pytest.raises(SSRFError):
            validate_outbound_url(bad)

    def test_validation_returns_pinned_public_ip(self, monkeypatch):
        def fake_getaddrinfo(host, port, proto=0):
            assert host == "assets.example"
            assert port == 443
            return [
                (None, None, None, None, ("2001:4860:4860::8888", port)),
                (None, None, None, None, ("93.184.216.34", port)),
            ]

        monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
        validated = validate_outbound_url("https://assets.example/avatar.png")

        assert validated.host == "assets.example"
        assert validated.port == 443
        assert [str(ip) for ip in validated.ip_addresses] == [
            "2001:4860:4860::8888",
            "93.184.216.34",
        ]
        assert str(validated.ip_address) == "2001:4860:4860::8888"

    def test_validation_rejects_mixed_public_private_dns_before_connect(
        self,
        monkeypatch,
    ):
        calls = []

        def fake_getaddrinfo(host, port, proto=0):
            return [
                (None, None, None, None, ("93.184.216.34", port)),
                (None, None, None, None, ("127.0.0.1", port)),
            ]

        async def fake_connect_tcp(self, host, port, **kwargs):
            calls.append((host, port))
            return object()

        monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
        monkeypatch.setattr(
            "httpcore._backends.anyio.AnyIOBackend.connect_tcp",
            fake_connect_tcp,
        )

        with pytest.raises(SSRFError):
            validate_outbound_url("https://mixed.example/avatar.png")

        assert calls == []


class TestPinnedConnections:
    @pytest.mark.asyncio
    async def test_httpx_backend_dials_pinned_ip_not_hostname(self, monkeypatch):
        calls = []

        async def fake_connect_tcp(
            self,
            host,
            port,
            timeout=None,
            local_address=None,
            socket_options=None,
        ):
            calls.append((host, port))
            return object()

        monkeypatch.setattr(
            "httpcore._backends.anyio.AnyIOBackend.connect_tcp",
            fake_connect_tcp,
        )
        validated = validate_outbound_url("https://93.184.216.34/avatar.png")
        backend = _PinnedAsyncNetworkBackend(validated)

        await backend.connect_tcp(validated.host, validated.port)

        assert calls == [("93.184.216.34", 443)]

    @pytest.mark.asyncio
    async def test_httpx_backend_falls_back_to_second_validated_ip(
        self,
        monkeypatch,
    ):
        calls = []

        def fake_getaddrinfo(host, port, proto=0):
            assert host == "dualstack.example"
            return [
                (None, None, None, None, ("2001:4860:4860::8888", port)),
                (None, None, None, None, ("93.184.216.34", port)),
            ]

        async def fake_connect_tcp(
            self,
            host,
            port,
            timeout=None,
            local_address=None,
            socket_options=None,
        ):
            calls.append((host, port))
            if host == "2001:4860:4860::8888":
                raise OSError("IPv6 route unavailable")
            return object()

        monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
        monkeypatch.setattr(
            "httpcore._backends.anyio.AnyIOBackend.connect_tcp",
            fake_connect_tcp,
        )
        validated = validate_outbound_url("https://dualstack.example/avatar.png")
        backend = _PinnedAsyncNetworkBackend(validated)

        await backend.connect_tcp(validated.host, validated.port)

        assert calls == [
            ("2001:4860:4860::8888", 443),
            ("93.184.216.34", 443),
        ]

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
