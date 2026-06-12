"""SSRF guard (#1727): block server-side fetches to non-public addresses."""
from __future__ import annotations

import pytest

from kestrel_sovereign.security.ssrf import (
    SSRFError,
    validate_outbound_url,
    _address_is_blocked,
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
        with pytest.raises(did_web.DidWebError, match="redirect"):
            did_web._default_fetcher("https://example.com/.well-known/did.json")
