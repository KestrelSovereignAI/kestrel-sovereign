"""DID and stdlib HTTPS callers use the shared SDK outbound policy."""
from __future__ import annotations

import ipaddress
from http.client import RemoteDisconnected
import socket
import ssl
import pytest
from unittest.mock import MagicMock
from urllib.error import URLError

from kestrel_sdk.security.outbound import SSRFError, ValidatedOutboundURL
from kestrel_sovereign.security.pinned_urllib import _PinnedHTTPSConnection


class TestPinnedUrllibConnections:
    def test_certificate_failure_does_not_retry_a_second_address(self):
        validated = ValidatedOutboundURL(
            url="https://example.test/did.json", scheme="https", host="example.test", port=443,
            ip_addresses=(ipaddress.ip_address("8.8.8.8"), ipaddress.ip_address("1.1.1.1")),
        )
        conn = _PinnedHTTPSConnection(validated.host, validated=validated, timeout=10)
        first, second = MagicMock(spec=socket.socket), MagicMock(spec=socket.socket)
        conn._create_connection = MagicMock(side_effect=[first, second])
        conn._context = MagicMock()
        conn._context.wrap_socket.side_effect = [ssl.SSLCertVerificationError("bad certificate"), second]
        with pytest.raises(ssl.SSLCertVerificationError, match="certificate"):
            conn.connect()
        conn._create_connection.assert_called_once()
        first.close.assert_called_once()
        second.close.assert_not_called()
        assert conn.sock is None

    def test_all_addresses_share_one_connection_deadline(self, monkeypatch):
        clock = [0.0]
        monkeypatch.setattr("time.monotonic", lambda: clock[0])
        validated = ValidatedOutboundURL(
            url="https://example.test/did.json", scheme="https", host="example.test", port=443,
            ip_addresses=(ipaddress.ip_address("8.8.8.8"), ipaddress.ip_address("1.1.1.1")),
        )
        conn = _PinnedHTTPSConnection(validated.host, validated=validated, timeout=0.5)
        dials = []

        def stalled(address, timeout, source_address):
            dials.append((address, timeout))
            clock[0] += timeout
            raise TimeoutError("address stalled")

        conn._create_connection = stalled
        with pytest.raises(TimeoutError) as error:
            conn.connect()
        assert len(dials) == 1
        assert "deadline" in str(error.value)
        assert conn.sock is None

    def test_tls_uses_remaining_budget_then_restores_the_read_timeout(self, monkeypatch):
        clock = [0.0]
        monkeypatch.setattr("time.monotonic", lambda: clock[0])
        validated = ValidatedOutboundURL(
            url="https://example.test/did.json", scheme="https", host="example.test", port=443,
            ip_addresses=(ipaddress.ip_address("8.8.8.8"),),
        )
        conn = _PinnedHTTPSConnection(validated.host, validated=validated, timeout=10)
        raw, wrapped = MagicMock(spec=socket.socket), MagicMock(spec=socket.socket)

        def dial(address, timeout, source_address):
            assert timeout == 10
            clock[0] += 3
            return raw

        def tls(sock, server_hostname):
            assert sock is raw and server_hostname == validated.host
            raw.settimeout.assert_called_once_with(7)
            clock[0] += 2
            return wrapped

        conn._create_connection = dial
        conn._context = MagicMock()
        conn._context.wrap_socket.side_effect = tls
        conn.connect()
        assert conn.sock is wrapped
        wrapped.settimeout.assert_called_once_with(10)
        raw.close.assert_not_called()
        wrapped.close.assert_not_called()

    @pytest.mark.parametrize("handshake_elapsed", [10, 10.25])
    def test_tls_finishing_at_the_deadline_closes_both_unpublished_sockets(self, monkeypatch, handshake_elapsed):
        clock = [0.0]
        monkeypatch.setattr("time.monotonic", lambda: clock[0])
        validated = ValidatedOutboundURL(
            url="https://example.test/did.json", scheme="https", host="example.test", port=443,
            ip_addresses=(ipaddress.ip_address("8.8.8.8"),),
        )
        conn = _PinnedHTTPSConnection(validated.host, validated=validated, timeout=10)
        raw, wrapped = MagicMock(spec=socket.socket), MagicMock(spec=socket.socket)
        conn._create_connection = MagicMock(return_value=raw)
        conn._context = MagicMock()

        def handshake(sock, server_hostname):
            assert sock is raw and server_hostname == validated.host
            raw.settimeout.assert_called_once_with(10)
            clock[0] += handshake_elapsed
            return wrapped

        conn._context.wrap_socket.side_effect = handshake
        with pytest.raises(TimeoutError, match="connection deadline exceeded"):
            conn.connect()
        conn._create_connection.assert_called_once()
        raw.close.assert_called_once()
        wrapped.close.assert_called_once()
        wrapped.settimeout.assert_not_called()
        assert conn.sock is None

    def test_reconnect_cannot_discard_an_existing_socket(self):
        validated = ValidatedOutboundURL(
            url="https://example.test/did.json", scheme="https", host="example.test", port=443,
            ip_addresses=(ipaddress.ip_address("8.8.8.8"),),
        )
        conn = _PinnedHTTPSConnection(validated.host, validated=validated)
        original = MagicMock(spec=socket.socket)
        conn.sock = original
        conn._create_connection = MagicMock(side_effect=AssertionError("reconnect guard allowed a new dial"))
        with pytest.raises(RuntimeError, match="connected"):
            conn.connect()
        conn._create_connection.assert_not_called()
        original.close.assert_not_called()
        assert conn.sock is original

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

    def test_all_tcp_failures_report_the_last_failure_without_tls(self):
        validated = ValidatedOutboundURL(
            url="https://example.test/did.json", scheme="https", host="example.test", port=443,
            ip_addresses=(ipaddress.ip_address("8.8.8.8"), ipaddress.ip_address("1.1.1.1")),
        )
        conn = _PinnedHTTPSConnection(validated.host, validated=validated)
        conn._create_connection = MagicMock(side_effect=[OSError("first unavailable"), OSError("second unavailable")])
        conn._context = MagicMock()
        with pytest.raises(OSError, match="second unavailable"):
            conn.connect()
        assert conn._create_connection.call_count == 2
        conn._context.wrap_socket.assert_not_called()
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

        class FakeContext:
            def wrap_socket(self, sock, server_hostname=None):
                wrapped.append(server_hostname)
                return sock

        def fake_create_connection(address, timeout, source_address):
            dialed.append(address)
            return MagicMock(spec=socket.socket)

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

        class FakeContext:
            def wrap_socket(self, sock, server_hostname=None):
                wrapped.append(server_hostname)
                return sock

        def fake_create_connection(address, timeout, source_address):
            dialed.append(address)
            return MagicMock(spec=socket.socket)

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

        class FakeContext:
            def wrap_socket(self, sock, server_hostname=None):
                wrapped.append(server_hostname)
                return sock

        def fake_create_connection(address, timeout, source_address):
            dialed.append(address)
            if address[0] == "2001:4860:4860::8888":
                raise OSError("IPv6 route unavailable")
            return MagicMock(spec=socket.socket)

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
        """Drive urllib's real redirect processing with a controlled HTTPS response."""
        from kestrel_sovereign.identity import did_web
        from email.message import Message
        from urllib.response import addinfourl
        import io

        requests = []

        def https_response(self, request):
            requests.append(request.full_url)
            headers = Message()
            if len(requests) == 1:
                headers["Location"] = "https://169.254.169.254/x"
                response = addinfourl(io.BytesIO(b""), headers, request.full_url, 302)
                response.msg = "Found"
            else:
                response = addinfourl(io.BytesIO(b"private metadata"), headers, request.full_url, 200)
                response.msg = "OK"
            return response

        def unexpected_socket(*args, **kwargs):
            pytest.fail("the controlled HTTPS response must not open a socket")

        monkeypatch.setattr("kestrel_sovereign.security.pinned_urllib._PinnedHTTPSHandler.https_open", https_response)
        monkeypatch.setattr("socket.create_connection", unexpected_socket)
        with pytest.raises(did_web.DidWebError, match="redirect"):
            did_web._default_fetcher("https://93.184.216.34/.well-known/did.json")
        assert requests == ["https://93.184.216.34/.well-known/did.json"]


@pytest.mark.parametrize("failure", [
    SSRFError("origin changed"),
    URLError("connection refused"),
    RemoteDisconnected("peer disconnected"),
    TimeoutError("body stalled"),
    ConnectionResetError("body disconnected"),
])
def test_did_fetch_network_failures_have_the_resolver_error_type(monkeypatch, failure):
    from kestrel_sovereign.identity import did_web
    opener = MagicMock()
    opener.open.side_effect = failure
    monkeypatch.setattr("kestrel_sovereign.security.pinned_urllib.pinned_urllib_https_opener", lambda *args: opener)
    with pytest.raises(did_web.DidWebError, match="failed"):
        did_web._default_fetcher("https://8.8.8.8/did.json")


def test_did_fetch_does_not_hide_programming_errors(monkeypatch):
    from kestrel_sovereign.identity import did_web
    opener = MagicMock()
    opener.open.side_effect = TypeError("invalid TLS configuration")
    monkeypatch.setattr("kestrel_sovereign.security.pinned_urllib.pinned_urllib_https_opener", lambda *args: opener)
    with pytest.raises(TypeError, match="configuration"):
        did_web._default_fetcher("https://8.8.8.8/did.json")


def test_pinned_connection_refuses_proxy_tunnels_before_dialing():
    validated = ValidatedOutboundURL(url="https://8.8.8.8/did.json", scheme="https", host="8.8.8.8", port=443,
                                     ip_addresses=(ipaddress.ip_address("8.8.8.8"),))
    conn = _PinnedHTTPSConnection(validated.host, validated=validated, timeout=1)
    conn.set_tunnel(validated.host)
    conn._create_connection = MagicMock(side_effect=AssertionError("proxy tunnel guard allowed a dial"))
    with pytest.raises(SSRFError, match="tunnel"):
        conn.connect()
    conn._create_connection.assert_not_called()


@pytest.mark.parametrize("status", [302, 404])
def test_did_http_errors_close_the_response_before_becoming_resolver_errors(monkeypatch, status):
    import io
    from urllib.error import HTTPError
    from kestrel_sovereign.identity import did_web

    body = io.BytesIO(b"remote error body")
    opener = MagicMock()
    opener.open.side_effect = HTTPError("https://8.8.8.8/did.json", status, "refused", {}, body)
    monkeypatch.setattr("kestrel_sovereign.security.pinned_urllib.pinned_urllib_https_opener", lambda *args: opener)
    with pytest.raises(did_web.DidWebError):
        did_web._default_fetcher("https://8.8.8.8/did.json")
    assert body.closed
