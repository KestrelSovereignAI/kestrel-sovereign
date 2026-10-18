"""Exercise Core's production fetch callers over real verified TLS and HTTP."""
from http.server import BaseHTTPRequestHandler
import json
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import pytest

from tests.unit.test_identity_constitution_endpoint_contracts import _prepare_app, _restore_app
from tests.utils.tls import make_tls_contexts, tls_http_server


@pytest.fixture
def outbound_tls(tmp_path, monkeypatch, request):
    """Route only a validated public IP to a private real TLS peer."""
    hostname = "xn--strae-oqa.example"
    certificate_hostname = getattr(request, "param", hostname)
    server_context, client_context, certificate = make_tls_contexts(tmp_path, certificate_hostname, ip_addresses=())
    monkeypatch.setenv("SSL_CERT_FILE", str(certificate))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    dns, dials, requests, sni = [], [], [], []
    actual_dns = socket.getaddrinfo
    actual_dial = socket.create_connection
    server_context.set_servername_callback(lambda sock, name, context: sni.append(name))

    def resolve(host, port, *args, **kwargs):
        if host == "127.0.0.1":
            return actual_dns(host, port, *args, **kwargs)
        dns.append(host)
        assert host == hostname
        assert dns == [hostname], "hostname was resolved again after validation"
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", port))]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers["Host"]))
            body = json.dumps({"id": "did:web:straße.example"}).encode()
            status = 302 if self.path == "/redirect" else 200
            self.send_response(status)
            if status == 302:
                self.send_header("Location", "/followed")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with tls_http_server(server_context, Handler) as server:
        port = server.server_port

        def dial(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, **kwargs):
            dials.append(address)
            assert address == ("8.8.8.8", port)
            return actual_dial(("127.0.0.1", port), timeout, source_address, **kwargs)

        monkeypatch.setattr(socket, "getaddrinfo", resolve)
        monkeypatch.setattr(socket, "create_connection", dial)
        # The native async backend connects directly, without socket.create_connection.
        from httpcore._backends.anyio import AnyIOBackend
        actual_async_dial = AnyIOBackend.connect_tcp

        async def async_dial(self, host, port, **kwargs):
            dials.append((host, port))
            assert (host, port) == ("8.8.8.8", server.server_port)
            return await actual_async_dial(self, host="127.0.0.1", port=port, **kwargs)

        monkeypatch.setattr(AnyIOBackend, "connect_tcp", async_dial)
        yield SimpleNamespace(url=f"https://straße.example:{port}", host=hostname, port=port,
                              dns=dns, dials=dials, requests=requests, sni=sni, client_context=client_context)


def test_did_fetch_preserves_idna2008_host_sni_and_proxy_isolation(outbound_tls):
    from kestrel_sovereign.identity.did_web import _default_fetcher
    peer = outbound_tls
    assert json.loads(_default_fetcher(peer.url + "/did.json"))["id"] == "did:web:straße.example"
    assert peer.dns == [peer.host]
    assert peer.dials == [("8.8.8.8", peer.port)]
    assert peer.sni == [peer.host]
    assert peer.requests == [("/did.json", f"{peer.host}:{peer.port}")]


@pytest.mark.parametrize("outbound_tls", ["wrong.example"], indirect=True)
def test_did_fetch_enforces_the_real_peer_certificate(outbound_tls):
    from kestrel_sovereign.identity.did_web import _default_fetcher, DidWebError
    peer = outbound_tls
    with pytest.raises(DidWebError, match="failed"):
        _default_fetcher(peer.url + "/did.json")
    assert peer.dials == [("8.8.8.8", peer.port)]
    assert peer.sni == [peer.host]
    assert peer.requests == []


@pytest.mark.usefixtures("isolated_process_rate_limiter")
@pytest.mark.parametrize("outbound_tls,path,status,served", [
    pytest.param("xn--strae-oqa.example", "/avatar.png", 200, True, id="verified-peer"),
    pytest.param("xn--strae-oqa.example", "/redirect", 400, True, id="redirect-refused"),
    pytest.param("wrong.example", "/avatar.png", 400, False, id="wrong-certificate"),
], indirect=["outbound_tls"])
def test_avatar_uses_real_tls_and_refuses_http_redirects(outbound_tls, path, status, served):
    peer = outbound_tls
    store = AsyncMock(return_value="avatar-hash")
    agent = SimpleNamespace(agent_id="did:test:avatar", storage=SimpleNamespace(files=SimpleNamespace(store_avatar=store)))
    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post("/api/identity/avatar", json={"url": peer.url + path}, headers={"X-API-Key": "test-key"})
        assert response.status_code == status
        assert peer.dns == [peer.host]
        assert peer.dials == [("8.8.8.8", peer.port)]
        assert peer.sni == [peer.host]
        assert peer.requests == ([(path, f"{peer.host}:{peer.port}")] if served else [])
        if status == 400:
            assert ("302" if served else "CERTIFICATE_VERIFY_FAILED") in response.json()["detail"]
            store.assert_not_awaited()
        else:
            store.assert_awaited_once()
            assert store.await_args.kwargs["source_url"] == peer.url + path
    finally:
        _restore_app(app, original)


def test_extracted_tls_fixture_still_supports_the_load_test_memory_bio(tmp_path):
    from tests.load.test_lighthouse_tls_write_ceiling import _memory_bio_pair
    server, client, _ = make_tls_contexts(tmp_path)
    assert _memory_bio_pair(server, client).write(b"small payload") == len(b"small payload")
