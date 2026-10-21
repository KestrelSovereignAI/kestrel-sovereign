"""One verified public-origin TLS fixture for production GET and POST callers."""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from tests.utils.tls import make_tls_contexts, tls_http_server


@pytest.fixture
def outbound_close_barrier(monkeypatch):
    """Retain the real response stream's close until its test owner releases it."""
    import asyncio

    import httpx

    entered, release = asyncio.Event(), asyncio.Event()
    original_init = httpx.AsyncClient.__init__

    async def response_owner(response):
        original_close = response.stream.aclose

        async def close():
            entered.set()
            await release.wait()
            await original_close()

        response.stream.aclose = close

    def client_owner(client, *args, **kwargs):
        original_init(client, *args, **kwargs)
        client.event_hooks["response"].append(response_owner)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", client_owner)
    try:
        yield SimpleNamespace(entered=entered, release=release)
    finally:
        release.set()


@pytest.fixture
def outbound_tls(tmp_path, monkeypatch, request):
    """Route only a validated public IP to a private real TLS peer."""
    hostname = "xn--strae-oqa.example"
    settings = getattr(request, "param", hostname)
    if isinstance(settings, dict):
        certificate_hostname = settings.get("certificate_hostname", hostname)
        dns_validations = settings["dns_validations"]
    else:
        certificate_hostname, dns_validations = settings, 1
    assert type(dns_validations) is int and dns_validations > 0
    server_context, client_context, certificate = make_tls_contexts(
        tmp_path, certificate_hostname, ip_addresses=()
    )
    monkeypatch.setenv("SSL_CERT_FILE", str(certificate))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    dns, dials, requests, posts, sni = [], [], [], [], []
    reply = threading.Event()
    reply.set()
    actual_dns = socket.getaddrinfo
    actual_dial = socket.create_connection
    server_context.set_servername_callback(lambda sock, name, context: sni.append(name))

    def resolve(host, port, *args, **kwargs):
        if host == "127.0.0.1":
            return actual_dns(host, port, *args, **kwargs)
        dns.append(host)
        assert host == hostname
        assert len(dns) <= dns_validations, (
            "hostname exceeded its independent validation budget"
        )
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("8.8.8.8", port),
            )
        ]

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

        def do_POST(self):
            route = urlsplit(self.path).path
            body = self.rfile.read(int(self.headers["Content-Length"]))
            posts.append((self.path, dict(self.headers), json.loads(body)))
            assert reply.wait(5), "test callback headers were not released"
            if route == "/drip-headers":
                # A real slow-loris peer: each read makes progress well within
                # the socket timeout, while the whole header never completes
                # within the sender's total exchange deadline.
                try:
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Probe: ")
                    for _ in range(40):
                        self.wfile.write(b"a")
                        self.wfile.flush()
                        time.sleep(0.04)
                    self.wfile.write(b"\r\nContent-Length: 0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # The sender's deadline physically closed its socket.
                return
            status = (
                302
                if route == "/redirect"
                else 503
                if route == "/failure"
                else 200
            )
            self.send_response(status)
            if status == 302:
                self.send_header("Location", "/followed")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    with tls_http_server(server_context, Handler) as server:
        port = server.server_port

        def dial(
            address,
            timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
            source_address=None,
            **kwargs,
        ):
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
        yield SimpleNamespace(
            url=f"https://straße.example:{port}",
            host=hostname,
            port=port,
            dns=dns,
            dials=dials,
            requests=requests,
            posts=posts,
            reply=reply,
            sni=sni,
            client_context=client_context,
        )
