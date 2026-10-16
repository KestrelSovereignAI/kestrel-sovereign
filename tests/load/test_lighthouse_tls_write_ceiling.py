"""The ~1.9 GB single-TLS-write ceiling behind #3662, on a real TLS stack.

Skipped by default - run with: pytest --run-load tests/load/test_lighthouse_tls_write_ceiling.py

``ssl.SSLObject.write`` (what anyio's TLS stream calls for each write httpcore
makes) encrypts a whole write into OpenSSL's memory BIO before anything is
sent, and that BIO stops growing near 1.9 GB. Emma's last snapshot CAR to
upload was 1,902,750,356 bytes; the next was ~1,906,760,000, and every
upload from then on raised ``OSError: [Errno 0] Error``. These tests use
those two sizes, need ~6 GB of memory, and take a few seconds each.
``tests/unit/test_lighthouse_upload_streaming.py`` covers the same failure
at a scaled-down ceiling on every run.
"""

from __future__ import annotations

import asyncio
import datetime
import ipaddress
import ssl
from pathlib import Path

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from kestrel_sovereign.storage.providers.lighthouse_rest import LighthouseRestClient

pytestmark = [pytest.mark.load, pytest.mark.slow]

#: The last CAR Emma uploaded (2026-10-07 20:00Z) and about the size of the
#: first that failed (2026-10-08 00:00Z).
LAST_GOOD_CAR_BYTES = 1_902_750_356
FIRST_FAILING_CAR_BYTES = 1_906_760_340


@pytest.fixture
def skip_unless_load(request):
    """Skip unless --run-load flag provided."""
    if not request.config.getoption("--run-load", default=False):
        pytest.skip("Load tests skipped. Use --run-load to enable.")


@pytest.fixture
def tls_contexts(tmp_path: Path) -> tuple[ssl.SSLContext, ssl.SSLContext]:
    """A TLS 1.3 server context for localhost and a client context trusting it."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=True,
                crl_sign=False, encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    cert_file = tmp_path / "localhost.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    secret_file = tmp_path / "localhost-secret.pem"
    secret_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.minimum_version = ssl.TLSVersion.TLSv1_3
    server.load_cert_chain(cert_file, secret_file)
    client = ssl.create_default_context(cafile=str(cert_file))
    client.minimum_version = ssl.TLSVersion.TLSv1_3
    return server, client


def _memory_bio_pair(server_ctx: ssl.SSLContext, client_ctx: ssl.SSLContext):
    """A handshaken client SSLObject over memory BIOs, as anyio uses one."""
    c_in, c_out, s_in, s_out = (ssl.MemoryBIO() for _ in range(4))
    client = client_ctx.wrap_bio(c_in, c_out, server_hostname="localhost")
    server = server_ctx.wrap_bio(s_in, s_out, server_side=True)
    done = {client: False, server: False}
    for _ in range(10):
        for side in (client, server):
            if not done[side]:
                try:
                    side.do_handshake()
                    done[side] = True
                except ssl.SSLWantReadError:
                    pass
        if c_out.pending:
            s_in.write(c_out.read())
        if s_out.pending:
            c_in.write(s_out.read())
        if all(done.values()):
            return client
    raise AssertionError("handshake did not complete")


def test_one_tls_write_past_the_memory_bio_fails_with_emmas_error(skip_unless_load, tls_contexts):
    payload = memoryview(bytearray(FIRST_FAILING_CAR_BYTES))

    assert _memory_bio_pair(*tls_contexts).write(payload[:LAST_GOOD_CAR_BYTES]) == LAST_GOOD_CAR_BYTES
    with pytest.raises(OSError) as raised:
        _memory_bio_pair(*tls_contexts).write(payload)
    if ssl.OPENSSL_VERSION_INFO >= (3,):
        assert raised.type is OSError and raised.value.errno == 0


@pytest.mark.asyncio
async def test_a_car_past_the_ceiling_uploads_over_real_tls(skip_unless_load, tls_contexts):
    """Before #3662 this raised ``OSError: [Errno 0] Error`` from
    ``ssl.SSLObject.write`` with 36 body bytes delivered."""
    server_ctx, client_ctx = tls_contexts
    received = {"length": None, "body": 0}

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                received["length"] = int(line.split(b":", 1)[1])
        while received["body"] < received["length"]:
            chunk = await reader.read(4 << 20)
            if not chunk:
                break
            received["body"] += len(chunk)
        payload = b'{"Hash": "QmLocal", "Size": "%d"}' % received["body"]
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\nConnection: close\r\n\r\n" % len(payload) + payload
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0, ssl=server_ctx)
    port = server.sockets[0].getsockname()[1]

    class _LocalClient(LighthouseRestClient):
        UPLOAD_URL = f"https://localhost:{port}"

        async def _get_client(self) -> httpx.AsyncClient:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=self.timeout, verify=client_ctx)
            return self._client

    client = _LocalClient(api_key="local")
    try:
        result = await client.upload_car(
            bytes(FIRST_FAILING_CAR_BYTES), tag="t", filename="kestrel_state.car"
        )
    finally:
        await client.close()
        server.close()
        await server.wait_closed()

    assert result["Hash"] == "QmLocal"
    assert received["body"] == received["length"] > FIRST_FAILING_CAR_BYTES
