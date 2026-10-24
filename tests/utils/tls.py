"""Shared real TLS certificate and isolated HTTP server fixtures."""
import datetime
import ipaddress
import ssl
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


class _NumericLoopbackHTTPServer(ThreadingHTTPServer):
    """Actual local TCP/TLS fixture without HTTPServer's reverse DNS lookup."""

    daemon_threads = False

    def server_bind(self):
        if self.server_address[0] != "127.0.0.1":
            raise ValueError("TLS test server requires numeric IPv4 loopback")
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address


def make_tls_contexts(directory: Path, hostname: str = "localhost", *, ip_addresses: tuple[str, ...] = ("127.0.0.1",)) -> tuple[ssl.SSLContext, ssl.SSLContext, Path]:
    """Make a short-lived TLS 1.3 certificate and a client that trusts it."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    now = datetime.datetime.now(datetime.UTC)
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
                [x509.DNSName(hostname), *(x509.IPAddress(ipaddress.ip_address(address)) for address in ip_addresses)]
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
    cert_file = directory / "certificate.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    secret_file = directory / "private-key.pem"
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
    return server, client, cert_file

@contextmanager
def tls_http_server(context: ssl.SSLContext, handler: type[BaseHTTPRequestHandler]):
    """Own a loopback HTTPS server, bound to an OS-assigned private port."""
    with _NumericLoopbackHTTPServer(("127.0.0.1", 0), handler) as server:
        server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive(), "test HTTPS server did not terminate"
