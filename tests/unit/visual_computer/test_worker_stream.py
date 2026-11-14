"""Broker peer loss with original Tasks/Futures; TLS cases use owned loopback IO."""
import asyncio
from asyncio import sslproto
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import os
import socket
import ssl
from types import SimpleNamespace

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
import pytest

from kestrel_sovereign._async_ownership import (
    await_owned_operation, await_owned_task, create_owned_task,
)
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.visual_computer import worker_broker as broker
from kestrel_sovereign.visual_computer.worker_stream import join_original_writer_close


class PeerTransport(asyncio.Transport):
    """Explicitly scripted IO, with a genuine stdlib close Future."""
    def __init__(self, protocol, error):
        self.protocol, self.error, self.closing = protocol, error, False
        self.close_calls = 0
    def is_closing(self):
        return self.closing
    def get_extra_info(self, name, default=None):
        return default
    def close(self):
        self.close_calls += 1
        if not self.closing:
            self.closing = True
            self.protocol.connection_lost(self.error)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["clean", "reset", "pipe", "timeout", "os-error", "local-error"])
async def test_broker_joins_original_peer_close_without_poisoning_lane(kind):
    loop = asyncio.get_running_loop()
    error = {"clean": None, "reset": ConnectionResetError("original peer reset"),
             "pipe": BrokenPipeError("original peer pipe"),
             "timeout": TimeoutError("unrecognized local timeout"),
             "os-error": OSError("unrecognized local close"),
             "local-error": RuntimeError("unrecognized local error")}[kind]
    lost_exchange = ValueError("scripted exchange ended before request")
    observed = {}
    class Loop:
        def __getattr__(self, name):
            return getattr(loop, name)
        async def connect_accepted_socket(self, factory, raw, **kwargs):
            protocol = factory()
            transport = PeerTransport(protocol, error)
            protocol.connection_made(transport)
            observed.update(protocol=protocol, transport=transport)
            return transport, protocol
    def check():
        raise lost_exchange
    slot = broker._Slot(SimpleNamespace(loop=Loop(), context=object(), check=check))
    task = create_owned_task(slot.exchange(), admission=slot.admission, name="original-broker-peer-close")
    outcome = await await_owned_task(task)
    writer, protocol, transport = slot.writer, observed["protocol"], observed["transport"]
    waiter = protocol._get_close_waiter(writer)
    assert slot.admission.task is task and task.done()
    assert slot.failure is lost_exchange
    assert waiter.done() and not waiter.cancelled() and waiter.exception() is error
    assert writer.is_closing() and transport.close_calls == 1
    if kind in {"clean", "reset", "pipe"}:
        assert outcome.error is None, "terminal peer close must not poison the broker lane"
        if error is not None:
            assert error in tuple(_explicit_control_chain(slot.failure))
    else:
        assert outcome.error is not None
        assert error in tuple(_explicit_control_chain(outcome.error))


@asynccontextmanager
async def stalled_tls_peer(tmp_path):
    """Real TLS1.3 handshake; owned peer never consumes the TLS close_notify.

    This measures local stream retirement only. The generated certificate is
    unrelated to the worker profile's authenticated client/server identities.
    """
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "owned-loopback-retirement")])
    now = datetime.now(UTC)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now-timedelta(minutes=1))
            .not_valid_after(now+timedelta(hours=1)).sign(key, hashes.SHA256()))
    for filename, raw in (
        ("cert.pem", cert.public_bytes(serialization.Encoding.PEM)),
        ("key.pem", key.private_bytes(serialization.Encoding.PEM,
                                     serialization.PrivateFormat.PKCS8, serialization.NoEncryption())),
    ):
        fd = os.open(tmp_path/filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(tmp_path/"cert.pem", tmp_path/"key.pem")
    client = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client.check_hostname = False
    client.verify_mode = ssl.CERT_NONE
    client.minimum_version = client.maximum_version = ssl.TLSVersion.TLSv1_3
    loop, accepted = asyncio.get_running_loop(), asyncio.get_running_loop().create_future()
    def accept(reader, writer):
        assert not accepted.done()
        accepted.set_result(writer)
    server = await asyncio.start_server(accept, "127.0.0.1", 0, ssl=context,
        ssl_handshake_timeout=1, ssl_shutdown_timeout=.05)
    listeners = tuple(server.sockets)
    raw = socket.socket()
    raw.setblocking(False)
    writer = None
    close_error = None
    try:
        async with asyncio.timeout(5):
            await loop.sock_connect(raw, server.sockets[0].getsockname())
            incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
            peer = client.wrap_bio(incoming, outgoing, server_side=False,
                                  server_hostname="owned-loopback-retirement")
            while True:
                try:
                    peer.do_handshake()
                    done = True
                except ssl.SSLWantReadError:
                    done = False
                if outgoing.pending:
                    await loop.sock_sendall(raw, outgoing.read())
                if done:
                    break
                data = await loop.sock_recv(raw, 65536)
                assert data
                incoming.write(data)
            writer = await accepted
        original_socket = writer.get_extra_info("socket")
        assert peer.version() == "TLSv1.3"
        yield writer
    finally:
        server.close()
        await server.wait_closed()
        try:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except TimeoutError as error:
                    close_error = error
                assert original_socket.fileno() == -1
                assert writer._protocol._get_close_waiter(writer).done()
                if close_error is not None:
                    assert writer._protocol._get_close_waiter(writer).exception() is close_error
        finally:
            # Keep the nonresponsive peer alive until the original native
            # server join, even when a counterfactual skips the product join.
            raw.close()
        assert raw.fileno() == -1 and all(s.fileno() == -1 for s in listeners)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["shared", "broker"])
async def test_actual_TLS_shutdown_timeout_retires_original_stream(tmp_path, boundary):
    async with stalled_tls_peer(tmp_path) as writer:
        protocol, transport = writer._protocol, writer.transport
        tls = transport._ssl_protocol
        waiter = protocol._get_close_waiter(writer)
        if boundary == "shared":
            writer.close()
            outcome = await await_owned_operation(join_original_writer_close(writer),
                                                  retirement_resource=writer, retain_originals=True)
        else:
            class RefusedUpgrade:
                async def connect_accepted_socket(self, *args, **kwargs):
                    raise ValueError("scripted lost exchange with original native TLS stream")
            slot = broker._Slot(SimpleNamespace(loop=RefusedUpgrade(), context=object()))
            slot.writer = writer
            task = create_owned_task(slot.exchange(), admission=slot.admission,
                                     name="original-broker-native-TLS-close")
            outcome = await await_owned_task(task)
            assert task.done() and slot.admission.task is task
        assert waiter.done() and not waiter.cancelled(), "original native close Future must finish before publication"
        error = waiter.exception()
        assert type(error) is TimeoutError and error.args == ("SSL shutdown timed out",)
        assert protocol._get_close_waiter(writer) is waiter and writer.is_closing()
        assert tls._transport is None and transport._closed
        assert outcome.error is None, "completed native TLS close must not poison owned stream retirement"
        if boundary == "shared":
            assert outcome.result is error
        else:
            assert error in tuple(_explicit_control_chain(slot.failure))


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", [
    "transport-type", "transport-identity", "ssl-identity", "ssl-type", "state", "app-state",
    "raw-transport", "app-transport", "app-protocol", "lost-count", "diagnostic",
])
async def test_TLS_timeout_recognition_requires_original_retired_protocol(tmp_path, monkeypatch, guard):
    """Change one observation after a native close; restore it before cleanup."""
    async with stalled_tls_peer(tmp_path) as writer:
        transport = writer.transport
        tls = transport._ssl_protocol
        waiter = writer._protocol._get_close_waiter(writer)
        original_wait = writer.wait_closed
        saved = []
        foreign_tls = None
        def replace(obj, name, value):
            saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)
        async def observed_wait():
            try:
                await original_wait()
            finally:
                if foreign_tls is not None:
                    foreign_tls.__dict__.update(tls.__dict__)
                if guard == "transport-identity":
                    foreign_transport = object.__new__(sslproto._SSLProtocolTransport)
                    foreign_transport.__dict__.update(transport.__dict__)
                    replace(writer, "_transport", foreign_transport)
                elif guard == "ssl-identity":
                    replace(transport, "_ssl_protocol", object())
                elif guard == "state":
                    replace(tls, "_state", sslproto.SSLProtocolState.SHUTDOWN)
                elif guard == "app-state":
                    replace(tls, "_app_state", sslproto.AppProtocolState.STATE_CON_MADE)
                elif guard in {"raw-transport", "app-transport", "app-protocol"}:
                    replace(tls, {"raw-transport": "_transport", "app-transport": "_app_transport",
                                  "app-protocol": "_app_protocol"}[guard], object())
                elif guard == "lost-count":
                    replace(tls, "_conn_lost", 0)
                elif guard == "diagnostic":
                    replace(waiter.exception(), "args", ("unrecognized local TLS timeout",))
        writer.close()
        try:
            if guard == "transport-type":
                class ForeignTLSTransport(sslproto._SSLProtocolTransport):
                    pass
                foreign_transport = object.__new__(ForeignTLSTransport)
                foreign_transport.__dict__.update(transport.__dict__)
                replace(writer, "_transport", foreign_transport)
            elif guard == "ssl-type":
                # The original stream has already begun shutdown. A foreign
                # SSLProtocol observation cannot prove its physical retirement.
                class ForeignSSLProtocol(sslproto.SSLProtocol):
                    pass
                foreign_tls = object.__new__(ForeignSSLProtocol)
                foreign_tls.__dict__.update(tls.__dict__)
                replace(transport, "_ssl_protocol", foreign_tls)
            monkeypatch.setattr(writer, "wait_closed", observed_wait)
            outcome = await await_owned_operation(join_original_writer_close(writer),
                                                  retirement_resource=writer, retain_originals=True)
        finally:
            for obj, name, value in reversed(saved):
                setattr(obj, name, value)
            monkeypatch.setattr(writer, "wait_closed", original_wait)
        assert outcome.error is not None
        assert waiter.exception() in tuple(_explicit_control_chain(outcome.error))



@pytest.mark.asyncio
async def test_TLS_close_refuses_foreign_timeout_subclass(tmp_path):
    async with stalled_tls_peer(tmp_path) as writer:
        protocol = writer._protocol
        original_waiter = protocol._get_close_waiter(writer)
        writer.close()
        try:
            await writer.wait_closed()
        except TimeoutError as native_error:
            assert type(native_error) is TimeoutError
            assert native_error.args == ("SSL shutdown timed out",)
        assert original_waiter.done()
        # Deliberately foreign Future/error observation, after the real native
        # close. This is a refusal test, never a physical close receipt.
        class ForeignTimeout(TimeoutError):
            pass
        foreign = ForeignTimeout("SSL shutdown timed out")
        replacement = asyncio.get_running_loop().create_future()
        replacement.set_exception(foreign)
        protocol._closed = replacement
        try:
            outcome = await await_owned_operation(join_original_writer_close(writer),
                                                  retirement_resource=writer, retain_originals=True)
        finally:
            protocol._closed = original_waiter
            assert replacement.exception() is foreign
        assert outcome.error is not None
        assert foreign in tuple(_explicit_control_chain(outcome.error))
